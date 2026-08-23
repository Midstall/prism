//! Linux virgl transport over the kernel virtio-gpu DRM uAPI (/dev/dri/renderD*).
//! Uses DRM_IOCTL_VIRTGPU_EXECBUFFER + RESOURCE_CREATE + TRANSFER_TO/FROM_HOST. No root.
//! Guest backing is the mmap of each resource's GEM BO. No conduit dependency.

const std = @import("std");
const types = @import("types.zig");
const prism_log = @import("../../../log.zig");

const Error = types.Error;
const Box = types.Box;
const Resource = types.Resource;
const enc = types.enc;

// DRM ioctl encoding (drm.h + virtgpu_drm.h): DRM_IOCTL_BASE='d' (0x64), DRM_COMMAND_BASE=0x40.
// _IOWR(nr,type) builds a 32-bit request: dir(2)|size(14)|type(8)|nr(8).

const DRM_IOCTL_BASE: u32 = 'd';
const DRM_COMMAND_BASE: u32 = 0x40;

const IOC_NONE: u32 = 0;
const IOC_WRITE: u32 = 1;
const IOC_READ: u32 = 2;
const IOC_NRSHIFT: u5 = 0;
const IOC_TYPESHIFT: u5 = 8;
const IOC_SIZESHIFT: u5 = 16;
const IOC_DIRSHIFT: u5 = 30;

fn ioc(dir: u32, typ: u32, nr: u32, size: u32) u32 {
    return (dir << IOC_DIRSHIFT) | (typ << IOC_TYPESHIFT) | (nr << IOC_NRSHIFT) | (size << IOC_SIZESHIFT);
}
fn drmIowr(comptime nr: u32, comptime T: type) u32 {
    return ioc(IOC_READ | IOC_WRITE, DRM_IOCTL_BASE, DRM_COMMAND_BASE + nr, @sizeOf(T));
}

// virtgpu command numbers (virtgpu_drm.h).
const DRM_VIRTGPU_MAP: u32 = 0x01;
const DRM_VIRTGPU_EXECBUFFER: u32 = 0x02;
const DRM_VIRTGPU_GETPARAM: u32 = 0x03;
const DRM_VIRTGPU_RESOURCE_CREATE: u32 = 0x04;
const DRM_VIRTGPU_RESOURCE_INFO: u32 = 0x05;
const DRM_VIRTGPU_TRANSFER_FROM_HOST: u32 = 0x06;
const DRM_VIRTGPU_TRANSFER_TO_HOST: u32 = 0x07;
const DRM_VIRTGPU_GET_CAPS: u32 = 0x09;
const DRM_VIRTGPU_CONTEXT_INIT: u32 = 0x0b;

const VIRTGPU_DRM_CAPSET_VIRGL2: u32 = 2;
const VIRTGPU_CONTEXT_PARAM_CAPSET_ID: u64 = 0x0001;

// uAPI structs (extern, matching virtgpu_drm.h layouts).

const drm_virtgpu_map = extern struct {
    offset: u64,
    handle: u32,
    pad: u32,
};

const drm_virtgpu_execbuffer = extern struct {
    flags: u32,
    size: u32,
    command: u64, // pointer to the virgl command-stream bytes
    bo_handles: u64, // pointer to a u32[] of bo handles referenced by the stream
    num_bo_handles: u32,
    fence_fd: i32,
    ring_idx: u32,
    syncobj_stride: u32,
    num_in_syncobjs: u32,
    num_out_syncobjs: u32,
    in_syncobjs: u64,
    out_syncobjs: u64,
};

const drm_virtgpu_getparam = extern struct {
    param: u64,
    value: u64,
};

const drm_virtgpu_resource_create = extern struct {
    target: u32,
    format: u32,
    bind: u32,
    width: u32,
    height: u32,
    depth: u32,
    array_size: u32,
    last_level: u32,
    nr_samples: u32,
    flags: u32,
    bo_handle: u32, // in: 0; out: kernel-assigned GEM handle
    res_handle: u32, // out: virgl resource id
    size: u32,
    stride: u32,
};

const drm_virtgpu_3d_box = extern struct { x: u32, y: u32, z: u32, w: u32, h: u32, d: u32 };

const drm_virtgpu_3d_transfer_to_host = extern struct {
    bo_handle: u32,
    box: drm_virtgpu_3d_box,
    level: u32,
    offset: u32,
    stride: u32,
    layer_stride: u32,
};

const drm_virtgpu_3d_transfer_from_host = extern struct {
    bo_handle: u32,
    box: drm_virtgpu_3d_box,
    level: u32,
    offset: u32,
    stride: u32,
    layer_stride: u32,
};

const drm_virtgpu_context_set_param = extern struct {
    param: u64,
    value: u64,
};

const drm_virtgpu_context_init = extern struct {
    num_params: u32,
    pad: u32,
    ctx_set_params: u64,
};

const DRM_IOCTL_VIRTGPU_MAP = drmIowr(DRM_VIRTGPU_MAP, drm_virtgpu_map);
const DRM_IOCTL_VIRTGPU_EXECBUFFER = drmIowr(DRM_VIRTGPU_EXECBUFFER, drm_virtgpu_execbuffer);
const DRM_IOCTL_VIRTGPU_GETPARAM = drmIowr(DRM_VIRTGPU_GETPARAM, drm_virtgpu_getparam);
const DRM_IOCTL_VIRTGPU_RESOURCE_CREATE = drmIowr(DRM_VIRTGPU_RESOURCE_CREATE, drm_virtgpu_resource_create);
const DRM_IOCTL_VIRTGPU_TRANSFER_FROM_HOST = drmIowr(DRM_VIRTGPU_TRANSFER_FROM_HOST, drm_virtgpu_3d_transfer_from_host);
const DRM_IOCTL_VIRTGPU_TRANSFER_TO_HOST = drmIowr(DRM_VIRTGPU_TRANSFER_TO_HOST, drm_virtgpu_3d_transfer_to_host);
const DRM_IOCTL_VIRTGPU_CONTEXT_INIT = drmIowr(DRM_VIRTGPU_CONTEXT_INIT, drm_virtgpu_context_init);

/// Linux transport construction args. `node` forces a specific /dev/dri/renderD*.
/// Null auto-scans for the virtio_gpu render node.
///
/// Zero-copy scanout: `external_fd` lets a display owner pass an already-open
/// /dev/dri/card0 fd (DRM master + KMS + virtgpu) so the virgl context and the
/// render target share the same fd the KMS modeset/page-flip uses. The transport
/// does not own/close that fd. `external_resource`, if set, is a virtgpu resource
/// the display owner already created (BIND_RENDER_TARGET|BIND_SCANOUT) so the same
/// resource virgl renders into is the KMS scanout framebuffer.
pub const InitArgs = struct {
    node: ?[]const u8 = null,
    external_fd: ?std.posix.fd_t = null,
};

/// A virtgpu resource created directly on a transport fd (for the zero-copy
/// scanout path): the kernel res_handle + GEM bo_handle + the mapped CPU view.
/// The display owner creates it (createScanoutResource) and both binds it as a
/// KMS framebuffer (ADDFB2 over bo_handle) AND feeds it to the HAL as the virgl
/// render target.
pub const ScanoutResource = struct {
    res_handle: u32,
    bo_handle: u32,
    bytes: []u8,
    width: u32,
    height: u32,
    stride: u32,
};

/// Return the bytes-per-pixel for a virgl format ordinal on the scanout path.
/// Virtio-gpu scanout is 4-byte-only; fp16 (R16G16B16A16_FLOAT = 94) is 8 bpp
/// and cannot scan out. All recognised 4-byte scanout formats return 4; fp16
/// returns 8 so the caller can gate on bpp != 4. Unknown formats default to 4.
fn scanoutBpp(virgl_format: u32) u32 {
    return switch (virgl_format) {
        enc.FORMAT_B8G8R8X8_UNORM => 4,
        enc.FORMAT_B8G8R8A8_UNORM => 4,
        enc.FORMAT_R10G10B10A2_UNORM => 4,
        enc.FORMAT_B10G10R10X2_UNORM => 4,
        enc.FORMAT_R16G16B16A16_FLOAT => 8,
        else => 4,
    };
}

pub const Transport = struct {
    gpa: std.mem.Allocator,
    fd: std.posix.fd_t,
    owns_fd: bool = true,
    ctx_ready: bool = false,
    log: prism_log.Logger = .{},
    // The bo handles created so far, referenced by EXECBUFFER's bo list. The
    // virgl stream references resources by res_handle, but the kernel needs the
    // BO handles of every resource the stream touches so it can pin them.
    bo_handles: [16]u32 = undefined,
    n_bo: usize = 0,

    pub fn init(gpa: std.mem.Allocator, args: InitArgs) Error!Transport {
        if (args.external_fd) |xfd| {
            // The display owner already opened the node (card0: master + KMS +
            // virtgpu). Share it. Do not close it on deinit.
            return .{ .gpa = gpa, .fd = xfd, .owns_fd = false };
        }
        const fd = if (args.node) |p|
            openNode(p) catch return error.InitializationFailed
        else
            discover() catch return error.InitializationFailed;
        return .{ .gpa = gpa, .fd = fd };
    }

    pub fn deinit(self: *Transport) void {
        if (self.owns_fd) _ = std.os.linux.close(self.fd);
    }

    /// Register an externally-created scanout resource (its GEM bo) so EXECBUFFER
    /// pins it for the host context. Used by the zero-copy path: the display owner
    /// creates the RT via createScanoutResource and adopts it as the HAL render
    /// target. The transport must know its bo handle to reference it in submits.
    pub fn trackResource(self: *Transport, res: Resource) void {
        self.trackBo(res.bo_handle);
    }

    /// Create a scanout-able render target directly on this transport's fd:
    /// RESOURCE_CREATE {TEXTURE_2D, given format, BIND_RENDER_TARGET|BIND_SCANOUT,
    /// w*h} + MAP its BO. Returns the res/bo handles + the CPU mapping so the
    /// display owner can ADDFB2 the bo as a KMS framebuffer and hand the resource
    /// back as the virgl render target. The bo is tracked for EXECBUFFER pinning.
    pub fn createScanoutResource(self: *Transport, width: u32, height: u32, format: u32) Error!ScanoutResource {
        const bpp = scanoutBpp(format);
        if (bpp != 4) return error.Unsupported;
        try self.ensureContext();
        const size: usize = @as(usize, width) * @as(usize, height) * @as(usize, bpp);
        var rc = drm_virtgpu_resource_create{
            .target = enc.TEXTURE_2D,
            .format = format,
            .bind = enc.BIND_RENDER_TARGET | enc.BIND_SCANOUT,
            .width = width,
            .height = height,
            .depth = 1,
            .array_size = 1,
            .last_level = 0,
            .nr_samples = 0,
            .flags = 0,
            .bo_handle = 0,
            .res_handle = 0,
            .size = @intCast(size),
            .stride = width * bpp,
        };
        try self.ioctlInit(DRM_IOCTL_VIRTGPU_RESOURCE_CREATE, @intFromPtr(&rc));
        const bytes = try self.mapBo(rc.bo_handle, size);
        @memset(bytes, 0);
        self.trackBo(rc.bo_handle);
        return .{
            .res_handle = rc.res_handle,
            .bo_handle = rc.bo_handle,
            .bytes = bytes,
            .width = width,
            .height = height,
            .stride = width * bpp,
        };
    }

    /// Open a render node O_RDWR | O_CLOEXEC.
    fn openNode(path: []const u8) !std.posix.fd_t {
        var buf: [64]u8 = undefined;
        const z = std.fmt.bufPrintZ(&buf, "{s}", .{path}) catch return error.OpenFailed;
        const rc = std.os.linux.open(z.ptr, .{ .ACCMODE = .RDWR, .CLOEXEC = true }, 0);
        switch (std.posix.errno(rc)) {
            .SUCCESS => return @intCast(rc),
            else => return error.OpenFailed,
        }
    }

    /// Scan /dev/dri/renderD128.. for the node whose driver is "virtio_gpu" (per
    /// /sys/class/drm/renderD<N>/device/driver symlink target). Falls back to
    /// renderD128 if the sysfs probe is inconclusive.
    fn discover() !std.posix.fd_t {
        var n: u32 = 128;
        while (n < 192) : (n += 1) {
            var nbuf: [80]u8 = undefined;
            const sys = std.fmt.bufPrint(&nbuf, "/sys/class/drm/renderD{d}/device/driver", .{n}) catch continue;
            var lbuf: [256]u8 = undefined;
            const link = std.os.linux.readlink(@ptrCast(sys.ptr), &lbuf, lbuf.len);
            // readlink returns the byte count. Errno is checked via the raw return.
            const ll: isize = @bitCast(link);
            if (ll <= 0) continue;
            const target = lbuf[0..@intCast(ll)];
            if (std.mem.indexOf(u8, target, "virtio_gpu") != null or std.mem.indexOf(u8, target, "virtio-gpu") != null) {
                var pbuf: [64]u8 = undefined;
                const path = std.fmt.bufPrint(&pbuf, "/dev/dri/renderD{d}", .{n}) catch continue;
                return openNode(path) catch continue;
            }
        }
        // Fallback: try renderD128 directly (single-GPU virtio guest).
        return openNode("/dev/dri/renderD128");
    }

    fn ioctl(self: *Transport, request: u32, arg: usize) Error!void {
        const rc = std.os.linux.ioctl(self.fd, request, arg);
        switch (std.posix.errno(rc)) {
            .SUCCESS => {},
            else => |e| {
                self.log.log("ioctl req=0x{x} errno={d}", .{ request, @intFromEnum(e) });
                return error.DeviceLost;
            },
        }
    }

    /// CONTEXT_INIT with the virgl2 capset, creating the 3D context the stream
    /// runs in. Idempotent.
    pub fn ensureContext(self: *Transport) Error!void {
        if (self.ctx_ready) return;
        var params = [_]drm_virtgpu_context_set_param{
            .{ .param = VIRTGPU_CONTEXT_PARAM_CAPSET_ID, .value = VIRTGPU_DRM_CAPSET_VIRGL2 },
        };
        var ci = drm_virtgpu_context_init{
            .num_params = 1,
            .pad = 0,
            .ctx_set_params = @intFromPtr(&params),
        };
        const rc = std.os.linux.ioctl(self.fd, DRM_IOCTL_VIRTGPU_CONTEXT_INIT, @intFromPtr(&ci));
        switch (std.posix.errno(rc)) {
            .SUCCESS => {},
            else => return error.InitializationFailed,
        }
        self.ctx_ready = true;
    }

    fn trackBo(self: *Transport, h: u32) void {
        if (self.n_bo < self.bo_handles.len) {
            self.bo_handles[self.n_bo] = h;
            self.n_bo += 1;
        }
    }

    /// RESOURCE_CREATE for a host vertex buffer + map its BO so the CPU can write
    /// the vertex data.
    pub fn createBuffer(self: *Transport, size: usize) Error!Resource {
        try self.ensureContext();
        var rc = drm_virtgpu_resource_create{
            .target = enc.BUFFER,
            .format = 0,
            .bind = enc.BIND_VERTEX_BUFFER,
            .width = @intCast(size),
            .height = 1,
            .depth = 1,
            .array_size = 1,
            .last_level = 0,
            .nr_samples = 0,
            .flags = 0,
            .bo_handle = 0,
            .res_handle = 0,
            .size = @intCast(size),
            .stride = 0,
        };
        try self.ioctlInit(DRM_IOCTL_VIRTGPU_RESOURCE_CREATE, @intFromPtr(&rc));
        const bytes = try self.mapBo(rc.bo_handle, size);
        @memset(bytes, 0);
        self.trackBo(rc.bo_handle);
        return .{ .res_id = rc.res_handle, .bytes = bytes, .bo_handle = rc.bo_handle, .is_vertex = true };
    }

    /// RESOURCE_CREATE for a host render target texture + map its BO for readback.
    /// `bpp` is the bytes per pixel of the format (4 for B8G8R8X8 / 10-bit, 8 for
    /// the fp16 HDR RT), so the BO + transfer stride fit high-precision targets.
    pub fn createImage(self: *Transport, width: u32, height: u32, format: u32, bpp: u32, samples: u32) Error!Resource {
        try self.ensureContext();
        const size: usize = @as(usize, width) * @as(usize, height) * @as(usize, bpp);
        // A depth/stencil format binds DEPTH_STENCIL (a ZETA target, never scanned out).
        // A color format binds RENDER_TARGET plus SCANOUT for 4-byte scanout-capable ones.
        // fp16 HDR RTs are render-target only.
        const bind: u32 = if (enc.isDepthFormat(format))
            enc.BIND_DEPTH_STENCIL
        else
            enc.BIND_RENDER_TARGET | enc.BIND_SAMPLER_VIEW | (if (bpp == 4) enc.BIND_SCANOUT else 0);
        var rc = drm_virtgpu_resource_create{
            .target = enc.TEXTURE_2D,
            .format = format,
            .bind = bind,
            .width = width,
            .height = height,
            .depth = 1,
            .array_size = 1,
            .last_level = 0,
            .nr_samples = samples,
            .flags = 0,
            .bo_handle = 0,
            .res_handle = 0,
            .size = @intCast(size),
            .stride = width * bpp,
        };
        try self.ioctlInit(DRM_IOCTL_VIRTGPU_RESOURCE_CREATE, @intFromPtr(&rc));
        const bytes = try self.mapBo(rc.bo_handle, size);
        @memset(bytes, 0);
        self.trackBo(rc.bo_handle);
        return .{ .res_id = rc.res_handle, .bytes = bytes, .bo_handle = rc.bo_handle, .width = width, .height = height, .bpp = bpp };
    }

    /// An ioctl whose failure is an init failure (resource create / map).
    fn ioctlInit(self: *Transport, request: u32, arg: usize) Error!void {
        const rc = std.os.linux.ioctl(self.fd, request, arg);
        switch (std.posix.errno(rc)) {
            .SUCCESS => {},
            else => |e| {
                self.log.log("ioctlInit req=0x{x} errno={d}", .{ request, @intFromEnum(e) });
                return error.InitializationFailed;
            },
        }
    }

    /// VIRTGPU_MAP a BO to get its mmap offset, then mmap it into the process.
    fn mapBo(self: *Transport, bo_handle: u32, size: usize) Error![]u8 {
        var m = drm_virtgpu_map{ .offset = 0, .handle = bo_handle, .pad = 0 };
        try self.ioctlInit(DRM_IOCTL_VIRTGPU_MAP, @intFromPtr(&m));
        const r = std.os.linux.mmap(null, size, .{ .READ = true, .WRITE = true }, .{ .TYPE = .SHARED }, self.fd, @intCast(m.offset));
        switch (std.posix.errno(r)) {
            .SUCCESS => {},
            else => return error.InitializationFailed,
        }
        const ptr: [*]u8 = @ptrFromInt(r);
        return ptr[0..size];
    }

    pub fn destroyResource(self: *Transport, res: Resource) void {
        _ = self;
        _ = std.os.linux.munmap(@ptrCast(res.bytes.ptr), res.bytes.len);
        // The GEM BO is reaped when the fd closes (close-on-last-ref). No
        // per-resource GEM_CLOSE needed for this short-lived test path.
    }

    pub fn map(self: *Transport, res: Resource) Error![]u8 {
        _ = self;
        return res.bytes;
    }

    /// TRANSFER_TO_HOST: upload `len` bytes of the BO to the host resource.
    pub fn transferToHost(self: *Transport, res: Resource, len: u32) Error!void {
        var t = drm_virtgpu_3d_transfer_to_host{
            .bo_handle = res.bo_handle,
            .box = .{ .x = 0, .y = 0, .z = 0, .w = len, .h = 1, .d = 1 },
            .level = 0,
            .offset = 0,
            .stride = 0,
            .layer_stride = 0,
        };
        try self.ioctl(DRM_IOCTL_VIRTGPU_TRANSFER_TO_HOST, @intFromPtr(&t));
    }

    /// TRANSFER_FROM_HOST: read the rendered RT pixels back into the BO mapping.
    pub fn transferFromHost(self: *Transport, res: Resource) Error!void {
        var t = drm_virtgpu_3d_transfer_from_host{
            .bo_handle = res.bo_handle,
            .box = .{ .x = 0, .y = 0, .z = 0, .w = res.width, .h = res.height, .d = 1 },
            .level = 0,
            .offset = 0,
            .stride = 0, // 0 -> kernel/host uses the resource's natural stride
            .layer_stride = 0,
        };
        try self.ioctl(DRM_IOCTL_VIRTGPU_TRANSFER_FROM_HOST, @intFromPtr(&t));
    }

    /// EXECBUFFER: submit the virgl command stream, referencing every tracked BO
    /// so the kernel pins them for the host context.
    pub fn submit(self: *Transport, stream_bytes: []const u8) Error!void {
        var eb = drm_virtgpu_execbuffer{
            .flags = 0,
            .size = @intCast(stream_bytes.len),
            .command = @intFromPtr(stream_bytes.ptr),
            .bo_handles = if (self.n_bo > 0) @intFromPtr(&self.bo_handles) else 0,
            .num_bo_handles = @intCast(self.n_bo),
            .fence_fd = -1,
            .ring_idx = 0,
            .syncobj_stride = 0,
            .num_in_syncobjs = 0,
            .num_out_syncobjs = 0,
            .in_syncobjs = 0,
            .out_syncobjs = 0,
        };
        try self.ioctl(DRM_IOCTL_VIRTGPU_EXECBUFFER, @intFromPtr(&eb));
    }

    /// No-op on Linux: there is no guest scanout in a headless render-node flow.
    /// The readback (transferFromHost) is the result path.
    pub fn present(self: *Transport, res: Resource) void {
        _ = self;
        _ = res;
    }
};

test "scanoutBpp: 4-byte scanout formats -> 4, fp16 -> 8" {
    try std.testing.expectEqual(@as(u32, 4), scanoutBpp(enc.FORMAT_B8G8R8X8_UNORM));
    try std.testing.expectEqual(@as(u32, 4), scanoutBpp(enc.FORMAT_R10G10B10A2_UNORM));
    try std.testing.expectEqual(@as(u32, 8), scanoutBpp(enc.FORMAT_R16G16B16A16_FLOAT));
}
