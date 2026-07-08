//! Freestanding virgl transport over a Conduit *Virtio (virtio-mmio).
//! Wraps ctxCreate / resourceCreate3D + attachBacking / transferToHost3D / submit3d / transferFromHost3D.
//! Comptime-selected on freestanding (transport.zig). Imports conduit, so never analysed on Linux.

const std = @import("std");
const types = @import("types.zig");

const conduit = @import("conduit");
const Virtio = conduit.driver.virtio_gpu.Virtio;
const vg = conduit.driver.virtio_gpu.virgl;

const Error = types.Error;
const Box = types.Box;
const Resource = types.Resource;
const enc = types.enc;

const RESP_OK_NODATA: u32 = 0x1100;
const CTX_ID: u32 = 1;
// 3D resource ids start at 16 (id 1 is reserved by the 2D scanout path).
const FIRST_RES_ID: u32 = 16;

/// The OS-specific construction argument: the freestanding transport is built
/// over a live, started, virgl-negotiated Conduit `*Virtio`. The command-stream
/// scratch lives in the Device (DMA-coherent, since the device is allocated from
/// the kernel's Normal-NC arena), so it is not part of the init args.
pub const InitArgs = struct {
    gpu: *Virtio,
};

pub const Transport = struct {
    gpa: std.mem.Allocator,
    gpu: *Virtio,
    next_res: u32 = FIRST_RES_ID,
    ctx_ready: bool = false,

    pub fn init(gpa: std.mem.Allocator, args: InitArgs) Error!Transport {
        return .{ .gpa = gpa, .gpu = args.gpu };
    }

    pub fn deinit(self: *Transport) void {
        _ = self;
        // The Conduit driver is caller-owned (a kernel static).
    }

    /// Create the 3D (virgl) context lazily. Safe to call repeatedly.
    pub fn ensureContext(self: *Transport) Error!void {
        if (self.ctx_ready) return;
        if (self.gpu.ctxCreate(CTX_ID, "prism-virgl") != RESP_OK_NODATA) return error.InitializationFailed;
        self.ctx_ready = true;
    }

    fn allocResId(self: *Transport) u32 {
        const id = self.next_res;
        self.next_res += 1;
        return id;
    }

    /// Create a host vertex buffer resource and attach `size` bytes of guest
    /// backing the CPU writes the vertex data into.
    pub fn createBuffer(self: *Transport, size: usize) Error!Resource {
        try self.ensureContext();
        const bytes = self.gpa.alloc(u8, size) catch return error.OutOfMemory;
        errdefer self.gpa.free(bytes);
        @memset(bytes, 0);
        const res_id = self.allocResId();
        if (self.gpu.resourceCreate3D(res_id, vg.BUFFER, 0, vg.BIND_VERTEX_BUFFER, @intCast(size), 1) != RESP_OK_NODATA)
            return error.InitializationFailed;
        if (self.gpu.attachBacking(res_id, @intFromPtr(bytes.ptr), @intCast(size)) != RESP_OK_NODATA)
            return error.InitializationFailed;
        if (self.gpu.ctxAttachResource(CTX_ID, res_id) != RESP_OK_NODATA)
            return error.InitializationFailed;
        return .{ .res_id = res_id, .bytes = bytes, .is_vertex = true };
    }

    /// Create a host render-target texture (`width`x`height`, `format`) and attach
    /// guest backing the rendered pixels are read back into. `bpp` is the bytes per
    /// pixel of the chosen format (4 for B8G8R8X8 / 10-bit, 8 for the fp16 HDR RT),
    /// so the backing + readback are sized for high-precision targets too.
    pub fn createImage(self: *Transport, width: u32, height: u32, format: u32, bpp: u32, samples: u32) Error!Resource {
        // Conduit's resourceCreate3D exposes no nr_samples field, so the freestanding
        // path is single-sample. MSAA (samples > 1) is a Linux-transport feature.
        _ = samples;
        try self.ensureContext();
        const size: usize = @as(usize, width) * @as(usize, height) * @as(usize, bpp);
        const bytes = self.gpa.alloc(u8, size) catch return error.OutOfMemory;
        errdefer self.gpa.free(bytes);
        @memset(bytes, 0);
        const res_id = self.allocResId();
        // Only request BIND_SCANOUT for 4-byte (8/10-bit) scanout-capable formats.
        // An fp16 (8 bpp) HDR render target is not a scanout format. Asking the
        // host to make it scanout-able confuses virglrenderer's surface allocation
        // (the draw then misses the surface). HDR RTs are render-target only.
        const bind: u32 = if (enc.isDepthFormat(format))
            enc.BIND_DEPTH_STENCIL
        else
            vg.BIND_RENDER_TARGET | enc.BIND_SAMPLER_VIEW | (if (bpp == 4) vg.BIND_SCANOUT else 0);
        if (self.gpu.resourceCreate3D(res_id, vg.TEXTURE_2D, format, bind, width, height) != RESP_OK_NODATA)
            return error.InitializationFailed;
        if (self.gpu.attachBacking(res_id, @intFromPtr(bytes.ptr), @intCast(size)) != RESP_OK_NODATA)
            return error.InitializationFailed;
        if (self.gpu.ctxAttachResource(CTX_ID, res_id) != RESP_OK_NODATA)
            return error.InitializationFailed;
        return .{ .res_id = res_id, .bytes = bytes, .width = width, .height = height, .bpp = bpp };
    }

    pub fn destroyResource(self: *Transport, res: Resource) void {
        self.gpa.free(res.bytes);
    }

    /// The CPU-visible backing of a resource.
    pub fn map(self: *Transport, res: Resource) Error![]u8 {
        _ = self;
        return res.bytes;
    }

    /// Upload `len` bytes of a resource's guest backing to the host copy.
    pub fn transferToHost(self: *Transport, res: Resource, len: u32) Error!void {
        const box = types.Box{ .x = 0, .y = 0, .z = 0, .w = len, .h = 1, .d = 1 };
        if (self.gpu.transferToHost3D(CTX_ID, res.res_id, @bitCast(box), len) != RESP_OK_NODATA)
            return error.DeviceLost;
    }

    /// Read a render target's host pixels back into its guest backing. The stride
    /// is width * bytes-per-pixel so an fp16 (8 bpp) RT reads back correctly too.
    pub fn transferFromHost(self: *Transport, res: Resource) Error!void {
        const box = types.Box{ .x = 0, .y = 0, .z = 0, .w = res.width, .h = res.height, .d = 1 };
        if (self.gpu.transferFromHost3D(CTX_ID, res.res_id, @bitCast(box), res.width * res.bpp) != RESP_OK_NODATA)
            return error.DeviceLost;
    }

    /// Submit a virgl command stream (raw little-endian dword bytes in the scratch
    /// buffer) to the 3D context.
    pub fn submit(self: *Transport, stream_bytes: []const u8) Error!void {
        if (self.gpu.submit3d(CTX_ID, stream_bytes) != RESP_OK_NODATA)
            return error.DeviceLost;
    }

    /// Best-effort scanout + flush of a render target to display 0 (for a host
    /// that has a readable surface). No-op failures are ignored.
    pub fn present(self: *Transport, res: Resource) void {
        _ = self.gpu.setScanoutRes(0, res.res_id, res.width, res.height);
        _ = self.gpu.flushRes(res.res_id, res.width, res.height);
    }
};
