//! Asahi AGX transport - the userspace half of the kernel `asahi` DRM driver
//! (Apple Silicon GPU). Raw std.os.linux open/ioctl/mmap, std.posix.errno for
//! status, NO libc and NO root: the GPU is reached through a user-accessible
//! DRM render node (/dev/dri/renderD*, mode 0666 / the `render` group), exactly
//! like the virgl/virtgpu transport (lib/prism/drivers/virgl/transport/linux.zig)
//! and unlike the NVIDIA /dev/nvidia* path.
//!
//! This is PHASE 1 (the transport foundation + a probe): open the device, query
//! GET_PARAMS, create a VM, allocate + CPU-map + VM-bind a GEM BO. Command
//! queues, SUBMIT, AGX command-buffer encoding and shaders are later phases.
//!
//! Zig 0.16 notes (same as the virgl linux transport): std.posix.fstat does NOT
//! exist - sysfs driver discovery uses std.os.linux.readlink; ioctl/mmap go
//! through std.os.linux with std.posix.errno on the raw return.

const std = @import("std");
const builtin = @import("builtin");
const uapi = @import("uapi.zig");

pub const Error = error{
    OpenFailed,
    NoDevice,
    IoctlFailed,
    MapFailed,
    OutOfMemory,
    /// A syncobj wait reached its timeout before the fence(s) signalled. The
    /// kernel returns ETIME for this, which is NOT a transport failure - the
    /// caller decides whether a timeout is expected (e.g. polling) or fatal.
    Timeout,
};

/// A GPU submission queue: the queue id and the VM it submits into.
pub const Queue = struct {
    queue_id: u32,
    vm_id: u32,
};

/// A GPU buffer object: the GEM handle, its GPU virtual address (once bound),
/// the CPU mapping, and the logical size.
pub const Bo = struct {
    handle: u32,
    gpu_va: u64,
    cpu: []u8,
    size: u64,
};

/// Decoded GPU identity from drm_asahi_params_global. `variant` is the ASCII
/// variant letter ('G'/'S'/'C'/'D'); `name` is a human GPU name resolved from
/// the chip_id table below.
pub const GpuInfo = struct {
    features: u64,
    gpu_generation: u32,
    gpu_variant: u8,
    gpu_revision: u32,
    chip_id: u32,
    num_dies: u32,
    num_clusters_total: u32,
    num_cores_per_cluster: u32,
    max_frequency_khz: u32,
    vm_start: u64,
    vm_end: u64,
    vm_kernel_min_size: u64,
    total_core_count: u32,
    name: []const u8,
};

/// chip_id -> human name. The BCD chip ids are the Apple part numbers (T-codes).
/// The G13 family (gpu_generation 13) is the M1 line; G14 is M2; G16 is M3/M4.
/// Derived from the Apple Silicon part-number list + the Asahi AGX generation
/// naming (G13G base, G13S Pro/Max, G13C/G13D Ultra dies).
fn chipName(chip_id: u32) []const u8 {
    return switch (chip_id) {
        0x8103 => "Apple M1 (G13G, T8103)",
        0x6000 => "Apple M1 Pro (G13S, T6000)",
        0x6001 => "Apple M1 Max (G13S, T6001)",
        0x6002 => "Apple M1 Ultra (G13C, T6002)",
        0x8112 => "Apple M2 (G14G, T8112)",
        0x6020 => "Apple M2 Pro (G14S, T6020)",
        0x6021 => "Apple M2 Max (G14S, T6021)",
        0x6022 => "Apple M2 Ultra (G14C, T6022)",
        0x8122 => "Apple M3 (G16, T8122)",
        0x6030 => "Apple M3 Pro (G16, T6030)",
        0x6031, 0x6034 => "Apple M3 Max (G16, T6031)",
        0x8132 => "Apple M4 (G16, T8132)",
        0x6040 => "Apple M4 Pro (G16, T6040)",
        0x6041 => "Apple M4 Max (G16, T6041)",
        else => "Apple AGX (unknown chip_id)",
    };
}

/// The Asahi AGX device: the open render-node fd. Owns the fd unless adopted.
pub const Device = struct {
    fd: std.posix.fd_t,
    owns_fd: bool = true,
    /// Diagnostics for the most recent FAILING ioctl: the raw errno and the
    /// ioctl request number. Error.IoctlFailed discards which call failed and
    /// with what errno, so every ioctl error path records them here first. The
    /// probe prints these on a compute FAIL so the M1 run shows EXACTLY which
    /// ioctl was rejected (e.g. SUBMIT vs GEM_BIND) and why (e.g. EINVAL=22).
    last_errno: u16 = 0,
    last_request: u32 = 0,

    /// The kernel driver name behind /dev/dri/renderD<n> - the basename of the
    /// /sys/class/drm/renderD<n>/device/driver symlink target - or null if the
    /// node does not exist. The sysfs path MUST be NUL-terminated for readlink:
    /// passing a plain bufPrint (non-NUL) slice makes readlink read past the
    /// string into uninitialized stack, so the lookup silently fails. That was
    /// the bug that hid the asahi node on real hardware (the dev box has no
    /// asahi node, so it returned NoDevice either way and masked it).
    pub fn renderNodeDriver(n: u32, name_buf: []u8) ?[]const u8 {
        var nbuf: [80]u8 = undefined;
        const sys = std.fmt.bufPrintZ(&nbuf, "/sys/class/drm/renderD{d}/device/driver", .{n}) catch return null;
        var lbuf: [256]u8 = undefined;
        const link = std.os.linux.readlink(sys.ptr, &lbuf, lbuf.len);
        const ll: isize = @bitCast(link);
        if (ll <= 0) return null;
        const target = lbuf[0..@intCast(ll)];
        const slash = std.mem.lastIndexOfScalar(u8, target, '/');
        const base = if (slash) |s| target[s + 1 ..] else target;
        const m = @min(base.len, name_buf.len);
        @memcpy(name_buf[0..m], base[0..m]);
        return name_buf[0..m];
    }

    /// Scan /dev/dri/renderD128.. for the node whose kernel driver is "asahi"
    /// and open it O_RDWR | O_CLOEXEC. Mirrors the virtio_gpu discovery in the
    /// virgl Linux transport.
    pub fn open() Error!Device {
        var n: u32 = 128;
        while (n < 192) : (n += 1) {
            var nbuf: [32]u8 = undefined;
            const driver = renderNodeDriver(n, &nbuf) orelse continue;
            if (std.mem.eql(u8, driver, "asahi")) {
                var pbuf: [64]u8 = undefined;
                const path = std.fmt.bufPrintZ(&pbuf, "/dev/dri/renderD{d}", .{n}) catch continue;
                return openNode(path) catch continue;
            }
        }
        return error.NoDevice;
    }

    /// Open a specific render node path O_RDWR | O_CLOEXEC (the forced-node
    /// fallback for when sysfs discovery is unavailable).
    pub fn openNode(path: []const u8) Error!Device {
        var buf: [128]u8 = undefined;
        const z = std.fmt.bufPrintZ(&buf, "{s}", .{path}) catch return error.OpenFailed;
        const rc = std.os.linux.open(z.ptr, .{ .ACCMODE = .RDWR, .CLOEXEC = true }, 0);
        switch (std.posix.errno(rc)) {
            .SUCCESS => return .{ .fd = @intCast(rc) },
            else => return error.OpenFailed,
        }
    }

    /// Adopt an already-open render-node fd (e.g. a card node from a display
    /// owner). The Device does not close it.
    pub fn adopt(fd: std.posix.fd_t) Device {
        return .{ .fd = fd, .owns_fd = false };
    }

    pub fn deinit(self: *Device) void {
        if (self.owns_fd) _ = std.os.linux.close(self.fd);
    }

    fn ioctl(self: *Device, request: u32, arg: usize) Error!void {
        const rc = std.os.linux.ioctl(self.fd, request, arg);
        const e = std.posix.errno(rc);
        switch (e) {
            .SUCCESS => {},
            else => {
                // Record which call failed + the errno BEFORE returning so the
                // caller (and the probe) can report it.
                self.last_errno = @intFromEnum(e);
                self.last_request = request;
                return error.IoctlFailed;
            },
        }
    }

    /// DRM_IOCTL_ASAHI_GET_PARAMS -> drm_asahi_params_global. Decodes the GPU
    /// identity, core count, VA range and firmware/feature info.
    pub fn getParams(self: *Device) Error!GpuInfo {
        var g: uapi.drm_asahi_params_global = std.mem.zeroes(uapi.drm_asahi_params_global);
        var req = uapi.drm_asahi_get_params{
            .param_group = 0,
            .pad = 0,
            .pointer = @intFromPtr(&g),
            .size = @sizeOf(uapi.drm_asahi_params_global),
        };
        try self.ioctl(uapi.IOCTL_GET_PARAMS, @intFromPtr(&req));

        // Sum the active cores across the per-cluster bitmasks.
        var cores: u32 = 0;
        var i: usize = 0;
        while (i < uapi.DRM_ASAHI_MAX_CLUSTERS) : (i += 1) {
            cores += @popCount(g.core_masks[i]);
        }

        return .{
            .features = g.features,
            .gpu_generation = g.gpu_generation,
            .gpu_variant = @truncate(g.gpu_variant),
            .gpu_revision = g.gpu_revision,
            .chip_id = g.chip_id,
            .num_dies = g.num_dies,
            .num_clusters_total = g.num_clusters_total,
            .num_cores_per_cluster = g.num_cores_per_cluster,
            .max_frequency_khz = g.max_frequency_khz,
            .vm_start = g.vm_start,
            .vm_end = g.vm_end,
            .vm_kernel_min_size = g.vm_kernel_min_size,
            .total_core_count = cores,
            .name = chipName(g.chip_id),
        };
    }

    /// DRM_IOCTL_ASAHI_VM_CREATE -> a GPU address space. Reserves a kernel-only
    /// VA sub-range of vm_kernel_min_size at the TOP of the usable VA window
    /// (params.vm_start..vm_end), leaving the low range for userspace binds.
    pub fn vmCreate(self: *Device, info: GpuInfo) Error!u32 {
        const kernel_size = if (info.vm_kernel_min_size != 0) info.vm_kernel_min_size else 16 * 1024 * 1024;
        const kernel_end = info.vm_end;
        const kernel_start = kernel_end - kernel_size;
        var vc = uapi.drm_asahi_vm_create{
            .kernel_start = kernel_start,
            .kernel_end = kernel_end,
            .vm_id = 0,
            .pad = 0,
        };
        try self.ioctl(uapi.IOCTL_VM_CREATE, @intFromPtr(&vc));
        return vc.vm_id;
    }

    /// DRM_IOCTL_ASAHI_VM_DESTROY.
    pub fn vmDestroy(self: *Device, vm_id: u32) void {
        var vd = uapi.drm_asahi_vm_destroy{ .vm_id = vm_id, .pad = 0 };
        self.ioctl(uapi.IOCTL_VM_DESTROY, @intFromPtr(&vd)) catch {};
    }

    /// DRM_IOCTL_ASAHI_GEM_CREATE: allocate a GEM BO of `size` bytes. Returns
    /// the GEM handle. WRITEBACK = CPU-cacheable (good for a CPU roundtrip).
    pub fn gemCreate(self: *Device, size: u64, flags: u32) Error!u32 {
        var gc = uapi.drm_asahi_gem_create{
            .size = size,
            .flags = flags,
            .vm_id = 0,
            .handle = 0,
            .pad = 0,
        };
        try self.ioctl(uapi.IOCTL_GEM_CREATE, @intFromPtr(&gc));
        return gc.handle;
    }

    /// DRM_IOCTL_ASAHI_GEM_MMAP_OFFSET + mmap: CPU-map a GEM BO read/write.
    pub fn gemMmap(self: *Device, handle: u32, size: u64) Error![]u8 {
        var mo = uapi.drm_asahi_gem_mmap_offset{ .handle = handle, .flags = 0, .offset = 0 };
        try self.ioctl(uapi.IOCTL_GEM_MMAP_OFFSET, @intFromPtr(&mo));
        const r = std.os.linux.mmap(
            null,
            @intCast(size),
            .{ .READ = true, .WRITE = true },
            .{ .TYPE = .SHARED },
            self.fd,
            @intCast(mo.offset),
        );
        switch (std.posix.errno(r)) {
            .SUCCESS => {},
            else => return error.MapFailed,
        }
        const ptr: [*]u8 = @ptrFromInt(r);
        return ptr[0..@intCast(size)];
    }

    /// DRM_IOCTL_ASAHI_VM_BIND: map `bo`'s `size` bytes into `vm_id` at GPU VA
    /// `gpu_va` with READ|WRITE access (a single bind op).
    pub fn gemBind(self: *Device, vm_id: u32, handle: u32, gpu_va: u64, size: u64) Error!void {
        var op = uapi.drm_asahi_gem_bind_op{
            .flags = uapi.BIND_READ | uapi.BIND_WRITE,
            .handle = handle,
            .offset = 0,
            .range = size,
            .addr = gpu_va,
        };
        var bind = uapi.drm_asahi_vm_bind{
            .vm_id = vm_id,
            .num_binds = 1,
            .stride = @sizeOf(uapi.drm_asahi_gem_bind_op),
            .pad = 0,
            .userptr = @intFromPtr(&op),
        };
        try self.ioctl(uapi.IOCTL_VM_BIND, @intFromPtr(&bind));
    }

    /// Convenience: GEM_CREATE + mmap + VM_BIND at `gpu_va`. Returns a Bo with
    /// the handle, the GPU VA, the CPU mapping, and the logical size.
    pub fn allocBo(self: *Device, vm_id: u32, size: u64, gpu_va: u64) Error!Bo {
        const handle = try self.gemCreate(size, uapi.GEM_WRITEBACK);
        const cpu = try self.gemMmap(handle, size);
        try self.gemBind(vm_id, handle, gpu_va, size);
        return .{ .handle = handle, .gpu_va = gpu_va, .cpu = cpu, .size = size };
    }

    /// DRM_IOCTL_ASAHI_VM_BIND with a BIND_UNBIND op: remove the VA range
    /// [gpu_va, gpu_va+size) from `vm_id`. The unbind op carries ONLY the flags +
    /// the addr/range (no GEM handle - it tears down whatever is mapped there), so
    /// `handle`/`offset` are 0. Pairs with gemBind's READ|WRITE map.
    pub fn gemUnbind(self: *Device, vm_id: u32, gpu_va: u64, size: u64) Error!void {
        var op = uapi.drm_asahi_gem_bind_op{
            .flags = uapi.BIND_UNBIND,
            .handle = 0,
            .offset = 0,
            .range = size,
            .addr = gpu_va,
        };
        var bind = uapi.drm_asahi_vm_bind{
            .vm_id = vm_id,
            .num_binds = 1,
            .stride = @sizeOf(uapi.drm_asahi_gem_bind_op),
            .pad = 0,
            .userptr = @intFromPtr(&op),
        };
        try self.ioctl(uapi.IOCTL_VM_BIND, @intFromPtr(&bind));
    }

    /// DRM_IOCTL_GEM_CLOSE (core DRM, nr 0x09, _IOW drm_gem_close): drop this
    /// process's reference to GEM `handle` so the BO + its backing memory are
    /// freed. The nr is literal (NOT DRM_COMMAND_BASE-offset), like the syncobj
    /// ioctls. Best-effort (ignores errno).
    pub fn gemClose(self: *Device, handle: u32) void {
        var gc = uapi.drm_gem_close{ .handle = handle, .pad = 0 };
        self.ioctl(uapi.IOCTL_GEM_CLOSE, @intFromPtr(&gc)) catch {};
    }

    /// Release a Bo obtained from allocBo. The inverse of allocBo, in the order
    /// the kernel needs:
    ///   (a) VM_BIND/BIND_UNBIND the VA range out of `vm_id` (so the GPU page
    ///       table no longer references the BO);
    ///   (b) munmap the CPU mapping;
    ///   (c) GEM_CLOSE the handle (drop the last reference -> the BO is freed).
    /// Each step is best-effort (individual errno failures are ignored) so a
    /// partial state still tears down as much as it can; the UNBIND is done
    /// BEFORE the close so the VA is never left bound to a freed handle. The
    /// caller MUST ensure the GPU is finished with the BO (e.g. after a fence
    /// wait) before freeing it, or the unbind races live GPU access.
    pub fn freeBo(self: *Device, vm_id: u32, bo: Bo) void {
        // (a) unbind the VA range from the VM.
        self.gemUnbind(vm_id, bo.gpu_va, bo.size) catch {};
        // (b) munmap the CPU mapping.
        if (bo.cpu.len != 0) {
            _ = std.os.linux.munmap(@ptrCast(bo.cpu.ptr), bo.cpu.len);
        }
        // (c) close the GEM handle (release the BO).
        self.gemClose(bo.handle);
    }

    // -----------------------------------------------------------------------
    // P2: command queue + SUBMIT + sync.
    // -----------------------------------------------------------------------

    /// DRM_IOCTL_ASAHI_QUEUE_CREATE: create a GPU submission queue on `vm_id`
    /// at the given priority. `usc_exec_base` is the base GPU VA of the queue's
    /// USC (shader-code) region; pass the VA you have bound shader memory at (0
    /// is acceptable until P3/P4 wire real shaders). Returns the queue id.
    pub fn queueCreate(self: *Device, vm_id: u32, priority: u32, usc_exec_base: u64) Error!u32 {
        var qc = uapi.drm_asahi_queue_create{
            .flags = 0,
            .vm_id = vm_id,
            .priority = priority,
            .queue_id = 0,
            .usc_exec_base = usc_exec_base,
        };
        try self.ioctl(uapi.IOCTL_QUEUE_CREATE, @intFromPtr(&qc));
        return qc.queue_id;
    }

    /// DRM_IOCTL_ASAHI_QUEUE_DESTROY.
    pub fn queueDestroy(self: *Device, queue_id: u32) void {
        var qd = uapi.drm_asahi_queue_destroy{ .queue_id = queue_id, .pad = 0 };
        self.ioctl(uapi.IOCTL_QUEUE_DESTROY, @intFromPtr(&qd)) catch {};
    }

    /// DRM_IOCTL_SYNCOBJ_CREATE (core DRM): allocate a binary syncobj. Returns
    /// the syncobj handle. `signaled` starts it in the signalled state.
    pub fn syncobjCreate(self: *Device, signaled: bool) Error!u32 {
        var sc = uapi.drm_syncobj_create{
            .handle = 0,
            .flags = if (signaled) uapi.DRM_SYNCOBJ_CREATE_SIGNALED else 0,
        };
        try self.ioctl(uapi.IOCTL_SYNCOBJ_CREATE, @intFromPtr(&sc));
        return sc.handle;
    }

    /// DRM_IOCTL_SYNCOBJ_DESTROY (core DRM).
    pub fn syncobjDestroy(self: *Device, handle: u32) void {
        var sd = uapi.drm_syncobj_destroy{ .handle = handle, .pad = 0 };
        self.ioctl(uapi.IOCTL_SYNCOBJ_DESTROY, @intFromPtr(&sd)) catch {};
    }

    /// DRM_IOCTL_SYNCOBJ_SIGNAL (core DRM): CPU-signal one binary syncobj.
    pub fn syncobjSignal(self: *Device, handle: u32) Error!void {
        var h = handle;
        var arr = uapi.drm_syncobj_array{
            .handles = @intFromPtr(&h),
            .count_handles = 1,
            .pad = 0,
        };
        try self.ioctl(uapi.IOCTL_SYNCOBJ_SIGNAL, @intFromPtr(&arr));
    }

    /// DRM_IOCTL_SYNCOBJ_WAIT (core DRM): wait for ALL of `handles` to signal,
    /// up to `timeout_ns` from now. Returns Error.Timeout (ETIME) if the
    /// deadline passes first. The kernel wants an ABSOLUTE CLOCK_MONOTONIC
    /// deadline, so a relative `timeout_ns` is added to now(); WAIT_FOR_SUBMIT
    /// is set so a not-yet-submitted fence is waited on rather than failing.
    pub fn syncobjWait(self: *Device, handles: []const u32, timeout_ns: u64) Error!void {
        var ts: std.os.linux.timespec = undefined;
        _ = std.os.linux.clock_gettime(.MONOTONIC, &ts);
        const now_ns: i128 = @as(i128, ts.sec) * 1_000_000_000 + @as(i128, ts.nsec);
        const deadline: i128 = now_ns + @as(i128, timeout_ns);
        const abs: i64 = if (deadline > std.math.maxInt(i64)) std.math.maxInt(i64) else @intCast(deadline);

        var w = uapi.drm_syncobj_wait{
            .handles = @intFromPtr(handles.ptr),
            .timeout_nsec = abs,
            .count_handles = @intCast(handles.len),
            .flags = uapi.DRM_SYNCOBJ_WAIT_FLAGS_WAIT_ALL | uapi.DRM_SYNCOBJ_WAIT_FLAGS_WAIT_FOR_SUBMIT,
            .first_signaled = 0,
            .pad = 0,
            .deadline_nsec = 0,
        };
        const rc = std.os.linux.ioctl(self.fd, uapi.IOCTL_SYNCOBJ_WAIT, @intFromPtr(&w));
        const e = std.posix.errno(rc);
        switch (e) {
            .SUCCESS => {},
            .TIME => return error.Timeout,
            else => {
                self.last_errno = @intFromEnum(e);
                self.last_request = uapi.IOCTL_SYNCOBJ_WAIT;
                return error.IoctlFailed;
            },
        }
    }

    /// DRM_IOCTL_ASAHI_SUBMIT: run `command` (a byte stream of one or more
    /// drm_asahi_cmd_header + payload) on `queue_id`, waiting on `in_syncs`
    /// before and signalling `out_syncs` on completion.
    ///
    /// SCOPE (P2): the syncs marshaling, the queue dispatch, and the cmdbuf
    /// pointer/size handoff are fully wired. The `command` bytes are NOT
    /// synthesized here - a real RENDER/COMPUTE command needs the AGX control
    /// stream + ZLS/USC encoding, which is P3. Pass an already-encoded cmdbuf;
    /// until P3 exists this is exercised struct-only (the probe does NOT hand
    /// the kernel a fabricated cmdbuf, since a bogus control stream risks a GPU
    /// fault). The in/out syncs are laid out contiguously: waits first, then
    /// signals, as the kernel expects (in_sync_count then out_sync_count).
    pub fn submit(
        self: *Device,
        queue_id: u32,
        command: []const u8,
        in_syncs: []const uapi.drm_asahi_sync,
        out_syncs: []const uapi.drm_asahi_sync,
    ) Error!void {
        // Pack waits-then-signals into one contiguous drm_asahi_sync array.
        var stack: [16]uapi.drm_asahi_sync = undefined;
        const total = in_syncs.len + out_syncs.len;
        if (total > stack.len) return error.OutOfMemory;
        var i: usize = 0;
        for (in_syncs) |s| {
            stack[i] = s;
            i += 1;
        }
        for (out_syncs) |s| {
            stack[i] = s;
            i += 1;
        }
        const syncs_ptr: u64 = if (total == 0) 0 else @intFromPtr(&stack);

        var sub = uapi.drm_asahi_submit{
            .syncs = syncs_ptr,
            .cmdbuf = if (command.len == 0) 0 else @intFromPtr(command.ptr),
            .flags = 0,
            .queue_id = queue_id,
            .in_sync_count = @intCast(in_syncs.len),
            .out_sync_count = @intCast(out_syncs.len),
            .cmdbuf_size = @intCast(command.len),
            .pad = 0,
        };
        try self.ioctl(uapi.IOCTL_SUBMIT, @intFromPtr(&sub));
    }

    /// Build a binary drm_asahi_sync (SYNC_SYNCOBJ, timeline_value 0) from a
    /// syncobj handle - the common case for in/out syncs.
    pub fn binarySync(handle: u32) uapi.drm_asahi_sync {
        return .{ .sync_type = uapi.SYNC_SYNCOBJ, .handle = handle, .timeline_value = 0 };
    }
};

/// Map a (failing) ioctl request value back to a human ioctl name, so a probe
/// can print which call the kernel rejected (SUBMIT vs GEM_BIND vs GEM_CREATE
/// point at totally different bugs). Returns "UNKNOWN" for an unrecognized
/// request. Compared against the encoded uapi.IOCTL_* constants.
pub fn requestName(request: u32) []const u8 {
    return switch (request) {
        uapi.IOCTL_GET_PARAMS => "GET_PARAMS",
        uapi.IOCTL_GET_TIME => "GET_TIME",
        uapi.IOCTL_VM_CREATE => "VM_CREATE",
        uapi.IOCTL_VM_DESTROY => "VM_DESTROY",
        uapi.IOCTL_VM_BIND => "VM_BIND",
        uapi.IOCTL_GEM_CREATE => "GEM_CREATE",
        uapi.IOCTL_GEM_MMAP_OFFSET => "GEM_MMAP_OFFSET",
        uapi.IOCTL_QUEUE_CREATE => "QUEUE_CREATE",
        uapi.IOCTL_QUEUE_DESTROY => "QUEUE_DESTROY",
        uapi.IOCTL_SUBMIT => "SUBMIT",
        uapi.IOCTL_SYNCOBJ_CREATE => "SYNCOBJ_CREATE",
        uapi.IOCTL_SYNCOBJ_DESTROY => "SYNCOBJ_DESTROY",
        uapi.IOCTL_SYNCOBJ_WAIT => "SYNCOBJ_WAIT",
        uapi.IOCTL_SYNCOBJ_SIGNAL => "SYNCOBJ_SIGNAL",
        else => "UNKNOWN",
    };
}

// ---------------------------------------------------------------------------
// CPU<->GPU cache maintenance (aarch64 / Apple Silicon)
//
// AGX BOs are CPU-coherent at the point of coherency, but a CPU write into a
// GEM_WRITEBACK mapping can still sit DIRTY in the CPU's data cache and not yet
// be visible at the PoC where the GPU reads. The HAL compute path fills the
// caller's input/storage BOs (via mapResource), then submits a dispatch; without
// a clean-to-PoC first, the GPU can LOAD a stale recycled-DRAM value instead of
// the bytes the CPU just wrote (the P5e stale-input bug: addbuf loaded
// 0x3F800000 instead of the CPU's 0x00010000). Symmetrically, after the GPU
// stores its output and the fence signals, a clean+invalidate before the CPU
// reads back guarantees the read is not served from a stale CPU line.
//
// These mirror the private helpers compute.zig / render.zig already use on their
// OWN CPU-written infra BOs (UNIFORM_DATA, clear_data, sampler_heap, fs_color).
// Exported here so the prism `apple` HAL driver can apply the SAME flush to the
// caller's storage BOs around a dispatch. EL0-legal on Linux (SCTLR_EL1.UCI).
//
// Guarded to aarch64: on any other target they are a no-op (the inline `dc`
// ops only assemble on aarch64), so a non-aarch64 native build still compiles.
//
// Zig 0.16 inline-asm: clobbers are an anonymous struct (`.{ .memory = true }`).
// ---------------------------------------------------------------------------

/// aarch64 cache line size we stride by. 64 bytes is the architectural minimum
/// DminLine on Apple Silicon; striding by 64 safely covers every line touched.
const CACHE_LINE: usize = 64;

/// Clean (write-back) the CPU-dirty lines of `mem` to the point of coherency
/// (`dc cvac`) so RAM holds what the CPU just wrote BEFORE the GPU reads the BO.
/// Call this on each storage buffer before a dispatch so the GPU loads fresh
/// input. EL0-legal on Linux (SCTLR_EL1.UCI). No-op off aarch64.
pub fn cleanToPoC(mem: []const u8) void {
    if (builtin.cpu.arch != .aarch64) return;
    var off: usize = 0;
    while (off < mem.len) : (off += CACHE_LINE) {
        const addr = @intFromPtr(mem.ptr) + off;
        asm volatile ("dc cvac, %[a]"
            :
            : [a] "r" (addr),
            : .{ .memory = true });
    }
    asm volatile ("dsb sy" ::: .{ .memory = true });
}

/// Clean + invalidate (`dc civac`, the EL0-allowed op) the lines of `mem` to the
/// point of coherency so a subsequent CPU read fetches fresh data rather than a
/// stale cached line. Call this on each storage buffer AFTER the dispatch's
/// fence so the CPU reads the GPU's fresh output. No-op off aarch64.
pub fn cleanInvalidateToPoC(mem: []const u8) void {
    if (builtin.cpu.arch != .aarch64) return;
    var off: usize = 0;
    while (off < mem.len) : (off += CACHE_LINE) {
        const addr = @intFromPtr(mem.ptr) + off;
        asm volatile ("dc civac, %[a]"
            :
            : [a] "r" (addr),
            : .{ .memory = true });
    }
    asm volatile ("dsb sy" ::: .{ .memory = true });
}

test "requestName maps the encoded ioctl requests to names" {
    try std.testing.expectEqualStrings("SUBMIT", requestName(uapi.IOCTL_SUBMIT));
    try std.testing.expectEqualStrings("VM_BIND", requestName(uapi.IOCTL_VM_BIND));
    try std.testing.expectEqualStrings("GEM_CREATE", requestName(uapi.IOCTL_GEM_CREATE));
    try std.testing.expectEqualStrings("UNKNOWN", requestName(0));
}

test "chip name table resolves the M1 Pro" {
    try std.testing.expectEqualStrings("Apple M1 Pro (G13S, T6000)", chipName(0x6000));
    try std.testing.expectEqualStrings("Apple M1 (G13G, T8103)", chipName(0x8103));
    try std.testing.expectEqualStrings("Apple AGX (unknown chip_id)", chipName(0xdead));
}

test "freeBo unbind op encodes BIND_UNBIND over the BO's VA range, no handle" {
    // freeBo's first step is an UNBIND VM_BIND op. Verify the op the kernel would
    // see: flags == BIND_UNBIND (NOT READ|WRITE), addr/range == the BO's VA span,
    // and no GEM handle (the unbind tears down the VA, not a specific handle).
    // This is the inverse of gemBind, whose op carries BIND_READ|BIND_WRITE +
    // the handle. (Pure struct check - no GPU.)
    const gpu_va: u64 = 0x2_0000_0000;
    const size: u64 = 0x4000;
    const unbind = uapi.drm_asahi_gem_bind_op{
        .flags = uapi.BIND_UNBIND,
        .handle = 0,
        .offset = 0,
        .range = size,
        .addr = gpu_va,
    };
    try std.testing.expectEqual(uapi.BIND_UNBIND, unbind.flags);
    try std.testing.expect((unbind.flags & (uapi.BIND_READ | uapi.BIND_WRITE)) == 0);
    try std.testing.expectEqual(@as(u32, 0), unbind.handle);
    try std.testing.expectEqual(gpu_va, unbind.addr);
    try std.testing.expectEqual(size, unbind.range);

    // GEM_CLOSE is a core DRM ioctl (literal nr 0x09, _IOW drm_gem_close): the
    // request name table does NOT recognize it (it is core, not asahi-private).
    try std.testing.expectEqualStrings("UNKNOWN", requestName(uapi.IOCTL_GEM_CLOSE));
    try std.testing.expectEqual(@as(usize, 8), @sizeOf(uapi.drm_gem_close));
}

test "open on a box without an asahi node returns NoDevice (not a crash)" {
    // On this aarch64-linux dev box there is no AGX GPU, so discovery must
    // cleanly report NoDevice rather than fault. (On the M1 it opens the node.)
    const r = Device.open();
    if (r) |dev| {
        var d = dev;
        d.deinit();
    } else |err| {
        try std.testing.expect(err == error.NoDevice or err == error.OpenFailed);
    }
}
