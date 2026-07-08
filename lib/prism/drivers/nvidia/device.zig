const std = @import("std");
const nvidia = @import("nvidia");
const hal = @import("../../hal.zig");
const Resource = @import("resource.zig").Resource;
const ShaderModule = @import("shader.zig").ShaderModule;
const Pipeline = @import("pipeline.zig").Pipeline;

/// Map an RM (subproject/nvidia) error onto the Prism-wide error set.
pub fn nverr(e: anyerror) hal.Error {
    return switch (e) {
        error.OpenFailed, error.BadVersion, error.NoDevice => error.InitializationFailed,
        error.RmAllocFailed => error.OutOfMemory,
        error.IoctlFailed, error.ControlFailed, error.MapFailed => error.DeviceLost,
        else => error.InitializationFailed,
    };
}

/// A freed memory chunk kept for reuse instead of being returned to the kernel.
/// Render-target churn during a window resize would otherwise alloc + GPU-map +
/// CPU-map a fresh buffer every frame. Pooling keeps both mappings live, so
/// reuse costs zero ioctls. `mem.size` is the chunk's capacity.
const Chunk = struct { mem: nvidia.Memory, va: u64, va_span: u64, mapping: ?nvidia.Mapping, gpu: nvidia.GpuMapping };

/// A reserved (or freed) GPU virtual-address range: `[va, va + size)`.
const VaRange = struct { va: u64, size: u64 };

/// The result of a VA reservation: the (aligned) base and the span actually consumed
/// (`alignForward(size, alignment)`), which freeVa returns to the free list verbatim.
const VaAlloc = struct { va: u64, span: u64 };

/// A HAL device backed by a real NVIDIA GPU: owns the RM client, the device
/// handle, and a GPU virtual-address space. Resources are device memory mapped
/// into that space. GPU VAs are handed out by a bump allocator. Freed
/// resource memory is recycled through `pool` rather than freed immediately.
pub const Device = struct {
    gpa: std.mem.Allocator,
    client: nvidia.Client,
    dev: nvidia.Device,
    vaspace: u32,
    next_va: u64 = 0x10000000,
    /// Freed GPU VA ranges available for reuse. `next_va` is a bump pointer that only
    /// ever climbs. Without recycling, a long-lived device (a UI framework that creates
    /// and destroys contexts/resources over its lifetime) marches `next_va` past the VA
    /// space limit and the next NV01_MEMORY_VIRTUAL reservation fails with OutOfMemory,
    /// reproducibly after a few dozen context cycles. reserveVa carves from this list
    /// first (first-fit, returning the head/tail remainder) and only bumps when nothing
    /// fits. freeVa returns a range here. Same-size churn reuses ranges exactly, so the
    /// list stays small.
    free_va: std.ArrayListUnmanaged(VaRange) = .empty,
    pool: [4]?Chunk = .{ null, null, null, null },
    /// The GPU's ASCII model name (e.g. "NVIDIA GeForce RTX 5070"), queried once
    /// at create time and stored so caps() can hand out a stable slice. `name` is
    /// the populated subslice. The rest is the backing buffer.
    name_buf: [64]u8 = [_]u8{0} ** 64,
    name: []const u8 = "NVIDIA GPU",
    /// Lazily-created copy-engine + a cached sysmem detile buffer, used by
    /// readbackPresent to detile a block-linear RT on the GPU (CE) instead of the
    /// CPU reading write-combined VRAM. Both freed in deinit. detile_buf is reused
    /// across frames and grown as needed.
    ce: ?*@import("ce.zig").CopyEngine = null,
    detile_buf: ?GpuBuf = null,
    /// Retired shader heaps awaiting free. A destroyed pipeline's shader heap is not freed
    /// immediately: the GPU channel retains SET_PIPELINE_PROGRAM_ADDRESS pointing at it, and a
    /// speculative shader prefetch during the idle window between submits reads that VA, so
    /// unmapping it (and recycling the sysmem VA for the next pipeline) mid-flight faults Xid 31
    /// (GPCCLIENT_GCC MMU VIRT_READ). Instead the heap is parked here (staying mapped) and freed at
    /// the start of the next submit, after the prior submit has fenced (GPU idle past any reference)
    /// and before that submit rebinds the shader, so its VA is unmapped only once nothing points at
    /// it. flushRetired() is the drain. Also drained at deinit.
    retired_heaps: std.ArrayListUnmanaged(GpuBuf) = .empty,
    /// Occlusion-query (ZPASS pixel count) report buffer. Each draw submit enables ZPASS counting and
    /// writes the running hardware count here via a REPORT_SEMAPHORE. occlusionSampleCount reads the
    /// 64-bit count so GL_ANY_SAMPLES_PASSED gets a real begin/end delta (not a conservative "passed").
    /// A 4-word report {count64, timestamp64}. The count is the first u64. Lazily allocated, freed in deinit.
    occlusion_buf: ?GpuBuf = null,
    /// Lazy dedicated context (channel) for GPU transform-feedback capture. captureTransformFeedback
    /// is a Device method but a stream-out draw needs a channel. The pipeline/vertex/UBO/output are
    /// all device resources bindable from any context, so a cached capture context serves. Freed in deinit.
    tf_ctx: ?hal.Context = null,

    pub fn create(gpa: std.mem.Allocator) hal.Error!hal.Device {
        const self = gpa.create(Device) catch return error.OutOfMemory;
        errdefer gpa.destroy(self);
        self.gpa = gpa;
        self.next_va = 0x10000000;
        // gpa.create does not apply struct field defaults (see ce/detile_buf below), so
        // the VA free list must be initialized explicitly or it reads as garbage.
        self.free_va = .empty;
        self.pool = .{ null, null, null, null };
        self.name = "NVIDIA GPU";
        // gpa.create does not apply struct field defaults, so these must be set
        // explicitly. Otherwise self.ce reads as garbage (non-null) and the present
        // path / deinit dereferences it (segfault at 0xaa..aa).
        self.ce = null;
        self.detile_buf = null;
        self.retired_heaps = .empty;
        self.occlusion_buf = null;
        self.tf_ctx = null;
        self.client = nvidia.Client.open() catch |e| return nverr(e);
        errdefer self.client.deinit();
        self.dev = self.client.allocDevice(0) catch |e| return nverr(e);
        errdefer self.client.freeDevice(self.dev);
        self.vaspace = self.client.allocVaSpace(self.dev) catch |e| return nverr(e);
        // Best-effort GPU model name (RM NV2080_CTRL_CMD_GPU_GET_NAME_STRING). On
        // any failure we keep the "NVIDIA GPU" fallback rather than fail bring-up.
        if (self.client.getGpuName(self.dev, &self.name_buf)) |n| {
            if (n.len > 0) self.name = self.name_buf[0..n.len];
        } else |_| {}
        return .{ .ptr = self, .vtable = &vtable };
    }

    /// Formats a real NVIDIA GPU render/texture path supports, including the HDR
    /// fp16 + 10-bit scanout formats (rgba16_float / rgb10a2 / rgb10x2).
    /// The DRM XR30 HDR10 path is implemented in platform/drm.zig.
    const supported_formats = [_]hal.Format{
        .rgba8_unorm,  .bgra8_unorm, .r8_unorm, .depth32_float,
        .rgba16_float, .rgb10a2,     .rgb10x2,
    };

    fn caps(ptr: *anyopaque) hal.DeviceCaps {
        const self: *Device = @ptrCast(@alignCast(ptr));
        return .{
            .device_name = self.name,
            .formats = &supported_formats,
            .hdr = hal.DeviceCaps.deriveHdr(&supported_formats),
            .spirv = true,
            .compute = true,
            .graphics = true,
            .present = true,
            // Ampere/Ada/Blackwell allow up to 32768 for 2D textures.
            .max_texture_dim = 32768,
        };
    }

    /// Reserve a GPU virtual address for `size` bytes. Small buffers are 64 KB
    /// aligned. Buffers >= 2 MB are 2 MB aligned, because RM maps large VRAM
    /// allocations with big (2 MB) GPU pages and rejects a sub-big-page-aligned VA.
    fn reserveVa(self: *Device, size: u64) VaAlloc {
        const big: u64 = 0x200000;
        const al: u64 = if (size >= big) big else 0x10000;
        return self.reserveVaAligned(size, al);
    }

    /// Reserve `size` bytes of GPU VA aligned to `al`, reusing a freed range when one
    /// fits (first-fit: the head before the aligned base and the tail after the span go
    /// back on the free list) and only bumping `next_va` when nothing fits. Returns the
    /// aligned base and the consumed span (`alignForward(size, al)`). Pass both to freeVa
    /// to release them.
    fn reserveVaAligned(self: *Device, size: u64, al: u64) VaAlloc {
        const span = std.mem.alignForward(u64, size, al);
        var i: usize = 0;
        while (i < self.free_va.items.len) : (i += 1) {
            const r = self.free_va.items[i];
            const start = std.mem.alignForward(u64, r.va, al);
            const head = start - r.va;
            if (head + span <= r.size) {
                _ = self.free_va.orderedRemove(i);
                if (head > 0) self.free_va.append(self.gpa, .{ .va = r.va, .size = head }) catch {};
                const tail_va = start + span;
                const tail = r.va + r.size - tail_va;
                if (tail > 0) self.free_va.append(self.gpa, .{ .va = tail_va, .size = tail }) catch {};
                return .{ .va = start, .span = span };
            }
        }
        const va = std.mem.alignForward(u64, self.next_va, al);
        self.next_va = va + span;
        return .{ .va = va, .span = span };
    }

    /// Return a reserved VA range to the free list for reuse. A failed append (OOM)
    /// forgets the range (correctness is preserved, it just won't be reused),
    /// so VA reservation can never be blocked by a bookkeeping allocation failure.
    fn freeVa(self: *Device, va: u64, span: u64) void {
        if (span == 0) return;
        self.free_va.append(self.gpa, .{ .va = va, .size = span }) catch {};
    }

    pub const GpuBuf = struct {
        mem: nvidia.Memory,
        va: u64,
        /// The VA span reserved for this buffer (>= the mapped size, alignment-rounded),
        /// returned to the device's free list by freeGpu so the range can be reused.
        va_span: u64 = 0,
        bytes: []u8,
        // The GPU VA binding + CPU mmap backing this buffer. Both are released by
        // freeGpu. Leaking them exhausts the kernel's VA/page-table resources and
        // open fds, so later allocations fail (NV status 0x36).
        gpu: nvidia.GpuMapping,
        cpu: ?nvidia.Mapping = null,
    };

    /// Allocate device memory, map it into this device's GPU address space, and
    /// CPU-map it. Used by the context for channel infrastructure (gpfifo,
    /// pushbuffer, semaphore). Caller frees with freeGpu.
    pub fn allocGpu(self: *Device, loc: nvidia.Memory.Location, size: u64) hal.Error!GpuBuf {
        const mem = self.client.allocMemory(self.dev, loc, size) catch |e| return nverr(e);
        errdefer self.client.freeMemory(self.dev, mem);
        const r = self.reserveVa(size);
        errdefer self.freeVa(r.va, r.span);
        const gm = self.client.mapToGpu(self.dev, self.vaspace, mem, r.va) catch |e| return nverr(e);
        errdefer self.client.unmapFromGpu(self.dev, gm);
        const m = self.client.mapMemory(self.dev, mem) catch |e| return nverr(e);
        return .{ .mem = mem, .va = r.va, .va_span = r.span, .bytes = m.bytes, .gpu = gm, .cpu = m };
    }
    pub fn freeGpu(self: *Device, buf: GpuBuf) void {
        if (buf.cpu) |c| self.client.unmapMemory(c);
        self.client.unmapFromGpu(self.dev, buf.gpu);
        self.client.freeMemory(self.dev, buf.mem);
        self.freeVa(buf.va, buf.va_span);
    }

    /// Park a shader heap for deferred free (see retired_heaps). Falls back to an immediate free if
    /// the parking list cannot grow. Freeing early is still correct on a fenced/idle channel.
    /// The deferral is a precaution against the speculative-prefetch window.
    pub fn retireHeap(self: *Device, buf: GpuBuf) void {
        self.retired_heaps.append(self.gpa, buf) catch {
            self.freeGpu(buf);
        };
    }

    /// Free all retired shader heaps. Called at the start of each submit (the prior submit has
    /// fenced, so the GPU is idle past any reference, and the current submit rebinds the shader
    /// before any draw) and at deinit.
    pub fn flushRetired(self: *Device) void {
        for (self.retired_heaps.items) |buf| self.freeGpu(buf);
        self.retired_heaps.clearRetainingCapacity();
    }

    /// Get-or-create the occlusion (ZPASS) report buffer's GPU VA (a draw submit writes the ZPASS
    /// count here). 64 bytes covers the 4-word report with headroom. Returns null if allocation fails.
    pub fn ensureOcclusionBuf(self: *Device) ?u64 {
        if (self.occlusion_buf == null) {
            self.occlusion_buf = self.allocGpu(.system_wc, 64) catch return null;
            @memset(self.occlusion_buf.?.bytes[0..64], 0);
        }
        return self.occlusion_buf.?.va;
    }

    /// HAL occlusionSampleCount: the cumulative ZPASS count as of the last draw submit's report. The
    /// GLES occlusion query snapshots this at begin/end. The delta is the samples that passed. 0 until
    /// the first reporting submit (a correct start-of-count baseline).
    fn occlusionSampleCount(ptr: *anyopaque) u64 {
        const self: *Device = @ptrCast(@alignCast(ptr));
        const b = self.occlusion_buf orelse return 0;
        return std.mem.readInt(u64, b.bytes[0..8], .little);
    }

    /// HAL captureTransformFeedback: delegate to a lazy capture context's GPU stream-out draw.
    fn captureTransformFeedback(ptr: *anyopaque, cap: hal.TransformFeedbackCapture) hal.Error!usize {
        const self: *Device = @ptrCast(@alignCast(ptr));
        if (self.tf_ctx == null) self.tf_ctx = try @import("context.zig").Context.create(self);
        const ctx: *@import("context.zig").Context = @ptrCast(@alignCast(self.tf_ctx.?.ptr));
        return ctx.captureTf(cap);
    }

    /// Allocate a ZF32 depth (ZETA) surface in VRAM with the BLOCK_LINEAR page kind
    /// the ROP-Z requires, GPU-mapped without a CPU mapping (the CPU never reads depth
    /// back). The returned `bytes` is empty. Free with freeGpu. `size` is the tiled
    /// footprint (nvidia.graphics.ztSizeBytes).
    fn allocDepthGpu(self: *Device, width: u32, height: u32, size: u64) hal.Error!GpuBuf {
        return self.allocDepthGpuKind(width, height, size, 0);
    }

    /// allocDepthGpu with an explicit PTE page kind for the GPU mapping. `page_kind` 0 = the
    /// default GENERIC kind (ZF32 depth, works as-is). A non-zero kind (e.g. ZF32_X24S8=0x4)
    /// is needed for a stencil surface whose format kind is not GENERIC, or the ROP-Z faults.
    fn allocDepthGpuKind(self: *Device, width: u32, height: u32, size: u64, page_kind: u32) hal.Error!GpuBuf {
        // The page kind is set at allocation (NVOS32 `format`), not at the GPU map. The open
        // RM rejects a map-time NVOS46 kind override (INVALID_ARGUMENT).
        const mem = self.client.allocDepthMemoryKind(self.dev, width, height, size, page_kind) catch |e| return nverr(e);
        errdefer self.client.freeMemory(self.dev, mem);
        // Block-linear ZETA needs big GPU pages: align the VA to the big-page
        // boundary (2 MB) and map with PAGE_SIZE_BIG. A small-page (4 KB) mapping of
        // a block-linear surface faults the depth draw (Xid 69 / ErrorCode 0x9c).
        const big: u64 = 0x200000;
        const need = @max(size, mem.size);
        const r = self.reserveVaAligned(need, big);
        errdefer self.freeVa(r.va, r.span);
        const gm = self.client.mapToGpuBig(self.dev, self.vaspace, mem, r.va) catch |e| return nverr(e);
        return .{ .mem = mem, .va = r.va, .va_span = r.span, .bytes = &.{}, .gpu = gm, .cpu = null };
    }

    /// Allocate an S8 (stencil-only, single-plane, 1 B/px) ZETA for `w`x`h`, used by the context
    /// as the internal stencil surface for a stencil-only UI clip. Mapped with the GENERIC page
    /// kind (kind 0, as the open RM will not let us set a stencil-format kind). The S8 single-plane
    /// format is the untried case (Z24S8 / ZF32_X24S8 both fault Xid 69 with the generic kind).
    /// If the ROP accepts an S8-only ZETA with the generic kind, GPU stencil works without it.
    pub fn allocStencilZeta(self: *Device, w: u32, h: u32) hal.Error!GpuBuf {
        return self.allocDepthGpuKind(w, h, @as(u64, nvidia.graphics.ztSizeBytesBpp(w, h, 1)), 0);
    }

    /// Allocate a block-linear color render target in VRAM, GPU-map it with big
    /// pages (block-linear PTE kinds need big pages), and CPU-map it (the CPU reads
    /// the rendered pixels back, de-swizzling per nvidia.graphics.blColorPixelOffset).
    /// A render target paired with a ZETA must be block-linear or the depth draw
    /// faults (Xid 69 / 0x9c). `size` is the GOB-tiled footprint (ztSizeBytes).
    pub fn allocColorBlGpu(self: *Device, width: u32, height: u32, size: u64) hal.Error!GpuBuf {
        const mem = self.client.allocColorBlMemory(self.dev, width, height, size) catch |e| return nverr(e);
        errdefer self.client.freeMemory(self.dev, mem);
        const big: u64 = 0x200000;
        const need = @max(size, mem.size);
        const r = self.reserveVaAligned(need, big);
        errdefer self.freeVa(r.va, r.span);
        const gm = self.client.mapToGpuBig(self.dev, self.vaspace, mem, r.va) catch |e| return nverr(e);
        errdefer self.client.unmapFromGpu(self.dev, gm);
        const m = self.client.mapMemory(self.dev, mem) catch |e| return nverr(e);
        return .{ .mem = mem, .va = r.va, .va_span = r.span, .bytes = m.bytes, .gpu = gm, .cpu = m };
    }

    /// Get memory of at least `size` bytes: reuse a pooled chunk if one fits
    /// (allocation-free), otherwise allocate + GPU-map a fresh one.
    fn acquire(self: *Device, size: u64) hal.Error!Chunk {
        for (&self.pool) |*slot| {
            if (slot.*) |c| {
                if (c.mem.size >= size) {
                    slot.* = null;
                    return c;
                }
            }
        }
        const mem = self.client.allocMemory(self.dev, .system_wc, size) catch |e| return nverr(e);
        errdefer self.client.freeMemory(self.dev, mem);
        const r = self.reserveVa(size);
        errdefer self.freeVa(r.va, r.span);
        const gm = self.client.mapToGpu(self.dev, self.vaspace, mem, r.va) catch |e| return nverr(e);
        return .{ .mem = mem, .va = r.va, .va_span = r.span, .mapping = null, .gpu = gm };
    }

    fn freeChunk(self: *Device, c: Chunk) void {
        if (c.mapping) |m| self.client.unmapMemory(m);
        self.client.unmapFromGpu(self.dev, c.gpu);
        self.client.freeMemory(self.dev, c.mem);
        self.freeVa(c.va, c.va_span);
    }

    /// Return a chunk for reuse. If the pool is full, keep the larger chunks
    /// (most reusable) and free the smaller, so the pool tracks the working set.
    fn release(self: *Device, c: Chunk) void {
        for (&self.pool) |*slot| {
            if (slot.* == null) {
                slot.* = c;
                return;
            }
        }
        var min_i: usize = 0;
        for (self.pool, 0..) |slot, i| {
            if (slot.?.mem.size < self.pool[min_i].?.mem.size) min_i = i;
        }
        if (c.mem.size > self.pool[min_i].?.mem.size) {
            self.freeChunk(self.pool[min_i].?);
            self.pool[min_i] = c;
        } else {
            self.freeChunk(c);
        }
    }

    fn createResource(ptr: *anyopaque, desc: hal.ResourceDesc) hal.Error!*hal.Resource {
        const self: *Device = @ptrCast(@alignCast(ptr));
        // A render-target color image is allocated block-linear (GOB-tiled), like the
        // ZETA, so it can be paired with a depth surface (a LINEAR color target with
        // a ZETA selected faults Xid 69 / 0x9c). All color images go block-linear so
        // any of them can serve a depth-tested render pass. The CPU de-swizzles on
        // readback/present. Buffers stay linear.
        // A sampled texture (combined-image-sampler source, not a render target) is
        // allocated block-linear with the TEXTURE block height, uploaded through a
        // linear staging buffer the context GOB-swizzles at draw time.
        // A sampled texture takes the block-linear texture path, including a depth32_float image
        // requested as sampled-only (a sampler2DShadow source). That builds a ZF32 sampled TIC over
        // plain block-linear memory (the HW DEPTH_COMPARE only engages on a depth format), instead of
        // the ZETA render-surface path. A depth32_float that is also a render target stays a ZETA.
        const is_sampled_tex = switch (desc) {
            .image => |i| i.usage.sampled and !i.usage.render_target,
            else => false,
        };
        const is_color_image = switch (desc) {
            .image => |i| i.format != .depth32_float and !is_sampled_tex,
            else => false,
        };
        const size: u64 = switch (desc) {
            .buffer => |b| b.size,
            // A sampled texture is PITCH-LINEAR (row-major, no GOB tiling): its GPU
            // footprint is the 32-byte-aligned row pitch times the height. DEPTH (ZETA)
            // and COLOR render targets are block-linear (ztSizeBytes: GOB-tiled).
            .image => |i| if (is_sampled_tex)
                // A cubemap is a 6-face-wide 2D atlas (6*width x height), with a mip chain of smaller
                // atlases when mipped. A 3D texture (depth > 1) stacks `depth` slices. A mipmapped 2D
                // texture needs the whole chain. Otherwise just the base level.
                @as(u64, if (i.cube)
                    nvidia.graphics.mippedTexSizeBytes(i.width * 6, i.height, i.mip_levels, @import("resource.zig").ticFormat(i.format))
                else if (i.depth > 1)
                    nvidia.graphics.tex3dSizeBytes(i.width, i.height, i.depth, @import("resource.zig").ticFormat(i.format))
                else if (i.mip_levels > 1)
                    nvidia.graphics.mippedTexSizeBytes(i.width, i.height, i.mip_levels, @import("resource.zig").ticFormat(i.format))
                else
                    nvidia.graphics.texSizeBytes(i.width, i.height, @import("resource.zig").ticFormat(i.format)))
            else
                @as(u64, nvidia.graphics.ztSizeBytesBpp(i.width, i.height, @import("resource.zig").colorTargetBpp(i.format))),
        };
        if (size == 0) return error.InvalidArgument;
        // A ZETA depth surface must live in VRAM (device-local): the depth/Z-cull/
        // Z-compression fixed-function hardware rejects a sysmem depth surface (an
        // Xid 69 Class Error at the draw). Allocate it directly in .vram (not the
        // system_wc pool) and never CPU-map it (the CPU never reads the depth back).
        // A depth32_float routes to the ZETA render-surface path unless it was requested sampled-only
        // (then it is a ZF32 sampled texture handled by is_sampled_tex above).
        const is_depth = switch (desc) {
            .image => |i| i.format == .depth32_float and !is_sampled_tex,
            else => false,
        };
        if (is_depth) {
            const i = desc.image;
            // A depth attachment for a supersampled (samples>1) pass must match the enlarged
            // color render dims, so the ZETA is allocated at ssScale times the logical size.
            const ss = @import("resource.zig").ssScale(i.samples);
            const dw = i.width * ss[0];
            const dh = i.height * ss[1];
            const dsize: u64 = if (i.samples > 1) @as(u64, nvidia.graphics.ztSizeBytes(dw, dh)) else size;
            const buf = try self.allocDepthGpu(dw, dh, dsize);
            errdefer self.client.freeMemory(self.dev, buf.mem);
            const r = self.gpa.create(Resource) catch return error.OutOfMemory;
            r.* = .{ .kind = .image, .mem = buf.mem, .gpu_va = buf.va, .va_span = buf.va_span, .size = dsize, .mapping = null, .gpu = buf.gpu, .width = i.width, .height = i.height, .format = i.format, .is_depth_vram = true, .samples = i.samples };
            return @ptrCast(r);
        }
        if (is_sampled_tex) {
            // Block-linear sampled texture: VRAM (big pages) GOB-tiled memory the TIC's
            // V2_BL header points at, plus a tightly-packed linear CPU staging buffer the
            // ICD fills via copyBufferToImage. The context GOB-swizzles staging into the
            // tiled GPU memory when it builds the TIC. Block-linear is the layout NVIDIA
            // sampled images use. The texture units read it via the same Fermi GOB tiling
            // the color render targets use.
            const i = desc.image;
            const buf = try self.allocColorBlGpu(i.width, i.height, size);
            errdefer self.client.freeMemory(self.dev, buf.mem);
            // The staging buffer is tightly-packed texels of the stored format (4 bytes for
            // rgba8/sRGB, 8 for fp16, 16 for fp32). The ICD/GLES fills it. The context swizzles.
            const bpt: usize = @import("resource.zig").ticFormat(i.format).bytesPerTexel();
            // The staging holds the tightly-packed texel chain (level 0, then w/2 x h/2, ...) the
            // ICD/GLES fills. uploadTexture GOB-swizzles each level to its block-linear offset.
            // A cubemap stages 6 faces, each with its own tightly-packed mip chain (face-major, then
            // level-major within a face: face f level L at f*mipChainStagingBytes + stagingMipLevelOffset).
            // A 3D texture stages `depth` slices. A mipmapped 2D texture stages the chain. Otherwise the base.
            const staging_bytes: usize = if (i.cube)
                6 * nvidia.graphics.mipChainStagingBytes(i.width, i.height, i.mip_levels, bpt)
            else if (i.depth > 1)
                @as(usize, i.width) * i.height * i.depth * bpt
            else if (i.mip_levels > 1)
                nvidia.graphics.mipChainStagingBytes(i.width, i.height, i.mip_levels, bpt)
            else
                @as(usize, i.width) * i.height * bpt;
            const staging = self.gpa.alloc(u8, staging_bytes) catch return error.OutOfMemory;
            @memset(staging, 0);
            const r = self.gpa.create(Resource) catch return error.OutOfMemory;
            r.* = .{ .kind = .image, .mem = buf.mem, .gpu_va = buf.va, .va_span = buf.va_span, .size = size, .mapping = buf.cpu, .gpu = buf.gpu, .width = i.width, .height = i.height, .format = i.format, .block_linear = true, .sampled = true, .linear_copy = staging, .mip_levels = i.mip_levels, .depth = if (i.cube) 6 else i.depth, .is_cube = i.cube, .is_array = i.array };
            return @ptrCast(r);
        }
        if (is_color_image) {
            const i = desc.image;
            // Supersampled (samples>1) color target: allocate the block-linear backing at
            // ssScale times the logical size so the GPU renders an enlarged image. The
            // logical width/height stay WxH and a resolve box-downsamples the block later.
            const ss = @import("resource.zig").ssScale(i.samples);
            const rw = i.width * ss[0];
            const rh = i.height * ss[1];
            const ss_size: u64 = if (i.samples > 1) @as(u64, nvidia.graphics.ztSizeBytesBpp(rw, rh, @import("resource.zig").colorTargetBpp(i.format))) else size;
            const buf = try self.allocColorBlGpu(rw, rh, ss_size);
            errdefer self.client.freeMemory(self.dev, buf.mem);
            const r = self.gpa.create(Resource) catch return error.OutOfMemory;
            r.* = .{ .kind = .image, .mem = buf.mem, .gpu_va = buf.va, .va_span = buf.va_span, .size = ss_size, .mapping = buf.cpu, .gpu = buf.gpu, .width = i.width, .height = i.height, .format = i.format, .block_linear = true, .samples = i.samples };
            return @ptrCast(r);
        }
        // .system_wc: CPU-writable and GPU-coherent without cache snooping, so
        // one memory type serves vertex data (CPU writes) and render targets
        // (GPU writes, CPU reads back) alike. acquire() recycles pooled memory.
        const c = try self.acquire(size);
        errdefer self.release(c);
        const r = self.gpa.create(Resource) catch return error.OutOfMemory;
        r.* = switch (desc) {
            .buffer => .{ .kind = .buffer, .mem = c.mem, .gpu_va = c.va, .va_span = c.va_span, .size = size, .mapping = c.mapping, .gpu = c.gpu },
            .image => |i| .{ .kind = .image, .mem = c.mem, .gpu_va = c.va, .va_span = c.va_span, .size = size, .mapping = c.mapping, .gpu = c.gpu, .width = i.width, .height = i.height, .format = i.format },
        };
        return @ptrCast(r);
    }

    fn destroyResource(ptr: *anyopaque, resource: *hal.Resource) void {
        const self: *Device = @ptrCast(@alignCast(ptr));
        const r: *Resource = @ptrCast(@alignCast(resource));
        // VRAM surfaces (depth ZETA + block-linear color RTs) are freed directly
        // (they never went through the sysmem pool). Everything else is recycled
        // (with its live GPU + CPU mappings).
        if (r.linear_copy) |lc| self.gpa.free(lc);
        if (r.bl_scratch) |bs| self.gpa.free(bs);
        if (r.is_depth_vram or r.block_linear) {
            // VRAM surfaces are freed directly: release the CPU mapping (color/sampled
            // surfaces have one, depth does not) and the GPU VA binding, then the memory.
            if (r.mapping) |m| self.client.unmapMemory(m);
            if (r.gpu) |g| self.client.unmapFromGpu(self.dev, g);
            self.client.freeMemory(self.dev, r.mem);
            self.freeVa(r.gpu_va, r.va_span);
        } else {
            self.release(.{ .mem = r.mem, .va = r.gpu_va, .va_span = r.va_span, .mapping = r.mapping, .gpu = r.gpu.? });
        }
        self.gpa.destroy(r);
    }

    fn mapResource(ptr: *anyopaque, resource: *hal.Resource) hal.Error![]u8 {
        const self: *Device = @ptrCast(@alignCast(ptr));
        const r: *Resource = @ptrCast(@alignCast(resource));
        if (r.mapping == null) {
            r.mapping = self.client.mapMemory(self.dev, r.mem) catch |e| return nverr(e);
        }
        // A sampled texture is filled linearly by the ICD (copyBufferToImage) into its
        // staging buffer. The context swizzles staging into the block-linear GPU memory
        // at draw time. Hand back the linear staging and mark it dirty so the next draw
        // re-uploads it. (Distinct from a block-linear RT, which de-swizzles for readback.)
        if (r.sampled) {
            r.tex_dirty = true;
            return r.linear_copy.?;
        }
        // A block-linear color RT is GOB-tiled. De-swizzle it into a linear scratch
        // buffer so the caller sees a normal row-major w*h*4 image (the GPU wrote it
        // tiled, mirroring the block-linear present de-tile).
        //
        // The 3D engine renders into an A8R8G8B8 color target, whose little-endian
        // bytes are [B,G,R,A]. A resource declared `.rgba8_unorm` (what the Vulkan ICD
        // creates for a VK_FORMAT_R8G8B8A8_* attachment) must therefore be handed back
        // with the red/blue lanes swapped so the consumer sees true RGBA, otherwise
        // every non-grey color reads with R<->B transposed (proven by a frame oracle:
        // a `vec4(texcoord.x,0,0,1)` output landed in B, not R). A resource declared
        // `.bgra8_unorm` already matches the engine's byte order, so it is copied
        // straight (the in-tree green-clear test relies on that A8R8G8B8 layout).
        if (r.block_linear) {
            const bpp = @import("resource.zig").colorTargetBpp(r.format);
            const lin_len = @as(usize, r.width) * r.height * bpp;
            if (r.linear_copy == null or r.linear_copy.?.len < lin_len) {
                if (r.linear_copy) |lc| self.gpa.free(lc);
                r.linear_copy = self.gpa.alloc(u8, lin_len) catch return error.OutOfMemory;
            }
            // A float RT (rgba16f/rgba32f) copies its `bpp` bytes straight (R,G,B,A in order).
            // An rgba8_unorm consumer wants true RGBA (swap the engine's B<->R). A bgra8_unorm
            // consumer matches the engine byte order (straight copy).
            const mode: DeswizzleOut = if (bpp != 4) .float_straight else if (r.format == .rgba8_unorm) .rgba_swap else .bgra_straight;
            try self.deswizzleBlockLinear(r, r.linear_copy.?, @as(usize, r.width) * bpp, mode, bpp);
            return r.linear_copy.?;
        }
        // The physical mapping is page-rounded. Expose only the logical size.
        return r.mapping.?.bytes[0..@intCast(r.size)];
    }

    fn unmapResource(ptr: *anyopaque, resource: *hal.Resource) void {
        // The CPU mapping is kept until the resource is destroyed (cheap to hold,
        // and the consumer typically maps once). Nothing to do here.
        _ = ptr;
        _ = resource;
    }

    const DeswizzleOut = enum {
        rgba_swap, // A8R8G8B8 VRAM [B,G,R,A] -> [R,G,B,A] (an rgba8_unorm consumer)
        bgra_straight, // copy [B,G,R,A] verbatim (a bgra8_unorm consumer)
        present_bgrx, // [B,G,R,0xff] straight to a wl_shm XRGB8888 present buffer
        float_straight, // copy `bpp` bytes verbatim (RF16/RF32 store R,G,B,A in order, no swap)
    };

    /// De-swizzle a block-linear (Blackwell TuringColor2D GOB) color surface into the
    /// row-major `dst` (`dst_stride` bytes per row). One copy of the GOB math, shared by
    /// `mapResource` (RGBA/BGRA for generic consumers) and `readbackPresent` (XRGB
    /// straight to a present buffer) so the de-swizzle/speckle oracles cover every path.
    ///
    /// The GPU mapping is write-combined / uncached VRAM: random per-pixel reads (the GOB
    /// gather) are catastrophically slow (~0.5 s/frame at 500x500, dominating the live
    /// present). Copy the whole tiled surface to a cached scratch in one sequential
    /// pass first (WC reads fast sequentially), then de-swizzle from the cached copy (the
    /// gather reads then hit L1/L2). The per-x byte offset (gob column + intra-GOB x) and
    /// per-y (gob row + intra-GOB y) are each precomputed once, so the inner loop is two
    /// array loads + an add. Byte-identical to graphics.blColorPixelOffset.
    fn deswizzleBlockLinear(self: *Device, r: *Resource, dst: []u8, dst_stride: usize, mode: DeswizzleOut, bpp: u32) hal.Error!void {
        const src_raw = r.mapping.?.bytes;
        const tiled_len = blk: {
            // The tiled footprint the de-swizzle indexes into (GOB-aligned width x
            // block-aligned height). Bounded by the resource's allocated size.
            const rb = std.mem.alignForward(u32, r.width * bpp, 64);
            const th = std.mem.alignForward(u32, r.height, 16 * 8);
            break :blk @min(@as(usize, rb) * th, src_raw.len);
        };
        if (r.bl_scratch == null or r.bl_scratch.?.len < tiled_len) {
            if (r.bl_scratch) |s| self.gpa.free(s);
            r.bl_scratch = self.gpa.alloc(u8, tiled_len) catch return error.OutOfMemory;
        }
        const src = r.bl_scratch.?;
        @memcpy(src[0..tiled_len], src_raw[0..tiled_len]);

        const GOB_W: u32 = 64; // 64 bytes wide
        const GOB_H: u32 = 8; // 8 rows
        const BH: u32 = 16; // ZT_BLOCK_HEIGHT_GOBS
        const block_bytes: usize = GOB_W * GOB_H * BH; // bytes per (1-GOB-wide x BH-tall) block
        const row_bytes = std.mem.alignForward(u32, r.width * bpp, GOB_W);
        const gobs_per_row = row_bytes / GOB_W;

        var xbuf: [4096]usize = undefined;
        const off_x: []usize = if (r.width <= xbuf.len) xbuf[0..r.width] else (self.gpa.alloc(usize, r.width) catch return error.OutOfMemory);
        defer if (r.width > xbuf.len) self.gpa.free(off_x);
        {
            var x: u32 = 0;
            while (x < r.width) : (x += 1) {
                const xb = (x * bpp) % GOB_W;
                const gob_col = (x * bpp) / GOB_W;
                // TuringColor2D intra-GOB x term (Blackwell >=4-byte color): the two
                // 16-byte sector columns are 64 bytes apart, not 32. A wide (8/16-byte) texel's
                // bytes are contiguous from off_x[x] (a 16-byte line never straddles).
                const intra_x = (xb / 32) * 256 + ((xb % 32) / 16) * 64 + (xb % 16);
                off_x[x] = @as(usize, gob_col) * block_bytes + intra_x;
            }
        }
        var y: u32 = 0;
        while (y < r.height) : (y += 1) {
            const gob_row = y / GOB_H;
            const block_row = gob_row / BH;
            const gob_in_block = gob_row % BH;
            const yb = y % GOB_H;
            // y contribution: block row stride + gob-in-block stride + intra-GOB y bytes.
            // TuringColor2D intra-GOB y term: 4 rows per 128-byte sector group.
            const y_base: usize = @as(usize, block_row) * gobs_per_row * block_bytes +
                @as(usize, gob_in_block) * (GOB_W * GOB_H) +
                (yb / 4) * 128 + (yb % 4) * 16;
            var di = @as(usize, y) * dst_stride;
            var x: u32 = 0;
            switch (mode) {
                .rgba_swap => while (x < r.width) : (x += 1) {
                    const off = y_base + off_x[x];
                    dst[di + 0] = src[off + 2];
                    dst[di + 1] = src[off + 1];
                    dst[di + 2] = src[off + 0];
                    dst[di + 3] = src[off + 3];
                    di += 4;
                },
                .bgra_straight => while (x < r.width) : (x += 1) {
                    const off = y_base + off_x[x];
                    @memcpy(dst[di .. di + 4], src[off .. off + 4]);
                    di += 4;
                },
                .present_bgrx => while (x < r.width) : (x += 1) {
                    const off = y_base + off_x[x];
                    dst[di + 0] = src[off + 0]; // B
                    dst[di + 1] = src[off + 1]; // G
                    dst[di + 2] = src[off + 2]; // R
                    dst[di + 3] = 0xff; // X
                    di += 4;
                },
                .float_straight => while (x < r.width) : (x += 1) {
                    const off = y_base + off_x[x];
                    @memcpy(dst[di .. di + bpp], src[off .. off + bpp]); // RF16/RF32: R,G,B,A in order
                    di += bpp;
                },
            }
        }
    }

    /// Re-swizzle (the inverse of deswizzleBlockLinear): write the row-major linear `src` (RGBA/BGRA/
    /// float, `src_stride` bytes/row) into the block-linear GPU surface, applying the inverse channel
    /// map. Used by flushMappedImage so a CPU write into mapResource()'s de-swizzled scratch (e.g.
    /// glBlitFramebuffer's destination) reaches the tiled GPU memory. Writes straight to the WC GPU
    /// mapping (sequential-ish writes are cheap on write-combined memory, no cached-scratch needed).
    fn reswizzleBlockLinear(self: *Device, r: *Resource, src: []const u8, src_stride: usize, mode: DeswizzleOut, bpp: u32) hal.Error!void {
        const dst = r.mapping.?.bytes;
        const GOB_W: u32 = 64;
        const GOB_H: u32 = 8;
        const BH: u32 = 16;
        const block_bytes: usize = GOB_W * GOB_H * BH;
        const row_bytes = std.mem.alignForward(u32, r.width * bpp, GOB_W);
        const gobs_per_row = row_bytes / GOB_W;

        var xbuf: [4096]usize = undefined;
        const off_x: []usize = if (r.width <= xbuf.len) xbuf[0..r.width] else (self.gpa.alloc(usize, r.width) catch return error.OutOfMemory);
        defer if (r.width > xbuf.len) self.gpa.free(off_x);
        {
            var x: u32 = 0;
            while (x < r.width) : (x += 1) {
                const xb = (x * bpp) % GOB_W;
                const gob_col = (x * bpp) / GOB_W;
                const intra_x = (xb / 32) * 256 + ((xb % 32) / 16) * 64 + (xb % 16);
                off_x[x] = @as(usize, gob_col) * block_bytes + intra_x;
            }
        }
        var y: u32 = 0;
        while (y < r.height) : (y += 1) {
            const gob_row = y / GOB_H;
            const block_row = gob_row / BH;
            const gob_in_block = gob_row % BH;
            const yb = y % GOB_H;
            const y_base: usize = @as(usize, block_row) * gobs_per_row * block_bytes +
                @as(usize, gob_in_block) * (GOB_W * GOB_H) +
                (yb / 4) * 128 + (yb % 4) * 16;
            var si = @as(usize, y) * src_stride;
            var x: u32 = 0;
            switch (mode) {
                .rgba_swap => while (x < r.width) : (x += 1) {
                    const off = y_base + off_x[x];
                    dst[off + 0] = src[si + 2]; // tiled B = linear B
                    dst[off + 1] = src[si + 1]; // G
                    dst[off + 2] = src[si + 0]; // tiled R-slot = linear R
                    dst[off + 3] = src[si + 3]; // A
                    si += 4;
                },
                .bgra_straight, .present_bgrx => while (x < r.width) : (x += 1) {
                    const off = y_base + off_x[x];
                    @memcpy(dst[off .. off + 4], src[si .. si + 4]);
                    si += 4;
                },
                .float_straight => while (x < r.width) : (x += 1) {
                    const off = y_base + off_x[x];
                    @memcpy(dst[off .. off + bpp], src[si .. si + bpp]);
                    si += bpp;
                },
            }
        }
    }

    /// HAL flushMappedImage: re-swizzle a block-linear color RT's mapResource() scratch back into its
    /// tiled GPU surface (see reswizzleBlockLinear). A no-op for non-block-linear resources (their
    /// mapResource returns the real backing, so writes already persist). Requires a prior mapResource
    /// (which populated linear_copy). The caller (glBlitFramebuffer) always maps before writing.
    fn flushMappedImage(ptr: *anyopaque, resource: *hal.Resource) void {
        const self: *Device = @ptrCast(@alignCast(ptr));
        const r: *Resource = @ptrCast(@alignCast(resource));
        if (!r.block_linear) return;
        const lc = r.linear_copy orelse return;
        const bpp = @import("resource.zig").colorTargetBpp(r.format);
        const mode: DeswizzleOut = if (bpp != 4) .float_straight else if (r.format == .rgba8_unorm) .rgba_swap else .bgra_straight;
        self.reswizzleBlockLinear(r, lc, @as(usize, r.width) * bpp, mode, bpp) catch {};
    }

    /// Read a rendered block-linear color RT straight into a wl_shm XRGB8888 present
    /// buffer (`dst`, `dst_stride` bytes/row), de-swizzling in one pass. This fuses what
    /// the EGL present did in two passes (mapResource de-swizzle -> linear_copy, then a
    /// separate RGBA->XRGB blit) and drops the linear_copy intermediate, halving the
    /// per-frame CPU work over the rendered image. A linear (non-tiled) RT just needs the
    /// XRGB byte arrangement, so it copies per row at the GPU pitch.
    fn readbackPresent(ptr: *anyopaque, resource: *hal.Resource, dst: []u8, dst_stride: usize) hal.Error!void {
        const self: *Device = @ptrCast(@alignCast(ptr));
        const r: *Resource = @ptrCast(@alignCast(resource));
        if (r.mapping == null) {
            r.mapping = self.client.mapMemory(self.dev, r.mem) catch |e| return nverr(e);
        }
        if (r.block_linear) {
            // GPU detile via the copy engine: the CE reads the block-linear RT at GPU
            // bandwidth and writes a pitch-linear copy into cached sysmem, so the CPU
            // never touches write-combined VRAM (the slow ~80ms/frame readback). Then
            // one cached pass repacks into the caller's XRGB buffer. If the CE is
            // unavailable (create failed), fall back to the CPU de-swizzle.
            if (self.ceDetile(r)) |buf| {
                const sb = buf.bytes;
                const src_pitch: usize = @as(usize, r.width) * 4;
                var y: u32 = 0;
                while (y < r.height) : (y += 1) {
                    var di = @as(usize, y) * dst_stride;
                    var so = @as(usize, y) * src_pitch;
                    var x: u32 = 0;
                    // buf is [B,G,R,A] verbatim VRAM bytes. Emit [B,G,R,0xff] (XRGB).
                    while (x < r.width) : (x += 1) {
                        dst[di + 0] = sb[so + 0];
                        dst[di + 1] = sb[so + 1];
                        dst[di + 2] = sb[so + 2];
                        dst[di + 3] = 0xff;
                        di += 4;
                        so += 4;
                    }
                }
                return;
            } else |_| {
                try self.deswizzleBlockLinear(r, dst, dst_stride, .present_bgrx, 4);
                return;
            }
        }
        // Linear RT: read at the 256-byte-aligned GPU row pitch, repack to XRGB.
        const src = r.mapping.?.bytes;
        const pitch = nvidia.threed.pitchBytes(r.width);
        var y: u32 = 0;
        while (y < r.height) : (y += 1) {
            var di = @as(usize, y) * dst_stride;
            var so = @as(usize, y) * pitch;
            var x: u32 = 0;
            while (x < r.width) : (x += 1) {
                dst[di + 0] = src[so + 0];
                dst[di + 1] = src[so + 1];
                dst[di + 2] = src[so + 2];
                dst[di + 3] = 0xff;
                di += 4;
                so += 4;
            }
        }
    }

    /// Detile a block-linear color RT into the device's cached sysmem detile buffer
    /// using the copy engine, returning the buffer (pitch-linear, width*4, [B,G,R,A]).
    /// Lazily creates the CE + the buffer (grown to fit). Errors if the CE is
    /// unavailable so the caller can fall back to the CPU de-swizzle.
    fn ceDetile(self: *Device, r: *Resource) hal.Error!GpuBuf {
        if (self.ce == null) self.ce = try @import("ce.zig").CopyEngine.create(self);
        const ce = self.ce.?;
        const need: u64 = @as(u64, r.width) * r.height * 4;
        if (self.detile_buf == null or self.detile_buf.?.bytes.len < need) {
            if (self.detile_buf) |b| self.freeGpu(b);
            self.detile_buf = try self.allocGpu(.system, need); // WRITE_BACK = cached CPU reads
        }
        const buf = self.detile_buf.?;
        try ce.detile(r, buf, r.width * 4);
        return buf;
    }

    /// Build a sampled ZF32 depth texture from a rendered depth surface (a ZETA). The GL
    /// depth-texture-sample path (sampler2DShadow) calls this after a shadow-map render pass: the
    /// rendered depth lives block-linear in the ZETA (not directly sampleable), so CE-detile it into
    /// a pitch-linear f32 buffer on the GPU and stage those depths into a real ZF32 sampled texture,
    /// the format the HW DEPTH_COMPARE engages on (an rgba8 "depth-as-color" image makes the compare
    /// a HW no-op). This is the HW analog of the software finalizeDepthTexture, reusing the exact
    /// ZETA -> CE-detile -> ZF32-sampled machinery proven by ORACLE-SHADOW-2PASS. Caller owns/destroys.
    fn finalizeDepthTexture(ptr: *anyopaque, depth_rt: *hal.Resource, w: u32, h: u32) hal.Error!*hal.Resource {
        const self: *Device = @ptrCast(@alignCast(ptr));
        const zr: *Resource = @ptrCast(@alignCast(depth_rt));
        // CE-detile the block-linear ZETA (ZF32, 4 B/px, generic PTE kind, byte-identical tiling to
        // a color RT) into the cached pitch-linear (width*4) f32 buffer on the GPU.
        const buf = try self.ceDetile(zr);
        // A depth32_float sampled image routes to the ZF32 sampled-texture path (createResource's
        // is_sampled_tex): a pitch-linear staging + a block-linear GPU image with the ZF32 R001 TIC.
        const tex = try createResource(ptr, .{ .image = .{ .width = w, .height = h, .format = .depth32_float, .usage = .{ .sampled = true } } });
        const staging = mapResource(ptr, tex) catch |e| {
            destroyResource(ptr, tex);
            return e;
        };
        @memcpy(staging[0 .. @as(usize, w) * h * 4], buf.bytes[0 .. @as(usize, w) * h * 4]);
        return tex;
    }

    fn createContext(ptr: *anyopaque) hal.Error!hal.Context {
        const self: *Device = @ptrCast(@alignCast(ptr));
        return @import("context.zig").Context.create(self);
    }
    fn createShaderModule(ptr: *anyopaque, desc: hal.ShaderModuleDesc) hal.Error!*hal.ShaderModule {
        const self: *Device = @ptrCast(@alignCast(ptr));
        const m = try ShaderModule.create(self.gpa, desc);
        return @ptrCast(m);
    }
    fn destroyShaderModule(ptr: *anyopaque, module: *hal.ShaderModule) void {
        const self: *Device = @ptrCast(@alignCast(ptr));
        const m: *ShaderModule = @ptrCast(@alignCast(module));
        m.destroy(self.gpa);
    }
    /// One-shot HAL compute dispatch. Stubbed on the nvidia driver: the NVIDIA
    /// real compute path is not wired into this HAL primitive (only the field is
    /// provided so the Device.VTable literal is complete). Not a behavior change to
    /// the nvidia SPIR-V/SASS compute path that exists elsewhere.
    fn dispatchCompute(ptr: *anyopaque, d: hal.ComputeDispatch) hal.Error!void {
        _ = ptr;
        _ = d;
        return error.NotImplemented;
    }
    fn createPipeline(ptr: *anyopaque, desc: hal.PipelineDesc) hal.Error!*hal.Pipeline {
        const self: *Device = @ptrCast(@alignCast(ptr));
        const p = try Pipeline.create(self, desc);
        return @ptrCast(p);
    }
    fn destroyPipeline(ptr: *anyopaque, pipeline: *hal.Pipeline) void {
        const self: *Device = @ptrCast(@alignCast(ptr));
        const p: *Pipeline = @ptrCast(@alignCast(pipeline));
        p.destroy(self);
    }
    fn createSurface(ptr: *anyopaque, platform_surface: *anyopaque) hal.Error!*hal.Surface {
        const self: *Device = @ptrCast(@alignCast(ptr));
        const s = self.gpa.create(@import("surface.zig").Surface) catch return error.OutOfMemory;
        s.* = .{ .platform = @ptrCast(@alignCast(platform_surface)) };
        return @ptrCast(s);
    }
    fn destroySurface(ptr: *anyopaque, surface: *hal.Surface) void {
        const self: *Device = @ptrCast(@alignCast(ptr));
        self.gpa.destroy(@as(*@import("surface.zig").Surface, @ptrCast(@alignCast(surface))));
    }

    fn deinit(ptr: *anyopaque) void {
        const self: *Device = @ptrCast(@alignCast(ptr));
        const gpa = self.gpa;
        if (self.tf_ctx) |c| c.deinit();
        if (self.ce) |ce| ce.deinit();
        if (self.detile_buf) |b| self.freeGpu(b);
        if (self.occlusion_buf) |b| self.freeGpu(b);
        self.flushRetired();
        self.retired_heaps.deinit(gpa);
        for (self.pool) |slot| if (slot) |c| self.freeChunk(c);
        self.free_va.deinit(gpa);
        self.client.rmFree(self.dev.client, self.dev.device, self.vaspace);
        self.client.freeDevice(self.dev);
        self.client.deinit();
        gpa.destroy(self);
    }

    const vtable = hal.Device.VTable{
        .caps = &caps,
        .createContext = &createContext,
        .createResource = &createResource,
        .destroyResource = &destroyResource,
        .mapResource = &mapResource,
        .unmapResource = &unmapResource,
        .readbackPresent = &readbackPresent,
        .finalizeDepthTexture = &finalizeDepthTexture,
        .flushMappedImage = &flushMappedImage,
        .occlusionSampleCount = &occlusionSampleCount,
        .captureTransformFeedback = &captureTransformFeedback,
        .createShaderModule = &createShaderModule,
        .destroyShaderModule = &destroyShaderModule,
        .dispatchCompute = &dispatchCompute,
        .createPipeline = &createPipeline,
        .destroyPipeline = &destroyPipeline,
        .createSurface = &createSurface,
        .destroySurface = &destroySurface,
        .deinit = &deinit,
    };
};

// Pull the SPIR-V compute + graphics paths' GPU-gated tests into the driver's
// test set.
test {
    _ = @import("spirv_compute.zig");
    _ = @import("spirv_graphics.zig");
    _ = @import("ce.zig");
}

test "nvidia device creates and maps a GPU buffer resource (skips without a GPU)" {
    const gpa = std.testing.allocator;
    const dev = Device.create(gpa) catch return error.SkipZigTest;
    defer dev.deinit();
    const r = try dev.createResource(.{ .buffer = .{ .size = 256, .usage = .{ .vertex = true } } });
    defer dev.destroyResource(r);
    const bytes = try dev.mapResource(r);
    try std.testing.expectEqual(@as(usize, 256), bytes.len);
    // CPU-writable (write-combining). A write-then-read round-trips.
    bytes[0] = 0xab;
    bytes[255] = 0xcd;
    try std.testing.expectEqual(@as(u8, 0xab), bytes[0]);
    try std.testing.expectEqual(@as(u8, 0xcd), bytes[255]);
}

test "nvidia device creates a sampled texture (block-linear + linear staging) (skips without a GPU)" {
    const gpa = std.testing.allocator;
    const dev = Device.create(gpa) catch return error.SkipZigTest;
    defer dev.deinit();
    // A combined-image-sampler source: sampled (no render_target). The nvidia driver
    // backs it with block-linear VRAM (the TIC describes block-linear) + a linear CPU
    // staging buffer the ICD fills via copyBufferToImage. mapResource hands back the
    // tightly-packed staging the caller writes row-major.
    const tex = try dev.createResource(.{ .image = .{ .width = 2, .height = 2, .format = .rgba8_unorm, .usage = .{ .sampled = true, .copy_dst = true } } });
    defer dev.destroyResource(tex);
    const r: *Resource = @ptrCast(@alignCast(tex));
    try std.testing.expect(r.sampled);
    try std.testing.expect(r.block_linear);
    // mapResource returns the linear staging (width*height*4) and marks it dirty so the
    // next draw re-uploads (GOB-swizzles) it.
    const px = try dev.mapResource(tex);
    try std.testing.expectEqual(@as(usize, 2 * 2 * 4), px.len);
    px[0] = 0x11;
    px[15] = 0x22;
    try std.testing.expectEqual(@as(u8, 0x11), px[0]);
    try std.testing.expectEqual(@as(u8, 0x22), px[15]);
    try std.testing.expect(r.tex_dirty);
}

test "nvidia createResource recycles freed sysmem buffers (skips without a GPU)" {
    const gpa = std.testing.allocator;
    const dev = Device.create(gpa) catch return error.SkipZigTest;
    defer dev.deinit();
    // The sysmem pool recycles buffers/staging allocations (color render targets
    // are now block-linear VRAM and freed directly, not pooled; they must be
    // GOB-tiled to pair with a ZETA, so they don't share the linear sysmem pool).
    const r1 = try dev.createResource(.{ .buffer = .{ .size = 0x10000, .usage = .{ .vertex = true } } });
    const va = @as(*Resource, @ptrCast(@alignCast(r1))).gpu_va;
    dev.destroyResource(r1); // returns the chunk to the pool, GPU+CPU maps still live
    // A same-or-smaller request must reuse that chunk: same GPU VA, no fresh
    // alloc/map.
    const r2 = try dev.createResource(.{ .buffer = .{ .size = 0x8000, .usage = .{ .vertex = true } } });
    defer dev.destroyResource(r2);
    try std.testing.expectEqual(va, @as(*Resource, @ptrCast(@alignCast(r2))).gpu_va);
}

test "nvidia readbackPresent (CE path) matches the CPU de-swizzle at 800x600 rgba8 (skips without a GPU)" {
    // Reproduces the EGL present call exactly: an rgba8_unorm block-linear backbuffer
    // at the glmark2 window size, read back through the copy-engine present path. The
    // CE output (XRGB) must equal the CPU de-swizzle present (present_bgrx) byte-for-byte.
    const gpa = std.testing.allocator;
    const dev = Device.create(gpa) catch return error.SkipZigTest;
    defer dev.deinit();
    const self: *Device = @ptrCast(@alignCast(dev.ptr));
    const W: u32 = 800;
    const H: u32 = 600;

    const rt = try dev.createResource(.{ .image = .{ .width = W, .height = H, .format = .rgba8_unorm, .usage = .{ .render_target = true } } });
    defer dev.destroyResource(rt);
    const r: *Resource = @ptrCast(@alignCast(rt));

    // Clear it to a known color so the RT holds valid block-linear content.
    {
        const ctx = try dev.createContext();
        defer ctx.deinit();
        const cb = try ctx.beginCommands();
        defer cb.deinit();
        try cb.setRenderTarget(rt);
        try cb.clear(.{ .r = 0.25, .g = 0.5, .b = 0.75, .a = 1 });
        try ctx.submit(cb);
    }

    const stride: usize = @as(usize, W) * 4;
    // CPU reference (present_bgrx): de-swizzle straight into a host buffer.
    const cpu_dst = try gpa.alloc(u8, stride * H);
    defer gpa.free(cpu_dst);
    if (r.mapping == null) r.mapping = self.client.mapMemory(self.dev, r.mem) catch return error.SkipZigTest;
    try self.deswizzleBlockLinear(r, cpu_dst, stride, .present_bgrx, 4);

    // Ensure the CE actually engages (so this exercises the GPU path, not the fallback).
    self.ce = @import("ce.zig").CopyEngine.create(self) catch return error.SkipZigTest;

    // CE present path (the exact vtable entry the EGL layer calls).
    const ce_dst = try gpa.alloc(u8, stride * H);
    defer gpa.free(ce_dst);
    @memset(ce_dst, 0);
    try Device.readbackPresent(dev.ptr, rt, ce_dst, stride);

    try std.testing.expectEqualSlices(u8, cpu_dst, ce_dst);
}

test "nvidia device VA allocator keeps next_va bounded across resource churn (skips without a GPU)" {
    const gpa = std.testing.allocator;
    const dev = Device.create(gpa) catch return error.SkipZigTest;
    defer dev.deinit();
    const self: *Device = @ptrCast(@alignCast(dev.ptr));
    // A long-lived device like a UI framework creates and destroys GPU surfaces for as
    // long as it runs. Block-linear render targets are VRAM surfaces that never pool, and
    // each reserves a fresh 2 MB-aligned GPU VA range. Without recycling, next_va climbs
    // 2 MB per RT and eventually runs off the end of the VA space. Recycling reuses each
    // freed range so next_va holds steady. Churn 400 RTs and check the bump pointer barely
    // moved.
    const start_va = self.next_va;
    var i: usize = 0;
    while (i < 400) : (i += 1) {
        const rt = try dev.createResource(.{ .image = .{ .width = 256, .height = 256, .format = .bgra8_unorm, .usage = .{ .render_target = true } } });
        dev.destroyResource(rt);
        // RM reclaims freed VRAM objects asynchronously. A real app paces itself by
        // submitting frames between surface swaps, giving the kernel time to drain. This
        // test does no GPU work, so a tight alloc/free burst outruns reclaim and trips
        // NV_ERR_INSUFFICIENT_RESOURCES. Yield now and then to match real pacing.
        if (i % 32 == 31) {
            var ts = std.os.linux.timespec{ .sec = 0, .nsec = 2_000_000 };
            _ = std.os.linux.nanosleep(&ts, &ts);
        }
    }
    // 400 RTs at ~2 MB each is ~800 MB of churn. Recycling keeps the bump pointer within a
    // few slots of where it started. Without it next_va would sit ~800 MB higher.
    try std.testing.expect(self.next_va - start_va < 0x1000000); // < 16 MB of growth
}
