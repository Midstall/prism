const std = @import("std");
const nvidia = @import("nvidia");
const hal = @import("../../hal.zig");
const Resource = @import("resource.zig").Resource;
const Pipeline = @import("pipeline.zig").Pipeline;
const NvDevice = @import("device.zig").Device;
const nverr = @import("device.zig").nverr;
const threed = nvidia.threed;
const gfx = nvidia.graphics;
const sdk = nvidia.sdk;
const platform = @import("../../platform.zig");

// Root-table-1 byte offset for UBO base addresses and texture handles. Must match
// vulcan encode.graphics_ubo_cb_base (the shader reads them from c[25] = HW root table 1).
// Moved off LOAD_CONSTANT_BUFFER (incoherent at high TPC occupancy on Blackwell,
// see [[prism-glmark2-perf-cliff]]) onto the HW root table.
const UBO_CB_BASE: u32 = 0x40;

/// CB0 byte offset of the cube half-texel (0.5/face_width, f32). Must match
/// vulcan-target.nvidia.encode.cube_halftexel_cb: the cube lowering LDCs it to clamp the within-face
/// u so a linear tap does not bleed across the 6-face atlas column boundaries.
const CUBE_HALFTEXEL_CB: u32 = 0x00; // root-table-1 offset; matches vulcan encode.cube_halftexel_cb

/// Map a HAL sampler address mode to the nvidia TSC wrap mode.
fn mapAddress(a: hal.AddressMode) gfx.TexAddress {
    return switch (a) {
        .repeat => .repeat,
        .clamp_to_edge => .clamp_to_edge,
        .mirrored_repeat => .mirror,
    };
}

/// Map a HAL BlendFactor to the NV9097 OGL_* blend coefficient.
fn mapBlendFactor(f: hal.BlendFactor) gfx.BlendCoeff {
    return switch (f) {
        .zero => .zero,
        .one => .one,
        .src_color => .src_color,
        .one_minus_src_color => .one_minus_src_color,
        .src_alpha => .src_alpha,
        .one_minus_src_alpha => .one_minus_src_alpha,
        .dst_alpha => .dst_alpha,
        .one_minus_dst_alpha => .one_minus_dst_alpha,
        .dst_color => .dst_color,
        .one_minus_dst_color => .one_minus_dst_color,
        .constant_color => .constant_color,
        .one_minus_constant_color => .one_minus_constant_color,
        .constant_alpha => .constant_alpha,
        .one_minus_constant_alpha => .one_minus_constant_alpha,
        .src_alpha_saturate => .src_alpha_saturate,
    };
}

/// Map a HAL BlendOp to the NV9097 OGL_* blend equation.
fn mapBlendOp(o: hal.BlendOp) gfx.BlendEquation {
    return switch (o) {
        .add => .add,
        .subtract => .subtract,
        .reverse_subtract => .reverse_subtract,
        .min => .min,
        .max => .max,
    };
}

/// Map the HAL BlendState the pipeline carries into the NV graphics BlendState setBlend takes.
fn halToNvBlend(b: hal.BlendState) gfx.BlendState {
    return .{
        .enable = b.enable,
        .src_color = mapBlendFactor(b.src_color),
        .dst_color = mapBlendFactor(b.dst_color),
        .src_alpha = mapBlendFactor(b.src_alpha),
        .dst_alpha = mapBlendFactor(b.dst_alpha),
        .color_op = mapBlendOp(b.color_op),
        .alpha_op = mapBlendOp(b.alpha_op),
        .constant = b.constant,
    };
}

/// A bound combined-image-sampler (the mapped form of a hal.TextureBinding): the image
/// resource + the nvidia sampler state. Named so the per-draw state emitter can take a slice.
const TexBinding = struct {
    binding: u32,
    image: *Resource,
    filter: gfx.TexFilter,
    mip_filter: gfx.TexMipFilter,
    address_u: gfx.TexAddress,
    address_v: gfx.TexAddress,
    max_anisotropy: f32,
    base_level: u32 = 0, // GL_TEXTURE_BASE_LEVEL: the finest mip the TIC view exposes
    max_level: u32 = 1000, // GL_TEXTURE_MAX_LEVEL: the coarsest (clamped to the chain length)
    swizzle: [4]u8 = .{ 0, 1, 2, 3 }, // GL_TEXTURE_SWIZZLE_R/G/B/A -> TIC *_SOURCE (0=r..3=a,4=zero,5=one)
    lod_bias: f32 = 0, // GL_TEXTURE_LOD_BIAS -> TSC MIP_LOD_BIAS
    min_lod: f32 = -1000, // GL_TEXTURE_MIN_LOD -> TSC MIN_LOD_CLAMP
    max_lod: f32 = 1000, // GL_TEXTURE_MAX_LOD -> TSC MAX_LOD_CLAMP
    compare_enable: bool = false, // GL_TEXTURE_COMPARE_MODE -> TSC DEPTH_COMPARE (sampler2DShadow)
    compare_op: hal.CompareOp = .less_or_equal, // GL_TEXTURE_COMPARE_FUNC -> TSC DEPTH_COMPARE_FUNC
};

fn mapMipFilter(m: hal.MipFilter) gfx.TexMipFilter {
    return switch (m) {
        .none => .none,
        .nearest => .nearest,
        .linear => .linear,
    };
}

/// A recorded command. The command buffer appends these. GPU work is built and
/// submitted in Context.submit (mirroring the software driver).
const Command = union(enum) {
    set_render_target: *Resource,
    set_color_target: struct { index: u8, target: *Resource },
    clear: hal.Color,
    bind_pipeline: *hal.Pipeline,
    bind_vertex_buffer: *Resource,
    bind_uniform_buffer: struct { binding: u32, buffer: *Resource, snapshot: []const u8 = &.{} },
    bind_texture: TexBinding,
    set_depth_target: struct { depth: *Resource, clear_value: ?f32 },
    set_stencil_target: struct { stencil: *Resource, clear_value: ?u8 },
    draw: struct { vertex_count: u32, first_vertex: u32, instance_count: u32 = 1, first_instance: u32 = 0 },
    set_scissor: ?hal.ScissorRect,
    set_viewport: ?hal.Viewport,
    resolve: struct { src: *Resource, dst: *Resource, width: u32, height: u32, samples: u8 },
};

pub const CommandBuffer = struct {
    gpa: std.mem.Allocator,
    cmds: std.ArrayListUnmanaged(Command) = .empty,
    // When non-null this buffer is the context's pooled command buffer: deinit resets it
    // (retaining its cmds capacity) instead of freeing, and clears the owner's busy flag, so
    // the per-draw beginCommands/submit/deinit cycle does no allocator work. The struct + list
    // are freed once, at Context.deinit. null = an ordinary heap-owned buffer (freed on deinit).
    owner: ?*Context = null,

    fn create(gpa: std.mem.Allocator) hal.Error!hal.CommandBuffer {
        const self = gpa.create(CommandBuffer) catch return error.OutOfMemory;
        self.* = .{ .gpa = gpa };
        return .{ .ptr = self, .vtable = &vtable };
    }

    fn setRenderTarget(ptr: *anyopaque, target: *hal.Resource) hal.Error!void {
        const self: *CommandBuffer = @ptrCast(@alignCast(ptr));
        self.cmds.append(self.gpa, .{ .set_render_target = @ptrCast(@alignCast(target)) }) catch return error.OutOfMemory;
    }
    fn setColorTarget(ptr: *anyopaque, index: u32, target: *hal.Resource) hal.Error!void {
        const self: *CommandBuffer = @ptrCast(@alignCast(ptr));
        // index 0 is the primary render target (same as setRenderTarget). 1+ are MRT extras.
        if (index == 0) return setRenderTarget(ptr, target);
        self.cmds.append(self.gpa, .{ .set_color_target = .{ .index = @intCast(index), .target = @ptrCast(@alignCast(target)) } }) catch return error.OutOfMemory;
    }
    fn clear(ptr: *anyopaque, color: hal.Color) hal.Error!void {
        const self: *CommandBuffer = @ptrCast(@alignCast(ptr));
        self.cmds.append(self.gpa, .{ .clear = color }) catch return error.OutOfMemory;
    }
    fn bindPipeline(ptr: *anyopaque, pipeline: *hal.Pipeline) hal.Error!void {
        const self: *CommandBuffer = @ptrCast(@alignCast(ptr));
        self.cmds.append(self.gpa, .{ .bind_pipeline = pipeline }) catch return error.OutOfMemory;
    }
    fn bindVertexBuffer(ptr: *anyopaque, buffer: *hal.Resource) hal.Error!void {
        const self: *CommandBuffer = @ptrCast(@alignCast(ptr));
        self.cmds.append(self.gpa, .{ .bind_vertex_buffer = @ptrCast(@alignCast(buffer)) }) catch return error.OutOfMemory;
    }
    fn bindUniformBuffer(ptr: *anyopaque, binding: u32, buffer: *hal.Resource) hal.Error!void {
        const self: *CommandBuffer = @ptrCast(@alignCast(ptr));
        const res: *Resource = @ptrCast(@alignCast(buffer));
        // Snapshot the UBO's bytes at record time. The GL layer reuses one UBO Resource
        // per program/stage and rewrites it in place before the next draw. Since our GPU
        // work is deferred to submit(), a live pointer would let a later glUniform*
        // retroactively recolor this draw. A private copy makes each draw's uniforms
        // immutable once recorded. submit() places it in the UBO ring. An unmapped
        // buffer (never CPU-written) snapshots nothing and submit() falls back to the live GPU VA.
        var snap: []const u8 = &.{};
        if (res.mapping) |m| {
            const n: usize = @intCast(res.size);
            snap = self.gpa.dupe(u8, m.bytes[0..n]) catch return error.OutOfMemory;
        }
        self.cmds.append(self.gpa, .{ .bind_uniform_buffer = .{ .binding = binding, .buffer = res, .snapshot = snap } }) catch {
            self.gpa.free(snap);
            return error.OutOfMemory;
        };
    }
    fn bindTexture(ptr: *anyopaque, b: hal.TextureBinding) hal.Error!void {
        const self: *CommandBuffer = @ptrCast(@alignCast(ptr));
        self.cmds.append(self.gpa, .{ .bind_texture = .{
            .binding = b.binding,
            .image = @ptrCast(@alignCast(b.image)),
            .filter = switch (b.filter) {
                .nearest => .nearest,
                .linear => .linear,
            },
            .mip_filter = mapMipFilter(b.mip_filter),
            .address_u = mapAddress(b.address_u),
            .address_v = mapAddress(b.address_v),
            .max_anisotropy = b.max_anisotropy,
            .base_level = b.base_level,
            .max_level = b.max_level,
            .swizzle = .{ @intFromEnum(b.swizzle[0]), @intFromEnum(b.swizzle[1]), @intFromEnum(b.swizzle[2]), @intFromEnum(b.swizzle[3]) },
            .lod_bias = b.lod_bias,
            .min_lod = b.min_lod,
            .max_lod = b.max_lod,
            .compare_enable = b.compare_enable,
            .compare_op = b.compare_op,
        } }) catch return error.OutOfMemory;
    }
    fn setDepthTarget(ptr: *anyopaque, depth: *hal.Resource, clear_value: ?f32) hal.Error!void {
        const self: *CommandBuffer = @ptrCast(@alignCast(ptr));
        self.cmds.append(self.gpa, .{ .set_depth_target = .{ .depth = @ptrCast(@alignCast(depth)), .clear_value = clear_value } }) catch return error.OutOfMemory;
    }
    fn setStencilTarget(ptr: *anyopaque, stencil: *hal.Resource, clear_value: ?u8) hal.Error!void {
        const self: *CommandBuffer = @ptrCast(@alignCast(ptr));
        self.cmds.append(self.gpa, .{ .set_stencil_target = .{ .stencil = @ptrCast(@alignCast(stencil)), .clear_value = clear_value } }) catch return error.OutOfMemory;
    }
    fn draw(ptr: *anyopaque, vertex_count: u32, first_vertex: u32) hal.Error!void {
        const self: *CommandBuffer = @ptrCast(@alignCast(ptr));
        self.cmds.append(self.gpa, .{ .draw = .{ .vertex_count = vertex_count, .first_vertex = first_vertex } }) catch return error.OutOfMemory;
    }
    fn drawInstanced(ptr: *anyopaque, vertex_count: u32, instance_count: u32, first_vertex: u32, first_instance: u32) hal.Error!void {
        const self: *CommandBuffer = @ptrCast(@alignCast(ptr));
        self.cmds.append(self.gpa, .{ .draw = .{ .vertex_count = vertex_count, .first_vertex = first_vertex, .instance_count = instance_count, .first_instance = first_instance } }) catch return error.OutOfMemory;
    }
    fn setScissor(ptr: *anyopaque, rect: ?hal.ScissorRect) hal.Error!void {
        const self: *CommandBuffer = @ptrCast(@alignCast(ptr));
        self.cmds.append(self.gpa, .{ .set_scissor = rect }) catch return error.OutOfMemory;
    }
    fn setViewport(ptr: *anyopaque, vp: ?hal.Viewport) hal.Error!void {
        const self: *CommandBuffer = @ptrCast(@alignCast(ptr));
        self.cmds.append(self.gpa, .{ .set_viewport = vp }) catch return error.OutOfMemory;
    }
    fn resolve(ptr: *anyopaque, src: *hal.Resource, dst: *hal.Resource, width: u32, height: u32, format: hal.Format, samples: u8) hal.Error!void {
        _ = format;
        const self: *CommandBuffer = @ptrCast(@alignCast(ptr));
        self.cmds.append(self.gpa, .{ .resolve = .{
            .src = @ptrCast(@alignCast(src)),
            .dst = @ptrCast(@alignCast(dst)),
            .width = width,
            .height = height,
            .samples = samples,
        } }) catch return error.OutOfMemory;
    }
    /// Free the per-draw UBO snapshots dup'd in bindUniformBuffer (called by reset + deinit).
    fn freeSnapshots(self: *CommandBuffer) void {
        for (self.cmds.items) |c| switch (c) {
            .bind_uniform_buffer => |u| if (u.snapshot.len > 0) self.gpa.free(u.snapshot),
            else => {},
        };
    }
    fn reset(ptr: *anyopaque) void {
        const self: *CommandBuffer = @ptrCast(@alignCast(ptr));
        self.freeSnapshots();
        self.cmds.clearRetainingCapacity();
    }
    fn deinit(ptr: *anyopaque) void {
        const self: *CommandBuffer = @ptrCast(@alignCast(ptr));
        // Pooled buffer: reset for reuse (keep the allocation) and release it back to the context.
        if (self.owner) |ctx| {
            self.freeSnapshots();
            self.cmds.clearRetainingCapacity();
            ctx.cb_busy = false;
            return;
        }
        const gpa = self.gpa;
        self.freeSnapshots();
        self.cmds.deinit(gpa);
        gpa.destroy(self);
    }

    const vtable = hal.CommandBuffer.VTable{
        .setRenderTarget = &setRenderTarget,
        .setColorTarget = &setColorTarget,
        .clear = &clear,
        .bindPipeline = &bindPipeline,
        .bindVertexBuffer = &bindVertexBuffer,
        .bindUniformBuffer = &bindUniformBuffer,
        .bindTexture = &bindTexture,
        .setDepthTarget = &setDepthTarget,
        .setStencilTarget = &setStencilTarget,
        .draw = &draw,
        .drawInstanced = &drawInstanced,
        .setScissor = &setScissor,
        .setViewport = &setViewport,
        .resolve = &resolve,
        .reset = &reset,
        .deinit = &deinit,
    };
};

/// A GPU command context: owns a GPFIFO channel bound to the 3D engine, the
/// USERMODE doorbell, and a pushbuffer + completion semaphore. Records go to a
/// CommandBuffer. submit() turns them into 3D-engine methods and runs them.
pub const Context = struct {
    dev: *NvDevice,
    channel: nvidia.Channel,
    object: sdk.NvHandle,
    usermode: sdk.NvHandle,
    userd: NvDevice.GpuBuf,
    gpfifo: NvDevice.GpuBuf,
    pbuf: NvDevice.GpuBuf,
    sem: NvDevice.GpuBuf,
    // CB0 (constant buffer 0, bound to all shader groups) + TLS (shader-local
    // memory). The clear path doesn't touch these. The draw path needs them.
    cb0: NvDevice.GpuBuf,
    tls: NvDevice.GpuBuf,
    // Texture image-control (TIC) + sampler-control (TSC) pools: one 32-byte
    // descriptor per bound combined-image-sampler, indexed by the shader's bindless
    // TEX handle. Built per-submit from the bound textures.
    tic_pool: NvDevice.GpuBuf,
    tsc_pool: NvDevice.GpuBuf,
    queue: nvidia.Queue,
    // CPU mapping of the USERMODE doorbell page. The doorbell lives in a limited
    // hardware aperture. Unmapped in deinit. Leaking it exhausts the aperture and
    // the next context's usermode alloc fails (NV status 0x36).
    doorbell_map: nvidia.Mapping,
    /// Lazily-allocated internal Z24S8 ZETA surface used as the stencil buffer for a
    /// stencil-only UI clip (the HAL's stencil Resource is a software u8 buffer, not a GPU
    /// ZETA, so the nvidia path owns its own). Sized to the largest render target seen.
    /// Reused across submits and freed in deinit. null until the first stencil draw.
    stencil_zeta: ?NvDevice.GpuBuf = null,
    stencil_zeta_w: u32 = 0,
    stencil_zeta_h: u32 = 0,
    /// Per-draw UBO snapshot ring. The GL layer reuses one UBO Resource per program/stage
    /// and overwrites it in place on the next glUniform*, but GPU draws are deferred
    /// (recorded now, executed at submit and read by the shader even later). Two
    /// consecutive draws sharing that one Resource would both sample the last-written value.
    /// submit() copies each draw's record-time snapshot into a fresh region of this ring
    /// (bump cursor, wraps at the end) so each draw's shader LDGs its own copy. Each region
    /// is written once per wrap cycle and never overwritten while an in-flight draw reads it.
    /// Lazily allocated. Freed in deinit.
    ubo_ring: ?NvDevice.GpuBuf = null,
    ubo_ring_off: u64 = 0,
    /// Lazy dummy 64x64 color RT for the transform-feedback capture draw: threed.begin needs a bound
    /// color surface even when the rasterizer is disabled (stream-out only). Never read/written.
    tf_dummy_rt: ?NvDevice.GpuBuf = null,
    /// One-time reset of the ZPASS occlusion counter: cleared on the first draw submit so the
    /// hardware counter (and the device occlusion buffer's memset-0) share a 0 baseline.
    /// Accumulates across submits. Each submit reports the running count. See gfx.reportZpass.
    zpass_cleared: bool = false,
    /// Pooled per-context command buffer (see CommandBuffer.owner). beginCommands hands this out
    /// reset instead of allocating a fresh CommandBuffer + growing its cmds list every draw, which
    /// was ~19 us of the per-draw CPU cost. `cb_busy` guards against a second overlapping
    /// beginCommands (none in the hot path, but a nested caller falls back to a fresh buffer).
    cb_pool: ?*CommandBuffer = null,
    cb_busy: bool = false,
    /// The static draw-state defaults (root tables, watermarks, fixed mode bits: ~580 of the ~1426
    /// per-submit dwords) are emitted once on the first draw submit. GPU method state persists
    /// across submits on the channel (same reasoning as scissor/zpass cross-submit state).
    /// Later submits emit only begin + the per-submit resets (vertex streams + CB0 zero) +
    /// the per-draw pipeline state. See gfx.initStaticState.
    draw_state_static_inited: bool = false,

    const FENCE: u32 = 0xC0DE;
    /// UBO snapshot ring size. Default-block UBOs are small, under 1 KB. 256 KB at 256 B min stride
    /// gives ~1024 in-flight draw slots before the cursor wraps back onto long-completed submits.
    const UBO_RING_SIZE: u64 = 256 * 1024;

    /// Get-or-create the internal Z24S8 stencil ZETA sized to at least `w`x`h` (reallocated
    /// when a larger target appears). Returns its GPU VA.
    fn ensureStencilZeta(self: *Context, w: u32, h: u32) hal.Error!u64 {
        if (self.stencil_zeta) |z| {
            if (w <= self.stencil_zeta_w and h <= self.stencil_zeta_h) return z.va;
            self.dev.freeGpu(z);
            self.stencil_zeta = null;
        }
        const z = try self.dev.allocStencilZeta(w, h);
        self.stencil_zeta = z;
        self.stencil_zeta_w = w;
        self.stencil_zeta_h = h;
        return z.va;
    }

    pub fn create(dev: *NvDevice) hal.Error!hal.Context {
        const self = dev.gpa.create(Context) catch return error.OutOfMemory;
        errdefer dev.gpa.destroy(self);
        self.dev = dev;
        self.stencil_zeta = null;
        self.stencil_zeta_w = 0;
        self.stencil_zeta_h = 0;
        self.ubo_ring = null;
        self.ubo_ring_off = 0;
        self.tf_dummy_rt = null;
        self.zpass_cleared = false;
        self.cb_pool = null;
        self.cb_busy = false;
        self.draw_state_static_inited = false;

        self.userd = try dev.allocGpu(.vram, 0x1000);
        errdefer dev.freeGpu(self.userd);
        self.gpfifo = try dev.allocGpu(.vram, 0x2000);
        errdefer dev.freeGpu(self.gpfifo);
        self.pbuf = try dev.allocGpu(.system, 0x4000);
        errdefer dev.freeGpu(self.pbuf);
        self.sem = try dev.allocGpu(.system, 0x1000);
        errdefer dev.freeGpu(self.sem);
        self.cb0 = try dev.allocGpu(.vram, 0x10000); // CB0, 64 KB
        errdefer dev.freeGpu(self.cb0);
        self.tls = try dev.allocGpu(.vram, 0x200000); // shader-local memory
        errdefer dev.freeGpu(self.tls);
        // TIC + TSC pools (one 4 KB page each holds 128 32-byte descriptors). nvk
        // allocates these in VRAM. The texture units read them through the bound pool
        // address. CPU-mapped so the draw path can write the descriptors.
        self.tic_pool = try dev.allocGpu(.vram, 0x1000);
        errdefer dev.freeGpu(self.tic_pool);
        self.tsc_pool = try dev.allocGpu(.vram, 0x1000);
        errdefer dev.freeGpu(self.tsc_pool);
        // Descriptor index 0 is the NULL descriptor: a TEX whose bindless handle is 0 reads it.
        // initDrawState zeroes CB0, so a shader that samples a sampler with NO bound texture
        // (e.g. glmark2 shadow's depth pass runs the shadow.frag that samples ShadowMap while
        // the map is being generated, with nothing bound) reads handle 0 -> TIC[0]. Point it at
        // a valid 1x1 image (the pool's own mapped VRAM) so that TEX returns defined bytes
        // instead of faulting the GPU (Xid 31 MMU fault) - matching the software driver, which
        // returns transparent black for an unbound sampler.
        {
            const ntic = gfx.fillTic(self.tic_pool.va, 1, 1, .rgba8_unorm, 0);
            const ntsc = gfx.fillTsc(.nearest, .clamp_to_edge, .clamp_to_edge, .none, 1);
            @memcpy(std.mem.bytesAsSlice(u32, self.tic_pool.bytes[0..32]), &ntic);
            @memcpy(std.mem.bytesAsSlice(u32, self.tsc_pool.bytes[0..32]), &ntsc);
        }

        self.channel = dev.client.allocChannel(dev.dev, sdk.BLACKWELL_CHANNEL_GPFIFO_B, dev.vaspace, self.gpfifo.va, 0x100, self.userd.mem) catch |e| return nverr(e);
        self.object = dev.client.allocObject(dev.dev, self.channel, threed.BLACKWELL_A) catch |e| return nverr(e);
        dev.client.bindChannel(dev.dev, self.channel, sdk.NV2080_ENGINE_TYPE_GRAPHICS) catch |e| return nverr(e);
        dev.client.scheduleChannel(dev.dev, self.channel, true) catch |e| return nverr(e);
        const token = dev.client.workSubmitToken(dev.dev, self.channel) catch |e| return nverr(e);
        self.usermode = dev.client.allocUsermode(dev.dev, sdk.BLACKWELL_USERMODE_A) catch |e| return nverr(e);
        const door = dev.client.mapMemory(dev.dev, .{ .handle = self.usermode, .size = 0x1000, .location = .vram }) catch |e| return nverr(e);
        self.doorbell_map = door;
        self.queue = .{ .channel = self.channel, .token = token, .userd = self.userd.bytes, .gpfifo = self.gpfifo.bytes, .doorbell = door.bytes };

        // One-time channel init: bind the 3D class, upload the SET_PRIV_REG MME
        // macro, and run the two Blackwell priv-reg writes nvk does in its draw-
        // state init. The second one (clearing the SM "Out Of Range Address"
        // exception report-mask bit) lets a draw with a bound ZETA depth surface
        // succeed instead of faulting Xid 69 / Class Error 0x9c. The firmware
        // handshake (SET_FALCON04 + a scratch spin) can only run inside the MME,
        // not as raw methods.
        try self.initMme();

        return .{ .ptr = self, .vtable = &vtable };
    }

    /// Upload the SET_PRIV_REG MME macro and perform nvk's two state-init priv-reg
    /// writes (FP-helper-loads enable + the OOR-address-exception disable). Runs
    /// once per context, fenced, so the firmware completes the writes before any
    /// draw is submitted.
    fn initMme(self: *Context) hal.Error!void {
        var s = threed.Stream{ .buf = @alignCast(std.mem.bytesAsSlice(u32, self.pbuf.bytes)) };
        s.m1(0x0000, threed.BLACKWELL_A); // SET_OBJECT: bind the 3D class
        gfx.uploadMmeMacros(&s);
        gfx.disableDrawExceptions(&s);
        threed.fence(&s, self.sem.va, FENCE);

        const semp: *volatile u32 = @ptrCast(@alignCast(self.sem.bytes.ptr));
        semp.* = 0;
        self.queue.submit(self.pbuf.va, s.dwords());
        var spins: u64 = 0;
        while (spins < 500_000_000) : (spins += 1) {
            if (semp.* == FENCE) return;
        }
        return error.DeviceLost; // the MME init fence never landed
    }

    fn beginCommands(ptr: *anyopaque) hal.Error!hal.CommandBuffer {
        const self: *Context = @ptrCast(@alignCast(ptr));
        // A second overlapping begin (no deinit yet): fall back to a fresh, non-pooled buffer.
        if (self.cb_busy) return CommandBuffer.create(self.dev.gpa);
        if (self.cb_pool == null) {
            const handle = try CommandBuffer.create(self.dev.gpa);
            const cb: *CommandBuffer = @ptrCast(@alignCast(handle.ptr));
            cb.owner = self;
            self.cb_pool = cb;
        }
        const cb = self.cb_pool.?;
        // Reset here too (deinit already reset it). Belt-and-suspenders if a caller skipped deinit.
        cb.freeSnapshots();
        cb.cmds.clearRetainingCapacity();
        self.cb_busy = true;
        return .{ .ptr = cb, .vtable = &CommandBuffer.vtable };
    }

    /// Emit the per-pipeline + per-current-binding draw state (depth/stencil test, UBOs,
    /// textures, vertex attributes, shaders, cull, blend) into the method stream. The
    /// per-submit setup (initDrawState, RT/ZETA binding, viewport, clears) is done once
    /// by the caller. Re-emitted whenever the pipeline or a UBO/texture binding changes
    /// between draws, so each draw gets its own state. A single-pipeline single-binding
    /// submit calls this once. `depth_present`/`stencil_zeta` say whether the depth /
    /// stencil ZETA was bound for this pass (their surfaces + clears are per-submit).
    fn emitDrawState(
        self: *Context,
        s: *threed.Stream,
        p: *Pipeline,
        ubo_va: []const u64,
        ubo_bound: []const bool,
        tex_bind: []const ?TexBinding,
        depth_present: bool,
        stencil_zeta: bool,
    ) hal.Error!void {
        // Depth test (per pipeline). The ZETA surface was bound once for the pass.
        if (depth_present) {
            if (p.depth.test_enable) gfx.setDepthTest(s, p.depthFunc(), p.depth.write_enable);
            // gl_FragDepth (depth-replace): force late-Z + unbounded ZCULL when the FS writes
            // its own depth, else the ROP does early-Z with the interpolated z (paint order).
            gfx.setDepthReplace(s, p.ps.writes_depth);
            // Depth bias (glPolygonOffset): emitted per-pipeline (a disable resets a prior
            // pipeline's bias). Only meaningful with the depth test, but harmless otherwise.
            gfx.setDepthBias(s, p.depth.bias_enable, p.depth.bias_constant, p.depth.bias_slope, p.depth.bias_clamp);
        }
        // Stencil test (per pipeline). Two shapes: stencil-only (no depth attachment) and
        // combined depth+stencil (a depth ZETA + a separate S8 stencil plane both bound).
        if (stencil_zeta and p.stencil.test_enable) {
            // Only force depth OFF for the stencil-only path (a bound ZETA with no depth test
            // would otherwise run against unset depth state). For combined depth+stencil, the
            // depth test was just enabled above and must stay. Disabling it here made green
            // overwrite red (no depth) in the combined oracle.
            if (!depth_present) gfx.setDepthDisabled(s);
            if (p.stencil_back) |b| {
                // Two-sided: front methods = `stencil`, back methods = `stencil_back`.
                gfx.setStencilTestTwoSided(
                    s,
                    p.stencilFunc(),
                    Pipeline.nvStencilOp(p.stencil.fail_op),
                    Pipeline.nvStencilOp(p.stencil.depth_fail_op),
                    Pipeline.nvStencilOp(p.stencil.pass_op),
                    p.stencil.reference,
                    p.stencil.compare_mask,
                    p.stencil.write_mask,
                    Pipeline.nvCompare(b.compare_op),
                    Pipeline.nvStencilOp(b.fail_op),
                    Pipeline.nvStencilOp(b.depth_fail_op),
                    Pipeline.nvStencilOp(b.pass_op),
                    b.reference,
                    b.compare_mask,
                    b.write_mask,
                );
            } else {
                gfx.setStencilTest(
                    s,
                    p.stencilFunc(),
                    Pipeline.nvStencilOp(p.stencil.fail_op),
                    Pipeline.nvStencilOp(p.stencil.depth_fail_op),
                    Pipeline.nvStencilOp(p.stencil.pass_op),
                    p.stencil.reference,
                    p.stencil.compare_mask,
                    p.stencil.write_mask,
                );
            }
        }
        // Bind each bound UBO's 64-bit GPU address into CB0 at UBO_CB_BASE + binding*8.
        for (ubo_va, ubo_bound, 0..) |va, bound, binding| {
            if (!bound) continue;
            const off: u32 = UBO_CB_BASE + @as(u32, @intCast(binding)) * 8;
            gfx.loadConstantBuffer(s, off, &.{ @truncate(va), @intCast(va >> 32) });
        }
        // Textures: build TIC/TSC per bound sampler, upload the pixels, write the bindless
        // handle into CB0, and bind the pools (index 0 = null descriptor, always in range).
        var max_tex_idx: u32 = 0;
        for (tex_bind, 0..) |maybe, binding| {
            const tb = maybe orelse continue;
            const idx: u32 = @intCast(binding);
            const img = tb.image;
            self.uploadTexture(img);
            const desc_idx: u32 = idx + 1;
            if (desc_idx > max_tex_idx) max_tex_idx = desc_idx;
            const max_mip: u32 = if (img.mip_levels > 1) img.mip_levels - 1 else 0;
            // A cubemap is bound as a 6-face-wide 2D atlas (faces side by side, width = 6*face_w). The
            // native cube TEX mode does not do direction->face selection on this from-scratch Blackwell
            // path (both CUBE and ARRAY_CUBE fail, see vulcan encode.TexDim), so the isel lowers a
            // cube sample to the major-axis math + a 2D sample at (u' = (face+u)/6, v). The 2D within-
            // texture addressing is the proven path (the 3D within-slice u,v is unreliable).
            var tic = if (img.is_cube)
                // The cube is a 6-face-wide 2D atlas with a mip chain of smaller atlases. The isel
                // samples it with an explicit LOD (TLD), so MAX_MIP_LEVEL must span the chain.
                gfx.fillTic(img.gpu_va, img.width * 6, img.height, @import("resource.zig").ticFormat(img.format), max_mip)
            else if (img.is_array)
                // A 2D array (sampler2DArray): TWO_D_ARRAY TIC, `depth` layers, same stacked
                // block-linear backing as 3D (the layer index selects one layer, no cross-filtering).
                gfx.fillTic2dArray(img.gpu_va, img.width, img.height, img.depth, @import("resource.zig").ticFormat(img.format))
            else if (img.depth > 1)
                gfx.fillTic3d(img.gpu_va, img.width, img.height, img.depth, @import("resource.zig").ticFormat(img.format))
            else
                gfx.fillTic(img.gpu_va, img.width, img.height, @import("resource.zig").ticFormat(img.format), max_mip);
            // GL_TEXTURE_BASE_LEVEL / GL_TEXTURE_MAX_LEVEL: clamp the sampled mip level to the view
            // range. Only meaningful with a real chain. Both are capped at the chain length so the
            // defaults (0 / 1000) span the whole chain unchanged.
            gfx.setTicMipView(&tic, @min(tb.base_level, max_mip), @min(tb.max_level, max_mip));
            // GL_TEXTURE_SWIZZLE_R/G/B/A: remap the sampled channels via the TIC *_SOURCE fields.
            // A ZF32 depth (sampler2DShadow) texture keeps fillTic's fixed R001 swizzle (depth in R,
            // what DEPTH_COMPARE fetches); GL does not swizzle depth textures, so skip the override.
            if (img.format != .depth32_float) gfx.setTicSwizzle(&tic, tb.swizzle);
            // The mip filter only engages when the image actually has a chain (else MIP_NONE keeps
            // the base level, matching the pre-mip behavior).
            var tsc = gfx.fillTsc(tb.filter, tb.address_u, tb.address_v, if (img.mip_levels > 1) tb.mip_filter else .none, tb.max_anisotropy);
            gfx.setTscLodBias(&tsc, tb.lod_bias); // GL_TEXTURE_LOD_BIAS
            gfx.setTscLodClamp(&tsc, tb.min_lod, tb.max_lod); // GL_TEXTURE_MIN_LOD / MAX_LOD
            // GL_TEXTURE_COMPARE_MODE (sampler2DShadow): the TSC holds the depth-compare enable + func.
            // The isel's texShadow sets the TEX z_cmpr bit so the HW compares dref vs stored depth (R).
            gfx.setTscDepthCompare(&tsc, tb.compare_enable, @intFromEnum(tb.compare_op));
            @memcpy(std.mem.bytesAsSlice(u32, self.tic_pool.bytes[desc_idx * 32 ..][0..32]), &tic);
            @memcpy(std.mem.bytesAsSlice(u32, self.tsc_pool.bytes[desc_idx * 32 ..][0..32]), &tsc);
            const handle: u32 = (desc_idx & 0xfffff) | (desc_idx << 20);
            gfx.loadConstantBuffer(s, UBO_CB_BASE + idx * 8, &.{ handle, 0 });
            // A cubemap is sampled as a 6-face-wide atlas. The isel clamps the within-face u to
            // [half_texel, 1-half_texel] so a linear tap near a face edge does not bleed into the
            // neighbouring atlas column. half_texel = 0.5/face_width is passed via the shared CB slot.
            if (img.is_cube) {
                const half_texel: f32 = 0.5 / @as(f32, @floatFromInt(img.width));
                gfx.loadConstantBuffer(s, CUBE_HALFTEXEL_CB, &.{@bitCast(half_texel)});
            }
        }
        gfx.bindTexturePools(s, self.tic_pool.va, @max(max_tex_idx, 1), self.tsc_pool.va, @max(max_tex_idx, 1));
        // Mark every attribute slot the layout doesn't use inactive.
        {
            var i: u32 = 0;
            while (i < 16) : (i += 1) {
                var used = false;
                for (p.vertex_layout.attributes) |a| {
                    if (a.location == i) used = true;
                }
                if (!used) gfx.setVertexAttributeInactive(s, i);
            }
        }
        gfx.bindShader(s, 1, p.vs_sph_va, p.regCountFor(p.vs), gfx.bindGroup(.vertex)); // VS slot 1
        gfx.bindShader(s, 5, p.ps_sph_va, p.regCountFor(p.ps), gfx.bindGroup(.pixel)); // PS slot 5
        gfx.setCull(s, switch (p.cull.mode) {
            .none => .none,
            .front => .front,
            .back => .back,
        }, switch (p.cull.front_face) {
            .counter_clockwise => .ccw,
            .clockwise => .cw,
        });
        gfx.setBlend(s, halToNvBlend(p.blend));
        // Color write mask (glColorMask): per-pipeline SET_CT_WRITE on target 0. Default all-true
        // writes every channel. A stencil-only mask pass sets all-false to touch no color.
        const wm = p.blend.write_mask;
        gfx.setColorWriteMask(s, 0, wm[0], wm[1], wm[2], wm[3]);
    }

    /// Copy a record-time UBO snapshot into a fresh 256-byte-aligned region of the UBO ring and
    /// return its GPU VA (what the shader LDGs). Returns null (fall back to the live buffer VA)
    /// for an empty snapshot, a snapshot larger than the ring, or if the ring can't be allocated.
    /// Each region is written exactly once until the bump cursor wraps, so an in-flight draw's
    /// UBO is never clobbered by a later draw (see the ubo_ring field doc).
    fn snapshotUbo(self: *Context, snap: []const u8) ?u64 {
        if (snap.len == 0) return null;
        if (self.ubo_ring == null) {
            self.ubo_ring = self.dev.allocGpu(.system_wc, UBO_RING_SIZE) catch return null;
        }
        const ring = self.ubo_ring.?;
        if (snap.len > ring.bytes.len) return null;
        var off = std.mem.alignForward(u64, self.ubo_ring_off, 256);
        if (off + snap.len > ring.bytes.len) off = 0;
        @memcpy(ring.bytes[@intCast(off)..][0..snap.len], snap);
        self.ubo_ring_off = off + snap.len;
        return ring.va + off;
    }

    /// Lazily allocate the 64x64 dummy color RT for the TF capture draw (see the field doc).
    fn ensureTfDummyRt(self: *Context) ?u64 {
        if (self.tf_dummy_rt == null) {
            self.tf_dummy_rt = self.dev.allocColorBlGpu(64, 64, gfx.ztSizeBytesBpp(64, 64, 4)) catch return null;
        }
        return self.tf_dummy_rt.?.va;
    }

    /// GPU transform-feedback capture: run the pipeline's VS over `cap.vertex_count` vertices with
    /// stream-out enabled and the rasterizer disabled, streaming the selected VS output varyings
    /// (cap.specs, interleaved f32) into cap.output. Mirrors the normal draw setup (begin + draw
    /// state + shaders + vertex stream) plus the stream-out state. Returns bytes written.
    /// Three load-bearing pieces (each was a debug cycle): the SPH STREAM_OUT_MASK (VS emits to
    /// a stream), the SET_STREAM_OUTPUT global enable (nothing streams without it), and the
    /// per-component attr layout (0x20 + location*4 + component). Verified {1,2,3,4} -> {2,4,6,8}.
    pub fn captureTf(self: *Context, cap: hal.TransformFeedbackCapture) hal.Error!usize {
        const p: *Pipeline = @ptrCast(@alignCast(cap.pipeline));
        const out: *Resource = @ptrCast(@alignCast(cap.output));
        // Stream-out layout: one byte per streamed component = the VS output attribute index. Generic
        // VS output `location` lives at byte 0x80 + location*0x10, so component c is at attr index
        // (0x80 + location*0x10 + c*4)/4 = 0x20 + location*4 + c. Specs are interleaved in order.
        var attr: [64]u8 = undefined;
        var total: usize = 0;
        for (cap.specs) |sp| {
            var c: u8 = 0;
            while (c < sp.components and total < attr.len) : (c += 1) {
                attr[total] = @intCast(0x20 + sp.location * 4 + sp.first_component + c);
                total += 1;
            }
        }
        if (total == 0 or cap.vertex_count == 0) return 0;
        const stride: u32 = @intCast(total * 4);

        // Snapshot the VS/FS UBOs into the ring so emitDrawState binds stable VAs (same as a draw).
        var ubo_va = [_]u64{0} ** 8;
        var ubo_bound = [_]bool{false} ** 8;
        for (cap.ubos, 0..) |ub, i| {
            if (i >= 8) break;
            if (ub) |r| {
                const rr: *Resource = @ptrCast(@alignCast(r));
                if (rr.mapping) |m| {
                    if (self.snapshotUbo(m.bytes[0..@intCast(rr.size)])) |va| {
                        ubo_va[i] = va;
                        ubo_bound[i] = true;
                    }
                }
            }
        }
        self.dev.flushRetired(); // free heaps retired since the last submit (shader-VA safety)

        var s = threed.Stream{ .buf = @alignCast(std.mem.bytesAsSlice(u32, self.pbuf.bytes)) };
        const rt_va = self.ensureTfDummyRt() orelse return error.OutOfMemory;
        gfx.beginBlockLinearFmt(&s, threed.BLACKWELL_A, rt_va, 64, 64, @import("resource.zig").colorTargetFormat(.rgba8_unorm), 4);
        gfx.initDrawState(&s, 64, 64, self.tls.va, self.cb0.va);
        var no_tex = [_]?TexBinding{null} ** 8;
        try self.emitDrawState(&s, p, &ubo_va, &ubo_bound, &no_tex, false, false);
        gfx.setViewport(&s, 0, 0, 64, 64, 0.0, 1.0);
        if (cap.vertex_buffer) |vbh| {
            const vb: *Resource = @ptrCast(@alignCast(vbh));
            gfx.setVertexStream(&s, 0, vb.gpu_va, @intCast(vb.size), p.vertex_layout.stride);
            for (p.vertex_layout.attributes) |a| gfx.setVertexAttribute(&s, a.location, 0, a.offset, a.format.componentCount());
        }
        // Stream-out: buffer 0 = the capture target. `total` f32 components per vertex.
        // Layout maps each component to its VS-output attribute. Rasterizer off (capture only).
        gfx.setStreamOutputEnable(&s, true); // global enable. without this nothing streams
        gfx.setStreamOutBuffer(&s, 0, out.gpu_va + cap.output_offset, cap.vertex_count * stride);
        gfx.setStreamOutControl(&s, 0, 0, @intCast(total), stride);
        gfx.setStreamOutLayout(&s, 0, attr[0..total]);
        gfx.setRasterEnable(&s, false);
        gfx.setBaseInstance(&s, cap.instance);
        gfx.drawInstanced(&s, .points, cap.first_vertex, cap.vertex_count, 1);
        // Restore channel state for later normal draws on this shared channel.
        gfx.setRasterEnable(&s, true);
        gfx.disableStreamOutBuffer(&s, 0);
        gfx.setStreamOutputEnable(&s, false);

        threed.fence(&s, self.sem.va, FENCE);
        const semp: *volatile u32 = @ptrCast(@alignCast(self.sem.bytes.ptr));
        semp.* = 0;
        self.queue.submit(self.pbuf.va, s.dwords());
        try waitFence(semp);
        return @as(usize, cap.vertex_count) * total * 4;
    }

    fn submit(ptr: *anyopaque, cb: hal.CommandBuffer) hal.Error!void {
        const self: *Context = @ptrCast(@alignCast(ptr));
        const buf: *CommandBuffer = @ptrCast(@alignCast(cb.ptr));

        // Free shader heaps retired since the last submit. The prior submit has fenced (GPU is
        // idle past any shader reference) and this submit rebinds the shader before any draw, so
        // retired VAs are unmapped only once nothing references them (the Xid 31 GCC prefetch fix).
        self.dev.flushRetired();

        // Replay the records into a 3D-engine method stream. Two paths share the
        // same begin/fence framing: a bare clear, and a shaded draw (which needs
        // the full draw-state init, vertex streams, shaders, and viewport).
        var rt: ?*Resource = null;
        // MRT extra color targets (index 1..7). index 0 is `rt`. n_color_targets = 1 + the
        // highest bound extra index, telling the ROP how many targets to write.
        var extra_ct = [_]?*Resource{null} ** hal.MAX_COLOR_TARGETS;
        var n_color_targets: u32 = 1;
        var color: ?hal.Color = null;
        var pipe: ?*Pipeline = null;
        var depth_res: ?*Resource = null;
        var depth_clear: ?f32 = null;
        var stencil_bound = false;
        var stencil_clear: ?u8 = null;
        var any_draw = false;
        // UBO addresses + bound textures are tracked per-draw in the replay loop below (so a
        // submit with several uniform sets / textures binds each draw's own), not gathered here.
        // Track the running scissor so a glClear under an enabled scissor is clipped to the
        // same rect (GL semantics). clear_scissor snapshots the scissor active at the clear.
        var scissor_state: ?hal.ScissorRect = null;
        var clear_scissor: ?hal.ScissorRect = null;
        for (buf.cmds.items) |cmd| switch (cmd) {
            .set_render_target => |t| rt = t,
            .set_color_target => |ct| if (ct.index < extra_ct.len) {
                extra_ct[ct.index] = ct.target;
                if (ct.index + 1 > n_color_targets) n_color_targets = ct.index + 1;
            },
            .set_scissor => |r| scissor_state = r,
            .set_viewport => {}, // tracked per-draw in the replay loop
            .clear => |c| {
                color = c;
                clear_scissor = scissor_state;
            },
            .bind_pipeline => |p| pipe = @ptrCast(@alignCast(p)),
            .bind_vertex_buffer => {},
            .bind_uniform_buffer => {}, // tracked per-draw in the replay loop
            .bind_texture => {}, // tracked per-draw in the replay loop
            .set_stencil_target => |st| {
                // The HAL stencil Resource is a software u8 buffer. On nvidia the stencil
                // lives in a ZETA, so we only take the signal (stencil active) + the clear
                // value here and bind our own internal Z24S8 ZETA below.
                stencil_bound = true;
                // Keep the first non-null clear: a batch re-binds the target once per draw, and only
                // the frame's first draw carries a clear (later draws pass null to preserve). Taking
                // the last value would erase the clear and the buffer would never be cleared.
                if (st.clear_value) |cv| stencil_clear = cv;
            },
            .set_depth_target => |dt| {
                depth_res = dt.depth;
                if (dt.clear_value) |cv| depth_clear = cv; // first non-null clear wins (see above)
            },
            .draw => any_draw = true,
            .resolve => {}, // handled post-fence (CPU box-downsample) after the GPU work lands
        };
        // A resolve-only command buffer (an MSAA swapBuffers): no render target is bound.
        // The MS surface was already rendered + fenced by the prior draw submits. Run the
        // CPU box-downsample only. There is no GPU work to submit.
        if (rt == null) {
            for (buf.cmds.items) |cmd| switch (cmd) {
                .resolve => |rc| try self.resolveSSAA(rc.src, rc.dst, rc.width, rc.height, rc.samples),
                else => {},
            };
            return;
        }
        const target = rt orelse return error.InvalidArgument;

        // Supersampled MSAA: a samples>1 color target is rendered at ssScale times its
        // logical size (SSAA is transparent to the shaders). rw/rh are the render dimensions
        // (the enlarged block-linear backing). A resolve box-downsamples them to WxH after.
        // ssx/ssy scale HAL scissor rects (logical pixels) into this render space.
        const ssv = @import("resource.zig").ssScale(target.samples);
        const ssx = ssv[0];
        const ssy = ssv[1];
        const rw = target.width * ssx;
        const rh = target.height * ssy;

        var s = threed.Stream{ .buf = @alignCast(std.mem.bytesAsSlice(u32, self.pbuf.bytes)) };
        // A block-linear color RT (every nvidia render-target image) uses the
        // block-linear begin so it can be paired with a ZETA depth surface. A
        // non-block-linear target (only legacy/linear cases) uses the linear begin.
        if (target.block_linear)
            gfx.beginBlockLinearFmt(&s, threed.BLACKWELL_A, target.gpu_va, rw, rh, @import("resource.zig").colorTargetFormat(target.format), @import("resource.zig").colorTargetBpp(target.format))
        else
            threed.begin(&s, threed.BLACKWELL_A, target.gpu_va, rw, rh);
        // MRT: bind the extra color targets (slots 1..N-1) and tell the ROP how many targets
        // are live. Each extra surface is block-linear like RT0 (same tiling for readback).
        if (n_color_targets > 1) {
            var i: u32 = 1;
            while (i < n_color_targets) : (i += 1) {
                if (extra_ct[i]) |ct| gfx.bindColorTargetBlockLinear(&s, i, ct.gpu_va, rw, rh);
            }
            gfx.setColorTargetCount(&s, n_color_targets);
        }
        // Set the surface clip up front (nvk emits SET_SURFACE_CLIP before the render
        // targets). The ROP validates the bound ZETA against the surface clip.
        s.mm(0x0ff4, &.{ rw << 16, rh << 16 });
        if (color) |c| {
            // A glClear under an enabled scissor clears only the sub-rect (CLEAR_SURFACE
            // respects the scissor). Apply the clamped rect before the clear, then disable
            // so it does not linger into the draw setup (the replay loop re-applies per draw).
            // The HAL rect is in logical pixels. Scale it by ssx/ssy into the render space.
            if (clear_scissor) |r| {
                const x0: i64 = @max(@as(i64, 0), @as(i64, r.x) * ssx);
                const y0: i64 = @max(@as(i64, 0), @as(i64, r.y) * ssy);
                const x1: i64 = @min(@as(i64, rw), (@as(i64, r.x) + @as(i64, r.width)) * ssx);
                const y1: i64 = @min(@as(i64, rh), (@as(i64, r.y) + @as(i64, r.height)) * ssy);
                if (x1 <= x0 or y1 <= y0) {
                    gfx.setScissor(&s, 0, 0, 0, 0);
                } else {
                    gfx.setScissor(&s, @intCast(x0), @intCast(y0), @intCast(x1 - x0), @intCast(y1 - y0));
                }
            }
            threed.clear(&s, rw, rh, c.r, c.g, c.b, c.a);
            if (clear_scissor != null) gfx.disableScissor(&s);
        }

        // Draw-less depth clear: glClear(GL_DEPTH_BUFFER_BIT) on a bound FBO depth attachment
        // arrives as a color RT + a depth target with no draw (state.Context.clearDepthTarget). The
        // per-draw clearDepth below is gated on any_draw, so without this branch the clear is
        // silently dropped. This would leave a render-to-depth-texture shadow map at an uninitialized
        // (zero) far plane, so every fragment fails GL_LESS and no depth is ever written (the
        // whole shadow map reads 0). Bind the ZETA and issue the CLEAR_SURFACE Z clear here, only
        // when there is no draw (a draw sets up + clears depth itself in the any_draw path).
        if (!any_draw and depth_res != null) {
            if (depth_clear) |cv| {
                // Cold path (draw-less depth clear): emit the full draw state. This also lays down
                // the static defaults, so mark them inited to spare a later draw submit the repeat.
                gfx.initDrawState(&s, rw, rh, self.tls.va, self.cb0.va);
                self.draw_state_static_inited = true;
                gfx.bindDepth(&s, depth_res.?.gpu_va, rw, rh);
                gfx.setViewport(&s, 0, 0, rw, rh, 0.0, 1.0);
                s.mm(0x0ff4, &.{ rw << 16, rh << 16 }); // SET_SCREEN_SCISSOR (setViewport no longer sets it)
                gfx.clearDepth(&s, cv);
            }
        }

        if (any_draw) {
            const last_pipe = pipe orelse return error.InvalidArgument;
            // Render dimensions (supersampled for MSAA; == logical size for samples==1).
            const w = rw;
            const h = rh;
            // Per-submit setup: draw state, MRT count, depth/stencil ZETA binding,
            // viewport, scissor reset. Pipeline-independent (depth/stencil test state and
            // shaders/cull/blend are per-pipeline, emitted by emitDrawState below).
            // Static defaults (root tables etc) are emitted once per context. Every submit
            // re-emits only the two reset-critical pieces (vertex streams + CB0 zero).
            // begin() already re-emits SET_CT_SELECT / RENDER_ENABLE_C each submit.
            if (!self.draw_state_static_inited) {
                gfx.initStaticState(&s, self.tls.va, self.cb0.va);
                self.draw_state_static_inited = true;
            }
            gfx.resetVertexStreams(&s);
            gfx.zeroConstantBuffer0(&s);
            // Occlusion (ZPASS) counting: clear the hardware counter once (shares a 0 baseline with the
            // device occlusion buffer), then enable it every submit so the ROP counts passing samples
            // into the cumulative counter. The count is reported after the draws (below).
            if (!self.zpass_cleared) {
                gfx.clearZpassCount(&s);
                self.zpass_cleared = true;
            }
            gfx.setZpassPixelCount(&s, true);
            // initDrawState resets SET_CT_SELECT to 1. Re-emit the MRT count + identity map.
            if (n_color_targets > 1) gfx.setColorTargetCount(&s, n_color_targets);
            // Depth + stencil attachment binding. Three shapes:
            //   - combined depth+stencil: a ZF32 depth ZETA + a separate S8 stencil plane, bound
            //     together (bindDepthStencilSeparate sets SET_ZT_FORMAT=ZF32|STENCIL_IS_SEPARATE +
            //     the SET_ST_* plane). Not a packed Z24S8 (page kind the open RM will not set).
            //   - depth-only: the ZF32 ZETA (bindDepth).
            //   - stencil-only: the S8 surface is the ZETA (bindStencilZeta = SET_ZT_FORMAT=S8 +
            //     the SET_ST_* plane). Known open bug: the GLES stencil-only path faults Xid 31
            //     (an intermittent, run-to-run MMU VIRT_READ @ 0x0_10a20000, GPCCLIENT_GCC) on the
            //     real GPU. Binding a depth ZETA alongside (the combined config) does not fix it.
            //     See [[prism-nvidia-fbo-stencil-fault]] (a race/scheduling-class bug, unsolved).
            // The stencil test state is gated on the (last) pipeline's stencil test. The per-draw
            // test state is emitted by emitDrawState.
            const stencil_active = stencil_bound and last_pipe.stencil.test_enable;
            if (stencil_active and depth_res != null) {
                const zeta_va = ensureStencilZeta(self, w, h) catch return error.DeviceLost;
                gfx.bindDepthStencilSeparate(&s, depth_res.?.gpu_va, zeta_va, w, h);
            } else if (depth_res) |dr| {
                gfx.bindDepth(&s, dr.gpu_va, w, h);
            } else if (stencil_active) {
                const zeta_va = ensureStencilZeta(self, w, h) catch return error.DeviceLost;
                gfx.bindStencilZeta(&s, zeta_va, w, h);
            }
            const stencil_zeta = stencil_active;
            gfx.setViewport(&s, 0, 0, w, h, 0.0, 1.0);
            // Reset the app scissor per submit: GPU scissor state persists on the channel, so
            // a submit with no set_scissor command must start unscissored (a prior frame's rect
            // would leak). A set_scissor command in the replay loop re-enables it.
            gfx.disableScissor(&s);
            // Per-draw replay: track the current pipeline + UBO/texture bindings, and emit
            // the pipeline state (via emitDrawState) whenever any of them changes before a draw,
            // so a submit with several pipelines / uniform sets / textures renders each draw
            // with its own state. A single-pipeline single-binding submit emits state once,
            // before the first draw. The depth/stencil clear happens once after the first
            // pipeline's state is set (so write masks apply) and the surface clip is valid.
            var cur_p: ?*Pipeline = null;
            var cur_vb: ?*Resource = null;
            var cur_ubo_va = [_]u64{0} ** 8;
            var cur_ubo_bound = [_]bool{false} ** 8;
            var cur_tex = [_]?TexBinding{null} ** 8;
            var state_dirty = true;
            var cleared = false;
            // Track the current viewport across the batch. null = full render target (what the
            // per-submit setup above programmed). A set_viewport command re-emits gfx.setViewport
            // only when it changes, so full-RT draws (the common case) cost nothing extra.
            var cur_vp: ?hal.Viewport = null;
            // The push buffer is a fixed-size ring of dwords. A large batch (many draws
            // accumulated to one render target, e.g. glmark2 [ideas], which fires hundreds
            // of tiny Lamp draws each re-emitting draw state) would run `s.n` past the end
            // and panic (index-out-of-bounds in ni/loadConstantBuffer). Before each command,
            // if the stream is near full, kick the accumulated work to the GPU, wait for it,
            // and reset the stream. Channel method state (bound RT, pipeline, CB0, vertex
            // streams, viewport, scissor, ZPASS-enable) persists across kicks on the same
            // channel, so the continuation needs no re-prologue. The tracked cur_* state
            // stays valid and unchanged draws still skip their re-emit. The margin covers a
            // single draw's worst-case state emission (bounded: emitDrawState writes UBO
            // addresses, not contents) plus the occlusion-report + fence tail.
            const PBUF_FLUSH_MARGIN: usize = 2048;
            for (buf.cmds.items) |cmd| {
                if (s.n + PBUF_FLUSH_MARGIN >= s.buf.len) {
                    threed.fence(&s, self.sem.va, FENCE);
                    const fsem: *volatile u32 = @ptrCast(@alignCast(self.sem.bytes.ptr));
                    fsem.* = 0;
                    self.queue.submit(self.pbuf.va, s.dwords());
                    try waitFence(fsem);
                    s.reset();
                }
                switch (cmd) {
                    .bind_pipeline => |pp| {
                        cur_p = @ptrCast(@alignCast(pp));
                        state_dirty = true;
                    },
                    .bind_uniform_buffer => |u| if (u.binding < cur_ubo_va.len) {
                        // Place this draw's record-time UBO snapshot in a fresh ring region so the
                        // shader reads this draw's uniforms, not whatever the shared Resource holds by
                        // the time the deferred draw executes. The bump cursor wraps at the ring end
                        // onto long-completed submits. A snapshot that exceeds the ring or an
                        // unmapped buffer (empty snapshot) falls back to the live GPU VA.
                        cur_ubo_va[u.binding] = self.snapshotUbo(u.snapshot) orelse u.buffer.gpu_va;
                        cur_ubo_bound[u.binding] = true;
                        state_dirty = true;
                    },
                    .bind_texture => |t| if (t.binding < cur_tex.len) {
                        cur_tex[t.binding] = t;
                        state_dirty = true;
                    },
                    .bind_vertex_buffer => |b| cur_vb = b,
                    .set_scissor => |maybe| {
                        // GPU scissor is window pixels, top-left origin (matching Prism's viewport
                        // and the HAL rect). The HAL rect is logical pixels. Scale by ssx/ssy into
                        // the (supersampled) render space and clamp so the u16 XMIN/XMAX fit. A
                        // degenerate rect clips everything.
                        if (maybe) |r| {
                            const x0: i64 = @max(@as(i64, 0), @as(i64, r.x) * ssx);
                            const y0: i64 = @max(@as(i64, 0), @as(i64, r.y) * ssy);
                            const x1: i64 = @min(@as(i64, rw), (@as(i64, r.x) + @as(i64, r.width)) * ssx);
                            const y1: i64 = @min(@as(i64, rh), (@as(i64, r.y) + @as(i64, r.height)) * ssy);
                            if (x1 <= x0 or y1 <= y0) {
                                gfx.setScissor(&s, 0, 0, 0, 0);
                            } else {
                                gfx.setScissor(&s, @intCast(x0), @intCast(y0), @intCast(x1 - x0), @intCast(y1 - y0));
                            }
                        } else {
                            gfx.disableScissor(&s);
                        }
                    },
                    .set_viewport => |maybe| {
                        // Re-emit the viewport only on a change (see cur_vp). null = full RT (0,0,rw,rh).
                        // A rect is scaled by ssx/ssy into the supersampled MSAA render space (like scissor).
                        const changed = blk: {
                            if (maybe) |v| {
                                if (cur_vp) |c| break :blk !(c.x == v.x and c.y == v.y and c.width == v.width and c.height == v.height and c.depth_near == v.depth_near and c.depth_far == v.depth_far) else break :blk true;
                            } else break :blk cur_vp != null;
                        };
                        if (changed) {
                            cur_vp = maybe;
                            if (maybe) |v| {
                                gfx.setViewport(&s, v.x * @as(i32, @intCast(ssx)), v.y * @as(i32, @intCast(ssy)), v.width * ssx, v.height * ssy, v.depth_near, v.depth_far);
                            } else {
                                gfx.setViewport(&s, 0, 0, rw, rh, 0.0, 1.0);
                            }
                        }
                    },
                    .draw => |d| {
                        const p = cur_p orelse return error.InvalidArgument;
                        // Emit this draw's pipeline + UBO/texture state if anything changed since
                        // the last draw (or this is the first draw). Single-pipeline single-binding
                        // submits hit this exactly once, before the first draw.
                        if (state_dirty) {
                            try self.emitDrawState(&s, p, &cur_ubo_va, &cur_ubo_bound, &cur_tex, depth_res != null, stencil_zeta);
                            state_dirty = false;
                        }
                        // Clear depth/stencil once (after the first pipeline's depth/stencil state
                        // sets the write masks and the surface clip is valid). A null clear value
                        // preserves the buffer (the GLES clear-once-draw-many contract).
                        if (!cleared) {
                            if (depth_res != null) {
                                if (depth_clear) |cv| gfx.clearDepth(&s, cv);
                            }
                            if (stencil_zeta) {
                                if (stencil_clear) |cv| gfx.clearStencil(&s, cv);
                            }
                            cleared = true;
                        }
                        // A vertex-pulling draw has no vertex buffer/attributes: the VS pulls
                        // vertices from a UBO array by gl_VertexIndex, so no stream is set up.
                        if (cur_vb) |vb| {
                            gfx.setVertexStream(&s, 0, vb.gpu_va, @intCast(vb.size), p.vertex_layout.stride);
                            // Declare each attribute with its real component count (from the HAL
                            // format): a tightly-packed vec2/vec3 attribute fetched as a vec4 would
                            // over-read the last vertex past the stream size and fault the DA.
                            for (p.vertex_layout.attributes) |a| gfx.setVertexAttribute(&s, a.location, 0, a.offset, a.format.componentCount());
                        }
                        // Instanced: one native instanced draw. The DA iterates the vertex
                        // array instance_count times, delivering gl_InstanceIndex = base_instance
                        // + 0..N-1 to the VS's ALD a[0x2f8]. first_instance is the base. The
                        // topology selects the hardware primitive assembler (tris/lines/points).
                        const nv_topo: gfx.Topology = switch (p.topology) {
                            .triangle_list => .triangles,
                            .line_list => .lines,
                            .point_list => .points,
                        };
                        if (p.topology == .line_list) gfx.setLineWidth(&s, p.line_width);
                        // POINTS with a VS-written gl_PointSize: take the size from the shader.
                        if (p.topology == .point_list) gfx.setAttributePointSize(&s, p.writes_point_size);
                        gfx.setBaseInstance(&s, d.first_instance);
                        gfx.drawInstanced(&s, nv_topo, d.first_vertex, d.vertex_count, @max(d.instance_count, 1));
                    },
                    else => {},
                }
            }
            // Report the cumulative ZPASS pixel count into the device occlusion buffer (read by
            // Device.occlusionSampleCount for GL_ANY_SAMPLES_PASSED). After the draws, before the fence.
            // WAIT_FOR_IDLE orders the report's posted write ahead of the fence the CPU waits on, so
            // occlusionSampleCount reads the committed count, not the untouched buffer (see waitForIdle).
            if (self.dev.ensureOcclusionBuf()) |occ_va| {
                gfx.reportZpass(&s, occ_va);
                gfx.waitForIdle(&s);
            }
        }

        threed.fence(&s, self.sem.va, FENCE);

        const semp: *volatile u32 = @ptrCast(@alignCast(self.sem.bytes.ptr));
        semp.* = 0;
        self.queue.submit(self.pbuf.va, s.dwords());
        try waitFence(semp);

        // MSAA resolve (post-fence): the GPU rendered the supersampled color target. Now that
        // the fence has landed (writes flushed), box-downsample each ssScale block into the
        // single-sample resolved image. CPU-side. Both surfaces are block-linear A8R8G8B8 and
        // averaging is byte-order agnostic, so the engine's BGRA lanes average straight (the
        // R<->B swap for an rgba8 consumer happens later at mapResource of the resolved image).
        for (buf.cmds.items) |cmd| switch (cmd) {
            .resolve => |rc| try self.resolveSSAA(rc.src, rc.dst, rc.width, rc.height, rc.samples),
            else => {},
        };
        return;
    }

    /// Box-downsample the supersampled `src` (block-linear, rendered at ssScale*WxH) into the
    /// single-sample `dst` (block-linear WxH). Reuses graphics.blColorPixelOffset for both
    /// surfaces' GOB addressing (byte-identical to the de-swizzle/present paths).
    fn resolveSSAA(self: *Context, src: *Resource, dst: *Resource, w: u32, h: u32, samples: u8) hal.Error!void {
        const ss = @import("resource.zig").ssScale(samples);
        const sx = ss[0];
        const sy = ss[1];
        const n: u32 = sx * sy;
        if (n <= 1) return; // samples==1: nothing to resolve
        const rw = w * sx;
        const rh = h * sy;
        // Bytes per pixel of the color target (4 = rgba8, 8 = rgba16f, 16 = rgba32f). A float MSAA
        // target box-averages in float (unpack -> average -> repack). rgba8 keeps the u8 fast path.
        const bpp = @import("resource.zig").colorTargetBpp(src.format);
        if (src.mapping == null) src.mapping = self.dev.client.mapMemory(self.dev.dev, src.mem) catch |e| return nverr(e);
        if (dst.mapping == null) dst.mapping = self.dev.client.mapMemory(self.dev.dev, dst.mem) catch |e| return nverr(e);
        // The MS source mapping is write-combined/uncached VRAM: a random per-pixel gather
        // (the box-average reads ssx*ssy tiled samples per output pixel) reads it repeatedly
        // and is catastrophically slow (the same trap deswizzleBlockLinear documents, ~0.5s at
        // 500x500). Copy the whole tiled surface to a cached scratch in one sequential pass
        // first (WC reads fast sequentially), then gather from the cache (hits L1/L2). The
        // scattered writes to the resolved surface stay direct (WC absorbs scattered writes).
        const src_raw = src.mapping.?.bytes;
        const tiled_len = blk: {
            const rb = std.mem.alignForward(u32, rw * bpp, 64);
            const th = std.mem.alignForward(u32, rh, 16 * 8);
            break :blk @min(@as(usize, rb) * th, src_raw.len);
        };
        if (src.bl_scratch == null or src.bl_scratch.?.len < tiled_len) {
            if (src.bl_scratch) |old| self.dev.gpa.free(old);
            src.bl_scratch = self.dev.gpa.alloc(u8, tiled_len) catch return error.OutOfMemory;
        }
        const sb = src.bl_scratch.?;
        @memcpy(sb[0..tiled_len], src_raw[0..tiled_len]);
        const db = dst.mapping.?.bytes;
        const inv: f32 = 1.0 / @as(f32, @floatFromInt(n));
        const is_f16 = src.format == .rgba16_float;
        var oy: u32 = 0;
        while (oy < h) : (oy += 1) {
            var ox: u32 = 0;
            while (ox < w) : (ox += 1) {
                if (bpp == 4) {
                    // rgba8 fast path: integer box-average (byte-identical to the pre-float code).
                    var acc = [4]u32{ 0, 0, 0, 0 };
                    var j: u32 = 0;
                    while (j < sy) : (j += 1) {
                        var i: u32 = 0;
                        while (i < sx) : (i += 1) {
                            const so = nvidia.graphics.blColorPixelOffset(ox * sx + i, oy * sy + j, rw);
                            for (0..4) |k| acc[k] += sb[so + k];
                        }
                    }
                    const dof = nvidia.graphics.blColorPixelOffset(ox, oy, w);
                    for (0..4) |k| db[dof + k] = @intCast(acc[k] / n);
                } else {
                    // Float path: unpack each sample's fp16/fp32 channels, average, repack.
                    var acc = [4]f32{ 0, 0, 0, 0 };
                    var j: u32 = 0;
                    while (j < sy) : (j += 1) {
                        var i: u32 = 0;
                        while (i < sx) : (i += 1) {
                            const so = nvidia.graphics.blColorPixelOffsetBpp(ox * sx + i, oy * sy + j, rw, bpp);
                            for (0..4) |k| acc[k] += if (is_f16)
                                @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, sb[so + k * 2 ..][0..2], .little))))
                            else
                                @bitCast(std.mem.readInt(u32, sb[so + k * 4 ..][0..4], .little));
                        }
                    }
                    const dof = nvidia.graphics.blColorPixelOffsetBpp(ox, oy, w, bpp);
                    for (0..4) |k| {
                        const avg = acc[k] * inv;
                        if (is_f16)
                            std.mem.writeInt(u16, db[dof + k * 2 ..][0..2], @bitCast(@as(f16, @floatCast(avg))), .little)
                        else
                            std.mem.writeInt(u32, db[dof + k * 4 ..][0..4], @bitCast(avg), .little);
                    }
                }
            }
        }
        // The resolved image's de-swizzle caches now hold stale bytes. Drop them so a later
        // mapResource/present re-reads the freshly-written GPU memory.
        if (dst.bl_scratch) |bs| {
            self.dev.gpa.free(bs);
            dst.bl_scratch = null;
        }
        if (dst.linear_copy) |lc| {
            self.dev.gpa.free(lc);
            dst.linear_copy = null;
        }
    }

    /// Wait for the GPU to write FENCE to the completion semaphore. A tight busy-spin
    /// covers the common case where the GPU is free (fence lands in microseconds),
    /// then falls back to a short sleep: a pure busy-spin would pin a full CPU core
    /// for the whole wait, starving the compositor and anything else sharing the GPU
    /// (e.g. a co-resident compute/LLM workload). When the GPU is contended a single
    /// submit can take many milliseconds, so burning a core that long is wasteful.
    /// The old fixed iteration count could time out a slow-but-valid submit. The
    /// timeout here is wall-clock (5 s).
    fn waitFence(semp: *volatile u32) hal.Error!void {
        var spins: u64 = 0;
        const start = nowNs();
        while (true) {
            if (semp.* == FENCE) return;
            spins += 1;
            if (spins >= 8192) {
                var req = std.os.linux.timespec{ .sec = 0, .nsec = 50_000 }; // 50 us
                _ = std.os.linux.nanosleep(&req, null);
                if (nowNs() - start > 5 * std.time.ns_per_s) return error.DeviceLost;
            }
        }
    }

    fn nowNs() u64 {
        var ts: std.os.linux.timespec = undefined;
        _ = std.os.linux.clock_gettime(.MONOTONIC, &ts);
        return @as(u64, @intCast(ts.sec)) * 1_000_000_000 + @as(u64, @intCast(ts.nsec));
    }

    /// Copy a sampled texture's tightly-packed linear staging (filled by the ICD via
    /// copyBufferToImage) into its pitch-linear GPU memory at the 32-byte-aligned row
    /// pitch the TIC's V2_PITCH header describes. Row-major, no GOB swizzle: the bytes
    /// the sampler reads are exactly the pixels the CPU wrote (per-row padded). Only
    /// re-uploads dirty staging.
    fn uploadTexture(self: *Context, img: *Resource) void {
        _ = self;
        if (!img.sampled or !img.tex_dirty) return;
        const staging = img.linear_copy orelse return;
        const gpu = img.mapping.?.bytes;
        // Zero the whole tiled footprint first. The sampler may fetch padding beyond the
        // texels at edges, and uninitialized GPU memory would bleed in.
        @memset(gpu[0..@intCast(img.size)], 0);
        // GOB-swizzle each texel from the tightly-packed staging into the block-linear
        // GPU memory at its tiled byte offset so the texture units read it correctly. The
        // texel size follows the stored format (4B rgba8/sRGB, 8B fp16, 16B fp32). Wide float
        // texels stay contiguous in the GOB, so one memcpy per texel is correct.
        const fmt = @import("resource.zig").ticFormat(img.format);
        const bpt: usize = fmt.bytesPerTexel();
        // Swizzle every mip level: tightly-packed staging (stagingMipLevelOffset) -> block-linear
        // GPU at the level's own offset (blMipLevelOffset), using the level's dimensions so
        // blTexPixelOffset picks the level's clamped block height. A single-level texture is just
        // level 0 (levels=1), byte-identical to the pre-mip path.
        // A cubemap is a 6-face-wide 2D atlas: face f is GOB-swizzled into the atlas column
        // [f*width, (f+1)*width) of a (6*width) x height block-linear 2D image. The isel samples it at
        // (u'=(face+u)/6, v). A mipped cube is a chain of such atlases (level L = 6 faces of
        // (width>>L)x(height>>L)). The staging is face-major then level-major (face f level L at
        // f*face_chain + stagingMipLevelOffset). Requires power-of-two faces so 6*(width>>L) == (6*width)>>L.
        if (img.is_cube) {
            const levels: u32 = @max(1, img.mip_levels);
            const face_chain = nvidia.graphics.mipChainStagingBytes(img.width, img.height, levels, bpt);
            var level: u32 = 0;
            while (level < levels) : (level += 1) {
                const dims = nvidia.graphics.mipLevelDims(img.width, img.height, level);
                const lw = dims[0];
                const lh = dims[1];
                const atlas_w: u32 = lw * 6;
                const dst_base = nvidia.graphics.blMipLevelOffset(img.width * 6, img.height, level, fmt);
                const src_level = nvidia.graphics.stagingMipLevelOffset(img.width, img.height, level, bpt);
                var f: u32 = 0;
                while (f < 6) : (f += 1) {
                    const face_base = @as(usize, f) * face_chain + src_level;
                    var y: u32 = 0;
                    while (y < lh) : (y += 1) {
                        var xf: u32 = 0;
                        while (xf < lw) : (xf += 1) {
                            const so = face_base + (@as(usize, y) * lw + xf) * bpt;
                            const off = dst_base + nvidia.graphics.blTexPixelOffset(f * lw + xf, y, atlas_w, lh, fmt);
                            @memcpy(gpu[off .. off + bpt], staging[so .. so + bpt]);
                        }
                    }
                }
            }
            img.tex_dirty = false;
            return;
        }
        // A 3D texture (depth > 1) is `depth` stacked 2D block-linear slices: swizzle each slice
        // as a 2D image (tightly-packed staging slice -> its block-linear slice offset).
        if (img.depth > 1) {
            const slice_bl = nvidia.graphics.texSizeBytes(img.width, img.height, fmt); // one slice's BL footprint
            const slice_staging = @as(usize, img.width) * img.height * bpt;
            var z: u32 = 0;
            while (z < img.depth) : (z += 1) {
                const dst_slice = @as(usize, z) * slice_bl;
                const src_slice = @as(usize, z) * slice_staging;
                var y: u32 = 0;
                while (y < img.height) : (y += 1) {
                    var x: u32 = 0;
                    while (x < img.width) : (x += 1) {
                        const so = src_slice + (@as(usize, y) * img.width + x) * bpt;
                        const off = dst_slice + nvidia.graphics.blTexPixelOffset(x, y, img.width, img.height, fmt);
                        @memcpy(gpu[off .. off + bpt], staging[so .. so + bpt]);
                    }
                }
            }
            img.tex_dirty = false;
            return;
        }
        const levels: u32 = @max(1, img.mip_levels);
        var level: u32 = 0;
        while (level < levels) : (level += 1) {
            const dims = nvidia.graphics.mipLevelDims(img.width, img.height, level);
            const lw = dims[0];
            const lh = dims[1];
            const src_base = nvidia.graphics.stagingMipLevelOffset(img.width, img.height, level, bpt);
            const dst_base = nvidia.graphics.blMipLevelOffset(img.width, img.height, level, fmt);
            var y: u32 = 0;
            while (y < lh) : (y += 1) {
                var x: u32 = 0;
                while (x < lw) : (x += 1) {
                    const so = src_base + (@as(usize, y) * lw + x) * bpt;
                    const off = dst_base + nvidia.graphics.blTexPixelOffset(x, y, lw, lh, fmt);
                    @memcpy(gpu[off .. off + bpt], staging[so .. so + bpt]);
                }
            }
        }
        img.tex_dirty = false;
    }

    fn present(ptr: *anyopaque, surface: *hal.Surface, source: *hal.Resource) hal.Error!void {
        const self: *Context = @ptrCast(@alignCast(ptr));
        const surf: *@import("surface.zig").Surface = @ptrCast(@alignCast(surface));
        const src: *Resource = @ptrCast(@alignCast(source));

        // CPU-map the source framebuffer (the 3D engine wrote it). The fence in
        // submit already flushed those writes to memory.
        if (src.mapping == null) {
            src.mapping = self.dev.client.mapMemory(self.dev.dev, src.mem) catch |e| return nverr(e);
        }
        const src_bytes = src.mapping.?.bytes;

        const buf = try surf.platform.currentBuffer();
        // The 3D engine's render target is A8R8G8B8, whose little-endian bytes are
        // B,G,R,A (exactly a bgra8/Wayland XRGB8888 buffer), so it blits straight across.
        // An rgba8 buffer needs the red/blue lanes swapped. A block-linear RT is GOB-tiled,
        // so each source pixel is fetched via the de-swizzle offset. A linear RT uses the
        // 256-byte-aligned GPU row pitch.
        const pitch = nvidia.threed.pitchBytes(src.width);
        const swap = buf.format == .rgba8_unorm;
        const copy_w = @min(src.width, buf.width);
        const copy_h = @min(src.height, buf.height);
        var y: u32 = 0;
        while (y < copy_h) : (y += 1) {
            const dst_row = buf.bytes[@as(usize, y) * buf.stride ..][0 .. copy_w * 4];
            var x: usize = 0;
            while (x < copy_w) : (x += 1) {
                const so: usize = if (src.block_linear)
                    nvidia.graphics.blColorPixelOffset(@intCast(x), y, src.width)
                else
                    @as(usize, y) * pitch + x * 4;
                const o = x * 4;
                if (swap) {
                    dst_row[o + 0] = src_bytes[so + 2]; // R <- B
                    dst_row[o + 1] = src_bytes[so + 1]; // G
                    dst_row[o + 2] = src_bytes[so + 0]; // B <- R
                    dst_row[o + 3] = src_bytes[so + 3]; // A
                } else {
                    @memcpy(dst_row[o .. o + 4], src_bytes[so .. so + 4]);
                }
            }
        }
        try surf.platform.commit();
    }

    fn deinit(ptr: *anyopaque) void {
        const self: *Context = @ptrCast(@alignCast(ptr));
        const dev = self.dev;
        // Disable scheduling (preempt the channel off the runlist) before freeing it.
        // A still-scheduled channel's runlist slot is reclaimed only lazily by RM,
        // so disabling first lets the free reclaim it promptly (same as the usermode
        // unmap's limited aperture reasoning). Hardens against NV status 0x36 on the
        // 0xCA6F class under heavy context churn. Best-effort: deinit cannot fail.
        dev.client.scheduleChannel(dev.dev, self.channel, false) catch {};
        // Best-effort: free the engine object, channel, and usermode, then the
        // backing memory. Anything missed is reclaimed when the client closes.
        dev.client.rmFree(dev.dev.client, self.channel.handle, self.object);
        // Unmap the doorbell CPU mapping before freeing the usermode object, so the
        // limited usermode aperture is released (otherwise repeated contexts exhaust
        // it and the next usermode alloc fails with NV status 0x36).
        dev.client.unmapMemory(self.doorbell_map);
        dev.client.rmFree(dev.dev.client, dev.dev.subdevice, self.usermode);
        dev.client.rmFree(dev.dev.client, dev.dev.device, self.channel.handle);
        dev.freeGpu(self.tsc_pool);
        dev.freeGpu(self.tic_pool);
        dev.freeGpu(self.tls);
        dev.freeGpu(self.cb0);
        dev.freeGpu(self.sem);
        dev.freeGpu(self.pbuf);
        dev.freeGpu(self.gpfifo);
        dev.freeGpu(self.userd);
        if (self.stencil_zeta) |z| dev.freeGpu(z);
        if (self.tf_dummy_rt) |r| dev.freeGpu(r);
        if (self.ubo_ring) |r| dev.freeGpu(r);
        // Truly free the pooled command buffer (clear owner so its deinit frees rather than resets).
        if (self.cb_pool) |cb| {
            cb.owner = null;
            cb.freeSnapshots();
            cb.cmds.deinit(dev.gpa);
            dev.gpa.destroy(cb);
        }
        dev.gpa.destroy(self);
    }

    const vtable = hal.Context.VTable{
        .beginCommands = &beginCommands,
        .submit = &submit,
        .present = &present,
        .deinit = &deinit,
    };
};

test "nvidia context clears a render target green on the GPU (skips without a GPU)" {
    const gpa = std.testing.allocator;
    const dev = NvDevice.create(gpa) catch return error.SkipZigTest;
    defer dev.deinit();
    const W: u32 = 64;
    const H: u32 = 64;
    const rt = try dev.createResource(.{ .image = .{ .width = W, .height = H, .format = .bgra8_unorm, .usage = .{ .render_target = true } } });
    defer dev.destroyResource(rt);
    const ctx = try dev.createContext();
    defer ctx.deinit();
    const cb = try ctx.beginCommands();
    defer cb.deinit();
    try cb.setRenderTarget(rt);
    try cb.clear(.{ .r = 0, .g = 1, .b = 0, .a = 1 }); // green
    try ctx.submit(cb);
    // Read the framebuffer back through the HAL. The 3D engine's RT is A8R8G8B8
    // (0xAARRGGBB), so a green clear lands as 0xff00ff00. W=64 makes the 256-byte
    // RT pitch exactly W*4, so rows are tightly packed.
    const px = std.mem.bytesAsSlice(u32, try dev.mapResource(rt));
    try std.testing.expectEqual(@as(u32, 0xff00ff00), px[(H / 2) * W + W / 2]);
    try std.testing.expectEqual(@as(u32, 0xff00ff00), px[0]);
}

test "nvidia mapResource swaps R<->B for an rgba8 RT (the engine renders BGRA) (skips without a GPU)" {
    // The 3D engine renders into an A8R8G8B8 color target (bytes [B,G,R,A]). A resource
    // declared .rgba8_unorm (what the Vulkan ICD creates for VK_FORMAT_R8G8B8A8_*) must
    // be handed back from mapResource as true RGBA with R and B swapped, or every
    // non-grey color reads with the red/blue lanes transposed (the vkcube "yellow cube"
    // had its computed R in the B byte). A pure-red clear isolates the swap: R must land
    // in byte 0, B in byte 2. (.bgra8_unorm RTs are not swapped, see the green test.)
    const gpa = std.testing.allocator;
    const dev = NvDevice.create(gpa) catch return error.SkipZigTest;
    defer dev.deinit();
    const W: u32 = 64;
    const H: u32 = 64;
    const rt = try dev.createResource(.{ .image = .{ .width = W, .height = H, .format = .rgba8_unorm, .usage = .{ .render_target = true } } });
    defer dev.destroyResource(rt);
    const ctx = try dev.createContext();
    defer ctx.deinit();
    const cb = try ctx.beginCommands();
    defer cb.deinit();
    try cb.setRenderTarget(rt);
    try cb.clear(.{ .r = 1, .g = 0, .b = 0, .a = 1 }); // red
    try ctx.submit(cb);
    const px = try dev.mapResource(rt);
    const o = ((H / 2) * W + W / 2) * 4;
    try std.testing.expectEqual(@as(u8, 255), px[o + 0]); // R in byte 0
    try std.testing.expectEqual(@as(u8, 0), px[o + 1]); // G
    try std.testing.expectEqual(@as(u8, 0), px[o + 2]); // B in byte 2 (not 255)
    try std.testing.expectEqual(@as(u8, 255), px[o + 3]); // A
}

test "nvidia draws a gradient triangle through the HAL draw path (skips without a GPU)" {
    const sass = nvidia.sass;
    const gpa = std.testing.allocator;
    const dev = NvDevice.create(gpa) catch return error.SkipZigTest;
    defer dev.deinit();
    const W: u32 = 256;
    const H: u32 = 256;

    // Render target (cleared dark grey, so triangle pixels are distinguishable).
    const rt = try dev.createResource(.{ .image = .{ .width = W, .height = H, .format = .bgra8_unorm, .usage = .{ .render_target = true } } });
    defer dev.destroyResource(rt);

    // Assemble the gradient VS + PS SASS, byte-for-byte the proven probe shaders.
    // VS: pad past the async sysval delivery window, ALD position (a[0x80]) and
    // color (a[0x90]), then AST position + the color varying (o[0x80]).
    var vs_code: [256]u32 = undefined;
    var vsa = sass.Assembler{ .code = &vs_code };
    {
        var p: u32 = 0;
        while (p < 6) : (p += 1) vsa.movImm(12, p, .{});
    }
    vsa.ald(4, sass.ATTR_GENERIC0, 4, .{ .wr_barrier = 0 }); // R4..R7 = position
    vsa.ald(8, sass.ATTR_GENERIC0 + 0x10, 4, .{ .wr_barrier = 1 }); // R8..R11 = color
    vsa.ast(sass.ATTR_POSITION, 4, 4, .{ .wait_mask = 1 });
    vsa.ast(sass.ATTR_GENERIC0, 8, 4, .{ .wait_mask = 2 });
    vsa.exit(.{ .stall = 1 });

    // PS: pad, IPA the 4 interpolated color components into R4..R7, drain, move
    // down to R0..R3 (the ROP reads color0 from R0..R3 at EXIT).
    var ps_code: [256]u32 = undefined;
    var psa = sass.Assembler{ .code = &ps_code };
    {
        var p: u32 = 0;
        while (p < 6) : (p += 1) psa.movImm(15, p, .{});
    }
    psa.ipa(4, sass.ATTR_GENERIC0, .{ .wr_barrier = 0 });
    psa.ipa(5, sass.ATTR_GENERIC0 + 4, .{ .wr_barrier = 1 });
    psa.ipa(6, sass.ATTR_GENERIC0 + 8, .{ .wr_barrier = 2 });
    psa.ipa(7, sass.ATTR_GENERIC0 + 12, .{ .wr_barrier = 3 });
    psa.movReg(0, 4, .{ .wait_mask = 0xf });
    psa.movReg(1, 5, .{});
    psa.movReg(2, 6, .{});
    psa.movReg(3, 7, .{});
    psa.exit(.{ .stall = 15 });

    const vs = try dev.createShaderModule(.{ .stage = .vertex, .code = std.mem.sliceAsBytes(vs_code[0..vsa.dwords()]) });
    defer dev.destroyShaderModule(vs);
    const ps = try dev.createShaderModule(.{ .stage = .fragment, .code = std.mem.sliceAsBytes(ps_code[0..psa.dwords()]) });
    defer dev.destroyShaderModule(ps);

    // Vertex buffer: 3 verts x 32 bytes (pos vec4 + color vec4). RGB corners.
    const vbuf = try dev.createResource(.{ .buffer = .{ .size = 3 * 32, .usage = .{ .vertex = true } } });
    defer dev.destroyResource(vbuf);
    const verts = [_]f32{
        0.0, 0.5, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, // top -> red
        -0.5, -0.5, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, // bottom-left -> green
        0.5, -0.5, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, // bottom-right -> blue
    };
    @memcpy(std.mem.bytesAsSlice(f32, try dev.mapResource(vbuf))[0..verts.len], &verts);

    const attrs = [_]hal.VertexAttribute{
        .{ .location = 0, .format = .rgba8_unorm, .offset = 0 }, // position (clip space)
        .{ .location = 1, .format = .rgba8_unorm, .offset = 16 }, // color varying
    };
    const pipe = try dev.createPipeline(.{
        .vertex = vs,
        .fragment = ps,
        .vertex_layout = .{ .stride = 32, .attributes = &attrs },
        .color_format = .bgra8_unorm,
    });
    defer dev.destroyPipeline(pipe);

    const ctx = try dev.createContext();
    defer ctx.deinit();
    const cb = try ctx.beginCommands();
    defer cb.deinit();
    try cb.setRenderTarget(rt);
    try cb.clear(.{ .r = 0.125, .g = 0.125, .b = 0.125, .a = 1 });
    try cb.bindPipeline(pipe);
    try cb.bindVertexBuffer(vbuf);
    try cb.draw(3, 0);
    try ctx.submit(cb);

    // The triangle should have rendered: lots of pixels differ from the clear,
    // and the three corners carry distinct R/G/B (the gradient).
    const fb = std.mem.bytesAsSlice(u32, try dev.mapResource(rt));
    const clear_px: u32 = 0xff202020;
    var changed: u32 = 0;
    for (fb) |p| {
        if (p != clear_px) changed += 1;
    }
    // The triangle (clip [-0.5,0.5] in x/y) maps to a ~128x128 right triangle =
    // ~8192 covered pixels. Require a substantial painted region.
    try std.testing.expect(changed > 4000);
    // Three interior samples near the corners carry the distinct R/G/B gradient.
    // With the software-matching Y-origin (NDC -1 -> row 0), the clip-top red vertex
    // (clip y=+0.5) lands LOW (~row 176); the clip-bottom green/blue verts (clip y=-0.5)
    // land HIGH (~row 78).
    const top = fb[176 * W + 128]; // near the clip-top (red) vertex, now low
    const bl = fb[78 * W + 80]; // near the clip-bottom-left (green) vertex, now high
    const br = fb[78 * W + 176]; // near the clip-bottom-right (blue) vertex, now high
    try std.testing.expect(top != clear_px and bl != clear_px and br != clear_px);
    const R = struct {
        fn c(p: u32) u8 {
            return @truncate(p >> 16);
        }
        fn g(p: u32) u8 {
            return @truncate(p >> 8);
        }
        fn b(p: u32) u8 {
            return @truncate(p);
        }
    };
    try std.testing.expect(R.c(top) > R.g(top) and R.c(top) > R.b(top)); // red corner
    try std.testing.expect(R.g(bl) > R.c(bl) and R.g(bl) > R.b(bl)); // green corner
    try std.testing.expect(R.b(br) > R.c(br) and R.b(br) > R.g(br)); // blue corner
}

test "nvidia depth test occludes draw-order-independently on the GPU (skips without a GPU)" {
    const sass = nvidia.sass;
    const gpa = std.testing.allocator;
    const dev = NvDevice.create(gpa) catch return error.SkipZigTest;
    defer dev.deinit();
    const W: u32 = 256;
    const H: u32 = 256;

    // Passthrough VS (position + a color varying) + IPA PS, the same shape as the
    // gradient draw, so per-vertex z (gl_Position.z) drives the fixed-function depth
    // test. The color is constant across each triangle (all 3 verts same color).
    var vs_code: [256]u32 = undefined;
    var vsa = sass.Assembler{ .code = &vs_code };
    {
        var p: u32 = 0;
        while (p < 6) : (p += 1) vsa.movImm(12, p, .{});
    }
    vsa.ald(4, sass.ATTR_GENERIC0, 4, .{ .wr_barrier = 0 }); // R4..R7 = position
    vsa.ald(8, sass.ATTR_GENERIC0 + 0x10, 4, .{ .wr_barrier = 1 }); // R8..R11 = color
    vsa.ast(sass.ATTR_POSITION, 4, 4, .{ .wait_mask = 1 });
    vsa.ast(sass.ATTR_GENERIC0, 8, 4, .{ .wait_mask = 2 });
    vsa.exit(.{ .stall = 1 });
    var ps_code: [256]u32 = undefined;
    var psa = sass.Assembler{ .code = &ps_code };
    {
        var p: u32 = 0;
        while (p < 6) : (p += 1) psa.movImm(15, p, .{});
    }
    psa.ipa(4, sass.ATTR_GENERIC0, .{ .wr_barrier = 0 });
    psa.ipa(5, sass.ATTR_GENERIC0 + 4, .{ .wr_barrier = 1 });
    psa.ipa(6, sass.ATTR_GENERIC0 + 8, .{ .wr_barrier = 2 });
    psa.ipa(7, sass.ATTR_GENERIC0 + 12, .{ .wr_barrier = 3 });
    psa.movReg(0, 4, .{ .wait_mask = 0xf });
    psa.movReg(1, 5, .{});
    psa.movReg(2, 6, .{});
    psa.movReg(3, 7, .{});
    psa.exit(.{ .stall = 15 });

    const vs = try dev.createShaderModule(.{ .stage = .vertex, .code = std.mem.sliceAsBytes(vs_code[0..vsa.dwords()]) });
    defer dev.destroyShaderModule(vs);
    const ps = try dev.createShaderModule(.{ .stage = .fragment, .code = std.mem.sliceAsBytes(ps_code[0..psa.dwords()]) });
    defer dev.destroyShaderModule(ps);

    const attrs = [_]hal.VertexAttribute{
        .{ .location = 0, .format = .rgba8_unorm, .offset = 0 },
        .{ .location = 1, .format = .rgba8_unorm, .offset = 16 },
    };
    // Depth-tested pipeline: test enabled, write enabled, compare op LESS.
    const pipe = try dev.createPipeline(.{
        .vertex = vs,
        .fragment = ps,
        .vertex_layout = .{ .stride = 32, .attributes = &attrs },
        .color_format = .bgra8_unorm,
        .depth = .{ .test_enable = true, .write_enable = true, .compare_op = .less },
    });
    defer dev.destroyPipeline(pipe);

    // Two overlapping triangles: FAR z=0.7 RED, NEAR z=0.3 GREEN. Same XY footprint.
    const far_verts = [_]f32{
        0.0,  0.6,  0.7, 1.0, 1, 0, 0, 1,
        -0.6, -0.6, 0.7, 1.0, 1, 0, 0, 1,
        0.6,  -0.6, 0.7, 1.0, 1, 0, 0, 1,
    };
    const near_verts = [_]f32{
        0.0,  0.6,  0.3, 1.0, 0, 1, 0, 1,
        -0.6, -0.6, 0.3, 1.0, 0, 1, 0, 1,
        0.6,  -0.6, 0.3, 1.0, 0, 1, 0, 1,
    };
    const far_vb = try dev.createResource(.{ .buffer = .{ .size = far_verts.len * 4, .usage = .{ .vertex = true } } });
    defer dev.destroyResource(far_vb);
    const near_vb = try dev.createResource(.{ .buffer = .{ .size = near_verts.len * 4, .usage = .{ .vertex = true } } });
    defer dev.destroyResource(near_vb);
    @memcpy(std.mem.bytesAsSlice(f32, try dev.mapResource(far_vb))[0..far_verts.len], &far_verts);
    @memcpy(std.mem.bytesAsSlice(f32, try dev.mapResource(near_vb))[0..near_verts.len], &near_verts);

    const ctx = try dev.createContext();
    defer ctx.deinit();

    // Render the scene with a given draw order. Return the center pixel.
    const S = struct {
        fn render(d: hal.Device, c: hal.Context, p: *hal.Pipeline, depth: *hal.Resource, first: *hal.Resource, second: *hal.Resource, w: u32, h: u32) !u32 {
            const rt = try d.createResource(.{ .image = .{ .width = w, .height = h, .format = .bgra8_unorm, .usage = .{ .render_target = true } } });
            defer d.destroyResource(rt);
            const cb = try c.beginCommands();
            defer cb.deinit();
            try cb.setRenderTarget(rt);
            try cb.clear(.{ .r = 0.125, .g = 0.125, .b = 0.125, .a = 1 });
            try cb.setDepthTarget(depth, 1.0);
            try cb.bindPipeline(p);
            try cb.bindVertexBuffer(first);
            try cb.draw(3, 0);
            try cb.bindVertexBuffer(second);
            try cb.draw(3, 0);
            try c.submit(cb);
            const fb = std.mem.bytesAsSlice(u32, try d.mapResource(rt));
            return fb[(h / 2) * w + w / 2];
        }
    };

    const depth_a = try dev.createResource(.{ .image = .{ .width = W, .height = H, .format = .depth32_float, .usage = .{} } });
    defer dev.destroyResource(depth_a);
    const depth_b = try dev.createResource(.{ .image = .{ .width = W, .height = H, .format = .depth32_float, .usage = .{} } });
    defer dev.destroyResource(depth_b);

    // FAR-then-NEAR and NEAR-then-FAR. In both orders the near (green) must win the center.
    // The ZETA depth wall is solved (the block-linear color RT requirement): both orders
    // render green here on the real GPU. The render does both draws in one submit (one
    // setDepthTarget clear + two draws + one fence), the in-pass depth-occlusion case.
    // A separate test below covers the multi-submit case (one draw per submit, depth
    // preserved across submits via a null clear_value).
    const center_far_first = S.render(dev, ctx, pipe, depth_a, far_vb, near_vb, W, H) catch return error.SkipZigTest;
    const center_near_first = S.render(dev, ctx, pipe, depth_b, near_vb, far_vb, W, H) catch return error.SkipZigTest;

    const R = struct {
        fn r(p: u32) u8 {
            return @truncate(p >> 16);
        }
        fn g(p: u32) u8 {
            return @truncate(p >> 8);
        }
        fn b(p: u32) u8 {
            return @truncate(p);
        }
    };
    // Green-dominant center in both orders = the GPU depth test occluded the far red
    // regardless of paint order (paint order alone would show red for near-then-far).
    try std.testing.expect(R.g(center_far_first) > 200 and R.r(center_far_first) < 80 and R.b(center_far_first) < 80);
    try std.testing.expect(R.g(center_near_first) > 200 and R.r(center_near_first) < 80 and R.b(center_near_first) < 80);
}

test "nvidia stencil clip: a REPLACE mask clips a later EQUAL draw on the GPU (skips without a GPU)" {
    const glsl = @import("../../glsl.zig");
    const gpa = std.testing.allocator;
    const dev = NvDevice.create(gpa) catch return error.SkipZigTest;
    defer dev.deinit();
    const W: u32 = 256;
    const H: u32 = 256;

    const vs_src =
        \\attribute vec2 position;
        \\attribute vec3 color;
        \\varying vec3 vColor;
        \\void main() { gl_Position = vec4(position, 0.0, 1.0); vColor = color; }
    ;
    const fs_src =
        \\precision mediump float;
        \\varying vec3 vColor;
        \\void main() { gl_FragColor = vec4(vColor, 1.0); }
    ;
    const vs_bytes = try glsl.compileForStage(gpa, vs_src, .vertex);
    defer gpa.free(vs_bytes);
    const fs_bytes = try glsl.compileForStage(gpa, fs_src, .fragment);
    defer gpa.free(fs_bytes);
    const vs = try dev.createShaderModule(.{ .stage = .vertex, .code = vs_bytes });
    defer dev.destroyShaderModule(vs);
    const fs = try dev.createShaderModule(.{ .stage = .fragment, .code = fs_bytes });
    defer dev.destroyShaderModule(fs);

    const attrs = [_]hal.VertexAttribute{
        .{ .location = 0, .format = .r32g32_float, .offset = 0 },
        .{ .location = 1, .format = .r32g32b32_float, .offset = 8 },
    };
    // MASK pipeline: stencil ALWAYS passes, REPLACE the stencil with ref=1 where drawn.
    const mask_pipe = try dev.createPipeline(.{
        .vertex = vs,
        .fragment = fs,
        .vertex_layout = .{ .stride = 20, .attributes = &attrs },
        .color_format = .bgra8_unorm,
        .stencil = .{ .test_enable = true, .compare_op = .always, .pass_op = .replace, .reference = 1 },
    });
    defer dev.destroyPipeline(mask_pipe);
    // CONTENT pipeline: draw only where stencil == 1 (the mask), keep.
    const content_pipe = try dev.createPipeline(.{
        .vertex = vs,
        .fragment = fs,
        .vertex_layout = .{ .stride = 20, .attributes = &attrs },
        .color_format = .bgra8_unorm,
        .stencil = .{ .test_enable = true, .compare_op = .equal, .pass_op = .keep, .reference = 1 },
    });
    defer dev.destroyPipeline(content_pipe);

    // 6 verts/quad: pos vec2 + color vec3. Mask = left half (x in [-1,0]) black; content =
    // full screen red.
    const quad = struct {
        fn make(x0: f32, x1: f32, r: f32, g: f32, b: f32) [30]f32 {
            return .{
                x0, -1, r, g, b, x1, -1, r, g, b, x1, 1, r, g, b,
                x0, -1, r, g, b, x1, 1,  r, g, b, x0, 1, r, g, b,
            };
        }
    };
    const mask_v = quad.make(-1, 0, 0, 0, 0);
    const content_v = quad.make(-1, 1, 1, 0, 0);
    const mask_vb = try dev.createResource(.{ .buffer = .{ .size = mask_v.len * 4, .usage = .{ .vertex = true } } });
    defer dev.destroyResource(mask_vb);
    @memcpy(std.mem.bytesAsSlice(f32, try dev.mapResource(mask_vb))[0..mask_v.len], &mask_v);
    const content_vb = try dev.createResource(.{ .buffer = .{ .size = content_v.len * 4, .usage = .{ .vertex = true } } });
    defer dev.destroyResource(content_vb);
    @memcpy(std.mem.bytesAsSlice(f32, try dev.mapResource(content_vb))[0..content_v.len], &content_v);

    // The HAL stencil Resource (a u8 buffer): on nvidia it is just the SIGNAL that stencil
    // is active. The driver binds its own internal Z24S8 ZETA.
    const stencil = try dev.createResource(.{ .buffer = .{ .size = W * H } });
    defer dev.destroyResource(stencil);
    const rt = try dev.createResource(.{ .image = .{ .width = W, .height = H, .format = .bgra8_unorm, .usage = .{ .render_target = true } } });
    defer dev.destroyResource(rt);

    const ctx = try dev.createContext();
    defer ctx.deinit();
    const cb = try ctx.beginCommands();
    defer cb.deinit();
    try cb.setRenderTarget(rt);
    try cb.clear(.{ .r = 0, .g = 0, .b = 0, .a = 1 });
    try cb.setStencilTarget(stencil, 0); // clear stencil to 0
    try cb.bindPipeline(mask_pipe);
    try cb.bindVertexBuffer(mask_vb);
    try cb.draw(6, 0);
    try cb.bindPipeline(content_pipe);
    try cb.bindVertexBuffer(content_vb);
    try cb.draw(6, 0);
    ctx.submit(cb) catch return error.SkipZigTest;

    const fb = std.mem.bytesAsSlice(u32, try dev.mapResource(rt));
    // bgra8 readback (the engine renders BGRA): red = byte2, green = byte1, blue = byte0.
    const red = struct {
        fn ok(p: u32) bool {
            return @as(u8, @truncate(p >> 16)) > 200 and @as(u8, @truncate(p >> 8)) < 60 and @as(u8, @truncate(p)) < 60;
        }
        fn black(p: u32) bool {
            return @as(u8, @truncate(p >> 16)) < 60 and @as(u8, @truncate(p >> 8)) < 60 and @as(u8, @truncate(p)) < 60;
        }
    };
    // NDC x in [-1,0] -> window x [0,128]. Left pixel (64,128) is inside the mask -> RED.
    // Right pixel (192,128) is outside -> stencil 0, content clipped -> BLACK.
    try std.testing.expect(red.ok(fb[128 * W + 64]));
    try std.testing.expect(red.black(fb[128 * W + 192]));
}

test "nvidia stencil nested clip: two INCR masks intersect, content shows only the overlap on the GPU (skips without a GPU)" {
    const glsl = @import("../../glsl.zig");
    const gpa = std.testing.allocator;
    const dev = NvDevice.create(gpa) catch return error.SkipZigTest;
    defer dev.deinit();
    const W: u32 = 256;
    const H: u32 = 256;

    // Nested clipping the way a UI toolkit intersects clip regions: two overlapping masks each
    // INCR the stencil, so the overlap reads 2 and the single-covered sides read 1. Content
    // drawn with EQUAL ref=2 shows only the intersection. Exercises the INCR stencil op + a
    // non-1 reference on the GPU (the first clip test only used REPLACE + ref 1).
    const vs_src =
        \\attribute vec2 position;
        \\attribute vec3 color;
        \\varying vec3 vColor;
        \\void main() { gl_Position = vec4(position, 0.0, 1.0); vColor = color; }
    ;
    const fs_src =
        \\precision mediump float;
        \\varying vec3 vColor;
        \\void main() { gl_FragColor = vec4(vColor, 1.0); }
    ;
    const vs_bytes = try glsl.compileForStage(gpa, vs_src, .vertex);
    defer gpa.free(vs_bytes);
    const fs_bytes = try glsl.compileForStage(gpa, fs_src, .fragment);
    defer gpa.free(fs_bytes);
    const vs = try dev.createShaderModule(.{ .stage = .vertex, .code = vs_bytes });
    defer dev.destroyShaderModule(vs);
    const fs = try dev.createShaderModule(.{ .stage = .fragment, .code = fs_bytes });
    defer dev.destroyShaderModule(fs);

    const attrs = [_]hal.VertexAttribute{
        .{ .location = 0, .format = .r32g32_float, .offset = 0 },
        .{ .location = 1, .format = .r32g32b32_float, .offset = 8 },
    };
    // INCR mask pipeline: stencil ALWAYS passes, INCR (saturating) the stencil where drawn.
    const incr_pipe = try dev.createPipeline(.{
        .vertex = vs,
        .fragment = fs,
        .vertex_layout = .{ .stride = 20, .attributes = &attrs },
        .color_format = .bgra8_unorm,
        .stencil = .{ .test_enable = true, .compare_op = .always, .pass_op = .incr_clamp },
    });
    defer dev.destroyPipeline(incr_pipe);
    // CONTENT pipeline: draw only where stencil == 2 (both masks overlapped).
    const content_pipe = try dev.createPipeline(.{
        .vertex = vs,
        .fragment = fs,
        .vertex_layout = .{ .stride = 20, .attributes = &attrs },
        .color_format = .bgra8_unorm,
        .stencil = .{ .test_enable = true, .compare_op = .equal, .pass_op = .keep, .reference = 2 },
    });
    defer dev.destroyPipeline(content_pipe);

    const quad = struct {
        fn make(x0: f32, x1: f32, r: f32, g: f32, b: f32) [30]f32 {
            return .{
                x0, -1, r, g, b, x1, -1, r, g, b, x1, 1, r, g, b,
                x0, -1, r, g, b, x1, 1,  r, g, b, x0, 1, r, g, b,
            };
        }
    };
    // Mask A = x in [-1, 0.2], Mask B = x in [-0.2, 1]. Overlap = [-0.2, 0.2] (window ~102..154).
    const a_v = quad.make(-1.0, 0.2, 0, 0, 0);
    const b_v = quad.make(-0.2, 1.0, 0, 0, 0);
    const content_v = quad.make(-1.0, 1.0, 1, 0, 0); // full-screen red, clipped to stencil==2
    const a_vb = try dev.createResource(.{ .buffer = .{ .size = a_v.len * 4, .usage = .{ .vertex = true } } });
    defer dev.destroyResource(a_vb);
    const b_vb = try dev.createResource(.{ .buffer = .{ .size = b_v.len * 4, .usage = .{ .vertex = true } } });
    defer dev.destroyResource(b_vb);
    const content_vb = try dev.createResource(.{ .buffer = .{ .size = content_v.len * 4, .usage = .{ .vertex = true } } });
    defer dev.destroyResource(content_vb);
    @memcpy(std.mem.bytesAsSlice(f32, try dev.mapResource(a_vb))[0..a_v.len], &a_v);
    @memcpy(std.mem.bytesAsSlice(f32, try dev.mapResource(b_vb))[0..b_v.len], &b_v);
    @memcpy(std.mem.bytesAsSlice(f32, try dev.mapResource(content_vb))[0..content_v.len], &content_v);

    const stencil = try dev.createResource(.{ .buffer = .{ .size = W * H } });
    defer dev.destroyResource(stencil);
    const rt = try dev.createResource(.{ .image = .{ .width = W, .height = H, .format = .bgra8_unorm, .usage = .{ .render_target = true } } });
    defer dev.destroyResource(rt);

    const ctx = try dev.createContext();
    defer ctx.deinit();
    const cb = try ctx.beginCommands();
    defer cb.deinit();
    try cb.setRenderTarget(rt);
    try cb.clear(.{ .r = 0, .g = 0, .b = 0, .a = 1 });
    try cb.setStencilTarget(stencil, 0);
    try cb.bindPipeline(incr_pipe);
    try cb.bindVertexBuffer(a_vb);
    try cb.draw(6, 0); // mask A -> INCR left region to 1
    try cb.bindVertexBuffer(b_vb);
    try cb.draw(6, 0); // mask B -> INCR right region. overlap becomes 2
    try cb.bindPipeline(content_pipe);
    try cb.bindVertexBuffer(content_vb);
    try cb.draw(6, 0); // red where stencil == 2 (the intersection)
    ctx.submit(cb) catch return error.SkipZigTest;

    const fb = std.mem.bytesAsSlice(u32, try dev.mapResource(rt));
    const px = struct {
        fn red(p: u32) bool {
            return @as(u8, @truncate(p >> 16)) > 200 and @as(u8, @truncate(p >> 8)) < 60 and @as(u8, @truncate(p)) < 60;
        }
        fn black(p: u32) bool {
            return @as(u8, @truncate(p >> 16)) < 60 and @as(u8, @truncate(p >> 8)) < 60 and @as(u8, @truncate(p)) < 60;
        }
    };
    // Center (128) = overlap (stencil 2) -> RED. Left (40, stencil 1) + right (216, stencil 1) -> BLACK.
    try std.testing.expect(px.red(fb[128 * W + 128]));
    try std.testing.expect(px.black(fb[128 * W + 40]));
    try std.testing.expect(px.black(fb[128 * W + 216]));
}

test "two-sided stencil: front and back faces write DIFFERENT stencil refs, on the NVIDIA GPU and in software (skips without a GPU)" {
    const glsl = @import("../../glsl.zig");
    const gpa = std.testing.allocator;
    const nv = NvDevice.create(gpa) catch return error.SkipZigTest;
    defer nv.deinit();
    const sw = @import("../software.zig");
    const sw_dev = try sw.driver.createDevice(gpa);
    defer sw_dev.deinit();
    const W: u32 = 256;
    const H: u32 = 256;

    // A mask pipeline with two-sided stencil: front faces REPLACE the stencil with ref=1, back
    // faces with ref=2 (both ALWAYS pass). A left triangle drawn front-facing writes 1; a right
    // triangle drawn back-facing writes 2. Content (EQUAL ref=1) then shows only the left (front)
    // region. If two-sided were ignored (front state on both), the right would also be 1 -> red.
    // The winding convention matches the gl_FrontFacing oracle: the (top, bl, br) order is back
    // facing here, so (top, br, bl) is front. Verified on both the GPU and the software raster.
    const vs_src =
        \\attribute vec2 position;
        \\attribute vec3 color;
        \\varying vec3 vColor;
        \\void main() { gl_Position = vec4(position, 0.0, 1.0); vColor = color; }
    ;
    const fs_src =
        \\precision mediump float;
        \\varying vec3 vColor;
        \\void main() { gl_FragColor = vec4(vColor, 1.0); }
    ;
    const check = struct {
        fn run(a: std.mem.Allocator, dev: hal.Device) !void {
            const vs_bytes = try glsl.compileForStage(a, vs_src, .vertex);
            defer a.free(vs_bytes);
            const fs_bytes = try glsl.compileForStage(a, fs_src, .fragment);
            defer a.free(fs_bytes);
            const vs = try dev.createShaderModule(.{ .stage = .vertex, .code = vs_bytes });
            defer dev.destroyShaderModule(vs);
            const fs = try dev.createShaderModule(.{ .stage = .fragment, .code = fs_bytes });
            defer dev.destroyShaderModule(fs);
            const attrs = [_]hal.VertexAttribute{
                .{ .location = 0, .format = .r32g32_float, .offset = 0 },
                .{ .location = 1, .format = .r32g32b32_float, .offset = 8 },
            };
            // Two-sided mask: front REPLACE ref=1, back REPLACE ref=2.
            const mask_pipe = try dev.createPipeline(.{
                .vertex = vs,
                .fragment = fs,
                .vertex_layout = .{ .stride = 20, .attributes = &attrs },
                .color_format = .bgra8_unorm,
                .stencil = .{ .test_enable = true, .compare_op = .always, .pass_op = .replace, .reference = 1 },
                .stencil_back = .{ .test_enable = true, .compare_op = .always, .pass_op = .replace, .reference = 2 },
            });
            defer dev.destroyPipeline(mask_pipe);
            // Content: draw where stencil == 1 (the FRONT-face region only).
            const content_pipe = try dev.createPipeline(.{
                .vertex = vs,
                .fragment = fs,
                .vertex_layout = .{ .stride = 20, .attributes = &attrs },
                .color_format = .bgra8_unorm,
                .stencil = .{ .test_enable = true, .compare_op = .equal, .pass_op = .keep, .reference = 1 },
            });
            defer dev.destroyPipeline(content_pipe);
            // Left triangle FRONT winding (top, br, bl); right triangle BACK winding (top, bl, br).
            const tri = struct {
                fn v(x0: f32, x1: f32, front: bool) [15]f32 {
                    const top = [2]f32{ (x0 + x1) * 0.5, 0.6 };
                    const bl = [2]f32{ x0, -0.6 };
                    const br = [2]f32{ x1, -0.6 };
                    // FRONT = (top, br, bl); BACK = (top, bl, br). Color is unused (content is red).
                    const a2 = if (front) br else bl;
                    const b2 = if (front) bl else br;
                    return .{ top[0], top[1], 0, 0, 0, a2[0], a2[1], 0, 0, 0, b2[0], b2[1], 0, 0, 0 };
                }
            };
            const left = tri.v(-0.9, -0.1, true); // FRONT
            const right = tri.v(0.1, 0.9, false); // BACK
            const content = [_]f32{ -1, -1, 1, 0, 0, 3, -1, 1, 0, 0, -1, 3, 1, 0, 0 }; // fullscreen red
            const lvb = try dev.createResource(.{ .buffer = .{ .size = 60, .usage = .{ .vertex = true } } });
            defer dev.destroyResource(lvb);
            const rvb = try dev.createResource(.{ .buffer = .{ .size = 60, .usage = .{ .vertex = true } } });
            defer dev.destroyResource(rvb);
            const cvb = try dev.createResource(.{ .buffer = .{ .size = 60, .usage = .{ .vertex = true } } });
            defer dev.destroyResource(cvb);
            @memcpy(std.mem.bytesAsSlice(f32, try dev.mapResource(lvb))[0..15], &left);
            @memcpy(std.mem.bytesAsSlice(f32, try dev.mapResource(rvb))[0..15], &right);
            @memcpy(std.mem.bytesAsSlice(f32, try dev.mapResource(cvb))[0..15], &content);
            const stencil = try dev.createResource(.{ .buffer = .{ .size = W * H } });
            defer dev.destroyResource(stencil);
            const rt = try dev.createResource(.{ .image = .{ .width = W, .height = H, .format = .bgra8_unorm, .usage = .{ .render_target = true } } });
            defer dev.destroyResource(rt);
            const ctx = try dev.createContext();
            defer ctx.deinit();
            const cb = try ctx.beginCommands();
            defer cb.deinit();
            try cb.setRenderTarget(rt);
            try cb.clear(.{ .r = 0, .g = 0, .b = 0, .a = 1 });
            try cb.setStencilTarget(stencil, 0);
            try cb.bindPipeline(mask_pipe);
            try cb.bindVertexBuffer(lvb);
            try cb.draw(3, 0); // left FRONT -> stencil 1
            try cb.bindVertexBuffer(rvb);
            try cb.draw(3, 0); // right BACK -> stencil 2
            try cb.bindPipeline(content_pipe);
            try cb.bindVertexBuffer(cvb);
            try cb.draw(3, 0); // red where stencil == 1 (the FRONT/left region)
            ctx.submit(cb) catch return error.SkipZigTest;
            const fb = std.mem.bytesAsSlice(u32, try dev.mapResource(rt));
            const isRed = struct {
                fn f(p: u32) bool {
                    return @as(u8, @truncate(p >> 16)) > 200 and @as(u8, @truncate(p >> 8)) < 60;
                }
            }.f;
            // Left (front, stencil 1 == ref 1) RED; right (back, stencil 2 != 1) BLACK.
            try std.testing.expect(isRed(fb[128 * W + 40]));
            try std.testing.expect(!isRed(fb[128 * W + 216]));
        }
    };
    try check.run(gpa, nv);
    try check.run(gpa, sw_dev);
}

test "nvidia combined depth+stencil: stencil clips AND depth occludes in ONE framebuffer on the GPU (skips without a GPU)" {
    const glsl = @import("../../glsl.zig");
    const gpa = std.testing.allocator;
    const dev = NvDevice.create(gpa) catch return error.SkipZigTest;
    defer dev.deinit();
    const W: u32 = 256;
    const H: u32 = 256;

    // pos vec3 (x,y,z) + color vec3, so each quad carries its own depth.
    const vs_src =
        \\attribute vec3 position;
        \\attribute vec3 color;
        \\varying vec3 vColor;
        \\void main() { gl_Position = vec4(position, 1.0); vColor = color; }
    ;
    const fs_src =
        \\precision mediump float;
        \\varying vec3 vColor;
        \\void main() { gl_FragColor = vec4(vColor, 1.0); }
    ;
    const vs_bytes = try glsl.compileForStage(gpa, vs_src, .vertex);
    defer gpa.free(vs_bytes);
    const fs_bytes = try glsl.compileForStage(gpa, fs_src, .fragment);
    defer gpa.free(fs_bytes);
    const vs = try dev.createShaderModule(.{ .stage = .vertex, .code = vs_bytes });
    defer dev.destroyShaderModule(vs);
    const fs = try dev.createShaderModule(.{ .stage = .fragment, .code = fs_bytes });
    defer dev.destroyShaderModule(fs);

    const attrs = [_]hal.VertexAttribute{
        .{ .location = 0, .format = .r32g32b32_float, .offset = 0 },
        .{ .location = 1, .format = .r32g32b32_float, .offset = 12 },
    };
    const depth_ls = hal.DepthState{ .test_enable = true, .write_enable = true, .compare_op = .less };
    // MASK: depth LESS+write, stencil ALWAYS -> REPLACE ref=1. Writes stencil=1 + depth on left.
    const mask_pipe = try dev.createPipeline(.{
        .vertex = vs,
        .fragment = fs,
        .vertex_layout = .{ .stride = 24, .attributes = &attrs },
        .color_format = .bgra8_unorm,
        .depth = depth_ls,
        .stencil = .{ .test_enable = true, .compare_op = .always, .pass_op = .replace, .reference = 1 },
    });
    defer dev.destroyPipeline(mask_pipe);
    // CONTENT: depth LESS+write, stencil EQUAL ref=1 (clip to the mask). Depth + stencil BOTH gate.
    const content_pipe = try dev.createPipeline(.{
        .vertex = vs,
        .fragment = fs,
        .vertex_layout = .{ .stride = 24, .attributes = &attrs },
        .color_format = .bgra8_unorm,
        .depth = depth_ls,
        .stencil = .{ .test_enable = true, .compare_op = .equal, .pass_op = .keep, .reference = 1 },
    });
    defer dev.destroyPipeline(content_pipe);

    // 6 verts/quad: pos vec3 (z carried) + color vec3.
    const quad = struct {
        fn make(x0: f32, x1: f32, z: f32, r: f32, g: f32, b: f32) [36]f32 {
            return .{
                x0, -1, z, r, g, b, x1, -1, z, r, g, b, x1, 1, z, r, g, b,
                x0, -1, z, r, g, b, x1, 1,  z, r, g, b, x0, 1, z, r, g, b,
            };
        }
    };
    // MASK: left half, z=0.5, black. RED: fullscreen, z=0.2 (in front). GREEN: left half, z=0.7
    // (behind red) - depth must REJECT it so the left stays RED, proving the depth plane works
    // alongside the stencil plane.
    const mask_v = quad.make(-1, 0, 0.5, 0, 0, 0);
    const red_v = quad.make(-1, 1, 0.2, 1, 0, 0);
    const green_v = quad.make(-1, 0, 0.7, 0, 1, 0);
    const mkvb = struct {
        fn f(d: hal.Device, v: []const f32) !*hal.Resource {
            const b = try d.createResource(.{ .buffer = .{ .size = v.len * 4, .usage = .{ .vertex = true } } });
            @memcpy(std.mem.bytesAsSlice(f32, try d.mapResource(b))[0..v.len], v);
            return b;
        }
    }.f;
    const mask_vb = try mkvb(dev, &mask_v);
    defer dev.destroyResource(mask_vb);
    const red_vb = try mkvb(dev, &red_v);
    defer dev.destroyResource(red_vb);
    const green_vb = try mkvb(dev, &green_v);
    defer dev.destroyResource(green_vb);

    const depth_a = try dev.createResource(.{ .image = .{ .width = W, .height = H, .format = .depth32_float, .usage = .{} } });
    defer dev.destroyResource(depth_a);
    const stencil = try dev.createResource(.{ .buffer = .{ .size = W * H } });
    defer dev.destroyResource(stencil);
    const rt = try dev.createResource(.{ .image = .{ .width = W, .height = H, .format = .bgra8_unorm, .usage = .{ .render_target = true } } });
    defer dev.destroyResource(rt);

    const ctx = try dev.createContext();
    defer ctx.deinit();
    const cb = try ctx.beginCommands();
    defer cb.deinit();
    try cb.setRenderTarget(rt);
    try cb.clear(.{ .r = 0, .g = 0, .b = 0, .a = 1 });
    try cb.setDepthTarget(depth_a, 1.0); // clear depth to far
    try cb.setStencilTarget(stencil, 0); // clear stencil to 0
    try cb.bindPipeline(mask_pipe);
    try cb.bindVertexBuffer(mask_vb);
    try cb.draw(6, 0);
    try cb.bindPipeline(content_pipe);
    try cb.bindVertexBuffer(red_vb);
    try cb.draw(6, 0);
    try cb.bindVertexBuffer(green_vb);
    try cb.draw(6, 0);
    ctx.submit(cb) catch return error.SkipZigTest;

    const fb = std.mem.bytesAsSlice(u32, try dev.mapResource(rt));
    const c = struct {
        fn red(p: u32) bool {
            return @as(u8, @truncate(p >> 16)) > 200 and @as(u8, @truncate(p >> 8)) < 60 and @as(u8, @truncate(p)) < 60;
        }
        fn black(p: u32) bool {
            return @as(u8, @truncate(p >> 16)) < 60 and @as(u8, @truncate(p >> 8)) < 60 and @as(u8, @truncate(p)) < 60;
        }
    };
    // LEFT (64,128): stencil==1 AND red (z=0.2) beat green (z=0.7) on depth -> RED. If the depth
    // plane were ignored, green would have overwritten -> the RED assert would fail.
    try std.testing.expect(c.red(fb[128 * W + 64]));
    // RIGHT (192,128): stencil==0 -> both content draws clipped -> BLACK (clear).
    try std.testing.expect(c.black(fb[128 * W + 192]));
}

test "nvidia depth bias (glPolygonOffset): a negative constant lets a coplanar draw win on the GPU (skips without a GPU)" {
    const glsl = @import("../../glsl.zig");
    const gpa = std.testing.allocator;
    const dev = NvDevice.create(gpa) catch return error.SkipZigTest;
    defer dev.deinit();
    const W: u32 = 256;
    const H: u32 = 256;

    const vs_src =
        \\attribute vec3 position;
        \\attribute vec3 color;
        \\varying vec3 vColor;
        \\void main() { gl_Position = vec4(position, 1.0); vColor = color; }
    ;
    const fs_src =
        \\precision mediump float;
        \\varying vec3 vColor;
        \\void main() { gl_FragColor = vec4(vColor, 1.0); }
    ;
    const vs_bytes = try glsl.compileForStage(gpa, vs_src, .vertex);
    defer gpa.free(vs_bytes);
    const fs_bytes = try glsl.compileForStage(gpa, fs_src, .fragment);
    defer gpa.free(fs_bytes);
    const vs = try dev.createShaderModule(.{ .stage = .vertex, .code = vs_bytes });
    defer dev.destroyShaderModule(vs);
    const fs = try dev.createShaderModule(.{ .stage = .fragment, .code = fs_bytes });
    defer dev.destroyShaderModule(fs);

    const attrs = [_]hal.VertexAttribute{
        .{ .location = 0, .format = .r32g32b32_float, .offset = 0 },
        .{ .location = 1, .format = .r32g32b32_float, .offset = 12 },
    };
    const depth_ls = hal.DepthState{ .test_enable = true, .write_enable = true, .compare_op = .less };
    const base_pipe = try dev.createPipeline(.{
        .vertex = vs,
        .fragment = fs,
        .vertex_layout = .{ .stride = 24, .attributes = &attrs },
        .color_format = .bgra8_unorm,
        .depth = depth_ls,
    });
    defer dev.destroyPipeline(base_pipe);
    // Large negative constant: with DEPTH_FORMAT_DEPENDENT the HW scales by the ZF32 r (~2^-24
    // near z=0.5), so -2^23 gives roughly a -0.5 offset - clearly below 0.5 so green wins LESS.
    var biased = depth_ls;
    biased.bias_enable = true;
    biased.bias_constant = -8388608;
    const biased_pipe = try dev.createPipeline(.{
        .vertex = vs,
        .fragment = fs,
        .vertex_layout = .{ .stride = 24, .attributes = &attrs },
        .color_format = .bgra8_unorm,
        .depth = biased,
    });
    defer dev.destroyPipeline(biased_pipe);

    const quad = struct {
        fn make(r: f32, g: f32, b: f32) [36]f32 {
            return .{
                -1, -1, 0.5, r, g, b, 1, -1, 0.5, r, g, b, 1,  1, 0.5, r, g, b,
                -1, -1, 0.5, r, g, b, 1, 1,  0.5, r, g, b, -1, 1, 0.5, r, g, b,
            };
        }
    };
    const red_v = quad.make(1, 0, 0);
    const green_v = quad.make(0, 1, 0);
    const mkvb = struct {
        fn f(d: hal.Device, v: []const f32) !*hal.Resource {
            const b = try d.createResource(.{ .buffer = .{ .size = v.len * 4, .usage = .{ .vertex = true } } });
            @memcpy(std.mem.bytesAsSlice(f32, try d.mapResource(b))[0..v.len], v);
            return b;
        }
    }.f;
    const red_vb = try mkvb(dev, &red_v);
    defer dev.destroyResource(red_vb);
    const green_vb = try mkvb(dev, &green_v);
    defer dev.destroyResource(green_vb);

    const depth_a = try dev.createResource(.{ .image = .{ .width = W, .height = H, .format = .depth32_float, .usage = .{} } });
    defer dev.destroyResource(depth_a);
    const rt = try dev.createResource(.{ .image = .{ .width = W, .height = H, .format = .bgra8_unorm, .usage = .{ .render_target = true } } });
    defer dev.destroyResource(rt);

    const ctx = try dev.createContext();
    defer ctx.deinit();
    const cb = try ctx.beginCommands();
    defer cb.deinit();
    try cb.setRenderTarget(rt);
    try cb.clear(.{ .r = 0, .g = 0, .b = 0, .a = 1 });
    try cb.setDepthTarget(depth_a, 1.0);
    try cb.bindPipeline(base_pipe);
    try cb.bindVertexBuffer(red_vb);
    try cb.draw(6, 0);
    try cb.bindPipeline(biased_pipe);
    try cb.bindVertexBuffer(green_vb);
    try cb.draw(6, 0);
    ctx.submit(cb) catch return error.SkipZigTest;

    const fb = std.mem.bytesAsSlice(u32, try dev.mapResource(rt));
    const center = fb[128 * W + 128];
    // bgra8 readback: green = byte1. The biased green (pulled in front) beat the red at z=0.5.
    const g: u8 = @truncate(center >> 8);
    const r: u8 = @truncate(center >> 16);
    try std.testing.expect(g > 200 and r < 60);
}

test "nvidia color write mask (glColorMask): masked channels keep the destination on the GPU (skips without a GPU)" {
    const glsl = @import("../../glsl.zig");
    const gpa = std.testing.allocator;
    const dev = NvDevice.create(gpa) catch return error.SkipZigTest;
    defer dev.deinit();
    const W: u32 = 256;
    const H: u32 = 256;

    const vs_src =
        \\attribute vec3 position;
        \\attribute vec3 color;
        \\varying vec3 vColor;
        \\void main() { gl_Position = vec4(position, 1.0); vColor = color; }
    ;
    const fs_src =
        \\precision mediump float;
        \\varying vec3 vColor;
        \\void main() { gl_FragColor = vec4(vColor, 1.0); }
    ;
    const vs_bytes = try glsl.compileForStage(gpa, vs_src, .vertex);
    defer gpa.free(vs_bytes);
    const fs_bytes = try glsl.compileForStage(gpa, fs_src, .fragment);
    defer gpa.free(fs_bytes);
    const vs = try dev.createShaderModule(.{ .stage = .vertex, .code = vs_bytes });
    defer dev.destroyShaderModule(vs);
    const fs = try dev.createShaderModule(.{ .stage = .fragment, .code = fs_bytes });
    defer dev.destroyShaderModule(fs);

    const attrs = [_]hal.VertexAttribute{
        .{ .location = 0, .format = .r32g32b32_float, .offset = 0 },
        .{ .location = 1, .format = .r32g32b32_float, .offset = 12 },
    };
    // Only R + A may write. G and B are masked (keep the black clear). A WHITE draw -> RED.
    const pipe = try dev.createPipeline(.{
        .vertex = vs,
        .fragment = fs,
        .vertex_layout = .{ .stride = 24, .attributes = &attrs },
        .color_format = .bgra8_unorm,
        .blend = .{ .write_mask = .{ true, false, false, true } },
    });
    defer dev.destroyPipeline(pipe);

    const white = [36]f32{
        -1, -1, 0, 1, 1, 1, 1, -1, 0, 1, 1, 1, 1,  1, 0, 1, 1, 1,
        -1, -1, 0, 1, 1, 1, 1, 1,  0, 1, 1, 1, -1, 1, 0, 1, 1, 1,
    };
    const vb = try dev.createResource(.{ .buffer = .{ .size = white.len * 4, .usage = .{ .vertex = true } } });
    defer dev.destroyResource(vb);
    @memcpy(std.mem.bytesAsSlice(f32, try dev.mapResource(vb))[0..white.len], &white);

    const rt = try dev.createResource(.{ .image = .{ .width = W, .height = H, .format = .bgra8_unorm, .usage = .{ .render_target = true } } });
    defer dev.destroyResource(rt);

    const ctx = try dev.createContext();
    defer ctx.deinit();
    const cb = try ctx.beginCommands();
    defer cb.deinit();
    try cb.setRenderTarget(rt);
    try cb.clear(.{ .r = 0, .g = 0, .b = 0, .a = 1 });
    try cb.bindPipeline(pipe);
    try cb.bindVertexBuffer(vb);
    try cb.draw(6, 0);
    ctx.submit(cb) catch return error.SkipZigTest;

    const fb = std.mem.bytesAsSlice(u32, try dev.mapResource(rt));
    const center = fb[128 * W + 128];
    // bgra8 readback: white through an R+A mask -> R=255 (byte2), G=0 (byte1), B=0 (byte0) = RED.
    const r: u8 = @truncate(center >> 16);
    const g: u8 = @truncate(center >> 8);
    const b: u8 = @truncate(center);
    try std.testing.expect(r > 200 and g < 40 and b < 40);
}

test "nvidia instancing: gl_InstanceIndex draws each instance at its UBO offset on the GPU (skips without a GPU)" {
    const gpa = std.testing.allocator;
    const vspirv = @import("vulcan-spirv");
    const opc = vspirv.opcodes;
    const sc = opc.StorageClass;

    // Vertex-pulling + INSTANCED VS (same shape as the software instancing oracle):
    //   gl_Position = u.pos[gl_VertexIndex] + u.offset[gl_InstanceIndex]; vc = u.col[vi].rgb;
    // gl_InstanceIndex -> the nvidia isel emits ALD a[0x2f8]; the pipeline declares
    // readsInstanceId, and the context replays the draw per instance bumping the base.
    var vsb = try vspirv.binary.Builder.init(gpa, 42);
    defer vsb.deinit(gpa);
    try vsb.emit(gpa, opc.EntryPoint, &.{ opc.ExecutionModel.vertex, 28, 0, 19, 20, 23, 25 });
    try vsb.emit(gpa, opc.Decorate, &.{ 11, opc.Decoration.block });
    try vsb.emit(gpa, opc.MemberDecorate, &.{ 11, 0, opc.Decoration.offset, 0 });
    try vsb.emit(gpa, opc.MemberDecorate, &.{ 11, 1, opc.Decoration.offset, 48 });
    try vsb.emit(gpa, opc.MemberDecorate, &.{ 11, 2, opc.Decoration.offset, 96 });
    try vsb.emit(gpa, opc.Decorate, &.{ 10, opc.Decoration.array_stride, 16 });
    try vsb.emit(gpa, opc.Decorate, &.{ 14, opc.Decoration.array_stride, 16 });
    try vsb.emit(gpa, opc.Decorate, &.{ 16, opc.Decoration.binding, 0 });
    try vsb.emit(gpa, opc.Decorate, &.{ 19, opc.Decoration.builtin, opc.BuiltIn.vertex_index });
    try vsb.emit(gpa, opc.Decorate, &.{ 20, opc.Decoration.builtin, opc.BuiltIn.instance_index });
    try vsb.emit(gpa, opc.Decorate, &.{ 21, opc.Decoration.block });
    try vsb.emit(gpa, opc.MemberDecorate, &.{ 21, 0, opc.Decoration.builtin, opc.BuiltIn.position });
    try vsb.emit(gpa, opc.Decorate, &.{ 25, opc.Decoration.location, 0 });
    try vsb.emit(gpa, opc.TypeVoid, &.{1});
    try vsb.emit(gpa, opc.TypeFloat, &.{ 2, 32 });
    try vsb.emit(gpa, opc.TypeVector, &.{ 3, 2, 4 });
    try vsb.emit(gpa, opc.TypeVector, &.{ 4, 2, 3 });
    try vsb.emit(gpa, opc.TypeInt, &.{ 5, 32, 1 });
    try vsb.emit(gpa, opc.Constant, &.{ 5, 6, 0 });
    try vsb.emit(gpa, opc.Constant, &.{ 5, 7, 1 });
    try vsb.emit(gpa, opc.TypeInt, &.{ 8, 32, 0 });
    try vsb.emit(gpa, opc.Constant, &.{ 8, 9, 3 });
    try vsb.emit(gpa, opc.TypeArray, &.{ 10, 3, 9 });
    try vsb.emit(gpa, opc.Constant, &.{ 5, 12, 2 });
    try vsb.emit(gpa, opc.Constant, &.{ 8, 13, 2 });
    try vsb.emit(gpa, opc.TypeArray, &.{ 14, 3, 13 });
    try vsb.emit(gpa, opc.TypeStruct, &.{ 11, 10, 10, 14 });
    try vsb.emit(gpa, opc.TypePointer, &.{ 15, sc.uniform, 11 });
    try vsb.emit(gpa, opc.Variable, &.{ 15, 16, sc.uniform });
    try vsb.emit(gpa, opc.TypePointer, &.{ 17, sc.uniform, 3 });
    try vsb.emit(gpa, opc.TypePointer, &.{ 18, sc.input, 5 });
    try vsb.emit(gpa, opc.Variable, &.{ 18, 19, sc.input });
    try vsb.emit(gpa, opc.Variable, &.{ 18, 20, sc.input });
    try vsb.emit(gpa, opc.TypeStruct, &.{ 21, 3 });
    try vsb.emit(gpa, opc.TypePointer, &.{ 22, sc.output, 21 });
    try vsb.emit(gpa, opc.Variable, &.{ 22, 23, sc.output });
    try vsb.emit(gpa, opc.TypePointer, &.{ 24, sc.output, 4 });
    try vsb.emit(gpa, opc.Variable, &.{ 24, 25, sc.output });
    try vsb.emit(gpa, opc.TypePointer, &.{ 26, sc.output, 3 });
    try vsb.emit(gpa, opc.TypeFunction, &.{ 27, 1 });
    try vsb.emit(gpa, opc.Function, &.{ 1, 28, 0, 27 });
    try vsb.emit(gpa, opc.Label, &.{29});
    try vsb.emit(gpa, opc.Load, &.{ 5, 30, 19 });
    try vsb.emit(gpa, opc.AccessChain, &.{ 17, 31, 16, 6, 30 });
    try vsb.emit(gpa, opc.Load, &.{ 3, 32, 31 });
    try vsb.emit(gpa, opc.Load, &.{ 5, 33, 20 });
    try vsb.emit(gpa, opc.AccessChain, &.{ 17, 34, 16, 12, 33 });
    try vsb.emit(gpa, opc.Load, &.{ 3, 35, 34 });
    try vsb.emit(gpa, opc.FAdd, &.{ 3, 36, 32, 35 });
    try vsb.emit(gpa, opc.AccessChain, &.{ 26, 37, 23, 6 });
    try vsb.emit(gpa, opc.Store, &.{ 37, 36 });
    try vsb.emit(gpa, opc.Load, &.{ 5, 38, 19 });
    try vsb.emit(gpa, opc.AccessChain, &.{ 17, 39, 16, 7, 38 });
    try vsb.emit(gpa, opc.Load, &.{ 3, 40, 39 });
    try vsb.emit(gpa, opc.VectorShuffle, &.{ 4, 41, 40, 40, 0, 1, 2 });
    try vsb.emit(gpa, opc.Store, &.{ 25, 41 });
    try vsb.emit(gpa, opc.Return, &.{});
    try vsb.emit(gpa, opc.FunctionEnd, &.{});

    const glsl = @import("../../glsl.zig");
    const fs_src =
        \\precision mediump float;
        \\varying vec3 vColor;
        \\void main() { gl_FragColor = vec4(vColor, 1.0); }
    ;
    const dev = NvDevice.create(gpa) catch return error.SkipZigTest;
    defer dev.deinit();
    const fs_bytes = try glsl.compileForStage(gpa, fs_src, .fragment);
    defer gpa.free(fs_bytes);
    const vs = try dev.createShaderModule(.{ .stage = .vertex, .code = std.mem.sliceAsBytes(vsb.words.items) });
    defer dev.destroyShaderModule(vs);
    const fs = try dev.createShaderModule(.{ .stage = .fragment, .code = fs_bytes });
    defer dev.destroyShaderModule(fs);

    const W: u32 = 256;
    const H: u32 = 256;
    const pipe = try dev.createPipeline(.{
        .vertex = vs,
        .fragment = fs,
        .vertex_layout = .{ .stride = 0, .attributes = &.{} }, // pulling: no attributes
        .color_format = .bgra8_unorm,
    });
    defer dev.destroyPipeline(pipe);

    // UBO: pos[3] (small triangle), col[3] (red), offset[2] (left/right).
    const ubo = try dev.createResource(.{ .buffer = .{ .size = 128, .usage = .{ .uniform = true } } });
    defer dev.destroyResource(ubo);
    {
        const f = std.mem.bytesAsSlice(f32, try dev.mapResource(ubo));
        const base = [3][2]f32{ .{ -0.3, -0.3 }, .{ 0.3, -0.3 }, .{ 0.0, 0.3 } };
        inline for (base, 0..) |v, k| {
            f[k * 4 + 0] = v[0];
            f[k * 4 + 1] = v[1];
            f[k * 4 + 2] = 0;
            f[k * 4 + 3] = 1;
        }
        inline for (0..3) |k| {
            f[12 + k * 4 + 0] = 1;
            f[12 + k * 4 + 1] = 0;
            f[12 + k * 4 + 2] = 0;
            f[12 + k * 4 + 3] = 1;
        }
        f[24] = -0.5;
        f[25] = 0;
        f[26] = 0;
        f[27] = 0;
        f[28] = 0.5;
        f[29] = 0;
        f[30] = 0;
        f[31] = 0;
    }

    const rt = try dev.createResource(.{ .image = .{ .width = W, .height = H, .format = .bgra8_unorm, .usage = .{ .render_target = true } } });
    defer dev.destroyResource(rt);
    const ctx = try dev.createContext();
    defer ctx.deinit();
    const cb = try ctx.beginCommands();
    defer cb.deinit();
    try cb.setRenderTarget(rt);
    try cb.clear(.{ .r = 0, .g = 0, .b = 0, .a = 1 });
    try cb.bindPipeline(pipe);
    try cb.bindUniformBuffer(0, ubo);
    try cb.drawInstanced(3, 2, 0, 0);
    ctx.submit(cb) catch return error.SkipZigTest;

    const fb = std.mem.bytesAsSlice(u32, try dev.mapResource(rt));
    const isRed = struct {
        fn f(p: u32) bool {
            return @as(u8, @truncate(p >> 16)) > 200 and @as(u8, @truncate(p >> 8)) < 60 and @as(u8, @truncate(p)) < 60;
        }
    }.f;
    // Instance 0 at NDC x -0.5 -> window x ~64 (LEFT); instance 1 at +0.5 -> x ~192.
    try std.testing.expect(isRed(fb[128 * W + 64])); // instance 0 drew
    try std.testing.expect(isRed(fb[128 * W + 192])); // instance 1 drew
}

test "nvidia scissor: SET_SCISSOR clips a fullscreen draw to the top-left quadrant on the GPU (skips without a GPU)" {
    const gpa = std.testing.allocator;
    const vspirv = @import("vulcan-spirv");
    const opc = vspirv.opcodes;
    const sc = opc.StorageClass;

    // Vertex-pulling fullscreen VS: gl_Position = u.pos[gl_VertexIndex] (a big [-1,3]
    // triangle covering the whole surface, so any un-drawn pixel proves the scissor clip).
    var vsb = try vspirv.binary.Builder.init(gpa, 40);
    defer vsb.deinit(gpa);
    try vsb.emit(gpa, opc.EntryPoint, &.{ opc.ExecutionModel.vertex, 28, 0, 19, 23 });
    try vsb.emit(gpa, opc.Decorate, &.{ 11, opc.Decoration.block });
    try vsb.emit(gpa, opc.MemberDecorate, &.{ 11, 0, opc.Decoration.offset, 0 });
    try vsb.emit(gpa, opc.Decorate, &.{ 10, opc.Decoration.array_stride, 16 });
    try vsb.emit(gpa, opc.Decorate, &.{ 16, opc.Decoration.binding, 0 });
    try vsb.emit(gpa, opc.Decorate, &.{ 19, opc.Decoration.builtin, opc.BuiltIn.vertex_index });
    try vsb.emit(gpa, opc.Decorate, &.{ 21, opc.Decoration.block });
    try vsb.emit(gpa, opc.MemberDecorate, &.{ 21, 0, opc.Decoration.builtin, opc.BuiltIn.position });
    try vsb.emit(gpa, opc.TypeVoid, &.{1});
    try vsb.emit(gpa, opc.TypeFloat, &.{ 2, 32 });
    try vsb.emit(gpa, opc.TypeVector, &.{ 3, 2, 4 });
    try vsb.emit(gpa, opc.TypeInt, &.{ 5, 32, 1 });
    try vsb.emit(gpa, opc.Constant, &.{ 5, 6, 0 });
    try vsb.emit(gpa, opc.TypeInt, &.{ 8, 32, 0 });
    try vsb.emit(gpa, opc.Constant, &.{ 8, 9, 3 });
    try vsb.emit(gpa, opc.TypeArray, &.{ 10, 3, 9 });
    try vsb.emit(gpa, opc.TypeStruct, &.{ 11, 10 });
    try vsb.emit(gpa, opc.TypePointer, &.{ 15, sc.uniform, 11 });
    try vsb.emit(gpa, opc.Variable, &.{ 15, 16, sc.uniform });
    try vsb.emit(gpa, opc.TypePointer, &.{ 17, sc.uniform, 3 });
    try vsb.emit(gpa, opc.TypePointer, &.{ 18, sc.input, 5 });
    try vsb.emit(gpa, opc.Variable, &.{ 18, 19, sc.input });
    try vsb.emit(gpa, opc.TypeStruct, &.{ 21, 3 });
    try vsb.emit(gpa, opc.TypePointer, &.{ 22, sc.output, 21 });
    try vsb.emit(gpa, opc.Variable, &.{ 22, 23, sc.output });
    try vsb.emit(gpa, opc.TypePointer, &.{ 26, sc.output, 3 });
    try vsb.emit(gpa, opc.TypeFunction, &.{ 27, 1 });
    try vsb.emit(gpa, opc.Function, &.{ 1, 28, 0, 27 });
    try vsb.emit(gpa, opc.Label, &.{29});
    try vsb.emit(gpa, opc.Load, &.{ 5, 30, 19 });
    try vsb.emit(gpa, opc.AccessChain, &.{ 17, 31, 16, 6, 30 });
    try vsb.emit(gpa, opc.Load, &.{ 3, 32, 31 });
    try vsb.emit(gpa, opc.AccessChain, &.{ 26, 37, 23, 6 });
    try vsb.emit(gpa, opc.Store, &.{ 37, 32 });
    try vsb.emit(gpa, opc.Return, &.{});
    try vsb.emit(gpa, opc.FunctionEnd, &.{});

    const glsl = @import("../../glsl.zig");
    const fs_src =
        \\precision mediump float;
        \\void main() { gl_FragColor = vec4(1.0, 0.0, 0.0, 1.0); }
    ;
    const dev = NvDevice.create(gpa) catch return error.SkipZigTest;
    defer dev.deinit();
    const fs_bytes = try glsl.compileForStage(gpa, fs_src, .fragment);
    defer gpa.free(fs_bytes);
    const vs = try dev.createShaderModule(.{ .stage = .vertex, .code = std.mem.sliceAsBytes(vsb.words.items) });
    defer dev.destroyShaderModule(vs);
    const fs = try dev.createShaderModule(.{ .stage = .fragment, .code = fs_bytes });
    defer dev.destroyShaderModule(fs);

    const W: u32 = 256;
    const H: u32 = 256;
    const pipe = try dev.createPipeline(.{
        .vertex = vs,
        .fragment = fs,
        .vertex_layout = .{ .stride = 0, .attributes = &.{} },
        .color_format = .bgra8_unorm,
    });
    defer dev.destroyPipeline(pipe);

    const ubo = try dev.createResource(.{ .buffer = .{ .size = 48, .usage = .{ .uniform = true } } });
    defer dev.destroyResource(ubo);
    {
        const f = std.mem.bytesAsSlice(f32, try dev.mapResource(ubo));
        const pos = [3][2]f32{ .{ -1, -1 }, .{ 3, -1 }, .{ -1, 3 } };
        inline for (pos, 0..) |v, k| {
            f[k * 4 + 0] = v[0];
            f[k * 4 + 1] = v[1];
            f[k * 4 + 2] = 0;
            f[k * 4 + 3] = 1;
        }
    }

    const rt = try dev.createResource(.{ .image = .{ .width = W, .height = H, .format = .bgra8_unorm, .usage = .{ .render_target = true } } });
    defer dev.destroyResource(rt);
    const ctx = try dev.createContext();
    defer ctx.deinit();
    const cb = try ctx.beginCommands();
    defer cb.deinit();
    try cb.setRenderTarget(rt);
    try cb.clear(.{ .r = 0, .g = 0, .b = 0, .a = 1 });
    try cb.bindPipeline(pipe);
    try cb.bindUniformBuffer(0, ubo);
    // Clip the fullscreen red draw to the TOP-LEFT 128x128 quadrant (HAL top-left origin,
    // which matches the nvidia viewport: NDC y=-1 -> row 0).
    try cb.setScissor(.{ .x = 0, .y = 0, .width = 128, .height = 128 });
    try cb.draw(3, 0);
    ctx.submit(cb) catch return error.SkipZigTest;

    const fb = std.mem.bytesAsSlice(u32, try dev.mapResource(rt));
    const isRed = struct {
        fn f(p: u32) bool {
            return @as(u8, @truncate(p >> 16)) > 200 and @as(u8, @truncate(p >> 8)) < 60 and @as(u8, @truncate(p)) < 60;
        }
    }.f;
    const isBlack = struct {
        fn f(p: u32) bool {
            return @as(u8, @truncate(p >> 16)) < 60 and @as(u8, @truncate(p >> 8)) < 60 and @as(u8, @truncate(p)) < 60;
        }
    }.f;
    try std.testing.expect(isRed(fb[64 * W + 64])); // inside the scissor: drawn red
    try std.testing.expect(isBlack(fb[192 * W + 192])); // bottom-right: clipped -> black
    try std.testing.expect(isBlack(fb[64 * W + 192])); // top-right: clipped -> black
    try std.testing.expect(isBlack(fb[192 * W + 64])); // bottom-left: clipped -> black
}

test "nvidia MSAA: a slanted edge anti-aliases via supersampling on the GPU (partial-coverage resolve) (skips without a GPU)" {
    const gpa = std.testing.allocator;
    const vspirv = @import("vulcan-spirv");
    const opc = vspirv.opcodes;
    const sc = opc.StorageClass;

    // Vertex-pulling VS: gl_Position = u.pos[gl_VertexIndex] (a slanted right triangle whose
    // hypotenuse is a diagonal edge). Constant-red FS. The samples=4 (2x2 SSAA) render
    // box-downsamples to partial-coverage red along the diagonal. samples=1 has hard edges.
    var vsb = try vspirv.binary.Builder.init(gpa, 40);
    defer vsb.deinit(gpa);
    try vsb.emit(gpa, opc.EntryPoint, &.{ opc.ExecutionModel.vertex, 28, 0, 19, 23 });
    try vsb.emit(gpa, opc.Decorate, &.{ 11, opc.Decoration.block });
    try vsb.emit(gpa, opc.MemberDecorate, &.{ 11, 0, opc.Decoration.offset, 0 });
    try vsb.emit(gpa, opc.Decorate, &.{ 10, opc.Decoration.array_stride, 16 });
    try vsb.emit(gpa, opc.Decorate, &.{ 16, opc.Decoration.binding, 0 });
    try vsb.emit(gpa, opc.Decorate, &.{ 19, opc.Decoration.builtin, opc.BuiltIn.vertex_index });
    try vsb.emit(gpa, opc.Decorate, &.{ 21, opc.Decoration.block });
    try vsb.emit(gpa, opc.MemberDecorate, &.{ 21, 0, opc.Decoration.builtin, opc.BuiltIn.position });
    try vsb.emit(gpa, opc.TypeVoid, &.{1});
    try vsb.emit(gpa, opc.TypeFloat, &.{ 2, 32 });
    try vsb.emit(gpa, opc.TypeVector, &.{ 3, 2, 4 });
    try vsb.emit(gpa, opc.TypeInt, &.{ 5, 32, 1 });
    try vsb.emit(gpa, opc.Constant, &.{ 5, 6, 0 });
    try vsb.emit(gpa, opc.TypeInt, &.{ 8, 32, 0 });
    try vsb.emit(gpa, opc.Constant, &.{ 8, 9, 3 });
    try vsb.emit(gpa, opc.TypeArray, &.{ 10, 3, 9 });
    try vsb.emit(gpa, opc.TypeStruct, &.{ 11, 10 });
    try vsb.emit(gpa, opc.TypePointer, &.{ 15, sc.uniform, 11 });
    try vsb.emit(gpa, opc.Variable, &.{ 15, 16, sc.uniform });
    try vsb.emit(gpa, opc.TypePointer, &.{ 17, sc.uniform, 3 });
    try vsb.emit(gpa, opc.TypePointer, &.{ 18, sc.input, 5 });
    try vsb.emit(gpa, opc.Variable, &.{ 18, 19, sc.input });
    try vsb.emit(gpa, opc.TypeStruct, &.{ 21, 3 });
    try vsb.emit(gpa, opc.TypePointer, &.{ 22, sc.output, 21 });
    try vsb.emit(gpa, opc.Variable, &.{ 22, 23, sc.output });
    try vsb.emit(gpa, opc.TypePointer, &.{ 26, sc.output, 3 });
    try vsb.emit(gpa, opc.TypeFunction, &.{ 27, 1 });
    try vsb.emit(gpa, opc.Function, &.{ 1, 28, 0, 27 });
    try vsb.emit(gpa, opc.Label, &.{29});
    try vsb.emit(gpa, opc.Load, &.{ 5, 30, 19 });
    try vsb.emit(gpa, opc.AccessChain, &.{ 17, 31, 16, 6, 30 });
    try vsb.emit(gpa, opc.Load, &.{ 3, 32, 31 });
    try vsb.emit(gpa, opc.AccessChain, &.{ 26, 37, 23, 6 });
    try vsb.emit(gpa, opc.Store, &.{ 37, 32 });
    try vsb.emit(gpa, opc.Return, &.{});
    try vsb.emit(gpa, opc.FunctionEnd, &.{});

    const glsl = @import("../../glsl.zig");
    const fs_src =
        \\precision mediump float;
        \\void main() { gl_FragColor = vec4(1.0, 0.0, 0.0, 1.0); }
    ;
    const dev = NvDevice.create(gpa) catch return error.SkipZigTest;
    defer dev.deinit();
    const fs_bytes = try glsl.compileForStage(gpa, fs_src, .fragment);
    defer gpa.free(fs_bytes);
    const vs = try dev.createShaderModule(.{ .stage = .vertex, .code = std.mem.sliceAsBytes(vsb.words.items) });
    defer dev.destroyShaderModule(vs);
    const fs = try dev.createShaderModule(.{ .stage = .fragment, .code = fs_bytes });
    defer dev.destroyShaderModule(fs);

    const W: u32 = 128;
    const H: u32 = 128;

    const renderAt = struct {
        fn go(d: anytype, vsm: anytype, fsm: anytype, samples: u8) ![W * H * 4]u8 {
            const pipe = try d.createPipeline(.{
                .vertex = vsm,
                .fragment = fsm,
                .vertex_layout = .{ .stride = 0, .attributes = &.{} },
                .color_format = .bgra8_unorm,
                .samples = samples,
            });
            defer d.destroyPipeline(pipe);

            const ubo = try d.createResource(.{ .buffer = .{ .size = 48, .usage = .{ .uniform = true } } });
            defer d.destroyResource(ubo);
            {
                const f = std.mem.bytesAsSlice(f32, try d.mapResource(ubo));
                // A slanted right triangle: the hypotenuse (bottom-right -> top-left) is a
                // diagonal edge that anti-aliases.
                const pos = [3][2]f32{ .{ -0.9, -0.9 }, .{ 0.9, -0.9 }, .{ -0.9, 0.9 } };
                inline for (pos, 0..) |v, k| {
                    f[k * 4 + 0] = v[0];
                    f[k * 4 + 1] = v[1];
                    f[k * 4 + 2] = 0;
                    f[k * 4 + 3] = 1;
                }
            }

            const ms = try d.createResource(.{ .image = .{ .width = W, .height = H, .format = .bgra8_unorm, .samples = samples, .usage = .{ .render_target = true } } });
            defer d.destroyResource(ms);
            const resolved = try d.createResource(.{ .image = .{ .width = W, .height = H, .format = .bgra8_unorm, .usage = .{ .render_target = true } } });
            defer d.destroyResource(resolved);

            const ctx = try d.createContext();
            defer ctx.deinit();
            const cb = try ctx.beginCommands();
            defer cb.deinit();
            try cb.setRenderTarget(ms);
            try cb.clear(.{ .r = 0, .g = 0, .b = 0, .a = 1 });
            try cb.bindPipeline(pipe);
            try cb.bindUniformBuffer(0, ubo);
            try cb.draw(3, 0);
            if (samples > 1) try cb.resolve(ms, resolved, W, H, .bgra8_unorm, samples);
            try ctx.submit(cb);

            var out: [W * H * 4]u8 = undefined;
            // For 1x there is no resolve. Read the single-sample target directly.
            const src = try d.mapResource(if (samples > 1) resolved else ms);
            @memcpy(&out, src[0 .. W * H * 4]);
            return out;
        }
    }.go;

    const px1 = renderAt(dev, vs, fs, 1) catch return error.SkipZigTest;
    const px4 = renderAt(dev, vs, fs, 4) catch return error.SkipZigTest;

    // Count partial-coverage RED edge pixels: bgra8 straight -> R is byte index 2. A partial
    // pixel has R strictly between the black background (0) and the solid triangle (255). The
    // 2x2-supersampled + resolved render MUST produce these along the diagonal. The 1x render
    // has hard edges (R is 0 or 255 only). That gap IS the anti-aliasing.
    var partial1: usize = 0;
    var partial4: usize = 0;
    var p: usize = 0;
    while (p < W * H) : (p += 1) {
        const r1 = px1[p * 4 + 2];
        const r4 = px4[p * 4 + 2];
        if (r1 > 30 and r1 < 225) partial1 += 1;
        if (r4 > 30 and r4 < 225) partial4 += 1;
    }
    try std.testing.expect(partial4 > 8);
    try std.testing.expect(partial4 > partial1 + 8);
}

test "nvidia MSAA: resolve in a SEPARATE submit (the EGL swapBuffers path) works on the GPU (skips without a GPU)" {
    // The window/pbuffer MSAA path renders into the multisampled backbuffer in the draw
    // submits, then eglSwapBuffers issues a RESOLVE-ONLY command buffer (no render target).
    // This exercises that resolve-only branch on real hardware (distinct from the same-cb
    // resolve the HAL/ICD MSAA oracle uses).
    const gpa = std.testing.allocator;
    const vspirv = @import("vulcan-spirv");
    const opc = vspirv.opcodes;
    const sc = opc.StorageClass;

    var vsb = try vspirv.binary.Builder.init(gpa, 40);
    defer vsb.deinit(gpa);
    try vsb.emit(gpa, opc.EntryPoint, &.{ opc.ExecutionModel.vertex, 28, 0, 19, 23 });
    try vsb.emit(gpa, opc.Decorate, &.{ 11, opc.Decoration.block });
    try vsb.emit(gpa, opc.MemberDecorate, &.{ 11, 0, opc.Decoration.offset, 0 });
    try vsb.emit(gpa, opc.Decorate, &.{ 10, opc.Decoration.array_stride, 16 });
    try vsb.emit(gpa, opc.Decorate, &.{ 16, opc.Decoration.binding, 0 });
    try vsb.emit(gpa, opc.Decorate, &.{ 19, opc.Decoration.builtin, opc.BuiltIn.vertex_index });
    try vsb.emit(gpa, opc.Decorate, &.{ 21, opc.Decoration.block });
    try vsb.emit(gpa, opc.MemberDecorate, &.{ 21, 0, opc.Decoration.builtin, opc.BuiltIn.position });
    try vsb.emit(gpa, opc.TypeVoid, &.{1});
    try vsb.emit(gpa, opc.TypeFloat, &.{ 2, 32 });
    try vsb.emit(gpa, opc.TypeVector, &.{ 3, 2, 4 });
    try vsb.emit(gpa, opc.TypeInt, &.{ 5, 32, 1 });
    try vsb.emit(gpa, opc.Constant, &.{ 5, 6, 0 });
    try vsb.emit(gpa, opc.TypeInt, &.{ 8, 32, 0 });
    try vsb.emit(gpa, opc.Constant, &.{ 8, 9, 3 });
    try vsb.emit(gpa, opc.TypeArray, &.{ 10, 3, 9 });
    try vsb.emit(gpa, opc.TypeStruct, &.{ 11, 10 });
    try vsb.emit(gpa, opc.TypePointer, &.{ 15, sc.uniform, 11 });
    try vsb.emit(gpa, opc.Variable, &.{ 15, 16, sc.uniform });
    try vsb.emit(gpa, opc.TypePointer, &.{ 17, sc.uniform, 3 });
    try vsb.emit(gpa, opc.TypePointer, &.{ 18, sc.input, 5 });
    try vsb.emit(gpa, opc.Variable, &.{ 18, 19, sc.input });
    try vsb.emit(gpa, opc.TypeStruct, &.{ 21, 3 });
    try vsb.emit(gpa, opc.TypePointer, &.{ 22, sc.output, 21 });
    try vsb.emit(gpa, opc.Variable, &.{ 22, 23, sc.output });
    try vsb.emit(gpa, opc.TypePointer, &.{ 26, sc.output, 3 });
    try vsb.emit(gpa, opc.TypeFunction, &.{ 27, 1 });
    try vsb.emit(gpa, opc.Function, &.{ 1, 28, 0, 27 });
    try vsb.emit(gpa, opc.Label, &.{29});
    try vsb.emit(gpa, opc.Load, &.{ 5, 30, 19 });
    try vsb.emit(gpa, opc.AccessChain, &.{ 17, 31, 16, 6, 30 });
    try vsb.emit(gpa, opc.Load, &.{ 3, 32, 31 });
    try vsb.emit(gpa, opc.AccessChain, &.{ 26, 37, 23, 6 });
    try vsb.emit(gpa, opc.Store, &.{ 37, 32 });
    try vsb.emit(gpa, opc.Return, &.{});
    try vsb.emit(gpa, opc.FunctionEnd, &.{});

    const glsl = @import("../../glsl.zig");
    const fs_src =
        \\precision mediump float;
        \\void main() { gl_FragColor = vec4(1.0, 0.0, 0.0, 1.0); }
    ;
    const dev = NvDevice.create(gpa) catch return error.SkipZigTest;
    defer dev.deinit();
    const fs_bytes = try glsl.compileForStage(gpa, fs_src, .fragment);
    defer gpa.free(fs_bytes);
    const vs = try dev.createShaderModule(.{ .stage = .vertex, .code = std.mem.sliceAsBytes(vsb.words.items) });
    defer dev.destroyShaderModule(vs);
    const fs = try dev.createShaderModule(.{ .stage = .fragment, .code = fs_bytes });
    defer dev.destroyShaderModule(fs);

    const W: u32 = 128;
    const H: u32 = 128;
    const pipe = try dev.createPipeline(.{
        .vertex = vs,
        .fragment = fs,
        .vertex_layout = .{ .stride = 0, .attributes = &.{} },
        .color_format = .bgra8_unorm,
    });
    defer dev.destroyPipeline(pipe);

    const ubo = try dev.createResource(.{ .buffer = .{ .size = 48, .usage = .{ .uniform = true } } });
    defer dev.destroyResource(ubo);
    {
        const f = std.mem.bytesAsSlice(f32, try dev.mapResource(ubo));
        const pos = [3][2]f32{ .{ -0.9, -0.9 }, .{ 0.9, -0.9 }, .{ -0.9, 0.9 } };
        inline for (pos, 0..) |v, k| {
            f[k * 4 + 0] = v[0];
            f[k * 4 + 1] = v[1];
            f[k * 4 + 2] = 0;
            f[k * 4 + 3] = 1;
        }
    }

    const ms = try dev.createResource(.{ .image = .{ .width = W, .height = H, .format = .bgra8_unorm, .samples = 4, .usage = .{ .render_target = true } } });
    defer dev.destroyResource(ms);
    const resolved = try dev.createResource(.{ .image = .{ .width = W, .height = H, .format = .bgra8_unorm, .usage = .{ .render_target = true } } });
    defer dev.destroyResource(resolved);

    const ctx = try dev.createContext();
    defer ctx.deinit();

    // Submit 1: DRAW into the multisampled backbuffer (no resolve).
    {
        const cb = try ctx.beginCommands();
        defer cb.deinit();
        try cb.setRenderTarget(ms);
        try cb.clear(.{ .r = 0, .g = 0, .b = 0, .a = 1 });
        try cb.bindPipeline(pipe);
        try cb.bindUniformBuffer(0, ubo);
        try cb.draw(3, 0);
        ctx.submit(cb) catch return error.SkipZigTest;
    }
    // Submit 2: RESOLVE-ONLY (the swapBuffers path) - no render target bound.
    {
        const cb = try ctx.beginCommands();
        defer cb.deinit();
        try cb.resolve(ms, resolved, W, H, .bgra8_unorm, 4);
        ctx.submit(cb) catch return error.SkipZigTest;
    }

    const fb = std.mem.bytesAsSlice(u32, try dev.mapResource(resolved));
    var partial: usize = 0;
    var i: usize = 0;
    while (i < W * H) : (i += 1) {
        const r = @as(u8, @truncate(fb[i] >> 16)); // bgra8: R at byte 2
        if (r > 30 and r < 225) partial += 1;
    }
    try std.testing.expect(partial > 8); // the resolved diagonal edge is anti-aliased
}

test "nvidia line primitive: GL_LINES draws a thin band via the hardware line rasterizer (skips without a GPU)" {
    const gpa = std.testing.allocator;
    const glsl = @import("../../glsl.zig");
    const dev = NvDevice.create(gpa) catch return error.SkipZigTest;
    defer dev.deinit();
    const W: u32 = 64;
    const H: u32 = 64;

    // Attribute VS (pos vec2) + constant-red FS.
    const vs_bytes = try glsl.compileForStage(gpa,
        \\attribute vec2 position;
        \\void main() { gl_Position = vec4(position, 0.0, 1.0); }
    , .vertex);
    defer gpa.free(vs_bytes);
    const fs_bytes = try glsl.compileForStage(gpa,
        \\precision mediump float;
        \\void main() { gl_FragColor = vec4(1.0, 0.0, 0.0, 1.0); }
    , .fragment);
    defer gpa.free(fs_bytes);
    const vs = try dev.createShaderModule(.{ .stage = .vertex, .code = vs_bytes });
    defer dev.destroyShaderModule(vs);
    const fs = try dev.createShaderModule(.{ .stage = .fragment, .code = fs_bytes });
    defer dev.destroyShaderModule(fs);

    const rt = try dev.createResource(.{ .image = .{ .width = W, .height = H, .format = .bgra8_unorm, .usage = .{ .render_target = true } } });
    defer dev.destroyResource(rt);

    // Two endpoints of a horizontal line across the middle.
    const verts = [_]f32{ -0.9, 0.0, 0.9, 0.0 };
    const vbuf = try dev.createResource(.{ .buffer = .{ .size = @sizeOf(@TypeOf(verts)), .usage = .{ .vertex = true } } });
    defer dev.destroyResource(vbuf);
    @memcpy(std.mem.bytesAsSlice(f32, try dev.mapResource(vbuf))[0..verts.len], &verts);

    const attrs = [_]hal.VertexAttribute{.{ .location = 0, .format = .r32g32_float, .offset = 0 }};
    const pipe = try dev.createPipeline(.{
        .vertex = vs,
        .fragment = fs,
        .vertex_layout = .{ .stride = 8, .attributes = &attrs },
        .color_format = .bgra8_unorm,
        .topology = .line_list, // the hardware line rasterizer
    });
    defer dev.destroyPipeline(pipe);

    const ctx = try dev.createContext();
    defer ctx.deinit();
    const cb = try ctx.beginCommands();
    defer cb.deinit();
    try cb.setRenderTarget(rt);
    try cb.clear(.{ .r = 0, .g = 0, .b = 0, .a = 1 });
    try cb.bindPipeline(pipe);
    try cb.bindVertexBuffer(vbuf);
    try cb.draw(2, 0); // GL_LINES: 2 vertices = 1 segment
    ctx.submit(cb) catch return error.SkipZigTest;

    const fb = std.mem.bytesAsSlice(u32, try dev.mapResource(rt));
    var red: usize = 0;
    var i: usize = 0;
    while (i < W * H) : (i += 1) {
        if (@as(u8, @truncate(fb[i] >> 16)) > 200 and @as(u8, @truncate(fb[i] >> 8)) < 60) red += 1;
    }
    // A hardware ~1px horizontal line across ~90% width: tens of pixels, not a fill, not blank.
    try std.testing.expect(red > 15 and red < 400);
    // Row 4 (far from the mid-line) is background black.
    var top_red = false;
    var x: usize = 0;
    while (x < W) : (x += 1) {
        if (@as(u8, @truncate(fb[4 * W + x] >> 16)) > 200) top_red = true;
    }
    try std.testing.expect(!top_red);
}

test "nvidia gl_PointSize: a larger point renders on the GPU (SPH OMAP_POINT_SIZE + hardware point rasterizer) (skips without a GPU)" {
    const gpa = std.testing.allocator;
    const glsl = @import("../../glsl.zig");
    const dev = NvDevice.create(gpa) catch return error.SkipZigTest;
    defer dev.deinit();
    const W: u32 = 64;
    const H: u32 = 64;

    // Render one GL_POINTS point at the center with gl_PointSize = `sz` (hardcoded per shader);
    // return the red pixel count.
    const pointRed = struct {
        fn go(d: anytype, comptime sz: []const u8) !usize {
            const vs_bytes = try glsl.compileForStage(gpa, "attribute vec2 position;\nvoid main() { gl_Position = vec4(position, 0.0, 1.0); gl_PointSize = " ++ sz ++ "; }", .vertex);
            defer gpa.free(vs_bytes);
            const fs_bytes = try glsl.compileForStage(gpa,
                \\precision mediump float;
                \\void main() { gl_FragColor = vec4(1.0, 0.0, 0.0, 1.0); }
            , .fragment);
            defer gpa.free(fs_bytes);
            const vs = try d.createShaderModule(.{ .stage = .vertex, .code = vs_bytes });
            defer d.destroyShaderModule(vs);
            const fs = try d.createShaderModule(.{ .stage = .fragment, .code = fs_bytes });
            defer d.destroyShaderModule(fs);
            const rt = try d.createResource(.{ .image = .{ .width = W, .height = H, .format = .bgra8_unorm, .usage = .{ .render_target = true } } });
            defer d.destroyResource(rt);
            const verts = [_]f32{ 0.0, 0.0 };
            const vbuf = try d.createResource(.{ .buffer = .{ .size = @sizeOf(@TypeOf(verts)), .usage = .{ .vertex = true } } });
            defer d.destroyResource(vbuf);
            @memcpy(std.mem.bytesAsSlice(f32, try d.mapResource(vbuf))[0..verts.len], &verts);
            const attrs = [_]hal.VertexAttribute{.{ .location = 0, .format = .r32g32_float, .offset = 0 }};
            const pipe = try d.createPipeline(.{
                .vertex = vs,
                .fragment = fs,
                .vertex_layout = .{ .stride = 8, .attributes = &attrs },
                .color_format = .bgra8_unorm,
                .topology = .point_list,
            });
            defer d.destroyPipeline(pipe);
            const ctx = try d.createContext();
            defer ctx.deinit();
            const cb = try ctx.beginCommands();
            defer cb.deinit();
            try cb.setRenderTarget(rt);
            try cb.clear(.{ .r = 0, .g = 0, .b = 0, .a = 1 });
            try cb.bindPipeline(pipe);
            try cb.bindVertexBuffer(vbuf);
            try cb.draw(1, 0); // one point
            ctx.submit(cb) catch return error.SkipZigTest;
            const fb = std.mem.bytesAsSlice(u32, try d.mapResource(rt));
            var red: usize = 0;
            var i: usize = 0;
            while (i < W * H) : (i += 1) {
                if (@as(u8, @truncate(fb[i] >> 16)) > 200 and @as(u8, @truncate(fb[i] >> 8)) < 60) red += 1;
            }
            return red;
        }
    }.go;

    const small = try pointRed(dev, "2.0");
    const large = try pointRed(dev, "8.0");
    try std.testing.expect(small > 0); // the point drew
    try std.testing.expect(large > small + 20); // gl_PointSize scaled it up on the GPU
}

test "ORACLE-POINTCOORD: gl_PointCoord matches software on the NVIDIA GPU (point sprite s/t gradient) (skips without a GPU)" {
    const gpa = std.testing.allocator;
    const glsl = @import("../../glsl.zig");
    const nv = NvDevice.create(gpa) catch return error.SkipZigTest;
    defer nv.deinit();
    const sw = @import("../software.zig");
    const sw_dev = try sw.driver.createDevice(gpa);
    defer sw_dev.deinit();
    const W: u32 = 64;
    const H: u32 = 64;

    // One GL_POINTS sprite of size 32 at the center: the FS writes gl_PointCoord into R (s)
    // and G (t). The sprite spans ~(16,16)..(48,48) with origin TOP-LEFT, so s runs 0..1
    // left->right and t runs 0..1 top->bottom. Render on BOTH software and the GPU. the two
    // must agree, proving the HW delivers gl_PointCoord (IPA of NAK_ATTR_POINT_SPRITE + the
    // SPH imap + ORIGIN_TOP) exactly like the software raster.
    const renderSprite = struct {
        fn go(d: hal.Device) ![]u32 {
            const vs_bytes = try glsl.compileForStage(gpa, "attribute vec2 position;\nvoid main() { gl_Position = vec4(position, 0.0, 1.0); gl_PointSize = 32.0; }", .vertex);
            defer gpa.free(vs_bytes);
            const fs_bytes = try glsl.compileForStage(gpa,
                \\precision mediump float;
                \\void main() { gl_FragColor = vec4(gl_PointCoord.x, gl_PointCoord.y, 1.0, 1.0); }
            , .fragment);
            defer gpa.free(fs_bytes);
            const vs = try d.createShaderModule(.{ .stage = .vertex, .code = vs_bytes });
            defer d.destroyShaderModule(vs);
            const fs = try d.createShaderModule(.{ .stage = .fragment, .code = fs_bytes });
            defer d.destroyShaderModule(fs);
            const rt = try d.createResource(.{ .image = .{ .width = W, .height = H, .format = .bgra8_unorm, .usage = .{ .render_target = true } } });
            defer d.destroyResource(rt);
            const verts = [_]f32{ 0.0, 0.0 };
            const vbuf = try d.createResource(.{ .buffer = .{ .size = @sizeOf(@TypeOf(verts)), .usage = .{ .vertex = true } } });
            defer d.destroyResource(vbuf);
            @memcpy(std.mem.bytesAsSlice(f32, try d.mapResource(vbuf))[0..verts.len], &verts);
            const attrs = [_]hal.VertexAttribute{.{ .location = 0, .format = .r32g32_float, .offset = 0 }};
            const pipe = try d.createPipeline(.{
                .vertex = vs,
                .fragment = fs,
                .vertex_layout = .{ .stride = 8, .attributes = &attrs },
                .color_format = .bgra8_unorm,
                .topology = .point_list,
            });
            defer d.destroyPipeline(pipe);
            const ctx = try d.createContext();
            defer ctx.deinit();
            const cb = try ctx.beginCommands();
            defer cb.deinit();
            try cb.setRenderTarget(rt);
            try cb.clear(.{ .r = 0, .g = 0, .b = 0, .a = 1 });
            try cb.bindPipeline(pipe);
            try cb.bindVertexBuffer(vbuf);
            try cb.draw(1, 0);
            ctx.submit(cb) catch return error.SkipZigTest;
            const fb = std.mem.bytesAsSlice(u32, try d.mapResource(rt));
            const out = try gpa.alloc(u32, W * H);
            @memcpy(out, fb[0 .. W * H]);
            return out;
        }
    }.go;

    const nv_fb = try renderSprite(nv);
    defer gpa.free(nv_fb);
    const sw_fb = try renderSprite(sw_dev);
    defer gpa.free(sw_fb);

    // Sample four inset corners of the sprite and read R (s) / G (t). Origin TOP-LEFT means:
    // top-left = low s, low t; top-right = high s, low t; bottom-left = low s, high t;
    // bottom-right = high s, high t. Both drivers must agree on each corner within tolerance.
    const pts = [_][2]u32{ .{ 20, 20 }, .{ 44, 20 }, .{ 20, 44 }, .{ 44, 44 } };
    const expect = [_][2]u8{ .{ 0, 0 }, .{ 1, 0 }, .{ 0, 1 }, .{ 1, 1 } }; // (s hi?, t hi?)
    for (pts, expect) |pt, ex| {
        const idx = pt[1] * W + pt[0];
        const nv_p = nv_fb[idx];
        const sw_p = sw_fb[idx];
        const nv_s: u8 = @truncate(nv_p >> 16);
        const nv_t: u8 = @truncate(nv_p >> 8);
        const sw_s: u8 = @truncate(sw_p >> 16);
        const sw_t: u8 = @truncate(sw_p >> 8);
        // GPU agrees with software (both in the same 0..1 sprite space; tolerance for the
        // half-texel pixel-center offset between the two rasterizers).
        try std.testing.expect(@abs(@as(i32, nv_s) - @as(i32, sw_s)) < 40);
        try std.testing.expect(@abs(@as(i32, nv_t) - @as(i32, sw_t)) < 40);
        // And the coord has the right orientation (low corner < 128 < high corner).
        if (ex[0] == 0) try std.testing.expect(nv_s < 128) else try std.testing.expect(nv_s > 128);
        if (ex[1] == 0) try std.testing.expect(nv_t < 128) else try std.testing.expect(nv_t > 128);
    }
}

test "nvidia gl_VertexID: a vertex-buffer-less full-screen triangle renders on the GPU (skips without a GPU)" {
    const gpa = std.testing.allocator;
    const glsl = @import("../../glsl.zig");
    const dev = NvDevice.create(gpa) catch return error.SkipZigTest;
    defer dev.deinit();
    const W: u32 = 64;
    const H: u32 = 64;
    // The canonical full-screen triangle from gl_VertexID: no vertex buffer, no attributes. The
    // VS derives clip-space position from the DA-delivered vertex id (ALD a[0x2fc]).
    const vs_bytes = try glsl.compileForStage(gpa,
        \\#version 300 es
        \\void main() {
        \\  float x = (gl_VertexID == 1) ? 3.0 : -1.0;
        \\  float y = (gl_VertexID == 2) ? 3.0 : -1.0;
        \\  gl_Position = vec4(x, y, 0.0, 1.0);
        \\}
    , .vertex);
    defer gpa.free(vs_bytes);
    const fs_bytes = try glsl.compileForStage(gpa,
        \\#version 300 es
        \\precision mediump float;
        \\out vec4 frag;
        \\void main() { frag = vec4(0.0, 1.0, 0.0, 1.0); }
    , .fragment);
    defer gpa.free(fs_bytes);
    const vs = try dev.createShaderModule(.{ .stage = .vertex, .code = vs_bytes });
    defer dev.destroyShaderModule(vs);
    const fs = try dev.createShaderModule(.{ .stage = .fragment, .code = fs_bytes });
    defer dev.destroyShaderModule(fs);
    const rt = try dev.createResource(.{ .image = .{ .width = W, .height = H, .format = .bgra8_unorm, .usage = .{ .render_target = true } } });
    defer dev.destroyResource(rt);
    // Empty vertex layout (no attributes) + a 1-byte dummy vertex buffer (unused): the VS reads
    // only gl_VertexID, so the DA needs no vertex stream.
    const dummy = try dev.createResource(.{ .buffer = .{ .size = 1, .usage = .{ .vertex = true } } });
    defer dev.destroyResource(dummy);
    const pipe = try dev.createPipeline(.{
        .vertex = vs,
        .fragment = fs,
        .vertex_layout = .{ .stride = 0, .attributes = &.{} },
        .color_format = .bgra8_unorm,
    });
    defer dev.destroyPipeline(pipe);
    const ctx = try dev.createContext();
    defer ctx.deinit();
    const cb = try ctx.beginCommands();
    defer cb.deinit();
    try cb.setRenderTarget(rt);
    try cb.clear(.{ .r = 0, .g = 0, .b = 0, .a = 1 });
    try cb.bindPipeline(pipe);
    try cb.bindVertexBuffer(dummy);
    try cb.draw(3, 0);
    ctx.submit(cb) catch return error.SkipZigTest;
    const fb = std.mem.bytesAsSlice(u32, try dev.mapResource(rt));
    const c = fb[(H / 2) * W + W / 2];
    const r: u8 = @truncate(c >> 16);
    const g: u8 = @truncate(c >> 8);
    const b: u8 = @truncate(c);
    try std.testing.expect(g > 200 and r < 60 and b < 60); // center green: the procedural triangle covered it
}

test "nvidia depth is PRESERVED across separate submits within a frame (skips without a GPU)" {
    // The GLES clear-once-then-draw-many contract: glClear(GL_DEPTH_BUFFER_BIT) clears the
    // depth buffer ONCE per frame, then each glDrawArrays/glDrawElements is its OWN HAL
    // submit that re-binds the SAME depth attachment WITHOUT re-clearing it (clear_value =
    // null). Earlier, setDepthTarget always cleared (a fixed 1.0), so every draw wiped the
    // accumulated depth and multi-primitive occlusion was wrong (glmark2/es2gears black/garbled).
    // This locks the fix: NEAR green is drawn FIRST (submit 1, depth cleared to 1.0), then FAR
    // red is drawn SECOND (submit 2, NO clear). The far red MUST be rejected at the center where
    // the near green already wrote a smaller depth - so the center stays GREEN.
    const sass = nvidia.sass;
    const gpa = std.testing.allocator;
    const dev = NvDevice.create(gpa) catch return error.SkipZigTest;
    defer dev.deinit();
    const W: u32 = 256;
    const H: u32 = 256;

    var vs_code: [256]u32 = undefined;
    var vsa = sass.Assembler{ .code = &vs_code };
    {
        var p: u32 = 0;
        while (p < 6) : (p += 1) vsa.movImm(12, p, .{});
    }
    vsa.ald(4, sass.ATTR_GENERIC0, 4, .{ .wr_barrier = 0 });
    vsa.ald(8, sass.ATTR_GENERIC0 + 0x10, 4, .{ .wr_barrier = 1 });
    vsa.ast(sass.ATTR_POSITION, 4, 4, .{ .wait_mask = 1 });
    vsa.ast(sass.ATTR_GENERIC0, 8, 4, .{ .wait_mask = 2 });
    vsa.exit(.{ .stall = 1 });
    var ps_code: [256]u32 = undefined;
    var psa = sass.Assembler{ .code = &ps_code };
    {
        var p: u32 = 0;
        while (p < 6) : (p += 1) psa.movImm(15, p, .{});
    }
    psa.ipa(4, sass.ATTR_GENERIC0, .{ .wr_barrier = 0 });
    psa.ipa(5, sass.ATTR_GENERIC0 + 4, .{ .wr_barrier = 1 });
    psa.ipa(6, sass.ATTR_GENERIC0 + 8, .{ .wr_barrier = 2 });
    psa.ipa(7, sass.ATTR_GENERIC0 + 12, .{ .wr_barrier = 3 });
    psa.movReg(0, 4, .{ .wait_mask = 0xf });
    psa.movReg(1, 5, .{});
    psa.movReg(2, 6, .{});
    psa.movReg(3, 7, .{});
    psa.exit(.{ .stall = 15 });

    const vs = try dev.createShaderModule(.{ .stage = .vertex, .code = std.mem.sliceAsBytes(vs_code[0..vsa.dwords()]) });
    defer dev.destroyShaderModule(vs);
    const ps = try dev.createShaderModule(.{ .stage = .fragment, .code = std.mem.sliceAsBytes(ps_code[0..psa.dwords()]) });
    defer dev.destroyShaderModule(ps);
    const attrs = [_]hal.VertexAttribute{
        .{ .location = 0, .format = .rgba8_unorm, .offset = 0 },
        .{ .location = 1, .format = .rgba8_unorm, .offset = 16 },
    };
    const pipe = try dev.createPipeline(.{
        .vertex = vs,
        .fragment = ps,
        .vertex_layout = .{ .stride = 32, .attributes = &attrs },
        .color_format = .bgra8_unorm,
        .depth = .{ .test_enable = true, .write_enable = true, .compare_op = .less },
    });
    defer dev.destroyPipeline(pipe);

    const near_verts = [_]f32{
        0.0,  0.6,  0.3, 1.0, 0, 1, 0, 1,
        -0.6, -0.6, 0.3, 1.0, 0, 1, 0, 1,
        0.6,  -0.6, 0.3, 1.0, 0, 1, 0, 1,
    };
    const far_verts = [_]f32{
        0.0,  0.6,  0.7, 1.0, 1, 0, 0, 1,
        -0.6, -0.6, 0.7, 1.0, 1, 0, 0, 1,
        0.6,  -0.6, 0.7, 1.0, 1, 0, 0, 1,
    };
    const near_vb = try dev.createResource(.{ .buffer = .{ .size = near_verts.len * 4, .usage = .{ .vertex = true } } });
    defer dev.destroyResource(near_vb);
    const far_vb = try dev.createResource(.{ .buffer = .{ .size = far_verts.len * 4, .usage = .{ .vertex = true } } });
    defer dev.destroyResource(far_vb);
    @memcpy(std.mem.bytesAsSlice(f32, try dev.mapResource(near_vb))[0..near_verts.len], &near_verts);
    @memcpy(std.mem.bytesAsSlice(f32, try dev.mapResource(far_vb))[0..far_verts.len], &far_verts);

    const rt = try dev.createResource(.{ .image = .{ .width = W, .height = H, .format = .bgra8_unorm, .usage = .{ .render_target = true } } });
    defer dev.destroyResource(rt);
    const depth = try dev.createResource(.{ .image = .{ .width = W, .height = H, .format = .depth32_float, .usage = .{} } });
    defer dev.destroyResource(depth);

    const ctx = try dev.createContext();
    defer ctx.deinit();

    // Submit 1: clear the depth buffer (1.0) + draw the NEAR green triangle.
    {
        const cb = ctx.beginCommands() catch return error.SkipZigTest;
        defer cb.deinit();
        try cb.setRenderTarget(rt);
        try cb.clear(.{ .r = 0.125, .g = 0.125, .b = 0.125, .a = 1 });
        try cb.setDepthTarget(depth, 1.0); // clear depth ONCE
        try cb.bindPipeline(pipe);
        try cb.bindVertexBuffer(near_vb);
        try cb.draw(3, 0);
        ctx.submit(cb) catch return error.SkipZigTest;
    }
    // Submit 2: NO depth clear (preserve) + draw the FAR red triangle. It must be rejected
    // at the center where the near green already wrote depth 0.3 (0.7 < 0.3 is false).
    {
        const cb = ctx.beginCommands() catch return error.SkipZigTest;
        defer cb.deinit();
        try cb.setRenderTarget(rt);
        try cb.setDepthTarget(depth, null); // PRESERVE the accumulated depth
        try cb.bindPipeline(pipe);
        try cb.bindVertexBuffer(far_vb);
        try cb.draw(3, 0);
        ctx.submit(cb) catch return error.SkipZigTest;
    }

    const fb = std.mem.bytesAsSlice(u32, try dev.mapResource(rt));
    const center = fb[(H / 2) * W + W / 2]; // bgra8 RT: u32 = 0xAARRGGBB
    const r: u8 = @truncate(center >> 16);
    const g: u8 = @truncate(center >> 8);
    const b: u8 = @truncate(center);
    // GREEN survives = the far red was depth-rejected = depth was PRESERVED across submits.
    try std.testing.expect(g > 200 and r < 80 and b < 80);
}

test "nvidia presents a rendered framebuffer to a surface (skips without a GPU)" {
    const gpa = std.testing.allocator;
    const dev = NvDevice.create(gpa) catch return error.SkipZigTest;
    defer dev.deinit();
    const W: u32 = 64;
    const H: u32 = 64;

    // Render: clear a framebuffer green on the GPU.
    const rt = try dev.createResource(.{ .image = .{ .width = W, .height = H, .format = .bgra8_unorm, .usage = .{ .render_target = true } } });
    defer dev.destroyResource(rt);
    const ctx = try dev.createContext();
    defer ctx.deinit();
    const cb = try ctx.beginCommands();
    defer cb.deinit();
    try cb.setRenderTarget(rt);
    try cb.clear(.{ .r = 0, .g = 1, .b = 0, .a = 1 });
    try ctx.submit(cb);

    // Present it onto a (display-free) headless surface, then read that surface's
    // buffer back: it must now hold the rendered frame.
    const display = try platform.headless.create(gpa);
    defer display.deinit();
    var psurf = try display.createSurface(.{ .width = W, .height = H, .format = .bgra8_unorm });
    defer psurf.deinit();
    const surf = try dev.createSurface(@ptrCast(&psurf));
    defer dev.destroySurface(surf);
    try ctx.present(surf, rt);

    const px = std.mem.bytesAsSlice(u32, (try psurf.currentBuffer()).bytes);
    try std.testing.expectEqual(@as(u32, 0xff00ff00), px[(H / 2) * W + W / 2]); // green A8R8G8B8
    try std.testing.expectEqual(@as(u32, 0xff00ff00), px[0]);
}
