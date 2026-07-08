const std = @import("std");
const hal = @import("../../hal.zig");
const Resource = @import("resource.zig").Resource;
const Pipeline = @import("pipeline.zig").Pipeline;
const raster = @import("raster.zig");
const spirv_jit = @import("spirv_jit.zig");
const sampler = @import("sampler.zig");
const GfxOut = spirv_jit.GfxOut;

const Command = union(enum) {
    set_render_target: *Resource,
    set_color_target: struct { index: u32, resource: *Resource }, // MRT color target (index 1+)
    set_depth_target: struct { depth: *Resource, clear_value: ?f32 },
    set_stencil_target: struct { stencil: *Resource, clear_value: ?u8 },
    clear: hal.Color,
    bind_pipeline: *Pipeline,
    bind_vertex_buffer: *Resource,
    // A record-time snapshot of the UBO bytes (not a live pointer): the GL layer reuses one UBO
    // Resource per program/stage and overwrites it before the next draw, so with deferred/batched
    // submits a live pointer would let a later draw retroactively change this draw's uniforms. Freed
    // by freeSnapshots (reset/deinit). Mirrors the nvidia driver's UBO snapshot.
    bind_uniform_buffer: struct { binding: u32, snapshot: []const u8 },
    bind_texture: hal.TextureBinding,
    draw: struct { vertex_count: u32, first_vertex: u32, instance_count: u32 = 1, first_instance: u32 = 0 },
    resolve: struct { src: *Resource, dst: *Resource, width: u32, height: u32, format: hal.Format, samples: u8 },
    set_scissor: ?hal.ScissorRect,
    set_viewport: ?hal.Viewport,
};

pub const CommandBuffer = struct {
    gpa: std.mem.Allocator,
    cmds: std.ArrayListUnmanaged(Command) = .empty,

    fn setRenderTarget(ptr: *anyopaque, target: *hal.Resource) hal.Error!void {
        const self: *CommandBuffer = @ptrCast(@alignCast(ptr));
        self.cmds.append(self.gpa, .{ .set_render_target = @ptrCast(@alignCast(target)) }) catch return error.OutOfMemory;
    }
    fn setDepthTarget(ptr: *anyopaque, depth: *hal.Resource, clear_value: ?f32) hal.Error!void {
        const self: *CommandBuffer = @ptrCast(@alignCast(ptr));
        self.cmds.append(self.gpa, .{ .set_depth_target = .{ .depth = @ptrCast(@alignCast(depth)), .clear_value = clear_value } }) catch return error.OutOfMemory;
    }
    fn setStencilTarget(ptr: *anyopaque, stencil: *hal.Resource, clear_value: ?u8) hal.Error!void {
        const self: *CommandBuffer = @ptrCast(@alignCast(ptr));
        self.cmds.append(self.gpa, .{ .set_stencil_target = .{ .stencil = @ptrCast(@alignCast(stencil)), .clear_value = clear_value } }) catch return error.OutOfMemory;
    }
    fn clear(ptr: *anyopaque, color: hal.Color) hal.Error!void {
        const self: *CommandBuffer = @ptrCast(@alignCast(ptr));
        self.cmds.append(self.gpa, .{ .clear = color }) catch return error.OutOfMemory;
    }
    fn bindPipeline(ptr: *anyopaque, pipeline: *hal.Pipeline) hal.Error!void {
        const self: *CommandBuffer = @ptrCast(@alignCast(ptr));
        self.cmds.append(self.gpa, .{ .bind_pipeline = @ptrCast(@alignCast(pipeline)) }) catch return error.OutOfMemory;
    }
    fn bindVertexBuffer(ptr: *anyopaque, buffer: *hal.Resource) hal.Error!void {
        const self: *CommandBuffer = @ptrCast(@alignCast(ptr));
        self.cmds.append(self.gpa, .{ .bind_vertex_buffer = @ptrCast(@alignCast(buffer)) }) catch return error.OutOfMemory;
    }
    fn bindUniformBuffer(ptr: *anyopaque, binding: u32, buffer: *hal.Resource) hal.Error!void {
        const self: *CommandBuffer = @ptrCast(@alignCast(ptr));
        const res: *Resource = @ptrCast(@alignCast(buffer));
        const snap = self.gpa.dupe(u8, res.bytes) catch return error.OutOfMemory;
        self.cmds.append(self.gpa, .{ .bind_uniform_buffer = .{ .binding = binding, .snapshot = snap } }) catch {
            self.gpa.free(snap);
            return error.OutOfMemory;
        };
    }
    fn setColorTarget(ptr: *anyopaque, index: u32, target: *hal.Resource) hal.Error!void {
        const self: *CommandBuffer = @ptrCast(@alignCast(ptr));
        self.cmds.append(self.gpa, .{ .set_color_target = .{ .index = index, .resource = @ptrCast(@alignCast(target)) } }) catch return error.OutOfMemory;
    }
    fn bindTexture(ptr: *anyopaque, binding: hal.TextureBinding) hal.Error!void {
        const self: *CommandBuffer = @ptrCast(@alignCast(ptr));
        self.cmds.append(self.gpa, .{ .bind_texture = binding }) catch return error.OutOfMemory;
    }
    fn draw(ptr: *anyopaque, vertex_count: u32, first_vertex: u32) hal.Error!void {
        const self: *CommandBuffer = @ptrCast(@alignCast(ptr));
        self.cmds.append(self.gpa, .{ .draw = .{ .vertex_count = vertex_count, .first_vertex = first_vertex } }) catch return error.OutOfMemory;
    }
    fn drawInstanced(ptr: *anyopaque, vertex_count: u32, instance_count: u32, first_vertex: u32, first_instance: u32) hal.Error!void {
        const self: *CommandBuffer = @ptrCast(@alignCast(ptr));
        self.cmds.append(self.gpa, .{ .draw = .{ .vertex_count = vertex_count, .first_vertex = first_vertex, .instance_count = instance_count, .first_instance = first_instance } }) catch return error.OutOfMemory;
    }
    fn resolve(ptr: *anyopaque, src: *hal.Resource, dst: *hal.Resource, width: u32, height: u32, format: hal.Format, samples: u8) hal.Error!void {
        const self: *CommandBuffer = @ptrCast(@alignCast(ptr));
        self.cmds.append(self.gpa, .{ .resolve = .{
            .src = @ptrCast(@alignCast(src)),
            .dst = @ptrCast(@alignCast(dst)),
            .width = width,
            .height = height,
            .format = format,
            .samples = samples,
        } }) catch return error.OutOfMemory;
    }
    fn setScissor(ptr: *anyopaque, rect: ?hal.ScissorRect) hal.Error!void {
        const self: *CommandBuffer = @ptrCast(@alignCast(ptr));
        self.cmds.append(self.gpa, .{ .set_scissor = rect }) catch return error.OutOfMemory;
    }
    fn setViewport(ptr: *anyopaque, vp: ?hal.Viewport) hal.Error!void {
        const self: *CommandBuffer = @ptrCast(@alignCast(ptr));
        self.cmds.append(self.gpa, .{ .set_viewport = vp }) catch return error.OutOfMemory;
    }
    /// Free the per-draw UBO snapshots dup'd in bindUniformBuffer (called by reset + deinit).
    fn freeSnapshots(self: *CommandBuffer) void {
        for (self.cmds.items) |c| switch (c) {
            .bind_uniform_buffer => |u| self.gpa.free(@constCast(u.snapshot)),
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
        .setDepthTarget = &setDepthTarget,
        .setStencilTarget = &setStencilTarget,
        .bindTexture = &bindTexture,
        .draw = &draw,
        .drawInstanced = &drawInstanced,
        .resolve = &resolve,
        .setScissor = &setScissor,
        .setViewport = &setViewport,
        .reset = &reset,
        .deinit = &deinit,
    };
};

fn readF32(bytes: []const u8, off: usize) f32 {
    return std.mem.bytesToValue(f32, bytes[off..][0..4]);
}

/// Gather a vertex's scalar VS inputs from the bound vertex buffer. Attributes are sorted
/// by ascending location to match the VS lowering's scalarized Input order. Returns the
/// scalar count written, or null if any read is out of bounds.
fn gatherVsInputs(p: *const Pipeline, vb: *const Resource, vertex_index: u32, out: []f32) ?usize {
    const attrs = p.layout.attributes;
    const stride = @as(usize, p.layout.stride);
    const base = @as(usize, vertex_index) * stride;

    // Attribute indices sorted by ascending location (small N: selection sort).
    var order: [raster.MAX_VS_INPUTS]usize = undefined;
    const na = @min(attrs.len, order.len);
    for (0..na) |i| order[i] = i;
    for (0..na) |i| {
        var min = i;
        for (i + 1..na) |j| if (attrs[order[j]].location < attrs[order[min]].location) {
            min = j;
        };
        const tmp = order[i];
        order[i] = order[min];
        order[min] = tmp;
    }

    var n: usize = 0;
    for (order[0..na]) |ai| {
        const attr = attrs[ai];
        const comps = attr.format.componentCount();
        const off = base + @as(usize, attr.offset);
        var c: u32 = 0;
        while (c < comps) : (c += 1) {
            if (n >= out.len) return n;
            const fo = off + @as(usize, c) * 4;
            if (fo + 4 > vb.bytes.len) return null;
            out[n] = readF32(vb.bytes, fo);
            n += 1;
        }
    }
    return n;
}

/// Run the VS for one vertex (`vertex_index`, `instance_index`) into `vout`, returning its
/// window-space screen position [x, y] and NDC z (clamped to [0,1] for the Vulkan depth
/// range), or null if the VS could not run (missing vertex buffer). Shared by the triangle,
/// line, and point paths so they transform vertices identically.
fn runVsVertex(t: *Resource, p: *Pipeline, prog: *const @import("pipeline.zig").ShaderProgram, vb: ?*Resource, pulling: bool, vertex_index: u32, instance_index: u32, vs_bufs: []const ?[*]const u8, vout: *[GfxOut.vertex_len]f32, viewport: ?hal.Viewport) ?[3]f32 {
    @memset(vout, 0);
    if (pulling) {
        const indices = [_]i32{ @intCast(vertex_index), @intCast(instance_index) };
        spirv_jit.runGraphicsPulling(&prog.vs, prog.vs_index_count, prog.vs_buffers, indices[0..prog.vs_index_count], vs_bufs[0..prog.vs_buffers], vout) catch return null;
    } else {
        var inputs = [_]f32{0} ** raster.MAX_VS_INPUTS;
        _ = gatherVsInputs(p, vb.?, vertex_index, &inputs) orelse return null;
        if (prog.vs_inputs > inputs.len) return null;
        spirv_jit.runGraphics(&prog.vs, prog.vs_inputs, prog.vs_buffers, &inputs, vs_bufs[0..prog.vs_buffers], vout) catch return null;
    }
    const w = if (vout[3] != 0) vout[3] else 1.0;
    // NDC -> window space (y down, no flip). glViewport maps NDC into
    // [vx, vx+vw] x [vy, vy+vh]. A null viewport is the full render target (byte-identical to
    // the pre-viewport formula). The GL bottom-left->top-left Y conversion is done by the GLES layer.
    var vx: f32 = 0;
    var vy: f32 = 0;
    var vw: f32 = @floatFromInt(t.width);
    var vh: f32 = @floatFromInt(t.height);
    var znear: f32 = 0.0;
    var zfar: f32 = 1.0;
    if (viewport) |vp| {
        vx = @floatFromInt(vp.x);
        vy = @floatFromInt(vp.y);
        vw = @floatFromInt(vp.width);
        vh = @floatFromInt(vp.height);
        znear = vp.depth_near;
        zfar = vp.depth_far;
    }
    // glDepthRangef: z_win = near + z_ndc*(far-near), clamped to the range (near/far may be inverted).
    const zw = znear + std.math.clamp(vout[2] / w, 0.0, 1.0) * (zfar - znear);
    return .{ vx + (vout[0] / w * 0.5 + 0.5) * vw, vy + (vout[1] / w * 0.5 + 0.5) * vh, std.math.clamp(zw, @min(znear, zfar), @max(znear, zfar)) };
}

fn writeF32(bytes: []u8, off: usize, v: f32) void {
    std.mem.bytesAsValue(f32, bytes[off..][0..4]).* = v;
}

/// Transform-feedback capture (GLES3): run the pipeline's vertex shader over `cap.vertex_count`
/// vertices (no rasterization) and write the selected output varyings, tightly interleaved as
/// f32 in `cap.specs` order, into `cap.output` starting at `cap.output_offset`. Returns the
/// number of bytes written. Reuses the exact per-vertex VS-run path the draw uses (attribute
/// gather or vertex pulling) so the captured varyings match what a normal draw would produce.
pub fn captureTransformFeedback(cap: hal.TransformFeedbackCapture) hal.Error!usize {
    const p: *Pipeline = @ptrCast(@alignCast(cap.pipeline));
    const prog: *const @import("pipeline.zig").ShaderProgram = if (p.program) |*pr| pr else return error.InvalidArgument;
    const out: *Resource = @ptrCast(@alignCast(cap.output));
    const vb: ?*Resource = if (cap.vertex_buffer) |b| @as(*Resource, @ptrCast(@alignCast(b))) else null;
    const pulling = prog.vs_index_count > 0;

    // Convert the binding-indexed UBO resources to raw byte pointers, then remap them to the VS's
    // entry-param order via vs_buffer_bindings (mirrors drawShaded's UBO wiring).
    var ubo_bytes: [spirv_jit.GfxBuffers.max]?[*]const u8 = .{null} ** spirv_jit.GfxBuffers.max;
    {
        var b: usize = 0;
        while (b < cap.ubos.len and b < ubo_bytes.len) : (b += 1) {
            if (cap.ubos[b]) |r| {
                const rr: *Resource = @ptrCast(@alignCast(r));
                ubo_bytes[b] = rr.bytes.ptr;
            }
        }
    }
    var vs_bufs: [spirv_jit.GfxBuffers.max]?[*]const u8 = .{null} ** spirv_jit.GfxBuffers.max;
    {
        var cursor: usize = 0;
        var k: usize = 0;
        while (k < prog.vs_buffers and k < vs_bufs.len) : (k += 1) {
            const bind = prog.vs_buffer_bindings[k];
            if (bind >= 0) {
                const bi: usize = @intCast(bind);
                vs_bufs[k] = if (bi < ubo_bytes.len) ubo_bytes[bi] else null;
            } else {
                vs_bufs[k] = if (cursor < ubo_bytes.len) ubo_bytes[cursor] else null;
                cursor += 1;
            }
        }
    }

    var write_off = cap.output_offset;
    var vout: [GfxOut.vertex_len]f32 = undefined;
    var v: u32 = 0;
    while (v < cap.vertex_count) : (v += 1) {
        @memset(&vout, 0);
        if (pulling) {
            const indices = [_]i32{ @intCast(cap.first_vertex + v), @intCast(cap.instance) };
            spirv_jit.runGraphicsPulling(&prog.vs, prog.vs_index_count, prog.vs_buffers, indices[0..prog.vs_index_count], vs_bufs[0..prog.vs_buffers], &vout) catch return error.InvalidArgument;
        } else {
            const vbb = vb orelse return error.InvalidArgument;
            var inputs = [_]f32{0} ** raster.MAX_VS_INPUTS;
            _ = gatherVsInputs(p, vbb, cap.first_vertex + v, &inputs) orelse return error.InvalidArgument;
            if (prog.vs_inputs > inputs.len) return error.InvalidArgument;
            spirv_jit.runGraphics(&prog.vs, prog.vs_inputs, prog.vs_buffers, &inputs, vs_bufs[0..prog.vs_buffers], &vout) catch return error.InvalidArgument;
        }
        // Write the captured varyings tightly interleaved (specs order, as f32).
        for (cap.specs) |sp| {
            var c: u8 = 0;
            while (c < sp.components) : (c += 1) {
                if (write_off + 4 > out.bytes.len) return write_off - cap.output_offset; // clamp at buffer end
                const comp = sp.first_component + c;
                const slot = GfxOut.varying_base + @as(usize, sp.location) * 4 + comp;
                const val = if (slot < vout.len) vout[slot] else 0;
                writeF32(out.bytes, write_off, val);
                write_off += 4;
            }
        }
    }
    return write_off - cap.output_offset;
}

/// The effective raster clip box = the scissor intersected with the viewport rect. glViewport clips
/// rasterization to the viewport (geometry beyond NDC [-1,1] maps outside the rect and must be
/// discarded). The scissor clips further. Reuses the software raster's scissor-box clamp. A null
/// viewport => just the scissor (full-RT viewport adds no clip).
fn effectiveScissor(scissor: ?hal.ScissorRect, viewport: ?hal.Viewport) ?hal.ScissorRect {
    const vp = viewport orelse return scissor;
    const vr = hal.ScissorRect{ .x = vp.x, .y = vp.y, .width = vp.width, .height = vp.height };
    const sc = scissor orelse return vr;
    const x0 = @max(sc.x, vr.x);
    const y0 = @max(sc.y, vr.y);
    const x1 = @min(sc.x +% @as(i32, @intCast(sc.width)), vr.x +% @as(i32, @intCast(vr.width)));
    const y1 = @min(sc.y +% @as(i32, @intCast(sc.height)), vr.y +% @as(i32, @intCast(vr.height)));
    if (x1 <= x0 or y1 <= y0) return .{ .x = 0, .y = 0, .width = 0, .height = 0 }; // empty intersection
    return .{ .x = x0, .y = y0, .width = @intCast(x1 - x0), .height = @intCast(y1 - y0) };
}

/// A zero descriptor for an unbound vertex sampler: sampleTexture reads width==0 and
/// returns transparent black instead of dereferencing the JIT's dummy pointer. File-scope so a
/// VS-buffer slot can point at it stably (a stack local would dangle past the builder's return).
const VS_EMPTY_TEX: sampler.TexDesc = .{ .pixels = undefined, .width = 0, .height = 0, .pitch = 0 };

/// Fill a vertex shader's pointer-param array in entry-ABI order per prog.vs_buffer_kinds, the VS
/// analogue of rasterShadedFmt's fs_bufs assembly. A `.descriptor` slot gets the UBO bound at its
/// tagged binding (else a declaration-order cursor). A `.sampler_desc` gets the bound texture (the
/// software driver's single-texture model, see execute's `tex`). A `.sampler_fn` gets the host
/// sampler entry point. A VS has no screen-space derivatives, so `.grad_buf` maps to null.
/// Vertex texture fetch (terrain displacement) rides this path.
fn buildVsBufs(prog: *const @import("pipeline.zig").ShaderProgram, ubos: []const ?[*]const u8, tex_ptr: ?*const sampler.TexDesc, out: *[spirv_jit.GfxBuffers.max]?[*]const u8) void {
    const total = @min(prog.vs_buffers, out.len);
    var ubo_cursor: usize = 0;
    var k: usize = 0;
    while (k < total) : (k += 1) {
        out[k] = switch (prog.vs_buffer_kinds[k]) {
            .descriptor => blk: {
                const bind = prog.vs_buffer_bindings[k];
                if (bind >= 0) {
                    const bi: usize = @intCast(bind);
                    break :blk if (bi < ubos.len) ubos[bi] else null;
                }
                const ptr: ?[*]const u8 = if (ubo_cursor < ubos.len) ubos[ubo_cursor] else null;
                ubo_cursor += 1;
                break :blk ptr;
            },
            .sampler_desc => if (tex_ptr) |t| @ptrCast(t) else @ptrCast(&VS_EMPTY_TEX),
            .sampler_fn => @ptrFromInt(@intFromPtr(&sampler.sampleTexture)),
            .sampler_cube_fn => @ptrFromInt(@intFromPtr(&sampler.sampleTextureCube)),
            .sampler_shadow_fn => @ptrFromInt(@intFromPtr(&sampler.sampleTextureShadow)),
            .sampler_cube_shadow_fn => @ptrFromInt(@intFromPtr(&sampler.sampleTextureCubeShadow)),
            .sampler_2darray_shadow_fn => @ptrFromInt(@intFromPtr(&sampler.sampleTexture2dArrayShadow)),
            .sampler_gather_fn => @ptrFromInt(@intFromPtr(&sampler.sampleTextureGather)),
            .sampler_fetch_fn => @ptrFromInt(@intFromPtr(&sampler.sampleTextureFetch)),
            .sampler_fetch3_fn => @ptrFromInt(@intFromPtr(&sampler.sampleTextureFetch3D)),
            .math_fn => @ptrFromInt(@intFromPtr(&sampler.mathFn)),
            .discard_fn => @ptrFromInt(@intFromPtr(&spirv_jit.discardThunk)),
            .grad_buf => null, // a vertex shader has no derivatives. no gradient buffer
        };
    }
}

fn drawShaded(t: *Resource, p: *Pipeline, vb: ?*Resource, first_vertex: u32, instance_index: u32, ubos: []const ?[*]const u8, depth_res: ?*Resource, stencil_res: ?*Resource, tex: ?*const sampler.TexDesc, scissor: ?hal.ScissorRect, viewport: ?hal.Viewport, extra_targets: []const raster.ColorTarget) void {
    const prog = &(p.program.?);
    if (t.width == 0 or t.height == 0) return;
    // A vertex-pulling VS (gl_VertexIndex, zero vertex attributes) needs no vertex buffer.
    // An attribute-fed VS does. Bail safely if the required buffer is absent.
    const pulling = prog.vs_index_count > 0;
    if (!pulling and vb == null) return;

    // `ubos` is binding-indexed (the draw fills ubos[binding]). The FS reads it by binding in
    // rasterShadedFmt. The VS receives its buffers in entry-param order, so remap each VS buffer
    // param to the UBO at its tagged binding (vs_buffer_bindings[k]). A param with no tag (-1)
    // falls back to declaration order via a running cursor (a hand-built VS without bindings).
    var ubo_ptrs: [spirv_jit.GfxBuffers.max]?[*]const u8 = .{null} ** spirv_jit.GfxBuffers.max;
    {
        var b: usize = 0;
        while (b < ubos.len and b < ubo_ptrs.len) : (b += 1) ubo_ptrs[b] = ubos[b];
    }
    var vs_bufs: [spirv_jit.GfxBuffers.max]?[*]const u8 = .{null} ** spirv_jit.GfxBuffers.max;
    buildVsBufs(prog, ubos, tex, &vs_bufs);

    // Per-vertex VS outputs: position(4) + varyings.
    var vouts: [3][GfxOut.vertex_len]f32 = undefined;
    var screen: [3][2]f32 = undefined; // screen x,y per vertex
    var screen_z: [3]f32 = undefined; // NDC z in [0,1] (Vulkan depth range) per vertex
    var i: u32 = 0;
    while (i < 3) : (i += 1) {
        // Vertex-pulling: gl_VertexIndex = first_vertex + i. gl_InstanceIndex = instance_index.
        const sv = runVsVertex(t, p, prog, vb, pulling, first_vertex + i, instance_index, vs_bufs[0..prog.vs_buffers], &vouts[i], viewport) orelse return;
        screen[i] = .{ sv[0], sv[1] };
        screen_z[i] = sv[2];
    }
    // The viewport also clips rasterization (fold it into the scissor box the raster clamps to).
    const clip = effectiveScissor(scissor, viewport);

    // Assemble the optional depth attachment: present only when a depth target is
    // bound and the pipeline enables the depth test. The depth Resource's bytes are
    // reinterpreted as an f32 depth buffer (one per pixel). When absent, the
    // rasterizer takes the color-only path (no depth test) unchanged.
    // Stencil forces the single-sample path (a UI clip/mask does not need MSAA, and the
    // rasterizer's stencil test lives only in the single-sample loop).
    const stencil_active = p.stencil.test_enable and stencil_res != null;
    // The MSAA sample count is driven by the target image's backing (width*height*samples
    // pixels), not the pipeline, so a draw into a multisampled render target (e.g. an
    // EGL_SAMPLES window backbuffer) is anti-aliased without the pipeline having to declare it,
    // matching how the nvidia driver keys supersampling off the target's sample count.
    // The pipeline's own p.samples still applies when it exceeds the target's (an explicitly
    // multisampled pipeline into a 1x target keeps working). Stencil forces single-sample.
    const target_samples: u8 = blk: {
        const bpp = t.format.bytesPerPixel();
        const per_pixel = @as(usize, t.width) * t.height * bpp;
        if (per_pixel == 0) break :blk 1;
        break :blk @intCast(@min(@as(usize, 8), t.bytes.len / per_pixel));
    };
    const eff_samples: u8 = @max(if (p.samples > 1) p.samples else 1, if (target_samples > 1) target_samples else 1);
    const samples: u8 = if (stencil_active) 1 else eff_samples;
    var depth_att: ?raster.DepthAttachment = null;
    if (depth_res) |dr| {
        // The depth buffer is w*h*samples f32s (one per color sample) when MSAA is on,
        // else w*h. Match the depth sample count to the pipeline's so a per-sample depth
        // test is possible.
        const depth_count = @as(usize, t.width) * t.height * samples;
        if (p.depth.test_enable and dr.bytes.len >= depth_count * 4) {
            const aligned: []align(@alignOf(f32)) u8 = @alignCast(dr.bytes[0 .. depth_count * 4]);
            const dbuf = std.mem.bytesAsSlice(f32, aligned);
            depth_att = .{ .buffer = dbuf, .state = p.depth, .screen_z = screen_z, .samples = samples };
        }
    }
    // Stencil attachment: present only when a stencil target is bound and the pipeline
    // enables the stencil test. One u8 per pixel (single-sample).
    // Triangle winding / facing (window space, y increasing downward as `screen` stores it):
    // signed_area < 0 == counter-clockwise. Map to the triangle's front/back facing per the
    // pipeline's front_face winding. Used for two-sided stencil (pick the front/back state) and
    // back-face culling below.
    const s_ax = screen[1][0] - screen[0][0];
    const s_ay = screen[1][1] - screen[0][1];
    const s_bx = screen[2][0] - screen[0][0];
    const s_by = screen[2][1] - screen[0][1];
    const signed_area = s_ax * s_by - s_ay * s_bx;
    const is_front = switch (p.cull.front_face) {
        .counter_clockwise => signed_area < 0,
        .clockwise => signed_area > 0,
    };

    var stencil_att: ?raster.StencilAttachment = null;
    if (stencil_res) |sr| {
        const sc = @as(usize, t.width) * t.height;
        if (p.stencil.test_enable and sr.bytes.len >= sc) {
            // Two-sided stencil: a back-facing triangle uses the back state (if set).
            // A front-facing one (or single-face pipeline) uses the front `stencil`.
            const face_state = if (!is_front) (p.stencil_back orelse p.stencil) else p.stencil;
            stencil_att = .{ .buffer = sr.bytes[0..sc], .state = face_state };
        }
    }

    // Back-face culling. Discard the face the pipeline culls so e.g. vkcube's unlit back
    // faces don't overdraw its lit front faces.
    if (p.cull.mode != .none) {
        const cull_this = switch (p.cull.mode) {
            .back => !is_front,
            .front => is_front,
            .none => false,
        };
        if (cull_this) return;
    }
    // The FS reads its `.descriptor` (UBO) params from `ubo_ptrs` by binding (rasterShadedFmt
    // uses prog.fs_buffer_bindings), so binding 0 (the VS block) and binding 1 (the FS block)
    // never collide. `ubo_ptrs` is binding-indexed (filled at ubos[binding] by the draw).
    raster.rasterShadedFmt(t.bytes, t.width, t.height, t.format, samples, screen, &vouts, prog, depth_att, stencil_att, tex, ubo_ptrs[0..], p.blend, clip, extra_targets, .{
        .alpha_to_coverage = p.alpha_to_coverage,
        .sample_coverage = p.sample_coverage,
        .sample_coverage_value = p.sample_coverage_value,
        .sample_coverage_invert = p.sample_coverage_invert,
    });
}

/// Draw one line segment (2 vertices) or point (1 vertex) by expanding it to a screen-space
/// quad (a `width`-px band perpendicular to the line, or a `width`x`width` square for a point)
/// and rasterizing the quad's two triangles through the same shaded-triangle path. The
/// varyings are the endpoint varyings (constant across the width) so the FS interpolates them
/// along the line exactly. Depth + scissor apply. No back-face cull (a line has no winding).
/// Axis-aligned lines rasterize exactly. A thin diagonal line may stipple (the quad-expansion
/// tradeoff). The nvidia driver uses the hardware line rasterizer, which is exact.
fn drawLinePoint(t: *Resource, p: *Pipeline, vb: ?*Resource, first_vertex: u32, vert_count: u32, instance_index: u32, ubos: []const ?[*]const u8, depth_res: ?*Resource, tex: ?*const sampler.TexDesc, scissor: ?hal.ScissorRect, viewport: ?hal.Viewport, width: f32) void {
    const prog = &(p.program.?);
    if (t.width == 0 or t.height == 0) return;
    const pulling = prog.vs_index_count > 0;
    if (!pulling and vb == null) return;

    var ubo_ptrs: [spirv_jit.GfxBuffers.max]?[*]const u8 = .{null} ** spirv_jit.GfxBuffers.max;
    {
        var b: usize = 0;
        while (b < ubos.len and b < ubo_ptrs.len) : (b += 1) ubo_ptrs[b] = ubos[b];
    }
    var vs_bufs: [spirv_jit.GfxBuffers.max]?[*]const u8 = .{null} ** spirv_jit.GfxBuffers.max;
    {
        var cursor: usize = 0;
        var k: usize = 0;
        while (k < prog.vs_buffers and k < vs_bufs.len) : (k += 1) {
            const bind = prog.vs_buffer_bindings[k];
            if (bind >= 0) {
                const bi: usize = @intCast(bind);
                vs_bufs[k] = if (bi < ubos.len) ubos[bi] else null;
            } else {
                vs_bufs[k] = if (cursor < ubos.len) ubos[cursor] else null;
                cursor += 1;
            }
        }
    }

    // Run the VS on the primitive's endpoints (1 for a point, 2 for a line).
    var ev: [2][GfxOut.vertex_len]f32 = undefined;
    var es: [2][2]f32 = undefined;
    var ez: [2]f32 = undefined;
    var vi: u32 = 0;
    while (vi < vert_count) : (vi += 1) {
        const sv = runVsVertex(t, p, prog, vb, pulling, first_vertex + vi, instance_index, vs_bufs[0..prog.vs_buffers], &ev[vi], viewport) orelse return;
        es[vi] = .{ sv[0], sv[1] };
        ez[vi] = sv[2];
    }
    const clip = effectiveScissor(scissor, viewport);

    const hw = @max(0.5, width * 0.5);
    var corners: [4][2]f32 = undefined;
    // Which endpoint (0/1) each corner a,b,c,d takes its varyings + depth from.
    var eidx: [4]usize = undefined;
    if (vert_count == 1) {
        ev[1] = ev[0];
        ez[1] = ez[0];
        // A point's size comes from the VS's gl_PointSize output (out[point_size_base]). 0 means
        // not written. Fall back to the default (glLineWidth-shared `width`, normally 1).
        const ps = ev[0][GfxOut.point_size_base];
        const phw = @max(0.5, (if (ps > 0) ps else width) * 0.5);
        corners = .{
            .{ es[0][0] - phw, es[0][1] - phw }, .{ es[0][0] + phw, es[0][1] - phw },
            .{ es[0][0] + phw, es[0][1] + phw }, .{ es[0][0] - phw, es[0][1] + phw },
        };
        eidx = .{ 0, 0, 0, 0 };
        // A point sprite: gl_PointCoord runs 0..1 across the phw*2 square (origin top-left).
        raster.current_point_sprite = .{ .ox = es[0][0] - phw, .oy = es[0][1] - phw, .inv_size = 1.0 / (2.0 * phw) };
    } else {
        var dx = es[1][0] - es[0][0];
        var dy = es[1][1] - es[0][1];
        const len = @sqrt(dx * dx + dy * dy);
        if (len < 1e-6) return;
        dx /= len;
        dy /= len;
        const px = -dy * hw; // perpendicular * half width
        const py = dx * hw;
        corners = .{
            .{ es[0][0] + px, es[0][1] + py }, .{ es[0][0] - px, es[0][1] - py },
            .{ es[1][0] - px, es[1][1] - py }, .{ es[1][0] + px, es[1][1] + py },
        };
        eidx = .{ 0, 0, 1, 1 };
    }

    // The band's two triangles: corners (a,b,c) and (a,c,d).
    const tris = [2][3]usize{ .{ 0, 1, 2 }, .{ 0, 2, 3 } };
    for (tris) |tri| {
        const scr = [3][2]f32{ corners[tri[0]], corners[tri[1]], corners[tri[2]] };
        const vo = [3][GfxOut.vertex_len]f32{ ev[eidx[tri[0]]], ev[eidx[tri[1]]], ev[eidx[tri[2]]] };
        const sz = [3]f32{ ez[eidx[tri[0]]], ez[eidx[tri[1]]], ez[eidx[tri[2]]] };
        var depth_att: ?raster.DepthAttachment = null;
        if (depth_res) |dr| {
            const dc = @as(usize, t.width) * t.height;
            if (p.depth.test_enable and dr.bytes.len >= dc * 4) {
                const aligned: []align(@alignOf(f32)) u8 = @alignCast(dr.bytes[0 .. dc * 4]);
                depth_att = .{ .buffer = std.mem.bytesAsSlice(f32, aligned), .state = p.depth, .screen_z = sz, .samples = 1 };
            }
        }
        raster.rasterShadedFmt(t.bytes, t.width, t.height, t.format, 1, scr, &vo, prog, depth_att, null, tex, ubo_ptrs[0..], p.blend, clip, &.{}, .{});
    }
    raster.current_point_sprite = null; // clear so a later triangle/line draw sees no sprite
}

fn execute(cb: *CommandBuffer) void {
    var target: ?*Resource = null;
    var depth_target: ?*Resource = null;
    var stencil_target: ?*Resource = null;
    var pipeline: ?*Pipeline = null;
    var vbuf: ?*Resource = null;
    var ubos: [spirv_jit.GfxBuffers.max]?[*]const u8 = .{null} ** spirv_jit.GfxBuffers.max;
    var tex: ?sampler.TexDesc = null;
    var scissor: ?hal.ScissorRect = null;
    var viewport: ?hal.Viewport = null;
    var color_rts: [7]?*Resource = .{null} ** 7; // MRT color targets 1..7 (0 = target)
    for (cb.cmds.items) |cmd| switch (cmd) {
        .set_render_target => |t| target = t,
        .set_color_target => |ct| if (ct.index == 0) {
            target = ct.resource;
        } else if (ct.index - 1 < color_rts.len) {
            color_rts[ct.index - 1] = ct.resource;
        },
        .set_scissor => |r| scissor = r,
        .set_viewport => |v| viewport = v,
        .set_depth_target => |d| {
            depth_target = d.depth;
            // Clear the depth buffer to the render-pass clear value (e.g. 1.0) when one
            // was given. A null clear_value binds the depth target without clearing, so
            // the accumulated depth from earlier draws this frame is preserved (the GLES
            // clear-once-then-draw-many contract). The depth Resource bytes are an
            // f32-per-(pixel*sample) buffer (page-aligned alloc). For MSAA it holds
            // w*h*samples depths. Clearing all of them is correct regardless of samples.
            if (d.clear_value) |cv| {
                const aligned: []align(@alignOf(f32)) u8 = @alignCast(d.depth.bytes);
                const dbuf = std.mem.bytesAsSlice(f32, aligned);
                for (dbuf) |*v| v.* = cv;
            }
        },
        .set_stencil_target => |s| {
            stencil_target = s.stencil;
            // Clear the stencil buffer (one u8 per pixel) to the clear value when given.
            // A null clear_value preserves the accumulated stencil across submits (the
            // GLES clear-once contract, mirroring the depth target above).
            if (s.clear_value) |cv| {
                for (s.stencil.bytes) |*v| v.* = cv;
            }
        },
        .bind_pipeline => |p| pipeline = p,
        .bind_vertex_buffer => |b| vbuf = b,
        .bind_uniform_buffer => |u| {
            if (u.binding < ubos.len) ubos[u.binding] = u.snapshot.ptr;
        },
        .bind_texture => |tb| {
            // Build the host sampler descriptor from the bound image Resource + the
            // sampler state. The image is rgba8 (the ICD's sampled-image path), so the
            // pitch is width*4.
            const img: *Resource = @ptrCast(@alignCast(tb.image));
            // The stored texel format drives the sampler's decode (sRGB EOTF, fp16/fp32).
            // Most sampled images are rgba8_unorm (the ICD/GLES path).
            const tex_fmt: sampler.TexFormat = switch (img.format) {
                .rgba8_srgb => .rgba8_srgb,
                .rgba16_float => .rgba16_float,
                .r32g32b32a32_float => .rgba32_float,
                else => .rgba8_unorm,
            };
            tex = .{
                .pixels = img.bytes.ptr,
                .width = img.width,
                .height = img.height,
                .pitch = img.width * tex_fmt.bytesPerTexel(),
                .format = tex_fmt,
                .filter = switch (tb.filter) {
                    .nearest => .nearest,
                    .linear => .linear,
                },
                .min_filter = switch (tb.min_filter) {
                    .nearest => .nearest,
                    .linear => .linear,
                },
                .mip_filter = switch (tb.mip_filter) {
                    .none => .none,
                    .nearest => .nearest,
                    .linear => .linear,
                },
                .levels = img.mip_levels,
                .max_anisotropy = tb.max_anisotropy,
                .base_level = tb.base_level,
                .tex_max_level = tb.max_level,
                .swizzle = .{ @intFromEnum(tb.swizzle[0]), @intFromEnum(tb.swizzle[1]), @intFromEnum(tb.swizzle[2]), @intFromEnum(tb.swizzle[3]) },
                .lod = tb.lod_bias, // GL_TEXTURE_LOD_BIAS: sampleTexture adds this to the computed LOD
                .tex_min_lod = tb.min_lod, // GL_TEXTURE_MIN_LOD / MAX_LOD: clamp the effective LOD
                .tex_max_lod = tb.max_lod,
                .address_u = switch (tb.address_u) {
                    .repeat => .repeat,
                    .clamp_to_edge => .clamp_to_edge,
                    .mirrored_repeat => .mirrored_repeat,
                },
                .address_v = switch (tb.address_v) {
                    .repeat => .repeat,
                    .clamp_to_edge => .clamp_to_edge,
                    .mirrored_repeat => .mirrored_repeat,
                },
                // GL_TEXTURE_COMPARE_MODE / _FUNC (sampler2DShadow): depth-compare sampling.
                .compare_enable = tb.compare_enable,
                .compare_op = @intFromEnum(tb.compare_op),
            };
            // A mip-chained image (mip_levels > 1) is tightly packed level-0-first. Record each
            // level's byte offset so the sampler can fetch any level. Base (0) is at offset 0.
            if (img.mip_levels > 1) {
                const bpp = tex_fmt.bytesPerTexel();
                var lvl: u8 = 0;
                while (lvl < img.mip_levels and lvl < sampler.MAX_MIP_LEVELS) : (lvl += 1) {
                    tex.?.level_off[lvl] = @intCast(hal.mipLevelOffset(img.width, img.height, lvl, bpp));
                }
            }
            // A cubemap image holds 6 faces packed contiguously. Each face is one image, or (when
            // the image is mip-chained) its own mip chain. Record each face's byte offset so
            // sampleTextureCube can pick the face. The per-face relative mip offsets reuse the
            // level_off set above (all faces are the same size).
            if (img.is_cube) {
                const bpp = tex_fmt.bytesPerTexel();
                const face_bytes: usize = if (img.mip_levels > 1)
                    hal.mipChainBytes(img.width, img.height, img.mip_levels, bpp)
                else
                    @as(usize, img.width) * img.height * bpp;
                tex.?.is_cube = true;
                for (0..6) |f| tex.?.face_off[f] = @intCast(@as(usize, f) * face_bytes);
            }
            // A 3D texture (sampler3D) or a 2D array (sampler2DArray): both have `depth` layers
            // packed layer-major. `is_array` distinguishes them: a 3D volume filters across slices
            // (trilinear), a 2D array selects one layer by a raw index (no cross-layer filtering).
            if (img.depth > 1) {
                if (img.is_array) tex.?.is_2darray = true else tex.?.is_3d = true;
                tex.?.depth = img.depth;
            }
        },
        .clear => |color| {
            if (target) |t| {
                // The target may be a multisampled image (bytes = w*h*samples*bpp).
                // Clear every pixel-sample to the clear color, format-aware. samples is
                // recovered from the backing size (1 for a normal single-sample target).
                const bpp = t.format.bytesPerPixel();
                const per_pixel = @as(usize, t.width) * t.height * bpp;
                const total_samples: u32 = if (per_pixel > 0) @intCast(t.bytes.len / per_pixel) else 1;
                // glClear honors an enabled scissor test: clear only the scissor sub-rect
                // (all samples) when one is set, else the whole surface.
                if (scissor) |r| {
                    raster.clearFmtScissor(t.bytes, t.width, t.height, @intCast(total_samples), t.format, color, r);
                } else {
                    raster.clearFmt(t.bytes, t.width, @max(1, t.height * total_samples), t.format, color);
                }
            }
        },
        .resolve => |r| {
            // MSAA resolve: box-average the N samples of the multisampled source into
            // the single-sample destination image, format-aware.
            raster.resolveMsaa(r.dst.bytes, r.src.bytes, r.width, r.height, r.format, r.samples);
        },
        .draw => |d| {
            const t = target orelse continue;
            const p = pipeline orelse continue;
            // Real-SPIR-V path: if the pipeline JITed a VS+FS program, execute it. A draw of N
            // vertices is N/vpp primitives (vpp = 3 triangles / 2 lines / 1 point), each run in
            // order. A vertex-pulling VS needs no vertex buffer (vbuf may be null). The
            // attribute-fed + declarative paths require one.
            if (p.program != null) {
                const tex_ptr: ?*const sampler.TexDesc = if (tex) |*td| td else null;
                // MRT: the additional bound color targets (contiguous from index 1).
                var extra_ct: [7]raster.ColorTarget = undefined;
                var n_extra: usize = 0;
                for (color_rts) |maybe| {
                    const cr = maybe orelse break; // stop at the first gap
                    extra_ct[n_extra] = .{ .bytes = cr.bytes, .format = cr.format };
                    n_extra += 1;
                }
                const extra_targets = extra_ct[0..n_extra];
                const vpp = p.topology.vertsPerPrimitive();
                if (d.vertex_count < vpp) continue;
                const prim_count = d.vertex_count / vpp;
                // Instanced: run the whole primitive list once per instance, feeding the VS
                // gl_InstanceIndex = first_instance + inst. A non-instanced draw is 1 instance.
                const ninst = @max(d.instance_count, 1);
                var inst: u32 = 0;
                while (inst < ninst) : (inst += 1) {
                    const iidx = d.first_instance + inst;
                    var pr: u32 = 0;
                    while (pr < prim_count) : (pr += 1) {
                        const fv = d.first_vertex + pr * vpp;
                        switch (p.topology) {
                            .triangle_list => drawShaded(t, p, vbuf, fv, iidx, &ubos, depth_target, stencil_target, tex_ptr, scissor, viewport, extra_targets),
                            .line_list => drawLinePoint(t, p, vbuf, fv, 2, iidx, &ubos, depth_target, tex_ptr, scissor, viewport, p.line_width),
                            .point_list => drawLinePoint(t, p, vbuf, fv, 1, iidx, &ubos, depth_target, tex_ptr, scissor, viewport, p.line_width),
                        }
                    }
                }
                continue;
            }
            if (d.vertex_count < 3) continue;
            const vb = vbuf orelse continue;
            const vir = p.vertex.vertex orelse continue;
            const stride = @as(usize, p.layout.stride);
            var pos_off: usize = 0;
            var col_off: usize = 0;
            // Component counts come from the attribute formats so a vec2 position
            // + a vec3 or vec4 color both read correctly (a vec3 color leaves
            // alpha defaulted to 1.0). The Vulkan ICD describes its vertices as
            // r32g32_float position + r32g32b32(a32)_float color via these formats.
            var pos_comps: u32 = 2;
            var col_comps: u32 = 4;
            var pos_found = false;
            var col_found = false;
            for (p.layout.attributes) |attr| {
                if (attr.location == vir.position_attr) {
                    pos_off = @as(usize, attr.offset);
                    pos_comps = attr.format.componentCount();
                    pos_found = true;
                }
                if (attr.location == vir.color_attr) {
                    col_off = @as(usize, attr.offset);
                    col_comps = attr.format.componentCount();
                    col_found = true;
                }
            }
            if (!pos_found or !col_found) continue;
            const pos_bytes = @as(usize, pos_comps) * 4;
            const col_bytes = @as(usize, col_comps) * 4;
            // Pre-check: all 3 vertex reads must be in bounds before filling any.
            {
                var i: u32 = 0;
                var in_bounds = true;
                while (i < 3) : (i += 1) {
                    const base = (@as(usize, d.first_vertex) + i) * stride;
                    const pos_end = base + pos_off + pos_bytes;
                    const col_end = base + col_off + col_bytes;
                    const required_end = if (pos_end > col_end) pos_end else col_end;
                    if (required_end > vb.bytes.len) {
                        in_bounds = false;
                        break;
                    }
                }
                if (!in_bounds) continue;
            }
            var verts: [3]raster.Vertex = undefined;
            var i: u32 = 0;
            while (i < 3) : (i += 1) {
                const base = (@as(usize, d.first_vertex) + i) * stride;
                // y defaults to 0 if the position is 1-component. Alpha defaults to
                // 1.0 if the color is vec3 (no alpha channel in the buffer).
                verts[i] = .{
                    .x = readF32(vb.bytes, base + pos_off + 0),
                    .y = if (pos_comps >= 2) readF32(vb.bytes, base + pos_off + 4) else 0,
                    .r = readF32(vb.bytes, base + col_off + 0),
                    .g = if (col_comps >= 2) readF32(vb.bytes, base + col_off + 4) else 0,
                    .b = if (col_comps >= 3) readF32(vb.bytes, base + col_off + 8) else 0,
                    .a = if (col_comps >= 4) readF32(vb.bytes, base + col_off + 12) else 1,
                };
            }
            raster.drawTriangle(t.bytes, t.width, t.height, verts);
        },
    };
}

pub const Context = struct {
    gpa: std.mem.Allocator,

    pub fn create(gpa: std.mem.Allocator) hal.Error!hal.Context {
        const self = gpa.create(Context) catch return error.OutOfMemory;
        self.* = .{ .gpa = gpa };
        return .{ .ptr = self, .vtable = &vtable };
    }

    fn beginCommands(ptr: *anyopaque) hal.Error!hal.CommandBuffer {
        const self: *Context = @ptrCast(@alignCast(ptr));
        const cb = self.gpa.create(CommandBuffer) catch return error.OutOfMemory;
        cb.* = .{ .gpa = self.gpa };
        return .{ .ptr = cb, .vtable = &CommandBuffer.vtable };
    }
    fn submit(ptr: *anyopaque, cb: hal.CommandBuffer) hal.Error!void {
        _ = ptr;
        const self_cb: *CommandBuffer = @ptrCast(@alignCast(cb.ptr));
        execute(self_cb);
    }
    fn present(ptr: *anyopaque, surface: *hal.Surface, source: *hal.Resource) hal.Error!void {
        _ = ptr;
        const sw_surface: *@import("surface.zig").Surface = @ptrCast(@alignCast(surface));
        const src: *Resource = @ptrCast(@alignCast(source));
        const buf = try sw_surface.platform.currentBuffer();
        const copy_w = @min(src.width, buf.width);
        const copy_h = @min(src.height, buf.height);
        // The render target is rgba8_unorm. If the surface buffer wants bgra8
        // (e.g. Wayland XRGB8888), convert during the blit so the buffer holds
        // the correct on-wire bytes once. The surface must not mutate pixels
        // after attach (the compositor reads the shm buffer asynchronously).
        const swap_rb = buf.format == .bgra8_unorm;
        var y: u32 = 0;
        while (y < copy_h) : (y += 1) {
            const src_row = src.bytes[(@as(usize, y) * src.width) * 4 ..][0 .. copy_w * 4];
            const dst_row = buf.bytes[@as(usize, y) * buf.stride ..][0 .. copy_w * 4];
            if (swap_rb) {
                // RGBA -> BGRA (XRGB8888 LE): swap R and B per pixel. Process 4 pixels (16
                // bytes) per @shuffle so the backend emits a single NEON TBL/byte-permute,
                // then a scalar remainder. The shuffle mask swaps bytes 0<->2 in each of the
                // four 4-byte lanes (R<->B), leaving G and A in place.
                const V16 = @Vector(16, u8);
                const mask = V16{ 2, 1, 0, 3, 6, 5, 4, 7, 10, 9, 8, 11, 14, 13, 12, 15 };
                var x: usize = 0;
                while (x + 4 <= copy_w) : (x += 4) {
                    const o = x * 4;
                    const v: V16 = src_row[o..][0..16].*;
                    dst_row[o..][0..16].* = @shuffle(u8, v, undefined, mask);
                }
                while (x < copy_w) : (x += 1) {
                    const o = x * 4;
                    dst_row[o + 0] = src_row[o + 2]; // B <- R
                    dst_row[o + 1] = src_row[o + 1]; // G
                    dst_row[o + 2] = src_row[o + 0]; // R <- B
                    dst_row[o + 3] = src_row[o + 3]; // A / X
                }
            } else {
                @memcpy(dst_row, src_row);
            }
        }
        try sw_surface.platform.commit();
    }
    fn deinit(ptr: *anyopaque) void {
        const self: *Context = @ptrCast(@alignCast(ptr));
        self.gpa.destroy(self);
    }

    const vtable = hal.Context.VTable{
        .beginCommands = &beginCommands,
        .submit = &submit,
        .present = &present,
        .deinit = &deinit,
    };
};

test "context creates a command buffer" {
    const gpa = std.testing.allocator;
    const ctx = try Context.create(gpa);
    defer ctx.deinit();
    const cb = try ctx.beginCommands();
    defer cb.deinit();
    try cb.clear(.{ .r = 0, .g = 0, .b = 0, .a = 1 });
}

test "draw with undersized vertex buffer is safely skipped" {
    const ShaderModule = std.meta.Child(@FieldType(Pipeline, "vertex"));
    const gpa = std.testing.allocator;

    // Render target: 4x4 rgba8 image, pre-cleared to 0.
    var target_pixels = [_]u8{0} ** (4 * 4 * 4);
    var target = Resource{
        .kind = .image,
        .bytes = &target_pixels,
        .width = 4,
        .height = 4,
        .format = .rgba8_unorm,
    };

    // Vertex buffer way too small (only 1 byte) to hold any vertex data.
    var tiny_buf = [_]u8{0} ** 1;
    var vbuf = Resource{ .kind = .buffer, .bytes = &tiny_buf };

    // Pipeline: stride=24 (8 pos + 16 col), two attributes at locations 0 and 1.
    var vs = ShaderModule{ .stage = .vertex, .vertex = .{ .position_attr = 0, .color_attr = 1 } };
    var fs = ShaderModule{ .stage = .fragment, .fragment = .{} };
    const attrs = [_]hal.VertexAttribute{
        .{ .location = 0, .format = .rgba8_unorm, .offset = 0 },
        .{ .location = 1, .format = .rgba8_unorm, .offset = 8 },
    };
    var pipe = Pipeline{
        .vertex = &vs,
        .fragment = &fs,
        .layout = .{ .stride = 24, .attributes = &attrs },
        .color_format = .rgba8_unorm,
    };

    const ctx = try Context.create(gpa);
    defer ctx.deinit();
    const cb = try ctx.beginCommands();
    defer cb.deinit();

    try cb.setRenderTarget(@ptrCast(&target));
    try cb.bindPipeline(@ptrCast(&pipe));
    try cb.bindVertexBuffer(@ptrCast(&vbuf));
    try cb.draw(3, 0);

    // submit must not panic. Pixels must stay zero (draw was skipped).
    try ctx.submit(cb);
    try std.testing.expectEqual(@as(u8, 0), target_pixels[0]);
}

test "draw with correct buffer writes a pixel" {
    const ShaderModule = std.meta.Child(@FieldType(Pipeline, "vertex"));
    const gpa = std.testing.allocator;

    // 8x8 render target.
    var target_pixels = [_]u8{0} ** (8 * 8 * 4);
    var target = Resource{
        .kind = .image,
        .bytes = &target_pixels,
        .width = 8,
        .height = 8,
        .format = .rgba8_unorm,
    };

    // stride=24: bytes 0..7 = xy (pos), bytes 8..23 = rgba (col).
    // Big white triangle covering most of the image in NDC.
    const stride: usize = 24;
    var vb_bytes = [_]u8{0} ** (3 * stride);
    const write_f32 = struct {
        fn f(buf: []u8, off: usize, v: f32) void {
            const bytes = std.mem.toBytes(v);
            buf[off + 0] = bytes[0];
            buf[off + 1] = bytes[1];
            buf[off + 2] = bytes[2];
            buf[off + 3] = bytes[3];
        }
    }.f;
    // vertex 0: pos(-1,-1), col(1,1,1,1)
    write_f32(&vb_bytes, 0, -1.0);
    write_f32(&vb_bytes, 4, -1.0);
    write_f32(&vb_bytes, 8, 1.0);
    write_f32(&vb_bytes, 12, 1.0);
    write_f32(&vb_bytes, 16, 1.0);
    write_f32(&vb_bytes, 20, 1.0);
    // vertex 1: pos(1,-1), col(1,1,1,1)
    write_f32(&vb_bytes, stride + 0, 1.0);
    write_f32(&vb_bytes, stride + 4, -1.0);
    write_f32(&vb_bytes, stride + 8, 1.0);
    write_f32(&vb_bytes, stride + 12, 1.0);
    write_f32(&vb_bytes, stride + 16, 1.0);
    write_f32(&vb_bytes, stride + 20, 1.0);
    // vertex 2: pos(0,1), col(1,1,1,1)
    write_f32(&vb_bytes, stride * 2 + 0, 0.0);
    write_f32(&vb_bytes, stride * 2 + 4, 1.0);
    write_f32(&vb_bytes, stride * 2 + 8, 1.0);
    write_f32(&vb_bytes, stride * 2 + 12, 1.0);
    write_f32(&vb_bytes, stride * 2 + 16, 1.0);
    write_f32(&vb_bytes, stride * 2 + 20, 1.0);

    var vbuf = Resource{ .kind = .buffer, .bytes = &vb_bytes };

    var vs = ShaderModule{ .stage = .vertex, .vertex = .{ .position_attr = 0, .color_attr = 1 } };
    var fs = ShaderModule{ .stage = .fragment, .fragment = .{} };
    const attrs = [_]hal.VertexAttribute{
        .{ .location = 0, .format = .rgba8_unorm, .offset = 0 },
        .{ .location = 1, .format = .rgba8_unorm, .offset = 8 },
    };
    var pipe = Pipeline{
        .vertex = &vs,
        .fragment = &fs,
        .layout = .{ .stride = @intCast(stride), .attributes = &attrs },
        .color_format = .rgba8_unorm,
    };

    const ctx = try Context.create(gpa);
    defer ctx.deinit();
    const cb = try ctx.beginCommands();
    defer cb.deinit();

    try cb.setRenderTarget(@ptrCast(&target));
    try cb.bindPipeline(@ptrCast(&pipe));
    try cb.bindVertexBuffer(@ptrCast(&vbuf));
    try cb.draw(3, 0);

    try ctx.submit(cb);

    // Center pixel (4,4) should be white (255) -- inside the triangle.
    const cx: usize = 4;
    const cy: usize = 4;
    const idx = (cy * 8 + cx) * 4;
    try std.testing.expectEqual(@as(u8, 255), target_pixels[idx + 0]);
}

test "real SPIR-V VS+FS draw: a channel-rotate fragment shader executes per fragment" {
    const gpa = std.testing.allocator;
    const Device = @import("device.zig").Device;
    const sw_shader = @import("shader.zig");
    const op = @import("vulcan-spirv").opcodes;
    const sc = op.StorageClass;

    // Passthrough VS: in vec2 pos(loc0) + vec3 color(loc1) -> gl_Position + out vc(loc0).
    var vsb = try @import("vulcan-spirv").binary.Builder.init(gpa, 24);
    defer vsb.deinit(gpa);
    try vsb.emit(gpa, op.EntryPoint, &.{ op.ExecutionModel.vertex, 15, 0, 11, 12, 13, 14 });
    try vsb.emit(gpa, op.Decorate, &.{ 13, op.Decoration.builtin, op.BuiltIn.position });
    try vsb.emit(gpa, op.Decorate, &.{ 11, op.Decoration.location, 0 });
    try vsb.emit(gpa, op.Decorate, &.{ 12, op.Decoration.location, 1 });
    try vsb.emit(gpa, op.Decorate, &.{ 14, op.Decoration.location, 0 });
    try vsb.emit(gpa, op.TypeVoid, &.{1});
    try vsb.emit(gpa, op.TypeFunction, &.{ 2, 1 });
    try vsb.emit(gpa, op.TypeFloat, &.{ 3, 32 });
    try vsb.emit(gpa, op.TypeVector, &.{ 4, 3, 2 });
    try vsb.emit(gpa, op.TypeVector, &.{ 5, 3, 3 });
    try vsb.emit(gpa, op.TypeVector, &.{ 6, 3, 4 });
    try vsb.emit(gpa, op.TypePointer, &.{ 7, sc.input, 4 });
    try vsb.emit(gpa, op.TypePointer, &.{ 8, sc.input, 5 });
    try vsb.emit(gpa, op.TypePointer, &.{ 9, sc.output, 6 });
    try vsb.emit(gpa, op.TypePointer, &.{ 10, sc.output, 5 });
    try vsb.emit(gpa, op.Constant, &.{ 3, 17, @bitCast(@as(f32, 0.0)) });
    try vsb.emit(gpa, op.Constant, &.{ 3, 18, @bitCast(@as(f32, 1.0)) });
    try vsb.emit(gpa, op.Variable, &.{ 7, 11, sc.input });
    try vsb.emit(gpa, op.Variable, &.{ 8, 12, sc.input });
    try vsb.emit(gpa, op.Variable, &.{ 9, 13, sc.output });
    try vsb.emit(gpa, op.Variable, &.{ 10, 14, sc.output });
    try vsb.emit(gpa, op.Function, &.{ 1, 15, 0, 2 });
    try vsb.emit(gpa, op.Label, &.{16});
    try vsb.emit(gpa, op.Load, &.{ 4, 19, 11 });
    try vsb.emit(gpa, op.Load, &.{ 5, 20, 12 });
    try vsb.emit(gpa, op.CompositeExtract, &.{ 3, 21, 19, 0 });
    try vsb.emit(gpa, op.CompositeExtract, &.{ 3, 22, 19, 1 });
    try vsb.emit(gpa, op.CompositeConstruct, &.{ 6, 23, 21, 22, 17, 18 });
    try vsb.emit(gpa, op.Store, &.{ 13, 23 });
    try vsb.emit(gpa, op.Store, &.{ 14, 20 });
    try vsb.emit(gpa, op.Return, &.{});
    try vsb.emit(gpa, op.FunctionEnd, &.{});

    // Channel-rotate FS: in vec3 vc(loc0) -> out vec4 o = vec4(vc.b, vc.r, vc.g, 1).
    var fsb = try @import("vulcan-spirv").binary.Builder.init(gpa, 18);
    defer fsb.deinit(gpa);
    try fsb.emit(gpa, op.EntryPoint, &.{ op.ExecutionModel.fragment, 10, 0, 8, 9 });
    try fsb.emit(gpa, op.Decorate, &.{ 8, op.Decoration.location, 0 });
    try fsb.emit(gpa, op.Decorate, &.{ 9, op.Decoration.location, 0 });
    try fsb.emit(gpa, op.TypeVoid, &.{1});
    try fsb.emit(gpa, op.TypeFunction, &.{ 2, 1 });
    try fsb.emit(gpa, op.TypeFloat, &.{ 3, 32 });
    try fsb.emit(gpa, op.TypeVector, &.{ 4, 3, 3 });
    try fsb.emit(gpa, op.TypeVector, &.{ 5, 3, 4 });
    try fsb.emit(gpa, op.TypePointer, &.{ 6, sc.input, 4 });
    try fsb.emit(gpa, op.TypePointer, &.{ 7, sc.output, 5 });
    try fsb.emit(gpa, op.Constant, &.{ 3, 12, @bitCast(@as(f32, 1.0)) });
    try fsb.emit(gpa, op.Variable, &.{ 6, 8, sc.input });
    try fsb.emit(gpa, op.Variable, &.{ 7, 9, sc.output });
    try fsb.emit(gpa, op.Function, &.{ 1, 10, 0, 2 });
    try fsb.emit(gpa, op.Label, &.{11});
    try fsb.emit(gpa, op.Load, &.{ 4, 13, 8 });
    try fsb.emit(gpa, op.CompositeExtract, &.{ 3, 14, 13, 2 }); // vc.b
    try fsb.emit(gpa, op.CompositeExtract, &.{ 3, 15, 13, 0 }); // vc.r
    try fsb.emit(gpa, op.CompositeExtract, &.{ 3, 16, 13, 1 }); // vc.g
    try fsb.emit(gpa, op.CompositeConstruct, &.{ 5, 17, 14, 15, 16, 12 });
    try fsb.emit(gpa, op.Store, &.{ 9, 17 });
    try fsb.emit(gpa, op.Return, &.{});
    try fsb.emit(gpa, op.FunctionEnd, &.{});

    const dev = try Device.create(gpa);
    defer dev.deinit();

    const vs = try dev.createShaderModule(.{ .stage = .vertex, .code = std.mem.sliceAsBytes(vsb.words.items) });
    defer dev.destroyShaderModule(vs);
    const fs = try dev.createShaderModule(.{ .stage = .fragment, .code = std.mem.sliceAsBytes(fsb.words.items) });
    defer dev.destroyShaderModule(fs);

    // The real shader modules must carry SPIR-V (not the declarative IR).
    const vs_mod: *sw_shader.ShaderModule = @ptrCast(@alignCast(vs));
    try std.testing.expect(vs_mod.compute_spirv != null);

    // 64x64 RGBA8 target.
    const W = 64;
    const H = 64;
    const target = try dev.createResource(.{ .image = .{ .width = W, .height = H, .format = .rgba8_unorm } });
    defer dev.destroyResource(target);
    const tpix = try dev.mapResource(target);

    // Vertex buffer: 3 vertices, stride 20 (vec2 pos + vec3 color), R/G/B corners.
    const stride: u32 = 20;
    const vbuf = try dev.createResource(.{ .buffer = .{ .size = 3 * stride } });
    defer dev.destroyResource(vbuf);
    const vb = try dev.mapResource(vbuf);
    const wf = struct {
        fn f(buf: []u8, off: usize, v: f32) void {
            @memcpy(buf[off..][0..4], &std.mem.toBytes(v));
        }
    }.f;
    // v0: pos(-0.8,-0.8) color RED(1,0,0)
    wf(vb, 0, -0.8);
    wf(vb, 4, -0.8);
    wf(vb, 8, 1);
    wf(vb, 12, 0);
    wf(vb, 16, 0);
    // v1: pos(0.8,-0.8) color GREEN(0,1,0)
    wf(vb, stride + 0, 0.8);
    wf(vb, stride + 4, -0.8);
    wf(vb, stride + 8, 0);
    wf(vb, stride + 12, 1);
    wf(vb, stride + 16, 0);
    // v2: pos(0.0,0.8) color BLUE(0,0,1)
    wf(vb, 2 * stride + 0, 0.0);
    wf(vb, 2 * stride + 4, 0.8);
    wf(vb, 2 * stride + 8, 0);
    wf(vb, 2 * stride + 12, 0);
    wf(vb, 2 * stride + 16, 1);

    const attrs = [_]hal.VertexAttribute{
        .{ .location = 0, .format = .r32g32_float, .offset = 0 },
        .{ .location = 1, .format = .r32g32b32_float, .offset = 8 },
    };
    const pipe = try dev.createPipeline(.{
        .vertex = vs,
        .fragment = fs,
        .vertex_layout = .{ .stride = stride, .attributes = &attrs },
        .color_format = .rgba8_unorm,
    });
    defer dev.destroyPipeline(pipe);

    // The pipeline must have JITed a real shader program (not the declarative path).
    const sw_pipe: *Pipeline = @ptrCast(@alignCast(pipe));
    try std.testing.expect(sw_pipe.program != null);

    const ctx = try dev.createContext();
    defer ctx.deinit();
    const cb = try ctx.beginCommands();
    defer cb.deinit();
    try cb.setRenderTarget(target);
    try cb.clear(.{ .r = 0, .g = 0, .b = 0, .a = 1 });
    try cb.bindPipeline(pipe);
    try cb.bindVertexBuffer(vbuf);
    try cb.draw(3, 0);
    try ctx.submit(cb);

    // Center pixel: the plain interpolation at the centroid is ~(R,G,B)=(1/3,1/3,1/3).
    // The channel-rotate FS outputs (vc.b, vc.r, vc.g). At the exact centroid these are
    // all 1/3 so a swap is invisible. Use an off-center point biased toward RED so the
    // rotation is unmistakable: there interp ~ (high R, low G, low B) and the rotated
    // output is (low B -> R, high R -> G, low G -> B): G must dominate, not R.
    // The RED vertex is at NDC (-0.8, -0.8). In Vulkan window space (origin top-left,
    // y down, no GL-style flip), NDC y = -0.8 maps near the TOP, so the RED corner is
    // at top-left: sample there (sx=20, sy=20).
    const sx: usize = 20; // near the RED (left-top) vertex
    const sy: usize = 20;
    const i = (sy * W + sx) * 4;
    const r = tpix[i + 0];
    const g = tpix[i + 1];
    const bch = tpix[i + 2];
    // Covered (non-clear) fragment.
    try std.testing.expect(r != 0 or g != 0 or bch != 0);
    // With the channel rotate, the GREEN channel (which received the high RED input)
    // must dominate the RED channel. The plain passthrough mapping (o = vc) would make
    // RED dominate here instead, so this asserts the FS SPIR-V actually executed.
    try std.testing.expect(g > r);
}

test "MSAA: a slanted triangle edge anti-aliases (partial-coverage resolve)" {
    const gpa = std.testing.allocator;
    const Device = @import("device.zig").Device;
    const op = @import("vulcan-spirv").opcodes;
    const sc = op.StorageClass;

    // Passthrough VS: in vec2 pos(loc0) + vec3 color(loc1) -> gl_Position + out vc(loc0).
    var vsb = try @import("vulcan-spirv").binary.Builder.init(gpa, 24);
    defer vsb.deinit(gpa);
    try vsb.emit(gpa, op.EntryPoint, &.{ op.ExecutionModel.vertex, 15, 0, 11, 12, 13, 14 });
    try vsb.emit(gpa, op.Decorate, &.{ 13, op.Decoration.builtin, op.BuiltIn.position });
    try vsb.emit(gpa, op.Decorate, &.{ 11, op.Decoration.location, 0 });
    try vsb.emit(gpa, op.Decorate, &.{ 12, op.Decoration.location, 1 });
    try vsb.emit(gpa, op.Decorate, &.{ 14, op.Decoration.location, 0 });
    try vsb.emit(gpa, op.TypeVoid, &.{1});
    try vsb.emit(gpa, op.TypeFunction, &.{ 2, 1 });
    try vsb.emit(gpa, op.TypeFloat, &.{ 3, 32 });
    try vsb.emit(gpa, op.TypeVector, &.{ 4, 3, 2 });
    try vsb.emit(gpa, op.TypeVector, &.{ 5, 3, 3 });
    try vsb.emit(gpa, op.TypeVector, &.{ 6, 3, 4 });
    try vsb.emit(gpa, op.TypePointer, &.{ 7, sc.input, 4 });
    try vsb.emit(gpa, op.TypePointer, &.{ 8, sc.input, 5 });
    try vsb.emit(gpa, op.TypePointer, &.{ 9, sc.output, 6 });
    try vsb.emit(gpa, op.TypePointer, &.{ 10, sc.output, 5 });
    try vsb.emit(gpa, op.Constant, &.{ 3, 17, @bitCast(@as(f32, 0.0)) });
    try vsb.emit(gpa, op.Constant, &.{ 3, 18, @bitCast(@as(f32, 1.0)) });
    try vsb.emit(gpa, op.Variable, &.{ 7, 11, sc.input });
    try vsb.emit(gpa, op.Variable, &.{ 8, 12, sc.input });
    try vsb.emit(gpa, op.Variable, &.{ 9, 13, sc.output });
    try vsb.emit(gpa, op.Variable, &.{ 10, 14, sc.output });
    try vsb.emit(gpa, op.Function, &.{ 1, 15, 0, 2 });
    try vsb.emit(gpa, op.Label, &.{16});
    try vsb.emit(gpa, op.Load, &.{ 4, 19, 11 });
    try vsb.emit(gpa, op.Load, &.{ 5, 20, 12 });
    try vsb.emit(gpa, op.CompositeExtract, &.{ 3, 21, 19, 0 });
    try vsb.emit(gpa, op.CompositeExtract, &.{ 3, 22, 19, 1 });
    try vsb.emit(gpa, op.CompositeConstruct, &.{ 6, 23, 21, 22, 17, 18 });
    try vsb.emit(gpa, op.Store, &.{ 13, 23 });
    try vsb.emit(gpa, op.Store, &.{ 14, 20 });
    try vsb.emit(gpa, op.Return, &.{});
    try vsb.emit(gpa, op.FunctionEnd, &.{});

    // Passthrough FS: in vec3 vc(loc0) -> out vec4 o = vec4(vc, 1).
    var fsb = try @import("vulcan-spirv").binary.Builder.init(gpa, 18);
    defer fsb.deinit(gpa);
    try fsb.emit(gpa, op.EntryPoint, &.{ op.ExecutionModel.fragment, 10, 0, 8, 9 });
    try fsb.emit(gpa, op.Decorate, &.{ 8, op.Decoration.location, 0 });
    try fsb.emit(gpa, op.Decorate, &.{ 9, op.Decoration.location, 0 });
    try fsb.emit(gpa, op.TypeVoid, &.{1});
    try fsb.emit(gpa, op.TypeFunction, &.{ 2, 1 });
    try fsb.emit(gpa, op.TypeFloat, &.{ 3, 32 });
    try fsb.emit(gpa, op.TypeVector, &.{ 4, 3, 3 });
    try fsb.emit(gpa, op.TypeVector, &.{ 5, 3, 4 });
    try fsb.emit(gpa, op.TypePointer, &.{ 6, sc.input, 4 });
    try fsb.emit(gpa, op.TypePointer, &.{ 7, sc.output, 5 });
    try fsb.emit(gpa, op.Constant, &.{ 3, 12, @bitCast(@as(f32, 1.0)) });
    try fsb.emit(gpa, op.Variable, &.{ 6, 8, sc.input });
    try fsb.emit(gpa, op.Variable, &.{ 7, 9, sc.output });
    try fsb.emit(gpa, op.Function, &.{ 1, 10, 0, 2 });
    try fsb.emit(gpa, op.Label, &.{11});
    try fsb.emit(gpa, op.Load, &.{ 4, 13, 8 });
    try fsb.emit(gpa, op.CompositeExtract, &.{ 3, 14, 13, 0 });
    try fsb.emit(gpa, op.CompositeExtract, &.{ 3, 15, 13, 1 });
    try fsb.emit(gpa, op.CompositeExtract, &.{ 3, 16, 13, 2 });
    try fsb.emit(gpa, op.CompositeConstruct, &.{ 5, 17, 14, 15, 16, 12 });
    try fsb.emit(gpa, op.Store, &.{ 9, 17 });
    try fsb.emit(gpa, op.Return, &.{});
    try fsb.emit(gpa, op.FunctionEnd, &.{});

    const dev = try Device.create(gpa);
    defer dev.deinit();
    const sw_shader = @import("shader.zig");

    const W = 64;
    const H = 64;
    const stride: u32 = 20;

    const renderAt = struct {
        fn go(d: hal.Device, vsm: *hal.ShaderModule, fsm: *hal.ShaderModule, samples: u8) ![W * H * 4]u8 {
            const attrs = [_]hal.VertexAttribute{
                .{ .location = 0, .format = .r32g32_float, .offset = 0 },
                .{ .location = 1, .format = .r32g32b32_float, .offset = 8 },
            };
            const pipe = try d.createPipeline(.{
                .vertex = vsm,
                .fragment = fsm,
                .vertex_layout = .{ .stride = stride, .attributes = &attrs },
                .color_format = .rgba8_unorm,
                .samples = samples,
            });
            defer d.destroyPipeline(pipe);

            // The multisampled color attachment (N samples/pixel) + a single-sample resolve.
            const ms = try d.createResource(.{ .image = .{ .width = W, .height = H, .format = .rgba8_unorm, .samples = samples } });
            defer d.destroyResource(ms);
            const resolved = try d.createResource(.{ .image = .{ .width = W, .height = H, .format = .rgba8_unorm } });
            defer d.destroyResource(resolved);

            const vbuf = try d.createResource(.{ .buffer = .{ .size = 3 * stride } });
            defer d.destroyResource(vbuf);
            const vb = try d.mapResource(vbuf);
            const wf = struct {
                fn f(buf: []u8, off: usize, v: f32) void {
                    @memcpy(buf[off..][0..4], &std.mem.toBytes(v));
                }
            }.f;
            // A slanted triangle (its hypotenuse is a diagonal edge) colored solid RED.
            const verts = [3][2]f32{ .{ -0.9, -0.9 }, .{ 0.9, -0.9 }, .{ -0.9, 0.9 } };
            inline for (verts, 0..) |v, k| {
                wf(vb, k * stride + 0, v[0]);
                wf(vb, k * stride + 4, v[1]);
                wf(vb, k * stride + 8, 1); // R
                wf(vb, k * stride + 12, 0); // G
                wf(vb, k * stride + 16, 0); // B
            }

            const ctx = try d.createContext();
            defer ctx.deinit();
            const cb = try ctx.beginCommands();
            defer cb.deinit();
            try cb.setRenderTarget(ms);
            try cb.clear(.{ .r = 0, .g = 0, .b = 0, .a = 1 }); // black bg
            try cb.bindPipeline(pipe);
            try cb.bindVertexBuffer(vbuf);
            try cb.draw(3, 0);
            if (samples > 1) try cb.resolve(ms, resolved, W, H, .rgba8_unorm, samples);
            try ctx.submit(cb);

            var out: [W * H * 4]u8 = undefined;
            // For 1x there is no resolve, so read the (single-sample) MSAA target directly.
            const src = try d.mapResource(if (samples > 1) resolved else ms);
            @memcpy(&out, src[0 .. W * H * 4]);
            return out;
        }
    }.go;

    const vs = try dev.createShaderModule(.{ .stage = .vertex, .code = std.mem.sliceAsBytes(vsb.words.items) });
    defer dev.destroyShaderModule(vs);
    const fs = try dev.createShaderModule(.{ .stage = .fragment, .code = std.mem.sliceAsBytes(fsb.words.items) });
    defer dev.destroyShaderModule(fs);
    const vs_mod: *sw_shader.ShaderModule = @ptrCast(@alignCast(vs));
    try std.testing.expect(vs_mod.compute_spirv != null);

    const px1 = try renderAt(dev, vs, fs, 1);
    const px4 = try renderAt(dev, vs, fs, 4);

    // Count partial-coverage RED pixels: the red channel is strictly between 0 (bg) and
    // 255 (full triangle). MSAA produces these along the diagonal edge (a sample-count
    // blend). The 1x render has hard edges (red is 0 or 255 only).
    var partial1: usize = 0;
    var partial4: usize = 0;
    var p: usize = 0;
    while (p < W * H) : (p += 1) {
        const r1 = px1[p * 4];
        const r4 = px4[p * 4];
        if (r1 > 10 and r1 < 245) partial1 += 1;
        if (r4 > 10 and r4 < 245) partial4 += 1;
    }
    // The MSAA render must have partial-coverage edge pixels. The 1x render must have
    // essentially none (hard edges). That gap is the anti-aliasing.
    try std.testing.expect(partial4 > 8);
    try std.testing.expect(partial4 > partial1 + 8);
}

test "depth test: the nearer triangle occludes the farther one regardless of draw order" {
    const gpa = std.testing.allocator;
    const Device = @import("device.zig").Device;
    const op = @import("vulcan-spirv").opcodes;
    const sc = op.StorageClass;

    // Passthrough VS that carries a per-vertex depth: in vec3 pos(loc0) [x,y,z]
    // + vec3 color(loc1) -> gl_Position = vec4(pos.x, pos.y, pos.z, 1); out vc(loc0).
    // ids: void1 fn2 f32:3 v3:4 v4:5 pInV3:6 pOutV4:7 pOutV3:8
    //      c1:17 pos:11 col:12 glpos:13 vcout:14 main:15 lbl:16
    //      pv:19 cv:20 x:21 y:22 z:23 gp:24.
    var vsb = try @import("vulcan-spirv").binary.Builder.init(gpa, 25);
    defer vsb.deinit(gpa);
    try vsb.emit(gpa, op.EntryPoint, &.{ op.ExecutionModel.vertex, 15, 0, 11, 12, 13, 14 });
    try vsb.emit(gpa, op.Decorate, &.{ 13, op.Decoration.builtin, op.BuiltIn.position });
    try vsb.emit(gpa, op.Decorate, &.{ 11, op.Decoration.location, 0 });
    try vsb.emit(gpa, op.Decorate, &.{ 12, op.Decoration.location, 1 });
    try vsb.emit(gpa, op.Decorate, &.{ 14, op.Decoration.location, 0 });
    try vsb.emit(gpa, op.TypeVoid, &.{1});
    try vsb.emit(gpa, op.TypeFunction, &.{ 2, 1 });
    try vsb.emit(gpa, op.TypeFloat, &.{ 3, 32 });
    try vsb.emit(gpa, op.TypeVector, &.{ 4, 3, 3 });
    try vsb.emit(gpa, op.TypeVector, &.{ 5, 3, 4 });
    try vsb.emit(gpa, op.TypePointer, &.{ 6, sc.input, 4 });
    try vsb.emit(gpa, op.TypePointer, &.{ 7, sc.output, 5 });
    try vsb.emit(gpa, op.TypePointer, &.{ 8, sc.output, 4 });
    try vsb.emit(gpa, op.Constant, &.{ 3, 17, @bitCast(@as(f32, 1.0)) });
    try vsb.emit(gpa, op.Variable, &.{ 6, 11, sc.input });
    try vsb.emit(gpa, op.Variable, &.{ 6, 12, sc.input });
    try vsb.emit(gpa, op.Variable, &.{ 7, 13, sc.output });
    try vsb.emit(gpa, op.Variable, &.{ 8, 14, sc.output });
    try vsb.emit(gpa, op.Function, &.{ 1, 15, 0, 2 });
    try vsb.emit(gpa, op.Label, &.{16});
    try vsb.emit(gpa, op.Load, &.{ 4, 19, 11 }); // pos
    try vsb.emit(gpa, op.Load, &.{ 4, 20, 12 }); // color
    try vsb.emit(gpa, op.CompositeExtract, &.{ 3, 21, 19, 0 }); // pos.x
    try vsb.emit(gpa, op.CompositeExtract, &.{ 3, 22, 19, 1 }); // pos.y
    try vsb.emit(gpa, op.CompositeExtract, &.{ 3, 23, 19, 2 }); // pos.z (the depth)
    try vsb.emit(gpa, op.CompositeConstruct, &.{ 5, 24, 21, 22, 23, 17 }); // (x,y,z,1)
    try vsb.emit(gpa, op.Store, &.{ 13, 24 });
    try vsb.emit(gpa, op.Store, &.{ 14, 20 });
    try vsb.emit(gpa, op.Return, &.{});
    try vsb.emit(gpa, op.FunctionEnd, &.{});

    // Passthrough FS: in vec3 vc(loc0) -> out vec4 o = vec4(vc, 1).
    var fsb = try @import("vulcan-spirv").binary.Builder.init(gpa, 18);
    defer fsb.deinit(gpa);
    try fsb.emit(gpa, op.EntryPoint, &.{ op.ExecutionModel.fragment, 10, 0, 8, 9 });
    try fsb.emit(gpa, op.Decorate, &.{ 8, op.Decoration.location, 0 });
    try fsb.emit(gpa, op.Decorate, &.{ 9, op.Decoration.location, 0 });
    try fsb.emit(gpa, op.TypeVoid, &.{1});
    try fsb.emit(gpa, op.TypeFunction, &.{ 2, 1 });
    try fsb.emit(gpa, op.TypeFloat, &.{ 3, 32 });
    try fsb.emit(gpa, op.TypeVector, &.{ 4, 3, 3 });
    try fsb.emit(gpa, op.TypeVector, &.{ 5, 3, 4 });
    try fsb.emit(gpa, op.TypePointer, &.{ 6, sc.input, 4 });
    try fsb.emit(gpa, op.TypePointer, &.{ 7, sc.output, 5 });
    try fsb.emit(gpa, op.Constant, &.{ 3, 12, @bitCast(@as(f32, 1.0)) });
    try fsb.emit(gpa, op.Variable, &.{ 6, 8, sc.input });
    try fsb.emit(gpa, op.Variable, &.{ 7, 9, sc.output });
    try fsb.emit(gpa, op.Function, &.{ 1, 10, 0, 2 });
    try fsb.emit(gpa, op.Label, &.{11});
    try fsb.emit(gpa, op.Load, &.{ 4, 13, 8 });
    try fsb.emit(gpa, op.CompositeExtract, &.{ 3, 14, 13, 0 });
    try fsb.emit(gpa, op.CompositeExtract, &.{ 3, 15, 13, 1 });
    try fsb.emit(gpa, op.CompositeExtract, &.{ 3, 16, 13, 2 });
    try fsb.emit(gpa, op.CompositeConstruct, &.{ 5, 17, 14, 15, 16, 12 });
    try fsb.emit(gpa, op.Store, &.{ 9, 17 });
    try fsb.emit(gpa, op.Return, &.{});
    try fsb.emit(gpa, op.FunctionEnd, &.{});

    const dev = try Device.create(gpa);
    defer dev.deinit();
    const vs = try dev.createShaderModule(.{ .stage = .vertex, .code = std.mem.sliceAsBytes(vsb.words.items) });
    defer dev.destroyShaderModule(vs);
    const fs = try dev.createShaderModule(.{ .stage = .fragment, .code = std.mem.sliceAsBytes(fsb.words.items) });
    defer dev.destroyShaderModule(fs);

    const W = 64;
    const H = 64;
    const stride: u32 = 24; // vec3 pos + vec3 color
    const attrs = [_]hal.VertexAttribute{
        .{ .location = 0, .format = .r32g32b32_float, .offset = 0 },
        .{ .location = 1, .format = .r32g32b32_float, .offset = 12 },
    };
    // Depth-tested pipeline: test LESS, write ON.
    const pipe = try dev.createPipeline(.{
        .vertex = vs,
        .fragment = fs,
        .vertex_layout = .{ .stride = stride, .attributes = &attrs },
        .color_format = .rgba8_unorm,
        .depth = .{ .test_enable = true, .write_enable = true, .compare_op = .less },
    });
    defer dev.destroyPipeline(pipe);
    const sw_pipe: *Pipeline = @ptrCast(@alignCast(pipe));
    try std.testing.expect(sw_pipe.program != null);

    const wf = struct {
        fn f(buf: []u8, off: usize, v: f32) void {
            @memcpy(buf[off..][0..4], &std.mem.toBytes(v));
        }
    }.f;
    // Far triangle (z=0.7, RED): covers the left + center region.
    const far_buf = try dev.createResource(.{ .buffer = .{ .size = 3 * stride } });
    defer dev.destroyResource(far_buf);
    {
        const b = try dev.mapResource(far_buf);
        // verts spanning x in [-0.9, 0.3], all z=0.7, color RED.
        const verts = [3][2]f32{ .{ -0.9, -0.9 }, .{ 0.3, -0.9 }, .{ -0.3, 0.9 } };
        inline for (verts, 0..) |v, k| {
            wf(b, k * stride + 0, v[0]);
            wf(b, k * stride + 4, v[1]);
            wf(b, k * stride + 8, 0.7);
            wf(b, k * stride + 12, 1);
            wf(b, k * stride + 16, 0);
            wf(b, k * stride + 20, 0);
        }
    }
    // Near triangle (z=0.3, GREEN): covers the center + right region.
    const near_buf = try dev.createResource(.{ .buffer = .{ .size = 3 * stride } });
    defer dev.destroyResource(near_buf);
    {
        const b = try dev.mapResource(near_buf);
        const verts = [3][2]f32{ .{ -0.3, -0.9 }, .{ 0.9, -0.9 }, .{ 0.3, 0.9 } };
        inline for (verts, 0..) |v, k| {
            wf(b, k * stride + 0, v[0]);
            wf(b, k * stride + 4, v[1]);
            wf(b, k * stride + 8, 0.3);
            wf(b, k * stride + 12, 0);
            wf(b, k * stride + 16, 1);
            wf(b, k * stride + 20, 0);
        }
    }

    // Render one pass with a given draw order. Return the color target pixels.
    const renderPass = struct {
        fn go(d: hal.Device, p: *hal.Pipeline, first: *hal.Resource, second: *hal.Resource) !struct { target: *hal.Resource, depth: *hal.Resource } {
            const target = try d.createResource(.{ .image = .{ .width = W, .height = H, .format = .rgba8_unorm } });
            const depth = try d.createResource(.{ .image = .{ .width = W, .height = H, .format = .depth32_float } });
            const ctx = try d.createContext();
            defer ctx.deinit();
            const cb = try ctx.beginCommands();
            defer cb.deinit();
            try cb.setRenderTarget(target);
            try cb.clear(.{ .r = 0, .g = 0, .b = 0, .a = 1 });
            try cb.setDepthTarget(depth, 1.0);
            try cb.bindPipeline(p);
            // First triangle.
            try cb.bindVertexBuffer(first);
            try cb.draw(3, 0);
            // Second triangle (same render pass, same depth buffer).
            try cb.bindVertexBuffer(second);
            try cb.draw(3, 0);
            try ctx.submit(cb);
            return .{ .target = target, .depth = depth };
        }
    }.go;

    // Sample on a row near the triangles' wide base where the two wide triangles
    // clearly overlap in the center and each covers its own side. The base verts are at
    // NDC y=-0.9. In Vulkan window space (y down, no flip) that maps near the top, so the
    // wide base + overlap is at the top of the image (row 16). The overlap pixel (x=32)
    // sits in both. Far-only (x=16) is only in the far (red) triangle. Near-only (x=48) is
    // only in the near (green) triangle.
    const row = 16;
    const overlap_i = (row * W + 32) * 4;
    const far_only_i = (row * W + 16) * 4;
    const near_only_i = (row * W + 48) * 4;

    // Pass 1: draw order FAR then NEAR.
    {
        const res = try renderPass(dev, pipe, far_buf, near_buf);
        defer dev.destroyResource(res.target);
        defer dev.destroyResource(res.depth);
        const px = try dev.mapResource(res.target);
        // Overlap -> GREEN (near wins).
        try std.testing.expect(px[overlap_i + 1] > px[overlap_i + 0]);
        try std.testing.expect(px[overlap_i + 1] > 100);
        // Far-only -> RED. Near-only -> GREEN.
        try std.testing.expect(px[far_only_i + 0] > px[far_only_i + 1]);
        try std.testing.expect(px[near_only_i + 1] > px[near_only_i + 0]);
    }

    // Pass 2: reverse draw order NEAR then FAR. The overlap must still be GREEN.
    // The far (red) fragments fail the less test against the already-written near
    // depth. Paint order would make it RED here. Real depth keeps it GREEN.
    {
        const res = try renderPass(dev, pipe, near_buf, far_buf);
        defer dev.destroyResource(res.target);
        defer dev.destroyResource(res.depth);
        const px = try dev.mapResource(res.target);
        try std.testing.expect(px[overlap_i + 1] > px[overlap_i + 0]);
        try std.testing.expect(px[overlap_i + 1] > 100);
        try std.testing.expect(px[far_only_i + 0] > px[far_only_i + 1]);
        try std.testing.expect(px[near_only_i + 1] > px[near_only_i + 0]);
    }
}

test "stencil test: a mask pass clips a later draw to only where the stencil was written" {
    const gpa = std.testing.allocator;
    const Device = @import("device.zig").Device;
    const op = @import("vulcan-spirv").opcodes;
    const sc = op.StorageClass;

    // Passthrough VS: in vec3 pos(loc0) + vec3 color(loc1) -> gl_Position=vec4(pos,1),
    // out vc(loc0). (Same shape as the depth test's VS.)
    var vsb = try @import("vulcan-spirv").binary.Builder.init(gpa, 25);
    defer vsb.deinit(gpa);
    try vsb.emit(gpa, op.EntryPoint, &.{ op.ExecutionModel.vertex, 15, 0, 11, 12, 13, 14 });
    try vsb.emit(gpa, op.Decorate, &.{ 13, op.Decoration.builtin, op.BuiltIn.position });
    try vsb.emit(gpa, op.Decorate, &.{ 11, op.Decoration.location, 0 });
    try vsb.emit(gpa, op.Decorate, &.{ 12, op.Decoration.location, 1 });
    try vsb.emit(gpa, op.Decorate, &.{ 14, op.Decoration.location, 0 });
    try vsb.emit(gpa, op.TypeVoid, &.{1});
    try vsb.emit(gpa, op.TypeFunction, &.{ 2, 1 });
    try vsb.emit(gpa, op.TypeFloat, &.{ 3, 32 });
    try vsb.emit(gpa, op.TypeVector, &.{ 4, 3, 3 });
    try vsb.emit(gpa, op.TypeVector, &.{ 5, 3, 4 });
    try vsb.emit(gpa, op.TypePointer, &.{ 6, sc.input, 4 });
    try vsb.emit(gpa, op.TypePointer, &.{ 7, sc.output, 5 });
    try vsb.emit(gpa, op.TypePointer, &.{ 8, sc.output, 4 });
    try vsb.emit(gpa, op.Constant, &.{ 3, 17, @bitCast(@as(f32, 1.0)) });
    try vsb.emit(gpa, op.Variable, &.{ 6, 11, sc.input });
    try vsb.emit(gpa, op.Variable, &.{ 6, 12, sc.input });
    try vsb.emit(gpa, op.Variable, &.{ 7, 13, sc.output });
    try vsb.emit(gpa, op.Variable, &.{ 8, 14, sc.output });
    try vsb.emit(gpa, op.Function, &.{ 1, 15, 0, 2 });
    try vsb.emit(gpa, op.Label, &.{16});
    try vsb.emit(gpa, op.Load, &.{ 4, 19, 11 });
    try vsb.emit(gpa, op.Load, &.{ 4, 20, 12 });
    try vsb.emit(gpa, op.CompositeExtract, &.{ 3, 21, 19, 0 });
    try vsb.emit(gpa, op.CompositeExtract, &.{ 3, 22, 19, 1 });
    try vsb.emit(gpa, op.CompositeExtract, &.{ 3, 23, 19, 2 });
    try vsb.emit(gpa, op.CompositeConstruct, &.{ 5, 24, 21, 22, 23, 17 });
    try vsb.emit(gpa, op.Store, &.{ 13, 24 });
    try vsb.emit(gpa, op.Store, &.{ 14, 20 });
    try vsb.emit(gpa, op.Return, &.{});
    try vsb.emit(gpa, op.FunctionEnd, &.{});

    // Passthrough FS: in vec3 vc(loc0) -> out vec4 o = vec4(vc, 1).
    var fsb = try @import("vulcan-spirv").binary.Builder.init(gpa, 18);
    defer fsb.deinit(gpa);
    try fsb.emit(gpa, op.EntryPoint, &.{ op.ExecutionModel.fragment, 10, 0, 8, 9 });
    try fsb.emit(gpa, op.Decorate, &.{ 8, op.Decoration.location, 0 });
    try fsb.emit(gpa, op.Decorate, &.{ 9, op.Decoration.location, 0 });
    try fsb.emit(gpa, op.TypeVoid, &.{1});
    try fsb.emit(gpa, op.TypeFunction, &.{ 2, 1 });
    try fsb.emit(gpa, op.TypeFloat, &.{ 3, 32 });
    try fsb.emit(gpa, op.TypeVector, &.{ 4, 3, 3 });
    try fsb.emit(gpa, op.TypeVector, &.{ 5, 3, 4 });
    try fsb.emit(gpa, op.TypePointer, &.{ 6, sc.input, 4 });
    try fsb.emit(gpa, op.TypePointer, &.{ 7, sc.output, 5 });
    try fsb.emit(gpa, op.Constant, &.{ 3, 12, @bitCast(@as(f32, 1.0)) });
    try fsb.emit(gpa, op.Variable, &.{ 6, 8, sc.input });
    try fsb.emit(gpa, op.Variable, &.{ 7, 9, sc.output });
    try fsb.emit(gpa, op.Function, &.{ 1, 10, 0, 2 });
    try fsb.emit(gpa, op.Label, &.{11});
    try fsb.emit(gpa, op.Load, &.{ 4, 13, 8 });
    try fsb.emit(gpa, op.CompositeExtract, &.{ 3, 14, 13, 0 });
    try fsb.emit(gpa, op.CompositeExtract, &.{ 3, 15, 13, 1 });
    try fsb.emit(gpa, op.CompositeExtract, &.{ 3, 16, 13, 2 });
    try fsb.emit(gpa, op.CompositeConstruct, &.{ 5, 17, 14, 15, 16, 12 });
    try fsb.emit(gpa, op.Store, &.{ 9, 17 });
    try fsb.emit(gpa, op.Return, &.{});
    try fsb.emit(gpa, op.FunctionEnd, &.{});

    const dev = try Device.create(gpa);
    defer dev.deinit();
    const vs = try dev.createShaderModule(.{ .stage = .vertex, .code = std.mem.sliceAsBytes(vsb.words.items) });
    defer dev.destroyShaderModule(vs);
    const fs = try dev.createShaderModule(.{ .stage = .fragment, .code = std.mem.sliceAsBytes(fsb.words.items) });
    defer dev.destroyShaderModule(fs);

    const W = 64;
    const H = 64;
    const stride: u32 = 24;
    const attrs = [_]hal.VertexAttribute{
        .{ .location = 0, .format = .r32g32b32_float, .offset = 0 },
        .{ .location = 1, .format = .r32g32b32_float, .offset = 12 },
    };
    // Mask pipeline: stencil always passes, writes reference=1 (replace). Color is drawn
    // too (black), but the point is the stencil side effect.
    const mask_pipe = try dev.createPipeline(.{
        .vertex = vs,
        .fragment = fs,
        .vertex_layout = .{ .stride = stride, .attributes = &attrs },
        .color_format = .rgba8_unorm,
        .stencil = .{ .test_enable = true, .compare_op = .always, .pass_op = .replace, .reference = 1 },
    });
    defer dev.destroyPipeline(mask_pipe);
    // Content pipeline: stencil passes only where stored == 1 (the mask), keep.
    const draw_pipe = try dev.createPipeline(.{
        .vertex = vs,
        .fragment = fs,
        .vertex_layout = .{ .stride = stride, .attributes = &attrs },
        .color_format = .rgba8_unorm,
        .stencil = .{ .test_enable = true, .compare_op = .equal, .pass_op = .keep, .reference = 1 },
    });
    defer dev.destroyPipeline(draw_pipe);

    const wf = struct {
        fn f(buf: []u8, off: usize, v: f32) void {
            @memcpy(buf[off..][0..4], &std.mem.toBytes(v));
        }
        fn quad(buf: []u8, st: u32, x0: f32, x1: f32, r: f32, g: f32, b: f32) void {
            // Two triangles spanning [x0,x1] x [-1,1] at z=0.5, given color.
            const pos = [6][2]f32{ .{ x0, -1 }, .{ x1, -1 }, .{ x1, 1 }, .{ x0, -1 }, .{ x1, 1 }, .{ x0, 1 } };
            inline for (pos, 0..) |p, k| {
                f(buf, k * st + 0, p[0]);
                f(buf, k * st + 4, p[1]);
                f(buf, k * st + 8, 0.5);
                f(buf, k * st + 12, r);
                f(buf, k * st + 16, g);
                f(buf, k * st + 20, b);
            }
        }
    };
    // Mask: the left half (x in [-1,0]), color black.
    const mask_vb = try dev.createResource(.{ .buffer = .{ .size = 6 * stride } });
    defer dev.destroyResource(mask_vb);
    wf.quad(try dev.mapResource(mask_vb), stride, -1.0, 0.0, 0, 0, 0);
    // Content: the full screen (x in [-1,1]), color RED.
    const content_vb = try dev.createResource(.{ .buffer = .{ .size = 6 * stride } });
    defer dev.destroyResource(content_vb);
    wf.quad(try dev.mapResource(content_vb), stride, -1.0, 1.0, 1, 0, 0);

    const target = try dev.createResource(.{ .image = .{ .width = W, .height = H, .format = .rgba8_unorm } });
    defer dev.destroyResource(target);
    // The stencil buffer: one u8 per pixel (a plain buffer resource).
    const stencil = try dev.createResource(.{ .buffer = .{ .size = W * H } });
    defer dev.destroyResource(stencil);

    const ctx = try dev.createContext();
    defer ctx.deinit();
    const cb = try ctx.beginCommands();
    defer cb.deinit();
    try cb.setRenderTarget(target);
    try cb.clear(.{ .r = 0, .g = 0, .b = 0, .a = 1 });
    try cb.setStencilTarget(stencil, 0); // clear stencil to 0
    // Mask pass: write stencil=1 over the left half.
    try cb.bindPipeline(mask_pipe);
    try cb.bindVertexBuffer(mask_vb);
    try cb.draw(6, 0);
    // Content pass: red, clipped to where stencil==1 (the left half).
    try cb.bindPipeline(draw_pipe);
    try cb.bindVertexBuffer(content_vb);
    try cb.draw(6, 0);
    try ctx.submit(cb);

    const px = try dev.mapResource(target);
    // NDC x in [-1,0] maps to window x in [0,32]. Left pixel (16,32) is inside the mask:
    // RED. Right pixel (48,32) is outside the mask: stencil==0, content clipped -> BLACK.
    const left_i = (32 * W + 16) * 4;
    const right_i = (32 * W + 48) * 4;
    try std.testing.expect(px[left_i + 0] > 200 and px[left_i + 1] < 40 and px[left_i + 2] < 40); // red
    try std.testing.expect(px[right_i + 0] < 40 and px[right_i + 1] < 40 and px[right_i + 2] < 40); // black
    // And the stencil buffer itself: 1 on the left, 0 on the right.
    const sb = try dev.mapResource(stencil);
    try std.testing.expectEqual(@as(u8, 1), sb[32 * W + 16]);
    try std.testing.expectEqual(@as(u8, 0), sb[32 * W + 48]);
}

test "depth bias (glPolygonOffset): a negative constant offset lets a coplanar draw win the LESS test" {
    const glsl = @import("../../glsl.zig");
    const Device = @import("device.zig").Device;
    const gpa = std.testing.allocator;
    const W = 64;
    const H = 64;

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
    const dev = try Device.create(gpa);
    defer dev.deinit();
    const vs = try dev.createShaderModule(.{ .stage = .vertex, .code = vs_bytes });
    defer dev.destroyShaderModule(vs);
    const fs = try dev.createShaderModule(.{ .stage = .fragment, .code = fs_bytes });
    defer dev.destroyShaderModule(fs);

    const stride: u32 = 24;
    const attrs = [_]hal.VertexAttribute{
        .{ .location = 0, .format = .r32g32b32_float, .offset = 0 },
        .{ .location = 1, .format = .r32g32b32_float, .offset = 12 },
    };
    const depth_ls = hal.DepthState{ .test_enable = true, .write_enable = true, .compare_op = .less };
    // First (red) writes depth 0.5. Second (green) is coplanar at z=0.5. With less it would fail
    // (0.5 < 0.5 is false) and stay red. A negative constant bias (-20000 * DEPTH_BIAS_R = -0.305)
    // pulls green to z~0.195 < 0.5 so it passes and wins. The control pipeline has no bias.
    const base_pipe = try dev.createPipeline(.{
        .vertex = vs,
        .fragment = fs,
        .vertex_layout = .{ .stride = stride, .attributes = &attrs },
        .color_format = .rgba8_unorm,
        .depth = depth_ls,
    });
    defer dev.destroyPipeline(base_pipe);
    var biased = depth_ls;
    biased.bias_enable = true;
    biased.bias_constant = -20000;
    const biased_pipe = try dev.createPipeline(.{
        .vertex = vs,
        .fragment = fs,
        .vertex_layout = .{ .stride = stride, .attributes = &attrs },
        .color_format = .rgba8_unorm,
        .depth = biased,
    });
    defer dev.destroyPipeline(biased_pipe);

    const wf = struct {
        fn f(buf: []u8, off: usize, v: f32) void {
            @memcpy(buf[off..][0..4], &std.mem.toBytes(v));
        }
        fn quad(buf: []u8, st: u32, r: f32, g: f32, b: f32) void {
            const pos = [6][2]f32{ .{ -1, -1 }, .{ 1, -1 }, .{ 1, 1 }, .{ -1, -1 }, .{ 1, 1 }, .{ -1, 1 } };
            inline for (pos, 0..) |p, k| {
                f(buf, k * st + 0, p[0]);
                f(buf, k * st + 4, p[1]);
                f(buf, k * st + 8, 0.5); // all at z=0.5 (coplanar)
                f(buf, k * st + 12, r);
                f(buf, k * st + 16, g);
                f(buf, k * st + 20, b);
            }
        }
    };
    const red_vb = try dev.createResource(.{ .buffer = .{ .size = 6 * stride } });
    defer dev.destroyResource(red_vb);
    wf.quad(try dev.mapResource(red_vb), stride, 1, 0, 0);
    const green_vb = try dev.createResource(.{ .buffer = .{ .size = 6 * stride } });
    defer dev.destroyResource(green_vb);
    wf.quad(try dev.mapResource(green_vb), stride, 0, 1, 0);

    const ctx = try dev.createContext();
    defer ctx.deinit();

    // Render with green BIASED: green wins (pulled in front). Depth is a fresh buffer cleared to 1.
    const render = struct {
        fn f(d: hal.Device, c: hal.Context, green_pipe: *hal.Pipeline, base_p: *hal.Pipeline, rvb: *hal.Resource, gvb: *hal.Resource) !u32 {
            const target = try d.createResource(.{ .image = .{ .width = W, .height = H, .format = .rgba8_unorm } });
            defer d.destroyResource(target);
            const depth = try d.createResource(.{ .buffer = .{ .size = W * H * 4 } });
            defer d.destroyResource(depth);
            const cb = try c.beginCommands();
            defer cb.deinit();
            try cb.setRenderTarget(target);
            try cb.clear(.{ .r = 0, .g = 0, .b = 0, .a = 1 });
            try cb.setDepthTarget(depth, 1.0);
            try cb.bindPipeline(base_p);
            try cb.bindVertexBuffer(rvb);
            try cb.draw(6, 0);
            try cb.bindPipeline(green_pipe);
            try cb.bindVertexBuffer(gvb);
            try cb.draw(6, 0);
            try c.submit(cb);
            const px = try d.mapResource(target);
            return std.mem.bytesToValue(u32, px[(32 * W + 32) * 4 ..][0..4]);
        }
    }.f;
    const biased_center = try render(dev, ctx, biased_pipe, base_pipe, red_vb, green_vb);
    const control_center = try render(dev, ctx, base_pipe, base_pipe, red_vb, green_vb);

    // rgba8: byte0=R, byte1=G, byte2=B.
    const g_of = struct {
        fn f(p: u32) u8 {
            return @truncate(p >> 8);
        }
        fn r(p: u32) u8 {
            return @truncate(p);
        }
    };
    // Biased: green won (pulled in front of red). Control: green failed LESS -> red stayed.
    try std.testing.expect(g_of.f(biased_center) > 200 and g_of.r(biased_center) < 60);
    try std.testing.expect(g_of.r(control_center) > 200 and g_of.f(control_center) < 60);
}

test "color write mask (glColorMask): masked channels keep the destination" {
    const glsl = @import("../../glsl.zig");
    const Device = @import("device.zig").Device;
    const gpa = std.testing.allocator;
    const W = 64;
    const H = 64;

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
    const dev = try Device.create(gpa);
    defer dev.deinit();
    const vs = try dev.createShaderModule(.{ .stage = .vertex, .code = vs_bytes });
    defer dev.destroyShaderModule(vs);
    const fs = try dev.createShaderModule(.{ .stage = .fragment, .code = fs_bytes });
    defer dev.destroyShaderModule(fs);

    const stride: u32 = 24;
    const attrs = [_]hal.VertexAttribute{
        .{ .location = 0, .format = .r32g32b32_float, .offset = 0 },
        .{ .location = 1, .format = .r32g32b32_float, .offset = 12 },
    };
    // Only R and A write. G and B are masked (keep the black clear). A WHITE draw -> RED result.
    const pipe = try dev.createPipeline(.{
        .vertex = vs,
        .fragment = fs,
        .vertex_layout = .{ .stride = stride, .attributes = &attrs },
        .color_format = .rgba8_unorm,
        .blend = .{ .write_mask = .{ true, false, false, true } },
    });
    defer dev.destroyPipeline(pipe);

    const vb = try dev.createResource(.{ .buffer = .{ .size = 6 * stride } });
    defer dev.destroyResource(vb);
    {
        const buf = try dev.mapResource(vb);
        const pos = [6][2]f32{ .{ -1, -1 }, .{ 1, -1 }, .{ 1, 1 }, .{ -1, -1 }, .{ 1, 1 }, .{ -1, 1 } };
        const wr = struct {
            fn f(b: []u8, off: usize, v: f32) void {
                @memcpy(b[off..][0..4], &std.mem.toBytes(v));
            }
        };
        inline for (pos, 0..) |p, k| {
            wr.f(buf, k * stride + 0, p[0]);
            wr.f(buf, k * stride + 4, p[1]);
            wr.f(buf, k * stride + 8, 0.0);
            wr.f(buf, k * stride + 12, 1.0); // white
            wr.f(buf, k * stride + 16, 1.0);
            wr.f(buf, k * stride + 20, 1.0);
        }
    }
    const target = try dev.createResource(.{ .image = .{ .width = W, .height = H, .format = .rgba8_unorm } });
    defer dev.destroyResource(target);

    const ctx = try dev.createContext();
    defer ctx.deinit();
    const cb = try ctx.beginCommands();
    defer cb.deinit();
    try cb.setRenderTarget(target);
    try cb.clear(.{ .r = 0, .g = 0, .b = 0, .a = 1 });
    try cb.bindPipeline(pipe);
    try cb.bindVertexBuffer(vb);
    try cb.draw(6, 0);
    try ctx.submit(cb);

    const px = try dev.mapResource(target);
    const ci = (32 * W + 32) * 4;
    // White written through an R+A mask -> R=255 (written), G=0, B=0 (masked, stayed black) = RED.
    try std.testing.expect(px[ci + 0] > 200 and px[ci + 1] < 40 and px[ci + 2] < 40);
}

test "vertex-pulling: a VS with gl_VertexIndex pulls its triangle from a UBO (no vertex buffer)" {
    const gpa = std.testing.allocator;
    const Device = @import("device.zig").Device;
    const op = @import("vulcan-spirv").opcodes;
    const sc = op.StorageClass;

    // The vkcube-shaped vertex-pulling VS (ZERO vertex attributes):
    //   layout(binding=0) uniform U { vec4 pos[3]; vec4 col[3]; } u;
    //   layout(location=0) out vec3 vc;
    //   void main(){ gl_Position = u.pos[gl_VertexIndex]; vc = u.col[gl_VertexIndex].rgb; }
    // ids: void1 f32:2 v4f:3 v3f:4 int:5 i0:6 i1:7 uint:8 u36:9 arr:10 U:11 pUniU:12
    //      u:13 pUniV4:14 pInInt:15 vidx:16 gpvS:17 pOutGpv:18 gpv:19 pOutV3:20 vc:21
    //      pOutV4:22 fn:23 main:24 entry:25 vi:26 acP:27 pull:28 acGP:29 viB:30 acC:31
    //      colv:32 col3:33.
    var vsb = try @import("vulcan-spirv").binary.Builder.init(gpa, 34);
    defer vsb.deinit(gpa);
    try vsb.emit(gpa, op.EntryPoint, &.{ op.ExecutionModel.vertex, 24, 0, 16, 19, 21 });
    try vsb.emit(gpa, op.Decorate, &.{ 11, op.Decoration.block });
    try vsb.emit(gpa, op.MemberDecorate, &.{ 11, 0, op.Decoration.offset, 0 });
    try vsb.emit(gpa, op.MemberDecorate, &.{ 11, 1, op.Decoration.offset, 48 });
    try vsb.emit(gpa, op.Decorate, &.{ 10, op.Decoration.array_stride, 16 });
    try vsb.emit(gpa, op.Decorate, &.{ 13, op.Decoration.binding, 0 });
    try vsb.emit(gpa, op.Decorate, &.{ 16, op.Decoration.builtin, op.BuiltIn.vertex_index });
    try vsb.emit(gpa, op.Decorate, &.{ 17, op.Decoration.block });
    try vsb.emit(gpa, op.MemberDecorate, &.{ 17, 0, op.Decoration.builtin, op.BuiltIn.position });
    try vsb.emit(gpa, op.Decorate, &.{ 21, op.Decoration.location, 0 });
    try vsb.emit(gpa, op.TypeVoid, &.{1});
    try vsb.emit(gpa, op.TypeFloat, &.{ 2, 32 });
    try vsb.emit(gpa, op.TypeVector, &.{ 3, 2, 4 });
    try vsb.emit(gpa, op.TypeVector, &.{ 4, 2, 3 });
    try vsb.emit(gpa, op.TypeInt, &.{ 5, 32, 1 });
    try vsb.emit(gpa, op.Constant, &.{ 5, 6, 0 });
    try vsb.emit(gpa, op.Constant, &.{ 5, 7, 1 });
    try vsb.emit(gpa, op.TypeInt, &.{ 8, 32, 0 });
    try vsb.emit(gpa, op.Constant, &.{ 8, 9, 3 });
    try vsb.emit(gpa, op.TypeArray, &.{ 10, 3, 9 }); // vec4[3]
    try vsb.emit(gpa, op.TypeStruct, &.{ 11, 10, 10 }); // U { vec4 pos[3]; vec4 col[3]; }
    try vsb.emit(gpa, op.TypePointer, &.{ 12, sc.uniform, 11 });
    try vsb.emit(gpa, op.Variable, &.{ 12, 13, sc.uniform });
    try vsb.emit(gpa, op.TypePointer, &.{ 14, sc.uniform, 3 });
    try vsb.emit(gpa, op.TypePointer, &.{ 15, sc.input, 5 });
    try vsb.emit(gpa, op.Variable, &.{ 15, 16, sc.input }); // gl_VertexIndex
    try vsb.emit(gpa, op.TypeStruct, &.{ 17, 3 });
    try vsb.emit(gpa, op.TypePointer, &.{ 18, sc.output, 17 });
    try vsb.emit(gpa, op.Variable, &.{ 18, 19, sc.output }); // gl_PerVertex
    try vsb.emit(gpa, op.TypePointer, &.{ 20, sc.output, 4 });
    try vsb.emit(gpa, op.Variable, &.{ 20, 21, sc.output }); // vc
    try vsb.emit(gpa, op.TypePointer, &.{ 22, sc.output, 3 });
    try vsb.emit(gpa, op.TypeFunction, &.{ 23, 1 });
    try vsb.emit(gpa, op.Function, &.{ 1, 24, 0, 23 });
    try vsb.emit(gpa, op.Label, &.{25});
    try vsb.emit(gpa, op.Load, &.{ 5, 26, 16 }); // vi
    try vsb.emit(gpa, op.AccessChain, &.{ 14, 27, 13, 6, 26 }); // &u.pos[vi]
    try vsb.emit(gpa, op.Load, &.{ 3, 28, 27 }); // pos vec4
    try vsb.emit(gpa, op.AccessChain, &.{ 22, 29, 19, 6 }); // &gl_Position
    try vsb.emit(gpa, op.Store, &.{ 29, 28 });
    try vsb.emit(gpa, op.Load, &.{ 5, 30, 16 }); // vi again
    try vsb.emit(gpa, op.AccessChain, &.{ 14, 31, 13, 7, 30 }); // &u.col[vi]
    try vsb.emit(gpa, op.Load, &.{ 3, 32, 31 }); // col vec4
    try vsb.emit(gpa, op.VectorShuffle, &.{ 4, 33, 32, 32, 0, 1, 2 }); // col.rgb
    try vsb.emit(gpa, op.Store, &.{ 21, 33 });
    try vsb.emit(gpa, op.Return, &.{});
    try vsb.emit(gpa, op.FunctionEnd, &.{});

    // Passthrough FS: in vec3 vc(loc0) -> out vec4 o = vec4(vc, 1).
    var fsb = try @import("vulcan-spirv").binary.Builder.init(gpa, 18);
    defer fsb.deinit(gpa);
    try fsb.emit(gpa, op.EntryPoint, &.{ op.ExecutionModel.fragment, 10, 0, 8, 9 });
    try fsb.emit(gpa, op.Decorate, &.{ 8, op.Decoration.location, 0 });
    try fsb.emit(gpa, op.Decorate, &.{ 9, op.Decoration.location, 0 });
    try fsb.emit(gpa, op.TypeVoid, &.{1});
    try fsb.emit(gpa, op.TypeFunction, &.{ 2, 1 });
    try fsb.emit(gpa, op.TypeFloat, &.{ 3, 32 });
    try fsb.emit(gpa, op.TypeVector, &.{ 4, 3, 3 });
    try fsb.emit(gpa, op.TypeVector, &.{ 5, 3, 4 });
    try fsb.emit(gpa, op.TypePointer, &.{ 6, sc.input, 4 });
    try fsb.emit(gpa, op.TypePointer, &.{ 7, sc.output, 5 });
    try fsb.emit(gpa, op.Constant, &.{ 3, 12, @bitCast(@as(f32, 1.0)) });
    try fsb.emit(gpa, op.Variable, &.{ 6, 8, sc.input });
    try fsb.emit(gpa, op.Variable, &.{ 7, 9, sc.output });
    try fsb.emit(gpa, op.Function, &.{ 1, 10, 0, 2 });
    try fsb.emit(gpa, op.Label, &.{11});
    try fsb.emit(gpa, op.Load, &.{ 4, 13, 8 });
    try fsb.emit(gpa, op.CompositeExtract, &.{ 3, 14, 13, 0 });
    try fsb.emit(gpa, op.CompositeExtract, &.{ 3, 15, 13, 1 });
    try fsb.emit(gpa, op.CompositeExtract, &.{ 3, 16, 13, 2 });
    try fsb.emit(gpa, op.CompositeConstruct, &.{ 5, 17, 14, 15, 16, 12 });
    try fsb.emit(gpa, op.Store, &.{ 9, 17 });
    try fsb.emit(gpa, op.Return, &.{});
    try fsb.emit(gpa, op.FunctionEnd, &.{});

    const dev = try Device.create(gpa);
    defer dev.deinit();
    const vs = try dev.createShaderModule(.{ .stage = .vertex, .code = std.mem.sliceAsBytes(vsb.words.items) });
    defer dev.destroyShaderModule(vs);
    const fs = try dev.createShaderModule(.{ .stage = .fragment, .code = std.mem.sliceAsBytes(fsb.words.items) });
    defer dev.destroyShaderModule(fs);

    const W = 64;
    const H = 64;
    // Zero vertex attributes. The VS pulls from the UBO by gl_VertexIndex.
    const pipe = try dev.createPipeline(.{
        .vertex = vs,
        .fragment = fs,
        .vertex_layout = .{ .stride = 0, .attributes = &.{} },
        .color_format = .rgba8_unorm,
    });
    defer dev.destroyPipeline(pipe);
    const sw_pipe: *Pipeline = @ptrCast(@alignCast(pipe));
    try std.testing.expect(sw_pipe.program != null); // the VS+FS JITed
    try std.testing.expect(sw_pipe.program.?.vs_index_count == 1); // gl_VertexIndex

    // UBO: pos[3] (std140, offset 0) covering the FB + col[3] (offset 48) R/G/B.
    const ubo = try dev.createResource(.{ .buffer = .{ .size = 96 } });
    defer dev.destroyResource(ubo);
    const ub = try dev.mapResource(ubo);
    const wf = struct {
        fn f(buf: []u8, off: usize, v: f32) void {
            @memcpy(buf[off..][0..4], &std.mem.toBytes(v));
        }
    }.f;
    const pos = [3][4]f32{ .{ -0.8, -0.8, 0, 1 }, .{ 0.8, -0.8, 0, 1 }, .{ 0.0, 0.8, 0, 1 } };
    inline for (pos, 0..) |p, k| inline for (0..4) |c| wf(ub, k * 16 + c * 4, p[c]);
    const col = [3][4]f32{ .{ 1, 0, 0, 1 }, .{ 0, 1, 0, 1 }, .{ 0, 0, 1, 1 } };
    inline for (col, 0..) |cc, k| inline for (0..4) |c| wf(ub, 48 + k * 16 + c * 4, cc[c]);

    const target = try dev.createResource(.{ .image = .{ .width = W, .height = H, .format = .rgba8_unorm } });
    defer dev.destroyResource(target);
    const tpix = try dev.mapResource(target);

    const ctx = try dev.createContext();
    defer ctx.deinit();
    const cb = try ctx.beginCommands();
    defer cb.deinit();
    try cb.setRenderTarget(target);
    try cb.clear(.{ .r = 0, .g = 0, .b = 0, .a = 1 });
    try cb.bindPipeline(pipe);
    try cb.bindUniformBuffer(0, ubo);
    // No bindVertexBuffer. The VS pulls from the UBO by gl_VertexIndex.
    try cb.draw(3, 0);
    try ctx.submit(cb);

    // Center is covered (interpolated R/G/B). A top corner is the black clear.
    const ci = (36 * W + 32) * 4;
    try std.testing.expect(tpix[ci + 0] != 0 or tpix[ci + 1] != 0 or tpix[ci + 2] != 0);
    const corner = (1 * W + 1) * 4;
    try std.testing.expectEqual(@as(u8, 0), tpix[corner + 0]);
    try std.testing.expectEqual(@as(u8, 0), tpix[corner + 1]);
    try std.testing.expectEqual(@as(u8, 0), tpix[corner + 2]);
    // Near the RED vertex red dominates, which proves the per-vertex UBO pull. That vertex is at
    // NDC (-0.8, -0.8). In Vulkan window space (y down, no flip) that is the top-left,
    // so sample near the top-left (row 12, col 12).
    const rr = (12 * W + 12) * 4;
    try std.testing.expect(tpix[rr + 0] > tpix[rr + 1] and tpix[rr + 0] > tpix[rr + 2]);
}

test "instancing: gl_InstanceIndex offsets each instance, drawInstanced renders all N" {
    const gpa = std.testing.allocator;
    const Device = @import("device.zig").Device;
    const op = @import("vulcan-spirv").opcodes;
    const sc = op.StorageClass;

    // A vertex-pulling + INSTANCED VS (zero vertex attributes):
    //   layout(binding=0) uniform U { vec4 pos[3]; vec4 col[3]; vec4 offset[2]; } u;
    //   layout(location=0) out vec3 vc;
    //   void main(){ gl_Position = u.pos[gl_VertexIndex] + u.offset[gl_InstanceIndex];
    //                vc = u.col[gl_VertexIndex].rgb; }
    // The per-instance offset[gl_InstanceIndex] is the UI instancing pattern (per-instance
    // data in a UBO, indexed by the instance), so the SAME 3 base verts draw at N places.
    var vsb = try @import("vulcan-spirv").binary.Builder.init(gpa, 42);
    defer vsb.deinit(gpa);
    try vsb.emit(gpa, op.EntryPoint, &.{ op.ExecutionModel.vertex, 28, 0, 19, 20, 23, 25 });
    try vsb.emit(gpa, op.Decorate, &.{ 11, op.Decoration.block });
    try vsb.emit(gpa, op.MemberDecorate, &.{ 11, 0, op.Decoration.offset, 0 });
    try vsb.emit(gpa, op.MemberDecorate, &.{ 11, 1, op.Decoration.offset, 48 });
    try vsb.emit(gpa, op.MemberDecorate, &.{ 11, 2, op.Decoration.offset, 96 });
    try vsb.emit(gpa, op.Decorate, &.{ 10, op.Decoration.array_stride, 16 });
    try vsb.emit(gpa, op.Decorate, &.{ 14, op.Decoration.array_stride, 16 });
    try vsb.emit(gpa, op.Decorate, &.{ 16, op.Decoration.binding, 0 });
    try vsb.emit(gpa, op.Decorate, &.{ 19, op.Decoration.builtin, op.BuiltIn.vertex_index });
    try vsb.emit(gpa, op.Decorate, &.{ 20, op.Decoration.builtin, op.BuiltIn.instance_index });
    try vsb.emit(gpa, op.Decorate, &.{ 21, op.Decoration.block });
    try vsb.emit(gpa, op.MemberDecorate, &.{ 21, 0, op.Decoration.builtin, op.BuiltIn.position });
    try vsb.emit(gpa, op.Decorate, &.{ 25, op.Decoration.location, 0 });
    try vsb.emit(gpa, op.TypeVoid, &.{1});
    try vsb.emit(gpa, op.TypeFloat, &.{ 2, 32 });
    try vsb.emit(gpa, op.TypeVector, &.{ 3, 2, 4 }); // vec4
    try vsb.emit(gpa, op.TypeVector, &.{ 4, 2, 3 }); // vec3
    try vsb.emit(gpa, op.TypeInt, &.{ 5, 32, 1 });
    try vsb.emit(gpa, op.Constant, &.{ 5, 6, 0 });
    try vsb.emit(gpa, op.Constant, &.{ 5, 7, 1 });
    try vsb.emit(gpa, op.TypeInt, &.{ 8, 32, 0 });
    try vsb.emit(gpa, op.Constant, &.{ 8, 9, 3 });
    try vsb.emit(gpa, op.TypeArray, &.{ 10, 3, 9 }); // vec4[3]
    try vsb.emit(gpa, op.Constant, &.{ 5, 12, 2 }); // int 2 (offset member index)
    try vsb.emit(gpa, op.Constant, &.{ 8, 13, 2 }); // uint 2 (offset array size)
    try vsb.emit(gpa, op.TypeArray, &.{ 14, 3, 13 }); // vec4[2]
    try vsb.emit(gpa, op.TypeStruct, &.{ 11, 10, 10, 14 }); // U { pos[3]; col[3]; offset[2]; }
    try vsb.emit(gpa, op.TypePointer, &.{ 15, sc.uniform, 11 });
    try vsb.emit(gpa, op.Variable, &.{ 15, 16, sc.uniform });
    try vsb.emit(gpa, op.TypePointer, &.{ 17, sc.uniform, 3 });
    try vsb.emit(gpa, op.TypePointer, &.{ 18, sc.input, 5 });
    try vsb.emit(gpa, op.Variable, &.{ 18, 19, sc.input }); // gl_VertexIndex
    try vsb.emit(gpa, op.Variable, &.{ 18, 20, sc.input }); // gl_InstanceIndex
    try vsb.emit(gpa, op.TypeStruct, &.{ 21, 3 });
    try vsb.emit(gpa, op.TypePointer, &.{ 22, sc.output, 21 });
    try vsb.emit(gpa, op.Variable, &.{ 22, 23, sc.output }); // gl_PerVertex
    try vsb.emit(gpa, op.TypePointer, &.{ 24, sc.output, 4 });
    try vsb.emit(gpa, op.Variable, &.{ 24, 25, sc.output }); // vc
    try vsb.emit(gpa, op.TypePointer, &.{ 26, sc.output, 3 });
    try vsb.emit(gpa, op.TypeFunction, &.{ 27, 1 });
    try vsb.emit(gpa, op.Function, &.{ 1, 28, 0, 27 });
    try vsb.emit(gpa, op.Label, &.{29});
    try vsb.emit(gpa, op.Load, &.{ 5, 30, 19 }); // vi
    try vsb.emit(gpa, op.AccessChain, &.{ 17, 31, 16, 6, 30 }); // &u.pos[vi]
    try vsb.emit(gpa, op.Load, &.{ 3, 32, 31 }); // pos
    try vsb.emit(gpa, op.Load, &.{ 5, 33, 20 }); // ii (gl_InstanceIndex)
    try vsb.emit(gpa, op.AccessChain, &.{ 17, 34, 16, 12, 33 }); // &u.offset[ii]
    try vsb.emit(gpa, op.Load, &.{ 3, 35, 34 }); // offset
    try vsb.emit(gpa, op.FAdd, &.{ 3, 36, 32, 35 }); // pos + offset
    try vsb.emit(gpa, op.AccessChain, &.{ 26, 37, 23, 6 }); // &gl_Position
    try vsb.emit(gpa, op.Store, &.{ 37, 36 });
    try vsb.emit(gpa, op.Load, &.{ 5, 38, 19 }); // vi again
    try vsb.emit(gpa, op.AccessChain, &.{ 17, 39, 16, 7, 38 }); // &u.col[vi]
    try vsb.emit(gpa, op.Load, &.{ 3, 40, 39 }); // col
    try vsb.emit(gpa, op.VectorShuffle, &.{ 4, 41, 40, 40, 0, 1, 2 }); // col.rgb
    try vsb.emit(gpa, op.Store, &.{ 25, 41 });
    try vsb.emit(gpa, op.Return, &.{});
    try vsb.emit(gpa, op.FunctionEnd, &.{});

    // Passthrough FS: in vec3 vc(loc0) -> vec4(vc, 1).
    var fsb = try @import("vulcan-spirv").binary.Builder.init(gpa, 18);
    defer fsb.deinit(gpa);
    try fsb.emit(gpa, op.EntryPoint, &.{ op.ExecutionModel.fragment, 10, 0, 8, 9 });
    try fsb.emit(gpa, op.Decorate, &.{ 8, op.Decoration.location, 0 });
    try fsb.emit(gpa, op.Decorate, &.{ 9, op.Decoration.location, 0 });
    try fsb.emit(gpa, op.TypeVoid, &.{1});
    try fsb.emit(gpa, op.TypeFunction, &.{ 2, 1 });
    try fsb.emit(gpa, op.TypeFloat, &.{ 3, 32 });
    try fsb.emit(gpa, op.TypeVector, &.{ 4, 3, 3 });
    try fsb.emit(gpa, op.TypeVector, &.{ 5, 3, 4 });
    try fsb.emit(gpa, op.TypePointer, &.{ 6, sc.input, 4 });
    try fsb.emit(gpa, op.TypePointer, &.{ 7, sc.output, 5 });
    try fsb.emit(gpa, op.Constant, &.{ 3, 12, @bitCast(@as(f32, 1.0)) });
    try fsb.emit(gpa, op.Variable, &.{ 6, 8, sc.input });
    try fsb.emit(gpa, op.Variable, &.{ 7, 9, sc.output });
    try fsb.emit(gpa, op.Function, &.{ 1, 10, 0, 2 });
    try fsb.emit(gpa, op.Label, &.{11});
    try fsb.emit(gpa, op.Load, &.{ 4, 13, 8 });
    try fsb.emit(gpa, op.CompositeExtract, &.{ 3, 14, 13, 0 });
    try fsb.emit(gpa, op.CompositeExtract, &.{ 3, 15, 13, 1 });
    try fsb.emit(gpa, op.CompositeExtract, &.{ 3, 16, 13, 2 });
    try fsb.emit(gpa, op.CompositeConstruct, &.{ 5, 17, 14, 15, 16, 12 });
    try fsb.emit(gpa, op.Store, &.{ 9, 17 });
    try fsb.emit(gpa, op.Return, &.{});
    try fsb.emit(gpa, op.FunctionEnd, &.{});

    const dev = try Device.create(gpa);
    defer dev.deinit();
    const vs = try dev.createShaderModule(.{ .stage = .vertex, .code = std.mem.sliceAsBytes(vsb.words.items) });
    defer dev.destroyShaderModule(vs);
    const fs = try dev.createShaderModule(.{ .stage = .fragment, .code = std.mem.sliceAsBytes(fsb.words.items) });
    defer dev.destroyShaderModule(fs);

    const W = 64;
    const H = 64;
    const pipe = try dev.createPipeline(.{
        .vertex = vs,
        .fragment = fs,
        .vertex_layout = .{ .stride = 0, .attributes = &.{} }, // pulling: no attributes
        .color_format = .rgba8_unorm,
    });
    defer dev.destroyPipeline(pipe);
    const sw_pipe: *Pipeline = @ptrCast(@alignCast(pipe));
    try std.testing.expect(sw_pipe.program != null);
    try std.testing.expectEqual(@as(usize, 2), sw_pipe.program.?.vs_index_count); // vi + ii

    // UBO: pos[3] (a small centered triangle), col[3] (red), offset[2] (left / right).
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
        // col[3] at f[12..24]: red.
        inline for (0..3) |k| {
            f[12 + k * 4 + 0] = 1;
            f[12 + k * 4 + 1] = 0;
            f[12 + k * 4 + 2] = 0;
            f[12 + k * 4 + 3] = 1;
        }
        // offset[2] at f[24..32]: instance 0 shifts LEFT (-0.5), instance 1 RIGHT (+0.5).
        f[24] = -0.5;
        f[25] = 0;
        f[26] = 0;
        f[27] = 0;
        f[28] = 0.5;
        f[29] = 0;
        f[30] = 0;
        f[31] = 0;
    }

    const target = try dev.createResource(.{ .image = .{ .width = W, .height = H, .format = .rgba8_unorm } });
    defer dev.destroyResource(target);
    const ctx = try dev.createContext();
    defer ctx.deinit();
    const cb = try ctx.beginCommands();
    defer cb.deinit();
    try cb.setRenderTarget(target);
    try cb.clear(.{ .r = 0, .g = 0, .b = 0, .a = 1 });
    try cb.bindPipeline(pipe);
    try cb.bindUniformBuffer(0, ubo);
    try cb.drawInstanced(3, 2, 0, 0); // 3 verts, 2 instances
    try ctx.submit(cb);

    const px = try dev.mapResource(target);
    // NDC x -0.5 -> window x 16; +0.5 -> x 48; the base triangle spans y about [22,42]
    // (window), center ~32. Instance 0 lands at x~16 (LEFT), instance 1 at x~48 (RIGHT).
    const left = (32 * W + 16) * 4;
    const right = (32 * W + 48) * 4;
    const center = (32 * W + 32) * 4; // between the two triangles -> background black
    try std.testing.expect(px[left + 0] > 200 and px[left + 1] < 50); // instance 0 drew (red)
    try std.testing.expect(px[right + 0] > 200 and px[right + 1] < 50); // instance 1 drew (red)
    try std.testing.expect(px[center + 0] < 50 and px[center + 1] < 50 and px[center + 2] < 50); // gap
}

// A vertex-pulling VS (gl_Position = u.pos[gl_VertexIndex]) + a constant-RED FS. Shared by
// the scissor oracles: a fullscreen triangle (u.pos = a big [-1,3] triangle covering the
// whole surface) makes coverage total, so any un-drawn pixel is proof of clipping.
fn buildFullscreenRedProgram(gpa: std.mem.Allocator, dev: hal.Device) !struct { vs: *hal.ShaderModule, fs: *hal.ShaderModule, ubo: *hal.Resource } {
    const op = @import("vulcan-spirv").opcodes;
    const sc = op.StorageClass;

    var vsb = try @import("vulcan-spirv").binary.Builder.init(gpa, 40);
    defer vsb.deinit(gpa);
    try vsb.emit(gpa, op.EntryPoint, &.{ op.ExecutionModel.vertex, 28, 0, 19, 23 });
    try vsb.emit(gpa, op.Decorate, &.{ 11, op.Decoration.block });
    try vsb.emit(gpa, op.MemberDecorate, &.{ 11, 0, op.Decoration.offset, 0 });
    try vsb.emit(gpa, op.Decorate, &.{ 10, op.Decoration.array_stride, 16 });
    try vsb.emit(gpa, op.Decorate, &.{ 16, op.Decoration.binding, 0 });
    try vsb.emit(gpa, op.Decorate, &.{ 19, op.Decoration.builtin, op.BuiltIn.vertex_index });
    try vsb.emit(gpa, op.Decorate, &.{ 21, op.Decoration.block });
    try vsb.emit(gpa, op.MemberDecorate, &.{ 21, 0, op.Decoration.builtin, op.BuiltIn.position });
    try vsb.emit(gpa, op.TypeVoid, &.{1});
    try vsb.emit(gpa, op.TypeFloat, &.{ 2, 32 });
    try vsb.emit(gpa, op.TypeVector, &.{ 3, 2, 4 }); // vec4
    try vsb.emit(gpa, op.TypeInt, &.{ 5, 32, 1 });
    try vsb.emit(gpa, op.Constant, &.{ 5, 6, 0 });
    try vsb.emit(gpa, op.TypeInt, &.{ 8, 32, 0 });
    try vsb.emit(gpa, op.Constant, &.{ 8, 9, 3 });
    try vsb.emit(gpa, op.TypeArray, &.{ 10, 3, 9 }); // vec4[3]
    try vsb.emit(gpa, op.TypeStruct, &.{ 11, 10 }); // U { pos[3]; }
    try vsb.emit(gpa, op.TypePointer, &.{ 15, sc.uniform, 11 });
    try vsb.emit(gpa, op.Variable, &.{ 15, 16, sc.uniform });
    try vsb.emit(gpa, op.TypePointer, &.{ 17, sc.uniform, 3 });
    try vsb.emit(gpa, op.TypePointer, &.{ 18, sc.input, 5 });
    try vsb.emit(gpa, op.Variable, &.{ 18, 19, sc.input }); // gl_VertexIndex
    try vsb.emit(gpa, op.TypeStruct, &.{ 21, 3 });
    try vsb.emit(gpa, op.TypePointer, &.{ 22, sc.output, 21 });
    try vsb.emit(gpa, op.Variable, &.{ 22, 23, sc.output }); // gl_PerVertex
    try vsb.emit(gpa, op.TypePointer, &.{ 26, sc.output, 3 });
    try vsb.emit(gpa, op.TypeFunction, &.{ 27, 1 });
    try vsb.emit(gpa, op.Function, &.{ 1, 28, 0, 27 });
    try vsb.emit(gpa, op.Label, &.{29});
    try vsb.emit(gpa, op.Load, &.{ 5, 30, 19 }); // vi
    try vsb.emit(gpa, op.AccessChain, &.{ 17, 31, 16, 6, 30 }); // &u.pos[vi]
    try vsb.emit(gpa, op.Load, &.{ 3, 32, 31 }); // pos
    try vsb.emit(gpa, op.AccessChain, &.{ 26, 37, 23, 6 }); // &gl_Position
    try vsb.emit(gpa, op.Store, &.{ 37, 32 });
    try vsb.emit(gpa, op.Return, &.{});
    try vsb.emit(gpa, op.FunctionEnd, &.{});

    // Constant-RED FS: out vec4 o(loc0) = vec4(1,0,0,1).
    var fsb = try @import("vulcan-spirv").binary.Builder.init(gpa, 20);
    defer fsb.deinit(gpa);
    try fsb.emit(gpa, op.EntryPoint, &.{ op.ExecutionModel.fragment, 10, 0, 9 });
    try fsb.emit(gpa, op.Decorate, &.{ 9, op.Decoration.location, 0 });
    try fsb.emit(gpa, op.TypeVoid, &.{1});
    try fsb.emit(gpa, op.TypeFunction, &.{ 2, 1 });
    try fsb.emit(gpa, op.TypeFloat, &.{ 3, 32 });
    try fsb.emit(gpa, op.TypeVector, &.{ 5, 3, 4 }); // vec4
    try fsb.emit(gpa, op.TypePointer, &.{ 7, sc.output, 5 });
    try fsb.emit(gpa, op.Constant, &.{ 3, 12, @bitCast(@as(f32, 1.0)) });
    try fsb.emit(gpa, op.Constant, &.{ 3, 13, @bitCast(@as(f32, 0.0)) });
    try fsb.emit(gpa, op.Variable, &.{ 7, 9, sc.output });
    try fsb.emit(gpa, op.Function, &.{ 1, 10, 0, 2 });
    try fsb.emit(gpa, op.Label, &.{11});
    try fsb.emit(gpa, op.CompositeConstruct, &.{ 5, 17, 12, 13, 13, 12 }); // vec4(1,0,0,1)
    try fsb.emit(gpa, op.Store, &.{ 9, 17 });
    try fsb.emit(gpa, op.Return, &.{});
    try fsb.emit(gpa, op.FunctionEnd, &.{});

    const vs = try dev.createShaderModule(.{ .stage = .vertex, .code = std.mem.sliceAsBytes(vsb.words.items) });
    errdefer dev.destroyShaderModule(vs);
    const fs = try dev.createShaderModule(.{ .stage = .fragment, .code = std.mem.sliceAsBytes(fsb.words.items) });
    errdefer dev.destroyShaderModule(fs);

    // The fullscreen triangle: (-1,-1),(3,-1),(-1,3) covers the whole [-1,1] clip square.
    const ubo = try dev.createResource(.{ .buffer = .{ .size = 48, .usage = .{ .uniform = true } } });
    errdefer dev.destroyResource(ubo);
    const f = std.mem.bytesAsSlice(f32, try dev.mapResource(ubo));
    const pos = [3][2]f32{ .{ -1, -1 }, .{ 3, -1 }, .{ -1, 3 } };
    inline for (pos, 0..) |v, k| {
        f[k * 4 + 0] = v[0];
        f[k * 4 + 1] = v[1];
        f[k * 4 + 2] = 0;
        f[k * 4 + 3] = 1;
    }
    return .{ .vs = vs, .fs = fs, .ubo = ubo };
}

test "scissor: a scissor rect clips a fullscreen draw to the rect; null re-enables full coverage" {
    const gpa = std.testing.allocator;
    const Device = @import("device.zig").Device;

    const dev = try Device.create(gpa);
    defer dev.deinit();
    const prog = try buildFullscreenRedProgram(gpa, dev);
    defer dev.destroyShaderModule(prog.vs);
    defer dev.destroyShaderModule(prog.fs);
    defer dev.destroyResource(prog.ubo);

    const W = 64;
    const H = 64;
    const pipe = try dev.createPipeline(.{
        .vertex = prog.vs,
        .fragment = prog.fs,
        .vertex_layout = .{ .stride = 0, .attributes = &.{} },
        .color_format = .rgba8_unorm,
    });
    defer dev.destroyPipeline(pipe);

    const target = try dev.createResource(.{ .image = .{ .width = W, .height = H, .format = .rgba8_unorm } });
    defer dev.destroyResource(target);
    const ctx = try dev.createContext();
    defer ctx.deinit();
    const cb = try ctx.beginCommands();
    defer cb.deinit();
    try cb.setRenderTarget(target);
    try cb.clear(.{ .r = 0, .g = 0, .b = 0, .a = 1 }); // black everywhere
    try cb.bindPipeline(pipe);
    try cb.bindUniformBuffer(0, prog.ubo);
    // Clip the fullscreen red draw to the TOP-LEFT 32x32 quadrant (top-left origin).
    try cb.setScissor(.{ .x = 0, .y = 0, .width = 32, .height = 32 });
    try cb.draw(3, 0);
    // Disable scissoring, then draw again: coverage returns to the full surface.
    try cb.setScissor(null);
    try cb.draw(3, 0);
    try ctx.submit(cb);

    const px = try dev.mapResource(target);
    const isRed = struct {
        fn f(p: []const u8, idx: usize) bool {
            return p[idx * 4 + 0] > 200 and p[idx * 4 + 1] < 60 and p[idx * 4 + 2] < 60;
        }
    }.f;
    // Inside the first (clipped) draw's rect: red from draw 1.
    try std.testing.expect(isRed(px, 16 * W + 16));
    // The second (unclipped) draw fills the rest: these were black after draw 1, red now.
    try std.testing.expect(isRed(px, 48 * W + 48));
    try std.testing.expect(isRed(px, 16 * W + 48));
    try std.testing.expect(isRed(px, 48 * W + 16));
}

test "scissor: glClear honors an enabled scissor (clears only the sub-rect)" {
    const gpa = std.testing.allocator;
    const Device = @import("device.zig").Device;

    const dev = try Device.create(gpa);
    defer dev.deinit();

    const W = 64;
    const H = 64;
    const target = try dev.createResource(.{ .image = .{ .width = W, .height = H, .format = .rgba8_unorm } });
    defer dev.destroyResource(target);
    const ctx = try dev.createContext();
    defer ctx.deinit();
    const cb = try ctx.beginCommands();
    defer cb.deinit();
    try cb.setRenderTarget(target);
    try cb.clear(.{ .r = 0, .g = 0, .b = 0, .a = 1 }); // whole surface black
    // Now clear RED, but only inside the BOTTOM-RIGHT 32x32 quadrant.
    try cb.setScissor(.{ .x = 32, .y = 32, .width = 32, .height = 32 });
    try cb.clear(.{ .r = 1, .g = 0, .b = 0, .a = 1 });
    try ctx.submit(cb);

    const px = try dev.mapResource(target);
    const isRed = struct {
        fn f(p: []const u8, idx: usize) bool {
            return p[idx * 4 + 0] > 200 and p[idx * 4 + 1] < 60 and p[idx * 4 + 2] < 60;
        }
    }.f;
    const isBlack = struct {
        fn f(p: []const u8, idx: usize) bool {
            return p[idx * 4 + 0] < 60 and p[idx * 4 + 1] < 60 and p[idx * 4 + 2] < 60;
        }
    }.f;
    try std.testing.expect(isRed(px, 48 * W + 48)); // inside the scissored clear
    try std.testing.expect(isBlack(px, 16 * W + 16)); // outside: untouched black
    try std.testing.expect(isBlack(px, 16 * W + 48)); // outside
    try std.testing.expect(isBlack(px, 48 * W + 16)); // outside
}

test "push constant: an FS reads a vec4 push-constant block as its output color" {
    const gpa = std.testing.allocator;
    const Device = @import("device.zig").Device;
    const op = @import("vulcan-spirv").opcodes;
    const sc = op.StorageClass;

    // Passthrough VS: in vec2 inPos(loc0) -> gl_Position = vec4(inPos, 0, 1).
    // ids: void1 f32:2 v2f:3 v4f:4 pInV2:5 inPos:6 gpvS:7 pOutGpv:8 gpv:9 fn:10 main:11
    //      entry:12 int:13 i0:14 c0:15 c1:16 p:17 x:18 y:19 pos:20 acGP:21 pOutV4:22.
    var vsb = try @import("vulcan-spirv").binary.Builder.init(gpa, 23);
    defer vsb.deinit(gpa);
    try vsb.emit(gpa, op.EntryPoint, &.{ op.ExecutionModel.vertex, 11, 0, 6, 9 });
    try vsb.emit(gpa, op.Decorate, &.{ 6, op.Decoration.location, 0 });
    try vsb.emit(gpa, op.Decorate, &.{ 7, op.Decoration.block });
    try vsb.emit(gpa, op.MemberDecorate, &.{ 7, 0, op.Decoration.builtin, op.BuiltIn.position });
    try vsb.emit(gpa, op.TypeVoid, &.{1});
    try vsb.emit(gpa, op.TypeFloat, &.{ 2, 32 });
    try vsb.emit(gpa, op.TypeVector, &.{ 3, 2, 2 }); // v2f
    try vsb.emit(gpa, op.TypeVector, &.{ 4, 2, 4 }); // v4f
    try vsb.emit(gpa, op.TypePointer, &.{ 5, sc.input, 3 });
    try vsb.emit(gpa, op.Variable, &.{ 5, 6, sc.input }); // inPos
    try vsb.emit(gpa, op.TypeStruct, &.{ 7, 4 });
    try vsb.emit(gpa, op.TypePointer, &.{ 8, sc.output, 7 });
    try vsb.emit(gpa, op.Variable, &.{ 8, 9, sc.output }); // gl_PerVertex
    try vsb.emit(gpa, op.TypeFunction, &.{ 10, 1 });
    try vsb.emit(gpa, op.TypeInt, &.{ 13, 32, 1 });
    try vsb.emit(gpa, op.Constant, &.{ 13, 14, 0 });
    try vsb.emit(gpa, op.Constant, &.{ 2, 15, 0 }); // f32 0.0
    try vsb.emit(gpa, op.Constant, &.{ 2, 16, @bitCast(@as(f32, 1.0)) }); // f32 1.0
    try vsb.emit(gpa, op.TypePointer, &.{ 22, sc.output, 4 });
    try vsb.emit(gpa, op.Function, &.{ 1, 11, 0, 10 });
    try vsb.emit(gpa, op.Label, &.{12});
    try vsb.emit(gpa, op.Load, &.{ 3, 17, 6 }); // vec2 p
    try vsb.emit(gpa, op.CompositeExtract, &.{ 2, 18, 17, 0 }); // p.x
    try vsb.emit(gpa, op.CompositeExtract, &.{ 2, 19, 17, 1 }); // p.y
    try vsb.emit(gpa, op.CompositeConstruct, &.{ 4, 20, 18, 19, 15, 16 }); // vec4(x,y,0,1)
    try vsb.emit(gpa, op.AccessChain, &.{ 22, 21, 9, 14 }); // &gl_Position
    try vsb.emit(gpa, op.Store, &.{ 21, 20 });
    try vsb.emit(gpa, op.Return, &.{});
    try vsb.emit(gpa, op.FunctionEnd, &.{});

    // FS reading a push-constant vec4 as its color:
    //   layout(push_constant) uniform PC { vec4 color; } pc;
    //   layout(location=0) out vec4 o; void main(){ o = pc.color; }
    // ids: void1 f32:2 v4f:3 int:4 i0:5 PC:6 pPC:7 pc:8 pPCv4:9 pOut:10 o:11 fn:12
    //      main:13 entry:14 acC:15 color:16.
    var fsb = try @import("vulcan-spirv").binary.Builder.init(gpa, 17);
    defer fsb.deinit(gpa);
    try fsb.emit(gpa, op.EntryPoint, &.{ op.ExecutionModel.fragment, 13, 0, 11 });
    try fsb.emit(gpa, op.Decorate, &.{ 6, op.Decoration.block });
    try fsb.emit(gpa, op.MemberDecorate, &.{ 6, 0, op.Decoration.offset, 0 });
    try fsb.emit(gpa, op.Decorate, &.{ 11, op.Decoration.location, 0 });
    try fsb.emit(gpa, op.TypeVoid, &.{1});
    try fsb.emit(gpa, op.TypeFloat, &.{ 2, 32 });
    try fsb.emit(gpa, op.TypeVector, &.{ 3, 2, 4 });
    try fsb.emit(gpa, op.TypeInt, &.{ 4, 32, 1 });
    try fsb.emit(gpa, op.Constant, &.{ 4, 5, 0 });
    try fsb.emit(gpa, op.TypeStruct, &.{ 6, 3 });
    try fsb.emit(gpa, op.TypePointer, &.{ 7, sc.push_constant, 6 });
    try fsb.emit(gpa, op.Variable, &.{ 7, 8, sc.push_constant });
    try fsb.emit(gpa, op.TypePointer, &.{ 9, sc.push_constant, 3 });
    try fsb.emit(gpa, op.TypePointer, &.{ 10, sc.output, 3 });
    try fsb.emit(gpa, op.Variable, &.{ 10, 11, sc.output });
    try fsb.emit(gpa, op.TypeFunction, &.{ 12, 1 });
    try fsb.emit(gpa, op.Function, &.{ 1, 13, 0, 12 });
    try fsb.emit(gpa, op.Label, &.{14});
    try fsb.emit(gpa, op.AccessChain, &.{ 9, 15, 8, 5 }); // &pc.color
    try fsb.emit(gpa, op.Load, &.{ 3, 16, 15 }); // vec4 color
    try fsb.emit(gpa, op.Store, &.{ 11, 16 }); // o = pc.color
    try fsb.emit(gpa, op.Return, &.{});
    try fsb.emit(gpa, op.FunctionEnd, &.{});

    const dev = try Device.create(gpa);
    defer dev.deinit();
    const vs = try dev.createShaderModule(.{ .stage = .vertex, .code = std.mem.sliceAsBytes(vsb.words.items) });
    defer dev.destroyShaderModule(vs);
    const fs = try dev.createShaderModule(.{ .stage = .fragment, .code = std.mem.sliceAsBytes(fsb.words.items) });
    defer dev.destroyShaderModule(fs);

    const W = 64;
    const H = 64;
    const attrs = [_]hal.VertexAttribute{.{ .location = 0, .format = .r32g32_float, .offset = 0 }};
    const pipe = try dev.createPipeline(.{
        .vertex = vs,
        .fragment = fs,
        .vertex_layout = .{ .stride = 8, .attributes = &attrs },
        .color_format = .rgba8_unorm,
    });
    defer dev.destroyPipeline(pipe);
    const sw_pipe: *Pipeline = @ptrCast(@alignCast(pipe));
    try std.testing.expect(sw_pipe.program != null); // the VS+FS JITed
    // The FS takes ONE buffer pointer param: the push-constant block.
    try std.testing.expectEqual(@as(usize, 1), sw_pipe.program.?.fs_buffers);

    // Vertex buffer: 3 verts, vec2 pos, a triangle covering the center.
    const verts = [_]f32{ -0.8, -0.8, 0.8, -0.8, 0.0, 0.8 };
    const vbuf = try dev.createResource(.{ .buffer = .{ .size = @sizeOf(@TypeOf(verts)) } });
    defer dev.destroyResource(vbuf);
    const vb = try dev.mapResource(vbuf);
    @memcpy(vb[0..@sizeOf(@TypeOf(verts))], std.mem.sliceAsBytes(&verts));

    // The push-constant block: a vec4 color (0.2,0.4,0.6,1.0) -> (51,102,153,255).
    const pc = try dev.createResource(.{ .buffer = .{ .size = 16 } });
    defer dev.destroyResource(pc);
    const pcb = try dev.mapResource(pc);
    const color = [4]f32{ 0.2, 0.4, 0.6, 1.0 };
    @memcpy(pcb[0..16], std.mem.sliceAsBytes(&color));

    const target = try dev.createResource(.{ .image = .{ .width = W, .height = H, .format = .rgba8_unorm } });
    defer dev.destroyResource(target);
    const tpix = try dev.mapResource(target);

    const ctx = try dev.createContext();
    defer ctx.deinit();
    const cb = try ctx.beginCommands();
    defer cb.deinit();
    try cb.setRenderTarget(target);
    try cb.clear(.{ .r = 0, .g = 0, .b = 0, .a = 1 });
    try cb.bindPipeline(pipe);
    try cb.bindVertexBuffer(vbuf);
    try cb.bindUniformBuffer(1, pc); // an FS push-constant block lowers to binding 1 (the FS slot)
    try cb.draw(3, 0);
    try ctx.submit(cb);

    // The center must be the pushed color (51,102,153,255), within rounding.
    const ci = (32 * W + 32) * 4;
    try std.testing.expectApproxEqAbs(@as(f32, 51), @as(f32, @floatFromInt(tpix[ci + 0])), 2);
    try std.testing.expectApproxEqAbs(@as(f32, 102), @as(f32, @floatFromInt(tpix[ci + 1])), 2);
    try std.testing.expectApproxEqAbs(@as(f32, 153), @as(f32, @floatFromInt(tpix[ci + 2])), 2);
    // A corner is the black clear.
    const corner = (1 * W + 1) * 4;
    try std.testing.expectEqual(@as(u8, 0), tpix[corner + 0]);
}

test "push constant + UBO: an FS reads BOTH (param order = UBO binding 0, push-constant last)" {
    const gpa = std.testing.allocator;
    const Device = @import("device.zig").Device;
    const op = @import("vulcan-spirv").opcodes;
    const sc = op.StorageClass;

    // Passthrough VS (pos vec2 loc0 -> gl_Position), identical to the prior test.
    var vsb = try @import("vulcan-spirv").binary.Builder.init(gpa, 23);
    defer vsb.deinit(gpa);
    try vsb.emit(gpa, op.EntryPoint, &.{ op.ExecutionModel.vertex, 11, 0, 6, 9 });
    try vsb.emit(gpa, op.Decorate, &.{ 6, op.Decoration.location, 0 });
    try vsb.emit(gpa, op.Decorate, &.{ 7, op.Decoration.block });
    try vsb.emit(gpa, op.MemberDecorate, &.{ 7, 0, op.Decoration.builtin, op.BuiltIn.position });
    try vsb.emit(gpa, op.TypeVoid, &.{1});
    try vsb.emit(gpa, op.TypeFloat, &.{ 2, 32 });
    try vsb.emit(gpa, op.TypeVector, &.{ 3, 2, 2 });
    try vsb.emit(gpa, op.TypeVector, &.{ 4, 2, 4 });
    try vsb.emit(gpa, op.TypePointer, &.{ 5, sc.input, 3 });
    try vsb.emit(gpa, op.Variable, &.{ 5, 6, sc.input });
    try vsb.emit(gpa, op.TypeStruct, &.{ 7, 4 });
    try vsb.emit(gpa, op.TypePointer, &.{ 8, sc.output, 7 });
    try vsb.emit(gpa, op.Variable, &.{ 8, 9, sc.output });
    try vsb.emit(gpa, op.TypeFunction, &.{ 10, 1 });
    try vsb.emit(gpa, op.TypeInt, &.{ 13, 32, 1 });
    try vsb.emit(gpa, op.Constant, &.{ 13, 14, 0 });
    try vsb.emit(gpa, op.Constant, &.{ 2, 15, 0 });
    try vsb.emit(gpa, op.Constant, &.{ 2, 16, @bitCast(@as(f32, 1.0)) });
    try vsb.emit(gpa, op.TypePointer, &.{ 22, sc.output, 4 });
    try vsb.emit(gpa, op.Function, &.{ 1, 11, 0, 10 });
    try vsb.emit(gpa, op.Label, &.{12});
    try vsb.emit(gpa, op.Load, &.{ 3, 17, 6 });
    try vsb.emit(gpa, op.CompositeExtract, &.{ 2, 18, 17, 0 });
    try vsb.emit(gpa, op.CompositeExtract, &.{ 2, 19, 17, 1 });
    try vsb.emit(gpa, op.CompositeConstruct, &.{ 4, 20, 18, 19, 15, 16 });
    try vsb.emit(gpa, op.AccessChain, &.{ 22, 21, 9, 14 });
    try vsb.emit(gpa, op.Store, &.{ 21, 20 });
    try vsb.emit(gpa, op.Return, &.{});
    try vsb.emit(gpa, op.FunctionEnd, &.{});

    // FS reading a UBO (binding 0) tint and a push-constant base. o = base + tint.
    //   layout(binding=0) uniform U { vec4 tint; } u;
    //   layout(push_constant) uniform PC { vec4 base; } pc;
    //   layout(location=0) out vec4 o; void main(){ o = pc.base + u.tint; }
    // ids: void1 f32:2 v4f:3 int:4 i0:5 U:6 pUniU:7 u:8 pUniV4:9 PC:10 pPC:11 pc:12
    //      pPCv4:13 pOut:14 o:15 fn:16 main:17 entry:18 acT:19 tint:20 acB:21 base:22 sum:23.
    var fsb = try @import("vulcan-spirv").binary.Builder.init(gpa, 24);
    defer fsb.deinit(gpa);
    try fsb.emit(gpa, op.EntryPoint, &.{ op.ExecutionModel.fragment, 17, 0, 15 });
    try fsb.emit(gpa, op.Decorate, &.{ 6, op.Decoration.block });
    try fsb.emit(gpa, op.MemberDecorate, &.{ 6, 0, op.Decoration.offset, 0 });
    try fsb.emit(gpa, op.Decorate, &.{ 8, op.Decoration.binding, 0 });
    try fsb.emit(gpa, op.Decorate, &.{ 8, op.Decoration.descriptor_set, 0 });
    try fsb.emit(gpa, op.Decorate, &.{ 10, op.Decoration.block });
    try fsb.emit(gpa, op.MemberDecorate, &.{ 10, 0, op.Decoration.offset, 0 });
    try fsb.emit(gpa, op.Decorate, &.{ 15, op.Decoration.location, 0 });
    try fsb.emit(gpa, op.TypeVoid, &.{1});
    try fsb.emit(gpa, op.TypeFloat, &.{ 2, 32 });
    try fsb.emit(gpa, op.TypeVector, &.{ 3, 2, 4 });
    try fsb.emit(gpa, op.TypeInt, &.{ 4, 32, 1 });
    try fsb.emit(gpa, op.Constant, &.{ 4, 5, 0 });
    try fsb.emit(gpa, op.TypeStruct, &.{ 6, 3 }); // U { vec4 tint; }
    try fsb.emit(gpa, op.TypePointer, &.{ 7, sc.uniform, 6 });
    try fsb.emit(gpa, op.Variable, &.{ 7, 8, sc.uniform });
    try fsb.emit(gpa, op.TypePointer, &.{ 9, sc.uniform, 3 });
    try fsb.emit(gpa, op.TypeStruct, &.{ 10, 3 }); // PC { vec4 base; }
    try fsb.emit(gpa, op.TypePointer, &.{ 11, sc.push_constant, 10 });
    try fsb.emit(gpa, op.Variable, &.{ 11, 12, sc.push_constant });
    try fsb.emit(gpa, op.TypePointer, &.{ 13, sc.push_constant, 3 });
    try fsb.emit(gpa, op.TypePointer, &.{ 14, sc.output, 3 });
    try fsb.emit(gpa, op.Variable, &.{ 14, 15, sc.output });
    try fsb.emit(gpa, op.TypeFunction, &.{ 16, 1 });
    try fsb.emit(gpa, op.Function, &.{ 1, 17, 0, 16 });
    try fsb.emit(gpa, op.Label, &.{18});
    try fsb.emit(gpa, op.AccessChain, &.{ 9, 19, 8, 5 }); // &u.tint
    try fsb.emit(gpa, op.Load, &.{ 3, 20, 19 }); // vec4 tint
    try fsb.emit(gpa, op.AccessChain, &.{ 13, 21, 12, 5 }); // &pc.base
    try fsb.emit(gpa, op.Load, &.{ 3, 22, 21 }); // vec4 base
    try fsb.emit(gpa, op.FAdd, &.{ 3, 23, 22, 20 }); // base + tint
    try fsb.emit(gpa, op.Store, &.{ 15, 23 });
    try fsb.emit(gpa, op.Return, &.{});
    try fsb.emit(gpa, op.FunctionEnd, &.{});

    const dev = try Device.create(gpa);
    defer dev.deinit();
    const vs = try dev.createShaderModule(.{ .stage = .vertex, .code = std.mem.sliceAsBytes(vsb.words.items) });
    defer dev.destroyShaderModule(vs);
    const fs = try dev.createShaderModule(.{ .stage = .fragment, .code = std.mem.sliceAsBytes(fsb.words.items) });
    defer dev.destroyShaderModule(fs);

    const W = 64;
    const H = 64;
    const attrs = [_]hal.VertexAttribute{.{ .location = 0, .format = .r32g32_float, .offset = 0 }};
    const pipe = try dev.createPipeline(.{
        .vertex = vs,
        .fragment = fs,
        .vertex_layout = .{ .stride = 8, .attributes = &attrs },
        .color_format = .rgba8_unorm,
    });
    defer dev.destroyPipeline(pipe);
    const sw_pipe: *Pipeline = @ptrCast(@alignCast(pipe));
    try std.testing.expect(sw_pipe.program != null);
    // Two buffer params: the UBO (binding 0, first) + the push constant (last).
    try std.testing.expectEqual(@as(usize, 2), sw_pipe.program.?.fs_buffers);

    const verts = [_]f32{ -0.8, -0.8, 0.8, -0.8, 0.0, 0.8 };
    const vbuf = try dev.createResource(.{ .buffer = .{ .size = @sizeOf(@TypeOf(verts)) } });
    defer dev.destroyResource(vbuf);
    @memcpy((try dev.mapResource(vbuf))[0..@sizeOf(@TypeOf(verts))], std.mem.sliceAsBytes(&verts));

    // UBO tint (0.1, 0.0, 0.0, 0.0) at binding 0. Push-constant base (0.0, 0.4, 0.6, 1.0)
    // is at the slot after descriptors (binding 1). o = base + tint =
    // (0.1, 0.4, 0.6, 1.0) -> (26, 102, 153, 255). If the UBO + PC were swapped the red
    // channel would be wrong, so a correct red proves the param order.
    const ubo = try dev.createResource(.{ .buffer = .{ .size = 16 } });
    defer dev.destroyResource(ubo);
    const tint = [4]f32{ 0.1, 0.0, 0.0, 0.0 };
    @memcpy((try dev.mapResource(ubo))[0..16], std.mem.sliceAsBytes(&tint));
    const pc = try dev.createResource(.{ .buffer = .{ .size = 16 } });
    defer dev.destroyResource(pc);
    const base = [4]f32{ 0.0, 0.4, 0.6, 1.0 };
    @memcpy((try dev.mapResource(pc))[0..16], std.mem.sliceAsBytes(&base));

    const target = try dev.createResource(.{ .image = .{ .width = W, .height = H, .format = .rgba8_unorm } });
    defer dev.destroyResource(target);
    const tpix = try dev.mapResource(target);

    const ctx = try dev.createContext();
    defer ctx.deinit();
    const cb = try ctx.beginCommands();
    defer cb.deinit();
    try cb.setRenderTarget(target);
    try cb.clear(.{ .r = 0, .g = 0, .b = 0, .a = 1 });
    try cb.bindPipeline(pipe);
    try cb.bindVertexBuffer(vbuf);
    try cb.bindUniformBuffer(0, ubo); // descriptor binding 0
    try cb.bindUniformBuffer(1, pc); // push-constant block at the slot after descriptors
    try cb.draw(3, 0);
    try ctx.submit(cb);

    const ci = (32 * W + 32) * 4;
    try std.testing.expectApproxEqAbs(@as(f32, 26), @as(f32, @floatFromInt(tpix[ci + 0])), 2); // 0.1 (tint) -> red
    try std.testing.expectApproxEqAbs(@as(f32, 102), @as(f32, @floatFromInt(tpix[ci + 1])), 2); // 0.4 (base)
    try std.testing.expectApproxEqAbs(@as(f32, 153), @as(f32, @floatFromInt(tpix[ci + 2])), 2); // 0.6 (base)
}
