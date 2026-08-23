//! Prism HAL context + command buffer over the virgl device.
//! Records render state. On submit, uploads vertex data, encodes a virgl command stream, and submits it.
//! present reads the render target back via transferFromHost and best-effort presents to a host surface.

const std = @import("std");
const hal = @import("../../hal.zig");
const Device = @import("device.zig").Device;
const stream_mod = @import("virgl_stream.zig");
const enc = @import("encoding.zig");
const formats = @import("formats.zig");
const transport_mod = @import("transport.zig");

/// The virgl PIPE_FORMAT for an `n`-component f32 vertex attribute. The Gallium
/// float vertex formats are contiguous: R32_FLOAT=28, R32G32_FLOAT=29,
/// R32G32B32_FLOAT=30, R32G32B32A32_FLOAT=31 (so n components -> 27 + n).
fn floatVertexFormat(components: u32) u32 {
    const n = std.math.clamp(components, 1, 4);
    return 27 + n;
}

/// HAL filter/address enums -> the Gallium PIPE_TEX_* ordinals the sampler state
/// carries. Prism's textures have no mip chain here, so MipFilter.none -> PIPE NONE.
fn pipeFilter(f: hal.Filter) u32 {
    return switch (f) {
        .nearest => 0,
        .linear => 1,
    };
}
fn pipeMipFilter(m: hal.MipFilter) u32 {
    return switch (m) {
        .nearest => 0,
        .linear => 1,
        .none => 2, // PIPE_TEX_MIPFILTER_NONE
    };
}
fn pipeWrap(a: hal.AddressMode) u32 {
    return switch (a) {
        .repeat => 0, // PIPE_TEX_WRAP_REPEAT
        .clamp_to_edge => 2, // PIPE_TEX_WRAP_CLAMP_TO_EDGE
        .mirrored_repeat => 1, // PIPE_TEX_WRAP_MIRROR_REPEAT
    };
}

/// HAL BlendFactor -> the Gallium PIPE_BLENDFACTOR_* ordinal.
fn pipeBlendFactor(f: hal.BlendFactor) u32 {
    return switch (f) {
        .one => 1,
        .src_color => 2,
        .src_alpha => 3,
        .dst_alpha => 4,
        .dst_color => 5,
        .src_alpha_saturate => 6,
        .constant_color => 7,
        .constant_alpha => 8,
        .zero => 0x11,
        .one_minus_src_color => 0x12,
        .one_minus_src_alpha => 0x13,
        .one_minus_dst_alpha => 0x14,
        .one_minus_dst_color => 0x15,
        .one_minus_constant_color => 0x17,
        .one_minus_constant_alpha => 0x18,
    };
}
/// HAL BlendOp -> PIPE_BLEND_* (add=0..max=4, a direct ordinal match).
fn pipeBlendOp(o: hal.BlendOp) u32 {
    return @intFromEnum(o);
}
/// Whether a blend factor references glBlendColor (a CONSTANT_* factor).
fn blendUsesConstant(f: hal.BlendFactor) bool {
    return switch (f) {
        .constant_color, .one_minus_constant_color, .constant_alpha, .one_minus_constant_alpha => true,
        else => false,
    };
}
/// HAL CullMode -> the PIPE_FACE mask (none=0, front=1, back=2).
fn pipeCullFace(m: hal.CullMode) u32 {
    return switch (m) {
        .none => 0,
        .front => 1,
        .back => 2,
    };
}

fn mapErr(e: transport_mod.Error) hal.Error {
    return switch (e) {
        error.OutOfMemory => error.OutOfMemory,
        error.InitializationFailed => error.InitializationFailed,
        error.DeviceLost => error.DeviceLost,
        error.Unsupported => error.Unsupported,
    };
}

const CommandBuffer = struct {
    dev: *Device,
    rt: ?*hal.Resource = null,
    clear_color: hal.Color = .{},
    pipeline: ?*hal.Pipeline = null,
    vbuf: ?*hal.Resource = null,
    vertex_count: u32 = 0,
    first_vertex: u32 = 0,
    instance_count: u32 = 1,
    first_instance: u32 = 0,
    scissor: ?stream_mod.Encoder.Scissor = null,
    depth_res: ?*hal.Resource = null,
    depth_clear: ?f32 = null,
    /// Bound uniform blocks, indexed by binding (the TGSI CONST unit). A driver-
    /// side cap of 8 bindings mirrors the small UBO count Prism shaders use.
    ubos: [8]?*hal.Resource = .{null} ** 8,
    /// Bound combined-image-samplers, indexed by binding (the TGSI SAMP unit).
    textures: [8]?hal.TextureBinding = .{null} ** 8,
    /// A pending MSAA resolve (multisample `src` -> single-sample `dst`), emitted as
    /// a BLIT after the draw at submit. Null when the pass has no resolve attachment.
    resolve_req: ?struct { src: *hal.Resource, dst: *hal.Resource, width: u32, height: u32, format: hal.Format } = null,
    /// Additional MRT color targets. Index 0 is `rt`. Indices 1..7 are the extras.
    /// A fragment shader's `layout(location = i) out` writes to color_rts[i-1].
    extra_rts: [7]?*hal.Resource = .{null} ** 7,

    fn setRenderTarget(ptr: *anyopaque, target: *hal.Resource) hal.Error!void {
        const self: *CommandBuffer = @ptrCast(@alignCast(ptr));
        self.rt = target;
    }
    fn clear(ptr: *anyopaque, color: hal.Color) hal.Error!void {
        const self: *CommandBuffer = @ptrCast(@alignCast(ptr));
        self.clear_color = color;
    }
    fn bindPipeline(ptr: *anyopaque, pipeline: *hal.Pipeline) hal.Error!void {
        const self: *CommandBuffer = @ptrCast(@alignCast(ptr));
        self.pipeline = pipeline;
    }
    fn bindVertexBuffer(ptr: *anyopaque, buffer: *hal.Resource) hal.Error!void {
        const self: *CommandBuffer = @ptrCast(@alignCast(ptr));
        self.vbuf = buffer;
    }
    fn draw(ptr: *anyopaque, vertex_count: u32, first_vertex: u32) hal.Error!void {
        const self: *CommandBuffer = @ptrCast(@alignCast(ptr));
        self.vertex_count = vertex_count;
        self.first_vertex = first_vertex;
        self.instance_count = 1;
        self.first_instance = 0;
    }
    fn drawInstanced(ptr: *anyopaque, vertex_count: u32, instance_count: u32, first_vertex: u32, first_instance: u32) hal.Error!void {
        const self: *CommandBuffer = @ptrCast(@alignCast(ptr));
        self.vertex_count = vertex_count;
        self.first_vertex = first_vertex;
        self.instance_count = instance_count;
        self.first_instance = first_instance;
    }
    fn bindUniformBuffer(ptr: *anyopaque, binding: u32, buffer: *hal.Resource) hal.Error!void {
        const self: *CommandBuffer = @ptrCast(@alignCast(ptr));
        if (binding >= self.ubos.len) return error.InvalidArgument;
        self.ubos[binding] = buffer;
    }
    fn setColorTarget(ptr: *anyopaque, index: u32, target: *hal.Resource) hal.Error!void {
        const self: *CommandBuffer = @ptrCast(@alignCast(ptr));
        if (index == 0) {
            self.rt = target; // color target 0 == the primary render target
        } else {
            if (index - 1 >= self.extra_rts.len) return error.InvalidArgument;
            self.extra_rts[index - 1] = target;
        }
    }
    fn setDepthTarget(ptr: *anyopaque, depth: *hal.Resource, clear_value: ?f32) hal.Error!void {
        const self: *CommandBuffer = @ptrCast(@alignCast(ptr));
        self.depth_res = depth;
        self.depth_clear = clear_value;
    }
    fn bindTexture(ptr: *anyopaque, binding: hal.TextureBinding) hal.Error!void {
        const self: *CommandBuffer = @ptrCast(@alignCast(ptr));
        if (binding.binding >= self.textures.len) return error.InvalidArgument;
        self.textures[binding.binding] = binding;
    }
    fn resolve(ptr: *anyopaque, src: *hal.Resource, dst: *hal.Resource, width: u32, height: u32, format: hal.Format, samples: u8) hal.Error!void {
        const self: *CommandBuffer = @ptrCast(@alignCast(ptr));
        _ = samples; // the src resource already carries its MSAA count
        self.resolve_req = .{ .src = src, .dst = dst, .width = width, .height = height, .format = format };
    }
    fn setScissor(ptr: *anyopaque, rect: ?hal.ScissorRect) hal.Error!void {
        const self: *CommandBuffer = @ptrCast(@alignCast(ptr));
        if (rect) |r| {
            // HAL ScissorRect is top-left {x, y, width, height}. Virgl wants a
            // min/max corner (max exclusive). Clamp a negative origin to 0.
            const minx: u32 = @intCast(@max(r.x, 0));
            const miny: u32 = @intCast(@max(r.y, 0));
            self.scissor = .{
                .minx = minx,
                .miny = miny,
                .maxx = minx + r.width,
                .maxy = miny + r.height,
            };
        } else {
            self.scissor = null;
        }
    }
    fn reset(ptr: *anyopaque) void {
        const self: *CommandBuffer = @ptrCast(@alignCast(ptr));
        self.rt = null;
        self.pipeline = null;
        self.vbuf = null;
        self.vertex_count = 0;
        self.first_vertex = 0;
        self.instance_count = 1;
        self.first_instance = 0;
        self.scissor = null;
        self.depth_res = null;
        self.depth_clear = null;
        self.ubos = .{null} ** 8;
        self.textures = .{null} ** 8;
        self.resolve_req = null;
        self.extra_rts = .{null} ** 7;
        self.clear_color = .{};
    }
    fn deinit(ptr: *anyopaque) void {
        const self: *CommandBuffer = @ptrCast(@alignCast(ptr));
        self.dev.gpa.destroy(self);
    }

    const vtable = hal.CommandBuffer.VTable{
        .setRenderTarget = &setRenderTarget,
        .clear = &clear,
        .bindPipeline = &bindPipeline,
        .bindVertexBuffer = &bindVertexBuffer,
        .draw = &draw,
        .drawInstanced = &drawInstanced,
        .setScissor = &setScissor,
        .setDepthTarget = &setDepthTarget,
        .bindUniformBuffer = &bindUniformBuffer,
        .bindTexture = &bindTexture,
        .resolve = &resolve,
        .setColorTarget = &setColorTarget,
        .reset = &reset,
        .deinit = &deinit,
    };
};

pub const Context = struct {
    dev: *Device,

    pub fn create(dev: *Device) hal.Error!hal.Context {
        const self = dev.gpa.create(Context) catch return error.OutOfMemory;
        self.* = .{ .dev = dev };
        return .{ .ptr = self, .vtable = &vtable };
    }

    fn beginCommands(ptr: *anyopaque) hal.Error!hal.CommandBuffer {
        const self: *Context = @ptrCast(@alignCast(ptr));
        const cb = self.dev.gpa.create(CommandBuffer) catch return error.OutOfMemory;
        cb.* = .{ .dev = self.dev };
        return .{ .ptr = cb, .vtable = &CommandBuffer.vtable };
    }

    fn submit(ptr: *anyopaque, cb: hal.CommandBuffer) hal.Error!void {
        const self: *Context = @ptrCast(@alignCast(ptr));
        const c: *CommandBuffer = @ptrCast(@alignCast(cb.ptr));
        const dev = self.dev;

        // A resolve-only command buffer (no draw bound): emit just the MSAA BLIT.
        // This is the EndRenderPass resolve the GLES/ICD layer issues on its own.
        if (c.pipeline == null and c.resolve_req != null) {
            try self.emitResolve(dev, c.resolve_req.?);
            return;
        }

        // A clear-only command buffer (render target bound, but no pipeline / vertex
        // buffer / draw and no resolve): emit just the framebuffer bind + CLEAR. The
        // draw path below requires a pipeline + vertex buffer, which a bare clear
        // (e.g. a compositor clearing a scanout target) does not have.
        if (c.pipeline == null and c.rt != null) {
            const rtc = Device.resourceOf(c.rt.?);
            var clear_enc = stream_mod.Encoder.init(dev.stream[0..]);
            const cwords = clear_enc.encodeClear(.{
                .rt_res = rtc.t.res_id,
                .rt_format = formats.virglColorFormat(rtc.format),
                .color = .{ c.clear_color.r, c.clear_color.g, c.clear_color.b, c.clear_color.a },
            }, .{});
            const cbytes: [*]const u8 = @ptrCast(&dev.stream);
            dev.transport.submit(cbytes[0 .. cwords * 4]) catch |e| return mapErr(e);
            return;
        }

        const rt_res = Device.resourceOf(c.rt orelse return error.InvalidArgument);
        const vb_res = Device.resourceOf(c.vbuf orelse return error.InvalidArgument);
        const pipe = Device.pipelineOf(c.pipeline orelse return error.InvalidArgument);

        // Upload the vertex buffer to the host copy.
        dev.transport.transferToHost(vb_res.t, @intCast(vb_res.t.bytes.len)) catch |e| return mapErr(e);

        // Derive each virgl vertex-attribute format from the real HAL layout. The
        // vertex data is f32 components. Each attribute's component count comes from
        // its byte span (the gap to the next attribute, or to the stride for the
        // last), so the virgl PIPE_FORMAT is the R32[..G32B32A32]_FLOAT of that width.
        var pos_off: u32 = 0;
        var col_off: u32 = 8;
        var pos_format: u32 = enc.FORMAT_R32G32_FLOAT;
        var color_format: u32 = enc.FORMAT_R32G32B32A32_FLOAT;
        if (pipe.layout.attributes.len >= 2) {
            const a = pipe.layout.attributes;
            pos_off = a[0].offset;
            col_off = a[1].offset;
            pos_format = floatVertexFormat((a[1].offset - a[0].offset) / 4);
            color_format = floatVertexFormat((pipe.layout.stride - a[1].offset) / 4);
        }

        // The render-target / surface format is the pipeline's color format mapped
        // to its virgl pipe format. For the SDR path this is B8G8R8X8. For HDR it
        // is the fp16 (R16G16B16A16_FLOAT) or 10-bit pipe format, so virglrenderer
        // allocates a high-precision surface that holds values past SDR white.
        const rt_format = formats.virglColorFormat(pipe.color_format);

        // A bound depth attachment lowers to a ZETA surface + the pipeline's
        // depth-test state (the HAL CompareOp ordinal is the Gallium PIPE_FUNC).
        const depth: ?stream_mod.Encoder.Depth = if (c.depth_res) |dr| blk: {
            const dres = Device.resourceOf(dr);
            break :blk .{
                .res = dres.t.res_id,
                .format = formats.virglColorFormat(dres.format),
                .test_enable = pipe.depth.test_enable,
                .write_enable = pipe.depth.write_enable,
                .func = @intFromEnum(pipe.depth.compare_op),
                .clear = c.depth_clear,
            };
        } else null;

        // Upload each bound uniform block to the host and point a Uniform at its
        // guest backing (as dwords), so encodeDraw emits the std140 data into the
        // shader's CONST[binding]. The words alias the resource backing for the
        // lifetime of this submit.
        var ubuf: [8]stream_mod.Encoder.Uniform = undefined;
        var ucount: usize = 0;
        for (c.ubos, 0..) |maybe, binding| {
            const ur = maybe orelse continue;
            const ures = Device.resourceOf(ur);
            dev.transport.transferToHost(ures.t, @intCast(ures.t.bytes.len)) catch |e| return mapErr(e);
            const words: []const u32 = @as([*]const u32, @ptrCast(@alignCast(ures.t.bytes.ptr)))[0 .. ures.t.bytes.len / 4];
            ubuf[ucount] = .{ .binding = @intCast(binding), .words = words };
            ucount += 1;
        }

        // Upload each bound texture's pixels to the host and build its sampler-view +
        // sampler-state description (the HAL filter/address enums mapped to pipe
        // ordinals). encodeDraw then creates the objects and binds them to SAMP[unit].
        var tbuf: [8]stream_mod.Encoder.Texture = undefined;
        var tcount: usize = 0;
        for (c.textures) |maybe| {
            const tb = maybe orelse continue;
            const tres = Device.resourceOf(tb.image);
            dev.transport.transferToHost(tres.t, @intCast(tres.t.bytes.len)) catch |e| return mapErr(e);
            tbuf[tcount] = .{
                .binding = tb.binding,
                .res_id = tres.t.res_id,
                .format = formats.virglSampledFormat(tres.format),
                .wrap_s = pipeWrap(tb.address_u),
                .wrap_t = pipeWrap(tb.address_v),
                .min_filter = pipeFilter(tb.min_filter),
                .mag_filter = pipeFilter(tb.filter),
                .mip_filter = pipeMipFilter(tb.mip_filter),
            };
            tcount += 1;
        }

        // Map the pipeline's HAL blend + cull state to the virgl encoder structs.
        const bl = pipe.blend;
        const blend: stream_mod.Encoder.Blend = .{
            .enable = bl.enable,
            .rgb_op = pipeBlendOp(bl.color_op),
            .rgb_src = pipeBlendFactor(bl.src_color),
            .rgb_dst = pipeBlendFactor(bl.dst_color),
            .alpha_op = pipeBlendOp(bl.alpha_op),
            .alpha_src = pipeBlendFactor(bl.src_alpha),
            .alpha_dst = pipeBlendFactor(bl.dst_alpha),
            .constant = bl.constant,
            .use_constant = blendUsesConstant(bl.src_color) or blendUsesConstant(bl.dst_color) or
                blendUsesConstant(bl.src_alpha) or blendUsesConstant(bl.dst_alpha),
        };
        const cull: stream_mod.Encoder.Cull = .{
            .cull_face = pipeCullFace(pipe.cull.mode),
            .front_ccw = pipe.cull.front_face == .counter_clockwise,
        };

        // Additional MRT color targets (contiguous from index 1). Each is a color image
        // resource. The framebuffer binds a surface over it at the matching cbuf slot.
        var crts: [7]stream_mod.Encoder.ColorRT = undefined;
        var ncrt: usize = 0;
        for (c.extra_rts) |maybe| {
            const cr = maybe orelse break; // stop at the first gap (contiguous targets)
            const cres = Device.resourceOf(cr);
            crts[ncrt] = .{ .res_id = cres.t.res_id, .format = formats.virglColorFormat(cres.format) };
            ncrt += 1;
        }

        var encoder = stream_mod.Encoder.init(dev.stream[0..]);
        const words = encoder.encodeDraw(.{
            .rt_res = rt_res.t.res_id,
            .vbuf_res = vb_res.t.res_id,
            .fb_w = rt_res.t.width,
            .fb_h = rt_res.t.height,
            .rt_format = rt_format,
            .vertex_stride = pipe.layout.stride,
            .pos_format = pos_format,
            .pos_offset = pos_off,
            .color_format = color_format,
            .color_offset = col_off,
            .vs_tgsi = pipe.vs.tgsi,
            .fs_tgsi = pipe.fs.tgsi,
            .clear = .{ c.clear_color.r, c.clear_color.g, c.clear_color.b, c.clear_color.a },
            .vertex_count = c.vertex_count,
            .mode = enc.primFromTopology(pipe.topology),
            .instance_count = c.instance_count,
            .first_vertex = c.first_vertex,
            .first_instance = c.first_instance,
            .line_width = pipe.line_width,
            .scissor = c.scissor,
            .depth = depth,
            .uniforms = ubuf[0..ucount],
            .textures = tbuf[0..tcount],
            .samples = rt_res.samples,
            .blend = blend,
            .cull = cull,
            .extra_color_rts = crts[0..ncrt],
        }, .{});

        const stream_bytes: [*]const u8 = @ptrCast(&dev.stream);
        dev.transport.submit(stream_bytes[0 .. words * 4]) catch |e| return mapErr(e);

        // A same-pass resolve attachment (MSAA): after the draw, resolve the
        // multisampled RT into its single-sample destination via a BLIT.
        if (c.resolve_req) |rr| try self.emitResolve(dev, rr);
    }

    /// Encode + submit a single MSAA resolve BLIT (multisample src -> single dst).
    fn emitResolve(self: *Context, dev: *Device, rr: anytype) hal.Error!void {
        _ = self;
        const src = Device.resourceOf(rr.src);
        const dst = Device.resourceOf(rr.dst);
        var encoder = stream_mod.Encoder.init(dev.stream[0..]);
        const words = encoder.encodeBlit(.{
            .src_res = src.t.res_id,
            .dst_res = dst.t.res_id,
            .width = rr.width,
            .height = rr.height,
            .format = formats.virglColorFormat(rr.format),
        });
        const bytes: [*]const u8 = @ptrCast(&dev.stream);
        dev.transport.submit(bytes[0 .. words * 4]) catch |e| return mapErr(e);
    }

    fn present(ptr: *anyopaque, surface: *hal.Surface, source: *hal.Resource) hal.Error!void {
        const self: *Context = @ptrCast(@alignCast(ptr));
        _ = surface;
        const dev = self.dev;
        const rt = Device.resourceOf(source);

        // Read the rendered render target back into its guest backing so the
        // caller can inspect or dump the pixels (the primary verification path on
        // a headless host that has no readable DisplaySurface / no guest scanout).
        dev.transport.transferFromHost(rt.t) catch |e| return mapErr(e);

        // Best-effort scanout + flush for a host/transport that supports it.
        dev.transport.present(rt.t);
    }

    fn deinit(ptr: *anyopaque) void {
        const self: *Context = @ptrCast(@alignCast(ptr));
        self.dev.gpa.destroy(self);
    }

    const vtable = hal.Context.VTable{
        .beginCommands = &beginCommands,
        .submit = &submit,
        .present = &present,
        .deinit = &deinit,
    };
};
