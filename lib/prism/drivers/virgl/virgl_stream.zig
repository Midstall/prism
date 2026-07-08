//! virgl command-stream encoder. Builds the little-endian u32 word stream a virtio-gpu SUBMIT_3D carries.
//! Encodes state objects (blend/rasterizer/dsa/surface), framebuffer + viewport, vertex elements, VS/FS shaders, CLEAR, DRAW_VBO.
//! The encoding matches conduit/test/qemu/virtio_gpu_virgl.zig. vrend's body sizes are load-bearing.

const std = @import("std");
const virgl = @import("encoding.zig");

/// A bump encoder over a caller-owned word buffer.
pub const Encoder = struct {
    words: []u32,
    len: usize = 0,

    pub fn init(words: []u32) Encoder {
        return .{ .words = words };
    }

    fn emit(self: *Encoder, w: u32) void {
        self.words[self.len] = w;
        self.len += 1;
    }
    fn emitF(self: *Encoder, f: f32) void {
        self.emit(@bitCast(f));
    }
    fn emitHdr(self: *Encoder, cmd: u32, obj: u32, body_len: u32) void {
        self.emit(virgl.cmd0(cmd, obj, body_len));
    }

    /// Append a TGSI shader-create command. vrend reads the body as:
    ///   [1] handle, [2] type, [3] offset(=byte len), [4] num_tokens(upper
    ///   bound), [5] so_num_outputs(=0), [6..] the NUL-terminated TGSI text
    ///   padded to a dword boundary. Body length is 5 + text_dwords.
    fn emitShader(self: *Encoder, handle: u32, stype: u32, src: []const u8) void {
        const text_dwords: u32 = @intCast((src.len + 3) / 4);
        const body_len: u32 = 5 + text_dwords;
        self.emitHdr(virgl.CCMD_CREATE_OBJECT, virgl.OBJ_SHADER, body_len);
        self.emit(handle);
        self.emit(stype);
        self.emit(@intCast(src.len));
        self.emit(256); // num_tokens upper bound
        self.emit(0); // so_num_outputs
        var i: usize = 0;
        while (i < text_dwords) : (i += 1) {
            var word: u32 = 0;
            var b: usize = 0;
            while (b < 4) : (b += 1) {
                const idx = i * 4 + b;
                const byte: u32 = if (idx < src.len) src[idx] else 0;
                word |= byte << @intCast(b * 8);
            }
            self.emit(word);
        }
    }

    /// Object handles, fixed for a single-pipeline draw (one per object type).
    pub const Handles = struct {
        blend: u32 = 1,
        rasterizer: u32 = 2,
        dsa: u32 = 3,
        surface: u32 = 4,
        vertex_elements: u32 = 5,
        vs: u32 = 6,
        fs: u32 = 7,
        zsurface: u32 = 8, // the depth (ZETA) surface, when a depth attachment is bound
        sampler_view_base: u32 = 16, // sampler-view handle = base + binding (up to 8)
        sampler_state_base: u32 = 24, // sampler-state handle = base + binding
        color_rt_base: u32 = 40, // MRT color-surface handle = base + (target-1), targets 1+
    };

    /// An additional MRT color render target. Target 0 is rt_res/rt_format. Targets 1+
    /// are filled from setColorTarget.
    pub const ColorRT = struct {
        res_id: u32,
        format: u32,
    };

    /// A bound combined-image-sampler: the texture resource plus the pipe sampler-
    /// state parameters (already mapped from the HAL filter/address enums). The
    /// context uploads the texel bytes to the host before the draw.
    pub const Texture = struct {
        binding: u32,
        res_id: u32,
        format: u32 = virgl.FORMAT_R8G8B8A8_UNORM, // the sampler-view pipe format
        wrap_s: u32,
        wrap_t: u32,
        min_filter: u32,
        mag_filter: u32,
        mip_filter: u32,
    };

    /// A bound depth attachment lowered to virgl: the ZETA resource, its pipe
    /// format, the pipeline's depth-test state, and an optional clear value. The
    /// context fills this from setDepthTarget + the pipeline's DepthState.
    pub const Depth = struct {
        res: u32, // depth resource id (a Z32_FLOAT ZETA texture)
        format: u32 = virgl.FORMAT_Z32_FLOAT,
        test_enable: bool,
        write_enable: bool,
        func: u32, // Gallium PIPE_FUNC (== HAL CompareOp ordinal)
        clear: ?f32 = null, // clear the depth buffer to this value before drawing
    };

    /// A scissor rectangle in the render target's pixel space, max corner
    /// exclusive (VIRGL pipe_scissor_state convention). The context translates a
    /// HAL ScissorRect into this before encoding.
    pub const Scissor = struct {
        minx: u32,
        miny: u32,
        maxx: u32,
        maxy: u32,
    };

    /// A uniform-block binding: the std140 bytes (as dwords) for the shader's
    /// CONST[binding]. The context points `words` at the uploaded UBO resource's
    /// guest backing. encodeDraw emits it into both shader stages' constant buffer
    /// at `binding` (each stage's TGSI reads only the CONST unit it declared).
    pub const Uniform = struct {
        binding: u32,
        words: []const u32,
    };

    /// Per-RT blend state (already mapped from the HAL BlendState to PIPE_BLEND_* /
    /// PIPE_BLENDFACTOR_* ordinals). Default = disabled with a full colormask, which
    /// is the pure-overwrite passthrough the color path used before.
    pub const Blend = struct {
        enable: bool = false,
        rgb_op: u32 = 0,
        rgb_src: u32 = 1, // PIPE_BLENDFACTOR_ONE
        rgb_dst: u32 = 0x11, // PIPE_BLENDFACTOR_ZERO
        alpha_op: u32 = 0,
        alpha_src: u32 = 1,
        alpha_dst: u32 = 0x11,
        colormask: u32 = 0xf,
        constant: [4]f32 = .{ 0, 0, 0, 0 },
        use_constant: bool = false, // a CONSTANT_* factor is in use -> SET_BLEND_COLOR
    };

    /// Rasterizer cull state (mapped from the HAL CullState). `cull_face` is a
    /// PIPE_FACE mask (0 none / 1 front / 2 back). `front_ccw` sets the front winding.
    pub const Cull = struct {
        cull_face: u32 = 0,
        front_ccw: bool = true,
    };

    /// Description of one gradient-triangle draw lowered to virgl.
    pub const Draw = struct {
        rt_res: u32, // render-target resource id (already created host-side)
        vbuf_res: u32, // vertex-buffer resource id
        fb_w: u32,
        fb_h: u32,
        rt_format: u32, // virgl color format of the RT/surface
        vertex_stride: u32, // bytes per vertex in vbuf_res
        // Two interleaved attributes: position (xy) at offset 0, color (rgba)
        // at color_offset. Formats are virgl FORMAT_* enums.
        pos_format: u32,
        pos_offset: u32,
        color_format: u32,
        color_offset: u32,
        vs_tgsi: []const u8, // NUL-terminated TGSI vertex source
        fs_tgsi: []const u8, // NUL-terminated TGSI fragment source
        clear: [4]f32, // clear color r,g,b,a
        vertex_count: u32,
        mode: u32 = virgl.PRIM_TRIANGLES, // PIPE_PRIM_* (topology)
        instance_count: u32 = 1,
        first_vertex: u32 = 0,
        first_instance: u32 = 0,
        line_width: f32 = 1.0,
        point_size: f32 = 1.0,
        scissor: ?Scissor = null, // null = no scissor test (draw the full RT)
        depth: ?Depth = null, // null = no depth attachment (color-only draw)
        uniforms: []const Uniform = &.{}, // bound uniform blocks (CONST[])
        textures: []const Texture = &.{}, // bound combined-image-samplers (SAMP[])
        samples: u8 = 1, // MSAA sample count of the render target (>1 -> multisample raster)
        blend: Blend = .{}, // alpha-blend state
        cull: Cull = .{}, // back-face cull + front winding
        extra_color_rts: []const ColorRT = &.{}, // MRT targets 1+ (target 0 = rt_res)
    };

    /// Encode a full draw into the buffer. Returns the word count to submit.
    pub fn encodeDraw(self: *Encoder, d: Draw, h: Handles) usize {
        self.len = 0;

        // CREATE + BIND blend. VIRGL_OBJ_BLEND_SIZE = 11. S2[0] carries RT0's blend
        // enable + funcs/factors + colormask (default = disabled, full colormask =
        // the pure-overwrite passthrough).
        self.emitHdr(virgl.CCMD_CREATE_OBJECT, virgl.OBJ_BLEND, 11);
        self.emit(h.blend);
        self.emit(0); // S0
        self.emit(0); // S1
        self.emit(virgl.blendS2(d.blend.enable, d.blend.rgb_op, d.blend.rgb_src, d.blend.rgb_dst, d.blend.alpha_op, d.blend.alpha_src, d.blend.alpha_dst, d.blend.colormask)); // S2[0]
        var cbuf: u32 = 1;
        while (cbuf < 8) : (cbuf += 1) self.emit(0);
        self.emitHdr(virgl.CCMD_BIND_OBJECT, virgl.OBJ_BLEND, 1);
        self.emit(h.blend);

        // SET_BLEND_COLOR (glBlendColor) when a CONSTANT_* factor is in use.
        if (d.blend.use_constant) {
            self.emitHdr(virgl.CCMD_SET_BLEND_COLOR, 0, 4);
            for (d.blend.constant) |ch| self.emitF(ch);
        }

        // CREATE + BIND rasterizer. VIRGL_OBJ_RS_SIZE = 9. Enable the scissor
        // test in S0 when the draw carries a scissor (else virglrenderer ignores
        // the SET_SCISSOR_STATE rectangle).
        const rs_s0: u32 = (1 << 1) | (1 << 29) | // depth_clip, half_pixel_center
            (if (d.cull.front_ccw) @as(u32, 1 << 15) else 0) | // front winding
            virgl.rsS0Cull(d.cull.cull_face) | // back-face culling
            (if (d.scissor != null) virgl.RS_S0_SCISSOR else 0) |
            (if (d.samples > 1) virgl.RS_S0_MULTISAMPLE else 0);
        self.emitHdr(virgl.CCMD_CREATE_OBJECT, virgl.OBJ_RASTERIZER, 9);
        self.emit(h.rasterizer);
        self.emit(rs_s0);
        self.emitF(d.point_size); // point_size (gl_PointSize default)
        self.emit(0); // sprite_coord_enable
        self.emit(0); // S3
        self.emitF(d.line_width); // line_width (glLineWidth)
        self.emitF(0.0); // offset_units
        self.emitF(0.0); // offset_scale
        self.emitF(0.0); // offset_clamp
        self.emitHdr(virgl.CCMD_BIND_OBJECT, virgl.OBJ_RASTERIZER, 1);
        self.emit(h.rasterizer);

        // CREATE + BIND DSA. VIRGL_OBJ_DSA_SIZE = 5. S0 carries the depth-test
        // state when a depth attachment is bound (else depth is disabled, the
        // color-only path). Stencil (S1/S2) stays disabled.
        const dsa_s0: u32 = if (d.depth) |dz|
            virgl.dsaS0(dz.test_enable, dz.write_enable, dz.func)
        else
            0;
        self.emitHdr(virgl.CCMD_CREATE_OBJECT, virgl.OBJ_DSA, 5);
        self.emit(h.dsa);
        self.emit(dsa_s0);
        self.emit(0);
        self.emit(0);
        self.emit(0);
        self.emitHdr(virgl.CCMD_BIND_OBJECT, virgl.OBJ_DSA, 1);
        self.emit(h.dsa);

        // CREATE surface over the primary RT (color target 0). VIRGL_OBJ_SURFACE_SIZE = 5.
        self.emitHdr(virgl.CCMD_CREATE_OBJECT, virgl.OBJ_SURFACE, 5);
        self.emit(h.surface);
        self.emit(d.rt_res);
        self.emit(d.rt_format);
        self.emit(0); // level
        self.emit(0); // layers

        // CREATE a surface per additional MRT color target (targets 1+).
        for (d.extra_color_rts, 0..) |crt, i| {
            self.emitHdr(virgl.CCMD_CREATE_OBJECT, virgl.OBJ_SURFACE, 5);
            self.emit(h.color_rt_base + @as(u32, @intCast(i)));
            self.emit(crt.res_id);
            self.emit(crt.format);
            self.emit(0); // level
            self.emit(0); // layers
        }

        // CREATE a second surface over the depth resource (the ZETA target) when a
        // depth attachment is bound.
        if (d.depth) |dz| {
            self.emitHdr(virgl.CCMD_CREATE_OBJECT, virgl.OBJ_SURFACE, 5);
            self.emit(h.zsurface);
            self.emit(dz.res);
            self.emit(dz.format);
            self.emit(0); // level
            self.emit(0); // layers
        }

        // SET_FRAMEBUFFER_STATE: nr_cbufs = 1 + the MRT extras, zsurf, then each color
        // surface handle. zsurf is the depth surface when bound, else 0.
        const nr_cbufs: u32 = 1 + @as(u32, @intCast(d.extra_color_rts.len));
        self.emitHdr(virgl.CCMD_SET_FRAMEBUFFER_STATE, 0, 2 + nr_cbufs);
        self.emit(nr_cbufs);
        self.emit(if (d.depth != null) h.zsurface else 0);
        self.emit(h.surface); // color target 0
        for (0..d.extra_color_rts.len) |i| self.emit(h.color_rt_base + @as(u32, @intCast(i)));

        // SET_VIEWPORT_STATE: 1 viewport mapping clip [-1,1] to the RT.
        self.emitHdr(virgl.CCMD_SET_VIEWPORT_STATE, 0, 7);
        self.emit(0); // start slot
        self.emitF(@as(f32, @floatFromInt(d.fb_w)) / 2.0); // scale x
        self.emitF(-@as(f32, @floatFromInt(d.fb_h)) / 2.0); // scale y
        self.emitF(0.5); // scale z
        self.emitF(@as(f32, @floatFromInt(d.fb_w)) / 2.0); // translate x
        self.emitF(@as(f32, @floatFromInt(d.fb_h)) / 2.0); // translate y
        self.emitF(0.5); // translate z

        // SET_SCISSOR_STATE: one scissor at slot 0 (min corner, max corner). Only
        // emitted when the draw carries a scissor. The rasterizer S0 scissor bit
        // above gates whether virglrenderer honors it.
        if (d.scissor) |sc| {
            self.emitHdr(virgl.CCMD_SET_SCISSOR_STATE, 0, 3);
            self.emit(0); // start slot
            self.emit(virgl.scissorMinWord(sc.minx, sc.miny));
            self.emit(virgl.scissorMaxWord(sc.maxx, sc.maxy));
        }

        // CREATE + BIND vertex elements: 2 elements * 4 dwords + handle.
        self.emitHdr(virgl.CCMD_CREATE_OBJECT, virgl.OBJ_VERTEX_ELEMENTS, 2 * 4 + 1);
        self.emit(h.vertex_elements);
        // element 0: position
        self.emit(d.pos_offset);
        self.emit(0); // instance_divisor
        self.emit(0); // vertex_buffer_index
        self.emit(d.pos_format);
        // element 1: color
        self.emit(d.color_offset);
        self.emit(0);
        self.emit(0);
        self.emit(d.color_format);
        self.emitHdr(virgl.CCMD_BIND_OBJECT, virgl.OBJ_VERTEX_ELEMENTS, 1);
        self.emit(h.vertex_elements);

        // SET_VERTEX_BUFFERS: 1 buffer * 3 dwords (stride, offset, res handle).
        self.emitHdr(virgl.CCMD_SET_VERTEX_BUFFERS, 0, 1 * 3);
        self.emit(d.vertex_stride);
        self.emit(0); // offset
        self.emit(d.vbuf_res);

        // VS + FS, each bound.
        self.emitShader(h.vs, virgl.SHADER_VERTEX, d.vs_tgsi);
        self.emitHdr(virgl.CCMD_BIND_SHADER, 0, 2);
        self.emit(h.vs);
        self.emit(virgl.SHADER_VERTEX);
        self.emitShader(h.fs, virgl.SHADER_FRAGMENT, d.fs_tgsi);
        self.emitHdr(virgl.CCMD_BIND_SHADER, 0, 2);
        self.emit(h.fs);
        self.emit(virgl.SHADER_FRAGMENT);

        // SET_CONSTANT_BUFFER for each bound uniform block. Body: [shader_type,
        // index, data...]. The std140 bytes go to both stages' constant buffer at
        // `binding` so whichever stage's TGSI declared CONST[binding] reads them.
        for (d.uniforms) |ub| {
            for ([_]u32{ virgl.SHADER_VERTEX, virgl.SHADER_FRAGMENT }) |stype| {
                self.emitHdr(virgl.CCMD_SET_CONSTANT_BUFFER, 0, 2 + @as(u32, @intCast(ub.words.len)));
                self.emit(stype);
                self.emit(ub.binding);
                for (ub.words) |w| self.emit(w);
            }
        }

        // Textures: one sampler-view + sampler-state object per bound texture, then
        // SET_SAMPLER_VIEWS + BIND_SAMPLER_STATES binding them to the fragment stage's
        // SAMP[binding] slots (the TGSI TEX reads SAMP[unit]).
        if (d.textures.len > 0) {
            var max_binding: u32 = 0;
            for (d.textures) |tx| {
                // CREATE sampler view. VIRGL_OBJ_SAMPLER_VIEW_SIZE = 6.
                self.emitHdr(virgl.CCMD_CREATE_OBJECT, virgl.OBJ_SAMPLER_VIEW, 6);
                self.emit(h.sampler_view_base + tx.binding);
                self.emit(tx.res_id);
                self.emit(tx.format | (virgl.PIPE_TEXTURE_2D << 24));
                self.emit(0); // first/last layer
                self.emit(0); // first/last level
                self.emit(virgl.SAMPLER_VIEW_SWIZZLE_IDENTITY);
                // CREATE sampler state. VIRGL_OBJ_SAMPLER_STATE_SIZE = 9.
                self.emitHdr(virgl.CCMD_CREATE_OBJECT, virgl.OBJ_SAMPLER_STATE, 9);
                self.emit(h.sampler_state_base + tx.binding);
                self.emit(virgl.samplerStateS0(tx.wrap_s, tx.wrap_t, tx.min_filter, tx.mip_filter, tx.mag_filter));
                self.emitF(0.0); // lod_bias
                self.emitF(0.0); // min_lod
                self.emitF(0.0); // max_lod (no mip chain)
                self.emitF(0.0); // border r
                self.emitF(0.0); // border g
                self.emitF(0.0); // border b
                self.emitF(0.0); // border a
                if (tx.binding > max_binding) max_binding = tx.binding;
            }
            const count = max_binding + 1;
            // SET_SAMPLER_VIEWS(FRAGMENT, slot 0, one handle per slot, 0 = unbound).
            self.emitHdr(virgl.CCMD_SET_SAMPLER_VIEWS, 0, 2 + count);
            self.emit(virgl.SHADER_FRAGMENT);
            self.emit(0); // start slot
            var slot: u32 = 0;
            while (slot < count) : (slot += 1) {
                var handle: u32 = 0;
                for (d.textures) |tx| {
                    if (tx.binding == slot) handle = h.sampler_view_base + tx.binding;
                }
                self.emit(handle);
            }
            // BIND_SAMPLER_STATES(FRAGMENT, slot 0, one handle per slot).
            self.emitHdr(virgl.CCMD_BIND_SAMPLER_STATES, 0, 2 + count);
            self.emit(virgl.SHADER_FRAGMENT);
            self.emit(0);
            slot = 0;
            while (slot < count) : (slot += 1) {
                var handle: u32 = 0;
                for (d.textures) |tx| {
                    if (tx.binding == slot) handle = h.sampler_state_base + tx.binding;
                }
                self.emit(handle);
            }
        }

        // CLEAR. VIRGL_OBJ_CLEAR_SIZE = 8. Always clears color. Also clears depth
        // when the bound depth attachment carries a clear value (the depth is a
        // f64 split into lo/hi dwords, the pipe_clear convention).
        const clear_depth: ?f32 = if (d.depth) |dz| dz.clear else null;
        const clear_mask: u32 = virgl.CLEAR_COLOR0 | (if (clear_depth != null) virgl.CLEAR_DEPTH else 0);
        const depth_bits: u64 = @bitCast(@as(f64, clear_depth orelse 0.0));
        self.emitHdr(virgl.CCMD_CLEAR, 0, 8);
        self.emit(clear_mask);
        self.emitF(d.clear[0]);
        self.emitF(d.clear[1]);
        self.emitF(d.clear[2]);
        self.emitF(d.clear[3]);
        self.emit(@truncate(depth_bits)); // depth lo
        self.emit(@truncate(depth_bits >> 32)); // depth hi
        self.emit(0); // stencil

        // DRAW_VBO. VIRGL_DRAW_VBO_SIZE = 12.
        self.emitHdr(virgl.CCMD_DRAW_VBO, 0, 12);
        self.emit(d.first_vertex); // start
        self.emit(d.vertex_count);
        self.emit(d.mode); // PIPE_PRIM_* (triangles / lines / points)
        self.emit(0); // indexed
        self.emit(@max(d.instance_count, 1)); // instance_count
        self.emit(0); // index_bias
        self.emit(d.first_instance); // start_instance
        self.emit(0); // primitive_restart
        self.emit(0); // restart_index
        self.emit(0); // min_index
        self.emit(0xffffffff); // max_index
        self.emit(0); // count_from_so

        return self.len;
    }

    /// One MSAA resolve, lowered to a virgl BLIT: the multisampled color resource
    /// `src` is resolved into the single-sample `dst` (both `width`x`height`, the
    /// same `format`). virglrenderer performs the sample average when the source
    /// is multisampled and the destination is not. Returns the word count.
    pub const Blit = struct {
        src_res: u32,
        dst_res: u32,
        width: u32,
        height: u32,
        format: u32,
    };
    pub fn encodeBlit(self: *Encoder, b: Blit) usize {
        self.len = 0;
        // BLIT. VIRGL_BLIT_SIZE = 21 body words.
        self.emitHdr(virgl.CCMD_BLIT, 0, 21);
        self.emit(virgl.blitS0(virgl.BLIT_MASK_RGBA, 0)); // mask RGBA, NEAREST filter
        self.emit(0); // scissor minx/miny (disabled)
        self.emit(0); // scissor maxx/maxy
        self.emit(b.dst_res);
        self.emit(0); // dst level
        self.emit(b.format); // dst format
        self.emit(0); // dst box x
        self.emit(0); // dst box y
        self.emit(0); // dst box z
        self.emit(b.width); // dst box w
        self.emit(b.height); // dst box h
        self.emit(1); // dst box d
        self.emit(b.src_res);
        self.emit(0); // src level
        self.emit(b.format); // src format
        self.emit(0); // src box x
        self.emit(0); // src box y
        self.emit(0); // src box z
        self.emit(b.width); // src box w
        self.emit(b.height); // src box h
        self.emit(1); // src box d
        return self.len;
    }
};

// Default passthrough TGSI for the gradient triangle: VS copies position +
// color through. FS emits the interpolated color. These are the hardcoded
// shaders for the initial virgl HAL. A SPIR-V -> TGSI front end replaces them.
pub const VS_PASSTHROUGH: []const u8 =
    \\VERT
    \\DCL IN[0]
    \\DCL IN[1]
    \\DCL OUT[0], POSITION
    \\DCL OUT[1], COLOR
    \\  1: MOV OUT[0], IN[0]
    \\  2: MOV OUT[1], IN[1]
    \\  3: END
    \\
++ "\x00";

pub const FS_COLOR: []const u8 =
    \\FRAG
    \\DCL IN[0], COLOR, COLOR
    \\DCL OUT[0], COLOR
    \\  1: MOV OUT[0], IN[0]
    \\  2: END
    \\
++ "\x00";

test "encodeDraw carries the topology mode + instancing + first-vertex into DRAW_VBO" {
    var buf: [8192]u32 = undefined;
    var enc = Encoder.init(&buf);
    const tgsi = "\x00"; // minimal (the shader-object body content is irrelevant to this check)
    const words = enc.encodeDraw(.{
        .rt_res = 1,
        .vbuf_res = 2,
        .fb_w = 64,
        .fb_h = 64,
        .rt_format = 1,
        .vertex_stride = 8,
        .pos_format = 0,
        .pos_offset = 0,
        .color_format = 0,
        .color_offset = 8,
        .vs_tgsi = tgsi,
        .fs_tgsi = tgsi,
        .clear = .{ 0, 0, 0, 1 },
        .vertex_count = 2,
        .mode = virgl.PRIM_LINES,
        .instance_count = 3,
        .first_vertex = 5,
        .first_instance = 1,
        .line_width = 4.0,
    }, .{});
    // DRAW_VBO is the trailing 13 words: [hdr, start, count, mode, indexed, instance_count,
    // index_bias, start_instance, prim_restart, restart_index, min, max, count_from_so].
    const b = buf[0..words];
    try std.testing.expectEqual(@as(u32, 5), b[words - 12]); // start = first_vertex
    try std.testing.expectEqual(@as(u32, 2), b[words - 11]); // count = vertex_count
    try std.testing.expectEqual(virgl.PRIM_LINES, b[words - 10]); // mode (topology)
    try std.testing.expectEqual(@as(u32, 3), b[words - 8]); // instance_count
    try std.testing.expectEqual(@as(u32, 1), b[words - 6]); // start_instance = first_instance
}

test "encodeDraw emits SET_SCISSOR_STATE + the rasterizer scissor bit only when scissored" {
    const tgsi = "\x00";
    const base = Encoder.Draw{
        .rt_res = 1,
        .vbuf_res = 2,
        .fb_w = 64,
        .fb_h = 64,
        .rt_format = 1,
        .vertex_stride = 8,
        .pos_format = 0,
        .pos_offset = 0,
        .color_format = 0,
        .color_offset = 8,
        .vs_tgsi = tgsi,
        .fs_tgsi = tgsi,
        .clear = .{ 0, 0, 0, 1 },
        .vertex_count = 3,
    };

    // Without a scissor: no SET_SCISSOR_STATE header anywhere, rasterizer S0 has
    // the scissor bit clear.
    var buf0: [8192]u32 = undefined;
    var enc0 = Encoder.init(&buf0);
    const w0 = enc0.encodeDraw(base, .{});
    const rs_hdr = virgl.cmd0(virgl.CCMD_CREATE_OBJECT, virgl.OBJ_RASTERIZER, 9);
    const scissor_hdr = virgl.cmd0(virgl.CCMD_SET_SCISSOR_STATE, 0, 3);
    var found_scissor0 = false;
    var rs_s0_off: usize = 0;
    for (buf0[0..w0], 0..) |word, i| {
        if (word == scissor_hdr) found_scissor0 = true;
        if (word == rs_hdr) rs_s0_off = i + 2; // header, handle, then S0
    }
    try std.testing.expect(!found_scissor0);
    try std.testing.expectEqual(@as(u32, 0), buf0[rs_s0_off] & virgl.RS_S0_SCISSOR);

    // With a scissor: the S0 scissor bit is set and the min/max corners encode.
    var buf1: [8192]u32 = undefined;
    var enc1 = Encoder.init(&buf1);
    var d = base;
    d.scissor = .{ .minx = 10, .miny = 20, .maxx = 40, .maxy = 55 };
    const w1 = enc1.encodeDraw(d, .{});
    var scissor_at: ?usize = 0;
    scissor_at = null;
    var rs_s0_off1: usize = 0;
    for (buf1[0..w1], 0..) |word, i| {
        if (word == scissor_hdr) scissor_at = i;
        if (word == rs_hdr) rs_s0_off1 = i + 2;
    }
    try std.testing.expect(scissor_at != null);
    try std.testing.expectEqual(virgl.RS_S0_SCISSOR, buf1[rs_s0_off1] & virgl.RS_S0_SCISSOR);
    const s = scissor_at.?;
    try std.testing.expectEqual(@as(u32, 0), buf1[s + 1]); // start slot
    try std.testing.expectEqual(virgl.scissorMinWord(10, 20), buf1[s + 2]);
    try std.testing.expectEqual(virgl.scissorMaxWord(40, 55), buf1[s + 3]);
}

test "encodeDraw binds a ZETA surface + depth DSA + depth clear when a depth attachment is present" {
    const tgsi = "\x00";
    const base = Encoder.Draw{
        .rt_res = 1,
        .vbuf_res = 2,
        .fb_w = 64,
        .fb_h = 64,
        .rt_format = virgl.FORMAT_B8G8R8X8_UNORM,
        .vertex_stride = 8,
        .pos_format = 0,
        .pos_offset = 0,
        .color_format = 0,
        .color_offset = 8,
        .vs_tgsi = tgsi,
        .fs_tgsi = tgsi,
        .clear = .{ 0, 0, 0, 1 },
        .vertex_count = 3,
    };

    // Color-only: DSA S0 is 0, framebuffer zsurf is 0, CLEAR mask is color only.
    var buf0: [8192]u32 = undefined;
    var enc0 = Encoder.init(&buf0);
    const w0 = enc0.encodeDraw(base, .{});
    const dsa_hdr = virgl.cmd0(virgl.CCMD_CREATE_OBJECT, virgl.OBJ_DSA, 5);
    const fb_hdr = virgl.cmd0(virgl.CCMD_SET_FRAMEBUFFER_STATE, 0, 3);
    const clear_hdr = virgl.cmd0(virgl.CCMD_CLEAR, 0, 8);
    var dsa0: usize = 0;
    var fb0: usize = 0;
    var clr0: usize = 0;
    for (buf0[0..w0], 0..) |word, i| {
        if (word == dsa_hdr) dsa0 = i;
        if (word == fb_hdr) fb0 = i;
        if (word == clear_hdr) clr0 = i;
    }
    try std.testing.expectEqual(@as(u32, 0), buf0[dsa0 + 2]); // DSA S0
    try std.testing.expectEqual(@as(u32, 0), buf0[fb0 + 2]); // zsurf = 0
    try std.testing.expectEqual(virgl.CLEAR_COLOR0, buf0[clr0 + 1]); // color only

    // Depth-tested (LEQUAL, write on) + a depth clear of 1.0. DSA S0 carries the
    // depth bits, the framebuffer references the zsurface handle, and CLEAR gets
    // the depth bit + the 1.0 depth as an f64 lo/hi pair.
    var buf1: [8192]u32 = undefined;
    var enc1 = Encoder.init(&buf1);
    var d = base;
    const h = Encoder.Handles{};
    d.depth = .{
        .res = 9,
        .format = virgl.FORMAT_Z32_FLOAT,
        .test_enable = true,
        .write_enable = true,
        .func = 3, // PIPE_FUNC_LEQUAL
        .clear = 1.0,
    };
    const w1 = enc1.encodeDraw(d, h);
    var dsa1: usize = 0;
    var fb1: usize = 0;
    var clr1: usize = 0;
    var zsurf_at: ?usize = null;
    const surf_hdr = virgl.cmd0(virgl.CCMD_CREATE_OBJECT, virgl.OBJ_SURFACE, 5);
    for (buf1[0..w1], 0..) |word, i| {
        if (word == dsa_hdr) dsa1 = i;
        if (word == fb_hdr) fb1 = i;
        if (word == clear_hdr) clr1 = i;
        // The ZETA surface is the CREATE_SURFACE whose resource id is the depth res.
        if (word == surf_hdr and i + 2 < w1 and buf1[i + 2] == 9) zsurf_at = i;
    }
    try std.testing.expectEqual(virgl.dsaS0(true, true, 3), buf1[dsa1 + 2]);
    try std.testing.expectEqual(h.zsurface, buf1[fb1 + 2]); // zsurf handle
    try std.testing.expect(zsurf_at != null);
    try std.testing.expectEqual(h.zsurface, buf1[zsurf_at.? + 1]); // surface handle
    try std.testing.expectEqual(virgl.FORMAT_Z32_FLOAT, buf1[zsurf_at.? + 3]); // format
    try std.testing.expectEqual(virgl.CLEAR_COLOR0 | virgl.CLEAR_DEPTH, buf1[clr1 + 1]);
    const depth_bits: u64 = @bitCast(@as(f64, 1.0));
    try std.testing.expectEqual(@as(u32, @truncate(depth_bits)), buf1[clr1 + 6]); // depth lo
    try std.testing.expectEqual(@as(u32, @truncate(depth_bits >> 32)), buf1[clr1 + 7]); // depth hi
}

test "encodeDraw uploads a bound uniform block to both shader stages' CONST[binding]" {
    const tgsi = "\x00";
    // A UBO at binding 0 with four std140 floats (e.g. a color / scale vec4).
    const ubo_words = [_]u32{ @bitCast(@as(f32, 1.0)), @bitCast(@as(f32, 0.5)), @bitCast(@as(f32, 0.25)), @bitCast(@as(f32, 1.0)) };
    const uniforms = [_]Encoder.Uniform{.{ .binding = 0, .words = &ubo_words }};
    var buf: [8192]u32 = undefined;
    var enc = Encoder.init(&buf);
    const words = enc.encodeDraw(.{
        .rt_res = 1,
        .vbuf_res = 2,
        .fb_w = 64,
        .fb_h = 64,
        .rt_format = virgl.FORMAT_B8G8R8X8_UNORM,
        .vertex_stride = 8,
        .pos_format = 0,
        .pos_offset = 0,
        .color_format = 0,
        .color_offset = 8,
        .vs_tgsi = tgsi,
        .fs_tgsi = tgsi,
        .clear = .{ 0, 0, 0, 1 },
        .vertex_count = 3,
        .uniforms = &uniforms,
    }, .{});

    // Expect a SET_CONSTANT_BUFFER for VERTEX (stype 0) and FRAGMENT (stype 1),
    // each at index 0, each carrying the four data words.
    const cbuf_hdr = virgl.cmd0(virgl.CCMD_SET_CONSTANT_BUFFER, 0, 2 + ubo_words.len);
    var seen_vert = false;
    var seen_frag = false;
    for (buf[0..words], 0..) |w, i| {
        if (w != cbuf_hdr) continue;
        const stype = buf[i + 1];
        try std.testing.expectEqual(@as(u32, 0), buf[i + 2]); // index = binding 0
        try std.testing.expectEqual(ubo_words[0], buf[i + 3]); // first data word
        try std.testing.expectEqual(ubo_words[3], buf[i + 6]); // last data word
        if (stype == virgl.SHADER_VERTEX) seen_vert = true;
        if (stype == virgl.SHADER_FRAGMENT) seen_frag = true;
    }
    try std.testing.expect(seen_vert);
    try std.testing.expect(seen_frag);
}

test "encodeDraw creates a sampler view + state and binds them for a bound texture" {
    const tgsi = "\x00";
    const textures = [_]Encoder.Texture{.{
        .binding = 0,
        .res_id = 42,
        .wrap_s = 0,
        .wrap_t = 0,
        .min_filter = 1, // linear
        .mag_filter = 1,
        .mip_filter = 2, // none
    }};
    var buf: [8192]u32 = undefined;
    var enc = Encoder.init(&buf);
    const h = Encoder.Handles{};
    const words = enc.encodeDraw(.{
        .rt_res = 1,
        .vbuf_res = 2,
        .fb_w = 64,
        .fb_h = 64,
        .rt_format = virgl.FORMAT_B8G8R8X8_UNORM,
        .vertex_stride = 8,
        .pos_format = 0,
        .pos_offset = 0,
        .color_format = 0,
        .color_offset = 8,
        .vs_tgsi = tgsi,
        .fs_tgsi = tgsi,
        .clear = .{ 0, 0, 0, 1 },
        .vertex_count = 3,
        .textures = &textures,
    }, h);

    const view_hdr = virgl.cmd0(virgl.CCMD_CREATE_OBJECT, virgl.OBJ_SAMPLER_VIEW, 6);
    const state_hdr = virgl.cmd0(virgl.CCMD_CREATE_OBJECT, virgl.OBJ_SAMPLER_STATE, 9);
    const set_views_hdr = virgl.cmd0(virgl.CCMD_SET_SAMPLER_VIEWS, 0, 3);
    const bind_states_hdr = virgl.cmd0(virgl.CCMD_BIND_SAMPLER_STATES, 0, 3);
    var view_at: ?usize = null;
    var set_at: ?usize = null;
    var bind_at: ?usize = null;
    var state_seen = false;
    for (buf[0..words], 0..) |w, i| {
        if (w == view_hdr) view_at = i;
        if (w == state_hdr) state_seen = true;
        if (w == set_views_hdr) set_at = i;
        if (w == bind_states_hdr) bind_at = i;
    }
    try std.testing.expect(view_at != null);
    try std.testing.expect(state_seen);
    // The sampler view references the texture resource + R8G8B8A8 with the 2D target.
    try std.testing.expectEqual(@as(u32, 42), buf[view_at.? + 2]); // res id
    try std.testing.expectEqual(virgl.FORMAT_R8G8B8A8_UNORM | (virgl.PIPE_TEXTURE_2D << 24), buf[view_at.? + 3]);
    // SET_SAMPLER_VIEWS / BIND_SAMPLER_STATES target the FRAGMENT stage at slot 0
    // with the view/state handles.
    try std.testing.expect(set_at != null);
    try std.testing.expectEqual(virgl.SHADER_FRAGMENT, buf[set_at.? + 1]);
    try std.testing.expectEqual(h.sampler_view_base + 0, buf[set_at.? + 3]);
    try std.testing.expect(bind_at != null);
    try std.testing.expectEqual(h.sampler_state_base + 0, buf[bind_at.? + 3]);
}

test "encodeDraw binds N color surfaces + nr_cbufs=N for MRT" {
    const tgsi = "\x00";
    const extras = [_]Encoder.ColorRT{.{ .res_id = 77, .format = virgl.FORMAT_B8G8R8X8_UNORM }};
    var buf: [8192]u32 = undefined;
    var enc = Encoder.init(&buf);
    const h = Encoder.Handles{};
    const words = enc.encodeDraw(.{
        .rt_res = 1,
        .vbuf_res = 2,
        .fb_w = 64,
        .fb_h = 64,
        .rt_format = virgl.FORMAT_B8G8R8X8_UNORM,
        .vertex_stride = 8,
        .pos_format = 0,
        .pos_offset = 0,
        .color_format = 0,
        .color_offset = 8,
        .vs_tgsi = tgsi,
        .fs_tgsi = tgsi,
        .clear = .{ 0, 0, 0, 1 },
        .vertex_count = 3,
        .extra_color_rts = &extras,
    }, h);

    // The framebuffer has nr_cbufs=2 with cbuf0=surface, cbuf1=color_rt_base.
    // A second surface is created over the extra RT (res 77).
    const fb_hdr = virgl.cmd0(virgl.CCMD_SET_FRAMEBUFFER_STATE, 0, 4); // 2 + nr_cbufs(2)
    const surf_hdr = virgl.cmd0(virgl.CCMD_CREATE_OBJECT, virgl.OBJ_SURFACE, 5);
    var fb: ?usize = null;
    var extra_surf = false;
    for (buf[0..words], 0..) |w, i| {
        if (w == fb_hdr) fb = i;
        if (w == surf_hdr and i + 2 < words and buf[i + 2] == 77) extra_surf = true;
    }
    try std.testing.expect(fb != null);
    try std.testing.expectEqual(@as(u32, 2), buf[fb.? + 1]); // nr_cbufs
    try std.testing.expectEqual(h.surface, buf[fb.? + 3]); // cbuf0
    try std.testing.expectEqual(h.color_rt_base, buf[fb.? + 4]); // cbuf1
    try std.testing.expect(extra_surf); // a surface over the second RT
}

test "encodeDraw programs blend S2 + rasterizer cull from the pipeline state" {
    const tgsi = "\x00";
    const base = Encoder.Draw{
        .rt_res = 1,
        .vbuf_res = 2,
        .fb_w = 64,
        .fb_h = 64,
        .rt_format = virgl.FORMAT_B8G8R8X8_UNORM,
        .vertex_stride = 8,
        .pos_format = 0,
        .pos_offset = 0,
        .color_format = 0,
        .color_offset = 8,
        .vs_tgsi = tgsi,
        .fs_tgsi = tgsi,
        .clear = .{ 0, 0, 0, 1 },
        .vertex_count = 3,
    };
    const blend_hdr = virgl.cmd0(virgl.CCMD_CREATE_OBJECT, virgl.OBJ_BLEND, 11);
    const rs_hdr = virgl.cmd0(virgl.CCMD_CREATE_OBJECT, virgl.OBJ_RASTERIZER, 9);

    // Default (no blend, no cull): S2 = disabled + full colormask, S0 cull bits clear,
    // front_ccw set.
    var buf0: [8192]u32 = undefined;
    var enc0 = Encoder.init(&buf0);
    const w0 = enc0.encodeDraw(base, .{});
    var s2_0: usize = 0;
    var rs0: usize = 0;
    for (buf0[0..w0], 0..) |w, i| {
        if (w == blend_hdr) s2_0 = i + 4; // header, handle, S0, S1, S2
        if (w == rs_hdr) rs0 = i + 2;
    }
    try std.testing.expectEqual(virgl.blendS2(false, 0, 1, 0x11, 0, 1, 0x11, 0xf), buf0[s2_0]);
    try std.testing.expectEqual(@as(u32, 0), buf0[rs0] & virgl.rsS0Cull(0x3)); // no cull bits

    // SRC_ALPHA / ONE_MINUS_SRC_ALPHA over blend + back-face cull + CW front.
    var buf1: [8192]u32 = undefined;
    var enc1 = Encoder.init(&buf1);
    var d = base;
    d.blend = .{ .enable = true, .rgb_src = 3, .rgb_dst = 0x13, .alpha_src = 3, .alpha_dst = 0x13 };
    d.cull = .{ .cull_face = 2, .front_ccw = false };
    const w1 = enc1.encodeDraw(d, .{});
    var s2_1: usize = 0;
    var rs1: usize = 0;
    for (buf1[0..w1], 0..) |w, i| {
        if (w == blend_hdr) s2_1 = i + 4;
        if (w == rs_hdr) rs1 = i + 2;
    }
    try std.testing.expectEqual(virgl.blendS2(true, 0, 3, 0x13, 0, 3, 0x13, 0xf), buf1[s2_1]);
    try std.testing.expectEqual(virgl.rsS0Cull(2), buf1[rs1] & virgl.rsS0Cull(0x3)); // cull back
    try std.testing.expectEqual(@as(u32, 0), buf1[rs1] & (1 << 15)); // front_ccw cleared (CW)
}

test "MSAA: samples>1 sets the rasterizer multisample bit and encodeBlit resolves src->dst" {
    const tgsi = "\x00";
    const base = Encoder.Draw{
        .rt_res = 1,
        .vbuf_res = 2,
        .fb_w = 64,
        .fb_h = 64,
        .rt_format = virgl.FORMAT_B8G8R8X8_UNORM,
        .vertex_stride = 8,
        .pos_format = 0,
        .pos_offset = 0,
        .color_format = 0,
        .color_offset = 8,
        .vs_tgsi = tgsi,
        .fs_tgsi = tgsi,
        .clear = .{ 0, 0, 0, 1 },
        .vertex_count = 3,
    };
    const rs_hdr = virgl.cmd0(virgl.CCMD_CREATE_OBJECT, virgl.OBJ_RASTERIZER, 9);

    // Single-sample: the multisample bit is clear.
    var buf0: [8192]u32 = undefined;
    var enc0 = Encoder.init(&buf0);
    const w0 = enc0.encodeDraw(base, .{});
    var rs0: usize = 0;
    for (buf0[0..w0], 0..) |w, i| if (w == rs_hdr) {
        rs0 = i + 2;
    };
    try std.testing.expectEqual(@as(u32, 0), buf0[rs0] & virgl.RS_S0_MULTISAMPLE);

    // 4x MSAA: the multisample bit is set on the rasterizer S0.
    var buf1: [8192]u32 = undefined;
    var enc1 = Encoder.init(&buf1);
    var d = base;
    d.samples = 4;
    const w1 = enc1.encodeDraw(d, .{});
    var rs1: usize = 0;
    for (buf1[0..w1], 0..) |w, i| if (w == rs_hdr) {
        rs1 = i + 2;
    };
    try std.testing.expectEqual(virgl.RS_S0_MULTISAMPLE, buf1[rs1] & virgl.RS_S0_MULTISAMPLE);

    // The resolve BLIT carries the multisample src + single-sample dst + extents.
    var bbuf: [64]u32 = undefined;
    var benc = Encoder.init(&bbuf);
    const bw = benc.encodeBlit(.{ .src_res = 7, .dst_res = 9, .width = 64, .height = 48, .format = virgl.FORMAT_B8G8R8X8_UNORM });
    try std.testing.expectEqual(virgl.cmd0(virgl.CCMD_BLIT, 0, 21), bbuf[0]);
    try std.testing.expectEqual(@as(u32, 9), bbuf[4]); // dst_res
    try std.testing.expectEqual(@as(u32, 64), bbuf[10]); // dst box w
    try std.testing.expectEqual(@as(u32, 48), bbuf[11]); // dst box h
    try std.testing.expectEqual(@as(u32, 7), bbuf[13]); // src_res
    try std.testing.expectEqual(@as(u32, 64), bbuf[19]); // src box w
    try std.testing.expectEqual(@as(usize, 22), bw); // header + 21 body words
}
