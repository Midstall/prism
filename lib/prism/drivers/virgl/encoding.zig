//! virgl command-stream encoding constants (command/object ids, formats, binds, pipe targets, cmd0 packer).
//! These are the host virglrenderer protocol values, shared by both the freestanding and Linux transports.
//! Matches conduit's virtio_gpu.virgl namespace. Keeping them here lets the encoder + device stay OS-agnostic.

pub const FORMAT_B8G8R8X8_UNORM: u32 = 2;
pub const FORMAT_B8G8R8A8_UNORM: u32 = 1;
pub const FORMAT_R32G32_FLOAT: u32 = 29;
pub const FORMAT_R32G32B32A32_FLOAT: u32 = 31;
// HDR color formats. These are VIRGL_FORMAT_* ordinals (virgl_hw.h), the values
// virglrenderer's vrend keys on, not the raw Gallium PIPE_FORMAT enum (which diverges
// for the higher formats). Verified against virglrenderer's src/virgl_hw.h.
//   VIRGL_FORMAT_R16G16B16A16_FLOAT = 94   (fp16, the scRGB/linear-HDR workhorse)
//   VIRGL_FORMAT_R10G10B10A2_UNORM  = 8    (10-bit + 2-bit alpha)
//   VIRGL_FORMAT_B10G10R10A2_UNORM  = 131  (the HDR10 scanout BGRA ordering)
//   VIRGL_FORMAT_B10G10R10X2_UNORM  = 233  (10-bit, no alpha)
pub const FORMAT_R16G16B16A16_FLOAT: u32 = 94;
pub const FORMAT_R10G10B10A2_UNORM: u32 = 8;
pub const FORMAT_B10G10R10A2_UNORM: u32 = 131;
pub const FORMAT_B10G10R10X2_UNORM: u32 = 233;
// Depth/stencil formats (VIRGL_FORMAT_* ordinals, virgl_hw.h). The HAL
// depth32_float maps to Z32_FLOAT (a depth-only ZETA surface).
pub const FORMAT_Z32_FLOAT: u32 = 18;
pub const FORMAT_Z24X8_UNORM: u32 = 21;
pub const FORMAT_Z24_UNORM_S8_UINT: u32 = 19;
/// True if a virgl pipe format is a depth/stencil format (ordinals 16..23), so
/// the transport binds the resource as DEPTH_STENCIL rather than a color target.
pub fn isDepthFormat(f: u32) bool {
    return f >= 16 and f <= 23;
}
pub const BIND_DEPTH_STENCIL: u32 = 1 << 0;
pub const BIND_RENDER_TARGET: u32 = 1 << 1;
pub const BIND_SAMPLER_VIEW: u32 = 1 << 3;
pub const BIND_VERTEX_BUFFER: u32 = 1 << 4;
pub const BIND_CONSTANT_BUFFER: u32 = 1 << 6;
pub const BIND_SCANOUT: u32 = 1 << 18;
pub const TEXTURE_2D: u32 = 2;
pub const BUFFER: u32 = 0;
// Gallium PIPE_PRIM_* values (the DRAW_VBO mode field).
pub const PRIM_POINTS: u32 = 0;
pub const PRIM_LINES: u32 = 1;
pub const PRIM_TRIANGLES: u32 = 4;

/// Map a HAL topology to its virgl/Gallium PIPE_PRIM mode.
pub fn primFromTopology(t: @import("../../hal.zig").Topology) u32 {
    return switch (t) {
        .triangle_list => PRIM_TRIANGLES,
        .line_list => PRIM_LINES,
        .point_list => PRIM_POINTS,
    };
}
pub const SHADER_VERTEX: u32 = 0;
pub const SHADER_FRAGMENT: u32 = 1;
// PIPE_CLEAR_* buffer bits for the CLEAR command's mask.
pub const CLEAR_DEPTH: u32 = 1 << 0;
pub const CLEAR_STENCIL: u32 = 1 << 1;
pub const CLEAR_COLOR0: u32 = 1 << 2;

/// Pack the DSA object's S0 word: depth-test enable (bit 0), depth writemask
/// (bit 1), depth compare func (bits 2..4, a Gallium PIPE_FUNC = HAL CompareOp
/// ordinal). Stencil lives in S1/S2 and is left disabled here.
pub fn dsaS0(depth_enable: bool, depth_write: bool, depth_func: u32) u32 {
    return (@as(u32, @intFromBool(depth_enable)) << 0) |
        (@as(u32, @intFromBool(depth_write)) << 1) |
        ((depth_func & 0x7) << 2);
}

pub const CCMD_CREATE_OBJECT: u32 = 1;
pub const CCMD_BIND_OBJECT: u32 = 2;
pub const CCMD_SET_VIEWPORT_STATE: u32 = 4;
pub const CCMD_SET_FRAMEBUFFER_STATE: u32 = 5;
pub const CCMD_SET_VERTEX_BUFFERS: u32 = 6;
pub const CCMD_CLEAR: u32 = 7;
pub const CCMD_DRAW_VBO: u32 = 8;
pub const CCMD_SET_SAMPLER_VIEWS: u32 = 10;
pub const CCMD_BLIT: u32 = 16;
pub const CCMD_SET_CONSTANT_BUFFER: u32 = 12;
pub const CCMD_SET_BLEND_COLOR: u32 = 14;
pub const CCMD_SET_SCISSOR_STATE: u32 = 15;
pub const CCMD_BIND_SAMPLER_STATES: u32 = 18;
pub const CCMD_BIND_SHADER: u32 = 31;

/// Rasterizer S0 bit for scissor-test enable (VIRGL_OBJ_RS_S0_SCISSOR, bit 14).
/// Without this bit set on the bound rasterizer, virglrenderer ignores the
/// SET_SCISSOR_STATE rectangle entirely.
pub const RS_S0_SCISSOR: u32 = 1 << 14;

/// Rasterizer S0 bit for multisample rasterization (VIRGL_OBJ_RS_S0_MULTISAMPLE,
/// bit 25). Set when rendering into a multisampled (nr_samples > 1) surface.
pub const RS_S0_MULTISAMPLE: u32 = 1 << 25;

/// The PIPE_MASK for a color (RGBA) blit: R=1, G=2, B=4, A=8.
pub const BLIT_MASK_RGBA: u32 = 0xf;

/// The rasterizer S0 cull_face field (VIRGL_OBJ_RS_S0_CULL_FACE, bits 8..9): a
/// PIPE_FACE mask (0 = none, 1 = front, 2 = back).
pub fn rsS0Cull(pipe_face: u32) u32 {
    return (pipe_face & 0x3) << 8;
}

/// Pack a per-RT blend S2 word (VIRGL_OBJ_BLEND_S2_RT_*): blend enable (bit 0), RGB
/// func (bits 1..3), RGB src/dst factor (bits 4..8 / 9..13), alpha func (bits 14..16),
/// alpha src/dst factor (bits 17..21 / 22..26), colormask (bits 27..30). The func
/// values are PIPE_BLEND_*, the factors PIPE_BLENDFACTOR_*.
pub fn blendS2(enable: bool, rgb_func: u32, rgb_src: u32, rgb_dst: u32, alpha_func: u32, alpha_src: u32, alpha_dst: u32, colormask: u32) u32 {
    return @as(u32, @intFromBool(enable)) |
        ((rgb_func & 0x7) << 1) |
        ((rgb_src & 0x1f) << 4) |
        ((rgb_dst & 0x1f) << 9) |
        ((alpha_func & 0x7) << 14) |
        ((alpha_src & 0x1f) << 17) |
        ((alpha_dst & 0x1f) << 22) |
        ((colormask & 0xf) << 27);
}

/// Pack the BLIT command's S0 word: the channel mask (bits 0..7) and the filter
/// (bits 8..9, PIPE_TEX_FILTER_*). A multisample resolve uses NEAREST (0).
/// The host performs the sample average.
pub fn blitS0(mask: u32, filter: u32) u32 {
    return (mask & 0xff) | ((filter & 0x3) << 8);
}

/// Pack a scissor rectangle's two body dwords (min corner, max corner). Each
/// dword is xlo in bits 0..15 and yhi in bits 16..31 (VIRGL_OBJ_SET_SCISSOR_*).
/// The max corner is exclusive (max = min + extent), matching pipe_scissor_state.
pub fn scissorMinWord(minx: u32, miny: u32) u32 {
    return (minx & 0xffff) | ((miny & 0xffff) << 16);
}
pub fn scissorMaxWord(maxx: u32, maxy: u32) u32 {
    return (maxx & 0xffff) | ((maxy & 0xffff) << 16);
}
pub const OBJ_BLEND: u32 = 1;
pub const OBJ_RASTERIZER: u32 = 2;
pub const OBJ_DSA: u32 = 3;
pub const OBJ_SHADER: u32 = 4;
pub const OBJ_VERTEX_ELEMENTS: u32 = 5;
pub const OBJ_SAMPLER_VIEW: u32 = 6;
pub const OBJ_SAMPLER_STATE: u32 = 7;
pub const OBJ_SURFACE: u32 = 8;

/// The virgl pipe format for a sampled RGBA8 texture. Prism uploads texel bytes in
/// internal RGBA order, so the sampler view reads R8G8B8A8_UNORM (byte 0 = R) and
/// the fragment shader gets correct RGBA (unlike the BGRA color/scanout formats).
pub const FORMAT_R8G8B8A8_UNORM: u32 = 67;

/// The pipe target ordinal a 2D sampler view carries in the upper 8 bits of its
/// format word (PIPE_TEXTURE_2D = 2).
pub const PIPE_TEXTURE_2D: u32 = 2;

/// The identity RGBA swizzle for a sampler view (PIPE_SWIZZLE X=0,Y=1,Z=2,W=3
/// packed 3 bits each): channel r<-x, g<-y, b<-z, a<-w.
pub const SAMPLER_VIEW_SWIZZLE_IDENTITY: u32 = 0 | (1 << 3) | (2 << 6) | (3 << 9);

/// Pack a sampler-state S0 word from the pipe filter/wrap enums: wrap_s (0..2),
/// wrap_t (3..5), wrap_r (6..8), min filter (9..10), mip filter (11..12), mag
/// filter (13..14). Values are the PIPE_TEX_* ordinals.
pub fn samplerStateS0(wrap_s: u32, wrap_t: u32, min_filter: u32, mip_filter: u32, mag_filter: u32) u32 {
    return (wrap_s & 0x7) | ((wrap_t & 0x7) << 3) | ((wrap_s & 0x7) << 6) | // wrap_r mirrors wrap_s (2D)
        ((min_filter & 0x3) << 9) | ((mip_filter & 0x3) << 11) | ((mag_filter & 0x3) << 13);
}

const CMD0_SHIFT_OBJ: u5 = 8;
const CMD0_SHIFT_LEN: u5 = 16;

/// Pack a virgl command-stream header dword: 8-bit command, 8-bit object type,
/// 16-bit length (in dwords, excluding the header word).
pub fn cmd0(cmd: u32, obj: u32, len: u32) u32 {
    return cmd | (obj << CMD0_SHIFT_OBJ) | (len << CMD0_SHIFT_LEN);
}
