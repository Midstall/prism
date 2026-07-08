//! The software driver's host texture-sampler runtime. A JITed fragment shader that
//! samples a `sampler2D` lowers an `OpImageSampleImplicitLod` to an indirect call through
//! a host function pointer. This file provides that host function (`sampleTexture`) and
//! the `TexDesc` it reads. The ABI mirrors the vulcan tag convention: the host backend
//! supplies the concrete sampler, just as a GPU backend would emit a TEX from the same op.

const std = @import("std");

/// The `req_lod` value the vulcan reader passes for an implicit 2D sample (`texture()`), signalling the
/// sampler to use the rasterizer-computed `desc.implicit_lod` instead. An absurd magnitude no real
/// explicit LOD uses (a textureLod's LOD is a small value). Must match vulcan-spirv/lower.zig's
/// `implicit_lod_sentinel`. A value at or below the threshold is the sentinel.
pub const IMPLICIT_LOD_SENTINEL: f32 = -1.0e30;
pub const IMPLICIT_LOD_SENTINEL_THRESHOLD: f32 = -1.0e29;

/// A sampler's minification/magnification filter.
pub const Filter = enum(u32) { nearest, linear };

/// How a sampler blends mip levels when minified. `none` = base level only; `nearest`
/// snaps to the nearest level; `linear` trilinearly blends the two bracketing levels.
pub const MipFilter = enum(u32) { none, nearest, linear };

/// How an out-of-`[0,1]` texture coordinate is wrapped.
pub const AddressMode = enum(u32) { repeat, clamp_to_edge, mirrored_repeat };

/// The stored texel format the sampler decodes to normalized/linear RGBA f32. rgba8_unorm
/// is the historical 8-bit path; rgba8_srgb decodes the sRGB EOTF on read (so filtering is
/// in LINEAR space, per the GL sRGB-texture rule); rgba16_float / rgba32_float carry IEEE
/// half / single per channel (HDR values outside 0..1 survive), tying into the fp16 HDR
/// render pipeline.
pub const TexFormat = enum(u32) {
    rgba8_unorm,
    rgba8_srgb,
    rgba16_float,
    rgba32_float,

    /// Bytes per texel (all four channels).
    pub fn bytesPerTexel(self: TexFormat) u32 {
        return switch (self) {
            .rgba8_unorm, .rgba8_srgb => 4,
            .rgba16_float => 8,
            .rgba32_float => 16,
        };
    }
};

/// sRGB electro-optical transfer function: decode one 8-bit sRGB-encoded channel to a
/// linear [0,1] float (the standard IEC 61966-2-1 curve). Applied to R/G/B of an sRGB
/// texture on sample so linear filtering + lighting are correct. Alpha stays linear.
fn srgbToLinear(u8v: u8) f32 {
    const s = @as(f32, @floatFromInt(u8v)) / 255.0;
    if (s <= 0.04045) return s / 12.92;
    return std.math.pow(f32, (s + 0.055) / 1.055, 2.4);
}

/// The maximum mip levels a TexDesc tracks (a 2D texture's chain down to 1x1: 16 covers
/// up to 32768 px). The base level is index 0.
pub const MAX_MIP_LEVELS = 16;

/// The bound combined-image-sampler descriptor passed to the host sampler. `pixels`
/// points at the bound image's tightly-packed RGBA8 texels (`pitch` bytes per row);
/// the sampler state (`mag_filter`/`min_filter`/`mip_filter`, address modes) comes from
/// the bound VkSampler. This is an `extern struct` so its layout is the fixed ABI the
/// JITed FS's call expects (the descriptor pointer is opaque to vulcan; the host backend
/// defines its shape).
///
/// Mipmaps: when `mip_filter != .none`, `levels` is the chain length and `level_off[i]`
/// holds the byte offset (within `pixels`) of mip level i (level 0 at offset 0). The
/// base width/height are `width`/`height` and level i is `max(1, width>>i) x
/// max(1, height>>i)`, tightly packed (pitch = level_width*4). `lod` is the per-draw
/// level-of-detail the rasterizer computed from the texture-coordinate footprint
/// (log2 texels-per-pixel). The sampler selects/blends levels around it. `mag_filter`
/// is used when `lod <= 0` (magnification), `min_filter` otherwise.
pub const TexDesc = extern struct {
    pixels: [*]const u8,
    width: u32,
    height: u32,
    // Max anisotropy (GL_TEXTURE_MAX_ANISOTROPY_EXT). 1 = isotropic. Carried through from the HAL
    // binding for ABI parity with the GPU path. The software sampler is isotropic and ignores it.
    max_anisotropy: f32 = 1,
    pitch: u32, // bytes per row of the base level (typically width*4)
    filter: Filter = .nearest, // magnification filter (legacy name; == mag_filter)
    min_filter: Filter = .nearest, // minification filter (within a level)
    mip_filter: MipFilter = .none,
    address_u: AddressMode = .repeat,
    address_v: AddressMode = .repeat,
    levels: u32 = 1, // mip chain length (1 = base only)
    // Mip level range (GL_TEXTURE_BASE_LEVEL / GL_TEXTURE_MAX_LEVEL): sampling clamps the selected
    // level to [base_level, min(levels-1, tex_max_level)]. Defaults span the whole chain.
    base_level: u32 = 0,
    tex_max_level: u32 = 1000,
    // Component swizzle (GL_TEXTURE_SWIZZLE_R/G/B/A): per output channel, the source 0=R 1=G 2=B 3=A
    // 4=zero 5=one. Identity {0,1,2,3} by default. Applied to the final sampled vec4.
    swizzle: [4]u8 = .{ 0, 1, 2, 3 },
    // GL_TEXTURE_MIN_LOD / GL_TEXTURE_MAX_LOD: the effective LOD is clamped to [tex_min_lod, tex_max_lod].
    tex_min_lod: f32 = -1000,
    tex_max_lod: f32 = 1000,
    lod: f32 = 0, // per-draw level-of-detail bias (added to the computed/explicit LOD; usually 0)
    // The rasterizer-computed implicit LOD (log2 texels-per-pixel from the texture-coordinate
    // screen-space derivatives). Used only for an implicit sample (texture()), which the reader
    // marks by passing `req_lod == IMPLICIT_LOD_SENTINEL`. An explicit textureLod ignores it. 0 when
    // the coordinate is computed (not a bare varying) or the texture is not mipmapped.
    implicit_lod: f32 = 0,
    // The rasterizer-computed UV-space texture-coordinate derivatives per screen pixel:
    // [du/dx, dv/dx, du/dy, dv/dy]. Used for anisotropic filtering (max_anisotropy > 1) on an
    // implicit sample: the footprint's major/minor axes give the sample count + the finer LOD. All
    // zero when the coord is computed / the texture is not mipmapped (falls back to isotropic).
    grad_uv: [4]f32 = .{ 0, 0, 0, 0 },
    level_off: [MAX_MIP_LEVELS]u32 = [_]u32{0} ** MAX_MIP_LEVELS,
    /// The stored texel format; the fetch decodes it to linear/normalized RGBA f32. Defaults
    /// to the historical 8-bit unorm path (so existing rgba8 descriptors are unchanged).
    format: TexFormat = .rgba8_unorm,
    /// CUBEMAP: when true this descriptor is a `samplerCube` (6 square faces). `face_off[f]`
    /// holds the byte offset (within `pixels`) of face f, in the GL cube-face order
    /// +X,-X,+Y,-Y,+Z,-Z (face 0..5). `width`/`height` are one face's dimensions (square).
    /// `sampleTextureCube` picks the face from the sampled direction. A 2D descriptor leaves
    /// this false and every field defaulted, so 2D sampling is byte-identical.
    is_cube: bool = false,
    face_off: [6]u32 = [_]u32{0} ** 6,
    /// 3D texture: when true this is a `sampler3D` with `depth` Z-slices of `width`x`height`,
    /// packed slice-major in `pixels`. The vec3-coordinate host sampler trilinearly interpolates
    /// the volume (mutually exclusive with is_cube). `depth` is the slice count.
    is_3d: bool = false,
    depth: u32 = 1,
    /// 2D array texture: when true this is a `sampler2DArray` with `depth` independent 2D layers of
    /// `width`x`height`, packed layer-major in `pixels` (same storage as a 3D texture). The vec3
    /// host sampler's third coordinate is a raw layer index (not normalized): it selects one layer
    /// (nearest, clamped to [0, depth-1]) and samples it as a plain 2D image with no cross-layer
    /// filtering (unlike 3D's trilinear). Mutually exclusive with is_cube / is_3d.
    is_2darray: bool = false,
    /// Depth compare (GL_TEXTURE_COMPARE_MODE == GL_COMPARE_REF_TO_TEXTURE, for sampler2DShadow). When
    /// true, sampleTextureShadow compares the shader reference against the stored depth (channel R)
    /// with `compare_op` and returns 0/1 (or a PCF-averaged fraction) instead of the raw texel. The
    /// ordinary sampleTexture path ignores this. Default false = a normal read.
    compare_enable: bool = false,
    /// The comparison as `reference <op> stored_depth`, encoded as the hal.CompareOp ordinal
    /// (0=never 1=less 2=equal 3=less_or_equal 4=greater 5=not_equal 6=greater_or_equal 7=always).
    /// Default 3 = GL_LEQUAL, the shadow-map convention.
    compare_op: u8 = 3,
};

/// Evaluate a hal.CompareOp ordinal as `reference <op> stored`, returning 1.0 on pass else 0.0.
/// Used by the shadow (sampler2DShadow) depth-compare path.
pub fn compareDepth(op: u8, reference: f32, stored: f32) f32 {
    const pass = switch (op) {
        0 => false, // never
        1 => reference < stored, // less
        2 => reference == stored, // equal
        3 => reference <= stored, // less_or_equal (GL_LEQUAL default)
        4 => reference > stored, // greater
        5 => reference != stored, // not_equal
        6 => reference >= stored, // greater_or_equal
        else => true, // always (7)
    };
    return if (pass) 1.0 else 0.0;
}

/// sampler2DShadow depth compare (SPIR-V OpImageSampleDref): compare the shader reference `dref`
/// against the stored depth (channel R) at (u,v) per `desc.compare_op` and return a scalar: 1.0
/// if the fragment passes (lit), 0.0 if occluded. Sampled at the base level (shadow maps sit there).
/// A nearest filter is a single exact compare. A linear filter is true PCF: the 4 footprint texels
/// are each compared, then the 0/1 results are bilinear-blended (compare-then-filter, the GL-correct
/// order), giving a soft anti-aliased shadow edge. The swizzle is intentionally not applied (GL
/// compares the raw depth). Mip-LOD PCF is a follow-up. Base level covers the common case.
pub fn sampleTextureShadow(desc: *const TexDesc, u: f32, v: f32, req_lod: f32, dref: f32) callconv(.c) f32 {
    _ = req_lod;
    const lw = desc.width;
    const lh = desc.height;
    if (lw == 0 or lh == 0) return 0; // unbound sampler: fully occluded (defined, no crash)
    // The mag/min filter decides single-tap vs PCF. Either being linear enables PCF.
    if (desc.filter != .linear and desc.min_filter != .linear) {
        const x = wrap(@intFromFloat(@floor(u * @as(f32, @floatFromInt(lw)))), lw, desc.address_u);
        const y = wrap(@intFromFloat(@floor(v * @as(f32, @floatFromInt(lh)))), lh, desc.address_v);
        var t: [4]f32 = undefined;
        texel(desc, 0, lw, x, y, &t);
        return compareDepth(desc.compare_op, dref, t[0]);
    }
    // PCF: the linear 2x2 footprint, but compare each tap first, then bilinear-blend the 0/1 results.
    const sx = u * @as(f32, @floatFromInt(lw)) - 0.5;
    const sy = v * @as(f32, @floatFromInt(lh)) - 0.5;
    const x0f = @floor(sx);
    const y0f = @floor(sy);
    const fx = sx - x0f;
    const fy = sy - y0f;
    const x0: i64 = @intFromFloat(x0f);
    const y0: i64 = @intFromFloat(y0f);
    const xa = wrap(x0, lw, desc.address_u);
    const xb = wrap(x0 + 1, lw, desc.address_u);
    const ya = wrap(y0, lh, desc.address_v);
    const yb = wrap(y0 + 1, lh, desc.address_v);
    var t00: [4]f32 = undefined;
    var t10: [4]f32 = undefined;
    var t01: [4]f32 = undefined;
    var t11: [4]f32 = undefined;
    texel(desc, 0, lw, xa, ya, &t00);
    texel(desc, 0, lw, xb, ya, &t10);
    texel(desc, 0, lw, xa, yb, &t01);
    texel(desc, 0, lw, xb, yb, &t11);
    const c00 = compareDepth(desc.compare_op, dref, t00[0]);
    const c10 = compareDepth(desc.compare_op, dref, t10[0]);
    const c01 = compareDepth(desc.compare_op, dref, t01[0]);
    const c11 = compareDepth(desc.compare_op, dref, t11[0]);
    const top = c00 * (1 - fx) + c10 * fx;
    const bot = c01 * (1 - fx) + c11 * fx;
    return top * (1 - fy) + bot * fy;
}

/// samplerCubeShadow depth compare (SPIR-V OpImageSampleDref on a Cube image): sample the cube
/// depth map along direction (x, y, z), then compare the shader reference `dref` against the stored
/// depth (channel R) per `desc.compare_op` and return a scalar: 1.0 if the fragment passes (lit),
/// 0.0 if occluded. Reuses `sampleTextureCube` for the face-select + fetch. A single-tap
/// compare (cube PCF across faces is a follow-up). The swizzle path is unused (compares raw depth).
pub fn sampleTextureCubeShadow(desc: *const TexDesc, x: f32, y: f32, z: f32, req_lod: f32, dref: f32) callconv(.c) f32 {
    if (desc.width == 0 or desc.height == 0) return 0; // unbound sampler: fully occluded (defined, no crash)
    var t: [4]f32 = undefined;
    sampleTextureCube(desc, x, y, z, req_lod, &t);
    return compareDepth(desc.compare_op, dref, t[0]);
}

/// sampler2DArrayShadow depth compare (SPIR-V OpImageSampleDref on a 2D-Arrayed image): sample the
/// depth map of layer `layer` at (u, v), then compare the shader reference `dref` against the stored
/// depth (channel R) per `desc.compare_op` and return a scalar: 1.0 if the fragment passes (lit),
/// 0.0 if occluded. Reuses `sampleTextureCube`, which dispatches on `desc.is_2darray` to sample
/// the right layer. Mirrors sampleTextureCubeShadow (a single-tap compare; array PCF is a follow-up).
pub fn sampleTexture2dArrayShadow(desc: *const TexDesc, u: f32, v: f32, layer: f32, req_lod: f32, dref: f32) callconv(.c) f32 {
    if (desc.width == 0 or desc.height == 0) return 0; // unbound sampler: fully occluded (defined, no crash)
    var t: [4]f32 = undefined;
    sampleTextureCube(desc, u, v, layer, req_lod, &t);
    return compareDepth(desc.compare_op, dref, t[0]);
}

/// Mip level i's dimensions for a `w`x`h` base (each axis halved, floored, clamped to 1).
fn levelSize(w: u32, h: u32, level: u32) [2]u32 {
    const sh: u5 = @intCast(@min(level, 31));
    return .{ @max(1, w >> sh), @max(1, h >> sh) };
}

/// Wrap an integer texel coordinate per the address mode (dim = the axis length).
fn wrap(coord: i64, dim: u32, mode: AddressMode) u32 {
    if (dim == 0) return 0;
    const d: i64 = @intCast(dim);
    switch (mode) {
        .clamp_to_edge => {
            if (coord < 0) return 0;
            if (coord >= d) return @intCast(d - 1);
            return @intCast(coord);
        },
        .repeat => {
            // Euclidean modulo so negative coordinates wrap correctly.
            var m = @mod(coord, d);
            if (m < 0) m += d;
            return @intCast(m);
        },
        .mirrored_repeat => {
            // Reflect at each integer boundary: over a period of 2*dim the coordinate runs
            // 0..dim-1 then dim-1..0. Euclidean-mod into [0, 2*dim) then fold the upper half.
            const period: i64 = 2 * d;
            var m = @mod(coord, period);
            if (m < 0) m += period;
            if (m >= d) m = period - 1 - m;
            return @intCast(m);
        },
    }
}

/// Fetch a single texel of mip `level`, decoded to 4 linear/normalized floats per the
/// descriptor's `format`. `lw`/`lh` are the level's dimensions. The level's pixels start at
/// `desc.level_off[level]` and are tightly packed (pitch = lw * bytesPerTexel).
fn texel(desc: *const TexDesc, level: u32, lw: u32, x: u32, y: u32, out: *[4]f32) void {
    const bpt: usize = desc.format.bytesPerTexel();
    const base = if (level < desc.level_off.len) desc.level_off[level] else 0;
    const off = @as(usize, base) + @as(usize, y) * (@as(usize, lw) * bpt) + @as(usize, x) * bpt;
    const p = desc.pixels;
    switch (desc.format) {
        .rgba8_unorm => inline for (0..4) |c| {
            out[c] = @as(f32, @floatFromInt(p[off + c])) / 255.0;
        },
        .rgba8_srgb => {
            out[0] = srgbToLinear(p[off + 0]);
            out[1] = srgbToLinear(p[off + 1]);
            out[2] = srgbToLinear(p[off + 2]);
            out[3] = @as(f32, @floatFromInt(p[off + 3])) / 255.0; // alpha is linear
        },
        .rgba16_float => inline for (0..4) |c| {
            const bits = @as(u16, p[off + c * 2]) | (@as(u16, p[off + c * 2 + 1]) << 8);
            out[c] = @floatCast(@as(f16, @bitCast(bits)));
        },
        .rgba32_float => inline for (0..4) |c| {
            var bits: u32 = 0;
            inline for (0..4) |b| bits |= @as(u32, p[off + c * 4 + b]) << (8 * b);
            out[c] = @bitCast(bits);
        },
    }
}

/// Sample mip `level` of the texture at `(u, v)` with the given in-level `filter`
/// (nearest or linear), writing RGBA into `out`.
fn sampleLevel(desc: *const TexDesc, level: u32, filter: Filter, u: f32, v: f32, out: *[4]f32) void {
    const lvl = @min(level, if (desc.levels > 0) desc.levels - 1 else 0);
    const sz = levelSize(desc.width, desc.height, lvl);
    const lw = sz[0];
    const lh = sz[1];
    switch (filter) {
        .nearest => {
            const fx = @floor(u * @as(f32, @floatFromInt(lw)));
            const fy = @floor(v * @as(f32, @floatFromInt(lh)));
            const x = wrap(@intFromFloat(fx), lw, desc.address_u);
            const y = wrap(@intFromFloat(fy), lh, desc.address_v);
            texel(desc, lvl, lw, x, y, out);
        },
        .linear => {
            const sx = u * @as(f32, @floatFromInt(lw)) - 0.5;
            const sy = v * @as(f32, @floatFromInt(lh)) - 0.5;
            const x0f = @floor(sx);
            const y0f = @floor(sy);
            const fx = sx - x0f;
            const fy = sy - y0f;
            const x0: i64 = @intFromFloat(x0f);
            const y0: i64 = @intFromFloat(y0f);
            const xa = wrap(x0, lw, desc.address_u);
            const xb = wrap(x0 + 1, lw, desc.address_u);
            const ya = wrap(y0, lh, desc.address_v);
            const yb = wrap(y0 + 1, lh, desc.address_v);
            var t00: [4]f32 = undefined;
            var t10: [4]f32 = undefined;
            var t01: [4]f32 = undefined;
            var t11: [4]f32 = undefined;
            texel(desc, lvl, lw, xa, ya, &t00);
            texel(desc, lvl, lw, xb, ya, &t10);
            texel(desc, lvl, lw, xa, yb, &t01);
            texel(desc, lvl, lw, xb, yb, &t11);
            inline for (0..4) |c| {
                const top = t00[c] * (1 - fx) + t10[c] * fx;
                const bot = t01[c] * (1 - fx) + t11[c] * fx;
                out[c] = top * (1 - fy) + bot * fy;
            }
        },
    }
}

/// Remap the sampled vec4 per the GL_TEXTURE_SWIZZLE_R/G/B/A sources (identity is a no-op). Reads
/// the original channels into a temp first so a swizzle that references an already-overwritten
/// channel (e.g. {a,r,g,b}) is correct.
pub fn applySwizzle(desc: *const TexDesc, out: *[4]f32) void {
    if (desc.swizzle[0] == 0 and desc.swizzle[1] == 1 and desc.swizzle[2] == 2 and desc.swizzle[3] == 3) return;
    const src = out.*;
    inline for (0..4) |c| out[c] = switch (desc.swizzle[c]) {
        0, 1, 2, 3 => src[desc.swizzle[c]],
        4 => 0.0,
        else => 1.0, // 5 = ONE
    };
}

/// The host sampler the JITed fragment shader calls (the fixed `sampler_fn` ABI).
/// Samples the bound 2D image at `(u, v)` per the descriptor's filter + address modes, writing
/// RGBA into `out`. An unbound / zero-size descriptor yields transparent black.
/// `req_lod` is the explicit LOD (textureLod), or `IMPLICIT_LOD_SENTINEL` for an implicit
/// sample (texture()), which uses `desc.implicit_lod` instead. Both add `desc.lod` as a bias.
pub fn sampleTexture(desc: *const TexDesc, u: f32, v: f32, req_lod: f32, out: *[4]f32) callconv(.c) void {
    sampleTexture2DImpl(desc, u, v, req_lod, out);
    applySwizzle(desc, out);
}
fn sampleTexture2DImpl(desc: *const TexDesc, u: f32, v: f32, req_lod: f32, out: *[4]f32) void {
    if (desc.width == 0 or desc.height == 0) {
        out.* = .{ 0, 0, 0, 0 };
        return;
    }
    const top_level: u32 = if (desc.levels > 0) desc.levels - 1 else 0;
    // GL_TEXTURE_MAX_LEVEL caps the coarsest usable level. GL_TEXTURE_BASE_LEVEL the finest (it can
    // never exceed the effective top). Sampling is confined to [base, eff_top].
    const eff_top = @min(top_level, desc.tex_max_level);
    const base = @min(desc.base_level, eff_top);
    const implicit = req_lod <= IMPLICIT_LOD_SENTINEL_THRESHOLD;
    // Anisotropic filtering: on an implicit sample of a mipmapped texture whose sampler asked for
    // anisotropy, take multiple taps along the footprint's major axis at a finer LOD (the derivatives
    // must be known). Otherwise isotropic.
    if (implicit and desc.max_anisotropy > 1.0 and desc.mip_filter != .none and eff_top > base) {
        const g = desc.grad_uv;
        if (g[0] != 0 or g[1] != 0 or g[2] != 0 or g[3] != 0) {
            sampleTextureAniso(desc, u, v, base, eff_top, out);
            return;
        }
    }
    // An implicit sample (the reader's sentinel) takes the rasterizer's derivative LOD. An explicit
    // textureLod takes its own req_lod. Both add the per-draw bias, then clamp to the GL_TEXTURE_MIN/
    // MAX_LOD range (a min_lod > 0 pins even a magnified surface to a coarser mip).
    const base_lod = if (implicit) desc.implicit_lod else req_lod;
    const eff_lod = std.math.clamp(base_lod + desc.lod, desc.tex_min_lod, desc.tex_max_lod);
    sampleAtLod(desc, u, v, eff_lod, base, eff_top, out);
}

/// Sample the mip chain at an effective LOD (isotropic): pick/blend levels per `mip_filter`, filter
/// within a level per `min_filter`/`mag_filter`. `eff_lod <= 0` is magnification (base level).
fn sampleAtLod(desc: *const TexDesc, u: f32, v: f32, eff_lod: f32, base: u32, top: u32, out: *[4]f32) void {
    // The effective level is base + LOD, confined to [base, top] (GL_TEXTURE_BASE_LEVEL /
    // GL_TEXTURE_MAX_LEVEL). Magnification / no-mip samples the BASE level (not always level 0).
    if (desc.mip_filter == .none or top == base or eff_lod <= 0.0) {
        const filt: Filter = if (eff_lod > 0.0) desc.min_filter else desc.filter;
        sampleLevel(desc, base, filt, u, v, out);
        return;
    }
    const span: f32 = @floatFromInt(top - base);
    const lod = std.math.clamp(eff_lod, 0.0, span);
    switch (desc.mip_filter) {
        .none => unreachable,
        .nearest => {
            const lvl: u32 = base + @as(u32, @intFromFloat(@round(lod)));
            sampleLevel(desc, lvl, desc.min_filter, u, v, out);
        },
        .linear => {
            const lo_f = @floor(lod);
            const frac = lod - lo_f;
            const lo: u32 = base + @as(u32, @intFromFloat(lo_f));
            const hi: u32 = @min(lo + 1, top);
            var a: [4]f32 = undefined;
            var b: [4]f32 = undefined;
            sampleLevel(desc, lo, desc.min_filter, u, v, &a);
            sampleLevel(desc, hi, desc.min_filter, u, v, &b);
            inline for (0..4) |c| out[c] = a[c] * (1 - frac) + b[c] * frac;
        },
    }
}

/// Anisotropic filtering: the per-pixel texel footprint is a parallelogram with sides `grad_x` /
/// `grad_y` (the uv derivatives scaled by the texture size). Isotropic filtering uses LOD =
/// log2(longer side) and over-blurs. Aniso samples N points along the major axis at the finer LOD
/// log2(longer/N), where N = min(ceil(longer/shorter), max_anisotropy), and averages them. This
/// keeps the fine axis sharp on grazing surfaces. Uses `desc.grad_uv` = [du/dx, dv/dx, du/dy, dv/dy].
fn sampleTextureAniso(desc: *const TexDesc, u: f32, v: f32, base: u32, top: u32, out: *[4]f32) void {
    const W: f32 = @floatFromInt(desc.width);
    const H: f32 = @floatFromInt(desc.height);
    const dudx = desc.grad_uv[0];
    const dvdx = desc.grad_uv[1];
    const dudy = desc.grad_uv[2];
    const dvdy = desc.grad_uv[3];
    const px = @sqrt((dudx * W) * (dudx * W) + (dvdx * H) * (dvdx * H)); // texels/pixel along screen x
    const py = @sqrt((dudy * W) * (dudy * W) + (dvdy * H) * (dvdy * H)); // and along screen y
    const pmax = @max(px, py);
    var pmin = @min(px, py);
    if (pmax <= 0.0) {
        sampleLevel(desc, base, desc.filter, u, v, out);
        return;
    }
    if (pmin < 1e-6) pmin = pmax; // a degenerate minor axis -> isotropic
    const ratio = std.math.clamp(pmax / pmin, 1.0, desc.max_anisotropy);
    const n: u32 = @min(@as(u32, @intFromFloat(@ceil(ratio))), 16);
    // Each of N taps covers pmax/ratio texels -> the finer LOD (isotropic would use log2(pmax)).
    const eff_lod = @log2(pmax / ratio) + desc.lod;
    // Spread taps along the major screen axis's uv step (the longer-gradient direction).
    const sdu = if (px >= py) dudx else dudy;
    const sdv = if (px >= py) dvdx else dvdy;
    var acc = [4]f32{ 0, 0, 0, 0 };
    var i: u32 = 0;
    while (i < n) : (i += 1) {
        const t = (@as(f32, @floatFromInt(i)) + 0.5) / @as(f32, @floatFromInt(n)) - 0.5;
        var tap: [4]f32 = undefined;
        sampleAtLod(desc, u + sdu * t, v + sdv * t, eff_lod, base, top, &tap);
        inline for (0..4) |c| acc[c] += tap[c];
    }
    const inv: f32 = 1.0 / @as(f32, @floatFromInt(n));
    inline for (0..4) |c| out[c] = acc[c] * inv;
}

/// Map a cube-sample direction to a face index (GL order +X,-X,+Y,-Y,+Z,-Z) and the
/// per-face 2D texture coordinate, following the GL ES cube-face selection (the major axis
/// picks the face. The other two axes, signed per the spec, give sc/tc; u,v = (s/|ma|+1)/2).
/// Returns `.{ face, u, v }`.
fn cubeFaceUv(rx: f32, ry: f32, rz: f32) struct { face: u32, u: f32, v: f32 } {
    const ax = @abs(rx);
    const ay = @abs(ry);
    const az = @abs(rz);
    var face: u32 = 0;
    var sc: f32 = 0;
    var tc: f32 = 0;
    var ma: f32 = 1;
    if (ax >= ay and ax >= az) {
        ma = ax;
        if (rx >= 0) {
            face = 0; // +X
            sc = -rz;
            tc = -ry;
        } else {
            face = 1; // -X
            sc = rz;
            tc = -ry;
        }
    } else if (ay >= az) {
        ma = ay;
        if (ry >= 0) {
            face = 2; // +Y
            sc = rx;
            tc = rz;
        } else {
            face = 3; // -Y
            sc = rx;
            tc = -rz;
        }
    } else {
        ma = az;
        if (rz >= 0) {
            face = 4; // +Z
            sc = rx;
            tc = -ry;
        } else {
            face = 5; // -Z
            sc = -rx;
            tc = -ry;
        }
    }
    if (ma == 0) ma = 1; // a zero direction is undefined. Avoid a div by zero
    return .{ .face = face, .u = (sc / ma + 1.0) * 0.5, .v = (tc / ma + 1.0) * 0.5 };
}

/// The host sampler the JITed fragment shader calls for a `samplerCube` (`sampler_cube_fn`
/// ABI: the same as `sampleTexture` but the coordinate is a three-component direction). It
/// picks the cube face from the direction, then samples that face's image at the derived
/// (u, v). When the cube has a mip chain (`levels > 1`), each face holds its own chain
/// (`face_off[f]` is the face's mip-0; `level_off` are the per-face relative level offsets)
/// and `req_lod` (from textureCubeLod / prefiltered env maps) selects the level. A zero-sized
/// descriptor returns transparent black.
pub fn sampleTextureCube(desc: *const TexDesc, x: f32, y: f32, z: f32, req_lod: f32, out: *[4]f32) callconv(.c) void {
    if (desc.width == 0 or desc.height == 0) {
        out.* = .{ 0, 0, 0, 0 };
        return;
    }
    // A vec3-coordinate sample dispatches on the bound descriptor. A 3D texture (`sampler3D`)
    // trilinearly interpolates the volume. A 2D-array (`sampler2DArray`) selects one layer by the
    // raw index z. A cube (`samplerCube`) picks a face from the direction.
    if (desc.is_2darray) {
        sampleTexture2DArray(desc, x, y, z, out);
        return applySwizzle(desc, out);
    }
    if (desc.is_3d) {
        sampleTexture3D(desc, x, y, z, out);
        return applySwizzle(desc, out);
    }
    const sel = cubeFaceUv(x, y, z);
    const fi = @min(sel.face, 5);
    // A view of the selected face: its data (a single texel, or a mip chain) starts at
    // face_off[face]. The face's mip levels are laid out exactly like a 2D texture, so the
    // 2D sampler handles LOD selection once given the chain + `req_lod`.
    const face_view = TexDesc{
        .pixels = desc.pixels + desc.face_off[fi],
        .width = desc.width,
        .height = desc.height,
        .pitch = desc.pitch,
        .filter = desc.filter,
        .min_filter = desc.min_filter,
        .mip_filter = desc.mip_filter,
        // Cube faces use edge clamping so a bilinear tap never wraps across a face seam.
        .address_u = .clamp_to_edge,
        .address_v = .clamp_to_edge,
        .levels = desc.levels,
        .lod = desc.lod,
        .level_off = desc.level_off,
        .format = desc.format,
    };
    sampleTexture2DImpl(&face_view, sel.u, sel.v, req_lod, out);
    applySwizzle(desc, out);
}

/// The host sampler the JITed fragment shader calls for `textureGather` (the `sampler_gather_fn`
/// ABI: `fn(desc, u, v, comp, out: *[4]f32)`). It returns the single component `comp` (0..3, passed
/// as an f32 the reader synthesized) of each of the 4 texels of the bilinear footprint at the base
/// level, in the GL/Vulkan gather order: out[0]=(i0,j1), out[1]=(i1,j1), out[2]=(i1,j0), out[3]=(i0,j0)
/// (counter-clockwise from the lower-left texel). The footprint is the exact 2x2 the linear filter
/// would tap, so a gather + manual blend reproduces bilinear filtering (used for PCF shadows and
/// custom filters). A GPU backend emits a TLD4 instead of this call. Cube / 3D gather is not modeled
/// (GL textureGather is 2D / 2D-array only for our surface). A zero-sized descriptor is black.
pub fn sampleTextureGather(desc: *const TexDesc, u: f32, v: f32, comp: f32, out: *[4]f32) callconv(.c) void {
    if (desc.width == 0 or desc.height == 0) {
        out.* = .{ 0, 0, 0, 0 };
        return;
    }
    // comp is a small non-negative float. Round to a channel index and clamp defensively.
    const ci: usize = @min(3, @as(usize, @intFromFloat(@round(@max(0.0, comp)))));
    const lw = desc.width;
    const lh = desc.height;
    // Same footprint as sampleLevel's linear branch: the 2x2 texels bracketing (u,v) - 0.5.
    const sx = u * @as(f32, @floatFromInt(lw)) - 0.5;
    const sy = v * @as(f32, @floatFromInt(lh)) - 0.5;
    const x0: i64 = @intFromFloat(@floor(sx));
    const y0: i64 = @intFromFloat(@floor(sy));
    const xa = wrap(x0, lw, desc.address_u); // i0
    const xb = wrap(x0 + 1, lw, desc.address_u); // i1
    const ya = wrap(y0, lh, desc.address_v); // j0
    const yb = wrap(y0 + 1, lh, desc.address_v); // j1
    var t: [4]f32 = undefined;
    // out[0]=(i0,j1) out[1]=(i1,j1) out[2]=(i1,j0) out[3]=(i0,j0), GL gather order.
    texel(desc, 0, lw, xa, yb, &t);
    out[0] = t[ci];
    texel(desc, 0, lw, xb, yb, &t);
    out[1] = t[ci];
    texel(desc, 0, lw, xb, ya, &t);
    out[2] = t[ci];
    texel(desc, 0, lw, xa, ya, &t);
    out[3] = t[ci];
}

/// The host sampler the JITed fragment shader calls for `texelFetch` (the `sampler_fetch_fn` ABI:
/// `fn(desc, x:i32, y:i32, lod:i32, out: *[4]f32)`). Fetches the exact texel at integer coords
/// (x, y) of mip level `lod` with no filtering, no normalization. Out-of-range coords clamp to the
/// level edge (GLSL leaves OOB undefined; clamping avoids an out-of-bounds read). A GPU backend
/// emits a TLD (OpTld). A zero-sized descriptor returns transparent black.
pub fn sampleTextureFetch(desc: *const TexDesc, x: i32, y: i32, lod: i32, out: *[4]f32) callconv(.c) void {
    if (desc.width == 0 or desc.height == 0) {
        out.* = .{ 0, 0, 0, 0 };
        return;
    }
    const max_level: u32 = if (desc.levels > 0) desc.levels - 1 else 0;
    const lvl: u32 = @intCast(std.math.clamp(lod, 0, @as(i32, @intCast(max_level))));
    const sz = levelSize(desc.width, desc.height, lvl);
    const lw = sz[0];
    const lh = sz[1];
    const cx: u32 = @intCast(std.math.clamp(x, 0, @as(i32, @intCast(lw)) - 1));
    const cy: u32 = @intCast(std.math.clamp(y, 0, @as(i32, @intCast(lh)) - 1));
    texel(desc, lvl, lw, cx, cy, out);
    applySwizzle(desc, out);
}

/// The host sampler for a `sampler2DArray` / `sampler3D` `texelFetch` (the `sampler_fetch_array_fn`
/// / `sampler_fetch_3d_fn` ABI: `fn(desc, x:i32, y:i32, z:i32, lod:i32, out)`). Fetches the exact
/// texel at integer (x, y) of layer/slice `z` with no filtering. Array and 3D share this (both store
/// layers/slices layer-major), differing only in the GPU TLD dim. `lod` selects the base level
/// (array/3D texelFetch is almost always base-level; mip-of-slice is not modeled). Out-of-range
/// coords clamp to the edge; a zero-sized descriptor is black.
pub fn sampleTextureFetch3D(desc: *const TexDesc, x: i32, y: i32, z: i32, lod: i32, out: *[4]f32) callconv(.c) void {
    _ = lod;
    if (desc.width == 0 or desc.height == 0) {
        out.* = .{ 0, 0, 0, 0 };
        return;
    }
    const d: u32 = @max(1, desc.depth);
    const cz: u32 = @intCast(std.math.clamp(z, 0, @as(i32, @intCast(d)) - 1));
    const cx: u32 = @intCast(std.math.clamp(x, 0, @as(i32, @intCast(desc.width)) - 1));
    const cy: u32 = @intCast(std.math.clamp(y, 0, @as(i32, @intCast(desc.height)) - 1));
    const bpt: usize = desc.format.bytesPerTexel();
    const slice_bytes: usize = @as(usize, desc.width) * desc.height * bpt;
    const view = TexDesc{
        .pixels = desc.pixels + @as(usize, cz) * slice_bytes,
        .width = desc.width,
        .height = desc.height,
        .pitch = desc.width * @as(u32, @intCast(bpt)),
        .format = desc.format,
    };
    texel(&view, 0, desc.width, cx, cy, out);
    applySwizzle(desc, out);
}

/// Sample a `sampler3D` volume at (u, v, w) in [0,1]. The volume is `depth` Z-slices of
/// `width`x`height` packed slice-major. Each slice is a 2D image. Bilinear within a slice plus a
/// linear blend across the two bracketing slices (trilinear) for a linear filter. Nearest snaps
/// all three axes. All axes clamp to the edge (the color-grading-LUT convention).
fn sampleTexture3D(desc: *const TexDesc, u: f32, v: f32, w: f32, out: *[4]f32) void {
    const d: u32 = @max(1, desc.depth);
    const slice_bytes: usize = @as(usize, desc.width) * desc.height * desc.format.bytesPerTexel();
    const linear = desc.filter == .linear;
    const df: f32 = @floatFromInt(d);
    if (!linear or d == 1) {
        const iz: i64 = @min(@max(@as(i64, @intFromFloat(@floor(w * df))), 0), @as(i64, d) - 1);
        sampleSlice(desc, @intCast(iz), slice_bytes, u, v, linear, out);
        return;
    }
    const sw = w * df - 0.5;
    const z0f = @floor(sw);
    const fz = sw - z0f;
    const z0: i64 = @min(@max(@as(i64, @intFromFloat(z0f)), 0), @as(i64, d) - 1);
    const z1: i64 = @min(@max(@as(i64, @intFromFloat(z0f)) + 1, 0), @as(i64, d) - 1);
    var a: [4]f32 = undefined;
    var b: [4]f32 = undefined;
    sampleSlice(desc, @intCast(z0), slice_bytes, u, v, true, &a);
    sampleSlice(desc, @intCast(z1), slice_bytes, u, v, true, &b);
    inline for (0..4) |c| out[c] = a[c] * (1 - fz) + b[c] * fz;
}

/// Sample a `sampler2DArray` at (u, v, layer). `layer` is a raw index (not normalized): it selects
/// one independent 2D layer, rounded to nearest and clamped to [0, depth-1], sampled as a plain 2D
/// image (bilinear or nearest per the filter). There is no filtering across layers. The GLSL
/// `texture(sampler2DArray, vec3)` layer-select rule is `floor(P.z + 0.5)`. Layers are packed
/// layer-major, identical storage to a 3D volume, so this reuses `sampleSlice`.
fn sampleTexture2DArray(desc: *const TexDesc, u: f32, v: f32, layer: f32, out: *[4]f32) void {
    const d: u32 = @max(1, desc.depth);
    const slice_bytes: usize = @as(usize, desc.width) * desc.height * desc.format.bytesPerTexel();
    const li: i64 = @min(@max(@as(i64, @intFromFloat(@floor(layer + 0.5))), 0), @as(i64, d) - 1);
    sampleSlice(desc, @intCast(li), slice_bytes, u, v, desc.filter == .linear, out);
}

/// Sample slice `iz` of a 3D volume at (u, v) as a plain 2D image (bilinear or nearest).
fn sampleSlice(desc: *const TexDesc, iz: u32, slice_bytes: usize, u: f32, v: f32, linear: bool, out: *[4]f32) void {
    const view = TexDesc{
        .pixels = desc.pixels + @as(usize, iz) * slice_bytes,
        .width = desc.width,
        .height = desc.height,
        .pitch = desc.width * desc.format.bytesPerTexel(),
        .filter = if (linear) .linear else .nearest,
        .address_u = .clamp_to_edge,
        .address_v = .clamp_to_edge,
        .levels = 1,
        .format = desc.format,
    };
    sampleLevel(&view, 0, view.filter, u, v, out);
}

/// The host math function the JITed shader calls for a transcendental ext-inst (vulcan's
/// `math_fn` ABI: `f32 mathFn(op: i32, a: f32, b: f32)`). `op` selects the function (the
/// vulcan-spirv MATH_* selectors). `a` is the operand. `b` is the exponent for pow. A GPU
/// backend would emit a MUFU instead of this call. This is the host interpretation of
/// vulcan's `math_fn` tag, parallel to `sampleTexture` for the sampler tag.
pub const MathOp = enum(i32) { pow = 0, exp = 1, log = 2, exp2 = 3, log2 = 4, sin = 5, cos = 6, _ };

pub fn mathFn(op: i32, a: f32, b: f32) callconv(.c) f32 {
    return switch (@as(MathOp, @enumFromInt(op))) {
        .pow => std.math.pow(f32, a, b),
        .exp => @exp(a),
        .log => @log(a),
        .exp2 => @exp2(a),
        .log2 => @log2(a),
        .sin => @sin(a),
        .cos => @cos(a),
        _ => 0,
    };
}

test "mathFn computes the transcendentals the JIT dispatches" {
    const eps = 1e-4;
    try std.testing.expectApproxEqAbs(@as(f32, 8.0), mathFn(@intFromEnum(MathOp.pow), 2.0, 3.0), eps);
    try std.testing.expectApproxEqAbs(std.math.pow(f32, 0.5, 1.0 / 2.4), mathFn(@intFromEnum(MathOp.pow), 0.5, 1.0 / 2.4), eps);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), mathFn(@intFromEnum(MathOp.exp2), 2.0, 0), eps);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), mathFn(@intFromEnum(MathOp.log2), 8.0, 0), eps);
}

test "sampleTextureCube picks the face from the direction (skybox oracle)" {
    // 6 single-texel faces, one distinct R value each, in GL cube-face order +X,-X,+Y,-Y,+Z,-Z.
    const face_r = [_]u8{ 10, 20, 30, 40, 50, 60 };
    var px: [6 * 4]u8 = undefined;
    for (0..6) |f| {
        px[f * 4 + 0] = face_r[f];
        px[f * 4 + 1] = 0;
        px[f * 4 + 2] = 0;
        px[f * 4 + 3] = 255;
    }
    var desc = TexDesc{ .pixels = &px, .width = 1, .height = 1, .pitch = 4, .filter = .nearest, .is_cube = true };
    for (0..6) |f| desc.face_off[f] = @intCast(f * 4);

    // A direction dominated by each axis must land on the matching face and read its color.
    const cases = [_]struct { d: [3]f32, face: usize }{
        .{ .d = .{ 1, 0, 0 }, .face = 0 }, // +X
        .{ .d = .{ -1, 0, 0 }, .face = 1 }, // -X
        .{ .d = .{ 0, 1, 0 }, .face = 2 }, // +Y
        .{ .d = .{ 0, -1, 0 }, .face = 3 }, // -Y
        .{ .d = .{ 0, 0, 1 }, .face = 4 }, // +Z
        .{ .d = .{ 0, 0, -1 }, .face = 5 }, // -Z
    };
    for (cases) |c| {
        var out: [4]f32 = undefined;
        sampleTextureCube(&desc, c.d[0], c.d[1], c.d[2], 0, &out);
        const want = @as(f32, @floatFromInt(face_r[c.face])) / 255.0;
        try std.testing.expectApproxEqAbs(want, out[0], 1e-4);
        try std.testing.expectApproxEqAbs(@as(f32, 1.0), out[3], 1e-4);
    }
    // A slightly-off direction still resolves by the major axis (here +Z dominates).
    var out: [4]f32 = undefined;
    sampleTextureCube(&desc, 0.2, -0.1, 0.9, 0, &out);
    try std.testing.expectApproxEqAbs(@as(f32, 50.0) / 255.0, out[0], 1e-4);
}

test "sampleTexture3D resolves WITHIN-slice (u,v) and ACROSS-slice (w) - the software 3D reference" {
    // A 2x2x2 volume, slice-major. Slice 0: 4 distinct in-slice colors (red/green/blue/white);
    // slice 1: all black. This is the software reference the nvidia GPU 3D oracles cross-check
    // against. It must resolve both the in-slice (u,v) texel and the slice (w), the exact pair the
    // GPU path regressed on before the coord-alignment fix (see prism-3d-textures memory).
    const vol = [_]u8{
        255, 0, 0, 255, 0, 255, 0, 255, // slice 0 row 0: red, green
        0, 0, 255, 255, 255, 255, 255, 255, // slice 0 row 1: blue, white
        0, 0, 0, 255, 0, 0, 0, 255, // slice 1 row 0: black, black
        0, 0, 0, 255, 0, 0, 0, 255, // slice 1 row 1: black, black
    };
    var desc = TexDesc{ .pixels = &vol, .width = 2, .height = 2, .pitch = 8, .filter = .nearest, .is_3d = true, .depth = 2 };
    var out: [4]f32 = undefined;
    // Nearest, slice 0 (w~0.25): each quadrant (u,v) picks its own in-slice texel.
    sampleTexture3D(&desc, 0.25, 0.25, 0.25, &out); // TL -> red
    try std.testing.expect(out[0] > 0.9 and out[1] < 0.1 and out[2] < 0.1);
    sampleTexture3D(&desc, 0.75, 0.25, 0.25, &out); // TR -> green
    try std.testing.expect(out[1] > 0.9 and out[0] < 0.1 and out[2] < 0.1);
    sampleTexture3D(&desc, 0.25, 0.75, 0.25, &out); // BL -> blue
    try std.testing.expect(out[2] > 0.9 and out[0] < 0.1 and out[1] < 0.1);
    sampleTexture3D(&desc, 0.75, 0.75, 0.25, &out); // BR -> white
    try std.testing.expect(out[0] > 0.9 and out[1] > 0.9 and out[2] > 0.9);
    // Nearest slice 1 (w~0.75) -> black regardless of (u,v).
    sampleTexture3D(&desc, 0.25, 0.25, 0.75, &out);
    try std.testing.expect(out[0] < 0.1 and out[1] < 0.1 and out[2] < 0.1);
    // Linear within slice 0 center (u=v=0.5): bilinear average of the 4 texels -> ~0.5 each.
    desc.filter = .linear;
    sampleTexture3D(&desc, 0.5, 0.5, 0.25, &out);
    inline for (0..3) |c| try std.testing.expectApproxEqAbs(@as(f32, 0.5), out[c], 0.2);
    // Linear across slices at w=0.5: blends slice-0 center (~0.5) with slice-1 black (0) -> ~0.25.
    sampleTexture3D(&desc, 0.5, 0.5, 0.5, &out);
    inline for (0..3) |c| try std.testing.expectApproxEqAbs(@as(f32, 0.25), out[c], 0.2);
}

test "sampleTexture2DArray selects the layer by a RAW index and reads its in-layer (u,v)" {
    // A 2x2x3 array: layer 0 = 4 distinct in-layer colors (red/green/blue/white); layer 1 = all
    // yellow; layer 2 = all magenta. The third coord is a raw layer index (floor(z+0.5)), not
    // normalized; z=0,1,2 pick layers 0,1,2 directly. No cross-layer filtering.
    const vol = [_]u8{
        255, 0, 0, 255, 0, 255, 0, 255, // layer 0 row 0: red, green
        0, 0, 255, 255, 255, 255, 255, 255, // layer 0 row 1: blue, white
        255, 255, 0, 255, 255, 255, 0, 255, // layer 1: yellow
        255, 255, 0, 255, 255, 255, 0, 255,
        255, 0, 255, 255, 255, 0, 255, 255, // layer 2: magenta
        255, 0, 255, 255, 255, 0, 255, 255,
    };
    var desc = TexDesc{ .pixels = &vol, .width = 2, .height = 2, .pitch = 8, .filter = .nearest, .is_2darray = true, .depth = 3 };
    var out: [4]f32 = undefined;
    // Layer 0 (z=0), each quadrant (u,v) reads its own in-layer texel.
    sampleTexture2DArray(&desc, 0.25, 0.25, 0.0, &out); // TL -> red
    try std.testing.expect(out[0] > 0.9 and out[1] < 0.1 and out[2] < 0.1);
    sampleTexture2DArray(&desc, 0.75, 0.75, 0.0, &out); // BR -> white
    try std.testing.expect(out[0] > 0.9 and out[1] > 0.9 and out[2] > 0.9);
    // z rounds to the nearest layer: z=0.9 -> layer 1 (yellow), z=1.4 -> layer 1, z=1.6 -> layer 2.
    sampleTexture2DArray(&desc, 0.5, 0.5, 0.9, &out);
    try std.testing.expect(out[0] > 0.9 and out[1] > 0.9 and out[2] < 0.1); // yellow
    sampleTexture2DArray(&desc, 0.5, 0.5, 2.0, &out);
    try std.testing.expect(out[0] > 0.9 and out[1] < 0.1 and out[2] > 0.9); // magenta
    // The raw index clamps to [0, depth-1]: z=5 -> layer 2 (magenta), z=-1 -> layer 0.
    sampleTexture2DArray(&desc, 0.5, 0.5, 5.0, &out);
    try std.testing.expect(out[0] > 0.9 and out[2] > 0.9 and out[1] < 0.1); // magenta (clamped)
    // Linear within a layer center (u=v=0.5) on layer 0: bilinear average -> ~0.5 each. No cross-layer.
    desc.filter = .linear;
    sampleTexture2DArray(&desc, 0.5, 0.5, 0.0, &out);
    inline for (0..3) |c| try std.testing.expectApproxEqAbs(@as(f32, 0.5), out[c], 0.2);
}

test "nearest sampling reads the 4 quadrants of a 2x2 texture" {
    // 2x2 RGBA8: (255,0,0) (0,255,0) / (0,0,255) (255,255,255).
    const px = [_]u8{
        255, 0, 0, 255, 0, 255, 0, 255, // row 0
        0, 0, 255, 255, 255, 255, 255, 255, // row 1
    };
    var desc = TexDesc{ .pixels = &px, .width = 2, .height = 2, .pitch = 8, .filter = .nearest };
    var out: [4]f32 = undefined;
    // Top-left quadrant -> red.
    sampleTexture(&desc, 0.25, 0.25, 0, &out);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), out[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), out[1], 1e-5);
    // Top-right -> green.
    sampleTexture(&desc, 0.75, 0.25, 0, &out);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), out[1], 1e-5);
    // Bottom-left -> blue.
    sampleTexture(&desc, 0.25, 0.75, 0, &out);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), out[2], 1e-5);
    // Bottom-right -> white.
    sampleTexture(&desc, 0.75, 0.75, 0, &out);
    try std.testing.expect(out[0] > 0.9 and out[1] > 0.9 and out[2] > 0.9);
}

test "sampleTextureFetch reads the EXACT texel at integer coords (no filter), clamps out-of-range" {
    // A 2x2 with 4 distinct texels. texelFetch(x,y) returns exactly that texel with no filter, no
    // normalization. Out-of-range coords clamp to the edge.
    const px = [_]u8{
        255, 0, 0, 255, 0, 255, 0, 255, // (0,0)=red, (1,0)=green
        0, 0, 255, 255, 255, 255, 255, 255, // (0,1)=blue, (1,1)=white
    };
    var desc = TexDesc{ .pixels = &px, .width = 2, .height = 2, .pitch = 8, .filter = .linear };
    var out: [4]f32 = undefined;
    sampleTextureFetch(&desc, 0, 0, 0, &out); // exact (0,0) -> red (linear filter is ignored)
    try std.testing.expect(out[0] > 0.99 and out[1] < 0.01 and out[2] < 0.01);
    sampleTextureFetch(&desc, 1, 0, 0, &out); // (1,0) -> green
    try std.testing.expect(out[1] > 0.99 and out[0] < 0.01 and out[2] < 0.01);
    sampleTextureFetch(&desc, 0, 1, 0, &out); // (0,1) -> blue
    try std.testing.expect(out[2] > 0.99 and out[0] < 0.01 and out[1] < 0.01);
    sampleTextureFetch(&desc, 1, 1, 0, &out); // (1,1) -> white
    try std.testing.expect(out[0] > 0.99 and out[1] > 0.99 and out[2] > 0.99);
    // Out-of-range x=5,y=-3 clamps to (1,0) -> green.
    sampleTextureFetch(&desc, 5, -3, 0, &out);
    try std.testing.expect(out[1] > 0.99 and out[0] < 0.01);
}

test "sampleTextureFetch3D reads the exact texel of a layer/slice by integer (x, y, z)" {
    // A 2x2x3 volume/array: layer 0 = red/green/blue/white (per-texel), layer 1 = yellow, layer 2 = magenta.
    const vol = [_]u8{
        255, 0, 0, 255, 0, 255, 0, 255, // layer 0 row 0: red, green
        0, 0, 255, 255, 255, 255, 255, 255, // layer 0 row 1: blue, white
        255, 255, 0, 255, 255, 255, 0, 255, // layer 1: yellow
        255, 255, 0, 255, 255, 255, 0, 255,
        255, 0, 255, 255, 255, 0, 255, 255, // layer 2: magenta
        255, 0, 255, 255, 255, 0, 255, 255,
    };
    const desc = TexDesc{ .pixels = &vol, .width = 2, .height = 2, .pitch = 8, .depth = 3 };
    var out: [4]f32 = undefined;
    sampleTextureFetch3D(&desc, 0, 0, 0, 0, &out); // layer 0 (0,0) -> red
    try std.testing.expect(out[0] > 0.99 and out[1] < 0.01 and out[2] < 0.01);
    sampleTextureFetch3D(&desc, 1, 1, 0, 0, &out); // layer 0 (1,1) -> white
    try std.testing.expect(out[0] > 0.99 and out[1] > 0.99 and out[2] > 0.99);
    sampleTextureFetch3D(&desc, 0, 0, 1, 0, &out); // layer 1 -> yellow
    try std.testing.expect(out[0] > 0.99 and out[1] > 0.99 and out[2] < 0.01);
    sampleTextureFetch3D(&desc, 1, 0, 2, 0, &out); // layer 2 -> magenta
    try std.testing.expect(out[0] > 0.99 and out[2] > 0.99 and out[1] < 0.01);
    // Out-of-range z clamps to the last layer (magenta).
    sampleTextureFetch3D(&desc, 0, 0, 9, 0, &out);
    try std.testing.expect(out[0] > 0.99 and out[2] > 0.99 and out[1] < 0.01);
}

test "sampleTextureGather returns the 4 footprint texels in GL gather order (i0j1,i1j1,i1j0,i0j0)" {
    // A 2x2 texture with a distinct R and G per texel so the gather order + component select are
    // both observable. Layout is row-major (row 0 = top / v-small, row 1 = bottom / v-large).
    const px = [_]u8{
        25, 10, 0, 255, // (col0,row0)
        51, 20, 0, 255, // (col1,row0)
        76, 30, 0, 255, // (col0,row1)
        102, 40, 0, 255, // (col1,row1)
    };
    var desc = TexDesc{ .pixels = &px, .width = 2, .height = 2, .pitch = 8, .filter = .linear, .address_u = .clamp_to_edge, .address_v = .clamp_to_edge };
    var out: [4]f32 = undefined;
    // At the center the footprint is the whole 2x2. GL order: out = (i0,j1),(i1,j1),(i1,j0),(i0,j0).
    sampleTextureGather(&desc, 0.5, 0.5, 0, &out); // comp 0 = R
    try std.testing.expectApproxEqAbs(@as(f32, 76.0 / 255.0), out[0], 1e-4); // (col0,row1)
    try std.testing.expectApproxEqAbs(@as(f32, 102.0 / 255.0), out[1], 1e-4); // (col1,row1)
    try std.testing.expectApproxEqAbs(@as(f32, 51.0 / 255.0), out[2], 1e-4); // (col1,row0)
    try std.testing.expectApproxEqAbs(@as(f32, 25.0 / 255.0), out[3], 1e-4); // (col0,row0)
    // comp 1 = G, same footprint, same order.
    sampleTextureGather(&desc, 0.5, 0.5, 1, &out);
    try std.testing.expectApproxEqAbs(@as(f32, 30.0 / 255.0), out[0], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 40.0 / 255.0), out[1], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 20.0 / 255.0), out[2], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 10.0 / 255.0), out[3], 1e-4);
    // A gather + a manual average of the 4 components reproduces the bilinear filter's R at center.
    sampleTextureGather(&desc, 0.5, 0.5, 0, &out);
    const avg = (out[0] + out[1] + out[2] + out[3]) / 4.0;
    var lin: [4]f32 = undefined;
    sampleTexture(&desc, 0.5, 0.5, 0, &lin);
    try std.testing.expectApproxEqAbs(avg, lin[0], 1e-4);
}

test "sRGB texture: the sampler decodes the sRGB EOTF on read (linear filtering)" {
    // A 1x1 sRGB texel: byte 188 in RGB (~mid), alpha 255. sRGB byte 188 -> linear ~0.5;
    // alpha stays linear (1.0). Reference: linear 0.5 <-> sRGB ~0.735 <-> 8-bit ~188.
    const px = [_]u8{ 188, 188, 188, 255 };
    var desc = TexDesc{ .pixels = &px, .width = 1, .height = 1, .pitch = 4, .filter = .nearest, .format = .rgba8_srgb };
    var out: [4]f32 = undefined;
    sampleTexture(&desc, 0.5, 0.5, 0, &out);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), out[0], 0.02);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), out[1], 0.02);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), out[2], 0.02);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), out[3], 1e-5); // alpha linear
    // Endpoints: sRGB 0 -> 0, 255 -> 1.
    const px2 = [_]u8{ 0, 255, 0, 0 };
    desc.pixels = &px2;
    sampleTexture(&desc, 0.5, 0.5, 0, &out);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), out[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), out[1], 1e-5);
}

test "half-float (rgba16_float) texture: the sampler reads fp16 texels, HDR values survive" {
    // 1x1 rgba16f: R=2.0, G=0.5, B=-1.0, A=1.0 (values outside 0..1 survive losslessly).
    const vals = [4]f16{ 2.0, 0.5, -1.0, 1.0 };
    var px: [8]u8 = undefined;
    inline for (vals, 0..) |h, c| {
        const bits: u16 = @bitCast(h);
        px[c * 2 + 0] = @truncate(bits);
        px[c * 2 + 1] = @truncate(bits >> 8);
    }
    var desc = TexDesc{ .pixels = &px, .width = 1, .height = 1, .pitch = 8, .filter = .nearest, .format = .rgba16_float };
    var out: [4]f32 = undefined;
    sampleTexture(&desc, 0.5, 0.5, 0, &out);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), out[0], 1e-3);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), out[1], 1e-3);
    try std.testing.expectApproxEqAbs(@as(f32, -1.0), out[2], 1e-3);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), out[3], 1e-3);
}

test "float (rgba32_float) texture: the sampler reads fp32 texels exactly" {
    // 1x1 rgba32f: R=3.5, G=100.0, B=0.0, A=-2.0.
    const vals = [4]f32{ 3.5, 100.0, 0.0, -2.0 };
    var px: [16]u8 = undefined;
    inline for (vals, 0..) |f, c| {
        const bits: u32 = @bitCast(f);
        inline for (0..4) |b| px[c * 4 + b] = @truncate(bits >> (8 * b));
    }
    var desc = TexDesc{ .pixels = &px, .width = 1, .height = 1, .pitch = 16, .filter = .nearest, .format = .rgba32_float };
    var out: [4]f32 = undefined;
    sampleTexture(&desc, 0.5, 0.5, 0, &out);
    try std.testing.expectEqual(@as(f32, 3.5), out[0]);
    try std.testing.expectEqual(@as(f32, 100.0), out[1]);
    try std.testing.expectEqual(@as(f32, 0.0), out[2]);
    try std.testing.expectEqual(@as(f32, -2.0), out[3]);
}

test "mip selection picks the level the LOD names (trilinear blends)" {
    // A 2x2 -> 1x1 chain: level 0 all red, level 1 all blue. (2x2 RGBA8 + 1x1 RGBA8.)
    const px = [_]u8{
        255, 0, 0, 255, 255, 0, 0, 255, // level 0 row 0
        255, 0, 0, 255, 255, 0, 0, 255, // level 0 row 1
        0, 0, 255, 255, // level 1 (1x1) blue
    };
    var desc = TexDesc{
        .pixels = &px,
        .width = 2,
        .height = 2,
        .pitch = 8,
        .filter = .nearest,
        .min_filter = .nearest,
        .mip_filter = .linear,
        .levels = 2,
    };
    desc.level_off = [_]u32{0} ** MAX_MIP_LEVELS;
    desc.level_off[1] = 16; // level 1 starts after level 0 (2*2*4 = 16 bytes)
    var out: [4]f32 = undefined;
    // lod 0 -> level 0 (red).
    desc.lod = 0;
    sampleTexture(&desc, 0.5, 0.5, 0, &out);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), out[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), out[2], 1e-5);
    // lod 1 -> level 1 (blue).
    desc.lod = 1;
    sampleTexture(&desc, 0.5, 0.5, 0, &out);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), out[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), out[2], 1e-5);
    // lod 0.5 -> a 50/50 trilinear blend of red and blue.
    desc.lod = 0.5;
    sampleTexture(&desc, 0.5, 0.5, 0, &out);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), out[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), out[2], 1e-5);
    // mip_filter nearest snaps: lod 0.6 -> level 1 (blue).
    desc.mip_filter = .nearest;
    desc.lod = 0.6;
    sampleTexture(&desc, 0.5, 0.5, 0, &out);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), out[2], 1e-5);
}

test "anisotropic filtering keeps the fine-axis mip on a grazing footprint (isotropic over-blurs)" {
    // An 8x8 texture, 2 mips: level 0 = white, level 1 (4x4) = red. A v-compressed footprint (fine
    // in u, ~16x coarse in v): grad_uv = [du/dx, dv/dx, du/dy, dv/dy] = [1/64, 0, 0, 16/64]. On an 8x8
    // texture that is px=0.125 texels/px (u) and py=2 texels/px (v). Isotropic LOD = log2(2) = 1 ->
    // Red. Anisotropic (16x) uses the fine u-axis LOD ~0 (+ taps along v) -> white. Same scene, both.
    var px: [8 * 8 * 4 + 4 * 4 * 4]u8 = undefined;
    for (0..8 * 8) |p| px[p * 4 ..][0..4].* = .{ 255, 255, 255, 255 }; // level 0: white
    const l1 = 8 * 8 * 4;
    for (0..4 * 4) |p| px[l1 + p * 4 ..][0..4].* = .{ 255, 0, 0, 255 }; // level 1: red
    var desc = TexDesc{
        .pixels = &px,
        .width = 8,
        .height = 8,
        .pitch = 32,
        .filter = .linear,
        .min_filter = .linear,
        .mip_filter = .nearest,
        .levels = 2,
        .address_u = .repeat,
        .address_v = .repeat,
    };
    desc.level_off = [_]u32{0} ** MAX_MIP_LEVELS;
    desc.level_off[1] = l1;
    desc.implicit_lod = 1.0; // the isotropic LOD the rasterizer would compute (log2(2))
    desc.grad_uv = .{ 1.0 / 64.0, 0, 0, 16.0 / 64.0 };
    var out: [4]f32 = undefined;
    // Isotropic (max_anisotropy = 1): the coarse-axis LOD 1 -> RED.
    desc.max_anisotropy = 1;
    sampleTexture(&desc, 0.5, 0.5, IMPLICIT_LOD_SENTINEL, &out);
    try std.testing.expect(out[0] > 0.9 and out[1] < 0.1 and out[2] < 0.1); // red
    // Anisotropic (16x): the fine u-axis LOD ~0 (with taps along v) -> WHITE.
    desc.max_anisotropy = 16;
    sampleTexture(&desc, 0.5, 0.5, IMPLICIT_LOD_SENTINEL, &out);
    try std.testing.expect(out[0] > 0.9 and out[1] > 0.9 and out[2] > 0.9); // white
}

test "explicit LOD (textureLod): the req_lod arg selects the mip level" {
    // Same 2x2(red) -> 1x1(blue) chain, but the level is chosen by the explicit req_lod arg
    // (what OpImageSampleExplicitLod / textureLod threads) rather than the per-draw desc.lod.
    const px = [_]u8{
        255, 0, 0,   255, 255, 0, 0, 255,
        255, 0, 0,   255, 255, 0, 0, 255,
        0,   0, 255, 255,
    };
    var desc = TexDesc{
        .pixels = &px,
        .width = 2,
        .height = 2,
        .pitch = 8,
        .filter = .nearest,
        .min_filter = .nearest,
        .mip_filter = .nearest,
        .levels = 2,
    };
    desc.level_off = [_]u32{0} ** MAX_MIP_LEVELS;
    desc.level_off[1] = 16;
    var out: [4]f32 = undefined;
    // req_lod 0 (desc.lod stays 0) -> level 0 (red).
    sampleTexture(&desc, 0.5, 0.5, 0.0, &out);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), out[0], 1e-5);
    // req_lod 1 -> level 1 (blue), with NO desc.lod set: proves the explicit arg drives selection.
    sampleTexture(&desc, 0.5, 0.5, 1.0, &out);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), out[2], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), out[0], 1e-5);
}

test "clamp_to_edge holds the border; repeat wraps" {
    const px = [_]u8{ 10, 20, 30, 40, 200, 210, 220, 230 }; // 2x1
    var desc = TexDesc{ .pixels = &px, .width = 2, .height = 1, .pitch = 8, .filter = .nearest, .address_u = .clamp_to_edge };
    var out: [4]f32 = undefined;
    // u=-0.5 clamps to texel 0.
    sampleTexture(&desc, -0.5, 0.5, 0, &out);
    try std.testing.expectApproxEqAbs(@as(f32, 10.0 / 255.0), out[0], 1e-5);
    // u=1.4 clamps to the last texel (1).
    sampleTexture(&desc, 1.4, 0.5, 0, &out);
    try std.testing.expectApproxEqAbs(@as(f32, 200.0 / 255.0), out[0], 1e-5);
    // With repeat, u=1.25 wraps to texel 0.
    desc.address_u = .repeat;
    sampleTexture(&desc, 1.25, 0.5, 0, &out);
    try std.testing.expectApproxEqAbs(@as(f32, 10.0 / 255.0), out[0], 1e-5);
}

test "mirrored_repeat reflects at each boundary (GL_MIRRORED_REPEAT)" {
    // A 2x1 texture: texel 0 = 10, texel 1 = 200. Over u in [0,4) the mirrored pattern of the
    // Nearest texel index is: [0]=t0, [1]=t1, [2]=t1(reflected), [3]=t0, then repeats.
    const px = [_]u8{ 10, 20, 30, 40, 200, 210, 220, 230 };
    var desc = TexDesc{ .pixels = &px, .width = 2, .height = 1, .pitch = 8, .filter = .nearest, .address_u = .mirrored_repeat };
    var out: [4]f32 = undefined;
    const at = struct {
        fn r(d: *const TexDesc, u: f32, o: *[4]f32) f32 {
            sampleTexture(d, u, 0.5, 0, o);
            return o[0] * 255.0;
        }
    }.r;
    // u in [0,1): texel 0 (10). u in [1,2): texel 1 (200). u in [2,3): reflected -> texel 1 (200).
    // u in [3,4): reflected -> texel 0 (10). Negative mirrors symmetrically.
    try std.testing.expectApproxEqAbs(@as(f32, 10), at(&desc, 0.25, &out), 1e-3);
    try std.testing.expectApproxEqAbs(@as(f32, 200), at(&desc, 0.75, &out), 1e-3);
    try std.testing.expectApproxEqAbs(@as(f32, 200), at(&desc, 1.25, &out), 1e-3); // reflected back
    try std.testing.expectApproxEqAbs(@as(f32, 10), at(&desc, 1.75, &out), 1e-3); // reflected to t0
    try std.testing.expectApproxEqAbs(@as(f32, 10), at(&desc, 2.25, &out), 1e-3); // period repeats
    try std.testing.expectApproxEqAbs(@as(f32, 10), at(&desc, -0.25, &out), 1e-3); // texel -1 reflects to t0
}
