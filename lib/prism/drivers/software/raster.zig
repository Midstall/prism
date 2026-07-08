const std = @import("std");
const hal = @import("../../hal.zig");
const spirv_jit = @import("spirv_jit.zig");
const pipeline = @import("pipeline.zig");
const sampler = @import("sampler.zig");
const GfxOut = spirv_jit.GfxOut;

/// The largest scalar VS input count the gather path supports (vec4 pos + several
/// vec4 attributes). Conservative cap for the fixed-size gather buffer.
pub const MAX_VS_INPUTS: usize = 16;

pub const Vertex = struct { x: f32, y: f32, r: f32, g: f32, b: f32, a: f32 };

/// A 4-lane f32 RGBA color. The non-SPIR-V rendering hot paths (pixel pack, clear,
/// vertex-color interpolation) operate on these so the compiler emits SIMD (NEON on
/// aarch64) for the clamp/scale/round/convert rather than 4 scalar ops per channel.
const F32x4 = @Vector(4, f32);

/// Convert an RGBA `F32x4` (each lane 0..1) to a packed little-endian RGBA8 u32. The
/// clamp -> *255 -> round -> truncate is one vector op per stage (4 lanes at once).
inline fn packRgba(c: F32x4) u32 {
    const lo: F32x4 = @splat(0.0);
    const hi: F32x4 = @splat(1.0);
    const scale: F32x4 = @splat(255.0);
    const clamped = @min(@max(c, lo), hi) * scale;
    const rounded: @Vector(4, u8) = @intFromFloat(@round(clamped));
    return @bitCast(rounded);
}

/// Write a packed RGBA8 pixel (the `F32x4` color) at (x, y).
inline fn putPixelV(pixels: []u8, width: u32, x: u32, y: u32, color: F32x4) void {
    const idx = (@as(usize, y) * width + x) * 4;
    const packed_px = packRgba(color);
    pixels[idx..][0..4].* = @bitCast(packed_px);
}

fn putPixel(pixels: []u8, width: u32, x: u32, y: u32, r: f32, g: f32, b: f32, a: f32) void {
    putPixelV(pixels, width, x, y, F32x4{ r, g, b, a });
}

// Render-target pixel format pack/unpack.
//
// A render target may be RGBA8 (the default, fast path) or a float / partial-channel
// format (R16F/R32F/RG16F/RGBA16F/R8/RG8). These helpers pack an `F32x4` color into,
// and unpack it back out of, one pixel of the given format. UNORM channels clamp to
// [0,1] then scale by 255. Float channels store the raw value without clamping (an HDR /
// out-of-range value survives losslessly, the whole point of a float target).

/// Bytes one pixel of `fmt` occupies in a render target's backing buffer.
pub inline fn pixelBytes(fmt: hal.Format) usize {
    return fmt.bytesPerPixel();
}

inline fn f32ToF16(v: f32) u16 {
    return @bitCast(@as(f16, @floatCast(v)));
}
inline fn f16ToF32(bits: u16) f32 {
    return @floatCast(@as(f16, @bitCast(bits)));
}
inline fn unorm8(v: f32) u8 {
    return @intFromFloat(@round(std.math.clamp(v, 0.0, 1.0) * 255.0));
}

/// Encode a linear channel to sRGB and quantize to u8 (the sRGB transfer function). Used when
/// writing to an rgba8_srgb render target: the shader's linear color is stored sRGB-encoded so the
/// presented / sampled image is gamma-correct (alpha stays linear, per the GL sRGB rule).
inline fn linearToSrgb8(c: f32) u8 {
    const x = std.math.clamp(c, 0.0, 1.0);
    const s = if (x <= 0.0031308) x * 12.92 else 1.055 * std.math.pow(f32, x, 1.0 / 2.4) - 0.055;
    return @intFromFloat(@round(s * 255.0));
}
/// Decode an sRGB u8 channel to linear (the inverse transfer function). Used when reading an
/// rgba8_srgb target back for blending. GL blends sRGB framebuffers in linear space then re-encodes.
inline fn srgbToLinearF(b: u8) f32 {
    const x = @as(f32, @floatFromInt(b)) / 255.0;
    return if (x <= 0.04045) x / 12.92 else std.math.pow(f32, (x + 0.055) / 1.055, 2.4);
}

/// Pack `color` (RGBA, each lane a shader value) into one pixel at byte offset `off`
/// of `pixels`, per `fmt`. Channels the format lacks are dropped. Missing source
/// channels default per the RGBA convention the caller supplies (alpha already 1.0).
pub fn packPixel(pixels: []u8, off: usize, fmt: hal.Format, color: F32x4) void {
    switch (fmt) {
        .rgba8_unorm => {
            const p = packRgba(color);
            pixels[off..][0..4].* = @bitCast(p);
        },
        // sRGB render target: encode the linear shader color to sRGB on write (alpha linear).
        .rgba8_srgb => {
            pixels[off + 0] = linearToSrgb8(color[0]);
            pixels[off + 1] = linearToSrgb8(color[1]);
            pixels[off + 2] = linearToSrgb8(color[2]);
            pixels[off + 3] = unorm8(color[3]);
        },
        .bgra8_unorm => {
            pixels[off + 0] = unorm8(color[2]); // B
            pixels[off + 1] = unorm8(color[1]); // G
            pixels[off + 2] = unorm8(color[0]); // R
            pixels[off + 3] = unorm8(color[3]); // A
        },
        .r8_unorm => pixels[off] = unorm8(color[0]),
        .r8g8_unorm => {
            pixels[off + 0] = unorm8(color[0]);
            pixels[off + 1] = unorm8(color[1]);
        },
        .r16_float => std.mem.writeInt(u16, pixels[off..][0..2], f32ToF16(color[0]), .little),
        .r16g16_float => {
            std.mem.writeInt(u16, pixels[off + 0 ..][0..2], f32ToF16(color[0]), .little);
            std.mem.writeInt(u16, pixels[off + 2 ..][0..2], f32ToF16(color[1]), .little);
        },
        .rgba16_float => {
            inline for (0..4) |c| std.mem.writeInt(u16, pixels[off + c * 2 ..][0..2], f32ToF16(color[c]), .little);
        },
        .r32_float => std.mem.writeInt(u32, pixels[off..][0..4], @bitCast(color[0]), .little),
        .r32g32b32a32_float => {
            inline for (0..4) |c| std.mem.writeInt(u32, pixels[off + c * 4 ..][0..4], @bitCast(color[c]), .little);
        },
        else => {
            // Any other format (e.g. a vertex-attr or packed-10bit format used as a
            // target) falls back to RGBA8 packing in its first 4 bytes.
            const p = packRgba(color);
            const n = @min(@as(usize, 4), pixels.len - off);
            const b: [4]u8 = @bitCast(p);
            @memcpy(pixels[off..][0..n], b[0..n]);
        },
    }
}

/// Unpack one pixel at byte offset `off` of `pixels`, per `fmt`, into RGBA floats.
/// Channels the format lacks read as 0 (G/B) or 1 (A), matching Vulkan's component
/// swizzle defaults for a sampled image.
pub fn unpackPixel(pixels: []const u8, off: usize, fmt: hal.Format) F32x4 {
    return switch (fmt) {
        .rgba8_unorm => F32x4{
            @as(f32, @floatFromInt(pixels[off + 0])) / 255.0,
            @as(f32, @floatFromInt(pixels[off + 1])) / 255.0,
            @as(f32, @floatFromInt(pixels[off + 2])) / 255.0,
            @as(f32, @floatFromInt(pixels[off + 3])) / 255.0,
        },
        .bgra8_unorm => F32x4{
            @as(f32, @floatFromInt(pixels[off + 2])) / 255.0,
            @as(f32, @floatFromInt(pixels[off + 1])) / 255.0,
            @as(f32, @floatFromInt(pixels[off + 0])) / 255.0,
            @as(f32, @floatFromInt(pixels[off + 3])) / 255.0,
        },
        // sRGB target read back (for blending / MSAA resolve): decode to LINEAR (alpha linear).
        .rgba8_srgb => F32x4{
            srgbToLinearF(pixels[off + 0]),
            srgbToLinearF(pixels[off + 1]),
            srgbToLinearF(pixels[off + 2]),
            @as(f32, @floatFromInt(pixels[off + 3])) / 255.0,
        },
        .r8_unorm => F32x4{ @as(f32, @floatFromInt(pixels[off])) / 255.0, 0, 0, 1 },
        .r8g8_unorm => F32x4{
            @as(f32, @floatFromInt(pixels[off + 0])) / 255.0,
            @as(f32, @floatFromInt(pixels[off + 1])) / 255.0,
            0,
            1,
        },
        .r16_float => F32x4{ f16ToF32(std.mem.readInt(u16, pixels[off..][0..2], .little)), 0, 0, 1 },
        .r16g16_float => F32x4{
            f16ToF32(std.mem.readInt(u16, pixels[off + 0 ..][0..2], .little)),
            f16ToF32(std.mem.readInt(u16, pixels[off + 2 ..][0..2], .little)),
            0,
            1,
        },
        .rgba16_float => F32x4{
            f16ToF32(std.mem.readInt(u16, pixels[off + 0 ..][0..2], .little)),
            f16ToF32(std.mem.readInt(u16, pixels[off + 2 ..][0..2], .little)),
            f16ToF32(std.mem.readInt(u16, pixels[off + 4 ..][0..2], .little)),
            f16ToF32(std.mem.readInt(u16, pixels[off + 6 ..][0..2], .little)),
        },
        .r32_float => F32x4{ @bitCast(std.mem.readInt(u32, pixels[off..][0..4], .little)), 0, 0, 1 },
        .r32g32b32a32_float => F32x4{
            @bitCast(std.mem.readInt(u32, pixels[off + 0 ..][0..4], .little)),
            @bitCast(std.mem.readInt(u32, pixels[off + 4 ..][0..4], .little)),
            @bitCast(std.mem.readInt(u32, pixels[off + 8 ..][0..4], .little)),
            @bitCast(std.mem.readInt(u32, pixels[off + 12 ..][0..4], .little)),
        },
        else => F32x4{ 0, 0, 0, 1 },
    };
}

/// Clear `pixels` (a single-sample image of `fmt`) to `color`, format-aware.
pub fn clearFmt(pixels: []u8, width: u32, height: u32, fmt: hal.Format, color: hal.Color) void {
    if (width == 0 or height == 0) return;
    if (fmt == .rgba8_unorm) return clear(pixels, width, height, color);
    const bpp = pixelBytes(fmt);
    const c = F32x4{ color.r, color.g, color.b, color.a };
    const total = @as(usize, width) * height;
    var i: usize = 0;
    while (i < total) : (i += 1) packPixel(pixels, i * bpp, fmt, c);
}

/// Clear only the pixels inside `rect` (framebuffer pixels, top-left origin), all
/// `samples` samples of each (sample-minor layout: pixel p's samples live at p*samples+s).
/// This is glClear under an enabled scissor test. `width`/`height` are the surface's
/// single-sample dimensions. The rect is clipped to the surface first.
pub fn clearFmtScissor(pixels: []u8, width: u32, height: u32, samples: u8, fmt: hal.Format, color: hal.Color, rect: hal.ScissorRect) void {
    if (width == 0 or height == 0) return;
    const bpp = pixelBytes(fmt);
    const ns: usize = if (samples > 1) samples else 1;
    const c = F32x4{ color.r, color.g, color.b, color.a };
    const sx1: i64 = @as(i64, rect.x) + @as(i64, rect.width); // exclusive
    const sy1: i64 = @as(i64, rect.y) + @as(i64, rect.height); // exclusive
    if (sx1 <= 0 or sy1 <= 0) return;
    const x0: u32 = @intCast(@max(@as(i64, 0), @as(i64, rect.x)));
    const y0: u32 = @intCast(@max(@as(i64, 0), @as(i64, rect.y)));
    const x1: u32 = @intCast(@min(@as(i64, width), sx1));
    const y1: u32 = @intCast(@min(@as(i64, height), sy1));
    if (x0 >= x1 or y0 >= y1) return;
    var y: u32 = y0;
    while (y < y1) : (y += 1) {
        var x: u32 = x0;
        while (x < x1) : (x += 1) {
            const p = (@as(usize, y) * width + x) * ns;
            var s: usize = 0;
            while (s < ns) : (s += 1) packPixel(pixels, (p + s) * bpp, fmt, c);
        }
    }
}

// --- MSAA standard sample positions -----------------------------------------
//
// The Vulkan "standard sample locations" for 2x and 4x MSAA, in the [0,1] pixel
// square (origin top-left). The rasterizer tests coverage at these offsets from each
// pixel's top-left corner. 1x = a single center sample (0.5, 0.5).
pub const sample_pos_1 = [_][2]f32{.{ 0.5, 0.5 }};
pub const sample_pos_2 = [_][2]f32{ .{ 0.75, 0.75 }, .{ 0.25, 0.25 } };
pub const sample_pos_4 = [_][2]f32{ .{ 0.375, 0.125 }, .{ 0.875, 0.375 }, .{ 0.125, 0.625 }, .{ 0.625, 0.875 } };

pub fn samplePositions(samples: u8) []const [2]f32 {
    return switch (samples) {
        2 => &sample_pos_2,
        4 => &sample_pos_4,
        else => &sample_pos_1,
    };
}

/// Resolve an N-sample MSAA color buffer (`src`, laid out as width*height*samples
/// pixels of `fmt`, sample-minor: pixel p's samples are contiguous at
/// (p*samples + s)*bpp) into a single-sample image `dst` of `fmt` by box-averaging
/// the N samples per pixel. This is the implicit resolve a render pass performs at
/// EndRenderPass (pResolveAttachments), turning per-sample coverage into smooth edges.
pub fn resolveMsaa(dst: []u8, src: []const u8, width: u32, height: u32, fmt: hal.Format, samples: u8) void {
    if (width == 0 or height == 0 or samples <= 1) return;
    const bpp = pixelBytes(fmt);
    const ns: usize = samples;
    const inv: f32 = 1.0 / @as(f32, @floatFromInt(samples));
    const total = @as(usize, width) * height;
    var p: usize = 0;
    while (p < total) : (p += 1) {
        var acc = F32x4{ 0, 0, 0, 0 };
        var s: usize = 0;
        while (s < ns) : (s += 1) {
            acc += unpackPixel(src, (p * ns + s) * bpp, fmt);
        }
        const avg = acc * @as(F32x4, @splat(inv));
        packPixel(dst, p * bpp, fmt, avg);
    }
}

pub fn clear(pixels: []u8, width: u32, height: u32, color: hal.Color) void {
    if (width == 0 or height == 0) return;
    // Pack the clear color once, then fill the buffer as a u32 array. `@memset` over the
    // aligned u32 view lets the backend emit wide vector stores (NEON) for the whole span
    // instead of a per-pixel dword store (the profile flagged the per-frame clear).
    const packed_px = packRgba(F32x4{ color.r, color.g, color.b, color.a });
    const total = @as(usize, width) * height;
    const byte_len = total * 4;
    if (byte_len > pixels.len) {
        // Defensive: undersized buffer; fall back to the bounded per-pixel store.
        var i: usize = 0;
        const n = pixels.len / 4;
        while (i < n) : (i += 1) pixels[i * 4 ..][0..4].* = @bitCast(packed_px);
        return;
    }
    const words: []align(1) u32 = std.mem.bytesAsSlice(u32, pixels[0..byte_len]);
    @memset(words, @bitCast(packed_px));
}

fn edge(ax: f32, ay: f32, bx: f32, by: f32, px: f32, py: f32) f32 {
    return (px - ax) * (by - ay) - (py - ay) * (bx - ax);
}

fn ndcToScreen(x: f32, y: f32, width: u32, height: u32) [2]f32 {
    const w: f32 = @floatFromInt(width);
    const h: f32 = @floatFromInt(height);
    return .{ (x * 0.5 + 0.5) * w, (1.0 - (y * 0.5 + 0.5)) * h };
}

pub fn drawTriangle(pixels: []u8, width: u32, height: u32, verts: [3]Vertex) void {
    if (width == 0 or height == 0) return;
    const p0 = ndcToScreen(verts[0].x, verts[0].y, width, height);
    const p1 = ndcToScreen(verts[1].x, verts[1].y, width, height);
    const p2 = ndcToScreen(verts[2].x, verts[2].y, width, height);

    var area = edge(p0[0], p0[1], p1[0], p1[1], p2[0], p2[1]);
    if (area == 0) return;
    // Normalize so the interior test uses >= 0 regardless of winding.
    const sign: f32 = if (area < 0) -1.0 else 1.0;
    area *= sign;

    const minx: u32 = @intFromFloat(@max(0.0, @floor(@min(p0[0], @min(p1[0], p2[0])))));
    const maxx: u32 = @intFromFloat(@min(@as(f32, @floatFromInt(width - 1)), @ceil(@max(p0[0], @max(p1[0], p2[0])))));
    const miny: u32 = @intFromFloat(@max(0.0, @floor(@min(p0[1], @min(p1[1], p2[1])))));
    const maxy: u32 = @intFromFloat(@min(@as(f32, @floatFromInt(height - 1)), @ceil(@max(p0[1], @max(p1[1], p2[1])))));

    var y: u32 = miny;
    while (y <= maxy) : (y += 1) {
        var x: u32 = minx;
        while (x <= maxx) : (x += 1) {
            const px = @as(f32, @floatFromInt(x)) + 0.5;
            const py = @as(f32, @floatFromInt(y)) + 0.5;
            const w0 = edge(p1[0], p1[1], p2[0], p2[1], px, py) * sign;
            const w1 = edge(p2[0], p2[1], p0[0], p0[1], px, py) * sign;
            const w2 = edge(p0[0], p0[1], p1[0], p1[1], px, py) * sign;
            if (w0 >= 0 and w1 >= 0 and w2 >= 0) {
                const b0: F32x4 = @splat(w0 / area);
                const b1: F32x4 = @splat(w1 / area);
                const b2: F32x4 = @splat(w2 / area);
                const c0 = F32x4{ verts[0].r, verts[0].g, verts[0].b, verts[0].a };
                const c1 = F32x4{ verts[1].r, verts[1].g, verts[1].b, verts[1].a };
                const c2 = F32x4{ verts[2].r, verts[2].g, verts[2].b, verts[2].a };
                // Barycentric RGBA blend, 4 channels at once.
                const color = b0 * c0 + b1 * c1 + b2 * c2;
                putPixelV(pixels, width, x, y, color);
            }
        }
    }
}

/// The software depth buffer's minimum resolvable depth step, used to scale glPolygonOffset's
/// constant factor (GL/Vulkan `r`). The depth buffer is f32 in [0,1]. `r` is implementation-
/// defined for a float buffer. We pick a fixed practical step (a `units` of 1 nudges depth
/// by this much). See DepthState.bias_* and the bias_offset computation in rasterShadedFmt.
pub const DEPTH_BIAS_R: f32 = 1.0 / 65536.0;

/// Optional depth attachment for a depth-tested draw: the per-pixel f32 depth buffer
/// (width*height, cleared at render-pass begin) plus the pipeline's depth state.
/// When absent (null), the rasterizer does no depth test and behaves exactly as the
/// color-only path. `screen_z` carries each vertex's NDC depth in [0,1].
pub const DepthAttachment = struct {
    buffer: []f32,
    state: hal.DepthState,
    screen_z: [3]f32,
    /// MSAA sample count of the depth buffer (matches the color target's). 1 = a
    /// single depth per pixel; N>1 = N contiguous depths per pixel (sample-minor),
    /// so a per-sample depth test is possible.
    samples: u8 = 1,
};

/// Optional stencil attachment for a stencil-tested draw: a per-pixel u8 buffer
/// (width*height, single-sample) plus the pipeline's stencil state. When absent (null)
/// the rasterizer does no stencil work. Only the single-sample path supports stencil
/// (a UI clip/mask does not need MSAA). The caller takes the single-sample path when a
/// stencil attachment is present.
pub const StencilAttachment = struct {
    buffer: []u8,
    /// The stencil state for this triangle's face. The caller (drawShaded) already selected
    /// front vs back (two-sided stencil) from the triangle's winding, so the raster is
    /// face-agnostic.
    state: hal.StencilState,
};

/// GL stencil comparison: passes iff `(ref) op (val)` (both already masked by the
/// compare mask). `ref` is the reference value, `val` the stored stencil value.
fn compareStencil(op: hal.CompareOp, ref: u8, val: u8) bool {
    return switch (op) {
        .never => false,
        .less => ref < val,
        .equal => ref == val,
        .less_or_equal => ref <= val,
        .greater => ref > val,
        .not_equal => ref != val,
        .greater_or_equal => ref >= val,
        .always => true,
    };
}

/// Compute the new stencil value for a given op (mirrors glStencilOp / VkStencilOp).
fn applyStencilOp(op: hal.StencilOp, stored: u8, ref: u8) u8 {
    return switch (op) {
        .keep => stored,
        .zero => 0,
        .replace => ref,
        .incr_clamp => if (stored == 0xff) 0xff else stored + 1,
        .decr_clamp => if (stored == 0) 0 else stored - 1,
        .invert => ~stored,
        .incr_wrap => stored +% 1,
        .decr_wrap => stored -% 1,
    };
}

/// Apply a stencil write under the write mask: only the masked bits of `new_val`
/// replace the corresponding bits of `stored`.
fn stencilWrite(stored: u8, new_val: u8, write_mask: u8) u8 {
    return (stored & ~write_mask) | (new_val & write_mask);
}

/// Evaluate a `BlendFactor` into the RGBA 4-vector it scales (`src`/`dst` are the
/// fragment / framebuffer colors, `c` the glBlendColor constant). The alpha-only factors
/// broadcast a single channel. `src_alpha_saturate` is min(src.a, 1-dst.a) for RGB, 1 for
/// alpha. This is the GL / NV9097 coefficient definition, used to compute the oracle.
inline fn blendFactor(f: hal.BlendFactor, src: F32x4, dst: F32x4, c: F32x4) F32x4 {
    const one: F32x4 = @splat(1.0);
    return switch (f) {
        .zero => @splat(0.0),
        .one => one,
        .src_color => src,
        .one_minus_src_color => one - src,
        .src_alpha => @splat(src[3]),
        .one_minus_src_alpha => @splat(1.0 - src[3]),
        .dst_alpha => @splat(dst[3]),
        .one_minus_dst_alpha => @splat(1.0 - dst[3]),
        .dst_color => dst,
        .one_minus_dst_color => one - dst,
        .constant_color => c,
        .one_minus_constant_color => one - c,
        .constant_alpha => @splat(c[3]),
        .one_minus_constant_alpha => @splat(1.0 - c[3]),
        .src_alpha_saturate => blk: {
            const s = @min(src[3], 1.0 - dst[3]);
            break :blk F32x4{ s, s, s, 1.0 };
        },
    };
}

/// Combine the factor-weighted source/destination per `op`. `min`/`max` ignore the factors
/// and use the raw source/destination (per the GL blend-equation spec).
inline fn blendCombine(op: hal.BlendOp, s_w: F32x4, d_w: F32x4, s_raw: F32x4, d_raw: F32x4) F32x4 {
    return switch (op) {
        .add => s_w + d_w,
        .subtract => s_w - d_w,
        .reverse_subtract => d_w - s_w,
        .min => @min(s_raw, d_raw),
        .max => @max(s_raw, d_raw),
    };
}

/// Compute the blended RGBA for a fragment: combine the source `src` (the FS output) with
/// the destination `dst` (the framebuffer pixel) per the pipeline's `BlendState`. RGB uses
/// the color factors/op, alpha the alpha factors/op. The result is clamped to [0,1].
/// This is Prism's blend oracle. The nvidia path emits the equivalent NV9097 state.
pub fn applyBlend(b: hal.BlendState, src: F32x4, dst: F32x4) F32x4 {
    const c: F32x4 = b.constant;
    const sf_c = blendFactor(b.src_color, src, dst, c);
    const df_c = blendFactor(b.dst_color, src, dst, c);
    const sf_a = blendFactor(b.src_alpha, src, dst, c);
    const df_a = blendFactor(b.dst_alpha, src, dst, c);
    const rgb = blendCombine(b.color_op, sf_c * src, df_c * dst, src, dst);
    const a = blendCombine(b.alpha_op, sf_a * src, df_a * dst, src, dst);
    const lo: F32x4 = @splat(0.0);
    const hi: F32x4 = @splat(1.0);
    const out = F32x4{ rgb[0], rgb[1], rgb[2], a[3] };
    return @min(@max(out, lo), hi);
}

/// Read the destination pixel at byte offset `off`, blend `src` over it per `b`, and return
/// the color to write. When `b.enable` is false this is a pure passthrough (returns `src`)
/// so the non-blend hot path is unchanged.
inline fn blendSrc(pixels: []const u8, off: usize, fmt: hal.Format, b: hal.BlendState, src: F32x4) F32x4 {
    if (!b.enable) return src;
    const dst = unpackPixel(pixels, off, fmt);
    return applyBlend(b, src, dst);
}

/// Apply a depth comparison: does a fragment at `frag` pass against the stored `stored`?
/// A point sprite being rasterized: the top-left corner (min window x/y) + 1/size, so
/// gl_PointCoord = ((fx - ox) * inv_size, (fy - oy) * inv_size) ranging 0..1 across the
/// sprite, origin at the upper-left (t = 0 at the smaller y, matching GL in this y-down window).
/// Set by `drawLinePoint` around a point draw. Null for triangles/lines (gl_PointCoord = 0).
pub const PointSprite = struct { ox: f32, oy: f32, inv_size: f32 };
pub threadlocal var current_point_sprite: ?PointSprite = null;

/// After the varyings are interpolated into `fs_in`, overwrite any fragment-builtin
/// input slots (gl_FragCoord / gl_PointCoord / gl_FrontFacing, marked by a sentinel in
/// `slots`) with the per-fragment window position / point-sprite coord / face. gl_FragCoord
/// is (x+0.5, y+0.5, z, 1); gl_FrontFacing is 1.0 front / 0.0 back; gl_PointCoord is the
/// point-sprite s/t (0 when not a point sprite).
inline fn fillFragBuiltins(fs_in: []f32, slots: []const u32, n: usize, fx: f32, fy: f32, z: f32, front: f32) void {
    var k: usize = 0;
    while (k < n and k < slots.len) : (k += 1) {
        const s = slots[k];
        if (s < spirv_jit.FRAG_COORD_INPUT_BASE) continue;
        if (s == spirv_jit.POINT_COORD_INPUT_BASE or s == spirv_jit.POINT_COORD_INPUT_BASE + 1) {
            const comp = s - spirv_jit.POINT_COORD_INPUT_BASE;
            fs_in[k] = if (current_point_sprite) |ps|
                (if (comp == 0) (fx + 0.5 - ps.ox) * ps.inv_size else (fy + 0.5 - ps.oy) * ps.inv_size)
            else
                0; // gl_PointCoord is undefined for non-points -> (0, 0)
            continue;
        }
        fs_in[k] = if (s == spirv_jit.FRONT_FACING_INPUT)
            front
        else switch (s - spirv_jit.FRAG_COORD_INPUT_BASE) {
            0 => fx + 0.5,
            1 => fy + 0.5,
            2 => z,
            else => 1.0,
        };
    }
}

/// An additional MRT color attachment (target index 1+): its pixel buffer + format.
pub const ColorTarget = struct { bytes: []u8, format: hal.Format };

/// Write the FS's MRT color outputs to the extra color targets at pixel index `pi`
/// (target t reads color[t*4 .. t*4+4]). Target 0 is written by the caller to `pixels`.
inline fn writeExtraTargets(extra: []const ColorTarget, pi: usize, color: []const f32) void {
    for (extra, 0..) |ct, i| {
        const t = i + 1;
        const cvec: F32x4 = color[t * 4 ..][0..4].*;
        packPixel(ct.bytes, pi * pixelBytes(ct.format), ct.format, cvec);
    }
}

/// Cumulative count of primary-color fragments (samples) written by the rasterizer: a fragment
/// that passed depth+stencil+coverage. glBeginQuery/glEndQuery (occlusion queries) read the delta
/// of this across a query span. The increment is one add on the write path (negligible).
pub threadlocal var samples_written: u64 = 0;
/// Write `color` to the pixel at `off` honoring the per-channel color write mask: a masked-off
/// channel (write_mask false) keeps the destination value. All-true (the default) writes verbatim
/// (a single packPixel, no read-back). glColorMask / VkColorComponentFlags map here.
inline fn packMasked(pixels: []u8, off: usize, fmt: hal.Format, color: F32x4, mask: [4]bool) void {
    samples_written +%= 1;
    if (mask[0] and mask[1] and mask[2] and mask[3]) {
        packPixel(pixels, off, fmt, color);
        return;
    }
    const dst = unpackPixel(pixels, off, fmt);
    var out = color;
    inline for (0..4) |c| {
        if (!mask[c]) out[c] = dst[c];
    }
    packPixel(pixels, off, fmt, out);
}

fn depthPasses(op: hal.CompareOp, frag: f32, stored: f32) bool {
    return switch (op) {
        .never => false,
        .less => frag < stored,
        .equal => frag == stored,
        .less_or_equal => frag <= stored,
        .greater => frag > stored,
        .not_equal => frag != stored,
        .greater_or_equal => frag >= stored,
        .always => true,
    };
}

/// The per-triangle implicit mip LOD and UV-space texture-coordinate derivatives, for a texture
/// sampled by bare varyings at indices `us`/`vs`. `lod` = `log2(max(rho_x, rho_y))` (rho = the
/// texel-space rate of change of (u, v) per screen pixel). `grad_uv` = [du/dx, dv/dx, du/dy, dv/dy]
/// in UV space (for anisotropic filtering). Uses the same Cramer's-rule plane gradients as dFdx/dFdy.
/// Returns lod 0 + zero grads for a degenerate triangle or a non-positive footprint (magnification).
const TexLodResult = struct { lod: f32 = 0, grad_uv: [4]f32 = .{ 0, 0, 0, 0 } };
fn triangleTexLod(vouts: *const [3][GfxOut.vertex_len]f32, p0: [2]f32, p1: [2]f32, p2: [2]f32, us: u32, vs: u32, tw: u32, th: u32) TexLodResult {
    const dx1 = p1[0] - p0[0];
    const dy1 = p1[1] - p0[1];
    const dx2 = p2[0] - p0[0];
    const dy2 = p2[1] - p0[1];
    const det = dx1 * dy2 - dx2 * dy1;
    if (det == 0) return .{};
    const inv_det = 1.0 / det;
    const uvi = GfxOut.varying_base + us;
    const vvi = GfxOut.varying_base + vs;
    if (uvi >= GfxOut.vertex_len or vvi >= GfxOut.vertex_len) return .{};
    // d(varying)/d(window_x), d(varying)/d(window_y) via Cramer (matches the derivative path).
    const du1 = vouts[1][uvi] - vouts[0][uvi];
    const du2 = vouts[2][uvi] - vouts[0][uvi];
    const dudx = (du1 * dy2 - du2 * dy1) * inv_det;
    const dudy = (dx1 * du2 - dx2 * du1) * inv_det;
    const dv1 = vouts[1][vvi] - vouts[0][vvi];
    const dv2 = vouts[2][vvi] - vouts[0][vvi];
    const dvdx = (dv1 * dy2 - dv2 * dy1) * inv_det;
    const dvdy = (dx1 * dv2 - dx2 * dv1) * inv_det;
    const twf: f32 = @floatFromInt(tw);
    const thf: f32 = @floatFromInt(th);
    // rho^2 = (texels moved per screen pixel)^2 along each axis. The LOD is log2 of the larger's sqrt.
    const rx2 = (dudx * twf) * (dudx * twf) + (dvdx * thf) * (dvdx * thf);
    const ry2 = (dudy * twf) * (dudy * twf) + (dvdy * thf) * (dvdy * thf);
    const rho2 = @max(rx2, ry2);
    const lod: f32 = if (rho2 <= 0) 0 else 0.5 * @log2(rho2); // log2(sqrt(rho2)) = 0.5 * log2(rho2)
    return .{ .lod = lod, .grad_uv = .{ dudx, dvdx, dudy, dvdy } };
}

/// Rasterize a triangle through a real JITed fragment shader. `screen` holds each
/// vertex's screen-space x,y. `vouts` holds each vertex's VS output buffer
/// (position(4) + varyings). For every covered fragment we barycentric-interpolate
/// the varyings (from `GfxOut.varying_base`) and run the fragment shader, writing its
/// RGBA output to the framebuffer. The varyings the FS reads are the first
/// `prog.fs_inputs` scalars of the varying region (locations packed from 0).
/// `depth` is the optional depth attachment. When present, each covered fragment's
/// depth (barycentric-interpolated from the per-vertex NDC z) is depth-tested against
/// the depth buffer per the pipeline's compare op. Failing fragments are discarded
/// (color + depth unchanged), and a passing fragment writes the new depth if depth
/// writes are enabled. When null, no depth test happens (color-only path).
pub fn rasterShaded(
    pixels: []u8,
    width: u32,
    height: u32,
    screen: [3][2]f32,
    vouts: *const [3][GfxOut.vertex_len]f32,
    prog: *const pipeline.ShaderProgram,
    depth: ?DepthAttachment,
    tex: ?*const sampler.TexDesc,
    // The bound UBO / push-constant base pointers, indexed by binding (the same array the
    // VS reads). A FS `.descriptor` (plain buffer) param is fed the next of these in binding
    // order. A `.sampler_desc` param is fed the texture descriptor.
    ubos: []const ?[*]const u8,
) void {
    // Default: single-sample, RGBA8 (the historical fast path), no blending.
    rasterShadedFmt(pixels, width, height, .rgba8_unorm, 1, screen, vouts, prog, depth, null, tex, ubos, .{}, null, &.{}, .{});
}

/// `rasterShaded` with an explicit render-target format and MSAA sample count.
///
/// - `fmt`: the color target's pixel format. `pixels` holds width*height*samples
///   pixels of `fmt` (sample-minor when samples>1).
/// - `samples`: MSAA sample count (1/2/4). 1 = one center sample per pixel, the FS
///   color written directly. N>1 = coverage is evaluated at the N standard sample
///   positions per pixel. The FS is shaded once per pixel (per-pixel shading) and its
///   color written to every covered sample that passes the per-sample depth test. The
///   caller resolves (box-averages) the samples afterward via `resolveMsaa`.
/// The multisample coverage state for a draw: how the per-sample MSAA coverage mask is reduced
/// after the fragment shader runs. Both features are no-ops on a single-sample target.
pub const CoverageState = struct {
    /// GL_SAMPLE_ALPHA_TO_COVERAGE: reduce each fragment's covered-sample set by its alpha.
    alpha_to_coverage: bool = false,
    /// GL_SAMPLE_COVERAGE: AND the covered-sample set with a fixed coverage value (screen-door
    /// or LOD-fade transparency independent of the fragment alpha).
    sample_coverage: bool = false,
    /// The coverage fraction in [0,1]. Ceil(value*samples) samples survive (before invert).
    sample_coverage_value: f32 = 1.0,
    /// GL_SAMPLE_COVERAGE_INVERT: invert which samples the coverage value keeps.
    sample_coverage_invert: bool = false,
};

pub fn rasterShadedFmt(
    pixels: []u8,
    width: u32,
    height: u32,
    fmt: hal.Format,
    samples: u8,
    screen: [3][2]f32,
    vouts: *const [3][GfxOut.vertex_len]f32,
    prog: *const pipeline.ShaderProgram,
    depth: ?DepthAttachment,
    stencil: ?StencilAttachment,
    tex: ?*const sampler.TexDesc,
    ubos: []const ?[*]const u8,
    blend: hal.BlendState,
    scissor: ?hal.ScissorRect,
    /// Additional MRT color targets (indices 1+; target 0 is `pixels`/`fmt`). The FS's
    /// `layout(location=i) out` is written here, from out[i*4 .. i*4+4]. Empty = single-RT.
    extra_targets: []const ColorTarget,
    /// Multisample coverage controls (GL_SAMPLE_ALPHA_TO_COVERAGE / GL_SAMPLE_COVERAGE). MSAA only.
    coverage: CoverageState,
) void {
    if (width == 0 or height == 0) return;
    const bpp = pixelBytes(fmt);
    const ns: usize = if (samples > 1) samples else 1;
    const positions = samplePositions(samples);

    const p0 = screen[0];
    const p1 = screen[1];
    const p2 = screen[2];

    var area = edge(p0[0], p0[1], p1[0], p1[1], p2[0], p2[1]);
    if (area == 0) return;
    const sign: f32 = if (area < 0) -1.0 else 1.0;
    area *= sign;

    // Depth bias (glPolygonOffset): a per-triangle constant added to the interpolated depth.
    // m = max(|dz/dx|, |dz/dy|) is the triangle's screen-space depth slope. Offset = slope*m +
    // constant*R (R = the software depth buffer's minimum resolvable step, DEPTH_BIAS_R). Zero
    // when bias is disabled, so the depth path is byte-identical for an un-biased draw.
    var bias_offset: f32 = 0;
    if (depth) |d| {
        if (d.state.bias_enable) {
            const dz0 = vouts_screen_z(depth, 0);
            const dz1 = vouts_screen_z(depth, 1);
            const dz2 = vouts_screen_z(depth, 2);
            const dvz1 = dz1 - dz0;
            const dvz2 = dz2 - dz0;
            const num_x = dvz1 * (p2[1] - p0[1]) - dvz2 * (p1[1] - p0[1]);
            const num_y = (p1[0] - p0[0]) * dvz2 - (p2[0] - p0[0]) * dvz1;
            const m = @max(@abs(num_x), @abs(num_y)) / area;
            var off = d.state.bias_slope * m + d.state.bias_constant * DEPTH_BIAS_R;
            if (d.state.bias_clamp > 0) {
                off = @min(off, d.state.bias_clamp);
            } else if (d.state.bias_clamp < 0) {
                off = @max(off, d.state.bias_clamp);
            }
            bias_offset = off;
        }
    }

    var minx: u32 = @intFromFloat(@max(0.0, @floor(@min(p0[0], @min(p1[0], p2[0])))));
    var maxx: u32 = @intFromFloat(@min(@as(f32, @floatFromInt(width - 1)), @ceil(@max(p0[0], @max(p1[0], p2[0])))));
    var miny: u32 = @intFromFloat(@max(0.0, @floor(@min(p0[1], @min(p1[1], p2[1])))));
    var maxy: u32 = @intFromFloat(@min(@as(f32, @floatFromInt(height - 1)), @ceil(@max(p0[1], @max(p1[1], p2[1])))));

    // Scissor: intersect the triangle's bounding box with the scissor rect (framebuffer
    // pixels, top-left origin). Every downstream path (scalar, 2x2 quad, MSAA) iterates
    // [minx..maxx] x [miny..maxy], so clamping the box here clips them all uniformly.
    if (scissor) |sc| {
        const sx1: i64 = @as(i64, sc.x) + @as(i64, sc.width); // exclusive right
        const sy1: i64 = @as(i64, sc.y) + @as(i64, sc.height); // exclusive bottom
        if (sx1 <= 0 or sy1 <= 0) return; // rect entirely left of / above the surface
        const sc_minx: u32 = @intCast(@max(@as(i64, 0), @as(i64, sc.x)));
        const sc_miny: u32 = @intCast(@max(@as(i64, 0), @as(i64, sc.y)));
        const sc_maxx: u32 = @intCast(@min(@as(i64, width - 1), sx1 - 1));
        const sc_maxy: u32 = @intCast(@min(@as(i64, height - 1), sy1 - 1));
        if (sc_minx > maxx or sc_miny > maxy or sc_maxx < minx or sc_maxy < miny) return;
        minx = @max(minx, sc_minx);
        miny = @max(miny, sc_miny);
        maxx = @min(maxx, sc_maxx);
        maxy = @min(maxy, sc_maxy);
    }

    // Screen-space derivatives: a linearly-interpolated varying has a constant per-triangle
    // screen-space gradient (the plane equation), so each dFdx/dFdy the FS reads is computed
    // once here from the 3 vertices' varying values + screen positions, written into a small
    // gradient buffer, and the FS loads them through its grad_buf pointer. When the FS uses
    // no derivatives (grad_count==0) this is skipped and the FS gets no grad_buf pointer.
    var grad_vals: [spirv_jit.MAX_GRAD_INPUTS]f32 = .{0} ** spirv_jit.MAX_GRAD_INPUTS;
    if (prog.fs_grad_count > 0) {
        const dx1 = p1[0] - p0[0];
        const dy1 = p1[1] - p0[1];
        const dx2 = p2[0] - p0[0];
        const dy2 = p2[1] - p0[1];
        const det = dx1 * dy2 - dx2 * dy1;
        const inv_det: f32 = if (det != 0) 1.0 / det else 0;
        var gi: usize = 0;
        while (gi < prog.fs_grad_count and gi < grad_vals.len) : (gi += 1) {
            const g = prog.fs_grads[gi];
            const vi = GfxOut.varying_base + g.varying_index;
            if (vi >= GfxOut.vertex_len) continue;
            const v0 = vouts[0][vi];
            const v1 = vouts[1][vi];
            const v2 = vouts[2][vi];
            const dvy1 = v1 - v0;
            const dvy2 = v2 - v0;
            // Screen-space derivatives via Cramer's rule on the 2x2 plane system. `p0/p1/p2`
            // are this rasterizer's stored screen positions in Vulkan window space (origin
            // top-left, x right, y down (see context.zig drawShaded, which maps NDC to
            // window with no y-flip). So `ddx = d(v)/d(window_x)` and `ddy = d(v)/d(window_y)`
            // are Vulkan's dFdx/dFdy, and cross(dFdx(frag_pos), dFdy(frag_pos)) gives the
            // correctly-handed face normal (front faces light in the positive direction) for flat-shading
            // shaders like vkcube's. Matches a real GPU (verified against an analytic
            // reference of vkcube's exact MVP+geometry) and vkderiv (worldPos = window
            // coords -> dFdx=(1,0,0), dFdy=(0,1,0), cross=(0,0,1)).
            const ddx = (dvy1 * dy2 - dvy2 * dy1) * inv_det;
            const ddy = (dx1 * dvy2 - dx2 * dvy1) * inv_det;
            grad_vals[gi] = switch (g.axis) {
                .x => ddx,
                .y => ddy,
            };
        }
    }

    // Implicit mip LOD (automatic mipmap selection): when the FS samples a mipmapped texture through
    // bare varyings (u, v), compute the per-triangle screen-space LOD from those varyings' gradients
    // + the texture size, so a minified surface picks a coarser mip level (instead of always the
    // base). A per-triangle constant is exact for a linearly-interpolated varying (flat/UI) and an
    // approximation under strong perspective. Non-mipmapped textures are unaffected (the sampler
    // clamps to level 0 with no chain), so this is inert for the common single-level case.
    var tex_lod_copy: sampler.TexDesc = undefined;
    var tex_eff = tex;
    if (tex) |t| {
        if (t.levels > 1) {
            if (prog.fs_tex_coord_slots[0]) |us| {
                if (prog.fs_tex_coord_slots[1]) |vs| {
                    const r = triangleTexLod(vouts, p0, p1, p2, us, vs, t.width, t.height);
                    tex_lod_copy = t.*;
                    tex_lod_copy.implicit_lod = r.lod;
                    tex_lod_copy.grad_uv = r.grad_uv;
                    tex_eff = &tex_lod_copy;
                }
            }
        }
    }

    // The FS's bound pointer params, in entry-ABI (param-append) order, filled per
    // prog.fs_buffer_kinds: a combined-image-sampler descriptor (the single bound TexDesc),
    // the host sampler function pointer, or the gradient buffer pointer. The sampler_fn /
    // grad_buf params are appended lazily and may interleave, so the kind sequence (read off
    // the IR) determines what each x-register slot holds.
    var fs_bufs: [spirv_jit.GfxBuffers.max]?[*]const u8 = .{null} ** spirv_jit.GfxBuffers.max;
    const fs_total_bufs = @min(prog.fs_buffers, fs_bufs.len);
    // A defined zero descriptor for an unbound sampler: sampleTexture reads width==0 and
    // returns transparent black instead of dereferencing the JIT's dummy pointer (a crash).
    const empty_tex: sampler.TexDesc = .{ .pixels = undefined, .width = 0, .height = 0, .pitch = 0 };
    {
        // `.descriptor` (plain buffer) params are fed the bound UBO / push-constant base pointer
        // at the descriptor's binding (prog.fs_buffer_bindings[b]). `ubos` is binding-indexed,
        // so binding 0 (the VS block) and binding 1 (the FS block) never collide. A param with
        // no binding tag (-1, a hand-built shader) falls back to a running declaration-order
        // cursor. A `.sampler_desc` gets the texture descriptor.
        var ubo_cursor: usize = 0;
        var b: usize = 0;
        while (b < fs_total_bufs) : (b += 1) {
            fs_bufs[b] = switch (prog.fs_buffer_kinds[b]) {
                .descriptor => blk: {
                    const bind = prog.fs_buffer_bindings[b];
                    if (bind >= 0) {
                        const bi: usize = @intCast(bind);
                        break :blk if (bi < ubos.len) ubos[bi] else null;
                    }
                    const ptr: ?[*]const u8 = if (ubo_cursor < ubos.len) ubos[ubo_cursor] else null;
                    ubo_cursor += 1;
                    break :blk ptr;
                },
                .sampler_desc => if (tex_eff) |t| @ptrCast(t) else @ptrCast(&empty_tex),
                .sampler_fn => @ptrFromInt(@intFromPtr(&sampler.sampleTexture)),
                .sampler_cube_fn => @ptrFromInt(@intFromPtr(&sampler.sampleTextureCube)),
                .sampler_shadow_fn => @ptrFromInt(@intFromPtr(&sampler.sampleTextureShadow)),
                .sampler_cube_shadow_fn => @ptrFromInt(@intFromPtr(&sampler.sampleTextureCubeShadow)),
                .sampler_2darray_shadow_fn => @ptrFromInt(@intFromPtr(&sampler.sampleTexture2dArrayShadow)),
                .sampler_gather_fn => @ptrFromInt(@intFromPtr(&sampler.sampleTextureGather)),
                .sampler_fetch_fn => @ptrFromInt(@intFromPtr(&sampler.sampleTextureFetch)),
                .sampler_fetch3_fn => @ptrFromInt(@intFromPtr(&sampler.sampleTextureFetch3D)),
                .grad_buf => @ptrCast(&grad_vals),
                .math_fn => @ptrFromInt(@intFromPtr(&sampler.mathFn)),
                .discard_fn => @ptrFromInt(@intFromPtr(&spirv_jit.discardThunk)),
            };
        }
    }

    const nvar = @min(prog.fs_inputs, GfxOut.vertex_len - GfxOut.varying_base);

    // Whether any FS input is a fragment builtin (gl_FragCoord / gl_FrontFacing), so the
    // hot pixel loop only pays the per-fragment builtin fill when the shader uses one.
    const has_frag_builtin = blk: {
        var i: usize = 0;
        while (i < nvar and i < prog.fs_input_slots.len) : (i += 1) {
            if (prog.fs_input_slots[i] >= spirv_jit.FRAG_COORD_INPUT_BASE) break :blk true;
        }
        break :blk false;
    };

    // Single-sample fast path (the common, non-MSAA case).
    //
    // The three edge functions are linear in (x, y), so each w steps by a constant when x
    // (or y) advances by 1: w(x+1) = w(x) + dwdx, w(y+1) = w(y) + dwdy. Computing them
    // incrementally turns the per-pixel 6 mul + 6 sub edge evaluation into 3 adds (then 3
    // adds at end of row), which the profile flagged as the dominant scalar cost. Coverage
    // is then a sign test. The barycentrics are w/area. Depth + varyings interpolate from
    // them. The MSAA path (ns>1) keeps the general per-sample loop below.
    if (ns == 1) {
        // 4-wide quad SIMD fast path.
        //
        // When the FS was SIMD-widened (prog.fs_quad_entry present: a straight-line,
        // buffer-free FS), shade fragments in 2x2 quads (4 at once) via the NEON quad
        // entry. Each quad's 4 fragments' interpolated varyings pack into <4 x f32>
        // lanes. Off-triangle / depth-failing lanes are masked out (not written). The
        // quad FS output is component-major (qout[c*4 + lane]). Lane k's RGBA is read
        // back from qout[0*4+k..3*4+k]. The scalar loop below remains for non-widenable
        // FSes (and as the golden reference the equivalence tests compare against).
        // The quad path writes color directly with no blend path, so when blending is
        // enabled we fall through to the scalar loop (which reads-modify-writes the dst).
        // Blend is the uncommon path, so this costs nothing on the hot non-blend draws. It also
        // has no stencil test/update, so an active stencil attachment likewise falls through
        // to the scalar loop (where the stencil logic lives).
        const stencil_on = if (stencil) |s| s.state.test_enable else false;
        // A gl_FragCoord/gl_FrontFacing FS needs the per-fragment builtin fill, which the
        // quad path does not do, so fall through to the scalar loop for those shaders.
        // A depth-biased draw (bias_offset != 0) also falls through to the scalar loop, which
        // adds the per-triangle offset to the interpolated depth (the quad path does not). A
        // partial color write mask likewise falls through (the quad path writes color verbatim).
        const mask_full = blend.write_mask[0] and blend.write_mask[1] and blend.write_mask[2] and blend.write_mask[3];
        if (!blend.enable and !stencil_on and !has_frag_builtin and !prog.fs_writes_frag_depth and extra_targets.len == 0 and bias_offset == 0 and mask_full) {
            if (prog.fs_quad_entry) |qfe| {
                rasterQuad(pixels, width, height, fmt, screen, vouts, prog, depth, area, sign, minx, maxx, miny, maxy, nvar, qfe, fs_bufs[0..fs_total_bufs], fs_total_bufs);
                return;
            }
        }
        // edge(ax,ay,bx,by,px,py) = (px-ax)*(by-ay) - (py-ay)*(bx-ax); d/dpx = (by-ay),
        // d/dpy = -(bx-ax). Apply the winding `sign` once into the deltas + base value.
        // Per-x edge-value deltas (d(edge)/dpx, winding applied). The per-row base is
        // recomputed directly at each row's first pixel center (cheap, once per row), so
        // the y-deltas aren't needed.
        const e0_dx = (p2[1] - p1[1]) * sign; // edge p1->p2
        const e1_dx = (p0[1] - p2[1]) * sign; // edge p2->p0
        const e2_dx = (p1[1] - p0[1]) * sign; // edge p0->p1
        const inv_area: f32 = 1.0 / area;

        // Per-vertex varying vectors the FS reads (loaded once, not per fragment): for each
        // of the FS's nvar inputs, the 3 vertices' values, so the barycentric blend is a
        // single F32x4 madd chain per input instead of 3 scalar loads + 3 muls + 2 adds.
        var vy0: [spirv_jit.MAX_FS_INPUTS]f32 = undefined;
        var vy1: [spirv_jit.MAX_FS_INPUTS]f32 = undefined;
        var vy2: [spirv_jit.MAX_FS_INPUTS]f32 = undefined;
        {
            var k: usize = 0;
            while (k < nvar) : (k += 1) {
                const rel = if (k < prog.fs_input_slots.len) prog.fs_input_slots[k] else @as(u32, @intCast(k));
                const vi = GfxOut.varying_base + @min(@as(usize, rel), GfxOut.max_varying_locations * 4 - 1);
                vy0[k] = vouts[0][vi];
                vy1[k] = vouts[1][vi];
                vy2[k] = vouts[2][vi];
            }
        }
        const dz = if (depth != null) [3]f32{ vouts_screen_z(depth, 0), vouts_screen_z(depth, 1), vouts_screen_z(depth, 2) } else [3]f32{ 0, 0, 0 };

        var y: u32 = miny;
        while (y <= maxy) : (y += 1) {
            // Edge values at the first pixel center of this row (minx + 0.5, y + 0.5).
            const sx0 = @as(f32, @floatFromInt(minx)) + 0.5;
            const sy0 = @as(f32, @floatFromInt(y)) + 0.5;
            var w0 = edge(p1[0], p1[1], p2[0], p2[1], sx0, sy0) * sign;
            var w1 = edge(p2[0], p2[1], p0[0], p0[1], sx0, sy0) * sign;
            var w2 = edge(p0[0], p0[1], p1[0], p1[1], sx0, sy0) * sign;

            var x: u32 = minx;
            while (x <= maxx) : ({
                x += 1;
                w0 += e0_dx;
                w1 += e1_dx;
                w2 += e2_dx;
            }) {
                // Coverage: inside iff all three edge values are non-negative.
                if (w0 < 0 or w1 < 0 or w2 < 0) continue;

                const b0 = w0 * inv_area;
                const b1 = w1 * inv_area;
                const b2 = w2 * inv_area;

                const px_idx_s = @as(usize, y) * width + x;

                // Stencil test (single sample, before depth per GL order). On a stencil
                // fail the fragment is discarded after applying fail_op. When no stencil
                // attachment is bound this whole block compiles to nothing taken.
                if (stencil) |s| {
                    if (s.state.test_enable and px_idx_s < s.buffer.len) {
                        const stored = s.buffer[px_idx_s];
                        const sref = s.state.reference & s.state.compare_mask;
                        const sval = stored & s.state.compare_mask;
                        if (!compareStencil(s.state.compare_op, sref, sval)) {
                            s.buffer[px_idx_s] = stencilWrite(stored, applyStencilOp(s.state.fail_op, stored, s.state.reference), s.state.write_mask);
                            continue;
                        }
                    }
                }

                // Depth test (single sample). With a stencil attachment bound, a depth
                // failure must still run the stencil depth_fail_op before discarding, so
                // record the result instead of continuing immediately.
                var pass_z: f32 = 0;
                var depth_failed = false;
                if (depth) |d| {
                    if (d.state.test_enable) {
                        pass_z = std.math.clamp(b0 * dz[0] + b1 * dz[1] + b2 * dz[2] + bias_offset, 0.0, 1.0);
                        const di = @as(usize, y) * width + x;
                        // A gl_FragDepth shader REPLACES the depth in the FS, so its depth test is
                        // late (against the shader-written depth, after the FS; see the write block
                        // below). Only an interpolated-z shader rejects early here. This matches the
                        // GPU's late-Z for depth-replace; without it, occlusion uses the wrong z.
                        if (!prog.fs_writes_frag_depth and di < d.buffer.len and !depthPasses(d.state.compare_op, pass_z, d.buffer[di])) depth_failed = true;
                    }
                }
                // Stencil update: the stencil test already passed above, so the outcome
                // selects depth_fail_op (depth failed) or pass_op (both passed).
                if (stencil) |s| {
                    if (s.state.test_enable and px_idx_s < s.buffer.len) {
                        const stored = s.buffer[px_idx_s];
                        const op = if (depth_failed) s.state.depth_fail_op else s.state.pass_op;
                        s.buffer[px_idx_s] = stencilWrite(stored, applyStencilOp(op, stored, s.state.reference), s.state.write_mask);
                    }
                }
                if (depth_failed) continue;

                // Interpolate the FS varyings (vectorized barycentric blend per input).
                // The FS reads exactly `prog.fs_inputs` scalars, and nvar == fs_inputs here
                // (fs_inputs <= 8 by the runGraphics ABI cap, well under the 32 varying cap),
                // so every read slot [0..nvar) is filled here. No zero-init is needed (the
                // per-fragment 144-byte memset the profile flagged as `memcpy` is removed).
                const bb0: F32x4 = @splat(b0);
                const bb1: F32x4 = @splat(b1);
                const bb2: F32x4 = @splat(b2);
                var fs_in: [spirv_jit.GfxOut.vertex_len]f32 = undefined;
                var k: usize = 0;
                while (k + 4 <= nvar) : (k += 4) {
                    const c0: F32x4 = vy0[k..][0..4].*;
                    const c1: F32x4 = vy1[k..][0..4].*;
                    const c2: F32x4 = vy2[k..][0..4].*;
                    fs_in[k..][0..4].* = bb0 * c0 + bb1 * c1 + bb2 * c2;
                }
                while (k < nvar) : (k += 1) {
                    fs_in[k] = b0 * vy0[k] + b1 * vy1[k] + b2 * vy2[k];
                }
                // Defensive: if the FS somehow reads beyond nvar (fs_inputs > nvar, only
                // possible if a future cap changes), zero the tail so it never reads garbage.
                while (k < prog.fs_inputs and k < fs_in.len) : (k += 1) fs_in[k] = 0;
                if (has_frag_builtin) fillFragBuiltins(&fs_in, prog.fs_input_slots[0..], nvar, @floatFromInt(x), @floatFromInt(y), pass_z, if (sign > 0) 1.0 else 0.0);

                var color = [_]f32{0} ** spirv_jit.GfxOut.fragment_len; // MRT colors + gl_FragDepth
                color[3] = 1; // target-0 default alpha
                spirv_jit.discardReset();
                if (prog.fs_entry) |fe| {
                    spirv_jit.runGraphicsAt(fe, prog.fs_inputs, fs_total_bufs, &fs_in, fs_bufs[0..fs_total_bufs], &color) catch return;
                } else {
                    spirv_jit.runGraphics(&prog.fs, prog.fs_inputs, fs_total_bufs, &fs_in, fs_bufs[0..fs_total_bufs], &color) catch return;
                }
                // Late-Z for gl_FragDepth: the shader-written depth is tested against the stored
                // depth after the FS (the early test above was skipped for a depth-replace shader).
                // On failure the fragment writes neither color nor depth. Normal shaders already
                // passed the early interpolated-z test, so `late_fail` stays false for them.
                var late_fail = false;
                const frag_z = if (prog.fs_writes_frag_depth) std.math.clamp(color[spirv_jit.GfxOut.frag_depth_index], 0, 1) else pass_z;
                if (prog.fs_writes_frag_depth) {
                    if (depth) |d| {
                        if (d.state.test_enable) {
                            const di = @as(usize, y) * width + x;
                            if (di < d.buffer.len and !depthPasses(d.state.compare_op, frag_z, d.buffer[di])) late_fail = true;
                        }
                    }
                }
                // A discarded fragment (OpKill) writes neither color nor depth.
                if (!spirv_jit.discarded() and !late_fail) {
                    const cvec: F32x4 = color[0..4].*;
                    const px_index = @as(usize, y) * width + x;
                    packMasked(pixels, px_index * bpp, fmt, blendSrc(pixels, px_index * bpp, fmt, blend, cvec), blend.write_mask);
                    if (extra_targets.len > 0) writeExtraTargets(extra_targets, px_index, &color);
                    if (depth) |d| {
                        if (d.state.test_enable and d.state.write_enable) {
                            const di = @as(usize, y) * width + x;
                            if (di < d.buffer.len) d.buffer[di] = frag_z; // gl_FragDepth replaces the interpolated depth
                        }
                    }
                }
            }
        }
        return;
    }

    var y: u32 = miny;
    while (y <= maxy) : (y += 1) {
        var x: u32 = minx;
        while (x <= maxx) : (x += 1) {
            const fx: f32 = @floatFromInt(x);
            const fy: f32 = @floatFromInt(y);

            // Determine which of this pixel's N sample positions the triangle covers
            // (1 center sample for non-MSAA). For each covered sample we also compute
            // its per-sample depth (interpolated at the sample location).
            var covered: [4]bool = .{ false, false, false, false };
            var samp_depth: [4]f32 = .{ 0, 0, 0, 0 };
            var any_covered = false;
            // Barycentrics at the PIXEL CENTER drive the FS (per-pixel shading), so the
            // FS is run once per pixel and its color written to every covered sample.
            var cb0: f32 = 0;
            var cb1: f32 = 0;
            var cb2: f32 = 0;
            var have_center_bary = false;
            var s: usize = 0;
            while (s < ns) : (s += 1) {
                const sx = fx + positions[s][0];
                const sy = fy + positions[s][1];
                const w0 = edge(p1[0], p1[1], p2[0], p2[1], sx, sy) * sign;
                const w1 = edge(p2[0], p2[1], p0[0], p0[1], sx, sy) * sign;
                const w2 = edge(p0[0], p0[1], p1[0], p1[1], sx, sy) * sign;
                if (!(w0 >= 0 and w1 >= 0 and w2 >= 0)) continue;
                covered[s] = true;
                any_covered = true;
                const b0 = w0 / area;
                const b1 = w1 / area;
                const b2 = w2 / area;
                samp_depth[s] = std.math.clamp(b0 * vouts_screen_z(depth, 0) + b1 * vouts_screen_z(depth, 1) + b2 * vouts_screen_z(depth, 2) + bias_offset, 0.0, 1.0);
                if (!have_center_bary) {
                    cb0 = b0;
                    cb1 = b1;
                    cb2 = b2;
                    have_center_bary = true;
                }
            }
            if (!any_covered) continue;

            // Use the pixel-center barycentrics for FS varyings when available. Otherwise
            // fall back to the first covered sample's (already loaded above). For the
            // single-sample path this is the center.
            if (ns == 1 or !have_center_bary) {
                // (already set from the single covered sample)
            } else {
                const ccx = fx + 0.5;
                const ccy = fy + 0.5;
                const w0 = edge(p1[0], p1[1], p2[0], p2[1], ccx, ccy) * sign;
                const w1 = edge(p2[0], p2[1], p0[0], p0[1], ccx, ccy) * sign;
                const w2 = edge(p0[0], p0[1], p1[0], p1[1], ccx, ccy) * sign;
                // If the center is inside the triangle use it. Else keep the first covered
                // sample's barycentrics (so an edge pixel still shades from a valid point).
                if (w0 >= 0 and w1 >= 0 and w2 >= 0) {
                    cb0 = w0 / area;
                    cb1 = w1 / area;
                    cb2 = w2 / area;
                }
            }

            // Per-sample depth test: a sample is written only if it is covered and passes
            // depth. Discard coverage on samples that fail. For non-MSAA this is the same
            // single-sample test as before.
            if (depth) |d| {
                if (d.state.test_enable) {
                    const dsamp: usize = if (d.samples > 1) d.samples else 1;
                    var sd: usize = 0;
                    while (sd < ns) : (sd += 1) {
                        if (!covered[sd]) continue;
                        const di = (@as(usize, y) * width + x) * dsamp + (if (dsamp > 1) sd else 0);
                        if (di < d.buffer.len) {
                            if (!depthPasses(d.state.compare_op, samp_depth[sd], d.buffer[di])) covered[sd] = false;
                        }
                    }
                }
            }
            // Recheck after depth test: if nothing survives, skip the FS.
            any_covered = false;
            for (covered[0..ns]) |c| {
                if (c) any_covered = true;
            }
            if (!any_covered) continue;

            // Interpolate the varyings the FS consumes (zero-padded to arity), using the
            // pixel-center barycentrics. The FS's screen-space derivatives come through its
            // grad_buf pointer (assembled in fs_bufs above), not as float inputs.
            var fs_in = [_]f32{0} ** spirv_jit.GfxOut.vertex_len;
            var k: usize = 0;
            while (k < nvar) : (k += 1) {
                const rel = if (k < prog.fs_input_slots.len) prog.fs_input_slots[k] else @as(u32, @intCast(k));
                const vi = GfxOut.varying_base + @min(@as(usize, rel), GfxOut.max_varying_locations * 4 - 1);
                fs_in[k] = cb0 * vouts[0][vi] + cb1 * vouts[1][vi] + cb2 * vouts[2][vi];
            }
            if (has_frag_builtin) {
                var fbz: f32 = 0;
                for (0..ns) |si| {
                    if (covered[si]) {
                        fbz = samp_depth[si];
                        break;
                    }
                }
                fillFragBuiltins(&fs_in, prog.fs_input_slots[0..], nvar, fx, fy, fbz, if (sign > 0) 1.0 else 0.0);
            }
            var color = [_]f32{0} ** spirv_jit.GfxOut.fragment_len; // MRT colors + gl_FragDepth
            color[3] = 1; // target-0 default alpha
            spirv_jit.discardReset();
            if (prog.fs_entry) |fe| {
                spirv_jit.runGraphicsAt(fe, prog.fs_inputs, fs_total_bufs, &fs_in, fs_bufs[0..fs_total_bufs], &color) catch return;
            } else {
                spirv_jit.runGraphics(&prog.fs, prog.fs_inputs, fs_total_bufs, &fs_in, fs_bufs[0..fs_total_bufs], &color) catch return;
            }
            if (spirv_jit.discarded()) continue; // OpKill: no color/depth for any sample
            const cvec: F32x4 = color[0..4].*;

            // GL_SAMPLE_ALPHA_TO_COVERAGE: turn the fragment alpha into a per-sample coverage mask
            // (MSAA only). A fragment with alpha `a` keeps sample s iff a > (s+0.5)/ns, so ~a of the
            // covered samples survive and the rest keep the background. The resolve then gives a
            // smooth alpha edge without blending (foliage / cut-out edges). A fully-opaque fragment
            // (a>=1) keeps every sample, so an ordinary draw is unaffected.
            if (coverage.alpha_to_coverage and ns > 1) {
                const a = cvec[3];
                var sa: usize = 0;
                while (sa < ns) : (sa += 1) {
                    if (covered[sa] and a <= (@as(f32, @floatFromInt(sa)) + 0.5) / @as(f32, @floatFromInt(ns))) covered[sa] = false;
                }
                any_covered = false;
                for (covered[0..ns]) |c| {
                    if (c) any_covered = true;
                }
                if (!any_covered) continue;
            }

            // GL_SAMPLE_COVERAGE: AND the covered set with a fixed coverage value (independent of the
            // fragment alpha; screen-door / LOD-fade transparency). k = ceil(value*ns) samples survive.
            // Non-invert keeps samples [0,k), invert keeps [k,ns). value 1.0 non-invert is a no-op.
            if (coverage.sample_coverage and ns > 1) {
                const v = std.math.clamp(coverage.sample_coverage_value, 0.0, 1.0);
                const scov_k: usize = @intFromFloat(@ceil(v * @as(f32, @floatFromInt(ns))));
                var sc: usize = 0;
                while (sc < ns) : (sc += 1) {
                    const in_mask = if (coverage.sample_coverage_invert) (sc >= scov_k) else (sc < scov_k);
                    if (covered[sc] and !in_mask) covered[sc] = false;
                }
                any_covered = false;
                for (covered[0..ns]) |c| {
                    if (c) any_covered = true;
                }
                if (!any_covered) continue;
            }

            // Write the shaded color into every covered+passing sample, and write that
            // sample's depth if depth writes are enabled.
            var ws: usize = 0;
            while (ws < ns) : (ws += 1) {
                if (!covered[ws]) continue;
                const px_index = (@as(usize, y) * width + x) * ns + (if (ns > 1) ws else 0);
                packMasked(pixels, px_index * bpp, fmt, blendSrc(pixels, px_index * bpp, fmt, blend, cvec), blend.write_mask);
                if (extra_targets.len > 0) writeExtraTargets(extra_targets, px_index, &color);
                if (depth) |d| {
                    if (d.state.test_enable and d.state.write_enable) {
                        const dsamp: usize = if (d.samples > 1) d.samples else 1;
                        const di = (@as(usize, y) * width + x) * dsamp + (if (dsamp > 1) ws else 0);
                        // gl_FragDepth (if written) replaces the per-sample interpolated depth.
                        const zw = if (prog.fs_writes_frag_depth) std.math.clamp(color[spirv_jit.GfxOut.frag_depth_index], 0, 1) else samp_depth[ws];
                        if (di < d.buffer.len) d.buffer[di] = zw;
                    }
                }
            }
        }
    }
}

/// Per-vertex NDC depth helper: 0 when there is no depth attachment (the value is
/// unused in that case). Keeps the per-sample depth interpolation branch-free.
inline fn vouts_screen_z(depth: ?DepthAttachment, i: usize) f32 {
    return if (depth) |d| d.screen_z[i] else 0;
}

/// 4-wide quad rasterization: shade fragments in 2x2 quads, running the SIMD-widened
/// fragment shader once per quad (4 fragments at once). Lane k of the quad = the k-th
/// fragment of the 2x2 block (order: top-left, top-right, bottom-left, bottom-right).
/// A lane that is off-triangle, out of bounds, or fails the depth test is masked off
/// (its color is never written). Single-sample, color + optional depth. The caller
/// guarantees `prog.fs_quad_entry != null`. Bit-identical to the scalar per-fragment
/// path for every covered lane (the equivalence tests prove the FS machine code;
/// the per-lane interpolation here uses the same barycentric blend the scalar path uses).
fn rasterQuad(
    pixels: []u8,
    width: u32,
    height: u32,
    fmt: hal.Format,
    screen: [3][2]f32,
    vouts: *const [3][GfxOut.vertex_len]f32,
    prog: *const pipeline.ShaderProgram,
    depth: ?DepthAttachment,
    area: f32,
    sign: f32,
    minx: u32,
    maxx: u32,
    miny: u32,
    maxy: u32,
    nvar: usize,
    qfe: *const anyopaque,
    fs_bufs: []const ?[*]const u8,
    fs_total_bufs: usize,
) void {
    const bpp = pixelBytes(fmt);
    const p0 = screen[0];
    const p1 = screen[1];
    const p2 = screen[2];
    const inv_area: f32 = 1.0 / area;

    // The FS's varying-input layout: input j reads the interpolated varying at slot
    // input_slots[j]. Precompute the 3 vertices' value for each input (loaded once).
    var vy0: [spirv_jit.MAX_FS_INPUTS]f32 = undefined;
    var vy1: [spirv_jit.MAX_FS_INPUTS]f32 = undefined;
    var vy2: [spirv_jit.MAX_FS_INPUTS]f32 = undefined;
    {
        var k: usize = 0;
        while (k < nvar) : (k += 1) {
            const rel = if (k < prog.fs_input_slots.len) prog.fs_input_slots[k] else @as(u32, @intCast(k));
            const vi = GfxOut.varying_base + @min(@as(usize, rel), GfxOut.max_varying_locations * 4 - 1);
            vy0[k] = vouts[0][vi];
            vy1[k] = vouts[1][vi];
            vy2[k] = vouts[2][vi];
        }
    }
    const dz = if (depth != null) [3]f32{ vouts_screen_z(depth, 0), vouts_screen_z(depth, 1), vouts_screen_z(depth, 2) } else [3]f32{ 0, 0, 0 };

    // Edge functions are affine: e(x+dx, y+dy) = e(x,y) + dx*e_dx + dy*e_dy. Compute the
    // per-pixel deltas once (winding applied) so each quad is 3 row-base evals + a handful
    // of adds for the 4 lanes (instead of 3 full edge evals per lane).
    const e0_dx = (p2[1] - p1[1]) * sign; // edge p1->p2, d/dx
    const e1_dx = (p0[1] - p2[1]) * sign; // edge p2->p0, d/dx
    const e2_dx = (p1[1] - p0[1]) * sign; // edge p0->p1, d/dx
    const e0_dy = -(p2[0] - p1[0]) * sign; // d/dy = -(bx-ax)
    const e1_dy = -(p0[0] - p2[0]) * sign;
    const e2_dy = -(p1[0] - p0[0]) * sign;

    const has_depth = depth != null and depth.?.state.test_enable;
    const write_depth = has_depth and depth.?.state.write_enable;

    var qy: u32 = miny;
    while (qy <= maxy) : (qy += 2) {
        // Edge values at the top-left lane's pixel center (qx + 0.5, qy + 0.5) of this row.
        const sy0 = @as(f32, @floatFromInt(qy)) + 0.5;
        const row_x0 = @as(f32, @floatFromInt(minx)) + 0.5;
        var rw0 = edge(p1[0], p1[1], p2[0], p2[1], row_x0, sy0) * sign;
        var rw1 = edge(p2[0], p2[1], p0[0], p0[1], row_x0, sy0) * sign;
        var rw2 = edge(p0[0], p0[1], p1[0], p1[1], row_x0, sy0) * sign;

        var qx: u32 = minx;
        while (qx <= maxx) : ({
            qx += 2;
            rw0 += 2 * e0_dx;
            rw1 += 2 * e1_dx;
            rw2 += 2 * e2_dx;
        }) {
            // The 4 lanes' edge values from the quad's top-left base via the deltas.
            // lane order: TL(0,0), TR(1,0), BL(0,1), BR(1,1).
            const lw0: [4]f32 = .{ rw0, rw0 + e0_dx, rw0 + e0_dy, rw0 + e0_dx + e0_dy };
            const lw1: [4]f32 = .{ rw1, rw1 + e1_dx, rw1 + e1_dy, rw1 + e1_dx + e1_dy };
            const lw2: [4]f32 = .{ rw2, rw2 + e2_dx, rw2 + e2_dy, rw2 + e2_dx + e2_dy };

            var lane_cov: [4]bool = .{ false, false, false, false };
            var lane_b: [4][3]f32 = .{.{ 0, 0, 0 }} ** 4;
            var lane_z: [4]f32 = .{ 0, 0, 0, 0 };
            var any = false;
            inline for (0..4) |l| {
                const lx_i = @as(i64, qx) + (l & 1);
                const ly_i = @as(i64, qy) + (l >> 1);
                if (lx_i <= @as(i64, maxx) and ly_i <= @as(i64, maxy) and lx_i < @as(i64, width) and ly_i < @as(i64, height) and
                    lw0[l] >= 0 and lw1[l] >= 0 and lw2[l] >= 0)
                {
                    const b0 = lw0[l] * inv_area;
                    const b1 = lw1[l] * inv_area;
                    const b2 = lw2[l] * inv_area;
                    var pass = true;
                    if (has_depth) {
                        const z = std.math.clamp(b0 * dz[0] + b1 * dz[1] + b2 * dz[2], 0.0, 1.0);
                        const di = @as(usize, @intCast(ly_i)) * width + @as(usize, @intCast(lx_i));
                        if (di < depth.?.buffer.len and !depthPasses(depth.?.state.compare_op, z, depth.?.buffer[di])) {
                            pass = false;
                        } else lane_z[l] = z;
                    }
                    if (pass) {
                        lane_cov[l] = true;
                        lane_b[l] = .{ b0, b1, b2 };
                        any = true;
                    }
                }
            }
            if (!any) continue;

            // Pack the quad's per-input varyings: input j -> <4 x f32> of the 4 lanes'
            // barycentric-interpolated value.
            var quad_in: [8]spirv_jit.Quad = undefined;
            var k: usize = 0;
            while (k < prog.fs_inputs and k < quad_in.len) : (k += 1) {
                const c0: spirv_jit.Quad = @splat(vy0[k]);
                const c1: spirv_jit.Quad = @splat(vy1[k]);
                const c2: spirv_jit.Quad = @splat(vy2[k]);
                const bb0: spirv_jit.Quad = .{ lane_b[0][0], lane_b[1][0], lane_b[2][0], lane_b[3][0] };
                const bb1: spirv_jit.Quad = .{ lane_b[0][1], lane_b[1][1], lane_b[2][1], lane_b[3][1] };
                const bb2: spirv_jit.Quad = .{ lane_b[0][2], lane_b[1][2], lane_b[2][2], lane_b[3][2] };
                quad_in[k] = bb0 * c0 + bb1 * c1 + bb2 * c2;
            }

            var qout: [spirv_jit.quad_out_len]f32 align(16) = .{0} ** spirv_jit.quad_out_len;
            // The FS's pointer params (sampler desc / sampler_fn / grad_buf / math_fn) are
            // lane-invariant per triangle (the grad_buf holds the per-triangle screen-space
            // gradients the FS broadcasts. The sampler/math fns and the texture descriptor are
            // shared), so the same fs_bufs the scalar path uses feed the quad entry. Per-lane
            // texture sampling happens inside the widened FS (the gather scalarizes the sampler
            // call across the 4 lanes' uv), matching the scalar per-fragment sample.
            spirv_jit.runGraphicsQuadAt(qfe, prog.fs_inputs, fs_total_bufs, quad_in[0..prog.fs_inputs], fs_bufs, &qout) catch return;

            // Write each covered lane's RGBA (component-major: qout[c*4 + lane]).
            inline for (0..4) |l| {
                if (lane_cov[l]) {
                    samples_written +%= 1; // occlusion query: a covered fragment (SIMD quad path)
                    const cvec: F32x4 = .{ qout[0 * 4 + l], qout[1 * 4 + l], qout[2 * 4 + l], qout[3 * 4 + l] };
                    const lx = qx + (l & 1);
                    const ly = qy + (l >> 1);
                    const px_index = @as(usize, ly) * width + lx;
                    packPixel(pixels, px_index * bpp, fmt, cvec);
                    if (write_depth) {
                        const di = @as(usize, ly) * width + lx;
                        if (di < depth.?.buffer.len) depth.?.buffer[di] = lane_z[l];
                    }
                }
            }
        }
    }
}

test "depthPasses honors the compare op" {
    try std.testing.expect(depthPasses(.less, 0.3, 0.7));
    try std.testing.expect(!depthPasses(.less, 0.7, 0.3));
    try std.testing.expect(depthPasses(.less_or_equal, 0.5, 0.5));
    try std.testing.expect(depthPasses(.greater, 0.7, 0.3));
    try std.testing.expect(depthPasses(.always, 0.9, 0.1));
    try std.testing.expect(!depthPasses(.never, 0.1, 0.9));
    try std.testing.expect(depthPasses(.equal, 0.4, 0.4));
}

test "clear fills every pixel" {
    var px = [_]u8{0} ** (2 * 2 * 4);
    clear(&px, 2, 2, .{ .r = 1, .g = 0, .b = 0, .a = 1 });
    try std.testing.expectEqual(@as(u8, 255), px[0]);
    try std.testing.expectEqual(@as(u8, 0), px[1]);
    try std.testing.expectEqual(@as(u8, 255), px[3]);
    try std.testing.expectEqual(@as(u8, 255), px[(3 * 4) + 0]);
}

test "clear and drawTriangle with zero dimensions do not crash" {
    var px = [_]u8{} ** 0;
    clear(&px, 0, 0, .{ .r = 1, .g = 0, .b = 0, .a = 1 });
    clear(&px, 0, 4, .{ .r = 1, .g = 0, .b = 0, .a = 1 });
    clear(&px, 4, 0, .{ .r = 1, .g = 0, .b = 0, .a = 1 });
    const verts = [3]Vertex{
        .{ .x = -1, .y = -1, .r = 1, .g = 1, .b = 1, .a = 1 },
        .{ .x = 1, .y = -1, .r = 1, .g = 1, .b = 1, .a = 1 },
        .{ .x = 0, .y = 1, .r = 1, .g = 1, .b = 1, .a = 1 },
    };
    drawTriangle(&px, 0, 0, verts);
    drawTriangle(&px, 0, 4, verts);
    drawTriangle(&px, 4, 0, verts);
}

test "triangle covers the center pixel and interpolates" {
    const W = 8;
    const H = 8;
    var px = [_]u8{0} ** (W * H * 4);
    // Big triangle covering the center, white at all corners.
    const verts = [3]Vertex{
        .{ .x = -1, .y = -1, .r = 1, .g = 1, .b = 1, .a = 1 },
        .{ .x = 1, .y = -1, .r = 1, .g = 1, .b = 1, .a = 1 },
        .{ .x = 0, .y = 1, .r = 1, .g = 1, .b = 1, .a = 1 },
    };
    drawTriangle(&px, W, H, verts);
    const cx = 4;
    const cy = 4;
    const idx = (cy * W + cx) * 4;
    try std.testing.expectEqual(@as(u8, 255), px[idx + 0]); // center is inside -> white
    // A top corner pixel is outside the triangle -> untouched (0)
    try std.testing.expectEqual(@as(u8, 0), px[(0 * W + 0) * 4 + 0]);
}

test "float render-target pack/unpack roundtrips within precision" {
    // R32F: exact roundtrip of an out-of-[0,1] value (a float target is not clamped).
    {
        var px = [_]u8{0} ** 4;
        packPixel(&px, 0, .r32_float, F32x4{ 7.5, 0, 0, 0 });
        const v = unpackPixel(&px, 0, .r32_float);
        try std.testing.expectEqual(@as(f32, 7.5), v[0]);
    }
    // R16F: roundtrip within f16 precision (and survives >1).
    {
        var px = [_]u8{0} ** 2;
        packPixel(&px, 0, .r16_float, F32x4{ 3.25, 0, 0, 0 });
        const v = unpackPixel(&px, 0, .r16_float);
        try std.testing.expectApproxEqAbs(@as(f32, 3.25), v[0], 0.01);
    }
    // RGBA16F: all four channels, an HDR value > 1 survives.
    {
        var px = [_]u8{0} ** 8;
        packPixel(&px, 0, .rgba16_float, F32x4{ 0.5, 2.0, 0.25, 1.0 });
        const v = unpackPixel(&px, 0, .rgba16_float);
        try std.testing.expectApproxEqAbs(@as(f32, 0.5), v[0], 0.01);
        try std.testing.expectApproxEqAbs(@as(f32, 2.0), v[1], 0.01);
        try std.testing.expectApproxEqAbs(@as(f32, 0.25), v[2], 0.01);
        try std.testing.expectApproxEqAbs(@as(f32, 1.0), v[3], 0.01);
    }
    // R8 / RG8 unorm: clamp to [0,1] then 8-bit quantize.
    {
        var px = [_]u8{0} ** 2;
        packPixel(&px, 0, .r8g8_unorm, F32x4{ 1.0, 0.0, 0, 0 });
        try std.testing.expectEqual(@as(u8, 255), px[0]);
        try std.testing.expectEqual(@as(u8, 0), px[1]);
        const v = unpackPixel(&px, 0, .r8g8_unorm);
        try std.testing.expectApproxEqAbs(@as(f32, 1.0), v[0], 1e-5);
        try std.testing.expectApproxEqAbs(@as(f32, 0.0), v[1], 1e-5);
    }
}

test "MSAA resolve box-averages the N samples per pixel" {
    // 1x1 image, 4 samples: two samples white, two black -> resolve = 0.5 gray.
    const fmt: hal.Format = .rgba8_unorm;
    const bpp = pixelBytes(fmt);
    var src = [_]u8{0} ** (4 * 4); // 4 samples * 4 bytes
    packPixel(&src, 0 * bpp, fmt, F32x4{ 1, 1, 1, 1 });
    packPixel(&src, 1 * bpp, fmt, F32x4{ 1, 1, 1, 1 });
    packPixel(&src, 2 * bpp, fmt, F32x4{ 0, 0, 0, 1 });
    packPixel(&src, 3 * bpp, fmt, F32x4{ 0, 0, 0, 1 });
    var dst = [_]u8{0} ** 4;
    resolveMsaa(&dst, &src, 1, 1, fmt, 4);
    // 2/4 white -> ~128. This INTERMEDIATE value is the MSAA edge-blend signature
    // (distinct from a pure 0 or 255).
    try std.testing.expect(dst[0] > 100 and dst[0] < 160);
    try std.testing.expectEqual(@as(u8, 255), dst[3]); // alpha resolves to 1
}

test "clearFmt fills an MSAA color buffer (all samples) for a float format" {
    // 1x1 R16F, 2 samples laid out as a 1x(1*2) image: clearFmt walks linearly.
    var px = [_]u8{0} ** (2 * 2); // 2 samples * 2 bytes
    clearFmt(&px, 1, 2, .r16_float, .{ .r = 0.5, .g = 0, .b = 0, .a = 1 });
    const v0 = unpackPixel(&px, 0, .r16_float);
    const v1 = unpackPixel(&px, 2, .r16_float);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), v0[0], 0.01);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), v1[0], 0.01);
}

test "quad SIMD rasterization matches the scalar per-fragment path over a whole triangle" {
    // The definitive raster-integration correctness gate: render the same triangle with
    // the same fragment shader through both the scalar per-fragment path (the golden
    // reference) and the 4-wide quad SIMD path, then compare every pixel. The quad path
    // must match the scalar path within a tiny tolerance. The FS machine code is proven
    // bit-exact by the spirv_jit equivalence tests. The only possible difference is
    // float-reorder in the rasterizer's incremental barycentric interpolation, which must
    // be sub-1-LSB (<= 1/255) for a correct vectorization.
    const front = @import("../../spirv.zig");
    const vspirv = @import("vulcan-spirv");
    const op = vspirv.opcodes;
    const sc = op.StorageClass;
    const gpa = std.testing.allocator;

    // Channel-rotate FS (a widenable straight-line FS): in vec3 vc -> o=(vc.b,vc.r,vc.g,1).
    var b = try vspirv.binary.Builder.init(gpa, 18);
    defer b.deinit(gpa);
    try b.emit(gpa, op.EntryPoint, &.{ op.ExecutionModel.fragment, 10, 0, 8, 9 });
    try b.emit(gpa, op.Decorate, &.{ 8, op.Decoration.location, 0 });
    try b.emit(gpa, op.Decorate, &.{ 9, op.Decoration.location, 0 });
    try b.emit(gpa, op.TypeVoid, &.{1});
    try b.emit(gpa, op.TypeFunction, &.{ 2, 1 });
    try b.emit(gpa, op.TypeFloat, &.{ 3, 32 });
    try b.emit(gpa, op.TypeVector, &.{ 4, 3, 3 });
    try b.emit(gpa, op.TypeVector, &.{ 5, 3, 4 });
    try b.emit(gpa, op.TypePointer, &.{ 6, sc.input, 4 });
    try b.emit(gpa, op.TypePointer, &.{ 7, sc.output, 5 });
    try b.emit(gpa, op.Constant, &.{ 3, 12, @bitCast(@as(f32, 1.0)) });
    try b.emit(gpa, op.Variable, &.{ 6, 8, sc.input });
    try b.emit(gpa, op.Variable, &.{ 7, 9, sc.output });
    try b.emit(gpa, op.Function, &.{ 1, 10, 0, 2 });
    try b.emit(gpa, op.Label, &.{11});
    try b.emit(gpa, op.Load, &.{ 4, 13, 8 });
    try b.emit(gpa, op.CompositeExtract, &.{ 3, 14, 13, 2 });
    try b.emit(gpa, op.CompositeExtract, &.{ 3, 15, 13, 0 });
    try b.emit(gpa, op.CompositeExtract, &.{ 3, 16, 13, 1 });
    try b.emit(gpa, op.CompositeConstruct, &.{ 5, 17, 14, 15, 16, 12 });
    try b.emit(gpa, op.Store, &.{ 9, 17 });
    try b.emit(gpa, op.Return, &.{});
    try b.emit(gpa, op.FunctionEnd, &.{});
    const fs_code = std.mem.sliceAsBytes(b.words.items);

    // Scalar FS entry.
    var sfunc = try front.parseSpirv(gpa, fs_code);
    defer sfunc.deinit();
    const sinfo = try spirv_jit.rewriteGraphics(&sfunc);
    var scompiled = try spirv_jit.jitFunction(gpa, &sfunc, "main");
    defer scompiled.deinit();

    // Quad FS entry.
    var qfunc = try front.parseSpirv(gpa, fs_code);
    defer qfunc.deinit();
    _ = try spirv_jit.rewriteGraphicsQuad(&qfunc);
    var qcompiled = try spirv_jit.jitFunction(gpa, &qfunc, "main");
    defer qcompiled.deinit();

    // A program that carries both entries. We render once with the quad entry present
    // (quad path) and once with it nulled (scalar path), into two buffers.
    var prog = pipeline.ShaderProgram{
        .vs = scompiled, // unused by rasterShaded (it takes vouts directly), placeholder
        .fs = scompiled,
        .vs_inputs = 0,
        .fs_inputs = sinfo.input_count,
        .vs_buffers = 0,
        .fs_buffers = 0,
    };
    prog.fs_input_slots = sinfo.input_slots;
    prog.fs_entry = spirv_jit.mainEntry(&scompiled);
    prog.fs_quad = null; // do not double-free scompiled via deinit. We manage modules here

    // A triangle covering most of a 37x37 image (odd size to exercise quad edge clamping),
    // with distinct per-vertex varying colors so the interpolation differs across the image.
    const W: u32 = 37;
    const H: u32 = 37;
    const screen: [3][2]f32 = .{ .{ 2.0, 35.0 }, .{ 35.0, 33.0 }, .{ 18.0, 1.0 } };
    var vouts: [3][GfxOut.vertex_len]f32 = .{[_]f32{0} ** GfxOut.vertex_len} ** 3;
    // Varyings at location 0 (slots varying_base+0..2): R/G/B per vertex.
    vouts[0][GfxOut.varying_base + 0] = 0.9;
    vouts[0][GfxOut.varying_base + 1] = 0.1;
    vouts[0][GfxOut.varying_base + 2] = 0.2;
    vouts[1][GfxOut.varying_base + 0] = 0.2;
    vouts[1][GfxOut.varying_base + 1] = 0.8;
    vouts[1][GfxOut.varying_base + 2] = 0.3;
    vouts[2][GfxOut.varying_base + 0] = 0.3;
    vouts[2][GfxOut.varying_base + 1] = 0.25;
    vouts[2][GfxOut.varying_base + 2] = 0.95;

    var px_scalar = [_]u8{0} ** (W * H * 4);
    var px_quad = [_]u8{0} ** (W * H * 4);

    // Scalar render (quad entry nulled).
    prog.fs_quad_entry = null;
    rasterShaded(&px_scalar, W, H, screen, &vouts, &prog, null, null, &.{});
    // Quad render (quad entry present).
    prog.fs_quad_entry = spirv_jit.mainEntry(&qcompiled);
    rasterShaded(&px_quad, W, H, screen, &vouts, &prog, null, null, &.{});

    // Compare every pixel. Coverage must match exactly (no extra/missing fragments) and
    // each channel within 1 LSB (float-reorder in the barycentric interpolation only).
    var max_diff: i32 = 0;
    var covered_count: usize = 0;
    for (0..W * H) |i| {
        const s = px_scalar[i * 4 ..][0..4];
        const q = px_quad[i * 4 ..][0..4];
        // Coverage: a written pixel has alpha 255 (the FS writes a=1). Both must agree.
        const s_cov = s[3] != 0 or s[0] != 0 or s[1] != 0 or s[2] != 0;
        const q_cov = q[3] != 0 or q[0] != 0 or q[1] != 0 or q[2] != 0;
        try std.testing.expectEqual(s_cov, q_cov);
        if (s_cov) covered_count += 1;
        for (0..4) |c| {
            const d = @as(i32, s[c]) - @as(i32, q[c]);
            const ad = if (d < 0) -d else d;
            if (ad > max_diff) max_diff = ad;
        }
    }
    // The triangle must actually cover a substantial area (not a no-op pass).
    try std.testing.expect(covered_count > 200);
    // Within 1 LSB everywhere (float-reorder tolerance, documented).
    try std.testing.expect(max_diff <= 1);
}

test "applyBlend computes SRC_ALPHA/ONE_MINUS_SRC_ALPHA over a destination (the blend oracle)" {
    // src = rgba(1,0,0,0.5) (a 50%-translucent red fragment), dst = (0,0,1,1) (opaque blue).
    // SRC_ALPHA / ONE_MINUS_SRC_ALPHA, FUNC_ADD on both color and alpha:
    //   rgb = 0.5*(1,0,0) + 0.5*(0,0,1) = (0.5, 0, 0.5)
    //   a   = 0.5*0.5     + 0.5*1.0     = 0.75
    const src = F32x4{ 1, 0, 0, 0.5 };
    const dst = F32x4{ 0, 0, 1, 1 };
    const b = hal.BlendState{
        .enable = true,
        .src_color = .src_alpha,
        .dst_color = .one_minus_src_alpha,
        .src_alpha = .src_alpha,
        .dst_alpha = .one_minus_src_alpha,
        .color_op = .add,
        .alpha_op = .add,
    };
    const out = applyBlend(b, src, dst);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), out[0], 1.0 / 255.0);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), out[1], 1.0 / 255.0);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), out[2], 1.0 / 255.0);
    try std.testing.expectApproxEqAbs(@as(f32, 0.75), out[3], 1.0 / 255.0);

    // Default passthrough (src=one dst=zero add) returns the source unchanged.
    const pass = applyBlend(.{ .enable = true }, src, dst);
    try std.testing.expectApproxEqAbs(src[0], pass[0], 1e-6);
    try std.testing.expectApproxEqAbs(src[3], pass[3], 1e-6);
}

/// Build a flat-color fragment shader (SPIR-V) that outputs the constant `color` at
/// location 0. Used by the blend integration test (a translucent fragment over a
/// pre-cleared destination). The caller owns and frees the returned words.
fn buildFlatColorFs(gpa: std.mem.Allocator, color: [4]f32) ![]u32 {
    const vspirv = @import("vulcan-spirv");
    const op = vspirv.opcodes;
    const sc = op.StorageClass;
    var b = try vspirv.binary.Builder.init(gpa, 20);
    errdefer b.deinit(gpa);
    try b.emit(gpa, op.EntryPoint, &.{ op.ExecutionModel.fragment, 10, 0, 9 });
    try b.emit(gpa, op.Decorate, &.{ 9, op.Decoration.location, 0 });
    try b.emit(gpa, op.TypeVoid, &.{1});
    try b.emit(gpa, op.TypeFunction, &.{ 2, 1 });
    try b.emit(gpa, op.TypeFloat, &.{ 3, 32 });
    try b.emit(gpa, op.TypeVector, &.{ 5, 3, 4 });
    try b.emit(gpa, op.TypePointer, &.{ 7, sc.output, 5 });
    try b.emit(gpa, op.Constant, &.{ 3, 12, @bitCast(color[0]) });
    try b.emit(gpa, op.Constant, &.{ 3, 13, @bitCast(color[1]) });
    try b.emit(gpa, op.Constant, &.{ 3, 14, @bitCast(color[2]) });
    try b.emit(gpa, op.Constant, &.{ 3, 15, @bitCast(color[3]) });
    try b.emit(gpa, op.Variable, &.{ 7, 9, sc.output });
    try b.emit(gpa, op.Function, &.{ 1, 10, 0, 2 });
    try b.emit(gpa, op.Label, &.{11});
    try b.emit(gpa, op.CompositeConstruct, &.{ 5, 16, 12, 13, 14, 15 });
    try b.emit(gpa, op.Store, &.{ 9, 16 });
    try b.emit(gpa, op.Return, &.{});
    try b.emit(gpa, op.FunctionEnd, &.{});
    const words = try gpa.dupe(u32, b.words.items);
    b.deinit(gpa);
    return words;
}

test "rasterShadedFmt blends a translucent fragment over a pre-cleared destination" {
    // Full-rasterizer blend oracle: a flat (1,0,0,0.5) fragment shaded over a buffer
    // pre-cleared to blue (0,0,1,1) with SRC_ALPHA/ONE_MINUS_SRC_ALPHA, FUNC_ADD must
    // give (0.5, 0, 0.5, 0.75) inside the triangle. Disabled blend overwrites with src.
    const front = @import("../../spirv.zig");
    const gpa = std.testing.allocator;

    const fs_code = try buildFlatColorFs(gpa, .{ 1.0, 0.0, 0.0, 0.5 });
    defer gpa.free(fs_code);
    var ffunc = try front.parseSpirv(gpa, std.mem.sliceAsBytes(fs_code));
    defer ffunc.deinit();
    const finfo = try spirv_jit.rewriteGraphics(&ffunc);
    var fcompiled = try spirv_jit.jitFunction(gpa, &ffunc, "main");
    defer fcompiled.deinit();

    var prog = pipeline.ShaderProgram{
        .vs = fcompiled, // placeholder (rasterShaded takes vouts directly)
        .fs = fcompiled,
        .vs_inputs = 0,
        .fs_inputs = finfo.input_count,
        .vs_buffers = 0,
        .fs_buffers = 0,
    };
    prog.fs_input_slots = finfo.input_slots;
    prog.fs_entry = spirv_jit.mainEntry(&fcompiled);
    prog.fs_quad = null;
    prog.fs_quad_entry = null;

    const W: u32 = 32;
    const H: u32 = 32;
    // A triangle covering the center of the buffer.
    const screen: [3][2]f32 = .{ .{ 1.0, 1.0 }, .{ 31.0, 1.0 }, .{ 16.0, 31.0 } };
    var vouts: [3][GfxOut.vertex_len]f32 = .{[_]f32{0} ** GfxOut.vertex_len} ** 3;

    const blend = hal.BlendState{
        .enable = true,
        .src_color = .src_alpha,
        .dst_color = .one_minus_src_alpha,
        .src_alpha = .src_alpha,
        .dst_alpha = .one_minus_src_alpha,
    };

    // Blended render over a blue destination.
    var px = [_]u8{0} ** (W * H * 4);
    clear(&px, W, H, .{ .r = 0, .g = 0, .b = 1, .a = 1 });
    rasterShadedFmt(&px, W, H, .rgba8_unorm, 1, screen, &vouts, &prog, null, null, null, &.{}, blend, null, &.{}, .{});

    // Interior pixel (16, 12) is well inside the triangle: must be the blended color.
    const idx = (12 * W + 16) * 4;
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), @as(f32, @floatFromInt(px[idx + 0])) / 255.0, 1.5 / 255.0);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), @as(f32, @floatFromInt(px[idx + 1])) / 255.0, 1.5 / 255.0);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), @as(f32, @floatFromInt(px[idx + 2])) / 255.0, 1.5 / 255.0);
    try std.testing.expectApproxEqAbs(@as(f32, 0.75), @as(f32, @floatFromInt(px[idx + 3])) / 255.0, 1.5 / 255.0);

    // Disabled blend: the fragment overwrites the destination (red, alpha 0.5).
    var px2 = [_]u8{0} ** (W * H * 4);
    clear(&px2, W, H, .{ .r = 0, .g = 0, .b = 1, .a = 1 });
    rasterShadedFmt(&px2, W, H, .rgba8_unorm, 1, screen, &vouts, &prog, null, null, null, &.{}, .{}, null, &.{}, .{});
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), @as(f32, @floatFromInt(px2[idx + 0])) / 255.0, 1.5 / 255.0);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), @as(f32, @floatFromInt(px2[idx + 2])) / 255.0, 1.5 / 255.0);
}

test "rasterShadedFmt MRT writes each color output to its own target" {
    const front = @import("../../spirv.zig");
    const spirv = @import("vulcan-spirv");
    const gpa = std.testing.allocator;
    const op = spirv.opcodes;
    const sc = op.StorageClass;

    // FS: layout(location=0) out vec4 o0; layout(location=1) out vec4 o1;
    //     void main(){ o0=vec4(1,0,0,1); o1=vec4(0,1,0,1); }
    // ids: void=1 fnty=2 f32=3 v4=4 pOut=5 o0=6 o1=7 main=8 entry=9 one=10 zero=11
    //      red=12 green=13.
    var b = try spirv.binary.Builder.init(gpa, 14);
    try b.emit(gpa, op.EntryPoint, &.{ op.ExecutionModel.fragment, 8, 0, 6, 7 });
    try b.emit(gpa, op.Decorate, &.{ 6, op.Decoration.location, 0 });
    try b.emit(gpa, op.Decorate, &.{ 7, op.Decoration.location, 1 });
    try b.emit(gpa, op.TypeVoid, &.{1});
    try b.emit(gpa, op.TypeFunction, &.{ 2, 1 });
    try b.emit(gpa, op.TypeFloat, &.{ 3, 32 });
    try b.emit(gpa, op.TypeVector, &.{ 4, 3, 4 });
    try b.emit(gpa, op.TypePointer, &.{ 5, sc.output, 4 });
    try b.emit(gpa, op.Constant, &.{ 3, 10, @bitCast(@as(f32, 1.0)) });
    try b.emit(gpa, op.Constant, &.{ 3, 11, @bitCast(@as(f32, 0.0)) });
    try b.emit(gpa, op.Variable, &.{ 5, 6, sc.output });
    try b.emit(gpa, op.Variable, &.{ 5, 7, sc.output });
    try b.emit(gpa, op.Function, &.{ 1, 8, 0, 2 });
    try b.emit(gpa, op.Label, &.{9});
    try b.emit(gpa, op.CompositeConstruct, &.{ 4, 12, 10, 11, 11, 10 }); // red
    try b.emit(gpa, op.Store, &.{ 6, 12 });
    try b.emit(gpa, op.CompositeConstruct, &.{ 4, 13, 11, 10, 11, 10 }); // green
    try b.emit(gpa, op.Store, &.{ 7, 13 });
    try b.emit(gpa, op.Return, &.{});
    try b.emit(gpa, op.FunctionEnd, &.{});
    const words = try gpa.dupe(u32, b.words.items);
    b.deinit(gpa);
    defer gpa.free(words);

    var ffunc = try front.parseSpirv(gpa, std.mem.sliceAsBytes(words));
    defer ffunc.deinit();
    const finfo = try spirv_jit.rewriteGraphics(&ffunc);
    var fcompiled = try spirv_jit.jitFunction(gpa, &ffunc, "main");
    defer fcompiled.deinit();

    var prog = pipeline.ShaderProgram{ .vs = fcompiled, .fs = fcompiled, .vs_inputs = 0, .fs_inputs = finfo.input_count, .vs_buffers = 0, .fs_buffers = 0 };
    prog.fs_input_slots = finfo.input_slots;
    prog.fs_entry = spirv_jit.mainEntry(&fcompiled);
    prog.fs_quad = null;
    prog.fs_quad_entry = null;

    const W: u32 = 32;
    const H: u32 = 32;
    const screen: [3][2]f32 = .{ .{ 1.0, 1.0 }, .{ 31.0, 1.0 }, .{ 16.0, 31.0 } };
    var vouts: [3][GfxOut.vertex_len]f32 = .{[_]f32{0} ** GfxOut.vertex_len} ** 3;

    var t0 = [_]u8{0} ** (W * H * 4); // color target 0
    var t1 = [_]u8{0} ** (W * H * 4); // color target 1 (MRT)
    const extra = [_]ColorTarget{.{ .bytes = &t1, .format = .rgba8_unorm }};
    rasterShadedFmt(&t0, W, H, .rgba8_unorm, 1, screen, &vouts, &prog, null, null, null, &.{}, .{}, null, &extra, .{});

    const idx = (12 * W + 16) * 4;
    // Target 0 = red, target 1 = green (each color output routed to its own buffer).
    try std.testing.expectEqual(@as(u8, 255), t0[idx + 0]); // red.r
    try std.testing.expectEqual(@as(u8, 0), t0[idx + 1]);
    try std.testing.expectEqual(@as(u8, 0), t1[idx + 0]); // green.r
    try std.testing.expectEqual(@as(u8, 255), t1[idx + 1]); // green.g
}
