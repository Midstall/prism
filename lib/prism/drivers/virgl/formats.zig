//! HAL-to-virgl format mapping and the fp16 (IEEE half) codec for the HDR path.
//! virgl color formats are Gallium PIPE_FORMAT ordinals. 8-bit SDR maps to B8G8R8X8. HDR adds fp16 + 10-bit formats.
//! The fp16 helpers convert between binary16 (u16) and f32 so the HDR readback can decode values past SDR white.

const std = @import("std");
const hal = @import("../../hal.zig");
const enc = @import("encoding.zig");

/// Map a HAL color format to its virgl (Gallium PIPE_FORMAT) value. SDR
/// rgba8/bgra8 share B8G8R8X8 (the readback interprets the bytes as 0x00RRGGBB
/// regardless). HDR formats map to their high-precision pipe formats so values
/// outside SDR white are preserved by the render target.
pub fn virglColorFormat(f: hal.Format) u32 {
    return switch (f) {
        .rgba8_unorm, .bgra8_unorm => enc.FORMAT_B8G8R8X8_UNORM,
        .rgba16_float => enc.FORMAT_R16G16B16A16_FLOAT,
        .rgb10a2 => enc.FORMAT_R10G10B10A2_UNORM,
        .rgb10x2 => enc.FORMAT_B10G10R10X2_UNORM,
        // A depth attachment allocates a Z32_FLOAT ZETA surface (the transport
        // binds it DEPTH_STENCIL, keyed off the depth format ordinal).
        .depth32_float => enc.FORMAT_Z32_FLOAT,
        // r8 is never used as a virgl target here. Fall back to the SDR color
        // format so the surface create still has a valid format.
        else => enc.FORMAT_B8G8R8X8_UNORM,
    };
}

/// The virgl pipe format a sampled texture is stored and viewed as. Prism uploads
/// internal RGBA texel bytes, so 8-bit formats read as R8G8B8A8_UNORM (byte 0 = R)
/// and fp16 as R16G16B16A16_FLOAT. sRGB + single/dual-channel float fall back to
/// R8G8B8A8_UNORM (their exact virgl ordinals are unverified here, and there is no
/// host to check against), so those sample as raw 8-bit until confirmed.
pub fn virglSampledFormat(f: hal.Format) u32 {
    return switch (f) {
        .rgba16_float => enc.FORMAT_R16G16B16A16_FLOAT,
        else => enc.FORMAT_R8G8B8A8_UNORM,
    };
}

/// Bytes per pixel of the virgl render target for a HAL format. This drives the
/// resource size + transfer stride: an fp16 RT is 8 bytes/pixel, the 8/10-bit
/// ones 4. Mirrors hal.Format.bytesPerPixel but lives here so the transports do
/// not import hal directly for it.
pub fn bytesPerPixel(f: hal.Format) u32 {
    return f.bytesPerPixel();
}

// fp16 (IEEE 754 binary16) codec. The fp16 render target stores each channel as a half.
// The readback decodes it to f32 to verify HDR values past 1.0.

/// Decode an IEEE binary16 (stored in a u16) to an f32. Handles subnormals,
/// zero, infinities and NaN. Used by the readback to recover the GPU-written
/// linear-light values from the fp16 render target.
pub fn halfToFloat(h: u16) f32 {
    const sign: u32 = @as(u32, h >> 15) & 0x1;
    const exp: u32 = @as(u32, h >> 10) & 0x1f;
    const mant: u32 = @as(u32, h) & 0x3ff;

    var bits: u32 = sign << 31;
    if (exp == 0) {
        if (mant == 0) {
            // signed zero
            return @bitCast(bits);
        }
        // subnormal half -> normalize into an f32
        var e: i32 = -1;
        var m = mant;
        while (m & 0x400 == 0) {
            m <<= 1;
            e -= 1;
        }
        m &= 0x3ff;
        const fexp: u32 = @intCast(127 - 15 + e + 1);
        bits |= (fexp << 23) | (m << 13);
        return @bitCast(bits);
    }
    if (exp == 0x1f) {
        // inf / nan
        bits |= (0xff << 23) | (mant << 13);
        return @bitCast(bits);
    }
    // normal
    const fexp: u32 = exp + (127 - 15);
    bits |= (fexp << 23) | (mant << 13);
    return @bitCast(bits);
}

/// Encode an f32 to an IEEE binary16 (round-toward-zero on the mantissa, which is
/// sufficient for the test's exact-ish values). Used by the host unit test to
/// round-trip values through the half representation.
pub fn floatToHalf(f: f32) u16 {
    const bits: u32 = @bitCast(f);
    const sign: u16 = @intCast((bits >> 16) & 0x8000);
    const exp_f: i32 = @intCast((bits >> 23) & 0xff);
    const mant_f: u32 = bits & 0x7fffff;

    if (exp_f == 0xff) {
        // inf / nan
        const m: u16 = if (mant_f != 0) 0x200 else 0;
        return sign | 0x7c00 | m;
    }
    const e: i32 = exp_f - 127 + 15;
    if (e >= 0x1f) {
        // overflow -> inf
        return sign | 0x7c00;
    }
    if (e <= 0) {
        // subnormal or underflow to zero
        if (e < -10) return sign;
        const m = (mant_f | 0x800000) >> @intCast(14 - e);
        return sign | @as(u16, @intCast(m));
    }
    const m: u16 = @intCast(mant_f >> 13);
    return sign | (@as(u16, @intCast(e)) << 10) | m;
}

test "hal format maps to the expected virgl pipe format" {
    try std.testing.expectEqual(enc.FORMAT_B8G8R8X8_UNORM, virglColorFormat(.rgba8_unorm));
    try std.testing.expectEqual(enc.FORMAT_B8G8R8X8_UNORM, virglColorFormat(.bgra8_unorm));
    try std.testing.expectEqual(@as(u32, 94), virglColorFormat(.rgba16_float));
    try std.testing.expectEqual(@as(u32, 8), virglColorFormat(.rgb10a2));
    try std.testing.expectEqual(@as(u32, 233), virglColorFormat(.rgb10x2));
    try std.testing.expectEqual(enc.FORMAT_Z32_FLOAT, virglColorFormat(.depth32_float));
    try std.testing.expect(enc.isDepthFormat(virglColorFormat(.depth32_float)));
    try std.testing.expect(!enc.isDepthFormat(virglColorFormat(.rgba8_unorm)));
}

test "virgl bytes-per-pixel tracks the format" {
    try std.testing.expectEqual(@as(u32, 4), bytesPerPixel(.rgba8_unorm));
    try std.testing.expectEqual(@as(u32, 8), bytesPerPixel(.rgba16_float));
    try std.testing.expectEqual(@as(u32, 4), bytesPerPixel(.rgb10a2));
}

test "fp16 decodes the canonical halves" {
    try std.testing.expectEqual(@as(f32, 0.0), halfToFloat(0x0000));
    try std.testing.expectEqual(@as(f32, 1.0), halfToFloat(0x3c00));
    try std.testing.expectEqual(@as(f32, 2.0), halfToFloat(0x4000));
    try std.testing.expectEqual(@as(f32, 4.0), halfToFloat(0x4400));
    try std.testing.expectEqual(@as(f32, -1.0), halfToFloat(0xbc00));
    try std.testing.expectEqual(@as(f32, 0.5), halfToFloat(0x3800));
}

test "fp16 round-trips HDR-range values" {
    // The exact HDR test values: red ramp to 4.0, green ramp to 2.0, plus 1.0.
    const vals = [_]f32{ 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0 };
    for (vals) |v| {
        const h = floatToHalf(v);
        const back = halfToFloat(h);
        try std.testing.expectEqual(v, back);
    }
    // The key HDR property: 4.0 is representable in fp16 (impossible in 8-bit
    // unorm, which clamps to 1.0).
    try std.testing.expect(halfToFloat(floatToHalf(4.0)) > 1.0);
}
