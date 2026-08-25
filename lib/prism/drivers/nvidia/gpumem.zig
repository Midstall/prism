//! CPU access to GPU-mapped memory.
//!
//! Memory the GPU hands back through `mapMemory` is not ordinary RAM, and on
//! aarch64 the difference is enforced by the hardware. A mapping of that kind
//! permits aligned, single-register accesses only: the wide multi-register forms
//! (`stp q`, `ldp q`, the NEON list forms) and unaligned accesses raise a
//! synchronous external abort, which reaches the process as SIGBUS.
//!
//! `@memcpy` is exactly the operation an optimising build turns into those wide
//! forms. In Debug and ReleaseSmall it stays narrow and the fault never appears,
//! so the whole GPU path worked until somebody built with ReleaseFast, where 92
//! tests died with `Bus error` inside a 32 byte `@memcpy` into a descriptor
//! pool.
//!
//! The helpers here do the copy as explicit 32 bit accesses through a volatile
//! pointer. Volatile is what holds it: it stops the optimiser merging the loop
//! back into the wide stores it just took apart, at any optimisation level.
//!
//! Use these for every CPU read or write of mapped GPU memory. A plain
//! `@memcpy` there is a latent SIGBUS that only shows up in a release build.
const std = @import("std");

/// Write `src` into GPU memory at `dst_bytes`, as 32 bit stores.
///
/// `dst_bytes` must be 4 byte aligned and at least `src.len * 4` long. Both
/// hold for every caller: the mappings are page aligned and the descriptors are
/// whole numbers of words.
pub fn writeWords(dst_bytes: []u8, src: []const u32) void {
    std.debug.assert(dst_bytes.len >= src.len * @sizeOf(u32));
    const dst: [*]volatile u32 = @ptrCast(@alignCast(dst_bytes.ptr));
    for (src, 0..) |word, i| dst[i] = word;
}

/// Write `src` into GPU memory at `dst_bytes`, as 32 bit stores.
///
/// Floats go out through the same width as words: what the mapping cares about
/// is the size of the access, not what the bits mean.
pub fn writeFloats(dst_bytes: []u8, src: []const f32) void {
    std.debug.assert(dst_bytes.len >= src.len * @sizeOf(f32));
    const dst: [*]volatile f32 = @ptrCast(@alignCast(dst_bytes.ptr));
    for (src, 0..) |value, i| dst[i] = value;
}

/// Copy `src` into GPU memory at `dst_bytes`, whatever its element type.
///
/// Takes a slice or a pointer to an array and copies the bytes behind it with
/// the widest access the mapping allows. Use it where the payload is a vertex
/// struct or a fixed array, which would otherwise need one typed helper per
/// layout.
pub fn write(dst_bytes: []u8, src: anytype) void {
    const S = @TypeOf(src);
    const bytes: []const u8 = switch (@typeInfo(S)) {
        .pointer => |p| switch (p.size) {
            .one => std.mem.asBytes(src),
            else => std.mem.sliceAsBytes(src),
        },
        else => @compileError("gpumem.write needs a slice or a pointer to an array, found " ++ @typeName(S)),
    };
    writeRaw(dst_bytes, bytes);
}

/// The byte copy every writer ends at: whole words first, then any remainder.
fn writeRaw(dst_bytes: []u8, src: []const u8) void {
    std.debug.assert(dst_bytes.len >= src.len);
    const words = src.len / @sizeOf(u32);
    if (words > 0) {
        const dst: [*]volatile u32 = @ptrCast(@alignCast(dst_bytes.ptr));
        const src_words: [*]align(1) const u32 = @ptrCast(src.ptr);
        var i: usize = 0;
        while (i < words) : (i += 1) dst[i] = src_words[i];
    }
    var i = words * @sizeOf(u32);
    const tail: [*]volatile u8 = @ptrCast(dst_bytes.ptr);
    while (i < src.len) : (i += 1) tail[i] = src[i];
}

/// Copy `src` into GPU memory at `dst_bytes`, byte by byte.
///
/// For a payload that is not a whole number of words, or whose start is not word
/// aligned. Prefer `writeWords` where the shape allows it: this is four times
/// the accesses.
pub fn writeBytes(dst_bytes: []u8, src: []const u8) void {
    std.debug.assert(dst_bytes.len >= src.len);
    const dst: [*]volatile u8 = @ptrCast(dst_bytes.ptr);
    for (src, 0..) |byte, i| dst[i] = byte;
}

/// Fill `len` bytes of GPU memory at `dst_bytes` with zero, as 32 bit stores
/// where it can and single bytes for any remainder.
///
/// `@memset` is widened by an optimising build exactly as `@memcpy` is.
pub fn zero(dst_bytes: []u8, len: usize) void {
    std.debug.assert(dst_bytes.len >= len);
    const words = len / @sizeOf(u32);
    if (words > 0) {
        const dst: [*]volatile u32 = @ptrCast(@alignCast(dst_bytes.ptr));
        var i: usize = 0;
        while (i < words) : (i += 1) dst[i] = 0;
    }
    var i = words * @sizeOf(u32);
    const tail: [*]volatile u8 = @ptrCast(dst_bytes.ptr);
    while (i < len) : (i += 1) tail[i] = 0;
}

/// Read `dst.len` words out of GPU memory at `src_bytes`, as 32 bit loads.
pub fn readWords(dst: []u32, src_bytes: []const u8) void {
    std.debug.assert(src_bytes.len >= dst.len * @sizeOf(u32));
    const src: [*]const volatile u32 = @ptrCast(@alignCast(src_bytes.ptr));
    for (dst, 0..) |*word, i| word.* = src[i];
}

/// Copy `len` bytes out of GPU memory at `src_bytes` into ordinary memory.
///
/// The whole-word part is copied as words and any remaining one to three bytes
/// one at a time, so a length that is not a multiple of four is still read with
/// accesses the mapping allows.
pub fn readBytes(dst: []u8, src_bytes: []const u8, len: usize) void {
    std.debug.assert(dst.len >= len and src_bytes.len >= len);
    const words = len / @sizeOf(u32);
    if (words > 0) {
        const src: [*]const volatile u32 = @ptrCast(@alignCast(src_bytes.ptr));
        const dst_words: [*]u32 = @ptrCast(@alignCast(dst.ptr));
        var i: usize = 0;
        while (i < words) : (i += 1) dst_words[i] = src[i];
    }
    var i = words * @sizeOf(u32);
    const tail: [*]const volatile u8 = @ptrCast(src_bytes.ptr);
    while (i < len) : (i += 1) dst[i] = tail[i];
}

test "a word copy round trips through ordinary memory" {
    // The helpers are about the ACCESS WIDTH, not about the destination, so
    // ordinary memory exercises the same code the GPU mapping runs.
    var buf: [32]u8 align(4) = undefined;
    const src = [_]u32{ 0x11223344, 0x55667788, 0xDEADBEEF, 0 };
    writeWords(&buf, &src);

    var back: [4]u32 = undefined;
    readWords(&back, &buf);
    try std.testing.expectEqualSlices(u32, &src, &back);
}

test "readBytes copies a length that is not a whole number of words" {
    var buf: [16]u8 align(4) = undefined;
    for (&buf, 0..) |*b, i| b.* = @intCast(i);
    var out: [16]u8 = undefined;
    // Seven bytes is one word plus three, which is the case the tail loop is for.
    readBytes(&out, &buf, 7);
    try std.testing.expectEqualSlices(u8, buf[0..7], out[0..7]);
}

test "readBytes of nothing touches nothing" {
    var buf: [4]u8 align(4) = undefined;
    var out: [4]u8 = .{ 9, 9, 9, 9 };
    readBytes(&out, &buf, 0);
    try std.testing.expectEqualSlices(u8, &.{ 9, 9, 9, 9 }, &out);
}
