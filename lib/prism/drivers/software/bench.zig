//! Software-render benchmark, gated behind `PRISM_BENCH`. Skipped in the normal test run.
//! Two workloads: "light" (channel-rotate FS, rasterizer-bound) and "heavy" (vkcube FS, sampler+pow/sRGB).
//! Run: PRISM_BENCH=1 zig build test ... | grep BENCH. Tune with PRISM_BENCH_DIM / PRISM_BENCH_FRAMES.

const std = @import("std");
const builtin = @import("builtin");
const hal = @import("../../hal.zig");
const Device = @import("device.zig").Device;
const Pipeline = @import("pipeline.zig").Pipeline;
const sw_shader = @import("shader.zig");

fn getEnv(name: []const u8) ?[]const u8 {
    return std.testing.environ.getPosix(name);
}

fn envUsize(name: []const u8, default: usize) usize {
    const v = getEnv(name) orelse return default;
    return std.fmt.parseInt(usize, std.mem.trim(u8, v, " \t\r\n"), 10) catch default;
}

fn benchEnabled() bool {
    return getEnv("PRISM_BENCH") != null;
}

/// Monotonic nanosecond clock via the raw linux syscall (this test binary may not
/// link libc, and std.time.Timer is unavailable in this std build).
fn nowNs() u64 {
    var ts: std.os.linux.timespec = undefined;
    _ = std.os.linux.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * 1_000_000_000 + @as(u64, @intCast(ts.nsec));
}

fn wf(buf: []u8, off: usize, v: f32) void {
    @memcpy(buf[off..][0..4], &std.mem.toBytes(v));
}

/// Build the passthrough VS (in vec2 pos loc0 + vec3 color loc1 -> gl_Position + out vc loc0).
fn buildVs(gpa: std.mem.Allocator) ![]u32 {
    const spirv = @import("vulcan-spirv");
    const op = spirv.opcodes;
    const sc = op.StorageClass;
    var vsb = try spirv.binary.Builder.init(gpa, 24);
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
    return gpa.dupe(u32, vsb.words.items);
}

/// Channel-rotate FS: in vec3 vc(loc0) -> out vec4 o = vec4(vc.b, vc.r, vc.g, 1).
fn buildLightFs(gpa: std.mem.Allocator) ![]u32 {
    const spirv = @import("vulcan-spirv");
    const op = spirv.opcodes;
    const sc = op.StorageClass;
    var fsb = try spirv.binary.Builder.init(gpa, 18);
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
    try fsb.emit(gpa, op.CompositeExtract, &.{ 3, 14, 13, 2 });
    try fsb.emit(gpa, op.CompositeExtract, &.{ 3, 15, 13, 0 });
    try fsb.emit(gpa, op.CompositeExtract, &.{ 3, 16, 13, 1 });
    try fsb.emit(gpa, op.CompositeConstruct, &.{ 5, 17, 14, 15, 16, 12 });
    try fsb.emit(gpa, op.Store, &.{ 9, 17 });
    try fsb.emit(gpa, op.Return, &.{});
    try fsb.emit(gpa, op.FunctionEnd, &.{});
    return gpa.dupe(u32, fsb.words.items);
}

fn runWorkload(gpa: std.mem.Allocator, name: []const u8, dim: usize, frames: usize, fs_code: []const u8) !void {
    const W: u32 = @intCast(dim);
    const H: u32 = @intCast(dim);

    const vs_code = try buildVs(gpa);
    defer gpa.free(vs_code);

    const dev = try Device.create(gpa);
    defer dev.deinit();

    const vs = try dev.createShaderModule(.{ .stage = .vertex, .code = std.mem.sliceAsBytes(vs_code) });
    defer dev.destroyShaderModule(vs);
    const fs = try dev.createShaderModule(.{ .stage = .fragment, .code = fs_code });
    defer dev.destroyShaderModule(fs);

    const target = try dev.createResource(.{ .image = .{ .width = W, .height = H, .format = .rgba8_unorm } });
    defer dev.destroyResource(target);

    // One big triangle covering ~the whole framebuffer so the per-fragment work dominates.
    const stride: u32 = 20;
    const vbuf = try dev.createResource(.{ .buffer = .{ .size = 3 * stride } });
    defer dev.destroyResource(vbuf);
    const vb = try dev.mapResource(vbuf);
    wf(vb, 0, -1.0);
    wf(vb, 4, -1.0);
    wf(vb, 8, 1);
    wf(vb, 12, 0);
    wf(vb, 16, 0);
    wf(vb, stride + 0, 3.0);
    wf(vb, stride + 4, -1.0);
    wf(vb, stride + 8, 0);
    wf(vb, stride + 12, 1);
    wf(vb, stride + 16, 0);
    wf(vb, 2 * stride + 0, -1.0);
    wf(vb, 2 * stride + 4, 3.0);
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

    const sw_pipe: *Pipeline = @ptrCast(@alignCast(pipe));
    if (sw_pipe.program == null) {
        std.debug.print("BENCH[{s}] SKIP: FS did not JIT (declarative fallback)\n", .{name});
        return;
    }
    // PRISM_NO_QUAD forces the scalar per-fragment FS path (drops the 4-wide quad entry)
    // so the SIMD speedup can be A/B-measured against the scalar baseline under identical
    // box load. The quad path is on by default (when the FS is widenable).
    const quad_on = sw_pipe.program.?.fs_quad_entry != null;
    if (getEnv("PRISM_NO_QUAD") != null) {
        sw_pipe.program.?.fs_quad_entry = null;
    }
    std.debug.print("BENCH[{s}] quad-simd={s} (widenable={s})\n", .{
        name,
        if (sw_pipe.program.?.fs_quad_entry != null) "ON" else "OFF",
        if (quad_on) "yes" else "no",
    });

    const ctx = try dev.createContext();
    defer ctx.deinit();

    // Warmup frame (touch pages, prime caches).
    {
        const cb = try ctx.beginCommands();
        defer cb.deinit();
        try cb.setRenderTarget(target);
        try cb.clear(.{ .r = 0, .g = 0, .b = 0, .a = 1 });
        try cb.bindPipeline(pipe);
        try cb.bindVertexBuffer(vbuf);
        try cb.draw(3, 0);
        try ctx.submit(cb);
    }

    const t0 = nowNs();
    var f: usize = 0;
    while (f < frames) : (f += 1) {
        const cb = try ctx.beginCommands();
        defer cb.deinit();
        try cb.setRenderTarget(target);
        try cb.clear(.{ .r = 0, .g = 0, .b = 0, .a = 1 });
        try cb.bindPipeline(pipe);
        try cb.bindVertexBuffer(vbuf);
        try cb.draw(3, 0);
        try ctx.submit(cb);
    }
    const ns = nowNs() - t0;

    const ms_total = @as(f64, @floatFromInt(ns)) / 1.0e6;
    const ms_frame = ms_total / @as(f64, @floatFromInt(frames));
    const pixels = @as(f64, @floatFromInt(@as(usize, W) * H));
    const mpix_s = (pixels * @as(f64, @floatFromInt(frames))) / (@as(f64, @floatFromInt(ns)) / 1.0e9) / 1.0e6;
    const fps = 1000.0 / ms_frame;
    std.debug.print("BENCH[{s}] {d}x{d} frames={d}: {d:.3} ms/frame  {d:.1} Mpix/s  {d:.1} fps-eq\n", .{ name, W, H, frames, ms_frame, mpix_s, fps });
}

test "PRISM_BENCH software render benchmark" {
    if (!benchEnabled()) return error.SkipZigTest;

    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const gpa = arena.allocator();

    const dim = envUsize("PRISM_BENCH_DIM", 1024);
    const frames = envUsize("PRISM_BENCH_FRAMES", 60);

    std.debug.print("\n=== PRISM software render benchmark (dim={d} frames={d}) ===\n", .{ dim, frames });

    // Light FS: channel-rotate (tiny body), rasterizer-bound.
    {
        const fs_code = try buildLightFs(gpa);
        try runWorkload(gpa, "light", dim, frames, std.mem.sliceAsBytes(fs_code));
    }

    // Heavy FS: vkcube's real FS (sampler + dFdx/dFdy + pow/sRGB).
    {
        const raw align(4) = @embedFile("testdata_vkcube_fs.spv").*;
        try runWorkload(gpa, "heavy", dim, frames, &raw);
    }
}
