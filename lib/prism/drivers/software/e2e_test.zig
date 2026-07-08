const std = @import("std");
const sw = @import("../software.zig");
const shader = sw.shader;

test "software driver rasterizes a triangle end to end" {
    const gpa = std.testing.allocator;
    const device = try sw.driver.createDevice(gpa);
    defer device.deinit();

    const W = 16;
    const H = 16;
    const target = try device.createResource(.{ .image = .{ .width = W, .height = H, .format = .rgba8_unorm, .usage = .{ .render_target = true } } });
    defer device.destroyResource(target);

    // 3 vertices: pos (2 f32) + color (4 f32) = 24 bytes stride.
    const Vtx = extern struct { x: f32, y: f32, r: f32, g: f32, b: f32, a: f32 };
    const tri = [3]Vtx{
        .{ .x = -1, .y = -1, .r = 1, .g = 0, .b = 0, .a = 1 },
        .{ .x = 1, .y = -1, .r = 0, .g = 1, .b = 0, .a = 1 },
        .{ .x = 0, .y = 1, .r = 0, .g = 0, .b = 1, .a = 1 },
    };
    const vbuf = try device.createResource(.{ .buffer = .{ .size = @sizeOf(@TypeOf(tri)), .usage = .{ .vertex = true } } });
    defer device.destroyResource(vbuf);
    @memcpy(try device.mapResource(vbuf), std.mem.asBytes(&tri));

    const vs_bytes = shader.encodeVertex(.{ .position_attr = 0, .color_attr = 1 });
    const vs = try device.createShaderModule(.{ .stage = .vertex, .code = &vs_bytes });
    defer device.destroyShaderModule(vs);
    const fs_bytes = shader.encodeFragment(.{});
    const fs = try device.createShaderModule(.{ .stage = .fragment, .code = &fs_bytes });
    defer device.destroyShaderModule(fs);

    const pipeline = try device.createPipeline(.{
        .vertex = vs,
        .fragment = fs,
        .vertex_layout = .{ .stride = 24, .attributes = &.{
            .{ .location = 0, .format = .rgba8_unorm, .offset = 0 },
            .{ .location = 1, .format = .rgba8_unorm, .offset = 8 },
        } },
        .color_format = .rgba8_unorm,
    });
    defer device.destroyPipeline(pipeline);

    const ctx = try device.createContext();
    defer ctx.deinit();
    const cb = try ctx.beginCommands();
    defer cb.deinit();
    try cb.setRenderTarget(target);
    try cb.clear(.{ .r = 0, .g = 0, .b = 0, .a = 1 });
    try cb.bindPipeline(pipeline);
    try cb.bindVertexBuffer(vbuf);
    try cb.draw(3, 0);
    try ctx.submit(cb);

    const pixels = try device.mapResource(target);
    // Bottom-center pixel sits inside the triangle near the bottom edge (mix of red+green), not the clear color.
    const bx = W / 2;
    const by = H - 2;
    const idx = (by * W + bx) * 4;
    const inside = pixels[idx + 0] != 0 or pixels[idx + 1] != 0 or pixels[idx + 2] != 0;
    try std.testing.expect(inside);
    // A top corner is outside the triangle -> still clear (black, alpha 255).
    const corner = (0 * W + 0) * 4;
    try std.testing.expectEqual(@as(u8, 0), pixels[corner + 0]);
    try std.testing.expectEqual(@as(u8, 0), pixels[corner + 1]);
    try std.testing.expectEqual(@as(u8, 0), pixels[corner + 2]);
}
