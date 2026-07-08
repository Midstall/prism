const std = @import("std");
const hal = @import("../../hal.zig");
const sw = @import("../software.zig");
const shader = sw.shader;
const platform = @import("../../platform.zig");

test "render triangle and present to headless surface" {
    const gpa = std.testing.allocator;

    // Create headless display and surface.
    const display = try platform.headless.create(gpa);
    defer display.deinit();

    const W = 16;
    const H = 16;

    var platform_surface = try display.createSurface(.{ .width = W, .height = H });
    defer platform_surface.deinit();

    // Create software device.
    const device = try sw.driver.createDevice(gpa);
    defer device.deinit();

    // Wrap platform surface into a HAL surface.
    const hal_surface = try device.createSurface(@ptrCast(&platform_surface));
    defer device.destroySurface(hal_surface);

    // Create render target image (rgba8, same size).
    const target = try device.createResource(.{ .image = .{
        .width = W,
        .height = H,
        .format = .rgba8_unorm,
        .usage = .{ .render_target = true },
    } });
    defer device.destroyResource(target);

    // Build triangle vertex data: pos(2xf32) + color(4xf32) = 24 bytes.
    const Vtx = extern struct { x: f32, y: f32, r: f32, g: f32, b: f32, a: f32 };
    const tri = [3]Vtx{
        .{ .x = -1, .y = -1, .r = 1, .g = 0, .b = 0, .a = 1 },
        .{ .x = 1, .y = -1, .r = 0, .g = 1, .b = 0, .a = 1 },
        .{ .x = 0, .y = 1, .r = 0, .g = 0, .b = 1, .a = 1 },
    };
    const vbuf = try device.createResource(.{ .buffer = .{
        .size = @sizeOf(@TypeOf(tri)),
        .usage = .{ .vertex = true },
    } });
    defer device.destroyResource(vbuf);
    @memcpy(try device.mapResource(vbuf), std.mem.asBytes(&tri));

    // Shaders.
    const vs_bytes = shader.encodeVertex(.{ .position_attr = 0, .color_attr = 1 });
    const vs = try device.createShaderModule(.{ .stage = .vertex, .code = &vs_bytes });
    defer device.destroyShaderModule(vs);
    const fs_bytes = shader.encodeFragment(.{});
    const fs = try device.createShaderModule(.{ .stage = .fragment, .code = &fs_bytes });
    defer device.destroyShaderModule(fs);

    // Pipeline.
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

    // Render into target.
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

    // Present the rendered image to the headless surface.
    try ctx.present(hal_surface, target);

    // Read back the headless surface buffer and assert.
    const buf = try platform_surface.currentBuffer();

    // Bottom-center pixel should be inside the triangle (non-black).
    const bx: usize = W / 2;
    const by: usize = H - 2;
    const idx = (by * W + bx) * 4;
    const inside = buf.bytes[idx + 0] != 0 or buf.bytes[idx + 1] != 0 or buf.bytes[idx + 2] != 0;
    try std.testing.expect(inside);

    // Top-left corner should be the clear color (black).
    const corner: usize = (0 * W + 0) * 4;
    try std.testing.expectEqual(@as(u8, 0), buf.bytes[corner + 0]);
    try std.testing.expectEqual(@as(u8, 0), buf.bytes[corner + 1]);
    try std.testing.expectEqual(@as(u8, 0), buf.bytes[corner + 2]);
}
