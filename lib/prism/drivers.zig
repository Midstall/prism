const std = @import("std");
const builtin = @import("builtin");
const build_options = @import("build-options");
const driver = @import("driver.zig");

pub const Driver = driver.Driver;
pub const software = @import("drivers/software.zig");

/// Compiled-in drivers, selected at build time via -Ddrivers (build_options flags).
/// An unselected driver's source is in a comptime-dead branch and never analyzed.
fn buildAll() []const Driver {
    // The apple, nvidia and virgl probes issue raw Linux syscalls (DRM render
    // nodes, the nvidia control device). On a darwin build those run a Linux
    // syscall number on the macOS kernel and the process dies with SIGSYS,
    // which is what the aarch64-darwin CI hit in the headless-backend tests.
    // The OS gate keeps that code comptime-dead off Linux: not analyzed, not
    // probed, and the software rasterizer is the universal fallback.
    const on_linux = builtin.os.tag == .linux;
    const sw_list: []const Driver = if (build_options.driver_software) &[_]Driver{@import("drivers/software.zig").driver} else &.{};
    const ap_list: []const Driver = if (build_options.driver_apple and on_linux and builtin.cpu.arch == .aarch64) &[_]Driver{@import("drivers/apple.zig").driver} else &.{};
    const nv_list: []const Driver = if (build_options.driver_nvidia and on_linux) &[_]Driver{@import("drivers/nvidia.zig").driver} else &.{};
    const vg_list: []const Driver = if (build_options.driver_virgl and on_linux) &[_]Driver{@import("drivers/virgl.zig").driver} else &.{};
    // Auto-selection preference order: real hardware drivers first, the software
    // rasterizer last as the universal fallback.
    return ap_list ++ nv_list ++ vg_list ++ sw_list;
}

pub const all: []const Driver = buildAll();

/// Pick a compiled-in driver by name, or the first available one (in preference
/// order) when name is null. Availability is a cheap predicate. A driver can
/// report available yet have an unimplemented createDevice. Use createBestDevice
/// when you want one that is actually usable.
pub fn select(name: ?[]const u8) ?Driver {
    if (name) |want| {
        for (all) |d| if (std.mem.eql(u8, d.name, want)) return d;
        return null;
    }
    for (all) |d| if (d.isAvailable()) return d;
    return null;
}

pub const Selected = struct { driver: Driver, device: driver.Device };

/// Bring up the best driver that actually works: walk the drivers in preference
/// order (hardware first, software last), skip the unavailable, and return the
/// first that starts AND draws.
///
/// Drawing is checked, not assumed. This used to return the first driver whose
/// `createDevice` succeeded, which is a weaker question than it looks: a
/// compute-only driver starts perfectly well and then rejects the first graphics
/// pipeline built on it. See `canDraw`.
///
/// The caller owns and must deinit the device. Returns null when no driver can
/// draw here, not even the software fallback.
pub fn createBestDevice(gpa: std.mem.Allocator) ?Selected {
    for (all) |d| {
        if (!d.isAvailable()) continue;
        const dev = d.createDevice(gpa) catch continue;
        if (!canDraw(gpa, dev)) {
            dev.deinit();
            continue;
        }
        return .{ .driver = d, .device = dev };
    }
    return null;
}

/// Whether `dev` can put geometry on a render target, rather than merely start.
///
/// Starting says almost nothing. The apple driver starts on any Asahi machine
/// because it is compute-capable, and then rejects every vertex and fragment
/// module at `createShaderModule` (see `drivers/apple/shader.zig`, where the
/// graphics path is a documented open item). A caller that took the first
/// driver to start got that one, never reached the software rasterizer behind
/// it, and failed at the first pipeline it built.
///
/// So this asks the only question that separates them: draw one triangle over a
/// contrasting background and read the middle pixel back. An error anywhere
/// along the way is an answer too, and the answer is no.
///
/// It costs a shader compile and a JIT per candidate, once per process. That is
/// worth more than handing back a device that cannot draw.
fn canDraw(gpa: std.mem.Allocator, dev: driver.Device) bool {
    return probeDraw(gpa, dev) catch false;
}

fn probeDraw(gpa: std.mem.Allocator, dev: driver.Device) !bool {
    const glsl = @import("glsl.zig");

    const vs_src =
        \\attribute vec2 aPos;
        \\void main() { gl_Position = vec4(aPos, 0.0, 1.0); }
    ;
    const fs_src =
        \\precision mediump float;
        \\void main() { gl_FragColor = vec4(1.0, 0.0, 0.0, 1.0); }
    ;

    const vs_spirv = try glsl.compileForStage(gpa, vs_src, .vertex);
    defer gpa.free(vs_spirv);
    const fs_spirv = try glsl.compileForStage(gpa, fs_src, .fragment);
    defer gpa.free(fs_spirv);

    const vs = try dev.createShaderModule(.{ .stage = .vertex, .code = vs_spirv });
    defer dev.destroyShaderModule(vs);
    const fs = try dev.createShaderModule(.{ .stage = .fragment, .code = fs_spirv });
    defer dev.destroyShaderModule(fs);

    const pipeline = try dev.createPipeline(.{
        .vertex = vs,
        .fragment = fs,
        .vertex_layout = .{
            .stride = @sizeOf([2]f32),
            .attributes = &.{.{ .location = 0, .format = .r32g32_float, .offset = 0 }},
        },
        .color_format = .rgba8_unorm,
    });
    defer dev.destroyPipeline(pipeline);

    // One triangle big enough to cover the whole target, so the middle pixel is
    // inside it whichever way the rasterizer rounds.
    const verts = [_][2]f32{ .{ -1, -1 }, .{ 3, -1 }, .{ -1, 3 } };
    const vbuf = try dev.createResource(.{ .buffer = .{ .size = @sizeOf(@TypeOf(verts)), .usage = .{ .vertex = true } } });
    defer dev.destroyResource(vbuf);
    @memcpy(try dev.mapResource(vbuf), std.mem.sliceAsBytes(verts[0..]));

    const size: u32 = 8;
    const target = try dev.createResource(.{ .image = .{
        .width = size,
        .height = size,
        .format = .rgba8_unorm,
        .usage = .{ .render_target = true },
    } });
    defer dev.destroyResource(target);

    const ctx = try dev.createContext();
    defer ctx.deinit();
    const cb = try ctx.beginCommands();
    defer cb.deinit();
    try cb.setRenderTarget(target);
    // Blue, which differs from the red the shader writes in every channel that
    // is checked. A target left at its clear colour cannot read as a drawn one.
    try cb.clear(.{ .r = 0, .g = 0, .b = 1, .a = 1 });
    try cb.bindPipeline(pipeline);
    try cb.bindVertexBuffer(vbuf);
    try cb.draw(verts.len, 0);
    try ctx.submit(cb);

    const px = try dev.mapResource(target);
    const centre = (size / 2 * size + size / 2) * 4;
    if (centre + 2 >= px.len) return false;
    return px[centre] > 200 and px[centre + 2] < 80;
}

/// Pick the compiled-in driver that drives the DRM device with the given dev_t,
/// e.g. the compositor's preferred render device from Wayland dmabuf
/// feedback's main_device. Resolves the dev_t to a kernel driver name (via
/// sysfs) and matches it to a driver's drm_driver. Returns null if nothing matches.
/// The caller can then fall back to createBestDevice.
pub fn selectForDrmDevice(dev: u64) ?Driver {
    var buf: [64]u8 = undefined;
    const kname = @import("platform/drm.zig").driverForDev(dev, &buf) orelse return null;
    for (all) |d| {
        const dd = d.drm_driver orelse continue;
        if (std.mem.eql(u8, dd, kname)) return d;
    }
    return null;
}

test "drivers list and selection" {
    try std.testing.expect(all.len >= 1);
    try std.testing.expect(select(all[0].name) != null);
    try std.testing.expect(select("does-not-exist") == null);
    try std.testing.expect(select(null) != null);
}

test "createBestDevice brings up a usable device (hardware if present, else software)" {
    const sel = createBestDevice(std.testing.allocator) orelse return error.NoWorkingDriver;
    defer sel.device.deinit();
    // It must be genuinely usable, not just selected.
    const r = try sel.device.createResource(.{ .buffer = .{ .size = 16 } });
    sel.device.destroyResource(r);
}

test "selectForDrmDevice matches the first render node to a driver (skips otherwise)" {
    if (builtin.os.tag != .linux) return error.SkipZigTest;

    // The first DRM render node (major 226, minor 128). On a box whose GPU has a
    // compiled-in driver (here the NVIDIA GPU) this maps to that driver. On any
    // other box (no render node, or an unsupported GPU) it skips.
    const dev = @import("platform/drm.zig").makedev(226, 128);
    const d = selectForDrmDevice(dev) orelse return error.SkipZigTest;
    try std.testing.expect(d.drm_driver != null);
}

test "the device createBestDevice returns can draw, not merely start" {
    // The whole point of the selection: a caller takes what this hands back and
    // builds a graphics pipeline on it. A driver that starts and cannot draw
    // used to win here and fail at that pipeline, with the working software
    // rasterizer behind it never tried.
    const gpa = std.testing.allocator;
    const sel = createBestDevice(gpa) orelse return error.SkipZigTest;
    defer sel.device.deinit();
    try std.testing.expect(canDraw(gpa, sel.device));
}

test "the software rasterizer draws, so it is a real fallback" {
    // It is the universal last resort, the only driver on a machine with no GPU
    // and the one in a build sandbox. A fallback that cannot draw is not one.
    const gpa = std.testing.allocator;
    const d = select("software") orelse return error.SkipZigTest;
    const dev = d.createDevice(gpa) catch return error.SkipZigTest;
    defer dev.deinit();
    try std.testing.expect(canDraw(gpa, dev));
}
