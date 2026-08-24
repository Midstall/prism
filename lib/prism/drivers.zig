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
/// first whose createDevice succeeds. Stub drivers (createDevice unimplemented)
/// are skipped automatically. The caller owns and must deinit the device.
/// Returns null only when no driver can start, not even the software fallback.
pub fn createBestDevice(gpa: std.mem.Allocator) ?Selected {
    for (all) |d| {
        if (!d.isAvailable()) continue;
        const dev = d.createDevice(gpa) catch continue;
        return .{ .driver = d, .device = dev };
    }
    return null;
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

test "hardware probes are compiled in only on Linux" {
    // The apple, nvidia and virgl probes issue raw Linux syscalls. Off Linux
    // they must not exist in the list at all: a probe there dies with SIGSYS
    // before the walk ever reaches the software fallback. This runs green
    // everywhere and would have caught the aarch64-darwin CI crash.
    if (builtin.os.tag != .linux) {
        for (all) |d| try std.testing.expectEqualStrings("software", d.name);
    }
}

test "selectForDrmDevice matches the first render node to a driver (skips otherwise)" {
    // The first DRM render node (major 226, minor 128). On a box whose GPU has a
    // compiled-in driver (here the NVIDIA GPU) this maps to that driver. On any
    // other box (no render node, or an unsupported GPU) it skips.
    const dev = @import("platform/drm.zig").makedev(226, 128);
    const d = selectForDrmDevice(dev) orelse return error.SkipZigTest;
    try std.testing.expect(d.drm_driver != null);
}
