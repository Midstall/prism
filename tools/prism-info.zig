//! prism-info: eglinfo-style probe. Prints the Prism version, then for each
//! compiled-in driver reports whether it is usable on THIS system (by actually
//! attempting to bring up a device AND querying its real capabilities through the
//! HAL), and which platform backends are available here.

const std = @import("std");
const builtin = @import("builtin");
const prism = @import("prism");
const hal = prism.hal;

fn linuxMain(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;
    var stdout_buf: [4096]u8 = undefined;
    var w = std.Io.File.stdout().writer(io, &stdout_buf);
    const out = &w.interface;

    try out.print("Prism {s}\n\n", .{prism.version});

    try out.print("Drivers ({d} compiled in):\n", .{prism.drivers.all.len});
    for (prism.drivers.all) |d| {
        const available = d.isAvailable();
        try out.print("  {s}\n", .{d.name});
        try out.print("    available:    {s}\n", .{if (available) "yes" else "no"});
        if (available) {
            // Actually try to bring the driver up on this machine (eglinfo-style).
            if (d.createDevice(gpa)) |device| {
                defer device.deinit();
                try out.print("    createDevice: ok\n", .{});
                try printCaps(out, device.caps());
            } else |err| {
                try out.print("    createDevice: {s}\n", .{@errorName(err)});
            }
        } else {
            try out.print("    createDevice: skipped (unavailable here)\n", .{});
        }
    }

    try out.print("\nPlatforms:\n", .{});
    try printPlatforms(out, init);

    try out.flush();
}

fn uefiMain() std.os.uefi.Status {
    return .success;
}

pub const main = if (builtin.os.tag == .uefi) uefiMain else linuxMain;

/// Print a brought-up device's capabilities, eglinfo-style: the device label, a
/// feature word list (omitting the false ones), the supported formats, and the
/// max texture dimension.
fn printCaps(out: anytype, caps: hal.DeviceCaps) !void {
    try out.print("    device:       {s}\n", .{caps.device_name});

    try out.print("    features:    ", .{});
    if (caps.graphics) try out.print(" graphics", .{});
    if (caps.compute) try out.print(" compute", .{});
    if (caps.present) try out.print(" present", .{});
    if (caps.spirv) try out.print(" spirv", .{});
    if (caps.hdr) try out.print(" hdr", .{});
    try out.print("\n", .{});

    try out.print("    formats:     ", .{});
    for (caps.formats) |f| try out.print(" {s}", .{@tagName(f)});
    try out.print("\n", .{});

    try out.print("    max texture:  {d}\n", .{caps.max_texture_dim});
}

/// Probe the platform present backends usable on THIS host and print each.
fn printPlatforms(out: anytype, init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;

    // headless: a pure allocator-backed framebuffer, always available.
    try out.print("  headless     available: yes\n", .{});

    // wayland: probe the socket, and when reachable ask the compositor which DRM
    // device it prefers + resolve that to a kernel driver name.
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    if (prism.platform.wayland.resolveSocketPath(init.environ_map, &path_buf)) |sock| {
        if (prism.platform.wayland.preferredDevice(gpa, io, sock)) |dev| {
            var name_buf: [64]u8 = undefined;
            if (prism.platform.drm.driverForDev(dev, &name_buf)) |kdriver| {
                try out.print("  wayland      available: yes  (socket: {s}; compositor prefers driver '{s}')\n", .{ sock, kdriver });
            } else {
                try out.print("  wayland      available: yes  (socket: {s}; compositor prefers an unrecognized device)\n", .{sock});
            }
        } else {
            try out.print("  wayland      available: yes  (socket: {s})\n", .{sock});
        }
    } else |_| {
        try out.print("  wayland      available: no   (WAYLAND_DISPLAY / XDG_RUNTIME_DIR unset)\n", .{});
    }

    // drm: read-only scan of /dev/dri/card* (no SET_MASTER, no modeset).
    try printDrm(out);

    // virtio_gpu: the present backend is the freestanding Conduit path only; on a
    // Linux host it is never usable directly.
    try out.print("  virtio_gpu   available: no   (freestanding/baremetal only)\n", .{});
}

/// List the DRM primary nodes that exist (/dev/dri/card0..card15), read-only. We
/// never take DRM master or modeset - just confirm the card device files exist.
fn printDrm(out: anytype) !void {
    var found: [16][]const u8 = undefined;
    var found_n: usize = 0;
    var name_store: [16][16]u8 = undefined;

    const linux = std.os.linux;
    var i: u32 = 0;
    while (i < 16) : (i += 1) {
        var path_buf: [32]u8 = undefined;
        const path = std.fmt.bufPrintZ(&path_buf, "/dev/dri/card{d}", .{i}) catch break;
        // Read-only existence probe: O_RDONLY|O_CLOEXEC. We never SET_MASTER or
        // modeset; opening the node read-only only confirms it exists. Close it
        // right back. (Mirrors how platform/drm.zig opens the node, minus RDWR.)
        const rc = linux.open(path.ptr, .{ .ACCMODE = .RDONLY, .CLOEXEC = true }, 0);
        if (std.posix.errno(rc) != .SUCCESS) continue;
        _ = linux.close(@intCast(rc));
        const name = std.fmt.bufPrint(&name_store[found_n], "card{d}", .{i}) catch continue;
        found[found_n] = name;
        found_n += 1;
    }

    if (found_n == 0) {
        try out.print("  drm          available: no   (no /dev/dri/card* nodes)\n", .{});
        return;
    }

    try out.print("  drm          available: yes  (", .{});
    for (found[0..found_n], 0..) |n, idx| {
        if (idx != 0) try out.print(", ", .{});
        try out.print("{s}", .{n});
    }
    try out.print(")\n", .{});
}
