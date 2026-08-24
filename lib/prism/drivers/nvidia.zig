const std = @import("std");
const drv = @import("../driver.zig");
const nvidia = @import("nvidia");
const Driver = drv.Driver;
const Device = drv.Device;
const Error = drv.Error;
const NvDevice = @import("nvidia/device.zig").Device;

const State = struct {};
var state: State = .{};

fn available(ptr: *anyopaque) bool {
    _ = ptr;
    // The nvidia control device exists only on Linux. Same guard as the apple
    // and virgl drivers: probing from a darwin build dies with SIGSYS.
    if (@import("builtin").os.tag != .linux) return false;
    // Probe for a usable NVIDIA GPU: open the control device and run the RM
    // version handshake. If that succeeds, the from-scratch driver can drive it.
    var client = nvidia.Client.open() catch return false;
    client.deinit();
    return true;
}
fn createDevice(ptr: *anyopaque, gpa: std.mem.Allocator) Error!Device {
    _ = ptr;
    return NvDevice.create(gpa);
}

const vtable = Driver.VTable{ .isAvailable = &available, .createDevice = &createDevice };

pub const driver = Driver{ .name = "nvidia", .ptr = &state, .vtable = &vtable, .drm_driver = "nvidia" };

test "nvidia driver exposes its name and a real device path" {
    try std.testing.expectEqualStrings("nvidia", driver.name);
    // Availability and device creation both need real hardware. Exercise the
    // full path when a GPU is present, otherwise just confirm it reports absent.
    if (driver.isAvailable()) {
        const dev = try driver.createDevice(std.testing.allocator);
        dev.deinit();
    }
}
