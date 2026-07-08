const std = @import("std");
const hal = @import("hal.zig");

pub const Error = @import("error.zig").Error;
pub const Device = hal.Device;

/// A graphics driver. Vtable fat-pointer (std.mem.Allocator style). The driver
/// owns its state behind `ptr`. Consumers call through the wrapper methods.
pub const Driver = struct {
    name: []const u8,
    ptr: *anyopaque,
    vtable: *const VTable,
    /// The kernel DRM driver this Prism driver corresponds to (e.g. "nvidia"),
    /// or null if it has no DRM device association (the software rasterizer).
    /// Used to match a compositor's preferred render device to a driver.
    drm_driver: ?[]const u8 = null,

    pub const VTable = struct {
        isAvailable: *const fn (ptr: *anyopaque) bool,
        createDevice: *const fn (ptr: *anyopaque, gpa: std.mem.Allocator) Error!Device,
    };

    pub fn isAvailable(self: Driver) bool {
        return self.vtable.isAvailable(self.ptr);
    }
    pub fn createDevice(self: Driver, gpa: std.mem.Allocator) Error!Device {
        return self.vtable.createDevice(self.ptr, gpa);
    }
};

fn testAvailable(ptr: *anyopaque) bool {
    _ = ptr;
    return true;
}
fn testCreateDevice(ptr: *anyopaque, gpa: std.mem.Allocator) Error!Device {
    _ = ptr;
    _ = gpa;
    return error.NotImplemented;
}

test "driver vtable dispatches name, availability, createDevice" {
    var dummy: u8 = 0;
    const vt = Driver.VTable{ .isAvailable = &testAvailable, .createDevice = &testCreateDevice };
    const d = Driver{ .name = "test", .ptr = &dummy, .vtable = &vt };
    try std.testing.expectEqualStrings("test", d.name);
    try std.testing.expect(d.isAvailable());
    try std.testing.expectError(error.NotImplemented, d.createDevice(std.testing.allocator));
}
