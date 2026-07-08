const std = @import("std");
const drv = @import("../driver.zig");
const Driver = drv.Driver;
const Device = drv.Device;
const Error = drv.Error;
const SwDevice = @import("software/device.zig").Device;

const State = struct {};
var state: State = .{};

fn available(ptr: *anyopaque) bool {
    _ = ptr;
    return true;
}
fn createDevice(ptr: *anyopaque, gpa: std.mem.Allocator) Error!Device {
    _ = ptr;
    return SwDevice.create(gpa);
}

const vtable = Driver.VTable{ .isAvailable = &available, .createDevice = &createDevice };

pub const driver = Driver{ .name = "software", .ptr = &state, .vtable = &vtable };

pub const shader = @import("software/shader.zig");
pub const pipeline = @import("software/pipeline.zig");

test {
    _ = @import("software/spirv_jit.zig");
    _ = @import("software/sampler.zig");
    _ = @import("software/raster.zig");
    _ = @import("software/context.zig");
}

test "software driver creates a real device" {
    const gpa = std.testing.allocator;
    const dev = try driver.createDevice(gpa);
    defer dev.deinit();
    const buf = try dev.createResource(.{ .buffer = .{ .size = 64 } });
    defer dev.destroyResource(buf);
    try std.testing.expectEqual(@as(usize, 64), (try dev.mapResource(buf)).len);
}
