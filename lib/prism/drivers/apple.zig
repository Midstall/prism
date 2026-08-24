const std = @import("std");
const drv = @import("../driver.zig");
const asahi = @import("asahi");
const Driver = drv.Driver;
const Device = drv.Device;
const Error = drv.Error;
const AppleDevice = @import("apple/device.zig").Device;

const State = struct {};
var state: State = .{};

fn available(ptr: *anyopaque) bool {
    _ = ptr;
    // The asahi DRM driver exists only on Linux. Probing it from a darwin build
    // runs Linux syscall numbers on the macOS kernel and the process dies with
    // SIGSYS. drivers.zig gates this driver out off Linux, and this guard keeps
    // an explicit -Ddrivers=apple build on darwin from crashing the same way.
    if (@import("builtin").os.tag != .linux) return false;
    // Probe for a usable Apple Silicon (AGX) GPU: open the kernel `asahi` DRM
    // render node. If that succeeds, the from-scratch driver can drive it. On a
    // box without an AGX GPU (the aarch64 dev box) open() returns NoDevice and
    // this reports absent, so driver auto-selection falls through cleanly.
    var dev = asahi.Device.open() catch return false;
    dev.deinit();
    return true;
}
fn createDevice(ptr: *anyopaque, gpa: std.mem.Allocator) Error!Device {
    _ = ptr;
    return AppleDevice.create(gpa);
}

const vtable = Driver.VTable{ .isAvailable = &available, .createDevice = &createDevice };

/// drm_driver = "asahi": the kernel DRM driver name, so the compositor's
/// preferred-render-device match (drivers.selectForDrmDevice / platform drm) maps
/// an asahi render node to this driver.
pub const driver = Driver{ .name = "apple", .ptr = &state, .vtable = &vtable, .drm_driver = "asahi" };

// Pull the device/context/surface tests into the prism test run.
test {
    _ = @import("apple/device.zig");
}

test "apple driver exposes name + asahi DRM association via vtable" {
    try std.testing.expectEqualStrings("apple", driver.name);
    try std.testing.expectEqualStrings("asahi", driver.drm_driver.?);
}

test "apple driver: availability + device creation (skips without an AGX GPU)" {
    // Exercise the full path when an AGX GPU is present. Otherwise confirm it
    // reports absent (the no-AGX dev box path: createBestDevice falls through
    // cleanly, exactly like nvidia on a non-NVIDIA box). Does not require a GPU.
    if (driver.isAvailable()) {
        const dev = try driver.createDevice(std.testing.allocator);
        dev.deinit();
    } else {
        // On this no-AGX box availability is false and createDevice would fail to
        // open the node (the documented dev-box state).
        try std.testing.expect(!driver.isAvailable());
    }
}
