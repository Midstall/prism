const std = @import("std");
const hal = @import("../../hal.zig");

/// A HAL shader module on the Apple Silicon (AGX) path. The AGX driver has no
/// graphics-shader compiler, so only compute modules are real: a compute module
/// wraps the caller's hand-assembled AGX kernel bytes (the bytecode dispatchCompute
/// uploads to the kernel BO and launches). The bytes are copied into an owned
/// buffer so the module outlives the caller's `code` slice.
///
/// .vertex / .fragment modules are rejected at createShaderModule (NotImplemented).
/// The AGX graphics draw path is a documented hard open-item.
pub const ShaderModule = struct {
    stage: hal.ShaderStage,
    /// Owned copy of the AGX kernel bytecode (compute only).
    code: []u8,

    pub fn create(gpa: std.mem.Allocator, desc: hal.ShaderModuleDesc) hal.Error!*ShaderModule {
        if (desc.stage != .compute) return error.NotImplemented;
        const m = gpa.create(ShaderModule) catch return error.OutOfMemory;
        errdefer gpa.destroy(m);
        const owned = gpa.alloc(u8, desc.code.len) catch return error.OutOfMemory;
        @memcpy(owned, desc.code);
        m.* = .{ .stage = desc.stage, .code = owned };
        return m;
    }

    pub fn destroy(self: *ShaderModule, gpa: std.mem.Allocator) void {
        gpa.free(self.code);
        gpa.destroy(self);
    }
};

test "apple compute shader module wraps the AGX kernel bytes" {
    const gpa = std.testing.allocator;
    const bytes = [_]u8{ 0x62, 0x01, 0x0d, 0xf0, 0xfe, 0xca };
    const m = try ShaderModule.create(gpa, .{ .stage = .compute, .code = &bytes });
    defer m.destroy(gpa);
    try std.testing.expectEqual(hal.ShaderStage.compute, m.stage);
    try std.testing.expectEqualSlices(u8, &bytes, m.code);
}

test "apple rejects vertex/fragment shader modules" {
    const gpa = std.testing.allocator;
    const bytes = [_]u8{0} ** 8;
    try std.testing.expectError(error.NotImplemented, ShaderModule.create(gpa, .{ .stage = .vertex, .code = &bytes }));
    try std.testing.expectError(error.NotImplemented, ShaderModule.create(gpa, .{ .stage = .fragment, .code = &bytes }));
}
