const std = @import("std");
const hal = @import("../../hal.zig");

pub const VertexIR = extern struct {
    magic: u32 = MAGIC,
    position_attr: u32,
    color_attr: u32,
    pub const MAGIC: u32 = 0x50525356; // "PRSV"
};

pub const FragmentIR = extern struct {
    magic: u32 = MAGIC,
    pub const MAGIC: u32 = 0x50525346; // "PRSF"
};

pub fn encodeVertex(ir: VertexIR) [@sizeOf(VertexIR)]u8 {
    return std.mem.toBytes(ir);
}
pub fn encodeFragment(ir: FragmentIR) [@sizeOf(FragmentIR)]u8 {
    return std.mem.toBytes(ir);
}

pub const ShaderModule = struct {
    stage: hal.ShaderStage,
    vertex: ?VertexIR = null,
    fragment: ?FragmentIR = null,
    /// The raw SPIR-V byte stream for a real-shader module (compute, or a vertex/
    /// fragment shader to be JITed and executed), owned by `gpa` (a copy of the
    /// caller's `code`, which is not guaranteed to outlive the module). Lowered +
    /// JITed lazily (at dispatch time for compute, at draw time for graphics).
    compute_spirv: ?[]u8 = null,
    /// The allocator that owns `compute_spirv` (so the module can free it).
    gpa: ?std.mem.Allocator = null,

    /// Decode a module. A `.vertex`/`.fragment` module accepts either the declarative
    /// Prism shader IR (the legacy passthrough magic, used by the encode/decode tests)
    /// or real SPIR-V (the Vulkan ICD path): a SPIR-V word stream that is not the
    /// declarative magic is retained as `compute_spirv` and JITed + executed at draw
    /// time. A `.compute` module always retains its SPIR-V. A copy is taken because
    /// `code` is caller-owned and may be freed after this returns. Pass a non-null
    /// `gpa` to retain SPIR-V (it owns the copy).
    pub fn decode(gpa: ?std.mem.Allocator, stage: hal.ShaderStage, code: []const u8) hal.Error!ShaderModule {
        switch (stage) {
            .vertex => {
                // Declarative passthrough IR (exact size + magic) takes the legacy path.
                if (code.len >= @sizeOf(VertexIR)) {
                    const ir = std.mem.bytesToValue(VertexIR, code[0..@sizeOf(VertexIR)]);
                    if (ir.magic == VertexIR.MAGIC) return .{ .stage = stage, .vertex = ir };
                }
                return retainSpirv(gpa, stage, code);
            },
            .fragment => {
                if (code.len >= @sizeOf(FragmentIR)) {
                    const ir = std.mem.bytesToValue(FragmentIR, code[0..@sizeOf(FragmentIR)]);
                    if (ir.magic == FragmentIR.MAGIC) return .{ .stage = stage, .fragment = ir };
                }
                return retainSpirv(gpa, stage, code);
            },
            // A SPIR-V compute kernel: retain the bytes. dispatchCompute lowers them
            // (SPIR-V -> Vulcan IR -> AArch64 JIT) and runs the kernel over the grid.
            .compute => return retainSpirv(gpa, stage, code),
        }
    }

    /// Retain a copy of a SPIR-V word stream for a real-shader module.
    fn retainSpirv(gpa: ?std.mem.Allocator, stage: hal.ShaderStage, code: []const u8) hal.Error!ShaderModule {
        const a = gpa orelse return error.InvalidArgument;
        if (code.len == 0 or code.len % 4 != 0) return error.InvalidArgument;
        const copy = a.alloc(u8, code.len) catch return error.OutOfMemory;
        @memcpy(copy, code);
        return .{ .stage = stage, .compute_spirv = copy, .gpa = a };
    }

    /// Release a module's retained SPIR-V (no-op for declarative graphics modules).
    pub fn deinit(self: *ShaderModule) void {
        if (self.compute_spirv) |s| {
            if (self.gpa) |a| a.free(s);
            self.compute_spirv = null;
        }
    }
};

test "vertex IR round-trips through encode/decode" {
    const bytes = encodeVertex(.{ .position_attr = 0, .color_attr = 1 });
    var m = try ShaderModule.decode(null, .vertex, &bytes);
    defer m.deinit();
    try std.testing.expectEqual(@as(u32, 0), m.vertex.?.position_attr);
    try std.testing.expectEqual(@as(u32, 1), m.vertex.?.color_attr);
}

test "bad magic is rejected" {
    var bytes = encodeVertex(.{ .position_attr = 0, .color_attr = 1 });
    bytes[0] = 0xFF;
    try std.testing.expectError(error.InvalidArgument, ShaderModule.decode(null, .vertex, &bytes));
}

test "compute module retains its SPIR-V bytes" {
    const gpa = std.testing.allocator;
    // A 4-word (16-byte) dummy SPIR-V-shaped blob. Decode only checks the length.
    const code = [_]u8{ 0x03, 0x02, 0x23, 0x07 } ** 4;
    var m = try ShaderModule.decode(gpa, .compute, &code);
    defer m.deinit();
    try std.testing.expect(m.compute_spirv != null);
    try std.testing.expectEqualSlices(u8, &code, m.compute_spirv.?);
    // A non-word-multiple length is rejected.
    try std.testing.expectError(error.InvalidArgument, ShaderModule.decode(gpa, .compute, "abc"));
}
