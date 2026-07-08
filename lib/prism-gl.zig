const std = @import("std");
const prism = @import("prism");

/// GLVND GLX vendor entry point (stub for Milestone 1).
export fn __glx_Main(version: u32, exports: ?*anyopaque, vendor: ?*anyopaque, imports: ?*anyopaque) ?*anyopaque {
    _ = version;
    _ = exports;
    _ = vendor;
    _ = imports;
    return null;
}

test "core reachable from gl frontend" {
    try std.testing.expect(prism.version.len > 0);
}
