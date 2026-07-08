const std = @import("std");

/// The Prism-wide error set. Used internally everywhere as `Error!T`.
/// Frontends translate it to VkResult / EGLint / GL error at the C-ABI boundary.
pub const Error = error{
    OutOfMemory,
    NotImplemented,
    Unsupported,
    InvalidArgument,
    DeviceLost,
    InitializationFailed,
};

test "error set contains the expected members" {
    const e: Error = error.NotImplemented;
    try std.testing.expect(e == Error.NotImplemented);
}
