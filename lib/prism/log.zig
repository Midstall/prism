const std = @import("std");

/// A per-instance log sink: `ctx` is opaque consumer state, `message` is the
/// already-formatted line. The consumer owns where it goes (stderr, a ring
/// buffer, a test capture, ...). Prism never logs unless a consumer wires one in.
pub const LogFn = *const fn (ctx: ?*anyopaque, message: []const u8) void;

/// A minimal per-instance logger. Default-constructed (`.{}`) it is silent: any
/// `log` call is a no-op. A consumer opts in by setting `func` (and optionally
/// `ctx`) on the thing's `log` field, e.g. `display.log = .{ .ctx = x, .func = f };`.
pub const Logger = struct {
    ctx: ?*anyopaque = null,
    func: ?LogFn = null,

    pub fn log(self: Logger, comptime fmt: []const u8, args: anytype) void {
        const f = self.func orelse return; // unset -> silent
        var buf: [256]u8 = undefined;
        const msg = std.fmt.bufPrint(&buf, fmt, args) catch return;
        f(self.ctx, msg);
    }
};

const Capture = struct {
    var buf: [256]u8 = undefined;
    var len: usize = 0;
    fn sink(ctx: ?*anyopaque, message: []const u8) void {
        _ = ctx;
        @memcpy(buf[0..message.len], message);
        len = message.len;
    }
    fn slice() []const u8 {
        return buf[0..len];
    }
};

test "Logger with a func receives the formatted message" {
    Capture.len = 0;
    const logger = Logger{ .func = &Capture.sink };
    logger.log("value={d} name={s}", .{ 42, "prism" });
    try std.testing.expectEqualStrings("value=42 name=prism", Capture.slice());
}

test "Logger with func=null is a silent no-op" {
    const logger = Logger{};
    // Must not crash and must do nothing.
    logger.log("this {s} go anywhere", .{"does not"});
}
