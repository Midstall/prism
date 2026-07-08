const std = @import("std");
const hal = @import("../hal.zig");
const platform = @import("../platform.zig");

const HeadlessSurface = struct {
    gpa: std.mem.Allocator,
    bytes: []u8,
    width: u32,
    height: u32,
    stride: u32,
    format: hal.Format,
    committed: bool,

    fn currentBuffer(ptr: *anyopaque) hal.Error!platform.Buffer {
        const self: *HeadlessSurface = @ptrCast(@alignCast(ptr));
        return platform.Buffer{
            .bytes = self.bytes,
            .width = self.width,
            .height = self.height,
            .stride = self.stride,
            .format = self.format,
        };
    }
    fn commit(ptr: *anyopaque) hal.Error!void {
        const self: *HeadlessSurface = @ptrCast(@alignCast(ptr));
        self.committed = true;
    }
    fn processEvents(ptr: *anyopaque) hal.Error!platform.WindowEvent {
        _ = ptr;
        // Headless has no windowing-system event source and is never resized or closed.
        return .none;
    }
    fn size(ptr: *anyopaque) [2]u32 {
        const self: *HeadlessSurface = @ptrCast(@alignCast(ptr));
        return .{ self.width, self.height };
    }
    fn deinit(ptr: *anyopaque) void {
        const self: *HeadlessSurface = @ptrCast(@alignCast(ptr));
        self.gpa.free(self.bytes);
        self.gpa.destroy(self);
    }

    const vtable = platform.Surface.VTable{
        .currentBuffer = &currentBuffer,
        .commit = &commit,
        .processEvents = &processEvents,
        .size = &size,
        .deinit = &deinit,
    };
};

const HeadlessDisplay = struct {
    gpa: std.mem.Allocator,

    fn createSurface(ptr: *anyopaque, desc: platform.SurfaceDesc) hal.Error!platform.Surface {
        const self: *HeadlessDisplay = @ptrCast(@alignCast(ptr));
        const stride = desc.width * desc.format.bytesPerPixel();
        const byte_count = @as(usize, stride) * desc.height;
        const bytes = self.gpa.alloc(u8, byte_count) catch return error.OutOfMemory;
        @memset(bytes, 0);
        const s = self.gpa.create(HeadlessSurface) catch {
            self.gpa.free(bytes);
            return error.OutOfMemory;
        };
        s.* = .{
            .gpa = self.gpa,
            .bytes = bytes,
            .width = desc.width,
            .height = desc.height,
            .stride = stride,
            .format = desc.format,
            .committed = false,
        };
        return platform.Surface{ .ptr = s, .vtable = &HeadlessSurface.vtable };
    }
    fn deinit(ptr: *anyopaque) void {
        const self: *HeadlessDisplay = @ptrCast(@alignCast(ptr));
        self.gpa.destroy(self);
    }

    const vtable = platform.Display.VTable{
        .createSurface = &createSurface,
        .deinit = &deinit,
    };
};

pub fn create(gpa: std.mem.Allocator) hal.Error!platform.Display {
    const d = gpa.create(HeadlessDisplay) catch return error.OutOfMemory;
    d.* = .{ .gpa = gpa };
    return platform.Display{ .ptr = d, .vtable = &HeadlessDisplay.vtable };
}

test "headless display creates surface and persists pixels" {
    const gpa = std.testing.allocator;
    const display = try create(gpa);
    defer display.deinit();

    var surf = try display.createSurface(.{ .width = 8, .height = 8 });
    defer surf.deinit();

    const buf = try surf.currentBuffer();
    try std.testing.expectEqual(@as(u32, 8), buf.width);
    try std.testing.expectEqual(@as(u32, 8), buf.height);
    try std.testing.expectEqual(@as(usize, 8 * 8 * 4), buf.bytes.len);

    buf.bytes[0] = 42;
    buf.bytes[1] = 99;

    try surf.commit();

    const buf2 = try surf.currentBuffer();
    try std.testing.expectEqual(@as(u8, 42), buf2.bytes[0]);
    try std.testing.expectEqual(@as(u8, 99), buf2.bytes[1]);
}
