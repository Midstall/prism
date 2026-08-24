//! GBM present backend. Adapts an app-owned gbm.Surface to a platform.Surface so
//! the EGL GBM platform presents through the same HAL path as Wayland and DRM.
//! The app owns the gbm.Device and gbm.Surface and drives scanout itself.

const std = @import("std");
const gbm = @import("gbm");
const hal = @import("../hal.zig");
const platform = @import("../platform.zig");

pub const Device = gbm.Device;
pub const Surface = gbm.Surface;
pub const MemoryBackend = gbm.MemoryBackend;
pub const BufferDesc = gbm.BufferDesc;
pub const BufferUsage = gbm.BufferUsage;
pub const format = gbm.format;

fn halErr(e: gbm.backend.Error) hal.Error {
    return switch (e) {
        error.OutOfMemory => error.OutOfMemory,
        error.Unsupported, error.InvalidArgument => error.InvalidArgument,
        // A failed dmabuf export at present time is a runtime device failure,
        // the same class the Wayland platform reports as DeviceLost.
        error.ExportFailed => error.DeviceLost,
    };
}

/// The HAL pixel format matching a DRM fourcc's byte order. XRGB/ARGB store bytes
/// B,G,R,(X or A), so the software present swaps R and B when writing the HAL
/// rgba8 backbuffer into them. XBGR/ABGR are already in R,G,B order.
fn halFormat(fourcc: u32) ?hal.Format {
    return switch (fourcc) {
        gbm.format.DRM_FORMAT_XRGB8888, gbm.format.DRM_FORMAT_ARGB8888 => .bgra8_unorm,
        gbm.format.DRM_FORMAT_XBGR8888, gbm.format.DRM_FORMAT_ABGR8888 => .rgba8_unorm,
        else => null,
    };
}

const GbmSurface = struct {
    gpa: std.mem.Allocator,
    surf: *gbm.Surface,
    format: hal.Format,

    fn currentBuffer(ptr: *anyopaque) hal.Error!platform.Buffer {
        const self: *GbmSurface = @ptrCast(@alignCast(ptr));
        const bo = self.surf.nextBuffer() catch |e| return halErr(e);
        return .{
            .bytes = bo.data,
            .width = bo.width,
            .height = bo.height,
            .stride = bo.stride,
            .format = self.format,
        };
    }

    fn commit(ptr: *anyopaque) hal.Error!void {
        const self: *GbmSurface = @ptrCast(@alignCast(ptr));
        // eglSwapBuffers posts the rendered back buffer as the new front. The app
        // then scans it out with gbm_surface_lock_front_buffer + releaseBuffer.
        self.surf.swapBuffers();
    }

    fn processEvents(ptr: *anyopaque) hal.Error!platform.WindowEvent {
        _ = ptr;
        return .none;
    }

    fn size(ptr: *anyopaque) [2]u32 {
        const self: *GbmSurface = @ptrCast(@alignCast(ptr));
        return .{ self.surf.desc.width, self.surf.desc.height };
    }

    fn deinit(ptr: *anyopaque) void {
        const self: *GbmSurface = @ptrCast(@alignCast(ptr));
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

/// Wrap an app-owned gbm.Surface as a platform.Surface. The returned surface is
/// Prism-owned (free it with surface.deinit) and does not own the gbm.Surface.
/// Fails with InvalidArgument if the surface format has no HAL equivalent.
pub fn wrapSurface(gpa: std.mem.Allocator, surf: *gbm.Surface) hal.Error!platform.Surface {
    const fmt = halFormat(surf.desc.format) orelse return error.InvalidArgument;
    const s = gpa.create(GbmSurface) catch return error.OutOfMemory;
    s.* = .{ .gpa = gpa, .surf = surf, .format = fmt };
    return .{ .ptr = s, .vtable = &GbmSurface.vtable };
}

test "gbm platform surface aliases the gbm back buffer and honors the swap" {
    var mb = gbm.MemoryBackend.init(std.testing.allocator);
    defer mb.deinit();
    var dev = gbm.Device.init(mb.allocator());
    var surf = gbm.Surface.init(&dev, .{
        .width = 8,
        .height = 4,
        .format = gbm.format.DRM_FORMAT_XRGB8888,
        .usage = .{ .scanout = true },
    });
    defer surf.deinit();

    const ps = try wrapSurface(std.testing.allocator, &surf);
    defer ps.deinit();

    try std.testing.expectEqual([2]u32{ 8, 4 }, ps.size());
    const buf = try ps.currentBuffer();
    try std.testing.expectEqual(@as(u32, 8), buf.width);
    try std.testing.expectEqual(@as(u32, 4), buf.height);
    try std.testing.expectEqual(hal.Format.bgra8_unorm, buf.format);

    // The bytes alias the gbm back buffer, so a write is visible when the app
    // locks the front buffer (the standard gbm swap).
    @memset(buf.bytes, 0x7f);
    try ps.commit();
    const front = surf.lockFrontBuffer().?;
    try std.testing.expectEqual(@as(u8, 0x7f), front.data[0]);
}

test "gbm wrapSurface rejects a format with no HAL equivalent" {
    var mb = gbm.MemoryBackend.init(std.testing.allocator);
    defer mb.deinit();
    var dev = gbm.Device.init(mb.allocator());
    var surf = gbm.Surface.init(&dev, .{
        .width = 4,
        .height = 4,
        .format = gbm.format.DRM_FORMAT_RGB565,
        .usage = .{ .scanout = true },
    });
    defer surf.deinit();
    try std.testing.expectError(error.InvalidArgument, wrapSurface(std.testing.allocator, &surf));
}
