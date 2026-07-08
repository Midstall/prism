//! wl_egl_window ABI and app-wl_display libwayland interop for Prism's EGL.
//! EGL equivalent of wsi_wayland.zig: parses wl_egl_window, presents via wl_shm
//! on the app's own wl_display. libwayland-client binds to the app's copy at runtime.

const std = @import("std");
const builtin = @import("builtin");
const wlz = @import("wayland");

// wl_egl_window ABI (wayland-egl-backend.h). First member is `const intptr_t version`.
// WL_EGL_WINDOW_VERSION >= 3 (every libwayland-egl since 2014) layout:
//
//   struct wl_egl_window {
//       const intptr_t version;        // offset 0
//       int width;                     // window size the app set
//       int height;
//       int dx;
//       int dy;
//       int attached_width;
//       int attached_height;
//       void *driver_private;
//       void (*resize_callback)(struct wl_egl_window *, void *);
//       void (*destroy_window_callback)(void *);
//       struct wl_surface *surface;    // last member (v>=3)
//   };
//
// Pre-v3 libwayland-egl put `surface` first. If `version` is a plausible small
// number (<= 0xffff) we trust the modern layout. Otherwise `version` is the
// wl_surface* (old layout). We model v3 and keep the legacy fallback.

pub const wl_surface = anyopaque;

/// The modern (version >= 3) `wl_egl_window` memory layout. `extern struct` so
/// the field offsets match the C ABI exactly (verified against
/// wayland-egl-backend.h: version@0, width@8, height@12, ..., surface last).
pub const WlEglWindowV3 = extern struct {
    version: isize,
    width: c_int,
    height: c_int,
    dx: c_int,
    dy: c_int,
    attached_width: c_int,
    attached_height: c_int,
    driver_private: ?*anyopaque,
    resize_callback: ?*const anyopaque,
    destroy_window_callback: ?*const anyopaque,
    surface: ?*wl_surface,
};

/// A parsed view of a `wl_egl_window *`: the app's wl_surface + the current
/// window size. Read fresh on each swap so a `wl_egl_window_resize` is honored.
pub const ParsedWindow = struct {
    surface: *wl_surface,
    width: u32,
    height: u32,
};

/// The largest version number we treat as a "real" version field. libwayland-egl
/// has never exceeded 3. Anything wildly larger means `version` is actually a
/// pointer (the legacy surface-first layout) and we fall back.
const MAX_PLAUSIBLE_VERSION: isize = 0xffff;

/// Parse a raw `wl_egl_window *` into the app's wl_surface + window size. Returns
/// null for a null or surface-less struct (maps to EGL_BAD_NATIVE_WINDOW). First
/// word <= 0xffff means the v3 layout. Otherwise the first word is the wl_surface*
/// (pre-v3 layout).
pub fn parse(raw: *anyopaque) ?ParsedWindow {
    const ver_ptr: *const isize = @ptrCast(@alignCast(raw));
    const ver = ver_ptr.*;
    if (ver >= 1 and ver <= MAX_PLAUSIBLE_VERSION) {
        // Modern versioned layout (v3 fields).
        const win: *const WlEglWindowV3 = @ptrCast(@alignCast(raw));
        const surf = win.surface orelse return null;
        if (win.width <= 0 or win.height <= 0) return null;
        return .{
            .surface = surf,
            .width = @intCast(win.width),
            .height = @intCast(win.height),
        };
    }
    // Legacy layout: { struct wl_surface *surface; int width; int height; ... }.
    // The first word is the surface pointer. Width/height follow it. We only
    // reach here when `version` looks like a pointer, which is the old ABI.
    const LegacyWindow = extern struct {
        surface: ?*wl_surface,
        width: c_int,
        height: c_int,
    };
    const win: *const LegacyWindow = @ptrCast(@alignCast(raw));
    const surf = win.surface orelse return null;
    if (win.width <= 0 or win.height <= 0) return null;
    return .{
        .surface = surf,
        .width = @intCast(win.width),
        .height = @intCast(win.height),
    };
}

// libwayland-client C API, resolved at runtime from the app's copy (build.zig
// linkSystemLibrary("wayland-client")). The dynamic linker binds these to the
// same copy the host app already loaded, so our proxies belong to the app's
// connection. We do not link libwayland-egl: wl_egl_window is a plain versioned
// struct read directly, adding no libwayland-egl dependency.

pub const wl_proxy = anyopaque;
pub const wl_display = anyopaque;

pub extern fn wl_proxy_marshal_flags(
    proxy: *wl_proxy,
    opcode: u32,
    interface: ?*const anyopaque,
    version: u32,
    flags: u32,
    ...,
) callconv(.c) ?*wl_proxy;
pub extern fn wl_proxy_get_version(proxy: *wl_proxy) callconv(.c) u32;
pub extern fn wl_proxy_add_listener(
    proxy: *wl_proxy,
    implementation: [*]const ?*const anyopaque,
    data: ?*anyopaque,
) callconv(.c) c_int;
pub extern fn wl_proxy_destroy(proxy: *wl_proxy) callconv(.c) void;
pub extern fn wl_display_roundtrip(display: *wl_display) callconv(.c) c_int;
pub extern fn wl_display_dispatch_pending(display: *wl_display) callconv(.c) c_int;
pub extern fn wl_display_dispatch(display: *wl_display) callconv(.c) c_int;
pub extern fn wl_display_flush(display: *wl_display) callconv(.c) c_int;

pub extern const wl_registry_interface: anyopaque;
pub extern const wl_shm_interface: anyopaque;
pub extern const wl_shm_pool_interface: anyopaque;
pub extern const wl_buffer_interface: anyopaque;

const WL_DISPLAY_GET_REGISTRY: u32 = 1;
const WL_REGISTRY_BIND: u32 = 0;
const WL_SHM_CREATE_POOL: u32 = 0;
const WL_SHM_POOL_CREATE_BUFFER: u32 = 0;
const WL_SHM_POOL_DESTROY: u32 = 1;
const WL_SURFACE_ATTACH: u32 = 1;
const WL_SURFACE_DAMAGE_BUFFER: u32 = 9;
const WL_SURFACE_COMMIT: u32 = 6;
const WL_MARSHAL_FLAG_DESTROY: u32 = 1;
const WL_SHM_FORMAT_XRGB8888: u32 = wlz.shm.FORMAT_XRGB8888;

const wl_registry_listener = extern struct {
    global: ?*const fn (data: ?*anyopaque, registry: *wl_proxy, name: u32, interface: [*:0]const u8, version: u32) callconv(.c) void,
    global_remove: ?*const fn (data: ?*anyopaque, registry: *wl_proxy, name: u32) callconv(.c) void,
};

const wl_buffer_listener = extern struct {
    release: ?*const fn (data: ?*anyopaque, buffer: *wl_proxy) callconv(.c) void,
};

const RegistryProbe = struct {
    shm_name: u32 = 0,
    shm_version: u32 = 0,
    found_shm: bool = false,
};

fn onGlobal(data: ?*anyopaque, registry: *wl_proxy, name: u32, interface: [*:0]const u8, version: u32) callconv(.c) void {
    _ = registry;
    const probe: *RegistryProbe = @ptrCast(@alignCast(data.?));
    const iface = std.mem.span(interface);
    if (std.mem.eql(u8, iface, "wl_shm")) {
        probe.shm_name = name;
        probe.shm_version = version;
        probe.found_shm = true;
    }
}

fn onGlobalRemove(data: ?*anyopaque, registry: *wl_proxy, name: u32) callconv(.c) void {
    _ = data;
    _ = registry;
    _ = name;
}

const registry_impl = wl_registry_listener{
    .global = onGlobal,
    .global_remove = onGlobalRemove,
};

fn onBufferRelease(data: ?*anyopaque, buffer: *wl_proxy) callconv(.c) void {
    _ = buffer;
    const b: *ShmBuffer = @ptrCast(@alignCast(data.?));
    b.free = true;
}

// --- A wl_shm-backed present surface on the app's wl_display ----------------

pub const ShmBuffer = struct {
    wl_buffer: *wl_proxy,
    pixels: []u8,
    free: bool = true,
    stride: u32,
    width: u32,
    height: u32,
};

/// Per-EGL-window libwayland present state: the bound wl_shm on the app's display
/// + N shm buffers + the app's wl_surface. Created on the first present (when the
/// EGL display knows it is app-wl_display-bound) and reused/re-created on resize.
pub const WaylandPresent = struct {
    gpa: std.mem.Allocator,
    display: *wl_display,
    surface: *wl_surface,
    shm: *wl_proxy,
    pool: *wl_proxy,
    shm_pool: wlz.shm.ShmPool,
    buffers: []ShmBuffer,
    buffer_impls: []wl_buffer_listener,
    width: u32,
    height: u32,

    /// Bind wl_shm on the app's `display` and create `count` XRGB8888 buffers of
    /// `width`x`height` (stride = width*4) over one shm pool. Marshals through the
    /// app's libwayland-client so the proxies belong to the app's connection.
    pub fn init(
        gpa: std.mem.Allocator,
        display: *wl_display,
        surface: *wl_surface,
        width: u32,
        height: u32,
        count: u32,
    ) !*WaylandPresent {
        const reg = wl_proxy_marshal_flags(
            display,
            WL_DISPLAY_GET_REGISTRY,
            &wl_registry_interface,
            wl_proxy_get_version(display),
            0,
            @as(?*wl_proxy, null),
        ) orelse return error.RegistryFailed;
        defer wl_proxy_destroy(reg);

        var probe = RegistryProbe{};
        if (wl_proxy_add_listener(reg, @ptrCast(&registry_impl), &probe) != 0)
            return error.ListenerFailed;
        _ = wl_display_roundtrip(display);
        if (!probe.found_shm) return error.NoWlShm;

        const bind_ver: u32 = @min(probe.shm_version, 1);
        const shm = wl_proxy_marshal_flags(
            reg,
            WL_REGISTRY_BIND,
            &wl_shm_interface,
            bind_ver,
            0,
            probe.shm_name,
            @as([*:0]const u8, "wl_shm"),
            bind_ver,
            @as(?*wl_proxy, null),
        ) orelse return error.BindFailed;
        errdefer wl_proxy_destroy(shm);

        const stride: u32 = width * 4;
        const img_bytes: usize = @as(usize, stride) * @as(usize, height);
        const total: usize = img_bytes * @as(usize, count);

        var shm_pool = try wlz.shm.ShmPool.create(total);
        errdefer shm_pool.deinit();
        const fd = shm_pool.fd;
        const region: []u8 = shm_pool.data;

        const pool = wl_proxy_marshal_flags(
            shm,
            WL_SHM_CREATE_POOL,
            &wl_shm_pool_interface,
            wl_proxy_get_version(shm),
            0,
            @as(?*wl_proxy, null),
            @as(i32, @intCast(fd)),
            @as(i32, @intCast(total)),
        ) orelse return error.PoolFailed;
        errdefer _ = wl_proxy_marshal_flags(pool, WL_SHM_POOL_DESTROY, null, wl_proxy_get_version(pool), WL_MARSHAL_FLAG_DESTROY);

        const buffers = try gpa.alloc(ShmBuffer, count);
        errdefer gpa.free(buffers);
        const buffer_impls = try gpa.alloc(wl_buffer_listener, count);
        errdefer gpa.free(buffer_impls);

        const self = try gpa.create(WaylandPresent);
        errdefer gpa.destroy(self);

        var made: u32 = 0;
        errdefer {
            var k: u32 = 0;
            while (k < made) : (k += 1) wl_proxy_destroy(buffers[k].wl_buffer);
        }
        while (made < count) : (made += 1) {
            const offset: i32 = @intCast(img_bytes * @as(usize, made));
            const wbuf = wl_proxy_marshal_flags(
                pool,
                WL_SHM_POOL_CREATE_BUFFER,
                &wl_buffer_interface,
                wl_proxy_get_version(pool),
                0,
                @as(?*wl_proxy, null),
                offset,
                @as(i32, @intCast(width)),
                @as(i32, @intCast(height)),
                @as(i32, @intCast(stride)),
                WL_SHM_FORMAT_XRGB8888,
            ) orelse return error.BufferFailed;
            const base = img_bytes * @as(usize, made);
            buffers[made] = .{
                .wl_buffer = wbuf,
                .pixels = region[base .. base + img_bytes],
                .free = true,
                .stride = stride,
                .width = width,
                .height = height,
            };
            buffer_impls[made] = .{ .release = onBufferRelease };
            _ = wl_proxy_add_listener(wbuf, @ptrCast(&buffer_impls[made]), &buffers[made]);
        }

        self.* = .{
            .gpa = gpa,
            .display = display,
            .surface = surface,
            .shm = shm,
            .pool = pool,
            .shm_pool = shm_pool,
            .buffers = buffers,
            .buffer_impls = buffer_impls,
            .width = width,
            .height = height,
        };
        return self;
    }

    pub fn deinit(self: *WaylandPresent) void {
        for (self.buffers) |b| wl_proxy_destroy(b.wl_buffer);
        _ = wl_proxy_marshal_flags(self.pool, WL_SHM_POOL_DESTROY, null, wl_proxy_get_version(self.pool), WL_MARSHAL_FLAG_DESTROY);
        wl_proxy_destroy(self.shm);
        self.shm_pool.deinit();
        self.gpa.free(self.buffer_impls);
        self.gpa.free(self.buffers);
        self.gpa.destroy(self);
    }

    /// Find a free buffer index for the next frame. If all buffers are in-flight
    /// (compositor still holds them), wait for a wl_buffer.release event rather than
    /// overwriting a buffer the compositor is still scanning out. Writing into an
    /// in-flight buffer causes tearing. Bounded so a wedged compositor falls back to
    /// buffer reuse instead of deadlocking.
    pub fn acquire(self: *WaylandPresent) u32 {
        // Drain already-arrived releases first (non-blocking).
        _ = wl_display_dispatch_pending(self.display);
        for (self.buffers, 0..) |b, i| if (b.free) return @intCast(i);
        // None free: flush so the compositor can release a buffer, then block on the
        // display fd for a release event. wl_display_dispatch waits for at least one
        // event, throttling to the compositor's pace instead of busy-spinning.
        var tries: usize = 0;
        while (tries < 200) : (tries += 1) {
            _ = wl_display_flush(self.display);
            if (wl_display_dispatch(self.display) < 0) break; // connection error
            for (self.buffers, 0..) |b, i| if (b.free) return @intCast(i);
        }
        return 0; // wedged compositor: reuse rather than hang forever
    }

    /// Present buffer `index`: mark it owned, attach + damage + commit + flush on
    /// the app's surface (its own event loop drives the rest).
    pub fn present(self: *WaylandPresent, index: u32) void {
        const b = &self.buffers[index];
        b.free = false;
        _ = wl_proxy_marshal_flags(
            self.surface,
            WL_SURFACE_ATTACH,
            null,
            wl_proxy_get_version(self.surface),
            0,
            b.wl_buffer,
            @as(i32, 0),
            @as(i32, 0),
        );
        _ = wl_proxy_marshal_flags(
            self.surface,
            WL_SURFACE_DAMAGE_BUFFER,
            null,
            wl_proxy_get_version(self.surface),
            0,
            @as(i32, 0),
            @as(i32, 0),
            @as(i32, @intCast(self.width)),
            @as(i32, @intCast(self.height)),
        );
        _ = wl_proxy_marshal_flags(
            self.surface,
            WL_SURFACE_COMMIT,
            null,
            wl_proxy_get_version(self.surface),
            0,
        );
        _ = wl_display_flush(self.display);
    }
};

/// Convert a rendered RGBA8 (R,G,B,A byte order) image into the shm buffer's
/// XRGB8888 little-endian layout (byte order B,G,R,X). Identical to the ICD's
/// blit (kept here so the EGL .so does not import the whole ICD).
pub fn blitRgbaToXrgb(dst: []u8, src: []const u8, width: u32, height: u32) void {
    const px = @as(usize, width) * @as(usize, height);
    const n = @min(@min(dst.len, src.len) / 4, px);
    var i: usize = 0;
    while (i < n) : (i += 1) {
        const s = i * 4;
        const r = src[s + 0];
        const g = src[s + 1];
        const b = src[s + 2];
        dst[s + 0] = b;
        dst[s + 1] = g;
        dst[s + 2] = r;
        dst[s + 3] = 0xff;
    }
}

// Tests (the struct parse + the blit are the live-compositor-free oracles).

test "wl_egl_window v3 layout offsets match wayland-egl-backend.h" {
    // version@0, width@8 (after the isize), surface as the LAST member.
    try std.testing.expectEqual(@as(usize, 0), @offsetOf(WlEglWindowV3, "version"));
    try std.testing.expectEqual(@sizeOf(isize), @offsetOf(WlEglWindowV3, "width"));
    try std.testing.expectEqual(@offsetOf(WlEglWindowV3, "width") + 4, @offsetOf(WlEglWindowV3, "height"));
    // surface follows three pointers after attached_height (driver_private,
    // resize_callback, destroy_window_callback).
    try std.testing.expect(@offsetOf(WlEglWindowV3, "surface") > @offsetOf(WlEglWindowV3, "attached_height"));
}

test "parse reads the wl_surface + size from a v3 wl_egl_window" {
    var fake_surface: u8 = 0;
    var win = WlEglWindowV3{
        .version = 3,
        .width = 640,
        .height = 480,
        .dx = 0,
        .dy = 0,
        .attached_width = 0,
        .attached_height = 0,
        .driver_private = null,
        .resize_callback = null,
        .destroy_window_callback = null,
        .surface = @ptrCast(&fake_surface),
    };
    const p = parse(@ptrCast(&win)).?;
    try std.testing.expectEqual(@as(u32, 640), p.width);
    try std.testing.expectEqual(@as(u32, 480), p.height);
    try std.testing.expectEqual(@as(*wl_surface, @ptrCast(&fake_surface)), p.surface);
}

test "parse honors a resize (reads width/height fresh each call)" {
    var fake_surface: u8 = 0;
    var win = WlEglWindowV3{
        .version = 3,
        .width = 100,
        .height = 100,
        .dx = 0,
        .dy = 0,
        .attached_width = 0,
        .attached_height = 0,
        .driver_private = null,
        .resize_callback = null,
        .destroy_window_callback = null,
        .surface = @ptrCast(&fake_surface),
    };
    try std.testing.expectEqual(@as(u32, 100), parse(@ptrCast(&win)).?.width);
    // The app calls wl_egl_window_resize, which mutates width/height in place.
    win.width = 1280;
    win.height = 720;
    const p = parse(@ptrCast(&win)).?;
    try std.testing.expectEqual(@as(u32, 1280), p.width);
    try std.testing.expectEqual(@as(u32, 720), p.height);
}

test "parse rejects a zero-size or surface-less window" {
    var win = WlEglWindowV3{
        .version = 3,
        .width = 0,
        .height = 0,
        .dx = 0,
        .dy = 0,
        .attached_width = 0,
        .attached_height = 0,
        .driver_private = null,
        .resize_callback = null,
        .destroy_window_callback = null,
        .surface = null,
    };
    try std.testing.expectEqual(@as(?ParsedWindow, null), parse(@ptrCast(&win)));
}

test "parse handles the legacy surface-first layout" {
    // Old libwayland-egl: { wl_surface *surface; int width; int height; }. The
    // first word is a pointer (huge as an isize), so we fall back to legacy.
    const Legacy = extern struct {
        surface: ?*anyopaque,
        width: c_int,
        height: c_int,
    };
    var fake_surface: u8 = 0;
    var win = Legacy{ .surface = @ptrCast(&fake_surface), .width = 320, .height = 240 };
    const p = parse(@ptrCast(&win)).?;
    try std.testing.expectEqual(@as(u32, 320), p.width);
    try std.testing.expectEqual(@as(u32, 240), p.height);
    try std.testing.expectEqual(@as(*wl_surface, @ptrCast(&fake_surface)), p.surface);
}

test "blitRgbaToXrgb swaps R and B channels" {
    var src = [_]u8{ 10, 20, 30, 40, 50, 60, 70, 80 };
    var dst = [_]u8{0} ** 8;
    blitRgbaToXrgb(&dst, &src, 2, 1);
    try std.testing.expectEqual(@as(u8, 30), dst[0]);
    try std.testing.expectEqual(@as(u8, 20), dst[1]);
    try std.testing.expectEqual(@as(u8, 10), dst[2]);
    try std.testing.expectEqual(@as(u8, 0xff), dst[3]);
}
