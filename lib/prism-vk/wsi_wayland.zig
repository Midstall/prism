//! libwayland-client WSI for Prism's Vulkan ICD. Presents through the host app's
//! wl_display + wl_surface via wl_shm buffers (RGBA8 to XRGB8888). Binds to the
//! app's libwayland-client.so at runtime, matching Mesa's wsi_common_wayland.

const std = @import("std");
/// Prism's Zig Wayland library. Used for `shm.ShmPool` (memfd + mmap) and format
/// consts for the swapchain's CPU buffers. Proxy marshaling goes through the app's
/// real libwayland to reach the app's wl_surface.
const wlz = @import("wayland");

// libwayland-client C API. Linked via pkg-config (build.zig). At runtime these
// bind to the host app's libwayland-client.so so the proxies belong to the app's
// connection, which is the only way to present to the app's real wl_surface.

pub const wl_proxy = anyopaque;
pub const wl_display = anyopaque;
pub const wl_surface = anyopaque;

// At runtime the dynamic linker resolves these against the libwayland-client.so
// already loaded by the host app (vkcube etc.), so the ICD drives the same
// connection and proxies the app made.

/// Variadic marshal entry point (libwayland >= 1.20). For requests that create
/// an object, pass the new object's `wl_interface*` as `interface`, the version,
/// and a NULL placeholder at the new-id position. The returned proxy is the new object.
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
pub extern fn wl_display_flush(display: *wl_display) callconv(.c) c_int;

// The DATA symbols (the wl_interface structs). Their address is what
// wl_proxy_marshal_flags needs to construct new proxies of the right type.
pub extern const wl_registry_interface: anyopaque;
pub extern const wl_shm_interface: anyopaque;
pub extern const wl_shm_pool_interface: anyopaque;
pub extern const wl_buffer_interface: anyopaque;
pub extern const wl_callback_interface: anyopaque;

// Request opcodes (from wayland.xml).
const WL_DISPLAY_GET_REGISTRY: u32 = 1;
const WL_REGISTRY_BIND: u32 = 0;
const WL_SHM_CREATE_POOL: u32 = 0;
const WL_SHM_POOL_CREATE_BUFFER: u32 = 0;
const WL_SHM_POOL_DESTROY: u32 = 1;
const WL_BUFFER_DESTROY: u32 = 0;
const WL_SURFACE_ATTACH: u32 = 1;
const WL_SURFACE_DAMAGE_BUFFER: u32 = 9;
const WL_SURFACE_COMMIT: u32 = 6;

const WL_MARSHAL_FLAG_DESTROY: u32 = 1;

// wl_shm pixel formats: reuse wayland.zig's consts (ARGB8888=0, XRGB8888=1).
const WL_SHM_FORMAT_XRGB8888: u32 = wlz.shm.FORMAT_XRGB8888;

// --- the registry listener: bind wl_shm -------------------------------------

const wl_registry_listener = extern struct {
    global: ?*const fn (data: ?*anyopaque, registry: *wl_proxy, name: u32, interface: [*:0]const u8, version: u32) callconv(.c) void,
    global_remove: ?*const fn (data: ?*anyopaque, registry: *wl_proxy, name: u32) callconv(.c) void,
};

const wl_buffer_listener = extern struct {
    release: ?*const fn (data: ?*anyopaque, buffer: *wl_proxy) callconv(.c) void,
};

/// State threaded through the registry-global callback to capture the wl_shm
/// global name + version.
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

// --- a single wl_shm-backed swapchain buffer --------------------------------

pub const ShmBuffer = struct {
    wl_buffer: *wl_proxy,
    /// The mmap'd CPU pixels (the wl_shm pool region for THIS image).
    pixels: []u8,
    /// Set false while the compositor owns the buffer, true once released.
    free: bool = true,
    stride: u32,
    width: u32,
    height: u32,
};

/// Per-swapchain libwayland WSI state: the bound wl_shm + the N shm buffers.
pub const WaylandWsi = struct {
    gpa: std.mem.Allocator,
    display: *wl_display,
    surface: *wl_surface,
    shm: *wl_proxy,
    pool: *wl_proxy,
    /// The memfd+mmap pool, from wayland.zig's shm helper.
    shm_pool: wlz.shm.ShmPool,
    buffers: []ShmBuffer,
    width: u32,
    height: u32,
    /// One listener instance per buffer (the data pointer is the ShmBuffer).
    buffer_impls: []wl_buffer_listener,

    /// Bind wl_shm on the app's display and create `count` wl_shm buffers of
    /// `width`x`height` XRGB8888 (stride = width*4).
    pub fn init(
        gpa: std.mem.Allocator,
        display: *wl_display,
        surface: *wl_surface,
        width: u32,
        height: u32,
        count: u32,
    ) !*WaylandWsi {
        // 1) Registry roundtrip to find + bind wl_shm.
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
        // The roundtrip drives the global advertisements into onGlobal.
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

        // 2) Allocate the shm pool: one contiguous region of count*stride*height.
        // The memfd + mmap is wayland.zig's ShmPool (the Zig wl_shm helper).
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

        // 3) One wl_buffer per image, each over its slice of the pool.
        const buffers = try gpa.alloc(ShmBuffer, count);
        errdefer gpa.free(buffers);
        const buffer_impls = try gpa.alloc(wl_buffer_listener, count);
        errdefer gpa.free(buffer_impls);

        const self = try gpa.create(WaylandWsi);
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
            .width = width,
            .height = height,
            .buffer_impls = buffer_impls,
        };
        return self;
    }

    pub fn deinit(self: *WaylandWsi) void {
        for (self.buffers) |b| wl_proxy_destroy(b.wl_buffer);
        _ = wl_proxy_marshal_flags(self.pool, WL_SHM_POOL_DESTROY, null, wl_proxy_get_version(self.pool), WL_MARSHAL_FLAG_DESTROY);
        wl_proxy_destroy(self.shm);
        self.shm_pool.deinit();
        self.gpa.free(self.buffer_impls);
        self.gpa.free(self.buffers);
        self.gpa.destroy(self);
    }

    /// Find a free buffer index, draining wl events (FIFO) until one frees.
    pub fn acquire(self: *WaylandWsi) u32 {
        var spins: usize = 0;
        while (true) : (spins += 1) {
            for (self.buffers, 0..) |b, i| {
                if (b.free) return @intCast(i);
            }
            // None free: flush + dispatch pending releases, then retry. Give up
            // after a bound and reuse index 0. Avoids a deadlock if the compositor
            // never releases. Present is best-effort.
            _ = wl_display_flush(self.display);
            _ = wl_display_dispatch_pending(self.display);
            if (spins > 1000) return 0;
        }
    }

    /// Present buffer `index`: mark it owned, attach + damage_buffer + commit +
    /// flush on the app's surface.
    pub fn present(self: *WaylandWsi, index: u32) void {
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

fn onBufferRelease(data: ?*anyopaque, buffer: *wl_proxy) callconv(.c) void {
    _ = buffer;
    const b: *ShmBuffer = @ptrCast(@alignCast(data.?));
    b.free = true;
}

/// Convert a rendered RGBA8 (R,G,B,A byte order) image into the shm buffer's
/// XRGB8888 little-endian layout (byte order B,G,R,X). `src` is the HAL image
/// bytes (width*height*4), `dst` is the wl_shm buffer pixels.
pub fn blitRgbaToXrgb(dst: []u8, src: []const u8, width: u32, height: u32) void {
    const px = @as(usize, width) * @as(usize, height);
    const n = @min(@min(dst.len, src.len) / 4, px);
    var i: usize = 0;
    while (i < n) : (i += 1) {
        const s = i * 4;
        const r = src[s + 0];
        const g = src[s + 1];
        const b = src[s + 2];
        // XRGB8888 little-endian in memory: byte0=B, byte1=G, byte2=R, byte3=X.
        dst[s + 0] = b;
        dst[s + 1] = g;
        dst[s + 2] = r;
        dst[s + 3] = 0xff;
    }
}

test "blitRgbaToXrgb swaps R and B channels" {
    var src = [_]u8{ 10, 20, 30, 40, 50, 60, 70, 80 };
    var dst = [_]u8{0} ** 8;
    blitRgbaToXrgb(&dst, &src, 2, 1);
    // pixel0: R=10 G=20 B=30 -> B,G,R,X
    try std.testing.expectEqual(@as(u8, 30), dst[0]);
    try std.testing.expectEqual(@as(u8, 20), dst[1]);
    try std.testing.expectEqual(@as(u8, 10), dst[2]);
    try std.testing.expectEqual(@as(u8, 0xff), dst[3]);
    // pixel1: R=50 G=60 B=70 -> B,G,R,X
    try std.testing.expectEqual(@as(u8, 70), dst[4]);
    try std.testing.expectEqual(@as(u8, 60), dst[5]);
    try std.testing.expectEqual(@as(u8, 50), dst[6]);
}
