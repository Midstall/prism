//! Wayland platform backend. Connects to a compositor via Unix socket, presents
//! via wl_shm/XRGB8888. Wayland XRGB8888 is B,G,R,X on little-endian. Reports
//! bgra8_unorm so the present blit writes correct bytes directly. No sockets at comptime.

const std = @import("std");
const hal = @import("../hal.zig");
const platform = @import("../platform.zig");
const prism_log = @import("../log.zig");
const wayland = @import("wayland");
const client = wayland.client;
const shm = wayland.shm;

/// Resolve the Wayland display socket path from the process environ map
/// (re-exported so callers only need the prism/platform module).
pub const resolveSocketPath = client.resolveSocketPath;

const WaylandSurface = struct {
    gpa: std.mem.Allocator,
    conn: *client.Connection,
    log: prism_log.Logger = .{},
    pool: shm.ShmPool,
    width: u32,
    height: u32,
    stride: u32,
    surface_id: u32,
    xdg_surface_id: u32,
    xdg_id: u32,
    shm_id: u32,
    pool_id: u32,
    buf_id: u32,
    toplevel_id: u32,
    // Pending size from an xdg_toplevel.configure, applied on the next
    // xdg_surface.configure. 0 means "no change requested".
    pending_width: u32 = 0,
    pending_height: u32 = 0,

    fn currentBuffer(ptr: *anyopaque) hal.Error!platform.Buffer {
        const self: *WaylandSurface = @ptrCast(@alignCast(ptr));
        return platform.Buffer{
            .bytes = self.pool.data,
            .width = self.width,
            .height = self.height,
            .stride = self.stride,
            // Wayland XRGB8888 is B,G,R,X in memory on little-endian. Report
            // bgra8 so the driver's present blit writes the correct byte order
            // directly into this shm buffer (no post-attach mutation).
            .format = .bgra8_unorm,
        };
    }

    /// Attach the current buffer + damage + commit. The buffer already holds correct
    /// XRGB8888 bytes from the present blit. Pixels are not mutated here since the
    /// compositor reads the shm buffer asynchronously.
    fn sendFrame(self: *WaylandSurface) hal.Error!void {
        const conn = self.conn;
        // wl_surface.attach (opcode 1): buffer, x=0, y=0
        conn.wire_writer.begin(conn.allocator, self.surface_id, 1) catch return error.DeviceLost;
        conn.wire_writer.writeObject(conn.allocator, self.buf_id) catch return error.DeviceLost;
        conn.wire_writer.writeInt(conn.allocator, 0) catch return error.DeviceLost;
        conn.wire_writer.writeInt(conn.allocator, 0) catch return error.DeviceLost;
        conn.sendMessage(conn.wire_writer.finish()) catch return error.DeviceLost;

        // wl_surface.damage (opcode 2): x, y, w, h
        conn.wire_writer.begin(conn.allocator, self.surface_id, 2) catch return error.DeviceLost;
        conn.wire_writer.writeInt(conn.allocator, 0) catch return error.DeviceLost;
        conn.wire_writer.writeInt(conn.allocator, 0) catch return error.DeviceLost;
        conn.wire_writer.writeInt(conn.allocator, @intCast(self.width)) catch return error.DeviceLost;
        conn.wire_writer.writeInt(conn.allocator, @intCast(self.height)) catch return error.DeviceLost;
        conn.sendMessage(conn.wire_writer.finish()) catch return error.DeviceLost;

        // wl_surface.commit (opcode 6)
        conn.wire_writer.begin(conn.allocator, self.surface_id, 6) catch return error.DeviceLost;
        conn.sendMessage(conn.wire_writer.finish()) catch return error.DeviceLost;
    }

    fn commit(ptr: *anyopaque) hal.Error!void {
        const self: *WaylandSurface = @ptrCast(@alignCast(ptr));
        try self.sendFrame();
    }

    fn size(ptr: *anyopaque) [2]u32 {
        const self: *WaylandSurface = @ptrCast(@alignCast(ptr));
        return .{ self.width, self.height };
    }

    /// Tear down the current wl_buffer + wl_shm_pool and create fresh ones at
    /// (w, h). Updates pool/buf/pool_id/width/height/stride. The caller then
    /// re-renders into currentBuffer() and present()s to attach the new buffer.
    fn recreateBuffer(self: *WaylandSurface, w: u32, h: u32) hal.Error!void {
        const conn = self.conn;
        // wl_buffer.destroy (opcode 0), wl_shm_pool.destroy (opcode 1).
        conn.wire_writer.begin(conn.allocator, self.buf_id, 0) catch return error.DeviceLost;
        conn.sendMessage(conn.wire_writer.finish()) catch return error.DeviceLost;
        conn.wire_writer.begin(conn.allocator, self.pool_id, 1) catch return error.DeviceLost;
        conn.sendMessage(conn.wire_writer.finish()) catch return error.DeviceLost;
        self.pool.deinit();

        const stride = w * 4;
        const pool_size = @as(usize, stride) * h;
        var pool = shm.ShmPool.create(pool_size) catch return error.InitializationFailed;
        errdefer pool.deinit();
        @memset(pool.data, 0);

        // wl_shm.create_pool (opcode 0): new_id, size (fd OOB via sendFd).
        const pool_id = conn.objects.allocId();
        conn.wire_writer.begin(conn.allocator, self.shm_id, 0) catch return error.OutOfMemory;
        conn.wire_writer.writeNewId(conn.allocator, pool_id) catch return error.OutOfMemory;
        conn.wire_writer.writeInt(conn.allocator, @intCast(pool_size)) catch return error.OutOfMemory;
        shm.sendFd(conn.stream.socket.handle, conn.wire_writer.finish(), pool.fd) catch return error.InitializationFailed;

        // wl_shm_pool.create_buffer (opcode 0): new_id, offset, w, h, stride, format=XRGB8888.
        const buf_id = conn.objects.allocId();
        conn.wire_writer.begin(conn.allocator, pool_id, 0) catch return error.OutOfMemory;
        conn.wire_writer.writeNewId(conn.allocator, buf_id) catch return error.OutOfMemory;
        conn.wire_writer.writeInt(conn.allocator, 0) catch return error.OutOfMemory;
        conn.wire_writer.writeInt(conn.allocator, @intCast(w)) catch return error.OutOfMemory;
        conn.wire_writer.writeInt(conn.allocator, @intCast(h)) catch return error.OutOfMemory;
        conn.wire_writer.writeInt(conn.allocator, @intCast(stride)) catch return error.OutOfMemory;
        conn.wire_writer.writeUint(conn.allocator, 1) catch return error.OutOfMemory;
        conn.sendMessage(conn.wire_writer.finish()) catch return error.InitializationFailed;

        self.pool = pool;
        self.pool_id = pool_id;
        self.buf_id = buf_id;
        self.width = w;
        self.height = h;
        self.stride = stride;
    }

    /// Block on the next compositor event and handle it. Pongs pings. Tracks the
    /// requested size from xdg_toplevel.configure. On xdg_surface.configure, acks
    /// and either resizes the buffer (returning .resized) or redraws.
    /// Returns .closed on xdg_toplevel.close.
    fn processEvents(ptr: *anyopaque) hal.Error!platform.WindowEvent {
        const self: *WaylandSurface = @ptrCast(@alignCast(ptr));
        const conn = self.conn;
        var msg_buf: [4096]u8 = undefined;
        const ev = client.dispatchOne(conn, &msg_buf) catch return error.DeviceLost;
        checkDisplayError(self.log, ev.object_id, ev.opcode, msg_buf[0..ev.size]);
        if (ev.object_id == self.xdg_id and ev.opcode == 0) {
            // xdg_wm_base.ping -> pong (opcode 3)
            var r = client.WireReader.init(msg_buf[0..ev.size]) catch return .none;
            const serial = r.readUint() catch return .none;
            conn.wire_writer.begin(conn.allocator, self.xdg_id, 3) catch return error.OutOfMemory;
            conn.wire_writer.writeUint(conn.allocator, serial) catch return error.OutOfMemory;
            conn.sendMessage(conn.wire_writer.finish()) catch return error.DeviceLost;
        } else if (ev.object_id == self.toplevel_id and ev.opcode == 0) {
            // xdg_toplevel.configure: width(int), height(int), states(array). 0 = client chooses.
            var r = client.WireReader.init(msg_buf[0..ev.size]) catch return .none;
            const w = r.readInt() catch return .none;
            const h = r.readInt() catch return .none;
            if (w > 0 and h > 0) {
                self.pending_width = @intCast(w);
                self.pending_height = @intCast(h);
            }
        } else if (ev.object_id == self.xdg_surface_id and ev.opcode == 0) {
            // xdg_surface.configure -> ack_configure (opcode 4).
            var r = client.WireReader.init(msg_buf[0..ev.size]) catch return .none;
            const serial = r.readUint() catch return .none;
            conn.wire_writer.begin(conn.allocator, self.xdg_surface_id, 4) catch return error.OutOfMemory;
            conn.wire_writer.writeUint(conn.allocator, serial) catch return error.OutOfMemory;
            conn.sendMessage(conn.wire_writer.finish()) catch return error.DeviceLost;
            const pw = self.pending_width;
            const ph = self.pending_height;
            self.pending_width = 0;
            self.pending_height = 0;
            if (pw != 0 and (pw != self.width or ph != self.height)) {
                try self.recreateBuffer(pw, ph);
                return .resized;
            }
            try self.sendFrame();
        } else if (ev.object_id == self.toplevel_id and ev.opcode == 1) {
            // xdg_toplevel.close
            return .closed;
        }
        return .none;
    }

    fn deinit(ptr: *anyopaque) void {
        const self: *WaylandSurface = @ptrCast(@alignCast(ptr));
        // Destroy toplevel, xdg_surface (best-effort).
        // xdg_toplevel.destroy opcode 0
        self.conn.wire_writer.begin(self.conn.allocator, self.toplevel_id, 0) catch {};
        self.conn.sendMessage(self.conn.wire_writer.finish()) catch {};
        // xdg_surface.destroy opcode 0
        self.conn.wire_writer.begin(self.conn.allocator, self.xdg_surface_id, 0) catch {};
        self.conn.sendMessage(self.conn.wire_writer.finish()) catch {};
        self.pool.deinit();
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

const WaylandDisplay = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    conn: client.Connection,
    compositor_id: u32,
    shm_id: u32,
    xdg_id: u32,
    log: prism_log.Logger = .{},

    fn createSurface(ptr: *anyopaque, desc: platform.SurfaceDesc) hal.Error!platform.Surface {
        const self: *WaylandDisplay = @ptrCast(@alignCast(ptr));
        const conn = &self.conn;

        const w: i32 = @intCast(desc.width);
        const h: i32 = @intCast(desc.height);
        const stride = desc.width * 4;
        const pool_size = @as(usize, stride) * desc.height;

        // Create SHM pool backed by memfd.
        var pool = shm.ShmPool.create(pool_size) catch return error.InitializationFailed;
        errdefer pool.deinit();
        @memset(pool.data, 0);

        // wl_shm.create_pool (opcode 0): new_id, fd, size.
        // The fd is transferred out-of-band via SCM_RIGHTS (shm.sendFd below) and
        // occupies zero bytes in the message body. The body is only new_id + size.
        // Writing a placeholder word for the fd shifts size and makes the compositor
        // read size 0 ("invalid wl_shm_pool size").
        const pool_id = conn.objects.allocId();
        conn.wire_writer.begin(conn.allocator, self.shm_id, 0) catch return error.OutOfMemory;
        conn.wire_writer.writeNewId(conn.allocator, pool_id) catch return error.OutOfMemory;
        conn.wire_writer.writeInt(conn.allocator, @intCast(pool_size)) catch return error.OutOfMemory;
        const pool_msg = conn.wire_writer.finish();
        shm.sendFd(conn.stream.socket.handle, pool_msg, pool.fd) catch return error.InitializationFailed;

        // wl_shm_pool.create_buffer (opcode 0): new_id, offset, w, h, stride, format=1(XRGB8888)
        const buf_id = conn.objects.allocId();
        conn.wire_writer.begin(conn.allocator, pool_id, 0) catch return error.OutOfMemory;
        conn.wire_writer.writeNewId(conn.allocator, buf_id) catch return error.OutOfMemory;
        conn.wire_writer.writeInt(conn.allocator, 0) catch return error.OutOfMemory;
        conn.wire_writer.writeInt(conn.allocator, w) catch return error.OutOfMemory;
        conn.wire_writer.writeInt(conn.allocator, h) catch return error.OutOfMemory;
        conn.wire_writer.writeInt(conn.allocator, @intCast(stride)) catch return error.OutOfMemory;
        conn.wire_writer.writeUint(conn.allocator, 1) catch return error.OutOfMemory;
        conn.sendMessage(conn.wire_writer.finish()) catch return error.InitializationFailed;

        // wl_compositor.create_surface (opcode 0): new_id
        const surface_id = conn.objects.allocId();
        conn.wire_writer.begin(conn.allocator, self.compositor_id, 0) catch return error.OutOfMemory;
        conn.wire_writer.writeNewId(conn.allocator, surface_id) catch return error.OutOfMemory;
        conn.sendMessage(conn.wire_writer.finish()) catch return error.InitializationFailed;

        // xdg_wm_base.get_xdg_surface (opcode 2): new_id, surface
        const xdg_surface_id = conn.objects.allocId();
        conn.wire_writer.begin(conn.allocator, self.xdg_id, 2) catch return error.OutOfMemory;
        conn.wire_writer.writeNewId(conn.allocator, xdg_surface_id) catch return error.OutOfMemory;
        conn.wire_writer.writeObject(conn.allocator, surface_id) catch return error.OutOfMemory;
        conn.sendMessage(conn.wire_writer.finish()) catch return error.InitializationFailed;

        // xdg_surface.get_toplevel (opcode 1): new_id
        const toplevel_id = conn.objects.allocId();
        conn.wire_writer.begin(conn.allocator, xdg_surface_id, 1) catch return error.OutOfMemory;
        conn.wire_writer.writeNewId(conn.allocator, toplevel_id) catch return error.OutOfMemory;
        conn.sendMessage(conn.wire_writer.finish()) catch return error.InitializationFailed;

        // Initial commit to trigger xdg_surface.configure.
        conn.wire_writer.begin(conn.allocator, surface_id, 6) catch return error.OutOfMemory;
        conn.sendMessage(conn.wire_writer.finish()) catch return error.InitializationFailed;

        // Wait for xdg_surface.configure and ack it.
        waitConfigure(self.log, conn, xdg_surface_id, self.xdg_id) catch return error.InitializationFailed;

        const s = self.gpa.create(WaylandSurface) catch return error.OutOfMemory;
        s.* = .{
            .gpa = self.gpa,
            .conn = &self.conn,
            .log = self.log,
            .pool = pool,
            .width = desc.width,
            .height = desc.height,
            .stride = stride,
            .surface_id = surface_id,
            .xdg_surface_id = xdg_surface_id,
            .xdg_id = self.xdg_id,
            .shm_id = self.shm_id,
            .pool_id = pool_id,
            .buf_id = buf_id,
            .toplevel_id = toplevel_id,
        };
        return platform.Surface{ .ptr = s, .vtable = &WaylandSurface.vtable };
    }

    fn deinit(ptr: *anyopaque) void {
        const self: *WaylandDisplay = @ptrCast(@alignCast(ptr));
        self.conn.deinit();
        self.gpa.destroy(self);
    }

    const vtable = platform.Display.VTable{
        .createSurface = &createSurface,
        .deinit = &deinit,
    };
};

/// Diagnostic: if the event is a wl_display.error (object 1, opcode 0), decode
/// and print it. The compositor sends this right before dropping a misbehaving
/// client, so it explains "no protocol error on our side, but no window".
fn checkDisplayError(logger: prism_log.Logger, object_id: u32, opcode: u16, buf: []const u8) void {
    if (object_id != 1 or opcode != 0) return;
    var r = client.WireReader.init(buf) catch return;
    const bad_obj = r.readUint() catch return;
    const code = r.readUint() catch return;
    const msg = (r.readString() catch null) orelse "<unreadable>";
    logger.log("wl_display.error: object={d} code={d} message={s}", .{ bad_obj, code, msg });
}

/// Block until xdg_surface.configure arrives, then send ack_configure.
fn waitConfigure(
    logger: prism_log.Logger,
    conn: *client.Connection,
    xdg_surface_id: u32,
    xdg_id: u32,
) !void {
    var msg_buf: [4096]u8 = undefined;
    var configured = false;
    while (!configured) {
        const ev = try client.dispatchOne(conn, &msg_buf);
        checkDisplayError(logger, ev.object_id, ev.opcode, msg_buf[0..ev.size]);
        if (ev.object_id == xdg_id and ev.opcode == 0) {
            // xdg_wm_base.ping -> pong
            var r = try client.WireReader.init(msg_buf[0..ev.size]);
            const serial = try r.readUint();
            try conn.wire_writer.begin(conn.allocator, xdg_id, 3);
            try conn.wire_writer.writeUint(conn.allocator, serial);
            try conn.sendMessage(conn.wire_writer.finish());
        } else if (ev.object_id == xdg_surface_id and ev.opcode == 0) {
            // xdg_surface.configure -> ack_configure
            var r = try client.WireReader.init(msg_buf[0..ev.size]);
            const serial = try r.readUint();
            try conn.wire_writer.begin(conn.allocator, xdg_surface_id, 4);
            try conn.wire_writer.writeUint(conn.allocator, serial);
            try conn.sendMessage(conn.wire_writer.finish());
            configured = true;
        }
    }
}

/// Connect to the Wayland compositor at `socket_path` and return a Display.
/// `socket_path` should be a resolved path (e.g. from client.resolveSocketPath).
/// Performs network I/O. Do not call from tests.
pub fn create(
    gpa: std.mem.Allocator,
    io: std.Io,
    socket_path: []const u8,
) hal.Error!platform.Display {
    var conn = client.connect(gpa, io, socket_path) catch return error.InitializationFailed;
    errdefer conn.deinit();

    const registry_id = client.getRegistry(&conn) catch return error.InitializationFailed;
    const sync_cb_id = client.sync(&conn) catch return error.InitializationFailed;

    var compositor_id: u32 = 0;
    var shm_id: u32 = 0;
    var xdg_id: u32 = 0;

    var msg_buf: [4096]u8 = undefined;
    var got_sync = false;
    while (!got_sync) {
        const ev = client.dispatchOne(&conn, &msg_buf) catch return error.InitializationFailed;
        checkDisplayError(.{}, ev.object_id, ev.opcode, msg_buf[0..ev.size]);
        if (ev.object_id == registry_id and ev.opcode == 0) {
            var r = client.WireReader.init(msg_buf[0..ev.size]) catch continue;
            const g = client.decodeRegistryGlobal(&r) catch continue;
            if (std.mem.eql(u8, g.interface, "wl_compositor")) {
                compositor_id = client.bindGlobal(
                    &conn,
                    registry_id,
                    g.name,
                    g.interface,
                    @min(g.version, 4),
                ) catch return error.InitializationFailed;
            } else if (std.mem.eql(u8, g.interface, "wl_shm")) {
                shm_id = client.bindGlobal(
                    &conn,
                    registry_id,
                    g.name,
                    g.interface,
                    @min(g.version, 1),
                ) catch return error.InitializationFailed;
            } else if (std.mem.eql(u8, g.interface, "xdg_wm_base")) {
                xdg_id = client.bindGlobal(
                    &conn,
                    registry_id,
                    g.name,
                    g.interface,
                    @min(g.version, 1),
                ) catch return error.InitializationFailed;
            }
        } else if (ev.object_id == sync_cb_id) {
            got_sync = true;
        }
    }

    if (compositor_id == 0 or shm_id == 0 or xdg_id == 0) return error.InitializationFailed;

    const d = gpa.create(WaylandDisplay) catch return error.OutOfMemory;
    d.* = .{
        .gpa = gpa,
        .io = io,
        .conn = conn,
        .compositor_id = compositor_id,
        .shm_id = shm_id,
        .xdg_id = xdg_id,
    };
    return platform.Display{ .ptr = d, .vtable = &WaylandDisplay.vtable };
}

/// Ask the compositor which DRM device it wants clients to render on, returned
/// as a system dev_t (pair with prism.platform.drm.driverForDev /
/// drivers.selectForDrmDevice to choose the matching driver). Prefers
/// zwp_linux_dmabuf_v1 default feedback `main_device`. Falls back to the drm_fd
/// from wp_drm_lease_device_v1 if linux-dmabuf is missing (e.g. COSMIC on NVIDIA).
/// Opens its own short-lived connection. Returns null if the compositor names no device.
pub fn preferredDevice(gpa: std.mem.Allocator, io: std.Io, socket_path: []const u8) ?u64 {
    var conn = client.connect(gpa, io, socket_path) catch return null;
    defer conn.deinit();

    const registry_id = client.getRegistry(&conn) catch return null;
    var sync_id = client.sync(&conn) catch return null;
    var msg_buf: [4096]u8 = undefined;

    // Round 1: bind zwp_linux_dmabuf_v1 (v4 needed for default feedback) and note
    // wp_drm_lease_device_v1 as a fallback for compositors that don't advertise
    // linux-dmabuf (e.g. COSMIC on NVIDIA).
    var dmabuf_id: u32 = 0;
    var lease_name: u32 = 0;
    var lease_ver: u32 = 0;
    var round_done = false;
    while (!round_done) {
        const ev = client.dispatchOne(&conn, &msg_buf) catch return null;
        if (ev.object_id == registry_id and ev.opcode == 0) {
            var r = client.WireReader.init(msg_buf[0..ev.size]) catch continue;
            const g = client.decodeRegistryGlobal(&r) catch continue;
            if (dmabuf_id == 0 and g.version >= 4 and std.mem.eql(u8, g.interface, "zwp_linux_dmabuf_v1")) {
                dmabuf_id = client.bindGlobal(&conn, registry_id, g.name, g.interface, 4) catch return null;
            } else if (lease_name == 0 and std.mem.eql(u8, g.interface, "wp_drm_lease_device_v1")) {
                lease_name = g.name;
                lease_ver = g.version;
            }
        } else if (ev.object_id == sync_id) {
            round_done = true;
        }
    }

    // Path A (preferred): linux-dmabuf default feedback's main_device.
    if (dmabuf_id != 0) {
        // get_default_feedback (request opcode 2): new_id.
        const feedback_id = conn.objects.allocId();
        conn.wire_writer.begin(conn.allocator, dmabuf_id, 2) catch return null;
        conn.wire_writer.writeNewId(conn.allocator, feedback_id) catch return null;
        conn.sendMessage(conn.wire_writer.finish()) catch return null;
        sync_id = client.sync(&conn) catch return null;
        var dev: ?u64 = null;
        round_done = false;
        while (!round_done) {
            const ev = client.dispatchOne(&conn, &msg_buf) catch return null;
            if (ev.object_id == feedback_id and ev.opcode == 2) { // main_device(array)
                var r = client.WireReader.init(msg_buf[0..ev.size]) catch continue;
                const arr = r.readArray() catch continue;
                if (arr.len >= 8) {
                    const p: *const [8]u8 = @ptrCast(arr.ptr);
                    dev = @bitCast(p.*); // dev_t in host byte order
                }
            } else if (ev.object_id == sync_id) {
                round_done = true;
            }
        }
        if (dev) |d| return d;
    }

    // Path B (fallback): wp_drm_lease_device_v1 sends a drm_fd on bind.
    // st_rdev is the DRM device. Binding triggers the events, so sync and drain.
    if (lease_name != 0) {
        const lease_id = client.bindGlobal(&conn, registry_id, lease_name, "wp_drm_lease_device_v1", @min(lease_ver, 1)) catch return null;
        sync_id = client.sync(&conn) catch return null;
        var dev: ?u64 = null;
        round_done = false;
        while (!round_done) {
            const ev = client.dispatchOne(&conn, &msg_buf) catch return null;
            if (ev.object_id == lease_id and ev.opcode == 0) { // drm_fd(fd), transferred OOB
                if (conn.takeFd()) |fd| {
                    const linux = std.os.linux;
                    var stx: linux.Statx = undefined;
                    const mask: linux.STATX = @bitCast(@as(u32, 0x7ff)); // STATX_BASIC_STATS
                    const rc = linux.statx(fd, "", 0x1000, mask, &stx); // AT_EMPTY_PATH
                    if (@as(isize, @bitCast(rc)) >= 0) {
                        dev = @import("drm.zig").makedev(stx.rdev_major, stx.rdev_minor);
                    }
                    _ = std.posix.system.close(fd);
                }
            } else if (ev.object_id == sync_id) {
                round_done = true;
            }
        }
        if (dev) |d| return d;
    }

    return null;
}

test "rgba->bgra swap restores under double application" {
    // The R<->B conversion lives in the software present blit. Verify the
    // swap is its own inverse (sanity for the byte mapping used there).
    var px = [_]u8{ 0xFF, 0x80, 0x00, 0xFF };
    const orig = px;
    const swap = struct {
        fn f(p: []u8) void {
            const r = p[0];
            p[0] = p[2];
            p[2] = r;
        }
    }.f;
    swap(&px);
    try std.testing.expectEqual(@as(u8, 0x00), px[0]);
    try std.testing.expectEqual(@as(u8, 0xFF), px[2]);
    swap(&px);
    try std.testing.expectEqualSlices(u8, &orig, &px);
}
