//! EGL C-ABI boundary and libglvnd vendor ABI. Exports EGL entry points with
//! their exact C signatures, maps Prism errors to per-thread EGLint codes, and
//! fills the __EGLapiImports table that libEGL.so reads from __egl_Main.

const std = @import("std");
const prism = @import("prism");
const egl = @import("egl.zig");
const state = @import("state.zig");
const gles = @import("gles.zig");

/// Oracle test helper: flush batched draws, then map the surface backbuffer directly.
/// Raw backbuffer reads bypass eglSwapBuffers/glReadPixels (the normal flush points), so draws
/// must be flushed first or the read races a pending deferred draw. No-op when not batching.
fn flushMap(dev: prism.hal.Device, s: *state.Surface) prism.Error![]u8 {
    if (state.currentContext()) |c| try c.flushDraws();
    return dev.mapResource(s.backbuffer);
}

const EGLint = egl.EGLint;
const EGLenum = egl.EGLenum;
const EGLBoolean = egl.EGLBoolean;
const EGLDisplay = egl.EGLDisplay;
const EGLConfig = egl.EGLConfig;
const EGLContext = egl.EGLContext;
const EGLSurface = egl.EGLSurface;

// --- Per-thread error + the GLVND callbacks --------------------------------

/// Per-thread EGL last-error. When libEGL is driving us, also forwarded via
/// exports.setEGLError so the loader's eglGetError agrees.
threadlocal var last_error: EGLint = egl.EGL_SUCCESS;

/// Exports table from __egl_Main. Null when called directly via dlopen (no-loader path).
var api_exports: ?*const egl.EGLapiExports = null;

/// Bound client API (eglBindAPI), per-thread. Defaults to OpenGL ES per the spec.
threadlocal var current_api: EGLenum = egl.EGL_OPENGL_ES_API;

/// Opt-in tracing for the GLVND-loader bring-up path (PRISM_EGL_DEBUG=1). Reads
/// the env directly from /proc/self/environ (NUL-separated KEY=VALUE records) so
/// it works inside a dlopen'd vendor without linking libc. The EGL vendor links
/// no C library. std.os.linux provides the raw open/read/close.
fn debugEnabled() bool {
    return envPresent("PRISM_EGL_DEBUG");
}

/// True if env var `name` is set (any value), read libc-free from /proc/self/environ.
fn envPresent(name: []const u8) bool {
    const linux = std.os.linux;
    const fd: i32 = blk: {
        const rc = linux.open("/proc/self/environ", .{ .ACCMODE = .RDONLY }, 0);
        if (@as(isize, @bitCast(rc)) < 0) return false;
        break :blk @intCast(rc);
    };
    defer _ = linux.close(fd);
    var buf: [16384]u8 = undefined;
    var total: usize = 0;
    while (total < buf.len) {
        const rc = linux.read(fd, buf[total..].ptr, buf.len - total);
        const n: isize = @bitCast(rc);
        if (n <= 0) break;
        total += @intCast(n);
    }
    var it = std.mem.splitScalar(u8, buf[0..total], 0);
    while (it.next()) |entry| {
        if (entry.len > name.len and entry[name.len] == '=' and std.mem.eql(u8, entry[0..name.len], name)) return true;
    }
    return false;
}

fn setError(code: EGLint) void {
    last_error = code;
    if (api_exports) |ex| {
        if (ex.setEGLError) |f| f(code);
    }
}

/// Translate a Prism Zig error into the closest EGL error code.
fn eglErrorFor(err: prism.Error) EGLint {
    return switch (err) {
        error.OutOfMemory => egl.EGL_BAD_ALLOC,
        error.InvalidArgument => egl.EGL_BAD_PARAMETER,
        error.Unsupported, error.NotImplemented => egl.EGL_BAD_MATCH,
        error.InitializationFailed, error.DeviceLost => egl.EGL_NOT_INITIALIZED,
    };
}

// --- Core EGL entry points (exported C ABI) --------------------------------

pub fn eglGetError() callconv(.c) EGLint {
    const e = last_error;
    last_error = egl.EGL_SUCCESS; // eglGetError resets to EGL_SUCCESS.
    return e;
}

pub fn eglGetPlatformDisplay(
    platform: EGLenum,
    native_display: ?*anyopaque,
    attrib_list: ?[*]const egl.EGLAttrib,
) callconv(.c) EGLDisplay {
    _ = attrib_list;
    const d = state.getPlatformDisplay(platform, native_display) orelse {
        // Not our platform. Return EGL_NO_DISPLAY with no error so libEGL tries
        // the next vendor (per the GLVND getPlatformDisplay contract).
        return egl.EGL_NO_DISPLAY;
    };
    setError(egl.EGL_SUCCESS);
    return @ptrCast(d);
}

pub fn eglGetPlatformDisplayEXT(
    platform: EGLenum,
    native_display: ?*anyopaque,
    attrib_list: ?[*]const EGLint,
) callconv(.c) EGLDisplay {
    _ = attrib_list; // EXT takes EGLint attribs. Ignored for M1.
    const d = state.getPlatformDisplay(platform, native_display) orelse return egl.EGL_NO_DISPLAY;
    setError(egl.EGL_SUCCESS);
    return @ptrCast(d);
}

pub fn eglGetDisplay(native_display: ?*anyopaque) callconv(.c) EGLDisplay {
    // eglGetDisplay(EGL_DEFAULT_DISPLAY) -> our surfaceless default display.
    const d = state.getPlatformDisplay(egl.EGL_PLATFORM_NONE, native_display) orelse return egl.EGL_NO_DISPLAY;
    setError(egl.EGL_SUCCESS);
    return @ptrCast(d);
}

pub fn eglInitialize(dpy: EGLDisplay, major: ?*EGLint, minor: ?*EGLint) callconv(.c) EGLBoolean {
    const d = state.lookupDisplay(dpy) orelse {
        setError(egl.EGL_BAD_DISPLAY);
        return egl.EGL_FALSE;
    };
    state.initialize(d) catch |e| {
        setError(eglErrorFor(e));
        return egl.EGL_FALSE;
    };
    if (major) |p| p.* = 1;
    if (minor) |p| p.* = 5;
    setError(egl.EGL_SUCCESS);
    return egl.EGL_TRUE;
}

pub fn eglTerminate(dpy: EGLDisplay) callconv(.c) EGLBoolean {
    const d = state.lookupDisplay(dpy) orelse {
        setError(egl.EGL_BAD_DISPLAY);
        return egl.EGL_FALSE;
    };
    state.terminate(d);
    setError(egl.EGL_SUCCESS);
    return egl.EGL_TRUE;
}

pub fn eglQueryString(dpy: EGLDisplay, name: EGLint) callconv(.c) ?[*:0]const u8 {
    // EGL_EXTENSIONS / EGL_VERSION can be queried with EGL_NO_DISPLAY for client
    // extensions in 1.5, but for M1 we require an initialized display (the common
    // path) and otherwise report it through the display query.
    const d = state.lookupDisplay(dpy) orelse {
        setError(egl.EGL_BAD_DISPLAY);
        return null;
    };
    if (!d.initialized) {
        setError(egl.EGL_NOT_INITIALIZED);
        return null;
    }
    const s = state.queryString(name) orelse {
        setError(egl.EGL_BAD_PARAMETER);
        return null;
    };
    setError(egl.EGL_SUCCESS);
    return s;
}

pub fn eglGetConfigs(
    dpy: EGLDisplay,
    configs: ?[*]EGLConfig,
    config_size: EGLint,
    num_config: ?*EGLint,
) callconv(.c) EGLBoolean {
    const d = state.lookupDisplay(dpy) orelse {
        setError(egl.EGL_BAD_DISPLAY);
        return egl.EGL_FALSE;
    };
    if (!d.initialized) {
        setError(egl.EGL_NOT_INITIALIZED);
        return egl.EGL_FALSE;
    }
    if (num_config == null) {
        setError(egl.EGL_BAD_PARAMETER);
        return egl.EGL_FALSE;
    }
    if (configs == null) {
        // Query-count form.
        num_config.?.* = @intCast(state.configs.len);
        setError(egl.EGL_SUCCESS);
        return egl.EGL_TRUE;
    }
    const cap: usize = if (config_size < 0) 0 else @intCast(config_size);
    var written: usize = 0;
    while (written < state.configs.len and written < cap) : (written += 1) {
        configs.?[written] = state.configHandle(written);
    }
    num_config.?.* = @intCast(written);
    setError(egl.EGL_SUCCESS);
    return egl.EGL_TRUE;
}

pub fn eglChooseConfig(
    dpy: EGLDisplay,
    attrib_list: ?[*]const EGLint,
    configs: ?[*]EGLConfig,
    config_size: EGLint,
    num_config: ?*EGLint,
) callconv(.c) EGLBoolean {
    const d = state.lookupDisplay(dpy) orelse {
        setError(egl.EGL_BAD_DISPLAY);
        return egl.EGL_FALSE;
    };
    if (!d.initialized) {
        setError(egl.EGL_NOT_INITIALIZED);
        return egl.EGL_FALSE;
    }
    if (num_config == null) {
        setError(egl.EGL_BAD_PARAMETER);
        return egl.EGL_FALSE;
    }
    var indices: [state.configs.len]usize = undefined;
    if (configs == null) {
        // Count-only form: validate the list and report the match count.
        const total = state.chooseConfig(attrib_list, &indices) catch {
            setError(egl.EGL_BAD_ATTRIBUTE);
            return egl.EGL_FALSE;
        };
        num_config.?.* = @intCast(total);
        setError(egl.EGL_SUCCESS);
        return egl.EGL_TRUE;
    }
    const total = state.chooseConfig(attrib_list, &indices) catch {
        setError(egl.EGL_BAD_ATTRIBUTE);
        return egl.EGL_FALSE;
    };
    const cap: usize = if (config_size < 0) 0 else @intCast(config_size);
    var written: usize = 0;
    while (written < total and written < cap) : (written += 1) {
        configs.?[written] = state.configHandle(indices[written]);
    }
    num_config.?.* = @intCast(written);
    setError(egl.EGL_SUCCESS);
    return egl.EGL_TRUE;
}

pub fn eglGetConfigAttrib(
    dpy: EGLDisplay,
    config: EGLConfig,
    attribute: EGLint,
    value: ?*EGLint,
) callconv(.c) EGLBoolean {
    const d = state.lookupDisplay(dpy) orelse {
        setError(egl.EGL_BAD_DISPLAY);
        return egl.EGL_FALSE;
    };
    if (!d.initialized) {
        setError(egl.EGL_NOT_INITIALIZED);
        return egl.EGL_FALSE;
    }
    const idx = state.configIndex(config) orelse {
        setError(egl.EGL_BAD_CONFIG);
        return egl.EGL_FALSE;
    };
    if (value == null) {
        setError(egl.EGL_BAD_PARAMETER);
        return egl.EGL_FALSE;
    }
    const v = state.configs[idx].attrib(attribute) orelse {
        setError(egl.EGL_BAD_ATTRIBUTE);
        return egl.EGL_FALSE;
    };
    value.?.* = v;
    setError(egl.EGL_SUCCESS);
    return egl.EGL_TRUE;
}

pub fn eglBindAPI(api: EGLenum) callconv(.c) EGLBoolean {
    if (!state.supportsAPI(api)) {
        setError(egl.EGL_BAD_PARAMETER);
        return egl.EGL_FALSE;
    }
    current_api = api;
    setError(egl.EGL_SUCCESS);
    return egl.EGL_TRUE;
}

pub fn eglQueryAPI() callconv(.c) EGLenum {
    return current_api;
}

pub fn eglReleaseThread() callconv(.c) EGLBoolean {
    last_error = egl.EGL_SUCCESS;
    current_api = egl.EGL_OPENGL_ES_API;
    return egl.EGL_TRUE;
}

// --- M2 surface/context + GLES clear/present (the render path) --------------

/// Parse an EGLint attrib list (key,value,...,EGL_NONE) for one key, returning
/// `default` if absent. Used for pbuffer EGL_WIDTH/EGL_HEIGHT and context version.
fn attribValue(attrib_list: ?[*]const EGLint, key: EGLint, default: EGLint) EGLint {
    const list = attrib_list orelse return default;
    var i: usize = 0;
    while (list[i] != egl.EGL_NONE) : (i += 2) {
        if (list[i] == key) return list[i + 1];
    }
    return default;
}

pub fn eglCreateContext(
    dpy: EGLDisplay,
    config: EGLConfig,
    share: EGLContext,
    attrib_list: ?[*]const EGLint,
) callconv(.c) EGLContext {
    _ = share; // share groups are not modeled in M2.
    const d = state.lookupDisplay(dpy) orelse {
        setError(egl.EGL_BAD_DISPLAY);
        return egl.EGL_NO_CONTEXT;
    };
    if (!d.initialized) {
        setError(egl.EGL_NOT_INITIALIZED);
        return egl.EGL_NO_CONTEXT;
    }
    const idx = state.configIndex(config) orelse {
        setError(egl.EGL_BAD_CONFIG);
        return egl.EGL_NO_CONTEXT;
    };
    _ = attribValue(attrib_list, egl.EGL_CONTEXT_CLIENT_VERSION, 1); // accepted. tracker is GLES-version-agnostic.
    const c = state.createContext(d, idx, current_api) catch |e| {
        setError(eglErrorFor(e));
        return egl.EGL_NO_CONTEXT;
    };
    setError(egl.EGL_SUCCESS);
    return @ptrCast(c);
}

pub fn eglDestroyContext(dpy: EGLDisplay, ctx: EGLContext) callconv(.c) EGLBoolean {
    if (state.lookupDisplay(dpy) == null) {
        setError(egl.EGL_BAD_DISPLAY);
        return egl.EGL_FALSE;
    }
    const c = state.lookupContext(ctx) orelse {
        setError(egl.EGL_BAD_CONTEXT);
        return egl.EGL_FALSE;
    };
    // EGL 1.5 3.7.1: if current, defer destruction until released by eglMakeCurrent.
    // Destroying now would unbind from the still-current GLVND dispatch, leaving subsequent
    // GL calls with no current context. Keep it alive and flagged instead.
    if (state.currentContext() == c) {
        c.pending_delete = true;
    } else {
        c.deinit();
    }
    setError(egl.EGL_SUCCESS);
    return egl.EGL_TRUE;
}

pub fn eglCreateWindowSurface(
    dpy: EGLDisplay,
    config: EGLConfig,
    win: ?*anyopaque,
    attrib_list: ?[*]const EGLint,
) callconv(.c) EGLSurface {
    _ = attrib_list;
    const d = state.lookupDisplay(dpy) orelse {
        setError(egl.EGL_BAD_DISPLAY);
        return egl.EGL_NO_SURFACE;
    };
    if (!d.initialized) {
        setError(egl.EGL_NOT_INITIALIZED);
        return egl.EGL_NO_SURFACE;
    }
    const idx = state.configIndex(config) orelse {
        setError(egl.EGL_BAD_CONFIG);
        return egl.EGL_NO_SURFACE;
    };
    const native = win orelse {
        setError(egl.EGL_BAD_NATIVE_WINDOW);
        return egl.EGL_NO_SURFACE;
    };
    // Two native-window conventions, selected by how the display was created:
    //   1. App-wl_display-bound (eglGetPlatformDisplay(EGL_PLATFORM_WAYLAND,
    //      app's wl_display)): the native window is a stock `wl_egl_window *`
    //      from libwayland-egl. Parse it + present via the system libwayland
    //      (wl_shm on the app's wl_display). Stock Wayland GLES path.
    //   2. Prism-platform-bound (Prism's self-contained EGL_EXT_platform_wayland):
    //      the native window is a Prism platform.Surface* the client created
    //      (it holds the Io + connection). Presents via the HAL platform layer.
    if (d.appWlDisplay() != null) {
        const s = state.createWaylandWindowSurface(d, idx, native) catch |e| {
            // A malformed wl_egl_window maps to BAD_NATIVE_WINDOW.
            setError(if (e == error.InvalidArgument) egl.EGL_BAD_NATIVE_WINDOW else eglErrorFor(e));
            return egl.EGL_NO_SURFACE;
        };
        setError(egl.EGL_SUCCESS);
        return @ptrCast(s);
    }
    // 3. GBM-bound (eglGetPlatformDisplay(EGL_PLATFORM_GBM, gbm_device)): the
    //    native window is the app's gbm.Surface. Render into its back buffer; the
    //    app scans out the front buffer via gbm_surface_lock_front_buffer.
    if (d.gbmDevice() != null) {
        const gbm_surface: *prism.platform.gbm.Surface = @ptrCast(@alignCast(native));
        const s = state.createGbmWindowSurface(d, idx, gbm_surface) catch |e| {
            setError(if (e == error.InvalidArgument) egl.EGL_BAD_NATIVE_WINDOW else eglErrorFor(e));
            return egl.EGL_NO_SURFACE;
        };
        setError(egl.EGL_SUCCESS);
        return @ptrCast(s);
    }
    const plat: *prism.platform.Surface = @ptrCast(@alignCast(native));
    const s = state.createWindowSurface(d, idx, plat) catch |e| {
        setError(eglErrorFor(e));
        return egl.EGL_NO_SURFACE;
    };
    setError(egl.EGL_SUCCESS);
    return @ptrCast(s);
}

pub fn eglCreatePbufferSurface(
    dpy: EGLDisplay,
    config: EGLConfig,
    attrib_list: ?[*]const EGLint,
) callconv(.c) EGLSurface {
    const d = state.lookupDisplay(dpy) orelse {
        setError(egl.EGL_BAD_DISPLAY);
        return egl.EGL_NO_SURFACE;
    };
    if (!d.initialized) {
        setError(egl.EGL_NOT_INITIALIZED);
        return egl.EGL_NO_SURFACE;
    }
    const idx = state.configIndex(config) orelse {
        setError(egl.EGL_BAD_CONFIG);
        return egl.EGL_NO_SURFACE;
    };
    const wi = attribValue(attrib_list, egl.EGL_WIDTH, 0);
    const hi = attribValue(attrib_list, egl.EGL_HEIGHT, 0);
    if (wi <= 0 or hi <= 0) {
        setError(egl.EGL_BAD_PARAMETER);
        return egl.EGL_NO_SURFACE;
    }
    const s = state.createPbufferSurface(d, idx, @intCast(wi), @intCast(hi)) catch |e| {
        setError(eglErrorFor(e));
        return egl.EGL_NO_SURFACE;
    };
    setError(egl.EGL_SUCCESS);
    return @ptrCast(s);
}

pub fn eglDestroySurface(dpy: EGLDisplay, surface: EGLSurface) callconv(.c) EGLBoolean {
    if (state.lookupDisplay(dpy) == null) {
        setError(egl.EGL_BAD_DISPLAY);
        return egl.EGL_FALSE;
    }
    const s = state.lookupSurface(surface) orelse {
        setError(egl.EGL_BAD_SURFACE);
        return egl.EGL_FALSE;
    };
    // Submit any batched draws before the surface's resources are freed (a pending draw may
    // target this surface's backbuffer/depth).
    if (state.currentContext()) |c| c.flushDraws() catch {};
    // Unbind it from the current thread if it is current (so we never present a
    // freed surface).
    if (state.currentDrawSurface() == s or state.currentReadSurface() == s) {
        state.makeCurrent(state.currentContext(), null, null);
    }
    s.deinit();
    setError(egl.EGL_SUCCESS);
    return egl.EGL_TRUE;
}

pub fn eglMakeCurrent(
    dpy: EGLDisplay,
    draw: EGLSurface,
    read: EGLSurface,
    ctx: EGLContext,
) callconv(.c) EGLBoolean {
    if (state.lookupDisplay(dpy) == null) {
        setError(egl.EGL_BAD_DISPLAY);
        return egl.EGL_FALSE;
    }
    // The context being released by this call: if it was flagged for deferred deletion
    // (eglDestroyContext while current) and we are switching away from it, reap it now.
    const releasing = state.currentContext();
    // Unbind: eglMakeCurrent(dpy, NO_SURFACE, NO_SURFACE, NO_CONTEXT) releases the
    // thread's current context + surfaces.
    if (ctx == egl.EGL_NO_CONTEXT) {
        if (draw != egl.EGL_NO_SURFACE or read != egl.EGL_NO_SURFACE) {
            setError(egl.EGL_BAD_MATCH);
            return egl.EGL_FALSE;
        }
        state.makeCurrent(null, null, null);
        if (releasing) |old| if (old.pending_delete) old.deinit();
        setError(egl.EGL_SUCCESS);
        return egl.EGL_TRUE;
    }
    const c = state.lookupContext(ctx) orelse {
        setError(egl.EGL_BAD_CONTEXT);
        return egl.EGL_FALSE;
    };
    // Both surfaces must be valid when a context is supplied. This tracker
    // requires bound draw+read surfaces. Surfaceless contexts are a later milestone.
    const ds = state.lookupSurface(draw) orelse {
        setError(egl.EGL_BAD_SURFACE);
        return egl.EGL_FALSE;
    };
    const rs = state.lookupSurface(read) orelse {
        setError(egl.EGL_BAD_SURFACE);
        return egl.EGL_FALSE;
    };
    // Binding a different context: reset this thread's GLES state to GL defaults so the
    // previously-current context's fixed-function state (depth test, cull, clear color,
    // bindings) does not leak into the newly-bound one. Re-binding the same context
    // preserves its accumulated state.
    if (releasing != c) gles.resetThreadState();
    state.makeCurrent(c, ds, rs);
    // Reap the previously-current context if it was destroyed while current and we are
    // now switching to a different one.
    if (releasing) |old| if (old != c and old.pending_delete) old.deinit();
    setError(egl.EGL_SUCCESS);
    return egl.EGL_TRUE;
}

pub fn eglSwapBuffers(dpy: EGLDisplay, surface: EGLSurface) callconv(.c) EGLBoolean {
    if (state.lookupDisplay(dpy) == null) {
        setError(egl.EGL_BAD_DISPLAY);
        return egl.EGL_FALSE;
    }
    const s = state.lookupSurface(surface) orelse {
        setError(egl.EGL_BAD_SURFACE);
        return egl.EGL_FALSE;
    };
    // Per the EGL spec a swap requires a current context (the rendering context
    // whose backbuffer is posted). Use the thread's current context.
    const c = state.currentContext() orelse {
        setError(egl.EGL_BAD_CONTEXT);
        return egl.EGL_FALSE;
    };
    state.swapBuffers(c, s) catch |e| {
        setError(eglErrorFor(e));
        return egl.EGL_FALSE;
    };
    setError(egl.EGL_SUCCESS);
    return egl.EGL_TRUE;
}

// --- Remaining EGL 1.0-1.5 static entry points ------------------------------
// GLVND builds its static dispatch table by asking the vendor's getProcAddress
// for every EGL 1.0-1.5 function at load time, and discards a vendor missing
// any of them. All of these must resolve. M1 implements the ones the enumeration
// path needs for real. The rest are honest stubs returning the correct "not yet"
// EGL error (query getters return sensible no-current values).

pub fn eglGetCurrentContext() callconv(.c) EGLContext {
    const c = state.currentContext() orelse return egl.EGL_NO_CONTEXT;
    return @ptrCast(c);
}
pub fn eglGetCurrentDisplay() callconv(.c) EGLDisplay {
    const c = state.currentContext() orelse return egl.EGL_NO_DISPLAY;
    return @ptrCast(c.display);
}
/// EGL_DRAW = 0x3059, EGL_READ = 0x305A. Returns the bound draw/read surface.
pub fn eglGetCurrentSurface(readdraw: EGLint) callconv(.c) EGLSurface {
    const s = switch (readdraw) {
        0x305A => state.currentReadSurface(), // EGL_READ
        else => state.currentDrawSurface(), // EGL_DRAW (default)
    } orelse return egl.EGL_NO_SURFACE;
    return @ptrCast(s);
}
pub fn eglQueryContext(dpy: EGLDisplay, ctx: EGLContext, attribute: EGLint, value: ?*EGLint) callconv(.c) EGLBoolean {
    if (state.lookupDisplay(dpy) == null) {
        setError(egl.EGL_BAD_DISPLAY);
        return egl.EGL_FALSE;
    }
    const c = state.lookupContext(ctx) orelse {
        setError(egl.EGL_BAD_CONTEXT);
        return egl.EGL_FALSE;
    };
    const v = value orelse {
        setError(egl.EGL_BAD_PARAMETER);
        return egl.EGL_FALSE;
    };
    switch (attribute) {
        egl.EGL_CONFIG_ID => v.* = state.configs[c.config].id,
        egl.EGL_CONTEXT_CLIENT_TYPE => v.* = @intCast(c.client_api),
        egl.EGL_CONTEXT_CLIENT_VERSION => v.* = 2,
        egl.EGL_RENDER_BUFFER => v.* = 0x3084, // EGL_BACK_BUFFER
        else => {
            setError(egl.EGL_BAD_ATTRIBUTE);
            return egl.EGL_FALSE;
        },
    }
    setError(egl.EGL_SUCCESS);
    return egl.EGL_TRUE;
}
pub fn eglQuerySurface(dpy: EGLDisplay, surface: EGLSurface, attribute: EGLint, value: ?*EGLint) callconv(.c) EGLBoolean {
    if (state.lookupDisplay(dpy) == null) {
        setError(egl.EGL_BAD_DISPLAY);
        return egl.EGL_FALSE;
    }
    const s = state.lookupSurface(surface) orelse {
        setError(egl.EGL_BAD_SURFACE);
        return egl.EGL_FALSE;
    };
    const v = value orelse {
        setError(egl.EGL_BAD_PARAMETER);
        return egl.EGL_FALSE;
    };
    switch (attribute) {
        egl.EGL_WIDTH => v.* = @intCast(s.width),
        egl.EGL_HEIGHT => v.* = @intCast(s.height),
        egl.EGL_CONFIG_ID => v.* = state.configs[s.config].id,
        egl.EGL_RENDER_BUFFER => v.* = 0x3084, // EGL_BACK_BUFFER
        egl.EGL_LARGEST_PBUFFER => v.* = egl.EGL_FALSE,
        else => {
            setError(egl.EGL_BAD_ATTRIBUTE);
            return egl.EGL_FALSE;
        },
    }
    setError(egl.EGL_SUCCESS);
    return egl.EGL_TRUE;
}
pub fn eglSurfaceAttrib(dpy: EGLDisplay, surface: EGLSurface, attribute: EGLint, value: EGLint) callconv(.c) EGLBoolean {
    _ = surface;
    _ = attribute;
    _ = value;
    if (state.lookupDisplay(dpy) == null) {
        setError(egl.EGL_BAD_DISPLAY);
        return egl.EGL_FALSE;
    }
    setError(egl.EGL_BAD_SURFACE);
    return egl.EGL_FALSE;
}
pub fn eglSwapInterval(dpy: EGLDisplay, interval: EGLint) callconv(.c) EGLBoolean {
    _ = interval;
    if (state.lookupDisplay(dpy) == null) {
        setError(egl.EGL_BAD_DISPLAY);
        return egl.EGL_FALSE;
    }
    setError(egl.EGL_BAD_SURFACE);
    return egl.EGL_FALSE;
}
pub fn eglBindTexImage(dpy: EGLDisplay, surface: EGLSurface, buffer: EGLint) callconv(.c) EGLBoolean {
    _ = surface;
    _ = buffer;
    if (state.lookupDisplay(dpy) == null) {
        setError(egl.EGL_BAD_DISPLAY);
        return egl.EGL_FALSE;
    }
    setError(egl.EGL_BAD_SURFACE);
    return egl.EGL_FALSE;
}
pub fn eglReleaseTexImage(dpy: EGLDisplay, surface: EGLSurface, buffer: EGLint) callconv(.c) EGLBoolean {
    return eglBindTexImage(dpy, surface, buffer);
}
pub fn eglCopyBuffers(dpy: EGLDisplay, surface: EGLSurface, target: ?*anyopaque) callconv(.c) EGLBoolean {
    _ = surface;
    _ = target;
    if (state.lookupDisplay(dpy) == null) {
        setError(egl.EGL_BAD_DISPLAY);
        return egl.EGL_FALSE;
    }
    setError(egl.EGL_BAD_SURFACE);
    return egl.EGL_FALSE;
}
pub fn eglCreatePixmapSurface(dpy: EGLDisplay, config: EGLConfig, pixmap: ?*anyopaque, attrib_list: ?[*]const EGLint) callconv(.c) EGLSurface {
    _ = config;
    _ = pixmap;
    _ = attrib_list;
    if (state.lookupDisplay(dpy) == null) {
        setError(egl.EGL_BAD_DISPLAY);
        return egl.EGL_NO_SURFACE;
    }
    setError(egl.EGL_BAD_MATCH);
    return egl.EGL_NO_SURFACE;
}
pub fn eglCreatePlatformWindowSurface(dpy: EGLDisplay, config: EGLConfig, native_window: ?*anyopaque, attrib_list: ?[*]const egl.EGLAttrib) callconv(.c) EGLSurface {
    _ = attrib_list; // EGL 1.5 takes EGLAttrib attribs. We have none to honor.
    // The EGL 1.5 form a stock Wayland app uses: native_window is the same
    // `wl_egl_window *`. Route it through the shared window-surface path (which
    // selects wl_egl_window vs Prism platform.Surface by the display binding).
    return eglCreateWindowSurface(dpy, config, native_window, null);
}
pub fn eglCreatePlatformPixmapSurface(dpy: EGLDisplay, config: EGLConfig, native_pixmap: ?*anyopaque, attrib_list: ?[*]const egl.EGLAttrib) callconv(.c) EGLSurface {
    return eglCreatePlatformWindowSurface(dpy, config, native_pixmap, attrib_list);
}
pub fn eglCreatePbufferFromClientBuffer(dpy: EGLDisplay, buftype: EGLenum, buffer: ?*anyopaque, config: EGLConfig, attrib_list: ?[*]const EGLint) callconv(.c) EGLSurface {
    _ = buftype;
    _ = buffer;
    _ = config;
    _ = attrib_list;
    if (state.lookupDisplay(dpy) == null) {
        setError(egl.EGL_BAD_DISPLAY);
        return egl.EGL_NO_SURFACE;
    }
    setError(egl.EGL_BAD_MATCH);
    return egl.EGL_NO_SURFACE;
}
pub fn eglWaitGL() callconv(.c) EGLBoolean {
    setError(egl.EGL_SUCCESS);
    return egl.EGL_TRUE;
}
pub fn eglWaitClient() callconv(.c) EGLBoolean {
    setError(egl.EGL_SUCCESS);
    return egl.EGL_TRUE;
}
pub fn eglWaitNative(engine: EGLint) callconv(.c) EGLBoolean {
    _ = engine;
    setError(egl.EGL_SUCCESS);
    return egl.EGL_TRUE;
}
// EGL 1.5 sync + image objects (M2+): resolve but fail with the correct error.
pub fn eglCreateSync(dpy: EGLDisplay, type_: EGLenum, attrib_list: ?[*]const egl.EGLAttrib) callconv(.c) ?*anyopaque {
    _ = type_;
    _ = attrib_list;
    if (state.lookupDisplay(dpy) == null) {
        setError(egl.EGL_BAD_DISPLAY);
        return null;
    }
    setError(egl.EGL_BAD_MATCH);
    return null;
}
pub fn eglDestroySync(dpy: EGLDisplay, sync: ?*anyopaque) callconv(.c) EGLBoolean {
    _ = sync;
    if (state.lookupDisplay(dpy) == null) {
        setError(egl.EGL_BAD_DISPLAY);
        return egl.EGL_FALSE;
    }
    setError(egl.EGL_BAD_PARAMETER);
    return egl.EGL_FALSE;
}
pub fn eglClientWaitSync(dpy: EGLDisplay, sync: ?*anyopaque, flags: EGLint, timeout: i64) callconv(.c) EGLint {
    _ = sync;
    _ = flags;
    _ = timeout;
    if (state.lookupDisplay(dpy) == null) {
        setError(egl.EGL_BAD_DISPLAY);
        return egl.EGL_FALSE;
    }
    setError(egl.EGL_BAD_PARAMETER);
    return egl.EGL_FALSE;
}
pub fn eglWaitSync(dpy: EGLDisplay, sync: ?*anyopaque, flags: EGLint) callconv(.c) EGLBoolean {
    _ = sync;
    _ = flags;
    if (state.lookupDisplay(dpy) == null) {
        setError(egl.EGL_BAD_DISPLAY);
        return egl.EGL_FALSE;
    }
    setError(egl.EGL_BAD_PARAMETER);
    return egl.EGL_FALSE;
}
pub fn eglGetSyncAttrib(dpy: EGLDisplay, sync: ?*anyopaque, attribute: EGLint, value: ?*egl.EGLAttrib) callconv(.c) EGLBoolean {
    _ = sync;
    _ = attribute;
    _ = value;
    if (state.lookupDisplay(dpy) == null) {
        setError(egl.EGL_BAD_DISPLAY);
        return egl.EGL_FALSE;
    }
    setError(egl.EGL_BAD_PARAMETER);
    return egl.EGL_FALSE;
}
pub fn eglCreateImage(dpy: EGLDisplay, ctx: EGLContext, target: EGLenum, buffer: ?*anyopaque, attrib_list: ?[*]const egl.EGLAttrib) callconv(.c) ?*anyopaque {
    _ = ctx;
    _ = target;
    _ = buffer;
    _ = attrib_list;
    if (state.lookupDisplay(dpy) == null) {
        setError(egl.EGL_BAD_DISPLAY);
        return null;
    }
    setError(egl.EGL_BAD_MATCH);
    return null;
}
pub fn eglDestroyImage(dpy: EGLDisplay, image: ?*anyopaque) callconv(.c) EGLBoolean {
    _ = image;
    if (state.lookupDisplay(dpy) == null) {
        setError(egl.EGL_BAD_DISPLAY);
        return egl.EGL_FALSE;
    }
    setError(egl.EGL_BAD_PARAMETER);
    return egl.EGL_FALSE;
}

// --- Minimal GLES entry points (the clear path) -----------------------------
// GLES is a separate client API. eglGetProcAddress resolves these too, and an
// EGL/GLES client links/dlsyms them. They drive the per-thread current EGL
// context's HAL backbuffer via the gles state tracker. Scope: clear only (M2).

pub fn glClearColor(r: gles.GLclampf, g: gles.GLclampf, b: gles.GLclampf, a: gles.GLclampf) callconv(.c) void {
    gles.clearColor(r, g, b, a);
}
pub fn glClear(mask: gles.GLbitfield) callconv(.c) void {
    gles.clear(mask);
}
pub fn glViewport(x: gles.GLint, y: gles.GLint, width: gles.GLsizei, height: gles.GLsizei) callconv(.c) void {
    gles.setViewport(x, y, width, height);
}
pub fn glScissor(x: gles.GLint, y: gles.GLint, width: gles.GLsizei, height: gles.GLsizei) callconv(.c) void {
    gles.setScissorBox(x, y, width, height);
}
pub fn glLineWidth(w: gles.GLfloat) callconv(.c) void {
    gles.setLineWidth(w);
}
pub fn glSampleCoverage(value: gles.GLfloat, invert: gles.GLboolean) callconv(.c) void {
    gles.sampleCoverage(value, invert);
}
pub fn glGetString(name: gles.GLenum) callconv(.c) ?[*:0]const gles.GLubyte {
    return gles.getString(name);
}
pub fn glGetStringi(name: gles.GLenum, index: gles.GLuint) callconv(.c) ?[*:0]const gles.GLubyte {
    return gles.getStringi(name, index);
}
pub fn glTexStorage2D(target: gles.GLenum, levels: gles.GLsizei, internalformat: gles.GLenum, width: gles.GLsizei, height: gles.GLsizei) callconv(.c) void {
    gles.texStorage2D(target, levels, internalformat, width, height);
}
pub fn glTexStorage3D(target: gles.GLenum, levels: gles.GLsizei, internalformat: gles.GLenum, width: gles.GLsizei, height: gles.GLsizei, depth: gles.GLsizei) callconv(.c) void {
    gles.texStorage3D(target, levels, internalformat, width, height, depth);
}
pub fn glInvalidateFramebuffer(target: gles.GLenum, num: gles.GLsizei, attachments: ?[*]const gles.GLenum) callconv(.c) void {
    gles.invalidateFramebuffer(target, num, attachments);
}
pub fn glInvalidateSubFramebuffer(target: gles.GLenum, num: gles.GLsizei, attachments: ?[*]const gles.GLenum, x: gles.GLint, y: gles.GLint, width: gles.GLsizei, height: gles.GLsizei) callconv(.c) void {
    gles.invalidateSubFramebuffer(target, num, attachments, x, y, width, height);
}
pub fn glFenceSync(condition: gles.GLenum, flags: u32) callconv(.c) ?*anyopaque {
    return gles.fenceSync(condition, flags);
}
pub fn glClientWaitSync(sync: ?*anyopaque, flags: u32, timeout: u64) callconv(.c) gles.GLenum {
    return gles.clientWaitSync(sync, flags, timeout);
}
pub fn glWaitSync(sync: ?*anyopaque, flags: u32, timeout: u64) callconv(.c) void {
    gles.waitSync(sync, flags, timeout);
}
pub fn glDeleteSync(sync: ?*anyopaque) callconv(.c) void {
    gles.deleteSync(sync);
}
pub fn glIsSync(sync: ?*anyopaque) callconv(.c) gles.GLboolean {
    return gles.isSync(sync);
}
pub fn glGetSynciv(sync: ?*anyopaque, pname: gles.GLenum, buf_size: gles.GLsizei, length: ?*gles.GLsizei, values: ?[*]gles.GLint) callconv(.c) void {
    gles.getSynciv(sync, pname, buf_size, length, values);
}
pub fn glBlitFramebuffer(sx0: gles.GLint, sy0: gles.GLint, sx1: gles.GLint, sy1: gles.GLint, dx0: gles.GLint, dy0: gles.GLint, dx1: gles.GLint, dy1: gles.GLint, mask: gles.GLbitfield, filter: gles.GLenum) callconv(.c) void {
    gles.blitFramebuffer(sx0, sy0, sx1, sy1, dx0, dy0, dx1, dy1, mask, filter);
}
pub fn glClearBufferfv(buffer: gles.GLenum, drawbuffer: gles.GLint, value: ?[*]const gles.GLfloat) callconv(.c) void {
    gles.clearBufferfv(buffer, drawbuffer, value);
}
pub fn glClearBufferiv(buffer: gles.GLenum, drawbuffer: gles.GLint, value: ?[*]const gles.GLint) callconv(.c) void {
    gles.clearBufferiv(buffer, drawbuffer, value);
}
pub fn glClearBufferuiv(buffer: gles.GLenum, drawbuffer: gles.GLint, value: ?[*]const u32) callconv(.c) void {
    gles.clearBufferuiv(buffer, drawbuffer, value);
}
pub fn glClearBufferfi(buffer: gles.GLenum, drawbuffer: gles.GLint, depth: gles.GLfloat, stencil: gles.GLint) callconv(.c) void {
    gles.clearBufferfi(buffer, drawbuffer, depth, stencil);
}
pub fn glCopyBufferSubData(read_target: gles.GLenum, write_target: gles.GLenum, read_offset: isize, write_offset: isize, size: isize) callconv(.c) void {
    gles.copyBufferSubData(read_target, write_target, read_offset, write_offset, size);
}
pub fn glReadBuffer(src: gles.GLenum) callconv(.c) void {
    gles.readBuffer(src);
}
pub fn glGenQueries(n: gles.GLsizei, ids: ?[*]gles.GLuint) callconv(.c) void {
    gles.genQueries(n, ids);
}
pub fn glDeleteQueries(n: gles.GLsizei, ids: ?[*]const gles.GLuint) callconv(.c) void {
    gles.deleteQueries(n, ids);
}
pub fn glIsQuery(id: gles.GLuint) callconv(.c) gles.GLboolean {
    return gles.isQuery(id);
}
pub fn glBeginQuery(target: gles.GLenum, id: gles.GLuint) callconv(.c) void {
    gles.beginQuery(target, id);
}
pub fn glEndQuery(target: gles.GLenum) callconv(.c) void {
    gles.endQuery(target);
}
pub fn glGetQueryObjectuiv(id: gles.GLuint, pname: gles.GLenum, params: ?*u32) callconv(.c) void {
    gles.getQueryObjectuiv(id, pname, params);
}
pub fn glGetQueryObjectiv(id: gles.GLuint, pname: gles.GLenum, params: ?*gles.GLint) callconv(.c) void {
    gles.getQueryObjectiv(id, pname, params);
}
pub fn glGetBufferSubData(target: gles.GLenum, offset: isize, size: isize, data: ?*anyopaque) callconv(.c) void {
    gles.getBufferSubData(target, offset, size, data);
}
pub fn glGetShaderSource(shader: gles.GLuint, buf_size: gles.GLsizei, length: ?*gles.GLint, source: ?[*]gles.GLchar) callconv(.c) void {
    gles.getShaderSource(shader, buf_size, length, source);
}
pub fn glDepthRangef(n: gles.GLfloat, f: gles.GLfloat) callconv(.c) void {
    gles.depthRangef(n, f);
}
pub fn glReleaseShaderCompiler() callconv(.c) void {
    gles.releaseShaderCompiler();
}
pub fn glHint(target: gles.GLenum, mode: gles.GLenum) callconv(.c) void {
    gles.hint(target, mode);
}
pub fn glDrawBuffers(n: gles.GLsizei, bufs: ?[*]const gles.GLenum) callconv(.c) void {
    gles.drawBuffers(n, bufs);
}
pub fn glGetInternalformativ(target: gles.GLenum, internalformat: gles.GLenum, pname: gles.GLenum, buf_size: gles.GLsizei, params: ?[*]gles.GLint) callconv(.c) void {
    gles.getInternalformativ(target, internalformat, pname, buf_size, params);
}
pub fn glGetIntegerv(pname: gles.GLenum, params: ?[*]gles.GLint) callconv(.c) void {
    gles.getIntegerv(pname, params);
}
pub fn glGetFloatv(pname: gles.GLenum, params: ?[*]gles.GLfloat) callconv(.c) void {
    gles.getFloatv(pname, params);
}
pub fn glGetBooleanv(pname: gles.GLenum, params: ?[*]gles.GLboolean) callconv(.c) void {
    gles.getBooleanv(pname, params);
}
pub fn glIsEnabled(cap: gles.GLenum) callconv(.c) gles.GLboolean {
    return gles.isEnabled(cap);
}
pub fn glGetShaderPrecisionFormat(shadertype: gles.GLenum, precisiontype: gles.GLenum, range: ?[*]gles.GLint, precision: ?*gles.GLint) callconv(.c) void {
    gles.getShaderPrecisionFormat(shadertype, precisiontype, range, precision);
}
pub fn glGetTexParameteriv(target: gles.GLenum, pname: gles.GLenum, params: ?[*]gles.GLint) callconv(.c) void {
    gles.getTexParameteriv(target, pname, params);
}
pub fn glGetTexParameterfv(target: gles.GLenum, pname: gles.GLenum, params: ?[*]gles.GLfloat) callconv(.c) void {
    gles.getTexParameterfv(target, pname, params);
}
pub fn glGetVertexAttribiv(index: gles.GLuint, pname: gles.GLenum, params: ?[*]gles.GLint) callconv(.c) void {
    gles.getVertexAttribiv(index, pname, params);
}
pub fn glGetVertexAttribfv(index: gles.GLuint, pname: gles.GLenum, params: ?[*]gles.GLfloat) callconv(.c) void {
    gles.getVertexAttribfv(index, pname, params);
}
pub fn glGetBufferParameteriv(target: gles.GLenum, pname: gles.GLenum, params: ?[*]gles.GLint) callconv(.c) void {
    gles.getBufferParameteriv(target, pname, params);
}
pub fn glGetRenderbufferParameteriv(target: gles.GLenum, pname: gles.GLenum, params: ?[*]gles.GLint) callconv(.c) void {
    gles.getRenderbufferParameteriv(target, pname, params);
}
pub fn glGetFramebufferAttachmentParameteriv(target: gles.GLenum, attachment: gles.GLenum, pname: gles.GLenum, params: ?[*]gles.GLint) callconv(.c) void {
    gles.getFramebufferAttachmentParameteriv(target, attachment, pname, params);
}
pub fn glGetAttachedShaders(program: gles.GLuint, max_count: gles.GLsizei, count: ?*gles.GLsizei, shaders: ?[*]gles.GLuint) callconv(.c) void {
    gles.getAttachedShaders(program, max_count, count, shaders);
}
pub fn glGetUniformfv(program: gles.GLuint, location: gles.GLint, params: ?[*]gles.GLfloat) callconv(.c) void {
    gles.getUniformfv(program, location, params);
}
pub fn glGetUniformiv(program: gles.GLuint, location: gles.GLint, params: ?[*]gles.GLint) callconv(.c) void {
    gles.getUniformiv(program, location, params);
}
pub fn glGetVertexAttribPointerv(index: gles.GLuint, pname: gles.GLenum, pointer: ?*?*anyopaque) callconv(.c) void {
    gles.getVertexAttribPointerv(index, pname, pointer);
}
pub fn glGetError() callconv(.c) gles.GLenum {
    return gles.getError();
}
pub fn glFinish() callconv(.c) void {
    // Submit any batched draws (glFinish/glFlush must make prior GL commands reach the GPU).
    // The submit fences, so on return the work is complete, matching glFinish's stronger guarantee.
    if (state.currentContext()) |c| c.flushDraws() catch {};
}
pub fn glFlush() callconv(.c) void {
    if (state.currentContext()) |c| c.flushDraws() catch {};
}

// --- GLES2 shader objects + vertex attributes + draw (EGL M3) ---------------
// Gradient-triangle path: SPIR-V shader binaries (GL_ARB_gl_spirv), vertex
// buffers, vertex-attribute arrays, and glDrawArrays. Drives the per-thread
// current EGL context's HAL device through the gles state tracker, building a
// HAL pipeline from the VS+FS SPIR-V and rasterizing via the software driver's
// SPIR-V -> Vulcan JIT path.

pub fn glGenBuffers(n: gles.GLsizei, buffers: ?[*]gles.GLuint) callconv(.c) void {
    gles.genBuffers(n, buffers);
}
pub fn glBindBuffer(target: gles.GLenum, buffer: gles.GLuint) callconv(.c) void {
    gles.bindBuffer(target, buffer);
}
pub fn glBufferData(target: gles.GLenum, size: gles.GLsizeiptr, data: ?*const anyopaque, usage: gles.GLenum) callconv(.c) void {
    gles.bufferData(target, size, data, usage);
}
pub fn glDeleteBuffers(n: gles.GLsizei, buffers: ?[*]const gles.GLuint) callconv(.c) void {
    gles.deleteBuffers(n, buffers);
}
pub fn glVertexAttribPointer(index: gles.GLuint, size: gles.GLint, gl_type: gles.GLenum, normalized: gles.GLboolean, stride: gles.GLsizei, pointer: ?*const anyopaque) callconv(.c) void {
    // For a bound vertex buffer (GLES2 VBO path), `pointer` is a byte offset
    // into the buffer, not a client pointer. Reinterpret its integer value.
    const offset: usize = @intFromPtr(pointer);
    gles.vertexAttribPointer(index, size, gl_type, normalized, stride, offset);
}
pub fn glVertexAttribIPointer(index: gles.GLuint, size: gles.GLint, gl_type: gles.GLenum, stride: gles.GLsizei, pointer: ?*const anyopaque) callconv(.c) void {
    // GLES3 integer vertex attributes: `pointer` is a byte offset into the bound VBO (same
    // convention as glVertexAttribPointer). Integers are delivered raw to an `in ivec4`/
    // `uvec4` VS input with no normalization.
    const offset: usize = @intFromPtr(pointer);
    gles.vertexAttribIPointer(index, size, gl_type, stride, offset);
}
pub fn glEnableVertexAttribArray(index: gles.GLuint) callconv(.c) void {
    gles.enableVertexAttribArray(index);
}
pub fn glDisableVertexAttribArray(index: gles.GLuint) callconv(.c) void {
    gles.disableVertexAttribArray(index);
}
pub fn glCreateShader(shader_type: gles.GLenum) callconv(.c) gles.GLuint {
    return gles.createShader(shader_type);
}
pub fn glShaderSource(shader: gles.GLuint, count: gles.GLsizei, string: ?[*]const ?[*:0]const gles.GLchar, length: ?[*]const gles.GLint) callconv(.c) void {
    gles.shaderSource(shader, count, string, length);
}
pub fn glShaderBinary(count: gles.GLsizei, shaders: ?[*]const gles.GLuint, binaryformat: gles.GLenum, binary: ?*const anyopaque, length: gles.GLsizei) callconv(.c) void {
    gles.shaderBinary(count, shaders, binaryformat, binary, length);
}
pub fn glCompileShader(shader: gles.GLuint) callconv(.c) void {
    gles.compileShader(shader);
}
pub fn glSpecializeShader(shader: gles.GLuint, entry: ?[*:0]const gles.GLchar, num: gles.GLuint, idx: ?[*]const gles.GLuint, val: ?[*]const gles.GLuint) callconv(.c) void {
    gles.specializeShader(shader, entry, num, idx, val);
}
pub fn glGetShaderiv(shader: gles.GLuint, pname: gles.GLenum, params: ?*gles.GLint) callconv(.c) void {
    gles.getShaderiv(shader, pname, params);
}
pub fn glGetShaderInfoLog(shader: gles.GLuint, buf_size: gles.GLsizei, length: ?*gles.GLint, info_log: ?[*]gles.GLchar) callconv(.c) void {
    gles.getShaderInfoLog(shader, buf_size, length, info_log);
}
pub fn glDeleteShader(shader: gles.GLuint) callconv(.c) void {
    gles.deleteShader(shader);
}
pub fn glCreateProgram() callconv(.c) gles.GLuint {
    return gles.createProgram();
}
pub fn glAttachShader(program: gles.GLuint, shader: gles.GLuint) callconv(.c) void {
    gles.attachShader(program, shader);
}
pub fn glDetachShader(program: gles.GLuint, shader: gles.GLuint) callconv(.c) void {
    gles.detachShader(program, shader);
}
pub fn glLinkProgram(program: gles.GLuint) callconv(.c) void {
    gles.linkProgram(program);
}
pub fn glUseProgram(program: gles.GLuint) callconv(.c) void {
    gles.useProgram(program);
}
pub fn glGetProgramiv(program: gles.GLuint, pname: gles.GLenum, params: ?*gles.GLint) callconv(.c) void {
    gles.getProgramiv(program, pname, params);
}
pub fn glDeleteProgram(program: gles.GLuint) callconv(.c) void {
    gles.deleteProgram(program);
}
pub fn glGetAttribLocation(program: gles.GLuint, name: ?[*:0]const gles.GLchar) callconv(.c) gles.GLint {
    return gles.getAttribLocation(program, name);
}
pub fn glDrawArrays(mode: gles.GLenum, first: gles.GLint, count: gles.GLsizei) callconv(.c) void {
    gles.drawArrays(mode, first, count);
}
pub fn glDrawElements(mode: gles.GLenum, count: gles.GLsizei, index_type: gles.GLenum, indices: ?*const anyopaque) callconv(.c) void {
    // For a bound element-array buffer, `indices` is a byte offset into it (GLES VBO path).
    gles.drawElements(mode, count, index_type, @intFromPtr(indices));
}
pub fn glDrawArraysInstanced(mode: gles.GLenum, first: gles.GLint, count: gles.GLsizei, instancecount: gles.GLsizei) callconv(.c) void {
    gles.drawArraysInstanced(mode, first, count, instancecount);
}
pub fn glDrawElementsInstanced(mode: gles.GLenum, count: gles.GLsizei, index_type: gles.GLenum, indices: ?*const anyopaque, instancecount: gles.GLsizei) callconv(.c) void {
    gles.drawElementsInstanced(mode, count, index_type, @intFromPtr(indices), instancecount);
}
pub fn glVertexAttribDivisor(index: gles.GLuint, divisor: gles.GLuint) callconv(.c) void {
    gles.vertexAttribDivisor(index, divisor);
}
pub fn glVertexAttrib1f(index: gles.GLuint, x: gles.GLfloat) callconv(.c) void {
    gles.vertexAttrib4f(index, x, 0, 0, 1);
}
pub fn glVertexAttrib2f(index: gles.GLuint, x: gles.GLfloat, y: gles.GLfloat) callconv(.c) void {
    gles.vertexAttrib4f(index, x, y, 0, 1);
}
pub fn glVertexAttrib3f(index: gles.GLuint, x: gles.GLfloat, y: gles.GLfloat, z: gles.GLfloat) callconv(.c) void {
    gles.vertexAttrib4f(index, x, y, z, 1);
}
pub fn glVertexAttrib4f(index: gles.GLuint, x: gles.GLfloat, y: gles.GLfloat, z: gles.GLfloat, w: gles.GLfloat) callconv(.c) void {
    gles.vertexAttrib4f(index, x, y, z, w);
}
pub fn glVertexAttrib1fv(index: gles.GLuint, v: ?[*]const gles.GLfloat) callconv(.c) void {
    if (v) |vv| gles.vertexAttrib4f(index, vv[0], 0, 0, 1);
}
pub fn glVertexAttrib2fv(index: gles.GLuint, v: ?[*]const gles.GLfloat) callconv(.c) void {
    if (v) |vv| gles.vertexAttrib4f(index, vv[0], vv[1], 0, 1);
}
pub fn glVertexAttrib3fv(index: gles.GLuint, v: ?[*]const gles.GLfloat) callconv(.c) void {
    if (v) |vv| gles.vertexAttrib4f(index, vv[0], vv[1], vv[2], 1);
}
pub fn glVertexAttrib4fv(index: gles.GLuint, v: ?[*]const gles.GLfloat) callconv(.c) void {
    if (v) |vv| gles.vertexAttrib4f(index, vv[0], vv[1], vv[2], vv[3]);
}
pub fn glGenVertexArrays(n: gles.GLsizei, arrays: ?[*]gles.GLuint) callconv(.c) void {
    gles.genVertexArrays(n, arrays);
}
pub fn glBindVertexArray(array: gles.GLuint) callconv(.c) void {
    gles.bindVertexArray(array);
}
pub fn glDeleteVertexArrays(n: gles.GLsizei, arrays: ?[*]const gles.GLuint) callconv(.c) void {
    gles.deleteVertexArrays(n, arrays);
}
pub fn glIsVertexArray(array: gles.GLuint) callconv(.c) gles.GLboolean {
    return gles.isVertexArray(array);
}

// --- GLES2 uniforms, fixed-function state (EGL es2gears milestone) -----------
pub fn glBindAttribLocation(program: gles.GLuint, index: gles.GLuint, name: ?[*:0]const gles.GLchar) callconv(.c) void {
    gles.bindAttribLocation(program, index, name);
}
pub fn glGetUniformLocation(program: gles.GLuint, name: ?[*:0]const gles.GLchar) callconv(.c) gles.GLint {
    return gles.getUniformLocation(program, name);
}
pub fn glGetActiveUniform(program: gles.GLuint, index: gles.GLuint, buf_size: gles.GLsizei, length: ?*gles.GLint, size: ?*gles.GLint, gl_type: ?*gles.GLenum, name: ?[*]gles.GLchar) callconv(.c) void {
    gles.getActiveUniform(program, index, buf_size, length, size, gl_type, name);
}
pub fn glGetActiveAttrib(program: gles.GLuint, index: gles.GLuint, buf_size: gles.GLsizei, length: ?*gles.GLint, size: ?*gles.GLint, gl_type: ?*gles.GLenum, name: ?[*]gles.GLchar) callconv(.c) void {
    gles.getActiveAttrib(program, index, buf_size, length, size, gl_type, name);
}
pub fn glGetProgramInfoLog(program: gles.GLuint, buf_size: gles.GLsizei, length: ?*gles.GLint, info_log: ?[*]gles.GLchar) callconv(.c) void {
    gles.getProgramInfoLog(program, buf_size, length, info_log);
}
pub fn glUniform1f(location: gles.GLint, v0: gles.GLfloat) callconv(.c) void {
    gles.uniform1f(location, v0);
}
pub fn glUniform2f(location: gles.GLint, v0: gles.GLfloat, v1: gles.GLfloat) callconv(.c) void {
    gles.uniform2f(location, v0, v1);
}
pub fn glUniform3f(location: gles.GLint, v0: gles.GLfloat, v1: gles.GLfloat, v2: gles.GLfloat) callconv(.c) void {
    gles.uniform3f(location, v0, v1, v2);
}
pub fn glUniform4f(location: gles.GLint, v0: gles.GLfloat, v1: gles.GLfloat, v2: gles.GLfloat, v3: gles.GLfloat) callconv(.c) void {
    gles.uniform4f(location, v0, v1, v2, v3);
}
pub fn glUniform1i(location: gles.GLint, v0: gles.GLint) callconv(.c) void {
    gles.uniform1i(location, v0);
}
pub fn glUniform1fv(location: gles.GLint, count: gles.GLsizei, value: ?[*]const gles.GLfloat) callconv(.c) void {
    gles.uniform1fv(location, count, value);
}
pub fn glUniform2fv(location: gles.GLint, count: gles.GLsizei, value: ?[*]const gles.GLfloat) callconv(.c) void {
    gles.uniform2fv(location, count, value);
}
pub fn glUniform3fv(location: gles.GLint, count: gles.GLsizei, value: ?[*]const gles.GLfloat) callconv(.c) void {
    gles.uniform3fv(location, count, value);
}
pub fn glUniform4fv(location: gles.GLint, count: gles.GLsizei, value: ?[*]const gles.GLfloat) callconv(.c) void {
    gles.uniform4fv(location, count, value);
}
pub fn glUniformMatrix2fv(location: gles.GLint, count: gles.GLsizei, transpose: gles.GLboolean, value: ?[*]const gles.GLfloat) callconv(.c) void {
    gles.uniformMatrix2fv(location, count, transpose, value);
}
pub fn glUniformMatrix3fv(location: gles.GLint, count: gles.GLsizei, transpose: gles.GLboolean, value: ?[*]const gles.GLfloat) callconv(.c) void {
    gles.uniformMatrix3fv(location, count, transpose, value);
}
pub fn glUniformMatrix4fv(location: gles.GLint, count: gles.GLsizei, transpose: gles.GLboolean, value: ?[*]const gles.GLfloat) callconv(.c) void {
    gles.uniformMatrix4fv(location, count, transpose, value);
}
// --- GLES3 uniform buffer objects (named uniform blocks) --------------------
pub fn glGetUniformBlockIndex(program: gles.GLuint, name: ?[*:0]const gles.GLchar) callconv(.c) gles.GLuint {
    return gles.getUniformBlockIndex(program, name);
}
pub fn glUniformBlockBinding(program: gles.GLuint, block_index: gles.GLuint, binding: gles.GLuint) callconv(.c) void {
    gles.uniformBlockBinding(program, block_index, binding);
}
pub fn glGetActiveUniformBlockiv(program: gles.GLuint, block_index: gles.GLuint, pname: gles.GLenum, params: ?[*]gles.GLint) callconv(.c) void {
    gles.getActiveUniformBlockiv(program, block_index, pname, params);
}
pub fn glGetUniformIndices(program: gles.GLuint, count: gles.GLsizei, names: ?[*]const ?[*:0]const gles.GLchar, indices: ?[*]gles.GLuint) callconv(.c) void {
    gles.getUniformIndices(program, count, names, indices);
}
pub fn glGetActiveUniformsiv(program: gles.GLuint, count: gles.GLsizei, indices: ?[*]const gles.GLuint, pname: gles.GLenum, params: ?[*]gles.GLint) callconv(.c) void {
    gles.getActiveUniformsiv(program, count, indices, pname, params);
}
pub fn glBindBufferBase(target: gles.GLenum, index: gles.GLuint, buffer: gles.GLuint) callconv(.c) void {
    gles.bindBufferBase(target, index, buffer);
}
pub fn glBindBufferRange(target: gles.GLenum, index: gles.GLuint, buffer: gles.GLuint, offset: gles.GLintptr, size: gles.GLsizeiptr) callconv(.c) void {
    gles.bindBufferRange(target, index, buffer, offset, size);
}
pub fn glTransformFeedbackVaryings(program: gles.GLuint, count: gles.GLsizei, varyings: ?[*]const ?[*:0]const gles.GLchar, buffer_mode: gles.GLenum) callconv(.c) void {
    gles.transformFeedbackVaryings(program, count, varyings, buffer_mode);
}
pub fn glBeginTransformFeedback(primitive_mode: gles.GLenum) callconv(.c) void {
    gles.beginTransformFeedback(primitive_mode);
}
pub fn glEndTransformFeedback() callconv(.c) void {
    gles.endTransformFeedback();
}
pub fn glPauseTransformFeedback() callconv(.c) void {
    gles.pauseTransformFeedback();
}
pub fn glResumeTransformFeedback() callconv(.c) void {
    gles.resumeTransformFeedback();
}
pub fn glEnable(cap: gles.GLenum) callconv(.c) void {
    gles.enable(cap);
}
pub fn glDisable(cap: gles.GLenum) callconv(.c) void {
    gles.disable(cap);
}
pub fn glDepthFunc(func: gles.GLenum) callconv(.c) void {
    gles.depthFunc(func);
}
pub fn glDepthMask(flag: gles.GLboolean) callconv(.c) void {
    gles.depthMask(flag);
}
pub fn glClearDepthf(d: gles.GLclampf) callconv(.c) void {
    gles.clearDepthf(d);
}
pub fn glPolygonOffset(factor: gles.GLfloat, units: gles.GLfloat) callconv(.c) void {
    gles.polygonOffset(factor, units);
}
pub fn glStencilFunc(func: gles.GLenum, ref: gles.GLint, mask: gles.GLuint) callconv(.c) void {
    gles.stencilFunc(func, ref, mask);
}
pub fn glStencilOp(sfail: gles.GLenum, dpfail: gles.GLenum, dppass: gles.GLenum) callconv(.c) void {
    gles.stencilOp(sfail, dpfail, dppass);
}
pub fn glStencilMask(mask: gles.GLuint) callconv(.c) void {
    gles.stencilMask(mask);
}
pub fn glStencilFuncSeparate(face: gles.GLenum, func: gles.GLenum, ref: gles.GLint, mask: gles.GLuint) callconv(.c) void {
    gles.stencilFuncSeparate(face, func, ref, mask);
}
pub fn glStencilOpSeparate(face: gles.GLenum, sfail: gles.GLenum, dpfail: gles.GLenum, dppass: gles.GLenum) callconv(.c) void {
    gles.stencilOpSeparate(face, sfail, dpfail, dppass);
}
pub fn glStencilMaskSeparate(face: gles.GLenum, mask: gles.GLuint) callconv(.c) void {
    gles.stencilMaskSeparate(face, mask);
}
pub fn glClearStencil(s: gles.GLint) callconv(.c) void {
    gles.clearStencil(s);
}
pub fn glCullFace(mode: gles.GLenum) callconv(.c) void {
    gles.cullFace(mode);
}
pub fn glFrontFace(mode: gles.GLenum) callconv(.c) void {
    gles.frontFace(mode);
}
pub fn glBlendFunc(sfactor: gles.GLenum, dfactor: gles.GLenum) callconv(.c) void {
    gles.blendFunc(sfactor, dfactor);
}
pub fn glBlendFuncSeparate(src_rgb: gles.GLenum, dst_rgb: gles.GLenum, src_alpha: gles.GLenum, dst_alpha: gles.GLenum) callconv(.c) void {
    gles.blendFuncSeparate(src_rgb, dst_rgb, src_alpha, dst_alpha);
}
pub fn glBlendEquation(mode: gles.GLenum) callconv(.c) void {
    gles.blendEquation(mode);
}
pub fn glBlendEquationSeparate(mode_rgb: gles.GLenum, mode_alpha: gles.GLenum) callconv(.c) void {
    gles.blendEquationSeparate(mode_rgb, mode_alpha);
}
pub fn glBlendColor(r: gles.GLclampf, g: gles.GLclampf, b: gles.GLclampf, a: gles.GLclampf) callconv(.c) void {
    gles.blendColor(r, g, b, a);
}
pub fn glColorMask(r: gles.GLboolean, g: gles.GLboolean, b: gles.GLboolean, a: gles.GLboolean) callconv(.c) void {
    gles.colorMask(r, g, b, a);
}

// --- GLES2 textures / samplers (glmark2-es2 texture milestone) --------------
pub fn glGenTextures(n: gles.GLsizei, txs: ?[*]gles.GLuint) callconv(.c) void {
    gles.genTextures(n, txs);
}
pub fn glDeleteTextures(n: gles.GLsizei, txs: ?[*]const gles.GLuint) callconv(.c) void {
    gles.deleteTextures(n, txs);
}
pub fn glBindTexture(target: gles.GLenum, texture: gles.GLuint) callconv(.c) void {
    gles.bindTexture(target, texture);
}
pub fn glActiveTexture(texture: gles.GLenum) callconv(.c) void {
    gles.activeTexture(texture);
}
pub fn glGenSamplers(n: gles.GLsizei, s: ?[*]gles.GLuint) callconv(.c) void {
    gles.genSamplers(n, s);
}
pub fn glDeleteSamplers(n: gles.GLsizei, s: ?[*]const gles.GLuint) callconv(.c) void {
    gles.deleteSamplers(n, s);
}
pub fn glIsSampler(id: gles.GLuint) callconv(.c) gles.GLboolean {
    return gles.isSampler(id);
}
pub fn glBindSampler(unit: gles.GLuint, sampler: gles.GLuint) callconv(.c) void {
    gles.bindSampler(unit, sampler);
}
pub fn glSamplerParameteri(sampler: gles.GLuint, pname: gles.GLenum, param: gles.GLint) callconv(.c) void {
    gles.samplerParameteri(sampler, pname, param);
}
pub fn glSamplerParameterf(sampler: gles.GLuint, pname: gles.GLenum, param: gles.GLfloat) callconv(.c) void {
    gles.samplerParameterf(sampler, pname, param);
}
pub fn glGetSamplerParameteriv(sampler: gles.GLuint, pname: gles.GLenum, out: ?[*]gles.GLint) callconv(.c) void {
    gles.getSamplerParameteriv(sampler, pname, out);
}
pub fn glIsTexture(texture: gles.GLuint) callconv(.c) gles.GLboolean {
    return gles.isTexture(texture);
}
pub fn glTexParameteri(target: gles.GLenum, pname: gles.GLenum, param: gles.GLint) callconv(.c) void {
    gles.texParameteri(target, pname, param);
}
pub fn glTexParameterf(target: gles.GLenum, pname: gles.GLenum, param: gles.GLfloat) callconv(.c) void {
    gles.texParameterf(target, pname, param);
}
pub fn glTexImage2D(target: gles.GLenum, level: gles.GLint, internalformat: gles.GLint, width: gles.GLsizei, height: gles.GLsizei, border: gles.GLint, format: gles.GLenum, gl_type: gles.GLenum, pixels: ?*const anyopaque) callconv(.c) void {
    gles.texImage2D(target, level, internalformat, width, height, border, format, gl_type, pixels);
}
pub fn glTexImage3D(target: gles.GLenum, level: gles.GLint, internalformat: gles.GLint, width: gles.GLsizei, height: gles.GLsizei, depth: gles.GLsizei, border: gles.GLint, format: gles.GLenum, gl_type: gles.GLenum, pixels: ?*const anyopaque) callconv(.c) void {
    gles.texImage3D(target, level, internalformat, width, height, depth, border, format, gl_type, pixels);
}
pub fn glTexSubImage3D(target: gles.GLenum, level: gles.GLint, xoffset: gles.GLint, yoffset: gles.GLint, zoffset: gles.GLint, width: gles.GLsizei, height: gles.GLsizei, depth: gles.GLsizei, format: gles.GLenum, gl_type: gles.GLenum, pixels: ?*const anyopaque) callconv(.c) void {
    gles.texSubImage3D(target, level, xoffset, yoffset, zoffset, width, height, depth, format, gl_type, pixels);
}
pub fn glTexSubImage2D(target: gles.GLenum, level: gles.GLint, xoffset: gles.GLint, yoffset: gles.GLint, width: gles.GLsizei, height: gles.GLsizei, format: gles.GLenum, gl_type: gles.GLenum, pixels: ?*const anyopaque) callconv(.c) void {
    gles.texSubImage2D(target, level, xoffset, yoffset, width, height, format, gl_type, pixels);
}
pub fn glCompressedTexImage2D(target: gles.GLenum, level: gles.GLint, internalformat: gles.GLenum, width: gles.GLsizei, height: gles.GLsizei, border: gles.GLint, imageSize: gles.GLsizei, data: ?*const anyopaque) callconv(.c) void {
    gles.compressedTexImage2D(target, level, internalformat, width, height, border, imageSize, data);
}
pub fn glReadPixels(x: gles.GLint, y: gles.GLint, width: gles.GLsizei, height: gles.GLsizei, format: gles.GLenum, gl_type: gles.GLenum, pixels: ?*anyopaque) callconv(.c) void {
    gles.readPixels(x, y, width, height, format, gl_type, pixels);
}
pub fn glCopyTexImage2D(target: gles.GLenum, level: gles.GLint, internalformat: gles.GLenum, x: gles.GLint, y: gles.GLint, width: gles.GLsizei, height: gles.GLsizei, border: gles.GLint) callconv(.c) void {
    gles.copyTexImage2D(target, level, @intCast(internalformat), x, y, width, height, border);
}
pub fn glCopyTexSubImage2D(target: gles.GLenum, level: gles.GLint, xoffset: gles.GLint, yoffset: gles.GLint, x: gles.GLint, y: gles.GLint, width: gles.GLsizei, height: gles.GLsizei) callconv(.c) void {
    gles.copyTexSubImage2D(target, level, xoffset, yoffset, x, y, width, height);
}
pub fn glPixelStorei(pname: gles.GLenum, param: gles.GLint) callconv(.c) void {
    gles.pixelStorei(pname, param);
}
pub fn glGenerateMipmap(target: gles.GLenum) callconv(.c) void {
    gles.generateMipmap(target);
}

// --- Framebuffer objects + renderbuffers (render-to-texture) ----------------
pub fn glGenFramebuffers(n: gles.GLsizei, fbs: ?[*]gles.GLuint) callconv(.c) void {
    gles.genFramebuffers(n, fbs);
}
pub fn glBindFramebuffer(target: gles.GLenum, framebuffer: gles.GLuint) callconv(.c) void {
    gles.bindFramebuffer(target, framebuffer);
}
pub fn glDeleteFramebuffers(n: gles.GLsizei, fbs: ?[*]const gles.GLuint) callconv(.c) void {
    gles.deleteFramebuffers(n, fbs);
}
pub fn glIsFramebuffer(framebuffer: gles.GLuint) callconv(.c) gles.GLboolean {
    return gles.isFramebuffer(framebuffer);
}
pub fn glFramebufferTexture2D(target: gles.GLenum, attachment: gles.GLenum, textarget: gles.GLenum, texture: gles.GLuint, level: gles.GLint) callconv(.c) void {
    gles.framebufferTexture2D(target, attachment, textarget, texture, level);
}
pub fn glFramebufferRenderbuffer(target: gles.GLenum, attachment: gles.GLenum, rbtarget: gles.GLenum, renderbuffer: gles.GLuint) callconv(.c) void {
    gles.framebufferRenderbuffer(target, attachment, rbtarget, renderbuffer);
}
pub fn glCheckFramebufferStatus(target: gles.GLenum) callconv(.c) gles.GLenum {
    return gles.checkFramebufferStatus(target);
}
pub fn glGenRenderbuffers(n: gles.GLsizei, rbs: ?[*]gles.GLuint) callconv(.c) void {
    gles.genRenderbuffers(n, rbs);
}
pub fn glBindRenderbuffer(target: gles.GLenum, renderbuffer: gles.GLuint) callconv(.c) void {
    gles.bindRenderbuffer(target, renderbuffer);
}
pub fn glRenderbufferStorage(target: gles.GLenum, internalformat: gles.GLenum, width: gles.GLsizei, height: gles.GLsizei) callconv(.c) void {
    gles.renderbufferStorage(target, internalformat, width, height);
}
pub fn glRenderbufferStorageMultisample(target: gles.GLenum, samples: gles.GLsizei, internalformat: gles.GLenum, width: gles.GLsizei, height: gles.GLsizei) callconv(.c) void {
    gles.renderbufferStorageMultisample(target, samples, internalformat, width, height);
}
pub fn glDeleteRenderbuffers(n: gles.GLsizei, rbs: ?[*]const gles.GLuint) callconv(.c) void {
    gles.deleteRenderbuffers(n, rbs);
}
pub fn glIsRenderbuffer(renderbuffer: gles.GLuint) callconv(.c) gles.GLboolean {
    return gles.isRenderbuffer(renderbuffer);
}

// --- GL_OES_mapbuffer -------------------------------------------------------
pub fn glMapBufferOES(target: gles.GLenum, access: gles.GLenum) callconv(.c) ?*anyopaque {
    return gles.mapBufferOES(target, access);
}
pub fn glUnmapBufferOES(target: gles.GLenum) callconv(.c) gles.GLboolean {
    return gles.unmapBufferOES(target);
}
pub fn glMapBufferRange(target: gles.GLenum, offset: gles.GLintptr, length: gles.GLsizeiptr, access: gles.GLbitfield) callconv(.c) ?*anyopaque {
    return gles.mapBufferRange(target, offset, length, access);
}
pub fn glUnmapBuffer(target: gles.GLenum) callconv(.c) gles.GLboolean {
    return gles.unmapBuffer(target);
}
pub fn glFlushMappedBufferRange(target: gles.GLenum, offset: gles.GLintptr, length: gles.GLsizeiptr) callconv(.c) void {
    gles.flushMappedBufferRange(target, offset, length);
}
pub fn glDrawRangeElements(mode: gles.GLenum, start: gles.GLuint, end: gles.GLuint, count: gles.GLsizei, index_type: gles.GLenum, indices: ?*const anyopaque) callconv(.c) void {
    gles.drawRangeElements(mode, start, end, count, index_type, @intFromPtr(indices));
}

// --- eglGetProcAddress ------------------------------------------------------

fn asProc(comptime f: anytype) egl.ProcFn {
    return @ptrCast(&f);
}

/// Name to entry-point table. eglGetProcAddress and the GLVND getProcAddress/
/// getDispatchAddress imports all resolve through here.
fn lookupProc(name: []const u8) egl.ProcFn {
    const map = .{
        .{ "eglGetError", asProc(eglGetError) },
        .{ "eglGetDisplay", asProc(eglGetDisplay) },
        .{ "eglGetPlatformDisplay", asProc(eglGetPlatformDisplay) },
        .{ "eglGetPlatformDisplayEXT", asProc(eglGetPlatformDisplayEXT) },
        .{ "eglInitialize", asProc(eglInitialize) },
        .{ "eglTerminate", asProc(eglTerminate) },
        .{ "eglQueryString", asProc(eglQueryString) },
        .{ "eglGetConfigs", asProc(eglGetConfigs) },
        .{ "eglChooseConfig", asProc(eglChooseConfig) },
        .{ "eglGetConfigAttrib", asProc(eglGetConfigAttrib) },
        .{ "eglBindAPI", asProc(eglBindAPI) },
        .{ "eglQueryAPI", asProc(eglQueryAPI) },
        .{ "eglReleaseThread", asProc(eglReleaseThread) },
        .{ "eglGetProcAddress", asProc(eglGetProcAddress) },
        // M2 stubs (resolvable so apps can bind them). They return errors until M2 lands.
        .{ "eglCreateContext", asProc(eglCreateContext) },
        .{ "eglDestroyContext", asProc(eglDestroyContext) },
        .{ "eglCreateWindowSurface", asProc(eglCreateWindowSurface) },
        .{ "eglCreatePbufferSurface", asProc(eglCreatePbufferSurface) },
        .{ "eglDestroySurface", asProc(eglDestroySurface) },
        .{ "eglMakeCurrent", asProc(eglMakeCurrent) },
        .{ "eglSwapBuffers", asProc(eglSwapBuffers) },
        // The remaining EGL 1.0-1.5 static entry points (GLVND requires the full set).
        .{ "eglGetCurrentContext", asProc(eglGetCurrentContext) },
        .{ "eglGetCurrentDisplay", asProc(eglGetCurrentDisplay) },
        .{ "eglGetCurrentSurface", asProc(eglGetCurrentSurface) },
        .{ "eglQueryContext", asProc(eglQueryContext) },
        .{ "eglQuerySurface", asProc(eglQuerySurface) },
        .{ "eglSurfaceAttrib", asProc(eglSurfaceAttrib) },
        .{ "eglSwapInterval", asProc(eglSwapInterval) },
        .{ "eglBindTexImage", asProc(eglBindTexImage) },
        .{ "eglReleaseTexImage", asProc(eglReleaseTexImage) },
        .{ "eglCopyBuffers", asProc(eglCopyBuffers) },
        .{ "eglCreatePixmapSurface", asProc(eglCreatePixmapSurface) },
        .{ "eglCreatePlatformWindowSurface", asProc(eglCreatePlatformWindowSurface) },
        .{ "eglCreatePlatformPixmapSurface", asProc(eglCreatePlatformPixmapSurface) },
        .{ "eglCreatePbufferFromClientBuffer", asProc(eglCreatePbufferFromClientBuffer) },
        .{ "eglWaitGL", asProc(eglWaitGL) },
        .{ "eglWaitClient", asProc(eglWaitClient) },
        .{ "eglWaitNative", asProc(eglWaitNative) },
        .{ "eglCreateSync", asProc(eglCreateSync) },
        .{ "eglDestroySync", asProc(eglDestroySync) },
        .{ "eglClientWaitSync", asProc(eglClientWaitSync) },
        .{ "eglWaitSync", asProc(eglWaitSync) },
        .{ "eglGetSyncAttrib", asProc(eglGetSyncAttrib) },
        .{ "eglCreateImage", asProc(eglCreateImage) },
        .{ "eglDestroyImage", asProc(eglDestroyImage) },
        // The minimal GLES clear path (resolvable via eglGetProcAddress).
        .{ "glClearColor", asProc(glClearColor) },
        .{ "glClear", asProc(glClear) },
        .{ "glViewport", asProc(glViewport) },
        .{ "glScissor", asProc(glScissor) },
        .{ "glLineWidth", asProc(glLineWidth) },
        .{ "glSampleCoverage", asProc(glSampleCoverage) },
        .{ "glGetString", asProc(glGetString) },
        .{ "glGetStringi", asProc(glGetStringi) },
        .{ "glTexStorage2D", asProc(glTexStorage2D) },
        .{ "glTexStorage2DEXT", asProc(glTexStorage2D) },
        .{ "glTexStorage3D", asProc(glTexStorage3D) },
        .{ "glInvalidateFramebuffer", asProc(glInvalidateFramebuffer) },
        .{ "glInvalidateSubFramebuffer", asProc(glInvalidateSubFramebuffer) },
        .{ "glFenceSync", asProc(glFenceSync) },
        .{ "glClientWaitSync", asProc(glClientWaitSync) },
        .{ "glWaitSync", asProc(glWaitSync) },
        .{ "glDeleteSync", asProc(glDeleteSync) },
        .{ "glIsSync", asProc(glIsSync) },
        .{ "glGetSynciv", asProc(glGetSynciv) },
        .{ "glBlitFramebuffer", asProc(glBlitFramebuffer) },
        .{ "glClearBufferfv", asProc(glClearBufferfv) },
        .{ "glClearBufferiv", asProc(glClearBufferiv) },
        .{ "glClearBufferuiv", asProc(glClearBufferuiv) },
        .{ "glClearBufferfi", asProc(glClearBufferfi) },
        .{ "glCopyBufferSubData", asProc(glCopyBufferSubData) },
        .{ "glReadBuffer", asProc(glReadBuffer) },
        .{ "glDrawBuffers", asProc(glDrawBuffers) },
        .{ "glGetInternalformativ", asProc(glGetInternalformativ) },
        .{ "glGenQueries", asProc(glGenQueries) },
        .{ "glDeleteQueries", asProc(glDeleteQueries) },
        .{ "glIsQuery", asProc(glIsQuery) },
        .{ "glBeginQuery", asProc(glBeginQuery) },
        .{ "glEndQuery", asProc(glEndQuery) },
        .{ "glGetQueryObjectuiv", asProc(glGetQueryObjectuiv) },
        .{ "glGetQueryObjectiv", asProc(glGetQueryObjectiv) },
        .{ "glGetBufferSubData", asProc(glGetBufferSubData) },
        .{ "glGetShaderSource", asProc(glGetShaderSource) },
        .{ "glDepthRangef", asProc(glDepthRangef) },
        .{ "glReleaseShaderCompiler", asProc(glReleaseShaderCompiler) },
        .{ "glHint", asProc(glHint) },
        .{ "glGetIntegerv", asProc(glGetIntegerv) },
        .{ "glGetFloatv", asProc(glGetFloatv) },
        .{ "glGetBooleanv", asProc(glGetBooleanv) },
        .{ "glIsEnabled", asProc(glIsEnabled) },
        .{ "glGetShaderPrecisionFormat", asProc(glGetShaderPrecisionFormat) },
        .{ "glGetTexParameteriv", asProc(glGetTexParameteriv) },
        .{ "glGetTexParameterfv", asProc(glGetTexParameterfv) },
        .{ "glGetVertexAttribiv", asProc(glGetVertexAttribiv) },
        .{ "glGetVertexAttribfv", asProc(glGetVertexAttribfv) },
        .{ "glGetBufferParameteriv", asProc(glGetBufferParameteriv) },
        .{ "glGetRenderbufferParameteriv", asProc(glGetRenderbufferParameteriv) },
        .{ "glGetFramebufferAttachmentParameteriv", asProc(glGetFramebufferAttachmentParameteriv) },
        .{ "glGetAttachedShaders", asProc(glGetAttachedShaders) },
        .{ "glGetUniformfv", asProc(glGetUniformfv) },
        .{ "glGetUniformiv", asProc(glGetUniformiv) },
        .{ "glGetVertexAttribPointerv", asProc(glGetVertexAttribPointerv) },
        .{ "glGetError", asProc(glGetError) },
        .{ "glFinish", asProc(glFinish) },
        .{ "glFlush", asProc(glFlush) },
        // The GLES2 triangle path (shaders + programs + buffers + attribs + draw).
        .{ "glGenBuffers", asProc(glGenBuffers) },
        .{ "glBindBuffer", asProc(glBindBuffer) },
        .{ "glBufferData", asProc(glBufferData) },
        .{ "glDeleteBuffers", asProc(glDeleteBuffers) },
        .{ "glVertexAttribPointer", asProc(glVertexAttribPointer) },
        .{ "glVertexAttribIPointer", asProc(glVertexAttribIPointer) },
        .{ "glEnableVertexAttribArray", asProc(glEnableVertexAttribArray) },
        .{ "glDisableVertexAttribArray", asProc(glDisableVertexAttribArray) },
        .{ "glCreateShader", asProc(glCreateShader) },
        .{ "glShaderSource", asProc(glShaderSource) },
        .{ "glShaderBinary", asProc(glShaderBinary) },
        .{ "glCompileShader", asProc(glCompileShader) },
        .{ "glSpecializeShader", asProc(glSpecializeShader) },
        .{ "glGetShaderiv", asProc(glGetShaderiv) },
        .{ "glGetShaderInfoLog", asProc(glGetShaderInfoLog) },
        .{ "glDeleteShader", asProc(glDeleteShader) },
        .{ "glCreateProgram", asProc(glCreateProgram) },
        .{ "glAttachShader", asProc(glAttachShader) },
        .{ "glDetachShader", asProc(glDetachShader) },
        .{ "glLinkProgram", asProc(glLinkProgram) },
        .{ "glUseProgram", asProc(glUseProgram) },
        .{ "glGetProgramiv", asProc(glGetProgramiv) },
        .{ "glDeleteProgram", asProc(glDeleteProgram) },
        .{ "glGetAttribLocation", asProc(glGetAttribLocation) },
        .{ "glDrawArrays", asProc(glDrawArrays) },
        .{ "glDrawElements", asProc(glDrawElements) },
        .{ "glDrawArraysInstanced", asProc(glDrawArraysInstanced) },
        .{ "glDrawElementsInstanced", asProc(glDrawElementsInstanced) },
        .{ "glVertexAttribDivisor", asProc(glVertexAttribDivisor) },
        .{ "glVertexAttrib1f", asProc(glVertexAttrib1f) },
        .{ "glVertexAttrib2f", asProc(glVertexAttrib2f) },
        .{ "glVertexAttrib3f", asProc(glVertexAttrib3f) },
        .{ "glVertexAttrib4f", asProc(glVertexAttrib4f) },
        .{ "glVertexAttrib1fv", asProc(glVertexAttrib1fv) },
        .{ "glVertexAttrib2fv", asProc(glVertexAttrib2fv) },
        .{ "glVertexAttrib3fv", asProc(glVertexAttrib3fv) },
        .{ "glVertexAttrib4fv", asProc(glVertexAttrib4fv) },
        .{ "glGenVertexArrays", asProc(glGenVertexArrays) },
        .{ "glBindVertexArray", asProc(glBindVertexArray) },
        .{ "glDeleteVertexArrays", asProc(glDeleteVertexArrays) },
        .{ "glIsVertexArray", asProc(glIsVertexArray) },
        .{ "glGenVertexArraysOES", asProc(glGenVertexArrays) },
        .{ "glBindVertexArrayOES", asProc(glBindVertexArray) },
        .{ "glDeleteVertexArraysOES", asProc(glDeleteVertexArrays) },
        .{ "glIsVertexArrayOES", asProc(glIsVertexArray) },
        // Uniforms + fixed-function state (the es2gears milestone).
        .{ "glBindAttribLocation", asProc(glBindAttribLocation) },
        // GLES3 uniform buffer objects (named uniform blocks).
        .{ "glGetUniformBlockIndex", asProc(glGetUniformBlockIndex) },
        .{ "glUniformBlockBinding", asProc(glUniformBlockBinding) },
        .{ "glGetActiveUniformBlockiv", asProc(glGetActiveUniformBlockiv) },
        .{ "glGetUniformIndices", asProc(glGetUniformIndices) },
        .{ "glGetActiveUniformsiv", asProc(glGetActiveUniformsiv) },
        .{ "glBindBufferBase", asProc(glBindBufferBase) },
        .{ "glBindBufferRange", asProc(glBindBufferRange) },
        // Transform feedback (GLES3).
        .{ "glTransformFeedbackVaryings", asProc(glTransformFeedbackVaryings) },
        .{ "glBeginTransformFeedback", asProc(glBeginTransformFeedback) },
        .{ "glEndTransformFeedback", asProc(glEndTransformFeedback) },
        .{ "glPauseTransformFeedback", asProc(glPauseTransformFeedback) },
        .{ "glResumeTransformFeedback", asProc(glResumeTransformFeedback) },
        .{ "glGetUniformLocation", asProc(glGetUniformLocation) },
        .{ "glGetActiveUniform", asProc(glGetActiveUniform) },
        .{ "glGetActiveAttrib", asProc(glGetActiveAttrib) },
        .{ "glGetProgramInfoLog", asProc(glGetProgramInfoLog) },
        .{ "glUniform1f", asProc(glUniform1f) },
        .{ "glUniform2f", asProc(glUniform2f) },
        .{ "glUniform3f", asProc(glUniform3f) },
        .{ "glUniform4f", asProc(glUniform4f) },
        .{ "glUniform1i", asProc(glUniform1i) },
        .{ "glUniform1fv", asProc(glUniform1fv) },
        .{ "glUniform2fv", asProc(glUniform2fv) },
        .{ "glUniform3fv", asProc(glUniform3fv) },
        .{ "glUniform4fv", asProc(glUniform4fv) },
        .{ "glUniformMatrix2fv", asProc(glUniformMatrix2fv) },
        .{ "glUniformMatrix3fv", asProc(glUniformMatrix3fv) },
        .{ "glUniformMatrix4fv", asProc(glUniformMatrix4fv) },
        .{ "glEnable", asProc(glEnable) },
        .{ "glDisable", asProc(glDisable) },
        .{ "glDepthFunc", asProc(glDepthFunc) },
        .{ "glDepthMask", asProc(glDepthMask) },
        .{ "glPolygonOffset", asProc(glPolygonOffset) },
        .{ "glStencilFunc", asProc(glStencilFunc) },
        .{ "glStencilOp", asProc(glStencilOp) },
        .{ "glStencilMask", asProc(glStencilMask) },
        .{ "glStencilFuncSeparate", asProc(glStencilFuncSeparate) },
        .{ "glStencilOpSeparate", asProc(glStencilOpSeparate) },
        .{ "glStencilMaskSeparate", asProc(glStencilMaskSeparate) },
        .{ "glClearStencil", asProc(glClearStencil) },
        .{ "glClearDepthf", asProc(glClearDepthf) },
        .{ "glCullFace", asProc(glCullFace) },
        .{ "glFrontFace", asProc(glFrontFace) },
        // Alpha blending (glmark2 desktop/effect2d composite translucent layers).
        .{ "glBlendFunc", asProc(glBlendFunc) },
        .{ "glBlendFuncSeparate", asProc(glBlendFuncSeparate) },
        .{ "glBlendEquation", asProc(glBlendEquation) },
        .{ "glBlendEquationSeparate", asProc(glBlendEquationSeparate) },
        .{ "glBlendColor", asProc(glBlendColor) },
        .{ "glColorMask", asProc(glColorMask) },
        // Textures / samplers (the glmark2-es2 texture milestone).
        .{ "glGenTextures", asProc(glGenTextures) },
        .{ "glDeleteTextures", asProc(glDeleteTextures) },
        .{ "glBindTexture", asProc(glBindTexture) },
        .{ "glGenSamplers", asProc(glGenSamplers) },
        .{ "glDeleteSamplers", asProc(glDeleteSamplers) },
        .{ "glIsSampler", asProc(glIsSampler) },
        .{ "glBindSampler", asProc(glBindSampler) },
        .{ "glSamplerParameteri", asProc(glSamplerParameteri) },
        .{ "glSamplerParameterf", asProc(glSamplerParameterf) },
        .{ "glGetSamplerParameteriv", asProc(glGetSamplerParameteriv) },
        .{ "glActiveTexture", asProc(glActiveTexture) },
        .{ "glIsTexture", asProc(glIsTexture) },
        .{ "glTexParameteri", asProc(glTexParameteri) },
        .{ "glTexParameterf", asProc(glTexParameterf) },
        .{ "glTexImage2D", asProc(glTexImage2D) },
        .{ "glTexImage3D", asProc(glTexImage3D) },
        .{ "glTexSubImage3D", asProc(glTexSubImage3D) },
        .{ "glCompressedTexImage2D", asProc(glCompressedTexImage2D) },
        .{ "glTexSubImage2D", asProc(glTexSubImage2D) },
        .{ "glReadPixels", asProc(glReadPixels) },
        .{ "glCopyTexImage2D", asProc(glCopyTexImage2D) },
        .{ "glCopyTexSubImage2D", asProc(glCopyTexSubImage2D) },
        .{ "glPixelStorei", asProc(glPixelStorei) },
        .{ "glGenerateMipmap", asProc(glGenerateMipmap) },
        // Framebuffer objects / renderbuffers (render-to-texture: glmark2 shadow/refract).
        .{ "glGenFramebuffers", asProc(glGenFramebuffers) },
        .{ "glBindFramebuffer", asProc(glBindFramebuffer) },
        .{ "glDeleteFramebuffers", asProc(glDeleteFramebuffers) },
        .{ "glIsFramebuffer", asProc(glIsFramebuffer) },
        .{ "glFramebufferTexture2D", asProc(glFramebufferTexture2D) },
        .{ "glFramebufferRenderbuffer", asProc(glFramebufferRenderbuffer) },
        .{ "glCheckFramebufferStatus", asProc(glCheckFramebufferStatus) },
        .{ "glGenRenderbuffers", asProc(glGenRenderbuffers) },
        .{ "glBindRenderbuffer", asProc(glBindRenderbuffer) },
        .{ "glRenderbufferStorage", asProc(glRenderbufferStorage) },
        .{ "glRenderbufferStorageMultisample", asProc(glRenderbufferStorageMultisample) },
        .{ "glDeleteRenderbuffers", asProc(glDeleteRenderbuffers) },
        .{ "glIsRenderbuffer", asProc(glIsRenderbuffer) },
        // GL_OES_mapbuffer (glmark2 buffer:update-method=map).
        .{ "glMapBufferOES", asProc(glMapBufferOES) },
        .{ "glMapBufferRange", asProc(glMapBufferRange) },
        .{ "glUnmapBuffer", asProc(glUnmapBuffer) },
        .{ "glFlushMappedBufferRange", asProc(glFlushMappedBufferRange) },
        .{ "glDrawRangeElements", asProc(glDrawRangeElements) },
        .{ "glUnmapBufferOES", asProc(glUnmapBufferOES) },
    };
    inline for (map) |entry| {
        if (std.mem.eql(u8, name, entry[0])) return entry[1];
    }
    return null;
}

pub fn eglGetProcAddress(procname: ?[*:0]const u8) callconv(.c) egl.ProcFn {
    const name = std.mem.span(procname orelse return null);
    return lookupProc(name);
}

// libglvnd EGL vendor ABI: the imports table + __egl_Main

fn imp_getPlatformDisplay(
    platform: EGLenum,
    nativeDisplay: ?*anyopaque,
    attrib_list: ?[*]const egl.EGLAttrib,
) callconv(.c) EGLDisplay {
    return eglGetPlatformDisplay(platform, nativeDisplay, attrib_list);
}

fn imp_getSupportsAPI(api: EGLenum) callconv(.c) EGLBoolean {
    return if (state.supportsAPI(api)) egl.EGL_TRUE else egl.EGL_FALSE;
}

fn imp_getVendorString(name: c_int) callconv(.c) ?[*:0]const u8 {
    return switch (name) {
        egl.VENDOR_STRING_PLATFORM_EXTENSIONS => state.platform_extensions_string,
        else => null,
    };
}

fn imp_getProcAddress(procName: ?[*:0]const u8) callconv(.c) ?*anyopaque {
    const name = std.mem.span(procName orelse return null);
    return @ptrCast(@constCast(lookupProc(name)));
}

/// We dispatch nothing through libEGL's dynamic table for M1 (no display-level
/// extension functions beyond the static set), so getDispatchAddress returns null
/// and setDispatchIndex is a no-op.
fn imp_getDispatchAddress(procName: ?[*:0]const u8) callconv(.c) ?*anyopaque {
    _ = procName;
    return null;
}

fn imp_setDispatchIndex(procName: ?[*:0]const u8, index: c_int) callconv(.c) void {
    _ = procName;
    _ = index;
}

/// libglvnd calls this for native displays in eglGetDisplay to determine the
/// platform. We don't sniff native handles (no false matches, per the contract):
/// return EGL_NONE so libglvnd falls back to its own detection.
fn imp_findNativeDisplayPlatform(native_display: ?*anyopaque) callconv(.c) EGLenum {
    _ = native_display;
    return @intCast(egl.EGL_NONE);
}

/// Fill in the GLVND imports table. Shared by __egl_Main and the tests.
pub fn fillImports(imports: *egl.EGLapiImports) void {
    imports.* = .{
        .getPlatformDisplay = imp_getPlatformDisplay,
        .getSupportsAPI = imp_getSupportsAPI,
        .getVendorString = imp_getVendorString,
        .getProcAddress = imp_getProcAddress,
        .getDispatchAddress = imp_getDispatchAddress,
        .setDispatchIndex = imp_setDispatchIndex,
        // Optional hooks left null (the entrypoint-patching optimization).
        .isPatchSupported = null,
        .initiatePatch = null,
        .releasePatch = null,
        .patchThreadAttach = null,
        .findNativeDisplayPlatform = imp_findNativeDisplayPlatform,
    };
}

/// GLVND EGL vendor entry point. libEGL.so dlopen's us, resolves "__egl_Main",
/// and calls it with its exports table plus a slot for our imports. Requires ABI
/// major version 0 (libglvnd 1.7.0 = 0.2).
pub fn eglMain(
    version: u32,
    exports: ?*const egl.EGLapiExports,
    vendor: ?*egl.EGLvendorInfo,
    imports: ?*egl.EGLapiImports,
) callconv(.c) EGLBoolean {
    _ = vendor;
    if (debugEnabled()) {
        std.debug.print("[prism-egl] __egl_Main version=0x{x} (major={d} minor={d})\n", .{ version, egl.abiMajor(version), egl.abiMinor(version) });
    }
    // Major-version handshake: we implement major 0. A different major breaks ABI.
    if (egl.abiMajor(version) != egl.EGL_VENDOR_ABI_MAJOR_VERSION) return egl.EGL_FALSE;
    const imp = imports orelse return egl.EGL_FALSE;
    api_exports = exports;
    fillImports(imp);
    return egl.EGL_TRUE;
}

// Tests

test "eglGetError resets to SUCCESS after a read" {
    setError(egl.EGL_BAD_DISPLAY);
    try std.testing.expectEqual(egl.EGL_BAD_DISPLAY, eglGetError());
    try std.testing.expectEqual(egl.EGL_SUCCESS, eglGetError());
}

test "enumeration path end to end: getDisplay -> initialize -> query -> configs" {
    const dpy = eglGetPlatformDisplay(egl.EGL_PLATFORM_SURFACELESS_MESA, null, null);
    try std.testing.expect(dpy != null);

    var major: EGLint = 0;
    var minor: EGLint = 0;
    try std.testing.expectEqual(egl.EGL_TRUE, eglInitialize(dpy, &major, &minor));
    try std.testing.expectEqual(@as(EGLint, 1), major);
    try std.testing.expectEqual(@as(EGLint, 5), minor);

    try std.testing.expectEqualStrings("Prism", std.mem.span(eglQueryString(dpy, egl.EGL_VENDOR).?));
    try std.testing.expectEqualStrings("1.5", std.mem.span(eglQueryString(dpy, egl.EGL_VERSION).?));

    var n: EGLint = 0;
    try std.testing.expectEqual(egl.EGL_TRUE, eglGetConfigs(dpy, null, 0, &n));
    try std.testing.expectEqual(@as(EGLint, @intCast(state.configs.len)), n);

    var got: [8]EGLConfig = undefined;
    var n2: EGLint = 0;
    try std.testing.expectEqual(egl.EGL_TRUE, eglGetConfigs(dpy, &got, 8, &n2));
    try std.testing.expectEqual(n, n2);

    // eglGetConfigAttrib on the first config.
    var red: EGLint = 0;
    try std.testing.expectEqual(egl.EGL_TRUE, eglGetConfigAttrib(dpy, got[0], egl.EGL_RED_SIZE, &red));
    try std.testing.expectEqual(@as(EGLint, 8), red);

    try std.testing.expectEqual(egl.EGL_TRUE, eglTerminate(dpy));
}

test "eglQueryString on an uninitialized display errors NOT_INITIALIZED" {
    // A fresh display the test owns (wayland platform, never initialized here).
    const dpy = eglGetPlatformDisplay(egl.EGL_PLATFORM_WAYLAND_KHR, @ptrFromInt(0xabc), null);
    try std.testing.expect(dpy != null);
    try std.testing.expectEqual(@as(?[*:0]const u8, null), eglQueryString(dpy, egl.EGL_VENDOR));
    try std.testing.expectEqual(egl.EGL_NOT_INITIALIZED, eglGetError());
}

test "eglGetConfigAttrib rejects bad config and bad attribute" {
    const dpy = eglGetPlatformDisplay(egl.EGL_PLATFORM_SURFACELESS_MESA, @ptrFromInt(0x1), null);
    _ = eglInitialize(dpy, null, null);
    var v: EGLint = 0;
    try std.testing.expectEqual(egl.EGL_FALSE, eglGetConfigAttrib(dpy, @ptrFromInt(0xdead), egl.EGL_RED_SIZE, &v));
    try std.testing.expectEqual(egl.EGL_BAD_CONFIG, eglGetError());
    const valid = state.configHandle(0);
    try std.testing.expectEqual(egl.EGL_FALSE, eglGetConfigAttrib(dpy, valid, 0x9999, &v));
    try std.testing.expectEqual(egl.EGL_BAD_ATTRIBUTE, eglGetError());
}

test "eglChooseConfig through the C ABI filters on depth" {
    const dpy = eglGetPlatformDisplay(egl.EGL_PLATFORM_SURFACELESS_MESA, @ptrFromInt(0x2), null);
    _ = eglInitialize(dpy, null, null);
    const list = [_]EGLint{ egl.EGL_DEPTH_SIZE, 24, egl.EGL_NONE };
    var out: [8]EGLConfig = undefined;
    var n: EGLint = 0;
    try std.testing.expectEqual(egl.EGL_TRUE, eglChooseConfig(dpy, &list, &out, 8, &n));
    try std.testing.expectEqual(@as(EGLint, 4), n); // configs 2, 3, and the two MSAA configs (5, 6)
}

test "eglBindAPI / eglQueryAPI" {
    try std.testing.expectEqual(egl.EGL_TRUE, eglBindAPI(egl.EGL_OPENGL_API));
    try std.testing.expectEqual(egl.EGL_OPENGL_API, eglQueryAPI());
    try std.testing.expectEqual(egl.EGL_FALSE, eglBindAPI(egl.EGL_OPENVG_API));
    try std.testing.expectEqual(egl.EGL_BAD_PARAMETER, eglGetError());
    _ = eglBindAPI(egl.EGL_OPENGL_ES_API);
}

test "eglGetProcAddress resolves core entry points" {
    try std.testing.expect(eglGetProcAddress("eglInitialize") != null);
    try std.testing.expect(eglGetProcAddress("eglChooseConfig") != null);
    try std.testing.expect(eglGetProcAddress("eglNonexistent") == null);
}

test "eglGetProcAddress resolves the GLES2 state-query entry points (the loader-facing wiring)" {
    // The query/introspection API a stock app resolves through the GLVND loader. A missing
    // entry here means the app silently gets a null pointer and skips (or crashes on) the call,
    // so this guards that every wrapper added to the proc table stays reachable.
    const names = [_][]const u8{
        "glGetIntegerv",                "glGetFloatv",
        "glGetBooleanv",                "glIsEnabled",
        "glGetShaderPrecisionFormat",   "glGetTexParameteriv",
        "glGetTexParameterfv",          "glGetVertexAttribiv",
        "glGetVertexAttribfv",          "glGetBufferParameteriv",
        "glGetRenderbufferParameteriv", "glGetFramebufferAttachmentParameteriv",
        "glGetAttachedShaders",         "glGetUniformfv",
        "glGetUniformiv",               "glGetVertexAttribPointerv",
        "glReadPixels",                 "glGetString",
    };
    for (names) |n| {
        const z = try std.testing.allocator.dupeZ(u8, n);
        defer std.testing.allocator.free(z);
        try std.testing.expect(eglGetProcAddress(z.ptr) != null);
    }
}

test "the FULL EGL 1.0-1.5 static entry-point set resolves (the GLVND vendor requirement)" {
    // GLVND builds its static dispatch table by asking getProcAddress for every
    // one of these at load time and discards a vendor missing any. This is the
    // exact set from EGL/egl.h. A regression here breaks GLVND-loader vendor
    // acceptance (the bug that initially blocked enumeration through libEGL.so).
    const required = [_][]const u8{
        "eglBindAPI",                     "eglBindTexImage",
        "eglChooseConfig",                "eglClientWaitSync",
        "eglCopyBuffers",                 "eglCreateContext",
        "eglCreateImage",                 "eglCreatePbufferFromClientBuffer",
        "eglCreatePbufferSurface",        "eglCreatePixmapSurface",
        "eglCreatePlatformPixmapSurface", "eglCreatePlatformWindowSurface",
        "eglCreateSync",                  "eglCreateWindowSurface",
        "eglDestroyContext",              "eglDestroyImage",
        "eglDestroySurface",              "eglDestroySync",
        "eglGetConfigAttrib",             "eglGetConfigs",
        "eglGetCurrentContext",           "eglGetCurrentDisplay",
        "eglGetCurrentSurface",           "eglGetDisplay",
        "eglGetError",                    "eglGetPlatformDisplay",
        "eglGetProcAddress",              "eglGetSyncAttrib",
        "eglInitialize",                  "eglMakeCurrent",
        "eglQueryAPI",                    "eglQueryContext",
        "eglQueryString",                 "eglQuerySurface",
        "eglReleaseTexImage",             "eglReleaseThread",
        "eglSurfaceAttrib",               "eglSwapBuffers",
        "eglSwapInterval",                "eglTerminate",
        "eglWaitClient",                  "eglWaitGL",
        "eglWaitNative",                  "eglWaitSync",
    };
    for (required) |name| {
        if (lookupProc(name) == null) {
            std.debug.print("missing required EGL entry point: {s}\n", .{name});
            return error.MissingEntryPoint;
        }
    }
}

test "__egl_Main fills the GLVND imports table and rejects a wrong major" {
    var imports: egl.EGLapiImports = undefined;
    const ok = eglMain((egl.EGL_VENDOR_ABI_MAJOR_VERSION << 16) | 2, null, null, &imports);
    try std.testing.expectEqual(egl.EGL_TRUE, ok);
    try std.testing.expect(imports.getPlatformDisplay != null);
    try std.testing.expect(imports.getProcAddress != null);
    try std.testing.expect(imports.getSupportsAPI != null);
    try std.testing.expect(imports.getVendorString != null);
    try std.testing.expect(imports.setDispatchIndex != null);
    // getSupportsAPI works.
    try std.testing.expectEqual(egl.EGL_TRUE, imports.getSupportsAPI.?(egl.EGL_OPENGL_ES_API));
    // A major-version mismatch is rejected.
    var imports2: egl.EGLapiImports = undefined;
    try std.testing.expectEqual(egl.EGL_FALSE, eglMain((1 << 16) | 0, null, null, &imports2));
}

// EGL M2: the render path (context + surface + GLES clear + swap)

// Deterministic oracle: an EGL client creates a context + a pbuffer surface,
// glClearColor to a known color, glClear, then reads the pbuffer's HAL backbuffer
// and asserts the exact clear color landed. Proves the full EGL -> GLES -> HAL
// (software driver) clear path end to end. No display needed.
test "EGL pbuffer clear reads back the exact color (EGL -> GLES -> HAL render path)" {
    _ = eglBindAPI(egl.EGL_OPENGL_ES_API);

    // A fresh display the test owns (unique native handle so it isn't shared with
    // other tests in this process), brought up.
    const dpy = eglGetPlatformDisplay(egl.EGL_PLATFORM_SURFACELESS_MESA, @ptrFromInt(0xE61C1), null);
    try std.testing.expect(dpy != null);
    try std.testing.expectEqual(egl.EGL_TRUE, eglInitialize(dpy, null, null));
    defer _ = eglTerminate(dpy);

    // Choose an RGBA8 config.
    const cfg = state.configHandle(0);

    // Context.
    const ctx = eglCreateContext(dpy, cfg, egl.EGL_NO_CONTEXT, null);
    try std.testing.expect(ctx != egl.EGL_NO_CONTEXT);

    // 64x64 pbuffer surface.
    const W: EGLint = 64;
    const H: EGLint = 64;
    const pb_attribs = [_]EGLint{ egl.EGL_WIDTH, W, egl.EGL_HEIGHT, H, egl.EGL_NONE };
    const surf = eglCreatePbufferSurface(dpy, cfg, &pb_attribs);
    try std.testing.expect(surf != egl.EGL_NO_SURFACE);

    // Query surface dims.
    var qw: EGLint = 0;
    var qh: EGLint = 0;
    try std.testing.expectEqual(egl.EGL_TRUE, eglQuerySurface(dpy, surf, egl.EGL_WIDTH, &qw));
    try std.testing.expectEqual(egl.EGL_TRUE, eglQuerySurface(dpy, surf, egl.EGL_HEIGHT, &qh));
    try std.testing.expectEqual(W, qw);
    try std.testing.expectEqual(H, qh);

    // Make current.
    try std.testing.expectEqual(egl.EGL_TRUE, eglMakeCurrent(dpy, surf, surf, ctx));
    try std.testing.expectEqual(ctx, eglGetCurrentContext());
    try std.testing.expectEqual(surf, eglGetCurrentSurface(0x3059)); // EGL_DRAW

    // glClearColor to a known, exact 8-bit-representable color, then glClear.
    // r=64/255, g=128/255, b=192/255, a=255/255 -> bytes 0x40,0x80,0xC0,0xFF.
    glClearColor(64.0 / 255.0, 128.0 / 255.0, 192.0 / 255.0, 1.0);
    glClear(gles.GL_COLOR_BUFFER_BIT);
    try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());

    // Read back the pbuffer's HAL backbuffer via the display's device + map.
    const s: *state.Surface = @ptrCast(@alignCast(surf.?));
    const dev = s.display.device.?;
    const px = try flushMap(dev, s);
    // Center pixel.
    const center = (@as(usize, 32) * 64 + 32) * 4;
    try std.testing.expectEqual(@as(u8, 0x40), px[center + 0]); // R
    try std.testing.expectEqual(@as(u8, 0x80), px[center + 1]); // G
    try std.testing.expectEqual(@as(u8, 0xC0), px[center + 2]); // B
    try std.testing.expectEqual(@as(u8, 0xFF), px[center + 3]); // A
    // Every pixel must be the clear color (a corner too).
    try std.testing.expectEqual(@as(u8, 0x40), px[0]);
    try std.testing.expectEqual(@as(u8, 0xC0), px[2]);

    // eglSwapBuffers on a pbuffer is a no-op success.
    try std.testing.expectEqual(egl.EGL_TRUE, eglSwapBuffers(dpy, surf));

    // Unbind + tear down.
    try std.testing.expectEqual(egl.EGL_TRUE, eglMakeCurrent(dpy, null, null, egl.EGL_NO_CONTEXT));
    try std.testing.expectEqual(egl.EGL_NO_CONTEXT, eglGetCurrentContext());
    try std.testing.expectEqual(egl.EGL_TRUE, eglDestroySurface(dpy, surf));
    try std.testing.expectEqual(egl.EGL_TRUE, eglDestroyContext(dpy, ctx));
}

test "glReadPixels + glCopyTexImage2D capture the framebuffer (the post-processing path)" {
    _ = eglBindAPI(egl.EGL_OPENGL_ES_API);
    const dpy = eglGetPlatformDisplay(egl.EGL_PLATFORM_SURFACELESS_MESA, @ptrFromInt(0xE61C2), null);
    try std.testing.expectEqual(egl.EGL_TRUE, eglInitialize(dpy, null, null));
    defer _ = eglTerminate(dpy);
    const cfg = state.configHandle(0);
    const ctx = eglCreateContext(dpy, cfg, egl.EGL_NO_CONTEXT, null);
    const W: EGLint = 64;
    const H: EGLint = 64;
    const pb_attribs = [_]EGLint{ egl.EGL_WIDTH, W, egl.EGL_HEIGHT, H, egl.EGL_NONE };
    const surf = eglCreatePbufferSurface(dpy, cfg, &pb_attribs);
    try std.testing.expectEqual(egl.EGL_TRUE, eglMakeCurrent(dpy, surf, surf, ctx));
    defer _ = eglDestroyContext(dpy, ctx);
    defer _ = eglDestroySurface(dpy, surf);
    defer _ = eglMakeCurrent(dpy, null, null, egl.EGL_NO_CONTEXT);

    // Clear to an exact 8-bit color, then read it back via glReadPixels.
    glClearColor(64.0 / 255.0, 128.0 / 255.0, 192.0 / 255.0, 1.0);
    glClear(gles.GL_COLOR_BUFFER_BIT);
    var buf: [64 * 64 * 4]u8 = undefined;
    glReadPixels(0, 0, 64, 64, gles.GL_RGBA, gles.GL_UNSIGNED_BYTE, &buf);
    try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());
    // Uniform clear: every readback pixel is the clear color (y-flip irrelevant here).
    const ci = (@as(usize, 32) * 64 + 32) * 4;
    try std.testing.expectEqual(@as(u8, 0x40), buf[ci + 0]);
    try std.testing.expectEqual(@as(u8, 0x80), buf[ci + 1]);
    try std.testing.expectEqual(@as(u8, 0xC0), buf[ci + 2]);

    // glReadPixels as GL_RGB drops alpha (3 bytes/pixel).
    var rgb: [64 * 64 * 3]u8 = undefined;
    glReadPixels(0, 0, 64, 64, gles.GL_RGB, gles.GL_UNSIGNED_BYTE, &rgb);
    try std.testing.expectEqual(@as(u8, 0x40), rgb[0]);
    try std.testing.expectEqual(@as(u8, 0x80), rgb[1]);
    try std.testing.expectEqual(@as(u8, 0xC0), rgb[2]);

    // glCopyTexImage2D captures the framebuffer into a texture's base level.
    var tex: gles.GLuint = 0;
    glGenTextures(1, @ptrCast(&tex));
    glBindTexture(gles.GL_TEXTURE_2D, tex);
    glCopyTexImage2D(gles.GL_TEXTURE_2D, 0, gles.GL_RGBA, 0, 0, 64, 64, 0);
    try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());
    const t = gles.findTexture(tex).?;
    try std.testing.expectEqual(@as(u32, 64), t.width);
    try std.testing.expectEqual(@as(u8, 0x40), t.bytes.items[ci + 0]);
    try std.testing.expectEqual(@as(u8, 0x80), t.bytes.items[ci + 1]);
    try std.testing.expectEqual(@as(u8, 0xC0), t.bytes.items[ci + 2]);
}

test "glReadPixels non-8bit: packed-short + half/single-float readback encode the framebuffer color" {
    _ = eglBindAPI(egl.EGL_OPENGL_ES_API);
    const dpy = eglGetPlatformDisplay(egl.EGL_PLATFORM_SURFACELESS_MESA, @ptrFromInt(0xE61C6), null);
    try std.testing.expectEqual(egl.EGL_TRUE, eglInitialize(dpy, null, null));
    defer _ = eglTerminate(dpy);
    const cfg = state.configHandle(0);
    const ctx = eglCreateContext(dpy, cfg, egl.EGL_NO_CONTEXT, null);
    const W: EGLint = 64;
    const pb_attribs = [_]EGLint{ egl.EGL_WIDTH, W, egl.EGL_HEIGHT, W, egl.EGL_NONE };
    const surf = eglCreatePbufferSurface(dpy, cfg, &pb_attribs);
    try std.testing.expectEqual(egl.EGL_TRUE, eglMakeCurrent(dpy, surf, surf, ctx));
    defer _ = eglDestroyContext(dpy, ctx);
    defer _ = eglDestroySurface(dpy, surf);
    defer _ = eglMakeCurrent(dpy, null, null, egl.EGL_NO_CONTEXT);

    // Pure red (opaque): R lands in the high field of each packed layout. Float channels are exactly
    // 1.0 / 0.0. A uniform clear makes the y-flip irrelevant.
    glClearColor(1.0, 0.0, 0.0, 1.0);
    glClear(gles.GL_COLOR_BUFFER_BIT);
    const p: usize = 32 * 64 + 32; // center pixel index

    // RGB565: R=31,G=0,B=0 -> 0xF800 (little-endian bytes {0x00,0xF8}).
    var p565: [64 * 64]u16 = undefined;
    glReadPixels(0, 0, 64, 64, gles.GL_RGB, gles.GL_UNSIGNED_SHORT_5_6_5, &p565);
    try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());
    try std.testing.expectEqual(@as(u16, 0xF800), p565[p]);

    // RGBA4444: R=15,G=0,B=0,A=15 -> 0xF00F.
    var p4444: [64 * 64]u16 = undefined;
    glReadPixels(0, 0, 64, 64, gles.GL_RGBA, gles.GL_UNSIGNED_SHORT_4_4_4_4, &p4444);
    try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());
    try std.testing.expectEqual(@as(u16, 0xF00F), p4444[p]);

    // RGBA5551: R=31,G=0,B=0,A=1 -> 0xF801.
    var p5551: [64 * 64]u16 = undefined;
    glReadPixels(0, 0, 64, 64, gles.GL_RGBA, gles.GL_UNSIGNED_SHORT_5_5_5_1, &p5551);
    try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());
    try std.testing.expectEqual(@as(u16, 0xF801), p5551[p]);

    // HALF_FLOAT_OES RGBA: normalized 1/255 per channel -> R=A=1.0, G=B=0.0.
    var phalf: [64 * 64 * 4]f16 = undefined;
    glReadPixels(0, 0, 64, 64, gles.GL_RGBA, gles.GL_HALF_FLOAT_OES, &phalf);
    try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());
    try std.testing.expectEqual(@as(f16, 1.0), phalf[p * 4 + 0]);
    try std.testing.expectEqual(@as(f16, 0.0), phalf[p * 4 + 1]);
    try std.testing.expectEqual(@as(f16, 0.0), phalf[p * 4 + 2]);
    try std.testing.expectEqual(@as(f16, 1.0), phalf[p * 4 + 3]);

    // FLOAT RGBA: same normalized channels as single-precision.
    var pflt: [64 * 64 * 4]f32 = undefined;
    glReadPixels(0, 0, 64, 64, gles.GL_RGBA, gles.GL_FLOAT, &pflt);
    try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());
    try std.testing.expectEqual(@as(f32, 1.0), pflt[p * 4 + 0]);
    try std.testing.expectEqual(@as(f32, 0.0), pflt[p * 4 + 1]);
    try std.testing.expectEqual(@as(f32, 1.0), pflt[p * 4 + 3]);

    // An invalid (format, type) pair is rejected with GL_INVALID_ENUM (5_5_5_1 is RGBA-only).
    var junk: [16]u8 = undefined;
    glReadPixels(0, 0, 1, 1, gles.GL_RGB, gles.GL_UNSIGNED_SHORT_5_5_5_1, &junk);
    try std.testing.expectEqual(gles.GL_INVALID_ENUM, glGetError());
}

test "glGetIntegerv/Floatv/Booleanv report limits, the readback pair, and live state" {
    _ = eglBindAPI(egl.EGL_OPENGL_ES_API);
    const dpy = eglGetPlatformDisplay(egl.EGL_PLATFORM_SURFACELESS_MESA, @ptrFromInt(0xE61C8), null);
    try std.testing.expectEqual(egl.EGL_TRUE, eglInitialize(dpy, null, null));
    defer _ = eglTerminate(dpy);
    const cfg = state.configHandle(0);
    const ctx = eglCreateContext(dpy, cfg, egl.EGL_NO_CONTEXT, null);
    const W: EGLint = 64;
    const pb_attribs = [_]EGLint{ egl.EGL_WIDTH, W, egl.EGL_HEIGHT, W, egl.EGL_NONE };
    const surf = eglCreatePbufferSurface(dpy, cfg, &pb_attribs);
    try std.testing.expectEqual(egl.EGL_TRUE, eglMakeCurrent(dpy, surf, surf, ctx));
    defer _ = eglDestroyContext(dpy, ctx);
    defer _ = eglDestroySurface(dpy, surf);
    defer _ = eglMakeCurrent(dpy, null, null, egl.EGL_NO_CONTEXT);

    glViewport(0, 0, 48, 32);
    glClearColor(0.25, 0.5, 0.75, 1.0);
    glEnable(gles.GL_DEPTH_TEST);
    glDisable(gles.GL_BLEND);

    // Implementation limits, honest to what Prism enforces (MAX_ATTRIBS=16, MAX_TEXTURE_UNITS=8).
    var mi: gles.GLint = -1;
    glGetIntegerv(gles.GL_MAX_VERTEX_ATTRIBS, @ptrCast(&mi));
    try std.testing.expectEqual(@as(gles.GLint, 16), mi);
    glGetIntegerv(gles.GL_MAX_TEXTURE_IMAGE_UNITS, @ptrCast(&mi));
    try std.testing.expectEqual(@as(gles.GLint, 8), mi);

    // The glReadPixels implementation-defined pair completes the readback story.
    glGetIntegerv(gles.GL_IMPLEMENTATION_COLOR_READ_FORMAT, @ptrCast(&mi));
    try std.testing.expectEqual(@as(gles.GLint, @intCast(gles.GL_RGBA)), mi);
    glGetIntegerv(gles.GL_IMPLEMENTATION_COLOR_READ_TYPE, @ptrCast(&mi));
    try std.testing.expectEqual(@as(gles.GLint, @intCast(gles.GL_UNSIGNED_BYTE)), mi);

    // Live viewport (4 ints).
    var vp: [4]gles.GLint = .{ 0, 0, 0, 0 };
    glGetIntegerv(gles.GL_VIEWPORT, &vp);
    try std.testing.expectEqual([4]gles.GLint{ 0, 0, 48, 32 }, vp);

    // Clear color as float, anisotropy limit as float.
    var cc: [4]gles.GLfloat = undefined;
    glGetFloatv(gles.GL_COLOR_CLEAR_VALUE, &cc);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), cc[1], 1e-6);
    var af: gles.GLfloat = 0;
    glGetFloatv(gles.GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT, @ptrCast(&af));
    try std.testing.expectEqual(@as(f32, 16), af);

    // Enable flags as booleans.
    var bd: gles.GLboolean = 2;
    glGetBooleanv(gles.GL_DEPTH_TEST, @ptrCast(&bd));
    try std.testing.expectEqual(gles.GL_TRUE, bd);
    glGetBooleanv(gles.GL_BLEND, @ptrCast(&bd));
    try std.testing.expectEqual(gles.GL_FALSE, bd);
    try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());

    // An unknown pname raises GL_INVALID_ENUM.
    glGetIntegerv(0x12345, @ptrCast(&mi));
    try std.testing.expectEqual(gles.GL_INVALID_ENUM, glGetError());
}

test "glIsEnabled + glGetShaderPrecisionFormat report capability state and full float/int precision" {
    _ = eglBindAPI(egl.EGL_OPENGL_ES_API);
    const dpy = eglGetPlatformDisplay(egl.EGL_PLATFORM_SURFACELESS_MESA, @ptrFromInt(0xE61CA), null);
    try std.testing.expectEqual(egl.EGL_TRUE, eglInitialize(dpy, null, null));
    defer _ = eglTerminate(dpy);
    const cfg = state.configHandle(0);
    const ctx = eglCreateContext(dpy, cfg, egl.EGL_NO_CONTEXT, null);
    const W: EGLint = 64;
    const pb_attribs = [_]EGLint{ egl.EGL_WIDTH, W, egl.EGL_HEIGHT, W, egl.EGL_NONE };
    const surf = eglCreatePbufferSurface(dpy, cfg, &pb_attribs);
    try std.testing.expectEqual(egl.EGL_TRUE, eglMakeCurrent(dpy, surf, surf, ctx));
    defer _ = eglDestroyContext(dpy, ctx);
    defer _ = eglDestroySurface(dpy, surf);
    defer _ = eglMakeCurrent(dpy, null, null, egl.EGL_NO_CONTEXT);

    // glIsEnabled tracks the live capability state.
    glEnable(gles.GL_SCISSOR_TEST);
    glDisable(gles.GL_CULL_FACE);
    try std.testing.expectEqual(gles.GL_TRUE, glIsEnabled(gles.GL_SCISSOR_TEST));
    try std.testing.expectEqual(gles.GL_FALSE, glIsEnabled(gles.GL_CULL_FACE));
    try std.testing.expectEqual(gles.GL_TRUE, glIsEnabled(gles.GL_DITHER)); // default-on
    try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());
    // An unknown capability is GL_INVALID_ENUM (and returns GL_FALSE).
    try std.testing.expectEqual(gles.GL_FALSE, glIsEnabled(0x12345));
    try std.testing.expectEqual(gles.GL_INVALID_ENUM, glGetError());

    // glGetShaderPrecisionFormat: full IEEE single-float / 32-bit-int precision.
    var range: [2]gles.GLint = .{ 0, 0 };
    var precision: gles.GLint = -1;
    glGetShaderPrecisionFormat(gles.GL_FRAGMENT_SHADER, gles.GL_HIGH_FLOAT, &range, &precision);
    try std.testing.expectEqual([2]gles.GLint{ 127, 127 }, range);
    try std.testing.expectEqual(@as(gles.GLint, 23), precision);
    glGetShaderPrecisionFormat(gles.GL_VERTEX_SHADER, gles.GL_HIGH_INT, &range, &precision);
    try std.testing.expectEqual([2]gles.GLint{ 31, 31 }, range);
    try std.testing.expectEqual(@as(gles.GLint, 0), precision);
    try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());
    // Invalid precision type -> GL_INVALID_ENUM.
    glGetShaderPrecisionFormat(gles.GL_FRAGMENT_SHADER, 0x1234, &range, &precision);
    try std.testing.expectEqual(gles.GL_INVALID_ENUM, glGetError());
}

test "glGetIntegerv framebuffer bits: the default framebuffer reports the surface config (depth/stencil/samples)" {
    _ = eglBindAPI(egl.EGL_OPENGL_ES_API);
    const dpy = eglGetPlatformDisplay(egl.EGL_PLATFORM_SURFACELESS_MESA, @ptrFromInt(0xE61CC), null);
    try std.testing.expectEqual(egl.EGL_TRUE, eglInitialize(dpy, null, null));
    defer _ = eglTerminate(dpy);
    const pb_attribs = [_]EGLint{ egl.EGL_WIDTH, 64, egl.EGL_HEIGHT, 64, egl.EGL_NONE };
    var v: gles.GLint = -1;

    // Config index 2 (id 3): 8/8/8/8 color, 24-bit depth, 8-bit stencil, no MSAA.
    {
        const cfg = state.configHandle(2);
        const ctx = eglCreateContext(dpy, cfg, egl.EGL_NO_CONTEXT, null);
        const surf = eglCreatePbufferSurface(dpy, cfg, &pb_attribs);
        try std.testing.expectEqual(egl.EGL_TRUE, eglMakeCurrent(dpy, surf, surf, ctx));
        defer _ = eglDestroyContext(dpy, ctx);
        defer _ = eglDestroySurface(dpy, surf);
        defer _ = eglMakeCurrent(dpy, null, null, egl.EGL_NO_CONTEXT);
        glGetIntegerv(gles.GL_RED_BITS, @ptrCast(&v));
        try std.testing.expectEqual(@as(gles.GLint, 8), v);
        glGetIntegerv(gles.GL_ALPHA_BITS, @ptrCast(&v));
        try std.testing.expectEqual(@as(gles.GLint, 8), v);
        glGetIntegerv(gles.GL_DEPTH_BITS, @ptrCast(&v));
        try std.testing.expectEqual(@as(gles.GLint, 24), v);
        glGetIntegerv(gles.GL_STENCIL_BITS, @ptrCast(&v));
        try std.testing.expectEqual(@as(gles.GLint, 8), v);
        glGetIntegerv(gles.GL_SAMPLES, @ptrCast(&v));
        try std.testing.expectEqual(@as(gles.GLint, 0), v);
        glGetIntegerv(gles.GL_SAMPLE_BUFFERS, @ptrCast(&v));
        try std.testing.expectEqual(@as(gles.GLint, 0), v);
        try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());
    }

    // Config index 4 (id 5): a 4x-MSAA surface -> GL_SAMPLES=4, GL_SAMPLE_BUFFERS=1.
    {
        const cfg = state.configHandle(4);
        const ctx = eglCreateContext(dpy, cfg, egl.EGL_NO_CONTEXT, null);
        const surf = eglCreatePbufferSurface(dpy, cfg, &pb_attribs);
        try std.testing.expectEqual(egl.EGL_TRUE, eglMakeCurrent(dpy, surf, surf, ctx));
        defer _ = eglDestroyContext(dpy, ctx);
        defer _ = eglDestroySurface(dpy, surf);
        defer _ = eglMakeCurrent(dpy, null, null, egl.EGL_NO_CONTEXT);
        glGetIntegerv(gles.GL_SAMPLES, @ptrCast(&v));
        try std.testing.expectEqual(@as(gles.GLint, 4), v);
        glGetIntegerv(gles.GL_SAMPLE_BUFFERS, @ptrCast(&v));
        try std.testing.expectEqual(@as(gles.GLint, 1), v);
        glGetIntegerv(gles.GL_DEPTH_BITS, @ptrCast(&v));
        try std.testing.expectEqual(@as(gles.GLint, 24), v);
        try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());
    }
}

test "per-object getters: glGetTexParameter / VertexAttrib / BufferParameter / Renderbuffer report object state" {
    _ = eglBindAPI(egl.EGL_OPENGL_ES_API);
    const dpy = eglGetPlatformDisplay(egl.EGL_PLATFORM_SURFACELESS_MESA, @ptrFromInt(0xE61CE), null);
    try std.testing.expectEqual(egl.EGL_TRUE, eglInitialize(dpy, null, null));
    defer _ = eglTerminate(dpy);
    const cfg = state.configHandle(0);
    const ctx = eglCreateContext(dpy, cfg, egl.EGL_NO_CONTEXT, null);
    const pb_attribs = [_]EGLint{ egl.EGL_WIDTH, 64, egl.EGL_HEIGHT, 64, egl.EGL_NONE };
    const surf = eglCreatePbufferSurface(dpy, cfg, &pb_attribs);
    try std.testing.expectEqual(egl.EGL_TRUE, eglMakeCurrent(dpy, surf, surf, ctx));
    defer _ = eglDestroyContext(dpy, ctx);
    defer _ = eglDestroySurface(dpy, surf);
    defer _ = eglMakeCurrent(dpy, null, null, egl.EGL_NO_CONTEXT);

    var iv: gles.GLint = -1;

    // --- Texture sampler state round-trips (the combined min-filter enum is reconstructed). ---
    var tex: gles.GLuint = 0;
    glGenTextures(1, @ptrCast(&tex));
    glBindTexture(gles.GL_TEXTURE_2D, tex);
    glTexParameteri(gles.GL_TEXTURE_2D, gles.GL_TEXTURE_MIN_FILTER, gles.GL_LINEAR_MIPMAP_NEAREST);
    glTexParameteri(gles.GL_TEXTURE_2D, gles.GL_TEXTURE_MAG_FILTER, gles.GL_NEAREST);
    glTexParameteri(gles.GL_TEXTURE_2D, gles.GL_TEXTURE_WRAP_S, gles.GL_CLAMP_TO_EDGE);
    glTexParameterf(gles.GL_TEXTURE_2D, gles.GL_TEXTURE_MAX_ANISOTROPY_EXT, 8.0);
    glGetTexParameteriv(gles.GL_TEXTURE_2D, gles.GL_TEXTURE_MIN_FILTER, @ptrCast(&iv));
    try std.testing.expectEqual(@as(gles.GLint, @intCast(gles.GL_LINEAR_MIPMAP_NEAREST)), iv);
    glGetTexParameteriv(gles.GL_TEXTURE_2D, gles.GL_TEXTURE_MAG_FILTER, @ptrCast(&iv));
    try std.testing.expectEqual(@as(gles.GLint, @intCast(gles.GL_NEAREST)), iv);
    glGetTexParameteriv(gles.GL_TEXTURE_2D, gles.GL_TEXTURE_WRAP_S, @ptrCast(&iv));
    try std.testing.expectEqual(@as(gles.GLint, @intCast(gles.GL_CLAMP_TO_EDGE)), iv);
    var fv: gles.GLfloat = 0;
    glGetTexParameterfv(gles.GL_TEXTURE_2D, gles.GL_TEXTURE_MAX_ANISOTROPY_EXT, @ptrCast(&fv));
    try std.testing.expectEqual(@as(f32, 8.0), fv);

    // --- Buffer size + usage hint. ---
    var vbo: gles.GLuint = 0;
    glGenBuffers(1, @ptrCast(&vbo));
    glBindBuffer(gles.GL_ARRAY_BUFFER, vbo);
    const data = [_]f32{ 1, 2, 3, 4, 5, 6 }; // 24 bytes
    glBufferData(gles.GL_ARRAY_BUFFER, @sizeOf(@TypeOf(data)), &data, gles.GL_DYNAMIC_DRAW);
    glGetBufferParameteriv(gles.GL_ARRAY_BUFFER, gles.GL_BUFFER_SIZE, @ptrCast(&iv));
    try std.testing.expectEqual(@as(gles.GLint, 24), iv);
    glGetBufferParameteriv(gles.GL_ARRAY_BUFFER, gles.GL_BUFFER_USAGE, @ptrCast(&iv));
    try std.testing.expectEqual(@as(gles.GLint, @intCast(gles.GL_DYNAMIC_DRAW)), iv);

    // --- Vertex-attrib array config (attrib 1: 3 floats, stride 20, sourced from vbo). ---
    glVertexAttribPointer(1, 3, gles.GL_FLOAT, gles.GL_FALSE, 20, @ptrFromInt(8));
    glEnableVertexAttribArray(1);
    glGetVertexAttribiv(1, gles.GL_VERTEX_ATTRIB_ARRAY_ENABLED, @ptrCast(&iv));
    try std.testing.expectEqual(@as(gles.GLint, 1), iv);
    glGetVertexAttribiv(1, gles.GL_VERTEX_ATTRIB_ARRAY_SIZE, @ptrCast(&iv));
    try std.testing.expectEqual(@as(gles.GLint, 3), iv);
    glGetVertexAttribiv(1, gles.GL_VERTEX_ATTRIB_ARRAY_STRIDE, @ptrCast(&iv));
    try std.testing.expectEqual(@as(gles.GLint, 20), iv);
    glGetVertexAttribiv(1, gles.GL_VERTEX_ATTRIB_ARRAY_TYPE, @ptrCast(&iv));
    try std.testing.expectEqual(@as(gles.GLint, @intCast(gles.GL_FLOAT)), iv);
    glGetVertexAttribiv(1, gles.GL_VERTEX_ATTRIB_ARRAY_BUFFER_BINDING, @ptrCast(&iv));
    try std.testing.expectEqual(@as(gles.GLint, @intCast(vbo)), iv);
    var cur: [4]gles.GLfloat = undefined;
    glGetVertexAttribfv(1, gles.GL_CURRENT_VERTEX_ATTRIB, &cur);
    try std.testing.expectEqual([4]gles.GLfloat{ 0, 0, 0, 1 }, cur);

    // --- Renderbuffer dimensions, sizes, and internal format (packed depth24-stencil8). ---
    var rb: gles.GLuint = 0;
    glGenRenderbuffers(1, @ptrCast(&rb));
    glBindRenderbuffer(gles.GL_RENDERBUFFER, rb);
    glRenderbufferStorage(gles.GL_RENDERBUFFER, gles.GL_DEPTH24_STENCIL8, 40, 30);
    glGetRenderbufferParameteriv(gles.GL_RENDERBUFFER, gles.GL_RENDERBUFFER_WIDTH, @ptrCast(&iv));
    try std.testing.expectEqual(@as(gles.GLint, 40), iv);
    glGetRenderbufferParameteriv(gles.GL_RENDERBUFFER, gles.GL_RENDERBUFFER_HEIGHT, @ptrCast(&iv));
    try std.testing.expectEqual(@as(gles.GLint, 30), iv);
    glGetRenderbufferParameteriv(gles.GL_RENDERBUFFER, gles.GL_RENDERBUFFER_DEPTH_SIZE, @ptrCast(&iv));
    try std.testing.expectEqual(@as(gles.GLint, 24), iv);
    glGetRenderbufferParameteriv(gles.GL_RENDERBUFFER, gles.GL_RENDERBUFFER_STENCIL_SIZE, @ptrCast(&iv));
    try std.testing.expectEqual(@as(gles.GLint, 8), iv);
    glGetRenderbufferParameteriv(gles.GL_RENDERBUFFER, gles.GL_RENDERBUFFER_INTERNAL_FORMAT, @ptrCast(&iv));
    try std.testing.expectEqual(@as(gles.GLint, @intCast(gles.GL_DEPTH24_STENCIL8)), iv);
    try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());

    // A bad target -> GL_INVALID_ENUM.
    glGetTexParameteriv(0x1234, gles.GL_TEXTURE_MIN_FILTER, @ptrCast(&iv));
    try std.testing.expectEqual(gles.GL_INVALID_ENUM, glGetError());

    glDeleteTextures(1, &[_]gles.GLuint{tex});
    glDeleteBuffers(1, &[_]gles.GLuint{vbo});
}

test "glGetFramebufferAttachmentParameteriv reports attachment type/name for a color-texture + depth-renderbuffer FBO" {
    _ = eglBindAPI(egl.EGL_OPENGL_ES_API);
    const dpy = eglGetPlatformDisplay(egl.EGL_PLATFORM_SURFACELESS_MESA, @ptrFromInt(0xE61D0), null);
    try std.testing.expectEqual(egl.EGL_TRUE, eglInitialize(dpy, null, null));
    defer _ = eglTerminate(dpy);
    const cfg = state.configHandle(0);
    const ctx = eglCreateContext(dpy, cfg, egl.EGL_NO_CONTEXT, null);
    const pb_attribs = [_]EGLint{ egl.EGL_WIDTH, 64, egl.EGL_HEIGHT, 64, egl.EGL_NONE };
    const surf = eglCreatePbufferSurface(dpy, cfg, &pb_attribs);
    try std.testing.expectEqual(egl.EGL_TRUE, eglMakeCurrent(dpy, surf, surf, ctx));
    defer _ = eglDestroyContext(dpy, ctx);
    defer _ = eglDestroySurface(dpy, surf);
    defer _ = eglMakeCurrent(dpy, null, null, egl.EGL_NO_CONTEXT);

    const TS: gles.GLsizei = 32;
    var tex: gles.GLuint = 0;
    glGenTextures(1, @ptrCast(&tex));
    glBindTexture(gles.GL_TEXTURE_2D, tex);
    glTexImage2D(gles.GL_TEXTURE_2D, 0, gles.GL_RGBA, TS, TS, 0, gles.GL_RGBA, gles.GL_UNSIGNED_BYTE, null);
    var rb: gles.GLuint = 0;
    glGenRenderbuffers(1, @ptrCast(&rb));
    glBindRenderbuffer(gles.GL_RENDERBUFFER, rb);
    glRenderbufferStorage(gles.GL_RENDERBUFFER, gles.GL_DEPTH_COMPONENT16, TS, TS);
    var fbo: gles.GLuint = 0;
    glGenFramebuffers(1, @ptrCast(&fbo));
    glBindFramebuffer(gles.GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(gles.GL_FRAMEBUFFER, gles.GL_COLOR_ATTACHMENT0, gles.GL_TEXTURE_2D, tex, 0);
    glFramebufferRenderbuffer(gles.GL_FRAMEBUFFER, gles.GL_DEPTH_ATTACHMENT, gles.GL_RENDERBUFFER, rb);
    try std.testing.expectEqual(gles.GL_FRAMEBUFFER_COMPLETE, glCheckFramebufferStatus(gles.GL_FRAMEBUFFER));

    var iv: gles.GLint = -1;
    // Color attachment: a texture named `tex`, mip level 0.
    glGetFramebufferAttachmentParameteriv(gles.GL_FRAMEBUFFER, gles.GL_COLOR_ATTACHMENT0, gles.GL_FRAMEBUFFER_ATTACHMENT_OBJECT_TYPE, @ptrCast(&iv));
    try std.testing.expectEqual(@as(gles.GLint, @intCast(gles.GL_TEXTURE)), iv);
    glGetFramebufferAttachmentParameteriv(gles.GL_FRAMEBUFFER, gles.GL_COLOR_ATTACHMENT0, gles.GL_FRAMEBUFFER_ATTACHMENT_OBJECT_NAME, @ptrCast(&iv));
    try std.testing.expectEqual(@as(gles.GLint, @intCast(tex)), iv);
    glGetFramebufferAttachmentParameteriv(gles.GL_FRAMEBUFFER, gles.GL_COLOR_ATTACHMENT0, gles.GL_FRAMEBUFFER_ATTACHMENT_TEXTURE_LEVEL, @ptrCast(&iv));
    try std.testing.expectEqual(@as(gles.GLint, 0), iv);
    // Depth attachment: a renderbuffer named `rb`.
    glGetFramebufferAttachmentParameteriv(gles.GL_FRAMEBUFFER, gles.GL_DEPTH_ATTACHMENT, gles.GL_FRAMEBUFFER_ATTACHMENT_OBJECT_TYPE, @ptrCast(&iv));
    try std.testing.expectEqual(@as(gles.GLint, @intCast(gles.GL_RENDERBUFFER)), iv);
    glGetFramebufferAttachmentParameteriv(gles.GL_FRAMEBUFFER, gles.GL_DEPTH_ATTACHMENT, gles.GL_FRAMEBUFFER_ATTACHMENT_OBJECT_NAME, @ptrCast(&iv));
    try std.testing.expectEqual(@as(gles.GLint, @intCast(rb)), iv);
    // An unattached point reports NONE (no error); OBJECT_NAME on it is GL_INVALID_ENUM.
    glGetFramebufferAttachmentParameteriv(gles.GL_FRAMEBUFFER, gles.GL_STENCIL_ATTACHMENT, gles.GL_FRAMEBUFFER_ATTACHMENT_OBJECT_TYPE, @ptrCast(&iv));
    try std.testing.expectEqual(@as(gles.GLint, @intCast(gles.GL_NONE)), iv);
    try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());
    glGetFramebufferAttachmentParameteriv(gles.GL_FRAMEBUFFER, gles.GL_STENCIL_ATTACHMENT, gles.GL_FRAMEBUFFER_ATTACHMENT_OBJECT_NAME, @ptrCast(&iv));
    try std.testing.expectEqual(gles.GL_INVALID_ENUM, glGetError());
    // TEXTURE_LEVEL on a renderbuffer attachment is GL_INVALID_ENUM.
    glGetFramebufferAttachmentParameteriv(gles.GL_FRAMEBUFFER, gles.GL_DEPTH_ATTACHMENT, gles.GL_FRAMEBUFFER_ATTACHMENT_TEXTURE_LEVEL, @ptrCast(&iv));
    try std.testing.expectEqual(gles.GL_INVALID_ENUM, glGetError());

    // The default framebuffer is not queryable in ES2 (GL_INVALID_OPERATION).
    glBindFramebuffer(gles.GL_FRAMEBUFFER, 0);
    glGetFramebufferAttachmentParameteriv(gles.GL_FRAMEBUFFER, gles.GL_COLOR_ATTACHMENT0, gles.GL_FRAMEBUFFER_ATTACHMENT_OBJECT_TYPE, @ptrCast(&iv));
    try std.testing.expectEqual(gles.GL_INVALID_OPERATION, glGetError());

    glDeleteFramebuffers(1, &[_]gles.GLuint{fbo});
    glDeleteRenderbuffers(1, &[_]gles.GLuint{rb});
    glDeleteTextures(1, &[_]gles.GLuint{tex});
}

test "program introspection: glGetAttachedShaders + glGetUniformfv + glGetVertexAttribPointerv report linked state" {
    _ = eglBindAPI(egl.EGL_OPENGL_ES_API);
    const dpy = eglGetPlatformDisplay(egl.EGL_PLATFORM_SURFACELESS_MESA, @ptrFromInt(0xE61D2), null);
    try std.testing.expectEqual(egl.EGL_TRUE, eglInitialize(dpy, null, null));
    defer _ = eglTerminate(dpy);
    const cfg = state.configHandle(0);
    const ctx = eglCreateContext(dpy, cfg, egl.EGL_NO_CONTEXT, null);
    const pb_attribs = [_]EGLint{ egl.EGL_WIDTH, 64, egl.EGL_HEIGHT, 64, egl.EGL_NONE };
    const surf = eglCreatePbufferSurface(dpy, cfg, &pb_attribs);
    try std.testing.expectEqual(egl.EGL_TRUE, eglMakeCurrent(dpy, surf, surf, ctx));
    defer _ = eglDestroyContext(dpy, ctx);
    defer _ = eglDestroySurface(dpy, surf);
    defer _ = eglMakeCurrent(dpy, null, null, egl.EGL_NO_CONTEXT);

    const vs_src: [*:0]const gles.GLchar =
        \\attribute vec2 position;
        \\uniform vec4 uColor;
        \\varying vec4 vColor;
        \\void main() { gl_Position = vec4(position, 0.0, 1.0); vColor = uColor; }
    ;
    const fs_src: [*:0]const gles.GLchar =
        \\precision mediump float;
        \\varying vec4 vColor;
        \\void main() { gl_FragColor = vColor; }
    ;
    const vs = glCreateShader(gles.GL_VERTEX_SHADER);
    glShaderSource(vs, 1, &[_]?[*:0]const gles.GLchar{vs_src}, null);
    glCompileShader(vs);
    const fs = glCreateShader(gles.GL_FRAGMENT_SHADER);
    glShaderSource(fs, 1, &[_]?[*:0]const gles.GLchar{fs_src}, null);
    glCompileShader(fs);
    const prog = glCreateProgram();
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glBindAttribLocation(prog, 0, "position");
    glLinkProgram(prog);
    glUseProgram(prog);

    // glGetAttachedShaders: the VS + FS ids, in attach order.
    var count: gles.GLsizei = -1;
    var shaders: [4]gles.GLuint = .{ 0, 0, 0, 0 };
    glGetAttachedShaders(prog, 4, &count, &shaders);
    try std.testing.expectEqual(@as(gles.GLsizei, 2), count);
    try std.testing.expectEqual(vs, shaders[0]);
    try std.testing.expectEqual(fs, shaders[1]);

    // glGetUniformfv round-trips a uniform's current value.
    const loc = glGetUniformLocation(prog, "uColor");
    try std.testing.expect(loc >= 0);
    glUniform4f(loc, 0.25, 0.5, 0.75, 1.0);
    var uv: [4]gles.GLfloat = undefined;
    glGetUniformfv(prog, loc, &uv);
    try std.testing.expectEqual([4]gles.GLfloat{ 0.25, 0.5, 0.75, 1.0 }, uv);

    // glGetVertexAttribPointerv returns the client byte offset of the attrib array.
    var vbo: gles.GLuint = 0;
    glGenBuffers(1, @ptrCast(&vbo));
    glBindBuffer(gles.GL_ARRAY_BUFFER, vbo);
    glVertexAttribPointer(2, 3, gles.GL_FLOAT, gles.GL_FALSE, 20, @ptrFromInt(12));
    var ptr: ?*anyopaque = null;
    glGetVertexAttribPointerv(2, gles.GL_VERTEX_ATTRIB_ARRAY_POINTER, &ptr);
    try std.testing.expectEqual(@as(usize, 12), @intFromPtr(ptr));
    try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());

    glDeleteBuffers(1, &[_]gles.GLuint{vbo});
    glDeleteProgram(prog);
    glDeleteShader(vs);
    glDeleteShader(fs);
}

test "float render target: an rgba32f FBO renders + reads back an HDR value (>1.0) unclamped (software)" {
    _ = eglBindAPI(egl.EGL_OPENGL_ES_API);
    const dpy = eglGetPlatformDisplay(egl.EGL_PLATFORM_SURFACELESS_MESA, @ptrFromInt(0xE61D4), null);
    try std.testing.expectEqual(egl.EGL_TRUE, eglInitialize(dpy, null, null));
    defer _ = eglTerminate(dpy);
    const cfg = state.configHandle(0);
    const ctx = eglCreateContext(dpy, cfg, egl.EGL_NO_CONTEXT, null);
    const pb_attribs = [_]EGLint{ egl.EGL_WIDTH, 64, egl.EGL_HEIGHT, 64, egl.EGL_NONE };
    const surf = eglCreatePbufferSurface(dpy, cfg, &pb_attribs);
    try std.testing.expectEqual(egl.EGL_TRUE, eglMakeCurrent(dpy, surf, surf, ctx));
    defer _ = eglDestroyContext(dpy, ctx);
    defer _ = eglDestroySurface(dpy, surf);
    defer _ = eglMakeCurrent(dpy, null, null, egl.EGL_NO_CONTEXT);

    // A fragment shader emitting an HDR color (red > 1.0), which an 8-bit RT would clamp to 1.0.
    const vs_src: [*:0]const gles.GLchar =
        \\attribute vec2 position;
        \\void main() { gl_Position = vec4(position, 0.0, 1.0); }
    ;
    const fs_src: [*:0]const gles.GLchar =
        \\precision highp float;
        \\void main() { gl_FragColor = vec4(2.0, 0.5, 0.25, 1.0); }
    ;
    const vs = glCreateShader(gles.GL_VERTEX_SHADER);
    glShaderSource(vs, 1, &[_]?[*:0]const gles.GLchar{vs_src}, null);
    glCompileShader(vs);
    const fs = glCreateShader(gles.GL_FRAGMENT_SHADER);
    glShaderSource(fs, 1, &[_]?[*:0]const gles.GLchar{fs_src}, null);
    glCompileShader(fs);
    const prog = glCreateProgram();
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glBindAttribLocation(prog, 0, "position");
    glLinkProgram(prog);
    glUseProgram(prog);

    // An rgba32f color texture attached to an FBO renders at full precision.
    const TS: gles.GLsizei = 8;
    var tex: gles.GLuint = 0;
    glGenTextures(1, @ptrCast(&tex));
    glBindTexture(gles.GL_TEXTURE_2D, tex);
    glTexImage2D(gles.GL_TEXTURE_2D, 0, gles.GL_RGBA, TS, TS, 0, gles.GL_RGBA, gles.GL_FLOAT, null);
    glTexParameteri(gles.GL_TEXTURE_2D, gles.GL_TEXTURE_MIN_FILTER, gles.GL_NEAREST);
    glTexParameteri(gles.GL_TEXTURE_2D, gles.GL_TEXTURE_MAG_FILTER, gles.GL_NEAREST);
    var fbo: gles.GLuint = 0;
    glGenFramebuffers(1, @ptrCast(&fbo));
    glBindFramebuffer(gles.GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(gles.GL_FRAMEBUFFER, gles.GL_COLOR_ATTACHMENT0, gles.GL_TEXTURE_2D, tex, 0);
    try std.testing.expectEqual(gles.GL_FRAMEBUFFER_COMPLETE, glCheckFramebufferStatus(gles.GL_FRAMEBUFFER));

    glViewport(0, 0, TS, TS);
    glClearColor(0, 0, 0, 1);
    glClear(gles.GL_COLOR_BUFFER_BIT);
    // A single fullscreen triangle covers the whole target.
    const tri = [_]f32{ -1, -1, 3, -1, -1, 3 };
    var vbo: gles.GLuint = 0;
    glGenBuffers(1, @ptrCast(&vbo));
    glBindBuffer(gles.GL_ARRAY_BUFFER, vbo);
    glBufferData(gles.GL_ARRAY_BUFFER, @sizeOf(@TypeOf(tri)), &tri, gles.GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, gles.GL_FLOAT, gles.GL_FALSE, 0, @ptrFromInt(0));
    glEnableVertexAttribArray(0);
    glDrawArrays(gles.GL_TRIANGLES, 0, 3);
    try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());

    // glReadPixels as GL_FLOAT returns the TRUE value: red is 2.0, NOT clamped to 1.0.
    var px: [8 * 8 * 4]gles.GLfloat = undefined;
    glReadPixels(0, 0, TS, TS, gles.GL_RGBA, gles.GL_FLOAT, &px);
    try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());
    const ci = (@as(usize, 4) * 8 + 4) * 4; // center pixel
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), px[ci + 0], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), px[ci + 1], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 0.25), px[ci + 2], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), px[ci + 3], 1e-4);

    // The SAME render read back as GL_UNSIGNED_BYTE clamps red to 255 (0..1 saturation).
    var pb: [8 * 8 * 4]u8 = undefined;
    glReadPixels(0, 0, TS, TS, gles.GL_RGBA, gles.GL_UNSIGNED_BYTE, &pb);
    try std.testing.expectEqual(@as(u8, 255), pb[ci + 0]);
    try std.testing.expectEqual(@as(u8, 128), pb[ci + 1]);

    glDeleteFramebuffers(1, &[_]gles.GLuint{fbo});
    glDeleteBuffers(1, &[_]gles.GLuint{vbo});
    glDeleteTextures(1, &[_]gles.GLuint{tex});
    glDeleteProgram(prog);
    glDeleteShader(vs);
    glDeleteShader(fs);
}

test "glVertexAttribDivisor: per-instance attributes place + tint each instance (2 instances, software)" {
    _ = eglBindAPI(egl.EGL_OPENGL_ES_API);
    const dpy = eglGetPlatformDisplay(egl.EGL_PLATFORM_SURFACELESS_MESA, @ptrFromInt(0xE61D8), null);
    try std.testing.expectEqual(egl.EGL_TRUE, eglInitialize(dpy, null, null));
    defer _ = eglTerminate(dpy);
    const cfg = state.configHandle(0);
    const ctx = eglCreateContext(dpy, cfg, egl.EGL_NO_CONTEXT, null);
    const pb_attribs = [_]EGLint{ egl.EGL_WIDTH, 64, egl.EGL_HEIGHT, 64, egl.EGL_NONE };
    const surf = eglCreatePbufferSurface(dpy, cfg, &pb_attribs);
    try std.testing.expectEqual(egl.EGL_TRUE, eglMakeCurrent(dpy, surf, surf, ctx));
    defer _ = eglDestroyContext(dpy, ctx);
    defer _ = eglDestroySurface(dpy, surf);
    defer _ = eglMakeCurrent(dpy, null, null, egl.EGL_NO_CONTEXT);

    const vs_src: [*:0]const gles.GLchar =
        \\attribute vec2 position;
        \\attribute vec2 iOffset;
        \\attribute vec3 iColor;
        \\varying vec3 vColor;
        \\void main() { gl_Position = vec4(position + iOffset, 0.0, 1.0); vColor = iColor; }
    ;
    const fs_src: [*:0]const gles.GLchar =
        \\precision mediump float;
        \\varying vec3 vColor;
        \\void main() { gl_FragColor = vec4(vColor, 1.0); }
    ;
    const vs = glCreateShader(gles.GL_VERTEX_SHADER);
    glShaderSource(vs, 1, &[_]?[*:0]const gles.GLchar{vs_src}, null);
    glCompileShader(vs);
    const fs = glCreateShader(gles.GL_FRAGMENT_SHADER);
    glShaderSource(fs, 1, &[_]?[*:0]const gles.GLchar{fs_src}, null);
    glCompileShader(fs);
    const prog = glCreateProgram();
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glBindAttribLocation(prog, 0, "position");
    glBindAttribLocation(prog, 1, "iOffset");
    glBindAttribLocation(prog, 2, "iColor");
    glLinkProgram(prog);
    glUseProgram(prog);

    // Per-vertex: a small quad at the origin (won't overlap the two instance offsets).
    const quad = [_]f32{ -0.15, -0.15, 0.15, -0.15, 0.15, 0.15, -0.15, -0.15, 0.15, 0.15, -0.15, 0.15 };
    var pvbo: gles.GLuint = 0;
    glGenBuffers(1, @ptrCast(&pvbo));
    glBindBuffer(gles.GL_ARRAY_BUFFER, pvbo);
    glBufferData(gles.GL_ARRAY_BUFFER, @sizeOf(@TypeOf(quad)), &quad, gles.GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, gles.GL_FLOAT, gles.GL_FALSE, 0, @ptrFromInt(0));
    glEnableVertexAttribArray(0);

    // Per-instance offset (divisor 1): instance 0 -> left, instance 1 -> right.
    const offs = [_]f32{ -0.5, 0.0, 0.5, 0.0 };
    var ovbo: gles.GLuint = 0;
    glGenBuffers(1, @ptrCast(&ovbo));
    glBindBuffer(gles.GL_ARRAY_BUFFER, ovbo);
    glBufferData(gles.GL_ARRAY_BUFFER, @sizeOf(@TypeOf(offs)), &offs, gles.GL_STATIC_DRAW);
    glVertexAttribPointer(1, 2, gles.GL_FLOAT, gles.GL_FALSE, 0, @ptrFromInt(0));
    glEnableVertexAttribArray(1);
    glVertexAttribDivisor(1, 1);

    // Per-instance color (divisor 1): instance 0 -> red, instance 1 -> green.
    const cols = [_]f32{ 1, 0, 0, 0, 1, 0 };
    var cvbo: gles.GLuint = 0;
    glGenBuffers(1, @ptrCast(&cvbo));
    glBindBuffer(gles.GL_ARRAY_BUFFER, cvbo);
    glBufferData(gles.GL_ARRAY_BUFFER, @sizeOf(@TypeOf(cols)), &cols, gles.GL_STATIC_DRAW);
    glVertexAttribPointer(2, 3, gles.GL_FLOAT, gles.GL_FALSE, 0, @ptrFromInt(0));
    glEnableVertexAttribArray(2);
    glVertexAttribDivisor(2, 1);

    glViewport(0, 0, 64, 64);
    glClearColor(0, 0, 0, 1);
    glClear(gles.GL_COLOR_BUFFER_BIT);
    glDrawArraysInstanced(gles.GL_TRIANGLES, 0, 6, 2);
    try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());

    const s: *state.Surface = @ptrCast(@alignCast(surf.?));
    const dev = s.display.device.?;
    const px = try flushMap(dev, s);
    // Instance 0 quad centered at clip (-0.5, 0) -> pixel (16, 32); instance 1 at (0.5, 0) -> (48, 32).
    const left = (@as(usize, 32) * 64 + 16) * 4;
    const right = (@as(usize, 32) * 64 + 48) * 4;
    try std.testing.expect(px[left + 0] > 200 and px[left + 1] < 60); // instance 0 = red
    try std.testing.expect(px[right + 1] > 200 and px[right + 0] < 60); // instance 1 = green

    glDeleteBuffers(1, &[_]gles.GLuint{ pvbo, ovbo, cvbo });
    glDeleteProgram(prog);
    glDeleteShader(vs);
    glDeleteShader(fs);
}

test "vertex array object: glBindVertexArray captures + restores attribute + element-buffer state" {
    _ = eglBindAPI(egl.EGL_OPENGL_ES_API);
    const dpy = eglGetPlatformDisplay(egl.EGL_PLATFORM_SURFACELESS_MESA, @ptrFromInt(0xE61DA), null);
    try std.testing.expectEqual(egl.EGL_TRUE, eglInitialize(dpy, null, null));
    defer _ = eglTerminate(dpy);
    const cfg = state.configHandle(0);
    const ctx = eglCreateContext(dpy, cfg, egl.EGL_NO_CONTEXT, null);
    const pb_attribs = [_]EGLint{ egl.EGL_WIDTH, 32, egl.EGL_HEIGHT, 32, egl.EGL_NONE };
    const surf = eglCreatePbufferSurface(dpy, cfg, &pb_attribs);
    try std.testing.expectEqual(egl.EGL_TRUE, eglMakeCurrent(dpy, surf, surf, ctx));
    defer _ = eglDestroyContext(dpy, ctx);
    defer _ = eglDestroySurface(dpy, surf);
    defer _ = eglMakeCurrent(dpy, null, null, egl.EGL_NO_CONTEXT);

    var vbo: gles.GLuint = 0;
    glGenBuffers(1, @ptrCast(&vbo));
    var ebo: gles.GLuint = 0;
    glGenBuffers(1, @ptrCast(&ebo));

    // The default VAO (0): attribute 1 stays disabled.
    var iv: gles.GLint = -1;
    glGetIntegerv(gles.GL_VERTEX_ARRAY_BINDING, @ptrCast(&iv));
    try std.testing.expectEqual(@as(gles.GLint, 0), iv);

    // A new VAO captures its own attribute + element-buffer state.
    var vao: gles.GLuint = 0;
    glGenVertexArrays(1, @ptrCast(&vao));
    try std.testing.expectEqual(gles.GL_FALSE, glIsVertexArray(vao)); // not a VAO until first bind
    glBindVertexArray(vao);
    try std.testing.expectEqual(gles.GL_TRUE, glIsVertexArray(vao));
    glGetIntegerv(gles.GL_VERTEX_ARRAY_BINDING, @ptrCast(&iv));
    try std.testing.expectEqual(@as(gles.GLint, @intCast(vao)), iv);
    glBindBuffer(gles.GL_ARRAY_BUFFER, vbo);
    glVertexAttribPointer(1, 3, gles.GL_FLOAT, gles.GL_FALSE, 12, @ptrFromInt(0));
    glEnableVertexAttribArray(1);
    glBindBuffer(gles.GL_ELEMENT_ARRAY_BUFFER, ebo);
    // In this VAO: attrib 1 enabled, sourced from vbo; element buffer = ebo.
    glGetVertexAttribiv(1, gles.GL_VERTEX_ATTRIB_ARRAY_ENABLED, @ptrCast(&iv));
    try std.testing.expectEqual(@as(gles.GLint, 1), iv);
    glGetVertexAttribiv(1, gles.GL_VERTEX_ATTRIB_ARRAY_BUFFER_BINDING, @ptrCast(&iv));
    try std.testing.expectEqual(@as(gles.GLint, @intCast(vbo)), iv);
    glGetIntegerv(gles.GL_ELEMENT_ARRAY_BUFFER_BINDING, @ptrCast(&iv));
    try std.testing.expectEqual(@as(gles.GLint, @intCast(ebo)), iv);

    // Switch back to the default VAO: attribute 1 is disabled and no element buffer is bound.
    glBindVertexArray(0);
    glGetVertexAttribiv(1, gles.GL_VERTEX_ATTRIB_ARRAY_ENABLED, @ptrCast(&iv));
    try std.testing.expectEqual(@as(gles.GLint, 0), iv);
    glGetIntegerv(gles.GL_ELEMENT_ARRAY_BUFFER_BINDING, @ptrCast(&iv));
    try std.testing.expectEqual(@as(gles.GLint, 0), iv);

    // Re-bind the VAO: its captured state is restored intact.
    glBindVertexArray(vao);
    glGetVertexAttribiv(1, gles.GL_VERTEX_ATTRIB_ARRAY_ENABLED, @ptrCast(&iv));
    try std.testing.expectEqual(@as(gles.GLint, 1), iv);
    glGetIntegerv(gles.GL_ELEMENT_ARRAY_BUFFER_BINDING, @ptrCast(&iv));
    try std.testing.expectEqual(@as(gles.GLint, @intCast(ebo)), iv);
    try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());

    // Deleting the bound VAO reverts to the default VAO.
    glDeleteVertexArrays(1, &[_]gles.GLuint{vao});
    glGetIntegerv(gles.GL_VERTEX_ARRAY_BINDING, @ptrCast(&iv));
    try std.testing.expectEqual(@as(gles.GLint, 0), iv);
    try std.testing.expectEqual(gles.GL_FALSE, glIsVertexArray(vao));

    glBindVertexArray(0);
    glDeleteBuffers(1, &[_]gles.GLuint{ vbo, ebo });
}

test "glGetStringi enumerates the extensions, matching GL_NUM_EXTENSIONS and the flat GL_EXTENSIONS string" {
    _ = eglBindAPI(egl.EGL_OPENGL_ES_API);
    const dpy = eglGetPlatformDisplay(egl.EGL_PLATFORM_SURFACELESS_MESA, @ptrFromInt(0xE61DC), null);
    try std.testing.expectEqual(egl.EGL_TRUE, eglInitialize(dpy, null, null));
    defer _ = eglTerminate(dpy);
    const cfg = state.configHandle(0);
    const ctx = eglCreateContext(dpy, cfg, egl.EGL_NO_CONTEXT, null);
    const pb_attribs = [_]EGLint{ egl.EGL_WIDTH, 16, egl.EGL_HEIGHT, 16, egl.EGL_NONE };
    const surf = eglCreatePbufferSurface(dpy, cfg, &pb_attribs);
    try std.testing.expectEqual(egl.EGL_TRUE, eglMakeCurrent(dpy, surf, surf, ctx));
    defer _ = eglDestroyContext(dpy, ctx);
    defer _ = eglDestroySurface(dpy, surf);
    defer _ = eglMakeCurrent(dpy, null, null, egl.EGL_NO_CONTEXT);

    var num: gles.GLint = 0;
    glGetIntegerv(gles.GL_NUM_EXTENSIONS, @ptrCast(&num));
    try std.testing.expect(num > 0);

    const flat = std.mem.span(@as([*:0]const u8, @ptrCast(glGetString(gles.GL_EXTENSIONS).?)));
    var i: gles.GLuint = 0;
    while (i < @as(gles.GLuint, @intCast(num))) : (i += 1) {
        const ext = std.mem.span(@as([*:0]const u8, @ptrCast(glGetStringi(gles.GL_EXTENSIONS, i).?)));
        // Every indexed extension is a substring of the flat GL_EXTENSIONS string.
        try std.testing.expect(std.mem.indexOf(u8, flat, ext) != null);
    }
    // A representative extension appears (the VAO one we just added).
    var found_vao = false;
    i = 0;
    while (i < @as(gles.GLuint, @intCast(num))) : (i += 1) {
        const ext = std.mem.span(@as([*:0]const u8, @ptrCast(glGetStringi(gles.GL_EXTENSIONS, i).?)));
        if (std.mem.eql(u8, ext, "GL_OES_vertex_array_object")) found_vao = true;
    }
    try std.testing.expect(found_vao);

    // Out-of-range index -> GL_INVALID_VALUE (null return).
    try std.testing.expect(glGetStringi(gles.GL_EXTENSIONS, @intCast(num)) == null);
    try std.testing.expectEqual(gles.GL_INVALID_VALUE, glGetError());
}

test "glTexStorage2D: immutable storage rejects glTexImage2D, accepts glTexSubImage2D, and samples" {
    _ = eglBindAPI(egl.EGL_OPENGL_ES_API);
    const dpy = eglGetPlatformDisplay(egl.EGL_PLATFORM_SURFACELESS_MESA, @ptrFromInt(0xE61DE), null);
    try std.testing.expectEqual(egl.EGL_TRUE, eglInitialize(dpy, null, null));
    defer _ = eglTerminate(dpy);
    const cfg = state.configHandle(0);
    const ctx = eglCreateContext(dpy, cfg, egl.EGL_NO_CONTEXT, null);
    const pb_attribs = [_]EGLint{ egl.EGL_WIDTH, 32, egl.EGL_HEIGHT, 32, egl.EGL_NONE };
    const surf = eglCreatePbufferSurface(dpy, cfg, &pb_attribs);
    try std.testing.expectEqual(egl.EGL_TRUE, eglMakeCurrent(dpy, surf, surf, ctx));
    defer _ = eglDestroyContext(dpy, ctx);
    defer _ = eglDestroySurface(dpy, surf);
    defer _ = eglMakeCurrent(dpy, null, null, egl.EGL_NO_CONTEXT);

    const vs_src: [*:0]const gles.GLchar =
        \\attribute vec2 position;
        \\attribute vec2 aUV;
        \\varying vec2 vUV;
        \\void main() { gl_Position = vec4(position, 0.0, 1.0); vUV = aUV; }
    ;
    const fs_src: [*:0]const gles.GLchar =
        \\precision mediump float;
        \\uniform sampler2D uTex;
        \\varying vec2 vUV;
        \\void main() { gl_FragColor = texture2D(uTex, vUV); }
    ;
    const vs = glCreateShader(gles.GL_VERTEX_SHADER);
    glShaderSource(vs, 1, &[_]?[*:0]const gles.GLchar{vs_src}, null);
    glCompileShader(vs);
    const fs = glCreateShader(gles.GL_FRAGMENT_SHADER);
    glShaderSource(fs, 1, &[_]?[*:0]const gles.GLchar{fs_src}, null);
    glCompileShader(fs);
    const prog = glCreateProgram();
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glBindAttribLocation(prog, 0, "position");
    glBindAttribLocation(prog, 1, "aUV");
    glLinkProgram(prog);
    glUseProgram(prog);

    // Immutable 2x2 rgba8 storage.
    var tex: gles.GLuint = 0;
    glGenTextures(1, @ptrCast(&tex));
    glBindTexture(gles.GL_TEXTURE_2D, tex);
    glTexStorage2D(gles.GL_TEXTURE_2D, 1, gles.GL_RGBA8, 2, 2);
    try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());
    var iv: gles.GLint = -1;
    glGetTexParameteriv(gles.GL_TEXTURE_2D, gles.GL_TEXTURE_IMMUTABLE_FORMAT, @ptrCast(&iv));
    try std.testing.expectEqual(@as(gles.GLint, 1), iv);

    // glTexImage2D on immutable storage is rejected.
    glTexImage2D(gles.GL_TEXTURE_2D, 0, gles.GL_RGBA, 2, 2, 0, gles.GL_RGBA, gles.GL_UNSIGNED_BYTE, null);
    try std.testing.expectEqual(gles.GL_INVALID_OPERATION, glGetError());

    // glTexSubImage2D fills it (solid green). Allowed on immutable storage.
    const green = [_]u8{ 0, 255, 0, 255, 0, 255, 0, 255, 0, 255, 0, 255, 0, 255, 0, 255 };
    glTexParameteri(gles.GL_TEXTURE_2D, gles.GL_TEXTURE_MIN_FILTER, @intCast(gles.GL_NEAREST));
    glTexParameteri(gles.GL_TEXTURE_2D, gles.GL_TEXTURE_MAG_FILTER, @intCast(gles.GL_NEAREST));
    glTexSubImage2D(gles.GL_TEXTURE_2D, 0, 0, 0, 2, 2, gles.GL_RGBA, gles.GL_UNSIGNED_BYTE, &green);
    try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());

    // Sample the immutable texture across a fullscreen quad.
    const QVtx = extern struct { x: f32, y: f32, u: f32, v: f32 };
    const quad = [6]QVtx{
        .{ .x = -1, .y = -1, .u = 0, .v = 0 }, .{ .x = 1, .y = -1, .u = 1, .v = 0 },
        .{ .x = 1, .y = 1, .u = 1, .v = 1 },   .{ .x = -1, .y = -1, .u = 0, .v = 0 },
        .{ .x = 1, .y = 1, .u = 1, .v = 1 },   .{ .x = -1, .y = 1, .u = 0, .v = 1 },
    };
    var qvbo: gles.GLuint = 0;
    glGenBuffers(1, @ptrCast(&qvbo));
    glBindBuffer(gles.GL_ARRAY_BUFFER, qvbo);
    glBufferData(gles.GL_ARRAY_BUFFER, @sizeOf(@TypeOf(quad)), &quad, gles.GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, gles.GL_FLOAT, gles.GL_FALSE, @sizeOf(QVtx), @ptrFromInt(0));
    glVertexAttribPointer(1, 2, gles.GL_FLOAT, gles.GL_FALSE, @sizeOf(QVtx), @ptrFromInt(8));
    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);
    glUniform1i(glGetUniformLocation(prog, "uTex"), 0);
    glActiveTexture(gles.GL_TEXTURE0);
    glBindTexture(gles.GL_TEXTURE_2D, tex);
    glViewport(0, 0, 32, 32);
    glClearColor(0, 0, 0, 1);
    glClear(gles.GL_COLOR_BUFFER_BIT);
    glDrawArrays(gles.GL_TRIANGLES, 0, 6);
    try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());

    const s: *state.Surface = @ptrCast(@alignCast(surf.?));
    const dev = s.display.device.?;
    const px = try flushMap(dev, s);
    const c = (@as(usize, 16) * 32 + 16) * 4;
    try std.testing.expect(px[c + 1] > 200 and px[c + 0] < 60 and px[c + 2] < 60); // green

    glDeleteBuffers(1, &[_]gles.GLuint{qvbo});
    glDeleteTextures(1, &[_]gles.GLuint{tex});
    glDeleteProgram(prog);
    glDeleteShader(vs);
    glDeleteShader(fs);
}

test "glTexStorage3D: immutable 2D-array storage rejects glTexImage3D, accepts glTexSubImage3D, samples a layer" {
    _ = eglBindAPI(egl.EGL_OPENGL_ES_API);
    const dpy = eglGetPlatformDisplay(egl.EGL_PLATFORM_SURFACELESS_MESA, @ptrFromInt(0xE61E4), null);
    try std.testing.expectEqual(egl.EGL_TRUE, eglInitialize(dpy, null, null));
    defer _ = eglTerminate(dpy);
    const cfg = state.configHandle(0);
    const ctx = eglCreateContext(dpy, cfg, egl.EGL_NO_CONTEXT, null);
    const pb_attribs = [_]EGLint{ egl.EGL_WIDTH, 32, egl.EGL_HEIGHT, 32, egl.EGL_NONE };
    const surf = eglCreatePbufferSurface(dpy, cfg, &pb_attribs);
    try std.testing.expectEqual(egl.EGL_TRUE, eglMakeCurrent(dpy, surf, surf, ctx));
    defer _ = eglDestroyContext(dpy, ctx);
    defer _ = eglDestroySurface(dpy, surf);
    defer _ = eglMakeCurrent(dpy, null, null, egl.EGL_NO_CONTEXT);

    const vs_src: [*:0]const gles.GLchar =
        \\attribute vec2 position;
        \\void main() { gl_Position = vec4(position, 0.0, 1.0); }
    ;
    const fs_src: [*:0]const gles.GLchar =
        \\precision mediump float;
        \\uniform sampler2DArray uArr;
        \\uniform vec3 uCoord;
        \\void main() { gl_FragColor = texture(uArr, uCoord); }
    ;
    const vs = glCreateShader(gles.GL_VERTEX_SHADER);
    glShaderSource(vs, 1, &[_]?[*:0]const gles.GLchar{vs_src}, null);
    glCompileShader(vs);
    const fs = glCreateShader(gles.GL_FRAGMENT_SHADER);
    glShaderSource(fs, 1, &[_]?[*:0]const gles.GLchar{fs_src}, null);
    glCompileShader(fs);
    const prog = glCreateProgram();
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glBindAttribLocation(prog, 0, "position");
    glLinkProgram(prog);
    glUseProgram(prog);
    defer {
        glDeleteProgram(prog);
        glDeleteShader(vs);
        glDeleteShader(fs);
    }

    // Immutable 2x2x2 array storage.
    var tex: gles.GLuint = 0;
    glGenTextures(1, @ptrCast(&tex));
    glActiveTexture(gles.GL_TEXTURE0);
    glBindTexture(gles.GL_TEXTURE_2D_ARRAY, tex);
    glTexStorage3D(gles.GL_TEXTURE_2D_ARRAY, 1, gles.GL_RGBA8, 2, 2, 2);
    try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());
    glTexParameteri(gles.GL_TEXTURE_2D_ARRAY, gles.GL_TEXTURE_MIN_FILTER, @intCast(gles.GL_NEAREST));
    glTexParameteri(gles.GL_TEXTURE_2D_ARRAY, gles.GL_TEXTURE_MAG_FILTER, @intCast(gles.GL_NEAREST));
    defer glDeleteTextures(1, &[_]gles.GLuint{tex});

    // glTexImage3D on immutable storage is rejected. glTexSubImage3D is allowed.
    glTexImage3D(gles.GL_TEXTURE_2D_ARRAY, 0, gles.GL_RGBA, 2, 2, 2, 0, gles.GL_RGBA, gles.GL_UNSIGNED_BYTE, null);
    try std.testing.expectEqual(gles.GL_INVALID_OPERATION, glGetError());
    // Layer 0 = red, layer 1 = blue.
    const red = [_]u8{ 255, 0, 0, 255 } ** 4;
    const blue = [_]u8{ 0, 0, 255, 255 } ** 4;
    glTexSubImage3D(gles.GL_TEXTURE_2D_ARRAY, 0, 0, 0, 0, 2, 2, 1, gles.GL_RGBA, gles.GL_UNSIGNED_BYTE, &red);
    glTexSubImage3D(gles.GL_TEXTURE_2D_ARRAY, 0, 0, 0, 1, 2, 2, 1, gles.GL_RGBA, gles.GL_UNSIGNED_BYTE, &blue);
    try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());

    const tri = [_]f32{ -1, -1, 3, -1, -1, 3 };
    var vbo: gles.GLuint = 0;
    glGenBuffers(1, @ptrCast(&vbo));
    glBindBuffer(gles.GL_ARRAY_BUFFER, vbo);
    glBufferData(gles.GL_ARRAY_BUFFER, @sizeOf(@TypeOf(tri)), &tri, gles.GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, gles.GL_FLOAT, gles.GL_FALSE, 0, @ptrFromInt(0));
    glEnableVertexAttribArray(0);
    defer glDeleteBuffers(1, &[_]gles.GLuint{vbo});
    glUniform1i(glGetUniformLocation(prog, "uArr"), 0);
    glUniform3f(glGetUniformLocation(prog, "uCoord"), 0.5, 0.5, 1.0); // layer 1
    glViewport(0, 0, 32, 32);
    glClearColor(0, 0, 0, 1);
    glClear(gles.GL_COLOR_BUFFER_BIT);
    glDrawArrays(gles.GL_TRIANGLES, 0, 3);
    try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());

    const s: *state.Surface = @ptrCast(@alignCast(surf.?));
    const dev = s.display.device.?;
    const px = try flushMap(dev, s);
    const c = (@as(usize, 16) * 32 + 16) * 4;
    // Sampling layer 1 -> blue (proves the immutable array storage + sub-updates work).
    try std.testing.expect(px[c + 2] > 200 and px[c + 0] < 60 and px[c + 1] < 60);
}

test "sampler2DShadow: GL_TEXTURE_COMPARE_MODE hardware depth compare (shadow mapping)" {
    // sampler2DShadow end-to-end: texture(uShadow, vec3(uv, ref)) does a hardware depth compare
    // (ref <op> stored_depth) and returns a scalar 1.0 (lit) / 0.0 (shadowed), not a vec4. The stored
    // "depth" here is a texel's R channel = 0.5; with GL_LEQUAL a ref of 0.2 passes (<=0.5 -> lit,
    // WHITE) and 0.9 fails (> 0.5 -> shadow, BLACK). Differential test of the whole chain: GLSL
    // sampler2DShadow -> Vulcan OpImageSampleDref -> the JIT sampler_shadow_fn -> sampleTextureShadow.
    _ = eglBindAPI(egl.EGL_OPENGL_ES_API);
    const dpy = eglGetPlatformDisplay(egl.EGL_PLATFORM_SURFACELESS_MESA, @ptrFromInt(0xE5AD0), null);
    try std.testing.expectEqual(egl.EGL_TRUE, eglInitialize(dpy, null, null));
    defer _ = eglTerminate(dpy);
    const cfg = state.configHandle(0);
    const ctx = eglCreateContext(dpy, cfg, egl.EGL_NO_CONTEXT, null);
    const pb_attribs = [_]EGLint{ egl.EGL_WIDTH, 32, egl.EGL_HEIGHT, 32, egl.EGL_NONE };
    const surf = eglCreatePbufferSurface(dpy, cfg, &pb_attribs);
    try std.testing.expectEqual(egl.EGL_TRUE, eglMakeCurrent(dpy, surf, surf, ctx));
    defer _ = eglDestroyContext(dpy, ctx);
    defer _ = eglDestroySurface(dpy, surf);
    defer _ = eglMakeCurrent(dpy, null, null, egl.EGL_NO_CONTEXT);

    const vs_src: [*:0]const gles.GLchar =
        \\attribute vec2 position;
        \\void main() { gl_Position = vec4(position, 0.0, 1.0); }
    ;
    const fs_src: [*:0]const gles.GLchar =
        \\precision mediump float;
        \\uniform sampler2DShadow uShadow;
        \\uniform vec2 uUv;
        \\uniform float uRef;
        \\void main() { float s = texture(uShadow, vec3(uUv, uRef)); gl_FragColor = vec4(s, s, s, 1.0); }
    ;
    const vs = glCreateShader(gles.GL_VERTEX_SHADER);
    glShaderSource(vs, 1, &[_]?[*:0]const gles.GLchar{vs_src}, null);
    glCompileShader(vs);
    const fs = glCreateShader(gles.GL_FRAGMENT_SHADER);
    glShaderSource(fs, 1, &[_]?[*:0]const gles.GLchar{fs_src}, null);
    glCompileShader(fs);
    const prog = glCreateProgram();
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glBindAttribLocation(prog, 0, "position");
    glLinkProgram(prog);
    glUseProgram(prog);
    defer {
        glDeleteProgram(prog);
        glDeleteShader(vs);
        glDeleteShader(fs);
    }

    // A 1x1 texel whose R channel = 128/255 ~ 0.5 is the "stored depth". Compare mode LEQUAL.
    const depth_texel = [_]u8{ 128, 0, 0, 255 };
    var tex: gles.GLuint = 0;
    glGenTextures(1, @ptrCast(&tex));
    glActiveTexture(gles.GL_TEXTURE0);
    glBindTexture(gles.GL_TEXTURE_2D, tex);
    glTexImage2D(gles.GL_TEXTURE_2D, 0, gles.GL_RGBA, 1, 1, 0, gles.GL_RGBA, gles.GL_UNSIGNED_BYTE, &depth_texel);
    glTexParameteri(gles.GL_TEXTURE_2D, gles.GL_TEXTURE_MIN_FILTER, gles.GL_NEAREST);
    glTexParameteri(gles.GL_TEXTURE_2D, gles.GL_TEXTURE_MAG_FILTER, gles.GL_NEAREST);
    glTexParameteri(gles.GL_TEXTURE_2D, gles.GL_TEXTURE_COMPARE_MODE, @intCast(gles.GL_COMPARE_REF_TO_TEXTURE));
    glTexParameteri(gles.GL_TEXTURE_2D, gles.GL_TEXTURE_COMPARE_FUNC, @intCast(gles.GL_LEQUAL));
    try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());
    // The compare state reads back.
    var qmode: gles.GLint = 0;
    glGetTexParameteriv(gles.GL_TEXTURE_2D, gles.GL_TEXTURE_COMPARE_MODE, @ptrCast(&qmode));
    try std.testing.expectEqual(@as(gles.GLint, @intCast(gles.GL_COMPARE_REF_TO_TEXTURE)), qmode);
    defer glDeleteTextures(1, &[_]gles.GLuint{tex});

    const tri = [_]f32{ -1, -1, 3, -1, -1, 3 };
    var vbo: gles.GLuint = 0;
    glGenBuffers(1, @ptrCast(&vbo));
    glBindBuffer(gles.GL_ARRAY_BUFFER, vbo);
    glBufferData(gles.GL_ARRAY_BUFFER, @sizeOf(@TypeOf(tri)), &tri, gles.GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, gles.GL_FLOAT, gles.GL_FALSE, 0, @ptrFromInt(0));
    glEnableVertexAttribArray(0);
    defer glDeleteBuffers(1, &[_]gles.GLuint{vbo});
    glUniform1i(glGetUniformLocation(prog, "uShadow"), 0);
    glUniform2f(glGetUniformLocation(prog, "uUv"), 0.5, 0.5);
    glViewport(0, 0, 32, 32);

    const s: *state.Surface = @ptrCast(@alignCast(surf.?));
    const dev = s.display.device.?;
    const c = (@as(usize, 16) * 32 + 16) * 4;

    // ref 0.2 <= 0.5 -> pass -> lit (white).
    glUniform1f(glGetUniformLocation(prog, "uRef"), 0.2);
    glClearColor(0, 0, 0, 1);
    glClear(gles.GL_COLOR_BUFFER_BIT);
    glDrawArrays(gles.GL_TRIANGLES, 0, 3);
    try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());
    {
        const px = try flushMap(dev, s);
        try std.testing.expect(px[c + 0] > 200); // lit
    }

    // ref 0.9 > 0.5 -> fail -> shadow (black).
    glUniform1f(glGetUniformLocation(prog, "uRef"), 0.9);
    glClear(gles.GL_COLOR_BUFFER_BIT);
    glDrawArrays(gles.GL_TRIANGLES, 0, 3);
    try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());
    {
        const px = try flushMap(dev, s);
        try std.testing.expect(px[c + 0] < 60); // in shadow
    }
}

test "samplerCubeShadow: GL_TEXTURE_COMPARE_MODE hardware cube depth compare (omni shadow mapping)" {
    // samplerCubeShadow end-to-end: texture(uShadow, vec4(dir, ref)) picks the cube face the direction
    // points at, does a hardware depth compare (ref <op> stored_depth) and returns a scalar 1.0 (lit) /
    // 0.0 (shadowed), not a vec4. Each face's texel R = 128/255 ~ 0.5 is the "depth". With GL_LEQUAL a
    // ref of 0.2 passes (<=0.5 -> lit, white) and 0.8 fails (> 0.5 -> shadow, black). Differential
    // test of the whole chain: GLSL samplerCubeShadow -> Vulcan OpImageSampleDref on a Cube image ->
    // the JIT sampler_cube_shadow_fn -> sampleTextureCubeShadow. Point uDir at +Z (face 4).
    _ = eglBindAPI(egl.EGL_OPENGL_ES_API);
    const dpy = eglGetPlatformDisplay(egl.EGL_PLATFORM_SURFACELESS_MESA, @ptrFromInt(0xCB5AD0), null);
    try std.testing.expectEqual(egl.EGL_TRUE, eglInitialize(dpy, null, null));
    defer _ = eglTerminate(dpy);
    const cfg = state.configHandle(0);
    const ctx = eglCreateContext(dpy, cfg, egl.EGL_NO_CONTEXT, null);
    const pb_attribs = [_]EGLint{ egl.EGL_WIDTH, 32, egl.EGL_HEIGHT, 32, egl.EGL_NONE };
    const surf = eglCreatePbufferSurface(dpy, cfg, &pb_attribs);
    try std.testing.expectEqual(egl.EGL_TRUE, eglMakeCurrent(dpy, surf, surf, ctx));
    defer _ = eglDestroyContext(dpy, ctx);
    defer _ = eglDestroySurface(dpy, surf);
    defer _ = eglMakeCurrent(dpy, null, null, egl.EGL_NO_CONTEXT);

    const vs_src: [*:0]const gles.GLchar =
        \\attribute vec2 position;
        \\void main() { gl_Position = vec4(position, 0.0, 1.0); }
    ;
    const fs_src: [*:0]const gles.GLchar =
        \\precision mediump float;
        \\uniform samplerCubeShadow uShadow;
        \\uniform vec3 uDir;
        \\uniform float uRef;
        \\void main() { float s = texture(uShadow, vec4(uDir, uRef)); gl_FragColor = vec4(s, s, s, 1.0); }
    ;
    const vs = glCreateShader(gles.GL_VERTEX_SHADER);
    glShaderSource(vs, 1, &[_]?[*:0]const gles.GLchar{vs_src}, null);
    glCompileShader(vs);
    const fs = glCreateShader(gles.GL_FRAGMENT_SHADER);
    glShaderSource(fs, 1, &[_]?[*:0]const gles.GLchar{fs_src}, null);
    glCompileShader(fs);
    const prog = glCreateProgram();
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glBindAttribLocation(prog, 0, "position");
    glLinkProgram(prog);
    glUseProgram(prog);
    defer {
        glDeleteProgram(prog);
        glDeleteShader(vs);
        glDeleteShader(fs);
    }

    // 6 single-texel faces, each R channel = 128/255 ~ 0.5 = the stored depth. Compare mode LEQUAL.
    const face_targets = [6]gles.GLenum{
        gles.GL_TEXTURE_CUBE_MAP_POSITIVE_X, gles.GL_TEXTURE_CUBE_MAP_NEGATIVE_X,
        gles.GL_TEXTURE_CUBE_MAP_POSITIVE_Y, gles.GL_TEXTURE_CUBE_MAP_NEGATIVE_Y,
        gles.GL_TEXTURE_CUBE_MAP_POSITIVE_Z, gles.GL_TEXTURE_CUBE_MAP_NEGATIVE_Z,
    };
    var cube: gles.GLuint = 0;
    glGenTextures(1, @ptrCast(&cube));
    glActiveTexture(gles.GL_TEXTURE0);
    glBindTexture(gles.GL_TEXTURE_CUBE_MAP, cube);
    const depth_texel = [_]u8{ 128, 0, 0, 255 };
    for (0..6) |f| {
        glTexImage2D(face_targets[f], 0, gles.GL_RGBA, 1, 1, 0, gles.GL_RGBA, gles.GL_UNSIGNED_BYTE, &depth_texel);
    }
    glTexParameteri(gles.GL_TEXTURE_CUBE_MAP, gles.GL_TEXTURE_MIN_FILTER, gles.GL_NEAREST);
    glTexParameteri(gles.GL_TEXTURE_CUBE_MAP, gles.GL_TEXTURE_MAG_FILTER, gles.GL_NEAREST);
    glTexParameteri(gles.GL_TEXTURE_CUBE_MAP, gles.GL_TEXTURE_COMPARE_MODE, @intCast(gles.GL_COMPARE_REF_TO_TEXTURE));
    glTexParameteri(gles.GL_TEXTURE_CUBE_MAP, gles.GL_TEXTURE_COMPARE_FUNC, @intCast(gles.GL_LEQUAL));
    try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());
    // The compare state reads back.
    var qmode: gles.GLint = 0;
    glGetTexParameteriv(gles.GL_TEXTURE_CUBE_MAP, gles.GL_TEXTURE_COMPARE_MODE, @ptrCast(&qmode));
    try std.testing.expectEqual(@as(gles.GLint, @intCast(gles.GL_COMPARE_REF_TO_TEXTURE)), qmode);
    defer glDeleteTextures(1, &[_]gles.GLuint{cube});

    const tri = [_]f32{ -1, -1, 3, -1, -1, 3 };
    var vbo: gles.GLuint = 0;
    glGenBuffers(1, @ptrCast(&vbo));
    glBindBuffer(gles.GL_ARRAY_BUFFER, vbo);
    glBufferData(gles.GL_ARRAY_BUFFER, @sizeOf(@TypeOf(tri)), &tri, gles.GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, gles.GL_FLOAT, gles.GL_FALSE, 0, @ptrFromInt(0));
    glEnableVertexAttribArray(0);
    defer glDeleteBuffers(1, &[_]gles.GLuint{vbo});
    glUniform1i(glGetUniformLocation(prog, "uShadow"), 0);
    glUniform3f(glGetUniformLocation(prog, "uDir"), 0.0, 0.0, 1.0); // +Z face
    glViewport(0, 0, 32, 32);

    const s: *state.Surface = @ptrCast(@alignCast(surf.?));
    const dev = s.display.device.?;
    const c = (@as(usize, 16) * 32 + 16) * 4;

    // ref 0.2 <= 0.5 -> pass -> lit (white).
    glUniform1f(glGetUniformLocation(prog, "uRef"), 0.2);
    glClearColor(0, 0, 0, 1);
    glClear(gles.GL_COLOR_BUFFER_BIT);
    glDrawArrays(gles.GL_TRIANGLES, 0, 3);
    try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());
    {
        const px = try flushMap(dev, s);
        try std.testing.expect(px[c + 0] > 200); // lit
    }

    // ref 0.8 > 0.5 -> fail -> shadow (black).
    glUniform1f(glGetUniformLocation(prog, "uRef"), 0.8);
    glClear(gles.GL_COLOR_BUFFER_BIT);
    glDrawArrays(gles.GL_TRIANGLES, 0, 3);
    try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());
    {
        const px = try flushMap(dev, s);
        try std.testing.expect(px[c + 0] < 60); // in shadow
    }
}

test "sampler2DArrayShadow: GL_TEXTURE_COMPARE_MODE hardware 2D-array depth compare (cascaded shadow mapping)" {
    // sampler2DArrayShadow end-to-end: texture(uShadow, vec4(uv, layer, ref)) selects array `layer`,
    // does a hardware depth compare (ref <op> stored_depth) and returns a scalar 1.0 (lit) / 0.0
    // (shadowed), not a vec4. Layer 1's texel R = 128/255 ~ 0.5 is the "depth". With GL_LEQUAL a ref of
    // 0.2 passes (<=0.5 -> lit, white) and 0.8 fails (> 0.5 -> shadow, black). Differential test of the
    // whole chain: GLSL sampler2DArrayShadow -> Vulcan OpImageSampleDref on a 2D-Arrayed image -> the JIT
    // sampler_2darray_shadow_fn -> sampleTexture2dArrayShadow. Point uUvLayer at (0.5, 0.5, layer 1).
    _ = eglBindAPI(egl.EGL_OPENGL_ES_API);
    const dpy = eglGetPlatformDisplay(egl.EGL_PLATFORM_SURFACELESS_MESA, @ptrFromInt(0xA45AD0), null);
    try std.testing.expectEqual(egl.EGL_TRUE, eglInitialize(dpy, null, null));
    defer _ = eglTerminate(dpy);
    const cfg = state.configHandle(0);
    const ctx = eglCreateContext(dpy, cfg, egl.EGL_NO_CONTEXT, null);
    const pb_attribs = [_]EGLint{ egl.EGL_WIDTH, 32, egl.EGL_HEIGHT, 32, egl.EGL_NONE };
    const surf = eglCreatePbufferSurface(dpy, cfg, &pb_attribs);
    try std.testing.expectEqual(egl.EGL_TRUE, eglMakeCurrent(dpy, surf, surf, ctx));
    defer _ = eglDestroyContext(dpy, ctx);
    defer _ = eglDestroySurface(dpy, surf);
    defer _ = eglMakeCurrent(dpy, null, null, egl.EGL_NO_CONTEXT);

    const vs_src: [*:0]const gles.GLchar =
        \\attribute vec2 position;
        \\void main() { gl_Position = vec4(position, 0.0, 1.0); }
    ;
    const fs_src: [*:0]const gles.GLchar =
        \\precision mediump float;
        \\uniform sampler2DArrayShadow uShadow;
        \\uniform vec3 uUvLayer;
        \\uniform float uRef;
        \\void main() { float s = texture(uShadow, vec4(uUvLayer, uRef)); gl_FragColor = vec4(s, s, s, 1.0); }
    ;
    const vs = glCreateShader(gles.GL_VERTEX_SHADER);
    glShaderSource(vs, 1, &[_]?[*:0]const gles.GLchar{vs_src}, null);
    glCompileShader(vs);
    const fs = glCreateShader(gles.GL_FRAGMENT_SHADER);
    glShaderSource(fs, 1, &[_]?[*:0]const gles.GLchar{fs_src}, null);
    glCompileShader(fs);
    const prog = glCreateProgram();
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glBindAttribLocation(prog, 0, "position");
    glLinkProgram(prog);
    glUseProgram(prog);
    defer {
        glDeleteProgram(prog);
        glDeleteShader(vs);
        glDeleteShader(fs);
    }

    // A 2-layer 1x1 array: layer 0 R = 0 (dummy), layer 1's R = 128/255 ~ 0.5 = the stored depth we
    // sample. Compare mode LEQUAL.
    var vol: [1 * 1 * 2 * 4]u8 = .{ 0, 0, 0, 255, 128, 0, 0, 255 };
    var tex: gles.GLuint = 0;
    glGenTextures(1, @ptrCast(&tex));
    glActiveTexture(gles.GL_TEXTURE0);
    glBindTexture(gles.GL_TEXTURE_2D_ARRAY, tex);
    glTexImage3D(gles.GL_TEXTURE_2D_ARRAY, 0, gles.GL_RGBA, 1, 1, 2, 0, gles.GL_RGBA, gles.GL_UNSIGNED_BYTE, &vol);
    glTexParameteri(gles.GL_TEXTURE_2D_ARRAY, gles.GL_TEXTURE_MIN_FILTER, @intCast(gles.GL_NEAREST));
    glTexParameteri(gles.GL_TEXTURE_2D_ARRAY, gles.GL_TEXTURE_MAG_FILTER, @intCast(gles.GL_NEAREST));
    glTexParameteri(gles.GL_TEXTURE_2D_ARRAY, gles.GL_TEXTURE_COMPARE_MODE, @intCast(gles.GL_COMPARE_REF_TO_TEXTURE));
    glTexParameteri(gles.GL_TEXTURE_2D_ARRAY, gles.GL_TEXTURE_COMPARE_FUNC, @intCast(gles.GL_LEQUAL));
    try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());
    // The compare state reads back.
    var qmode: gles.GLint = 0;
    glGetTexParameteriv(gles.GL_TEXTURE_2D_ARRAY, gles.GL_TEXTURE_COMPARE_MODE, @ptrCast(&qmode));
    try std.testing.expectEqual(@as(gles.GLint, @intCast(gles.GL_COMPARE_REF_TO_TEXTURE)), qmode);
    defer glDeleteTextures(1, &[_]gles.GLuint{tex});

    const tri = [_]f32{ -1, -1, 3, -1, -1, 3 };
    var vbo: gles.GLuint = 0;
    glGenBuffers(1, @ptrCast(&vbo));
    glBindBuffer(gles.GL_ARRAY_BUFFER, vbo);
    glBufferData(gles.GL_ARRAY_BUFFER, @sizeOf(@TypeOf(tri)), &tri, gles.GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, gles.GL_FLOAT, gles.GL_FALSE, 0, @ptrFromInt(0));
    glEnableVertexAttribArray(0);
    defer glDeleteBuffers(1, &[_]gles.GLuint{vbo});
    glUniform1i(glGetUniformLocation(prog, "uShadow"), 0);
    glUniform3f(glGetUniformLocation(prog, "uUvLayer"), 0.5, 0.5, 1.0); // layer 1
    glViewport(0, 0, 32, 32);

    const s: *state.Surface = @ptrCast(@alignCast(surf.?));
    const dev = s.display.device.?;
    const c = (@as(usize, 16) * 32 + 16) * 4;

    // ref 0.2 <= 0.5 -> pass -> lit (white).
    glUniform1f(glGetUniformLocation(prog, "uRef"), 0.2);
    glClearColor(0, 0, 0, 1);
    glClear(gles.GL_COLOR_BUFFER_BIT);
    glDrawArrays(gles.GL_TRIANGLES, 0, 3);
    try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());
    {
        const px = try flushMap(dev, s);
        try std.testing.expect(px[c + 0] > 200); // lit
    }

    // ref 0.8 > 0.5 -> fail -> shadow (black).
    glUniform1f(glGetUniformLocation(prog, "uRef"), 0.8);
    glClear(gles.GL_COLOR_BUFFER_BIT);
    glDrawArrays(gles.GL_TRIANGLES, 0, 3);
    try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());
    {
        const px = try flushMap(dev, s);
        try std.testing.expect(px[c + 0] < 60); // in shadow
    }
}

test "sampler2DShadow PCF: a LINEAR filter compares each tap THEN blends (soft shadow edge)" {
    // True PCF: with a LINEAR filter the 4 footprint texels are each depth-compared and the 0/1
    // results are bilinear-blended (compare-THEN-filter). A 2-wide depth map [0.8 | 0.2] sampled at
    // the boundary with ref 0.5 (LEQUAL): the left tap passes (0.5<=0.8 -> 1) and the right fails
    // (0.5<=0.2 -> 0), so PCF blends to ~0.5 (a mid gray, a soft edge). The old filter-THEN-compare
    // would average the depths first (0.5) then compare (0.5<=0.5 -> fully lit 1.0), so a mid gray
    // here is the differential proof that the compare happens per-tap.
    _ = eglBindAPI(egl.EGL_OPENGL_ES_API);
    const dpy = eglGetPlatformDisplay(egl.EGL_PLATFORM_SURFACELESS_MESA, @ptrFromInt(0xE5AD2), null);
    try std.testing.expectEqual(egl.EGL_TRUE, eglInitialize(dpy, null, null));
    defer _ = eglTerminate(dpy);
    const cfg = state.configHandle(0);
    const ctx = eglCreateContext(dpy, cfg, egl.EGL_NO_CONTEXT, null);
    const pb_attribs = [_]EGLint{ egl.EGL_WIDTH, 32, egl.EGL_HEIGHT, 32, egl.EGL_NONE };
    const surf = eglCreatePbufferSurface(dpy, cfg, &pb_attribs);
    try std.testing.expectEqual(egl.EGL_TRUE, eglMakeCurrent(dpy, surf, surf, ctx));
    defer _ = eglDestroyContext(dpy, ctx);
    defer _ = eglDestroySurface(dpy, surf);
    defer _ = eglMakeCurrent(dpy, null, null, egl.EGL_NO_CONTEXT);

    const vs_src: [*:0]const gles.GLchar =
        \\attribute vec2 position;
        \\void main() { gl_Position = vec4(position, 0.0, 1.0); }
    ;
    const fs_src: [*:0]const gles.GLchar =
        \\precision mediump float;
        \\uniform sampler2DShadow uShadow;
        \\uniform vec2 uUv;
        \\uniform float uRef;
        \\void main() { float s = texture(uShadow, vec3(uUv, uRef)); gl_FragColor = vec4(s, s, s, 1.0); }
    ;
    const vs = glCreateShader(gles.GL_VERTEX_SHADER);
    glShaderSource(vs, 1, &[_]?[*:0]const gles.GLchar{vs_src}, null);
    glCompileShader(vs);
    const fs = glCreateShader(gles.GL_FRAGMENT_SHADER);
    glShaderSource(fs, 1, &[_]?[*:0]const gles.GLchar{fs_src}, null);
    glCompileShader(fs);
    const prog = glCreateProgram();
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glBindAttribLocation(prog, 0, "position");
    glLinkProgram(prog);
    glUseProgram(prog);
    defer {
        glDeleteProgram(prog);
        glDeleteShader(vs);
        glDeleteShader(fs);
    }

    // A 2x2 depth map: column 0 = 0.8 (R=204), column 1 = 0.2 (R=51); both rows identical.
    const far = [_]u8{ 204, 0, 0, 255 };
    const near = [_]u8{ 51, 0, 0, 255 };
    const texels = far ++ near ++ far ++ near; // (0,0)(1,0)(0,1)(1,1)
    var tex: gles.GLuint = 0;
    glGenTextures(1, @ptrCast(&tex));
    glActiveTexture(gles.GL_TEXTURE0);
    glBindTexture(gles.GL_TEXTURE_2D, tex);
    glTexImage2D(gles.GL_TEXTURE_2D, 0, gles.GL_RGBA, 2, 2, 0, gles.GL_RGBA, gles.GL_UNSIGNED_BYTE, &texels);
    glTexParameteri(gles.GL_TEXTURE_2D, gles.GL_TEXTURE_MIN_FILTER, gles.GL_LINEAR);
    glTexParameteri(gles.GL_TEXTURE_2D, gles.GL_TEXTURE_MAG_FILTER, gles.GL_LINEAR);
    glTexParameteri(gles.GL_TEXTURE_2D, gles.GL_TEXTURE_WRAP_S, @intCast(gles.GL_CLAMP_TO_EDGE));
    glTexParameteri(gles.GL_TEXTURE_2D, gles.GL_TEXTURE_WRAP_T, @intCast(gles.GL_CLAMP_TO_EDGE));
    glTexParameteri(gles.GL_TEXTURE_2D, gles.GL_TEXTURE_COMPARE_MODE, @intCast(gles.GL_COMPARE_REF_TO_TEXTURE));
    glTexParameteri(gles.GL_TEXTURE_2D, gles.GL_TEXTURE_COMPARE_FUNC, @intCast(gles.GL_LEQUAL));
    try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());
    defer glDeleteTextures(1, &[_]gles.GLuint{tex});

    const tri = [_]f32{ -1, -1, 3, -1, -1, 3 };
    var vbo: gles.GLuint = 0;
    glGenBuffers(1, @ptrCast(&vbo));
    glBindBuffer(gles.GL_ARRAY_BUFFER, vbo);
    glBufferData(gles.GL_ARRAY_BUFFER, @sizeOf(@TypeOf(tri)), &tri, gles.GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, gles.GL_FLOAT, gles.GL_FALSE, 0, @ptrFromInt(0));
    glEnableVertexAttribArray(0);
    defer glDeleteBuffers(1, &[_]gles.GLuint{vbo});
    glUniform1i(glGetUniformLocation(prog, "uShadow"), 0);
    // u=0.5 sits exactly between the two texel columns -> the footprint straddles both taps.
    glUniform2f(glGetUniformLocation(prog, "uUv"), 0.5, 0.5);
    glUniform1f(glGetUniformLocation(prog, "uRef"), 0.5);
    glViewport(0, 0, 32, 32);
    glClearColor(0, 0, 0, 1);
    glClear(gles.GL_COLOR_BUFFER_BIT);
    glDrawArrays(gles.GL_TRIANGLES, 0, 3);
    try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());

    const s: *state.Surface = @ptrCast(@alignCast(surf.?));
    const dev = s.display.device.?;
    const c = (@as(usize, 16) * 32 + 16) * 4;
    const px = try flushMap(dev, s);
    // ~0.5 = a MID gray (PCF blended 1 and 0). NOT the fully-lit 255 that filter-then-compare gives.
    try std.testing.expect(px[c + 0] > 90 and px[c + 0] < 165);
}

test "sampler2DShadow real depth-texture shadow map: render depth into an FBO, then shadow-lookup it" {
    // Real shadow-mapping pipeline (no RGBA stand-in): pass 1 renders a full-screen quad at a
    // known window depth (0.5) into a GL_DEPTH_COMPONENT texture attached to an FBO. Pass 2 binds that
    // depth texture as a sampler2DShadow with GL_COMPARE_REF_TO_TEXTURE and looks it up. ref 0.2 is
    // in front of the stored 0.5 (LEQUAL passes -> lit white), ref 0.8 is behind it (fails -> shadow
    // black). Proves the compare works against a genuine rendered depth texture end-to-end.
    _ = eglBindAPI(egl.EGL_OPENGL_ES_API);
    const dpy = eglGetPlatformDisplay(egl.EGL_PLATFORM_SURFACELESS_MESA, @ptrFromInt(0xE5AD4), null);
    try std.testing.expectEqual(egl.EGL_TRUE, eglInitialize(dpy, null, null));
    defer _ = eglTerminate(dpy);
    const cfg = state.configHandle(0);
    const ctx = eglCreateContext(dpy, cfg, egl.EGL_NO_CONTEXT, null);
    const pb_attribs = [_]EGLint{ egl.EGL_WIDTH, 32, egl.EGL_HEIGHT, 32, egl.EGL_NONE };
    const surf = eglCreatePbufferSurface(dpy, cfg, &pb_attribs);
    try std.testing.expectEqual(egl.EGL_TRUE, eglMakeCurrent(dpy, surf, surf, ctx));
    defer _ = eglDestroyContext(dpy, ctx);
    defer _ = eglDestroySurface(dpy, surf);
    defer _ = eglMakeCurrent(dpy, null, null, egl.EGL_NO_CONTEXT);

    // Pass 1 program: a VS that plants gl_Position.z = uZ (window depth = uZ in Prism's [0,1] clip).
    const depth_vs: [*:0]const gles.GLchar =
        \\attribute vec2 position;
        \\uniform float uZ;
        \\void main() { gl_Position = vec4(position, uZ, 1.0); }
    ;
    const white_fs: [*:0]const gles.GLchar =
        \\precision mediump float;
        \\void main() { gl_FragColor = vec4(1.0); }
    ;
    const dvs = glCreateShader(gles.GL_VERTEX_SHADER);
    glShaderSource(dvs, 1, &[_]?[*:0]const gles.GLchar{depth_vs}, null);
    glCompileShader(dvs);
    const dfs = glCreateShader(gles.GL_FRAGMENT_SHADER);
    glShaderSource(dfs, 1, &[_]?[*:0]const gles.GLchar{white_fs}, null);
    glCompileShader(dfs);
    const dprog = glCreateProgram();
    glAttachShader(dprog, dvs);
    glAttachShader(dprog, dfs);
    glBindAttribLocation(dprog, 0, "position");
    glLinkProgram(dprog);

    // Pass 2 program: the sampler2DShadow lookup.
    const shadow_vs: [*:0]const gles.GLchar =
        \\attribute vec2 position;
        \\void main() { gl_Position = vec4(position, 0.0, 1.0); }
    ;
    const shadow_fs: [*:0]const gles.GLchar =
        \\precision mediump float;
        \\uniform sampler2DShadow uShadow;
        \\uniform vec2 uUv;
        \\uniform float uRef;
        \\void main() { float s = texture(uShadow, vec3(uUv, uRef)); gl_FragColor = vec4(s, s, s, 1.0); }
    ;
    const svs = glCreateShader(gles.GL_VERTEX_SHADER);
    glShaderSource(svs, 1, &[_]?[*:0]const gles.GLchar{shadow_vs}, null);
    glCompileShader(svs);
    const sfs = glCreateShader(gles.GL_FRAGMENT_SHADER);
    glShaderSource(sfs, 1, &[_]?[*:0]const gles.GLchar{shadow_fs}, null);
    glCompileShader(sfs);
    const sprog = glCreateProgram();
    glAttachShader(sprog, svs);
    glAttachShader(sprog, sfs);
    glBindAttribLocation(sprog, 0, "position");
    glLinkProgram(sprog);

    const SZ: gles.GLsizei = 16;
    // A GL_DEPTH_COMPONENT depth texture + a color texture as the FBO attachments.
    var dtex: gles.GLuint = 0;
    glGenTextures(1, @ptrCast(&dtex));
    glActiveTexture(gles.GL_TEXTURE0);
    glBindTexture(gles.GL_TEXTURE_2D, dtex);
    glTexImage2D(gles.GL_TEXTURE_2D, 0, @bitCast(gles.GL_DEPTH_COMPONENT), SZ, SZ, 0, gles.GL_DEPTH_COMPONENT, gles.GL_UNSIGNED_SHORT, null);
    glTexParameteri(gles.GL_TEXTURE_2D, gles.GL_TEXTURE_MIN_FILTER, gles.GL_NEAREST);
    glTexParameteri(gles.GL_TEXTURE_2D, gles.GL_TEXTURE_MAG_FILTER, gles.GL_NEAREST);
    var ctex: gles.GLuint = 0;
    glGenTextures(1, @ptrCast(&ctex));
    glBindTexture(gles.GL_TEXTURE_2D, ctex);
    glTexImage2D(gles.GL_TEXTURE_2D, 0, gles.GL_RGBA, SZ, SZ, 0, gles.GL_RGBA, gles.GL_UNSIGNED_BYTE, null);
    var fbo: gles.GLuint = 0;
    glGenFramebuffers(1, @ptrCast(&fbo));
    glBindFramebuffer(gles.GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(gles.GL_FRAMEBUFFER, gles.GL_COLOR_ATTACHMENT0, gles.GL_TEXTURE_2D, ctex, 0);
    glFramebufferTexture2D(gles.GL_FRAMEBUFFER, gles.GL_DEPTH_ATTACHMENT, gles.GL_TEXTURE_2D, dtex, 0);
    defer {
        glDeleteFramebuffers(1, &[_]gles.GLuint{fbo});
        glDeleteTextures(1, &[_]gles.GLuint{dtex});
        glDeleteTextures(1, &[_]gles.GLuint{ctex});
    }

    const tri = [_]f32{ -1, -1, 3, -1, -1, 3 };
    var vbo: gles.GLuint = 0;
    glGenBuffers(1, @ptrCast(&vbo));
    glBindBuffer(gles.GL_ARRAY_BUFFER, vbo);
    glBufferData(gles.GL_ARRAY_BUFFER, @sizeOf(@TypeOf(tri)), &tri, gles.GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, gles.GL_FLOAT, gles.GL_FALSE, 0, @ptrFromInt(0));
    glEnableVertexAttribArray(0);
    defer glDeleteBuffers(1, &[_]gles.GLuint{vbo});

    // Pass 1: render window-depth 0.5 into the depth texture.
    glUseProgram(dprog);
    glViewport(0, 0, SZ, SZ);
    glEnable(gles.GL_DEPTH_TEST);
    glDepthFunc(gles.GL_LESS);
    glClearColor(0, 0, 0, 1);
    glClear(gles.GL_COLOR_BUFFER_BIT | gles.GL_DEPTH_BUFFER_BIT);
    glUniform1f(glGetUniformLocation(dprog, "uZ"), 0.5);
    glDrawArrays(gles.GL_TRIANGLES, 0, 3);
    glDisable(gles.GL_DEPTH_TEST);
    try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());

    // Pass 2: the depth texture becomes a sampler2DShadow (compare mode LEQUAL).
    glBindFramebuffer(gles.GL_FRAMEBUFFER, 0);
    glActiveTexture(gles.GL_TEXTURE0);
    glBindTexture(gles.GL_TEXTURE_2D, dtex);
    glTexParameteri(gles.GL_TEXTURE_2D, gles.GL_TEXTURE_COMPARE_MODE, @intCast(gles.GL_COMPARE_REF_TO_TEXTURE));
    glTexParameteri(gles.GL_TEXTURE_2D, gles.GL_TEXTURE_COMPARE_FUNC, @intCast(gles.GL_LEQUAL));
    glUseProgram(sprog);
    glViewport(0, 0, 32, 32);
    glUniform1i(glGetUniformLocation(sprog, "uShadow"), 0);
    glUniform2f(glGetUniformLocation(sprog, "uUv"), 0.5, 0.5);

    const s: *state.Surface = @ptrCast(@alignCast(surf.?));
    const dev = s.display.device.?;
    const c = (@as(usize, 16) * 32 + 16) * 4;

    // ref 0.2 in front of the stored 0.5 -> lit (white).
    glUniform1f(glGetUniformLocation(sprog, "uRef"), 0.2);
    glClear(gles.GL_COLOR_BUFFER_BIT);
    glDrawArrays(gles.GL_TRIANGLES, 0, 3);
    try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());
    {
        const px = try flushMap(dev, s);
        try std.testing.expect(px[c + 0] > 200); // lit
    }

    // ref 0.8 behind the stored 0.5 -> shadow (black).
    glUniform1f(glGetUniformLocation(sprog, "uRef"), 0.8);
    glClear(gles.GL_COLOR_BUFFER_BIT);
    glDrawArrays(gles.GL_TRIANGLES, 0, 3);
    try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());
    {
        const px = try flushMap(dev, s);
        try std.testing.expect(px[c + 0] < 60); // in shadow
    }
}

test "shadow-mapped scene: two-pass light-space shadow map casts a correctly-placed shadow on the ground" {
    // Software oracle: deterministic (pinned_driver defaults to "software" under is_test). Full
    // scene (grey ground + blue occluder) with the blue-occluder assertions.
    try runShadowMappedScene(0x5AD00, false);
}

test "glViewport sub-rectangle: a fullscreen draw fills ONLY the viewport rect and is clipped outside (software)" {
    try runViewportSubrect(0x71E409);
}

test "glViewport sub-rectangle on NVIDIA GPU: the sub-viewport transform + clip act on the real RTX (skips without a GPU)" {
    const saved = state.pinned_driver;
    state.pinned_driver = "nvidia";
    defer state.pinned_driver = saved;
    try runViewportSubrect(0x71E40A);
}

fn runViewportSubrect(dpy_magic: usize) !void {
    _ = eglBindAPI(egl.EGL_OPENGL_ES_API);
    const dpy = eglGetPlatformDisplay(egl.EGL_PLATFORM_SURFACELESS_MESA, @ptrFromInt(dpy_magic), null);
    if (eglInitialize(dpy, null, null) != egl.EGL_TRUE) return error.SkipZigTest;
    defer _ = eglTerminate(dpy);
    const cfg = state.configHandle(0);
    const ctx = eglCreateContext(dpy, cfg, egl.EGL_NO_CONTEXT, null);
    const W: EGLint = 64;
    const pb = [_]EGLint{ egl.EGL_WIDTH, W, egl.EGL_HEIGHT, W, egl.EGL_NONE };
    const surf = eglCreatePbufferSurface(dpy, cfg, &pb);
    _ = eglMakeCurrent(dpy, surf, surf, ctx);
    defer _ = eglDestroyContext(dpy, ctx);
    defer _ = eglDestroySurface(dpy, surf);
    defer _ = eglMakeCurrent(dpy, null, null, egl.EGL_NO_CONTEXT);

    const vs: [*:0]const gles.GLchar = "attribute vec2 p; void main(){ gl_Position=vec4(p,0.0,1.0); }";
    const fs: [*:0]const gles.GLchar = "precision mediump float; void main(){ gl_FragColor=vec4(1.0,0.0,0.0,1.0); }";
    const v = glCreateShader(gles.GL_VERTEX_SHADER);
    glShaderSource(v, 1, &[_]?[*:0]const gles.GLchar{vs}, null);
    glCompileShader(v);
    const f = glCreateShader(gles.GL_FRAGMENT_SHADER);
    glShaderSource(f, 1, &[_]?[*:0]const gles.GLchar{fs}, null);
    glCompileShader(f);
    const prog = glCreateProgram();
    glAttachShader(prog, v);
    glAttachShader(prog, f);
    glBindAttribLocation(prog, 0, "p");
    glLinkProgram(prog);
    glUseProgram(prog);
    // A big triangle that covers the whole clip volume [-1,1]^2, so the viewport rect is what limits it.
    const tri = [_]f32{ -1, -1, 3, -1, -1, 3 };
    var vbo: gles.GLuint = 0;
    glGenBuffers(1, @ptrCast(&vbo));
    glBindBuffer(gles.GL_ARRAY_BUFFER, vbo);
    glBufferData(gles.GL_ARRAY_BUFFER, @sizeOf(@TypeOf(tri)), &tri, gles.GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, gles.GL_FLOAT, gles.GL_FALSE, 0, @ptrFromInt(0));
    glEnableVertexAttribArray(0);

    glClearColor(0, 0, 0, 1);
    glClear(gles.GL_COLOR_BUFFER_BIT);
    // GL viewport (0,0,32,32) = the bottom-left quadrant (GL bottom-left origin). In Prism's top-left
    // window that is cols [0,32), rows [32,64): the lower-left quadrant of the readback image.
    glViewport(0, 0, 32, 32);
    glDrawArrays(gles.GL_TRIANGLES, 0, 3);
    try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());

    const s: *state.Surface = @ptrCast(@alignCast(surf.?));
    const dev = s.display.device.?;
    const px = try flushMap(dev, s);
    const at = struct {
        fn red(buf: []const u8, x: usize, y: usize) bool {
            const i = (y * 64 + x) * 4;
            return buf[i] > 200 and buf[i + 1] < 60 and buf[i + 2] < 60;
        }
    }.red;
    // Inside the viewport (lower-left quadrant): red.
    try std.testing.expect(at(px, 16, 48));
    // Outside the viewport: clipped -> black. Check the other three quadrants.
    try std.testing.expect(!at(px, 48, 48)); // lower-right
    try std.testing.expect(!at(px, 16, 16)); // upper-left
    try std.testing.expect(!at(px, 48, 16)); // upper-right
    glDeleteProgram(prog);
    glDeleteBuffers(1, &[_]gles.GLuint{vbo});
}

test "glDepthRangef: remaps NDC z into the window depth range so a farther-NDC draw can win GL_LESS (software)" {
    try runDepthRange(0xDE7409);
}

test "glDepthRangef on NVIDIA GPU: the SET_VIEWPORT scale/offset Z remaps depth on the real RTX (skips without a GPU)" {
    const saved = state.pinned_driver;
    state.pinned_driver = "nvidia";
    defer state.pinned_driver = saved;
    try runDepthRange(0xDE740A);
}

fn runDepthRange(dpy_magic: usize) !void {
    _ = eglBindAPI(egl.EGL_OPENGL_ES_API);
    const dpy = eglGetPlatformDisplay(egl.EGL_PLATFORM_SURFACELESS_MESA, @ptrFromInt(dpy_magic), null);
    if (eglInitialize(dpy, null, null) != egl.EGL_TRUE) return error.SkipZigTest;
    defer _ = eglTerminate(dpy);
    const cfg = state.configHandle(1); // RGBA8 + D24
    const ctx = eglCreateContext(dpy, cfg, egl.EGL_NO_CONTEXT, null);
    const pb = [_]EGLint{ egl.EGL_WIDTH, 64, egl.EGL_HEIGHT, 64, egl.EGL_NONE };
    const surf = eglCreatePbufferSurface(dpy, cfg, &pb);
    _ = eglMakeCurrent(dpy, surf, surf, ctx);
    defer _ = eglDestroyContext(dpy, ctx);
    defer _ = eglDestroySurface(dpy, surf);
    defer _ = eglMakeCurrent(dpy, null, null, egl.EGL_NO_CONTEXT);

    const vs: [*:0]const gles.GLchar = "attribute vec2 p; void main(){ gl_Position=vec4(p,0.0,1.0); }"; // z_ndc = 0
    const fs: [*:0]const gles.GLchar = "precision mediump float; uniform vec3 uC; void main(){ gl_FragColor=vec4(uC,1.0); }";
    const v = glCreateShader(gles.GL_VERTEX_SHADER);
    glShaderSource(v, 1, &[_]?[*:0]const gles.GLchar{vs}, null);
    glCompileShader(v);
    const f = glCreateShader(gles.GL_FRAGMENT_SHADER);
    glShaderSource(f, 1, &[_]?[*:0]const gles.GLchar{fs}, null);
    glCompileShader(f);
    const prog = glCreateProgram();
    glAttachShader(prog, v);
    glAttachShader(prog, f);
    glBindAttribLocation(prog, 0, "p");
    glLinkProgram(prog);
    glUseProgram(prog);
    const tri = [_]f32{ -1, -1, 3, -1, -1, 3 }; // fullscreen
    var vbo: gles.GLuint = 0;
    glGenBuffers(1, @ptrCast(&vbo));
    glBindBuffer(gles.GL_ARRAY_BUFFER, vbo);
    glBufferData(gles.GL_ARRAY_BUFFER, @sizeOf(@TypeOf(tri)), &tri, gles.GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, gles.GL_FLOAT, gles.GL_FALSE, 0, @ptrFromInt(0));
    glEnableVertexAttribArray(0);
    glEnable(gles.GL_DEPTH_TEST);
    glDepthFunc(gles.GL_LESS);
    const uC = glGetUniformLocation(prog, "uC");

    glClearColor(0, 0, 0, 1);
    glClearDepthf(1.0);
    glClear(gles.GL_COLOR_BUFFER_BIT | gles.GL_DEPTH_BUFFER_BIT);
    // Draw red with depth range [0.5, 1.0]: z_ndc 0 -> window z 0.5. Passes vs the 1.0 clear, writes 0.5.
    glUniform3f(uC, 1, 0, 0);
    glDepthRangef(0.5, 1.0);
    glDrawArrays(gles.GL_TRIANGLES, 0, 3);
    // Draw green with depth range [0.0, 0.5]: z_ndc 0 -> window z 0.0 < 0.5, so it wins GL_LESS.
    // Without depth-range support both map to z_ndc 0, and green (0 < 0) fails -> stays red.
    glUniform3f(uC, 0, 1, 0);
    glDepthRangef(0.0, 0.5);
    glDrawArrays(gles.GL_TRIANGLES, 0, 3);
    try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());

    const s: *state.Surface = @ptrCast(@alignCast(surf.?));
    const dev = s.display.device.?;
    const px = try flushMap(dev, s);
    const c = (32 * 64 + 32) * 4;
    // GREEN won: the nearer window-z draw (via the remapped depth range) survived GL_LESS.
    try std.testing.expect(px[c + 1] > 200 and px[c + 0] < 60); // green, not red
    glDeleteProgram(prog);
    glDeleteBuffers(1, &[_]gles.GLuint{vbo});
}

// GPU per-draw throughput micro-benchmark. Skipped unless PRISM_BENCH is set (it forces the
// nvidia driver + needs the real GPU, and adds ~1000 GPU draws to a suite run). Run it with:
//   PRISM_BENCH=1 zig build test -Dtest-filter="perf bench: nvidia per-draw"
// It reports us/draw + draws/s for the immediate-mode glDrawArrays path (begin/record/submit/
// fence per draw). Kept as a regression guard for the per-draw CPU cost work (scratch vertex
// buffer + pooled command buffer removed the per-draw alloc/free churn: ~57 -> ~33 us/draw).
test "perf bench: nvidia per-draw cost" {
    if (std.c.getenv("PRISM_BENCH") == null) return error.SkipZigTest;
    const saved = state.pinned_driver;
    state.pinned_driver = "nvidia";
    defer state.pinned_driver = saved;
    _ = eglBindAPI(egl.EGL_OPENGL_ES_API);
    const dpy = eglGetPlatformDisplay(egl.EGL_PLATFORM_SURFACELESS_MESA, @ptrFromInt(0xBEEF01), null);
    if (eglInitialize(dpy, null, null) != egl.EGL_TRUE) return error.SkipZigTest;
    defer _ = eglTerminate(dpy);
    const cfg = state.configHandle(0);
    const ctx = eglCreateContext(dpy, cfg, egl.EGL_NO_CONTEXT, null);
    const pb = [_]EGLint{ egl.EGL_WIDTH, 64, egl.EGL_HEIGHT, 64, egl.EGL_NONE };
    const surf = eglCreatePbufferSurface(dpy, cfg, &pb);
    _ = eglMakeCurrent(dpy, surf, surf, ctx);
    defer _ = eglDestroyContext(dpy, ctx);
    defer _ = eglDestroySurface(dpy, surf);
    defer _ = eglMakeCurrent(dpy, null, null, egl.EGL_NO_CONTEXT);
    const vs: [*:0]const gles.GLchar = "attribute vec2 p; void main(){ gl_Position=vec4(p,0.0,1.0); }";
    const fs: [*:0]const gles.GLchar = "precision mediump float; void main(){ gl_FragColor=vec4(1.0,0.5,0.25,1.0); }";
    const v = glCreateShader(gles.GL_VERTEX_SHADER);
    glShaderSource(v, 1, &[_]?[*:0]const gles.GLchar{vs}, null);
    glCompileShader(v);
    const f = glCreateShader(gles.GL_FRAGMENT_SHADER);
    glShaderSource(f, 1, &[_]?[*:0]const gles.GLchar{fs}, null);
    glCompileShader(f);
    const prog = glCreateProgram();
    glAttachShader(prog, v);
    glAttachShader(prog, f);
    glBindAttribLocation(prog, 0, "p");
    glLinkProgram(prog);
    glUseProgram(prog);
    const tri = [_]f32{ -0.9, -0.9, 0.9, -0.9, 0.0, 0.9 };
    var vbo: gles.GLuint = 0;
    glGenBuffers(1, @ptrCast(&vbo));
    glBindBuffer(gles.GL_ARRAY_BUFFER, vbo);
    glBufferData(gles.GL_ARRAY_BUFFER, @sizeOf(@TypeOf(tri)), &tri, gles.GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, gles.GL_FLOAT, gles.GL_FALSE, 0, @ptrFromInt(0));
    glEnableVertexAttribArray(0);
    glViewport(0, 0, 64, 64);
    var i: usize = 0;
    while (i < 500) : (i += 1) { // warm up (flush every 50 so the vertex pool reaches steady size)
        glDrawArrays(gles.GL_TRIANGLES, 0, 3);
        if ((i + 1) % 50 == 0) glFinish();
    }
    const nowNs = struct {
        fn mono() i128 {
            var ts: std.os.linux.timespec = undefined;
            _ = std.os.linux.clock_gettime(std.os.linux.CLOCK.MONOTONIC, &ts);
            return @as(i128, ts.sec) * 1_000_000_000 + ts.nsec;
        }
    }.mono;
    const N = 500;
    const FRAME = 50; // simulate 50-draw frames: glFinish flushes at each boundary (a swap would too)
    const t0 = nowNs();
    i = 0;
    while (i < N) : (i += 1) {
        glDrawArrays(gles.GL_TRIANGLES, 0, 3);
        if ((i + 1) % FRAME == 0) glFinish(); // frame boundary: batched -> one submit per frame
    }
    const ns: f64 = @floatFromInt(nowNs() - t0);
    std.debug.print("\n[PERF] {} draws ({}-draw frames) in {d:.2} ms = {d:.1} us/draw = {d:.0} draws/s\n", .{ N, FRAME, ns / 1e6, ns / @as(f64, N) / 1e3, @as(f64, N) * 1e9 / ns });
    glDeleteProgram(prog);
}

/// Two consecutive draws sharing one default-uniform-block UBO, each setting a different value
/// via glUniform* before its draw. Draw 1 paints the left half red, draw 2 the right half blue.
/// Each draw must sample the uniform value as it was when that draw was issued. Driver-agnostic.
/// Caller pins the device via state.pinned_driver. Skips if it cannot come up.
fn runTwoObjectUniform(dpy_magic: usize) !void {
    _ = eglBindAPI(egl.EGL_OPENGL_ES_API);
    const dpy = eglGetPlatformDisplay(egl.EGL_PLATFORM_SURFACELESS_MESA, @ptrFromInt(dpy_magic), null);
    if (eglInitialize(dpy, null, null) != egl.EGL_TRUE) return error.SkipZigTest;
    defer _ = eglTerminate(dpy);
    const cfg = state.configHandle(0); // RGBA8, no depth
    const ctx = eglCreateContext(dpy, cfg, egl.EGL_NO_CONTEXT, null);
    const pb_attribs = [_]EGLint{ egl.EGL_WIDTH, 64, egl.EGL_HEIGHT, 64, egl.EGL_NONE };
    const surf = eglCreatePbufferSurface(dpy, cfg, &pb_attribs);
    try std.testing.expectEqual(egl.EGL_TRUE, eglMakeCurrent(dpy, surf, surf, ctx));
    defer _ = eglDestroyContext(dpy, ctx);
    defer _ = eglDestroySurface(dpy, surf);
    defer _ = eglMakeCurrent(dpy, null, null, egl.EGL_NO_CONTEXT);

    const vs_src: [*:0]const gles.GLchar =
        \\attribute vec2 position;
        \\void main() { gl_Position = vec4(position, 0.0, 1.0); }
    ;
    const fs_src: [*:0]const gles.GLchar =
        \\precision highp float;
        \\uniform vec3 uColor;
        \\void main() { gl_FragColor = vec4(uColor, 1.0); }
    ;
    const vs = glCreateShader(gles.GL_VERTEX_SHADER);
    glShaderSource(vs, 1, &[_]?[*:0]const gles.GLchar{vs_src}, null);
    glCompileShader(vs);
    const fs = glCreateShader(gles.GL_FRAGMENT_SHADER);
    glShaderSource(fs, 1, &[_]?[*:0]const gles.GLchar{fs_src}, null);
    glCompileShader(fs);
    const prog = glCreateProgram();
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glBindAttribLocation(prog, 0, "position");
    glLinkProgram(prog);
    var link: gles.GLint = 0;
    glGetProgramiv(prog, gles.GL_LINK_STATUS, &link);
    try std.testing.expectEqual(@as(gles.GLint, gles.GL_TRUE), link);
    defer {
        glDeleteProgram(prog);
        glDeleteShader(vs);
        glDeleteShader(fs);
    }

    // Left half quad (x in [-1,0]) and right half quad (x in [0,1]); both span the full height.
    const left = [_]f32{ -1, -1, 0, -1, 0, 1, -1, -1, 0, 1, -1, 1 };
    const right = [_]f32{ 0, -1, 1, -1, 1, 1, 0, -1, 1, 1, 0, 1 };
    var lvbo: gles.GLuint = 0;
    var rvbo: gles.GLuint = 0;
    glGenBuffers(1, @ptrCast(&lvbo));
    glGenBuffers(1, @ptrCast(&rvbo));
    glBindBuffer(gles.GL_ARRAY_BUFFER, lvbo);
    glBufferData(gles.GL_ARRAY_BUFFER, @sizeOf(@TypeOf(left)), &left, gles.GL_STATIC_DRAW);
    glBindBuffer(gles.GL_ARRAY_BUFFER, rvbo);
    glBufferData(gles.GL_ARRAY_BUFFER, @sizeOf(@TypeOf(right)), &right, gles.GL_STATIC_DRAW);
    defer glDeleteBuffers(2, &[_]gles.GLuint{ lvbo, rvbo });

    glViewport(0, 0, 64, 64);
    glUseProgram(prog);
    glEnableVertexAttribArray(0);
    const uColor = glGetUniformLocation(prog, "uColor");

    glClearColor(0, 0, 0, 1);
    glClear(gles.GL_COLOR_BUFFER_BIT);
    // Draw 1: left half, red.
    glUniform3f(uColor, 1.0, 0.0, 0.0);
    glBindBuffer(gles.GL_ARRAY_BUFFER, lvbo);
    glVertexAttribPointer(0, 2, gles.GL_FLOAT, gles.GL_FALSE, 0, @ptrFromInt(0));
    glDrawArrays(gles.GL_TRIANGLES, 0, 6);
    // Draw 2: right half, blue. This second glUniform* must not retroactively recolor draw 1.
    glUniform3f(uColor, 0.0, 0.0, 1.0);
    glBindBuffer(gles.GL_ARRAY_BUFFER, rvbo);
    glVertexAttribPointer(0, 2, gles.GL_FLOAT, gles.GL_FALSE, 0, @ptrFromInt(0));
    glDrawArrays(gles.GL_TRIANGLES, 0, 6);
    try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());

    const s: *state.Surface = @ptrCast(@alignCast(surf.?));
    const dev = s.display.device.?;
    const px = try flushMap(dev, s);
    const at = struct {
        fn p(buf: []const u8, x: usize, y: usize) [3]u8 {
            const i = (y * 64 + x) * 4;
            return .{ buf[i], buf[i + 1], buf[i + 2] };
        }
    }.p;
    const l = at(px, 16, 32); // left half: draw 1 (red)
    const r = at(px, 48, 32); // right half: draw 2 (blue)
    // Differential: left is red (draw 1's uniform), right is blue (draw 2's uniform). The bug
    // makes the left half blue too (both draws read the last-written uColor).
    try std.testing.expect(l[0] > 200 and l[2] < 60); // left red, not blue
    try std.testing.expect(r[2] > 200 and r[0] < 60); // right blue
}

test "per-draw default-block uniforms: two draws sharing one UBO keep their own values (software)" {
    // SOFTWARE (deterministic under is_test): the shared-UBO snapshot semantics baseline.
    try runTwoObjectUniform(0x2B300);
}

test "per-draw default-block uniforms on NVIDIA GPU: two draws sharing one UBO keep their own values (skips without a GPU)" {
    // The GPU differential that proves per-draw uniform snapshotting: forced onto the real nvidia
    // device. Without the fix both halves come out blue (the last-written uColor).
    const saved = state.pinned_driver;
    state.pinned_driver = "nvidia";
    defer state.pinned_driver = saved;
    try runTwoObjectUniform(0x2B301);
}

/// Cross-submit depth occlusion: draw a near quad (red) first, then a far quad (blue) that overlaps
/// it. Each glDrawArrays is its own nvidia submit, so the depth buffer written by draw 1 must persist
/// into draw 2's submit for GL_LESS to reject the farther fragment. If the driver re-cleared depth on
/// every submit (the historically-flagged "submit re-clears depth every bind" concern), the far blue
/// would pass against a fresh far plane and overwrite the near red at the overlap. This is the
/// discriminating order: a far-then-near draw (the shadow scene) passes either way because the nearer
/// fragment always wins. Here near-first means only preserved depth keeps the overlap red.
fn runDepthOcclusion(dpy_magic: usize) !void {
    _ = eglBindAPI(egl.EGL_OPENGL_ES_API);
    const dpy = eglGetPlatformDisplay(egl.EGL_PLATFORM_SURFACELESS_MESA, @ptrFromInt(dpy_magic), null);
    if (eglInitialize(dpy, null, null) != egl.EGL_TRUE) return error.SkipZigTest;
    defer _ = eglTerminate(dpy);
    const cfg = state.configHandle(1); // RGBA8 + D24
    const ctx = eglCreateContext(dpy, cfg, egl.EGL_NO_CONTEXT, null);
    const pb_attribs = [_]EGLint{ egl.EGL_WIDTH, 64, egl.EGL_HEIGHT, 64, egl.EGL_NONE };
    const surf = eglCreatePbufferSurface(dpy, cfg, &pb_attribs);
    try std.testing.expectEqual(egl.EGL_TRUE, eglMakeCurrent(dpy, surf, surf, ctx));
    defer _ = eglDestroyContext(dpy, ctx);
    defer _ = eglDestroySurface(dpy, surf);
    defer _ = eglMakeCurrent(dpy, null, null, egl.EGL_NO_CONTEXT);

    // z carried per-vertex so each quad sits at a fixed NDC depth (Vulkan clip: z in [0,1]).
    const vs_src: [*:0]const gles.GLchar =
        \\attribute vec3 position;
        \\void main() { gl_Position = vec4(position, 1.0); }
    ;
    const fs_src: [*:0]const gles.GLchar =
        \\precision highp float;
        \\uniform vec3 uColor;
        \\void main() { gl_FragColor = vec4(uColor, 1.0); }
    ;
    const vs = glCreateShader(gles.GL_VERTEX_SHADER);
    glShaderSource(vs, 1, &[_]?[*:0]const gles.GLchar{vs_src}, null);
    glCompileShader(vs);
    const fs = glCreateShader(gles.GL_FRAGMENT_SHADER);
    glShaderSource(fs, 1, &[_]?[*:0]const gles.GLchar{fs_src}, null);
    glCompileShader(fs);
    const prog = glCreateProgram();
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glBindAttribLocation(prog, 0, "position");
    glLinkProgram(prog);
    var link: gles.GLint = 0;
    glGetProgramiv(prog, gles.GL_LINK_STATUS, &link);
    try std.testing.expectEqual(@as(gles.GLint, gles.GL_TRUE), link);
    defer {
        glDeleteProgram(prog);
        glDeleteShader(vs);
        glDeleteShader(fs);
    }

    // Near quad (z=0.2): x in [-1, 0.5]. Far quad (z=0.8): x in [-0.5, 1]. They overlap x in
    // [-0.5, 0.5] = screen pixels [16, 48], centred at pixel 32.
    const near = [_]f32{ -1, -1, 0.2, 0.5, -1, 0.2, 0.5, 1, 0.2, -1, -1, 0.2, 0.5, 1, 0.2, -1, 1, 0.2 };
    const far = [_]f32{ -0.5, -1, 0.8, 1, -1, 0.8, 1, 1, 0.8, -0.5, -1, 0.8, 1, 1, 0.8, -0.5, 1, 0.8 };
    var nvbo: gles.GLuint = 0;
    var fvbo: gles.GLuint = 0;
    glGenBuffers(1, @ptrCast(&nvbo));
    glGenBuffers(1, @ptrCast(&fvbo));
    glBindBuffer(gles.GL_ARRAY_BUFFER, nvbo);
    glBufferData(gles.GL_ARRAY_BUFFER, @sizeOf(@TypeOf(near)), &near, gles.GL_STATIC_DRAW);
    glBindBuffer(gles.GL_ARRAY_BUFFER, fvbo);
    glBufferData(gles.GL_ARRAY_BUFFER, @sizeOf(@TypeOf(far)), &far, gles.GL_STATIC_DRAW);
    defer glDeleteBuffers(2, &[_]gles.GLuint{ nvbo, fvbo });

    glViewport(0, 0, 64, 64);
    glUseProgram(prog);
    glEnableVertexAttribArray(0);
    glEnable(gles.GL_DEPTH_TEST);
    glDepthFunc(gles.GL_LESS);
    const uColor = glGetUniformLocation(prog, "uColor");

    glClearColor(0, 0, 0, 1);
    glClearDepthf(1.0);
    glClear(gles.GL_COLOR_BUFFER_BIT | gles.GL_DEPTH_BUFFER_BIT);
    // Draw 1 (submit 1): near red. Writes depth 0.2 across x in [-1, 0.5].
    glUniform3f(uColor, 1.0, 0.0, 0.0);
    glBindBuffer(gles.GL_ARRAY_BUFFER, nvbo);
    glVertexAttribPointer(0, 3, gles.GL_FLOAT, gles.GL_FALSE, 0, @ptrFromInt(0));
    glDrawArrays(gles.GL_TRIANGLES, 0, 6);
    // Draw 2 (submit 2): far blue. At the overlap it must lose GL_LESS against the retained 0.2.
    glUniform3f(uColor, 0.0, 0.0, 1.0);
    glBindBuffer(gles.GL_ARRAY_BUFFER, fvbo);
    glVertexAttribPointer(0, 3, gles.GL_FLOAT, gles.GL_FALSE, 0, @ptrFromInt(0));
    glDrawArrays(gles.GL_TRIANGLES, 0, 6);
    glDisable(gles.GL_DEPTH_TEST);
    try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());

    const s: *state.Surface = @ptrCast(@alignCast(surf.?));
    const dev = s.display.device.?;
    const px = try flushMap(dev, s);
    const at = struct {
        fn p(buf: []const u8, x: usize, y: usize) [3]u8 {
            const i = (y * 64 + x) * 4;
            return .{ buf[i], buf[i + 1], buf[i + 2] };
        }
    }.p;
    const near_only = at(px, 8, 32); // x in [-1,-0.5]: near red only
    const overlap = at(px, 32, 32); // x in [-0.5,0.5]: near red drawn first occludes far blue
    const far_only = at(px, 56, 32); // x in [0.5,1]: far blue only
    // Sanity: the two non-overlapping halves render their own colour regardless of depth.
    try std.testing.expect(near_only[0] > 200 and near_only[2] < 60); // near-only red
    try std.testing.expect(far_only[2] > 200 and far_only[0] < 60); // far-only blue
    // Discriminator: the overlap is red because draw 1's depth persisted into draw 2's submit and
    // rejected the farther blue. A per-submit depth re-clear would make this blue.
    try std.testing.expect(overlap[0] > 200 and overlap[2] < 60); // overlap red, not blue
}

test "cross-submit depth occlusion: near-first quad keeps the overlap when a far quad is drawn after (software)" {
    try runDepthOcclusion(0x2B310);
}

test "cross-submit depth occlusion on NVIDIA GPU: near-first draw's depth persists across submits so the later far quad is rejected (skips without a GPU)" {
    // The GPU discriminator for the historically-flagged "submit re-clears depth every bind" concern:
    // forced onto the real nvidia device. If depth were re-cleared per submit the overlap comes out
    // blue (the far quad wrongly wins against a fresh far plane).
    const saved = state.pinned_driver;
    state.pinned_driver = "nvidia";
    defer state.pinned_driver = saved;
    try runDepthOcclusion(0x2B311);
}

/// Combined depth and stencil in the same draw (a 3D UI panel clipped by a stencil mask): the
/// depth test and stencil test are both enabled simultaneously, so on nvidia the ZF32 depth ZETA
/// and a separate S8 stencil plane must be bound together (bindDepthStencilSeparate). Stencil clips
/// where; depth resolves occlusion within the clip. A driver that bound only one plane (or clobbered
/// the other) fails one of the two assertions.
fn runDepthStencilCombined(dpy_magic: usize) !void {
    _ = eglBindAPI(egl.EGL_OPENGL_ES_API);
    const dpy = eglGetPlatformDisplay(egl.EGL_PLATFORM_SURFACELESS_MESA, @ptrFromInt(dpy_magic), null);
    if (eglInitialize(dpy, null, null) != egl.EGL_TRUE) return error.SkipZigTest;
    defer _ = eglTerminate(dpy);
    const cfg = state.configHandle(2); // RGBA8 + D24 + S8
    const ctx = eglCreateContext(dpy, cfg, egl.EGL_NO_CONTEXT, null);
    const pb_attribs = [_]EGLint{ egl.EGL_WIDTH, 64, egl.EGL_HEIGHT, 64, egl.EGL_NONE };
    const surf = eglCreatePbufferSurface(dpy, cfg, &pb_attribs);
    try std.testing.expectEqual(egl.EGL_TRUE, eglMakeCurrent(dpy, surf, surf, ctx));
    defer _ = eglDestroyContext(dpy, ctx);
    defer _ = eglDestroySurface(dpy, surf);
    defer _ = eglMakeCurrent(dpy, null, null, egl.EGL_NO_CONTEXT);

    const vs_src: [*:0]const gles.GLchar =
        \\attribute vec3 position;
        \\void main() { gl_Position = vec4(position, 1.0); }
    ;
    const fs_src: [*:0]const gles.GLchar =
        \\precision highp float;
        \\uniform vec3 uColor;
        \\void main() { gl_FragColor = vec4(uColor, 1.0); }
    ;
    const vs = glCreateShader(gles.GL_VERTEX_SHADER);
    glShaderSource(vs, 1, &[_]?[*:0]const gles.GLchar{vs_src}, null);
    glCompileShader(vs);
    const fs = glCreateShader(gles.GL_FRAGMENT_SHADER);
    glShaderSource(fs, 1, &[_]?[*:0]const gles.GLchar{fs_src}, null);
    glCompileShader(fs);
    const prog = glCreateProgram();
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glBindAttribLocation(prog, 0, "position");
    glLinkProgram(prog);
    var link: gles.GLint = 0;
    glGetProgramiv(prog, gles.GL_LINK_STATUS, &link);
    try std.testing.expectEqual(@as(gles.GLint, gles.GL_TRUE), link);
    defer {
        glDeleteProgram(prog);
        glDeleteShader(vs);
        glDeleteShader(fs);
    }

    // Left-half quad (x in [-1,0]) and full-screen quad, each at a fixed NDC z.
    const left_z5 = [_]f32{ -1, -1, 0.5, 0, -1, 0.5, 0, 1, 0.5, -1, -1, 0.5, 0, 1, 0.5, -1, 1, 0.5 };
    const full_z3 = [_]f32{ -1, -1, 0.3, 1, -1, 0.3, 1, 1, 0.3, -1, -1, 0.3, 1, 1, 0.3, -1, 1, 0.3 };
    const full_z7 = [_]f32{ -1, -1, 0.7, 1, -1, 0.7, 1, 1, 0.7, -1, -1, 0.7, 1, 1, 0.7, -1, 1, 0.7 };
    var vbo: [3]gles.GLuint = .{ 0, 0, 0 };
    glGenBuffers(3, &vbo);
    inline for (.{ left_z5, full_z3, full_z7 }, 0..) |data, i| {
        glBindBuffer(gles.GL_ARRAY_BUFFER, vbo[i]);
        glBufferData(gles.GL_ARRAY_BUFFER, @sizeOf(@TypeOf(data)), &data, gles.GL_STATIC_DRAW);
    }
    defer glDeleteBuffers(3, &vbo);
    const bindQuad = struct {
        fn f(b: gles.GLuint) void {
            glBindBuffer(gles.GL_ARRAY_BUFFER, b);
            glVertexAttribPointer(0, 3, gles.GL_FLOAT, gles.GL_FALSE, 0, @ptrFromInt(0));
        }
    }.f;

    glViewport(0, 0, 64, 64);
    glUseProgram(prog);
    glEnableVertexAttribArray(0);
    glEnable(gles.GL_DEPTH_TEST);
    glDepthFunc(gles.GL_LESS);
    glEnable(gles.GL_STENCIL_TEST);
    const uColor = glGetUniformLocation(prog, "uColor");

    glClearColor(0, 0, 0, 1);
    glClearDepthf(1.0);
    glClearStencil(0);
    glClear(gles.GL_COLOR_BUFFER_BIT | gles.GL_DEPTH_BUFFER_BIT | gles.GL_STENCIL_BUFFER_BIT);

    // Plant: stencil=1 across the left half (ALWAYS/REPLACE). Colour black, depth 0.5 (< the 1.0 far
    // clear, so it writes). Right half is left untouched (stencil 0, depth 1.0, colour black).
    glStencilFunc(gles.GL_ALWAYS, 1, 0xFF);
    glStencilOp(gles.GL_KEEP, gles.GL_KEEP, gles.GL_REPLACE);
    glUniform3f(uColor, 0.0, 0.0, 0.0);
    bindQuad(vbo[0]);
    glDrawArrays(gles.GL_TRIANGLES, 0, 6);

    // Main: full-screen red at z=0.3. Stencil EQUAL 1 (passes only the left half), depth LESS (0.3 <
    // planted 0.5, passes). KEEP so it does not disturb the mask. Red in the left half only.
    glStencilFunc(gles.GL_EQUAL, 1, 0xFF);
    glStencilOp(gles.GL_KEEP, gles.GL_KEEP, gles.GL_KEEP);
    glUniform3f(uColor, 1.0, 0.0, 0.0);
    bindQuad(vbo[1]);
    glDrawArrays(gles.GL_TRIANGLES, 0, 6);

    // Depth proof: full-screen blue at z=0.7, same stencil EQUAL 1. Where stencil passes (left half)
    // the depth test must reject it (0.7 < 0.3 is false) so the left half stays red. If the combined
    // path dropped the depth test, the left half would turn blue.
    glUniform3f(uColor, 0.0, 0.0, 1.0);
    bindQuad(vbo[2]);
    glDrawArrays(gles.GL_TRIANGLES, 0, 6);
    glDisable(gles.GL_DEPTH_TEST);
    glDisable(gles.GL_STENCIL_TEST);
    try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());

    const s: *state.Surface = @ptrCast(@alignCast(surf.?));
    const dev = s.display.device.?;
    const px = try flushMap(dev, s);
    const at = struct {
        fn p(buf: []const u8, x: usize, y: usize) [3]u8 {
            const i = (y * 64 + x) * 4;
            return .{ buf[i], buf[i + 1], buf[i + 2] };
        }
    }.p;
    const left = at(px, 16, 32); // stencil==1: RED (depth kept it over the later blue)
    const right = at(px, 48, 32); // stencil==0: clipped, stays black
    // Stencil clip: the right half is masked out (black), not painted red by the full-screen draw.
    try std.testing.expect(right[0] < 60 and right[1] < 60 and right[2] < 60); // right black
    // Depth and stencil both active: the left half is red (stencil let it in, depth rejected the blue).
    try std.testing.expect(left[0] > 200 and left[2] < 60); // left red, not blue, not black
}

test "combined depth+stencil: a stencil mask and depth test both act in one draw (software)" {
    try runDepthStencilCombined(0x2B320);
}

test "combined depth+stencil on NVIDIA GPU: ZF32 depth ZETA + separate S8 plane bound together, both tests act (skips without a GPU)" {
    // Forces the real nvidia device: exercises bindDepthStencilSeparate (a ZF32 depth ZETA and a
    // separate S8 stencil plane bound together). Stencil must clip the right half AND depth must
    // reject the later far blue inside the clip. Both planes live at once.
    const saved = state.pinned_driver;
    state.pinned_driver = "nvidia";
    defer state.pinned_driver = saved;
    try runDepthStencilCombined(0x2B321);
}

/// FBO stencil: stencil-clip a draw while rendering into an off-screen framebuffer (a color RB
/// and a STENCIL_INDEX8 RB), the compositor "clip a layer, then composite it" case. On nvidia
/// the stencil lives in an internal ZETA (the HAL stencil RB is a CPU u8 buffer), proving the
/// stencil test engages against an FBO color surface (block-linear render target), not just
/// the default window.
fn runFboStencil(dpy_magic: usize) !void {
    _ = eglBindAPI(egl.EGL_OPENGL_ES_API);
    const dpy = eglGetPlatformDisplay(egl.EGL_PLATFORM_SURFACELESS_MESA, @ptrFromInt(dpy_magic), null);
    if (eglInitialize(dpy, null, null) != egl.EGL_TRUE) return error.SkipZigTest;
    defer _ = eglTerminate(dpy);
    const cfg = state.configHandle(0); // RGBA8 default surface; the FBO carries its own stencil RB
    const ctx = eglCreateContext(dpy, cfg, egl.EGL_NO_CONTEXT, null);
    const W: EGLint = 64;
    const pb_attribs = [_]EGLint{ egl.EGL_WIDTH, W, egl.EGL_HEIGHT, W, egl.EGL_NONE };
    const surf = eglCreatePbufferSurface(dpy, cfg, &pb_attribs);
    try std.testing.expectEqual(egl.EGL_TRUE, eglMakeCurrent(dpy, surf, surf, ctx));
    defer _ = eglDestroyContext(dpy, ctx);
    defer _ = eglDestroySurface(dpy, surf);
    defer _ = eglMakeCurrent(dpy, null, null, egl.EGL_NO_CONTEXT);

    const vs_src: [*:0]const gles.GLchar =
        \\attribute vec2 position;
        \\void main() { gl_Position = vec4(position, 0.0, 1.0); }
    ;
    const fs_src: [*:0]const gles.GLchar =
        \\precision highp float;
        \\uniform vec3 uColor;
        \\void main() { gl_FragColor = vec4(uColor, 1.0); }
    ;
    const vs = glCreateShader(gles.GL_VERTEX_SHADER);
    glShaderSource(vs, 1, &[_]?[*:0]const gles.GLchar{vs_src}, null);
    glCompileShader(vs);
    const fs = glCreateShader(gles.GL_FRAGMENT_SHADER);
    glShaderSource(fs, 1, &[_]?[*:0]const gles.GLchar{fs_src}, null);
    glCompileShader(fs);
    const prog = glCreateProgram();
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glBindAttribLocation(prog, 0, "position");
    glLinkProgram(prog);
    var link: gles.GLint = 0;
    glGetProgramiv(prog, gles.GL_LINK_STATUS, &link);
    try std.testing.expectEqual(@as(gles.GLint, gles.GL_TRUE), link);
    defer {
        glDeleteProgram(prog);
        glDeleteShader(vs);
        glDeleteShader(fs);
    }

    // Off-screen FBO: a colour renderbuffer + a stencil renderbuffer.
    var fbo: gles.GLuint = 0;
    glGenFramebuffers(1, @ptrCast(&fbo));
    glBindFramebuffer(gles.GL_FRAMEBUFFER, fbo);
    var color_rb: gles.GLuint = 0;
    glGenRenderbuffers(1, @ptrCast(&color_rb));
    glBindRenderbuffer(gles.GL_RENDERBUFFER, color_rb);
    glRenderbufferStorage(gles.GL_RENDERBUFFER, gles.GL_RGBA8, W, W);
    glFramebufferRenderbuffer(gles.GL_FRAMEBUFFER, gles.GL_COLOR_ATTACHMENT0, gles.GL_RENDERBUFFER, color_rb);
    var stencil_rb: gles.GLuint = 0;
    glGenRenderbuffers(1, @ptrCast(&stencil_rb));
    glBindRenderbuffer(gles.GL_RENDERBUFFER, stencil_rb);
    glRenderbufferStorage(gles.GL_RENDERBUFFER, gles.GL_STENCIL_INDEX8, W, W);
    glFramebufferRenderbuffer(gles.GL_FRAMEBUFFER, gles.GL_STENCIL_ATTACHMENT, gles.GL_RENDERBUFFER, stencil_rb);
    defer {
        glDeleteFramebuffers(1, &[_]gles.GLuint{fbo});
        glDeleteRenderbuffers(1, &[_]gles.GLuint{ color_rb, stencil_rb });
    }

    const left = [_]f32{ -1, -1, 0, -1, 0, 1, -1, -1, 0, 1, -1, 1 }; // x in [-1,0]
    const full = [_]f32{ -1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1 }; // x in [-1,1]
    var vbo: [2]gles.GLuint = .{ 0, 0 };
    glGenBuffers(2, &vbo);
    inline for (.{ left, full }, 0..) |data, i| {
        glBindBuffer(gles.GL_ARRAY_BUFFER, vbo[i]);
        glBufferData(gles.GL_ARRAY_BUFFER, @sizeOf(@TypeOf(data)), &data, gles.GL_STATIC_DRAW);
    }
    defer glDeleteBuffers(2, &vbo);
    const bindQuad = struct {
        fn f(b: gles.GLuint) void {
            glBindBuffer(gles.GL_ARRAY_BUFFER, b);
            glVertexAttribPointer(0, 2, gles.GL_FLOAT, gles.GL_FALSE, 0, @ptrFromInt(0));
        }
    }.f;

    glViewport(0, 0, W, W);
    glUseProgram(prog);
    glEnableVertexAttribArray(0);
    glEnable(gles.GL_STENCIL_TEST);
    const uColor = glGetUniformLocation(prog, "uColor");

    glClearColor(0, 0, 0, 1);
    glClearStencil(0);
    glClear(gles.GL_COLOR_BUFFER_BIT | gles.GL_STENCIL_BUFFER_BIT);

    // Plant: stencil=1 across the left half (ALWAYS/REPLACE), colour black.
    glStencilFunc(gles.GL_ALWAYS, 1, 0xFF);
    glStencilOp(gles.GL_KEEP, gles.GL_KEEP, gles.GL_REPLACE);
    glUniform3f(uColor, 0.0, 0.0, 0.0);
    bindQuad(vbo[0]);
    glDrawArrays(gles.GL_TRIANGLES, 0, 6);

    // Clip: full-screen red, stencil EQUAL 1 (KEEP) -> only the left half (stencil==1) is painted.
    glStencilFunc(gles.GL_EQUAL, 1, 0xFF);
    glStencilOp(gles.GL_KEEP, gles.GL_KEEP, gles.GL_KEEP);
    glUniform3f(uColor, 1.0, 0.0, 0.0);
    bindQuad(vbo[1]);
    glDrawArrays(gles.GL_TRIANGLES, 0, 6);
    glDisable(gles.GL_STENCIL_TEST);
    try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());

    // Read the FBO colour back with glReadPixels (the FBO stays bound as READ). Maps the FBO
    // colour renderbuffer directly through the de-swizzle READ path. Unlike a blit-into-the-default-
    // fb, whose CPU write side lands in a throwaway de-swizzle scratch on nvidia and never reaches
    // the GPU surface. glReadPixels returns rows bottom-up (GL origin bottom-left); the left/right
    // columns span full height so any row discriminates.
    var rgba: [64 * 64 * 4]u8 = .{0} ** (64 * 64 * 4);
    glReadPixels(0, 0, W, W, gles.GL_RGBA, gles.GL_UNSIGNED_BYTE, &rgba);
    try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());
    const at = struct {
        fn p(buf: []const u8, x: usize, y: usize) [3]u8 {
            const i = (y * 64 + x) * 4;
            return .{ buf[i], buf[i + 1], buf[i + 2] };
        }
    }.p;
    const lft = at(&rgba, 16, 32); // stencil==1: RED
    const rgt = at(&rgba, 48, 32); // stencil==0: clipped, black
    try std.testing.expect(lft[0] > 200 and lft[2] < 60); // left red
    try std.testing.expect(rgt[0] < 60 and rgt[1] < 60 and rgt[2] < 60); // right black
}

test "FBO stencil clip: a stencil mask clips a draw rendered into an off-screen framebuffer (software)" {
    try runFboStencil(0x2B330);
}

test "FBO stencil clip on NVIDIA GPU: stencil test engages against an off-screen FBO color surface (skips without a GPU)" {
    // Forces the real nvidia device: the stencil TEST must clip a draw into an FBO colour surface (the
    // compositor layer case). This once faulted Xid 31 (GPCCLIENT_GCC) intermittently: the pipeline is
    // rebuilt per draw and its shader-heap sysmem VA was freed + immediately recycled, so the GPU's
    // speculative shader prefetch (off the persistent SET_PIPELINE_PROGRAM_ADDRESS) read the recycled,
    // transiently-unmapped VA. Fixed by deferring the shader-heap free (Device.retired_heaps): a heap's
    // VA is unmapped only at the next submit, after a fence and after the shader is rebound away from
    // it. Right half stays black only if the stencil clip is honoured off-screen.
    const saved = state.pinned_driver;
    state.pinned_driver = "nvidia";
    defer state.pinned_driver = saved;
    try runFboStencil(0x2B331);
}

/// glBlitFramebuffer write-back: draw a red left half into an off-screen FBO, then blit the whole
/// FBO colour into the default framebuffer and read it back. The blit's destination write must reach
/// the default fb's GPU storage. On nvidia that is a block-linear surface whose mapResource hands
/// back a de-swizzled scratch, so the write is only visible if the driver re-swizzles it back
/// (hal flushMappedImage). Asserts left red, right black (placement and write-back).
fn runBlitWriteback(dpy_magic: usize) !void {
    _ = eglBindAPI(egl.EGL_OPENGL_ES_API);
    const dpy = eglGetPlatformDisplay(egl.EGL_PLATFORM_SURFACELESS_MESA, @ptrFromInt(dpy_magic), null);
    if (eglInitialize(dpy, null, null) != egl.EGL_TRUE) return error.SkipZigTest;
    defer _ = eglTerminate(dpy);
    const cfg = state.configHandle(0);
    const W: EGLint = 64;
    const ctx = eglCreateContext(dpy, cfg, egl.EGL_NO_CONTEXT, null);
    const pb_attribs = [_]EGLint{ egl.EGL_WIDTH, W, egl.EGL_HEIGHT, W, egl.EGL_NONE };
    const surf = eglCreatePbufferSurface(dpy, cfg, &pb_attribs);
    try std.testing.expectEqual(egl.EGL_TRUE, eglMakeCurrent(dpy, surf, surf, ctx));
    defer _ = eglDestroyContext(dpy, ctx);
    defer _ = eglDestroySurface(dpy, surf);
    defer _ = eglMakeCurrent(dpy, null, null, egl.EGL_NO_CONTEXT);

    const vs_src: [*:0]const gles.GLchar =
        \\attribute vec2 position;
        \\void main() { gl_Position = vec4(position, 0.0, 1.0); }
    ;
    const fs_src: [*:0]const gles.GLchar =
        \\precision highp float;
        \\uniform vec3 uColor;
        \\void main() { gl_FragColor = vec4(uColor, 1.0); }
    ;
    const vs = glCreateShader(gles.GL_VERTEX_SHADER);
    glShaderSource(vs, 1, &[_]?[*:0]const gles.GLchar{vs_src}, null);
    glCompileShader(vs);
    const fs = glCreateShader(gles.GL_FRAGMENT_SHADER);
    glShaderSource(fs, 1, &[_]?[*:0]const gles.GLchar{fs_src}, null);
    glCompileShader(fs);
    const prog = glCreateProgram();
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glBindAttribLocation(prog, 0, "position");
    glLinkProgram(prog);
    defer {
        glDeleteProgram(prog);
        glDeleteShader(vs);
        glDeleteShader(fs);
    }

    var fbo: gles.GLuint = 0;
    glGenFramebuffers(1, @ptrCast(&fbo));
    glBindFramebuffer(gles.GL_FRAMEBUFFER, fbo);
    var color_rb: gles.GLuint = 0;
    glGenRenderbuffers(1, @ptrCast(&color_rb));
    glBindRenderbuffer(gles.GL_RENDERBUFFER, color_rb);
    glRenderbufferStorage(gles.GL_RENDERBUFFER, gles.GL_RGBA8, W, W);
    glFramebufferRenderbuffer(gles.GL_FRAMEBUFFER, gles.GL_COLOR_ATTACHMENT0, gles.GL_RENDERBUFFER, color_rb);
    defer {
        glDeleteFramebuffers(1, &[_]gles.GLuint{fbo});
        glDeleteRenderbuffers(1, &[_]gles.GLuint{color_rb});
    }

    // Left half quad (x in [-1,0]) red, on a black-cleared FBO.
    const left = [_]f32{ -1, -1, 0, -1, 0, 1, -1, -1, 0, 1, -1, 1 };
    var vbo: gles.GLuint = 0;
    glGenBuffers(1, @ptrCast(&vbo));
    glBindBuffer(gles.GL_ARRAY_BUFFER, vbo);
    glBufferData(gles.GL_ARRAY_BUFFER, @sizeOf(@TypeOf(left)), &left, gles.GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, gles.GL_FLOAT, gles.GL_FALSE, 0, @ptrFromInt(0));
    glEnableVertexAttribArray(0);
    defer glDeleteBuffers(1, &[_]gles.GLuint{vbo});

    glViewport(0, 0, W, W);
    glUseProgram(prog);
    glClearColor(0, 0, 0, 1);
    glClear(gles.GL_COLOR_BUFFER_BIT);
    glUniform3f(glGetUniformLocation(prog, "uColor"), 1.0, 0.0, 0.0);
    glDrawArrays(gles.GL_TRIANGLES, 0, 6);
    try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());

    // Blit the whole FBO colour into the DEFAULT framebuffer, then read the default fb back.
    glBindFramebuffer(gles.GL_READ_FRAMEBUFFER, fbo);
    glBindFramebuffer(gles.GL_DRAW_FRAMEBUFFER, 0);
    glBlitFramebuffer(0, 0, W, W, 0, 0, W, W, gles.GL_COLOR_BUFFER_BIT, gles.GL_NEAREST);
    try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());

    const s: *state.Surface = @ptrCast(@alignCast(surf.?));
    const dev = s.display.device.?;
    const px = try flushMap(dev, s);
    const at = struct {
        fn p(buf: []const u8, x: usize, y: usize) [3]u8 {
            const i = (y * 64 + x) * 4;
            return .{ buf[i], buf[i + 1], buf[i + 2] };
        }
    }.p;
    const lft = at(px, 16, 32);
    const rgt = at(px, 48, 32);
    try std.testing.expect(lft[0] > 200 and lft[2] < 60); // blit landed: left red
    try std.testing.expect(rgt[0] < 60 and rgt[1] < 60 and rgt[2] < 60); // right black
}

test "glBlitFramebuffer write-back: an FBO color blit into the default framebuffer lands correctly (software)" {
    try runBlitWriteback(0x2B340);
}

test "glBlitFramebuffer write-back on NVIDIA GPU: the blit re-swizzles into the block-linear default fb (skips without a GPU)" {
    // Forces the real nvidia device: the default fb is a block-linear surface, so the blit's CPU
    // write into mapResource's de-swizzled scratch must be re-swizzled back (hal flushMappedImage) or
    // the readback is all black (the write is lost).
    const saved = state.pinned_driver;
    state.pinned_driver = "nvidia";
    defer state.pinned_driver = saved;
    try runBlitWriteback(0x2B341);
}

test "shadow-mapped scene on NVIDIA GPU: two-pass light-space shadow map casts a correctly-placed shadow on the real RTX (skips without a GPU)" {
    // Hardware culmination of the shadow work: the same two-pass light-space shadow-mapped scene,
    // forced onto the real nvidia device (glmark2/vkcube auto-select it; the oracles otherwise pin
    // "software" for determinism). The pass-1 depth render lands in a ZETA depth surface and the
    // sampler2DShadow lookup in pass 2 does a real hardware DEPTH_COMPARE against it: the GL depth-
    // texture-sample path on nvidia de-tiles the rendered ZETA into a ZF32 sampled texture (hal
    // Device.finalizeDepthTexture -> CE-detile -> ZF32), the format the HW compare engages on (an
    // rgba8 "depth-as-color" image, the software path, would make the compare a HW no-op). The FBO
    // depth clear is routed through a real render pass so the nvidia driver actually clears the ZETA
    // (a draw-less depth clear was a no-op). Asserts the same core shadow: shadowed ground dark, lit
    // ground bright, correctly placed. Uses per-draw base colours (grey ground + blue occluder, the
    // full assertions): the nvidia per-draw UBO snapshot ring now gives each draw its own copy of the
    // shared default-block UBO, so the two draws' distinct uBase values no longer alias (this once
    // needed unify_base to sidestep that bug). Skips cleanly if there is no nvidia GPU.
    const saved = state.pinned_driver;
    state.pinned_driver = "nvidia";
    defer state.pinned_driver = saved;
    try runShadowMappedScene(0x5AD01, false);
}

/// Two-pass light-space shadow-mapped scene, driver-agnostic (caller pins device via
/// state.pinned_driver). A ground plane, an occluder slab, a directional light. Pass 1 renders
/// scene depth from the light's POV into a GL_DEPTH_COMPONENT FBO texture. Pass 2 samples it
/// with a sampler2DShadow and light-space projection. Skips if the pinned device cannot come up.
/// `unify_base`: use one grey base colour for both ground and occluder (no per-draw uBase change),
/// dropping the blue-occluder assertions to let the nvidia GPU path sidestep a pre-existing
/// multi-draw uniform-aliasing bug while still asserting the core cast-shadow differential.
fn runShadowMappedScene(dpy_magic: usize, unify_base: bool) !void {
    // A real 3D scene (a ground plane + an occluder quad above it)
    // lit by a directional light, with a genuine cast shadow via two-pass shadow mapping.
    //
    // Scene (world space, +y up):
    //  - Ground: a large flat quad at y=0, x,z in [-5,5].
    //  - Occluder: a small flat quad at y=2, x,z in [-1,1] (a "floating slab" above the ground).
    //  - Directional light travelling in dir (-1,-1,0)/sqrt2 (i.e. FROM up-and-+x). The direction
    //    toward the light is (1,1,0)/sqrt2.
    //
    // Expected shadow (derived from ray geometry, independent of the matrices):
    //  A point P on the occluder casts its shadow where the ray P + t*(-1,-1,0) hits y=0: for the
    //  occluder plane y=2, t=2, so (x0,2,z0) -> ground (x0-2, 0, z0). The occluder footprint x0 in
    //  [-1,1] therefore shadows ground x in [-3,-1], z in [-1,1] (shadow centre (-2,0,0)).
    //  Cross-check via the light ray: ground (-2,0,0) walking toward the light (1,1,0)/sqrt2 by
    //  2*sqrt2 reaches (0,2,0), the occluder centre, so (-2,0,0) is occluded. Ground (3,0,0)
    //  walking the same way reaches (5,2,0), off the occluder (x in [-1,1]), so it is lit.
    //
    // Camera (pass 2): a top-down orthographic view looking down -y. World x,z map linearly to the
    // screen (pixel_x = 8*worldx + 32, pixel_y = 8*worldz + 32 on a 64x64 buffer) and world y drives
    // depth (higher y = nearer), so the occluder (y=2) occludes the ground beneath it. A top-down
    // camera makes the pixel prediction principled; the shadow offset (occluder footprint at screen
    // x in [24,40] but its shadow at screen x in [16,24]) is produced purely by the light-space
    // projection. A bug that made light-space == camera-space would put the shadow under the
    // occluder and fail the asserts.
    _ = eglBindAPI(egl.EGL_OPENGL_ES_API);
    const dpy = eglGetPlatformDisplay(egl.EGL_PLATFORM_SURFACELESS_MESA, @ptrFromInt(dpy_magic), null);
    // A forced hardware device that is not present fails to initialize -> skip (not a failure).
    if (eglInitialize(dpy, null, null) != egl.EGL_TRUE) return error.SkipZigTest;
    defer _ = eglTerminate(dpy);
    const cfg = state.configHandle(1); // RGBA8 + depth
    const ctx = eglCreateContext(dpy, cfg, egl.EGL_NO_CONTEXT, null);
    const W: EGLint = 64;
    const pb_attribs = [_]EGLint{ egl.EGL_WIDTH, W, egl.EGL_HEIGHT, W, egl.EGL_NONE };
    const surf = eglCreatePbufferSurface(dpy, cfg, &pb_attribs);
    try std.testing.expectEqual(egl.EGL_TRUE, eglMakeCurrent(dpy, surf, surf, ctx));
    defer _ = eglDestroyContext(dpy, ctx);
    defer _ = eglDestroySurface(dpy, surf);
    defer _ = eglMakeCurrent(dpy, null, null, egl.EGL_NO_CONTEXT);

    // --- Matrices (column-major, index = col*4 + row) ------------------------------------------
    const s2: f32 = std.math.sqrt2;
    const inv8s2: f32 = 1.0 / (8.0 * s2);
    const inv10s2: f32 = 1.0 / (10.0 * s2);
    // Light view-projection (orthographic, for the directional light).
    //  clipL.x = p.z / 8                         (lx = dot(p, right=(0,0,1)))
    //  clipL.y = (-p.x + p.y - 1) / (8*sqrt2)     (ly = dot(p-c, up_light), c=(0,1,0))
    //  clipL.z = (-p.x - p.y + 1) / (10*sqrt2) + 0.5   (depth along the light dir, remapped to [0,1])
    //  clipL.w = 1
    const light_vp = [16]f32{
        // col0 (x)
        0,     -inv8s2, -inv10s2,      0,
        // col1 (y)
        0,     inv8s2,  -inv10s2,      0,
        // col2 (z)
        0.125, 0,       0,             0,
        // col3 (const)
        0,     -inv8s2, inv10s2 + 0.5, 1,
    };
    // Camera view-projection (top-down orthographic).
    //  clip.x = x/4, clip.y = z/4, clip.z = 0.5 - y/6 (nearer = higher y), clip.w = 1.
    //  The z map keeps the ground (y=0 -> 0.5) and occluder (y=2 -> 0.167) both strictly inside
    //  (0,1). A ground depth of exactly 1.0 would fail GL_LESS against the cleared far plane and
    //  leave speckled holes.
    const cam_vp = [16]f32{
        0.25, 0, 0, 0, // col0 (x)
        0, 0, -1.0 / 6.0, 0, // col1 (y)
        0, 0.25, 0, 0, // col2 (z)
        0, 0, 0.5, 1, // col3 (const)
    };

    // --- Pass 1 program: render scene depth from the light's POV --------------------------------
    const depth_vs: [*:0]const gles.GLchar =
        \\attribute vec3 position;
        \\uniform mat4 uLightVP;
        \\void main() { gl_Position = uLightVP * vec4(position, 1.0); }
    ;
    const white_fs: [*:0]const gles.GLchar =
        \\precision mediump float;
        \\void main() { gl_FragColor = vec4(1.0); }
    ;
    const dvs = glCreateShader(gles.GL_VERTEX_SHADER);
    glShaderSource(dvs, 1, &[_]?[*:0]const gles.GLchar{depth_vs}, null);
    glCompileShader(dvs);
    const dfs = glCreateShader(gles.GL_FRAGMENT_SHADER);
    glShaderSource(dfs, 1, &[_]?[*:0]const gles.GLchar{white_fs}, null);
    glCompileShader(dfs);
    const dprog = glCreateProgram();
    glAttachShader(dprog, dvs);
    glAttachShader(dprog, dfs);
    glBindAttribLocation(dprog, 0, "position");
    glLinkProgram(dprog);
    var dlink: gles.GLint = 0;
    glGetProgramiv(dprog, gles.GL_LINK_STATUS, &dlink);
    try std.testing.expectEqual(@as(gles.GLint, gles.GL_TRUE), dlink);

    // --- Pass 2 program: camera render + shadow lookup -----------------------------------------
    // The VS computes the light-space clip position per vertex and forwards it. The FS projects
    // that to a shadow-map UV (xy: [-1,1]->[0,1]) and a reference depth (z is already in Prism's
    // [0,1] window range, so no 0.5*z+0.5 remap, only a small acne bias), then samples the
    // sampler2DShadow (LEQUAL) for a 0/1 lit factor and applies ambient + lit*diffuse Lambert.
    const scene_vs: [*:0]const gles.GLchar =
        \\attribute vec3 position;
        \\uniform mat4 uCamVP;
        \\uniform mat4 uLightVP;
        \\varying vec4 vLightClip;
        \\void main() {
        \\  vLightClip = uLightVP * vec4(position, 1.0);
        \\  gl_Position = uCamVP * vec4(position, 1.0);
        \\}
    ;
    const scene_fs: [*:0]const gles.GLchar =
        \\precision highp float;
        \\uniform sampler2DShadow uShadow;
        \\uniform vec3 uBase;
        \\uniform vec3 uNormal;
        \\uniform vec3 uLightToLight;
        \\uniform float uBias;
        \\varying vec4 vLightClip;
        \\void main() {
        \\  vec3 proj = vLightClip.xyz / vLightClip.w;
        \\  vec2 uv = proj.xy * 0.5 + 0.5;
        \\  float ref = proj.z - uBias;
        \\  float lit = texture(uShadow, vec3(uv, ref));
        \\  float diff = max(dot(normalize(uNormal), normalize(uLightToLight)), 0.0);
        \\  vec3 color = uBase * (0.2 + lit * diff);
        \\  gl_FragColor = vec4(color, 1.0);
        \\}
    ;
    const svs = glCreateShader(gles.GL_VERTEX_SHADER);
    glShaderSource(svs, 1, &[_]?[*:0]const gles.GLchar{scene_vs}, null);
    glCompileShader(svs);
    var svs_ok: gles.GLint = 0;
    glGetShaderiv(svs, gles.GL_COMPILE_STATUS, &svs_ok);
    try std.testing.expectEqual(@as(gles.GLint, gles.GL_TRUE), svs_ok);
    const sfs = glCreateShader(gles.GL_FRAGMENT_SHADER);
    glShaderSource(sfs, 1, &[_]?[*:0]const gles.GLchar{scene_fs}, null);
    glCompileShader(sfs);
    var sfs_ok: gles.GLint = 0;
    glGetShaderiv(sfs, gles.GL_COMPILE_STATUS, &sfs_ok);
    if (sfs_ok != gles.GL_TRUE) {
        var lb: [512]gles.GLchar = undefined;
        glGetShaderInfoLog(sfs, lb.len, null, &lb);
        std.debug.print("scene FS: {s}\n", .{std.mem.sliceTo(&lb, 0)});
    }
    try std.testing.expectEqual(@as(gles.GLint, gles.GL_TRUE), sfs_ok);
    const sprog = glCreateProgram();
    glAttachShader(sprog, svs);
    glAttachShader(sprog, sfs);
    glBindAttribLocation(sprog, 0, "position");
    glLinkProgram(sprog);
    var slink: gles.GLint = 0;
    glGetProgramiv(sprog, gles.GL_LINK_STATUS, &slink);
    try std.testing.expectEqual(@as(gles.GLint, gles.GL_TRUE), slink);

    // --- Geometry: ground quad (y=0) + occluder quad (y=2), two triangles each ------------------
    const ground = [_]f32{
        -5, 0, -5, 5, 0, -5, 5,  0, 5,
        -5, 0, -5, 5, 0, 5,  -5, 0, 5,
    };
    const occluder = [_]f32{
        -1, 2, -1, 1, 2, -1, 1,  2, 1,
        -1, 2, -1, 1, 2, 1,  -1, 2, 1,
    };
    var gvbo: gles.GLuint = 0;
    var ovbo: gles.GLuint = 0;
    glGenBuffers(1, @ptrCast(&gvbo));
    glGenBuffers(1, @ptrCast(&ovbo));
    glBindBuffer(gles.GL_ARRAY_BUFFER, gvbo);
    glBufferData(gles.GL_ARRAY_BUFFER, @sizeOf(@TypeOf(ground)), &ground, gles.GL_STATIC_DRAW);
    glBindBuffer(gles.GL_ARRAY_BUFFER, ovbo);
    glBufferData(gles.GL_ARRAY_BUFFER, @sizeOf(@TypeOf(occluder)), &occluder, gles.GL_STATIC_DRAW);
    defer glDeleteBuffers(2, &[_]gles.GLuint{ gvbo, ovbo });

    // --- Shadow-map FBO: a GL_DEPTH_COMPONENT depth texture + a throwaway color texture ---------
    const SM: gles.GLsizei = 128;
    var dtex: gles.GLuint = 0;
    glGenTextures(1, @ptrCast(&dtex));
    glActiveTexture(gles.GL_TEXTURE0);
    glBindTexture(gles.GL_TEXTURE_2D, dtex);
    glTexImage2D(gles.GL_TEXTURE_2D, 0, @bitCast(gles.GL_DEPTH_COMPONENT), SM, SM, 0, gles.GL_DEPTH_COMPONENT, gles.GL_UNSIGNED_SHORT, null);
    glTexParameteri(gles.GL_TEXTURE_2D, gles.GL_TEXTURE_MIN_FILTER, gles.GL_NEAREST);
    glTexParameteri(gles.GL_TEXTURE_2D, gles.GL_TEXTURE_MAG_FILTER, gles.GL_NEAREST);
    var smcol: gles.GLuint = 0;
    glGenTextures(1, @ptrCast(&smcol));
    glBindTexture(gles.GL_TEXTURE_2D, smcol);
    glTexImage2D(gles.GL_TEXTURE_2D, 0, gles.GL_RGBA, SM, SM, 0, gles.GL_RGBA, gles.GL_UNSIGNED_BYTE, null);
    var fbo: gles.GLuint = 0;
    glGenFramebuffers(1, @ptrCast(&fbo));
    glBindFramebuffer(gles.GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(gles.GL_FRAMEBUFFER, gles.GL_COLOR_ATTACHMENT0, gles.GL_TEXTURE_2D, smcol, 0);
    glFramebufferTexture2D(gles.GL_FRAMEBUFFER, gles.GL_DEPTH_ATTACHMENT, gles.GL_TEXTURE_2D, dtex, 0);
    defer {
        glDeleteFramebuffers(1, &[_]gles.GLuint{fbo});
        glDeleteTextures(1, &[_]gles.GLuint{dtex});
        glDeleteTextures(1, &[_]gles.GLuint{smcol});
    }

    // === Pass 1: render the scene depth from the light's POV into the shadow map =================
    glUseProgram(dprog);
    glUniformMatrix4fv(glGetUniformLocation(dprog, "uLightVP"), 1, gles.GL_FALSE, &light_vp);
    glViewport(0, 0, SM, SM);
    glEnable(gles.GL_DEPTH_TEST);
    glDepthFunc(gles.GL_LESS);
    glClearColor(0, 0, 0, 1);
    glClear(gles.GL_COLOR_BUFFER_BIT | gles.GL_DEPTH_BUFFER_BIT);
    glEnableVertexAttribArray(0);
    glBindBuffer(gles.GL_ARRAY_BUFFER, gvbo);
    glVertexAttribPointer(0, 3, gles.GL_FLOAT, gles.GL_FALSE, 0, @ptrFromInt(0));
    glDrawArrays(gles.GL_TRIANGLES, 0, 6);
    glBindBuffer(gles.GL_ARRAY_BUFFER, ovbo);
    glVertexAttribPointer(0, 3, gles.GL_FLOAT, gles.GL_FALSE, 0, @ptrFromInt(0));
    glDrawArrays(gles.GL_TRIANGLES, 0, 6);
    try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());

    // === Pass 2: render from the camera, sampling the shadow map ================================
    glBindFramebuffer(gles.GL_FRAMEBUFFER, 0);
    glActiveTexture(gles.GL_TEXTURE0);
    glBindTexture(gles.GL_TEXTURE_2D, dtex);
    glTexParameteri(gles.GL_TEXTURE_2D, gles.GL_TEXTURE_COMPARE_MODE, @intCast(gles.GL_COMPARE_REF_TO_TEXTURE));
    glTexParameteri(gles.GL_TEXTURE_2D, gles.GL_TEXTURE_COMPARE_FUNC, @intCast(gles.GL_LEQUAL));
    glUseProgram(sprog);
    glUniformMatrix4fv(glGetUniformLocation(sprog, "uCamVP"), 1, gles.GL_FALSE, &cam_vp);
    glUniformMatrix4fv(glGetUniformLocation(sprog, "uLightVP"), 1, gles.GL_FALSE, &light_vp);
    glUniform1i(glGetUniformLocation(sprog, "uShadow"), 0);
    glUniform3f(glGetUniformLocation(sprog, "uNormal"), 0, 1, 0);
    glUniform3f(glGetUniformLocation(sprog, "uLightToLight"), 0.70710678, 0.70710678, 0);
    glUniform1f(glGetUniformLocation(sprog, "uBias"), 0.02);
    glViewport(0, 0, W, W);
    glEnable(gles.GL_DEPTH_TEST);
    glDepthFunc(gles.GL_LESS);
    glClearColor(0, 0, 0, 1);
    glClear(gles.GL_COLOR_BUFFER_BIT | gles.GL_DEPTH_BUFFER_BIT);
    // Ground: neutral grey base.
    glUniform3f(glGetUniformLocation(sprog, "uBase"), 0.9, 0.9, 0.9);
    glBindBuffer(gles.GL_ARRAY_BUFFER, gvbo);
    glVertexAttribPointer(0, 3, gles.GL_FLOAT, gles.GL_FALSE, 0, @ptrFromInt(0));
    glDrawArrays(gles.GL_TRIANGLES, 0, 6);
    // Occluder: blue base so its lit top face is identifiable. If unify_base, keeps the
    // grey base for both draws (no per-draw uBase change) to sidestep the nvidia multi-draw uniform-
    // aliasing bug (the occluder then renders lit-grey). The cast-shadow differential is unaffected.
    if (!unify_base) glUniform3f(glGetUniformLocation(sprog, "uBase"), 0.2, 0.4, 1.0);
    glBindBuffer(gles.GL_ARRAY_BUFFER, ovbo);
    glVertexAttribPointer(0, 3, gles.GL_FLOAT, gles.GL_FALSE, 0, @ptrFromInt(0));
    glDrawArrays(gles.GL_TRIANGLES, 0, 6);
    glDisable(gles.GL_DEPTH_TEST);
    try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());

    // --- Read back + assert -------------------------------------------------------------------
    const s: *state.Surface = @ptrCast(@alignCast(surf.?));
    const dev = s.display.device.?;
    const px = try flushMap(dev, s);
    const at = struct {
        fn p(buf: []const u8, x: usize, y: usize) [3]u8 {
            const i = (y * 64 + x) * 4;
            return .{ buf[i], buf[i + 1], buf[i + 2] };
        }
    }.p;
    // Expected Lambert: diffuse = dot((0,1,0),(1,1,0)/sqrt2) = 0.7071.
    //  lit grey ground   = 0.9 * (0.2 + 0.7071) = ~0.816 -> ~208
    //  shadowed grey     = 0.9 * 0.2            = 0.18   -> ~46
    //  lit blue occluder = blue 1.0*(0.907)     = ~0.907 -> ~231
    // Shadowed ground: world (-2,0,0) -> pixel (16,32). Inside the occluder's cast shadow, and the
    // occluder footprint (screen x [24,40]) does not cover it, so we see darkened ground.
    const shadowed = at(px, 16, 32);
    // Lit ground: world (3,0,0) -> pixel (56,32). Outside the shadow (shadow is world x [-3,-1]).
    const lit_ground = at(px, 56, 32);
    // Lit occluder top: world (0,2,0) -> pixel (32,32). Nearer than the ground, casts (not receives)
    // the shadow, so its top is fully lit.
    const occ_top = at(px, 32, 32);

    // Differential (the core cast-shadow assertion, driver-agnostic): the shadowed ground is
    // clearly darker than the lit ground, and the shadow is correctly placed (not just "some dark").
    try std.testing.expect(shadowed[0] < 90); // shadow -> ambient only (~46)
    try std.testing.expect(lit_ground[0] > 170); // lit -> ambient + diffuse (~208)
    try std.testing.expect(lit_ground[0] > shadowed[0] + 100); // placed correctly, not "some dark"
    if (unify_base) {
        // Grey occluder: its lit top face is bright (grey, so the red channel is high). The slab
        // occludes (casts) the shadow, so its own top is fully lit.
        try std.testing.expect(occ_top[0] > 170); // grey base, lit (~208)
    } else {
        // The occluder's lit top face is bright (dominant blue channel).
        try std.testing.expect(occ_top[2] > 170); // blue base, lit (~231)
        // Sanity: the occluder is blue, not grey (so pixel (32,32) is really the slab, not the ground).
        try std.testing.expect(occ_top[2] > occ_top[0] + 80);
    }

    glDeleteProgram(dprog);
    glDeleteProgram(sprog);
    glDeleteShader(dvs);
    glDeleteShader(dfs);
    glDeleteShader(svs);
    glDeleteShader(sfs);
}

test "GL_UNPACK_IMAGE_HEIGHT / SKIP_IMAGES: glTexImage3D reads a sub-volume of a wider source (array layers)" {
    // The 3D/array analog of GL_UNPACK_ROW_LENGTH: a glTexImage3D upload must address a sub-volume of
    // a wider source. image_height = source rows per slice (0 = the upload height), skip_images = the
    // leading source slices to skip. Differential: the source has a leading padding image (skipped)
    // and each source image is 2 rows tall while we upload height=1, so only the correct combination
    // of skip_images=1 + image_height=2 lands green at layer 1 (any other addressing lands red/white).
    _ = eglBindAPI(egl.EGL_OPENGL_ES_API);
    const dpy = eglGetPlatformDisplay(egl.EGL_PLATFORM_SURFACELESS_MESA, @ptrFromInt(0xE610B), null);
    try std.testing.expectEqual(egl.EGL_TRUE, eglInitialize(dpy, null, null));
    defer _ = eglTerminate(dpy);
    const cfg = state.configHandle(0);
    const ctx = eglCreateContext(dpy, cfg, egl.EGL_NO_CONTEXT, null);
    const pb_attribs = [_]EGLint{ egl.EGL_WIDTH, 32, egl.EGL_HEIGHT, 32, egl.EGL_NONE };
    const surf = eglCreatePbufferSurface(dpy, cfg, &pb_attribs);
    try std.testing.expectEqual(egl.EGL_TRUE, eglMakeCurrent(dpy, surf, surf, ctx));
    defer _ = eglDestroyContext(dpy, ctx);
    defer _ = eglDestroySurface(dpy, surf);
    defer _ = eglMakeCurrent(dpy, null, null, egl.EGL_NO_CONTEXT);

    const vs_src: [*:0]const gles.GLchar =
        \\attribute vec2 position;
        \\void main() { gl_Position = vec4(position, 0.0, 1.0); }
    ;
    const fs_src: [*:0]const gles.GLchar =
        \\precision mediump float;
        \\uniform sampler2DArray uArr;
        \\uniform vec3 uCoord;
        \\void main() { gl_FragColor = texture(uArr, uCoord); }
    ;
    const vs = glCreateShader(gles.GL_VERTEX_SHADER);
    glShaderSource(vs, 1, &[_]?[*:0]const gles.GLchar{vs_src}, null);
    glCompileShader(vs);
    const fs = glCreateShader(gles.GL_FRAGMENT_SHADER);
    glShaderSource(fs, 1, &[_]?[*:0]const gles.GLchar{fs_src}, null);
    glCompileShader(fs);
    const prog = glCreateProgram();
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glBindAttribLocation(prog, 0, "position");
    glLinkProgram(prog);
    glUseProgram(prog);
    defer {
        glDeleteProgram(prog);
        glDeleteShader(vs);
        glDeleteShader(fs);
    }

    // A 1x1 x 2-layer array. Source is 3 images, each image_height=2 rows of one texel:
    //   image 0 (skipped): [black, black]   image 1 -> layer 0: [red, white]   image 2 -> layer 1: [green, white]
    // With skip_images=1 + image_height=2 the two uploaded slices read image1.row0 (red) + image2.row0 (green).
    const K = [_]u8{ 0, 0, 0, 255 }; // black
    const R = [_]u8{ 255, 0, 0, 255 }; // red
    const G = [_]u8{ 0, 255, 0, 255 }; // green
    const Wt = [_]u8{ 255, 255, 255, 255 }; // white padding
    const srcbuf = K ++ K ++ R ++ Wt ++ G ++ Wt;

    var tex: gles.GLuint = 0;
    glGenTextures(1, @ptrCast(&tex));
    glActiveTexture(gles.GL_TEXTURE0);
    glBindTexture(gles.GL_TEXTURE_2D_ARRAY, tex);
    glTexParameteri(gles.GL_TEXTURE_2D_ARRAY, gles.GL_TEXTURE_MIN_FILTER, @intCast(gles.GL_NEAREST));
    glTexParameteri(gles.GL_TEXTURE_2D_ARRAY, gles.GL_TEXTURE_MAG_FILTER, @intCast(gles.GL_NEAREST));
    defer glDeleteTextures(1, &[_]gles.GLuint{tex});
    glPixelStorei(gles.GL_UNPACK_IMAGE_HEIGHT, 2); // source images are 2 rows tall
    glPixelStorei(gles.GL_UNPACK_SKIP_IMAGES, 1); // skip the leading black image
    glTexImage3D(gles.GL_TEXTURE_2D_ARRAY, 0, gles.GL_RGBA, 1, 1, 2, 0, gles.GL_RGBA, gles.GL_UNSIGNED_BYTE, &srcbuf);
    try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());
    glPixelStorei(gles.GL_UNPACK_IMAGE_HEIGHT, 0); // reset so later tests are unaffected
    glPixelStorei(gles.GL_UNPACK_SKIP_IMAGES, 0);

    const tri = [_]f32{ -1, -1, 3, -1, -1, 3 };
    var vbo: gles.GLuint = 0;
    glGenBuffers(1, @ptrCast(&vbo));
    glBindBuffer(gles.GL_ARRAY_BUFFER, vbo);
    glBufferData(gles.GL_ARRAY_BUFFER, @sizeOf(@TypeOf(tri)), &tri, gles.GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, gles.GL_FLOAT, gles.GL_FALSE, 0, @ptrFromInt(0));
    glEnableVertexAttribArray(0);
    defer glDeleteBuffers(1, &[_]gles.GLuint{vbo});
    glUniform1i(glGetUniformLocation(prog, "uArr"), 0);
    glViewport(0, 0, 32, 32);

    const s: *state.Surface = @ptrCast(@alignCast(surf.?));
    const dev = s.display.device.?;
    const c = (@as(usize, 16) * 32 + 16) * 4;

    // Layer 0 -> red (image 1, row 0).
    glUniform3f(glGetUniformLocation(prog, "uCoord"), 0.5, 0.5, 0.0);
    glClearColor(0, 0, 0, 1);
    glClear(gles.GL_COLOR_BUFFER_BIT);
    glDrawArrays(gles.GL_TRIANGLES, 0, 3);
    try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());
    {
        const px = try flushMap(dev, s);
        try std.testing.expect(px[c + 0] > 200 and px[c + 1] < 60 and px[c + 2] < 60); // red
    }

    // Layer 1 -> green (image 2, row 0). The combined skip_images + image_height differential.
    glUniform3f(glGetUniformLocation(prog, "uCoord"), 0.5, 0.5, 1.0);
    glClear(gles.GL_COLOR_BUFFER_BIT);
    glDrawArrays(gles.GL_TRIANGLES, 0, 3);
    try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());
    {
        const px = try flushMap(dev, s);
        try std.testing.expect(px[c + 1] > 200 and px[c + 0] < 60 and px[c + 2] < 60); // green
    }
}

test "glInvalidateFramebuffer validates the target + attachments and no-ops" {
    _ = eglBindAPI(egl.EGL_OPENGL_ES_API);
    const dpy = eglGetPlatformDisplay(egl.EGL_PLATFORM_SURFACELESS_MESA, @ptrFromInt(0xE61E0), null);
    try std.testing.expectEqual(egl.EGL_TRUE, eglInitialize(dpy, null, null));
    defer _ = eglTerminate(dpy);
    const cfg = state.configHandle(0);
    const ctx = eglCreateContext(dpy, cfg, egl.EGL_NO_CONTEXT, null);
    const pb_attribs = [_]EGLint{ egl.EGL_WIDTH, 16, egl.EGL_HEIGHT, 16, egl.EGL_NONE };
    const surf = eglCreatePbufferSurface(dpy, cfg, &pb_attribs);
    try std.testing.expectEqual(egl.EGL_TRUE, eglMakeCurrent(dpy, surf, surf, ctx));
    defer _ = eglDestroyContext(dpy, ctx);
    defer _ = eglDestroySurface(dpy, surf);
    defer _ = eglMakeCurrent(dpy, null, null, egl.EGL_NO_CONTEXT);

    // Valid: the default framebuffer's GL_COLOR/GL_DEPTH attachments.
    const dflt = [_]gles.GLenum{ gles.GL_COLOR, gles.GL_DEPTH };
    glInvalidateFramebuffer(gles.GL_FRAMEBUFFER, 2, &dflt);
    try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());
    // Valid: an FBO's GL_COLOR_ATTACHMENT0 via the sub-rectangle form.
    const fbo_att = [_]gles.GLenum{gles.GL_COLOR_ATTACHMENT0};
    glInvalidateSubFramebuffer(gles.GL_DRAW_FRAMEBUFFER, 1, &fbo_att, 0, 0, 8, 8);
    try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());

    // A bad target -> GL_INVALID_ENUM.
    glInvalidateFramebuffer(0x1234, 1, &fbo_att);
    try std.testing.expectEqual(gles.GL_INVALID_ENUM, glGetError());
    // A bad attachment enum -> GL_INVALID_ENUM.
    const bad = [_]gles.GLenum{0x9999};
    glInvalidateFramebuffer(gles.GL_FRAMEBUFFER, 1, &bad);
    try std.testing.expectEqual(gles.GL_INVALID_ENUM, glGetError());
    // A negative sub-rectangle extent -> GL_INVALID_VALUE.
    glInvalidateSubFramebuffer(gles.GL_FRAMEBUFFER, 1, &fbo_att, 0, 0, -1, 8);
    try std.testing.expectEqual(gles.GL_INVALID_VALUE, glGetError());
}

test "fence sync: glFenceSync is immediately signaled (Prism submits synchronously)" {
    _ = eglBindAPI(egl.EGL_OPENGL_ES_API);
    const dpy = eglGetPlatformDisplay(egl.EGL_PLATFORM_SURFACELESS_MESA, @ptrFromInt(0xE61E2), null);
    try std.testing.expectEqual(egl.EGL_TRUE, eglInitialize(dpy, null, null));
    defer _ = eglTerminate(dpy);
    const cfg = state.configHandle(0);
    const ctx = eglCreateContext(dpy, cfg, egl.EGL_NO_CONTEXT, null);
    const pb_attribs = [_]EGLint{ egl.EGL_WIDTH, 16, egl.EGL_HEIGHT, 16, egl.EGL_NONE };
    const surf = eglCreatePbufferSurface(dpy, cfg, &pb_attribs);
    try std.testing.expectEqual(egl.EGL_TRUE, eglMakeCurrent(dpy, surf, surf, ctx));
    defer _ = eglDestroyContext(dpy, ctx);
    defer _ = eglDestroySurface(dpy, surf);
    defer _ = eglMakeCurrent(dpy, null, null, egl.EGL_NO_CONTEXT);

    const sync = glFenceSync(gles.GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
    try std.testing.expect(sync != null);
    try std.testing.expectEqual(gles.GL_TRUE, glIsSync(sync));

    // The work is already complete: a client wait returns ALREADY_SIGNALED.
    try std.testing.expectEqual(gles.GL_ALREADY_SIGNALED, glClientWaitSync(sync, 0, 0));

    var status: gles.GLint = -1;
    var len: gles.GLsizei = -1;
    glGetSynciv(sync, gles.GL_SYNC_STATUS, 1, &len, @ptrCast(&status));
    try std.testing.expectEqual(@as(gles.GLsizei, 1), len);
    try std.testing.expectEqual(@as(gles.GLint, @intCast(gles.GL_SIGNALED)), status);
    var otype: gles.GLint = 0;
    glGetSynciv(sync, gles.GL_OBJECT_TYPE, 1, null, @ptrCast(&otype));
    try std.testing.expectEqual(@as(gles.GLint, @intCast(gles.GL_SYNC_FENCE)), otype);
    try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());

    // A bad condition is rejected.
    try std.testing.expect(glFenceSync(0x1234, 0) == null);
    try std.testing.expectEqual(gles.GL_INVALID_ENUM, glGetError());

    // Delete: the handle is no longer a sync, and a stale wait fails.
    glDeleteSync(sync);
    try std.testing.expectEqual(gles.GL_FALSE, glIsSync(sync));
    try std.testing.expectEqual(gles.GL_WAIT_FAILED, glClientWaitSync(sync, 0, 0));
    try std.testing.expectEqual(gles.GL_INVALID_VALUE, glGetError());
}

test "glBlitFramebuffer copies (scaled) an FBO's color to the default framebuffer" {
    _ = eglBindAPI(egl.EGL_OPENGL_ES_API);
    const dpy = eglGetPlatformDisplay(egl.EGL_PLATFORM_SURFACELESS_MESA, @ptrFromInt(0xE61E4), null);
    try std.testing.expectEqual(egl.EGL_TRUE, eglInitialize(dpy, null, null));
    defer _ = eglTerminate(dpy);
    const cfg = state.configHandle(0);
    const ctx = eglCreateContext(dpy, cfg, egl.EGL_NO_CONTEXT, null);
    const W: EGLint = 64;
    const pb_attribs = [_]EGLint{ egl.EGL_WIDTH, W, egl.EGL_HEIGHT, W, egl.EGL_NONE };
    const surf = eglCreatePbufferSurface(dpy, cfg, &pb_attribs);
    try std.testing.expectEqual(egl.EGL_TRUE, eglMakeCurrent(dpy, surf, surf, ctx));
    defer _ = eglDestroyContext(dpy, ctx);
    defer _ = eglDestroySurface(dpy, surf);
    defer _ = eglMakeCurrent(dpy, null, null, egl.EGL_NO_CONTEXT);

    // A fill shader renders solid green into a 16x16 FBO color texture.
    const vs_src: [*:0]const gles.GLchar =
        \\attribute vec2 position;
        \\void main() { gl_Position = vec4(position, 0.0, 1.0); }
    ;
    const fs_src: [*:0]const gles.GLchar =
        \\precision mediump float;
        \\void main() { gl_FragColor = vec4(0.0, 1.0, 0.0, 1.0); }
    ;
    const vs = glCreateShader(gles.GL_VERTEX_SHADER);
    glShaderSource(vs, 1, &[_]?[*:0]const gles.GLchar{vs_src}, null);
    glCompileShader(vs);
    const fs = glCreateShader(gles.GL_FRAGMENT_SHADER);
    glShaderSource(fs, 1, &[_]?[*:0]const gles.GLchar{fs_src}, null);
    glCompileShader(fs);
    const prog = glCreateProgram();
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glBindAttribLocation(prog, 0, "position");
    glLinkProgram(prog);
    glUseProgram(prog);

    const TS: gles.GLsizei = 16;
    var tex: gles.GLuint = 0;
    glGenTextures(1, @ptrCast(&tex));
    glBindTexture(gles.GL_TEXTURE_2D, tex);
    glTexImage2D(gles.GL_TEXTURE_2D, 0, gles.GL_RGBA, TS, TS, 0, gles.GL_RGBA, gles.GL_UNSIGNED_BYTE, null);
    var fbo: gles.GLuint = 0;
    glGenFramebuffers(1, @ptrCast(&fbo));
    glBindFramebuffer(gles.GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(gles.GL_FRAMEBUFFER, gles.GL_COLOR_ATTACHMENT0, gles.GL_TEXTURE_2D, tex, 0);

    const tri = [_]f32{ -1, -1, 3, -1, -1, 3 };
    var vbo: gles.GLuint = 0;
    glGenBuffers(1, @ptrCast(&vbo));
    glBindBuffer(gles.GL_ARRAY_BUFFER, vbo);
    glBufferData(gles.GL_ARRAY_BUFFER, @sizeOf(@TypeOf(tri)), &tri, gles.GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, gles.GL_FLOAT, gles.GL_FALSE, 0, @ptrFromInt(0));
    glEnableVertexAttribArray(0);
    glViewport(0, 0, TS, TS);
    glDrawArrays(gles.GL_TRIANGLES, 0, 3);
    try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());

    // Blit the FBO (READ) up to the full 64x64 default framebuffer (DRAW), scaled 16 -> 64.
    glBindFramebuffer(gles.GL_READ_FRAMEBUFFER, fbo);
    glBindFramebuffer(gles.GL_DRAW_FRAMEBUFFER, 0);
    glClearColor(0, 0, 0, 1);
    glClear(gles.GL_COLOR_BUFFER_BIT); // clear default to black first
    glBlitFramebuffer(0, 0, TS, TS, 0, 0, W, W, gles.GL_COLOR_BUFFER_BIT, gles.GL_NEAREST);
    try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());

    const s: *state.Surface = @ptrCast(@alignCast(surf.?));
    const dev = s.display.device.?;
    const px = try flushMap(dev, s);
    const c = (@as(usize, 32) * 64 + 32) * 4;
    try std.testing.expect(px[c + 1] > 200 and px[c + 0] < 60 and px[c + 2] < 60); // green blitted across
    // A corner too (the blit filled the whole target).
    const corner = (@as(usize, 4) * 64 + 4) * 4;
    try std.testing.expect(px[corner + 1] > 200);

    glBindFramebuffer(gles.GL_FRAMEBUFFER, 0);
    glDeleteFramebuffers(1, &[_]gles.GLuint{fbo});
    glDeleteBuffers(1, &[_]gles.GLuint{vbo});
    glDeleteTextures(1, &[_]gles.GLuint{tex});
    glDeleteProgram(prog);
    glDeleteShader(vs);
    glDeleteShader(fs);
}

test "glClearBufferfv clears the color buffer to a typed value" {
    _ = eglBindAPI(egl.EGL_OPENGL_ES_API);
    const dpy = eglGetPlatformDisplay(egl.EGL_PLATFORM_SURFACELESS_MESA, @ptrFromInt(0xE61E6), null);
    try std.testing.expectEqual(egl.EGL_TRUE, eglInitialize(dpy, null, null));
    defer _ = eglTerminate(dpy);
    const cfg = state.configHandle(0);
    const ctx = eglCreateContext(dpy, cfg, egl.EGL_NO_CONTEXT, null);
    const pb_attribs = [_]EGLint{ egl.EGL_WIDTH, 32, egl.EGL_HEIGHT, 32, egl.EGL_NONE };
    const surf = eglCreatePbufferSurface(dpy, cfg, &pb_attribs);
    try std.testing.expectEqual(egl.EGL_TRUE, eglMakeCurrent(dpy, surf, surf, ctx));
    defer _ = eglDestroyContext(dpy, ctx);
    defer _ = eglDestroySurface(dpy, surf);
    defer _ = eglMakeCurrent(dpy, null, null, egl.EGL_NO_CONTEXT);

    // A glClear to black first, then glClearBufferfv overrides the color to a known value.
    glClearColor(0, 0, 0, 1);
    glClear(gles.GL_COLOR_BUFFER_BIT);
    const val = [_]gles.GLfloat{ 64.0 / 255.0, 128.0 / 255.0, 192.0 / 255.0, 1.0 };
    glClearBufferfv(gles.GL_COLOR, 0, &val);
    try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());

    const s: *state.Surface = @ptrCast(@alignCast(surf.?));
    const dev = s.display.device.?;
    const px = try flushMap(dev, s);
    const c = (@as(usize, 16) * 32 + 16) * 4;
    try std.testing.expectEqual(@as(u8, 0x40), px[c + 0]);
    try std.testing.expectEqual(@as(u8, 0x80), px[c + 1]);
    try std.testing.expectEqual(@as(u8, 0xC0), px[c + 2]);

    // glClearColor state is NOT disturbed by glClearBufferfv (a subsequent glClear stays black).
    glClear(gles.GL_COLOR_BUFFER_BIT);
    const px2 = try flushMap(dev, s);
    try std.testing.expectEqual(@as(u8, 0), px2[c + 0]);

    // A bad buffer enum is rejected.
    glClearBufferfv(0x1234, 0, &val);
    try std.testing.expectEqual(gles.GL_INVALID_ENUM, glGetError());
}

test "glCopyBufferSubData copies bytes between two buffers" {
    _ = eglBindAPI(egl.EGL_OPENGL_ES_API);
    const dpy = eglGetPlatformDisplay(egl.EGL_PLATFORM_SURFACELESS_MESA, @ptrFromInt(0xE61E8), null);
    try std.testing.expectEqual(egl.EGL_TRUE, eglInitialize(dpy, null, null));
    defer _ = eglTerminate(dpy);
    const cfg = state.configHandle(0);
    const ctx = eglCreateContext(dpy, cfg, egl.EGL_NO_CONTEXT, null);
    const pb_attribs = [_]EGLint{ egl.EGL_WIDTH, 16, egl.EGL_HEIGHT, 16, egl.EGL_NONE };
    const surf = eglCreatePbufferSurface(dpy, cfg, &pb_attribs);
    try std.testing.expectEqual(egl.EGL_TRUE, eglMakeCurrent(dpy, surf, surf, ctx));
    defer _ = eglDestroyContext(dpy, ctx);
    defer _ = eglDestroySurface(dpy, surf);
    defer _ = eglMakeCurrent(dpy, null, null, egl.EGL_NO_CONTEXT);

    const src_data = [_]u8{ 10, 20, 30, 40, 50, 60, 70, 80 };
    var src_buf: gles.GLuint = 0;
    glGenBuffers(1, @ptrCast(&src_buf));
    glBindBuffer(gles.GL_COPY_READ_BUFFER, src_buf);
    glBufferData(gles.GL_COPY_READ_BUFFER, src_data.len, &src_data, gles.GL_STATIC_DRAW);

    var dst_buf: gles.GLuint = 0;
    glGenBuffers(1, @ptrCast(&dst_buf));
    glBindBuffer(gles.GL_COPY_WRITE_BUFFER, dst_buf);
    glBufferData(gles.GL_COPY_WRITE_BUFFER, 8, null, gles.GL_STATIC_DRAW);

    // Copy 4 bytes from src[2..6] to dst[1..5].
    glCopyBufferSubData(gles.GL_COPY_READ_BUFFER, gles.GL_COPY_WRITE_BUFFER, 2, 1, 4);
    try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());

    // Read the destination back via glGetBufferSubData-equivalent: map the buffer object.
    const db = gles.findBuffer(dst_buf).?;
    try std.testing.expectEqual(@as(u8, 30), db.bytes.items[1]);
    try std.testing.expectEqual(@as(u8, 40), db.bytes.items[2]);
    try std.testing.expectEqual(@as(u8, 50), db.bytes.items[3]);
    try std.testing.expectEqual(@as(u8, 60), db.bytes.items[4]);

    // Out-of-range copy is rejected.
    glCopyBufferSubData(gles.GL_COPY_READ_BUFFER, gles.GL_COPY_WRITE_BUFFER, 0, 0, 100);
    try std.testing.expectEqual(gles.GL_INVALID_VALUE, glGetError());

    glDeleteBuffers(1, &[_]gles.GLuint{ src_buf, dst_buf });
}

test "glReadBuffer / glDrawBuffers validate + store the color-buffer selection" {
    _ = eglBindAPI(egl.EGL_OPENGL_ES_API);
    const dpy = eglGetPlatformDisplay(egl.EGL_PLATFORM_SURFACELESS_MESA, @ptrFromInt(0xE61EA), null);
    try std.testing.expectEqual(egl.EGL_TRUE, eglInitialize(dpy, null, null));
    defer _ = eglTerminate(dpy);
    const cfg = state.configHandle(0);
    const ctx = eglCreateContext(dpy, cfg, egl.EGL_NO_CONTEXT, null);
    const pb_attribs = [_]EGLint{ egl.EGL_WIDTH, 16, egl.EGL_HEIGHT, 16, egl.EGL_NONE };
    const surf = eglCreatePbufferSurface(dpy, cfg, &pb_attribs);
    try std.testing.expectEqual(egl.EGL_TRUE, eglMakeCurrent(dpy, surf, surf, ctx));
    defer _ = eglDestroyContext(dpy, ctx);
    defer _ = eglDestroySurface(dpy, surf);
    defer _ = eglMakeCurrent(dpy, null, null, egl.EGL_NO_CONTEXT);

    glReadBuffer(gles.GL_COLOR_ATTACHMENT0);
    try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());
    var iv: gles.GLint = -1;
    glGetIntegerv(gles.GL_READ_BUFFER, @ptrCast(&iv));
    try std.testing.expectEqual(@as(gles.GLint, @intCast(gles.GL_COLOR_ATTACHMENT0)), iv);

    const bufs = [_]gles.GLenum{gles.GL_COLOR_ATTACHMENT0};
    glDrawBuffers(1, &bufs);
    try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());
    glGetIntegerv(gles.GL_DRAW_BUFFER0, @ptrCast(&iv));
    try std.testing.expectEqual(@as(gles.GLint, @intCast(gles.GL_COLOR_ATTACHMENT0)), iv);

    // GL_MAX_DRAW_BUFFERS is queryable (MRT count).
    glGetIntegerv(gles.GL_MAX_DRAW_BUFFERS, @ptrCast(&iv));
    try std.testing.expect(iv >= 1);

    // A bad read-buffer enum is rejected.
    glReadBuffer(0x1234);
    try std.testing.expectEqual(gles.GL_INVALID_ENUM, glGetError());
}

fn runOcclusionQuery(dpy_magic: usize) !void {
    _ = eglBindAPI(egl.EGL_OPENGL_ES_API);
    const dpy = eglGetPlatformDisplay(egl.EGL_PLATFORM_SURFACELESS_MESA, @ptrFromInt(dpy_magic), null);
    if (eglInitialize(dpy, null, null) != egl.EGL_TRUE) return error.SkipZigTest;
    defer _ = eglTerminate(dpy);
    const cfg = state.configHandle(0);
    const ctx = eglCreateContext(dpy, cfg, egl.EGL_NO_CONTEXT, null);
    const pb_attribs = [_]EGLint{ egl.EGL_WIDTH, 32, egl.EGL_HEIGHT, 32, egl.EGL_NONE };
    const surf = eglCreatePbufferSurface(dpy, cfg, &pb_attribs);
    try std.testing.expectEqual(egl.EGL_TRUE, eglMakeCurrent(dpy, surf, surf, ctx));
    defer _ = eglDestroyContext(dpy, ctx);
    defer _ = eglDestroySurface(dpy, surf);
    defer _ = eglMakeCurrent(dpy, null, null, egl.EGL_NO_CONTEXT);

    const vs_src: [*:0]const gles.GLchar =
        \\attribute vec2 position;
        \\void main() { gl_Position = vec4(position, 0.0, 1.0); }
    ;
    const fs_src: [*:0]const gles.GLchar =
        \\precision mediump float;
        \\void main() { gl_FragColor = vec4(1.0, 1.0, 1.0, 1.0); }
    ;
    const vs = glCreateShader(gles.GL_VERTEX_SHADER);
    glShaderSource(vs, 1, &[_]?[*:0]const gles.GLchar{vs_src}, null);
    glCompileShader(vs);
    const fs = glCreateShader(gles.GL_FRAGMENT_SHADER);
    glShaderSource(fs, 1, &[_]?[*:0]const gles.GLchar{fs_src}, null);
    glCompileShader(fs);
    const prog = glCreateProgram();
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glBindAttribLocation(prog, 0, "position");
    glLinkProgram(prog);
    glUseProgram(prog);

    var vbo: gles.GLuint = 0;
    glGenBuffers(1, @ptrCast(&vbo));
    glBindBuffer(gles.GL_ARRAY_BUFFER, vbo);
    // Verts 0..2 = a big visible triangle; verts 3..5 = a degenerate (zero-area) triangle.
    const verts = [_]f32{ -0.8, -0.8, 0.8, -0.8, 0.0, 0.8, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1 };
    glBufferData(gles.GL_ARRAY_BUFFER, @sizeOf(@TypeOf(verts)), &verts, gles.GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, gles.GL_FLOAT, gles.GL_FALSE, 0, @ptrFromInt(0));
    glEnableVertexAttribArray(0);
    glViewport(0, 0, 32, 32);

    var q: gles.GLuint = 0;
    glGenQueries(1, @ptrCast(&q));

    // Query 1: the visible triangle -> some samples pass.
    glClearColor(0, 0, 0, 1);
    glClear(gles.GL_COLOR_BUFFER_BIT);
    glBeginQuery(gles.GL_ANY_SAMPLES_PASSED, q);
    try std.testing.expectEqual(gles.GL_TRUE, glIsQuery(q));
    glDrawArrays(gles.GL_TRIANGLES, 0, 3);
    glEndQuery(gles.GL_ANY_SAMPLES_PASSED);
    var res: u32 = 99;
    glGetQueryObjectuiv(q, gles.GL_QUERY_RESULT_AVAILABLE, &res);
    try std.testing.expectEqual(@as(u32, 1), res); // synchronous -> immediately available
    glGetQueryObjectuiv(q, gles.GL_QUERY_RESULT, &res);
    try std.testing.expectEqual(@as(u32, 1), res); // samples passed

    // Query 2: the degenerate triangle -> no samples pass.
    glBeginQuery(gles.GL_ANY_SAMPLES_PASSED, q);
    glDrawArrays(gles.GL_TRIANGLES, 3, 3);
    glEndQuery(gles.GL_ANY_SAMPLES_PASSED);
    glGetQueryObjectuiv(q, gles.GL_QUERY_RESULT, &res);
    try std.testing.expectEqual(@as(u32, 0), res); // nothing rasterized
    try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());

    glDeleteQueries(1, &[_]gles.GLuint{q});
    glDeleteBuffers(1, &[_]gles.GLuint{vbo});
    glDeleteProgram(prog);
    glDeleteShader(vs);
    glDeleteShader(fs);
}

test "occlusion query: GL_ANY_SAMPLES_PASSED reports a visible draw and a zero-coverage draw (software)" {
    try runOcclusionQuery(0xE61EC);
}

test "occlusion query on NVIDIA GPU: GL_ANY_SAMPLES_PASSED counts real ZPASS samples (skips without a GPU)" {
    // Forces the real nvidia device: the visible triangle must report samples-passed=1 and the
    // Degenerate triangle must report 0. Discriminator: the GPU ZPASS counter (not the old
    // conservative "always passed=1") drives the result. Reads the hardware ZPASS pixel count.
    const saved = state.pinned_driver;
    state.pinned_driver = "nvidia";
    defer state.pinned_driver = saved;
    try runOcclusionQuery(0xE61ED);
}

test "glGetBufferSubData reads back a sub-range of a buffer's data" {
    _ = eglBindAPI(egl.EGL_OPENGL_ES_API);
    const dpy = eglGetPlatformDisplay(egl.EGL_PLATFORM_SURFACELESS_MESA, @ptrFromInt(0xE61EE), null);
    try std.testing.expectEqual(egl.EGL_TRUE, eglInitialize(dpy, null, null));
    defer _ = eglTerminate(dpy);
    const cfg = state.configHandle(0);
    const ctx = eglCreateContext(dpy, cfg, egl.EGL_NO_CONTEXT, null);
    const pb_attribs = [_]EGLint{ egl.EGL_WIDTH, 16, egl.EGL_HEIGHT, 16, egl.EGL_NONE };
    const surf = eglCreatePbufferSurface(dpy, cfg, &pb_attribs);
    try std.testing.expectEqual(egl.EGL_TRUE, eglMakeCurrent(dpy, surf, surf, ctx));
    defer _ = eglDestroyContext(dpy, ctx);
    defer _ = eglDestroySurface(dpy, surf);
    defer _ = eglMakeCurrent(dpy, null, null, egl.EGL_NO_CONTEXT);

    const data = [_]f32{ 1.5, 2.5, 3.5, 4.5, 5.5, 6.5 };
    var vbo: gles.GLuint = 0;
    glGenBuffers(1, @ptrCast(&vbo));
    glBindBuffer(gles.GL_ARRAY_BUFFER, vbo);
    glBufferData(gles.GL_ARRAY_BUFFER, @sizeOf(@TypeOf(data)), &data, gles.GL_STATIC_DRAW);

    // Read back floats 1..3 (bytes 4..16).
    var got: [3]f32 = undefined;
    glGetBufferSubData(gles.GL_ARRAY_BUFFER, 4, 12, &got);
    try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());
    try std.testing.expectEqual(@as(f32, 2.5), got[0]);
    try std.testing.expectEqual(@as(f32, 3.5), got[1]);
    try std.testing.expectEqual(@as(f32, 4.5), got[2]);

    // Out-of-range read is rejected.
    glGetBufferSubData(gles.GL_ARRAY_BUFFER, 0, 1000, &got);
    try std.testing.expectEqual(gles.GL_INVALID_VALUE, glGetError());

    glDeleteBuffers(1, &[_]gles.GLuint{vbo});

    // glGetInternalformativ reports the supported MSAA sample counts (4x, 2x).
    var nc: gles.GLint = -1;
    glGetInternalformativ(gles.GL_RENDERBUFFER, gles.GL_RGBA8, gles.GL_NUM_SAMPLE_COUNTS, 1, @ptrCast(&nc));
    try std.testing.expectEqual(@as(gles.GLint, 2), nc);
    var samples: [2]gles.GLint = .{ 0, 0 };
    glGetInternalformativ(gles.GL_RENDERBUFFER, gles.GL_RGBA8, gles.GL_SAMPLES, 2, &samples);
    try std.testing.expectEqual([2]gles.GLint{ 4, 2 }, samples);
    try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());
}

test "glVertexAttrib4f feeds a DISABLED attribute's generic value to the whole draw" {
    _ = eglBindAPI(egl.EGL_OPENGL_ES_API);
    const dpy = eglGetPlatformDisplay(egl.EGL_PLATFORM_SURFACELESS_MESA, @ptrFromInt(0xE61F0), null);
    try std.testing.expectEqual(egl.EGL_TRUE, eglInitialize(dpy, null, null));
    defer _ = eglTerminate(dpy);
    const cfg = state.configHandle(0);
    const ctx = eglCreateContext(dpy, cfg, egl.EGL_NO_CONTEXT, null);
    const pb_attribs = [_]EGLint{ egl.EGL_WIDTH, 32, egl.EGL_HEIGHT, 32, egl.EGL_NONE };
    const surf = eglCreatePbufferSurface(dpy, cfg, &pb_attribs);
    try std.testing.expectEqual(egl.EGL_TRUE, eglMakeCurrent(dpy, surf, surf, ctx));
    defer _ = eglDestroyContext(dpy, ctx);
    defer _ = eglDestroySurface(dpy, surf);
    defer _ = eglMakeCurrent(dpy, null, null, egl.EGL_NO_CONTEXT);

    const vs_src: [*:0]const gles.GLchar =
        \\attribute vec2 position;
        \\attribute vec4 aColor;
        \\varying vec4 vColor;
        \\void main() { gl_Position = vec4(position, 0.0, 1.0); vColor = aColor; }
    ;
    const fs_src: [*:0]const gles.GLchar =
        \\precision mediump float;
        \\varying vec4 vColor;
        \\void main() { gl_FragColor = vColor; }
    ;
    const vs = glCreateShader(gles.GL_VERTEX_SHADER);
    glShaderSource(vs, 1, &[_]?[*:0]const gles.GLchar{vs_src}, null);
    glCompileShader(vs);
    const fs = glCreateShader(gles.GL_FRAGMENT_SHADER);
    glShaderSource(fs, 1, &[_]?[*:0]const gles.GLchar{fs_src}, null);
    glCompileShader(fs);
    const prog = glCreateProgram();
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glBindAttribLocation(prog, 0, "position");
    glBindAttribLocation(prog, 1, "aColor");
    glLinkProgram(prog);
    glUseProgram(prog);

    const tri = [_]f32{ -0.8, -0.8, 0.8, -0.8, 0.0, 0.8 };
    var vbo: gles.GLuint = 0;
    glGenBuffers(1, @ptrCast(&vbo));
    glBindBuffer(gles.GL_ARRAY_BUFFER, vbo);
    glBufferData(gles.GL_ARRAY_BUFFER, @sizeOf(@TypeOf(tri)), &tri, gles.GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, gles.GL_FLOAT, gles.GL_FALSE, 0, @ptrFromInt(0));
    glEnableVertexAttribArray(0);
    // aColor (attrib 1) is disabled. Its constant value comes from glVertexAttrib4f: solid red.
    glVertexAttrib4f(1, 1.0, 0.0, 0.0, 1.0);

    glViewport(0, 0, 32, 32);
    glClearColor(0, 0, 0, 1);
    glClear(gles.GL_COLOR_BUFFER_BIT);
    glDrawArrays(gles.GL_TRIANGLES, 0, 3);
    try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());

    const s: *state.Surface = @ptrCast(@alignCast(surf.?));
    const dev = s.display.device.?;
    const px = try flushMap(dev, s);
    const c = (@as(usize, 16) * 32 + 16) * 4;
    try std.testing.expect(px[c + 0] > 200 and px[c + 1] < 60 and px[c + 2] < 60); // the generic red

    // glGetVertexAttribfv(GL_CURRENT_VERTEX_ATTRIB) reflects the set value.
    var cur: [4]gles.GLfloat = undefined;
    glGetVertexAttribfv(1, gles.GL_CURRENT_VERTEX_ATTRIB, &cur);
    try std.testing.expectEqual([4]gles.GLfloat{ 1, 0, 0, 1 }, cur);

    glDeleteBuffers(1, &[_]gles.GLuint{vbo});
    glDeleteProgram(prog);
    glDeleteShader(vs);
    glDeleteShader(fs);
}

test "glGetShaderSource reads back the stored source; glDepthRangef + glHint validate" {
    _ = eglBindAPI(egl.EGL_OPENGL_ES_API);
    const dpy = eglGetPlatformDisplay(egl.EGL_PLATFORM_SURFACELESS_MESA, @ptrFromInt(0xE61F4), null);
    try std.testing.expectEqual(egl.EGL_TRUE, eglInitialize(dpy, null, null));
    defer _ = eglTerminate(dpy);
    const cfg = state.configHandle(0);
    const ctx = eglCreateContext(dpy, cfg, egl.EGL_NO_CONTEXT, null);
    const pb_attribs = [_]EGLint{ egl.EGL_WIDTH, 16, egl.EGL_HEIGHT, 16, egl.EGL_NONE };
    const surf = eglCreatePbufferSurface(dpy, cfg, &pb_attribs);
    try std.testing.expectEqual(egl.EGL_TRUE, eglMakeCurrent(dpy, surf, surf, ctx));
    defer _ = eglDestroyContext(dpy, ctx);
    defer _ = eglDestroySurface(dpy, surf);
    defer _ = eglMakeCurrent(dpy, null, null, egl.EGL_NO_CONTEXT);

    const src: [*:0]const gles.GLchar = "void main() { gl_FragColor = vec4(1.0); }";
    const sh = glCreateShader(gles.GL_FRAGMENT_SHADER);
    glShaderSource(sh, 1, &[_]?[*:0]const gles.GLchar{src}, null);
    // glGetShaderSource reads the stored text back, NUL-terminated.
    var buf: [128]gles.GLchar = undefined;
    var len: gles.GLint = -1;
    glGetShaderSource(sh, buf.len, &len, &buf);
    try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());
    const want = std.mem.span(src);
    try std.testing.expectEqual(@as(gles.GLint, @intCast(want.len)), len);
    try std.testing.expectEqualStrings(want, std.mem.span(@as([*:0]const u8, @ptrCast(&buf))));
    glDeleteShader(sh);

    // glDepthRangef stores the range (reported by GL_DEPTH_RANGE).
    glDepthRangef(0.25, 0.75);
    var dr: [2]gles.GLfloat = undefined;
    glGetFloatv(gles.GL_DEPTH_RANGE, &dr);
    try std.testing.expectApproxEqAbs(@as(f32, 0.25), dr[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.75), dr[1], 1e-6);

    // glReleaseShaderCompiler is an accepted no-op. glHint validates.
    glReleaseShaderCompiler();
    glHint(gles.GL_GENERATE_MIPMAP_HINT, gles.GL_NICEST);
    try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());
    glHint(gles.GL_GENERATE_MIPMAP_HINT, 0x9999);
    try std.testing.expectEqual(gles.GL_INVALID_ENUM, glGetError());
}

test "glVertexAttribPointer packed types: a normalized GL_UNSIGNED_BYTE color attribute samples correctly" {
    _ = eglBindAPI(egl.EGL_OPENGL_ES_API);
    const dpy = eglGetPlatformDisplay(egl.EGL_PLATFORM_SURFACELESS_MESA, @ptrFromInt(0xE61F2), null);
    try std.testing.expectEqual(egl.EGL_TRUE, eglInitialize(dpy, null, null));
    defer _ = eglTerminate(dpy);
    const cfg = state.configHandle(0);
    const ctx = eglCreateContext(dpy, cfg, egl.EGL_NO_CONTEXT, null);
    const pb_attribs = [_]EGLint{ egl.EGL_WIDTH, 32, egl.EGL_HEIGHT, 32, egl.EGL_NONE };
    const surf = eglCreatePbufferSurface(dpy, cfg, &pb_attribs);
    try std.testing.expectEqual(egl.EGL_TRUE, eglMakeCurrent(dpy, surf, surf, ctx));
    defer _ = eglDestroyContext(dpy, ctx);
    defer _ = eglDestroySurface(dpy, surf);
    defer _ = eglMakeCurrent(dpy, null, null, egl.EGL_NO_CONTEXT);

    const vs_src: [*:0]const gles.GLchar =
        \\attribute vec2 position;
        \\attribute vec4 aColor;
        \\varying vec4 vColor;
        \\void main() { gl_Position = vec4(position, 0.0, 1.0); vColor = aColor; }
    ;
    const fs_src: [*:0]const gles.GLchar =
        \\precision mediump float;
        \\varying vec4 vColor;
        \\void main() { gl_FragColor = vColor; }
    ;
    const vs = glCreateShader(gles.GL_VERTEX_SHADER);
    glShaderSource(vs, 1, &[_]?[*:0]const gles.GLchar{vs_src}, null);
    glCompileShader(vs);
    const fs = glCreateShader(gles.GL_FRAGMENT_SHADER);
    glShaderSource(fs, 1, &[_]?[*:0]const gles.GLchar{fs_src}, null);
    glCompileShader(fs);
    const prog = glCreateProgram();
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glBindAttribLocation(prog, 0, "position");
    glBindAttribLocation(prog, 1, "aColor");
    glLinkProgram(prog);
    glUseProgram(prog);

    // Interleaved: position (2 floats) + color packed as 4 unsigned bytes (green, normalized).
    const Vtx = extern struct { x: f32, y: f32, r: u8, g: u8, b: u8, a: u8 };
    const tri = [3]Vtx{
        .{ .x = -0.8, .y = -0.8, .r = 0, .g = 255, .b = 0, .a = 255 },
        .{ .x = 0.8, .y = -0.8, .r = 0, .g = 255, .b = 0, .a = 255 },
        .{ .x = 0.0, .y = 0.8, .r = 0, .g = 255, .b = 0, .a = 255 },
    };
    var vbo: gles.GLuint = 0;
    glGenBuffers(1, @ptrCast(&vbo));
    glBindBuffer(gles.GL_ARRAY_BUFFER, vbo);
    glBufferData(gles.GL_ARRAY_BUFFER, @sizeOf(@TypeOf(tri)), &tri, gles.GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, gles.GL_FLOAT, gles.GL_FALSE, @sizeOf(Vtx), @ptrFromInt(0));
    glVertexAttribPointer(1, 4, gles.GL_UNSIGNED_BYTE, gles.GL_TRUE, @sizeOf(Vtx), @ptrFromInt(8));
    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);

    glViewport(0, 0, 32, 32);
    glClearColor(0, 0, 0, 1);
    glClear(gles.GL_COLOR_BUFFER_BIT);
    glDrawArrays(gles.GL_TRIANGLES, 0, 3);
    try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());

    const s: *state.Surface = @ptrCast(@alignCast(surf.?));
    const dev = s.display.device.?;
    const px = try flushMap(dev, s);
    const c = (@as(usize, 16) * 32 + 16) * 4;
    // The packed byte color (0,255,0) normalized to (0,1,0) -> green center.
    try std.testing.expect(px[c + 1] > 200 and px[c + 0] < 60 and px[c + 2] < 60);

    glDeleteBuffers(1, &[_]gles.GLuint{vbo});
    glDeleteProgram(prog);
    glDeleteShader(vs);
    glDeleteShader(fs);
}

test "glVertexAttribIPointer: a GLES3 integer attribute reaches an `in ivec4` VS input RAW (not normalized)" {
    _ = eglBindAPI(egl.EGL_OPENGL_ES_API);
    const dpy = eglGetPlatformDisplay(egl.EGL_PLATFORM_SURFACELESS_MESA, @ptrFromInt(0xE61FA), null);
    try std.testing.expectEqual(egl.EGL_TRUE, eglInitialize(dpy, null, null));
    defer _ = eglTerminate(dpy);
    const cfg = state.configHandle(0);
    const ctx = eglCreateContext(dpy, cfg, egl.EGL_NO_CONTEXT, null);
    const pb_attribs = [_]EGLint{ egl.EGL_WIDTH, 32, egl.EGL_HEIGHT, 32, egl.EGL_NONE };
    const surf = eglCreatePbufferSurface(dpy, cfg, &pb_attribs);
    try std.testing.expectEqual(egl.EGL_TRUE, eglMakeCurrent(dpy, surf, surf, ctx));
    defer _ = eglDestroyContext(dpy, ctx);
    defer _ = eglDestroySurface(dpy, surf);
    defer _ = eglMakeCurrent(dpy, null, null, egl.EGL_NO_CONTEXT);

    // The VS reads `aData` as an `in ivec4` INTEGER input and routes the RAW integer x
    // (200) through the color: float(200)/255.0 ~= 0.784 -> R ~= 200. If the attribute were
    // wrongly NORMALIZED (like glVertexAttribPointer with GL_TRUE) the value would already be
    // 200/255 ~= 0.784, and float(0.784)/255 ~= 0.003 -> R ~= 1. So R ~200 vs R ~1 is the
    // raw-integer-vs-normalized differential.
    const vs_src: [*:0]const gles.GLchar =
        \\#version 300 es
        \\in vec2 position;
        \\in ivec4 aData;
        \\out vec4 vColor;
        \\void main() {
        \\  gl_Position = vec4(position, 0.0, 1.0);
        \\  vColor = vec4(float(aData.x) / 255.0, float(aData.y) / 255.0, 0.0, 1.0);
        \\}
    ;
    const fs_src: [*:0]const gles.GLchar =
        \\#version 300 es
        \\precision mediump float;
        \\in vec4 vColor;
        \\out vec4 frag;
        \\void main() { frag = vColor; }
    ;
    const vs = glCreateShader(gles.GL_VERTEX_SHADER);
    glShaderSource(vs, 1, &[_]?[*:0]const gles.GLchar{vs_src}, null);
    glCompileShader(vs);
    var vs_ok: gles.GLint = 0;
    glGetShaderiv(vs, gles.GL_COMPILE_STATUS, @ptrCast(&vs_ok));
    try std.testing.expect(vs_ok == gles.GL_TRUE);
    const fs = glCreateShader(gles.GL_FRAGMENT_SHADER);
    glShaderSource(fs, 1, &[_]?[*:0]const gles.GLchar{fs_src}, null);
    glCompileShader(fs);
    const prog = glCreateProgram();
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glBindAttribLocation(prog, 0, "position");
    glBindAttribLocation(prog, 1, "aData");
    glLinkProgram(prog);
    glUseProgram(prog);

    // Interleaved: position (2 floats) + 4 unsigned bytes (200, 40, 0, 0) delivered raw.
    const Vtx = extern struct { x: f32, y: f32, r: u8, g: u8, b: u8, a: u8 };
    const tri = [3]Vtx{
        .{ .x = -0.8, .y = -0.8, .r = 200, .g = 40, .b = 0, .a = 0 },
        .{ .x = 0.8, .y = -0.8, .r = 200, .g = 40, .b = 0, .a = 0 },
        .{ .x = 0.0, .y = 0.8, .r = 200, .g = 40, .b = 0, .a = 0 },
    };
    var vbo: gles.GLuint = 0;
    glGenBuffers(1, @ptrCast(&vbo));
    glBindBuffer(gles.GL_ARRAY_BUFFER, vbo);
    glBufferData(gles.GL_ARRAY_BUFFER, @sizeOf(@TypeOf(tri)), &tri, gles.GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, gles.GL_FLOAT, gles.GL_FALSE, @sizeOf(Vtx), @ptrFromInt(0));
    glVertexAttribIPointer(1, 4, gles.GL_UNSIGNED_BYTE, @sizeOf(Vtx), @ptrFromInt(8));
    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);

    // glGetVertexAttribiv reports attribute 1 as an integer array, attribute 0 as not.
    var is_int: gles.GLint = -1;
    glGetVertexAttribiv(1, gles.GL_VERTEX_ATTRIB_ARRAY_INTEGER, @ptrCast(&is_int));
    try std.testing.expectEqual(@as(gles.GLint, 1), is_int);
    glGetVertexAttribiv(0, gles.GL_VERTEX_ATTRIB_ARRAY_INTEGER, @ptrCast(&is_int));
    try std.testing.expectEqual(@as(gles.GLint, 0), is_int);

    glViewport(0, 0, 32, 32);
    glClearColor(0, 0, 0, 1);
    glClear(gles.GL_COLOR_BUFFER_BIT);
    glDrawArrays(gles.GL_TRIANGLES, 0, 3);
    try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());

    const s: *state.Surface = @ptrCast(@alignCast(surf.?));
    const dev = s.display.device.?;
    const px = try flushMap(dev, s);
    const c = (@as(usize, 16) * 32 + 16) * 4;
    // RAW integer 200 -> 200/255 -> R ~200 (well above the ~1 a normalized read would give);
    // RAW integer 40 -> 40/255 -> G ~40. B stays 0.
    try std.testing.expect(px[c + 0] > 180); // R reflects the RAW 200, NOT a normalized ~1
    try std.testing.expect(px[c + 1] > 25 and px[c + 1] < 60); // G reflects the RAW 40
    try std.testing.expect(px[c + 2] < 20); // B ~0

    glDeleteBuffers(1, &[_]gles.GLuint{vbo});
    glDeleteProgram(prog);
    glDeleteShader(vs);
    glDeleteShader(fs);
}

test "glRenderbufferStorageMultisample: an MSAA FBO renders anti-aliased + resolves via glBlitFramebuffer" {
    _ = eglBindAPI(egl.EGL_OPENGL_ES_API);
    const dpy = eglGetPlatformDisplay(egl.EGL_PLATFORM_SURFACELESS_MESA, @ptrFromInt(0xE61F6), null);
    try std.testing.expectEqual(egl.EGL_TRUE, eglInitialize(dpy, null, null));
    defer _ = eglTerminate(dpy);
    const cfg = state.configHandle(0);
    const ctx = eglCreateContext(dpy, cfg, egl.EGL_NO_CONTEXT, null);
    const W: EGLint = 64;
    const pb_attribs = [_]EGLint{ egl.EGL_WIDTH, W, egl.EGL_HEIGHT, W, egl.EGL_NONE };
    const surf = eglCreatePbufferSurface(dpy, cfg, &pb_attribs);
    try std.testing.expectEqual(egl.EGL_TRUE, eglMakeCurrent(dpy, surf, surf, ctx));
    defer _ = eglDestroyContext(dpy, ctx);
    defer _ = eglDestroySurface(dpy, surf);
    defer _ = eglMakeCurrent(dpy, null, null, egl.EGL_NO_CONTEXT);

    const vs_src: [*:0]const gles.GLchar =
        \\attribute vec2 position;
        \\void main() { gl_Position = vec4(position, 0.0, 1.0); }
    ;
    const fs_src: [*:0]const gles.GLchar =
        \\precision mediump float;
        \\void main() { gl_FragColor = vec4(1.0); }
    ;
    const vs = glCreateShader(gles.GL_VERTEX_SHADER);
    glShaderSource(vs, 1, &[_]?[*:0]const gles.GLchar{vs_src}, null);
    glCompileShader(vs);
    const fs = glCreateShader(gles.GL_FRAGMENT_SHADER);
    glShaderSource(fs, 1, &[_]?[*:0]const gles.GLchar{fs_src}, null);
    glCompileShader(fs);
    const prog = glCreateProgram();
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glBindAttribLocation(prog, 0, "position");
    glLinkProgram(prog);
    glUseProgram(prog);

    // A 4x-MULTISAMPLED color renderbuffer attached to an FBO.
    const TS: gles.GLsizei = 32;
    var msrb: gles.GLuint = 0;
    glGenRenderbuffers(1, @ptrCast(&msrb));
    glBindRenderbuffer(gles.GL_RENDERBUFFER, msrb);
    glRenderbufferStorageMultisample(gles.GL_RENDERBUFFER, 4, gles.GL_RGBA8, TS, TS);
    try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());
    var msfbo: gles.GLuint = 0;
    glGenFramebuffers(1, @ptrCast(&msfbo));
    glBindFramebuffer(gles.GL_FRAMEBUFFER, msfbo);
    glFramebufferRenderbuffer(gles.GL_FRAMEBUFFER, gles.GL_COLOR_ATTACHMENT0, gles.GL_RENDERBUFFER, msrb);
    try std.testing.expectEqual(gles.GL_FRAMEBUFFER_COMPLETE, glCheckFramebufferStatus(gles.GL_FRAMEBUFFER));

    // A triangle with a slanted edge (the diagonal y = -x): covers the lower-left half.
    const tri = [_]f32{ -1, -1, 1, -1, -1, 1 };
    var vbo: gles.GLuint = 0;
    glGenBuffers(1, @ptrCast(&vbo));
    glBindBuffer(gles.GL_ARRAY_BUFFER, vbo);
    glBufferData(gles.GL_ARRAY_BUFFER, @sizeOf(@TypeOf(tri)), &tri, gles.GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, gles.GL_FLOAT, gles.GL_FALSE, 0, @ptrFromInt(0));
    glEnableVertexAttribArray(0);
    glViewport(0, 0, TS, TS);
    glClearColor(0, 0, 0, 1);
    glClear(gles.GL_COLOR_BUFFER_BIT);
    glDrawArrays(gles.GL_TRIANGLES, 0, 3);
    try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());

    // Resolve the MSAA FBO into the default framebuffer via glBlitFramebuffer (1:1, NEAREST).
    glBindFramebuffer(gles.GL_READ_FRAMEBUFFER, msfbo);
    glBindFramebuffer(gles.GL_DRAW_FRAMEBUFFER, 0);
    glClearColor(0, 0, 0, 1);
    glClear(gles.GL_COLOR_BUFFER_BIT);
    // Blit-resolve the whole MSAA FBO up to the full default framebuffer (32 -> 64, NEAREST).
    glBlitFramebuffer(0, 0, TS, TS, 0, 0, W, W, gles.GL_COLOR_BUFFER_BIT, gles.GL_NEAREST);
    try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());

    // The MSAA FBO rendered a covered region (white) with an anti-aliased edge (partial pixels a
    // 1x render can't produce). Scan the whole resolved framebuffer.
    const s: *state.Surface = @ptrCast(@alignCast(surf.?));
    const dev = s.display.device.?;
    const pxs = try flushMap(dev, s);
    var white: usize = 0;
    var partial: usize = 0;
    var i: usize = 0;
    while (i < 64 * 64) : (i += 1) {
        const r = pxs[i * 4 + 0];
        if (r > 200) white += 1;
        if (r > 30 and r < 225) partial += 1;
    }
    // The covered region resolves to white. The 4x-sampled diagonal edge produces
    // partial-coverage pixels a 1x render cannot (this is the anti-aliasing proof).
    try std.testing.expect(white > 500); // ~half the triangle is fully covered
    try std.testing.expect(partial > 6); // the resolved 4x edge has partial-coverage pixels

    glDeleteFramebuffers(1, &[_]gles.GLuint{msfbo});
    glDeleteRenderbuffers(1, &[_]gles.GLuint{msrb});
    glDeleteBuffers(1, &[_]gles.GLuint{vbo});
    glDeleteProgram(prog);
    glDeleteShader(vs);
    glDeleteShader(fs);
}

test "FBO render-to-texture: a triangle drawn into a texture-backed FBO is non-black when read back" {
    _ = eglBindAPI(egl.EGL_OPENGL_ES_API);
    const dpy = eglGetPlatformDisplay(egl.EGL_PLATFORM_SURFACELESS_MESA, @ptrFromInt(0xE63F0), null);
    try std.testing.expectEqual(egl.EGL_TRUE, eglInitialize(dpy, null, null));
    defer _ = eglTerminate(dpy);
    const cfg = state.configHandle(0);
    const ctx = eglCreateContext(dpy, cfg, egl.EGL_NO_CONTEXT, null);
    const W: EGLint = 64;
    const pb_attribs = [_]EGLint{ egl.EGL_WIDTH, W, egl.EGL_HEIGHT, W, egl.EGL_NONE };
    const surf = eglCreatePbufferSurface(dpy, cfg, &pb_attribs);
    try std.testing.expectEqual(egl.EGL_TRUE, eglMakeCurrent(dpy, surf, surf, ctx));
    defer _ = eglDestroyContext(dpy, ctx);
    defer _ = eglDestroySurface(dpy, surf);
    defer _ = eglMakeCurrent(dpy, null, null, egl.EGL_NO_CONTEXT);

    const vs = glCreateShader(gles.GL_VERTEX_SHADER);
    glShaderSource(vs, 1, &[_]?[*:0]const gles.GLchar{tri_vert_glsl}, null);
    glCompileShader(vs);
    const fs = glCreateShader(gles.GL_FRAGMENT_SHADER);
    glShaderSource(fs, 1, &[_]?[*:0]const gles.GLchar{tri_frag_glsl}, null);
    glCompileShader(fs);
    const prog = glCreateProgram();
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glLinkProgram(prog);
    glUseProgram(prog);

    const Vtx = extern struct { x: f32, y: f32, r: f32, g: f32, b: f32 };
    const tri = [3]Vtx{
        .{ .x = 0.0, .y = 0.8, .r = 1, .g = 0, .b = 0 },
        .{ .x = -0.8, .y = -0.8, .r = 0, .g = 1, .b = 0 },
        .{ .x = 0.8, .y = -0.8, .r = 0, .g = 0, .b = 1 },
    };
    var vbo: gles.GLuint = 0;
    glGenBuffers(1, @ptrCast(&vbo));
    glBindBuffer(gles.GL_ARRAY_BUFFER, vbo);
    glBufferData(gles.GL_ARRAY_BUFFER, @sizeOf(@TypeOf(tri)), &tri, gles.GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, gles.GL_FLOAT, gles.GL_FALSE, @sizeOf(Vtx), @ptrFromInt(0));
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, gles.GL_FLOAT, gles.GL_FALSE, @sizeOf(Vtx), @ptrFromInt(8));
    glEnableVertexAttribArray(1);

    const TS: gles.GLsizei = 32;
    var tex: gles.GLuint = 0;
    glGenTextures(1, @ptrCast(&tex));
    glBindTexture(gles.GL_TEXTURE_2D, tex);
    glTexImage2D(gles.GL_TEXTURE_2D, 0, gles.GL_RGBA, TS, TS, 0, gles.GL_RGBA, gles.GL_UNSIGNED_BYTE, null);
    var fbo: gles.GLuint = 0;
    glGenFramebuffers(1, @ptrCast(&fbo));
    glBindFramebuffer(gles.GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(gles.GL_FRAMEBUFFER, gles.GL_COLOR_ATTACHMENT0, gles.GL_TEXTURE_2D, tex, 0);

    glViewport(0, 0, TS, TS);
    glClearColor(0, 0, 0, 1);
    glClear(gles.GL_COLOR_BUFFER_BIT);
    glDrawArrays(gles.GL_TRIANGLES, 0, 3);
    try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());

    var buf: [32 * 32 * 4]u8 = undefined;
    glReadPixels(0, 0, TS, TS, gles.GL_RGBA, gles.GL_UNSIGNED_BYTE, &buf);
    const ci = (@as(usize, 16) * 32 + 16) * 4;
    try std.testing.expect(buf[ci + 0] > 0 or buf[ci + 1] > 0 or buf[ci + 2] > 0); // FBO got the triangle
}

test "MRT: a fragment shader writes GL_COLOR_ATTACHMENT0 + 1 independently (deferred G-buffer path)" {
    _ = eglBindAPI(egl.EGL_OPENGL_ES_API);
    const dpy = eglGetPlatformDisplay(egl.EGL_PLATFORM_SURFACELESS_MESA, @ptrFromInt(0xE6F01), null);
    try std.testing.expectEqual(egl.EGL_TRUE, eglInitialize(dpy, null, null));
    defer _ = eglTerminate(dpy);
    const cfg = state.configHandle(0);
    const ctx = eglCreateContext(dpy, cfg, egl.EGL_NO_CONTEXT, null);
    const W: EGLint = 64;
    const pb_attribs = [_]EGLint{ egl.EGL_WIDTH, W, egl.EGL_HEIGHT, W, egl.EGL_NONE };
    const surf = eglCreatePbufferSurface(dpy, cfg, &pb_attribs);
    try std.testing.expectEqual(egl.EGL_TRUE, eglMakeCurrent(dpy, surf, surf, ctx));
    defer _ = eglDestroyContext(dpy, ctx);
    defer _ = eglDestroySurface(dpy, surf);
    defer _ = eglMakeCurrent(dpy, null, null, egl.EGL_NO_CONTEXT);

    // GL_MAX_COLOR_ATTACHMENTS is advertised (>= 2 so MRT is usable).
    var max_ca: gles.GLint = 0;
    glGetIntegerv(gles.GL_MAX_COLOR_ATTACHMENTS, @ptrCast(&max_ca));
    try std.testing.expect(max_ca >= 2);

    // A GLES3 fragment shader with two located outputs: c0 -> attachment 0 (red),
    // c1 -> attachment 1 (green). The full-screen triangle covers the whole target.
    const vs_src: [*:0]const gles.GLchar =
        \\#version 300 es
        \\in vec2 position;
        \\void main() { gl_Position = vec4(position, 0.0, 1.0); }
    ;
    const fs_src: [*:0]const gles.GLchar =
        \\#version 300 es
        \\precision mediump float;
        \\layout(location = 0) out vec4 c0;
        \\layout(location = 1) out vec4 c1;
        \\void main() { c0 = vec4(1.0, 0.0, 0.0, 1.0); c1 = vec4(0.0, 1.0, 0.0, 1.0); }
    ;
    const vs = glCreateShader(gles.GL_VERTEX_SHADER);
    glShaderSource(vs, 1, &[_]?[*:0]const gles.GLchar{vs_src}, null);
    glCompileShader(vs);
    var vs_ok: gles.GLint = 0;
    glGetShaderiv(vs, gles.GL_COMPILE_STATUS, @ptrCast(&vs_ok));
    try std.testing.expect(vs_ok == gles.GL_TRUE);
    const fs = glCreateShader(gles.GL_FRAGMENT_SHADER);
    glShaderSource(fs, 1, &[_]?[*:0]const gles.GLchar{fs_src}, null);
    glCompileShader(fs);
    var fs_ok: gles.GLint = 0;
    glGetShaderiv(fs, gles.GL_COMPILE_STATUS, @ptrCast(&fs_ok));
    try std.testing.expect(fs_ok == gles.GL_TRUE);
    const prog = glCreateProgram();
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glLinkProgram(prog);
    glUseProgram(prog);

    const tri = [_]f32{ -1.0, -1.0, 3.0, -1.0, -1.0, 3.0 }; // covers the whole viewport
    var vbo: gles.GLuint = 0;
    glGenBuffers(1, @ptrCast(&vbo));
    glBindBuffer(gles.GL_ARRAY_BUFFER, vbo);
    glBufferData(gles.GL_ARRAY_BUFFER, @sizeOf(@TypeOf(tri)), &tri, gles.GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, gles.GL_FLOAT, gles.GL_FALSE, 8, @ptrFromInt(0));
    glEnableVertexAttribArray(0);

    const TS: gles.GLsizei = 32;
    var tex: [2]gles.GLuint = .{ 0, 0 };
    glGenTextures(2, &tex);
    for (tex) |t| {
        glBindTexture(gles.GL_TEXTURE_2D, t);
        glTexImage2D(gles.GL_TEXTURE_2D, 0, gles.GL_RGBA, TS, TS, 0, gles.GL_RGBA, gles.GL_UNSIGNED_BYTE, null);
    }
    var fbo: gles.GLuint = 0;
    glGenFramebuffers(1, @ptrCast(&fbo));
    glBindFramebuffer(gles.GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(gles.GL_FRAMEBUFFER, gles.GL_COLOR_ATTACHMENT0, gles.GL_TEXTURE_2D, tex[0], 0);
    glFramebufferTexture2D(gles.GL_FRAMEBUFFER, gles.GL_COLOR_ATTACHMENT0 + 1, gles.GL_TEXTURE_2D, tex[1], 0);
    var bufs = [_]gles.GLenum{ gles.GL_COLOR_ATTACHMENT0, gles.GL_COLOR_ATTACHMENT0 + 1 };
    glDrawBuffers(2, &bufs);

    glViewport(0, 0, TS, TS);
    glClearColor(0, 0, 0, 1);
    glClear(gles.GL_COLOR_BUFFER_BIT);
    glDrawArrays(gles.GL_TRIANGLES, 0, 3);
    try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());

    // Attachment 0 (bound FBO reads attachment 0) must be red.
    var buf0: [32 * 32 * 4]u8 = undefined;
    glReadPixels(0, 0, TS, TS, gles.GL_RGBA, gles.GL_UNSIGNED_BYTE, &buf0);
    const ci = (@as(usize, 16) * 32 + 16) * 4;
    try std.testing.expect(buf0[ci + 0] > 200 and buf0[ci + 1] < 60 and buf0[ci + 2] < 60); // red

    // Attachment 1 (tex[1]): read it back via a second FBO that binds tex[1] as attachment 0. GREEN.
    var fbo2: gles.GLuint = 0;
    glGenFramebuffers(1, @ptrCast(&fbo2));
    glBindFramebuffer(gles.GL_FRAMEBUFFER, fbo2);
    glFramebufferTexture2D(gles.GL_FRAMEBUFFER, gles.GL_COLOR_ATTACHMENT0, gles.GL_TEXTURE_2D, tex[1], 0);
    var buf1: [32 * 32 * 4]u8 = undefined;
    glReadPixels(0, 0, TS, TS, gles.GL_RGBA, gles.GL_UNSIGNED_BYTE, &buf1);
    try std.testing.expect(buf1[ci + 0] < 60 and buf1[ci + 1] > 200 and buf1[ci + 2] < 60); // green
}

test "MRT: glReadBuffer selects an attachment for glReadPixels + glClearBufferfv clears one attachment" {
    _ = eglBindAPI(egl.EGL_OPENGL_ES_API);
    const dpy = eglGetPlatformDisplay(egl.EGL_PLATFORM_SURFACELESS_MESA, @ptrFromInt(0xE6F02), null);
    try std.testing.expectEqual(egl.EGL_TRUE, eglInitialize(dpy, null, null));
    defer _ = eglTerminate(dpy);
    const cfg = state.configHandle(0);
    const ctx = eglCreateContext(dpy, cfg, egl.EGL_NO_CONTEXT, null);
    const W: EGLint = 64;
    const pb_attribs = [_]EGLint{ egl.EGL_WIDTH, W, egl.EGL_HEIGHT, W, egl.EGL_NONE };
    const surf = eglCreatePbufferSurface(dpy, cfg, &pb_attribs);
    try std.testing.expectEqual(egl.EGL_TRUE, eglMakeCurrent(dpy, surf, surf, ctx));
    defer _ = eglDestroyContext(dpy, ctx);
    defer _ = eglDestroySurface(dpy, surf);
    defer _ = eglMakeCurrent(dpy, null, null, egl.EGL_NO_CONTEXT);

    const vs_src: [*:0]const gles.GLchar =
        \\#version 300 es
        \\in vec2 position;
        \\void main() { gl_Position = vec4(position, 0.0, 1.0); }
    ;
    const fs_src: [*:0]const gles.GLchar =
        \\#version 300 es
        \\precision mediump float;
        \\layout(location = 0) out vec4 c0;
        \\layout(location = 1) out vec4 c1;
        \\void main() { c0 = vec4(1.0, 0.0, 0.0, 1.0); c1 = vec4(0.0, 1.0, 0.0, 1.0); }
    ;
    const vs = glCreateShader(gles.GL_VERTEX_SHADER);
    glShaderSource(vs, 1, &[_]?[*:0]const gles.GLchar{vs_src}, null);
    glCompileShader(vs);
    const fs = glCreateShader(gles.GL_FRAGMENT_SHADER);
    glShaderSource(fs, 1, &[_]?[*:0]const gles.GLchar{fs_src}, null);
    glCompileShader(fs);
    const prog = glCreateProgram();
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glLinkProgram(prog);
    glUseProgram(prog);

    const tri = [_]f32{ -1.0, -1.0, 3.0, -1.0, -1.0, 3.0 };
    var vbo: gles.GLuint = 0;
    glGenBuffers(1, @ptrCast(&vbo));
    glBindBuffer(gles.GL_ARRAY_BUFFER, vbo);
    glBufferData(gles.GL_ARRAY_BUFFER, @sizeOf(@TypeOf(tri)), &tri, gles.GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, gles.GL_FLOAT, gles.GL_FALSE, 8, @ptrFromInt(0));
    glEnableVertexAttribArray(0);

    const TS: gles.GLsizei = 32;
    var tex: [2]gles.GLuint = .{ 0, 0 };
    glGenTextures(2, &tex);
    for (tex) |t| {
        glBindTexture(gles.GL_TEXTURE_2D, t);
        glTexImage2D(gles.GL_TEXTURE_2D, 0, gles.GL_RGBA, TS, TS, 0, gles.GL_RGBA, gles.GL_UNSIGNED_BYTE, null);
    }
    var fbo: gles.GLuint = 0;
    glGenFramebuffers(1, @ptrCast(&fbo));
    glBindFramebuffer(gles.GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(gles.GL_FRAMEBUFFER, gles.GL_COLOR_ATTACHMENT0, gles.GL_TEXTURE_2D, tex[0], 0);
    glFramebufferTexture2D(gles.GL_FRAMEBUFFER, gles.GL_COLOR_ATTACHMENT0 + 1, gles.GL_TEXTURE_2D, tex[1], 0);
    var bufs = [_]gles.GLenum{ gles.GL_COLOR_ATTACHMENT0, gles.GL_COLOR_ATTACHMENT0 + 1 };
    glDrawBuffers(2, &bufs);
    glViewport(0, 0, TS, TS);
    glDrawArrays(gles.GL_TRIANGLES, 0, 3);

    const ci = (@as(usize, 16) * 32 + 16) * 4;
    var buf: [32 * 32 * 4]u8 = undefined;

    // glReadBuffer(GL_COLOR_ATTACHMENT1): glReadPixels now reads attachment 1 (GREEN) without a
    // second FBO. Attachment 0 (default read) is RED.
    glReadBuffer(gles.GL_COLOR_ATTACHMENT0 + 1);
    glReadPixels(0, 0, TS, TS, gles.GL_RGBA, gles.GL_UNSIGNED_BYTE, &buf);
    try std.testing.expect(buf[ci + 0] < 60 and buf[ci + 1] > 200 and buf[ci + 2] < 60); // green

    // glClearBufferfv(GL_COLOR, 1, blue): clear ONLY attachment 1 to blue. Attachment 0 stays red.
    const blue = [_]gles.GLfloat{ 0, 0, 1, 1 };
    glClearBufferfv(gles.GL_COLOR, 1, &blue);
    try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());

    glReadBuffer(gles.GL_COLOR_ATTACHMENT0 + 1);
    glReadPixels(0, 0, TS, TS, gles.GL_RGBA, gles.GL_UNSIGNED_BYTE, &buf);
    try std.testing.expect(buf[ci + 0] < 60 and buf[ci + 1] < 60 and buf[ci + 2] > 200); // now blue

    // Attachment 0 was untouched by the attachment-1 clear: still red.
    glReadBuffer(gles.GL_COLOR_ATTACHMENT0);
    glReadPixels(0, 0, TS, TS, gles.GL_RGBA, gles.GL_UNSIGNED_BYTE, &buf);
    try std.testing.expect(buf[ci + 0] > 200 and buf[ci + 1] < 60 and buf[ci + 2] < 60); // red
}

test "gl_VertexID: a vertex-buffer-less full-screen triangle (procedural VS) covers the screen" {
    _ = eglBindAPI(egl.EGL_OPENGL_ES_API);
    const dpy = eglGetPlatformDisplay(egl.EGL_PLATFORM_SURFACELESS_MESA, @ptrFromInt(0xE7A01), null);
    try std.testing.expectEqual(egl.EGL_TRUE, eglInitialize(dpy, null, null));
    defer _ = eglTerminate(dpy);
    const cfg = state.configHandle(0);
    const ctx = eglCreateContext(dpy, cfg, egl.EGL_NO_CONTEXT, null);
    const W: EGLint = 64;
    const pb_attribs = [_]EGLint{ egl.EGL_WIDTH, W, egl.EGL_HEIGHT, W, egl.EGL_NONE };
    const surf = eglCreatePbufferSurface(dpy, cfg, &pb_attribs);
    try std.testing.expectEqual(egl.EGL_TRUE, eglMakeCurrent(dpy, surf, surf, ctx));
    defer _ = eglDestroyContext(dpy, ctx);
    defer _ = eglDestroySurface(dpy, surf);
    defer _ = eglMakeCurrent(dpy, null, null, egl.EGL_NO_CONTEXT);

    // The canonical full-screen triangle from gl_VertexID, no vertex buffer:
    //   id 0 -> (-1,-1), id 1 -> (3,-1), id 2 -> (-1,3) -> covers the whole [-1,1] viewport.
    const vs_src: [*:0]const gles.GLchar =
        \\#version 300 es
        \\void main() {
        \\  float x = (gl_VertexID == 1) ? 3.0 : -1.0;
        \\  float y = (gl_VertexID == 2) ? 3.0 : -1.0;
        \\  gl_Position = vec4(x, y, 0.0, 1.0);
        \\}
    ;
    const fs_src: [*:0]const gles.GLchar =
        \\#version 300 es
        \\precision mediump float;
        \\out vec4 frag;
        \\void main() { frag = vec4(0.0, 1.0, 0.0, 1.0); }
    ;
    const vs = glCreateShader(gles.GL_VERTEX_SHADER);
    glShaderSource(vs, 1, &[_]?[*:0]const gles.GLchar{vs_src}, null);
    glCompileShader(vs);
    var vs_ok: gles.GLint = 0;
    glGetShaderiv(vs, gles.GL_COMPILE_STATUS, @ptrCast(&vs_ok));
    try std.testing.expect(vs_ok == gles.GL_TRUE);
    const fs = glCreateShader(gles.GL_FRAGMENT_SHADER);
    glShaderSource(fs, 1, &[_]?[*:0]const gles.GLchar{fs_src}, null);
    glCompileShader(fs);
    const prog = glCreateProgram();
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glLinkProgram(prog);
    glUseProgram(prog);

    glViewport(0, 0, W, W);
    glClearColor(0, 0, 0, 1);
    glClear(gles.GL_COLOR_BUFFER_BIT);
    glDrawArrays(gles.GL_TRIANGLES, 0, 3); // no VBO, no attributes: the VS derives position from gl_VertexID
    try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());

    var buf: [64 * 64 * 4]u8 = undefined;
    glReadPixels(0, 0, W, W, gles.GL_RGBA, gles.GL_UNSIGNED_BYTE, &buf);
    const ci = (@as(usize, 32) * 64 + 32) * 4;
    try std.testing.expect(buf[ci + 0] < 60 and buf[ci + 1] > 200 and buf[ci + 2] < 60); // center is green
}

test "vertex texture fetch: a VS texture2D reads the bound texel and drives the color (software)" {
    try runVertexTextureFetch(0x71F409);
}

test "vertex texture fetch on NVIDIA GPU: a VS samples a texture on the real RTX (skips without a GPU)" {
    const saved = state.pinned_driver;
    state.pinned_driver = "nvidia";
    defer state.pinned_driver = saved;
    try runVertexTextureFetch(0x71F40A);
}

/// Vertex texture fetch (VTF): the VS calls `texture2D` (a vertex shader has no derivatives, so it
/// lowers to an explicit LOD-0 sample) and passes the red channel to the FS as a varying. All three
/// verts sample the same texel, so the interpolated value is constant and the whole triangle carries
/// it. The center pixel's red equals the bound texel's red (~200). A broken VTF would read 0/garbage.
/// This is the terrain-heightmap idiom (sample a texture in the VS) reduced to a color read-back.
fn runVertexTextureFetch(dpy_magic: usize) !void {
    _ = eglBindAPI(egl.EGL_OPENGL_ES_API);
    const dpy = eglGetPlatformDisplay(egl.EGL_PLATFORM_SURFACELESS_MESA, @ptrFromInt(dpy_magic), null);
    if (eglInitialize(dpy, null, null) != egl.EGL_TRUE) return error.SkipZigTest;
    defer _ = eglTerminate(dpy);
    const cfg = state.configHandle(0);
    const ctx = eglCreateContext(dpy, cfg, egl.EGL_NO_CONTEXT, null);
    const W: EGLint = 64;
    const pb = [_]EGLint{ egl.EGL_WIDTH, W, egl.EGL_HEIGHT, W, egl.EGL_NONE };
    const surf = eglCreatePbufferSurface(dpy, cfg, &pb);
    _ = eglMakeCurrent(dpy, surf, surf, ctx);
    defer _ = eglDestroyContext(dpy, ctx);
    defer _ = eglDestroySurface(dpy, surf);
    defer _ = eglMakeCurrent(dpy, null, null, egl.EGL_NO_CONTEXT);

    const vs: [*:0]const gles.GLchar =
        \\attribute vec2 p;
        \\uniform sampler2D uTex;
        \\varying float vVal;
        \\void main() {
        \\  vVal = texture2D(uTex, vec2(0.5, 0.5)).x;
        \\  gl_Position = vec4(p, 0.0, 1.0);
        \\}
    ;
    const fs: [*:0]const gles.GLchar = "precision mediump float; varying float vVal; void main(){ gl_FragColor = vec4(vVal, 0.0, 0.0, 1.0); }";
    const v = glCreateShader(gles.GL_VERTEX_SHADER);
    glShaderSource(v, 1, &[_]?[*:0]const gles.GLchar{vs}, null);
    glCompileShader(v);
    var vok: gles.GLint = 0;
    glGetShaderiv(v, gles.GL_COMPILE_STATUS, @ptrCast(&vok));
    try std.testing.expect(vok == gles.GL_TRUE);
    const f = glCreateShader(gles.GL_FRAGMENT_SHADER);
    glShaderSource(f, 1, &[_]?[*:0]const gles.GLchar{fs}, null);
    glCompileShader(f);
    const prog = glCreateProgram();
    glAttachShader(prog, v);
    glAttachShader(prog, f);
    glBindAttribLocation(prog, 0, "p");
    glLinkProgram(prog);
    glUseProgram(prog);

    // A 2x2 texture, all texels red = 200 (uv 0.5 lands between texels; NEAREST -> one of them, all equal).
    var tex: gles.GLuint = 0;
    glGenTextures(1, @ptrCast(&tex));
    glBindTexture(gles.GL_TEXTURE_2D, tex);
    const px = [_]u8{ 200, 0, 0, 255 } ** 4;
    glTexImage2D(gles.GL_TEXTURE_2D, 0, gles.GL_RGBA, 2, 2, 0, gles.GL_RGBA, gles.GL_UNSIGNED_BYTE, &px);
    glTexParameteri(gles.GL_TEXTURE_2D, gles.GL_TEXTURE_MIN_FILTER, gles.GL_NEAREST);
    glTexParameteri(gles.GL_TEXTURE_2D, gles.GL_TEXTURE_MAG_FILTER, gles.GL_NEAREST);
    glActiveTexture(gles.GL_TEXTURE0);
    const loc = glGetUniformLocation(prog, "uTex");
    glUniform1i(loc, 0);

    const tri = [_]f32{ -1, -1, 3, -1, -1, 3 }; // full-screen triangle
    var vbo: gles.GLuint = 0;
    glGenBuffers(1, @ptrCast(&vbo));
    glBindBuffer(gles.GL_ARRAY_BUFFER, vbo);
    glBufferData(gles.GL_ARRAY_BUFFER, @sizeOf(@TypeOf(tri)), &tri, gles.GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, gles.GL_FLOAT, gles.GL_FALSE, 0, @ptrFromInt(0));
    glEnableVertexAttribArray(0);

    glViewport(0, 0, W, W);
    glClearColor(0, 0, 0, 1);
    glClear(gles.GL_COLOR_BUFFER_BIT);
    glDrawArrays(gles.GL_TRIANGLES, 0, 3);
    try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());

    const s: *state.Surface = @ptrCast(@alignCast(surf.?));
    const dev = s.display.device.?;
    const rd = try flushMap(dev, s);
    const ci = (@as(usize, 32) * 64 + 32) * 4;
    // The VS read the texel (red ~200) and it flowed through the varying to the center pixel's red.
    try std.testing.expect(rd[ci + 0] > 185 and rd[ci + 0] < 215);
    try std.testing.expect(rd[ci + 1] < 40 and rd[ci + 2] < 40);
    glDeleteProgram(prog);
    glDeleteTextures(1, &[_]gles.GLuint{tex});
    glDeleteBuffers(1, &[_]gles.GLuint{vbo});
}

test "MRT: glClear(GL_COLOR_BUFFER_BIT) clears EVERY bound draw buffer, not just attachment 0" {
    _ = eglBindAPI(egl.EGL_OPENGL_ES_API);
    const dpy = eglGetPlatformDisplay(egl.EGL_PLATFORM_SURFACELESS_MESA, @ptrFromInt(0xE6F03), null);
    try std.testing.expectEqual(egl.EGL_TRUE, eglInitialize(dpy, null, null));
    defer _ = eglTerminate(dpy);
    const cfg = state.configHandle(0);
    const ctx = eglCreateContext(dpy, cfg, egl.EGL_NO_CONTEXT, null);
    const W: EGLint = 64;
    const pb_attribs = [_]EGLint{ egl.EGL_WIDTH, W, egl.EGL_HEIGHT, W, egl.EGL_NONE };
    const surf = eglCreatePbufferSurface(dpy, cfg, &pb_attribs);
    try std.testing.expectEqual(egl.EGL_TRUE, eglMakeCurrent(dpy, surf, surf, ctx));
    defer _ = eglDestroyContext(dpy, ctx);
    defer _ = eglDestroySurface(dpy, surf);
    defer _ = eglMakeCurrent(dpy, null, null, egl.EGL_NO_CONTEXT);

    const TS: gles.GLsizei = 32;
    var tex: [3]gles.GLuint = .{ 0, 0, 0 };
    glGenTextures(3, &tex);
    for (tex) |t| {
        glBindTexture(gles.GL_TEXTURE_2D, t);
        glTexImage2D(gles.GL_TEXTURE_2D, 0, gles.GL_RGBA, TS, TS, 0, gles.GL_RGBA, gles.GL_UNSIGNED_BYTE, null);
    }
    var fbo: gles.GLuint = 0;
    glGenFramebuffers(1, @ptrCast(&fbo));
    glBindFramebuffer(gles.GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(gles.GL_FRAMEBUFFER, gles.GL_COLOR_ATTACHMENT0, gles.GL_TEXTURE_2D, tex[0], 0);
    glFramebufferTexture2D(gles.GL_FRAMEBUFFER, gles.GL_COLOR_ATTACHMENT0 + 1, gles.GL_TEXTURE_2D, tex[1], 0);
    glFramebufferTexture2D(gles.GL_FRAMEBUFFER, gles.GL_COLOR_ATTACHMENT0 + 2, gles.GL_TEXTURE_2D, tex[2], 0);
    var bufs = [_]gles.GLenum{ gles.GL_COLOR_ATTACHMENT0, gles.GL_COLOR_ATTACHMENT0 + 1, gles.GL_COLOR_ATTACHMENT0 + 2 };
    glDrawBuffers(3, &bufs);

    glViewport(0, 0, TS, TS);
    // A single glClear must clear all three bound attachments to the clear color (orange-ish).
    glClearColor(1.0, 0.5, 0.0, 1.0);
    glClear(gles.GL_COLOR_BUFFER_BIT);
    try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());

    const ci = (@as(usize, 16) * 32 + 16) * 4;
    var buf: [32 * 32 * 4]u8 = undefined;
    for (0..3) |att| {
        glReadBuffer(gles.GL_COLOR_ATTACHMENT0 + @as(gles.GLenum, @intCast(att)));
        glReadPixels(0, 0, TS, TS, gles.GL_RGBA, gles.GL_UNSIGNED_BYTE, &buf);
        try std.testing.expect(buf[ci + 0] > 200 and buf[ci + 1] > 100 and buf[ci + 1] < 160 and buf[ci + 2] < 40); // clear color reached this attachment
    }
}

test "EGL depth-bias oracle: glPolygonOffset lets a coplanar draw win the depth test (decal path)" {
    _ = eglBindAPI(egl.EGL_OPENGL_ES_API);
    const dpy = eglGetPlatformDisplay(egl.EGL_PLATFORM_SURFACELESS_MESA, @ptrFromInt(0xE640C), null);
    try std.testing.expect(dpy != null);
    try std.testing.expectEqual(egl.EGL_TRUE, eglInitialize(dpy, null, null));
    defer _ = eglTerminate(dpy);
    const cfg = state.configHandle(3); // RGBA8 + D24S8 (has a depth buffer)
    const ctx = eglCreateContext(dpy, cfg, egl.EGL_NO_CONTEXT, null);
    const W: EGLint = 64;
    const pb_attribs = [_]EGLint{ egl.EGL_WIDTH, W, egl.EGL_HEIGHT, W, egl.EGL_NONE };
    const surf = eglCreatePbufferSurface(dpy, cfg, &pb_attribs);
    try std.testing.expectEqual(egl.EGL_TRUE, eglMakeCurrent(dpy, surf, surf, ctx));
    defer _ = eglDestroyContext(dpy, ctx);
    defer _ = eglDestroySurface(dpy, surf);
    defer _ = eglMakeCurrent(dpy, null, null, egl.EGL_NO_CONTEXT);

    const vs_src: [*:0]const gles.GLchar =
        \\attribute vec3 position;
        \\uniform vec4 uColor;
        \\varying vec4 vColor;
        \\void main() { gl_Position = vec4(position, 1.0); vColor = uColor; }
    ;
    const fs_src: [*:0]const gles.GLchar =
        \\precision mediump float;
        \\varying vec4 vColor;
        \\void main() { gl_FragColor = vColor; }
    ;
    const vs = glCreateShader(gles.GL_VERTEX_SHADER);
    glShaderSource(vs, 1, &[_]?[*:0]const gles.GLchar{vs_src}, null);
    glCompileShader(vs);
    const fs = glCreateShader(gles.GL_FRAGMENT_SHADER);
    glShaderSource(fs, 1, &[_]?[*:0]const gles.GLchar{fs_src}, null);
    glCompileShader(fs);
    const prog = glCreateProgram();
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glBindAttribLocation(prog, 0, "position");
    glLinkProgram(prog);
    glUseProgram(prog);
    const col_loc = glGetUniformLocation(prog, "uColor");

    const Vtx = extern struct { x: f32, y: f32, z: f32 };
    const full = [4]Vtx{
        .{ .x = -1, .y = -1, .z = 0.5 }, .{ .x = 1, .y = -1, .z = 0.5 },
        .{ .x = -1, .y = 1, .z = 0.5 },  .{ .x = 1, .y = 1, .z = 0.5 },
    };
    var vbo: gles.GLuint = 0;
    glGenBuffers(1, @ptrCast(&vbo));
    glBindBuffer(gles.GL_ARRAY_BUFFER, vbo);
    glBufferData(gles.GL_ARRAY_BUFFER, @sizeOf(@TypeOf(full)), &full, gles.GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, gles.GL_FLOAT, gles.GL_FALSE, @sizeOf(Vtx), @ptrFromInt(0));
    glEnableVertexAttribArray(0);

    glClearColor(0, 0, 0, 1);
    glClearDepthf(1.0);
    glClear(gles.GL_COLOR_BUFFER_BIT | gles.GL_DEPTH_BUFFER_BIT);
    glEnable(gles.GL_DEPTH_TEST);
    glDepthFunc(gles.GL_LESS);

    // Red coplanar quad at z=0.5 (writes depth 0.5).
    glUniform4f(col_loc, 1, 0, 0, 1);
    glDrawArrays(gles.GL_TRIANGLE_STRIP, 0, 4);
    // Green coplanar quad at z=0.5 with a negative polygon offset: pulled in front -> passes LESS.
    // Without the offset it would fail (0.5 < 0.5 false) and the center would stay red.
    glEnable(gles.GL_POLYGON_OFFSET_FILL);
    glPolygonOffset(0, -20000);
    glUniform4f(col_loc, 0, 1, 0, 1);
    glDrawArrays(gles.GL_TRIANGLE_STRIP, 0, 4);
    try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());

    const s: *state.Surface = @ptrCast(@alignCast(surf.?));
    const dev = s.display.device.?;
    const px = try flushMap(dev, s);
    const ci = (32 * 64 + 32) * 4;
    try std.testing.expect(px[ci + 1] > 200 and px[ci + 0] < 60); // green won via polygon offset

    glDeleteBuffers(1, &[_]gles.GLuint{vbo});
    glDeleteProgram(prog);
    glDeleteShader(vs);
    glDeleteShader(fs);
}

test "EGL mipmap oracle: a minified mipmapped texture samples an averaged mip level (implicit LOD)" {
    _ = eglBindAPI(egl.EGL_OPENGL_ES_API);
    const dpy = eglGetPlatformDisplay(egl.EGL_PLATFORM_SURFACELESS_MESA, @ptrFromInt(0xE6413), null);
    try std.testing.expect(dpy != null);
    try std.testing.expectEqual(egl.EGL_TRUE, eglInitialize(dpy, null, null));
    defer _ = eglTerminate(dpy);
    const cfg = state.configHandle(0); // RGBA8
    const ctx = eglCreateContext(dpy, cfg, egl.EGL_NO_CONTEXT, null);
    const W: EGLint = 64;
    const pb_attribs = [_]EGLint{ egl.EGL_WIDTH, W, egl.EGL_HEIGHT, W, egl.EGL_NONE };
    const surf = eglCreatePbufferSurface(dpy, cfg, &pb_attribs);
    try std.testing.expectEqual(egl.EGL_TRUE, eglMakeCurrent(dpy, surf, surf, ctx));
    defer _ = eglDestroyContext(dpy, ctx);
    defer _ = eglDestroySurface(dpy, surf);
    defer _ = eglMakeCurrent(dpy, null, null, egl.EGL_NO_CONTEXT);

    const vs_src: [*:0]const gles.GLchar =
        \\attribute vec2 position;
        \\attribute vec2 uv;
        \\varying vec2 vUv;
        \\void main() { gl_Position = vec4(position, 0.0, 1.0); vUv = uv; }
    ;
    const fs_src: [*:0]const gles.GLchar =
        \\precision mediump float;
        \\uniform sampler2D tex;
        \\varying vec2 vUv;
        \\void main() { gl_FragColor = texture2D(tex, vUv); }
    ;
    const vs = glCreateShader(gles.GL_VERTEX_SHADER);
    glShaderSource(vs, 1, &[_]?[*:0]const gles.GLchar{vs_src}, null);
    glCompileShader(vs);
    const fs = glCreateShader(gles.GL_FRAGMENT_SHADER);
    glShaderSource(fs, 1, &[_]?[*:0]const gles.GLchar{fs_src}, null);
    glCompileShader(fs);
    const prog = glCreateProgram();
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glBindAttribLocation(prog, 0, "position");
    glBindAttribLocation(prog, 1, "uv");
    glLinkProgram(prog);
    glUseProgram(prog);

    // An 8x8 texture of 1px-wide red/blue vertical stripes. Box-downsampling averages adjacent
    // stripes, so mip level 1+ is uniformly purple (~128,0,128). The base is pure red or blue.
    const TS: gles.GLsizei = 8;
    var texels: [8 * 8 * 4]u8 = undefined;
    var yy: usize = 0;
    while (yy < 8) : (yy += 1) {
        var xx: usize = 0;
        while (xx < 8) : (xx += 1) {
            const i = (yy * 8 + xx) * 4;
            const red = (xx % 2) == 0;
            texels[i + 0] = if (red) 255 else 0;
            texels[i + 1] = 0;
            texels[i + 2] = if (red) 0 else 255;
            texels[i + 3] = 255;
        }
    }
    var tex: gles.GLuint = 0;
    glGenTextures(1, @ptrCast(&tex));
    glBindTexture(gles.GL_TEXTURE_2D, tex);
    glTexImage2D(gles.GL_TEXTURE_2D, 0, gles.GL_RGBA, TS, TS, 0, gles.GL_RGBA, gles.GL_UNSIGNED_BYTE, &texels);
    glTexParameteri(gles.GL_TEXTURE_2D, gles.GL_TEXTURE_MIN_FILTER, gles.GL_NEAREST_MIPMAP_NEAREST);
    glTexParameteri(gles.GL_TEXTURE_2D, gles.GL_TEXTURE_MAG_FILTER, gles.GL_NEAREST);
    glGenerateMipmap(gles.GL_TEXTURE_2D);

    // A fullscreen quad whose uv tiles the texture 16x across the 64px pbuffer: du/dpixel = 0.25,
    // so the texel-space derivative is 0.25*8 = 2 texels/pixel -> LOD ~1 -> mip level 1 (PURPLE).
    // Without implicit LOD the sampler would read the base stripes (pure red or blue).
    const Vtx = extern struct { x: f32, y: f32, u: f32, v: f32 };
    const quad = [4]Vtx{
        .{ .x = -1, .y = -1, .u = 0, .v = 0 }, .{ .x = 1, .y = -1, .u = 16, .v = 0 },
        .{ .x = -1, .y = 1, .u = 0, .v = 16 }, .{ .x = 1, .y = 1, .u = 16, .v = 16 },
    };
    var vbo: gles.GLuint = 0;
    glGenBuffers(1, @ptrCast(&vbo));
    glBindBuffer(gles.GL_ARRAY_BUFFER, vbo);
    glBufferData(gles.GL_ARRAY_BUFFER, @sizeOf(@TypeOf(quad)), &quad, gles.GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, gles.GL_FLOAT, gles.GL_FALSE, @sizeOf(Vtx), @ptrFromInt(0));
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 2, gles.GL_FLOAT, gles.GL_FALSE, @sizeOf(Vtx), @ptrFromInt(8));
    glEnableVertexAttribArray(1);

    glClearColor(0, 0, 0, 1);
    glClear(gles.GL_COLOR_BUFFER_BIT);
    glDrawArrays(gles.GL_TRIANGLE_STRIP, 0, 4);
    try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());

    const s: *state.Surface = @ptrCast(@alignCast(surf.?));
    const dev = s.display.device.?;
    const px = try flushMap(dev, s);
    const ci = (32 * 64 + 32) * 4;
    // Purple = both R and B present (the averaged mip), not a pure-red/blue base stripe. Implicit
    // LOD picked a higher level. Without it the center would be R=255,B=0 or R=0,B=255.
    try std.testing.expect(px[ci + 0] > 60 and px[ci + 2] > 60);

    // TRILINEAR (GL_LINEAR_MIPMAP_LINEAR): re-draw with the mip-blend filter to exercise the
    // sampler's mip_filter=linear path (blends the two bracketing levels). The lower levels are
    // uniform purple, so the trilinear blend is still purple. It must not regress to a base stripe.
    glTexParameteri(gles.GL_TEXTURE_2D, gles.GL_TEXTURE_MIN_FILTER, gles.GL_LINEAR_MIPMAP_LINEAR);
    glClear(gles.GL_COLOR_BUFFER_BIT);
    glDrawArrays(gles.GL_TRIANGLE_STRIP, 0, 4);
    try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());
    const px2 = try flushMap(dev, s);
    try std.testing.expect(px2[ci + 0] > 40 and px2[ci + 2] > 40); // still purple (trilinear blend)

    glDeleteTextures(1, &[_]gles.GLuint{tex});
    glDeleteBuffers(1, &[_]gles.GLuint{vbo});
    glDeleteProgram(prog);
    glDeleteShader(vs);
    glDeleteShader(fs);
}

test "EGL textureLod oracle: an explicit LOD selects the mip level directly (not the derivative)" {
    _ = eglBindAPI(egl.EGL_OPENGL_ES_API);
    const dpy = eglGetPlatformDisplay(egl.EGL_PLATFORM_SURFACELESS_MESA, @ptrFromInt(0xE6417), null);
    try std.testing.expect(dpy != null);
    try std.testing.expectEqual(egl.EGL_TRUE, eglInitialize(dpy, null, null));
    defer _ = eglTerminate(dpy);
    const cfg = state.configHandle(0);
    const ctx = eglCreateContext(dpy, cfg, egl.EGL_NO_CONTEXT, null);
    const W: EGLint = 64;
    const pb_attribs = [_]EGLint{ egl.EGL_WIDTH, W, egl.EGL_HEIGHT, W, egl.EGL_NONE };
    const surf = eglCreatePbufferSurface(dpy, cfg, &pb_attribs);
    try std.testing.expectEqual(egl.EGL_TRUE, eglMakeCurrent(dpy, surf, surf, ctx));
    defer _ = eglDestroyContext(dpy, ctx);
    defer _ = eglDestroySurface(dpy, surf);
    defer _ = eglMakeCurrent(dpy, null, null, egl.EGL_NO_CONTEXT);

    const vs_src: [*:0]const gles.GLchar =
        \\attribute vec2 position;
        \\attribute vec2 uv;
        \\varying vec2 vUv;
        \\void main() { gl_Position = vec4(position, 0.0, 1.0); vUv = uv; }
    ;
    // Explicit-LOD sampling: the LOD is a uniform, so the same geometry can read any level.
    const fs_src: [*:0]const gles.GLchar =
        \\precision mediump float;
        \\uniform sampler2D tex;
        \\uniform float uLod;
        \\varying vec2 vUv;
        \\void main() { gl_FragColor = textureLod(tex, vUv, uLod); }
    ;
    const vs = glCreateShader(gles.GL_VERTEX_SHADER);
    glShaderSource(vs, 1, &[_]?[*:0]const gles.GLchar{vs_src}, null);
    glCompileShader(vs);
    const fs = glCreateShader(gles.GL_FRAGMENT_SHADER);
    glShaderSource(fs, 1, &[_]?[*:0]const gles.GLchar{fs_src}, null);
    glCompileShader(fs);
    const prog = glCreateProgram();
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glBindAttribLocation(prog, 0, "position");
    glBindAttribLocation(prog, 1, "uv");
    glLinkProgram(prog);
    glUseProgram(prog);
    defer {
        glDeleteProgram(prog);
        glDeleteShader(vs);
        glDeleteShader(fs);
    }

    // 8x8 red/blue vertical stripes; box-downsampled mips are uniformly purple (top level 1x1).
    const TS: gles.GLsizei = 8;
    var texels: [8 * 8 * 4]u8 = undefined;
    for (0..8) |y| for (0..8) |x| {
        const i = (y * 8 + x) * 4;
        const red = (x % 2) == 0;
        texels[i + 0] = if (red) 255 else 0;
        texels[i + 1] = 0;
        texels[i + 2] = if (red) 0 else 255;
        texels[i + 3] = 255;
    };
    var tex: gles.GLuint = 0;
    glGenTextures(1, @ptrCast(&tex));
    glBindTexture(gles.GL_TEXTURE_2D, tex);
    glTexImage2D(gles.GL_TEXTURE_2D, 0, gles.GL_RGBA, TS, TS, 0, gles.GL_RGBA, gles.GL_UNSIGNED_BYTE, &texels);
    glTexParameteri(gles.GL_TEXTURE_2D, gles.GL_TEXTURE_MIN_FILTER, gles.GL_NEAREST_MIPMAP_NEAREST);
    glTexParameteri(gles.GL_TEXTURE_2D, gles.GL_TEXTURE_MAG_FILTER, gles.GL_NEAREST);
    glGenerateMipmap(gles.GL_TEXTURE_2D);
    glUniform1i(glGetUniformLocation(prog, "tex"), 0);
    const lod_loc = glGetUniformLocation(prog, "uLod");

    // A fullscreen quad with uv 0..1: the derivative LOD is ~0 (no minification), so the mip
    // level is chosen entirely by the explicit LOD uniform.
    const Vtx = extern struct { x: f32, y: f32, u: f32, v: f32 };
    const quad = [4]Vtx{
        .{ .x = -1, .y = -1, .u = 0, .v = 0 }, .{ .x = 1, .y = -1, .u = 1, .v = 0 },
        .{ .x = -1, .y = 1, .u = 0, .v = 1 },  .{ .x = 1, .y = 1, .u = 1, .v = 1 },
    };
    var vbo: gles.GLuint = 0;
    glGenBuffers(1, @ptrCast(&vbo));
    glBindBuffer(gles.GL_ARRAY_BUFFER, vbo);
    glBufferData(gles.GL_ARRAY_BUFFER, @sizeOf(@TypeOf(quad)), &quad, gles.GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, gles.GL_FLOAT, gles.GL_FALSE, @sizeOf(Vtx), @ptrFromInt(0));
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 2, gles.GL_FLOAT, gles.GL_FALSE, @sizeOf(Vtx), @ptrFromInt(8));
    glEnableVertexAttribArray(1);
    defer glDeleteBuffers(1, &[_]gles.GLuint{vbo});
    defer glDeleteTextures(1, &[_]gles.GLuint{tex});

    const s: *state.Surface = @ptrCast(@alignCast(surf.?));
    const dev = s.display.device.?;
    const ci = (32 * 64 + 32) * 4;

    // LOD 0: the base level -> a pure stripe (exactly one of R / B saturated, the other 0).
    glUniform1f(lod_loc, 0.0);
    glClearColor(0, 0, 0, 1);
    glClear(gles.GL_COLOR_BUFFER_BIT);
    glDrawArrays(gles.GL_TRIANGLE_STRIP, 0, 4);
    try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());
    const px0 = try flushMap(dev, s);
    const base_pure = (px0[ci + 0] > 200 and px0[ci + 2] < 60) or (px0[ci + 0] < 60 and px0[ci + 2] > 200);
    try std.testing.expect(base_pure);

    // LOD 3: the top (1x1) level -> the fully averaged purple (both R and B present). This can
    // only come from the explicit LOD (the geometry's implicit LOD is ~0 = the base stripe).
    glUniform1f(lod_loc, 3.0);
    glClear(gles.GL_COLOR_BUFFER_BIT);
    glDrawArrays(gles.GL_TRIANGLE_STRIP, 0, 4);
    try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());
    const px3 = try flushMap(dev, s);
    try std.testing.expect(px3[ci + 0] > 60 and px3[ci + 2] > 60);
}

test "GL_TEXTURE_BASE_LEVEL: sampling never uses a level finer than base_level (mip range control)" {
    _ = eglBindAPI(egl.EGL_OPENGL_ES_API);
    const dpy = eglGetPlatformDisplay(egl.EGL_PLATFORM_SURFACELESS_MESA, @ptrFromInt(0xE7B02), null);
    try std.testing.expectEqual(egl.EGL_TRUE, eglInitialize(dpy, null, null));
    defer _ = eglTerminate(dpy);
    const cfg = state.configHandle(0);
    const ctx = eglCreateContext(dpy, cfg, egl.EGL_NO_CONTEXT, null);
    const W: EGLint = 64;
    const pb_attribs = [_]EGLint{ egl.EGL_WIDTH, W, egl.EGL_HEIGHT, W, egl.EGL_NONE };
    const surf = eglCreatePbufferSurface(dpy, cfg, &pb_attribs);
    try std.testing.expectEqual(egl.EGL_TRUE, eglMakeCurrent(dpy, surf, surf, ctx));
    defer _ = eglDestroyContext(dpy, ctx);
    defer _ = eglDestroySurface(dpy, surf);
    defer _ = eglMakeCurrent(dpy, null, null, egl.EGL_NO_CONTEXT);

    const vs_src: [*:0]const gles.GLchar =
        \\attribute vec2 position;
        \\attribute vec2 uv;
        \\varying vec2 vUv;
        \\void main() { gl_Position = vec4(position, 0.0, 1.0); vUv = uv; }
    ;
    // Sample at explicit LOD 0. With base_level 0 this is the fine (striped) level; with
    // base_level 1 the sampler clamps to level 1 (the box-downsampled purple).
    const fs_src: [*:0]const gles.GLchar =
        \\precision mediump float;
        \\uniform sampler2D tex;
        \\uniform float uLod;
        \\varying vec2 vUv;
        \\void main() { gl_FragColor = textureLod(tex, vUv, uLod); }
    ;
    const vs = glCreateShader(gles.GL_VERTEX_SHADER);
    glShaderSource(vs, 1, &[_]?[*:0]const gles.GLchar{vs_src}, null);
    glCompileShader(vs);
    const fs = glCreateShader(gles.GL_FRAGMENT_SHADER);
    glShaderSource(fs, 1, &[_]?[*:0]const gles.GLchar{fs_src}, null);
    glCompileShader(fs);
    const prog = glCreateProgram();
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glBindAttribLocation(prog, 0, "position");
    glBindAttribLocation(prog, 1, "uv");
    glLinkProgram(prog);
    glUseProgram(prog);

    // 8x8 red/blue vertical stripes; box-downsampled mips are uniformly purple.
    const TS: gles.GLsizei = 8;
    var texels: [8 * 8 * 4]u8 = undefined;
    for (0..8) |y| for (0..8) |x| {
        const i = (y * 8 + x) * 4;
        const red = (x % 2) == 0;
        texels[i + 0] = if (red) 255 else 0;
        texels[i + 1] = 0;
        texels[i + 2] = if (red) 0 else 255;
        texels[i + 3] = 255;
    };
    var tex: gles.GLuint = 0;
    glGenTextures(1, @ptrCast(&tex));
    glBindTexture(gles.GL_TEXTURE_2D, tex);
    glTexImage2D(gles.GL_TEXTURE_2D, 0, gles.GL_RGBA, TS, TS, 0, gles.GL_RGBA, gles.GL_UNSIGNED_BYTE, &texels);
    glTexParameteri(gles.GL_TEXTURE_2D, gles.GL_TEXTURE_MIN_FILTER, gles.GL_NEAREST_MIPMAP_NEAREST);
    glTexParameteri(gles.GL_TEXTURE_2D, gles.GL_TEXTURE_MAG_FILTER, gles.GL_NEAREST);
    glGenerateMipmap(gles.GL_TEXTURE_2D);
    glUniform1i(glGetUniformLocation(prog, "tex"), 0);
    glUniform1f(glGetUniformLocation(prog, "uLod"), 0.0); // explicit LOD 0 (the finest usable level)

    const Vtx = extern struct { x: f32, y: f32, u: f32, v: f32 };
    const quad = [4]Vtx{
        .{ .x = -1, .y = -1, .u = 0, .v = 0 }, .{ .x = 1, .y = -1, .u = 1, .v = 0 },
        .{ .x = -1, .y = 1, .u = 0, .v = 1 },  .{ .x = 1, .y = 1, .u = 1, .v = 1 },
    };
    var vbo: gles.GLuint = 0;
    glGenBuffers(1, @ptrCast(&vbo));
    glBindBuffer(gles.GL_ARRAY_BUFFER, vbo);
    glBufferData(gles.GL_ARRAY_BUFFER, @sizeOf(@TypeOf(quad)), &quad, gles.GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, gles.GL_FLOAT, gles.GL_FALSE, @sizeOf(Vtx), @ptrFromInt(0));
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 2, gles.GL_FLOAT, gles.GL_FALSE, @sizeOf(Vtx), @ptrFromInt(8));
    glEnableVertexAttribArray(1);

    const s: *state.Surface = @ptrCast(@alignCast(surf.?));
    const dev = s.display.device.?;
    const ci = (32 * 64 + 32) * 4;

    // base_level 0 (default): LOD 0 samples the fine striped level -> a pure stripe (R or B).
    glDrawArrays(gles.GL_TRIANGLE_STRIP, 0, 4);
    try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());
    const px0 = try flushMap(dev, s);
    const base_pure = (px0[ci + 0] > 200 and px0[ci + 2] < 60) or (px0[ci + 0] < 60 and px0[ci + 2] > 200);
    try std.testing.expect(base_pure); // base_level 0: LOD 0 is the fine striped level

    // base_level 1: LOD 0 is clamped to level 1 -> the box-downsampled purple (both R and B).
    glTexParameteri(gles.GL_TEXTURE_2D, gles.GL_TEXTURE_BASE_LEVEL, 1);
    var qbl: gles.GLint = -1;
    glGetTexParameteriv(gles.GL_TEXTURE_2D, gles.GL_TEXTURE_BASE_LEVEL, @ptrCast(&qbl));
    try std.testing.expectEqual(@as(gles.GLint, 1), qbl); // the set base level reads back
    glClearColor(0, 0, 0, 1);
    glClear(gles.GL_COLOR_BUFFER_BIT);
    glDrawArrays(gles.GL_TRIANGLE_STRIP, 0, 4);
    try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());
    const px1 = try flushMap(dev, s);
    try std.testing.expect(px1[ci + 0] > 60 and px1[ci + 2] > 60); // purple: base_level forced level 1
}

test "GL_TEXTURE_SWIZZLE: remaps sampled channels (the font-atlas broadcast idiom)" {
    _ = eglBindAPI(egl.EGL_OPENGL_ES_API);
    const dpy = eglGetPlatformDisplay(egl.EGL_PLATFORM_SURFACELESS_MESA, @ptrFromInt(0xE7C03), null);
    try std.testing.expectEqual(egl.EGL_TRUE, eglInitialize(dpy, null, null));
    defer _ = eglTerminate(dpy);
    const cfg = state.configHandle(0);
    const ctx = eglCreateContext(dpy, cfg, egl.EGL_NO_CONTEXT, null);
    const W: EGLint = 64;
    const pb_attribs = [_]EGLint{ egl.EGL_WIDTH, W, egl.EGL_HEIGHT, W, egl.EGL_NONE };
    const surf = eglCreatePbufferSurface(dpy, cfg, &pb_attribs);
    try std.testing.expectEqual(egl.EGL_TRUE, eglMakeCurrent(dpy, surf, surf, ctx));
    defer _ = eglDestroyContext(dpy, ctx);
    defer _ = eglDestroySurface(dpy, surf);
    defer _ = eglMakeCurrent(dpy, null, null, egl.EGL_NO_CONTEXT);

    const vs_src: [*:0]const gles.GLchar =
        \\attribute vec2 position;
        \\attribute vec2 uv;
        \\varying vec2 vUv;
        \\void main() { gl_Position = vec4(position, 0.0, 1.0); vUv = uv; }
    ;
    const fs_src: [*:0]const gles.GLchar =
        \\precision mediump float;
        \\uniform sampler2D tex;
        \\varying vec2 vUv;
        \\void main() { gl_FragColor = texture2D(tex, vUv); }
    ;
    const vs = glCreateShader(gles.GL_VERTEX_SHADER);
    glShaderSource(vs, 1, &[_]?[*:0]const gles.GLchar{vs_src}, null);
    glCompileShader(vs);
    const fs = glCreateShader(gles.GL_FRAGMENT_SHADER);
    glShaderSource(fs, 1, &[_]?[*:0]const gles.GLchar{fs_src}, null);
    glCompileShader(fs);
    const prog = glCreateProgram();
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glBindAttribLocation(prog, 0, "position");
    glBindAttribLocation(prog, 1, "uv");
    glLinkProgram(prog);
    glUseProgram(prog);

    // A 2x2 texture: R=200, G=100, B=40, A=255 in every texel.
    const TS: gles.GLsizei = 2;
    var texels: [2 * 2 * 4]u8 = undefined;
    for (0..4) |p| {
        texels[p * 4 + 0] = 200;
        texels[p * 4 + 1] = 100;
        texels[p * 4 + 2] = 40;
        texels[p * 4 + 3] = 255;
    }
    var tex: gles.GLuint = 0;
    glGenTextures(1, @ptrCast(&tex));
    glBindTexture(gles.GL_TEXTURE_2D, tex);
    glTexImage2D(gles.GL_TEXTURE_2D, 0, gles.GL_RGBA, TS, TS, 0, gles.GL_RGBA, gles.GL_UNSIGNED_BYTE, &texels);
    glTexParameteri(gles.GL_TEXTURE_2D, gles.GL_TEXTURE_MIN_FILTER, gles.GL_NEAREST);
    glTexParameteri(gles.GL_TEXTURE_2D, gles.GL_TEXTURE_MAG_FILTER, gles.GL_NEAREST);
    glUniform1i(glGetUniformLocation(prog, "tex"), 0);

    const Vtx = extern struct { x: f32, y: f32, u: f32, v: f32 };
    const quad = [4]Vtx{
        .{ .x = -1, .y = -1, .u = 0, .v = 0 }, .{ .x = 1, .y = -1, .u = 1, .v = 0 },
        .{ .x = -1, .y = 1, .u = 0, .v = 1 },  .{ .x = 1, .y = 1, .u = 1, .v = 1 },
    };
    var vbo: gles.GLuint = 0;
    glGenBuffers(1, @ptrCast(&vbo));
    glBindBuffer(gles.GL_ARRAY_BUFFER, vbo);
    glBufferData(gles.GL_ARRAY_BUFFER, @sizeOf(@TypeOf(quad)), &quad, gles.GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, gles.GL_FLOAT, gles.GL_FALSE, @sizeOf(Vtx), @ptrFromInt(0));
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 2, gles.GL_FLOAT, gles.GL_FALSE, @sizeOf(Vtx), @ptrFromInt(8));
    glEnableVertexAttribArray(1);

    const s: *state.Surface = @ptrCast(@alignCast(surf.?));
    const dev = s.display.device.?;
    const ci = (32 * 64 + 32) * 4;

    // Identity swizzle: the sampled color is (R=200, G=100, B=40).
    glDrawArrays(gles.GL_TRIANGLE_STRIP, 0, 4);
    try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());
    const p0 = try flushMap(dev, s);
    try std.testing.expect(p0[ci + 0] > 180 and p0[ci + 1] > 80 and p0[ci + 1] < 130 and p0[ci + 2] < 70);

    // Broadcast R to every channel (the font-coverage idiom): all channels read 200.
    glTexParameteri(gles.GL_TEXTURE_2D, gles.GL_TEXTURE_SWIZZLE_R, @intCast(gles.GL_RED));
    glTexParameteri(gles.GL_TEXTURE_2D, gles.GL_TEXTURE_SWIZZLE_G, @intCast(gles.GL_RED));
    glTexParameteri(gles.GL_TEXTURE_2D, gles.GL_TEXTURE_SWIZZLE_B, @intCast(gles.GL_RED));
    glClearColor(0, 0, 0, 1);
    glClear(gles.GL_COLOR_BUFFER_BIT);
    glDrawArrays(gles.GL_TRIANGLE_STRIP, 0, 4);
    try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());
    const p1 = try flushMap(dev, s);
    // R, G, B all now carry R (200). The green channel jumped from ~100 to ~200.
    try std.testing.expect(p1[ci + 0] > 180 and p1[ci + 1] > 180 and p1[ci + 2] > 180);

    // The swizzle reads back via glGetTexParameteriv.
    var qsw: gles.GLint = 0;
    glGetTexParameteriv(gles.GL_TEXTURE_2D, gles.GL_TEXTURE_SWIZZLE_G, @ptrCast(&qsw));
    try std.testing.expectEqual(@as(gles.GLint, @intCast(gles.GL_RED)), qsw);
}

test "GL_TEXTURE_LOD_BIAS: a positive bias pushes a magnified texture to a coarser mip" {
    _ = eglBindAPI(egl.EGL_OPENGL_ES_API);
    const dpy = eglGetPlatformDisplay(egl.EGL_PLATFORM_SURFACELESS_MESA, @ptrFromInt(0xE7D04), null);
    try std.testing.expectEqual(egl.EGL_TRUE, eglInitialize(dpy, null, null));
    defer _ = eglTerminate(dpy);
    const cfg = state.configHandle(0);
    const ctx = eglCreateContext(dpy, cfg, egl.EGL_NO_CONTEXT, null);
    const W: EGLint = 64;
    const pb_attribs = [_]EGLint{ egl.EGL_WIDTH, W, egl.EGL_HEIGHT, W, egl.EGL_NONE };
    const surf = eglCreatePbufferSurface(dpy, cfg, &pb_attribs);
    try std.testing.expectEqual(egl.EGL_TRUE, eglMakeCurrent(dpy, surf, surf, ctx));
    defer _ = eglDestroyContext(dpy, ctx);
    defer _ = eglDestroySurface(dpy, surf);
    defer _ = eglMakeCurrent(dpy, null, null, egl.EGL_NO_CONTEXT);

    const vs_src: [*:0]const gles.GLchar =
        \\attribute vec2 position;
        \\attribute vec2 uv;
        \\varying vec2 vUv;
        \\void main() { gl_Position = vec4(position, 0.0, 1.0); vUv = uv; }
    ;
    // Implicit-LOD sample: the LOD bias shifts which mip the derivative LOD selects.
    const fs_src: [*:0]const gles.GLchar =
        \\precision mediump float;
        \\uniform sampler2D tex;
        \\varying vec2 vUv;
        \\void main() { gl_FragColor = texture2D(tex, vUv); }
    ;
    const vs = glCreateShader(gles.GL_VERTEX_SHADER);
    glShaderSource(vs, 1, &[_]?[*:0]const gles.GLchar{vs_src}, null);
    glCompileShader(vs);
    const fs = glCreateShader(gles.GL_FRAGMENT_SHADER);
    glShaderSource(fs, 1, &[_]?[*:0]const gles.GLchar{fs_src}, null);
    glCompileShader(fs);
    const prog = glCreateProgram();
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glBindAttribLocation(prog, 0, "position");
    glBindAttribLocation(prog, 1, "uv");
    glLinkProgram(prog);
    glUseProgram(prog);

    // 8x8 red/blue stripes; box-downsampled mips are purple.
    const TS: gles.GLsizei = 8;
    var texels: [8 * 8 * 4]u8 = undefined;
    for (0..8) |y| for (0..8) |x| {
        const i = (y * 8 + x) * 4;
        const red = (x % 2) == 0;
        texels[i + 0] = if (red) 255 else 0;
        texels[i + 1] = 0;
        texels[i + 2] = if (red) 0 else 255;
        texels[i + 3] = 255;
    };
    var tex: gles.GLuint = 0;
    glGenTextures(1, @ptrCast(&tex));
    glBindTexture(gles.GL_TEXTURE_2D, tex);
    glTexImage2D(gles.GL_TEXTURE_2D, 0, gles.GL_RGBA, TS, TS, 0, gles.GL_RGBA, gles.GL_UNSIGNED_BYTE, &texels);
    glTexParameteri(gles.GL_TEXTURE_2D, gles.GL_TEXTURE_MIN_FILTER, gles.GL_NEAREST_MIPMAP_NEAREST);
    glTexParameteri(gles.GL_TEXTURE_2D, gles.GL_TEXTURE_MAG_FILTER, gles.GL_NEAREST);
    glGenerateMipmap(gles.GL_TEXTURE_2D);
    glUniform1i(glGetUniformLocation(prog, "tex"), 0);

    const Vtx = extern struct { x: f32, y: f32, u: f32, v: f32 };
    // uv 0..1 over the whole 64x64 target: an 8x8 texture is magnified (implicit LOD < 0).
    const quad = [4]Vtx{
        .{ .x = -1, .y = -1, .u = 0, .v = 0 }, .{ .x = 1, .y = -1, .u = 1, .v = 0 },
        .{ .x = -1, .y = 1, .u = 0, .v = 1 },  .{ .x = 1, .y = 1, .u = 1, .v = 1 },
    };
    var vbo: gles.GLuint = 0;
    glGenBuffers(1, @ptrCast(&vbo));
    glBindBuffer(gles.GL_ARRAY_BUFFER, vbo);
    glBufferData(gles.GL_ARRAY_BUFFER, @sizeOf(@TypeOf(quad)), &quad, gles.GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, gles.GL_FLOAT, gles.GL_FALSE, @sizeOf(Vtx), @ptrFromInt(0));
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 2, gles.GL_FLOAT, gles.GL_FALSE, @sizeOf(Vtx), @ptrFromInt(8));
    glEnableVertexAttribArray(1);

    const s: *state.Surface = @ptrCast(@alignCast(surf.?));
    const dev = s.display.device.?;
    const ci = (32 * 64 + 32) * 4;

    // LOD_BIAS 0: the magnified sample reads the fine striped base level (pure R or B).
    glDrawArrays(gles.GL_TRIANGLE_STRIP, 0, 4);
    try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());
    const p0 = try flushMap(dev, s);
    const base_pure = (p0[ci + 0] > 200 and p0[ci + 2] < 60) or (p0[ci + 0] < 60 and p0[ci + 2] > 200);
    try std.testing.expect(base_pure);

    // LOD_BIAS +8: the effective LOD jumps to a coarse level -> the averaged purple.
    glTexParameterf(gles.GL_TEXTURE_2D, gles.GL_TEXTURE_LOD_BIAS, 8.0);
    var qb: gles.GLfloat = 0;
    glGetTexParameterfv(gles.GL_TEXTURE_2D, gles.GL_TEXTURE_LOD_BIAS, @ptrCast(&qb));
    try std.testing.expect(qb > 7.9 and qb < 8.1); // the float bias reads back (fraction preserved)
    glClearColor(0, 0, 0, 1);
    glClear(gles.GL_COLOR_BUFFER_BIT);
    glDrawArrays(gles.GL_TRIANGLE_STRIP, 0, 4);
    try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());
    const p1 = try flushMap(dev, s);
    try std.testing.expect(p1[ci + 0] > 60 and p1[ci + 2] > 60); // purple: the bias forced a coarse mip
}

test "sRGB framebuffer: rendering a linear color to a GL_SRGB8_ALPHA8 attachment encodes on write" {
    _ = eglBindAPI(egl.EGL_OPENGL_ES_API);
    const dpy = eglGetPlatformDisplay(egl.EGL_PLATFORM_SURFACELESS_MESA, @ptrFromInt(0xE7E05), null);
    try std.testing.expectEqual(egl.EGL_TRUE, eglInitialize(dpy, null, null));
    defer _ = eglTerminate(dpy);
    const cfg = state.configHandle(0);
    const ctx = eglCreateContext(dpy, cfg, egl.EGL_NO_CONTEXT, null);
    const W: EGLint = 64;
    const pb_attribs = [_]EGLint{ egl.EGL_WIDTH, W, egl.EGL_HEIGHT, W, egl.EGL_NONE };
    const surf = eglCreatePbufferSurface(dpy, cfg, &pb_attribs);
    try std.testing.expectEqual(egl.EGL_TRUE, eglMakeCurrent(dpy, surf, surf, ctx));
    defer _ = eglDestroyContext(dpy, ctx);
    defer _ = eglDestroySurface(dpy, surf);
    defer _ = eglMakeCurrent(dpy, null, null, egl.EGL_NO_CONTEXT);

    const vs_src: [*:0]const gles.GLchar =
        \\attribute vec2 position;
        \\void main() { gl_Position = vec4(position, 0.0, 1.0); }
    ;
    // The fragment output is LINEAR. An sRGB attachment encodes it on write.
    const fs_src: [*:0]const gles.GLchar =
        \\precision mediump float;
        \\void main() { gl_FragColor = vec4(0.5, 0.5, 0.5, 1.0); }
    ;
    const vs = glCreateShader(gles.GL_VERTEX_SHADER);
    glShaderSource(vs, 1, &[_]?[*:0]const gles.GLchar{vs_src}, null);
    glCompileShader(vs);
    const fs = glCreateShader(gles.GL_FRAGMENT_SHADER);
    glShaderSource(fs, 1, &[_]?[*:0]const gles.GLchar{fs_src}, null);
    glCompileShader(fs);
    const prog = glCreateProgram();
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glBindAttribLocation(prog, 0, "position");
    glLinkProgram(prog);
    glUseProgram(prog);

    const TS: gles.GLsizei = 32;
    var tex: gles.GLuint = 0;
    glGenTextures(1, @ptrCast(&tex));
    glBindTexture(gles.GL_TEXTURE_2D, tex);
    glTexImage2D(gles.GL_TEXTURE_2D, 0, @intCast(gles.GL_SRGB8_ALPHA8), TS, TS, 0, gles.GL_RGBA, gles.GL_UNSIGNED_BYTE, null);
    var fbo: gles.GLuint = 0;
    glGenFramebuffers(1, @ptrCast(&fbo));
    glBindFramebuffer(gles.GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(gles.GL_FRAMEBUFFER, gles.GL_COLOR_ATTACHMENT0, gles.GL_TEXTURE_2D, tex, 0);

    const tri = [_]f32{ -1, -1, 3, -1, -1, 3 }; // covers the whole target
    var vbo: gles.GLuint = 0;
    glGenBuffers(1, @ptrCast(&vbo));
    glBindBuffer(gles.GL_ARRAY_BUFFER, vbo);
    glBufferData(gles.GL_ARRAY_BUFFER, @sizeOf(@TypeOf(tri)), &tri, gles.GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, gles.GL_FLOAT, gles.GL_FALSE, 8, @ptrFromInt(0));
    glEnableVertexAttribArray(0);

    glViewport(0, 0, TS, TS);
    glDrawArrays(gles.GL_TRIANGLES, 0, 3);
    try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());

    var buf: [32 * 32 * 4]u8 = undefined;
    glReadPixels(0, 0, TS, TS, gles.GL_RGBA, gles.GL_UNSIGNED_BYTE, &buf);
    const ci = (16 * 32 + 16) * 4;
    // Linear 0.5 sRGB-encoded is ~188 (0.735*255), NOT the un-encoded 128. That proves the ROP
    // encoded on write. Alpha stays linear (255).
    try std.testing.expect(buf[ci + 0] > 180 and buf[ci + 0] < 196);
    try std.testing.expect(buf[ci + 3] > 250);
}

test "GL_TEXTURE_MIN_LOD: clamps the LOD up so even a magnified surface reads a coarser mip" {
    _ = eglBindAPI(egl.EGL_OPENGL_ES_API);
    const dpy = eglGetPlatformDisplay(egl.EGL_PLATFORM_SURFACELESS_MESA, @ptrFromInt(0xE7F06), null);
    try std.testing.expectEqual(egl.EGL_TRUE, eglInitialize(dpy, null, null));
    defer _ = eglTerminate(dpy);
    const cfg = state.configHandle(0);
    const ctx = eglCreateContext(dpy, cfg, egl.EGL_NO_CONTEXT, null);
    const W: EGLint = 64;
    const pb_attribs = [_]EGLint{ egl.EGL_WIDTH, W, egl.EGL_HEIGHT, W, egl.EGL_NONE };
    const surf = eglCreatePbufferSurface(dpy, cfg, &pb_attribs);
    try std.testing.expectEqual(egl.EGL_TRUE, eglMakeCurrent(dpy, surf, surf, ctx));
    defer _ = eglDestroyContext(dpy, ctx);
    defer _ = eglDestroySurface(dpy, surf);
    defer _ = eglMakeCurrent(dpy, null, null, egl.EGL_NO_CONTEXT);

    const vs_src: [*:0]const gles.GLchar =
        \\attribute vec2 position;
        \\attribute vec2 uv;
        \\varying vec2 vUv;
        \\void main() { gl_Position = vec4(position, 0.0, 1.0); vUv = uv; }
    ;
    const fs_src: [*:0]const gles.GLchar =
        \\precision mediump float;
        \\uniform sampler2D tex;
        \\varying vec2 vUv;
        \\void main() { gl_FragColor = texture2D(tex, vUv); }
    ;
    const vs = glCreateShader(gles.GL_VERTEX_SHADER);
    glShaderSource(vs, 1, &[_]?[*:0]const gles.GLchar{vs_src}, null);
    glCompileShader(vs);
    const fs = glCreateShader(gles.GL_FRAGMENT_SHADER);
    glShaderSource(fs, 1, &[_]?[*:0]const gles.GLchar{fs_src}, null);
    glCompileShader(fs);
    const prog = glCreateProgram();
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glBindAttribLocation(prog, 0, "position");
    glBindAttribLocation(prog, 1, "uv");
    glLinkProgram(prog);
    glUseProgram(prog);

    const TS: gles.GLsizei = 8;
    var texels: [8 * 8 * 4]u8 = undefined;
    for (0..8) |y| for (0..8) |x| {
        const i = (y * 8 + x) * 4;
        const red = (x % 2) == 0;
        texels[i + 0] = if (red) 255 else 0;
        texels[i + 1] = 0;
        texels[i + 2] = if (red) 0 else 255;
        texels[i + 3] = 255;
    };
    var tex: gles.GLuint = 0;
    glGenTextures(1, @ptrCast(&tex));
    glBindTexture(gles.GL_TEXTURE_2D, tex);
    glTexImage2D(gles.GL_TEXTURE_2D, 0, gles.GL_RGBA, TS, TS, 0, gles.GL_RGBA, gles.GL_UNSIGNED_BYTE, &texels);
    glTexParameteri(gles.GL_TEXTURE_2D, gles.GL_TEXTURE_MIN_FILTER, gles.GL_NEAREST_MIPMAP_NEAREST);
    glTexParameteri(gles.GL_TEXTURE_2D, gles.GL_TEXTURE_MAG_FILTER, gles.GL_NEAREST);
    glGenerateMipmap(gles.GL_TEXTURE_2D);
    glUniform1i(glGetUniformLocation(prog, "tex"), 0);

    const Vtx = extern struct { x: f32, y: f32, u: f32, v: f32 };
    const quad = [4]Vtx{
        .{ .x = -1, .y = -1, .u = 0, .v = 0 }, .{ .x = 1, .y = -1, .u = 1, .v = 0 },
        .{ .x = -1, .y = 1, .u = 0, .v = 1 },  .{ .x = 1, .y = 1, .u = 1, .v = 1 },
    };
    var vbo: gles.GLuint = 0;
    glGenBuffers(1, @ptrCast(&vbo));
    glBindBuffer(gles.GL_ARRAY_BUFFER, vbo);
    glBufferData(gles.GL_ARRAY_BUFFER, @sizeOf(@TypeOf(quad)), &quad, gles.GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, gles.GL_FLOAT, gles.GL_FALSE, @sizeOf(Vtx), @ptrFromInt(0));
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 2, gles.GL_FLOAT, gles.GL_FALSE, @sizeOf(Vtx), @ptrFromInt(8));
    glEnableVertexAttribArray(1);

    const s: *state.Surface = @ptrCast(@alignCast(surf.?));
    const dev = s.display.device.?;
    const ci = (32 * 64 + 32) * 4;

    // Default min_lod: the magnified sample reads the fine striped base level.
    glDrawArrays(gles.GL_TRIANGLE_STRIP, 0, 4);
    try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());
    const p0 = try flushMap(dev, s);
    const base_pure = (p0[ci + 0] > 200 and p0[ci + 2] < 60) or (p0[ci + 0] < 60 and p0[ci + 2] > 200);
    try std.testing.expect(base_pure);

    // GL_TEXTURE_MIN_LOD 2: the effective LOD is clamped up to 2 -> the coarse PURPLE mip.
    glTexParameterf(gles.GL_TEXTURE_2D, gles.GL_TEXTURE_MIN_LOD, 2.0);
    var qml: gles.GLfloat = 0;
    glGetTexParameterfv(gles.GL_TEXTURE_2D, gles.GL_TEXTURE_MIN_LOD, @ptrCast(&qml));
    try std.testing.expect(qml > 1.9 and qml < 2.1); // the float reads back
    glClearColor(0, 0, 0, 1);
    glClear(gles.GL_COLOR_BUFFER_BIT);
    glDrawArrays(gles.GL_TRIANGLE_STRIP, 0, 4);
    try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());
    const p1 = try flushMap(dev, s);
    try std.testing.expect(p1[ci + 0] > 60 and p1[ci + 2] > 60); // purple: min_lod pinned a coarse mip
}

test "GL_SAMPLE_ALPHA_TO_COVERAGE: a fragment's alpha becomes MSAA coverage (foliage/cut-out edges)" {
    _ = eglBindAPI(egl.EGL_OPENGL_ES_API);
    const dpy = eglGetPlatformDisplay(egl.EGL_PLATFORM_SURFACELESS_MESA, @ptrFromInt(0xE8006), null);
    try std.testing.expectEqual(egl.EGL_TRUE, eglInitialize(dpy, null, null));
    defer _ = eglTerminate(dpy);
    const cfg = state.configHandle(0);
    const ctx = eglCreateContext(dpy, cfg, egl.EGL_NO_CONTEXT, null);
    const W: EGLint = 64;
    const pb_attribs = [_]EGLint{ egl.EGL_WIDTH, W, egl.EGL_HEIGHT, W, egl.EGL_NONE };
    const surf = eglCreatePbufferSurface(dpy, cfg, &pb_attribs);
    try std.testing.expectEqual(egl.EGL_TRUE, eglMakeCurrent(dpy, surf, surf, ctx));
    defer _ = eglDestroyContext(dpy, ctx);
    defer _ = eglDestroySurface(dpy, surf);
    defer _ = eglMakeCurrent(dpy, null, null, egl.EGL_NO_CONTEXT);

    const vs_src: [*:0]const gles.GLchar =
        \\attribute vec2 position;
        \\void main() { gl_Position = vec4(position, 0.0, 1.0); }
    ;
    // White with 0.5 alpha: with alpha-to-coverage this covers ~half the samples.
    const fs_src: [*:0]const gles.GLchar =
        \\precision mediump float;
        \\void main() { gl_FragColor = vec4(1.0, 1.0, 1.0, 0.5); }
    ;
    const vs = glCreateShader(gles.GL_VERTEX_SHADER);
    glShaderSource(vs, 1, &[_]?[*:0]const gles.GLchar{vs_src}, null);
    glCompileShader(vs);
    const fs = glCreateShader(gles.GL_FRAGMENT_SHADER);
    glShaderSource(fs, 1, &[_]?[*:0]const gles.GLchar{fs_src}, null);
    glCompileShader(fs);
    const prog = glCreateProgram();
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glBindAttribLocation(prog, 0, "position");
    glLinkProgram(prog);
    glUseProgram(prog);

    const TS: gles.GLsizei = 32;
    var msrb: gles.GLuint = 0;
    glGenRenderbuffers(1, @ptrCast(&msrb));
    glBindRenderbuffer(gles.GL_RENDERBUFFER, msrb);
    glRenderbufferStorageMultisample(gles.GL_RENDERBUFFER, 4, gles.GL_RGBA8, TS, TS);
    var msfbo: gles.GLuint = 0;
    glGenFramebuffers(1, @ptrCast(&msfbo));
    glBindFramebuffer(gles.GL_FRAMEBUFFER, msfbo);
    glFramebufferRenderbuffer(gles.GL_FRAMEBUFFER, gles.GL_COLOR_ATTACHMENT0, gles.GL_RENDERBUFFER, msrb);

    // A full-screen triangle: the center pixel is fully geometrically covered, so only the
    // alpha-to-coverage mask reduces it.
    const tri = [_]f32{ -1, -1, 3, -1, -1, 3 };
    var vbo: gles.GLuint = 0;
    glGenBuffers(1, @ptrCast(&vbo));
    glBindBuffer(gles.GL_ARRAY_BUFFER, vbo);
    glBufferData(gles.GL_ARRAY_BUFFER, @sizeOf(@TypeOf(tri)), &tri, gles.GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, gles.GL_FLOAT, gles.GL_FALSE, 0, @ptrFromInt(0));
    glEnableVertexAttribArray(0);
    glViewport(0, 0, TS, TS);

    glEnable(gles.GL_SAMPLE_ALPHA_TO_COVERAGE);
    try std.testing.expect(glIsEnabled(gles.GL_SAMPLE_ALPHA_TO_COVERAGE) == gles.GL_TRUE);
    glClearColor(0, 0, 0, 1);
    glClear(gles.GL_COLOR_BUFFER_BIT);
    glDrawArrays(gles.GL_TRIANGLES, 0, 3);
    try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());

    // Resolve to the default framebuffer.
    glBindFramebuffer(gles.GL_READ_FRAMEBUFFER, msfbo);
    glBindFramebuffer(gles.GL_DRAW_FRAMEBUFFER, 0);
    glClearColor(0, 0, 0, 1);
    glClear(gles.GL_COLOR_BUFFER_BIT);
    glBlitFramebuffer(0, 0, TS, TS, 0, 0, W, W, gles.GL_COLOR_BUFFER_BIT, gles.GL_NEAREST);
    try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());

    const s: *state.Surface = @ptrCast(@alignCast(surf.?));
    const dev = s.display.device.?;
    const px = try flushMap(dev, s);
    const ci = (32 * 64 + 32) * 4;
    // 0.5 alpha over 4 samples -> ~2 covered -> the resolve averages white+black to a MID gray
    // (~128), NOT the full 255 an opaque draw (or no alpha-to-coverage) would give.
    try std.testing.expect(px[ci + 0] > 90 and px[ci + 0] < 170);
}

test "GL_SAMPLE_COVERAGE: a fixed coverage value ANDs the MSAA sample mask (screen-door / LOD fade)" {
    // GL_SAMPLE_COVERAGE reduces the covered-sample set by a FIXED value (independent of alpha):
    // ceil(value*samples) samples survive, inverted keeps the rest. An opaque white full-screen
    // triangle over black -> the resolve averages the surviving samples. value=0.25 over 4 samples
    // keeps 1 -> ~64 (dark). Invert keeps 3 -> ~191 (light). Double differential: value picks the
    // count, invert flips which. Without sample coverage both would be the opaque 255.
    _ = eglBindAPI(egl.EGL_OPENGL_ES_API);
    const dpy = eglGetPlatformDisplay(egl.EGL_PLATFORM_SURFACELESS_MESA, @ptrFromInt(0xE800C), null);
    try std.testing.expectEqual(egl.EGL_TRUE, eglInitialize(dpy, null, null));
    defer _ = eglTerminate(dpy);
    const cfg = state.configHandle(0);
    const ctx = eglCreateContext(dpy, cfg, egl.EGL_NO_CONTEXT, null);
    const W: EGLint = 64;
    const pb_attribs = [_]EGLint{ egl.EGL_WIDTH, W, egl.EGL_HEIGHT, W, egl.EGL_NONE };
    const surf = eglCreatePbufferSurface(dpy, cfg, &pb_attribs);
    try std.testing.expectEqual(egl.EGL_TRUE, eglMakeCurrent(dpy, surf, surf, ctx));
    defer _ = eglDestroyContext(dpy, ctx);
    defer _ = eglDestroySurface(dpy, surf);
    defer _ = eglMakeCurrent(dpy, null, null, egl.EGL_NO_CONTEXT);

    const vs_src: [*:0]const gles.GLchar =
        \\attribute vec2 position;
        \\void main() { gl_Position = vec4(position, 0.0, 1.0); }
    ;
    const fs_src: [*:0]const gles.GLchar =
        \\precision mediump float;
        \\void main() { gl_FragColor = vec4(1.0, 1.0, 1.0, 1.0); }
    ;
    const vs = glCreateShader(gles.GL_VERTEX_SHADER);
    glShaderSource(vs, 1, &[_]?[*:0]const gles.GLchar{vs_src}, null);
    glCompileShader(vs);
    const fs = glCreateShader(gles.GL_FRAGMENT_SHADER);
    glShaderSource(fs, 1, &[_]?[*:0]const gles.GLchar{fs_src}, null);
    glCompileShader(fs);
    const prog = glCreateProgram();
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glBindAttribLocation(prog, 0, "position");
    glLinkProgram(prog);
    glUseProgram(prog);

    const TS: gles.GLsizei = 32;
    var msrb: gles.GLuint = 0;
    glGenRenderbuffers(1, @ptrCast(&msrb));
    glBindRenderbuffer(gles.GL_RENDERBUFFER, msrb);
    glRenderbufferStorageMultisample(gles.GL_RENDERBUFFER, 4, gles.GL_RGBA8, TS, TS);
    var msfbo: gles.GLuint = 0;
    glGenFramebuffers(1, @ptrCast(&msfbo));
    glBindFramebuffer(gles.GL_FRAMEBUFFER, msfbo);
    glFramebufferRenderbuffer(gles.GL_FRAMEBUFFER, gles.GL_COLOR_ATTACHMENT0, gles.GL_RENDERBUFFER, msrb);

    const tri = [_]f32{ -1, -1, 3, -1, -1, 3 };
    var vbo: gles.GLuint = 0;
    glGenBuffers(1, @ptrCast(&vbo));
    glBindBuffer(gles.GL_ARRAY_BUFFER, vbo);
    glBufferData(gles.GL_ARRAY_BUFFER, @sizeOf(@TypeOf(tri)), &tri, gles.GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, gles.GL_FLOAT, gles.GL_FALSE, 0, @ptrFromInt(0));
    glEnableVertexAttribArray(0);

    glEnable(gles.GL_SAMPLE_COVERAGE);
    try std.testing.expect(glIsEnabled(gles.GL_SAMPLE_COVERAGE) == gles.GL_TRUE);

    const s: *state.Surface = @ptrCast(@alignCast(surf.?));
    const dev = s.display.device.?;
    const ci = (32 * 64 + 32) * 4;

    // Pass 1: value 0.25, non-invert -> 1 of 4 samples survive -> dark (~64).
    glSampleCoverage(0.25, gles.GL_FALSE);
    var qv: gles.GLfloat = 0;
    glGetFloatv(gles.GL_SAMPLE_COVERAGE_VALUE, @ptrCast(&qv));
    try std.testing.expect(qv > 0.24 and qv < 0.26); // the value reads back
    glBindFramebuffer(gles.GL_FRAMEBUFFER, msfbo);
    glViewport(0, 0, TS, TS);
    glClearColor(0, 0, 0, 1);
    glClear(gles.GL_COLOR_BUFFER_BIT);
    glDrawArrays(gles.GL_TRIANGLES, 0, 3);
    glBindFramebuffer(gles.GL_READ_FRAMEBUFFER, msfbo);
    glBindFramebuffer(gles.GL_DRAW_FRAMEBUFFER, 0);
    glClear(gles.GL_COLOR_BUFFER_BIT);
    glBlitFramebuffer(0, 0, TS, TS, 0, 0, W, W, gles.GL_COLOR_BUFFER_BIT, gles.GL_NEAREST);
    try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());
    {
        const px = try flushMap(dev, s);
        try std.testing.expect(px[ci + 0] > 40 and px[ci + 0] < 95); // ~1/4 white
    }

    // Pass 2: value 0.25, invert -> the other 3 of 4 samples survive -> light (~191).
    glSampleCoverage(0.25, gles.GL_TRUE);
    var qinv: gles.GLboolean = 0;
    glGetBooleanv(gles.GL_SAMPLE_COVERAGE_INVERT, @ptrCast(&qinv));
    try std.testing.expect(qinv == gles.GL_TRUE);
    glBindFramebuffer(gles.GL_FRAMEBUFFER, msfbo);
    glViewport(0, 0, TS, TS);
    glClearColor(0, 0, 0, 1);
    glClear(gles.GL_COLOR_BUFFER_BIT);
    glDrawArrays(gles.GL_TRIANGLES, 0, 3);
    glBindFramebuffer(gles.GL_READ_FRAMEBUFFER, msfbo);
    glBindFramebuffer(gles.GL_DRAW_FRAMEBUFFER, 0);
    glClear(gles.GL_COLOR_BUFFER_BIT);
    glBlitFramebuffer(0, 0, TS, TS, 0, 0, W, W, gles.GL_COLOR_BUFFER_BIT, gles.GL_NEAREST);
    try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());
    {
        const px = try flushMap(dev, s);
        try std.testing.expect(px[ci + 0] > 160 and px[ci + 0] < 215); // ~3/4 white
    }
}

test "GLES3 caps queries: MAX_3D_TEXTURE_SIZE / MAX_ARRAY_TEXTURE_LAYERS / MAX_SAMPLES are advertised" {
    // Prism supports sampler3D, sampler2DArray, and up to 4x MSAA, so glGetIntegerv must ANSWER these
    // caps (not GL_INVALID_ENUM). An engine's startup caps probe reads them to size its resources.
    // Differential: before these were unhandled -> value 0 + GL_INVALID_ENUM; now real values, no error.
    _ = eglBindAPI(egl.EGL_OPENGL_ES_API);
    const dpy = eglGetPlatformDisplay(egl.EGL_PLATFORM_SURFACELESS_MESA, @ptrFromInt(0xE8207), null);
    try std.testing.expectEqual(egl.EGL_TRUE, eglInitialize(dpy, null, null));
    defer _ = eglTerminate(dpy);
    const cfg = state.configHandle(0);
    const ctx = eglCreateContext(dpy, cfg, egl.EGL_NO_CONTEXT, null);
    const pb_attribs = [_]EGLint{ egl.EGL_WIDTH, 16, egl.EGL_HEIGHT, 16, egl.EGL_NONE };
    const surf = eglCreatePbufferSurface(dpy, cfg, &pb_attribs);
    try std.testing.expectEqual(egl.EGL_TRUE, eglMakeCurrent(dpy, surf, surf, ctx));
    defer _ = eglDestroyContext(dpy, ctx);
    defer _ = eglDestroySurface(dpy, surf);
    defer _ = eglMakeCurrent(dpy, null, null, egl.EGL_NO_CONTEXT);

    var v: gles.GLint = -1;
    glGetIntegerv(gles.GL_MAX_3D_TEXTURE_SIZE, @ptrCast(&v));
    try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());
    try std.testing.expect(v >= 256); // GLES3 minimum; Prism reports 2048

    v = -1;
    glGetIntegerv(gles.GL_MAX_ARRAY_TEXTURE_LAYERS, @ptrCast(&v));
    try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());
    try std.testing.expect(v >= 256); // GLES3 minimum; Prism reports 2048

    v = -1;
    glGetIntegerv(gles.GL_MAX_SAMPLES, @ptrCast(&v));
    try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());
    try std.testing.expectEqual(@as(gles.GLint, 4), v); // the software rasterizer resolves up to 4x
}

test "GLES3 caps queries 2: compressed-texture-formats list + MAX_ELEMENT_INDEX / MAX_TEXTURE_LOD_BIAS" {
    // Prism decodes 7 compressed formats (ETC1 + S3TC DXT1/1a/3/5 + RGTC1/2) but never advertised
    // them, so an app's glGetIntegerv(GL_NUM_COMPRESSED_TEXTURE_FORMATS)+GL_COMPRESSED_TEXTURE_FORMATS
    // probe found none and skipped compressed assets. Also advertise MAX_ELEMENT_INDEX (32-bit indices
    // work), the MAX_ELEMENTS_* draw hints, and MAX_TEXTURE_LOD_BIAS (the sampler honors LOD bias).
    _ = eglBindAPI(egl.EGL_OPENGL_ES_API);
    const dpy = eglGetPlatformDisplay(egl.EGL_PLATFORM_SURFACELESS_MESA, @ptrFromInt(0xE8307), null);
    try std.testing.expectEqual(egl.EGL_TRUE, eglInitialize(dpy, null, null));
    defer _ = eglTerminate(dpy);
    const cfg = state.configHandle(0);
    const ctx = eglCreateContext(dpy, cfg, egl.EGL_NO_CONTEXT, null);
    const pb_attribs = [_]EGLint{ egl.EGL_WIDTH, 16, egl.EGL_HEIGHT, 16, egl.EGL_NONE };
    const surf = eglCreatePbufferSurface(dpy, cfg, &pb_attribs);
    try std.testing.expectEqual(egl.EGL_TRUE, eglMakeCurrent(dpy, surf, surf, ctx));
    defer _ = eglDestroyContext(dpy, ctx);
    defer _ = eglDestroySurface(dpy, surf);
    defer _ = eglMakeCurrent(dpy, null, null, egl.EGL_NO_CONTEXT);

    // The count, then the list, must agree and include the common ETC1 + DXT5 tokens.
    var n: gles.GLint = -1;
    glGetIntegerv(gles.GL_NUM_COMPRESSED_TEXTURE_FORMATS, @ptrCast(&n));
    try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());
    try std.testing.expect(n >= 5);
    var fmts = [_]gles.GLint{0} ** 32;
    glGetIntegerv(gles.GL_COMPRESSED_TEXTURE_FORMATS, @ptrCast(&fmts));
    try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());
    var saw_etc1 = false;
    var saw_dxt5 = false;
    for (fmts[0..@intCast(n)]) |f| {
        if (f == @as(gles.GLint, @intCast(gles.GL_ETC1_RGB8_OES))) saw_etc1 = true;
        if (f == @as(gles.GLint, @intCast(gles.GL_COMPRESSED_RGBA_S3TC_DXT5_EXT))) saw_dxt5 = true;
    }
    try std.testing.expect(saw_etc1 and saw_dxt5);
    // The slot past the list stays untouched (the count bounds the write).
    try std.testing.expectEqual(@as(gles.GLint, 0), fmts[@intCast(n)]);

    var v: gles.GLint = -1;
    glGetIntegerv(gles.GL_MAX_ELEMENT_INDEX, @ptrCast(&v));
    try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());
    try std.testing.expect(v >= 0x00FFFFFF);

    v = -1;
    glGetIntegerv(gles.GL_MAX_ELEMENTS_INDICES, @ptrCast(&v));
    try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());
    try std.testing.expect(v > 0);

    var fv: gles.GLfloat = -1;
    glGetFloatv(gles.GL_MAX_TEXTURE_LOD_BIAS, @ptrCast(&fv));
    try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());
    try std.testing.expect(fv >= 2.0); // GLES3 minimum
}

test "GL_UNPACK_ROW_LENGTH / SKIP_PIXELS: upload a sub-rectangle from a wider source buffer (atlas tile)" {
    _ = eglBindAPI(egl.EGL_OPENGL_ES_API);
    const dpy = eglGetPlatformDisplay(egl.EGL_PLATFORM_SURFACELESS_MESA, @ptrFromInt(0xE8107), null);
    try std.testing.expectEqual(egl.EGL_TRUE, eglInitialize(dpy, null, null));
    defer _ = eglTerminate(dpy);
    const cfg = state.configHandle(0);
    const ctx = eglCreateContext(dpy, cfg, egl.EGL_NO_CONTEXT, null);
    const W: EGLint = 64;
    const pb_attribs = [_]EGLint{ egl.EGL_WIDTH, W, egl.EGL_HEIGHT, W, egl.EGL_NONE };
    const surf = eglCreatePbufferSurface(dpy, cfg, &pb_attribs);
    try std.testing.expectEqual(egl.EGL_TRUE, eglMakeCurrent(dpy, surf, surf, ctx));
    defer _ = eglDestroyContext(dpy, ctx);
    defer _ = eglDestroySurface(dpy, surf);
    defer _ = eglMakeCurrent(dpy, null, null, egl.EGL_NO_CONTEXT);

    const vs_src: [*:0]const gles.GLchar =
        \\attribute vec2 position;
        \\attribute vec2 uv;
        \\varying vec2 vUv;
        \\void main() { gl_Position = vec4(position, 0.0, 1.0); vUv = uv; }
    ;
    const fs_src: [*:0]const gles.GLchar =
        \\precision mediump float;
        \\uniform sampler2D tex;
        \\varying vec2 vUv;
        \\void main() { gl_FragColor = texture2D(tex, vUv); }
    ;
    const vs = glCreateShader(gles.GL_VERTEX_SHADER);
    glShaderSource(vs, 1, &[_]?[*:0]const gles.GLchar{vs_src}, null);
    glCompileShader(vs);
    const fs = glCreateShader(gles.GL_FRAGMENT_SHADER);
    glShaderSource(fs, 1, &[_]?[*:0]const gles.GLchar{fs_src}, null);
    glCompileShader(fs);
    const prog = glCreateProgram();
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glBindAttribLocation(prog, 0, "position");
    glBindAttribLocation(prog, 1, "uv");
    glLinkProgram(prog);
    glUseProgram(prog);

    // A 4-wide x 2-tall source: each row is [red, green, blue, white]. We upload a 2x2 texture
    // starting at column 1 (skip 1) with a row stride of 4 -> the texture is [green, blue].
    var srcbuf: [4 * 2 * 4]u8 = undefined;
    const cols = [4][3]u8{ .{ 255, 0, 0 }, .{ 0, 255, 0 }, .{ 0, 0, 255 }, .{ 255, 255, 255 } };
    for (0..2) |ry| for (0..4) |rx| {
        const i = (ry * 4 + rx) * 4;
        srcbuf[i + 0] = cols[rx][0];
        srcbuf[i + 1] = cols[rx][1];
        srcbuf[i + 2] = cols[rx][2];
        srcbuf[i + 3] = 255;
    };
    var tex: gles.GLuint = 0;
    glGenTextures(1, @ptrCast(&tex));
    glBindTexture(gles.GL_TEXTURE_2D, tex);
    glPixelStorei(gles.GL_UNPACK_ROW_LENGTH, 4); // source rows are 4 pixels wide
    glPixelStorei(gles.GL_UNPACK_SKIP_PIXELS, 1); // start at column 1 (green)
    glTexImage2D(gles.GL_TEXTURE_2D, 0, gles.GL_RGBA, 2, 2, 0, gles.GL_RGBA, gles.GL_UNSIGNED_BYTE, &srcbuf);
    try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());
    glPixelStorei(gles.GL_UNPACK_ROW_LENGTH, 0); // reset so later tests are unaffected
    glPixelStorei(gles.GL_UNPACK_SKIP_PIXELS, 0);
    glTexParameteri(gles.GL_TEXTURE_2D, gles.GL_TEXTURE_MIN_FILTER, gles.GL_NEAREST);
    glTexParameteri(gles.GL_TEXTURE_2D, gles.GL_TEXTURE_MAG_FILTER, gles.GL_NEAREST);
    glUniform1i(glGetUniformLocation(prog, "tex"), 0);

    const Vtx = extern struct { x: f32, y: f32, u: f32, v: f32 };
    const quad = [4]Vtx{
        .{ .x = -1, .y = -1, .u = 0, .v = 0 }, .{ .x = 1, .y = -1, .u = 1, .v = 0 },
        .{ .x = -1, .y = 1, .u = 0, .v = 1 },  .{ .x = 1, .y = 1, .u = 1, .v = 1 },
    };
    var vbo: gles.GLuint = 0;
    glGenBuffers(1, @ptrCast(&vbo));
    glBindBuffer(gles.GL_ARRAY_BUFFER, vbo);
    glBufferData(gles.GL_ARRAY_BUFFER, @sizeOf(@TypeOf(quad)), &quad, gles.GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, gles.GL_FLOAT, gles.GL_FALSE, @sizeOf(Vtx), @ptrFromInt(0));
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 2, gles.GL_FLOAT, gles.GL_FALSE, @sizeOf(Vtx), @ptrFromInt(8));
    glEnableVertexAttribArray(1);

    glViewport(0, 0, W, W);
    glClearColor(0, 0, 0, 1);
    glClear(gles.GL_COLOR_BUFFER_BIT);
    glDrawArrays(gles.GL_TRIANGLE_STRIP, 0, 4);
    try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());

    const s: *state.Surface = @ptrCast(@alignCast(surf.?));
    const dev = s.display.device.?;
    const px = try flushMap(dev, s);
    // NEAREST 2x2 texture on a full quad: left half = texel 0 (green), right half = texel 1 (blue).
    const left = (32 * 64 + 16) * 4;
    const right = (32 * 64 + 48) * 4;
    try std.testing.expect(px[left + 1] > 200 and px[left + 0] < 60); // green (the skip-1 column)
    try std.testing.expect(px[right + 2] > 200 and px[right + 0] < 60); // blue (column 2)
}

test "GL_PACK_ROW_LENGTH / SKIP_PIXELS / SKIP_ROWS: read a region into a sub-rectangle of a wider dest" {
    // The readback mirror of GL_UNPACK_ROW_LENGTH: glReadPixels must write into a SUB-RECTANGLE of a
    // wider destination buffer. pack_row_length is the destination row stride in pixels, skip_* the
    // destination start offset. The use case: read one region straight into a packed CPU atlas slot.
    // Differential: pre-fill the destination with a sentinel, read a 2x2 red region with a 5-wide
    // stride + skip(1,1), and verify only the 4 target pixels became red. Every sentinel byte around
    // them stays untouched (without the pack params they would land tightly-packed at offset 0).
    _ = eglBindAPI(egl.EGL_OPENGL_ES_API);
    const dpy = eglGetPlatformDisplay(egl.EGL_PLATFORM_SURFACELESS_MESA, @ptrFromInt(0xE8109), null);
    try std.testing.expectEqual(egl.EGL_TRUE, eglInitialize(dpy, null, null));
    defer _ = eglTerminate(dpy);
    const cfg = state.configHandle(0);
    const ctx = eglCreateContext(dpy, cfg, egl.EGL_NO_CONTEXT, null);
    const W: EGLint = 16;
    const pb_attribs = [_]EGLint{ egl.EGL_WIDTH, W, egl.EGL_HEIGHT, W, egl.EGL_NONE };
    const surf = eglCreatePbufferSurface(dpy, cfg, &pb_attribs);
    try std.testing.expectEqual(egl.EGL_TRUE, eglMakeCurrent(dpy, surf, surf, ctx));
    defer _ = eglDestroyContext(dpy, ctx);
    defer _ = eglDestroySurface(dpy, surf);
    defer _ = eglMakeCurrent(dpy, null, null, egl.EGL_NO_CONTEXT);

    glViewport(0, 0, W, W);
    glClearColor(1, 0, 0, 1); // solid red framebuffer
    glClear(gles.GL_COLOR_BUFFER_BIT);
    try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());

    // Destination: 5 pixels wide x 4 tall of RGBA8, all bytes sentinel 0xAB.
    var dstbuf: [5 * 4 * 4]u8 = [_]u8{0xAB} ** (5 * 4 * 4);
    glPixelStorei(gles.GL_PACK_ROW_LENGTH, 5); // dest rows are 5 pixels wide
    glPixelStorei(gles.GL_PACK_SKIP_PIXELS, 1); // start at column 1
    glPixelStorei(gles.GL_PACK_SKIP_ROWS, 1); // start at row 1
    glReadPixels(0, 0, 2, 2, gles.GL_RGBA, gles.GL_UNSIGNED_BYTE, &dstbuf);
    try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());
    glPixelStorei(gles.GL_PACK_ROW_LENGTH, 0); // reset so later tests are unaffected
    glPixelStorei(gles.GL_PACK_SKIP_PIXELS, 0);
    glPixelStorei(gles.GL_PACK_SKIP_ROWS, 0);

    // dst_stride = 5*4 = 20 bytes; dst_skip = 1*20 + 1*4 = 24. The 2x2 red region lands at:
    //   row0: offsets 24, 28   row1: offsets 44, 48.
    const targets = [_]usize{ 24, 28, 44, 48 };
    for (targets) |t| {
        try std.testing.expectEqual(@as(u8, 255), dstbuf[t + 0]); // R
        try std.testing.expectEqual(@as(u8, 0), dstbuf[t + 1]); // G
        try std.testing.expectEqual(@as(u8, 0), dstbuf[t + 2]); // B
        try std.testing.expectEqual(@as(u8, 255), dstbuf[t + 3]); // A
    }
    // Everything else stays the sentinel: the leading skip (offsets 0 and 20, the skipped col/row)
    // and past the read width (offset 32, col 3 of row 1) must be untouched.
    for ([_]usize{ 0, 20, 32 }) |t| {
        try std.testing.expectEqual(@as(u8, 0xAB), dstbuf[t + 0]);
    }
}

test "EGL implicit-mipmap oracle: a MINIFIED texture() auto-selects a coarser mip from the derivatives" {
    // Automatic mipmapping: a plain texture() on a minified surface must pick a coarser mip level
    // (from the texture-coordinate screen-space derivatives), not always the base level. An 8x8
    // red/blue stripe texture, box-downsampled mips = purple. Drawn with uv 0..1 (magnified) the
    // center is a pure stripe (base level); drawn with uv 0..16 (minified ~8x, LOD >= 1) the center
    // is purple (a coarse averaged level). The difference proves derivative LOD selection works.
    _ = eglBindAPI(egl.EGL_OPENGL_ES_API);
    const dpy = eglGetPlatformDisplay(egl.EGL_PLATFORM_SURFACELESS_MESA, @ptrFromInt(0x7E654), null);
    try std.testing.expectEqual(egl.EGL_TRUE, eglInitialize(dpy, null, null));
    const cfg = state.configHandle(0);
    const ctx = eglCreateContext(dpy, cfg, egl.EGL_NO_CONTEXT, null);
    const Wd: EGLint = 64;
    const pb = [_]EGLint{ egl.EGL_WIDTH, Wd, egl.EGL_HEIGHT, Wd, egl.EGL_NONE };
    const surf = eglCreatePbufferSurface(dpy, cfg, &pb);
    try std.testing.expectEqual(egl.EGL_TRUE, eglMakeCurrent(dpy, surf, surf, ctx));
    defer {
        _ = eglMakeCurrent(dpy, null, null, egl.EGL_NO_CONTEXT);
        _ = eglDestroySurface(dpy, surf);
        _ = eglDestroyContext(dpy, ctx);
        _ = eglTerminate(dpy);
    }

    const vs_src: [*:0]const gles.GLchar =
        \\attribute vec2 position;
        \\attribute vec2 uv;
        \\varying vec2 vUv;
        \\void main() { gl_Position = vec4(position, 0.0, 1.0); vUv = uv; }
    ;
    const fs_src: [*:0]const gles.GLchar =
        \\precision mediump float;
        \\uniform sampler2D tex;
        \\varying vec2 vUv;
        \\void main() { gl_FragColor = texture2D(tex, vUv); }
    ;
    const vs = glCreateShader(gles.GL_VERTEX_SHADER);
    glShaderSource(vs, 1, &[_]?[*:0]const gles.GLchar{vs_src}, null);
    glCompileShader(vs);
    const fs = glCreateShader(gles.GL_FRAGMENT_SHADER);
    glShaderSource(fs, 1, &[_]?[*:0]const gles.GLchar{fs_src}, null);
    glCompileShader(fs);
    const prog = glCreateProgram();
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glBindAttribLocation(prog, 0, "position");
    glBindAttribLocation(prog, 1, "uv");
    glLinkProgram(prog);
    glUseProgram(prog);
    defer {
        glDeleteProgram(prog);
        glDeleteShader(vs);
        glDeleteShader(fs);
    }

    var texels: [8 * 8 * 4]u8 = undefined;
    for (0..8) |y| for (0..8) |x| {
        const i = (y * 8 + x) * 4;
        const red = (x % 2) == 0;
        texels[i + 0] = if (red) 255 else 0;
        texels[i + 1] = 0;
        texels[i + 2] = if (red) 0 else 255;
        texels[i + 3] = 255;
    };
    var tex: gles.GLuint = 0;
    glGenTextures(1, @ptrCast(&tex));
    glActiveTexture(gles.GL_TEXTURE0);
    glBindTexture(gles.GL_TEXTURE_2D, tex);
    glTexImage2D(gles.GL_TEXTURE_2D, 0, gles.GL_RGBA, 8, 8, 0, gles.GL_RGBA, gles.GL_UNSIGNED_BYTE, &texels);
    glTexParameteri(gles.GL_TEXTURE_2D, gles.GL_TEXTURE_MIN_FILTER, gles.GL_NEAREST_MIPMAP_NEAREST);
    glTexParameteri(gles.GL_TEXTURE_2D, gles.GL_TEXTURE_MAG_FILTER, gles.GL_NEAREST);
    glTexParameteri(gles.GL_TEXTURE_2D, gles.GL_TEXTURE_WRAP_S, @intCast(gles.GL_REPEAT));
    glTexParameteri(gles.GL_TEXTURE_2D, gles.GL_TEXTURE_WRAP_T, @intCast(gles.GL_REPEAT));
    glGenerateMipmap(gles.GL_TEXTURE_2D);
    glUniform1i(glGetUniformLocation(prog, "tex"), 0);
    defer glDeleteTextures(1, &[_]gles.GLuint{tex});

    const Vtx = extern struct { x: f32, y: f32, u: f32, v: f32 };
    var vbo: gles.GLuint = 0;
    glGenBuffers(1, @ptrCast(&vbo));
    glBindBuffer(gles.GL_ARRAY_BUFFER, vbo);
    glVertexAttribPointer(0, 2, gles.GL_FLOAT, gles.GL_FALSE, @sizeOf(Vtx), @ptrFromInt(0));
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 2, gles.GL_FLOAT, gles.GL_FALSE, @sizeOf(Vtx), @ptrFromInt(8));
    glEnableVertexAttribArray(1);
    defer glDeleteBuffers(1, &[_]gles.GLuint{vbo});

    const s: *state.Surface = @ptrCast(@alignCast(surf.?));
    const dev = s.display.device.?;
    const ci = (32 * 64 + 32) * 4;

    const draw = struct {
        fn d(vb: gles.GLuint, uvmax: f32) void {
            const Vt = extern struct { x: f32, y: f32, u: f32, v: f32 };
            const quad = [4]Vt{
                .{ .x = -1, .y = -1, .u = 0, .v = 0 },    .{ .x = 1, .y = -1, .u = uvmax, .v = 0 },
                .{ .x = -1, .y = 1, .u = 0, .v = uvmax }, .{ .x = 1, .y = 1, .u = uvmax, .v = uvmax },
            };
            glBindBuffer(gles.GL_ARRAY_BUFFER, vb);
            glBufferData(gles.GL_ARRAY_BUFFER, @sizeOf(@TypeOf(quad)), &quad, gles.GL_STATIC_DRAW);
            glClearColor(0, 0, 0, 1);
            glClear(gles.GL_COLOR_BUFFER_BIT);
            glDrawArrays(gles.GL_TRIANGLE_STRIP, 0, 4);
        }
    }.d;

    // uv 0..1: magnified (LOD ~0) -> the base level = a pure stripe (exactly one of R / B saturated).
    draw(vbo, 1.0);
    const pmag = try flushMap(dev, s);
    const pure = (pmag[ci + 0] > 200 and pmag[ci + 2] < 60) or (pmag[ci + 0] < 60 and pmag[ci + 2] > 200);
    try std.testing.expect(pure);
    // uv 0..16: minified ~8x (LOD >= 1) -> a coarse (box-averaged) level = purple (both R and B).
    draw(vbo, 16.0);
    const pmin = try flushMap(dev, s);
    try std.testing.expect(pmin[ci + 0] > 40 and pmin[ci + 2] > 40);
}

test "EGL depth-blit oracle: glBlitFramebuffer(GL_DEPTH_BUFFER_BIT) copies the depth buffer between FBOs" {
    _ = eglBindAPI(egl.EGL_OPENGL_ES_API);
    const dpy = eglGetPlatformDisplay(egl.EGL_PLATFORM_SURFACELESS_MESA, @ptrFromInt(0xE6418), null);
    try std.testing.expect(dpy != null);
    try std.testing.expectEqual(egl.EGL_TRUE, eglInitialize(dpy, null, null));
    defer _ = eglTerminate(dpy);
    const cfg = state.configHandle(0);
    const ctx = eglCreateContext(dpy, cfg, egl.EGL_NO_CONTEXT, null);
    const W: EGLint = 64;
    const pb_attribs = [_]EGLint{ egl.EGL_WIDTH, W, egl.EGL_HEIGHT, W, egl.EGL_NONE };
    const surf = eglCreatePbufferSurface(dpy, cfg, &pb_attribs);
    try std.testing.expectEqual(egl.EGL_TRUE, eglMakeCurrent(dpy, surf, surf, ctx));
    defer _ = eglDestroyContext(dpy, ctx);
    defer _ = eglDestroySurface(dpy, surf);
    defer _ = eglMakeCurrent(dpy, null, null, egl.EGL_NO_CONTEXT);

    // A shader that draws a fullscreen quad at a uniform NDC z (so we can plant a known depth)
    // in a uniform color.
    const vs_src: [*:0]const gles.GLchar =
        \\attribute vec2 position;
        \\uniform float uZ;
        \\void main() { gl_Position = vec4(position, uZ, 1.0); }
    ;
    const fs_src: [*:0]const gles.GLchar =
        \\precision mediump float;
        \\uniform vec4 uColor;
        \\void main() { gl_FragColor = uColor; }
    ;
    const vs = glCreateShader(gles.GL_VERTEX_SHADER);
    glShaderSource(vs, 1, &[_]?[*:0]const gles.GLchar{vs_src}, null);
    glCompileShader(vs);
    const fs = glCreateShader(gles.GL_FRAGMENT_SHADER);
    glShaderSource(fs, 1, &[_]?[*:0]const gles.GLchar{fs_src}, null);
    glCompileShader(fs);
    const prog = glCreateProgram();
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glBindAttribLocation(prog, 0, "position");
    glLinkProgram(prog);
    glUseProgram(prog);
    defer {
        glDeleteProgram(prog);
        glDeleteShader(vs);
        glDeleteShader(fs);
    }
    const z_loc = glGetUniformLocation(prog, "uZ");
    const c_loc = glGetUniformLocation(prog, "uColor");

    const tri = [_]f32{ -1, -1, 3, -1, -1, 3 };
    var vbo: gles.GLuint = 0;
    glGenBuffers(1, @ptrCast(&vbo));
    glBindBuffer(gles.GL_ARRAY_BUFFER, vbo);
    glBufferData(gles.GL_ARRAY_BUFFER, @sizeOf(@TypeOf(tri)), &tri, gles.GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, gles.GL_FLOAT, gles.GL_FALSE, 0, @ptrFromInt(0));
    glEnableVertexAttribArray(0);
    defer glDeleteBuffers(1, &[_]gles.GLuint{vbo});

    // Build an FBO with a color + a depth renderbuffer.
    const makeFbo = struct {
        fn f(w: gles.GLsizei) struct { fbo: gles.GLuint, color: gles.GLuint, depth: gles.GLuint } {
            var fbo: gles.GLuint = 0;
            glGenFramebuffers(1, @ptrCast(&fbo));
            glBindFramebuffer(gles.GL_FRAMEBUFFER, fbo);
            var color: gles.GLuint = 0;
            glGenRenderbuffers(1, @ptrCast(&color));
            glBindRenderbuffer(gles.GL_RENDERBUFFER, color);
            glRenderbufferStorage(gles.GL_RENDERBUFFER, gles.GL_RGBA8, w, w);
            glFramebufferRenderbuffer(gles.GL_FRAMEBUFFER, gles.GL_COLOR_ATTACHMENT0, gles.GL_RENDERBUFFER, color);
            var depth: gles.GLuint = 0;
            glGenRenderbuffers(1, @ptrCast(&depth));
            glBindRenderbuffer(gles.GL_RENDERBUFFER, depth);
            glRenderbufferStorage(gles.GL_RENDERBUFFER, gles.GL_DEPTH_COMPONENT16, w, w);
            glFramebufferRenderbuffer(gles.GL_FRAMEBUFFER, gles.GL_DEPTH_ATTACHMENT, gles.GL_RENDERBUFFER, depth);
            return .{ .fbo = fbo, .color = color, .depth = depth };
        }
    }.f;
    const src_fb = makeFbo(W);
    const dst_fb = makeFbo(W);

    glEnable(gles.GL_DEPTH_TEST);
    glViewport(0, 0, W, W);

    // Plant a near depth (z=-0.5) in src and a far depth (z=+0.5) in dst by drawing with
    // GL_ALWAYS (unconditional depth write).
    glDepthFunc(gles.GL_ALWAYS);
    glUniform4f(c_loc, 0, 0, 0, 1);
    glBindFramebuffer(gles.GL_FRAMEBUFFER, src_fb.fbo);
    glUniform1f(z_loc, 0.25);
    glClear(gles.GL_COLOR_BUFFER_BIT | gles.GL_DEPTH_BUFFER_BIT);
    glDrawArrays(gles.GL_TRIANGLES, 0, 3);
    glBindFramebuffer(gles.GL_FRAMEBUFFER, dst_fb.fbo);
    glUniform1f(z_loc, 0.75);
    glClear(gles.GL_COLOR_BUFFER_BIT | gles.GL_DEPTH_BUFFER_BIT);
    glDrawArrays(gles.GL_TRIANGLES, 0, 3);
    try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());

    // Blit the depth from src into dst: dst's depth becomes the near z=-0.5 depth.
    glBindFramebuffer(gles.GL_READ_FRAMEBUFFER, src_fb.fbo);
    glBindFramebuffer(gles.GL_DRAW_FRAMEBUFFER, dst_fb.fbo);
    glBlitFramebuffer(0, 0, W, W, 0, 0, W, W, gles.GL_DEPTH_BUFFER_BIT, gles.GL_NEAREST);
    try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());

    const s: *state.Surface = @ptrCast(@alignCast(surf.?));
    const dev = s.display.device.?;

    // Probe the blitted depth by drawing with GL_LESS (depth-write off so probes don't disturb it):
    glBindFramebuffer(gles.GL_FRAMEBUFFER, dst_fb.fbo);
    glDepthFunc(gles.GL_LESS);
    glDepthMask(gles.GL_FALSE);

    // A mid quad (z=0.5, farther than the blitted near z=0.25) must be rejected -> stays black.
    glUniform1f(z_loc, 0.5);
    glUniform4f(c_loc, 1, 0, 0, 1);
    glClear(gles.GL_COLOR_BUFFER_BIT);
    glDrawArrays(gles.GL_TRIANGLES, 0, 3);
    try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());
    // Read the dst FBO color back through a blit to the default fb (the surfaceless read path).
    glBindFramebuffer(gles.GL_READ_FRAMEBUFFER, dst_fb.fbo);
    glBindFramebuffer(gles.GL_DRAW_FRAMEBUFFER, 0);
    glBlitFramebuffer(0, 0, W, W, 0, 0, W, W, gles.GL_COLOR_BUFFER_BIT, gles.GL_NEAREST);
    const pxMid = try flushMap(dev, s);
    const ci = (32 * 64 + 32) * 4;
    try std.testing.expect(pxMid[ci + 0] < 60); // rejected -> not red

    // A nearer quad (z=0.1, in front of the blitted z=0.25) must pass -> red. This proves the
    // blitted depth really is the near value (a control against "everything is rejected").
    glBindFramebuffer(gles.GL_FRAMEBUFFER, dst_fb.fbo);
    glUniform1f(z_loc, 0.1);
    glClear(gles.GL_COLOR_BUFFER_BIT);
    glDrawArrays(gles.GL_TRIANGLES, 0, 3);
    glBindFramebuffer(gles.GL_READ_FRAMEBUFFER, dst_fb.fbo);
    glBindFramebuffer(gles.GL_DRAW_FRAMEBUFFER, 0);
    glBlitFramebuffer(0, 0, W, W, 0, 0, W, W, gles.GL_COLOR_BUFFER_BIT, gles.GL_NEAREST);
    const pxNear = try flushMap(dev, s);
    try std.testing.expect(pxNear[ci + 0] > 200); // passes -> red

    glDepthMask(gles.GL_TRUE);
    glDisable(gles.GL_DEPTH_TEST);
    glDeleteFramebuffers(1, &[_]gles.GLuint{src_fb.fbo});
    glDeleteFramebuffers(1, &[_]gles.GLuint{dst_fb.fbo});
    glDeleteRenderbuffers(1, &[_]gles.GLuint{ src_fb.color, src_fb.depth, dst_fb.color, dst_fb.depth });
}

test "EGL stencil-blit oracle: glBlitFramebuffer(GL_STENCIL_BUFFER_BIT) copies the stencil buffer between FBOs" {
    _ = eglBindAPI(egl.EGL_OPENGL_ES_API);
    const dpy = eglGetPlatformDisplay(egl.EGL_PLATFORM_SURFACELESS_MESA, @ptrFromInt(0xE6419), null);
    try std.testing.expect(dpy != null);
    try std.testing.expectEqual(egl.EGL_TRUE, eglInitialize(dpy, null, null));
    defer _ = eglTerminate(dpy);
    const cfg = state.configHandle(0);
    const ctx = eglCreateContext(dpy, cfg, egl.EGL_NO_CONTEXT, null);
    const W: EGLint = 64;
    const pb_attribs = [_]EGLint{ egl.EGL_WIDTH, W, egl.EGL_HEIGHT, W, egl.EGL_NONE };
    const surf = eglCreatePbufferSurface(dpy, cfg, &pb_attribs);
    try std.testing.expectEqual(egl.EGL_TRUE, eglMakeCurrent(dpy, surf, surf, ctx));
    defer _ = eglDestroyContext(dpy, ctx);
    defer _ = eglDestroySurface(dpy, surf);
    defer _ = eglMakeCurrent(dpy, null, null, egl.EGL_NO_CONTEXT);

    const vs_src: [*:0]const gles.GLchar =
        \\attribute vec2 position;
        \\void main() { gl_Position = vec4(position, 0.5, 1.0); }
    ;
    const fs_src: [*:0]const gles.GLchar =
        \\precision mediump float;
        \\uniform vec4 uColor;
        \\void main() { gl_FragColor = uColor; }
    ;
    const vs = glCreateShader(gles.GL_VERTEX_SHADER);
    glShaderSource(vs, 1, &[_]?[*:0]const gles.GLchar{vs_src}, null);
    glCompileShader(vs);
    const fs = glCreateShader(gles.GL_FRAGMENT_SHADER);
    glShaderSource(fs, 1, &[_]?[*:0]const gles.GLchar{fs_src}, null);
    glCompileShader(fs);
    const prog = glCreateProgram();
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glBindAttribLocation(prog, 0, "position");
    glLinkProgram(prog);
    glUseProgram(prog);
    defer {
        glDeleteProgram(prog);
        glDeleteShader(vs);
        glDeleteShader(fs);
    }
    const c_loc = glGetUniformLocation(prog, "uColor");

    const tri = [_]f32{ -1, -1, 3, -1, -1, 3 };
    var vbo: gles.GLuint = 0;
    glGenBuffers(1, @ptrCast(&vbo));
    glBindBuffer(gles.GL_ARRAY_BUFFER, vbo);
    glBufferData(gles.GL_ARRAY_BUFFER, @sizeOf(@TypeOf(tri)), &tri, gles.GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, gles.GL_FLOAT, gles.GL_FALSE, 0, @ptrFromInt(0));
    glEnableVertexAttribArray(0);
    defer glDeleteBuffers(1, &[_]gles.GLuint{vbo});

    const makeFbo = struct {
        fn f(w: gles.GLsizei) struct { fbo: gles.GLuint, color: gles.GLuint, stencil: gles.GLuint } {
            var fbo: gles.GLuint = 0;
            glGenFramebuffers(1, @ptrCast(&fbo));
            glBindFramebuffer(gles.GL_FRAMEBUFFER, fbo);
            var color: gles.GLuint = 0;
            glGenRenderbuffers(1, @ptrCast(&color));
            glBindRenderbuffer(gles.GL_RENDERBUFFER, color);
            glRenderbufferStorage(gles.GL_RENDERBUFFER, gles.GL_RGBA8, w, w);
            glFramebufferRenderbuffer(gles.GL_FRAMEBUFFER, gles.GL_COLOR_ATTACHMENT0, gles.GL_RENDERBUFFER, color);
            var stencil: gles.GLuint = 0;
            glGenRenderbuffers(1, @ptrCast(&stencil));
            glBindRenderbuffer(gles.GL_RENDERBUFFER, stencil);
            glRenderbufferStorage(gles.GL_RENDERBUFFER, gles.GL_STENCIL_INDEX8, w, w);
            glFramebufferRenderbuffer(gles.GL_FRAMEBUFFER, gles.GL_STENCIL_ATTACHMENT, gles.GL_RENDERBUFFER, stencil);
            return .{ .fbo = fbo, .color = color, .stencil = stencil };
        }
    }.f;
    const src_fb = makeFbo(W);
    const dst_fb = makeFbo(W);

    glViewport(0, 0, W, W);
    glEnable(gles.GL_STENCIL_TEST);

    // Plant stencil = 2 in src (REPLACE, always), and stencil = 0 in dst.
    glStencilOp(gles.GL_KEEP, gles.GL_KEEP, gles.GL_REPLACE);
    glUniform4f(c_loc, 0, 0, 0, 1);
    glBindFramebuffer(gles.GL_FRAMEBUFFER, src_fb.fbo);
    glStencilFunc(gles.GL_ALWAYS, 2, 0xFF);
    glClear(gles.GL_COLOR_BUFFER_BIT | gles.GL_STENCIL_BUFFER_BIT);
    glDrawArrays(gles.GL_TRIANGLES, 0, 3);
    glBindFramebuffer(gles.GL_FRAMEBUFFER, dst_fb.fbo);
    glStencilFunc(gles.GL_ALWAYS, 0, 0xFF);
    glClear(gles.GL_COLOR_BUFFER_BIT | gles.GL_STENCIL_BUFFER_BIT);
    glDrawArrays(gles.GL_TRIANGLES, 0, 3);
    try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());

    // Blit the stencil from src into dst: dst's stencil becomes 2.
    glBindFramebuffer(gles.GL_READ_FRAMEBUFFER, src_fb.fbo);
    glBindFramebuffer(gles.GL_DRAW_FRAMEBUFFER, dst_fb.fbo);
    glBlitFramebuffer(0, 0, W, W, 0, 0, W, W, gles.GL_STENCIL_BUFFER_BIT, gles.GL_NEAREST);
    try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());

    const s: *state.Surface = @ptrCast(@alignCast(surf.?));
    const dev = s.display.device.?;
    const ci = (32 * 64 + 32) * 4;

    // Probe: draw red where stencil == 2 (KEEP so the probe doesn't rewrite it). The blitted
    // stencil is 2 everywhere -> the red quad passes.
    glStencilOp(gles.GL_KEEP, gles.GL_KEEP, gles.GL_KEEP);
    glBindFramebuffer(gles.GL_FRAMEBUFFER, dst_fb.fbo);
    glStencilFunc(gles.GL_EQUAL, 2, 0xFF);
    glUniform4f(c_loc, 1, 0, 0, 1);
    glClear(gles.GL_COLOR_BUFFER_BIT);
    glDrawArrays(gles.GL_TRIANGLES, 0, 3);
    glBindFramebuffer(gles.GL_READ_FRAMEBUFFER, dst_fb.fbo);
    glBindFramebuffer(gles.GL_DRAW_FRAMEBUFFER, 0);
    glBlitFramebuffer(0, 0, W, W, 0, 0, W, W, gles.GL_COLOR_BUFFER_BIT, gles.GL_NEAREST);
    const pxEq = try flushMap(dev, s);
    try std.testing.expect(pxEq[ci + 0] > 200); // stencil==2 -> passes -> red

    // Control: stencil == 5 must NOT match (proves the value is 2, not "anything").
    glBindFramebuffer(gles.GL_FRAMEBUFFER, dst_fb.fbo);
    glStencilFunc(gles.GL_EQUAL, 5, 0xFF);
    glClear(gles.GL_COLOR_BUFFER_BIT);
    glDrawArrays(gles.GL_TRIANGLES, 0, 3);
    glBindFramebuffer(gles.GL_READ_FRAMEBUFFER, dst_fb.fbo);
    glBindFramebuffer(gles.GL_DRAW_FRAMEBUFFER, 0);
    glBlitFramebuffer(0, 0, W, W, 0, 0, W, W, gles.GL_COLOR_BUFFER_BIT, gles.GL_NEAREST);
    const pxNe = try flushMap(dev, s);
    try std.testing.expect(pxNe[ci + 0] < 60); // stencil!=5 -> rejected -> black

    glDisable(gles.GL_STENCIL_TEST);
    glDeleteFramebuffers(1, &[_]gles.GLuint{src_fb.fbo});
    glDeleteFramebuffers(1, &[_]gles.GLuint{dst_fb.fbo});
    glDeleteRenderbuffers(1, &[_]gles.GLuint{ src_fb.color, src_fb.stencil, dst_fb.color, dst_fb.stencil });
}

test "EGL color-mask oracle: glColorMask masks off channels (a white draw through R+A -> red)" {
    _ = eglBindAPI(egl.EGL_OPENGL_ES_API);
    const dpy = eglGetPlatformDisplay(egl.EGL_PLATFORM_SURFACELESS_MESA, @ptrFromInt(0xE6410), null);
    try std.testing.expect(dpy != null);
    try std.testing.expectEqual(egl.EGL_TRUE, eglInitialize(dpy, null, null));
    defer _ = eglTerminate(dpy);
    const cfg = state.configHandle(0); // RGBA8
    const ctx = eglCreateContext(dpy, cfg, egl.EGL_NO_CONTEXT, null);
    const W: EGLint = 64;
    const pb_attribs = [_]EGLint{ egl.EGL_WIDTH, W, egl.EGL_HEIGHT, W, egl.EGL_NONE };
    const surf = eglCreatePbufferSurface(dpy, cfg, &pb_attribs);
    try std.testing.expectEqual(egl.EGL_TRUE, eglMakeCurrent(dpy, surf, surf, ctx));
    defer _ = eglDestroyContext(dpy, ctx);
    defer _ = eglDestroySurface(dpy, surf);
    defer _ = eglMakeCurrent(dpy, null, null, egl.EGL_NO_CONTEXT);

    const vs_src: [*:0]const gles.GLchar =
        \\attribute vec3 position;
        \\void main() { gl_Position = vec4(position, 1.0); }
    ;
    const fs_src: [*:0]const gles.GLchar =
        \\precision mediump float;
        \\void main() { gl_FragColor = vec4(1.0, 1.0, 1.0, 1.0); }
    ;
    const vs = glCreateShader(gles.GL_VERTEX_SHADER);
    glShaderSource(vs, 1, &[_]?[*:0]const gles.GLchar{vs_src}, null);
    glCompileShader(vs);
    const fs = glCreateShader(gles.GL_FRAGMENT_SHADER);
    glShaderSource(fs, 1, &[_]?[*:0]const gles.GLchar{fs_src}, null);
    glCompileShader(fs);
    const prog = glCreateProgram();
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glBindAttribLocation(prog, 0, "position");
    glLinkProgram(prog);
    glUseProgram(prog);

    const Vtx = extern struct { x: f32, y: f32, z: f32 };
    const full = [4]Vtx{
        .{ .x = -1, .y = -1, .z = 0 }, .{ .x = 1, .y = -1, .z = 0 },
        .{ .x = -1, .y = 1, .z = 0 },  .{ .x = 1, .y = 1, .z = 0 },
    };
    var vbo: gles.GLuint = 0;
    glGenBuffers(1, @ptrCast(&vbo));
    glBindBuffer(gles.GL_ARRAY_BUFFER, vbo);
    glBufferData(gles.GL_ARRAY_BUFFER, @sizeOf(@TypeOf(full)), &full, gles.GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, gles.GL_FLOAT, gles.GL_FALSE, @sizeOf(Vtx), @ptrFromInt(0));
    glEnableVertexAttribArray(0);

    glClearColor(0, 0, 0, 1);
    glClear(gles.GL_COLOR_BUFFER_BIT);
    // Only R and A may write. G and B are masked. A white fragment leaves G=B=0 -> red.
    glColorMask(1, 0, 0, 1);
    glDrawArrays(gles.GL_TRIANGLE_STRIP, 0, 4);
    try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());

    const s: *state.Surface = @ptrCast(@alignCast(surf.?));
    const dev = s.display.device.?;
    const px = try flushMap(dev, s);
    const ci = (32 * 64 + 32) * 4;
    try std.testing.expect(px[ci + 0] > 200 and px[ci + 1] < 40 and px[ci + 2] < 40); // red (G,B masked)

    glDeleteBuffers(1, &[_]gles.GLuint{vbo});
    glDeleteProgram(prog);
    glDeleteShader(vs);
    glDeleteShader(fs);
}

test "FBO stencil oracle: a stencil renderbuffer clips a draw INTO an FBO (offscreen UI clip)" {
    _ = eglBindAPI(egl.EGL_OPENGL_ES_API);
    const dpy = eglGetPlatformDisplay(egl.EGL_PLATFORM_SURFACELESS_MESA, @ptrFromInt(0xE63F7), null);
    try std.testing.expectEqual(egl.EGL_TRUE, eglInitialize(dpy, null, null));
    defer _ = eglTerminate(dpy);
    const cfg = state.configHandle(0); // the FBO carries its own stencil renderbuffer, so any config works
    const ctx = eglCreateContext(dpy, cfg, egl.EGL_NO_CONTEXT, null);
    const W: EGLint = 64;
    const pb_attribs = [_]EGLint{ egl.EGL_WIDTH, W, egl.EGL_HEIGHT, W, egl.EGL_NONE };
    const surf = eglCreatePbufferSurface(dpy, cfg, &pb_attribs);
    try std.testing.expectEqual(egl.EGL_TRUE, eglMakeCurrent(dpy, surf, surf, ctx));
    defer _ = eglDestroyContext(dpy, ctx);
    defer _ = eglDestroySurface(dpy, surf);
    defer _ = eglMakeCurrent(dpy, null, null, egl.EGL_NO_CONTEXT);

    const vs_src: [*:0]const gles.GLchar =
        \\attribute vec3 position;
        \\uniform vec4 uColor;
        \\varying vec4 vColor;
        \\void main() { gl_Position = vec4(position, 1.0); vColor = uColor; }
    ;
    const fs_src: [*:0]const gles.GLchar =
        \\precision mediump float;
        \\varying vec4 vColor;
        \\void main() { gl_FragColor = vColor; }
    ;
    const vs = glCreateShader(gles.GL_VERTEX_SHADER);
    glShaderSource(vs, 1, &[_]?[*:0]const gles.GLchar{vs_src}, null);
    glCompileShader(vs);
    const fs = glCreateShader(gles.GL_FRAGMENT_SHADER);
    glShaderSource(fs, 1, &[_]?[*:0]const gles.GLchar{fs_src}, null);
    glCompileShader(fs);
    const prog = glCreateProgram();
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glBindAttribLocation(prog, 0, "position");
    glLinkProgram(prog);
    glUseProgram(prog);
    const col_loc = glGetUniformLocation(prog, "uColor");

    const Vtx = extern struct { x: f32, y: f32, z: f32 };
    var vbo: gles.GLuint = 0;
    glGenBuffers(1, @ptrCast(&vbo));
    glBindBuffer(gles.GL_ARRAY_BUFFER, vbo);
    glVertexAttribPointer(0, 3, gles.GL_FLOAT, gles.GL_FALSE, @sizeOf(Vtx), @ptrFromInt(0));
    glEnableVertexAttribArray(0);

    // An FBO with a color texture + a STENCIL_INDEX8 stencil renderbuffer.
    const TS: gles.GLsizei = 32;
    var tex: gles.GLuint = 0;
    glGenTextures(1, @ptrCast(&tex));
    glBindTexture(gles.GL_TEXTURE_2D, tex);
    glTexImage2D(gles.GL_TEXTURE_2D, 0, gles.GL_RGBA, TS, TS, 0, gles.GL_RGBA, gles.GL_UNSIGNED_BYTE, null);
    var srb: gles.GLuint = 0;
    glGenRenderbuffers(1, @ptrCast(&srb));
    glBindRenderbuffer(gles.GL_RENDERBUFFER, srb);
    glRenderbufferStorage(gles.GL_RENDERBUFFER, gles.GL_STENCIL_INDEX8, TS, TS);
    var fbo: gles.GLuint = 0;
    glGenFramebuffers(1, @ptrCast(&fbo));
    glBindFramebuffer(gles.GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(gles.GL_FRAMEBUFFER, gles.GL_COLOR_ATTACHMENT0, gles.GL_TEXTURE_2D, tex, 0);
    glFramebufferRenderbuffer(gles.GL_FRAMEBUFFER, gles.GL_STENCIL_ATTACHMENT, gles.GL_RENDERBUFFER, srb);
    try std.testing.expectEqual(gles.GL_FRAMEBUFFER_COMPLETE, glCheckFramebufferStatus(gles.GL_FRAMEBUFFER));

    glViewport(0, 0, TS, TS);
    glClearColor(0, 0, 0, 1);
    glClearStencil(0);
    glClear(gles.GL_COLOR_BUFFER_BIT | gles.GL_STENCIL_BUFFER_BIT);
    glEnable(gles.GL_STENCIL_TEST);

    // MASK: REPLACE stencil with 1 over the LEFT half.
    glStencilFunc(gles.GL_ALWAYS, 1, 0xFF);
    glStencilOp(gles.GL_KEEP, gles.GL_KEEP, gles.GL_REPLACE);
    glStencilMask(0xFF);
    const left = [4]Vtx{
        .{ .x = -1, .y = -1, .z = 0 }, .{ .x = 0, .y = -1, .z = 0 },
        .{ .x = -1, .y = 1, .z = 0 },  .{ .x = 0, .y = 1, .z = 0 },
    };
    glBufferData(gles.GL_ARRAY_BUFFER, @sizeOf(@TypeOf(left)), &left, gles.GL_STATIC_DRAW);
    glUniform4f(col_loc, 0, 0, 0, 1);
    glDrawArrays(gles.GL_TRIANGLE_STRIP, 0, 4);

    // Content: red fullscreen, clipped to stencil==1 (the left half).
    glStencilFunc(gles.GL_EQUAL, 1, 0xFF);
    glStencilOp(gles.GL_KEEP, gles.GL_KEEP, gles.GL_KEEP);
    const full = [4]Vtx{
        .{ .x = -1, .y = -1, .z = 0 }, .{ .x = 1, .y = -1, .z = 0 },
        .{ .x = -1, .y = 1, .z = 0 },  .{ .x = 1, .y = 1, .z = 0 },
    };
    glBufferData(gles.GL_ARRAY_BUFFER, @sizeOf(@TypeOf(full)), &full, gles.GL_STATIC_DRAW);
    glUniform4f(col_loc, 1, 0, 0, 1);
    glDrawArrays(gles.GL_TRIANGLE_STRIP, 0, 4);
    try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());

    var buf: [32 * 32 * 4]u8 = undefined;
    glReadPixels(0, 0, TS, TS, gles.GL_RGBA, gles.GL_UNSIGNED_BYTE, &buf);
    const at = struct {
        fn p(b: []const u8, x: usize, y: usize) [3]u8 {
            const i = (y * 32 + x) * 4;
            return .{ b[i], b[i + 1], b[i + 2] };
        }
    }.p;
    const l = at(&buf, 8, 16); // left half: stencil==1 -> red passed
    try std.testing.expect(l[0] > 200 and l[1] < 50 and l[2] < 50);
    const r = at(&buf, 24, 16); // right half: stencil==0 -> clipped, black
    try std.testing.expect(r[0] < 50 and r[1] < 50 and r[2] < 50);

    glDeleteRenderbuffers(1, &[_]gles.GLuint{srb});
    glDeleteFramebuffers(1, &[_]gles.GLuint{fbo});
    glDeleteTextures(1, &[_]gles.GLuint{tex});
    glDeleteBuffers(1, &[_]gles.GLuint{vbo});
    glDeleteProgram(prog);
    glDeleteShader(vs);
    glDeleteShader(fs);
}

test "FBO texture sampled by a flipped default-fb blit is non-black (the glmark2 desktop final blit)" {
    // The desktop scene renders its composite into an FBO color texture, then blits that
    // texture to the default framebuffer with a trivial pass-through shader. The default-fb
    // VS is y-flipped (GL bottom-left origin). The FBO texture is is_rt. This reproduces the
    // exact failure: flip + sampling an is_rt texture wrote black to the backbuffer.
    _ = eglBindAPI(egl.EGL_OPENGL_ES_API);
    const dpy = eglGetPlatformDisplay(egl.EGL_PLATFORM_SURFACELESS_MESA, @ptrFromInt(0xE6401), null);
    try std.testing.expectEqual(egl.EGL_TRUE, eglInitialize(dpy, null, null));
    defer _ = eglTerminate(dpy);
    const cfg = state.configHandle(0);
    const ctx = eglCreateContext(dpy, cfg, egl.EGL_NO_CONTEXT, null);
    const W: EGLint = 64;
    const pb_attribs = [_]EGLint{ egl.EGL_WIDTH, W, egl.EGL_HEIGHT, W, egl.EGL_NONE };
    const surf = eglCreatePbufferSurface(dpy, cfg, &pb_attribs);
    try std.testing.expectEqual(egl.EGL_TRUE, eglMakeCurrent(dpy, surf, surf, ctx));
    defer _ = eglDestroyContext(dpy, ctx);
    defer _ = eglDestroySurface(dpy, surf);
    defer _ = eglMakeCurrent(dpy, null, null, egl.EGL_NO_CONTEXT);

    // One blit program (samples uTex), used twice. Exactly like the desktop scene reuses
    // its composite/blit shader: first NOFLIP to render into an FBO texture, then FLIPPED to
    // blit that FBO texture to the default framebuffer. This is the faithful reproduction:
    // the failing path is the same program's flipped variant sampling an is_rt texture that
    // its own noflip variant produced.
    const bvs_src: [*:0]const gles.GLchar =
        \\attribute vec2 position;
        \\attribute vec2 aUV;
        \\varying vec2 vUV;
        \\void main() { gl_Position = vec4(position, 0.0, 1.0); vUV = aUV; }
    ;
    const bfs_src: [*:0]const gles.GLchar =
        \\precision mediump float;
        \\uniform sampler2D uTex;
        \\varying vec2 vUV;
        \\void main() { gl_FragColor = texture2D(uTex, vUV); }
    ;
    const bvs = glCreateShader(gles.GL_VERTEX_SHADER);
    glShaderSource(bvs, 1, &[_]?[*:0]const gles.GLchar{bvs_src}, null);
    glCompileShader(bvs);
    const bfs = glCreateShader(gles.GL_FRAGMENT_SHADER);
    glShaderSource(bfs, 1, &[_]?[*:0]const gles.GLchar{bfs_src}, null);
    glCompileShader(bfs);
    const bprog = glCreateProgram();
    glAttachShader(bprog, bvs);
    glAttachShader(bprog, bfs);
    glBindAttribLocation(bprog, 0, "position");
    glBindAttribLocation(bprog, 1, "aUV");
    glLinkProgram(bprog);
    glUseProgram(bprog);

    const QVtx = extern struct { x: f32, y: f32, u: f32, v: f32 };
    const quad = [6]QVtx{
        .{ .x = -1, .y = -1, .u = 0, .v = 0 },
        .{ .x = 1, .y = -1, .u = 1, .v = 0 },
        .{ .x = 1, .y = 1, .u = 1, .v = 1 },
        .{ .x = -1, .y = -1, .u = 0, .v = 0 },
        .{ .x = 1, .y = 1, .u = 1, .v = 1 },
        .{ .x = -1, .y = 1, .u = 0, .v = 1 },
    };
    var qvbo: gles.GLuint = 0;
    glGenBuffers(1, @ptrCast(&qvbo));
    glBindBuffer(gles.GL_ARRAY_BUFFER, qvbo);
    glBufferData(gles.GL_ARRAY_BUFFER, @sizeOf(@TypeOf(quad)), &quad, gles.GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, gles.GL_FLOAT, gles.GL_FALSE, @sizeOf(QVtx), @ptrFromInt(0));
    glVertexAttribPointer(1, 2, gles.GL_FLOAT, gles.GL_FALSE, @sizeOf(QVtx), @ptrFromInt(8));
    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);

    // A plain (non-rt) source texture: solid yellow 2x2.
    const src_px = [_]u8{
        255, 255, 0, 255, 255, 255, 0, 255,
        255, 255, 0, 255, 255, 255, 0, 255,
    };
    var src_tex: gles.GLuint = 0;
    glGenTextures(1, @ptrCast(&src_tex));
    glActiveTexture(gles.GL_TEXTURE0);
    glBindTexture(gles.GL_TEXTURE_2D, src_tex);
    glTexImage2D(gles.GL_TEXTURE_2D, 0, gles.GL_RGBA, 2, 2, 0, gles.GL_RGBA, gles.GL_UNSIGNED_BYTE, &src_px);
    glTexParameteri(gles.GL_TEXTURE_2D, gles.GL_TEXTURE_MIN_FILTER, @intCast(gles.GL_NEAREST));
    glTexParameteri(gles.GL_TEXTURE_2D, gles.GL_TEXTURE_MAG_FILTER, @intCast(gles.GL_NEAREST));

    // The is_rt FBO color texture (32x32).
    const TS: gles.GLsizei = 32;
    var rt_tex: gles.GLuint = 0;
    glGenTextures(1, @ptrCast(&rt_tex));
    glBindTexture(gles.GL_TEXTURE_2D, rt_tex);
    glTexImage2D(gles.GL_TEXTURE_2D, 0, gles.GL_RGBA, TS, TS, 0, gles.GL_RGBA, gles.GL_UNSIGNED_BYTE, null);
    glTexParameteri(gles.GL_TEXTURE_2D, gles.GL_TEXTURE_MIN_FILTER, @intCast(gles.GL_NEAREST));
    glTexParameteri(gles.GL_TEXTURE_2D, gles.GL_TEXTURE_MAG_FILTER, @intCast(gles.GL_NEAREST));
    var fbo: gles.GLuint = 0;
    glGenFramebuffers(1, @ptrCast(&fbo));
    glBindFramebuffer(gles.GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(gles.GL_FRAMEBUFFER, gles.GL_COLOR_ATTACHMENT0, gles.GL_TEXTURE_2D, rt_tex, 0);

    glUniform1i(glGetUniformLocation(bprog, "uTex"), 0);

    // --- Pass 1 (NOFLIP): sample the yellow source into the FBO texture. ---
    glActiveTexture(gles.GL_TEXTURE0);
    glBindTexture(gles.GL_TEXTURE_2D, src_tex);
    glViewport(0, 0, TS, TS);
    glClearColor(0, 0, 0, 1);
    glClear(gles.GL_COLOR_BUFFER_BIT);
    glDrawArrays(gles.GL_TRIANGLES, 0, 6);
    try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());

    // --- Pass 2 (FLIP): blit the FBO texture to the default framebuffer. ---
    glEnable(gles.GL_CULL_FACE); // the desktop blit runs with GL_CULL_FACE enabled
    glActiveTexture(gles.GL_TEXTURE0);
    glBindTexture(gles.GL_TEXTURE_2D, rt_tex); // the is_rt FBO texture
    glBindFramebuffer(gles.GL_FRAMEBUFFER, 0); // default fb -> flipped VS path
    glViewport(0, 0, W, W);
    glClearColor(0, 0, 0, 1);
    glClear(gles.GL_COLOR_BUFFER_BIT);
    glDrawArrays(gles.GL_TRIANGLES, 0, 6);
    try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());

    const s: *state.Surface = @ptrCast(@alignCast(surf.?));
    const dev = s.display.device.?;
    const px = try flushMap(dev, s);
    const c = (@as(usize, 32) * 64 + 32) * 4;
    // The blit sampled the yellow FBO texture: center must be non-black (R and G high).
    try std.testing.expect(px[c + 0] > 0 or px[c + 1] > 0 or px[c + 2] > 0);

    glDeleteProgram(bprog);
}

test "float RT sampling: an rgba16f RT rendered with HDR 2.0 samples back UNCLAMPED in a 2nd pass (software)" {
    // The HDR post-processing building block: pass 1 renders a >1.0 value into a float FBO, pass 2
    // samples that float texture and scales by 0.5. If sampling preserved the HDR 2.0, the 8-bit
    // result is 1.0 -> 255. If the float RT had clamped to 1.0, it would be 0.5 -> 128.
    _ = eglBindAPI(egl.EGL_OPENGL_ES_API);
    const dpy = eglGetPlatformDisplay(egl.EGL_PLATFORM_SURFACELESS_MESA, @ptrFromInt(0xE61D6), null);
    try std.testing.expectEqual(egl.EGL_TRUE, eglInitialize(dpy, null, null));
    defer _ = eglTerminate(dpy);
    const cfg = state.configHandle(0);
    const ctx = eglCreateContext(dpy, cfg, egl.EGL_NO_CONTEXT, null);
    const W: EGLint = 64;
    const pb_attribs = [_]EGLint{ egl.EGL_WIDTH, W, egl.EGL_HEIGHT, W, egl.EGL_NONE };
    const surf = eglCreatePbufferSurface(dpy, cfg, &pb_attribs);
    try std.testing.expectEqual(egl.EGL_TRUE, eglMakeCurrent(dpy, surf, surf, ctx));
    defer _ = eglDestroyContext(dpy, ctx);
    defer _ = eglDestroySurface(dpy, surf);
    defer _ = eglMakeCurrent(dpy, null, null, egl.EGL_NO_CONTEXT);

    const vs_src: [*:0]const gles.GLchar =
        \\attribute vec2 position;
        \\attribute vec2 aUV;
        \\varying vec2 vUV;
        \\void main() { gl_Position = vec4(position, 0.0, 1.0); vUV = aUV; }
    ;
    // Pass 1 fills the float RT with an HDR constant.
    const fill_fs: [*:0]const gles.GLchar =
        \\precision highp float;
        \\void main() { gl_FragColor = vec4(2.0, 0.5, 0.25, 1.0); }
    ;
    // Pass 2 samples the float RT and halves it (2.0 -> 1.0 if HDR survived; 1.0 -> 0.5 if clamped).
    const sample_fs: [*:0]const gles.GLchar =
        \\precision highp float;
        \\uniform sampler2D uTex;
        \\varying vec2 vUV;
        \\void main() { gl_FragColor = texture2D(uTex, vUV) * 0.5; }
    ;
    const mkProg = struct {
        fn f(vsrc: [*:0]const gles.GLchar, fsrc: [*:0]const gles.GLchar) gles.GLuint {
            const v = glCreateShader(gles.GL_VERTEX_SHADER);
            glShaderSource(v, 1, &[_]?[*:0]const gles.GLchar{vsrc}, null);
            glCompileShader(v);
            const fr = glCreateShader(gles.GL_FRAGMENT_SHADER);
            glShaderSource(fr, 1, &[_]?[*:0]const gles.GLchar{fsrc}, null);
            glCompileShader(fr);
            const p = glCreateProgram();
            glAttachShader(p, v);
            glAttachShader(p, fr);
            glBindAttribLocation(p, 0, "position");
            glBindAttribLocation(p, 1, "aUV");
            glLinkProgram(p);
            return p;
        }
    }.f;
    const fill_prog = mkProg(vs_src, fill_fs);
    const sample_prog = mkProg(vs_src, sample_fs);

    const QVtx = extern struct { x: f32, y: f32, u: f32, v: f32 };
    const quad = [6]QVtx{
        .{ .x = -1, .y = -1, .u = 0, .v = 0 }, .{ .x = 1, .y = -1, .u = 1, .v = 0 },
        .{ .x = 1, .y = 1, .u = 1, .v = 1 },   .{ .x = -1, .y = -1, .u = 0, .v = 0 },
        .{ .x = 1, .y = 1, .u = 1, .v = 1 },   .{ .x = -1, .y = 1, .u = 0, .v = 1 },
    };
    var qvbo: gles.GLuint = 0;
    glGenBuffers(1, @ptrCast(&qvbo));
    glBindBuffer(gles.GL_ARRAY_BUFFER, qvbo);
    glBufferData(gles.GL_ARRAY_BUFFER, @sizeOf(@TypeOf(quad)), &quad, gles.GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, gles.GL_FLOAT, gles.GL_FALSE, @sizeOf(QVtx), @ptrFromInt(0));
    glVertexAttribPointer(1, 2, gles.GL_FLOAT, gles.GL_FALSE, @sizeOf(QVtx), @ptrFromInt(8));
    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);

    // An rgba16f FBO color texture (half-float, sampled back in pass 2).
    const TS: gles.GLsizei = 32;
    var rt_tex: gles.GLuint = 0;
    glGenTextures(1, @ptrCast(&rt_tex));
    glBindTexture(gles.GL_TEXTURE_2D, rt_tex);
    glTexImage2D(gles.GL_TEXTURE_2D, 0, gles.GL_RGBA, TS, TS, 0, gles.GL_RGBA, gles.GL_HALF_FLOAT_OES, null);
    glTexParameteri(gles.GL_TEXTURE_2D, gles.GL_TEXTURE_MIN_FILTER, @intCast(gles.GL_NEAREST));
    glTexParameteri(gles.GL_TEXTURE_2D, gles.GL_TEXTURE_MAG_FILTER, @intCast(gles.GL_NEAREST));
    var fbo: gles.GLuint = 0;
    glGenFramebuffers(1, @ptrCast(&fbo));
    glBindFramebuffer(gles.GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(gles.GL_FRAMEBUFFER, gles.GL_COLOR_ATTACHMENT0, gles.GL_TEXTURE_2D, rt_tex, 0);

    // --- Pass 1: fill the float RT with the HDR constant (2.0 red). ---
    glUseProgram(fill_prog);
    glViewport(0, 0, TS, TS);
    glDrawArrays(gles.GL_TRIANGLES, 0, 6);
    try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());

    // --- Pass 2: sample the float RT * 0.5 into the default framebuffer. ---
    glUseProgram(sample_prog);
    glUniform1i(glGetUniformLocation(sample_prog, "uTex"), 0);
    glActiveTexture(gles.GL_TEXTURE0);
    glBindTexture(gles.GL_TEXTURE_2D, rt_tex);
    glBindFramebuffer(gles.GL_FRAMEBUFFER, 0);
    glViewport(0, 0, W, W);
    glClearColor(0, 0, 0, 1);
    glClear(gles.GL_COLOR_BUFFER_BIT);
    glDrawArrays(gles.GL_TRIANGLES, 0, 6);
    try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());

    const s: *state.Surface = @ptrCast(@alignCast(surf.?));
    const dev = s.display.device.?;
    const px = try flushMap(dev, s);
    const c = (@as(usize, 32) * 64 + 32) * 4;
    // Sampled HDR 2.0 * 0.5 = 1.0 -> 255 red. A clamped (1.0) RT would give 0.5 -> ~128.
    try std.testing.expect(px[c + 0] > 240); // red survived as full-scale (NOT ~128)
    try std.testing.expectApproxEqAbs(@as(f32, 64), @as(f32, @floatFromInt(px[c + 1])), 8); // 0.5*0.5=0.25 -> ~64

    glDeleteProgram(fill_prog);
    glDeleteProgram(sample_prog);
}

test "eglDestroyContext on a current context defers destruction (the glmark2 scene-switch bug)" {
    // EGL 1.5 3.7.1: destroying a context that is current to a thread must defer the
    // teardown until it is released. glmark2 destroys the previous scene's context while
    // it is still current, then keeps issuing GL (the GLVND loader keeps routing to us)
    // before binding the next scene's context. A premature unbind made the next
    // glLinkProgram find no current context -> "Set up failed". This guards the fix.
    _ = eglBindAPI(egl.EGL_OPENGL_ES_API);
    const dpy = eglGetPlatformDisplay(egl.EGL_PLATFORM_SURFACELESS_MESA, @ptrFromInt(0xDEFE12), null);
    try std.testing.expect(dpy != null);
    try std.testing.expectEqual(egl.EGL_TRUE, eglInitialize(dpy, null, null));
    defer _ = eglTerminate(dpy);

    const cfg = state.configHandle(0);
    const pb_attribs = [_]EGLint{ egl.EGL_WIDTH, 32, egl.EGL_HEIGHT, 32, egl.EGL_NONE };

    const ctx_a = eglCreateContext(dpy, cfg, egl.EGL_NO_CONTEXT, null);
    try std.testing.expect(ctx_a != egl.EGL_NO_CONTEXT);
    const surf = eglCreatePbufferSurface(dpy, cfg, &pb_attribs);
    try std.testing.expect(surf != egl.EGL_NO_SURFACE);
    try std.testing.expectEqual(egl.EGL_TRUE, eglMakeCurrent(dpy, surf, surf, ctx_a));

    // Destroy ctx_a while it is current: must succeed, must stay current + usable.
    try std.testing.expectEqual(egl.EGL_TRUE, eglDestroyContext(dpy, ctx_a));
    try std.testing.expectEqual(ctx_a, eglGetCurrentContext()); // still current (deferred)
    // GL still works on the deferred-deleted-but-current context (this is the regression:
    // before the fix the device was gone and the current binding was null).
    glClearColor(0.0, 1.0, 0.0, 1.0);
    glClear(gles.GL_COLOR_BUFFER_BIT);
    try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());

    // Bind a different context (the next scene): this releases ctx_a, reaping it now.
    const ctx_b = eglCreateContext(dpy, cfg, egl.EGL_NO_CONTEXT, null);
    try std.testing.expect(ctx_b != egl.EGL_NO_CONTEXT);
    try std.testing.expectEqual(egl.EGL_TRUE, eglMakeCurrent(dpy, surf, surf, ctx_b));
    try std.testing.expectEqual(ctx_b, eglGetCurrentContext());
    // A link/clear on the freshly-bound context succeeds (no leftover desync).
    glClear(gles.GL_COLOR_BUFFER_BIT);
    try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());

    try std.testing.expectEqual(egl.EGL_TRUE, eglMakeCurrent(dpy, null, null, egl.EGL_NO_CONTEXT));
    try std.testing.expectEqual(egl.EGL_TRUE, eglDestroySurface(dpy, surf));
    try std.testing.expectEqual(egl.EGL_TRUE, eglDestroyContext(dpy, ctx_b));
}

// EGL M3 deterministic oracle: an EGL client creates a context + a pbuffer
// surface, loads the RGB gradient triangle's SPIR-V VS+FS via glShaderBinary
// (GL_ARB_gl_spirv), uploads a position+color vertex buffer, glDrawArrays, then
// reads the pbuffer's HAL backbuffer back and asserts an off-center interior pixel
// is the interpolated barycentric blend (not the clear color, not any single vertex
// color). Proves the full EGL -> GLES2 shader+attribute+draw -> HAL (software
// SPIR-V JIT + rasterizer) path end to end, no display needed.
const tri_vert_spv align(4) = @embedFile("testdata/tri.vert.spv").*;
const tri_frag_spv align(4) = @embedFile("testdata/tri.frag.spv").*;

test "EGL pbuffer GLES2 triangle reads back the interpolated pixel (EGL -> GLES2 -> HAL draw path)" {
    _ = eglBindAPI(egl.EGL_OPENGL_ES_API);

    const dpy = eglGetPlatformDisplay(egl.EGL_PLATFORM_SURFACELESS_MESA, @ptrFromInt(0xE63A1), null);
    try std.testing.expect(dpy != null);
    try std.testing.expectEqual(egl.EGL_TRUE, eglInitialize(dpy, null, null));
    defer _ = eglTerminate(dpy);

    const cfg = state.configHandle(0); // RGBA8
    const ctx = eglCreateContext(dpy, cfg, egl.EGL_NO_CONTEXT, null);
    try std.testing.expect(ctx != egl.EGL_NO_CONTEXT);

    const W: EGLint = 64;
    const H: EGLint = 64;
    const pb_attribs = [_]EGLint{ egl.EGL_WIDTH, W, egl.EGL_HEIGHT, H, egl.EGL_NONE };
    const surf = eglCreatePbufferSurface(dpy, cfg, &pb_attribs);
    try std.testing.expect(surf != egl.EGL_NO_SURFACE);
    try std.testing.expectEqual(egl.EGL_TRUE, eglMakeCurrent(dpy, surf, surf, ctx));

    // --- Build the GLES2 program from the SPIR-V gradient-triangle VS+FS. -------
    const vs = glCreateShader(gles.GL_VERTEX_SHADER);
    try std.testing.expect(vs != 0);
    glShaderBinary(1, &[_]gles.GLuint{vs}, gles.GL_SHADER_BINARY_FORMAT_SPIR_V, &tri_vert_spv, @intCast(tri_vert_spv.len));
    glSpecializeShader(vs, "main", 0, null, null);
    var vs_ok: gles.GLint = 0;
    glGetShaderiv(vs, gles.GL_COMPILE_STATUS, &vs_ok);
    try std.testing.expectEqual(@as(gles.GLint, gles.GL_TRUE), vs_ok);

    const fs = glCreateShader(gles.GL_FRAGMENT_SHADER);
    glShaderBinary(1, &[_]gles.GLuint{fs}, gles.GL_SHADER_BINARY_FORMAT_SPIR_V, &tri_frag_spv, @intCast(tri_frag_spv.len));
    glSpecializeShader(fs, "main", 0, null, null);

    const prog = glCreateProgram();
    try std.testing.expect(prog != 0);
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glLinkProgram(prog);
    var link_ok: gles.GLint = 0;
    glGetProgramiv(prog, gles.GL_LINK_STATUS, &link_ok);
    try std.testing.expectEqual(@as(gles.GLint, gles.GL_TRUE), link_ok);
    glUseProgram(prog);

    // --- The vertex buffer: pos(vec2, loc 0) + color(vec3, loc 1), 20-byte stride.
    const Vtx = extern struct { x: f32, y: f32, r: f32, g: f32, b: f32 };
    const tri = [3]Vtx{
        .{ .x = 0.0, .y = 0.8, .r = 1, .g = 0, .b = 0 }, // top    -> red
        .{ .x = -0.8, .y = -0.8, .r = 0, .g = 1, .b = 0 }, // bl   -> green
        .{ .x = 0.8, .y = -0.8, .r = 0, .g = 0, .b = 1 }, // br    -> blue
    };
    var vbo: gles.GLuint = 0;
    glGenBuffers(1, @ptrCast(&vbo));
    try std.testing.expect(vbo != 0);
    glBindBuffer(gles.GL_ARRAY_BUFFER, vbo);
    glBufferData(gles.GL_ARRAY_BUFFER, @sizeOf(@TypeOf(tri)), &tri, gles.GL_STATIC_DRAW);

    glVertexAttribPointer(0, 2, gles.GL_FLOAT, gles.GL_FALSE, @sizeOf(Vtx), @ptrFromInt(0));
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, gles.GL_FLOAT, gles.GL_FALSE, @sizeOf(Vtx), @ptrFromInt(8));
    glEnableVertexAttribArray(1);

    // Clear to black, then draw the triangle over it.
    glClearColor(0, 0, 0, 1);
    glClear(gles.GL_COLOR_BUFFER_BIT);
    glDrawArrays(gles.GL_TRIANGLES, 0, 3);
    try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());

    // --- Read back + assert the interpolated triangle. -------------------------
    const s: *state.Surface = @ptrCast(@alignCast(surf.?));
    const dev = s.display.device.?;
    const px = try flushMap(dev, s);

    // Center pixel (32,32): a barycentric blend of all three vertex colors -> all
    // three channels nonzero, and NOT equal to any pure vertex color (1,0,0)/(0,1,0)/
    // (0,0,1). This is the load-bearing proof the GLES2 shader+attribute draw path
    // interpolated through the SPIR-V VS/FS.
    const c = (32 * 64 + 32) * 4;
    const cr = px[c + 0];
    const cg = px[c + 1];
    const cb = px[c + 2];
    try std.testing.expect(cr > 0 and cg > 0 and cb > 0); // a true blend, not clear black
    // Not a single vertex color (no channel is ~255 while the others are ~0).
    const is_pure = (cr > 250 and cg < 5 and cb < 5) or (cg > 250 and cr < 5 and cb < 5) or (cb > 250 and cr < 5 and cg < 5);
    try std.testing.expect(!is_pure);

    // A top corner pixel (0,0) is outside the triangle -> still the clear color (black).
    try std.testing.expectEqual(@as(u8, 0), px[0]);
    try std.testing.expectEqual(@as(u8, 0), px[1]);
    try std.testing.expectEqual(@as(u8, 0), px[2]);

    // The gradient VARIES across the triangle. The EGL/GLES path negates gl_Position.y for
    // the GL bottom-left framebuffer origin, so the red (NDC +y) vertex lands near the TOP of
    // the window. A pixel toward the top (32,14) is red-heavier than the center.
    const lo = (14 * 64 + 32) * 4;
    try std.testing.expect(px[lo + 0] > cr); // redder than center (red vertex -> window top)
    // And a pixel toward the bottom of the window (near the green/blue edge) is less red:
    // sample (32,46).
    const hi = (46 * 64 + 32) * 4;
    try std.testing.expect(px[hi + 0] < cr); // less red than center

    // Tear down.
    glDeleteBuffers(1, &[_]gles.GLuint{vbo});
    glDeleteProgram(prog);
    glDeleteShader(vs);
    glDeleteShader(fs);
    try std.testing.expectEqual(egl.EGL_TRUE, eglMakeCurrent(dpy, null, null, egl.EGL_NO_CONTEXT));
    try std.testing.expectEqual(egl.EGL_TRUE, eglDestroySurface(dpy, surf));
    try std.testing.expectEqual(egl.EGL_TRUE, eglDestroyContext(dpy, ctx));
}

// The same gradient triangle, but the VS+FS are compiled from GLSL ES 1.00 SOURCE
// through glShaderSource + glCompileShader (the Vulcan GLSL front end) instead of a
// pre-baked SPIR-V binary. The pbuffer must read back the SAME interpolated interior
// pixel as the SPIR-V-binary M3 oracle above, proving the GLSL-source -> SPIR-V ->
// Vulcan JIT -> HAL render slice works end to end. This is the GLSL-ES->SPIR-V M1 gate.
const tri_vert_glsl =
    \\attribute vec2 aPos;
    \\attribute vec3 aColor;
    \\varying vec3 vColor;
    \\void main() {
    \\    gl_Position = vec4(aPos, 0.0, 1.0);
    \\    vColor = aColor;
    \\}
;
const tri_frag_glsl =
    \\precision mediump float;
    \\varying vec3 vColor;
    \\void main() {
    \\    gl_FragColor = vec4(vColor, 1.0);
    \\}
;

test "EGL pbuffer GLES2 triangle compiled from GLSL ES SOURCE reads back the interpolated pixel" {
    _ = eglBindAPI(egl.EGL_OPENGL_ES_API);

    const dpy = eglGetPlatformDisplay(egl.EGL_PLATFORM_SURFACELESS_MESA, @ptrFromInt(0xE63A2), null);
    try std.testing.expect(dpy != null);
    try std.testing.expectEqual(egl.EGL_TRUE, eglInitialize(dpy, null, null));
    defer _ = eglTerminate(dpy);

    const cfg = state.configHandle(0); // RGBA8
    const ctx = eglCreateContext(dpy, cfg, egl.EGL_NO_CONTEXT, null);
    try std.testing.expect(ctx != egl.EGL_NO_CONTEXT);

    const W: EGLint = 64;
    const H: EGLint = 64;
    const pb_attribs = [_]EGLint{ egl.EGL_WIDTH, W, egl.EGL_HEIGHT, H, egl.EGL_NONE };
    const surf = eglCreatePbufferSurface(dpy, cfg, &pb_attribs);
    try std.testing.expect(surf != egl.EGL_NO_SURFACE);
    try std.testing.expectEqual(egl.EGL_TRUE, eglMakeCurrent(dpy, surf, surf, ctx));

    // --- Build the GLES2 program from GLSL ES 1.00 SOURCE (glShaderSource + glCompileShader). ---
    const vs = glCreateShader(gles.GL_VERTEX_SHADER);
    try std.testing.expect(vs != 0);
    glShaderSource(vs, 1, &[_]?[*:0]const gles.GLchar{tri_vert_glsl}, null);
    glCompileShader(vs);
    var vs_ok: gles.GLint = 0;
    glGetShaderiv(vs, gles.GL_COMPILE_STATUS, &vs_ok);
    if (vs_ok != gles.GL_TRUE) {
        var log_buf: [256]gles.GLchar = undefined;
        glGetShaderInfoLog(vs, log_buf.len, null, &log_buf);
        std.debug.print("VS compile failed: {s}\n", .{std.mem.sliceTo(&log_buf, 0)});
    }
    try std.testing.expectEqual(@as(gles.GLint, gles.GL_TRUE), vs_ok);

    const fs = glCreateShader(gles.GL_FRAGMENT_SHADER);
    glShaderSource(fs, 1, &[_]?[*:0]const gles.GLchar{tri_frag_glsl}, null);
    glCompileShader(fs);
    var fs_ok: gles.GLint = 0;
    glGetShaderiv(fs, gles.GL_COMPILE_STATUS, &fs_ok);
    try std.testing.expectEqual(@as(gles.GLint, gles.GL_TRUE), fs_ok);

    const prog = glCreateProgram();
    try std.testing.expect(prog != 0);
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glLinkProgram(prog);
    var link_ok: gles.GLint = 0;
    glGetProgramiv(prog, gles.GL_LINK_STATUS, &link_ok);
    try std.testing.expectEqual(@as(gles.GLint, gles.GL_TRUE), link_ok);
    glUseProgram(prog);

    // --- The vertex buffer: pos(vec2, loc 0) + color(vec3, loc 1), 20-byte stride. ---
    const Vtx = extern struct { x: f32, y: f32, r: f32, g: f32, b: f32 };
    const tri = [3]Vtx{
        .{ .x = 0.0, .y = 0.8, .r = 1, .g = 0, .b = 0 }, // top    -> red
        .{ .x = -0.8, .y = -0.8, .r = 0, .g = 1, .b = 0 }, // bl   -> green
        .{ .x = 0.8, .y = -0.8, .r = 0, .g = 0, .b = 1 }, // br    -> blue
    };
    var vbo: gles.GLuint = 0;
    glGenBuffers(1, @ptrCast(&vbo));
    glBindBuffer(gles.GL_ARRAY_BUFFER, vbo);
    glBufferData(gles.GL_ARRAY_BUFFER, @sizeOf(@TypeOf(tri)), &tri, gles.GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, gles.GL_FLOAT, gles.GL_FALSE, @sizeOf(Vtx), @ptrFromInt(0));
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, gles.GL_FLOAT, gles.GL_FALSE, @sizeOf(Vtx), @ptrFromInt(8));
    glEnableVertexAttribArray(1);

    glClearColor(0, 0, 0, 1);
    glClear(gles.GL_COLOR_BUFFER_BIT);
    glDrawArrays(gles.GL_TRIANGLES, 0, 3);
    try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());

    // --- Read back + assert the SAME interpolated triangle as the SPIR-V oracle. ---
    const s: *state.Surface = @ptrCast(@alignCast(surf.?));
    const dev = s.display.device.?;
    const px = try flushMap(dev, s);

    const c = (32 * 64 + 32) * 4;
    const cr = px[c + 0];
    const cg = px[c + 1];
    const cb = px[c + 2];
    try std.testing.expect(cr > 0 and cg > 0 and cb > 0); // a true blend, not clear black
    const is_pure = (cr > 250 and cg < 5 and cb < 5) or (cg > 250 and cr < 5 and cb < 5) or (cb > 250 and cr < 5 and cg < 5);
    try std.testing.expect(!is_pure);
    // Corner (0,0) outside the triangle stays clear black.
    try std.testing.expectEqual(@as(u8, 0), px[0]);
    try std.testing.expectEqual(@as(u8, 0), px[1]);
    try std.testing.expectEqual(@as(u8, 0), px[2]);
    // The gradient varies: with the GL bottom-left origin (the EGL/GLES y-flip), the red
    // (NDC +y) vertex lands near the window TOP, so a top pixel (32,14) is redder than center
    // and a bottom pixel (32,46) is less red.
    const lo = (14 * 64 + 32) * 4;
    try std.testing.expect(px[lo + 0] > cr);
    const hi = (46 * 64 + 32) * 4;
    try std.testing.expect(px[hi + 0] < cr);

    glDeleteBuffers(1, &[_]gles.GLuint{vbo});
    glDeleteProgram(prog);
    glDeleteShader(vs);
    glDeleteShader(fs);
    try std.testing.expectEqual(egl.EGL_TRUE, eglMakeCurrent(dpy, null, null, egl.EGL_NO_CONTEXT));
    try std.testing.expectEqual(egl.EGL_TRUE, eglDestroySurface(dpy, surf));
    try std.testing.expectEqual(egl.EGL_TRUE, eglDestroyContext(dpy, ctx));
}

test "EGL pbuffer depth+uniform oracle: MVP transforms geometry, near triangle occludes far (es2gears path)" {
    _ = eglBindAPI(egl.EGL_OPENGL_ES_API);
    const dpy = eglGetPlatformDisplay(egl.EGL_PLATFORM_SURFACELESS_MESA, @ptrFromInt(0xE63D7), null);
    try std.testing.expect(dpy != null);
    try std.testing.expectEqual(egl.EGL_TRUE, eglInitialize(dpy, null, null));
    defer _ = eglTerminate(dpy);

    const cfg = state.configHandle(1); // RGBA8 + D24 (has a depth buffer)
    const ctx = eglCreateContext(dpy, cfg, egl.EGL_NO_CONTEXT, null);
    try std.testing.expect(ctx != egl.EGL_NO_CONTEXT);
    const W: EGLint = 64;
    const H: EGLint = 64;
    const pb_attribs = [_]EGLint{ egl.EGL_WIDTH, W, egl.EGL_HEIGHT, H, egl.EGL_NONE };
    const surf = eglCreatePbufferSurface(dpy, cfg, &pb_attribs);
    try std.testing.expect(surf != egl.EGL_NO_SURFACE);
    try std.testing.expectEqual(egl.EGL_TRUE, eglMakeCurrent(dpy, surf, surf, ctx));

    // VS: an MVP-transformed position + a uniform color forwarded as a varying. This is the
    // es2gears shape (all uniforms in the VS). The FS only reads varyings. The uniforms lower
    // through Vulcan into one default-uniform-block the HAL binds as a UBO.
    const vs_src: [*:0]const gles.GLchar =
        \\attribute vec3 position;
        \\uniform mat4 uMVP;
        \\uniform vec4 uColor;
        \\varying vec4 vColor;
        \\void main() { gl_Position = uMVP * vec4(position, 1.0); vColor = uColor; }
    ;
    const fs_src: [*:0]const gles.GLchar =
        \\precision mediump float;
        \\varying vec4 vColor;
        \\void main() { gl_FragColor = vColor; }
    ;
    const vs = glCreateShader(gles.GL_VERTEX_SHADER);
    glShaderSource(vs, 1, &[_]?[*:0]const gles.GLchar{vs_src}, null);
    glCompileShader(vs);
    var vs_ok: gles.GLint = 0;
    glGetShaderiv(vs, gles.GL_COMPILE_STATUS, &vs_ok);
    if (vs_ok != gles.GL_TRUE) {
        var lb: [256]gles.GLchar = undefined;
        glGetShaderInfoLog(vs, lb.len, null, &lb);
        std.debug.print("VS: {s}\n", .{std.mem.sliceTo(&lb, 0)});
    }
    try std.testing.expectEqual(@as(gles.GLint, gles.GL_TRUE), vs_ok);
    const fs = glCreateShader(gles.GL_FRAGMENT_SHADER);
    glShaderSource(fs, 1, &[_]?[*:0]const gles.GLchar{fs_src}, null);
    glCompileShader(fs);

    const prog = glCreateProgram();
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glBindAttribLocation(prog, 0, "position");
    glLinkProgram(prog);
    var link_ok: gles.GLint = 0;
    glGetProgramiv(prog, gles.GL_LINK_STATUS, &link_ok);
    try std.testing.expectEqual(@as(gles.GLint, gles.GL_TRUE), link_ok);
    glUseProgram(prog);

    // The uniform locations must resolve.
    const mvp_loc = glGetUniformLocation(prog, "uMVP");
    const col_loc = glGetUniformLocation(prog, "uColor");
    try std.testing.expect(mvp_loc >= 0);
    try std.testing.expect(col_loc >= 0);

    // A full-screen-ish quad (two triangles) as a TRIANGLE_STRIP (es2gears uses strips).
    // position z carries depth; the MVP is identity so screen.z = clamp(z,0,1).
    const Vtx = extern struct { x: f32, y: f32, z: f32 };
    var vbo: gles.GLuint = 0;
    glGenBuffers(1, @ptrCast(&vbo));
    glBindBuffer(gles.GL_ARRAY_BUFFER, vbo);
    glVertexAttribPointer(0, 3, gles.GL_FLOAT, gles.GL_FALSE, @sizeOf(Vtx), @ptrFromInt(0));
    glEnableVertexAttribArray(0);
    glDisableVertexAttribArray(1); // this shader uses only attribute 0

    glEnable(gles.GL_DEPTH_TEST);
    glDepthFunc(gles.GL_LESS);

    // Identity MVP (column-major). The uniform MVP actually transforms: with identity the
    // known vertex (0.5,0.5) lands at screen (48,48). Also verify a scaled MVP moves it.
    const identity = [16]f32{ 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1 };
    glUniformMatrix4fv(mvp_loc, 1, gles.GL_FALSE, &identity);

    glClearColor(0, 0, 0, 1);
    glClear(gles.GL_COLOR_BUFFER_BIT | gles.GL_DEPTH_BUFFER_BIT);

    // Far quad: depth z=0.8, red. Covers the whole NDC square as a strip (bl,br,tl,tr).
    const far = [4]Vtx{
        .{ .x = -0.9, .y = -0.9, .z = 0.8 },
        .{ .x = 0.9, .y = -0.9, .z = 0.8 },
        .{ .x = -0.9, .y = 0.9, .z = 0.8 },
        .{ .x = 0.9, .y = 0.9, .z = 0.8 },
    };
    glBufferData(gles.GL_ARRAY_BUFFER, @sizeOf(@TypeOf(far)), &far, gles.GL_STATIC_DRAW);
    glUniform4f(col_loc, 1, 0, 0, 1); // red
    glDrawArrays(gles.GL_TRIANGLE_STRIP, 0, 4);
    try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());

    // Near quad: depth z=0.2, green, covering the center half. Being nearer (LESS), it must
    // occlude the far red where they overlap.
    const near = [4]Vtx{
        .{ .x = -0.5, .y = -0.5, .z = 0.2 },
        .{ .x = 0.5, .y = -0.5, .z = 0.2 },
        .{ .x = -0.5, .y = 0.5, .z = 0.2 },
        .{ .x = 0.5, .y = 0.5, .z = 0.2 },
    };
    glBufferData(gles.GL_ARRAY_BUFFER, @sizeOf(@TypeOf(near)), &near, gles.GL_STATIC_DRAW);
    glUniform4f(col_loc, 0, 1, 0, 1); // green
    glDrawArrays(gles.GL_TRIANGLE_STRIP, 0, 4);
    try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());

    const s: *state.Surface = @ptrCast(@alignCast(surf.?));
    const dev = s.display.device.?;
    const px = try flushMap(dev, s);
    const at = struct {
        fn p(buf: []const u8, x: usize, y: usize) [3]u8 {
            const i = (y * 64 + x) * 4;
            return .{ buf[i], buf[i + 1], buf[i + 2] };
        }
    }.p;
    // Center (32,32): both quads cover it. The nearer green wins (depth occlusion).
    const c = at(px, 32, 32);
    try std.testing.expect(c[1] > 200 and c[0] < 60); // green dominates, not red
    // Edge (8,32): only the far red quad covers it (near quad is the center half) -> red.
    const e = at(px, 8, 32);
    try std.testing.expect(e[0] > 200 and e[1] < 60); // red, the uniform color reached here
    // Corner (1,1): outside both quads -> clear black (the MVP placed geometry correctly,
    // did not fill the whole buffer).
    const k = at(px, 1, 1);
    try std.testing.expect(k[0] < 10 and k[1] < 10 and k[2] < 10);

    glDeleteBuffers(1, &[_]gles.GLuint{vbo});
    glDeleteProgram(prog);
    glDeleteShader(vs);
    glDeleteShader(fs);
    try std.testing.expectEqual(egl.EGL_TRUE, eglMakeCurrent(dpy, null, null, egl.EGL_NO_CONTEXT));
    try std.testing.expectEqual(egl.EGL_TRUE, eglDestroySurface(dpy, surf));
    try std.testing.expectEqual(egl.EGL_TRUE, eglDestroyContext(dpy, ctx));
}

test "EGL pbuffer stencil oracle: a GL_REPLACE mask clips a later GL_EQUAL draw (UI clip path)" {
    _ = eglBindAPI(egl.EGL_OPENGL_ES_API);
    const dpy = eglGetPlatformDisplay(egl.EGL_PLATFORM_SURFACELESS_MESA, @ptrFromInt(0xE63D9), null);
    try std.testing.expect(dpy != null);
    try std.testing.expectEqual(egl.EGL_TRUE, eglInitialize(dpy, null, null));
    defer _ = eglTerminate(dpy);

    const cfg = state.configHandle(3); // RGBA8 + D24S8 (has a stencil buffer)
    const ctx = eglCreateContext(dpy, cfg, egl.EGL_NO_CONTEXT, null);
    try std.testing.expect(ctx != egl.EGL_NO_CONTEXT);
    const W: EGLint = 64;
    const H: EGLint = 64;
    const pb_attribs = [_]EGLint{ egl.EGL_WIDTH, W, egl.EGL_HEIGHT, H, egl.EGL_NONE };
    const surf = eglCreatePbufferSurface(dpy, cfg, &pb_attribs);
    try std.testing.expect(surf != egl.EGL_NO_SURFACE);
    try std.testing.expectEqual(egl.EGL_TRUE, eglMakeCurrent(dpy, surf, surf, ctx));

    const vs_src: [*:0]const gles.GLchar =
        \\attribute vec3 position;
        \\uniform vec4 uColor;
        \\varying vec4 vColor;
        \\void main() { gl_Position = vec4(position, 1.0); vColor = uColor; }
    ;
    const fs_src: [*:0]const gles.GLchar =
        \\precision mediump float;
        \\varying vec4 vColor;
        \\void main() { gl_FragColor = vColor; }
    ;
    const vs = glCreateShader(gles.GL_VERTEX_SHADER);
    glShaderSource(vs, 1, &[_]?[*:0]const gles.GLchar{vs_src}, null);
    glCompileShader(vs);
    const fs = glCreateShader(gles.GL_FRAGMENT_SHADER);
    glShaderSource(fs, 1, &[_]?[*:0]const gles.GLchar{fs_src}, null);
    glCompileShader(fs);
    const prog = glCreateProgram();
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glBindAttribLocation(prog, 0, "position");
    glLinkProgram(prog);
    var link_ok: gles.GLint = 0;
    glGetProgramiv(prog, gles.GL_LINK_STATUS, &link_ok);
    try std.testing.expectEqual(@as(gles.GLint, gles.GL_TRUE), link_ok);
    glUseProgram(prog);
    const col_loc = glGetUniformLocation(prog, "uColor");
    try std.testing.expect(col_loc >= 0);

    const Vtx = extern struct { x: f32, y: f32, z: f32 };
    var vbo: gles.GLuint = 0;
    glGenBuffers(1, @ptrCast(&vbo));
    glBindBuffer(gles.GL_ARRAY_BUFFER, vbo);
    glVertexAttribPointer(0, 3, gles.GL_FLOAT, gles.GL_FALSE, @sizeOf(Vtx), @ptrFromInt(0));
    glEnableVertexAttribArray(0);

    // Clear color + stencil to 0. The stencil clear is deferred to the first stencil draw.
    glClearColor(0, 0, 0, 1);
    glClearStencil(0);
    glClear(gles.GL_COLOR_BUFFER_BIT | gles.GL_STENCIL_BUFFER_BIT);

    glEnable(gles.GL_STENCIL_TEST);

    // MASK pass: always pass, REPLACE the stencil with ref=1 wherever drawn (the left half).
    glStencilFunc(gles.GL_ALWAYS, 1, 0xFF);
    glStencilOp(gles.GL_KEEP, gles.GL_KEEP, gles.GL_REPLACE);
    glStencilMask(0xFF);
    const left = [4]Vtx{
        .{ .x = -1, .y = -1, .z = 0 }, .{ .x = 0, .y = -1, .z = 0 },
        .{ .x = -1, .y = 1, .z = 0 },  .{ .x = 0, .y = 1, .z = 0 },
    };
    glBufferData(gles.GL_ARRAY_BUFFER, @sizeOf(@TypeOf(left)), &left, gles.GL_STATIC_DRAW);
    glUniform4f(col_loc, 0, 0, 0, 1); // black (the mask shape is not the content)
    glDrawArrays(gles.GL_TRIANGLE_STRIP, 0, 4);
    try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());

    // Content pass: draw red over the full screen, but only where stencil == 1 (the mask).
    glStencilFunc(gles.GL_EQUAL, 1, 0xFF);
    glStencilOp(gles.GL_KEEP, gles.GL_KEEP, gles.GL_KEEP);
    const full = [4]Vtx{
        .{ .x = -1, .y = -1, .z = 0 }, .{ .x = 1, .y = -1, .z = 0 },
        .{ .x = -1, .y = 1, .z = 0 },  .{ .x = 1, .y = 1, .z = 0 },
    };
    glBufferData(gles.GL_ARRAY_BUFFER, @sizeOf(@TypeOf(full)), &full, gles.GL_STATIC_DRAW);
    glUniform4f(col_loc, 1, 0, 0, 1); // red
    glDrawArrays(gles.GL_TRIANGLE_STRIP, 0, 4);
    try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());

    const s: *state.Surface = @ptrCast(@alignCast(surf.?));
    const dev = s.display.device.?;
    const px = try flushMap(dev, s);
    const at = struct {
        fn p(buf: []const u8, x: usize, y: usize) [3]u8 {
            const i = (y * 64 + x) * 4;
            return .{ buf[i], buf[i + 1], buf[i + 2] };
        }
    }.p;
    // Left (16,32): stencil==1 (masked) -> the red content passed the GL_EQUAL test.
    const l = at(px, 16, 32);
    try std.testing.expect(l[0] > 200 and l[1] < 50 and l[2] < 50); // red
    // Right (48,32): stencil==0 -> the content was clipped, leaving the clear black.
    const r = at(px, 48, 32);
    try std.testing.expect(r[0] < 50 and r[1] < 50 and r[2] < 50); // black

    glDeleteBuffers(1, &[_]gles.GLuint{vbo});
    glDeleteProgram(prog);
    glDeleteShader(vs);
    glDeleteShader(fs);
    try std.testing.expectEqual(egl.EGL_TRUE, eglMakeCurrent(dpy, null, null, egl.EGL_NO_CONTEXT));
    try std.testing.expectEqual(egl.EGL_TRUE, eglDestroySurface(dpy, surf));
    try std.testing.expectEqual(egl.EGL_TRUE, eglDestroyContext(dpy, ctx));
}

fn runTwoSidedStencil(dpy_magic: usize) !void {
    _ = eglBindAPI(egl.EGL_OPENGL_ES_API);
    const dpy = eglGetPlatformDisplay(egl.EGL_PLATFORM_SURFACELESS_MESA, @ptrFromInt(dpy_magic), null);
    if (eglInitialize(dpy, null, null) != egl.EGL_TRUE) return error.SkipZigTest;
    defer _ = eglTerminate(dpy);

    const cfg = state.configHandle(3); // RGBA8 + D24S8 (has a stencil buffer)
    const ctx = eglCreateContext(dpy, cfg, egl.EGL_NO_CONTEXT, null);
    try std.testing.expect(ctx != egl.EGL_NO_CONTEXT);
    const W: EGLint = 64;
    const H: EGLint = 64;
    const pb_attribs = [_]EGLint{ egl.EGL_WIDTH, W, egl.EGL_HEIGHT, H, egl.EGL_NONE };
    const surf = eglCreatePbufferSurface(dpy, cfg, &pb_attribs);
    try std.testing.expect(surf != egl.EGL_NO_SURFACE);
    try std.testing.expectEqual(egl.EGL_TRUE, eglMakeCurrent(dpy, surf, surf, ctx));

    const vs_src: [*:0]const gles.GLchar =
        \\attribute vec3 position;
        \\uniform vec4 uColor;
        \\varying vec4 vColor;
        \\void main() { gl_Position = vec4(position, 1.0); vColor = uColor; }
    ;
    const fs_src: [*:0]const gles.GLchar =
        \\precision mediump float;
        \\varying vec4 vColor;
        \\void main() { gl_FragColor = vColor; }
    ;
    const vs = glCreateShader(gles.GL_VERTEX_SHADER);
    glShaderSource(vs, 1, &[_]?[*:0]const gles.GLchar{vs_src}, null);
    glCompileShader(vs);
    const fs = glCreateShader(gles.GL_FRAGMENT_SHADER);
    glShaderSource(fs, 1, &[_]?[*:0]const gles.GLchar{fs_src}, null);
    glCompileShader(fs);
    const prog = glCreateProgram();
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glBindAttribLocation(prog, 0, "position");
    glLinkProgram(prog);
    glUseProgram(prog);
    const col_loc = glGetUniformLocation(prog, "uColor");
    try std.testing.expect(col_loc >= 0);

    const Vtx = extern struct { x: f32, y: f32, z: f32 };
    var vbo: gles.GLuint = 0;
    glGenBuffers(1, @ptrCast(&vbo));
    glBindBuffer(gles.GL_ARRAY_BUFFER, vbo);
    glVertexAttribPointer(0, 3, gles.GL_FLOAT, gles.GL_FALSE, @sizeOf(Vtx), @ptrFromInt(0));
    glEnableVertexAttribArray(0);

    glClearColor(0, 0, 0, 1);
    glClearStencil(0);
    glClear(gles.GL_COLOR_BUFFER_BIT | gles.GL_STENCIL_BUFFER_BIT);
    glEnable(gles.GL_STENCIL_TEST);

    // Two-sided mask: a FRONT-facing triangle REPLACEs stencil with 1, a BACK-facing one with 2.
    // Both always pass the test. Only the write ref differs by face.
    glStencilFuncSeparate(gles.GL_FRONT, gles.GL_ALWAYS, 1, 0xFF);
    glStencilFuncSeparate(gles.GL_BACK, gles.GL_ALWAYS, 2, 0xFF);
    glStencilOpSeparate(gles.GL_FRONT, gles.GL_KEEP, gles.GL_KEEP, gles.GL_REPLACE);
    glStencilOpSeparate(gles.GL_BACK, gles.GL_KEEP, gles.GL_KEEP, gles.GL_REPLACE);
    glStencilMask(0xFF);
    glUniform4f(col_loc, 0, 0, 0, 1); // color irrelevant, only the stencil side effect matters

    // A left triangle and a right triangle with opposite vertex winding, so one is front-facing
    // (writes stencil 1) and the other back-facing (writes stencil 2). Asserts they differ
    // without depending on which absolute winding the y-flip makes "front". (GL_TRIANGLES with
    // explicit winding. A mirrored strip preserves winding, which is why this is not a strip.)
    const left = [3]Vtx{
        .{ .x = -0.9, .y = -0.9, .z = 0 }, .{ .x = -0.1, .y = -0.9, .z = 0 }, .{ .x = -0.5, .y = 0.9, .z = 0 },
    };
    glBufferData(gles.GL_ARRAY_BUFFER, @sizeOf(@TypeOf(left)), &left, gles.GL_STATIC_DRAW);
    glDrawArrays(gles.GL_TRIANGLES, 0, 3);
    const right = [3]Vtx{
        .{ .x = 0.1, .y = -0.9, .z = 0 }, .{ .x = 0.5, .y = 0.9, .z = 0 }, .{ .x = 0.9, .y = -0.9, .z = 0 },
    };
    glBufferData(gles.GL_ARRAY_BUFFER, @sizeOf(@TypeOf(right)), &right, gles.GL_STATIC_DRAW);
    glDrawArrays(gles.GL_TRIANGLES, 0, 3);
    try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());

    // Content: KEEP on both faces now. Red where stencil==1, green where stencil==2, fullscreen.
    glStencilOp(gles.GL_KEEP, gles.GL_KEEP, gles.GL_KEEP);
    const full = [4]Vtx{
        .{ .x = -1, .y = -1, .z = 0 }, .{ .x = 1, .y = -1, .z = 0 },
        .{ .x = -1, .y = 1, .z = 0 },  .{ .x = 1, .y = 1, .z = 0 },
    };
    glBufferData(gles.GL_ARRAY_BUFFER, @sizeOf(@TypeOf(full)), &full, gles.GL_STATIC_DRAW);
    glStencilFunc(gles.GL_EQUAL, 1, 0xFF); // both faces == 1
    glUniform4f(col_loc, 1, 0, 0, 1); // red
    glDrawArrays(gles.GL_TRIANGLE_STRIP, 0, 4);
    glStencilFunc(gles.GL_EQUAL, 2, 0xFF); // both faces == 2
    glUniform4f(col_loc, 0, 1, 0, 1); // green
    glDrawArrays(gles.GL_TRIANGLE_STRIP, 0, 4);
    try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());

    const s: *state.Surface = @ptrCast(@alignCast(surf.?));
    const dev = s.display.device.?;
    const px = try flushMap(dev, s);
    const at = struct {
        fn p(buf: []const u8, x: usize, y: usize) [3]u8 {
            const i = (y * 64 + x) * 4;
            return .{ buf[i], buf[i + 1], buf[i + 2] };
        }
    }.p;
    const l = at(px, 16, 32);
    const r = at(px, 48, 32);
    const l_red = l[0] > 200 and l[1] < 50 and l[2] < 50;
    const l_green = l[1] > 200 and l[0] < 50 and l[2] < 50;
    const r_red = r[0] > 200 and r[1] < 50 and r[2] < 50;
    const r_green = r[1] > 200 and r[0] < 50 and r[2] < 50;
    // Each half is a pure red OR green, and the two halves disagree. This proves the two faces
    // wrote different stencil refs (a single-face pipeline would color both halves identically).
    try std.testing.expect((l_red or l_green) and (r_red or r_green));
    try std.testing.expect(l_red != r_red);

    glDeleteBuffers(1, &[_]gles.GLuint{vbo});
    glDeleteProgram(prog);
    glDeleteShader(vs);
    glDeleteShader(fs);
    try std.testing.expectEqual(egl.EGL_TRUE, eglMakeCurrent(dpy, null, null, egl.EGL_NO_CONTEXT));
    try std.testing.expectEqual(egl.EGL_TRUE, eglDestroySurface(dpy, surf));
    try std.testing.expectEqual(egl.EGL_TRUE, eglDestroyContext(dpy, ctx));
}

test "EGL pbuffer two-sided stencil oracle: front/back faces write different stencil refs (glStencilOpSeparate) (software)" {
    try runTwoSidedStencil(0xE63E5);
}

test "two-sided stencil on NVIDIA GPU: glStencilFuncSeparate front/back write different refs on the real RTX (skips without a GPU)" {
    // Forces the real nvidia device: glStencilFuncSeparate/OpSeparate set per-face stencil, so a
    // front-facing tri writes ref 1 and a back-facing one ref 2 (nvidia SET_TWO_SIDED + BACK_* methods
    // over the combined depth+stencil path). Left half red / right half green must disagree.
    const saved = state.pinned_driver;
    state.pinned_driver = "nvidia";
    defer state.pinned_driver = saved;
    try runTwoSidedStencil(0xE63E6);
}

test "EGL pbuffer scissor oracle: glScissor clips a fullscreen draw to a sub-rect (UI clip path)" {
    _ = eglBindAPI(egl.EGL_OPENGL_ES_API);
    const dpy = eglGetPlatformDisplay(egl.EGL_PLATFORM_SURFACELESS_MESA, @ptrFromInt(0xE63DA), null);
    try std.testing.expect(dpy != null);
    try std.testing.expectEqual(egl.EGL_TRUE, eglInitialize(dpy, null, null));
    defer _ = eglTerminate(dpy);

    const cfg = state.configHandle(0); // RGBA8
    const ctx = eglCreateContext(dpy, cfg, egl.EGL_NO_CONTEXT, null);
    try std.testing.expect(ctx != egl.EGL_NO_CONTEXT);
    const W: EGLint = 64;
    const H: EGLint = 64;
    const pb_attribs = [_]EGLint{ egl.EGL_WIDTH, W, egl.EGL_HEIGHT, H, egl.EGL_NONE };
    const surf = eglCreatePbufferSurface(dpy, cfg, &pb_attribs);
    try std.testing.expect(surf != egl.EGL_NO_SURFACE);
    try std.testing.expectEqual(egl.EGL_TRUE, eglMakeCurrent(dpy, surf, surf, ctx));

    const vs_src: [*:0]const gles.GLchar =
        \\attribute vec3 position;
        \\void main() { gl_Position = vec4(position, 1.0); }
    ;
    const fs_src: [*:0]const gles.GLchar =
        \\precision mediump float;
        \\void main() { gl_FragColor = vec4(1.0, 0.0, 0.0, 1.0); }
    ;
    const vs = glCreateShader(gles.GL_VERTEX_SHADER);
    glShaderSource(vs, 1, &[_]?[*:0]const gles.GLchar{vs_src}, null);
    glCompileShader(vs);
    const fs = glCreateShader(gles.GL_FRAGMENT_SHADER);
    glShaderSource(fs, 1, &[_]?[*:0]const gles.GLchar{fs_src}, null);
    glCompileShader(fs);
    const prog = glCreateProgram();
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glBindAttribLocation(prog, 0, "position");
    glLinkProgram(prog);
    var link_ok: gles.GLint = 0;
    glGetProgramiv(prog, gles.GL_LINK_STATUS, &link_ok);
    try std.testing.expectEqual(@as(gles.GLint, gles.GL_TRUE), link_ok);
    glUseProgram(prog);

    const Vtx = extern struct { x: f32, y: f32, z: f32 };
    var vbo: gles.GLuint = 0;
    glGenBuffers(1, @ptrCast(&vbo));
    glBindBuffer(gles.GL_ARRAY_BUFFER, vbo);
    glVertexAttribPointer(0, 3, gles.GL_FLOAT, gles.GL_FALSE, @sizeOf(Vtx), @ptrFromInt(0));
    glEnableVertexAttribArray(0);

    glClearColor(0, 0, 0, 1);
    glClear(gles.GL_COLOR_BUFFER_BIT);

    // Clip the fullscreen red draw to the bottom-left 32x32 (GL bottom-left origin: y=0 is
    // the bottom). In HAL/readback terms (top-left origin) this is the lower-left quadrant.
    glEnable(gles.GL_SCISSOR_TEST);
    glScissor(0, 0, 32, 32);
    const full = [4]Vtx{
        .{ .x = -1, .y = -1, .z = 0 }, .{ .x = 1, .y = -1, .z = 0 },
        .{ .x = -1, .y = 1, .z = 0 },  .{ .x = 1, .y = 1, .z = 0 },
    };
    glBufferData(gles.GL_ARRAY_BUFFER, @sizeOf(@TypeOf(full)), &full, gles.GL_STATIC_DRAW);
    glDrawArrays(gles.GL_TRIANGLE_STRIP, 0, 4);
    try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());

    const s: *state.Surface = @ptrCast(@alignCast(surf.?));
    const dev = s.display.device.?;
    const px = try flushMap(dev, s);
    const at = struct {
        fn p(buf: []const u8, x: usize, y: usize) [3]u8 {
            const i = (y * 64 + x) * 4;
            return .{ buf[i], buf[i + 1], buf[i + 2] };
        }
    }.p;
    // Readback is top-left origin. GL scissor (0,0,32,32) is the bottom-left quadrant, so in
    // readback rows that is the lower half (y in [32,64)) x left half (x in [0,32)).
    const inside = at(px, 16, 48); // lower-left: drawn red
    try std.testing.expect(inside[0] > 200 and inside[1] < 50 and inside[2] < 50);
    const upper_left = at(px, 16, 16); // upper-left: clipped -> black
    try std.testing.expect(upper_left[0] < 50 and upper_left[1] < 50 and upper_left[2] < 50);
    const lower_right = at(px, 48, 48); // lower-right: clipped -> black
    try std.testing.expect(lower_right[0] < 50 and lower_right[1] < 50 and lower_right[2] < 50);

    glDeleteBuffers(1, &[_]gles.GLuint{vbo});
    glDeleteProgram(prog);
    glDeleteShader(vs);
    glDeleteShader(fs);
    try std.testing.expectEqual(egl.EGL_TRUE, eglMakeCurrent(dpy, null, null, egl.EGL_NO_CONTEXT));
    try std.testing.expectEqual(egl.EGL_TRUE, eglDestroySurface(dpy, surf));
    try std.testing.expectEqual(egl.EGL_TRUE, eglDestroyContext(dpy, ctx));
}

fn runMsaaEdge(dpy_magic: usize) !void {
    _ = eglBindAPI(egl.EGL_OPENGL_ES_API);
    const dpy = eglGetPlatformDisplay(egl.EGL_PLATFORM_SURFACELESS_MESA, @ptrFromInt(dpy_magic), null);
    if (eglInitialize(dpy, null, null) != egl.EGL_TRUE) return error.SkipZigTest;
    defer _ = eglTerminate(dpy);

    // eglChooseConfig(EGL_SAMPLES=4) must select a multisampled config.
    var msaa_cfgs: [8]egl.EGLConfig = undefined;
    var n_msaa: EGLint = 0;
    const msaa_attrs = [_]EGLint{ egl.EGL_SAMPLES, 4, egl.EGL_NONE };
    try std.testing.expectEqual(egl.EGL_TRUE, eglChooseConfig(dpy, &msaa_attrs, &msaa_cfgs, 8, &n_msaa));
    try std.testing.expect(n_msaa >= 1);
    var got_samples: EGLint = 0;
    _ = eglGetConfigAttrib(dpy, msaa_cfgs[0], egl.EGL_SAMPLES, &got_samples);
    try std.testing.expectEqual(@as(EGLint, 4), got_samples);

    const W: EGLint = 64;
    const H: EGLint = 64;

    // Render a slanted red triangle (diagonal hypotenuse) into a pbuffer of `cfg`, swap
    // (resolving MSAA), read back, and count partial-coverage red edge pixels (0<R<255).
    const renderPartial = struct {
        fn go(d: EGLDisplay, cfg: egl.EGLConfig) !usize {
            const ctx = eglCreateContext(d, cfg, egl.EGL_NO_CONTEXT, null);
            defer _ = eglDestroyContext(d, ctx);
            const pb_attribs = [_]EGLint{ egl.EGL_WIDTH, W, egl.EGL_HEIGHT, H, egl.EGL_NONE };
            const surf = eglCreatePbufferSurface(d, cfg, &pb_attribs);
            defer _ = eglDestroySurface(d, surf);
            _ = eglMakeCurrent(d, surf, surf, ctx);
            defer _ = eglMakeCurrent(d, null, null, egl.EGL_NO_CONTEXT);

            const vs_src: [*:0]const gles.GLchar =
                \\attribute vec2 position;
                \\void main() { gl_Position = vec4(position, 0.0, 1.0); }
            ;
            const fs_src: [*:0]const gles.GLchar =
                \\precision mediump float;
                \\void main() { gl_FragColor = vec4(1.0, 0.0, 0.0, 1.0); }
            ;
            const vs = glCreateShader(gles.GL_VERTEX_SHADER);
            glShaderSource(vs, 1, &[_]?[*:0]const gles.GLchar{vs_src}, null);
            glCompileShader(vs);
            const fs = glCreateShader(gles.GL_FRAGMENT_SHADER);
            glShaderSource(fs, 1, &[_]?[*:0]const gles.GLchar{fs_src}, null);
            glCompileShader(fs);
            const prog = glCreateProgram();
            glAttachShader(prog, vs);
            glAttachShader(prog, fs);
            glBindAttribLocation(prog, 0, "position");
            glLinkProgram(prog);
            glUseProgram(prog);

            const tri = [3][2]f32{ .{ -0.9, -0.9 }, .{ 0.9, -0.9 }, .{ -0.9, 0.9 } };
            var vbo: gles.GLuint = 0;
            glGenBuffers(1, @ptrCast(&vbo));
            glBindBuffer(gles.GL_ARRAY_BUFFER, vbo);
            glBufferData(gles.GL_ARRAY_BUFFER, @sizeOf(@TypeOf(tri)), &tri, gles.GL_STATIC_DRAW);
            glVertexAttribPointer(0, 2, gles.GL_FLOAT, gles.GL_FALSE, 8, @ptrFromInt(0));
            glEnableVertexAttribArray(0);

            glClearColor(0, 0, 0, 1);
            glClear(gles.GL_COLOR_BUFFER_BIT);
            glDrawArrays(gles.GL_TRIANGLES, 0, 3);
            _ = eglSwapBuffers(d, surf); // resolves the MSAA render into the backbuffer

            const s: *state.Surface = @ptrCast(@alignCast(surf.?));
            const dev = s.display.device.?;
            const px = try flushMap(dev, s);
            var partial: usize = 0;
            var i: usize = 0;
            while (i < @as(usize, W) * H) : (i += 1) {
                const r = px[i * 4 + 0]; // rgba8 backbuffer: R at byte 0
                if (r > 30 and r < 225) partial += 1;
            }
            glDeleteBuffers(1, &[_]gles.GLuint{vbo});
            glDeleteProgram(prog);
            glDeleteShader(vs);
            glDeleteShader(fs);
            return partial;
        }
    }.go;

    const partial_msaa = try renderPartial(dpy, msaa_cfgs[0]);
    const partial_1x = try renderPartial(dpy, state.configHandle(0)); // samples=0 config
    // The MSAA render's resolved diagonal edge carries partial-coverage red. The 1x render
    // has hard edges (R is 0 or 255). That gap IS the anti-aliasing.
    try std.testing.expect(partial_msaa > 8);
    try std.testing.expect(partial_msaa > partial_1x + 8);
}

test "EGL pbuffer MSAA oracle: an EGL_SAMPLES=4 config anti-aliases a slanted edge (vs a 1x config) (software)" {
    try runMsaaEdge(0xE63DB);
}

test "EGL MSAA on NVIDIA GPU: an EGL_SAMPLES=4 config anti-aliases a slanted edge on the real RTX (skips without a GPU)" {
    // Forces the real nvidia device: a samples=4 color target renders supersampled (ssScale x size)
    // and the swap resolves (box-downsample) it, so the slanted edge carries partial-coverage red.
    // Real hardware MSAA/SSAA antialiasing, more edge pixels than the 1x config.
    const saved = state.pinned_driver;
    state.pinned_driver = "nvidia";
    defer state.pinned_driver = saved;
    try runMsaaEdge(0xE63DC);
}

test "glCompileShader reports a GLSL compile error via the info log (no crash on bad input)" {
    _ = eglBindAPI(egl.EGL_OPENGL_ES_API);
    const dpy = eglGetPlatformDisplay(egl.EGL_PLATFORM_SURFACELESS_MESA, @ptrFromInt(0xE63A3), null);
    _ = eglInitialize(dpy, null, null);
    defer _ = eglTerminate(dpy);

    const fs = glCreateShader(gles.GL_FRAGMENT_SHADER);
    defer glDeleteShader(fs);
    // Malformed GLSL: must not crash. Compile status FALSE with a non-empty info log.
    glShaderSource(fs, 1, &[_]?[*:0]const gles.GLchar{"void main( { gl_FragColor = "}, null);
    glCompileShader(fs);
    var ok: gles.GLint = 1;
    glGetShaderiv(fs, gles.GL_COMPILE_STATUS, &ok);
    try std.testing.expectEqual(@as(gles.GLint, gles.GL_FALSE), ok);
    var log_len: gles.GLint = 0;
    glGetShaderiv(fs, gles.GL_INFO_LOG_LENGTH, &log_len);
    try std.testing.expect(log_len > 0);
    var log_buf: [256]gles.GLchar = undefined;
    var written: gles.GLint = 0;
    glGetShaderInfoLog(fs, log_buf.len, &written, &log_buf);
    try std.testing.expect(written > 0);
    try std.testing.expect(std.mem.indexOf(u8, log_buf[0..@intCast(written)], "compile error") != null);
}

test "eglMakeCurrent / unbind / current getters" {
    const dpy = eglGetPlatformDisplay(egl.EGL_PLATFORM_SURFACELESS_MESA, @ptrFromInt(0xE61C2), null);
    _ = eglInitialize(dpy, null, null);
    defer _ = eglTerminate(dpy);
    const cfg = state.configHandle(2);
    const ctx = eglCreateContext(dpy, cfg, egl.EGL_NO_CONTEXT, null);
    const attribs = [_]EGLint{ egl.EGL_WIDTH, 16, egl.EGL_HEIGHT, 16, egl.EGL_NONE };
    const surf = eglCreatePbufferSurface(dpy, cfg, &attribs);

    // No context current initially.
    try std.testing.expectEqual(egl.EGL_NO_CONTEXT, eglGetCurrentContext());

    try std.testing.expectEqual(egl.EGL_TRUE, eglMakeCurrent(dpy, surf, surf, ctx));
    try std.testing.expectEqual(dpy, eglGetCurrentDisplay());

    // Unbind with a non-null surface + NO_CONTEXT is BAD_MATCH.
    try std.testing.expectEqual(egl.EGL_FALSE, eglMakeCurrent(dpy, surf, surf, egl.EGL_NO_CONTEXT));
    try std.testing.expectEqual(egl.EGL_BAD_MATCH, eglGetError());

    // Proper unbind.
    try std.testing.expectEqual(egl.EGL_TRUE, eglMakeCurrent(dpy, null, null, egl.EGL_NO_CONTEXT));
    try std.testing.expectEqual(egl.EGL_NO_CONTEXT, eglGetCurrentContext());

    _ = eglDestroySurface(dpy, surf);
    _ = eglDestroyContext(dpy, ctx);
}

test "eglCreateContext / Surface reject a bad config + bad display" {
    const dpy = eglGetPlatformDisplay(egl.EGL_PLATFORM_SURFACELESS_MESA, @ptrFromInt(0xE61C3), null);
    _ = eglInitialize(dpy, null, null);
    defer _ = eglTerminate(dpy);

    // Bad config handle.
    try std.testing.expectEqual(egl.EGL_NO_CONTEXT, eglCreateContext(dpy, @ptrFromInt(0xdead), egl.EGL_NO_CONTEXT, null));
    try std.testing.expectEqual(egl.EGL_BAD_CONFIG, eglGetError());

    // Bad display.
    try std.testing.expectEqual(egl.EGL_NO_CONTEXT, eglCreateContext(@ptrFromInt(0xbeef), state.configHandle(0), egl.EGL_NO_CONTEXT, null));
    try std.testing.expectEqual(egl.EGL_BAD_DISPLAY, eglGetError());

    // Pbuffer with no width/height is BAD_PARAMETER.
    try std.testing.expectEqual(egl.EGL_NO_SURFACE, eglCreatePbufferSurface(dpy, state.configHandle(0), null));
    try std.testing.expectEqual(egl.EGL_BAD_PARAMETER, eglGetError());

    // Window surface with a null native window is BAD_NATIVE_WINDOW.
    try std.testing.expectEqual(egl.EGL_NO_SURFACE, eglCreateWindowSurface(dpy, state.configHandle(0), null, null));
    try std.testing.expectEqual(egl.EGL_BAD_NATIVE_WINDOW, eglGetError());
}

test "GLES + EGL render entry points all resolve via getProcAddress" {
    const names = [_][]const u8{
        "glClearColor",            "glClear",                 "glViewport",             "glScissor",            "glGetString",          "glGetError",
        "eglCreateContext",        "eglCreatePbufferSurface", "eglCreateWindowSurface", "eglMakeCurrent",       "eglSwapBuffers",       "eglQuerySurface",
        "eglQueryContext",         "eglGetCurrentContext",    "eglGetCurrentDisplay",   "eglGetCurrentSurface",
        // The es2gears milestone surface: uniforms, fixed-function state, indexed draw.
        "glGetUniformLocation", "glUniform1f",
        "glUniform2f",             "glUniform3f",             "glUniform4f",            "glUniform1i",          "glUniform1fv",         "glUniform2fv",
        "glUniform3fv",            "glUniform4fv",            "glUniformMatrix2fv",     "glUniformMatrix3fv",   "glUniformMatrix4fv",   "glBindAttribLocation",
        "glGetProgramInfoLog",     "glEnable",                "glDisable",              "glDepthFunc",          "glDepthMask",          "glClearDepthf",
        "glCullFace",              "glFrontFace",             "glDrawElements",         "glBlendFunc",          "glBlendFuncSeparate",  "glBlendEquation",
        "glBlendEquationSeparate", "glBlendColor",            "glGenTextures",          "glDeleteTextures",     "glBindTexture",        "glActiveTexture",
        "glIsTexture",             "glTexParameteri",         "glTexParameterf",        "glTexImage2D",         "glTexSubImage2D",      "glPixelStorei",
        "glGenerateMipmap",        "glPolygonOffset",         "glColorMask",
    };
    for (names) |n| {
        if (lookupProc(n) == null) {
            std.debug.print("missing render entry point: {s}\n", .{n});
            return error.MissingEntryPoint;
        }
    }
}

test "glDrawElements indexed draw renders the same quad as two GL_TRIANGLES" {
    _ = eglBindAPI(egl.EGL_OPENGL_ES_API);
    const dpy = eglGetPlatformDisplay(egl.EGL_PLATFORM_SURFACELESS_MESA, @ptrFromInt(0xE63E9), null);
    try std.testing.expectEqual(egl.EGL_TRUE, eglInitialize(dpy, null, null));
    defer _ = eglTerminate(dpy);
    const cfg = state.configHandle(0);
    const ctx = eglCreateContext(dpy, cfg, egl.EGL_NO_CONTEXT, null);
    const W: EGLint = 64;
    const pb = [_]EGLint{ egl.EGL_WIDTH, W, egl.EGL_HEIGHT, W, egl.EGL_NONE };
    const surf = eglCreatePbufferSurface(dpy, cfg, &pb);
    try std.testing.expectEqual(egl.EGL_TRUE, eglMakeCurrent(dpy, surf, surf, ctx));

    const vs_src: [*:0]const gles.GLchar =
        \\attribute vec3 position;
        \\uniform vec4 uColor;
        \\varying vec4 vColor;
        \\void main() { gl_Position = vec4(position, 1.0); vColor = uColor; }
    ;
    const fs_src: [*:0]const gles.GLchar =
        \\precision mediump float;
        \\varying vec4 vColor;
        \\void main() { gl_FragColor = vColor; }
    ;
    const vs = glCreateShader(gles.GL_VERTEX_SHADER);
    glShaderSource(vs, 1, &[_]?[*:0]const gles.GLchar{vs_src}, null);
    glCompileShader(vs);
    const fs = glCreateShader(gles.GL_FRAGMENT_SHADER);
    glShaderSource(fs, 1, &[_]?[*:0]const gles.GLchar{fs_src}, null);
    glCompileShader(fs);
    const prog = glCreateProgram();
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glBindAttribLocation(prog, 0, "position");
    glLinkProgram(prog);
    glUseProgram(prog);
    const col_loc = glGetUniformLocation(prog, "uColor");
    glUniform4f(col_loc, 0.2, 0.9, 0.4, 1);

    const Vtx = extern struct { x: f32, y: f32, z: f32 };
    const quad = [4]Vtx{
        .{ .x = -0.8, .y = -0.8, .z = 0 },
        .{ .x = 0.8, .y = -0.8, .z = 0 },
        .{ .x = 0.8, .y = 0.8, .z = 0 },
        .{ .x = -0.8, .y = 0.8, .z = 0 },
    };
    const idx = [6]u16{ 0, 1, 2, 0, 2, 3 };
    var vbo: gles.GLuint = 0;
    var ibo: gles.GLuint = 0;
    glGenBuffers(1, @ptrCast(&vbo));
    glGenBuffers(1, @ptrCast(&ibo));
    glBindBuffer(gles.GL_ARRAY_BUFFER, vbo);
    glBufferData(gles.GL_ARRAY_BUFFER, @sizeOf(@TypeOf(quad)), &quad, gles.GL_STATIC_DRAW);
    glBindBuffer(gles.GL_ELEMENT_ARRAY_BUFFER, ibo);
    glBufferData(gles.GL_ELEMENT_ARRAY_BUFFER, @sizeOf(@TypeOf(idx)), &idx, gles.GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, gles.GL_FLOAT, gles.GL_FALSE, @sizeOf(Vtx), @ptrFromInt(0));
    glEnableVertexAttribArray(0);
    glDisableVertexAttribArray(1);

    glClearColor(0, 0, 0, 1);
    glClear(gles.GL_COLOR_BUFFER_BIT);
    glDrawElements(gles.GL_TRIANGLES, 6, gles.GL_UNSIGNED_SHORT, @ptrFromInt(0));
    try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());

    const s: *state.Surface = @ptrCast(@alignCast(surf.?));
    const dev = s.display.device.?;
    const px = try flushMap(dev, s);
    const c = (32 * 64 + 32) * 4;
    // Center is inside the indexed quad -> the uniform color (~0.2,0.9,0.4 -> ~51,229,102).
    try std.testing.expect(px[c + 0] > 30 and px[c + 0] < 90);
    try std.testing.expect(px[c + 1] > 200);
    try std.testing.expect(px[c + 2] > 70 and px[c + 2] < 140);
    // Corner outside the quad stays clear black.
    try std.testing.expectEqual(@as(u8, 0), px[0]);

    glDeleteBuffers(1, &[_]gles.GLuint{ vbo, ibo });
    glDeleteProgram(prog);
    glDeleteShader(vs);
    glDeleteShader(fs);
    _ = eglMakeCurrent(dpy, null, null, egl.EGL_NO_CONTEXT);
    _ = eglDestroySurface(dpy, surf);
    _ = eglDestroyContext(dpy, ctx);
}

test "glEnable(GL_BLEND) + glBlendFunc composites a translucent quad over the cleared background" {
    // End-to-end GLES2 blend: clear to opaque blue, then draw a full-coverage quad whose FS
    // outputs rgba(1,0,0,0.5) with SRC_ALPHA / ONE_MINUS_SRC_ALPHA. The center must be the
    // blended (0.5,0,0.5) = (~128,0,~128); without blending it would be solid red.
    _ = eglBindAPI(egl.EGL_OPENGL_ES_API);
    const dpy = eglGetPlatformDisplay(egl.EGL_PLATFORM_SURFACELESS_MESA, @ptrFromInt(0xB1E4D), null);
    try std.testing.expectEqual(egl.EGL_TRUE, eglInitialize(dpy, null, null));
    defer _ = eglTerminate(dpy);
    const cfg = state.configHandle(0);
    const ctx = eglCreateContext(dpy, cfg, egl.EGL_NO_CONTEXT, null);
    const W: EGLint = 64;
    const pb = [_]EGLint{ egl.EGL_WIDTH, W, egl.EGL_HEIGHT, W, egl.EGL_NONE };
    const surf = eglCreatePbufferSurface(dpy, cfg, &pb);
    try std.testing.expectEqual(egl.EGL_TRUE, eglMakeCurrent(dpy, surf, surf, ctx));

    const vs_src: [*:0]const gles.GLchar =
        \\attribute vec3 position;
        \\void main() { gl_Position = vec4(position, 1.0); }
    ;
    const fs_src: [*:0]const gles.GLchar =
        \\precision mediump float;
        \\uniform vec4 uColor;
        \\void main() { gl_FragColor = uColor; }
    ;
    const vs = glCreateShader(gles.GL_VERTEX_SHADER);
    glShaderSource(vs, 1, &[_]?[*:0]const gles.GLchar{vs_src}, null);
    glCompileShader(vs);
    const fs = glCreateShader(gles.GL_FRAGMENT_SHADER);
    glShaderSource(fs, 1, &[_]?[*:0]const gles.GLchar{fs_src}, null);
    glCompileShader(fs);
    const prog = glCreateProgram();
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glBindAttribLocation(prog, 0, "position");
    glLinkProgram(prog);
    glUseProgram(prog);
    glUniform4f(glGetUniformLocation(prog, "uColor"), 1.0, 0.0, 0.0, 0.5);

    const Vtx = extern struct { x: f32, y: f32, z: f32 };
    const quad = [4]Vtx{
        .{ .x = -1, .y = -1, .z = 0 },
        .{ .x = 1, .y = -1, .z = 0 },
        .{ .x = 1, .y = 1, .z = 0 },
        .{ .x = -1, .y = 1, .z = 0 },
    };
    const idx = [6]u16{ 0, 1, 2, 0, 2, 3 };
    var vbo: gles.GLuint = 0;
    var ibo: gles.GLuint = 0;
    glGenBuffers(1, @ptrCast(&vbo));
    glGenBuffers(1, @ptrCast(&ibo));
    glBindBuffer(gles.GL_ARRAY_BUFFER, vbo);
    glBufferData(gles.GL_ARRAY_BUFFER, @sizeOf(@TypeOf(quad)), &quad, gles.GL_STATIC_DRAW);
    glBindBuffer(gles.GL_ELEMENT_ARRAY_BUFFER, ibo);
    glBufferData(gles.GL_ELEMENT_ARRAY_BUFFER, @sizeOf(@TypeOf(idx)), &idx, gles.GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, gles.GL_FLOAT, gles.GL_FALSE, @sizeOf(Vtx), @ptrFromInt(0));
    glEnableVertexAttribArray(0);

    glEnable(gles.GL_BLEND);
    glBlendFunc(gles.GL_SRC_ALPHA, gles.GL_ONE_MINUS_SRC_ALPHA);
    try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());

    glClearColor(0, 0, 1, 1); // opaque blue background
    glClear(gles.GL_COLOR_BUFFER_BIT);
    glDrawElements(gles.GL_TRIANGLES, 6, gles.GL_UNSIGNED_SHORT, @ptrFromInt(0));
    try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());

    const s: *state.Surface = @ptrCast(@alignCast(surf.?));
    const dev = s.display.device.?;
    const px = try flushMap(dev, s);
    const c = (32 * 64 + 32) * 4;
    // Blended center: r ~128 (0.5), g 0, b ~128 (0.5*blue). NOT solid red (255,0,0).
    try std.testing.expect(px[c + 0] > 110 and px[c + 0] < 145); // ~0.5 red
    try std.testing.expectEqual(@as(u8, 0), px[c + 1]); // no green
    try std.testing.expect(px[c + 2] > 110 and px[c + 2] < 145); // ~0.5 blue (survived)

    glDeleteBuffers(1, &[_]gles.GLuint{ vbo, ibo });
    glDeleteProgram(prog);
    glDeleteShader(vs);
    glDeleteShader(fs);
    _ = eglMakeCurrent(dpy, null, null, egl.EGL_NO_CONTEXT);
    _ = eglDestroySurface(dpy, surf);
    _ = eglDestroyContext(dpy, ctx);
}

/// Render a textured full-pbuffer quad through the GLES2 texture API and return the 64x64
/// RGBA backbuffer bytes. `filter` is the GL min/mag filter (GL_NEAREST / GL_LINEAR). The
/// texture is a 2x2 RGBA8 checkerboard: texel(0,0)=red (255,0,0), (1,0)=green (0,255,0),
/// (0,1)=blue (0,0,255), (1,1)=white. The quad spans NDC [-1,1] with uv (0,0)..(1,1).
/// Passthrough-uv VS + a `texture2D(uTex, vUV)` FS samples it. The 64*64*4 backbuffer bytes
/// are copied into `out` before EGL is torn down (the live map dangles after teardown).
fn renderTexturedQuad(filter: gles.GLenum, out: *[64 * 64 * 4]u8) !void {
    _ = eglBindAPI(egl.EGL_OPENGL_ES_API);
    const dpy = eglGetPlatformDisplay(egl.EGL_PLATFORM_SURFACELESS_MESA, @ptrFromInt(0x7E640), null);
    try std.testing.expectEqual(egl.EGL_TRUE, eglInitialize(dpy, null, null));
    const cfg = state.configHandle(0);
    const ctx = eglCreateContext(dpy, cfg, egl.EGL_NO_CONTEXT, null);
    const W: EGLint = 64;
    const pb = [_]EGLint{ egl.EGL_WIDTH, W, egl.EGL_HEIGHT, W, egl.EGL_NONE };
    const surf = eglCreatePbufferSurface(dpy, cfg, &pb);
    try std.testing.expectEqual(egl.EGL_TRUE, eglMakeCurrent(dpy, surf, surf, ctx));

    const vs_src: [*:0]const gles.GLchar =
        \\attribute vec2 position;
        \\attribute vec2 aUV;
        \\varying vec2 vUV;
        \\void main() { gl_Position = vec4(position, 0.0, 1.0); vUV = aUV; }
    ;
    const fs_src: [*:0]const gles.GLchar =
        \\precision mediump float;
        \\uniform sampler2D uTex;
        \\varying vec2 vUV;
        \\void main() { gl_FragColor = texture2D(uTex, vUV); }
    ;
    const vs = glCreateShader(gles.GL_VERTEX_SHADER);
    glShaderSource(vs, 1, &[_]?[*:0]const gles.GLchar{vs_src}, null);
    glCompileShader(vs);
    const fs = glCreateShader(gles.GL_FRAGMENT_SHADER);
    glShaderSource(fs, 1, &[_]?[*:0]const gles.GLchar{fs_src}, null);
    glCompileShader(fs);
    const prog = glCreateProgram();
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glBindAttribLocation(prog, 0, "position");
    glBindAttribLocation(prog, 1, "aUV");
    glLinkProgram(prog);
    glUseProgram(prog);

    // A 2x2 RGBA8 checkerboard.
    const tex_px = [_]u8{
        255, 0, 0, 255, 0, 255, 0, 255, // row 0: red, green
        0, 0, 255, 255, 255, 255, 255, 255, // row 1: blue, white
    };
    var tex: gles.GLuint = 0;
    glGenTextures(1, @ptrCast(&tex));
    glActiveTexture(gles.GL_TEXTURE0);
    glBindTexture(gles.GL_TEXTURE_2D, tex);
    glTexImage2D(gles.GL_TEXTURE_2D, 0, gles.GL_RGBA, 2, 2, 0, gles.GL_RGBA, gles.GL_UNSIGNED_BYTE, &tex_px);
    glTexParameteri(gles.GL_TEXTURE_2D, gles.GL_TEXTURE_MIN_FILTER, @intCast(filter));
    glTexParameteri(gles.GL_TEXTURE_2D, gles.GL_TEXTURE_MAG_FILTER, @intCast(filter));
    glTexParameteri(gles.GL_TEXTURE_2D, gles.GL_TEXTURE_WRAP_S, @intCast(gles.GL_CLAMP_TO_EDGE));
    glTexParameteri(gles.GL_TEXTURE_2D, gles.GL_TEXTURE_WRAP_T, @intCast(gles.GL_CLAMP_TO_EDGE));
    const tex_loc = glGetUniformLocation(prog, "uTex");
    glUniform1i(tex_loc, 0); // sampler reads texture unit 0

    // A full-pbuffer quad (two triangles), interleaved position(x,y) + uv(u,v).
    const Vtx = extern struct { x: f32, y: f32, u: f32, v: f32 };
    const quad = [6]Vtx{
        .{ .x = -1, .y = -1, .u = 0, .v = 0 },
        .{ .x = 1, .y = -1, .u = 1, .v = 0 },
        .{ .x = 1, .y = 1, .u = 1, .v = 1 },
        .{ .x = -1, .y = -1, .u = 0, .v = 0 },
        .{ .x = 1, .y = 1, .u = 1, .v = 1 },
        .{ .x = -1, .y = 1, .u = 0, .v = 1 },
    };
    var vbo: gles.GLuint = 0;
    glGenBuffers(1, @ptrCast(&vbo));
    glBindBuffer(gles.GL_ARRAY_BUFFER, vbo);
    glBufferData(gles.GL_ARRAY_BUFFER, @sizeOf(@TypeOf(quad)), &quad, gles.GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, gles.GL_FLOAT, gles.GL_FALSE, @sizeOf(Vtx), @ptrFromInt(0));
    glVertexAttribPointer(1, 2, gles.GL_FLOAT, gles.GL_FALSE, @sizeOf(Vtx), @ptrFromInt(8));
    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);

    glClearColor(0, 0, 0, 1);
    glClear(gles.GL_COLOR_BUFFER_BIT);
    glDrawArrays(gles.GL_TRIANGLES, 0, 6);
    try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());

    const s: *state.Surface = @ptrCast(@alignCast(surf.?));
    const dev = s.display.device.?;
    const px = try flushMap(dev, s);
    @memcpy(out, px[0 .. 64 * 64 * 4]);

    glDeleteBuffers(1, &[_]gles.GLuint{vbo});
    glDeleteTextures(1, &[_]gles.GLuint{tex});
    glDeleteProgram(prog);
    glDeleteShader(vs);
    glDeleteShader(fs);
    _ = eglMakeCurrent(dpy, null, null, egl.EGL_NO_CONTEXT);
    _ = eglDestroySurface(dpy, surf);
    _ = eglDestroyContext(dpy, ctx);
    _ = eglTerminate(dpy);
}

test "EGL 3D-texture oracle: a sampler3D LUT selects/blends Z-slices by the vec3 coordinate" {
    _ = eglBindAPI(egl.EGL_OPENGL_ES_API);
    const dpy = eglGetPlatformDisplay(egl.EGL_PLATFORM_SURFACELESS_MESA, @ptrFromInt(0x7E646), null);
    try std.testing.expectEqual(egl.EGL_TRUE, eglInitialize(dpy, null, null));
    const cfg = state.configHandle(0);
    const ctx = eglCreateContext(dpy, cfg, egl.EGL_NO_CONTEXT, null);
    const W: EGLint = 64;
    const pb = [_]EGLint{ egl.EGL_WIDTH, W, egl.EGL_HEIGHT, W, egl.EGL_NONE };
    const surf = eglCreatePbufferSurface(dpy, cfg, &pb);
    try std.testing.expectEqual(egl.EGL_TRUE, eglMakeCurrent(dpy, surf, surf, ctx));
    defer {
        _ = eglMakeCurrent(dpy, null, null, egl.EGL_NO_CONTEXT);
        _ = eglDestroySurface(dpy, surf);
        _ = eglDestroyContext(dpy, ctx);
        _ = eglTerminate(dpy);
    }

    const vs_src: [*:0]const gles.GLchar =
        \\attribute vec2 position;
        \\void main() { gl_Position = vec4(position, 0.0, 1.0); }
    ;
    const fs_src: [*:0]const gles.GLchar =
        \\precision mediump float;
        \\uniform sampler3D uLut;
        \\uniform vec3 uCoord;
        \\void main() { gl_FragColor = texture(uLut, uCoord); }
    ;
    const vs = glCreateShader(gles.GL_VERTEX_SHADER);
    glShaderSource(vs, 1, &[_]?[*:0]const gles.GLchar{vs_src}, null);
    glCompileShader(vs);
    const fs = glCreateShader(gles.GL_FRAGMENT_SHADER);
    glShaderSource(fs, 1, &[_]?[*:0]const gles.GLchar{fs_src}, null);
    glCompileShader(fs);
    const prog = glCreateProgram();
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glBindAttribLocation(prog, 0, "position");
    glLinkProgram(prog);
    glUseProgram(prog);
    defer {
        glDeleteProgram(prog);
        glDeleteShader(vs);
        glDeleteShader(fs);
    }

    // A 2x2x2 LUT: slice 0 (z=0) all red, slice 1 (z=1) all blue, packed slice-major.
    var vol: [2 * 2 * 2 * 4]u8 = undefined;
    for (0..4) |i| { // slice 0: red
        vol[i * 4 + 0] = 255;
        vol[i * 4 + 1] = 0;
        vol[i * 4 + 2] = 0;
        vol[i * 4 + 3] = 255;
    }
    for (4..8) |i| { // slice 1: blue
        vol[i * 4 + 0] = 0;
        vol[i * 4 + 1] = 0;
        vol[i * 4 + 2] = 255;
        vol[i * 4 + 3] = 255;
    }
    var lut: gles.GLuint = 0;
    glGenTextures(1, @ptrCast(&lut));
    glActiveTexture(gles.GL_TEXTURE0);
    glBindTexture(gles.GL_TEXTURE_3D, lut);
    glTexImage3D(gles.GL_TEXTURE_3D, 0, gles.GL_RGBA, 2, 2, 2, 0, gles.GL_RGBA, gles.GL_UNSIGNED_BYTE, &vol);
    glTexParameteri(gles.GL_TEXTURE_3D, gles.GL_TEXTURE_MIN_FILTER, @intCast(gles.GL_NEAREST));
    glTexParameteri(gles.GL_TEXTURE_3D, gles.GL_TEXTURE_MAG_FILTER, @intCast(gles.GL_NEAREST));
    glUniform1i(glGetUniformLocation(prog, "uLut"), 0);
    defer glDeleteTextures(1, &[_]gles.GLuint{lut});

    const tri = [_]f32{ -1, -1, 3, -1, -1, 3 };
    var vbo: gles.GLuint = 0;
    glGenBuffers(1, @ptrCast(&vbo));
    glBindBuffer(gles.GL_ARRAY_BUFFER, vbo);
    glBufferData(gles.GL_ARRAY_BUFFER, @sizeOf(@TypeOf(tri)), &tri, gles.GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, gles.GL_FLOAT, gles.GL_FALSE, 0, @ptrFromInt(0));
    glEnableVertexAttribArray(0);
    defer glDeleteBuffers(1, &[_]gles.GLuint{vbo});

    const co_loc = glGetUniformLocation(prog, "uCoord");
    const s: *state.Surface = @ptrCast(@alignCast(surf.?));
    const dev = s.display.device.?;
    const ci = (32 * 64 + 32) * 4;
    const draw = struct {
        fn d(cl: gles.GLint, x: f32, y: f32, z: f32) void {
            glUniform3f(cl, x, y, z);
            glClearColor(0, 0, 0, 1);
            glClear(gles.GL_COLOR_BUFFER_BIT);
            glDrawArrays(gles.GL_TRIANGLES, 0, 3);
        }
    }.d;

    // NEAREST: coord z=0.25 -> slice 0 (red); z=0.75 -> slice 1 (blue).
    draw(co_loc, 0.5, 0.5, 0.25);
    const p0 = try flushMap(dev, s);
    try std.testing.expect(p0[ci + 0] > 200 and p0[ci + 2] < 60);
    draw(co_loc, 0.5, 0.5, 0.75);
    const p1 = try flushMap(dev, s);
    try std.testing.expect(p1[ci + 2] > 200 and p1[ci + 0] < 60);

    // LINEAR: z=0.5 blends the red + blue slices -> both channels present (purple).
    glTexParameteri(gles.GL_TEXTURE_3D, gles.GL_TEXTURE_MAG_FILTER, @intCast(gles.GL_LINEAR));
    glTexParameteri(gles.GL_TEXTURE_3D, gles.GL_TEXTURE_MIN_FILTER, @intCast(gles.GL_LINEAR));
    draw(co_loc, 0.5, 0.5, 0.5);
    const p2 = try flushMap(dev, s);
    try std.testing.expect(p2[ci + 0] > 60 and p2[ci + 2] > 60);

    // glTexSubImage3D: overwrite slice 1 (z=1) with GREEN, then NEAREST z=0.75 -> green (the
    // sub-update landed on the right slice) while z=0.25 -> still red.
    glTexParameteri(gles.GL_TEXTURE_3D, gles.GL_TEXTURE_MAG_FILTER, @intCast(gles.GL_NEAREST));
    const green = [_]u8{ 0, 255, 0, 255 } ** 4; // 4 texels (one slice)
    glTexSubImage3D(gles.GL_TEXTURE_3D, 0, 0, 0, 1, 2, 2, 1, gles.GL_RGBA, gles.GL_UNSIGNED_BYTE, &green);
    try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());
    draw(co_loc, 0.5, 0.5, 0.75);
    const p3 = try flushMap(dev, s);
    try std.testing.expect(p3[ci + 1] > 200 and p3[ci + 0] < 60 and p3[ci + 2] < 60);
    draw(co_loc, 0.5, 0.5, 0.25);
    const p4 = try flushMap(dev, s);
    try std.testing.expect(p4[ci + 0] > 200 and p4[ci + 1] < 60); // slice 0 untouched -> red
}

test "EGL 2D-array oracle: a sampler2DArray selects a layer by a raw index (glTexImage3D GL_TEXTURE_2D_ARRAY)" {
    // The stock GLES3 path: glTexImage3D(GL_TEXTURE_2D_ARRAY) + `texture(sampler2DArray, vec3)` where
    // P.z is a RAW layer index (not normalized). A 2x2x3 array: layer 0 = 4 distinct in-layer colors,
    // layer 1 = green, layer 2 = blue. Sampling with z=0/1/2 selects that layer. z=0 with different
    // (u,v) reads the in-layer texels. Proves the whole GLES-to-HAL 2D-array plumbing.
    _ = eglBindAPI(egl.EGL_OPENGL_ES_API);
    const dpy = eglGetPlatformDisplay(egl.EGL_PLATFORM_SURFACELESS_MESA, @ptrFromInt(0x7E650), null);
    try std.testing.expectEqual(egl.EGL_TRUE, eglInitialize(dpy, null, null));
    const cfg = state.configHandle(0);
    const ctx = eglCreateContext(dpy, cfg, egl.EGL_NO_CONTEXT, null);
    const Wd: EGLint = 64;
    const pb = [_]EGLint{ egl.EGL_WIDTH, Wd, egl.EGL_HEIGHT, Wd, egl.EGL_NONE };
    const surf = eglCreatePbufferSurface(dpy, cfg, &pb);
    try std.testing.expectEqual(egl.EGL_TRUE, eglMakeCurrent(dpy, surf, surf, ctx));
    defer {
        _ = eglMakeCurrent(dpy, null, null, egl.EGL_NO_CONTEXT);
        _ = eglDestroySurface(dpy, surf);
        _ = eglDestroyContext(dpy, ctx);
        _ = eglTerminate(dpy);
    }

    const vs_src: [*:0]const gles.GLchar =
        \\attribute vec2 position;
        \\void main() { gl_Position = vec4(position, 0.0, 1.0); }
    ;
    const fs_src: [*:0]const gles.GLchar =
        \\precision mediump float;
        \\uniform sampler2DArray uArr;
        \\uniform vec3 uCoord;
        \\void main() { gl_FragColor = texture(uArr, uCoord); }
    ;
    const vs = glCreateShader(gles.GL_VERTEX_SHADER);
    glShaderSource(vs, 1, &[_]?[*:0]const gles.GLchar{vs_src}, null);
    glCompileShader(vs);
    const fs = glCreateShader(gles.GL_FRAGMENT_SHADER);
    glShaderSource(fs, 1, &[_]?[*:0]const gles.GLchar{fs_src}, null);
    glCompileShader(fs);
    const prog = glCreateProgram();
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glBindAttribLocation(prog, 0, "position");
    glLinkProgram(prog);
    glUseProgram(prog);
    defer {
        glDeleteProgram(prog);
        glDeleteShader(vs);
        glDeleteShader(fs);
    }

    // 2x2x3 array, layer-major. Layer 0: red/green/blue/white in-layer; layer 1: all green; layer 2: all blue.
    var vol: [2 * 2 * 3 * 4]u8 = undefined;
    const l0 = [_]u8{ 255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255, 255, 255, 255 };
    @memcpy(vol[0..16], &l0);
    for (0..4) |i| {
        vol[16 + i * 4 ..][0..4].* = .{ 0, 255, 0, 255 }; // layer 1: green
        vol[32 + i * 4 ..][0..4].* = .{ 0, 0, 255, 255 }; // layer 2: blue
    }
    var arr: gles.GLuint = 0;
    glGenTextures(1, @ptrCast(&arr));
    glActiveTexture(gles.GL_TEXTURE0);
    glBindTexture(gles.GL_TEXTURE_2D_ARRAY, arr);
    glTexImage3D(gles.GL_TEXTURE_2D_ARRAY, 0, gles.GL_RGBA, 2, 2, 3, 0, gles.GL_RGBA, gles.GL_UNSIGNED_BYTE, &vol);
    glTexParameteri(gles.GL_TEXTURE_2D_ARRAY, gles.GL_TEXTURE_MIN_FILTER, @intCast(gles.GL_NEAREST));
    glTexParameteri(gles.GL_TEXTURE_2D_ARRAY, gles.GL_TEXTURE_MAG_FILTER, @intCast(gles.GL_NEAREST));
    glUniform1i(glGetUniformLocation(prog, "uArr"), 0);
    try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());
    defer glDeleteTextures(1, &[_]gles.GLuint{arr});

    const tri = [_]f32{ -1, -1, 3, -1, -1, 3 };
    var vbo: gles.GLuint = 0;
    glGenBuffers(1, @ptrCast(&vbo));
    glBindBuffer(gles.GL_ARRAY_BUFFER, vbo);
    glBufferData(gles.GL_ARRAY_BUFFER, @sizeOf(@TypeOf(tri)), &tri, gles.GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, gles.GL_FLOAT, gles.GL_FALSE, 0, @ptrFromInt(0));
    glEnableVertexAttribArray(0);
    defer glDeleteBuffers(1, &[_]gles.GLuint{vbo});

    const co_loc = glGetUniformLocation(prog, "uCoord");
    const s: *state.Surface = @ptrCast(@alignCast(surf.?));
    const dev = s.display.device.?;
    const ci = (32 * 64 + 32) * 4;
    const draw = struct {
        fn d(cl: gles.GLint, x: f32, y: f32, z: f32) void {
            glUniform3f(cl, x, y, z);
            glClearColor(0, 0, 0, 1);
            glClear(gles.GL_COLOR_BUFFER_BIT);
            glDrawArrays(gles.GL_TRIANGLES, 0, 3);
        }
    }.d;

    // Layer 0 (z=0): u<0.5,v<0.5 -> red (TL in-layer texel); u>0.5,v>0.5 -> white (BR).
    draw(co_loc, 0.25, 0.25, 0.0);
    const p0 = try flushMap(dev, s);
    try std.testing.expect(p0[ci + 0] > 200 and p0[ci + 1] < 60 and p0[ci + 2] < 60);
    draw(co_loc, 0.75, 0.75, 0.0);
    const p1 = try flushMap(dev, s);
    try std.testing.expect(p1[ci + 0] > 200 and p1[ci + 1] > 200 and p1[ci + 2] > 200);
    // Layer 1 (z=1) -> green. Layer 2 (z=2) -> blue. The raw index selects the layer.
    draw(co_loc, 0.5, 0.5, 1.0);
    const p2 = try flushMap(dev, s);
    try std.testing.expect(p2[ci + 1] > 200 and p2[ci + 0] < 60 and p2[ci + 2] < 60);
    draw(co_loc, 0.5, 0.5, 2.0);
    const p3 = try flushMap(dev, s);
    try std.testing.expect(p3[ci + 2] > 200 and p3[ci + 0] < 60 and p3[ci + 1] < 60);
}

test "EGL texelFetch oracle: fetches the EXACT texel at integer coords (no filter), through GLSL" {
    // The whole texelFetch path: GLSL texelFetch(sampler2D, ivec2, int) -> vulcan-glsl -> SPIR-V
    // OpImageFetch -> the reader's sampler_fetch_fn host call -> software sampleTextureFetch. A 2x2
    // texture with 4 distinct texels. A uCoord uniform picks the integer texel. The sampler is
    // LINEAR, but texelFetch ignores filtering (exact texel), which the oracle verifies.
    _ = eglBindAPI(egl.EGL_OPENGL_ES_API);
    const dpy = eglGetPlatformDisplay(egl.EGL_PLATFORM_SURFACELESS_MESA, @ptrFromInt(0x7E652), null);
    try std.testing.expectEqual(egl.EGL_TRUE, eglInitialize(dpy, null, null));
    const cfg = state.configHandle(0);
    const ctx = eglCreateContext(dpy, cfg, egl.EGL_NO_CONTEXT, null);
    const Wd: EGLint = 64;
    const pb = [_]EGLint{ egl.EGL_WIDTH, Wd, egl.EGL_HEIGHT, Wd, egl.EGL_NONE };
    const surf = eglCreatePbufferSurface(dpy, cfg, &pb);
    try std.testing.expectEqual(egl.EGL_TRUE, eglMakeCurrent(dpy, surf, surf, ctx));
    defer {
        _ = eglMakeCurrent(dpy, null, null, egl.EGL_NO_CONTEXT);
        _ = eglDestroySurface(dpy, surf);
        _ = eglDestroyContext(dpy, ctx);
        _ = eglTerminate(dpy);
    }

    const vs_src: [*:0]const gles.GLchar =
        \\attribute vec2 position;
        \\void main() { gl_Position = vec4(position, 0.0, 1.0); }
    ;
    const fs_src: [*:0]const gles.GLchar =
        \\precision mediump float;
        \\uniform sampler2D uTex;
        \\uniform vec2 uCoord;
        \\void main() { gl_FragColor = texelFetch(uTex, ivec2(uCoord), 0); }
    ;
    const vs = glCreateShader(gles.GL_VERTEX_SHADER);
    glShaderSource(vs, 1, &[_]?[*:0]const gles.GLchar{vs_src}, null);
    glCompileShader(vs);
    const fs = glCreateShader(gles.GL_FRAGMENT_SHADER);
    glShaderSource(fs, 1, &[_]?[*:0]const gles.GLchar{fs_src}, null);
    glCompileShader(fs);
    const prog = glCreateProgram();
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glBindAttribLocation(prog, 0, "position");
    glLinkProgram(prog);
    glUseProgram(prog);
    defer {
        glDeleteProgram(prog);
        glDeleteShader(vs);
        glDeleteShader(fs);
    }

    // 2x2 RGBA8: (0,0)=red (1,0)=green / (0,1)=blue (1,1)=white.
    const px = [_]u8{ 255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255, 255, 255, 255 };
    var tex: gles.GLuint = 0;
    glGenTextures(1, @ptrCast(&tex));
    glActiveTexture(gles.GL_TEXTURE0);
    glBindTexture(gles.GL_TEXTURE_2D, tex);
    glTexImage2D(gles.GL_TEXTURE_2D, 0, gles.GL_RGBA, 2, 2, 0, gles.GL_RGBA, gles.GL_UNSIGNED_BYTE, &px);
    glTexParameteri(gles.GL_TEXTURE_2D, gles.GL_TEXTURE_MIN_FILTER, @intCast(gles.GL_LINEAR));
    glTexParameteri(gles.GL_TEXTURE_2D, gles.GL_TEXTURE_MAG_FILTER, @intCast(gles.GL_LINEAR));
    glUniform1i(glGetUniformLocation(prog, "uTex"), 0);
    try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());
    defer glDeleteTextures(1, &[_]gles.GLuint{tex});

    const tri = [_]f32{ -1, -1, 3, -1, -1, 3 };
    var vbo: gles.GLuint = 0;
    glGenBuffers(1, @ptrCast(&vbo));
    glBindBuffer(gles.GL_ARRAY_BUFFER, vbo);
    glBufferData(gles.GL_ARRAY_BUFFER, @sizeOf(@TypeOf(tri)), &tri, gles.GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, gles.GL_FLOAT, gles.GL_FALSE, 0, @ptrFromInt(0));
    glEnableVertexAttribArray(0);
    defer glDeleteBuffers(1, &[_]gles.GLuint{vbo});

    const co_loc = glGetUniformLocation(prog, "uCoord");
    const s: *state.Surface = @ptrCast(@alignCast(surf.?));
    const dev = s.display.device.?;
    const ci = (32 * 64 + 32) * 4;
    const draw = struct {
        fn d(cl: gles.GLint, x: f32, y: f32) void {
            glUniform2f(cl, x, y);
            glClearColor(0, 0, 0, 1);
            glClear(gles.GL_COLOR_BUFFER_BIT);
            glDrawArrays(gles.GL_TRIANGLES, 0, 3);
        }
    }.d;

    // Exact texel by integer coord (LINEAR filter ignored): (0,0)=red, (1,0)=green, (1,1)=white.
    draw(co_loc, 0, 0);
    const p0 = try flushMap(dev, s);
    try std.testing.expect(p0[ci + 0] > 250 and p0[ci + 1] < 5 and p0[ci + 2] < 5);
    draw(co_loc, 1, 0);
    const p1 = try flushMap(dev, s);
    try std.testing.expect(p1[ci + 1] > 250 and p1[ci + 0] < 5 and p1[ci + 2] < 5);
    draw(co_loc, 1, 1);
    const p2 = try flushMap(dev, s);
    try std.testing.expect(p2[ci + 0] > 250 and p2[ci + 1] > 250 and p2[ci + 2] > 250);
}

test "EGL textureGather oracle: returns the 4 footprint texels of one component as a vec4 (GL order)" {
    // Proves the whole textureGather path: GLSL textureGather -> vulcan-glsl -> SPIR-V OpImageGather
    // -> the reader's sampler_gather_fn host call -> the software sampleTextureGather. A 2x2 texture
    // with distinct R and B per texel; a fixed sample coord (uCoord = 0.5,0.5) so the footprint is
    // exactly the 2x2. gl_FragColor = textureGather(uTex, uCoord[, comp]) writes the 4 gathered
    // texels into RGBA, so a single center-pixel readback reveals the gather order + component.
    _ = eglBindAPI(egl.EGL_OPENGL_ES_API);
    const dpy = eglGetPlatformDisplay(egl.EGL_PLATFORM_SURFACELESS_MESA, @ptrFromInt(0x7E64C), null);
    try std.testing.expectEqual(egl.EGL_TRUE, eglInitialize(dpy, null, null));
    const cfg = state.configHandle(0);
    const ctx = eglCreateContext(dpy, cfg, egl.EGL_NO_CONTEXT, null);
    const W: EGLint = 64;
    const pb = [_]EGLint{ egl.EGL_WIDTH, W, egl.EGL_HEIGHT, W, egl.EGL_NONE };
    const surf = eglCreatePbufferSurface(dpy, cfg, &pb);
    try std.testing.expectEqual(egl.EGL_TRUE, eglMakeCurrent(dpy, surf, surf, ctx));
    defer {
        _ = eglMakeCurrent(dpy, null, null, egl.EGL_NO_CONTEXT);
        _ = eglDestroySurface(dpy, surf);
        _ = eglDestroyContext(dpy, ctx);
        _ = eglTerminate(dpy);
    }

    // A 2x2 texture: distinct R and B per texel so both the gather order and the component select
    // are observable. Row-major (row 0 = top / v-small). RGBA per texel:
    //   (col0,row0) R=26  B=200   (col1,row0) R=51  B=150
    //   (col0,row1) R=77  B=100   (col1,row1) R=102 B=50
    const tex_px = [_]u8{
        26, 0, 200, 255, 51, 0, 150, 255, // row 0
        77, 0, 100, 255, 102, 0, 50, 255, // row 1
    };
    const s: *state.Surface = @ptrCast(@alignCast(surf.?));
    const dev = s.display.device.?;
    const ci = (32 * 64 + 32) * 4; // center pixel byte offset

    // Render with a given gather FS and return the center RGBA. GL gather order for coord 0.5,0.5:
    // out = (i0,j1),(i1,j1),(i1,j0),(i0,j0) = (col0,row1),(col1,row1),(col1,row0),(col0,row0).
    const Runner = struct {
        fn run(prog_fs: [*:0]const gles.GLchar, tp: *const [16]u8, cif: usize, device: anytype, surface: *state.Surface) [4]u8 {
            const v = glCreateShader(gles.GL_VERTEX_SHADER);
            const vsrc: [*:0]const gles.GLchar =
                \\attribute vec2 position;
                \\void main() { gl_Position = vec4(position, 0.0, 1.0); }
            ;
            glShaderSource(v, 1, &[_]?[*:0]const gles.GLchar{vsrc}, null);
            glCompileShader(v);
            const f = glCreateShader(gles.GL_FRAGMENT_SHADER);
            glShaderSource(f, 1, &[_]?[*:0]const gles.GLchar{prog_fs}, null);
            glCompileShader(f);
            const p = glCreateProgram();
            glAttachShader(p, v);
            glAttachShader(p, f);
            glBindAttribLocation(p, 0, "position");
            glLinkProgram(p);
            glUseProgram(p);
            var tex: gles.GLuint = 0;
            glGenTextures(1, @ptrCast(&tex));
            glActiveTexture(gles.GL_TEXTURE0);
            glBindTexture(gles.GL_TEXTURE_2D, tex);
            glTexImage2D(gles.GL_TEXTURE_2D, 0, gles.GL_RGBA, 2, 2, 0, gles.GL_RGBA, gles.GL_UNSIGNED_BYTE, tp);
            glTexParameteri(gles.GL_TEXTURE_2D, gles.GL_TEXTURE_MIN_FILTER, @intCast(gles.GL_NEAREST));
            glTexParameteri(gles.GL_TEXTURE_2D, gles.GL_TEXTURE_MAG_FILTER, @intCast(gles.GL_NEAREST));
            glTexParameteri(gles.GL_TEXTURE_2D, gles.GL_TEXTURE_WRAP_S, @intCast(gles.GL_CLAMP_TO_EDGE));
            glTexParameteri(gles.GL_TEXTURE_2D, gles.GL_TEXTURE_WRAP_T, @intCast(gles.GL_CLAMP_TO_EDGE));
            glUniform1i(glGetUniformLocation(p, "uTex"), 0);
            glUniform2f(glGetUniformLocation(p, "uCoord"), 0.5, 0.5);
            const tri = [_]f32{ -1, -1, 3, -1, -1, 3 };
            var vbo: gles.GLuint = 0;
            glGenBuffers(1, @ptrCast(&vbo));
            glBindBuffer(gles.GL_ARRAY_BUFFER, vbo);
            glBufferData(gles.GL_ARRAY_BUFFER, @sizeOf(@TypeOf(tri)), &tri, gles.GL_STATIC_DRAW);
            glVertexAttribPointer(0, 2, gles.GL_FLOAT, gles.GL_FALSE, 0, @ptrFromInt(0));
            glEnableVertexAttribArray(0);
            glClearColor(0, 0, 0, 1);
            glClear(gles.GL_COLOR_BUFFER_BIT);
            glDrawArrays(gles.GL_TRIANGLES, 0, 3);
            const px = flushMap(device, surface) catch unreachable;
            const rgba = [4]u8{ px[cif + 0], px[cif + 1], px[cif + 2], px[cif + 3] };
            glDeleteBuffers(1, &[_]gles.GLuint{vbo});
            glDeleteTextures(1, &[_]gles.GLuint{tex});
            glDeleteProgram(p);
            glDeleteShader(v);
            glDeleteShader(f);
            return rgba;
        }
    };

    // comp omitted (defaults to 0 = R): gathers the R channel. Proves the 2-arg textureGather path.
    const fs_r: [*:0]const gles.GLchar =
        \\precision mediump float;
        \\uniform sampler2D uTex;
        \\uniform vec2 uCoord;
        \\void main() { gl_FragColor = textureGather(uTex, uCoord); }
    ;
    const r = Runner.run(fs_r, &tex_px, ci, dev, s);
    // R = (col0,row1),(col1,row1),(col1,row0),(col0,row0) = 77,102,51,26.
    try std.testing.expectApproxEqAbs(@as(f32, 77), @as(f32, @floatFromInt(r[0])), 2);
    try std.testing.expectApproxEqAbs(@as(f32, 102), @as(f32, @floatFromInt(r[1])), 2);
    try std.testing.expectApproxEqAbs(@as(f32, 51), @as(f32, @floatFromInt(r[2])), 2);
    try std.testing.expectApproxEqAbs(@as(f32, 26), @as(f32, @floatFromInt(r[3])), 2);

    // comp = 2 (B): gathers the B channel. Proves the 3-arg (explicit component) path.
    const fs_b: [*:0]const gles.GLchar =
        \\precision mediump float;
        \\uniform sampler2D uTex;
        \\uniform vec2 uCoord;
        \\void main() { gl_FragColor = textureGather(uTex, uCoord, 2); }
    ;
    const b = Runner.run(fs_b, &tex_px, ci, dev, s);
    // B = (col0,row1),(col1,row1),(col1,row0),(col0,row0) = 100,50,150,200.
    try std.testing.expectApproxEqAbs(@as(f32, 100), @as(f32, @floatFromInt(b[0])), 2);
    try std.testing.expectApproxEqAbs(@as(f32, 50), @as(f32, @floatFromInt(b[1])), 2);
    try std.testing.expectApproxEqAbs(@as(f32, 150), @as(f32, @floatFromInt(b[2])), 2);
    try std.testing.expectApproxEqAbs(@as(f32, 200), @as(f32, @floatFromInt(b[3])), 2);
}

test "EGL cube-mipmap oracle: textureLod on a samplerCube selects a per-face mip level (prefiltered env)" {
    _ = eglBindAPI(egl.EGL_OPENGL_ES_API);
    const dpy = eglGetPlatformDisplay(egl.EGL_PLATFORM_SURFACELESS_MESA, @ptrFromInt(0x7E642), null);
    try std.testing.expectEqual(egl.EGL_TRUE, eglInitialize(dpy, null, null));
    const cfg = state.configHandle(0);
    const ctx = eglCreateContext(dpy, cfg, egl.EGL_NO_CONTEXT, null);
    const W: EGLint = 64;
    const pb = [_]EGLint{ egl.EGL_WIDTH, W, egl.EGL_HEIGHT, W, egl.EGL_NONE };
    const surf = eglCreatePbufferSurface(dpy, cfg, &pb);
    try std.testing.expectEqual(egl.EGL_TRUE, eglMakeCurrent(dpy, surf, surf, ctx));
    defer {
        _ = eglMakeCurrent(dpy, null, null, egl.EGL_NO_CONTEXT);
        _ = eglDestroySurface(dpy, surf);
        _ = eglDestroyContext(dpy, ctx);
        _ = eglTerminate(dpy);
    }

    const vs_src: [*:0]const gles.GLchar =
        \\attribute vec2 position;
        \\void main() { gl_Position = vec4(position, 0.0, 1.0); }
    ;
    const fs_src: [*:0]const gles.GLchar =
        \\precision mediump float;
        \\uniform samplerCube uCube;
        \\uniform vec3 uDir;
        \\uniform float uLod;
        \\void main() { gl_FragColor = textureLod(uCube, uDir, uLod); }
    ;
    const vs = glCreateShader(gles.GL_VERTEX_SHADER);
    glShaderSource(vs, 1, &[_]?[*:0]const gles.GLchar{vs_src}, null);
    glCompileShader(vs);
    const fs = glCreateShader(gles.GL_FRAGMENT_SHADER);
    glShaderSource(fs, 1, &[_]?[*:0]const gles.GLchar{fs_src}, null);
    glCompileShader(fs);
    const prog = glCreateProgram();
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glBindAttribLocation(prog, 0, "position");
    glLinkProgram(prog);
    glUseProgram(prog);
    defer {
        glDeleteProgram(prog);
        glDeleteShader(vs);
        glDeleteShader(fs);
    }

    // Each 4x4 face is a red/blue checkerboard. Box-downsampled mips average to purple.
    const TS: gles.GLsizei = 4;
    var facepx: [4 * 4 * 4]u8 = undefined;
    for (0..4) |y| for (0..4) |x| {
        const i = (y * 4 + x) * 4;
        const red = ((x + y) % 2) == 0;
        facepx[i + 0] = if (red) 255 else 0;
        facepx[i + 1] = 0;
        facepx[i + 2] = if (red) 0 else 255;
        facepx[i + 3] = 255;
    };
    const face_targets = [6]gles.GLenum{
        gles.GL_TEXTURE_CUBE_MAP_POSITIVE_X, gles.GL_TEXTURE_CUBE_MAP_NEGATIVE_X,
        gles.GL_TEXTURE_CUBE_MAP_POSITIVE_Y, gles.GL_TEXTURE_CUBE_MAP_NEGATIVE_Y,
        gles.GL_TEXTURE_CUBE_MAP_POSITIVE_Z, gles.GL_TEXTURE_CUBE_MAP_NEGATIVE_Z,
    };
    var cube: gles.GLuint = 0;
    glGenTextures(1, @ptrCast(&cube));
    glActiveTexture(gles.GL_TEXTURE0);
    glBindTexture(gles.GL_TEXTURE_CUBE_MAP, cube);
    for (0..6) |f| {
        glTexImage2D(face_targets[f], 0, gles.GL_RGBA, TS, TS, 0, gles.GL_RGBA, gles.GL_UNSIGNED_BYTE, &facepx);
    }
    glTexParameteri(gles.GL_TEXTURE_CUBE_MAP, gles.GL_TEXTURE_MIN_FILTER, @intCast(gles.GL_NEAREST_MIPMAP_NEAREST));
    glTexParameteri(gles.GL_TEXTURE_CUBE_MAP, gles.GL_TEXTURE_MAG_FILTER, @intCast(gles.GL_NEAREST));
    glGenerateMipmap(gles.GL_TEXTURE_CUBE_MAP);
    glUniform1i(glGetUniformLocation(prog, "uCube"), 0);
    glUniform3f(glGetUniformLocation(prog, "uDir"), 0, 0, 1); // the +Z face
    defer glDeleteTextures(1, &[_]gles.GLuint{cube});

    const tri = [_]f32{ -1, -1, 3, -1, -1, 3 };
    var vbo: gles.GLuint = 0;
    glGenBuffers(1, @ptrCast(&vbo));
    glBindBuffer(gles.GL_ARRAY_BUFFER, vbo);
    glBufferData(gles.GL_ARRAY_BUFFER, @sizeOf(@TypeOf(tri)), &tri, gles.GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, gles.GL_FLOAT, gles.GL_FALSE, 0, @ptrFromInt(0));
    glEnableVertexAttribArray(0);
    defer glDeleteBuffers(1, &[_]gles.GLuint{vbo});

    const lod_loc = glGetUniformLocation(prog, "uLod");
    const s: *state.Surface = @ptrCast(@alignCast(surf.?));
    const dev = s.display.device.?;
    const ci = (32 * 64 + 32) * 4;

    // LOD 0: a base texel -> a pure stripe (one of R / B saturated).
    glUniform1f(lod_loc, 0.0);
    glClearColor(0, 0, 0, 1);
    glClear(gles.GL_COLOR_BUFFER_BIT);
    glDrawArrays(gles.GL_TRIANGLES, 0, 3);
    try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());
    const px0 = try flushMap(dev, s);
    const pure = (px0[ci + 0] > 200 and px0[ci + 2] < 60) or (px0[ci + 0] < 60 and px0[ci + 2] > 200);
    try std.testing.expect(pure);

    // LOD 2: the top (1x1) mip of the +Z face -> the averaged purple (both R and B present).
    glUniform1f(lod_loc, 2.0);
    glClear(gles.GL_COLOR_BUFFER_BIT);
    glDrawArrays(gles.GL_TRIANGLES, 0, 3);
    try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());
    const px2 = try flushMap(dev, s);
    try std.testing.expect(px2[ci + 0] > 60 and px2[ci + 2] > 60);
}

test "EGL skybox oracle: a samplerCube shader samples the face the direction points at (end-to-end)" {
    _ = eglBindAPI(egl.EGL_OPENGL_ES_API);
    const dpy = eglGetPlatformDisplay(egl.EGL_PLATFORM_SURFACELESS_MESA, @ptrFromInt(0x7E641), null);
    try std.testing.expectEqual(egl.EGL_TRUE, eglInitialize(dpy, null, null));
    const cfg = state.configHandle(0);
    const ctx = eglCreateContext(dpy, cfg, egl.EGL_NO_CONTEXT, null);
    const W: EGLint = 64;
    const pb = [_]EGLint{ egl.EGL_WIDTH, W, egl.EGL_HEIGHT, W, egl.EGL_NONE };
    const surf = eglCreatePbufferSurface(dpy, cfg, &pb);
    try std.testing.expectEqual(egl.EGL_TRUE, eglMakeCurrent(dpy, surf, surf, ctx));
    defer {
        _ = eglMakeCurrent(dpy, null, null, egl.EGL_NO_CONTEXT);
        _ = eglDestroySurface(dpy, surf);
        _ = eglDestroyContext(dpy, ctx);
        _ = eglTerminate(dpy);
    }

    // A full-screen triangle; the direction is a uniform so one setup samples any face.
    const vs_src: [*:0]const gles.GLchar =
        \\attribute vec2 position;
        \\void main() { gl_Position = vec4(position, 0.0, 1.0); }
    ;
    const fs_src: [*:0]const gles.GLchar =
        \\precision mediump float;
        \\uniform samplerCube uCube;
        \\uniform vec3 uDir;
        \\void main() { gl_FragColor = textureCube(uCube, uDir); }
    ;
    const vs = glCreateShader(gles.GL_VERTEX_SHADER);
    glShaderSource(vs, 1, &[_]?[*:0]const gles.GLchar{vs_src}, null);
    glCompileShader(vs);
    const fs = glCreateShader(gles.GL_FRAGMENT_SHADER);
    glShaderSource(fs, 1, &[_]?[*:0]const gles.GLchar{fs_src}, null);
    glCompileShader(fs);
    const prog = glCreateProgram();
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glBindAttribLocation(prog, 0, "position");
    glLinkProgram(prog);
    glUseProgram(prog);
    defer {
        glDeleteProgram(prog);
        glDeleteShader(vs);
        glDeleteShader(fs);
    }

    // 6 single-texel faces, one distinct color each (GL order +X,-X,+Y,-Y,+Z,-Z).
    const face_rgb = [6][3]u8{
        .{ 255, 0, 0 }, // +X red
        .{ 0, 255, 0 }, // -X green
        .{ 0, 0, 255 }, // +Y blue
        .{ 255, 255, 0 }, // -Y yellow
        .{ 255, 0, 255 }, // +Z magenta
        .{ 0, 255, 255 }, // -Z cyan
    };
    const face_targets = [6]gles.GLenum{
        gles.GL_TEXTURE_CUBE_MAP_POSITIVE_X, gles.GL_TEXTURE_CUBE_MAP_NEGATIVE_X,
        gles.GL_TEXTURE_CUBE_MAP_POSITIVE_Y, gles.GL_TEXTURE_CUBE_MAP_NEGATIVE_Y,
        gles.GL_TEXTURE_CUBE_MAP_POSITIVE_Z, gles.GL_TEXTURE_CUBE_MAP_NEGATIVE_Z,
    };
    var cube: gles.GLuint = 0;
    glGenTextures(1, @ptrCast(&cube));
    glActiveTexture(gles.GL_TEXTURE0);
    glBindTexture(gles.GL_TEXTURE_CUBE_MAP, cube);
    for (0..6) |f| {
        const texel = [_]u8{ face_rgb[f][0], face_rgb[f][1], face_rgb[f][2], 255 };
        glTexImage2D(face_targets[f], 0, gles.GL_RGBA, 1, 1, 0, gles.GL_RGBA, gles.GL_UNSIGNED_BYTE, &texel);
    }
    glTexParameteri(gles.GL_TEXTURE_CUBE_MAP, gles.GL_TEXTURE_MIN_FILTER, @intCast(gles.GL_NEAREST));
    glTexParameteri(gles.GL_TEXTURE_CUBE_MAP, gles.GL_TEXTURE_MAG_FILTER, @intCast(gles.GL_NEAREST));
    glUniform1i(glGetUniformLocation(prog, "uCube"), 0);
    defer glDeleteTextures(1, &[_]gles.GLuint{cube});

    const tri = [_]f32{ -1, -1, 3, -1, -1, 3 }; // one big triangle covering the pbuffer
    var vbo: gles.GLuint = 0;
    glGenBuffers(1, @ptrCast(&vbo));
    glBindBuffer(gles.GL_ARRAY_BUFFER, vbo);
    glBufferData(gles.GL_ARRAY_BUFFER, @sizeOf(@TypeOf(tri)), &tri, gles.GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, gles.GL_FLOAT, gles.GL_FALSE, 0, @ptrFromInt(0));
    glEnableVertexAttribArray(0);
    defer glDeleteBuffers(1, &[_]gles.GLuint{vbo});

    const dir_loc = glGetUniformLocation(prog, "uDir");
    const s: *state.Surface = @ptrCast(@alignCast(surf.?));
    const dev = s.display.device.?;

    // Each axis-aligned direction must sample the matching face's color at the center pixel.
    const cases = [_]struct { d: [3]f32, face: usize }{
        .{ .d = .{ 1, 0, 0 }, .face = 0 },
        .{ .d = .{ -1, 0, 0 }, .face = 1 },
        .{ .d = .{ 0, 1, 0 }, .face = 2 },
        .{ .d = .{ 0, -1, 0 }, .face = 3 },
        .{ .d = .{ 0, 0, 1 }, .face = 4 },
        .{ .d = .{ 0, 0, -1 }, .face = 5 },
    };
    for (cases) |c| {
        glUniform3f(dir_loc, c.d[0], c.d[1], c.d[2]);
        glClearColor(0, 0, 0, 1);
        glClear(gles.GL_COLOR_BUFFER_BIT);
        glDrawArrays(gles.GL_TRIANGLES, 0, 3);
        try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());
        const px = try flushMap(dev, s);
        const o = (32 * 64 + 32) * 4; // center pixel
        const want = face_rgb[c.face];
        try std.testing.expect(px[o + 0] == want[0] and px[o + 1] == want[1] and px[o + 2] == want[2]);
    }
}

test "textured quad (GLSL): NEAREST reads the 4 checkerboard texels at the right quadrants" {
    var buf: [64 * 64 * 4]u8 = undefined;
    try renderTexturedQuad(gles.GL_NEAREST, &buf);
    const px: []const u8 = &buf;
    // 64x64 backbuffer. GL bottom-left framebuffer origin (the EGL/GLES y-flip): the quad's
    // uv.v=0 vertex is at NDC y=-1 = the BOTTOM of the screen, so uv.v increases UPWARD. Pick
    // pixels well inside each quadrant: x=16 left half (u<0.5), x=48 right half (u>0.5);
    // y=16 top (v>0.5), y=48 bottom (v<0.5).
    const at = struct {
        fn p(b: []const u8, x: usize, y: usize) [4]u8 {
            const o = (y * 64 + x) * 4;
            return .{ b[o], b[o + 1], b[o + 2], b[o + 3] };
        }
    }.p;
    // v=0 (texel row 0 = red,green) is at the screen bottom; v=1 (texel row 1 = blue,white) at
    // the top. So:
    // Top-left (u<0.5, v>0.5) -> texel(0,1) = blue.
    const tl = at(px, 16, 16);
    try std.testing.expect(tl[2] > 200 and tl[0] < 60 and tl[1] < 60);
    // Top-right (u>0.5, v>0.5) -> texel(1,1) = white.
    const tr = at(px, 48, 16);
    try std.testing.expect(tr[0] > 200 and tr[1] > 200 and tr[2] > 200);
    // Bottom-left (u<0.5, v<0.5) -> texel(0,0) = red.
    const bl = at(px, 16, 48);
    try std.testing.expect(bl[0] > 200 and bl[1] < 60 and bl[2] < 60);
    // Bottom-right (u>0.5, v<0.5) -> texel(1,0) = green.
    const br = at(px, 48, 48);
    try std.testing.expect(br[1] > 200 and br[0] < 60 and br[2] < 60);
}

test "textured quad (GLSL): LINEAR blends toward the texel grid center" {
    var buf: [64 * 64 * 4]u8 = undefined;
    try renderTexturedQuad(gles.GL_LINEAR, &buf);
    const px: []const u8 = &buf;
    const at = struct {
        fn p(b: []const u8, x: usize, y: usize) [4]u8 {
            const o = (y * 64 + x) * 4;
            return .{ b[o], b[o + 1], b[o + 2], b[o + 3] };
        }
    }.p;
    // The exact center (uv ~ 0.5,0.5) sits at the meeting of all 4 texels. With CLAMP_TO_EDGE
    // bilinear it averages red+green+blue+white -> a mid grey-ish blend, not a pure texel.
    const c = at(px, 32, 32);
    // Every channel is partial (none saturated to 0 or 255): a real blend, not a single texel.
    var partial = true;
    for (0..3) |i| if (c[i] < 40 or c[i] > 230) {
        partial = false;
    };
    try std.testing.expect(partial);
    // A deep-quadrant pixel still leans to its texel. With the GL bottom-left origin, the
    // top-left deep pixel (8,8) is u<0.5, v>0.5 -> texel(0,1) = blue dominates.
    const tl = at(px, 8, 8);
    try std.testing.expect(tl[2] >= tl[0] and tl[2] >= tl[1]);
}

test "EGL primitive-restart oracle: GL_PRIMITIVE_RESTART_FIXED_INDEX splits a strip into two quads" {
    _ = eglBindAPI(egl.EGL_OPENGL_ES_API);
    const dpy = eglGetPlatformDisplay(egl.EGL_PLATFORM_SURFACELESS_MESA, @ptrFromInt(0x7E645), null);
    try std.testing.expectEqual(egl.EGL_TRUE, eglInitialize(dpy, null, null));
    const cfg = state.configHandle(0);
    const ctx = eglCreateContext(dpy, cfg, egl.EGL_NO_CONTEXT, null);
    const W: EGLint = 64;
    const pb = [_]EGLint{ egl.EGL_WIDTH, W, egl.EGL_HEIGHT, W, egl.EGL_NONE };
    const surf = eglCreatePbufferSurface(dpy, cfg, &pb);
    try std.testing.expectEqual(egl.EGL_TRUE, eglMakeCurrent(dpy, surf, surf, ctx));
    defer {
        _ = eglMakeCurrent(dpy, null, null, egl.EGL_NO_CONTEXT);
        _ = eglDestroySurface(dpy, surf);
        _ = eglDestroyContext(dpy, ctx);
        _ = eglTerminate(dpy);
    }

    const vs_src: [*:0]const gles.GLchar =
        \\attribute vec2 position;
        \\void main() { gl_Position = vec4(position, 0.0, 1.0); }
    ;
    const fs_src: [*:0]const gles.GLchar =
        \\precision mediump float;
        \\void main() { gl_FragColor = vec4(1.0, 0.0, 0.0, 1.0); }
    ;
    const vs = glCreateShader(gles.GL_VERTEX_SHADER);
    glShaderSource(vs, 1, &[_]?[*:0]const gles.GLchar{vs_src}, null);
    glCompileShader(vs);
    const fs = glCreateShader(gles.GL_FRAGMENT_SHADER);
    glShaderSource(fs, 1, &[_]?[*:0]const gles.GLchar{fs_src}, null);
    glCompileShader(fs);
    const prog = glCreateProgram();
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glBindAttribLocation(prog, 0, "position");
    glLinkProgram(prog);
    glUseProgram(prog);
    defer {
        glDeleteProgram(prog);
        glDeleteShader(vs);
        glDeleteShader(fs);
    }

    // Two quads as one triangle strip: a left quad [x -1..-0.4], a right quad [x 0.4..1], with a
    // gap in the middle. A restart index between them stops the strip connecting across the gap.
    const verts = [_]f32{
        -1, -0.2, -1, 0.2, -0.4, -0.2, -0.4, 0.2, // left quad (indices 0..3)
        0.4, -0.2, 0.4, 0.2, 1, -0.2, 1, 0.2, // right quad (indices 4..7)
    };
    var vbo: gles.GLuint = 0;
    glGenBuffers(1, @ptrCast(&vbo));
    glBindBuffer(gles.GL_ARRAY_BUFFER, vbo);
    glBufferData(gles.GL_ARRAY_BUFFER, @sizeOf(@TypeOf(verts)), &verts, gles.GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, gles.GL_FLOAT, gles.GL_FALSE, 0, @ptrFromInt(0));
    glEnableVertexAttribArray(0);
    defer glDeleteBuffers(1, &[_]gles.GLuint{vbo});

    const idx = [_]u16{ 0, 1, 2, 3, 0xFFFF, 4, 5, 6, 7 };
    var ibo: gles.GLuint = 0;
    glGenBuffers(1, @ptrCast(&ibo));
    glBindBuffer(gles.GL_ELEMENT_ARRAY_BUFFER, ibo);
    glBufferData(gles.GL_ELEMENT_ARRAY_BUFFER, @sizeOf(@TypeOf(idx)), &idx, gles.GL_STATIC_DRAW);
    defer glDeleteBuffers(1, &[_]gles.GLuint{ibo});

    glEnable(gles.GL_PRIMITIVE_RESTART_FIXED_INDEX);
    try std.testing.expectEqual(gles.GL_TRUE, glIsEnabled(gles.GL_PRIMITIVE_RESTART_FIXED_INDEX));
    glClearColor(0, 0, 0, 1);
    glClear(gles.GL_COLOR_BUFFER_BIT);
    glDrawElements(gles.GL_TRIANGLE_STRIP, idx.len, gles.GL_UNSIGNED_SHORT, @ptrFromInt(0));
    try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());

    const s: *state.Surface = @ptrCast(@alignCast(surf.?));
    const dev = s.display.device.?;
    const at = struct {
        fn r(px: []const u8, x: usize) u8 {
            return px[(32 * 64 + x) * 4 + 0];
        }
    }.r;
    const px = try flushMap(dev, s);
    try std.testing.expect(at(px, 10) > 200); // left quad -> red
    try std.testing.expect(at(px, 54) > 200); // right quad -> red
    try std.testing.expect(at(px, 32) < 60); // the gap -> black (the restart split the strip)
}

test "EGL map-buffer-range oracle: positions written through glMapBufferRange draw via glDrawRangeElements" {
    _ = eglBindAPI(egl.EGL_OPENGL_ES_API);
    const dpy = eglGetPlatformDisplay(egl.EGL_PLATFORM_SURFACELESS_MESA, @ptrFromInt(0x7E644), null);
    try std.testing.expectEqual(egl.EGL_TRUE, eglInitialize(dpy, null, null));
    const cfg = state.configHandle(0);
    const ctx = eglCreateContext(dpy, cfg, egl.EGL_NO_CONTEXT, null);
    const W: EGLint = 64;
    const pb = [_]EGLint{ egl.EGL_WIDTH, W, egl.EGL_HEIGHT, W, egl.EGL_NONE };
    const surf = eglCreatePbufferSurface(dpy, cfg, &pb);
    try std.testing.expectEqual(egl.EGL_TRUE, eglMakeCurrent(dpy, surf, surf, ctx));
    defer {
        _ = eglMakeCurrent(dpy, null, null, egl.EGL_NO_CONTEXT);
        _ = eglDestroySurface(dpy, surf);
        _ = eglDestroyContext(dpy, ctx);
        _ = eglTerminate(dpy);
    }

    const vs_src: [*:0]const gles.GLchar =
        \\attribute vec2 position;
        \\void main() { gl_Position = vec4(position, 0.0, 1.0); }
    ;
    const fs_src: [*:0]const gles.GLchar =
        \\precision mediump float;
        \\void main() { gl_FragColor = vec4(1.0, 0.0, 0.0, 1.0); }
    ;
    const vs = glCreateShader(gles.GL_VERTEX_SHADER);
    glShaderSource(vs, 1, &[_]?[*:0]const gles.GLchar{vs_src}, null);
    glCompileShader(vs);
    const fs = glCreateShader(gles.GL_FRAGMENT_SHADER);
    glShaderSource(fs, 1, &[_]?[*:0]const gles.GLchar{fs_src}, null);
    glCompileShader(fs);
    const prog = glCreateProgram();
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glBindAttribLocation(prog, 0, "position");
    glLinkProgram(prog);
    glUseProgram(prog);
    defer {
        glDeleteProgram(prog);
        glDeleteShader(vs);
        glDeleteShader(fs);
    }

    // A VBO sized for 3 vec2s, filled with ZEROS (a degenerate triangle). glMapBufferRange then
    // writes the real fullscreen-triangle positions through the mapped pointer.
    const zeros = [_]f32{0} ** 6;
    var vbo: gles.GLuint = 0;
    glGenBuffers(1, @ptrCast(&vbo));
    glBindBuffer(gles.GL_ARRAY_BUFFER, vbo);
    glBufferData(gles.GL_ARRAY_BUFFER, @sizeOf(@TypeOf(zeros)), &zeros, gles.GL_DYNAMIC_DRAW);
    const ptr = glMapBufferRange(gles.GL_ARRAY_BUFFER, 0, @sizeOf(@TypeOf(zeros)), gles.GL_MAP_WRITE_BIT | gles.GL_MAP_INVALIDATE_BUFFER_BIT);
    try std.testing.expect(ptr != null);
    const verts = [_]f32{ -1, -1, 3, -1, -1, 3 };
    const dst: [*]f32 = @ptrCast(@alignCast(ptr.?));
    for (verts, 0..) |v, i| dst[i] = v;
    try std.testing.expectEqual(gles.GL_TRUE, glUnmapBuffer(gles.GL_ARRAY_BUFFER));
    glVertexAttribPointer(0, 2, gles.GL_FLOAT, gles.GL_FALSE, 0, @ptrFromInt(0));
    glEnableVertexAttribArray(0);
    defer glDeleteBuffers(1, &[_]gles.GLuint{vbo});

    const idx = [_]u16{ 0, 1, 2 };
    var ibo: gles.GLuint = 0;
    glGenBuffers(1, @ptrCast(&ibo));
    glBindBuffer(gles.GL_ELEMENT_ARRAY_BUFFER, ibo);
    glBufferData(gles.GL_ELEMENT_ARRAY_BUFFER, @sizeOf(@TypeOf(idx)), &idx, gles.GL_STATIC_DRAW);
    defer glDeleteBuffers(1, &[_]gles.GLuint{ibo});

    glClearColor(0, 0, 0, 1);
    glClear(gles.GL_COLOR_BUFFER_BIT);
    glDrawRangeElements(gles.GL_TRIANGLES, 0, 2, 3, gles.GL_UNSIGNED_SHORT, @ptrFromInt(0));
    try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());

    const s: *state.Surface = @ptrCast(@alignCast(surf.?));
    const dev = s.display.device.?;
    const px = try flushMap(dev, s);
    const ci = (32 * 64 + 32) * 4;
    // The mapped positions cover the screen -> center red. (Zeros would leave it black.)
    try std.testing.expect(px[ci + 0] > 200 and px[ci + 1] < 60 and px[ci + 2] < 60);
}

test "EGL sampler-object oracle: a bound sampler object overrides the texture's filter (NEAREST tex + LINEAR sampler)" {
    _ = eglBindAPI(egl.EGL_OPENGL_ES_API);
    const dpy = eglGetPlatformDisplay(egl.EGL_PLATFORM_SURFACELESS_MESA, @ptrFromInt(0x7E643), null);
    try std.testing.expectEqual(egl.EGL_TRUE, eglInitialize(dpy, null, null));
    const cfg = state.configHandle(0);
    const ctx = eglCreateContext(dpy, cfg, egl.EGL_NO_CONTEXT, null);
    const W: EGLint = 64;
    const pb = [_]EGLint{ egl.EGL_WIDTH, W, egl.EGL_HEIGHT, W, egl.EGL_NONE };
    const surf = eglCreatePbufferSurface(dpy, cfg, &pb);
    try std.testing.expectEqual(egl.EGL_TRUE, eglMakeCurrent(dpy, surf, surf, ctx));
    defer {
        _ = eglMakeCurrent(dpy, null, null, egl.EGL_NO_CONTEXT);
        _ = eglDestroySurface(dpy, surf);
        _ = eglDestroyContext(dpy, ctx);
        _ = eglTerminate(dpy);
    }

    const vs_src: [*:0]const gles.GLchar =
        \\attribute vec2 position;
        \\attribute vec2 aUV;
        \\varying vec2 vUV;
        \\void main() { gl_Position = vec4(position, 0.0, 1.0); vUV = aUV; }
    ;
    const fs_src: [*:0]const gles.GLchar =
        \\precision mediump float;
        \\uniform sampler2D uTex;
        \\varying vec2 vUV;
        \\void main() { gl_FragColor = texture2D(uTex, vUV); }
    ;
    const vs = glCreateShader(gles.GL_VERTEX_SHADER);
    glShaderSource(vs, 1, &[_]?[*:0]const gles.GLchar{vs_src}, null);
    glCompileShader(vs);
    const fs = glCreateShader(gles.GL_FRAGMENT_SHADER);
    glShaderSource(fs, 1, &[_]?[*:0]const gles.GLchar{fs_src}, null);
    glCompileShader(fs);
    const prog = glCreateProgram();
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glBindAttribLocation(prog, 0, "position");
    glBindAttribLocation(prog, 1, "aUV");
    glLinkProgram(prog);
    glUseProgram(prog);
    defer {
        glDeleteProgram(prog);
        glDeleteShader(vs);
        glDeleteShader(fs);
    }

    // A 2x2 checkerboard, uploaded with the TEXTURE's own filter = NEAREST.
    const tex_px = [_]u8{
        255, 0, 0,   255, 0,   255, 0,   255,
        0,   0, 255, 255, 255, 255, 255, 255,
    };
    var tex: gles.GLuint = 0;
    glGenTextures(1, @ptrCast(&tex));
    glActiveTexture(gles.GL_TEXTURE0);
    glBindTexture(gles.GL_TEXTURE_2D, tex);
    glTexImage2D(gles.GL_TEXTURE_2D, 0, gles.GL_RGBA, 2, 2, 0, gles.GL_RGBA, gles.GL_UNSIGNED_BYTE, &tex_px);
    glTexParameteri(gles.GL_TEXTURE_2D, gles.GL_TEXTURE_MIN_FILTER, @intCast(gles.GL_NEAREST));
    glTexParameteri(gles.GL_TEXTURE_2D, gles.GL_TEXTURE_MAG_FILTER, @intCast(gles.GL_NEAREST));
    glTexParameteri(gles.GL_TEXTURE_2D, gles.GL_TEXTURE_WRAP_S, @intCast(gles.GL_CLAMP_TO_EDGE));
    glTexParameteri(gles.GL_TEXTURE_2D, gles.GL_TEXTURE_WRAP_T, @intCast(gles.GL_CLAMP_TO_EDGE));
    glUniform1i(glGetUniformLocation(prog, "uTex"), 0);
    defer glDeleteTextures(1, &[_]gles.GLuint{tex});

    // A sampler object with LINEAR filtering + edge clamp, bound to unit 0. It overrides the
    // texture's NEAREST for the sampling.
    var so: gles.GLuint = 0;
    glGenSamplers(1, @ptrCast(&so));
    glSamplerParameteri(so, gles.GL_TEXTURE_MIN_FILTER, @intCast(gles.GL_LINEAR));
    glSamplerParameteri(so, gles.GL_TEXTURE_MAG_FILTER, @intCast(gles.GL_LINEAR));
    glSamplerParameteri(so, gles.GL_TEXTURE_WRAP_S, @intCast(gles.GL_CLAMP_TO_EDGE));
    glSamplerParameteri(so, gles.GL_TEXTURE_WRAP_T, @intCast(gles.GL_CLAMP_TO_EDGE));
    defer glDeleteSamplers(1, &[_]gles.GLuint{so});

    const Vtx = extern struct { x: f32, y: f32, u: f32, v: f32 };
    const quad = [4]Vtx{
        .{ .x = -1, .y = -1, .u = 0, .v = 0 }, .{ .x = 1, .y = -1, .u = 1, .v = 0 },
        .{ .x = -1, .y = 1, .u = 0, .v = 1 },  .{ .x = 1, .y = 1, .u = 1, .v = 1 },
    };
    var vbo: gles.GLuint = 0;
    glGenBuffers(1, @ptrCast(&vbo));
    glBindBuffer(gles.GL_ARRAY_BUFFER, vbo);
    glBufferData(gles.GL_ARRAY_BUFFER, @sizeOf(@TypeOf(quad)), &quad, gles.GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, gles.GL_FLOAT, gles.GL_FALSE, @sizeOf(Vtx), @ptrFromInt(0));
    glVertexAttribPointer(1, 2, gles.GL_FLOAT, gles.GL_FALSE, @sizeOf(Vtx), @ptrFromInt(8));
    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);
    defer glDeleteBuffers(1, &[_]gles.GLuint{vbo});

    const s: *state.Surface = @ptrCast(@alignCast(surf.?));
    const dev = s.display.device.?;
    const ci = (32 * 64 + 32) * 4; // the exact center = the 4-texel meeting point

    // With the LINEAR sampler object bound, the center blends all 4 texels: every channel partial.
    glBindSampler(0, so);
    // A gen'd sampler name becomes a real sampler object only once bound (GL spec).
    try std.testing.expectEqual(gles.GL_TRUE, glIsSampler(so));
    glClearColor(0, 0, 0, 1);
    glClear(gles.GL_COLOR_BUFFER_BIT);
    glDrawArrays(gles.GL_TRIANGLE_STRIP, 0, 4);
    try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());
    const pxL = try flushMap(dev, s);
    var blended = true;
    for (0..3) |k| if (pxL[ci + k] < 30 or pxL[ci + k] > 230) {
        blended = false;
    };
    try std.testing.expect(blended);

    // Unbind the sampler object -> the texture's own NEAREST applies: the center is a pure texel
    // (a corner-clamped single texel, so at least one channel saturates and another is ~0).
    glBindSampler(0, 0);
    glClear(gles.GL_COLOR_BUFFER_BIT);
    glDrawArrays(gles.GL_TRIANGLE_STRIP, 0, 4);
    try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());
    const pxN = try flushMap(dev, s);
    var saturated = false;
    for (0..3) |k| if (pxN[ci + k] > 230 or pxN[ci + k] < 20) {
        saturated = true;
    };
    try std.testing.expect(saturated);
}

test "textured quad (SPIR-V): a SPIR-V textured FS samples the same checkerboard" {
    // The GLSL frontend produces the SPIR-V the draw path JITs. Prove a textured shader
    // compiled from GLSL ES source (the glmark2 case) reads the bound texture by sampling
    // a uniform-color 1x1 texture -> a flat fill the FS multiplies through unchanged.
    _ = eglBindAPI(egl.EGL_OPENGL_ES_API);
    const dpy = eglGetPlatformDisplay(egl.EGL_PLATFORM_SURFACELESS_MESA, @ptrFromInt(0x7E1F), null);
    try std.testing.expectEqual(egl.EGL_TRUE, eglInitialize(dpy, null, null));
    defer _ = eglTerminate(dpy);
    const cfg = state.configHandle(0);
    const ctx = eglCreateContext(dpy, cfg, egl.EGL_NO_CONTEXT, null);
    const W: EGLint = 64;
    const pb = [_]EGLint{ egl.EGL_WIDTH, W, egl.EGL_HEIGHT, W, egl.EGL_NONE };
    const surf = eglCreatePbufferSurface(dpy, cfg, &pb);
    try std.testing.expectEqual(egl.EGL_TRUE, eglMakeCurrent(dpy, surf, surf, ctx));

    const vs_src: [*:0]const gles.GLchar =
        \\attribute vec2 position;
        \\attribute vec2 aUV;
        \\varying vec2 vUV;
        \\void main() { gl_Position = vec4(position, 0.0, 1.0); vUV = aUV; }
    ;
    // A textured FS that tints the sampled texel by a uniform: proves a sampler AND a UBO
    // uniform coexist (the binding-model gotcha). Magenta texture * a half tint.
    const fs_src: [*:0]const gles.GLchar =
        \\precision mediump float;
        \\uniform sampler2D uTex;
        \\uniform vec4 uTint;
        \\varying vec2 vUV;
        \\void main() { gl_FragColor = texture2D(uTex, vUV) * uTint; }
    ;
    const vs = glCreateShader(gles.GL_VERTEX_SHADER);
    glShaderSource(vs, 1, &[_]?[*:0]const gles.GLchar{vs_src}, null);
    glCompileShader(vs);
    const fs = glCreateShader(gles.GL_FRAGMENT_SHADER);
    glShaderSource(fs, 1, &[_]?[*:0]const gles.GLchar{fs_src}, null);
    glCompileShader(fs);
    var ok: gles.GLint = 0;
    glGetShaderiv(fs, gles.GL_COMPILE_STATUS, &ok);
    try std.testing.expectEqual(@as(gles.GLint, gles.GL_TRUE), ok);
    const prog = glCreateProgram();
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glBindAttribLocation(prog, 0, "position");
    glBindAttribLocation(prog, 1, "aUV");
    glLinkProgram(prog);
    glUseProgram(prog);

    // A 1x1 magenta texture (255,0,255,255).
    const tex_px = [_]u8{ 255, 0, 255, 255 };
    var tex: gles.GLuint = 0;
    glGenTextures(1, @ptrCast(&tex));
    glActiveTexture(gles.GL_TEXTURE0);
    glBindTexture(gles.GL_TEXTURE_2D, tex);
    glTexImage2D(gles.GL_TEXTURE_2D, 0, gles.GL_RGBA, 1, 1, 0, gles.GL_RGBA, gles.GL_UNSIGNED_BYTE, &tex_px);
    glTexParameteri(gles.GL_TEXTURE_2D, gles.GL_TEXTURE_MIN_FILTER, @intCast(gles.GL_NEAREST));
    glTexParameteri(gles.GL_TEXTURE_2D, gles.GL_TEXTURE_MAG_FILTER, @intCast(gles.GL_NEAREST));
    glUniform1i(glGetUniformLocation(prog, "uTex"), 0);
    glUniform4f(glGetUniformLocation(prog, "uTint"), 1.0, 1.0, 1.0, 1.0); // pass the texel through

    const Vtx = extern struct { x: f32, y: f32, u: f32, v: f32 };
    const quad = [6]Vtx{
        .{ .x = -1, .y = -1, .u = 0, .v = 0 }, .{ .x = 1, .y = -1, .u = 1, .v = 0 }, .{ .x = 1, .y = 1, .u = 1, .v = 1 },
        .{ .x = -1, .y = -1, .u = 0, .v = 0 }, .{ .x = 1, .y = 1, .u = 1, .v = 1 },  .{ .x = -1, .y = 1, .u = 0, .v = 1 },
    };
    var vbo: gles.GLuint = 0;
    glGenBuffers(1, @ptrCast(&vbo));
    glBindBuffer(gles.GL_ARRAY_BUFFER, vbo);
    glBufferData(gles.GL_ARRAY_BUFFER, @sizeOf(@TypeOf(quad)), &quad, gles.GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, gles.GL_FLOAT, gles.GL_FALSE, @sizeOf(Vtx), @ptrFromInt(0));
    glVertexAttribPointer(1, 2, gles.GL_FLOAT, gles.GL_FALSE, @sizeOf(Vtx), @ptrFromInt(8));
    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);

    glClearColor(0, 0, 0, 1);
    glClear(gles.GL_COLOR_BUFFER_BIT);
    glDrawArrays(gles.GL_TRIANGLES, 0, 6);
    try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());

    const s: *state.Surface = @ptrCast(@alignCast(surf.?));
    const dev = s.display.device.?;
    const px = try flushMap(dev, s);
    const c = (32 * 64 + 32) * 4;
    // The whole quad samples the 1x1 magenta texel * white tint -> magenta.
    try std.testing.expect(px[c + 0] > 200 and px[c + 1] < 60 and px[c + 2] > 200);

    glDeleteBuffers(1, &[_]gles.GLuint{vbo});
    glDeleteTextures(1, &[_]gles.GLuint{tex});
    glDeleteProgram(prog);
    glDeleteShader(vs);
    glDeleteShader(fs);
    _ = eglMakeCurrent(dpy, null, null, egl.EGL_NO_CONTEXT);
    _ = eglDestroySurface(dpy, surf);
    _ = eglDestroyContext(dpy, ctx);
}

test "texture formats: a GL_BGRA_EXT upload samples with B<->R swapped (the compositor case)" {
    // A compositor sampling a client's BGRA8888 buffer uploads it as GL_BGRA_EXT. The texel
    // bytes [B=255,G=0,R=0,A=255] must sample as BLUE (0,0,255). A naive RGBA read would give
    // red. A blue result proves the BGRA decode works end-to-end through the sampler.
    _ = eglBindAPI(egl.EGL_OPENGL_ES_API);
    const dpy = eglGetPlatformDisplay(egl.EGL_PLATFORM_SURFACELESS_MESA, @ptrFromInt(0x7E2A), null);
    try std.testing.expectEqual(egl.EGL_TRUE, eglInitialize(dpy, null, null));
    defer _ = eglTerminate(dpy);
    const cfg = state.configHandle(0);
    const ctx = eglCreateContext(dpy, cfg, egl.EGL_NO_CONTEXT, null);
    const W: EGLint = 64;
    const pb = [_]EGLint{ egl.EGL_WIDTH, W, egl.EGL_HEIGHT, W, egl.EGL_NONE };
    const surf = eglCreatePbufferSurface(dpy, cfg, &pb);
    try std.testing.expectEqual(egl.EGL_TRUE, eglMakeCurrent(dpy, surf, surf, ctx));

    const vs_src: [*:0]const gles.GLchar =
        \\attribute vec2 position;
        \\attribute vec2 aUV;
        \\varying vec2 vUV;
        \\void main() { gl_Position = vec4(position, 0.0, 1.0); vUV = aUV; }
    ;
    const fs_src: [*:0]const gles.GLchar =
        \\precision mediump float;
        \\uniform sampler2D uTex;
        \\varying vec2 vUV;
        \\void main() { gl_FragColor = texture2D(uTex, vUV); }
    ;
    const vs = glCreateShader(gles.GL_VERTEX_SHADER);
    glShaderSource(vs, 1, &[_]?[*:0]const gles.GLchar{vs_src}, null);
    glCompileShader(vs);
    const fs = glCreateShader(gles.GL_FRAGMENT_SHADER);
    glShaderSource(fs, 1, &[_]?[*:0]const gles.GLchar{fs_src}, null);
    glCompileShader(fs);
    const prog = glCreateProgram();
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glBindAttribLocation(prog, 0, "position");
    glBindAttribLocation(prog, 1, "aUV");
    glLinkProgram(prog);
    glUseProgram(prog);

    // 1x1 texture uploaded as GL_BGRA_EXT: bytes B=255,G=0,R=0,A=255 -> RGBA (0,0,255,255).
    const bgra_px = [_]u8{ 255, 0, 0, 255 };
    var tex: gles.GLuint = 0;
    glGenTextures(1, @ptrCast(&tex));
    glActiveTexture(gles.GL_TEXTURE0);
    glBindTexture(gles.GL_TEXTURE_2D, tex);
    glTexImage2D(gles.GL_TEXTURE_2D, 0, @intCast(gles.GL_BGRA_EXT), 1, 1, 0, gles.GL_BGRA_EXT, gles.GL_UNSIGNED_BYTE, &bgra_px);
    try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());
    glTexParameteri(gles.GL_TEXTURE_2D, gles.GL_TEXTURE_MIN_FILTER, @intCast(gles.GL_NEAREST));
    glTexParameteri(gles.GL_TEXTURE_2D, gles.GL_TEXTURE_MAG_FILTER, @intCast(gles.GL_NEAREST));
    glUniform1i(glGetUniformLocation(prog, "uTex"), 0);

    const Vtx = extern struct { x: f32, y: f32, u: f32, v: f32 };
    const quad = [6]Vtx{
        .{ .x = -1, .y = -1, .u = 0, .v = 0 }, .{ .x = 1, .y = -1, .u = 1, .v = 0 }, .{ .x = 1, .y = 1, .u = 1, .v = 1 },
        .{ .x = -1, .y = -1, .u = 0, .v = 0 }, .{ .x = 1, .y = 1, .u = 1, .v = 1 },  .{ .x = -1, .y = 1, .u = 0, .v = 1 },
    };
    var vbo: gles.GLuint = 0;
    glGenBuffers(1, @ptrCast(&vbo));
    glBindBuffer(gles.GL_ARRAY_BUFFER, vbo);
    glBufferData(gles.GL_ARRAY_BUFFER, @sizeOf(@TypeOf(quad)), &quad, gles.GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, gles.GL_FLOAT, gles.GL_FALSE, @sizeOf(Vtx), @ptrFromInt(0));
    glVertexAttribPointer(1, 2, gles.GL_FLOAT, gles.GL_FALSE, @sizeOf(Vtx), @ptrFromInt(8));
    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);

    glClearColor(0, 0, 0, 1);
    glClear(gles.GL_COLOR_BUFFER_BIT);
    glDrawArrays(gles.GL_TRIANGLES, 0, 6);
    try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());

    const s: *state.Surface = @ptrCast(@alignCast(surf.?));
    const dev = s.display.device.?;
    const px = try flushMap(dev, s);
    const c = (32 * 64 + 32) * 4;
    // Blue (not red): the BGRA decode swapped B<->R correctly.
    try std.testing.expect(px[c + 0] < 60 and px[c + 1] < 60 and px[c + 2] > 200);

    glDeleteBuffers(1, &[_]gles.GLuint{vbo});
    glDeleteTextures(1, &[_]gles.GLuint{tex});
    glDeleteProgram(prog);
    glDeleteShader(vs);
    glDeleteShader(fs);
    _ = eglMakeCurrent(dpy, null, null, egl.EGL_NO_CONTEXT);
    _ = eglDestroySurface(dpy, surf);
    _ = eglDestroyContext(dpy, ctx);
}

test "texture formats: sRGB decodes the EOTF on sample; half-float stores fp16 texels (end-to-end)" {
    _ = eglBindAPI(egl.EGL_OPENGL_ES_API);
    const dpy = eglGetPlatformDisplay(egl.EGL_PLATFORM_SURFACELESS_MESA, @ptrFromInt(0x7E3B), null);
    try std.testing.expectEqual(egl.EGL_TRUE, eglInitialize(dpy, null, null));
    defer _ = eglTerminate(dpy);

    // Render a 1x1 texture (uploaded with the given internalformat/format/type/bytes) through
    // a passthrough textured quad; return the center pixel's R channel from the rgba8 backbuffer.
    const sampleR = struct {
        fn go(d: EGLDisplay, internalformat: gles.GLint, format: gles.GLenum, gl_type: gles.GLenum, px_bytes: []const u8) !u8 {
            const cfg = state.configHandle(0);
            const ctx = eglCreateContext(d, cfg, egl.EGL_NO_CONTEXT, null);
            defer _ = eglDestroyContext(d, ctx);
            const pb = [_]EGLint{ egl.EGL_WIDTH, 64, egl.EGL_HEIGHT, 64, egl.EGL_NONE };
            const surf = eglCreatePbufferSurface(d, cfg, &pb);
            defer _ = eglDestroySurface(d, surf);
            _ = eglMakeCurrent(d, surf, surf, ctx);
            defer _ = eglMakeCurrent(d, null, null, egl.EGL_NO_CONTEXT);

            const vs_src: [*:0]const gles.GLchar =
                \\attribute vec2 position;
                \\attribute vec2 aUV;
                \\varying vec2 vUV;
                \\void main() { gl_Position = vec4(position, 0.0, 1.0); vUV = aUV; }
            ;
            const fs_src: [*:0]const gles.GLchar =
                \\precision mediump float;
                \\uniform sampler2D uTex;
                \\varying vec2 vUV;
                \\void main() { gl_FragColor = texture2D(uTex, vUV); }
            ;
            const vs = glCreateShader(gles.GL_VERTEX_SHADER);
            glShaderSource(vs, 1, &[_]?[*:0]const gles.GLchar{vs_src}, null);
            glCompileShader(vs);
            const fs = glCreateShader(gles.GL_FRAGMENT_SHADER);
            glShaderSource(fs, 1, &[_]?[*:0]const gles.GLchar{fs_src}, null);
            glCompileShader(fs);
            const prog = glCreateProgram();
            glAttachShader(prog, vs);
            glAttachShader(prog, fs);
            glBindAttribLocation(prog, 0, "position");
            glBindAttribLocation(prog, 1, "aUV");
            glLinkProgram(prog);
            glUseProgram(prog);

            var tex: gles.GLuint = 0;
            glGenTextures(1, @ptrCast(&tex));
            glActiveTexture(gles.GL_TEXTURE0);
            glBindTexture(gles.GL_TEXTURE_2D, tex);
            glTexImage2D(gles.GL_TEXTURE_2D, 0, internalformat, 1, 1, 0, format, gl_type, px_bytes.ptr);
            try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());
            glTexParameteri(gles.GL_TEXTURE_2D, gles.GL_TEXTURE_MIN_FILTER, @intCast(gles.GL_NEAREST));
            glTexParameteri(gles.GL_TEXTURE_2D, gles.GL_TEXTURE_MAG_FILTER, @intCast(gles.GL_NEAREST));
            glUniform1i(glGetUniformLocation(prog, "uTex"), 0);

            const Vtx = extern struct { x: f32, y: f32, u: f32, v: f32 };
            const quad = [6]Vtx{
                .{ .x = -1, .y = -1, .u = 0, .v = 0 }, .{ .x = 1, .y = -1, .u = 1, .v = 0 }, .{ .x = 1, .y = 1, .u = 1, .v = 1 },
                .{ .x = -1, .y = -1, .u = 0, .v = 0 }, .{ .x = 1, .y = 1, .u = 1, .v = 1 },  .{ .x = -1, .y = 1, .u = 0, .v = 1 },
            };
            var vbo: gles.GLuint = 0;
            glGenBuffers(1, @ptrCast(&vbo));
            glBindBuffer(gles.GL_ARRAY_BUFFER, vbo);
            glBufferData(gles.GL_ARRAY_BUFFER, @sizeOf(@TypeOf(quad)), &quad, gles.GL_STATIC_DRAW);
            glVertexAttribPointer(0, 2, gles.GL_FLOAT, gles.GL_FALSE, @sizeOf(Vtx), @ptrFromInt(0));
            glVertexAttribPointer(1, 2, gles.GL_FLOAT, gles.GL_FALSE, @sizeOf(Vtx), @ptrFromInt(8));
            glEnableVertexAttribArray(0);
            glEnableVertexAttribArray(1);
            glClearColor(0, 0, 0, 1);
            glClear(gles.GL_COLOR_BUFFER_BIT);
            glDrawArrays(gles.GL_TRIANGLES, 0, 6);
            try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());

            const s: *state.Surface = @ptrCast(@alignCast(surf.?));
            const dev = s.display.device.?;
            const px = try flushMap(dev, s);
            const r = px[(32 * 64 + 32) * 4 + 0];
            glDeleteBuffers(1, &[_]gles.GLuint{vbo});
            glDeleteTextures(1, &[_]gles.GLuint{tex});
            glDeleteProgram(prog);
            glDeleteShader(vs);
            glDeleteShader(fs);
            return r;
        }
    }.go;

    // sRGB: an 8-bit value 188 decodes to LINEAR ~0.5 on sample -> pixel ~128. A plain rgba8
    // upload of the same byte samples 188/255 -> ~188. So sRGB (~128) vs unorm (~188) proves
    // the sample-time sRGB decode.
    const srgb_r = try sampleR(dpy, @intCast(gles.GL_SRGB_ALPHA_EXT), gles.GL_SRGB_ALPHA_EXT, gles.GL_UNSIGNED_BYTE, &[_]u8{ 188, 188, 188, 255 });
    try std.testing.expect(srgb_r > 118 and srgb_r < 140);
    const unorm_r = try sampleR(dpy, @intCast(gles.GL_RGBA), gles.GL_RGBA, gles.GL_UNSIGNED_BYTE, &[_]u8{ 188, 188, 188, 255 });
    try std.testing.expect(unorm_r > 180);

    // Half-float (fp16): a texture uploaded as GL_HALF_FLOAT_OES stores IEEE half texels. R=0.5
    // (fp16 0x3800) samples back to ~128 in the rgba8 backbuffer, proving the fp16 upload +
    // the sampler's fp16 read work end-to-end.
    const half_half: u16 = @bitCast(@as(f16, 0.5));
    const hf = [_]u8{ @truncate(half_half), @truncate(half_half >> 8) } ** 4; // R=G=B=A=0.5 fp16
    const half_r = try sampleR(dpy, @intCast(gles.GL_RGBA), gles.GL_RGBA, gles.GL_HALF_FLOAT_OES, &hf);
    try std.testing.expect(half_r > 118 and half_r < 140);
}

test "texture formats: an ETC1-compressed texture decompresses and samples (end-to-end)" {
    _ = eglBindAPI(egl.EGL_OPENGL_ES_API);
    const dpy = eglGetPlatformDisplay(egl.EGL_PLATFORM_SURFACELESS_MESA, @ptrFromInt(0x7E4C), null);
    try std.testing.expectEqual(egl.EGL_TRUE, eglInitialize(dpy, null, null));
    defer _ = eglTerminate(dpy);
    const cfg = state.configHandle(0);
    const ctx = eglCreateContext(dpy, cfg, egl.EGL_NO_CONTEXT, null);
    const W: EGLint = 64;
    const pb = [_]EGLint{ egl.EGL_WIDTH, W, egl.EGL_HEIGHT, W, egl.EGL_NONE };
    const surf = eglCreatePbufferSurface(dpy, cfg, &pb);
    try std.testing.expectEqual(egl.EGL_TRUE, eglMakeCurrent(dpy, surf, surf, ctx));

    const vs_src: [*:0]const gles.GLchar =
        \\attribute vec2 position;
        \\attribute vec2 aUV;
        \\varying vec2 vUV;
        \\void main() { gl_Position = vec4(position, 0.0, 1.0); vUV = aUV; }
    ;
    const fs_src: [*:0]const gles.GLchar =
        \\precision mediump float;
        \\uniform sampler2D uTex;
        \\varying vec2 vUV;
        \\void main() { gl_FragColor = texture2D(uTex, vUV); }
    ;
    const vs = glCreateShader(gles.GL_VERTEX_SHADER);
    glShaderSource(vs, 1, &[_]?[*:0]const gles.GLchar{vs_src}, null);
    glCompileShader(vs);
    const fs = glCreateShader(gles.GL_FRAGMENT_SHADER);
    glShaderSource(fs, 1, &[_]?[*:0]const gles.GLchar{fs_src}, null);
    glCompileShader(fs);
    const prog = glCreateProgram();
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glBindAttribLocation(prog, 0, "position");
    glBindAttribLocation(prog, 1, "aUV");
    glLinkProgram(prog);
    glUseProgram(prog);

    // A 4x4 ETC1 block: individual mode, both sub-blocks red (RGB444 F,0,0), table 0, all
    // pixel selectors 0 -> every texel ~= (255, 2, 2). Proves the CPU ETC1 decode + sampling.
    const etc1_block = [_]u8{ 0xFF, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 };
    var tex: gles.GLuint = 0;
    glGenTextures(1, @ptrCast(&tex));
    glActiveTexture(gles.GL_TEXTURE0);
    glBindTexture(gles.GL_TEXTURE_2D, tex);
    glCompressedTexImage2D(gles.GL_TEXTURE_2D, 0, gles.GL_ETC1_RGB8_OES, 4, 4, 0, etc1_block.len, &etc1_block);
    try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());
    glTexParameteri(gles.GL_TEXTURE_2D, gles.GL_TEXTURE_MIN_FILTER, @intCast(gles.GL_NEAREST));
    glTexParameteri(gles.GL_TEXTURE_2D, gles.GL_TEXTURE_MAG_FILTER, @intCast(gles.GL_NEAREST));
    glUniform1i(glGetUniformLocation(prog, "uTex"), 0);

    const Vtx = extern struct { x: f32, y: f32, u: f32, v: f32 };
    const quad = [6]Vtx{
        .{ .x = -1, .y = -1, .u = 0, .v = 0 }, .{ .x = 1, .y = -1, .u = 1, .v = 0 }, .{ .x = 1, .y = 1, .u = 1, .v = 1 },
        .{ .x = -1, .y = -1, .u = 0, .v = 0 }, .{ .x = 1, .y = 1, .u = 1, .v = 1 },  .{ .x = -1, .y = 1, .u = 0, .v = 1 },
    };
    var vbo: gles.GLuint = 0;
    glGenBuffers(1, @ptrCast(&vbo));
    glBindBuffer(gles.GL_ARRAY_BUFFER, vbo);
    glBufferData(gles.GL_ARRAY_BUFFER, @sizeOf(@TypeOf(quad)), &quad, gles.GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, gles.GL_FLOAT, gles.GL_FALSE, @sizeOf(Vtx), @ptrFromInt(0));
    glVertexAttribPointer(1, 2, gles.GL_FLOAT, gles.GL_FALSE, @sizeOf(Vtx), @ptrFromInt(8));
    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);
    glClearColor(0, 0, 0, 1);
    glClear(gles.GL_COLOR_BUFFER_BIT);
    glDrawArrays(gles.GL_TRIANGLES, 0, 6);
    try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());

    const s: *state.Surface = @ptrCast(@alignCast(surf.?));
    const dev = s.display.device.?;
    const px = try flushMap(dev, s);
    const c = (32 * 64 + 32) * 4;
    // The decompressed ETC1 texel is red.
    try std.testing.expect(px[c + 0] > 200 and px[c + 1] < 40 and px[c + 2] < 40);

    glDeleteBuffers(1, &[_]gles.GLuint{vbo});
    glDeleteTextures(1, &[_]gles.GLuint{tex});
    glDeleteProgram(prog);
    glDeleteShader(vs);
    glDeleteShader(fs);
    _ = eglMakeCurrent(dpy, null, null, egl.EGL_NO_CONTEXT);
    _ = eglDestroySurface(dpy, surf);
    _ = eglDestroyContext(dpy, ctx);
}

test "texture formats: an S3TC/DXT1-compressed texture decompresses and samples (end-to-end)" {
    _ = eglBindAPI(egl.EGL_OPENGL_ES_API);
    const dpy = eglGetPlatformDisplay(egl.EGL_PLATFORM_SURFACELESS_MESA, @ptrFromInt(0x7E5D), null);
    try std.testing.expectEqual(egl.EGL_TRUE, eglInitialize(dpy, null, null));
    defer _ = eglTerminate(dpy);
    const cfg = state.configHandle(0);
    const ctx = eglCreateContext(dpy, cfg, egl.EGL_NO_CONTEXT, null);
    const W: EGLint = 64;
    const pb = [_]EGLint{ egl.EGL_WIDTH, W, egl.EGL_HEIGHT, W, egl.EGL_NONE };
    const surf = eglCreatePbufferSurface(dpy, cfg, &pb);
    try std.testing.expectEqual(egl.EGL_TRUE, eglMakeCurrent(dpy, surf, surf, ctx));

    const vs_src: [*:0]const gles.GLchar =
        \\attribute vec2 position;
        \\attribute vec2 aUV;
        \\varying vec2 vUV;
        \\void main() { gl_Position = vec4(position, 0.0, 1.0); vUV = aUV; }
    ;
    const fs_src: [*:0]const gles.GLchar =
        \\precision mediump float;
        \\uniform sampler2D uTex;
        \\varying vec2 vUV;
        \\void main() { gl_FragColor = texture2D(uTex, vUV); }
    ;
    const vs = glCreateShader(gles.GL_VERTEX_SHADER);
    glShaderSource(vs, 1, &[_]?[*:0]const gles.GLchar{vs_src}, null);
    glCompileShader(vs);
    const fs = glCreateShader(gles.GL_FRAGMENT_SHADER);
    glShaderSource(fs, 1, &[_]?[*:0]const gles.GLchar{fs_src}, null);
    glCompileShader(fs);
    const prog = glCreateProgram();
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glBindAttribLocation(prog, 0, "position");
    glBindAttribLocation(prog, 1, "aUV");
    glLinkProgram(prog);
    glUseProgram(prog);

    // A 4x4 DXT1 block: c0 = c1 = green (RGB565 0x07E0), all selectors 0 -> every texel green.
    const dxt1_block = [_]u8{ 0xE0, 0x07, 0xE0, 0x07, 0x00, 0x00, 0x00, 0x00 };
    var tex: gles.GLuint = 0;
    glGenTextures(1, @ptrCast(&tex));
    glActiveTexture(gles.GL_TEXTURE0);
    glBindTexture(gles.GL_TEXTURE_2D, tex);
    glCompressedTexImage2D(gles.GL_TEXTURE_2D, 0, gles.GL_COMPRESSED_RGB_S3TC_DXT1_EXT, 4, 4, 0, dxt1_block.len, &dxt1_block);
    try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());
    glTexParameteri(gles.GL_TEXTURE_2D, gles.GL_TEXTURE_MIN_FILTER, @intCast(gles.GL_NEAREST));
    glTexParameteri(gles.GL_TEXTURE_2D, gles.GL_TEXTURE_MAG_FILTER, @intCast(gles.GL_NEAREST));
    glUniform1i(glGetUniformLocation(prog, "uTex"), 0);

    const Vtx = extern struct { x: f32, y: f32, u: f32, v: f32 };
    const quad = [6]Vtx{
        .{ .x = -1, .y = -1, .u = 0, .v = 0 }, .{ .x = 1, .y = -1, .u = 1, .v = 0 }, .{ .x = 1, .y = 1, .u = 1, .v = 1 },
        .{ .x = -1, .y = -1, .u = 0, .v = 0 }, .{ .x = 1, .y = 1, .u = 1, .v = 1 },  .{ .x = -1, .y = 1, .u = 0, .v = 1 },
    };
    var vbo: gles.GLuint = 0;
    glGenBuffers(1, @ptrCast(&vbo));
    glBindBuffer(gles.GL_ARRAY_BUFFER, vbo);
    glBufferData(gles.GL_ARRAY_BUFFER, @sizeOf(@TypeOf(quad)), &quad, gles.GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, gles.GL_FLOAT, gles.GL_FALSE, @sizeOf(Vtx), @ptrFromInt(0));
    glVertexAttribPointer(1, 2, gles.GL_FLOAT, gles.GL_FALSE, @sizeOf(Vtx), @ptrFromInt(8));
    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);
    glClearColor(0, 0, 0, 1);
    glClear(gles.GL_COLOR_BUFFER_BIT);
    glDrawArrays(gles.GL_TRIANGLES, 0, 6);
    try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());

    const s: *state.Surface = @ptrCast(@alignCast(surf.?));
    const dev = s.display.device.?;
    const px = try flushMap(dev, s);
    const c = (32 * 64 + 32) * 4;
    // The decompressed DXT1 texel is green.
    try std.testing.expect(px[c + 1] > 200 and px[c + 0] < 40 and px[c + 2] < 40);

    glDeleteBuffers(1, &[_]gles.GLuint{vbo});
    glDeleteTextures(1, &[_]gles.GLuint{tex});
    glDeleteProgram(prog);
    glDeleteShader(vs);
    glDeleteShader(fs);
    _ = eglMakeCurrent(dpy, null, null, egl.EGL_NO_CONTEXT);
    _ = eglDestroySurface(dpy, surf);
    _ = eglDestroyContext(dpy, ctx);
}

test "texture formats: RGTC/BC4 (RED) and BC5 (RG) compressed textures decompress and sample (end-to-end)" {
    _ = eglBindAPI(egl.EGL_OPENGL_ES_API);
    const dpy = eglGetPlatformDisplay(egl.EGL_PLATFORM_SURFACELESS_MESA, @ptrFromInt(0x7E5E), null);
    try std.testing.expectEqual(egl.EGL_TRUE, eglInitialize(dpy, null, null));
    defer _ = eglTerminate(dpy);
    const cfg = state.configHandle(0);
    const ctx = eglCreateContext(dpy, cfg, egl.EGL_NO_CONTEXT, null);
    const W: EGLint = 64;
    const pb = [_]EGLint{ egl.EGL_WIDTH, W, egl.EGL_HEIGHT, W, egl.EGL_NONE };
    const surf = eglCreatePbufferSurface(dpy, cfg, &pb);
    try std.testing.expectEqual(egl.EGL_TRUE, eglMakeCurrent(dpy, surf, surf, ctx));

    const vs_src: [*:0]const gles.GLchar =
        \\attribute vec2 position;
        \\attribute vec2 aUV;
        \\varying vec2 vUV;
        \\void main() { gl_Position = vec4(position, 0.0, 1.0); vUV = aUV; }
    ;
    const fs_src: [*:0]const gles.GLchar =
        \\precision mediump float;
        \\uniform sampler2D uTex;
        \\varying vec2 vUV;
        \\void main() { gl_FragColor = texture2D(uTex, vUV); }
    ;
    const vs = glCreateShader(gles.GL_VERTEX_SHADER);
    glShaderSource(vs, 1, &[_]?[*:0]const gles.GLchar{vs_src}, null);
    glCompileShader(vs);
    const fs = glCreateShader(gles.GL_FRAGMENT_SHADER);
    glShaderSource(fs, 1, &[_]?[*:0]const gles.GLchar{fs_src}, null);
    glCompileShader(fs);
    const prog = glCreateProgram();
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glBindAttribLocation(prog, 0, "position");
    glBindAttribLocation(prog, 1, "aUV");
    glLinkProgram(prog);
    glUseProgram(prog);

    const Vtx = extern struct { x: f32, y: f32, u: f32, v: f32 };
    const quad = [6]Vtx{
        .{ .x = -1, .y = -1, .u = 0, .v = 0 }, .{ .x = 1, .y = -1, .u = 1, .v = 0 }, .{ .x = 1, .y = 1, .u = 1, .v = 1 },
        .{ .x = -1, .y = -1, .u = 0, .v = 0 }, .{ .x = 1, .y = 1, .u = 1, .v = 1 },  .{ .x = -1, .y = 1, .u = 0, .v = 1 },
    };
    var vbo: gles.GLuint = 0;
    glGenBuffers(1, @ptrCast(&vbo));
    glBindBuffer(gles.GL_ARRAY_BUFFER, vbo);
    glBufferData(gles.GL_ARRAY_BUFFER, @sizeOf(@TypeOf(quad)), &quad, gles.GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, gles.GL_FLOAT, gles.GL_FALSE, @sizeOf(Vtx), @ptrFromInt(0));
    glVertexAttribPointer(1, 2, gles.GL_FLOAT, gles.GL_FALSE, @sizeOf(Vtx), @ptrFromInt(8));
    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);
    glUniform1i(glGetUniformLocation(prog, "uTex"), 0);

    const s: *state.Surface = @ptrCast(@alignCast(surf.?));
    const dev = s.display.device.?;
    const c = (32 * 64 + 32) * 4;

    // BC4: red0=200, red1=100 (200>100 -> a[0]=200); all 16 indices 0 -> every texel R=200, G=B=0.
    {
        const bc4 = [_]u8{ 200, 100, 0, 0, 0, 0, 0, 0 };
        var tex: gles.GLuint = 0;
        glGenTextures(1, @ptrCast(&tex));
        glActiveTexture(gles.GL_TEXTURE0);
        glBindTexture(gles.GL_TEXTURE_2D, tex);
        glCompressedTexImage2D(gles.GL_TEXTURE_2D, 0, gles.GL_COMPRESSED_RED_RGTC1_EXT, 4, 4, 0, bc4.len, &bc4);
        try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());
        glTexParameteri(gles.GL_TEXTURE_2D, gles.GL_TEXTURE_MIN_FILTER, @intCast(gles.GL_NEAREST));
        glTexParameteri(gles.GL_TEXTURE_2D, gles.GL_TEXTURE_MAG_FILTER, @intCast(gles.GL_NEAREST));
        glClearColor(0, 0, 0, 1);
        glClear(gles.GL_COLOR_BUFFER_BIT);
        glDrawArrays(gles.GL_TRIANGLES, 0, 6);
        try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());
        const px = try flushMap(dev, s);
        try std.testing.expect(px[c + 0] > 180 and px[c + 0] < 220 and px[c + 1] < 40 and px[c + 2] < 40);
        glDeleteTextures(1, &[_]gles.GLuint{tex});
    }
    // BC5: red block (200) + green block (150) -> R=200, G=150, B=0.
    {
        const bc5 = [_]u8{ 200, 100, 0, 0, 0, 0, 0, 0, 150, 100, 0, 0, 0, 0, 0, 0 };
        var tex: gles.GLuint = 0;
        glGenTextures(1, @ptrCast(&tex));
        glBindTexture(gles.GL_TEXTURE_2D, tex);
        glCompressedTexImage2D(gles.GL_TEXTURE_2D, 0, gles.GL_COMPRESSED_RG_RGTC2_EXT, 4, 4, 0, bc5.len, &bc5);
        try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());
        glTexParameteri(gles.GL_TEXTURE_2D, gles.GL_TEXTURE_MIN_FILTER, @intCast(gles.GL_NEAREST));
        glTexParameteri(gles.GL_TEXTURE_2D, gles.GL_TEXTURE_MAG_FILTER, @intCast(gles.GL_NEAREST));
        glClear(gles.GL_COLOR_BUFFER_BIT);
        glDrawArrays(gles.GL_TRIANGLES, 0, 6);
        try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());
        const px = try flushMap(dev, s);
        try std.testing.expect(px[c + 0] > 180 and px[c + 0] < 220 and px[c + 1] > 130 and px[c + 1] < 170 and px[c + 2] < 40);
        glDeleteTextures(1, &[_]gles.GLuint{tex});
    }

    glDeleteBuffers(1, &[_]gles.GLuint{vbo});
    glDeleteProgram(prog);
    glDeleteShader(vs);
    glDeleteShader(fs);
    _ = eglMakeCurrent(dpy, null, null, egl.EGL_NO_CONTEXT);
    _ = eglDestroySurface(dpy, surf);
    _ = eglDestroyContext(dpy, ctx);
}

test "primitives: GL_LINES draws a horizontal line (a thin band, not a fill, not blank)" {
    _ = eglBindAPI(egl.EGL_OPENGL_ES_API);
    const dpy = eglGetPlatformDisplay(egl.EGL_PLATFORM_SURFACELESS_MESA, @ptrFromInt(0x7E6E), null);
    try std.testing.expectEqual(egl.EGL_TRUE, eglInitialize(dpy, null, null));
    defer _ = eglTerminate(dpy);
    const cfg = state.configHandle(0);
    const ctx = eglCreateContext(dpy, cfg, egl.EGL_NO_CONTEXT, null);
    const W: EGLint = 64;
    const pb = [_]EGLint{ egl.EGL_WIDTH, W, egl.EGL_HEIGHT, W, egl.EGL_NONE };
    const surf = eglCreatePbufferSurface(dpy, cfg, &pb);
    try std.testing.expectEqual(egl.EGL_TRUE, eglMakeCurrent(dpy, surf, surf, ctx));

    const vs_src: [*:0]const gles.GLchar =
        \\attribute vec2 position;
        \\void main() { gl_Position = vec4(position, 0.0, 1.0); }
    ;
    const fs_src: [*:0]const gles.GLchar =
        \\precision mediump float;
        \\void main() { gl_FragColor = vec4(1.0, 0.0, 0.0, 1.0); }
    ;
    const vs = glCreateShader(gles.GL_VERTEX_SHADER);
    glShaderSource(vs, 1, &[_]?[*:0]const gles.GLchar{vs_src}, null);
    glCompileShader(vs);
    const fs = glCreateShader(gles.GL_FRAGMENT_SHADER);
    glShaderSource(fs, 1, &[_]?[*:0]const gles.GLchar{fs_src}, null);
    glCompileShader(fs);
    const prog = glCreateProgram();
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glBindAttribLocation(prog, 0, "position");
    glLinkProgram(prog);
    glUseProgram(prog);

    // A single horizontal line segment across the middle (NDC y = 0 -> window row ~32).
    const line = [2][2]f32{ .{ -0.9, 0.0 }, .{ 0.9, 0.0 } };
    var vbo: gles.GLuint = 0;
    glGenBuffers(1, @ptrCast(&vbo));
    glBindBuffer(gles.GL_ARRAY_BUFFER, vbo);
    glBufferData(gles.GL_ARRAY_BUFFER, @sizeOf(@TypeOf(line)), &line, gles.GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, gles.GL_FLOAT, gles.GL_FALSE, 8, @ptrFromInt(0));
    glEnableVertexAttribArray(0);

    glClearColor(0, 0, 0, 1);
    glClear(gles.GL_COLOR_BUFFER_BIT);
    glDrawArrays(gles.GL_LINES, 0, 2);
    try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());

    const s: *state.Surface = @ptrCast(@alignCast(surf.?));
    const dev = s.display.device.?;
    const px = try flushMap(dev, s);
    var red: usize = 0;
    var i: usize = 0;
    while (i < @as(usize, W) * W) : (i += 1) {
        if (px[i * 4 + 0] > 200 and px[i * 4 + 1] < 60) red += 1;
    }
    // A ~1px horizontal line across ~90% of the width: tens of pixels, NOT a fill (~thousands)
    // and NOT blank. That range distinguishes a line primitive from a triangle or nothing.
    try std.testing.expect(red > 15 and red < 500);
    // The center band has the line. The top rows are clear.
    var top_red = false;
    var x: usize = 0;
    while (x < W) : (x += 1) {
        if (px[(4 * @as(usize, W) + x) * 4 + 0] > 200) top_red = true;
    }
    try std.testing.expect(!top_red); // row 4 (far from the mid-line) is background

    glDeleteBuffers(1, &[_]gles.GLuint{vbo});
    glDeleteProgram(prog);
    glDeleteShader(vs);
    glDeleteShader(fs);
    _ = eglMakeCurrent(dpy, null, null, egl.EGL_NO_CONTEXT);
    _ = eglDestroySurface(dpy, surf);
    _ = eglDestroyContext(dpy, ctx);
}

test "primitives: glLineWidth makes a line thicker (more covered pixels)" {
    _ = eglBindAPI(egl.EGL_OPENGL_ES_API);
    const dpy = eglGetPlatformDisplay(egl.EGL_PLATFORM_SURFACELESS_MESA, @ptrFromInt(0x7E7F), null);
    try std.testing.expectEqual(egl.EGL_TRUE, eglInitialize(dpy, null, null));
    defer _ = eglTerminate(dpy);

    // Draw a horizontal line at the given width. Return the red pixel count.
    const lineRed = struct {
        fn go(d: EGLDisplay, w: gles.GLfloat) !usize {
            const cfg = state.configHandle(0);
            const ctx = eglCreateContext(d, cfg, egl.EGL_NO_CONTEXT, null);
            defer _ = eglDestroyContext(d, ctx);
            const pb = [_]EGLint{ egl.EGL_WIDTH, 64, egl.EGL_HEIGHT, 64, egl.EGL_NONE };
            const surf = eglCreatePbufferSurface(d, cfg, &pb);
            defer _ = eglDestroySurface(d, surf);
            _ = eglMakeCurrent(d, surf, surf, ctx);
            defer _ = eglMakeCurrent(d, null, null, egl.EGL_NO_CONTEXT);

            const vs_src: [*:0]const gles.GLchar =
                \\attribute vec2 position;
                \\void main() { gl_Position = vec4(position, 0.0, 1.0); }
            ;
            const fs_src: [*:0]const gles.GLchar =
                \\precision mediump float;
                \\void main() { gl_FragColor = vec4(1.0, 0.0, 0.0, 1.0); }
            ;
            const vs = glCreateShader(gles.GL_VERTEX_SHADER);
            glShaderSource(vs, 1, &[_]?[*:0]const gles.GLchar{vs_src}, null);
            glCompileShader(vs);
            const fs = glCreateShader(gles.GL_FRAGMENT_SHADER);
            glShaderSource(fs, 1, &[_]?[*:0]const gles.GLchar{fs_src}, null);
            glCompileShader(fs);
            const prog = glCreateProgram();
            glAttachShader(prog, vs);
            glAttachShader(prog, fs);
            glBindAttribLocation(prog, 0, "position");
            glLinkProgram(prog);
            glUseProgram(prog);

            const line = [2][2]f32{ .{ -0.9, 0.0 }, .{ 0.9, 0.0 } };
            var vbo: gles.GLuint = 0;
            glGenBuffers(1, @ptrCast(&vbo));
            glBindBuffer(gles.GL_ARRAY_BUFFER, vbo);
            glBufferData(gles.GL_ARRAY_BUFFER, @sizeOf(@TypeOf(line)), &line, gles.GL_STATIC_DRAW);
            glVertexAttribPointer(0, 2, gles.GL_FLOAT, gles.GL_FALSE, 8, @ptrFromInt(0));
            glEnableVertexAttribArray(0);
            glLineWidth(w);
            glClearColor(0, 0, 0, 1);
            glClear(gles.GL_COLOR_BUFFER_BIT);
            glDrawArrays(gles.GL_LINES, 0, 2);
            try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());

            const s: *state.Surface = @ptrCast(@alignCast(surf.?));
            const dev = s.display.device.?;
            const px = try flushMap(dev, s);
            var red: usize = 0;
            var i: usize = 0;
            while (i < 64 * 64) : (i += 1) {
                if (px[i * 4 + 0] > 200 and px[i * 4 + 1] < 60) red += 1;
            }
            glDeleteBuffers(1, &[_]gles.GLuint{vbo});
            glDeleteProgram(prog);
            glDeleteShader(vs);
            glDeleteShader(fs);
            return red;
        }
    }.go;

    const thin = try lineRed(dpy, 1.0);
    const thick = try lineRed(dpy, 5.0);
    // A 5px line covers clearly more pixels than a 1px line (roughly 5x the rows).
    try std.testing.expect(thick > thin + 50);
}

test "gl_PointSize: a VS may write gl_PointSize WITHOUT corrupting a location-0 varying" {
    // gl_PointSize is a scalar builtin output. If it were mis-routed to varying location 0
    // (the old behavior before the ATTR_POINT_SIZE routing), it would clobber vColor and the
    // triangle would NOT be clean green. This proves the builtin is accepted + isolated (its
    // value is currently dropped by the software JIT. Variable point size is a follow-up).
    _ = eglBindAPI(egl.EGL_OPENGL_ES_API);
    const dpy = eglGetPlatformDisplay(egl.EGL_PLATFORM_SURFACELESS_MESA, @ptrFromInt(0x7E90), null);
    try std.testing.expectEqual(egl.EGL_TRUE, eglInitialize(dpy, null, null));
    defer _ = eglTerminate(dpy);
    const cfg = state.configHandle(0);
    const ctx = eglCreateContext(dpy, cfg, egl.EGL_NO_CONTEXT, null);
    const W: EGLint = 64;
    const pb = [_]EGLint{ egl.EGL_WIDTH, W, egl.EGL_HEIGHT, W, egl.EGL_NONE };
    const surf = eglCreatePbufferSurface(dpy, cfg, &pb);
    try std.testing.expectEqual(egl.EGL_TRUE, eglMakeCurrent(dpy, surf, surf, ctx));

    const vs_src: [*:0]const gles.GLchar =
        \\attribute vec2 position;
        \\varying vec4 vColor;
        \\void main() {
        \\  gl_Position = vec4(position, 0.0, 1.0);
        \\  gl_PointSize = 4.0;
        \\  vColor = vec4(0.0, 1.0, 0.0, 1.0);
        \\}
    ;
    const fs_src: [*:0]const gles.GLchar =
        \\precision mediump float;
        \\varying vec4 vColor;
        \\void main() { gl_FragColor = vColor; }
    ;
    const vs = glCreateShader(gles.GL_VERTEX_SHADER);
    glShaderSource(vs, 1, &[_]?[*:0]const gles.GLchar{vs_src}, null);
    glCompileShader(vs);
    var ok: gles.GLint = 0;
    glGetShaderiv(vs, gles.GL_COMPILE_STATUS, &ok);
    try std.testing.expectEqual(@as(gles.GLint, gles.GL_TRUE), ok); // gl_PointSize compiles
    const fs = glCreateShader(gles.GL_FRAGMENT_SHADER);
    glShaderSource(fs, 1, &[_]?[*:0]const gles.GLchar{fs_src}, null);
    glCompileShader(fs);
    const prog = glCreateProgram();
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glBindAttribLocation(prog, 0, "position");
    glLinkProgram(prog);
    glUseProgram(prog);

    const tri = [3][2]f32{ .{ -0.8, -0.8 }, .{ 0.8, -0.8 }, .{ 0.0, 0.8 } };
    var vbo: gles.GLuint = 0;
    glGenBuffers(1, @ptrCast(&vbo));
    glBindBuffer(gles.GL_ARRAY_BUFFER, vbo);
    glBufferData(gles.GL_ARRAY_BUFFER, @sizeOf(@TypeOf(tri)), &tri, gles.GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, gles.GL_FLOAT, gles.GL_FALSE, 8, @ptrFromInt(0));
    glEnableVertexAttribArray(0);
    glClearColor(0, 0, 0, 1);
    glClear(gles.GL_COLOR_BUFFER_BIT);
    glDrawArrays(gles.GL_TRIANGLES, 0, 3);
    try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());

    const s: *state.Surface = @ptrCast(@alignCast(surf.?));
    const dev = s.display.device.?;
    const px = try flushMap(dev, s);
    const c = (32 * 64 + 32) * 4;
    // Clean green: the varying survived (gl_PointSize did not overwrite location 0).
    try std.testing.expect(px[c + 1] > 200 and px[c + 0] < 40 and px[c + 2] < 40);

    try std.testing.expect(px[c + 1] > 200 and px[c + 0] < 40 and px[c + 2] < 40);

    glDeleteBuffers(1, &[_]gles.GLuint{vbo});
    glDeleteProgram(prog);
    glDeleteShader(vs);
    glDeleteShader(fs);
    _ = eglMakeCurrent(dpy, null, null, egl.EGL_NO_CONTEXT);
    _ = eglDestroySurface(dpy, surf);
    _ = eglDestroyContext(dpy, ctx);
}

test "gl_PointSize: a larger gl_PointSize renders a bigger point (software consumes it)" {
    _ = eglBindAPI(egl.EGL_OPENGL_ES_API);
    const dpy = eglGetPlatformDisplay(egl.EGL_PLATFORM_SURFACELESS_MESA, @ptrFromInt(0x7EA1), null);
    try std.testing.expectEqual(egl.EGL_TRUE, eglInitialize(dpy, null, null));
    defer _ = eglTerminate(dpy);

    // Render one GL_POINTS point at the center with gl_PointSize = `sz`; return the red count.
    const pointRed = struct {
        fn go(d: EGLDisplay, sz: gles.GLfloat) !usize {
            const cfg = state.configHandle(0);
            const ctx = eglCreateContext(d, cfg, egl.EGL_NO_CONTEXT, null);
            defer _ = eglDestroyContext(d, ctx);
            const pb = [_]EGLint{ egl.EGL_WIDTH, 64, egl.EGL_HEIGHT, 64, egl.EGL_NONE };
            const surf = eglCreatePbufferSurface(d, cfg, &pb);
            defer _ = eglDestroySurface(d, surf);
            _ = eglMakeCurrent(d, surf, surf, ctx);
            defer _ = eglMakeCurrent(d, null, null, egl.EGL_NO_CONTEXT);

            const vs_src: [*:0]const gles.GLchar =
                \\attribute vec2 position;
                \\uniform float uSize;
                \\void main() { gl_Position = vec4(position, 0.0, 1.0); gl_PointSize = uSize; }
            ;
            const fs_src: [*:0]const gles.GLchar =
                \\precision mediump float;
                \\void main() { gl_FragColor = vec4(1.0, 0.0, 0.0, 1.0); }
            ;
            const vs = glCreateShader(gles.GL_VERTEX_SHADER);
            glShaderSource(vs, 1, &[_]?[*:0]const gles.GLchar{vs_src}, null);
            glCompileShader(vs);
            const fs = glCreateShader(gles.GL_FRAGMENT_SHADER);
            glShaderSource(fs, 1, &[_]?[*:0]const gles.GLchar{fs_src}, null);
            glCompileShader(fs);
            const prog = glCreateProgram();
            glAttachShader(prog, vs);
            glAttachShader(prog, fs);
            glBindAttribLocation(prog, 0, "position");
            glLinkProgram(prog);
            glUseProgram(prog);
            glUniform1f(glGetUniformLocation(prog, "uSize"), sz);

            const pt = [1][2]f32{.{ 0.0, 0.0 }}; // one point at the center
            var vbo: gles.GLuint = 0;
            glGenBuffers(1, @ptrCast(&vbo));
            glBindBuffer(gles.GL_ARRAY_BUFFER, vbo);
            glBufferData(gles.GL_ARRAY_BUFFER, @sizeOf(@TypeOf(pt)), &pt, gles.GL_STATIC_DRAW);
            glVertexAttribPointer(0, 2, gles.GL_FLOAT, gles.GL_FALSE, 8, @ptrFromInt(0));
            glEnableVertexAttribArray(0);
            glClearColor(0, 0, 0, 1);
            glClear(gles.GL_COLOR_BUFFER_BIT);
            glDrawArrays(gles.GL_POINTS, 0, 1);
            try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());

            const s: *state.Surface = @ptrCast(@alignCast(surf.?));
            const dev = s.display.device.?;
            const px = try flushMap(dev, s);
            var red: usize = 0;
            var i: usize = 0;
            while (i < 64 * 64) : (i += 1) {
                if (px[i * 4 + 0] > 200 and px[i * 4 + 1] < 60) red += 1;
            }
            glDeleteBuffers(1, &[_]gles.GLuint{vbo});
            glDeleteProgram(prog);
            glDeleteShader(vs);
            glDeleteShader(fs);
            return red;
        }
    }.go;

    const small = try pointRed(dpy, 2.0); // ~2x2 = ~4 px
    const large = try pointRed(dpy, 8.0); // ~8x8 = ~64 px
    try std.testing.expect(small > 0); // the point drew
    try std.testing.expect(large > small + 20); // gl_PointSize scaled it up
}

test "gl_PointCoord: a point sprite's FS reads the 0..1 s/t coord (textured points / particles)" {
    // A point sprite of size 16 at the screen center; the FS writes gl_PointCoord into R,G. The coord
    // runs 0..1 across the sprite with the origin at the upper-left: s = 0 left / 1 right, t = 0 top /
    // 1 bottom. So the top-right of the sprite has high s (red) + low t (little green); the
    // bottom-left has low s + high t (green). The gradient proves gl_PointCoord is supplied.
    _ = eglBindAPI(egl.EGL_OPENGL_ES_API);
    const dpy = eglGetPlatformDisplay(egl.EGL_PLATFORM_SURFACELESS_MESA, @ptrFromInt(0x7EA5), null);
    try std.testing.expectEqual(egl.EGL_TRUE, eglInitialize(dpy, null, null));
    defer _ = eglTerminate(dpy);
    const cfg = state.configHandle(0);
    const ctx = eglCreateContext(dpy, cfg, egl.EGL_NO_CONTEXT, null);
    const W: EGLint = 64;
    const pb = [_]EGLint{ egl.EGL_WIDTH, W, egl.EGL_HEIGHT, W, egl.EGL_NONE };
    const surf = eglCreatePbufferSurface(dpy, cfg, &pb);
    try std.testing.expectEqual(egl.EGL_TRUE, eglMakeCurrent(dpy, surf, surf, ctx));
    defer {
        _ = eglMakeCurrent(dpy, null, null, egl.EGL_NO_CONTEXT);
        _ = eglDestroySurface(dpy, surf);
        _ = eglDestroyContext(dpy, ctx);
    }

    const vs_src: [*:0]const gles.GLchar =
        \\attribute vec2 position;
        \\void main() { gl_Position = vec4(position, 0.0, 1.0); gl_PointSize = 16.0; }
    ;
    const fs_src: [*:0]const gles.GLchar =
        \\precision mediump float;
        \\void main() { gl_FragColor = vec4(gl_PointCoord.x, gl_PointCoord.y, 0.0, 1.0); }
    ;
    const vs = glCreateShader(gles.GL_VERTEX_SHADER);
    glShaderSource(vs, 1, &[_]?[*:0]const gles.GLchar{vs_src}, null);
    glCompileShader(vs);
    const fs = glCreateShader(gles.GL_FRAGMENT_SHADER);
    glShaderSource(fs, 1, &[_]?[*:0]const gles.GLchar{fs_src}, null);
    glCompileShader(fs);
    const prog = glCreateProgram();
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glBindAttribLocation(prog, 0, "position");
    glLinkProgram(prog);
    glUseProgram(prog);
    defer {
        glDeleteProgram(prog);
        glDeleteShader(vs);
        glDeleteShader(fs);
    }

    const pt = [1][2]f32{.{ 0.0, 0.0 }};
    var vbo: gles.GLuint = 0;
    glGenBuffers(1, @ptrCast(&vbo));
    glBindBuffer(gles.GL_ARRAY_BUFFER, vbo);
    glBufferData(gles.GL_ARRAY_BUFFER, @sizeOf(@TypeOf(pt)), &pt, gles.GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, gles.GL_FLOAT, gles.GL_FALSE, 8, @ptrFromInt(0));
    glEnableVertexAttribArray(0);
    defer glDeleteBuffers(1, &[_]gles.GLuint{vbo});
    glClearColor(0, 0, 0, 1);
    glClear(gles.GL_COLOR_BUFFER_BIT);
    glDrawArrays(gles.GL_POINTS, 0, 1);
    try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());

    const s: *state.Surface = @ptrCast(@alignCast(surf.?));
    const dev = s.display.device.?;
    const px = try flushMap(dev, s);
    // The sprite covers ~[24,40)x[24,40) (size 16 centered at 32). Sample well inside each corner.
    const at = struct {
        fn p(buf: []const u8, x: usize, y: usize) [2]u8 {
            const o = (y * 64 + x) * 4;
            return .{ buf[o + 0], buf[o + 1] }; // R (=s), G (=t)
        }
    }.p;
    const tr = at(px, 37, 27); // top-right: high s, low t
    const bl = at(px, 27, 37); // bottom-left: low s, high t
    try std.testing.expect(tr[0] > 150 and tr[1] < 110); // s high, t low
    try std.testing.expect(bl[1] > 150 and bl[0] < 110); // t high, s low
}

// glmark2's texture/effect2d scenes declare `attribute vec3 position; vec3 normal; vec2
// texcoord;` and resolve each by name (no glBindAttribLocation). The old substring heuristic
// returned -1 for `texcoord` (and aliased normal/color to location 1), so the texture
// coordinates were never bound -> texture2D sampled garbage -> a black scene. This test proves
// the real attribute resolution: `texcoord` resolves to its true location (declaration order
// after position+normal), the UV reaches the FS, and the texture samples correctly.
test "glmark2-shaped attributes (position/normal/texcoord) resolve by NAME and texcoord samples" {
    _ = eglBindAPI(egl.EGL_OPENGL_ES_API);
    const dpy = eglGetPlatformDisplay(egl.EGL_PLATFORM_SURFACELESS_MESA, @ptrFromInt(0x7EC0), null);
    try std.testing.expectEqual(egl.EGL_TRUE, eglInitialize(dpy, null, null));
    defer _ = eglTerminate(dpy);
    const cfg = state.configHandle(0);
    const ctx = eglCreateContext(dpy, cfg, egl.EGL_NO_CONTEXT, null);
    const W: EGLint = 64;
    const pb = [_]EGLint{ egl.EGL_WIDTH, W, egl.EGL_HEIGHT, W, egl.EGL_NONE };
    const surf = eglCreatePbufferSurface(dpy, cfg, &pb);
    try std.testing.expectEqual(egl.EGL_TRUE, eglMakeCurrent(dpy, surf, surf, ctx));

    // The glmark2 texture-scene attribute set (position, normal, texcoord), declaration order.
    const vs_src: [*:0]const gles.GLchar =
        \\attribute vec3 position;
        \\attribute vec3 normal;
        \\attribute vec2 texcoord;
        \\varying vec2 TextureCoord;
        \\void main() {
        \\    TextureCoord = texcoord;
        \\    gl_Position = vec4(position.xy + normal.x * 0.0, 0.0, 1.0);
        \\}
    ;
    const fs_src: [*:0]const gles.GLchar =
        \\precision mediump float;
        \\uniform sampler2D uTex;
        \\varying vec2 TextureCoord;
        \\void main() { gl_FragColor = texture2D(uTex, TextureCoord); }
    ;
    const vs = glCreateShader(gles.GL_VERTEX_SHADER);
    glShaderSource(vs, 1, &[_]?[*:0]const gles.GLchar{vs_src}, null);
    glCompileShader(vs);
    const fs = glCreateShader(gles.GL_FRAGMENT_SHADER);
    glShaderSource(fs, 1, &[_]?[*:0]const gles.GLchar{fs_src}, null);
    glCompileShader(fs);
    const prog = glCreateProgram();
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    // NO glBindAttribLocation: the app resolves every attribute by name (the glmark2 path).
    glLinkProgram(prog);
    var link_ok: gles.GLint = 0;
    glGetProgramiv(prog, gles.GL_LINK_STATUS, &link_ok);
    try std.testing.expectEqual(@as(gles.GLint, gles.GL_TRUE), link_ok);
    glUseProgram(prog);

    // The real attribute resolution: position/normal/texcoord get distinct locations by name,
    // and texcoord is NOT -1 (the old heuristic's bug). A uniform name is still not an attribute.
    const pos_loc = glGetAttribLocation(prog, "position");
    const nrm_loc = glGetAttribLocation(prog, "normal");
    const uv_loc = glGetAttribLocation(prog, "texcoord");
    try std.testing.expect(pos_loc >= 0 and nrm_loc >= 0 and uv_loc >= 0);
    try std.testing.expect(pos_loc != uv_loc and nrm_loc != uv_loc and pos_loc != nrm_loc);
    try std.testing.expectEqual(@as(gles.GLint, -1), glGetAttribLocation(prog, "uTex")); // a sampler is not an attribute

    // A 2x2 checkerboard so a wrong UV would NOT yield the expected per-corner texel.
    const tex_px = [_]u8{
        255, 0, 0,   255, 0,   255, 0,   255,
        0,   0, 255, 255, 255, 255, 255, 255,
    };
    var tex: gles.GLuint = 0;
    glGenTextures(1, @ptrCast(&tex));
    glActiveTexture(gles.GL_TEXTURE0);
    glBindTexture(gles.GL_TEXTURE_2D, tex);
    glTexImage2D(gles.GL_TEXTURE_2D, 0, gles.GL_RGBA, 2, 2, 0, gles.GL_RGBA, gles.GL_UNSIGNED_BYTE, &tex_px);
    glTexParameteri(gles.GL_TEXTURE_2D, gles.GL_TEXTURE_MIN_FILTER, @intCast(gles.GL_NEAREST));
    glTexParameteri(gles.GL_TEXTURE_2D, gles.GL_TEXTURE_MAG_FILTER, @intCast(gles.GL_NEAREST));
    glTexParameteri(gles.GL_TEXTURE_2D, gles.GL_TEXTURE_WRAP_S, @intCast(gles.GL_CLAMP_TO_EDGE));
    glTexParameteri(gles.GL_TEXTURE_2D, gles.GL_TEXTURE_WRAP_T, @intCast(gles.GL_CLAMP_TO_EDGE));
    glUniform1i(glGetUniformLocation(prog, "uTex"), 0);

    // Separate per-attribute VBOs (glmark2's Mesh layout), bound to the resolved locations.
    const positions = [_]f32{ -1, -1, 0, 1, -1, 0, 1, 1, 0, -1, -1, 0, 1, 1, 0, -1, 1, 0 };
    const normals = [_]f32{ 0, 0, 1 } ** 6;
    const uvs = [_]f32{ 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1 };
    var vbos: [3]gles.GLuint = undefined;
    glGenBuffers(3, &vbos);
    glBindBuffer(gles.GL_ARRAY_BUFFER, vbos[0]);
    glBufferData(gles.GL_ARRAY_BUFFER, @sizeOf(@TypeOf(positions)), &positions, gles.GL_STATIC_DRAW);
    glVertexAttribPointer(@intCast(pos_loc), 3, gles.GL_FLOAT, gles.GL_FALSE, 0, @ptrFromInt(0));
    glEnableVertexAttribArray(@intCast(pos_loc));
    glBindBuffer(gles.GL_ARRAY_BUFFER, vbos[1]);
    glBufferData(gles.GL_ARRAY_BUFFER, @sizeOf(@TypeOf(normals)), &normals, gles.GL_STATIC_DRAW);
    glVertexAttribPointer(@intCast(nrm_loc), 3, gles.GL_FLOAT, gles.GL_FALSE, 0, @ptrFromInt(0));
    glEnableVertexAttribArray(@intCast(nrm_loc));
    glBindBuffer(gles.GL_ARRAY_BUFFER, vbos[2]);
    glBufferData(gles.GL_ARRAY_BUFFER, @sizeOf(@TypeOf(uvs)), &uvs, gles.GL_STATIC_DRAW);
    glVertexAttribPointer(@intCast(uv_loc), 2, gles.GL_FLOAT, gles.GL_FALSE, 0, @ptrFromInt(0));
    glEnableVertexAttribArray(@intCast(uv_loc));

    glClearColor(0, 0, 0, 1);
    glClear(gles.GL_COLOR_BUFFER_BIT);
    glDrawArrays(gles.GL_TRIANGLES, 0, 6);
    try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());

    const s: *state.Surface = @ptrCast(@alignCast(surf.?));
    const dev = s.display.device.?;
    const px = try flushMap(dev, s);
    const at = struct {
        fn p(b: []const u8, x: usize, y: usize) [3]u8 {
            const o = (y * 64 + x) * 4;
            return .{ b[o], b[o + 1], b[o + 2] };
        }
    }.p;
    // The texture sampled correctly (NOT black): with the GL bottom-left origin, v=0 (red,green)
    // is at the bottom, v=1 (blue,white) at the top. So the bottom-left pixel is red, the
    // load-bearing proof that texcoord bound and the UV reached the FS (the bug-2 fix).
    const bl = at(px, 16, 48);
    try std.testing.expect(bl[0] > 200 and bl[1] < 60 and bl[2] < 60); // red, not black
    const tl = at(px, 16, 16);
    try std.testing.expect(tl[2] > 200 and tl[0] < 60 and tl[1] < 60); // blue, not black

    glDeleteBuffers(3, &vbos);
    glDeleteTextures(1, &[_]gles.GLuint{tex});
    glDeleteProgram(prog);
    glDeleteShader(vs);
    glDeleteShader(fs);
    _ = eglMakeCurrent(dpy, null, null, egl.EGL_NO_CONTEXT);
    _ = eglDestroySurface(dpy, surf);
    _ = eglDestroyContext(dpy, ctx);
}

fn runNamedUbo(dpy_magic: usize) !void {
    _ = eglBindAPI(egl.EGL_OPENGL_ES_API);
    const dpy = eglGetPlatformDisplay(egl.EGL_PLATFORM_SURFACELESS_MESA, @ptrFromInt(dpy_magic), null);
    if (eglInitialize(dpy, null, null) != egl.EGL_TRUE) return error.SkipZigTest;
    defer _ = eglTerminate(dpy);

    const cfg = state.configHandle(0); // RGBA8
    const ctx = eglCreateContext(dpy, cfg, egl.EGL_NO_CONTEXT, null);
    try std.testing.expect(ctx != egl.EGL_NO_CONTEXT);
    const W: EGLint = 64;
    const H: EGLint = 64;
    const pb_attribs = [_]EGLint{ egl.EGL_WIDTH, W, egl.EGL_HEIGHT, H, egl.EGL_NONE };
    const surf = eglCreatePbufferSurface(dpy, cfg, &pb_attribs);
    try std.testing.expect(surf != egl.EGL_NO_SURFACE);
    try std.testing.expectEqual(egl.EGL_TRUE, eglMakeCurrent(dpy, surf, surf, ctx));

    // A named std140 uniform block `Blk { vec4 uColor; }` in the VS: uColor is read from the
    // block and forwarded as a varying the FS outputs. The block's bytes come from a USER buffer
    // filled with glBufferData + bound with glBindBufferBase (NOT glUniform*), so the rendered
    // color is the load-bearing proof the app's UBO reached the shader's block descriptor.
    const vs_src: [*:0]const gles.GLchar =
        \\attribute vec2 aPos;
        \\layout(std140) uniform Blk { vec4 uColor; };
        \\varying vec4 vColor;
        \\void main() { gl_Position = vec4(aPos, 0.0, 1.0); vColor = uColor; }
    ;
    const fs_src: [*:0]const gles.GLchar =
        \\precision mediump float;
        \\varying vec4 vColor;
        \\void main() { gl_FragColor = vColor; }
    ;
    const vs = glCreateShader(gles.GL_VERTEX_SHADER);
    glShaderSource(vs, 1, &[_]?[*:0]const gles.GLchar{vs_src}, null);
    glCompileShader(vs);
    var vs_ok: gles.GLint = 0;
    glGetShaderiv(vs, gles.GL_COMPILE_STATUS, &vs_ok);
    if (vs_ok != gles.GL_TRUE) {
        var lb: [256]gles.GLchar = undefined;
        glGetShaderInfoLog(vs, lb.len, null, &lb);
        std.debug.print("VS: {s}\n", .{std.mem.sliceTo(&lb, 0)});
    }
    try std.testing.expectEqual(@as(gles.GLint, gles.GL_TRUE), vs_ok);
    const fs = glCreateShader(gles.GL_FRAGMENT_SHADER);
    glShaderSource(fs, 1, &[_]?[*:0]const gles.GLchar{fs_src}, null);
    glCompileShader(fs);

    const prog = glCreateProgram();
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glBindAttribLocation(prog, 0, "aPos");
    glLinkProgram(prog);
    var link_ok: gles.GLint = 0;
    glGetProgramiv(prog, gles.GL_LINK_STATUS, &link_ok);
    try std.testing.expectEqual(@as(gles.GLint, gles.GL_TRUE), link_ok);
    glUseProgram(prog);

    // The block index resolves, and its std140 data size is 16 bytes (one vec4).
    const blk_index = glGetUniformBlockIndex(prog, "Blk");
    try std.testing.expect(blk_index != gles.GL_INVALID_INDEX);
    var data_size = [_]gles.GLint{0};
    glGetActiveUniformBlockiv(prog, blk_index, gles.GL_UNIFORM_BLOCK_DATA_SIZE, &data_size);
    try std.testing.expectEqual(@as(gles.GLint, 16), data_size[0]);
    // Point the block at uniform-buffer binding point 0.
    glUniformBlockBinding(prog, blk_index, 0);

    // The UBO: a known vec4 filled via glBufferData, bound at binding point 0.
    const ubo_color = [4]f32{ 0.25, 0.5, 0.75, 1.0 };
    var ubo: gles.GLuint = 0;
    glGenBuffers(1, @ptrCast(&ubo));
    try std.testing.expect(ubo != 0);
    glBindBuffer(gles.GL_UNIFORM_BUFFER, ubo);
    glBufferData(gles.GL_UNIFORM_BUFFER, @sizeOf(@TypeOf(ubo_color)), &ubo_color, gles.GL_STATIC_DRAW);
    glBindBufferBase(gles.GL_UNIFORM_BUFFER, 0, ubo);
    try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());

    // A full-screen triangle (covers the center pixel).
    const Vtx = extern struct { x: f32, y: f32 };
    const tri = [3]Vtx{ .{ .x = -1, .y = -1 }, .{ .x = 3, .y = -1 }, .{ .x = -1, .y = 3 } };
    var vbo: gles.GLuint = 0;
    glGenBuffers(1, @ptrCast(&vbo));
    glBindBuffer(gles.GL_ARRAY_BUFFER, vbo);
    glBufferData(gles.GL_ARRAY_BUFFER, @sizeOf(@TypeOf(tri)), &tri, gles.GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, gles.GL_FLOAT, gles.GL_FALSE, @sizeOf(Vtx), @ptrFromInt(0));
    glEnableVertexAttribArray(0);

    glClearColor(0, 0, 0, 1);
    glClear(gles.GL_COLOR_BUFFER_BIT);
    glDrawArrays(gles.GL_TRIANGLES, 0, 3);
    try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());

    // Read back the center pixel: it must be the UBO color (0.25,0.5,0.75) -> ~ (64,128,191).
    // NOT black (which is what a zero/default block would give: the differential proof the
    // glBindBufferBase'd buffer, not the empty glUniform storage, fed the block).
    const s: *state.Surface = @ptrCast(@alignCast(surf.?));
    const dev = s.display.device.?;
    const px = try flushMap(dev, s);
    const c = (32 * 64 + 32) * 4;
    try std.testing.expect(px[c + 0] > 50 and px[c + 0] < 78); // ~64
    try std.testing.expect(px[c + 1] > 114 and px[c + 1] < 142); // ~128
    try std.testing.expect(px[c + 2] > 177 and px[c + 2] < 205); // ~191
    // Load-bearing differential: not black (the UBO data really reached the shader).
    const sum: u32 = @as(u32, px[c + 0]) + px[c + 1] + px[c + 2];
    try std.testing.expect(sum > 30);

    glDeleteBuffers(1, &[_]gles.GLuint{ubo});
    glDeleteBuffers(1, &[_]gles.GLuint{vbo});
    glDeleteProgram(prog);
    glDeleteShader(vs);
    glDeleteShader(fs);
    _ = eglMakeCurrent(dpy, null, null, egl.EGL_NO_CONTEXT);
    _ = eglDestroySurface(dpy, surf);
    _ = eglDestroyContext(dpy, ctx);
}

test "EGL UBO oracle: a std140 named uniform block bound via glBindBufferBase colors the draw (software)" {
    try runNamedUbo(0xE6B10);
}

test "EGL UBO oracle on NVIDIA GPU: a std140 named uniform block feeds the shader on the real RTX (skips without a GPU)" {
    // Named UBOs (glBindBufferBase) resolve to a HAL Resource and ride the SAME per-draw bind path as
    // default-block uniforms, so the nvidia UBO snapshot ring feeds them too. Forced onto the GPU.
    const saved = state.pinned_driver;
    state.pinned_driver = "nvidia";
    defer state.pinned_driver = saved;
    try runNamedUbo(0xE6B11);
}

fn runTransformFeedback(dpy_magic: usize) !void {
    _ = eglBindAPI(egl.EGL_OPENGL_ES_API);
    const dpy = eglGetPlatformDisplay(egl.EGL_PLATFORM_SURFACELESS_MESA, @ptrFromInt(dpy_magic), null);
    if (eglInitialize(dpy, null, null) != egl.EGL_TRUE) return error.SkipZigTest;
    defer _ = eglTerminate(dpy);

    const cfg = state.configHandle(0); // RGBA8
    const ctx = eglCreateContext(dpy, cfg, egl.EGL_NO_CONTEXT, null);
    try std.testing.expect(ctx != egl.EGL_NO_CONTEXT);
    const W: EGLint = 16;
    const H: EGLint = 16;
    const pb_attribs = [_]EGLint{ egl.EGL_WIDTH, W, egl.EGL_HEIGHT, H, egl.EGL_NONE };
    const surf = eglCreatePbufferSurface(dpy, cfg, &pb_attribs);
    try std.testing.expect(surf != egl.EGL_NO_SURFACE);
    try std.testing.expectEqual(egl.EGL_TRUE, eglMakeCurrent(dpy, surf, surf, ctx));

    // A VS that computes vDoubled = aValue * 2.0 into an output varying. Transform feedback captures
    // vDoubled per vertex INDEPENDENT of rasterization; with GL_RASTERIZER_DISCARD the fragment stage
    // never runs, so the buffer is the only output (the particle-system / skinning-cache pattern).
    const vs_src: [*:0]const gles.GLchar =
        \\attribute float aValue;
        \\varying float vDoubled;
        \\void main() { vDoubled = aValue * 2.0; gl_Position = vec4(0.0, 0.0, 0.0, 1.0); }
    ;
    const fs_src: [*:0]const gles.GLchar =
        \\precision mediump float;
        \\varying float vDoubled;
        \\void main() { gl_FragColor = vec4(vDoubled, 0.0, 0.0, 1.0); }
    ;
    const vs = glCreateShader(gles.GL_VERTEX_SHADER);
    glShaderSource(vs, 1, &[_]?[*:0]const gles.GLchar{vs_src}, null);
    glCompileShader(vs);
    var vs_ok: gles.GLint = 0;
    glGetShaderiv(vs, gles.GL_COMPILE_STATUS, &vs_ok);
    if (vs_ok != gles.GL_TRUE) {
        var lb: [256]gles.GLchar = undefined;
        glGetShaderInfoLog(vs, lb.len, null, &lb);
        std.debug.print("VS: {s}\n", .{std.mem.sliceTo(&lb, 0)});
    }
    try std.testing.expectEqual(@as(gles.GLint, gles.GL_TRUE), vs_ok);
    const fs = glCreateShader(gles.GL_FRAGMENT_SHADER);
    glShaderSource(fs, 1, &[_]?[*:0]const gles.GLchar{fs_src}, null);
    glCompileShader(fs);

    const prog = glCreateProgram();
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glBindAttribLocation(prog, 0, "aValue");
    // Record the capture varying BEFORE linking, per spec (GL_INTERLEAVED_ATTRIBS = one buffer).
    glTransformFeedbackVaryings(prog, 1, &[_]?[*:0]const gles.GLchar{"vDoubled"}, gles.GL_INTERLEAVED_ATTRIBS);
    try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());
    glLinkProgram(prog);
    var link_ok: gles.GLint = 0;
    glGetProgramiv(prog, gles.GL_LINK_STATUS, &link_ok);
    try std.testing.expectEqual(@as(gles.GLint, gles.GL_TRUE), link_ok);
    glUseProgram(prog);

    // The input attribute stream: four values -> expect four doubled captures.
    const N = 4;
    const in_values = [N]f32{ 1.0, 2.0, 3.0, 4.0 };
    var vbo: gles.GLuint = 0;
    glGenBuffers(1, @ptrCast(&vbo));
    glBindBuffer(gles.GL_ARRAY_BUFFER, vbo);
    glBufferData(gles.GL_ARRAY_BUFFER, @sizeOf(@TypeOf(in_values)), &in_values, gles.GL_STATIC_DRAW);
    glVertexAttribPointer(0, 1, gles.GL_FLOAT, gles.GL_FALSE, @sizeOf(f32), @ptrFromInt(0));
    glEnableVertexAttribArray(0);

    // The capture buffer: sized for N floats, bound at transform-feedback binding point 0.
    var tfbo: gles.GLuint = 0;
    glGenBuffers(1, @ptrCast(&tfbo));
    glBindBuffer(gles.GL_TRANSFORM_FEEDBACK_BUFFER, tfbo);
    const zero = [N]f32{ 0, 0, 0, 0 };
    glBufferData(gles.GL_TRANSFORM_FEEDBACK_BUFFER, @sizeOf(@TypeOf(zero)), &zero, gles.GL_STATIC_DRAW);
    glBindBufferBase(gles.GL_TRANSFORM_FEEDBACK_BUFFER, 0, tfbo);
    try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());

    // Pure capture: discard rasterization, run the VS over N points, capture vDoubled per vertex.
    glEnable(gles.GL_RASTERIZER_DISCARD);
    glBeginTransformFeedback(gles.GL_POINTS);
    glDrawArrays(gles.GL_POINTS, 0, N);
    glEndTransformFeedback();
    glDisable(gles.GL_RASTERIZER_DISCARD);
    try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());

    // Read back the captured floats: each must be exactly its input doubled (1,2,3,4 -> 2,4,6,8).
    var captured = [N]f32{ 0, 0, 0, 0 };
    glGetBufferSubData(gles.GL_TRANSFORM_FEEDBACK_BUFFER, 0, @sizeOf(@TypeOf(captured)), &captured);
    try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());
    for (0..N) |i| {
        try std.testing.expectApproxEqAbs(in_values[i] * 2.0, captured[i], 1e-5);
    }
    // Differential: the buffer is no longer all zero (the VS output really reached the capture).
    try std.testing.expect(captured[0] != 0 and captured[3] != 0);

    glDeleteBuffers(1, &[_]gles.GLuint{tfbo});
    glDeleteBuffers(1, &[_]gles.GLuint{vbo});
    glDeleteProgram(prog);
    glDeleteShader(vs);
    glDeleteShader(fs);
    _ = eglMakeCurrent(dpy, null, null, egl.EGL_NO_CONTEXT);
    _ = eglDestroySurface(dpy, surf);
    _ = eglDestroyContext(dpy, ctx);
}

test "EGL transform-feedback oracle: a VS captures a doubled per-vertex varying into a buffer (software)" {
    try runTransformFeedback(0x7F00B);
}

test "transform feedback on NVIDIA GPU: GPU stream-out captures the doubled varying on the real RTX (skips without a GPU)" {
    // Forces the real nvidia device: the VS runs with GPU STREAM-OUT (global SET_STREAM_OUTPUT +
    // SET_STREAM_OUT_BUFFER/CONTROL/LAYOUT + SPH STREAM_OUT_MASK) and the rasterizer disabled,
    // streaming vDoubled per vertex. {1,2,3,4} -> {2,4,6,8} proves real hardware stream-out.
    const saved = state.pinned_driver;
    state.pinned_driver = "nvidia";
    defer state.pinned_driver = saved;
    try runTransformFeedback(0x7F00C);
}

test "EGL transform-feedback oracle: GL_SEPARATE_ATTRIBS captures each varying into its OWN buffer" {
    _ = eglBindAPI(egl.EGL_OPENGL_ES_API);
    const dpy = eglGetPlatformDisplay(egl.EGL_PLATFORM_SURFACELESS_MESA, @ptrFromInt(0x5EBA2), null);
    try std.testing.expect(dpy != null);
    try std.testing.expectEqual(egl.EGL_TRUE, eglInitialize(dpy, null, null));
    defer _ = eglTerminate(dpy);

    const cfg = state.configHandle(0); // RGBA8
    const ctx = eglCreateContext(dpy, cfg, egl.EGL_NO_CONTEXT, null);
    try std.testing.expect(ctx != egl.EGL_NO_CONTEXT);
    const pb_attribs = [_]EGLint{ egl.EGL_WIDTH, 16, egl.EGL_HEIGHT, 16, egl.EGL_NONE };
    const surf = eglCreatePbufferSurface(dpy, cfg, &pb_attribs);
    try std.testing.expect(surf != egl.EGL_NO_SURFACE);
    try std.testing.expectEqual(egl.EGL_TRUE, eglMakeCurrent(dpy, surf, surf, ctx));
    defer _ = eglDestroyContext(dpy, ctx);
    defer _ = eglDestroySurface(dpy, surf);
    defer _ = eglMakeCurrent(dpy, null, null, egl.EGL_NO_CONTEXT);

    // A VS with TWO output varyings: vA = aValue*2, vB = aValue+10. With GL_SEPARATE_ATTRIBS each
    // varying is captured into its OWN buffer (vA -> binding 0, vB -> binding 1), so the two must
    // land in different buffers rather than interleaved in one.
    const vs_src: [*:0]const gles.GLchar =
        \\attribute float aValue;
        \\varying float vA;
        \\varying float vB;
        \\void main() { vA = aValue * 2.0; vB = aValue + 10.0; gl_Position = vec4(0.0, 0.0, 0.0, 1.0); }
    ;
    const fs_src: [*:0]const gles.GLchar =
        \\precision mediump float;
        \\varying float vA;
        \\varying float vB;
        \\void main() { gl_FragColor = vec4(vA, vB, 0.0, 1.0); }
    ;
    const vs = glCreateShader(gles.GL_VERTEX_SHADER);
    glShaderSource(vs, 1, &[_]?[*:0]const gles.GLchar{vs_src}, null);
    glCompileShader(vs);
    var vs_ok: gles.GLint = 0;
    glGetShaderiv(vs, gles.GL_COMPILE_STATUS, &vs_ok);
    try std.testing.expectEqual(@as(gles.GLint, gles.GL_TRUE), vs_ok);
    const fs = glCreateShader(gles.GL_FRAGMENT_SHADER);
    glShaderSource(fs, 1, &[_]?[*:0]const gles.GLchar{fs_src}, null);
    glCompileShader(fs);

    const prog = glCreateProgram();
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glBindAttribLocation(prog, 0, "aValue");
    // Capturing MORE than GL_MAX_TRANSFORM_FEEDBACK_SEPARATE_ATTRIBS varyings in SEPARATE mode is
    // GL_INVALID_VALUE (advertised = 4 here).
    var max_sep: gles.GLint = 0;
    glGetIntegerv(gles.GL_MAX_TRANSFORM_FEEDBACK_SEPARATE_ATTRIBS, @ptrCast(&max_sep));
    try std.testing.expectEqual(@as(gles.GLint, gles.MAX_TRANSFORM_FEEDBACK_BUFFERS), max_sep);
    const too_many = [_]?[*:0]const gles.GLchar{ "a", "b", "c", "d", "e" };
    glTransformFeedbackVaryings(prog, 5, &too_many, gles.GL_SEPARATE_ATTRIBS);
    try std.testing.expectEqual(@as(gles.GLenum, gles.GL_INVALID_VALUE), glGetError());

    // Record the two capture varyings in separate mode before linking, per spec.
    glTransformFeedbackVaryings(prog, 2, &[_]?[*:0]const gles.GLchar{ "vA", "vB" }, gles.GL_SEPARATE_ATTRIBS);
    try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());
    glLinkProgram(prog);
    var link_ok: gles.GLint = 0;
    glGetProgramiv(prog, gles.GL_LINK_STATUS, &link_ok);
    try std.testing.expectEqual(@as(gles.GLint, gles.GL_TRUE), link_ok);
    glUseProgram(prog);

    const N = 4;
    const in_values = [N]f32{ 1.0, 2.0, 3.0, 4.0 };
    var vbo: gles.GLuint = 0;
    glGenBuffers(1, @ptrCast(&vbo));
    glBindBuffer(gles.GL_ARRAY_BUFFER, vbo);
    glBufferData(gles.GL_ARRAY_BUFFER, @sizeOf(@TypeOf(in_values)), &in_values, gles.GL_STATIC_DRAW);
    glVertexAttribPointer(0, 1, gles.GL_FLOAT, gles.GL_FALSE, @sizeOf(f32), @ptrFromInt(0));
    glEnableVertexAttribArray(0);

    // Two capture buffers: buffer0 at binding 0 (vA), buffer1 at binding 1 (vB).
    const zero = [N]f32{ 0, 0, 0, 0 };
    var tf0: gles.GLuint = 0;
    glGenBuffers(1, @ptrCast(&tf0));
    glBindBuffer(gles.GL_TRANSFORM_FEEDBACK_BUFFER, tf0);
    glBufferData(gles.GL_TRANSFORM_FEEDBACK_BUFFER, @sizeOf(@TypeOf(zero)), &zero, gles.GL_STATIC_DRAW);
    glBindBufferBase(gles.GL_TRANSFORM_FEEDBACK_BUFFER, 0, tf0);
    var tf1: gles.GLuint = 0;
    glGenBuffers(1, @ptrCast(&tf1));
    glBindBuffer(gles.GL_TRANSFORM_FEEDBACK_BUFFER, tf1);
    glBufferData(gles.GL_TRANSFORM_FEEDBACK_BUFFER, @sizeOf(@TypeOf(zero)), &zero, gles.GL_STATIC_DRAW);
    glBindBufferBase(gles.GL_TRANSFORM_FEEDBACK_BUFFER, 1, tf1);
    try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());

    glEnable(gles.GL_RASTERIZER_DISCARD);
    glBeginTransformFeedback(gles.GL_POINTS);
    glDrawArrays(gles.GL_POINTS, 0, N);
    glEndTransformFeedback();
    glDisable(gles.GL_RASTERIZER_DISCARD);
    try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());

    // buffer0 = vA = {2,4,6,8}; buffer1 = vB = {11,12,13,14}. Each varying in its own buffer.
    var cap0 = [N]f32{ 0, 0, 0, 0 };
    glBindBuffer(gles.GL_TRANSFORM_FEEDBACK_BUFFER, tf0);
    glGetBufferSubData(gles.GL_TRANSFORM_FEEDBACK_BUFFER, 0, @sizeOf(@TypeOf(cap0)), &cap0);
    var cap1 = [N]f32{ 0, 0, 0, 0 };
    glBindBuffer(gles.GL_TRANSFORM_FEEDBACK_BUFFER, tf1);
    glGetBufferSubData(gles.GL_TRANSFORM_FEEDBACK_BUFFER, 0, @sizeOf(@TypeOf(cap1)), &cap1);
    try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());
    for (0..N) |i| {
        try std.testing.expectApproxEqAbs(in_values[i] * 2.0, cap0[i], 1e-5); // vA
        try std.testing.expectApproxEqAbs(in_values[i] + 10.0, cap1[i], 1e-5); // vB
    }
    // Differential: vB really landed in buffer1 (NOT interleaved into buffer0 after vA).
    try std.testing.expect(cap0[3] != cap1[3]);

    glDeleteBuffers(1, &[_]gles.GLuint{tf0});
    glDeleteBuffers(1, &[_]gles.GLuint{tf1});
    glDeleteBuffers(1, &[_]gles.GLuint{vbo});
    glDeleteProgram(prog);
    glDeleteShader(vs);
    glDeleteShader(fs);
}

test "EGL transform-feedback oracle: glPause/glResumeTransformFeedback skip captures while paused" {
    _ = eglBindAPI(egl.EGL_OPENGL_ES_API);
    const dpy = eglGetPlatformDisplay(egl.EGL_PLATFORM_SURFACELESS_MESA, @ptrFromInt(0x9A05E), null);
    try std.testing.expect(dpy != null);
    try std.testing.expectEqual(egl.EGL_TRUE, eglInitialize(dpy, null, null));
    defer _ = eglTerminate(dpy);

    const cfg = state.configHandle(0);
    const ctx = eglCreateContext(dpy, cfg, egl.EGL_NO_CONTEXT, null);
    try std.testing.expect(ctx != egl.EGL_NO_CONTEXT);
    const pb_attribs = [_]EGLint{ egl.EGL_WIDTH, 16, egl.EGL_HEIGHT, 16, egl.EGL_NONE };
    const surf = eglCreatePbufferSurface(dpy, cfg, &pb_attribs);
    try std.testing.expect(surf != egl.EGL_NO_SURFACE);
    try std.testing.expectEqual(egl.EGL_TRUE, eglMakeCurrent(dpy, surf, surf, ctx));
    defer _ = eglDestroyContext(dpy, ctx);
    defer _ = eglDestroySurface(dpy, surf);
    defer _ = eglMakeCurrent(dpy, null, null, egl.EGL_NO_CONTEXT);

    // glPause before any span is active is GL_INVALID_OPERATION.
    glPauseTransformFeedback();
    try std.testing.expectEqual(@as(gles.GLenum, gles.GL_INVALID_OPERATION), glGetError());
    // glResume before any span is active is GL_INVALID_OPERATION too.
    glResumeTransformFeedback();
    try std.testing.expectEqual(@as(gles.GLenum, gles.GL_INVALID_OPERATION), glGetError());

    const vs_src: [*:0]const gles.GLchar =
        \\attribute float aValue;
        \\varying float vDoubled;
        \\void main() { vDoubled = aValue * 2.0; gl_Position = vec4(0.0, 0.0, 0.0, 1.0); }
    ;
    const fs_src: [*:0]const gles.GLchar =
        \\precision mediump float;
        \\varying float vDoubled;
        \\void main() { gl_FragColor = vec4(vDoubled, 0.0, 0.0, 1.0); }
    ;
    const vs = glCreateShader(gles.GL_VERTEX_SHADER);
    glShaderSource(vs, 1, &[_]?[*:0]const gles.GLchar{vs_src}, null);
    glCompileShader(vs);
    const fs = glCreateShader(gles.GL_FRAGMENT_SHADER);
    glShaderSource(fs, 1, &[_]?[*:0]const gles.GLchar{fs_src}, null);
    glCompileShader(fs);
    const prog = glCreateProgram();
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glBindAttribLocation(prog, 0, "aValue");
    glTransformFeedbackVaryings(prog, 1, &[_]?[*:0]const gles.GLchar{"vDoubled"}, gles.GL_INTERLEAVED_ATTRIBS);
    glLinkProgram(prog);
    var link_ok: gles.GLint = 0;
    glGetProgramiv(prog, gles.GL_LINK_STATUS, &link_ok);
    try std.testing.expectEqual(@as(gles.GLint, gles.GL_TRUE), link_ok);
    glUseProgram(prog);

    // Two input points, redrawn each pass: {5,6} -> captured {10,12} per non-paused draw.
    const in_values = [2]f32{ 5.0, 6.0 };
    var vbo: gles.GLuint = 0;
    glGenBuffers(1, @ptrCast(&vbo));
    glBindBuffer(gles.GL_ARRAY_BUFFER, vbo);
    glBufferData(gles.GL_ARRAY_BUFFER, @sizeOf(@TypeOf(in_values)), &in_values, gles.GL_STATIC_DRAW);
    glVertexAttribPointer(0, 1, gles.GL_FLOAT, gles.GL_FALSE, @sizeOf(f32), @ptrFromInt(0));
    glEnableVertexAttribArray(0);

    // Capture buffer sized for 6 floats, pre-filled with a 99 sentinel: only the 4 NON-paused
    // captures should overwrite. the last two slots must stay 99.
    const sentinel = [6]f32{ 99, 99, 99, 99, 99, 99 };
    var tfbo: gles.GLuint = 0;
    glGenBuffers(1, @ptrCast(&tfbo));
    glBindBuffer(gles.GL_TRANSFORM_FEEDBACK_BUFFER, tfbo);
    glBufferData(gles.GL_TRANSFORM_FEEDBACK_BUFFER, @sizeOf(@TypeOf(sentinel)), &sentinel, gles.GL_STATIC_DRAW);
    glBindBufferBase(gles.GL_TRANSFORM_FEEDBACK_BUFFER, 0, tfbo);

    glEnable(gles.GL_RASTERIZER_DISCARD);
    glBeginTransformFeedback(gles.GL_POINTS);
    glDrawArrays(gles.GL_POINTS, 0, 2); // captured -> {10,12}
    // glResume while active-but-not-paused is GL_INVALID_OPERATION.
    glResumeTransformFeedback();
    try std.testing.expectEqual(@as(gles.GLenum, gles.GL_INVALID_OPERATION), glGetError());
    glPauseTransformFeedback();
    try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());
    // glPause while already paused is GL_INVALID_OPERATION.
    glPauseTransformFeedback();
    try std.testing.expectEqual(@as(gles.GLenum, gles.GL_INVALID_OPERATION), glGetError());
    glDrawArrays(gles.GL_POINTS, 0, 2); // NOT captured (paused)
    glResumeTransformFeedback();
    try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());
    glDrawArrays(gles.GL_POINTS, 0, 2); // captured -> {10,12}, appended after the first pair
    glEndTransformFeedback();
    glDisable(gles.GL_RASTERIZER_DISCARD);
    try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());

    // Only 4 values captured ({10,12,10,12}). The paused draw wrote nothing, so slots 4,5 stay 99.
    var captured = [6]f32{ 0, 0, 0, 0, 0, 0 };
    glGetBufferSubData(gles.GL_TRANSFORM_FEEDBACK_BUFFER, 0, @sizeOf(@TypeOf(captured)), &captured);
    try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());
    try std.testing.expectApproxEqAbs(@as(f32, 10.0), captured[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 12.0), captured[1], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 10.0), captured[2], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 12.0), captured[3], 1e-5);
    // The pause held the cursor: the sixth/fifth slots were never written (still the 99 sentinel).
    try std.testing.expectApproxEqAbs(@as(f32, 99.0), captured[4], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 99.0), captured[5], 1e-5);

    glDeleteBuffers(1, &[_]gles.GLuint{tfbo});
    glDeleteBuffers(1, &[_]gles.GLuint{vbo});
    glDeleteProgram(prog);
    glDeleteShader(vs);
    glDeleteShader(fs);
}

test "EGL UBO oracle: glBindBufferRange reads a block from a non-zero buffer OFFSET (sub-range)" {
    // glBindBufferRange binds a sub-range of a buffer as the UBO (dynamic UBO streaming / packing
    // several blocks in one buffer). A 32-byte buffer holds a decoy color (red) in bytes [0,16) and
    // the real color in [16,32); binding the range at offset 16 must make the shader read the second
    // vec4, not the first. Differential: with the offset ignored (the old deferral) it would read the
    // red decoy at byte 0.
    _ = eglBindAPI(egl.EGL_OPENGL_ES_API);
    const dpy = eglGetPlatformDisplay(egl.EGL_PLATFORM_SURFACELESS_MESA, @ptrFromInt(0xE6B11), null);
    try std.testing.expectEqual(egl.EGL_TRUE, eglInitialize(dpy, null, null));
    defer _ = eglTerminate(dpy);
    const cfg = state.configHandle(0);
    const ctx = eglCreateContext(dpy, cfg, egl.EGL_NO_CONTEXT, null);
    const pb_attribs = [_]EGLint{ egl.EGL_WIDTH, 64, egl.EGL_HEIGHT, 64, egl.EGL_NONE };
    const surf = eglCreatePbufferSurface(dpy, cfg, &pb_attribs);
    try std.testing.expectEqual(egl.EGL_TRUE, eglMakeCurrent(dpy, surf, surf, ctx));
    defer _ = eglDestroyContext(dpy, ctx);
    defer _ = eglDestroySurface(dpy, surf);
    defer _ = eglMakeCurrent(dpy, null, null, egl.EGL_NO_CONTEXT);

    const vs_src: [*:0]const gles.GLchar =
        \\attribute vec2 aPos;
        \\layout(std140) uniform Blk { vec4 uColor; };
        \\varying vec4 vColor;
        \\void main() { gl_Position = vec4(aPos, 0.0, 1.0); vColor = uColor; }
    ;
    const fs_src: [*:0]const gles.GLchar =
        \\precision mediump float;
        \\varying vec4 vColor;
        \\void main() { gl_FragColor = vColor; }
    ;
    const vs = glCreateShader(gles.GL_VERTEX_SHADER);
    glShaderSource(vs, 1, &[_]?[*:0]const gles.GLchar{vs_src}, null);
    glCompileShader(vs);
    const fs = glCreateShader(gles.GL_FRAGMENT_SHADER);
    glShaderSource(fs, 1, &[_]?[*:0]const gles.GLchar{fs_src}, null);
    glCompileShader(fs);
    const prog = glCreateProgram();
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glBindAttribLocation(prog, 0, "aPos");
    glLinkProgram(prog);
    glUseProgram(prog);
    defer {
        glDeleteProgram(prog);
        glDeleteShader(vs);
        glDeleteShader(fs);
    }
    glUniformBlockBinding(prog, glGetUniformBlockIndex(prog, "Blk"), 0);

    // 32-byte buffer: [0,16) = red decoy, [16,32) = the real (0.25,0.5,0.75,1).
    const data = [8]f32{ 1.0, 0.0, 0.0, 1.0, 0.25, 0.5, 0.75, 1.0 };
    var ubo: gles.GLuint = 0;
    glGenBuffers(1, @ptrCast(&ubo));
    glBindBuffer(gles.GL_UNIFORM_BUFFER, ubo);
    glBufferData(gles.GL_UNIFORM_BUFFER, @sizeOf(@TypeOf(data)), &data, gles.GL_STATIC_DRAW);
    glBindBufferRange(gles.GL_UNIFORM_BUFFER, 0, ubo, 16, 16); // the SECOND vec4
    try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());
    defer glDeleteBuffers(1, &[_]gles.GLuint{ubo});

    const Vtx = extern struct { x: f32, y: f32 };
    const tri = [3]Vtx{ .{ .x = -1, .y = -1 }, .{ .x = 3, .y = -1 }, .{ .x = -1, .y = 3 } };
    var vbo: gles.GLuint = 0;
    glGenBuffers(1, @ptrCast(&vbo));
    glBindBuffer(gles.GL_ARRAY_BUFFER, vbo);
    glBufferData(gles.GL_ARRAY_BUFFER, @sizeOf(@TypeOf(tri)), &tri, gles.GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, gles.GL_FLOAT, gles.GL_FALSE, @sizeOf(Vtx), @ptrFromInt(0));
    glEnableVertexAttribArray(0);
    defer glDeleteBuffers(1, &[_]gles.GLuint{vbo});

    glClearColor(0, 0, 0, 1);
    glClear(gles.GL_COLOR_BUFFER_BIT);
    glDrawArrays(gles.GL_TRIANGLES, 0, 3);
    try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());

    const s: *state.Surface = @ptrCast(@alignCast(surf.?));
    const dev = s.display.device.?;
    const px = try flushMap(dev, s);
    const c = (32 * 64 + 32) * 4;
    // The OFFSET color (0.25,0.5,0.75) -> ~(64,128,191), NOT the red decoy at byte 0 (255,0,0).
    try std.testing.expect(px[c + 0] > 50 and px[c + 0] < 78); // ~64, not 255
    try std.testing.expect(px[c + 1] > 114 and px[c + 1] < 142); // ~128, not 0
    try std.testing.expect(px[c + 2] > 177 and px[c + 2] < 205); // ~191
}

test "EGL UBO introspection oracle: glGetUniformIndices + glGetActiveUniformsiv discover a std140 block layout" {
    _ = eglBindAPI(egl.EGL_OPENGL_ES_API);
    const dpy = eglGetPlatformDisplay(egl.EGL_PLATFORM_SURFACELESS_MESA, @ptrFromInt(0x0B10C0), null);
    try std.testing.expect(dpy != null);
    try std.testing.expectEqual(egl.EGL_TRUE, eglInitialize(dpy, null, null));
    defer _ = eglTerminate(dpy);

    const cfg = state.configHandle(0); // RGBA8
    const ctx = eglCreateContext(dpy, cfg, egl.EGL_NO_CONTEXT, null);
    try std.testing.expect(ctx != egl.EGL_NO_CONTEXT);
    const pb_attribs = [_]EGLint{ egl.EGL_WIDTH, 16, egl.EGL_HEIGHT, 16, egl.EGL_NONE };
    const surf = eglCreatePbufferSurface(dpy, cfg, &pb_attribs);
    try std.testing.expect(surf != egl.EGL_NO_SURFACE);
    try std.testing.expectEqual(egl.EGL_TRUE, eglMakeCurrent(dpy, surf, surf, ctx));

    // The same padded std140 block as the layout oracle: `float uA; vec3 uB; vec2 uC;`. Per
    // std140 the members land at bytes {0, 16, 32}. Introspection must discover exactly that
    // layout (offsets + GL types + member count) purely from the query API.
    const vs_src: [*:0]const gles.GLchar =
        \\attribute vec2 aPos;
        \\layout(std140) uniform Blk { float uA; vec3 uB; vec2 uC; };
        \\varying vec4 vColor;
        \\void main() { gl_Position = vec4(aPos, 0.0, 1.0); vColor = vec4(uA, uB.z, uC.y, 1.0); }
    ;
    const fs_src: [*:0]const gles.GLchar =
        \\precision mediump float;
        \\varying vec4 vColor;
        \\void main() { gl_FragColor = vColor; }
    ;
    const vs = glCreateShader(gles.GL_VERTEX_SHADER);
    glShaderSource(vs, 1, &[_]?[*:0]const gles.GLchar{vs_src}, null);
    glCompileShader(vs);
    var vs_ok: gles.GLint = 0;
    glGetShaderiv(vs, gles.GL_COMPILE_STATUS, &vs_ok);
    try std.testing.expectEqual(@as(gles.GLint, gles.GL_TRUE), vs_ok);
    const fs = glCreateShader(gles.GL_FRAGMENT_SHADER);
    glShaderSource(fs, 1, &[_]?[*:0]const gles.GLchar{fs_src}, null);
    glCompileShader(fs);

    const prog = glCreateProgram();
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glBindAttribLocation(prog, 0, "aPos");
    glLinkProgram(prog);
    var link_ok: gles.GLint = 0;
    glGetProgramiv(prog, gles.GL_LINK_STATUS, &link_ok);
    try std.testing.expectEqual(@as(gles.GLint, gles.GL_TRUE), link_ok);
    glUseProgram(prog);

    // glGetUniformIndices: each member name resolves to a valid (non-INVALID) uniform index.
    const names = [_]?[*:0]const gles.GLchar{ "uA", "uB", "uC" };
    var idx = [_]gles.GLuint{ gles.GL_INVALID_INDEX, gles.GL_INVALID_INDEX, gles.GL_INVALID_INDEX };
    glGetUniformIndices(prog, 3, &names, &idx);
    for (idx) |ix| try std.testing.expect(ix != gles.GL_INVALID_INDEX);
    // A name not in any block returns GL_INVALID_INDEX.
    const bogus = [_]?[*:0]const gles.GLchar{"uNope"};
    var bogus_idx = [_]gles.GLuint{0};
    glGetUniformIndices(prog, 1, &bogus, &bogus_idx);
    try std.testing.expectEqual(gles.GL_INVALID_INDEX, bogus_idx[0]);

    // glGetActiveUniformsiv GL_UNIFORM_OFFSET: the std140 byte offsets {0, 16, 32}.
    var offsets = [_]gles.GLint{ -1, -1, -1 };
    glGetActiveUniformsiv(prog, 3, &idx, gles.GL_UNIFORM_OFFSET, &offsets);
    try std.testing.expectEqual(@as(gles.GLint, 0), offsets[0]); // uA
    try std.testing.expectEqual(@as(gles.GLint, 16), offsets[1]); // uB
    try std.testing.expectEqual(@as(gles.GLint, 32), offsets[2]); // uC

    // glGetActiveUniformsiv GL_UNIFORM_TYPE: float, vec3, vec2.
    var types = [_]gles.GLint{ 0, 0, 0 };
    glGetActiveUniformsiv(prog, 3, &idx, gles.GL_UNIFORM_TYPE, &types);
    try std.testing.expectEqual(@as(gles.GLint, @intCast(gles.GL_FLOAT)), types[0]);
    try std.testing.expectEqual(@as(gles.GLint, @intCast(gles.GL_FLOAT_VEC3)), types[1]);
    try std.testing.expectEqual(@as(gles.GLint, @intCast(gles.GL_FLOAT_VEC2)), types[2]);

    // GL_UNIFORM_SIZE is 1 for each (no arrays); GL_UNIFORM_BLOCK_INDEX is the same block for all.
    var sizes = [_]gles.GLint{ 0, 0, 0 };
    glGetActiveUniformsiv(prog, 3, &idx, gles.GL_UNIFORM_SIZE, &sizes);
    for (sizes) |sz| try std.testing.expectEqual(@as(gles.GLint, 1), sz);
    const blk_index = glGetUniformBlockIndex(prog, "Blk");
    try std.testing.expect(blk_index != gles.GL_INVALID_INDEX);
    var blk_of = [_]gles.GLint{ -1, -1, -1 };
    glGetActiveUniformsiv(prog, 3, &idx, gles.GL_UNIFORM_BLOCK_INDEX, &blk_of);
    for (blk_of) |bi| try std.testing.expectEqual(@as(gles.GLint, @intCast(blk_index)), bi);

    // glGetActiveUniformBlockiv: the block has 3 active members, and their indices round-trip.
    var active = [_]gles.GLint{0};
    glGetActiveUniformBlockiv(prog, blk_index, gles.GL_UNIFORM_BLOCK_ACTIVE_UNIFORMS, &active);
    try std.testing.expectEqual(@as(gles.GLint, 3), active[0]);
    var member_idx = [_]gles.GLint{ -1, -1, -1 };
    glGetActiveUniformBlockiv(prog, blk_index, gles.GL_UNIFORM_BLOCK_ACTIVE_UNIFORM_INDICES, &member_idx);
    // The reported member indices are exactly the uA/uB/uC uniform indices (some order).
    for (idx) |want| {
        var seen = false;
        for (member_idx) |got| {
            if (got == @as(gles.GLint, @intCast(want))) seen = true;
        }
        try std.testing.expect(seen);
    }
    // GL_UNIFORM_BLOCK_DATA_SIZE still works (regression): 48 bytes for this padded block.
    var data_size = [_]gles.GLint{0};
    glGetActiveUniformBlockiv(prog, blk_index, gles.GL_UNIFORM_BLOCK_DATA_SIZE, &data_size);
    try std.testing.expectEqual(@as(gles.GLint, 48), data_size[0]);
    try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());

    glDeleteProgram(prog);
    glDeleteShader(vs);
    glDeleteShader(fs);
    _ = eglMakeCurrent(dpy, null, null, egl.EGL_NO_CONTEXT);
    _ = eglDestroySurface(dpy, surf);
    _ = eglDestroyContext(dpy, ctx);
}

test "EGL UBO oracle: a std140 block with padding (float, vec3, vec2) reads each member from its std140 offset" {
    _ = eglBindAPI(egl.EGL_OPENGL_ES_API);
    const dpy = eglGetPlatformDisplay(egl.EGL_PLATFORM_SURFACELESS_MESA, @ptrFromInt(0x57D140), null);
    try std.testing.expect(dpy != null);
    try std.testing.expectEqual(egl.EGL_TRUE, eglInitialize(dpy, null, null));
    defer _ = eglTerminate(dpy);

    const cfg = state.configHandle(0); // RGBA8
    const ctx = eglCreateContext(dpy, cfg, egl.EGL_NO_CONTEXT, null);
    try std.testing.expect(ctx != egl.EGL_NO_CONTEXT);
    const W: EGLint = 64;
    const H: EGLint = 64;
    const pb_attribs = [_]EGLint{ egl.EGL_WIDTH, W, egl.EGL_HEIGHT, H, egl.EGL_NONE };
    const surf = eglCreatePbufferSurface(dpy, cfg, &pb_attribs);
    try std.testing.expect(surf != egl.EGL_NO_SURFACE);
    try std.testing.expectEqual(egl.EGL_TRUE, eglMakeCurrent(dpy, surf, surf, ctx));

    // The load-bearing block: `float uA; vec3 uB; vec2 uC;`. Per std140, uA is at byte 0, uB at
    // byte 16 (vec3 rounds its base alignment up to 16), uC at byte 32 (after uB's 12 bytes ends
    // at 28, rounded up to vec2's 8-byte alignment). The block's std140 size is 48. The shader
    // reads the members tight-packed (uA @ 0, uB @ 4, uC @ 16). If the host did not repack, uB
    // would read the std140 padding after uA (zeros) and uC would read uB.xy: a wrong color.
    // Color forwards uA (red), uB.z (green) and uC.y (blue), each drawn from deep in a padded
    // region, so the rendered pixel proves each member landed at its std140 offset.
    const vs_src: [*:0]const gles.GLchar =
        \\attribute vec2 aPos;
        \\layout(std140) uniform Blk { float uA; vec3 uB; vec2 uC; };
        \\varying vec4 vColor;
        \\void main() { gl_Position = vec4(aPos, 0.0, 1.0); vColor = vec4(uA, uB.z, uC.y, 1.0); }
    ;
    const fs_src: [*:0]const gles.GLchar =
        \\precision mediump float;
        \\varying vec4 vColor;
        \\void main() { gl_FragColor = vColor; }
    ;
    const vs = glCreateShader(gles.GL_VERTEX_SHADER);
    glShaderSource(vs, 1, &[_]?[*:0]const gles.GLchar{vs_src}, null);
    glCompileShader(vs);
    var vs_ok: gles.GLint = 0;
    glGetShaderiv(vs, gles.GL_COMPILE_STATUS, &vs_ok);
    if (vs_ok != gles.GL_TRUE) {
        var lb: [256]gles.GLchar = undefined;
        glGetShaderInfoLog(vs, lb.len, null, &lb);
        std.debug.print("VS: {s}\n", .{std.mem.sliceTo(&lb, 0)});
    }
    try std.testing.expectEqual(@as(gles.GLint, gles.GL_TRUE), vs_ok);
    const fs = glCreateShader(gles.GL_FRAGMENT_SHADER);
    glShaderSource(fs, 1, &[_]?[*:0]const gles.GLchar{fs_src}, null);
    glCompileShader(fs);

    const prog = glCreateProgram();
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glBindAttribLocation(prog, 0, "aPos");
    glLinkProgram(prog);
    var link_ok: gles.GLint = 0;
    glGetProgramiv(prog, gles.GL_LINK_STATUS, &link_ok);
    try std.testing.expectEqual(@as(gles.GLint, gles.GL_TRUE), link_ok);
    glUseProgram(prog);

    // The std140 block data size is 48 bytes (float@0 + vec3@16 [ends 28] + vec2@32 [ends 40],
    // rounded up to 16). With the old tight packing it would have been 24 (a hard regression check).
    const blk_index = glGetUniformBlockIndex(prog, "Blk");
    try std.testing.expect(blk_index != gles.GL_INVALID_INDEX);
    var data_size = [_]gles.GLint{0};
    glGetActiveUniformBlockiv(prog, blk_index, gles.GL_UNIFORM_BLOCK_DATA_SIZE, &data_size);
    try std.testing.expectEqual(@as(gles.GLint, 48), data_size[0]);
    glUniformBlockBinding(prog, blk_index, 0);

    // Fill a user buffer per the std140 layout (48 bytes = 12 floats): uA@float0, padding,
    // uB@floats4..7, padding, uC@floats8..10, padding.
    var blk_bytes = [_]f32{0} ** 12;
    blk_bytes[0] = 0.2; // uA
    blk_bytes[4] = 0.4; // uB.x
    blk_bytes[5] = 0.6; // uB.y
    blk_bytes[6] = 0.8; // uB.z
    blk_bytes[8] = 0.5; // uC.x
    blk_bytes[9] = 0.9; // uC.y
    var ubo: gles.GLuint = 0;
    glGenBuffers(1, @ptrCast(&ubo));
    try std.testing.expect(ubo != 0);
    glBindBuffer(gles.GL_UNIFORM_BUFFER, ubo);
    glBufferData(gles.GL_UNIFORM_BUFFER, @sizeOf(@TypeOf(blk_bytes)), &blk_bytes, gles.GL_STATIC_DRAW);
    glBindBufferBase(gles.GL_UNIFORM_BUFFER, 0, ubo);
    try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());

    const Vtx = extern struct { x: f32, y: f32 };
    const tri = [3]Vtx{ .{ .x = -1, .y = -1 }, .{ .x = 3, .y = -1 }, .{ .x = -1, .y = 3 } };
    var vbo: gles.GLuint = 0;
    glGenBuffers(1, @ptrCast(&vbo));
    glBindBuffer(gles.GL_ARRAY_BUFFER, vbo);
    glBufferData(gles.GL_ARRAY_BUFFER, @sizeOf(@TypeOf(tri)), &tri, gles.GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, gles.GL_FLOAT, gles.GL_FALSE, @sizeOf(Vtx), @ptrFromInt(0));
    glEnableVertexAttribArray(0);

    glClearColor(0, 0, 0, 1);
    glClear(gles.GL_COLOR_BUFFER_BIT);
    glDrawArrays(gles.GL_TRIANGLES, 0, 3);
    try std.testing.expectEqual(gles.GL_NO_ERROR, glGetError());

    // Center pixel must be (uA, uB.z, uC.y) = (0.2, 0.8, 0.9) -> ~(51, 204, 229). With the OLD
    // tight packing the shader would read uB from the std140 padding (green -> 0) and uC.y from
    // uB.y (blue -> 153): the differential proof each member came from its std140 offset.
    const s: *state.Surface = @ptrCast(@alignCast(surf.?));
    const dev = s.display.device.?;
    const px = try flushMap(dev, s);
    const c = (32 * 64 + 32) * 4;
    try std.testing.expect(px[c + 0] > 39 and px[c + 0] < 63); // uA ~51
    try std.testing.expect(px[c + 1] > 192 and px[c + 1] < 216); // uB.z ~204 (old packing -> 0)
    try std.testing.expect(px[c + 2] > 217 and px[c + 2] < 241); // uC.y ~229 (old packing -> ~153)

    glDeleteBuffers(1, &[_]gles.GLuint{ubo});
    glDeleteBuffers(1, &[_]gles.GLuint{vbo});
    glDeleteProgram(prog);
    glDeleteShader(vs);
    glDeleteShader(fs);
    _ = eglMakeCurrent(dpy, null, null, egl.EGL_NO_CONTEXT);
    _ = eglDestroySurface(dpy, surf);
    _ = eglDestroyContext(dpy, ctx);
}
