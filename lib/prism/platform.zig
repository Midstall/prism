const builtin = @import("builtin");

pub const Buffer = @import("platform/surface.zig").Buffer;
pub const SurfaceDesc = @import("platform/surface.zig").SurfaceDesc;
pub const WindowEvent = @import("platform/surface.zig").WindowEvent;
pub const Surface = @import("platform/surface.zig").Surface;
pub const Display = @import("platform/surface.zig").Display;

// headless is host-agnostic (pure allocator-backed framebuffer) and compiles
// on any target, freestanding included.
pub const headless = @import("platform/headless.zig");

// Wayland and DRM backends use std.os.linux / std.posix / the wayland
// subproject module, none of which exist on freestanding targets. Both are
// gated behind a comptime os check so a freestanding build never analyzes them.
pub const drm = if (builtin.target.os.tag == .linux) @import("platform/drm.zig") else struct {};
pub const wayland = if (builtin.target.os.tag == .linux) @import("platform/wayland.zig") else struct {};

// GBM present backend: adapts the gbm package's Surface to a platform.Surface for
// the EGL GBM platform. The gbm package is portable, but the EGL frontend that
// drives it is host-only, so it is gated with the other host backends.
pub const gbm = if (builtin.target.os.tag == .linux) @import("platform/gbm.zig") else struct {};

// virtio-gpu present backend (Conduit driver). Only meaningful on freestanding
// where the guest kernel owns the device and identity-maps it. Gated to
// freestanding so the Linux build never needs conduit.
pub const virtio_gpu = if (builtin.target.os.tag == .freestanding) @import("platform/virtio_gpu.zig") else struct {};
