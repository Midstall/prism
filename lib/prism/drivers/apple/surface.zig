const platform = @import("../../platform.zig");

/// A HAL surface is a handle onto a platform (Wayland / headless / DRM) surface.
/// present() CPU-blits the rendered framebuffer into its current buffer.
/// Mirrors the nvidia driver's surface shim.
pub const Surface = struct {
    platform: *platform.Surface,
};
