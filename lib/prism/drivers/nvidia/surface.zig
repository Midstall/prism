const platform = @import("../../platform.zig");

/// A HAL surface is a handle onto a platform (Wayland / headless / DRM) surface.
/// present() blits the rendered framebuffer into its current buffer.
pub const Surface = struct {
    platform: *platform.Surface,
};
