//! virtio-gpu present backend (freestanding). Presents through Conduit via
//! TRANSFER_TO_HOST_2D. Scans out B8G8R8X8, reports bgra8_unorm so the present
//! blit writes correct bytes in one pass. Caller owns the Virtio and framebuffer.

const builtin = @import("builtin");
const hal = @import("../hal.zig");
const platform = @import("../platform.zig");

// Conduit import: only referenced in freestanding builds, so the host prism
// module never needs a conduit dependency.
const conduit = @import("conduit");
const Virtio = conduit.driver.virtio_gpu.Virtio;

/// Configuration for a virtio-gpu present display. The caller has already run
/// `bind`/`start`/`setup` on `gpu` against the same `bytes` framebuffer.
pub const Config = struct {
    /// The live Conduit virtio-gpu driver, at its final (stable) address.
    gpu: *Virtio,
    /// The scanout framebuffer backing (B8G8R8X8, stride = width*4). This is the
    /// exact buffer `gpu.setup` attached, so writes here reach the host on present.
    bytes: []u8,
    width: u32,
    height: u32,
};

const VirtioSurface = struct {
    gpu: *Virtio,
    bytes: []u8,
    width: u32,
    height: u32,
    stride: u32,

    fn currentBuffer(ptr: *anyopaque) hal.Error!platform.Buffer {
        const self: *VirtioSurface = @ptrCast(@alignCast(ptr));
        return platform.Buffer{
            .bytes = self.bytes,
            .width = self.width,
            .height = self.height,
            .stride = self.stride,
            // bgra8_unorm => the present blit swaps R<->B, yielding the
            // B,G,R,X byte order the virtio-gpu B8G8R8X8 scanout expects.
            .format = .bgra8_unorm,
        };
    }
    fn commit(ptr: *anyopaque) hal.Error!void {
        const self: *VirtioSurface = @ptrCast(@alignCast(ptr));
        if (!self.gpu.present()) return error.DeviceLost;
    }
    fn processEvents(ptr: *anyopaque) hal.Error!platform.WindowEvent {
        _ = ptr;
        // No windowing-system event source. Never resized or closed from the guest.
        return .none;
    }
    fn size(ptr: *anyopaque) [2]u32 {
        const self: *VirtioSurface = @ptrCast(@alignCast(ptr));
        return .{ self.width, self.height };
    }
    fn deinit(ptr: *anyopaque) void {
        // Caller owns the Virtio driver and framebuffer storage. The surface holds
        // only borrows, so nothing to release here.
        _ = ptr;
    }

    const vtable = platform.Surface.VTable{
        .currentBuffer = &currentBuffer,
        .commit = &commit,
        .processEvents = &processEvents,
        .size = &size,
        .deinit = &deinit,
    };
};

const VirtioDisplay = struct {
    cfg: Config,
    surface: VirtioSurface = undefined,

    fn createSurface(ptr: *anyopaque, desc: platform.SurfaceDesc) hal.Error!platform.Surface {
        const self: *VirtioDisplay = @ptrCast(@alignCast(ptr));
        _ = desc; // the surface size is fixed by the scanout setup, not the desc.
        self.surface = .{
            .gpu = self.cfg.gpu,
            .bytes = self.cfg.bytes,
            .width = self.cfg.width,
            .height = self.cfg.height,
            .stride = self.cfg.width * 4,
        };
        return platform.Surface{ .ptr = &self.surface, .vtable = &VirtioSurface.vtable };
    }
    fn deinit(ptr: *anyopaque) void {
        _ = ptr;
    }

    const vtable = platform.Display.VTable{
        .createSurface = &createSurface,
        .deinit = &deinit,
    };
};

/// Build a virtio-gpu present Display over an already-set-up Conduit driver.
/// `storage` holds the display state and must outlive every surface created from
/// it (a static var in the kernel is the intended use). No allocation needed.
pub fn create(storage: *VirtioDisplay, cfg: Config) platform.Display {
    storage.* = .{ .cfg = cfg };
    return platform.Display{ .ptr = storage, .vtable = &VirtioDisplay.vtable };
}

/// Opaque display storage handle (so callers can declare a static var).
pub const Display = VirtioDisplay;
