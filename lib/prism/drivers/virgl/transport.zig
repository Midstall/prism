//! Transport seam for the virgl HAL driver. Comptime-selects the backend by OS.
//! Freestanding: Conduit *Virtio (virtio-mmio SUBMIT_3D). Linux: /dev/dri/renderD* DRM uAPI (EXECBUFFER).

const builtin = @import("builtin");

pub const Transport = if (builtin.target.os.tag == .freestanding)
    @import("transport/freestanding.zig").Transport
else
    @import("transport/linux.zig").Transport;

/// The OS-specific construction argument for `Transport.init`.
pub const InitArgs = if (builtin.target.os.tag == .freestanding)
    @import("transport/freestanding.zig").InitArgs
else
    @import("transport/linux.zig").InitArgs;

// Shared, OS-agnostic types re-exported so the device/context have one source.
pub const types = @import("transport/types.zig");
pub const Error = types.Error;
pub const Box = types.Box;
pub const Resource = types.Resource;
pub const enc = types.enc;
