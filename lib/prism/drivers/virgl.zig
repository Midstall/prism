//! virgl HAL driver. Lowers Prism render flow to virgl commands executed by virglrenderer on the host GPU.
//! OS-agnostic over a transport seam. Freestanding: Conduit *Virtio (virtio-mmio SUBMIT_3D). Linux: /dev/dri/renderD* (EXECBUFFER).
//! Shaders are SPIR-V lowered to TGSI (prism.spirv.parseSpirv -> vulcan-target.tgsi). Not in the driver registry. Call createDevice(gpa, args) directly.

const std = @import("std");
const builtin = @import("builtin");
const hal = @import("../hal.zig");
const drv = @import("../driver.zig");
const DeviceImpl = @import("virgl/device.zig").Device;
const transport_mod = @import("virgl/transport.zig");

const Driver = drv.Driver;
const Error = drv.Error;

/// The OS-specific construction argument for `createDevice`:
///   - freestanding: `.{ .gpu = &virtio }` (a started, virgl-negotiated Conduit
///     `*Virtio` at a stable address)
///   - linux: `.{ .node = "/dev/dri/renderD128" }` or `.{}` to auto-discover the
///     virtio_gpu render node
pub const InitArgs = transport_mod.InitArgs;

/// The hardcoded passthrough TGSI shaders for the gradient triangle. Retained as
/// the reference virgl encoding the SPIR-V-derived TGSI matches (used by tests).
pub const shader = struct {
    pub const VS_PASSTHROUGH = @import("virgl/virgl_stream.zig").VS_PASSTHROUGH;
    pub const FS_COLOR = @import("virgl/virgl_stream.zig").FS_COLOR;
};

/// The IEEE binary16 (fp16) codec used by the HDR path: decode the fp16 render
/// target the GPU wrote (halfToFloat) to recover linear-light HDR values, and the
/// matching encoder (floatToHalf) for tests. Exposed so the HDR harness can decode
/// the readback. Also re-exports the HAL<->virgl format mapping.
pub const fp16 = struct {
    pub const halfToFloat = @import("virgl/formats.zig").halfToFloat;
    pub const floatToHalf = @import("virgl/formats.zig").floatToHalf;
};

/// Map a HAL color format to its virgl (Gallium PIPE_FORMAT) value.
pub const virglColorFormat = @import("virgl/formats.zig").virglColorFormat;

/// Build a HAL device over the comptime-selected virgl transport. `args` is the
/// platform's construction argument (see `InitArgs`). The device owns its 3D
/// context (created lazily) and the command-stream scratch.
pub fn createDevice(gpa: std.mem.Allocator, args: InitArgs) hal.Error!hal.Device {
    return DeviceImpl.create(gpa, args);
}

/// The concrete virgl Device implementation, exposed so a display owner can reach
/// the zero-copy scanout helper (`Device.adoptScanout`) given a `hal.Device` whose
/// backing is a virgl device. Recover the impl with `deviceOf(hal_device)`.
pub const Device = DeviceImpl;

/// Recover the concrete virgl `Device` from a `hal.Device` produced by
/// `createDevice` (so the caller can call `adoptScanout` for zero-copy scanout).
pub fn deviceOf(dev: hal.Device) *DeviceImpl {
    return @ptrCast(@alignCast(dev.ptr));
}

/// The scanout descriptor returned by the zero-copy path (re-exported).
pub const Scanout = DeviceImpl.Scanout;

const on_linux = builtin.target.os.tag == .linux;

const State = struct {};
var state: State = .{};

/// On Linux, true iff a usable virtio-gpu DRM render node exists: scan
/// /dev/dri/renderD128.. for one whose /sys/class/drm/renderD<N>/device/driver
/// symlink target basename is "virtio_gpu" (side-effect-free sysfs probe, no node opened).
/// False on freestanding.
fn available(ptr: *anyopaque) bool {
    _ = ptr;
    if (!on_linux) return false;
    return hasVirtioGpuRenderNode();
}

fn registryCreateDevice(ptr: *anyopaque, gpa: std.mem.Allocator) Error!hal.Device {
    _ = ptr;
    if (!on_linux) return error.NotImplemented;
    // Auto-discover the virtio_gpu render node (node = null). If none exists the
    // transport init fails and the error propagates, so createBestDevice skips
    // virgl gracefully on a box with no virtio-gpu.
    return createDevice(gpa, .{ .node = null });
}

/// Side-effect-free probe: does a /dev/dri/renderD* node bound to the
/// "virtio_gpu" kernel driver exist? Mirrors transport/linux.zig's discover()
/// sysfs walk, but only reads the driver symlink (it never open()s the node).
fn hasVirtioGpuRenderNode() bool {
    var n: u32 = 128;
    while (n < 192) : (n += 1) {
        var nbuf: [80]u8 = undefined;
        const sys = std.fmt.bufPrint(&nbuf, "/sys/class/drm/renderD{d}/device/driver", .{n}) catch continue;
        var lbuf: [256]u8 = undefined;
        const link = std.os.linux.readlink(@ptrCast(sys.ptr), &lbuf, lbuf.len);
        const ll: isize = @bitCast(link);
        if (ll <= 0) continue;
        const target = lbuf[0..@intCast(ll)];
        if (std.mem.indexOf(u8, target, "virtio_gpu") != null or std.mem.indexOf(u8, target, "virtio-gpu") != null) {
            return true;
        }
    }
    return false;
}

const vtable = Driver.VTable{ .isAvailable = &available, .createDevice = &registryCreateDevice };

/// The registry entry for the virgl driver. `drm_driver = "virtio_gpu"` lets the
/// compositor-device-match path (drivers.selectForDrmDevice) resolve a virtio-gpu
/// render device to this driver.
pub const driver = Driver{ .name = "virgl", .ptr = &state, .vtable = &vtable, .drm_driver = "virtio_gpu" };

test "virgl registry driver exposes its name and drm_driver" {
    try std.testing.expectEqualStrings("virgl", driver.name);
    try std.testing.expectEqualStrings("virtio_gpu", driver.drm_driver.?);
    // Availability needs a real virtio-gpu render node. Exercise the full path
    // when present, otherwise confirm createDevice is skipped cleanly.
    if (driver.isAvailable()) {
        const dev = try driver.createDevice(std.testing.allocator);
        dev.deinit();
    }
}

test "virgl stream encoder produces a non-empty draw stream" {
    // A host-side sanity check of the encoder shape (no device needed). It runs
    // on any target because it only touches the local virgl encoding namespace.
    const stream_mod = @import("virgl/virgl_stream.zig");
    const enc = @import("virgl/encoding.zig");
    var words: [1024]u32 = undefined;
    var encoder = stream_mod.Encoder.init(&words);
    const n = encoder.encodeDraw(.{
        .rt_res = 16,
        .vbuf_res = 17,
        .fb_w = 256,
        .fb_h = 256,
        .rt_format = enc.FORMAT_B8G8R8X8_UNORM,
        .vertex_stride = 24,
        .pos_format = enc.FORMAT_R32G32_FLOAT,
        .pos_offset = 0,
        .color_format = enc.FORMAT_R32G32B32A32_FLOAT,
        .color_offset = 8,
        .vs_tgsi = shader.VS_PASSTHROUGH,
        .fs_tgsi = shader.FS_COLOR,
        .clear = .{ 0.1, 0.1, 0.15, 1.0 },
        .vertex_count = 3,
    }, .{});
    try std.testing.expect(n > 40); // a real draw stream is dozens of words
}

test {
    // Pull the format-mapping + fp16 codec tests (formats.zig) into the run.
    _ = @import("virgl/formats.zig");
}

test "hdr render target lowers to the fp16 virgl pipe format" {
    // The HDR proof at the encoding layer: an rgba16_float color target maps to
    // R16G16B16A16_FLOAT (65) so virglrenderer allocates a high-precision surface.
    const formats = @import("virgl/formats.zig");
    const enc = @import("virgl/encoding.zig");
    try std.testing.expectEqual(enc.FORMAT_R16G16B16A16_FLOAT, formats.virglColorFormat(.rgba16_float));
    try std.testing.expectEqual(@as(u32, 94), formats.virglColorFormat(.rgba16_float));
    try std.testing.expectEqual(@as(u32, 8), formats.bytesPerPixel(.rgba16_float));
    // And the SDR default is unchanged (the 8-bit path must not regress).
    try std.testing.expectEqual(enc.FORMAT_B8G8R8X8_UNORM, formats.virglColorFormat(.bgra8_unorm));
}

test "virgl device + transport type-checks on the host (linux) target" {
    // Force Sema-analysis of the full driver -> transport seam on Linux: the
    // device type and the selected (linux) transport's InitArgs must compile.
    // We do not open a render node here (CI has no virtio_gpu node), so this is a
    // compile-only check. The live DRM path runs in the Linux-guest harness.
    _ = DeviceImpl;
    _ = InitArgs;
    try std.testing.expect(@hasDecl(transport_mod, "Transport"));
}
