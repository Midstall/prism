//! KMS/DRM present backend. Prism owns the display: opens /dev/dri/card0,
//! becomes DRM master, modesets a connected connector, and double-buffers dumb
//! framebuffers. Linux-only. Opposite of platform/wayland.zig (a Wayland client).

const std = @import("std");
const builtin = @import("builtin");
const drm = @import("drm");
const hal = @import("../hal.zig");
const platform = @import("../platform.zig");
const prism_log = @import("../log.zig");

const linux = std.os.linux;

// KMS enum values compared against here. The master/fb/crtc/blob/property/atomic
// ioctls now all route through drm.Node + drm.types, so no local request encoders
// remain. What stays are the plain KMS constants the module tests against.
const DRM_MODE_CONNECTED: u32 = 1;
const DRM_MODE_TYPE_PREFERRED: u32 = 1 << 3;
const DRM_MODE_PAGE_FLIP_EVENT: u32 = 0x01;

/// FourCC 'XR24' = DRM_FORMAT_XRGB8888: 32bpp, 0x00RRGGBB little-endian. The
/// dumb fbs and the virgl B8G8R8X8 render target use the same byte order.
const DRM_FORMAT_XRGB8888: u32 = fourcc('X', 'R', '2', '4');

/// FourCC 'XR30' = DRM_FORMAT_XRGB2101010: 32bpp, the 10-bit-per-channel HDR10
/// scanout format (2-10-10-10, x:R:G:B from MSB, little-endian dword). This is
/// the hal.Format.rgb10x2 identity on the wire (matches drm_fourcc.h).
const DRM_FORMAT_XRGB2101010: u32 = fourcc('X', 'R', '3', '0');

fn fourcc(a: u8, b: u8, c: u8, d: u8) u32 {
    return @as(u32, a) | (@as(u32, b) << 8) | (@as(u32, c) << 16) | (@as(u32, d) << 24);
}

// HDR metadata uAPI (drm_mode.h). Byte layout verified against linux-headers 6.16.7
// with a C offsetof probe: infoframe is 26 bytes (align 2), eotf=0, metadata_type=1,
// display_primaries=2 (3*4), white_point=14, max_dml=18, min_dml=20, max_cll=22,
// max_fall=24. hdr_output_metadata is 32 bytes: u32 metadata_type at 0, infoframe
// union at offset 4 (4 + 26 padded to 32 by 4-byte alignment). Kept local: the
// drm module does not model the HDR infoframe blob payload.

/// drm_mode.h `struct { __u16 x, y; }` chromaticity, units of 0.00002 (0xC350 ==
/// 1.0000). align 2 so this packs with no padding inside the infoframe.
const drm_chromaticity = extern struct {
    x: u16 = 0,
    y: u16 = 0,
};

const hdr_metadata_infoframe = extern struct {
    eotf: u8 = 0,
    metadata_type: u8 = 0,
    display_primaries: [3]drm_chromaticity = [_]drm_chromaticity{.{}} ** 3,
    white_point: drm_chromaticity = .{},
    max_display_mastering_luminance: u16 = 0,
    min_display_mastering_luminance: u16 = 0,
    max_cll: u16 = 0,
    max_fall: u16 = 0,
};

const hdr_output_metadata = extern struct {
    metadata_type: u32 = 0,
    // The kernel union is just hdmi_metadata_type1 today. We inline the only
    // member. The u32 above forces 4-byte alignment so the struct is 32 bytes,
    // matching sizeof(struct hdr_output_metadata).
    hdmi_metadata_type1: hdr_metadata_infoframe = .{},
};

// CTA-861.G EOTF codes (kernel include/linux/hdmi.h enum hdmi_eotf): the value
// written into the infoframe's `eotf` byte. Stable ABI (an HDMI/CTA spec enum).
const HDMI_EOTF_TRADITIONAL_GAMMA_SDR: u8 = 0;
const HDMI_EOTF_TRADITIONAL_GAMMA_HDR: u8 = 1;
const HDMI_EOTF_SMPTE_ST2084: u8 = 2; // PQ (HDR10)
const HDMI_EOTF_BT_2100_HLG: u8 = 3; // HLG
// Static_Metadata_Descriptor_ID 0 == Static Metadata Type 1 (the only one
// defined). Written into both hdr_output_metadata.metadata_type and the
// infoframe.metadata_type byte.
const HDMI_STATIC_METADATA_TYPE1: u8 = 0;

// One scanout framebuffer: a dumb BO, its KMS fb_id, and the CPU mapping.
const FrameBuffer = struct {
    handle: u32,
    fb_id: u32,
    pitch: u32,
    size: u64,
    bytes: []u8,
};

// Route a failing open's errno through the display logger. Default-silent: an
// unset `log` emits nothing, so tests and graceful-fail paths produce no output.
fn dbg(logger: prism_log.Logger, comptime what: []const u8, e: std.posix.E) void {
    logger.log("{s} errno={d}", .{ what, @intFromEnum(e) });
}

// Log a note when a node operation failed. The typed drm.Node methods map errno
// to error unions, so the specific code is no longer visible here.
fn note(logger: prism_log.Logger, comptime what: []const u8) void {
    logger.log("{s}", .{what});
}

// Modeset `crtc_id` onto `fb_id` for `connector_id` with `mode` (the standard
// full-surface SETCRTC: origin 0,0, mode valid). Routes through node.setModeCrtc.
fn setCrtcOn(node: drm.Node, connector_id: u32, crtc_id: u32, mode: drm.types.ModeInfo, fb_id: u32) !void {
    var conn = [_]u32{connector_id};
    try node.setModeCrtc(.{
        .setConnectorsPtr = @intFromPtr(&conn),
        .countConnectors = 1,
        .crtcId = crtc_id,
        .fbId = fb_id,
        .modeValid = 1,
        .mode = mode,
    });
}

// Read the current CRTC state so deinit can restore it. A failed getCrtc leaves
// a zeroed state (no mode, no fb), which restoreSavedCrtc then skips.
fn savedCrtcFor(node: drm.Node, crtc_id: u32) drm.types.ModeGetCrtc {
    return node.getCrtc(crtc_id) catch drm.types.ModeGetCrtc{ .crtcId = crtc_id };
}

// Best-effort restore of the CRTC state captured by getCrtc before modesetting.
// A saved state with no mode and no fb means there was nothing to restore.
fn restoreSavedCrtc(node: drm.Node, connector_id: u32, saved: drm.types.ModeGetCrtc) void {
    if (saved.modeValid == 0 and saved.fbId == 0) return;
    var conn = [_]u32{connector_id};
    node.setModeCrtc(.{
        .setConnectorsPtr = @intFromPtr(&conn),
        .countConnectors = 1,
        .crtcId = saved.crtcId,
        .fbId = saved.fbId,
        .x = saved.x,
        .y = saved.y,
        .gammaSize = saved.gammaSize,
        .modeValid = saved.modeValid,
        .mode = saved.mode,
    }) catch {};
}

// Surface: one DRM CRTC scanning out a double-buffered dumb framebuffer.
const DrmSurface = struct {
    gpa: std.mem.Allocator,
    node: drm.Node, // shared with the owning DrmDisplay, not closed here
    crtc_id: u32,
    connector_id: u32,
    mode: drm.types.ModeInfo,
    width: u32,
    height: u32,
    fbs: [2]FrameBuffer,
    front: usize, // currently scanned-out buffer
    use_page_flip: bool,
    log: prism_log.Logger = .{},

    fn back(self: *DrmSurface) usize {
        return self.front ^ 1;
    }

    fn currentBuffer(ptr: *anyopaque) hal.Error!platform.Buffer {
        const self: *DrmSurface = @ptrCast(@alignCast(ptr));
        const fb = self.fbs[self.back()];
        return platform.Buffer{
            .bytes = fb.bytes,
            .width = self.width,
            .height = self.height,
            .stride = fb.pitch,
            // bgra8_unorm => the software present blit swaps R<->B, yielding the
            // B,G,R,X byte order the dumb XRGB8888 (0x00RRGGBB LE) fb expects.
            .format = .bgra8_unorm,
        };
    }

    fn commit(ptr: *anyopaque) hal.Error!void {
        const self: *DrmSurface = @ptrCast(@alignCast(ptr));
        const b = self.back();
        const fb = self.fbs[b];

        if (self.use_page_flip) {
            if (self.node.pageFlip(self.crtc_id, fb.fb_id, DRM_MODE_PAGE_FLIP_EVENT)) |_| {
                self.waitFlip();
                self.front = b;
                return;
            } else |_| {
                // Page-flip not usable on this device. Drop to SETCRTC per frame.
                note(self.log, "PAGE_FLIP (falling back to SETCRTC)");
                self.use_page_flip = false;
            }
        }

        // Fallback: SETCRTC scans out the back buffer directly.
        self.setCrtc(fb.fb_id) catch return error.DeviceLost;
        self.front = b;
    }

    /// Block on the drm fd for the FLIP_COMPLETE event of the page flip we just
    /// queued. Drains one event. A raw read (not drm.Node.getEvent) because the
    /// module's event body layout does not match the kernel drm_event_vblank.
    fn waitFlip(self: *DrmSurface) void {
        var buf: [256]u8 align(8) = undefined;
        const n = linux.read(self.node.fd, &buf, buf.len);
        const got: isize = @bitCast(n);
        if (got < @as(isize, @sizeOf(drm.types.Event.Base))) return;
        // The buffer may hold one or more events back to back. We only need to
        // consume the flip-complete, which the read above delivered.
    }

    fn setCrtc(self: *DrmSurface, fb_id: u32) !void {
        return setCrtcOn(self.node, self.connector_id, self.crtc_id, self.mode, fb_id);
    }

    fn processEvents(ptr: *anyopaque) hal.Error!platform.WindowEvent {
        _ = ptr;
        // No windowing-system event source, no input wired. Never resized or closed.
        return .none;
    }

    fn size(ptr: *anyopaque) [2]u32 {
        const self: *DrmSurface = @ptrCast(@alignCast(ptr));
        return .{ self.width, self.height };
    }

    fn deinit(ptr: *anyopaque) void {
        const self: *DrmSurface = @ptrCast(@alignCast(ptr));
        const gpa = self.gpa;
        for (&self.fbs) |*fb| {
            _ = linux.munmap(@ptrCast(fb.bytes.ptr), fb.bytes.len);
            self.node.rmFb(fb.fb_id) catch {};
            self.node.destroyDumb(fb.handle) catch {};
        }
        gpa.destroy(self);
    }

    const vtable = platform.Surface.VTable{
        .currentBuffer = &currentBuffer,
        .commit = &commit,
        .processEvents = &processEvents,
        .size = &size,
        .deinit = &deinit,
    };
};

// Display: owns the DRM node (fd), connector/crtc/mode, and saved CRTC state.
const DrmDisplay = struct {
    gpa: std.mem.Allocator,
    node: drm.Node,
    have_master: bool,
    connector_id: u32,
    crtc_id: u32,
    mode: drm.types.ModeInfo,
    saved_crtc: drm.types.ModeGetCrtc,
    log: prism_log.Logger = .{},

    fn createSurface(ptr: *anyopaque, desc: platform.SurfaceDesc) hal.Error!platform.Surface {
        const self: *DrmDisplay = @ptrCast(@alignCast(ptr));
        _ = desc; // the surface size is fixed by the modeset, not the desc.

        const w: u32 = self.mode.hdisplay;
        const h: u32 = self.mode.vdisplay;

        var fbs: [2]FrameBuffer = undefined;
        var created: usize = 0;
        errdefer for (fbs[0..created]) |*fb| {
            _ = linux.munmap(@ptrCast(fb.bytes.ptr), fb.bytes.len);
            self.node.rmFb(fb.fb_id) catch {};
            self.node.destroyDumb(fb.handle) catch {};
        };

        while (created < 2) : (created += 1) {
            fbs[created] = try self.createFb(w, h);
        }

        const s = self.gpa.create(DrmSurface) catch {
            return error.OutOfMemory;
        };
        s.* = .{
            .gpa = self.gpa,
            .node = self.node,
            .crtc_id = self.crtc_id,
            .connector_id = self.connector_id,
            .mode = self.mode,
            .width = w,
            .height = h,
            .fbs = fbs,
            .front = 0,
            .use_page_flip = true,
            .log = self.log,
        };

        // Modeset on framebuffer 0. Requires DRM master, which Display.init secured.
        s.setCrtc(fbs[0].fb_id) catch {
            self.gpa.destroy(s);
            return error.DeviceLost;
        };
        return platform.Surface{ .ptr = s, .vtable = &DrmSurface.vtable };
    }

    /// Allocate one dumb scanout buffer at w*h, add it as a KMS fb, and mmap it.
    fn createFb(self: *DrmDisplay, w: u32, h: u32) hal.Error!FrameBuffer {
        const cd = self.node.createDumb(w, h, 32) catch return error.DeviceLost;
        errdefer self.node.destroyDumb(cd.handle) catch {};

        // Legacy ADDFB carries a depth field (24) the ADDFB2 path omits.
        const fb = self.node.addFb(w, h, cd.pitch, 32, 24, cd.handle) catch return error.DeviceLost;
        errdefer self.node.rmFb(fb.fbId) catch {};

        const md = self.node.mapDumb(cd.handle) catch return error.DeviceLost;

        const r = linux.mmap(null, cd.size, .{ .READ = true, .WRITE = true }, .{ .TYPE = .SHARED }, self.node.fd, @intCast(md.offset));
        if (std.posix.errno(r) != .SUCCESS) return error.DeviceLost;
        const ptr: [*]u8 = @ptrFromInt(r);
        const bytes = ptr[0..cd.size];
        @memset(bytes, 0);

        return .{ .handle = cd.handle, .fb_id = fb.fbId, .pitch = cd.pitch, .size = cd.size, .bytes = bytes };
    }

    fn deinit(ptr: *anyopaque) void {
        const self: *DrmDisplay = @ptrCast(@alignCast(ptr));
        // Restore the CRTC to the state we saved before modesetting (best effort).
        restoreSavedCrtc(self.node, self.connector_id, self.saved_crtc);
        if (self.have_master) self.node.dropMaster() catch {};
        self.node.deinit();
        self.gpa.destroy(self);
    }

    const vtable = platform.Display.VTable{
        .createSurface = &createSurface,
        .deinit = &deinit,
    };
};

// Display construction: open card0, become master, pick connector/crtc/mode.

/// Open the primary DRM node (default /dev/dri/card0), become DRM master, and
/// resolve a connected connector + its preferred mode + a usable CRTC. Returns a
/// platform.Display that owns the display. `path` overrides the device path.
pub fn create(gpa: std.mem.Allocator) hal.Error!platform.Display {
    return createNode(gpa, "/dev/dri/card0");
}

pub fn createNode(gpa: std.mem.Allocator, path: []const u8) hal.Error!platform.Display {
    var zbuf: [80]u8 = undefined;
    const z = std.fmt.bufPrintZ(&zbuf, "{s}", .{path}) catch return error.InvalidArgument;
    const ofd = linux.open(z.ptr, .{ .ACCMODE = .RDWR, .CLOEXEC = true }, 0);
    if (std.posix.errno(ofd) != .SUCCESS) {
        dbg(.{}, "open card0", std.posix.errno(ofd));
        return error.InitializationFailed;
    }
    const node = drm.Node{ .allocator = gpa, .fd = @intCast(ofd) };
    errdefer node.deinit();

    // Become DRM master. In a guest with no other master the opener already is
    // master implicitly, so a failure here is non-fatal (we proceed and let the
    // first modeset confirm). Track success so deinit drops it.
    const have_master = if (node.setMaster()) true else |_| false;

    const pick = try resolveOutput(node);
    const saved = savedCrtcFor(node, pick.crtc_id);

    const d = gpa.create(DrmDisplay) catch return error.OutOfMemory;
    d.* = .{
        .gpa = gpa,
        .node = node,
        .have_master = have_master,
        .connector_id = pick.connector_id,
        .crtc_id = pick.crtc_id,
        .mode = pick.mode,
        .saved_crtc = saved,
    };
    return platform.Display{ .ptr = d, .vtable = &DrmDisplay.vtable };
}

const Output = struct {
    connector_id: u32,
    crtc_id: u32,
    mode: drm.types.ModeInfo,
};

// HDR display output layer: enumerate connector KMS properties, resolve
// HDR_OUTPUT_METADATA and Colorspace by name, build a CTA-861.G infoframe,
// upload as a blob. Degrades to SDR on non-HDR connectors without error.
// Real-hardware verified: HDR panel receives PQ/BT.2020 and shows the 10-bit ramp.

/// EOTF the display should apply. Mirrors the Wayland color-management EOTFs.
pub const Eotf = enum {
    traditional_sdr,
    traditional_hdr,
    pq, // SMPTE ST 2084 (HDR10 transfer function)
    hlg, // BT.2100 Hybrid Log-Gamma

    fn hdmiCode(self: Eotf) u8 {
        return switch (self) {
            .traditional_sdr => HDMI_EOTF_TRADITIONAL_GAMMA_SDR,
            .traditional_hdr => HDMI_EOTF_TRADITIONAL_GAMMA_HDR,
            .pq => HDMI_EOTF_SMPTE_ST2084,
            .hlg => HDMI_EOTF_BT_2100_HLG,
        };
    }
};

/// A chromaticity in 0.00002 units (drm_mode.h: 0xC350 == 1.0000), i.e. the
/// CIE 1931 xy coordinate the kernel infoframe expects.
pub const Chromaticity = struct {
    x: u16,
    y: u16,

    /// Build from CIE xy floats in [0,1]. 1.0000 -> 0xC350 (50000), per spec.
    pub fn fromXy(x: f32, y: f32) Chromaticity {
        return .{ .x = encodeChroma(x), .y = encodeChroma(y) };
    }
};

fn encodeChroma(v: f32) u16 {
    const scaled = @round(v * 50000.0);
    if (scaled <= 0) return 0;
    if (scaled >= 65535) return 65535;
    return @intFromFloat(scaled);
}

/// Compositor-facing HDR metadata input. Mirrors the shape the Wayland
/// color-management layer produces (EOTF + primaries/white in 0.00002-unit
/// chromaticities + mastering luminance + content light levels). Feeds the
/// CTA-861.G Static Metadata Type 1 infoframe.
pub const HdrMetadata = struct {
    eotf: Eotf,
    /// RGB primaries in display order (red, green, blue).
    display_primaries: [3]Chromaticity,
    white_point: Chromaticity,
    /// Max mastering display luminance, units of 1 cd/m2 (1..65535).
    max_display_mastering_luminance: u16,
    /// Min mastering display luminance, units of 0.0001 cd/m2.
    min_display_mastering_luminance: u16,
    /// Max content light level (MaxCLL), units of 1 cd/m2.
    max_cll: u16,
    /// Max frame-average light level (MaxFALL), units of 1 cd/m2.
    max_fall: u16,

    /// Fill a CTA-861.G hdr_output_metadata (Static Metadata Type 1) from this
    /// input. metadata_type == HDMI_STATIC_METADATA_TYPE1 (0) on both the outer
    /// descriptor id and the infoframe byte. The EOTF byte carries the transfer
    /// function. The byte layout is asserted by the unit tests below.
    pub fn toOutputMetadata(self: HdrMetadata) hdr_output_metadata {
        var out = hdr_output_metadata{ .metadata_type = HDMI_STATIC_METADATA_TYPE1 };
        out.hdmi_metadata_type1 = .{
            .eotf = self.eotf.hdmiCode(),
            .metadata_type = HDMI_STATIC_METADATA_TYPE1,
            .display_primaries = .{
                .{ .x = self.display_primaries[0].x, .y = self.display_primaries[0].y },
                .{ .x = self.display_primaries[1].x, .y = self.display_primaries[1].y },
                .{ .x = self.display_primaries[2].x, .y = self.display_primaries[2].y },
            },
            .white_point = .{ .x = self.white_point.x, .y = self.white_point.y },
            .max_display_mastering_luminance = self.max_display_mastering_luminance,
            .min_display_mastering_luminance = self.min_display_mastering_luminance,
            .max_cll = self.max_cll,
            .max_fall = self.max_fall,
        };
        return out;
    }

    /// The common HDR10 case: PQ (SMPTE ST 2084) transfer + BT.2020 (Rec.2020)
    /// primaries + D65 white. `min_nits`/`max_nits` are the mastering display
    /// luminance (e.g. 0.005..1000). `max_cll`/`max_fall` are content light
    /// levels in cd/m2. BT.2020 (ITU-R BT.2020): R 0.708/0.292, G 0.170/0.797,
    /// B 0.131/0.046. D65 white 0.3127/0.3290.
    pub fn pqRec2020(min_nits: f32, max_nits: f32, max_cll: u16, max_fall: u16) HdrMetadata {
        return .{
            .eotf = .pq,
            .display_primaries = .{
                Chromaticity.fromXy(0.708, 0.292), // red
                Chromaticity.fromXy(0.170, 0.797), // green
                Chromaticity.fromXy(0.131, 0.046), // blue
            },
            .white_point = Chromaticity.fromXy(0.3127, 0.3290), // D65
            .max_display_mastering_luminance = nitsToU16(max_nits),
            .min_display_mastering_luminance = minNitsToU16(min_nits),
            .max_cll = max_cll,
            .max_fall = max_fall,
        };
    }
};

/// Max-luminance field: 1 cd/m2 per unit (1..65535).
fn nitsToU16(nits: f32) u16 {
    const v = @round(nits);
    if (v <= 1) return 1;
    if (v >= 65535) return 65535;
    return @intFromFloat(v);
}

/// Min-luminance field: 0.0001 cd/m2 per unit (so 0.005 nits -> 50).
fn minNitsToU16(nits: f32) u16 {
    const v = @round(nits * 10000.0);
    if (v <= 0) return 0;
    if (v >= 65535) return 65535;
    return @intFromFloat(v);
}

/// Pack an 8-bit RGB triple into an XRGB2101010 (XR30) dword: x(2):R(10):G(10):
/// B(10) from the MSB, scaling each 8-bit channel up to 10 bits. Little-endian
/// in memory, matching DRM_FORMAT_XRGB2101010. The 8->10 scale uses (c<<2)|(c>>6)
/// so 0xFF maps to 0x3FF (full white stays full).
pub fn packXrgb2101010(r8: u8, g8: u8, b8: u8) u32 {
    const r10: u32 = expand8to10(r8);
    const g10: u32 = expand8to10(g8);
    const b10: u32 = expand8to10(b8);
    return (r10 << 20) | (g10 << 10) | b10; // x bits (31:30) left zero
}

fn expand8to10(c: u8) u32 {
    const v: u32 = c;
    return (v << 2) | (v >> 6);
}

/// A connector's resolved HDR-relevant property ids (0 == the connector does not
/// expose that property). HDR_OUTPUT_METADATA + Colorspace presence is what
/// makes a connector "HDR-capable".
const HdrProps = struct {
    hdr_output_metadata_id: u32 = 0,
    colorspace_id: u32 = 0,
    colorspace_bt2020_rgb: u64 = 0, // resolved enum value, by name
    max_bpc_id: u32 = 0,

    fn isHdrCapable(self: HdrProps) bool {
        return self.hdr_output_metadata_id != 0;
    }
};

/// Enumerate a connector's properties (objGetProperties) and resolve the
/// HDR-relevant ones by name: "HDR_OUTPUT_METADATA", "Colorspace" (plus its
/// "BT2020_RGB" enum value from the enum table, since drivers order enums
/// differently and hardcoding the integer is wrong), and "max bpc" (for 10-bit).
/// On virtio-gpu none of these exist. Returned ids stay 0 (not HDR-capable).
fn resolveHdrProps(node: drm.Node, connector_id: u32) HdrProps {
    var result = HdrProps{};

    var op = node.objGetProperties(connector_id, drm.types.OBJECT_CONNECTOR) catch return result;
    defer op.deinit(node.allocator);
    const ids = op.props() orelse return result; // no properties -> not HDR-capable

    for (ids) |prop_id| {
        var gp = node.getProperty(prop_id) catch continue;
        defer gp.deinit(node.allocator);
        const name = cstr(&gp.name);
        if (std.mem.eql(u8, name, "HDR_OUTPUT_METADATA")) {
            result.hdr_output_metadata_id = prop_id;
        } else if (std.mem.eql(u8, name, "Colorspace")) {
            result.colorspace_id = prop_id;
            result.colorspace_bt2020_rgb = resolveEnumValue(&gp, "BT2020_RGB") orelse 0;
        } else if (std.mem.eql(u8, name, "max bpc")) {
            result.max_bpc_id = prop_id;
        }
    }
    return result;
}

/// For an ENUM property (the getProperty result `gp`, already allocated), read
/// its (value -> name) table and return the numeric value whose name matches
/// `want`. Returns null when `gp` is not an enum property.
fn resolveEnumValue(gp: *const drm.types.ModeGetProperty, want: []const u8) ?u64 {
    const ens = gp.enums() orelse return null;
    for (ens) |en| {
        if (std.mem.eql(u8, cstr(&en.name), want)) return en.value;
    }
    return null;
}

/// A NUL-terminated (or full-length) fixed buffer -> the string slice.
fn cstr(buf: []const u8) []const u8 {
    const end = std.mem.indexOfScalar(u8, buf, 0) orelse buf.len;
    return buf[0..end];
}

// GPU zero-copy scanout: holds the card0 fd so a virgl context uses the same fd,
// creates a virtio-gpu resource (BIND_RENDER_TARGET|BIND_SCANOUT), wraps its GEM
// handle via ADDFB2, and modesets/page-flips it. No readback, no blit.
// Caller shares this fd as virgl InitArgs.external_fd, then calls addScanoutFb
// to get a fb_id for setCrtc/pageFlip. Optionally drives an atomic commit path.

/// Cached KMS property ids for the atomic modeset/present path. Populated by
/// enableAtomicMode. `mode_blob_id` tracks the last MODE_ID blob so it is freed
/// on the next modeset or teardown.
const AtomicState = struct {
    plane_id: u32,
    crtc_mode_id: u32,
    crtc_active: u32,
    conn_crtc_id: u32,
    plane_fb_id: u32,
    plane_crtc_id: u32,
    plane_src_x: u32,
    plane_src_y: u32,
    plane_src_w: u32,
    plane_src_h: u32,
    plane_crtc_x: u32,
    plane_crtc_y: u32,
    plane_crtc_w: u32,
    plane_crtc_h: u32,
    mode_blob_id: u32 = 0,
};

/// 16.16 fixed point: a plane SRC_W/SRC_H is width/height in the high 16 bits.
fn fp16(v: u32) u64 {
    return @as(u64, v) << 16;
}

pub const GpuDisplay = struct {
    gpa: std.mem.Allocator,
    node: drm.Node,
    have_master: bool,
    connector_id: u32,
    crtc_id: u32,
    mode: drm.types.ModeInfo,
    saved_crtc: drm.types.ModeGetCrtc,
    atomic: ?AtomicState = null,
    log: prism_log.Logger = .{},

    /// The shared DRM fd (card0): master + KMS + virtgpu on one node. Pass this as
    /// the virgl Linux transport's `external_fd` so the virgl 3D context and the
    /// scanout resource live on the same fd this object modesets/page-flips.
    pub fn deviceFd(self: *GpuDisplay) linux.fd_t {
        return self.node.fd;
    }

    pub fn modeWidth(self: *GpuDisplay) u32 {
        return self.mode.hdisplay;
    }
    pub fn modeHeight(self: *GpuDisplay) u32 {
        return self.mode.vdisplay;
    }

    /// Wrap a virtio-gpu resource's GEM handle as an XRGB8888 KMS framebuffer via
    /// ADDFB2. `handle` is the GEM bo handle for the scanout-able resource (created
    /// with BIND_RENDER_TARGET|BIND_SCANOUT). The returned fb_id is what setCrtc/
    /// pageFlip reference. `pitch` is bytes/row (width*4 for XRGB8888).
    pub fn addScanoutFb(self: *GpuDisplay, handle: u32, width: u32, height: u32, pitch: u32) hal.Error!u32 {
        var fb = drm.types.ModeFbCmd2{
            .width = width,
            .height = height,
            .pixelFormat = DRM_FORMAT_XRGB8888,
            .flags = .{},
        };
        fb.handles[0] = handle;
        fb.pitches[0] = pitch;
        fb.offsets[0] = 0;
        self.node.addFb2(&fb) catch {
            note(self.log, "ADDFB2 (virtgpu scanout resource)");
            return error.DeviceLost;
        };
        return fb.fbId;
    }

    /// MODE_SETCRTC: modeset the connector onto `fb_id`. The first present uses
    /// SETCRTC. Subsequent frames can page-flip.
    pub fn setCrtc(self: *GpuDisplay, fb_id: u32) hal.Error!void {
        setCrtcOn(self.node, self.connector_id, self.crtc_id, self.mode, fb_id) catch {
            note(self.log, "SETCRTC (gpu scanout)");
            return error.DeviceLost;
        };
    }

    /// MODE_PAGE_FLIP `fb_id` and block on the FLIP_COMPLETE event. Returns
    /// DeviceLost if the flip ioctl fails (the caller may fall back to setCrtc).
    pub fn pageFlip(self: *GpuDisplay, fb_id: u32) hal.Error!void {
        self.node.pageFlip(self.crtc_id, fb_id, DRM_MODE_PAGE_FLIP_EVENT) catch {
            note(self.log, "PAGE_FLIP (gpu scanout)");
            return error.DeviceLost;
        };
        // Drain one FLIP_COMPLETE event.
        var buf: [256]u8 align(8) = undefined;
        _ = linux.read(self.node.fd, &buf, buf.len);
    }

    /// Opt into atomic mode setting (a compositor capability). Enables the
    /// UNIVERSAL_PLANES + ATOMIC client caps and resolves the CRTC/connector/
    /// primary-plane property ids. On success setCrtcAtomic/pageFlipAtomic drive
    /// the KMS atomic path. Returns false (leaving the legacy path in force) if
    /// the driver lacks atomic or a primary plane cannot be found.
    pub fn enableAtomicMode(self: *GpuDisplay) bool {
        const node = self.node;
        if (!enableAtomic(node)) return false;
        const crtc_index = crtcIndexOf(node, self.crtc_id) orelse return false;
        const plane_id = primaryPlaneFor(node, crtc_index) orelse return false;
        self.atomic = AtomicState{
            .plane_id = plane_id,
            .crtc_mode_id = propId(node, self.crtc_id, drm.types.OBJECT_CRTC, "MODE_ID") orelse return false,
            .crtc_active = propId(node, self.crtc_id, drm.types.OBJECT_CRTC, "ACTIVE") orelse return false,
            .conn_crtc_id = propId(node, self.connector_id, drm.types.OBJECT_CONNECTOR, "CRTC_ID") orelse return false,
            .plane_fb_id = propId(node, plane_id, drm.types.OBJECT_PLANE, "FB_ID") orelse return false,
            .plane_crtc_id = propId(node, plane_id, drm.types.OBJECT_PLANE, "CRTC_ID") orelse return false,
            .plane_src_x = propId(node, plane_id, drm.types.OBJECT_PLANE, "SRC_X") orelse return false,
            .plane_src_y = propId(node, plane_id, drm.types.OBJECT_PLANE, "SRC_Y") orelse return false,
            .plane_src_w = propId(node, plane_id, drm.types.OBJECT_PLANE, "SRC_W") orelse return false,
            .plane_src_h = propId(node, plane_id, drm.types.OBJECT_PLANE, "SRC_H") orelse return false,
            .plane_crtc_x = propId(node, plane_id, drm.types.OBJECT_PLANE, "CRTC_X") orelse return false,
            .plane_crtc_y = propId(node, plane_id, drm.types.OBJECT_PLANE, "CRTC_Y") orelse return false,
            .plane_crtc_w = propId(node, plane_id, drm.types.OBJECT_PLANE, "CRTC_W") orelse return false,
            .plane_crtc_h = propId(node, plane_id, drm.types.OBJECT_PLANE, "CRTC_H") orelse return false,
        };
        return true;
    }

    /// Modeset `fb_id` via an atomic commit when atomic mode is enabled, else via
    /// legacy SETCRTC. A failed atomic commit disables atomic and retries legacy.
    pub fn setCrtcAtomic(self: *GpuDisplay, fb_id: u32) hal.Error!void {
        if (self.atomic == null) return self.setCrtc(fb_id);
        self.atomicModeset(fb_id) catch {
            note(self.log, "atomic modeset (falling back to SETCRTC)");
            self.disableAtomic();
            return self.setCrtc(fb_id);
        };
    }

    /// Present `fb_id` via an atomic page-flip commit when atomic mode is enabled,
    /// else via legacy PAGE_FLIP. A failed commit disables atomic and retries.
    pub fn pageFlipAtomic(self: *GpuDisplay, fb_id: u32) hal.Error!void {
        if (self.atomic == null) return self.pageFlip(fb_id);
        self.atomicPresent(fb_id) catch {
            note(self.log, "atomic present (falling back to PAGE_FLIP)");
            self.disableAtomic();
            return self.pageFlip(fb_id);
        };
    }

    fn atomicModeset(self: *GpuDisplay, fb_id: u32) hal.Error!void {
        const node = self.node;
        const blob = node.createBlob(std.mem.asBytes(&self.mode)) catch return error.DeviceLost;
        var req = buildModesetRequest(node.allocator, self.crtc_id, self.connector_id, &self.atomic.?, blob, fb_id, self.modeWidth(), self.modeHeight()) catch {
            node.destroyBlob(blob) catch {};
            return error.OutOfMemory;
        };
        defer req.deinit();
        node.atomicCommit(&req, drm.types.atomic.FLAG_ALLOW_MODESET, 0) catch {
            node.destroyBlob(blob) catch {};
            return error.DeviceLost;
        };
        // Commit took: free the previous MODE_ID blob and track the new one.
        if (self.atomic.?.mode_blob_id != 0) node.destroyBlob(self.atomic.?.mode_blob_id) catch {};
        self.atomic.?.mode_blob_id = blob;
    }

    fn atomicPresent(self: *GpuDisplay, fb_id: u32) hal.Error!void {
        const node = self.node;
        var req = drm.types.atomic.Request.init(node.allocator);
        defer req.deinit();
        req.addProperty(self.atomic.?.plane_id, self.atomic.?.plane_fb_id, fb_id) catch return error.OutOfMemory;
        node.atomicCommit(&req, drm.types.atomic.FLAG_PAGE_FLIP_EVENT | drm.types.atomic.FLAG_NONBLOCK, 0) catch return error.DeviceLost;
        // Drain one FLIP_COMPLETE event.
        var buf: [256]u8 align(8) = undefined;
        _ = linux.read(self.node.fd, &buf, buf.len);
    }

    fn disableAtomic(self: *GpuDisplay) void {
        if (self.atomic) |st| {
            if (st.mode_blob_id != 0) self.node.destroyBlob(st.mode_blob_id) catch {};
        }
        self.atomic = null;
    }

    pub fn deinit(self: *GpuDisplay) void {
        self.disableAtomic();
        restoreSavedCrtc(self.node, self.connector_id, self.saved_crtc);
        if (self.have_master) self.node.dropMaster() catch {};
        self.node.deinit();
        self.gpa.destroy(self);
    }
};

// Atomic helpers, all driven through drm.Node.

/// Enable the UNIVERSAL_PLANES + ATOMIC client caps. False if either is rejected.
fn enableAtomic(node: drm.Node) bool {
    node.setClientCap(drm.types.cap.UNIVERSAL_PLANES, 1) catch return false;
    node.setClientCap(drm.types.cap.ATOMIC, 1) catch return false;
    return true;
}

/// Resolve the property id named `name` on object `obj_id` (of `obj_type`).
fn propId(node: drm.Node, obj_id: u32, obj_type: u32, name: []const u8) ?u32 {
    var props = node.objGetProperties(obj_id, obj_type) catch return null;
    defer props.deinit(node.allocator);
    const ids = props.props() orelse return null;
    for (ids) |pid| {
        var gp = node.getProperty(pid) catch continue;
        defer gp.deinit(node.allocator);
        if (std.mem.eql(u8, cstr(&gp.name), name)) return pid;
    }
    return null;
}

/// The index of `crtc_id` in the resources' crtc array (the bit position used by
/// a plane's possible_crtcs mask).
fn crtcIndexOf(node: drm.Node, crtc_id: u32) ?u32 {
    var res = node.getModeCardRes() catch return null;
    defer res.deinit(node.allocator);
    const crtcs = res.crtcIds() orelse return null;
    for (crtcs, 0..) |c, i| if (c == crtc_id) return @intCast(i);
    return null;
}

/// The primary plane usable on the CRTC at `crtc_index`: a plane whose
/// possible_crtcs includes that index and whose "type" property is "Primary".
fn primaryPlaneFor(node: drm.Node, crtc_index: u32) ?u32 {
    var res = node.getPlaneRes() catch return null;
    defer res.deinit(node.allocator);
    const planes = res.planeIds() orelse return null;
    for (planes) |pid| {
        var pl = drm.types.ModeGetPlane{ .planeId = pid };
        pl.get(node.fd) catch continue;
        if (pl.possibleCrtcs & (@as(u32, 1) << @intCast(crtc_index)) == 0) continue;
        if (planeTypeIsPrimary(node, pid)) return pid;
    }
    return null;
}

/// True if the plane's "type" enum property currently resolves to "Primary".
fn planeTypeIsPrimary(node: drm.Node, plane_id: u32) bool {
    var props = node.objGetProperties(plane_id, drm.types.OBJECT_PLANE) catch return false;
    defer props.deinit(node.allocator);
    const ids = props.props() orelse return false;
    const vals = props.values() orelse return false;
    for (ids, vals) |pid, val| {
        var gp = node.getProperty(pid) catch continue;
        defer gp.deinit(node.allocator);
        if (!std.mem.eql(u8, cstr(&gp.name), "type")) continue;
        const ens = gp.enums() orelse return false;
        for (ens) |en| {
            if (en.value == val and std.mem.eql(u8, cstr(&en.name), "Primary")) return true;
        }
        return false;
    }
    return false;
}

/// Build the atomic modeset property set (CRTC MODE_ID/ACTIVE, connector CRTC_ID,
/// primary-plane FB_ID/CRTC_ID + full-surface src/dst rects). Src rects are 16.16
/// fixed point. The caller owns the returned Request (deinit) and the mode blob.
fn buildModesetRequest(
    alloc: std.mem.Allocator,
    crtc_id: u32,
    conn_id: u32,
    st: *const AtomicState,
    mode_blob_id: u32,
    fb_id: u32,
    w: u32,
    h: u32,
) !drm.types.atomic.Request {
    var req = drm.types.atomic.Request.init(alloc);
    errdefer req.deinit();
    try req.addProperty(crtc_id, st.crtc_mode_id, mode_blob_id);
    try req.addProperty(crtc_id, st.crtc_active, 1);
    try req.addProperty(conn_id, st.conn_crtc_id, crtc_id);
    try req.addProperty(st.plane_id, st.plane_fb_id, fb_id);
    try req.addProperty(st.plane_id, st.plane_crtc_id, crtc_id);
    try req.addProperty(st.plane_id, st.plane_src_x, 0);
    try req.addProperty(st.plane_id, st.plane_src_y, 0);
    try req.addProperty(st.plane_id, st.plane_src_w, fp16(w));
    try req.addProperty(st.plane_id, st.plane_src_h, fp16(h));
    try req.addProperty(st.plane_id, st.plane_crtc_x, 0);
    try req.addProperty(st.plane_id, st.plane_crtc_y, 0);
    try req.addProperty(st.plane_id, st.plane_crtc_w, w);
    try req.addProperty(st.plane_id, st.plane_crtc_h, h);
    return req;
}

/// Open the DRM primary node (default /dev/dri/card0) for the zero-copy GPU
/// scanout path: become DRM master, resolve a connected connector + mode + CRTC,
/// and return a GpuDisplay that owns the fd. The caller shares `deviceFd()` with
/// the virgl Linux transport (external_fd) so the virgl context + the scanout
/// resource live on this same fd, then ADDFB2's the resource + modesets it.
pub fn createGpuScanout(gpa: std.mem.Allocator) hal.Error!*GpuDisplay {
    return createGpuScanoutNode(gpa, "/dev/dri/card0");
}

pub fn createGpuScanoutNode(gpa: std.mem.Allocator, path: []const u8) hal.Error!*GpuDisplay {
    var zbuf: [80]u8 = undefined;
    const z = std.fmt.bufPrintZ(&zbuf, "{s}", .{path}) catch return error.InvalidArgument;
    const ofd = linux.open(z.ptr, .{ .ACCMODE = .RDWR, .CLOEXEC = true }, 0);
    if (std.posix.errno(ofd) != .SUCCESS) {
        dbg(.{}, "open card0 (gpu scanout)", std.posix.errno(ofd));
        return error.InitializationFailed;
    }
    const node = drm.Node{ .allocator = gpa, .fd = @intCast(ofd) };
    errdefer node.deinit();

    const have_master = if (node.setMaster()) true else |_| false;
    const pick = try resolveOutput(node);
    const saved = savedCrtcFor(node, pick.crtc_id);

    const d = gpa.create(GpuDisplay) catch return error.OutOfMemory;
    d.* = .{
        .gpa = gpa,
        .node = node,
        .have_master = have_master,
        .connector_id = pick.connector_id,
        .crtc_id = pick.crtc_id,
        .mode = pick.mode,
        .saved_crtc = saved,
    };
    return d;
}

// HdrDisplay: opens card0 and resolves connector/mode/crtc like GpuDisplay, and
// enumerates HDR properties (resolveHdrProps). On an HDR connector, createHdrScanout
// sets HDR_OUTPUT_METADATA (CTA-861.G blob) + Colorspace=BT2020_RGB, allocates a
// 10-bit XR30 dumb fb. On a non-HDR connector it degrades to 8-bit XRGB8888 without error.

pub const HdrScanout = struct {
    /// The KMS framebuffer id that was modeset (10-bit on HDR, 8-bit on SDR).
    fb_id: u32,
    /// True if the connector was HDR-capable and the HDR metadata+colorspace
    /// were applied with a 10-bit fb scanned out. False means it degraded to SDR.
    is_hdr: bool,
    /// The scanned-out framebuffer format on the wire.
    format: hal.Format,
    width: u32,
    height: u32,
};

pub const HdrDisplay = struct {
    gpa: std.mem.Allocator,
    node: drm.Node,
    have_master: bool,
    connector_id: u32,
    crtc_id: u32,
    mode: drm.types.ModeInfo,
    saved_crtc: drm.types.ModeGetCrtc,
    hdr_props: HdrProps,
    log: prism_log.Logger = .{},
    // The scanout fb resources, kept so deinit can tear them down.
    fb_id: u32 = 0,
    fb_handle: u32 = 0,
    fb_bytes: []u8 = &.{},
    hdr_blob_id: u32 = 0,

    pub fn modeWidth(self: *HdrDisplay) u32 {
        return self.mode.hdisplay;
    }
    pub fn modeHeight(self: *HdrDisplay) u32 {
        return self.mode.vdisplay;
    }

    /// Whether the resolved connector exposes HDR_OUTPUT_METADATA (the property a
    /// compositor needs to drive HDR). False on virtio-gpu and any SDR connector.
    pub fn outputIsHdrCapable(self: *HdrDisplay) bool {
        return self.hdr_props.isHdrCapable();
    }

    /// The compositor entry point. On an HDR connector: set HDR_OUTPUT_METADATA
    /// from `metadata` (CTA-861.G blob) + Colorspace=BT2020_RGB, create a 10-bit
    /// XRGB2101010 scanout fb, render a wide-gamut test gradient, and modeset it.
    /// On a non-HDR connector: log + degrade to an 8-bit SDR dumb fb (no error).
    /// Returns what was actually presented (is_hdr tells you which path ran).
    pub fn createHdrScanout(self: *HdrDisplay, metadata: HdrMetadata) hal.Error!HdrScanout {
        const w: u32 = self.mode.hdisplay;
        const h: u32 = self.mode.vdisplay;

        if (!self.outputIsHdrCapable()) {
            // Graceful degradation: virtio-gpu or any SDR connector. The QEMU
            // present path takes this branch.
            logLine("[drm-hdr] connector is not HDR-capable, presenting SDR\n");
            return self.scanoutSdr(w, h);
        }

        // 1. HDR_OUTPUT_METADATA: build the infoframe, upload it as a blob, set it.
        const meta = metadata.toOutputMetadata();
        const blob_id = self.node.createBlob(std.mem.asBytes(&meta)) catch return error.DeviceLost;
        self.node.objSetProperty(self.connector_id, drm.types.OBJECT_CONNECTOR, self.hdr_props.hdr_output_metadata_id, blob_id) catch {
            self.node.destroyBlob(blob_id) catch {};
            note(self.log, "OBJ_SETPROPERTY HDR_OUTPUT_METADATA");
            return error.DeviceLost;
        };
        self.hdr_blob_id = blob_id;

        // 2. Colorspace = BT2020_RGB (enum value resolved by name). Best-effort:
        //    some HDR connectors have HDR_OUTPUT_METADATA without a settable
        //    Colorspace enum. That should not fail the HDR present.
        if (self.hdr_props.colorspace_id != 0) {
            self.node.objSetProperty(self.connector_id, drm.types.OBJECT_CONNECTOR, self.hdr_props.colorspace_id, self.hdr_props.colorspace_bt2020_rgb) catch {};
        }

        // 3. 10-bit XR30 scanout fb + wide-gamut ramp + modeset.
        return self.scanoutHdr10(w, h);
    }

    /// 10-bit XRGB2101010 (XR30) scanout: CREATE_DUMB bpp=32, ADDFB2 with
    /// pixel_format=XR30, fill a 10-bit wide-gamut gradient, SETCRTC.
    fn scanoutHdr10(self: *HdrDisplay, w: u32, h: u32) hal.Error!HdrScanout {
        const cd = self.node.createDumb(w, h, 32) catch return error.DeviceLost;
        errdefer self.node.destroyDumb(cd.handle) catch {};

        var fb = drm.types.ModeFbCmd2{
            .width = w,
            .height = h,
            .pixelFormat = DRM_FORMAT_XRGB2101010,
            .flags = .{},
        };
        fb.handles[0] = cd.handle;
        fb.pitches[0] = cd.pitch;
        fb.offsets[0] = 0;
        self.node.addFb2(&fb) catch {
            note(self.log, "ADDFB2 XR30 (10-bit HDR scanout)");
            return error.DeviceLost;
        };
        errdefer self.node.rmFb(fb.fbId) catch {};

        const bytes = try self.mapDumb(cd.handle, cd.size);
        fillHdrRamp(bytes, w, h, cd.pitch);

        try self.modeset(fb.fbId);
        self.fb_id = fb.fbId;
        self.fb_handle = cd.handle;
        self.fb_bytes = bytes;
        return .{ .fb_id = fb.fbId, .is_hdr = true, .format = .rgb10x2, .width = w, .height = h };
    }

    /// SDR fallback: the original 8-bit XRGB8888 dumb-fb path (ADDFB depth24).
    fn scanoutSdr(self: *HdrDisplay, w: u32, h: u32) hal.Error!HdrScanout {
        const cd = self.node.createDumb(w, h, 32) catch return error.DeviceLost;
        errdefer self.node.destroyDumb(cd.handle) catch {};

        const fb = self.node.addFb(w, h, cd.pitch, 32, 24, cd.handle) catch return error.DeviceLost;
        errdefer self.node.rmFb(fb.fbId) catch {};

        const bytes = try self.mapDumb(cd.handle, cd.size);
        fillSdrRamp(bytes, w, h, cd.pitch);

        try self.modeset(fb.fbId);
        self.fb_id = fb.fbId;
        self.fb_handle = cd.handle;
        self.fb_bytes = bytes;
        return .{ .fb_id = fb.fbId, .is_hdr = false, .format = .bgra8_unorm, .width = w, .height = h };
    }

    fn mapDumb(self: *HdrDisplay, handle: u32, sz: u64) hal.Error![]u8 {
        const md = self.node.mapDumb(handle) catch return error.DeviceLost;
        const r = linux.mmap(null, sz, .{ .READ = true, .WRITE = true }, .{ .TYPE = .SHARED }, self.node.fd, @intCast(md.offset));
        if (std.posix.errno(r) != .SUCCESS) return error.DeviceLost;
        const ptr: [*]u8 = @ptrFromInt(r);
        return ptr[0..sz];
    }

    fn modeset(self: *HdrDisplay, fb_id: u32) hal.Error!void {
        setCrtcOn(self.node, self.connector_id, self.crtc_id, self.mode, fb_id) catch return error.DeviceLost;
    }

    pub fn deinit(self: *HdrDisplay) void {
        if (self.fb_bytes.len != 0) _ = linux.munmap(@ptrCast(self.fb_bytes.ptr), self.fb_bytes.len);
        if (self.fb_id != 0) self.node.rmFb(self.fb_id) catch {};
        if (self.fb_handle != 0) self.node.destroyDumb(self.fb_handle) catch {};
        if (self.hdr_blob_id != 0) self.node.destroyBlob(self.hdr_blob_id) catch {};
        restoreSavedCrtc(self.node, self.connector_id, self.saved_crtc);
        if (self.have_master) self.node.dropMaster() catch {};
        self.node.deinit();
        self.gpa.destroy(self);
    }
};

/// Fill a 10-bit XRGB2101010 framebuffer with a wide-gamut horizontal ramp: red
/// rises left->right, blue falls, green tracks the vertical. A real HDR display
/// shows it as a smooth wide-gamut wedge (banding-free vs 8-bit).
fn fillHdrRamp(bytes: []u8, w: u32, h: u32, pitch: u32) void {
    var y: u32 = 0;
    while (y < h) : (y += 1) {
        const row = bytes[y * pitch ..];
        var x: u32 = 0;
        while (x < w) : (x += 1) {
            const r8: u8 = @intCast((x * 255) / @max(w - 1, 1));
            const b8: u8 = @intCast(255 - (x * 255) / @max(w - 1, 1));
            const g8: u8 = @intCast((y * 255) / @max(h - 1, 1));
            const px = packXrgb2101010(r8, g8, b8);
            std.mem.writeInt(u32, row[x * 4 ..][0..4], px, .little);
        }
    }
}

/// SDR fallback ramp: the same wedge written as XRGB8888 (0x00RRGGBB LE).
fn fillSdrRamp(bytes: []u8, w: u32, h: u32, pitch: u32) void {
    var y: u32 = 0;
    while (y < h) : (y += 1) {
        const row = bytes[y * pitch ..];
        var x: u32 = 0;
        while (x < w) : (x += 1) {
            const r8: u32 = (x * 255) / @max(w - 1, 1);
            const b8: u32 = 255 - (x * 255) / @max(w - 1, 1);
            const g8: u32 = (y * 255) / @max(h - 1, 1);
            const px = (r8 << 16) | (g8 << 8) | b8;
            std.mem.writeInt(u32, row[x * 4 ..][0..4], px, .little);
        }
    }
}

fn logLine(s: []const u8) void {
    if (builtin.is_test) return;
    _ = linux.write(2, s.ptr, s.len);
}

/// Open the DRM primary node (default /dev/dri/card0) for the HDR present path:
/// become master, resolve a connected connector + mode + CRTC, and enumerate the
/// connector's HDR properties. The returned HdrDisplay degrades to SDR cleanly if
/// the connector is not HDR-capable. The caller drives it via createHdrScanout.
pub fn createHdrDisplay(gpa: std.mem.Allocator) hal.Error!*HdrDisplay {
    return createHdrDisplayNode(gpa, "/dev/dri/card0");
}

pub fn createHdrDisplayNode(gpa: std.mem.Allocator, path: []const u8) hal.Error!*HdrDisplay {
    var zbuf: [80]u8 = undefined;
    const z = std.fmt.bufPrintZ(&zbuf, "{s}", .{path}) catch return error.InvalidArgument;
    const ofd = linux.open(z.ptr, .{ .ACCMODE = .RDWR, .CLOEXEC = true }, 0);
    if (std.posix.errno(ofd) != .SUCCESS) {
        dbg(.{}, "open card0 (hdr)", std.posix.errno(ofd));
        return error.InitializationFailed;
    }
    const node = drm.Node{ .allocator = gpa, .fd = @intCast(ofd) };
    errdefer node.deinit();

    const have_master = if (node.setMaster()) true else |_| false;
    const pick = try resolveOutput(node);
    const hdr_props = resolveHdrProps(node, pick.connector_id);
    const saved = savedCrtcFor(node, pick.crtc_id);

    const d = gpa.create(HdrDisplay) catch return error.OutOfMemory;
    d.* = .{
        .gpa = gpa,
        .node = node,
        .have_master = have_master,
        .connector_id = pick.connector_id,
        .crtc_id = pick.crtc_id,
        .mode = pick.mode,
        .saved_crtc = saved,
        .hdr_props = hdr_props,
    };
    return d;
}

/// GETRESOURCES, then find a connected connector with at least one mode, take its
/// preferred (or first) mode, and a CRTC reachable through its encoder.
fn resolveOutput(node: drm.Node) hal.Error!Output {
    var res = node.getModeCardRes() catch return error.InitializationFailed;
    defer res.deinit(node.allocator);
    if (res.countConnectors == 0 or res.countCrtcs == 0)
        return error.InitializationFailed;

    const crtcs = res.crtcIds() orelse return error.InitializationFailed;
    const conns = res.connectorIds() orelse return error.InitializationFailed;

    for (conns) |conn_id| {
        const out = pickConnector(node, conn_id, crtcs) catch continue orelse continue;
        return out;
    }
    return error.InitializationFailed;
}

/// Probe one connector. If connected with a mode, find a reachable CRTC via its
/// current encoder (or any of its encoders' possible_crtcs) and return the pick.
fn pickConnector(node: drm.Node, conn_id: u32, crtcs: []const u32) hal.Error!?Output {
    var c = node.getConnector(conn_id) catch return null;
    defer c.deinit(node.allocator);
    if (c.connection != DRM_MODE_CONNECTED or c.countModes == 0)
        return null;

    const modes = c.modes() orelse return null;

    // Pick the preferred mode, else the first. The first is the highest-res by
    // convention. Both work for our scanout test.
    var mode = modes[0];
    for (modes) |m| {
        if (m.type & DRM_MODE_TYPE_PREFERRED != 0) {
            mode = m;
            break;
        }
    }

    // Find a CRTC. Try the connector's current encoder first, then every encoder
    // it lists, matching against possible_crtcs (a bitmask over the resources'
    // crtc array order).
    if (crtcForEncoder(node, c.encoderId, crtcs)) |id| return Output{ .connector_id = conn_id, .crtc_id = id, .mode = mode };
    if (c.encoderIds()) |encs| {
        for (encs) |enc_id| {
            if (crtcForEncoder(node, enc_id, crtcs)) |id|
                return Output{ .connector_id = conn_id, .crtc_id = id, .mode = mode };
        }
    }
    return null;
}

/// Resolve an encoder to a usable CRTC id: its current crtc_id if set, else the
/// first crtc in possible_crtcs.
fn crtcForEncoder(node: drm.Node, encoder_id: u32, crtcs: []const u32) ?u32 {
    if (encoder_id == 0) return null;
    const e = node.getEncoder(encoder_id) catch return null;
    if (e.crtcId != 0) return e.crtcId;
    var i: usize = 0;
    while (i < crtcs.len) : (i += 1) {
        if (e.possibleCrtcs & (@as(u32, 1) << @intCast(i)) != 0) return crtcs[i];
    }
    return null;
}

// dev_t major/minor + sysfs driver lookup. Used by the Wayland-client path to
// match a compositor's preferred device to a Prism driver.

/// Major/minor from a system dev_t (glibc encoding, which matches the Wayland
/// dmabuf `main_device` and a stat's st_rdev).
pub fn major(dev: u64) u64 {
    return ((dev >> 8) & 0xfff) | ((dev >> 32) & ~@as(u64, 0xfff));
}
pub fn minor(dev: u64) u64 {
    return (dev & 0xff) | ((dev >> 12) & ~@as(u64, 0xff));
}
pub fn makedev(maj: u64, min: u64) u64 {
    return ((maj & 0xfff) << 8) | (min & 0xff) | ((maj & ~@as(u64, 0xfff)) << 32) | ((min & ~@as(u64, 0xff)) << 12);
}

/// The kernel driver backing the DRM device with the given dev_t (e.g.
/// "nvidia", "amdgpu", "i915"), resolved via sysfs. Returns null if not found.
/// Used to match a compositor's preferred render device to a Prism driver.
/// The returned slice points into `buf`. 64 bytes is plenty.
pub fn driverForDev(dev: u64, buf: []u8) ?[]const u8 {
    var path_buf: [80]u8 = undefined;
    const link = std.fmt.bufPrintZ(&path_buf, "/sys/dev/char/{d}:{d}/device/driver", .{ major(dev), minor(dev) }) catch return null;
    const rc = linux.readlinkat(linux.AT.FDCWD, link.ptr, buf.ptr, buf.len);
    if (@as(isize, @bitCast(rc)) < 0) return null; // -errno (e.g. ENOENT: no such device)
    return std.fs.path.basename(buf[0..rc]); // ".../drivers/nvidia" -> "nvidia"
}

test "driverForDev resolves the first DRM render node to its kernel driver (skips if none)" {
    // DRM major is 226, render nodes start at minor 128. Resolves renderD128.
    var buf: [64]u8 = undefined;
    const name = driverForDev(makedev(226, 128), &buf) orelse return error.SkipZigTest;
    try std.testing.expect(name.len > 0);
    try std.testing.expect(std.mem.indexOfScalar(u8, name, '/') == null); // a name, not a path
}

test "drm ioctl encodings match the DRM uAPI" {
    // The resource getters, dumb create, master, and legacy ADDFB all route
    // through drm.types now, so assert their package .req values are the encoding
    // the kernel wants (DRM_IO(WR)('d', nr, struct)).
    const res_req: u32 = (3 << 30) | (@as(u32, @sizeOf(drm.types.ModeCardRes)) << 16) | (0x64 << 8) | 0xA0;
    try std.testing.expectEqual(res_req, drm.types.ModeCardRes.req);
    const dumb_req: u32 = (3 << 30) | (@as(u32, @sizeOf(drm.types.DumbCreate)) << 16) | (0x64 << 8) | 0xB2;
    try std.testing.expectEqual(dumb_req, drm.types.DumbCreate.req);
    // SET_MASTER is DRM_IO('d', 0x1e): dir 0, size 0.
    const set_master: u32 = (0x64 << 8) | 0x1e;
    try std.testing.expectEqual(set_master, drm.types.master.SET_MASTER);
    // Legacy ADDFB is DRM_IOWR('d', 0xAE, ModeFbCmd), a 28-byte depth-bearing cmd.
    const addfb: u32 = (3 << 30) | (@as(u32, @sizeOf(drm.types.ModeFbCmd)) << 16) | (0x64 << 8) | 0xAE;
    try std.testing.expectEqual(addfb, drm.types.ModeFbCmd.reqAdd);
    try std.testing.expectEqual(@as(usize, 28), @sizeOf(drm.types.ModeFbCmd));
    // Struct sizes the kernel checks via the embedded size field.
    try std.testing.expectEqual(@as(usize, 68), @sizeOf(drm.types.ModeInfo));
    // ModeGetCrtc: u64 + 7*u32 (=36) header, then the embedded modeinfo.
    try std.testing.expectEqual(@as(usize, 36), @sizeOf(drm.types.ModeGetCrtc) - @sizeOf(drm.types.ModeInfo));
}

test "drm display open fails gracefully without a card (no panic)" {
    const gpa = std.testing.allocator;
    // A path that does not exist must yield InitializationFailed, not a crash.
    const r = createNode(gpa, "/dev/dri/prism-nonexistent-card");
    try std.testing.expectError(error.InitializationFailed, r);
}

test "hdr property ioctl encodings match the DRM uAPI" {
    // GETPROPERTY + the object-property + blob ioctls all come from drm.types now.
    // Assert each package .req is DRM_IOWR('d', nr, struct), from drm_mode.h.
    const getprop: u32 = (3 << 30) | (@as(u32, @sizeOf(drm.types.ModeGetProperty)) << 16) | (0x64 << 8) | 0xAA;
    try std.testing.expectEqual(getprop, drm.types.ModeGetProperty.req);
    const objget: u32 = (3 << 30) | (@as(u32, @sizeOf(drm.types.ModeObjGetProperties)) << 16) | (0x64 << 8) | 0xB9;
    try std.testing.expectEqual(objget, drm.types.ModeObjGetProperties.req);
    const objset: u32 = (3 << 30) | (@as(u32, @sizeOf(drm.types.ModeObjSetProperty)) << 16) | (0x64 << 8) | 0xBA;
    try std.testing.expectEqual(objset, drm.types.ModeObjSetProperty.req);
    const createblob: u32 = (3 << 30) | (@as(u32, @sizeOf(drm.types.ModeCreateBlob)) << 16) | (0x64 << 8) | 0xBD;
    try std.testing.expectEqual(createblob, drm.types.ModeCreateBlob.req);
    const destroyblob: u32 = (3 << 30) | (@as(u32, @sizeOf(drm.types.ModeDestroyBlob)) << 16) | (0x64 << 8) | 0xBE;
    try std.testing.expectEqual(destroyblob, drm.types.ModeDestroyBlob.req);
}

test "atomic src rect uses 16.16 fixed point" {
    try std.testing.expectEqual(@as(u64, 0), fp16(0));
    try std.testing.expectEqual(@as(u64, 0x10000), fp16(1));
    try std.testing.expectEqual(@as(u64, 1920) << 16, fp16(1920));
    try std.testing.expectEqual(@as(u64, 0x07800000), fp16(1920));
    try std.testing.expectEqual(@as(u64, 0x04380000), fp16(1080));
}

test "atomic modeset request groups props per object with 16.16 src rects" {
    const a = std.testing.allocator;
    // Fake, distinct property ids so the grouping and values are easy to read.
    const st = AtomicState{
        .plane_id = 30,
        .crtc_mode_id = 1,
        .crtc_active = 2,
        .conn_crtc_id = 3,
        .plane_fb_id = 4,
        .plane_crtc_id = 5,
        .plane_src_x = 6,
        .plane_src_y = 7,
        .plane_src_w = 8,
        .plane_src_h = 9,
        .plane_crtc_x = 10,
        .plane_crtc_y = 11,
        .plane_crtc_w = 12,
        .plane_crtc_h = 13,
    };
    var req = try buildModesetRequest(a, 100, 200, &st, 55, 42, 1920, 1080);
    defer req.deinit();

    // Three grouped objects: crtc(100) with 2 props, connector(200) with 1,
    // plane(30) with 10 (fb, crtc, src x/y/w/h, crtc x/y/w/h).
    try std.testing.expectEqual(@as(usize, 3), req.groups.items.len);

    var flat = try req.flatten(a);
    defer flat.deinit();

    try std.testing.expectEqualSlices(u32, &.{ 100, 200, 30 }, flat.objs);
    try std.testing.expectEqualSlices(u32, &.{ 2, 1, 10 }, flat.countProps);
    try std.testing.expectEqualSlices(u32, &.{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13 }, flat.props);
    // MODE_ID blob (55), ACTIVE (1), connector CRTC_ID (100), plane FB_ID (42),
    // plane CRTC_ID (100), src x/y (0,0), src w/h in 16.16 (1920<<16, 1080<<16),
    // crtc x/y (0,0), crtc w/h (1920, 1080).
    try std.testing.expectEqualSlices(u64, &.{
        55,        1,        100, 42, 100,  0,    0,
        125829120, 70778880, 0,   0,  1920, 1080,
    }, flat.values);
}

test "hdr_output_metadata struct layout matches the kernel uAPI (CTA-861.G)" {
    // Verified against linux-headers 6.16.7 drm_mode.h with a C offsetof probe.
    try std.testing.expectEqual(@as(usize, 26), @sizeOf(hdr_metadata_infoframe));
    try std.testing.expectEqual(@as(usize, 32), @sizeOf(hdr_output_metadata));
    try std.testing.expectEqual(@as(usize, 0), @offsetOf(hdr_metadata_infoframe, "eotf"));
    try std.testing.expectEqual(@as(usize, 1), @offsetOf(hdr_metadata_infoframe, "metadata_type"));
    try std.testing.expectEqual(@as(usize, 2), @offsetOf(hdr_metadata_infoframe, "display_primaries"));
    try std.testing.expectEqual(@as(usize, 14), @offsetOf(hdr_metadata_infoframe, "white_point"));
    try std.testing.expectEqual(@as(usize, 18), @offsetOf(hdr_metadata_infoframe, "max_display_mastering_luminance"));
    try std.testing.expectEqual(@as(usize, 20), @offsetOf(hdr_metadata_infoframe, "min_display_mastering_luminance"));
    try std.testing.expectEqual(@as(usize, 22), @offsetOf(hdr_metadata_infoframe, "max_cll"));
    try std.testing.expectEqual(@as(usize, 24), @offsetOf(hdr_metadata_infoframe, "max_fall"));
    // The infoframe union sits at offset 4 inside hdr_output_metadata.
    try std.testing.expectEqual(@as(usize, 4), @offsetOf(hdr_output_metadata, "hdmi_metadata_type1"));
}

test "HdrMetadata -> hdr_output_metadata infoframe byte layout" {
    // Hand-built metadata with distinct field values. Assert each lands at the
    // right byte offset in the raw blob the kernel will receive.
    const md = HdrMetadata{
        .eotf = .pq,
        .display_primaries = .{
            .{ .x = 0x1111, .y = 0x2222 },
            .{ .x = 0x3333, .y = 0x4444 },
            .{ .x = 0x5555, .y = 0x6666 },
        },
        .white_point = .{ .x = 0x7777, .y = 0x8888 },
        .max_display_mastering_luminance = 0x99AA,
        .min_display_mastering_luminance = 0xBBCC,
        .max_cll = 0xDDEE,
        .max_fall = 0x0F1F,
    };
    const out = md.toOutputMetadata();
    const raw = std.mem.asBytes(&out);
    try std.testing.expectEqual(@as(usize, 32), raw.len);

    // outer metadata_type (u32 LE) == HDMI_STATIC_METADATA_TYPE1 (0).
    try std.testing.expectEqual(@as(u32, 0), std.mem.readInt(u32, raw[0..4], .little));
    // infoframe at offset 4: eotf == SMPTE ST 2084 (PQ) == 2.
    try std.testing.expectEqual(@as(u8, HDMI_EOTF_SMPTE_ST2084), raw[4]);
    try std.testing.expectEqual(@as(u8, 2), raw[4]);
    // infoframe.metadata_type byte == 0.
    try std.testing.expectEqual(@as(u8, 0), raw[5]);
    // display_primaries[0].x at offset 4+2 == 6 (u16 LE).
    try std.testing.expectEqual(@as(u16, 0x1111), std.mem.readInt(u16, raw[6..8], .little));
    try std.testing.expectEqual(@as(u16, 0x2222), std.mem.readInt(u16, raw[8..10], .little));
    try std.testing.expectEqual(@as(u16, 0x3333), std.mem.readInt(u16, raw[10..12], .little));
    try std.testing.expectEqual(@as(u16, 0x4444), std.mem.readInt(u16, raw[12..14], .little));
    try std.testing.expectEqual(@as(u16, 0x5555), std.mem.readInt(u16, raw[14..16], .little));
    try std.testing.expectEqual(@as(u16, 0x6666), std.mem.readInt(u16, raw[16..18], .little));
    // white_point at offset 4+14 == 18.
    try std.testing.expectEqual(@as(u16, 0x7777), std.mem.readInt(u16, raw[18..20], .little));
    try std.testing.expectEqual(@as(u16, 0x8888), std.mem.readInt(u16, raw[20..22], .little));
    // luminance + content light levels at 4+18..4+24 == 22..28.
    try std.testing.expectEqual(@as(u16, 0x99AA), std.mem.readInt(u16, raw[22..24], .little));
    try std.testing.expectEqual(@as(u16, 0xBBCC), std.mem.readInt(u16, raw[24..26], .little));
    try std.testing.expectEqual(@as(u16, 0xDDEE), std.mem.readInt(u16, raw[26..28], .little));
    try std.testing.expectEqual(@as(u16, 0x0F1F), std.mem.readInt(u16, raw[28..30], .little));
}

test "pqRec2020 builder: BT.2020 primaries + D65 + PQ EOTF + luminance encoding" {
    // The common HDR10 case: 0.005 .. 1000 nits, MaxCLL 1000, MaxFALL 400.
    const md = HdrMetadata.pqRec2020(0.005, 1000.0, 1000, 400);
    try std.testing.expectEqual(Eotf.pq, md.eotf);

    // BT.2020 chromaticities in 0.00002 units (xy * 50000):
    //   R 0.708/0.292 -> 35400/14600, G 0.170/0.797 -> 8500/39850,
    //   B 0.131/0.046 -> 6550/2300, white D65 0.3127/0.3290 -> 15635/16450.
    try std.testing.expectEqual(@as(u16, 35400), md.display_primaries[0].x);
    try std.testing.expectEqual(@as(u16, 14600), md.display_primaries[0].y);
    try std.testing.expectEqual(@as(u16, 8500), md.display_primaries[1].x);
    try std.testing.expectEqual(@as(u16, 39850), md.display_primaries[1].y);
    try std.testing.expectEqual(@as(u16, 6550), md.display_primaries[2].x);
    try std.testing.expectEqual(@as(u16, 2300), md.display_primaries[2].y);
    try std.testing.expectEqual(@as(u16, 15635), md.white_point.x);
    try std.testing.expectEqual(@as(u16, 16450), md.white_point.y);

    // max luminance: 1 cd/m2 per unit -> 1000. min: 0.0001 cd/m2 per unit ->
    // 0.005 * 10000 == 50.
    try std.testing.expectEqual(@as(u16, 1000), md.max_display_mastering_luminance);
    try std.testing.expectEqual(@as(u16, 50), md.min_display_mastering_luminance);
    try std.testing.expectEqual(@as(u16, 1000), md.max_cll);
    try std.testing.expectEqual(@as(u16, 400), md.max_fall);

    // And the EOTF byte in the blob is the PQ code.
    const out = md.toOutputMetadata();
    try std.testing.expectEqual(HDMI_EOTF_SMPTE_ST2084, out.hdmi_metadata_type1.eotf);
}

test "XRGB2101010 (XR30) pixel packing" {
    // Full white: every channel 0xFF -> 0x3FF (1023), x bits zero.
    //   (1023<<20)|(1023<<10)|1023 == 0x3FFFFFFF.
    try std.testing.expectEqual(@as(u32, 0x3FFFFFFF), packXrgb2101010(0xFF, 0xFF, 0xFF));
    // Pure red: R=0x3FF in bits 29:20 -> 0x3FF00000.
    try std.testing.expectEqual(@as(u32, 0x3FF00000), packXrgb2101010(0xFF, 0, 0));
    // Pure green: bits 19:10 -> 0x000FFC00.
    try std.testing.expectEqual(@as(u32, 0x000FFC00), packXrgb2101010(0, 0xFF, 0));
    // Pure blue: bits 9:0 -> 0x000003FF.
    try std.testing.expectEqual(@as(u32, 0x000003FF), packXrgb2101010(0, 0, 0xFF));
    // Black: all zero, and the top 2 (x) bits are never set.
    try std.testing.expectEqual(@as(u32, 0), packXrgb2101010(0, 0, 0));
    // 8->10 expansion keeps mid grays monotonic: 0x80 -> (0x80<<2)|(0x80>>6).
    try std.testing.expectEqual(@as(u32, (0x80 << 2) | (0x80 >> 6)), packXrgb2101010(0, 0, 0x80) & 0x3FF);
}

test "DRM_FORMAT_XRGB2101010 fourcc is 'XR30'" {
    // drm_fourcc.h: DRM_FORMAT_XRGB2101010 == fourcc_code('X','R','3','0').
    try std.testing.expectEqual(@as(u32, fourcc('X', 'R', '3', '0')), DRM_FORMAT_XRGB2101010);
    // sanity: byte order is little (X==0x58 in the low byte).
    try std.testing.expectEqual(@as(u8, 'X'), @as(u8, @truncate(DRM_FORMAT_XRGB2101010)));
}

test "hdr display open fails gracefully without a card (no panic)" {
    const gpa = std.testing.allocator;
    const r = createHdrDisplayNode(gpa, "/dev/dri/prism-nonexistent-card");
    try std.testing.expectError(error.InitializationFailed, r);
}

test "non-HDR HdrProps reports not HDR-capable" {
    const p = HdrProps{};
    try std.testing.expect(!p.isHdrCapable());
    const p2 = HdrProps{ .hdr_output_metadata_id = 42 };
    try std.testing.expect(p2.isHdrCapable());
}
