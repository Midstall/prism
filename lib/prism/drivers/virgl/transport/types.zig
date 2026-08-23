//! Shared OS-agnostic types for the virgl transport seam.
//! Both the freestanding (Conduit *Virtio) and Linux (DRM uAPI) transports share these
//! so the virgl Device, Context, and encoder stay OS-agnostic.

const builtin = @import("builtin");

/// A 3D transfer box (x,y,z + w,h,d). Identical layout to conduit's GpuBox and
/// the kernel's `struct drm_virtgpu_3d_box`.
pub const Box = extern struct { x: u32, y: u32, z: u32, w: u32, h: u32, d: u32 };

/// Transport errors. InitializationFailed = a setup op failed (open/context/resource create).
/// DeviceLost = a submit/transfer failed on a reachable device.
/// Unsupported = operation not supported for the given parameters (e.g. non-4-byte scanout format).
pub const Error = error{ InitializationFailed, DeviceLost, OutOfMemory, Unsupported };

/// A virgl 3D resource as the transport tracks it: a host resource id plus the
/// guest backing the CPU reads/writes. Freestanding: caller-allocated DMA-coherent
/// memory attached to the host resource. Linux: mmap of the resource's BO.
pub const Resource = struct {
    /// Host-visible virgl resource id used in the command stream (the value the
    /// DRAW_VBO / SET_FRAMEBUFFER surface reference). On Linux this is the
    /// kernel-returned res_handle. On freestanding it is the id the driver chose.
    res_id: u32,
    /// The guest backing bytes (the vertex data the CPU writes, or the RT the GPU
    /// renders into and the CPU reads back).
    bytes: []u8,
    /// Linux-only: the GEM bo handle for transfer/map/execbuffer bo lists. Unused
    /// (0) on freestanding.
    bo_handle: u32 = 0,
    /// Whether this resource is the vertex buffer (vs. the render-target image).
    is_vertex: bool = false,
    width: u32 = 0,
    height: u32 = 0,
    /// Bytes per pixel of an image resource (4 for the SDR B8G8R8X8 / 10-bit
    /// formats, 8 for the fp16 R16G16B16A16_FLOAT HDR render target). Drives the
    /// resource size + the TRANSFER stride. Unused (0) for buffers.
    bpp: u32 = 4,
};

/// virgl/Gallium encoding constants used to build resources + the command stream.
/// These are the host virglrenderer protocol values, identical on both transports
/// (the freestanding transport's conduit namespace uses the same numbers).
pub const enc = struct {
    pub const FORMAT_B8G8R8X8_UNORM: u32 = 2;
    pub const FORMAT_B8G8R8A8_UNORM: u32 = 1;
    pub const FORMAT_R32G32_FLOAT: u32 = 29;
    pub const FORMAT_R32G32B32A32_FLOAT: u32 = 31;
    pub const FORMAT_R8G8B8A8_UNORM: u32 = 67;
    // HDR / 10-bit formats (VIRGL_FORMAT_* ordinals, virgl_hw.h).
    pub const FORMAT_R16G16B16A16_FLOAT: u32 = 94;
    pub const FORMAT_R10G10B10A2_UNORM: u32 = 8;
    pub const FORMAT_B10G10R10X2_UNORM: u32 = 233;
    pub const BIND_DEPTH_STENCIL: u32 = 1 << 0;
    pub const BIND_RENDER_TARGET: u32 = 1 << 1;
    pub const BIND_SAMPLER_VIEW: u32 = 1 << 3;
    pub const BIND_VERTEX_BUFFER: u32 = 1 << 4;
    pub const BIND_SCANOUT: u32 = 1 << 18;
    pub const TEXTURE_2D: u32 = 2;
    pub const BUFFER: u32 = 0;
    /// True if a virgl pipe format is a depth/stencil format (ordinals 16..23),
    /// so createImage binds it DEPTH_STENCIL rather than a color/scanout target.
    pub fn isDepthFormat(f: u32) bool {
        return f >= 16 and f <= 23;
    }
};
