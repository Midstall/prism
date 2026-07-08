const nvidia = @import("nvidia");
const hal = @import("../../hal.zig");

/// A GPU-backed HAL resource: a chunk of device memory with a GPU virtual
/// address and an optional CPU mapping (created lazily by mapResource). Backed
/// by the from-scratch RM driver (subproject/nvidia). Buffers and images share
/// this. An image just carries its dimensions and format.
pub const Resource = struct {
    kind: hal.ResourceKind,
    mem: nvidia.Memory,
    gpu_va: u64,
    /// The GPU VA SPAN reserved for this resource (alignment-rounded, >= size). The
    /// device returns it to its VA free list on destroy so the address range is reused
    /// (a long-lived device otherwise marches its bump pointer past the VA space limit).
    va_span: u64 = 0,
    /// Logical byte size requested by the consumer. The physical allocation is
    /// page-rounded (often larger), so mapResource returns a slice of this size.
    size: u64,
    mapping: ?nvidia.Mapping = null,
    /// The GPU VA binding (NV01_MEMORY_VIRTUAL) for this resource's memory. It must
    /// be released on destroy (unmapFromGpu) or the kernel's VA/page-table resources
    /// leak and later allocations fail (NV status 0x36 insufficient resources).
    /// Pooled (sysmem) resources hand it back to the chunk pool. VRAM surfaces
    /// (depth/color block-linear) free it directly.
    gpu: ?nvidia.GpuMapping = null,
    // image-only metadata (zero for buffers)
    width: u32 = 0,
    height: u32 = 0,
    format: hal.Format = .rgba8_unorm,
    /// A ZETA depth surface allocated in VRAM (not the system_wc pool): the
    /// fixed-function depth/Z-cull/Z-compression hardware requires the depth
    /// surface in device-local memory. Freed directly (not pooled) on destroy.
    is_depth_vram: bool = false,
    /// A block-linear (GOB-tiled) color render target. Required so the RT can be
    /// paired with a ZETA depth surface (a LINEAR color target + a selected ZETA
    /// faults Xid 69 / 0x9c). The GPU writes it tiled. CPU readback/present
    /// de-swizzles via nvidia.graphics.blColorPixelOffset. Like the ZETA it lives
    /// in VRAM with big pages and is freed directly (not pooled).
    block_linear: bool = false,
    /// Lazily-allocated linear de-swizzle scratch for a block_linear RT, so
    /// mapResource can hand back a normal row-major image. Owned by the resource.
    /// Freed on destroy. width*height*4 bytes.
    linear_copy: ?[]u8 = null,
    /// Lazily-allocated CACHED scratch holding a bulk sequential copy of the tiled
    /// (block-linear) GPU surface, so the per-pixel de-swizzle gather reads cached
    /// memory instead of uncached/write-combined VRAM (the live-present hot path).
    /// Owned by the resource. Freed on destroy.
    bl_scratch: ?[]u8 = null,
    /// A sampled texture (combined-image-sampler source). Its GPU memory is
    /// block-linear (the TIC describes block-linear tiling), but the CPU/ICD fills it
    /// linearly via vkCmdCopyBufferToImage. mapResource hands back a linear staging
    /// buffer (`linear_copy`). The context GOB-swizzles that staging into the GPU
    /// memory when it builds the texture's TIC at draw time. `tex_dirty` marks staging
    /// that has been written and not yet swizzled to the GPU.
    sampled: bool = false,
    tex_dirty: bool = false,
    /// Mip level count (1 = base only). >1 = a glGenerateMipmap'd sampled texture whose GPU memory
    /// holds a block-linear mip chain (blMipLevelOffset per level) and whose staging holds the
    /// tightly-packed chain. uploadTexture GOB-swizzles each level, and the TIC sets MAX_MIP_LEVEL.
    mip_levels: u8 = 1,
    /// 3D texture Z-slice count (1 = a 2D image). >1 = a sampler3D volume whose block-linear
    /// backing stacks `depth` 2D block-linear slices (GOBS_PER_BLOCK_DEPTH=0). uploadTexture
    /// GOB-swizzles each slice, and the TIC sets TEXTURE_TYPE=THREE_D + DEPTH_MINUS_ONE.
    depth: u32 = 1,
    /// CUBEMAP: 6 block-linear faces stacked like a depth-6 3D texture (GL order). The TIC uses
    /// TEXTURE_TYPE=CUBEMAP + DEPTH_MINUS_ONE=5 and the TEX a cube dim (HW face-selection).
    is_cube: bool = false,
    /// 2D ARRAY (sampler2DArray): `depth` independent layers, same stacked block-linear backing as
    /// a 3D volume, but the TIC uses TEXTURE_TYPE=TWO_D_ARRAY and the TEX an Array2D dim (a raw
    /// layer index, no cross-layer filtering). Distinguishes the two `depth > 1` image kinds.
    is_array: bool = false,
    /// MSAA sample count (from ImageDesc.samples). >1 means this color target is
    /// supersampled: its block-linear backing is allocated at ssScale(samples) times the
    /// logical width/height, the GPU renders the whole enlarged image (SSAA is transparent
    /// to the shaders), and a resolve box-downsamples the block back to a single sample.
    /// `width`/`height` stay the logical (resolved) dimensions.
    samples: u8 = 1,
};

/// Supersample scale (x, y) for a HAL MSAA sample count: 4 -> 2x2, 2 -> 2x1, else 1x1.
/// The product is the number of source samples the resolve box-averages per output pixel.
pub fn ssScale(samples: u8) [2]u32 {
    return switch (samples) {
        4 => .{ 2, 2 },
        2 => .{ 2, 1 },
        else => .{ 1, 1 },
    };
}

/// Map a HAL pixel format to the nvidia sampled-texture TIC format (drives the TIC's
/// COMPONENTS / DATA_TYPE / sRGB bit and the per-texel storage size). Anything not sRGB /
/// float samples as the plain 8-bit rgba8_unorm path.
pub fn ticFormat(fmt: hal.Format) nvidia.graphics.TicFormat {
    return switch (fmt) {
        .rgba8_srgb => .rgba8_srgb,
        .rgba16_float => .rgba16_float,
        .r32g32b32a32_float => .rgba32_float,
        // A sampled depth texture (sampler2DShadow source): ZF32 = one 32-bit float in R, the
        // format the HW DEPTH_COMPARE engages on. (A depth32_float used as a render target goes
        // to the ZETA path in device.createResource and never reaches ticFormat.)
        .depth32_float => .zf32,
        else => .rgba8_unorm,
    };
}

/// The SET_COLOR_TARGET_FORMAT_V value for a hal color-target format. Float RTs render at full
/// precision. Every 8-bit format uses A8R8G8B8 (the historical default).
pub fn colorTargetFormat(fmt: hal.Format) u32 {
    return switch (fmt) {
        .rgba16_float => nvidia.graphics.CT_FORMAT_RF16_GF16_BF16_AF16,
        .r32g32b32a32_float => nvidia.graphics.CT_FORMAT_RF32_GF32_BF32_AF32,
        else => nvidia.graphics.CT_FORMAT_A8R8G8B8,
    };
}

/// Bytes per pixel of a color render target's format (drives the block-linear footprint + the
/// de-swizzle stride). 4 for every 8-bit format, 8 for rgba16f, 16 for rgba32f.
pub fn colorTargetBpp(fmt: hal.Format) u32 {
    return switch (fmt) {
        .rgba16_float => 8,
        .r32g32b32a32_float => 16,
        else => 4,
    };
}
