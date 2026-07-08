const std = @import("std");

pub const Error = @import("error.zig").Error;

pub const Color = struct {
    r: f32 = 0,
    g: f32 = 0,
    b: f32 = 0,
    a: f32 = 1,
};

pub const Format = enum {
    rgba8_unorm,
    // 8-bit sRGB-encoded RGBA. RGB channels carry the sRGB EOTF. Alpha is linear.
    // Sampled-texture format: the sampler decodes sRGB -> linear on read so filtering
    // happens in linear space. Same 4-byte storage as rgba8_unorm.
    rgba8_srgb,
    bgra8_unorm,
    r8_unorm,
    // Two-channel 8-bit unorm (R,G), e.g. a normal-map / mask texture.
    r8g8_unorm,
    depth32_float,
    // HDR formats. rgba16_float is the scRGB / linear-HDR workhorse (16-bit IEEE
    // half per channel, holds values well outside the SDR 0..1 range losslessly).
    // rgb10a2 / rgb10x2 are the 10-bit-per-channel + 2-bit-alpha HDR10 scanout
    // formats (10-bit unorm gives the extra precision HDR10 transfer needs).
    rgba16_float,
    rgb10a2,
    rgb10x2,
    // Single/dual-channel floating-point render-target + sampled formats. r16_float /
    // r32_float carry one IEEE half / single per pixel (a value far outside 0..1
    // survives a render losslessly). r16g16_float is the 2-channel half variant. These
    // are real pixel formats (unlike the r32g32* vertex-attribute formats below).
    r16_float,
    r32_float,
    r16g16_float,

    // 32-bit-float vertex-attribute formats. These are not pixel/render-target
    // formats. They describe vertex-buffer attributes (a vec2 position, a vec3 or
    // vec4 color) so the graphics path knows how many f32 components to read per
    // attribute. bytesPerPixel doubles as bytes-per-attribute here.
    r32g32_float,
    r32g32b32_float,
    r32g32b32a32_float,

    pub fn bytesPerPixel(self: Format) u32 {
        return switch (self) {
            .rgba8_unorm, .rgba8_srgb, .bgra8_unorm, .depth32_float => 4,
            .r8_unorm => 1,
            .r8g8_unorm => 2,
            .rgba16_float => 8, // 4 channels * 16-bit half
            .rgb10a2, .rgb10x2 => 4, // packed 10/10/10/2 in one dword
            .r16_float => 2, // 1 channel * 16-bit half
            .r32_float => 4, // 1 channel * 32-bit single
            .r16g16_float => 4, // 2 channels * 16-bit half
            .r32g32_float => 8,
            .r32g32b32_float => 12,
            .r32g32b32a32_float => 16,
        };
    }

    /// Number of f32 components a vertex attribute of this format holds (used by
    /// the software graphics path to read vec2 positions and vec3/vec4 colors).
    /// Non-float formats report their natural channel count.
    pub fn componentCount(self: Format) u32 {
        return switch (self) {
            .r8_unorm, .depth32_float, .r16_float, .r32_float => 1,
            .r8g8_unorm, .r16g16_float, .r32g32_float => 2,
            .r32g32b32_float => 3,
            .rgba8_unorm, .rgba8_srgb, .bgra8_unorm, .rgba16_float, .rgb10a2, .rgb10x2, .r32g32b32a32_float => 4,
        };
    }

    /// Whether the format carries high-dynamic-range pixels (values outside the
    /// SDR 8-bit 0..1 range survive a render into it).
    pub fn isHdr(self: Format) bool {
        return switch (self) {
            .rgba16_float, .rgb10a2, .rgb10x2 => true,
            else => false,
        };
    }
};

pub const ResourceUsage = packed struct {
    vertex: bool = false,
    index: bool = false,
    uniform: bool = false,
    sampled: bool = false,
    render_target: bool = false,
    copy_src: bool = false,
    copy_dst: bool = false,
    /// A compute storage buffer: the kernel reads and/or writes it directly (a
    /// `ComputeDispatch.buffers` entry). On the apple driver createResource allocs
    /// a GEM BO regardless of usage, so this is mainly semantic (a driver that
    /// distinguishes storage vs vertex/uniform allocations keys off it).
    storage: bool = false,
};

pub const ResourceKind = enum { buffer, image };

pub const ResourceDesc = union(ResourceKind) {
    buffer: struct { size: usize, usage: ResourceUsage = .{} },
    image: struct {
        width: u32,
        height: u32,
        format: Format,
        usage: ResourceUsage = .{},
        /// MSAA sample count (1/2/4). Default 1 (single-sample). When > 1 the image's
        /// backing holds width*height*samples pixels (sample-minor) for the software
        /// MSAA path. The logical width/height stay the image's dimensions.
        samples: u8 = 1,
        /// Mip level count. Default 1 (base level only). When > 1 the image holds a full
        /// mip chain (base width*height, then w/2 x h/2, ... down to 1x1, clamped to this
        /// count). The driver sizes the backing to hold every level and lays them out so a
        /// sampler with `mip_filter != .none` can read any level. `mapResource` returns the
        /// whole packed-level backing (level 0 first). The caller fills each level (see the
        /// software driver's `mipLevelOffset`).
        mip_levels: u8 = 1,
        /// CUBEMAP: when true this image is a `samplerCube` (6 square faces). The backing
        /// holds 6 * width*height*bpp bytes, faces packed in GL order (+X,-X,+Y,-Y,+Z,-Z).
        /// `width`/`height` are one face's dimensions. Cube + mip do not co-occur here (cube
        /// mip is a follow-up). Default false = an ordinary 2D image (backing unchanged).
        cube: bool = false,
        /// 3D TEXTURE: the Z-slice count of a `sampler3D` volume (default 1 = a 2D image). When
        /// > 1 the backing holds width*height*depth texels packed slice-major. Also the layer count
        /// of a 2D array (see `array`).
        depth: u32 = 1,
        /// 2D ARRAY TEXTURE: when true this image is a `sampler2DArray` with `depth` independent
        /// layers (same layer-major backing as a 3D volume, but sampled by a raw layer index with no
        /// cross-layer filtering). Distinguishes a 2D-array from a 3D volume, which are both
        /// `depth > 1` (the driver can not tell them apart from `depth` alone). Default false.
        array: bool = false,
    },
};

/// The full mip-chain level count for a `w`x`h` 2D texture: `floor(log2(max(w,h))) + 1`,
/// i.e. levels down to 1x1. A glGenerateMipmap / a trilinear sampler uses this.
pub fn mipLevelCount(w: u32, h: u32) u8 {
    var m = @max(w, h);
    if (m == 0) return 1;
    var n: u8 = 1;
    while (m > 1) : (m >>= 1) n += 1;
    return n;
}

/// The dimensions of mip `level` of a `w`x`h` base (each axis halved per level, floored,
/// clamped to >= 1). Level 0 is the base.
pub fn mipLevelSize(w: u32, h: u32, level: u8) [2]u32 {
    return .{ @max(1, w >> @intCast(level)), @max(1, h >> @intCast(level)) };
}

/// The byte offset of mip `level` within a tightly-packed (row-major, `bytes_per_pixel`)
/// mip-chain backing: the sum of the byte sizes of all lower-numbered levels. The base
/// level is at offset 0. Used by the software driver to lay out + sample the chain.
pub fn mipLevelOffset(w: u32, h: u32, level: u8, bytes_per_pixel: usize) usize {
    var off: usize = 0;
    var l: u8 = 0;
    while (l < level) : (l += 1) {
        const s = mipLevelSize(w, h, l);
        off += @as(usize, s[0]) * s[1] * bytes_per_pixel;
    }
    return off;
}

/// The total byte size of a tightly-packed `levels`-level mip chain for a `w`x`h`,
/// `bytes_per_pixel` 2D texture.
pub fn mipChainBytes(w: u32, h: u32, levels: u8, bytes_per_pixel: usize) usize {
    return mipLevelOffset(w, h, levels, bytes_per_pixel);
}

pub const ShaderStage = enum { vertex, fragment, compute };

/// The per-fragment depth comparison a depth-tested pipeline applies (mirrors
/// VkCompareOp). `always` (the default) plus `never` make depth testing a no-op
/// pass/fail. The rest compare the fragment's depth against the stored depth.
pub const CompareOp = enum {
    never,
    less,
    equal,
    less_or_equal,
    greater,
    not_equal,
    greater_or_equal,
    always,
};

/// Optional depth-test state for a graphics pipeline. When `test_enable` is false
/// (the default) the pipeline does no depth testing and the color-only render path
/// is unchanged. When true, each fragment's depth is compared to the depth buffer
/// via `compare_op`. On pass the fragment is shaded and, if `write_enable`, its
/// depth is written. A driver with no depth path ignores this.
pub const DepthState = struct {
    test_enable: bool = false,
    write_enable: bool = false,
    compare_op: CompareOp = .less,
    /// Depth bias (glPolygonOffset / VkPipelineRasterizationStateCreateInfo depthBias*). When
    /// enabled, each fragment's depth gets `slope*m + constant*r` added before the test/write,
    /// where m = max(|dz/dx|, |dz/dy|) (the triangle's screen-space depth slope) and r is the
    /// minimum resolvable depth step. Used to push coplanar geometry (decals, UI layers, shadow
    /// receivers) off the surface so it wins/loses depth consistently instead of z-fighting. A
    /// driver with no bias support ignores this.
    bias_enable: bool = false,
    bias_constant: f32 = 0, // GL `units` / Vulkan depthBiasConstantFactor
    bias_slope: f32 = 0, // GL `factor` / Vulkan depthBiasSlopeFactor
    bias_clamp: f32 = 0, // Vulkan depthBiasClamp (0 = no clamp)
};

/// How a stencil value is updated when the stencil test (and depth test) resolve
/// (mirrors VkStencilOp / GLES glStencilOp). The op is selected per outcome: the
/// stencil-fail op, the depth-fail op (stencil passed, depth failed), and the
/// pass op (both passed). `keep` (the default) leaves the value unchanged.
pub const StencilOp = enum {
    keep,
    zero,
    replace, // write the reference value
    incr_clamp, // increment, saturating at 0xff
    decr_clamp, // decrement, saturating at 0
    invert, // bitwise NOT
    incr_wrap, // increment, wrapping 0xff -> 0
    decr_wrap, // decrement, wrapping 0 -> 0xff
};

/// Per-fragment stencil test + update state for a graphics pipeline (a single face
/// state applied to both windings, matching GLES2 glStencilFunc/glStencilOp which set
/// both faces). When `test_enable` is false (the default) the pipeline does no stencil
/// work and the render path is unchanged. When true and a stencil attachment is bound,
/// each fragment compares `(reference & compare_mask)` against `(stored & compare_mask)`
/// via `compare_op`. The resulting fail/pass (combined with the depth test) selects
/// `fail_op` / `depth_fail_op` / `pass_op`, whose result is written under `write_mask`.
/// A fragment whose stencil test fails is discarded (color + depth unchanged). A driver
/// with no stencil path ignores this.
pub const StencilState = struct {
    test_enable: bool = false,
    compare_op: CompareOp = .always,
    fail_op: StencilOp = .keep,
    depth_fail_op: StencilOp = .keep,
    pass_op: StencilOp = .keep,
    /// ANDed into both the reference and the stored value before comparing.
    compare_mask: u8 = 0xff,
    /// Which stencil bits a write may modify.
    write_mask: u8 = 0xff,
    /// The reference value compared against (and written by `replace`).
    reference: u8 = 0,
};

pub const ShaderModuleDesc = struct {
    stage: ShaderStage,
    code: []const u8,
};

/// Which triangle faces a graphics pipeline discards (back-face culling), and which
/// winding the front face is. Default = no culling (draw both windings), so any
/// pipeline that doesn't request culling renders exactly as before. The software path
/// determines a triangle's winding from its signed screen-space area.
pub const CullMode = enum { none, front, back };
pub const FrontFace = enum { counter_clockwise, clockwise };
pub const CullState = struct {
    mode: CullMode = .none,
    front_face: FrontFace = .counter_clockwise,
};

/// A blend factor (GLES2 glBlendFunc coefficient). Each factor evaluates to an RGBA
/// 4-vector applied component-wise to the source or destination color. The alpha-only
/// variants (`src_alpha`, `dst_alpha`, ...) broadcast a single channel. `src_alpha_saturate`
/// is min(src.a, 1-dst.a) for RGB and 1 for alpha (GL's standard saturate factor).
pub const BlendFactor = enum {
    zero,
    one,
    src_color,
    one_minus_src_color,
    src_alpha,
    one_minus_src_alpha,
    dst_alpha,
    one_minus_dst_alpha,
    dst_color,
    one_minus_dst_color,
    constant_color,
    one_minus_constant_color,
    constant_alpha,
    one_minus_constant_alpha,
    src_alpha_saturate,
};

/// A blend equation (GLES2 glBlendEquation). `add`/`subtract`/`reverse_subtract` combine
/// the factor-weighted source and destination. `min`/`max` ignore the factors and take the
/// component-wise min/max of the raw source and destination (per the GL spec).
pub const BlendOp = enum {
    add,
    subtract,
    reverse_subtract,
    min,
    max,
};

/// Optional alpha-blending state for a graphics pipeline. Default = disabled, which is a
/// pure passthrough (the fragment overwrites the destination), so any pipeline that does
/// not request blending renders exactly as before. When `enable` is true the per-fragment
/// color is combined with the framebuffer color via the separate RGB / alpha factors and
/// ops. `constant` is glBlendColor (the CONSTANT_* factors). The default factors (src=one
/// dst=zero, op=add) are themselves a no-op even when enable is set.
pub const BlendState = struct {
    enable: bool = false,
    src_color: BlendFactor = .one,
    dst_color: BlendFactor = .zero,
    src_alpha: BlendFactor = .one,
    dst_alpha: BlendFactor = .zero,
    color_op: BlendOp = .add,
    alpha_op: BlendOp = .add,
    constant: [4]f32 = .{ 0, 0, 0, 0 },
    /// Per-channel color write mask (glColorMask / VkPipelineColorBlendAttachmentState
    /// .colorWriteMask). A false channel keeps the destination value for that channel. The
    /// default (all true) writes every channel. Applied whether or not blending is enabled.
    /// A stencil-only mask pass sets all-false so the mask draw touches stencil but not color.
    write_mask: [4]bool = .{ true, true, true, true },
};

/// A texture filter for a sampled image (magnification / minification within a level).
pub const Filter = enum { nearest, linear };

/// How a sampler selects / blends between mip levels when a texture is minified.
/// `none` samples the base level only (no mip chain). `nearest` snaps to the closest
/// level (GL_*_MIPMAP_NEAREST). `linear` trilinearly blends the two bracketing levels
/// (GL_*_MIPMAP_LINEAR). The driver computes the LOD from the texture-coordinate
/// screen-space footprint. `none` means the bound `image` has only the base level.
pub const MipFilter = enum { none, nearest, linear };

/// How a sampler wraps an out-of-`[0,1]` texture coordinate.
pub const AddressMode = enum { repeat, clamp_to_edge, mirrored_repeat };

/// A combined-image-sampler binding for a graphics draw: the sampled image Resource
/// plus the sampler state (min/mag/mip filters + per-axis address modes). Optional in
/// the HAL: a driver with no texture path leaves `bindTexture` null and the binding is
/// ignored. The software driver builds its host sampler descriptor (TexDesc) from this.
///
/// `filter` is the magnification filter (kept as the legacy single field so existing
/// callers that set only `filter` get mag==min and no mipmapping). `min_filter` is the
/// minification filter (within a level). `mip_filter` selects how levels are blended.
/// When `mip_filter != .none`, `image` must carry a full mip chain (the driver samples
/// `mip_levels` levels, the base + box-downsampled levels, the GPU/host computing the
/// LOD from the texcoord derivatives).
pub const TextureBinding = struct {
    binding: u32,
    image: *Resource,
    filter: Filter = .nearest, // magnification filter (legacy field)
    min_filter: Filter = .nearest, // minification filter (within a level)
    mip_filter: MipFilter = .none, // how mip levels are blended
    address_u: AddressMode = .repeat,
    address_v: AddressMode = .repeat,
    /// Max anisotropy (GL_TEXTURE_MAX_ANISOTROPY_EXT / VkSamplerCreateInfo.maxAnisotropy). 1 = off
    /// (isotropic). >1 samples along the anisotropy axis at a lower LOD so a minified texture on a
    /// grazing surface stays sharp. The nvidia HW honors it (TSC MAX_ANISOTROPY). The software path
    /// currently samples isotropically (accepted, a follow-up).
    max_anisotropy: f32 = 1,
    /// Mip level range (GL_TEXTURE_BASE_LEVEL / GL_TEXTURE_MAX_LEVEL). Sampling never uses a level
    /// finer than `base_level` nor coarser than `max_level` (clamped to the image's real level count).
    /// Defaults sample the full chain (base 0, max = the GL default 1000). Used for mip streaming and
    /// to bound a texture-atlas mip range.
    base_level: u32 = 0,
    max_level: u32 = 1000,
    /// Component swizzle (GL_TEXTURE_SWIZZLE_R/G/B/A): each sampled output channel is remapped from a
    /// source channel or a constant. Identity by default. A single-channel coverage texture (a font
    /// atlas in R) uses e.g. {one, one, one, r} to broadcast coverage into alpha for text blending.
    swizzle: [4]Swizzle = .{ .r, .g, .b, .a },
    /// LOD bias (GL_TEXTURE_LOD_BIAS): added to the computed mip level-of-detail. Positive = a coarser
    /// (blurrier) mip. Negative = a finer (sharper) mip. 0 = no bias. Used to soften or sharpen a
    /// mipmapped texture (e.g. a small negative bias for crisper text).
    lod_bias: f32 = 0,
    /// LOD clamp (GL_TEXTURE_MIN_LOD / GL_TEXTURE_MAX_LOD): the computed LOD is clamped to
    /// [min_lod, max_lod] before selecting the mip level. Defaults span the full range (GL defaults
    /// -1000 / 1000). Distinct from base/max LEVEL (an integer level index): this clamps the
    /// fractional LOD, so it can also pin a magnified surface to a coarser mip.
    min_lod: f32 = -1000,
    max_lod: f32 = 1000,
    /// Depth compare (GL_TEXTURE_COMPARE_MODE == GL_COMPARE_REF_TO_TEXTURE, for sampler2DShadow). When
    /// true the sampler compares the shader-supplied reference against the texture's stored depth with
    /// `compare_op` and returns the 0/1 (or PCF-averaged) pass fraction instead of the raw texel. Used
    /// for shadow mapping. Default false = an ordinary texture read.
    compare_enable: bool = false,
    /// GL_TEXTURE_COMPARE_FUNC: the comparison applied as `reference <op> stored_depth`. Default
    /// less_or_equal (GL_LEQUAL, the shadow-map convention). Ignored unless `compare_enable`.
    compare_op: CompareOp = .less_or_equal,
};

/// A per-output-channel texture swizzle source (GL_TEXTURE_SWIZZLE_*). `zero`/`one` supply the
/// constant 0.0 / 1.0. The rest select a sampled source channel.
pub const Swizzle = enum(u8) { r, g, b, a, zero, one };

pub const VertexAttribute = struct {
    location: u32,
    format: Format,
    offset: u32,
};

pub const VertexLayout = struct {
    stride: u32,
    attributes: []const VertexAttribute,
};

/// The primitive a draw assembles from its vertex stream. Strips/loops/fans are expanded to
/// these base lists by the API layer (GLES/Vulkan), so a driver only handles the three lists:
/// `triangle_list` (3 verts/tri, the default), `line_list` (2 verts/segment), `point_list`
/// (1 vert/point). UI toolkits draw borders/grids/focus rings as lines and dots as points.
pub const Topology = enum {
    triangle_list,
    line_list,
    point_list,

    /// Vertices consumed per primitive.
    pub fn vertsPerPrimitive(self: Topology) u32 {
        return switch (self) {
            .triangle_list => 3,
            .line_list => 2,
            .point_list => 1,
        };
    }
};

pub const PipelineDesc = struct {
    vertex: *ShaderModule,
    fragment: *ShaderModule,
    vertex_layout: VertexLayout,
    color_format: Format,
    /// The primitive topology the draw assembles. Default = triangle_list (existing
    /// pipelines unchanged). line_list / point_list drive the line / point rasterizer.
    topology: Topology = .triangle_list,
    /// Line width in pixels for a line_list draw (glLineWidth; default 1). The software
    /// rasterizer widens the expanded quad. Nvidia programs SET_ALIASED_LINE_WIDTH_FLOAT.
    /// Ignored for triangle/point topologies.
    line_width: f32 = 1.0,
    /// Depth-test state. Default = disabled, so a pipeline with no depth attachment
    /// (the M4/M5/UBO color-only path) behaves exactly as before.
    depth: DepthState = .{},
    /// Back-face culling state. Default = none (draw both windings), so existing
    /// pipelines are unchanged. vkcube culls back faces (so its unlit interior/back
    /// faces don't overdraw the lit front faces).
    cull: CullState = .{},
    /// Alpha-blend state. Default = disabled (passthrough), so existing pipelines are
    /// unchanged. glmark2's desktop/effect2d composite translucent layers with
    /// SRC_ALPHA / ONE_MINUS_SRC_ALPHA blending.
    blend: BlendState = .{},
    /// MSAA sample count (1/2/4). Default = 1 (no MSAA: the fast single-sample path).
    /// When > 1 the rasterizer evaluates per-sample coverage at the standard sample
    /// positions and writes the FS color (shaded once per pixel) to each covered
    /// sample. The render pass resolves (box-averages) the samples afterward.
    samples: u8 = 1,
    /// GL_SAMPLE_ALPHA_TO_COVERAGE: when set (and samples > 1) the fragment's alpha is turned into a
    /// per-sample coverage mask. A fragment with alpha a covers ~a of the samples (the rest keep the
    /// background), so the MSAA resolve gives smooth alpha edges without blending. Used for foliage /
    /// leaf cut-outs / alpha-tested edges. Default false. Ignored for single-sample.
    alpha_to_coverage: bool = false,
    /// GL_SAMPLE_COVERAGE: when set (and samples > 1) the fragment's covered-sample set is ANDed with
    /// a fixed coverage value. ceil(value*samples) samples survive (inverted if `sample_coverage_invert`),
    /// independent of the fragment alpha. Used for screen-door / LOD-fade transparency. Default false.
    sample_coverage: bool = false,
    /// The GL_SAMPLE_COVERAGE fraction in [0,1] (glSampleCoverage value). Ignored unless sample_coverage.
    sample_coverage_value: f32 = 1.0,
    /// GL_SAMPLE_COVERAGE_INVERT: invert which samples the coverage value keeps. Ignored unless sample_coverage.
    sample_coverage_invert: bool = false,
    /// Stencil-test state (the FRONT face, and both faces when `stencil_back` is null). Default
    /// = disabled, so existing pipelines are unchanged. A UI toolkit uses this for clipping/
    /// masking (draw a mask shape into stencil, then draw content only where the stencil test
    /// passes, e.g. rounded-rect clips).
    stencil: StencilState = .{},
    /// Optional per-face BACK stencil state (two-sided stencil, GLES glStencilFuncSeparate /
    /// glStencilOpSeparate, Vulkan front/back VkStencilOpState). null = single-face (`stencil`
    /// applies to both windings, the common UI-clip case). When set, front-facing fragments use
    /// `stencil` and back-facing use `stencil_back`, e.g. shadow-volume z-fail. A driver with no
    /// two-sided path ignores it (falls back to `stencil` for both faces).
    stencil_back: ?StencilState = null,
};

/// Opaque, driver-owned handles. The driver allocates and interprets these.
/// Consumers only ever hold pointers to them.
pub const Resource = opaque {};
pub const ShaderModule = opaque {};
pub const Pipeline = opaque {};
pub const Surface = opaque {};

/// A minimal one-shot GPU compute dispatch: a compute `shader` module (built via
/// createShaderModule with stage = .compute), the storage `buffers` the kernel
/// binds in order, and the threadgroup-grid dimensions `groups` (x,y,z). The
/// driver runs the kernel to completion (allocs its own launch infra, binds the
/// caller-owned `buffers`, submits, waits the fence) and returns once the results
/// are visible in the buffers.
///
/// BUFFER BINDING ORDER: `buffers[i]` is bound to the kernel's uniform pair
/// u(2i)_u(2i+1) (each buffer's 64-bit GPU VA occupies two 32-bit uniform halves).
/// So buffers[0] -> u0_u1, buffers[1] -> u2_u3, buffers[2] -> u4_u5, etc. A
/// read-input / write-output kernel therefore reads from buffers[0] (u0_u1) and
/// writes to buffers[1] (u2_u3). A single-buffer dispatch (the proven constant
/// kernel) just passes `buffers = &.{out}` and binds u0_u1.
///
/// Every Resource in `buffers` is caller-owned: the driver must not free or
/// re-allocate any of them.
pub const ComputeDispatch = struct {
    shader: *ShaderModule,
    buffers: []const *Resource,
    groups: [3]u32,
};

/// A scissor rectangle in framebuffer pixels, top-left origin (matching the render
/// target's stored pixel layout). Fragments and clears outside [x, x+width) x
/// [y, y+height) are discarded. The GLES layer converts GL's bottom-left glScissor
/// to this top-left convention (as it already does for viewport/winding). The Vulkan
/// layer passes VkRect2D through unchanged.
pub const ScissorRect = struct {
    x: i32,
    y: i32,
    width: u32,
    height: u32,
};

/// The viewport rectangle in WINDOW pixels, top-left origin (the HAL convention, matching the
/// render targets and scissor). NDC [-1,1] maps into [x, x+width] x [y, y+height], and rasterization
/// is clipped to it. The GL bottom-left glViewport is converted to this top-left rect by the GLES
/// layer (which knows the render-target height + the default-fb Y-flip). A null/unset viewport means
/// the full render target (the pre-viewport behavior), so a driver treats "no setViewport" as full-RT.
pub const Viewport = struct {
    x: i32,
    y: i32,
    width: u32,
    height: u32,
    /// glDepthRangef: NDC z (Prism uses the Vulkan [0,1] convention) maps to window z via
    /// z_win = near + z_ndc * (far - near), clamped to [min,max]. Default [0,1] = identity.
    depth_near: f32 = 0.0,
    depth_far: f32 = 1.0,
};

/// The maximum number of simultaneous color render targets (MRT). Matches the common
/// GL/Vulkan minimum guarantee.
pub const MAX_COLOR_TARGETS: u32 = 8;

pub const CommandBuffer = struct {
    ptr: *anyopaque,
    vtable: *const VTable,

    pub const VTable = struct {
        setRenderTarget: *const fn (ptr: *anyopaque, target: *Resource) Error!void,
        /// Bind an additional color render target at `index` for MRT (multiple render
        /// targets): the fragment shader's `layout(location = index) out` writes here.
        /// `index` 0 is the primary target (equivalent to setRenderTarget). Optional: a
        /// driver with no MRT path leaves this null and only index-0 (setRenderTarget)
        /// binds, so single-target rendering is unchanged.
        setColorTarget: ?*const fn (ptr: *anyopaque, index: u32, target: *Resource) Error!void = null,
        clear: *const fn (ptr: *anyopaque, color: Color) Error!void,
        bindPipeline: *const fn (ptr: *anyopaque, pipeline: *Pipeline) Error!void,
        bindVertexBuffer: *const fn (ptr: *anyopaque, buffer: *Resource) Error!void,
        /// Bind a uniform buffer (UBO) at `binding` for the next draw's shaders to
        /// read (std140). Optional: a driver that has no graphics-uniform path leaves
        /// this null and a `bindUniformBuffer` call is a no-op. The software driver
        /// uses it to feed the JITed VS/FS the UBO base pointer.
        bindUniformBuffer: ?*const fn (ptr: *anyopaque, binding: u32, buffer: *Resource) Error!void = null,
        /// Bind a depth attachment (a Resource with a depth32_float-sized buffer) for
        /// this render pass. `clear_value` clears the depth buffer to that value when
        /// non-null. Pass null to bind without clearing so the existing (accumulated)
        /// depth is preserved across draws issued in separate submits within one frame
        /// (the GLES clear-once-then-draw-many contract). Optional: a driver with no
        /// depth path leaves this null and the call is a no-op (the color-only path is
        /// unaffected). The software driver uses it to allocate/clear its depth buffer.
        setDepthTarget: ?*const fn (ptr: *anyopaque, depth: *Resource, clear_value: ?f32) Error!void = null,
        /// Bind a stencil attachment (a Resource holding a u8-per-pixel buffer) for this
        /// render pass. `clear_value` clears the buffer to that value when non-null. Pass
        /// null to bind without clearing (preserve the accumulated stencil across submits,
        /// mirroring setDepthTarget). Optional: a driver with no stencil path leaves this
        /// null and the call is a no-op (the color/depth path is unaffected).
        setStencilTarget: ?*const fn (ptr: *anyopaque, stencil: *Resource, clear_value: ?u8) Error!void = null,
        /// Bind a combined-image-sampler texture for the next draw's fragment shader to
        /// sample. Optional: a driver with no texture path leaves this null and the call
        /// is a no-op. The software driver records it and builds a host sampler
        /// descriptor (TexDesc) for the JITed FS's sampler calls.
        bindTexture: ?*const fn (ptr: *anyopaque, binding: TextureBinding) Error!void = null,
        draw: *const fn (ptr: *anyopaque, vertex_count: u32, first_vertex: u32) Error!void,
        /// Instanced draw: render `vertex_count` vertices `instance_count` times, with the VS
        /// seeing gl_InstanceIndex = first_instance + the instance number each time (so a
        /// vertex-pulling VS can index per-instance data in a UBO). Optional: a driver without
        /// an instancing path leaves this null and the CommandBuffer wrapper falls back to a
        /// single (instance-0) draw. The software driver loops the draw feeding the index.
        drawInstanced: ?*const fn (ptr: *anyopaque, vertex_count: u32, instance_count: u32, first_vertex: u32, first_instance: u32) Error!void = null,
        /// MSAA resolve: box-average the N samples of the multisampled render target
        /// `src` into the single-sample image `dst` (both `width`x`height`, `format`,
        /// `samples` the MSAA count). Issued at EndRenderPass for a pass with a resolve
        /// attachment. Optional: a driver with no MSAA path leaves this null (no-op).
        resolve: ?*const fn (ptr: *anyopaque, src: *Resource, dst: *Resource, width: u32, height: u32, format: Format, samples: u8) Error!void = null,
        /// Set the scissor rectangle for subsequent draws and clears: fragments outside
        /// `rect` are discarded. Pass null to disable scissoring (clip to the full render
        /// target). Optional: a driver with no scissor path leaves this null and the call
        /// is a no-op (no clipping). Implemented on the software and nvidia drivers.
        setScissor: ?*const fn (ptr: *anyopaque, rect: ?ScissorRect) Error!void = null,
        /// Set the viewport rectangle (window pixels, top-left) for subsequent draws: NDC maps into
        /// it and rasterization is clipped to it. Pass null for the full render target (the default).
        /// Optional: a driver without a viewport path leaves this null (full-RT rendering, the
        /// pre-viewport behavior). Implemented on the software and nvidia drivers.
        setViewport: ?*const fn (ptr: *anyopaque, vp: ?Viewport) Error!void = null,
        reset: *const fn (ptr: *anyopaque) void,
        deinit: *const fn (ptr: *anyopaque) void,
    };

    pub fn setRenderTarget(self: CommandBuffer, target: *Resource) Error!void {
        return self.vtable.setRenderTarget(self.ptr, target);
    }
    /// Bind an additional MRT color target at `index` (index 0 is setRenderTarget). A
    /// driver without an MRT path leaves the vtable slot null and the call is a no-op.
    pub fn setColorTarget(self: CommandBuffer, index: u32, target: *Resource) Error!void {
        if (self.vtable.setColorTarget) |f| return f(self.ptr, index, target);
    }
    pub fn clear(self: CommandBuffer, color: Color) Error!void {
        return self.vtable.clear(self.ptr, color);
    }
    pub fn bindPipeline(self: CommandBuffer, pipeline: *Pipeline) Error!void {
        return self.vtable.bindPipeline(self.ptr, pipeline);
    }
    pub fn bindVertexBuffer(self: CommandBuffer, buffer: *Resource) Error!void {
        return self.vtable.bindVertexBuffer(self.ptr, buffer);
    }
    pub fn bindUniformBuffer(self: CommandBuffer, binding: u32, buffer: *Resource) Error!void {
        if (self.vtable.bindUniformBuffer) |f| return f(self.ptr, binding, buffer);
    }
    pub fn setDepthTarget(self: CommandBuffer, depth: *Resource, clear_value: ?f32) Error!void {
        if (self.vtable.setDepthTarget) |f| return f(self.ptr, depth, clear_value);
    }
    pub fn setStencilTarget(self: CommandBuffer, stencil: *Resource, clear_value: ?u8) Error!void {
        if (self.vtable.setStencilTarget) |f| return f(self.ptr, stencil, clear_value);
    }
    pub fn bindTexture(self: CommandBuffer, binding: TextureBinding) Error!void {
        if (self.vtable.bindTexture) |f| return f(self.ptr, binding);
    }
    pub fn draw(self: CommandBuffer, vertex_count: u32, first_vertex: u32) Error!void {
        return self.vtable.draw(self.ptr, vertex_count, first_vertex);
    }
    pub fn drawInstanced(self: CommandBuffer, vertex_count: u32, instance_count: u32, first_vertex: u32, first_instance: u32) Error!void {
        if (self.vtable.drawInstanced) |f| return f(self.ptr, vertex_count, instance_count, first_vertex, first_instance);
        // Fallback for a driver without an instancing path: draw instance 0 only (the VS's
        // gl_InstanceIndex is whatever the non-instanced path supplies, typically 0).
        if (instance_count > 0) return self.vtable.draw(self.ptr, vertex_count, first_vertex);
    }
    pub fn resolve(self: CommandBuffer, src: *Resource, dst: *Resource, width: u32, height: u32, format: Format, samples: u8) Error!void {
        if (self.vtable.resolve) |f| return f(self.ptr, src, dst, width, height, format, samples);
    }
    pub fn setScissor(self: CommandBuffer, rect: ?ScissorRect) Error!void {
        if (self.vtable.setScissor) |f| return f(self.ptr, rect);
    }
    pub fn setViewport(self: CommandBuffer, vp: ?Viewport) Error!void {
        if (self.vtable.setViewport) |f| return f(self.ptr, vp);
    }
    pub fn reset(self: CommandBuffer) void {
        self.vtable.reset(self.ptr);
    }
    pub fn deinit(self: CommandBuffer) void {
        self.vtable.deinit(self.ptr);
    }
};

pub const Context = struct {
    ptr: *anyopaque,
    vtable: *const VTable,

    pub const VTable = struct {
        beginCommands: *const fn (ptr: *anyopaque) Error!CommandBuffer,
        submit: *const fn (ptr: *anyopaque, cb: CommandBuffer) Error!void,
        present: *const fn (ptr: *anyopaque, surface: *Surface, source: *Resource) Error!void,
        deinit: *const fn (ptr: *anyopaque) void,
    };

    pub fn beginCommands(self: Context) Error!CommandBuffer {
        return self.vtable.beginCommands(self.ptr);
    }
    pub fn submit(self: Context, cb: CommandBuffer) Error!void {
        return self.vtable.submit(self.ptr, cb);
    }
    pub fn present(self: Context, surface: *Surface, source: *Resource) Error!void {
        return self.vtable.present(self.ptr, surface, source);
    }
    pub fn deinit(self: Context) void {
        self.vtable.deinit(self.ptr);
    }
};

/// What a brought-up device can actually do, eglinfo-style. A driver fills this
/// out truthfully from its real backend (the GPU model, the formats it renders,
/// the shader paths it accepts). `formats` is a static slice the driver owns. The
/// caller only reads it. Keep this focused. Do not advertise a capability the
/// driver cannot honor.
pub const DeviceCaps = struct {
    /// Human label for the device (a GPU model/arch, or "software renderer").
    device_name: []const u8,
    /// Render/texture formats this device supports (a driver-owned static slice).
    formats: []const Format,
    /// Supports any HDR format (derive from `formats`, see `deriveHdr`).
    hdr: bool,
    /// Accepts SPIR-V shader modules.
    spirv: bool,
    compute: bool,
    graphics: bool,
    present: bool,
    /// Max 2D texture / render-target edge length in pixels.
    max_texture_dim: u32,

    /// True when any of `formats` is an HDR format. Drivers use this so `hdr`
    /// never disagrees with the advertised format set.
    pub fn deriveHdr(formats: []const Format) bool {
        for (formats) |f| if (f.isHdr()) return true;
        return false;
    }
};

/// One captured transform-feedback varying: a run of components taken from a VS output
/// varying. `location` is the pipeline output location the GLSL front end assigned the varying.
/// `first_component` + `components` select which of its (up to 4) components to capture. The
/// driver reads the VS output slot for (location, first_component + k) for k in 0..components and
/// writes each as one f32.
pub const TfCaptureSpec = struct {
    location: u32,
    first_component: u8 = 0,
    components: u8,
};

/// A transform-feedback capture request: run the pipeline's vertex shader over `vertex_count`
/// vertices (independent of rasterization) and write the selected output varyings, tightly
/// interleaved as f32 in `specs` order, into `output` starting at `output_offset`. Only the
/// software driver implements this (see Device.VTable.captureTransformFeedback). Other drivers
/// leave the slot null and the caller reports transform feedback as unsupported.
pub const TransformFeedbackCapture = struct {
    /// The pipeline whose VS is run (built from the same VS+FS the draw uses).
    pipeline: *Pipeline,
    /// The interleaved per-vertex attribute buffer (one vertex per index, already expanded), or
    /// null for a vertex-pulling VS (gl_VertexIndex-only, no attributes).
    vertex_buffer: ?*Resource,
    /// Binding-indexed UBO resources the VS may read (binding 0 = VS block, 1 = FS block, ...).
    ubos: []const ?*Resource,
    /// The GL_TRANSFORM_FEEDBACK_BUFFER capture target.
    output: *Resource,
    /// Byte offset into `output` to start writing.
    output_offset: usize,
    /// The captured varyings, tightly interleaved as f32 in this order per vertex.
    specs: []const TfCaptureSpec,
    /// Number of vertices to run the VS over.
    vertex_count: u32,
    /// The first vertex index (gl_VertexIndex base for a pulling VS, the base row otherwise).
    first_vertex: u32 = 0,
    /// gl_InstanceIndex fed to the VS.
    instance: u32 = 0,
};

pub const Device = struct {
    ptr: *anyopaque,
    vtable: *const VTable,

    pub const VTable = struct {
        caps: *const fn (ptr: *anyopaque) DeviceCaps,
        createContext: *const fn (ptr: *anyopaque) Error!Context,
        createResource: *const fn (ptr: *anyopaque, desc: ResourceDesc) Error!*Resource,
        destroyResource: *const fn (ptr: *anyopaque, resource: *Resource) void,
        mapResource: *const fn (ptr: *anyopaque, resource: *Resource) Error![]u8,
        unmapResource: *const fn (ptr: *anyopaque, resource: *Resource) void,
        /// Optional fast present readback: de-swizzle + format-convert a rendered color
        /// resource straight into a caller XRGB8888 buffer (`dst`, `dst_stride` bytes/row)
        /// in one pass. A driver that sets this lets the present path skip the
        /// mapResource->linear-copy + separate blit (two passes). Null = not supported.
        /// The caller falls back to mapResource + a CPU blit.
        readbackPresent: ?*const fn (ptr: *anyopaque, resource: *Resource, dst: []u8, dst_stride: usize) Error!void = null,
        /// Optional cumulative count of primary-color fragments (samples) the driver has written
        /// (a fragment that passed depth+stencil+coverage). Occlusion queries (glBeginQuery /
        /// glEndQuery, GL_ANY_SAMPLES_PASSED) read the DELTA across a query span. Null = the driver
        /// cannot count. The API layer treats such a query conservatively as "samples passed".
        occlusionSampleCount: ?*const fn (ptr: *anyopaque) u64 = null,
        /// Optional transform-feedback capture: run a pipeline's vertex shader over N vertices and
        /// write the selected output varyings into a buffer (GLES3 transform feedback / GL_RASTERIZER_
        /// DISCARD capture), independent of rasterization. Returns the number of bytes written. Null =
        /// the driver cannot capture. The API layer reports transform feedback as unsupported.
        captureTransformFeedback: ?*const fn (ptr: *anyopaque, cap: TransformFeedbackCapture) Error!usize = null,
        /// Optional: build a sampled depth texture from a rendered depth surface. `depth_rt` is a
        /// depth32_float render-target image that has just been rendered (a shadow-map pass). The
        /// driver returns a new `depth32_float` image with `usage.sampled` holding the same depths,
        /// ready to be bound as a sampler2DShadow source with a hardware depth compare. On a driver
        /// whose render-depth surface is tiled/non-sampleable (nvidia's ZETA) this de-tiles the
        /// rendered depth into a real sampled depth texture (the format the HW DEPTH_COMPARE engages
        /// on). Null = not needed. The API layer falls back to reading the depth as an rgba8 image
        /// and doing the compare in the shader (the software path). Caller owns + destroys the result.
        finalizeDepthTexture: ?*const fn (ptr: *anyopaque, depth_rt: *Resource, w: u32, h: u32) Error!*Resource = null,
        /// Optional: write a resource's mapResource() bytes back to its GPU storage. On a driver
        /// whose mapResource hands back a DE-SWIZZLED linear scratch of a tiled surface (nvidia's
        /// block-linear color RTs), a CPU write into that scratch is otherwise discarded, so a
        /// glBlitFramebuffer into such a target silently does nothing. This re-swizzles the scratch
        /// into the tiled GPU surface. Null = mapResource returns the real backing (software), so
        /// CPU writes already persist and no flush is needed. flushMappedImage() then no-ops.
        flushMappedImage: ?*const fn (ptr: *anyopaque, resource: *Resource) void = null,
        createShaderModule: *const fn (ptr: *anyopaque, desc: ShaderModuleDesc) Error!*ShaderModule,
        destroyShaderModule: *const fn (ptr: *anyopaque, module: *ShaderModule) void,
        dispatchCompute: *const fn (ptr: *anyopaque, d: ComputeDispatch) Error!void,
        createPipeline: *const fn (ptr: *anyopaque, desc: PipelineDesc) Error!*Pipeline,
        destroyPipeline: *const fn (ptr: *anyopaque, pipeline: *Pipeline) void,
        createSurface: *const fn (ptr: *anyopaque, platform_surface: *anyopaque) Error!*Surface,
        destroySurface: *const fn (ptr: *anyopaque, surface: *Surface) void,
        deinit: *const fn (ptr: *anyopaque) void,
    };

    pub fn caps(self: Device) DeviceCaps {
        return self.vtable.caps(self.ptr);
    }
    pub fn createContext(self: Device) Error!Context {
        return self.vtable.createContext(self.ptr);
    }
    pub fn createResource(self: Device, desc: ResourceDesc) Error!*Resource {
        return self.vtable.createResource(self.ptr, desc);
    }
    pub fn destroyResource(self: Device, resource: *Resource) void {
        self.vtable.destroyResource(self.ptr, resource);
    }
    pub fn mapResource(self: Device, resource: *Resource) Error![]u8 {
        return self.vtable.mapResource(self.ptr, resource);
    }
    pub fn unmapResource(self: Device, resource: *Resource) void {
        self.vtable.unmapResource(self.ptr, resource);
    }
    /// Flush a resource's mapResource() bytes back to its GPU storage (see the vtable field). A no-op
    /// on drivers whose mapResource returns the real backing (software). CPU writes already persist.
    pub fn flushMappedImage(self: Device, resource: *Resource) void {
        if (self.vtable.flushMappedImage) |f| f(self.ptr, resource);
    }
    /// Build a sampled depth texture from a rendered depth surface (see the vtable field). The
    /// caller must first confirm `vtable.finalizeDepthTexture != null` (drivers without a
    /// tiled/non-sampleable depth surface leave it null and use the rgba8 shader-compare path).
    pub fn finalizeDepthTexture(self: Device, depth_rt: *Resource, w: u32, h: u32) Error!*Resource {
        return self.vtable.finalizeDepthTexture.?(self.ptr, depth_rt, w, h);
    }
    /// Try the fused present readback (de-swizzle + XRGB convert into `dst` in one pass).
    /// Returns true if the driver handled it. Returns false if unsupported (caller should fall
    /// back to mapResource + a CPU blit).
    pub fn tryReadbackPresent(self: Device, resource: *Resource, dst: []u8, dst_stride: usize) Error!bool {
        if (self.vtable.readbackPresent) |f| {
            try f(self.ptr, resource, dst, dst_stride);
            return true;
        }
        return false;
    }
    /// The driver's cumulative written-sample counter, or null if unsupported (occlusion queries
    /// then fall back to a conservative "passed"). See VTable.occlusionSampleCount.
    pub fn occlusionCounter(self: Device) ?u64 {
        if (self.vtable.occlusionSampleCount) |f| return f(self.ptr);
        return null;
    }
    /// Run transform-feedback capture if the driver supports it, returning the bytes written.
    /// Returns null if unsupported (the caller then reports transform feedback as a no-op).
    pub fn captureTransformFeedback(self: Device, cap: TransformFeedbackCapture) Error!?usize {
        if (self.vtable.captureTransformFeedback) |f| return try f(self.ptr, cap);
        return null;
    }
    pub fn createShaderModule(self: Device, desc: ShaderModuleDesc) Error!*ShaderModule {
        return self.vtable.createShaderModule(self.ptr, desc);
    }
    pub fn destroyShaderModule(self: Device, module: *ShaderModule) void {
        self.vtable.destroyShaderModule(self.ptr, module);
    }
    pub fn dispatchCompute(self: Device, d: ComputeDispatch) Error!void {
        return self.vtable.dispatchCompute(self.ptr, d);
    }
    pub fn createPipeline(self: Device, desc: PipelineDesc) Error!*Pipeline {
        return self.vtable.createPipeline(self.ptr, desc);
    }
    pub fn destroyPipeline(self: Device, pipeline: *Pipeline) void {
        self.vtable.destroyPipeline(self.ptr, pipeline);
    }
    pub fn createSurface(self: Device, platform_surface: *anyopaque) Error!*Surface {
        return self.vtable.createSurface(self.ptr, platform_surface);
    }
    pub fn destroySurface(self: Device, surface: *Surface) void {
        self.vtable.destroySurface(self.ptr, surface);
    }
    pub fn deinit(self: Device) void {
        self.vtable.deinit(self.ptr);
    }
};

test "vtable objects have the expected shape" {
    try std.testing.expect(@hasField(CommandBuffer, "vtable"));
    try std.testing.expect(@hasField(Context, "vtable"));
    try std.testing.expect(@hasField(Device, "vtable"));
}

test "format bytes-per-pixel" {
    try std.testing.expectEqual(@as(u32, 4), Format.rgba8_unorm.bytesPerPixel());
    try std.testing.expectEqual(@as(u32, 1), Format.r8_unorm.bytesPerPixel());
    try std.testing.expectEqual(@as(u32, 8), Format.rgba16_float.bytesPerPixel());
    try std.testing.expectEqual(@as(u32, 4), Format.rgb10a2.bytesPerPixel());
}

test "hdr formats are flagged" {
    try std.testing.expect(Format.rgba16_float.isHdr());
    try std.testing.expect(Format.rgb10a2.isHdr());
    try std.testing.expect(Format.rgb10x2.isHdr());
    try std.testing.expect(!Format.rgba8_unorm.isHdr());
    try std.testing.expect(!Format.bgra8_unorm.isHdr());
}

test "resource desc union tags" {
    const b = ResourceDesc{ .buffer = .{ .size = 1024 } };
    const i = ResourceDesc{ .image = .{ .width = 16, .height = 16, .format = .rgba8_unorm } };
    try std.testing.expect(b == .buffer);
    try std.testing.expect(i == .image);
    try std.testing.expectEqual(@as(usize, 1024), b.buffer.size);
}
