//! Software driver SPIR-V JIT: parse SPIR-V into Vulcan IR, lower to native host code,
//! and call in-process. CPU mirror of the NVIDIA SASS path. The host is the device.
//! Supported targets: aarch64, x86_64, x86, riscv64 (vulcan `native` backend).

const std = @import("std");
const target = @import("vulcan-target");
const spirv = @import("vulcan-spirv");
const sampler = @import("sampler.zig");
const ir = @import("vulcan-ir");
const front = @import("../../spirv.zig");

/// The native (host-arch) JIT backend. `target.native` dispatches to
/// aarch64/x86_64/x86/riscv64 codegen by `builtin.cpu.arch` at comptime.
const native = target.native;

/// JIT a parsed Vulcan IR function and return a live callable `native.JittedModule`.
/// The function is exposed under `name` so the caller can look it up.
/// The caller owns the result and must `deinit()` it.
pub fn jitFunction(gpa: std.mem.Allocator, func: *const front.Function, name: []const u8) !native.JittedModule {
    return native.jitModule(gpa, &.{.{ .name = name, .func = func }});
}

// Graphics (vertex / fragment) SPIR-V JIT.
//
// vulcan's `lowerModule` lowers vertex/fragment shaders: each Input variable
// becomes a scalarized f32 entry-block parameter (one per vector component, in
// SPIR-V declaration order), and each Output store becomes `store value, <slot>`
// where <slot> is an iconst pointer tagged for the NVIDIA backend:
//   vertex:   #[vulcan.gpu.out_attr = N]   N = 0x70..0x7c -> gl_Position xyzw
//                                           N = 0x80 + loc*0x10 + comp*4 -> varying
//   fragment: #[vulcan.gpu.color_out = c]  c = 0..3 -> render-target color RGBA
//
// The NVIDIA backend dispatches on those tags. The host CPU JIT cannot store to
// literal slot addresses. This path rewrites every tagged output store to write
// into a caller-provided f32 output buffer at a dense, known index, then JITs the
// function. Only the output ABI is re-pointed at host memory. Shader math is
// unchanged from vulcan's lowering.
//
// Rewritten entry ABI (AArch64 AAPCS): every f32 input in v0,v1,... (declaration
// order), then one appended pointer param (the output buffer) in x0.

/// The fixed output-buffer layout a JITed graphics shader writes into. Slots map
/// to dense f32 indices so the rasterizer can read them by name.
pub const GfxOut = struct {
    /// vertex: gl_Position.x/y/z/w live at out[0..4].
    pub const position_base: usize = 0;
    /// vertex: varyings start at out[4]. A varying at (location, component) lives at
    /// out[varying_base + location*4 + component] (max 4 components per location).
    pub const varying_base: usize = 4;
    /// vertex: the max varying location count we route (locations 0..7).
    pub const max_varying_locations: usize = 8;
    /// vertex: gl_PointSize (a scalar float) at out[point_size_base]. 0 = the VS did not write
    /// it (the point path falls back to the default/glLineWidth size).
    pub const point_size_base: usize = varying_base + max_varying_locations * 4; // 36
    /// vertex output-buffer length in f32: position(4) + 8 locations * 4 comps + point size.
    pub const vertex_len: usize = point_size_base + 1;
    /// fragment: the max simultaneous MRT color targets we route (locations 0..3). Each
    /// target's RGBA lives at out[target*4 .. target*4+4] (color_out = target*4 + comp).
    pub const max_color_targets: usize = 4;
    /// fragment: color R/G/B/A of target 0 at out[0..4].
    pub const fragment_color_len: usize = 4;
    /// fragment: gl_FragDepth (a scalar) at out[frag_depth_index], after all color targets.
    pub const frag_depth_index: usize = max_color_targets * 4; // 16
    /// fragment output-buffer length: MRT colors (4 * targets) + the gl_FragDepth scalar.
    pub const fragment_len: usize = max_color_targets * 4 + 1; // 17
};

pub const GfxStage = enum { vertex, fragment };

/// Fragment discard (OpKill): the JITed FS calls the host `discard_fn` pointer, which
/// sets this per-thread flag. The rasterizer resets it before each fragment invocation
/// and, after, skips writing the pixel if it was set. Thread-local so tiled/parallel
/// rasterization stays correct.
threadlocal var g_discarded: bool = false;
pub fn discardReset() void {
    g_discarded = false;
}
pub fn discarded() bool {
    return g_discarded;
}
/// The host function the FS's OpKill `call_indirect(discard_fn)` targets (C ABI, no args).
pub fn discardThunk() callconv(.c) void {
    g_discarded = true;
}

/// Caps for the graphics buffer (UBO / storage / sampler-descriptor) pointer params a
/// JITed VS/FS takes.
pub const GfxBuffers = struct {
    /// The maximum number of uniform/storage/sampler-descriptor/helper pointer params we
    /// route into a graphics shader (vulcan lowers each UBO/SSBO, each combined-image-
    /// sampler, the host sampler_fn, the grad_buf, and the host math_fn to one pointer entry
    /// param). `runGraphics` provides concrete call signatures for up to 4 such pointers -
    /// vkcube's fragment shader needs all four (sampler descriptor + sampler_fn + grad_buf +
    /// math_fn). A shader needing more JIT-runs as TooManyInputs and the draw produces
    /// nothing (a documented limit).
    pub const max: usize = 4;
};

/// The maximum number of screen-space-derivative (dFdx/dFdy/Fwidth) gradient-buffer slots
/// a fragment shader can take. vkcube's FS derivatives one vec3 varying = 3 dFdx + 3 dFdy =
/// 6; cap generously.
pub const MAX_GRAD_INPUTS: usize = 16;

/// The maximum number of scalar f32 varying input params a fragment shader takes (the
/// varying buffer holds max_varying_locations*4 = 32 components).
pub const MAX_FS_INPUTS: usize = 32;

/// One entry in a fragment shader's gradient buffer: the per-triangle screen-space
/// derivative of a varying scalar. `axis` selects dFdx vs dFdy. `varying_index` is the
/// scalar's index in the VS output / interpolation buffer (so the rasterizer knows which
/// varying to take the gradient of). Vulcan synthesizes ONE `grad_buf` pointer param the FS
/// loads each gradient from at `grad_buf[index]` (so the FS's float-register varying
/// interface is unchanged no matter how many derivatives are taken). The rasterizer fills
/// `grad_buf[index]` for each entry, in `grads` order.
pub const GradParam = struct {
    axis: enum { x, y },
    /// The varying scalar index in the VS output / interpolation buffer this gradient is
    /// of: (slot - ATTR_GENERIC0)/0x10 * 4 + ((slot - ATTR_GENERIC0) % 0x10)/4.
    varying_index: usize,
};

/// ATTR_GENERIC0: the first generic varying attribute byte slot (mirrors vulcan's
/// lower.zig). A varying at (location, component) is at slot ATTR_GENERIC0 + loc*0x10 +
/// comp*4. The grad_slot func attrs carry this slot.
const ATTR_GENERIC0: u32 = 0x80;

/// Metadata describing a lowered graphics shader's interface, recovered from the IR.
pub const GfxInfo = struct {
    stage: GfxStage,
    /// Number of leading i32 BuiltIn index parameters (gl_VertexIndex / gl_InstanceIndex,
    /// in that order) vulcan synthesized for a vertex-pulling VS. They come first in the
    /// entry ABI (GPRs x0.., before the f32 inputs and the buffer pointers), so the draw
    /// supplies the per-vertex / per-instance index. 0 for an attribute-fed VS or any FS.
    index_count: usize = 0,
    /// Number of scalar f32 input parameters (the shader's Input varying interface,
    /// scalarized). These are passed in v0..v{input_count-1}. Screen-space-derivative
    /// gradients do not count here (they come through the grad_buf pointer), so this stays
    /// small (within the 8 FP arg registers for typical shaders).
    input_count: usize,
    /// Each f32 input param's varying interpolation-buffer index, in entry-param
    /// (declaration / ABI v0,v1,...) order. The varying buffer is laid out by location
    /// (out[varying_base + location*4 + component]), but vulcan appends the FS's input
    /// params in SPIR-V variable-declaration order, which need not match location order
    /// (vkcube declares frag_pos (loc 1) before texcoord (loc 0)). The rasterizer must
    /// fill v{k} from the varying slot named by input_slots[k], not a packed varying_base+k,
    /// or the varyings get swapped (texcoord.xy would read frag_pos's components). The
    /// index is relative to varying_base (i.e. location*4 + component).
    input_slots: [MAX_FS_INPUTS]u32 = undefined,
    /// The screen-space-derivative gradient entries (dFdx/dFdy of varyings) the FS reads,
    /// in grad_buf index order. The rasterizer computes each per triangle and fills
    /// grad_buf[index]. `grad_count` is how many.
    grads: [MAX_GRAD_INPUTS]GradParam = undefined,
    grad_count: usize = 0,
    /// Number of buffer (UBO / storage / combined-image-sampler descriptor / grad_buf)
    /// pointer parameters vulcan synthesized, after the f32 inputs and before the appended
    /// output pointer. Passed in x0..x{n-1}. The output pointer follows in x{n}.
    buffer_count: usize,
    /// The kind of each pointer param, in ABI (entry-param append) order, so the draw path
    /// supplies the right pointer in each x-register slot (the sampler_fn / grad_buf params
    /// are appended lazily and may interleave, so the order is read off the IR, not assumed).
    buffer_kinds: [GfxBuffers.max]BufferKind = undefined,
    /// The Vulkan binding of each `.descriptor` pointer param (the `vulcan.gpu.binding` tag the
    /// lowering attaches), so the rasterizer feeds it the UBO bound at that binding rather than
    /// in declaration order. -1 = no binding tag (a hand-built shader without one): the draw
    /// then falls back to declaration order (the running descriptor index). Indexed in ABI
    /// (entry-param append) order, parallel to buffer_kinds. Non-`.descriptor` slots are -1.
    buffer_bindings: [GfxBuffers.max]i32 = .{-1} ** GfxBuffers.max,
    /// Whether vulcan appended a `sampler_fn` host-sampler function-pointer entry param
    /// (the first time an OpImageSample lowered).
    has_sampler_fn: bool = false,
    /// Whether the FS writes gl_FragDepth (a `frag_depth`-tagged store). When true the
    /// rasterizer takes the fragment's depth from out[frag_depth_index] instead of the
    /// interpolated triangle depth.
    writes_frag_depth: bool = false,
    /// The varying-buffer indices (location*4 + component) of the first texture sample's
    /// (u, v) coordinate, when each is a direct input varying (the common `texture(s, vUV)`
    /// case). Used by the rasterizer to compute the per-draw mip LOD from the texcoord's
    /// screen-space footprint. `null` when the coordinate is computed (not a bare varying)
    /// or the shader does no sampling. The rasterizer then samples the base level (LOD 0).
    tex_coord_slots: [2]?u32 = .{ null, null },
};

/// The role of a graphics-shader pointer parameter (in entry-ABI order). `descriptor` is
/// a plain buffer descriptor (UBO/SSBO/push-constant block). The rasterizer feeds it the
/// bound buffer's base pointer. `sampler_desc` is a combined-image-sampler descriptor
/// (fed a texture descriptor). They are distinct so a FS that reads both a UBO/push-constant
/// and a texture gets the right pointer in each slot.
pub const BufferKind = enum { descriptor, sampler_desc, sampler_fn, sampler_cube_fn, sampler_shadow_fn, sampler_cube_shadow_fn, sampler_2darray_shadow_fn, sampler_gather_fn, sampler_fetch_fn, sampler_fetch3_fn, grad_buf, math_fn, discard_fn };

/// The function-level stage attribute vulcan tags onto a graphics shader.
fn readStage(func: *const front.Function) ?GfxStage {
    var it = func.attributesOf(.func);
    while (it.next()) |attr| switch (attr) {
        .custom => |c| {
            if (std.mem.eql(u8, c.namespace, "vulcan.gpu") and std.mem.eql(u8, c.key, "stage")) {
                switch (c.value) {
                    .string => |s| {
                        if (std.mem.eql(u8, s, "vertex")) return .vertex;
                        if (std.mem.eql(u8, s, "fragment")) return .fragment;
                    },
                    else => {},
                }
            }
        },
        else => {},
    };
    return null;
}

/// Read the `out_attr` / `color_out` integer tag attached to a value (the slot-
/// pointer iconst's result value of a graphics output store), or null if neither.
const OutTag = union(enum) { out_attr: u32, color_out: u32, frag_depth };
fn readOutTag(func: *const front.Function, value: ir.function.Value) ?OutTag {
    var it = func.attributesOf(.{ .value = value });
    while (it.next()) |attr| switch (attr) {
        .custom => |c| {
            if (!std.mem.eql(u8, c.namespace, "vulcan.gpu")) continue;
            if (std.mem.eql(u8, c.key, "frag_depth")) return .frag_depth; // gl_FragDepth
            const n: u32 = switch (c.value) {
                .int => |v| @intCast(v),
                else => continue,
            };
            if (std.mem.eql(u8, c.key, "out_attr")) return .{ .out_attr = n };
            if (std.mem.eql(u8, c.key, "color_out")) return .{ .color_out = n };
        },
        else => {},
    };
    return null;
}

/// Sentinel `input_slots[k]` values marking a fragment-builtin input the rasterizer
/// fills with the fragment position or face, not an interpolated varying. Placed
/// far above any real varying slot (location*4 + component).
pub const FRAG_COORD_INPUT_BASE: u32 = 0x2000; // + component (0=x,1=y,2=z,3=w)
pub const FRONT_FACING_INPUT: u32 = 0x2010;
pub const POINT_COORD_INPUT_BASE: u32 = 0x2020; // gl_PointCoord: + component (0=s,1=t)

/// The `bicomp` component index tagged on a gl_FragCoord/gl_FrontFacing input param.
fn readBiComp(func: *const front.Function, value: ir.function.Value) u32 {
    var it = func.attributesOf(.{ .value = value });
    while (it.next()) |attr| switch (attr) {
        .custom => |c| if (std.mem.eql(u8, c.namespace, "vulcan.gpu") and std.mem.eql(u8, c.key, "bicomp")) {
            return switch (c.value) {
                .int => |v| @intCast(v),
                else => 0,
            };
        },
        else => {},
    };
    return 0;
}

fn readBuiltinParam(func: *const front.Function, value: ir.function.Value) ?u32 {
    var it = func.attributesOf(.{ .value = value });
    while (it.next()) |attr| switch (attr) {
        .custom => |c| {
            if (std.mem.eql(u8, c.namespace, "vulcan.gpu") and std.mem.eql(u8, c.key, "builtin")) {
                return switch (c.value) {
                    .int => |v| @intCast(v),
                    else => null,
                };
            }
        },
        else => {},
    };
    return null;
}

/// Read a value's `vulcan.gpu.attr` slot tag (the input varying attribute slot vulcan
/// tagged onto each scalarized FS/VS input param: ATTR_GENERIC0 + location*0x10 +
/// component*4). Returns null if the value carries no `attr` tag.
fn readAttrSlot(func: *const front.Function, value: ir.function.Value) ?u32 {
    var it = func.attributesOf(.{ .value = value });
    while (it.next()) |attr| switch (attr) {
        .custom => |c| {
            if (std.mem.eql(u8, c.namespace, "vulcan.gpu") and std.mem.eql(u8, c.key, "attr")) {
                return switch (c.value) {
                    .int => |v| @intCast(v),
                    else => null,
                };
            }
        },
        else => {},
    };
    return null;
}

/// Whether a value carries the `vulcan.gpu.grad_buf` tag (the synthesized gradient-buffer
/// pointer param a fragment shader loads its screen-space derivatives from).
fn isGradBufParam(func: *const front.Function, value: ir.function.Value) bool {
    var it = func.attributesOf(.{ .value = value });
    while (it.next()) |attr| switch (attr) {
        .custom => |c| if (std.mem.eql(u8, c.namespace, "vulcan.gpu") and std.mem.eql(u8, c.key, "grad_buf")) return true,
        else => {},
    };
    return false;
}

/// Read the FS gradient-buffer layout from the function's `grad_slot` attrs (one per
/// buffer index, in order): each packs (slot << 1 | axis), axis 0 = dFdx, 1 = dFdy. Fills
/// `out` with the (axis, varying_index) for each gradient and returns the count.
fn readGradSlots(func: *const front.Function, out: *[MAX_GRAD_INPUTS]GradParam) usize {
    var n: usize = 0;
    var it = func.attributesOf(.func);
    while (it.next()) |attr| switch (attr) {
        .custom => |c| {
            if (!std.mem.eql(u8, c.namespace, "vulcan.gpu") or !std.mem.eql(u8, c.key, "grad_slot")) continue;
            const packed_val: u32 = switch (c.value) {
                .int => |v| @intCast(v),
                else => continue,
            };
            if (n >= out.len) continue;
            const is_y = (packed_val & 1) != 0;
            const slot = packed_val >> 1;
            const rel = slot - ATTR_GENERIC0;
            const loc = rel / 0x10;
            const comp = (rel % 0x10) / 4;
            out[n] = .{ .axis = if (is_y) .y else .x, .varying_index = loc * 4 + comp };
            n += 1;
        },
        else => {},
    };
    return n;
}

/// Whether a value carries the `vulcan.gpu.sampler_fn` flag tag (the host-sampler
/// function-pointer entry param vulcan appends for a texturing shader).
fn isSamplerFnParam(func: *const front.Function, value: ir.function.Value) bool {
    var it = func.attributesOf(.{ .value = value });
    while (it.next()) |attr| switch (attr) {
        .custom => |c| if (std.mem.eql(u8, c.namespace, "vulcan.gpu") and std.mem.eql(u8, c.key, "sampler_fn")) return true,
        else => {},
    };
    return false;
}

/// Whether a value carries the `vulcan.gpu.sampler_cube_fn` or `.sampler_3d_fn` flag tag (the
/// host vec3-coordinate sampler param vulcan appends for a `samplerCube` or `sampler3D` sample).
/// Both bind the same host function (`sampleTextureCube`), which dispatches cube-face-select vs
/// trilinear-3D on the bound descriptor. Only the GPU backend distinguishes the tags (TEX dim).
fn isSamplerCubeFnParam(func: *const front.Function, value: ir.function.Value) bool {
    var it = func.attributesOf(.{ .value = value });
    while (it.next()) |attr| switch (attr) {
        .custom => |c| if (std.mem.eql(u8, c.namespace, "vulcan.gpu") and (std.mem.eql(u8, c.key, "sampler_cube_fn") or std.mem.eql(u8, c.key, "sampler_3d_fn") or std.mem.eql(u8, c.key, "sampler_2darray_fn"))) return true,
        else => {},
    };
    return false;
}

/// Whether a value carries the `vulcan.gpu.sampler_shadow_fn` flag tag (the host depth-compare
/// sampler param vulcan appends for a `sampler2DShadow` sample). Binds to `sampleTextureShadow`,
/// which returns a scalar depth-compare fraction.
fn isSamplerShadowFnParam(func: *const front.Function, value: ir.function.Value) bool {
    var it = func.attributesOf(.{ .value = value });
    while (it.next()) |attr| switch (attr) {
        .custom => |c| if (std.mem.eql(u8, c.namespace, "vulcan.gpu") and std.mem.eql(u8, c.key, "sampler_shadow_fn")) return true,
        else => {},
    };
    return false;
}

/// Whether a value carries the `vulcan.gpu.sampler_cube_shadow_fn` flag tag (the host cube
/// depth-compare sampler param vulcan appends for a `samplerCubeShadow` sample). Binds to
/// `sampleTextureCubeShadow`, which returns a scalar depth-compare fraction.
fn isSamplerCubeShadowFnParam(func: *const front.Function, value: ir.function.Value) bool {
    var it = func.attributesOf(.{ .value = value });
    while (it.next()) |attr| switch (attr) {
        .custom => |c| if (std.mem.eql(u8, c.namespace, "vulcan.gpu") and std.mem.eql(u8, c.key, "sampler_cube_shadow_fn")) return true,
        else => {},
    };
    return false;
}

/// Whether a value carries the `vulcan.gpu.sampler_2darray_shadow_fn` flag tag (the host 2D-array
/// depth-compare sampler param vulcan appends for a `sampler2DArrayShadow` sample). Binds to
/// `sampleTexture2dArrayShadow`, which returns a scalar depth-compare fraction.
fn isSampler2dArrayShadowFnParam(func: *const front.Function, value: ir.function.Value) bool {
    var it = func.attributesOf(.{ .value = value });
    while (it.next()) |attr| switch (attr) {
        .custom => |c| if (std.mem.eql(u8, c.namespace, "vulcan.gpu") and std.mem.eql(u8, c.key, "sampler_2darray_shadow_fn")) return true,
        else => {},
    };
    return false;
}

/// Whether a value carries the `vulcan.gpu.sampler_gather_fn` flag tag (the host-gather
/// function-pointer entry param vulcan appends for a `textureGather`). Binds `sampleTextureGather`,
/// which returns one component of the 4-texel bilinear footprint. A GPU backend emits a TLD4.
fn isSamplerGatherFnParam(func: *const front.Function, value: ir.function.Value) bool {
    var it = func.attributesOf(.{ .value = value });
    while (it.next()) |attr| switch (attr) {
        .custom => |c| if (std.mem.eql(u8, c.namespace, "vulcan.gpu") and std.mem.eql(u8, c.key, "sampler_gather_fn")) return true,
        else => {},
    };
    return false;
}

/// Whether a value carries the `vulcan.gpu.sampler_fetch_fn` flag tag (the host-fetch function-
/// pointer entry param vulcan appends for a `texelFetch`). Binds `sampleTextureFetch`, which returns
/// the exact texel at integer coords. A GPU backend emits a TLD.
fn isSamplerFetchFnParam(func: *const front.Function, value: ir.function.Value) bool {
    var it = func.attributesOf(.{ .value = value });
    while (it.next()) |attr| switch (attr) {
        .custom => |c| if (std.mem.eql(u8, c.namespace, "vulcan.gpu") and std.mem.eql(u8, c.key, "sampler_fetch_fn")) return true,
        else => {},
    };
    return false;
}

/// Whether a value carries a 2D-array or 3D `texelFetch` host-fn tag (`sampler_fetch_array_fn` or
/// `sampler_fetch_3d_fn`). Both bind `sampleTextureFetch3D` (fetch the exact texel of layer/slice z).
/// Only the GPU backend distinguishes them (the TLD dim).
fn isSamplerFetch3FnParam(func: *const front.Function, value: ir.function.Value) bool {
    var it = func.attributesOf(.{ .value = value });
    while (it.next()) |attr| switch (attr) {
        .custom => |c| if (std.mem.eql(u8, c.namespace, "vulcan.gpu") and (std.mem.eql(u8, c.key, "sampler_fetch_array_fn") or std.mem.eql(u8, c.key, "sampler_fetch_3d_fn"))) return true,
        else => {},
    };
    return false;
}

/// Whether a value carries the `vulcan.gpu.sampler_desc` tag: a combined-image-sampler
/// descriptor pointer param (vs a plain UBO/SSBO/push-constant buffer pointer param).
/// The rasterizer feeds a sampler_desc slot the bound texture descriptor and a plain
/// descriptor slot the bound buffer's base pointer (the UBO / push-constant block).
fn isSamplerDescParam(func: *const front.Function, value: ir.function.Value) bool {
    var it = func.attributesOf(.{ .value = value });
    while (it.next()) |attr| switch (attr) {
        .custom => |c| if (std.mem.eql(u8, c.namespace, "vulcan.gpu") and std.mem.eql(u8, c.key, "sampler_desc")) return true,
        else => {},
    };
    return false;
}

/// The `vulcan.gpu.binding` integer tag on a buffer pointer param (the descriptor's Vulkan
/// binding number), or null when absent. The lowering tags every UBO / push-constant param.
/// A hand-built shader without the tag falls back to declaration-order slotting.
fn paramBinding(func: *const front.Function, value: ir.function.Value) ?i32 {
    var it = func.attributesOf(.{ .value = value });
    while (it.next()) |attr| switch (attr) {
        .custom => |c| if (std.mem.eql(u8, c.namespace, "vulcan.gpu") and std.mem.eql(u8, c.key, "binding")) {
            return switch (c.value) {
                .int => |v| @intCast(v),
                else => null,
            };
        },
        else => {},
    };
    return null;
}

/// Whether a value carries the `vulcan.gpu.math_fn` flag tag (the host-math function-pointer
/// entry param vulcan appends for a transcendental ext-inst (pow / exp / log / sin / cos).
fn isMathFnParam(func: *const front.Function, value: ir.function.Value) bool {
    var it = func.attributesOf(.{ .value = value });
    while (it.next()) |attr| switch (attr) {
        .custom => |c| if (std.mem.eql(u8, c.namespace, "vulcan.gpu") and std.mem.eql(u8, c.key, "math_fn")) return true,
        else => {},
    };
    return false;
}

/// Whether a value carries the `vulcan.gpu.discard_fn` flag tag (the host-discard
/// function-pointer entry param vulcan appends for an OpKill. Calling it kills the
/// fragment. The rasterizer's thunk sets a per-invocation discard flag.
fn isDiscardFnParam(func: *const front.Function, value: ir.function.Value) bool {
    var it = func.attributesOf(.{ .value = value });
    while (it.next()) |attr| switch (attr) {
        .custom => |c| if (std.mem.eql(u8, c.namespace, "vulcan.gpu") and std.mem.eql(u8, c.key, "discard_fn")) return true,
        else => {},
    };
    return false;
}

/// Whether the FS writes gl_FragDepth: any block has a store whose pointer carries the
/// `frag_depth` tag. Scanned before the store-rewrite re-points those pointers.
fn fsWritesFragDepth(func: *const front.Function) bool {
    var bi: usize = 0;
    while (bi < func.blockCount()) : (bi += 1) {
        for (func.blockInsts(@enumFromInt(bi))) |inst| {
            if (func.opcode(inst) == .store) {
                const st = func.opcode(inst).store;
                if (readOutTag(func, st.ptr)) |tag| if (tag == .frag_depth) return true;
            }
        }
    }
    return false;
}

/// Map a tagged output slot to its dense f32 index in the output buffer.
fn slotToIndex(stage: GfxStage, tag: OutTag) ?usize {
    switch (stage) {
        .fragment => switch (tag) {
            .color_out => |c| return if (c < GfxOut.max_color_targets * 4) @as(usize, c) else null, // target*4+comp (MRT)
            .frag_depth => return GfxOut.frag_depth_index, // gl_FragDepth
            .out_attr => return null,
        },
        .vertex => switch (tag) {
            .out_attr => |slot| {
                if (slot == 0x6c) return GfxOut.point_size_base; // gl_PointSize (a scalar)
                if (slot >= 0x70 and slot < 0x80) {
                    // gl_Position component: (slot - 0x70)/4.
                    return GfxOut.position_base + @as(usize, (slot - 0x70) / 4);
                }
                if (slot >= 0x80) {
                    const rel = slot - 0x80;
                    const loc = rel / 0x10;
                    const comp = (rel % 0x10) / 4;
                    if (loc >= GfxOut.max_varying_locations) return null;
                    return GfxOut.varying_base + loc * 4 + comp;
                }
                return null;
            },
            .color_out, .frag_depth => return null, // fragment-only tags
        },
    }
}

/// Rewrite a lowered graphics `Function` in place so every tagged output store
/// writes into an appended output-buffer pointer parameter (instead of the
/// NVIDIA-only slot-pointer), then return its interface metadata. The shader's
/// inputs and arithmetic are left untouched. `func` must be a single-entry-block
/// graphics shader (what a flat passthrough/channel-rotate VS/FS lowers to).
pub fn rewriteGraphics(func: *front.Function) !GfxInfo {
    const stage = readStage(func) orelse return error.NotGraphics;
    const entry: ir.function.Block = @enumFromInt(0);

    // Partition the entry block's current params (before we append the output ptr):
    // vulcan emits the f32 input scalars first, then one pointer param per uniform /
    // storage buffer. Count each class by its param type.
    var index_count: usize = 0;
    var input_count: usize = 0;
    var buffer_count: usize = 0;
    var has_sampler_fn = false;
    var buffer_kinds: [GfxBuffers.max]BufferKind = undefined;
    var buffer_bindings: [GfxBuffers.max]i32 = .{-1} ** GfxBuffers.max;
    var input_slots: [MAX_FS_INPUTS]u32 = undefined;
    for (func.blockParams(entry)) |pv| {
        switch (func.types.type_kind(func.valueType(pv))) {
            .ptr => {
                // Each pointer param is a descriptor (UBO/storage/sampler), the host-
                // sampler function pointer, or the gradient buffer. Record its kind in
                // entry-ABI (append) order so the draw supplies the right pointer per slot.
                const kind: BufferKind = if (isSamplerFnParam(func, pv)) blk: {
                    has_sampler_fn = true;
                    break :blk .sampler_fn;
                } else if (isSamplerCubeFnParam(func, pv)) .sampler_cube_fn else if (isSamplerCubeShadowFnParam(func, pv)) .sampler_cube_shadow_fn else if (isSampler2dArrayShadowFnParam(func, pv)) .sampler_2darray_shadow_fn else if (isSamplerShadowFnParam(func, pv)) .sampler_shadow_fn else if (isSamplerGatherFnParam(func, pv)) .sampler_gather_fn else if (isSamplerFetchFnParam(func, pv)) .sampler_fetch_fn else if (isSamplerFetch3FnParam(func, pv)) .sampler_fetch3_fn else if (isMathFnParam(func, pv)) .math_fn else if (isDiscardFnParam(func, pv)) .discard_fn else if (isGradBufParam(func, pv)) .grad_buf else if (isSamplerDescParam(func, pv)) .sampler_desc else .descriptor;
                if (buffer_count < buffer_kinds.len) {
                    buffer_kinds[buffer_count] = kind;
                    buffer_bindings[buffer_count] = if (kind == .descriptor) (paramBinding(func, pv) orelse -1) else -1;
                }
                buffer_count += 1;
            },
            // A leading i32 BuiltIn index param (gl_VertexIndex / gl_InstanceIndex) the
            // draw supplies, a fragment builtin (gl_FragCoord / gl_FrontFacing) the
            // rasterizer fills, or an ordinary scalar f32 varying input.
            else => if (readBuiltinParam(func, pv)) |bi| switch (bi) {
                42, 43 => index_count += 1, // gl_VertexIndex / gl_InstanceIndex (i32)
                15 => { // gl_FragCoord (a component of the fragment window position)
                    if (input_count < input_slots.len) input_slots[input_count] = FRAG_COORD_INPUT_BASE + readBiComp(func, pv);
                    input_count += 1;
                },
                16 => { // gl_PointCoord (s/t across a point sprite)
                    if (input_count < input_slots.len) input_slots[input_count] = POINT_COORD_INPUT_BASE + readBiComp(func, pv);
                    input_count += 1;
                },
                17 => { // gl_FrontFacing
                    if (input_count < input_slots.len) input_slots[input_count] = FRONT_FACING_INPUT;
                    input_count += 1;
                },
                else => index_count += 1,
            } else {
                // An f32 varying input. Record its varying-buffer index (location*4 +
                // component) from its `attr` slot tag, so the rasterizer fills this param's
                // register from the right interpolated varying (the params are in
                // declaration order, which need not equal location order).
                if (input_count < input_slots.len) {
                    const vidx: u32 = if (readAttrSlot(func, pv)) |slot| blk: {
                        const rel = slot -% ATTR_GENERIC0;
                        const loc = rel / 0x10;
                        const comp = (rel % 0x10) / 4;
                        break :blk loc * 4 + comp;
                    } else @intCast(input_count); // untagged: fall back to packed order.
                    input_slots[input_count] = vidx;
                }
                input_count += 1;
            },
        }
    }
    // The FS gradient-buffer layout (one entry per grad_buf slot, in index order).
    var grads: [MAX_GRAD_INPUTS]GradParam = undefined;
    const grad_count = readGradSlots(func, &grads);

    // Detect gl_FragDepth before the store-rewrite below re-points its tagged pointer.
    const writes_frag_depth = fsWritesFragDepth(func);

    // Append the output-buffer pointer parameter (GPR x0 at the ABI level).
    const ptr_ty = try func.types.intern(.ptr);
    const outbuf = try func.appendBlockParam(entry, ptr_ty);

    // Rewrite every tagged output store in every block (not only the entry block): an
    // inlined shader (e.g. vkcube's FS, whose linearToSrgb helper inlines into branching
    // blocks) lands its color/position stores in a non-entry block. Each tagged store's
    // pointer is a slot-iconst (color component 0..3 for a FS, ATTR_POSITION+ for a VS).
    // Re-point it to `outbuf + idx*4`. `outbuf` is an entry-block param, so it dominates
    // every block. We can place the address-compute immediately before the store in the
    // store's own block and dominance always holds. (Leaving these unrewritten stores to
    // the literal slot address faults: a FS color_out=0 store writes to address 0.)
    var nblock: usize = 0;
    while (nblock < func.blockCount()) : (nblock += 1) {
        const block: ir.function.Block = @enumFromInt(nblock);
        const orig_insts = func.blockInsts(block);

        // Snapshot the list (createInst/setBlockInsts may realloc the block's list).
        const snapshot = try func.allocator.dupe(ir.function.Inst, orig_insts);
        defer func.allocator.free(snapshot);

        var new_insts = std.ArrayListUnmanaged(ir.function.Inst).empty;
        defer new_insts.deinit(func.allocator);

        var rewrote = false;
        for (snapshot) |inst| {
            // A store whose pointer operand is a slot-iconst tagged as an output.
            const op = func.opcode(inst);
            if (op == .store) {
                const st = op.store;
                if (readOutTag(func, st.ptr)) |tag| {
                    rewrote = true;
                    if (slotToIndex(stage, tag)) |idx| {
                        // new ptr = outbuf + idx*4  (byte offset of the f32 slot). The
                        // address-compute is placed right before the store, in this block.
                        const addr = try func.createInst(ptr_ty, .{ .arith_imm = .{
                            .op = .add,
                            .lhs = outbuf,
                            .imm = @intCast(idx * 4),
                        } });
                        try new_insts.append(func.allocator, func.definingInst(addr).?);
                        // Re-point this store at the computed address.
                        func.opcodeMut(inst).* = .{ .store = .{ .value = st.value, .ptr = addr } };
                        try new_insts.append(func.allocator, inst);
                        continue;
                    } else {
                        // An output slot we don't route (e.g. a varying beyond our cap):
                        // drop the store rather than fault on the literal slot address.
                        continue;
                    }
                }
            }
            try new_insts.append(func.allocator, inst);
        }
        if (rewrote) try func.setBlockInsts(block, new_insts.items);
    }

    // Trace the first texture sample's (u, v) coordinate back to its input varying slots,
    // when each is a direct input varying (the common `texture(s, vUV)` case), so the
    // rasterizer can compute the per-draw mip LOD from the texcoord footprint.
    const tex_coord_slots = if (has_sampler_fn) traceTexCoordSlots(func, entry) else .{ null, null };

    return .{ .stage = stage, .index_count = index_count, .input_count = input_count, .input_slots = input_slots, .grads = grads, .grad_count = grad_count, .buffer_count = buffer_count, .buffer_kinds = buffer_kinds, .buffer_bindings = buffer_bindings, .has_sampler_fn = has_sampler_fn, .writes_frag_depth = writes_frag_depth, .tex_coord_slots = tex_coord_slots };
}

/// Find the first `sampler_fn` call_indirect in the function and, if its (u, v) coordinate
/// args are direct input varyings (entry-block params tagged with an `attr` slot), return
/// their varying-buffer indices (location*4 + component). Returns `.{ null, null }` when the
/// coordinate is computed (not a bare varying). The rasterizer then samples the base level.
/// The sampler ABI is call_indirect(sampler_fn, {desc, u, v, out}). We match args[1]/args[2].
fn traceTexCoordSlots(func: *const front.Function, entry: ir.function.Block) [2]?u32 {
    // Map each entry-block input-varying param Value -> its varying index, via the `attr` slot.
    var slot_of = std.AutoHashMapUnmanaged(ir.function.Value, u32).empty;
    defer slot_of.deinit(func.allocator);
    for (func.blockParams(entry)) |pv| {
        if (func.types.type_kind(func.valueType(pv)) == .ptr) continue;
        if (readBuiltinParam(func, pv) != null) continue;
        if (readAttrSlot(func, pv)) |slot| {
            const rel = slot -% ATTR_GENERIC0;
            const idx = (rel / 0x10) * 4 + (rel % 0x10) / 4;
            slot_of.put(func.allocator, pv, idx) catch return .{ null, null };
        }
    }
    // Scan every block for the sampler_fn call_indirect. Read args {desc, u, v, out}.
    var nb: usize = 0;
    while (nb < func.blockCount()) : (nb += 1) {
        const block: ir.function.Block = @enumFromInt(nb);
        for (func.blockInsts(block)) |inst| {
            const op = func.opcode(inst);
            if (op != .call_indirect) continue;
            const ci = op.call_indirect;
            if (!isSamplerFnParam(func, ci.target)) continue;
            const args = func.valueList(ci.args);
            if (args.len < 4) continue;
            return .{ slot_of.get(args[1]), slot_of.get(args[2]) };
        }
    }
    return .{ null, null };
}

// 4-wide quad fragment shading.
//
// An optional fragment-shader entry that shades a 2x2 pixel quad (4 fragments) at
// once: vulcan's `widenGraphics` turns every f32 value into a `<4 x f32>` (lane k =
// fragment k of the quad), so each SPIR-V op becomes one NEON 4-lane op. The scalar
// `rewriteGraphics` entry remains the golden reference and the fallback for FSes outside
// the straight-line/buffer-free widenable subset. The two entries are compiled separately
// from the same SPIR-V.
//
// Quad output ABI: the FS writes each color component as a `<4 x f32>` into a quad
// output buffer, component-major: component c at qout[c*4 .. c*4+4], so
// qout[c*4 + lane] = fragment `lane`'s channel `c`. Each tagged output store
// re-points to `qout + idx*16` (16 bytes per component vector). 16-aligned, so the
// NEON `str q` is aligned. The rasterizer reads back fragment k's RGBA from
// qout[0*4+k], qout[1*4+k], qout[2*4+k], qout[3*4+k].

/// A 4-lane f32 input/quad scalar for the widened FS ABI: each lane is one of the 2x2
/// quad's fragments. Passed by value, one per varying scalar (in v0..v7), the host
/// C-ABI placing each 16-byte vector in a single SIMD arg register (AAPCS).
pub const Quad = @Vector(4, f32);

/// The widened FS's quad output buffer length in f32: 4 color components, each a 4-lane
/// vector = 16 floats (component-major: out[c*4 + lane]).
pub const quad_out_len: usize = 16;

/// Widen a lowered, single-block fragment `Function` to 4-wide SIMD (via vulcan's
/// `widenGraphics`), then re-point every tagged color-output store to a quad output
/// buffer (component-major `<4 x f32>` slots at qout + comp*16). Returns the FS interface
/// metadata (the scalar varying-input layout still applies; each input is now a Quad).
/// Returns error.NotWidenable for an FS outside the vectorizable subset (the caller keeps
/// the scalar path). `func` must be a freshly-parsed FS, not already scalar-rewritten.
pub fn rewriteGraphicsQuad(func: *front.Function) !GfxInfo {
    const stage = readStage(func) orelse return error.NotGraphics;
    if (stage != .fragment) return error.NotGraphics;
    const entry: ir.function.Block = @enumFromInt(0);

    // Capture the input + buffer-param layout before widening. Widening retypes f32 -> vector
    // (each input becomes a Quad) and rewrites the body, but does not reorder/add/remove the
    // entry-block params (it keeps f32 inputs as <4 x f32> and ptr params as scalar pointers),
    // so the input_slots + buffer_kinds captured here stay valid. The heavy widener (vkcube-
    // class) keeps the sampler-descriptor / sampler_fn / grad_buf / math_fn pointer params. The
    // quad raster supplies them exactly like the scalar path. The grad_buf is broadcast-
    // invariant (per-triangle gradients), so the same per-triangle gradient buffer the scalar
    // path computes is passed unchanged.
    var input_count: usize = 0;
    var input_slots: [MAX_FS_INPUTS]u32 = undefined;
    var buffer_count: usize = 0;
    var has_sampler_fn = false;
    var buffer_kinds: [GfxBuffers.max]BufferKind = undefined;
    var buffer_bindings: [GfxBuffers.max]i32 = .{-1} ** GfxBuffers.max;
    for (func.blockParams(entry)) |pv| {
        switch (func.types.type_kind(func.valueType(pv))) {
            .ptr => {
                const kind: BufferKind = if (isSamplerFnParam(func, pv)) blk: {
                    has_sampler_fn = true;
                    break :blk .sampler_fn;
                } else if (isSamplerCubeFnParam(func, pv)) .sampler_cube_fn else if (isSamplerCubeShadowFnParam(func, pv)) .sampler_cube_shadow_fn else if (isSampler2dArrayShadowFnParam(func, pv)) .sampler_2darray_shadow_fn else if (isSamplerShadowFnParam(func, pv)) .sampler_shadow_fn else if (isSamplerGatherFnParam(func, pv)) .sampler_gather_fn else if (isSamplerFetchFnParam(func, pv)) .sampler_fetch_fn else if (isSamplerFetch3FnParam(func, pv)) .sampler_fetch3_fn else if (isMathFnParam(func, pv)) .math_fn else if (isDiscardFnParam(func, pv)) .discard_fn else if (isGradBufParam(func, pv)) .grad_buf else if (isSamplerDescParam(func, pv)) .sampler_desc else .descriptor;
                if (buffer_count < buffer_kinds.len) {
                    buffer_kinds[buffer_count] = kind;
                    buffer_bindings[buffer_count] = if (kind == .descriptor) (paramBinding(func, pv) orelse -1) else -1;
                }
                buffer_count += 1;
            },
            else => {
                if (readBuiltinParam(func, pv) != null) return error.NotWidenable;
                if (input_count < input_slots.len) {
                    const vidx: u32 = if (readAttrSlot(func, pv)) |slot| blk: {
                        const rel = slot -% ATTR_GENERIC0;
                        const loc = rel / 0x10;
                        const comp = (rel % 0x10) / 4;
                        break :blk loc * 4 + comp;
                    } else @intCast(input_count);
                    input_slots[input_count] = vidx;
                }
                input_count += 1;
            },
        }
    }
    if (buffer_count > GfxBuffers.max) return error.NotWidenable;
    // The FS gradient-buffer layout (one entry per grad_buf slot, in index order). The raster
    // fills grad_buf[i] per triangle. The FS broadcasts each grad_buf load across the 4 lanes.
    var grads: [MAX_GRAD_INPUTS]GradParam = undefined;
    const grad_count = readGradSlots(func, &grads);

    // Widen f32 -> <4 x f32> throughout (vulcan, at the root). Bails (NotWidenable) if the FS
    // is outside both the straight-line and the heavy (broadcast/gather/flatten) subsets.
    try spirv.widenGraphics(func);

    // Re-point every tagged color-output store (now of a <4 x f32>) to qout + comp*16.
    const ptr_ty = try func.types.intern(.ptr);
    const qout = try func.appendBlockParam(entry, ptr_ty);

    const orig_insts = func.blockInsts(entry);
    const snapshot = try func.allocator.dupe(ir.function.Inst, orig_insts);
    defer func.allocator.free(snapshot);

    var new_insts = std.ArrayListUnmanaged(ir.function.Inst).empty;
    defer new_insts.deinit(func.allocator);

    for (snapshot) |inst| {
        const op = func.opcode(inst);
        if (op == .store) {
            const st = op.store;
            if (readOutTag(func, st.ptr)) |tag| {
                if (slotToIndex(.fragment, tag)) |idx| {
                    // qout + idx*16 (16 bytes per <4 x f32> component slot).
                    const addr = try func.createInst(ptr_ty, .{ .arith_imm = .{
                        .op = .add,
                        .lhs = qout,
                        .imm = @intCast(idx * 16),
                    } });
                    try new_insts.append(func.allocator, func.definingInst(addr).?);
                    func.opcodeMut(inst).* = .{ .store = .{ .value = st.value, .ptr = addr } };
                    try new_insts.append(func.allocator, inst);
                    continue;
                } else continue; // an unrouted slot: drop (don't fault).
            }
        }
        try new_insts.append(func.allocator, inst);
    }
    try func.setBlockInsts(entry, new_insts.items);

    return .{ .stage = .fragment, .index_count = 0, .input_count = input_count, .input_slots = input_slots, .grads = grads, .grad_count = grad_count, .buffer_count = buffer_count, .buffer_kinds = buffer_kinds, .buffer_bindings = buffer_bindings, .has_sampler_fn = has_sampler_fn };
}

// Quad FS entry signatures: ni quad (<4 x f32>) inputs in v0.. (SIMD arg regs), then nb
// buffer/sampler/grad_buf/math_fn pointers in x0.. (GPRs), then the quad output ptr. AAPCS
// partitions params by class, so the Quads occupy v0..v{ni-1} and the pointers x0..x{nb}.
// A straight-line FS is buffer-free (nb=0). The heavy (vkcube-class) FS broadcasts/gathers
// through nb up to 4 pointers (sampler desc + sampler_fn + grad_buf + math_fn). ni in 0..8.
const QO = [*]f32;
const QP = [*]const u8;

/// The most value inputs a JITed graphics entry can be called with.
///
/// Each input is one call parameter. AArch64's C ABI passes the first eight
/// floats in v0..v7 and every one after that on the stack, and vulcan's code
/// generator follows the ABI, so the count is NOT limited by the machine. It is
/// limited by how many concrete signatures this file offers to call through.
///
/// That number used to be eight, which is exactly a vec2 plus a vec2 plus a
/// vec4. One attribute more and the call reported `TooManyInputs`, and because
/// every caller in `raster.zig` drops the error and moves on (`catch return`),
/// the draw produced NOTHING, with no diagnostic anywhere. A whole frame came
/// back the colour it was cleared to. phantom's rounded-rect shader, which takes
/// twelve, is what found it: its images and its text drew correctly because both
/// sit at exactly eight.
///
/// Sixteen leaves room for four vec4s and is what a caller can rely on. Beyond
/// it the error is still the honest answer.
pub const max_gfx_inputs: usize = 16;

/// The C ABI signature of a JITed graphics entry: `nf` value parameters, then
/// `nb` pointer parameters, then the output pointer.
///
/// Built here rather than written out. One table for each value type is
/// `(max_gfx_inputs + 1) * (GfxBuffers.max + 1)` entries, and the pair of them
/// spelled by hand is what pinned the limit at eight for as long as it held.
fn GfxSig(comptime Value: type, comptime Ptr: type, comptime Out: type, comptime nf: usize, comptime nb: usize) type {
    @setEvalBranchQuota(100_000);
    const types = comptime blk: {
        var t: [nf + nb + 1]type = undefined;
        for (t[0..nf]) |*e| e.* = Value;
        for (t[nf..][0..nb]) |*e| e.* = Ptr;
        t[nf + nb] = Out;
        break :blk t;
    };
    const attrs: [nf + nb + 1]std.builtin.Type.Fn.Param.Attributes = @splat(.{});
    return *const @Fn(&types, &attrs, void, .{ .@"callconv" = .c });
}

/// Call `f` with the first `nf` of `values`, then the first `nb` of `ptrs`, then
/// `out`, matching the signature `GfxSig` built.
fn invokeGfx(
    comptime Sig: type,
    comptime nf: usize,
    comptime nb: usize,
    f: Sig,
    values: anytype,
    ptrs: anytype,
    out: anytype,
) void {
    var args: std.meta.ArgsTuple(@typeInfo(Sig).pointer.child) = undefined;
    inline for (0..nf) |i| args[i] = values[i];
    inline for (0..nb) |i| args[nf + i] = ptrs[i];
    args[nf + nb] = out;
    @call(.auto, f, args);
}

/// The buffer pointers a graphics entry takes, with every unused slot filled by
/// a benign non-null address. A JITed entry never dereferences a slot its shader
/// did not ask for, and a null there would still have to be passed.
fn gfxPtrs(comptime Ptr: type, bufs: []const ?[*]const u8) [GfxBuffers.max]Ptr {
    const dummy: Ptr = @ptrFromInt(0x1000);
    var out: [GfxBuffers.max]Ptr = @splat(dummy);
    for (0..@min(bufs.len, GfxBuffers.max)) |i| out[i] = bufs[i] orelse dummy;
    return out;
}

/// Invoke a JITed 4-wide quad FS through a PRE-RESOLVED entry address (`entry_ptr`, from
/// `mainEntry`). `n` quad inputs from `inputs` go in v0.. (each a `<4 x f32>` of the quad's
/// 4 fragments' interpolated varying). `nb` buffer/sampler/grad_buf/math_fn pointers from
/// `bufs` go in x0.. (a null slot passes a benign non-null dummy, matching the scalar path).
/// `qout` is the 16-f32 component-major output buffer (out[c*4 + lane]).
pub fn runGraphicsQuadAt(entry_ptr: *const anyopaque, n: usize, nb: usize, inputs: []const Quad, bufs: []const ?[*]const u8, qout: QO) !void {
    if (n > max_gfx_inputs or nb > GfxBuffers.max) return error.TooManyInputs;
    const fnptr: *align(@alignOf(fn () callconv(.c) void)) const anyopaque = @alignCast(entry_ptr);
    const ptrs = gfxPtrs(QP, bufs);
    switch (nb) {
        inline 0...GfxBuffers.max => |NB| switch (n) {
            inline 0...max_gfx_inputs => |N| {
                const Sig = GfxSig(Quad, QP, QO, N, NB);
                invokeGfx(Sig, N, NB, @as(Sig, @ptrCast(fnptr)), inputs, ptrs, qout);
            },
            else => return error.TooManyInputs,
        },
        else => return error.TooManyInputs,
    }
}

// Typed entry callers for a JITed graphics shader. The AArch64 AAPCS ABI is: every
// f32 input scalar in v0.. (declaration order), then every buffer (UBO/storage)
// pointer in x0.. (binding order), then the appended output-buffer pointer in the
// next x register. AAPCS partitions params by class, so floats and pointers occupy
// separate register files regardless of IR param order.
//
// Zig 0.16 forbids synthesizing a `.fn` type with @Type, so the concrete signatures
// are written out explicitly: one per (input-scalar count nf in 0..8) x (buffer
// pointer count nb in 0..2). A `P = [*]const u8` buffer pointer and the trailing
// `[*]f32` output pointer are both ordinary pointers. nb<=2 covers a single MVP UBO
// (multiple UBOs in one graphics shader is a documented later limit).
const P = [*]const u8;
const O = [*]f32;

// nb = 3: nf floats + three buffer/sampler pointers + the output pointer. The third
// pointer is typically the host sampler_fn (a texturing FS = a sampler descriptor + the
// sampler_fn), or a UBO + sampler descriptor + sampler_fn.
const G0_3 = *const fn (P, P, P, O) callconv(.c) void;
const G1_3 = *const fn (f32, P, P, P, O) callconv(.c) void;
const G2_3 = *const fn (f32, f32, P, P, P, O) callconv(.c) void;
const G3_3 = *const fn (f32, f32, f32, P, P, P, O) callconv(.c) void;
const G4_3 = *const fn (f32, f32, f32, f32, P, P, P, O) callconv(.c) void;
const G5_3 = *const fn (f32, f32, f32, f32, f32, P, P, P, O) callconv(.c) void;
const G6_3 = *const fn (f32, f32, f32, f32, f32, f32, P, P, P, O) callconv(.c) void;
const G7_3 = *const fn (f32, f32, f32, f32, f32, f32, f32, P, P, P, O) callconv(.c) void;
const G8_3 = *const fn (f32, f32, f32, f32, f32, f32, f32, f32, P, P, P, O) callconv(.c) void;

// nb = 4: nf floats + four buffer/sampler pointers + the output pointer. vkcube's fragment
// shader hits this: a sampler descriptor + the host sampler_fn + the grad_buf (dFdx/dFdy) +
// the host math_fn (pow), all at once.
const G0_4 = *const fn (P, P, P, P, O) callconv(.c) void;
const G1_4 = *const fn (f32, P, P, P, P, O) callconv(.c) void;
const G2_4 = *const fn (f32, f32, P, P, P, P, O) callconv(.c) void;
const G3_4 = *const fn (f32, f32, f32, P, P, P, P, O) callconv(.c) void;
const G4_4 = *const fn (f32, f32, f32, f32, P, P, P, P, O) callconv(.c) void;
const G5_4 = *const fn (f32, f32, f32, f32, f32, P, P, P, P, O) callconv(.c) void;
const G6_4 = *const fn (f32, f32, f32, f32, f32, f32, P, P, P, P, O) callconv(.c) void;
const G7_4 = *const fn (f32, f32, f32, f32, f32, f32, f32, P, P, P, P, O) callconv(.c) void;
const G8_4 = *const fn (f32, f32, f32, f32, f32, f32, f32, f32, P, P, P, P, O) callconv(.c) void;

// Vertex-pulling entry signatures: ni leading i32 index params (gl_VertexIndex [+
// gl_InstanceIndex]), then zero f32 inputs (a pulling VS has no vertex attributes), then
// nb buffer pointers (the UBO the vertices are pulled from), then the output pointer. In
// AAPCS the i32 indices and the pointers share the GPR file in declaration order (indices
// first), and there are no FPR args. ni in {1,2}, nb in {0..3}.
const VP1_0 = *const fn (i32, O) callconv(.c) void;
const VP1_1 = *const fn (i32, P, O) callconv(.c) void;
const VP1_2 = *const fn (i32, P, P, O) callconv(.c) void;
const VP1_3 = *const fn (i32, P, P, P, O) callconv(.c) void;
const VP2_0 = *const fn (i32, i32, O) callconv(.c) void;
const VP2_1 = *const fn (i32, i32, P, O) callconv(.c) void;
const VP2_2 = *const fn (i32, i32, P, P, O) callconv(.c) void;
const VP2_3 = *const fn (i32, i32, P, P, P, O) callconv(.c) void;

/// Resolve the JITed "main" entry's raw code address once. The address is independent of
/// the typed signature we later cast it to (all the G*/VP* fn types point at the same code),
/// so the rasterizer can cache this pointer in the pipeline and skip the per-fragment symbol
/// string lookup (`JittedModule.entry` does a linear `std.mem.eql` over the symbol table;
/// the profile showed this dominating the per-fragment call overhead). Null if no "main".
pub fn mainEntry(compiled: *const native.JittedModule) ?*const anyopaque {
    return compiled.entry(*const anyopaque, "main");
}

/// Like `runGraphics`, but dispatches through a pre-resolved entry code address (`fnptr`,
/// from `mainEntry`) instead of looking the "main" symbol up by string on every call. This
/// is the per-fragment hot path: the rasterizer resolves `fnptr` once per pipeline and calls
/// this for each covered fragment. Same ABI and arity dispatch as `runGraphics`.
pub fn runGraphicsAt(entry_ptr: *const anyopaque, n: usize, nb: usize, inputs: []const f32, bufs: []const ?[*]const u8, out: O) !void {
    if (n > max_gfx_inputs or nb > GfxBuffers.max) return error.TooManyInputs;
    // The JIT code address is function-aligned. Re-establish that alignment so the
    // fn-pointer casts below don't trip @ptrCast's alignment check (anyopaque is align 1).
    const fnptr: *align(@alignOf(fn () callconv(.c) void)) const anyopaque = @alignCast(entry_ptr);
    const ptrs = gfxPtrs(P, bufs);
    switch (nb) {
        inline 0...GfxBuffers.max => |NB| switch (n) {
            inline 0...max_gfx_inputs => |N| {
                const Sig = GfxSig(f32, P, O, N, NB);
                invokeGfx(Sig, N, NB, @as(Sig, @ptrCast(fnptr)), inputs, ptrs, out);
            },
            else => return error.TooManyInputs,
        },
        else => return error.TooManyInputs,
    }
}

/// Invoke a JITed vertex-pulling shader: `ni` leading i32 index params from `indices`
/// (gl_VertexIndex first, then gl_InstanceIndex) in x0.., `nb` buffer pointers from `bufs`
/// (the pulled-from UBO, binding order) next in the GPRs, the output pointer last. The VS
/// reads its position + attributes from the UBO indexed by the index params. There is no
/// vertex buffer and no f32 attribute inputs. ni in {1,2}, nb in {0..3}.
pub fn runGraphicsPulling(compiled: *const native.JittedModule, ni: usize, nb: usize, indices: []const i32, bufs: []const ?[*]const u8, out: O) !void {
    if (ni == 0 or ni > 2 or nb > 3) return error.TooManyInputs;
    const idx = indices;
    const dummy: P = @ptrFromInt(0x1000);
    const b0: P = if (bufs.len > 0) (bufs[0] orelse dummy) else dummy;
    const b1: P = if (bufs.len > 1) (bufs[1] orelse dummy) else dummy;
    const b2: P = if (bufs.len > 2) (bufs[2] orelse dummy) else dummy;
    const idx0 = idx[0];
    const idx1: i32 = if (idx.len > 1) idx[1] else 0;
    const c = compiled;
    const fp = struct {
        fn get(comptime T: type, cc: *const native.JittedModule) ?T {
            return cc.entry(T, "main");
        }
    }.get;
    switch (ni) {
        1 => switch (nb) {
            0 => (fp(VP1_0, c) orelse return error.NoEntry)(idx0, out),
            1 => (fp(VP1_1, c) orelse return error.NoEntry)(idx0, b0, out),
            2 => (fp(VP1_2, c) orelse return error.NoEntry)(idx0, b0, b1, out),
            3 => (fp(VP1_3, c) orelse return error.NoEntry)(idx0, b0, b1, b2, out),
            else => return error.TooManyInputs,
        },
        2 => switch (nb) {
            0 => (fp(VP2_0, c) orelse return error.NoEntry)(idx0, idx1, out),
            1 => (fp(VP2_1, c) orelse return error.NoEntry)(idx0, idx1, b0, out),
            2 => (fp(VP2_2, c) orelse return error.NoEntry)(idx0, idx1, b0, b1, out),
            3 => (fp(VP2_3, c) orelse return error.NoEntry)(idx0, idx1, b0, b1, b2, out),
            else => return error.TooManyInputs,
        },
        else => return error.TooManyInputs,
    }
}

/// Invoke a JITed graphics shader entry ("main"). `n` f32 inputs from `inputs` go in
/// v0..; `nb` buffer pointers from `bufs` (each a UBO/storage base, binding order) go
/// in x0..; `out` is the appended output-buffer pointer. A null/unbound buffer slot
/// passes a benign non-null pointer (the shader must not read an unbound buffer).
pub fn runGraphics(compiled: *const native.JittedModule, n: usize, nb: usize, inputs: []const f32, bufs: []const ?[*]const u8, out: O) !void {
    if (n > max_gfx_inputs or nb > GfxBuffers.max) return error.TooManyInputs;
    const ptrs = gfxPtrs(P, bufs);
    switch (nb) {
        inline 0...GfxBuffers.max => |NB| switch (n) {
            inline 0...max_gfx_inputs => |N| {
                const Sig = GfxSig(f32, P, O, N, NB);
                const f = compiled.entry(Sig, "main") orelse return error.NoEntry;
                invokeGfx(Sig, N, NB, f, inputs, ptrs, out);
            },
            else => return error.TooManyInputs,
        },
        else => return error.TooManyInputs,
    }
}

test "spirv compute control flow: a glslang for-loop + if JITs and runs natively" {
    const gpa = std.testing.allocator;

    // Real glslang output for:
    //   void main(){ uint i=gl_GlobalInvocationID.x; uint s=0u;
    //                for(uint k=0u;k<=i;k++){ s+=k; }
    //                if((i&1u)==1u){ s+=1000u; } b[i]=s; }
    // The loop and the if make this a multi-block body whose locals (i, s, k) are
    // stored/loaded across blocks. The multi-block mem2reg (SSA + OpPhi) path handles this.
    const spv align(4) = @embedFile("vkcf_test.spv").*;

    var func = try front.parseSpirv(gpa, &spv);
    defer func.deinit();

    var compiled = try jitFunction(gpa, &func, "main");
    defer compiled.deinit();

    // Compute entry ABI: main(gid: i32, buf0: ptr). Dispatch over N=256 invocations.
    const Fn = *const fn (i32, [*]u8) callconv(.c) void;
    const f = compiled.entry(Fn, "main") orelse return error.NoEntry;

    const N = 256;
    const buf = try gpa.alloc(u32, N);
    defer gpa.free(buf);
    @memset(buf, 0);
    const base: [*]u8 = @ptrCast(buf.ptr);
    var i: i32 = 0;
    while (i < N) : (i += 1) f(i, base);

    // out[i] = i*(i+1)/2 + (i odd ? 1000 : 0).
    for (0..N) |k| {
        const expect: u32 = @intCast(k * (k + 1) / 2 + (if (k & 1 == 1) @as(usize, 1000) else 0));
        try std.testing.expectEqual(expect, buf[k]);
    }
}

test "spirv compute: x*y - x parses, JITs, and runs natively" {
    const gpa = std.testing.allocator;
    const op = spirv.opcodes;

    // int f(int x, int y) { return x*y - x; } Built exactly as the parseSpirv
    // test in lib/prism/spirv.zig, so it exercises the real front end.
    // ids: int=1, fnty=2, f=3, x=4, y=5, entry=6, prod=7, diff=8.
    var b = try spirv.binary.Builder.init(gpa, 9);
    defer b.deinit(gpa);
    try b.emit(gpa, op.TypeInt, &.{ 1, 32, 1 });
    try b.emit(gpa, op.TypeFunction, &.{ 2, 1, 1, 1 });
    try b.emit(gpa, op.Function, &.{ 1, 3, 0, 2 });
    try b.emit(gpa, op.FunctionParameter, &.{ 1, 4 });
    try b.emit(gpa, op.FunctionParameter, &.{ 1, 5 });
    try b.emit(gpa, op.Label, &.{6});
    try b.emit(gpa, op.IMul, &.{ 1, 7, 4, 5 });
    try b.emit(gpa, op.ISub, &.{ 1, 8, 7, 4 });
    try b.emit(gpa, op.ReturnValue, &.{8});
    try b.emit(gpa, op.FunctionEnd, &.{});

    // SPIR-V bytes -> Vulcan IR (via the shared parseSpirv front end).
    var func = try front.parseSpirv(gpa, std.mem.sliceAsBytes(b.words.items));
    defer func.deinit();

    // Vulcan IR -> AArch64 (isel + link) -> mapped W^X image, callable in-process.
    var compiled = try jitFunction(gpa, &func, "f");
    defer compiled.deinit();

    const f = compiled.entry(*const fn (i32, i32) callconv(.c) i32, "f").?;

    // f(3, 4) = 3*4 - 3 = 9, run as real machine code on the host.
    try std.testing.expectEqual(@as(i32, 9), f(3, 4));
    // A couple more pairs to prove it is computing, not returning a constant.
    try std.testing.expectEqual(@as(i32, 5), f(5, 2)); // 5*2 - 5 = 5
    try std.testing.expectEqual(@as(i32, 7), f(-7, 0)); // (-7)*0 - (-7) = 7
}

test "spirv graphics: a channel-rotate fragment shader JITs and runs natively" {
    const gpa = std.testing.allocator;
    const op = spirv.opcodes;
    const sc = op.StorageClass;

    // FS: layout(location=0) in vec3 vc; layout(location=0) out vec4 o;
    //     void main(){ o = vec4(vc.b, vc.r, vc.g, 1.0); }  -- channels rotated.
    // ids: void=1 fnty=2 f32=3 v3=4 v4=5 pInV3=6 pOutV4=7 vc=8 o=9 main=10 entry=11
    //      one=12 loaded=13 b=14 r=15 g=16 res=17.
    var b = try spirv.binary.Builder.init(gpa, 18);
    defer b.deinit(gpa);
    try b.emit(gpa, op.EntryPoint, &.{ op.ExecutionModel.fragment, 10, 0, 8, 9 });
    try b.emit(gpa, op.Decorate, &.{ 8, op.Decoration.location, 0 });
    try b.emit(gpa, op.Decorate, &.{ 9, op.Decoration.location, 0 });
    try b.emit(gpa, op.TypeVoid, &.{1});
    try b.emit(gpa, op.TypeFunction, &.{ 2, 1 });
    try b.emit(gpa, op.TypeFloat, &.{ 3, 32 });
    try b.emit(gpa, op.TypeVector, &.{ 4, 3, 3 });
    try b.emit(gpa, op.TypeVector, &.{ 5, 3, 4 });
    try b.emit(gpa, op.TypePointer, &.{ 6, sc.input, 4 });
    try b.emit(gpa, op.TypePointer, &.{ 7, sc.output, 5 });
    try b.emit(gpa, op.Constant, &.{ 3, 12, @bitCast(@as(f32, 1.0)) });
    try b.emit(gpa, op.Variable, &.{ 6, 8, sc.input });
    try b.emit(gpa, op.Variable, &.{ 7, 9, sc.output });
    try b.emit(gpa, op.Function, &.{ 1, 10, 0, 2 });
    try b.emit(gpa, op.Label, &.{11});
    try b.emit(gpa, op.Load, &.{ 4, 13, 8 }); // vc
    try b.emit(gpa, op.CompositeExtract, &.{ 3, 14, 13, 2 }); // vc.b
    try b.emit(gpa, op.CompositeExtract, &.{ 3, 15, 13, 0 }); // vc.r
    try b.emit(gpa, op.CompositeExtract, &.{ 3, 16, 13, 1 }); // vc.g
    try b.emit(gpa, op.CompositeConstruct, &.{ 5, 17, 14, 15, 16, 12 }); // vec4(b,r,g,1)
    try b.emit(gpa, op.Store, &.{ 9, 17 });
    try b.emit(gpa, op.Return, &.{});
    try b.emit(gpa, op.FunctionEnd, &.{});

    var func = try front.parseSpirv(gpa, std.mem.sliceAsBytes(b.words.items));
    defer func.deinit();

    const info = try rewriteGraphics(&func);
    try std.testing.expectEqual(GfxStage.fragment, info.stage);
    try std.testing.expectEqual(@as(usize, 3), info.input_count); // vec3 input
    try std.testing.expectEqual(@as(usize, 0), info.buffer_count); // no UBO

    var compiled = try jitFunction(gpa, &func, "main");
    defer compiled.deinit();

    // Input color (R,G,B) = (0.1, 0.5, 0.9); the rotate makes (out.r,out.g,out.b)
    // = (B, R, G) = (0.9, 0.1, 0.5), out.a = 1.0. The plain passthrough mapping
    // physically cannot produce this swap.
    var out = [_]f32{0} ** GfxOut.fragment_len;
    try runGraphics(&compiled, info.input_count, info.buffer_count, &.{ 0.1, 0.5, 0.9 }, &.{}, &out);
    try std.testing.expectApproxEqAbs(@as(f32, 0.9), out[0], 1e-6); // R <- B
    try std.testing.expectApproxEqAbs(@as(f32, 0.1), out[1], 1e-6); // G <- R
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), out[2], 1e-6); // B <- G
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), out[3], 1e-6); // A
}

// 4-wide quad SIMD == scalar equivalence. For each representative widenable FS we JIT
// both the scalar per-fragment entry (the golden reference) and the 4-wide quad entry,
// feed 4 distinct fragments, and assert the quad result matches the scalar result
// lane-by-lane (bit-exact: the same NEON ops, just packed; no float reordering).

/// A per-test buffer binder: a texture descriptor (for sampler_desc), a per-triangle
/// gradient buffer (for grad_buf), and the host sampler/math fns. The same pointers are fed
/// to both the scalar and the quad path, so the grad_buf (per-triangle, lane-invariant) and
/// texture are identical between them. Only the packing differs (which is what we validate).
/// A null `tex` binds a defined empty descriptor.
const BindCfg = struct {
    tex: ?*const sampler.TexDesc = null,
    grad_buf: ?[]const f32 = null,
};

/// Build the FS's pointer-param list (in entry-ABI order, by `kinds`) from a BindCfg, for
/// the scalar or quad path (the kinds and count are identical for both; widening keeps the
/// ptr params untouched).
fn bindBufs(kinds: []const BufferKind, cfg: BindCfg, empty_tex: *const sampler.TexDesc, out: *[GfxBuffers.max]?[*]const u8) usize {
    for (kinds, 0..) |kind, i| {
        out[i] = switch (kind) {
            .descriptor, .sampler_desc => if (cfg.tex) |t| @ptrCast(t) else @ptrCast(empty_tex),
            .sampler_fn => @ptrFromInt(@intFromPtr(&sampler.sampleTexture)),
            .sampler_cube_fn => @ptrFromInt(@intFromPtr(&sampler.sampleTextureCube)),
            .sampler_shadow_fn => @ptrFromInt(@intFromPtr(&sampler.sampleTextureShadow)),
            .sampler_cube_shadow_fn => @ptrFromInt(@intFromPtr(&sampler.sampleTextureCubeShadow)),
            .sampler_2darray_shadow_fn => @ptrFromInt(@intFromPtr(&sampler.sampleTexture2dArrayShadow)),
            .sampler_gather_fn => @ptrFromInt(@intFromPtr(&sampler.sampleTextureGather)),
            .sampler_fetch_fn => @ptrFromInt(@intFromPtr(&sampler.sampleTextureFetch)),
            .sampler_fetch3_fn => @ptrFromInt(@intFromPtr(&sampler.sampleTextureFetch3D)),
            .grad_buf => if (cfg.grad_buf) |g| @ptrCast(g.ptr) else @ptrCast(empty_tex),
            .math_fn => @ptrFromInt(@intFromPtr(&sampler.mathFn)),
            .discard_fn => @ptrFromInt(@intFromPtr(&discardThunk)),
        };
    }
    return kinds.len;
}

/// Run the scalar FS for one fragment's inputs -> RGBA, and the quad FS for the same 4
/// fragments at once, and assert lane k of the quad output equals the scalar output for
/// fragment k. `fs_code` is raw FS SPIR-V. `inputs4[k]` are fragment k's `nin` varying
/// scalars (k in 0..4). Buffer-free FSes pass the default `cfg`. Bit-exact comparison.
fn assertQuadMatchesScalar(gpa: std.mem.Allocator, fs_code: []const u8, nin: usize, inputs4: [4][]const f32) !void {
    return assertQuadMatchesScalarCfg(gpa, fs_code, nin, inputs4, .{}, 0);
}

fn assertQuadMatchesScalarCfg(gpa: std.mem.Allocator, fs_code: []const u8, nin: usize, inputs4: [4][]const f32, cfg: BindCfg, tol: f32) !void {
    const empty_tex: sampler.TexDesc = .{ .pixels = undefined, .width = 0, .height = 0, .pitch = 0 };

    // Scalar reference: JIT the scalar-rewritten FS, run it per fragment.
    var sfunc = try front.parseSpirv(gpa, fs_code);
    defer sfunc.deinit();
    const sinfo = try rewriteGraphics(&sfunc);
    try std.testing.expectEqual(GfxStage.fragment, sinfo.stage);
    var scompiled = try jitFunction(gpa, &sfunc, "main");
    defer scompiled.deinit();
    const sentry = mainEntry(&scompiled) orelse return error.NoEntry;

    var sbufs: [GfxBuffers.max]?[*]const u8 = .{null} ** GfxBuffers.max;
    const snb = bindBufs(sinfo.buffer_kinds[0..sinfo.buffer_count], cfg, &empty_tex, &sbufs);

    var scalar_out: [4][GfxOut.fragment_len]f32 = undefined;
    for (0..4) |k| {
        scalar_out[k] = [_]f32{0} ** GfxOut.fragment_len;
        scalar_out[k][3] = 1;
        try runGraphicsAt(sentry, sinfo.input_count, snb, inputs4[k][0..nin], sbufs[0..snb], &scalar_out[k]);
    }

    // Quad: JIT the widened FS, run all 4 fragments at once.
    var qfunc = try front.parseSpirv(gpa, fs_code);
    defer qfunc.deinit();
    const qinfo = try rewriteGraphicsQuad(&qfunc);
    try std.testing.expectEqual(sinfo.input_count, qinfo.input_count);
    try std.testing.expectEqual(sinfo.buffer_count, qinfo.buffer_count);
    var qcompiled = try jitFunction(gpa, &qfunc, "main");
    defer qcompiled.deinit();
    const qentry = mainEntry(&qcompiled) orelse return error.NoEntry;

    var qbufs: [GfxBuffers.max]?[*]const u8 = .{null} ** GfxBuffers.max;
    const qnb = bindBufs(qinfo.buffer_kinds[0..qinfo.buffer_count], cfg, &empty_tex, &qbufs);

    // Build the quad inputs: input j is a <4 x f32> of (frag0.j, frag1.j, frag2.j, frag3.j).
    var quad_in: [8]Quad = undefined;
    for (0..qinfo.input_count) |j| {
        quad_in[j] = .{ inputs4[0][j], inputs4[1][j], inputs4[2][j], inputs4[3][j] };
    }
    var qout: [quad_out_len]f32 align(16) = .{0} ** quad_out_len;
    try runGraphicsQuadAt(qentry, qinfo.input_count, qnb, quad_in[0..qinfo.input_count], qbufs[0..qnb], &qout);

    // qout is component-major: out[c*4 + lane]. Compare lane-by-lane to the scalar golden.
    for (0..4) |lane| {
        for (0..4) |c| {
            if (tol == 0) {
                try std.testing.expectEqual(scalar_out[lane][c], qout[c * 4 + lane]);
            } else {
                try std.testing.expectApproxEqAbs(scalar_out[lane][c], qout[c * 4 + lane], tol);
            }
        }
    }
}

test "spirv graphics quad: channel-rotate FS - 4-wide SIMD matches scalar lane-by-lane" {
    const gpa = std.testing.allocator;
    const fs = try buildChannelRotateFs(gpa);
    defer gpa.free(fs);
    try assertQuadMatchesScalar(gpa, std.mem.sliceAsBytes(fs), 3, .{
        &.{ 0.1, 0.5, 0.9 },
        &.{ 0.2, 0.4, 0.8 },
        &.{ 0.7, 0.3, 0.6 },
        &.{ 0.0, 1.0, 0.25 },
    });
}

test "spirv graphics quad: arithmetic FS (add/mul/sqrt) - 4-wide SIMD matches scalar" {
    const gpa = std.testing.allocator;
    const fs = try buildMathFs(gpa);
    defer gpa.free(fs);
    try assertQuadMatchesScalar(gpa, std.mem.sliceAsBytes(fs), 2, .{
        &.{ 0.25, 0.5 },
        &.{ 1.0, 0.0 },
        &.{ 4.0, 2.0 },
        &.{ 0.16, 0.81 },
    });
}

test "spirv graphics quad: masked-select FS (clamp/min/max) - 4-wide SIMD matches scalar" {
    const gpa = std.testing.allocator;
    const fs = try buildClampFs(gpa);
    defer gpa.free(fs);
    // Spread inputs across the clamp range so different lanes hit different branches of
    // the min/max (the masked-blend / bsl path).
    try assertQuadMatchesScalar(gpa, std.mem.sliceAsBytes(fs), 1, .{
        &.{-0.5}, // below 0 -> clamps to 0
        &.{0.3}, // inside -> passes through
        &.{1.7}, // above 1 -> clamps to 1
        &.{0.99}, // inside near top
    });
}

/// Channel-rotate FS: in vec3 vc(loc0) -> out vec4 o = vec4(vc.b, vc.r, vc.g, 1).
fn buildChannelRotateFs(gpa: std.mem.Allocator) ![]u32 {
    const op = spirv.opcodes;
    const sc = op.StorageClass;
    var b = try spirv.binary.Builder.init(gpa, 18);
    defer b.deinit(gpa);
    try b.emit(gpa, op.EntryPoint, &.{ op.ExecutionModel.fragment, 10, 0, 8, 9 });
    try b.emit(gpa, op.Decorate, &.{ 8, op.Decoration.location, 0 });
    try b.emit(gpa, op.Decorate, &.{ 9, op.Decoration.location, 0 });
    try b.emit(gpa, op.TypeVoid, &.{1});
    try b.emit(gpa, op.TypeFunction, &.{ 2, 1 });
    try b.emit(gpa, op.TypeFloat, &.{ 3, 32 });
    try b.emit(gpa, op.TypeVector, &.{ 4, 3, 3 });
    try b.emit(gpa, op.TypeVector, &.{ 5, 3, 4 });
    try b.emit(gpa, op.TypePointer, &.{ 6, sc.input, 4 });
    try b.emit(gpa, op.TypePointer, &.{ 7, sc.output, 5 });
    try b.emit(gpa, op.Constant, &.{ 3, 12, @bitCast(@as(f32, 1.0)) });
    try b.emit(gpa, op.Variable, &.{ 6, 8, sc.input });
    try b.emit(gpa, op.Variable, &.{ 7, 9, sc.output });
    try b.emit(gpa, op.Function, &.{ 1, 10, 0, 2 });
    try b.emit(gpa, op.Label, &.{11});
    try b.emit(gpa, op.Load, &.{ 4, 13, 8 });
    try b.emit(gpa, op.CompositeExtract, &.{ 3, 14, 13, 2 });
    try b.emit(gpa, op.CompositeExtract, &.{ 3, 15, 13, 0 });
    try b.emit(gpa, op.CompositeExtract, &.{ 3, 16, 13, 1 });
    try b.emit(gpa, op.CompositeConstruct, &.{ 5, 17, 14, 15, 16, 12 });
    try b.emit(gpa, op.Store, &.{ 9, 17 });
    try b.emit(gpa, op.Return, &.{});
    try b.emit(gpa, op.FunctionEnd, &.{});
    return gpa.dupe(u32, b.words.items);
}

/// Arithmetic FS: in vec2 v(loc0) -> o = vec4(v.x*v.x + v.y, sqrt(v.x), v.x - v.y, 1).
/// Exercises vector fmul/fadd/fsub/fsqrt + a splat constant.
fn buildMathFs(gpa: std.mem.Allocator) ![]u32 {
    const op = spirv.opcodes;
    const sc = op.StorageClass;
    // ids: void1 fnty2 f32=3 v2=4 v4=5 pIn6 pOut7 in8 out9 main10 entry11 one12
    //      load13 x14 y15 xx16 r0=17 (xx+y) r1=18 sqrt(x) r2=19 (x-y) res20. SET100.
    var b = try spirv.binary.Builder.init(gpa, 21);
    defer b.deinit(gpa);
    try b.emit(gpa, op.EntryPoint, &.{ op.ExecutionModel.fragment, 10, 0, 8, 9 });
    try b.emit(gpa, op.Decorate, &.{ 8, op.Decoration.location, 0 });
    try b.emit(gpa, op.Decorate, &.{ 9, op.Decoration.location, 0 });
    try b.emit(gpa, op.TypeVoid, &.{1});
    try b.emit(gpa, op.TypeFunction, &.{ 2, 1 });
    try b.emit(gpa, op.TypeFloat, &.{ 3, 32 });
    try b.emit(gpa, op.TypeVector, &.{ 4, 3, 2 });
    try b.emit(gpa, op.TypeVector, &.{ 5, 3, 4 });
    try b.emit(gpa, op.TypePointer, &.{ 6, sc.input, 4 });
    try b.emit(gpa, op.TypePointer, &.{ 7, sc.output, 5 });
    try b.emit(gpa, op.Constant, &.{ 3, 12, @bitCast(@as(f32, 1.0)) });
    try b.emit(gpa, op.Variable, &.{ 6, 8, sc.input });
    try b.emit(gpa, op.Variable, &.{ 7, 9, sc.output });
    try b.emit(gpa, op.Function, &.{ 1, 10, 0, 2 });
    try b.emit(gpa, op.Label, &.{11});
    try b.emit(gpa, op.Load, &.{ 4, 13, 8 });
    try b.emit(gpa, op.CompositeExtract, &.{ 3, 14, 13, 0 }); // x
    try b.emit(gpa, op.CompositeExtract, &.{ 3, 15, 13, 1 }); // y
    try b.emit(gpa, op.FMul, &.{ 3, 16, 14, 14 }); // x*x
    try b.emit(gpa, op.FAdd, &.{ 3, 17, 16, 15 }); // x*x + y
    try b.emit(gpa, op.ExtInst, &.{ 3, 18, 100, op.Glsl.sqrt, 14 }); // sqrt(x)
    try b.emit(gpa, op.FSub, &.{ 3, 19, 14, 15 }); // x - y
    try b.emit(gpa, op.CompositeConstruct, &.{ 5, 20, 17, 18, 19, 12 });
    try b.emit(gpa, op.Store, &.{ 9, 20 });
    try b.emit(gpa, op.Return, &.{});
    try b.emit(gpa, op.FunctionEnd, &.{});
    return gpa.dupe(u32, b.words.items);
}

/// Masked-select FS: in float v(loc0) -> o = vec4(clamp(v,0,1), clamp(v,0,1), clamp(v,0,1), 1).
/// clamp lowers to min(max(v,0),1) = nested select(icmp(...)) -> vector fcmp masks + bsl.
fn buildClampFs(gpa: std.mem.Allocator) ![]u32 {
    const op = spirv.opcodes;
    const sc = op.StorageClass;
    // ids: void1 fnty2 f32=3 v4=5 pInF6 pOut7 in8 out9 main10 entry11 one12 zero13
    //      load14 cl15 res16. SET100.
    var b = try spirv.binary.Builder.init(gpa, 17);
    defer b.deinit(gpa);
    try b.emit(gpa, op.EntryPoint, &.{ op.ExecutionModel.fragment, 10, 0, 8, 9 });
    try b.emit(gpa, op.Decorate, &.{ 8, op.Decoration.location, 0 });
    try b.emit(gpa, op.Decorate, &.{ 9, op.Decoration.location, 0 });
    try b.emit(gpa, op.TypeVoid, &.{1});
    try b.emit(gpa, op.TypeFunction, &.{ 2, 1 });
    try b.emit(gpa, op.TypeFloat, &.{ 3, 32 });
    try b.emit(gpa, op.TypeVector, &.{ 5, 3, 4 });
    try b.emit(gpa, op.TypePointer, &.{ 6, sc.input, 3 });
    try b.emit(gpa, op.TypePointer, &.{ 7, sc.output, 5 });
    try b.emit(gpa, op.Constant, &.{ 3, 12, @bitCast(@as(f32, 1.0)) });
    try b.emit(gpa, op.Constant, &.{ 3, 13, @bitCast(@as(f32, 0.0)) });
    try b.emit(gpa, op.Variable, &.{ 6, 8, sc.input });
    try b.emit(gpa, op.Variable, &.{ 7, 9, sc.output });
    try b.emit(gpa, op.Function, &.{ 1, 10, 0, 2 });
    try b.emit(gpa, op.Label, &.{11});
    try b.emit(gpa, op.Load, &.{ 3, 14, 8 }); // v (a float)
    try b.emit(gpa, op.ExtInst, &.{ 3, 15, 100, op.Glsl.f_clamp, 14, 13, 12 }); // clamp(v,0,1)
    try b.emit(gpa, op.CompositeConstruct, &.{ 5, 16, 15, 15, 15, 12 });
    try b.emit(gpa, op.Store, &.{ 9, 16 });
    try b.emit(gpa, op.Return, &.{});
    try b.emit(gpa, op.FunctionEnd, &.{});
    return gpa.dupe(u32, b.words.items);
}

// Heavy-tier quad validation: a textured FS (sampler gather), a derivative FS (grad_buf
// broadcast), a multi-block branchy FS (flatten to select), and the real vkcube FS. Each
// JITs both the scalar golden and the heavy-widened quad and asserts they match lane-by-lane
// when fed 4 distinct fragments (per-lane uv -> per-lane texel; per-triangle grad broadcast;
// per-lane branch outcome via the flattened select).

/// A textured FS: in vec2 uv(loc0) -> o = texture(s, uv). The OpImageSample lowers to the
/// sampler_fn call_indirect + reloads. The heavy widener gathers it per-lane (4 calls, one
/// per fragment's uv), so each lane reads its own texel, exactly matching the scalar per-fragment
/// sample. ids: void1 fnty2 f32=3 v2=4 v4=5 img6 simg7 pUV8 pImg9 pOut10 uvv11 simgv12 outv13
/// main14 entry15 uvld16 imld17 samp18. The sampler descriptor + sampler_fn become 2 ptr params.
fn buildTexturedFs(gpa: std.mem.Allocator) ![]u32 {
    const op = spirv.opcodes;
    const sc = op.StorageClass;
    var b = try spirv.binary.Builder.init(gpa, 19);
    defer b.deinit(gpa);
    try b.emit(gpa, op.EntryPoint, &.{ op.ExecutionModel.fragment, 14, 0, 11, 13 });
    try b.emit(gpa, op.Decorate, &.{ 11, op.Decoration.location, 0 });
    try b.emit(gpa, op.Decorate, &.{ 13, op.Decoration.location, 0 });
    try b.emit(gpa, op.Decorate, &.{ 12, op.Decoration.descriptor_set, 0 });
    try b.emit(gpa, op.Decorate, &.{ 12, op.Decoration.binding, 0 });
    try b.emit(gpa, op.TypeVoid, &.{1});
    try b.emit(gpa, op.TypeFunction, &.{ 2, 1 });
    try b.emit(gpa, op.TypeFloat, &.{ 3, 32 });
    try b.emit(gpa, op.TypeVector, &.{ 4, 3, 2 });
    try b.emit(gpa, op.TypeVector, &.{ 5, 3, 4 });
    try b.emit(gpa, op.TypeImage, &.{ 6, 3, op.Dim.dim_2d, 0, 0, 0, 1, 0 });
    try b.emit(gpa, op.TypeSampledImage, &.{ 7, 6 });
    try b.emit(gpa, op.TypePointer, &.{ 8, sc.input, 4 });
    try b.emit(gpa, op.TypePointer, &.{ 9, sc.uniform_constant, 7 });
    try b.emit(gpa, op.TypePointer, &.{ 10, sc.output, 5 });
    try b.emit(gpa, op.Variable, &.{ 8, 11, sc.input });
    try b.emit(gpa, op.Variable, &.{ 9, 12, sc.uniform_constant });
    try b.emit(gpa, op.Variable, &.{ 10, 13, sc.output });
    try b.emit(gpa, op.Function, &.{ 1, 14, 0, 2 });
    try b.emit(gpa, op.Label, &.{15});
    try b.emit(gpa, op.Load, &.{ 7, 17, 12 }); // sampledImage = load sampler
    try b.emit(gpa, op.Load, &.{ 4, 16, 11 }); // uv
    try b.emit(gpa, op.ImageSampleImplicitLod, &.{ 5, 18, 17, 16 }); // texture(s, uv)
    try b.emit(gpa, op.Store, &.{ 13, 18 });
    try b.emit(gpa, op.Return, &.{});
    try b.emit(gpa, op.FunctionEnd, &.{});
    return gpa.dupe(u32, b.words.items);
}

/// A derivative FS: in vec3 fp(loc0) -> o = vec4(dFdx(fp) + dFdy(fp), 1). The dFdx/dFdy lower
/// to grad_buf loads (lane-invariant per triangle). The heavy widener broadcasts each. ids:
/// void1 fnty2 f32=3 v3=4 v4=5 pIn6 pOut7 in8 out9 main10 entry11 one12 v13 dx14 dy15 sum16 res17.
fn buildDerivFs(gpa: std.mem.Allocator) ![]u32 {
    const op = spirv.opcodes;
    const sc = op.StorageClass;
    var b = try spirv.binary.Builder.init(gpa, 18);
    defer b.deinit(gpa);
    try b.emit(gpa, op.EntryPoint, &.{ op.ExecutionModel.fragment, 10, 0, 8, 9 });
    try b.emit(gpa, op.Decorate, &.{ 8, op.Decoration.location, 0 });
    try b.emit(gpa, op.Decorate, &.{ 9, op.Decoration.location, 0 });
    try b.emit(gpa, op.TypeVoid, &.{1});
    try b.emit(gpa, op.TypeFunction, &.{ 2, 1 });
    try b.emit(gpa, op.TypeFloat, &.{ 3, 32 });
    try b.emit(gpa, op.TypeVector, &.{ 4, 3, 3 });
    try b.emit(gpa, op.TypeVector, &.{ 5, 3, 4 });
    try b.emit(gpa, op.TypePointer, &.{ 6, sc.input, 4 });
    try b.emit(gpa, op.TypePointer, &.{ 7, sc.output, 5 });
    try b.emit(gpa, op.Constant, &.{ 3, 12, @bitCast(@as(f32, 1.0)) });
    try b.emit(gpa, op.Variable, &.{ 6, 8, sc.input });
    try b.emit(gpa, op.Variable, &.{ 7, 9, sc.output });
    try b.emit(gpa, op.Function, &.{ 1, 10, 0, 2 });
    try b.emit(gpa, op.Label, &.{11});
    try b.emit(gpa, op.Load, &.{ 4, 13, 8 }); // fp
    try b.emit(gpa, op.DPdx, &.{ 4, 14, 13 });
    try b.emit(gpa, op.DPdy, &.{ 4, 15, 13 });
    try b.emit(gpa, op.FAdd, &.{ 4, 16, 14, 15 }); // dFdx + dFdy (a vec3)
    try b.emit(gpa, op.CompositeConstruct, &.{ 5, 17, 16, 12 }); // vec4(sum, 1)
    try b.emit(gpa, op.Store, &.{ 9, 17 });
    try b.emit(gpa, op.Return, &.{});
    try b.emit(gpa, op.FunctionEnd, &.{});
    return gpa.dupe(u32, b.words.items);
}

test "spirv graphics quad HEAVY: a pow() FS (math_fn gather) - 4-wide SIMD matches scalar" {
    const gpa = std.testing.allocator;
    const op = spirv.opcodes;
    const sc = op.StorageClass;
    // FS: in float x(loc0) -> o = vec4(pow(x, 2.0), 0, 0, 1). pow -> math_fn call_indirect,
    // Gathered per-lane by the heavy widener (4 scalar pow calls packed back to <4 x f32>).
    var b = try spirv.binary.Builder.init(gpa, 17);
    defer b.deinit(gpa);
    try b.emit(gpa, op.EntryPoint, &.{ op.ExecutionModel.fragment, 9, 0, 7, 8 });
    try b.emit(gpa, op.Decorate, &.{ 7, op.Decoration.location, 0 });
    try b.emit(gpa, op.Decorate, &.{ 8, op.Decoration.location, 0 });
    try b.emit(gpa, op.TypeVoid, &.{1});
    try b.emit(gpa, op.TypeFunction, &.{ 2, 1 });
    try b.emit(gpa, op.TypeFloat, &.{ 3, 32 });
    try b.emit(gpa, op.TypeVector, &.{ 4, 3, 4 });
    try b.emit(gpa, op.TypePointer, &.{ 5, sc.input, 3 });
    try b.emit(gpa, op.TypePointer, &.{ 6, sc.output, 4 });
    try b.emit(gpa, op.Constant, &.{ 3, 11, @bitCast(@as(f32, 2.0)) });
    try b.emit(gpa, op.Constant, &.{ 3, 12, @bitCast(@as(f32, 0.0)) });
    try b.emit(gpa, op.Constant, &.{ 3, 13, @bitCast(@as(f32, 1.0)) });
    try b.emit(gpa, op.Variable, &.{ 5, 7, sc.input });
    try b.emit(gpa, op.Variable, &.{ 6, 8, sc.output });
    try b.emit(gpa, op.Function, &.{ 1, 9, 0, 2 });
    try b.emit(gpa, op.Label, &.{10});
    try b.emit(gpa, op.Load, &.{ 3, 14, 7 });
    try b.emit(gpa, op.ExtInst, &.{ 3, 15, 100, op.Glsl.pow, 14, 11 });
    try b.emit(gpa, op.CompositeConstruct, &.{ 4, 16, 15, 12, 12, 13 });
    try b.emit(gpa, op.Store, &.{ 8, 16 });
    try b.emit(gpa, op.Return, &.{});
    try b.emit(gpa, op.FunctionEnd, &.{});
    const fs = try gpa.dupe(u32, b.words.items);
    defer gpa.free(fs);
    try assertQuadMatchesScalarCfg(gpa, std.mem.sliceAsBytes(fs), 1, .{
        &.{0.5}, &.{2.0}, &.{3.0}, &.{1.5},
    }, .{}, 1e-5);
}

test "spirv graphics quad REPRO: glmark2 phong (normalize+dot+max+pow) - sweep Normal for SIMD divergence" {
    const gpa = std.testing.allocator;
    const glsl = @import("../../glsl.zig");

    // The glmark2 light-advanced.frag construct (Blinn-Phong), uniforms folded to local
    // constants so the only per-fragment input is the interpolated `Normal` varying. The
    // pow() forces the heavy widener (math_fn gather + flattened max-selects), which is the
    // exact path the real shader takes and where the per-quad speckle lives.
    const src =
        \\varying vec3 Normal;
        \\void main(void) {
        \\    vec3 LightSourcePosition = vec3(20.0, 20.0, 10.0);
        \\    vec3 LightSourceHalfVector = vec3(0.0, 0.30151, 0.95346);
        \\    vec3 N = normalize(Normal);
        \\    vec3 L = normalize(LightSourcePosition);
        \\    vec3 H = normalize(LightSourceHalfVector);
        \\    float d = max(dot(N, L), 0.0);
        \\    float sp = pow(max(dot(N, H), 0.0), 100.0);
        \\    vec3 diffuse = vec3(0.0, 0.0, 1.0) * 0.8 * d;
        \\    vec3 ambient = vec3(0.1);
        \\    vec3 specular = vec3(0.8) * sp;
        \\    gl_FragColor = vec4(ambient + specular + diffuse, 1.0);
        \\}
    ;
    const fs = try glsl.compileForStage(gpa, src, .fragment);
    defer gpa.free(fs);

    var sfunc = try front.parseSpirv(gpa, fs);
    defer sfunc.deinit();
    const sinfo = try rewriteGraphics(&sfunc);
    var scompiled = try jitFunction(gpa, &sfunc, "main");
    defer scompiled.deinit();
    const sentry = mainEntry(&scompiled) orelse return error.NoEntry;

    var qfunc = try front.parseSpirv(gpa, fs);
    defer qfunc.deinit();
    const qinfo = try rewriteGraphicsQuad(&qfunc);
    var qcompiled = try jitFunction(gpa, &qfunc, "main");
    defer qcompiled.deinit();
    const qentry = mainEntry(&qcompiled) orelse return error.NoEntry;

    const empty_tex: sampler.TexDesc = .{ .pixels = undefined, .width = 0, .height = 0, .pitch = 0 };
    var sbufs: [GfxBuffers.max]?[*]const u8 = .{null} ** GfxBuffers.max;
    const snb = bindBufs(sinfo.buffer_kinds[0..sinfo.buffer_count], .{}, &empty_tex, &sbufs);
    var qbufs: [GfxBuffers.max]?[*]const u8 = .{null} ** GfxBuffers.max;
    const qnb = bindBufs(qinfo.buffer_kinds[0..qinfo.buffer_count], .{}, &empty_tex, &qbufs);

    const ni = sinfo.input_count;
    try std.testing.expectEqual(@as(usize, 3), ni);

    var worst: f32 = 0;
    var worst_in: [4][3]f32 = undefined;
    var worst_s: [4][4]f32 = undefined;
    var worst_q: [quad_out_len]f32 = undefined;

    const steps = [_]f32{ -1.0, -0.6, -0.3, -0.1, 0.0, 0.1, 0.3, 0.6, 1.0 };
    var pts: [4][3]f32 = undefined;
    var fill: usize = 0;

    for (steps) |x| for (steps) |y| for (steps) |z| {
        pts[fill] = .{ x, y, z };
        fill += 1;
        if (fill < 4) continue;
        fill = 0;

        var scal: [4][4]f32 = undefined;
        for (0..4) |k| {
            scal[k] = .{ 0, 0, 0, 1 };
            try runGraphicsAt(sentry, sinfo.input_count, snb, pts[k][0..ni], sbufs[0..snb], &scal[k]);
        }
        var quad_in: [3]Quad = undefined;
        for (0..ni) |j| quad_in[j] = .{ pts[0][j], pts[1][j], pts[2][j], pts[3][j] };
        var qout: [quad_out_len]f32 align(16) = .{0} ** quad_out_len;
        try runGraphicsQuadAt(qentry, qinfo.input_count, qnb, quad_in[0..ni], qbufs[0..qnb], &qout);

        for (0..4) |lane| for (0..4) |c| {
            const diff = @abs(scal[lane][c] - qout[c * 4 + lane]);
            if (diff > worst) {
                worst = diff;
                worst_in = pts;
                worst_s = scal;
                worst_q = qout;
            }
        };
    };

    if (worst > 1e-3) {
        std.debug.print("\nPHONG QUAD DIVERGENCE worst={d}\n", .{worst});
        for (0..4) |k| std.debug.print("  lane{d} in=({d:.3},{d:.3},{d:.3}) scalar=({d:.4},{d:.4},{d:.4}) quad=({d:.4},{d:.4},{d:.4})\n", .{
            k,                  worst_in[k][0],     worst_in[k][1], worst_in[k][2],
            worst_s[k][0],      worst_s[k][1],      worst_s[k][2],  worst_q[0 * 4 + k],
            worst_q[1 * 4 + k], worst_q[2 * 4 + k],
        });
        return error.QuadDiverges;
    }
}

test "spirv graphics quad HEAVY: a TEXTURED FS (sampler gather) - 4-wide SIMD matches scalar" {
    const gpa = std.testing.allocator;
    const fs = try buildTexturedFs(gpa);
    defer gpa.free(fs);
    // A 2x2 RGBA8 texture: distinct texel per quadrant, so the 4 lanes' uv hit 4 colors.
    var px = [_]u8{
        255, 0, 0, 255, 0, 255, 0, 255, // (red)  (green)
        0, 0, 255, 255, 255, 255, 0, 255, // (blue) (yellow)
    };
    var tex = sampler.TexDesc{ .pixels = &px, .width = 2, .height = 2, .pitch = 8, .filter = .nearest };
    // 4 lanes sample the 4 quadrant centers -> 4 distinct texels; gather must keep them apart.
    try assertQuadMatchesScalarCfg(gpa, std.mem.sliceAsBytes(fs), 2, .{
        &.{ 0.25, 0.25 }, // red
        &.{ 0.75, 0.25 }, // green
        &.{ 0.25, 0.75 }, // blue
        &.{ 0.75, 0.75 }, // yellow
    }, .{ .tex = &tex }, 0);
}

test "spirv graphics quad HEAVY: a DERIVATIVE FS (grad_buf broadcast) - 4-wide SIMD matches scalar" {
    const gpa = std.testing.allocator;
    const fs = try buildDerivFs(gpa);
    defer gpa.free(fs);
    // The grad_buf holds the per-triangle screen-space gradients (lane-invariant): 3 dFdx + 3
    // dFdy entries. The heavy widener broadcasts each grad_buf load to all 4 lanes. Distinct
    // varying inputs per lane (they pass through unused here, but exercise the input packing).
    const grad = [_]f32{ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6 } ++ [_]f32{0} ** (MAX_GRAD_INPUTS - 6);
    try assertQuadMatchesScalarCfg(gpa, std.mem.sliceAsBytes(fs), 3, .{
        &.{ 0.1, 0.2, 0.3 },
        &.{ 0.4, 0.5, 0.6 },
        &.{ 0.7, 0.8, 0.9 },
        &.{ 0.0, 0.5, 1.0 },
    }, .{ .grad_buf = &grad }, 0);
}

test "spirv graphics quad HEAVY: the REAL vkcube FS - 4-wide SIMD matches scalar lane-by-lane" {
    const gpa = std.testing.allocator;
    const raw = @embedFile("testdata_vkcube_fs.spv");

    // The grad_buf holds a non-degenerate per-triangle gradient so normalize(cross(dFdx,dFdy))
    // is well-defined. We read the grad layout off the scalar info (same for the quad).
    var sf = try front.parseSpirv(gpa, raw);
    defer sf.deinit();
    const si = try rewriteGraphics(&sf);
    var min_vi: u32 = std.math.maxInt(u32);
    for (si.grads[0..si.grad_count]) |g| min_vi = @min(min_vi, g.varying_index);
    var grad_buf = [_]f32{0} ** MAX_GRAD_INPUTS;
    for (si.grads[0..si.grad_count], 0..) |g, i| {
        grad_buf[i] = switch (g.axis) {
            .x => if (g.varying_index == min_vi) 1.0 else 0.0,
            .y => if (g.varying_index == min_vi + 1) 1.0 else 0.0,
        };
    }
    var px = [_]u8{ 200, 180, 160, 255 } ** 4; // 2x2 textured (non-white so the sample varies)
    var tex = sampler.TexDesc{ .pixels = &px, .width = 2, .height = 2, .pitch = 8, .filter = .nearest };

    // 4 distinct fragments (frag_pos + texcoord varyings) so the lanes diverge through the
    // texture sample, the normal/lighting math, and the per-channel linearToSrgb branch. A
    // small float tolerance: the host pow() in math_fn is the same fn per lane, so the only
    // slack is FP reassociation across the packed vs scalar arithmetic (none expected, but
    // allow 1e-4 for safety).
    try assertQuadMatchesScalarCfg(gpa, raw, @min(si.input_count, 8), .{
        &.{ 0.2, 0.3, 0.5, 0.25, 0.25, 0.1, 0.4, 0.6 },
        &.{ 0.6, 0.1, 0.2, 0.75, 0.25, 0.3, 0.2, 0.5 },
        &.{ 0.1, 0.7, 0.4, 0.25, 0.75, 0.5, 0.6, 0.2 },
        &.{ 0.4, 0.4, 0.8, 0.75, 0.75, 0.2, 0.1, 0.9 },
    }, .{ .tex = &tex, .grad_buf = &grad_buf }, 1e-4);
}

test "spirv graphics: a flat-normal FS (dFdx/dFdy + cross + normalize) JITs + runs via grad_buf" {
    const gpa = std.testing.allocator;
    const op = spirv.opcodes;
    const sc = op.StorageClass;

    // FS: layout(location=0) in vec3 frag_pos; layout(location=0) out vec4 o;
    //     void main(){ o = vec4(normalize(cross(dFdx(frag_pos), dFdy(frag_pos))), 1.0); }
    // The dFdx/dFdy of the varying lower to grad_buf loads (the rasterizer supplies the
    // per-triangle gradients). Cross + normalize lower to native arithmetic.
    // ids: void=1 fnty=2 f32=3 v3=4 v4=5 pIn=6 pOut=7 in=8 out=9 main=10 entry=11
    //      one=12 v=13 dx=14 dy=15 cr=16 nm=17 res=18. SET=100.
    var b = try spirv.binary.Builder.init(gpa, 19);
    defer b.deinit(gpa);
    try b.emit(gpa, op.EntryPoint, &.{ op.ExecutionModel.fragment, 10, 0, 8, 9 });
    try b.emit(gpa, op.Decorate, &.{ 8, op.Decoration.location, 0 });
    try b.emit(gpa, op.Decorate, &.{ 9, op.Decoration.location, 0 });
    try b.emit(gpa, op.TypeVoid, &.{1});
    try b.emit(gpa, op.TypeFunction, &.{ 2, 1 });
    try b.emit(gpa, op.TypeFloat, &.{ 3, 32 });
    try b.emit(gpa, op.TypeVector, &.{ 4, 3, 3 });
    try b.emit(gpa, op.TypeVector, &.{ 5, 3, 4 });
    try b.emit(gpa, op.TypePointer, &.{ 6, sc.input, 4 });
    try b.emit(gpa, op.TypePointer, &.{ 7, sc.output, 5 });
    try b.emit(gpa, op.Constant, &.{ 3, 12, @bitCast(@as(f32, 1.0)) });
    try b.emit(gpa, op.Variable, &.{ 6, 8, sc.input });
    try b.emit(gpa, op.Variable, &.{ 7, 9, sc.output });
    try b.emit(gpa, op.Function, &.{ 1, 10, 0, 2 });
    try b.emit(gpa, op.Label, &.{11});
    try b.emit(gpa, op.Load, &.{ 4, 13, 8 }); // frag_pos
    try b.emit(gpa, op.DPdx, &.{ 4, 14, 13 }); // dFdx(frag_pos)
    try b.emit(gpa, op.DPdy, &.{ 4, 15, 13 }); // dFdy(frag_pos)
    try b.emit(gpa, op.ExtInst, &.{ 4, 16, 100, op.Glsl.cross, 14, 15 }); // cross(dx, dy)
    try b.emit(gpa, op.ExtInst, &.{ 4, 17, 100, op.Glsl.normalize, 16 }); // normalize(...)
    try b.emit(gpa, op.CompositeConstruct, &.{ 5, 18, 17, 12 }); // vec4(normal, 1.0)
    try b.emit(gpa, op.Store, &.{ 9, 18 });
    try b.emit(gpa, op.Return, &.{});
    try b.emit(gpa, op.FunctionEnd, &.{});

    var func = try front.parseSpirv(gpa, std.mem.sliceAsBytes(b.words.items));
    defer func.deinit();

    const info = try rewriteGraphics(&func);
    try std.testing.expectEqual(GfxStage.fragment, info.stage);
    try std.testing.expectEqual(@as(usize, 3), info.input_count); // vec3 varying (not gradients)
    try std.testing.expectEqual(@as(usize, 6), info.grad_count); // 3 dFdx + 3 dFdy
    try std.testing.expectEqual(@as(usize, 1), info.buffer_count); // just the grad_buf
    try std.testing.expectEqual(BufferKind.grad_buf, info.buffer_kinds[0]);

    var compiled = try jitFunction(gpa, &func, "main");
    defer compiled.deinit();

    // Fill grad_buf so dFdx(frag_pos) = (1,0,0) and dFdy(frag_pos) = (0,1,0) (a flat plane
    // in the XY screen). cross = (0,0,1); normalize = (0,0,1); o = (0,0,1,1). The grad_buf
    // index order is the order the derivatives were taken (info.grads).
    var grad_buf = [_]f32{0} ** MAX_GRAD_INPUTS;
    for (info.grads[0..info.grad_count], 0..) |g, i| {
        // varying_index identifies the component; axis the gradient. dFdx of comp c -> the
        // c-th basis along x; dFdy of comp c -> the c-th basis along y.
        grad_buf[i] = switch (g.axis) {
            .x => if (g.varying_index == 0) 1.0 else 0.0,
            .y => if (g.varying_index == 1) 1.0 else 0.0,
        };
    }
    const gbptr: [*]const u8 = @ptrCast(&grad_buf);
    var out = [_]f32{0} ** GfxOut.fragment_len;
    // The varyings (frag_pos) are unused for the normal here, pass zeros.
    try runGraphics(&compiled, info.input_count, info.buffer_count, &.{ 0, 0, 0 }, &.{gbptr}, &out);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), out[0], 1e-5); // normal.x
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), out[1], 1e-5); // normal.y
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), out[2], 1e-5); // normal.z (+Z)
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), out[3], 1e-5); // a
}

test "spirv graphics: vkcube's real fragment shader inlines, lowers, JITs, and EXECUTES" {
    const gpa = std.testing.allocator;
    const raw = @embedFile("testdata_vkcube_fs.spv");

    // The FS calls helper functions (linearToSrgb) via OpFunctionCall. ParseSpirv inlines
    // them, lowers the sampler + dFdx/dFdy + pow, and JIT-compiles to native code. The inlined
    // linearToSrgb has branching control flow, so the FS's color-output stores land in a
    // non-entry block. rewriteGraphics now re-points tagged output stores in every block (not
    // just the entry), so the stores hit `outbuf + idx*4` instead of the literal slot-iconst
    // (a `color_out=0` store to the literal address 0 was the `str s15, [x28]` x28==0 fault).
    var func = try front.parseSpirv(gpa, raw);
    defer func.deinit();
    const info = try rewriteGraphics(&func);
    try std.testing.expectEqual(GfxStage.fragment, info.stage);
    // sampler descriptor + grad_buf + sampler_fn + math_fn = 4 buffer pointer params.
    try std.testing.expectEqual(@as(usize, 4), info.buffer_count);
    try std.testing.expect(info.input_count <= 8);

    var compiled = try jitFunction(gpa, &func, "main");
    defer compiled.deinit();

    // Bind the four buffers in entry-ABI order (info.buffer_kinds): a real 2x2 white texture
    // descriptor, the host sampler_fn, the grad_buf, and the host math_fn. Execute the FS
    // and assert it writes a finite RGBA color to the output buffer without faulting (the bug
    // faulted before producing any output). The exact color depends on the inlined sRGB curve.
    // We assert it ran (all 4 components finite, alpha written). The proof is no fault.
    var px = [_]u8{ 255, 255, 255, 255 } ** 4; // 2x2 white RGBA8
    var tex = sampler.TexDesc{ .pixels = &px, .width = 2, .height = 2, .pitch = 8, .filter = .nearest };
    const tex_ptr: [*]const u8 = @ptrCast(&tex);
    const samplerfn: [*]const u8 = @ptrFromInt(@intFromPtr(&sampler.sampleTexture));
    const samplercubefn: [*]const u8 = @ptrFromInt(@intFromPtr(&sampler.sampleTextureCube));
    const samplershadowfn: [*]const u8 = @ptrFromInt(@intFromPtr(&sampler.sampleTextureShadow));
    const samplercubeshadowfn: [*]const u8 = @ptrFromInt(@intFromPtr(&sampler.sampleTextureCubeShadow));
    const sampler2darrayshadowfn: [*]const u8 = @ptrFromInt(@intFromPtr(&sampler.sampleTexture2dArrayShadow));
    const samplergatherfn: [*]const u8 = @ptrFromInt(@intFromPtr(&sampler.sampleTextureGather));
    const samplerfetchfn: [*]const u8 = @ptrFromInt(@intFromPtr(&sampler.sampleTextureFetch));
    const samplerfetch3fn: [*]const u8 = @ptrFromInt(@intFromPtr(&sampler.sampleTextureFetch3D));
    const mathfn: [*]const u8 = @ptrFromInt(@intFromPtr(&sampler.mathFn));
    // Fill grad_buf for a non-degenerate flat triangle so normalize(cross(dFdx,dFdy)) is
    // well-defined (all-zero gradients give cross=0 -> normalize 0/0 = NaN). frag_pos is the
    // varying with derivatives. Make dFdx(frag_pos) point along its first component and
    // dFdy(frag_pos) along its second (the lowest two grad varying indices), so the cross
    // product is a unit normal. info.grads lists the (varying_index, axis) per grad slot.
    var min_vi: u32 = std.math.maxInt(u32);
    for (info.grads[0..info.grad_count]) |g| min_vi = @min(min_vi, g.varying_index);
    var grad_buf = [_]f32{0} ** MAX_GRAD_INPUTS;
    for (info.grads[0..info.grad_count], 0..) |g, i| {
        grad_buf[i] = switch (g.axis) {
            .x => if (g.varying_index == min_vi) 1.0 else 0.0,
            .y => if (g.varying_index == min_vi + 1) 1.0 else 0.0,
        };
    }
    const gbptr: [*]const u8 = @ptrCast(&grad_buf);

    var bufs: [GfxBuffers.max]?[*]const u8 = undefined;
    for (info.buffer_kinds[0..info.buffer_count], 0..) |kind, i| {
        bufs[i] = switch (kind) {
            .descriptor, .sampler_desc => tex_ptr,
            .sampler_fn => samplerfn,
            .sampler_cube_fn => samplercubefn,
            .sampler_shadow_fn => samplershadowfn,
            .sampler_cube_shadow_fn => samplercubeshadowfn,
            .sampler_2darray_shadow_fn => sampler2darrayshadowfn,
            .sampler_gather_fn => samplergatherfn,
            .sampler_fetch_fn => samplerfetchfn,
            .sampler_fetch3_fn => samplerfetch3fn,
            .grad_buf => gbptr,
            .math_fn => mathfn,
            .discard_fn => @ptrFromInt(@intFromPtr(&discardThunk)),
        };
    }

    const inputs = [_]f32{0.5} ** 8; // benign finite varyings (frag_pos + texcoord = white texel)
    const sentinel: f32 = -123456.0;
    var out = [_]f32{sentinel} ** GfxOut.fragment_len;
    try runGraphics(&compiled, info.input_count, info.buffer_count, inputs[0..info.input_count], bufs[0..info.buffer_count], &out);
    // The fix: the FS's four color-output stores live in a non-entry block (the inlined
    // branching linearToSrgb returns from two arms, merged by a phi). They are now re-pointed to
    // `outbuf + idx*4` in every block, so all four RGBA slots are written. Before the fix they
    // kept their literal slot-iconst pointers and a `color_out=0` store to address 0 faulted
    // (`str s15, [x28]`, x28==0). Reaching this assert means the crash is gone.
    for (0..4) |c| try std.testing.expect(out[c] != sentinel);
    // It also computes the correct shaded color: light = dot((0.424,0.566,0.707),(0,0,1)) = 0.707,
    // white texel * 0.707 = 0.707, linearToSrgb(0.707) = 1.055*pow(0.707,0.4167)-0.055 ~= 0.858.
    for (0..3) |c| try std.testing.expectApproxEqAbs(@as(f32, 0.858), out[c], 1e-2);
    try std.testing.expectApproxEqAbs(@as(f32, 0.707), out[3], 1e-2); // alpha = linearToSrgb(0.707) input is 0.707
}

test "spirv graphics: a FS using pow() JITs, calls the host math_fn, runs natively" {
    const gpa = std.testing.allocator;
    const op = spirv.opcodes;
    const sc = op.StorageClass;

    // FS: layout(location=0) in float x; layout(location=0) out vec4 o;
    //     void main(){ o = vec4(pow(x, 2.0), 0, 0, 1); }
    // pow lowers to a call through the synthesized host math_fn pointer param.
    // ids: void=1 fnty=2 f32=3 v4=4 pIn=5 pOut=6 in=7 out=8 main=9 entry=10
    //      two=11 zero=12 one=13 x=14 p=15 res=16. SET=100.
    var b = try spirv.binary.Builder.init(gpa, 17);
    defer b.deinit(gpa);
    try b.emit(gpa, op.EntryPoint, &.{ op.ExecutionModel.fragment, 9, 0, 7, 8 });
    try b.emit(gpa, op.Decorate, &.{ 7, op.Decoration.location, 0 });
    try b.emit(gpa, op.Decorate, &.{ 8, op.Decoration.location, 0 });
    try b.emit(gpa, op.TypeVoid, &.{1});
    try b.emit(gpa, op.TypeFunction, &.{ 2, 1 });
    try b.emit(gpa, op.TypeFloat, &.{ 3, 32 });
    try b.emit(gpa, op.TypeVector, &.{ 4, 3, 4 });
    try b.emit(gpa, op.TypePointer, &.{ 5, sc.input, 3 });
    try b.emit(gpa, op.TypePointer, &.{ 6, sc.output, 4 });
    try b.emit(gpa, op.Constant, &.{ 3, 11, @bitCast(@as(f32, 2.0)) });
    try b.emit(gpa, op.Constant, &.{ 3, 12, @bitCast(@as(f32, 0.0)) });
    try b.emit(gpa, op.Constant, &.{ 3, 13, @bitCast(@as(f32, 1.0)) });
    try b.emit(gpa, op.Variable, &.{ 5, 7, sc.input });
    try b.emit(gpa, op.Variable, &.{ 6, 8, sc.output });
    try b.emit(gpa, op.Function, &.{ 1, 9, 0, 2 });
    try b.emit(gpa, op.Label, &.{10});
    try b.emit(gpa, op.Load, &.{ 3, 14, 7 }); // x
    try b.emit(gpa, op.ExtInst, &.{ 3, 15, 100, op.Glsl.pow, 14, 11 }); // pow(x, 2.0)
    try b.emit(gpa, op.CompositeConstruct, &.{ 4, 16, 15, 12, 12, 13 }); // vec4(p,0,0,1)
    try b.emit(gpa, op.Store, &.{ 8, 16 });
    try b.emit(gpa, op.Return, &.{});
    try b.emit(gpa, op.FunctionEnd, &.{});

    var func = try front.parseSpirv(gpa, std.mem.sliceAsBytes(b.words.items));
    defer func.deinit();

    const info = try rewriteGraphics(&func);
    try std.testing.expectEqual(GfxStage.fragment, info.stage);
    try std.testing.expectEqual(@as(usize, 1), info.buffer_count); // just the math_fn pointer
    try std.testing.expectEqual(BufferKind.math_fn, info.buffer_kinds[0]);

    var compiled = try jitFunction(gpa, &func, "main");
    defer compiled.deinit();

    const mathfn: [*]const u8 = @ptrFromInt(@intFromPtr(&sampler.mathFn));
    var out = [_]f32{0} ** GfxOut.fragment_len;
    // x = 3.0 -> pow(3,2) = 9.0.
    try runGraphics(&compiled, info.input_count, info.buffer_count, &.{3.0}, &.{mathfn}, &out);
    try std.testing.expectApproxEqAbs(@as(f32, 9.0), out[0], 1e-3);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), out[1], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), out[3], 1e-5);
}

test "spirv graphics: a discarding FS calls the host discard_fn and sets the kill flag" {
    const gpa = std.testing.allocator;
    const op = spirv.opcodes;

    // FS: layout(location=0) out vec4 o; void main() { o = vec4(1,0,0,1); discard; }
    // OpKill synthesizes a discard_fn pointer param the FS calls. The rasterizer's thunk
    // sets the per-thread discard flag (which suppresses the color/depth write).
    // ids: void=1 fnty=2 f32=3 v4=4 pOut=5 out=6 main=7 entry=8 one=9 zero=10 col=11.
    const sc = op.StorageClass;
    var b = try spirv.binary.Builder.init(gpa, 12);
    defer b.deinit(gpa);
    try b.emit(gpa, op.EntryPoint, &.{ op.ExecutionModel.fragment, 7, 0, 6 });
    try b.emit(gpa, op.Decorate, &.{ 6, op.Decoration.location, 0 });
    try b.emit(gpa, op.TypeVoid, &.{1});
    try b.emit(gpa, op.TypeFunction, &.{ 2, 1 });
    try b.emit(gpa, op.TypeFloat, &.{ 3, 32 });
    try b.emit(gpa, op.TypeVector, &.{ 4, 3, 4 });
    try b.emit(gpa, op.TypePointer, &.{ 5, sc.output, 4 });
    try b.emit(gpa, op.Constant, &.{ 3, 9, @bitCast(@as(f32, 1.0)) });
    try b.emit(gpa, op.Constant, &.{ 3, 10, @bitCast(@as(f32, 0.0)) });
    try b.emit(gpa, op.Variable, &.{ 5, 6, sc.output });
    try b.emit(gpa, op.Function, &.{ 1, 7, 0, 2 });
    try b.emit(gpa, op.Label, &.{8});
    try b.emit(gpa, op.CompositeConstruct, &.{ 4, 11, 9, 10, 10, 9 }); // vec4(1,0,0,1)
    try b.emit(gpa, op.Store, &.{ 6, 11 });
    try b.emit(gpa, op.Kill, &.{});
    try b.emit(gpa, op.FunctionEnd, &.{});

    var func = try front.parseSpirv(gpa, std.mem.sliceAsBytes(b.words.items));
    defer func.deinit();

    const info = try rewriteGraphics(&func);
    try std.testing.expectEqual(GfxStage.fragment, info.stage);
    try std.testing.expectEqual(@as(usize, 1), info.buffer_count); // just the discard_fn pointer
    try std.testing.expectEqual(BufferKind.discard_fn, info.buffer_kinds[0]);

    var compiled = try jitFunction(gpa, &func, "main");
    defer compiled.deinit();

    const dfn: [*]const u8 = @ptrFromInt(@intFromPtr(&discardThunk));
    var out = [_]f32{0} ** GfxOut.fragment_len;
    discardReset();
    try std.testing.expect(!discarded());
    try runGraphics(&compiled, info.input_count, info.buffer_count, &.{}, &.{dfn}, &out);
    try std.testing.expect(discarded()); // the FS killed the fragment
}

test "spirv graphics: a FS reading gl_FragCoord classifies its builtin inputs + passes them through" {
    const gpa = std.testing.allocator;
    const op = spirv.opcodes;
    const sc = op.StorageClass;

    // FS: layout(location=0) out vec4 o; void main(){ o = gl_FragCoord; }
    // gl_FragCoord is a vec4 BuiltIn input. The 4 components become builtin input params
    // the rasterizer fills with the fragment window position.
    // ids: void=1 fnty=2 f32=3 v4=4 pIn=5 pOut=6 fc=7 out=8 main=9 entry=10 v=11.
    var b = try spirv.binary.Builder.init(gpa, 12);
    defer b.deinit(gpa);
    try b.emit(gpa, op.EntryPoint, &.{ op.ExecutionModel.fragment, 9, 0, 7, 8 });
    try b.emit(gpa, op.Decorate, &.{ 7, op.Decoration.builtin, op.BuiltIn.frag_coord });
    try b.emit(gpa, op.Decorate, &.{ 8, op.Decoration.location, 0 });
    try b.emit(gpa, op.TypeVoid, &.{1});
    try b.emit(gpa, op.TypeFunction, &.{ 2, 1 });
    try b.emit(gpa, op.TypeFloat, &.{ 3, 32 });
    try b.emit(gpa, op.TypeVector, &.{ 4, 3, 4 });
    try b.emit(gpa, op.TypePointer, &.{ 5, sc.input, 4 });
    try b.emit(gpa, op.TypePointer, &.{ 6, sc.output, 4 });
    try b.emit(gpa, op.Variable, &.{ 5, 7, sc.input });
    try b.emit(gpa, op.Variable, &.{ 6, 8, sc.output });
    try b.emit(gpa, op.Function, &.{ 1, 9, 0, 2 });
    try b.emit(gpa, op.Label, &.{10});
    try b.emit(gpa, op.Load, &.{ 4, 11, 7 });
    try b.emit(gpa, op.Store, &.{ 8, 11 });
    try b.emit(gpa, op.Return, &.{});
    try b.emit(gpa, op.FunctionEnd, &.{});

    var func = try front.parseSpirv(gpa, std.mem.sliceAsBytes(b.words.items));
    defer func.deinit();

    const info = try rewriteGraphics(&func);
    try std.testing.expectEqual(@as(usize, 4), info.input_count); // the 4 gl_FragCoord components
    try std.testing.expectEqual(@as(usize, 0), info.buffer_count);
    // Each input carries the gl_FragCoord sentinel slot (component 0..3).
    for (0..4) |c| try std.testing.expectEqual(FRAG_COORD_INPUT_BASE + @as(u32, @intCast(c)), info.input_slots[c]);

    var compiled = try jitFunction(gpa, &func, "main");
    defer compiled.deinit();

    // The rasterizer fills the 4 inputs with the fragment window position. The FS passes
    // them through to the color output.
    var out = [_]f32{0} ** GfxOut.fragment_len;
    try runGraphics(&compiled, info.input_count, info.buffer_count, &.{ 12.5, 34.5, 0.25, 1.0 }, &.{}, &out);
    try std.testing.expectApproxEqAbs(@as(f32, 12.5), out[0], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 34.5), out[1], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 0.25), out[2], 1e-4);
}

test "spirv graphics: a FS writing gl_FragDepth routes it to the depth output slot" {
    const gpa = std.testing.allocator;
    const op = spirv.opcodes;
    const sc = op.StorageClass;

    // FS: layout(location=0) out vec4 o; void main(){ o=vec4(1,0,0,1); gl_FragDepth=0.25; }
    // ids: void=1 fnty=2 f32=3 v4=4 pOut=5 pDepth=6 out=7 fd=8 main=9 entry=10
    //      one=11 zero=12 quarter=13 col=14.
    var b = try spirv.binary.Builder.init(gpa, 15);
    defer b.deinit(gpa);
    try b.emit(gpa, op.EntryPoint, &.{ op.ExecutionModel.fragment, 9, 0, 7, 8 });
    try b.emit(gpa, op.Decorate, &.{ 7, op.Decoration.location, 0 });
    try b.emit(gpa, op.Decorate, &.{ 8, op.Decoration.builtin, op.BuiltIn.frag_depth });
    try b.emit(gpa, op.TypeVoid, &.{1});
    try b.emit(gpa, op.TypeFunction, &.{ 2, 1 });
    try b.emit(gpa, op.TypeFloat, &.{ 3, 32 });
    try b.emit(gpa, op.TypeVector, &.{ 4, 3, 4 });
    try b.emit(gpa, op.TypePointer, &.{ 5, sc.output, 4 });
    try b.emit(gpa, op.TypePointer, &.{ 6, sc.output, 3 });
    try b.emit(gpa, op.Constant, &.{ 3, 11, @bitCast(@as(f32, 1.0)) });
    try b.emit(gpa, op.Constant, &.{ 3, 12, @bitCast(@as(f32, 0.0)) });
    try b.emit(gpa, op.Constant, &.{ 3, 13, @bitCast(@as(f32, 0.25)) });
    try b.emit(gpa, op.Variable, &.{ 5, 7, sc.output });
    try b.emit(gpa, op.Variable, &.{ 6, 8, sc.output });
    try b.emit(gpa, op.Function, &.{ 1, 9, 0, 2 });
    try b.emit(gpa, op.Label, &.{10});
    try b.emit(gpa, op.CompositeConstruct, &.{ 4, 14, 11, 12, 12, 11 });
    try b.emit(gpa, op.Store, &.{ 7, 14 });
    try b.emit(gpa, op.Store, &.{ 8, 13 });
    try b.emit(gpa, op.Return, &.{});
    try b.emit(gpa, op.FunctionEnd, &.{});

    var func = try front.parseSpirv(gpa, std.mem.sliceAsBytes(b.words.items));
    defer func.deinit();

    const info = try rewriteGraphics(&func);
    try std.testing.expect(info.writes_frag_depth);

    var compiled = try jitFunction(gpa, &func, "main");
    defer compiled.deinit();

    var out = [_]f32{0} ** GfxOut.fragment_len;
    try runGraphics(&compiled, info.input_count, info.buffer_count, &.{}, &.{}, &out);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), out[0], 1e-4); // color.r
    try std.testing.expectApproxEqAbs(@as(f32, 0.25), out[GfxOut.frag_depth_index], 1e-4); // gl_FragDepth
}

test "spirv graphics: a UBO mat4-MVP vertex shader JITs, reads the uniform, runs natively" {
    const gpa = std.testing.allocator;

    // The real glslang VS:
    //   layout(binding=0) uniform U { mat4 mvp; } u;
    //   layout(location=0) in vec2 p; layout(location=1) in vec3 c;
    //   layout(location=0) out vec3 vc;
    //   void main(){ gl_Position = u.mvp * vec4(p, 0.0, 1.0); vc = c; }
    // It reads a UBO mat4 (Load %mat4 + OpMatrixTimesVector), builds vec4(p, 0, 1) from
    // a vec2 input read per-component (pattern A), and writes gl_Position via gl_PerVertex
    // (pattern B), all lowered natively by vulcan-spirv, no Prism rewrite.
    const spv align(4) = @embedFile("vkubo_vert.spv").*;

    var func = try front.parseSpirv(gpa, &spv);
    defer func.deinit();

    const info = try rewriteGraphics(&func);
    try std.testing.expectEqual(GfxStage.vertex, info.stage);
    try std.testing.expectEqual(@as(usize, 5), info.input_count); // vec2 pos + vec3 color
    try std.testing.expectEqual(@as(usize, 1), info.buffer_count); // the one UBO

    var compiled = try jitFunction(gpa, &func, "main");
    defer compiled.deinit();

    // UBO = a column-major mat4 diag(0.5, 0.5, 1, 1): element (col j, row i) at j*4+i.
    var mvp = [_]f32{0} ** 16;
    mvp[0] = 0.5; // [0][0]
    mvp[5] = 0.5; // [1][1]
    mvp[10] = 1.0; // [2][2]
    mvp[15] = 1.0; // [3][3]
    const ubo: [*]const u8 = @ptrCast(&mvp);

    // VS inputs: p = (0.8, -0.8), c = (0.1, 0.2, 0.3). The VS computes
    // gl_Position = mvp * vec4(p, 0, 1) = (0.4, -0.4, 0, 1) under the scale.
    var out = [_]f32{0} ** GfxOut.vertex_len;
    try runGraphics(&compiled, info.input_count, info.buffer_count, &.{ 0.8, -0.8, 0.1, 0.2, 0.3 }, &.{ubo}, &out);
    try std.testing.expectApproxEqAbs(@as(f32, 0.4), out[0], 1e-5); // x scaled by 0.5
    try std.testing.expectApproxEqAbs(@as(f32, -0.4), out[1], 1e-5); // y scaled by 0.5
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), out[2], 1e-5); // z
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), out[3], 1e-5); // w
    // The varyings (vc = c) pass through at location 0.
    try std.testing.expectApproxEqAbs(@as(f32, 0.1), out[GfxOut.varying_base + 0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.2), out[GfxOut.varying_base + 1], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.3), out[GfxOut.varying_base + 2], 1e-5);
}

// General control flow (loops + branches + switch). The multi-block mem2reg
// (lib/prism/spirv.zig: Cytron SSA construction over the real CFG incl. back-edges)
// places loop-header / merge phis. Vulcan's lower.zig turns OpBranch/OpBranchConditional/
// OpSwitch + OpPhi into IR if/jump/block-params; the aarch64 isel emits the branches
// (forward + back-edges) and resolves phis with edge moves. These tests run glslang
// output end-to-end (JIT + execute), asserting the hand-computed result.

test "control flow: a glslang for-loop accumulator JITs + runs (loop-carried + per-iter use)" {
    const gpa = std.testing.allocator;
    // for(i<n) acc += i*2; out[gid]=acc, n=in[gid]. A loop-carried `acc` (loop-header phi)
    // plus a back-edge and a per-iteration use of the induction variable `i`.
    const spv align(4) = @embedFile("loop.spv").*;
    var func = try front.parseSpirv(gpa, &spv);
    defer func.deinit();
    var compiled = try jitFunction(gpa, &func, "main");
    defer compiled.deinit();
    const Fn = *const fn (i32, [*]u8, [*]u8) callconv(.c) void;
    const f = compiled.entry(Fn, "main") orelse return error.NoEntry;
    const N = 8;
    var in = [_]i32{ 0, 1, 2, 3, 4, 5, 6, 10 };
    var out = [_]i32{0} ** N;
    var i: i32 = 0;
    while (i < N) : (i += 1) f(i, @ptrCast(&in), @ptrCast(&out));
    for (0..N) |k| {
        var expect: i32 = 0;
        var j: i32 = 0;
        while (j < in[k]) : (j += 1) expect += j * 2;
        try std.testing.expectEqual(expect, out[k]);
    }
}

test "control flow: a glslang nested if JITs + runs (selection within selection)" {
    const gpa = std.testing.allocator;
    // if(x<10){ if((x&1)==0) r=x*2 else r=x*3 } else { r=x+100 }. Nested OpSelectionMerge +
    // OpBranchConditional with merge-block phis for `r`.
    const spv align(4) = @embedFile("nested.spv").*;
    var func = try front.parseSpirv(gpa, &spv);
    defer func.deinit();
    var compiled = try jitFunction(gpa, &func, "main");
    defer compiled.deinit();
    const Fn = *const fn (i32, [*]u8, [*]u8) callconv(.c) void;
    const f = compiled.entry(Fn, "main") orelse return error.NoEntry;
    const N = 16;
    var in: [N]i32 = undefined;
    for (0..N) |k| in[k] = @intCast(k);
    var out = [_]i32{0} ** N;
    var i: i32 = 0;
    while (i < N) : (i += 1) f(i, @ptrCast(&in), @ptrCast(&out));
    for (0..N) |k| {
        const x = in[k];
        const expect: i32 = if (x < 10) (if (@mod(x, 2) == 0) x * 2 else x * 3) else x + 100;
        try std.testing.expectEqual(expect, out[k]);
    }
}

test "control flow: a glslang OpSwitch JITs + runs (case + default, merge phi)" {
    const gpa = std.testing.allocator;
    // switch(x){case 0:100; case 1:200; case 2:300; default:x*10}. OpSwitch lowered to an
    // equality-test chain in vulcan; the merge block's `r` phi has the case blocks as preds.
    const spv align(4) = @embedFile("switch.spv").*;
    var func = try front.parseSpirv(gpa, &spv);
    defer func.deinit();
    var compiled = try jitFunction(gpa, &func, "main");
    defer compiled.deinit();
    const Fn = *const fn (i32, [*]u8, [*]u8) callconv(.c) void;
    const f = compiled.entry(Fn, "main") orelse return error.NoEntry;
    const N = 6;
    var in = [_]i32{ 0, 1, 2, 3, 4, 5 };
    var out = [_]i32{0} ** N;
    var i: i32 = 0;
    while (i < N) : (i += 1) f(i, @ptrCast(&in), @ptrCast(&out));
    const want = [_]i32{ 100, 200, 300, 30, 40, 50 };
    for (0..N) |k| try std.testing.expectEqual(want[k], out[k]);
}

test "control flow: the vkflow FS (loop + per-iteration conditional) JITs + runs per-fragment" {
    const gpa = std.testing.allocator;
    // FS: vec3 c=0; for(i<4){ if(((i+int(uv.x*4))&1)==0) c+=(0.2,0.1,0.05); } o=vec4(c,1).
    // A loop with a loop-carried vec3 accumulator + a per-iteration conditional, in a graphics
    // (fragment) shader run through rewriteGraphics + the JIT.
    const spv align(4) = @embedFile("flow.frag.spv").*;
    var func = try front.parseSpirv(gpa, &spv);
    defer func.deinit();
    const info = try rewriteGraphics(&func);
    try std.testing.expectEqual(GfxStage.fragment, info.stage);
    var compiled = try jitFunction(gpa, &func, "main");
    defer compiled.deinit();
    const bufs: []const ?[*]const u8 = &.{};
    // uv.x in [0,0.25) -> int(uv.x*4)=0 -> even i (0,2) pass -> 2 adds: c=(0.4,0.2,0.1).
    var out0 = [_]f32{0} ** GfxOut.fragment_len;
    try runGraphics(&compiled, info.input_count, info.buffer_count, &.{ 0.1, 0.0 }, bufs, &out0);
    try std.testing.expectApproxEqAbs(@as(f32, 0.4), out0[0], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 0.2), out0[1], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 0.1), out0[2], 1e-4);
    // uv.x in [0.25,0.5) -> base=1 -> odd i (1,3) pass -> 2 adds, same accumulation.
    var out1 = [_]f32{0} ** GfxOut.fragment_len;
    try runGraphics(&compiled, info.input_count, info.buffer_count, &.{ 0.3, 0.0 }, bufs, &out1);
    try std.testing.expectApproxEqAbs(@as(f32, 0.4), out1[0], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 0.2), out1[1], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 0.1), out1[2], 1e-4);
}

test "control flow: a FS vector loop-carried accumulator JITs + runs (vector phi over a loop)" {
    const gpa = std.testing.allocator;
    // FS: vec3 c=0; for(i<3) c += vec3(0.1*i, 0.2, 0.05); o=vec4(c,1). The loop-carried value
    // is a vector (scalarized vector phi at the loop header).
    const spv align(4) = @embedFile("vecloop.frag.spv").*;
    var func = try front.parseSpirv(gpa, &spv);
    defer func.deinit();
    const info = try rewriteGraphics(&func);
    var compiled = try jitFunction(gpa, &func, "main");
    defer compiled.deinit();
    var out = [_]f32{0} ** GfxOut.fragment_len;
    const bufs: []const ?[*]const u8 = &.{};
    try runGraphics(&compiled, info.input_count, info.buffer_count, &.{ 0.0, 0.0 }, bufs, &out);
    // sum i=0..2 of (0.1*i, 0.2, 0.05) = (0.3, 0.6, 0.15).
    try std.testing.expectApproxEqAbs(@as(f32, 0.3), out[0], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 0.6), out[1], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 0.15), out[2], 1e-4);
}

test "a graphics entry takes more than eight inputs, in order" {
    // Eight is where the signature table used to stop, and it is exactly a vec2
    // plus a vec2 plus a vec4. A vertex shader with one attribute more reported
    // `TooManyInputs`, and every caller in `raster.zig` drops that error, so the
    // draw silently produced an empty frame. phantom's rounded-rect shader takes
    // twelve, which is how it was found.
    //
    // A native function stands in for a JITed one: the dispatch and the C ABI are
    // the same either way, and this is the part that was broken. Twelve also
    // reaches past AArch64's eight SIMD argument registers, so inputs 9 to 12
    // arrive on the stack, which is the case worth pinning.
    const Entry = struct {
        fn f(
            a0: f32,
            a1: f32,
            a2: f32,
            a3: f32,
            a4: f32,
            a5: f32,
            a6: f32,
            a7: f32,
            a8: f32,
            a9: f32,
            a10: f32,
            a11: f32,
            out: O,
        ) callconv(.c) void {
            const all = [_]f32{ a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11 };
            for (all, 0..) |v, i| out[i] = v;
        }
    };

    const inputs = [_]f32{ 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21 };
    var out = [_]f32{0} ** 12;
    // Every input is distinct, so a signature that dropped one or swapped two
    // fails here rather than passing on a sum that happens to match.
    try runGraphicsAt(@ptrCast(&Entry.f), inputs.len, 0, &inputs, &.{}, &out);
    try std.testing.expectEqualSlices(f32, &inputs, &out);
}

test "a graphics entry past the limit reports it rather than drawing nothing" {
    // The limit still exists; it is just no longer eight. Past it the error is
    // the honest answer, and a caller that swallows it draws nothing, which is
    // the failure this whole path is about.
    const Entry = struct {
        fn f(out: O) callconv(.c) void {
            out[0] = 1;
        }
    };
    const inputs = [_]f32{0} ** (max_gfx_inputs + 1);
    var out = [_]f32{0};
    try std.testing.expectError(
        error.TooManyInputs,
        runGraphicsAt(@ptrCast(&Entry.f), inputs.len, 0, &inputs, &.{}, &out),
    );
}

test "a quad entry takes more than eight inputs too" {
    // The fragment side has the same table and the same limit. phantom's
    // rounded-rect fragment shader takes ten varyings, so fixing only the vertex
    // side would have moved the blank frame one stage along.
    const Entry = struct {
        fn f(q0: Quad, q1: Quad, q2: Quad, q3: Quad, q4: Quad, q5: Quad, q6: Quad, q7: Quad, q8: Quad, q9: Quad, out: QO) callconv(.c) void {
            const all = [_]Quad{ q0, q1, q2, q3, q4, q5, q6, q7, q8, q9 };
            for (all, 0..) |q, i| out[i] = q[0];
        }
    };

    var inputs: [10]Quad = undefined;
    for (&inputs, 0..) |*q, i| q.* = @splat(@floatFromInt(i + 30));
    var out = [_]f32{0} ** 10;
    try runGraphicsQuadAt(@ptrCast(&Entry.f), inputs.len, 0, &inputs, &.{}, &out);
    for (out, 0..) |v, i| try std.testing.expectEqual(@as(f32, @floatFromInt(i + 30)), v);
}
