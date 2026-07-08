//! End-to-end SPIR-V graphics path for the NVIDIA GPU: SPIR-V VS+FS -> Vulcan IR -> SASS,
//! drawn through the 3D pipeline to a readback RT. VS inputs are ALD a[0x80..0x8c],
//! position AST o[0x70..0x7c]. PS color goes to R0..R3. Counterpart to spirv_compute.zig.

const std = @import("std");
const nvidia = @import("nvidia");
const hal = @import("../../hal.zig");
const spirv = @import("../../spirv.zig");
const vspirv = @import("vulcan-spirv");
const isel = @import("vulcan-target").nvidia.isel;

const op = vspirv.opcodes;
const Builder = vspirv.binary.Builder;

/// Flat-color vertex shader as SPIR-V: input location 0 = position vec4<f32>,
/// output = the Position builtin vec4. Pass-through body. Returns the owned SPIR-V words.
fn buildVertexSpirv(gpa: std.mem.Allocator) !Builder {
    // ids: void=1 f32=2 v4f=3 pIn=4 pOut=5 voidfn=6 inVar=7 outVar=8 main=9
    //      entry=10 loaded=11.
    var b = try Builder.init(gpa, 12);
    errdefer b.deinit(gpa);
    try b.emit(gpa, op.EntryPoint, &.{ op.ExecutionModel.vertex, 9, 0, 7, 8 });
    try b.emit(gpa, op.Decorate, &.{ 7, op.Decoration.location, 0 });
    try b.emit(gpa, op.Decorate, &.{ 8, op.Decoration.builtin, op.BuiltIn.position });
    try b.emit(gpa, op.TypeVoid, &.{1});
    try b.emit(gpa, op.TypeFloat, &.{ 2, 32 });
    try b.emit(gpa, op.TypeVector, &.{ 3, 2, 4 }); // v4float
    try b.emit(gpa, op.TypePointer, &.{ 4, op.StorageClass.input, 3 });
    try b.emit(gpa, op.TypePointer, &.{ 5, op.StorageClass.output, 3 });
    try b.emit(gpa, op.TypeFunction, &.{ 6, 1 });
    try b.emit(gpa, op.Variable, &.{ 4, 7, op.StorageClass.input });
    try b.emit(gpa, op.Variable, &.{ 5, 8, op.StorageClass.output });
    try b.emit(gpa, op.Function, &.{ 1, 9, 0, 6 });
    try b.emit(gpa, op.Label, &.{10});
    try b.emit(gpa, op.Load, &.{ 3, 11, 7 });
    try b.emit(gpa, op.Store, &.{ 8, 11 });
    try b.emit(gpa, op.Return, &.{});
    try b.emit(gpa, op.FunctionEnd, &.{});
    return b;
}

/// Flat-color fragment shader as SPIR-V: output location 0 = color vec4<f32>.
/// The body stores a constant color via OpConstantComposite. `rgba` is the four f32 components.
/// Returns the owned SPIR-V words.
fn buildFragmentSpirv(gpa: std.mem.Allocator, rgba: [4]f32) !Builder {
    // ids: void=1 f32=2 v4f=3 pOut=4 voidfn=5 c_r=6 c_g=7 c_b=8 c_a=9 color=10
    //      outVar=11 main=12 entry=13.
    var b = try Builder.init(gpa, 14);
    errdefer b.deinit(gpa);
    try b.emit(gpa, op.EntryPoint, &.{ op.ExecutionModel.fragment, 12, 0, 11 });
    try b.emit(gpa, op.Decorate, &.{ 11, op.Decoration.location, 0 });
    try b.emit(gpa, op.TypeVoid, &.{1});
    try b.emit(gpa, op.TypeFloat, &.{ 2, 32 });
    try b.emit(gpa, op.TypeVector, &.{ 3, 2, 4 });
    try b.emit(gpa, op.TypePointer, &.{ 4, op.StorageClass.output, 3 });
    try b.emit(gpa, op.TypeFunction, &.{ 5, 1 });
    try b.emit(gpa, op.Constant, &.{ 2, 6, @bitCast(rgba[0]) });
    try b.emit(gpa, op.Constant, &.{ 2, 7, @bitCast(rgba[1]) });
    try b.emit(gpa, op.Constant, &.{ 2, 8, @bitCast(rgba[2]) });
    try b.emit(gpa, op.Constant, &.{ 2, 9, @bitCast(rgba[3]) });
    try b.emit(gpa, op.ConstantComposite, &.{ 3, 10, 6, 7, 8, 9 });
    try b.emit(gpa, op.Variable, &.{ 4, 11, op.StorageClass.output });
    try b.emit(gpa, op.Function, &.{ 1, 12, 0, 5 });
    try b.emit(gpa, op.Label, &.{13});
    try b.emit(gpa, op.Store, &.{ 11, 10 });
    try b.emit(gpa, op.Return, &.{});
    try b.emit(gpa, op.FunctionEnd, &.{});
    return b;
}

/// Fragment shader that always discards as SPIR-V: stores a constant color to output
/// location 0 then executes OpKill unconditionally. The Vulcan frontend synthesizes a
/// `discard_fn` param and lowers OpKill to a call of it. The nvidia isel emits SASS
/// `KIL` (opcode 0x95b). KIL masks the fragment at the ROP so nothing is written.
/// Returns the owned SPIR-V words.
fn buildDiscardFragmentSpirv(gpa: std.mem.Allocator, rgba: [4]f32) !Builder {
    // ids: void=1 f32=2 v4f=3 pOut=4 voidfn=5 c_r=6 c_g=7 c_b=8 c_a=9 color=10
    //      outVar=11 main=12 entry=13.
    var b = try Builder.init(gpa, 14);
    errdefer b.deinit(gpa);
    try b.emit(gpa, op.EntryPoint, &.{ op.ExecutionModel.fragment, 12, 0, 11 });
    try b.emit(gpa, op.Decorate, &.{ 11, op.Decoration.location, 0 });
    try b.emit(gpa, op.TypeVoid, &.{1});
    try b.emit(gpa, op.TypeFloat, &.{ 2, 32 });
    try b.emit(gpa, op.TypeVector, &.{ 3, 2, 4 });
    try b.emit(gpa, op.TypePointer, &.{ 4, op.StorageClass.output, 3 });
    try b.emit(gpa, op.TypeFunction, &.{ 5, 1 });
    try b.emit(gpa, op.Constant, &.{ 2, 6, @bitCast(rgba[0]) });
    try b.emit(gpa, op.Constant, &.{ 2, 7, @bitCast(rgba[1]) });
    try b.emit(gpa, op.Constant, &.{ 2, 8, @bitCast(rgba[2]) });
    try b.emit(gpa, op.Constant, &.{ 2, 9, @bitCast(rgba[3]) });
    try b.emit(gpa, op.ConstantComposite, &.{ 3, 10, 6, 7, 8, 9 });
    try b.emit(gpa, op.Variable, &.{ 4, 11, op.StorageClass.output });
    try b.emit(gpa, op.Function, &.{ 1, 12, 0, 5 });
    try b.emit(gpa, op.Label, &.{13});
    try b.emit(gpa, op.Store, &.{ 11, 10 }); // out_color = color (never reaches ROP)
    try b.emit(gpa, op.Kill, &.{}); // discard: block terminator, no Return follows
    try b.emit(gpa, op.FunctionEnd, &.{});
    return b;
}

/// gl_FragCoord fragment shader as SPIR-V: reads the FragCoord builtin (window-space
/// pixel position) and writes color = (FragCoord.x/dim, FragCoord.y/dim, 0, 1).
/// Red rises left-to-right, green top-to-bottom. Pixel (px,py) reads back r~=px/dim,
/// g~=py/dim. Proves the raster delivers the pixel position through the POSITION IPA.
/// `dim` is the RT size the coords are normalized by. Returns owned words.
fn buildFragCoordFragmentSpirv(gpa: std.mem.Allocator, dim: f32) !Builder {
    // ids: void=1 f32=2 v4f=3 pOut=4 voidfn=5 pIn=6 fcVar=7 outVar=8 main=9 entry=10
    //      loaded=11 x=12 y=13 scale=14 sx=15 sy=16 zero=17 one=18 result=19.
    var b = try Builder.init(gpa, 20);
    errdefer b.deinit(gpa);
    try b.emit(gpa, op.EntryPoint, &.{ op.ExecutionModel.fragment, 9, 0, 7, 8 });
    try b.emit(gpa, op.Decorate, &.{ 7, op.Decoration.builtin, op.BuiltIn.frag_coord });
    try b.emit(gpa, op.Decorate, &.{ 8, op.Decoration.location, 0 });
    try b.emit(gpa, op.TypeVoid, &.{1});
    try b.emit(gpa, op.TypeFloat, &.{ 2, 32 });
    try b.emit(gpa, op.TypeVector, &.{ 3, 2, 4 });
    try b.emit(gpa, op.TypePointer, &.{ 4, op.StorageClass.output, 3 });
    try b.emit(gpa, op.TypeFunction, &.{ 5, 1 });
    try b.emit(gpa, op.TypePointer, &.{ 6, op.StorageClass.input, 3 });
    try b.emit(gpa, op.Variable, &.{ 6, 7, op.StorageClass.input }); // gl_FragCoord
    try b.emit(gpa, op.Variable, &.{ 4, 8, op.StorageClass.output }); // color out
    try b.emit(gpa, op.Constant, &.{ 2, 14, @bitCast(@as(f32, 1.0) / dim) });
    try b.emit(gpa, op.Constant, &.{ 2, 17, @bitCast(@as(f32, 0.0)) });
    try b.emit(gpa, op.Constant, &.{ 2, 18, @bitCast(@as(f32, 1.0)) });
    try b.emit(gpa, op.Function, &.{ 1, 9, 0, 5 });
    try b.emit(gpa, op.Label, &.{10});
    try b.emit(gpa, op.Load, &.{ 3, 11, 7 });
    try b.emit(gpa, op.CompositeExtract, &.{ 2, 12, 11, 0 }); // FragCoord.x
    try b.emit(gpa, op.CompositeExtract, &.{ 2, 13, 11, 1 }); // FragCoord.y
    try b.emit(gpa, op.FMul, &.{ 2, 15, 12, 14 }); // x / dim
    try b.emit(gpa, op.FMul, &.{ 2, 16, 13, 14 }); // y / dim
    try b.emit(gpa, op.CompositeConstruct, &.{ 3, 19, 15, 16, 17, 18 });
    try b.emit(gpa, op.Store, &.{ 8, 19 });
    try b.emit(gpa, op.Return, &.{});
    try b.emit(gpa, op.FunctionEnd, &.{});
    return b;
}

/// gl_FrontFacing fragment shader as SPIR-V: reads the FrontFacing builtin (true for
/// a front-facing primitive) and outputs green (0,1,0,1) when front, red (1,0,0,1)
/// when back. Proves the raster delivers the facing flag through the FRONT_FACE attribute.
/// Returns owned words.
fn buildFrontFaceFragmentSpirv(gpa: std.mem.Allocator) !Builder {
    // ids: void=1 f32=2 v4f=3 pOut=4 voidfn=5 bool=6 pInBool=7 ffVar=8 outVar=9 main=10
    //      entry=11 loaded=12 c0=13 c1=14 green=15 red=16 result=17.
    var b = try Builder.init(gpa, 18);
    errdefer b.deinit(gpa);
    try b.emit(gpa, op.EntryPoint, &.{ op.ExecutionModel.fragment, 10, 0, 8, 9 });
    try b.emit(gpa, op.Decorate, &.{ 8, op.Decoration.builtin, op.BuiltIn.front_facing });
    try b.emit(gpa, op.Decorate, &.{ 9, op.Decoration.location, 0 });
    try b.emit(gpa, op.TypeVoid, &.{1});
    try b.emit(gpa, op.TypeFloat, &.{ 2, 32 });
    try b.emit(gpa, op.TypeVector, &.{ 3, 2, 4 });
    try b.emit(gpa, op.TypePointer, &.{ 4, op.StorageClass.output, 3 });
    try b.emit(gpa, op.TypeFunction, &.{ 5, 1 });
    try b.emit(gpa, op.TypeBool, &.{6});
    try b.emit(gpa, op.TypePointer, &.{ 7, op.StorageClass.input, 6 });
    try b.emit(gpa, op.Variable, &.{ 7, 8, op.StorageClass.input }); // gl_FrontFacing
    try b.emit(gpa, op.Variable, &.{ 4, 9, op.StorageClass.output }); // color out
    try b.emit(gpa, op.Constant, &.{ 2, 13, @bitCast(@as(f32, 0.0)) });
    try b.emit(gpa, op.Constant, &.{ 2, 14, @bitCast(@as(f32, 1.0)) });
    try b.emit(gpa, op.ConstantComposite, &.{ 3, 15, 13, 14, 13, 14 }); // green
    try b.emit(gpa, op.ConstantComposite, &.{ 3, 16, 14, 13, 13, 14 }); // red
    try b.emit(gpa, op.Function, &.{ 1, 10, 0, 5 });
    try b.emit(gpa, op.Label, &.{11});
    try b.emit(gpa, op.Load, &.{ 6, 12, 8 });
    try b.emit(gpa, op.Select, &.{ 3, 17, 12, 15, 16 }); // front ? green : red
    try b.emit(gpa, op.Store, &.{ 9, 17 });
    try b.emit(gpa, op.Return, &.{});
    try b.emit(gpa, op.FunctionEnd, &.{});
    return b;
}

/// gl_FragDepth fragment shader as SPIR-V: writes a constant color to output location 0
/// and a constant depth to the FragDepth builtin. The shader-written depth drives the
/// depth test, not the interpolated gl_Position.z, so two overlapping triangles at the
/// same geometry z occlude by their FragDepth values. `rgba` is the color, `depth` the
/// [0,1] fragment depth. Returns the owned SPIR-V words.
fn buildDepthFragmentSpirv(gpa: std.mem.Allocator, rgba: [4]f32, depth: f32) !Builder {
    // ids: void=1 f32=2 v4f=3 pOutV4=4 voidfn=5 pOutF32=6 c_r=7 c_g=8 c_b=9 c_a=10
    //      color=11 depthC=12 colorVar=13 depthVar=14 main=15 entry=16.
    var b = try Builder.init(gpa, 17);
    errdefer b.deinit(gpa);
    try b.emit(gpa, op.EntryPoint, &.{ op.ExecutionModel.fragment, 15, 0, 13, 14 });
    try b.emit(gpa, op.Decorate, &.{ 13, op.Decoration.location, 0 });
    try b.emit(gpa, op.Decorate, &.{ 14, op.Decoration.builtin, op.BuiltIn.frag_depth });
    try b.emit(gpa, op.TypeVoid, &.{1});
    try b.emit(gpa, op.TypeFloat, &.{ 2, 32 });
    try b.emit(gpa, op.TypeVector, &.{ 3, 2, 4 });
    try b.emit(gpa, op.TypePointer, &.{ 4, op.StorageClass.output, 3 });
    try b.emit(gpa, op.TypeFunction, &.{ 5, 1 });
    try b.emit(gpa, op.TypePointer, &.{ 6, op.StorageClass.output, 2 });
    try b.emit(gpa, op.Constant, &.{ 2, 7, @bitCast(rgba[0]) });
    try b.emit(gpa, op.Constant, &.{ 2, 8, @bitCast(rgba[1]) });
    try b.emit(gpa, op.Constant, &.{ 2, 9, @bitCast(rgba[2]) });
    try b.emit(gpa, op.Constant, &.{ 2, 10, @bitCast(rgba[3]) });
    try b.emit(gpa, op.ConstantComposite, &.{ 3, 11, 7, 8, 9, 10 });
    try b.emit(gpa, op.Constant, &.{ 2, 12, @bitCast(depth) });
    try b.emit(gpa, op.Variable, &.{ 4, 13, op.StorageClass.output }); // color out (loc 0)
    try b.emit(gpa, op.Variable, &.{ 6, 14, op.StorageClass.output }); // gl_FragDepth
    try b.emit(gpa, op.Function, &.{ 1, 15, 0, 5 });
    try b.emit(gpa, op.Label, &.{16});
    try b.emit(gpa, op.Store, &.{ 13, 11 }); // color = rgba
    try b.emit(gpa, op.Store, &.{ 14, 12 }); // gl_FragDepth = depth
    try b.emit(gpa, op.Return, &.{});
    try b.emit(gpa, op.FunctionEnd, &.{});
    return b;
}

/// MRT fragment shader as SPIR-V: two color outputs, location 0 = `c0` and location 1 = `c1`.
/// The isel routes RT0 to R0-R3 and RT1 to R4-R7. The pipeline declares two SPH omap targets
/// and binds two color surfaces. Returns owned words.
fn buildMrtFragmentSpirv(gpa: std.mem.Allocator, c0: [4]f32, c1: [4]f32) !Builder {
    // ids: void=1 f32=2 v4f=3 pOut=4 voidfn=5 r0=6 g0=7 b0=8 a0=9 col0=10
    //      r1=11 g1=12 b1=13 a1=14 col1=15 outVar0=16 outVar1=17 main=18 entry=19.
    var b = try Builder.init(gpa, 20);
    errdefer b.deinit(gpa);
    try b.emit(gpa, op.EntryPoint, &.{ op.ExecutionModel.fragment, 18, 0, 16, 17 });
    try b.emit(gpa, op.Decorate, &.{ 16, op.Decoration.location, 0 });
    try b.emit(gpa, op.Decorate, &.{ 17, op.Decoration.location, 1 });
    try b.emit(gpa, op.TypeVoid, &.{1});
    try b.emit(gpa, op.TypeFloat, &.{ 2, 32 });
    try b.emit(gpa, op.TypeVector, &.{ 3, 2, 4 });
    try b.emit(gpa, op.TypePointer, &.{ 4, op.StorageClass.output, 3 });
    try b.emit(gpa, op.TypeFunction, &.{ 5, 1 });
    try b.emit(gpa, op.Constant, &.{ 2, 6, @bitCast(c0[0]) });
    try b.emit(gpa, op.Constant, &.{ 2, 7, @bitCast(c0[1]) });
    try b.emit(gpa, op.Constant, &.{ 2, 8, @bitCast(c0[2]) });
    try b.emit(gpa, op.Constant, &.{ 2, 9, @bitCast(c0[3]) });
    try b.emit(gpa, op.ConstantComposite, &.{ 3, 10, 6, 7, 8, 9 });
    try b.emit(gpa, op.Constant, &.{ 2, 11, @bitCast(c1[0]) });
    try b.emit(gpa, op.Constant, &.{ 2, 12, @bitCast(c1[1]) });
    try b.emit(gpa, op.Constant, &.{ 2, 13, @bitCast(c1[2]) });
    try b.emit(gpa, op.Constant, &.{ 2, 14, @bitCast(c1[3]) });
    try b.emit(gpa, op.ConstantComposite, &.{ 3, 15, 11, 12, 13, 14 });
    try b.emit(gpa, op.Variable, &.{ 4, 16, op.StorageClass.output }); // RT0 (location 0)
    try b.emit(gpa, op.Variable, &.{ 4, 17, op.StorageClass.output }); // RT1 (location 1)
    try b.emit(gpa, op.Function, &.{ 1, 18, 0, 5 });
    try b.emit(gpa, op.Label, &.{19});
    try b.emit(gpa, op.Store, &.{ 16, 10 }); // RT0 = c0
    try b.emit(gpa, op.Store, &.{ 17, 15 }); // RT1 = c1
    try b.emit(gpa, op.Return, &.{});
    try b.emit(gpa, op.FunctionEnd, &.{});
    return b;
}

/// Gradient vertex shader as SPIR-V: input location 0 = position vec4<f32>, input
/// location 1 = color vec4<f32>; outputs = Position builtin and color varying at output
/// location 0. Pure pass-through body. Position lowered to ATTR_POSITION, color varying
/// to ATTR_GENERIC0 (0x80); inputs to ALD slots a[0x80..] / a[0x90..].
/// Returns the owned SPIR-V words.
fn buildGradientVertexSpirv(gpa: std.mem.Allocator) !Builder {
    // ids: void=1 f32=2 v4f=3 pIn=4 pOut=5 voidfn=6 posIn=7 colIn=8 posOut=9
    //      colOut=10 main=11 entry=12 ldPos=13 ldCol=14.
    var b = try Builder.init(gpa, 15);
    errdefer b.deinit(gpa);
    try b.emit(gpa, op.EntryPoint, &.{ op.ExecutionModel.vertex, 11, 0, 7, 8, 9, 10 });
    try b.emit(gpa, op.Decorate, &.{ 7, op.Decoration.location, 0 }); // position input
    try b.emit(gpa, op.Decorate, &.{ 8, op.Decoration.location, 1 }); // color input
    try b.emit(gpa, op.Decorate, &.{ 9, op.Decoration.builtin, op.BuiltIn.position });
    try b.emit(gpa, op.Decorate, &.{ 10, op.Decoration.location, 0 }); // color varying output
    try b.emit(gpa, op.TypeVoid, &.{1});
    try b.emit(gpa, op.TypeFloat, &.{ 2, 32 });
    try b.emit(gpa, op.TypeVector, &.{ 3, 2, 4 }); // v4float
    try b.emit(gpa, op.TypePointer, &.{ 4, op.StorageClass.input, 3 });
    try b.emit(gpa, op.TypePointer, &.{ 5, op.StorageClass.output, 3 });
    try b.emit(gpa, op.TypeFunction, &.{ 6, 1 });
    try b.emit(gpa, op.Variable, &.{ 4, 7, op.StorageClass.input }); // position input (loc 0)
    try b.emit(gpa, op.Variable, &.{ 4, 8, op.StorageClass.input }); // color input (loc 1)
    try b.emit(gpa, op.Variable, &.{ 5, 9, op.StorageClass.output }); // gl_Position
    try b.emit(gpa, op.Variable, &.{ 5, 10, op.StorageClass.output }); // color varying (loc 0)
    try b.emit(gpa, op.Function, &.{ 1, 11, 0, 6 });
    try b.emit(gpa, op.Label, &.{12});
    try b.emit(gpa, op.Load, &.{ 3, 13, 7 }); // pos = load position input
    try b.emit(gpa, op.Store, &.{ 9, 13 }); // gl_Position = pos
    try b.emit(gpa, op.Load, &.{ 3, 14, 8 }); // col = load color input
    try b.emit(gpa, op.Store, &.{ 10, 14 }); // out_color = col
    try b.emit(gpa, op.Return, &.{});
    try b.emit(gpa, op.FunctionEnd, &.{});
    return b;
}

/// Gradient fragment shader as SPIR-V: input location 0 = color varying vec4<f32>
/// (the interpolated VS output); output location 0 = color vec4<f32>.
/// Body: out_color = in_color, a plain read of the interpolated varying.
/// Lowering scalarizes the input to IPA params at a[0x80..0x8c], output to R0..R3.
/// Returns the owned SPIR-V words.
fn buildGradientFragmentSpirv(gpa: std.mem.Allocator) !Builder {
    // ids: void=1 f32=2 v4f=3 pIn=4 pOut=5 voidfn=6 colIn=7 colOut=8 main=9
    //      entry=10 loaded=11.
    var b = try Builder.init(gpa, 12);
    errdefer b.deinit(gpa);
    try b.emit(gpa, op.EntryPoint, &.{ op.ExecutionModel.fragment, 9, 0, 7, 8 });
    try b.emit(gpa, op.Decorate, &.{ 7, op.Decoration.location, 0 }); // color varying input
    try b.emit(gpa, op.Decorate, &.{ 8, op.Decoration.location, 0 }); // color output
    try b.emit(gpa, op.TypeVoid, &.{1});
    try b.emit(gpa, op.TypeFloat, &.{ 2, 32 });
    try b.emit(gpa, op.TypeVector, &.{ 3, 2, 4 });
    try b.emit(gpa, op.TypePointer, &.{ 4, op.StorageClass.input, 3 });
    try b.emit(gpa, op.TypePointer, &.{ 5, op.StorageClass.output, 3 });
    try b.emit(gpa, op.TypeFunction, &.{ 6, 1 });
    try b.emit(gpa, op.Variable, &.{ 4, 7, op.StorageClass.input }); // color varying in (loc 0)
    try b.emit(gpa, op.Variable, &.{ 5, 8, op.StorageClass.output }); // color out (loc 0)
    try b.emit(gpa, op.Function, &.{ 1, 9, 0, 6 });
    try b.emit(gpa, op.Label, &.{10});
    try b.emit(gpa, op.Load, &.{ 3, 11, 7 }); // col = load interpolated varying
    try b.emit(gpa, op.Store, &.{ 8, 11 }); // out_color = col
    try b.emit(gpa, op.Return, &.{});
    try b.emit(gpa, op.FunctionEnd, &.{});
    return b;
}

/// Compile a SPIR-V shader (`code` byte stream) to NVIDIA SASS for `stage`.
/// Caller owns the returned dword slice. Public so shader.zig can compile a
/// Vulkan VS/FS SPIR-V to SASS when the ICD hands it a SPIR-V word stream.
pub fn compileToSass(gpa: std.mem.Allocator, code: []const u8, stage: isel.Stage) ![]u32 {
    return (try compileToSassRegs(gpa, code, stage)).code;
}

pub const CompiledSass = struct { code: []u32, reg_count: u32, writes_depth: bool = false, color_targets: u8 = 1 };

/// Like `compileToSass` but also returns the GPR count the isel allocated.
/// SET_PIPELINE_REGISTER_COUNT needs to cover this usage exactly. A heavier shader
/// (e.g. the glmark2 desktop blur) allocates more than a tight blit. A register count
/// below the shader's usage faults the SM with Xid 13 "Out Of Range Register".
pub fn compileToSassRegs(gpa: std.mem.Allocator, code: []const u8, stage: isel.Stage) !CompiledSass {
    var func = try spirv.parseSpirv(gpa, code);
    defer func.deinit();
    var kernel = try isel.compileShader(gpa, &func, stage);
    defer kernel.deinit(gpa);
    return .{ .code = try gpa.dupe(u32, kernel.code), .reg_count = kernel.reg_count, .writes_depth = kernel.writes_depth, .color_targets = kernel.color_targets };
}

/// Map a HAL shader stage to the Vulcan isel graphics stage. Compute is not a
/// graphics stage and returns error.InvalidArgument. See spirv_compute.zig.
pub fn iselStage(stage: hal.ShaderStage) hal.Error!isel.Stage {
    return switch (stage) {
        .vertex => .vertex,
        .fragment => .fragment,
        .compute => error.InvalidArgument,
    };
}

/// The SPIR-V magic word (first dword of any SPIR-V module). A `code` stream that
/// starts with this is a Vulkan SPIR-V shader. The nvidia driver compiles it to SASS.
/// Otherwise `code` is raw SASS.
pub const SPIRV_MAGIC: u32 = 0x07230203;

/// Like `run`, but the pipeline carries an explicit `blend` state so the flat-color
/// triangle is alpha-composited over the `clear` background. Used by the blend oracle:
/// the same translucent-over-solid scene renders on both nvidia and software, and both
/// readbacks are compared. Works on any `hal.Device`. Returns A8R8G8B8 u32s packed
/// w*h, caller-owned.
pub fn runBlend(gpa: std.mem.Allocator, dev: hal.Device, w: u32, h: u32, clear: hal.Color, color: [4]f32, blend: hal.BlendState) ![]u32 {
    // Pass raw SPIR-V to createShaderModule: the nvidia driver compiles it to SASS
    // internally, the software driver JITs it directly. The same module source drives
    // both devices, which is required for the cross-driver blend comparison.
    var vs_b = try buildVertexSpirv(gpa);
    defer vs_b.deinit(gpa);
    var ps_b = try buildFragmentSpirv(gpa, color);
    defer ps_b.deinit(gpa);

    const rt = try dev.createResource(.{ .image = .{ .width = w, .height = h, .format = .bgra8_unorm, .usage = .{ .render_target = true } } });
    defer dev.destroyResource(rt);

    const vs = try dev.createShaderModule(.{ .stage = .vertex, .code = std.mem.sliceAsBytes(vs_b.words.items) });
    defer dev.destroyShaderModule(vs);
    const ps = try dev.createShaderModule(.{ .stage = .fragment, .code = std.mem.sliceAsBytes(ps_b.words.items) });
    defer dev.destroyShaderModule(ps);

    const vbuf = try dev.createResource(.{ .buffer = .{ .size = 3 * 16, .usage = .{ .vertex = true } } });
    defer dev.destroyResource(vbuf);
    const verts = [_]f32{
        0.0, 0.5, 0.0, 1.0, // top
        -0.5, -0.5, 0.0, 1.0, // bottom-left
        0.5, -0.5, 0.0, 1.0, // bottom-right
    };
    @memcpy(std.mem.bytesAsSlice(f32, try dev.mapResource(vbuf))[0..verts.len], &verts);

    const attrs = [_]hal.VertexAttribute{
        .{ .location = 0, .format = .rgba8_unorm, .offset = 0 },
    };
    const pipe = try dev.createPipeline(.{
        .vertex = vs,
        .fragment = ps,
        .vertex_layout = .{ .stride = 16, .attributes = &attrs },
        .color_format = .bgra8_unorm,
        .blend = blend,
    });
    defer dev.destroyPipeline(pipe);

    const ctx = try dev.createContext();
    defer ctx.deinit();
    {
        const cb = try ctx.beginCommands();
        defer cb.deinit();
        try cb.setRenderTarget(rt);
        try cb.clear(clear);
        try cb.bindPipeline(pipe);
        try cb.bindVertexBuffer(vbuf);
        try cb.draw(3, 0);
        try ctx.submit(cb);
    }

    const fb = std.mem.bytesAsSlice(u32, try dev.mapResource(rt));
    const out = try gpa.alloc(u32, @as(usize, w) * h);
    errdefer gpa.free(out);
    const pitch_px = w;
    var y: u32 = 0;
    while (y < h) : (y += 1) {
        @memcpy(out[@as(usize, y) * w ..][0..w], fb[@as(usize, y) * pitch_px ..][0..w]);
    }
    return out;
}

/// Render a flat-color triangle from SPIR-V VS/PS to a freshly created render target,
/// copy the framebuffer (A8R8G8B8, tightly repacked to `w*h*4`) into caller-owned
/// memory, and tear down every GPU resource. `clear` is the background color.
/// `color` is the triangle's constant color. Caller frees the returned slice.
pub fn run(gpa: std.mem.Allocator, dev: hal.Device, w: u32, h: u32, clear: hal.Color, color: [4]f32) ![]u32 {
    // SPIR-V -> Vulcan IR -> SASS for both stages.
    var vs_b = try buildVertexSpirv(gpa);
    defer vs_b.deinit(gpa);
    var ps_b = try buildFragmentSpirv(gpa, color);
    defer ps_b.deinit(gpa);
    const vs_sass = try compileToSass(gpa, std.mem.sliceAsBytes(vs_b.words.items), .vertex);
    defer gpa.free(vs_sass);
    const ps_sass = try compileToSass(gpa, std.mem.sliceAsBytes(ps_b.words.items), .fragment);
    defer gpa.free(ps_sass);

    // Render target (cleared to `clear` so the triangle is distinguishable).
    const rt = try dev.createResource(.{ .image = .{ .width = w, .height = h, .format = .bgra8_unorm, .usage = .{ .render_target = true } } });
    defer dev.destroyResource(rt);

    // Shader modules from the SPIR-V-compiled SASS.
    const vs = try dev.createShaderModule(.{ .stage = .vertex, .code = std.mem.sliceAsBytes(vs_sass) });
    defer dev.destroyShaderModule(vs);
    const ps = try dev.createShaderModule(.{ .stage = .fragment, .code = std.mem.sliceAsBytes(ps_sass) });
    defer dev.destroyShaderModule(ps);

    // Vertex buffer: 3 verts of a vec4 clip-space position (the triangle_hal
    // geometry: top, bottom-left, bottom-right). Single position-only attribute.
    const vbuf = try dev.createResource(.{ .buffer = .{ .size = 3 * 16, .usage = .{ .vertex = true } } });
    defer dev.destroyResource(vbuf);
    const verts = [_]f32{
        0.0, 0.5, 0.0, 1.0, // top
        -0.5, -0.5, 0.0, 1.0, // bottom-left
        0.5, -0.5, 0.0, 1.0, // bottom-right
    };
    @memcpy(std.mem.bytesAsSlice(f32, try dev.mapResource(vbuf))[0..verts.len], &verts);

    const attrs = [_]hal.VertexAttribute{
        .{ .location = 0, .format = .rgba8_unorm, .offset = 0 }, // clip-space position
    };
    const pipe = try dev.createPipeline(.{
        .vertex = vs,
        .fragment = ps,
        .vertex_layout = .{ .stride = 16, .attributes = &attrs },
        .color_format = .bgra8_unorm,
    });
    defer dev.destroyPipeline(pipe);

    const ctx = try dev.createContext();
    defer ctx.deinit();
    {
        const cb = try ctx.beginCommands();
        defer cb.deinit();
        try cb.setRenderTarget(rt);
        try cb.clear(clear);
        try cb.bindPipeline(pipe);
        try cb.bindVertexBuffer(vbuf);
        try cb.draw(3, 0);
        try ctx.submit(cb);
    }

    // Copy the GPU-pitched framebuffer (256-byte aligned rows) into a tightly
    // packed w*h buffer the caller owns. For w=256 the pitch is exactly w*4.
    const fb = std.mem.bytesAsSlice(u32, try dev.mapResource(rt));
    const out = try gpa.alloc(u32, @as(usize, w) * h);
    errdefer gpa.free(out);
    // Tight (w*4) readback: the software RT is tight and nvidia mapResource de-swizzles
    // the block-linear RT into a tight w*4 buffer. (pitchBytes(w)/4 == w only at 256.)
    const pitch_px = w;
    var y: u32 = 0;
    while (y < h) : (y += 1) {
        @memcpy(out[@as(usize, y) * w ..][0..w], fb[@as(usize, y) * pitch_px ..][0..w]);
    }
    return out;
}

/// Like `run`, but the fragment shader always discards (`buildDiscardFragmentSpirv`).
/// The triangle rasterizes and the PS runs, but every fragment executes SASS `KIL`
/// so the ROP writes nothing. The returned framebuffer is the pure clear color.
/// Proves OpKill->KIL masks fragments on the real GPU. Returns A8R8G8B8 u32s
/// packed w*h, caller-owned.
pub fn runDiscard(gpa: std.mem.Allocator, dev: hal.Device, w: u32, h: u32, clear: hal.Color, color: [4]f32) ![]u32 {
    var vs_b = try buildVertexSpirv(gpa);
    defer vs_b.deinit(gpa);
    var ps_b = try buildDiscardFragmentSpirv(gpa, color);
    defer ps_b.deinit(gpa);

    const rt = try dev.createResource(.{ .image = .{ .width = w, .height = h, .format = .bgra8_unorm, .usage = .{ .render_target = true } } });
    defer dev.destroyResource(rt);

    // Raw SPIR-V so the discard FS runs on any device: nvidia -> SASS KIL internally,
    // software -> the discard_fn thunk. See the runFragCoord note.
    const vs = try dev.createShaderModule(.{ .stage = .vertex, .code = std.mem.sliceAsBytes(vs_b.words.items) });
    defer dev.destroyShaderModule(vs);
    const ps = try dev.createShaderModule(.{ .stage = .fragment, .code = std.mem.sliceAsBytes(ps_b.words.items) });
    defer dev.destroyShaderModule(ps);

    const vbuf = try dev.createResource(.{ .buffer = .{ .size = 3 * 16, .usage = .{ .vertex = true } } });
    defer dev.destroyResource(vbuf);
    const verts = [_]f32{
        0.0, 0.5, 0.0, 1.0, // top
        -0.5, -0.5, 0.0, 1.0, // bottom-left
        0.5, -0.5, 0.0, 1.0, // bottom-right
    };
    @memcpy(std.mem.bytesAsSlice(f32, try dev.mapResource(vbuf))[0..verts.len], &verts);

    const attrs = [_]hal.VertexAttribute{
        .{ .location = 0, .format = .rgba8_unorm, .offset = 0 },
    };
    const pipe = try dev.createPipeline(.{
        .vertex = vs,
        .fragment = ps,
        .vertex_layout = .{ .stride = 16, .attributes = &attrs },
        .color_format = .bgra8_unorm,
    });
    defer dev.destroyPipeline(pipe);

    const ctx = try dev.createContext();
    defer ctx.deinit();
    {
        const cb = try ctx.beginCommands();
        defer cb.deinit();
        try cb.setRenderTarget(rt);
        try cb.clear(clear);
        try cb.bindPipeline(pipe);
        try cb.bindVertexBuffer(vbuf);
        try cb.draw(3, 0);
        try ctx.submit(cb);
    }

    const fb = std.mem.bytesAsSlice(u32, try dev.mapResource(rt));
    const out = try gpa.alloc(u32, @as(usize, w) * h);
    errdefer gpa.free(out);
    const pitch_px = w;
    var y: u32 = 0;
    while (y < h) : (y += 1) {
        @memcpy(out[@as(usize, y) * w ..][0..w], fb[@as(usize, y) * pitch_px ..][0..w]);
    }
    return out;
}

/// Like `run`, but the fragment shader outputs gl_FragCoord.xy/`dim` as color
/// (`buildFragCoordFragmentSpirv`). The result is a screen-space gradient: pixel
/// (px, py) reads back r ~= px/dim, g ~= py/dim. Proves the window-space position
/// reaches the FS on the GPU. Returns A8R8G8B8 u32s tightly packed `w*h`, caller-owned.
pub fn runFragCoord(gpa: std.mem.Allocator, dev: hal.Device, w: u32, h: u32, clear: hal.Color, dim: f32) ![]u32 {
    var vs_b = try buildVertexSpirv(gpa);
    defer vs_b.deinit(gpa);
    var ps_b = try buildFragCoordFragmentSpirv(gpa, dim);
    defer ps_b.deinit(gpa);

    const rt = try dev.createResource(.{ .image = .{ .width = w, .height = h, .format = .bgra8_unorm, .usage = .{ .render_target = true } } });
    defer dev.destroyResource(rt);

    // Pass raw SPIR-V so the shader runs on any device: nvidia compiles it to SASS internally,
    // the software driver JITs it. Pre-compiling to nvidia SASS here would break the software path.
    const vs = try dev.createShaderModule(.{ .stage = .vertex, .code = std.mem.sliceAsBytes(vs_b.words.items) });
    defer dev.destroyShaderModule(vs);
    const ps = try dev.createShaderModule(.{ .stage = .fragment, .code = std.mem.sliceAsBytes(ps_b.words.items) });
    defer dev.destroyShaderModule(ps);

    const vbuf = try dev.createResource(.{ .buffer = .{ .size = 3 * 16, .usage = .{ .vertex = true } } });
    defer dev.destroyResource(vbuf);
    // Fullscreen triangle (clip -1..3): the GPU guardband-clips it to cover every pixel
    // of the render target, giving a full-frame gradient to sample.
    const verts = [_]f32{
        -1.0, 3.0, 0.0, 1.0, // top (off-screen high), covers the top
        -1.0, -1.0, 0.0, 1.0, // bottom-left
        3.0, -1.0, 0.0, 1.0, // bottom-right (off-screen right)
    };
    @memcpy(std.mem.bytesAsSlice(f32, try dev.mapResource(vbuf))[0..verts.len], &verts);

    const attrs = [_]hal.VertexAttribute{
        .{ .location = 0, .format = .rgba8_unorm, .offset = 0 },
    };
    const pipe = try dev.createPipeline(.{
        .vertex = vs,
        .fragment = ps,
        .vertex_layout = .{ .stride = 16, .attributes = &attrs },
        .color_format = .bgra8_unorm,
    });
    defer dev.destroyPipeline(pipe);

    const ctx = try dev.createContext();
    defer ctx.deinit();
    {
        const cb = try ctx.beginCommands();
        defer cb.deinit();
        try cb.setRenderTarget(rt);
        try cb.clear(clear);
        try cb.bindPipeline(pipe);
        try cb.bindVertexBuffer(vbuf);
        try cb.draw(3, 0);
        try ctx.submit(cb);
    }

    const fb = std.mem.bytesAsSlice(u32, try dev.mapResource(rt));
    const out = try gpa.alloc(u32, @as(usize, w) * h);
    errdefer gpa.free(out);
    const pitch_px = w;
    var y: u32 = 0;
    while (y < h) : (y += 1) {
        @memcpy(out[@as(usize, y) * w ..][0..w], fb[@as(usize, y) * pitch_px ..][0..w]);
    }
    return out;
}

/// Like `run`, but the fragment shader colors the triangle by gl_FrontFacing (green
/// front, red back) via `buildFrontFaceFragmentSpirv`. `reversed` swaps two vertices
/// to flip the primitive winding. With culling disabled, the standard winding renders
/// front (green) and the reversed one back (red). Works on any `hal.Device` so both
/// nvidia and software can be verified. Returns A8R8G8B8 u32s packed w*h, caller-owned.
pub fn runFrontFace(gpa: std.mem.Allocator, dev: hal.Device, w: u32, h: u32, clear: hal.Color, reversed: bool) ![]u32 {
    var vs_b = try buildVertexSpirv(gpa);
    defer vs_b.deinit(gpa);
    var ps_b = try buildFrontFaceFragmentSpirv(gpa);
    defer ps_b.deinit(gpa);

    const rt = try dev.createResource(.{ .image = .{ .width = w, .height = h, .format = .bgra8_unorm, .usage = .{ .render_target = true } } });
    defer dev.destroyResource(rt);

    const vs = try dev.createShaderModule(.{ .stage = .vertex, .code = std.mem.sliceAsBytes(vs_b.words.items) });
    defer dev.destroyShaderModule(vs);
    const ps = try dev.createShaderModule(.{ .stage = .fragment, .code = std.mem.sliceAsBytes(ps_b.words.items) });
    defer dev.destroyShaderModule(ps);

    const vbuf = try dev.createResource(.{ .buffer = .{ .size = 3 * 16, .usage = .{ .vertex = true } } });
    defer dev.destroyResource(vbuf);
    const top = [4]f32{ 0.0, 0.5, 0.0, 1.0 };
    const bl = [4]f32{ -0.5, -0.5, 0.0, 1.0 };
    const br = [4]f32{ 0.5, -0.5, 0.0, 1.0 };
    // reversed swaps bl<->br, flipping the winding (front <-> back).
    const verts: [12]f32 = if (reversed)
        top ++ br ++ bl
    else
        top ++ bl ++ br;
    @memcpy(std.mem.bytesAsSlice(f32, try dev.mapResource(vbuf))[0..verts.len], &verts);

    const attrs = [_]hal.VertexAttribute{
        .{ .location = 0, .format = .rgba8_unorm, .offset = 0 },
    };
    const pipe = try dev.createPipeline(.{
        .vertex = vs,
        .fragment = ps,
        .vertex_layout = .{ .stride = 16, .attributes = &attrs },
        .color_format = .bgra8_unorm,
    });
    defer dev.destroyPipeline(pipe);

    const ctx = try dev.createContext();
    defer ctx.deinit();
    {
        const cb = try ctx.beginCommands();
        defer cb.deinit();
        try cb.setRenderTarget(rt);
        try cb.clear(clear);
        try cb.bindPipeline(pipe);
        try cb.bindVertexBuffer(vbuf);
        try cb.draw(3, 0);
        try ctx.submit(cb);
    }

    const fb = std.mem.bytesAsSlice(u32, try dev.mapResource(rt));
    const out = try gpa.alloc(u32, @as(usize, w) * h);
    errdefer gpa.free(out);
    const pitch_px = w;
    var y: u32 = 0;
    while (y < h) : (y += 1) {
        @memcpy(out[@as(usize, y) * w ..][0..w], fb[@as(usize, y) * pitch_px ..][0..w]);
    }
    return out;
}

/// Render the gradient triangle from SPIR-V VS/PS: two-attribute layout (vec4 position
/// at offset 0, vec4 color at offset 16, stride 32). The VS passes position and a color
/// varying through. The PS reads the interpolated varying (not a constant). Geometry is
/// buildNvidia's gold-reference gradient: top=red, bottom-left=green, bottom-right=blue.
/// Returns the caller-owned tightly-packed w*h readback.
pub fn runGradient(gpa: std.mem.Allocator, dev: hal.Device, w: u32, h: u32, clear: hal.Color) ![]u32 {
    // SPIR-V -> Vulcan IR -> SASS for both stages.
    var vs_b = try buildGradientVertexSpirv(gpa);
    defer vs_b.deinit(gpa);
    var ps_b = try buildGradientFragmentSpirv(gpa);
    defer ps_b.deinit(gpa);
    const vs_sass = try compileToSass(gpa, std.mem.sliceAsBytes(vs_b.words.items), .vertex);
    defer gpa.free(vs_sass);
    const ps_sass = try compileToSass(gpa, std.mem.sliceAsBytes(ps_b.words.items), .fragment);
    defer gpa.free(ps_sass);

    const rt = try dev.createResource(.{ .image = .{ .width = w, .height = h, .format = .bgra8_unorm, .usage = .{ .render_target = true } } });
    defer dev.destroyResource(rt);

    const vs = try dev.createShaderModule(.{ .stage = .vertex, .code = std.mem.sliceAsBytes(vs_sass) });
    defer dev.destroyShaderModule(vs);
    const ps = try dev.createShaderModule(.{ .stage = .fragment, .code = std.mem.sliceAsBytes(ps_sass) });
    defer dev.destroyShaderModule(ps);

    // Vertex buffer: 3 verts of (vec4 position, vec4 color), the exact gradient
    // geometry from triangle_hal's buildNvidia. Stride 32, two attributes.
    const vbuf = try dev.createResource(.{ .buffer = .{ .size = 3 * 32, .usage = .{ .vertex = true } } });
    defer dev.destroyResource(vbuf);
    const verts = [_]f32{
        0.0, 0.5, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, // top          -> red
        -0.5, -0.5, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, // bottom-left  -> green
        0.5, -0.5, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, // bottom-right -> blue
    };
    @memcpy(std.mem.bytesAsSlice(f32, try dev.mapResource(vbuf))[0..verts.len], &verts);

    const attrs = [_]hal.VertexAttribute{
        .{ .location = 0, .format = .rgba8_unorm, .offset = 0 }, // clip-space position
        .{ .location = 1, .format = .rgba8_unorm, .offset = 16 }, // per-vertex color
    };
    const pipe = try dev.createPipeline(.{
        .vertex = vs,
        .fragment = ps,
        .vertex_layout = .{ .stride = 32, .attributes = &attrs },
        .color_format = .bgra8_unorm,
    });
    defer dev.destroyPipeline(pipe);

    const ctx = try dev.createContext();
    defer ctx.deinit();
    {
        const cb = try ctx.beginCommands();
        defer cb.deinit();
        try cb.setRenderTarget(rt);
        try cb.clear(clear);
        try cb.bindPipeline(pipe);
        try cb.bindVertexBuffer(vbuf);
        try cb.draw(3, 0);
        try ctx.submit(cb);
    }

    const fb = std.mem.bytesAsSlice(u32, try dev.mapResource(rt));
    const out = try gpa.alloc(u32, @as(usize, w) * h);
    errdefer gpa.free(out);
    // Tight (w*4) readback: the software RT is tight and nvidia mapResource de-swizzles
    // the block-linear RT into a tight w*4 buffer. (pitchBytes(w)/4 == w only at 256.)
    const pitch_px = w;
    var y: u32 = 0;
    while (y < h) : (y += 1) {
        @memcpy(out[@as(usize, y) * w ..][0..w], fb[@as(usize, y) * pitch_px ..][0..w]);
    }
    return out;
}

const NvDevice = @import("device.zig").Device;

/// Render a gradient triangle by handing the HAL `createShaderModule` raw SPIR-V
/// word streams, not pre-compiled SASS. This exercises the exact path the Vulkan ICD
/// takes: the nvidia HAL detects the SPIR-V magic and compiles to SASS internally.
/// Returns the caller-owned tightly-packed readback.
pub fn runGradientFromSpirvModule(gpa: std.mem.Allocator, dev: hal.Device, w: u32, h: u32, clear: hal.Color) ![]u32 {
    // The VS/FS SPIR-V word streams, fed to the HAL AS SPIR-V (the ICD path).
    var vs_b = try buildGradientVertexSpirv(gpa);
    defer vs_b.deinit(gpa);
    var ps_b = try buildGradientFragmentSpirv(gpa);
    defer ps_b.deinit(gpa);
    const vs_spirv = std.mem.sliceAsBytes(vs_b.words.items);
    const ps_spirv = std.mem.sliceAsBytes(ps_b.words.items);

    const rt = try dev.createResource(.{ .image = .{ .width = w, .height = h, .format = .bgra8_unorm, .usage = .{ .render_target = true } } });
    defer dev.destroyResource(rt);

    // KEY: pass SPIR-V, not SASS. createShaderModule compiles it to SASS.
    const vs = try dev.createShaderModule(.{ .stage = .vertex, .code = vs_spirv });
    defer dev.destroyShaderModule(vs);
    const ps = try dev.createShaderModule(.{ .stage = .fragment, .code = ps_spirv });
    defer dev.destroyShaderModule(ps);

    const vbuf = try dev.createResource(.{ .buffer = .{ .size = 3 * 32, .usage = .{ .vertex = true } } });
    defer dev.destroyResource(vbuf);
    const verts = [_]f32{
        0.0, 0.5, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, // top          -> red
        -0.5, -0.5, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, // bottom-left  -> green
        0.5, -0.5, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, // bottom-right -> blue
    };
    @memcpy(std.mem.bytesAsSlice(f32, try dev.mapResource(vbuf))[0..verts.len], &verts);

    const attrs = [_]hal.VertexAttribute{
        .{ .location = 0, .format = .rgba8_unorm, .offset = 0 },
        .{ .location = 1, .format = .rgba8_unorm, .offset = 16 },
    };
    const pipe = try dev.createPipeline(.{
        .vertex = vs,
        .fragment = ps,
        .vertex_layout = .{ .stride = 32, .attributes = &attrs },
        .color_format = .bgra8_unorm,
    });
    defer dev.destroyPipeline(pipe);

    const ctx = try dev.createContext();
    defer ctx.deinit();
    {
        const cb = try ctx.beginCommands();
        defer cb.deinit();
        try cb.setRenderTarget(rt);
        try cb.clear(clear);
        try cb.bindPipeline(pipe);
        try cb.bindVertexBuffer(vbuf);
        try cb.draw(3, 0);
        try ctx.submit(cb);
    }

    const fb = std.mem.bytesAsSlice(u32, try dev.mapResource(rt));
    const out = try gpa.alloc(u32, @as(usize, w) * h);
    errdefer gpa.free(out);
    // Tight (w*4) readback: the software RT is tight and nvidia mapResource de-swizzles
    // the block-linear RT into a tight w*4 buffer. (pitchBytes(w)/4 == w only at 256.)
    const pitch_px = w;
    var y: u32 = 0;
    while (y < h) : (y += 1) {
        @memcpy(out[@as(usize, y) * w ..][0..w], fb[@as(usize, y) * pitch_px ..][0..w]);
    }
    return out;
}

/// UBO fragment shader as SPIR-V: a uniform block (binding 0) holds a single vec4
/// `color`. Output location 0 = that color. The body does out = u.color (a std140 UBO
/// vec4 load). Tests per-draw uniform binding: several draws in one submit each bind a
/// different UBO and read their own color. Returns owned words.
fn buildUboColorFragmentSpirv(gpa: std.mem.Allocator) !Builder {
    // ids: void=1 f32=2 v4f=3 pOut=4 voidfn=5 ublock=6 pUbo=7 uboVar=8 outVar=9 main=10
    //      entry=11 int=12 c0=13 pColor=14 colorPtr=15 loaded=16.
    var b = try Builder.init(gpa, 17);
    errdefer b.deinit(gpa);
    try b.emit(gpa, op.EntryPoint, &.{ op.ExecutionModel.fragment, 10, 0, 9 });
    try b.emit(gpa, op.Decorate, &.{ 9, op.Decoration.location, 0 });
    try b.emit(gpa, op.Decorate, &.{ 6, op.Decoration.block });
    try b.emit(gpa, op.MemberDecorate, &.{ 6, 0, op.Decoration.offset, 0 });
    try b.emit(gpa, op.Decorate, &.{ 8, op.Decoration.binding, 0 });
    try b.emit(gpa, op.Decorate, &.{ 8, op.Decoration.descriptor_set, 0 });
    try b.emit(gpa, op.TypeVoid, &.{1});
    try b.emit(gpa, op.TypeFloat, &.{ 2, 32 });
    try b.emit(gpa, op.TypeVector, &.{ 3, 2, 4 });
    try b.emit(gpa, op.TypePointer, &.{ 4, op.StorageClass.output, 3 });
    try b.emit(gpa, op.TypeFunction, &.{ 5, 1 });
    try b.emit(gpa, op.TypeStruct, &.{ 6, 3 }); // UBO block { vec4 color }
    try b.emit(gpa, op.TypePointer, &.{ 7, op.StorageClass.uniform, 6 });
    try b.emit(gpa, op.Variable, &.{ 7, 8, op.StorageClass.uniform }); // the UBO
    try b.emit(gpa, op.Variable, &.{ 4, 9, op.StorageClass.output }); // color out
    try b.emit(gpa, op.TypeInt, &.{ 12, 32, 1 });
    try b.emit(gpa, op.Constant, &.{ 12, 13, 0 }); // member index 0
    try b.emit(gpa, op.TypePointer, &.{ 14, op.StorageClass.uniform, 3 }); // ptr to vec4
    try b.emit(gpa, op.Function, &.{ 1, 10, 0, 5 });
    try b.emit(gpa, op.Label, &.{11});
    try b.emit(gpa, op.AccessChain, &.{ 14, 15, 8, 13 }); // &u.color
    try b.emit(gpa, op.Load, &.{ 3, 16, 15 }); // color = load u.color
    try b.emit(gpa, op.Store, &.{ 9, 16 }); // out = color
    try b.emit(gpa, op.Return, &.{});
    try b.emit(gpa, op.FunctionEnd, &.{});
    return b;
}

/// A UBO vertex shader as SPIR-V: input location 0 = position vec4; a uniform block
/// (binding 0) holding a mat4 `mvp`; output = the Position builtin. The body does
/// gl_Position = mvp * inPos (a UBO matrix*vector transform). The Vulcan lowering
/// appends the UBO as a pointer entry param, reads the std140 mat4 members through it
/// (load), and lowers the multiply to FMUL/FADD. The NVIDIA backend sources the UBO
/// pointer from constant bank (LDC) and loads the members (LDG). Returns owned words.
fn buildUboVertexSpirv(gpa: std.mem.Allocator) !Builder {
    // ids: void=1 f32=2 v4f=3 m4=4 pIn=5 pOut=6 voidfn=7 posIn=8 posOut=9 ublock=10
    //      pUbo=11 uboVar=12 main=13 entry=14 int=15 c0=16 mvpPtr=17 ldM=18 ldPos=19
    //      prod=20, mvp=22.
    var b = try Builder.init(gpa, 23);
    errdefer b.deinit(gpa);
    try b.emit(gpa, op.EntryPoint, &.{ op.ExecutionModel.vertex, 13, 0, 8, 9 });
    try b.emit(gpa, op.Decorate, &.{ 8, op.Decoration.location, 0 });
    try b.emit(gpa, op.Decorate, &.{ 9, op.Decoration.builtin, op.BuiltIn.position });
    // The UBO block: a mat4 member at offset 0, std140 matrix stride 16, ColMajor.
    try b.emit(gpa, op.Decorate, &.{ 10, op.Decoration.block });
    try b.emit(gpa, op.MemberDecorate, &.{ 10, 0, op.Decoration.col_major });
    try b.emit(gpa, op.MemberDecorate, &.{ 10, 0, op.Decoration.matrix_stride, 16 });
    try b.emit(gpa, op.MemberDecorate, &.{ 10, 0, op.Decoration.offset, 0 });
    try b.emit(gpa, op.Decorate, &.{ 12, op.Decoration.binding, 0 });
    try b.emit(gpa, op.Decorate, &.{ 12, op.Decoration.descriptor_set, 0 });
    try b.emit(gpa, op.TypeVoid, &.{1});
    try b.emit(gpa, op.TypeFloat, &.{ 2, 32 });
    try b.emit(gpa, op.TypeVector, &.{ 3, 2, 4 }); // v4float
    try b.emit(gpa, op.TypeMatrix, &.{ 4, 3, 4 }); // mat4 (4 columns of v4float)
    try b.emit(gpa, op.TypePointer, &.{ 5, op.StorageClass.input, 3 });
    try b.emit(gpa, op.TypePointer, &.{ 6, op.StorageClass.output, 3 });
    try b.emit(gpa, op.TypeFunction, &.{ 7, 1 });
    try b.emit(gpa, op.Variable, &.{ 5, 8, op.StorageClass.input }); // position input
    try b.emit(gpa, op.Variable, &.{ 6, 9, op.StorageClass.output }); // gl_Position
    try b.emit(gpa, op.TypeStruct, &.{ 10, 4 }); // UBO block { mat4 }
    try b.emit(gpa, op.TypePointer, &.{ 11, op.StorageClass.uniform, 10 });
    try b.emit(gpa, op.Variable, &.{ 11, 12, op.StorageClass.uniform }); // the UBO
    try b.emit(gpa, op.TypeInt, &.{ 15, 32, 1 });
    try b.emit(gpa, op.Constant, &.{ 15, 16, 0 }); // member index 0
    try b.emit(gpa, op.TypePointer, &.{ 17, op.StorageClass.uniform, 4 }); // ptr to mat4
    try b.emit(gpa, op.Function, &.{ 1, 13, 0, 7 });
    try b.emit(gpa, op.Label, &.{14});
    try b.emit(gpa, op.Load, &.{ 3, 19, 8 }); // pos = load position input
    try b.emit(gpa, op.AccessChain, &.{ 17, 18, 12, 16 }); // &u.mvp
    const mat_load = 22;
    try b.emit(gpa, op.Load, &.{ 4, mat_load, 18 }); // mvp = load the mat4
    try b.emit(gpa, op.MatrixTimesVector, &.{ 3, 20, mat_load, 19 }); // mvp * pos
    try b.emit(gpa, op.Store, &.{ 9, 20 }); // gl_Position = mvp * pos
    try b.emit(gpa, op.Return, &.{});
    try b.emit(gpa, op.FunctionEnd, &.{});
    return b;
}

/// Vertex-pulling vertex shader as SPIR-V (vkcube's pattern): no vertex inputs. A
/// uniform block (binding 0) holds a vec4 array `pos[3]`. The body reads gl_VertexIndex
/// and does gl_Position = u.pos[gl_VertexIndex] via a dynamic UBO-array index. Vulcan
/// synthesizes gl_VertexIndex as a leading i32 builtin param and lowers OpAccessChain to
/// base + index*ArrayStride. The NVIDIA backend sources gl_VertexIndex from ALD a[0x2fc]
/// (DA vertex id), scales it (IMAD), adds the UBO base (LDC), and LDGs.
fn buildPullVertexSpirv(gpa: std.mem.Allocator) !Builder {
    // ids: void=1 f32=2 v4f=3 pOut=4 voidfn=5 posOut=6 uint=7 u3=8 arr=9 ublock=10
    //      pUbo=11 uboVar=12 int=13 c0=14 pInInt=15 viVar=16 pUboV4=17 main=18 entry=19
    //      vi=20 elemPtr=21 posVal=22 posPtr=23.
    var b = try Builder.init(gpa, 24);
    errdefer b.deinit(gpa);
    try b.emit(gpa, op.EntryPoint, &.{ op.ExecutionModel.vertex, 18, 0, 6, 16 });
    try b.emit(gpa, op.Decorate, &.{ 6, op.Decoration.builtin, op.BuiltIn.position });
    try b.emit(gpa, op.Decorate, &.{ 16, op.Decoration.builtin, op.BuiltIn.vertex_index });
    // The UBO block: a vec4[3] array member at offset 0, std140 array stride 16.
    try b.emit(gpa, op.Decorate, &.{ 9, op.Decoration.array_stride, 16 });
    try b.emit(gpa, op.Decorate, &.{ 10, op.Decoration.block });
    try b.emit(gpa, op.MemberDecorate, &.{ 10, 0, op.Decoration.offset, 0 });
    try b.emit(gpa, op.Decorate, &.{ 12, op.Decoration.binding, 0 });
    try b.emit(gpa, op.Decorate, &.{ 12, op.Decoration.descriptor_set, 0 });
    try b.emit(gpa, op.TypeVoid, &.{1});
    try b.emit(gpa, op.TypeFloat, &.{ 2, 32 });
    try b.emit(gpa, op.TypeVector, &.{ 3, 2, 4 }); // v4float
    try b.emit(gpa, op.TypePointer, &.{ 4, op.StorageClass.output, 3 });
    try b.emit(gpa, op.TypeFunction, &.{ 5, 1 });
    try b.emit(gpa, op.Variable, &.{ 4, 6, op.StorageClass.output }); // gl_Position
    try b.emit(gpa, op.TypeInt, &.{ 7, 32, 0 }); // uint
    try b.emit(gpa, op.Constant, &.{ 7, 8, 3 }); // array length 3
    try b.emit(gpa, op.TypeArray, &.{ 9, 3, 8 }); // vec4[3]
    try b.emit(gpa, op.TypeStruct, &.{ 10, 9 }); // UBO block { vec4[3] }
    try b.emit(gpa, op.TypePointer, &.{ 11, op.StorageClass.uniform, 10 });
    try b.emit(gpa, op.Variable, &.{ 11, 12, op.StorageClass.uniform }); // the UBO
    try b.emit(gpa, op.TypeInt, &.{ 13, 32, 1 }); // int
    try b.emit(gpa, op.Constant, &.{ 13, 14, 0 }); // member index 0
    try b.emit(gpa, op.TypePointer, &.{ 15, op.StorageClass.input, 13 }); // ptr Input int
    try b.emit(gpa, op.Variable, &.{ 15, 16, op.StorageClass.input }); // gl_VertexIndex
    try b.emit(gpa, op.TypePointer, &.{ 17, op.StorageClass.uniform, 3 }); // ptr to vec4
    try b.emit(gpa, op.Function, &.{ 1, 18, 0, 5 });
    try b.emit(gpa, op.Label, &.{19});
    try b.emit(gpa, op.Load, &.{ 13, 20, 16 }); // vi = load gl_VertexIndex
    try b.emit(gpa, op.AccessChain, &.{ 17, 21, 12, 14, 20 }); // &u.pos[vi]
    try b.emit(gpa, op.Load, &.{ 3, 22, 21 }); // pos = load u.pos[vi]
    try b.emit(gpa, op.Store, &.{ 6, 22 }); // gl_Position = pos
    try b.emit(gpa, op.Return, &.{});
    try b.emit(gpa, op.FunctionEnd, &.{});
    return b;
}

test "a vertex-pulling VS (gl_VertexIndex -> UBO array) compiles to SASS with ALD a[0x2fc] + IMAD + LDC + LDG" {
    const gpa = std.testing.allocator;
    var b = try buildPullVertexSpirv(gpa);
    defer b.deinit(gpa);
    const sass = try compileToSass(gpa, std.mem.sliceAsBytes(b.words.items), .vertex);
    defer gpa.free(sass);

    // gl_VertexIndex from ALD a[0x2fc] (the DA-delivered vertex id), IMAD (index*stride),
    // LDC (the UBO base address from constant bank), LDG (the dynamic array element).
    var has_ald_vid = false;
    var has_imad = false;
    var has_ldc = false;
    var has_ldg = false;
    var i: usize = 0;
    while (i < sass.len) : (i += 4) {
        switch (sass[i] & 0xfff) {
            0x321 => if ((sass[i + 1] >> 8) & 0x3ff == 0x2fc) {
                has_ald_vid = true;
            },
            0x224 => has_imad = true,
            0xb82 => has_ldc = true,
            0x981 => has_ldg = true,
            else => {},
        }
    }
    try std.testing.expect(has_ald_vid); // gl_VertexIndex sourced from the vertex-id attribute
    try std.testing.expect(has_imad); // index * std140 stride
    try std.testing.expect(has_ldc); // UBO base from constant bank
    try std.testing.expect(has_ldg); // the dynamic array element load
}

test "a UBO mat4 vertex shader compiles to SASS with an LDC address load + LDG + FMUL" {
    const gpa = std.testing.allocator;
    var b = try buildUboVertexSpirv(gpa);
    defer b.deinit(gpa);
    const sass = try compileToSass(gpa, std.mem.sliceAsBytes(b.words.items), .vertex);
    defer gpa.free(sass);

    // The UBO path emits: LDC (the UBO base address from constant bank), LDG (the
    // std140 mat4 member loads), and FMUL (the matrix*vector multiply). None of these
    // appear in the pass-through milestone-1 shader.
    var has_ldc = false;
    var has_ldg = false;
    var has_fmul = false;
    var i: usize = 0;
    while (i < sass.len) : (i += 4) {
        switch (sass[i] & 0xfff) {
            0xb82 => has_ldc = true,
            0x981 => has_ldg = true,
            0x220 => has_fmul = true,
            else => {},
        }
    }
    try std.testing.expect(has_ldc);
    try std.testing.expect(has_ldg);
    try std.testing.expect(has_fmul);
}

test "HAL createShaderModule compiles SPIR-V to SASS and renders on the NVIDIA GPU (skips without a GPU)" {
    const gpa = std.testing.allocator;
    const dev = NvDevice.create(gpa) catch return error.SkipZigTest;
    defer dev.deinit();
    const W: u32 = 256;
    const H: u32 = 256;

    const clear = hal.Color{ .r = 0.125, .g = 0.125, .b = 0.125, .a = 1 };
    const fb = try runGradientFromSpirvModule(gpa, dev, W, H, clear);
    defer gpa.free(fb);

    const clear_px: u32 = 0xff202020;
    const get = struct {
        fn r(p: u32) u8 {
            return @truncate(p >> 16);
        }
        fn g(p: u32) u8 {
            return @truncate(p >> 8);
        }
        fn b(p: u32) u8 {
            return @truncate(p);
        }
    };
    var changed: u32 = 0;
    for (fb) |p| {
        if (p != clear_px) changed += 1;
    }
    try std.testing.expect(changed > 4000);
    // The gradient corners. With the software-matching Y-origin (NDC -1 -> row 0),
    // the clip-top red vertex (clip y=+0.5) lands LOW (~row 176); the clip-bottom
    // green/blue verts (clip y=-0.5) land HIGH (~row 78).
    const red_c = fb[176 * W + 128];
    const grn_c = fb[78 * W + 84];
    const blu_c = fb[78 * W + 172];
    try std.testing.expect(red_c != clear_px and grn_c != clear_px and blu_c != clear_px);
    try std.testing.expect(get.r(red_c) > get.g(red_c) and get.r(red_c) > get.b(red_c));
    try std.testing.expect(get.g(grn_c) > get.r(grn_c) and get.g(grn_c) > get.b(grn_c));
    try std.testing.expect(get.b(blu_c) > get.r(blu_c) and get.b(blu_c) > get.g(blu_c));
}

test "SPIR-V flat-color triangle renders on the NVIDIA GPU (skips without a GPU)" {
    const gpa = std.testing.allocator;
    const dev = NvDevice.create(gpa) catch return error.SkipZigTest;
    defer dev.deinit();
    const W: u32 = 256;
    const H: u32 = 256;

    // Clear dark grey (0xff202020 as A8R8G8B8) and draw a solid red triangle.
    const clear = hal.Color{ .r = 0.125, .g = 0.125, .b = 0.125, .a = 1 };
    const red = [4]f32{ 1.0, 0.0, 0.0, 1.0 };
    const fb = try run(gpa, dev, W, H, clear, red);
    defer gpa.free(fb);

    const clear_px: u32 = 0xff202020;

    // A substantial region is painted (the clip [-0.5,0.5] triangle covers ~8192
    // of the 256x256 pixels).
    var changed: u32 = 0;
    for (fb) |p| {
        if (p != clear_px) changed += 1;
    }
    try std.testing.expect(changed > 4000);

    // Sample points: the triangle centroid is the constant color (red); a render-
    // target corner is the clear color. A8R8G8B8: red = 0xffff0000.
    const get = struct {
        fn r(p: u32) u8 {
            return @truncate(p >> 16);
        }
        fn g(p: u32) u8 {
            return @truncate(p >> 8);
        }
        fn b(p: u32) u8 {
            return @truncate(p);
        }
    };
    // Centroid of (clip-top=(0,0.5), bl=(-0.5,-0.5), br=(0.5,-0.5)) maps to pixel
    // (~128, ~107) for a 256x256 RT (clip-x in [-1,1] -> [0,W], py=(y+1)/2*H, NDC -1
    // -> row 0 = software-matching). The solid triangle spans py 64..192.
    const centroid = fb[107 * W + 128];
    try std.testing.expect(centroid != clear_px);
    try std.testing.expect(get.r(centroid) > 200 and get.g(centroid) < 64 and get.b(centroid) < 64); // red
    // Top-left corner is outside the triangle: the clear color.
    try std.testing.expectEqual(clear_px, fb[0]);
}

test "ORACLE-FLOAT-RT: an HDR value (>1.0) renders into an rgba16f RT and reads back UNCLAMPED on the NVIDIA GPU (skips without a GPU)" {
    const gpa = std.testing.allocator;
    const dev = NvDevice.create(gpa) catch return error.SkipZigTest;
    defer dev.deinit();
    const W: u32 = 64;
    const H: u32 = 64;

    var vs_b = try buildVertexSpirv(gpa);
    defer vs_b.deinit(gpa);
    // HDR red = 2.0, which an 8-bit color target would clamp to 1.0. 2.0/0.5/0.25 are exact in fp16.
    var ps_b = try buildFragmentSpirv(gpa, .{ 2.0, 0.5, 0.25, 1.0 });
    defer ps_b.deinit(gpa);

    // An rgba16_float block-linear color target (rendered at full precision by the ROP).
    const rt = try dev.createResource(.{ .image = .{ .width = W, .height = H, .format = .rgba16_float, .usage = .{ .render_target = true } } });
    defer dev.destroyResource(rt);
    const vs = try dev.createShaderModule(.{ .stage = .vertex, .code = std.mem.sliceAsBytes(vs_b.words.items) });
    defer dev.destroyShaderModule(vs);
    const ps = try dev.createShaderModule(.{ .stage = .fragment, .code = std.mem.sliceAsBytes(ps_b.words.items) });
    defer dev.destroyShaderModule(ps);
    const vbuf = try dev.createResource(.{ .buffer = .{ .size = 3 * 16, .usage = .{ .vertex = true } } });
    defer dev.destroyResource(vbuf);
    const verts = [_]f32{
        0.0, 0.8, 0.0, 1.0, // top
        -0.8, -0.8, 0.0, 1.0, // bottom-left
        0.8, -0.8, 0.0, 1.0, // bottom-right
    };
    @memcpy(std.mem.bytesAsSlice(f32, try dev.mapResource(vbuf))[0..verts.len], &verts);
    const attrs = [_]hal.VertexAttribute{.{ .location = 0, .format = .rgba8_unorm, .offset = 0 }};
    const pipe = try dev.createPipeline(.{
        .vertex = vs,
        .fragment = ps,
        .vertex_layout = .{ .stride = 16, .attributes = &attrs },
        .color_format = .rgba16_float,
    });
    defer dev.destroyPipeline(pipe);

    const ctx = try dev.createContext();
    defer ctx.deinit();
    {
        const cb = try ctx.beginCommands();
        defer cb.deinit();
        try cb.setRenderTarget(rt);
        try cb.clear(.{ .r = 0, .g = 0, .b = 0, .a = 1 });
        try cb.bindPipeline(pipe);
        try cb.bindVertexBuffer(vbuf);
        try cb.draw(3, 0);
        try ctx.submit(cb);
    }

    // mapResource de-swizzles the block-linear rgba16f RT to tight row-major fp16 texels.
    const raw = try dev.mapResource(rt);
    const ci = (@as(usize, 32) * W + 32) * 8; // center pixel, 8 bytes/texel (4x fp16)
    const rd: f16 = @bitCast(std.mem.readInt(u16, raw[ci + 0 ..][0..2], .little));
    const gd: f16 = @bitCast(std.mem.readInt(u16, raw[ci + 2 ..][0..2], .little));
    const bd: f16 = @bitCast(std.mem.readInt(u16, raw[ci + 4 ..][0..2], .little));
    // The GPU wrote the true HDR value: red is 2.0, not clamped to 1.0.
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), @as(f32, @floatCast(rd)), 1e-2);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), @as(f32, @floatCast(gd)), 1e-2);
    try std.testing.expectApproxEqAbs(@as(f32, 0.25), @as(f32, @floatCast(bd)), 1e-2);
}

test "ORACLE-FLOAT-RT32: an HDR value renders into an rgba32f RT and reads back UNCLAMPED on the NVIDIA GPU (skips without a GPU)" {
    const gpa = std.testing.allocator;
    const dev = NvDevice.create(gpa) catch return error.SkipZigTest;
    defer dev.deinit();
    const W: u32 = 64;
    const H: u32 = 64;

    var vs_b = try buildVertexSpirv(gpa);
    defer vs_b.deinit(gpa);
    var ps_b = try buildFragmentSpirv(gpa, .{ 3.5, 0.25, 0.75, 1.0 }); // HDR red 3.5 (>1.0)
    defer ps_b.deinit(gpa);

    // A 16-byte-per-texel rgba32f block-linear color target (exercises the bpp=16 de-swizzle).
    const rt = try dev.createResource(.{ .image = .{ .width = W, .height = H, .format = .r32g32b32a32_float, .usage = .{ .render_target = true } } });
    defer dev.destroyResource(rt);
    const vs = try dev.createShaderModule(.{ .stage = .vertex, .code = std.mem.sliceAsBytes(vs_b.words.items) });
    defer dev.destroyShaderModule(vs);
    const ps = try dev.createShaderModule(.{ .stage = .fragment, .code = std.mem.sliceAsBytes(ps_b.words.items) });
    defer dev.destroyShaderModule(ps);
    const vbuf = try dev.createResource(.{ .buffer = .{ .size = 3 * 16, .usage = .{ .vertex = true } } });
    defer dev.destroyResource(vbuf);
    const verts = [_]f32{ 0.0, 0.8, 0.0, 1.0, -0.8, -0.8, 0.0, 1.0, 0.8, -0.8, 0.0, 1.0 };
    @memcpy(std.mem.bytesAsSlice(f32, try dev.mapResource(vbuf))[0..verts.len], &verts);
    const attrs = [_]hal.VertexAttribute{.{ .location = 0, .format = .rgba8_unorm, .offset = 0 }};
    const pipe = try dev.createPipeline(.{
        .vertex = vs,
        .fragment = ps,
        .vertex_layout = .{ .stride = 16, .attributes = &attrs },
        .color_format = .r32g32b32a32_float,
    });
    defer dev.destroyPipeline(pipe);

    const ctx = try dev.createContext();
    defer ctx.deinit();
    {
        const cb = try ctx.beginCommands();
        defer cb.deinit();
        try cb.setRenderTarget(rt);
        try cb.clear(.{ .r = 0, .g = 0, .b = 0, .a = 1 });
        try cb.bindPipeline(pipe);
        try cb.bindVertexBuffer(vbuf);
        try cb.draw(3, 0);
        try ctx.submit(cb);
    }

    // mapResource de-swizzles the block-linear rgba32f RT to tight row-major fp32 texels.
    const raw = try dev.mapResource(rt);
    const ci = (@as(usize, 32) * W + 32) * 16; // center pixel, 16 bytes/texel (4x fp32)
    const rd: f32 = @bitCast(std.mem.readInt(u32, raw[ci + 0 ..][0..4], .little));
    const gd: f32 = @bitCast(std.mem.readInt(u32, raw[ci + 4 ..][0..4], .little));
    try std.testing.expectApproxEqAbs(@as(f32, 3.5), rd, 1e-4); // full fp32 HDR value, not clamped
    try std.testing.expectApproxEqAbs(@as(f32, 0.25), gd, 1e-4);
}

test "ORACLE-FLOAT-MSAA: a 4x-MSAA rgba16f target resolves an HDR value UNCLAMPED on the NVIDIA GPU (skips without a GPU)" {
    const gpa = std.testing.allocator;
    const dev = NvDevice.create(gpa) catch return error.SkipZigTest;
    defer dev.deinit();
    const W: u32 = 64;
    const H: u32 = 64;

    var vs_b = try buildVertexSpirv(gpa);
    defer vs_b.deinit(gpa);
    var ps_b = try buildFragmentSpirv(gpa, .{ 2.0, 0.5, 0.25, 1.0 }); // HDR red 2.0
    defer ps_b.deinit(gpa);

    // A 4x-multisampled rgba16f color target + a single-sample rgba16f resolve destination.
    const ms = try dev.createResource(.{ .image = .{ .width = W, .height = H, .format = .rgba16_float, .samples = 4, .usage = .{ .render_target = true } } });
    defer dev.destroyResource(ms);
    const resolved = try dev.createResource(.{ .image = .{ .width = W, .height = H, .format = .rgba16_float, .usage = .{ .render_target = true } } });
    defer dev.destroyResource(resolved);
    const vs = try dev.createShaderModule(.{ .stage = .vertex, .code = std.mem.sliceAsBytes(vs_b.words.items) });
    defer dev.destroyShaderModule(vs);
    const ps = try dev.createShaderModule(.{ .stage = .fragment, .code = std.mem.sliceAsBytes(ps_b.words.items) });
    defer dev.destroyShaderModule(ps);
    const vbuf = try dev.createResource(.{ .buffer = .{ .size = 3 * 16, .usage = .{ .vertex = true } } });
    defer dev.destroyResource(vbuf);
    // A large triangle fully covering the center (all 4 samples there average to the same HDR value).
    const verts = [_]f32{ 0.0, 0.9, 0.0, 1.0, -0.9, -0.9, 0.0, 1.0, 0.9, -0.9, 0.0, 1.0 };
    @memcpy(std.mem.bytesAsSlice(f32, try dev.mapResource(vbuf))[0..verts.len], &verts);
    const attrs = [_]hal.VertexAttribute{.{ .location = 0, .format = .rgba8_unorm, .offset = 0 }};
    const pipe = try dev.createPipeline(.{
        .vertex = vs,
        .fragment = ps,
        .vertex_layout = .{ .stride = 16, .attributes = &attrs },
        .color_format = .rgba16_float,
    });
    defer dev.destroyPipeline(pipe);

    const ctx = try dev.createContext();
    defer ctx.deinit();
    {
        const cb = try ctx.beginCommands();
        defer cb.deinit();
        try cb.setRenderTarget(ms);
        try cb.clear(.{ .r = 0, .g = 0, .b = 0, .a = 1 });
        try cb.bindPipeline(pipe);
        try cb.bindVertexBuffer(vbuf);
        try cb.draw(3, 0);
        try cb.resolve(ms, resolved, W, H, .rgba16_float, 4);
        try ctx.submit(cb);
    }

    // The resolve box-averaged the fp16 samples in float: a fully-covered center is still 2.0.
    const raw = try dev.mapResource(resolved);
    const ci = (@as(usize, 32) * W + 32) * 8; // 8 bytes/texel (4x fp16)
    const rd: f16 = @bitCast(std.mem.readInt(u16, raw[ci + 0 ..][0..2], .little));
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), @as(f32, @floatCast(rd)), 1e-2); // HDR survived the resolve
}

test "ORACLE-FLOAT-CLEAR: glClear writes an EXACT float value into an rgba16f RT on the NVIDIA GPU (skips without a GPU)" {
    const gpa = std.testing.allocator;
    const dev = NvDevice.create(gpa) catch return error.SkipZigTest;
    defer dev.deinit();
    const W: u32 = 32;
    const H: u32 = 32;

    // A clear-only pass (no draw): the ROP writes the float clear value in the target's format.
    const rt = try dev.createResource(.{ .image = .{ .width = W, .height = H, .format = .rgba16_float, .usage = .{ .render_target = true } } });
    defer dev.destroyResource(rt);
    const ctx = try dev.createContext();
    defer ctx.deinit();
    {
        const cb = try ctx.beginCommands();
        defer cb.deinit();
        try cb.setRenderTarget(rt);
        try cb.clear(.{ .r = 0.5, .g = 0.25, .b = 0.75, .a = 1.0 });
        try ctx.submit(cb);
    }

    // fp16 stores 0.5/0.25/0.75 exactly. An 8-bit RT would round (0.5 -> 128/255 = 0.502).
    const raw = try dev.mapResource(rt);
    const ci = (@as(usize, 16) * W + 16) * 8;
    const rd: f16 = @bitCast(std.mem.readInt(u16, raw[ci + 0 ..][0..2], .little));
    const gd: f16 = @bitCast(std.mem.readInt(u16, raw[ci + 2 ..][0..2], .little));
    const bd: f16 = @bitCast(std.mem.readInt(u16, raw[ci + 4 ..][0..2], .little));
    try std.testing.expectEqual(@as(f16, 0.5), rd);
    try std.testing.expectEqual(@as(f16, 0.25), gd);
    try std.testing.expectEqual(@as(f16, 0.75), bd);
}

test "ORACLE-DISCARD: an always-discard FS writes NO pixels on the NVIDIA GPU or in software (skips without a GPU)" {
    const gpa = std.testing.allocator;
    const nv = NvDevice.create(gpa) catch return error.SkipZigTest;
    defer nv.deinit();
    const sw = @import("../software.zig");
    const sw_dev = try sw.driver.createDevice(gpa);
    defer sw_dev.deinit();
    const W: u32 = 256;
    const H: u32 = 256;

    // Same scene as the flat-color oracle (a red triangle over dark grey), but the FS discards
    // every fragment. On nvidia, SASS KIL masks them. In software, the discard_fn thunk skips
    // the write. Either way the render target stays pure clear color (0 pixels changed).
    const clear = hal.Color{ .r = 0.125, .g = 0.125, .b = 0.125, .a = 1 };
    const red = [4]f32{ 1.0, 0.0, 0.0, 1.0 };
    const clear_px: u32 = 0xff202020;
    const check = struct {
        fn run(a: std.mem.Allocator, dev: hal.Device) !void {
            const fb = try runDiscard(a, dev, W, H, clear, red);
            defer a.free(fb);
            var changed: u32 = 0;
            for (fb) |p| {
                if (p != clear_px) changed += 1;
            }
            try std.testing.expectEqual(@as(u32, 0), changed); // every fragment discarded
            try std.testing.expectEqual(clear_px, fb[107 * W + 128]); // would-be centroid is clear
        }
    };
    try check.run(gpa, nv);
    try check.run(gpa, sw_dev);
}

test "ORACLE-FRAGCOORD: gl_FragCoord delivers the window-space pixel position on the NVIDIA GPU (skips without a GPU)" {
    const gpa = std.testing.allocator;
    const nv = NvDevice.create(gpa) catch return error.SkipZigTest;
    defer nv.deinit();
    const sw = @import("../software.zig");
    const sw_dev = try sw.driver.createDevice(gpa);
    defer sw_dev.deinit();
    const W: u32 = 256;
    const H: u32 = 256;
    const clear = hal.Color{ .r = 0, .g = 0, .b = 0, .a = 1 };

    // A full-screen triangle whose FS writes color = (FragCoord.x/256, FragCoord.y/256, 0, 1).
    // Pixel (px, py) reads back r ~= (px+0.5)/256, g ~= (py+0.5)/256. The GPU and the
    // software rasterizer share the window convention (y=0 at the top row), so both produce
    // the same screen-space gradient (sampled away from the (0,0) top-left vertex, which the
    // fill rule excludes on both).
    const check = struct {
        fn run(a: std.mem.Allocator, dev: hal.Device) !void {
            const fb = try runFragCoord(a, dev, W, H, clear, 256.0);
            defer a.free(fb);
            const get = struct {
                fn r(p: u32) u8 {
                    return @truncate(p >> 16);
                }
                fn g(p: u32) u8 {
                    return @truncate(p >> 8);
                }
            };
            // Expected component for coord c (pixel center c+0.5): round((c+0.5)/256*255).
            const expect = struct {
                fn v(c: u32) i32 {
                    const f: f32 = (@as(f32, @floatFromInt(c)) + 0.5) / 256.0 * 255.0;
                    return @intFromFloat(@round(f));
                }
            };
            const my: u32 = 128;
            inline for (.{ 32, 96, 160, 224 }) |px| {
                try std.testing.expect(@abs(@as(i32, get.r(fb[my * W + px])) - expect.v(px)) <= 4);
            }
            const mx: u32 = 128;
            inline for (.{ 32, 96, 160, 224 }) |py| {
                try std.testing.expect(@abs(@as(i32, get.g(fb[py * W + mx])) - expect.v(py)) <= 4);
            }
        }
    };
    try check.run(gpa, nv);
    try check.run(gpa, sw_dev);
}

test "ORACLE-FRONTFACE: gl_FrontFacing matches winding identically on software and the NVIDIA GPU (skips without a GPU)" {
    const gpa = std.testing.allocator;
    const nv = NvDevice.create(gpa) catch return error.SkipZigTest;
    defer nv.deinit();
    const sw = @import("../software.zig");
    const sw_dev = try sw.driver.createDevice(gpa);
    defer sw_dev.deinit();

    const W: u32 = 256;
    const H: u32 = 256;
    const clear = hal.Color{ .r = 0.125, .g = 0.125, .b = 0.125, .a = 1 };

    // Classify the triangle centroid: 'g' = green (front), 'r' = red (back), '?' = neither.
    const Cls = struct {
        fn of(fb: []const u32) u8 {
            const p = fb[107 * W + 128];
            const rr: u8 = @truncate(p >> 16);
            const gg: u8 = @truncate(p >> 8);
            if (gg > 180 and rr < 80) return 'g';
            if (rr > 180 and gg < 80) return 'r';
            return '?';
        }
    };

    // For both windings, the GPU and software classify the facing identically, and
    // the two windings classify oppositely (front vs back). This proves gl_FrontFacing
    // is delivered and its true/false paths both work, matching software exactly.
    var cls: [2]u8 = undefined;
    inline for (.{ false, true }, 0..) |rev, i| {
        const nv_fb = try runFrontFace(gpa, nv, W, H, clear, rev);
        defer gpa.free(nv_fb);
        const sw_fb = try runFrontFace(gpa, sw_dev, W, H, clear, rev);
        defer gpa.free(sw_fb);
        const nv_c = Cls.of(nv_fb);
        const sw_c = Cls.of(sw_fb);
        try std.testing.expect(nv_c == 'g' or nv_c == 'r'); // a definite facing color
        try std.testing.expectEqual(sw_c, nv_c); // GPU agrees with software
        cls[i] = nv_c;
    }
    try std.testing.expect(cls[0] != cls[1]); // the two windings face oppositely
}

test "ORACLE-FRAGDEPTH: gl_FragDepth governs the depth test AND write on the NVIDIA GPU and in software (skips without a GPU)" {
    const gpa = std.testing.allocator;
    const nv = NvDevice.create(gpa) catch return error.SkipZigTest;
    defer nv.deinit();
    const sw = @import("../software.zig");
    const sw_dev = try sw.driver.createDevice(gpa);
    defer sw_dev.deinit();
    const W: u32 = 128;
    const H: u32 = 128;

    // Two overlapping triangles with opposite interpolated z and gl_FragDepth, each drawn with
    // its own pipeline in a separate submit (one pipeline per submit, depth carries across via
    // the null-clear preserve). Green: geometry z=0.1 (near) but FragDepth 0.9 (far). Red:
    // geometry z=0.9 (far) but FragDepth 0.1 (near). Green first, then red, depth less + write:
    //  - correct (FragDepth governs test and write): green writes 0.9; red 0.1<0.9 passes -> red.
    //  - interpolated z (bug): green writes 0.1; red 0.9<0.1 fails -> green.
    // Red proves gl_FragDepth drives both the depth test and write on the GPU (late-Z) and in
    // software (the raster's late-Z path for depth-replace shaders).
    const check = struct {
        fn run(a: std.mem.Allocator, dev: hal.Device) !void {
            var vs_b = try buildVertexSpirv(a);
            defer vs_b.deinit(a);
            var green_b = try buildDepthFragmentSpirv(a, .{ 0, 1, 0, 1 }, 0.9); // green, FragDepth far
            defer green_b.deinit(a);
            var red_b = try buildDepthFragmentSpirv(a, .{ 1, 0, 0, 1 }, 0.1); // red, FragDepth near
            defer red_b.deinit(a);
            const vs = try dev.createShaderModule(.{ .stage = .vertex, .code = std.mem.sliceAsBytes(vs_b.words.items) });
            defer dev.destroyShaderModule(vs);
            const ps_green = try dev.createShaderModule(.{ .stage = .fragment, .code = std.mem.sliceAsBytes(green_b.words.items) });
            defer dev.destroyShaderModule(ps_green);
            const ps_red = try dev.createShaderModule(.{ .stage = .fragment, .code = std.mem.sliceAsBytes(red_b.words.items) });
            defer dev.destroyShaderModule(ps_red);
            const attrs = [_]hal.VertexAttribute{.{ .location = 0, .format = .rgba8_unorm, .offset = 0 }};
            const mkPipe = struct {
                fn f(d: hal.Device, v: *hal.ShaderModule, p: *hal.ShaderModule, at: []const hal.VertexAttribute) !*hal.Pipeline {
                    return d.createPipeline(.{ .vertex = v, .fragment = p, .vertex_layout = .{ .stride = 16, .attributes = at }, .color_format = .bgra8_unorm, .depth = .{ .test_enable = true, .write_enable = true, .compare_op = .less } });
                }
            }.f;
            const pipe_green = try mkPipe(dev, vs, ps_green, &attrs);
            defer dev.destroyPipeline(pipe_green);
            const pipe_red = try mkPipe(dev, vs, ps_red, &attrs);
            defer dev.destroyPipeline(pipe_red);
            const tri = struct {
                fn at(z: f32) [12]f32 {
                    return .{ 0.0, 0.6, z, 1.0, -0.6, -0.6, z, 1.0, 0.6, -0.6, z, 1.0 };
                }
            };
            const green_verts = tri.at(0.1);
            const red_verts = tri.at(0.9);
            const green_vb = try dev.createResource(.{ .buffer = .{ .size = 12 * 4, .usage = .{ .vertex = true } } });
            defer dev.destroyResource(green_vb);
            const red_vb = try dev.createResource(.{ .buffer = .{ .size = 12 * 4, .usage = .{ .vertex = true } } });
            defer dev.destroyResource(red_vb);
            @memcpy(std.mem.bytesAsSlice(f32, try dev.mapResource(green_vb))[0..12], &green_verts);
            @memcpy(std.mem.bytesAsSlice(f32, try dev.mapResource(red_vb))[0..12], &red_verts);
            const ctx = try dev.createContext();
            defer ctx.deinit();
            const rt = try dev.createResource(.{ .image = .{ .width = W, .height = H, .format = .bgra8_unorm, .usage = .{ .render_target = true } } });
            defer dev.destroyResource(rt);
            const depth = try dev.createResource(.{ .image = .{ .width = W, .height = H, .format = .depth32_float, .usage = .{} } });
            defer dev.destroyResource(depth);
            { // Submit 1: green (FragDepth 0.9), clear color + depth 1.0.
                const cb = try ctx.beginCommands();
                defer cb.deinit();
                try cb.setRenderTarget(rt);
                try cb.clear(.{ .r = 0.125, .g = 0.125, .b = 0.125, .a = 1 });
                try cb.setDepthTarget(depth, 1.0);
                try cb.bindPipeline(pipe_green);
                try cb.bindVertexBuffer(green_vb);
                try cb.draw(3, 0);
                ctx.submit(cb) catch return error.SkipZigTest;
            }
            { // Submit 2: red (FragDepth 0.1), preserve color + depth (null clear).
                const cb = try ctx.beginCommands();
                defer cb.deinit();
                try cb.setRenderTarget(rt);
                try cb.setDepthTarget(depth, null);
                try cb.bindPipeline(pipe_red);
                try cb.bindVertexBuffer(red_vb);
                try cb.draw(3, 0);
                ctx.submit(cb) catch return error.SkipZigTest;
            }
            const center = std.mem.bytesAsSlice(u32, try dev.mapResource(rt))[(H / 2) * W + W / 2];
            const r: u8 = @truncate(center >> 16);
            const g: u8 = @truncate(center >> 8);
            const b: u8 = @truncate(center);
            try std.testing.expect(r > 200 and g < 80 and b < 80); // red wins via gl_FragDepth
        }
    };
    try check.run(gpa, nv);
    try check.run(gpa, sw_dev);
}

test "ORACLE-MRT: a fragment shader writes two render targets on the NVIDIA GPU and in software (skips without a GPU)" {
    const gpa = std.testing.allocator;
    const nv = NvDevice.create(gpa) catch return error.SkipZigTest;
    defer nv.deinit();
    const sw = @import("../software.zig");
    const sw_dev = try sw.driver.createDevice(gpa);
    defer sw_dev.deinit();
    const W: u32 = 256;
    const H: u32 = 256;

    // An MRT fragment shader: RT0 = red, RT1 = green. nvidia routes RT0 -> R0-R3, RT1 -> R4-R7,
    // declares two SPH omap targets and binds two color surfaces. Software writes each output to
    // its own ColorTarget. The triangle centroid reads red on RT0, green on RT1.
    const check = struct {
        fn run(a: std.mem.Allocator, dev: hal.Device) !void {
            var vs_b = try buildVertexSpirv(a);
            defer vs_b.deinit(a);
            var ps_b = try buildMrtFragmentSpirv(a, .{ 1, 0, 0, 1 }, .{ 0, 1, 0, 1 });
            defer ps_b.deinit(a);
            const vs = try dev.createShaderModule(.{ .stage = .vertex, .code = std.mem.sliceAsBytes(vs_b.words.items) });
            defer dev.destroyShaderModule(vs);
            const ps = try dev.createShaderModule(.{ .stage = .fragment, .code = std.mem.sliceAsBytes(ps_b.words.items) });
            defer dev.destroyShaderModule(ps);
            const rt0 = try dev.createResource(.{ .image = .{ .width = W, .height = H, .format = .bgra8_unorm, .usage = .{ .render_target = true } } });
            defer dev.destroyResource(rt0);
            const rt1 = try dev.createResource(.{ .image = .{ .width = W, .height = H, .format = .bgra8_unorm, .usage = .{ .render_target = true } } });
            defer dev.destroyResource(rt1);
            const vbuf = try dev.createResource(.{ .buffer = .{ .size = 3 * 16, .usage = .{ .vertex = true } } });
            defer dev.destroyResource(vbuf);
            const verts = [_]f32{ 0.0, 0.5, 0.0, 1.0, -0.5, -0.5, 0.0, 1.0, 0.5, -0.5, 0.0, 1.0 };
            @memcpy(std.mem.bytesAsSlice(f32, try dev.mapResource(vbuf))[0..verts.len], &verts);
            const attrs = [_]hal.VertexAttribute{.{ .location = 0, .format = .rgba8_unorm, .offset = 0 }};
            const pipe = try dev.createPipeline(.{ .vertex = vs, .fragment = ps, .vertex_layout = .{ .stride = 16, .attributes = &attrs }, .color_format = .bgra8_unorm });
            defer dev.destroyPipeline(pipe);
            const ctx = try dev.createContext();
            defer ctx.deinit();
            {
                const cb = try ctx.beginCommands();
                defer cb.deinit();
                try cb.setRenderTarget(rt0);
                try cb.setColorTarget(1, rt1);
                try cb.clear(.{ .r = 0.125, .g = 0.125, .b = 0.125, .a = 1 });
                try cb.bindPipeline(pipe);
                try cb.bindVertexBuffer(vbuf);
                try cb.draw(3, 0);
                ctx.submit(cb) catch return error.SkipZigTest;
            }
            const c0 = std.mem.bytesAsSlice(u32, try dev.mapResource(rt0))[107 * W + 128];
            const c1 = std.mem.bytesAsSlice(u32, try dev.mapResource(rt1))[107 * W + 128];
            const get = struct {
                fn r(p: u32) u8 {
                    return @truncate(p >> 16);
                }
                fn g(p: u32) u8 {
                    return @truncate(p >> 8);
                }
                fn b(p: u32) u8 {
                    return @truncate(p);
                }
            };
            try std.testing.expect(get.r(c0) > 200 and get.g(c0) < 80 and get.b(c0) < 80); // RT0 red
            try std.testing.expect(get.g(c1) > 200 and get.r(c1) < 80 and get.b(c1) < 80); // RT1 green
        }
    };
    try check.run(gpa, nv);
    try check.run(gpa, sw_dev);
}

test "ORACLE-PERDRAW-UBO: draws in ONE submit each read their OWN bound UBO on the NVIDIA GPU and in software (skips without a GPU)" {
    const gpa = std.testing.allocator;
    const nv = NvDevice.create(gpa) catch return error.SkipZigTest;
    defer nv.deinit();
    const sw = @import("../software.zig");
    const sw_dev = try sw.driver.createDevice(gpa);
    defer sw_dev.deinit();
    const W: u32 = 128;
    const H: u32 = 128;

    // One pipeline, but two draws each with a different bound UBO (the FS outputs u.color): a
    // left triangle after bindUniformBuffer(green), a right triangle after bindUniformBuffer(red).
    // Per-draw uniform binding gives left=green, right=red. If the driver bound only the last
    // UBO for all draws (the pre-refactor bug), both would be red. This is the es2gears / UI
    // "many objects, same shader, different uniforms per draw, one submit" pattern.
    const check = struct {
        fn run(a: std.mem.Allocator, dev: hal.Device) !void {
            var vs_b = try buildVertexSpirv(a);
            defer vs_b.deinit(a);
            var ps_b = try buildUboColorFragmentSpirv(a);
            defer ps_b.deinit(a);
            const vs = try dev.createShaderModule(.{ .stage = .vertex, .code = std.mem.sliceAsBytes(vs_b.words.items) });
            defer dev.destroyShaderModule(vs);
            const ps = try dev.createShaderModule(.{ .stage = .fragment, .code = std.mem.sliceAsBytes(ps_b.words.items) });
            defer dev.destroyShaderModule(ps);
            const attrs = [_]hal.VertexAttribute{.{ .location = 0, .format = .rgba8_unorm, .offset = 0 }};
            const pipe = try dev.createPipeline(.{ .vertex = vs, .fragment = ps, .vertex_layout = .{ .stride = 16, .attributes = &attrs }, .color_format = .bgra8_unorm });
            defer dev.destroyPipeline(pipe);
            // Two UBOs, each a single vec4 color (std140 offset 0).
            const green_ubo = try dev.createResource(.{ .buffer = .{ .size = 16, .usage = .{ .uniform = true } } });
            defer dev.destroyResource(green_ubo);
            const red_ubo = try dev.createResource(.{ .buffer = .{ .size = 16, .usage = .{ .uniform = true } } });
            defer dev.destroyResource(red_ubo);
            @memcpy(std.mem.bytesAsSlice(f32, try dev.mapResource(green_ubo))[0..4], &[_]f32{ 0, 1, 0, 1 });
            @memcpy(std.mem.bytesAsSlice(f32, try dev.mapResource(red_ubo))[0..4], &[_]f32{ 1, 0, 0, 1 });
            // Left triangle (x in [-0.9,-0.1]) and right (x in [0.1,0.9]).
            const lv = [_]f32{ -0.5, 0.6, 0, 1, -0.9, -0.6, 0, 1, -0.1, -0.6, 0, 1 };
            const rv = [_]f32{ 0.5, 0.6, 0, 1, 0.1, -0.6, 0, 1, 0.9, -0.6, 0, 1 };
            const lvb = try dev.createResource(.{ .buffer = .{ .size = 48, .usage = .{ .vertex = true } } });
            defer dev.destroyResource(lvb);
            const rvb = try dev.createResource(.{ .buffer = .{ .size = 48, .usage = .{ .vertex = true } } });
            defer dev.destroyResource(rvb);
            @memcpy(std.mem.bytesAsSlice(f32, try dev.mapResource(lvb))[0..12], &lv);
            @memcpy(std.mem.bytesAsSlice(f32, try dev.mapResource(rvb))[0..12], &rv);
            const rt = try dev.createResource(.{ .image = .{ .width = W, .height = H, .format = .bgra8_unorm, .usage = .{ .render_target = true } } });
            defer dev.destroyResource(rt);
            const ctx = try dev.createContext();
            defer ctx.deinit();
            {
                const cb = try ctx.beginCommands();
                defer cb.deinit();
                try cb.setRenderTarget(rt);
                try cb.clear(.{ .r = 0.125, .g = 0.125, .b = 0.125, .a = 1 });
                try cb.bindPipeline(pipe);
                try cb.bindUniformBuffer(0, green_ubo);
                try cb.bindVertexBuffer(lvb);
                try cb.draw(3, 0);
                try cb.bindUniformBuffer(0, red_ubo);
                try cb.bindVertexBuffer(rvb);
                try cb.draw(3, 0);
                ctx.submit(cb) catch return error.SkipZigTest;
            }
            const fb = std.mem.bytesAsSlice(u32, try dev.mapResource(rt));
            const left = fb[70 * W + 40];
            const right = fb[70 * W + 88];
            const get = struct {
                fn r(p: u32) u8 {
                    return @truncate(p >> 16);
                }
                fn g(p: u32) u8 {
                    return @truncate(p >> 8);
                }
            };
            try std.testing.expect(get.g(left) > 200 and get.r(left) < 80); // left = green UBO
            try std.testing.expect(get.r(right) > 200 and get.g(right) < 80); // right = red UBO
        }
    };
    try check.run(gpa, nv);
    try check.run(gpa, sw_dev);
}

test "ORACLE-MULTIPIPELINE: two pipelines in ONE submit each render with their own state on the NVIDIA GPU (skips without a GPU)" {
    const gpa = std.testing.allocator;
    const dev = NvDevice.create(gpa) catch return error.SkipZigTest;
    defer dev.deinit();
    const W: u32 = 128;
    const H: u32 = 128;

    // A red pipeline draws a left triangle, a green pipeline a right triangle, both in one
    // command buffer / submit. Each draw renders with its own pipeline (left red, right
    // green). The context emits per-draw pipeline state instead of only the last one.
    var vs_b = try buildVertexSpirv(gpa);
    defer vs_b.deinit(gpa);
    var red_b = try buildFragmentSpirv(gpa, .{ 1, 0, 0, 1 });
    defer red_b.deinit(gpa);
    var grn_b = try buildFragmentSpirv(gpa, .{ 0, 1, 0, 1 });
    defer grn_b.deinit(gpa);
    const vs = try dev.createShaderModule(.{ .stage = .vertex, .code = std.mem.sliceAsBytes(vs_b.words.items) });
    defer dev.destroyShaderModule(vs);
    const ps_r = try dev.createShaderModule(.{ .stage = .fragment, .code = std.mem.sliceAsBytes(red_b.words.items) });
    defer dev.destroyShaderModule(ps_r);
    const ps_g = try dev.createShaderModule(.{ .stage = .fragment, .code = std.mem.sliceAsBytes(grn_b.words.items) });
    defer dev.destroyShaderModule(ps_g);
    const attrs = [_]hal.VertexAttribute{.{ .location = 0, .format = .rgba8_unorm, .offset = 0 }};
    const mk = struct {
        fn f(d: hal.Device, v: *hal.ShaderModule, ps: *hal.ShaderModule, a: []const hal.VertexAttribute) !*hal.Pipeline {
            return d.createPipeline(.{ .vertex = v, .fragment = ps, .vertex_layout = .{ .stride = 16, .attributes = a }, .color_format = .bgra8_unorm });
        }
    }.f;
    const pr = try mk(dev, vs, ps_r, &attrs);
    defer dev.destroyPipeline(pr);
    const pg = try mk(dev, vs, ps_g, &attrs);
    defer dev.destroyPipeline(pg);
    // Left triangle spans x in [-0.9,-0.1] (center pixel ~x=40); right x in [0.1,0.9] (~x=88).
    const lv = [_]f32{ -0.5, 0.6, 0, 1, -0.9, -0.6, 0, 1, -0.1, -0.6, 0, 1 };
    const rv = [_]f32{ 0.5, 0.6, 0, 1, 0.1, -0.6, 0, 1, 0.9, -0.6, 0, 1 };
    const lvb = try dev.createResource(.{ .buffer = .{ .size = 48, .usage = .{ .vertex = true } } });
    defer dev.destroyResource(lvb);
    const rvb = try dev.createResource(.{ .buffer = .{ .size = 48, .usage = .{ .vertex = true } } });
    defer dev.destroyResource(rvb);
    @memcpy(std.mem.bytesAsSlice(f32, try dev.mapResource(lvb))[0..12], &lv);
    @memcpy(std.mem.bytesAsSlice(f32, try dev.mapResource(rvb))[0..12], &rv);
    const rt = try dev.createResource(.{ .image = .{ .width = W, .height = H, .format = .bgra8_unorm, .usage = .{ .render_target = true } } });
    defer dev.destroyResource(rt);
    const ctx = try dev.createContext();
    defer ctx.deinit();
    {
        const cb = try ctx.beginCommands();
        defer cb.deinit();
        try cb.setRenderTarget(rt);
        try cb.clear(.{ .r = 0.125, .g = 0.125, .b = 0.125, .a = 1 });
        try cb.bindPipeline(pr);
        try cb.bindVertexBuffer(lvb);
        try cb.draw(3, 0);
        try cb.bindPipeline(pg);
        try cb.bindVertexBuffer(rvb);
        try cb.draw(3, 0);
        ctx.submit(cb) catch return error.SkipZigTest;
    }
    const fb = std.mem.bytesAsSlice(u32, try dev.mapResource(rt));
    const left = fb[70 * W + 40]; // inside the left (red-pipeline) triangle
    const right = fb[70 * W + 88]; // inside the right (green-pipeline) triangle
    const get = struct {
        fn r(p: u32) u8 {
            return @truncate(p >> 16);
        }
        fn g(p: u32) u8 {
            return @truncate(p >> 8);
        }
    };
    try std.testing.expect(get.r(left) > 200 and get.g(left) < 80); // left = red pipeline
    try std.testing.expect(get.g(right) > 200 and get.r(right) < 80); // right = green pipeline
}

test "ORACLE-BLEND: a translucent triangle alpha-composites the SAME on the NVIDIA GPU and in software (skips without a GPU)" {
    const gpa = std.testing.allocator;
    const nv = NvDevice.create(gpa) catch return error.SkipZigTest;
    defer nv.deinit();

    const sw = @import("../software.zig");
    const sw_dev = try sw.driver.createDevice(gpa);
    defer sw_dev.deinit();

    const W: u32 = 256;
    const H: u32 = 256;

    // A 50%-translucent red triangle over an opaque blue background, SRC_ALPHA /
    // ONE_MINUS_SRC_ALPHA, FUNC_ADD. Interior = 0.5*red + 0.5*blue = (0.5, 0, 0.5).
    const clear = hal.Color{ .r = 0, .g = 0, .b = 1, .a = 1 };
    const red_half = [4]f32{ 1.0, 0.0, 0.0, 0.5 };
    const blend = hal.BlendState{
        .enable = true,
        .src_color = .src_alpha,
        .dst_color = .one_minus_src_alpha,
        .src_alpha = .src_alpha,
        .dst_alpha = .one_minus_src_alpha,
    };

    const nv_fb = try runBlend(gpa, nv, W, H, clear, red_half, blend);
    defer gpa.free(nv_fb);
    const sw_fb = try runBlend(gpa, sw_dev, W, H, clear, red_half, blend);
    defer gpa.free(sw_fb);

    const get = struct {
        fn r(p: u32) i32 {
            return @intCast((p >> 16) & 0xff);
        }
        fn g(p: u32) i32 {
            return @intCast((p >> 8) & 0xff);
        }
        fn b(p: u32) i32 {
            return @intCast(p & 0xff);
        }
    };

    // The triangle centroid maps to ~(128,107). Software is the oracle: it must read the
    // exact blended (~128, 0, ~128). Then assert the GPU matches software within tolerance.
    const sw_c = sw_fb[107 * W + 128];
    try std.testing.expect(get.r(sw_c) > 110 and get.r(sw_c) < 145); // ~0.5 red
    try std.testing.expectEqual(@as(i32, 0), get.g(sw_c)); // no green
    try std.testing.expect(get.b(sw_c) > 110 and get.b(sw_c) < 145); // blue survived (0.5)

    // Compare nvidia vs software over the whole frame: per-channel within a small tolerance
    // (GPU fixed-function blend vs the software oracle round identically to within 2 LSB).
    var max_diff: i32 = 0;
    var sampled: usize = 0;
    for (0..@as(usize, W) * H) |i| {
        const n = nv_fb[i];
        const sft = sw_fb[i];
        inline for (.{ get.r, get.g, get.b }) |ch| {
            const d = ch(n) - ch(sft);
            const ad = if (d < 0) -d else d;
            if (ad > max_diff) max_diff = ad;
        }
        sampled += 1;
    }
    try std.testing.expect(sampled == @as(usize, W) * H);
    std.debug.print("[ORACLE-BLEND] translucent-over-blue centroid nv=0x{x:0>8} sw=0x{x:0>8} (A8R8G8B8) max_diff={d} LSB\n", .{ nv_fb[107 * W + 128], sw_c, max_diff });
    try std.testing.expect(max_diff <= 2);
}

test "SPIR-V gradient triangle renders the interpolated varying on the NVIDIA GPU (skips without a GPU)" {
    const gpa = std.testing.allocator;
    const dev = NvDevice.create(gpa) catch return error.SkipZigTest;
    defer dev.deinit();
    const W: u32 = 256;
    const H: u32 = 256;

    // Clear dark grey; draw the red/green/blue gradient triangle from SPIR-V. The
    // PS reads the interpolated color varying (not a constant). That is the whole point
    // of phase 2. Geometry: top=red, bottom-left=green, bottom-right=blue.
    const clear = hal.Color{ .r = 0.125, .g = 0.125, .b = 0.125, .a = 1 };
    const fb = try runGradient(gpa, dev, W, H, clear);
    defer gpa.free(fb);

    const clear_px: u32 = 0xff202020;
    const get = struct {
        fn r(p: u32) u8 {
            return @truncate(p >> 16);
        }
        fn g(p: u32) u8 {
            return @truncate(p >> 8);
        }
        fn b(p: u32) u8 {
            return @truncate(p);
        }
    };

    // The triangle covers a substantial region.
    var changed: u32 = 0;
    for (fb) |p| {
        if (p != clear_px) changed += 1;
    }
    try std.testing.expect(changed > 4000);

    // Clip -> pixel for a 256x256 RT: px = (x+1)*128, py = (y+1)*128 (NDC -1 -> row 0,
    // same Y-origin as software). The clip-top red vertex (clip y=+0.5) lands LOW
    // (~row 176); the clip-bottom green/blue verts (clip y=-0.5) land HIGH (~row 78);
    // the centroid at (~128,~107). Sample a few pixels pulled in toward the centroid
    // from each vertex (so coverage/edge AA at the exact vertex never makes the sample
    // fall on the clear color), and assert the dominant channel is the vertex's color.
    const sample = struct {
        fn at(buf: []const u32, px: u32, py: u32) u32 {
            return buf[@as(usize, py) * W + px];
        }
    }.at;

    // Near the clip-top vertex (red), now LOW: pull up toward the centroid.
    const top = sample(fb, 128, 176);
    try std.testing.expect(top != clear_px);
    try std.testing.expect(get.r(top) > get.g(top) and get.r(top) > get.b(top)); // red dominant

    // Near the clip-BL vertex (green), now HIGH-left: pull right + down toward centroid.
    const bl = sample(fb, 84, 78);
    try std.testing.expect(bl != clear_px);
    try std.testing.expect(get.g(bl) > get.r(bl) and get.g(bl) > get.b(bl)); // green dominant

    // Near the clip-BR vertex (blue), now HIGH-right: pull left + down toward centroid.
    const br = sample(fb, 172, 78);
    try std.testing.expect(br != clear_px);
    try std.testing.expect(get.b(br) > get.r(br) and get.b(br) > get.g(br)); // blue dominant

    // The centroid is a roughly equal mix of all three corner colors: every
    // channel present and no single channel dominating wildly.
    const mid = sample(fb, 128, 107);
    try std.testing.expect(mid != clear_px);
    try std.testing.expect(get.r(mid) > 40 and get.g(mid) > 40 and get.b(mid) > 40); // all three contribute

    // The strongest cross-check: render the hand-assembled gold gradient with the
    // same geometry/RT and compare the readback. Both compute the same perspective
    // interpolation, so the buffers should match within a small tolerance.
    const ref = renderGoldGradient(gpa, dev, W, H, clear) catch null;
    if (ref) |gold| {
        defer gpa.free(gold);
        var max_diff: u32 = 0;
        var diffs: u32 = 0;
        for (fb, gold) |a, c| {
            if (a == c) continue;
            inline for (.{ 0, 8, 16, 24 }) |sh| {
                const da = @as(i32, @as(u8, @truncate(a >> sh)));
                const db = @as(i32, @as(u8, @truncate(c >> sh)));
                const d: u32 = @intCast(@abs(da - db));
                if (d > max_diff) max_diff = d;
            }
            diffs += 1;
        }
        // A handful of edge/coverage pixels may differ. The bulk agrees and no
        // channel may differ by more than a small tolerance.
        try std.testing.expect(max_diff <= 8);
        try std.testing.expect(diffs < changed / 10);
    }
}

/// Render the hand-assembled gradient (triangle_hal's buildNvidia VS/PS) to a
/// fresh render target with the same geometry and read it back. This is the
/// hardware-verified gold reference: the SPIR-V gradient should match it. Returns the
/// caller-owned readback (free with `gpa`).
fn renderGoldGradient(gpa: std.mem.Allocator, dev: hal.Device, w: u32, h: u32, clear: hal.Color) ![]u32 {
    const sass = nvidia.sass;

    // The exact gold VS/PS from examples/triangle_hal.zig buildNvidia.
    var vs_code: [256]u32 = undefined;
    var vsa = sass.Assembler{ .code = &vs_code };
    {
        var p: u32 = 0;
        while (p < 6) : (p += 1) vsa.movImm(12, p, .{});
    }
    vsa.ald(4, sass.ATTR_GENERIC0, 4, .{ .wr_barrier = 0 });
    vsa.ald(8, sass.ATTR_GENERIC0 + 0x10, 4, .{ .wr_barrier = 1 });
    vsa.ast(sass.ATTR_POSITION, 4, 4, .{ .wait_mask = 1 });
    vsa.ast(sass.ATTR_GENERIC0, 8, 4, .{ .wait_mask = 2 });
    vsa.exit(.{ .stall = 1 });

    var ps_code: [256]u32 = undefined;
    var psa = sass.Assembler{ .code = &ps_code };
    {
        var p: u32 = 0;
        while (p < 6) : (p += 1) psa.movImm(15, p, .{});
    }
    psa.ipa(4, sass.ATTR_GENERIC0, .{ .wr_barrier = 0 });
    psa.ipa(5, sass.ATTR_GENERIC0 + 4, .{ .wr_barrier = 1 });
    psa.ipa(6, sass.ATTR_GENERIC0 + 8, .{ .wr_barrier = 2 });
    psa.ipa(7, sass.ATTR_GENERIC0 + 12, .{ .wr_barrier = 3 });
    psa.movReg(0, 4, .{ .wait_mask = 0xf });
    psa.movReg(1, 5, .{});
    psa.movReg(2, 6, .{});
    psa.movReg(3, 7, .{});
    psa.exit(.{ .stall = 15 });

    const rt = try dev.createResource(.{ .image = .{ .width = w, .height = h, .format = .bgra8_unorm, .usage = .{ .render_target = true } } });
    defer dev.destroyResource(rt);
    const vs = try dev.createShaderModule(.{ .stage = .vertex, .code = std.mem.sliceAsBytes(vs_code[0..vsa.dwords()]) });
    defer dev.destroyShaderModule(vs);
    const ps = try dev.createShaderModule(.{ .stage = .fragment, .code = std.mem.sliceAsBytes(ps_code[0..psa.dwords()]) });
    defer dev.destroyShaderModule(ps);

    const vbuf = try dev.createResource(.{ .buffer = .{ .size = 3 * 32, .usage = .{ .vertex = true } } });
    defer dev.destroyResource(vbuf);
    const verts = [_]f32{
        0.0,  0.5,  0.0, 1.0, 1.0, 0.0, 0.0, 1.0,
        -0.5, -0.5, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0,
        0.5,  -0.5, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0,
    };
    @memcpy(std.mem.bytesAsSlice(f32, try dev.mapResource(vbuf))[0..verts.len], &verts);

    const attrs = [_]hal.VertexAttribute{
        .{ .location = 0, .format = .rgba8_unorm, .offset = 0 },
        .{ .location = 1, .format = .rgba8_unorm, .offset = 16 },
    };
    const pipe = try dev.createPipeline(.{
        .vertex = vs,
        .fragment = ps,
        .vertex_layout = .{ .stride = 32, .attributes = &attrs },
        .color_format = .bgra8_unorm,
    });
    defer dev.destroyPipeline(pipe);

    const ctx = try dev.createContext();
    defer ctx.deinit();
    {
        const cb = try ctx.beginCommands();
        defer cb.deinit();
        try cb.setRenderTarget(rt);
        try cb.clear(clear);
        try cb.bindPipeline(pipe);
        try cb.bindVertexBuffer(vbuf);
        try cb.draw(3, 0);
        try ctx.submit(cb);
    }

    const fb = std.mem.bytesAsSlice(u32, try dev.mapResource(rt));
    const out = try gpa.alloc(u32, @as(usize, w) * h);
    errdefer gpa.free(out);
    // Tight (w*4) readback: the software RT is tight and nvidia mapResource de-swizzles
    // the block-linear RT into a tight w*4 buffer. (pitchBytes(w)/4 == w only at 256.)
    const pitch_px = w;
    var y: u32 = 0;
    while (y < h) : (y += 1) {
        @memcpy(out[@as(usize, y) * w ..][0..w], fb[@as(usize, y) * pitch_px ..][0..w]);
    }
    return out;
}

// Mid-triangle linear-ramp interpolation oracle: the gradient check in ORACLE only
// samples near-vertex pixels where one barycentric weight dominates, so a 2x-too-steep
// plane is invisible there. This oracle sweeps many interior pixels and checks the
// interpolated varying against the analytic barycentric plane (all clip-w=1, exact
// affine). Expected color = w0*c0 + w1*c1 + w2*c2. Permanent oracle for rasterizer-W
// and interpolation fixes.
test "ORACLE: gradient varying interpolates LINEARLY across the triangle interior on the NVIDIA GPU (skips without a GPU)" {
    const gpa = std.testing.allocator;
    const dev = NvDevice.create(gpa) catch return error.SkipZigTest;
    defer dev.deinit();
    const W: u32 = 256;
    const H: u32 = 256;

    const clear = hal.Color{ .r = 0.125, .g = 0.125, .b = 0.125, .a = 1 };
    const fb = try runGradient(gpa, dev, W, H, clear);
    defer gpa.free(fb);

    const clear_px: u32 = 0xff202020;

    // Screen-space vertices for the runGradient geometry on a WxH RT.
    // clip -> pixel: px = (x+1)*W/2, py = (y+1)*H/2 (NDC -1 -> row 0, same as software).
    // The clip-space "top" vertex (clip y=+0.5) lands LOW (py=192). The clip-space
    // "bottom" verts (clip y=-0.5) land HIGH (py=64), the Vulkan-standard Y-origin.
    const Vtx = struct { x: f32, y: f32, c: [3]f32 };
    const v0 = Vtx{ .x = 128, .y = 192, .c = .{ 1, 0, 0 } }; // clip-top red  -> low
    const v1 = Vtx{ .x = 64, .y = 64, .c = .{ 0, 1, 0 } }; // clip-BL  green -> high-left
    const v2 = Vtx{ .x = 192, .y = 64, .c = .{ 0, 0, 1 } }; // clip-BR  blue  -> high-right

    // Twice the signed area of the screen triangle (constant denominator).
    const denom = (v1.y - v2.y) * (v0.x - v2.x) + (v2.x - v1.x) * (v0.y - v2.y);
    try std.testing.expect(@abs(denom) > 1.0);

    const get = struct {
        fn r(p: u32) f32 {
            return @as(f32, @floatFromInt(@as(u8, @truncate(p >> 16)))) / 255.0;
        }
        fn g(p: u32) f32 {
            return @as(f32, @floatFromInt(@as(u8, @truncate(p >> 8)))) / 255.0;
        }
        fn b(p: u32) f32 {
            return @as(f32, @floatFromInt(@as(u8, @truncate(p)))) / 255.0;
        }
    };

    // Sweep a dense grid of interior points. For each one well inside the
    // triangle (every barycentric weight >= a margin, so we never sample a
    // coverage edge), compare the readback color to the analytic linear plane.
    var checked: u32 = 0;
    var bad: u32 = 0;
    var worst: f32 = 0;
    var worst_px: u32 = 0;
    var worst_py: u32 = 0;
    var py: u32 = 66;
    while (py < 190) : (py += 4) {
        var px: u32 = 66;
        while (px < 190) : (px += 4) {
            const fx: f32 = @floatFromInt(px);
            const fy: f32 = @floatFromInt(py);
            const w0 = ((v1.y - v2.y) * (fx - v2.x) + (v2.x - v1.x) * (fy - v2.y)) / denom;
            const w1 = ((v2.y - v0.y) * (fx - v2.x) + (v0.x - v2.x) * (fy - v2.y)) / denom;
            const w2 = 1.0 - w0 - w1;
            const margin: f32 = 0.06;
            if (w0 < margin or w1 < margin or w2 < margin) continue;

            const p = fb[py * W + px];
            if (p == clear_px) {
                bad += 1; // a genuinely interior point should be covered
                continue;
            }
            const exp_r = w0 * v0.c[0] + w1 * v1.c[0] + w2 * v2.c[0];
            const exp_g = w0 * v0.c[1] + w1 * v1.c[1] + w2 * v2.c[1];
            const exp_b = w0 * v0.c[2] + w1 * v1.c[2] + w2 * v2.c[2];
            const d = @max(@abs(get.r(p) - exp_r), @max(@abs(get.g(p) - exp_g), @abs(get.b(p) - exp_b)));
            checked += 1;
            if (d > worst) {
                worst = d;
                worst_px = px;
                worst_py = py;
            }
            // 8-bit quantization + small rasterizer rounding. A 2x-too-steep
            // plane misses by ~0.3-0.5 here, far above this bar.
            if (d > 0.08) bad += 1;
        }
    }

    if (bad > 0) std.debug.print(
        "\n[ORACLE] interior checked={d} bad={d} worst_dev={d:.4} at ({d},{d})\n",
        .{ checked, bad, worst, worst_px, worst_py },
    );

    try std.testing.expect(checked > 60); // the sweep actually hit the interior
    try std.testing.expectEqual(@as(u32, 0), bad);
}

/// Build a VS that passes the input position through to a generic varying (loc 0)
/// in addition to gl_Position, so the interpolated varying aliases the clip
/// coordinates (Session 6's exact trigger: frag_pos == gl_Position). One input
/// attribute (loc 0), one varying output (loc 0). No separate color attribute.
fn buildPosPassthroughVertexSpirv(gpa: std.mem.Allocator) !Builder {
    // ids: void=1 f32=2 v4f=3 pIn=4 pOut=5 voidfn=6 posIn=7 posOut=9 varOut=10
    //      main=11 entry=12 ldPos=13.
    var b = try Builder.init(gpa, 14);
    errdefer b.deinit(gpa);
    try b.emit(gpa, op.EntryPoint, &.{ op.ExecutionModel.vertex, 11, 0, 7, 9, 10 });
    try b.emit(gpa, op.Decorate, &.{ 7, op.Decoration.location, 0 }); // position input
    try b.emit(gpa, op.Decorate, &.{ 9, op.Decoration.builtin, op.BuiltIn.position });
    try b.emit(gpa, op.Decorate, &.{ 10, op.Decoration.location, 0 }); // varying output (= position)
    try b.emit(gpa, op.TypeVoid, &.{1});
    try b.emit(gpa, op.TypeFloat, &.{ 2, 32 });
    try b.emit(gpa, op.TypeVector, &.{ 3, 2, 4 });
    try b.emit(gpa, op.TypePointer, &.{ 4, op.StorageClass.input, 3 });
    try b.emit(gpa, op.TypePointer, &.{ 5, op.StorageClass.output, 3 });
    try b.emit(gpa, op.TypeFunction, &.{ 6, 1 });
    try b.emit(gpa, op.Variable, &.{ 4, 7, op.StorageClass.input });
    try b.emit(gpa, op.Variable, &.{ 5, 9, op.StorageClass.output });
    try b.emit(gpa, op.Variable, &.{ 5, 10, op.StorageClass.output });
    try b.emit(gpa, op.Function, &.{ 1, 11, 0, 6 });
    try b.emit(gpa, op.Label, &.{12});
    try b.emit(gpa, op.Load, &.{ 3, 13, 7 }); // pos = load position
    try b.emit(gpa, op.Store, &.{ 9, 13 }); // gl_Position = pos
    try b.emit(gpa, op.Store, &.{ 10, 13 }); // varying = pos (aliases gl_Position)
    try b.emit(gpa, op.Return, &.{});
    try b.emit(gpa, op.FunctionEnd, &.{});
    return b;
}

/// Render the position-passthrough triangle (varying = gl_Position) to a fresh RT.
/// The clip positions are chosen in [0,1] so the interpolated varying is directly
/// readable as a color. Returns the caller-owned tightly-packed readback.
fn runPosPassthrough(gpa: std.mem.Allocator, dev: hal.Device, w: u32, h: u32, clear: hal.Color) ![]u32 {
    var vs_b = try buildPosPassthroughVertexSpirv(gpa);
    defer vs_b.deinit(gpa);
    var ps_b = try buildGradientFragmentSpirv(gpa); // out = varying (pure passthrough)
    defer ps_b.deinit(gpa);
    const vs_sass = try compileToSass(gpa, std.mem.sliceAsBytes(vs_b.words.items), .vertex);
    defer gpa.free(vs_sass);
    const ps_sass = try compileToSass(gpa, std.mem.sliceAsBytes(ps_b.words.items), .fragment);
    defer gpa.free(ps_sass);

    const rt = try dev.createResource(.{ .image = .{ .width = w, .height = h, .format = .bgra8_unorm, .usage = .{ .render_target = true } } });
    defer dev.destroyResource(rt);
    const vs = try dev.createShaderModule(.{ .stage = .vertex, .code = std.mem.sliceAsBytes(vs_sass) });
    defer dev.destroyShaderModule(vs);
    const ps = try dev.createShaderModule(.{ .stage = .fragment, .code = std.mem.sliceAsBytes(ps_sass) });
    defer dev.destroyShaderModule(ps);

    // 3 verts: vec4 position only (stride 16). Positions in clip [0,1] so the
    // passed-through varying reads back directly as a color (r=x, g=y).
    const vbuf = try dev.createResource(.{ .buffer = .{ .size = 3 * 16, .usage = .{ .vertex = true } } });
    defer dev.destroyResource(vbuf);
    const verts = [_]f32{
        0.5, 0.9, 0.5, 1.0, // top
        0.1, 0.1, 0.5, 1.0, // bottom-left
        0.9, 0.1, 0.5, 1.0, // bottom-right
    };
    @memcpy(std.mem.bytesAsSlice(f32, try dev.mapResource(vbuf))[0..verts.len], &verts);

    const attrs = [_]hal.VertexAttribute{
        .{ .location = 0, .format = .rgba8_unorm, .offset = 0 },
    };
    const pipe = try dev.createPipeline(.{
        .vertex = vs,
        .fragment = ps,
        .vertex_layout = .{ .stride = 16, .attributes = &attrs },
        .color_format = .bgra8_unorm,
    });
    defer dev.destroyPipeline(pipe);

    const ctx = try dev.createContext();
    defer ctx.deinit();
    {
        const cb = try ctx.beginCommands();
        defer cb.deinit();
        try cb.setRenderTarget(rt);
        try cb.clear(clear);
        try cb.bindPipeline(pipe);
        try cb.bindVertexBuffer(vbuf);
        try cb.draw(3, 0);
        try ctx.submit(cb);
    }

    const fb = std.mem.bytesAsSlice(u32, try dev.mapResource(rt));
    const out = try gpa.alloc(u32, @as(usize, w) * h);
    errdefer gpa.free(out);
    // Tight (w*4) readback: the software RT is tight and nvidia mapResource de-swizzles
    // the block-linear RT into a tight w*4 buffer. (pitchBytes(w)/4 == w only at 256.)
    const pitch_px = w;
    var y: u32 = 0;
    while (y < h) : (y += 1) {
        @memcpy(out[@as(usize, y) * w ..][0..w], fb[@as(usize, y) * pitch_px ..][0..w]);
    }
    return out;
}

test "ORACLE2: position-passthrough varying (aliases gl_Position) interpolates LINEARLY on the NVIDIA GPU (skips without a GPU)" {
    const gpa = std.testing.allocator;
    const dev = NvDevice.create(gpa) catch return error.SkipZigTest;
    defer dev.deinit();
    const W: u32 = 256;
    const H: u32 = 256;

    const clear = hal.Color{ .r = 0.125, .g = 0.125, .b = 0.125, .a = 1 };
    const fb = try runPosPassthrough(gpa, dev, W, H, clear);
    defer gpa.free(fb);

    const clear_px: u32 = 0xff202020;

    // Clip [0,1] -> pixel: px = (x+1)*W/2, py = (y+1)*H/2 (NDC -1 -> row 0, software-
    // matching). top (0.5,0.9)->(192, 243.2); BL (0.1,0.1)->(140.8, 140.8);
    // BR (0.9,0.1)->(243.2, 140.8). The interpolated varying value at a pixel is
    // the clip position (cx, cy); we expect readback r = cx, g = cy.
    const Vtx = struct { x: f32, y: f32, cx: f32, cy: f32 };
    const v0 = Vtx{ .x = (0.5 + 1) * 128, .y = (0.9 + 1) * 128, .cx = 0.5, .cy = 0.9 };
    const v1 = Vtx{ .x = (0.1 + 1) * 128, .y = (0.1 + 1) * 128, .cx = 0.1, .cy = 0.1 };
    const v2 = Vtx{ .x = (0.9 + 1) * 128, .y = (0.1 + 1) * 128, .cx = 0.9, .cy = 0.1 };

    const denom = (v1.y - v2.y) * (v0.x - v2.x) + (v2.x - v1.x) * (v0.y - v2.y);
    try std.testing.expect(@abs(denom) > 1.0);

    const get = struct {
        fn r(p: u32) f32 {
            return @as(f32, @floatFromInt(@as(u8, @truncate(p >> 16)))) / 255.0;
        }
        fn g(p: u32) f32 {
            return @as(f32, @floatFromInt(@as(u8, @truncate(p >> 8)))) / 255.0;
        }
    };

    var checked: u32 = 0;
    var bad: u32 = 0;
    var worst: f32 = 0;
    var worst_px: u32 = 0;
    var worst_py: u32 = 0;
    var py: u32 = 146;
    while (py < 238) : (py += 3) {
        var px: u32 = 145;
        while (px < 240) : (px += 3) {
            const fx: f32 = @floatFromInt(px);
            const fy: f32 = @floatFromInt(py);
            const w0 = ((v1.y - v2.y) * (fx - v2.x) + (v2.x - v1.x) * (fy - v2.y)) / denom;
            const w1 = ((v2.y - v0.y) * (fx - v2.x) + (v0.x - v2.x) * (fy - v2.y)) / denom;
            const w2 = 1.0 - w0 - w1;
            const margin: f32 = 0.06;
            if (w0 < margin or w1 < margin or w2 < margin) continue;

            const p = fb[py * W + px];
            if (p == clear_px) {
                bad += 1;
                continue;
            }
            const exp_cx = w0 * v0.cx + w1 * v1.cx + w2 * v2.cx;
            const exp_cy = w0 * v0.cy + w1 * v1.cy + w2 * v2.cy;
            const d = @max(@abs(get.r(p) - exp_cx), @abs(get.g(p) - exp_cy));
            checked += 1;
            if (d > worst) {
                worst = d;
                worst_px = px;
                worst_py = py;
            }
            if (d > 0.08) bad += 1;
        }
    }

    if (bad > 0) std.debug.print(
        "\n[ORACLE2] interior checked={d} bad={d} worst_dev={d:.4} at ({d},{d})\n",
        .{ checked, bad, worst, worst_px, worst_py },
    );

    try std.testing.expect(checked > 50);
    try std.testing.expectEqual(@as(u32, 0), bad);
}

/// Build a VS with two generic varying outputs: location 0 = per-vertex color
/// (slot 0x80), location 1 = a second per-vertex attribute (slot 0x90). Session
/// 4/5 claimed a second varying (index>=1, slot 0x90) interpolates to a constant
/// 0. This exercises that exact path: two fetched attributes -> two varyings.
fn buildTwoVaryingVertexSpirv(gpa: std.mem.Allocator) !Builder {
    // ids: void=1 f32=2 v4f=3 pIn=4 pOut=5 voidfn=6 posIn=7 colIn=8 col2In=9
    //      posOut=10 colOut=11 col2Out=12 main=13 entry=14 ldP=15 ldC=16 ldC2=17.
    var b = try Builder.init(gpa, 18);
    errdefer b.deinit(gpa);
    try b.emit(gpa, op.EntryPoint, &.{ op.ExecutionModel.vertex, 13, 0, 7, 8, 9, 10, 11, 12 });
    try b.emit(gpa, op.Decorate, &.{ 7, op.Decoration.location, 0 }); // position in
    try b.emit(gpa, op.Decorate, &.{ 8, op.Decoration.location, 1 }); // color in
    try b.emit(gpa, op.Decorate, &.{ 9, op.Decoration.location, 2 }); // color2 in
    try b.emit(gpa, op.Decorate, &.{ 10, op.Decoration.builtin, op.BuiltIn.position });
    try b.emit(gpa, op.Decorate, &.{ 11, op.Decoration.location, 0 }); // varying 0 (slot 0x80)
    try b.emit(gpa, op.Decorate, &.{ 12, op.Decoration.location, 1 }); // varying 1 (slot 0x90)
    try b.emit(gpa, op.TypeVoid, &.{1});
    try b.emit(gpa, op.TypeFloat, &.{ 2, 32 });
    try b.emit(gpa, op.TypeVector, &.{ 3, 2, 4 });
    try b.emit(gpa, op.TypePointer, &.{ 4, op.StorageClass.input, 3 });
    try b.emit(gpa, op.TypePointer, &.{ 5, op.StorageClass.output, 3 });
    try b.emit(gpa, op.TypeFunction, &.{ 6, 1 });
    try b.emit(gpa, op.Variable, &.{ 4, 7, op.StorageClass.input });
    try b.emit(gpa, op.Variable, &.{ 4, 8, op.StorageClass.input });
    try b.emit(gpa, op.Variable, &.{ 4, 9, op.StorageClass.input });
    try b.emit(gpa, op.Variable, &.{ 5, 10, op.StorageClass.output });
    try b.emit(gpa, op.Variable, &.{ 5, 11, op.StorageClass.output });
    try b.emit(gpa, op.Variable, &.{ 5, 12, op.StorageClass.output });
    try b.emit(gpa, op.Function, &.{ 1, 13, 0, 6 });
    try b.emit(gpa, op.Label, &.{14});
    try b.emit(gpa, op.Load, &.{ 3, 15, 7 });
    try b.emit(gpa, op.Store, &.{ 10, 15 }); // gl_Position = pos
    try b.emit(gpa, op.Load, &.{ 3, 16, 8 });
    try b.emit(gpa, op.Store, &.{ 11, 16 }); // varying0 = color
    try b.emit(gpa, op.Load, &.{ 3, 17, 9 });
    try b.emit(gpa, op.Store, &.{ 12, 17 }); // varying1 = color2
    try b.emit(gpa, op.Return, &.{});
    try b.emit(gpa, op.FunctionEnd, &.{});
    return b;
}

/// FS that reads only varying location 1 (slot 0x90) and writes it as the color.
fn buildSecondVaryingFragmentSpirv(gpa: std.mem.Allocator) !Builder {
    // ids: void=1 f32=2 v4f=3 pIn=4 pOut=5 voidfn=6 var1In=7 colOut=8 main=9
    //      entry=10 loaded=11.
    var b = try Builder.init(gpa, 12);
    errdefer b.deinit(gpa);
    try b.emit(gpa, op.EntryPoint, &.{ op.ExecutionModel.fragment, 9, 0, 7, 8 });
    try b.emit(gpa, op.Decorate, &.{ 7, op.Decoration.location, 1 }); // read varying loc 1
    try b.emit(gpa, op.Decorate, &.{ 8, op.Decoration.location, 0 });
    try b.emit(gpa, op.TypeVoid, &.{1});
    try b.emit(gpa, op.TypeFloat, &.{ 2, 32 });
    try b.emit(gpa, op.TypeVector, &.{ 3, 2, 4 });
    try b.emit(gpa, op.TypePointer, &.{ 4, op.StorageClass.input, 3 });
    try b.emit(gpa, op.TypePointer, &.{ 5, op.StorageClass.output, 3 });
    try b.emit(gpa, op.TypeFunction, &.{ 6, 1 });
    try b.emit(gpa, op.Variable, &.{ 4, 7, op.StorageClass.input });
    try b.emit(gpa, op.Variable, &.{ 5, 8, op.StorageClass.output });
    try b.emit(gpa, op.Function, &.{ 1, 9, 0, 6 });
    try b.emit(gpa, op.Label, &.{10});
    try b.emit(gpa, op.Load, &.{ 3, 11, 7 });
    try b.emit(gpa, op.Store, &.{ 8, 11 });
    try b.emit(gpa, op.Return, &.{});
    try b.emit(gpa, op.FunctionEnd, &.{});
    return b;
}

fn runSecondVarying(gpa: std.mem.Allocator, dev: hal.Device, w: u32, h: u32, clear: hal.Color) ![]u32 {
    var vs_b = try buildTwoVaryingVertexSpirv(gpa);
    defer vs_b.deinit(gpa);
    var ps_b = try buildSecondVaryingFragmentSpirv(gpa);
    defer ps_b.deinit(gpa);
    const vs_sass = try compileToSass(gpa, std.mem.sliceAsBytes(vs_b.words.items), .vertex);
    defer gpa.free(vs_sass);
    const ps_sass = try compileToSass(gpa, std.mem.sliceAsBytes(ps_b.words.items), .fragment);
    defer gpa.free(ps_sass);

    const rt = try dev.createResource(.{ .image = .{ .width = w, .height = h, .format = .bgra8_unorm, .usage = .{ .render_target = true } } });
    defer dev.destroyResource(rt);
    const vs = try dev.createShaderModule(.{ .stage = .vertex, .code = std.mem.sliceAsBytes(vs_sass) });
    defer dev.destroyShaderModule(vs);
    const ps = try dev.createShaderModule(.{ .stage = .fragment, .code = std.mem.sliceAsBytes(ps_sass) });
    defer dev.destroyShaderModule(ps);

    // 3 verts: vec4 position, vec4 color0, vec4 color1 (stride 48). color1 is a
    // distinct ramp (cyan/magenta/yellow) so a constant-0 second varying is obvious.
    const vbuf = try dev.createResource(.{ .buffer = .{ .size = 3 * 48, .usage = .{ .vertex = true } } });
    defer dev.destroyResource(vbuf);
    const verts = [_]f32{
        0.0, 0.5, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, // top:    col0 red,   col1 cyan
        -0.5, -0.5, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, // BL:     col0 green, col1 magenta
        0.5, -0.5, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, // BR:     col0 blue,  col1 yellow
    };
    @memcpy(std.mem.bytesAsSlice(f32, try dev.mapResource(vbuf))[0..verts.len], &verts);

    const attrs = [_]hal.VertexAttribute{
        .{ .location = 0, .format = .rgba8_unorm, .offset = 0 },
        .{ .location = 1, .format = .rgba8_unorm, .offset = 16 },
        .{ .location = 2, .format = .rgba8_unorm, .offset = 32 },
    };
    const pipe = try dev.createPipeline(.{
        .vertex = vs,
        .fragment = ps,
        .vertex_layout = .{ .stride = 48, .attributes = &attrs },
        .color_format = .bgra8_unorm,
    });
    defer dev.destroyPipeline(pipe);

    const ctx = try dev.createContext();
    defer ctx.deinit();
    {
        const cb = try ctx.beginCommands();
        defer cb.deinit();
        try cb.setRenderTarget(rt);
        try cb.clear(clear);
        try cb.bindPipeline(pipe);
        try cb.bindVertexBuffer(vbuf);
        try cb.draw(3, 0);
        try ctx.submit(cb);
    }

    const fb = std.mem.bytesAsSlice(u32, try dev.mapResource(rt));
    const out = try gpa.alloc(u32, @as(usize, w) * h);
    errdefer gpa.free(out);
    // Tight (w*4) readback: the software RT is tight and nvidia mapResource de-swizzles
    // the block-linear RT into a tight w*4 buffer. (pitchBytes(w)/4 == w only at 256.)
    const pitch_px = w;
    var y: u32 = 0;
    while (y < h) : (y += 1) {
        @memcpy(out[@as(usize, y) * w ..][0..w], fb[@as(usize, y) * pitch_px ..][0..w]);
    }
    return out;
}

test "ORACLE3: SECOND varying (slot 0x90) interpolates LINEARLY (not constant 0) on the NVIDIA GPU (skips without a GPU)" {
    const gpa = std.testing.allocator;
    const dev = NvDevice.create(gpa) catch return error.SkipZigTest;
    defer dev.deinit();
    const W: u32 = 256;
    const H: u32 = 256;

    const clear = hal.Color{ .r = 0.125, .g = 0.125, .b = 0.125, .a = 1 };
    const fb = try runSecondVarying(gpa, dev, W, H, clear);
    defer gpa.free(fb);

    const clear_px: u32 = 0xff202020;

    // Same geometry as the gradient (NDC -1 -> row 0 Y-origin, software-matching: the
    // clip-top vertex lands LOW at y=192, the clip-bottom verts HIGH at y=64). col1:
    // clip-top=cyan(0,1,1), clip-BL=magenta(1,0,1), clip-BR=yellow(1,1,0). The
    // interpolated second varying must match this plane.
    const Vtx = struct { x: f32, y: f32, c: [3]f32 };
    const v0 = Vtx{ .x = 128, .y = 192, .c = .{ 0, 1, 1 } };
    const v1 = Vtx{ .x = 64, .y = 64, .c = .{ 1, 0, 1 } };
    const v2 = Vtx{ .x = 192, .y = 64, .c = .{ 1, 1, 0 } };
    const denom = (v1.y - v2.y) * (v0.x - v2.x) + (v2.x - v1.x) * (v0.y - v2.y);

    const get = struct {
        fn r(p: u32) f32 {
            return @as(f32, @floatFromInt(@as(u8, @truncate(p >> 16)))) / 255.0;
        }
        fn g(p: u32) f32 {
            return @as(f32, @floatFromInt(@as(u8, @truncate(p >> 8)))) / 255.0;
        }
        fn b(p: u32) f32 {
            return @as(f32, @floatFromInt(@as(u8, @truncate(p)))) / 255.0;
        }
    };

    var checked: u32 = 0;
    var bad: u32 = 0;
    var worst: f32 = 0;
    var py: u32 = 66;
    while (py < 190) : (py += 4) {
        var px: u32 = 66;
        while (px < 190) : (px += 4) {
            const fx: f32 = @floatFromInt(px);
            const fy: f32 = @floatFromInt(py);
            const w0 = ((v1.y - v2.y) * (fx - v2.x) + (v2.x - v1.x) * (fy - v2.y)) / denom;
            const w1 = ((v2.y - v0.y) * (fx - v2.x) + (v0.x - v2.x) * (fy - v2.y)) / denom;
            const w2 = 1.0 - w0 - w1;
            if (w0 < 0.06 or w1 < 0.06 or w2 < 0.06) continue;
            const p = fb[py * W + px];
            if (p == clear_px) {
                bad += 1;
                continue;
            }
            const er = w0 * v0.c[0] + w1 * v1.c[0] + w2 * v2.c[0];
            const eg = w0 * v0.c[1] + w1 * v1.c[1] + w2 * v2.c[1];
            const eb = w0 * v0.c[2] + w1 * v1.c[2] + w2 * v2.c[2];
            const d = @max(@abs(get.r(p) - er), @max(@abs(get.g(p) - eg), @abs(get.b(p) - eb)));
            checked += 1;
            if (d > worst) worst = d;
            if (d > 0.08) bad += 1;
        }
    }
    if (bad > 0) std.debug.print("\n[ORACLE3] secondvarying checked={d} bad={d} worst_dev={d:.4}\n", .{ checked, bad, worst });
    try std.testing.expect(checked > 60);
    try std.testing.expectEqual(@as(u32, 0), bad);
}

// vkcube's real fragment shader uses GLSL.std.450 Pow (the linearToSrgb transfer curve).
// Before the math_fn lowering, the NVIDIA backend returned error.Unsupported on the
// host-math call_indirect (vkCreateGraphicsPipelines failed -> the cube showed the clear).
// Now pow/exp/log/sin/cos lower to the MUFU special-function unit. This guards that the
// real vkcube FS compiles to SASS and emits MUFU (a compile-only test, runs in the sandbox).
test "vkcube's fragment shader compiles to SASS with MUFU (the host-math / pow lowering)" {
    const gpa = std.testing.allocator;
    const sass = try compileToSass(gpa, @embedFile("vkcube_fs.spv"), .fragment);
    defer gpa.free(sass);
    var has_mufu = false;
    var i: usize = 0;
    while (i + 4 <= sass.len) : (i += 4) {
        // MUFU encodes as low-12-bits 0x308 (the base 0x108 | the register-srcB form bit).
        if (sass[i] & 0xfff == 0x308) has_mufu = true;
    }
    try std.testing.expect(has_mufu);
}

// vkcube's exact vertex shader (gl_VertexIndex pulls a vec4 position from a UBO array,
// multiplied by a UBO mat4 MVP, with a texcoord + frag_pos varying) draws an on-screen
// triangle through the full NVIDIA GPU path. This exercises vertex-pulling + the UBO
// matrix + multi-varying + the descriptor constant-bank binding slot (a UBO at binding 0)
// together. This is the integration the per-feature tests covered only in isolation. Uses a
// trivial constant FS so the test asserts the VS geometry/coverage, not the FS shading.
test "vkcube's exact VS draws an on-screen triangle on the NVIDIA GPU (skips without a GPU)" {
    const gpa = std.testing.allocator;
    const dev = NvDevice.create(gpa) catch return error.SkipZigTest;
    defer dev.deinit();
    const W: u32 = 256;
    const H: u32 = 256;

    const vs_sass = try compileToSass(gpa, @embedFile("vkcube_vs.spv"), .vertex);
    defer gpa.free(vs_sass);
    var fs_b = try buildFragmentSpirv(gpa, .{ 1.0, 0.0, 0.0, 1.0 });
    defer fs_b.deinit(gpa);
    const fs_sass = try compileToSass(gpa, std.mem.sliceAsBytes(fs_b.words.items), .fragment);
    defer gpa.free(fs_sass);

    const rt = try dev.createResource(.{ .image = .{ .width = W, .height = H, .format = .rgba8_unorm, .usage = .{ .render_target = true } } });
    defer dev.destroyResource(rt);
    const vs = try dev.createShaderModule(.{ .stage = .vertex, .code = std.mem.sliceAsBytes(vs_sass) });
    defer dev.destroyShaderModule(vs);
    const fs = try dev.createShaderModule(.{ .stage = .fragment, .code = std.mem.sliceAsBytes(fs_sass) });
    defer dev.destroyShaderModule(fs);

    // UBO: { mat4 mvp; vec4 position[36]; vec4 attr[36] } = 1216 bytes. Identity MVP +
    // a big on-screen triangle in position[0..2] (the cube's actual data is many tris;
    // we feed a known on-screen one so the coverage is deterministic).
    const ubo = try dev.createResource(.{ .buffer = .{ .size = 1216, .usage = .{ .uniform = true } } });
    defer dev.destroyResource(ubo);
    {
        const f = std.mem.bytesAsSlice(f32, try dev.mapResource(ubo));
        @memset(f[0..304], 0);
        f[0] = 1;
        f[5] = 1;
        f[10] = 1;
        f[15] = 1;
        const pos = [_]f32{ 0.0, -0.8, 0.0, 1.0, -0.8, 0.8, 0.0, 1.0, 0.8, 0.8, 0.0, 1.0 };
        @memcpy(f[16 .. 16 + 12], &pos);
    }

    const pipe = try dev.createPipeline(.{
        .vertex = vs,
        .fragment = fs,
        .vertex_layout = .{ .stride = 0, .attributes = &.{} }, // vertex-pulling
        .color_format = .rgba8_unorm,
    });
    defer dev.destroyPipeline(pipe);

    const ctx = try dev.createContext();
    defer ctx.deinit();
    {
        const cb = try ctx.beginCommands();
        defer cb.deinit();
        try cb.setRenderTarget(rt);
        try cb.clear(.{ .r = 0.2, .g = 0.2, .b = 0.2, .a = 1 });
        try cb.bindPipeline(pipe);
        try cb.bindUniformBuffer(0, ubo);
        try cb.draw(3, 0);
        try ctx.submit(cb);
    }

    const fb = std.mem.bytesAsSlice(u32, try dev.mapResource(rt));
    var nonclear: u32 = 0;
    for (fb[0 .. @as(usize, W) * H]) |p| {
        const r: u8 = @truncate(p);
        const g: u8 = @truncate(p >> 8);
        const b: u8 = @truncate(p >> 16);
        if (!(r > 40 and r < 60 and g > 40 and g < 60 and b > 40 and b < 60)) nonclear += 1;
    }
    try std.testing.expect(nonclear > 2000); // the pulled+transformed triangle covers the interior
}

// ARITHMETIC-ON-IPA ORACLE support (see ORACLE4 below).
// FS: vx = CompositeExtract(load varying, 0); o = vec4(vx*0.5 + 0.5, 0, 0, 1).
// An arithmetic (FMul/FAdd) consumer of a freshly-IPA'd varying. Pure-passthrough
// (ORACLE2) of the same slot reads linearly. This checks the FP-ALU read of the IPA
// result (the frame-based oracle for the FMUL PDIV fix).
fn buildArithIpaFragmentSpirv(gpa: std.mem.Allocator) !Builder {
    // ids: void=1 f32=2 v4f=3 pIn=4 pOut=5 voidfn=6 colIn=7 colOut=8 main=9 entry=10
    //      loaded=11 vx=12 half=13 vhalf=14 zero=15 one=16 r=17 outv=18.
    var b = try Builder.init(gpa, 19);
    errdefer b.deinit(gpa);
    try b.emit(gpa, op.EntryPoint, &.{ op.ExecutionModel.fragment, 9, 0, 7, 8 });
    try b.emit(gpa, op.Decorate, &.{ 7, op.Decoration.location, 0 });
    try b.emit(gpa, op.Decorate, &.{ 8, op.Decoration.location, 0 });
    try b.emit(gpa, op.TypeVoid, &.{1});
    try b.emit(gpa, op.TypeFloat, &.{ 2, 32 });
    try b.emit(gpa, op.TypeVector, &.{ 3, 2, 4 });
    try b.emit(gpa, op.TypePointer, &.{ 4, op.StorageClass.input, 3 });
    try b.emit(gpa, op.TypePointer, &.{ 5, op.StorageClass.output, 3 });
    try b.emit(gpa, op.TypeFunction, &.{ 6, 1 });
    try b.emit(gpa, op.Variable, &.{ 4, 7, op.StorageClass.input });
    try b.emit(gpa, op.Variable, &.{ 5, 8, op.StorageClass.output });
    try b.emit(gpa, op.Constant, &.{ 2, 13, @bitCast(@as(f32, 0.5)) }); // 0.5
    try b.emit(gpa, op.Constant, &.{ 2, 15, @bitCast(@as(f32, 0.0)) }); // 0.0
    try b.emit(gpa, op.Constant, &.{ 2, 16, @bitCast(@as(f32, 1.0)) }); // 1.0
    try b.emit(gpa, op.Function, &.{ 1, 9, 0, 6 });
    try b.emit(gpa, op.Label, &.{10});
    try b.emit(gpa, op.Load, &.{ 3, 11, 7 }); // v = load interpolated varying
    try b.emit(gpa, op.CompositeExtract, &.{ 2, 12, 11, 0 }); // vx = v.x
    try b.emit(gpa, op.FMul, &.{ 2, 14, 12, 13 }); // vhalf = vx * 0.5
    try b.emit(gpa, op.FAdd, &.{ 2, 17, 14, 13 }); // r = vhalf + 0.5
    try b.emit(gpa, op.CompositeConstruct, &.{ 3, 18, 17, 15, 15, 16 }); // vec4(r,0,0,1)
    try b.emit(gpa, op.Store, &.{ 8, 18 });
    try b.emit(gpa, op.Return, &.{});
    try b.emit(gpa, op.FunctionEnd, &.{});
    return b;
}

fn runArithIpa(gpa: std.mem.Allocator, dev: hal.Device, w: u32, h: u32, clear: hal.Color) ![]u32 {
    var vs_b = try buildPosPassthroughVertexSpirv(gpa);
    defer vs_b.deinit(gpa);
    var ps_b = try buildArithIpaFragmentSpirv(gpa);
    defer ps_b.deinit(gpa);
    const vs_sass = try compileToSass(gpa, std.mem.sliceAsBytes(vs_b.words.items), .vertex);
    defer gpa.free(vs_sass);
    const ps_sass = try compileToSass(gpa, std.mem.sliceAsBytes(ps_b.words.items), .fragment);
    defer gpa.free(ps_sass);

    const rt = try dev.createResource(.{ .image = .{ .width = w, .height = h, .format = .bgra8_unorm, .usage = .{ .render_target = true } } });
    defer dev.destroyResource(rt);
    const vs = try dev.createShaderModule(.{ .stage = .vertex, .code = std.mem.sliceAsBytes(vs_sass) });
    defer dev.destroyShaderModule(vs);
    const ps = try dev.createShaderModule(.{ .stage = .fragment, .code = std.mem.sliceAsBytes(ps_sass) });
    defer dev.destroyShaderModule(ps);

    const vbuf = try dev.createResource(.{ .buffer = .{ .size = 3 * 16, .usage = .{ .vertex = true } } });
    defer dev.destroyResource(vbuf);
    const verts = [_]f32{
        0.5, 0.9, 0.5, 1.0,
        0.1, 0.1, 0.5, 1.0,
        0.9, 0.1, 0.5, 1.0,
    };
    @memcpy(std.mem.bytesAsSlice(f32, try dev.mapResource(vbuf))[0..verts.len], &verts);
    const attrs = [_]hal.VertexAttribute{.{ .location = 0, .format = .rgba8_unorm, .offset = 0 }};
    const pipe = try dev.createPipeline(.{
        .vertex = vs,
        .fragment = ps,
        .vertex_layout = .{ .stride = 16, .attributes = &attrs },
        .color_format = .bgra8_unorm,
    });
    defer dev.destroyPipeline(pipe);

    const ctx = try dev.createContext();
    defer ctx.deinit();
    {
        const cb = try ctx.beginCommands();
        defer cb.deinit();
        try cb.setRenderTarget(rt);
        try cb.clear(clear);
        try cb.bindPipeline(pipe);
        try cb.bindVertexBuffer(vbuf);
        try cb.draw(3, 0);
        try ctx.submit(cb);
    }
    const fb = std.mem.bytesAsSlice(u32, try dev.mapResource(rt));
    const out = try gpa.alloc(u32, @as(usize, w) * h);
    errdefer gpa.free(out);
    // Tight (w*4) readback: the software RT is tight and nvidia mapResource de-swizzles
    // the block-linear RT into a tight w*4 buffer. (pitchBytes(w)/4 == w only at 256.)
    const pitch_px = w;
    var y: u32 = 0;
    while (y < h) : (y += 1) {
        @memcpy(out[@as(usize, y) * w ..][0..w], fb[@as(usize, y) * pitch_px ..][0..w]);
    }
    return out;
}

// ORACLE4: an arithmetic fragment shader (FMul/FAdd on a freshly-IPA'd varying)
// reads the interpolated value correctly on the GPU. The bare-passthrough ORACLE2
// only proves the IPA + store path. This proves the FP-ALU consumer. It is the
// regression guard for the FMUL PDIV-field bug (NAK OpFMul sets bits 84..86 = 4;
// leaving the PT default 7 made `0.5*0.5` saturate, corrupting every FP-multiply on
// the GPU: vkcube's lighting/sRGB, derivatives, all read garbage). FS: r = vx*0.5+0.5.
test "ORACLE4: arithmetic FS (FMul on a freshly-IPA'd varying) reads correctly on the NVIDIA GPU (skips without a GPU)" {
    const gpa = std.testing.allocator;
    const dev = NvDevice.create(gpa) catch return error.SkipZigTest;
    defer dev.deinit();
    const W: u32 = 256;
    const H: u32 = 256;
    const clear = hal.Color{ .r = 0.125, .g = 0.125, .b = 0.125, .a = 1 };
    const fb = try runArithIpa(gpa, dev, W, H, clear);
    defer gpa.free(fb);

    const clear_px: u32 = 0xff202020;
    // Geometry (clip [0,1]): top=(0.5,0.9), BL=(0.1,0.1), BR=(0.9,0.1).
    // clip -> pixel: px = (cx+1)*W/2, py = (cy+1)*H/2 (NDC -1 -> row 0, software-
    // matching; the clip-top vertex lands LOW at py~243, the base HIGH at py~140).
    const Vtx = struct { x: f32, y: f32 };
    const v0 = Vtx{ .x = (0.5 + 1) * @as(f32, W) / 2, .y = (0.9 + 1) * @as(f32, H) / 2 };
    const v1 = Vtx{ .x = (0.1 + 1) * @as(f32, W) / 2, .y = (0.1 + 1) * @as(f32, H) / 2 };
    const v2 = Vtx{ .x = (0.9 + 1) * @as(f32, W) / 2, .y = (0.1 + 1) * @as(f32, H) / 2 };
    // varying.x at each vertex = clip x = 0.5, 0.1, 0.9. Expected R = vx*0.5+0.5.
    const cx = [3]f32{ 0.5, 0.1, 0.9 };
    const denom = (v1.y - v2.y) * (v0.x - v2.x) + (v2.x - v1.x) * (v0.y - v2.y);

    var checked: u32 = 0;
    var bad: u32 = 0;
    var worst: f32 = 0;
    var py: u32 = 142;
    while (py < 240) : (py += 4) {
        var px: u32 = 130;
        while (px < 250) : (px += 4) {
            const fx: f32 = @floatFromInt(px);
            const fy: f32 = @floatFromInt(py);
            const w0 = ((v1.y - v2.y) * (fx - v2.x) + (v2.x - v1.x) * (fy - v2.y)) / denom;
            const w1 = ((v2.y - v0.y) * (fx - v2.x) + (v0.x - v2.x) * (fy - v2.y)) / denom;
            const w2 = 1.0 - w0 - w1;
            if (w0 < 0.08 or w1 < 0.08 or w2 < 0.08) continue; // interior only
            const p = fb[py * W + px];
            try std.testing.expect(p != clear_px); // the triangle covers the interior
            const vx = w0 * cx[0] + w1 * cx[1] + w2 * cx[2];
            const exp_r = vx * 0.5 + 0.5;
            const got_r = @as(f32, @floatFromInt(@as(u8, @truncate(p >> 16)))) / 255.0;
            const d = @abs(got_r - exp_r);
            checked += 1;
            if (d > worst) worst = d;
            if (d > 0.08) bad += 1;
        }
    }
    try std.testing.expect(checked > 60);
    try std.testing.expectEqual(@as(u32, 0), bad);
}

// DIVERGENT-BRANCH ORACLE support (see ORACLE5 below).
// FS: vx = varying.x; if (vx <= 0.5) o = 1.0 else o = 0.25; out = vec4(o,o,o,1).
// A per-lane divergent predicated branch (OpBranchConditional + OpSelectionMerge +
// OpPhi). vx ranges across the triangle so neighbouring quad lanes take different
// arms. The warp diverges then reconverges at the merge before the store.
// Regression guard for the predicated-BRA encoding (NAK puts the taken condition at
// bits 87..89 + negate 90, not the 12..14 guard. sm>=100 word-unit rel offset is
// split across 16..24 + 34..82) and the BSSY/BSYNC convergence. A wrong
// predicate/offset made the branch take the wrong arm (the vkcube sRGB G/B-saturate
// bug) or run the warp off the end (Xid 13).
fn buildDivergentFragmentSpirv(gpa: std.mem.Allocator) !Builder {
    // ids: void=1 f32=2 v4f=3 bool=4 pIn=5 pOut=6 voidfn=7 colIn=8 colOut=9 main=10
    //      entry=11 loaded=12 vx=13 half=14 one=15 quarter=16 zero=17 cond=18
    //      Lthen=19 Lelse=20 Lmerge=21 o=22 outv=23.
    var b = try Builder.init(gpa, 24);
    errdefer b.deinit(gpa);
    try b.emit(gpa, op.EntryPoint, &.{ op.ExecutionModel.fragment, 10, 0, 8, 9 });
    try b.emit(gpa, op.Decorate, &.{ 8, op.Decoration.location, 0 });
    try b.emit(gpa, op.Decorate, &.{ 9, op.Decoration.location, 0 });
    try b.emit(gpa, op.TypeVoid, &.{1});
    try b.emit(gpa, op.TypeFloat, &.{ 2, 32 });
    try b.emit(gpa, op.TypeVector, &.{ 3, 2, 4 });
    try b.emit(gpa, op.TypeBool, &.{4});
    try b.emit(gpa, op.TypePointer, &.{ 5, op.StorageClass.input, 3 });
    try b.emit(gpa, op.TypePointer, &.{ 6, op.StorageClass.output, 3 });
    try b.emit(gpa, op.TypeFunction, &.{ 7, 1 });
    try b.emit(gpa, op.Variable, &.{ 5, 8, op.StorageClass.input });
    try b.emit(gpa, op.Variable, &.{ 6, 9, op.StorageClass.output });
    try b.emit(gpa, op.Constant, &.{ 2, 14, @bitCast(@as(f32, 0.5)) }); // 0.5 (threshold)
    try b.emit(gpa, op.Constant, &.{ 2, 15, @bitCast(@as(f32, 1.0)) }); // 1.0 (then arm)
    try b.emit(gpa, op.Constant, &.{ 2, 16, @bitCast(@as(f32, 0.25)) }); // 0.25 (else arm)
    try b.emit(gpa, op.Constant, &.{ 2, 17, @bitCast(@as(f32, 0.0)) }); // 0.0
    try b.emit(gpa, op.Function, &.{ 1, 10, 0, 7 });
    try b.emit(gpa, op.Label, &.{11});
    try b.emit(gpa, op.Load, &.{ 3, 12, 8 }); // v = load interpolated varying
    try b.emit(gpa, op.CompositeExtract, &.{ 2, 13, 12, 0 }); // vx = v.x
    try b.emit(gpa, op.FOrdLessThanEqual, &.{ 4, 18, 13, 14 }); // cond = vx <= 0.5
    try b.emit(gpa, op.SelectionMerge, &.{ 21, 0 });
    try b.emit(gpa, op.BranchConditional, &.{ 18, 19, 20 });
    try b.emit(gpa, op.Label, &.{19}); // then
    try b.emit(gpa, op.Branch, &.{21});
    try b.emit(gpa, op.Label, &.{20}); // else
    try b.emit(gpa, op.Branch, &.{21});
    try b.emit(gpa, op.Label, &.{21}); // merge
    try b.emit(gpa, op.Phi, &.{ 2, 22, 15, 19, 16, 20 }); // o = phi(1.0 from then, 0.25 from else)
    try b.emit(gpa, op.CompositeConstruct, &.{ 3, 23, 22, 22, 22, 15 }); // vec4(o,o,o,1)
    try b.emit(gpa, op.Store, &.{ 9, 23 });
    try b.emit(gpa, op.Return, &.{});
    try b.emit(gpa, op.FunctionEnd, &.{});
    return b;
}

fn runDivergent(gpa: std.mem.Allocator, dev: hal.Device, w: u32, h: u32, clear: hal.Color) ![]u32 {
    var vs_b = try buildPosPassthroughVertexSpirv(gpa);
    defer vs_b.deinit(gpa);
    var ps_b = try buildDivergentFragmentSpirv(gpa);
    defer ps_b.deinit(gpa);
    const vs_sass = try compileToSass(gpa, std.mem.sliceAsBytes(vs_b.words.items), .vertex);
    defer gpa.free(vs_sass);
    const ps_sass = try compileToSass(gpa, std.mem.sliceAsBytes(ps_b.words.items), .fragment);
    defer gpa.free(ps_sass);

    const rt = try dev.createResource(.{ .image = .{ .width = w, .height = h, .format = .bgra8_unorm, .usage = .{ .render_target = true } } });
    defer dev.destroyResource(rt);
    const vs = try dev.createShaderModule(.{ .stage = .vertex, .code = std.mem.sliceAsBytes(vs_sass) });
    defer dev.destroyShaderModule(vs);
    const ps = try dev.createShaderModule(.{ .stage = .fragment, .code = std.mem.sliceAsBytes(ps_sass) });
    defer dev.destroyShaderModule(ps);

    const vbuf = try dev.createResource(.{ .buffer = .{ .size = 3 * 16, .usage = .{ .vertex = true } } });
    defer dev.destroyResource(vbuf);
    const verts = [_]f32{
        0.5, 0.9, 0.5, 1.0,
        0.1, 0.1, 0.5, 1.0,
        0.9, 0.1, 0.5, 1.0,
    };
    @memcpy(std.mem.bytesAsSlice(f32, try dev.mapResource(vbuf))[0..verts.len], &verts);
    const attrs = [_]hal.VertexAttribute{.{ .location = 0, .format = .rgba8_unorm, .offset = 0 }};
    const pipe = try dev.createPipeline(.{
        .vertex = vs,
        .fragment = ps,
        .vertex_layout = .{ .stride = 16, .attributes = &attrs },
        .color_format = .bgra8_unorm,
    });
    defer dev.destroyPipeline(pipe);

    const ctx = try dev.createContext();
    defer ctx.deinit();
    {
        const cb = try ctx.beginCommands();
        defer cb.deinit();
        try cb.setRenderTarget(rt);
        try cb.clear(clear);
        try cb.bindPipeline(pipe);
        try cb.bindVertexBuffer(vbuf);
        try cb.draw(3, 0);
        try ctx.submit(cb);
    }
    const fb = std.mem.bytesAsSlice(u32, try dev.mapResource(rt));
    const out = try gpa.alloc(u32, @as(usize, w) * h);
    errdefer gpa.free(out);
    // Tight (w*4) readback: the software RT is tight and nvidia mapResource de-swizzles
    // the block-linear RT into a tight w*4 buffer. (pitchBytes(w)/4 == w only at 256.)
    const pitch_px = w;
    var y: u32 = 0;
    while (y < h) : (y += 1) {
        @memcpy(out[@as(usize, y) * w ..][0..w], fb[@as(usize, y) * pitch_px ..][0..w]);
    }
    return out;
}

// ORACLE5: a divergent predicated branch takes the correct arm per-lane and the
// warp reconverges. FS: if (vx <= 0.5) o = 1.0 else o = 0.25. vx varies across the
// triangle so neighbouring lanes diverge. Each interior pixel reads the arm its own
// vx selects (1.0 or 0.25), away from the seam. Guards the predicated-BRA encoding
// (condition @87..89 + negate 90; sm>=100 word-unit rel offset split 16..24 + 34..82)
// and the BSSY/BSYNC convergence. This is the fix that made stock vkcube's per-channel
// sRGB branches stop taking the wrong arm (G/B saturating to 255).
test "ORACLE5: divergent predicated branch takes the correct arm per-lane on the NVIDIA GPU (skips without a GPU)" {
    const gpa = std.testing.allocator;
    const dev = NvDevice.create(gpa) catch return error.SkipZigTest;
    defer dev.deinit();
    const W: u32 = 256;
    const H: u32 = 256;
    const clear = hal.Color{ .r = 0.125, .g = 0.125, .b = 0.125, .a = 1 };
    const fb = try runDivergent(gpa, dev, W, H, clear);
    defer gpa.free(fb);

    const clear_px: u32 = 0xff202020;
    const Vtx = struct { x: f32, y: f32 };
    // py = (cy+1)/2*H (NDC -1 -> row 0, software-matching): clip-top lands LOW (~243),
    // the base HIGH (~140).
    const v0 = Vtx{ .x = (0.5 + 1) * @as(f32, W) / 2, .y = (0.9 + 1) * @as(f32, H) / 2 };
    const v1 = Vtx{ .x = (0.1 + 1) * @as(f32, W) / 2, .y = (0.1 + 1) * @as(f32, H) / 2 };
    const v2 = Vtx{ .x = (0.9 + 1) * @as(f32, W) / 2, .y = (0.1 + 1) * @as(f32, H) / 2 };
    const cx = [3]f32{ 0.5, 0.1, 0.9 }; // varying.x at each vertex = clip x
    const denom = (v1.y - v2.y) * (v0.x - v2.x) + (v2.x - v1.x) * (v0.y - v2.y);

    var checked: u32 = 0;
    var bad: u32 = 0;
    var py: u32 = 142;
    while (py < 240) : (py += 4) {
        var px: u32 = 130;
        while (px < 250) : (px += 4) {
            const fx: f32 = @floatFromInt(px);
            const fy: f32 = @floatFromInt(py);
            const w0 = ((v1.y - v2.y) * (fx - v2.x) + (v2.x - v1.x) * (fy - v2.y)) / denom;
            const w1 = ((v2.y - v0.y) * (fx - v2.x) + (v0.x - v2.x) * (fy - v2.y)) / denom;
            const w2 = 1.0 - w0 - w1;
            if (w0 < 0.08 or w1 < 0.08 or w2 < 0.08) continue; // interior only
            const vx = w0 * cx[0] + w1 * cx[1] + w2 * cx[2];
            // Skip a band around the seam (vx == 0.5) where rasterizer rounding can
            // flip the arm by a pixel: a real branch BUG saturates a whole half, not
            // a thin seam.
            if (@abs(vx - 0.5) < 0.04) continue;
            const p = fb[py * W + px];
            try std.testing.expect(p != clear_px); // the triangle covers the interior
            const exp: f32 = if (vx <= 0.5) 1.0 else 0.25;
            const got = @as(f32, @floatFromInt(@as(u8, @truncate(p >> 16)))) / 255.0;
            checked += 1;
            if (@abs(got - exp) > 0.08) bad += 1;
        }
    }
    try std.testing.expect(checked > 60);
    try std.testing.expectEqual(@as(u32, 0), bad);
}

// UNIFORM-BLOCK (EGL/GLES default-uniform-block) ORACLE on the GPU.
//
// The GLSL ES front end packs all `uniform` scalars/vectors/matrices into one
// PushConstant block of floats and eagerly loads every float in the entry block,
// including ones the shader never reads (e.g. LightSourcePosition.w). On NVIDIA each
// load is a decoupled LDG, and the linear-scan allocator reuses a dead load's
// destination register for a later synchronous MOV (an address offset). Without a
// write-after-write scoreboard wait, the dead async load lands after the MOV and
// clobbers the offset, so the next member's LDG reads a garbage address and the
// uniform reads the wrong value (the es2gears MaterialColor came back (1,1,0), so a
// green material rendered yellow). This oracle compiles the exact es2gears VS+FS via
// the GLSL front end, binds a UBO with a green MaterialColor, and asserts the rendered
// fragment is green. Permanent regression guard for the schedule.zig WAW fix.
test "ORACLE: a default-uniform-block (es2gears) VS reads its bound MaterialColor on the NVIDIA GPU (skips without a GPU)" {
    const gpa = std.testing.allocator;
    const glsl = @import("../../glsl.zig");
    const dev = NvDevice.create(gpa) catch return error.SkipZigTest;
    defer dev.deinit();
    const W: u32 = 128;
    const H: u32 = 128;

    const vs_src =
        \\attribute vec3 position;
        \\attribute vec3 normal;
        \\uniform mat4 ModelViewProjectionMatrix;
        \\uniform mat4 NormalMatrix;
        \\uniform vec4 LightSourcePosition;
        \\uniform vec4 MaterialColor;
        \\varying vec4 Color;
        \\void main(void) {
        \\    vec3 N = normalize(vec3(NormalMatrix * vec4(normal, 1.0)));
        \\    vec3 L = normalize(LightSourcePosition.xyz);
        \\    float diffuse = max(dot(N, L), 0.0);
        \\    float ambient = 0.2;
        \\    Color = vec4((ambient + diffuse) * MaterialColor.xyz, MaterialColor.a);
        \\    gl_Position = ModelViewProjectionMatrix * vec4(position, 1.0);
        \\}
    ;
    const fs_src =
        \\precision mediump float;
        \\varying vec4 Color;
        \\void main(void) { gl_FragColor = Color; }
    ;
    const vs_spv = try glsl.compileForStage(gpa, vs_src, .vertex);
    defer gpa.free(vs_spv);
    const fs_spv = try glsl.compileForStage(gpa, fs_src, .fragment);
    defer gpa.free(fs_spv);

    const rt = try dev.createResource(.{ .image = .{ .width = W, .height = H, .format = .bgra8_unorm, .usage = .{ .render_target = true } } });
    defer dev.destroyResource(rt);
    const vs = try dev.createShaderModule(.{ .stage = .vertex, .code = vs_spv });
    defer dev.destroyShaderModule(vs);
    const fs = try dev.createShaderModule(.{ .stage = .fragment, .code = fs_spv });
    defer dev.destroyShaderModule(fs);

    // A full-screen quad as two triangles: position (vec3) + normal (vec3 = +z, facing
    // the light so it is fully lit). The HAL draw is triangle-list, so 6 verts.
    const Vtx = extern struct { px: f32, py: f32, pz: f32, nx: f32, ny: f32, nz: f32 };
    const verts = [_]Vtx{
        .{ .px = -0.9, .py = -0.9, .pz = 0, .nx = 0, .ny = 0, .nz = 1 },
        .{ .px = 0.9, .py = -0.9, .pz = 0, .nx = 0, .ny = 0, .nz = 1 },
        .{ .px = -0.9, .py = 0.9, .pz = 0, .nx = 0, .ny = 0, .nz = 1 },
        .{ .px = 0.9, .py = -0.9, .pz = 0, .nx = 0, .ny = 0, .nz = 1 },
        .{ .px = 0.9, .py = 0.9, .pz = 0, .nx = 0, .ny = 0, .nz = 1 },
        .{ .px = -0.9, .py = 0.9, .pz = 0, .nx = 0, .ny = 0, .nz = 1 },
    };
    const vbuf = try dev.createResource(.{ .buffer = .{ .size = verts.len * @sizeOf(Vtx), .usage = .{ .vertex = true } } });
    defer dev.destroyResource(vbuf);
    @memcpy(std.mem.bytesAsSlice(Vtx, try dev.mapResource(vbuf))[0..verts.len], &verts);

    // The std140 block: MVP@0 (identity), NormalMatrix@64 (identity), Light@128 (+z),
    // MaterialColor@144 = green (0,1,0,1). 40 floats = 160 bytes. Light.w (float 35) is
    // never read. It is the dead load whose register reuse triggered the WAW bug.
    const ubo = try dev.createResource(.{ .buffer = .{ .size = 160, .usage = .{ .uniform = true } } });
    defer dev.destroyResource(ubo);
    {
        const f = std.mem.bytesAsSlice(f32, try dev.mapResource(ubo));
        @memset(f[0..40], 0);
        f[0] = 1;
        f[5] = 1;
        f[10] = 1;
        f[15] = 1; // MVP identity
        f[16] = 1;
        f[21] = 1;
        f[26] = 1;
        f[31] = 1; // NormalMatrix identity
        f[32] = 0;
        f[33] = 0;
        f[34] = 1;
        f[35] = 0; // LightSourcePosition (0,0,1,0)
        f[36] = 0;
        f[37] = 1;
        f[38] = 0;
        f[39] = 1; // MaterialColor = green
    }

    const attrs = [_]hal.VertexAttribute{
        .{ .location = 0, .format = .r32g32b32_float, .offset = 0 },
        .{ .location = 1, .format = .r32g32b32_float, .offset = 12 },
    };
    const pipe = try dev.createPipeline(.{
        .vertex = vs,
        .fragment = fs,
        .vertex_layout = .{ .stride = @sizeOf(Vtx), .attributes = &attrs },
        .color_format = .bgra8_unorm,
    });
    defer dev.destroyPipeline(pipe);

    const ctx = try dev.createContext();
    defer ctx.deinit();
    {
        const cb = try ctx.beginCommands();
        defer cb.deinit();
        try cb.setRenderTarget(rt);
        try cb.clear(.{ .r = 0.1, .g = 0.1, .b = 0.1, .a = 1 });
        try cb.bindPipeline(pipe);
        try cb.bindUniformBuffer(0, ubo);
        try cb.bindVertexBuffer(vbuf);
        try cb.draw(6, 0);
        try ctx.submit(cb);
    }

    const fb = std.mem.bytesAsSlice(u32, try dev.mapResource(rt));
    const pitch_px = nvidia.threed.pitchBytes(W) / 4;
    const center = fb[64 * pitch_px + 64];
    const r: u8 = @truncate(center >> 16);
    const g: u8 = @truncate(center >> 8);
    const b: u8 = @truncate(center);
    // The fully-lit (ambient 0.2 + diffuse ~0.8 = 1.0) green material renders green. The WAW
    // bug rendered this yellow (r ~= 255) because MaterialColor.x read 1.0 not 0.0.
    if (!(g > 180 and r < 80 and b < 80)) std.debug.print("\n[ORACLE-UBO] center=({d},{d},{d}) (expect green)\n", .{ r, g, b });
    try std.testing.expect(g > 180);
    try std.testing.expect(r < 80);
    try std.testing.expect(b < 80);
}

// A fragment shader producing a boolean value from comparisons combined by a logical
// operator (`&&`), then consumed by a select. This is the construct that panicked the
// nvidia isel (a bool-typed `.binary` bit_and routed to `gprOf` -> unreachable,
// isel.zig:570). glmark2's light-phong FS hit this. The fix lowers the bool combine to
// PLOP3 (predicate logic) instead of a GPR LOP3. This oracle renders the FS on the real
// GPU and checks the bool-selected color against the software golden: a varying color
// (0.8, 0.2) makes `(r>0.5 && g>0.5)` false, so the select returns blue, not green. A
// green readback means the predicate combine was wrong. Permanent regression guard for
// the PLOP3 codegen.
test "ORACLE: a boolean VALUE from a comparison-and (the light-phong panic construct) selects correctly on the NVIDIA GPU (skips without a GPU)" {
    const gpa = std.testing.allocator;
    const glsl = @import("../../glsl.zig");
    const dev = NvDevice.create(gpa) catch return error.SkipZigTest;
    defer dev.deinit();
    const W: u32 = 128;
    const H: u32 = 128;

    // The VS passes a constant color varying (0.8, 0.2, 0.0) so the FS bool is the same at
    // every covered pixel (a deterministic oracle, not corner-dependent).
    const vs_src =
        \\attribute vec3 position;
        \\varying vec4 Color;
        \\void main(void) {
        \\    Color = vec4(0.8, 0.2, 0.0, 1.0);
        \\    gl_Position = vec4(position, 1.0);
        \\}
    ;
    // bool both = (Color.r > 0.5) && (Color.g > 0.5); gl_FragColor = both ? green : blue.
    // Color = (0.8, 0.2): r>0.5 is true, g>0.5 is false -> both false -> blue.
    const fs_src =
        \\precision mediump float;
        \\varying vec4 Color;
        \\void main(void) {
        \\    bool br = Color.r > 0.5;
        \\    bool bg = Color.g > 0.5;
        \\    bool both = br && bg;
        \\    gl_FragColor = both ? vec4(0.0, 1.0, 0.0, 1.0) : vec4(0.0, 0.0, 1.0, 1.0);
        \\}
    ;
    const vs_spv = try glsl.compileForStage(gpa, vs_src, .vertex);
    defer gpa.free(vs_spv);
    const fs_spv = try glsl.compileForStage(gpa, fs_src, .fragment);
    defer gpa.free(fs_spv);

    const rt = try dev.createResource(.{ .image = .{ .width = W, .height = H, .format = .bgra8_unorm, .usage = .{ .render_target = true } } });
    defer dev.destroyResource(rt);
    const vs = try dev.createShaderModule(.{ .stage = .vertex, .code = vs_spv });
    defer dev.destroyShaderModule(vs);
    const fs = try dev.createShaderModule(.{ .stage = .fragment, .code = fs_spv });
    defer dev.destroyShaderModule(fs);

    const Vtx = extern struct { px: f32, py: f32, pz: f32 };
    const verts = [_]Vtx{
        .{ .px = -0.9, .py = -0.9, .pz = 0 },
        .{ .px = 0.9, .py = -0.9, .pz = 0 },
        .{ .px = -0.9, .py = 0.9, .pz = 0 },
        .{ .px = 0.9, .py = -0.9, .pz = 0 },
        .{ .px = 0.9, .py = 0.9, .pz = 0 },
        .{ .px = -0.9, .py = 0.9, .pz = 0 },
    };
    const vbuf = try dev.createResource(.{ .buffer = .{ .size = verts.len * @sizeOf(Vtx), .usage = .{ .vertex = true } } });
    defer dev.destroyResource(vbuf);
    @memcpy(std.mem.bytesAsSlice(Vtx, try dev.mapResource(vbuf))[0..verts.len], &verts);

    const attrs = [_]hal.VertexAttribute{.{ .location = 0, .format = .r32g32b32_float, .offset = 0 }};
    const pipe = try dev.createPipeline(.{
        .vertex = vs,
        .fragment = fs,
        .vertex_layout = .{ .stride = @sizeOf(Vtx), .attributes = &attrs },
        .color_format = .bgra8_unorm,
    });
    defer dev.destroyPipeline(pipe);

    const ctx = try dev.createContext();
    defer ctx.deinit();
    {
        const cb = try ctx.beginCommands();
        defer cb.deinit();
        try cb.setRenderTarget(rt);
        try cb.clear(.{ .r = 0.1, .g = 0.1, .b = 0.1, .a = 1 });
        try cb.bindPipeline(pipe);
        try cb.bindVertexBuffer(vbuf);
        try cb.draw(6, 0);
        try ctx.submit(cb);
    }

    const fb = std.mem.bytesAsSlice(u32, try dev.mapResource(rt));
    const pitch_px = nvidia.threed.pitchBytes(W) / 4;
    const center = fb[64 * pitch_px + 64];
    const r: u8 = @truncate(center >> 16);
    const g: u8 = @truncate(center >> 8);
    const b: u8 = @truncate(center);
    // (0.8>0.5) && (0.2>0.5) == false -> blue (the false arm of the select).
    if (!(b > 180 and r < 80 and g < 80)) std.debug.print("\n[ORACLE-BOOL] center=({d},{d},{d}) (expect blue)\n", .{ r, g, b });
    try std.testing.expect(b > 180);
    try std.testing.expect(r < 80);
    try std.testing.expect(g < 80);
}

/// Render a flat-color triangle into a w*h render target and return the de-swizzled
/// image as a tightly-packed (w*h) RGBA8 slice, read straight from mapResource (which
/// de-swizzles the block-linear surface). Unlike `run`, this does not re-pitch the
/// result by pitchBytes(w): mapResource already hands back row-major w*h*4, so a
/// non-tile-aligned width (e.g. 500, where pitchBytes(w)/4 != w) reads correctly.
/// The caller owns and frees the returned slice.
fn runTight(gpa: std.mem.Allocator, dev: hal.Device, w: u32, h: u32, clear: hal.Color, color: [4]f32) ![]u32 {
    var vs_b = try buildVertexSpirv(gpa);
    defer vs_b.deinit(gpa);
    var ps_b = try buildFragmentSpirv(gpa, color);
    defer ps_b.deinit(gpa);
    const vs_sass = try compileToSass(gpa, std.mem.sliceAsBytes(vs_b.words.items), .vertex);
    defer gpa.free(vs_sass);
    const ps_sass = try compileToSass(gpa, std.mem.sliceAsBytes(ps_b.words.items), .fragment);
    defer gpa.free(ps_sass);

    const rt = try dev.createResource(.{ .image = .{ .width = w, .height = h, .format = .bgra8_unorm, .usage = .{ .render_target = true } } });
    defer dev.destroyResource(rt);
    const vs = try dev.createShaderModule(.{ .stage = .vertex, .code = std.mem.sliceAsBytes(vs_sass) });
    defer dev.destroyShaderModule(vs);
    const ps = try dev.createShaderModule(.{ .stage = .fragment, .code = std.mem.sliceAsBytes(ps_sass) });
    defer dev.destroyShaderModule(ps);
    const vbuf = try dev.createResource(.{ .buffer = .{ .size = 3 * 16, .usage = .{ .vertex = true } } });
    defer dev.destroyResource(vbuf);
    // A near-fullscreen triangle so a large interior region is the flat `color`.
    const verts = [_]f32{
        0.0,  0.9,  0.0, 1.0,
        -0.9, -0.9, 0.0, 1.0,
        0.9,  -0.9, 0.0, 1.0,
    };
    @memcpy(std.mem.bytesAsSlice(f32, try dev.mapResource(vbuf))[0..verts.len], &verts);
    const attrs = [_]hal.VertexAttribute{.{ .location = 0, .format = .rgba8_unorm, .offset = 0 }};
    const pipe = try dev.createPipeline(.{
        .vertex = vs,
        .fragment = ps,
        .vertex_layout = .{ .stride = 16, .attributes = &attrs },
        .color_format = .bgra8_unorm,
    });
    defer dev.destroyPipeline(pipe);
    const ctx = try dev.createContext();
    defer ctx.deinit();
    {
        const cb = try ctx.beginCommands();
        defer cb.deinit();
        try cb.setRenderTarget(rt);
        try cb.clear(clear);
        try cb.bindPipeline(pipe);
        try cb.bindVertexBuffer(vbuf);
        try cb.draw(3, 0);
        try ctx.submit(cb);
    }
    const fb = std.mem.bytesAsSlice(u32, try dev.mapResource(rt));
    const out = try gpa.alloc(u32, @as(usize, w) * h);
    @memcpy(out, fb[0 .. @as(usize, w) * h]);
    return out;
}

// ORACLE (de-swizzle math): the fast block-linear de-swizzle in device.zig
// mapResource (the precomputed off_x[] + y_base path) is byte-identical to the
// reference graphics.blColorPixelOffset for every pixel, at the speckle-prone sizes:
// 800x600 (a GOB-aligned 3200-byte row = exactly 50 GOBs, no padding), 500x500 (the
// padded vkcube size, 2000->2048-byte stride), GOB-misaligned widths, and a partial
// last block-row (601 rows = 75 GOB-rows, last block-row only 11 of 16 GOBs). This
// pins the perf optimization to the proven-correct gather and catches any off-by-one
// in the table math at a GOB or block boundary. (CPU-only; no GPU needed.)
test "ORACLE: the fast de-swizzle table equals blColorPixelOffset at every pixel (GOB-aligned + partial block-row)" {
    const sizes = [_][2]u32{ .{ 500, 500 }, .{ 800, 600 }, .{ 640, 480 }, .{ 1280, 720 }, .{ 17, 200 }, .{ 800, 601 } };
    for (sizes) |sz| {
        const w = sz[0];
        const h = sz[1];
        const GOB_W: u32 = 64;
        const GOB_H: u32 = 8;
        const BH: u32 = 16;
        const block_bytes: usize = GOB_W * GOB_H * BH;
        const row_bytes = std.mem.alignForward(u32, w * 4, GOB_W);
        const gobs_per_row = row_bytes / GOB_W;
        var mismatches: u32 = 0;
        var first: [2]u32 = .{ 0, 0 };
        var y: u32 = 0;
        while (y < h) : (y += 1) {
            const gob_row = y / GOB_H;
            const block_row = gob_row / BH;
            const gob_in_block = gob_row % BH;
            const yb = y % GOB_H;
            const y_base: usize = @as(usize, block_row) * gobs_per_row * block_bytes +
                @as(usize, gob_in_block) * (GOB_W * GOB_H) +
                (yb / 4) * 128 + (yb % 4) * 16;
            var x: u32 = 0;
            while (x < w) : (x += 1) {
                const xb = (x * 4) % GOB_W;
                const gob_col = (x * 4) / GOB_W;
                const intra_x = (xb / 32) * 256 + ((xb % 32) / 16) * 64 + (xb % 16);
                const fast = y_base + @as(usize, gob_col) * block_bytes + intra_x;
                const ref = nvidia.graphics.blColorPixelOffset(x, y, w);
                if (fast != ref) {
                    if (mismatches == 0) first = .{ x, y };
                    mismatches += 1;
                }
            }
        }
        if (mismatches != 0) std.debug.print("\n[ORACLE-DESWIZZLE-MATH] {d}x{d}: {d} mismatches, first at ({d},{d})\n", .{ w, h, mismatches, first[0], first[1] });
        try std.testing.expectEqual(@as(u32, 0), mismatches);
    }
}

// ORACLE (de-swizzle is not the source of the live-present black "speckle"): render a
// steep-edged triangle (the speckle signature; the live glmark2 `build` model holes
// concentrate at x%8 in {3,4}, the 8-px GPU raster-tile center) into an 800x600
// block-linear color RT. For every black "hole" pixel (black with non-black left+right
// neighbours) prove the black is really in the tiled VRAM by reading the same pixel two
// independent ways: (a) the device's fast de-swizzle (mapResource) and (b) a fresh
// blColorPixelOffset gather from the raw VRAM mapping. Both agree on every hole, so the
// de-swizzle faithfully reports the GPU's own output. The speckle is a GPU rasterization
// coverage gap (open item), not a readback bug. Permanently guards that a future
// "speckle" cannot be reintroduced by the block-linear de-swizzle. Skips without a GPU.
test "ORACLE: the block-linear de-swizzle faithfully reports the NVIDIA GPU output at speckle pixels (no readback-introduced holes) (skips without a GPU)" {
    const gpa = std.testing.allocator;
    const dev = NvDevice.create(gpa) catch return error.SkipZigTest;
    defer dev.deinit();
    const w: u32 = 800;
    const h: u32 = 600;

    var vs_b = try buildVertexSpirv(gpa);
    defer vs_b.deinit(gpa);
    var ps_b = try buildFragmentSpirv(gpa, .{ 0.2, 0.4, 0.9, 1.0 });
    defer ps_b.deinit(gpa);
    const vs_sass = try compileToSass(gpa, std.mem.sliceAsBytes(vs_b.words.items), .vertex);
    defer gpa.free(vs_sass);
    const ps_sass = try compileToSass(gpa, std.mem.sliceAsBytes(ps_b.words.items), .fragment);
    defer gpa.free(ps_sass);
    const rt = try dev.createResource(.{ .image = .{ .width = w, .height = h, .format = .bgra8_unorm, .usage = .{ .render_target = true } } });
    defer dev.destroyResource(rt);
    const vs = try dev.createShaderModule(.{ .stage = .vertex, .code = std.mem.sliceAsBytes(vs_sass) });
    defer dev.destroyShaderModule(vs);
    const ps = try dev.createShaderModule(.{ .stage = .fragment, .code = std.mem.sliceAsBytes(ps_sass) });
    defer dev.destroyShaderModule(ps);
    const vbuf = try dev.createResource(.{ .buffer = .{ .size = 3 * 16, .usage = .{ .vertex = true } } });
    defer dev.destroyResource(vbuf);
    // One big triangle with a long steep (near-vertical) hypotenuse. That is the
    // configuration that produces the most edge speckles (gaps in horizontal runs).
    const verts = [_]f32{
        0.6,  0.9,  0.0, 1.0,
        0.55, 0.9,  0.0, 1.0,
        -0.6, -0.9, 0.0, 1.0,
    };
    @memcpy(std.mem.bytesAsSlice(f32, try dev.mapResource(vbuf))[0..verts.len], &verts);
    const attrs = [_]hal.VertexAttribute{.{ .location = 0, .format = .rgba8_unorm, .offset = 0 }};
    const pipe = try dev.createPipeline(.{ .vertex = vs, .fragment = ps, .vertex_layout = .{ .stride = 16, .attributes = &attrs }, .color_format = .bgra8_unorm });
    defer dev.destroyPipeline(pipe);
    const ctx = try dev.createContext();
    defer ctx.deinit();
    {
        const cb = try ctx.beginCommands();
        defer cb.deinit();
        try cb.setRenderTarget(rt);
        try cb.clear(.{ .r = 0, .g = 0, .b = 0, .a = 1 });
        try cb.bindPipeline(pipe);
        try cb.bindVertexBuffer(vbuf);
        try cb.draw(3, 0);
        try ctx.submit(cb);
    }

    // de-swizzled readback (the device's fast path) ...
    const fb = std.mem.bytesAsSlice(u32, try dev.mapResource(rt));
    // ... vs an independent gather straight from the raw tiled VRAM mapping.
    const r: *@import("resource.zig").Resource = @ptrCast(@alignCast(rt));
    const raw = r.mapping.?.bytes;

    var holes: u32 = 0;
    var disagree: u32 = 0;
    var y: u32 = 1;
    while (y < h - 1) : (y += 1) {
        var x: u32 = 1;
        while (x < w - 1) : (x += 1) {
            const c = fb[y * w + x] & 0x00ffffff;
            const l = fb[y * w + x - 1] & 0x00ffffff;
            const rr = fb[y * w + x + 1] & 0x00ffffff;
            if (c == 0 and l != 0 and rr != 0) {
                holes += 1;
                // independent VRAM gather of the same three pixels
                const off = nvidia.graphics.blColorPixelOffset(x, y, w);
                const offl = nvidia.graphics.blColorPixelOffset(x - 1, y, w);
                const offr = nvidia.graphics.blColorPixelOffset(x + 1, y, w);
                const vc = (@as(u32, raw[off]) | @as(u32, raw[off + 1]) << 8 | @as(u32, raw[off + 2]) << 16) & 0x00ffffff;
                const vl = (@as(u32, raw[offl]) | @as(u32, raw[offl + 1]) << 8 | @as(u32, raw[offl + 2]) << 16) & 0x00ffffff;
                const vr = (@as(u32, raw[offr]) | @as(u32, raw[offr + 1]) << 8 | @as(u32, raw[offr + 2]) << 16) & 0x00ffffff;
                // The de-swizzled readback and the raw VRAM gather agree: both
                // production de-swizzle paths (the fast inline path in device.zig and
                // graphics.blColorPixelOffset) read the same byte for every pixel.
                if (!(vc == c and vl == l and vr == rr)) disagree += 1;
            }
        }
    }
    std.debug.print("\n[ORACLE-DESWIZZLE-FAITHFUL] 800x600 steep-edge holes={d} readback-disagreements={d}\n", .{ holes, disagree });
    try std.testing.expectEqual(@as(u32, 0), disagree);
    // The "speckle" was a de-swizzle GOB-layout bug, not a rasterizer coverage gap:
    // Prism read back the block-linear color RT with the old Fermi 16x2 sector
    // layout, but Blackwell stores >=4-byte color as TuringColor2D (4 rows / 128-B
    // sector group, 64-B-apart sector columns). The Fermi mapping mis-read half the
    // pixels in every GOB. For a uniform interior the permutation is invisible, but
    // at a covered region's steep edge it read an adjacent uncovered (black) byte =
    // the holes that clustered at x%8 in {3,4}. With the correct TuringColor2D layout
    // there are zero holes. Assert it so the swizzle bug cannot be reintroduced.
    try std.testing.expectEqual(@as(u32, 0), holes);
}

// ORACLE (block-linear de-swizzle at a non-tile-aligned size): the live vkcube
// swapchain is 500x500 (500*4 = 2000 bytes/row, not a multiple of the 256-byte
// pitch nor a whole number of 16-px GOBs), which is the first size that exercises
// blColorPixelOffset where the GOB-aligned row stride (512 elem) != the pixel
// width (500). A de-swizzle that mishandles the partial trailing GOB scatters
// wrong pixels into flat regions (the salt-and-pepper "speckle" the live present
// showed). This renders a flat-color triangle at 500x500 and asserts the painted
// interior is perfectly uniform. Any single off-color pixel is the speckle bug.
test "ORACLE: block-linear de-swizzle is speckle-free at a non-tile-aligned 500x500 on the NVIDIA GPU (skips without a GPU)" {
    const gpa = std.testing.allocator;
    const dev = NvDevice.create(gpa) catch return error.SkipZigTest;
    defer dev.deinit();
    const W: u32 = 500;
    const H: u32 = 500;
    const clear = hal.Color{ .r = 0.1, .g = 0.45, .b = 0.48, .a = 1 }; // teal (vkcube-like)
    const tri = [4]f32{ 0.5, 0.2, 0.2, 1.0 }; // a distinct gray-ish color
    const fb = try runTight(gpa, dev, W, H, clear, tri);
    defer gpa.free(fb);

    // Scan the interior for speckles: a pixel that differs from the modal value of
    // its 4-neighbors while that neighborhood is flat. The flat triangle interior +
    // flat clear background are both uniform, so a correct de-swizzle has zero such
    // pixels. (A de-swizzle that gathers the wrong byte offset produces exactly the
    // scattered single-pixel anomalies the live 500x500 present showed.)
    var speckles: u32 = 0;
    var y: u32 = 1;
    while (y < H - 1) : (y += 1) {
        var x: u32 = 1;
        while (x < W - 1) : (x += 1) {
            const c = fb[y * W + x];
            const l = fb[y * W + x - 1];
            const r = fb[y * W + x + 1];
            const u = fb[(y - 1) * W + x];
            const d = fb[(y + 1) * W + x];
            if (l == r and l == u and l == d and c != l) speckles += 1;
        }
    }
    if (speckles != 0) std.debug.print("\n[ORACLE-DESWIZZLE] {d} speckles at 500x500\n", .{speckles});
    try std.testing.expectEqual(@as(u32, 0), speckles);
}

// GLES TEXTURE ORACLE (the EGL/GLES textured-quad path on the real GPU): the same
// GLSL shaders tools/egl-texture.zig drives (a `uniform sampler2D` + texture2D over a
// full quad) are compiled through the GLSL front end -> SPIR-V -> SASS, a 2x2 RGBA8
// checkerboard is uploaded as a sampled texture, and the quad is drawn through the
// nvidia HAL. Each quadrant reads its texel in the software-matching orientation
// (NDC -1 -> framebuffer row 0): TL=red, TR=green, BL=blue, BR=white. Permanent
// frame-verified guard that (a) the nvidia texture path samples correctly from the GLES
// API and (b) the nvidia viewport Y-origin matches the software driver. A vertical flip
// would swap the top/bottom quadrant rows. That was the bug the positive-Y-scale
// setViewport fixed. Skips without a GPU.
test "ORACLE: GLES textured quad samples the 2x2 checkerboard in software-matching orientation on the NVIDIA GPU (skips without a GPU)" {
    const glsl = @import("../../glsl.zig");
    const gpa = std.testing.allocator;
    const dev = NvDevice.create(gpa) catch return error.SkipZigTest;
    defer dev.deinit();
    const W: u32 = 64;
    const H: u32 = 64;

    // The exact GLSL the EGL texture tool uses (passthrough-uv VS + texture2D FS).
    const vs_src =
        \\attribute vec2 position;
        \\attribute vec2 aUV;
        \\varying vec2 vUV;
        \\void main() { gl_Position = vec4(position, 0.0, 1.0); vUV = aUV; }
    ;
    const fs_src =
        \\precision mediump float;
        \\uniform sampler2D uTex;
        \\varying vec2 vUV;
        \\void main() { gl_FragColor = texture2D(uTex, vUV); }
    ;
    const vs_bytes = try glsl.compileForStage(gpa, vs_src, .vertex);
    defer gpa.free(vs_bytes);
    const fs_bytes = try glsl.compileForStage(gpa, fs_src, .fragment);
    defer gpa.free(fs_bytes);

    const rt = try dev.createResource(.{ .image = .{ .width = W, .height = H, .format = .rgba8_unorm, .usage = .{ .render_target = true } } });
    defer dev.destroyResource(rt);
    const vs = try dev.createShaderModule(.{ .stage = .vertex, .code = vs_bytes });
    defer dev.destroyShaderModule(vs);
    const fs = try dev.createShaderModule(.{ .stage = .fragment, .code = fs_bytes });
    defer dev.destroyShaderModule(fs);

    // 2x2 RGBA8 checkerboard: red, green / blue, white (= the egl-texture checker).
    const tex = try dev.createResource(.{ .image = .{ .width = 2, .height = 2, .format = .rgba8_unorm, .usage = .{ .sampled = true } } });
    defer dev.destroyResource(tex);
    {
        const checker = [_]u8{
            255, 0, 0,   255, 0,   255, 0,   255,
            0,   0, 255, 255, 255, 255, 255, 255,
        };
        const dst = try dev.mapResource(tex);
        @memcpy(dst[0..checker.len], &checker);
    }

    // Full-screen quad: position (clip xy) + UV. v=0 at clip y=-1.
    const Vtx = extern struct { x: f32, y: f32, u: f32, v: f32 };
    const quad = [6]Vtx{
        .{ .x = -1, .y = -1, .u = 0, .v = 0 },
        .{ .x = 1, .y = -1, .u = 1, .v = 0 },
        .{ .x = 1, .y = 1, .u = 1, .v = 1 },
        .{ .x = -1, .y = -1, .u = 0, .v = 0 },
        .{ .x = 1, .y = 1, .u = 1, .v = 1 },
        .{ .x = -1, .y = 1, .u = 0, .v = 1 },
    };
    const vbuf = try dev.createResource(.{ .buffer = .{ .size = @sizeOf(@TypeOf(quad)), .usage = .{ .vertex = true } } });
    defer dev.destroyResource(vbuf);
    @memcpy(std.mem.bytesAsSlice(Vtx, try dev.mapResource(vbuf))[0..quad.len], &quad);

    const attrs = [_]hal.VertexAttribute{
        .{ .location = 0, .format = .r32g32_float, .offset = 0 }, // position
        .{ .location = 1, .format = .r32g32_float, .offset = 8 }, // uv
    };
    const pipe = try dev.createPipeline(.{
        .vertex = vs,
        .fragment = fs,
        .vertex_layout = .{ .stride = @sizeOf(Vtx), .attributes = &attrs },
        .color_format = .rgba8_unorm,
    });
    defer dev.destroyPipeline(pipe);

    const ctx = try dev.createContext();
    defer ctx.deinit();
    {
        const cb = try ctx.beginCommands();
        defer cb.deinit();
        try cb.setRenderTarget(rt);
        try cb.clear(.{ .r = 0, .g = 0, .b = 0, .a = 1 });
        try cb.bindPipeline(pipe);
        try cb.bindTexture(.{ .binding = 2, .image = tex, .filter = .nearest, .address_u = .clamp_to_edge, .address_v = .clamp_to_edge });
        try cb.bindVertexBuffer(vbuf);
        try cb.draw(6, 0);
        try ctx.submit(cb);
    }

    const fb = std.mem.bytesAsSlice(u32, try dev.mapResource(rt));
    const at = struct {
        fn p(buf: []align(1) const u32, x: usize, y: usize) [3]u8 {
            const v = buf[y * W + x];
            return .{ @truncate(v), @truncate(v >> 8), @truncate(v >> 16) };
        }
    }.p;
    // v=0 -> row 0 (top), software-matching: TL=red, TR=green, BL=blue, BR=white.
    const tl = at(fb, 16, 16);
    const tr = at(fb, 48, 16);
    const bl = at(fb, 16, 48);
    const br = at(fb, 48, 48);
    try std.testing.expect(tl[0] > 200 and tl[1] < 60 and tl[2] < 60); // red
    try std.testing.expect(tr[1] > 200 and tr[0] < 60 and tr[2] < 60); // green
    try std.testing.expect(bl[2] > 200 and bl[0] < 60 and bl[1] < 60); // blue
    try std.testing.expect(br[0] > 200 and br[1] > 200 and br[2] > 200); // white
}

// A sampler2DShadow FS lowers to a depth-compare TEX: OpTex (0xd61) with z_cmpr set
// (bit 78) and a scalar (R-only, channel_mask == 1) result. Offline check (no GPU) guards the
// nvidia isel's texShadow encoding + the whole GLSL sampler2DShadow -> OpImageSampleDref -> sampler_
// shadow_fn tag -> texShadow chain, independent of hardware. Pairs with ORACLE-SHADOW (the GPU render).
test "ORACLE-SHADOW-SASS: a sampler2DShadow FS compiles to a TEX with z_cmpr (bit 78) + an R-only scalar result (no GPU)" {
    const glsl = @import("../../glsl.zig");
    const gpa = std.testing.allocator;
    const fs_src =
        \\precision mediump float;
        \\uniform sampler2DShadow uShadow;
        \\void main() { float s = texture(uShadow, vec3(0.5, 0.5, 0.2)); gl_FragColor = vec4(s, s, s, 1.0); }
    ;
    const fs_bytes = try glsl.compileForStage(gpa, fs_src, .fragment);
    defer gpa.free(fs_bytes);
    const sass = try compileToSass(gpa, fs_bytes, .fragment);
    defer gpa.free(sass);
    var found = false;
    var i: usize = 0;
    while (i + 4 <= sass.len) : (i += 4) {
        if (sass[i] & 0xfff == 0xd61) { // OpTex (bindless)
            const zcmpr = (sass[i + 2] >> (78 - 64)) & 1; // z_cmpr @ bit 78
            const chmask = (sass[i + 2] >> (72 - 64)) & 0xf; // channel_mask @ 72..76
            const src0 = (sass[i] >> 24) & 0xff; // coord base @ 24..32
            const src1 = (sass[i + 1] >> 0) & 0xff; // handle/src1 base @ 32..40
            try std.testing.expectEqual(@as(u32, 1), zcmpr); // depth-compare enabled
            try std.testing.expectEqual(@as(u32, 1), chmask); // scalar (R only)
            // A 2D z_cmpr TEX reads a 2-register coord pair (u, v) at src0 and a 2-register
            // [handle, dref] pair at src1 = coord+2. Both power-of-2 vector operands need to be
            // even-aligned or the SM faults Xid 13 "Misaligned Register".
            try std.testing.expectEqual(@as(u32, 0), src0 % 2); // coord pair even-aligned
            try std.testing.expectEqual(src0 + 2, src1); // src1 = coord+2 (even)
            found = true;
        }
    }
    try std.testing.expect(found); // a shadow sample becomes a z_cmpr TEX
}

// samplerCubeShadow lowers to a z_cmpr TEX over the 6-face atlas: OpTex (0xd61) with
// z_cmpr (bit 78), an R-only scalar (channel_mask == 1) result, and DIM = 2D (=1, the atlas is sampled
// as 2D, mirroring the non-shadow samplerCube path). Offline guard for the isel's cube-shadow lowering.
test "ORACLE-CUBE-SHADOW-SASS: a samplerCubeShadow FS compiles to a 2D z_cmpr TEX (bit 78) + R-only scalar (no GPU)" {
    const glsl = @import("../../glsl.zig");
    const gpa = std.testing.allocator;
    const fs_src =
        \\precision mediump float;
        \\uniform samplerCubeShadow uShadow;
        \\void main() { float s = texture(uShadow, vec4(0.0, 0.0, 1.0, 0.2)); gl_FragColor = vec4(s, s, s, 1.0); }
    ;
    const fs_bytes = try glsl.compileForStage(gpa, fs_src, .fragment);
    defer gpa.free(fs_bytes);
    const sass = try compileToSass(gpa, fs_bytes, .fragment);
    defer gpa.free(sass);
    var found = false;
    var i: usize = 0;
    while (i + 4 <= sass.len) : (i += 4) {
        if (sass[i] & 0xfff == 0xd61) { // OpTex (bindless)
            const zcmpr = (sass[i + 2] >> (78 - 64)) & 1; // z_cmpr @ bit 78
            const chmask = (sass[i + 2] >> (72 - 64)) & 0xf; // channel_mask @ 72..76
            const dim = (sass[i + 1] >> (61 - 32)) & 0x7; // dim @ 61..64
            const src0 = (sass[i] >> 24) & 0xff; // coord base @ 24..32
            const src1 = (sass[i + 1] >> 0) & 0xff; // handle/src1 base @ 32..40
            try std.testing.expectEqual(@as(u32, 1), zcmpr);
            try std.testing.expectEqual(@as(u32, 1), chmask);
            try std.testing.expectEqual(@as(u32, 1), dim); // 2D atlas
            // 2D atlas coord pair (u', v); src1 = [handle, dref] at coord+2 (even). Both the
            // coord pair and the [handle, dref] pair are 2-register operands, so coord needs to be
            // even-aligned. An odd base faults the SM Xid 13 "Misaligned Register" (the exact
            // samplerCubeShadow full-suite wall this guards against).
            try std.testing.expectEqual(@as(u32, 0), src0 % 2); // coord pair even-aligned
            try std.testing.expectEqual(src0 + 2, src1); // src1 = coord+2
            try std.testing.expectEqual(@as(u32, 0), src1 % 2); // even (2-reg src1 pair)
            found = true;
        }
    }
    try std.testing.expect(found);
}

// sampler2DArrayShadow lowers to a native TWO_D_ARRAY z_cmpr TEX: OpTex (0xd61) with
// z_cmpr (bit 78), an R-only scalar result, and DIM = Array2D (=5). Offline guard for the array-shadow
// lowering (the layer-first integer coord + [handle, dref] src1).
test "ORACLE-ARRAY-SHADOW-SASS: a sampler2DArrayShadow FS compiles to an Array2D z_cmpr TEX (bit 78) + R-only scalar (no GPU)" {
    const glsl = @import("../../glsl.zig");
    const gpa = std.testing.allocator;
    const fs_src =
        \\precision mediump float;
        \\uniform sampler2DArrayShadow uShadow;
        \\void main() { float s = texture(uShadow, vec4(0.5, 0.5, 1.0, 0.2)); gl_FragColor = vec4(s, s, s, 1.0); }
    ;
    const fs_bytes = try glsl.compileForStage(gpa, fs_src, .fragment);
    defer gpa.free(fs_bytes);
    const sass = try compileToSass(gpa, fs_bytes, .fragment);
    defer gpa.free(sass);
    var found = false;
    var i: usize = 0;
    while (i + 4 <= sass.len) : (i += 4) {
        if (sass[i] & 0xfff == 0xd61) { // OpTex (bindless)
            const zcmpr = (sass[i + 2] >> (78 - 64)) & 1; // z_cmpr @ bit 78
            const chmask = (sass[i + 2] >> (72 - 64)) & 0xf; // channel_mask @ 72..76
            const dim = (sass[i + 1] >> (61 - 32)) & 0x7; // dim @ 61..64
            const src0 = (sass[i] >> 24) & 0xff; // coord base @ 24..32
            const src1 = (sass[i + 1] >> 0) & 0xff; // handle/src1 base @ 32..40
            try std.testing.expectEqual(@as(u32, 1), zcmpr);
            try std.testing.expectEqual(@as(u32, 1), chmask);
            try std.testing.expectEqual(@as(u32, 5), dim); // Array2D
            // A 3-register array coord (layer, u, v) needs 4-alignment (else Xid 13 "Misaligned
            // Register"). src1 = [handle, dref] follows at coord+4 (even). Guard both alignments.
            try std.testing.expectEqual(@as(u32, 0), src0 % 4); // coord 4-aligned
            try std.testing.expectEqual(src0 + 4, src1); // src1 = coord+4 (even)
            found = true;
        }
    }
    try std.testing.expect(found);
}

// sampler2DShadow depth compare on the real GPU: bind a 1x1 texture whose R channel = 128/255 ~ 0.5 as
// the "stored depth" with compare_enable + LEQUAL, and sample it with two references. The FS
// texture(uShadow, vec3(0.5, 0.5, ref)) returns a scalar: ref 0.2 <= 0.5 passes -> lit (white), ref 0.8 >
// 0.5 fails -> shadow (black). GPU counterpart of the software vendor.zig sampler2DShadow oracle, proving
// the whole chain end to end on hardware: GLSL sampler2DShadow -> OpImageSampleDref -> the isel's z_cmpr
// TEX + the TSC DEPTH_COMPARE fields. Skips without a GPU.
test "ORACLE-SHADOW: a sampler2DShadow depth-compare TEX lights ref<=depth and shadows ref>depth on the NVIDIA GPU (skips without a GPU)" {
    const glsl = @import("../../glsl.zig");
    const gpa = std.testing.allocator;
    const dev = NvDevice.create(gpa) catch return error.SkipZigTest;
    defer dev.deinit();
    const W: u32 = 64;
    const H: u32 = 64;

    const vs_src =
        \\attribute vec2 position;
        \\void main() { gl_Position = vec4(position, 0.0, 1.0); }
    ;
    // Two FS variants baking the compare reference: 0.2 (lit) and 0.8 (shadow). Constant coords (the
    // texture is 1x1) so no varying / vertex UV is needed.
    const fs_lit_src =
        \\precision mediump float;
        \\uniform sampler2DShadow uShadow;
        \\void main() { float s = texture(uShadow, vec3(0.5, 0.5, 0.2)); gl_FragColor = vec4(s, s, s, 1.0); }
    ;
    const fs_shadow_src =
        \\precision mediump float;
        \\uniform sampler2DShadow uShadow;
        \\void main() { float s = texture(uShadow, vec3(0.5, 0.5, 0.8)); gl_FragColor = vec4(s, s, s, 1.0); }
    ;
    const vs_bytes = try glsl.compileForStage(gpa, vs_src, .vertex);
    defer gpa.free(vs_bytes);
    const fs_lit_bytes = try glsl.compileForStage(gpa, fs_lit_src, .fragment);
    defer gpa.free(fs_lit_bytes);
    const fs_shadow_bytes = try glsl.compileForStage(gpa, fs_shadow_src, .fragment);
    defer gpa.free(fs_shadow_bytes);

    const rt = try dev.createResource(.{ .image = .{ .width = W, .height = H, .format = .rgba8_unorm, .usage = .{ .render_target = true } } });
    defer dev.destroyResource(rt);
    const vs = try dev.createShaderModule(.{ .stage = .vertex, .code = vs_bytes });
    defer dev.destroyShaderModule(vs);
    const fs_lit = try dev.createShaderModule(.{ .stage = .fragment, .code = fs_lit_bytes });
    defer dev.destroyShaderModule(fs_lit);
    const fs_shadow = try dev.createShaderModule(.{ .stage = .fragment, .code = fs_shadow_bytes });
    defer dev.destroyShaderModule(fs_shadow);

    // 1x1 depth-format (ZF32) sampled texture with the stored depth = 0.5. A depth32_float image
    // requested sampled-only builds a ZF32 sampled TIC (COMPONENTS=ZF32, DATA_TYPE=FLOAT, swizzle
    // R001) over plain block-linear memory. That is the format the HW DEPTH_COMPARE engages on (a
    // color texture returns the raw texel and the compare is a no-op). The staging is one 32-bit float.
    const tex = try dev.createResource(.{ .image = .{ .width = 1, .height = 1, .format = .depth32_float, .usage = .{ .sampled = true } } });
    defer dev.destroyResource(tex);
    {
        const depth: f32 = 0.5;
        const dst = try dev.mapResource(tex);
        @memcpy(dst[0..4], std.mem.asBytes(&depth));
    }

    const quad = [6][2]f32{ .{ -1, -1 }, .{ 1, -1 }, .{ 1, 1 }, .{ -1, -1 }, .{ 1, 1 }, .{ -1, 1 } };
    const vbuf = try dev.createResource(.{ .buffer = .{ .size = @sizeOf(@TypeOf(quad)), .usage = .{ .vertex = true } } });
    defer dev.destroyResource(vbuf);
    @memcpy(std.mem.bytesAsSlice([2]f32, try dev.mapResource(vbuf))[0..quad.len], &quad);

    const attrs = [_]hal.VertexAttribute{.{ .location = 0, .format = .r32g32_float, .offset = 0 }};
    const pipe_lit = try dev.createPipeline(.{ .vertex = vs, .fragment = fs_lit, .vertex_layout = .{ .stride = 8, .attributes = &attrs }, .color_format = .rgba8_unorm });
    defer dev.destroyPipeline(pipe_lit);
    const pipe_shadow = try dev.createPipeline(.{ .vertex = vs, .fragment = fs_shadow, .vertex_layout = .{ .stride = 8, .attributes = &attrs }, .color_format = .rgba8_unorm });
    defer dev.destroyPipeline(pipe_shadow);

    const ctx = try dev.createContext();
    defer ctx.deinit();
    const at = struct {
        fn r(buf: []align(1) const u32, x: usize, y: usize) u8 {
            return @truncate(buf[y * W + x]);
        }
    }.r;

    // Draw 1: ref 0.2 <= 0.5 -> lit (white). The bound texture carries compare_enable + LEQUAL.
    {
        const cb = try ctx.beginCommands();
        defer cb.deinit();
        try cb.setRenderTarget(rt);
        try cb.clear(.{ .r = 0, .g = 0, .b = 0, .a = 1 });
        try cb.bindPipeline(pipe_lit);
        try cb.bindTexture(.{ .binding = 2, .image = tex, .filter = .nearest, .address_u = .clamp_to_edge, .address_v = .clamp_to_edge, .compare_enable = true, .compare_op = .less_or_equal });
        try cb.bindVertexBuffer(vbuf);
        try cb.draw(6, 0);
        try ctx.submit(cb);
    }
    const lit_v = at(std.mem.bytesAsSlice(u32, try dev.mapResource(rt)), 32, 32);

    // Draw 2: ref 0.8 > 0.5 -> shadow (black).
    {
        const cb = try ctx.beginCommands();
        defer cb.deinit();
        try cb.setRenderTarget(rt);
        try cb.clear(.{ .r = 0, .g = 0, .b = 0, .a = 1 });
        try cb.bindPipeline(pipe_shadow);
        try cb.bindTexture(.{ .binding = 2, .image = tex, .filter = .nearest, .address_u = .clamp_to_edge, .address_v = .clamp_to_edge, .compare_enable = true, .compare_op = .less_or_equal });
        try cb.bindVertexBuffer(vbuf);
        try cb.draw(6, 0);
        try ctx.submit(cb);
    }
    const shadow_v = at(std.mem.bytesAsSlice(u32, try dev.mapResource(rt)), 32, 32);

    // The HW sampler DEPTH_COMPARE only engages on a depth-format sampled texture (GL_TEXTURE_
    // COMPARE_MODE is undefined for color formats; NVIDIA HW then returns the raw texel and the
    // compare is a no-op). The ZF32 sampled texture built above (COMPONENTS=ZF32, DATA_TYPE=FLOAT,
    // R001 swizzle) is that depth format, so the z_cmpr TEX (bit 78) + the TSC DEPTH_COMPARE/FUNC
    // engage: texture(uShadow, vec3(u,v,ref)) returns 1.0 (lit) when ref <= stored depth, else 0.0
    // (shadow). We assert the differential when it engages. If a driver/HW config leaves the
    // compare a raw-texel passthrough (~128 for both, depth 0.5 -> 0.5*255), fall back to a skip so
    // the offline ORACLE-SHADOW-SASS still guards the encoding. The compare engaging here proves the
    // whole chain: GLSL sampler2DShadow -> ZF32 sampled TIC -> z_cmpr TEX + TSC on the real GPU.
    const compare_engaged = lit_v > 200 and shadow_v < 60;
    if (!compare_engaged) {
        // Sanity: the render ran and sampled the texel (not a crash / cleared black).
        try std.testing.expect(lit_v > 100 and shadow_v > 100); // raw-texel passthrough (~128)
        return error.SkipZigTest;
    }
    try std.testing.expect(lit_v > 200); // ref 0.2 <= depth 0.5 -> lit
    try std.testing.expect(shadow_v < 60); // ref 0.8 > depth 0.5 -> shadow
}

// The real two-pass shadow map on the GPU (ORACLE-SHADOW-2PASS): unlike ORACLE-SHADOW (which uploads a
// ZF32 depth value), the depth here comes from an actual GPU depth render. PASS 1 renders a full-screen
// quad at window depth 0.5 into a ZETA depth surface (the fixed-function depth test writes it). The
// rendered ZETA is block-linear (GOB-tiled, block height 16, ZF32 = 4 bytes/px, generic PTE kind), same
// tiling as a block-linear color RT. The existing copy-engine (CE) detile copies it into a pitch-linear
// f32 buffer on the GPU. Those rendered depths are staged into a ZF32 sampled texture (the format the HW
// DEPTH_COMPARE engages on). PASS 2 binds that rendered depth texture as a sampler2DShadow with
// COMPARE_REF_TO_TEXTURE + LEQUAL and samples it: ref 0.2 <= 0.5 -> lit (v>200), ref 0.8 > 0.5 ->
// shadow (v<60). Proves the whole shadow-map loop (render depth from the light -> copy -> shadow-lookup)
// on real hardware, mirroring the software "real depth-texture shadow map" oracle
// (finalizeDepthTexture there == the ZETA->CE-detile->ZF32-staging here). Asserts unconditionally.
// Skips only if there is no GPU / no CE.
test "ORACLE-SHADOW-2PASS: PASS 1 RENDERS depth into a ZETA, the CE copies it into a ZF32 sampled texture, PASS 2 shadow-samples the RENDERED depth on the NVIDIA GPU (skips without a GPU)" {
    const glsl = @import("../../glsl.zig");
    const CopyEngine = @import("ce.zig").CopyEngine;
    const Resource = @import("resource.zig").Resource;
    const gpa = std.testing.allocator;
    const dev = NvDevice.create(gpa) catch return error.SkipZigTest;
    defer dev.deinit();
    const nv: *NvDevice = @ptrCast(@alignCast(dev.ptr));
    // 64x64: width*4 = 256 is GOB-row-aligned, so the CE detile lands tightly packed and the ZF32
    // staging (w*h*4) is a direct memcpy of the detiled f32 depths.
    const W: u32 = 64;
    const H: u32 = 64;

    // --- PASS 1 programs: a VS that plants gl_Position.z = 0.5 (window depth 0.5 in Prism's [0,1]
    // clip), and a white FS (the color target is a throwaway; only the depth matters). ---
    const depth_vs_src =
        \\attribute vec2 position;
        \\void main() { gl_Position = vec4(position, 0.5, 1.0); }
    ;
    const white_fs_src =
        \\precision mediump float;
        \\void main() { gl_FragColor = vec4(1.0); }
    ;
    // --- PASS 2 programs: the sampler2DShadow lookup, baking the compare reference (0.2 lit / 0.8
    // shadow). Constant coords (the depth is uniform 0.5 across the surface). ---
    const shadow_vs_src =
        \\attribute vec2 position;
        \\void main() { gl_Position = vec4(position, 0.0, 1.0); }
    ;
    const fs_lit_src =
        \\precision mediump float;
        \\uniform sampler2DShadow uShadow;
        \\void main() { float s = texture(uShadow, vec3(0.5, 0.5, 0.2)); gl_FragColor = vec4(s, s, s, 1.0); }
    ;
    const fs_shadow_src =
        \\precision mediump float;
        \\uniform sampler2DShadow uShadow;
        \\void main() { float s = texture(uShadow, vec3(0.5, 0.5, 0.8)); gl_FragColor = vec4(s, s, s, 1.0); }
    ;
    const depth_vs_bytes = try glsl.compileForStage(gpa, depth_vs_src, .vertex);
    defer gpa.free(depth_vs_bytes);
    const white_fs_bytes = try glsl.compileForStage(gpa, white_fs_src, .fragment);
    defer gpa.free(white_fs_bytes);
    const shadow_vs_bytes = try glsl.compileForStage(gpa, shadow_vs_src, .vertex);
    defer gpa.free(shadow_vs_bytes);
    const fs_lit_bytes = try glsl.compileForStage(gpa, fs_lit_src, .fragment);
    defer gpa.free(fs_lit_bytes);
    const fs_shadow_bytes = try glsl.compileForStage(gpa, fs_shadow_src, .fragment);
    defer gpa.free(fs_shadow_bytes);

    const depth_vs = try dev.createShaderModule(.{ .stage = .vertex, .code = depth_vs_bytes });
    defer dev.destroyShaderModule(depth_vs);
    const white_fs = try dev.createShaderModule(.{ .stage = .fragment, .code = white_fs_bytes });
    defer dev.destroyShaderModule(white_fs);
    const shadow_vs = try dev.createShaderModule(.{ .stage = .vertex, .code = shadow_vs_bytes });
    defer dev.destroyShaderModule(shadow_vs);
    const fs_lit = try dev.createShaderModule(.{ .stage = .fragment, .code = fs_lit_bytes });
    defer dev.destroyShaderModule(fs_lit);
    const fs_shadow = try dev.createShaderModule(.{ .stage = .fragment, .code = fs_shadow_bytes });
    defer dev.destroyShaderModule(fs_shadow);

    // A full-screen quad, shared by both passes.
    const quad = [6][2]f32{ .{ -1, -1 }, .{ 1, -1 }, .{ 1, 1 }, .{ -1, -1 }, .{ 1, 1 }, .{ -1, 1 } };
    const vbuf = try dev.createResource(.{ .buffer = .{ .size = @sizeOf(@TypeOf(quad)), .usage = .{ .vertex = true } } });
    defer dev.destroyResource(vbuf);
    @memcpy(std.mem.bytesAsSlice([2]f32, try dev.mapResource(vbuf))[0..quad.len], &quad);
    const attrs = [_]hal.VertexAttribute{.{ .location = 0, .format = .r32g32_float, .offset = 0 }};

    // The color RT for PASS 1 (throwaway) and the ZETA depth surface it renders into. A depth32_float
    // with an EMPTY usage routes to the ZETA render-surface path (not the sampled ZF32 texture).
    const rt1 = try dev.createResource(.{ .image = .{ .width = W, .height = H, .format = .rgba8_unorm, .usage = .{ .render_target = true } } });
    defer dev.destroyResource(rt1);
    const zeta = try dev.createResource(.{ .image = .{ .width = W, .height = H, .format = .depth32_float, .usage = .{} } });
    defer dev.destroyResource(zeta);

    // Depth-tested PASS 1 pipeline: test + write enabled, LESS. Clear depth to far (1.0), the quad at
    // z=0.5 passes and writes 0.5 everywhere it covers (the whole surface).
    const pipe_depth = try dev.createPipeline(.{ .vertex = depth_vs, .fragment = white_fs, .vertex_layout = .{ .stride = 8, .attributes = &attrs }, .color_format = .rgba8_unorm, .depth = .{ .test_enable = true, .write_enable = true, .compare_op = .less } });
    defer dev.destroyPipeline(pipe_depth);

    const ctx = try dev.createContext();
    defer ctx.deinit();

    // PASS 1: render window-depth 0.5 into the ZETA on the GPU.
    {
        const cb = try ctx.beginCommands();
        defer cb.deinit();
        try cb.setRenderTarget(rt1);
        try cb.clear(.{ .r = 0, .g = 0, .b = 0, .a = 1 });
        try cb.setDepthTarget(zeta, 1.0);
        try cb.bindPipeline(pipe_depth);
        try cb.bindVertexBuffer(vbuf);
        try cb.draw(6, 0);
        try ctx.submit(cb);
    }

    // COPY: detile the block-linear ZETA (ZF32, 4 B/px, block height 16, same tiling as a
    // color RT) into a pitch-linear f32 buffer on the GPU via the copy engine, then stage those
    // rendered depths into a ZF32 sampled texture (the format the HW DEPTH_COMPARE engages on).
    const zeta_res: *Resource = @ptrCast(@alignCast(zeta));
    const dst_pitch: u32 = W * 4;
    const dst = try nv.allocGpu(.system, @as(u64, dst_pitch) * H);
    defer nv.freeGpu(dst);
    @memset(dst.bytes[0 .. @as(usize, dst_pitch) * H], 0);
    const ce = CopyEngine.create(nv) catch return error.SkipZigTest;
    defer ce.deinit();
    try ce.detile(zeta_res, dst, dst_pitch);

    // The rendered depth (0.5) now lives row-major in dst. Sanity: the center depth is ~0.5.
    {
        const depths = std.mem.bytesAsSlice(f32, @as([]align(@alignOf(f32)) const u8, @alignCast(dst.bytes[0 .. @as(usize, dst_pitch) * H])));
        const center = depths[(H / 2) * W + W / 2];
        try std.testing.expect(center > 0.4 and center < 0.6); // the GPU depth render landed 0.5
    }

    const tex = try dev.createResource(.{ .image = .{ .width = W, .height = H, .format = .depth32_float, .usage = .{ .sampled = true } } });
    defer dev.destroyResource(tex);
    {
        const staging = try dev.mapResource(tex); // a sampled texture hands back its linear staging
        @memcpy(staging[0 .. @as(usize, W) * H * 4], dst.bytes[0 .. @as(usize, W) * H * 4]);
    }

    // PASS 2 pipelines: the sampler2DShadow lookup against the rendered depth texture.
    const pipe_lit = try dev.createPipeline(.{ .vertex = shadow_vs, .fragment = fs_lit, .vertex_layout = .{ .stride = 8, .attributes = &attrs }, .color_format = .rgba8_unorm });
    defer dev.destroyPipeline(pipe_lit);
    const pipe_shadow = try dev.createPipeline(.{ .vertex = shadow_vs, .fragment = fs_shadow, .vertex_layout = .{ .stride = 8, .attributes = &attrs }, .color_format = .rgba8_unorm });
    defer dev.destroyPipeline(pipe_shadow);

    const rt2 = try dev.createResource(.{ .image = .{ .width = W, .height = H, .format = .rgba8_unorm, .usage = .{ .render_target = true } } });
    defer dev.destroyResource(rt2);
    const at = struct {
        fn r(buf: []align(1) const u32, x: usize, y: usize) u8 {
            return @truncate(buf[y * W + x]);
        }
    }.r;

    // Draw 1: ref 0.2 <= rendered 0.5 -> lit (white).
    {
        const cb = try ctx.beginCommands();
        defer cb.deinit();
        try cb.setRenderTarget(rt2);
        try cb.clear(.{ .r = 0, .g = 0, .b = 0, .a = 1 });
        try cb.bindPipeline(pipe_lit);
        try cb.bindTexture(.{ .binding = 2, .image = tex, .filter = .nearest, .address_u = .clamp_to_edge, .address_v = .clamp_to_edge, .compare_enable = true, .compare_op = .less_or_equal });
        try cb.bindVertexBuffer(vbuf);
        try cb.draw(6, 0);
        try ctx.submit(cb);
    }
    const lit_v = at(std.mem.bytesAsSlice(u32, try dev.mapResource(rt2)), 32, 32);

    // Draw 2: ref 0.8 > rendered 0.5 -> shadow (black).
    {
        const cb = try ctx.beginCommands();
        defer cb.deinit();
        try cb.setRenderTarget(rt2);
        try cb.clear(.{ .r = 0, .g = 0, .b = 0, .a = 1 });
        try cb.bindPipeline(pipe_shadow);
        try cb.bindTexture(.{ .binding = 2, .image = tex, .filter = .nearest, .address_u = .clamp_to_edge, .address_v = .clamp_to_edge, .compare_enable = true, .compare_op = .less_or_equal });
        try cb.bindVertexBuffer(vbuf);
        try cb.draw(6, 0);
        try ctx.submit(cb);
    }
    const shadow_v = at(std.mem.bytesAsSlice(u32, try dev.mapResource(rt2)), 32, 32);

    // Assert unconditionally: the compare is proven to engage on a ZF32 sampled texture (ORACLE-SHADOW),
    // so a garbage copy or a non-engaging compare fails here. A green run proves the full two-pass
    // GPU shadow-map loop (render depth -> CE copy -> shadow-lookup) on real hardware.
    try std.testing.expect(lit_v > 200); // ref 0.2 <= rendered depth 0.5 -> lit
    try std.testing.expect(shadow_v < 60); // ref 0.8 > rendered depth 0.5 -> shadow
}

// samplerCubeShadow depth compare on the real GPU: a depth32_float CUBE (6 ZF32 faces, each stored
// depth ~0.5) sampled by a direction with GL_LEQUAL. The FS texture(uShadow, vec4(dir, ref)) returns a
// scalar: ref 0.2 <= 0.5 -> lit (white), ref 0.8 > 0.5 -> shadow (black). The nvidia isel reuses the
// samplerCube atlas major-axis lowering (a 2D sample of the 6-face-wide ZF32 atlas) with the z_cmpr bit
// set + the TSC DEPTH_COMPARE fields. Point the direction at +Z (face 4). Unlike ORACLE-SHADOW (2D)
// this asserts unconditionally. A non-engaging compare (raw-texel passthrough ~128) fails the assert,
// so a green run proves the cube depth compare engaged on hardware. Skips only if there is no GPU.
test "ORACLE-CUBE-SHADOW: a samplerCubeShadow depth-compare TEX lights ref<=depth and shadows ref>depth on the NVIDIA GPU (skips without a GPU)" {
    const glsl = @import("../../glsl.zig");
    const gpa = std.testing.allocator;
    const dev = NvDevice.create(gpa) catch return error.SkipZigTest;
    defer dev.deinit();
    const W: u32 = 64;
    const H: u32 = 64;

    const vs_src =
        \\attribute vec2 position;
        \\void main() { gl_Position = vec4(position, 0.0, 1.0); }
    ;
    // Constant +Z direction (face 4) + baked compare reference: 0.2 (lit) and 0.8 (shadow).
    const fs_lit_src =
        \\precision mediump float;
        \\uniform samplerCubeShadow uShadow;
        \\void main() { float s = texture(uShadow, vec4(0.0, 0.0, 1.0, 0.2)); gl_FragColor = vec4(s, s, s, 1.0); }
    ;
    const fs_shadow_src =
        \\precision mediump float;
        \\uniform samplerCubeShadow uShadow;
        \\void main() { float s = texture(uShadow, vec4(0.0, 0.0, 1.0, 0.8)); gl_FragColor = vec4(s, s, s, 1.0); }
    ;
    const vs_bytes = try glsl.compileForStage(gpa, vs_src, .vertex);
    defer gpa.free(vs_bytes);
    const fs_lit_bytes = try glsl.compileForStage(gpa, fs_lit_src, .fragment);
    defer gpa.free(fs_lit_bytes);
    const fs_shadow_bytes = try glsl.compileForStage(gpa, fs_shadow_src, .fragment);
    defer gpa.free(fs_shadow_bytes);

    const rt = try dev.createResource(.{ .image = .{ .width = W, .height = H, .format = .rgba8_unorm, .usage = .{ .render_target = true } } });
    defer dev.destroyResource(rt);
    const vs = try dev.createShaderModule(.{ .stage = .vertex, .code = vs_bytes });
    defer dev.destroyShaderModule(vs);
    const fs_lit = try dev.createShaderModule(.{ .stage = .fragment, .code = fs_lit_bytes });
    defer dev.destroyShaderModule(fs_lit);
    const fs_shadow = try dev.createShaderModule(.{ .stage = .fragment, .code = fs_shadow_bytes });
    defer dev.destroyShaderModule(fs_shadow);

    // A 1x1-face depth32_float CUBE requested sampled-only builds a 6-face-wide ZF32 sampled TIC (the
    // depth format the HW DEPTH_COMPARE engages on). The staging is 6 f32 faces, each = 0.5.
    const tex = try dev.createResource(.{ .image = .{ .width = 1, .height = 1, .format = .depth32_float, .cube = true, .usage = .{ .sampled = true } } });
    defer dev.destroyResource(tex);
    {
        const dst = try dev.mapResource(tex);
        var f: usize = 0;
        while (f < 6) : (f += 1) {
            const depth: f32 = 0.5;
            @memcpy(dst[f * 4 ..][0..4], std.mem.asBytes(&depth));
        }
    }

    const quad = [6][2]f32{ .{ -1, -1 }, .{ 1, -1 }, .{ 1, 1 }, .{ -1, -1 }, .{ 1, 1 }, .{ -1, 1 } };
    const vbuf = try dev.createResource(.{ .buffer = .{ .size = @sizeOf(@TypeOf(quad)), .usage = .{ .vertex = true } } });
    defer dev.destroyResource(vbuf);
    @memcpy(std.mem.bytesAsSlice([2]f32, try dev.mapResource(vbuf))[0..quad.len], &quad);

    const attrs = [_]hal.VertexAttribute{.{ .location = 0, .format = .r32g32_float, .offset = 0 }};
    const pipe_lit = try dev.createPipeline(.{ .vertex = vs, .fragment = fs_lit, .vertex_layout = .{ .stride = 8, .attributes = &attrs }, .color_format = .rgba8_unorm });
    defer dev.destroyPipeline(pipe_lit);
    const pipe_shadow = try dev.createPipeline(.{ .vertex = vs, .fragment = fs_shadow, .vertex_layout = .{ .stride = 8, .attributes = &attrs }, .color_format = .rgba8_unorm });
    defer dev.destroyPipeline(pipe_shadow);

    const ctx = try dev.createContext();
    defer ctx.deinit();
    const at = struct {
        fn r(buf: []align(1) const u32, x: usize, y: usize) u8 {
            return @truncate(buf[y * W + x]);
        }
    }.r;

    // Draw 1: ref 0.2 <= 0.5 -> lit (white).
    {
        const cb = try ctx.beginCommands();
        defer cb.deinit();
        try cb.setRenderTarget(rt);
        try cb.clear(.{ .r = 0, .g = 0, .b = 0, .a = 1 });
        try cb.bindPipeline(pipe_lit);
        try cb.bindTexture(.{ .binding = 2, .image = tex, .filter = .nearest, .address_u = .clamp_to_edge, .address_v = .clamp_to_edge, .compare_enable = true, .compare_op = .less_or_equal });
        try cb.bindVertexBuffer(vbuf);
        try cb.draw(6, 0);
        try ctx.submit(cb);
    }
    const lit_v = at(std.mem.bytesAsSlice(u32, try dev.mapResource(rt)), 32, 32);

    // Draw 2: ref 0.8 > 0.5 -> shadow (black).
    {
        const cb = try ctx.beginCommands();
        defer cb.deinit();
        try cb.setRenderTarget(rt);
        try cb.clear(.{ .r = 0, .g = 0, .b = 0, .a = 1 });
        try cb.bindPipeline(pipe_shadow);
        try cb.bindTexture(.{ .binding = 2, .image = tex, .filter = .nearest, .address_u = .clamp_to_edge, .address_v = .clamp_to_edge, .compare_enable = true, .compare_op = .less_or_equal });
        try cb.bindVertexBuffer(vbuf);
        try cb.draw(6, 0);
        try ctx.submit(cb);
    }
    const shadow_v = at(std.mem.bytesAsSlice(u32, try dev.mapResource(rt)), 32, 32);

    // Assert the differential unconditionally: a green run proves the cube depth compare engaged on
    // the real GPU (GLSL samplerCubeShadow -> atlas z_cmpr TEX + TSC DEPTH_COMPARE + ZF32 cube TIC).
    try std.testing.expect(lit_v > 200); // ref 0.2 <= depth 0.5 -> lit
    try std.testing.expect(shadow_v < 60); // ref 0.8 > depth 0.5 -> shadow
}

// sampler2DArrayShadow depth compare on the real GPU: a depth32_float 2D-ARRAY (2 ZF32 layers; layer 0
// = 0.9, layer 1 = 0.5) sampled at (0.5, 0.5, layer 1) with GL_LEQUAL. The FS texture(uShadow, vec4(u,
// v, layer, ref)) returns a scalar: ref 0.2 <= 0.5 -> lit, ref 0.8 > 0.5 -> shadow. The nvidia isel
// emits a native TWO_D_ARRAY z_cmpr TEX (coord = layer,u,v with the integer layer first) + the TSC
// DEPTH_COMPARE fields. The ZF32 TWO_D_ARRAY TIC is the depth format the compare engages on. Sampling
// layer 1 (0.5) not layer 0 (0.9) also confirms layer selection (reading 0.9 would light ref 0.8).
// Asserts unconditionally. A green run proves the array depth compare engaged. Skips only without a GPU.
test "ORACLE-ARRAY-SHADOW: a sampler2DArrayShadow depth-compare TEX lights ref<=depth and shadows ref>depth on the NVIDIA GPU (skips without a GPU)" {
    const glsl = @import("../../glsl.zig");
    const gpa = std.testing.allocator;
    const dev = NvDevice.create(gpa) catch return error.SkipZigTest;
    defer dev.deinit();
    const W: u32 = 64;
    const H: u32 = 64;

    const vs_src =
        \\attribute vec2 position;
        \\void main() { gl_Position = vec4(position, 0.0, 1.0); }
    ;
    // (u,v) = (0.5,0.5), layer 1, baked compare reference: 0.2 (lit) and 0.8 (shadow).
    const fs_lit_src =
        \\precision mediump float;
        \\uniform sampler2DArrayShadow uShadow;
        \\void main() { float s = texture(uShadow, vec4(0.5, 0.5, 1.0, 0.2)); gl_FragColor = vec4(s, s, s, 1.0); }
    ;
    const fs_shadow_src =
        \\precision mediump float;
        \\uniform sampler2DArrayShadow uShadow;
        \\void main() { float s = texture(uShadow, vec4(0.5, 0.5, 1.0, 0.8)); gl_FragColor = vec4(s, s, s, 1.0); }
    ;
    const vs_bytes = try glsl.compileForStage(gpa, vs_src, .vertex);
    defer gpa.free(vs_bytes);
    const fs_lit_bytes = try glsl.compileForStage(gpa, fs_lit_src, .fragment);
    defer gpa.free(fs_lit_bytes);
    const fs_shadow_bytes = try glsl.compileForStage(gpa, fs_shadow_src, .fragment);
    defer gpa.free(fs_shadow_bytes);

    const rt = try dev.createResource(.{ .image = .{ .width = W, .height = H, .format = .rgba8_unorm, .usage = .{ .render_target = true } } });
    defer dev.destroyResource(rt);
    const vs = try dev.createShaderModule(.{ .stage = .vertex, .code = vs_bytes });
    defer dev.destroyShaderModule(vs);
    const fs_lit = try dev.createShaderModule(.{ .stage = .fragment, .code = fs_lit_bytes });
    defer dev.destroyShaderModule(fs_lit);
    const fs_shadow = try dev.createShaderModule(.{ .stage = .fragment, .code = fs_shadow_bytes });
    defer dev.destroyShaderModule(fs_shadow);

    // A 1x1 depth32_float 2D-ARRAY (2 layers) requested sampled-only builds a ZF32 TWO_D_ARRAY sampled
    // TIC. Staging is layer-major: layer 0 = 0.9, layer 1 = 0.5 (the sampled layer).
    const tex = try dev.createResource(.{ .image = .{ .width = 1, .height = 1, .format = .depth32_float, .depth = 2, .array = true, .usage = .{ .sampled = true } } });
    defer dev.destroyResource(tex);
    {
        const dst = try dev.mapResource(tex);
        const l0: f32 = 0.9;
        const l1: f32 = 0.5;
        @memcpy(dst[0..4], std.mem.asBytes(&l0));
        @memcpy(dst[4..8], std.mem.asBytes(&l1));
    }

    const quad = [6][2]f32{ .{ -1, -1 }, .{ 1, -1 }, .{ 1, 1 }, .{ -1, -1 }, .{ 1, 1 }, .{ -1, 1 } };
    const vbuf = try dev.createResource(.{ .buffer = .{ .size = @sizeOf(@TypeOf(quad)), .usage = .{ .vertex = true } } });
    defer dev.destroyResource(vbuf);
    @memcpy(std.mem.bytesAsSlice([2]f32, try dev.mapResource(vbuf))[0..quad.len], &quad);

    const attrs = [_]hal.VertexAttribute{.{ .location = 0, .format = .r32g32_float, .offset = 0 }};
    const pipe_lit = try dev.createPipeline(.{ .vertex = vs, .fragment = fs_lit, .vertex_layout = .{ .stride = 8, .attributes = &attrs }, .color_format = .rgba8_unorm });
    defer dev.destroyPipeline(pipe_lit);
    const pipe_shadow = try dev.createPipeline(.{ .vertex = vs, .fragment = fs_shadow, .vertex_layout = .{ .stride = 8, .attributes = &attrs }, .color_format = .rgba8_unorm });
    defer dev.destroyPipeline(pipe_shadow);

    const ctx = try dev.createContext();
    defer ctx.deinit();
    const at = struct {
        fn r(buf: []align(1) const u32, x: usize, y: usize) u8 {
            return @truncate(buf[y * W + x]);
        }
    }.r;

    // Draw 1: ref 0.2 <= layer-1 depth 0.5 -> lit (white).
    {
        const cb = try ctx.beginCommands();
        defer cb.deinit();
        try cb.setRenderTarget(rt);
        try cb.clear(.{ .r = 0, .g = 0, .b = 0, .a = 1 });
        try cb.bindPipeline(pipe_lit);
        try cb.bindTexture(.{ .binding = 2, .image = tex, .filter = .nearest, .address_u = .clamp_to_edge, .address_v = .clamp_to_edge, .compare_enable = true, .compare_op = .less_or_equal });
        try cb.bindVertexBuffer(vbuf);
        try cb.draw(6, 0);
        try ctx.submit(cb);
    }
    const lit_v = at(std.mem.bytesAsSlice(u32, try dev.mapResource(rt)), 32, 32);

    // Draw 2: ref 0.8 > layer-1 depth 0.5 -> shadow (black). (0.8 <= layer-0 0.9 would light if the
    // layer index were ignored, so a black here also proves the correct layer was sampled.)
    {
        const cb = try ctx.beginCommands();
        defer cb.deinit();
        try cb.setRenderTarget(rt);
        try cb.clear(.{ .r = 0, .g = 0, .b = 0, .a = 1 });
        try cb.bindPipeline(pipe_shadow);
        try cb.bindTexture(.{ .binding = 2, .image = tex, .filter = .nearest, .address_u = .clamp_to_edge, .address_v = .clamp_to_edge, .compare_enable = true, .compare_op = .less_or_equal });
        try cb.bindVertexBuffer(vbuf);
        try cb.draw(6, 0);
        try ctx.submit(cb);
    }
    const shadow_v = at(std.mem.bytesAsSlice(u32, try dev.mapResource(rt)), 32, 32);

    try std.testing.expect(lit_v > 200); // ref 0.2 <= layer-1 depth 0.5 -> lit
    try std.testing.expect(shadow_v < 60); // ref 0.8 > layer-1 depth 0.5 -> shadow
}

// The real two-pass CUBE (point-light) shadow map on the GPU (ORACLE-CUBE-SHADOW-2PASS): the cube
// extension of ORACLE-SHADOW-2PASS. Instead of uploading 6 face depths (ORACLE-CUBE-SHADOW), each of
// the 6 faces' depth is rendered on the GPU: PASS 1 renders a full-screen quad at a per-face window
// depth into a shared ZETA (block-linear ZF32), the copy engine (CE) detiles that face's rendered
// depth into a pitch-linear f32 buffer, and that row-major face is memcpy'd into the ZF32 sampled
// cube texture's staging at the face slot (face-major: face f at f*W*H*4, the exact layout the
// uploadTexture cube-atlas swizzle consumes). One ZETA is reused across all 6 faces (render -> detile
// -> copy -> repeat), lowest VRAM. We plant +Z (face 4) at depth 0.5 and the other 5 faces at 0.9, so
// PASS 2's samplerCubeShadow at dir (0,0,1) (-> face 4) with ref 0.2 -> lit / 0.8 -> shadow proves both
// the compare engaged and the correct face was sampled (reading a 0.9 face would light ref 0.8). This
// is the full point-light shadow-cube loop (render 6 face depths -> CE copy -> cube shadow-lookup) on
// real hardware. Asserts unconditionally. Skips only if there is no GPU / no CE.
test "ORACLE-CUBE-SHADOW-2PASS: PASS 1 RENDERS depth into 6 cube faces via ZETA+CE, PASS 2 cube-shadow-samples the RENDERED +Z face depth on the NVIDIA GPU (skips without a GPU)" {
    const glsl = @import("../../glsl.zig");
    const CopyEngine = @import("ce.zig").CopyEngine;
    const Resource = @import("resource.zig").Resource;
    const gpa = std.testing.allocator;
    const dev = NvDevice.create(gpa) catch return error.SkipZigTest;
    defer dev.deinit();
    const nv: *NvDevice = @ptrCast(@alignCast(dev.ptr));
    // 64x64 faces: width*4 = 256 is GOB-row-aligned, so the CE detile lands tightly packed and each
    // face's staging slot (W*H*4) is a direct memcpy of the detiled f32 depths.
    const W: u32 = 64;
    const H: u32 = 64;
    const face_bytes: usize = @as(usize, W) * H * 4;

    // PASS 1: a VS that plants gl_Position.z at a per-face window depth (near 0.5 for +Z, far 0.9 for
    // the rest) and a white FS (the color RT is a throwaway; only the ZETA depth matters).
    const depth_vs_near_src =
        \\attribute vec2 position;
        \\void main() { gl_Position = vec4(position, 0.5, 1.0); }
    ;
    const depth_vs_far_src =
        \\attribute vec2 position;
        \\void main() { gl_Position = vec4(position, 0.9, 1.0); }
    ;
    const white_fs_src =
        \\precision mediump float;
        \\void main() { gl_FragColor = vec4(1.0); }
    ;
    // PASS 2: the samplerCubeShadow lookup at +Z (face 4), baking the compare reference.
    const shadow_vs_src =
        \\attribute vec2 position;
        \\void main() { gl_Position = vec4(position, 0.0, 1.0); }
    ;
    const fs_lit_src =
        \\precision mediump float;
        \\uniform samplerCubeShadow uShadow;
        \\void main() { float s = texture(uShadow, vec4(0.0, 0.0, 1.0, 0.2)); gl_FragColor = vec4(s, s, s, 1.0); }
    ;
    const fs_shadow_src =
        \\precision mediump float;
        \\uniform samplerCubeShadow uShadow;
        \\void main() { float s = texture(uShadow, vec4(0.0, 0.0, 1.0, 0.8)); gl_FragColor = vec4(s, s, s, 1.0); }
    ;
    const depth_vs_near_bytes = try glsl.compileForStage(gpa, depth_vs_near_src, .vertex);
    defer gpa.free(depth_vs_near_bytes);
    const depth_vs_far_bytes = try glsl.compileForStage(gpa, depth_vs_far_src, .vertex);
    defer gpa.free(depth_vs_far_bytes);
    const white_fs_bytes = try glsl.compileForStage(gpa, white_fs_src, .fragment);
    defer gpa.free(white_fs_bytes);
    const shadow_vs_bytes = try glsl.compileForStage(gpa, shadow_vs_src, .vertex);
    defer gpa.free(shadow_vs_bytes);
    const fs_lit_bytes = try glsl.compileForStage(gpa, fs_lit_src, .fragment);
    defer gpa.free(fs_lit_bytes);
    const fs_shadow_bytes = try glsl.compileForStage(gpa, fs_shadow_src, .fragment);
    defer gpa.free(fs_shadow_bytes);

    const depth_vs_near = try dev.createShaderModule(.{ .stage = .vertex, .code = depth_vs_near_bytes });
    defer dev.destroyShaderModule(depth_vs_near);
    const depth_vs_far = try dev.createShaderModule(.{ .stage = .vertex, .code = depth_vs_far_bytes });
    defer dev.destroyShaderModule(depth_vs_far);
    const white_fs = try dev.createShaderModule(.{ .stage = .fragment, .code = white_fs_bytes });
    defer dev.destroyShaderModule(white_fs);
    const shadow_vs = try dev.createShaderModule(.{ .stage = .vertex, .code = shadow_vs_bytes });
    defer dev.destroyShaderModule(shadow_vs);
    const fs_lit = try dev.createShaderModule(.{ .stage = .fragment, .code = fs_lit_bytes });
    defer dev.destroyShaderModule(fs_lit);
    const fs_shadow = try dev.createShaderModule(.{ .stage = .fragment, .code = fs_shadow_bytes });
    defer dev.destroyShaderModule(fs_shadow);

    const quad = [6][2]f32{ .{ -1, -1 }, .{ 1, -1 }, .{ 1, 1 }, .{ -1, -1 }, .{ 1, 1 }, .{ -1, 1 } };
    const vbuf = try dev.createResource(.{ .buffer = .{ .size = @sizeOf(@TypeOf(quad)), .usage = .{ .vertex = true } } });
    defer dev.destroyResource(vbuf);
    @memcpy(std.mem.bytesAsSlice([2]f32, try dev.mapResource(vbuf))[0..quad.len], &quad);
    const attrs = [_]hal.VertexAttribute{.{ .location = 0, .format = .r32g32_float, .offset = 0 }};

    // The shared PASS-1 color RT (throwaway) + the ZETA depth surface reused across all 6 faces.
    const rt1 = try dev.createResource(.{ .image = .{ .width = W, .height = H, .format = .rgba8_unorm, .usage = .{ .render_target = true } } });
    defer dev.destroyResource(rt1);
    const zeta = try dev.createResource(.{ .image = .{ .width = W, .height = H, .format = .depth32_float, .usage = .{} } });
    defer dev.destroyResource(zeta);

    const pipe_depth_near = try dev.createPipeline(.{ .vertex = depth_vs_near, .fragment = white_fs, .vertex_layout = .{ .stride = 8, .attributes = &attrs }, .color_format = .rgba8_unorm, .depth = .{ .test_enable = true, .write_enable = true, .compare_op = .less } });
    defer dev.destroyPipeline(pipe_depth_near);
    const pipe_depth_far = try dev.createPipeline(.{ .vertex = depth_vs_far, .fragment = white_fs, .vertex_layout = .{ .stride = 8, .attributes = &attrs }, .color_format = .rgba8_unorm, .depth = .{ .test_enable = true, .write_enable = true, .compare_op = .less } });
    defer dev.destroyPipeline(pipe_depth_far);

    // The ZF32 sampled cube: a 6-face-wide TWO_D atlas (the format the HW DEPTH_COMPARE engages on).
    const tex = try dev.createResource(.{ .image = .{ .width = W, .height = H, .format = .depth32_float, .cube = true, .usage = .{ .sampled = true } } });
    defer dev.destroyResource(tex);

    const ctx = try dev.createContext();
    defer ctx.deinit();

    // The CE detile scratch (one 64x64 f32 face) + the copy engine, reused for all 6 faces.
    const zeta_res: *Resource = @ptrCast(@alignCast(zeta));
    const dst_pitch: u32 = W * 4;
    const dst = try nv.allocGpu(.system, @as(u64, dst_pitch) * H);
    defer nv.freeGpu(dst);
    const ce = CopyEngine.create(nv) catch return error.SkipZigTest;
    defer ce.deinit();

    // Map the cube staging once. Fill all 6 face slots, then a single bind re-swizzles it to the GPU.
    const cube_staging = try dev.mapResource(tex);
    var plus_z_center: f32 = -1;
    var face: u32 = 0;
    while (face < 6) : (face += 1) {
        // PASS 1: render this face's window depth (+Z=face 4 near 0.5, else far 0.9) into the ZETA.
        {
            const cb = try ctx.beginCommands();
            defer cb.deinit();
            try cb.setRenderTarget(rt1);
            try cb.clear(.{ .r = 0, .g = 0, .b = 0, .a = 1 });
            try cb.setDepthTarget(zeta, 1.0);
            try cb.bindPipeline(if (face == 4) pipe_depth_near else pipe_depth_far);
            try cb.bindVertexBuffer(vbuf);
            try cb.draw(6, 0);
            try ctx.submit(cb);
        }
        // COPY: CE-detile the block-linear ZETA into pitch-linear f32, then memcpy this face's
        // row-major depths into its cube-staging slot (face-major: face f at f*W*H*4).
        @memset(dst.bytes[0 .. @as(usize, dst_pitch) * H], 0);
        try ce.detile(zeta_res, dst, dst_pitch);
        @memcpy(cube_staging[face * face_bytes ..][0..face_bytes], dst.bytes[0..face_bytes]);
        if (face == 4) {
            const depths = std.mem.bytesAsSlice(f32, @as([]align(@alignOf(f32)) const u8, @alignCast(dst.bytes[0..face_bytes])));
            plus_z_center = depths[(H / 2) * W + W / 2];
        }
    }
    // Sanity: the +Z face's rendered center depth landed ~0.5 (proves the CE read the real render).
    try std.testing.expect(plus_z_center > 0.4 and plus_z_center < 0.6);

    // PASS 2 pipelines: the samplerCubeShadow lookup against the rendered cube depth.
    const pipe_lit = try dev.createPipeline(.{ .vertex = shadow_vs, .fragment = fs_lit, .vertex_layout = .{ .stride = 8, .attributes = &attrs }, .color_format = .rgba8_unorm });
    defer dev.destroyPipeline(pipe_lit);
    const pipe_shadow = try dev.createPipeline(.{ .vertex = shadow_vs, .fragment = fs_shadow, .vertex_layout = .{ .stride = 8, .attributes = &attrs }, .color_format = .rgba8_unorm });
    defer dev.destroyPipeline(pipe_shadow);

    const rt2 = try dev.createResource(.{ .image = .{ .width = W, .height = H, .format = .rgba8_unorm, .usage = .{ .render_target = true } } });
    defer dev.destroyResource(rt2);
    const at = struct {
        fn r(buf: []align(1) const u32, x: usize, y: usize) u8 {
            return @truncate(buf[y * W + x]);
        }
    }.r;

    // Draw 1: ref 0.2 <= +Z-face rendered 0.5 -> lit (white).
    {
        const cb = try ctx.beginCommands();
        defer cb.deinit();
        try cb.setRenderTarget(rt2);
        try cb.clear(.{ .r = 0, .g = 0, .b = 0, .a = 1 });
        try cb.bindPipeline(pipe_lit);
        try cb.bindTexture(.{ .binding = 2, .image = tex, .filter = .nearest, .address_u = .clamp_to_edge, .address_v = .clamp_to_edge, .compare_enable = true, .compare_op = .less_or_equal });
        try cb.bindVertexBuffer(vbuf);
        try cb.draw(6, 0);
        try ctx.submit(cb);
    }
    const lit_v = at(std.mem.bytesAsSlice(u32, try dev.mapResource(rt2)), 32, 32);

    // Draw 2: ref 0.8 > +Z-face rendered 0.5 -> shadow (black). (0.8 <= the 0.9 far faces would light
    // if the face index were ignored, so a black here also proves +Z (face 4) was sampled.)
    {
        const cb = try ctx.beginCommands();
        defer cb.deinit();
        try cb.setRenderTarget(rt2);
        try cb.clear(.{ .r = 0, .g = 0, .b = 0, .a = 1 });
        try cb.bindPipeline(pipe_shadow);
        try cb.bindTexture(.{ .binding = 2, .image = tex, .filter = .nearest, .address_u = .clamp_to_edge, .address_v = .clamp_to_edge, .compare_enable = true, .compare_op = .less_or_equal });
        try cb.bindVertexBuffer(vbuf);
        try cb.draw(6, 0);
        try ctx.submit(cb);
    }
    const shadow_v = at(std.mem.bytesAsSlice(u32, try dev.mapResource(rt2)), 32, 32);

    // A green run proves the full two-pass point-light shadow-cube loop (render 6 face depths -> CE
    // copy -> cube shadow-lookup) engaged on real hardware.
    try std.testing.expect(lit_v > 200); // ref 0.2 <= +Z rendered depth 0.5 -> lit
    try std.testing.expect(shadow_v < 60); // ref 0.8 > +Z rendered depth 0.5 -> shadow
}

// The real two-pass 2D-ARRAY (cascade) shadow map on the GPU (ORACLE-ARRAY-SHADOW-2PASS): the array
// extension of ORACLE-SHADOW-2PASS. Instead of uploading N layer depths (ORACLE-ARRAY-SHADOW), each
// layer's depth is rendered on the GPU (PASS 1 into a shared ZETA), CE-detiled, and memcpy'd into the
// ZF32 sampled TWO_D_ARRAY texture's staging at the layer slot (layer-major: layer l at l*W*H*4). One
// ZETA is reused across the layers. Layer 0 is rendered at 0.9, layer 1 at 0.5; PASS 2's
// sampler2DArrayShadow at (0.5,0.5,layer 1) with ref 0.2 -> lit / 0.8 -> shadow proves both the compare
// engaged and the correct layer was sampled (reading layer 0's 0.9 would light ref 0.8). The full
// cascade shadow loop (render N layer depths -> CE copy -> array shadow-lookup) on real hardware.
// Asserts unconditionally. Skips only if there is no GPU / no CE.
test "ORACLE-ARRAY-SHADOW-2PASS: PASS 1 RENDERS depth into 2 array layers via ZETA+CE, PASS 2 array-shadow-samples the RENDERED layer-1 depth on the NVIDIA GPU (skips without a GPU)" {
    const glsl = @import("../../glsl.zig");
    const CopyEngine = @import("ce.zig").CopyEngine;
    const Resource = @import("resource.zig").Resource;
    const gpa = std.testing.allocator;
    const dev = NvDevice.create(gpa) catch return error.SkipZigTest;
    defer dev.deinit();
    const nv: *NvDevice = @ptrCast(@alignCast(dev.ptr));
    const W: u32 = 64;
    const H: u32 = 64;
    const LAYERS: u32 = 2;
    const layer_bytes: usize = @as(usize, W) * H * 4;

    // PASS 1: per-layer window depth (0.9 for layer 0, 0.5 for layer 1) + a white FS.
    const depth_vs_l0_src =
        \\attribute vec2 position;
        \\void main() { gl_Position = vec4(position, 0.9, 1.0); }
    ;
    const depth_vs_l1_src =
        \\attribute vec2 position;
        \\void main() { gl_Position = vec4(position, 0.5, 1.0); }
    ;
    const white_fs_src =
        \\precision mediump float;
        \\void main() { gl_FragColor = vec4(1.0); }
    ;
    // PASS 2: the sampler2DArrayShadow lookup at (0.5,0.5,layer 1), baking the compare reference.
    const shadow_vs_src =
        \\attribute vec2 position;
        \\void main() { gl_Position = vec4(position, 0.0, 1.0); }
    ;
    const fs_lit_src =
        \\precision mediump float;
        \\uniform sampler2DArrayShadow uShadow;
        \\void main() { float s = texture(uShadow, vec4(0.5, 0.5, 1.0, 0.2)); gl_FragColor = vec4(s, s, s, 1.0); }
    ;
    const fs_shadow_src =
        \\precision mediump float;
        \\uniform sampler2DArrayShadow uShadow;
        \\void main() { float s = texture(uShadow, vec4(0.5, 0.5, 1.0, 0.8)); gl_FragColor = vec4(s, s, s, 1.0); }
    ;
    const depth_vs_l0_bytes = try glsl.compileForStage(gpa, depth_vs_l0_src, .vertex);
    defer gpa.free(depth_vs_l0_bytes);
    const depth_vs_l1_bytes = try glsl.compileForStage(gpa, depth_vs_l1_src, .vertex);
    defer gpa.free(depth_vs_l1_bytes);
    const white_fs_bytes = try glsl.compileForStage(gpa, white_fs_src, .fragment);
    defer gpa.free(white_fs_bytes);
    const shadow_vs_bytes = try glsl.compileForStage(gpa, shadow_vs_src, .vertex);
    defer gpa.free(shadow_vs_bytes);
    const fs_lit_bytes = try glsl.compileForStage(gpa, fs_lit_src, .fragment);
    defer gpa.free(fs_lit_bytes);
    const fs_shadow_bytes = try glsl.compileForStage(gpa, fs_shadow_src, .fragment);
    defer gpa.free(fs_shadow_bytes);

    const depth_vs_l0 = try dev.createShaderModule(.{ .stage = .vertex, .code = depth_vs_l0_bytes });
    defer dev.destroyShaderModule(depth_vs_l0);
    const depth_vs_l1 = try dev.createShaderModule(.{ .stage = .vertex, .code = depth_vs_l1_bytes });
    defer dev.destroyShaderModule(depth_vs_l1);
    const white_fs = try dev.createShaderModule(.{ .stage = .fragment, .code = white_fs_bytes });
    defer dev.destroyShaderModule(white_fs);
    const shadow_vs = try dev.createShaderModule(.{ .stage = .vertex, .code = shadow_vs_bytes });
    defer dev.destroyShaderModule(shadow_vs);
    const fs_lit = try dev.createShaderModule(.{ .stage = .fragment, .code = fs_lit_bytes });
    defer dev.destroyShaderModule(fs_lit);
    const fs_shadow = try dev.createShaderModule(.{ .stage = .fragment, .code = fs_shadow_bytes });
    defer dev.destroyShaderModule(fs_shadow);

    const quad = [6][2]f32{ .{ -1, -1 }, .{ 1, -1 }, .{ 1, 1 }, .{ -1, -1 }, .{ 1, 1 }, .{ -1, 1 } };
    const vbuf = try dev.createResource(.{ .buffer = .{ .size = @sizeOf(@TypeOf(quad)), .usage = .{ .vertex = true } } });
    defer dev.destroyResource(vbuf);
    @memcpy(std.mem.bytesAsSlice([2]f32, try dev.mapResource(vbuf))[0..quad.len], &quad);
    const attrs = [_]hal.VertexAttribute{.{ .location = 0, .format = .r32g32_float, .offset = 0 }};

    const rt1 = try dev.createResource(.{ .image = .{ .width = W, .height = H, .format = .rgba8_unorm, .usage = .{ .render_target = true } } });
    defer dev.destroyResource(rt1);
    const zeta = try dev.createResource(.{ .image = .{ .width = W, .height = H, .format = .depth32_float, .usage = .{} } });
    defer dev.destroyResource(zeta);

    const pipe_depth_l0 = try dev.createPipeline(.{ .vertex = depth_vs_l0, .fragment = white_fs, .vertex_layout = .{ .stride = 8, .attributes = &attrs }, .color_format = .rgba8_unorm, .depth = .{ .test_enable = true, .write_enable = true, .compare_op = .less } });
    defer dev.destroyPipeline(pipe_depth_l0);
    const pipe_depth_l1 = try dev.createPipeline(.{ .vertex = depth_vs_l1, .fragment = white_fs, .vertex_layout = .{ .stride = 8, .attributes = &attrs }, .color_format = .rgba8_unorm, .depth = .{ .test_enable = true, .write_enable = true, .compare_op = .less } });
    defer dev.destroyPipeline(pipe_depth_l1);

    // The ZF32 sampled TWO_D_ARRAY (the format the HW DEPTH_COMPARE engages on).
    const tex = try dev.createResource(.{ .image = .{ .width = W, .height = H, .format = .depth32_float, .depth = LAYERS, .array = true, .usage = .{ .sampled = true } } });
    defer dev.destroyResource(tex);

    const ctx = try dev.createContext();
    defer ctx.deinit();

    const zeta_res: *Resource = @ptrCast(@alignCast(zeta));
    const dst_pitch: u32 = W * 4;
    const dst = try nv.allocGpu(.system, @as(u64, dst_pitch) * H);
    defer nv.freeGpu(dst);
    const ce = CopyEngine.create(nv) catch return error.SkipZigTest;
    defer ce.deinit();

    const arr_staging = try dev.mapResource(tex);
    var l1_center: f32 = -1;
    var layer: u32 = 0;
    while (layer < LAYERS) : (layer += 1) {
        {
            const cb = try ctx.beginCommands();
            defer cb.deinit();
            try cb.setRenderTarget(rt1);
            try cb.clear(.{ .r = 0, .g = 0, .b = 0, .a = 1 });
            try cb.setDepthTarget(zeta, 1.0);
            try cb.bindPipeline(if (layer == 1) pipe_depth_l1 else pipe_depth_l0);
            try cb.bindVertexBuffer(vbuf);
            try cb.draw(6, 0);
            try ctx.submit(cb);
        }
        @memset(dst.bytes[0 .. @as(usize, dst_pitch) * H], 0);
        try ce.detile(zeta_res, dst, dst_pitch);
        @memcpy(arr_staging[layer * layer_bytes ..][0..layer_bytes], dst.bytes[0..layer_bytes]);
        if (layer == 1) {
            const depths = std.mem.bytesAsSlice(f32, @as([]align(@alignOf(f32)) const u8, @alignCast(dst.bytes[0..layer_bytes])));
            l1_center = depths[(H / 2) * W + W / 2];
        }
    }
    // Sanity: layer 1's rendered center depth landed ~0.5 (proves the CE read the real render).
    try std.testing.expect(l1_center > 0.4 and l1_center < 0.6);

    const pipe_lit = try dev.createPipeline(.{ .vertex = shadow_vs, .fragment = fs_lit, .vertex_layout = .{ .stride = 8, .attributes = &attrs }, .color_format = .rgba8_unorm });
    defer dev.destroyPipeline(pipe_lit);
    const pipe_shadow = try dev.createPipeline(.{ .vertex = shadow_vs, .fragment = fs_shadow, .vertex_layout = .{ .stride = 8, .attributes = &attrs }, .color_format = .rgba8_unorm });
    defer dev.destroyPipeline(pipe_shadow);

    const rt2 = try dev.createResource(.{ .image = .{ .width = W, .height = H, .format = .rgba8_unorm, .usage = .{ .render_target = true } } });
    defer dev.destroyResource(rt2);
    const at = struct {
        fn r(buf: []align(1) const u32, x: usize, y: usize) u8 {
            return @truncate(buf[y * W + x]);
        }
    }.r;

    // Draw 1: ref 0.2 <= layer-1 rendered 0.5 -> lit (white).
    {
        const cb = try ctx.beginCommands();
        defer cb.deinit();
        try cb.setRenderTarget(rt2);
        try cb.clear(.{ .r = 0, .g = 0, .b = 0, .a = 1 });
        try cb.bindPipeline(pipe_lit);
        try cb.bindTexture(.{ .binding = 2, .image = tex, .filter = .nearest, .address_u = .clamp_to_edge, .address_v = .clamp_to_edge, .compare_enable = true, .compare_op = .less_or_equal });
        try cb.bindVertexBuffer(vbuf);
        try cb.draw(6, 0);
        try ctx.submit(cb);
    }
    const lit_v = at(std.mem.bytesAsSlice(u32, try dev.mapResource(rt2)), 32, 32);

    // Draw 2: ref 0.8 > layer-1 rendered 0.5 -> shadow (black). (0.8 <= layer-0 0.9 would light if the
    // layer index were ignored, so a black here also proves layer 1 was sampled.)
    {
        const cb = try ctx.beginCommands();
        defer cb.deinit();
        try cb.setRenderTarget(rt2);
        try cb.clear(.{ .r = 0, .g = 0, .b = 0, .a = 1 });
        try cb.bindPipeline(pipe_shadow);
        try cb.bindTexture(.{ .binding = 2, .image = tex, .filter = .nearest, .address_u = .clamp_to_edge, .address_v = .clamp_to_edge, .compare_enable = true, .compare_op = .less_or_equal });
        try cb.bindVertexBuffer(vbuf);
        try cb.draw(6, 0);
        try ctx.submit(cb);
    }
    const shadow_v = at(std.mem.bytesAsSlice(u32, try dev.mapResource(rt2)), 32, 32);

    // A green run proves the full two-pass cascade shadow loop (render N layer depths -> CE copy ->
    // array shadow-lookup) engaged on real hardware.
    try std.testing.expect(lit_v > 200); // ref 0.2 <= layer-1 rendered depth 0.5 -> lit
    try std.testing.expect(shadow_v < 60); // ref 0.8 > layer-1 rendered depth 0.5 -> shadow
}

test "ORACLE-GATHER: textureGather returns the 4 footprint texels of one component in GL order on the NVIDIA GPU (TLD4), matching software (skips without a GPU)" {
    const glsl = @import("../../glsl.zig");
    const sw_sampler = @import("../software/sampler.zig");
    const gpa = std.testing.allocator;
    const dev = NvDevice.create(gpa) catch return error.SkipZigTest;
    defer dev.deinit();
    const W: u32 = 64;
    const H: u32 = 64;

    // A full-screen quad + a fragment shader that gathers ONE component of a 2x2 texture into RGBA.
    // At the screen center the footprint is exactly the 2x2, so the center pixel reveals the gather
    // order + the selected component. Proves the whole GPU path: GLSL textureGather -> vulcan-glsl ->
    // SPIR-V OpImageGather -> the nvidia isel's TLD4 (0xd64). The expected values are the SOFTWARE
    // golden (sampleTextureGather), so this cross-checks the HW gather order against software.
    const vs_src =
        \\attribute vec2 position;
        \\attribute vec2 aUV;
        \\varying vec2 vUV;
        \\void main() { gl_Position = vec4(position, 0.0, 1.0); vUV = aUV; }
    ;
    const fs_r =
        \\precision mediump float;
        \\uniform sampler2D uTex;
        \\varying vec2 vUV;
        \\void main() { gl_FragColor = textureGather(uTex, vUV); }
    ;
    const fs_b =
        \\precision mediump float;
        \\uniform sampler2D uTex;
        \\varying vec2 vUV;
        \\void main() { gl_FragColor = textureGather(uTex, vUV, 2); }
    ;
    const vs_bytes = try glsl.compileForStage(gpa, vs_src, .vertex);
    defer gpa.free(vs_bytes);

    // The 2x2 texture: distinct R and B per texel. Same layout as the EGL gather oracle.
    //   (col0,row0) R=26  B=200   (col1,row0) R=51  B=150
    //   (col0,row1) R=77  B=100   (col1,row1) R=102 B=50
    const tex_px = [_]u8{
        26, 0, 200, 255, 51,  0, 150, 255,
        77, 0, 100, 255, 102, 0, 50,  255,
    };
    const tex = try dev.createResource(.{ .image = .{ .width = 2, .height = 2, .format = .rgba8_unorm, .usage = .{ .sampled = true } } });
    defer dev.destroyResource(tex);
    @memcpy((try dev.mapResource(tex))[0..tex_px.len], &tex_px);

    // The software golden for the center coord (0.5,0.5): the 4 gathered texels in GL order.
    var sw_desc = sw_sampler.TexDesc{ .pixels = &tex_px, .width = 2, .height = 2, .pitch = 8, .filter = .nearest, .address_u = .clamp_to_edge, .address_v = .clamp_to_edge };
    var gold_r: [4]f32 = undefined;
    var gold_b: [4]f32 = undefined;
    sw_sampler.sampleTextureGather(&sw_desc, 0.5, 0.5, 0, &gold_r);
    sw_sampler.sampleTextureGather(&sw_desc, 0.5, 0.5, 2, &gold_b);

    const Vtx = extern struct { x: f32, y: f32, u: f32, v: f32 };
    const quad = [6]Vtx{
        .{ .x = -1, .y = -1, .u = 0, .v = 0 },
        .{ .x = 1, .y = -1, .u = 1, .v = 0 },
        .{ .x = 1, .y = 1, .u = 1, .v = 1 },
        .{ .x = -1, .y = -1, .u = 0, .v = 0 },
        .{ .x = 1, .y = 1, .u = 1, .v = 1 },
        .{ .x = -1, .y = 1, .u = 0, .v = 1 },
    };
    const vbuf = try dev.createResource(.{ .buffer = .{ .size = @sizeOf(@TypeOf(quad)), .usage = .{ .vertex = true } } });
    defer dev.destroyResource(vbuf);
    @memcpy(std.mem.bytesAsSlice(Vtx, try dev.mapResource(vbuf))[0..quad.len], &quad);
    const attrs = [_]hal.VertexAttribute{
        .{ .location = 0, .format = .r32g32_float, .offset = 0 },
        .{ .location = 1, .format = .r32g32_float, .offset = 8 },
    };

    // Render one gather FS and return the center pixel's RGBA (the 4 gathered values).
    const Render = struct {
        fn run(device: hal.Device, vsb: []const u8, fs_src: []const u8, texh: anytype, vb: anytype, at: []const hal.VertexAttribute, alloc: std.mem.Allocator) ![4]u8 {
            const fs_bytes = try glsl.compileForStage(alloc, fs_src, .fragment);
            defer alloc.free(fs_bytes);
            const rt = try device.createResource(.{ .image = .{ .width = W, .height = H, .format = .rgba8_unorm, .usage = .{ .render_target = true } } });
            defer device.destroyResource(rt);
            const vs = try device.createShaderModule(.{ .stage = .vertex, .code = vsb });
            defer device.destroyShaderModule(vs);
            const fs = try device.createShaderModule(.{ .stage = .fragment, .code = fs_bytes });
            defer device.destroyShaderModule(fs);
            const pipe = try device.createPipeline(.{
                .vertex = vs,
                .fragment = fs,
                .vertex_layout = .{ .stride = @sizeOf(Vtx), .attributes = at },
                .color_format = .rgba8_unorm,
            });
            defer device.destroyPipeline(pipe);
            const ctx = try device.createContext();
            defer ctx.deinit();
            {
                const cb = try ctx.beginCommands();
                defer cb.deinit();
                try cb.setRenderTarget(rt);
                try cb.clear(.{ .r = 0, .g = 0, .b = 0, .a = 1 });
                try cb.bindPipeline(pipe);
                try cb.bindTexture(.{ .binding = 2, .image = texh, .filter = .nearest, .address_u = .clamp_to_edge, .address_v = .clamp_to_edge });
                try cb.bindVertexBuffer(vb);
                try cb.draw(6, 0);
                try ctx.submit(cb);
            }
            const fb = std.mem.bytesAsSlice(u32, try device.mapResource(rt));
            const v = fb[32 * W + 32];
            return .{ @truncate(v), @truncate(v >> 8), @truncate(v >> 16), @truncate(v >> 24) };
        }
    };

    const r = try Render.run(dev, vs_bytes, fs_r, tex, vbuf, &attrs, gpa);
    const b = try Render.run(dev, vs_bytes, fs_b, tex, vbuf, &attrs, gpa);
    // The GPU TLD4 gather must match the software golden (tolerance for the 8-bit quantization).
    inline for (0..4) |c| {
        try std.testing.expectApproxEqAbs(gold_r[c] * 255.0, @as(f32, @floatFromInt(r[c])), 6);
        try std.testing.expectApproxEqAbs(gold_b[c] * 255.0, @as(f32, @floatFromInt(b[c])), 6);
    }
}

test "ORACLE-FETCH: texelFetch returns the EXACT texel at integer coords on the NVIDIA GPU (TLD), matching software (skips without a GPU)" {
    // Proves the whole texelFetch path: GLSL texelFetch(sampler2D, ivec2, int) -> vulcan-glsl ->
    // SPIR-V OpImageFetch -> the nvidia isel's TLD (0xd67, integer coords + explicit LOD). A 2x2
    // texture with 4 distinct texels. A uCoord uniform picks the integer texel. The bound sampler is
    // linear, but texelFetch ignores filtering. Cross-checks the software golden (sampleTextureFetch).
    const glsl = @import("../../glsl.zig");
    const sw_sampler = @import("../software/sampler.zig");
    const gpa = std.testing.allocator;
    const dev = NvDevice.create(gpa) catch return error.SkipZigTest;
    defer dev.deinit();
    const W: u32 = 64;
    const H: u32 = 64;

    // The fetch coord comes in as a varying (constant across the quad) so each draw picks one texel
    // without a uniform block. ivec2(vCoord) converts float->int in the shader.
    const vs_src =
        \\attribute vec2 position;
        \\attribute vec2 aCoord;
        \\varying vec2 vCoord;
        \\void main() { gl_Position = vec4(position, 0.0, 1.0); vCoord = aCoord; }
    ;
    const fs_src =
        \\precision mediump float;
        \\uniform sampler2D uTex;
        \\varying vec2 vCoord;
        \\void main() { gl_FragColor = texelFetch(uTex, ivec2(vCoord), 0); }
    ;
    const vs_bytes = try glsl.compileForStage(gpa, vs_src, .vertex);
    defer gpa.free(vs_bytes);
    const fs_bytes = try glsl.compileForStage(gpa, fs_src, .fragment);
    defer gpa.free(fs_bytes);

    const rt = try dev.createResource(.{ .image = .{ .width = W, .height = H, .format = .rgba8_unorm, .usage = .{ .render_target = true } } });
    defer dev.destroyResource(rt);
    const vs = try dev.createShaderModule(.{ .stage = .vertex, .code = vs_bytes });
    defer dev.destroyShaderModule(vs);
    const fs = try dev.createShaderModule(.{ .stage = .fragment, .code = fs_bytes });
    defer dev.destroyShaderModule(fs);

    // 2x2 RGBA8: (0,0)=red (1,0)=green / (0,1)=blue (1,1)=white.
    const tex_px = [_]u8{ 255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255, 255, 255, 255 };
    const tex = try dev.createResource(.{ .image = .{ .width = 2, .height = 2, .format = .rgba8_unorm, .usage = .{ .sampled = true } } });
    defer dev.destroyResource(tex);
    @memcpy((try dev.mapResource(tex))[0..tex_px.len], &tex_px);
    var sw_desc = sw_sampler.TexDesc{ .pixels = &tex_px, .width = 2, .height = 2, .pitch = 8, .filter = .linear };

    const Vtx = extern struct { x: f32, y: f32, cx: f32, cy: f32 };
    const attrs = [_]hal.VertexAttribute{
        .{ .location = 0, .format = .r32g32_float, .offset = 0 },
        .{ .location = 1, .format = .r32g32_float, .offset = 8 },
    };
    const vbuf = try dev.createResource(.{ .buffer = .{ .size = 3 * @sizeOf(Vtx), .usage = .{ .vertex = true } } });
    defer dev.destroyResource(vbuf);
    const pipe = try dev.createPipeline(.{
        .vertex = vs,
        .fragment = fs,
        .vertex_layout = .{ .stride = @sizeOf(Vtx), .attributes = &attrs },
        .color_format = .rgba8_unorm,
    });
    defer dev.destroyPipeline(pipe);
    const ctx = try dev.createContext();
    defer ctx.deinit();

    const Case = struct { x: f32, y: f32 };
    const cases = [_]Case{ .{ .x = 0, .y = 0 }, .{ .x = 1, .y = 0 }, .{ .x = 0, .y = 1 }, .{ .x = 1, .y = 1 } };
    for (cases) |cse| {
        const tri = [3]Vtx{
            .{ .x = -1, .y = -1, .cx = cse.x, .cy = cse.y },
            .{ .x = 3, .y = -1, .cx = cse.x, .cy = cse.y },
            .{ .x = -1, .y = 3, .cx = cse.x, .cy = cse.y },
        };
        @memcpy(std.mem.bytesAsSlice(Vtx, try dev.mapResource(vbuf))[0..3], &tri);
        {
            const cb = try ctx.beginCommands();
            defer cb.deinit();
            try cb.setRenderTarget(rt);
            try cb.clear(.{ .r = 0, .g = 0, .b = 0, .a = 1 });
            try cb.bindPipeline(pipe);
            try cb.bindTexture(.{ .binding = 2, .image = tex, .filter = .linear, .min_filter = .linear, .address_u = .clamp_to_edge, .address_v = .clamp_to_edge });
            try cb.bindVertexBuffer(vbuf);
            try cb.draw(3, 0);
            try ctx.submit(cb);
        }
        const fb = std.mem.bytesAsSlice(u32, try dev.mapResource(rt));
        const val = fb[32 * W + 32];
        const g = [3]u8{ @truncate(val), @truncate(val >> 8), @truncate(val >> 16) };
        var gold: [4]f32 = undefined;
        sw_sampler.sampleTextureFetch(&sw_desc, @intFromFloat(cse.x), @intFromFloat(cse.y), 0, &gold);
        inline for (0..3) |ch| try std.testing.expectApproxEqAbs(gold[ch] * 255.0, @as(f32, @floatFromInt(g[ch])), 8);
    }
}

test "ORACLE-FETCH3: texelFetch on a sampler2DArray + sampler3D fetches the exact (x,y,z) texel on the NVIDIA GPU (TLD), matching software (skips without a GPU)" {
    // Proves the array/3D texelFetch GPU path: GLSL texelFetch(samplerARRAY/3D, ivec3, int) -> the
    // nvidia isel's TLD with dim=Array2D(5) or 3D(2) + a 3-register integer coord (array = layer-first
    // (layer,x,y); 3D = (x,y,z)). A 2x2x3 texture. Each (kind, coord) case cross-checks the software
    // golden sampleTextureFetch3D. Runs the array and 3D samplers in one test.
    const glsl = @import("../../glsl.zig");
    const sw_sampler = @import("../software/sampler.zig");
    const gpa = std.testing.allocator;
    const dev = NvDevice.create(gpa) catch return error.SkipZigTest;
    defer dev.deinit();
    const W: u32 = 64;
    const H: u32 = 64;

    const vs_src =
        \\attribute vec2 position;
        \\attribute vec3 aCoord;
        \\varying vec3 vCoord;
        \\void main() { gl_Position = vec4(position, 0.0, 1.0); vCoord = aCoord; }
    ;
    const vs_bytes = try glsl.compileForStage(gpa, vs_src, .vertex);
    defer gpa.free(vs_bytes);
    // 2x2x3, layer/slice-major. Layer 0: red/green/blue/white; layer 1: yellow; layer 2: magenta.
    const vol = [_]u8{
        255, 0,   0,   255, 0,   255, 0,   255,
        0,   0,   255, 255, 255, 255, 255, 255,
        255, 255, 0,   255, 255, 255, 0,   255,
        255, 255, 0,   255, 255, 255, 0,   255,
        255, 0,   255, 255, 255, 0,   255, 255,
        255, 0,   255, 255, 255, 0,   255, 255,
    };
    var sw_desc = sw_sampler.TexDesc{ .pixels = &vol, .width = 2, .height = 2, .pitch = 8, .depth = 3 };

    const Vtx = extern struct { x: f32, y: f32, cx: f32, cy: f32, cz: f32 };
    const attrs = [_]hal.VertexAttribute{
        .{ .location = 0, .format = .r32g32_float, .offset = 0 },
        .{ .location = 1, .format = .r32g32b32_float, .offset = 8 },
    };

    // Render `fs` with `arr`/`d3` bound and a fixed integer (x,y,z) coord, return the center RGB.
    const Runner = struct {
        fn run(device: anytype, vsb: []const u8, is_array: bool, x: i32, y: i32, z: i32, at: []const hal.VertexAttribute, alloc: std.mem.Allocator) ![3]u8 {
            const fs_src = if (is_array)
                \\precision mediump float;
                \\uniform sampler2DArray uT;
                \\varying vec3 vCoord;
                \\void main() { gl_FragColor = texelFetch(uT, ivec3(vCoord), 0); }
            else
                \\precision mediump float;
                \\uniform sampler3D uT;
                \\varying vec3 vCoord;
                \\void main() { gl_FragColor = texelFetch(uT, ivec3(vCoord), 0); }
            ;
            const fsb = try glsl.compileForStage(alloc, fs_src, .fragment);
            defer alloc.free(fsb);
            const rt = try device.createResource(.{ .image = .{ .width = W, .height = H, .format = .rgba8_unorm, .usage = .{ .render_target = true } } });
            defer device.destroyResource(rt);
            const vs = try device.createShaderModule(.{ .stage = .vertex, .code = vsb });
            defer device.destroyShaderModule(vs);
            const fs = try device.createShaderModule(.{ .stage = .fragment, .code = fsb });
            defer device.destroyShaderModule(fs);
            const vol2 = [_]u8{
                255, 0,   0,   255, 0,   255, 0,   255, 0,   0,   255, 255, 255, 255, 255, 255,
                255, 255, 0,   255, 255, 255, 0,   255, 255, 255, 0,   255, 255, 255, 0,   255,
                255, 0,   255, 255, 255, 0,   255, 255, 255, 0,   255, 255, 255, 0,   255, 255,
            };
            const tex = try device.createResource(.{ .image = .{ .width = 2, .height = 2, .depth = 3, .array = is_array, .format = .rgba8_unorm, .usage = .{ .sampled = true } } });
            defer device.destroyResource(tex);
            @memcpy((try device.mapResource(tex))[0..vol2.len], &vol2);
            const vbuf = try device.createResource(.{ .buffer = .{ .size = 3 * @sizeOf(Vtx), .usage = .{ .vertex = true } } });
            defer device.destroyResource(vbuf);
            const fx: f32 = @floatFromInt(x);
            const fy: f32 = @floatFromInt(y);
            const fz: f32 = @floatFromInt(z);
            const tri = [3]Vtx{
                .{ .x = -1, .y = -1, .cx = fx, .cy = fy, .cz = fz },
                .{ .x = 3, .y = -1, .cx = fx, .cy = fy, .cz = fz },
                .{ .x = -1, .y = 3, .cx = fx, .cy = fy, .cz = fz },
            };
            @memcpy(std.mem.bytesAsSlice(Vtx, try device.mapResource(vbuf))[0..3], &tri);
            const pipe = try device.createPipeline(.{ .vertex = vs, .fragment = fs, .vertex_layout = .{ .stride = @sizeOf(Vtx), .attributes = at }, .color_format = .rgba8_unorm });
            defer device.destroyPipeline(pipe);
            const ctx = try device.createContext();
            defer ctx.deinit();
            const cb = try ctx.beginCommands();
            defer cb.deinit();
            try cb.setRenderTarget(rt);
            try cb.clear(.{ .r = 0, .g = 0, .b = 0, .a = 1 });
            try cb.bindPipeline(pipe);
            try cb.bindTexture(.{ .binding = 2, .image = tex, .filter = .nearest, .address_u = .clamp_to_edge, .address_v = .clamp_to_edge });
            try cb.bindVertexBuffer(vbuf);
            try cb.draw(3, 0);
            try ctx.submit(cb);
            const fb = std.mem.bytesAsSlice(u32, try device.mapResource(rt));
            const val = fb[32 * W + 32];
            return .{ @truncate(val), @truncate(val >> 8), @truncate(val >> 16) };
        }
    };

    const Case = struct { arr: bool, x: i32, y: i32, z: i32 };
    const cases = [_]Case{
        .{ .arr = true, .x = 0, .y = 0, .z = 0 }, // array layer 0 (0,0) -> red
        .{ .arr = true, .x = 1, .y = 1, .z = 0 }, // array layer 0 (1,1) -> white
        .{ .arr = true, .x = 0, .y = 0, .z = 1 }, // array layer 1 -> yellow
        .{ .arr = true, .x = 1, .y = 0, .z = 2 }, // array layer 2 -> magenta
        .{ .arr = false, .x = 0, .y = 0, .z = 0 }, // 3D slice 0 (0,0) -> red
        .{ .arr = false, .x = 1, .y = 1, .z = 0 }, // 3D slice 0 (1,1) -> white
        .{ .arr = false, .x = 0, .y = 0, .z = 2 }, // 3D slice 2 -> magenta
    };
    for (cases) |cse| {
        const g = try Runner.run(dev, vs_bytes, cse.arr, cse.x, cse.y, cse.z, &attrs, gpa);
        var gold: [4]f32 = undefined;
        sw_sampler.sampleTextureFetch3D(&sw_desc, cse.x, cse.y, cse.z, 0, &gold);
        inline for (0..3) |ch| try std.testing.expectApproxEqAbs(gold[ch] * 255.0, @as(f32, @floatFromInt(g[ch])), 10);
    }
}

test "ORACLE-3D-TEX: a sampler3D LUT selects the right Z-slice by the w coordinate on the NVIDIA GPU (skips without a GPU)" {
    const glsl = @import("../../glsl.zig");
    const gpa = std.testing.allocator;
    const dev = NvDevice.create(gpa) catch return error.SkipZigTest;
    defer dev.deinit();
    const W: u32 = 64;
    const H: u32 = 64;

    const vs_src =
        \\attribute vec2 position;
        \\attribute vec3 aCoord;
        \\varying vec3 vCoord;
        \\void main() { gl_Position = vec4(position, 0.0, 1.0); vCoord = aCoord; }
    ;
    const fs_src =
        \\precision mediump float;
        \\uniform sampler3D uLut;
        \\varying vec3 vCoord;
        \\void main() { gl_FragColor = texture(uLut, vCoord); }
    ;
    const vs_bytes = try glsl.compileForStage(gpa, vs_src, .vertex);
    defer gpa.free(vs_bytes);
    const fs_bytes = try glsl.compileForStage(gpa, fs_src, .fragment);
    defer gpa.free(fs_bytes);

    const rt = try dev.createResource(.{ .image = .{ .width = W, .height = H, .format = .rgba8_unorm, .usage = .{ .render_target = true } } });
    defer dev.destroyResource(rt);
    const vs = try dev.createShaderModule(.{ .stage = .vertex, .code = vs_bytes });
    defer dev.destroyShaderModule(vs);
    const fs = try dev.createShaderModule(.{ .stage = .fragment, .code = fs_bytes });
    defer dev.destroyShaderModule(fs);

    // A 1x1x4 LUT with 4 distinct slice colors (red, green, blue, white) to read the w->slice map.
    const DEPTH: u32 = 4;
    const lut = try dev.createResource(.{ .image = .{ .width = 1, .height = 1, .depth = DEPTH, .format = .rgba8_unorm, .usage = .{ .sampled = true } } });
    defer dev.destroyResource(lut);
    {
        const cols = [4][4]u8{ .{ 255, 0, 0, 255 }, .{ 0, 255, 0, 255 }, .{ 0, 0, 255, 255 }, .{ 255, 255, 255, 255 } };
        var vol: [4 * 4]u8 = undefined;
        for (0..4) |s| @memcpy(vol[s * 4 ..][0..4], &cols[s]);
        const dst = try dev.mapResource(lut);
        @memcpy(dst[0..vol.len], &vol);
    }

    const Vtx = extern struct { x: f32, y: f32, cx: f32, cy: f32, cz: f32 };
    const attrs = [_]hal.VertexAttribute{
        .{ .location = 0, .format = .r32g32_float, .offset = 0 },
        .{ .location = 1, .format = .r32g32b32_float, .offset = 8 },
    };
    const vbuf = try dev.createResource(.{ .buffer = .{ .size = 6 * @sizeOf(Vtx), .usage = .{ .vertex = true } } });
    defer dev.destroyResource(vbuf);
    const pipe = try dev.createPipeline(.{
        .vertex = vs,
        .fragment = fs,
        .vertex_layout = .{ .stride = @sizeOf(Vtx), .attributes = &attrs },
        .color_format = .rgba8_unorm,
    });
    defer dev.destroyPipeline(pipe);
    const ctx = try dev.createContext();
    defer ctx.deinit();

    // One draw with a w-gradient across the quad (left edge cz=0, right edge cz=1); reading
    // several x positions samples several w values in a single submit (avoids the multi-draw hang).
    {
        const V = extern struct { x: f32, y: f32, cx: f32, cy: f32, cz: f32 };
        const q = [6]V{
            .{ .x = -1, .y = -1, .cx = 0.5, .cy = 0.5, .cz = 0 }, .{ .x = 1, .y = -1, .cx = 0.5, .cy = 0.5, .cz = 1 },
            .{ .x = 1, .y = 1, .cx = 0.5, .cy = 0.5, .cz = 1 },   .{ .x = -1, .y = -1, .cx = 0.5, .cy = 0.5, .cz = 0 },
            .{ .x = 1, .y = 1, .cx = 0.5, .cy = 0.5, .cz = 1 },   .{ .x = -1, .y = 1, .cx = 0.5, .cy = 0.5, .cz = 0 },
        };
        @memcpy(std.mem.bytesAsSlice(V, try dev.mapResource(vbuf))[0..6], &q);
        const cb = try ctx.beginCommands();
        defer cb.deinit();
        try cb.setRenderTarget(rt);
        try cb.clear(.{ .r = 0, .g = 0, .b = 0, .a = 1 });
        try cb.bindPipeline(pipe);
        try cb.bindTexture(.{ .binding = 2, .image = lut, .filter = .nearest, .address_u = .clamp_to_edge, .address_v = .clamp_to_edge });
        try cb.bindVertexBuffer(vbuf);
        try cb.draw(6, 0);
        try ctx.submit(cb);
    }
    const px = struct {
        fn p(d: anytype, r: *hal.Resource, x: usize) [3]u8 {
            const fb = std.mem.bytesAsSlice(u32, (d.mapResource(r) catch unreachable));
            const v = fb[32 * W + x];
            return .{ @truncate(v), @truncate(v >> 8), @truncate(v >> 16) };
        }
    }.p;
    // The w coordinate (0 at left, 1 at right) selects the LUT slice: leftmost -> slice 0 red,
    // rightmost -> slice 3 white, green + blue in between. Per-slice selection on the GPU.
    const s0 = px(dev, rt, 2); // w~0.03 -> slice 0 = red
    try std.testing.expect(s0[0] > 200 and s0[1] < 60 and s0[2] < 60);
    const s1 = px(dev, rt, 22); // w~0.34 -> slice 1 = green
    try std.testing.expect(s1[1] > 200 and s1[0] < 60 and s1[2] < 60);
    const s2 = px(dev, rt, 42); // w~0.66 -> slice 2 = blue
    try std.testing.expect(s2[2] > 200 and s2[0] < 60 and s2[1] < 60);
    const s3 = px(dev, rt, 60); // w~0.94 -> slice 3 = white
    try std.testing.expect(s3[0] > 200 and s3[1] > 200 and s3[2] > 200);
}

test "ORACLE-3D-WITHIN-SLICE: a sampler3D with a 2x2 within-slice picks the right in-slice texel by (u,v) on the NVIDIA GPU (skips without a GPU)" {
    // Previously a hard wall: 3D within-slice u,v read texel 0. The root cause was a
    // 2-aligned 3D coord (which also faulted Xid 13) scrambling the coordinate order. With the
    // coord 4-aligned + emitted in natural (u,v,w) order, in-slice addressing works. A 2x2x2 LUT:
    // slice 0 has 4 distinct in-slice colors (red/green/blue/white), slice 1 all yellow. Sample
    // slice 0 (w~0.25) at the four quadrant (u,v) centers and assert each picks its own texel.
    const glsl = @import("../../glsl.zig");
    const gpa = std.testing.allocator;
    const dev = NvDevice.create(gpa) catch return error.SkipZigTest;
    defer dev.deinit();
    const W: u32 = 64;
    const H: u32 = 64;

    const vs_src =
        \\attribute vec2 position;
        \\attribute vec3 aCoord;
        \\varying vec3 vCoord;
        \\void main() { gl_Position = vec4(position, 0.0, 1.0); vCoord = aCoord; }
    ;
    const fs_src =
        \\precision mediump float;
        \\uniform sampler3D uLut;
        \\varying vec3 vCoord;
        \\void main() { gl_FragColor = texture(uLut, vCoord); }
    ;
    const vs_bytes = try glsl.compileForStage(gpa, vs_src, .vertex);
    defer gpa.free(vs_bytes);
    const fs_bytes = try glsl.compileForStage(gpa, fs_src, .fragment);
    defer gpa.free(fs_bytes);

    const rt = try dev.createResource(.{ .image = .{ .width = W, .height = H, .format = .rgba8_unorm, .usage = .{ .render_target = true } } });
    defer dev.destroyResource(rt);
    const vs = try dev.createShaderModule(.{ .stage = .vertex, .code = vs_bytes });
    defer dev.destroyShaderModule(vs);
    const fs = try dev.createShaderModule(.{ .stage = .fragment, .code = fs_bytes });
    defer dev.destroyShaderModule(fs);

    // 2x2x2 LUT, slice-major, row-major within a slice. Slice 0: (col0,row0)=red (col1,row0)=green
    // (col0,row1)=blue (col1,row1)=white. Slice 1: all yellow.
    const lut = try dev.createResource(.{ .image = .{ .width = 2, .height = 2, .depth = 2, .format = .rgba8_unorm, .usage = .{ .sampled = true } } });
    defer dev.destroyResource(lut);
    {
        const vol = [_]u8{
            255, 0, 0, 255, 0, 255, 0, 255, // slice 0 row 0: red, green
            0, 0, 255, 255, 255, 255, 255, 255, // slice 0 row 1: blue, white
            255, 255, 0, 255, 255, 255, 0, 255, // slice 1 row 0: yellow, yellow
            255, 255, 0, 255, 255, 255, 0, 255, // slice 1 row 1: yellow, yellow
        };
        const dst = try dev.mapResource(lut);
        @memcpy(dst[0..vol.len], &vol);
    }

    const Vtx = extern struct { x: f32, y: f32, cx: f32, cy: f32, cz: f32 };
    const attrs = [_]hal.VertexAttribute{
        .{ .location = 0, .format = .r32g32_float, .offset = 0 },
        .{ .location = 1, .format = .r32g32b32_float, .offset = 8 },
    };
    const vbuf = try dev.createResource(.{ .buffer = .{ .size = 6 * @sizeOf(Vtx), .usage = .{ .vertex = true } } });
    defer dev.destroyResource(vbuf);
    const pipe = try dev.createPipeline(.{
        .vertex = vs,
        .fragment = fs,
        .vertex_layout = .{ .stride = @sizeOf(Vtx), .attributes = &attrs },
        .color_format = .rgba8_unorm,
    });
    defer dev.destroyPipeline(pipe);
    const ctx = try dev.createContext();
    defer ctx.deinit();

    // One draw: a full-screen quad whose (u,v) spans 0..1 across the screen, w fixed at 0.25 (slice
    // 0). Reading the four quadrant centers samples the four in-slice (u,v) texels. Nearest filter.
    {
        const V = extern struct { x: f32, y: f32, cx: f32, cy: f32, cz: f32 };
        const q = [6]V{
            .{ .x = -1, .y = -1, .cx = 0, .cy = 0, .cz = 0.25 }, .{ .x = 1, .y = -1, .cx = 1, .cy = 0, .cz = 0.25 },
            .{ .x = 1, .y = 1, .cx = 1, .cy = 1, .cz = 0.25 },   .{ .x = -1, .y = -1, .cx = 0, .cy = 0, .cz = 0.25 },
            .{ .x = 1, .y = 1, .cx = 1, .cy = 1, .cz = 0.25 },   .{ .x = -1, .y = 1, .cx = 0, .cy = 1, .cz = 0.25 },
        };
        @memcpy(std.mem.bytesAsSlice(V, try dev.mapResource(vbuf))[0..6], &q);
        const cb = try ctx.beginCommands();
        defer cb.deinit();
        try cb.setRenderTarget(rt);
        try cb.clear(.{ .r = 0, .g = 0, .b = 0, .a = 1 });
        try cb.bindPipeline(pipe);
        try cb.bindTexture(.{ .binding = 2, .image = lut, .filter = .nearest, .address_u = .clamp_to_edge, .address_v = .clamp_to_edge });
        try cb.bindVertexBuffer(vbuf);
        try cb.draw(6, 0);
        try ctx.submit(cb);
    }
    const at = struct {
        fn p(d: anytype, r: *hal.Resource, x: usize, y: usize) [3]u8 {
            const fb = std.mem.bytesAsSlice(u32, (d.mapResource(r) catch unreachable));
            const v = fb[y * W + x];
            return .{ @truncate(v), @truncate(v >> 8), @truncate(v >> 16) };
        }
    }.p;
    // v=0 at screen top (clip y=-1 maps to row 0 here since UV follows position). The four quadrant
    // centers pick the four in-slice texels of slice 0. TL(u<.5,v<.5)=red, TR=green, BL=blue, BR=white.
    const tl = at(dev, rt, 16, 16);
    const tr = at(dev, rt, 48, 16);
    const bl = at(dev, rt, 16, 48);
    const br = at(dev, rt, 48, 48);
    try std.testing.expect(tl[0] > 200 and tl[1] < 60 and tl[2] < 60); // red
    try std.testing.expect(tr[1] > 200 and tr[0] < 60 and tr[2] < 60); // green
    try std.testing.expect(bl[2] > 200 and bl[0] < 60 and bl[1] < 60); // blue
    try std.testing.expect(br[0] > 200 and br[1] > 200 and br[2] > 200); // white
}

test "ORACLE-3D-LINEAR: a sampler3D LINEAR-filters WITHIN a slice (bilinear) AND ACROSS slices (trilinear) on the NVIDIA GPU (skips without a GPU)" {
    // Completes the 3D-texture story now that within-slice addressing works (the coord-alignment fix):
    // Linear filtering blends the in-slice 2x2 (bilinear) and blends the two bracketing slices
    // (trilinear). Slice 0 = a 2x2 red/green/blue/white; slice 1 = all black. Sampling the center of
    // slice 0 (u=v=0.5, w=0.25) bilinearly averages the 4 in-slice texels -> gray ~127. Sampling
    // between the slices (w=0.5) trilinearly blends that slice-0 center gray with slice-1 black ->
    // ~half (~64). Two draws (a fixed coord per draw), read the center pixel.
    const glsl = @import("../../glsl.zig");
    const gpa = std.testing.allocator;
    const dev = NvDevice.create(gpa) catch return error.SkipZigTest;
    defer dev.deinit();
    const W: u32 = 64;
    const H: u32 = 64;

    const vs_src =
        \\attribute vec2 position;
        \\attribute vec3 aCoord;
        \\varying vec3 vCoord;
        \\void main() { gl_Position = vec4(position, 0.0, 1.0); vCoord = aCoord; }
    ;
    const fs_src =
        \\precision mediump float;
        \\uniform sampler3D uLut;
        \\varying vec3 vCoord;
        \\void main() { gl_FragColor = texture(uLut, vCoord); }
    ;
    const vs_bytes = try glsl.compileForStage(gpa, vs_src, .vertex);
    defer gpa.free(vs_bytes);
    const fs_bytes = try glsl.compileForStage(gpa, fs_src, .fragment);
    defer gpa.free(fs_bytes);

    const rt = try dev.createResource(.{ .image = .{ .width = W, .height = H, .format = .rgba8_unorm, .usage = .{ .render_target = true } } });
    defer dev.destroyResource(rt);
    const vs = try dev.createShaderModule(.{ .stage = .vertex, .code = vs_bytes });
    defer dev.destroyShaderModule(vs);
    const fs = try dev.createShaderModule(.{ .stage = .fragment, .code = fs_bytes });
    defer dev.destroyShaderModule(fs);

    // 2x2x2 LUT: slice 0 = red/green/blue/white (in-slice), slice 1 = all black.
    const lut = try dev.createResource(.{ .image = .{ .width = 2, .height = 2, .depth = 2, .format = .rgba8_unorm, .usage = .{ .sampled = true } } });
    defer dev.destroyResource(lut);
    {
        const vol = [_]u8{
            255, 0, 0, 255, 0, 255, 0, 255, // slice 0 row 0: red, green
            0, 0, 255, 255, 255, 255, 255, 255, // slice 0 row 1: blue, white
            0, 0, 0, 255, 0, 0, 0, 255, // slice 1 row 0: black, black
            0, 0, 0, 255, 0, 0, 0, 255, // slice 1 row 1: black, black
        };
        const dst = try dev.mapResource(lut);
        @memcpy(dst[0..vol.len], &vol);
    }

    const Vtx = extern struct { x: f32, y: f32, cx: f32, cy: f32, cz: f32 };
    const attrs = [_]hal.VertexAttribute{
        .{ .location = 0, .format = .r32g32_float, .offset = 0 },
        .{ .location = 1, .format = .r32g32b32_float, .offset = 8 },
    };
    const vbuf = try dev.createResource(.{ .buffer = .{ .size = 6 * @sizeOf(Vtx), .usage = .{ .vertex = true } } });
    defer dev.destroyResource(vbuf);
    const pipe = try dev.createPipeline(.{
        .vertex = vs,
        .fragment = fs,
        .vertex_layout = .{ .stride = @sizeOf(Vtx), .attributes = &attrs },
        .color_format = .rgba8_unorm,
    });
    defer dev.destroyPipeline(pipe);
    const ctx = try dev.createContext();
    defer ctx.deinit();

    // Render the full quad with a single (u,v,w) for every vertex, linear filter, and return the
    // center pixel. clamp-to-edge so the in-slice bilinear taps the true 4 texels at (0.5,0.5).
    const draw = struct {
        fn go(device: anytype, context: anytype, pl: anytype, texh: anytype, vb: anytype, target: anytype, cz: f32) [3]u8 {
            const V = extern struct { x: f32, y: f32, cx: f32, cy: f32, cz: f32 };
            const q = [6]V{
                .{ .x = -1, .y = -1, .cx = 0.5, .cy = 0.5, .cz = cz }, .{ .x = 1, .y = -1, .cx = 0.5, .cy = 0.5, .cz = cz },
                .{ .x = 1, .y = 1, .cx = 0.5, .cy = 0.5, .cz = cz },   .{ .x = -1, .y = -1, .cx = 0.5, .cy = 0.5, .cz = cz },
                .{ .x = 1, .y = 1, .cx = 0.5, .cy = 0.5, .cz = cz },   .{ .x = -1, .y = 1, .cx = 0.5, .cy = 0.5, .cz = cz },
            };
            @memcpy(std.mem.bytesAsSlice(V, device.mapResource(vb) catch unreachable)[0..6], &q);
            const cb = context.beginCommands() catch unreachable;
            defer cb.deinit();
            cb.setRenderTarget(target) catch unreachable;
            cb.clear(.{ .r = 0, .g = 0, .b = 0, .a = 1 }) catch unreachable;
            cb.bindPipeline(pl) catch unreachable;
            cb.bindTexture(.{ .binding = 2, .image = texh, .filter = .linear, .min_filter = .linear, .address_u = .clamp_to_edge, .address_v = .clamp_to_edge }) catch unreachable;
            cb.bindVertexBuffer(vb) catch unreachable;
            cb.draw(6, 0) catch unreachable;
            context.submit(cb) catch unreachable;
            const fb = std.mem.bytesAsSlice(u32, device.mapResource(target) catch unreachable);
            const v = fb[32 * W + 32];
            return .{ @truncate(v), @truncate(v >> 8), @truncate(v >> 16) };
        }
    }.go;

    // Within-slice bilinear: slice 0 center = average(red,green,blue,white) = ~(127,127,127) gray.
    const in_slice = draw(dev, ctx, pipe, lut, vbuf, rt, 0.25);
    inline for (0..3) |c| try std.testing.expectApproxEqAbs(@as(f32, 127), @as(f32, @floatFromInt(in_slice[c])), 40);

    // Cross-slice trilinear: w=0.5 blends slice-0-center gray (~127) with slice-1 black (0) -> ~64.
    const cross = draw(dev, ctx, pipe, lut, vbuf, rt, 0.5);
    inline for (0..3) |c| try std.testing.expectApproxEqAbs(@as(f32, 64), @as(f32, @floatFromInt(cross[c])), 40);
    // And it is clearly darker than the pure in-slice sample (proves cross-slice blending happened).
    try std.testing.expect(cross[0] < in_slice[0] and cross[1] < in_slice[1] and cross[2] < in_slice[2]);
}

test "ORACLE-2DARRAY: a sampler2DArray selects a LAYER by a raw index AND reads its in-layer (u,v) on the NVIDIA GPU, matching software (skips without a GPU)" {
    // Proves the whole sampler2DArray path: GLSL sampler2DArray -> vulcan-glsl (Dim 2D + Arrayed) ->
    // SPIR-V -> the nvidia isel's Array2D (=5) TEX with the (layer, u, v) coord + a TWO_D_ARRAY TIC.
    // Distinct from 3D: the third coord is a raw layer index (not normalized), one layer selected,
    // no cross-layer filtering. A 2x2x3 array: layer 0 = red/green/blue/white (in-layer), layer 1 =
    // all yellow, layer 2 = all magenta. Draw per (layer, corner) and cross-check the software golden.
    const glsl = @import("../../glsl.zig");
    const sw_sampler = @import("../software/sampler.zig");
    const gpa = std.testing.allocator;
    const dev = NvDevice.create(gpa) catch return error.SkipZigTest;
    defer dev.deinit();
    const W: u32 = 64;
    const H: u32 = 64;

    const vs_src =
        \\attribute vec2 position;
        \\attribute vec3 aCoord;
        \\varying vec3 vCoord;
        \\void main() { gl_Position = vec4(position, 0.0, 1.0); vCoord = aCoord; }
    ;
    const fs_src =
        \\precision mediump float;
        \\uniform sampler2DArray uArr;
        \\varying vec3 vCoord;
        \\void main() { gl_FragColor = texture(uArr, vCoord); }
    ;
    const vs_bytes = try glsl.compileForStage(gpa, vs_src, .vertex);
    defer gpa.free(vs_bytes);
    const fs_bytes = try glsl.compileForStage(gpa, fs_src, .fragment);
    defer gpa.free(fs_bytes);

    const rt = try dev.createResource(.{ .image = .{ .width = W, .height = H, .format = .rgba8_unorm, .usage = .{ .render_target = true } } });
    defer dev.destroyResource(rt);
    const vs = try dev.createShaderModule(.{ .stage = .vertex, .code = vs_bytes });
    defer dev.destroyShaderModule(vs);
    const fs = try dev.createShaderModule(.{ .stage = .fragment, .code = fs_bytes });
    defer dev.destroyShaderModule(fs);

    // 2x2x3 array, layer-major. Layer 0: red/green/blue/white in-layer; layer 1: yellow; layer 2: magenta.
    const vol = [_]u8{
        255, 0, 0, 255, 0, 255, 0, 255, // layer 0 row 0: red, green
        0, 0, 255, 255, 255, 255, 255, 255, // layer 0 row 1: blue, white
        255, 255, 0, 255, 255, 255, 0, 255, // layer 1: yellow
        255, 255, 0, 255, 255, 255, 0, 255,
        255, 0, 255, 255, 255, 0, 255, 255, // layer 2: magenta
        255, 0, 255, 255, 255, 0, 255, 255,
    };
    const arr = try dev.createResource(.{ .image = .{ .width = 2, .height = 2, .depth = 3, .array = true, .format = .rgba8_unorm, .usage = .{ .sampled = true } } });
    defer dev.destroyResource(arr);
    @memcpy((try dev.mapResource(arr))[0..vol.len], &vol);

    // Software golden for the same descriptor (layer-major, is_2darray).
    var sw_desc = sw_sampler.TexDesc{ .pixels = &vol, .width = 2, .height = 2, .pitch = 8, .filter = .nearest, .is_2darray = true, .depth = 3 };

    const Vtx = extern struct { x: f32, y: f32, cx: f32, cy: f32, cz: f32 };
    const attrs = [_]hal.VertexAttribute{
        .{ .location = 0, .format = .r32g32_float, .offset = 0 },
        .{ .location = 1, .format = .r32g32b32_float, .offset = 8 },
    };
    const vbuf = try dev.createResource(.{ .buffer = .{ .size = 6 * @sizeOf(Vtx), .usage = .{ .vertex = true } } });
    defer dev.destroyResource(vbuf);
    const pipe = try dev.createPipeline(.{
        .vertex = vs,
        .fragment = fs,
        .vertex_layout = .{ .stride = @sizeOf(Vtx), .attributes = &attrs },
        .color_format = .rgba8_unorm,
    });
    defer dev.destroyPipeline(pipe);
    const ctx = try dev.createContext();
    defer ctx.deinit();

    // Draw the full quad with (u,v) spanning the screen and a fixed layer, nearest. Read a pixel at
    // screen (px,py) -> that (u,v) of `layer`. Returns the GPU RGB.
    const draw = struct {
        fn go(device: anytype, context: anytype, pl: anytype, texh: anytype, vb: anytype, target: anytype, layer: f32, px: usize, py: usize) [3]u8 {
            const V = extern struct { x: f32, y: f32, cx: f32, cy: f32, cz: f32 };
            const q = [6]V{
                .{ .x = -1, .y = -1, .cx = 0, .cy = 0, .cz = layer }, .{ .x = 1, .y = -1, .cx = 1, .cy = 0, .cz = layer },
                .{ .x = 1, .y = 1, .cx = 1, .cy = 1, .cz = layer },   .{ .x = -1, .y = -1, .cx = 0, .cy = 0, .cz = layer },
                .{ .x = 1, .y = 1, .cx = 1, .cy = 1, .cz = layer },   .{ .x = -1, .y = 1, .cx = 0, .cy = 1, .cz = layer },
            };
            @memcpy(std.mem.bytesAsSlice(V, device.mapResource(vb) catch unreachable)[0..6], &q);
            const cb = context.beginCommands() catch unreachable;
            defer cb.deinit();
            cb.setRenderTarget(target) catch unreachable;
            cb.clear(.{ .r = 0, .g = 0, .b = 0, .a = 1 }) catch unreachable;
            cb.bindPipeline(pl) catch unreachable;
            cb.bindTexture(.{ .binding = 2, .image = texh, .filter = .nearest, .address_u = .clamp_to_edge, .address_v = .clamp_to_edge }) catch unreachable;
            cb.bindVertexBuffer(vb) catch unreachable;
            cb.draw(6, 0) catch unreachable;
            context.submit(cb) catch unreachable;
            const fb = std.mem.bytesAsSlice(u32, device.mapResource(target) catch unreachable);
            const val = fb[py * W + px];
            return .{ @truncate(val), @truncate(val >> 8), @truncate(val >> 16) };
        }
    }.go;

    const Case = struct { layer: f32, px: usize, py: usize, u: f32, v: f32 };
    const cases = [_]Case{
        .{ .layer = 0, .px = 16, .py = 16, .u = 0.25, .v = 0.25 }, // layer 0 TL -> red
        .{ .layer = 0, .px = 48, .py = 48, .u = 0.75, .v = 0.75 }, // layer 0 BR -> white
        .{ .layer = 1, .px = 32, .py = 32, .u = 0.5, .v = 0.5 }, // layer 1 -> yellow
        .{ .layer = 2, .px = 32, .py = 32, .u = 0.5, .v = 0.5 }, // layer 2 -> magenta
    };
    for (cases) |c| {
        const g = draw(dev, ctx, pipe, arr, vbuf, rt, c.layer, c.px, c.py);
        var gold: [4]f32 = undefined;
        sw_sampler.sampleTextureCube(&sw_desc, c.u, c.v, c.layer, 0, &gold);
        inline for (0..3) |ch| try std.testing.expectApproxEqAbs(gold[ch] * 255.0, @as(f32, @floatFromInt(g[ch])), 12);
    }
}

test "ORACLE-CUBE-TEX: a samplerCube on the NVIDIA GPU selects the correct GL face AND within-face texel from the direction (cube lowered to a 6-face-wide 2D atlas), matching the software convention (skips without a GPU)" {
    const glsl = @import("../../glsl.zig");
    const gpa = std.testing.allocator;
    const dev = NvDevice.create(gpa) catch return error.SkipZigTest;
    defer dev.deinit();
    const W: u32 = 64;

    const vs_src =
        \\attribute vec2 position;
        \\attribute vec3 aDir;
        \\varying vec3 vDir;
        \\void main() { gl_Position = vec4(position, 0.0, 1.0); vDir = aDir; }
    ;
    const fs_src =
        \\precision mediump float;
        \\uniform samplerCube uSky;
        \\varying vec3 vDir;
        \\void main() { gl_FragColor = textureCube(uSky, vDir); }
    ;
    const vs_bytes = try glsl.compileForStage(gpa, vs_src, .vertex);
    defer gpa.free(vs_bytes);
    const fs_bytes = try glsl.compileForStage(gpa, fs_src, .fragment);
    defer gpa.free(fs_bytes);

    const rt = try dev.createResource(.{ .image = .{ .width = W, .height = W, .format = .rgba8_unorm, .usage = .{ .render_target = true } } });
    defer dev.destroyResource(rt);
    const vs = try dev.createShaderModule(.{ .stage = .vertex, .code = vs_bytes });
    defer dev.destroyShaderModule(vs);
    const fs = try dev.createShaderModule(.{ .stage = .fragment, .code = fs_bytes });
    defer dev.destroyShaderModule(fs);

    // A 16x16-face cubemap where each texel encodes its position + face: R = px gradient,
    // G = py gradient, B = face marker. So the sampled color reveals both the face (B) and the
    // within-face texel (R,G). The whole direction->(face,u,v) convention is checkable.
    const FW: u32 = 16;
    const cube = try dev.createResource(.{ .image = .{ .width = FW, .height = FW, .format = .rgba8_unorm, .cube = true, .usage = .{ .sampled = true } } });
    defer dev.destroyResource(cube);
    {
        const dst = try dev.mapResource(cube);
        for (0..6) |f| {
            for (0..FW) |py| {
                for (0..FW) |px| {
                    const o = (f * FW * FW + py * FW + px) * 4;
                    dst[o + 0] = @intCast(px * 17); // R = u column (0..255)
                    dst[o + 1] = @intCast(py * 17); // G = v row
                    dst[o + 2] = @intCast(f * 42); // B = face marker
                    dst[o + 3] = 255;
                }
            }
        }
    }

    // Reference GL cube convention (byte-identical to software cubeFaceUv).
    const Ref = struct {
        fn uv(rx: f32, ry: f32, rz: f32) struct { face: u32, u: f32, v: f32 } {
            const ax = @abs(rx);
            const ay = @abs(ry);
            const az = @abs(rz);
            var face: u32 = 0;
            var sc: f32 = 0;
            var tc: f32 = 0;
            var ma: f32 = 1;
            if (ax >= ay and ax >= az) {
                ma = ax;
                if (rx >= 0) {
                    face = 0;
                    sc = -rz;
                    tc = -ry;
                } else {
                    face = 1;
                    sc = rz;
                    tc = -ry;
                }
            } else if (ay >= az) {
                ma = ay;
                if (ry >= 0) {
                    face = 2;
                    sc = rx;
                    tc = rz;
                } else {
                    face = 3;
                    sc = rx;
                    tc = -rz;
                }
            } else {
                ma = az;
                if (rz >= 0) {
                    face = 4;
                    sc = rx;
                    tc = -ry;
                } else {
                    face = 5;
                    sc = -rx;
                    tc = -ry;
                }
            }
            if (ma == 0) ma = 1;
            return .{ .face = face, .u = (sc / ma + 1.0) * 0.5, .v = (tc / ma + 1.0) * 0.5 };
        }
    };

    const Vtx = extern struct { x: f32, y: f32, dx: f32, dy: f32, dz: f32 };
    const attrs = [_]hal.VertexAttribute{
        .{ .location = 0, .format = .r32g32_float, .offset = 0 },
        .{ .location = 1, .format = .r32g32b32_float, .offset = 8 },
    };
    const vbuf = try dev.createResource(.{ .buffer = .{ .size = 6 * @sizeOf(Vtx), .usage = .{ .vertex = true } } });
    defer dev.destroyResource(vbuf);
    const pipe = try dev.createPipeline(.{
        .vertex = vs,
        .fragment = fs,
        .vertex_layout = .{ .stride = @sizeOf(Vtx), .attributes = &attrs },
        .color_format = .rgba8_unorm,
    });
    defer dev.destroyPipeline(pipe);

    // Off-center directions (each unambiguously major on one axis, u,v away from face edges).
    const dirs = [_][3]f32{
        .{ 1.0, 0.4, -0.5 },  .{ -1.0, -0.3, 0.6 }, .{ 0.2, 1.0, 0.5 },
        .{ 0.3, -1.0, -0.4 }, .{ 0.5, 0.4, 1.0 },   .{ -0.5, 0.3, -1.0 },
    };
    var fails: u32 = 0;
    for (dirs) |d| {
        const q = [6]Vtx{
            .{ .x = -1, .y = -1, .dx = d[0], .dy = d[1], .dz = d[2] }, .{ .x = 1, .y = -1, .dx = d[0], .dy = d[1], .dz = d[2] },
            .{ .x = 1, .y = 1, .dx = d[0], .dy = d[1], .dz = d[2] },   .{ .x = -1, .y = -1, .dx = d[0], .dy = d[1], .dz = d[2] },
            .{ .x = 1, .y = 1, .dx = d[0], .dy = d[1], .dz = d[2] },   .{ .x = -1, .y = 1, .dx = d[0], .dy = d[1], .dz = d[2] },
        };
        @memcpy(std.mem.bytesAsSlice(Vtx, try dev.mapResource(vbuf))[0..6], &q);
        const dctx = try dev.createContext();
        defer dctx.deinit();
        const cb = try dctx.beginCommands();
        defer cb.deinit();
        try cb.setRenderTarget(rt);
        try cb.clear(.{ .r = 0, .g = 0, .b = 0, .a = 1 });
        try cb.bindPipeline(pipe);
        try cb.bindTexture(.{ .binding = 2, .image = cube, .filter = .nearest, .address_u = .clamp_to_edge, .address_v = .clamp_to_edge });
        try cb.bindVertexBuffer(vbuf);
        try cb.draw(6, 0);
        try dctx.submit(cb);
        const fb = std.mem.bytesAsSlice(u32, try dev.mapResource(rt));
        const vv = fb[32 * W + 32];
        const gr: [3]u8 = .{ @truncate(vv), @truncate(vv >> 8), @truncate(vv >> 16) };
        // Expected: reference (face,u,v) -> nearest texel color.
        const r = Ref.uv(d[0], d[1], d[2]);
        const col: u32 = @min(@as(u32, @intFromFloat(r.u * FW)), FW - 1);
        const row: u32 = @min(@as(u32, @intFromFloat(r.v * FW)), FW - 1);
        const exp: [3]u8 = .{ @intCast(col * 17), @intCast(row * 17), @intCast(r.face * 42) };
        // Tolerance 20 (~1 texel of 17) absorbs the nearest u*W floor rounding at texel boundaries.
        const ok = @abs(@as(i32, gr[0]) - exp[0]) <= 20 and @abs(@as(i32, gr[1]) - exp[1]) <= 20 and @abs(@as(i32, gr[2]) - exp[2]) <= 20;
        if (!ok) {
            fails += 1;
            std.debug.print("[CUBE-TEX] dir ({d:.1},{d:.1},{d:.1}) face{d} gpu=({d},{d},{d}) exp=({d},{d},{d}) MISMATCH\n", .{ d[0], d[1], d[2], r.face, gr[0], gr[1], gr[2], exp[0], exp[1], exp[2] });
        }
    }
    try std.testing.expectEqual(@as(u32, 0), fails);
}

test "ORACLE-CUBE-SEAM: LINEAR sampling near a cube face edge does NOT bleed into the neighbouring face on the NVIDIA GPU (per-face clamp-to-edge via the atlas half-texel clamp) (skips without a GPU)" {
    const glsl = @import("../../glsl.zig");
    const gpa = std.testing.allocator;
    const dev = NvDevice.create(gpa) catch return error.SkipZigTest;
    defer dev.deinit();
    const W: u32 = 64;

    const vs_src =
        \\attribute vec2 position;
        \\attribute vec3 aDir;
        \\varying vec3 vDir;
        \\void main() { gl_Position = vec4(position, 0.0, 1.0); vDir = aDir; }
    ;
    const fs_src =
        \\precision mediump float;
        \\uniform samplerCube uSky;
        \\varying vec3 vDir;
        \\void main() { gl_FragColor = textureCube(uSky, vDir); }
    ;
    const vs_bytes = try glsl.compileForStage(gpa, vs_src, .vertex);
    defer gpa.free(vs_bytes);
    const fs_bytes = try glsl.compileForStage(gpa, fs_src, .fragment);
    defer gpa.free(fs_bytes);

    const rt = try dev.createResource(.{ .image = .{ .width = W, .height = W, .format = .rgba8_unorm, .usage = .{ .render_target = true } } });
    defer dev.destroyResource(rt);
    const vs = try dev.createShaderModule(.{ .stage = .vertex, .code = vs_bytes });
    defer dev.destroyShaderModule(vs);
    const fs = try dev.createShaderModule(.{ .stage = .fragment, .code = fs_bytes });
    defer dev.destroyShaderModule(fs);

    // Solid distinct faces: +Z=magenta(255,0,255), its atlas neighbour -Y(face3)=yellow(255,255,0).
    const FW: u32 = 16;
    const face_rgb = [6][4]u8{ .{ 255, 0, 0, 255 }, .{ 0, 255, 0, 255 }, .{ 0, 0, 255, 255 }, .{ 255, 255, 0, 255 }, .{ 255, 0, 255, 255 }, .{ 0, 255, 255, 255 } };
    const cube = try dev.createResource(.{ .image = .{ .width = FW, .height = FW, .format = .rgba8_unorm, .cube = true, .usage = .{ .sampled = true } } });
    defer dev.destroyResource(cube);
    {
        const dst = try dev.mapResource(cube);
        const px = FW * FW;
        for (0..6) |f| {
            for (0..px) |p| @memcpy(dst[(f * px + p) * 4 ..][0..4], &face_rgb[f]);
        }
    }

    const Vtx = extern struct { x: f32, y: f32, dx: f32, dy: f32, dz: f32 };
    const attrs = [_]hal.VertexAttribute{
        .{ .location = 0, .format = .r32g32_float, .offset = 0 },
        .{ .location = 1, .format = .r32g32b32_float, .offset = 8 },
    };
    const vbuf = try dev.createResource(.{ .buffer = .{ .size = 6 * @sizeOf(Vtx), .usage = .{ .vertex = true } } });
    defer dev.destroyResource(vbuf);
    const pipe = try dev.createPipeline(.{
        .vertex = vs,
        .fragment = fs,
        .vertex_layout = .{ .stride = @sizeOf(Vtx), .attributes = &attrs },
        .color_format = .rgba8_unorm,
    });
    defer dev.destroyPipeline(pipe);

    // A +Z-major direction whose within-face u lands ~0.01 (deep in the left edge texel), so a linear
    // tap at the raw coord would straddle the atlas column boundary into face3 (yellow). The clamp
    // keeps it inside face 4 -> pure magenta, no yellow bleed (G stays low).
    const d = [3]f32{ -0.98, 0.0, 1.0 };
    const q = [6]Vtx{
        .{ .x = -1, .y = -1, .dx = d[0], .dy = d[1], .dz = d[2] }, .{ .x = 1, .y = -1, .dx = d[0], .dy = d[1], .dz = d[2] },
        .{ .x = 1, .y = 1, .dx = d[0], .dy = d[1], .dz = d[2] },   .{ .x = -1, .y = -1, .dx = d[0], .dy = d[1], .dz = d[2] },
        .{ .x = 1, .y = 1, .dx = d[0], .dy = d[1], .dz = d[2] },   .{ .x = -1, .y = 1, .dx = d[0], .dy = d[1], .dz = d[2] },
    };
    @memcpy(std.mem.bytesAsSlice(Vtx, try dev.mapResource(vbuf))[0..6], &q);
    const ctx = try dev.createContext();
    defer ctx.deinit();
    const cb = try ctx.beginCommands();
    defer cb.deinit();
    try cb.setRenderTarget(rt);
    try cb.clear(.{ .r = 0, .g = 0, .b = 0, .a = 1 });
    try cb.bindPipeline(pipe);
    try cb.bindTexture(.{ .binding = 2, .image = cube, .filter = .linear, .address_u = .clamp_to_edge, .address_v = .clamp_to_edge });
    try cb.bindVertexBuffer(vbuf);
    try cb.draw(6, 0);
    try ctx.submit(cb);

    const fb = std.mem.bytesAsSlice(u32, try dev.mapResource(rt));
    const vv = fb[32 * W + 32];
    const c = [3]u8{ @truncate(vv), @truncate(vv >> 8), @truncate(vv >> 16) };
    // Magenta face, no yellow bleed: R high, G low (bleed from yellow would push G up), B high.
    try std.testing.expect(c[0] > 200 and c[2] > 200);
    try std.testing.expect(c[1] < 60);
}

test "ORACLE-2D-MIPMAP: implicit-LOD minification of a mipmapped 2D texture selects a higher mip on the NVIDIA GPU (fragment-stage derivatives work with deriv_mode = Auto on Blackwell) (skips without a GPU)" {
    const glsl = @import("../../glsl.zig");
    const gpa = std.testing.allocator;
    const dev = NvDevice.create(gpa) catch return error.SkipZigTest;
    defer dev.deinit();
    const W: u32 = 64;

    const vs_src =
        \\attribute vec2 position;
        \\attribute vec2 aUV;
        \\varying vec2 vUV;
        \\void main() { gl_Position = vec4(position, 0.0, 1.0); vUV = aUV; }
    ;
    const fs_src =
        \\precision mediump float;
        \\uniform sampler2D uTex;
        \\varying vec2 vUV;
        \\void main() { gl_FragColor = texture2D(uTex, vUV); }
    ;
    const vs_bytes = try glsl.compileForStage(gpa, vs_src, .vertex);
    defer gpa.free(vs_bytes);
    const fs_bytes = try glsl.compileForStage(gpa, fs_src, .fragment);
    defer gpa.free(fs_bytes);

    const rt = try dev.createResource(.{ .image = .{ .width = W, .height = W, .format = .rgba8_unorm, .usage = .{ .render_target = true } } });
    defer dev.destroyResource(rt);
    const vs = try dev.createShaderModule(.{ .stage = .vertex, .code = vs_bytes });
    defer dev.destroyShaderModule(vs);
    const fs = try dev.createShaderModule(.{ .stage = .fragment, .code = fs_bytes });
    defer dev.destroyShaderModule(fs);

    // A 16x16 texture with 2 mips: level 0 = red, level 1 (8x8) = green. A full-screen quad whose UV
    // spans 0..8 minifies ~2 texels/pixel -> implicit LOD ~1 -> the sampler picks the green mip. This
    // confirms fragment-stage screen-space derivatives (deriv_mode = Auto) drive the implicit LOD on
    // Blackwell (NAK only forces deriv_mode = DerivXY for COMPUTE-stage sampling, not fragment).
    const TW: u32 = 16;
    const tex = try dev.createResource(.{ .image = .{ .width = TW, .height = TW, .format = .rgba8_unorm, .mip_levels = 2, .usage = .{ .sampled = true } } });
    defer dev.destroyResource(tex);
    {
        const dst = try dev.mapResource(tex);
        for (0..TW * TW) |p| {
            dst[p * 4 + 0] = 255;
            dst[p * 4 + 1] = 0;
            dst[p * 4 + 2] = 0;
            dst[p * 4 + 3] = 255;
        }
        const l1 = TW * TW * 4;
        for (0..(TW / 2) * (TW / 2)) |p| {
            dst[l1 + p * 4 + 0] = 0;
            dst[l1 + p * 4 + 1] = 255;
            dst[l1 + p * 4 + 2] = 0;
            dst[l1 + p * 4 + 3] = 255;
        }
    }

    const Vtx = extern struct { x: f32, y: f32, u: f32, v: f32 };
    const attrs = [_]hal.VertexAttribute{
        .{ .location = 0, .format = .r32g32_float, .offset = 0 },
        .{ .location = 1, .format = .r32g32_float, .offset = 8 },
    };
    const vbuf = try dev.createResource(.{ .buffer = .{ .size = 6 * @sizeOf(Vtx), .usage = .{ .vertex = true } } });
    defer dev.destroyResource(vbuf);
    const pipe = try dev.createPipeline(.{
        .vertex = vs,
        .fragment = fs,
        .vertex_layout = .{ .stride = @sizeOf(Vtx), .attributes = &attrs },
        .color_format = .rgba8_unorm,
    });
    defer dev.destroyPipeline(pipe);
    const ctx = try dev.createContext();
    defer ctx.deinit();

    // UV 0..8 across the full-screen quad (minifies).
    const q = [6]Vtx{
        .{ .x = -1, .y = -1, .u = 0, .v = 0 }, .{ .x = 1, .y = -1, .u = 8, .v = 0 }, .{ .x = 1, .y = 1, .u = 8, .v = 8 },
        .{ .x = -1, .y = -1, .u = 0, .v = 0 }, .{ .x = 1, .y = 1, .u = 8, .v = 8 },  .{ .x = -1, .y = 1, .u = 0, .v = 8 },
    };
    @memcpy(std.mem.bytesAsSlice(Vtx, try dev.mapResource(vbuf))[0..6], &q);
    const cb = try ctx.beginCommands();
    defer cb.deinit();
    try cb.setRenderTarget(rt);
    try cb.clear(.{ .r = 0, .g = 0, .b = 0, .a = 1 });
    try cb.bindPipeline(pipe);
    try cb.bindTexture(.{ .binding = 2, .image = tex, .filter = .nearest, .address_u = .repeat, .address_v = .repeat, .mip_filter = .nearest });
    try cb.bindVertexBuffer(vbuf);
    try cb.draw(6, 0);
    try ctx.submit(cb);

    const fb = std.mem.bytesAsSlice(u32, try dev.mapResource(rt));
    const vv = fb[32 * W + 32];
    const c = [3]u8{ @truncate(vv), @truncate(vv >> 8), @truncate(vv >> 16) };
    // Minified -> the green mip (level 1), not the red base.
    try std.testing.expect(c[1] > 200 and c[0] < 60);
}

test "ORACLE-BASE-LEVEL: GL_TEXTURE_BASE_LEVEL clamps the sampled mip on the NVIDIA GPU (TIC RES_VIEW_MIN_MIP_LEVEL) (skips without a GPU)" {
    const glsl = @import("../../glsl.zig");
    const gpa = std.testing.allocator;
    const dev = NvDevice.create(gpa) catch return error.SkipZigTest;
    defer dev.deinit();
    const W: u32 = 64;

    const vs_src =
        \\attribute vec2 position;
        \\attribute vec2 aUV;
        \\varying vec2 vUV;
        \\void main() { gl_Position = vec4(position, 0.0, 1.0); vUV = aUV; }
    ;
    const fs_src =
        \\precision mediump float;
        \\uniform sampler2D uTex;
        \\varying vec2 vUV;
        \\void main() { gl_FragColor = texture2D(uTex, vUV); }
    ;
    const vs_bytes = try glsl.compileForStage(gpa, vs_src, .vertex);
    defer gpa.free(vs_bytes);
    const fs_bytes = try glsl.compileForStage(gpa, fs_src, .fragment);
    defer gpa.free(fs_bytes);
    const rt = try dev.createResource(.{ .image = .{ .width = W, .height = W, .format = .rgba8_unorm, .usage = .{ .render_target = true } } });
    defer dev.destroyResource(rt);
    const vs = try dev.createShaderModule(.{ .stage = .vertex, .code = vs_bytes });
    defer dev.destroyShaderModule(vs);
    const fs = try dev.createShaderModule(.{ .stage = .fragment, .code = fs_bytes });
    defer dev.destroyShaderModule(fs);

    // 16x16, 2 mips: level 0 = red, level 1 (8x8) = green.
    const TW: u32 = 16;
    const tex = try dev.createResource(.{ .image = .{ .width = TW, .height = TW, .format = .rgba8_unorm, .mip_levels = 2, .usage = .{ .sampled = true } } });
    defer dev.destroyResource(tex);
    {
        const dst = try dev.mapResource(tex);
        for (0..TW * TW) |p| {
            dst[p * 4 + 0] = 255;
            dst[p * 4 + 1] = 0;
            dst[p * 4 + 2] = 0;
            dst[p * 4 + 3] = 255;
        }
        const l1 = TW * TW * 4;
        for (0..(TW / 2) * (TW / 2)) |p| {
            dst[l1 + p * 4 + 0] = 0;
            dst[l1 + p * 4 + 1] = 255;
            dst[l1 + p * 4 + 2] = 0;
            dst[l1 + p * 4 + 3] = 255;
        }
    }

    const Vtx = extern struct { x: f32, y: f32, u: f32, v: f32 };
    const attrs = [_]hal.VertexAttribute{
        .{ .location = 0, .format = .r32g32_float, .offset = 0 },
        .{ .location = 1, .format = .r32g32_float, .offset = 8 },
    };
    const vbuf = try dev.createResource(.{ .buffer = .{ .size = 6 * @sizeOf(Vtx), .usage = .{ .vertex = true } } });
    defer dev.destroyResource(vbuf);
    const pipe = try dev.createPipeline(.{
        .vertex = vs,
        .fragment = fs,
        .vertex_layout = .{ .stride = @sizeOf(Vtx), .attributes = &attrs },
        .color_format = .rgba8_unorm,
    });
    defer dev.destroyPipeline(pipe);
    const ctx = try dev.createContext();
    defer ctx.deinit();

    // UV 0..1 (magnified, implicit LOD ~0): level 0 (red) normally. base_level 1 clamps to green.
    const q = [6]Vtx{
        .{ .x = -1, .y = -1, .u = 0, .v = 0 }, .{ .x = 1, .y = -1, .u = 1, .v = 0 }, .{ .x = 1, .y = 1, .u = 1, .v = 1 },
        .{ .x = -1, .y = -1, .u = 0, .v = 0 }, .{ .x = 1, .y = 1, .u = 1, .v = 1 },  .{ .x = -1, .y = 1, .u = 0, .v = 1 },
    };
    @memcpy(std.mem.bytesAsSlice(Vtx, try dev.mapResource(vbuf))[0..6], &q);

    // base_level 0: magnified sample reads the red base level.
    {
        const cb = try ctx.beginCommands();
        defer cb.deinit();
        try cb.setRenderTarget(rt);
        try cb.clear(.{ .r = 0, .g = 0, .b = 0, .a = 1 });
        try cb.bindPipeline(pipe);
        try cb.bindTexture(.{ .binding = 2, .image = tex, .filter = .nearest, .address_u = .repeat, .address_v = .repeat, .mip_filter = .nearest, .base_level = 0 });
        try cb.bindVertexBuffer(vbuf);
        try cb.draw(6, 0);
        ctx.submit(cb) catch return error.SkipZigTest;
        const vv = std.mem.bytesAsSlice(u32, try dev.mapResource(rt))[32 * W + 32];
        try std.testing.expect(@as(u8, @truncate(vv)) > 200 and @as(u8, @truncate(vv >> 8)) < 60); // red base
    }
    // base_level 1: the TIC RES_VIEW_MIN clamps the sampled level to 1 -> green, even magnified.
    {
        const cb = try ctx.beginCommands();
        defer cb.deinit();
        try cb.setRenderTarget(rt);
        try cb.clear(.{ .r = 0, .g = 0, .b = 0, .a = 1 });
        try cb.bindPipeline(pipe);
        try cb.bindTexture(.{ .binding = 2, .image = tex, .filter = .nearest, .address_u = .repeat, .address_v = .repeat, .mip_filter = .nearest, .base_level = 1 });
        try cb.bindVertexBuffer(vbuf);
        try cb.draw(6, 0);
        ctx.submit(cb) catch return error.SkipZigTest;
        const vv = std.mem.bytesAsSlice(u32, try dev.mapResource(rt))[32 * W + 32];
        try std.testing.expect(@as(u8, @truncate(vv >> 8)) > 200 and @as(u8, @truncate(vv)) < 60); // green (clamped to level 1)
    }
}

test "ORACLE-SWIZZLE: GL_TEXTURE_SWIZZLE remaps sampled channels on the NVIDIA GPU (TIC X/Y/Z/W_SOURCE) (skips without a GPU)" {
    const glsl = @import("../../glsl.zig");
    const gpa = std.testing.allocator;
    const dev = NvDevice.create(gpa) catch return error.SkipZigTest;
    defer dev.deinit();
    const W: u32 = 64;
    const vs_src =
        \\attribute vec2 position;
        \\attribute vec2 aUV;
        \\varying vec2 vUV;
        \\void main() { gl_Position = vec4(position, 0.0, 1.0); vUV = aUV; }
    ;
    const fs_src =
        \\precision mediump float;
        \\uniform sampler2D uTex;
        \\varying vec2 vUV;
        \\void main() { gl_FragColor = texture2D(uTex, vUV); }
    ;
    const vs_bytes = try glsl.compileForStage(gpa, vs_src, .vertex);
    defer gpa.free(vs_bytes);
    const fs_bytes = try glsl.compileForStage(gpa, fs_src, .fragment);
    defer gpa.free(fs_bytes);
    const rt = try dev.createResource(.{ .image = .{ .width = W, .height = W, .format = .rgba8_unorm, .usage = .{ .render_target = true } } });
    defer dev.destroyResource(rt);
    const vs = try dev.createShaderModule(.{ .stage = .vertex, .code = vs_bytes });
    defer dev.destroyShaderModule(vs);
    const fs = try dev.createShaderModule(.{ .stage = .fragment, .code = fs_bytes });
    defer dev.destroyShaderModule(fs);
    // A 2x2 texture: R=200, G=100, B=40, A=255.
    const tex = try dev.createResource(.{ .image = .{ .width = 2, .height = 2, .format = .rgba8_unorm, .usage = .{ .sampled = true } } });
    defer dev.destroyResource(tex);
    {
        const dst = try dev.mapResource(tex);
        for (0..4) |p| {
            dst[p * 4 + 0] = 200;
            dst[p * 4 + 1] = 100;
            dst[p * 4 + 2] = 40;
            dst[p * 4 + 3] = 255;
        }
    }
    const Vtx = extern struct { x: f32, y: f32, u: f32, v: f32 };
    const attrs = [_]hal.VertexAttribute{
        .{ .location = 0, .format = .r32g32_float, .offset = 0 },
        .{ .location = 1, .format = .r32g32_float, .offset = 8 },
    };
    const vbuf = try dev.createResource(.{ .buffer = .{ .size = 6 * @sizeOf(Vtx), .usage = .{ .vertex = true } } });
    defer dev.destroyResource(vbuf);
    const pipe = try dev.createPipeline(.{
        .vertex = vs,
        .fragment = fs,
        .vertex_layout = .{ .stride = @sizeOf(Vtx), .attributes = &attrs },
        .color_format = .rgba8_unorm,
    });
    defer dev.destroyPipeline(pipe);
    const ctx = try dev.createContext();
    defer ctx.deinit();
    const q = [6]Vtx{
        .{ .x = -1, .y = -1, .u = 0, .v = 0 }, .{ .x = 1, .y = -1, .u = 1, .v = 0 }, .{ .x = 1, .y = 1, .u = 1, .v = 1 },
        .{ .x = -1, .y = -1, .u = 0, .v = 0 }, .{ .x = 1, .y = 1, .u = 1, .v = 1 },  .{ .x = -1, .y = 1, .u = 0, .v = 1 },
    };
    @memcpy(std.mem.bytesAsSlice(Vtx, try dev.mapResource(vbuf))[0..6], &q);
    // Broadcast R to R/G/B (the font-coverage idiom): the TIC X/Y/Z_SOURCE all read IN_R.
    const cb = try ctx.beginCommands();
    defer cb.deinit();
    try cb.setRenderTarget(rt);
    try cb.clear(.{ .r = 0, .g = 0, .b = 0, .a = 1 });
    try cb.bindPipeline(pipe);
    try cb.bindTexture(.{ .binding = 2, .image = tex, .filter = .nearest, .address_u = .repeat, .address_v = .repeat, .mip_filter = .none, .swizzle = .{ .r, .r, .r, .a } });
    try cb.bindVertexBuffer(vbuf);
    try cb.draw(6, 0);
    ctx.submit(cb) catch return error.SkipZigTest;
    const vv = std.mem.bytesAsSlice(u32, try dev.mapResource(rt))[32 * W + 32];
    const r: u8 = @truncate(vv);
    const g: u8 = @truncate(vv >> 8);
    const b: u8 = @truncate(vv >> 16);
    // All three read R (~200): green jumped from ~100 to ~200, blue from ~40 to ~200.
    try std.testing.expect(r > 180 and g > 180 and b > 180);
}

test "ORACLE-LOD-BIAS: GL_TEXTURE_LOD_BIAS pushes the sampled mip coarser on the NVIDIA GPU (TSC MIP_LOD_BIAS) (skips without a GPU)" {
    const glsl = @import("../../glsl.zig");
    const gpa = std.testing.allocator;
    const dev = NvDevice.create(gpa) catch return error.SkipZigTest;
    defer dev.deinit();
    const W: u32 = 64;
    const vs_src =
        \\attribute vec2 position;
        \\attribute vec2 aUV;
        \\varying vec2 vUV;
        \\void main() { gl_Position = vec4(position, 0.0, 1.0); vUV = aUV; }
    ;
    const fs_src =
        \\precision mediump float;
        \\uniform sampler2D uTex;
        \\varying vec2 vUV;
        \\void main() { gl_FragColor = texture2D(uTex, vUV); }
    ;
    const vs_bytes = try glsl.compileForStage(gpa, vs_src, .vertex);
    defer gpa.free(vs_bytes);
    const fs_bytes = try glsl.compileForStage(gpa, fs_src, .fragment);
    defer gpa.free(fs_bytes);
    const rt = try dev.createResource(.{ .image = .{ .width = W, .height = W, .format = .rgba8_unorm, .usage = .{ .render_target = true } } });
    defer dev.destroyResource(rt);
    const vs = try dev.createShaderModule(.{ .stage = .vertex, .code = vs_bytes });
    defer dev.destroyShaderModule(vs);
    const fs = try dev.createShaderModule(.{ .stage = .fragment, .code = fs_bytes });
    defer dev.destroyShaderModule(fs);
    // 16x16, 2 mips: L0 = red, L1 (8x8) = green.
    const TW: u32 = 16;
    const tex = try dev.createResource(.{ .image = .{ .width = TW, .height = TW, .format = .rgba8_unorm, .mip_levels = 2, .usage = .{ .sampled = true } } });
    defer dev.destroyResource(tex);
    {
        const dst = try dev.mapResource(tex);
        for (0..TW * TW) |p| {
            dst[p * 4 + 0] = 255;
            dst[p * 4 + 1] = 0;
            dst[p * 4 + 2] = 0;
            dst[p * 4 + 3] = 255;
        }
        const l1 = TW * TW * 4;
        for (0..(TW / 2) * (TW / 2)) |p| {
            dst[l1 + p * 4 + 0] = 0;
            dst[l1 + p * 4 + 1] = 255;
            dst[l1 + p * 4 + 2] = 0;
            dst[l1 + p * 4 + 3] = 255;
        }
    }
    const Vtx = extern struct { x: f32, y: f32, u: f32, v: f32 };
    const attrs = [_]hal.VertexAttribute{
        .{ .location = 0, .format = .r32g32_float, .offset = 0 },
        .{ .location = 1, .format = .r32g32_float, .offset = 8 },
    };
    const vbuf = try dev.createResource(.{ .buffer = .{ .size = 6 * @sizeOf(Vtx), .usage = .{ .vertex = true } } });
    defer dev.destroyResource(vbuf);
    const pipe = try dev.createPipeline(.{
        .vertex = vs,
        .fragment = fs,
        .vertex_layout = .{ .stride = @sizeOf(Vtx), .attributes = &attrs },
        .color_format = .rgba8_unorm,
    });
    defer dev.destroyPipeline(pipe);
    const ctx = try dev.createContext();
    defer ctx.deinit();
    // UV 0..1 (magnified, implicit LOD ~0): level 0 red normally. A big +LOD bias -> the green mip.
    const q = [6]Vtx{
        .{ .x = -1, .y = -1, .u = 0, .v = 0 }, .{ .x = 1, .y = -1, .u = 1, .v = 0 }, .{ .x = 1, .y = 1, .u = 1, .v = 1 },
        .{ .x = -1, .y = -1, .u = 0, .v = 0 }, .{ .x = 1, .y = 1, .u = 1, .v = 1 },  .{ .x = -1, .y = 1, .u = 0, .v = 1 },
    };
    @memcpy(std.mem.bytesAsSlice(Vtx, try dev.mapResource(vbuf))[0..6], &q);
    const cb = try ctx.beginCommands();
    defer cb.deinit();
    try cb.setRenderTarget(rt);
    try cb.clear(.{ .r = 0, .g = 0, .b = 0, .a = 1 });
    try cb.bindPipeline(pipe);
    // mip_filter must be a real mip mode so the LOD bias engages. +4 LOD forces the (single extra) L1.
    try cb.bindTexture(.{ .binding = 2, .image = tex, .filter = .nearest, .address_u = .repeat, .address_v = .repeat, .mip_filter = .nearest, .lod_bias = 4.0 });
    try cb.bindVertexBuffer(vbuf);
    try cb.draw(6, 0);
    ctx.submit(cb) catch return error.SkipZigTest;
    const vv = std.mem.bytesAsSlice(u32, try dev.mapResource(rt))[32 * W + 32];
    // The +4 bias pushed the magnified sample from red (L0) to green (L1).
    try std.testing.expect(@as(u8, @truncate(vv >> 8)) > 180 and @as(u8, @truncate(vv)) < 70);
}

test "ORACLE-ANISO: anisotropic filtering keeps a fine-axis mip on a grazing (v-compressed) surface on the NVIDIA GPU (skips without a GPU)" {
    const glsl = @import("../../glsl.zig");
    const gpa = std.testing.allocator;
    const dev = NvDevice.create(gpa) catch return error.SkipZigTest;
    defer dev.deinit();
    const W: u32 = 64;

    const vs_src =
        \\attribute vec2 position;
        \\attribute vec2 aUV;
        \\varying vec2 vUV;
        \\void main() { gl_Position = vec4(position, 0.0, 1.0); vUV = aUV; }
    ;
    const fs_src =
        \\precision mediump float;
        \\uniform sampler2D uTex;
        \\varying vec2 vUV;
        \\void main() { gl_FragColor = texture2D(uTex, vUV); }
    ;
    const vs_bytes = try glsl.compileForStage(gpa, vs_src, .vertex);
    defer gpa.free(vs_bytes);
    const fs_bytes = try glsl.compileForStage(gpa, fs_src, .fragment);
    defer gpa.free(fs_bytes);
    const rt = try dev.createResource(.{ .image = .{ .width = W, .height = W, .format = .rgba8_unorm, .usage = .{ .render_target = true } } });
    defer dev.destroyResource(rt);
    const vs = try dev.createShaderModule(.{ .stage = .vertex, .code = vs_bytes });
    defer dev.destroyShaderModule(vs);
    const fs = try dev.createShaderModule(.{ .stage = .fragment, .code = fs_bytes });
    defer dev.destroyShaderModule(fs);

    // An 8x8 texture, 2 mips: level 0 = white, level 1 (4x4) = red. The quad is v-compressed (v spans
    // 0..16 over 64px -> a ~16:1 anisotropic minification: fine in u (LOD 0), coarse in v (LOD 1)).
    // Isotropic filtering picks LOD 1 -> red. Anisotropic filtering uses the fine u-axis LOD 0 (+ taps
    // along v) -> white. The same scene reads red with aniso off and white with aniso on.
    const TW: u32 = 8;
    const tex = try dev.createResource(.{ .image = .{ .width = TW, .height = TW, .format = .rgba8_unorm, .mip_levels = 2, .usage = .{ .sampled = true } } });
    defer dev.destroyResource(tex);
    {
        const dst = try dev.mapResource(tex);
        for (0..TW * TW) |p| {
            dst[p * 4 + 0] = 255;
            dst[p * 4 + 1] = 255;
            dst[p * 4 + 2] = 255;
            dst[p * 4 + 3] = 255;
        }
        const l1 = TW * TW * 4;
        for (0..(TW / 2) * (TW / 2)) |p| {
            dst[l1 + p * 4 + 0] = 255;
            dst[l1 + p * 4 + 1] = 0;
            dst[l1 + p * 4 + 2] = 0;
            dst[l1 + p * 4 + 3] = 255;
        }
    }

    const Vtx = extern struct { x: f32, y: f32, u: f32, v: f32 };
    const attrs = [_]hal.VertexAttribute{
        .{ .location = 0, .format = .r32g32_float, .offset = 0 },
        .{ .location = 1, .format = .r32g32_float, .offset = 8 },
    };
    const vbuf = try dev.createResource(.{ .buffer = .{ .size = 6 * @sizeOf(Vtx), .usage = .{ .vertex = true } } });
    defer dev.destroyResource(vbuf);
    const pipe = try dev.createPipeline(.{ .vertex = vs, .fragment = fs, .vertex_layout = .{ .stride = @sizeOf(Vtx), .attributes = &attrs }, .color_format = .rgba8_unorm });
    defer dev.destroyPipeline(pipe);
    const q = [6]Vtx{
        .{ .x = -1, .y = -1, .u = 0, .v = 0 }, .{ .x = 1, .y = -1, .u = 1, .v = 0 }, .{ .x = 1, .y = 1, .u = 1, .v = 16 },
        .{ .x = -1, .y = -1, .u = 0, .v = 0 }, .{ .x = 1, .y = 1, .u = 1, .v = 16 }, .{ .x = -1, .y = 1, .u = 0, .v = 16 },
    };
    @memcpy(std.mem.bytesAsSlice(Vtx, try dev.mapResource(vbuf))[0..6], &q);

    const anisos = [2]f32{ 1.0, 16.0 };
    var got: [2][3]u8 = undefined;
    for (anisos, 0..) |aniso, i| {
        const ctx = try dev.createContext();
        defer ctx.deinit();
        const cb = try ctx.beginCommands();
        defer cb.deinit();
        try cb.setRenderTarget(rt);
        try cb.clear(.{ .r = 0, .g = 0, .b = 0, .a = 1 });
        try cb.bindPipeline(pipe);
        try cb.bindTexture(.{ .binding = 2, .image = tex, .filter = .linear, .address_u = .repeat, .address_v = .repeat, .mip_filter = .nearest, .max_anisotropy = aniso });
        try cb.bindVertexBuffer(vbuf);
        try cb.draw(6, 0);
        try ctx.submit(cb);
        const fb = std.mem.bytesAsSlice(u32, try dev.mapResource(rt));
        const vv = fb[32 * W + 32];
        got[i] = .{ @truncate(vv), @truncate(vv >> 8), @truncate(vv >> 16) };
    }
    // aniso 1x -> isotropic LOD 1 -> red (R high, G/B low). aniso 16x -> LOD 0 -> white (all high).
    try std.testing.expect(got[0][0] > 200 and got[0][1] < 80 and got[0][2] < 80);
    try std.testing.expect(got[1][0] > 200 and got[1][1] > 200 and got[1][2] > 200);
}

test "ORACLE-CUBE-MIP: textureCubeLod selects the requested cube mip level on the NVIDIA GPU (explicit LOD / TEX.LL over the 6-face atlas mip chain) (skips without a GPU)" {
    const glsl = @import("../../glsl.zig");
    const gpa = std.testing.allocator;
    const dev = NvDevice.create(gpa) catch return error.SkipZigTest;
    defer dev.deinit();
    const W: u32 = 64;

    const vs_src =
        \\attribute vec2 position;
        \\attribute vec3 aDir;
        \\attribute float aLod;
        \\varying vec3 vDir;
        \\varying float vLod;
        \\void main() { gl_Position = vec4(position, 0.0, 1.0); vDir = aDir; vLod = aLod; }
    ;
    const fs_src =
        \\precision mediump float;
        \\uniform samplerCube uSky;
        \\varying vec3 vDir;
        \\varying float vLod;
        \\void main() { gl_FragColor = textureCubeLod(uSky, vDir, vLod); }
    ;
    const vs_bytes = try glsl.compileForStage(gpa, vs_src, .vertex);
    defer gpa.free(vs_bytes);
    const fs_bytes = try glsl.compileForStage(gpa, fs_src, .fragment);
    defer gpa.free(fs_bytes);

    const rt = try dev.createResource(.{ .image = .{ .width = W, .height = W, .format = .rgba8_unorm, .usage = .{ .render_target = true } } });
    defer dev.destroyResource(rt);
    const vs = try dev.createShaderModule(.{ .stage = .vertex, .code = vs_bytes });
    defer dev.destroyShaderModule(vs);
    const fs = try dev.createShaderModule(.{ .stage = .fragment, .code = fs_bytes });
    defer dev.destroyShaderModule(fs);

    // A 2-level 16x16 cubemap: level 0 = red on every face, level 1 = green. Explicit LOD picks the
    // level, so lod 0 -> red, lod 1 -> green (independent of the direction/face).
    const FW: u32 = 16;
    const cube = try dev.createResource(.{ .image = .{ .width = FW, .height = FW, .format = .rgba8_unorm, .cube = true, .mip_levels = 2, .usage = .{ .sampled = true } } });
    defer dev.destroyResource(cube);
    {
        // Staging is face-major, level-major within a face: 16x16x2 -> face_chain = (256+64)*4 = 1280,
        // level 1 at +1024.
        const face_chain: usize = (FW * FW + (FW / 2) * (FW / 2)) * 4;
        const dst = try dev.mapResource(cube);
        for (0..6) |f| {
            const base = f * face_chain;
            for (0..FW * FW) |p| {
                dst[base + p * 4 + 0] = 255;
                dst[base + p * 4 + 1] = 0;
                dst[base + p * 4 + 2] = 0;
                dst[base + p * 4 + 3] = 255;
            }
            const l1 = base + FW * FW * 4;
            for (0..(FW / 2) * (FW / 2)) |p| {
                dst[l1 + p * 4 + 0] = 0;
                dst[l1 + p * 4 + 1] = 255;
                dst[l1 + p * 4 + 2] = 0;
                dst[l1 + p * 4 + 3] = 255;
            }
        }
    }

    const Vtx = extern struct { x: f32, y: f32, dx: f32, dy: f32, dz: f32, lod: f32 };
    const attrs = [_]hal.VertexAttribute{
        .{ .location = 0, .format = .r32g32_float, .offset = 0 },
        .{ .location = 1, .format = .r32g32b32_float, .offset = 8 },
        .{ .location = 2, .format = .r32_float, .offset = 20 },
    };
    const vbuf = try dev.createResource(.{ .buffer = .{ .size = 6 * @sizeOf(Vtx), .usage = .{ .vertex = true } } });
    defer dev.destroyResource(vbuf);
    const pipe = try dev.createPipeline(.{
        .vertex = vs,
        .fragment = fs,
        .vertex_layout = .{ .stride = @sizeOf(Vtx), .attributes = &attrs },
        .color_format = .rgba8_unorm,
    });
    defer dev.destroyPipeline(pipe);

    const dir = [3]f32{ 0.2, 0.1, 1.0 }; // +Z face (any face works; both levels are face-uniform)
    const lods = [2]f32{ 0.0, 1.0 };
    const want = [2][3]u8{ .{ 255, 0, 0 }, .{ 0, 255, 0 } };
    for (lods, want) |lod, exp| {
        var q: [6]Vtx = undefined;
        const corners = [6][2]f32{ .{ -1, -1 }, .{ 1, -1 }, .{ 1, 1 }, .{ -1, -1 }, .{ 1, 1 }, .{ -1, 1 } };
        for (0..6) |k| q[k] = .{ .x = corners[k][0], .y = corners[k][1], .dx = dir[0], .dy = dir[1], .dz = dir[2], .lod = lod };
        @memcpy(std.mem.bytesAsSlice(Vtx, try dev.mapResource(vbuf))[0..6], &q);
        const ctx = try dev.createContext();
        defer ctx.deinit();
        const cb = try ctx.beginCommands();
        defer cb.deinit();
        try cb.setRenderTarget(rt);
        try cb.clear(.{ .r = 0, .g = 0, .b = 0, .a = 1 });
        try cb.bindPipeline(pipe);
        try cb.bindTexture(.{ .binding = 2, .image = cube, .filter = .nearest, .address_u = .clamp_to_edge, .address_v = .clamp_to_edge, .mip_filter = .nearest });
        try cb.bindVertexBuffer(vbuf);
        try cb.draw(6, 0);
        try ctx.submit(cb);
        const fb = std.mem.bytesAsSlice(u32, try dev.mapResource(rt));
        const vv = fb[32 * W + 32];
        const c = [3]u8{ @truncate(vv), @truncate(vv >> 8), @truncate(vv >> 16) };
        try std.testing.expect(@abs(@as(i32, c[0]) - exp[0]) < 40 and @abs(@as(i32, c[1]) - exp[1]) < 40 and @abs(@as(i32, c[2]) - exp[2]) < 40);
    }
}

test "ORACLE-FLOAT-RT-SAMPLE: a rendered rgba16f RT samples back its HDR value UNCLAMPED on the NVIDIA GPU (skips without a GPU)" {
    const glsl = @import("../../glsl.zig");
    const gpa = std.testing.allocator;
    const dev = NvDevice.create(gpa) catch return error.SkipZigTest;
    defer dev.deinit();
    const W: u32 = 64;
    const H: u32 = 64;

    const vs_src =
        \\attribute vec2 position;
        \\attribute vec2 aUV;
        \\varying vec2 vUV;
        \\void main() { gl_Position = vec4(position, 0.0, 1.0); vUV = aUV; }
    ;
    const fill_src =
        \\precision highp float;
        \\void main() { gl_FragColor = vec4(2.0, 0.5, 0.25, 1.0); }
    ;
    const sample_src =
        \\precision highp float;
        \\uniform sampler2D uTex;
        \\varying vec2 vUV;
        \\void main() { gl_FragColor = texture2D(uTex, vUV) * 0.5; }
    ;
    const vs_bytes = try glsl.compileForStage(gpa, vs_src, .vertex);
    defer gpa.free(vs_bytes);
    const fill_bytes = try glsl.compileForStage(gpa, fill_src, .fragment);
    defer gpa.free(fill_bytes);
    const sample_bytes = try glsl.compileForStage(gpa, sample_src, .fragment);
    defer gpa.free(sample_bytes);

    const vs = try dev.createShaderModule(.{ .stage = .vertex, .code = vs_bytes });
    defer dev.destroyShaderModule(vs);
    const fill_fs = try dev.createShaderModule(.{ .stage = .fragment, .code = fill_bytes });
    defer dev.destroyShaderModule(fill_fs);
    const sample_fs = try dev.createShaderModule(.{ .stage = .fragment, .code = sample_bytes });
    defer dev.destroyShaderModule(sample_fs);

    // The float RT is both rendered into (pass 1) and sampled (pass 2).
    const float_rt = try dev.createResource(.{ .image = .{ .width = W, .height = H, .format = .rgba16_float, .usage = .{ .render_target = true, .sampled = true } } });
    defer dev.destroyResource(float_rt);
    const out_rt = try dev.createResource(.{ .image = .{ .width = W, .height = H, .format = .rgba8_unorm, .usage = .{ .render_target = true } } });
    defer dev.destroyResource(out_rt);

    const Vtx = extern struct { x: f32, y: f32, u: f32, v: f32 };
    const quad = [6]Vtx{
        .{ .x = -1, .y = -1, .u = 0, .v = 0 },
        .{ .x = 1, .y = -1, .u = 1, .v = 0 },
        .{ .x = 1, .y = 1, .u = 1, .v = 1 },
        .{ .x = -1, .y = -1, .u = 0, .v = 0 },
        .{ .x = 1, .y = 1, .u = 1, .v = 1 },
        .{ .x = -1, .y = 1, .u = 0, .v = 1 },
    };
    const vbuf = try dev.createResource(.{ .buffer = .{ .size = @sizeOf(@TypeOf(quad)), .usage = .{ .vertex = true } } });
    defer dev.destroyResource(vbuf);
    @memcpy(std.mem.bytesAsSlice(Vtx, try dev.mapResource(vbuf))[0..quad.len], &quad);

    const attrs = [_]hal.VertexAttribute{
        .{ .location = 0, .format = .r32g32_float, .offset = 0 },
        .{ .location = 1, .format = .r32g32_float, .offset = 8 },
    };
    const fill_pipe = try dev.createPipeline(.{
        .vertex = vs,
        .fragment = fill_fs,
        .vertex_layout = .{ .stride = @sizeOf(Vtx), .attributes = &attrs },
        .color_format = .rgba16_float,
    });
    defer dev.destroyPipeline(fill_pipe);
    const sample_pipe = try dev.createPipeline(.{
        .vertex = vs,
        .fragment = sample_fs,
        .vertex_layout = .{ .stride = @sizeOf(Vtx), .attributes = &attrs },
        .color_format = .rgba8_unorm,
    });
    defer dev.destroyPipeline(sample_pipe);

    const ctx = try dev.createContext();
    defer ctx.deinit();
    // Pass 1: fill the float RT with the HDR constant.
    {
        const cb = try ctx.beginCommands();
        defer cb.deinit();
        try cb.setRenderTarget(float_rt);
        try cb.clear(.{ .r = 0, .g = 0, .b = 0, .a = 1 });
        try cb.bindPipeline(fill_pipe);
        try cb.bindVertexBuffer(vbuf);
        try cb.draw(6, 0);
        try ctx.submit(cb);
    }
    // Pass 2: sample the float RT * 0.5 into the 8-bit output.
    {
        const cb = try ctx.beginCommands();
        defer cb.deinit();
        try cb.setRenderTarget(out_rt);
        try cb.clear(.{ .r = 0, .g = 0, .b = 0, .a = 1 });
        try cb.bindPipeline(sample_pipe);
        try cb.bindTexture(.{ .binding = 2, .image = float_rt, .filter = .nearest, .address_u = .clamp_to_edge, .address_v = .clamp_to_edge });
        try cb.bindVertexBuffer(vbuf);
        try cb.draw(6, 0);
        try ctx.submit(cb);
    }

    const fb = std.mem.bytesAsSlice(u32, try dev.mapResource(out_rt));
    const center = fb[@as(usize, 32) * W + 32];
    const r: u8 = @truncate(center); // rgba8 de-swizzle puts R at byte 0
    // Sampled HDR 2.0 * 0.5 = 1.0 -> 255. A clamped (1.0) float RT would give 0.5 -> ~128.
    try std.testing.expect(r > 240);
}

test "ORACLE: a mipmapped texture minifies to a lower mip level on the NVIDIA GPU via HW implicit LOD (skips without a GPU)" {
    const glsl = @import("../../glsl.zig");
    const gpa = std.testing.allocator;
    const dev = NvDevice.create(gpa) catch return error.SkipZigTest;
    defer dev.deinit();
    const W: u32 = 64;
    const H: u32 = 64;

    const vs_src =
        \\attribute vec2 position;
        \\attribute vec2 aUV;
        \\varying vec2 vUV;
        \\void main() { gl_Position = vec4(position, 0.0, 1.0); vUV = aUV; }
    ;
    const fs_src =
        \\precision mediump float;
        \\uniform sampler2D uTex;
        \\varying vec2 vUV;
        \\void main() { gl_FragColor = texture2D(uTex, vUV); }
    ;
    const vs_bytes = try glsl.compileForStage(gpa, vs_src, .vertex);
    defer gpa.free(vs_bytes);
    const fs_bytes = try glsl.compileForStage(gpa, fs_src, .fragment);
    defer gpa.free(fs_bytes);

    const rt = try dev.createResource(.{ .image = .{ .width = W, .height = H, .format = .rgba8_unorm, .usage = .{ .render_target = true } } });
    defer dev.destroyResource(rt);
    const vs = try dev.createShaderModule(.{ .stage = .vertex, .code = vs_bytes });
    defer dev.destroyShaderModule(vs);
    const fs = try dev.createShaderModule(.{ .stage = .fragment, .code = fs_bytes });
    defer dev.destroyShaderModule(fs);

    // An 8x8 texture with a 4-level chain (8,4,2,1). Level 0 is red. Every lower level is blue.
    // A minified draw makes the HW pick a lower level -> blue. Without mip sampling it reads the
    // red base. (Distinct per-level colors isolate the level selection from any averaging.)
    const levels: u8 = 4;
    const tex = try dev.createResource(.{ .image = .{ .width = 8, .height = 8, .format = .rgba8_unorm, .usage = .{ .sampled = true }, .mip_levels = levels } });
    defer dev.destroyResource(tex);
    {
        const dst = try dev.mapResource(tex);
        const dims = [4][2]u32{ .{ 8, 8 }, .{ 4, 4 }, .{ 2, 2 }, .{ 1, 1 } };
        const level_rgb = [4][3]u8{ .{ 255, 0, 0 }, .{ 0, 255, 0 }, .{ 0, 0, 255 }, .{ 255, 255, 255 } };
        var off: usize = 0;
        for (dims, 0..) |d, lvl| {
            const count = d[0] * d[1];
            var i: usize = 0;
            while (i < count) : (i += 1) {
                const p = off + i * 4;
                dst[p + 0] = level_rgb[lvl][0];
                dst[p + 1] = level_rgb[lvl][1];
                dst[p + 2] = level_rgb[lvl][2];
                dst[p + 3] = 255;
            }
            off += count * 4;
        }
    }

    // Fullscreen quad, uv tiled 16x: du/dpixel = 16/64 = 0.25 -> 2 texels/pixel -> LOD ~1.
    const Vtx = extern struct { x: f32, y: f32, u: f32, v: f32 };
    const quad = [6]Vtx{
        .{ .x = -1, .y = -1, .u = 0, .v = 0 }, .{ .x = 1, .y = -1, .u = 64, .v = 0 },
        .{ .x = 1, .y = 1, .u = 64, .v = 64 }, .{ .x = -1, .y = -1, .u = 0, .v = 0 },
        .{ .x = 1, .y = 1, .u = 64, .v = 64 }, .{ .x = -1, .y = 1, .u = 0, .v = 64 },
    };
    const vbuf = try dev.createResource(.{ .buffer = .{ .size = @sizeOf(@TypeOf(quad)), .usage = .{ .vertex = true } } });
    defer dev.destroyResource(vbuf);
    @memcpy(std.mem.bytesAsSlice(Vtx, try dev.mapResource(vbuf))[0..quad.len], &quad);

    const attrs = [_]hal.VertexAttribute{
        .{ .location = 0, .format = .r32g32_float, .offset = 0 },
        .{ .location = 1, .format = .r32g32_float, .offset = 8 },
    };
    const pipe = try dev.createPipeline(.{
        .vertex = vs,
        .fragment = fs,
        .vertex_layout = .{ .stride = @sizeOf(Vtx), .attributes = &attrs },
        .color_format = .rgba8_unorm,
    });
    defer dev.destroyPipeline(pipe);

    const ctx = try dev.createContext();
    defer ctx.deinit();
    {
        const cb = try ctx.beginCommands();
        defer cb.deinit();
        try cb.setRenderTarget(rt);
        try cb.clear(.{ .r = 0, .g = 0, .b = 0, .a = 1 });
        try cb.bindPipeline(pipe);
        try cb.bindTexture(.{ .binding = 2, .image = tex, .filter = .nearest, .min_filter = .nearest, .mip_filter = .nearest, .address_u = .repeat, .address_v = .repeat });
        try cb.bindVertexBuffer(vbuf);
        try cb.draw(6, 0);
        ctx.submit(cb) catch return error.SkipZigTest;
    }

    const fb = std.mem.bytesAsSlice(u32, try dev.mapResource(rt));
    const c = fb[32 * W + 32];
    const r: u8 = @truncate(c);
    const g: u8 = @truncate(c >> 8);
    const b: u8 = @truncate(c >> 16);
    // The heavily-minified draw (uv tiled 64x -> ~8 texels/pixel -> LOD ~3) makes the HW pick a
    // lower mip level via implicit LOD. Levels are distinct colors (L0 red, L1 green, L2 blue, L3
    // white), so a lower level is anything with green or blue high. The red base (L0) has both low.
    // Green/blue high therefore proves the HW minified through the block-linear chain (MAX_MIP_LEVEL
    // + RES_VIEW_MAX_MIP_LEVEL + MIP_POINT + Auto-LOD TEX + per-level upload all correct).
    try std.testing.expect((g > 200 or b > 200) and !(r > 200 and g < 60 and b < 60));

    // TRILINEAR (TSC MIP_LINEAR): re-draw with the mip-blend filter. At LOD ~3 the HW blends the two
    // bracketing lower levels (blue/white here), still a non-base color (b high). Proves the GPU
    // MIP_LINEAR path samples the chain (not the red base).
    {
        const cb2 = try ctx.beginCommands();
        defer cb2.deinit();
        try cb2.setRenderTarget(rt);
        try cb2.clear(.{ .r = 0, .g = 0, .b = 0, .a = 1 });
        try cb2.bindPipeline(pipe);
        try cb2.bindTexture(.{ .binding = 2, .image = tex, .filter = .linear, .min_filter = .linear, .mip_filter = .linear, .address_u = .repeat, .address_v = .repeat });
        try cb2.bindVertexBuffer(vbuf);
        try cb2.draw(6, 0);
        ctx.submit(cb2) catch return error.SkipZigTest;
    }
    const fb2 = std.mem.bytesAsSlice(u32, try dev.mapResource(rt));
    const c2 = fb2[32 * W + 32];
    const r2: u8 = @truncate(c2);
    const g2: u8 = @truncate(c2 >> 8);
    const b2: u8 = @truncate(c2 >> 16);
    try std.testing.expect((g2 > 150 or b2 > 150) and !(r2 > 200 and g2 < 60 and b2 < 60));
}

test "ORACLE: anisotropic filtering samples a sharper (lower) mip level than isotropic on the NVIDIA GPU (skips without a GPU)" {
    const glsl = @import("../../glsl.zig");
    const gpa = std.testing.allocator;
    const dev = NvDevice.create(gpa) catch return error.SkipZigTest;
    defer dev.deinit();
    const W: u32 = 64;
    const H: u32 = 64;

    const vs_src =
        \\attribute vec2 position;
        \\attribute vec2 aUV;
        \\varying vec2 vUV;
        \\void main() { gl_Position = vec4(position, 0.0, 1.0); vUV = aUV; }
    ;
    const fs_src =
        \\precision mediump float;
        \\uniform sampler2D uTex;
        \\varying vec2 vUV;
        \\void main() { gl_FragColor = texture2D(uTex, vUV); }
    ;
    const vs_bytes = try glsl.compileForStage(gpa, vs_src, .vertex);
    defer gpa.free(vs_bytes);
    const fs_bytes = try glsl.compileForStage(gpa, fs_src, .fragment);
    defer gpa.free(fs_bytes);
    const vs = try dev.createShaderModule(.{ .stage = .vertex, .code = vs_bytes });
    defer dev.destroyShaderModule(vs);
    const fs = try dev.createShaderModule(.{ .stage = .fragment, .code = fs_bytes });
    defer dev.destroyShaderModule(fs);

    // 8x8, 4-level chain, distinct per-level colors (L0 red, L1 green, L2 blue, L3 white).
    const tex = try dev.createResource(.{ .image = .{ .width = 8, .height = 8, .format = .rgba8_unorm, .usage = .{ .sampled = true }, .mip_levels = 4 } });
    defer dev.destroyResource(tex);
    {
        const dst = try dev.mapResource(tex);
        const dims = [4][2]u32{ .{ 8, 8 }, .{ 4, 4 }, .{ 2, 2 }, .{ 1, 1 } };
        const rgb = [4][3]u8{ .{ 255, 0, 0 }, .{ 0, 255, 0 }, .{ 0, 0, 255 }, .{ 255, 255, 255 } };
        var off: usize = 0;
        for (dims, 0..) |d, lvl| {
            var i: usize = 0;
            while (i < d[0] * d[1]) : (i += 1) {
                dst[off + i * 4 + 0] = rgb[lvl][0];
                dst[off + i * 4 + 1] = rgb[lvl][1];
                dst[off + i * 4 + 2] = rgb[lvl][2];
                dst[off + i * 4 + 3] = 255;
            }
            off += d[0] * d[1] * 4;
        }
    }

    // Anisotropic footprint: uv tiles 16x in X but stays [0,1] in Y. So d(uv)/dx is large (~2
    // texels/pixel) but d(uv)/dy is tiny, a 16:1 anisotropic footprint. Isotropic LOD = the MAX
    // axis -> a lower mip level (green-ish); anisotropy samples along X at the minor-axis LOD ->
    // a sharper, lower-index level (redder, toward the base). So aniso should shift the color
    // toward red vs isotropic.
    const Vtx = extern struct { x: f32, y: f32, u: f32, v: f32 };
    const quad = [6]Vtx{
        .{ .x = -1, .y = -1, .u = 0, .v = 0 }, .{ .x = 1, .y = -1, .u = 16, .v = 0 },
        .{ .x = 1, .y = 1, .u = 16, .v = 1 },  .{ .x = -1, .y = -1, .u = 0, .v = 0 },
        .{ .x = 1, .y = 1, .u = 16, .v = 1 },  .{ .x = -1, .y = 1, .u = 0, .v = 1 },
    };
    const vbuf = try dev.createResource(.{ .buffer = .{ .size = @sizeOf(@TypeOf(quad)), .usage = .{ .vertex = true } } });
    defer dev.destroyResource(vbuf);
    @memcpy(std.mem.bytesAsSlice(Vtx, try dev.mapResource(vbuf))[0..quad.len], &quad);
    const attrs = [_]hal.VertexAttribute{
        .{ .location = 0, .format = .r32g32_float, .offset = 0 },
        .{ .location = 1, .format = .r32g32_float, .offset = 8 },
    };
    const pipe = try dev.createPipeline(.{ .vertex = vs, .fragment = fs, .vertex_layout = .{ .stride = @sizeOf(Vtx), .attributes = &attrs }, .color_format = .rgba8_unorm });
    defer dev.destroyPipeline(pipe);

    const render = struct {
        // A FRESH context + render target per call: isolates the two draws so a stale sampler/TSC
        // descriptor from the prior submit cannot leak into the next (that made aniso==iso flaky).
        fn f(d: hal.Device, p: *hal.Pipeline, t: *hal.Resource, vb: *hal.Resource, aniso: f32) !u32 {
            const c = try d.createContext();
            defer c.deinit();
            const target = try d.createResource(.{ .image = .{ .width = W, .height = H, .format = .rgba8_unorm, .usage = .{ .render_target = true } } });
            defer d.destroyResource(target);
            const cb = try c.beginCommands();
            defer cb.deinit();
            try cb.setRenderTarget(target);
            try cb.clear(.{ .r = 0, .g = 0, .b = 0, .a = 1 });
            try cb.bindPipeline(p);
            try cb.bindTexture(.{ .binding = 2, .image = t, .filter = .nearest, .min_filter = .nearest, .mip_filter = .nearest, .address_u = .repeat, .address_v = .repeat, .max_anisotropy = aniso });
            try cb.bindVertexBuffer(vb);
            try cb.draw(6, 0);
            try c.submit(cb);
            const fb = std.mem.bytesAsSlice(u32, try d.mapResource(target));
            return fb[32 * W + 32];
        }
    }.f;
    const iso = render(dev, pipe, tex, vbuf, 1) catch return error.SkipZigTest;
    const aniso = render(dev, pipe, tex, vbuf, 16) catch return error.SkipZigTest;
    const iso_r: u8 = @truncate(iso);
    const ani_r: u8 = @truncate(aniso);
    // Anisotropy retains detail along the minor axis -> a lower/sharper mip -> more red (toward the
    // base) than the isotropic sample. Proves the TSC MAX_ANISOTROPY is applied by the HW.
    try std.testing.expect(ani_r > iso_r + 30);
}

test "ORACLE: sRGB + fp16 textures sample correctly on the NVIDIA GPU (TIC sRGB decode + FLOAT format) (skips without a GPU)" {
    const glsl = @import("../../glsl.zig");
    const gpa = std.testing.allocator;
    const dev = NvDevice.create(gpa) catch return error.SkipZigTest;
    defer dev.deinit();
    const W: u32 = 64;
    const H: u32 = 64;

    const vs_src =
        \\attribute vec2 position;
        \\attribute vec2 aUV;
        \\varying vec2 vUV;
        \\void main() { gl_Position = vec4(position, 0.0, 1.0); vUV = aUV; }
    ;
    const fs_src =
        \\precision mediump float;
        \\uniform sampler2D uTex;
        \\varying vec2 vUV;
        \\void main() { gl_FragColor = texture2D(uTex, vUV); }
    ;
    const vs_bytes = try glsl.compileForStage(gpa, vs_src, .vertex);
    defer gpa.free(vs_bytes);
    const fs_bytes = try glsl.compileForStage(gpa, fs_src, .fragment);
    defer gpa.free(fs_bytes);
    const vs = try dev.createShaderModule(.{ .stage = .vertex, .code = vs_bytes });
    defer dev.destroyShaderModule(vs);
    const fs = try dev.createShaderModule(.{ .stage = .fragment, .code = fs_bytes });
    defer dev.destroyShaderModule(fs);

    const Vtx = extern struct { x: f32, y: f32, u: f32, v: f32 };
    const quad = [6]Vtx{
        .{ .x = -1, .y = -1, .u = 0, .v = 0 }, .{ .x = 1, .y = -1, .u = 1, .v = 0 }, .{ .x = 1, .y = 1, .u = 1, .v = 1 },
        .{ .x = -1, .y = -1, .u = 0, .v = 0 }, .{ .x = 1, .y = 1, .u = 1, .v = 1 },  .{ .x = -1, .y = 1, .u = 0, .v = 1 },
    };
    const vbuf = try dev.createResource(.{ .buffer = .{ .size = @sizeOf(@TypeOf(quad)), .usage = .{ .vertex = true } } });
    defer dev.destroyResource(vbuf);
    @memcpy(std.mem.bytesAsSlice(Vtx, try dev.mapResource(vbuf))[0..quad.len], &quad);
    const attrs = [_]hal.VertexAttribute{
        .{ .location = 0, .format = .r32g32_float, .offset = 0 },
        .{ .location = 1, .format = .r32g32_float, .offset = 8 },
    };
    const pipe = try dev.createPipeline(.{
        .vertex = vs,
        .fragment = fs,
        .vertex_layout = .{ .stride = @sizeOf(Vtx), .attributes = &attrs },
        .color_format = .rgba8_unorm,
    });
    defer dev.destroyPipeline(pipe);

    // Render `tex` full-screen and return the center pixel's R channel (0..255).
    const sampleR = struct {
        fn go(d: anytype, p: anytype, vbf: anytype, tex: anytype) !u8 {
            const rt = try d.createResource(.{ .image = .{ .width = W, .height = H, .format = .rgba8_unorm, .usage = .{ .render_target = true } } });
            defer d.destroyResource(rt);
            const ctx = try d.createContext();
            defer ctx.deinit();
            const cb = try ctx.beginCommands();
            defer cb.deinit();
            try cb.setRenderTarget(rt);
            try cb.clear(.{ .r = 0, .g = 0, .b = 0, .a = 1 });
            try cb.bindPipeline(p);
            try cb.bindTexture(.{ .binding = 2, .image = tex, .filter = .nearest, .address_u = .clamp_to_edge, .address_v = .clamp_to_edge });
            try cb.bindVertexBuffer(vbf);
            try cb.draw(6, 0);
            ctx.submit(cb) catch return error.SkipZigTest;
            const fb = std.mem.bytesAsSlice(u32, try d.mapResource(rt));
            return @truncate(fb[32 * W + 32]); // rgba8: R at byte 0
        }
    }.go;

    // sRGB texture: a mid-gray sRGB texel (188) must be decoded by the TIC's sRGB bit to
    // linear ~0.5 -> ~128 in the rgba8 output. A plain rgba8 read would give ~188.
    {
        const tex = try dev.createResource(.{ .image = .{ .width = 1, .height = 1, .format = .rgba8_srgb, .usage = .{ .sampled = true } } });
        defer dev.destroyResource(tex);
        @memcpy((try dev.mapResource(tex))[0..4], &[_]u8{ 188, 188, 188, 255 });
        const r = try sampleR(dev, pipe, vbuf, tex);
        try std.testing.expect(r > 110 and r < 145); // sRGB decoded to linear, not 188
    }
    // fp16 texture: R=0.5 stored as IEEE half -> sampled 0.5 -> ~128. Proves the FLOAT TIC
    // format + the 8-byte-per-texel block-linear swizzle read correctly on the GPU.
    {
        const tex = try dev.createResource(.{ .image = .{ .width = 1, .height = 1, .format = .rgba16_float, .usage = .{ .sampled = true } } });
        defer dev.destroyResource(tex);
        const h: u16 = @bitCast(@as(f16, 0.5));
        const bytes = [_]u8{ @truncate(h), @truncate(h >> 8) } ** 4;
        @memcpy((try dev.mapResource(tex))[0..8], &bytes);
        const r = try sampleR(dev, pipe, vbuf, tex);
        try std.testing.expect(r > 110 and r < 145); // fp16 0.5 sampled correctly
    }
}

// GLES larger-texture oracle (the glmark2 `texture` scene size): the 2x2 oracle above
// lives entirely in the first GOB sector, so it never exercises the real BLOCK-LINEAR
// texture tiling (texBlockHeightLog2 > 0, multiple GOBs wide and tall, the GOB sector
// swizzle). glmark2's crate texture is a 512x512 file image, far larger than one GOB.
// Here a 512x512 RGBA8 texture (the same size) with four distinct per-quadrant colors
// (each quadrant 256x256 px = many GOBs) is uploaded as a sampled texture and sampled
// through the same GLES texture2D FS. If the upload-tiling (blTexPixelOffset) disagrees
// with what the TEX unit reads back at this width (8 GOBs wide), the sampler returns
// zero/garbage -> the quadrants are wrong or black. This is the permanent frame-verified
// guard that the wide+tall multi-GOB block-linear texture path samples correctly on the GPU.
test "ORACLE: GLES textured quad samples a 512x512 multi-GOB texture correctly on the NVIDIA GPU (skips without a GPU)" {
    const glsl = @import("../../glsl.zig");
    const gpa = std.testing.allocator;
    const dev = NvDevice.create(gpa) catch return error.SkipZigTest;
    defer dev.deinit();
    const W: u32 = 512;
    const H: u32 = 512;
    const TW: u32 = 512;
    const TH: u32 = 512;

    const vs_src =
        \\attribute vec2 position;
        \\attribute vec2 aUV;
        \\varying vec2 vUV;
        \\void main() { gl_Position = vec4(position, 0.0, 1.0); vUV = aUV; }
    ;
    const fs_src =
        \\precision mediump float;
        \\uniform sampler2D uTex;
        \\varying vec2 vUV;
        \\void main() { gl_FragColor = texture2D(uTex, vUV); }
    ;
    const vs_bytes = try glsl.compileForStage(gpa, vs_src, .vertex);
    defer gpa.free(vs_bytes);
    const fs_bytes = try glsl.compileForStage(gpa, fs_src, .fragment);
    defer gpa.free(fs_bytes);

    const rt = try dev.createResource(.{ .image = .{ .width = W, .height = H, .format = .rgba8_unorm, .usage = .{ .render_target = true } } });
    defer dev.destroyResource(rt);
    const vs = try dev.createShaderModule(.{ .stage = .vertex, .code = vs_bytes });
    defer dev.destroyShaderModule(vs);
    const fs = try dev.createShaderModule(.{ .stage = .fragment, .code = fs_bytes });
    defer dev.destroyShaderModule(fs);

    // 128x128 RGBA8 texture, four 64x64 quadrants with distinct, non-grey colors so a
    // wrong sample is unambiguous: TL red, TR green, BL blue, BR yellow. The interiors
    // are flat per quadrant, so a sampled value mismatching the expected quadrant color
    // means the GOB tiling (upload vs TEX read) disagrees.
    const tex = try dev.createResource(.{ .image = .{ .width = TW, .height = TH, .format = .rgba8_unorm, .usage = .{ .sampled = true } } });
    defer dev.destroyResource(tex);
    {
        const dst = try dev.mapResource(tex);
        var ty: u32 = 0;
        while (ty < TH) : (ty += 1) {
            var tx: u32 = 0;
            while (tx < TW) : (tx += 1) {
                const o = (@as(usize, ty) * TW + tx) * 4;
                const left = tx < TW / 2;
                const top = ty < TH / 2;
                const c: [4]u8 = if (top and left)
                    .{ 255, 0, 0, 255 } // TL red
                else if (top and !left)
                    .{ 0, 255, 0, 255 } // TR green
                else if (!top and left)
                    .{ 0, 0, 255, 255 } // BL blue
                else
                    .{ 255, 255, 0, 255 }; // BR yellow
                @memcpy(dst[o .. o + 4], &c);
            }
        }
    }

    const Vtx = extern struct { x: f32, y: f32, u: f32, v: f32 };
    const quad = [6]Vtx{
        .{ .x = -1, .y = -1, .u = 0, .v = 0 },
        .{ .x = 1, .y = -1, .u = 1, .v = 0 },
        .{ .x = 1, .y = 1, .u = 1, .v = 1 },
        .{ .x = -1, .y = -1, .u = 0, .v = 0 },
        .{ .x = 1, .y = 1, .u = 1, .v = 1 },
        .{ .x = -1, .y = 1, .u = 0, .v = 1 },
    };
    const vbuf = try dev.createResource(.{ .buffer = .{ .size = @sizeOf(@TypeOf(quad)), .usage = .{ .vertex = true } } });
    defer dev.destroyResource(vbuf);
    @memcpy(std.mem.bytesAsSlice(Vtx, try dev.mapResource(vbuf))[0..quad.len], &quad);

    const attrs = [_]hal.VertexAttribute{
        .{ .location = 0, .format = .r32g32_float, .offset = 0 },
        .{ .location = 1, .format = .r32g32_float, .offset = 8 },
    };
    const pipe = try dev.createPipeline(.{
        .vertex = vs,
        .fragment = fs,
        .vertex_layout = .{ .stride = @sizeOf(Vtx), .attributes = &attrs },
        .color_format = .rgba8_unorm,
    });
    defer dev.destroyPipeline(pipe);

    const ctx = try dev.createContext();
    defer ctx.deinit();
    {
        const cb = try ctx.beginCommands();
        defer cb.deinit();
        try cb.setRenderTarget(rt);
        try cb.clear(.{ .r = 0, .g = 0, .b = 0, .a = 1 });
        try cb.bindPipeline(pipe);
        try cb.bindTexture(.{ .binding = 2, .image = tex, .filter = .nearest, .address_u = .clamp_to_edge, .address_v = .clamp_to_edge });
        try cb.bindVertexBuffer(vbuf);
        try cb.draw(6, 0);
        try ctx.submit(cb);
    }

    const fb = std.mem.bytesAsSlice(u32, try dev.mapResource(rt));
    const at = struct {
        fn p(buf: []align(1) const u32, x: usize, y: usize) [3]u8 {
            const v = buf[y * W + x];
            return .{ @truncate(v), @truncate(v >> 8), @truncate(v >> 16) };
        }
    }.p;
    // Sample the center of each on-screen quadrant. v=0 -> row 0 (top), matching 2x2.
    const tl = at(fb, W / 4, H / 4);
    const tr = at(fb, 3 * W / 4, H / 4);
    const bl = at(fb, W / 4, 3 * H / 4);
    const br = at(fb, 3 * W / 4, 3 * H / 4);
    try std.testing.expect(tl[0] > 200 and tl[1] < 60 and tl[2] < 60); // red
    try std.testing.expect(tr[1] > 200 and tr[0] < 60 and tr[2] < 60); // green
    try std.testing.expect(bl[2] > 200 and bl[0] < 60 and bl[1] < 60); // blue
    try std.testing.expect(br[0] > 200 and br[1] > 200 and br[2] < 60); // yellow
}

// GLMARK2 `texture`-SCENE-SHAPE ORACLE (the actual blank-on-nvidia repro): glmark2's
// default texture scene draws a model with three attributes (position(vec3, loc 0),
// normal(vec3, loc 1), texcoord(vec2, loc 2)) and FS `gl_FragColor = texture2D(tex,
// TextureCoord) * Color`, where TextureCoord is the vec2 attribute passed through. The
// 2x2 + 128x128 oracles above use only two attributes (position+uv) with the uv at
// location 1; they never exercised a vec2 texcoord at location 2 behind a vec3 normal.
// If the Data Assembler fetches the 3rd (vec2) attribute from the wrong stream offset,
// TextureCoord is garbage -> the sampler reads the clamped (0,0) border = black, even
// though the lit Color is non-zero -> the whole textured model goes black (the exact
// glmark2 symptom). This permanent guard reproduces glmark2's attribute shape with a
// flat quad so the expected sampled color is deterministic.
test "ORACLE: GLES textured quad with glmark2's 3-attribute (pos/normal/texcoord) layout samples correctly on the NVIDIA GPU (skips without a GPU)" {
    const glsl = @import("../../glsl.zig");
    const gpa = std.testing.allocator;
    const dev = NvDevice.create(gpa) catch return error.SkipZigTest;
    defer dev.deinit();
    const W: u32 = 64;
    const H: u32 = 64;

    // glmark2's light-basic.vert shape: 3 attributes, lit Color, texcoord passthrough.
    const vs_src =
        \\attribute vec3 position;
        \\attribute vec3 normal;
        \\attribute vec2 texcoord;
        \\varying vec4 Color;
        \\varying vec2 TextureCoord;
        \\void main(void) {
        \\  Color = vec4(abs(normal), 1.0);
        \\  TextureCoord = texcoord;
        \\  gl_Position = vec4(position, 1.0);
        \\}
    ;
    // glmark2's light-basic-tex.frag: texel * Color.
    const fs_src =
        \\precision mediump float;
        \\uniform sampler2D MaterialTexture0;
        \\varying vec4 Color;
        \\varying vec2 TextureCoord;
        \\void main(void) {
        \\  vec4 texel = texture2D(MaterialTexture0, TextureCoord);
        \\  gl_FragColor = texel * Color;
        \\}
    ;
    const vs_bytes = try glsl.compileForStage(gpa, vs_src, .vertex);
    defer gpa.free(vs_bytes);
    const fs_bytes = try glsl.compileForStage(gpa, fs_src, .fragment);
    defer gpa.free(fs_bytes);

    const rt = try dev.createResource(.{ .image = .{ .width = W, .height = H, .format = .rgba8_unorm, .usage = .{ .render_target = true } } });
    defer dev.destroyResource(rt);
    const vs = try dev.createShaderModule(.{ .stage = .vertex, .code = vs_bytes });
    defer dev.destroyShaderModule(vs);
    const fs = try dev.createShaderModule(.{ .stage = .fragment, .code = fs_bytes });
    defer dev.destroyShaderModule(fs);

    // A solid-white 64x64 texture so `texel * Color` == Color (= abs(normal), all 1 here):
    // any black output means the texcoord/sampler returned 0 (the blank-glmark2 failure).
    const TW: u32 = 64;
    const TH: u32 = 64;
    const tex = try dev.createResource(.{ .image = .{ .width = TW, .height = TH, .format = .rgba8_unorm, .usage = .{ .sampled = true } } });
    defer dev.destroyResource(tex);
    {
        const dst = try dev.mapResource(tex);
        @memset(dst[0 .. @as(usize, TW) * TH * 4], 255);
    }

    // glmark2 binds position/normal/texcoord in SEPARATE VBOs, but the GLES layer repacks
    // them into one interleaved stream before the HAL (drawTriangleList). Mirror that here:
    // one interleaved vertex = pos(vec3) + normal(vec3) + texcoord(vec2). normal = (1,1,1)
    // so Color = white. texcoord spans the quad.
    const Vtx = extern struct { px: f32, py: f32, pz: f32, nx: f32, ny: f32, nz: f32, u: f32, v: f32 };
    const n: [3]f32 = .{ 1, 1, 1 };
    const quad = [6]Vtx{
        .{ .px = -1, .py = -1, .pz = 0, .nx = n[0], .ny = n[1], .nz = n[2], .u = 0, .v = 0 },
        .{ .px = 1, .py = -1, .pz = 0, .nx = n[0], .ny = n[1], .nz = n[2], .u = 1, .v = 0 },
        .{ .px = 1, .py = 1, .pz = 0, .nx = n[0], .ny = n[1], .nz = n[2], .u = 1, .v = 1 },
        .{ .px = -1, .py = -1, .pz = 0, .nx = n[0], .ny = n[1], .nz = n[2], .u = 0, .v = 0 },
        .{ .px = 1, .py = 1, .pz = 0, .nx = n[0], .ny = n[1], .nz = n[2], .u = 1, .v = 1 },
        .{ .px = -1, .py = 1, .pz = 0, .nx = n[0], .ny = n[1], .nz = n[2], .u = 0, .v = 1 },
    };
    const vbuf = try dev.createResource(.{ .buffer = .{ .size = @sizeOf(@TypeOf(quad)), .usage = .{ .vertex = true } } });
    defer dev.destroyResource(vbuf);
    @memcpy(std.mem.bytesAsSlice(Vtx, try dev.mapResource(vbuf))[0..quad.len], &quad);

    const attrs = [_]hal.VertexAttribute{
        .{ .location = 0, .format = .r32g32b32_float, .offset = 0 }, // position vec3
        .{ .location = 1, .format = .r32g32b32_float, .offset = 12 }, // normal vec3
        .{ .location = 2, .format = .r32g32_float, .offset = 24 }, // texcoord vec2
    };
    const pipe = try dev.createPipeline(.{
        .vertex = vs,
        .fragment = fs,
        .vertex_layout = .{ .stride = @sizeOf(Vtx), .attributes = &attrs },
        .color_format = .rgba8_unorm,
    });
    defer dev.destroyPipeline(pipe);

    const ctx = try dev.createContext();
    defer ctx.deinit();
    {
        const cb = try ctx.beginCommands();
        defer cb.deinit();
        try cb.setRenderTarget(rt);
        try cb.clear(.{ .r = 0, .g = 0, .b = 0, .a = 1 });
        try cb.bindPipeline(pipe);
        try cb.bindTexture(.{ .binding = 2, .image = tex, .filter = .linear, .address_u = .clamp_to_edge, .address_v = .clamp_to_edge });
        try cb.bindVertexBuffer(vbuf);
        try cb.draw(6, 0);
        try ctx.submit(cb);
    }

    const fb = std.mem.bytesAsSlice(u32, try dev.mapResource(rt));
    // Center pixel: texel(white) * Color(abs(normal)=white) = white. A black center means
    // the texcoord/sampler returned 0 -> the glmark2-blank bug.
    const c = fb[(H / 2) * W + W / 2];
    const r: u8 = @truncate(c);
    const g: u8 = @truncate(c >> 8);
    const b: u8 = @truncate(c >> 16);
    try std.testing.expect(r > 200 and g > 200 and b > 200);
}

// DROPPED-TRIANGLE ORACLE (glmark2 build/shading: scattered whole black triangles).
// glmark2 `build`/`shading` render a lit, indexed, many-small-triangle model
// (a horse/cat) and on nvidia show scattered whole-mesh-sized black triangles
// (the background showing through) that flicker as the model rotates. The
// software driver renders the same scene clean. The defect is a per-triangle
// geometry drop in the nvidia path (a constant-FS substitution did not fill the
// holes, so the triangles are not rasterized, not a shading bug).
//
// This oracle reproduces it deterministically: a tessellated grid of many small
// triangles is transformed by an MVP (light-basic.vert), lit per-vertex (the
// exact normalize(NormalMatrix*normal) / dot / max), and rendered through the
// nvidia HAL. The grid is a flat plane facing the camera so every triangle is
// fully covered. Any interior background-colored hole = a dropped triangle. The
// software render of the same mesh is the golden (it has no holes).
const DropMesh = struct {
    verts: []f32, // interleaved pos(vec3) + normal(vec3) + texcoord(vec2)
    indices: []u32,
};

// glmark2 build/shading enable GL_DEPTH_TEST. the oracle matches that fixed-function
// state so it exercises the same depth-tested draw path the real scene drops under.
const drop_depth = true;

// Build a tessellated UV SPHERE (radius 1) with N longitude x N latitude quads, each
// split into two triangles (an indexed triangle list, exactly glmark2's build/horse
// topology). Per-vertex normal = the surface normal (= the unit position), so the
// triangles face every direction (many at grazing angles to the camera, like the
// real model), the case a flat camera-facing grid never exercised.
fn buildDropGrid(gpa: std.mem.Allocator, comptime N: u32) !DropMesh {
    const stride = 8; // pos3 + normal3 + uv2
    const vcount = (N + 1) * (N + 1);
    const verts = try gpa.alloc(f32, vcount * stride);
    errdefer gpa.free(verts);
    const pi = std.math.pi;
    var j: u32 = 0;
    while (j <= N) : (j += 1) {
        var i: u32 = 0;
        while (i <= N) : (i += 1) {
            const u = @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(N));
            const v = @as(f32, @floatFromInt(j)) / @as(f32, @floatFromInt(N));
            const theta = u * 2.0 * pi; // longitude
            const phi = v * pi; // latitude
            const sx = @sin(phi) * @cos(theta);
            const sy = @cos(phi);
            const sz = @sin(phi) * @sin(theta);
            const o = (j * (N + 1) + i) * stride;
            verts[o + 0] = sx;
            verts[o + 1] = sy;
            verts[o + 2] = sz;
            verts[o + 3] = sx; // normal = unit position
            verts[o + 4] = sy;
            verts[o + 5] = sz;
            verts[o + 6] = u;
            verts[o + 7] = v;
        }
    }
    const indices = try gpa.alloc(u32, N * N * 6);
    errdefer gpa.free(indices);
    var t: usize = 0;
    j = 0;
    while (j < N) : (j += 1) {
        var i: u32 = 0;
        while (i < N) : (i += 1) {
            const a = j * (N + 1) + i;
            const b = a + 1;
            const c2 = a + (N + 1);
            const d = c2 + 1;
            indices[t + 0] = a;
            indices[t + 1] = b;
            indices[t + 2] = c2;
            indices[t + 3] = b;
            indices[t + 4] = d;
            indices[t + 5] = c2;
            t += 6;
        }
    }
    return .{ .verts = verts, .indices = indices };
}

// Render the lit mesh to a fresh RT through `dev` and return the caller-owned
// tightly-packed readback. The VS is light-basic.vert verbatim (the `mvp` transform +
// per-vertex diffuse lighting via `normal_mat`); the FS passes Color through. The CPU
// expands the index list into a non-indexed vertex stream (exactly what the GLES layer
// does before the nvidia HAL, which only does DRAW_VERTEX_ARRAY), with depth testing
// on to match glmark2 build/shading.
fn renderDropGrid(gpa: std.mem.Allocator, dev: hal.Device, w: u32, h: u32, mesh: DropMesh, mvp: [16]f32, normal_mat: [16]f32) ![]u32 {
    return renderDropGridCull(gpa, dev, w, h, mesh, mvp, normal_mat, .{});
}

fn renderDropGridCull(gpa: std.mem.Allocator, dev: hal.Device, w: u32, h: u32, mesh: DropMesh, mvp: [16]f32, normal_mat: [16]f32, cull: hal.CullState) ![]u32 {
    const glsl = @import("../../glsl.zig");
    // light-basic.vert verbatim, with glmark2's LightSourcePosition / MaterialDiffuse
    // (compile-time #defines in the real shader) promoted to uniforms so they live in
    // the default uniform block at binding 0, the path the nvidia driver binds. The
    // lighting is per-vertex (gouraud); the FS just passes Color through.
    const vs_src =
        \\attribute vec3 position;
        \\attribute vec3 normal;
        \\attribute vec2 texcoord;
        \\uniform mat4 ModelViewProjectionMatrix;
        \\uniform mat4 NormalMatrix;
        \\uniform vec4 LightSourcePosition;
        \\uniform vec4 MaterialDiffuse;
        \\varying vec4 Color;
        \\varying vec2 TextureCoord;
        \\void main(void) {
        \\  vec3 N = normalize(vec3(NormalMatrix * vec4(normal, 1.0)));
        \\  vec3 L = normalize(LightSourcePosition.xyz);
        \\  float diffuse = max(dot(N, L), 0.0);
        \\  Color = vec4(diffuse * MaterialDiffuse.rgb, MaterialDiffuse.a);
        \\  TextureCoord = texcoord;
        \\  gl_Position = ModelViewProjectionMatrix * vec4(position, 1.0);
        \\}
    ;
    const fs_src =
        \\precision mediump float;
        \\varying vec4 Color;
        \\varying vec2 TextureCoord;
        \\void main(void) { gl_FragColor = Color; }
    ;
    var vsc = try glsl.compileForStageWithLayout(gpa, vs_src, .vertex);
    defer vsc.deinit(gpa);
    const fs_bytes = try glsl.compileForStage(gpa, fs_src, .fragment);
    defer gpa.free(fs_bytes);

    const rt = try dev.createResource(.{ .image = .{ .width = w, .height = h, .format = .bgra8_unorm, .usage = .{ .render_target = true } } });
    defer dev.destroyResource(rt);
    const vs = try dev.createShaderModule(.{ .stage = .vertex, .code = vsc.spirv });
    defer dev.destroyShaderModule(vs);
    const fs = try dev.createShaderModule(.{ .stage = .fragment, .code = fs_bytes });
    defer dev.destroyShaderModule(fs);

    // Fill the default uniform block: the caller's MVP + NormalMatrix, a light from
    // the upper-front, white diffuse. A front-facing lit triangle is bright. A dropped
    // triangle leaves a black background hole in the surface.
    const ublock = try gpa.alloc(u8, vsc.block_size);
    defer gpa.free(ublock);
    @memset(ublock, 0);
    const fset = struct {
        fn set(blk: []u8, members: []const glsl.UniformMember, name: []const u8, vals: []const f32) void {
            for (members) |m| {
                if (std.mem.eql(u8, m.name, name)) {
                    const dst = std.mem.bytesAsSlice(f32, blk[m.offset..][0 .. vals.len * 4]);
                    @memcpy(dst, vals);
                    return;
                }
            }
        }
    }.set;
    fset(ublock, vsc.uniforms, "ModelViewProjectionMatrix", &mvp);
    fset(ublock, vsc.uniforms, "NormalMatrix", &normal_mat);
    fset(ublock, vsc.uniforms, "LightSourcePosition", &.{ 0.5, 0.5, 1.0, 0.0 });
    fset(ublock, vsc.uniforms, "MaterialDiffuse", &.{ 1, 1, 1, 1 });

    const ubo = try dev.createResource(.{ .buffer = .{ .size = vsc.block_size, .usage = .{ .uniform = true } } });
    defer dev.destroyResource(ubo);
    @memcpy((try dev.mapResource(ubo))[0..ublock.len], ublock);

    // Expand the index list into a non-indexed interleaved vertex stream (the GLES
    // layer's drawTriangleList step). 8 floats per vertex.
    const stride_f = 8;
    const exp = try gpa.alloc(f32, mesh.indices.len * stride_f);
    defer gpa.free(exp);
    for (mesh.indices, 0..) |vi, k| {
        @memcpy(exp[k * stride_f ..][0..stride_f], mesh.verts[vi * stride_f ..][0..stride_f]);
    }
    const vbuf = try dev.createResource(.{ .buffer = .{ .size = exp.len * 4, .usage = .{ .vertex = true } } });
    defer dev.destroyResource(vbuf);
    @memcpy(std.mem.bytesAsSlice(f32, try dev.mapResource(vbuf))[0..exp.len], exp);

    const attrs = [_]hal.VertexAttribute{
        .{ .location = 0, .format = .r32g32b32_float, .offset = 0 },
        .{ .location = 1, .format = .r32g32b32_float, .offset = 12 },
        .{ .location = 2, .format = .r32g32_float, .offset = 24 },
    };
    // glmark2 build enables GL_DEPTH_TEST (depth LESS) + GL_CULL_FACE. Match it so
    // the oracle exercises the same fixed-function state the real scene drops under.
    const depth_state: hal.DepthState = if (drop_depth)
        .{ .test_enable = true, .write_enable = true, .compare_op = .less }
    else
        .{};
    const pipe = try dev.createPipeline(.{
        .vertex = vs,
        .fragment = fs,
        .vertex_layout = .{ .stride = stride_f * 4, .attributes = &attrs },
        .color_format = .bgra8_unorm,
        .depth = depth_state,
        .cull = cull,
    });
    defer dev.destroyPipeline(pipe);

    var depth_res: ?*hal.Resource = null;
    if (drop_depth) {
        depth_res = try dev.createResource(.{ .image = .{ .width = w, .height = h, .format = .depth32_float, .usage = .{} } });
    }
    defer if (depth_res) |dr| dev.destroyResource(dr);

    const ctx = try dev.createContext();
    defer ctx.deinit();
    {
        const cb = try ctx.beginCommands();
        defer cb.deinit();
        try cb.setRenderTarget(rt);
        try cb.clear(.{ .r = 0, .g = 0, .b = 0, .a = 1 });
        if (depth_res) |dr| try cb.setDepthTarget(dr, 1.0);
        try cb.bindPipeline(pipe);
        try cb.bindUniformBuffer(0, ubo);
        try cb.bindVertexBuffer(vbuf);
        try cb.draw(@intCast(mesh.indices.len), 0);
        try ctx.submit(cb);
    }

    const fb = std.mem.bytesAsSlice(u32, try dev.mapResource(rt));
    const out = try gpa.alloc(u32, @as(usize, w) * h);
    errdefer gpa.free(out);
    // Tight (w*4) readback: the software RT is tight and nvidia mapResource de-swizzles
    // the block-linear RT into a tight w*4 buffer. (pitchBytes(w)/4 == w only at 256.)
    const pitch_px = w;
    var y: u32 = 0;
    while (y < h) : (y += 1) {
        @memcpy(out[@as(usize, y) * w ..][0..w], fb[@as(usize, y) * pitch_px ..][0..w]);
    }
    return out;
}

// Column-major 4x4 multiply: r = a * b (each [16]f32 is column-major, GL std140).
fn mat4mul(a: [16]f32, b: [16]f32) [16]f32 {
    var r: [16]f32 = undefined;
    var c: usize = 0;
    while (c < 4) : (c += 1) {
        var rr: usize = 0;
        while (rr < 4) : (rr += 1) {
            var s: f32 = 0;
            var k: usize = 0;
            while (k < 4) : (k += 1) s += a[k * 4 + rr] * b[c * 4 + k];
            r[c * 4 + rr] = s;
        }
    }
    return r;
}

test "ORACLE: a lit indexed many-triangle SPHERE under a perspective MVP renders WATERTIGHT (no background-through-surface holes) on the NVIDIA GPU (skips without a GPU)" {
    const gpa = std.testing.allocator;
    const dev = NvDevice.create(gpa) catch return error.SkipZigTest;
    defer dev.deinit();
    const W: u32 = 256;
    const H: u32 = 256;

    // A dense tessellated UV sphere: 48x48 quads = 4608 small triangles, ~13.8k
    // expanded (non-indexed) vertices, glmark2 build/shading's lit-model shape
    // (a perspective MVP, per-vertex normals, light-basic.vert lighting, indexed
    // GL_TRIANGLES expanded to a vertex array). The triangles span every grazing
    // angle, the rotating-perspective case that the flat camera-facing grid never
    // exercises. glmark2 build/shading showed scattered whole black triangles (the
    // background showing through the lit surface) on nvidia. This guards that the
    // nvidia draw path stays watertight for this exact shape.
    const mesh = try buildDropGrid(gpa, 48);
    defer gpa.free(mesh.verts);
    defer gpa.free(mesh.indices);

    // A real perspective MVP (the glmark2 build/horse camera): perspective * translate(z)
    // * rotate. Column-major. f = 1/tan(fovy/2), aspect 1, near 1, far 100.
    const f: f32 = 2.4142136; // fovy ~45deg
    const near: f32 = 1.0;
    const far: f32 = 100.0;
    const persp = [16]f32{
        f, 0, 0,                               0,
        0, f, 0,                               0,
        0, 0, (far + near) / (near - far),     -1,
        0, 0, (2 * far * near) / (near - far), 0,
    };
    // Rotate 0.6 rad about Y then 0.3 about X, translate back along -Z by 4.
    const cy = @cos(@as(f32, 0.6));
    const sy = @sin(@as(f32, 0.6));
    const cx = @cos(@as(f32, 0.3));
    const sx = @sin(@as(f32, 0.3));
    const roty = [16]f32{ cy, 0, -sy, 0, 0, 1, 0, 0, sy, 0, cy, 0, 0, 0, 0, 1 };
    const rotx = [16]f32{ 1, 0, 0, 0, 0, cx, sx, 0, 0, -sx, cx, 0, 0, 0, 0, 1 };
    const trans = [16]f32{ 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, -4, 1 };
    const model = mat4mul(rotx, roty); // rotate
    const mv = mat4mul(trans, model); // then translate
    const mvp = mat4mul(persp, mv);
    // NormalMatrix = the rotation part (orthonormal, so inverse-transpose == itself).
    const normal_mat = model;

    const fb = try renderDropGrid(gpa, dev, W, H, mesh, mvp, normal_mat);
    defer gpa.free(fb);

    const lit = struct {
        fn isLit(p: u32) bool {
            return (p & 0x00ffffff) != 0;
        }
    }.isLit;

    // The sphere actually rendered a substantial lit area.
    var bright: u32 = 0;
    for (fb) |p| {
        if (lit(p)) bright += 1;
    }
    try std.testing.expect(bright > @as(u32, W) * H / 8);

    // Watertight: a dropped triangle leaves a black pixel surrounded by lit surface.
    // Count black pixels bracketed by lit pixels on the row and lit above + below
    // (a true background-through-surface hole, not the model's outer silhouette).
    var holes: u32 = 0;
    var yy: u32 = 1;
    while (yy + 1 < H) : (yy += 1) {
        const row = fb[yy * W ..][0..W];
        var first: i64 = -1;
        var last: i64 = -1;
        var xx: u32 = 0;
        while (xx < W) : (xx += 1) {
            if (lit(row[xx])) {
                if (first < 0) first = xx;
                last = xx;
            }
        }
        if (first < 1) continue;
        var x2: u32 = @intCast(first);
        while (x2 < @as(u32, @intCast(last))) : (x2 += 1) {
            if (!lit(row[x2]) and lit(fb[(yy - 1) * W + x2]) and lit(fb[(yy + 1) * W + x2])) holes += 1;
        }
    }
    if (holes > 0) std.debug.print("\n[DROP-ORACLE] watertight check: interior holes={d} (lit={d})\n", .{ holes, bright });
    try std.testing.expectEqual(@as(u32, 0), holes);
}

// Real glmark2 build/shading HORSE-MESH replay oracle (the decisive instrument).
// The synthetic sphere above renders watertight on nvidia, yet stock glmark2
// build/shading (the lit horse model) drops scattered whole triangles on the real
// GPU while the software driver renders it clean. The drop is mesh-specific: only
// the real artist mesh (thin, irregular, adjacent triangles) reproduces it. This
// oracle replays the actual horse.3ds geometry (extracted from glmark2's data dir:
// 3582 vertices / 7172 triangles, expanded to a non-indexed vertex array the way
// the GLES layer does) under the exact build-scene perspective MVP, through the same
// GLES->HAL lit path (light-basic.vert + depth less + the solid-color FS), on the
// nvidia HAL and the software HAL, then diffs: a pixel covered+lit in software but
// black on nvidia is a dropped-geometry hole. The oracle asserts zero such holes.
//
// horse_mesh.bin layout (little-endian): u32 ntris, u32 pad, 3 f32 min, 3 f32 max,
// then ntris*9 f32 = the model-space triangle-list positions (3 verts x xyz).
const SwDevice = @import("../software/device.zig").Device;

// Build the build-scene perspective MVP for a given Y rotation (degrees) exactly as
// glmark2 SceneBuild::draw does: perspective(fovy, aspect, 2, 2+diameter) * translate(
// -center, -(center.z + 2 + radius)) * rotateY(rot). Column-major (GL std140).
fn buildHorseMvp(mn: [3]f32, mx: [3]f32, rot_deg: f32, aspect: f32) [16]f32 {
    var diff: [3]f32 = undefined;
    var center: [3]f32 = undefined;
    for (0..3) |k| {
        diff[k] = mx[k] - mn[k];
        center[k] = (mx[k] + mn[k]) / 2.0;
    }
    const diameter = @sqrt(diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2]);
    const radius = diameter / 2.0;
    var fovy = 2.0 * std.math.atan(radius / (2.0 + radius));
    fovy = fovy / std.math.pi * 180.0;
    const znear: f32 = 2.0;
    const zfar: f32 = 2.0 + diameter;
    const fv = 1.0 / std.math.tan(fovy * std.math.pi / 180.0 / 2.0);
    const persp = [16]f32{
        fv / aspect, 0,  0,                                   0,
        0,           fv, 0,                                   0,
        0,           0,  (zfar + znear) / (znear - zfar),     -1,
        0,           0,  (2 * zfar * znear) / (znear - zfar), 0,
    };
    const trans = [16]f32{ 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, -center[0], -center[1], -(center[2] + 2.0 + radius), 1 };
    const a = rot_deg * std.math.pi / 180.0;
    const c = @cos(a);
    const s = @sin(a);
    const roty = [16]f32{ c, 0, -s, 0, 0, 1, 0, 0, s, 0, c, 0, 0, 0, 0, 1 };
    const mv = mat4mul(trans, roty);
    return mat4mul(persp, mv);
}

// Parse horse_mesh.bin into a DropMesh: per-vertex pos(vec3)+normal(vec3)+uv(vec2),
// with a face normal per triangle (so the lit FS gives every triangle a solid color),
// indices 0..3N (already a flat triangle list). Caller owns verts+indices.
fn loadHorse(gpa: std.mem.Allocator, bin: []const u8, mn: *[3]f32, mx: *[3]f32) !DropMesh {
    const ntris = std.mem.readInt(u32, bin[0..4], .little);
    for (0..3) |k| mn[k] = @bitCast(std.mem.readInt(u32, bin[8 + k * 4 ..][0..4], .little));
    for (0..3) |k| mx[k] = @bitCast(std.mem.readInt(u32, bin[20 + k * 4 ..][0..4], .little));
    const pf = std.mem.bytesAsSlice(f32, bin[32..][0 .. ntris * 9 * 4]);
    const stride = 8;
    var verts = try gpa.alloc(f32, ntris * 3 * stride);
    errdefer gpa.free(verts);
    var indices = try gpa.alloc(u32, ntris * 3);
    errdefer gpa.free(indices);
    var t: usize = 0;
    while (t < ntris) : (t += 1) {
        const p0 = [3]f32{ pf[t * 9 + 0], pf[t * 9 + 1], pf[t * 9 + 2] };
        const p1 = [3]f32{ pf[t * 9 + 3], pf[t * 9 + 4], pf[t * 9 + 5] };
        const p2 = [3]f32{ pf[t * 9 + 6], pf[t * 9 + 7], pf[t * 9 + 8] };
        const e1 = [3]f32{ p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2] };
        const e2 = [3]f32{ p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2] };
        var n = [3]f32{ e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0] };
        const nl = @sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
        if (nl > 1e-12) for (0..3) |k| {
            n[k] /= nl;
        };
        inline for (.{ p0, p1, p2 }, 0..) |p, vi| {
            const o = (t * 3 + vi) * stride;
            verts[o + 0] = p[0];
            verts[o + 1] = p[1];
            verts[o + 2] = p[2];
            verts[o + 3] = n[0];
            verts[o + 4] = n[1];
            verts[o + 5] = n[2];
            verts[o + 6] = 0;
            verts[o + 7] = 0;
            indices[t * 3 + vi] = @intCast(t * 3 + vi);
        }
    }
    return .{ .verts = verts, .indices = indices };
}

// Render the horse on `dev` and read it back tightly. `is_nvidia` selects the
// block-linear de-swizzle pitch (the nvidia RT) vs a tight w*4 pitch (software).
fn renderHorse(gpa: std.mem.Allocator, dev: hal.Device, w: u32, h: u32, mesh: DropMesh, mvp: [16]f32, normal_mat: [16]f32, cull: hal.CullState) ![]u32 {
    // renderDropGridCull reads back with the nvidia pitch. for the software device at
    // this width pitchBytes(w)==w*4 (256-aligned) so it coincides. We only call this at
    // W==256, where both pitches are identical.
    return renderDropGridCull(gpa, dev, w, h, mesh, mvp, normal_mat, cull);
}

test "DROP-ORACLE: the REAL glmark2 build HORSE mesh renders with NO interior dropped-triangle holes vs the software golden on the NVIDIA GPU (skips without a GPU)" {
    const gpa = std.testing.allocator;
    const dev = NvDevice.create(gpa) catch return error.SkipZigTest;
    defer dev.deinit();
    const W: u32 = 256; // pitchBytes(256)==256*4; software + nvidia readback coincide
    const H: u32 = 256;

    var mn: [3]f32 = undefined;
    var mx: [3]f32 = undefined;
    const mesh = try loadHorse(gpa, @embedFile("horse_mesh.bin"), &mn, &mx);
    defer gpa.free(mesh.verts);
    defer gpa.free(mesh.indices);

    const sw = try SwDevice.create(gpa);
    defer sw.deinit();

    // Sweep several rotations: the drop is view-dependent (a shared edge opens at
    // particular angles). Each frame is a deterministic build-scene MVP.
    const rotations = [_]f32{ 0, 30, 45, 60, 90, 120, 137, 180, 220, 270, 315 };
    var worst_holes: u32 = 0;
    var worst_rot: f32 = 0;
    var first_holes: [8][2]u32 = undefined;
    var first_n: usize = 0;
    for (rotations) |rot| {
        const mvp = buildHorseMvp(mn, mx, rot, 1.0);
        // NormalMatrix = inverse-transpose of the model-view. For a pure rotation+
        // translation the upper-3x3 is the rotation, orthonormal, so it equals itself.
        const a = rot * std.math.pi / 180.0;
        const c = @cos(a);
        const s = @sin(a);
        const normal_mat = [16]f32{ c, 0, -s, 0, 0, 1, 0, 0, s, 0, c, 0, 0, 0, 0, 1 };

        // glmark2 build/shading: glEnable(GL_CULL_FACE) with the default glFrontFace(CCW)
        // + glCullFace(BACK). The GLES layer's wantCullState flips the front winding (GL
        // y-up vs framebuffer y-down) to CW, so the HAL CullState is {back, clockwise},
        // exactly what the real scene binds (and the nvidia draw path must honor it).
        // Verified: with this winding nvidia disagrees with the software golden on only
        // ~200 px (silhouette/crease fill-rule), vs ~7000 px for the inverted winding
        // (which would cull the front faces). Clockwise is unambiguously correct.
        const cull: hal.CullState = .{ .mode = .back, .front_face = .clockwise };
        const nv_fb = try renderHorse(gpa, dev, W, H, mesh, mvp, normal_mat, cull);
        defer gpa.free(nv_fb);
        const sw_fb = try renderHorse(gpa, sw, W, H, mesh, mvp, normal_mat, cull);
        defer gpa.free(sw_fb);

        const lit = struct {
            fn isLit(p: u32) bool {
                return (p & 0x00ffffff) != 0;
            }
        }.isLit;

        // A dropped whole triangle = a connected cluster of interior holes (the back-
        // facing, diffuse=0 triangle overdrawing the lit front surface covers many
        // adjacent pixels, e.g. the head/neck blob seen with culling off). An interior
        // hole pixel: software-lit, nvidia-black, with its full 3x3 software neighbourhood
        // lit (deep in the surface, not on the silhouette where both backends are black on
        // the outer side). We do not require the nvidia neighbours to be lit: the
        // interior of a multi-pixel dropped-triangle blob is itself nvidia-black, so a
        // "surrounded" test would only match the blob's 1px rim and miss the blob. The
        // largest 4-connected component of these holes is the dropped triangle's size: a
        // genuine drop is a multi-pixel blob (the head/neck back-face cluster with culling
        // off), while the unavoidable fill-rule differences between two distinct
        // rasterizers at grazing silhouette creases are isolated single pixels (size-1).
        const is_hole = try gpa.alloc(bool, @as(usize, W) * H);
        defer gpa.free(is_hole);
        @memset(is_hole, false);
        var yy: u32 = 1;
        while (yy + 1 < H) : (yy += 1) {
            var xx: u32 = 1;
            while (xx + 1 < W) : (xx += 1) {
                const i = yy * W + xx;
                if (!(lit(sw_fb[i]) and !lit(nv_fb[i]))) continue;
                var sw_interior = true;
                var dy: i32 = -1;
                while (dy <= 1) : (dy += 1) {
                    var dx: i32 = -1;
                    while (dx <= 1) : (dx += 1) {
                        const j: usize = @intCast(@as(i32, @intCast(i)) + dy * @as(i32, W) + dx);
                        if (!lit(sw_fb[j])) sw_interior = false;
                    }
                }
                if (sw_interior) is_hole[i] = true;
            }
        }
        // Largest 4-connected component of interior holes (iterative flood fill).
        const seen = try gpa.alloc(bool, @as(usize, W) * H);
        defer gpa.free(seen);
        @memset(seen, false);
        var stack = std.ArrayList(usize).empty;
        defer stack.deinit(gpa);
        var max_cluster: u32 = 0;
        var max_at: [2]u32 = .{ 0, 0 };
        var k: usize = 0;
        while (k < is_hole.len) : (k += 1) {
            if (!is_hole[k] or seen[k]) continue;
            var size: u32 = 0;
            stack.clearRetainingCapacity();
            try stack.append(gpa, k);
            seen[k] = true;
            while (stack.pop()) |cur| {
                size += 1;
                const cx = cur % W;
                const cy = cur / W;
                const nbrs = [_]?usize{
                    if (cx > 0) cur - 1 else null,
                    if (cx + 1 < W) cur + 1 else null,
                    if (cy > 0) cur - W else null,
                    if (cy + 1 < H) cur + W else null,
                };
                for (nbrs) |mn2| if (mn2) |n2| {
                    if (is_hole[n2] and !seen[n2]) {
                        seen[n2] = true;
                        try stack.append(gpa, n2);
                    }
                };
            }
            if (size > max_cluster) {
                max_cluster = size;
                max_at = .{ @intCast(k % W), @intCast(k / W) };
            }
        }
        if (max_cluster > worst_holes) {
            worst_holes = max_cluster;
            worst_rot = rot;
            first_holes[0] = max_at;
            first_n = 1;
        }
    }

    if (worst_holes > 0) {
        std.debug.print("\n[HORSE-DROP-ORACLE] worst rot={d}deg largest interior dropped-triangle cluster={d}px near ({d},{d})\n", .{ worst_rot, worst_holes, first_holes[0][0], first_holes[0][1] });
    }
    // A dropped whole triangle is a large interior blob. The unavoidable fill-rule
    // differences between the software and nvidia rasterizers at grazing silhouette
    // creases are tiny. Measured on the real RTX 5070: with the cull fix the largest
    // interior cluster is 3 px (a crease run); without the fix (the bug) the back-face
    // overdraw forms a 92 px blob (a whole dropped triangle near the head). The <= 8
    // threshold sits firmly between the two, so this is a real regression guard: it
    // passes the fixed state with margin and fires if the nvidia draw path stops honoring
    // the pipeline's back-face cull (re-introducing the whole-triangle drops).
    try std.testing.expect(worst_holes <= 8);
}
