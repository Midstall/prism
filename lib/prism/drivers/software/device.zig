const std = @import("std");
const hal = @import("../../hal.zig");
const Resource = @import("resource.zig").Resource;
const ShaderModule = @import("shader.zig").ShaderModule;
const Pipeline = @import("pipeline.zig").Pipeline;
const spirv = @import("../../spirv.zig");
const spirv_jit = @import("spirv_jit.zig");

// The callconv(.c) signatures of a JITed compute kernel by bound-buffer arity:
// `main(gid: i32, buf0: [*]u8, ..., buf{n-1}: [*]u8) void`. This mirrors the entry
// ABI the SPIR-V frontend synthesizes (invocation id, then each buffer base pointer
// in declaration order). AArch64 passes these in x0, x1, ... x{n}.
const ComputeFn1 = *const fn (i32, [*]u8) callconv(.c) void;
const ComputeFn2 = *const fn (i32, [*]u8, [*]u8) callconv(.c) void;
const ComputeFn3 = *const fn (i32, [*]u8, [*]u8, [*]u8) callconv(.c) void;
const ComputeFn4 = *const fn (i32, [*]u8, [*]u8, [*]u8, [*]u8) callconv(.c) void;
const ComputeFn5 = *const fn (i32, [*]u8, [*]u8, [*]u8, [*]u8, [*]u8) callconv(.c) void;
const ComputeFn6 = *const fn (i32, [*]u8, [*]u8, [*]u8, [*]u8, [*]u8, [*]u8) callconv(.c) void;
const ComputeFn7 = *const fn (i32, [*]u8, [*]u8, [*]u8, [*]u8, [*]u8, [*]u8, [*]u8) callconv(.c) void;
const ComputeFn8 = *const fn (i32, [*]u8, [*]u8, [*]u8, [*]u8, [*]u8, [*]u8, [*]u8, [*]u8) callconv(.c) void;

/// Run an `n`-buffer compute kernel over `total` global invocation ids, supplying
/// each linear thread index as gid and the bound buffer base pointers. The typed
/// entry is resolved from the JIT image by name ("main"). Returns InvalidArgument
/// if the symbol is missing.
fn runGrid(comptime n: usize, compiled: anytype, total: u64, bases: []const [*]u8) hal.Error!void {
    const Fn = switch (n) {
        1 => ComputeFn1,
        2 => ComputeFn2,
        3 => ComputeFn3,
        4 => ComputeFn4,
        5 => ComputeFn5,
        6 => ComputeFn6,
        7 => ComputeFn7,
        8 => ComputeFn8,
        else => unreachable,
    };
    const f = compiled.entry(Fn, "main") orelse return error.InvalidArgument;
    var g: u64 = 0;
    while (g < total) : (g += 1) {
        const gid: i32 = @intCast(g);
        switch (n) {
            1 => f(gid, bases[0]),
            2 => f(gid, bases[0], bases[1]),
            3 => f(gid, bases[0], bases[1], bases[2]),
            4 => f(gid, bases[0], bases[1], bases[2], bases[3]),
            5 => f(gid, bases[0], bases[1], bases[2], bases[3], bases[4]),
            6 => f(gid, bases[0], bases[1], bases[2], bases[3], bases[4], bases[5]),
            7 => f(gid, bases[0], bases[1], bases[2], bases[3], bases[4], bases[5], bases[6]),
            8 => f(gid, bases[0], bases[1], bases[2], bases[3], bases[4], bases[5], bases[6], bases[7]),
            else => unreachable,
        }
    }
}

pub const Device = struct {
    gpa: std.mem.Allocator,

    pub fn create(gpa: std.mem.Allocator) hal.Error!hal.Device {
        const self = gpa.create(Device) catch return error.OutOfMemory;
        self.* = .{ .gpa = gpa };
        return .{ .ptr = self, .vtable = &vtable };
    }

    /// Formats the rasterizer + present blit actually produce correct pixels for.
    /// The clear/triangle paths and the present blit work on 4-byte RGBA/BGRA
    /// pixels only (no 8-byte fp16 or packed 10-bit path), so we advertise just
    /// the two SDR color formats. No HDR. (Truthful over aspirational.)
    const supported_formats = [_]hal.Format{ .rgba8_unorm, .bgra8_unorm };

    fn caps(ptr: *anyopaque) hal.DeviceCaps {
        _ = ptr;
        return .{
            .device_name = "software renderer",
            .formats = &supported_formats,
            .hdr = hal.DeviceCaps.deriveHdr(&supported_formats),
            // The software driver has a SPIR-V JIT shader path.
            .spirv = true,
            .compute = true,
            .graphics = true,
            .present = true,
            // The rasterizer is allocator-backed with no hardware limit. 16384 is
            // a sane conservative 2D edge cap (matches common GL/Vulkan minimums).
            .max_texture_dim = 16384,
        };
    }

    fn createResource(ptr: *anyopaque, desc: hal.ResourceDesc) hal.Error!*hal.Resource {
        const self: *Device = @ptrCast(@alignCast(ptr));
        const r = self.gpa.create(Resource) catch return error.OutOfMemory;
        errdefer self.gpa.destroy(r);
        const bytes = self.gpa.alloc(u8, Resource.sizeOf(desc)) catch return error.OutOfMemory;
        @memset(bytes, 0);
        r.* = switch (desc) {
            .buffer => .{ .kind = .buffer, .bytes = bytes },
            .image => |i| .{ .kind = .image, .bytes = bytes, .width = i.width, .height = i.height, .format = i.format, .mip_levels = i.mip_levels, .is_cube = i.cube, .depth = i.depth, .is_array = i.array },
        };
        return @ptrCast(r);
    }
    fn destroyResource(ptr: *anyopaque, resource: *hal.Resource) void {
        const self: *Device = @ptrCast(@alignCast(ptr));
        const r: *Resource = @ptrCast(@alignCast(resource));
        self.gpa.free(r.bytes);
        self.gpa.destroy(r);
    }
    fn mapResource(ptr: *anyopaque, resource: *hal.Resource) hal.Error![]u8 {
        _ = ptr;
        const r: *Resource = @ptrCast(@alignCast(resource));
        return r.bytes;
    }
    fn unmapResource(ptr: *anyopaque, resource: *hal.Resource) void {
        _ = ptr;
        _ = resource;
    }
    fn createContext(ptr: *anyopaque) hal.Error!hal.Context {
        const self: *Device = @ptrCast(@alignCast(ptr));
        return @import("context.zig").Context.create(self.gpa);
    }
    fn createShaderModule(ptr: *anyopaque, desc: hal.ShaderModuleDesc) hal.Error!*hal.ShaderModule {
        const self: *Device = @ptrCast(@alignCast(ptr));
        const m = self.gpa.create(ShaderModule) catch return error.OutOfMemory;
        errdefer self.gpa.destroy(m);
        // A compute module retains a copy of the SPIR-V (owned by self.gpa).
        m.* = try ShaderModule.decode(self.gpa, desc.stage, desc.code);
        return @ptrCast(m);
    }
    fn destroyShaderModule(ptr: *anyopaque, module: *hal.ShaderModule) void {
        const self: *Device = @ptrCast(@alignCast(ptr));
        const m: *ShaderModule = @ptrCast(@alignCast(module));
        m.deinit();
        self.gpa.destroy(m);
    }
    /// One-shot HAL compute dispatch on the host CPU. Lowers the .compute ShaderModule's
    /// SPIR-V to Vulcan IR (prism.spirv.parseSpirv), JITs to native AArch64 (spirv_jit),
    /// and runs the kernel once per global invocation id. Entry ABI:
    ///   main(gid: i32, buf0: ptr, buf1: ptr, ...) -- AArch64: x0=gid, x1..=buffers.
    /// Grid: total = groups.x * groups.y * groups.z invocations, each gets its linear index as gid.
    fn dispatchCompute(ptr: *anyopaque, d: hal.ComputeDispatch) hal.Error!void {
        const self: *Device = @ptrCast(@alignCast(ptr));
        const m: *ShaderModule = @ptrCast(@alignCast(d.shader));
        if (m.stage != .compute) return error.InvalidArgument;
        const code = m.compute_spirv orelse return error.InvalidArgument;
        if (d.buffers.len == 0 or d.buffers.len > 8) return error.InvalidArgument;

        // SPIR-V -> Vulcan IR. Owned here. Freed after the JIT image is built.
        var func = spirv.parseSpirv(self.gpa, code) catch return error.InvalidArgument;
        defer func.deinit();

        // Vulcan IR -> live AArch64 image, callable in-process.
        var compiled = spirv_jit.jitFunction(self.gpa, &func, "main") catch return error.InvalidArgument;
        defer compiled.deinit();

        // Buffer base pointers, in HAL bind order (buffers[i] -> the i-th storage
        // buffer parameter), matching the frontend's declaration-order synthesis.
        var bases: [8][*]u8 = undefined;
        for (d.buffers, 0..) |res, i| {
            const r: *Resource = @ptrCast(@alignCast(res));
            bases[i] = r.bytes.ptr;
        }

        const total: u64 = @as(u64, d.groups[0]) * d.groups[1] * d.groups[2];

        // Pick the typed entry by buffer arity and run the grid. Each kernel call is
        // one invocation: gid = the linear thread index.
        switch (d.buffers.len) {
            inline 1, 2, 3, 4, 5, 6, 7, 8 => |n| try runGrid(n, &compiled, total, bases[0..n]),
            else => unreachable,
        }
    }
    fn createPipeline(ptr: *anyopaque, desc: hal.PipelineDesc) hal.Error!*hal.Pipeline {
        const self: *Device = @ptrCast(@alignCast(ptr));
        const p = self.gpa.create(Pipeline) catch return error.OutOfMemory;
        // Keep a copy of the vertex attributes: desc.vertex_layout.attributes is borrowed (the
        // GLES path builds it in a per-draw stack array), but a cached pipeline outlives that
        // call and is replayed later. Retaining the borrowed slice would read freed stack memory.
        // createPipeline-built pipelines are freed via destroyPipeline, which frees this copy.
        // In-module tests that construct Pipeline directly keep their own slice.
        const owned_attrs = self.gpa.dupe(hal.VertexAttribute, desc.vertex_layout.attributes) catch {
            self.gpa.destroy(p);
            return error.OutOfMemory;
        };
        p.* = .{
            .vertex = @ptrCast(@alignCast(desc.vertex)),
            .fragment = @ptrCast(@alignCast(desc.fragment)),
            .layout = .{ .stride = desc.vertex_layout.stride, .attributes = owned_attrs },
            .owns_attributes = true,
            .color_format = desc.color_format,
            .depth = desc.depth,
            .cull = desc.cull,
            .blend = desc.blend,
            .samples = desc.samples,
            .alpha_to_coverage = desc.alpha_to_coverage,
            .sample_coverage = desc.sample_coverage,
            .sample_coverage_value = desc.sample_coverage_value,
            .sample_coverage_invert = desc.sample_coverage_invert,
            .stencil = desc.stencil,
            .stencil_back = desc.stencil_back,
            .topology = desc.topology,
            .line_width = desc.line_width,
        };
        // If the modules carry real SPIR-V, lower + JIT the VS/FS now (reused per
        // draw). A declarative passthrough module leaves program null.
        p.buildProgram(self.gpa);
        return @ptrCast(p);
    }
    fn destroyPipeline(ptr: *anyopaque, pipeline: *hal.Pipeline) void {
        const self: *Device = @ptrCast(@alignCast(ptr));
        const p: *Pipeline = @ptrCast(@alignCast(pipeline));
        if (p.owns_attributes) self.gpa.free(p.layout.attributes);
        p.deinit();
        self.gpa.destroy(p);
    }
    fn createSurface(ptr: *anyopaque, platform_surface: *anyopaque) hal.Error!*hal.Surface {
        const self: *Device = @ptrCast(@alignCast(ptr));
        const s = self.gpa.create(@import("surface.zig").Surface) catch return error.OutOfMemory;
        s.* = .{ .platform = @ptrCast(@alignCast(platform_surface)) };
        return @ptrCast(s);
    }
    fn destroySurface(ptr: *anyopaque, surface: *hal.Surface) void {
        const self: *Device = @ptrCast(@alignCast(ptr));
        self.gpa.destroy(@as(*@import("surface.zig").Surface, @ptrCast(@alignCast(surface))));
    }
    fn deinit(ptr: *anyopaque) void {
        const self: *Device = @ptrCast(@alignCast(ptr));
        const gpa = self.gpa;
        gpa.destroy(self);
    }

    const vtable = hal.Device.VTable{
        .caps = &caps,
        .createContext = &createContext,
        .createResource = &createResource,
        .destroyResource = &destroyResource,
        .mapResource = &mapResource,
        .unmapResource = &unmapResource,
        .createShaderModule = &createShaderModule,
        .destroyShaderModule = &destroyShaderModule,
        .dispatchCompute = &dispatchCompute,
        .createPipeline = &createPipeline,
        .destroyPipeline = &destroyPipeline,
        .createSurface = &createSurface,
        .destroySurface = &destroySurface,
        .occlusionSampleCount = &occlusionSampleCount,
        .captureTransformFeedback = &captureTransformFeedback,
        .deinit = &deinit,
    };

    /// Transform-feedback capture: delegate to the context's per-vertex VS-run-and-write path.
    /// Device-independent (operates on the passed pipeline/buffers), so `ptr` is unused.
    fn captureTransformFeedback(ptr: *anyopaque, cap: hal.TransformFeedbackCapture) hal.Error!usize {
        _ = ptr;
        return @import("context.zig").captureTransformFeedback(cap);
    }

    /// The rasterizer's cumulative primary-color sample-write counter (occlusion queries read its
    /// delta). Device-independent (a process-wide threadlocal in raster.zig), so `ptr` is unused.
    fn occlusionSampleCount(ptr: *anyopaque) u64 {
        _ = ptr;
        return @import("raster.zig").samples_written;
    }
};

test "software device allocates and maps a resource" {
    const gpa = std.testing.allocator;
    const dev = try Device.create(gpa);
    defer dev.deinit();
    const r = try dev.createResource(.{ .image = .{ .width = 2, .height = 2, .format = .rgba8_unorm } });
    defer dev.destroyResource(r);
    const mapped = try dev.mapResource(r);
    try std.testing.expectEqual(@as(usize, 2 * 2 * 4), mapped.len);
}

test "software compute dispatch: output[i] = input[i] + 0x100 over a grid" {
    const gpa = std.testing.allocator;
    const op = @import("vulcan-spirv").opcodes;
    const sc = op.StorageClass;

    // The exact M3 test kernel, hand-built as SPIR-V:
    //   void main() { uint i = gl_GlobalInvocationID.x; out[i] = in[i] + 0x100; }
    // ids: void=1 int=2 uint=3 v3uint=4 pInV3=5 pInU=6 arr=7 struct=8
    //      pSb=9 pSbInt=10 voidfn=11 c0=12 c100=13 gid=14 inBuf=15 outBuf=16
    //      main=17 entry=18 xptr=19 i=20 inElem=21 v=22 sum=23 outElem=24.
    var b = try @import("vulcan-spirv").binary.Builder.init(gpa, 25);
    defer b.deinit(gpa);
    try b.emit(gpa, op.Decorate, &.{ 14, op.Decoration.builtin, op.BuiltIn.global_invocation_id });
    try b.emit(gpa, op.TypeVoid, &.{1});
    try b.emit(gpa, op.TypeInt, &.{ 2, 32, 1 });
    try b.emit(gpa, op.TypeInt, &.{ 3, 32, 0 });
    try b.emit(gpa, op.TypeVector, &.{ 4, 3, 3 });
    try b.emit(gpa, op.TypePointer, &.{ 5, sc.input, 4 });
    try b.emit(gpa, op.TypePointer, &.{ 6, sc.input, 3 });
    try b.emit(gpa, op.TypeRuntimeArray, &.{ 7, 2 });
    try b.emit(gpa, op.TypeStruct, &.{ 8, 7 });
    try b.emit(gpa, op.TypePointer, &.{ 9, sc.storage_buffer, 8 });
    try b.emit(gpa, op.TypePointer, &.{ 10, sc.storage_buffer, 2 });
    try b.emit(gpa, op.TypeFunction, &.{ 11, 1 });
    try b.emit(gpa, op.Constant, &.{ 3, 12, 0 });
    try b.emit(gpa, op.Constant, &.{ 2, 13, 0x100 });
    try b.emit(gpa, op.Variable, &.{ 5, 14, sc.input });
    try b.emit(gpa, op.Variable, &.{ 9, 15, sc.storage_buffer }); // in (buffer 0)
    try b.emit(gpa, op.Variable, &.{ 9, 16, sc.storage_buffer }); // out (buffer 1)
    try b.emit(gpa, op.Function, &.{ 1, 17, 0, 11 });
    try b.emit(gpa, op.Label, &.{18});
    try b.emit(gpa, op.AccessChain, &.{ 6, 19, 14, 12 }); // &gid.x
    try b.emit(gpa, op.Load, &.{ 3, 20, 19 }); // i = gid.x
    try b.emit(gpa, op.AccessChain, &.{ 10, 21, 15, 12, 20 }); // &in[i]
    try b.emit(gpa, op.Load, &.{ 2, 22, 21 }); // v = in[i]
    try b.emit(gpa, op.IAdd, &.{ 2, 23, 22, 13 }); // v + 0x100
    try b.emit(gpa, op.AccessChain, &.{ 10, 24, 16, 12, 20 }); // &out[i]
    try b.emit(gpa, op.Store, &.{ 24, 23 }); // out[i] = v + 0x100
    try b.emit(gpa, op.Return, &.{});
    try b.emit(gpa, op.FunctionEnd, &.{});

    const spv = std.mem.sliceAsBytes(b.words.items);

    const dev = try Device.create(gpa);
    defer dev.deinit();

    const shader = try dev.createShaderModule(.{ .stage = .compute, .code = spv });
    defer dev.destroyShaderModule(shader);

    const n = 256;
    const in_buf = try dev.createResource(.{ .buffer = .{ .size = n * 4 } });
    defer dev.destroyResource(in_buf);
    const out_buf = try dev.createResource(.{ .buffer = .{ .size = n * 4 } });
    defer dev.destroyResource(out_buf);

    const in_map = std.mem.bytesAsSlice(u32, try dev.mapResource(in_buf));
    const out_map = std.mem.bytesAsSlice(u32, try dev.mapResource(out_buf));
    for (0..n) |i| in_map[i] = @intCast(i);
    @memset(out_map[0..n], 0);

    try dev.dispatchCompute(.{
        .shader = shader,
        .buffers = &.{ in_buf, out_buf },
        .groups = .{ n, 1, 1 },
    });

    for (0..n) |i| {
        try std.testing.expectEqual(@as(u32, @as(u32, @intCast(i)) + 0x100), out_map[i]);
    }
}
