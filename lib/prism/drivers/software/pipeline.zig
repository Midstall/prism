const std = @import("std");

/// When set, the software pipeline skips building the 4-wide SIMD quad FS, forcing the scalar
/// fragment path. The EGL layer wires this to PRISM_NOQUAD as a diagnostic toggle (compare the
/// SIMD path against the scalar golden). The quad path is correct: the per-quad speckle that
/// once justified this as a fallback was a dangling-args realloc in vulcan's gather widener
/// (now fixed), so this is no longer a correctness crutch. It is just a knob. Default off.
pub var disable_quad_diag: bool = false;
const hal = @import("../../hal.zig");
const ShaderModule = @import("shader.zig").ShaderModule;
const spirv = @import("../../spirv.zig");
const spirv_jit = @import("spirv_jit.zig");
const target = @import("vulcan-target");

/// A live, callable JIT image (vulcan's host-arch JittedModule), the type
/// `spirv_jit.jitFunction` returns.
const JittedModule = target.native.JittedModule;

/// A real-SPIR-V graphics program: the JITed vertex + fragment shaders plus their
/// interface metadata, built once at pipeline create and reused for every draw.
/// Present only when both stages carried SPIR-V (the general path). Absent for the
/// declarative passthrough path (which the rasterizer handles directly).
pub const ShaderProgram = struct {
    vs: JittedModule,
    fs: JittedModule,
    /// Leading i32 BuiltIn index params the VS takes (gl_VertexIndex [+ gl_InstanceIndex]).
    /// Nonzero = a vertex-pulling VS (zero vertex attributes, positions pulled from a UBO
    /// by the index). The draw supplies the per-vertex index instead of gathering inputs.
    vs_index_count: usize = 0,
    vs_inputs: usize, // scalar VS input count (vertex attributes, scalarized)
    fs_inputs: usize, // scalar FS input count (varyings, scalarized, not gradients)
    /// Each FS f32 input param's varying-buffer index (location*4 + component), in entry-ABI
    /// (declaration / v0,v1,...) order. The rasterizer fills v{k} from the interpolated
    /// varying at fs_input_slots[k]. The FS input param order need not match location order.
    fs_input_slots: [spirv_jit.MAX_FS_INPUTS]u32 = undefined,
    /// The screen-space-derivative gradient entries (dFdx/dFdy of varyings) the FS reads
    /// through its grad_buf pointer, in buffer-index order. The rasterizer computes each
    /// from the per-triangle varying plane and writes them into a small gradient buffer
    /// whose pointer it passes to the FS.
    fs_grads: [spirv_jit.MAX_GRAD_INPUTS]spirv_jit.GradParam = undefined,
    fs_grad_count: usize = 0,
    /// The varying indices feeding the sampler's (u, v) coordinate when they are bare varyings
    /// (null when computed or no sampling). The rasterizer computes the implicit mip LOD from these
    /// varyings' screen-space gradients + the texture size, so a minified mipmapped texture selects
    /// a coarser level (instead of always the base level). See traceTexCoordSlots.
    fs_tex_coord_slots: [2]?u32 = .{ null, null },
    vs_buffers: usize, // UBO/storage buffer pointer params the VS takes (binding order)
    fs_buffers: usize, // UBO/storage/sampler-descriptor/grad_buf pointer params the FS takes
    /// The kind of each FS pointer param (descriptor / sampler_fn / grad_buf), in entry-ABI
    /// order, so the rasterizer supplies the right pointer in each slot.
    fs_buffer_kinds: [spirv_jit.GfxBuffers.max]spirv_jit.BufferKind = undefined,
    /// The binding of each FS `.descriptor` param (-1 = no tag -> declaration order), so the
    /// rasterizer feeds it the UBO bound at that binding. Parallel to fs_buffer_kinds.
    fs_buffer_bindings: [spirv_jit.GfxBuffers.max]i32 = .{-1} ** spirv_jit.GfxBuffers.max,
    /// The binding of each VS `.descriptor` param (-1 = no tag -> declaration order).
    vs_buffer_bindings: [spirv_jit.GfxBuffers.max]i32 = .{-1} ** spirv_jit.GfxBuffers.max,
    /// The kind of each VS pointer param (descriptor / sampler_fn / sampler_desc / ...), in
    /// entry-ABI order. Vertex texture fetch (terrain heightmap displacement) appends a host
    /// sampler-fn + sampler-desc param to the VS, exactly like the FS. The draw path fills each
    /// VS buffer slot per this kind (the sampler fn pointer, the bound texture descriptor, etc.).
    vs_buffer_kinds: [spirv_jit.GfxBuffers.max]spirv_jit.BufferKind = undefined,
    /// Whether the VS samples a texture (vertex texture fetch): vulcan appended a host-sampler
    /// function pointer param to the VS. The draw path passes the software sampler fn there.
    vs_has_sampler_fn: bool = false,
    /// Whether the FS samples a texture: vulcan appended a host-sampler function
    /// pointer param. The draw path passes the software sampler fn there.
    fs_has_sampler_fn: bool = false,
    /// Whether the FS writes gl_FragDepth: the rasterizer takes the fragment depth from
    /// the shader's output (out[frag_depth_index]) rather than the interpolated depth.
    fs_writes_frag_depth: bool = false,
    /// The FS's JITed "main" code address, resolved once at pipeline build. The
    /// per-fragment hot path (`runGraphicsAt`) dispatches through this cached pointer
    /// instead of looking the symbol up by string for every covered fragment (the
    /// profile showed `JittedModule.entry`'s linear symbol-name compare dominating the
    /// per-fragment call overhead). Null only if the JIT produced no "main" symbol.
    fs_entry: ?*const anyopaque = null,

    /// The optional 4-wide quad FS: a second compilation of the same fragment shader,
    /// SIMD-widened so each invocation shades a 2x2 quad (4 fragments at once) lane-wise.
    /// Present only when the FS is in the straight-line / buffer-free widenable subset
    /// (vulcan's `widenGraphics` succeeded). When present the rasterizer's single-sample
    /// path walks fragments in 2x2 quads and runs this once per quad. Otherwise it falls
    /// back to the proven scalar per-fragment `fs`/`fs_entry`. The scalar FS is always
    /// built (golden reference + the fallback), so this is a pure speedup, never a
    /// correctness dependency.
    fs_quad: ?JittedModule = null,
    /// The quad FS's pre-resolved "main" entry address (per the per-quad hot path).
    fs_quad_entry: ?*const anyopaque = null,

    pub fn deinit(self: *ShaderProgram, gpa: std.mem.Allocator) void {
        _ = gpa; // the JittedModule owns its allocator
        self.vs.deinit();
        self.fs.deinit();
        if (self.fs_quad) |*q| q.deinit();
    }
};

pub const Pipeline = struct {
    vertex: *ShaderModule,
    fragment: *ShaderModule,
    layout: hal.VertexLayout,
    color_format: hal.Format,
    /// Depth-test state (default disabled). When test_enable is set and a depth
    /// target is bound, the rasterizer depth-tests each fragment via compare_op and,
    /// if write_enable, writes the passing fragment's depth.
    depth: hal.DepthState = .{},
    /// Back-face cull state (default none). The rasterizer discards triangles whose
    /// screen-space winding is the culled face.
    cull: hal.CullState = .{},
    /// Alpha-blend state (default disabled = passthrough). When enabled the rasterizer
    /// reads the destination pixel and combines it with each fragment per the factors/op.
    blend: hal.BlendState = .{},
    /// MSAA sample count (1/2/4). Default 1 = no MSAA (the single-sample path). When
    /// > 1 the rasterizer renders into an N-sample buffer and the pass resolves it.
    samples: u8 = 1,
    /// GL_SAMPLE_ALPHA_TO_COVERAGE: the fragment alpha becomes a per-sample coverage mask (MSAA only).
    alpha_to_coverage: bool = false,
    /// GL_SAMPLE_COVERAGE: and the covered-sample set with a fixed coverage value (MSAA only).
    sample_coverage: bool = false,
    sample_coverage_value: f32 = 1.0,
    sample_coverage_invert: bool = false,
    /// Stencil-test state (default disabled). When enabled and a stencil target is bound,
    /// the single-sample rasterizer tests/updates the per-pixel stencil buffer (UI clip).
    stencil: hal.StencilState = .{},
    /// Optional back-face stencil state (two-sided stencil). Null = single-face (`stencil` for
    /// both). drawShaded picks per-triangle by winding.
    stencil_back: ?hal.StencilState = null,
    /// Primitive topology (default triangle_list). line_list / point_list drive the line /
    /// point rasterizer instead of triangle coverage.
    topology: hal.Topology = .triangle_list,
    /// Line width in pixels (glLineWidth, default 1) for a line_list draw.
    line_width: f32 = 1.0,
    /// The JITed VS+FS (real SPIR-V execution). Null = declarative passthrough.
    program: ?ShaderProgram = null,
    gpa: ?std.mem.Allocator = null,
    /// Whether `layout.attributes` is an allocation this pipeline owns (createPipeline dupes the
    /// borrowed desc slice). destroyPipeline frees it. Directly-constructed test pipelines leave
    /// this false and keep their borrowed slice.
    owns_attributes: bool = false,

    /// If both shader modules carry SPIR-V, lower + JIT them into a ShaderProgram so
    /// draws run the real shaders. A declarative module leaves `program` null and the
    /// rasterizer uses its position/interpolated-color path. JIT failure (e.g. an
    /// unsupported SPIR-V pattern) also falls back to the declarative path.
    pub fn buildProgram(self: *Pipeline, gpa: std.mem.Allocator) void {
        const vs_code = self.vertex.compute_spirv orelse return;
        const fs_code = self.fragment.compute_spirv orelse return;

        var vfunc = spirv.parseSpirv(gpa, vs_code) catch return;
        defer vfunc.deinit();
        const vinfo = spirv_jit.rewriteGraphics(&vfunc) catch return;
        if (vinfo.stage != .vertex) return;
        var vcompiled = spirv_jit.jitFunction(gpa, &vfunc, "main") catch return;

        var ffunc = spirv.parseSpirv(gpa, fs_code) catch {
            vcompiled.deinit();
            return;
        };
        defer ffunc.deinit();
        const finfo = spirv_jit.rewriteGraphics(&ffunc) catch {
            vcompiled.deinit();
            return;
        };
        if (finfo.stage != .fragment) {
            vcompiled.deinit();
            return;
        }
        const fcompiled = spirv_jit.jitFunction(gpa, &ffunc, "main") catch {
            vcompiled.deinit();
            return;
        };

        self.program = .{
            .vs = vcompiled,
            .fs = fcompiled,
            .vs_index_count = vinfo.index_count,
            .vs_inputs = vinfo.input_count,
            .fs_inputs = finfo.input_count,
            .fs_input_slots = finfo.input_slots,
            .fs_grads = finfo.grads,
            .fs_grad_count = finfo.grad_count,
            .fs_tex_coord_slots = finfo.tex_coord_slots,
            .vs_buffers = vinfo.buffer_count,
            .fs_buffers = finfo.buffer_count,
            .fs_buffer_kinds = finfo.buffer_kinds,
            .fs_buffer_bindings = finfo.buffer_bindings,
            .vs_buffer_bindings = vinfo.buffer_bindings,
            .vs_buffer_kinds = vinfo.buffer_kinds,
            .vs_has_sampler_fn = vinfo.has_sampler_fn,
            .fs_has_sampler_fn = finfo.has_sampler_fn,
            .fs_writes_frag_depth = finfo.writes_frag_depth,
        };
        // Resolve the FS "main" entry address once (the per-fragment caller reuses it).
        self.program.?.fs_entry = spirv_jit.mainEntry(&self.program.?.fs);
        self.gpa = gpa;

        // Try the optional 4-wide quad FS (a separate compilation of the same FS, SIMD-
        // widened). Re-parse from scratch (rewriteGraphicsQuad needs a fresh, non-scalar-
        // rewritten function). On any failure (FS outside the widenable subset, JIT error)
        // the quad path is absent and the rasterizer keeps the scalar path.
        // The scalar program above is unaffected, so this never breaks correctness.
        buildQuad(&self.program.?, gpa, fs_code);
    }

    fn buildQuad(prog: *ShaderProgram, gpa: std.mem.Allocator, fs_code: []const u8) void {
        if (disable_quad_diag) return; // DIAG: force the scalar FS path
        var qfunc = spirv.parseSpirv(gpa, fs_code) catch return;
        defer qfunc.deinit();
        _ = spirv_jit.rewriteGraphicsQuad(&qfunc) catch return;
        var qcompiled = spirv_jit.jitFunction(gpa, &qfunc, "main") catch return;
        const qentry = spirv_jit.mainEntry(&qcompiled) orelse {
            qcompiled.deinit();
            return;
        };
        prog.fs_quad = qcompiled;
        prog.fs_quad_entry = qentry;
    }

    pub fn deinit(self: *Pipeline) void {
        if (self.program) |*p| {
            if (self.gpa) |gpa| p.deinit(gpa);
            self.program = null;
        }
    }
};

test "pipeline holds its shaders and layout" {
    var vs = ShaderModule{ .stage = .vertex, .vertex = .{ .position_attr = 0, .color_attr = 1 } };
    var fs = ShaderModule{ .stage = .fragment, .fragment = .{} };
    const p = Pipeline{ .vertex = &vs, .fragment = &fs, .layout = .{ .stride = 24, .attributes = &.{} }, .color_format = .rgba8_unorm };
    try std.testing.expectEqual(hal.ShaderStage.vertex, p.vertex.stage);
}
