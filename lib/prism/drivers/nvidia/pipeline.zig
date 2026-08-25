const std = @import("std");
const gpumem = @import("gpumem.zig");
const nvidia = @import("nvidia");
const hal = @import("../../hal.zig");
const NvDevice = @import("device.zig").Device;
const ShaderModule = @import("shader.zig").ShaderModule;
const gfx = nvidia.graphics;

/// Shader-heap layout: [VS SPH][VS code][PS SPH][PS code], each SPH 0x80-aligned
/// with its code 0x80 (one SPHV4 header, 32 dwords / 128 B) past it. The VS SPH is
/// at offset 0. The PS SPH is placed after the VS code (aligned up to 0x80) so a
/// large VS never overruns the PS block. The offsets are computed per-pipeline from
/// the actual code sizes: a fixed 0x200 PS offset corrupts the PS SPH/code for any
/// VS over 96 dwords (e.g. real glslang output) and traps an illegal shader.
const VS_SPH_OFF: u32 = 0;
const VS_CODE_OFF: u32 = 0x80;
/// The byte gap from an SPH to its code (the SPHV4 header is 32 dwords = 128 B).
const SPH_CODE_GAP: u32 = 0x80;
/// SPH/code blocks are aligned to this so the shader-program addresses stay valid.
const HEAP_ALIGN: u32 = 0x80;

/// SET_PIPELINE_REGISTER_COUNT floor: even tiny shaders get at least this many GPRs
/// so hardware-delivered FS inputs (barycentrics/sysvals) always have room.
/// A tighter count provokes a benign "Out Of Range Register" warp exception.
const REG_COUNT: u32 = 64;
/// Headroom added above the shader's own GPR usage. The SM delivers FS barycentric
/// coefficients/sysvals into registers above the shader's allocation, so the declared
/// count must exceed the isel's reg_count. Without it the glmark2 desktop blur (120 GPRs)
/// faulted with Xid 13 "Out Of Range Register": 120 declared was not enough.
const REG_HEADROOM: u32 = 16;

/// A HAL pipeline backed by a real NVIDIA GPU: owns the VS+PS shader modules
/// (borrowed, not freed here), the vertex layout, and a GPU shader heap holding
/// the two SPHs and their SASS code. The SPH input/output maps are derived from
/// the vertex layout + a fixed varying convention (location 0 = clip-space
/// position; locations 1..N-1 = generic varyings, passed VS -> PS in order).
pub const Pipeline = struct {
    vs: *ShaderModule,
    ps: *ShaderModule,
    vertex_layout: hal.VertexLayout,
    color_format: hal.Format,
    /// Depth-test state from the pipeline (default disabled). When test_enable is
    /// set and a depth target is bound at submit, the context enables the ZETA
    /// fixed-function depth test (SET_DEPTH_TEST/FUNC/WRITE) for this draw.
    depth: hal.DepthState,
    /// Back-face cull state from the pipeline (default none). Applied at submit via
    /// OGL_SET_CULL / OGL_SET_CULL_FACE / OGL_SET_FRONT_FACE so back faces facing
    /// away from the light don't overdraw the lit front surface (the glmark2
    /// build/shading whole-triangle drops). The default initDrawState disables cull.
    cull: hal.CullState,
    /// Alpha-blend state from the pipeline (default disabled). Applied at submit via
    /// gfx.setBlend (NV9097 SET_BLEND* methods). The default initDrawState leaves
    /// SET_BLEND(0) at reset. A disabled BlendState explicitly clears it to 0.
    blend: hal.BlendState,
    /// Stencil-test state from the pipeline (default disabled). When test_enable is set
    /// and a stencil target is bound at submit, the context binds a Z24S8 ZETA and enables
    /// the fixed-function stencil test (SET_STENCIL_TEST/FUNC/OP/REF/MASK) for this draw.
    stencil: hal.StencilState,
    /// Optional BACK-face stencil state (two-sided stencil). null = single-face. When set + a
    /// stencil target is bound, the context enables SET_TWO_SIDED_STENCIL_TEST and programs the
    /// front (`stencil`) + back (`stencil_back`) methods.
    stencil_back: ?hal.StencilState = null,
    /// Primitive topology (default triangle_list). The submit maps it to the NVCE97 draw
    /// topology so the DA assembles lines / points via the hardware line/point rasterizer.
    topology: hal.Topology = .triangle_list,
    /// Line width in pixels (glLineWidth; default 1) - programmed via SET_ALIASED_LINE_WIDTH_FLOAT.
    line_width: f32 = 1.0,
    /// Whether the VS writes gl_PointSize (declared OMAP_POINT_SIZE). A POINTS draw with this
    /// enables SET_ATTRIBUTE_POINT_SIZE so the DA takes the shader's per-vertex point size.
    writes_point_size: bool = false,
    heap: NvDevice.GpuBuf,
    vs_sph_va: u64,
    ps_sph_va: u64,

    /// Map a HAL CompareOp to the NVIDIA OGL compare value (shared by depth + stencil).
    pub fn nvCompare(op: hal.CompareOp) gfx.DepthFunc {
        return switch (op) {
            .never => .never,
            .less => .less,
            .equal => .equal,
            .less_or_equal => .less_or_equal,
            .greater => .greater,
            .not_equal => .not_equal,
            .greater_or_equal => .greater_or_equal,
            .always => .always,
        };
    }

    /// The NVIDIA SET_DEPTH_FUNC value for this pipeline's compare op.
    pub fn depthFunc(self: *const Pipeline) gfx.DepthFunc {
        return nvCompare(self.depth.compare_op);
    }

    /// The NVIDIA SET_STENCIL_FUNC value for this pipeline's stencil compare op.
    pub fn stencilFunc(self: *const Pipeline) gfx.DepthFunc {
        return nvCompare(self.stencil.compare_op);
    }

    /// Map a HAL StencilOp to the NVIDIA OGL stencil-op value.
    pub fn nvStencilOp(op: hal.StencilOp) gfx.StencilOp {
        return switch (op) {
            .keep => .keep,
            .zero => .zero,
            .replace => .replace,
            .incr_clamp => .incr_clamp,
            .decr_clamp => .decr_clamp,
            .invert => .invert,
            .incr_wrap => .incr_wrap,
            .decr_wrap => .decr_wrap,
        };
    }

    pub fn create(dev: *NvDevice, desc: hal.PipelineDesc) hal.Error!*Pipeline {
        const vs: *ShaderModule = @ptrCast(@alignCast(desc.vertex));
        const ps: *ShaderModule = @ptrCast(@alignCast(desc.fragment));
        const attrs = desc.vertex_layout.attributes;
        // N = vertex-attribute count. location 0 = position, 1..N-1 = varyings.
        // A zero-attribute pipeline is a vertex-pulling draw: the VS reads no fetched
        // attributes (no readsGeneric) and instead pulls its vertices from a UBO array
        // by gl_VertexIndex (sourced from S2R). The varying count then can't come from
        // the attribute layout, so it is recovered from the compiled VS SASS (the
        // number of distinct generic-varying AST stores it emits).
        const n: u32 = @intCast(attrs.len);
        // The number of VS->PS varyings is not the vertex-attribute count minus one:
        // a VS can derive a varying from a vertex input (e.g. frag_pos = f(inPos)) so
        // it produces more varyings than (attr_count - 1), or pass an attribute through
        // unchanged. The compiled VS SASS is authoritative: countVsVaryings counts the
        // distinct generic-output slots it actually stores (AST o[0x80 + slot*0x10]).
        // The old (n - 1) heuristic undercounted whenever a varying did not correspond
        // 1:1 to a fetched attribute (e.g. the derivative test's single position
        // attribute feeding a computed frag_pos varying), leaving the PS imap unset so
        // the FS IPA read 0. Take the max so a pass-through attribute that the optimizer
        // might route differently is never dropped either.
        const sass_varyings = countVsVaryings(vs.code);
        const attr_varyings: u32 = if (n > 0) n - 1 else 0;
        const varyings: u32 = @max(sass_varyings, attr_varyings);

        // Build the VS SPH: reads every fetched generic attribute (none for a pulling
        // VS), writes position, writes one varying per VS->PS varying.
        var vs_sph = gfx.Sph.vertex();
        vs_sph.writesPosition();
        // Enable stream 0 for transform-feedback capture (harmless without SO buffers bound).
        vs_sph.enablesStreamOut();
        // A vertex-pulling VS reads the DA-delivered vertex id (ALD a[0x2fc]). The SPH
        // must declare that sysval input or the DA won't deliver it. Detect it from the
        // compiled SASS (the vertex-id ALD) and declare it.
        if (readsVertexId(vs.code)) vs_sph.readsVertexId();
        if (readsInstanceId(vs.code)) vs_sph.readsInstanceId();
        // gl_PointSize: a VS that stores to a[0x6c] (AST) writes the per-vertex point size.
        // Declare OMAP_POINT_SIZE so the DA takes it (else the fixed SET_POINT_SIZE is used).
        const vs_writes_point_size = writesPointSize(vs.code);
        if (vs_writes_point_size) vs_sph.writesPointSize();
        {
            var i: u32 = 0;
            while (i < n) : (i += 1) vs_sph.readsGeneric(i);
        }
        {
            var v: u32 = 0;
            while (v < varyings) : (v += 1) vs_sph.writesVarying(v);
        }
        // Build the PS SPH: reads each varying (perspective), writes each render target it
        // outputs (MRT: writesColorTarget per target; a single-RT FS declares just RT0).
        var ps_sph = gfx.Sph.fragment();
        {
            var t: u32 = 0;
            while (t < ps.color_targets) : (t += 1) ps_sph.writesColorTarget(t);
        }
        {
            var v: u32 = 0;
            while (v < varyings) : (v += 1) ps_sph.readsVarying(v);
        }
        // gl_FragCoord: a FS that IPA's the window-space POSITION (a[0x70..0x80]) must
        // declare the position input in the imap (SCREEN_LINEAR) or the raster delivers 0.
        if (readsFragCoord(ps.code)) ps_sph.readsPosition();
        // gl_PointCoord: a FS that IPA's the point-sprite attribute (a[0x2e0..0x2e8]) must
        // declare the sprite inputs in the imap (IMAP_POINT_SPRITE_S/T) or the raster
        // delivers garbage. The draw-state enables SET_POINT_SPRITE once at channel init.
        if (readsPointCoord(ps.code)) ps_sph.readsPointSprite();
        // gl_FragDepth: a FS that writes the depth output (isel-detected) must set OMAP_DEPTH
        // so the ROP takes the fragment depth from the shader instead of the interpolated z.
        if (ps.writes_depth) ps_sph.writesDepth();

        // Compute the heap offsets from the actual code sizes: the PS SPH follows
        // the VS code, 0x80-aligned, so a large VS never overruns the PS block.
        const lo = layout(@intCast(vs.code.len), @intCast(ps.code.len));
        const ps_sph_off = lo.ps_sph_off;
        const ps_code_off = lo.ps_code_off;
        const heap_size = lo.heap_size;

        // Lay out + upload the shader heap (write-combining, GPU-coherent).
        const heap = try dev.allocGpu(.system_wc, heap_size);
        errdefer dev.freeGpu(heap);
        const h = heap.bytes;
        // The shader heap is GPU memory, so every one of these goes out through
        // `gpumem` rather than a plain copy. See that file: an optimising build
        // widens a plain copy into stores the mapping refuses.
        gpumem.writeBytes(h[VS_SPH_OFF..], vs_sph.bytes());
        gpumem.writeBytes(h[VS_CODE_OFF..], vs.code);
        gpumem.writeBytes(h[ps_sph_off..], ps_sph.bytes());
        gpumem.writeBytes(h[ps_code_off..], ps.code);

        // Own a copy of the vertex attributes: desc.vertex_layout.attributes is borrowed from
        // the caller (the GLES path builds it in a stack array per draw), but a cached pipeline
        // outlives that call and is replayed on later draws/frames. Retaining the borrowed slice
        // would read freed stack memory. A garbage attribute location then overflows the
        // SET_VERTEX_ATTRIBUTE method index (0x1160 + i*4). Dupe it into pipeline-owned memory.
        const owned_attrs = dev.gpa.dupe(hal.VertexAttribute, desc.vertex_layout.attributes) catch return error.OutOfMemory;
        errdefer dev.gpa.free(owned_attrs);

        const self = dev.gpa.create(Pipeline) catch return error.OutOfMemory;
        self.* = .{
            .vs = vs,
            .ps = ps,
            .vertex_layout = .{ .stride = desc.vertex_layout.stride, .attributes = owned_attrs },
            .color_format = desc.color_format,
            .depth = desc.depth,
            .cull = desc.cull,
            .blend = desc.blend,
            .stencil = desc.stencil,
            .stencil_back = desc.stencil_back,
            .topology = desc.topology,
            .line_width = desc.line_width,
            .writes_point_size = vs_writes_point_size,
            .heap = heap,
            .vs_sph_va = heap.va + VS_SPH_OFF,
            .ps_sph_va = heap.va + ps_sph_off,
        };
        return self;
    }

    /// The computed shader-heap offsets for a VS/PS code pair. The PS SPH is placed
    /// after the VS code (0x80-aligned), its code 0x80 past it, and the heap sized to
    /// fit (min 0x1000). Keeps a large VS (e.g. real-glslang output) from overrunning
    /// the PS SPH/code and trapping an illegal shader.
    const Layout = struct { ps_sph_off: u32, ps_code_off: u32, heap_size: u32 };
    fn layout(vs_code_len: u32, ps_code_len: u32) Layout {
        const vs_code_end = VS_CODE_OFF + vs_code_len;
        const ps_sph_off = std.mem.alignForward(u32, vs_code_end, HEAP_ALIGN);
        const ps_code_off = ps_sph_off + SPH_CODE_GAP;
        const heap_end = ps_code_off + ps_code_len;
        const heap_size = std.mem.alignForward(u32, @max(heap_end, 0x1000), 0x1000);
        return .{ .ps_sph_off = ps_sph_off, .ps_code_off = ps_code_off, .heap_size = heap_size };
    }

    pub fn destroy(self: *Pipeline, dev: *NvDevice) void {
        dev.gpa.free(self.vertex_layout.attributes);
        // Defer the shader-heap free: the GPU channel still points at this heap's VA and a
        // speculative shader prefetch can read it during the idle window. It is freed at the next
        // submit, after a fence and after the shader is rebound. See Device.retired_heaps.
        dev.retireHeap(self.heap);
        dev.gpa.destroy(self);
    }

    /// SET_PIPELINE_REGISTER_COUNT for a shader stage: the GPRs the isel actually
    /// allocated, floored at REG_COUNT so the count is never below the
    /// hardware-delivered-input headroom. A heavier shader (the glmark2 desktop blur
    /// uses > REG_COUNT GPRs) gets its larger real count. A count below its usage
    /// faults the SM with Xid 13 "Out Of Range Register".
    pub fn regCountFor(self: *const Pipeline, mod: *const ShaderModule) u32 {
        _ = self;
        return @max(REG_COUNT, mod.reg_count + REG_HEADROOM);
    }
};

/// Count the VS->PS varyings a compiled vertex shader produces by scanning its SASS
/// for distinct generic-varying attribute stores (AST o[0x80 + slot*0x10]). Used for
/// the vertex-pulling (zero-attribute) path where the varying count can't be derived
/// from the vertex layout. The Vulcan isel emits one AST per scalar component, so a
/// single vec4 varying becomes 4 ASTs at o[0x80..0x8c], all the same 0x10-aligned
/// slot. Hence counting distinct slots. AST = opcode 0x322 (bits 0..11), attribute
/// byte address at bits 40..49. The position store (o[0x70]) is below ATTR_GENERIC0
/// (0x80) and is not a varying.
fn countVsVaryings(code_bytes: []const u8) u32 {
    const ATTR_GENERIC0: u32 = 0x80;
    const VARYING_STRIDE: u32 = 0x10;
    const MAX_VARYINGS: u32 = 32;
    const code = std.mem.bytesAsSlice(u32, code_bytes);
    var seen = [_]bool{false} ** MAX_VARYINGS;
    var i: usize = 0;
    while (i + 4 <= code.len) : (i += 4) {
        if (code[i] & 0xfff != 0x322) continue; // AST
        const addr: u32 = (code[i + 1] >> 8) & 0x3ff; // attribute byte address (bits 40..49)
        if (addr < ATTR_GENERIC0) continue; // the position store, not a varying
        const slot = (addr - ATTR_GENERIC0) / VARYING_STRIDE;
        if (slot < MAX_VARYINGS) seen[slot] = true;
    }
    var count: u32 = 0;
    for (seen) |s| {
        if (s) count += 1;
    }
    return count;
}

/// Whether a compiled VS reads the DA-delivered vertex id (gl_VertexIndex): an ALD
/// from the vertex-id system-value attribute a[0x2fc]. A vertex-pulling VS (S2R-free,
/// no vertex buffer) does this to index its UBO vertex array. ALD = opcode 0x321
/// (bits 0..11), attribute byte address at bits 40..49.
fn readsVertexId(code_bytes: []const u8) bool {
    return readsAttr(code_bytes, 0x2fc);
}

/// Whether a compiled VS reads the DA-delivered instance id (gl_InstanceIndex): an ALD
/// from the instance-id system-value attribute a[0x2f8]. An instanced draw's VS does this
/// to index per-instance data. The SPH must declare it (readsInstanceId) or the DA won't
/// deliver the id and the ALD reads 0.
fn readsInstanceId(code_bytes: []const u8) bool {
    return readsAttr(code_bytes, 0x2f8);
}

/// Whether the SASS contains an ALD (opcode 0x321) from system-value attribute byte
/// address `attr` (bits 40..49). ALD opcode is bits 0..11, addr is bits 40..49.
fn readsAttr(code_bytes: []const u8, attr: u32) bool {
    const code = std.mem.bytesAsSlice(u32, code_bytes);
    var i: usize = 0;
    while (i + 4 <= code.len) : (i += 4) {
        if (code[i] & 0xfff != 0x321) continue; // ALD
        const addr: u32 = (code[i + 1] >> 8) & 0x3ff; // attribute byte address (bits 40..49)
        if (addr == attr) return true;
    }
    return false;
}

/// Whether the SASS stores to attribute `attr` (AST, opcode 0x322). The attribute byte
/// address is at the same bit position as ALD's (bits 40..49). Used to detect a VS writing
/// gl_PointSize (a[0x6c]).
fn writesAttr(code_bytes: []const u8, attr: u32) bool {
    const code = std.mem.bytesAsSlice(u32, code_bytes);
    var i: usize = 0;
    while (i + 4 <= code.len) : (i += 4) {
        if (code[i] & 0xfff != 0x322) continue; // AST
        const addr: u32 = (code[i + 1] >> 8) & 0x3ff;
        if (addr == attr) return true;
    }
    return false;
}
fn writesPointSize(code_bytes: []const u8) bool {
    return writesAttr(code_bytes, 0x6c); // ATTR_POINT_SIZE
}

/// Whether the SASS interpolates (IPA, opcode 0x326) an attribute in byte range
/// [lo, hi). The IPA attribute byte address is stored as addr>>2 at bits 64..72
/// (word 2, low 8 bits). Used to detect a fragment shader reading gl_FragCoord
/// (a[0x70..0x80], the window-space position) or gl_FrontFacing (a[0x3fc]).
fn interpolatesAttrIn(code_bytes: []const u8, lo: u32, hi: u32) bool {
    const code = std.mem.bytesAsSlice(u32, code_bytes);
    var i: usize = 0;
    while (i + 4 <= code.len) : (i += 4) {
        if (code[i] & 0xfff != 0x326) continue; // IPA
        const addr: u32 = (code[i + 2] & 0xff) << 2; // addr>>2 at bits 64..72
        if (addr >= lo and addr < hi) return true;
    }
    return false;
}

/// Whether a compiled FS reads gl_FragCoord: an IPA of the window-space POSITION
/// attribute a[0x70..0x80] (below ATTR_GENERIC0, so it is never a varying IPA).
fn readsFragCoord(code_bytes: []const u8) bool {
    return interpolatesAttrIn(code_bytes, 0x70, 0x80);
}

/// Whether a compiled FS reads gl_PointCoord: an IPA of the point-sprite attribute
/// a[0x2e0] (s) / a[0x2e4] (t). These sit in the system-value region (above any
/// varying), so an IPA there is unambiguously the sprite coord.
fn readsPointCoord(code_bytes: []const u8) bool {
    return interpolatesAttrIn(code_bytes, 0x2e0, 0x2e8);
}
// gl_FrontFacing (a[0x3fc]) needs no SPH imap: the raster always delivers the facing
// attribute (NAK marks no imap bit for 0x3fc either), so there is nothing to declare.

test "nvidia shader-heap layout: a large VS never overruns the PS block" {
    // The PS SPH must always start at or after the end of the VS code, and the PS
    // code must follow its SPH with room for the 0x80 header gap, regardless of size.
    // A fixed 0x200 PS offset corrupted the PS for any VS over 96 dwords. Real
    // glslang VS output is 100 dwords / 400 bytes.
    const cases = [_]struct { vs: u32, ps: u32 }{
        .{ .vs = 92 * 4, .ps = 60 * 4 }, // the hand-built gradient (fit the old layout)
        .{ .vs = 100 * 4, .ps = 60 * 4 }, // real glslang VS (overran the old 0x200 PS)
        .{ .vs = 4096, .ps = 4096 }, // a deliberately huge pair
    };
    for (cases) |c| {
        const lo = Pipeline.layout(c.vs, c.ps);
        // PS SPH starts at/after the VS code end (no overlap).
        try std.testing.expect(lo.ps_sph_off >= VS_CODE_OFF + c.vs);
        // PS code is exactly the header gap past the PS SPH (SET_PIPELINE_PROGRAM
        // points at the SPH, code is 0x80 after it).
        try std.testing.expectEqual(lo.ps_sph_off + SPH_CODE_GAP, lo.ps_code_off);
        // The heap is large enough to hold everything.
        try std.testing.expect(lo.heap_size >= lo.ps_code_off + c.ps);
        // Both blocks are 0x80-aligned.
        try std.testing.expectEqual(@as(u32, 0), lo.ps_sph_off % HEAP_ALIGN);
    }
}
