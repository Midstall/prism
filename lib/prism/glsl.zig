//! GLSL ES 1.00 shader front end: thin seam onto Vulcan's vulcan-glsl compiler.
//! Emits SPIR-V with the same glslang-style shapes spirv.zig already lowers natively.
//! No change to the SPIR-V -> native lowering path is needed.

const std = @import("std");
const glsl = @import("vulcan-glsl");
const hal = @import("hal.zig");

pub const Error = glsl.SpirvError;

/// The pipeline stage a GLSL shader compiles for. Re-exported from Vulcan. Mapped from
/// the HAL `ShaderStage` by `compileToSpirv`.
pub const Stage = glsl.Stage;

/// Compile GLSL ES 1.00 source for `stage` to a SPIR-V binary (a `[]u32` word stream).
/// Caller owns the result (free it). The stage resolves GLSL ES 1.00 `varying` (an
/// `out` in a vertex shader, an `in` in a fragment shader) and the implicit
/// `gl_FragColor` color output.
pub fn compileShaderToSpirv(allocator: std.mem.Allocator, source: []const u8, stage: Stage) Error![]u32 {
    return glsl.compileShaderToSpirv(allocator, source, stage);
}

/// Compile GLSL ES source for a HAL `ShaderStage` (the EGL/GLES caller's stage type)
/// to a SPIR-V *byte* stream the HAL `createShaderModule`/`prism.spirv.parseSpirv`
/// path consumes. Caller owns the returned bytes (free with the same allocator).
pub fn compileForStage(allocator: std.mem.Allocator, source: []const u8, stage: hal.ShaderStage) Error![]u8 {
    const vk_stage: Stage = switch (stage) {
        .vertex => .vertex,
        .fragment => .fragment,
        .compute => .compute,
    };
    const words = try compileShaderToSpirv(allocator, source, vk_stage);
    defer allocator.free(words);
    const bytes = try allocator.alloc(u8, words.len * 4);
    @memcpy(bytes, std.mem.sliceAsBytes(words));
    return bytes;
}

/// One member of the default uniform block: its name (heap-duped), byte offset within the
/// block, and float count. Re-exported from Vulcan's lowering so the GLES layer can resolve
/// glGetUniformLocation + write glUniform* bytes at the matching offset.
pub const UniformMember = struct {
    name: []const u8,
    /// Byte offset of this uniform within the default uniform block (tight-packed std140
    /// for the all-vec4/mat4 case es2gears uses).
    offset: u32,
    /// Number of floats one element occupies (mat4 = 16, vec4 = 4, scalar = 1). For an
    /// array uniform this is the per-element stride. For a scalar uniform it is the whole
    /// member.
    float_count: u32,
    /// Matrix dimension (2/3/4) for a matN uniform, else 0.
    mat_dim: u8,
    /// Array length for an array uniform (`uniform vec3 c[3];` -> 3), else 1. Element `i`
    /// of the array lives at byte `offset + i * float_count * 4`.
    array_len: u32 = 1,
};

/// One `uniform sampler2D` member: its name (heap-duped) and SPIR-V binding. Samplers
/// start at binding 2. Bindings 0/1 are reserved for the per-stage default uniform
/// blocks, so a sampler descriptor never shares the nvidia backend's shared-constant-bank
/// slot the uniform-block pointer uses. The GLES layer maps a sampler name -> binding so
/// the texture unit set via glUniform1i binds the right texture at draw time.
pub const SamplerMember = struct {
    name: []const u8,
    binding: u32,
    /// true for a `samplerCube` (sampled by a vec3 direction). false for a plain sampler2D.
    /// The GLES layer resolves a cube sampler's texture from GL_TEXTURE_CUBE_MAP at draw.
    cube: bool = false,
    /// true for a `sampler3D` (a vec3 coordinate into a volume). Resolved from GL_TEXTURE_3D.
    tex3d: bool = false,
    /// true for a `sampler2DArray` (a vec3 (u,v,layer)). Resolved from GL_TEXTURE_2D_ARRAY.
    tex2darray: bool = false,
};

/// One vertex-attribute member (re-exported from Vulcan): its name (heap-duped) and the
/// pipeline location the GLSL lowering assigned it. The GLES layer records these at link
/// time and resolves glGetAttribLocation/glGetActiveAttrib + the draw-time vertex layout
/// against them, instead of a name-substring heuristic.
pub const AttributeMember = struct {
    name: []const u8,
    location: u32,
    /// Component count (1 scalar, 2..4 vector), reported by glGetActiveAttrib.
    components: u8,
};

/// One vertex-shader output varying (re-exported from Vulcan): its name (heap-duped) and the
/// pipeline location the GLSL lowering assigned it, plus the component count. The GLES layer
/// records these at link time and maps a glTransformFeedbackVaryings capture name to the VS
/// output slot (location*4 + component) the software driver reads when capturing transform
/// feedback. Builtins (gl_Position/gl_PointSize) are not surfaced here.
pub const OutputMember = struct {
    name: []const u8,
    location: u32,
    components: u8,
};

/// One named uniform interface block (re-exported from Vulcan): its name (heap-duped), its
/// GL binding point (`layout(binding=N)`, else its declaration index), and its byte size
/// (GL_UNIFORM_BLOCK_DATA_SIZE). The GLES layer resolves glGetUniformBlockIndex to this and
/// routes the glBindBufferBase'd buffer to the block at draw time.
pub const UniformBlock = struct {
    name: []const u8,
    binding: u32,
    byte_offset: u32,
    /// std140 block size (GL_UNIFORM_BLOCK_DATA_SIZE), rounded up to 16.
    byte_size: u32,
    /// The std140<->tight repack table (one entry per member, see prism.glsl.UniformBlockMember).
    /// For an all-16-byte-member block each entry is an identity copy. Heap-owned. Freed by
    /// CompiledShader.deinit. Empty only for a block with no members.
    members: []const UniformBlockMember = &.{},
};

/// One member of a named uniform block (re-exported from Vulcan): how to repack it from a
/// std140 user buffer (glBindBufferBase) into the tight layout the shader reads. All offsets
/// are block-relative. The host copies `unit_count` chunks of `copy_bytes`. Chunk `u` moves
/// `std140_offset + u*std140_stride` (user buffer) to `tight_offset + u*tight_stride` (tight UBO).
pub const UniformBlockMember = glsl.UniformBlockMember;

/// A compiled shader: SPIR-V bytes + the default-uniform-block layout + the sampler
/// members + the vertex attributes + the named uniform blocks. Caller owns all and frees
/// them via `deinit`.
pub const CompiledShader = struct {
    spirv: []u8,
    uniforms: []UniformMember,
    samplers: []SamplerMember,
    attributes: []AttributeMember,
    outputs: []OutputMember,
    uniform_blocks: []UniformBlock,
    block_size: u32,

    pub fn deinit(self: *CompiledShader, allocator: std.mem.Allocator) void {
        allocator.free(self.spirv);
        for (self.uniforms) |u| allocator.free(u.name);
        allocator.free(self.uniforms);
        for (self.samplers) |s| allocator.free(s.name);
        allocator.free(self.samplers);
        for (self.attributes) |a| allocator.free(a.name);
        allocator.free(self.attributes);
        for (self.outputs) |o| allocator.free(o.name);
        allocator.free(self.outputs);
        for (self.uniform_blocks) |b| {
            allocator.free(b.name);
            allocator.free(b.members);
        }
        allocator.free(self.uniform_blocks);
        self.* = undefined;
    }
};

/// Compile GLSL ES source for a HAL `ShaderStage` to SPIR-V bytes and the
/// default-uniform-block layout (name -> byte offset), so the GLES layer can implement
/// glGetUniformLocation + glUniform*. Caller owns the result (`deinit`).
pub fn compileForStageWithLayout(allocator: std.mem.Allocator, source: []const u8, stage: hal.ShaderStage) Error!CompiledShader {
    const vk_stage: Stage = switch (stage) {
        .vertex => .vertex,
        .fragment => .fragment,
        .compute => .compute,
    };
    var c = try glsl.compileShaderWithLayout(allocator, source, vk_stage);
    // Transfer the SPIR-V words into a byte buffer. Transfer the uniform names verbatim.
    const bytes = allocator.alloc(u8, c.spirv.len * 4) catch |e| {
        c.deinit(allocator);
        return e;
    };
    @memcpy(bytes, std.mem.sliceAsBytes(c.spirv));
    allocator.free(c.spirv);
    c.spirv = &.{};

    const members = allocator.alloc(UniformMember, c.uniforms.len) catch |e| {
        c.deinit(allocator);
        allocator.free(bytes);
        return e;
    };
    for (c.uniforms, 0..) |u, i| {
        members[i] = .{ .name = u.name, .offset = u.offset_floats * 4, .float_count = u.float_count, .mat_dim = u.mat_dim, .array_len = u.array_len };
    }
    // Transfer the sampler members (name -> binding). Names transferred verbatim.
    const samplers = allocator.alloc(SamplerMember, c.samplers.len) catch |e| {
        c.deinit(allocator);
        allocator.free(bytes);
        allocator.free(members);
        return e;
    };
    for (c.samplers, 0..) |s, i| samplers[i] = .{ .name = s.name, .binding = s.binding, .cube = s.cube, .tex3d = s.tex3d, .tex2darray = s.tex2darray };
    // Transfer the attribute members (name -> location + components). Names transferred verbatim.
    const attributes = allocator.alloc(AttributeMember, c.attributes.len) catch |e| {
        c.deinit(allocator);
        allocator.free(bytes);
        allocator.free(members);
        allocator.free(samplers);
        return e;
    };
    for (c.attributes, 0..) |a, i| attributes[i] = .{ .name = a.name, .location = a.location, .components = a.components };
    // Transfer the VS output varyings (name -> location + components). Names transferred verbatim.
    const outputs = allocator.alloc(OutputMember, c.outputs.len) catch |e| {
        c.deinit(allocator);
        allocator.free(bytes);
        allocator.free(members);
        allocator.free(samplers);
        allocator.free(attributes);
        return e;
    };
    for (c.outputs, 0..) |o, i| outputs[i] = .{ .name = o.name, .location = o.location, .components = o.components };
    // Transfer the named uniform blocks (names transferred verbatim).
    const uniform_blocks = allocator.alloc(UniformBlock, c.uniform_blocks.len) catch |e| {
        c.deinit(allocator);
        allocator.free(bytes);
        allocator.free(members);
        allocator.free(samplers);
        allocator.free(attributes);
        allocator.free(outputs);
        return e;
    };
    // Transfer both the name and the std140 member table verbatim (freed by our deinit).
    for (c.uniform_blocks, 0..) |b, i| uniform_blocks[i] = .{ .name = b.name, .binding = b.binding, .byte_offset = b.byte_offset, .byte_size = b.byte_size, .members = b.members };
    const block_size = c.block_size;
    // Free the vulcan member arrays (not the names, which were transferred above).
    allocator.free(c.uniforms);
    c.uniforms = &.{};
    allocator.free(c.samplers);
    c.samplers = &.{};
    allocator.free(c.attributes);
    c.attributes = &.{};
    allocator.free(c.outputs);
    c.outputs = &.{};
    allocator.free(c.uniform_blocks);
    c.uniform_blocks = &.{};
    return .{ .spirv = bytes, .uniforms = members, .samplers = samplers, .attributes = attributes, .outputs = outputs, .uniform_blocks = uniform_blocks, .block_size = block_size };
}

test "glsl: the gradient-triangle VS+FS compile from GLSL ES 1.00 source to SPIR-V bytes" {
    const gpa = std.testing.allocator;
    const vs_src =
        \\attribute vec2 aPos;
        \\attribute vec3 aColor;
        \\varying vec3 vColor;
        \\void main() { gl_Position = vec4(aPos, 0.0, 1.0); vColor = aColor; }
    ;
    const fs_src =
        \\precision mediump float;
        \\varying vec3 vColor;
        \\void main() { gl_FragColor = vec4(vColor, 1.0); }
    ;
    const vs = try compileForStage(gpa, vs_src, .vertex);
    defer gpa.free(vs);
    const fs = try compileForStage(gpa, fs_src, .fragment);
    defer gpa.free(fs);
    // SPIR-V magic in the first word (little-endian bytes).
    try std.testing.expectEqual(@as(u8, 0x03), vs[0]);
    try std.testing.expectEqual(@as(u8, 0x02), vs[1]);
    try std.testing.expect(vs.len % 4 == 0 and fs.len % 4 == 0);

    // Both lower back to Vulcan IR through Prism's SPIR-V seam (the draw-time path).
    const prism_spirv = @import("spirv.zig");
    var vf = try prism_spirv.parseSpirv(gpa, vs);
    vf.deinit();
    var ff = try prism_spirv.parseSpirv(gpa, fs);
    ff.deinit();
}

test "glsl: the es2gears VS compiles + surfaces its uniform layout + round-trips" {
    const gpa = std.testing.allocator;
    const vs_src =
        \\attribute vec3 position;
        \\attribute vec3 normal;
        \\uniform mat4 ModelViewProjectionMatrix;
        \\uniform mat4 NormalMatrix;
        \\uniform vec4 LightSourcePosition;
        \\uniform vec4 MaterialColor;
        \\varying vec4 Color;
        \\void main(void) {
        \\  vec3 N = normalize(vec3(NormalMatrix * vec4(normal, 1.0)));
        \\  vec3 L = normalize(LightSourcePosition.xyz);
        \\  float diffuse = max(dot(N, L), 0.0);
        \\  float ambient = 0.2;
        \\  Color = vec4((ambient + diffuse) * MaterialColor.xyz, MaterialColor.a);
        \\  gl_Position = ModelViewProjectionMatrix * vec4(position, 1.0);
        \\}
    ;
    var c = try compileForStageWithLayout(gpa, vs_src, .vertex);
    defer c.deinit(gpa);
    try std.testing.expectEqual(@as(usize, 4), c.uniforms.len);
    try std.testing.expectEqualStrings("ModelViewProjectionMatrix", c.uniforms[0].name);
    try std.testing.expectEqual(@as(u32, 0), c.uniforms[0].offset);
    try std.testing.expectEqual(@as(u32, 64), c.uniforms[1].offset); // NormalMatrix @ byte 64
    try std.testing.expectEqual(@as(u32, 128), c.uniforms[2].offset); // LightSourcePosition @ 128
    try std.testing.expectEqual(@as(u32, 144), c.uniforms[3].offset); // MaterialColor @ 144
    try std.testing.expectEqual(@as(u32, 160), c.block_size);
    // Round-trips through Prism's SPIR-V seam (the draw-time path).
    const prism_spirv = @import("spirv.zig");
    var vf = try prism_spirv.parseSpirv(gpa, c.spirv);
    vf.deinit();
}

test "glsl: a textured FS surfaces its sampler members + coexisting UBO uniforms" {
    const gpa = std.testing.allocator;
    const fs_src =
        \\precision mediump float;
        \\uniform sampler2D uTex;
        \\uniform vec4 uTint;
        \\varying vec2 vUV;
        \\void main(void) { gl_FragColor = texture2D(uTex, vUV) * uTint; }
    ;
    var c = try compileForStageWithLayout(gpa, fs_src, .fragment);
    defer c.deinit(gpa);
    try std.testing.expectEqual(@as(usize, 1), c.samplers.len);
    try std.testing.expectEqualStrings("uTex", c.samplers[0].name);
    // The first sampler is at binding 2: bindings 0 and 1 are reserved for the per-stage default
    // uniform blocks (VS block at 0, FS block at 1, the `uTint` block here), so the sampler
    // descriptor never shares the shared-constant-bank slot a uniform-block pointer uses (which
    // made a textured + uniform shader render black, or fault the GPU, on the nvidia backend).
    // Coexisting UBO uniform still lives at its block's byte offset 0.
    try std.testing.expectEqual(@as(u32, 2), c.samplers[0].binding);
    // The non-sampler uniform is a UBO member (byte offset 0, 4 floats).
    try std.testing.expectEqual(@as(usize, 1), c.uniforms.len);
    try std.testing.expectEqualStrings("uTint", c.uniforms[0].name);
    try std.testing.expectEqual(@as(u32, 0), c.uniforms[0].offset);
    // Round-trips through Prism's SPIR-V seam.
    const prism_spirv = @import("spirv.zig");
    var ff = try prism_spirv.parseSpirv(gpa, c.spirv);
    ff.deinit();
}

test "glsl: a malformed shader surfaces a compile error (no crash)" {
    const gpa = std.testing.allocator;
    // Missing semicolon / garbage: the frontend must return an error, not panic.
    try std.testing.expectError(error.ParseError, compileForStage(gpa, "void main( { ", .fragment));
    // No main() is a defined failure too.
    try std.testing.expectError(error.MissingMain, compileForStage(gpa, "float x;", .vertex));
}
