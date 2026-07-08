//! OpenGL ES / OpenGL state tracker. Writes to the EGL-current draw surface's
//! HAL backbuffer through the software driver's clear path, so an EGL client can
//! glClearColor + glClear + eglSwapBuffers and get a cleared frame.
//! GL types are defined in Zig (zero C deps) and cross-checked against GLES2/gl2.h.
//! GLenum=u32, GLbitfield=u32, GLclampf=f32, GLint=i32, GLsizei=i32, GLubyte=u8.

const std = @import("std");
const prism = @import("prism");
const state = @import("state.zig");

extern "c" fn getenv(name: [*:0]const u8) ?[*:0]const u8;
/// PRISM_GLES_DEBUG gates verbose GLES diagnostics (e.g. the failing GLSL source).
fn debugEnabled() bool {
    return getenv("PRISM_GLES_DEBUG") != null;
}

// --- GL C-ABI scalar types (GL/gl.h, GLES2/gl2.h) --------------------------

pub const GLenum = c_uint;
pub const GLbitfield = c_uint;
pub const GLint = i32;
pub const GLuint = u32;
pub const GLsizei = i32;
pub const GLsizeiptr = isize;
pub const GLintptr = isize;
pub const GLclampf = f32;
pub const GLfloat = f32;
pub const GLubyte = u8;
pub const GLchar = u8;
pub const GLboolean = u8;

// --- GL constants -----------------------------------------------------------

// glClear bits.
pub const GL_DEPTH_BUFFER_BIT: GLbitfield = 0x00000100;
pub const GL_STENCIL_BUFFER_BIT: GLbitfield = 0x00000400;
pub const GL_COLOR_BUFFER_BIT: GLbitfield = 0x00004000;

// glGetString names.
pub const GL_VENDOR: GLenum = 0x1F00;
pub const GL_RENDERER: GLenum = 0x1F01;
pub const GL_VERSION: GLenum = 0x1F02;
pub const GL_EXTENSIONS: GLenum = 0x1F03;
pub const GL_SHADING_LANGUAGE_VERSION: GLenum = 0x8B8C;

// glGetError codes.
pub const GL_NO_ERROR: GLenum = 0;
pub const GL_INVALID_ENUM: GLenum = 0x0500;
pub const GL_INVALID_VALUE: GLenum = 0x0501;
pub const GL_INVALID_OPERATION: GLenum = 0x0502;
pub const GL_OUT_OF_MEMORY: GLenum = 0x0505;

// glGetBooleanv / type tags + draw primitives.
pub const GL_FALSE: GLboolean = 0;
pub const GL_TRUE: GLboolean = 1;
pub const GL_FLOAT: GLenum = 0x1406;
pub const GL_BYTE: GLenum = 0x1400;
pub const GL_UNSIGNED_BYTE: GLenum = 0x1401;
pub const GL_SHORT: GLenum = 0x1402;
pub const GL_FIXED: GLenum = 0x140C;
pub const GL_HALF_FLOAT: GLenum = 0x140B; // GLES3 spelling (GL_HALF_FLOAT_OES = 0x8D61)
pub const GL_UNSIGNED_SHORT: GLenum = 0x1403;
pub const GL_UNSIGNED_INT: GLenum = 0x1405;

// glDrawArrays / glDrawElements primitive modes (only TRIANGLES is rasterized; the
// rest are accepted enums for a later milestone).
pub const GL_POINTS: GLenum = 0x0000;
pub const GL_LINES: GLenum = 0x0001;
pub const GL_LINE_LOOP: GLenum = 0x0002;
pub const GL_LINE_STRIP: GLenum = 0x0003;
pub const GL_TRIANGLES: GLenum = 0x0004;
pub const GL_TRIANGLE_STRIP: GLenum = 0x0005;
pub const GL_TRIANGLE_FAN: GLenum = 0x0006;

// Buffer targets + usage (glBindBuffer / glBufferData).
pub const GL_ARRAY_BUFFER: GLenum = 0x8892;
pub const GL_ELEMENT_ARRAY_BUFFER: GLenum = 0x8893;
pub const GL_COPY_READ_BUFFER: GLenum = 0x8F36;
pub const GL_COPY_WRITE_BUFFER: GLenum = 0x8F37;
pub const GL_UNIFORM_BUFFER: GLenum = 0x8A11;
// Transform feedback (GLES3): the capture buffer target, the two capture modes, and the
// rasterizer-discard toggle.
pub const GL_TRANSFORM_FEEDBACK_BUFFER: GLenum = 0x8C8E;
pub const GL_INTERLEAVED_ATTRIBS: GLenum = 0x8C8C;
pub const GL_SEPARATE_ATTRIBS: GLenum = 0x8C8D;
pub const GL_RASTERIZER_DISCARD: GLenum = 0x8C89;
pub const GL_TRANSFORM_FEEDBACK: GLenum = 0x8E22;
// Transform-feedback implementation limits (glGetIntegerv).
pub const GL_MAX_TRANSFORM_FEEDBACK_SEPARATE_ATTRIBS: GLenum = 0x8C8A;
pub const GL_MAX_TRANSFORM_FEEDBACK_INTERLEAVED_COMPONENTS: GLenum = 0x8C8B;
pub const GL_MAX_TRANSFORM_FEEDBACK_SEPARATE_COMPONENTS: GLenum = 0x8C80;
pub const GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN: GLenum = 0x8C88;
// glGetActiveUniformBlockiv pnames + glGetUniformBlockIndex sentinel.
pub const GL_UNIFORM_BLOCK_BINDING: GLenum = 0x8A3F;
pub const GL_UNIFORM_BLOCK_DATA_SIZE: GLenum = 0x8A40;
pub const GL_UNIFORM_BLOCK_ACTIVE_UNIFORMS: GLenum = 0x8A42;
pub const GL_UNIFORM_BLOCK_ACTIVE_UNIFORM_INDICES: GLenum = 0x8A43;
pub const GL_ACTIVE_UNIFORM_BLOCKS: GLenum = 0x8A36;
pub const GL_INVALID_INDEX: GLuint = 0xFFFFFFFF;
// glGetActiveUniformsiv pnames (per-block-member introspection).
pub const GL_UNIFORM_TYPE: GLenum = 0x8A37;
pub const GL_UNIFORM_SIZE: GLenum = 0x8A38;
pub const GL_UNIFORM_BLOCK_INDEX: GLenum = 0x8A3A;
pub const GL_UNIFORM_OFFSET: GLenum = 0x8A3B;
pub const GL_UNIFORM_ARRAY_STRIDE: GLenum = 0x8A3C;
pub const GL_UNIFORM_MATRIX_STRIDE: GLenum = 0x8A3D;
pub const GL_STREAM_DRAW: GLenum = 0x88E0;
pub const GL_STATIC_DRAW: GLenum = 0x88E4;
pub const GL_DYNAMIC_DRAW: GLenum = 0x88E8;

// Shader stages (glCreateShader).
pub const GL_FRAGMENT_SHADER: GLenum = 0x8B30;
pub const GL_VERTEX_SHADER: GLenum = 0x8B31;

// glGetShaderPrecisionFormat precision types.
pub const GL_LOW_FLOAT: GLenum = 0x8DF0;
pub const GL_MEDIUM_FLOAT: GLenum = 0x8DF1;
pub const GL_HIGH_FLOAT: GLenum = 0x8DF2;
pub const GL_LOW_INT: GLenum = 0x8DF3;
pub const GL_MEDIUM_INT: GLenum = 0x8DF4;
pub const GL_HIGH_INT: GLenum = 0x8DF5;

// glGetShaderiv / glGetProgramiv pnames + the SPIR-V binary path.
pub const GL_COMPILE_STATUS: GLenum = 0x8B81;
pub const GL_LINK_STATUS: GLenum = 0x8B82;
pub const GL_INFO_LOG_LENGTH: GLenum = 0x8B84;
pub const GL_SHADER_SOURCE_LENGTH: GLenum = 0x8B88;
// glHint targets + modes.
pub const GL_DONT_CARE: GLenum = 0x1100;
pub const GL_FASTEST: GLenum = 0x1101;
pub const GL_NICEST: GLenum = 0x1102;
pub const GL_GENERATE_MIPMAP_HINT: GLenum = 0x8192;
pub const GL_FRAGMENT_SHADER_DERIVATIVE_HINT: GLenum = 0x8B8B;
pub const GL_SHADER_TYPE: GLenum = 0x8B4F;
pub const GL_DELETE_STATUS: GLenum = 0x8B80;
pub const GL_VALIDATE_STATUS: GLenum = 0x8B83;
pub const GL_ATTACHED_SHADERS: GLenum = 0x8B85;
pub const GL_ACTIVE_ATTRIBUTES: GLenum = 0x8B89;
pub const GL_ACTIVE_UNIFORMS: GLenum = 0x8B86;
// glGetActiveUniform/glGetActiveAttrib + the max-name-length queries an app uses to size its
// enumeration name buffers (glmark2's Program builds its uniform/attribute map this way).
pub const GL_ACTIVE_UNIFORM_MAX_LENGTH: GLenum = 0x8B87;
pub const GL_ACTIVE_ATTRIBUTE_MAX_LENGTH: GLenum = 0x8B8A;

// GLSL variable type tags reported by glGetActiveUniform/glGetActiveAttrib.
pub const GL_FLOAT_VEC2: GLenum = 0x8B50;
pub const GL_FLOAT_VEC3: GLenum = 0x8B51;
pub const GL_FLOAT_VEC4: GLenum = 0x8B52;
pub const GL_INT: GLenum = 0x1404;
pub const GL_FLOAT_MAT2: GLenum = 0x8B5A;
pub const GL_FLOAT_MAT3: GLenum = 0x8B5B;
pub const GL_FLOAT_MAT4: GLenum = 0x8B5C;
pub const GL_SAMPLER_2D: GLenum = 0x8B5E;
pub const GL_SAMPLER_CUBE: GLenum = 0x8B60;

// GL_ARB_gl_spirv: glShaderBinary format + glSpecializeShader.
pub const GL_SHADER_BINARY_FORMAT_SPIR_V: GLenum = 0x9551;
pub const GL_SPIR_V_BINARY: GLenum = 0x9552;

// glEnable / glDisable capabilities (the subset es2gears touches).
pub const GL_CULL_FACE: GLenum = 0x0B44;
pub const GL_DEPTH_TEST: GLenum = 0x0B71;
pub const GL_BLEND: GLenum = 0x0BE2;
pub const GL_SCISSOR_TEST: GLenum = 0x0C11;
pub const GL_STENCIL_TEST: GLenum = 0x0B90;
pub const GL_DITHER: GLenum = 0x0BD0;
pub const GL_POLYGON_OFFSET_FILL: GLenum = 0x8037;
// GLES3 primitive restart: when enabled, the type's max index (0xFF/0xFFFF/0xFFFFFFFF) restarts
// a strip/fan in an indexed draw instead of being a real vertex.
pub const GL_PRIMITIVE_RESTART_FIXED_INDEX: GLenum = 0x8D69;
pub const GL_SAMPLE_ALPHA_TO_COVERAGE: GLenum = 0x809E;
pub const GL_SAMPLE_COVERAGE: GLenum = 0x80A0;
pub const GL_SAMPLE_COVERAGE_VALUE: GLenum = 0x80AA;
pub const GL_SAMPLE_COVERAGE_INVERT: GLenum = 0x80AB;

// glCullFace face + glFrontFace winding.
pub const GL_FRONT: GLenum = 0x0404;
pub const GL_BACK: GLenum = 0x0405;
pub const GL_FRONT_AND_BACK: GLenum = 0x0408;
pub const GL_CW: GLenum = 0x0900;
pub const GL_CCW: GLenum = 0x0901;

// glBlendFunc / glBlendFuncSeparate coefficients.
pub const GL_ZERO: GLenum = 0x0000;
pub const GL_ONE: GLenum = 0x0001;
pub const GL_SRC_COLOR: GLenum = 0x0300;
pub const GL_ONE_MINUS_SRC_COLOR: GLenum = 0x0301;
pub const GL_SRC_ALPHA: GLenum = 0x0302;
pub const GL_ONE_MINUS_SRC_ALPHA: GLenum = 0x0303;
pub const GL_DST_ALPHA: GLenum = 0x0304;
pub const GL_ONE_MINUS_DST_ALPHA: GLenum = 0x0305;
pub const GL_DST_COLOR: GLenum = 0x0306;
pub const GL_ONE_MINUS_DST_COLOR: GLenum = 0x0307;
pub const GL_SRC_ALPHA_SATURATE: GLenum = 0x0308;
pub const GL_CONSTANT_COLOR: GLenum = 0x8001;
pub const GL_ONE_MINUS_CONSTANT_COLOR: GLenum = 0x8002;
pub const GL_CONSTANT_ALPHA: GLenum = 0x8003;
pub const GL_ONE_MINUS_CONSTANT_ALPHA: GLenum = 0x8004;

// glBlendEquation / glBlendEquationSeparate modes.
pub const GL_FUNC_ADD: GLenum = 0x8006;
pub const GL_FUNC_SUBTRACT: GLenum = 0x800A;
pub const GL_FUNC_REVERSE_SUBTRACT: GLenum = 0x800B;
pub const GL_MIN: GLenum = 0x8007;
pub const GL_MAX: GLenum = 0x8008;

// glDepthFunc compare ops.
pub const GL_NEVER: GLenum = 0x0200;
pub const GL_LESS: GLenum = 0x0201;
pub const GL_EQUAL: GLenum = 0x0202;
pub const GL_LEQUAL: GLenum = 0x0203;
pub const GL_GREATER: GLenum = 0x0204;
pub const GL_NOTEQUAL: GLenum = 0x0205;
pub const GL_GEQUAL: GLenum = 0x0206;
pub const GL_ALWAYS: GLenum = 0x0207;

// --- Texture targets / parameters / formats (glTexImage2D, glTexParameteri) -

// glActiveTexture units (GL_TEXTURE0 + i).
pub const GL_TEXTURE0: GLenum = 0x84C0;
pub const MAX_TEXTURE_UNITS = 8;

// glBindTexture / glTexImage2D target.
pub const GL_TEXTURE_2D: GLenum = 0x0DE1;

// Cubemap bind target + the 6 face targets for glTexImage2D (GL order +X,-X,+Y,-Y,+Z,-Z).
pub const GL_TEXTURE_CUBE_MAP: GLenum = 0x8513;
pub const GL_TEXTURE_BINDING_CUBE_MAP: GLenum = 0x8514;
pub const GL_TEXTURE_CUBE_MAP_POSITIVE_X: GLenum = 0x8515;
pub const GL_TEXTURE_CUBE_MAP_NEGATIVE_X: GLenum = 0x8516;
pub const GL_TEXTURE_CUBE_MAP_POSITIVE_Y: GLenum = 0x8517;
pub const GL_TEXTURE_CUBE_MAP_NEGATIVE_Y: GLenum = 0x8518;
pub const GL_TEXTURE_CUBE_MAP_POSITIVE_Z: GLenum = 0x8519;
pub const GL_TEXTURE_CUBE_MAP_NEGATIVE_Z: GLenum = 0x851A;

// GLES3 3D texture (sampler3D): bind target + binding query.
pub const GL_TEXTURE_3D: GLenum = 0x806F;
pub const GL_TEXTURE_BINDING_3D: GLenum = 0x806A;
pub const GL_TEXTURE_2D_ARRAY: GLenum = 0x8C1A;
pub const GL_TEXTURE_BINDING_2D_ARRAY: GLenum = 0x8C1D;

/// The face index (0..5, GL order) for a cube-face glTexImage2D target, or null if `t` is
/// not one of the 6 face targets.
fn cubeFaceIndex(t: GLenum) ?u32 {
    return switch (t) {
        GL_TEXTURE_CUBE_MAP_POSITIVE_X => 0,
        GL_TEXTURE_CUBE_MAP_NEGATIVE_X => 1,
        GL_TEXTURE_CUBE_MAP_POSITIVE_Y => 2,
        GL_TEXTURE_CUBE_MAP_NEGATIVE_Y => 3,
        GL_TEXTURE_CUBE_MAP_POSITIVE_Z => 4,
        GL_TEXTURE_CUBE_MAP_NEGATIVE_Z => 5,
        else => null,
    };
}

// glGetIntegerv / glGetFloatv / glGetBooleanv query pnames (implementation limits + live state).
pub const GL_MAX_TEXTURE_SIZE: GLenum = 0x0D33;
pub const GL_MAX_VIEWPORT_DIMS: GLenum = 0x0D3A;
pub const GL_SUBPIXEL_BITS: GLenum = 0x0D50;
pub const GL_MAX_CUBE_MAP_TEXTURE_SIZE: GLenum = 0x851C;
pub const GL_MAX_RENDERBUFFER_SIZE: GLenum = 0x84E8;
pub const GL_MAX_3D_TEXTURE_SIZE: GLenum = 0x8073;
pub const GL_MAX_ARRAY_TEXTURE_LAYERS: GLenum = 0x88FF;
pub const GL_MAX_SAMPLES: GLenum = 0x8D57;
pub const GL_NUM_COMPRESSED_TEXTURE_FORMATS: GLenum = 0x86A2;
pub const GL_COMPRESSED_TEXTURE_FORMATS: GLenum = 0x86A3;
pub const GL_MAX_ELEMENT_INDEX: GLenum = 0x8D6B;
pub const GL_MAX_ELEMENTS_VERTICES: GLenum = 0x80E8;
pub const GL_MAX_ELEMENTS_INDICES: GLenum = 0x80E9;
pub const GL_MAX_TEXTURE_LOD_BIAS: GLenum = 0x84FD;
pub const GL_MAX_VERTEX_ATTRIBS: GLenum = 0x8869;
pub const GL_MAX_VERTEX_UNIFORM_VECTORS: GLenum = 0x8DFB;
pub const GL_MAX_VARYING_VECTORS: GLenum = 0x8DFC;
pub const GL_MAX_FRAGMENT_UNIFORM_VECTORS: GLenum = 0x8DFD;
pub const GL_MAX_TEXTURE_IMAGE_UNITS: GLenum = 0x8872;
pub const GL_MAX_VERTEX_TEXTURE_IMAGE_UNITS: GLenum = 0x8B4C;
pub const GL_MAX_COMBINED_TEXTURE_IMAGE_UNITS: GLenum = 0x8B4D;
pub const GL_IMPLEMENTATION_COLOR_READ_TYPE: GLenum = 0x8B9A;
pub const GL_IMPLEMENTATION_COLOR_READ_FORMAT: GLenum = 0x8B9B;
pub const GL_VIEWPORT: GLenum = 0x0BA2;
pub const GL_SCISSOR_BOX: GLenum = 0x0C10;
pub const GL_COLOR_CLEAR_VALUE: GLenum = 0x0C22;
pub const GL_BLEND_COLOR: GLenum = 0x8005;
pub const GL_DEPTH_RANGE: GLenum = 0x0B70;
pub const GL_LINE_WIDTH: GLenum = 0x0B21;
pub const GL_ALIASED_POINT_SIZE_RANGE: GLenum = 0x846D;
pub const GL_ALIASED_LINE_WIDTH_RANGE: GLenum = 0x846E;
pub const GL_ACTIVE_TEXTURE: GLenum = 0x84E0;
pub const GL_CURRENT_PROGRAM: GLenum = 0x8B8D;
pub const GL_ARRAY_BUFFER_BINDING: GLenum = 0x8894;
pub const GL_ELEMENT_ARRAY_BUFFER_BINDING: GLenum = 0x8895;
pub const GL_FRAMEBUFFER_BINDING: GLenum = 0x8CA6;
pub const GL_TEXTURE_BINDING_2D: GLenum = 0x8069;
pub const GL_VERTEX_ARRAY_BINDING: GLenum = 0x85B5;
pub const GL_NUM_EXTENSIONS: GLenum = 0x821D;

// Fence sync objects (glFenceSync et al, GLES3). Prism's software submit is synchronous, so a
// fence is immediately signaled.
pub const GL_SYNC_GPU_COMMANDS_COMPLETE: GLenum = 0x9117;
pub const GL_SYNC_FENCE: GLenum = 0x9116;
pub const GL_OBJECT_TYPE: GLenum = 0x9112;
pub const GL_SYNC_CONDITION: GLenum = 0x9113;
pub const GL_SYNC_STATUS: GLenum = 0x9114;
pub const GL_SYNC_FLAGS: GLenum = 0x9115;
pub const GL_UNSIGNALED: GLenum = 0x9118;
pub const GL_SIGNALED: GLenum = 0x9119;
pub const GL_ALREADY_SIGNALED: GLenum = 0x911A;
pub const GL_TIMEOUT_EXPIRED: GLenum = 0x911B;
pub const GL_CONDITION_SATISFIED: GLenum = 0x911C;
pub const GL_WAIT_FAILED: GLenum = 0x911D;

// Occlusion query objects (glGenQueries et al, GLES3).
pub const GL_ANY_SAMPLES_PASSED: GLenum = 0x8C2F;
pub const GL_ANY_SAMPLES_PASSED_CONSERVATIVE: GLenum = 0x8D6A;
pub const GL_CURRENT_QUERY: GLenum = 0x8865;
pub const GL_QUERY_RESULT: GLenum = 0x8866;
pub const GL_QUERY_RESULT_AVAILABLE: GLenum = 0x8867;
// Current-framebuffer channel/depth/stencil/sample bit counts.
pub const GL_RED_BITS: GLenum = 0x0D52;
pub const GL_GREEN_BITS: GLenum = 0x0D53;
pub const GL_BLUE_BITS: GLenum = 0x0D54;
pub const GL_ALPHA_BITS: GLenum = 0x0D55;
pub const GL_DEPTH_BITS: GLenum = 0x0D56;
pub const GL_STENCIL_BITS: GLenum = 0x0D57;
pub const GL_SAMPLE_BUFFERS: GLenum = 0x80A8;
pub const GL_SAMPLES: GLenum = 0x80A9;

// glTexParameteri pnames.
pub const GL_TEXTURE_MAG_FILTER: GLenum = 0x2800;
pub const GL_TEXTURE_MIN_FILTER: GLenum = 0x2801;
pub const GL_TEXTURE_WRAP_S: GLenum = 0x2802;
pub const GL_TEXTURE_WRAP_T: GLenum = 0x2803;
// GLES3 sampler-object pnames (accepted by glSamplerParameter*; the LOD/compare ones are stored
// as no-ops - Prism's sampler clamps LOD to the mip chain and does not model shadow compare).
pub const GL_TEXTURE_WRAP_R: GLenum = 0x8072;
pub const GL_TEXTURE_MIN_LOD: GLenum = 0x813A;
pub const GL_TEXTURE_MAX_LOD: GLenum = 0x813B;
pub const GL_TEXTURE_BASE_LEVEL: GLenum = 0x813C;
pub const GL_TEXTURE_MAX_LEVEL: GLenum = 0x813D;
// GL_TEXTURE_SWIZZLE_R/G/B/A (GLES3): per-output-channel component remap (font atlases, etc.).
pub const GL_TEXTURE_SWIZZLE_R: GLenum = 0x8E42;
pub const GL_TEXTURE_SWIZZLE_G: GLenum = 0x8E43;
pub const GL_TEXTURE_SWIZZLE_B: GLenum = 0x8E44;
pub const GL_TEXTURE_SWIZZLE_A: GLenum = 0x8E45;
pub const GL_TEXTURE_LOD_BIAS: GLenum = 0x8501; // added to the computed mip LOD (sharpen/soften)
pub const GL_RED: GLenum = 0x1903;
pub const GL_GREEN: GLenum = 0x1904;
pub const GL_BLUE: GLenum = 0x1905;
pub const GL_TEXTURE_COMPARE_MODE: GLenum = 0x884C;
pub const GL_TEXTURE_COMPARE_FUNC: GLenum = 0x884D;
pub const GL_COMPARE_REF_TO_TEXTURE: GLenum = 0x884E;
// GL_EXT_texture_filter_anisotropic: per-texture max anisotropy + the implementation limit.
pub const GL_TEXTURE_MAX_ANISOTROPY_EXT: GLenum = 0x84FE;
pub const GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT: GLenum = 0x84FF;

// glTexParameteri filter values.
pub const GL_NEAREST: GLenum = 0x2600;
pub const GL_LINEAR: GLenum = 0x2601;
pub const GL_NEAREST_MIPMAP_NEAREST: GLenum = 0x2700;
pub const GL_LINEAR_MIPMAP_NEAREST: GLenum = 0x2701;
pub const GL_NEAREST_MIPMAP_LINEAR: GLenum = 0x2702;
pub const GL_LINEAR_MIPMAP_LINEAR: GLenum = 0x2703;

// glTexParameteri wrap values.
pub const GL_REPEAT: GLenum = 0x2901;
pub const GL_CLAMP_TO_EDGE: GLenum = 0x812F;
pub const GL_MIRRORED_REPEAT: GLenum = 0x8370;

// glGetBufferParameteriv pnames.
pub const GL_BUFFER_SIZE: GLenum = 0x8764;
pub const GL_BUFFER_USAGE: GLenum = 0x8765;
// glGetVertexAttrib*v pnames.
pub const GL_VERTEX_ATTRIB_ARRAY_ENABLED: GLenum = 0x8622;
pub const GL_VERTEX_ATTRIB_ARRAY_SIZE: GLenum = 0x8623;
pub const GL_VERTEX_ATTRIB_ARRAY_STRIDE: GLenum = 0x8624;
pub const GL_VERTEX_ATTRIB_ARRAY_TYPE: GLenum = 0x8625;
pub const GL_CURRENT_VERTEX_ATTRIB: GLenum = 0x8626;
pub const GL_VERTEX_ATTRIB_ARRAY_NORMALIZED: GLenum = 0x886A;
pub const GL_VERTEX_ATTRIB_ARRAY_BUFFER_BINDING: GLenum = 0x889F;
pub const GL_VERTEX_ATTRIB_ARRAY_POINTER: GLenum = 0x8645;
// GLES3: the attribute is an INTEGER array (glVertexAttribIPointer), reported by glGetVertexAttribiv.
pub const GL_VERTEX_ATTRIB_ARRAY_INTEGER: GLenum = 0x88FD;
// glGetRenderbufferParameteriv pnames + GL_RGBA8 (GL_OES_rgb8_rgba8).
pub const GL_RENDERBUFFER_WIDTH: GLenum = 0x8D42;
pub const GL_RENDERBUFFER_HEIGHT: GLenum = 0x8D43;
pub const GL_RENDERBUFFER_INTERNAL_FORMAT: GLenum = 0x8D44;
pub const GL_RENDERBUFFER_RED_SIZE: GLenum = 0x8D50;
pub const GL_RENDERBUFFER_GREEN_SIZE: GLenum = 0x8D51;
pub const GL_RENDERBUFFER_BLUE_SIZE: GLenum = 0x8D52;
pub const GL_RENDERBUFFER_ALPHA_SIZE: GLenum = 0x8D53;
pub const GL_RENDERBUFFER_DEPTH_SIZE: GLenum = 0x8D54;
pub const GL_RENDERBUFFER_STENCIL_SIZE: GLenum = 0x8D55;
pub const GL_RGBA8: GLenum = 0x8058;

// Sized internal formats for glTexStorage2D (immutable texture storage).
pub const GL_RGB8: GLenum = 0x8051;
pub const GL_R8: GLenum = 0x8229;
pub const GL_RG8: GLenum = 0x822B;
pub const GL_SRGB8_ALPHA8: GLenum = 0x8C43;
pub const GL_RGBA16F: GLenum = 0x881A;
pub const GL_RGBA32F: GLenum = 0x8814;
pub const GL_TEXTURE_IMMUTABLE_FORMAT: GLenum = 0x912F; // glGetTexParameteriv query (0/1)

// glTexImage2D internalformat / format.
pub const GL_RGBA: GLenum = 0x1908;
pub const GL_RGB: GLenum = 0x1907;
pub const GL_LUMINANCE: GLenum = 0x1909;
pub const GL_ALPHA: GLenum = 0x1906;
pub const GL_LUMINANCE_ALPHA: GLenum = 0x190A;
/// GL_BGRA_EXT (EXT_texture_format_BGRA8888): a BGRA8 upload, common for compositors
/// sampling client buffers. Stored expanded to Prism's internal RGBA8 (B<->R swapped).
pub const GL_BGRA_EXT: GLenum = 0x80E1;

// Packed 16-bit pixel types (core GLES2): each texel is one native-endian u16 whose
// components are packed MSB-first. Decoded to RGBA8 on upload.
pub const GL_UNSIGNED_SHORT_5_6_5: GLenum = 0x8363; // with GL_RGB
pub const GL_UNSIGNED_SHORT_4_4_4_4: GLenum = 0x8033; // with GL_RGBA
pub const GL_UNSIGNED_SHORT_5_5_5_1: GLenum = 0x8034; // with GL_RGBA

// EXT_sRGB texture formats: the 8-bit RGB(A) texels carry the sRGB EOTF. Stored verbatim
// (rgba8) and tagged sRGB. The sampler decodes sRGB->linear on read (linear filtering).
pub const GL_SRGB_EXT: GLenum = 0x8C40; // RGB, sRGB
pub const GL_SRGB_ALPHA_EXT: GLenum = 0x8C42; // RGBA, sRGB RGB + linear A

// OES_texture_half_float / OES_texture_float pixel types: a float texture stores IEEE
// half / single per channel (HDR values outside 0..1 survive). Stored in native fp16/fp32.
pub const GL_HALF_FLOAT_OES: GLenum = 0x8D61;

// GL_OES_compressed_ETC1_RGB8_texture: 4x4-block ETC1 RGB. Decoded to RGBA8 at upload
// (no GPU/sampler change) - the block decoder runs on the CPU, like the packed formats.
pub const GL_ETC1_RGB8_OES: GLenum = 0x8D64;

// GL_EXT_texture_compression_s3tc (DXT1/3/5): the desktop-game 4x4 block formats. DXT1 is
// 8 bytes/block (RGB, optional 1-bit punchthrough alpha). DXT3/DXT5 are 16 bytes/block (an
// alpha block + a DXT1-style color block). All decoded to RGBA8 at upload.
pub const GL_COMPRESSED_RGB_S3TC_DXT1_EXT: GLenum = 0x83F0;
pub const GL_COMPRESSED_RGBA_S3TC_DXT1_EXT: GLenum = 0x83F1;
pub const GL_COMPRESSED_RGBA_S3TC_DXT3_EXT: GLenum = 0x83F2;
pub const GL_COMPRESSED_RGBA_S3TC_DXT5_EXT: GLenum = 0x83F3;

// GL_EXT_texture_compression_rgtc (BC4/BC5): single- and dual-channel 4x4 block formats used for
// height / roughness / normal (XY) maps. RGTC1 (BC4) is 8 bytes/block: one interpolated-value block
// (identical to the DXT5 alpha block) -> the RED channel. RGTC2 (BC5) is 16 bytes/block: two such
// blocks -> RED then GREEN. Decoded to RGBA8 at upload (unsigned variants; R/RG, other channels
// 0/0/255).
pub const GL_COMPRESSED_RED_RGTC1_EXT: GLenum = 0x8DBB;
pub const GL_COMPRESSED_RG_RGTC2_EXT: GLenum = 0x8DBD;

/// The compressed internal formats Prism can decode at upload (see compressedTexImage2D +
/// blockBytes). glGetIntegerv(GL_COMPRESSED_TEXTURE_FORMATS) enumerates these and
/// GL_NUM_COMPRESSED_TEXTURE_FORMATS reports the count, so an app discovers which compressed
/// assets it may load. Kept in sync with the decode switch in decompressBlock.
const compressed_texture_formats = [_]GLint{
    @intCast(GL_ETC1_RGB8_OES),
    @intCast(GL_COMPRESSED_RGB_S3TC_DXT1_EXT),
    @intCast(GL_COMPRESSED_RGBA_S3TC_DXT1_EXT),
    @intCast(GL_COMPRESSED_RGBA_S3TC_DXT3_EXT),
    @intCast(GL_COMPRESSED_RGBA_S3TC_DXT5_EXT),
    @intCast(GL_COMPRESSED_RED_RGTC1_EXT),
    @intCast(GL_COMPRESSED_RG_RGTC2_EXT),
};

// glPixelStorei pnames.
pub const GL_UNPACK_ALIGNMENT: GLenum = 0x0CF5;
pub const GL_PACK_ALIGNMENT: GLenum = 0x0D05;
pub const GL_UNPACK_ROW_LENGTH: GLenum = 0x0CF2;
pub const GL_UNPACK_SKIP_ROWS: GLenum = 0x0CF3;
pub const GL_UNPACK_SKIP_PIXELS: GLenum = 0x0CF4;
pub const GL_PACK_ROW_LENGTH: GLenum = 0x0D02;
pub const GL_PACK_SKIP_ROWS: GLenum = 0x0D03;
pub const GL_PACK_SKIP_PIXELS: GLenum = 0x0D04;
pub const GL_UNPACK_SKIP_IMAGES: GLenum = 0x806D;
pub const GL_UNPACK_IMAGE_HEIGHT: GLenum = 0x806E;

// --- Framebuffer objects (render-to-texture) + GL_OES_depth_texture --------

// glBindFramebuffer / glBindRenderbuffer targets.
pub const GL_FRAMEBUFFER: GLenum = 0x8D40;
pub const GL_READ_FRAMEBUFFER: GLenum = 0x8CA8;
pub const GL_DRAW_FRAMEBUFFER: GLenum = 0x8CA9;
// glInvalidateFramebuffer default-framebuffer attachment names.
pub const GL_COLOR: GLenum = 0x1800;
pub const GL_DEPTH: GLenum = 0x1801;
pub const GL_STENCIL: GLenum = 0x1802;
pub const GL_DEPTH_STENCIL: GLenum = 0x84F9;
pub const GL_READ_BUFFER: GLenum = 0x0C02;
pub const GL_DRAW_BUFFER0: GLenum = 0x8825;
pub const GL_MAX_DRAW_BUFFERS: GLenum = 0x8824;
pub const GL_MAX_COLOR_ATTACHMENTS: GLenum = 0x8CDF;
pub const GL_NUM_SAMPLE_COUNTS: GLenum = 0x9380; // glGetInternalformativ
pub const GL_RENDERBUFFER: GLenum = 0x8D41;

// Framebuffer attachment points.
pub const GL_COLOR_ATTACHMENT0: GLenum = 0x8CE0;
pub const GL_DEPTH_ATTACHMENT: GLenum = 0x8D00;
pub const GL_STENCIL_ATTACHMENT: GLenum = 0x8D20;

// glCheckFramebufferStatus return values.
pub const GL_FRAMEBUFFER_COMPLETE: GLenum = 0x8CD5;
pub const GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT: GLenum = 0x8CD6;
pub const GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT: GLenum = 0x8CD7;
pub const GL_FRAMEBUFFER_UNSUPPORTED: GLenum = 0x8CDD;

// glGetFramebufferAttachmentParameteriv pnames + object-type values.
pub const GL_NONE: GLenum = 0;
pub const GL_TEXTURE: GLenum = 0x1702;
pub const GL_FRAMEBUFFER_ATTACHMENT_OBJECT_TYPE: GLenum = 0x8CD0;
pub const GL_FRAMEBUFFER_ATTACHMENT_OBJECT_NAME: GLenum = 0x8CD1;
pub const GL_FRAMEBUFFER_ATTACHMENT_TEXTURE_LEVEL: GLenum = 0x8CD2;
pub const GL_FRAMEBUFFER_ATTACHMENT_TEXTURE_CUBE_MAP_FACE: GLenum = 0x8CD3;

// glRenderbufferStorage internalformats (the depth/stencil renderbuffer flavours).
pub const GL_DEPTH_COMPONENT16: GLenum = 0x81A5;
pub const GL_DEPTH_COMPONENT24: GLenum = 0x81A6;
pub const GL_DEPTH_COMPONENT32: GLenum = 0x81A7;
pub const GL_DEPTH_COMPONENT32_OES: GLenum = 0x81A7;
pub const GL_STENCIL_INDEX8: GLenum = 0x8D48;
// OES_packed_depth_stencil: a single renderbuffer/attachment carrying both depth and stencil.
pub const GL_DEPTH_STENCIL_ATTACHMENT: GLenum = 0x821A;
pub const GL_DEPTH24_STENCIL8: GLenum = 0x88F0;
pub const GL_DEPTH24_STENCIL8_OES: GLenum = 0x88F0;
pub const GL_RGBA4: GLenum = 0x8056;
pub const GL_RGB565: GLenum = 0x8D62;
pub const GL_RGB5_A1: GLenum = 0x8057;

// GL_OES_depth_texture: a GL_DEPTH_COMPONENT texture (sampled as a shadow/distance map).
pub const GL_DEPTH_COMPONENT: GLenum = 0x1902;

// GL_OES_mapbuffer (glMapBufferOES access).
pub const GL_WRITE_ONLY_OES: GLenum = 0x88B9;
pub const GL_BUFFER_ACCESS_OES: GLenum = 0x88BB;
pub const GL_BUFFER_MAPPED_OES: GLenum = 0x88BC;
pub const GL_BUFFER_MAP_POINTER_OES: GLenum = 0x88BD;
// GLES3 glMapBufferRange access bitfield + the buffer map-pointer query.
pub const GL_MAP_READ_BIT: GLbitfield = 0x0001;
pub const GL_MAP_WRITE_BIT: GLbitfield = 0x0002;
pub const GL_MAP_INVALIDATE_RANGE_BIT: GLbitfield = 0x0004;
pub const GL_MAP_INVALIDATE_BUFFER_BIT: GLbitfield = 0x0008;
pub const GL_MAP_FLUSH_EXPLICIT_BIT: GLbitfield = 0x0010;
pub const GL_MAP_UNSYNCHRONIZED_BIT: GLbitfield = 0x0020;
pub const GL_BUFFER_MAP_POINTER: GLenum = 0x88BD;

// --- Driver strings (honest, Prism-branded) --------------------------------

pub const vendor_string = "Prism";
pub const renderer_string = "Prism";
pub const version_string = "OpenGL ES 2.0 Prism";
pub const glsl_version_string = "OpenGL ES GLSL ES 1.00";
/// Extensions Prism's GLES2 frontend implements, one per entry. glGetString(GL_EXTENSIONS)
/// joins them with spaces. glGetStringi(GL_EXTENSIONS, i) returns entry i and
/// glGetIntegerv(GL_NUM_EXTENSIONS) returns the count.
pub const extension_names = [_][:0]const u8{
    "GL_OES_depth_texture",            "GL_OES_depth24",
    "GL_OES_depth32",                  "GL_OES_packed_depth_stencil",
    "GL_OES_rgb8_rgba8",               "GL_OES_mapbuffer",
    "GL_OES_required_internalformat",  "GL_OES_compressed_ETC1_RGB8_texture",
    "GL_EXT_texture_compression_s3tc", "GL_EXT_texture_format_BGRA8888",
    "GL_OES_texture_half_float",       "GL_OES_texture_float",
    "GL_EXT_sRGB",                     "GL_EXT_texture_filter_anisotropic",
    "GL_OES_vertex_array_object",      "GL_ANGLE_instanced_arrays",
    "GL_EXT_instanced_arrays",         "GL_EXT_texture_compression_rgtc",
};
pub const extensions_string: [:0]const u8 = blk: {
    var len: usize = 0;
    for (extension_names, 0..) |e, i| len += e.len + @as(usize, if (i == 0) 0 else 1);
    var arr: [len:0]u8 = undefined;
    var p: usize = 0;
    for (extension_names, 0..) |e, i| {
        if (i != 0) {
            arr[p] = ' ';
            p += 1;
        }
        @memcpy(arr[p..][0..e.len], e);
        p += e.len;
    }
    arr[len] = 0;
    const final = arr;
    break :blk &final;
};

pub fn getString(name: GLenum) ?[*:0]const GLubyte {
    const s: ?[*:0]const u8 = switch (name) {
        GL_VENDOR => vendor_string,
        GL_RENDERER => renderer_string,
        GL_VERSION => version_string,
        GL_SHADING_LANGUAGE_VERSION => glsl_version_string,
        GL_EXTENSIONS => extensions_string,
        else => null,
    };
    return @ptrCast(s);
}

/// glGetStringi(GL_EXTENSIONS, index): the index-th supported extension (the GLES3 / core-profile
/// enumeration, paired with glGetIntegerv(GL_NUM_EXTENSIONS)). Out-of-range index -> GL_INVALID_VALUE.
pub fn getStringi(name: GLenum, index: GLuint) ?[*:0]const GLubyte {
    switch (name) {
        GL_EXTENSIONS => {
            if (index >= extension_names.len) {
                setError(GL_INVALID_VALUE);
                return null;
            }
            return @ptrCast(extension_names[index].ptr);
        },
        else => {
            setError(GL_INVALID_ENUM);
            return null;
        },
    }
}

// --- Per-thread GLES state --------------------------------------------------

/// Per-thread GLES error flag. glGetError reports and resets it. Single slot; that is the common minimal behavior.
threadlocal var gl_error: GLenum = GL_NO_ERROR;

/// The per-thread clear color (glClearColor), defaults to transparent black per GL.
pub const ClearColor = struct {
    r: GLclampf = 0,
    g: GLclampf = 0,
    b: GLclampf = 0,
    a: GLclampf = 0,
};
threadlocal var clear_color: ClearColor = .{};

/// Per-thread viewport. The minimal clear path clears the whole surface, so the
/// viewport is tracked + queryable but does not scissor the clear (no glScissor in
/// this milestone). x,y,width,height in pixels.
threadlocal var viewport: [4]GLint = .{ 0, 0, 0, 0 };

/// Per-thread fixed-function state es2gears toggles: depth test + cull. Defaults match
/// GL (depth test OFF, depth func LESS, depth writes ON, cull OFF, front face CCW, cull
/// BACK). These feed the HAL pipeline's DepthState + CullState at link/draw time.
const FixedState = struct {
    depth_test: bool = false,
    depth_func: prism.hal.CompareOp = .less,
    depth_write: bool = true,
    // Depth bias (glEnable(GL_POLYGON_OFFSET_FILL) + glPolygonOffset(factor, units)). GL default:
    // off, factor 0, units 0. Applied only when depth testing is on (bias shifts tested depth).
    polygon_offset: bool = false,
    polygon_offset_factor: f32 = 0, // GL `factor` -> HAL bias_slope
    polygon_offset_units: f32 = 0, // GL `units` -> HAL bias_constant
    depth_clear: f32 = 1.0,
    cull_face: bool = false,
    cull_mode: prism.hal.CullMode = .back,
    front_face: prism.hal.FrontFace = .counter_clockwise,
    // Alpha blend (glEnable(GL_BLEND) + glBlendFunc*/glBlendEquation*/glBlendColor). The
    // GL defaults: src=ONE dst=ZERO, equation FUNC_ADD, constant (0,0,0,0) - a no-op even
    // when GL_BLEND is enabled, so a program that enables blending without setting a func
    // still renders as passthrough (matching GL).
    blend: bool = false,
    blend_src_rgb: prism.hal.BlendFactor = .one,
    blend_dst_rgb: prism.hal.BlendFactor = .zero,
    blend_src_alpha: prism.hal.BlendFactor = .one,
    blend_dst_alpha: prism.hal.BlendFactor = .zero,
    blend_op_rgb: prism.hal.BlendOp = .add,
    blend_op_alpha: prism.hal.BlendOp = .add,
    blend_color: [4]f32 = .{ 0, 0, 0, 0 },
    // Per-channel color write mask (glColorMask). GL default: all enabled. Independent of blend
    // (applies whether or not GL_BLEND is on). A stencil-only mask pass sets all-false.
    color_mask: [4]bool = .{ true, true, true, true },
    // Stencil (glEnable(GL_STENCIL_TEST) + glStencilFunc/glStencilOp/glStencilMask/
    // glClearStencil). GL defaults: func ALWAYS, ref 0, masks 0xff, all ops KEEP, clear 0.
    // A no-op even when enabled, so a program that turns stencil on without configuring it
    // still renders unchanged (matching GL).
    stencil_test: bool = false,
    stencil_func: prism.hal.CompareOp = .always,
    stencil_ref: u8 = 0,
    stencil_value_mask: u8 = 0xff,
    stencil_write_mask: u8 = 0xff,
    stencil_sfail: prism.hal.StencilOp = .keep,
    stencil_dpfail: prism.hal.StencilOp = .keep,
    stencil_dppass: prism.hal.StencilOp = .keep,
    // Back-face stencil (glStencilFuncSeparate/glStencilOpSeparate/glStencilMaskSeparate with
    // GL_BACK). GL keeps front and back state independent. The single-face glStencilFunc/Op/Mask
    // entry points write BOTH faces (GL_FRONT_AND_BACK). These mirror the front-face defaults so
    // a program that never touches the separate variants stays two-sided-identical (back == front),
    // which wantStencilBack() detects and reports as "no separate back state".
    stencil_back_func: prism.hal.CompareOp = .always,
    stencil_back_ref: u8 = 0,
    stencil_back_value_mask: u8 = 0xff,
    stencil_back_write_mask: u8 = 0xff,
    stencil_back_sfail: prism.hal.StencilOp = .keep,
    stencil_back_dpfail: prism.hal.StencilOp = .keep,
    stencil_back_dppass: prism.hal.StencilOp = .keep,
    stencil_clear: u8 = 0,
    // Scissor (glEnable(GL_SCISSOR_TEST) + glScissor). GL default: test OFF, box = the full
    // window. `scissor_set` tracks whether glScissor has been called. Until then the box is
    // treated as the full render target (GL initializes it to the window dimensions). The
    // box is in GL's BOTTOM-LEFT window coords. The draw path flips y to the HAL's top-left.
    scissor_test: bool = false,
    scissor_set: bool = false,
    scissor_box: [4]GLint = .{ 0, 0, 0, 0 },
    sample_alpha_to_coverage: bool = false, // GL_SAMPLE_ALPHA_TO_COVERAGE
    sample_coverage: bool = false, // GL_SAMPLE_COVERAGE (enable)
    sample_coverage_value: GLfloat = 1.0, // glSampleCoverage value, clamped [0,1]
    sample_coverage_invert: bool = false, // glSampleCoverage invert
    // GL_RASTERIZER_DISCARD (GLES3): when on, primitives are discarded before rasterization (the
    // fragment stage never runs). Used with transform feedback for pure GPU-computed capture.
    rasterizer_discard: bool = false,
};
threadlocal var fixed: FixedState = .{};

/// Reset this thread's GLES fixed-function + binding state to GL defaults. Called when a
/// different context is made current (each GL context owns its own default state, so the
/// previously-current context's depth-test/cull/clear-color/bindings must not leak into
/// the newly-bound one). Without this, a context that left GL_DEPTH_TEST enabled would
/// make the next context bind an uncleared depth attachment (the depth clear is deferred
/// to glClear(GL_DEPTH_BUFFER_BIT)), discarding fragments against garbage depth.
pub fn resetThreadState() void {
    fixed = .{};
    clear_color = .{};
    viewport = .{ 0, 0, 0, 0 };
    depth_range = .{ 0, 1 }; // glDepthRangef default; per-context, must not leak into the next context
    gl_error = GL_NO_ERROR; // each GL context owns its error state. a prior context's unread error must not carry over
    line_width = 1.0;
    primitive_restart_fixed = false;
    pending_depth_clear = false;
    pending_stencil_clear = false;
    bound_array_buffer = 0;
    bound_element_buffer = 0;
    bound_copy_read_buffer = 0;
    bound_copy_write_buffer = 0;
    bound_uniform_buffer = 0;
    uniform_buffer_bindings = [_]GLuint{0} ** MAX_UNIFORM_BUFFER_BINDINGS;
    uniform_buffer_offsets = [_]GLintptr{0} ** MAX_UNIFORM_BUFFER_BINDINGS;
    transform_feedback_bindings = [_]GLuint{0} ** MAX_TRANSFORM_FEEDBACK_BUFFERS;
    bound_transform_feedback_buffer = 0;
    tf_active = false;
    tf_primitive_mode = 0;
    tf_write_offsets = [_]usize{0} ** MAX_TRANSFORM_FEEDBACK_BUFFERS;
    tf_paused = false;
    current_program = 0;
    attribs = [_]AttribArray{.{}} ** MAX_ATTRIBS;
    bound_framebuffer = 0;
    bound_renderbuffer = 0;
    bound_read_framebuffer = 0;
    // glReadBuffer / glDrawBuffers selection is per-context GL state: reset it so a context
    // that selected an MRT attachment (GL_COLOR_ATTACHMENT_i) does not leak that read/draw
    // buffer into the next context (which would read/write the wrong attachment).
    read_buffer = GL_BACK;
    draw_buffer0 = GL_BACK;
    active_occlusion_query = 0;
    bound_vao = 0;
    default_vao = .{ .attribs = [_]AttribArray{.{}} ** MAX_ATTRIBS, .element_buffer = 0 };
    active_texture_unit = 0;
    bound_texture_2d = [_]GLuint{0} ** MAX_TEXTURE_UNITS;
    bound_texture_cube = [_]GLuint{0} ** MAX_TEXTURE_UNITS;
    bound_texture_3d = [_]GLuint{0} ** MAX_TEXTURE_UNITS;
    bound_texture_2darray = [_]GLuint{0} ** MAX_TEXTURE_UNITS;
    bound_sampler = [_]GLuint{0} ** MAX_TEXTURE_UNITS;
    unpack_alignment = 4;
    unpack_row_length = 0;
    unpack_skip_pixels = 0;
    unpack_skip_rows = 0;
    unpack_image_height = 0;
    unpack_skip_images = 0;
    pack_alignment = 4;
    pack_row_length = 0;
    pack_skip_pixels = 0;
    pack_skip_rows = 0;
}

fn setError(code: GLenum) void {
    if (gl_error == GL_NO_ERROR) gl_error = code;
}

/// glGetError: report and reset the current error flag.
pub fn getError() GLenum {
    const e = gl_error;
    gl_error = GL_NO_ERROR;
    return e;
}

fn clampUnit(v: GLclampf) f32 {
    return std.math.clamp(v, 0.0, 1.0);
}

pub fn clearColor(r: GLclampf, g: GLclampf, b: GLclampf, a: GLclampf) void {
    clear_color = .{ .r = clampUnit(r), .g = clampUnit(g), .b = clampUnit(b), .a = clampUnit(a) };
}

/// glEnable / glDisable: toggle a fixed-function capability. DEPTH_TEST, CULL_FACE,
/// GL_BLEND, GL_SCISSOR_TEST, and GL_STENCIL_TEST are honored. Other valid caps are
/// accepted as no-ops to keep stock apps from erroring. An unknown enum raises GL_INVALID_ENUM.
fn setCap(cap: GLenum, on: bool) void {
    switch (cap) {
        GL_DEPTH_TEST => fixed.depth_test = on,
        GL_CULL_FACE => fixed.cull_face = on,
        GL_BLEND => fixed.blend = on,
        GL_STENCIL_TEST => fixed.stencil_test = on,
        GL_SCISSOR_TEST => fixed.scissor_test = on,
        GL_POLYGON_OFFSET_FILL => fixed.polygon_offset = on,
        GL_PRIMITIVE_RESTART_FIXED_INDEX => primitive_restart_fixed = on,
        GL_SAMPLE_ALPHA_TO_COVERAGE => fixed.sample_alpha_to_coverage = on,
        GL_SAMPLE_COVERAGE => fixed.sample_coverage = on,
        GL_RASTERIZER_DISCARD => fixed.rasterizer_discard = on,
        GL_DITHER,
        => {}, // accepted, not yet modeled (no-op with correct semantics for es2gears)
        else => setError(GL_INVALID_ENUM),
    }
}
pub fn enable(cap: GLenum) void {
    setCap(cap, true);
}
pub fn disable(cap: GLenum) void {
    setCap(cap, false);
}

/// Returns the current framebuffer's channel / depth / stencil / sample bit counts for
/// GL_*_BITS / GL_SAMPLES / GL_SAMPLE_BUFFERS queries. The default framebuffer reports the
/// draw surface's EGL config. A bound FBO reports its attachments: color is always rgba8
/// (8/8/8/8), depth 24-bit, stencil 8-bit (GL_OES_depth24 / packed depth-stencil). FBOs
/// are single-sample. Returns 0 when nothing is resolvable.
fn fbBits(pname: GLenum) GLint {
    if (bound_framebuffer != 0) {
        obj_lock.lock();
        defer obj_lock.unlock();
        const f = findFramebuffer(bound_framebuffer) orelse return 0;
        const has_color = f.color_tex != 0 or f.color_rb != 0;
        const has_depth = f.depth_tex != 0 or f.depth_rb != 0;
        const has_stencil = f.stencil_rb != 0;
        return switch (pname) {
            GL_RED_BITS, GL_GREEN_BITS, GL_BLUE_BITS, GL_ALPHA_BITS => if (has_color) 8 else 0,
            GL_DEPTH_BITS => if (has_depth) 24 else 0,
            GL_STENCIL_BITS => if (has_stencil) 8 else 0,
            else => 0, // GL_SAMPLES / GL_SAMPLE_BUFFERS: FBOs are single-sample
        };
    }
    const surf = state.currentDrawSurface() orelse return 0;
    const cfg = state.configs[surf.config];
    return switch (pname) {
        GL_RED_BITS => cfg.red,
        GL_GREEN_BITS => cfg.green,
        GL_BLUE_BITS => cfg.blue,
        GL_ALPHA_BITS => cfg.alpha,
        GL_DEPTH_BITS => cfg.depth,
        GL_STENCIL_BITS => cfg.stencil,
        GL_SAMPLES => cfg.samples,
        GL_SAMPLE_BUFFERS => if (cfg.samples > 0) 1 else 0,
        else => 0,
    };
}

/// Fill `out` with the value(s) of a glGet* `pname` as f32, returning the component count
/// (0 = unknown pname, raises GL_INVALID_ENUM). One core for glGetInteger/Float/Booleanv.
/// Each getter converts these f32s per the GL rule (round to int, != 0 to boolean). Limits
/// reflect what Prism actually enforces (MAX_ATTRIBS / MAX_TEXTURE_UNITS) or what the
/// software and GPU paths handle. State comes from the live per-thread GLES state.
fn queryState(pname: GLenum, out: *[4]GLfloat) usize {
    const b = struct {
        fn f(v: bool) GLfloat {
            return if (v) 1 else 0;
        }
    }.f;
    switch (pname) {
        // --- implementation limits ---
        GL_MAX_TEXTURE_SIZE, GL_MAX_CUBE_MAP_TEXTURE_SIZE, GL_MAX_RENDERBUFFER_SIZE => {
            out[0] = 8192;
            return 1;
        },
        // GLES3 caps Prism actually supports (sampler3D / sampler2DArray - see prism-3d-textures /
        // prism-sampler2darray). Reported so an engine's caps probe does not get GL_INVALID_ENUM.
        GL_MAX_3D_TEXTURE_SIZE => {
            out[0] = 2048;
            return 1;
        },
        GL_MAX_ARRAY_TEXTURE_LAYERS => {
            out[0] = 2048;
            return 1;
        },
        // The software rasterizer resolves up to 4x MSAA (raster.samplePositions: 1/2/4).
        GL_MAX_SAMPLES => {
            out[0] = 4;
            return 1;
        },
        // The count of decodable compressed formats. The list itself (an N-element array) is
        // handled directly in getIntegerv since it exceeds this 4-float scratch buffer.
        GL_NUM_COMPRESSED_TEXTURE_FORMATS => {
            out[0] = @floatFromInt(compressed_texture_formats.len);
            return 1;
        },
        // GL_UNSIGNED_INT indices are accepted (drawElements), so advertise the GLES3-minimum max
        // index 2^24-1 (also exactly representable as f32 so it survives this scratch buffer).
        GL_MAX_ELEMENT_INDEX => {
            out[0] = 0x00FFFFFF;
            return 1;
        },
        // Advisory glDrawRangeElements batch-size hints (no hard limit in the software path).
        GL_MAX_ELEMENTS_VERTICES, GL_MAX_ELEMENTS_INDICES => {
            out[0] = 1048576;
            return 1;
        },
        // The sampler honors GL_TEXTURE_LOD_BIAS (see prism-texture-mip-levels). Advertise the cap.
        GL_MAX_TEXTURE_LOD_BIAS => {
            out[0] = 16;
            return 1;
        },
        GL_MAX_VIEWPORT_DIMS => {
            out[0] = 8192;
            out[1] = 8192;
            return 2;
        },
        GL_MAX_VERTEX_ATTRIBS => {
            out[0] = @floatFromInt(MAX_ATTRIBS);
            return 1;
        },
        GL_MAX_TEXTURE_IMAGE_UNITS, GL_MAX_COMBINED_TEXTURE_IMAGE_UNITS => {
            out[0] = @floatFromInt(MAX_TEXTURE_UNITS);
            return 1;
        },
        // Vertex texture fetch: a vertex shader may sample textures (terrain heightmap
        // displacement). A VS has no derivatives, so a `texture()` in the vertex stage lowers
        // to an explicit LOD-0 sample (vulcan-glsl). Advertise the same unit budget as the
        // fragment stage. A shared combined-image-sampler binding feeds either stage.
        GL_MAX_VERTEX_TEXTURE_IMAGE_UNITS => {
            out[0] = @floatFromInt(MAX_TEXTURE_UNITS);
            return 1;
        },
        GL_MAX_VERTEX_UNIFORM_VECTORS, GL_MAX_FRAGMENT_UNIFORM_VECTORS => {
            out[0] = 1024;
            return 1;
        },
        GL_MAX_VARYING_VECTORS => {
            out[0] = 16;
            return 1;
        },
        // --- transform-feedback limits ---
        // The number of separate-mode capture buffers (one varying each) = the number of TF binding
        // points we track; and the per-mode component budgets.
        GL_MAX_TRANSFORM_FEEDBACK_SEPARATE_ATTRIBS => {
            out[0] = @floatFromInt(MAX_TRANSFORM_FEEDBACK_BUFFERS);
            return 1;
        },
        GL_MAX_TRANSFORM_FEEDBACK_INTERLEAVED_COMPONENTS => {
            out[0] = 64;
            return 1;
        },
        GL_MAX_TRANSFORM_FEEDBACK_SEPARATE_COMPONENTS => {
            out[0] = 4;
            return 1;
        },
        GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT => {
            out[0] = 16;
            return 1;
        },
        GL_SUBPIXEL_BITS => {
            out[0] = 4;
            return 1;
        },
        // --- current framebuffer channel / depth / stencil / sample bit counts ---
        GL_RED_BITS, GL_GREEN_BITS, GL_BLUE_BITS, GL_ALPHA_BITS, GL_DEPTH_BITS, GL_STENCIL_BITS, GL_SAMPLES, GL_SAMPLE_BUFFERS => {
            out[0] = @floatFromInt(fbBits(pname));
            return 1;
        },
        // --- glReadPixels' implementation-defined readback pair (the always-supported combo) ---
        GL_IMPLEMENTATION_COLOR_READ_FORMAT => {
            out[0] = @floatFromInt(GL_RGBA);
            return 1;
        },
        GL_IMPLEMENTATION_COLOR_READ_TYPE => {
            out[0] = @floatFromInt(GL_UNSIGNED_BYTE);
            return 1;
        },
        // --- ranges ---
        GL_ALIASED_POINT_SIZE_RANGE => {
            out[0] = 1;
            out[1] = 1024;
            return 2;
        },
        GL_ALIASED_LINE_WIDTH_RANGE => {
            out[0] = 1;
            out[1] = 1;
            return 2;
        },
        GL_DEPTH_RANGE => {
            out[0] = depth_range[0];
            out[1] = depth_range[1];
            return 2;
        },
        // --- live state ---
        GL_VIEWPORT => {
            for (0..4) |i| out[i] = @floatFromInt(viewport[i]);
            return 4;
        },
        GL_SCISSOR_BOX => {
            for (0..4) |i| out[i] = @floatFromInt(fixed.scissor_box[i]);
            return 4;
        },
        GL_COLOR_CLEAR_VALUE => {
            out[0] = clear_color.r;
            out[1] = clear_color.g;
            out[2] = clear_color.b;
            out[3] = clear_color.a;
            return 4;
        },
        GL_BLEND_COLOR => {
            for (0..4) |i| out[i] = fixed.blend_color[i];
            return 4;
        },
        GL_LINE_WIDTH => {
            out[0] = line_width;
            return 1;
        },
        GL_ACTIVE_TEXTURE => {
            out[0] = @floatFromInt(GL_TEXTURE0 + active_texture_unit);
            return 1;
        },
        GL_CURRENT_PROGRAM => {
            out[0] = @floatFromInt(current_program);
            return 1;
        },
        GL_ARRAY_BUFFER_BINDING => {
            out[0] = @floatFromInt(bound_array_buffer);
            return 1;
        },
        GL_ELEMENT_ARRAY_BUFFER_BINDING => {
            out[0] = @floatFromInt(bound_element_buffer);
            return 1;
        },
        GL_FRAMEBUFFER_BINDING => {
            out[0] = @floatFromInt(bound_framebuffer);
            return 1;
        },
        GL_TEXTURE_BINDING_2D => {
            out[0] = @floatFromInt(bound_texture_2d[active_texture_unit]);
            return 1;
        },
        GL_VERTEX_ARRAY_BINDING => {
            out[0] = @floatFromInt(bound_vao);
            return 1;
        },
        GL_NUM_EXTENSIONS => {
            out[0] = @floatFromInt(extension_names.len);
            return 1;
        },
        GL_READ_BUFFER => {
            out[0] = @floatFromInt(read_buffer);
            return 1;
        },
        GL_DRAW_BUFFER0 => {
            out[0] = @floatFromInt(draw_buffer0);
            return 1;
        },
        GL_MAX_DRAW_BUFFERS, GL_MAX_COLOR_ATTACHMENTS => {
            out[0] = @floatFromInt(prism.hal.MAX_COLOR_TARGETS);
            return 1;
        },
        // --- enable flags (queried as 0/1) ---
        GL_DEPTH_TEST => {
            out[0] = b(fixed.depth_test);
            return 1;
        },
        GL_CULL_FACE => {
            out[0] = b(fixed.cull_face);
            return 1;
        },
        GL_BLEND => {
            out[0] = b(fixed.blend);
            return 1;
        },
        GL_SCISSOR_TEST => {
            out[0] = b(fixed.scissor_test);
            return 1;
        },
        GL_STENCIL_TEST => {
            out[0] = b(fixed.stencil_test);
            return 1;
        },
        GL_POLYGON_OFFSET_FILL => {
            out[0] = b(fixed.polygon_offset);
            return 1;
        },
        GL_SAMPLE_ALPHA_TO_COVERAGE => {
            out[0] = b(fixed.sample_alpha_to_coverage);
            return 1;
        },
        GL_SAMPLE_COVERAGE => {
            out[0] = b(fixed.sample_coverage);
            return 1;
        },
        GL_SAMPLE_COVERAGE_VALUE => {
            out[0] = fixed.sample_coverage_value;
            return 1;
        },
        GL_SAMPLE_COVERAGE_INVERT => {
            out[0] = b(fixed.sample_coverage_invert);
            return 1;
        },
        // Dither is enabled by default in GL. Prism does not dither, but the query is honest
        // about the default state a stock app expects.
        GL_DITHER => {
            out[0] = 1;
            return 1;
        },
        else => return 0,
    }
}

/// glGetIntegerv: write the queried state as GLint(s) (float values round to nearest per GL).
pub fn getIntegerv(pname: GLenum, params: ?[*]GLint) void {
    const p = params orelse return;
    // GL_COMPRESSED_TEXTURE_FORMATS returns a variable-length array of format enums, more than the
    // 4-value queryState scratch buffer holds, so write it straight into the caller's buffer.
    if (pname == GL_COMPRESSED_TEXTURE_FORMATS) {
        for (compressed_texture_formats, 0..) |fmt, i| p[i] = fmt;
        return;
    }
    var buf: [4]GLfloat = undefined;
    const n = queryState(pname, &buf);
    if (n == 0) {
        setError(GL_INVALID_ENUM);
        return;
    }
    for (0..n) |i| p[i] = @intFromFloat(@round(buf[i]));
}

/// glGetFloatv: write the queried state as GLfloat(s) verbatim.
pub fn getFloatv(pname: GLenum, params: ?[*]GLfloat) void {
    const p = params orelse return;
    var buf: [4]GLfloat = undefined;
    const n = queryState(pname, &buf);
    if (n == 0) {
        setError(GL_INVALID_ENUM);
        return;
    }
    for (0..n) |i| p[i] = buf[i];
}

/// glGetBooleanv: write the queried state as GLboolean(s) (each component != 0 -> GL_TRUE).
pub fn getBooleanv(pname: GLenum, params: ?[*]GLboolean) void {
    const p = params orelse return;
    var buf: [4]GLfloat = undefined;
    const n = queryState(pname, &buf);
    if (n == 0) {
        setError(GL_INVALID_ENUM);
        return;
    }
    for (0..n) |i| p[i] = if (buf[i] != 0) GL_TRUE else GL_FALSE;
}

/// glIsEnabled: report whether a fixed-function capability is on. Only capability enums are
/// valid (an unknown enum is GL_INVALID_ENUM + GL_FALSE). No-op-but-accepted caps (DITHER
/// default-on, SAMPLE_* off) report their modeled default.
pub fn isEnabled(cap: GLenum) GLboolean {
    return switch (cap) {
        GL_DEPTH_TEST => if (fixed.depth_test) GL_TRUE else GL_FALSE,
        GL_CULL_FACE => if (fixed.cull_face) GL_TRUE else GL_FALSE,
        GL_BLEND => if (fixed.blend) GL_TRUE else GL_FALSE,
        GL_SCISSOR_TEST => if (fixed.scissor_test) GL_TRUE else GL_FALSE,
        GL_STENCIL_TEST => if (fixed.stencil_test) GL_TRUE else GL_FALSE,
        GL_POLYGON_OFFSET_FILL => if (fixed.polygon_offset) GL_TRUE else GL_FALSE,
        GL_PRIMITIVE_RESTART_FIXED_INDEX => if (primitive_restart_fixed) GL_TRUE else GL_FALSE,
        GL_DITHER => GL_TRUE, // enabled by default in GL (a no-op in Prism)
        GL_SAMPLE_ALPHA_TO_COVERAGE => if (fixed.sample_alpha_to_coverage) GL_TRUE else GL_FALSE,
        GL_SAMPLE_COVERAGE => if (fixed.sample_coverage) GL_TRUE else GL_FALSE,
        else => {
            setError(GL_INVALID_ENUM);
            return GL_FALSE;
        },
    };
}

/// glGetShaderPrecisionFormat(shadertype, precisiontype, range[2], precision): report the
/// numeric range/precision of a shader numeric format. Prism compiles every GLES precision
/// qualifier to IEEE-754 single float / 32-bit int (no reduced-precision path). Reports the
/// full-precision values a desktop GL exposes: float -> range {127,127}, precision 23.
/// int -> range {31,31}, precision 0. `range` gets log2 of the min/max representable magnitude.
/// `precision` gets log2 of the precision (0 for exact integers).
pub fn getShaderPrecisionFormat(shadertype: GLenum, precisiontype: GLenum, range: ?[*]GLint, precision: ?*GLint) void {
    if (shadertype != GL_VERTEX_SHADER and shadertype != GL_FRAGMENT_SHADER) {
        setError(GL_INVALID_ENUM);
        return;
    }
    const is_float = switch (precisiontype) {
        GL_LOW_FLOAT, GL_MEDIUM_FLOAT, GL_HIGH_FLOAT => true,
        GL_LOW_INT, GL_MEDIUM_INT, GL_HIGH_INT => false,
        else => {
            setError(GL_INVALID_ENUM);
            return;
        },
    };
    if (range) |r| {
        r[0] = if (is_float) 127 else 31;
        r[1] = if (is_float) 127 else 31;
    }
    if (precision) |pr| pr.* = if (is_float) 23 else 0;
}

// --- Per-object getters (glGetTexParameter / VertexAttrib / BufferParameter / Renderbuffer) --

/// Reconstruct the GL_TEXTURE_MIN_FILTER enum from the split base-min + mip-filter storage.
/// Prism keeps them separate. glTexParameteri decomposed the combined GL enum.
fn minFilterEnum(t: *const Texture) GLenum {
    const base_lin = t.min_filter == .linear;
    return switch (t.mip_filter) {
        .none => if (base_lin) GL_LINEAR else GL_NEAREST,
        .nearest => if (base_lin) GL_LINEAR_MIPMAP_NEAREST else GL_NEAREST_MIPMAP_NEAREST,
        .linear => if (base_lin) GL_LINEAR_MIPMAP_LINEAR else GL_NEAREST_MIPMAP_LINEAR,
    };
}
fn wrapEnum(w: prism.hal.AddressMode) GLenum {
    return switch (w) {
        .repeat => GL_REPEAT,
        .clamp_to_edge => GL_CLAMP_TO_EDGE,
        .mirrored_repeat => GL_MIRRORED_REPEAT,
    };
}
/// The active texture's sampler-state value for a glGetTexParameter* pname, as f32 (null =
/// unknown pname).
fn texParamValue(t: *const Texture, pname: GLenum) ?GLfloat {
    return switch (pname) {
        GL_TEXTURE_MIN_FILTER => @floatFromInt(minFilterEnum(t)),
        GL_TEXTURE_MAG_FILTER => @floatFromInt(@as(GLenum, if (t.mag_filter == .linear) GL_LINEAR else GL_NEAREST)),
        GL_TEXTURE_WRAP_S => @floatFromInt(wrapEnum(t.wrap_s)),
        GL_TEXTURE_WRAP_T => @floatFromInt(wrapEnum(t.wrap_t)),
        GL_TEXTURE_MAX_ANISOTROPY_EXT => t.max_anisotropy,
        GL_TEXTURE_BASE_LEVEL => @floatFromInt(t.base_level),
        GL_TEXTURE_MAX_LEVEL => @floatFromInt(t.max_level),
        GL_TEXTURE_SWIZZLE_R => @floatFromInt(swizzleToEnum(t.swizzle[0])),
        GL_TEXTURE_SWIZZLE_G => @floatFromInt(swizzleToEnum(t.swizzle[1])),
        GL_TEXTURE_SWIZZLE_B => @floatFromInt(swizzleToEnum(t.swizzle[2])),
        GL_TEXTURE_SWIZZLE_A => @floatFromInt(swizzleToEnum(t.swizzle[3])),
        GL_TEXTURE_LOD_BIAS => t.lod_bias,
        GL_TEXTURE_MIN_LOD => t.min_lod,
        GL_TEXTURE_MAX_LOD => t.max_lod,
        GL_TEXTURE_COMPARE_MODE => @floatFromInt(t.compare_mode),
        GL_TEXTURE_COMPARE_FUNC => @floatFromInt(t.compare_func),
        GL_TEXTURE_IMMUTABLE_FORMAT => if (t.immutable) 1 else 0,
        else => null,
    };
}
/// glGetTexParameteriv: the active GL_TEXTURE_2D texture's sampler state. Filter enums round-trip
/// to the combined GL enum. Max-anisotropy rounds to int.
pub fn getTexParameteriv(target: GLenum, pname: GLenum, params: ?[*]GLint) void {
    const p = params orelse return;
    if (target != GL_TEXTURE_2D and target != GL_TEXTURE_CUBE_MAP and target != GL_TEXTURE_3D and target != GL_TEXTURE_2D_ARRAY) {
        setError(GL_INVALID_ENUM);
        return;
    }
    obj_lock.lock();
    defer obj_lock.unlock();
    const t = (switch (target) {
        GL_TEXTURE_CUBE_MAP => activeCubeTextureObj(),
        GL_TEXTURE_3D => active3dTextureObj(),
        GL_TEXTURE_2D_ARRAY => active2dArrayTextureObj(),
        else => activeTextureObj(),
    }) orelse {
        setError(GL_INVALID_OPERATION);
        return;
    };
    const v = texParamValue(t, pname) orelse {
        setError(GL_INVALID_ENUM);
        return;
    };
    p[0] = @intFromFloat(@round(v));
}
/// glGetTexParameterfv: same as the integer form, keeping GL_TEXTURE_MAX_ANISOTROPY_EXT exact.
pub fn getTexParameterfv(target: GLenum, pname: GLenum, params: ?[*]GLfloat) void {
    const p = params orelse return;
    if (target != GL_TEXTURE_2D and target != GL_TEXTURE_CUBE_MAP and target != GL_TEXTURE_3D and target != GL_TEXTURE_2D_ARRAY) {
        setError(GL_INVALID_ENUM);
        return;
    }
    obj_lock.lock();
    defer obj_lock.unlock();
    const t = (switch (target) {
        GL_TEXTURE_CUBE_MAP => activeCubeTextureObj(),
        GL_TEXTURE_3D => active3dTextureObj(),
        GL_TEXTURE_2D_ARRAY => active2dArrayTextureObj(),
        else => activeTextureObj(),
    }) orelse {
        setError(GL_INVALID_OPERATION);
        return;
    };
    const v = texParamValue(t, pname) orelse {
        setError(GL_INVALID_ENUM);
        return;
    };
    p[0] = v;
}

/// One vertex-attrib array's `pname` value as f32 (null = unknown pname). GL_CURRENT_VERTEX_ATTRIB
/// is handled separately (it is 4-wide).
fn vertexAttribValue(a: *const AttribArray, pname: GLenum) ?GLfloat {
    return switch (pname) {
        GL_VERTEX_ATTRIB_ARRAY_ENABLED => if (a.enabled) 1 else 0,
        GL_VERTEX_ATTRIB_ARRAY_SIZE => @floatFromInt(a.size),
        GL_VERTEX_ATTRIB_ARRAY_STRIDE => @floatFromInt(a.stride),
        GL_VERTEX_ATTRIB_ARRAY_TYPE => @floatFromInt(a.gl_type),
        GL_VERTEX_ATTRIB_ARRAY_NORMALIZED => if (a.normalized) 1 else 0,
        GL_VERTEX_ATTRIB_ARRAY_BUFFER_BINDING => @floatFromInt(a.buffer),
        GL_VERTEX_ATTRIB_ARRAY_INTEGER => if (a.integer) 1 else 0,
        else => null,
    };
}
/// glGetVertexAttribiv: a vertex-attrib array's configuration. GL_CURRENT_VERTEX_ATTRIB reports the
/// generic attrib. Prism keeps it at the GL default (0,0,0,1) with no glVertexAttrib* setters.
pub fn getVertexAttribiv(index: GLuint, pname: GLenum, params: ?[*]GLint) void {
    const p = params orelse return;
    if (index >= MAX_ATTRIBS) {
        setError(GL_INVALID_VALUE);
        return;
    }
    if (pname == GL_CURRENT_VERTEX_ATTRIB) {
        p[0] = 0;
        p[1] = 0;
        p[2] = 0;
        p[3] = 1;
        return;
    }
    const v = vertexAttribValue(&attribs[index], pname) orelse {
        setError(GL_INVALID_ENUM);
        return;
    };
    p[0] = @intFromFloat(@round(v));
}
/// glGetVertexAttribfv: the float form (GL_CURRENT_VERTEX_ATTRIB is the 4-float generic attrib).
pub fn getVertexAttribfv(index: GLuint, pname: GLenum, params: ?[*]GLfloat) void {
    const p = params orelse return;
    if (index >= MAX_ATTRIBS) {
        setError(GL_INVALID_VALUE);
        return;
    }
    if (pname == GL_CURRENT_VERTEX_ATTRIB) {
        const g = attribs[index].generic; // the glVertexAttrib*f value (default 0,0,0,1)
        p[0] = g[0];
        p[1] = g[1];
        p[2] = g[2];
        p[3] = g[3];
        return;
    }
    const v = vertexAttribValue(&attribs[index], pname) orelse {
        setError(GL_INVALID_ENUM);
        return;
    };
    p[0] = v;
}

/// glGetBufferParameteriv: the bound buffer's size (bytes) or usage hint.
pub fn getBufferParameteriv(target: GLenum, pname: GLenum, params: ?[*]GLint) void {
    const p = params orelse return;
    const id = switch (target) {
        GL_ARRAY_BUFFER => bound_array_buffer,
        GL_ELEMENT_ARRAY_BUFFER => bound_element_buffer,
        else => {
            setError(GL_INVALID_ENUM);
            return;
        },
    };
    if (id == 0) {
        setError(GL_INVALID_OPERATION);
        return;
    }
    obj_lock.lock();
    defer obj_lock.unlock();
    const b = findBuffer(id) orelse {
        setError(GL_INVALID_OPERATION);
        return;
    };
    p[0] = switch (pname) {
        GL_BUFFER_SIZE => @intCast(b.bytes.items.len),
        GL_BUFFER_USAGE => @intCast(b.usage),
        else => {
            setError(GL_INVALID_ENUM);
            return;
        },
    };
}

/// glGetRenderbufferParameteriv: the bound renderbuffer's dimensions, internal format, and
/// per-channel bit sizes. Color renderbuffers are rgba8. Depth is 24-bit, stencil 8-bit.
pub fn getRenderbufferParameteriv(target: GLenum, pname: GLenum, params: ?[*]GLint) void {
    const p = params orelse return;
    if (target != GL_RENDERBUFFER) {
        setError(GL_INVALID_ENUM);
        return;
    }
    if (bound_renderbuffer == 0) {
        setError(GL_INVALID_OPERATION);
        return;
    }
    obj_lock.lock();
    defer obj_lock.unlock();
    const rb = findRenderbuffer(bound_renderbuffer) orelse {
        setError(GL_INVALID_OPERATION);
        return;
    };
    const color = !rb.is_depth and !rb.is_stencil;
    p[0] = switch (pname) {
        GL_RENDERBUFFER_WIDTH => @intCast(rb.width),
        GL_RENDERBUFFER_HEIGHT => @intCast(rb.height),
        GL_RENDERBUFFER_INTERNAL_FORMAT => @intCast(if (rb.is_depth and rb.is_stencil)
            GL_DEPTH24_STENCIL8
        else if (rb.is_depth)
            GL_DEPTH_COMPONENT24
        else if (rb.is_stencil)
            GL_STENCIL_INDEX8
        else
            GL_RGBA8),
        GL_RENDERBUFFER_RED_SIZE, GL_RENDERBUFFER_GREEN_SIZE, GL_RENDERBUFFER_BLUE_SIZE, GL_RENDERBUFFER_ALPHA_SIZE => if (color) 8 else 0,
        GL_RENDERBUFFER_DEPTH_SIZE => if (rb.is_depth) 24 else 0,
        GL_RENDERBUFFER_STENCIL_SIZE => if (rb.is_stencil) 8 else 0,
        else => {
            setError(GL_INVALID_ENUM);
            return;
        },
    };
}

pub fn depthFunc(func: GLenum) void {
    fixed.depth_func = switch (func) {
        GL_NEVER => .never,
        GL_LESS => .less,
        GL_EQUAL => .equal,
        GL_LEQUAL => .less_or_equal,
        GL_GREATER => .greater,
        GL_NOTEQUAL => .not_equal,
        GL_GEQUAL => .greater_or_equal,
        GL_ALWAYS => .always,
        else => {
            setError(GL_INVALID_ENUM);
            return;
        },
    };
}

/// Map a GL comparison enum (GL_LEQUAL etc.) to a HAL CompareOp; null (GL_INVALID_ENUM) on unknown.
/// Shared by GL_TEXTURE_COMPARE_FUNC (sampler2DShadow) and any other compare-func consumer.
fn compareOpFromGl(func: GLenum) ?prism.hal.CompareOp {
    return switch (func) {
        GL_NEVER => .never,
        GL_LESS => .less,
        GL_EQUAL => .equal,
        GL_LEQUAL => .less_or_equal,
        GL_GREATER => .greater,
        GL_NOTEQUAL => .not_equal,
        GL_GEQUAL => .greater_or_equal,
        GL_ALWAYS => .always,
        else => null,
    };
}

pub fn depthMask(flag: GLboolean) void {
    fixed.depth_write = flag != 0;
}

/// glPolygonOffset: set the depth-bias factor (slope-scaled) + units (constant). Takes effect
/// only while GL_POLYGON_OFFSET_FILL is enabled (wired through wantDepthState -> the HAL bias).
pub fn polygonOffset(factor: GLfloat, units: GLfloat) void {
    fixed.polygon_offset_factor = factor;
    fixed.polygon_offset_units = units;
}

pub fn clearDepthf(d: GLclampf) void {
    fixed.depth_clear = clampUnit(d);
}

// glStencilOp action enums.
pub const GL_KEEP: GLenum = 0x1E00;
pub const GL_REPLACE: GLenum = 0x1E01;
pub const GL_INCR: GLenum = 0x1E02;
pub const GL_DECR: GLenum = 0x1E03;
pub const GL_INVERT: GLenum = 0x150A;
pub const GL_INCR_WRAP: GLenum = 0x8507;
pub const GL_DECR_WRAP: GLenum = 0x8508;

/// Map a GL stencil-op action enum to the HAL StencilOp. Returns null on an unknown enum
/// (the caller raises GL_INVALID_ENUM).
fn mapStencilOp(action: GLenum) ?prism.hal.StencilOp {
    return switch (action) {
        GL_KEEP => .keep,
        GL_ZERO => .zero,
        GL_REPLACE => .replace,
        GL_INCR => .incr_clamp,
        GL_DECR => .decr_clamp,
        GL_INVERT => .invert,
        GL_INCR_WRAP => .incr_wrap,
        GL_DECR_WRAP => .decr_wrap,
        else => null,
    };
}

/// Map a GL stencil compare-func enum to the HAL CompareOp. Returns null on an unknown enum
/// (the caller raises GL_INVALID_ENUM).
fn mapStencilFunc(func: GLenum) ?prism.hal.CompareOp {
    return switch (func) {
        GL_NEVER => .never,
        GL_LESS => .less,
        GL_EQUAL => .equal,
        GL_LEQUAL => .less_or_equal,
        GL_GREATER => .greater,
        GL_NOTEQUAL => .not_equal,
        GL_GEQUAL => .greater_or_equal,
        GL_ALWAYS => .always,
        else => null,
    };
}

/// Which stencil face(s) a separate-variant call targets. Returns null on an unknown enum.
const StencilFace = enum { front, back, both };
fn mapStencilFace(face: GLenum) ?StencilFace {
    return switch (face) {
        GL_FRONT => .front,
        GL_BACK => .back,
        GL_FRONT_AND_BACK => .both,
        else => null,
    };
}

/// glStencilFunc: set the stencil compare op, reference value, and compare mask on both faces
/// (equivalent to glStencilFuncSeparate(GL_FRONT_AND_BACK, ...)). `ref` is clamped to the
/// 8-bit stencil range.
pub fn stencilFunc(func: GLenum, ref: GLint, mask: GLuint) void {
    stencilFuncSeparate(GL_FRONT_AND_BACK, func, ref, mask);
}

/// glStencilFuncSeparate: like glStencilFunc but only for the given face(s).
pub fn stencilFuncSeparate(face: GLenum, func: GLenum, ref: GLint, mask: GLuint) void {
    const which = mapStencilFace(face) orelse return setError(GL_INVALID_ENUM);
    const op = mapStencilFunc(func) orelse return setError(GL_INVALID_ENUM);
    // GL clamps the reference to [0, 2^stencilbits - 1] (8-bit stencil here).
    const clamped_ref: u8 = @intCast(std.math.clamp(ref, 0, 255));
    const vmask: u8 = @truncate(mask);
    if (which != .back) {
        fixed.stencil_func = op;
        fixed.stencil_ref = clamped_ref;
        fixed.stencil_value_mask = vmask;
    }
    if (which != .front) {
        fixed.stencil_back_func = op;
        fixed.stencil_back_ref = clamped_ref;
        fixed.stencil_back_value_mask = vmask;
    }
}

/// glStencilOp: set the stencil-fail / depth-fail / depth-pass actions on both faces.
pub fn stencilOp(sfail: GLenum, dpfail: GLenum, dppass: GLenum) void {
    stencilOpSeparate(GL_FRONT_AND_BACK, sfail, dpfail, dppass);
}

/// glStencilOpSeparate: like glStencilOp but only for the given face(s).
pub fn stencilOpSeparate(face: GLenum, sfail: GLenum, dpfail: GLenum, dppass: GLenum) void {
    const which = mapStencilFace(face) orelse return setError(GL_INVALID_ENUM);
    const sf = mapStencilOp(sfail) orelse return setError(GL_INVALID_ENUM);
    const df = mapStencilOp(dpfail) orelse return setError(GL_INVALID_ENUM);
    const dp = mapStencilOp(dppass) orelse return setError(GL_INVALID_ENUM);
    if (which != .back) {
        fixed.stencil_sfail = sf;
        fixed.stencil_dpfail = df;
        fixed.stencil_dppass = dp;
    }
    if (which != .front) {
        fixed.stencil_back_sfail = sf;
        fixed.stencil_back_dpfail = df;
        fixed.stencil_back_dppass = dp;
    }
}

/// glStencilMask: set which stencil bits a write may modify on both faces.
pub fn stencilMask(mask: GLuint) void {
    stencilMaskSeparate(GL_FRONT_AND_BACK, mask);
}

/// glStencilMaskSeparate: like glStencilMask but only for the given face(s).
pub fn stencilMaskSeparate(face: GLenum, mask: GLuint) void {
    const which = mapStencilFace(face) orelse return setError(GL_INVALID_ENUM);
    const wmask: u8 = @truncate(mask);
    if (which != .back) fixed.stencil_write_mask = wmask;
    if (which != .front) fixed.stencil_back_write_mask = wmask;
}

/// glClearStencil: the value glClear(GL_STENCIL_BUFFER_BIT) clears the stencil buffer to.
pub fn clearStencil(s: GLint) void {
    fixed.stencil_clear = @truncate(@as(u32, @bitCast(s)));
}

pub fn cullFace(mode: GLenum) void {
    fixed.cull_mode = switch (mode) {
        GL_FRONT => .front,
        GL_BACK => .back,
        // FRONT_AND_BACK would cull everything. The software cull path has no such mode,
        // so map it to back (the common case) and accept it without erroring.
        GL_FRONT_AND_BACK => .back,
        else => {
            setError(GL_INVALID_ENUM);
            return;
        },
    };
}

pub fn frontFace(mode: GLenum) void {
    fixed.front_face = switch (mode) {
        GL_CW => .clockwise,
        GL_CCW => .counter_clockwise,
        else => {
            setError(GL_INVALID_ENUM);
            return;
        },
    };
}

/// Map a GL blend-factor enum to the HAL BlendFactor; null (with GL_INVALID_ENUM set) on
/// an unknown value so the caller leaves the tracked state unchanged.
fn blendFactorFromGl(f: GLenum) ?prism.hal.BlendFactor {
    return switch (f) {
        GL_ZERO => .zero,
        GL_ONE => .one,
        GL_SRC_COLOR => .src_color,
        GL_ONE_MINUS_SRC_COLOR => .one_minus_src_color,
        GL_SRC_ALPHA => .src_alpha,
        GL_ONE_MINUS_SRC_ALPHA => .one_minus_src_alpha,
        GL_DST_ALPHA => .dst_alpha,
        GL_ONE_MINUS_DST_ALPHA => .one_minus_dst_alpha,
        GL_DST_COLOR => .dst_color,
        GL_ONE_MINUS_DST_COLOR => .one_minus_dst_color,
        GL_CONSTANT_COLOR => .constant_color,
        GL_ONE_MINUS_CONSTANT_COLOR => .one_minus_constant_color,
        GL_CONSTANT_ALPHA => .constant_alpha,
        GL_ONE_MINUS_CONSTANT_ALPHA => .one_minus_constant_alpha,
        GL_SRC_ALPHA_SATURATE => .src_alpha_saturate,
        else => null,
    };
}

/// Map a GL blend-equation enum to the HAL BlendOp; null (GL_INVALID_ENUM) on unknown.
fn blendOpFromGl(m: GLenum) ?prism.hal.BlendOp {
    return switch (m) {
        GL_FUNC_ADD => .add,
        GL_FUNC_SUBTRACT => .subtract,
        GL_FUNC_REVERSE_SUBTRACT => .reverse_subtract,
        GL_MIN => .min,
        GL_MAX => .max,
        else => null,
    };
}

pub fn blendFunc(sfactor: GLenum, dfactor: GLenum) void {
    const s = blendFactorFromGl(sfactor) orelse return setError(GL_INVALID_ENUM);
    const d = blendFactorFromGl(dfactor) orelse return setError(GL_INVALID_ENUM);
    fixed.blend_src_rgb = s;
    fixed.blend_dst_rgb = d;
    fixed.blend_src_alpha = s;
    fixed.blend_dst_alpha = d;
}

pub fn blendFuncSeparate(src_rgb: GLenum, dst_rgb: GLenum, src_alpha: GLenum, dst_alpha: GLenum) void {
    const sr = blendFactorFromGl(src_rgb) orelse return setError(GL_INVALID_ENUM);
    const dr = blendFactorFromGl(dst_rgb) orelse return setError(GL_INVALID_ENUM);
    const sa = blendFactorFromGl(src_alpha) orelse return setError(GL_INVALID_ENUM);
    const da = blendFactorFromGl(dst_alpha) orelse return setError(GL_INVALID_ENUM);
    fixed.blend_src_rgb = sr;
    fixed.blend_dst_rgb = dr;
    fixed.blend_src_alpha = sa;
    fixed.blend_dst_alpha = da;
}

pub fn blendEquation(mode: GLenum) void {
    const op = blendOpFromGl(mode) orelse return setError(GL_INVALID_ENUM);
    fixed.blend_op_rgb = op;
    fixed.blend_op_alpha = op;
}

pub fn blendEquationSeparate(mode_rgb: GLenum, mode_alpha: GLenum) void {
    const opr = blendOpFromGl(mode_rgb) orelse return setError(GL_INVALID_ENUM);
    const opa = blendOpFromGl(mode_alpha) orelse return setError(GL_INVALID_ENUM);
    fixed.blend_op_rgb = opr;
    fixed.blend_op_alpha = opa;
}

pub fn blendColor(r: GLclampf, g: GLclampf, b: GLclampf, a: GLclampf) void {
    fixed.blend_color = .{ clampUnit(r), clampUnit(g), clampUnit(b), clampUnit(a) };
}

/// glColorMask: enable/disable writes to each color channel. Applies to every draw until
/// changed, independent of blending. A stencil-only mask pass sets all false.
pub fn colorMask(r: GLboolean, g: GLboolean, b: GLboolean, a: GLboolean) void {
    fixed.color_mask = .{ r != 0, g != 0, b != 0, a != 0 };
}

pub fn setViewport(x: GLint, y: GLint, width: GLsizei, height: GLsizei) void {
    if (width < 0 or height < 0) {
        setError(GL_INVALID_VALUE);
        return;
    }
    viewport = .{ x, y, width, height };
}

pub fn getViewport() [4]GLint {
    return viewport;
}

/// The line width (glLineWidth) for GL_LINES draws, clamped to >= 1 (GLES supports 1..N).
/// Fed into the HAL pipeline (line_list) at build time. The pipeline is cache-keyed on it.
threadlocal var line_width: f32 = 1.0;
/// GL_PRIMITIVE_RESTART_FIXED_INDEX enabled (glEnable): the type's max index restarts strips/fans.
threadlocal var primitive_restart_fixed: bool = false;

/// glLineWidth: set the line width in pixels (must be > 0, else GL_INVALID_VALUE).
pub fn setLineWidth(w: GLfloat) void {
    if (w <= 0) {
        setError(GL_INVALID_VALUE);
        return;
    }
    line_width = w;
}

/// glSampleCoverage: set the GL_SAMPLE_COVERAGE fraction (clamped to [0,1]) and invert flag. Takes
/// effect only while GL_SAMPLE_COVERAGE is enabled and the draw target is multisampled.
pub fn sampleCoverage(value: GLfloat, invert: GLboolean) void {
    fixed.sample_coverage_value = std.math.clamp(value, 0.0, 1.0);
    fixed.sample_coverage_invert = invert != GL_FALSE;
}

/// glScissor: set the scissor box (GL bottom-left window coords). Negative width/height
/// is GL_INVALID_VALUE. The box only clips when GL_SCISSOR_TEST is enabled. The draw path
/// flips y to the HAL's top-left origin at submit time.
pub fn setScissorBox(x: GLint, y: GLint, width: GLsizei, height: GLsizei) void {
    if (width < 0 or height < 0) {
        setError(GL_INVALID_VALUE);
        return;
    }
    fixed.scissor_box = .{ x, y, width, height };
    fixed.scissor_set = true;
}

/// The glViewport rect as a HAL Viewport (window pixels, top-left origin), or null for the
/// full render target (unset/degenerate, or a viewport that already covers the whole RT).
/// A null viewport is byte-identical to the pre-viewport path. Y conversion is the same as
/// currentScissorRect: top = height - (y + h), correct for both the default framebuffer and FBOs.
fn currentViewportRect(target_width: u32, target_height: u32) ?prism.hal.Viewport {
    const v = viewport; // GL {x, y, w, h}, bottom-left origin
    const dr = depth_range; // glDepthRangef {near, far}
    // A "full" xy-viewport = unset/degenerate, or one that already covers the whole render target.
    const full_xy = (v[2] <= 0 or v[3] <= 0) or
        (v[0] == 0 and v[1] == 0 and v[2] == @as(GLint, @intCast(target_width)) and v[3] == @as(GLint, @intCast(target_height)));
    const default_z = dr[0] == 0.0 and dr[1] == 1.0;
    if (full_xy and default_z) return null; // nothing to apply -> full RT (byte-identical to before)
    // xy rect: full-RT when the xy-viewport is full (but the depth range is not, so a viewport is
    // still emitted to carry it). Otherwise the GL bottom-left rect converted to top-left (top = h-(y+h)).
    const th: i64 = @intCast(target_height);
    const rx: i32 = if (full_xy) 0 else v[0];
    const ry: i32 = if (full_xy) 0 else @intCast(th - (@as(i64, v[1]) + @as(i64, v[3])));
    const rw: u32 = if (full_xy) target_width else @intCast(v[2]);
    const rh: u32 = if (full_xy) target_height else @intCast(v[3]);
    return .{ .x = rx, .y = ry, .width = rw, .height = rh, .depth_near = dr[0], .depth_far = dr[1] };
}

/// Effective HAL scissor rect for a draw/clear into a `target_height`-tall render target.
/// Returns null when the scissor test is off. Otherwise returns the GL box with y flipped
/// from bottom-left to the HAL top-left origin.
fn currentScissorRect(target_height: u32) ?prism.hal.ScissorRect {
    if (!fixed.scissor_test) return null;
    // Before glScissor is called GL's box is the full window. Model that as "no clip".
    if (!fixed.scissor_set) return null;
    const b = fixed.scissor_box;
    const th: i64 = @intCast(target_height);
    // GL y is the bottom edge (origin bottom-left). The HAL rect's y is the TOP edge:
    // top = height - (y + h).
    const top: i64 = th - (@as(i64, b[1]) + @as(i64, b[3]));
    return .{
        .x = b[0],
        .y = @intCast(top),
        .width = @intCast(b[2]),
        .height = @intCast(b[3]),
    };
}

/// glClear: clear the current EGL-draw surface's HAL backbuffer to the clear
/// color through the software driver's proven clear path (beginCommands ->
/// setRenderTarget -> clear -> submit). GL_COLOR_BUFFER_BIT does the color clear;
/// GL_DEPTH/STENCIL_BUFFER_BIT are accepted (no depth attachment in this minimal
/// path, so they are honored as no-ops on the color-only backbuffer). An unknown
/// bit raises GL_INVALID_VALUE. With no current context/draw surface this is a
/// GL_INVALID_OPERATION (matches GL: a clear with no framebuffer is invalid).
/// Per-thread depth clear pending flag. glClear(GL_DEPTH_BUFFER_BIT) sets it. The next
/// depth-tested draw consumes it by clearing the depth attachment via the render pass
/// (setDepthTarget clear_value). This defers the depth clear to the draw that binds the
/// depth buffer (the software depth path has no standalone depth-clear command). Correct
/// for the clear-then-draw-all-gears frame es2gears renders.
threadlocal var pending_depth_clear: bool = false;

/// Per-thread stencil clear flag (glClear(GL_STENCIL_BUFFER_BIT)). Consumed by the next
/// stencil-tested draw via setStencilTarget's clear value. Same deferred-clear contract as
/// depth. Default-framebuffer path only (v1).
threadlocal var pending_stencil_clear: bool = false;

pub fn clear(mask: GLbitfield) void {
    const known = GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT;
    if ((mask & ~known) != 0) {
        setError(GL_INVALID_VALUE);
        return;
    }
    const ctx = state.currentContext() orelse {
        setError(GL_INVALID_OPERATION);
        return;
    };
    const surf = state.currentDrawSurface() orelse {
        setError(GL_INVALID_OPERATION);
        return;
    };
    if ((mask & GL_DEPTH_BUFFER_BIT) != 0) pending_depth_clear = true;
    if ((mask & GL_STENCIL_BUFFER_BIT) != 0) pending_stencil_clear = true;

    // Render-to-texture: when an FBO is bound, the clear targets the FBO's color/depth
    // attachments instead of the surface backbuffer.
    if (bound_framebuffer != 0) {
        obj_lock.lock();
        // resolveRenderTargets wants want_depth = whether the FBO has a depth attachment so
        // a depth clear can land. pass true so the depth attachment is resolved when present.
        const rt = resolveRenderTargets(ctx.device(), surf, true);
        const depth_tex = if (rt) |r| r.depth_tex else null;
        obj_lock.unlock();
        if (rt) |r| {
            if ((mask & GL_COLOR_BUFFER_BIT) != 0) {
                const sc = currentScissorRect(r.color_h);
                const cc = prism.hal.Color{ .r = clear_color.r, .g = clear_color.g, .b = clear_color.b, .a = clear_color.a };
                ctx.clearTarget(r.color, cc, sc) catch setError(GL_INVALID_OPERATION);
                // MRT: glClear clears EVERY bound draw buffer, not just attachment 0.
                for (r.extra_colors[0..r.extra_color_count]) |extra| {
                    if (extra) |e| ctx.clearTarget(e, cc, sc) catch setError(GL_INVALID_OPERATION);
                }
            }
            if ((mask & GL_DEPTH_BUFFER_BIT) != 0) {
                if (r.depth) |d| {
                    ctx.clearDepthTarget(r.color, d, fixed.depth_clear) catch setError(GL_INVALID_OPERATION);
                    pending_depth_clear = false;
                    // A standalone depth clear changes the depth-texture contents.
                    if (depth_tex) |dt| {
                        obj_lock.lock();
                        dt.depth_dirty = true;
                        obj_lock.unlock();
                    }
                }
            }
        }
        return;
    }

    if ((mask & GL_COLOR_BUFFER_BIT) == 0) return; // nothing to do on the color buffer
    const sc = currentScissorRect(surf.height);
    ctx.clearBackbuffer(surf, .{
        .r = clear_color.r,
        .g = clear_color.g,
        .b = clear_color.b,
        .a = clear_color.a,
    }, sc) catch {
        setError(GL_INVALID_OPERATION);
    };
}

/// Clear the current draw framebuffer's color to `rgba` immediately. Reuses the glClear color
/// path via a temporary clear color, so it lands on the FBO attachment or the backbuffer alike.
/// A float RT keeps HDR values (clear_color is not re-clamped here).
fn clearColorNow(rgba: [4]f32) void {
    const saved = clear_color;
    clear_color = .{ .r = rgba[0], .g = rgba[1], .b = rgba[2], .a = rgba[3] };
    clear(GL_COLOR_BUFFER_BIT);
    clear_color = saved;
}

/// glClearBufferfv(buffer, drawbuffer, value): the GLES3 typed clear. GL_COLOR clears the color to
/// the 4 floats (HDR-preserving on a float RT). GL_DEPTH clears depth to value[0].
/// Immediately clear the bound draw framebuffer's MRT color attachment `index` (index >= 1) to
/// `rgba`. Attachment 0 uses the deferred clearColorNow path. The extras have no deferred slot so
/// they clear via an immediate HAL submit (state.Context.clearColorTargetNow). No-op if there is no
/// bound FBO or that attachment is unbound. Caller must not hold obj_lock.
fn clearColorAttachmentExtra(index: u32, rgba: [4]f32) void {
    if (bound_framebuffer == 0) return; // the default framebuffer has no MRT extras
    const ctx = state.currentContext() orelse return;
    const dev = ctx.device();
    obj_lock.lock();
    const img: ?*prism.hal.Resource = blk: {
        const f = findFramebuffer(bound_framebuffer) orelse break :blk null;
        const slot = index - 1;
        if (slot >= f.extra_color_tex.len) break :blk null;
        if (f.extra_color_tex[slot] != 0) {
            const ct = findTexture(f.extra_color_tex[slot]) orelse break :blk null;
            const im = ensureTextureHal(dev, ct) catch break :blk null;
            ct.hal_dirty = false;
            break :blk im;
        } else if (f.extra_color_rb[slot] != 0) {
            const cr = findRenderbuffer(f.extra_color_rb[slot]) orelse break :blk null;
            break :blk ensureRenderbufferHal(dev, cr) catch break :blk null;
        }
        break :blk null;
    };
    obj_lock.unlock();
    if (img) |im| ctx.clearTarget(im, .{ .r = rgba[0], .g = rgba[1], .b = rgba[2], .a = rgba[3] }, null) catch {};
}

pub fn clearBufferfv(buffer: GLenum, drawbuffer: GLint, value: ?[*]const GLfloat) void {
    const v = value orelse return;
    switch (buffer) {
        // glClearBufferfv(GL_COLOR, drawbuffer, rgba): clear ONE draw buffer. drawbuffer 0 is
        // attachment 0 (the deferred clear). drawbuffer >= 1 is an MRT extra attachment (immediate).
        GL_COLOR => {
            if (drawbuffer < 0 or drawbuffer >= @as(GLint, @intCast(prism.hal.MAX_COLOR_TARGETS))) {
                setError(GL_INVALID_VALUE);
                return;
            }
            if (drawbuffer == 0) clearColorNow(.{ v[0], v[1], v[2], v[3] }) else clearColorAttachmentExtra(@intCast(drawbuffer), .{ v[0], v[1], v[2], v[3] });
        },
        GL_DEPTH => {
            fixed.depth_clear = v[0];
            clear(GL_DEPTH_BUFFER_BIT); // deferred to the next draw, matching glClear(GL_DEPTH)
        },
        else => setError(GL_INVALID_ENUM),
    }
}

/// glClearBufferiv: GL_STENCIL clears the stencil to value[0]. GL_COLOR clears an integer color
/// target (Prism stores unorm, so the ints are taken as the color directly, clamped).
pub fn clearBufferiv(buffer: GLenum, drawbuffer: GLint, value: ?[*]const GLint) void {
    _ = drawbuffer;
    const v = value orelse return;
    switch (buffer) {
        GL_STENCIL => {
            fixed.stencil_clear = @intCast(v[0] & 0xff);
            clear(GL_STENCIL_BUFFER_BIT);
        },
        GL_COLOR => clearColorNow(.{ @floatFromInt(v[0]), @floatFromInt(v[1]), @floatFromInt(v[2]), @floatFromInt(v[3]) }),
        else => setError(GL_INVALID_ENUM),
    }
}

/// glClearBufferuiv: GL_COLOR clears an unsigned-integer color target.
pub fn clearBufferuiv(buffer: GLenum, drawbuffer: GLint, value: ?[*]const u32) void {
    _ = drawbuffer;
    const v = value orelse return;
    switch (buffer) {
        GL_COLOR => clearColorNow(.{ @floatFromInt(v[0]), @floatFromInt(v[1]), @floatFromInt(v[2]), @floatFromInt(v[3]) }),
        else => setError(GL_INVALID_ENUM),
    }
}

/// glReadBuffer / glDrawBuffers select which color buffer glReadPixels reads / the fragment
/// outputs write. Prism reads/writes color attachment 0 (or the back buffer) with MRT driven by the
/// FBO attachments + the FS, so these VALIDATE the selection + store it for GL_READ_BUFFER queries;
/// the actual routing is unchanged. Valid: GL_NONE, GL_BACK (default fb), GL_COLOR_ATTACHMENTi (FBO).
threadlocal var read_buffer: GLenum = GL_BACK;
threadlocal var draw_buffer0: GLenum = GL_BACK;
fn validColorBufferEnum(b: GLenum) bool {
    return b == GL_NONE or b == GL_BACK or (b >= GL_COLOR_ATTACHMENT0 and b < GL_COLOR_ATTACHMENT0 + prism.hal.MAX_COLOR_TARGETS);
}
/// glGetInternalformativ(GL_RENDERBUFFER, fmt, pname, ...): report the MSAA sample counts Prism
/// supports for a renderable format. GL_NUM_SAMPLE_COUNTS = how many; GL_SAMPLES = the counts in
/// DESCENDING order (Prism does 4x and 2x supersampling). Apps call this before an MSAA renderbuffer.
pub fn getInternalformativ(target: GLenum, internalformat: GLenum, pname: GLenum, buf_size: GLsizei, params: ?[*]GLint) void {
    _ = internalformat;
    if (target != GL_RENDERBUFFER) {
        setError(GL_INVALID_ENUM);
        return;
    }
    if (buf_size < 0) {
        setError(GL_INVALID_VALUE);
        return;
    }
    const p = params orelse return;
    const counts = [_]GLint{ 4, 2 }; // supported multisample counts, descending
    switch (pname) {
        GL_NUM_SAMPLE_COUNTS => {
            if (buf_size >= 1) p[0] = counts.len;
        },
        GL_SAMPLES => {
            var i: usize = 0;
            while (i < counts.len and i < @as(usize, @intCast(buf_size))) : (i += 1) p[i] = counts[i];
        },
        else => setError(GL_INVALID_ENUM),
    }
}

pub fn readBuffer(src: GLenum) void {
    if (!validColorBufferEnum(src)) {
        setError(GL_INVALID_ENUM);
        return;
    }
    read_buffer = src;
}
pub fn drawBuffers(n: GLsizei, bufs: ?[*]const GLenum) void {
    if (n < 0) {
        setError(GL_INVALID_VALUE);
        return;
    }
    const b = bufs orelse return;
    var i: usize = 0;
    while (i < @as(usize, @intCast(n))) : (i += 1) {
        if (!validColorBufferEnum(b[i])) {
            setError(GL_INVALID_ENUM);
            return;
        }
    }
    if (n >= 1) draw_buffer0 = b[0];
}

/// glClearBufferfi(GL_DEPTH_STENCIL, 0, depth, stencil): clear the packed depth+stencil at once.
pub fn clearBufferfi(buffer: GLenum, drawbuffer: GLint, depth: GLfloat, stencil: GLint) void {
    _ = drawbuffer;
    if (buffer != GL_DEPTH_STENCIL) {
        setError(GL_INVALID_ENUM);
        return;
    }
    fixed.depth_clear = depth;
    fixed.stencil_clear = @intCast(stencil & 0xff);
    clear(GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
}

// GLES2 shader objects, vertex attributes, and glDrawArrays.
// GL object ids are process-wide GLuints. Buffers/shaders/programs live in
// id-keyed registries guarded by a CAS spinlock (GLES is callable from any thread;
// no Io handle for a mutex, matching state.zig's display registry). Per-thread
// bindings (bound program, bound array buffer, attrib arrays) are threadlocal,
// matching the GL per-context model. eglMakeCurrent binds a single current context.

const gpa = state.gpa;

/// A tiny CAS spinlock matching state.zig's pattern. GLES is callable from any thread
/// and there is no Io handle for a real mutex.
const SpinLock = struct {
    flag: std.atomic.Value(bool) = .init(false),
    fn lock(self: *SpinLock) void {
        while (self.flag.cmpxchgWeak(false, true, .acquire, .monotonic) != null) std.atomic.spinLoopHint();
    }
    fn unlock(self: *SpinLock) void {
        self.flag.store(false, .release);
    }
};
var obj_lock: SpinLock = .{};

/// A GLES buffer object (glGenBuffers). `bytes` holds the CPU-side data uploaded
/// by glBufferData. `hal` is the lazily-(re)built HAL vertex Resource the draw
/// binds. The HAL Resource is rebuilt whenever the data or size changes.
const Buffer = struct {
    id: GLuint,
    bytes: std.ArrayListUnmanaged(u8) = .empty,
    hal: ?*prism.hal.Resource = null,
    hal_dev: ?prism.hal.Device = null, // the device that owns `hal` (to destroy it)
    // A separate HAL Resource for when this buffer is used as a UBO (glBindBufferBase +
    // GL_UNIFORM_BUFFER), so a buffer used as both a VBO and a UBO keeps distinct allocations.
    // Lazily built at draw and re-uploaded from `bytes`. Invalidated on glBufferData (size change).
    ubo_hal: ?*prism.hal.Resource = null,
    ubo_hal_dev: ?prism.hal.Device = null,
    mapped: bool = false, // GL_OES_mapbuffer: a glMapBufferOES region is outstanding
    usage: GLenum = GL_STATIC_DRAW, // the glBufferData hint (reported by glGetBufferParameteriv)
};

/// A GLES texture object (glGenTextures). `bytes` holds the CPU-side RGBA8 texels
/// (always expanded to 4 channels regardless of the upload format, so the software
/// sampler's rgba8 TexDesc reads them directly). `width`/`height` are the base-level
/// dimensions. `hal` is the lazily-(re)built HAL image Resource the draw binds as a
/// combined-image-sampler. Sampler state (filter/wrap) lives here per GL's "texture
/// object carries its sampler state" model and feeds the HAL TextureBinding at draw.
const Texture = struct {
    id: GLuint,
    width: u32 = 0,
    height: u32 = 0,
    bytes: std.ArrayListUnmanaged(u8) = .empty, // tightly-packed texels of `format` (width*height*bpp)
    /// The stored texel format. rgba8_unorm is the historical 8-bit path. rgba8_srgb tags an
    /// sRGB-encoded upload (the sampler decodes on read). rgba16_float / r32g32b32a32_float
    /// store IEEE half / single per channel (HDR textures). Drives the HAL image format +
    /// the per-texel storage size.
    format: prism.hal.Format = .rgba8_unorm,
    bound_once: bool = false, // glBindTexture(GL_TEXTURE_2D) was issued (glIsTexture)
    // CUBEMAP: bound to GL_TEXTURE_CUBE_MAP. `bytes` holds 6 faces packed contiguously in GL
    // order (+X,-X,+Y,-Y,+Z,-Z), each width*height*bpp. A face is uploaded via glTexImage2D
    // with a GL_TEXTURE_CUBE_MAP_* face target. The HAL image is built with `.cube = true`.
    is_cube: bool = false,
    // 3D TEXTURE: bound to GL_TEXTURE_3D. `bytes` holds `depth` slices of width*height texels,
    // packed slice-major (glTexImage3D). The HAL image is built with `.depth = depth`.
    is_3d: bool = false,
    depth: u32 = 1,
    // 2D ARRAY TEXTURE: bound to GL_TEXTURE_2D_ARRAY. Same `depth`-layer slice-major backing as a
    // 3D texture (uploaded via glTexImage3D), but the HAL image is built with `.array = true` and a
    // `sampler2DArray` selects one layer by a raw index (no cross-layer filtering).
    is_array: bool = false,
    // Sampler state (defaults match GL: min/mag LINEAR-ish; GL's true min default is
    // NEAREST_MIPMAP_LINEAR but with no mipmaps we sample the base level, so LINEAR/NEAREST
    // is the meaningful choice. GL's mag default is LINEAR).
    min_filter: prism.hal.Filter = .nearest,
    mag_filter: prism.hal.Filter = .linear,
    // The mip-filter mode from the GL_TEXTURE_MIN_FILTER (GL_*_MIPMAP_* -> nearest/linear; the
    // non-mipmap filters -> none). Only takes effect once the texture has a mip chain (has_mipmaps).
    mip_filter: prism.hal.MipFilter = .none,
    wrap_s: prism.hal.AddressMode = .repeat,
    wrap_t: prism.hal.AddressMode = .repeat,
    // GL_TEXTURE_MAX_ANISOTROPY_EXT (default 1 = off). Clamped to the advertised limit.
    max_anisotropy: f32 = 1,
    // GL_TEXTURE_BASE_LEVEL / GL_TEXTURE_MAX_LEVEL: the finest / coarsest mip level sampling uses
    // (GL defaults 0 / 1000). Passed into the HAL TextureBinding so the sampler clamps its level.
    base_level: u32 = 0,
    max_level: u32 = 1000,
    // GL_TEXTURE_SWIZZLE_R/G/B/A: per-output-channel source remap (identity r,g,b,a by default).
    swizzle: [4]prism.hal.Swizzle = .{ .r, .g, .b, .a },
    // GL_TEXTURE_LOD_BIAS: added to the computed mip LOD (default 0).
    lod_bias: f32 = 0,
    // GL_TEXTURE_MIN_LOD / GL_TEXTURE_MAX_LOD: clamp the computed LOD (GL defaults -1000 / 1000).
    min_lod: f32 = -1000,
    max_lod: f32 = 1000,
    // GL_TEXTURE_COMPARE_MODE (GL_NONE or GL_COMPARE_REF_TO_TEXTURE) + GL_TEXTURE_COMPARE_FUNC
    // (default GL_LEQUAL): sampler2DShadow depth compare. GL_NONE (default) = an ordinary read.
    compare_mode: GLenum = GL_NONE,
    compare_func: GLenum = GL_LEQUAL,
    // glGenerateMipmap was called: the HAL image is (re)built with a full box-downsampled mip
    // chain and the sampler may minify through it (given a mip min-filter).
    has_mipmaps: bool = false,
    // glTexStorage2D declared this texture's storage immutable: its dimensions/format are fixed,
    // glTexImage2D is rejected (GL_INVALID_OPERATION), and only glTexSubImage2D may update texels.
    immutable: bool = false,
    // The lazily-(re)built HAL image Resource + the device that owns it.
    hal: ?*prism.hal.Resource = null,
    hal_dev: ?prism.hal.Device = null,
    hal_levels: u8 = 1, // mip level count the current `hal` image was built with (recreate on change)
    hal_dirty: bool = true, // CPU texels changed -> re-upload to the HAL image
    // GL_OES_depth_texture: a texture created with GL_DEPTH_COMPONENT. Its render-target
    // is a HAL depth32_float image (`depth_hal`). When it is later sampled as a sampler2D
    // the f32 depths are converted into `bytes` (R=G=B=depth, A=1) so the existing RGBA8
    // host sampler reads the depth in `.x` (what shadow.frag / refract's maps expect).
    is_depth: bool = false,
    depth_hal: ?*prism.hal.Resource = null, // the depth32_float attachment image
    depth_dirty: bool = false, // depth render happened -> rebuild `bytes` before sampling
    // On a driver whose rendered depth surface is tiled/non-sampleable (nvidia's ZETA), the
    // rgba8 "depth-as-color" conversion below would make a sampler2DShadow hardware compare a
    // no-op. Such a driver exposes hal Device.finalizeDepthTexture, which de-tiles the rendered
    // depth into a real ZF32 sampled texture (the format the HW DEPTH_COMPARE engages on). It is
    // stored here and bound instead of the rgba8 `hal` image. Null on the software path.
    depth_sampled_hal: ?*prism.hal.Resource = null,
    // This texture is also used as an FBO GL_COLOR_ATTACHMENT0: it is rendered into,
    // not just sampled. Its HAL image must be created with render_target usage (a
    // sampled-only image is not a valid render target. On nvidia an RT must be
    // block-linear). Set by glFramebufferTexture2D. Render-to-texture (glmark2 desktop's
    // window+shadow passes, jellyfish, etc.) depends on this.
    is_rt: bool = false,
};

/// A GLES framebuffer object (glGenFramebuffers). Holds the color + depth attachments
/// (each a texture id, or a renderbuffer id, or 0). When bound (glBindFramebuffer), the
/// GLES clear/draw path redirects its render-target to the color attachment's HAL image
/// and its depth-target to the depth attachment's HAL depth32_float buffer. This is the
/// render-to-texture path glmark2's shadow/refract scenes use. The default framebuffer (id 0)
/// is the window/pbuffer backbuffer and is not an object here (a null bound FBO means it).
const Framebuffer = struct {
    id: GLuint,
    color_tex: GLuint = 0, // GL_COLOR_ATTACHMENT0 texture id (0 = none)
    color_rb: GLuint = 0, // GL_COLOR_ATTACHMENT0 renderbuffer id (0 = none)
    // MRT: GL_COLOR_ATTACHMENT1..N-1 (index 0 here = attachment 1). A fragment shader with
    // multiple located `out`s (or gl_FragData[]) writes each to its matching attachment. The
    // draw binds these as HAL extra color targets (cb.setColorTarget). Deferred-shading G-buffers.
    extra_color_tex: [prism.hal.MAX_COLOR_TARGETS - 1]GLuint = .{0} ** (prism.hal.MAX_COLOR_TARGETS - 1),
    extra_color_rb: [prism.hal.MAX_COLOR_TARGETS - 1]GLuint = .{0} ** (prism.hal.MAX_COLOR_TARGETS - 1),
    depth_tex: GLuint = 0, // GL_DEPTH_ATTACHMENT texture id (0 = none)
    depth_rb: GLuint = 0, // GL_DEPTH_ATTACHMENT renderbuffer id (0 = none)
    stencil_rb: GLuint = 0, // GL_STENCIL_ATTACHMENT renderbuffer id (0 = none). A packed
    // GL_DEPTH_STENCIL_ATTACHMENT (GL_DEPTH24_STENCIL8) sets BOTH depth_rb and stencil_rb to
    // the same renderbuffer, which then backs a depth image AND a u8/pixel stencil buffer.
    bound_once: bool = false,
};

/// A GLES renderbuffer object (glGenRenderbuffers). A plain off-screen render target with
/// no sampling (glRenderbufferStorage sets its format + size). Depth renderbuffers back a
/// HAL depth32_float buffer. Color renderbuffers back an rgba8_unorm image. Used by an FBO
/// that needs a depth buffer it never samples (the common refract distance/normal pass).
const Renderbuffer = struct {
    id: GLuint,
    width: u32 = 0,
    height: u32 = 0,
    is_depth: bool = false,
    is_stencil: bool = false, // GL_STENCIL_INDEX8, or (with is_depth) a packed GL_DEPTH24_STENCIL8
    samples: u8 = 1, // glRenderbufferStorageMultisample: >1 = a multisampled attachment
    hal: ?*prism.hal.Resource = null, // color rgba8 image OR depth32_float image
    stencil_hal: ?*prism.hal.Resource = null, // u8/pixel stencil BUFFER (is_stencil renderbuffers)
    resolved_hal: ?*prism.hal.Resource = null, // single-sample resolve target (blit source for MSAA)
    hal_dev: ?prism.hal.Device = null,
    bound_once: bool = false,
};

/// A GLES shader object (glCreateShader). For the SPIR-V binary path (glShaderBinary
/// with GL_SHADER_BINARY_FORMAT_SPIR_V) `spirv` holds the word stream that is handed
/// to the HAL createShaderModule verbatim at link time. `source` retains GLSL-ES text
/// from glShaderSource for a future GLSL front-end milestone (not compiled here).
const Shader = struct {
    id: GLuint,
    stage: prism.hal.ShaderStage, // .vertex / .fragment
    source: std.ArrayListUnmanaged(u8) = .empty,
    spirv: std.ArrayListUnmanaged(u8) = .empty,
    compiled: bool = false,
    deleted: bool = false,
    // The GLSL compile diagnostic (glGetShaderInfoLog). Empty on success or when the
    // SPIR-V binary path is used. NUL-terminated content is appended on demand.
    info_log: std.ArrayListUnmanaged(u8) = .empty,
    // The default-uniform-block layout resolved when GLSL source was compiled (name ->
    // byte offset). Empty for the SPIR-V binary path (no GLSL source) or a shader with no
    // uniforms. linkProgram copies these onto the program for glUniform* resolution.
    uniforms: std.ArrayListUnmanaged(prism.glsl.UniformMember) = .empty,
    uniform_block_size: u32 = 0,
    // The `uniform sampler2D` members (name -> SPIR-V binding) from GLSL compilation.
    // linkProgram copies these onto the program for glUniform1i(sampler) resolution.
    samplers: std.ArrayListUnmanaged(prism.glsl.SamplerMember) = .empty,
    // The real vertex attributes (name -> location + components) from GLSL compilation.
    // Empty for a fragment shader or the SPIR-V binary path (no GLSL source -> no names).
    // linkProgram copies the VS's onto the program for glGetAttribLocation resolution.
    attributes: std.ArrayListUnmanaged(prism.glsl.AttributeMember) = .empty,
    // The VS OUTPUT varyings (name -> location + components) from GLSL compilation. Empty for a
    // fragment shader. linkProgram copies the VS's onto the program so glTransformFeedbackVaryings
    // can map a capture name to the VS output location the transform-feedback path reads.
    outputs: std.ArrayListUnmanaged(prism.glsl.OutputMember) = .empty,
    // The named uniform interface blocks (glGetUniformBlockIndex) from GLSL compilation.
    // linkProgram copies these onto the program, tagging each with this shader's stage so
    // the draw binds the glBindBufferBase'd buffer at the right per-stage UBO slot.
    uniform_blocks: std.ArrayListUnmanaged(prism.glsl.UniformBlock) = .empty,
};

/// A GLES program object (glCreateProgram). Holds the attached VS/FS shader ids and,
/// after glLinkProgram, the built HAL ShaderModules + Pipeline (built from the
/// shaders' SPIR-V, which rides spirv.zig + the Vulcan JIT in the software driver).
const Program = struct {
    id: GLuint,
    vs_shader: ?GLuint = null,
    fs_shader: ?GLuint = null,
    linked: bool = false,
    deleted: bool = false,
    // The link diagnostic for glGetProgramInfoLog (NUL-terminated when non-empty). Set on a
    // link failure so the app (e.g. glmark2) sees WHY the link failed instead of a blank log.
    info_log: std.ArrayListUnmanaged(u8) = .empty,
    // HAL objects built at link time (owned by the program; destroyed on delete).
    hal_dev: ?prism.hal.Device = null,
    hal_vs: ?*prism.hal.ShaderModule = null,
    hal_fs: ?*prism.hal.ShaderModule = null,
    hal_pipeline: ?*prism.hal.Pipeline = null,
    // The y-flip (GL bottom-left origin) is baked into hal_vs and is correct only for the
    // default framebuffer (which is presented top-down). Rendering to an FBO texture that
    // is later sampled must not flip: the flip would store the texture upside-down and
    // invert the winding so cull discards it. hal_vs_noflip is the unflipped VS used for FBO
    // render targets, with its own pipeline (no cull-winding inversion).
    hal_vs_noflip: ?*prism.hal.ShaderModule = null,
    hal_pipeline_noflip: ?*prism.hal.Pipeline = null,
    pipeline_noflip_depth: prism.hal.DepthState = .{},
    pipeline_noflip_cull: prism.hal.CullState = .{},
    pipeline_noflip_blend: prism.hal.BlendState = .{},
    pipeline_noflip_stencil: prism.hal.StencilState = .{},
    pipeline_noflip_stencil_back: ?prism.hal.StencilState = null,
    pipeline_noflip_topology: prism.hal.Topology = .triangle_list,
    pipeline_noflip_line_width: f32 = 1.0,
    // The color format the pipeline was built for (the surface backbuffer's
    // rgba8_unorm). A relink would rebuild against the current draw surface.
    color_format: prism.hal.Format = .rgba8_unorm,
    // The depth/cull state baked into hal_pipeline (so the pipeline is rebuilt if the
    // app toggles depth/cull after the first draw).
    pipeline_depth: prism.hal.DepthState = .{},
    pipeline_cull: prism.hal.CullState = .{},
    pipeline_blend: prism.hal.BlendState = .{},
    pipeline_stencil: prism.hal.StencilState = .{},
    pipeline_stencil_back: ?prism.hal.StencilState = null,
    pipeline_topology: prism.hal.Topology = .triangle_list,
    pipeline_line_width: f32 = 1.0,
    pipeline_a2c: bool = false, // GL_SAMPLE_ALPHA_TO_COVERAGE (flip pipeline)
    pipeline_noflip_a2c: bool = false, // (noflip pipeline)
    pipeline_scov: bool = false, // GL_SAMPLE_COVERAGE enable (flip pipeline)
    pipeline_scov_value: f32 = 1.0,
    pipeline_scov_invert: bool = false,
    pipeline_noflip_scov: bool = false, // (noflip pipeline)
    pipeline_noflip_scov_value: f32 = 1.0,
    pipeline_noflip_scov_invert: bool = false,

    // --- Default-uniform-block store (glUniform*) ---------------------------
    // GLES2's loose `uniform`s form one default block per stage, and each stage lays its block
    // out independently from offset 0. A VS-only uniform (e.g. MVP) and an FS-only uniform
    // (e.g. NormalMatrix) both sit at offset 0. The two stages therefore get separate backing
    // buffers bound at separate bindings (VS block -> binding 0, FS block -> binding 1; samplers
    // start at binding 2). `uniforms` lists each member once (glGetUniformLocation returns its
    // index) with its offset in each stage that declares it (-1 if absent). glUniform* writes the
    // value into every stage's buffer that carries it. vs_/fs_uniform_bytes are the CPU blocks.
    // vs_/fs_uniform_hal are the lazily-(re)built HAL UBO Resources (null = that stage declares
    // no default-block uniforms).
    uniforms: std.ArrayListUnmanaged(ProgUniform) = .empty,
    vs_uniform_bytes: std.ArrayListUnmanaged(u8) = .empty,
    fs_uniform_bytes: std.ArrayListUnmanaged(u8) = .empty,
    uniform_dirty: bool = true, // the CPU bytes changed -> re-upload to the HAL UBOs
    vs_uniform_hal: ?*prism.hal.Resource = null,
    fs_uniform_hal: ?*prism.hal.Resource = null,

    // App-overridden attribute locations (glBindAttribLocation, name -> location). Take
    // precedence over the program's real attribute list. Applied at getAttribLocation.
    attrib_bindings: std.ArrayListUnmanaged(struct { name: []u8, location: GLuint }) = .empty,

    // The program's vertex attributes (name -> location + components), copied from the VS
    // at link time (from the GLSL front end). glGetAttribLocation/glGetActiveAttrib + the
    // draw-time vertex layout resolve against this list (a glBindAttribLocation override wins).
    // Empty for the SPIR-V binary path (no GLSL source -> no attribute names). Those programs
    // rely on glBindAttribLocation, as the M3 oracles do.
    attributes: std.ArrayListUnmanaged(prism.glsl.AttributeMember) = .empty,

    // --- Sampler uniforms (a `uniform sampler2D` -> a texture unit -> a HAL binding) ---
    // Each `uniform sampler2D` the linked shaders declare: its name, its SPIR-V binding (the
    // HAL bindTexture binding), and the texture UNIT glUniform1i(loc, unit) selected (default
    // 0 per GL). glGetUniformLocation returns SAMPLER_LOCATION_FLAG|index for a sampler name;
    // glUniform1i on that location sets `unit`. At draw, the texture bound to `unit` on
    // GL_TEXTURE_2D is bound to the HAL at `binding`.
    samplers: std.ArrayListUnmanaged(SamplerUniform) = .empty,

    // --- Named uniform blocks (GLES3 UBOs) -----------------------------------
    // Each `layout(std140) uniform Blk { ... };` the linked shaders declare. glGetUniformBlockIndex
    // returns the block's index here. glUniformBlockBinding rebinds its `binding_point`.
    // glBindBufferBase(GL_UNIFORM_BUFFER, point, buf) then routes a user buffer to it. At draw the
    // block's stage (VS/FS) UBO slot is fed the buffer bound at `binding_point` instead of the
    // program's glUniform* storage.
    uniform_blocks: std.ArrayListUnmanaged(ProgUniformBlock) = .empty,

    // --- Transform feedback (GLES3) ------------------------------------------
    // The VS output varyings (name -> location + components), copied from the VS at link time.
    // glTransformFeedbackVaryings resolves each captured name to a VS output location here so the
    // transform-feedback path knows which output slots to capture.
    vs_outputs: std.ArrayListUnmanaged(prism.glsl.OutputMember) = .empty,
    // The capture list set by glTransformFeedbackVaryings (owned varying-name copies), in capture
    // order. Per spec this is recorded before glLinkProgram. The capture resolves the names against
    // vs_outputs at draw time. Empty = transform feedback not configured for this program.
    tf_varyings: std.ArrayListUnmanaged([]u8) = .empty,
    // GL_INTERLEAVED_ATTRIBS (all varyings in one buffer, the minimal target) or GL_SEPARATE_ATTRIBS.
    tf_buffer_mode: GLenum = GL_INTERLEAVED_ATTRIBS,
};

/// A linked program's named uniform block: name (owned), the shader stage that declares it
/// (its per-stage HAL UBO slot: VS -> 0, FS -> 1), the current GL binding point (default from
/// `layout(binding=N)` / declaration index, retargeted by glUniformBlockBinding), and the
/// std140 block size (GL_UNIFORM_BLOCK_DATA_SIZE).
const ProgUniformBlock = struct {
    name: []u8,
    stage: prism.hal.ShaderStage,
    binding_point: GLuint,
    data_size: u32,
    /// Absolute byte offset of the block's first member in the stage's default uniform block
    /// (its members lower as contiguous default-block uniforms). Introspection resolves a
    /// member's name/GL-type by matching `byte_offset + member.tight_offset` against the
    /// program's default-block uniform list (which carries the name + float_count + mat_dim).
    byte_offset: u32 = 0,
    /// The std140<->tight repack table for this block's members (one entry per member; owned,
    /// duped at link from the shader). resolveNamedStageUbo uses it to gather a std140 user buffer
    /// into the tight layout the shader reads (an identity copy for all-16-byte members). Empty
    /// only for a block with no members.
    members: []const prism.glsl.UniformBlockMember = &.{},
};

/// A linked program's sampler uniform: name (owned), SPIR-V/HAL binding, and the texture
/// unit it currently samples (set by glUniform1i, default 0).
const SamplerUniform = struct {
    name: []u8,
    binding: u32,
    unit: GLint = 0,
    // true for a `samplerCube` uniform: at draw it resolves the texture bound to
    // GL_TEXTURE_CUBE_MAP on its unit (not GL_TEXTURE_2D).
    cube: bool = false,
    // true for a `sampler3D` uniform: resolves GL_TEXTURE_3D on its unit.
    tex3d: bool = false,
    // true for a `sampler2DArray` uniform: resolves GL_TEXTURE_2D_ARRAY on its unit.
    tex2darray: bool = false,
};

/// A default-block uniform member as the linked program tracks it: one entry per unique name
/// (its glUniform* location is its index in `Program.uniforms`), with the std-layout byte offset
/// it occupies in each stage that declares it (-1 = the stage does not). The VS and FS lay their
/// blocks out independently, so a member shared by both can sit at different offsets in each.
/// glUniform* writes the value into every stage's buffer at that stage's offset.
const ProgUniform = struct {
    name: []u8,
    vs_offset: i32 = -1,
    fs_offset: i32 = -1,
    float_count: u32,
    mat_dim: u32,
    array_len: u32,
};

/// glGetUniformLocation returns a sampler uniform's location with this high bit set so
/// glUniform1i can distinguish it from a default-block (UBO) uniform's member index. The
/// low bits hold the sampler's index in `Program.samplers`.
const SAMPLER_LOCATION_FLAG: GLint = @bitCast(@as(u32, 0x4000_0000));

/// A GLES vertex array object (glGenVertexArrays, OES_vertex_array_object / GLES3). Captures the
/// whole vertex-attribute array state (the per-attribute pointers/enables/divisors) plus the bound
/// GL_ELEMENT_ARRAY_BUFFER, so binding a VAO swaps that state in one call. The currently-bound VAO's
/// state lives in the live globals (`attribs` + `bound_element_buffer`). Bind saves/restores them.
const VertexArray = struct {
    id: GLuint,
    attribs: [MAX_ATTRIBS]AttribArray = [_]AttribArray{.{}} ** MAX_ATTRIBS,
    element_buffer: GLuint = 0,
    bound_once: bool = false,
};

var buffers: std.ArrayListUnmanaged(*Buffer) = .empty;
var shaders: std.ArrayListUnmanaged(*Shader) = .empty;
var programs: std.ArrayListUnmanaged(*Program) = .empty;
var textures: std.ArrayListUnmanaged(*Texture) = .empty;
var framebuffers: std.ArrayListUnmanaged(*Framebuffer) = .empty;
var renderbuffers: std.ArrayListUnmanaged(*Renderbuffer) = .empty;
var vertex_arrays: std.ArrayListUnmanaged(*VertexArray) = .empty;

/// A GLES3 fence sync object (glFenceSync). Prism submits synchronously, so by the time the
/// app gets the GLsync back the commands have already completed. It is created already-signaled.
const Sync = struct { condition: GLenum, flags: u32 };
var syncs: std.ArrayListUnmanaged(*Sync) = .empty;

/// An occlusion query object (glGenQueries). Records the driver's written-sample counter at
/// glBeginQuery. glEndQuery stores the delta as the result. `conservative` means the driver
/// cannot count (e.g. the GPU path), so the query reports "passed" (never wrongly culls).
const Query = struct { id: GLuint, target: GLenum = 0, start: u64 = 0, conservative: bool = false, result: u64 = 0, has_result: bool = false, used: bool = false };
var queries: std.ArrayListUnmanaged(*Query) = .empty;
threadlocal var active_occlusion_query: GLuint = 0;
fn findQuery(id: GLuint) ?*Query {
    for (queries.items) |q| if (q.id == id) return q;
    return null;
}
fn findSync(sync: ?*anyopaque) ?*Sync {
    const p = sync orelse return null;
    const s: *Sync = @ptrCast(@alignCast(p));
    for (syncs.items) |x| if (x == s) return x;
    return null;
}

/// A GLES3 sampler object (glGenSamplers). Holds sampler state (filter / wrap / LOD / aniso)
/// independent of any texture. When bound to a texture unit via glBindSampler it overrides the
/// texture's own sampler state for draws sampling that unit, so one texture can be sampled with
/// different filtering on different units. Defaults match the GL sampler-object defaults.
const Sampler = struct {
    id: GLuint,
    bound_once: bool = false,
    min_filter: prism.hal.Filter = .nearest, // GL default GL_NEAREST_MIPMAP_LINEAR -> base NEAREST
    mag_filter: prism.hal.Filter = .linear,
    mip_filter: prism.hal.MipFilter = .linear, // ... + mip LINEAR (applies only if the texture has mips)
    wrap_s: prism.hal.AddressMode = .repeat,
    wrap_t: prism.hal.AddressMode = .repeat,
    max_anisotropy: f32 = 1,
    lod_bias: f32 = 0, // GL_TEXTURE_LOD_BIAS
    min_lod: f32 = -1000, // GL_TEXTURE_MIN_LOD
    max_lod: f32 = 1000, // GL_TEXTURE_MAX_LOD
    compare_mode: GLenum = GL_NONE, // GL_TEXTURE_COMPARE_MODE (sampler2DShadow)
    compare_func: GLenum = GL_LEQUAL, // GL_TEXTURE_COMPARE_FUNC
};
var samplers_list: std.ArrayListUnmanaged(*Sampler) = .empty;
fn findSampler(id: GLuint) ?*Sampler {
    for (samplers_list.items) |s| if (s.id == id) return s;
    return null;
}
/// The sampler object bound to each texture unit (0 = none -> the texture's own sampler state).
threadlocal var bound_sampler: [MAX_TEXTURE_UNITS]GLuint = [_]GLuint{0} ** MAX_TEXTURE_UNITS;

var next_id: GLuint = 1; // 0 is "no object" in GL. ids are dense+monotonic.

/// The VAO bound by glBindVertexArray (0 = the default VAO, whose state is the plain globals). The
/// default VAO's state when a non-0 VAO is current is stashed in `default_vao`.
threadlocal var bound_vao: GLuint = 0;
threadlocal var default_vao: struct { attribs: [MAX_ATTRIBS]AttribArray, element_buffer: GLuint } = .{ .attribs = [_]AttribArray{.{}} ** MAX_ATTRIBS, .element_buffer = 0 };

fn findVertexArray(id: GLuint) ?*VertexArray {
    for (vertex_arrays.items) |v| if (v.id == id) return v;
    return null;
}

fn allocId() GLuint {
    const id = next_id;
    next_id += 1;
    return id;
}

pub fn findBuffer(id: GLuint) ?*Buffer {
    for (buffers.items) |b| if (b.id == id) return b;
    return null;
}
fn findShader(id: GLuint) ?*Shader {
    for (shaders.items) |s| if (s.id == id) return s;
    return null;
}
fn findProgram(id: GLuint) ?*Program {
    for (programs.items) |p| if (p.id == id) return p;
    return null;
}
pub fn findTexture(id: GLuint) ?*Texture {
    for (textures.items) |t| if (t.id == id) return t;
    return null;
}

// --- Vertex array objects (glGenVertexArrays / glBindVertexArray) ------------

pub fn genVertexArrays(n: GLsizei, out: ?[*]GLuint) void {
    if (n < 0 or out == null) {
        if (n < 0) setError(GL_INVALID_VALUE);
        return;
    }
    obj_lock.lock();
    defer obj_lock.unlock();
    var i: usize = 0;
    while (i < @as(usize, @intCast(n))) : (i += 1) {
        const v = gpa.create(VertexArray) catch {
            setError(GL_INVALID_OPERATION);
            return;
        };
        v.* = .{ .id = allocId() };
        vertex_arrays.append(gpa, v) catch {
            gpa.destroy(v);
            setError(GL_INVALID_OPERATION);
            return;
        };
        out.?[i] = v.id;
    }
}

/// glBindVertexArray: save the current attribute+element-buffer state into the currently-bound
/// VAO, then load the target VAO's state into the live globals. VAO 0 is the default.
pub fn bindVertexArray(id: GLuint) void {
    obj_lock.lock();
    defer obj_lock.unlock();
    // Stash the live state into whatever VAO is currently bound.
    if (bound_vao == 0) {
        default_vao.attribs = attribs;
        default_vao.element_buffer = bound_element_buffer;
    } else if (findVertexArray(bound_vao)) |cur| {
        cur.attribs = attribs;
        cur.element_buffer = bound_element_buffer;
    }
    // Load the target VAO's state.
    if (id == 0) {
        attribs = default_vao.attribs;
        bound_element_buffer = default_vao.element_buffer;
    } else if (findVertexArray(id)) |v| {
        v.bound_once = true;
        attribs = v.attribs;
        bound_element_buffer = v.element_buffer;
    } else {
        setError(GL_INVALID_OPERATION); // a name never returned by glGenVertexArrays
        return;
    }
    bound_vao = id;
}

pub fn deleteVertexArrays(n: GLsizei, ids: ?[*]const GLuint) void {
    if (n < 0 or ids == null) {
        if (n < 0) setError(GL_INVALID_VALUE);
        return;
    }
    obj_lock.lock();
    defer obj_lock.unlock();
    var i: usize = 0;
    while (i < @as(usize, @intCast(n))) : (i += 1) {
        const id = ids.?[i];
        if (id == 0) continue;
        // Deleting the bound VAO reverts the binding to the default VAO (GL spec).
        if (bound_vao == id) {
            attribs = default_vao.attribs;
            bound_element_buffer = default_vao.element_buffer;
            bound_vao = 0;
        }
        for (vertex_arrays.items, 0..) |v, idx| {
            if (v.id == id) {
                gpa.destroy(v);
                _ = vertex_arrays.swapRemove(idx);
                break;
            }
        }
    }
}

pub fn isVertexArray(id: GLuint) GLboolean {
    if (id == 0) return GL_FALSE;
    obj_lock.lock();
    defer obj_lock.unlock();
    const v = findVertexArray(id) orelse return GL_FALSE;
    return if (v.bound_once) GL_TRUE else GL_FALSE;
}

// --- Fence sync objects (glFenceSync / glClientWaitSync, GLES3) ---------------

pub fn fenceSync(condition: GLenum, flags: u32) ?*anyopaque {
    flushBatch(); // the fence must represent all prior draws
    if (condition != GL_SYNC_GPU_COMMANDS_COMPLETE) {
        setError(GL_INVALID_ENUM);
        return null;
    }
    if (flags != 0) {
        setError(GL_INVALID_VALUE);
        return null;
    }
    obj_lock.lock();
    defer obj_lock.unlock();
    const s = gpa.create(Sync) catch {
        setError(GL_INVALID_OPERATION);
        return null;
    };
    s.* = .{ .condition = condition, .flags = flags };
    syncs.append(gpa, s) catch {
        gpa.destroy(s);
        setError(GL_INVALID_OPERATION);
        return null;
    };
    return @ptrCast(s);
}

/// glClientWaitSync: block until the sync signals. Prism's work is already done, so a valid sync
/// always reports GL_ALREADY_SIGNALED (an invalid sync -> GL_WAIT_FAILED + GL_INVALID_VALUE).
pub fn clientWaitSync(sync: ?*anyopaque, flags: u32, timeout: u64) GLenum {
    flushBatch();
    _ = flags;
    _ = timeout;
    obj_lock.lock();
    defer obj_lock.unlock();
    if (findSync(sync) == null) {
        setError(GL_INVALID_VALUE);
        return GL_WAIT_FAILED;
    }
    return GL_ALREADY_SIGNALED;
}

/// glWaitSync: a server-side wait (glFlush-and-return). No-op on Prism (already complete).
pub fn waitSync(sync: ?*anyopaque, flags: u32, timeout: u64) void {
    _ = flags;
    _ = timeout;
    obj_lock.lock();
    defer obj_lock.unlock();
    if (findSync(sync) == null) setError(GL_INVALID_VALUE);
}

pub fn deleteSync(sync: ?*anyopaque) void {
    obj_lock.lock();
    defer obj_lock.unlock();
    const s = findSync(sync) orelse {
        if (sync != null) setError(GL_INVALID_VALUE);
        return;
    };
    for (syncs.items, 0..) |x, i| {
        if (x == s) {
            gpa.destroy(x);
            _ = syncs.swapRemove(i);
            break;
        }
    }
}

pub fn isSync(sync: ?*anyopaque) GLboolean {
    obj_lock.lock();
    defer obj_lock.unlock();
    return if (findSync(sync) != null) GL_TRUE else GL_FALSE;
}

// --- Occlusion query objects (glBeginQuery / glGetQueryObjectuiv, GLES3) ------

pub fn genQueries(n: GLsizei, out: ?[*]GLuint) void {
    if (n < 0 or out == null) {
        if (n < 0) setError(GL_INVALID_VALUE);
        return;
    }
    obj_lock.lock();
    defer obj_lock.unlock();
    var i: usize = 0;
    while (i < @as(usize, @intCast(n))) : (i += 1) {
        const q = gpa.create(Query) catch {
            setError(GL_INVALID_OPERATION);
            return;
        };
        q.* = .{ .id = allocId() };
        queries.append(gpa, q) catch {
            gpa.destroy(q);
            setError(GL_INVALID_OPERATION);
            return;
        };
        out.?[i] = q.id;
    }
}

pub fn deleteQueries(n: GLsizei, ids: ?[*]const GLuint) void {
    if (n < 0 or ids == null) {
        if (n < 0) setError(GL_INVALID_VALUE);
        return;
    }
    obj_lock.lock();
    defer obj_lock.unlock();
    var i: usize = 0;
    while (i < @as(usize, @intCast(n))) : (i += 1) {
        const id = ids.?[i];
        if (id == 0) continue;
        if (active_occlusion_query == id) active_occlusion_query = 0;
        for (queries.items, 0..) |q, idx| {
            if (q.id == id) {
                gpa.destroy(q);
                _ = queries.swapRemove(idx);
                break;
            }
        }
    }
}

pub fn isQuery(id: GLuint) GLboolean {
    if (id == 0) return GL_FALSE;
    obj_lock.lock();
    defer obj_lock.unlock();
    const q = findQuery(id) orelse return GL_FALSE;
    return if (q.used) GL_TRUE else GL_FALSE;
}

/// glBeginQuery(GL_ANY_SAMPLES_PASSED[_CONSERVATIVE], id): start counting passing samples. Snapshots
/// the driver's written-sample counter. A driver that can't count marks the query conservative.
pub fn beginQuery(target: GLenum, id: GLuint) void {
    flushBatch(); // the query's start baseline must exclude prior batched draws
    if (target != GL_ANY_SAMPLES_PASSED and target != GL_ANY_SAMPLES_PASSED_CONSERVATIVE) {
        setError(GL_INVALID_ENUM);
        return;
    }
    if (id == 0 or active_occlusion_query != 0) {
        setError(GL_INVALID_OPERATION); // no name 0, and one query per target at a time
        return;
    }
    const ctx = state.currentContext() orelse {
        setError(GL_INVALID_OPERATION);
        return;
    };
    obj_lock.lock();
    defer obj_lock.unlock();
    const q = findQuery(id) orelse {
        setError(GL_INVALID_OPERATION);
        return;
    };
    q.target = target;
    q.has_result = false;
    q.used = true;
    if (ctx.device().occlusionCounter()) |c| {
        q.start = c;
        q.conservative = false;
    } else {
        q.conservative = true; // GPU path: cannot count -> report "passed"
    }
    active_occlusion_query = id;
}

/// glEndQuery: finalize the active query's result (the counter delta). Draws are already executed
/// since the software submit is synchronous.
pub fn endQuery(target: GLenum) void {
    flushBatch(); // the draws inside the query span must be counted
    if (target != GL_ANY_SAMPLES_PASSED and target != GL_ANY_SAMPLES_PASSED_CONSERVATIVE) {
        setError(GL_INVALID_ENUM);
        return;
    }
    if (active_occlusion_query == 0) {
        setError(GL_INVALID_OPERATION);
        return;
    }
    const ctx = state.currentContext() orelse {
        setError(GL_INVALID_OPERATION);
        return;
    };
    obj_lock.lock();
    defer obj_lock.unlock();
    const q = findQuery(active_occlusion_query) orelse {
        active_occlusion_query = 0;
        return;
    };
    if (q.conservative) {
        q.result = 1;
    } else {
        const c = ctx.device().occlusionCounter() orelse q.start;
        q.result = c -% q.start;
    }
    q.has_result = true;
    active_occlusion_query = 0;
}

/// glGetQueryObjectuiv: GL_QUERY_RESULT is 0/1 for GL_ANY_SAMPLES_PASSED (any sample survived);
/// GL_QUERY_RESULT_AVAILABLE is 1 once the query ended (Prism resolves synchronously).
pub fn getQueryObjectuiv(id: GLuint, pname: GLenum, params: ?*u32) void {
    const p = params orelse return;
    obj_lock.lock();
    defer obj_lock.unlock();
    const q = findQuery(id) orelse {
        setError(GL_INVALID_OPERATION);
        return;
    };
    switch (pname) {
        GL_QUERY_RESULT => p.* = if (q.result > 0) 1 else 0,
        GL_QUERY_RESULT_AVAILABLE => p.* = if (q.has_result) 1 else 0,
        else => setError(GL_INVALID_ENUM),
    }
}

/// glGetQueryObjectiv: the signed-int form of glGetQueryObjectuiv.
pub fn getQueryObjectiv(id: GLuint, pname: GLenum, params: ?*GLint) void {
    const p = params orelse return;
    var u: u32 = 0;
    getQueryObjectuiv(id, pname, &u);
    p.* = @intCast(u);
}

/// glGetSynciv: query a sync object's properties. Prism's sync is always signaled.
pub fn getSynciv(sync: ?*anyopaque, pname: GLenum, buf_size: GLsizei, length: ?*GLsizei, values: ?[*]GLint) void {
    obj_lock.lock();
    defer obj_lock.unlock();
    const s = findSync(sync) orelse {
        setError(GL_INVALID_VALUE);
        return;
    };
    const v: GLint = switch (pname) {
        GL_OBJECT_TYPE => @intCast(GL_SYNC_FENCE),
        GL_SYNC_CONDITION => @intCast(s.condition),
        GL_SYNC_FLAGS => @intCast(s.flags),
        GL_SYNC_STATUS => @intCast(GL_SIGNALED), // synchronous submit -> always signaled
        else => {
            setError(GL_INVALID_ENUM);
            return;
        },
    };
    if (buf_size >= 1) {
        if (values) |vv| vv[0] = v;
        if (length) |l| l.* = 1;
    } else if (length) |l| l.* = 0;
}
fn findFramebuffer(id: GLuint) ?*Framebuffer {
    for (framebuffers.items) |f| if (f.id == id) return f;
    return null;
}
fn findRenderbuffer(id: GLuint) ?*Renderbuffer {
    for (renderbuffers.items) |r| if (r.id == id) return r;
    return null;
}

// --- Per-thread GLES2 binding state ----------------------------------------

/// One vertex-attribute array slot (glVertexAttribPointer + glEnableVertexAttribArray).
/// `size` = component count (1..4), `gl_type` = GL_FLOAT (only floats supported), `stride`
/// = bytes between consecutive vertices (0 = tightly packed), `offset` = byte offset of
/// the first component within the bound array buffer.
const AttribArray = struct {
    enabled: bool = false,
    size: GLint = 4,
    gl_type: GLenum = GL_FLOAT,
    normalized: bool = false,
    stride: GLsizei = 0,
    offset: usize = 0,
    buffer: GLuint = 0, // the GL_ARRAY_BUFFER bound when glVertexAttribPointer was called
    // glVertexAttribDivisor (ANGLE_instanced_arrays / GLES3): 0 = advance per vertex (normal).
    // D>0 = advance once per D instances, so instance i reads array element floor(i/D). The draw
    // emulates this by a per-instance sub-draw that pins each divisor attribute to its element.
    divisor: u32 = 0,
    // The generic (constant) attribute value glVertexAttrib*f set. Used for every vertex when this
    // array is disabled (the GL fixed value). `has_generic` = a setter was called, so a disabled
    // attribute the shader reads is fed this constant instead of undefined.
    generic: [4]f32 = .{ 0, 0, 0, 1 },
    has_generic: bool = false,
    // glVertexAttribIPointer (GLES3 integer vertex attributes): the array feeds an integer VS
    // input (`in ivec4`/`uvec4`, skinning bone indices, packed per-vertex ints). The raw source
    // value is delivered unconverted (never normalized). The shader reads the integer itself,
    // not a 0..1 normalized float. Distinct from glVertexAttribPointer, which produces a float
    // input. Reported by glGetVertexAttribiv(GL_VERTEX_ATTRIB_ARRAY_INTEGER).
    integer: bool = false,
};

/// glVertexAttrib{1,2,3,4}f: set the generic (constant) value for attribute `index`, used when the
/// array is disabled. Missing components default per GL (y,z=0, w=1).
pub fn vertexAttrib4f(index: GLuint, x: GLfloat, y: GLfloat, z: GLfloat, w: GLfloat) void {
    if (index >= MAX_ATTRIBS) {
        setError(GL_INVALID_VALUE);
        return;
    }
    attribs[index].generic = .{ x, y, z, w };
    attribs[index].has_generic = true;
}

/// glVertexAttribDivisor(index, divisor): set how the attribute advances (0 = per vertex).
pub fn vertexAttribDivisor(index: GLuint, divisor: GLuint) void {
    if (index >= MAX_ATTRIBS) {
        setError(GL_INVALID_VALUE);
        return;
    }
    attribs[index].divisor = divisor;
}

const MAX_ATTRIBS = 16;

threadlocal var bound_array_buffer: GLuint = 0;
threadlocal var bound_element_buffer: GLuint = 0;
threadlocal var bound_copy_read_buffer: GLuint = 0; // GL_COPY_READ_BUFFER (glCopyBufferSubData)
threadlocal var bound_copy_write_buffer: GLuint = 0; // GL_COPY_WRITE_BUFFER
threadlocal var bound_uniform_buffer: GLuint = 0; // GL_UNIFORM_BUFFER (glBufferData for a UBO)
/// GLES3 uniform-buffer binding points: the buffer object bound at each point by
/// glBindBufferBase(GL_UNIFORM_BUFFER, point, buffer). A program's named uniform block reads
/// from the buffer at its (glUniformBlockBinding-assigned) point at draw time. Only the low
/// binding points are honored (the software UBO descriptor space is narrow).
pub const MAX_UNIFORM_BUFFER_BINDINGS = 8;
threadlocal var uniform_buffer_bindings: [MAX_UNIFORM_BUFFER_BINDINGS]GLuint = [_]GLuint{0} ** MAX_UNIFORM_BUFFER_BINDINGS;
/// The byte offset into the bound buffer for each binding point (glBindBufferRange). 0 for a
/// glBindBufferBase (whole-buffer) binding. A non-zero offset selects a sub-range (dynamic UBO
/// streaming / packing several blocks in one buffer). The block's data_size bounds the read.
threadlocal var uniform_buffer_offsets: [MAX_UNIFORM_BUFFER_BINDINGS]GLintptr = [_]GLintptr{0} ** MAX_UNIFORM_BUFFER_BINDINGS;

// --- Transform feedback (GLES3) state ---------------------------------------
/// The number of transform-feedback capture binding points tracked (index in glBindBufferBase(
/// GL_TRANSFORM_FEEDBACK_BUFFER, index, buffer)). The minimal interleaved path uses index 0.
pub const MAX_TRANSFORM_FEEDBACK_BUFFERS = 4;
/// The buffer object bound at each transform-feedback binding point (glBindBufferBase). Index 0 is
/// the interleaved capture target.
threadlocal var transform_feedback_bindings: [MAX_TRANSFORM_FEEDBACK_BUFFERS]GLuint = [_]GLuint{0} ** MAX_TRANSFORM_FEEDBACK_BUFFERS;
/// The generic GL_TRANSFORM_FEEDBACK_BUFFER binding (glBindBuffer / glBindBufferBase set it too);
/// glBufferData + glGetBufferSubData(GL_TRANSFORM_FEEDBACK_BUFFER, ...) operate on it.
threadlocal var bound_transform_feedback_buffer: GLuint = 0;
/// Whether a transform-feedback capture span is active (between glBeginTransformFeedback and
/// glEndTransformFeedback). While active, a draw captures the program's VS output varyings.
threadlocal var tf_active: bool = false;
/// The primitive mode passed to glBeginTransformFeedback (GL_POINTS/LINES/TRIANGLES).
threadlocal var tf_primitive_mode: GLenum = 0;
/// The running byte write cursor into each capture buffer, advancing across draws within a span.
/// GL_INTERLEAVED_ATTRIBS uses only index 0 (all varyings tightly packed into binding 0);
/// GL_SEPARATE_ATTRIBS uses index i for varying i (its own buffer at binding point i).
threadlocal var tf_write_offsets: [MAX_TRANSFORM_FEEDBACK_BUFFERS]usize = [_]usize{0} ** MAX_TRANSFORM_FEEDBACK_BUFFERS;
/// Whether the active transform-feedback span is paused (glPauseTransformFeedback). While paused
/// a draw does not capture. The write cursors hold so glResumeTransformFeedback appends after them.
threadlocal var tf_paused: bool = false;

threadlocal var current_program: GLuint = 0;
threadlocal var attribs: [MAX_ATTRIBS]AttribArray = [_]AttribArray{.{}} ** MAX_ATTRIBS;
/// The framebuffer bound to GL_DRAW_FRAMEBUFFER (glBindFramebuffer / GL_FRAMEBUFFER). 0 = the
/// default framebuffer (the window/pbuffer backbuffer). Non-zero redirects clears/draws to the
/// FBO's attachments. `bound_read_framebuffer` is the GLES3 READ target (glReadPixels / the source
/// of glBlitFramebuffer). glBindFramebuffer(GL_FRAMEBUFFER) sets BOTH, so GLES2 read==draw.
threadlocal var bound_framebuffer: GLuint = 0;
threadlocal var bound_read_framebuffer: GLuint = 0;
/// The renderbuffer bound to GL_RENDERBUFFER (glBindRenderbuffer). glRenderbufferStorage
/// targets it.
threadlocal var bound_renderbuffer: GLuint = 0;

// --- Per-thread texture binding state ---------------------------------------
// The active texture unit (glActiveTexture, default GL_TEXTURE0) and the GL_TEXTURE_2D
// texture id bound to each unit (glBindTexture). At draw, a sampler uniform's selected
// unit -> the texture bound here -> the HAL combined-image-sampler binding.
threadlocal var active_texture_unit: u32 = 0;
threadlocal var bound_texture_2d: [MAX_TEXTURE_UNITS]GLuint = [_]GLuint{0} ** MAX_TEXTURE_UNITS;
// The GL_TEXTURE_CUBE_MAP texture id bound to each unit (a separate binding point from 2D,
// per the GL spec). A `samplerCube` uniform samples the cube bound here on its selected unit.
threadlocal var bound_texture_cube: [MAX_TEXTURE_UNITS]GLuint = [_]GLuint{0} ** MAX_TEXTURE_UNITS;
// The GL_TEXTURE_3D texture id bound to each unit. A `sampler3D` uniform samples it.
threadlocal var bound_texture_3d: [MAX_TEXTURE_UNITS]GLuint = [_]GLuint{0} ** MAX_TEXTURE_UNITS;
// The GL_TEXTURE_2D_ARRAY texture id bound to each unit. A `sampler2DArray` uniform samples it.
threadlocal var bound_texture_2darray: [MAX_TEXTURE_UNITS]GLuint = [_]GLuint{0} ** MAX_TEXTURE_UNITS;
// glPixelStorei(GL_UNPACK_ALIGNMENT): row alignment for glTexImage2D pixel data (1/2/4/8).
threadlocal var unpack_alignment: GLint = 4;
// glPixelStorei(GL_UNPACK_ROW_LENGTH / SKIP_PIXELS / SKIP_ROWS): let a texture upload read a
// sub-rectangle of a wider source buffer. row_length is the source row stride in PIXELS (0 = the
// upload width), skip_pixels/rows offset the source start. Used to upload one glyph / atlas tile
// out of a packed source without repacking. GL defaults: all 0.
threadlocal var unpack_row_length: GLint = 0;
threadlocal var unpack_skip_pixels: GLint = 0;
threadlocal var unpack_skip_rows: GLint = 0;
// glPixelStorei(GL_UNPACK_IMAGE_HEIGHT / SKIP_IMAGES): the 3D/array analog of row_length/skip_rows.
// Lets a glTexImage3D / glTexSubImage3D upload read a sub-volume of a wider source. image_height is
// the source rows per slice (0 = the upload height), skip_images the number of leading source slices
// to skip. Upload one sub-brick of a volume / a range of array layers without repacking. Defaults 0.
threadlocal var unpack_image_height: GLint = 0;
threadlocal var unpack_skip_images: GLint = 0;
// glPixelStorei(GL_PACK_ALIGNMENT): row alignment for glReadPixels DESTINATION rows (1/2/4/8).
threadlocal var pack_alignment: GLint = 4;
// glPixelStorei(GL_PACK_ROW_LENGTH / SKIP_PIXELS / SKIP_ROWS): the readback mirror of the unpack
// versions. Writes a read-back rectangle into a sub-rectangle of a wider destination buffer.
// row_length is the destination row stride in PIXELS (0 = the read width), skip_pixels/rows offset
// the destination start. Read one region straight into a packed CPU atlas. GL defaults: all 0.
threadlocal var pack_row_length: GLint = 0;
threadlocal var pack_skip_pixels: GLint = 0;
threadlocal var pack_skip_rows: GLint = 0;

// --- Buffer objects ---------------------------------------------------------

pub fn genBuffers(n: GLsizei, out: ?[*]GLuint) void {
    if (n < 0 or out == null) {
        if (n < 0) setError(GL_INVALID_VALUE);
        return;
    }
    obj_lock.lock();
    defer obj_lock.unlock();
    var i: usize = 0;
    while (i < @as(usize, @intCast(n))) : (i += 1) {
        const b = gpa.create(Buffer) catch {
            setError(GL_INVALID_OPERATION);
            return;
        };
        b.* = .{ .id = allocId() };
        buffers.append(gpa, b) catch {
            gpa.destroy(b);
            setError(GL_INVALID_OPERATION);
            return;
        };
        out.?[i] = b.id;
    }
}

pub fn bindBuffer(target: GLenum, id: GLuint) void {
    switch (target) {
        GL_ARRAY_BUFFER => bound_array_buffer = id,
        GL_ELEMENT_ARRAY_BUFFER => bound_element_buffer = id,
        GL_COPY_READ_BUFFER => bound_copy_read_buffer = id,
        GL_COPY_WRITE_BUFFER => bound_copy_write_buffer = id,
        GL_UNIFORM_BUFFER => bound_uniform_buffer = id,
        GL_TRANSFORM_FEEDBACK_BUFFER => bound_transform_feedback_buffer = id,
        else => setError(GL_INVALID_ENUM),
    }
}

/// The buffer id bound to a generic buffer `target` (0 = none / unknown target).
fn boundBufferForTarget(target: GLenum) GLuint {
    return switch (target) {
        GL_ARRAY_BUFFER => bound_array_buffer,
        GL_ELEMENT_ARRAY_BUFFER => bound_element_buffer,
        GL_COPY_READ_BUFFER => bound_copy_read_buffer,
        GL_COPY_WRITE_BUFFER => bound_copy_write_buffer,
        GL_UNIFORM_BUFFER => bound_uniform_buffer,
        GL_TRANSFORM_FEEDBACK_BUFFER => bound_transform_feedback_buffer,
        else => 0,
    };
}

/// glGetBufferSubData(target, offset, size, data): read `size` bytes from the bound buffer at
/// `offset` into `data` (the readback counterpart of glBufferSubData; GL_EXT/OES + GLES3).
pub fn getBufferSubData(target: GLenum, offset: isize, size: isize, data: ?*anyopaque) void {
    flushBatch(); // the buffer may have been written by pending transform-feedback/draws
    if (offset < 0 or size < 0) {
        setError(GL_INVALID_VALUE);
        return;
    }
    if (target != GL_ARRAY_BUFFER and target != GL_ELEMENT_ARRAY_BUFFER and target != GL_COPY_READ_BUFFER and target != GL_COPY_WRITE_BUFFER and target != GL_UNIFORM_BUFFER and target != GL_TRANSFORM_FEEDBACK_BUFFER) {
        setError(GL_INVALID_ENUM);
        return;
    }
    const d = data orelse return;
    const id = boundBufferForTarget(target);
    if (id == 0) {
        setError(GL_INVALID_OPERATION);
        return;
    }
    obj_lock.lock();
    defer obj_lock.unlock();
    const b = findBuffer(id) orelse {
        setError(GL_INVALID_OPERATION);
        return;
    };
    const o: usize = @intCast(offset);
    const n: usize = @intCast(size);
    if (o + n > b.bytes.items.len) {
        setError(GL_INVALID_VALUE);
        return;
    }
    @memcpy(@as([*]u8, @ptrCast(d))[0..n], b.bytes.items[o .. o + n]);
}

/// glCopyBufferSubData(readTarget, writeTarget, readOffset, writeOffset, size): copy `size` bytes
/// between two bound buffers (GL_COPY_READ/WRITE_BUFFER avoid disturbing the vertex/index bindings).
pub fn copyBufferSubData(read_target: GLenum, write_target: GLenum, read_offset: isize, write_offset: isize, size: isize) void {
    if (read_offset < 0 or write_offset < 0 or size < 0) {
        setError(GL_INVALID_VALUE);
        return;
    }
    const rid = boundBufferForTarget(read_target);
    const wid = boundBufferForTarget(write_target);
    if (rid == 0 or wid == 0) {
        setError(GL_INVALID_OPERATION); // an unknown target -> 0, or no buffer bound
        return;
    }
    obj_lock.lock();
    defer obj_lock.unlock();
    const rb = findBuffer(rid) orelse {
        setError(GL_INVALID_OPERATION);
        return;
    };
    const wb = findBuffer(wid) orelse {
        setError(GL_INVALID_OPERATION);
        return;
    };
    const ro: usize = @intCast(read_offset);
    const wo: usize = @intCast(write_offset);
    const n: usize = @intCast(size);
    if (ro + n > rb.bytes.items.len or wo + n > wb.bytes.items.len) {
        setError(GL_INVALID_VALUE); // out of a buffer's range
        return;
    }
    // Same-buffer overlapping copy is GL_INVALID_VALUE. Distinct buffers copy directly.
    if (rid == wid and !(wo + n <= ro or ro + n <= wo)) {
        setError(GL_INVALID_VALUE);
        return;
    }
    std.mem.copyForwards(u8, wb.bytes.items[wo .. wo + n], rb.bytes.items[ro .. ro + n]);
    // The write buffer's CPU mirror changed. Drop its HAL resource so the next draw re-uploads.
    if (wb.hal) |h| {
        if (wb.hal_dev) |d| d.destroyResource(h);
        wb.hal = null;
    }
}

pub fn bufferData(target: GLenum, size: GLsizeiptr, data: ?*const anyopaque, usage: GLenum) void {
    // GL_STATIC/DYNAMIC/STREAM_DRAW are accepted hints. The software path ignores them for storage
    // but records the hint so glGetBufferParameteriv(GL_BUFFER_USAGE) reports it faithfully.
    if (target != GL_ARRAY_BUFFER and target != GL_ELEMENT_ARRAY_BUFFER and target != GL_COPY_READ_BUFFER and target != GL_COPY_WRITE_BUFFER and target != GL_UNIFORM_BUFFER and target != GL_TRANSFORM_FEEDBACK_BUFFER) {
        setError(GL_INVALID_ENUM);
        return;
    }
    const id = boundBufferForTarget(target);
    if (id == 0) {
        setError(GL_INVALID_OPERATION); // no buffer bound to the target
        return;
    }
    if (size < 0) {
        setError(GL_INVALID_VALUE);
        return;
    }
    obj_lock.lock();
    defer obj_lock.unlock();
    const b = findBuffer(id) orelse {
        setError(GL_INVALID_OPERATION);
        return;
    };
    b.usage = usage;
    const n: usize = @intCast(size);
    b.bytes.clearRetainingCapacity();
    b.bytes.resize(gpa, n) catch {
        setError(GL_INVALID_OPERATION);
        return;
    };
    if (data) |d| {
        const src = @as([*]const u8, @ptrCast(d))[0..n];
        @memcpy(b.bytes.items, src);
    } else {
        @memset(b.bytes.items, 0);
    }
    // Invalidate any previously-built HAL resource (rebuilt at draw time).
    if (b.hal) |h| {
        if (b.hal_dev) |dev| dev.destroyResource(h);
        b.hal = null;
        b.hal_dev = null;
    }
    if (b.ubo_hal) |h| {
        if (b.ubo_hal_dev) |dev| dev.destroyResource(h);
        b.ubo_hal = null;
        b.ubo_hal_dev = null;
    }
}

pub fn deleteBuffers(n: GLsizei, ids: ?[*]const GLuint) void {
    flushBatch(); // a pending draw may reference these buffers (VBO/UBO)
    if (n < 0 or ids == null) {
        if (n < 0) setError(GL_INVALID_VALUE);
        return;
    }
    obj_lock.lock();
    defer obj_lock.unlock();
    var i: usize = 0;
    while (i < @as(usize, @intCast(n))) : (i += 1) {
        const id = ids.?[i];
        if (id == 0) continue;
        for (buffers.items, 0..) |b, idx| {
            if (b.id == id) {
                if (b.hal) |h| if (b.hal_dev) |dev| dev.destroyResource(h);
                if (b.ubo_hal) |h| if (b.ubo_hal_dev) |dev| dev.destroyResource(h);
                b.bytes.deinit(gpa);
                gpa.destroy(b);
                _ = buffers.swapRemove(idx);
                break;
            }
        }
    }
}

// --- Texture objects --------------------------------------------------------

pub fn pixelStorei(pname: GLenum, param: GLint) void {
    switch (pname) {
        GL_UNPACK_ALIGNMENT, GL_PACK_ALIGNMENT => {
            // GL accepts 1, 2, 4, 8.
            if (param != 1 and param != 2 and param != 4 and param != 8) {
                setError(GL_INVALID_VALUE);
                return;
            }
            if (pname == GL_UNPACK_ALIGNMENT) unpack_alignment = param else pack_alignment = param;
        },
        // GL_UNPACK_ROW_LENGTH / SKIP_PIXELS / SKIP_ROWS: sub-rectangle upload from a wider source.
        // Negative is GL_INVALID_VALUE.
        GL_UNPACK_ROW_LENGTH, GL_UNPACK_SKIP_PIXELS, GL_UNPACK_SKIP_ROWS => {
            if (param < 0) return setError(GL_INVALID_VALUE);
            switch (pname) {
                GL_UNPACK_ROW_LENGTH => unpack_row_length = param,
                GL_UNPACK_SKIP_PIXELS => unpack_skip_pixels = param,
                else => unpack_skip_rows = param,
            }
        },
        // GL_UNPACK_IMAGE_HEIGHT / SKIP_IMAGES: the 3D/array sub-volume upload params.
        GL_UNPACK_IMAGE_HEIGHT, GL_UNPACK_SKIP_IMAGES => {
            if (param < 0) return setError(GL_INVALID_VALUE);
            if (pname == GL_UNPACK_IMAGE_HEIGHT) unpack_image_height = param else unpack_skip_images = param;
        },
        // GL_PACK_ROW_LENGTH / SKIP_PIXELS / SKIP_ROWS: sub-rectangle readback into a wider dest.
        // Negative is GL_INVALID_VALUE.
        GL_PACK_ROW_LENGTH, GL_PACK_SKIP_PIXELS, GL_PACK_SKIP_ROWS => {
            if (param < 0) return setError(GL_INVALID_VALUE);
            switch (pname) {
                GL_PACK_ROW_LENGTH => pack_row_length = param,
                GL_PACK_SKIP_PIXELS => pack_skip_pixels = param,
                else => pack_skip_rows = param,
            }
        },
        else => setError(GL_INVALID_ENUM),
    }
}

pub fn genTextures(n: GLsizei, out: ?[*]GLuint) void {
    if (n < 0 or out == null) {
        if (n < 0) setError(GL_INVALID_VALUE);
        return;
    }
    obj_lock.lock();
    defer obj_lock.unlock();
    var i: usize = 0;
    while (i < @as(usize, @intCast(n))) : (i += 1) {
        const t = gpa.create(Texture) catch {
            setError(GL_INVALID_OPERATION);
            return;
        };
        t.* = .{ .id = allocId() };
        textures.append(gpa, t) catch {
            gpa.destroy(t);
            setError(GL_INVALID_OPERATION);
            return;
        };
        out.?[i] = t.id;
    }
}

pub fn activeTexture(texture: GLenum) void {
    if (texture < GL_TEXTURE0 or texture >= GL_TEXTURE0 + MAX_TEXTURE_UNITS) {
        setError(GL_INVALID_ENUM);
        return;
    }
    active_texture_unit = texture - GL_TEXTURE0;
}

pub fn bindTexture(target: GLenum, id: GLuint) void {
    if (target != GL_TEXTURE_2D and target != GL_TEXTURE_CUBE_MAP and target != GL_TEXTURE_3D and target != GL_TEXTURE_2D_ARRAY) {
        setError(GL_INVALID_ENUM);
        return;
    }
    if (id != 0) {
        obj_lock.lock();
        const t = findTexture(id);
        if (t) |tex| {
            tex.bound_once = true;
            // First bind marks the texture object's kind.
            if (target == GL_TEXTURE_CUBE_MAP) tex.is_cube = true;
            if (target == GL_TEXTURE_3D) tex.is_3d = true;
            if (target == GL_TEXTURE_2D_ARRAY) tex.is_array = true;
        }
        obj_lock.unlock();
        if (t == null) {
            setError(GL_INVALID_OPERATION);
            return;
        }
    }
    switch (target) {
        GL_TEXTURE_CUBE_MAP => bound_texture_cube[active_texture_unit] = id,
        GL_TEXTURE_3D => bound_texture_3d[active_texture_unit] = id,
        GL_TEXTURE_2D_ARRAY => bound_texture_2darray[active_texture_unit] = id,
        else => bound_texture_2d[active_texture_unit] = id,
    }
}

// --- GLES3 sampler objects --------------------------------------------------

pub fn genSamplers(n: GLsizei, out: ?[*]GLuint) void {
    if (n < 0 or out == null) {
        if (n < 0) setError(GL_INVALID_VALUE);
        return;
    }
    obj_lock.lock();
    defer obj_lock.unlock();
    var i: usize = 0;
    while (i < @as(usize, @intCast(n))) : (i += 1) {
        const s = gpa.create(Sampler) catch {
            setError(GL_INVALID_OPERATION);
            return;
        };
        s.* = .{ .id = next_id };
        next_id += 1;
        samplers_list.append(gpa, s) catch {
            gpa.destroy(s);
            setError(GL_INVALID_OPERATION);
            return;
        };
        out.?[i] = s.id;
    }
}

pub fn deleteSamplers(n: GLsizei, ids: ?[*]const GLuint) void {
    if (n < 0 or ids == null) {
        if (n < 0) setError(GL_INVALID_VALUE);
        return;
    }
    obj_lock.lock();
    defer obj_lock.unlock();
    var i: usize = 0;
    while (i < @as(usize, @intCast(n))) : (i += 1) {
        const id = ids.?[i];
        if (id == 0) continue;
        // A deleted sampler that is bound to a unit reverts that unit to the texture's own state.
        for (&bound_sampler) |*bs| if (bs.* == id) {
            bs.* = 0;
        };
        var j: usize = 0;
        while (j < samplers_list.items.len) : (j += 1) {
            if (samplers_list.items[j].id == id) {
                gpa.destroy(samplers_list.items[j]);
                _ = samplers_list.swapRemove(j);
                break;
            }
        }
    }
}

pub fn isSampler(id: GLuint) GLboolean {
    if (id == 0) return GL_FALSE;
    obj_lock.lock();
    defer obj_lock.unlock();
    const s = findSampler(id) orelse return GL_FALSE;
    return if (s.bound_once) GL_TRUE else GL_FALSE;
}

pub fn bindSampler(unit: GLuint, sampler: GLuint) void {
    if (unit >= MAX_TEXTURE_UNITS) {
        setError(GL_INVALID_VALUE);
        return;
    }
    if (sampler != 0) {
        obj_lock.lock();
        const s = findSampler(sampler);
        if (s) |sm| sm.bound_once = true;
        obj_lock.unlock();
        if (s == null) {
            setError(GL_INVALID_OPERATION);
            return;
        }
    }
    bound_sampler[unit] = sampler;
}

pub fn samplerParameteri(sampler: GLuint, pname: GLenum, param: GLint) void {
    obj_lock.lock();
    defer obj_lock.unlock();
    const s = findSampler(sampler) orelse {
        setError(GL_INVALID_OPERATION);
        return;
    };
    const e: GLenum = @intCast(param);
    switch (pname) {
        GL_TEXTURE_MIN_FILTER => {
            s.min_filter = halFilter(e) orelse return setError(GL_INVALID_ENUM);
            s.mip_filter = mipFilterOf(e);
        },
        GL_TEXTURE_MAG_FILTER => s.mag_filter = halFilter(e) orelse return setError(GL_INVALID_ENUM),
        GL_TEXTURE_WRAP_S => s.wrap_s = halWrap(e) orelse return setError(GL_INVALID_ENUM),
        GL_TEXTURE_WRAP_T => s.wrap_t = halWrap(e) orelse return setError(GL_INVALID_ENUM),
        GL_TEXTURE_WRAP_R => _ = halWrap(e) orelse return setError(GL_INVALID_ENUM), // accepted; 2D sampling ignores R
        GL_TEXTURE_MAX_ANISOTROPY_EXT => {
            if (param < 1) return setError(GL_INVALID_VALUE);
            s.max_anisotropy = @floatFromInt(@min(param, 16));
        },
        GL_TEXTURE_LOD_BIAS => s.lod_bias = @floatFromInt(param), // integer entry; the float entry keeps the fraction
        GL_TEXTURE_MIN_LOD => s.min_lod = @floatFromInt(param),
        GL_TEXTURE_MAX_LOD => s.max_lod = @floatFromInt(param),
        GL_TEXTURE_COMPARE_MODE => {
            if (e != GL_NONE and e != GL_COMPARE_REF_TO_TEXTURE) return setError(GL_INVALID_ENUM);
            s.compare_mode = e;
        },
        GL_TEXTURE_COMPARE_FUNC => {
            if (compareOpFromGl(e) == null) return setError(GL_INVALID_ENUM);
            s.compare_func = e;
        },
        else => setError(GL_INVALID_ENUM),
    }
}

pub fn samplerParameterf(sampler: GLuint, pname: GLenum, param: GLfloat) void {
    // GL_TEXTURE_LOD_BIAS / MIN_LOD / MAX_LOD are genuine floats: store them directly, not int-cast.
    if (pname == GL_TEXTURE_LOD_BIAS or pname == GL_TEXTURE_MIN_LOD or pname == GL_TEXTURE_MAX_LOD) {
        obj_lock.lock();
        defer obj_lock.unlock();
        const s = findSampler(sampler) orelse return setError(GL_INVALID_OPERATION);
        switch (pname) {
            GL_TEXTURE_LOD_BIAS => s.lod_bias = param,
            GL_TEXTURE_MIN_LOD => s.min_lod = param,
            else => s.max_lod = param,
        }
        return;
    }
    samplerParameteri(sampler, pname, @intFromFloat(param));
}

pub fn getSamplerParameteriv(sampler: GLuint, pname: GLenum, out: ?[*]GLint) void {
    const o = out orelse return;
    obj_lock.lock();
    defer obj_lock.unlock();
    const s = findSampler(sampler) orelse {
        setError(GL_INVALID_OPERATION);
        return;
    };
    o[0] = switch (pname) {
        GL_TEXTURE_MIN_FILTER => @intCast(minFilterEnumOf(s.min_filter, s.mip_filter)),
        GL_TEXTURE_MAG_FILTER => @intCast(if (s.mag_filter == .linear) GL_LINEAR else GL_NEAREST),
        GL_TEXTURE_WRAP_S => @intCast(wrapEnumOf(s.wrap_s)),
        GL_TEXTURE_WRAP_T => @intCast(wrapEnumOf(s.wrap_t)),
        GL_TEXTURE_MAX_ANISOTROPY_EXT => @intFromFloat(s.max_anisotropy),
        GL_TEXTURE_COMPARE_MODE => @intCast(s.compare_mode),
        GL_TEXTURE_COMPARE_FUNC => @intCast(s.compare_func),
        else => {
            setError(GL_INVALID_ENUM);
            return;
        },
    };
}

/// The GL_TEXTURE_MIN_FILTER enum for a (base-min, mip) filter pair (the inverse of halFilter +
/// mipFilterOf), for glGetSamplerParameteriv.
fn minFilterEnumOf(base: prism.hal.Filter, mip: prism.hal.MipFilter) GLenum {
    return switch (mip) {
        .none => if (base == .linear) GL_LINEAR else GL_NEAREST,
        .nearest => if (base == .linear) GL_LINEAR_MIPMAP_NEAREST else GL_NEAREST_MIPMAP_NEAREST,
        .linear => if (base == .linear) GL_LINEAR_MIPMAP_LINEAR else GL_NEAREST_MIPMAP_LINEAR,
    };
}
fn wrapEnumOf(w: prism.hal.AddressMode) GLenum {
    return switch (w) {
        .repeat => GL_REPEAT,
        .mirrored_repeat => GL_MIRRORED_REPEAT,
        .clamp_to_edge => GL_CLAMP_TO_EDGE,
    };
}

pub fn isTexture(id: GLuint) GLboolean {
    if (id == 0) return GL_FALSE;
    obj_lock.lock();
    defer obj_lock.unlock();
    const t = findTexture(id) orelse return GL_FALSE;
    // glIsTexture is FALSE until the name has been bound at least once.
    return if (t.bound_once) GL_TRUE else GL_FALSE;
}

pub fn deleteTextures(n: GLsizei, ids: ?[*]const GLuint) void {
    flushBatch(); // a pending draw may sample these textures
    if (n < 0 or ids == null) {
        if (n < 0) setError(GL_INVALID_VALUE);
        return;
    }
    obj_lock.lock();
    defer obj_lock.unlock();
    var i: usize = 0;
    while (i < @as(usize, @intCast(n))) : (i += 1) {
        const id = ids.?[i];
        if (id == 0) continue;
        for (textures.items, 0..) |t, idx| {
            if (t.id == id) {
                if (t.hal) |h| if (t.hal_dev) |dev| dev.destroyResource(h);
                if (t.depth_hal) |dh| if (t.hal_dev) |dev| dev.destroyResource(dh);
                if (t.depth_sampled_hal) |ds| if (t.hal_dev) |dev| dev.destroyResource(ds);
                t.bytes.deinit(gpa);
                gpa.destroy(t);
                _ = textures.swapRemove(idx);
                break;
            }
        }
    }
    // GL: deleting a bound texture resets the binding to 0 for every unit it is bound to.
    for (&bound_texture_2d) |*b| {
        var k: usize = 0;
        while (k < @as(usize, @intCast(n))) : (k += 1) {
            if (b.* == ids.?[k] and b.* != 0) b.* = 0;
        }
    }
}

/// Map a GL filter enum to the HAL in-level filter (the min/mag filter within a mip level).
/// Mipmap modes keep their base 2D filter. The mip-blend mode is separate (mipFilterOf).
fn halFilter(f: GLenum) ?prism.hal.Filter {
    return switch (f) {
        GL_NEAREST, GL_NEAREST_MIPMAP_NEAREST, GL_NEAREST_MIPMAP_LINEAR => .nearest,
        GL_LINEAR, GL_LINEAR_MIPMAP_NEAREST, GL_LINEAR_MIPMAP_LINEAR => .linear,
        else => null,
    };
}

/// Map a GL_TEXTURE_MIN_FILTER enum to the HAL mip-blend mode: the non-mipmap filters select the
/// base level only (none). GL_*_MIPMAP_NEAREST snaps to the nearest level. GL_*_MIPMAP_LINEAR
/// trilinearly blends the two bracketing levels.
fn mipFilterOf(f: GLenum) prism.hal.MipFilter {
    return switch (f) {
        GL_NEAREST_MIPMAP_NEAREST, GL_LINEAR_MIPMAP_NEAREST => .nearest,
        GL_NEAREST_MIPMAP_LINEAR, GL_LINEAR_MIPMAP_LINEAR => .linear,
        else => .none,
    };
}

/// Map a GL wrap enum to the HAL address mode.
fn halWrap(w: GLenum) ?prism.hal.AddressMode {
    return switch (w) {
        GL_REPEAT => .repeat,
        GL_MIRRORED_REPEAT => .mirrored_repeat,
        GL_CLAMP_TO_EDGE => .clamp_to_edge,
        else => null,
    };
}

/// The texture bound to GL_TEXTURE_2D on the active unit (or null / texture 0). Caller
/// holds obj_lock.
fn activeTextureObj() ?*Texture {
    const id = bound_texture_2d[active_texture_unit];
    if (id == 0) return null;
    return findTexture(id);
}

/// The texture bound to GL_TEXTURE_CUBE_MAP on the active unit (or null / texture 0).
fn activeCubeTextureObj() ?*Texture {
    const id = bound_texture_cube[active_texture_unit];
    if (id == 0) return null;
    return findTexture(id);
}

/// The texture bound to GL_TEXTURE_3D on the active unit (or null / texture 0).
fn active3dTextureObj() ?*Texture {
    const id = bound_texture_3d[active_texture_unit];
    if (id == 0) return null;
    return findTexture(id);
}

/// The texture bound to GL_TEXTURE_2D_ARRAY on the active unit (or null / texture 0).
fn active2dArrayTextureObj() ?*Texture {
    const id = bound_texture_2darray[active_texture_unit];
    if (id == 0) return null;
    return findTexture(id);
}

pub fn texParameteri(target: GLenum, pname: GLenum, param: GLint) void {
    if (target != GL_TEXTURE_2D and target != GL_TEXTURE_CUBE_MAP and target != GL_TEXTURE_3D and target != GL_TEXTURE_2D_ARRAY) {
        setError(GL_INVALID_ENUM);
        return;
    }
    obj_lock.lock();
    defer obj_lock.unlock();
    const t = (switch (target) {
        GL_TEXTURE_CUBE_MAP => activeCubeTextureObj(),
        GL_TEXTURE_3D => active3dTextureObj(),
        GL_TEXTURE_2D_ARRAY => active2dArrayTextureObj(),
        else => activeTextureObj(),
    }) orelse {
        setError(GL_INVALID_OPERATION); // no texture bound to the active unit
        return;
    };
    const e: GLenum = @intCast(param);
    switch (pname) {
        GL_TEXTURE_MIN_FILTER => {
            t.min_filter = halFilter(e) orelse {
                setError(GL_INVALID_ENUM);
                return;
            };
            t.mip_filter = mipFilterOf(e); // the GL_*_MIPMAP_* blend mode (none if not a mipmap filter)
        },
        GL_TEXTURE_MAG_FILTER => t.mag_filter = halFilter(e) orelse {
            setError(GL_INVALID_ENUM);
            return;
        },
        GL_TEXTURE_WRAP_S => t.wrap_s = halWrap(e) orelse {
            setError(GL_INVALID_ENUM);
            return;
        },
        GL_TEXTURE_WRAP_T => t.wrap_t = halWrap(e) orelse {
            setError(GL_INVALID_ENUM);
            return;
        },
        // GL_TEXTURE_MAX_ANISOTROPY_EXT: clamp to [1, the advertised 16x limit]. A value < 1 is
        // GL_INVALID_VALUE per the extension.
        GL_TEXTURE_MAX_ANISOTROPY_EXT => {
            if (param < 1) {
                setError(GL_INVALID_VALUE);
                return;
            }
            t.max_anisotropy = @floatFromInt(@min(param, 16));
        },
        // GL_TEXTURE_BASE_LEVEL / GL_TEXTURE_MAX_LEVEL: the mip level range sampling uses. A negative
        // value is GL_INVALID_VALUE. The sampler clamps to the image's real level count.
        GL_TEXTURE_BASE_LEVEL => {
            if (param < 0) return setError(GL_INVALID_VALUE);
            t.base_level = @intCast(param);
        },
        GL_TEXTURE_MAX_LEVEL => {
            if (param < 0) return setError(GL_INVALID_VALUE);
            t.max_level = @intCast(param);
        },
        // GL_TEXTURE_SWIZZLE_R/G/B/A: remap output channel R/G/B/A to a source (RED/GREEN/BLUE/ALPHA/
        // ZERO/ONE). The classic font-atlas idiom is a GL_R8 coverage texture with SWIZZLE {1,1,1,R}.
        GL_TEXTURE_SWIZZLE_R, GL_TEXTURE_SWIZZLE_G, GL_TEXTURE_SWIZZLE_B, GL_TEXTURE_SWIZZLE_A => {
            const sw = swizzleFromEnum(e) orelse return setError(GL_INVALID_ENUM);
            t.swizzle[pname - GL_TEXTURE_SWIZZLE_R] = sw;
        },
        // GL_TEXTURE_LOD_BIAS via the integer entry (glTexParameteri): the fraction is lost, but an
        // integer bias is still valid. glTexParameterf (below) preserves the fraction.
        GL_TEXTURE_LOD_BIAS => t.lod_bias = @floatFromInt(param),
        GL_TEXTURE_MIN_LOD => t.min_lod = @floatFromInt(param), // integer entry; the float entry keeps the fraction
        GL_TEXTURE_MAX_LOD => t.max_lod = @floatFromInt(param),
        // GL_TEXTURE_COMPARE_MODE / _FUNC: sampler2DShadow depth compare.
        GL_TEXTURE_COMPARE_MODE => {
            if (e != GL_NONE and e != GL_COMPARE_REF_TO_TEXTURE) return setError(GL_INVALID_ENUM);
            t.compare_mode = e;
        },
        GL_TEXTURE_COMPARE_FUNC => {
            if (compareOpFromGl(e) == null) return setError(GL_INVALID_ENUM);
            t.compare_func = e;
        },
        else => setError(GL_INVALID_ENUM),
    }
}

/// Map a GL_TEXTURE_SWIZZLE_* value (GL_RED/GREEN/BLUE/ALPHA/ZERO/ONE) to the HAL Swizzle source.
fn swizzleFromEnum(e: GLenum) ?prism.hal.Swizzle {
    return switch (e) {
        GL_RED => .r,
        GL_GREEN => .g,
        GL_BLUE => .b,
        GL_ALPHA => .a,
        GL_ZERO => .zero,
        GL_ONE => .one,
        else => null,
    };
}
fn swizzleToEnum(s: prism.hal.Swizzle) GLenum {
    return switch (s) {
        .r => GL_RED,
        .g => GL_GREEN,
        .b => GL_BLUE,
        .a => GL_ALPHA,
        .zero => GL_ZERO,
        .one => GL_ONE,
    };
}

pub fn texParameterf(target: GLenum, pname: GLenum, param: GLfloat) void {
    // GL_TEXTURE_LOD_BIAS / MIN_LOD / MAX_LOD are genuine floats (e.g. -0.5): store them directly,
    // not via the int cast that would drop the fraction.
    if (pname == GL_TEXTURE_LOD_BIAS or pname == GL_TEXTURE_MIN_LOD or pname == GL_TEXTURE_MAX_LOD) {
        if (target != GL_TEXTURE_2D and target != GL_TEXTURE_CUBE_MAP and target != GL_TEXTURE_3D and target != GL_TEXTURE_2D_ARRAY) return setError(GL_INVALID_ENUM);
        obj_lock.lock();
        defer obj_lock.unlock();
        const t = (switch (target) {
            GL_TEXTURE_CUBE_MAP => activeCubeTextureObj(),
            GL_TEXTURE_3D => active3dTextureObj(),
            GL_TEXTURE_2D_ARRAY => active2dArrayTextureObj(),
            else => activeTextureObj(),
        }) orelse return setError(GL_INVALID_OPERATION);
        switch (pname) {
            GL_TEXTURE_LOD_BIAS => t.lod_bias = param,
            GL_TEXTURE_MIN_LOD => t.min_lod = param,
            else => t.max_lod = param,
        }
        return;
    }
    texParameteri(target, pname, @intFromFloat(param));
}

/// Source bytes per texel for a (format, type) upload, or null if the combination is not
/// a supported one. GL_UNSIGNED_BYTE is 1 byte per channel. The packed 16-bit types are one
/// u16 per texel and only pair with their canonical format.
fn texelBytes(format: GLenum, gl_type: GLenum) ?usize {
    return switch (gl_type) {
        GL_UNSIGNED_BYTE => switch (format) {
            GL_RGBA, GL_BGRA_EXT, GL_SRGB_ALPHA_EXT => 4,
            GL_RGB, GL_SRGB_EXT => 3,
            GL_LUMINANCE_ALPHA => 2,
            GL_LUMINANCE, GL_ALPHA => 1,
            else => null,
        },
        GL_UNSIGNED_SHORT_5_6_5 => if (format == GL_RGB) @as(usize, 2) else null,
        GL_UNSIGNED_SHORT_4_4_4_4, GL_UNSIGNED_SHORT_5_5_5_1 => if (format == GL_RGBA) @as(usize, 2) else null,
        // Float source texels: 2 bytes/channel (half) or 4 (single); RGBA = 4 ch, RGB = 3.
        GL_HALF_FLOAT_OES => switch (format) {
            GL_RGBA => 8,
            GL_RGB => 6,
            else => null,
        },
        GL_FLOAT => switch (format) {
            GL_RGBA => 16,
            GL_RGB => 12,
            else => null,
        },
        else => null,
    };
}

/// The internal HAL texel format for a (internalformat, format, type) upload. sRGB formats tag
/// rgba8_srgb (8-bit sRGB, sampler decodes). Half/single-float sources store rgba16_float or
/// r32g32b32a32_float (HDR). Everything else is the 8-bit rgba8_unorm path.
fn internalTexFormat(format: GLenum, gl_type: GLenum) prism.hal.Format {
    if (format == GL_SRGB_EXT or format == GL_SRGB_ALPHA_EXT) return .rgba8_srgb;
    return switch (gl_type) {
        GL_HALF_FLOAT_OES => .rgba16_float,
        GL_FLOAT => .r32g32b32a32_float,
        else => .rgba8_unorm,
    };
}

// Expand an N-bit channel to 8 bits by replicating the high bits into the low ones (so
// full-scale maps to 0xff, matching GL's normalized conversion).
fn ex5(v: u16) u8 {
    return @intCast((v << 3) | (v >> 2));
}
fn ex6(v: u16) u8 {
    return @intCast((v << 2) | (v >> 4));
}
fn ex4(v: u16) u8 {
    return @intCast(v * 17); // (v<<4)|v
}

/// Decode one source texel of (format, type) to RGBA8 (Prism's internal texture storage the
/// software sampler + nvidia TIC read). Handles the byte formats (LUMINANCE replicates L,
/// ALPHA is (0,0,0,A), RGB is opaque, BGRA swaps B<->R) and the packed 16-bit types
/// (native-endian u16, components MSB-first per the GLES spec).
fn decodeTexel(format: GLenum, gl_type: GLenum, src: []const u8, dst: *[4]u8) void {
    switch (gl_type) {
        GL_UNSIGNED_BYTE => switch (format) {
            // sRGB texels are stored VERBATIM (the sampler applies the sRGB->linear decode);
            // GL_SRGB_EXT is RGB (opaque), GL_SRGB_ALPHA_EXT is RGBA.
            GL_RGBA, GL_SRGB_ALPHA_EXT => dst.* = .{ src[0], src[1], src[2], src[3] },
            GL_BGRA_EXT => dst.* = .{ src[2], src[1], src[0], src[3] }, // B,G,R,A -> R,G,B,A
            GL_RGB, GL_SRGB_EXT => dst.* = .{ src[0], src[1], src[2], 255 },
            GL_LUMINANCE => dst.* = .{ src[0], src[0], src[0], 255 },
            GL_ALPHA => dst.* = .{ 0, 0, 0, src[0] },
            GL_LUMINANCE_ALPHA => dst.* = .{ src[0], src[0], src[0], src[1] },
            else => dst.* = .{ 0, 0, 0, 255 },
        },
        GL_UNSIGNED_SHORT_5_6_5 => {
            const v: u16 = @as(u16, src[0]) | (@as(u16, src[1]) << 8);
            dst.* = .{ ex5((v >> 11) & 0x1f), ex6((v >> 5) & 0x3f), ex5(v & 0x1f), 255 };
        },
        GL_UNSIGNED_SHORT_4_4_4_4 => {
            const v: u16 = @as(u16, src[0]) | (@as(u16, src[1]) << 8);
            dst.* = .{ ex4((v >> 12) & 0xf), ex4((v >> 8) & 0xf), ex4((v >> 4) & 0xf), ex4(v & 0xf) };
        },
        GL_UNSIGNED_SHORT_5_5_5_1 => {
            const v: u16 = @as(u16, src[0]) | (@as(u16, src[1]) << 8);
            dst.* = .{ ex5((v >> 11) & 0x1f), ex5((v >> 6) & 0x1f), ex5((v >> 1) & 0x1f), if (v & 1 == 1) 255 else 0 };
        },
        else => dst.* = .{ 0, 0, 0, 255 },
    }
}

// Quantize an 8-bit channel down to an N-bit packed field (GL's normalized round-to-nearest,
// so full-scale 0xff maps to the N-bit max). The inverse direction of ex4/ex5/ex6.
// Quantize a [0,1] float channel to an N-bit packed field (GL normalized round-to-nearest).
inline fn quantN(v: f32, maxn: u16) u16 {
    return @intFromFloat(@round(std.math.clamp(v, 0.0, 1.0) * @as(f32, @floatFromInt(maxn))));
}
inline fn unorm8f(v: f32) u8 {
    return @intFromFloat(@round(std.math.clamp(v, 0.0, 1.0) * 255.0));
}

/// Encode one RGBA framebuffer texel `src` (already unpacked to floats: 0..1 for an 8-bit source,
/// the true value for a float render target) into the glReadPixels destination `dst` in the
/// (format, gl_type) pair. Unorm/packed outputs clamp to [0,1]. Float outputs preserve
/// the full value (HDR > 1.0 survives from a float RT). `dst` is exactly one destination texel.
fn encodeReadTexel(format: GLenum, gl_type: GLenum, src: *const [4]f32, dst: []u8) void {
    switch (gl_type) {
        GL_UNSIGNED_BYTE => {
            dst[0] = unorm8f(src[0]);
            dst[1] = unorm8f(src[1]);
            dst[2] = unorm8f(src[2]);
            if (format == GL_RGBA) dst[3] = unorm8f(src[3]);
        },
        GL_UNSIGNED_SHORT_5_6_5 => {
            const v: u16 = (quantN(src[0], 31) << 11) | (quantN(src[1], 63) << 5) | quantN(src[2], 31);
            dst[0] = @truncate(v);
            dst[1] = @truncate(v >> 8);
        },
        GL_UNSIGNED_SHORT_4_4_4_4 => {
            const v: u16 = (quantN(src[0], 15) << 12) | (quantN(src[1], 15) << 8) | (quantN(src[2], 15) << 4) | quantN(src[3], 15);
            dst[0] = @truncate(v);
            dst[1] = @truncate(v >> 8);
        },
        GL_UNSIGNED_SHORT_5_5_5_1 => {
            const a1: u16 = if (src[3] >= 0.5) 1 else 0;
            const v: u16 = (quantN(src[0], 31) << 11) | (quantN(src[1], 31) << 6) | (quantN(src[2], 31) << 1) | a1;
            dst[0] = @truncate(v);
            dst[1] = @truncate(v >> 8);
        },
        GL_HALF_FLOAT_OES => {
            var i: usize = 0;
            while (i < 4) : (i += 1) {
                const hv: f16 = @floatCast(src[i]);
                const bits: u16 = @bitCast(hv);
                dst[i * 2 + 0] = @truncate(bits);
                dst[i * 2 + 1] = @truncate(bits >> 8);
            }
        },
        GL_FLOAT => {
            var i: usize = 0;
            while (i < 4) : (i += 1) {
                const bits: u32 = @bitCast(src[i]);
                dst[i * 4 + 0] = @truncate(bits);
                dst[i * 4 + 1] = @truncate(bits >> 8);
                dst[i * 4 + 2] = @truncate(bits >> 16);
                dst[i * 4 + 3] = @truncate(bits >> 24);
            }
        },
        else => {},
    }
}

/// Unpack one framebuffer texel at byte offset `off` of `px` into RGBA floats, per the source
/// render-target `fmt`. rgba8_unorm normalizes 0..1. Float RT formats read their true value
/// (preserving HDR). Any other format falls back to the rgba8 interpretation.
fn unpackColor(px: []const u8, off: usize, fmt: prism.hal.Format) [4]f32 {
    switch (fmt) {
        .rgba16_float => {
            var out: [4]f32 = undefined;
            inline for (0..4) |c| out[c] = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, px[off + c * 2 ..][0..2], .little))));
            return out;
        },
        .r32g32b32a32_float => {
            var out: [4]f32 = undefined;
            inline for (0..4) |c| out[c] = @bitCast(std.mem.readInt(u32, px[off + c * 4 ..][0..4], .little));
            return out;
        },
        else => return .{
            @as(f32, @floatFromInt(px[off + 0])) / 255.0,
            @as(f32, @floatFromInt(px[off + 1])) / 255.0,
            @as(f32, @floatFromInt(px[off + 2])) / 255.0,
            @as(f32, @floatFromInt(px[off + 3])) / 255.0,
        },
    }
}

/// Write one source texel into `dst` in the internal `target` format's storage. The 8-bit
/// targets (rgba8_unorm / rgba8_srgb) decode via `decodeTexel` (4 bytes). Float targets
/// copy the source's per-channel half/single floats verbatim, expanding a 3-channel (RGB)
/// upload to opaque alpha (1.0). `dst` is exactly one target texel wide.
fn decodeToTarget(target: prism.hal.Format, format: GLenum, gl_type: GLenum, src: []const u8, dst: []u8) void {
    switch (target) {
        .rgba16_float => {
            const nch: usize = if (format == GL_RGBA) 4 else 3;
            inline for (0..4) |c| {
                const h: u16 = if (c < nch)
                    (@as(u16, src[c * 2]) | (@as(u16, src[c * 2 + 1]) << 8))
                else if (c == 3) @as(u16, 0x3C00) else 0; // fp16 1.0 for the missing alpha
                dst[c * 2] = @truncate(h);
                dst[c * 2 + 1] = @truncate(h >> 8);
            }
        },
        .r32g32b32a32_float => {
            const nch: usize = if (format == GL_RGBA) 4 else 3;
            inline for (0..4) |c| {
                const bits: u32 = if (c < nch) blk: {
                    var v: u32 = 0;
                    inline for (0..4) |b| v |= @as(u32, src[c * 4 + b]) << (8 * b);
                    break :blk v;
                } else if (c == 3) @as(u32, @bitCast(@as(f32, 1.0))) else 0;
                inline for (0..4) |b| dst[c * 4 + b] = @truncate(bits >> (8 * b));
            }
        },
        else => { // rgba8_unorm / rgba8_srgb
            var rgba: [4]u8 = undefined;
            decodeTexel(format, gl_type, src, &rgba);
            @memcpy(dst[0..4], &rgba);
        },
    }
}

/// glTexImage2D: upload the base level of a 2D texture. Only GL_UNSIGNED_BYTE pixel data
/// for GL_RGBA/GL_RGB/GL_LUMINANCE/GL_ALPHA/GL_LUMINANCE_ALPHA is supported (the common
/// case). All are stored expanded to tightly-packed RGBA8 (width*height*4) so the software
/// sampler's rgba8 TexDesc reads them directly. GL_UNPACK_ALIGNMENT is honored for the
/// source row stride. `level` > 0 (mipmaps) is accepted but ignored (base level is enough).
/// The Prism texel format for a glTexStorage2D sized internal format (null = unsupported). RGB8/R8/
/// RG8 fold into rgba8_unorm (Prism's storage expands to 4 channels). Float and sRGB sized formats
/// map to their native storage so the sampler decodes them.
fn sizedFormat(internalformat: GLenum) ?prism.hal.Format {
    return switch (internalformat) {
        GL_RGBA8, GL_RGB8, GL_R8, GL_RG8 => .rgba8_unorm,
        GL_SRGB8_ALPHA8 => .rgba8_srgb,
        GL_RGBA16F => .rgba16_float,
        GL_RGBA32F => .r32g32b32a32_float,
        else => null,
    };
}

/// glTexStorage2D(target, levels, internalformat, w, h): declare the active texture's storage as
/// immutable. Fixed dimensions/format, a `levels`-deep mip chain filled only by glTexSubImage2D.
/// Modern GLES3 / EXT_texture_storage code allocates this way instead of per-level glTexImage2D.
pub fn texStorage2D(target: GLenum, levels: GLsizei, internalformat: GLenum, width: GLsizei, height: GLsizei) void {
    if (target != GL_TEXTURE_2D) {
        setError(GL_INVALID_ENUM);
        return;
    }
    if (levels < 1 or width < 1 or height < 1) {
        setError(GL_INVALID_VALUE);
        return;
    }
    const fmt = sizedFormat(internalformat) orelse {
        setError(GL_INVALID_ENUM);
        return;
    };
    obj_lock.lock();
    defer obj_lock.unlock();
    const t = activeTextureObj() orelse {
        setError(GL_INVALID_OPERATION);
        return;
    };
    if (t.immutable) {
        setError(GL_INVALID_OPERATION); // storage already declared immutable
        return;
    }
    t.width = @intCast(width);
    t.height = @intCast(height);
    t.format = fmt;
    t.is_depth = false;
    t.immutable = true;
    t.has_mipmaps = levels > 1; // a multi-level store minifies through the chain (filled later)
    const bpp = fmt.bytesPerPixel();
    t.bytes.clearRetainingCapacity();
    t.bytes.resize(gpa, @as(usize, @intCast(width)) * @as(usize, @intCast(height)) * bpp) catch {
        setError(GL_INVALID_OPERATION);
        return;
    };
    @memset(t.bytes.items, 0);
    t.hal_dirty = true;
}

/// glTexStorage3D (GLES3): immutable storage for a GL_TEXTURE_3D volume or a GL_TEXTURE_2D_ARRAY.
/// Allocates the base level's `width`x`height`x`depth` texels and marks the texture immutable (later
/// glTexImage3D is rejected; glTexSubImage3D is allowed). Mirrors glTexStorage2D but with a depth /
/// layer count. Like the software glTexImage3D path, only the base level is stored + sampled, so a
/// multi-level store still allocates the base (3D/array mip chains are not modeled).
pub fn texStorage3D(target: GLenum, levels: GLsizei, internalformat: GLenum, width: GLsizei, height: GLsizei, depth: GLsizei) void {
    if (target != GL_TEXTURE_3D and target != GL_TEXTURE_2D_ARRAY) {
        setError(GL_INVALID_ENUM);
        return;
    }
    if (levels < 1 or width < 1 or height < 1 or depth < 1) {
        setError(GL_INVALID_VALUE);
        return;
    }
    const fmt = sizedFormat(internalformat) orelse {
        setError(GL_INVALID_ENUM);
        return;
    };
    const is_arr = target == GL_TEXTURE_2D_ARRAY;
    obj_lock.lock();
    defer obj_lock.unlock();
    const t = (if (is_arr) active2dArrayTextureObj() else active3dTextureObj()) orelse {
        setError(GL_INVALID_OPERATION);
        return;
    };
    if (t.immutable) {
        setError(GL_INVALID_OPERATION); // storage already declared immutable
        return;
    }
    t.width = @intCast(width);
    t.height = @intCast(height);
    t.depth = @intCast(depth);
    t.is_3d = !is_arr;
    t.is_array = is_arr;
    t.is_depth = false;
    t.immutable = true;
    const bpp = fmt.bytesPerPixel();
    t.format = fmt;
    t.bytes.clearRetainingCapacity();
    t.bytes.resize(gpa, @as(usize, @intCast(width)) * @as(usize, @intCast(height)) * @as(usize, @intCast(depth)) * bpp) catch {
        setError(GL_INVALID_OPERATION);
        return;
    };
    @memset(t.bytes.items, 0);
    t.hal_dirty = true;
}

pub fn texImage2D(target: GLenum, level: GLint, internalformat: GLint, width: GLsizei, height: GLsizei, border: GLint, format: GLenum, gl_type: GLenum, pixels: ?*const anyopaque) void {
    flushBatch(); // a pending draw may sample this texture. its pixels must not change under it
    if (cubeFaceIndex(target)) |face| {
        texImageCubeFace(face, level, width, height, border, format, gl_type, pixels);
        return;
    }
    if (target != GL_TEXTURE_2D) {
        setError(GL_INVALID_ENUM);
        return;
    }
    if (border != 0 or width < 0 or height < 0 or level < 0) {
        setError(GL_INVALID_VALUE);
        return;
    }
    // GL_OES_depth_texture: a GL_DEPTH_COMPONENT texture allocates a depth32_float render
    // target (no CPU texels uploaded - it is rendered into, then sampled). The data type is
    // GL_UNSIGNED_SHORT / GL_UNSIGNED_INT (depth) per the extension. pixels is normally null.
    if (format == GL_DEPTH_COMPONENT or @as(GLenum, @bitCast(internalformat)) == GL_DEPTH_COMPONENT) {
        if (gl_type != GL_UNSIGNED_SHORT and gl_type != GL_UNSIGNED_INT and gl_type != GL_UNSIGNED_BYTE and gl_type != GL_FLOAT) {
            setError(GL_INVALID_ENUM);
            return;
        }
        obj_lock.lock();
        defer obj_lock.unlock();
        const td = activeTextureObj() orelse {
            setError(GL_INVALID_OPERATION);
            return;
        };
        if (level != 0) return;
        td.width = @intCast(width);
        td.height = @intCast(height);
        td.is_depth = true;
        td.depth_dirty = false;
        // Reset any prior color storage. depth_hal is rebuilt lazily when attached.
        td.bytes.clearRetainingCapacity();
        td.hal_dirty = true;
        return;
    }
    const tbytes = texelBytes(format, gl_type) orelse {
        setError(GL_INVALID_ENUM); // unsupported (format, type) combination
        return;
    };
    obj_lock.lock();
    defer obj_lock.unlock();
    const t = activeTextureObj() orelse {
        setError(GL_INVALID_OPERATION);
        return;
    };
    if (t.immutable) {
        setError(GL_INVALID_OPERATION); // glTexStorage2D made the storage immutable
        return;
    }
    // Mipmap levels are accepted but only the base level is stored (sampled).
    if (level != 0) return;
    const w: u32 = @intCast(width);
    const h: u32 = @intCast(height);
    t.is_depth = false;
    t.width = w;
    t.height = h;
    // The internal storage format (sRGB tag / fp16 / fp32 / plain rgba8) is decided by the SIZED
    // internalformat when it is one (GL_SRGB8_ALPHA8 -> rgba8_srgb, GL_RGBA16F -> fp16, ...), else
    // by the (format, type). If it changed from a prior definition, the lazily-built HAL image must
    // be rebuilt too (different HAL format), so drop it.
    const new_fmt = sizedFormat(@intCast(internalformat)) orelse internalTexFormat(format, gl_type);
    if (new_fmt != t.format) {
        if (t.hal) |old| {
            if (t.hal_dev) |d| d.destroyResource(old);
            t.hal = null;
        }
        t.format = new_fmt;
    }
    t.bytes.clearRetainingCapacity();
    t.bytes.resize(gpa, @as(usize, w) * h * t.format.bytesPerPixel()) catch {
        setError(GL_INVALID_OPERATION);
        return;
    };
    if (pixels) |pp| {
        uploadTexels(t, 0, 0, w, h, format, gl_type, tbytes, @ptrCast(pp));
    } else {
        @memset(t.bytes.items, 0); // a null upload allocates storage (undefined contents -> 0)
    }
    t.hal_dirty = true;
}

/// glTexImage2D with a GL_TEXTURE_CUBE_MAP_* face target: upload one of the 6 faces of the
/// cube bound to GL_TEXTURE_CUBE_MAP on the active unit. All faces share one square size +
/// format. `bytes` holds the 6 faces packed contiguously (face f at f*faceSize).
fn texImageCubeFace(face: u32, level: GLint, width: GLsizei, height: GLsizei, border: GLint, format: GLenum, gl_type: GLenum, pixels: ?*const anyopaque) void {
    if (border != 0 or width < 0 or height < 0 or level < 0) {
        setError(GL_INVALID_VALUE);
        return;
    }
    const tbytes = texelBytes(format, gl_type) orelse {
        setError(GL_INVALID_ENUM);
        return;
    };
    obj_lock.lock();
    defer obj_lock.unlock();
    const t = activeCubeTextureObj() orelse {
        setError(GL_INVALID_OPERATION); // no cubemap bound to the active unit
        return;
    };
    if (t.immutable) {
        setError(GL_INVALID_OPERATION);
        return;
    }
    if (level != 0) return; // only the base level is stored/sampled
    const w: u32 = @intCast(width);
    const h: u32 = @intCast(height);
    t.is_cube = true;
    t.is_depth = false;
    const new_fmt = internalTexFormat(format, gl_type);
    // A dimension/format change resizes the 6-face backing and drops the stale HAL image. The
    // first face upload after that allocation zero-fills, so faces uploaded in any order coexist.
    const face_size = @as(usize, w) * h * new_fmt.bytesPerPixel();
    const needed = 6 * face_size;
    if (t.width != w or t.height != h or t.format != new_fmt or t.bytes.items.len != needed) {
        if (t.hal) |old| {
            if (t.hal_dev) |d| d.destroyResource(old);
            t.hal = null;
        }
        t.width = w;
        t.height = h;
        t.format = new_fmt;
        t.bytes.clearRetainingCapacity();
        t.bytes.resize(gpa, needed) catch {
            setError(GL_INVALID_OPERATION);
            return;
        };
        @memset(t.bytes.items, 0);
    }
    const base_off = @as(usize, face) * face_size;
    if (pixels) |pp| {
        uploadTexelsOff(t, base_off, 0, 0, w, h, format, gl_type, tbytes, @ptrCast(pp));
    }
    t.hal_dirty = true;
}

/// glTexImage3D(GL_TEXTURE_3D, level, internalformat, w, h, depth, ...): upload a 3D texture
/// (sampler3D). `depth` slices of `w`x`h` texels packed slice-major. A vec3-coordinate sample
/// trilinearly interpolates the volume (color-grading LUTs, volume data).
pub fn texImage3D(target: GLenum, level: GLint, internalformat: GLint, width: GLsizei, height: GLsizei, depth: GLsizei, border: GLint, format: GLenum, gl_type: GLenum, pixels: ?*const anyopaque) void {
    _ = internalformat;
    // GL_TEXTURE_3D uploads a volume (sampler3D). GL_TEXTURE_2D_ARRAY uploads independent layers
    // (sampler2DArray). Identical slice-major storage. Only the sampled semantics differ.
    if (target != GL_TEXTURE_3D and target != GL_TEXTURE_2D_ARRAY) {
        setError(GL_INVALID_ENUM);
        return;
    }
    if (border != 0 or width < 0 or height < 0 or depth < 0 or level < 0) {
        setError(GL_INVALID_VALUE);
        return;
    }
    const tbytes = texelBytes(format, gl_type) orelse {
        setError(GL_INVALID_ENUM);
        return;
    };
    obj_lock.lock();
    defer obj_lock.unlock();
    const is_arr = target == GL_TEXTURE_2D_ARRAY;
    const t = (if (is_arr) active2dArrayTextureObj() else active3dTextureObj()) orelse {
        setError(GL_INVALID_OPERATION);
        return;
    };
    if (t.immutable) {
        setError(GL_INVALID_OPERATION);
        return;
    }
    if (level != 0) return; // only the base level is stored/sampled
    const uw: u32 = @intCast(width);
    const uh: u32 = @intCast(height);
    const ud: u32 = @intCast(depth);
    t.is_3d = !is_arr;
    t.is_array = is_arr;
    t.is_depth = false;
    t.width = uw;
    t.height = uh;
    t.depth = ud;
    const new_fmt = internalTexFormat(format, gl_type);
    if (new_fmt != t.format) {
        if (t.hal) |old| {
            if (t.hal_dev) |d| d.destroyResource(old);
            t.hal = null;
        }
        t.format = new_fmt;
    }
    t.bytes.clearRetainingCapacity();
    t.bytes.resize(gpa, @as(usize, uw) * uh * ud * t.format.bytesPerPixel()) catch {
        setError(GL_INVALID_OPERATION);
        return;
    };
    @memset(t.bytes.items, 0);
    if (pixels) |pp| {
        const src: [*]const u8 = @ptrCast(pp);
        const align_u: usize = @intCast(@max(1, unpack_alignment));
        // The per-3D-slice source stride honors GL_UNPACK_ROW_LENGTH too (2D uploads set ud=1 so the
        // per-slice offset is 0; uploadTexelsOff applies the row-length/skip within a slice).
        const row_px: usize = if (unpack_row_length > 0) @intCast(unpack_row_length) else @as(usize, uw);
        const src_row = (row_px * tbytes + align_u - 1) / align_u * align_u;
        const slice_dst = @as(usize, uw) * uh * t.format.bytesPerPixel();
        // GL_UNPACK_IMAGE_HEIGHT sets the source rows per slice (0 = the upload height). The source
        // slice byte stride is that many aligned rows. GL_UNPACK_SKIP_IMAGES skips leading slices.
        // (skip_rows/skip_pixels within each slice are applied by uploadTexelsOff.)
        const img_rows: usize = if (unpack_image_height > 0) @intCast(unpack_image_height) else @as(usize, uh);
        const src_slice = img_rows * src_row;
        const src_base: usize = @as(usize, @intCast(unpack_skip_images)) * src_slice;
        var z: u32 = 0;
        while (z < ud) : (z += 1) {
            uploadTexelsOff(t, @as(usize, z) * slice_dst, 0, 0, uw, uh, format, gl_type, tbytes, src + src_base + @as(usize, z) * src_slice);
        }
    }
    t.hal_dirty = true;
}

/// glTexSubImage2D: replace a sub-rectangle of the base level. Same format constraints as
/// glTexImage2D. The texture must already have base-level storage.
pub fn texSubImage2D(target: GLenum, level: GLint, xoffset: GLint, yoffset: GLint, width: GLsizei, height: GLsizei, format: GLenum, gl_type: GLenum, pixels: ?*const anyopaque) void {
    flushBatch(); // a pending draw may sample this texture
    if (target != GL_TEXTURE_2D) {
        setError(GL_INVALID_ENUM);
        return;
    }
    if (width < 0 or height < 0 or level < 0 or xoffset < 0 or yoffset < 0) {
        setError(GL_INVALID_VALUE);
        return;
    }
    const tbytes = texelBytes(format, gl_type) orelse {
        setError(GL_INVALID_ENUM);
        return;
    };
    obj_lock.lock();
    defer obj_lock.unlock();
    const t = activeTextureObj() orelse {
        setError(GL_INVALID_OPERATION);
        return;
    };
    if (level != 0) return;
    const x: u32 = @intCast(xoffset);
    const y: u32 = @intCast(yoffset);
    const w: u32 = @intCast(width);
    const h: u32 = @intCast(height);
    if (x + w > t.width or y + h > t.height) {
        setError(GL_INVALID_VALUE);
        return;
    }
    if (pixels) |pp| {
        uploadTexels(t, x, y, w, h, format, gl_type, tbytes, @ptrCast(pp));
        t.hal_dirty = true;
    }
}

/// glTexSubImage3D: update a sub-box of an existing 3D texture (GL_TEXTURE_3D). Streams new
/// slices/sub-rects into a LUT/volume without reallocating. Each affected slice's sub-rect is
/// updated via uploadTexelsOff at the slice's byte offset.
pub fn texSubImage3D(target: GLenum, level: GLint, xoffset: GLint, yoffset: GLint, zoffset: GLint, width: GLsizei, height: GLsizei, depth: GLsizei, format: GLenum, gl_type: GLenum, pixels: ?*const anyopaque) void {
    if (target != GL_TEXTURE_3D and target != GL_TEXTURE_2D_ARRAY) {
        setError(GL_INVALID_ENUM);
        return;
    }
    if (width < 0 or height < 0 or depth < 0 or level < 0 or xoffset < 0 or yoffset < 0 or zoffset < 0) {
        setError(GL_INVALID_VALUE);
        return;
    }
    const tbytes = texelBytes(format, gl_type) orelse {
        setError(GL_INVALID_ENUM);
        return;
    };
    obj_lock.lock();
    defer obj_lock.unlock();
    const t = (if (target == GL_TEXTURE_2D_ARRAY) active2dArrayTextureObj() else active3dTextureObj()) orelse {
        setError(GL_INVALID_OPERATION);
        return;
    };
    if (level != 0) return;
    const x: u32 = @intCast(xoffset);
    const y: u32 = @intCast(yoffset);
    const z: u32 = @intCast(zoffset);
    const w: u32 = @intCast(width);
    const h: u32 = @intCast(height);
    const dd: u32 = @intCast(depth);
    if (x + w > t.width or y + h > t.height or z + dd > t.depth) {
        setError(GL_INVALID_VALUE);
        return;
    }
    if (pixels) |pp| {
        const src: [*]const u8 = @ptrCast(pp);
        const align_u: usize = @intCast(@max(1, unpack_alignment));
        const row_px: usize = if (unpack_row_length > 0) @intCast(unpack_row_length) else @as(usize, w);
        const src_row = (row_px * tbytes + align_u - 1) / align_u * align_u;
        const slice_dst = @as(usize, t.width) * t.height * t.format.bytesPerPixel();
        // GL_UNPACK_IMAGE_HEIGHT / SKIP_IMAGES address a sub-volume of a wider source (see texImage3D).
        const img_rows: usize = if (unpack_image_height > 0) @intCast(unpack_image_height) else @as(usize, h);
        const src_slice = img_rows * src_row;
        const src_base: usize = @as(usize, @intCast(unpack_skip_images)) * src_slice;
        var s: u32 = 0;
        while (s < dd) : (s += 1) {
            // Source slice s starts at src_base + s*src_slice. Dest lands in slice z+s. skip_rows/
            // skip_pixels within a slice are applied by uploadTexelsOff.
            uploadTexelsOff(t, @as(usize, z + s) * slice_dst, x, y, w, h, format, gl_type, tbytes, src + src_base + @as(usize, s) * src_slice);
        }
        t.hal_dirty = true;
    }
}

/// The current framebuffer's color, read back into a freshly-allocated tight RGBA8 buffer
/// (row 0 = top, Prism's storage order). Default framebuffer = the surface backbuffer. A
/// bound FBO = its color texture/renderbuffer image. Caller frees `px` with `gpa`. Used by
/// glReadPixels / glCopyTex(Sub)Image2D (the framebuffer-to-CPU / framebuffer-to-texture
/// paths glmark2's effect2d/desktop and many post-processing apps rely on). Caller holds
/// obj_lock. Returns null (and sets GL error) if no color is resolvable.
/// The pixel format of the currently-bound color target: the bound FBO's float color attachment
/// (rgba16f / rgba32f) or rgba8_unorm for the default framebuffer and every 8-bit target. Drives
/// the draw pipeline's color_format so a float FBO renders at full precision. Caller holds obj_lock.
fn currentColorFormat() prism.hal.Format {
    if (bound_framebuffer == 0) return .rgba8_unorm;
    const f = findFramebuffer(bound_framebuffer) orelse return .rgba8_unorm;
    if (f.color_tex != 0) {
        const ct = findTexture(f.color_tex) orelse return .rgba8_unorm;
        // A float attachment renders unclamped. An sRGB attachment (GL_SRGB8_ALPHA8) makes the ROP
        // encode the linear fragment color to sRGB on write (gamma-correct output).
        if (ct.format == .rgba16_float or ct.format == .r32g32b32a32_float or ct.format == .rgba8_srgb) return ct.format;
    }
    return .rgba8_unorm;
}

const FbColor = struct { px: []u8, w: u32, h: u32, format: prism.hal.Format = .rgba8_unorm };
/// Read a specific framebuffer's color into a fresh CPU buffer. glReadPixels / glCopyTex* / the
/// glBlitFramebuffer source read the read framebuffer (`bound_read_framebuffer`). The default
/// framebuffer (0) is the surface backbuffer.
fn readFramebufferColor(ctx: *state.Context, surf: *state.Surface, fb_id: GLuint) ?FbColor {
    // Submit pending batched draws so the readback sees them (covers glReadPixels, glCopyTexImage2D,
    // glCopyTexSubImage2D, glBlitFramebuffer's source read).
    ctx.flushDraws() catch {};
    const dev = ctx.device();
    var res: *prism.hal.Resource = undefined;
    var w: u32 = 0;
    var h: u32 = 0;
    var fmt: prism.hal.Format = .rgba8_unorm; // the default surface + color renderbuffers are rgba8
    if (fb_id == 0) {
        res = surf.backbuffer;
        w = surf.width;
        h = surf.height;
    } else {
        const f = findFramebuffer(fb_id) orelse return null;
        // MRT: glReadBuffer(GL_COLOR_ATTACHMENT_i) selects which attachment glReadPixels reads.
        // read_buffer defaults to attachment 0 (GL_BACK / GL_COLOR_ATTACHMENT0). i>0 -> the extra slot.
        var ctex = f.color_tex;
        var crb = f.color_rb;
        if (read_buffer > GL_COLOR_ATTACHMENT0 and read_buffer < GL_COLOR_ATTACHMENT0 + prism.hal.MAX_COLOR_TARGETS) {
            const idx = read_buffer - GL_COLOR_ATTACHMENT0 - 1;
            ctex = f.extra_color_tex[idx];
            crb = f.extra_color_rb[idx];
        }
        if (ctex != 0) {
            const ct = findTexture(ctex) orelse return null;
            res = ct.hal orelse return null;
            w = ct.width;
            h = ct.height;
            // A float color attachment (rgba16f / rgba32f) reads its true bytes. Other RTs are rgba8.
            if (ct.format == .rgba16_float or ct.format == .r32g32b32a32_float) fmt = ct.format;
        } else if (crb != 0) {
            const cr = findRenderbuffer(crb) orelse return null;
            // A multisampled color renderbuffer must be resolved (box-downsampled) before its
            // pixels can be read. The multisampled image is sample-minor. Single-sample reads
            // its image directly.
            res = resolveRenderbufferColor(ctx, cr) orelse return null;
            w = cr.width;
            h = cr.height;
        } else return null;
    }
    if (w == 0 or h == 0) return null;
    const bpp = fmt.bytesPerPixel();
    const src = dev.mapResource(res) catch return null; // tight, top-down; de-swizzled on nvidia
    const px = gpa.alloc(u8, @as(usize, w) * h * bpp) catch return null;
    const n = @min(px.len, src.len);
    @memcpy(px[0..n], src[0..n]);
    if (n < px.len) @memset(px[n..], 0);
    return .{ .px = px, .w = w, .h = h, .format = fmt };
}
/// The read framebuffer's color (glReadPixels / glCopyTex* source).
fn readCurrentColor(ctx: *state.Context, surf: *state.Surface) ?FbColor {
    return readFramebufferColor(ctx, surf, bound_read_framebuffer);
}

/// glReadPixels(x,y,w,h,format,type,pixels): read a rectangle of the current framebuffer's
/// color into `pixels`. GL's framebuffer origin is bottom-left, so rows are returned bottom-
/// to-top (flipped from Prism's top-down storage). Accepted (format, type) pairs encoded from
/// the 8-bit framebuffer via encodeReadTexel: GL_RGBA/GL_RGB + GL_UNSIGNED_BYTE, GL_RGB +
/// GL_UNSIGNED_SHORT_5_6_5, GL_RGBA + GL_UNSIGNED_SHORT_4_4_4_4 / _5_5_5_1, and GL_RGBA +
/// GL_HALF_FLOAT_OES / GL_FLOAT (normalized 0..1).
pub fn readPixels(x: GLint, y: GLint, width: GLsizei, height: GLsizei, format: GLenum, gl_type: GLenum, pixels: ?*anyopaque) void {
    if (width < 0 or height < 0) {
        setError(GL_INVALID_VALUE);
        return;
    }
    // Destination bytes per pixel for the accepted read combos (0 = rejected).
    const out_bytes: usize = switch (gl_type) {
        GL_UNSIGNED_BYTE => switch (format) {
            GL_RGBA => 4,
            GL_RGB => 3,
            else => 0,
        },
        GL_UNSIGNED_SHORT_5_6_5 => if (format == GL_RGB) @as(usize, 2) else 0,
        GL_UNSIGNED_SHORT_4_4_4_4, GL_UNSIGNED_SHORT_5_5_5_1 => if (format == GL_RGBA) @as(usize, 2) else 0,
        GL_HALF_FLOAT_OES => if (format == GL_RGBA) @as(usize, 8) else 0,
        GL_FLOAT => if (format == GL_RGBA) @as(usize, 16) else 0,
        else => 0,
    };
    if (out_bytes == 0) {
        setError(GL_INVALID_ENUM);
        return;
    }
    const dst_ptr = pixels orelse return;
    const ctx = state.currentContext() orelse {
        setError(GL_INVALID_OPERATION);
        return;
    };
    const surf = state.currentDrawSurface() orelse {
        setError(GL_INVALID_OPERATION);
        return;
    };
    obj_lock.lock();
    defer obj_lock.unlock();
    const fb = readCurrentColor(ctx, surf) orelse {
        setError(GL_INVALID_OPERATION);
        return;
    };
    defer gpa.free(fb.px);
    const w: usize = @intCast(width);
    const h: usize = @intCast(height);
    const src_bpp = fb.format.bytesPerPixel();
    const dst = @as([*]u8, @ptrCast(dst_ptr));
    // GL_PACK_ROW_LENGTH overrides the DESTINATION row length (in pixels); 0 = the read width `w`.
    // The row stride is then aligned up to GL_PACK_ALIGNMENT, and GL_PACK_SKIP_ROWS / SKIP_PIXELS
    // offset the write start so a region lands inside a wider destination atlas.
    const align_d: usize = @intCast(@max(1, pack_alignment));
    const dst_row_pixels: usize = if (pack_row_length > 0) @intCast(pack_row_length) else w;
    const dst_stride = (dst_row_pixels * out_bytes + align_d - 1) / align_d * align_d;
    const dst_skip: usize = @as(usize, @intCast(pack_skip_rows)) * dst_stride + @as(usize, @intCast(pack_skip_pixels)) * out_bytes;
    var r: usize = 0;
    while (r < h) : (r += 1) {
        // GL row r (0 = bottom) maps to framebuffer row (fb.h-1 - (y+r)) in top-down storage.
        const sy = @as(i64, fb.h) - 1 - (@as(i64, y) + @as(i64, @intCast(r)));
        var c: usize = 0;
        while (c < w) : (c += 1) {
            const sx = @as(i64, x) + @as(i64, @intCast(c));
            const di = dst_skip + r * dst_stride + c * out_bytes;
            if (sy < 0 or sy >= fb.h or sx < 0 or sx >= fb.w) {
                @memset(dst[di .. di + out_bytes], 0);
                continue;
            }
            const si = (@as(usize, @intCast(sy)) * fb.w + @as(usize, @intCast(sx))) * src_bpp;
            const rgba = unpackColor(fb.px, si, fb.format);
            encodeReadTexel(format, gl_type, &rgba, dst[di .. di + out_bytes]);
        }
    }
}

/// Pack an RGBA-float color into `dst` at byte `off` per the render target `fmt` (the inverse of
/// unpackColor). Used by glBlitFramebuffer's write side.
fn packColorHal(dst: []u8, off: usize, fmt: prism.hal.Format, c: [4]f32) void {
    switch (fmt) {
        .rgba16_float => {
            inline for (0..4) |k| std.mem.writeInt(u16, dst[off + k * 2 ..][0..2], @bitCast(@as(f16, @floatCast(c[k]))), .little);
        },
        .r32g32b32a32_float => {
            inline for (0..4) |k| std.mem.writeInt(u32, dst[off + k * 4 ..][0..4], @bitCast(c[k]), .little);
        },
        else => {
            dst[off + 0] = unorm8f(c[0]);
            dst[off + 1] = unorm8f(c[1]);
            dst[off + 2] = unorm8f(c[2]);
            dst[off + 3] = unorm8f(c[3]);
        },
    }
}

/// Resolve the draw framebuffer's color image (the glBlitFramebuffer destination) to its HAL
/// resource + dimensions + format. Caller holds obj_lock.
const FbTarget = struct { res: *prism.hal.Resource, w: u32, h: u32, format: prism.hal.Format };
fn drawFramebufferColorTarget(surf: *state.Surface) ?FbTarget {
    if (bound_framebuffer == 0) return .{ .res = surf.backbuffer, .w = surf.width, .h = surf.height, .format = .rgba8_unorm };
    const f = findFramebuffer(bound_framebuffer) orelse return null;
    if (f.color_tex != 0) {
        const ct = findTexture(f.color_tex) orelse return null;
        const res = ct.hal orelse return null;
        const fmt: prism.hal.Format = if (ct.format == .rgba16_float or ct.format == .r32g32b32a32_float) ct.format else .rgba8_unorm;
        return .{ .res = res, .w = ct.width, .h = ct.height, .format = fmt };
    } else if (f.color_rb != 0) {
        const cr = findRenderbuffer(f.color_rb) orelse return null;
        return .{ .res = cr.hal orelse return null, .w = cr.width, .h = cr.height, .format = .rgba8_unorm };
    }
    return null;
}

/// glBlitFramebuffer: copy a rectangle of the READ framebuffer's color to a (possibly scaled)
/// rectangle of the DRAW framebuffer's color, with GL_NEAREST or GL_LINEAR filtering. Framebuffer
/// coords are bottom-left origin (GL). Prism stores top-down, so each axis is flipped on read and
/// write, and an inverted src/dst rect mirrors as GL specifies. Only GL_COLOR_BUFFER_BIT is blitted.
/// Depth/stencil blit is accepted but not copied (no-ops those planes without faulting).
pub fn blitFramebuffer(sx0: GLint, sy0: GLint, sx1: GLint, sy1: GLint, dx0: GLint, dy0: GLint, dx1: GLint, dy1: GLint, mask: GLbitfield, filter: GLenum) void {
    flushBatch(); // pending draws to the source (and dest) must land before the blit reads/writes
    if (filter != GL_NEAREST and filter != GL_LINEAR) {
        setError(GL_INVALID_ENUM);
        return;
    }
    // LINEAR filtering is only defined for a COLOR blit. A depth/stencil blit must be NEAREST.
    if (filter == GL_LINEAR and (mask & (GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT)) != 0) {
        setError(GL_INVALID_OPERATION);
        return;
    }
    if (mask & (GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT) == 0) return;
    const ctx = state.currentContext() orelse {
        setError(GL_INVALID_OPERATION);
        return;
    };
    const surf = state.currentDrawSurface() orelse {
        setError(GL_INVALID_OPERATION);
        return;
    };
    obj_lock.lock();
    defer obj_lock.unlock();
    const dev = ctx.device();

    const rect = BlitRect{ .sx0 = sx0, .sy0 = sy0, .sx1 = sx1, .sy1 = sy1, .dx0 = dx0, .dy0 = dy0, .dx1 = dx1, .dy1 = dy1 };

    if (mask & GL_COLOR_BUFFER_BIT != 0) {
        if (readFramebufferColor(ctx, surf, bound_read_framebuffer)) |src| {
            defer gpa.free(src.px);
            if (drawFramebufferColorTarget(surf)) |dst| {
                if (dev.mapResource(dst.res)) |dmap| {
                    const dbpp = dst.format.bytesPerPixel();
                    const sbpp = src.format.bytesPerPixel();
                    var it = rect.iterator(dst.w, dst.h);
                    while (it.next()) |p| {
                        const c = sampleSrc(src, p.suf, p.svf, sbpp, filter == GL_LINEAR);
                        packColorHal(dmap, (@as(usize, p.drow) * dst.w + p.dcol) * dbpp, dst.format, c);
                    }
                    // Persist the writes to GPU storage: on a driver whose mapResource returns a
                    // de-swizzled scratch of a tiled surface (nvidia block-linear), the writes above
                    // land in the scratch and must be re-swizzled back. No-op on software.
                    dev.flushMappedImage(dst.res);
                } else |_| {}
            }
        }
    }

    // Depth (f32/pixel) and stencil (u8/pixel) are element-copied NEAREST (no filtering). Both
    // are stored top-down like color, so the same GL bottom-left -> top-down row flip applies.
    if (mask & GL_DEPTH_BUFFER_BIT != 0) {
        if (framebufferDepthStencil(ctx, bound_read_framebuffer, .depth)) |src| {
            if (framebufferDepthStencil(ctx, bound_framebuffer, .depth)) |dst| {
                blitElements(dev, src, dst, 4, rect);
            }
        }
    }
    if (mask & GL_STENCIL_BUFFER_BIT != 0) {
        if (framebufferDepthStencil(ctx, bound_read_framebuffer, .stencil)) |src| {
            if (framebufferDepthStencil(ctx, bound_framebuffer, .stencil)) |dst| {
                blitElements(dev, src, dst, 1, rect);
            }
        }
    }
}

/// A glBlitFramebuffer src+dst rectangle (GL bottom-left coords). `iterator` walks the dst
/// pixels and maps each back to its (scaled/inverted) source coordinate.
const BlitRect = struct {
    sx0: GLint,
    sy0: GLint,
    sx1: GLint,
    sy1: GLint,
    dx0: GLint,
    dy0: GLint,
    dx1: GLint,
    dy1: GLint,

    const Pixel = struct { dcol: u32, drow: u32, suf: f32, svf: f32 };
    const Iter = struct {
        r: BlitRect,
        dw: u32,
        dh: u32,
        dxlo: GLint,
        dxhi: GLint,
        dylo: GLint,
        dyhi: GLint,
        dw_span: f32,
        dh_span: f32,
        dx: GLint,
        dy: GLint,

        fn next(self: *Iter) ?Pixel {
            if (self.dw_span == 0 or self.dh_span == 0) return null;
            while (self.dy < self.dyhi) : ({
                self.dy += 1;
                self.dx = self.dxlo;
            }) {
                if (self.dy < 0 or self.dy >= self.dh) continue;
                const tv = (@as(f32, @floatFromInt(self.dy)) + 0.5 - @as(f32, @floatFromInt(self.r.dy0))) / self.dh_span;
                const svf = @as(f32, @floatFromInt(self.r.sy0)) + tv * @as(f32, @floatFromInt(self.r.sy1 - self.r.sy0));
                const drow = self.dh - 1 - @as(u32, @intCast(self.dy));
                while (self.dx < self.dxhi) : (self.dx += 1) {
                    if (self.dx < 0 or self.dx >= self.dw) continue;
                    const tu = (@as(f32, @floatFromInt(self.dx)) + 0.5 - @as(f32, @floatFromInt(self.r.dx0))) / self.dw_span;
                    const suf = @as(f32, @floatFromInt(self.r.sx0)) + tu * @as(f32, @floatFromInt(self.r.sx1 - self.r.sx0));
                    const dcol = @as(u32, @intCast(self.dx));
                    self.dx += 1;
                    return .{ .dcol = dcol, .drow = drow, .suf = suf, .svf = svf };
                }
            }
            return null;
        }
    };

    fn iterator(self: BlitRect, dw: u32, dh: u32) Iter {
        return .{
            .r = self,
            .dw = dw,
            .dh = dh,
            .dxlo = @min(self.dx0, self.dx1),
            .dxhi = @max(self.dx0, self.dx1),
            .dylo = @min(self.dy0, self.dy1),
            .dyhi = @max(self.dy0, self.dy1),
            .dw_span = @floatFromInt(self.dx1 - self.dx0),
            .dh_span = @floatFromInt(self.dy1 - self.dy0),
            .dx = @min(self.dx0, self.dx1),
            .dy = @min(self.dy0, self.dy1),
        };
    }
};

const FbAspect = enum { depth, stencil };
const FbDS = struct { res: *prism.hal.Resource, w: u32, h: u32 };

/// Resolve a framebuffer's depth (f32/pixel) or stencil (u8/pixel) HAL resource for a blit.
/// The default framebuffer's depth/stencil is not exposed here (blits target FBO attachments,
/// the common shadow/depth-copy case). A missing attachment returns null (that aspect no-ops).
fn framebufferDepthStencil(ctx: *state.Context, fb_id: GLuint, aspect: FbAspect) ?FbDS {
    if (fb_id == 0) return null;
    const dev = ctx.device();
    const f = findFramebuffer(fb_id) orelse return null;
    switch (aspect) {
        .depth => {
            if (f.depth_tex != 0) {
                const dt = findTexture(f.depth_tex) orelse return null;
                return .{ .res = dt.depth_hal orelse return null, .w = dt.width, .h = dt.height };
            }
            if (f.depth_rb != 0) {
                const dr = findRenderbuffer(f.depth_rb) orelse return null;
                const res = ensureRenderbufferHal(dev, dr) catch return null;
                return .{ .res = res, .w = dr.width, .h = dr.height };
            }
            return null;
        },
        .stencil => {
            if (f.stencil_rb == 0) return null;
            const sr = findRenderbuffer(f.stencil_rb) orelse return null;
            const res = ensureRenderbufferStencilHal(dev, sr) catch return null;
            return .{ .res = res, .w = sr.width, .h = sr.height };
        },
    }
}

/// Scaled nearest copy of `elem`-byte pixels (depth = 4, stencil = 1) from `src` to `dst`,
/// using the same GL bottom-left -> top-down orientation as the color blit.
fn blitElements(dev: prism.hal.Device, src: FbDS, dst: FbDS, elem: usize, rect: BlitRect) void {
    const smap = dev.mapResource(src.res) catch return;
    const dmap = dev.mapResource(dst.res) catch return;
    var it = rect.iterator(dst.w, dst.h);
    while (it.next()) |p| {
        const scx: i64 = @min(@max(@as(i64, @intFromFloat(@floor(p.suf))), 0), @as(i64, src.w) - 1);
        const scy: i64 = @min(@max(@as(i64, @intFromFloat(@floor(p.svf))), 0), @as(i64, src.h) - 1);
        const srow = src.h - 1 - @as(u32, @intCast(scy)); // GL bottom-left -> top-down
        const soff = (@as(usize, srow) * src.w + @as(usize, @intCast(scx))) * elem;
        const doff = (@as(usize, p.drow) * dst.w + p.dcol) * elem;
        if (soff + elem > smap.len or doff + elem > dmap.len) continue;
        @memcpy(dmap[doff..][0..elem], smap[soff..][0..elem]);
    }
}

/// Sample the blit source `fb` at GL coords (`gx`, `gy`) (bottom-left origin), nearest or bilinear.
fn sampleSrc(fb: FbColor, gx: f32, gy: f32, sbpp: usize, linear: bool) [4]f32 {
    const at = struct {
        fn p(f: FbColor, ix: i64, iy: i64, bpp: usize) [4]f32 {
            const cx: i64 = @min(@max(ix, 0), @as(i64, f.w) - 1);
            const cy: i64 = @min(@max(iy, 0), @as(i64, f.h) - 1);
            const row = @as(usize, @intCast(@as(i64, f.h) - 1 - cy)); // GL -> top-down
            return unpackColor(f.px, (row * f.w + @as(usize, @intCast(cx))) * bpp, f.format);
        }
    }.p;
    if (!linear) {
        return at(fb, @intFromFloat(@floor(gx)), @intFromFloat(@floor(gy)), sbpp);
    }
    const fx = gx - 0.5;
    const fy = gy - 0.5;
    const x0: i64 = @intFromFloat(@floor(fx));
    const y0: i64 = @intFromFloat(@floor(fy));
    const tx = fx - @floor(fx);
    const ty = fy - @floor(fy);
    const c00 = at(fb, x0, y0, sbpp);
    const c10 = at(fb, x0 + 1, y0, sbpp);
    const c01 = at(fb, x0, y0 + 1, sbpp);
    const c11 = at(fb, x0 + 1, y0 + 1, sbpp);
    var out: [4]f32 = undefined;
    inline for (0..4) |k| {
        const a = c00[k] + (c10[k] - c00[k]) * tx;
        const b = c01[k] + (c11[k] - c01[k]) * tx;
        out[k] = a + (b - a) * ty;
    }
    return out;
}

/// glCopyTexImage2D(target,level,internalformat,x,y,w,h,border): define the active texture's
/// base level from a `w`x`h` rectangle of the current framebuffer. Prism stores both top-down,
/// so a direct row copy preserves the image. This is what glmark2's effect2d expects when it
/// captures the scene to a texture to convolve it.
pub fn copyTexImage2D(target: GLenum, level: GLint, internalformat: GLint, x: GLint, y: GLint, width: GLsizei, height: GLsizei, border: GLint) void {
    _ = internalformat;
    _ = border;
    if (target != GL_TEXTURE_2D or level != 0) {
        if (level != 0) return; // only the base level is stored
        setError(GL_INVALID_ENUM);
        return;
    }
    if (width < 0 or height < 0) {
        setError(GL_INVALID_VALUE);
        return;
    }
    const ctx = state.currentContext() orelse {
        setError(GL_INVALID_OPERATION);
        return;
    };
    const surf = state.currentDrawSurface() orelse {
        setError(GL_INVALID_OPERATION);
        return;
    };
    obj_lock.lock();
    defer obj_lock.unlock();
    const t = activeTextureObj() orelse {
        setError(GL_INVALID_OPERATION);
        return;
    };
    const fb = readCurrentColor(ctx, surf) orelse {
        setError(GL_INVALID_OPERATION);
        return;
    };
    defer gpa.free(fb.px);
    const w: u32 = @intCast(width);
    const h: u32 = @intCast(height);
    t.is_depth = false;
    t.width = w;
    t.height = h;
    t.bytes.clearRetainingCapacity();
    t.bytes.resize(gpa, @as(usize, w) * h * 4) catch {
        setError(GL_INVALID_OPERATION);
        return;
    };
    copyFbRectToTexels(t, 0, 0, fb, x, y, w, h);
    t.hal_dirty = true;
}

/// glCopyTexSubImage2D(target,level,xoffset,yoffset,x,y,w,h): update a sub-rectangle of the
/// active texture's base level from the current framebuffer.
pub fn copyTexSubImage2D(target: GLenum, level: GLint, xoffset: GLint, yoffset: GLint, x: GLint, y: GLint, width: GLsizei, height: GLsizei) void {
    if (target != GL_TEXTURE_2D) {
        setError(GL_INVALID_ENUM);
        return;
    }
    if (level != 0) return;
    if (width < 0 or height < 0 or xoffset < 0 or yoffset < 0) {
        setError(GL_INVALID_VALUE);
        return;
    }
    const ctx = state.currentContext() orelse {
        setError(GL_INVALID_OPERATION);
        return;
    };
    const surf = state.currentDrawSurface() orelse {
        setError(GL_INVALID_OPERATION);
        return;
    };
    obj_lock.lock();
    defer obj_lock.unlock();
    const t = activeTextureObj() orelse {
        setError(GL_INVALID_OPERATION);
        return;
    };
    if (t.width == 0 or t.height == 0) {
        setError(GL_INVALID_OPERATION);
        return;
    }
    const fb = readCurrentColor(ctx, surf) orelse {
        setError(GL_INVALID_OPERATION);
        return;
    };
    defer gpa.free(fb.px);
    copyFbRectToTexels(t, @intCast(xoffset), @intCast(yoffset), fb, x, y, @intCast(width), @intCast(height));
    t.hal_dirty = true;
}

/// Copy a `w`x`h` framebuffer rectangle starting at framebuffer (`fx`,`fy`) into texture `t`
/// at (`tx`,`ty`), as RGBA8 (direct top-down copy; out-of-range source reads as 0).
fn copyFbRectToTexels(t: *Texture, tx: u32, ty: u32, fb: FbColor, fx: GLint, fy: GLint, w: u32, h: u32) void {
    var r: u32 = 0;
    while (r < h) : (r += 1) {
        const dty = ty + r;
        if (dty >= t.height) break;
        const sy = @as(i64, fy) + @as(i64, r);
        var c: u32 = 0;
        while (c < w) : (c += 1) {
            const dtx = tx + c;
            if (dtx >= t.width) break;
            const di = (@as(usize, dty) * t.width + dtx) * 4;
            const sx = @as(i64, fx) + @as(i64, c);
            if (sy < 0 or sy >= fb.h or sx < 0 or sx >= fb.w) {
                t.bytes.items[di + 0] = 0;
                t.bytes.items[di + 1] = 0;
                t.bytes.items[di + 2] = 0;
                t.bytes.items[di + 3] = 255;
                continue;
            }
            const si = (@as(usize, @intCast(sy)) * fb.w + @as(usize, @intCast(sx))) * fb.format.bytesPerPixel();
            // Texture storage is rgba8. A float source render target is clamped to 0..1 on capture.
            const rgba = unpackColor(fb.px, si, fb.format);
            t.bytes.items[di + 0] = unorm8f(rgba[0]);
            t.bytes.items[di + 1] = unorm8f(rgba[1]);
            t.bytes.items[di + 2] = unorm8f(rgba[2]);
            t.bytes.items[di + 3] = unorm8f(rgba[3]);
        }
    }
}

/// Copy a `w`x`h` block of source texels (rows aligned to GL_UNPACK_ALIGNMENT, `channels`
/// bytes per texel) into the texture's RGBA8 storage at (`x`,`y`), expanding each texel to
/// RGBA. Caller holds obj_lock and has validated the bounds.
fn uploadTexels(t: *Texture, x: u32, y: u32, w: u32, h: u32, format: GLenum, gl_type: GLenum, texel_bytes: usize, src: [*]const u8) void {
    uploadTexelsOff(t, 0, x, y, w, h, format, gl_type, texel_bytes, src);
}

/// Decode `src` into `t.bytes` at `base_off + (y*t.width + x)*bpp`. `base_off` is 0 for a 2D
/// upload. A cube face passes its face byte offset so the 6 faces pack contiguously in `bytes`.
fn uploadTexelsOff(t: *Texture, base_off: usize, x: u32, y: u32, w: u32, h: u32, format: GLenum, gl_type: GLenum, texel_bytes: usize, src: [*]const u8) void {
    const align_u: usize = @intCast(@max(1, unpack_alignment));
    // GL_UNPACK_ROW_LENGTH overrides the source row length (in pixels); 0 = the upload width `w`.
    // The row byte stride is that rounded up to the unpack alignment. GL_UNPACK_SKIP_ROWS /
    // SKIP_PIXELS offset the source start, so a sub-rectangle of a wider buffer uploads directly.
    const src_row_pixels: usize = if (unpack_row_length > 0) @intCast(unpack_row_length) else w;
    const src_stride = (src_row_pixels * texel_bytes + align_u - 1) / align_u * align_u;
    const skip: usize = @as(usize, @intCast(unpack_skip_rows)) * src_stride + @as(usize, @intCast(unpack_skip_pixels)) * texel_bytes;
    const s: [*]const u8 = src + skip;
    const dst_bpp: usize = t.format.bytesPerPixel(); // internal storage texel size
    var row: u32 = 0;
    while (row < h) : (row += 1) {
        const src_row = s[row * src_stride ..];
        const dst_y = y + row;
        var col: u32 = 0;
        while (col < w) : (col += 1) {
            const doff = base_off + (@as(usize, dst_y) * t.width + (x + col)) * dst_bpp;
            decodeToTarget(t.format, format, gl_type, src_row[col * texel_bytes ..][0..texel_bytes], t.bytes.items[doff..][0..dst_bpp]);
        }
    }
}

/// glGenerateMipmap: mark the bound texture for a full mip chain. The chain is box-downsampled
/// from the base level and uploaded to a mip_levels HAL image at the next ensureTextureHal (so
/// this stays cheap and a later glTexImage2D re-dirties it). The sampler minifies through the
/// chain once the min-filter selects a mipmap mode.
pub fn generateMipmap(target: GLenum) void {
    flushBatch(); // a pending draw may sample this texture. mips are derived from its base level
    if (target != GL_TEXTURE_2D and target != GL_TEXTURE_CUBE_MAP) {
        setError(GL_INVALID_ENUM);
        return;
    }
    obj_lock.lock();
    defer obj_lock.unlock();
    const t = (if (target == GL_TEXTURE_CUBE_MAP) activeCubeTextureObj() else activeTextureObj()) orelse {
        setError(GL_INVALID_OPERATION);
        return;
    };
    if (t.is_depth or t.is_rt) return; // depth / render-target textures are not mip-generated here
    t.has_mipmaps = true;
    t.hal_dirty = true; // force ensureTextureHal to (re)build the chained image
}

// --- ETC1 texture decompression (GL_OES_compressed_ETC1_RGB8_texture) --------
// ETC1 packs a 4x4 RGB block into 8 bytes: two base colors (one per 2x4 / 4x2 sub-block),
// two 3-bit intensity-table indices, and 16 2-bit per-pixel modifier selectors. We decode
// each block to RGBA8 at upload so the rest of the stack (sampler / nvidia TIC) is unchanged.

/// The 8 ETC1 intensity modifier tables (4 signed offsets each; the 2-bit pixel index picks one).
const etc1_modifier = [8][4]i32{
    .{ 2, 8, -2, -8 },     .{ 5, 17, -5, -17 },   .{ 9, 29, -9, -29 },     .{ 13, 42, -13, -42 },
    .{ 18, 60, -18, -60 }, .{ 24, 80, -24, -80 }, .{ 33, 106, -33, -106 }, .{ 47, 183, -47, -183 },
};

fn clamp8(v: i32) u8 {
    return @intCast(std.math.clamp(v, 0, 255));
}

/// Decode one 8-byte ETC1 block into RGBA8, writing the valid `bw`x`bh` region (<=4x4, edge
/// blocks may be partial) row-major into `out` at `out_stride` bytes/row (top-left of the block).
fn etc1DecodeBlock(blk: []const u8, out: []u8, out_stride: usize, bw: u32, bh: u32) void {
    const diff = (blk[3] & 0x02) != 0;
    const flip = (blk[3] & 0x01) != 0;
    var base1: [3]u8 = undefined;
    var base2: [3]u8 = undefined;
    if (diff) {
        // Differential: base1 is 5-bit. base2 = base1 + 3-bit two's-complement delta.
        inline for (0..3) |c| {
            const v5: i32 = @intCast(blk[c] >> 3);
            var d3: i32 = @intCast(blk[c] & 0x7);
            if (d3 >= 4) d3 -= 8;
            base1[c] = ex5(@intCast(std.math.clamp(v5, 0, 31)));
            base2[c] = ex5(@intCast(std.math.clamp(v5 + d3, 0, 31)));
        }
    } else {
        // Individual: two RGB444 base colors.
        base1[0] = ex4(blk[0] >> 4);
        base2[0] = ex4(blk[0] & 0xF);
        base1[1] = ex4(blk[1] >> 4);
        base2[1] = ex4(blk[1] & 0xF);
        base1[2] = ex4(blk[2] >> 4);
        base2[2] = ex4(blk[2] & 0xF);
    }
    const table1: usize = blk[3] >> 5;
    const table2: usize = (blk[3] >> 2) & 0x7;
    // Pixel selectors: the MSB plane in bytes 4-5, the LSB plane in bytes 6-7. Pixel (x,y)
    // uses bit (x*4 + y) of each plane (ETC's column-major pixel order).
    const msb: u32 = (@as(u32, blk[4]) << 8) | blk[5];
    const lsb: u32 = (@as(u32, blk[6]) << 8) | blk[7];
    var x: u32 = 0;
    while (x < 4) : (x += 1) {
        var y: u32 = 0;
        while (y < 4) : (y += 1) {
            const i: u5 = @intCast(x * 4 + y);
            const idx: usize = (((msb >> i) & 1) << 1) | ((lsb >> i) & 1);
            const sub2 = if (flip) (y >= 2) else (x >= 2);
            const base = if (sub2) base2 else base1;
            const table = if (sub2) table2 else table1;
            const mod = etc1_modifier[table][idx];
            if (x < bw and y < bh) {
                const o = @as(usize, y) * out_stride + @as(usize, x) * 4;
                out[o + 0] = clamp8(@as(i32, base[0]) + mod);
                out[o + 1] = clamp8(@as(i32, base[1]) + mod);
                out[o + 2] = clamp8(@as(i32, base[2]) + mod);
                out[o + 3] = 255;
            }
        }
    }
}

// --- S3TC / DXT texture decompression (GL_EXT_texture_compression_s3tc) ------
// DXT packs a 4x4 RGB(A) block: two RGB565 color endpoints interpolated to 4 colors chosen
// per texel by 2-bit indices (row-major), plus (DXT3/5) a 16-byte block with an alpha block.

/// Expand one RGB565 color to RGBA8 (opaque). Reuses the packed-format bit expanders.
fn rgb565(c: u16) [4]u8 {
    return .{ ex5(@intCast((c >> 11) & 0x1f)), ex6(@intCast((c >> 5) & 0x3f)), ex5(@intCast(c & 0x1f)), 255 };
}

/// Decode the 8-byte DXT color block into RGBA8 (alpha 255, except DXT1 punchthrough index 3
/// = transparent black). `punchthrough` = DXT1 semantics (c0<=c1 -> 3-color + transparent);
/// DXT3/DXT5 pass false (always 4 opaque colors, alpha comes from their separate alpha block).
fn dxtColorBlock(blk: []const u8, out: []u8, out_stride: usize, bw: u32, bh: u32, punchthrough: bool) void {
    const c0: u16 = @as(u16, blk[0]) | (@as(u16, blk[1]) << 8);
    const c1: u16 = @as(u16, blk[2]) | (@as(u16, blk[3]) << 8);
    var col: [4][4]u8 = undefined;
    col[0] = rgb565(c0);
    col[1] = rgb565(c1);
    if (c0 > c1 or !punchthrough) {
        // 4-color: c2 = (2c0+c1)/3, c3 = (c0+2c1)/3.
        inline for (0..3) |k| col[2][k] = @intCast((2 * @as(u32, col[0][k]) + col[1][k]) / 3);
        inline for (0..3) |k| col[3][k] = @intCast((@as(u32, col[0][k]) + 2 * @as(u32, col[1][k])) / 3);
        col[2][3] = 255;
        col[3][3] = 255;
    } else {
        // 3-color + transparent: c2 = (c0+c1)/2, c3 = transparent black.
        inline for (0..3) |k| col[2][k] = @intCast((@as(u32, col[0][k]) + col[1][k]) / 2);
        col[2][3] = 255;
        col[3] = .{ 0, 0, 0, 0 };
    }
    const idx: u32 = @as(u32, blk[4]) | (@as(u32, blk[5]) << 8) | (@as(u32, blk[6]) << 16) | (@as(u32, blk[7]) << 24);
    var y: u32 = 0;
    while (y < 4) : (y += 1) {
        var x: u32 = 0;
        while (x < 4) : (x += 1) {
            const ci = (idx >> @as(u5, @intCast((y * 4 + x) * 2))) & 3;
            if (x < bw and y < bh) {
                const o = @as(usize, y) * out_stride + @as(usize, x) * 4;
                @memcpy(out[o .. o + 4], &col[ci]);
            }
        }
    }
}

/// Overwrite the alpha channel from a DXT3 explicit-alpha block (8 bytes, 4 bits/texel).
fn dxt3Alpha(ablk: []const u8, out: []u8, out_stride: usize, bw: u32, bh: u32) void {
    var bits: u64 = 0;
    inline for (0..8) |b| bits |= @as(u64, ablk[b]) << (8 * b);
    var y: u32 = 0;
    while (y < 4) : (y += 1) {
        var x: u32 = 0;
        while (x < 4) : (x += 1) {
            const a4: u8 = @intCast((bits >> @as(u6, @intCast((y * 4 + x) * 4))) & 0xF);
            if (x < bw and y < bh) out[@as(usize, y) * out_stride + @as(usize, x) * 4 + 3] = ex4(a4);
        }
    }
}

/// Decode one 8-byte interpolated-value block (two 8-bit endpoints + 16 3-bit indices selecting
/// from an 8-entry interpolated table) into channel `ch` (0=R,1=G,2=B,3=A) of the output. This is
/// the shared kernel for the DXT5 alpha block AND the RGTC/BC4/BC5 red/green blocks.
fn rgtcChannel(ablk: []const u8, out: []u8, out_stride: usize, bw: u32, bh: u32, ch: usize) void {
    const a0: u32 = ablk[0];
    const a1: u32 = ablk[1];
    var a: [8]u8 = undefined;
    a[0] = @intCast(a0);
    a[1] = @intCast(a1);
    if (a0 > a1) {
        inline for (2..8) |i| a[i] = @intCast(((8 - i) * a0 + (i - 1) * a1) / 7);
    } else {
        inline for (2..6) |i| a[i] = @intCast(((6 - i) * a0 + (i - 1) * a1) / 5);
        a[6] = 0;
        a[7] = 255;
    }
    // The 16 3-bit indices are packed little-endian across bytes 2..7 (48 bits).
    var bits: u64 = 0;
    inline for (0..6) |b| bits |= @as(u64, ablk[2 + b]) << (8 * b);
    var y: u32 = 0;
    while (y < 4) : (y += 1) {
        var x: u32 = 0;
        while (x < 4) : (x += 1) {
            const ai: usize = @intCast((bits >> @as(u6, @intCast((y * 4 + x) * 3))) & 0x7);
            if (x < bw and y < bh) out[@as(usize, y) * out_stride + @as(usize, x) * 4 + ch] = a[ai];
        }
    }
}

/// Fill a `bw`x`bh` region with constant per-channel bytes (used to set the non-decoded channels of
/// an RGTC/BC4/BC5 block before the interpolated red/green channels are written).
fn fillBlockConst(out: []u8, out_stride: usize, bw: u32, bh: u32, rgba: [4]u8) void {
    var y: u32 = 0;
    while (y < bh) : (y += 1) {
        var x: u32 = 0;
        while (x < bw) : (x += 1) {
            @memcpy(out[@as(usize, y) * out_stride + @as(usize, x) * 4 ..][0..4], &rgba);
        }
    }
}

/// Bytes per 4x4 block for a compressed internal format, or null if unsupported.
fn compressedBlockBytes(internalformat: GLenum) ?usize {
    return switch (internalformat) {
        GL_ETC1_RGB8_OES, GL_COMPRESSED_RGB_S3TC_DXT1_EXT, GL_COMPRESSED_RGBA_S3TC_DXT1_EXT, GL_COMPRESSED_RED_RGTC1_EXT => 8,
        GL_COMPRESSED_RGBA_S3TC_DXT3_EXT, GL_COMPRESSED_RGBA_S3TC_DXT5_EXT, GL_COMPRESSED_RG_RGTC2_EXT => 16,
        else => null,
    };
}

/// Decode one compressed 4x4 block of `internalformat` into RGBA8 at `out` (`out_stride`
/// bytes/row), writing only the valid `bw`x`bh` region.
fn decodeCompressedBlock(internalformat: GLenum, blk: []const u8, out: []u8, out_stride: usize, bw: u32, bh: u32) void {
    switch (internalformat) {
        GL_ETC1_RGB8_OES => etc1DecodeBlock(blk, out, out_stride, bw, bh),
        GL_COMPRESSED_RGB_S3TC_DXT1_EXT => dxtColorBlock(blk, out, out_stride, bw, bh, false),
        GL_COMPRESSED_RGBA_S3TC_DXT1_EXT => dxtColorBlock(blk, out, out_stride, bw, bh, true),
        GL_COMPRESSED_RGBA_S3TC_DXT3_EXT => {
            dxtColorBlock(blk[8..16], out, out_stride, bw, bh, false); // color = second 8 bytes
            dxt3Alpha(blk[0..8], out, out_stride, bw, bh); // alpha = first 8 bytes
        },
        GL_COMPRESSED_RGBA_S3TC_DXT5_EXT => {
            dxtColorBlock(blk[8..16], out, out_stride, bw, bh, false);
            rgtcChannel(blk[0..8], out, out_stride, bw, bh, 3); // alpha
        },
        GL_COMPRESSED_RED_RGTC1_EXT => {
            // BC4: interpolated RED, other channels (0, 0, 255). Samples as (R, 0, 0, 1).
            fillBlockConst(out, out_stride, bw, bh, .{ 0, 0, 0, 255 });
            rgtcChannel(blk[0..8], out, out_stride, bw, bh, 0); // red
        },
        GL_COMPRESSED_RG_RGTC2_EXT => {
            // BC5: interpolated RED (first block) + GREEN (second block); B=0, A=255.
            fillBlockConst(out, out_stride, bw, bh, .{ 0, 0, 0, 255 });
            rgtcChannel(blk[0..8], out, out_stride, bw, bh, 0); // red
            rgtcChannel(blk[8..16], out, out_stride, bw, bh, 1); // green
        },
        else => {},
    }
}

/// glCompressedTexImage2D: upload a compressed texture (ETC1 or S3TC/DXT1/3/5). The 4x4
/// blocks are decoded to a tightly-packed RGBA8 base level (so the sampler + nvidia TIC read
/// it like any rgba8 texture). `imageSize` must equal the block count times the format's
/// block size. `level` > 0 is accepted but only the base level is stored.
pub fn compressedTexImage2D(target: GLenum, level: GLint, internalformat: GLenum, width: GLsizei, height: GLsizei, border: GLint, imageSize: GLsizei, data: ?*const anyopaque) void {
    if (target != GL_TEXTURE_2D) {
        setError(GL_INVALID_ENUM);
        return;
    }
    const block_bytes = compressedBlockBytes(internalformat) orelse {
        setError(GL_INVALID_ENUM);
        return;
    };
    if (border != 0 or width < 0 or height < 0 or level < 0 or imageSize < 0) {
        setError(GL_INVALID_VALUE);
        return;
    }
    const w: u32 = @intCast(width);
    const h: u32 = @intCast(height);
    const bx = (w + 3) / 4;
    const by = (h + 3) / 4;
    if (@as(usize, @intCast(imageSize)) != @as(usize, bx) * by * block_bytes) {
        setError(GL_INVALID_VALUE); // block count * the format's block size
        return;
    }
    obj_lock.lock();
    defer obj_lock.unlock();
    const t = activeTextureObj() orelse {
        setError(GL_INVALID_OPERATION);
        return;
    };
    if (level != 0) return;
    // A compressed texture decodes to a plain rgba8 texture. If the prior definition was a
    // different internal format, drop the lazily-built HAL image so it rebuilds as rgba8_unorm.
    if (t.format != .rgba8_unorm) {
        if (t.hal) |old| {
            if (t.hal_dev) |d| d.destroyResource(old);
            t.hal = null;
        }
        t.format = .rgba8_unorm;
    }
    t.is_depth = false;
    t.width = w;
    t.height = h;
    t.bytes.clearRetainingCapacity();
    t.bytes.resize(gpa, @as(usize, w) * h * 4) catch {
        setError(GL_INVALID_OPERATION);
        return;
    };
    @memset(t.bytes.items, 0);
    if (data) |dp| {
        const src: [*]const u8 = @ptrCast(dp);
        const stride = @as(usize, w) * 4;
        var byi: u32 = 0;
        while (byi < by) : (byi += 1) {
            var bxi: u32 = 0;
            while (bxi < bx) : (bxi += 1) {
                const blk = src[(@as(usize, byi) * bx + bxi) * block_bytes ..][0..block_bytes];
                const px0 = bxi * 4;
                const py0 = byi * 4;
                const out_off = (@as(usize, py0) * w + px0) * 4;
                decodeCompressedBlock(internalformat, blk, t.bytes.items[out_off..], stride, @min(4, w - px0), @min(4, h - py0));
            }
        }
    }
    t.hal_dirty = true;
}

// --- GL_OES_mapbuffer -------------------------------------------------------
// glMapBufferOES returns a pointer into the bound Buffer's CPU bytes the app writes. The
// matching glUnmapBufferOES commits the writes (invalidating the lazily-built HAL vertex
// resource so the next draw re-uploads them) - the same effect as glBufferSubData, which
// is glmark2 buffer:update-method=map's path.

/// glMapBufferOES(target, access): map the buffer bound to `target` for CPU access. Only
/// GL_WRITE_ONLY_OES is meaningful. Returns the address of the buffer's CPU storage (the
/// app writes vertices through it). Returns null + GL_INVALID_OPERATION if no buffer is
/// bound, the buffer has no storage, or it is already mapped.
pub fn mapBufferOES(target: GLenum, access: GLenum) ?*anyopaque {
    flushBatch(); // the map may read data a pending draw / transform feedback wrote
    if (access != GL_WRITE_ONLY_OES) {
        // GLES only defines WRITE_ONLY for GL_OES_mapbuffer.
        setError(GL_INVALID_ENUM);
        return null;
    }
    const id = switch (target) {
        GL_ARRAY_BUFFER => bound_array_buffer,
        GL_ELEMENT_ARRAY_BUFFER => bound_element_buffer,
        else => {
            setError(GL_INVALID_ENUM);
            return null;
        },
    };
    if (id == 0) {
        setError(GL_INVALID_OPERATION);
        return null;
    }
    obj_lock.lock();
    defer obj_lock.unlock();
    const b = findBuffer(id) orelse {
        setError(GL_INVALID_OPERATION);
        return null;
    };
    if (b.mapped or b.bytes.items.len == 0) {
        setError(GL_INVALID_OPERATION);
        return null;
    }
    b.mapped = true;
    return @ptrCast(b.bytes.items.ptr);
}

/// glUnmapBufferOES(target): commit a mapped buffer's writes. Invalidates the HAL vertex
/// resource so the next draw rebuilds it from the (now-updated) CPU bytes. Returns GL_FALSE
/// (+ GL_INVALID_OPERATION) if the buffer was not mapped.
pub fn unmapBufferOES(target: GLenum) GLboolean {
    const id = switch (target) {
        GL_ARRAY_BUFFER => bound_array_buffer,
        GL_ELEMENT_ARRAY_BUFFER => bound_element_buffer,
        else => {
            setError(GL_INVALID_ENUM);
            return GL_FALSE;
        },
    };
    if (id == 0) {
        setError(GL_INVALID_OPERATION);
        return GL_FALSE;
    }
    obj_lock.lock();
    defer obj_lock.unlock();
    const b = findBuffer(id) orelse {
        setError(GL_INVALID_OPERATION);
        return GL_FALSE;
    };
    if (!b.mapped) {
        setError(GL_INVALID_OPERATION);
        return GL_FALSE;
    }
    b.mapped = false;
    // Invalidate the built HAL resource: the next draw re-uploads from b.bytes (the same
    // path glBufferData's invalidation drives).
    if (b.hal) |h| {
        if (b.hal_dev) |dev| dev.destroyResource(h);
        b.hal = null;
        b.hal_dev = null;
    }
    return GL_TRUE;
}

/// glMapBufferRange(target, offset, length, access): map a SUB-RANGE of the bound buffer for CPU
/// access (GLES3). Returns the address of `bytes[offset]`. The app writes vertices/indices there
/// and glUnmapBuffer commits them. Only array/element targets (the streaming case) are mapped.
/// GL_MAP_INVALIDATE_*_BIT zero the mapped range (a discard-and-refill, avoiding a stale read).
pub fn mapBufferRange(target: GLenum, offset: GLintptr, length: GLsizeiptr, access: GLbitfield) ?*anyopaque {
    flushBatch(); // the map may read data a pending draw / transform feedback wrote
    if (access & (GL_MAP_READ_BIT | GL_MAP_WRITE_BIT) == 0) {
        setError(GL_INVALID_OPERATION); // must map for read and/or write
        return null;
    }
    const id = switch (target) {
        GL_ARRAY_BUFFER => bound_array_buffer,
        GL_ELEMENT_ARRAY_BUFFER => bound_element_buffer,
        else => {
            setError(GL_INVALID_ENUM);
            return null;
        },
    };
    if (id == 0) {
        setError(GL_INVALID_OPERATION);
        return null;
    }
    obj_lock.lock();
    defer obj_lock.unlock();
    const b = findBuffer(id) orelse {
        setError(GL_INVALID_OPERATION);
        return null;
    };
    if (b.mapped) {
        setError(GL_INVALID_OPERATION); // already mapped
        return null;
    }
    if (offset < 0 or length <= 0 or @as(usize, @intCast(offset)) + @as(usize, @intCast(length)) > b.bytes.items.len) {
        setError(GL_INVALID_VALUE);
        return null;
    }
    const off: usize = @intCast(offset);
    const len: usize = @intCast(length);
    if (access & (GL_MAP_INVALIDATE_RANGE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT) != 0) {
        @memset(b.bytes.items[off..][0..len], 0);
    }
    b.mapped = true;
    return @ptrCast(b.bytes.items.ptr + off);
}

/// glUnmapBuffer(target): the GLES3 name for committing a mapped buffer. Same effect as
/// glUnmapBufferOES; re-uploads on the next draw.
pub fn unmapBuffer(target: GLenum) GLboolean {
    return unmapBufferOES(target);
}

/// glFlushMappedBufferRange(target, offset, length): explicitly flush a sub-range of a buffer
/// mapped with GL_MAP_FLUSH_EXPLICIT_BIT. Prism commits the whole buffer at glUnmapBuffer, so
/// this is a validated no-op (target must be a mapped array/element buffer).
pub fn flushMappedBufferRange(target: GLenum, offset: GLintptr, length: GLsizeiptr) void {
    const id = switch (target) {
        GL_ARRAY_BUFFER => bound_array_buffer,
        GL_ELEMENT_ARRAY_BUFFER => bound_element_buffer,
        else => return setError(GL_INVALID_ENUM),
    };
    obj_lock.lock();
    defer obj_lock.unlock();
    const b = findBuffer(id) orelse return setError(GL_INVALID_OPERATION);
    if (!b.mapped or offset < 0 or length < 0) setError(GL_INVALID_VALUE);
}

/// glDrawRangeElements(mode, start, end, count, type, indices): glDrawElements with an advisory
/// [start, end] index-range hint (Prism ignores the hint; it draws the same). `start > end` is
/// GL_INVALID_VALUE.
pub fn drawRangeElements(mode: GLenum, start: GLuint, end: GLuint, count: GLsizei, index_type: GLenum, offset: usize) void {
    if (start > end) {
        setError(GL_INVALID_VALUE);
        return;
    }
    drawElements(mode, count, index_type, offset);
}

// --- Framebuffer objects (render-to-texture) --------------------------------

pub fn genFramebuffers(n: GLsizei, out: ?[*]GLuint) void {
    if (n < 0 or out == null) {
        if (n < 0) setError(GL_INVALID_VALUE);
        return;
    }
    obj_lock.lock();
    defer obj_lock.unlock();
    var i: usize = 0;
    while (i < @as(usize, @intCast(n))) : (i += 1) {
        const f = gpa.create(Framebuffer) catch {
            setError(GL_INVALID_OPERATION);
            return;
        };
        f.* = .{ .id = allocId() };
        framebuffers.append(gpa, f) catch {
            gpa.destroy(f);
            setError(GL_INVALID_OPERATION);
            return;
        };
        out.?[i] = f.id;
    }
}

pub fn bindFramebuffer(target: GLenum, id: GLuint) void {
    flushBatch(); // pending draws belong to the OLD render target. submit before retargeting
    if (target != GL_FRAMEBUFFER and target != GL_READ_FRAMEBUFFER and target != GL_DRAW_FRAMEBUFFER) {
        setError(GL_INVALID_ENUM);
        return;
    }
    if (id != 0) {
        obj_lock.lock();
        const f = findFramebuffer(id);
        if (f) |fb| fb.bound_once = true;
        obj_lock.unlock();
        if (f == null) {
            setError(GL_INVALID_OPERATION);
            return;
        }
    }
    // GL_FRAMEBUFFER binds BOTH the draw and read targets. The split targets bind one each.
    if (target == GL_FRAMEBUFFER or target == GL_DRAW_FRAMEBUFFER) bound_framebuffer = id;
    if (target == GL_FRAMEBUFFER or target == GL_READ_FRAMEBUFFER) bound_read_framebuffer = id;
}

pub fn deleteFramebuffers(n: GLsizei, ids: ?[*]const GLuint) void {
    flushBatch(); // a pending draw may target this FBO's attachments
    if (n < 0 or ids == null) {
        if (n < 0) setError(GL_INVALID_VALUE);
        return;
    }
    obj_lock.lock();
    defer obj_lock.unlock();
    var i: usize = 0;
    while (i < @as(usize, @intCast(n))) : (i += 1) {
        const id = ids.?[i];
        if (id == 0) continue;
        for (framebuffers.items, 0..) |f, idx| {
            if (f.id == id) {
                gpa.destroy(f);
                _ = framebuffers.swapRemove(idx);
                break;
            }
        }
        if (bound_framebuffer == id) bound_framebuffer = 0;
    }
}

pub fn isFramebuffer(id: GLuint) GLboolean {
    if (id == 0) return GL_FALSE;
    obj_lock.lock();
    defer obj_lock.unlock();
    const f = findFramebuffer(id) orelse return GL_FALSE;
    return if (f.bound_once) GL_TRUE else GL_FALSE;
}

/// glFramebufferTexture2D: attach a texture's image to the bound FBO's color or depth
/// attachment. `textarget` must be GL_TEXTURE_2D; `level` 0. A GL_DEPTH_ATTACHMENT of a
/// GL_DEPTH_COMPONENT texture is the GL_OES_depth_texture shadow/distance map.
pub fn framebufferTexture2D(target: GLenum, attachment: GLenum, textarget: GLenum, texture: GLuint, level: GLint) void {
    flushBatch(); // pending draws reference the FBO's current attachments. submit before changing them
    _ = level;
    if (target != GL_FRAMEBUFFER) {
        setError(GL_INVALID_ENUM);
        return;
    }
    if (textarget != GL_TEXTURE_2D) {
        setError(GL_INVALID_ENUM);
        return;
    }
    if (bound_framebuffer == 0) {
        setError(GL_INVALID_OPERATION); // cannot attach to the default framebuffer
        return;
    }
    obj_lock.lock();
    defer obj_lock.unlock();
    const f = findFramebuffer(bound_framebuffer) orelse {
        setError(GL_INVALID_OPERATION);
        return;
    };
    if (texture != 0 and findTexture(texture) == null) {
        setError(GL_INVALID_OPERATION);
        return;
    }
    switch (attachment) {
        GL_COLOR_ATTACHMENT0 => {
            f.color_tex = texture;
            f.color_rb = 0;
            // This texture will be rendered into: mark it so ensureTextureHal builds a
            // render-target-capable HAL image. If a sampled-only image was already built,
            // drop it so it is recreated with render_target usage.
            if (texture != 0) {
                if (findTexture(texture)) |ct| {
                    if (!ct.is_rt) {
                        ct.is_rt = true;
                        if (ct.hal) |old| {
                            if (ct.hal_dev) |hd| hd.destroyResource(old);
                            ct.hal = null;
                            ct.hal_dirty = true;
                        }
                    }
                }
            }
        },
        GL_DEPTH_ATTACHMENT => {
            f.depth_tex = texture;
            f.depth_rb = 0;
        },
        GL_STENCIL_ATTACHMENT => {}, // accepted; stencil is not modeled (no-op)
        else => {
            // MRT: GL_COLOR_ATTACHMENT1..N-1 -> extra_color_tex[attachment - ATTACHMENT0 - 1].
            if (attachment > GL_COLOR_ATTACHMENT0 and attachment < GL_COLOR_ATTACHMENT0 + prism.hal.MAX_COLOR_TARGETS) {
                const idx = attachment - GL_COLOR_ATTACHMENT0 - 1;
                f.extra_color_tex[idx] = texture;
                f.extra_color_rb[idx] = 0;
                if (texture != 0) markTextureRenderTarget(texture);
            } else setError(GL_INVALID_ENUM);
        },
    }
}

/// Mark a texture as an FBO color attachment: ensureTextureHal builds a render-target-capable
/// HAL image, and any sampled-only image already built is dropped so it is recreated with
/// render_target usage. Shared by GL_COLOR_ATTACHMENT0 and the MRT attachments.
fn markTextureRenderTarget(texture: GLuint) void {
    if (findTexture(texture)) |ct| {
        if (!ct.is_rt) {
            ct.is_rt = true;
            if (ct.hal) |old| {
                if (ct.hal_dev) |hd| hd.destroyResource(old);
                ct.hal = null;
                ct.hal_dirty = true;
            }
        }
    }
}

pub fn framebufferRenderbuffer(target: GLenum, attachment: GLenum, rbtarget: GLenum, renderbuffer: GLuint) void {
    flushBatch(); // pending draws reference the FBO's current attachments. submit before changing them
    if (target != GL_FRAMEBUFFER) {
        setError(GL_INVALID_ENUM);
        return;
    }
    if (rbtarget != GL_RENDERBUFFER) {
        setError(GL_INVALID_ENUM);
        return;
    }
    if (bound_framebuffer == 0) {
        setError(GL_INVALID_OPERATION);
        return;
    }
    obj_lock.lock();
    defer obj_lock.unlock();
    const f = findFramebuffer(bound_framebuffer) orelse {
        setError(GL_INVALID_OPERATION);
        return;
    };
    if (renderbuffer != 0 and findRenderbuffer(renderbuffer) == null) {
        setError(GL_INVALID_OPERATION);
        return;
    }
    switch (attachment) {
        GL_COLOR_ATTACHMENT0 => {
            f.color_rb = renderbuffer;
            f.color_tex = 0;
        },
        GL_DEPTH_ATTACHMENT => {
            f.depth_rb = renderbuffer;
            f.depth_tex = 0;
        },
        GL_STENCIL_ATTACHMENT => {
            f.stencil_rb = renderbuffer;
        },
        // Packed depth+stencil (OES_packed_depth_stencil): one renderbuffer feeds BOTH the depth
        // and stencil attachment points.
        GL_DEPTH_STENCIL_ATTACHMENT => {
            f.depth_rb = renderbuffer;
            f.depth_tex = 0;
            f.stencil_rb = renderbuffer;
        },
        else => {
            // MRT: GL_COLOR_ATTACHMENT1..N-1 renderbuffer -> extra_color_rb slot.
            if (attachment > GL_COLOR_ATTACHMENT0 and attachment < GL_COLOR_ATTACHMENT0 + prism.hal.MAX_COLOR_TARGETS) {
                const idx = attachment - GL_COLOR_ATTACHMENT0 - 1;
                f.extra_color_rb[idx] = renderbuffer;
                f.extra_color_tex[idx] = 0;
            } else setError(GL_INVALID_ENUM);
        },
    }
}

/// glCheckFramebufferStatus: report whether the bound FBO is renderable. The default
/// framebuffer (0) is always complete. An FBO is complete when it has at least one
/// attachment (color or depth) that has storage.
/// Validate the attachment list for glInvalidate(Sub)Framebuffer. Prism is not a tiled/deferred
/// renderer so there is nothing to discard. The call is a no-op after validation.
fn invalidateValidate(target: GLenum, num: GLsizei, attachments: ?[*]const GLenum) bool {
    if (target != GL_FRAMEBUFFER and target != GL_READ_FRAMEBUFFER and target != GL_DRAW_FRAMEBUFFER) {
        setError(GL_INVALID_ENUM);
        return false;
    }
    if (num < 0) {
        setError(GL_INVALID_VALUE);
        return false;
    }
    if (attachments) |att| {
        var i: usize = 0;
        while (i < @as(usize, @intCast(num))) : (i += 1) {
            switch (att[i]) {
                GL_COLOR_ATTACHMENT0, GL_DEPTH_ATTACHMENT, GL_STENCIL_ATTACHMENT, GL_DEPTH_STENCIL_ATTACHMENT, GL_COLOR, GL_DEPTH, GL_STENCIL => {},
                else => {
                    setError(GL_INVALID_ENUM);
                    return false;
                },
            }
        }
    }
    return true;
}

/// glInvalidateFramebuffer: a hint that the listed attachments' contents are no longer needed (a
/// tiled-GPU bandwidth optimization). Prism keeps every attachment, so it validates and no-ops.
pub fn invalidateFramebuffer(target: GLenum, num: GLsizei, attachments: ?[*]const GLenum) void {
    _ = invalidateValidate(target, num, attachments);
}

/// glInvalidateSubFramebuffer: the sub-rectangle form; same validation + no-op (a negative extent
/// is GL_INVALID_VALUE).
pub fn invalidateSubFramebuffer(target: GLenum, num: GLsizei, attachments: ?[*]const GLenum, x: GLint, y: GLint, width: GLsizei, height: GLsizei) void {
    _ = x;
    _ = y;
    if (width < 0 or height < 0) {
        setError(GL_INVALID_VALUE);
        return;
    }
    _ = invalidateValidate(target, num, attachments);
}

pub fn checkFramebufferStatus(target: GLenum) GLenum {
    if (target != GL_FRAMEBUFFER) {
        setError(GL_INVALID_ENUM);
        return 0;
    }
    if (bound_framebuffer == 0) return GL_FRAMEBUFFER_COMPLETE;
    obj_lock.lock();
    defer obj_lock.unlock();
    const f = findFramebuffer(bound_framebuffer) orelse return GL_FRAMEBUFFER_UNSUPPORTED;
    if (f.color_tex == 0 and f.color_rb == 0 and f.depth_tex == 0 and f.depth_rb == 0 and f.stencil_rb == 0) {
        return GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT;
    }
    return GL_FRAMEBUFFER_COMPLETE;
}

/// glGetFramebufferAttachmentParameteriv: introspect one attachment point of the bound FBO
/// (OBJECT_TYPE = TEXTURE / RENDERBUFFER / NONE, OBJECT_NAME = the object id, and the level /
/// cube-map-face which are 0 for Prism's 2D-only attachments). ES2 rules: the default
/// framebuffer is not queryable (GL_INVALID_OPERATION), and OBJECT_NAME / TEXTURE_LEVEL /
/// CUBE_MAP_FACE are only valid when the attachment's type matches (else GL_INVALID_ENUM).
pub fn getFramebufferAttachmentParameteriv(target: GLenum, attachment: GLenum, pname: GLenum, params: ?[*]GLint) void {
    const p = params orelse return;
    if (target != GL_FRAMEBUFFER) {
        setError(GL_INVALID_ENUM);
        return;
    }
    if (bound_framebuffer == 0) {
        setError(GL_INVALID_OPERATION); // ES2: only a bound FBO object is queryable
        return;
    }
    obj_lock.lock();
    defer obj_lock.unlock();
    const f = findFramebuffer(bound_framebuffer) orelse {
        setError(GL_INVALID_OPERATION);
        return;
    };
    var tex_id: GLuint = 0;
    var rb_id: GLuint = 0;
    switch (attachment) {
        GL_COLOR_ATTACHMENT0 => {
            tex_id = f.color_tex;
            rb_id = f.color_rb;
        },
        GL_DEPTH_ATTACHMENT => {
            tex_id = f.depth_tex;
            rb_id = f.depth_rb;
        },
        GL_STENCIL_ATTACHMENT => rb_id = f.stencil_rb,
        else => {
            setError(GL_INVALID_ENUM);
            return;
        },
    }
    const obj_type: GLenum = if (tex_id != 0) GL_TEXTURE else if (rb_id != 0) GL_RENDERBUFFER else GL_NONE;
    switch (pname) {
        GL_FRAMEBUFFER_ATTACHMENT_OBJECT_TYPE => p[0] = @intCast(obj_type),
        GL_FRAMEBUFFER_ATTACHMENT_OBJECT_NAME => {
            if (obj_type == GL_NONE) {
                setError(GL_INVALID_ENUM);
                return;
            }
            p[0] = @intCast(if (tex_id != 0) tex_id else rb_id);
        },
        GL_FRAMEBUFFER_ATTACHMENT_TEXTURE_LEVEL, GL_FRAMEBUFFER_ATTACHMENT_TEXTURE_CUBE_MAP_FACE => {
            if (obj_type != GL_TEXTURE) {
                setError(GL_INVALID_ENUM);
                return;
            }
            p[0] = 0; // only mip level 0 attaches. 2D textures have no cube face
        },
        else => {
            setError(GL_INVALID_ENUM);
            return;
        },
    }
}

/// glGetAttachedShaders: list the shader ids attached to `program` (Prism attaches at most a VS
/// and an FS), up to `max_count`. `count` receives how many were written.
pub fn getAttachedShaders(program: GLuint, max_count: GLsizei, count: ?*GLsizei, out_shaders: ?[*]GLuint) void {
    if (max_count < 0) {
        setError(GL_INVALID_VALUE);
        return;
    }
    obj_lock.lock();
    defer obj_lock.unlock();
    const p = findProgram(program) orelse {
        setError(GL_INVALID_OPERATION);
        return;
    };
    const cap: usize = @intCast(max_count);
    var n: usize = 0;
    if (p.vs_shader) |vs| {
        if (n < cap) {
            if (out_shaders) |s| s[n] = vs;
            n += 1;
        }
    }
    if (p.fs_shader) |fs| {
        if (n < cap) {
            if (out_shaders) |s| s[n] = fs;
            n += 1;
        }
    }
    if (count) |c| c.* = @intCast(n);
}

/// Read the current value(s) of the uniform at `location` into `out` (up to out.len floats),
/// returning the count written (null = invalid program/location). A default-block member yields
/// its float_count components from whichever stage's block declares it. A sampler uniform yields
/// its bound texture unit as one value. Caller holds obj_lock.
fn readUniform(program: GLuint, location: GLint, out: []GLfloat) ?usize {
    if (location < 0) return null;
    const p = findProgram(program) orelse return null;
    if (!p.linked) return null;
    if ((location & SAMPLER_LOCATION_FLAG) != 0) {
        const si: usize = @intCast(location & ~SAMPLER_LOCATION_FLAG);
        if (si >= p.samplers.items.len) return null;
        if (out.len > 0) out[0] = @floatFromInt(p.samplers.items[si].unit);
        return 1;
    }
    const idx: usize = @intCast(location);
    if (idx >= p.uniforms.items.len) return null;
    const m = p.uniforms.items[idx];
    const bytes = if (m.vs_offset >= 0) &p.vs_uniform_bytes else if (m.fs_offset >= 0) &p.fs_uniform_bytes else return null;
    const off: usize = if (m.vs_offset >= 0) @intCast(m.vs_offset) else @intCast(m.fs_offset);
    const cap = @min(out.len, m.float_count);
    var i: usize = 0;
    while (i < cap and off + (i + 1) * 4 <= bytes.items.len) : (i += 1) {
        out[i] = std.mem.bytesToValue(f32, bytes.items[off + i * 4 ..][0..4]);
    }
    return i;
}
/// glGetUniformfv: read a uniform's current value (mat4 is the widest at 16 floats).
pub fn getUniformfv(program: GLuint, location: GLint, params: ?[*]GLfloat) void {
    const p = params orelse return;
    obj_lock.lock();
    defer obj_lock.unlock();
    var buf: [16]GLfloat = undefined;
    const n = readUniform(program, location, &buf) orelse {
        setError(GL_INVALID_OPERATION);
        return;
    };
    for (0..n) |i| p[i] = buf[i];
}
/// glGetUniformiv: the integer form (each component rounds; a sampler reports its texture unit).
pub fn getUniformiv(program: GLuint, location: GLint, params: ?[*]GLint) void {
    const p = params orelse return;
    obj_lock.lock();
    defer obj_lock.unlock();
    var buf: [16]GLfloat = undefined;
    const n = readUniform(program, location, &buf) orelse {
        setError(GL_INVALID_OPERATION);
        return;
    };
    for (0..n) |i| p[i] = @intFromFloat(@round(buf[i]));
}

/// glGetVertexAttribPointerv: the client byte offset (GLES VBO convention) of a vertex-attrib
/// array, returned as a void* (the offset reinterpreted as a pointer, per GL).
pub fn getVertexAttribPointerv(index: GLuint, pname: GLenum, pointer: ?*?*anyopaque) void {
    if (index >= MAX_ATTRIBS) {
        setError(GL_INVALID_VALUE);
        return;
    }
    if (pname != GL_VERTEX_ATTRIB_ARRAY_POINTER) {
        setError(GL_INVALID_ENUM);
        return;
    }
    if (pointer) |pp| pp.* = @ptrFromInt(attribs[index].offset);
}

// --- Renderbuffer objects ---------------------------------------------------

pub fn genRenderbuffers(n: GLsizei, out: ?[*]GLuint) void {
    if (n < 0 or out == null) {
        if (n < 0) setError(GL_INVALID_VALUE);
        return;
    }
    obj_lock.lock();
    defer obj_lock.unlock();
    var i: usize = 0;
    while (i < @as(usize, @intCast(n))) : (i += 1) {
        const r = gpa.create(Renderbuffer) catch {
            setError(GL_INVALID_OPERATION);
            return;
        };
        r.* = .{ .id = allocId() };
        renderbuffers.append(gpa, r) catch {
            gpa.destroy(r);
            setError(GL_INVALID_OPERATION);
            return;
        };
        out.?[i] = r.id;
    }
}

pub fn bindRenderbuffer(target: GLenum, id: GLuint) void {
    if (target != GL_RENDERBUFFER) {
        setError(GL_INVALID_ENUM);
        return;
    }
    if (id != 0) {
        obj_lock.lock();
        const r = findRenderbuffer(id);
        if (r) |rb| rb.bound_once = true;
        obj_lock.unlock();
        if (r == null) {
            setError(GL_INVALID_OPERATION);
            return;
        }
    }
    bound_renderbuffer = id;
}

/// glRenderbufferStorage: set the bound renderbuffer's format + size. A depth format backs
/// a HAL depth32_float buffer. A color format an rgba8_unorm image (lazily built when the
/// FBO that owns it is first used).
pub fn renderbufferStorage(target: GLenum, internalformat: GLenum, width: GLsizei, height: GLsizei) void {
    renderbufferStorageMultisample(target, 0, internalformat, width, height);
}

/// glRenderbufferStorageMultisample(target, samples, internalformat, w, h): storage for a
/// multisampled renderbuffer. samples clamps to Prism's supported counts: 0/1 = single-sample,
/// else 2 or 4 (the SSAA levels the software raster and resolve support). An FBO with such a
/// color attachment renders anti-aliased. glBlitFramebuffer to a single-sample target resolves it.
pub fn renderbufferStorageMultisample(target: GLenum, samples: GLsizei, internalformat: GLenum, width: GLsizei, height: GLsizei) void {
    if (target != GL_RENDERBUFFER) {
        setError(GL_INVALID_ENUM);
        return;
    }
    if (width < 0 or height < 0 or samples < 0) {
        setError(GL_INVALID_VALUE);
        return;
    }
    if (bound_renderbuffer == 0) {
        setError(GL_INVALID_OPERATION);
        return;
    }
    // A renderbuffer is depth-only, stencil-only, packed depth+stencil, or color.
    var is_depth = false;
    var is_stencil = false;
    switch (internalformat) {
        GL_DEPTH_COMPONENT16, GL_DEPTH_COMPONENT24, GL_DEPTH_COMPONENT32, GL_DEPTH_COMPONENT => is_depth = true,
        GL_STENCIL_INDEX8 => is_stencil = true,
        GL_DEPTH24_STENCIL8 => {
            is_depth = true;
            is_stencil = true;
        },
        GL_RGBA4, GL_RGB565, GL_RGB5_A1, GL_RGBA, GL_RGB, GL_RGBA8, GL_RGB8 => {},
        else => {
            setError(GL_INVALID_ENUM);
            return;
        },
    }
    obj_lock.lock();
    defer obj_lock.unlock();
    const r = findRenderbuffer(bound_renderbuffer) orelse {
        setError(GL_INVALID_OPERATION);
        return;
    };
    // Re-spec frees any prior HAL backing (rebuilt lazily for the new size/format).
    if (r.hal) |h| if (r.hal_dev) |dev| dev.destroyResource(h);
    if (r.stencil_hal) |h| if (r.hal_dev) |dev| dev.destroyResource(h);
    if (r.resolved_hal) |h| if (r.hal_dev) |dev| dev.destroyResource(h);
    r.hal = null;
    r.stencil_hal = null;
    r.resolved_hal = null;
    r.hal_dev = null;
    r.width = @intCast(width);
    r.height = @intCast(height);
    r.is_depth = is_depth;
    r.is_stencil = is_stencil;
    // Clamp the requested sample count to Prism's supported set (1 / 2 / 4).
    r.samples = if (samples <= 1) 1 else if (samples == 2) 2 else 4;
}

pub fn deleteRenderbuffers(n: GLsizei, ids: ?[*]const GLuint) void {
    flushBatch(); // a pending draw may use these as color/depth/stencil attachments
    if (n < 0 or ids == null) {
        if (n < 0) setError(GL_INVALID_VALUE);
        return;
    }
    obj_lock.lock();
    defer obj_lock.unlock();
    var i: usize = 0;
    while (i < @as(usize, @intCast(n))) : (i += 1) {
        const id = ids.?[i];
        if (id == 0) continue;
        for (renderbuffers.items, 0..) |r, idx| {
            if (r.id == id) {
                if (r.hal) |h| if (r.hal_dev) |dev| dev.destroyResource(h);
                if (r.stencil_hal) |h| if (r.hal_dev) |dev| dev.destroyResource(h);
                gpa.destroy(r);
                _ = renderbuffers.swapRemove(idx);
                break;
            }
        }
        if (bound_renderbuffer == id) bound_renderbuffer = 0;
    }
}

pub fn isRenderbuffer(id: GLuint) GLboolean {
    if (id == 0) return GL_FALSE;
    obj_lock.lock();
    defer obj_lock.unlock();
    const r = findRenderbuffer(id) orelse return GL_FALSE;
    return if (r.bound_once) GL_TRUE else GL_FALSE;
}

// --- Vertex attribute arrays ------------------------------------------------

pub fn vertexAttribPointer(index: GLuint, size: GLint, gl_type: GLenum, normalized: GLboolean, stride: GLsizei, offset: usize) void {
    if (index >= MAX_ATTRIBS) {
        setError(GL_INVALID_VALUE);
        return;
    }
    if (size < 1 or size > 4) {
        setError(GL_INVALID_VALUE);
        return;
    }
    if (attribTypeSize(gl_type) == 0) {
        setError(GL_INVALID_ENUM); // an unknown vertex-attribute component type
        return;
    }
    if (stride < 0) {
        setError(GL_INVALID_VALUE);
        return;
    }
    attribs[index] = .{
        .enabled = attribs[index].enabled,
        .size = size,
        .gl_type = gl_type,
        .normalized = normalized != 0,
        .stride = stride,
        .offset = offset,
        .buffer = bound_array_buffer,
    };
}

/// glVertexAttribIPointer (GLES3): declare an integer vertex-attribute array. Unlike
/// glVertexAttribPointer, the source integers are delivered raw to an integer VS input
/// (`in ivec4`/`uvec4`), sign/zero-extended to 32 bits. Not normalized and not reinterpreted.
/// Accepts BYTE/UNSIGNED_BYTE/SHORT/UNSIGNED_SHORT/INT/UNSIGNED_INT. The raw value rides the
/// f32 gather path (e.g. 200 -> 200.0) and the GLSL front end converts it back at the VS entry.
pub fn vertexAttribIPointer(index: GLuint, size: GLint, gl_type: GLenum, stride: GLsizei, offset: usize) void {
    if (index >= MAX_ATTRIBS) {
        setError(GL_INVALID_VALUE);
        return;
    }
    if (size < 1 or size > 4) {
        setError(GL_INVALID_VALUE);
        return;
    }
    // Only INTEGER source types are valid for glVertexAttribIPointer (no FLOAT/HALF/FIXED).
    switch (gl_type) {
        GL_BYTE, GL_UNSIGNED_BYTE, GL_SHORT, GL_UNSIGNED_SHORT, GL_INT, GL_UNSIGNED_INT => {},
        else => {
            setError(GL_INVALID_ENUM);
            return;
        },
    }
    if (stride < 0) {
        setError(GL_INVALID_VALUE);
        return;
    }
    attribs[index] = .{
        .enabled = attribs[index].enabled,
        .size = size,
        .gl_type = gl_type,
        .normalized = false, // integer attributes are not normalized
        .stride = stride,
        .offset = offset,
        .buffer = bound_array_buffer,
        .integer = true,
    };
}

pub fn enableVertexAttribArray(index: GLuint) void {
    if (index >= MAX_ATTRIBS) {
        setError(GL_INVALID_VALUE);
        return;
    }
    attribs[index].enabled = true;
}

pub fn disableVertexAttribArray(index: GLuint) void {
    if (index >= MAX_ATTRIBS) {
        setError(GL_INVALID_VALUE);
        return;
    }
    attribs[index].enabled = false;
}

// --- Shader objects ---------------------------------------------------------

pub fn createShader(shader_type: GLenum) GLuint {
    const stage: prism.hal.ShaderStage = switch (shader_type) {
        GL_VERTEX_SHADER => .vertex,
        GL_FRAGMENT_SHADER => .fragment,
        else => {
            setError(GL_INVALID_ENUM);
            return 0;
        },
    };
    obj_lock.lock();
    defer obj_lock.unlock();
    const s = gpa.create(Shader) catch {
        setError(GL_INVALID_OPERATION);
        return 0;
    };
    s.* = .{ .id = allocId(), .stage = stage };
    shaders.append(gpa, s) catch {
        gpa.destroy(s);
        setError(GL_INVALID_OPERATION);
        return 0;
    };
    return s.id;
}

/// glShaderSource: retain the GLSL-ES source (concatenated). Stored for a future
/// GLSL front-end milestone. The M3 triangle uses the SPIR-V binary path instead.
pub fn shaderSource(shader: GLuint, count: GLsizei, strings: ?[*]const ?[*:0]const GLchar, lengths: ?[*]const GLint) void {
    if (count < 0 or strings == null) {
        if (count < 0) setError(GL_INVALID_VALUE);
        return;
    }
    obj_lock.lock();
    defer obj_lock.unlock();
    const s = findShader(shader) orelse {
        setError(GL_INVALID_OPERATION);
        return;
    };
    s.source.clearRetainingCapacity();
    var i: usize = 0;
    while (i < @as(usize, @intCast(count))) : (i += 1) {
        const str = strings.?[i] orelse continue;
        const slice: []const u8 = if (lengths) |lens| blk: {
            const l = lens[i];
            if (l < 0) break :blk std.mem.span(str);
            break :blk @as([*]const u8, @ptrCast(str))[0..@intCast(l)];
        } else std.mem.span(str);
        s.source.appendSlice(gpa, slice) catch {
            setError(GL_INVALID_OPERATION);
            return;
        };
    }
}

/// glShaderBinary (GL_ARB_gl_spirv): load a SPIR-V binary into one or more shader
/// objects. Accepts GL_SHADER_BINARY_FORMAT_SPIR_V and stores the word stream for
/// each named shader. glSpecializeShader/glCompileShader then validates it (no-op).
/// The bytes feed HAL createShaderModule verbatim at link time.
pub fn shaderBinary(count: GLsizei, shaderlist: ?[*]const GLuint, binaryformat: GLenum, binary: ?*const anyopaque, length: GLsizei) void {
    if (binaryformat != GL_SHADER_BINARY_FORMAT_SPIR_V) {
        setError(GL_INVALID_ENUM);
        return;
    }
    if (count < 0 or shaderlist == null or binary == null or length < 0 or @rem(length, 4) != 0) {
        setError(GL_INVALID_VALUE);
        return;
    }
    const bytes = @as([*]const u8, @ptrCast(binary.?))[0..@intCast(length)];
    obj_lock.lock();
    defer obj_lock.unlock();
    var i: usize = 0;
    while (i < @as(usize, @intCast(count))) : (i += 1) {
        const s = findShader(shaderlist.?[i]) orelse {
            setError(GL_INVALID_OPERATION);
            return;
        };
        s.spirv.clearRetainingCapacity();
        s.spirv.appendSlice(gpa, bytes) catch {
            setError(GL_INVALID_OPERATION);
            return;
        };
        s.compiled = true; // a loaded SPIR-V binary is ready (glSpecializeShader is a no-op here)
    }
}

/// glCompileShader: compile the shader object's source.
/// SPIR-V binary path (glShaderBinary already loaded a word stream): a no-op success.
/// The binary feeds HAL createShaderModule verbatim at link time.
/// GLSL-source path (glShaderSource set the text): compile the GLSL ES 1.00
/// source to SPIR-V via the Vulcan GLSL front end (prism.glsl), keyed by the
/// shader's stage, and store the resulting SPIR-V exactly like the binary path.
/// On a GLSL compile error the shader is marked NOT compiled and the diagnostic is
/// recorded for glGetShaderInfoLog. glGetShaderiv(GL_COMPILE_STATUS) then reports
/// GL_FALSE. Never crashes on bad input.
var spv_dump_seq: u32 = 0; // DIAG: sequence counter for PRISM_DUMP_SPV shader dumps

pub fn compileShader(shader: GLuint) void {
    obj_lock.lock();
    defer obj_lock.unlock();
    const s = findShader(shader) orelse {
        setError(GL_INVALID_OPERATION);
        return;
    };
    s.info_log.clearRetainingCapacity();
    // A pre-loaded SPIR-V binary is already "compiled".
    if (s.spirv.items.len > 0) {
        s.compiled = true;
        return;
    }
    // GLSL source -> SPIR-V via the Vulcan GLSL front end.
    if (s.source.items.len == 0) {
        s.compiled = false;
        s.info_log.appendSlice(gpa, "no shader source\x00") catch {};
        return;
    }
    var compiled = prism.glsl.compileForStageWithLayout(gpa, s.source.items, s.stage) catch |e| {
        s.compiled = false;
        // A real (human-readable) message. The front end's error name pinpoints the gap.
        s.info_log.appendSlice(gpa, "GLSL compile error: ") catch {};
        s.info_log.appendSlice(gpa, @errorName(e)) catch {};
        s.info_log.append(gpa, 0) catch {};
        if (debugEnabled()) {
            std.debug.print("[prism-gles] compile FAIL stage={s} err={s}\n----\n{s}\n----\n", .{ @tagName(s.stage), @errorName(e), s.source.items });
        }
        return;
    };
    defer compiled.deinit(gpa);
    s.spirv.clearRetainingCapacity();
    s.spirv.appendSlice(gpa, compiled.spirv) catch {
        s.compiled = false;
        s.info_log.appendSlice(gpa, "out of memory\x00") catch {};
        return;
    };
    // DIAG (PRISM_DUMP_SPV): write each compiled shader's SPIR-V to /tmp/prism_spv so it can be
    // fed to the sass_disasm tool (the perf-hunt oracle for the complex-fragment-shader cliff).
    if (getenv("PRISM_DUMP_SPV") != null) {
        const linux = std.os.linux;
        _ = linux.mkdir("/tmp/prism_spv", 0o755);
        var namebuf: [64]u8 = undefined;
        const nm = std.fmt.bufPrintZ(&namebuf, "/tmp/prism_spv/{s}_{d}.spv", .{ @tagName(s.stage), spv_dump_seq }) catch return;
        spv_dump_seq += 1;
        const fd_us = linux.open(nm.ptr, .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true }, 0o644);
        if (@as(isize, @bitCast(fd_us)) >= 0) {
            const fd: i32 = @intCast(fd_us);
            _ = linux.write(fd, compiled.spirv.ptr, compiled.spirv.len);
            _ = linux.close(fd);
        }
    }
    // Record the default-uniform-block layout (own the names: dupe them; compiled.deinit
    // frees its copies). linkProgram copies these onto the program.
    for (s.uniforms.items) |u| gpa.free(@constCast(u.name));
    s.uniforms.clearRetainingCapacity();
    for (compiled.uniforms) |u| {
        const owned = gpa.dupe(u8, u.name) catch {
            s.compiled = false;
            s.info_log.appendSlice(gpa, "out of memory\x00") catch {};
            return;
        };
        s.uniforms.append(gpa, .{ .name = owned, .offset = u.offset, .float_count = u.float_count, .mat_dim = u.mat_dim, .array_len = u.array_len }) catch {
            gpa.free(owned);
            s.compiled = false;
            return;
        };
    }
    s.uniform_block_size = compiled.block_size;
    // Record the sampler members (name -> binding). Dupe the names (own them).
    for (s.samplers.items) |sm| gpa.free(@constCast(sm.name));
    s.samplers.clearRetainingCapacity();
    for (compiled.samplers) |sm| {
        const owned = gpa.dupe(u8, sm.name) catch {
            s.compiled = false;
            s.info_log.appendSlice(gpa, "out of memory\x00") catch {};
            return;
        };
        s.samplers.append(gpa, .{ .name = owned, .binding = sm.binding, .cube = sm.cube, .tex3d = sm.tex3d, .tex2darray = sm.tex2darray }) catch {
            gpa.free(owned);
            s.compiled = false;
            return;
        };
    }
    // Record the vertex attributes (name -> location + components). Dupe the names (own them).
    for (s.attributes.items) |a| gpa.free(@constCast(a.name));
    s.attributes.clearRetainingCapacity();
    for (compiled.attributes) |a| {
        const owned = gpa.dupe(u8, a.name) catch {
            s.compiled = false;
            s.info_log.appendSlice(gpa, "out of memory\x00") catch {};
            return;
        };
        s.attributes.append(gpa, .{ .name = owned, .location = a.location, .components = a.components }) catch {
            gpa.free(owned);
            s.compiled = false;
            return;
        };
    }
    // Record the VS output varyings (name -> location + components) for transform feedback. Dupe
    // the names (own them). Empty for a fragment shader.
    for (s.outputs.items) |o| gpa.free(@constCast(o.name));
    s.outputs.clearRetainingCapacity();
    for (compiled.outputs) |o| {
        const owned = gpa.dupe(u8, o.name) catch {
            s.compiled = false;
            s.info_log.appendSlice(gpa, "out of memory\x00") catch {};
            return;
        };
        s.outputs.append(gpa, .{ .name = owned, .location = o.location, .components = o.components }) catch {
            gpa.free(owned);
            s.compiled = false;
            return;
        };
    }
    // Record the named uniform blocks (name -> binding point + data size + std140 repack table).
    // Dupe both the names and the member tables (own them).
    for (s.uniform_blocks.items) |b| {
        gpa.free(@constCast(b.name));
        gpa.free(@constCast(b.members));
    }
    s.uniform_blocks.clearRetainingCapacity();
    for (compiled.uniform_blocks) |b| {
        const owned = gpa.dupe(u8, b.name) catch {
            s.compiled = false;
            s.info_log.appendSlice(gpa, "out of memory\x00") catch {};
            return;
        };
        const owned_members = gpa.dupe(prism.glsl.UniformBlockMember, b.members) catch {
            gpa.free(owned);
            s.compiled = false;
            s.info_log.appendSlice(gpa, "out of memory\x00") catch {};
            return;
        };
        s.uniform_blocks.append(gpa, .{ .name = owned, .binding = b.binding, .byte_offset = b.byte_offset, .byte_size = b.byte_size, .members = owned_members }) catch {
            gpa.free(owned);
            gpa.free(owned_members);
            s.compiled = false;
            return;
        };
    }
    s.compiled = true;
}

/// glSpecializeShader (GL_ARB_gl_spirv): bind the SPIR-V entry point + specialization
/// constants. We only support entry "main" with no spec constants (the triangle's
/// shaders). The binary is already loaded so this just confirms readiness.
pub fn specializeShader(shader: GLuint, entry: ?[*:0]const GLchar, num_spec: GLuint, spec_index: ?[*]const GLuint, spec_value: ?[*]const GLuint) void {
    _ = spec_index;
    _ = spec_value;
    obj_lock.lock();
    defer obj_lock.unlock();
    const s = findShader(shader) orelse {
        setError(GL_INVALID_OPERATION);
        return;
    };
    if (s.spirv.items.len == 0) {
        setError(GL_INVALID_OPERATION);
        return;
    }
    if (num_spec != 0) {
        setError(GL_INVALID_VALUE); // spec constants not modeled in M3
        return;
    }
    if (entry) |e| {
        if (!std.mem.eql(u8, std.mem.span(e), "main")) {
            setError(GL_INVALID_VALUE);
            return;
        }
    }
    s.compiled = true;
}

pub fn getShaderiv(shader: GLuint, pname: GLenum, params: ?*GLint) void {
    const p = params orelse return;
    obj_lock.lock();
    defer obj_lock.unlock();
    const s = findShader(shader) orelse {
        setError(GL_INVALID_OPERATION);
        return;
    };
    p.* = switch (pname) {
        GL_COMPILE_STATUS => if (s.compiled) @as(GLint, GL_TRUE) else GL_FALSE,
        GL_SHADER_TYPE => @intCast(if (s.stage == .vertex) GL_VERTEX_SHADER else GL_FRAGMENT_SHADER),
        GL_DELETE_STATUS => if (s.deleted) @as(GLint, GL_TRUE) else GL_FALSE,
        // Includes the NUL terminator, per GL (0 when there is no log).
        GL_INFO_LOG_LENGTH => @intCast(s.info_log.items.len),
        // Source length including the NUL terminator, per GL (0 when no source set).
        GL_SHADER_SOURCE_LENGTH => if (s.source.items.len == 0) 0 else @intCast(s.source.items.len + 1),
        else => {
            setError(GL_INVALID_ENUM);
            return;
        },
    };
}

/// glGetShaderInfoLog: copy the (NUL-terminated) compile diagnostic into `info_log`,
/// truncated to `buf_size`, writing the byte count (excluding NUL) to `length`. Empty
/// when compilation succeeded or used the SPIR-V binary path.
/// glGetShaderSource: copy the shader's stored GLSL source into `source` (NUL-terminated, at most
/// buf_size bytes). glShaderSource stored it. Debuggers and some frameworks read it back.
pub fn getShaderSource(shader: GLuint, buf_size: GLsizei, length: ?*GLint, source: ?[*]GLchar) void {
    obj_lock.lock();
    defer obj_lock.unlock();
    const s = findShader(shader) orelse {
        setError(GL_INVALID_OPERATION);
        return;
    };
    if (buf_size < 0) {
        setError(GL_INVALID_VALUE);
        return;
    }
    const src = s.source.items; // raw GLSL (no stored NUL)
    const out = source orelse {
        if (length) |lp| lp.* = 0;
        return;
    };
    if (buf_size == 0) {
        if (length) |lp| lp.* = 0;
        return;
    }
    const n = @min(@as(usize, @intCast(buf_size - 1)), src.len);
    var i: usize = 0;
    while (i < n) : (i += 1) out[i] = src[i];
    out[n] = 0;
    if (length) |lp| lp.* = @intCast(n);
}

/// glDepthRangef(n, f): the window-space depth range NDC z maps into. Stored + reported by
/// glGetFloatv(GL_DEPTH_RANGE). the default [0,1] is Prism's native Vulkan depth range.
threadlocal var depth_range: [2]f32 = .{ 0, 1 };
pub fn depthRangef(n: GLfloat, f: GLfloat) void {
    depth_range = .{ std.math.clamp(n, 0, 1), std.math.clamp(f, 0, 1) };
}

/// glReleaseShaderCompiler: a hint that the shader compiler's resources may be freed. Prism's
/// compiler is always available, so this is a no-op.
pub fn releaseShaderCompiler() void {}

/// glHint(target, mode): a quality/performance hint (mipmap generation, derivative accuracy).
/// Validated + ignored (Prism has one implementation per operation).
pub fn hint(target: GLenum, mode: GLenum) void {
    if (mode != GL_FASTEST and mode != GL_NICEST and mode != GL_DONT_CARE) {
        setError(GL_INVALID_ENUM);
        return;
    }
    switch (target) {
        GL_GENERATE_MIPMAP_HINT, GL_FRAGMENT_SHADER_DERIVATIVE_HINT => {},
        else => setError(GL_INVALID_ENUM),
    }
}

pub fn getShaderInfoLog(shader: GLuint, buf_size: GLsizei, length: ?*GLint, info_log: ?[*]GLchar) void {
    obj_lock.lock();
    defer obj_lock.unlock();
    const s = findShader(shader) orelse {
        setError(GL_INVALID_OPERATION);
        return;
    };
    if (buf_size < 0) {
        setError(GL_INVALID_VALUE);
        return;
    }
    const log = s.info_log.items; // includes a trailing NUL when non-empty
    const out = info_log orelse {
        if (length) |lp| lp.* = 0;
        return;
    };
    if (buf_size == 0) {
        if (length) |lp| lp.* = 0;
        return;
    }
    // Copy at most buf_size-1 bytes then NUL-terminate (GL semantics).
    const max_copy: usize = @intCast(buf_size - 1);
    const src_len = if (log.len > 0) log.len - 1 else 0; // exclude our stored NUL
    const n = @min(max_copy, src_len);
    var i: usize = 0;
    while (i < n) : (i += 1) out[i] = log[i];
    out[n] = 0;
    if (length) |lp| lp.* = @intCast(n);
}

pub fn deleteShader(shader: GLuint) void {
    if (shader == 0) return;
    obj_lock.lock();
    defer obj_lock.unlock();
    for (shaders.items, 0..) |s, idx| {
        if (s.id == shader) {
            s.source.deinit(gpa);
            s.spirv.deinit(gpa);
            s.info_log.deinit(gpa);
            for (s.uniforms.items) |u| gpa.free(@constCast(u.name));
            s.uniforms.deinit(gpa);
            for (s.samplers.items) |sm| gpa.free(@constCast(sm.name));
            s.samplers.deinit(gpa);
            for (s.attributes.items) |a| gpa.free(@constCast(a.name));
            s.attributes.deinit(gpa);
            for (s.outputs.items) |o| gpa.free(@constCast(o.name));
            s.outputs.deinit(gpa);
            for (s.uniform_blocks.items) |b| {
                gpa.free(@constCast(b.name));
                gpa.free(@constCast(b.members));
            }
            s.uniform_blocks.deinit(gpa);
            gpa.destroy(s);
            _ = shaders.swapRemove(idx);
            return;
        }
    }
}

// --- Program objects --------------------------------------------------------

pub fn createProgram() GLuint {
    obj_lock.lock();
    defer obj_lock.unlock();
    const p = gpa.create(Program) catch {
        setError(GL_INVALID_OPERATION);
        return 0;
    };
    p.* = .{ .id = allocId() };
    programs.append(gpa, p) catch {
        gpa.destroy(p);
        setError(GL_INVALID_OPERATION);
        return 0;
    };
    return p.id;
}

pub fn attachShader(program: GLuint, shader: GLuint) void {
    obj_lock.lock();
    defer obj_lock.unlock();
    const p = findProgram(program) orelse {
        setError(GL_INVALID_OPERATION);
        return;
    };
    const s = findShader(shader) orelse {
        setError(GL_INVALID_OPERATION);
        return;
    };
    switch (s.stage) {
        .vertex => p.vs_shader = shader,
        .fragment => p.fs_shader = shader,
        .compute => setError(GL_INVALID_OPERATION),
    }
}

pub fn detachShader(program: GLuint, shader: GLuint) void {
    obj_lock.lock();
    defer obj_lock.unlock();
    const p = findProgram(program) orelse {
        setError(GL_INVALID_OPERATION);
        return;
    };
    if (p.vs_shader == shader) p.vs_shader = null;
    if (p.fs_shader == shader) p.fs_shader = null;
}

/// glLinkProgram: build the HAL ShaderModules (from each attached shader's SPIR-V)
/// and a HAL Pipeline. The pipeline's vertex layout is derived from the enabled
/// vertex-attribute arrays at link time when a draw surface is current. Otherwise the
/// modules are built and the pipeline is deferred to the first glDrawArrays (which
/// always has the current draw surface + bound arrays). Builds on the proven HAL graphics
/// path: createShaderModule(SPIR-V) feeds spirv.zig and the Vulcan JIT in the software
/// driver. createPipeline JITs the VS+FS.
/// Record a NUL-terminated link diagnostic for glGetProgramInfoLog. Surfacing the reason
/// (a missing/uncompiled stage, or a stage that failed native codegen) lets apps print a
/// real error instead of a blank "Failed to link" message.
fn setProgramLog(p: *Program, msg: []const u8) void {
    p.info_log.clearRetainingCapacity();
    p.info_log.appendSlice(gpa, msg) catch {};
    p.info_log.append(gpa, 0) catch {};
}

pub fn linkProgram(program: GLuint) void {
    obj_lock.lock();
    const p = findProgram(program) orelse {
        obj_lock.unlock();
        setError(GL_INVALID_OPERATION);
        return;
    };
    p.info_log.clearRetainingCapacity();
    const vs_id = p.vs_shader;
    const fs_id = p.fs_shader;
    const vs = if (vs_id) |id| findShader(id) else null;
    const fs = if (fs_id) |id| findShader(id) else null;
    // Both stages must be attached and carry SPIR-V to link (M3 SPIR-V path).
    if (vs == null or fs == null or vs.?.spirv.items.len == 0 or fs.?.spirv.items.len == 0) {
        const why = if (vs == null or (vs.?.spirv.items.len == 0))
            "link failed: the vertex shader is not attached or did not compile"
        else
            "link failed: the fragment shader is not attached or did not compile";
        setProgramLog(p, why);
        p.linked = false;
        obj_lock.unlock();
        return; // link "fails" honestly. glGetProgramiv(LINK_STATUS) reports GL_FALSE
    }
    // Copy the SPIR-V out under the lock, then build the HAL modules without it
    // (createShaderModule may allocate; keep the critical section small).
    obj_lock.unlock();

    const ctx = state.currentContext() orelse {
        setProgramLog(p, "link failed: no current context");
        setError(GL_INVALID_OPERATION); // need a current context's HAL device to build
        return;
    };
    const dev = ctx.device();

    // Tear down any previous link products.
    destroyProgramHal(p);

    // GL vs Vulkan framebuffer origin: GLES uses a BOTTOM-LEFT origin (NDC +y = TOP) but the
    // HAL drivers rasterize y-down (Vulkan convention, the ICD relies on it). Negate the VS's
    // gl_Position.y so GL +y lands at the top - upright GLES content on software AND nvidia
    // without touching the drivers or the Vulkan ICD. The cull winding flips with this (handled
    // in drawTriangleList by inverting the front face). flipPositionY returns a verbatim copy
    // if there is no gl_Position store, so a malformed VS still links as before.
    const flipped_vs = prism.spirv.flipPositionY(gpa, vs.?.spirv.items) catch {
        setProgramLog(p, "link failed: out of memory rewriting the vertex shader for the GL framebuffer origin");
        setError(GL_INVALID_OPERATION);
        return;
    };
    defer gpa.free(flipped_vs);
    const vs_mod = dev.createShaderModule(.{ .stage = .vertex, .code = flipped_vs }) catch {
        setProgramLog(p, "link failed: the vertex shader could not be compiled to native code for this driver (unsupported shader construct)");
        setError(GL_INVALID_OPERATION);
        return;
    };
    const fs_mod = dev.createShaderModule(.{ .stage = .fragment, .code = fs.?.spirv.items }) catch {
        dev.destroyShaderModule(vs_mod);
        setProgramLog(p, "link failed: the fragment shader could not be compiled to native code for this driver (unsupported shader construct)");
        setError(GL_INVALID_OPERATION);
        return;
    };
    // The unflipped VS for FBO render-to-texture targets (no GL-origin y-flip).
    const vs_noflip = dev.createShaderModule(.{ .stage = .vertex, .code = vs.?.spirv.items }) catch {
        dev.destroyShaderModule(vs_mod);
        dev.destroyShaderModule(fs_mod);
        setProgramLog(p, "link failed: the vertex shader could not be compiled to native code for this driver (unsupported shader construct)");
        setError(GL_INVALID_OPERATION);
        return;
    };
    p.hal_dev = dev;
    p.hal_vs = vs_mod;
    p.hal_vs_noflip = vs_noflip;
    p.hal_fs = fs_mod;
    p.hal_pipeline = null; // built lazily at draw time (needs the live vertex layout)
    p.hal_pipeline_noflip = null;
    p.linked = true;

    // Resolve the per-stage default-uniform-block layout for glUniform*/glGetUniformLocation.
    // Each stage lays its block out from offset 0, so they get SEPARATE buffers (bound at
    // bindings 0=VS / 1=FS). `uniforms` lists each member once. A member declared in both stages
    // records BOTH its offsets so glUniform* writes it into both blocks.
    obj_lock.lock();
    defer obj_lock.unlock();
    for (p.uniforms.items) |u| gpa.free(u.name);
    p.uniforms.clearRetainingCapacity();
    var vs_block_size: u32 = 0;
    var fs_block_size: u32 = 0;
    // The VS members first (each gets a fresh entry with vs_offset set), then the FS members
    // (a name already present from the VS gets fs_offset set on the SAME entry; a new name is
    // appended with only fs_offset). `is_vs` selects which stage's offset field this pass sets.
    inline for (.{ vs, fs }) |maybe_shader| {
        const is_vs = maybe_shader == vs;
        if (maybe_shader) |sh| {
            for (sh.uniforms.items) |u| {
                const end = u.offset + u.float_count * @max(u.array_len, 1) * 4;
                if (is_vs) vs_block_size = @max(vs_block_size, end) else fs_block_size = @max(fs_block_size, end);
                // Find an existing entry for this name (a uniform shared between VS and FS).
                var found: ?*ProgUniform = null;
                for (p.uniforms.items) |*existing| {
                    if (std.mem.eql(u8, existing.name, u.name)) {
                        found = existing;
                        break;
                    }
                }
                if (found) |e| {
                    if (is_vs) e.vs_offset = @intCast(u.offset) else e.fs_offset = @intCast(u.offset);
                    continue;
                }
                const owned = gpa.dupe(u8, u.name) catch continue;
                p.uniforms.append(gpa, .{
                    .name = owned,
                    .vs_offset = if (is_vs) @intCast(u.offset) else -1,
                    .fs_offset = if (is_vs) -1 else @intCast(u.offset),
                    .float_count = u.float_count,
                    .mat_dim = u.mat_dim,
                    .array_len = u.array_len,
                }) catch {
                    gpa.free(owned);
                    continue;
                };
            }
        }
    }
    // Size the per-stage CPU uniform blocks (zero-initialized) the glUniform* writes target.
    p.vs_uniform_bytes.clearRetainingCapacity();
    p.vs_uniform_bytes.resize(gpa, vs_block_size) catch {};
    @memset(p.vs_uniform_bytes.items, 0);
    p.fs_uniform_bytes.clearRetainingCapacity();
    p.fs_uniform_bytes.resize(gpa, fs_block_size) catch {};
    @memset(p.fs_uniform_bytes.items, 0);
    p.uniform_dirty = true;

    // Merge the sampler uniforms (a `uniform sampler2D` in either stage). The FS samples in
    // practice. Merge both so the name -> binding resolves regardless of where it is declared.
    for (p.samplers.items) |sm| gpa.free(sm.name);
    p.samplers.clearRetainingCapacity();
    inline for (.{ vs, fs }) |maybe_shader| {
        if (maybe_shader) |sh| {
            for (sh.samplers.items) |sm| {
                var dup = false;
                for (p.samplers.items) |existing| {
                    if (std.mem.eql(u8, existing.name, sm.name)) {
                        dup = true;
                        break;
                    }
                }
                if (dup) continue;
                const owned = gpa.dupe(u8, sm.name) catch continue;
                p.samplers.append(gpa, .{ .name = owned, .binding = sm.binding, .unit = 0, .cube = sm.cube, .tex3d = sm.tex3d, .tex2darray = sm.tex2darray }) catch {
                    gpa.free(owned);
                    continue;
                };
            }
        }
    }

    // Record the program's REAL vertex attributes from the VS (the GLSL front end surfaced
    // their names + locations). glGetAttribLocation/glGetActiveAttrib + the draw-time vertex
    // layout resolve against these (a glBindAttribLocation override still wins). Empty for the
    // SPIR-V binary path (no GLSL -> no names). Those programs use glBindAttribLocation.
    for (p.attributes.items) |a| gpa.free(@constCast(a.name));
    p.attributes.clearRetainingCapacity();
    if (vs) |sh| {
        for (sh.attributes.items) |a| {
            const owned = gpa.dupe(u8, a.name) catch continue;
            p.attributes.append(gpa, .{ .name = owned, .location = a.location, .components = a.components }) catch {
                gpa.free(owned);
                continue;
            };
        }
    }

    // Record the VS output varyings (name -> location + components) for transform feedback.
    for (p.vs_outputs.items) |o| gpa.free(@constCast(o.name));
    p.vs_outputs.clearRetainingCapacity();
    if (vs) |sh| {
        for (sh.outputs.items) |o| {
            const owned = gpa.dupe(u8, o.name) catch continue;
            p.vs_outputs.append(gpa, .{ .name = owned, .location = o.location, .components = o.components }) catch {
                gpa.free(owned);
                continue;
            };
        }
    }

    // Record the named uniform blocks from each stage (tagged with that stage, so the draw
    // routes the user buffer to the VS UBO slot (0) or FS UBO slot (1)). A block name declared
    // in both stages appears once per stage (each stage binds its own slot from the same buffer).
    for (p.uniform_blocks.items) |b| {
        gpa.free(b.name);
        gpa.free(@constCast(b.members));
    }
    p.uniform_blocks.clearRetainingCapacity();
    inline for (.{ vs, fs }) |maybe_shader| {
        const stage: prism.hal.ShaderStage = if (maybe_shader == vs) .vertex else .fragment;
        if (maybe_shader) |sh| {
            for (sh.uniform_blocks.items) |b| {
                const owned = gpa.dupe(u8, b.name) catch continue;
                const owned_members = gpa.dupe(prism.glsl.UniformBlockMember, b.members) catch {
                    gpa.free(owned);
                    continue;
                };
                p.uniform_blocks.append(gpa, .{ .name = owned, .stage = stage, .binding_point = b.binding, .data_size = b.byte_size, .byte_offset = b.byte_offset, .members = owned_members }) catch {
                    gpa.free(owned);
                    gpa.free(owned_members);
                    continue;
                };
            }
        }
    }
}

fn destroyProgramHal(p: *Program) void {
    if (p.hal_dev) |dev| {
        if (p.hal_pipeline) |pl| dev.destroyPipeline(pl);
        if (p.hal_pipeline_noflip) |pl| dev.destroyPipeline(pl);
        if (p.hal_vs) |m| dev.destroyShaderModule(m);
        if (p.hal_vs_noflip) |m| dev.destroyShaderModule(m);
        if (p.hal_fs) |m| dev.destroyShaderModule(m);
        if (p.vs_uniform_hal) |ub| dev.destroyResource(ub);
        if (p.fs_uniform_hal) |ub| dev.destroyResource(ub);
    }
    p.hal_pipeline = null;
    p.hal_pipeline_noflip = null;
    p.hal_vs = null;
    p.hal_vs_noflip = null;
    p.hal_fs = null;
    p.vs_uniform_hal = null;
    p.fs_uniform_hal = null;
    p.hal_dev = null;
}

pub fn useProgram(program: GLuint) void {
    if (program != 0) {
        obj_lock.lock();
        const ok = findProgram(program) != null;
        obj_lock.unlock();
        if (!ok) {
            setError(GL_INVALID_OPERATION);
            return;
        }
    }
    current_program = program;
}

pub fn getProgramiv(program: GLuint, pname: GLenum, params: ?*GLint) void {
    const p = params orelse return;
    obj_lock.lock();
    defer obj_lock.unlock();
    const prog = findProgram(program) orelse {
        setError(GL_INVALID_OPERATION);
        return;
    };
    p.* = switch (pname) {
        GL_LINK_STATUS, GL_VALIDATE_STATUS => if (prog.linked) @as(GLint, GL_TRUE) else GL_FALSE,
        GL_DELETE_STATUS => if (prog.deleted) @as(GLint, GL_TRUE) else GL_FALSE,
        GL_ATTACHED_SHADERS => blk: {
            var c: GLint = 0;
            if (prog.vs_shader != null) c += 1;
            if (prog.fs_shader != null) c += 1;
            break :blk c;
        },
        GL_INFO_LOG_LENGTH => @intCast(prog.info_log.items.len),
        GL_ACTIVE_ATTRIBUTES => @intCast(activeAttribCount(prog)),
        GL_ACTIVE_UNIFORMS => @intCast(prog.uniforms.items.len + prog.samplers.items.len),
        GL_ACTIVE_UNIFORM_MAX_LENGTH => blk: {
            var maxn: usize = 0;
            for (prog.uniforms.items) |u| maxn = @max(maxn, u.name.len);
            for (prog.samplers.items) |sm| maxn = @max(maxn, sm.name.len);
            break :blk if (maxn == 0) 0 else @intCast(maxn + 1); // + NUL per the GL spec
        },
        GL_ACTIVE_ATTRIBUTE_MAX_LENGTH => blk: {
            var maxn: usize = 0;
            if (prog.attributes.items.len > 0) {
                for (prog.attributes.items) |a| maxn = @max(maxn, a.name.len);
            } else {
                for (prog.attrib_bindings.items) |ab| maxn = @max(maxn, ab.name.len);
            }
            break :blk if (maxn == 0) 0 else @intCast(maxn + 1);
        },
        else => {
            setError(GL_INVALID_ENUM);
            return;
        },
    };
}

/// glGetActiveUniform: enumerate the program's active uniforms by index (the default-block
/// members first, then the sampler uniforms). Reports the member's name, GLSL type, and array
/// size. An app (glmark2's Program) calls glGetProgramiv(GL_ACTIVE_UNIFORMS) for the count
/// then this per index to build a name -> location map. Without it those uniforms (e.g.
/// glmark2 build's NormalMatrix) are never discovered and stay unset (zero) -> a black frame.
pub fn getActiveUniform(program: GLuint, index: GLuint, buf_size: GLsizei, length: ?*GLint, size: ?*GLint, gl_type: ?*GLenum, name: ?[*]GLchar) void {
    obj_lock.lock();
    defer obj_lock.unlock();
    const p = findProgram(program) orelse {
        setError(GL_INVALID_OPERATION);
        return;
    };
    const nu = p.uniforms.items.len;
    const idx: usize = index;
    if (idx >= nu + p.samplers.items.len) {
        setError(GL_INVALID_VALUE);
        return;
    }
    var member_name: []const u8 = undefined;
    var ty: GLenum = GL_FLOAT;
    var arr: GLint = 1;
    if (idx < nu) {
        const u = p.uniforms.items[idx];
        member_name = u.name;
        arr = @intCast(@max(u.array_len, 1));
        ty = uniformGlType(u);
    } else {
        const sm = p.samplers.items[idx - nu];
        member_name = sm.name;
        ty = GL_SAMPLER_2D;
        arr = 1;
    }
    if (size) |sp| sp.* = arr;
    if (gl_type) |tp| tp.* = ty;
    writeName(member_name, buf_size, length, name);
}

/// glGetActiveAttrib: enumerate the program's active vertex attributes by index. The attribute
/// set is the app's glBindAttribLocation bindings (the names + locations it declared). The type
/// is reported as a float vector (the only attribute kind this frontend feeds). The exact
/// component count is not tracked per-attribute, so vec4 is reported (apps key off the location).
pub fn getActiveAttrib(program: GLuint, index: GLuint, buf_size: GLsizei, length: ?*GLint, size: ?*GLint, gl_type: ?*GLenum, name: ?[*]GLchar) void {
    obj_lock.lock();
    defer obj_lock.unlock();
    const p = findProgram(program) orelse {
        setError(GL_INVALID_OPERATION);
        return;
    };
    const idx: usize = index;
    // Enumerate the program's REAL attributes when known (the GLSL path). Fall back to the
    // glBindAttribLocation list for the SPIR-V binary path.
    if (p.attributes.items.len > 0) {
        if (idx >= p.attributes.items.len) {
            setError(GL_INVALID_VALUE);
            return;
        }
        const a = p.attributes.items[idx];
        if (size) |sp| sp.* = 1;
        if (gl_type) |tp| tp.* = attribGlType(a.components);
        writeName(a.name, buf_size, length, name);
        return;
    }
    if (idx >= p.attrib_bindings.items.len) {
        setError(GL_INVALID_VALUE);
        return;
    }
    const ab = p.attrib_bindings.items[idx];
    if (size) |sp| sp.* = 1;
    if (gl_type) |tp| tp.* = GL_FLOAT_VEC4;
    writeName(ab.name, buf_size, length, name);
}

/// The number of active attributes a program reports (the real GLSL attribute list when
/// known, else the glBindAttribLocation bindings for the SPIR-V binary path).
fn activeAttribCount(p: *Program) usize {
    return if (p.attributes.items.len > 0) p.attributes.items.len else p.attrib_bindings.items.len;
}

/// The GL_FLOAT_VECn type enum for an attribute's component count (glGetActiveAttrib).
fn attribGlType(components: u8) GLenum {
    return switch (components) {
        1 => GL_FLOAT,
        2 => GL_FLOAT_VEC2,
        3 => GL_FLOAT_VEC3,
        else => GL_FLOAT_VEC4,
    };
}

/// The GL type enum for a resolved default-block uniform member (a matrix by its dimension, or
/// a float vector by its float count). Used by glGetActiveUniform.
fn uniformGlType(u: ProgUniform) GLenum {
    if (u.mat_dim == 2) return GL_FLOAT_MAT2;
    if (u.mat_dim == 3) return GL_FLOAT_MAT3;
    if (u.mat_dim == 4) return GL_FLOAT_MAT4;
    return switch (u.float_count) {
        1 => GL_FLOAT,
        2 => GL_FLOAT_VEC2,
        3 => GL_FLOAT_VEC3,
        else => GL_FLOAT_VEC4,
    };
}

/// Copy `src` into the caller's `name` buffer (capped to `buf_size - 1` + a NUL) and report the
/// written length (excluding the NUL) in `length`, matching the glGetActive* name convention.
fn writeName(src: []const u8, buf_size: GLsizei, length: ?*GLint, name: ?[*]GLchar) void {
    if (name) |out| {
        if (buf_size > 0) {
            const cap: usize = @intCast(buf_size - 1);
            const n = @min(src.len, cap);
            @memcpy(out[0..n], src[0..n]);
            out[n] = 0;
            if (length) |lp| lp.* = @intCast(n);
            return;
        }
    }
    if (length) |lp| lp.* = 0;
}

/// glGetProgramInfoLog: programs link successfully or fail silently here (the SPIR-V/GLSL
/// diagnostics live on the shaders), so the program info log is empty. Write a NUL.
pub fn getProgramInfoLog(program: GLuint, buf_size: GLsizei, length: ?*GLint, info_log: ?[*]GLchar) void {
    obj_lock.lock();
    const log: []const u8 = if (findProgram(program)) |p| p.info_log.items else &.{};
    // Copy the diagnostic (it already includes a trailing NUL when non-empty), truncated to
    // the caller's buffer. `length` is the count EXCLUDING the NUL, per the GL spec.
    if (buf_size > 0) {
        if (info_log) |out| {
            if (log.len == 0) {
                out[0] = 0;
                if (length) |lp| lp.* = 0;
            } else {
                const n = @min(@as(usize, @intCast(buf_size)) - 1, log.len - 1); // leave room for NUL, drop the stored NUL
                @memcpy(out[0..n], log[0..n]);
                out[n] = 0;
                if (length) |lp| lp.* = @intCast(n);
            }
        }
    } else if (length) |lp| lp.* = 0;
    obj_lock.unlock();
}

pub fn deleteProgram(program: GLuint) void {
    if (program == 0) return;
    flushBatch(); // a pending draw may reference this program's cached pipeline/shaders
    obj_lock.lock();
    defer obj_lock.unlock();
    for (programs.items, 0..) |p, idx| {
        if (p.id == program) {
            destroyProgramHal(p);
            for (p.uniforms.items) |u| gpa.free(u.name);
            p.uniforms.deinit(gpa);
            p.vs_uniform_bytes.deinit(gpa);
            p.fs_uniform_bytes.deinit(gpa);
            for (p.attrib_bindings.items) |ab| gpa.free(ab.name);
            p.attrib_bindings.deinit(gpa);
            for (p.attributes.items) |a| gpa.free(@constCast(a.name));
            p.attributes.deinit(gpa);
            for (p.vs_outputs.items) |o| gpa.free(@constCast(o.name));
            p.vs_outputs.deinit(gpa);
            for (p.tf_varyings.items) |name| gpa.free(name);
            p.tf_varyings.deinit(gpa);
            for (p.samplers.items) |sm| gpa.free(sm.name);
            p.samplers.deinit(gpa);
            for (p.uniform_blocks.items) |b| {
                gpa.free(b.name);
                gpa.free(@constCast(b.members));
            }
            p.uniform_blocks.deinit(gpa);
            p.info_log.deinit(gpa);
            gpa.destroy(p);
            _ = programs.swapRemove(idx);
            if (current_program == program) current_program = 0;
            return;
        }
    }
}

/// glGetAttribLocation: the M3 triangle binds position to location 0 and color to
/// location 1 (matching the SPIR-V shaders' `layout(location=...)`). We return the
/// conventional location by attribute name so a client that asks resolves correctly.
pub fn getAttribLocation(program: GLuint, name: ?[*:0]const GLchar) GLint {
    obj_lock.lock();
    defer obj_lock.unlock();
    const p = findProgram(program) orelse {
        setError(GL_INVALID_OPERATION);
        return -1;
    };
    const n = std.mem.span(name orelse return -1);
    // An app-set glBindAttribLocation binding wins (es2gears binds position=0, normal=1).
    for (p.attrib_bindings.items) |ab| {
        if (std.mem.eql(u8, ab.name, n)) return @intCast(ab.location);
    }
    // The program's REAL attributes (from the GLSL front end). Resolving by NAME is the
    // correct, non-heuristic mapping: glmark2's texture scenes declare position/normal/texcoord,
    // and `texcoord` (which the old substring heuristic returned -1 for) must bind so the texture
    // coordinates reach the FS - otherwise texture2D samples garbage and the scene is black.
    if (p.attributes.items.len > 0) {
        for (p.attributes.items) |a| {
            if (std.mem.eql(u8, a.name, n)) return @intCast(a.location);
        }
        // A linked GLSL program with a known attribute list: an unmatched name is not an
        // attribute (e.g. a uniform/sampler queried via glGetAttribLocation -> -1).
        return -1;
    }
    // --- Fallback for the SPIR-V binary path (no GLSL source -> no attribute names) ---
    // A name that is a UNIFORM/SAMPLER is never a vertex attribute. glmark2 queries
    // glGetAttribLocation BEFORE glGetUniformLocation and treats a >=0 result as an
    // attribute - so the substring heuristic below must not shadow a uniform. Without
    // this, "NormalMatrix" matched the "normal" attribute heuristic, glmark2 took it as an
    // attribute and never uploaded the NormalMatrix uniform -> zero -> black lit scenes.
    for (p.uniforms.items) |u| if (std.mem.eql(u8, u.name, n)) return -1;
    for (p.samplers.items) |sm| if (std.mem.eql(u8, sm.name, n)) return -1;
    // Heuristic name->location mapping for the gradient triangle's attributes.
    if (std.mem.indexOf(u8, n, "position") != null or std.mem.indexOf(u8, n, "Pos") != null or std.mem.indexOf(u8, n, "pos") != null) return 0;
    if (std.mem.indexOf(u8, n, "normal") != null or std.mem.indexOf(u8, n, "Normal") != null) return 1;
    if (std.mem.indexOf(u8, n, "color") != null or std.mem.indexOf(u8, n, "Color") != null or std.mem.indexOf(u8, n, "col") != null) return 1;
    return -1;
}

/// glBindAttribLocation: bind a vertex attribute name to a generic attribute index. Per GL,
/// call before glLinkProgram. Recorded on the program and honored in getAttribLocation (and
/// the VS lowering assigns attributes in declaration order, which es2gears matches:
/// position decl 0 -> loc 0, normal decl 1 -> loc 1).
pub fn bindAttribLocation(program: GLuint, index: GLuint, name: ?[*:0]const GLchar) void {
    if (index >= MAX_ATTRIBS) {
        setError(GL_INVALID_VALUE);
        return;
    }
    const nm = std.mem.span(name orelse return);
    obj_lock.lock();
    defer obj_lock.unlock();
    const p = findProgram(program) orelse {
        setError(GL_INVALID_OPERATION);
        return;
    };
    // Replace an existing binding for the same name, else append.
    for (p.attrib_bindings.items) |*ab| {
        if (std.mem.eql(u8, ab.name, nm)) {
            ab.location = index;
            return;
        }
    }
    const owned = gpa.dupe(u8, nm) catch {
        setError(GL_INVALID_OPERATION);
        return;
    };
    p.attrib_bindings.append(gpa, .{ .name = owned, .location = index }) catch {
        gpa.free(owned);
        setError(GL_INVALID_OPERATION);
    };
}

// --- Uniforms (the default uniform block -> a HAL UBO) ----------------------

/// glGetUniformLocation: resolve a default-block uniform name to its location. We encode
/// the location as the uniform member's index in the program's resolved layout (0-based),
/// so glUniform* can find the member's byte offset. Returns -1 for an unknown name.
pub fn getUniformLocation(program: GLuint, name: ?[*:0]const GLchar) GLint {
    obj_lock.lock();
    defer obj_lock.unlock();
    const p = findProgram(program) orelse {
        setError(GL_INVALID_OPERATION);
        return -1;
    };
    const n = std.mem.span(name orelse return -1);
    for (p.uniforms.items, 0..) |u, i| {
        if (std.mem.eql(u8, u.name, n)) return @intCast(i);
    }
    // A `uniform sampler2D` name: return its sampler index with the high bit flagged so
    // glUniform1i routes to the texture-unit selection (not the UBO block).
    for (p.samplers.items, 0..) |sm, i| {
        if (std.mem.eql(u8, sm.name, n)) return SAMPLER_LOCATION_FLAG | @as(GLint, @intCast(i));
    }
    return -1;
}

/// Write `floats` into the current program's uniform block at the member identified by
/// `location` (the index returned by glGetUniformLocation). `expect_floats` is the member's
/// expected float count (validated against the resolved layout). Marks the block dirty so
/// the next draw re-uploads it to the HAL UBO. A negative location is silently ignored
/// (GL: glUniform* with location -1 is a no-op).
fn writeUniform(location: GLint, floats: []const f32) void {
    if (location < 0) return;
    if (current_program == 0) {
        setError(GL_INVALID_OPERATION);
        return;
    }
    obj_lock.lock();
    defer obj_lock.unlock();
    const p = findProgram(current_program) orelse {
        setError(GL_INVALID_OPERATION);
        return;
    };
    const idx: usize = @intCast(location);
    if (idx >= p.uniforms.items.len) {
        setError(GL_INVALID_OPERATION);
        return;
    }
    const m = p.uniforms.items[idx];
    // The member's total capacity in floats: per-element stride * array length (tight, no
    // std140 padding, matching the front end's block layout). A non-array uniform has
    // array_len == 1, so this is just float_count. An array uniform (`vec3 c[N]`) accepts up
    // to N*float_count floats so glUniform*fv(loc, count, ptr) for count up to N lands each
    // element at base + i*stride.
    const capacity = m.float_count * @max(m.array_len, 1);
    const n = @min(floats.len, capacity);
    // Write into every stage's block that declares this member (a member shared by VS+FS lives
    // in both, at each stage's own offset). The two blocks bind at separate slots at draw time.
    if (m.vs_offset >= 0) writeUniformBytes(&p.vs_uniform_bytes, @intCast(m.vs_offset), floats[0..n]);
    if (m.fs_offset >= 0) writeUniformBytes(&p.fs_uniform_bytes, @intCast(m.fs_offset), floats[0..n]);
    p.uniform_dirty = true;
}

/// Copy `floats` into a stage's CPU uniform block at byte `off` (4 bytes each), bounded by the
/// block size. Out-of-range writes are clamped (a malformed offset never escapes the block).
fn writeUniformBytes(bytes: *std.ArrayListUnmanaged(u8), off: usize, floats: []const f32) void {
    var i: usize = 0;
    while (i < floats.len and off + (i + 1) * 4 <= bytes.items.len) : (i += 1) {
        @memcpy(bytes.items[off + i * 4 ..][0..4], &std.mem.toBytes(floats[i]));
    }
}

/// Lazily (re)build a stage's HAL UBO from its CPU block bytes and return it (null when the
/// stage has no default-block uniforms). `dirty` forces a re-upload of the CPU bytes. Caller
/// holds obj_lock (the program's HAL fields are touched).
fn buildStageUbo(dev: prism.hal.Device, bytes: *std.ArrayListUnmanaged(u8), hal_slot: *?*prism.hal.Resource, dirty: bool) prism.Error!?*prism.hal.Resource {
    if (bytes.items.len == 0) return null;
    var rebuilt = false;
    if (hal_slot.* == null) {
        hal_slot.* = try dev.createResource(.{ .buffer = .{ .size = bytes.items.len, .usage = .{ .uniform = true } } });
        rebuilt = true;
    }
    if (dirty or rebuilt) {
        const map = try dev.mapResource(hal_slot.*.?);
        @memcpy(map[0..bytes.items.len], bytes.items);
    }
    return hal_slot.*.?;
}

// --- Named uniform blocks (GLES3 UBOs: glBindBufferBase, glGetUniformBlockIndex) --------

/// Lazily (re)build the user buffer `b`'s HAL UBO Resource and re-upload its bytes from the
/// buffer object storage, returning it. Null for an empty buffer. Caller holds obj_lock.
///
/// `members` is the block's std140<->tight repack table (one entry per member; empty only for a
/// block with no members, or the SPIR-V-binary path with no GLSL block). When non-empty the app
/// filled the buffer per std140 (vec3 16-align, vec2 8-align, arrays 16-stride, mat columns
/// 16-stride) but the shader reads the members tight-packed. Gather each member from its std140
/// offset into its tight offset instead of a straight copy (an identity copy for all-16-byte
/// members). The gap-filling zero-init keeps any tight padding deterministic.
fn ensureUserUboHal(dev: prism.hal.Device, b: *Buffer, members: []const prism.glsl.UniformBlockMember, base_off: usize) prism.Error!?*prism.hal.Resource {
    if (b.bytes.items.len == 0 or base_off >= b.bytes.items.len) return null;
    if (b.ubo_hal == null) {
        b.ubo_hal = try dev.createResource(.{ .buffer = .{ .size = b.bytes.items.len, .usage = .{ .uniform = true } } });
        b.ubo_hal_dev = dev;
    }
    const map = try dev.mapResource(b.ubo_hal.?);
    // glBindBufferRange: the block starts at `base_off` into the buffer (0 for glBindBufferBase).
    // The members' std140 offsets are relative to this block start.
    const src = b.bytes.items[base_off..];
    if (members.len == 0) {
        // std140 == tight (all 16-byte members): the user buffer feeds the shader unchanged.
        @memcpy(map[0..src.len], src);
        return b.ubo_hal.?;
    }
    // std140 -> tight repack. Zero the destination first (leaves tight padding well-defined),
    // then gather each member's contiguous chunks from the user buffer's std140 offsets.
    @memset(map[0..src.len], 0);
    for (members) |m| {
        var u: u32 = 0;
        while (u < m.unit_count) : (u += 1) {
            const s0 = m.std140_offset + u * m.std140_stride;
            const d0 = m.tight_offset + u * m.tight_stride;
            // Clamp defensively: a short user buffer must not over-read/over-write.
            if (s0 + m.copy_bytes > src.len or d0 + m.copy_bytes > map.len) continue;
            @memcpy(map[d0 .. d0 + m.copy_bytes], src[s0 .. s0 + m.copy_bytes]);
        }
    }
    return b.ubo_hal.?;
}

/// If program `p` declares a named uniform block for `stage`, resolve the buffer bound at that
/// block's GL binding point (glBindBufferBase) into a HAL UBO Resource, else null. Caller holds
/// obj_lock. The first block for the stage wins (the minimal path: one block per stage).
fn resolveNamedStageUbo(dev: prism.hal.Device, p: *Program, stage: prism.hal.ShaderStage) ?*prism.hal.Resource {
    for (p.uniform_blocks.items) |blk| {
        if (blk.stage != stage) continue;
        if (blk.binding_point >= MAX_UNIFORM_BUFFER_BINDINGS) return null;
        const buf_id = uniform_buffer_bindings[blk.binding_point];
        if (buf_id == 0) return null;
        const b = findBuffer(buf_id) orelse return null;
        const off: usize = @intCast(uniform_buffer_offsets[blk.binding_point]);
        return ensureUserUboHal(dev, b, blk.members, off) catch null;
    }
    return null;
}

/// glGetUniformBlockIndex(program, name): the index of a named uniform block, or GL_INVALID_INDEX.
pub fn getUniformBlockIndex(program: GLuint, name: ?[*:0]const GLchar) GLuint {
    obj_lock.lock();
    defer obj_lock.unlock();
    const p = findProgram(program) orelse {
        setError(GL_INVALID_OPERATION);
        return GL_INVALID_INDEX;
    };
    const n = std.mem.span(name orelse return GL_INVALID_INDEX);
    for (p.uniform_blocks.items, 0..) |blk, i| {
        if (std.mem.eql(u8, blk.name, n)) return @intCast(i);
    }
    return GL_INVALID_INDEX;
}

/// glUniformBlockBinding(program, blockIndex, binding): retarget a block's GL binding point (the
/// glBindBufferBase point it reads from at draw).
pub fn uniformBlockBinding(program: GLuint, block_index: GLuint, binding: GLuint) void {
    obj_lock.lock();
    defer obj_lock.unlock();
    const p = findProgram(program) orelse {
        setError(GL_INVALID_OPERATION);
        return;
    };
    if (block_index >= p.uniform_blocks.items.len) {
        setError(GL_INVALID_VALUE);
        return;
    }
    p.uniform_blocks.items[block_index].binding_point = binding;
}

/// glGetActiveUniformBlockiv(program, blockIndex, pname, params): query a block's std140 data
/// size (GL_UNIFORM_BLOCK_DATA_SIZE) or its current binding point (GL_UNIFORM_BLOCK_BINDING).
pub fn getActiveUniformBlockiv(program: GLuint, block_index: GLuint, pname: GLenum, params: ?[*]GLint) void {
    const out = params orelse return;
    obj_lock.lock();
    defer obj_lock.unlock();
    const p = findProgram(program) orelse {
        setError(GL_INVALID_OPERATION);
        return;
    };
    if (block_index >= p.uniform_blocks.items.len) {
        setError(GL_INVALID_VALUE);
        return;
    }
    const blk = p.uniform_blocks.items[block_index];
    switch (pname) {
        GL_UNIFORM_BLOCK_DATA_SIZE => out[0] = @intCast(blk.data_size),
        GL_UNIFORM_BLOCK_BINDING => out[0] = @intCast(blk.binding_point),
        // The number of members the block declares (each is a queryable "active uniform").
        GL_UNIFORM_BLOCK_ACTIVE_UNIFORMS => out[0] = @intCast(blk.members.len),
        // The program-wide uniform indices of this block's members, written into `params[]`
        // (the caller sizes it to GL_UNIFORM_BLOCK_ACTIVE_UNIFORMS). The flattened index of
        // member `mi` is the count of all earlier blocks' members plus `mi`.
        GL_UNIFORM_BLOCK_ACTIVE_UNIFORM_INDICES => {
            var base: usize = 0;
            for (p.uniform_blocks.items[0..block_index]) |b| base += b.members.len;
            for (0..blk.members.len) |mi| out[mi] = @intCast(base + mi);
        },
        else => setError(GL_INVALID_ENUM),
    }
}

/// Resolve the program's default-block uniform that carries a block member's name + GL type.
/// A block member lowers to a default-block uniform whose absolute byte offset is
/// `block.byte_offset + member.tight_offset`. Match it against the stage's offset field.
fn progUniformForBlockMember(p: *Program, blk: ProgUniformBlock, m: prism.glsl.UniformBlockMember) ?*ProgUniform {
    const abs: i32 = @intCast(blk.byte_offset + m.tight_offset);
    for (p.uniforms.items) |*u| {
        const so = if (blk.stage == .vertex) u.vs_offset else u.fs_offset;
        if (so == abs) return u;
    }
    return null;
}

/// Resolve a flattened block-member "uniform index" (block order, then member order within a
/// block) to its owning block index, the member record, and the default-block uniform carrying
/// its name/type. Returns null if the index is out of range.
const BlockMemberRef = struct {
    block_index: usize,
    blk: ProgUniformBlock,
    member: prism.glsl.UniformBlockMember,
    uni: ?*ProgUniform,
};
fn resolveBlockMemberIndex(p: *Program, index: usize) ?BlockMemberRef {
    var running: usize = 0;
    for (p.uniform_blocks.items, 0..) |blk, bi| {
        if (index < running + blk.members.len) {
            const m = blk.members[index - running];
            return .{ .block_index = bi, .blk = blk, .member = m, .uni = progUniformForBlockMember(p, blk, m) };
        }
        running += blk.members.len;
    }
    return null;
}

/// glGetUniformIndices(program, count, names, indices): for each requested uniform name, its
/// program-wide uniform index (a flattened index into the named blocks' members, block order
/// then member order), or GL_INVALID_INDEX if no block member has that name. The uniform index
/// this returns is exactly what glGetActiveUniformsiv accepts.
pub fn getUniformIndices(program: GLuint, count: GLsizei, names: ?[*]const ?[*:0]const GLchar, indices: ?[*]GLuint) void {
    const name_arr = names orelse return;
    const out = indices orelse return;
    obj_lock.lock();
    defer obj_lock.unlock();
    const p = findProgram(program) orelse {
        setError(GL_INVALID_OPERATION);
        return;
    };
    const n: usize = if (count < 0) 0 else @intCast(count);
    for (0..n) |i| {
        const req = std.mem.span(name_arr[i] orelse {
            out[i] = GL_INVALID_INDEX;
            continue;
        });
        out[i] = GL_INVALID_INDEX;
        var running: usize = 0;
        outer: for (p.uniform_blocks.items) |blk| {
            for (blk.members) |m| {
                if (progUniformForBlockMember(p, blk, m)) |u| {
                    if (std.mem.eql(u8, u.name, req)) {
                        out[i] = @intCast(running);
                        break :outer;
                    }
                }
                running += 1;
            }
        }
    }
}

/// glGetActiveUniformsiv(program, count, indices, pname, params): for each uniform index (a
/// flattened named-block member index, see getUniformIndices) return the requested property.
pub fn getActiveUniformsiv(program: GLuint, count: GLsizei, indices: ?[*]const GLuint, pname: GLenum, params: ?[*]GLint) void {
    const idx_arr = indices orelse return;
    const out = params orelse return;
    obj_lock.lock();
    defer obj_lock.unlock();
    const p = findProgram(program) orelse {
        setError(GL_INVALID_OPERATION);
        return;
    };
    const n: usize = if (count < 0) 0 else @intCast(count);
    for (0..n) |i| {
        const ref = resolveBlockMemberIndex(p, idx_arr[i]) orelse {
            setError(GL_INVALID_VALUE);
            return;
        };
        const is_matrix = if (ref.uni) |u| u.mat_dim != 0 else false;
        const array_len: u32 = if (ref.uni) |u| @max(u.array_len, 1) else 1;
        out[i] = switch (pname) {
            GL_UNIFORM_TYPE => @intCast(if (ref.uni) |u| uniformGlType(u.*) else GL_FLOAT),
            GL_UNIFORM_SIZE => @intCast(array_len),
            GL_UNIFORM_OFFSET => @intCast(ref.member.std140_offset),
            GL_UNIFORM_BLOCK_INDEX => @intCast(ref.block_index),
            // std140 array stride is 16 for an array member, else 0 (a non-array reports no
            // stride; GL permits -1, but 0 is unambiguous here since 0 is never a valid stride).
            GL_UNIFORM_ARRAY_STRIDE => if (array_len > 1) @intCast(ref.member.std140_stride) else 0,
            // std140 matrix column stride is 16 for a matrix member, else 0.
            GL_UNIFORM_MATRIX_STRIDE => if (is_matrix) 16 else 0,
            else => {
                setError(GL_INVALID_ENUM);
                return;
            },
        };
    }
}

/// glBindBufferBase(target, index, buffer): bind a whole buffer object to an indexed binding
/// point. For GL_UNIFORM_BUFFER this records the buffer at uniform-buffer binding point `index`
/// (a named block whose binding point is `index` reads from it at draw). Also updates the
/// generic GL_UNIFORM_BUFFER binding, per GL.
pub fn bindBufferBase(target: GLenum, index: GLuint, buffer: GLuint) void {
    if (target == GL_TRANSFORM_FEEDBACK_BUFFER) {
        if (index >= MAX_TRANSFORM_FEEDBACK_BUFFERS) {
            setError(GL_INVALID_VALUE);
            return;
        }
        transform_feedback_bindings[index] = buffer;
        bound_transform_feedback_buffer = buffer; // per GL, the generic binding is set too
        return;
    }
    if (target != GL_UNIFORM_BUFFER) {
        setError(GL_INVALID_ENUM);
        return;
    }
    if (index >= MAX_UNIFORM_BUFFER_BINDINGS) {
        setError(GL_INVALID_VALUE);
        return;
    }
    uniform_buffer_bindings[index] = buffer;
    uniform_buffer_offsets[index] = 0; // BindBufferBase = the whole buffer (offset 0)
    bound_uniform_buffer = buffer;
}

/// glTransformFeedbackVaryings(program, count, varyings, bufferMode): record which VS output
/// varyings a transform-feedback capture writes, in capture order, and the buffer mode. Per spec
/// this is called BEFORE glLinkProgram. The capture resolves each name against the linked VS's
/// output varyings at draw time. Only GL_INTERLEAVED_ATTRIBS is honored end to end here (all
/// varyings tightly packed into one buffer). With GL_SEPARATE_ATTRIBS each varying is captured into
/// its own buffer (the buffer bound at GL_TRANSFORM_FEEDBACK_BUFFER binding point i).
pub fn transformFeedbackVaryings(program: GLuint, count: GLsizei, varyings: ?[*]const ?[*:0]const GLchar, buffer_mode: GLenum) void {
    if (count < 0) {
        setError(GL_INVALID_VALUE);
        return;
    }
    if (buffer_mode != GL_INTERLEAVED_ATTRIBS and buffer_mode != GL_SEPARATE_ATTRIBS) {
        setError(GL_INVALID_ENUM);
        return;
    }
    // GL_SEPARATE_ATTRIBS puts one varying per buffer, so the count may not exceed the number of
    // separate-mode capture buffers (GL_MAX_TRANSFORM_FEEDBACK_SEPARATE_ATTRIBS).
    if (buffer_mode == GL_SEPARATE_ATTRIBS and count > MAX_TRANSFORM_FEEDBACK_BUFFERS) {
        setError(GL_INVALID_VALUE);
        return;
    }
    obj_lock.lock();
    defer obj_lock.unlock();
    const p = findProgram(program) orelse {
        setError(GL_INVALID_VALUE);
        return;
    };
    for (p.tf_varyings.items) |name| gpa.free(name);
    p.tf_varyings.clearRetainingCapacity();
    p.tf_buffer_mode = buffer_mode;
    const list = varyings orelse return;
    var i: usize = 0;
    while (i < @as(usize, @intCast(count))) : (i += 1) {
        const cname = list[i] orelse continue;
        const owned = gpa.dupe(u8, std.mem.span(cname)) catch {
            setError(GL_OUT_OF_MEMORY);
            return;
        };
        p.tf_varyings.append(gpa, owned) catch {
            gpa.free(owned);
            setError(GL_OUT_OF_MEMORY);
            return;
        };
    }
}

/// glBeginTransformFeedback(primitiveMode): open a capture span. Subsequent draws capture the
/// current program's recorded varyings into the bound GL_TRANSFORM_FEEDBACK_BUFFER, appending at a
/// running offset that resets here. Must be paired with glEndTransformFeedback.
pub fn beginTransformFeedback(primitive_mode: GLenum) void {
    flushBatch(); // start the capture span cleanly after prior draws
    if (tf_active) {
        setError(GL_INVALID_OPERATION); // already active
        return;
    }
    tf_active = true;
    tf_paused = false;
    tf_primitive_mode = primitive_mode;
    tf_write_offsets = [_]usize{0} ** MAX_TRANSFORM_FEEDBACK_BUFFERS;
}

/// glEndTransformFeedback(): close the capture span.
pub fn endTransformFeedback() void {
    flushBatch(); // captured draws must land before the buffer is read
    if (!tf_active) {
        setError(GL_INVALID_OPERATION);
        return;
    }
    tf_active = false;
    tf_paused = false;
}

/// glPauseTransformFeedback(): pause the active capture span. While paused a draw still rasterizes
/// (unless GL_RASTERIZER_DISCARD) but does NOT capture. The per-buffer write cursors hold so a later
/// glResumeTransformFeedback appends after them. Error if TF is not active or already paused.
pub fn pauseTransformFeedback() void {
    if (!tf_active or tf_paused) {
        setError(GL_INVALID_OPERATION);
        return;
    }
    tf_paused = true;
}

/// glResumeTransformFeedback(): resume a paused capture span. Subsequent draws capture again,
/// appending at the held cursors. Error if TF is not active or not currently paused.
pub fn resumeTransformFeedback() void {
    if (!tf_active or !tf_paused) {
        setError(GL_INVALID_OPERATION);
        return;
    }
    tf_paused = false;
}

/// glBindBufferRange(target, index, buffer, offset, size): bind a sub-range of `buffer` starting at
/// `offset` to the uniform binding point. Useful for dynamic UBO streaming or packing several blocks
/// in one buffer. `size` is advisory here (the block's std140 data_size bounds the read). `offset`
/// must be >= 0. GLES also requires it be a multiple of GL_UNIFORM_BUFFER_OFFSET_ALIGNMENT, which
/// Prism advertises as 1, so any offset is legal.
pub fn bindBufferRange(target: GLenum, index: GLuint, buffer: GLuint, offset: GLintptr, size: GLsizeiptr) void {
    if (target != GL_UNIFORM_BUFFER) return setError(GL_INVALID_ENUM);
    if (index >= MAX_UNIFORM_BUFFER_BINDINGS) return setError(GL_INVALID_VALUE);
    if (offset < 0 or size < 0) return setError(GL_INVALID_VALUE);
    uniform_buffer_bindings[index] = buffer;
    uniform_buffer_offsets[index] = offset;
    bound_uniform_buffer = buffer;
}

pub fn uniform1f(location: GLint, v0: GLfloat) void {
    writeUniform(location, &.{v0});
}
pub fn uniform2f(location: GLint, v0: GLfloat, v1: GLfloat) void {
    writeUniform(location, &.{ v0, v1 });
}
pub fn uniform3f(location: GLint, v0: GLfloat, v1: GLfloat, v2: GLfloat) void {
    writeUniform(location, &.{ v0, v1, v2 });
}
pub fn uniform4f(location: GLint, v0: GLfloat, v1: GLfloat, v2: GLfloat, v3: GLfloat) void {
    writeUniform(location, &.{ v0, v1, v2, v3 });
}
/// glUniform1i: an integer uniform. A sampler uniform (its location carries
/// SAMPLER_LOCATION_FLAG) selects the texture unit the sampler reads (`glUniform1i(loc,
/// unit)`). At draw the texture bound to that unit binds to the HAL at the sampler's
/// binding. A non-sampler int uniform is stored in the default block as a float (the
/// scalarized common subset).
pub fn uniform1i(location: GLint, v0: GLint) void {
    if (location < 0) return;
    if ((location & SAMPLER_LOCATION_FLAG) != 0) {
        if (current_program == 0) {
            setError(GL_INVALID_OPERATION);
            return;
        }
        const idx: usize = @intCast(location & ~SAMPLER_LOCATION_FLAG);
        obj_lock.lock();
        defer obj_lock.unlock();
        const p = findProgram(current_program) orelse {
            setError(GL_INVALID_OPERATION);
            return;
        };
        if (idx >= p.samplers.items.len) {
            setError(GL_INVALID_OPERATION);
            return;
        }
        p.samplers.items[idx].unit = v0;
        return;
    }
    writeUniform(location, &.{@floatFromInt(v0)});
}

fn uniformFv(location: GLint, count: GLsizei, comps: usize, value: ?[*]const GLfloat) void {
    if (count < 0 or value == null) {
        if (count < 0) setError(GL_INVALID_VALUE);
        return;
    }
    // `count` is the number of array elements (1 for a non-array uniform). Write all
    // `count * comps` floats contiguously. writeUniform caps to the member's array capacity
    // and lays element `i` at base + i*stride (the front end packs array elements tightly).
    const total = @as(usize, @intCast(count)) * comps;
    const v = value.?[0..total];
    writeUniform(location, v);
}
pub fn uniform1fv(location: GLint, count: GLsizei, value: ?[*]const GLfloat) void {
    uniformFv(location, count, 1, value);
}
pub fn uniform2fv(location: GLint, count: GLsizei, value: ?[*]const GLfloat) void {
    uniformFv(location, count, 2, value);
}
pub fn uniform3fv(location: GLint, count: GLsizei, value: ?[*]const GLfloat) void {
    uniformFv(location, count, 3, value);
}
pub fn uniform4fv(location: GLint, count: GLsizei, value: ?[*]const GLfloat) void {
    uniformFv(location, count, 4, value);
}

/// glUniformMatrix{2,3,4}fv: write a column-major matrix uniform. `transpose` must be
/// GL_FALSE in GLES2. If GL_TRUE, transposes into column-major before storing (the SPIR-V
/// matrix lowering reads column-major). es2gears passes GL_FALSE.
fn uniformMatrixFv(location: GLint, dim: usize, count: GLsizei, transpose: GLboolean, value: ?[*]const GLfloat) void {
    if (count < 0 or value == null) {
        if (count < 0) setError(GL_INVALID_VALUE);
        return;
    }
    const n = dim * dim;
    const src = value.?[0..n];
    if (transpose == 0) {
        writeUniform(location, src);
    } else {
        var tmp: [16]f32 = undefined;
        var c: usize = 0;
        while (c < dim) : (c += 1) {
            var r: usize = 0;
            while (r < dim) : (r += 1) {
                // src is row-major (transpose=TRUE). Store column-major: dst[col*dim+row].
                tmp[c * dim + r] = src[r * dim + c];
            }
        }
        writeUniform(location, tmp[0..n]);
    }
}
pub fn uniformMatrix2fv(location: GLint, count: GLsizei, transpose: GLboolean, value: ?[*]const GLfloat) void {
    uniformMatrixFv(location, 2, count, transpose, value);
}
pub fn uniformMatrix3fv(location: GLint, count: GLsizei, transpose: GLboolean, value: ?[*]const GLfloat) void {
    uniformMatrixFv(location, 3, count, transpose, value);
}
pub fn uniformMatrix4fv(location: GLint, count: GLsizei, transpose: GLboolean, value: ?[*]const GLfloat) void {
    uniformMatrixFv(location, 4, count, transpose, value);
}

// --- glDrawArrays: the triangle ---------------------------------------------

/// Map a GL_FLOAT attribute of `size` components to the HAL vertex-attribute format.
fn attribFormat(size: GLint) ?prism.hal.Format {
    return switch (size) {
        1 => .r32_float,
        2 => .r32g32_float,
        3 => .r32g32b32_float,
        4 => .r32g32b32a32_float,
        else => null,
    };
}

/// Bytes per component of a vertex-attribute source type (0 = unsupported type).
fn attribTypeSize(gl_type: GLenum) u32 {
    return switch (gl_type) {
        GL_BYTE, GL_UNSIGNED_BYTE => 1,
        GL_SHORT, GL_UNSIGNED_SHORT, GL_HALF_FLOAT, GL_HALF_FLOAT_OES => 2,
        GL_INT, GL_UNSIGNED_INT, GL_FLOAT, GL_FIXED => 4,
        else => 0,
    };
}

/// Read one attribute component of `gl_type` at byte `off` of `src` and convert to f32 (the format
/// the software VS reads). Integer types either NORMALIZE (unorm 0..1 / snorm -1..1) or cast. Float
/// / half / fixed convert directly. Packed attributes (glTF byte colors, short positions) ride this.
fn readAttribF32(src: []const u8, off: usize, gl_type: GLenum, normalized: bool) f32 {
    switch (gl_type) {
        GL_FLOAT => return std.mem.bytesToValue(f32, src[off..][0..4]),
        GL_HALF_FLOAT, GL_HALF_FLOAT_OES => return @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, src[off..][0..2], .little)))),
        GL_FIXED => return @as(f32, @floatFromInt(@as(i32, @bitCast(std.mem.readInt(u32, src[off..][0..4], .little))))) / 65536.0,
        GL_BYTE => {
            const v: i8 = @bitCast(src[off]);
            return if (normalized) @max(@as(f32, @floatFromInt(v)) / 127.0, -1.0) else @floatFromInt(v);
        },
        GL_UNSIGNED_BYTE => {
            const v = src[off];
            return if (normalized) @as(f32, @floatFromInt(v)) / 255.0 else @floatFromInt(v);
        },
        GL_SHORT => {
            const v: i16 = @bitCast(std.mem.readInt(u16, src[off..][0..2], .little));
            return if (normalized) @max(@as(f32, @floatFromInt(v)) / 32767.0, -1.0) else @floatFromInt(v);
        },
        GL_UNSIGNED_SHORT => {
            const v = std.mem.readInt(u16, src[off..][0..2], .little);
            return if (normalized) @as(f32, @floatFromInt(v)) / 65535.0 else @floatFromInt(v);
        },
        GL_INT => {
            const v: i32 = @bitCast(std.mem.readInt(u32, src[off..][0..4], .little));
            return if (normalized) @max(@as(f32, @floatFromInt(v)) / 2147483647.0, -1.0) else @floatFromInt(v);
        },
        GL_UNSIGNED_INT => {
            const v = std.mem.readInt(u32, src[off..][0..4], .little);
            return if (normalized) @as(f32, @floatFromInt(v)) / 4294967295.0 else @floatFromInt(v);
        },
        else => return 0,
    }
}

/// One enabled vertex attribute's SOURCE in the draw: which buffer it reads from, the byte
/// stride between vertices in that buffer, the byte offset of the first component, and its
/// component count. The attribute may come from its own VBO (glmark2 stores position/normal/
/// texcoord in SEPARATE tightly-packed VBOs, not one interleaved buffer).
const DrawSource = struct {
    location: u32,
    src: []const u8, // the source bytes: a VBO's CPU mirror, or a client-array CPU range
    src_stride: u32, // bytes between consecutive vertices in `src`
    src_offset: u32, // byte offset of the first component within `src`
    comp_bytes: u32, // DEST bytes per vertex (size * 4 - the interleaved buffer is always f32)
    format: prism.hal.Format,
    divisor: u32, // 0 = per-vertex; D>0 = per-instance (element = floor(instance/D))
    gl_type: GLenum = GL_FLOAT, // SOURCE component type; non-float is converted to f32 at build
    normalized: bool = false,
    size: u8 = 4, // component count (1..4)
    src_comp_bytes: u32 = 16, // SOURCE bytes per vertex (size * sizeof(gl_type))
};

/// The resolved per-attribute source descriptors for a draw. Each attribute is copied into a
/// single INTERLEAVED expanded vertex buffer the HAL pipeline reads. The destination layout
/// (the HAL VertexAttribute offsets + the interleaved stride) is computed in drawTriangleList.
const DrawLayout = struct {
    sources: [MAX_ATTRIBS]DrawSource,
    n_attrs: usize,
};

/// Fetch one attribute's SOURCE (at byte `so`) into the interleaved DEST at `dst_off` as f32(s).
/// GL_FLOAT is a straight memcpy (byte-identical fast path). A packed type (byte/short/half/fixed/
/// int) converts per component (normalizing if requested). Out-of-range source reads leave the
/// dest as its zeroed default.
fn fetchAttrib(emap: []u8, dst_off: usize, s: DrawSource, so: usize) void {
    if (s.gl_type == GL_FLOAT) {
        if (so + s.comp_bytes <= s.src.len) @memcpy(emap[dst_off .. dst_off + s.comp_bytes], s.src[so .. so + s.comp_bytes]);
        return;
    }
    const tsz = attribTypeSize(s.gl_type);
    if (so + @as(usize, s.size) * tsz > s.src.len) return;
    var comp: usize = 0;
    while (comp < s.size) : (comp += 1) {
        const f = readAttribF32(s.src, so + comp * tsz, s.gl_type, s.normalized);
        @memcpy(emap[dst_off + comp * 4 ..][0..4], std.mem.asBytes(&f));
    }
}

/// Collect the enabled vertex-attribute arrays into per-attribute source descriptors. Each
/// attribute may read from its OWN bound array buffer (separate VBOs), the common
/// non-interleaved layout glmark2's Mesh uses. The interleaved single-VBO layout es2gears
/// uses is the n=1-buffer special case. Caller holds obj_lock. Returns null on error.
fn resolveDrawLayout(max_vi: u32) ?DrawLayout {
    var sources: [MAX_ATTRIBS]DrawSource = undefined;
    var n_attrs: usize = 0;
    for (attribs, 0..) |a, loc| {
        if (!a.enabled) {
            // A DISABLED array with a glVertexAttrib*f generic value: feed that constant to EVERY
            // vertex (a zero-stride source pointing at the stored 4 floats). A disabled attribute
            // with no generic set is left out (the VS reads it as the pipeline's default).
            if (a.has_generic) {
                sources[n_attrs] = .{
                    .location = @intCast(loc),
                    .src = std.mem.asBytes(&attribs[loc].generic),
                    .src_stride = 0, // constant across vertices
                    .src_offset = 0,
                    .comp_bytes = 16, // vec4 of f32
                    .format = .r32g32b32a32_float,
                    .divisor = 0,
                };
                n_attrs += 1;
            }
            continue;
        }
        const fmt = attribFormat(a.size) orelse {
            setError(GL_INVALID_OPERATION);
            return null;
        };
        const comp_bytes: u32 = @as(u32, @intCast(a.size)) * 4; // DEST (interleaved f32) bytes
        const src_comp_bytes: u32 = @as(u32, @intCast(a.size)) * attribTypeSize(a.gl_type); // SOURCE bytes
        // A zero stride means tightly packed (this attribute's SOURCE component bytes); otherwise
        // the app's interleaving stride.
        const src_stride: u32 = if (a.stride > 0) @intCast(a.stride) else src_comp_bytes;
        var src: []const u8 = undefined;
        var src_offset: u32 = 0;
        if (a.buffer != 0) {
            // VBO-backed: read from the bound buffer's CPU mirror at the byte offset.
            const b = findBuffer(a.buffer) orelse {
                setError(GL_INVALID_OPERATION);
                return null;
            };
            if (a.offset > std.math.maxInt(u32)) {
                setError(GL_INVALID_OPERATION);
                return null;
            }
            src = b.bytes.items;
            src_offset = @intCast(a.offset);
        } else {
            // CLIENT-SIDE array (no VBO bound): a.offset holds the app's raw CPU pointer
            // (glmark2's desktop/effect2d scenes draw quads this way). Wrap the range the
            // draw touches - [0 .. max_vi*stride + src_comp_bytes) from that pointer - as a
            // slice. the pointer's vertex 0 is the first component, so src_offset is 0.
            if (a.offset == 0) {
                setError(GL_INVALID_OPERATION); // enabled client array with a null pointer
                return null;
            }
            const need: usize = @as(usize, max_vi) * src_stride + src_comp_bytes;
            src = @as([*]const u8, @ptrFromInt(a.offset))[0..need];
            src_offset = 0;
        }
        sources[n_attrs] = .{
            .location = @intCast(loc),
            .src = src,
            .src_stride = src_stride,
            .src_offset = src_offset,
            .comp_bytes = comp_bytes,
            .format = fmt,
            .divisor = a.divisor,
            .gl_type = a.gl_type,
            .normalized = a.normalized,
            .size = @intCast(a.size),
            .src_comp_bytes = src_comp_bytes,
        };
        n_attrs += 1;
    }
    // n_attrs == 0 is VALID: a vertex-buffer-less draw whose VS derives everything from
    // gl_VertexID / gl_InstanceID (the procedural full-screen-triangle idiom). The draw builds
    // an empty vertex layout + a dummy vertex buffer. the VS reads only the DA-delivered index.
    return .{ .sources = sources, .n_attrs = n_attrs };
}

/// The HAL depth state for the live GLES fixed-function depth settings, including the
/// glPolygonOffset depth bias (GL factor -> bias_slope, units -> bias_constant) when
/// GL_POLYGON_OFFSET_FILL is enabled.
fn wantDepthState() prism.hal.DepthState {
    if (!fixed.depth_test) return .{};
    return .{
        .test_enable = true,
        .write_enable = fixed.depth_write,
        .compare_op = fixed.depth_func,
        .bias_enable = fixed.polygon_offset,
        .bias_slope = if (fixed.polygon_offset) fixed.polygon_offset_factor else 0,
        .bias_constant = if (fixed.polygon_offset) fixed.polygon_offset_units else 0,
    };
}

/// The HAL stencil state for the live GLES fixed-function stencil settings. When
/// GL_STENCIL_TEST is disabled this is the passthrough default. When enabled it carries the
/// tracked glStencilFunc/glStencilOp/glStencilMask values.
fn wantStencilState() prism.hal.StencilState {
    if (!fixed.stencil_test) return .{};
    return .{
        .test_enable = true,
        .compare_op = fixed.stencil_func,
        .fail_op = fixed.stencil_sfail,
        .depth_fail_op = fixed.stencil_dpfail,
        .pass_op = fixed.stencil_dppass,
        .compare_mask = fixed.stencil_value_mask,
        .write_mask = fixed.stencil_write_mask,
        .reference = fixed.stencil_ref,
    };
}

/// The back-face HAL stencil state for two-sided stencil (glStencilFuncSeparate/glStencilOpSeparate
/// with GL_BACK). Returns null when stencil is disabled or the back state is byte-identical to the
/// front state. A program that never diverged the faces takes the single-face path (stencil_back null).
/// Only a genuinely two-sided program builds a two-sided pipeline. The driver's front/back facing is
/// picked by wantCullState's flip-aware front_face, so the GL front state maps to the driver front
/// (p.stencil) and GL back to p.stencil_back with no
/// swap needed here.
fn wantStencilBack() ?prism.hal.StencilState {
    if (!fixed.stencil_test) return null;
    const back: prism.hal.StencilState = .{
        .test_enable = true,
        .compare_op = fixed.stencil_back_func,
        .fail_op = fixed.stencil_back_sfail,
        .depth_fail_op = fixed.stencil_back_dpfail,
        .pass_op = fixed.stencil_back_dppass,
        .compare_mask = fixed.stencil_back_value_mask,
        .write_mask = fixed.stencil_back_write_mask,
        .reference = fixed.stencil_back_ref,
    };
    if (std.meta.eql(back, wantStencilState())) return null;
    return back;
}

/// HAL cull state for the live GLES cull settings. `flip` is whether the VS applies the
/// GL-origin y-flip (true for the default framebuffer). The y-flip negates window-space
/// winding, so the front face is inverted to keep GL front faces front. An FBO render
/// target uses the unflipped VS (flip=false), so the GL winding is used verbatim.
/// Inverting it there wrongly culled the glmark2 desktop windows.
fn wantCullState(flip: bool) prism.hal.CullState {
    // GL's window space is y-UP. The HAL rasterizer is y-DOWN (Vulkan). For the default
    // framebuffer the VS bakes in the GL-origin y-flip, so the presented image lands pixel-for-
    // pixel where GL would put it. Triangle winding in final window space matches GL exactly,
    // so the GL front face is used verbatim. An FBO render target uses the unflipped VS, so its
    // stored image is y-flipped relative to GL and the winding is negated. The GL front face is
    // inverted there. (Inverting the default-fb path wrongly culled the glmark2 desktop final
    // blit; not inverting the FBO path wrongly culled the desktop windows.)
    // The facing is computed even when culling is OFF: two-sided stencil (wantStencilBack) picks
    // the front/back state by this same front_face, so it must reflect glFrontFace + flip whether
    // or not GL_CULL_FACE is enabled. mode=.none then draws both faces but keeps the facing.
    const front: prism.hal.FrontFace = if (flip) fixed.front_face else switch (fixed.front_face) {
        .counter_clockwise => .clockwise,
        .clockwise => .counter_clockwise,
    };
    return .{ .mode = if (fixed.cull_face) fixed.cull_mode else .none, .front_face = front };
}

/// The HAL blend state for the live GLES fixed-function blend settings. When GL_BLEND is
/// disabled this is the passthrough default (enable=false). When enabled it carries the
/// tracked glBlendFunc*/glBlendEquation*/glBlendColor factors, ops, and constant.
fn wantBlendState() prism.hal.BlendState {
    // The color write mask (glColorMask) is independent of GL_BLEND, so it rides the BlendState
    // in BOTH the blend-off and blend-on cases.
    if (!fixed.blend) return .{ .write_mask = fixed.color_mask };
    return .{
        .enable = true,
        .src_color = fixed.blend_src_rgb,
        .dst_color = fixed.blend_dst_rgb,
        .src_alpha = fixed.blend_src_alpha,
        .dst_alpha = fixed.blend_dst_alpha,
        .color_op = fixed.blend_op_rgb,
        .alpha_op = fixed.blend_op_alpha,
        .constant = fixed.blend_color,
        .write_mask = fixed.color_mask,
    };
}

/// Whether the program's pipeline must be rebuilt because the live depth/cull/blend state
/// changed since it was baked (the fixed-function state lives outside the program).
fn pipelineStateMatches(p: *Program, flip: bool) bool {
    const want_depth = wantDepthState();
    const want_cull = wantCullState(flip);
    const want_blend = wantBlendState();
    const want_stencil = wantStencilState();
    const want_stencil_back = wantStencilBack();
    if (flip) return std.meta.eql(p.pipeline_depth, want_depth) and std.meta.eql(p.pipeline_cull, want_cull) and std.meta.eql(p.pipeline_blend, want_blend) and std.meta.eql(p.pipeline_stencil, want_stencil) and std.meta.eql(p.pipeline_stencil_back, want_stencil_back) and p.pipeline_topology == draw_topology and p.pipeline_line_width == line_width and p.pipeline_a2c == fixed.sample_alpha_to_coverage and p.pipeline_scov == fixed.sample_coverage and p.pipeline_scov_value == fixed.sample_coverage_value and p.pipeline_scov_invert == fixed.sample_coverage_invert;
    return std.meta.eql(p.pipeline_noflip_depth, want_depth) and std.meta.eql(p.pipeline_noflip_cull, want_cull) and std.meta.eql(p.pipeline_noflip_blend, want_blend) and std.meta.eql(p.pipeline_noflip_stencil, want_stencil) and std.meta.eql(p.pipeline_noflip_stencil_back, want_stencil_back) and p.pipeline_noflip_topology == draw_topology and p.pipeline_noflip_line_width == line_width and p.pipeline_noflip_a2c == fixed.sample_alpha_to_coverage and p.pipeline_noflip_scov == fixed.sample_coverage and p.pipeline_noflip_scov_value == fixed.sample_coverage_value and p.pipeline_noflip_scov_invert == fixed.sample_coverage_invert;
}

// --- Bound render target (default framebuffer OR a bound FBO) ---------------

/// The render-target resources a clear/draw writes: a color image, an optional depth image,
/// the dimensions, and (for an FBO depth-texture attachment) the Texture whose `bytes` must
/// be rebuilt from the rendered f32 depths before it is next sampled.
const RenderTargets = struct {
    color: *prism.hal.Resource,
    color_w: u32,
    color_h: u32,
    depth: ?*prism.hal.Resource = null,
    depth_tex: ?*Texture = null, // a GL_OES_depth_texture attachment to finalize after draw
    stencil: ?*prism.hal.Resource = null, // u8/pixel stencil BUFFER (a stencil renderbuffer)
    stencil_w: u32 = 0,
    stencil_h: u32 = 0,
    // MRT: HAL images for GL_COLOR_ATTACHMENT1..N-1 (index 0 = attachment 1), contiguous from
    // the first bound extra attachment up to extra_color_count. The draw binds these via
    // cb.setColorTarget(1+i, ...). A fragment shader's located `out`s land in each.
    extra_colors: [prism.hal.MAX_COLOR_TARGETS - 1]?*prism.hal.Resource = .{null} ** (prism.hal.MAX_COLOR_TARGETS - 1),
    extra_color_count: u32 = 0,
};

/// Get-or-create the HAL depth32_float image backing a depth texture attachment. Caller
/// holds obj_lock.
fn ensureDepthTextureHal(dev: prism.hal.Device, t: *Texture, w: u32, h: u32) prism.Error!*prism.hal.Resource {
    if (t.depth_hal) |d| return d;
    const d = try dev.createResource(.{ .image = .{
        .width = w,
        .height = h,
        .format = .depth32_float,
        .usage = .{ .render_target = true },
    } });
    t.depth_hal = d;
    t.hal_dev = dev;
    return d;
}

/// Get-or-create the HAL image backing a renderbuffer (depth32_float for depth, rgba8_unorm
/// for color). Caller holds obj_lock.
fn ensureRenderbufferHal(dev: prism.hal.Device, r: *Renderbuffer) prism.Error!*prism.hal.Resource {
    if (r.hal) |h| return h;
    const h = try dev.createResource(.{
        .image = .{
            .width = r.width,
            .height = r.height,
            .format = if (r.is_depth) .depth32_float else .rgba8_unorm,
            // A multisampled color attachment renders N samples (the software raster keys off the
            // target's sample count). It is not directly sampled - glBlitFramebuffer resolves it.
            .samples = if (r.is_depth) 1 else r.samples,
            .usage = .{ .render_target = true, .sampled = if (r.is_depth or r.samples > 1) false else true },
        },
    });
    r.hal = h;
    r.hal_dev = dev;
    return h;
}

/// The single-sample resolve target for a multisampled color renderbuffer (the glBlitFramebuffer
/// source when reading an MSAA FBO). Box-downsamples the multisampled image into it via the HAL
/// resolve command. Caller holds obj_lock. Returns the renderbuffer's HAL image directly if it is
/// single-sample.
fn resolveRenderbufferColor(ctx: *state.Context, r: *Renderbuffer) ?*prism.hal.Resource {
    const dev = ctx.device();
    const ms = ensureRenderbufferHal(dev, r) catch return null;
    if (r.samples <= 1) return ms;
    if (r.resolved_hal == null) {
        r.resolved_hal = dev.createResource(.{ .image = .{
            .width = r.width,
            .height = r.height,
            .format = .rgba8_unorm,
            .usage = .{ .render_target = true, .copy_src = true },
        } }) catch return null;
    }
    ctx.resolveMsaa(ms, r.resolved_hal.?, r.width, r.height, .rgba8_unorm, r.samples) catch return null;
    return r.resolved_hal.?;
}

/// Get-or-create the u8/pixel stencil buffer backing a stencil (or packed depth+stencil)
/// renderbuffer. A stencil surface is a plain buffer (one byte per pixel), matching the HAL
/// stencil-target contract and the surface stencil attachment. Caller holds obj_lock.
fn ensureRenderbufferStencilHal(dev: prism.hal.Device, r: *Renderbuffer) prism.Error!*prism.hal.Resource {
    if (r.stencil_hal) |h| return h;
    const h = try dev.createResource(.{ .buffer = .{ .size = @as(usize, r.width) * r.height } });
    r.stencil_hal = h;
    r.hal_dev = dev;
    return h;
}

/// A scratch color image for an FBO that has only a depth attachment (a depth-only shadow
/// pass still needs a color render target for the software draw to run + write depth). Keyed
/// by dimensions. Rebuilt when the size changes.
var scratch_color: ?*prism.hal.Resource = null;
var scratch_color_dev: ?prism.hal.Device = null;
var scratch_color_w: u32 = 0;
var scratch_color_h: u32 = 0;

fn ensureScratchColor(dev: prism.hal.Device, w: u32, h: u32) prism.Error!*prism.hal.Resource {
    if (scratch_color != null and scratch_color_w == w and scratch_color_h == h) return scratch_color.?;
    if (scratch_color) |s| if (scratch_color_dev) |d| d.destroyResource(s);
    const s = try dev.createResource(.{ .image = .{
        .width = w,
        .height = h,
        .format = .rgba8_unorm,
        .usage = .{ .render_target = true, .copy_src = true },
    } });
    scratch_color = s;
    scratch_color_dev = dev;
    scratch_color_w = w;
    scratch_color_h = h;
    return s;
}

/// Resolve the bound render targets: the default framebuffer (the surface backbuffer +
/// lazily-created surface depth) when no FBO is bound, else the bound FBO's color/depth
/// attachment HAL resources (render-to-texture). Caller holds obj_lock. `want_depth` is
/// whether the draw needs a depth attachment (depth test on) so the default-FB surface
/// depth is created on demand only then.
fn resolveRenderTargets(dev: prism.hal.Device, surf: *state.Surface, want_depth: bool) ?RenderTargets {
    if (bound_framebuffer == 0) {
        // Draw into the RENDER target: the multisampled backbuffer for an MSAA surface (the
        // driver anti-aliases against its sample count), else the plain backbuffer. color_w/h
        // stay the LOGICAL size. swapBuffers resolves the MS target into the presented image.
        var rt: RenderTargets = .{ .color = surf.renderTarget(), .color_w = surf.width, .color_h = surf.height };
        if (want_depth) rt.depth = surf.depthAttachment() catch null;
        return rt;
    }
    const f = findFramebuffer(bound_framebuffer) orelse {
        setError(GL_INVALID_OPERATION);
        return null;
    };
    // Color attachment: a texture image, a renderbuffer image, or (depth-only FBO) a
    // scratch color image sized to the depth attachment.
    var color: ?*prism.hal.Resource = null;
    var cw: u32 = 0;
    var ch: u32 = 0;
    if (f.color_tex != 0) {
        const ct = findTexture(f.color_tex) orelse {
            setError(GL_INVALID_OPERATION);
            return null;
        };
        if (ct.width == 0 or ct.height == 0) {
            setError(GL_INVALID_OPERATION);
            return null;
        }
        // The color texture is sampled later, so it needs its rgba8 HAL image as the RT.
        const img = ensureTextureHal(dev, ct) catch {
            setError(GL_INVALID_OPERATION);
            return null;
        };
        ct.hal_dirty = false; // the GPU now owns it. do not clobber with stale CPU bytes
        color = img;
        cw = ct.width;
        ch = ct.height;
    } else if (f.color_rb != 0) {
        const cr = findRenderbuffer(f.color_rb) orelse {
            setError(GL_INVALID_OPERATION);
            return null;
        };
        const img = ensureRenderbufferHal(dev, cr) catch {
            setError(GL_INVALID_OPERATION);
            return null;
        };
        color = img;
        cw = cr.width;
        ch = cr.height;
    }

    // Depth attachment: a GL_OES_depth_texture (finalized to RGBA8 after the pass) or a
    // depth renderbuffer.
    var depth: ?*prism.hal.Resource = null;
    var depth_tex: ?*Texture = null;
    var dw: u32 = 0;
    var dh: u32 = 0;
    if (f.depth_tex != 0) {
        const dt = findTexture(f.depth_tex) orelse {
            setError(GL_INVALID_OPERATION);
            return null;
        };
        const dim = ensureDepthTextureHal(dev, dt, dt.width, dt.height) catch {
            setError(GL_INVALID_OPERATION);
            return null;
        };
        depth = dim;
        depth_tex = dt;
        dw = dt.width;
        dh = dt.height;
    } else if (f.depth_rb != 0) {
        const dr = findRenderbuffer(f.depth_rb) orelse {
            setError(GL_INVALID_OPERATION);
            return null;
        };
        const dim = ensureRenderbufferHal(dev, dr) catch {
            setError(GL_INVALID_OPERATION);
            return null;
        };
        depth = dim;
        dw = dr.width;
        dh = dr.height;
    }

    // Stencil attachment: a stencil (or packed depth+stencil) renderbuffer -> a u8/pixel buffer.
    var stencil: ?*prism.hal.Resource = null;
    var sw: u32 = 0;
    var sh: u32 = 0;
    if (f.stencil_rb != 0) {
        const sr = findRenderbuffer(f.stencil_rb) orelse {
            setError(GL_INVALID_OPERATION);
            return null;
        };
        const sim = ensureRenderbufferStencilHal(dev, sr) catch {
            setError(GL_INVALID_OPERATION);
            return null;
        };
        stencil = sim;
        sw = sr.width;
        sh = sr.height;
    }

    if (color == null) {
        // Depth-only / stencil-only FBO: synthesize a color target sized to the attachment.
        const aw = if (dw != 0) dw else sw;
        const ah = if (dh != 0) dh else sh;
        if (aw == 0 or ah == 0) {
            setError(GL_INVALID_OPERATION);
            return null;
        }
        const s = ensureScratchColor(dev, aw, ah) catch {
            setError(GL_INVALID_OPERATION);
            return null;
        };
        color = s;
        cw = aw;
        ch = ah;
    }
    // MRT: resolve GL_COLOR_ATTACHMENT1..N-1 into HAL images. Only a contiguous run from
    // attachment 1 is bound (a gap would leave the ROP without a target for that slot). The
    // count stops at the first empty slot. Each becomes a HAL extra color target at the draw.
    var extra_colors: [prism.hal.MAX_COLOR_TARGETS - 1]?*prism.hal.Resource = .{null} ** (prism.hal.MAX_COLOR_TARGETS - 1);
    var extra_count: u32 = 0;
    for (f.extra_color_tex, f.extra_color_rb, 0..) |etex, erb, i| {
        const img: ?*prism.hal.Resource = if (etex != 0) blk: {
            const ct = findTexture(etex) orelse {
                setError(GL_INVALID_OPERATION);
                return null;
            };
            const im = ensureTextureHal(dev, ct) catch {
                setError(GL_INVALID_OPERATION);
                return null;
            };
            ct.hal_dirty = false;
            break :blk im;
        } else if (erb != 0) blk: {
            const cr = findRenderbuffer(erb) orelse {
                setError(GL_INVALID_OPERATION);
                return null;
            };
            break :blk ensureRenderbufferHal(dev, cr) catch {
                setError(GL_INVALID_OPERATION);
                return null;
            };
        } else null;
        if (img == null) break; // stop at the first unbound attachment (contiguous run only)
        extra_colors[i] = img;
        extra_count = @intCast(i + 1);
    }
    return .{ .color = color.?, .color_w = cw, .color_h = ch, .depth = depth, .depth_tex = depth_tex, .stencil = stencil, .stencil_w = sw, .stencil_h = sh, .extra_colors = extra_colors, .extra_color_count = extra_count };
}

/// After a depth-texture render pass, rebuild the texture's RGBA8 `bytes` from the rendered
/// f32 depths so a later texture2D(ShadowMap, ...) reads the depth in `.x` (R=G=B=depth,
/// A=1). The host RGBA8 sampler then samples it like any color texture. Caller holds obj_lock.
fn finalizeDepthTexture(dev: prism.hal.Device, t: *Texture) void {
    const dh = t.depth_hal orelse return;
    const w = t.width;
    const h = t.height;
    if (w == 0 or h == 0) return;
    // Hardware shadow path (nvidia): the rendered depth lives in a tiled/non-sampleable ZETA. Build
    // a real ZF32 sampled texture from it via the driver seam (CE-detile -> sampled ZF32) so a
    // sampler2DShadow does a HW DEPTH_COMPARE against the rendered depth. The rgba8 conversion below
    // (a HW compare no-op) is skipped. Software leaves finalizeDepthTexture null and uses rgba8.
    if (dev.vtable.finalizeDepthTexture != null) {
        if (t.depth_sampled_hal) |old| dev.destroyResource(old);
        t.depth_sampled_hal = dev.finalizeDepthTexture(dh, w, h) catch null;
        t.depth_dirty = false;
        return;
    }
    const dmap = dev.mapResource(dh) catch return;
    const aligned: []align(@alignOf(f32)) const u8 = @alignCast(dmap[0 .. @as(usize, w) * h * 4]);
    const depths = std.mem.bytesAsSlice(f32, aligned);
    t.bytes.resize(gpa, @as(usize, w) * h * 4) catch return;
    // Build the rgba8 "depth-as-color" image: replicate the [0,1] depth into R,G,B; A=255.
    for (depths, 0..) |z, i| {
        const v: u8 = @intFromFloat(std.math.clamp(z, 0.0, 1.0) * 255.0 + 0.5);
        t.bytes.items[i * 4 + 0] = v;
        t.bytes.items[i * 4 + 1] = v;
        t.bytes.items[i * 4 + 2] = v;
        t.bytes.items[i * 4 + 3] = 255;
    }
    // Force the sampled rgba8 image (t.hal) to be rebuilt from these depth bytes.
    if (t.hal) |old| dev.destroyResource(old);
    t.hal = null;
    t.hal_dirty = true;
    t.depth_dirty = false;
}

/// Core draw: the `indices` sequence selects vertices from the bound array buffer, already
/// resolved as a triangle list (strips/indexed draws are expanded by the caller). Builds a
/// fresh triangle-list vertex Resource by gathering each indexed vertex's bytes so the software
/// driver's sequential first_vertex+tri*3 read is correct for any topology. Rebuilds the
/// pipeline with the live depth+cull state, binds the program's uniform UBO and the surface
/// depth attachment, and draws via the proven HAL flow.
/// Run the software capture path for `specs` (tightly interleaved) into the GL buffer bound at
/// transform-feedback binding point `binding`, appending at that binding's running cursor
/// (tf_write_offsets[binding]). `total_floats` is the sum of the specs' components (bytes/vertex/4).
/// A driver without capture support (nvidia/apple) or an unbound buffer is a no-op. Holds obj_lock.
fn captureVaryingsIntoBinding(dev: prism.hal.Device, pipeline: *prism.hal.Pipeline, expanded: *prism.hal.Resource, vcount: u32, vs_ubo: ?*prism.hal.Resource, fs_ubo: ?*prism.hal.Resource, binding: usize, specs: []const prism.hal.TfCaptureSpec, total_floats: usize) void {
    if (specs.len == 0 or total_floats == 0) return;
    const tf_id = transform_feedback_bindings[binding];
    if (tf_id == 0) return;
    const tf_buf = findBuffer(tf_id) orelse return;

    const bytes_needed = @as(usize, vcount) * total_floats * 4;
    if (bytes_needed == 0) return;

    // Capture into a scratch HAL resource, then copy the result into the GL buffer's CPU storage
    // (which glGetBufferSubData / mapping reads) at this binding's running write offset.
    const scratch = dev.createResource(.{ .buffer = .{ .size = bytes_needed } }) catch return;
    defer dev.destroyResource(scratch);
    var ubos = [_]?*prism.hal.Resource{ vs_ubo, fs_ubo };
    const written = dev.captureTransformFeedback(.{
        .pipeline = pipeline,
        .vertex_buffer = expanded,
        .ubos = &ubos,
        .output = scratch,
        .output_offset = 0,
        .specs = specs,
        .vertex_count = vcount,
        .first_vertex = 0,
        .instance = 0,
    }) catch return;
    const nbytes = written orelse return; // driver does not support capture (non-software)
    if (nbytes == 0) return;
    const src = dev.mapResource(scratch) catch return;
    // Copy into the GL buffer's CPU mirror at this binding's running cursor, clamped to its size.
    const cursor = tf_write_offsets[binding];
    const dst_end = @min(cursor + nbytes, tf_buf.bytes.items.len);
    if (dst_end > cursor) {
        const ncopy = dst_end - cursor;
        @memcpy(tf_buf.bytes.items[cursor..dst_end], src[0..ncopy]);
    }
    tf_write_offsets[binding] += nbytes;
    // The GL buffer's CPU mirror changed. Drop any stale HAL mirror so a later use re-uploads.
    if (tf_buf.hal) |h| {
        if (tf_buf.hal_dev) |d| d.destroyResource(h);
        tf_buf.hal = null;
    }
}

/// Capture this draw's transform-feedback varyings. Resolves each of the program's
/// glTransformFeedbackVaryings names to a VS output location (via p.vs_outputs), runs the VS over
/// `vcount` vertices through the software driver's capture path, and appends the captured floats
/// to the bound GL_TRANSFORM_FEEDBACK_BUFFER(s) at the running write cursor(s). GL_INTERLEAVED_ATTRIBS
/// tightly packs every varying into the buffer at binding 0. GL_SEPARATE_ATTRIBS captures varying i
/// into its own buffer at binding point i. A driver without capture support (nvidia/apple) is a
/// no-op. Caller holds obj_lock.
fn captureTransformFeedbackForDraw(dev: prism.hal.Device, p: *Program, pipeline: *prism.hal.Pipeline, expanded: *prism.hal.Resource, vcount: u32, vs_ubo: ?*prism.hal.Resource, fs_ubo: ?*prism.hal.Resource) void {
    // Resolve each recorded capture name to a VS output (location + components). Skip any name the
    // linked VS does not export (a robust minimal path; a full impl would fail the link).
    var specs: [16]prism.hal.TfCaptureSpec = undefined;
    var n_specs: usize = 0;
    var total_floats: usize = 0;
    for (p.tf_varyings.items) |cap_name| {
        if (n_specs >= specs.len) break;
        for (p.vs_outputs.items) |o| {
            if (std.mem.eql(u8, o.name, cap_name)) {
                specs[n_specs] = .{ .location = o.location, .first_component = 0, .components = o.components };
                n_specs += 1;
                total_floats += o.components;
                break;
            }
        }
    }
    if (n_specs == 0 or total_floats == 0) return;

    if (p.tf_buffer_mode == GL_SEPARATE_ATTRIBS) {
        // Each varying -> its own buffer at binding point i, with its own running cursor.
        for (specs[0..n_specs], 0..) |sp, i| {
            if (i >= MAX_TRANSFORM_FEEDBACK_BUFFERS) break;
            const one = [_]prism.hal.TfCaptureSpec{sp};
            captureVaryingsIntoBinding(dev, pipeline, expanded, vcount, vs_ubo, fs_ubo, i, &one, sp.components);
        }
        return;
    }

    // GL_INTERLEAVED_ATTRIBS: all varyings tightly packed into the single buffer at binding 0.
    captureVaryingsIntoBinding(dev, pipeline, expanded, vcount, vs_ubo, fs_ubo, 0, specs[0..n_specs], total_floats);
}

/// Submit any pending batched draws so a subsequent GPU-result read / present / render-target
/// change / resource free observes their output. No-op when not batching or the batch is empty.
/// Call at the top of every entry point that reads GPU-rendered data, retargets, syncs, or frees a
/// resource a pending draw could reference. See state.Context.flushDraws.
fn flushBatch() void {
    const ctx = state.currentContext() orelse return;
    ctx.flushDraws() catch {};
}

fn drawTriangleList(indices: []const u32) void {
    drawTriangleListInstanced(indices, 1);
}

fn drawTriangleListInstanced(indices: []const u32, instance_count: u32) void {
    const ctx = state.currentContext() orelse {
        setError(GL_INVALID_OPERATION);
        return;
    };
    const surf = state.currentDrawSurface() orelse {
        setError(GL_INVALID_OPERATION);
        return;
    };
    if (current_program == 0) {
        setError(GL_INVALID_OPERATION);
        return;
    }
    if (indices.len < draw_topology.vertsPerPrimitive()) return; // 3 tris / 2 lines / 1 point

    obj_lock.lock();
    const p = findProgram(current_program) orelse {
        obj_lock.unlock();
        setError(GL_INVALID_OPERATION);
        return;
    };
    if (!p.linked or p.hal_vs == null or p.hal_fs == null) {
        obj_lock.unlock();
        setError(GL_INVALID_OPERATION);
        return;
    }
    // The highest vertex index this draw gathers - bounds the client-array CPU range.
    var max_vi: u32 = 0;
    for (indices) |vi| {
        if (vi > max_vi) max_vi = vi;
    }
    const layout = resolveDrawLayout(max_vi) orelse {
        obj_lock.unlock();
        return;
    };
    const dev = ctx.device();

    // Compute the INTERLEAVED destination layout: pack each attribute's component bytes back
    // to back into one vertex stride, in the order resolveDrawLayout collected them. The HAL
    // pipeline reads this tightly-interleaved buffer regardless of whether the app supplied
    // one interleaved VBO (es2gears) or separate per-attribute VBOs (glmark2's Mesh).
    var dst_attrs: [MAX_ATTRIBS]prism.hal.VertexAttribute = undefined;
    var stride: u32 = 0;
    for (layout.sources[0..layout.n_attrs], 0..) |s, i| {
        dst_attrs[i] = .{ .location = s.location, .format = s.format, .offset = stride };
        stride += s.comp_bytes;
    }

    // Build an expanded triangle-list vertex buffer: one interleaved vertex (stride bytes) per
    // index, gathering each attribute from its OWN source buffer/stride/offset.
    const vcount = indices.len;
    // A gl_VertexID-only draw has stride 0 (no attributes). Still allocate a 1-byte dummy
    // vertex buffer so the HAL has a valid (unused) binding - the VS reads only gl_VertexID.
    const exp_size = @max(vcount * stride, 1);
    // Check out a pooled vertex buffer instead of allocating + mapping + freeing a fresh GPU
    // Resource every draw (that alloc/mmap/free churn dominated per-draw CPU cost). The buffer is
    // fully rewritten below. In batched mode each draw gets its own pool slot (they coexist in one
    // deferred submit). Non-batched reuses one buffer (the prior draw fenced before reuse).
    const pooled_vb = ctx.checkoutVertexBuffer(exp_size) catch {
        obj_lock.unlock();
        setError(GL_INVALID_OPERATION);
        return;
    };
    const emap = pooled_vb.map;
    const expanded = pooled_vb.res;
    @memset(emap[0..exp_size], 0);
    for (indices, 0..) |vi, out_i| {
        const doff = out_i * stride;
        for (layout.sources[0..layout.n_attrs], 0..) |s, i| {
            fetchAttrib(emap, doff + dst_attrs[i].offset, s, @as(usize, vi) * s.src_stride + s.src_offset);
        }
    }

    // The GL-origin y-flip is correct only for the default framebuffer (presented top-down).
    // An FBO render target uses the unflipped VS (and un-inverted cull) so its sampled texture
    // is upright and not wrongly culled. Pick the matching VS + pipeline slot.
    const flip = bound_framebuffer == 0;
    const vs_mod = if (flip) p.hal_vs.? else p.hal_vs_noflip.?;
    const cur_pipe = if (flip) p.hal_pipeline else p.hal_pipeline_noflip;
    if (cur_pipe == null or !pipelineStateMatches(p, flip)) {
        if (cur_pipe) |old| {
            // A batched draw may still reference this cached pipeline. Submit the batch before the
            // state change frees + rebuilds it (else the deferred draw dereferences a freed pipeline).
            ctx.flushDraws() catch {};
            // Null the slot BEFORE freeing: if the createPipeline below FAILS (e.g. a shader that
            // overflows the register budget) it returns early, so the slot must not be left pointing
            // at the freed pipeline - the next draw would re-enter here and double-free it (crash).
            if (flip) p.hal_pipeline = null else p.hal_pipeline_noflip = null;
            dev.destroyPipeline(old);
        }
        const want_depth = wantDepthState();
        const want_cull = wantCullState(flip);
        const want_blend = wantBlendState();
        const want_stencil = wantStencilState();
        const want_stencil_back = wantStencilBack();
        const pipe = dev.createPipeline(.{
            .vertex = vs_mod,
            .fragment = p.hal_fs.?,
            .vertex_layout = .{ .stride = stride, .attributes = dst_attrs[0..layout.n_attrs] },
            .color_format = currentColorFormat(),
            .depth = want_depth,
            .cull = want_cull,
            .blend = want_blend,
            .stencil = want_stencil,
            .stencil_back = want_stencil_back,
            .topology = draw_topology,
            .line_width = line_width,
            .alpha_to_coverage = fixed.sample_alpha_to_coverage,
            .sample_coverage = fixed.sample_coverage,
            .sample_coverage_value = fixed.sample_coverage_value,
            .sample_coverage_invert = fixed.sample_coverage_invert,
        }) catch {
            obj_lock.unlock();
            setError(GL_INVALID_OPERATION);
            return;
        };
        if (flip) {
            p.hal_pipeline = pipe;
            p.pipeline_depth = want_depth;
            p.pipeline_cull = want_cull;
            p.pipeline_blend = want_blend;
            p.pipeline_stencil = want_stencil;
            p.pipeline_stencil_back = want_stencil_back;
            p.pipeline_topology = draw_topology;
            p.pipeline_line_width = line_width;
            p.pipeline_a2c = fixed.sample_alpha_to_coverage;
            p.pipeline_scov = fixed.sample_coverage;
            p.pipeline_scov_value = fixed.sample_coverage_value;
            p.pipeline_scov_invert = fixed.sample_coverage_invert;
        } else {
            p.hal_pipeline_noflip = pipe;
            p.pipeline_noflip_depth = want_depth;
            p.pipeline_noflip_cull = want_cull;
            p.pipeline_noflip_blend = want_blend;
            p.pipeline_noflip_stencil = want_stencil;
            p.pipeline_noflip_stencil_back = want_stencil_back;
            p.pipeline_noflip_topology = draw_topology;
            p.pipeline_noflip_line_width = line_width;
            p.pipeline_noflip_a2c = fixed.sample_alpha_to_coverage;
            p.pipeline_noflip_scov = fixed.sample_coverage;
            p.pipeline_noflip_scov_value = fixed.sample_coverage_value;
            p.pipeline_noflip_scov_invert = fixed.sample_coverage_invert;
        }
    }

    // (Re)build + upload each stage's uniform UBO from its CPU block bytes. The VS block binds
    // at binding 0, the FS block at binding 1 (samplers are at binding 2+).
    var vs_ubo = buildStageUbo(dev, &p.vs_uniform_bytes, &p.vs_uniform_hal, p.uniform_dirty) catch {
        obj_lock.unlock();
        setError(GL_INVALID_OPERATION);
        return;
    };
    var fs_ubo = buildStageUbo(dev, &p.fs_uniform_bytes, &p.fs_uniform_hal, p.uniform_dirty) catch {
        obj_lock.unlock();
        setError(GL_INVALID_OPERATION);
        return;
    };
    p.uniform_dirty = false;
    // A named uniform block (GLES3 UBO) OVERRIDES the stage's glUniform* storage: the block's
    // binding-point buffer (glBindBufferBase) supplies the stage's UBO bytes instead. The
    // std140 block occupies the same per-stage UBO slot the shader's default block reads.
    if (resolveNamedStageUbo(dev, p, .vertex)) |ub| vs_ubo = ub;
    if (resolveNamedStageUbo(dev, p, .fragment)) |ub| fs_ubo = ub;

    // Collect the combined-image-sampler bindings: for each sampler uniform, the texture
    // bound to its selected unit on GL_TEXTURE_2D -> a HAL TextureBinding at the sampler's
    // binding. The texture's HAL image Resource is lazily (re)built + uploaded here.
    var tex_bindings: [MAX_TEXTURE_UNITS]prism.hal.TextureBinding = undefined;
    var tex_count: usize = 0;
    for (p.samplers.items) |sm| {
        if (sm.unit < 0 or sm.unit >= MAX_TEXTURE_UNITS) continue;
        // A samplerCube resolves GL_TEXTURE_CUBE_MAP, a sampler3D GL_TEXTURE_3D, a sampler2DArray
        // GL_TEXTURE_2D_ARRAY, a sampler2D GL_TEXTURE_2D on its unit (each a separate binding point).
        const tex_id = if (sm.cube)
            bound_texture_cube[@intCast(sm.unit)]
        else if (sm.tex3d)
            bound_texture_3d[@intCast(sm.unit)]
        else if (sm.tex2darray)
            bound_texture_2darray[@intCast(sm.unit)]
        else
            bound_texture_2d[@intCast(sm.unit)];
        if (tex_id == 0) continue;
        const tex = findTexture(tex_id) orelse continue;
        if (tex.width == 0 or tex.height == 0) continue;
        // A GL_OES_depth_texture sampled after a depth render pass: rebuild its RGBA8 bytes
        // (R=G=B=depth) from the rendered f32 depths so texture2D(...).x reads the depth.
        if (tex.is_depth and tex.depth_dirty) finalizeDepthTexture(dev, tex);
        // A depth texture finalized on the HW-shadow path binds the ZF32 sampled image (a real
        // hardware depth compare). Every other texture (incl. the software rgba8 depth-as-color
        // path) binds the ordinary sampled `hal` image.
        const img = if (tex.is_depth and tex.depth_sampled_hal != null)
            tex.depth_sampled_hal.?
        else
            ensureTextureHal(dev, tex) catch {
                obj_lock.unlock();
                setError(GL_INVALID_OPERATION);
                return;
            };
        // A GLES3 sampler object bound to this unit OVERRIDES the texture's own sampler state
        // (glBindSampler). Unit with no sampler object (0) uses the texture's state.
        const so: ?*Sampler = if (bound_sampler[@intCast(sm.unit)] != 0) findSampler(bound_sampler[@intCast(sm.unit)]) else null;
        tex_bindings[tex_count] = .{
            .binding = sm.binding,
            .image = img,
            .filter = if (so) |s| s.mag_filter else tex.mag_filter, // magnification filter
            .min_filter = if (so) |s| s.min_filter else tex.min_filter, // minification filter (within a level)
            // The mip-blend mode takes effect only when the image actually has a chain (a texture
            // with a mipmap min-filter but no glGenerateMipmap samples the base level).
            .mip_filter = if (tex.has_mipmaps) (if (so) |s| s.mip_filter else tex.mip_filter) else .none,
            .address_u = if (so) |s| s.wrap_s else tex.wrap_s,
            .address_v = if (so) |s| s.wrap_t else tex.wrap_t,
            .max_anisotropy = if (so) |s| s.max_anisotropy else tex.max_anisotropy,
            // GL_TEXTURE_BASE_LEVEL / GL_TEXTURE_MAX_LEVEL are TEXTURE state (not part of a sampler
            // object), so always sourced from the texture.
            .base_level = tex.base_level,
            .max_level = tex.max_level,
            // GL_TEXTURE_SWIZZLE_* is texture state too.
            .swizzle = tex.swizzle,
            // GL_TEXTURE_LOD_BIAS: a sampler object overrides the texture's own bias when bound.
            .lod_bias = if (so) |s| s.lod_bias else tex.lod_bias,
            // GL_TEXTURE_MIN_LOD / MAX_LOD: sampler object overrides too.
            .min_lod = if (so) |s| s.min_lod else tex.min_lod,
            .max_lod = if (so) |s| s.max_lod else tex.max_lod,
            // GL_TEXTURE_COMPARE_MODE / _FUNC (sampler2DShadow depth compare). Sampler object overrides.
            .compare_enable = (if (so) |s| s.compare_mode else tex.compare_mode) == GL_COMPARE_REF_TO_TEXTURE,
            .compare_op = compareOpFromGl(if (so) |s| s.compare_func else tex.compare_func) orelse .less_or_equal,
        };
        tex_count += 1;
    }

    const pipeline = if (flip) p.hal_pipeline.? else p.hal_pipeline_noflip.?;

    // Transform feedback (GLES3): if a capture span is active and the program has recorded
    // varyings, run the VS over this draw's vertices and append the captured varyings to the bound
    // GL_TRANSFORM_FEEDBACK_BUFFER (independent of rasterization). Runs under obj_lock (touches the
    // program's varying tables + the buffer registry).
    // A paused span (glPauseTransformFeedback) still draws but captures nothing (the cursors hold).
    if (tf_active and !tf_paused and p.tf_varyings.items.len > 0) {
        captureTransformFeedbackForDraw(dev, p, pipeline, expanded, @intCast(vcount), vs_ubo, fs_ubo);
    }
    // GL_RASTERIZER_DISCARD: primitives are discarded before rasterization, so skip the entire
    // fragment/render path (the capture above already ran). This is the pure-capture idiom.
    if (fixed.rasterizer_discard) {
        obj_lock.unlock();
        return;
    }

    // Resolve the render targets: the default framebuffer's backbuffer, or a bound FBO's
    // color/depth attachments (render-to-texture). Done under obj_lock (it touches the FBO +
    // texture + renderbuffer registries).
    const rt = resolveRenderTargets(dev, surf, fixed.depth_test) orelse {
        obj_lock.unlock();
        return;
    };
    const depth_tex_to_finalize = rt.depth_tex;
    obj_lock.unlock();

    // The depth attachment + its clear: bind only when depth testing is on. A pending
    // glClear(GL_DEPTH_BUFFER_BIT) clears it this draw (the software depth path clears via
    // setDepthTarget). Subsequent draws in the frame keep the accumulated depth.
    var depth_res: ?*prism.hal.Resource = null;
    var depth_clear: ?f32 = null;
    if (fixed.depth_test) {
        depth_res = rt.depth;
        if (depth_res != null and pending_depth_clear) {
            depth_clear = fixed.depth_clear;
            pending_depth_clear = false;
        }
    }

    // The stencil attachment + its clear: the default framebuffer's surface stencil buffer, or a
    // bound FBO's GL_STENCIL_ATTACHMENT renderbuffer (resolved above). Same deferred-clear contract
    // as depth: a pending glClear(GL_STENCIL_BUFFER_BIT) clears it on this draw, later draws
    // accumulate.
    var stencil_res: ?*prism.hal.Resource = null;
    var stencil_clear: ?u8 = null;
    if (fixed.stencil_test) {
        stencil_res = if (bound_framebuffer == 0) (surf.stencilAttachment() catch null) else rt.stencil;
        if (stencil_res != null and pending_stencil_clear) {
            stencil_clear = fixed.stencil_clear;
            pending_stencil_clear = false;
        }
    }

    // Scissor: when GL_SCISSOR_TEST is on, clip this draw to the glScissor box (its GL
    // bottom-left y flipped to the HAL's top-left origin for THIS render target's height).
    const scissor_rect = currentScissorRect(rt.color_h);
    // glViewport: the NDC->window transform + raster clip for THIS render target (null = full RT).
    const viewport_rect = currentViewportRect(rt.color_w, rt.color_h);
    var has_divisor = false;
    for (layout.sources[0..layout.n_attrs]) |s| {
        if (s.divisor > 0) has_divisor = true;
    }
    // MRT: compact the resolved extra color attachments (a contiguous non-null run) into a
    // slice of non-optional resources for the draw. Empty when the FBO has only attachment 0.
    var mrt_buf: [prism.hal.MAX_COLOR_TARGETS - 1]*prism.hal.Resource = undefined;
    for (0..rt.extra_color_count) |i| mrt_buf[i] = rt.extra_colors[i].?;
    const mrt = mrt_buf[0..rt.extra_color_count];
    if (!has_divisor) {
        // Fast path: one draw, all instances (per-vertex attributes; gl_InstanceIndex 0..N-1).
        ctx.drawArraysUboTarget(rt.color, pipeline, expanded, 0, @intCast(vcount), instance_count, 0, null, vs_ubo, fs_ubo, depth_res, depth_clear, stencil_res, stencil_clear, tex_bindings[0..tex_count], scissor_rect, viewport_rect, mrt) catch {
            setError(GL_INVALID_OPERATION);
        };
    } else {
        // glVertexAttribDivisor emulation: one sub-draw per instance, each pinning every
        // divisor attribute to its element floor(instance/divisor) across all vertices, and
        // carrying first_instance = inst so gl_InstanceIndex stays correct. The clear (if any)
        // fires only on the first sub-draw so later instances accumulate into the same target.
        var inst: u32 = 0;
        while (inst < instance_count) : (inst += 1) {
            // Per-instance vertex buffer: instance 0 uses the base `expanded`. Later instances get
            // their OWN pooled buffer (a copy of the base, then the divisor attributes overwritten).
            // Without this, batched sub-draws would all share one buffer and render the last
            // instance's data (in non-batched mode checkout returns the same slot, so this is a no-op
            // self-copy and behaves exactly as before).
            var inst_res = expanded;
            var inst_map = emap;
            if (inst > 0) {
                const pv = ctx.checkoutVertexBuffer(exp_size) catch {
                    setError(GL_INVALID_OPERATION);
                    break;
                };
                inst_res = pv.res;
                inst_map = pv.map;
                // Copy the base vertex data only when a different buffer was returned (batched).
                // Non-batched, checkout returns the same pool slot as `expanded`, so inst_map IS emap
                // and a memcpy would self-alias (Zig @memcpy panics). The base data is already there.
                if (inst_res != expanded) @memcpy(inst_map[0..exp_size], emap[0..exp_size]);
            }
            for (layout.sources[0..layout.n_attrs], 0..) |s, si| {
                if (s.divisor == 0) continue;
                const elem = inst / s.divisor;
                const so = @as(usize, elem) * s.src_stride + s.src_offset;
                var out_i: usize = 0;
                while (out_i < vcount) : (out_i += 1) {
                    fetchAttrib(inst_map, out_i * stride + dst_attrs[si].offset, s, so);
                }
            }
            const dc: ?f32 = if (inst == 0) depth_clear else null;
            const sc: ?u8 = if (inst == 0) stencil_clear else null;
            ctx.drawArraysUboTarget(rt.color, pipeline, inst_res, 0, @intCast(vcount), 1, inst, null, vs_ubo, fs_ubo, depth_res, dc, stencil_res, sc, tex_bindings[0..tex_count], scissor_rect, viewport_rect, mrt) catch {
                setError(GL_INVALID_OPERATION);
            };
        }
    }
    // A depth render into an FBO depth-texture: mark it dirty so the next sample finalizes
    // its RGBA8 bytes from the rendered depths.
    if (depth_tex_to_finalize) |dt| {
        obj_lock.lock();
        dt.depth_dirty = true;
        obj_lock.unlock();
    }
}

/// Box-downsample one mip level: each `dst` texel (bpp bytes) is the per-channel average of the
/// 2x2 block of `src` texels it covers (edges clamped for odd source dimensions). Works on any
/// byte-per-channel-averaging format (rgba8_unorm / rgba8_srgb here); tightly packed rows.
fn boxDownsample(dst: []u8, dw: u32, dh: u32, src: []const u8, sw: u32, sh: u32, bpp: u32) void {
    var y: u32 = 0;
    while (y < dh) : (y += 1) {
        const sy0 = @min(y * 2, sh - 1);
        const sy1 = @min(y * 2 + 1, sh - 1);
        var x: u32 = 0;
        while (x < dw) : (x += 1) {
            const sx0 = @min(x * 2, sw - 1);
            const sx1 = @min(x * 2 + 1, sw - 1);
            const o00 = (@as(usize, sy0) * sw + sx0) * bpp;
            const o01 = (@as(usize, sy0) * sw + sx1) * bpp;
            const o10 = (@as(usize, sy1) * sw + sx0) * bpp;
            const o11 = (@as(usize, sy1) * sw + sx1) * bpp;
            const doff = (@as(usize, y) * dw + x) * bpp;
            var c: u32 = 0;
            while (c < bpp) : (c += 1) {
                const sum = @as(u32, src[o00 + c]) + src[o01 + c] + src[o10 + c] + src[o11 + c];
                dst[doff + c] = @intCast((sum + 2) / 4);
            }
        }
    }
}

/// Lazily build (or rebuild on a dirty upload) the texture's HAL image Resource from its
/// CPU RGBA8 texels, and return it. Caller holds obj_lock. The image is rgba8_unorm with
/// the .sampled usage so the software driver samples it (the ICD's vkcube texture path).
/// When the texture requested mipmaps (glGenerateMipmap) and is an 8-bit format, the image
/// carries a full box-downsampled mip chain (mip_levels > 1) the sampler minifies through.
fn ensureTextureHal(dev: prism.hal.Device, tex: *Texture) prism.Error!*prism.hal.Resource {
    if (tex.hal != null and !tex.hal_dirty) return tex.hal.?;
    // A render target is rgba8_unorm UNLESS it is a float attachment (rgba16f / rgba32f, rendered +
    // read back at full precision) or an sRGB attachment (GL_SRGB8_ALPHA8, which makes the ROP encode
    // linear->sRGB on write). A sampled-only texture always uses its stored texel format.
    const rt_keep = tex.is_rt and (tex.format == .rgba16_float or tex.format == .r32g32b32a32_float or tex.format == .rgba8_srgb);
    const img_format: prism.hal.Format = if (tex.is_rt and !rt_keep) .rgba8_unorm else tex.format;
    const bpp = img_format.bytesPerPixel();
    // A mip chain is built only for a glGenerateMipmap'd, non-RT/-depth, 8-bit texture (the box
    // downsample averages bytes per channel; float formats keep a single level for now).
    const want_levels: u8 = if (tex.has_mipmaps and !tex.is_rt and !tex.is_depth and bpp == 4)
        prism.hal.mipLevelCount(tex.width, tex.height)
    else
        1;
    // (Re)create the image if absent OR the level count changed (glGenerateMipmap toggled mips on).
    if (tex.hal != null and tex.hal_levels != want_levels) {
        dev.destroyResource(tex.hal.?);
        tex.hal = null;
    }
    if (tex.hal == null) {
        const img = try dev.createResource(.{
            .image = .{
                .width = tex.width,
                .height = tex.height,
                // A render-to-texture target is always the plain 8-bit color format (the RT
                // render path is rgba8). A sampled-only texture uses its stored texel format
                // (rgba8_srgb / rgba16_float / r32g32b32a32_float) so the sampler decodes it.
                .format = img_format,
                .mip_levels = want_levels,
                // A cubemap image: 6 faces packed in `bytes`. The driver sizes the backing to
                // 6 * width*height*bpp and the sampler picks the face from the direction.
                .cube = tex.is_cube,
                // A 3D image: `depth` slices packed slice-major. A sampler3D interpolates it. A 2D
                // array uses the SAME layer-major backing (`array = true` selects sampler2DArray).
                .depth = if (tex.is_3d or tex.is_array) tex.depth else 1,
                .array = tex.is_array,
                // A render-to-texture target is rendered into AND sampled. A plain texture is
                // only sampled. The RT needs render_target usage (+ copy_src for readback).
                .usage = if (tex.is_rt) .{ .sampled = true, .render_target = true, .copy_src = true } else .{ .sampled = true },
            },
        });
        tex.hal = img;
        tex.hal_dev = dev;
        tex.hal_levels = want_levels;
    }
    const map = try dev.mapResource(tex.hal.?);
    if (tex.is_cube) {
        // A cubemap: `tex.bytes` holds the 6 face BASE levels packed tightly (each w*h*bpp),
        // while the HAL image lays out 6 face mip CHAINS. Copy each face base into its chain
        // slot, then generate that face's mip levels in place.
        const face_base = @as(usize, tex.width) * tex.height * bpp;
        const face_chain = if (want_levels > 1) prism.hal.mipChainBytes(tex.width, tex.height, want_levels, bpp) else face_base;
        var f: usize = 0;
        while (f < 6) : (f += 1) {
            const src_face = f * face_base;
            const dst_face = f * face_chain;
            if (src_face + face_base > tex.bytes.items.len or dst_face + face_base > map.len) break;
            @memcpy(map[dst_face..][0..face_base], tex.bytes.items[src_face..][0..face_base]);
            var level: u8 = 1;
            while (level < want_levels) : (level += 1) {
                genMipLevel(map, dst_face, tex.width, tex.height, level, bpp);
            }
        }
    } else {
        // Level 0 = the CPU texels.
        const n = @min(map.len, tex.bytes.items.len);
        @memcpy(map[0..n], tex.bytes.items[0..n]);
        // Generate + write levels 1..N by box-downsampling the previous level in place in the
        // mapped chain (each level starts at hal.mipLevelOffset, tightly packed per level).
        var level: u8 = 1;
        while (level < want_levels) : (level += 1) {
            genMipLevel(map, 0, tex.width, tex.height, level, bpp);
        }
    }
    tex.hal_dirty = false;
    return tex.hal.?;
}

/// Box-downsample mip `level` from level-1 within a chain based at `base` byte offset in `map`
/// (0 for a 2D texture, the face's chain offset for a cubemap). Bounds-checked (no-op if the
/// destination would overflow `map`).
fn genMipLevel(map: []u8, base: usize, w: u32, h: u32, level: u8, bpp: u32) void {
    const src_sz = prism.hal.mipLevelSize(w, h, level - 1);
    const dst_sz = prism.hal.mipLevelSize(w, h, level);
    const src_off = base + prism.hal.mipLevelOffset(w, h, level - 1, bpp);
    const dst_off = base + prism.hal.mipLevelOffset(w, h, level, bpp);
    const src_len = @as(usize, src_sz[0]) * src_sz[1] * bpp;
    const dst_len = @as(usize, dst_sz[0]) * dst_sz[1] * bpp;
    if (dst_off + dst_len > map.len) return;
    boxDownsample(map[dst_off..][0..dst_len], dst_sz[0], dst_sz[1], map[src_off..][0..src_len], src_sz[0], src_sz[1], bpp);
}

/// Expand a draw `mode` over `count` sequential vertices starting at `first` into a
/// triangle-list index sequence written to `out` (caller-sized). Returns the index count,
/// or null for an unsupported mode (sets GL_INVALID_ENUM).
/// The primitive topology a GL draw `mode` maps to (strips/loops/fans expand to base lists in
/// buildArrayIndices, so the pipeline only ever sees list topologies). null = invalid mode.
fn modeTopology(mode: GLenum) ?prism.hal.Topology {
    return switch (mode) {
        GL_TRIANGLES, GL_TRIANGLE_STRIP, GL_TRIANGLE_FAN => .triangle_list,
        GL_LINES, GL_LINE_STRIP, GL_LINE_LOOP => .line_list,
        GL_POINTS => .point_list,
        else => null,
    };
}

/// The topology the currently-recorded draw uses; set by each draw entry from its mode and
/// read at pipeline (re)build so the pipeline is keyed on the primitive kind.
threadlocal var draw_topology: prism.hal.Topology = .triangle_list;

fn buildArrayIndices(mode: GLenum, first: u32, count: u32, out: *std.ArrayListUnmanaged(u32)) ?usize {
    out.clearRetainingCapacity();
    switch (mode) {
        GL_TRIANGLES => {
            var i: u32 = 0;
            while (i + 3 <= count) : (i += 3) {
                out.append(gpa, first + i) catch return null;
                out.append(gpa, first + i + 1) catch return null;
                out.append(gpa, first + i + 2) catch return null;
            }
        },
        GL_LINES => {
            var i: u32 = 0;
            while (i + 2 <= count) : (i += 2) {
                out.append(gpa, first + i) catch return null;
                out.append(gpa, first + i + 1) catch return null;
            }
        },
        GL_LINE_STRIP => {
            var i: u32 = 0;
            while (i + 1 < count) : (i += 1) {
                out.append(gpa, first + i) catch return null;
                out.append(gpa, first + i + 1) catch return null;
            }
        },
        GL_LINE_LOOP => {
            var i: u32 = 0;
            while (i + 1 < count) : (i += 1) {
                out.append(gpa, first + i) catch return null;
                out.append(gpa, first + i + 1) catch return null;
            }
            if (count >= 2) { // close the loop: last -> first
                out.append(gpa, first + count - 1) catch return null;
                out.append(gpa, first) catch return null;
            }
        },
        GL_POINTS => {
            var i: u32 = 0;
            while (i < count) : (i += 1) out.append(gpa, first + i) catch return null;
        },
        GL_TRIANGLE_STRIP => {
            // v0,v1,v2 ; v2,v1,v3 ; v2,v3,v4 ; ... (alternate winding so all tris share the
            // front face - the standard strip-to-list expansion).
            var i: u32 = 0;
            while (i + 3 <= count) : (i += 1) {
                if (i % 2 == 0) {
                    out.append(gpa, first + i) catch return null;
                    out.append(gpa, first + i + 1) catch return null;
                    out.append(gpa, first + i + 2) catch return null;
                } else {
                    out.append(gpa, first + i + 1) catch return null;
                    out.append(gpa, first + i) catch return null;
                    out.append(gpa, first + i + 2) catch return null;
                }
            }
        },
        GL_TRIANGLE_FAN => {
            // v0,v1,v2 ; v0,v2,v3 ; ...
            var i: u32 = 1;
            while (i + 1 < count) : (i += 1) {
                out.append(gpa, first) catch return null;
                out.append(gpa, first + i) catch return null;
                out.append(gpa, first + i + 1) catch return null;
            }
        },
        else => {
            setError(GL_INVALID_ENUM);
            return null;
        },
    }
    return out.items.len;
}

threadlocal var draw_indices: std.ArrayListUnmanaged(u32) = .empty;

/// glDrawArrays(mode, first, count): rasterize `count` vertices of the bound program from
/// the enabled vertex-attribute arrays. Supports GL_TRIANGLES, GL_TRIANGLE_STRIP (es2gears'
/// gears), and GL_TRIANGLE_FAN by expanding to a triangle list. Drives the depth-tested,
/// culled, uniform-bound HAL flow (drawTriangleList). Same software path the Vulkan ICD's
/// depth and UBO draws use. Nothing is reimplemented here.
pub fn drawArrays(mode: GLenum, first: GLint, count: GLsizei) void {
    if (count < 0 or first < 0) {
        setError(GL_INVALID_VALUE);
        return;
    }
    if (count == 0) return;
    const n = buildArrayIndices(mode, @intCast(first), @intCast(count), &draw_indices) orelse return;
    if (n == 0) return;
    draw_topology = modeTopology(mode) orelse .triangle_list;
    drawTriangleList(draw_indices.items[0..n]);
}

/// glDrawArraysInstanced(mode, first, count, instancecount): draw `count` vertices
/// `instancecount` times, the VS seeing gl_InstanceID = 0..instancecount-1 (GLES3 / the
/// ANGLE_instanced_arrays path). Per-instance data is typically a uniform-block array the VS
/// indexes by gl_InstanceID. This is the common many-quad UI pattern.
pub fn drawArraysInstanced(mode: GLenum, first: GLint, count: GLsizei, instancecount: GLsizei) void {
    if (count < 0 or first < 0 or instancecount < 0) {
        setError(GL_INVALID_VALUE);
        return;
    }
    if (count == 0 or instancecount == 0) return;
    const n = buildArrayIndices(mode, @intCast(first), @intCast(count), &draw_indices) orelse return;
    if (n == 0) return;
    draw_topology = modeTopology(mode) orelse .triangle_list;
    drawTriangleListInstanced(draw_indices.items[0..n], @intCast(instancecount));
}

/// glDrawElements(mode, count, type, offset): indexed draw. The index buffer is the bound
/// GL_ELEMENT_ARRAY_BUFFER. `offset` is a byte offset into it (the GLES VBO convention).
/// Supports UNSIGNED_BYTE/SHORT/INT indices + GL_TRIANGLES/STRIP/FAN. Expands to a triangle
/// list and rides the same depth+UBO HAL draw path.
pub fn drawElements(mode: GLenum, count: GLsizei, index_type: GLenum, offset: usize) void {
    drawElementsInstanced(mode, count, index_type, offset, 1);
}

/// glDrawElementsInstanced: glDrawElements `instancecount` times with gl_InstanceID = the
/// instance number (GLES3). Shares the index-expansion path. Only the draw repeats.
pub fn drawElementsInstanced(mode: GLenum, count: GLsizei, index_type: GLenum, offset: usize, instancecount: GLsizei) void {
    if (count < 0 or instancecount < 0) {
        setError(GL_INVALID_VALUE);
        return;
    }
    if (count == 0 or instancecount == 0) return;
    if (modeTopology(mode) == null) {
        setError(GL_INVALID_ENUM); // reject an unknown primitive mode before the expansion switch
        return;
    }
    const elem_size: usize = switch (index_type) {
        GL_UNSIGNED_BYTE => 1,
        GL_UNSIGNED_SHORT => 2,
        GL_UNSIGNED_INT => 4,
        else => {
            setError(GL_INVALID_ENUM);
            return;
        },
    };
    if (mode != GL_TRIANGLES and mode != GL_TRIANGLE_STRIP and mode != GL_TRIANGLE_FAN) {
        setError(GL_INVALID_ENUM);
        return;
    }
    obj_lock.lock();
    const ib = findBuffer(bound_element_buffer) orelse {
        obj_lock.unlock();
        setError(GL_INVALID_OPERATION); // no element-array buffer bound
        return;
    };
    // Read the raw indices out of the bound element buffer at `offset`.
    var raw: std.ArrayListUnmanaged(u32) = .empty;
    defer raw.deinit(gpa);
    const ucount: usize = @intCast(count);
    const bytes = ib.bytes.items;
    var i: usize = 0;
    var ok = true;
    while (i < ucount) : (i += 1) {
        const bo = offset + i * elem_size;
        if (bo + elem_size > bytes.len) {
            ok = false;
            break;
        }
        const v: u32 = switch (elem_size) {
            1 => bytes[bo],
            2 => std.mem.bytesToValue(u16, bytes[bo..][0..2]),
            4 => std.mem.bytesToValue(u32, bytes[bo..][0..4]),
            else => unreachable,
        };
        raw.append(gpa, v) catch {
            ok = false;
            break;
        };
    }
    obj_lock.unlock();
    if (!ok or raw.items.len == 0) {
        if (!ok) setError(GL_INVALID_OPERATION);
        return;
    }
    // Expand the index list to a base-list (triangle/line/point) per the primitive mode.
    draw_topology = modeTopology(mode) orelse .triangle_list;
    draw_indices.clearRetainingCapacity();
    const r = raw.items;
    switch (mode) {
        GL_TRIANGLES => {
            var k: usize = 0;
            while (k + 3 <= r.len) : (k += 3) {
                draw_indices.append(gpa, r[k]) catch return;
                draw_indices.append(gpa, r[k + 1]) catch return;
                draw_indices.append(gpa, r[k + 2]) catch return;
            }
        },
        GL_LINES => {
            var k: usize = 0;
            while (k + 2 <= r.len) : (k += 2) {
                draw_indices.append(gpa, r[k]) catch return;
                draw_indices.append(gpa, r[k + 1]) catch return;
            }
        },
        GL_LINE_STRIP => {
            var k: usize = 0;
            while (k + 1 < r.len) : (k += 1) {
                draw_indices.append(gpa, r[k]) catch return;
                draw_indices.append(gpa, r[k + 1]) catch return;
            }
        },
        GL_LINE_LOOP => {
            var k: usize = 0;
            while (k + 1 < r.len) : (k += 1) {
                draw_indices.append(gpa, r[k]) catch return;
                draw_indices.append(gpa, r[k + 1]) catch return;
            }
            if (r.len >= 2) {
                draw_indices.append(gpa, r[r.len - 1]) catch return;
                draw_indices.append(gpa, r[0]) catch return;
            }
        },
        GL_POINTS => {
            for (r) |v| draw_indices.append(gpa, v) catch return;
        },
        GL_TRIANGLE_STRIP => forEachRestartRun(r, index_type, expandStripRun),
        GL_TRIANGLE_FAN => forEachRestartRun(r, index_type, expandFanRun),
        else => unreachable,
    }
    drawTriangleListInstanced(draw_indices.items, @intCast(instancecount));
}

/// The GL_PRIMITIVE_RESTART_FIXED_INDEX value for an index type (its max representable value).
fn restartIndexFor(index_type: GLenum) u32 {
    return switch (index_type) {
        GL_UNSIGNED_BYTE => 0xFF,
        GL_UNSIGNED_SHORT => 0xFFFF,
        else => 0xFFFFFFFF,
    };
}

/// Call `expand` on each maximal RUN of `r` that contains no restart index. When primitive
/// restart is disabled the whole list is one run (unchanged behavior). When enabled, a restart
/// index (the type's max value) splits the strip/fan so each run is an independent primitive.
fn forEachRestartRun(r: []const u32, index_type: GLenum, comptime expand: fn ([]const u32) void) void {
    if (!primitive_restart_fixed) {
        expand(r);
        return;
    }
    const restart = restartIndexFor(index_type);
    var start: usize = 0;
    var k: usize = 0;
    while (k <= r.len) : (k += 1) {
        if (k == r.len or r[k] == restart) {
            if (k > start) expand(r[start..k]);
            start = k + 1;
        }
    }
}

/// Expand one triangle-strip run into the draw_indices triangle list (alternating winding).
fn expandStripRun(run: []const u32) void {
    var k: usize = 0;
    while (k + 3 <= run.len) : (k += 1) {
        if (k % 2 == 0) {
            draw_indices.append(gpa, run[k]) catch return;
            draw_indices.append(gpa, run[k + 1]) catch return;
            draw_indices.append(gpa, run[k + 2]) catch return;
        } else {
            draw_indices.append(gpa, run[k + 1]) catch return;
            draw_indices.append(gpa, run[k]) catch return;
            draw_indices.append(gpa, run[k + 2]) catch return;
        }
    }
}

/// Expand one triangle-fan run into the draw_indices triangle list (pivot = the run's first vertex).
fn expandFanRun(run: []const u32) void {
    if (run.len < 3) return;
    var k: usize = 1;
    while (k + 1 < run.len) : (k += 1) {
        draw_indices.append(gpa, run[0]) catch return;
        draw_indices.append(gpa, run[k]) catch return;
        draw_indices.append(gpa, run[k + 1]) catch return;
    }
}

// --- Tests ------------------------------------------------------------------

test "boxDownsample averages each 2x2 block per channel (mip generation)" {
    // A 2x2 RGBA8 source: R=(0,100,200,255) in raster order across one channel. The 1x1 dst is
    // the average. Build distinct per-channel values to catch a channel mix-up.
    const src = [_]u8{
        0, 10, 20, 30, // texel (0,0)
        100, 110, 120, 130, // texel (1,0)
        200, 210, 220, 230, // texel (0,1)
        40, 50, 60, 70, // texel (1,1)
    };
    var dst = [_]u8{0} ** 4;
    boxDownsample(&dst, 1, 1, &src, 2, 2, 4);
    // Channel 0: round((0+100+200+40)/4) = round(85) = 85; ch1: (10+110+210+50)/4=95; etc.
    try std.testing.expectEqual(@as(u8, 85), dst[0]);
    try std.testing.expectEqual(@as(u8, 95), dst[1]);
    try std.testing.expectEqual(@as(u8, 105), dst[2]);
    try std.testing.expectEqual(@as(u8, 115), dst[3]);

    // A 1x1 source (odd/degenerate): every dst texel clamps to the single source texel.
    const src1 = [_]u8{ 42, 43, 44, 45 };
    var dst1 = [_]u8{0} ** 4;
    boxDownsample(&dst1, 1, 1, &src1, 1, 1, 4);
    try std.testing.expectEqual(@as(u8, 42), dst1[0]);
    try std.testing.expectEqual(@as(u8, 45), dst1[3]);
}

test "mipFilterOf maps the GL min-filter to the mip-blend mode" {
    try std.testing.expectEqual(prism.hal.MipFilter.none, mipFilterOf(GL_NEAREST));
    try std.testing.expectEqual(prism.hal.MipFilter.none, mipFilterOf(GL_LINEAR));
    try std.testing.expectEqual(prism.hal.MipFilter.nearest, mipFilterOf(GL_NEAREST_MIPMAP_NEAREST));
    try std.testing.expectEqual(prism.hal.MipFilter.nearest, mipFilterOf(GL_LINEAR_MIPMAP_NEAREST));
    try std.testing.expectEqual(prism.hal.MipFilter.linear, mipFilterOf(GL_NEAREST_MIPMAP_LINEAR));
    try std.testing.expectEqual(prism.hal.MipFilter.linear, mipFilterOf(GL_LINEAR_MIPMAP_LINEAR));
}

test "texelBytes accepts the new (format, type) combinations and rejects mismatches" {
    try std.testing.expectEqual(@as(?usize, 4), texelBytes(GL_BGRA_EXT, GL_UNSIGNED_BYTE));
    try std.testing.expectEqual(@as(?usize, 2), texelBytes(GL_RGB, GL_UNSIGNED_SHORT_5_6_5));
    try std.testing.expectEqual(@as(?usize, 2), texelBytes(GL_RGBA, GL_UNSIGNED_SHORT_4_4_4_4));
    try std.testing.expectEqual(@as(?usize, 2), texelBytes(GL_RGBA, GL_UNSIGNED_SHORT_5_5_5_1));
    // 5_6_5 only pairs with RGB. 4_4_4_4 / 5_5_5_1 only with RGBA.
    try std.testing.expectEqual(@as(?usize, null), texelBytes(GL_RGBA, GL_UNSIGNED_SHORT_5_6_5));
    try std.testing.expectEqual(@as(?usize, null), texelBytes(GL_RGB, GL_UNSIGNED_SHORT_4_4_4_4));
    // The existing byte formats still work.
    try std.testing.expectEqual(@as(?usize, 4), texelBytes(GL_RGBA, GL_UNSIGNED_BYTE));
    try std.testing.expectEqual(@as(?usize, 1), texelBytes(GL_LUMINANCE, GL_UNSIGNED_BYTE));
}

test "decodeTexel: BGRA swaps B<->R; packed 16-bit types expand to full-scale RGBA8" {
    var d: [4]u8 = undefined;
    // BGRA8 [B,G,R,A] = [30,20,10,40] -> RGBA [10,20,30,40].
    decodeTexel(GL_BGRA_EXT, GL_UNSIGNED_BYTE, &[_]u8{ 30, 20, 10, 40 }, &d);
    try std.testing.expectEqual([4]u8{ 10, 20, 30, 40 }, d);

    // 5_6_5 (little-endian u16). 0xF800 = R max -> (255,0,0,255).
    decodeTexel(GL_RGB, GL_UNSIGNED_SHORT_5_6_5, &[_]u8{ 0x00, 0xF8 }, &d);
    try std.testing.expectEqual([4]u8{ 255, 0, 0, 255 }, d);
    // 0x07E0 = G max -> (0,255,0,255).
    decodeTexel(GL_RGB, GL_UNSIGNED_SHORT_5_6_5, &[_]u8{ 0xE0, 0x07 }, &d);
    try std.testing.expectEqual([4]u8{ 0, 255, 0, 255 }, d);
    // 0xFFFF = white opaque.
    decodeTexel(GL_RGB, GL_UNSIGNED_SHORT_5_6_5, &[_]u8{ 0xFF, 0xFF }, &d);
    try std.testing.expectEqual([4]u8{ 255, 255, 255, 255 }, d);

    // 4_4_4_4: 0xF00F -> R=0xF(255), A=0xF(255), G=B=0.
    decodeTexel(GL_RGBA, GL_UNSIGNED_SHORT_4_4_4_4, &[_]u8{ 0x0F, 0xF0 }, &d);
    try std.testing.expectEqual([4]u8{ 255, 0, 0, 255 }, d);

    // 5_5_5_1: 0xF801 -> R max, A bit set -> (255,0,0,255).
    decodeTexel(GL_RGBA, GL_UNSIGNED_SHORT_5_5_5_1, &[_]u8{ 0x01, 0xF8 }, &d);
    try std.testing.expectEqual([4]u8{ 255, 0, 0, 255 }, d);
    // 0x0000 -> fully transparent black (the 1-bit alpha is 0).
    decodeTexel(GL_RGBA, GL_UNSIGNED_SHORT_5_5_5_1, &[_]u8{ 0x00, 0x00 }, &d);
    try std.testing.expectEqual([4]u8{ 0, 0, 0, 0 }, d);
}

test "dxtColorBlock decodes RGB565 endpoints + interpolates the 4-color palette" {
    // c0 = red (RGB565 0xF800), c1 = blue (0x001F); 4-color mode (c0 > c1). Index byte 0xE4
    // selects texels (0,0)=idx0, (1,0)=idx1, (2,0)=idx2, (3,0)=idx3; the rest idx0 (red).
    const blk = [_]u8{ 0x00, 0xF8, 0x1F, 0x00, 0xE4, 0x00, 0x00, 0x00 };
    var out: [4 * 4 * 4]u8 = undefined;
    @memset(&out, 0);
    dxtColorBlock(&blk, &out, 4 * 4, 4, 4, false);
    // texel(0,0) = c0 (red), texel(1,0) = c1 (blue), both opaque.
    try std.testing.expect(out[0] > 250 and out[1] < 8 and out[2] < 8 and out[3] == 255);
    try std.testing.expect(out[4] < 8 and out[5] < 8 and out[6] > 250 and out[7] == 255);
    // texel(2,0) = (2*c0+c1)/3: mostly red, some blue.
    try std.testing.expect(out[8] > 150 and out[10] > 40 and out[10] < 130);
    // DXT1 punchthrough: c0<=c1 with index 3 -> transparent black.
    const pblk = [_]u8{ 0x1F, 0x00, 0x00, 0xF8, 0xFF, 0xFF, 0xFF, 0xFF }; // c0=blue<c1=red, all idx3
    @memset(&out, 9);
    dxtColorBlock(&pblk, &out, 4 * 4, 4, 4, true);
    try std.testing.expectEqual(@as(u8, 0), out[3]); // alpha 0 (transparent)
}

test "rgtcChannel decodes the interpolated table (the DXT5-alpha / BC4-red kernel)" {
    // a0=255, a1=0 (a0>a1 -> 8-value table). All 16 indices 0 -> value 255; written to channel 3.
    var out: [4 * 4 * 4]u8 = undefined;
    @memset(&out, 0);
    const ablk = [_]u8{ 255, 0, 0, 0, 0, 0, 0, 0 }; // endpoints 255/0, all indices 0
    rgtcChannel(&ablk, &out, 4 * 4, 4, 4, 3);
    try std.testing.expectEqual(@as(u8, 255), out[3]); // index 0 -> a0 = 255
    try std.testing.expectEqual(@as(u8, 255), out[(15) * 4 + 3]);
    // All indices 1 -> a1 = 0. Index 1 in every texel: each 3-bit field = 1 -> pack 0b001 x16.
    @memset(&out, 7);
    var ablk2 = [_]u8{ 255, 0, 0, 0, 0, 0, 0, 0 };
    // 16 3-bit indices all = 1: bit pattern 001 repeated. Fill bytes 2..7 (48 bits).
    var bits: u64 = 0;
    inline for (0..16) |i| bits |= @as(u64, 1) << (i * 3);
    inline for (0..6) |b| ablk2[2 + b] = @truncate(bits >> (8 * b));
    rgtcChannel(&ablk2, &out, 4 * 4, 4, 4, 3);
    try std.testing.expectEqual(@as(u8, 0), out[3]); // index 1 -> a1 = 0
    // Same kernel to channel 0 (BC4 red): index 0 -> 255 in R.
    @memset(&out, 0);
    rgtcChannel(&ablk, &out, 4 * 4, 4, 4, 0);
    try std.testing.expectEqual(@as(u8, 255), out[0]);
}

test "etc1DecodeBlock decodes an individual-mode solid block to RGBA8" {
    // Individual mode (diff=0, flip=0): both sub-blocks RGB444 = (F,0,0) -> red; table 0;
    // all 16 pixel selectors 0 -> modifier[0][0] = +2. Each texel = (255, 0+2, 0+2, 255).
    //   byte0 = R1<<4|R2 = 0xFF, byte1 = G1<<4|G2 = 0, byte2 = B1<<4|B2 = 0,
    //   byte3 = table1<<5|table2<<2|diff<<1|flip = 0, bytes4-7 = 0 (all indices 0).
    const blk = [_]u8{ 0xFF, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 };
    var out: [4 * 4 * 4]u8 = undefined;
    @memset(&out, 0);
    etc1DecodeBlock(&blk, &out, 4 * 4, 4, 4);
    var i: usize = 0;
    while (i < 16) : (i += 1) {
        try std.testing.expect(out[i * 4 + 0] > 250); // red
        try std.testing.expect(out[i * 4 + 1] < 20);
        try std.testing.expect(out[i * 4 + 2] < 20);
        try std.testing.expectEqual(@as(u8, 255), out[i * 4 + 3]); // opaque
    }
    // A partial edge block (bw=2, bh=3) must only touch its valid region.
    var out2: [4 * 4 * 4]u8 = undefined;
    @memset(&out2, 7);
    etc1DecodeBlock(&blk, &out2, 4 * 4, 2, 3);
    // (x=3,y=3) is outside the 2x3 valid region -> untouched sentinel.
    try std.testing.expectEqual(@as(u8, 7), out2[(3 * 4 + 3) * 4 + 0]);
    // (x=1,y=2) is inside -> decoded red.
    try std.testing.expect(out2[(2 * 4 + 1) * 4 + 0] > 250);
}

test "getString reports Prism GLES strings" {
    try std.testing.expectEqualStrings("Prism", std.mem.span(@as([*:0]const u8, @ptrCast(getString(GL_VENDOR).?))));
    try std.testing.expectEqualStrings("Prism", std.mem.span(@as([*:0]const u8, @ptrCast(getString(GL_RENDERER).?))));
    try std.testing.expect(std.mem.indexOf(u8, std.mem.span(@as([*:0]const u8, @ptrCast(getString(GL_VERSION).?))), "Prism") != null);
    try std.testing.expect(getString(0x1234) == null);
}

test "clearColor clamps to [0,1]" {
    clearColor(2.0, -1.0, 0.5, 0.25);
    try std.testing.expectEqual(@as(f32, 1.0), clear_color.r);
    try std.testing.expectEqual(@as(f32, 0.0), clear_color.g);
    try std.testing.expectEqual(@as(f32, 0.5), clear_color.b);
    try std.testing.expectEqual(@as(f32, 0.25), clear_color.a);
}

test "glViewport rejects negative dims with GL_INVALID_VALUE" {
    _ = getError();
    setViewport(0, 0, -4, 4);
    try std.testing.expectEqual(GL_INVALID_VALUE, getError());
    setViewport(1, 2, 8, 16);
    try std.testing.expectEqual(GL_NO_ERROR, getError());
    try std.testing.expectEqualSlices(GLint, &.{ 1, 2, 8, 16 }, &getViewport());
}

test "glClear with no current context is GL_INVALID_OPERATION" {
    state.makeCurrent(null, null, null); // ensure no leftover current from another test
    _ = getError();
    clear(GL_COLOR_BUFFER_BIT);
    try std.testing.expectEqual(GL_INVALID_OPERATION, getError());
}

test "glClear rejects an unknown mask bit" {
    _ = getError();
    clear(0x1); // not a valid GL clear bit
    try std.testing.expectEqual(GL_INVALID_VALUE, getError());
}

test "GL_EXTENSIONS advertises the extensions Prism implements" {
    const ext = std.mem.span(@as([*:0]const u8, @ptrCast(getString(GL_EXTENSIONS).?)));
    // glmark2 shadow/refract gate on depth_texture. buffer:map gates on mapbuffer.
    try std.testing.expect(std.mem.indexOf(u8, ext, "GL_OES_depth_texture") != null);
    try std.testing.expect(std.mem.indexOf(u8, ext, "GL_OES_mapbuffer") != null);
}

test "glGenFramebuffers / glIsFramebuffer / glBindFramebuffer object lifecycle" {
    _ = getError();
    var fb: GLuint = 0;
    genFramebuffers(1, @ptrCast(&fb));
    try std.testing.expect(fb != 0);
    // glIsFramebuffer is FALSE until the name is first bound (GL semantics).
    try std.testing.expectEqual(GL_FALSE, isFramebuffer(fb));
    bindFramebuffer(GL_FRAMEBUFFER, fb);
    try std.testing.expectEqual(GL_TRUE, isFramebuffer(fb));
    try std.testing.expectEqual(GL_NO_ERROR, getError());
    // An empty FBO is incomplete (no attachment).
    try std.testing.expectEqual(GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT, checkFramebufferStatus(GL_FRAMEBUFFER));
    // Binding a bogus name errors.
    bindFramebuffer(GL_FRAMEBUFFER, 0xDEAD);
    try std.testing.expectEqual(GL_INVALID_OPERATION, getError());
    bindFramebuffer(GL_FRAMEBUFFER, 0); // back to the default framebuffer
    deleteFramebuffers(1, @ptrCast(&fb));
    try std.testing.expectEqual(GL_FALSE, isFramebuffer(fb));
}

test "glGenRenderbuffers + glRenderbufferStorage records a depth renderbuffer" {
    _ = getError();
    var rb: GLuint = 0;
    genRenderbuffers(1, @ptrCast(&rb));
    bindRenderbuffer(GL_RENDERBUFFER, rb);
    renderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT16, 32, 24);
    try std.testing.expectEqual(GL_NO_ERROR, getError());
    obj_lock.lock();
    const r = findRenderbuffer(rb).?;
    try std.testing.expect(r.is_depth);
    try std.testing.expectEqual(@as(u32, 32), r.width);
    try std.testing.expectEqual(@as(u32, 24), r.height);
    obj_lock.unlock();
    bindRenderbuffer(GL_RENDERBUFFER, 0);
    deleteRenderbuffers(1, @ptrCast(&rb));
}

test "GL_OES_mapbuffer: glMapBufferOES returns the buffer bytes; unmap invalidates" {
    _ = getError();
    var vbo: GLuint = 0;
    genBuffers(1, @ptrCast(&vbo));
    bindBuffer(GL_ARRAY_BUFFER, vbo);
    const init_data = [_]f32{ 1, 2, 3, 4 };
    bufferData(GL_ARRAY_BUFFER, @sizeOf(@TypeOf(init_data)), &init_data, GL_DYNAMIC_DRAW);
    // Map and write a new vertex through the returned pointer.
    const ptr = mapBufferOES(GL_ARRAY_BUFFER, GL_WRITE_ONLY_OES) orelse return error.MapFailed;
    try std.testing.expectEqual(GL_NO_ERROR, getError());
    const fp: [*]f32 = @ptrCast(@alignCast(ptr));
    fp[0] = 9.0;
    fp[3] = 42.0;
    // A second map while still mapped is an error.
    try std.testing.expect(mapBufferOES(GL_ARRAY_BUFFER, GL_WRITE_ONLY_OES) == null);
    try std.testing.expectEqual(GL_INVALID_OPERATION, getError());
    try std.testing.expectEqual(GL_TRUE, unmapBufferOES(GL_ARRAY_BUFFER));
    // The CPU bytes carry the written values (what the next draw uploads).
    obj_lock.lock();
    const b = findBuffer(vbo).?;
    const stored = std.mem.bytesAsSlice(f32, b.bytes.items[0..16]);
    try std.testing.expectEqual(@as(f32, 9.0), stored[0]);
    try std.testing.expectEqual(@as(f32, 42.0), stored[3]);
    try std.testing.expect(!b.mapped);
    obj_lock.unlock();
    // Unmapping a non-mapped buffer errors.
    try std.testing.expectEqual(GL_FALSE, unmapBufferOES(GL_ARRAY_BUFFER));
    deleteBuffers(1, @ptrCast(&vbo));
}

test "glTexImage2D with GL_DEPTH_COMPONENT marks a depth texture (GL_OES_depth_texture)" {
    _ = getError();
    var tex: GLuint = 0;
    genTextures(1, @ptrCast(&tex));
    activeTexture(GL_TEXTURE0);
    bindTexture(GL_TEXTURE_2D, tex);
    texImage2D(GL_TEXTURE_2D, 0, @bitCast(GL_DEPTH_COMPONENT), 16, 16, 0, GL_DEPTH_COMPONENT, GL_UNSIGNED_SHORT, null);
    try std.testing.expectEqual(GL_NO_ERROR, getError());
    obj_lock.lock();
    const t = findTexture(tex).?;
    try std.testing.expect(t.is_depth);
    try std.testing.expectEqual(@as(u32, 16), t.width);
    obj_lock.unlock();
    bindTexture(GL_TEXTURE_2D, 0);
    deleteTextures(1, @ptrCast(&tex));
}

test "a failed glLinkProgram populates glGetProgramInfoLog (the reason reaches the app)" {
    _ = getError();
    const prog = createProgram();
    defer deleteProgram(prog);
    // No shaders attached -> link fails. The info log must say why, not be blank.
    linkProgram(prog);
    var status: GLint = 1;
    getProgramiv(prog, GL_LINK_STATUS, &status);
    try std.testing.expectEqual(@as(GLint, GL_FALSE), status);
    var len: GLint = 0;
    getProgramiv(prog, GL_INFO_LOG_LENGTH, &len);
    try std.testing.expect(len > 0); // a real diagnostic, not the old hard-coded 0
    var buf: [256]GLchar = undefined;
    var written: GLint = 0;
    getProgramInfoLog(prog, buf.len, &written, &buf);
    try std.testing.expect(written > 0);
    try std.testing.expect(std.mem.indexOf(u8, buf[0..@intCast(written)], "vertex shader") != null);
}
