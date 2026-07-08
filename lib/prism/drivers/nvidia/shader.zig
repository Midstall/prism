const std = @import("std");
const hal = @import("../../hal.zig");
const spirv_graphics = @import("spirv_graphics.zig");

/// A HAL shader module backed by a real NVIDIA GPU: an owned copy of the native
/// assembled SASS bytes plus the stage it targets.
///
/// The HAL `code` is either native SASS (the hand-assembled / gold-reference path)
/// or a Vulkan SPIR-V word stream (the Vulkan ICD path): when `code` starts with
/// the SPIR-V magic word, it is compiled to SASS here (parseSpirv -> Vulcan IR ->
/// vulcan-target.nvidia.isel.compileShader for the stage), so the pipeline always
/// uploads native machine code. The pipeline derives the SPH input/output maps
/// from its vertex layout. The SASS the compiler emits follows that convention.
pub const ShaderModule = struct {
    stage: hal.ShaderStage,
    code: []u8,
    /// GPRs the shader uses, from the isel register allocation (SPIR-V path). 0 = a
    /// native/hand-assembled SASS module whose usage was not measured. The pipeline
    /// floors the SET_PIPELINE_REGISTER_COUNT so a 0 here just keeps the safe default.
    reg_count: u32 = 0,
    /// Whether a fragment module writes gl_FragDepth (from the isel). The pipeline sets
    /// the SPH OMAP_DEPTH bit so the ROP takes the fragment depth from the shader.
    writes_depth: bool = false,
    /// Render targets a fragment module writes (MRT); 1 for single-RT / VS. The pipeline
    /// declares this many color targets in the SPH omap and binds that many color surfaces.
    color_targets: u8 = 1,

    pub fn create(gpa: std.mem.Allocator, desc: hal.ShaderModuleDesc) hal.Error!*ShaderModule {
        const self = gpa.create(ShaderModule) catch return error.OutOfMemory;
        errdefer gpa.destroy(self);
        const a = try assemble(gpa, desc.stage, desc.code);
        self.* = .{ .stage = desc.stage, .code = a.code, .reg_count = a.reg_count, .writes_depth = a.writes_depth, .color_targets = a.color_targets };
        return self;
    }

    const Assembled = struct { code: []u8, reg_count: u32, writes_depth: bool = false, color_targets: u8 = 1 };

    /// Produce the native SASS bytes for a shader module's `code`. If `code` is a
    /// SPIR-V word stream (first dword == the SPIR-V magic) compile it to SASS for
    /// the stage (and capture the isel register count). Otherwise it is already native
    /// SASS and is copied verbatim (reg_count 0 = use the pipeline's safe default).
    fn assemble(gpa: std.mem.Allocator, stage: hal.ShaderStage, code: []const u8) hal.Error!Assembled {
        if (isSpirv(code)) {
            const isel_stage = try spirv_graphics.iselStage(stage);
            const sass = spirv_graphics.compileToSassRegs(gpa, code, isel_stage) catch |e| switch (e) {
                error.OutOfMemory => return error.OutOfMemory,
                // A SPIR-V parse/lowering/codegen failure (an unsupported pattern,
                // a malformed module) is an invalid shader for this backend.
                else => return error.InvalidArgument,
            };
            defer gpa.free(sass.code);
            // Copy the compiled dwords into a byte allocation the module owns (so
            // destroy() frees a plain []u8, matching the alloc's element type).
            const owned = gpa.dupe(u8, std.mem.sliceAsBytes(sass.code)) catch return error.OutOfMemory;
            return .{ .code = owned, .reg_count = sass.reg_count, .writes_depth = sass.writes_depth, .color_targets = sass.color_targets };
        }
        return .{ .code = gpa.dupe(u8, code) catch return error.OutOfMemory, .reg_count = 0 };
    }

    /// Whether a HAL shader `code` byte stream is a SPIR-V module (vs native SASS).
    fn isSpirv(code: []const u8) bool {
        if (code.len < 4 or code.len % 4 != 0) return false;
        const first = std.mem.readInt(u32, code[0..4], .little);
        return first == spirv_graphics.SPIRV_MAGIC;
    }

    pub fn destroy(self: *ShaderModule, gpa: std.mem.Allocator) void {
        gpa.free(self.code);
        gpa.destroy(self);
    }
};
