const std = @import("std");
const builtin = @import("builtin");
const build_options = @import("build-options");

pub const hal = @import("prism/hal.zig");
pub const Device = hal.Device;
pub const Context = hal.Context;
pub const CommandBuffer = hal.CommandBuffer;
pub const Resource = hal.Resource;
pub const ShaderModule = hal.ShaderModule;
pub const Pipeline = hal.Pipeline;
pub const Error = @import("prism/error.zig").Error;

pub const platform = @import("prism/platform.zig");
pub const drivers = @import("prism/drivers.zig");

/// The target-agnostic SPIR-V shader front end (SPIR-V -> Vulcan IR). Drivers
/// lower the IR to their target (NVIDIA SASS, aarch64 JIT).
pub const spirv = @import("prism/spirv.zig");

/// The GLSL-ES shader front end (GLSL ES 1.00 source -> SPIR-V), via Vulcan's
/// `vulcan-glsl`. The EGL/GLES stack uses it so glShaderSource + glCompileShader
/// compile real GLSL into SPIR-V the `spirv` seam above then lowers.
pub const glsl = @import("prism/glsl.zig");

/// The software reference driver (driver + declarative shader IR helpers).
pub const software = drivers.software;

/// The virgl HAL driver: a hardware-accelerated backend that lowers the HAL flow
/// to virgl commands run on the host GPU by virglrenderer, over a comptime-
/// selected transport (lib/prism/drivers/virgl/transport.zig):
///   - freestanding: the Conduit virtio-gpu 3D driver (virtio-mmio SUBMIT_3D)
///   - linux: the kernel virtio-gpu DRM uAPI via /dev/dri/renderD* (EXECBUFFER)
/// On freestanding the transport imports `conduit`. On Linux it uses raw
/// std.os.linux ioctls and needs no conduit dep. It is available on both. (The
/// transport seam is comptime-selected, so the conduit import is only analysed on
/// the freestanding path. The Linux prism build is unaffected and needs no
/// conduit dependency.)
pub const virgl = @import("prism/drivers/virgl.zig");

/// Apple Silicon (AGX) helpers exposed for HAL consumers that need to supply a
/// real AGX compute kernel to Device.dispatchCompute. The `apple` HAL driver
/// takes the kernel as raw AGX bytecode (a .compute ShaderModule). Re-exports
/// the proven hand-assembled "store a 32-bit constant" kernel builder so an
/// example/tool can build one importing only prism. (asahi is always a module dep
/// of prism, so this is present regardless of -Ddrivers.)
pub const apple = struct {
    const asahi = @import("asahi");
    /// Assemble the proven AGX "store a 32-bit constant" compute kernel for a
    /// given value into `out`. Returns the byte length.
    pub const buildConstantKernel = asahi.buildConstantKernel;
    /// Length buildConstantKernel writes (size a caller's kernel buffer must hold).
    pub const CONSTANT_KERNEL_SIZE = asahi.CONSTANT_KERNEL_SIZE;
    /// The constant the default constant-store kernel writes (0xCAFEF00D).
    pub const kComputeConstant = asahi.kComputeConstant;
    /// Assemble the proven multi-buffer "read input, add an immediate, write
    /// output" AGX compute kernel (reads buffers[0]=u0_u1, writes buffers[1]=u2_u3).
    pub const buildAddKernel = asahi.buildAddKernel;
    /// Length buildAddKernel writes (size a caller's kernel buffer must hold).
    pub const ADD_KERNEL_SIZE = asahi.ADD_KERNEL_SIZE;
    /// Assemble the data-parallel "input[i] -> output[i] + addend" AGX compute
    /// kernel: each thread reads its own global index (get_sr sr80) and processes
    /// its own element via indexed device_load/store (reads buffers[0]=u0_u1,
    /// writes buffers[1]=u2_u3). Dispatched over groups {N,1,1} (N threads), all N
    /// elements are processed in parallel.
    pub const buildParallelAddKernel = asahi.buildParallelAddKernel;
    /// Length buildParallelAddKernel writes (size a caller's kernel buffer must hold).
    pub const PARALLEL_ADD_KERNEL_SIZE = asahi.PARALLEL_ADD_KERNEL_SIZE;
};

pub const version: []const u8 = build_options.version;

pub const Driver = @import("prism/driver.zig").Driver;
pub const selectDriver = drivers.select;

test "version is non-empty" {
    try std.testing.expect(version.len > 0);
}

// Pull every internal test into the single prism module's test run. The
// behavior modules are reached transitively via the re-exports above. The
// test-only files are referenced explicitly here.
test {
    _ = drivers;
    _ = platform;
    _ = spirv;
    _ = glsl;
    // The virgl driver + its transport seam (the Linux DRM transport is selected
    // on the host target, so this Sema-checks the DRM ioctl backend).
    _ = virgl;
    _ = @import("prism/log.zig");
    _ = @import("prism/hal_mock_test.zig");
    _ = @import("prism/drivers/software/e2e_test.zig");
    _ = @import("prism/drivers/software/present_e2e_test.zig");
    // The software render benchmark (inert unless PRISM_BENCH is set).
    _ = @import("prism/drivers/software/bench.zig");
}
