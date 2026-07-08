//! Userspace Apple Silicon (AGX) GPU driver for the kernel `asahi` DRM driver,
//! the GPU bring-up path for Asahi Linux on Apple M-series machines. Zero C
//! deps. Built as its own subproject (mirrors subproject/nvidia); to be wired
//! into Prism's HAL `apple` driver (lib/prism/drivers/apple.zig) in a later
//! phase.
//!
//! Status (PHASE 3 - AGX compute launch): on top of P1's transport and P2's
//! queue/SUBMIT/sync wiring, this adds src/asahi/compute.zig - the from-scratch
//! AGX COMPUTE encoding (the drm_asahi_cmd_compute body, the CDM control stream,
//! the USC descriptor, and a hand-assembled store-a-constant kernel) plus
//! runComputeConstant, which submits it on a real AGX GPU and reads the result
//! back. The encodings are grounded in Mesa src/asahi + dougallj's applegpu; the
//! live GPU execution is the user's M1 test.
//!
//! Earlier (PHASE 2 - command queue + SUBMIT): on top of P1's transport (open,
//! GET_PARAMS, VM_CREATE, GEM alloc + CPU map + VM_BIND), this adds the
//! QUEUE_CREATE/QUEUE_DESTROY + SUBMIT structs, the core DRM syncobj ioctls
//! (create/destroy/signal/wait), and the Device queue/syncobj/submit methods.
//! The submit struct + sync marshaling are fully wired; the GPU-executing
//! command needs the AGX control-stream/cmdbuf encoding, which is P3 - so SUBMIT
//! is proven up to "queue created + syncobj signal/wait roundtrip" and does NOT
//! hand the kernel a fabricated cmdbuf. AGX command buffers and shaders are the
//! next phases. See examples/asahi-info.zig for the probe the user runs on the M1.

const std = @import("std");

pub const uapi = @import("asahi/uapi.zig");
pub const device = @import("asahi/device.zig");
pub const compute = @import("asahi/compute.zig");
pub const render = @import("asahi/render.zig");

pub const Device = device.Device;
pub const Bo = device.Bo;
pub const GpuInfo = device.GpuInfo;
pub const Queue = device.Queue;
pub const Error = device.Error;

/// CPU<->GPU cache maintenance for CPU-written / GPU-written BOs (aarch64).
/// `cleanToPoC` flushes CPU writes to the point of coherency before the GPU
/// reads (call before a dispatch on each input/storage BO); `cleanInvalidateToPoC`
/// clean+invalidates before the CPU reads the GPU's output (call after the fence).
/// No-op off aarch64. The prism `apple` HAL driver brackets dispatchCompute with
/// these so the GPU loads fresh input instead of a stale recycled-DRAM value.
pub const cleanToPoC = device.cleanToPoC;
pub const cleanInvalidateToPoC = device.cleanInvalidateToPoC;

/// P3: the from-scratch AGX compute launch - build a trivial kernel that stores
/// a known constant to a GPU buffer, submit it, wait the fence, read it back.
pub const runComputeConstant = compute.runComputeConstant;
pub const kComputeConstant = compute.kConstant;
/// Generalized one-shot AGX compute (the Prism `apple` driver's dispatchCompute
/// backend): run a CALLER-supplied kernel with the store target at a CALLER-owned
/// GPU VA, over a caller grid. Allocates/frees ONLY the launch infra (not the
/// caller's storage), so it is repeatable. Same proven path as runComputeConstant.
pub const runCompute = compute.runCompute;
/// Assemble the proven "store a 32-bit constant" AGX compute kernel for a given
/// value (the example + tests supply this as a real AGX kernel to dispatch).
pub const buildConstantKernel = compute.buildConstantKernel;
pub const CONSTANT_KERNEL_SIZE = compute.CONSTANT_KERNEL_SIZE;
/// Assemble the proven "read input, add an immediate, write output" AGX compute
/// kernel: it device_LOADs buffer 0 (u0_u1), iadds the addend, and device_STOREs
/// buffer 1 (u2_u3). The first REAL multi-buffer input->output compute kernel.
pub const buildAddKernel = compute.buildAddKernel;
pub const ADD_KERNEL_SIZE = compute.ADD_KERNEL_SIZE;
/// Assemble the DATA-PARALLEL "input[i] -> output[i] + addend" AGX compute kernel:
/// each hardware thread reads its OWN global index (get_sr sr80), device_LOADs
/// input[idx] (INDEXED device_load on buffer 0 = u0_u1), iadds the addend, and
/// device_STOREs output[idx] (INDEXED device_store on buffer 1 = u2_u3).
/// Dispatched over groups {N,1,1} (local 1x1x1 = N threads), it processes N
/// elements in parallel, each thread its own element - real data-parallel GPU
/// compute on the M1. NO HAL/dispatch change is needed (the dispatch already
/// launches N threads); only this kernel reads the thread index.
pub const buildParallelAddKernel = compute.buildParallelAddKernel;
pub const PARALLEL_ADD_KERNEL_SIZE = compute.PARALLEL_ADD_KERNEL_SIZE;

/// P4: the from-scratch AGX render launch - clear a 64x64 linear RGBA8 buffer +
/// rasterize one solid-color triangle into it, submit it, wait the fence, read a
/// center pixel back. The first from-scratch GPU triangle on Apple Silicon.
pub const runTriangle = render.runTriangle;
pub const kTriColor = render.kTriColor;
/// P4 clear-only discriminator: the same render pass WITHOUT the geometry draw
/// (bg clear + eot store only), to split a render hang into infrastructure vs draw.
pub const runClear = render.runClear;
pub const kClearColor = render.kClearColor;
/// P5 HAL primitive: the PROVEN clear render pass writing a CALLER-supplied color
/// to a CALLER-supplied output BO (the Prism `apple` driver's clear path). Same
/// proven bg/eot/PBE/tile encoding as runClear; only the baked color + output VA
/// are threaded through. 64x64 RGBA8.
pub const clearColorTo = render.clearColorTo;
/// Render geometry the clearColorTo / clear pass is encoded for (64x64 RGBA8).
pub const FB_WIDTH = render.FB_WIDTH;
pub const FB_HEIGHT = render.FB_HEIGHT;
pub const COLOR_SIZE = render.COLOR_SIZE;
pub const COLOR_STRIDE = render.COLOR_STRIDE;
pub const USC_EXEC_BASE = render.USC_EXEC_BASE;
pub const PAGE = render.PAGE;
/// RUNTIME ground-truth dump: decode + print OUR triangle render stream (VDM
/// blocks, PPP records, drm_asahi_cmd_render fields, shader bytes) for a
/// field-by-field diff against a real M1 Mesa capture. NO GPU - runs anywhere.
pub const dumpStream = render.dumpStream;

// The UEFI/baremetal path of asahi-info reads the SoC + AGX GPU identity from
// the FDT (Apple Silicon has no PCI) using Midstall's dtree + conduit libraries
// directly in examples/asahi-info.zig, not a hand-rolled parser here.

test {
    std.testing.refAllDecls(@This());
}
