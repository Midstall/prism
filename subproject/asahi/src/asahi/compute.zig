//! AGX compute launch - the first from-scratch GPU EXECUTION on Apple Silicon
//! via Prism (P3). Builds, from scratch, a trivial COMPUTE kernel that writes a
//! known 32-bit constant to a GPU buffer, then submits it through the kernel
//! `asahi` DRM driver, waits the fence, and reads the buffer back to prove the
//! AGX cores ran. This is the AGX analogue of the NVIDIA hand-assembled compute
//! SASS milestone.
//!
//! GROUNDING. Every GPU-side byte here is grounded in Mesa's proven AGX
//! encoding (src/asahi) and dougallj's applegpu, NOT guessed:
//!   - the COMPUTE command body: include/uapi/drm/asahi_drm.h
//!     drm_asahi_cmd_compute (see uapi.zig).
//!   - the CDM (compute) control-stream words + the USC descriptor words: the
//!     genxml bit layouts in Mesa src/asahi/genxml/cmdbuf.xml, and the assembly
//!     order in libagx/libagx_dgc.h agx_cdm_launch / agx_cdm_barrier /
//!     agx_cdm_terminate and lib/agx_bg_eot.c (the minimal USC build).
//!   - the shader ISA (mov_imm / device_store / wait / stop): dougallj's
//!     applegpu assembler+disassembler (round-trip verified).
//!
//! The CONSTANT the GPU writes is `kConstant` (a pub const) so the probe and a
//! test can compare it against the read-back word.
//!
//! WHAT IS UNVERIFIED (candidates for M1 iteration - see the report): the live
//! GPU execution itself cannot be tested on this box (no AGX GPU). The struct
//! layouts, the word encodings and the assembled sizes ARE tested here. The CDM
//! barrier word's exact "unk" bits, the helper-program "must be present" rule,
//! and the precise queue usc_exec_base relationship are matched to Mesa but only
//! the M1 run proves them end-to-end.

const std = @import("std");
const uapi = @import("uapi.zig");
const dev_mod = @import("device.zig");

const Device = dev_mod.Device;
const Bo = dev_mod.Bo;
const Error = dev_mod.Error;

/// The constant the kernel stores to output[0]. 0xCAFEF00D = 3405705229.
pub const kConstant: u32 = 0xCAFEF00D;

/// The sentinel the CPU pre-fills the output BO with BEFORE the submit. After
/// the fence, the readback is decoded three ways so the result is HONEST:
///   == kConstant  -> the AGX executed our kernel and the store landed (SUCCESS)
///   == kSentinel  -> the GPU never wrote output[0] (an encoding bug, NOT a
///                    coherency / stale-readback artefact)
///   anything else -> the store hit but wrote a wrong value (note + iterate)
/// This distinguishes "did not store" from "stored but stale". On AGX that
/// distinction is meaningful because the GPU is cache-coherent with the CPU
/// (Asahi docs: "all memory is coherent ... we haven't used a single cache
/// management instruction and everything still works"), so a sentinel survivor
/// means the store really did not execute, not that the read is stale.
pub const kSentinel: u32 = 0xDEADBEEF;

// ---------------------------------------------------------------------------
// GPU virtual-address layout.
//
// Mesa puts the "USC heap" (the region all USC code addresses + the launch
// pipeline offset are relative to) at a 4 GiB-aligned base = usc_exec_base,
// and makes every USC `code` address and the CDM `pipeline` value a 32-bit
// offset from it (lib/agx_device.c agx_usc_addr: addr - shader_base; the queue
// is created with usc_exec_base = shader_base). We mirror that: usc_exec_base
// is 4 GiB-aligned, and the shader binary + USC descriptor live just above it
// so their offsets fit in 32 bits and are small/aligned. The output BO lives
// well above the USC window; the shader reaches it by its FULL 64-bit GPU VA
// (passed in as a uniform), not a USC-relative offset.
// ---------------------------------------------------------------------------

/// 4 GiB-aligned base of the USC executable window == the queue's usc_exec_base.
pub const USC_EXEC_BASE: u64 = 0x1_0000_0000;

/// GPU VA of the shader binary (USC-relative offset = SHADER_VA - USC_EXEC_BASE).
/// Each BO is bound as its own 16 KiB page, so every VA MUST be PAGE-aligned:
/// AGX's MMU is 16 KiB-page and VM_BIND rejects a non-16-KiB-aligned addr with
/// EINVAL (the bug the M1 hit on the first - shader - bind). PAGE alignment also
/// covers the USC needs (16 KiB is a multiple of the shader's 128-byte and the
/// USC block's 64-byte alignment).
pub const SHADER_VA: u64 = USC_EXEC_BASE + PAGE;

/// GPU VA of the USC descriptor block (2*PAGE; still 64-byte aligned so the CDM
/// pipeline field, USC offset >> 6, is exact).
pub const USC_VA: u64 = USC_EXEC_BASE + 2 * PAGE;

/// GPU VA of the CDM control stream (3*PAGE).
pub const CTRL_STREAM_VA: u64 = USC_EXEC_BASE + 3 * PAGE;

/// GPU VA of the UNIFORM-DATA BO (4*PAGE). This buffer HOLDS the 8-byte
/// little-endian OUTPUT_VA: the USC Uniform descriptor's Buffer field points
/// HERE, and the hardware LOADS the 4 halves (= the 64-bit OUTPUT_VA) from it
/// into the uniform registers u0_u1. It is NOT the store target - it is the
/// little buffer that contains the store-target pointer. See buildUsc.
pub const UNIFORM_DATA_VA: u64 = USC_EXEC_BASE + 4 * PAGE;

/// GPU VA of the 4 KiB output BO (the shader's store TARGET). Placed above the
/// 4 GiB USC window so it is clearly outside it; the kernel only needs it
/// mapped, the shader addresses it by the full VA it loads from the uniform-data
/// buffer.
pub const OUTPUT_VA: u64 = USC_EXEC_BASE + 0x1_0000_0000;

/// Page size for the BO allocations.
pub const PAGE: u64 = 0x4000; // 16 KiB (Apple Silicon page size)

// ---------------------------------------------------------------------------
// The compute shader binary.
//
// A minimal AGX (G13 / M1 Pro) kernel. The uniform registers u0_u1 are
// PRELOADED by the hardware with the output BO's 64-bit GPU VA: the USC Uniform
// descriptor names a uniform-data buffer that CONTAINS that 64-bit pointer, and
// the HW loads 4 halves (8 bytes) of it into u0_u1 at half-offset 0 before the
// shader runs (see buildUsc + UNIFORM_DATA_VA). So `[u0_u1 + 0]` resolves to
// OUTPUT_VA, the real store target. The kernel:
//   mov_imm      r0, 0xCAFEF00D     ; load the constant into r0 (r0l:r0h)
//   device_store 0, i32, x, r0, u0_u1, 0, signed   ; store r0 to [u0_u1 + 0]
//   wait         0                  ; ordering barrier
//   stop                            ; end_execution
//
// The shader + the device_store are CORRECT and UNCHANGED. The bug an M1 dmesg
// GPU fault exposed (a WRITE to 0xefdeadbec0, an address that CONTAINS the
// 0xDEADBEEF sentinel) was that the USC Uniform Buffer used to point AT the
// output BO, so the HW loaded u0_u1 = the sentinel the CPU had pre-filled, then
// the store wrote to ~0xDEADBEEF (unmapped). Only the uniform SOURCE was wrong.
//
// Each instruction byte sequence was produced + round-tripped by dougallj's
// applegpu assembler/disassembler (the reference RE tool). Sources (applegpu.py
// master): MovImm32InstructionDesc L2341-2351; DeviceStore/DeviceLoadStore
// L4893-4915 + L5173-5180 (MEMORY_FORMATS i32=2 @L4486, MASK 'x'=1 @L4507,
// MemoryBaseDesc uniform pair @L1952); WaitInstructionDesc L4472; Stop L3552.
// ---------------------------------------------------------------------------

/// The raw little-endian AGX machine code (18 bytes). Comment per instruction
/// cites the applegpu encoding it was assembled from.
pub const shader_code = [_]u8{
    // mov_imm r0, 0xCAFEF00D   (6 bytes)
    //   opcode low7 = 0b1100010 (0x62); bit8=1 selects 32-bit imm form;
    //   imm32 is a contiguous field at bit 16 -> bytes [2..6] little-endian.
    0x62, 0x01, 0x0d, 0xf0, 0xfe, 0xca,
    // device_store 0, i32, x, r0, u0_u1, 0, signed   (8 bytes)
    //   opcode low7 = 0b1000101 (0x45, store); F=i32(2), mask=x(1),
    //   R=r0 (Rt=1), A=u0_u1 (At=1 uniform pair), O=0 imm (Ot=1), signed.
    0x45, 0x01, 0x00, 0x0d, 0x00, 0xc0,
    0x12, 0x00,
    // wait 0   (2 bytes) - wait_for_loads, ordering barrier before stop.
    0x38, 0x00,
    // stop     (2 bytes) - end_execution.
    0x88, 0x00,
};

/// Size of the constant-store kernel buildConstantKernel writes.
pub const CONSTANT_KERNEL_SIZE: usize = shader_code.len;

/// Assemble the "store a 32-bit constant" AGX compute kernel into `out`,
/// returning the byte length (= CONSTANT_KERNEL_SIZE = 18). The kernel is
/// EXACTLY the proven P3 kernel - mov_imm r0, value / device_store i32 x
/// [u0_u1+0], r0 / wait 0 / stop - but with the caller's `value` baked into the
/// mov_imm's 32-bit immediate (bytes [2..6], little-endian). The shape is the
/// round-tripped-through-dougallj-applegpu byte stream documented on shader_code;
/// only the immediate changes. Lets the example + tests supply a real AGX kernel
/// (e.g. buildConstantKernel(buf, 0xCAFEF00D)) to dispatch through the HAL.
pub fn buildConstantKernel(out: []u8, value: u32) usize {
    std.debug.assert(out.len >= shader_code.len);
    @memcpy(out[0..shader_code.len], &shader_code);
    // Patch the mov_imm 32-bit immediate (contiguous field at byte offset 2).
    std.mem.writeInt(u32, out[2..6], value, .little);
    return shader_code.len;
}

// ---------------------------------------------------------------------------
// The multi-buffer "add" compute kernel (input -> output).
//
// A minimal AGX (G13 / M1 Pro) kernel that READS an input buffer, adds an
// immediate, and WRITES an output buffer - the first REAL input->output compute
// through the Prism HAL. The two storage buffers are bound IN ORDER by
// runCompute: buffers[0] -> u0_u1 (input), buffers[1] -> u2_u3 (output) (see the
// MULTI-BUFFER BINDING note on runCompute). The kernel:
//   mov_imm      r1, addend          ; load the addend into r1 (r1l:r1h)
//   device_load  0, i32, x, r0, u0_u1, 0, signed   ; r0 = input  [u0_u1 + 0]
//   iadd         r0, r0, r1                          ; r0 = r0 + addend
//   device_store 0, i32, x, r0, u2_u3, 0, signed   ; output[u2_u3 + 0] = r0
//   wait         0                  ; ordering barrier
//   stop                            ; end_execution
//
// Every instruction was ROUND-TRIPPED through dougallj's applegpu
// assembler+disassembler (full applegpu.py): the whole 34-byte stream walks
// instruction-by-instruction to `stop` with no desync, disassembling to exactly
// the listing above. The addend is loaded into r1 first (rather than as an inline
// iadd immediate) because the AGX iadd inline immediate is only 8-bit (max 255),
// so a 16/32-bit addend like 0xCAFE must come from a register. Sources (applegpu
// master): MovImm32 (62 05 for r1, imm32 @[2..6]); DeviceLoad (05, base u0_u1)/
// DeviceStore (45, base u2_u3 = A-field value 4 at the merged (16,4)+(36,4) bits)
// with F=i32(2), mask=x(1); IAdd reg+reg (0e, Da=r0 Sa=r0 Sb=r1); Wait(38);
// Stop(88).
// ---------------------------------------------------------------------------

/// The raw little-endian AGX machine code for the add kernel (34 bytes). The
/// addend at byte offset 2 (the mov_imm r1 32-bit immediate) is patched by
/// buildAddKernel; the template carries 0 there. Comment per instruction cites
/// the applegpu encoding it was round-tripped from.
pub const add_shader_code = [_]u8{
    // mov_imm r1, <addend>  (6 bytes) - opcode 0x62, dest-r1 selector 0x05,
    //   imm32 contiguous at bytes [2..6] little-endian (patched by buildAddKernel).
    0x62, 0x05, 0x00, 0x00, 0x00, 0x00,
    // device_load 0, i32, x, r0, u0_u1, 0, signed  (8 bytes) - r0 = input[u0_u1+0].
    //   opcode 0x05 (load); F=i32(2), mask=x(1); A=u0_u1 (value 0); R=r0.
    0x05, 0x01, 0x00, 0x0d, 0x00, 0xc0,
    0x12, 0x00,
    // iadd r0, r0, r1  (8 bytes) - r0 = r0 + r1.
    0x0e, 0x01, 0x40, 0x22,
    0x24, 0x00, 0x00, 0x00,
    // device_store 0, i32, x, r0, u2_u3, 0, signed  (8 bytes) - output[u2_u3+0] = r0.
    //   opcode 0x45 (store); A=u2_u3 (merged-field value 4 = low nibble of byte 2).
    0x45, 0x01,
    0x04, 0x0d, 0x00, 0xc0, 0x12, 0x00,
    // wait 0   (2 bytes) - wait_for_loads, ordering barrier before stop.
    0x38, 0x00,
    // stop     (2 bytes) - end_execution.
    0x88, 0x00,
};

/// Size of the add kernel buildAddKernel writes (= 34).
pub const ADD_KERNEL_SIZE: usize = add_shader_code.len;

/// Assemble the "read input, add an immediate, write output" AGX compute kernel
/// into `out`, returning the byte length (= ADD_KERNEL_SIZE = 34). The kernel
/// reads buffer 0 (u0_u1), adds `addend`, and writes buffer 1 (u2_u3). The
/// `addend` is baked into the mov_imm r1 32-bit immediate (bytes [2..6],
/// little-endian) of the round-tripped add_shader_code template. Lets the example
/// + tests supply a real input->output AGX kernel to dispatch through the HAL
/// (e.g. buildAddKernel(buf, 0x0000CAFE)).
pub fn buildAddKernel(out: []u8, addend: u32) usize {
    std.debug.assert(out.len >= add_shader_code.len);
    @memcpy(out[0..add_shader_code.len], &add_shader_code);
    // Patch the mov_imm r1 32-bit immediate (contiguous field at byte offset 2).
    std.mem.writeInt(u32, out[2..6], addend, .little);
    return add_shader_code.len;
}

// ---------------------------------------------------------------------------
// The PARALLEL "add" compute kernel - DATA-PARALLEL input[i] -> output[i].
//
// THE PROOF OF REAL GPU DATA-PARALLELISM. The HAL dispatches this kernel over
// `groups = {N,1,1}` threadgroups with the LOCAL (workgroup) size fixed 1x1x1
// (see buildControlStream), so the launch spawns exactly N hardware threads, and
// thread `t` (t = 0..N-1) sees its own GLOBAL thread index. THE DISPATCH ALREADY
// LAUNCHES N THREADS - no runCompute / dispatchCompute / HAL change is needed for
// parallelism; ONLY the kernel must read its thread index and address its own
// element. Each thread then transforms input[t] -> output[t] entirely
// independently, so all N elements are processed in parallel on the AGX cores.
//
// THE THREAD INDEX. Each thread reads its global thread index from the AGX
// special register `thread_position_in_grid.x` (SR number 80) via `get_sr`. With
// the local size fixed 1x1x1, thread_position_in_grid.x == threadgroup-id.x == the
// thread's index 0..N-1 (the threadgroup-id SR is 0; SR 80 is the equivalent
// global index and is what dougallj's applegpu models in its get_sr executor,
// `D[thread] = thread`). SOURCES for the SR number + the get_sr encoding:
//   - dougallj applegpu (applegpu.py): SR_NAMES[80] = 'thread_position_in_grid.x'
//     and SR_NAMES[0] = 'threadgroup_position_in_grid.x' (the SR enum/table,
//     L24-62); MovFromSrInstructionDesc ('get_sr', size=4): opcode low7=0b1110010
//     (0x72), constant bit15=0, ALUDstDesc('D',28), SReg32Desc('SR', 16, 26) =
//     a 6+2-bit merged SR field (L2373-2392). get_sr ROUND-TRIPS through applegpu:
//     bytes 72 09 10 04 disassemble to `get_sr r2, sr80 (thread_position_in_grid.x)`.
//   - Mesa (src/asahi/compiler/agx_compile.c): nir_intrinsic_load_workgroup_id
//     lowers to AGX_SR_THREADGROUP_POSITION_IN_GRID_{X,Y,Z} and
//     load_local_invocation_id to AGX_SR_THREAD_POSITION_IN_THREADGROUP_*; the
//     load_global_invocation_id / thread_position_in_grid path maps to the same
//     thread-position SRs read by a `get_sr` (agx_emit_intrinsic ->
//     agx_get_sr/agx_get_sr_coverage). With local 1x1x1 the workgroup-id IS the
//     global thread index, matching our SR-80 read.
//
// THE INDEXED ADDRESSING. The per-thread element is reached with the INDEXED
// device_load / device_store form: the offset operand `O` is a REGISTER (the
// thread-index reg r2) instead of an immediate, so the hardware computes the
// address as `base + (O + i) * item_size` with item_size = 4 for i32 (applegpu
// DeviceLoadStoreInstructionDesc: `offset = get_reg32(O>>1); offset <<= s;
// load_address = address + (offset+i)*item_size`, L5006-5067). With `O = r2`
// (the index), `s = 0` (no extra shift) and i32 (item_size 4), the address is
// `base + idx*4` = the idx-th 32-bit element. THE INDEX REGISTER lives in the
// `O` field (MemoryIndexDesc merged (20,4)/(32,4)/(56,8), with `Ot` at bit 24
// selecting register(0) vs immediate(1), L1810-1840): r2 encodes as n<<1=4 at bit
// 20 -> byte 2 = 0x40, and `Ot=0` (register) makes byte 3 = 0x0e (vs the
// immediate form's 0x0d / byte2 0x00 that the single-element add kernel uses).
//   - device_load  0, i32, x, r0, u0_u1, r2, unsigned -> 05 01 40 0e 00 c0 12 00
//   - device_store 0, i32, x, r0, u2_u3, r2, unsigned -> 45 01 44 0e 00 c0 12 00
// The addend goes in r1 via mov_imm (the inline iadd immediate is 8-bit only).
//
// The kernel (per thread t):
//   get_sr       r2, sr80               ; r2 = thread_position_in_grid.x = t
//   device_load  0, i32, x, r0, u0_u1, r2, unsigned  ; r0 = input[u0_u1 + t*4]
//   mov_imm      r1, addend                           ; addend in a register
//   iadd         r0, r0, r1                           ; r0 = input[t] + addend
//   device_store 0, i32, x, r0, u2_u3, r2, unsigned  ; output[u2_u3 + t*4] = r0
//   wait         0                       ; ordering barrier
//   stop                                 ; end_execution
//
// The WHOLE 38-byte stream was ROUND-TRIPPED through dougallj's applegpu
// assembler+disassembler: it walks instruction-by-instruction to `stop` with no
// desync, disassembling to exactly the listing above (the get_sr and the two
// INDEXED device_load/device_store are the new encodings verified here). Buffer
// binding is the proven multi-buffer scheme: buffers[0] (input) -> u0_u1,
// buffers[1] (output) -> u2_u3 (see runCompute's MULTI-BUFFER BINDING note).
// ---------------------------------------------------------------------------

/// The raw little-endian AGX machine code for the parallel add kernel (38 bytes).
/// Each thread reads its own global index (get_sr sr80) and processes its own
/// element via indexed device_load/store. The addend at byte offset 14 (the
/// mov_imm r1 32-bit immediate) is patched by buildParallelAddKernel; the template
/// carries 0 there. Comment per instruction cites the applegpu encoding it was
/// round-tripped from.
pub const parallel_add_shader_code = [_]u8{
    // get_sr r2, sr80  (4 bytes) - r2 = thread_position_in_grid.x = global thread idx.
    //   opcode 0x72; D=r2 (32-bit reg, value n<<1=4, Dt=2); SR=80 (sr80).
    0x72, 0x09, 0x10, 0x04,
    // device_load 0, i32, x, r0, u0_u1, r2, unsigned  (8 bytes) - r0 = input[u0_u1 + r2*4].
    //   opcode 0x05 (load); F=i32(2), mask=x(1); A=u0_u1 (value 0); R=r0; O=r2
    //   (register index, n<<1=4 @byte2=0x40), Ot=0 (register), s=0 -> base+idx*4.
    0x05, 0x01, 0x40, 0x0e,
    0x00, 0xc0, 0x12, 0x00,
    // mov_imm r1, <addend>  (6 bytes) - opcode 0x62, dest-r1 selector 0x05,
    //   imm32 contiguous at bytes [14..18] little-endian (patched).
    0x62, 0x05, 0x00, 0x00,
    0x00, 0x00,
    // iadd r0, r0, r1  (8 bytes) - r0 = r0 + r1 (= input[idx] + addend).
    0x0e, 0x01,
    0x40, 0x22, 0x24, 0x00,
    0x00, 0x00,
    // device_store 0, i32, x, r0, u2_u3, r2, unsigned  (8 bytes) - output[u2_u3 + r2*4] = r0.
    //   opcode 0x45 (store); A=u2_u3 (merged-field value 4); O=r2 (index, byte2=0x44), Ot=0.
    0x45, 0x01,
    0x44, 0x0e, 0x00, 0xc0,
    0x12, 0x00,
    // wait 0   (2 bytes) - wait_for_loads, ordering barrier before stop.
    0x38, 0x00,
    // stop     (2 bytes) - end_execution.
    0x88, 0x00,
};

/// Byte offset of the mov_imm r1 32-bit immediate (the addend) in the parallel
/// kernel template.
const PARALLEL_ADDEND_OFF: usize = 14;

/// Size of the parallel add kernel buildParallelAddKernel writes (= 38).
pub const PARALLEL_ADD_KERNEL_SIZE: usize = parallel_add_shader_code.len;

/// Assemble the DATA-PARALLEL "input[i] -> output[i] + addend" AGX compute kernel
/// into `out`, returning the byte length (= PARALLEL_ADD_KERNEL_SIZE = 38). Each
/// hardware thread reads its OWN global index (get_sr sr80), loads input[idx] via
/// the INDEXED device_load, adds `addend`, and stores it to output[idx] via the
/// INDEXED device_store - so dispatching this over `groups = {N,1,1}` (local
/// 1x1x1 = N threads) processes N elements in parallel, each thread its own
/// element. The kernel reads buffer 0 (u0_u1) and writes buffer 1 (u2_u3). The
/// `addend` is baked into the mov_imm r1 32-bit immediate (bytes [14..18],
/// little-endian) of the round-tripped parallel_add_shader_code template. NO
/// runCompute / dispatchCompute / HAL change is needed (groups {N,1,1} already
/// launches N threads); only this kernel reads the thread index + addresses its
/// own element.
pub fn buildParallelAddKernel(out: []u8, addend: u32) usize {
    std.debug.assert(out.len >= parallel_add_shader_code.len);
    @memcpy(out[0..parallel_add_shader_code.len], &parallel_add_shader_code);
    // Patch the mov_imm r1 32-bit immediate (contiguous field at byte offset 14).
    std.mem.writeInt(u32, out[PARALLEL_ADDEND_OFF..][0..4], addend, .little);
    return parallel_add_shader_code.len;
}

// ---------------------------------------------------------------------------
// USC descriptor words.
//
// The USC (Uniform/Shader Cache) descriptor is a little byte stream of tagged
// words the hardware walks before launching the shader. For a minimal compute
// kernel (mirroring lib/agx_bg_eot.c's "Bake USC" block, minus textures), the
// words are, in order:
//   USC Uniform   - LOAD the 64-bit store-target VA into uniforms u0_u1 (half 0)
//                   FROM the uniform-data buffer that holds it (NOT from the
//                   store target itself - that is the bug this fixes).
//   USC Shared    - shared-memory config (vertex/compute layout, none used).
//   USC Shader    - the shader code address (USC-relative).
//   USC Registers - GPR count.
//   USC No Preshader - no preamble.
// Bit layouts: Mesa src/asahi/genxml/cmdbuf.xml. USC Control tags:
//   Uniform=0x1d, Shared=0x4d, Shader=0x0d, Registers=0x8d, No preshader=0x88.
// ---------------------------------------------------------------------------

/// USC Control tag bytes (cmdbuf.xml enum "USC Control").
const USC_TAG_SHADER: u8 = 0x0d;
const USC_TAG_UNIFORM: u8 = 0x1d;
const USC_TAG_SHARED: u8 = 0x4d;
const USC_TAG_REGISTERS: u8 = 0x8d;
const USC_TAG_NO_PRESHADER: u8 = 0x88;

/// AGX_SHARED_LAYOUT_VERTEX_COMPUTE (cmdbuf.xml enum "Shared layout").
const SHARED_LAYOUT_VERTEX_COMPUTE: u6 = 0x24;

/// Little-endian bit writer over a fixed scratch buffer - the genxml structs
/// are little-endian bit arrays, so we OR fields in at absolute bit offsets.
const BitWriter = struct {
    words: [16]u32 = [_]u32{0} ** 16,

    fn set(self: *BitWriter, start: u32, size: u32, value: u64) void {
        var bit = start;
        var rem = size;
        var v = value;
        while (rem > 0) {
            const word = bit / 32;
            const off: u5 = @intCast(bit % 32);
            const here = @min(rem, 32 - @as(u32, off));
            const mask: u64 = if (here >= 32) 0xFFFF_FFFF else ((@as(u64, 1) << @intCast(here)) - 1);
            self.words[word] |= @truncate((v & mask) << off);
            v >>= @intCast(here);
            bit += here;
            rem -= here;
        }
    }

    /// Emit `nbytes` of the packed words into `out`, returns bytes written.
    fn emit(self: *const BitWriter, out: []u8, nbytes: usize) usize {
        var i: usize = 0;
        while (i < nbytes) : (i += 1) {
            out[i] = @truncate(self.words[i / 4] >> @intCast((i % 4) * 8));
        }
        return nbytes;
    }
};

/// `__gen_to_groups(value, group_size, length)` from agx_pack_header.h:
/// stored = ceil(value/group_size); 0 clamps to 1; full (==2^length) -> 0.
fn toGroups(value: u32, group_size: u32, length: u5) u32 {
    if (value == 0) return 1;
    const groups = (value + group_size - 1) / group_size;
    if (groups == (@as(u32, 1) << length)) return 0;
    return groups;
}

/// Build the USC descriptor block for the constant-store kernel. `uniform_va`
/// is the GPU VA of the UNIFORM-DATA buffer that CONTAINS the 64-bit store
/// target (it is NOT the store target). The hardware loads 4 halves (8 bytes)
/// from `uniform_va` into uniform registers u0_u1, so the shader's
/// `device_store [u0_u1+0]` then writes to whatever 64-bit VA those bytes hold
/// (we fill them with OUTPUT_VA). `shader_off` is the shader's USC-relative
/// offset (SHADER_VA - USC_EXEC_BASE). Returns the number of bytes written.
///
/// THE BUG THIS FIXES: the USC Uniform Buffer field is the address the HW LOADS
/// the uniform registers FROM, not the address the shader stores TO. Passing
/// OUTPUT_VA here made the HW load u0_u1 from the output buffer's bytes (the
/// 0xDEADBEEF sentinel), then store to ~0xDEADBEEF (a GPU fault). Pass the
/// uniform-data buffer's VA (whose first 8 bytes = OUTPUT_VA, little-endian).
pub fn buildUsc(buf: []u8, uniform_va: u64, shader_off: u32, gpr_count: u32, uniform_halfs: u32) usize {
    var n: usize = 0;

    // USC Uniform (8 bytes): LOAD `uniform_halfs` uniform halves [0..uniform_halfs)
    // from `uniform_va`. The N buffers' VAs are stored consecutively there as N
    // 64-bit pointers (= N*4 halves), so buffer i lands in the uniform pair
    // u(2i)_u(2i+1).
    //   Tag(8)@0=Uniform; Start(halfs)(8)@8=0; Size(halfs)(6)@20 groups(1)=N*4;
    //   Buffer(38)@26 = the source address shr(2) = uniform_va>>2 (genxml encodes
    //   the load-FROM address shifted right by 2; the HW shifts it back).
    {
        var w = BitWriter{};
        w.set(0, 8, USC_TAG_UNIFORM);
        w.set(8, 8, 0); // start half = 0 -> buffers[0] lands in u0_u1
        w.set(20, 6, toGroups(uniform_halfs, 1, 6)); // N*4 halves = N 64-bit pointers
        w.set(26, 38, uniform_va >> 2); // LOAD-FROM addr (the uniform-data buffer)
        n += w.emit(buf[n..], 8);
    }

    // USC Shared (4 bytes): no shared memory used.
    //   Tag(8)@0=Shared; Uses shared memory(1)@8=0; Layout(6)@10=Vertex/compute;
    //   Bytes per threadgroup(8)@24 groups(256). agx_usc_shared_none sets
    //   bytes_per_threadgroup=65536 -> to_groups(65536,256,8) = 256 == 2^8 -> 0.
    {
        var w = BitWriter{};
        w.set(0, 8, USC_TAG_SHARED);
        w.set(10, 6, SHARED_LAYOUT_VERTEX_COMPUTE);
        w.set(24, 8, toGroups(65536, 256, 8));
        n += w.emit(buf[n..], 4);
    }

    // USC Shader (6 bytes): the shader code address (USC-relative).
    //   Tag(8)@0=Shader; Loads varyings(1)@8=0; Unk2(6)@10=3 (bg_eot sets 3);
    //   Code(32)@16 = shader_off.
    {
        var w = BitWriter{};
        w.set(0, 8, USC_TAG_SHADER);
        w.set(10, 6, 3); // cfg.unk_2 = 3 (lib/agx_bg_eot.c "Bake USC")
        w.set(16, 32, shader_off);
        n += w.emit(buf[n..], 6);
    }

    // USC Registers (4 bytes): GPR count.
    //   Tag(8)@0=Registers; Register count(5)@8 groups(8)=gpr_count; spill=0.
    {
        var w = BitWriter{};
        w.set(0, 8, USC_TAG_REGISTERS);
        w.set(8, 5, toGroups(gpr_count, 8, 5));
        n += w.emit(buf[n..], 4);
    }

    // USC No Preshader (2 bytes): no preamble program.
    {
        var w = BitWriter{};
        w.set(0, 8, USC_TAG_NO_PRESHADER);
        n += w.emit(buf[n..], 2);
    }

    return n;
}

// ---------------------------------------------------------------------------
// The CDM (compute) control stream.
//
// A direct (non-indirect) single-dispatch control stream, mirroring
// libagx_dgc.h agx_cdm_launch + agx_cdm_barrier + agx_cdm_terminate for a
// non-G14X chip (the M1 Pro is G13S, so NO CDM_UNK_G14X word):
//   CDM Launch Word 0 (4B) - reg counts + mode(Direct) + block(Launch)
//   CDM Launch Word 1 (4B) - pipeline = USC descriptor offset (>>6)
//   CDM Global size  (12B) - x,y,z = num_groups * local_size  (= 1,1,1)
//   CDM Local size   (12B) - x,y,z = workgroup dims           (= 1,1,1)
//   CDM Barrier      (4B)  - cache flush / invalidate after the launch
//   CDM Stream Terminate (8B) - end of the control stream
// Bit layouts: src/asahi/genxml/cmdbuf.xml. Block types: Launch=0,
// Stream Terminate=2, Barrier=3. CDM Mode Direct=0.
// ---------------------------------------------------------------------------

const CDM_BLOCK_LAUNCH: u3 = 0;
const CDM_BLOCK_TERMINATE: u3 = 2;
const CDM_BLOCK_BARRIER: u3 = 3;
const CDM_MODE_DIRECT: u2 = 0;

/// Build the CDM control stream for a single direct dispatch over `groups`
/// threadgroups (x,y,z). `usc_off` is the USC descriptor's offset from
/// USC_EXEC_BASE (so the pipeline field = usc_off >> 6). `uniform_halfs` is the
/// uniform half count the launch preloads (4 for our 64-bit pointer). The local
/// (workgroup) size is fixed at 1x1x1, so the GLOBAL size equals `groups` (global
/// = num_groups * local). Returns bytes written into `buf`.
pub fn buildControlStream(buf: []u8, usc_off: u32, uniform_halfs: u32, groups: [3]u32) usize {
    var n: usize = 0;
    // global = num_groups * local_size; local is 1x1x1, so global == groups. A
    // zero group dim is clamped to 1 (an empty dispatch would be a no-op / hang
    // risk; the proven path uses 1x1x1).
    const gx: u32 = if (groups[0] == 0) 1 else groups[0];
    const gy: u32 = if (groups[1] == 0) 1 else groups[1];
    const gz: u32 = if (groups[2] == 0) 1 else groups[2];

    // CDM Launch Word 0 (4 bytes).
    //   Uniform register count(3)@1 groups(64) = ceil(uniform_halfs/64);
    //   Texture state register count(5)@4 groups(8) = 0 textures -> 1;
    //   Sampler state register count(3)@9 = AGX_SAMPLER_STATES_0 = 0;
    //   Preshader register count(4)@12 groups(16) = 0 -> 1;
    //   Mode(2)@27 = Direct; Block Type(3)@29 = Launch.
    {
        var w = BitWriter{};
        w.set(1, 3, toGroups(uniform_halfs, 64, 3));
        w.set(4, 5, toGroups(0, 8, 5)); // 0 textures -> encodes 1
        w.set(9, 3, 0); // AGX_SAMPLER_STATES_0
        w.set(12, 4, toGroups(0, 16, 4)); // 0 preshader -> encodes 1
        w.set(27, 2, CDM_MODE_DIRECT);
        w.set(29, 3, CDM_BLOCK_LAUNCH);
        n += w.emit(buf[n..], 4);
    }

    // CDM Launch Word 1 (4 bytes): Pipeline(26)@6 shr(6) = usc_off >> 6.
    {
        var w = BitWriter{};
        w.set(6, 26, usc_off >> 6);
        n += w.emit(buf[n..], 4);
    }

    // CDM Global size (12 bytes): X,Y,Z each u32. global = groups * local; local
    // is 1x1x1, so global == groups.
    {
        var w = BitWriter{};
        w.set(0, 32, gx);
        w.set(32, 32, gy);
        w.set(64, 32, gz);
        n += w.emit(buf[n..], 12);
    }

    // CDM Local size (12 bytes): X,Y,Z workgroup dims = 1,1,1.
    {
        var w = BitWriter{};
        w.set(0, 32, 1);
        w.set(32, 32, 1);
        w.set(64, 32, 1);
        n += w.emit(buf[n..], 12);
    }

    // CDM Barrier (4 bytes): post-launch cache flush + USC cache invalidate.
    // Mirrors agx_cdm_barrier's "set these after every launch to be safe" block
    // for a non-G13X chip: unk_0..unk_20 + usc_cache_inval. Block Type=Barrier.
    {
        var w = BitWriter{};
        // unk_0..unk_2 (bits 0,1,2)
        w.set(0, 1, 1);
        w.set(1, 1, 1);
        w.set(2, 1, 1);
        // USC cache inval (bit 3)
        w.set(3, 1, 1);
        // unk_4..unk_20 (bits 4..20 inclusive)
        var b: u32 = 4;
        while (b <= 20) : (b += 1) w.set(b, 1, 1);
        w.set(29, 3, CDM_BLOCK_BARRIER);
        n += w.emit(buf[n..], 4);
    }

    // CDM Stream Terminate (8 bytes): Block Type(3)@29 = Stream Terminate.
    {
        var w = BitWriter{};
        w.set(29, 3, CDM_BLOCK_TERMINATE);
        n += w.emit(buf[n..], 8);
    }

    return n;
}

/// Assemble the full SUBMIT cmdbuf: a drm_asahi_cmd_header{COMPUTE} followed by
/// the drm_asahi_cmd_compute body. The control-stream base/end point at the BO
/// holding the bytes from buildControlStream. Returns bytes written into `buf`.
///
/// The header barriers are 0/0: for the FIRST (and only) CDM command in the
/// submit, Mesa passes vdm_barrier=nr_vdm=0, cdm_barrier=nr_cdm=0 (hk_queue.c
/// the submit loop). `ctrl_len` is the control-stream byte length (so end =
/// base + ctrl_len, spanning the launch words through the terminate).
pub fn buildCmdbuf(buf: []u8, ctrl_base: u64, ctrl_len: usize) usize {
    const header = uapi.drm_asahi_cmd_header{
        .cmd_type = uapi.CMD_COMPUTE,
        .size = @sizeOf(uapi.drm_asahi_cmd_compute),
        .vdm_barrier = 0, // first command -> barrier index 0
        .cdm_barrier = 0,
    };
    const body = uapi.drm_asahi_cmd_compute{
        .flags = 0,
        .sampler_count = 0,
        .cdm_ctrl_stream_base = ctrl_base,
        .cdm_ctrl_stream_end = ctrl_base + ctrl_len,
        .sampler_heap = 0,
        .helper = .{ .binary = 0, .cfg = 0, .data = 0 },
        .ts = .{ .start = .{ .handle = 0, .offset = 0 }, .end = .{ .handle = 0, .offset = 0 } },
    };

    const hbytes = std.mem.asBytes(&header);
    const bbytes = std.mem.asBytes(&body);
    @memcpy(buf[0..hbytes.len], hbytes);
    @memcpy(buf[hbytes.len..][0..bbytes.len], bbytes);
    return hbytes.len + bbytes.len;
}

// ---------------------------------------------------------------------------
// CPU<->GPU coherency.
//
// RESEARCH (what Mesa / the kernel actually do). The AGX GPU is IO-COHERENT
// with the CPU: the Asahi reverse-engineering docs state "all memory is
// coherent as far as we can tell - we haven't used a single cache management
// instruction and everything still works", and Mesa's asahi driver performs NO
// CPU cache maintenance around GPU work. The mainlined drm/asahi UAPI exposes
// exactly TWO gem_create flags, WRITEBACK and VM_PRIVATE; WRITEBACK only picks
// a CPU-cacheable mapping "optimized for CPU reads" vs the default write-combine
// (kernel comment: "Map as writeback instead of write-combine. This optimizes
// for CPU reads."). There is NO separate "coherent"/"uncached" gem flag to
// switch to - WRITEBACK is already the coherent, CPU-read-friendly choice, so we
// keep allocBo(GEM_WRITEBACK) for the output BO.
//
// Because the hardware is coherent, a sentinel that SURVIVES a successful submit
// means the GPU genuinely did not store (an encoding bug), NOT a stale read. To
// make that conclusion airtight we still bracket the readback with explicit
// aarch64 CPU cache maintenance + a barrier: a clean-to-PoC before the submit so
// RAM holds the sentinel, and a clean+invalidate (the EL0-allowed `dc civac`)
// after the fence so the readback cannot come from a stale CPU line. If the
// result is then still the sentinel, coherency is definitively ruled out.
//
// Zig 0.16 inline-asm: clobbers are an anonymous struct (`.{ .memory = true }`),
// not the old string form.
// ---------------------------------------------------------------------------

/// aarch64 cache line size we stride by. 64 bytes is the architectural minimum
/// DminLine on Apple Silicon; striding by 64 safely covers every line touched.
const CACHE_LINE: usize = 64;

/// Full system data-synchronization barrier (`dsb sy`): orders the cache ops
/// against the surrounding memory accesses / the GPU's coherent view.
inline fn dsbSy() void {
    asm volatile ("dsb sy" ::: .{ .memory = true });
}

/// Clean (write-back) the CPU-dirty lines of `mem` to the point of coherency
/// (`dc cvac`) so RAM holds what the CPU just wrote (the sentinel) before the
/// GPU reads/writes the BO. EL0-legal on Linux (SCTLR_EL1.UCI enables it).
fn cleanToPoC(mem: []const u8) void {
    var off: usize = 0;
    while (off < mem.len) : (off += CACHE_LINE) {
        const addr = @intFromPtr(mem.ptr) + off;
        asm volatile ("dc cvac, %[a]"
            :
            : [a] "r" (addr),
            : .{ .memory = true });
    }
    dsbSy();
}

/// Clean + invalidate (`dc civac`, the EL0-allowed op) the lines of `mem` to the
/// point of coherency so a subsequent CPU read fetches fresh data rather than a
/// stale cached line. Used AFTER the fence, BEFORE readback.
fn cleanInvalidateToPoC(mem: []const u8) void {
    var off: usize = 0;
    while (off < mem.len) : (off += CACHE_LINE) {
        const addr = @intFromPtr(mem.ptr) + off;
        asm volatile ("dc civac, %[a]"
            :
            : [a] "r" (addr),
            : .{ .memory = true });
    }
    dsbSy();
}

// ---------------------------------------------------------------------------
// The live launch.
// ---------------------------------------------------------------------------

/// Max storage buffers a single dispatch binds. Each buffer is a 64-bit pointer
/// = 4 uniform halves; 8 buffers = 32 halves, comfortably inside a single 16-KiB
/// uniform-data page and the launch's uniform-register budget.
pub const MAX_BUFFERS: usize = 8;

/// Generalized one-shot AGX compute launch (the HAL `dispatchCompute` backend).
/// Runs a CALLER-supplied compute `kernel_code` on a real AGX GPU binding N
/// CALLER-owned storage buffers (`storage_vas`, their GPU VAs) IN ORDER, then
/// dispatching over `groups` (x,y,z) threadgroups.
///
/// MULTI-BUFFER BINDING. The kernel addresses buffer i as the uniform pair
/// u(2i)_u(2i+1): the UNIFORM_DATA BO holds the N VAs as consecutive 8-byte
/// little-endian pointers (storage_vas[0] at +0, storage_vas[1] at +8, ...,
/// storage_vas[i] at +8*i), and the USC Uniform descriptor loads N*4 halves from
/// it into u0..u(2N-1). So buffer 0 -> u0_u1, buffer 1 -> u2_u3, buffer i ->
/// u(2i)_u(2i+1). A `device_load [u0_u1+0]` reads the first buffer, a
/// `device_store [u2_u3+0]` writes the second, etc. (the single-buffer constant
/// kernel passes one VA and addresses u0_u1, UNREGRESSED).
///
/// This is EXACTLY the PROVEN runComputeConstant path, parameterized:
///   (a) the caller's `kernel_code` is uploaded to the kernel BO;
///   (b) the caller's `storage_vas` are bound via the SAME USC Uniform
///       indirection - the UNIFORM_DATA BO holds them as N LE pointers, the USC
///       Uniform Buffer points at that BO, so the HW loads u(2i)_u(2i+1) =
///       storage_vas[i] before the shader runs (the proven mechanism; the
///       USC-Uniform-source bug fix);
///   (c) the caller's `groups` go into the CDM Launch global/local dims.
///
/// The storage buffers are CALLER-OWNED: this does NOT allocBo at any storage_va
/// and MUST NOT touch their lifetime (the P5b lesson - re-allocating a
/// caller-owned BO would double-bind/leak it). Only the FOUR launch-infra BOs
/// (shader / USC / control stream / uniform-data) are allocated here, and they
/// are FREED via dev.freeBo AFTER the fence (the P5d lesson) so dispatchCompute is
/// REPEATABLE - a second call re-binds the same fixed infra VAs without an EINVAL
/// "already bound" collision.
///
/// The caller passes an EXISTING `vm_id` and `queue_id`; the queue MUST have been
/// created with usc_exec_base = USC_EXEC_BASE so the USC-relative addresses
/// resolve. Every storage_va MUST already be mapped in `vm_id` (the caller's
/// resource BOs are bound there). Cleans up the out-syncobj on the way out.
///
/// NOTE: this performs a REAL GPU submit - it is the user's M1 test and cannot be
/// exercised on a box without an AGX GPU (allocBo / submit fail, surfaced as an
/// Error, not a crash).
pub fn runCompute(
    dev: *Device,
    vm_id: u32,
    queue_id: u32,
    kernel_code: []const u8,
    storage_vas: []const u64,
    groups: [3]u32,
) Error!void {
    std.debug.assert(kernel_code.len <= PAGE);
    // 1..MAX_BUFFERS storage buffers. The HAL boundary (apple dispatchCompute)
    // already rejects an out-of-range count with error.InvalidArgument; this
    // internal path asserts it (the asahi Error set is transport-only). N*8 bytes
    // of VAs must also fit the single uniform-data page.
    std.debug.assert(storage_vas.len >= 1 and storage_vas.len <= MAX_BUFFERS);
    std.debug.assert(storage_vas.len * 8 <= PAGE);

    // Allocate the FOUR launch-infra BOs at their fixed GPU VAs. The storage BOs
    // are CALLER-owned and already bound at their VAs - we do NOT allocate them.
    var shader_bo = try dev.allocBo(vm_id, PAGE, SHADER_VA);
    var usc_bo = try dev.allocBo(vm_id, PAGE, USC_VA);
    var ctrl_bo = try dev.allocBo(vm_id, PAGE, CTRL_STREAM_VA);
    var uniform_bo = try dev.allocBo(vm_id, PAGE, UNIFORM_DATA_VA);
    // Free the infra AFTER the fence so a repeat dispatch can re-bind these VAs.
    // (Deferred, runs on every exit path including an error after a partial
    // setup, since all four are bound by this point.)
    defer dev.freeBo(vm_id, shader_bo);
    defer dev.freeBo(vm_id, usc_bo);
    defer dev.freeBo(vm_id, ctrl_bo);
    defer dev.freeBo(vm_id, uniform_bo);

    // Fill the uniform-data buffer with the N 8-byte little-endian storage VAs,
    // consecutively (vas[0] at +0, vas[1] at +8, ...). The USC Uniform descriptor
    // points the hardware HERE, so the HW loads u(2i)_u(2i+1) = storage_vas[i]
    // from these bytes before the shader runs. This buffer HOLDS the pointers; it
    // is NOT any store target. (The P3 bug was pointing the USC Uniform at the
    // output BO, so the HW loaded u0_u1 from its bytes.)
    for (storage_vas, 0..) |va, i| {
        std.mem.writeInt(u64, uniform_bo.cpu[i * 8 ..][0..8], va, .little);
    }
    const uniform_halfs: u32 = @intCast(storage_vas.len * 4); // 4 halves per 64-bit ptr

    // Upload the caller's kernel binary.
    @memcpy(shader_bo.cpu[0..kernel_code.len], kernel_code);

    // Build + upload the USC descriptor. The USC Uniform Buffer points at the
    // UNIFORM-DATA buffer (which holds the N VAs), NOT at any storage buffer - the
    // HW LOADS u0..u(2N-1) from there. The shader's USC-relative offset is
    // SHADER_VA - USC_EXEC_BASE; 8 GPRs (covers the kernels' working registers).
    const shader_off: u32 = @intCast(SHADER_VA - USC_EXEC_BASE);
    _ = buildUsc(usc_bo.cpu[0..], UNIFORM_DATA_VA, shader_off, 8, uniform_halfs);

    // Build + upload the CDM control stream with the caller's grid dims. The
    // pipeline offset is the USC descriptor's offset from USC_EXEC_BASE; the
    // launch preloads `uniform_halfs` uniform halves.
    const usc_off: u32 = @intCast(USC_VA - USC_EXEC_BASE);
    const ctrl_len = buildControlStream(ctrl_bo.cpu[0..], usc_off, uniform_halfs, groups);

    // Assemble the SUBMIT cmdbuf (header + compute body).
    var cmdbuf: [@sizeOf(uapi.drm_asahi_cmd_header) + @sizeOf(uapi.drm_asahi_cmd_compute)]u8 = undefined;
    const cmd_len = buildCmdbuf(cmdbuf[0..], CTRL_STREAM_VA, ctrl_len);

    // An out-syncobj the kernel signals on completion.
    const out_sync = try dev.syncobjCreate(false);
    defer dev.syncobjDestroy(out_sync);

    // Clean the uniform-data buffer so the GPU's uniform LOAD reads the N VAs we
    // just wrote from RAM, not a stale line (AGX is coherent, so this is
    // belt-and-suspenders).
    cleanToPoC(uniform_bo.cpu[0 .. storage_vas.len * 8]);

    const out_syncs = [_]uapi.drm_asahi_sync{Device.binarySync(out_sync)};
    try dev.submit(queue_id, cmdbuf[0..cmd_len], &.{}, &out_syncs);

    // Wait up to 1 second for the GPU to finish.
    const handles = [_]u32{out_sync};
    try dev.syncobjWait(&handles, 1_000_000_000);
}

/// Run the constant-store compute kernel on a real AGX GPU and return the u32 the
/// GPU wrote to output[0..4]. This is the P3 self-test, now a thin wrapper over
/// the generalized `runCompute`: it allocates its OWN 4 KiB output BO (the
/// caller-owned storage), sentinel-prefills it, dispatches the hand-assembled
/// constant kernel over a 1x1x1 grid, then reads output[0..4] back. UNREGRESSED:
/// a successful run still returns kConstant (0xCAFEF00D).
///
/// The caller passes an EXISTING `vm_id` and `queue_id` - the queue MUST have
/// been created with usc_exec_base = USC_EXEC_BASE so the USC-relative addresses
/// resolve. The output BO is freed on the way out (so this is repeatable too).
pub fn runComputeConstant(dev: *Device, vm_id: u32, queue_id: u32) Error!u32 {
    // The constant-store kernel's OWN output BO (the "caller-owned storage" from
    // runCompute's point of view).
    var out_bo = try dev.allocBo(vm_id, PAGE, OUTPUT_VA);
    defer dev.freeBo(vm_id, out_bo);

    // Pre-fill the output BO with the SENTINEL (not zero) so a non-write is
    // distinguishable from a write-of-zero. The whole BO is filled; the readback
    // only inspects output[0..4], which the shader targets.
    {
        var off: usize = 0;
        while (off + 4 <= out_bo.cpu.len) : (off += 4) {
            std.mem.writeInt(u32, out_bo.cpu[off..][0..4], kSentinel, .little);
        }
    }
    // Coherency bracket #1: clean the sentinel out to RAM BEFORE the submit so a
    // post-fence re-read sees the GPU's store, not a stale CPU line.
    cleanToPoC(out_bo.cpu[0..]);

    // Build the hand-assembled constant kernel and dispatch it over a 1x1x1 grid
    // binding a SINGLE storage buffer (u0_u1 = OUTPUT_VA) - the proven P3 path,
    // now expressed as a one-element storage_vas slice.
    var kernel: [CONSTANT_KERNEL_SIZE]u8 = undefined;
    const klen = buildConstantKernel(&kernel, kConstant);
    const storage_vas = [_]u64{OUTPUT_VA};
    try runCompute(dev, vm_id, queue_id, kernel[0..klen], &storage_vas, .{ 1, 1, 1 });

    // Coherency bracket #2: clean+invalidate AFTER the fence, BEFORE readback, so
    // the read cannot return a stale cached line. With AGX coherent this is a
    // no-op for correctness, but it means a surviving sentinel definitively
    // implicates the shader/dispatch encoding, not the CPU cache.
    cleanInvalidateToPoC(out_bo.cpu[0..]);

    // Read back the word the GPU stored. Decoded by the caller against kConstant
    // (success) / kSentinel (store never landed) / other (wrong value).
    return std.mem.readInt(u32, out_bo.cpu[0..4], .little);
}

// ---------------------------------------------------------------------------
// Structural tests (no GPU on this box - these check the encoders' shapes).
// ---------------------------------------------------------------------------

test "shader binary is the expected 18 bytes and starts with mov_imm" {
    try std.testing.expectEqual(@as(usize, 18), shader_code.len);
    // mov_imm opcode byte.
    try std.testing.expectEqual(@as(u8, 0x62), shader_code[0]);
    // The 0xCAFEF00D immediate is little-endian at byte offset 2.
    const imm = std.mem.readInt(u32, shader_code[2..6], .little);
    try std.testing.expectEqual(kConstant, imm);
    // device_store opcode, then wait, then stop.
    try std.testing.expectEqual(@as(u8, 0x45), shader_code[6]);
    try std.testing.expectEqual(@as(u8, 0x38), shader_code[14]);
    try std.testing.expectEqual(@as(u8, 0x88), shader_code[16]);
}

test "toGroups matches __gen_to_groups semantics" {
    // 0 clamps to 1.
    try std.testing.expectEqual(@as(u32, 1), toGroups(0, 64, 3));
    // 4 halves over group 64 -> ceil = 1.
    try std.testing.expectEqual(@as(u32, 1), toGroups(4, 64, 3));
    // full (==2^length) encodes as 0: 65536/256 = 256 == 2^8.
    try std.testing.expectEqual(@as(u32, 0), toGroups(65536, 256, 8));
    // 8 over group 8 -> 1 (same encoding as 0 textures).
    try std.testing.expectEqual(@as(u32, 1), toGroups(8, 8, 5));
}

test "USC block has the expected size and tag order" {
    var buf: [64]u8 = undefined;
    const n = buildUsc(&buf, UNIFORM_DATA_VA, 0x1000, 8, 4);
    // Uniform(8) + Shared(4) + Shader(6) + Registers(4) + NoPreshader(2) = 24.
    try std.testing.expectEqual(@as(usize, 24), n);
    // Tag bytes appear at the start of each word.
    try std.testing.expectEqual(USC_TAG_UNIFORM, buf[0]);
    try std.testing.expectEqual(USC_TAG_SHARED, buf[8]);
    try std.testing.expectEqual(USC_TAG_SHADER, buf[12]);
    try std.testing.expectEqual(USC_TAG_REGISTERS, buf[18]);
    try std.testing.expectEqual(USC_TAG_NO_PRESHADER, buf[22]);
}

test "USC Uniform word loads u0_u1 FROM the uniform-data buffer (not the store target)" {
    var buf: [64]u8 = undefined;
    _ = buildUsc(&buf, UNIFORM_DATA_VA, 0x1000, 8, 4);
    const word = std.mem.readInt(u64, buf[0..8], .little);
    // Tag (bits 0..8).
    try std.testing.expectEqual(@as(u64, USC_TAG_UNIFORM), word & 0xFF);
    // Start halfs (bits 8..16) = 0.
    try std.testing.expectEqual(@as(u64, 0), (word >> 8) & 0xFF);
    // Size halfs (bits 20..26) groups(1) = 4 (a single 64-bit pointer).
    try std.testing.expectEqual(@as(u64, 4), (word >> 20) & 0x3F);
    // Buffer (bits 26..64) = the LOAD-FROM address = UNIFORM_DATA_VA >> 2. This
    // is the uniform-data buffer that HOLDS OUTPUT_VA, NOT OUTPUT_VA itself - the
    // regression guard for the dmesg GPU-fault bug.
    try std.testing.expectEqual(UNIFORM_DATA_VA >> 2, word >> 26);
    try std.testing.expect((word >> 26) != (OUTPUT_VA >> 2));
}

test "USC Uniform Size scales with the buffer count (N*4 halves)" {
    // Two buffers (input+output) = 8 halves; the USC Uniform binds N*4 halves so
    // the HW loads N 64-bit pointers into u0..u(2N-1).
    var buf: [64]u8 = undefined;
    _ = buildUsc(&buf, UNIFORM_DATA_VA, 0x1000, 8, 8);
    const word = std.mem.readInt(u64, buf[0..8], .little);
    try std.testing.expectEqual(@as(u64, USC_TAG_UNIFORM), word & 0xFF);
    // Size halfs groups(1) = 8 for 2 buffers.
    try std.testing.expectEqual(@as(u64, 8), (word >> 20) & 0x3F);
    // The LOAD-FROM address is unchanged (still the uniform-data buffer).
    try std.testing.expectEqual(UNIFORM_DATA_VA >> 2, word >> 26);
}

test "uniform-data buffer VAs are distinct, PAGE-aligned, and ordered" {
    // The uniform-data buffer must be its own 16 KiB page, distinct from the
    // output BO, so the USC Uniform LOAD source and the store TARGET never alias.
    try std.testing.expect(UNIFORM_DATA_VA != OUTPUT_VA);
    try std.testing.expectEqual(@as(u64, 0), UNIFORM_DATA_VA % PAGE);
    try std.testing.expectEqual(@as(u64, 0), OUTPUT_VA % PAGE);
    // It lives in the USC window just above the control stream.
    try std.testing.expectEqual(USC_EXEC_BASE + 4 * PAGE, UNIFORM_DATA_VA);
}

test "control stream has the expected size and launch/terminate shape" {
    var buf: [64]u8 = undefined;
    const n = buildControlStream(&buf, 0x2000, 4, .{ 1, 1, 1 });
    // Launch0(4)+Launch1(4)+Global(12)+Local(12)+Barrier(4)+Terminate(8) = 44.
    try std.testing.expectEqual(@as(usize, 44), n);

    // Launch Word 0: Block Type (bits 29..32) = Launch (0), Mode (27..29) = 0.
    const w0 = std.mem.readInt(u32, buf[0..4], .little);
    try std.testing.expectEqual(@as(u32, CDM_BLOCK_LAUNCH), w0 >> 29);
    // Uniform register count (bits 1..4) = 1 (4 halves / group 64).
    try std.testing.expectEqual(@as(u32, 1), (w0 >> 1) & 0x7);

    // Launch Word 1: Pipeline (bits 6..32) = usc_off >> 6.
    const w1 = std.mem.readInt(u32, buf[4..8], .little);
    try std.testing.expectEqual(@as(u32, 0x2000 >> 6), w1 >> 6);

    // Global + Local sizes are all 1.
    try std.testing.expectEqual(@as(u32, 1), std.mem.readInt(u32, buf[8..12], .little));
    try std.testing.expectEqual(@as(u32, 1), std.mem.readInt(u32, buf[20..24], .little));

    // Terminate word (last 8 bytes): Block Type = Stream Terminate (2).
    const term = std.mem.readInt(u32, buf[36..40], .little);
    try std.testing.expectEqual(@as(u32, CDM_BLOCK_TERMINATE), term >> 29);
}

test "control stream carries the caller's group dims in the global size" {
    var buf: [64]u8 = undefined;
    _ = buildControlStream(&buf, 0x2000, 4, .{ 4, 2, 3 });
    // Global size X,Y,Z (the 12 bytes after the two launch words) == groups.
    try std.testing.expectEqual(@as(u32, 4), std.mem.readInt(u32, buf[8..12], .little));
    try std.testing.expectEqual(@as(u32, 2), std.mem.readInt(u32, buf[12..16], .little));
    try std.testing.expectEqual(@as(u32, 3), std.mem.readInt(u32, buf[16..20], .little));
    // Local size stays 1x1x1.
    try std.testing.expectEqual(@as(u32, 1), std.mem.readInt(u32, buf[20..24], .little));
    // A zero group dim clamps to 1 (no empty/no-op dispatch).
    _ = buildControlStream(&buf, 0x2000, 4, .{ 0, 0, 0 });
    try std.testing.expectEqual(@as(u32, 1), std.mem.readInt(u32, buf[8..12], .little));
}

test "buildConstantKernel matches the proven shader for the default constant" {
    var buf: [CONSTANT_KERNEL_SIZE]u8 = undefined;
    const n = buildConstantKernel(&buf, kConstant);
    // Same length + bytes as the hand-assembled shader_code (the P3 kernel).
    try std.testing.expectEqual(shader_code.len, n);
    try std.testing.expectEqualSlices(u8, &shader_code, buf[0..n]);
}

test "buildConstantKernel bakes an arbitrary constant into the mov_imm immediate" {
    var buf: [CONSTANT_KERNEL_SIZE]u8 = undefined;
    _ = buildConstantKernel(&buf, 0x12345678);
    // mov_imm opcode + the patched 32-bit immediate at byte offset 2.
    try std.testing.expectEqual(@as(u8, 0x62), buf[0]);
    try std.testing.expectEqual(@as(u32, 0x12345678), std.mem.readInt(u32, buf[2..6], .little));
    // The device_store / wait / stop tail is untouched.
    try std.testing.expectEqual(@as(u8, 0x45), buf[6]);
    try std.testing.expectEqual(@as(u8, 0x38), buf[14]);
    try std.testing.expectEqual(@as(u8, 0x88), buf[16]);
}

test "buildAddKernel byte-shape: the proven input->output add stream" {
    var buf: [ADD_KERNEL_SIZE]u8 = undefined;
    const n = buildAddKernel(&buf, 0x0000CAFE);
    // 34 bytes: mov_imm(6)+device_load(8)+iadd(8)+device_store(8)+wait(2)+stop(2).
    try std.testing.expectEqual(@as(usize, 34), n);
    // mov_imm r1, addend  - opcode 0x62, dest-r1 selector 0x05, imm @[2..6].
    try std.testing.expectEqual(@as(u8, 0x62), buf[0]);
    try std.testing.expectEqual(@as(u8, 0x05), buf[1]);
    try std.testing.expectEqual(@as(u32, 0x0000CAFE), std.mem.readInt(u32, buf[2..6], .little));
    // device_load r0, [u0_u1+0] (input = buffer 0): opcode 0x05, base value 0.
    try std.testing.expectEqual(@as(u8, 0x05), buf[6]);
    try std.testing.expectEqual(@as(u8, 0x00), buf[8]); // A-field low nibble = 0 (u0_u1)
    // iadd r0, r0, r1: opcode 0x0e.
    try std.testing.expectEqual(@as(u8, 0x0e), buf[14]);
    // device_store r0, [u2_u3+0] (output = buffer 1): opcode 0x45, base value 4.
    try std.testing.expectEqual(@as(u8, 0x45), buf[22]);
    try std.testing.expectEqual(@as(u8, 0x04), buf[24]); // A-field low nibble = 4 (u2_u3)
    // wait 0, then stop.
    try std.testing.expectEqual(@as(u8, 0x38), buf[30]);
    try std.testing.expectEqual(@as(u8, 0x88), buf[32]);
}

test "buildAddKernel bakes an arbitrary addend, leaving the load/iadd/store tail intact" {
    var a: [ADD_KERNEL_SIZE]u8 = undefined;
    var b: [ADD_KERNEL_SIZE]u8 = undefined;
    _ = buildAddKernel(&a, 0x0000CAFE);
    _ = buildAddKernel(&b, 0x12345678);
    // Only the mov_imm immediate (bytes [2..6]) differs between addends.
    try std.testing.expectEqual(@as(u32, 0x12345678), std.mem.readInt(u32, b[2..6], .little));
    try std.testing.expectEqualSlices(u8, a[0..2], b[0..2]);
    try std.testing.expectEqualSlices(u8, a[6..], b[6..]);
    // The template (default addend 0) is add_shader_code.
    var t: [ADD_KERNEL_SIZE]u8 = undefined;
    _ = buildAddKernel(&t, 0);
    try std.testing.expectEqualSlices(u8, &add_shader_code, &t);
}

test "buildParallelAddKernel byte-shape: per-thread get_sr + indexed load/store stream" {
    var buf: [PARALLEL_ADD_KERNEL_SIZE]u8 = undefined;
    const n = buildParallelAddKernel(&buf, 0x0000CAFE);
    // 38 bytes: get_sr(4)+device_load(8)+mov_imm(6)+iadd(8)+device_store(8)+wait(2)+stop(2).
    try std.testing.expectEqual(@as(usize, 38), n);

    // get_sr r2, sr80 - the THREAD-INDEX read. opcode 0x72, SR number 80 (=
    // thread_position_in_grid.x, the global thread index). The SR field is the
    // merged SReg32Desc (6 bits @16 + 2 bits @26); for SR=80 the low 6 bits (80 &
    // 0x3F = 16 = 0x10) land at byte 2, the high 2 bits (80 >> 6 = 1) at bit 26.
    // The whole get_sr round-trips through applegpu to `get_sr r2, sr80`.
    try std.testing.expectEqual(@as(u8, 0x72), buf[0]); // get_sr opcode
    const gsr = std.mem.readInt(u32, buf[0..4], .little);
    // SR = (bits 16..22) | (bits 26..28 << 6).
    const sr = ((gsr >> 16) & 0x3F) | (((gsr >> 26) & 0x3) << 6);
    try std.testing.expectEqual(@as(u32, 80), sr); // thread_position_in_grid.x

    // device_load r0, [u0_u1 + r2*4] (input = buffer 0), INDEXED by r2 - the load
    // occupies buf[4..12] = 05 01 40 0e 00 c0 12 00. opcode 0x05; the index reg r2
    // is in the O field at the load's byte 2 (buf[6]=0x40 = r2 encoded n<<1=4 at
    // bit 20) with Ot=0 (register) -> the load's byte 3 (buf[7]=0x0e) vs the
    // immediate-offset form's 0x0d / byte2 0x00 that the single-element add kernel
    // uses. The A-field (u0_u1=0) and the i32/mask bits sit in the remaining bytes.
    try std.testing.expectEqual(@as(u8, 0x05), buf[4]); // device_load opcode
    try std.testing.expectEqual(@as(u8, 0x40), buf[6]); // O field byte = r2 index
    try std.testing.expectEqual(@as(u8, 0x0e), buf[7]); // Ot=0 register (not 0x0d immediate)
    try std.testing.expectEqualSlices(u8, &[_]u8{ 0x05, 0x01, 0x40, 0x0e, 0x00, 0xc0, 0x12, 0x00 }, buf[4..12]);

    // mov_imm r1, addend - opcode 0x62, dest-r1 selector 0x05, imm @[14..18].
    try std.testing.expectEqual(@as(u8, 0x62), buf[12]);
    try std.testing.expectEqual(@as(u8, 0x05), buf[13]);
    try std.testing.expectEqual(@as(u32, 0x0000CAFE), std.mem.readInt(u32, buf[14..18], .little));

    // iadd r0, r0, r1: opcode 0x0e.
    try std.testing.expectEqual(@as(u8, 0x0e), buf[18]);

    // device_store r0, [u2_u3 + r2*4] (output = buffer 1), INDEXED by r2 - the
    // store occupies buf[26..34] = 45 01 44 0e 00 c0 12 00. opcode 0x45; the store's
    // byte 2 (buf[28]=0x44) carries the A-field low nibble (u2_u3 = 4) merged with
    // the r2 index O bits, and Ot=0 (register) -> byte 3 (buf[29]=0x0e).
    try std.testing.expectEqual(@as(u8, 0x45), buf[26]); // device_store opcode
    try std.testing.expectEqual(@as(u8, 0x44), buf[28]); // A (u2_u3) low nibble | O index (r2)
    try std.testing.expectEqual(@as(u8, 0x0e), buf[29]); // Ot=0 register
    try std.testing.expectEqualSlices(u8, &[_]u8{ 0x45, 0x01, 0x44, 0x0e, 0x00, 0xc0, 0x12, 0x00 }, buf[26..34]);

    // wait 0, then stop.
    try std.testing.expectEqual(@as(u8, 0x38), buf[34]);
    try std.testing.expectEqual(@as(u8, 0x88), buf[36]);
}

test "buildParallelAddKernel bakes an arbitrary addend, leaving the get_sr/load/iadd/store intact" {
    var a: [PARALLEL_ADD_KERNEL_SIZE]u8 = undefined;
    var b: [PARALLEL_ADD_KERNEL_SIZE]u8 = undefined;
    _ = buildParallelAddKernel(&a, 0x0000CAFE);
    _ = buildParallelAddKernel(&b, 0x12345678);
    // Only the mov_imm immediate (bytes [14..18]) differs between addends.
    try std.testing.expectEqual(@as(u32, 0x12345678), std.mem.readInt(u32, b[14..18], .little));
    try std.testing.expectEqualSlices(u8, a[0..14], b[0..14]);
    try std.testing.expectEqualSlices(u8, a[18..], b[18..]);
    // The template (default addend 0) is parallel_add_shader_code.
    var t: [PARALLEL_ADD_KERNEL_SIZE]u8 = undefined;
    _ = buildParallelAddKernel(&t, 0);
    try std.testing.expectEqualSlices(u8, &parallel_add_shader_code, &t);
    // The parallel kernel is DISTINCT from the single-element add kernel (it has
    // the extra get_sr + uses indexed addressing) and is longer.
    try std.testing.expect(PARALLEL_ADD_KERNEL_SIZE > ADD_KERNEL_SIZE);
}

test "multi-VA UNIFORM_DATA layout: N pointers consecutive little-endian" {
    // runCompute writes storage_vas[i] at uniform_data[+8*i] as a LE u64 - the
    // layout the USC Uniform load (N*4 halves) consumes so buffer i lands in
    // u(2i)_u(2i+1). Reproduce that fill and assert the byte layout.
    const vas = [_]u64{ 0x2_0000_0000, 0x2_0000_4000, 0x2_0000_8000 };
    var udata: [64]u8 = [_]u8{0} ** 64;
    for (vas, 0..) |va, i| {
        std.mem.writeInt(u64, udata[i * 8 ..][0..8], va, .little);
    }
    // buffer 0 at +0 -> u0_u1, buffer 1 at +8 -> u2_u3, buffer 2 at +16 -> u4_u5.
    try std.testing.expectEqual(vas[0], std.mem.readInt(u64, udata[0..8], .little));
    try std.testing.expectEqual(vas[1], std.mem.readInt(u64, udata[8..16], .little));
    try std.testing.expectEqual(vas[2], std.mem.readInt(u64, udata[16..24], .little));
    // The USC Uniform binds N*4 = 12 halves for these 3 buffers.
    var ubuf: [64]u8 = undefined;
    _ = buildUsc(&ubuf, UNIFORM_DATA_VA, 0x1000, 8, @intCast(vas.len * 4));
    const word = std.mem.readInt(u64, ubuf[0..8], .little);
    try std.testing.expectEqual(@as(u64, 12), (word >> 20) & 0x3F);
}

test "cmdbuf is header + compute body with correct type and size fields" {
    var buf: [@sizeOf(uapi.drm_asahi_cmd_header) + @sizeOf(uapi.drm_asahi_cmd_compute)]u8 = undefined;
    const n = buildCmdbuf(&buf, CTRL_STREAM_VA, 44);
    try std.testing.expectEqual(@as(usize, 8 + 64), n);

    const header: *const uapi.drm_asahi_cmd_header = @ptrCast(@alignCast(&buf[0]));
    try std.testing.expectEqual(uapi.CMD_COMPUTE, header.cmd_type);
    try std.testing.expectEqual(@as(u16, @sizeOf(uapi.drm_asahi_cmd_compute)), header.size);
    try std.testing.expectEqual(@as(u16, 0), header.vdm_barrier);
    try std.testing.expectEqual(@as(u16, 0), header.cdm_barrier);

    const body: *const uapi.drm_asahi_cmd_compute = @ptrCast(@alignCast(&buf[8]));
    try std.testing.expectEqual(CTRL_STREAM_VA, body.cdm_ctrl_stream_base);
    try std.testing.expectEqual(CTRL_STREAM_VA + 44, body.cdm_ctrl_stream_end);
    try std.testing.expectEqual(@as(u32, 0), body.flags);
    try std.testing.expectEqual(@as(u32, 0), body.helper.binary);
}
