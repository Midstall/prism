//! AGX render launch (P4) - the first from-scratch GPU TRIANGLE on Apple
//! Silicon via Prism. Builds, entirely from scratch, a minimal render pass that
//! clears a 64x64 linear RGBA8 color buffer and rasterizes ONE solid-color
//! triangle into it, submits it through the kernel `asahi` DRM driver, waits the
//! fence, and reads a center pixel back to prove the AGX TBDR pipeline ran. This
//! is the graphics analogue of P3's hand-assembled compute launch (compute.zig),
//! and reuses P3's PROVEN patterns: PAGE-spaced 16 KiB-aligned VAs, the
//! USC_EXEC_BASE window, the USC-uniform-indirection lesson, the sentinel + 3-way
//! readback, and the EL0 cache bracket.
//!
//! GROUNDING. Every GPU-side byte here is grounded in Mesa's proven AGX encoding
//! and dougallj's applegpu, NOT guessed:
//!   - drm_asahi_cmd_render + the bg/eot/zls/attachment sub-structs: the kernel
//!     UAPI include/uapi/drm/asahi_drm.h (byte-confirmed in uapi.zig).
//!   - the VDM control stream (barrier / PPP-state / VDM-state / index-list /
//!     terminate), the PPP per-pass words, the USC descriptor words, the bg/eot
//!     program references, the PBE image descriptor, the tile config: the genxml
//!     bit layouts in Mesa src/asahi/genxml/cmdbuf.xml and the assembly order in
//!     src/asahi/lib/agx_bg_eot.c + agx_ppp.h + agx_tilebuffer.c + libagx_dgc.h
//!     (agx_vdm_draw) + src/gallium/drivers/asahi/agx_state.c (agx_encode_state,
//!     agx_build_bg_eot, agx_batch_upload_pbe) + vulkan/hk_queue.c (the
//!     drm_asahi_cmd_render fill: ppp_ctrl, ppp_multisamplectl, isp_merge).
//!   - the shader ISA (mov_imm / st_var / st_tile / block_image_store / wait /
//!     stop): dougallj's applegpu (the same reference compute.zig round-tripped).
//!
//! HONESTY. The AGX render encoding is large and parts are under-documented (the
//! "unk" bits of PPP/ZLS/bg-eot, the exact st_var/st_tile fields, the vertex_id
//! ABI register, the PBE linear layout). Where a field is genuinely uncertain it
//! is flagged inline "UNVERIFIED - M1 iteration candidate". A wrong render
//! cmdbuf GPU-FAULTS (asahi dmesg "gpu fault address 0x... reason:Unmapped"),
//! exactly the iteration signal the compute bring-up used. The live render is
//! the USER's M1 test - it CANNOT be executed on this box (no AGX GPU). This is a
//! Mesa-grounded FIRST ATTEMPT and will likely need M1 iterations.

const std = @import("std");
const uapi = @import("uapi.zig");
const dev_mod = @import("device.zig");

const Device = dev_mod.Device;
const Bo = dev_mod.Bo;
const Error = dev_mod.Error;

// ---------------------------------------------------------------------------
// COLORS - FLOAT in the shader, packed RGBA8 only for the CPU compare.
//
// FIX 2 (the M1 clear_readback=0xCA6CA178 finding): the AGX tilebuffer holds
// per-component FLOAT color registers, NOT a packed u32.
//
// FIX 3 (the M1 R3 clear_readback=0xFF261B1A finding - alpha correct, RGB junk):
// the tilebuffer color registers are FULL 32-bit (fp32) registers r0,r1,r2,r3,
// NOT fp16 half-registers. Mesa's background clear (agx_bg_eot.c
// build_background_op) loads the clear color via nir_load_preamble(.., 32, ..) =
// 4x 32-BIT values and stores them with nir_store_output (untyped 32-bit);
// agx_nir_lower_tilebuffer.store_tilebuffer keeps that 32-bit width and emits a
// single st_tile whose DATA REGISTER GROUP is the 4 consecutive 32-bit registers
// r0_r1_r2_r3 (agx_pack_alu_dst => D=2, the size-bit set). The previous version
// wrote 4 fp16 HALVES (r0l,r0h,r1l,r1h, D=0) - a register-SIZE mismatch: the HW
// read r0,r1,r2,r3 as 32-bit floats but only r0l/r0h/r1l/r1h-spanning halves
// were set, so R,G,B landed as leftover-register junk (the M1's 0x1A,0x1B,0x26)
// while alpha happened to fall correct. The fix loads each component as a full
// fp32 via the PROVEN 32-bit mov_imm (compute.zig's 62 01 .. form) into
// r0,r1,r2,r3, and st_tile reads r0_r1_r2_r3 (D=2) with MEMORY_FORMAT u8norm and
// sample mask 0xFF (broadcast). Round-tripped through dougallj applegpu:
//   mov_imm r0..r3, <fp32 bits>; wait 0; st_tile r0_r1_r2_r3, u8norm, .., 255; stop.
// The PBE then converts each float component to RGBA8-unorm on the end-of-tile
// store:
//   byte = round(clamp(component, 0, 1) * 255).
// Channel/byte order in the linear RGBA8 BO (PBE Channels=R8G8B8A8, identity
// swizzle): byte0=R, byte1=G, byte2=B, byte3=A; the CPU reads it as a
// little-endian u32, so u32 = R | G<<8 | B<<16 | A<<24.
//
// The chosen colors + their EXPECTED packed-u32 readbacks (this is what the
// probe compares the center pixel against):
//   triangle = MAGENTA   (R=1.0, G=0.0, B=1.0, A=1.0)
//              -> bytes FF 00 FF FF -> u32 0xFFFF00FF = kTriColor
//   clear    = DARK BLUE (R=0.0, G=0.0, B=0.5, A=1.0)
//              -> round(0.5*255)=128=0x80 -> bytes 00 00 80 FF -> u32 0xFF800000
//                 = kClearColor
// So the packed compare constants are UNCHANGED by the float fix - the float
// values were chosen to land on exactly these readbacks.
// ---------------------------------------------------------------------------

/// Expected packed RGBA8 readback for the MAGENTA triangle (FF 00 FF FF LE).
pub const kTriColor: u32 = 0xFFFF00FF;

/// The triangle color as 4 float components (R,G,B,A) in [0,1].
pub const tri_rgba: [4]f32 = .{ 1.0, 0.0, 1.0, 1.0 };
/// The clear color as 4 float components (R,G,B,A) in [0,1].
pub const clear_rgba: [4]f32 = .{ 0.0, 0.0, 0.5, 1.0 };

/// The sentinel the CPU pre-fills the color BO with BEFORE the submit, so the
/// readback decodes three ways (mirrors compute.zig's honest discriminator):
///   == kTriColor -> the AGX rasterized our triangle and the eot store landed.
///   == kSentinel -> nothing was stored to the center pixel (the bg clear and/or
///                   the eot store and/or the draw did not reach it - an ENCODING
///                   bug, not a stale read: AGX is CPU-coherent, see compute.zig).
///   == kClearColor -> the bg clear ran but the triangle did not cover/rasterize
///                   the center pixel (draw/VDM/PPP/viewport bug).
///   anything else -> a partial/wrong store (note + iterate).
pub const kSentinel: u32 = 0xDEADBEEF;

/// Expected packed RGBA8 readback for the DARK BLUE clear (00 00 80 FF LE).
/// Distinct from BOTH the sentinel and the triangle color: a center pixel ==
/// kClearColor means "clear ran, triangle missed".
pub const kClearColor: u32 = 0xFF800000;

// ---------------------------------------------------------------------------
// GPU virtual-address layout. Same scheme as compute.zig: a 4 GiB-aligned USC
// window (USC_EXEC_BASE == the queue's usc_exec_base), every BO its own 16 KiB
// page at a PAGE-spaced, 16-KiB-aligned VA (AGX MMU is 16-KiB-page; VM_BIND
// rejects a misaligned addr with EINVAL - the bug the M1 hit in P3 iter 1).
// The USC-relative shader code addresses and the VDM/PPP pipeline offsets are
// (VA - USC_EXEC_BASE), which fits in 32 bits. The color BO sits well above the
// USC window; the eot store reaches it via a PBE descriptor (full VA).
// ---------------------------------------------------------------------------

/// 4 GiB-aligned base of the USC executable window == the queue's usc_exec_base.
/// SHARED with compute.zig so a single queue (created with this usc_exec_base)
/// serves both paths.
pub const USC_EXEC_BASE: u64 = 0x1_0000_0000;

/// Page size for BO allocations (Apple Silicon 16 KiB page).
pub const PAGE: u64 = 0x4000;

// Each render BO is its own PAGE, PAGE-spaced inside the USC window so all USC
// offsets are small + aligned. Ordered: VS code, PS code, EOT code, USC blocks,
// VDM stream, PPP block, PBE descriptor, clear-color uniform data. The color
// attachment BO lives above the window (full-VA addressed via the PBE).
pub const VS_CODE_VA: u64 = USC_EXEC_BASE + 1 * PAGE;
pub const PS_CODE_VA: u64 = USC_EXEC_BASE + 2 * PAGE;
pub const EOT_CODE_VA: u64 = USC_EXEC_BASE + 3 * PAGE;
pub const VS_USC_VA: u64 = USC_EXEC_BASE + 4 * PAGE;
pub const PS_USC_VA: u64 = USC_EXEC_BASE + 5 * PAGE;
pub const BG_USC_VA: u64 = USC_EXEC_BASE + 6 * PAGE;
pub const EOT_USC_VA: u64 = USC_EXEC_BASE + 7 * PAGE;
pub const VDM_STREAM_VA: u64 = USC_EXEC_BASE + 8 * PAGE;
pub const PPP_VA: u64 = USC_EXEC_BASE + 9 * PAGE;
pub const PBE_VA: u64 = USC_EXEC_BASE + 10 * PAGE;
pub const CLEAR_DATA_VA: u64 = USC_EXEC_BASE + 11 * PAGE;
/// The bg (background/clear) program's OWN code BO. Previously the bg USC SHADER
/// pointed at the FS code (a placeholder + a prime hang/wrong-color suspect); now
/// the clear program has its own dedicated, self-contained, terminating shader.
pub const BG_CODE_VA: u64 = USC_EXEC_BASE + 12 * PAGE;
/// The mandatory per-control-stream init PPP word block (W-clamp setup). Its own
/// 16-KiB page so the init PPP State Update can point a clean VA at it.
pub const INIT_PPP_VA: u64 = USC_EXEC_BASE + 13 * PAGE;
/// The scissor descriptor array (isp_scissor_base) - one full-framebuffer entry.
pub const SCISSOR_VA: u64 = USC_EXEC_BASE + 14 * PAGE;
/// The depth-bias array (isp_dbias_base) - one zero entry.
pub const DBIAS_VA: u64 = USC_EXEC_BASE + 15 * PAGE;
/// The viewport/scissor/region-clip PPP State Update record (Mesa's separate
/// agx_upload_viewport_scissor block).
pub const VPS_PPP_VA: u64 = USC_EXEC_BASE + 16 * PAGE;
/// The MANDATORY graphics sampler heap - a 1-entry heap holding the reserved
/// txf sampler #0. Mesa's sampler_count() unconditionally adds +1 (reserves
/// sampler #0 for txf), so EVERY graphics VS/FS USC binds a Sampler state word
/// pointing at this heap and declares sampler_state_register_count = 4 compact
/// (1 sampler). agx_state.c sampler_count()+1 + agx_helpers.h agx_pack_txf_sampler.
pub const SAMPLER_HEAP_VA: u64 = USC_EXEC_BASE + 17 * PAGE;
/// The FRAGMENT color uniform data BO - 4 fp16 RGBA halves the FS reads via the
/// USC Uniform word (Mesa's proven uniform-sourced color path: the noddtri FS
/// does `mov r4,u8 / mov r5,u9 / st_tile r4l_r4h_r5l_r5h`, reading the color out
/// of uniform registers loaded from this buffer). mesa_noddtri.txt:265-274.
pub const FS_COLOR_VA: u64 = USC_EXEC_BASE + 18 * PAGE;
/// The 64x64 linear RGBA8 color attachment, above the USC window.
pub const COLOR_VA: u64 = USC_EXEC_BASE + 0x1_0000_0000;

// Framebuffer geometry: 64x64 RGBA8, single sample, single layer. A 64x64
// surface is trivial to read back a center pixel from (row*stride + col*4).
pub const FB_WIDTH: u16 = 64;
pub const FB_HEIGHT: u16 = 64;
/// 4 bytes/pixel, 64 px/row, cacheline-aligned (128) per Mesa
/// ail_initialize_linear: ALIGN_POT(64*4, 0x80) = 256 (already 128-aligned).
pub const COLOR_STRIDE: u32 = 64 * 4;
pub const COLOR_SIZE: u64 = @as(u64, COLOR_STRIDE) * FB_HEIGHT;

// Tile config for 64x64 RGBA8 1-sample, from agx_select_tile_size +
// agx_tilebuffer (CONFIRMED): sample_size_B = align(4,8) = 8; tile = 32x32 =>
// utile 32x32; samples 1.
pub const UTILE_W: u8 = 32;
pub const UTILE_H: u8 = 32;
pub const SAMPLE_SIZE_B: u8 = 8;

// ---------------------------------------------------------------------------
// Little-endian bit writer (same as compute.zig - the genxml structs are LE bit
// arrays). Sized for the largest single word group we build.
// ---------------------------------------------------------------------------
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

    fn emit(self: *const BitWriter, out: []u8, nbytes: usize) usize {
        var i: usize = 0;
        while (i < nbytes) : (i += 1) {
            out[i] = @truncate(self.words[i / 4] >> @intCast((i % 4) * 8));
        }
        return nbytes;
    }
};

/// __gen_to_groups(value, group_size, length): stored = ceil(value/group_size);
/// 0 clamps to 1; full (==2^length) -> 0. (Same as compute.zig.)
fn toGroups(value: u32, group_size: u32, length: u5) u32 {
    if (value == 0) return 1;
    const groups = (value + group_size - 1) / group_size;
    if (groups == (@as(u32, 1) << length)) return 0;
    return groups;
}

/// Bit pattern of a float literal (the genxml f32 register fields take the raw
/// IEEE-754 bits). comptime-friendly.
fn fui(comptime f: f32) u32 {
    return @bitCast(f);
}

/// Runtime variant of `fui`: the raw IEEE-754 bits of a runtime f32 value.
fn fui_rt(f: f32) u32 {
    return @bitCast(f);
}

/// IEEE-754 binary16 (fp16) bit pattern of an f32 value. The AGX tilebuffer
/// color registers are fp16 halves, so the FS/bg color components are loaded as
/// fp16 immediates. Zig's f16 cast does the proper round-to-nearest.
fn f16bits(f: f32) u16 {
    return @bitCast(@as(f16, @floatCast(f)));
}

// ===========================================================================
// SHADER ISA. Three hand-assembled AGX (G13/M1 Pro) shaders. The mov_imm / wait
// / stop encodings are IDENTICAL to compute.zig (already round-tripped through
// dougallj's applegpu + PROVEN live on the M1). The graphics-specific
// instructions (st_var, st_tile, block_image_store) are encoded from the Mesa
// agx_pack.c packers + applegpu.py field layouts; they are NOT yet round-tripped
// on hardware, so they are the prime first-fault suspects (flagged below).
// ===========================================================================

// --- st_var (store varying / position) -------------------------------------
// Mesa agx_pack.c AGX_OPCODE_ST_VARY packer (the authoritative bit layout):
//   raw = 0x11 | (last?1<<7:0) | ((value&0x3F)<<9) | ((index&0x3F)<<16)
//       | (imm_index?1<<23:0) | ((value>>6)<<24) | ((index>>6)<<26) | (0x8<<28)
// where `value` is the SOURCE register in 16-bit-granular units (a 32-bit reg
// rN => value = 2*N), `index` is the UVS vertex-output slot, `last` marks the
// final varying store, `imm_index` = use an immediate slot index (our case).
// applegpu.py table: 0x11 st_var / 0x91 st_var_final / 0x51 no_var (opcode in
// low 10 bits); ExReg32Desc('r',10,24) (it drops the reg low bit, == Mesa's
// value>>1 convention); UVSIndexDesc the slot. 4-byte instruction.
//
// UNVERIFIED - M1 iteration candidate: the (0x8<<28) constant (applegpu marks it
// "XXX"), the imm-index bit23 semantics, and the 16-bit-granular register
// convention. If the VS faults, st_var encoding is the first suspect.

/// Encode one `st_var` (4 bytes LE) storing 32-bit register `reg` to UVS slot
/// `slot`, with an immediate slot index. `last` sets the final-varying bit.
fn encodeStVar(out: []u8, reg: u32, slot: u32, last: bool) void {
    const value: u32 = reg * 2; // 32-bit reg -> 16-bit-granular unit
    var raw: u64 = 0x11;
    if (last) raw |= (1 << 7);
    raw |= @as(u64, value & 0x3F) << 9;
    raw |= @as(u64, slot & 0x3F) << 16;
    raw |= (1 << 23); // immediate slot index
    raw |= @as(u64, value >> 6) << 24;
    raw |= @as(u64, slot >> 6) << 26;
    raw |= @as(u64, 0x8) << 28;
    std.mem.writeInt(u32, out[0..4], @truncate(raw), .little);
}

// --- mov_imm (load a 32-bit immediate into a register) ----------------------
// IDENTICAL to compute.zig (PROVEN): opcode low7 = 0x62, bit8=1 (32-bit imm
// form), imm32 at byte offset 2 LE. 6 bytes. The first byte's bit8 lives in the
// second byte; for r0 the bytes are 62 01 <imm32 LE>. For a general reg rN we
// follow compute.zig's exact form (it only ever used r0); higher regs change the
// destination field. To stay on the PROVEN encoding we MOV into r0..r3 using the
// same 6-byte shape with the register selector in byte 1.
//
// UNVERIFIED - M1 iteration candidate: the destination-register field for r1..r3
// (compute.zig only proved r0). applegpu MovImm32InstructionDesc: the dest is a
// register field; 62 01 is r0. We set the dest by ORing the reg number into the
// proven layout (see encodeMovImm). If a position component lands in the wrong
// register, this is the suspect.

/// Encode `mov_imm rN, imm32` (6 bytes). Mirrors compute.zig's proven 62 01 ..
/// shape, with the destination register encoded. r0 reproduces compute.zig
/// byte-for-byte.
fn encodeMovImm(out: []u8, reg: u32, imm: u32) void {
    // applegpu MovImm32: opcode 0b1100010 (0x62) low7; the destination register
    // and the 32-bit-imm select share the second halfword. compute.zig proved
    // `62 01 <imm LE>` == mov_imm r0, imm (bit8 select + dest r0). The dest reg
    // is a 6-bit field; r0 -> the 0x01 here is the (bit8 imm-select | dest=0).
    // For rN we OR (reg<<1) into that byte (reg in 16-bit-granular units like
    // st_var). UNVERIFIED for reg>0.
    out[0] = 0x62;
    out[1] = @truncate(0x01 | ((reg * 2) << 1));
    std.mem.writeInt(u32, out[2..6], imm, .little);
}

// --- mov_imm16 (load a 16-bit immediate into a 16-bit half register) --------
// applegpu MovImm16InstructionDesc (round-tripped through the disassembler):
//   opcode low7 = 0b1100010 (0x62); bit8 = 0 (16-bit imm form selector, vs the
//   32-bit form's bit8=1); the destination half-register D in bits [9..14] +
//   [44..45], with the size-flags field Dt @7(2) = 0 for a 16-bit register;
//   the imm16 at bits [16..31]. The length bit @15 selects 4 vs 6 bytes; we
//   FORCE it to 1 (6-byte form) so the on-chip decoder always advances exactly
//   6 bytes - emitting a short (4-byte) form but reserving 6 would leave a
//   trailing 00 00 that decodes as `jmp_incomplete pc+0` (an infinite self-jump
//   = a hang). Half registers: r0l=0, r0h=1, r1l=2, r1h=3, ...
//
// The destination is addressed in 16-bit-half units (`half_reg`): a 32-bit
// register rN occupies halves 2N (low) and 2N+1 (high).

/// Encode `mov_imm <half_reg>, imm16` (6 bytes, forced long form). `half_reg`
/// is a 16-bit half-register index (r0l=0, r0h=1, r1l=2, ...).
fn encodeMovImm16(out: []u8, half_reg: u32, imm16: u16) void {
    var raw: u64 = 0b1100010; // opcode low7
    // bit8 stays 0 (16-bit imm form).
    raw |= @as(u64, half_reg & 0x3F) << 9; // D low
    raw |= @as(u64, half_reg >> 6) << 44; // D high
    raw |= @as(u64, 1) << 15; // force the 6-byte length form
    raw |= @as(u64, imm16) << 16; // imm16
    var tmp: [8]u8 = undefined;
    std.mem.writeInt(u64, &tmp, raw, .little);
    @memcpy(out[0..6], tmp[0..6]);
}

/// Encode `mov rN, uM` (6 bytes) - copy 32-bit uniform register `uM` into the
/// 32-bit register `rN`. This is the `mov` (bitop-alias) read of a uniform that
/// Mesa's noddtri FS uses to source the color (mesa_noddtri.txt:287 `mov r4,u8`).
/// Encoding (round-tripped through dougallj applegpu's assembler):
///   byte0=0x7e, byte1=(rN<<2)|1, byte2=0x80|(uM<<1), byte3..5=09 80 00.
/// VERIFIED byte-identical for r4,u8 (7e1190098000) and r5,u9 (7e1592098000).
fn encodeMovFromUniform(out: []u8, reg: u32, uni: u32) void {
    out[0] = 0x7e;
    out[1] = @truncate((reg << 2) | 1);
    out[2] = @truncate(0x80 | (uni << 1));
    out[3] = 0x09;
    out[4] = 0x80;
    out[5] = 0x00;
}

/// `wait 0` (2 bytes) - PROVEN in compute.zig. Waits for outstanding LOADS, NOT
/// the pixel-ready fence (that is wait_pix, see WAIT_PIX below).
const WAIT: [2]u8 = .{ 0x38, 0x00 };
/// `stop` (2 bytes) - PROVEN in compute.zig.
const STOP: [2]u8 = .{ 0x88, 0x00 };

/// `wait_pix <mask>` (4 bytes): the FRAGMENT-stage pixel-ordering fence. The AGX
/// compiler (agx_compile.c agx_emit_local_store_pixel L674-676) UNCONDITIONALLY
/// emits `agx_wait_pixel_mask(b, 0xC)` before every fragment-shader st_tile, so
/// the tilebuffer store synchronises against the ISP pixel-ready signal. Mesa's
/// egltri FS shows it verbatim: `480c0000 = pixwait 12, 0`. WITHOUT it a real
/// fragment shader's st_tile can stall the pixel pipe with NO fault address - the
/// exact draw-hangs-but-clear-works signature (the bg/eot tile-control programs
/// are NOT fragment-stage, line 674 gates on MESA_SHADER_FRAGMENT, so they never
/// get a wait_pix - which is precisely why the clear path works without one).
/// Encoding (dougallj applegpu wait_pix): constant 0x48 @byte0, mask `i` @bits
/// 8..15, `j` @bits 22..23 = 0. For mask 0xC: bytes 48 0C 00 00.
const WAIT_PIX_C: [4]u8 = .{ 0x48, 0x0C, 0x00, 0x00 };

// ---------------------------------------------------------------------------
// (A) The VERTEX shader (REAL, vertex_id-indexed - the P4 R3 fix). AGX has NO
// fixed-function vertex fetch; the VS runs once per vertex with the vertex id
// preloaded in an ABI input register, and writes gl_Position (4 floats) to UVS
// slots 0..3 via st_var (position is always the first 4 UVS slots -
// agx_nir_lower_uvs.c group_offs[UVS_POSITION]==0, x->0 y->1 z->2 w->3).
//
// VERTEX_ID ABI (Mesa src/asahi/lib/agx_abi.h): AGX_ABI_VIN_VERTEX_ID = 2*5 = 10,
// a 16-bit-HALF-register index, so vertex_id is preloaded as a 32-bit value in
// register r5 (halves 10,11). agx_compile.c agx_vertex_id() reads
// agx_register(10, AGX_SIZE_32) == r5. The VS reads r5 directly.
//
// THE COVERING "FULLSCREEN" TRIANGLE selected by vertex_id (the standard
// gl_VertexID trick), giving 3 DISTINCT clip-space positions:
//   xi = (vid << 1) & 2  -> vid0=0, vid1=2, vid2=0
//   yi =  vid       & 2  -> vid0=0, vid1=0, vid2=2
//   x  = float(xi)*2 - 1 -> -1,  3, -1
//   y  = float(yi)*2 - 1 -> -1, -1,  3
//   z  = 0, w = 1
// => verts (-1,-1), (3,-1), (-1,3): a triangle that strictly covers the whole
// [-1,1]^2 clip square, so the framebuffer CENTER (0,0) is definitely inside.
//
// All integer/float ALU bytes below were ASSEMBLED + ROUND-TRIPPED through
// dougallj's applegpu (assemble_line -> disassemble_n confirms each mnemonic +
// the whole stream walks to `stop` with no desync):
//   iadd    r0, r5, r5            ; r0 = vid*2  (vid<<1)
//   and     r0, r0, 2            ; r0 = (vid*2) & 2   (xi)
//   and     r1, r5, 2            ; r1 = vid & 2       (yi)
//   convert u32_to_f, r0, r0, rte; r0 = float(xi)
//   convert u32_to_f, r1, r1, rte; r1 = float(yi)
//   fmadd32 r0, r0, 2.0, -1.0    ; x = xi*2 - 1
//   fmadd32 r1, r1, 2.0, -1.0    ; y = yi*2 - 1
//   mov_imm r2, 0.0 ; mov_imm r3, 1.0
//   st_var{0,1,2,3}; stop
// The VS now uses r0,r1,r2,r3,r5 -> 6 GPRs (the USC REGISTERS word must cover
// r5; buildVsUsc is called with gpr_count=6).
// ---------------------------------------------------------------------------

/// `iadd r0, r5, r5` (8B) - r0 = vid*2. Disassembler-verified bytes.
const VS_IADD_R0_VID2: [8]u8 = .{ 0x0E, 0x01, 0x4A, 0xA2, 0x24, 0x00, 0x00, 0x00 };
/// `and r0, r0, 2` (6B) - r0 = (vid*2) & 2.
const VS_AND_R0_2: [6]u8 = .{ 0x7E, 0x01, 0x40, 0x22, 0x80, 0x00 };
/// `and r1, r5, 2` (6B) - r1 = vid & 2.
const VS_AND_R1_VID_2: [6]u8 = .{ 0x7E, 0x05, 0x4A, 0x22, 0x80, 0x00 };
/// `convert u32_to_f, r0, r0, rte` (6B) - r0 = float(r0).
const VS_CVT_R0: [6]u8 = .{ 0x3E, 0x81, 0x0A, 0x04, 0x24, 0x00 };
/// `convert u32_to_f, r1, r1, rte` (6B) - r1 = float(r1).
const VS_CVT_R1: [6]u8 = .{ 0x3E, 0x85, 0x0A, 0x24, 0x24, 0x00 };
/// `fmadd32 r0, r0, 2.0, -1.0` (8B) - x = r0*2 - 1.
const VS_FMADD_R0: [8]u8 = .{ 0x3A, 0x81, 0x40, 0x02, 0x00, 0x30, 0x80, 0x01 };
/// `fmadd32 r1, r1, 2.0, -1.0` (8B) - y = r1*2 - 1.
const VS_FMADD_R1: [8]u8 = .{ 0x3A, 0x85, 0x42, 0x02, 0x00, 0x30, 0x80, 0x01 };

/// The number of GPRs the (real) vertex shader uses: r0..r3 + r5 (vertex_id).
pub const VS_GPR_COUNT: u32 = 6;

/// Build the REAL vertex_id-indexed vertex shader binary into `out`, returns
/// bytes written. Computes the covering fullscreen-triangle clip position from
/// the vertex_id in r5, writes (x,y,z,w) to UVS slots 0..3, then stop. The 3
/// vertices are DISTINCT ((-1,-1),(3,-1),(-1,3)), so the triangle covers the
/// framebuffer center.
pub fn buildVertexShader(out: []u8) usize {
    var n: usize = 0;
    @memcpy(out[n..][0..8], &VS_IADD_R0_VID2);
    n += 8;
    @memcpy(out[n..][0..6], &VS_AND_R0_2);
    n += 6;
    @memcpy(out[n..][0..6], &VS_AND_R1_VID_2);
    n += 6;
    @memcpy(out[n..][0..6], &VS_CVT_R0);
    n += 6;
    @memcpy(out[n..][0..6], &VS_CVT_R1);
    n += 6;
    @memcpy(out[n..][0..8], &VS_FMADD_R0);
    n += 8;
    @memcpy(out[n..][0..8], &VS_FMADD_R1);
    n += 8;
    encodeMovImm(out[n..], 2, fui(0.0)); // r2 = z = 0.0
    n += 6;
    encodeMovImm(out[n..], 3, fui(1.0)); // r3 = w = 1.0
    n += 6;
    // st_var r0->slot0 (x), r1->slot1 (y), r2->slot2 (z), r3->slot3 (w, last).
    encodeStVar(out[n..], 0, 0, false);
    n += 4;
    encodeStVar(out[n..], 1, 1, false);
    n += 4;
    encodeStVar(out[n..], 2, 2, false);
    n += 4;
    encodeStVar(out[n..], 3, 3, true); // last varying store sets the final bit
    n += 4;
    @memcpy(out[n..][0..2], &STOP);
    n += 2;
    return n;
}

// ---------------------------------------------------------------------------
// (B) The FRAGMENT shader. Writes a constant color to color output 0 (the
// tilebuffer) via st_tile, then stop. The tilebuffer eot store later copies it
// to memory. A real FS issues wait_pix before st_tile; for a minimal single-draw
// constant-color FS we follow the proven `wait 0` + store + stop shape.
//
// Mesa agx_pack.c AGX_OPCODE_ST_TILE packer (8 bytes), mirrored EXACTLY:
//   D  = agx_pack_alu_dst(data reg). For a 32-bit register rN that is
//        (2N << 1) | 1 (the low size-bit set marks 32-bit). For a tilebuffer
//        u8norm color the data is a contiguous group of `popcount(mask)` 32-BIT
//        registers starting at that base, so RGBA8 xyzw reads r0,r1,r2,r3 with
//        D=2 (FIX 3: the OLD D=0 read fp16 halves r0l_r0h_r1l_r1h, a register-
//        size mismatch vs Mesa's 32-bit clear/output values - the RGB-junk bug).
//   raw = 0x09
//       | ((D>>1)&0x7F)<<8         data reg low
//       | (St)<<22                 sample-mask is register (0 = immediate)
//       | (format)<<24             MEMORY_FORMAT (u8norm=4)
//       | (C&0x3F)<<16             coords reg (0, implicit)
//       | (pixel_offset&0x7F)<<28  tilebuffer byte offset for the RT (RT0 -> 0)
//       | (load||explicit ? 1<<35 : 0)  (0 here)
//       | (mask)<<36               write mask (0xF = xyzw)
//       | (pixel_offset>>7)<<40
//       | (S&0x3F)<<42             sample mask low  (S=0xFF broadcast!)
//       | (S>>6)<<56               sample mask high
//       | ((D>>8)&0xF)<<60         data reg high
// THE OLD CODE WROTE S=0 (no sample mask) = store to NO samples = nothing
// landed in the tilebuffer. Mesa's tilebuffer store defaults samples to
// ALL_SAMPLES = 0xFF (agx_nir_lower_tilebuffer.c). That is the broadcast value.
// ---------------------------------------------------------------------------

/// MEMORY_FORMAT / AGX_FORMAT u8norm (= 4). The PBE converts these float
/// tilebuffer components to RGBA8-unorm on the end-of-tile store.
const AGX_FORMAT_U8NORM: u64 = 4;
/// st_tile sample mask: broadcast to all samples (ALL_SAMPLES in Mesa).
const ST_TILE_ALL_SAMPLES: u64 = 0xFF;

/// Encode one `st_tile` (8 bytes LE): store the 4-component fp32 color held in
/// the FULL 32-bit register group starting at register `reg32_base` (r0_r1_r2_r3
/// for reg32_base=0) to the tilebuffer at byte offset `pixel_offset` (0 for RT0),
/// mask 0xF (RGBA), format u8norm, sample mask 0xFF (broadcast).
///
/// agx_pack_alu_dst for a 32-bit register rN yields D = (N << 2) | (1 << 1) (the
/// bit-1 "size>=32" flag set, the register number in bits 2+). For r0 that is
/// D=2, which the dougallj disassembler decodes as `st_tile r0_r1_r2_r3` (4
/// consecutive 32-bit regs) - the Mesa-faithful encoding. (D=0, the old value,
/// decoded as `r0l_r0h_r1l_r1h`, the fp16-halves register-size mismatch that
/// produced the M1's RGB junk.)
fn encodeStTile(out: []u8, reg32_base: u32, pixel_offset: u32) void {
    const d: u64 = (@as(u64, reg32_base) << 2) | (1 << 1); // agx_pack_alu_dst (32-bit reg)
    const fmt: u64 = AGX_FORMAT_U8NORM;
    const mask: u64 = 0xF; // xyzw
    const s: u64 = ST_TILE_ALL_SAMPLES;
    const po: u64 = pixel_offset;
    var raw: u64 = 0x09; // opcode, store
    raw |= ((d >> 1) & 0x7F) << 8; // data reg low
    raw |= fmt << 24; // MEMORY_FORMAT
    // coords reg C = 0 (implicit) at bits 16..21 -> nothing to OR.
    raw |= (po & 0x7F) << 28; // pixel offset low
    raw |= mask << 36; // write mask
    raw |= (po >> 7) << 40; // pixel offset high
    raw |= (s & 0x3F) << 42; // sample mask low
    raw |= (s >> 6) << 56; // sample mask high
    raw |= ((d >> 8) & 0xF) << 60; // data reg high
    std.mem.writeInt(u64, out[0..8], raw, .little);
}

/// Encode one `st_tile` (8 bytes LE) reading the 4 color components from a group
/// of FOUR consecutive 16-bit HALF registers starting at half index `half_base`
/// (r4l_r4h_r5l_r5h for half_base=8). This is MESA'S PROVEN fragment-shader
/// tilebuffer store (mesa_noddtri.txt:291): the FS loads the color as fp16 halves
/// from a uniform (mov r4,u8 / mov r5,u9) and stores them. The genxml ld/st_tile
/// data register field is `Rt`@8 (the 32-bit-size bit, 0 for 16-bit halves) +
/// `R`@9(6) (the register base, in 16-bit-half units) + `Rx`@60(2) (high bits).
/// For r4l_r4h_r5l_r5h: Rt=0, R=8 -> byte1 = 0x10, byte-identical to Mesa's
/// 09 10 00 04 f0 fc 00 03 (round-tripped through dougallj applegpu).
fn encodeStTileHalves(out: []u8, half_base: u32, pixel_offset: u32) void {
    const r: u64 = half_base; // 16-bit-half register base
    const fmt: u64 = AGX_FORMAT_U8NORM;
    const mask: u64 = 0xF; // xyzw
    const s: u64 = ST_TILE_ALL_SAMPLES;
    const po: u64 = pixel_offset;
    var raw: u64 = 0x09; // opcode, store
    // Rt@8 = 0 (16-bit half registers, NOT the 32-bit form).
    raw |= (r & 0x3F) << 9; // R: register base (halves)
    raw |= (r >> 6) << 60; // Rx: register base high
    raw |= fmt << 24; // MEMORY_FORMAT (u8norm)
    raw |= (po & 0x7F) << 28; // pixel offset low
    raw |= mask << 36; // write mask (xyzw)
    raw |= (po >> 7) << 40; // pixel offset high
    raw |= (s & 0x3F) << 42; // sample mask low (broadcast 0xFF)
    raw |= (s >> 6) << 56; // sample mask high
    std.mem.writeInt(u64, out[0..8], raw, .little);
}

/// Emit a constant-color tilebuffer-store program into `out`: load the 4 float
/// components `rgba` as fp32 into the FULL registers r0,r1,r2,r3 (the proven
/// 32-bit mov_imm form), then wait + st_tile (r0_r1_r2_r3, RT0, u8norm, mask 0xF,
/// broadcast sample mask) + stop. Returns bytes written (FS=38 with wait_pix,
/// bg=36 with the wait no-op). This is the shared shape of the fragment shader
/// and the background/clear shader - both write a constant color, differing in
/// the color AND the pixel fence. Mesa-faithful: the tilebuffer color regs are
/// 32-bit (FIX 3).
/// `frag` selects the pixel-synchronisation instruction before the st_tile:
///   - frag=true  (the real FRAGMENT shader): emit `wait_pix 0xC`, the mandatory
///     fragment-stage pixel-ready fence Mesa emits before every FS st_tile. This
///     is the draw-hang fix - a fragment-stage st_tile WITHOUT it stalls the
///     pixel pipe (no fault address).
///   - frag=false (the bg/clear tile-control program): NO pixel fence (Mesa's
///     background program is not a fragment-stage shader and has no wait at all;
///     we keep the proven `wait 0` no-op here to leave the working clear path
///     byte-stable).
fn buildColorStore(out: []u8, rgba: [4]f32, frag: bool) usize {
    var n: usize = 0;
    // 4 fp32 components into the 4 full registers r0,r1,r2,r3.
    var i: u32 = 0;
    while (i < 4) : (i += 1) {
        encodeMovImm(out[n..], i, fui_rt(rgba[i]));
        n += 6;
    }
    if (frag) {
        // FRAGMENT stage: the mandatory pixel-ready fence before the tile store.
        @memcpy(out[n..][0..4], &WAIT_PIX_C);
        n += 4;
    } else {
        // bg/clear tile-control program: keep the proven (load) wait no-op.
        @memcpy(out[n..][0..2], &WAIT);
        n += 2;
    }
    encodeStTile(out[n..], 0, 0); // st_tile r0_r1_r2_r3 -> RT0, offset 0
    n += 8;
    @memcpy(out[n..][0..2], &STOP);
    n += 2;
    return n;
}

/// The fp16 RGBA color the FS reads out of its uniform (4 halves: R,G,B,A).
/// These are written into the FS_COLOR_VA BO and DMA'd into uniform registers
/// u8,u9 (halves 16..19) by the FS USC Uniform word. For MAGENTA (1,0,1,1) ->
/// [0x3C00, 0x0000, 0x3C00, 0x3C00].
pub fn fsColorHalves() [4]u16 {
    return .{ f16bits(tri_rgba[0]), f16bits(tri_rgba[1]), f16bits(tri_rgba[2]), f16bits(tri_rgba[3]) };
}

/// Build the fragment shader binary into `out`, returns bytes written. This is
/// MESA'S PROVEN noddtri fragment shader (mesa_noddtri.txt:287-292), the
/// config-matched reference for our solid-color no-depth triangle:
///   mov     r4, u8     ; r4l=R(fp16), r4h=G(fp16)  [uniform halves 16,17]
///   mov     r5, u9     ; r5l=B(fp16), r5h=A(fp16)  [uniform halves 18,19]
///   wait_pix 12, 0     ; mandatory fragment-stage pixel-ready fence
///   st_tile r4l_r4h_r5l_r5h, u8norm, 0, xyzw, 0, 255, 0
///   stop
/// The color is SOURCED FROM A UNIFORM (loaded by the FS USC Uniform word from
/// FS_COLOR_VA), NOT baked - matching Mesa AND making the FS consistent with the
/// non-zero uniform_register_count the shader-word now declares (a graphics shader
/// must declare its register file; a uniform-sourced read keeps the declaration,
/// the USC Uniform word, and the program in lockstep, the way Mesa does it).
/// The tilebuffer color registers are the fp16 HALVES r4l/r4h/r5l/r5h (Mesa's
/// store form for a u8norm RT loaded from fp16 uniforms - the noddtri ground
/// truth), NOT the 4x fp32 r0_r1_r2_r3 form the bg-clear still uses.
pub fn buildFragmentShader(out: []u8) usize {
    var n: usize = 0;
    encodeMovFromUniform(out[n..], 4, 8); // mov r4, u8  (R,G)
    n += 6;
    encodeMovFromUniform(out[n..], 5, 9); // mov r5, u9  (B,A)
    n += 6;
    @memcpy(out[n..][0..4], &WAIT_PIX_C); // wait_pix 12, 0
    n += 4;
    encodeStTileHalves(out[n..], 8, 0); // st_tile r4l_r4h_r5l_r5h (half base 8)
    n += 8;
    @memcpy(out[n..][0..2], &STOP);
    n += 2;
    return n;
}

// ---------------------------------------------------------------------------
// (B') The BACKGROUND (clear) shader. The hardware auto-runs this per tile (on
// load/PROCESS_EMPTY_TILES) to initialise the on-chip tilebuffer before the
// draw. Mesa's clear program (src/asahi/lib/agx_bg_eot.c build_background_op +
// agx_build_background_shader, lines 100-129) is a NIR `load_preamble (uniform)
// -> store_output`, lowered by agx_nir_lower_tilebuffer.c (store_tilebuffer ->
// store_local_pixel_agx) to a single st_tile of the loaded clear color, then
// `stop` (NO trap footer - only the eot store gets the trap footer).
//
// It has its OWN dedicated code BO (no FS-code aliasing) and writes the clear
// color as 4 FLOAT components (buildColorStore), exactly like the FS but with
// the dark-blue clear color. This is the FIX 2 correction: the previous version
// baked a PACKED u32 (and -inf bits), which the float tilebuffer + PBE u8norm
// conversion turned into garbage (the M1's 0xCA6CA178). With 4 fp16 floats +
// the broadcast sample mask, the PBE converts to exactly 00 00 80 FF =
// kClearColor (0xFF800000).
//
// HONESTY (the one deviation from Mesa, flagged): Mesa loads the clear color
// from the bg USC Uniform (load_preamble of 4 fp32) rather than baking it. We
// BAKE the 4 float components as fp16 immediates so the clear program needs no
// uniform-register read (not yet round-tripped on hardware) - the bg USC still
// binds the CLEAR_DATA uniform (harmless, ignored by this baked program) so the
// USC block stays Mesa-shaped. Baking is a valid, self-contained, TERMINATING
// clear. Reading the uniform is a follow-up M1 iteration.
// ---------------------------------------------------------------------------

/// Build the background/clear shader binary into `out`, returns bytes written.
/// Writes the DARK BLUE clear color as 4 float components (same shape as the FS,
/// distinct color, its OWN code BO - no FS-code aliasing).
pub fn buildBgClearShader(out: []u8) usize {
    return buildColorStore(out, clear_rgba, false);
}

/// Build the background/clear shader baking an ARBITRARY caller-supplied clear
/// color (4 fp32 RGBA components in [0,1]). Byte-for-byte the PROVEN bg-clear
/// encoding (buildColorStore frag=false: 4x fp32 mov_imm + wait + st_tile
/// r0_r1_r2_r3 u8norm 0xFF + stop), only the immediate color words differ. This
/// is what threads the HAL clear color through to the GPU without touching the
/// proven st_tile / tile-config encoding. `buildBgClearShader` is exactly this
/// called with `clear_rgba`, so the self-test still reproduces 0xFF800000.
pub fn buildBgClearShaderColor(out: []u8, rgba: [4]f32) usize {
    return buildColorStore(out, rgba, false);
}

// ---------------------------------------------------------------------------
// (C) The END-OF-TILE store shader. The hardware auto-runs this per tile to copy
// the on-chip tilebuffer to the linear color attachment in memory, via a PBE
// (image) descriptor bound by the eot USC Texture word. It is a compute-style
// kernel doing one block_image_store then stop.
//
// Mesa agx_pack.c AGX_OPCODE_BLOCK_IMAGE_STORE packer (10 bytes, 3 LE words):
//   word0 = 0xB1 | (1<<15) | ((F&1)<<8) | ((R&0x3F)<<9) | ((C&0x3F)<<16)
//         | (Ct<<22) | (explicit<<23) | (1u<<31)
//   word1 = (T&0x3F) | (Tt<<6) | ((dim&7)<<8) | (9<<11) | (Cs<<15) | (U<<16)
//         | (((dim&8)?1:0)<<23) | ((R>>6)<<24) | ((C>>6)<<26)
//   word2 = (F>>1) | (1<<3) | ((T>>6)<<14)
// F = agx_format (RGBA8 tilebuffer -> U8NORM = 4); dim = 2 (2D); R = the GPR
// holding the tilebuffer byte offset (RT0 -> 0, so a reg pre-set to 0); T = the
// texture/PBE state index (RT0 -> 0); C = coords reg (0); U = 0 (base zero).
// Mesa appends `stop` then an 8x `trap` (08 00) footer.
//
// UNVERIFIED - M1 iteration candidate (HIGH): the whole block_image_store
// encoding, the tilebuffer-offset register needing to actually hold 0, and the
// agx_format value. If the eot faults (the color BO never gets written) this is
// the suspect. This is the program that ACTUALLY moves pixels to memory.
// ---------------------------------------------------------------------------

/// Encode one `block_image_store` (10 bytes). RT0, 2D, U8NORM, offset reg `r_off`
/// (must hold the tilebuffer byte offset, 0 for RT0), PBE index `tex`.
fn encodeBlockImageStore(out: []u8, r_off: u32, tex: u32) void {
    const f: u64 = 4; // agx_format U8NORM
    const dim: u64 = 2; // 2D
    const r: u64 = r_off;
    const c: u64 = 0; // coords reg (implicit)
    const t: u64 = tex; // PBE state index
    var word0: u64 = 0xB1;
    word0 |= (1 << 15);
    word0 |= (f & 1) << 8;
    word0 |= (r & 0x3F) << 9;
    word0 |= (c & 0x3F) << 16;
    // Ct (bit22)=0 coords discard; explicit(bit23)=0; unk1(bit31)=1.
    word0 |= (1 << 31);
    var word1: u64 = (t & 0x3F);
    // Tt(bit6)=0 immediate texture index; dim bits 8..10; (9<<11) constant.
    word1 |= (dim & 7) << 8;
    word1 |= (9 << 11);
    // Cs(bit15)=0; U(bit16)=0; dim&8 high bit (bit23)=0; R high (bit24); C high.
    word1 |= (r >> 6) << 24;
    word1 |= (c >> 6) << 26;
    var word2: u64 = (f >> 1);
    word2 |= (1 << 3); // unk3 forced
    word2 |= (t >> 6) << 14;
    std.mem.writeInt(u32, out[0..4], @truncate(word0), .little);
    std.mem.writeInt(u32, out[4..8], @truncate(word1), .little);
    std.mem.writeInt(u16, out[8..10], @truncate(word2), .little);
}

/// Build the eot store shader binary into `out`, returns bytes written. One
/// block_image_store (RT0, offset reg r0 which we leave as whatever launches as
/// 0) + stop + an 8x trap footer (Mesa's terminator).
pub fn buildEotShader(out: []u8) usize {
    var n: usize = 0;
    // r0 := 0 (the tilebuffer byte offset for RT0). mov_imm r0,0 keeps the
    // store's offset register well-defined rather than trusting launch state.
    encodeMovImm(out[n..], 0, 0);
    n += 6;
    encodeBlockImageStore(out[n..], 0, 0); // offset reg r0, PBE index 0
    n += 10;
    @memcpy(out[n..][0..2], &STOP);
    n += 2;
    // 8x trap footer (08 00) - Mesa agx_pack_binary appends this after stop.
    var t: usize = 0;
    while (t < 8) : (t += 1) {
        out[n] = 0x08;
        out[n + 1] = 0x00;
        n += 2;
    }
    return n;
}

// ===========================================================================
// USC descriptor words (the "pipeline" the VDM/PPP/ISP bind). Tags + the
// BitWriter are the SAME as compute.zig. Each USC word is a tagged LE byte
// block. Tag bytes from genxml enum "USC Control" (byte-confirmed):
//   Shader=0x0d Uniform=0x1d UniformHigh=0x3d Shared=0x4d Registers=0x8d
//   Sampler=0x9d Texture=0xdd NoPreshader=0x88 Preshader=0x38 FragProps=0x58.
// ===========================================================================

const USC_TAG_SHADER: u8 = 0x0d;
const USC_TAG_UNIFORM: u8 = 0x1d;
const USC_TAG_SHARED: u8 = 0x4d;
const USC_TAG_REGISTERS: u8 = 0x8d;
const USC_TAG_SAMPLER: u8 = 0x9d;
const USC_TAG_TEXTURE: u8 = 0xdd;
const USC_TAG_NO_PRESHADER: u8 = 0x88;
const USC_TAG_FRAGMENT_PROPERTIES: u8 = 0x58;

/// genxml "Sampler states" enum value for 1 sampler ("4 compact" = 1). The
/// shader-word Sampler-state-register-count field stores this for a count of 1
/// (agx_helpers.h agx_translate_sampler_state_count: count<=4 -> 4_COMPACT).
const SAMPLER_STATES_4_COMPACT: u3 = 1;

/// AGX_SHARED_LAYOUT_VERTEX_COMPUTE (for the VS/eot SHARED word).
const SHARED_LAYOUT_VERTEX_COMPUTE: u6 = 0x24;
/// AGX_SHARED_LAYOUT_32X32 (for the FS/eot tilebuffer SHARED word).
const SHARED_LAYOUT_32X32: u6 = 0x2f;

/// Emit a USC SHARED word (4 bytes) for the vertex/compute layout (no shared
/// memory used) - mirrors compute.zig's SHARED word. Returns bytes written.
fn uscSharedVertexCompute(buf: []u8) usize {
    var w = BitWriter{};
    w.set(0, 8, USC_TAG_SHARED);
    w.set(10, 6, SHARED_LAYOUT_VERTEX_COMPUTE);
    w.set(24, 8, toGroups(65536, 256, 8)); // 65536 -> encodes 0 (none)
    return w.emit(buf, 4);
}

/// Emit the USC SHARED word (4 bytes) describing the 32x32 RGBA8 1-sample
/// tilebuffer (agx_tilebuffer_pack_usc): uses_shared_memory=1, layout=32x32,
/// sample_count=log2(1)=0, sample_stride_in_8B = SAMPLE_SIZE_B/8 = 1,
/// bytes_per_threadgroup = SAMPLE_SIZE_B*1*1024 = 8192 -> groups(256) = 32.
fn uscSharedTilebuffer(buf: []u8) usize {
    var w = BitWriter{};
    w.set(0, 8, USC_TAG_SHARED);
    w.set(8, 1, 1); // uses shared memory
    w.set(10, 6, SHARED_LAYOUT_32X32);
    w.set(16, 2, 0); // sample count log2(1)
    w.set(20, 4, SAMPLE_SIZE_B / 8); // sample stride in 8-byte units = 1
    w.set(24, 8, (8 * 1 * 1024) / 256); // bytes/threadgroup groups(256) = 32
    return w.emit(buf, 4);
}

/// Emit a USC SHADER word (6 bytes). genxml "USC Shader": Tag@0(8),
/// Loads varyings@8(1), Unk1@9(1), Unk2@10(6), Code@16(32). Mesa sets Unk2 =
/// fragment ? 2 : 3 (agx_linker.c:185; agx_bg_eot/compute use 3); loads_varyings
/// = (fragment && nr_cf_bindings>0). For our solid-color FS there are no varyings,
/// so loads_varyings stays 0, but Unk2 MUST be 2 for the fragment-pipeline shader
/// (the M1 FS dump shows "Unk 2: 2"). Returns bytes written.
fn uscShader(buf: []u8, code_off: u32, unk2: u6, loads_varyings: bool) usize {
    var w = BitWriter{};
    w.set(0, 8, USC_TAG_SHADER);
    if (loads_varyings) w.set(8, 1, 1);
    w.set(10, 6, unk2);
    w.set(16, 32, code_off);
    return w.emit(buf, 6);
}

/// Emit a USC REGISTERS word (4 bytes). genxml "USC Registers": Tag@0(8),
/// Register count@8(5) groups(8), Unk1@13(1), Spill@18(4), Unk4@24(8). Mesa's
/// linker (agx_linker.c:191) sets Unk1 = fragment and Unk4 = 1 for VS/FS; the
/// bg/eot compute-style path (agx_bg_eot.c) leaves both default (Unk4=0). The M1
/// dump confirms: VS Unk1=false/Unk4=0x1, FS Unk1=true/Unk4=0x1, bg/eot
/// Unk1=false/Unk4=0x0.
fn uscRegisters(buf: []u8, gpr_count: u32, unk1: bool, unk4: u8) usize {
    var w = BitWriter{};
    w.set(0, 8, USC_TAG_REGISTERS);
    w.set(8, 5, toGroups(gpr_count, 8, 5));
    if (unk1) w.set(13, 1, 1);
    w.set(24, 8, unk4);
    return w.emit(buf, 4);
}

/// Emit a USC FRAGMENT PROPERTIES word (4 bytes) - MANDATORY for any
/// fragment-pipeline shader (agx_state.c:2968 always pushes it for
/// MESA_SHADER_FRAGMENT). genxml "USC Fragment Properties": Tag@0(8),
/// early_z_testing@8(1), unk_2@9(1), unconditional_discard_1@10, _2@11,
/// unk_3@12(4), unk_4@16(8), unk_5@24(8). Mesa (agx_linker.c:200) fills
/// early_z_testing=!writes_sample_mask (true for a normal FS that does not write
/// the sample mask), unk_2=true, unk_3=0xf, unk_4=0x2, unk_5=0x0 - exactly the M1
/// FS dump (lines 350-357). WITHOUT this word the fragment stage has no
/// properties programmed and stalls = the draw TIMEOUT. Returns bytes written.
fn uscFragmentProperties(buf: []u8) usize {
    var w = BitWriter{};
    w.set(0, 8, USC_TAG_FRAGMENT_PROPERTIES);
    w.set(8, 1, 1); // early_z_testing = !writes_sample_mask = true
    w.set(9, 1, 1); // unk_2 = true
    w.set(12, 4, 0xf); // unk_3 = 0xf
    w.set(16, 8, 0x2); // unk_4 = 0x2
    w.set(24, 8, 0x0); // unk_5 = 0x0
    return w.emit(buf, 4);
}

/// Emit a USC NO-PRESHADER word (2 bytes).
fn uscNoPreshader(buf: []u8) usize {
    var w = BitWriter{};
    w.set(0, 8, USC_TAG_NO_PRESHADER);
    return w.emit(buf, 2);
}

/// Emit a USC TEXTURE word (8 bytes): bind `count` texture/PBE descriptors
/// starting at index `start`, from the descriptor at GPU VA `buffer` (encoded
/// shr(3) per genxml). Used by the eot store to bind the PBE.
fn uscTexture(buf: []u8, start: u32, count: u32, buffer: u64) usize {
    var w = BitWriter{};
    w.set(0, 8, USC_TAG_TEXTURE);
    w.set(8, 8, start);
    w.set(20, 7, count);
    w.set(27, 36, buffer >> 3);
    return w.emit(buf, 8);
}

/// Emit a USC SAMPLER word (8 bytes): bind `count` sampler descriptors starting
/// at index `start`, from the sampler heap at GPU VA `buffer` (encoded shr(3)).
/// genxml "USC Sampler" (tag 0x9d): Start@8(8), Count@20(7), Buffer@27(36) shr(3).
/// MANDATORY for any graphics shader: Mesa's sampler_count() unconditionally adds
/// the reserved txf sampler #0 (agx_state.c:2615 `return ... + 1`), so EVERY
/// graphics VS/FS USC binds 1 sampler from the heap. mesa_noddtri.txt:32-50
/// (the "Sampler state" + "Sampler" the decoder prints for both VS and FS).
fn uscSampler(buf: []u8, start: u32, count: u32, buffer: u64) usize {
    var w = BitWriter{};
    w.set(0, 8, USC_TAG_SAMPLER);
    w.set(8, 8, start);
    w.set(20, 7, count);
    w.set(27, 36, buffer >> 3);
    return w.emit(buf, 8);
}

/// Emit a USC UNIFORM word (8 bytes): load `size_halfs` halves starting at
/// uniform half `start_half` FROM the buffer at GPU VA `buffer` (encoded shr(2)).
/// Used by the bg clear to load the clear color into uniforms. SAME indirection
/// lesson as compute.zig: `buffer` is the LOAD-FROM source.
fn uscUniform(buf: []u8, start_half: u32, size_halfs: u32, buffer: u64) usize {
    var w = BitWriter{};
    w.set(0, 8, USC_TAG_UNIFORM);
    w.set(8, 8, start_half);
    w.set(20, 6, toGroups(size_halfs, 1, 6));
    w.set(26, 38, buffer >> 2);
    return w.emit(buf, 8);
}

/// Number of fixed-point LOD units per 1.0 (genxml `lod` type: value * 64).
const LOD_FRAC: u32 = 64;

/// Build the reserved txf SAMPLER descriptor (8 bytes) into `buf`. This is the
/// sampler Mesa reserves at index 0 for txf and binds in EVERY graphics USC
/// (agx_helpers.h agx_pack_txf_sampler). genxml "Sampler" (8B): Min LOD@0(10),
/// Max LOD@10(10) [lod = value*64, clamp 0..0x380=14.0], Max aniso@20(3) log2,
/// Magnify@23(2)/Minify@25(2) Filter, Mip filter@27(2), Wrap S/T/R@29/32/35(3),
/// Pixel coords@38, Compare func@39(3), Compare enable@42, Border@55(2),
/// Seamful@57. The decoded values match mesa_noddtri.txt:38-49 exactly: Min LOD 0,
/// Max LOD 14, aniso 1, Nearest mag/min/mip, Clamp-to-border S/T/R, Compare
/// Lequal, Border Transparent black. Bytes 00 00 0E 68 1B 00 00 00.
pub fn buildTxfSampler(buf: []u8) usize {
    var w = BitWriter{};
    w.set(0, 10, 0); // Minimum LOD = 0.0
    w.set(10, 10, 14 * LOD_FRAC); // Maximum LOD = 14.0 (0x380)
    w.set(20, 3, 0); // Maximum anisotropy = log2(1) = 0
    w.set(23, 2, 0); // Magnify = Nearest
    w.set(25, 2, 0); // Minify  = Nearest
    w.set(27, 2, 1); // Mip filter = Nearest
    w.set(29, 3, 3); // Wrap S = Clamp to border
    w.set(32, 3, 3); // Wrap T = Clamp to border
    w.set(35, 3, 3); // Wrap R = Clamp to border
    w.set(39, 3, 0); // Compare func = Lequal
    w.set(55, 2, 0); // Border colour = Transparent black
    return w.emit(buf, 8);
}

/// Build the VERTEX-shader USC pipeline block. The MANDATORY graphics root
/// descriptor Mesa emits for every graphics shader (agx_build_pipeline,
/// agx_state.c:2873) comes FIRST: the reserved txf SAMPLER (Mesa unconditionally
/// reserves sampler #0), then SHARED + SHADER + REGISTERS + NO_PRESHADER. The VS
/// reads no uniforms/textures itself (the fullscreen-triangle is vertex_id-
/// indexed, no vertex buffer), but the sampler binding is the mandatory part of
/// the graphics register-file setup. `sampler_heap_va` is the 1-entry sampler
/// heap (buildTxfSampler). Returns bytes written. mesa_noddtri.txt:30-102.
pub fn buildVsUsc(buf: []u8, code_off: u32, gpr_count: u32, sampler_heap_va: u64) usize {
    var n: usize = 0;
    // MANDATORY: the reserved txf sampler (1 entry) - the graphics root descriptor.
    n += uscSampler(buf[n..], 0, 1, sampler_heap_va);
    n += uscSharedVertexCompute(buf[n..]);
    // VS is non-fragment: Unk2=3, no loads_varyings, REGISTERS Unk1=false, Unk4=1.
    n += uscShader(buf[n..], code_off, 3, false);
    n += uscRegisters(buf[n..], gpr_count, false, 1);
    n += uscNoPreshader(buf[n..]);
    return n;
}

/// Build the FRAGMENT-shader USC pipeline block (SHARED[tilebuffer] + SHADER +
/// REGISTERS + FRAGMENT_PROPERTIES + NO_PRESHADER), returns bytes written. The FS
/// writes the tilebuffer so it carries the 32x32 tilebuffer SHARED word.
///
/// FIX (diff vs Mesa): the FS USC was MISSING the mandatory FRAGMENT_PROPERTIES
/// word and used the wrong SHADER Unk2 / REGISTERS flags. Mesa
/// (agx_state.c:2956-2968 + agx_linker.c:183-205) emits, for a fragment shader:
/// SHARED, SHADER(Unk2=2, loads_varyings=nr_bindings>0), REGISTERS(Unk1=true,
/// Unk4=1), FRAGMENT_PROPERTIES(early_z, unk2=1, unk3=0xf, unk4=2), then
/// NO_PRESHADER. Our solid-color FS reads NO varyings, so loads_varyings=false and
/// there are zero coefficient bindings (Output Select=0, no CF binding block - see
/// buildPpp); but FRAGMENT_PROPERTIES is unconditional for ANY fragment shader and
/// is the prime cause of the draw TIMEOUT (the fragment stage stalls without it).
pub fn buildFsUsc(buf: []u8, code_off: u32, gpr_count: u32, sampler_heap_va: u64, color_va: u64) usize {
    var n: usize = 0;
    // MANDATORY graphics root descriptor (Mesa agx_build_pipeline, emitted FIRST):
    //   (1) the reserved txf SAMPLER (sampler #0, 1 entry from the heap).
    //   (2) the color UNIFORM: 4 fp16 halves at uniform half 16 (= regs u8,u9),
    //       DMA'd from FS_COLOR_VA. The FS reads it via `mov r4,u8 / mov r5,u9`.
    //       This is the noddtri color source (mesa_noddtri.txt:265-274); it keeps
    //       the declared uniform_register_count, the USC Uniform word, and the
    //       shader's uniform reads all in lockstep (Mesa's invariant).
    n += uscSampler(buf[n..], 0, 1, sampler_heap_va);
    n += uscUniform(buf[n..], 16, 4, color_va); // halves 16..19 -> regs u8,u9
    n += uscSharedTilebuffer(buf[n..]);
    n += uscShader(buf[n..], code_off, 2, false); // fragment: Unk2=2, no varyings
    n += uscRegisters(buf[n..], gpr_count, true, 1); // fragment: Unk1=true, Unk4=1
    n += uscFragmentProperties(buf[n..]); // MANDATORY for a fragment shader
    n += uscNoPreshader(buf[n..]);
    return n;
}

/// Build the BACKGROUND (clear) USC block: SHARED[tile] + SHADER + REGISTERS +
/// NO_PRESHADER. Same shape as the FRAGMENT USC.
///
/// FIX (R4 - the bg-clear was IGNORED, R2-fp16 and R3-fp32 gave IDENTICAL
/// readback 0xFF261B1A): the previous version emitted a USC UNIFORM word binding
/// the clear-color buffer (Mesa's `agx_usc_uniform(b, 4, 8, clear_color)`), BUT
/// our clear shader BAKES the color and never reads a uniform, AND the bg Counts
/// (rsrc_spec) `uniform_register_count` was 0. That is an INCONSISTENT pipeline:
/// the USC UNIFORM word tells the hardware to DMA 8 uniform halves into uniform
/// registers starting at half 4 before the program runs, while the Counts says
/// the program uses 0 uniform registers and the program never reads them. Mesa
/// keeps these in lockstep - if a clear binds the uniform, its Counts
/// `uniform_register_count = shader->info.push_count` is non-zero AND the compiled
/// clear shader reads those uniform registers (agx_bg_eot.c build_background_op:
/// `nir_load_preamble(b, nr, 32, 4 + rt*8)`; the Counts in agx_state.c:3239
/// `cfg.uniform_register_count = shader->info.push_count`). Binding a uniform the
/// program doesn't consume, with a zero register count, leaves the uniform-load
/// stage mis-set for the fragment-like background dispatch (it does not affect the
/// COMPUTE end-of-tile store, which is exactly why the eot kept landing alpha
/// while the bg-clear color was dropped). Dropping the UNIFORM word makes the
/// USC + Counts + (baked) shader all agree on "no uniforms", matching Mesa's
/// invariant for a self-contained program. Returns bytes written.
pub fn buildBgUsc(buf: []u8, code_off: u32, gpr_count: u32) usize {
    var n: usize = 0;
    // The bg clear is a COMPUTE-style tile-control program (M1 dump "Load
    // pipeline": SHARED 32x32, shader Unk2=0, REGISTERS Unk4=0, NO Fragment
    // Properties). NOT a fragment-pipeline shader, so it gets no Fragment
    // Properties word - which is exactly why the clear path works without it.
    n += uscSharedTilebuffer(buf[n..]);
    n += uscShader(buf[n..], code_off, 0, false);
    n += uscRegisters(buf[n..], gpr_count, false, 0);
    n += uscNoPreshader(buf[n..]);
    return n;
}

/// Build the END-OF-TILE (store) USC block: TEXTURE[PBE] + SHARED[tile] + SHADER
/// + REGISTERS + NO_PRESHADER. The TEXTURE word binds the PBE descriptor at index
/// 0 so the store finds the color attachment. Returns bytes written.
pub fn buildEotUsc(buf: []u8, code_off: u32, gpr_count: u32, pbe_va: u64) usize {
    var n: usize = 0;
    // The eot store is also a COMPUTE-style program (M1 "Store pipeline": shader
    // Unk2=0, REGISTERS Unk4=0, no Fragment Properties).
    n += uscTexture(buf[n..], 0, 1, pbe_va);
    n += uscSharedTilebuffer(buf[n..]);
    n += uscShader(buf[n..], code_off, 0, false);
    n += uscRegisters(buf[n..], gpr_count, false, 0);
    n += uscNoPreshader(buf[n..]);
    return n;
}

/// The packed "Counts" / rsrc_spec word for a bg/eot program. genxml "Counts"
/// (4 bytes): Unknown0@0, UniformRegCount@1 groups(64), TextureStateRegCount@4
/// groups(8), SamplerStateRegCount@9, PreshaderRegCount@12 groups(16),
/// CFBindingCount@16. `unknown_ffff` is set (0xFFFF in the high bits, field
/// "Unknown") for the bg/clear path, cleared for the eot/store path (Mesa
/// agx_build_bg_eot: `if (!store) cfg.unknown = 0xFFFF`). We pass the texture
/// count (1 for the eot PBE, 0 for the bg).
pub fn buildCounts(nr_tex: u32, set_unknown: bool) u32 {
    var w = BitWriter{};
    w.set(4, 5, toGroups(nr_tex, 8, 5));
    if (set_unknown) w.set(16, 16, 0xFFFF); // the "Unknown" bg-only field
    return w.words[0];
}

// ===========================================================================
// The PBE (image / pixel-back-end) descriptor. Built for a LINEAR RGBA8 64x64
// store target. genxml "PBE" (32 bytes); for the linear single-layer case:
//   Dimension@0 = 2 (2D); Layout@4 = 0 (Linear); Channels@6 = 0x28 (R8G8B8A8);
//   Type@13 = 0 (Unorm); Swizzle R/G/B/A @16/18/20/22 = 0/1/2/3 (identity);
//   Width@24 minus(1) = 63; Height@38 minus(1) = 63; Samples@56 = 0 (1 sample);
//   Mode@60 = 0; Buffer@64 shr(4) = COLOR_VA>>4; Level@100 = 0;
//   Stride@104 (hex, linear) = linear_stride_B - 4; Extended@127 = 0.
// The Stride field (bits 104..125) aliases Levels/Layers - for linear we write
// Stride there (Mesa agx_batch_upload_pbe linear branch).
//
// UNVERIFIED - M1 iteration candidate (MEDIUM): the exact linear PBE layout (the
// Stride union, the "-4" offset, whether linear needs Depth(linear)/Layer-stride
// fields). If the eot store writes to a wrong address/garbage, the PBE is the
// suspect after the eot shader itself.
// ===========================================================================

const PBE_LEN: usize = 32;

/// Build the PBE descriptor for a linear RGBA8 W x H color attachment at
/// `color_va`, stride `stride_b`. Returns bytes written (32).
///
/// SIZE-DEPENDENT fields, each grounded in Mesa src/gallium/drivers/asahi/
/// agx_state.c agx_pack(out, PBE, cfg) (the render-target / linear branch, lines
/// 1218-1239):
///   cfg.width  = view->resource->width0   -> Width  minus(1) = W-1
///   cfg.height = view->resource->height0  -> Height minus(1) = H-1
///   cfg.stride = ail_get_linear_stride_B(layout) - 4. For a LINEAR RGBA8 image
///     ail_get_linear_stride_B = W * blocksize = W*4, so Stride = W*4 - 4 =
///     stride_b - 4 (the genxml "-4" convention). The caller passes stride_b =
///     W*4 (COLOR_STRIDE for the self-test; width*4 for the HAL path).
pub fn buildPbe(buf: []u8, color_va: u64, stride_b: u32, width: u32, height: u32) usize {
    var w = BitWriter{};
    w.set(0, 4, 2); // Dimension = 2D
    w.set(4, 2, 0); // Layout = Linear
    w.set(6, 7, 0x28); // Channels = R8G8B8A8
    w.set(13, 3, 0); // Type = Unorm
    w.set(16, 2, 0); // Swizzle R = R
    w.set(18, 2, 1); // Swizzle G = G
    w.set(20, 2, 2); // Swizzle B = B
    w.set(22, 2, 3); // Swizzle A = A
    w.set(24, 14, width - 1); // Width minus(1)   (Mesa cfg.width = width0)
    w.set(38, 14, height - 1); // Height minus(1)  (Mesa cfg.height = height0)
    w.set(56, 1, 0); // Samples = 1
    w.set(60, 2, 0); // Mode = Normal
    w.set(64, 36, color_va >> 4); // Buffer shr(4)
    w.set(100, 4, 0); // Level = 0
    w.set(104, 21, stride_b - 4); // Stride (linear) = W*4 - 4 (Mesa "-4")
    return w.emit(buf, PBE_LEN);
}

// ===========================================================================
// The VDM control stream. A flat LE byte stream of tagged blocks (Block Type in
// bits 29..31 of the first word, same scheme as CDM):
//   VDM Barrier (4B)        - usc_cache_inval at batch start
//   PPP State Update (8B)   - point at the PPP word block
//   VDM State Update (4B header) + present sub-words:
//     Vertex Shader Word 0 (4B) - VS register counts (all 0 here)
//     Vertex Shader Word 1 (4B) - VS pipeline = VS USC offset >> 6
//     Vertex Outputs (4B)       - output count = UVS size in words = 4 (position)
//     Vertex Unknown (4B)       - flat shading control
//     + 4B zero pad (Mesa keeps the stream 8-byte aligned)
//   Index List (4B header) + Count(4B)=3 + Instances(4B)=1 + Start(4B)=0
//   VDM Stream Terminate (4B; the struct is 32B but only word0 carries Block
//     Type - we emit a single 4B terminate word, padded if needed)
// Block types (genxml "VDM Block Type"): Barrier=1, PPP State=0, VDM State=2,
// Index List=3, Stream Terminate=6. Primitive Triangles=6.
//
// UNVERIFIED - M1 iteration candidate (HIGH): the Vertex Outputs count, the VDM
// State present-bit set, the 8-byte alignment padding, and the terminate length.
// ===========================================================================

const VDM_BARRIER: u3 = 1;
const VDM_PPP_STATE: u3 = 0;
const VDM_STATE: u3 = 2;
const VDM_INDEX_LIST: u3 = 3;
const VDM_TERMINATE: u3 = 6;
const PRIMITIVE_TRIANGLES: u8 = 6;

/// Build the VDM control stream into `buf`. `vs_usc_off` is the VS USC block's
/// offset from USC_EXEC_BASE (the pipeline is vs_usc_off>>6); `ppp_va` is the GPU
/// VA of the PPP word block; `ppp_words` is its length in 32-bit words. `init_ppp_va`
/// / `init_ppp_words` point at the MANDATORY per-control-stream init PPP block
/// (buildInitPpp - the W-clamp setup; see buildInitPpp for why it is the draw-hang
/// fix). Returns bytes written.
pub fn buildVdmStream(buf: []u8, vs_usc_off: u32, ppp_va: u64, ppp_words: u32, init_ppp_va: u64, init_ppp_words: u32, vps_va: u64, vps_words: u32) usize {
    var n: usize = 0;

    // VDM Barrier (4B): usc_cache_inval (bit3) + Block Type Barrier.
    {
        var w = BitWriter{};
        w.set(3, 1, 1);
        w.set(29, 3, VDM_BARRIER);
        n += w.emit(buf[n..], 4);
    }

    // INIT PPP State Update (8B): points at the mandatory init PPP word block
    // (W_CLAMP=1e-10 + zero words). This is the Honeykrisp hk_cs_init_graphics
    // setup - WITHOUT it, ppp_ctrl's enable_w_clamp clamps W against an undefined
    // register and primitive setup hangs (TIMEOUT, no fault). Emitted right after
    // the barrier, before the VDM state + per-draw PPP record, exactly like hk
    // (hk_cs_init_graphics runs once per control stream, then the draw state).
    {
        var w = BitWriter{};
        w.set(0, 8, @truncate((init_ppp_va >> 32) & 0xFF)); // pointer hi
        w.set(8, 8, init_ppp_words); // size in words
        w.set(29, 3, VDM_PPP_STATE);
        w.set(32, 32, @truncate(init_ppp_va & 0xFFFF_FFFF)); // pointer lo
        n += w.emit(buf[n..], 8);
    }

    // VDM State Update (4B header): present bits for VS word0/word1, vertex
    // outputs, vertex unknown.
    //
    // ORDER FIX (diff vs Mesa): Mesa's agx_encode_state emits the VDM State Update
    // FIRST, then the per-draw PPP State Update via agx_ppp_fini at the END of the
    // function. We previously emitted the per-draw PPP BEFORE the VDM state; now we
    // match Mesa: Barrier, init-PPP, VDM-State, per-draw-PPP, Index List, Terminate.
    {
        var w = BitWriter{};
        w.set(1, 1, 1); // vertex shader word 0 present
        w.set(2, 1, 1); // vertex shader word 1 present
        w.set(3, 1, 1); // vertex outputs present
        w.set(5, 1, 1); // vertex unknown present
        w.set(29, 3, VDM_STATE);
        n += w.emit(buf[n..], 4);
    }
    // VDM State Vertex Shader Word 0 (4B): the graphics register-file declaration.
    // genxml: Uniform reg count@1(3) groups(64), Texture state@4(5) groups(8),
    // Sampler state@9(3) Sampler-states, Preshader@12(4) groups(16). Mesa fills
    // these from the COMPILED VS (agx_state.c:3477) - and sampler_state_register_
    // count is ALWAYS >= 1 because sampler_count() unconditionally reserves the
    // txf sampler #0 (agx_state.c:2615 `+ 1`). Our vertex_id-indexed VS reads no
    // uniforms/textures and has no preamble, so those stay 0; but the sampler
    // declaration is MANDATORY and must match the USC Sampler word we now bind
    // (4 compact = enum 1 = the M1 dump's "Sampler state register count: 4
    // compact", mesa_noddtri.txt:28). A graphics shader-word with a 0 sampler
    // count contradicts the bound sampler heap and leaves the register file
    // half-set = the draw stall.
    {
        var w = BitWriter{};
        // Match Mesa noddtri's VS Word 0 EXACTLY (mesa_noddtri.txt:24-29):
        // Uniform 64 (1 group), Texture 8 (1 group), Sampler 4 compact (1),
        // Preshader 16 (1 group). A stored value of 0 in a groups() field decodes
        // to "all" (the max), NOT zero, so a graphics shader-word must store the
        // 1-group encoding to declare a bounded register file. The sampler field
        // is the unconditionally-mandatory part (reserved txf sampler #0); the
        // others mirror Mesa's proven config-matched reference so the register
        // file is declared exactly as the known-good driver declares it.
        w.set(1, 3, toGroups(64, 64, 3)); // uniform register count = 64
        w.set(4, 5, toGroups(8, 8, 5)); // texture state register count = 8
        w.set(9, 3, SAMPLER_STATES_4_COMPACT); // sampler state count = 4 compact (1)
        w.set(12, 4, toGroups(16, 16, 4)); // preshader register count = 16
        n += w.emit(buf[n..], 4);
    }
    // VDM State Vertex Shader Word 1 (4B): Pipeline@6 shr(6) = vs_usc_off>>6.
    {
        var w = BitWriter{};
        w.set(6, 26, vs_usc_off >> 6);
        n += w.emit(buf[n..], 4);
    }
    // VDM State Vertex Outputs (4B): Output count 1@0(8), Output count 2@8(8).
    // Position-only UVS = 4 words.
    {
        var w = BitWriter{};
        w.set(0, 8, 4);
        w.set(8, 8, 4);
        n += w.emit(buf[n..], 4);
    }
    // VDM State Vertex Unknown (4B): flat shading control = 2 (AGX_VDM_VERTEX_2,
    // when flatshade_first is false). The rest 0.
    {
        var w = BitWriter{};
        w.set(0, 2, 2);
        n += w.emit(buf[n..], 4);
    }
    // 4B zero pad to keep the stream 8-byte aligned (Mesa memsets 4 here).
    {
        var w = BitWriter{};
        n += w.emit(buf[n..], 4);
    }

    // VIEWPORT/SCISSOR PPP State Update (8B): Mesa emits this block (depth-bias/
    // scissor + region-clip + viewport-control + viewport) right after the VDM
    // State and before the per-draw fragment PPP record (agx_encode_state ->
    // agx_upload_viewport_scissor -> agx_ppp_fini). It points at the
    // buildViewportScissorPpp record.
    {
        var w = BitWriter{};
        w.set(0, 8, @truncate((vps_va >> 32) & 0xFF)); // pointer hi
        w.set(8, 8, vps_words); // size in words
        w.set(29, 3, VDM_PPP_STATE);
        w.set(32, 32, @truncate(vps_va & 0xFFFF_FFFF)); // pointer lo
        n += w.emit(buf[n..], 8);
    }

    // PER-DRAW PPP State Update (8B), emitted AFTER the VDM State + viewport/scissor
    // (matching Mesa's agx_encode_state -> agx_ppp_fini order). word0 =
    // Pointer(hi)@0(8) | Size(words)@8(8) | Block Type@29; word1 = Pointer(lo).
    {
        var w = BitWriter{};
        w.set(0, 8, @truncate((ppp_va >> 32) & 0xFF)); // pointer hi
        w.set(8, 8, ppp_words); // size in words
        w.set(29, 3, VDM_PPP_STATE);
        w.set(32, 32, @truncate(ppp_va & 0xFFFF_FFFF)); // pointer lo
        n += w.emit(buf[n..], 8);
    }

    // Index List (4B header): Primitive@8 = Triangles(6); Index count present
    // (bit22), Instance count present (bit23), Start present (bit24); non-indexed
    // (no index buffer). Block Type Index List.
    {
        var w = BitWriter{};
        w.set(8, 8, PRIMITIVE_TRIANGLES);
        w.set(22, 1, 1); // index count present
        w.set(23, 1, 1); // instance count present
        w.set(24, 1, 1); // start present
        w.set(29, 3, VDM_INDEX_LIST);
        n += w.emit(buf[n..], 4);
    }
    // Index List Count (4B) = 3 vertices.
    {
        var w = BitWriter{};
        w.set(0, 32, 3);
        n += w.emit(buf[n..], 4);
    }
    // Index List Instances (4B) = 1.
    {
        var w = BitWriter{};
        w.set(0, 32, 1);
        n += w.emit(buf[n..], 4);
    }
    // Index List Start (4B) = 0 (base vertex).
    {
        var w = BitWriter{};
        w.set(0, 32, 0);
        n += w.emit(buf[n..], 4);
    }

    // VDM Stream Terminate (4B word carrying Block Type; the genxml struct is 32B
    // but only the first word is meaningful - we emit one word).
    {
        var w = BitWriter{};
        w.set(29, 3, VDM_TERMINATE);
        n += w.emit(buf[n..], 4);
    }

    return n;
}

/// Build a CLEAR-ONLY VDM control stream: a Barrier followed immediately by a
/// Stream Terminate, with NO PPP/VDM-state/index-list draw. This is the
/// discriminator stream: the render pass still runs the per-tile background clear
/// + end-of-tile store (driven by the bg/eot programs in drm_asahi_cmd_render +
/// PROCESS_EMPTY_TILES, which are INDEPENDENT of the VDM draw), but issues NO
/// geometry. It isolates the tile-renderer INFRASTRUCTURE (bg clear + eot store +
/// tile/render config) from the DRAW (VDM/PPP/VS/FS). If the clear-only readback
/// is kClearColor, the infrastructure works and the hang/miss is in the draw; if
/// it still times out / stays kSentinel, the bg/eot/config is the culprit.
///
/// A Barrier + Terminate is the minimal well-formed VDM stream (the command
/// processor reads the Barrier, advances by its 4-byte length, reads Terminate
/// and stops). Mesa pads the cmdbuf tail generously because the VDM overreads
/// (~0x800 bytes) past the terminate; the terminate here sits at the very start
/// of a 16 KiB BO whose tail is zero-filled, so the overread stays in mapped,
/// zeroed memory. Returns bytes written.
pub fn buildVdmStreamClear(buf: []u8) usize {
    var n: usize = 0;
    // VDM Barrier (4B): usc_cache_inval (bit3) + Block Type Barrier.
    {
        var w = BitWriter{};
        w.set(3, 1, 1);
        w.set(29, 3, VDM_BARRIER);
        n += w.emit(buf[n..], 4);
    }
    // VDM Stream Terminate (4B).
    {
        var w = BitWriter{};
        w.set(29, 3, VDM_TERMINATE);
        n += w.emit(buf[n..], 4);
    }
    return n;
}

// ===========================================================================
// The PPP (per-pipeline-pass) word block. A PPP_HEADER (4B) of present bits
// followed by exactly the present sub-words IN HEADER-BIT ORDER. The on-chip PPP
// parser advances through the record word-by-word in present-bit order (the
// decode.c agxdecode_record / agx_ppp model): if a present bit is set the parser
// consumes that word, if not it skips it. THE BLOCK IS ONLY VALID IF THE EMITTED
// WORDS EXACTLY MATCH THE SET PRESENT BITS, IN BIT ORDER.
//
// THE OLD BLOCK DESYNCED THE PPP PARSER (a prime cause of the M1 draw TIMEOUT,
// no fault address): it emitted Fragment Face _2 words for front + back and a
// Viewport Control word WITHOUT setting their present bits, and used the wrong
// viewport-count encoding. Every word after the first unflagged extra was then
// misread -> a garbage PPP record -> the ISP/PPP processor hangs. This rewrite
// makes the header present-bit set EXACTLY match Mesa's agx_encode_state combined
// dirty set (agx_state.c L3536-3705) and emits the words in that exact order.
//
// We also DISABLE the scissor (Fragment Control scissor_enable=0) and OMIT the
// Depth-bias/Scissor + Region-clip words: those index into the scissor /
// depth-bias ARRAYS (isp_scissor_base / isp_dbias_base), which we do not upload
// (they are 0 in drm_asahi_cmd_render). Enabling the scissor without those arrays
// would make the ISP read a null scissor base. Clipping to the framebuffer is
// handled by the viewport instead. The Viewport word IS emitted (it is what maps
// clip space to the 64x64 framebuffer), so we set the Viewport present bit + the
// matching Viewport Control + Viewport words.
//
// Field layouts byte-confirmed against genxml cmdbuf.xml:
//   Fragment control (4B): scissor_enable@16, depth_bias_enable@17,
//     stencil_test_enable@18, ..., pass_type@29(3).
//   Fragment face (4B): stencil_ref@0, line_width@8, polygon_mode@18(2),
//     disable_depth_write@21, depth_function@24(3) [ZS Func: Always=7].
//   Fragment face 2 (4B): disable_depth_write@21, depth_function@24(3),
//     object_type@28(4) [Triangle=0].
//   Output Select (4B): all clip/varying-enable bits 0 (no varyings).
//   Varying Counts (4B): smooth@0(8), flat@8(8), linear@16(8) = 0.
//   Cull (4B): cull_front@0, cull_back@1, front_face_ccw@16 (none here).
//   Cull 2 (4B): clamp_w@5.
//   Fragment Shader Word 0 (4B): reg counts 0, cf_binding_count@16 = 0.
//   Fragment Shader Word 1 (4B): pipeline@6 shr(6) = fs_usc_off>>6.
//   Fragment Shader Word 2 (4B): cf_bindings@2 shr(2) = 0 (no varyings).
//   Fragment Shader Word 3 (4B): unknown@0(4) = 0 (<4 textures).
//   Output Size (4B): count@0(32) = UVS size = 4.
//   Viewport control (4B): zero. Viewport (24B): translate_x,scale_x,
//     translate_y,scale_y,translate_z,scale_z (f32).
// ===========================================================================

// PPP_HEADER present bit indices (genxml "PPP Header" field starts).
const PPP_FRAGMENT_CONTROL: u32 = 0;
const PPP_FRAGMENT_CONTROL_2: u32 = 1;
const PPP_FRAGMENT_FRONT_FACE: u32 = 2;
const PPP_FRAGMENT_FRONT_FACE_2: u32 = 3;
const PPP_FRAGMENT_BACK_FACE: u32 = 5;
const PPP_FRAGMENT_BACK_FACE_2: u32 = 6;
const PPP_DEPTH_BIAS_SCISSOR: u32 = 8;
const PPP_REGION_CLIP: u32 = 10;
const PPP_VIEWPORT: u32 = 11;
const PPP_VIEWPORT_COUNT: u32 = 12; // 4-bit field, minus(1) encoded
const PPP_OUTPUT_SELECT: u32 = 17;
const PPP_VARYING_COUNTS_32: u32 = 18;
const PPP_VARYING_COUNTS_16: u32 = 19;
const PPP_CULL: u32 = 21;
const PPP_CULL_2: u32 = 22;
const PPP_FRAGMENT_SHADER: u32 = 23;
const PPP_OUTPUT_SIZE: u32 = 27;

/// ZS Func "Always" (depth test always passes), genxml ZS Func enum.
const ZS_FUNC_ALWAYS: u64 = 7;

// PPP_HEADER present bits for the MANDATORY per-control-stream INIT PPP block
// (the Honeykrisp render-path setup that gallium's batch init also implies).
const PPP_W_CLAMP: u32 = 16;
const PPP_OCCLUSION_QUERY_2: u32 = 25;
const PPP_OUTPUT_UNKNOWN: u32 = 26;
const PPP_VARYING_WORD_2: u32 = 28;

/// THE FIX (R5 - the draw-hang-no-fault). Build the MANDATORY initial PPP State
/// record that the kernel-UAPI render path (drm_asahi_cmd_render.vdm_ctrl_stream_
/// base) requires once per control stream, BEFORE any draw. This is byte-for-byte
/// the block Honeykrisp emits in hk_cs_init_graphics (src/asahi/vulkan/
/// hk_cmd_buffer.c:797-823): a VDM Barrier (already emitted by the stream) followed
/// by a PPP State Update whose header sets w_clamp + occlusion_query_2 +
/// output_unknown + varying_word_2, pushing:
///   W_CLAMP                 = 1e-10  (genxml "W Clamp", 4B, a single f32)
///   FRAGMENT_OCCLUSION_QUERY_2 = 0   (4B)
///   OUTPUT_UNKNOWN          = 0      (4B)
///   VARYING_2               = 0      (8B)
/// = header(4) + 4 + 4 + 4 + 8 = 24 bytes, 6 words.
///
/// WHY THIS IS THE HANG. drm_asahi_cmd_render.ppp_ctrl = 0x202 sets
/// enable_w_clamp (CR_PPP_CONTROL.enable_w_clamp, hk_queue.c:114). With W-clamp
/// ENABLED but no W_CLAMP value ever programmed, the W-clamp register is
/// undefined, so the geometry/primitive-setup stage clamps W against garbage and
/// can stall forever waiting on vertex output it never accepts - a GPU TIMEOUT
/// with NO fault address (exactly the observed signature). The CLEAR-only path
/// never runs a VS / primitive setup, so it is unaffected (which is precisely why
/// the clear works while the draw hangs). The decode walk (decode.c:487
/// PPP_PRINT(w_clamp, W_CLAMP)) consumes this word in header-bit order, so the
/// block stays in sync. Returns bytes written (24).
pub fn buildInitPpp(buf: []u8) usize {
    var n: usize = 0;

    // PPP_HEADER (4B): w_clamp + occlusion_query_2 + output_unknown +
    // varying_word_2, in ascending bit order. viewport_count is irrelevant here
    // (no viewport word in this block) but Mesa sets it to 1 (minus(1) -> 0).
    {
        var w = BitWriter{};
        w.set(PPP_W_CLAMP, 1, 1);
        w.set(PPP_OCCLUSION_QUERY_2, 1, 1);
        w.set(PPP_OUTPUT_UNKNOWN, 1, 1);
        w.set(PPP_VARYING_WORD_2, 1, 1);
        n += w.emit(buf[n..], 4);
    }
    // W_CLAMP (4B): a single f32 = 1e-10 (Mesa's exact value).
    {
        var w = BitWriter{};
        w.set(0, 32, fui(1e-10));
        n += w.emit(buf[n..], 4);
    }
    // FRAGMENT_OCCLUSION_QUERY_2 (4B) = 0.
    {
        var w = BitWriter{};
        n += w.emit(buf[n..], 4);
    }
    // OUTPUT_UNKNOWN (4B) = 0.
    {
        var w = BitWriter{};
        n += w.emit(buf[n..], 4);
    }
    // VARYING_2 (8B) = 0.
    {
        var w = BitWriter{};
        n += w.emit(buf[n..], 8);
    }

    return n;
}

// ---------------------------------------------------------------------------
// Scissor + depth-bias arrays and the viewport/scissor PPP State Update.
//
// Mesa ALWAYS uploads a SCISSOR descriptor array (isp_scissor_base) and a
// DEPTH_BIAS array (isp_dbias_base) and emits a SEPARATE PPP State Update
// containing Depth-bias/Scissor + Region-clip + Viewport-control + Viewport
// (agx_upload_viewport_scissor, agx_state.c:937). The FRAGMENT_CONTROL
// scissor_enable is always true so the rasterizer clips to the viewport box, and
// the REGION_CLIP bounds the tile region the ISP processes. We previously omitted
// all of this (null bases, scissor disabled, viewport in the per-draw record) -
// the diff vs Mesa flagged it as a candidate hang/clip cause. Now we upload a
// single full-framebuffer scissor + a zero depth-bias and emit the matching PPP
// block, so isp_scissor_base/isp_dbias_base are valid and the indices resolve.
// ---------------------------------------------------------------------------

/// genxml "Scissor" (16 bytes): Max X@0(16), Min X@16(16), Max Y@32(16),
/// Min Y@48(16), Min Z@64(32 f32), Max Z@96(32 f32). One full-framebuffer
/// scissor covering [0,width] x [0,height], z [0,1]. Returns 16. (Mesa
/// agx_state.c agx_upload_viewport_scissor: the scissor box is the full
/// framebuffer maxx/maxy = width/height.)
pub fn buildScissor(buf: []u8, width: u32, height: u32) usize {
    var w = BitWriter{};
    w.set(0, 16, width); // Max X
    w.set(16, 16, 0); // Min X
    w.set(32, 16, height); // Max Y
    w.set(48, 16, 0); // Min Y
    w.set(64, 32, fui(0.0)); // Min Z
    w.set(96, 32, fui(1.0)); // Max Z
    return w.emit(buf, 16);
}

/// genxml "Depth bias" (12 bytes): Depth bias@0(f32), Slope scale@32(f32),
/// Clamp@64(f32). All zero (no depth bias). Returns 12.
pub fn buildDepthBias(buf: []u8) usize {
    var w = BitWriter{};
    // all zero
    return w.emit(buf, 12);
}

/// Build the viewport/scissor PPP State Update record (the Mesa
/// agx_upload_viewport_scissor block): header (depth_bias_scissor + region_clip +
/// viewport + viewport_count=1) then DEPTH_BIAS_SCISSOR(4B) + REGION_CLIP(8B) +
/// VIEWPORT_CONTROL(4B) + VIEWPORT(24B). The viewport maps clip [-1,1] -> [0,FB]
/// in x/y, z [0,1]. The region clip bounds the rasterizer to the tile-aligned
/// framebuffer box (min/32 .. DIV_ROUND_UP(max,32)). Returns bytes written.
pub fn buildViewportScissorPpp(buf: []u8, width: u32, height: u32) usize {
    var n: usize = 0;

    // PPP_HEADER (4B): depth_bias_scissor@8, region_clip@10, viewport@11,
    // viewport_count@12(4) = 1 (minus(1) -> 0).
    {
        var w = BitWriter{};
        w.set(PPP_DEPTH_BIAS_SCISSOR, 1, 1);
        w.set(PPP_REGION_CLIP, 1, 1);
        w.set(PPP_VIEWPORT, 1, 1);
        w.set(PPP_VIEWPORT_COUNT, 4, 0); // viewport count = 1 (minus(1) -> 0)
        n += w.emit(buf[n..], 4);
    }

    // DEPTH_BIAS_SCISSOR (4B): Scissor index@0(16) = 0, Depth bias index@16(16)=0.
    {
        var w = BitWriter{};
        n += w.emit(buf[n..], 4);
    }

    // REGION_CLIP (8B): Max X@0(9) minus(1), Min X@16(9), Enable@31, Max Y@32(9)
    // minus(1), Min Y@48(9). Tile-aligned: min/32, DIV_ROUND_UP(max,32).
    {
        var w = BitWriter{};
        // Mesa agx_state.c:1000 cfg.max_x = DIV_ROUND_UP(MAX2(maxx,1), 32); the
        // genxml "Region clip" Max X is modifier="minus(1)", so the raw field is
        // DIV_ROUND_UP(W,32) - 1 (utile 32x32). For W=64 -> 2-1=1; W=256 -> 8-1=7.
        const max_x_tiles: u32 = (width + 31) / 32; // DIV_ROUND_UP(W,32)
        const max_y_tiles: u32 = (height + 31) / 32; // DIV_ROUND_UP(H,32)
        w.set(0, 9, max_x_tiles - 1); // Max X minus(1)
        w.set(16, 9, 0); // Min X
        w.set(31, 1, 1); // Enable
        w.set(32, 9, max_y_tiles - 1); // Max Y minus(1)
        w.set(48, 9, 0); // Min Y
        n += w.emit(buf[n..], 8);
    }

    // VIEWPORT_CONTROL (4B): zero.
    {
        var w = BitWriter{};
        n += w.emit(buf[n..], 4);
    }

    // VIEWPORT (24B): translate/scale X/Y/Z. clip [-1,1] -> [0,FB]: scale = FB/2,
    // translate = FB/2; z [0,1]. (Mesa adjusts z for non-half-z; for our z in
    // [0,1] with clear depth handling we keep translate_z=0, scale_z=1.)
    {
        var w = BitWriter{};
        // clip [-1,1] -> [0,W] / [0,H]: scale = dim/2, translate = dim/2 (Mesa
        // viewport scale_x = width/2, translate_x = width/2).
        const half_w: f32 = @as(f32, @floatFromInt(width)) / 2.0;
        const half_h: f32 = @as(f32, @floatFromInt(height)) / 2.0;
        w.set(0, 32, fui_rt(half_w)); // translate x = W/2
        w.set(32, 32, fui_rt(half_w)); // scale x = W/2
        w.set(64, 32, fui_rt(half_h)); // translate y = H/2
        w.set(96, 32, fui_rt(half_h)); // scale y = H/2
        w.set(128, 32, fui(0.0)); // translate z
        w.set(160, 32, fui(1.0)); // scale z
        n += w.emit(buf[n..], 24);
    }

    return n;
}

/// Build the PPP word block into `buf`. `fs_usc_off` is the FS USC block's offset
/// from USC_EXEC_BASE (the FS pipeline is fs_usc_off>>6). Returns bytes written.
/// The viewport maps clip [-1,1] -> [0,64] in x and y, z [0,1]. The header
/// present-bit set EXACTLY matches the emitted words, in bit order, so the PPP
/// parser stays in sync (the desync fix).
pub fn buildPpp(buf: []u8, fs_usc_off: u32) usize {
    var n: usize = 0;

    // PPP_HEADER (4B): present bits for EXACTLY the words emitted below, in bit
    // order: FragControl, FragControl2, FrontFace, FrontFace2, BackFace,
    // BackFace2, Viewport(+count), OutputSelect, VaryingCounts32, VaryingCounts16,
    // Cull, Cull2, FragmentShader, OutputSize.
    {
        var w = BitWriter{};
        w.set(PPP_FRAGMENT_CONTROL, 1, 1);
        w.set(PPP_FRAGMENT_CONTROL_2, 1, 1);
        w.set(PPP_FRAGMENT_FRONT_FACE, 1, 1);
        w.set(PPP_FRAGMENT_FRONT_FACE_2, 1, 1);
        w.set(PPP_FRAGMENT_BACK_FACE, 1, 1);
        w.set(PPP_FRAGMENT_BACK_FACE_2, 1, 1);
        // Viewport/scissor/region-clip live in their OWN PPP State Update
        // (buildViewportScissorPpp), exactly like Mesa's separate
        // agx_upload_viewport_scissor block. They are NOT in this per-draw record.
        w.set(PPP_OUTPUT_SELECT, 1, 1);
        w.set(PPP_VARYING_COUNTS_32, 1, 1);
        w.set(PPP_VARYING_COUNTS_16, 1, 1);
        w.set(PPP_CULL, 1, 1);
        w.set(PPP_CULL_2, 1, 1);
        w.set(PPP_FRAGMENT_SHADER, 1, 1);
        w.set(PPP_OUTPUT_SIZE, 1, 1);
        n += w.emit(buf[n..], 4);
    }

    // FRAGMENT_CONTROL (4B): scissor ENABLED (Mesa always sets scissor_enable to
    // clip to the viewport - agx_state.c:3583 "Always enable scissoring"; we now
    // upload the scissor + depth-bias arrays so isp_scissor_base/isp_dbias_base are
    // non-null and the indices are valid). pass_type=Opaque(0), depth/stencil off.
    {
        var w = BitWriter{};
        w.set(16, 1, 1); // scissor_enable = true
        n += w.emit(buf[n..], 4);
    }
    // FRAGMENT_CONTROL_2 (4B): tag_write_disable off (we DO want pixels). 0.
    {
        var w = BitWriter{};
        n += w.emit(buf[n..], 4);
    }
    // FRAGMENT_FACE front (4B): depth function Always(7) @24, depth write
    // disabled @21 (no ZLS buffer attached).
    {
        var w = BitWriter{};
        w.set(21, 1, 1); // disable depth write
        w.set(24, 3, ZS_FUNC_ALWAYS); // depth function Always
        n += w.emit(buf[n..], 4);
    }
    // FRAGMENT_FACE_2 front (4B): depth function Always @24, disable depth write
    // @21, object type Triangle(0) @28.
    {
        var w = BitWriter{};
        w.set(21, 1, 1);
        w.set(24, 3, ZS_FUNC_ALWAYS);
        // object_type Triangle = 0 @28 -> nothing to OR.
        n += w.emit(buf[n..], 4);
    }
    // FRAGMENT_FACE back (4B) + FACE_2 back (4B): same as front.
    {
        var w = BitWriter{};
        w.set(21, 1, 1);
        w.set(24, 3, ZS_FUNC_ALWAYS);
        n += w.emit(buf[n..], 4);
    }
    {
        var w = BitWriter{};
        w.set(21, 1, 1);
        w.set(24, 3, ZS_FUNC_ALWAYS);
        n += w.emit(buf[n..], 4);
    }

    // (Viewport/scissor/region-clip are emitted in their own PPP State Update -
    // buildViewportScissorPpp - not here, matching Mesa.)

    // OUTPUT_SELECT (4B) = 0 (position-only VS, constant-color FS, no varyings).
    {
        var w = BitWriter{};
        n += w.emit(buf[n..], 4);
    }
    // VARYING_COUNTS 32 (4B) = 0, VARYING_COUNTS 16 (4B) = 0.
    {
        var w = BitWriter{};
        n += w.emit(buf[n..], 4);
    }
    {
        var w = BitWriter{};
        n += w.emit(buf[n..], 4);
    }
    // CULL (4B): no face culling, but match Mesa's default rasterizer
    // (agx_create_rs_state): depth_clip @10 = 1 (depth_clip_near), depth_clamp
    // @11 = 0, flat_shading_vertex @7 = AGX_PPP_VERTEX_2 (enum value 3, since
    // flatshade_first is false). Leaving depth_clip/clamp both 0 is an undefined
    // clip mode for primitive setup; Mesa always programs exactly one.
    {
        var w = BitWriter{};
        w.set(7, 2, 3); // flat_shading_vertex = AGX_PPP_VERTEX_2 (encoded 3)
        w.set(10, 1, 1); // depth_clip (near) enabled
        n += w.emit(buf[n..], 4);
    }
    // CULL_2 (4B): clamp_w=1 @5 (Mesa always sets it).
    {
        var w = BitWriter{};
        w.set(5, 1, 1); // clamp_w
        n += w.emit(buf[n..], 4);
    }

    // FRAGMENT_SHADER group (4 words):
    //   WORD_0 (4B): the graphics register-file declaration for the FS. genxml
    //   "Fragment Shader Word 0": Uniform reg count@1(3) groups(64), Texture
    //   state@4(5) groups(8), Sampler state@9(3) Sampler-states, Preshader@12(4)
    //   groups(16), CF binding count@16(7). Mesa fills these from the compiled FS
    //   (agx_state.c:3667). OUR FS now reads the color from a uniform (mov r4,u8 /
    //   mov r5,u9 -> uniform regs u8,u9), so uniform_register_count MUST be
    //   non-zero (we declare 1 group = 64 regs, Mesa's noddtri value, covering the
    //   color's u8,u9). sampler_state_register_count = 4 compact (1, the reserved
    //   txf sampler, MANDATORY - matches the USC Sampler word). No textures, no
    //   preshader, no CF bindings (solid color, no interpolated varyings). With
    //   all-zero counts (the old value) the FS had a bound USC Uniform + Sampler
    //   but a register file declared as empty = the fragment stage stalls = the
    //   draw TIMEOUT-no-fault. mesa_noddtri.txt:308-314.
    {
        var w = BitWriter{};
        // Match Mesa noddtri's FS Word 0 EXACTLY (mesa_noddtri.txt:308-314):
        // Uniform 64, Texture 8, Sampler 4 compact, Preshader 16, CF binding 0.
        // (A stored 0 in a groups() field decodes to "all", not zero - so we store
        // the 1-group encoding to declare a bounded register file, matching the
        // known-good driver. CF binding count stays 0: our solid-color FS reads no
        // interpolated varyings, so no coefficient bindings - agx_state.c sets it
        // from nr_cf_bindings, which is 0 here, exactly Mesa-faithful.)
        w.set(1, 3, toGroups(64, 64, 3)); // uniform register count = 64
        w.set(4, 5, toGroups(8, 8, 5)); // texture state register count = 8
        w.set(9, 3, SAMPLER_STATES_4_COMPACT); // sampler state count = 4 compact (1)
        w.set(12, 4, toGroups(16, 16, 4)); // preshader register count = 16
        // CF binding count @16(7) = 0 (no varyings).
        n += w.emit(buf[n..], 4);
    }
    //   WORD_1 (4B): Pipeline@6 shr(6) = fs_usc_off>>6.
    {
        var w = BitWriter{};
        w.set(6, 26, fs_usc_off >> 6);
        n += w.emit(buf[n..], 4);
    }
    //   WORD_2 (4B): CF bindings = 0 (no varyings).
    {
        var w = BitWriter{};
        n += w.emit(buf[n..], 4);
    }
    //   WORD_3 (4B): unknown, 0 for <4 textures.
    {
        var w = BitWriter{};
        n += w.emit(buf[n..], 4);
    }

    // OUTPUT_SIZE (4B): Count = UVS size = 4.
    {
        var w = BitWriter{};
        w.set(0, 8, 4);
        n += w.emit(buf[n..], 4);
    }

    return n;
}

// ===========================================================================
// Assemble the SUBMIT cmdbuf. A SET_FRAGMENT_ATTACHMENTS software command (header
// + 1 drm_asahi_attachment for the color buffer region) followed by the RENDER
// hardware command (header + drm_asahi_cmd_render body). The attachment is a hint
// (optional per the UAPI) but we provide it for the eot store's write region.
// ===========================================================================

/// ppp_multisamplectl for 1 sample (agx_default_sample_positions(1) = 0x88).
pub const PPP_MULTISAMPLECTL_1: u64 = 0x88;
/// ppp_ctrl (hk_queue.c): enable_w_clamp (bit1) | fixed_point_format=1 (bit9) =
/// 0x202. (A GL driver would also set bit0; we follow Honeykrisp's value.)
pub const PPP_CTRL: u32 = 0x202;

/// Build the full SUBMIT cmdbuf into `buf`: SET_FRAGMENT_ATTACHMENTS(1) + RENDER.
/// Returns bytes written. `vdm_va`/`vdm_len`, the bg/eot programs, the PBE-backed
/// color region, and the tile/fb config are wired into drm_asahi_cmd_render.
pub fn buildCmdbuf(
    buf: []u8,
    vdm_va: u64,
    vdm_len: usize,
    color_va: u64,
    color_size: u64,
    bg: uapi.drm_asahi_bg_eot,
    eot: uapi.drm_asahi_bg_eot,
    scissor_base: u64,
    dbias_base: u64,
    width: u32,
    height: u32,
) usize {
    var n: usize = 0;

    // SET_FRAGMENT_ATTACHMENTS: header (BARRIER_NONE for software cmds) + one
    // attachment describing the color buffer region the eot store writes.
    const set_hdr = uapi.drm_asahi_cmd_header{
        .cmd_type = uapi.SET_FRAGMENT_ATTACHMENTS,
        .size = @sizeOf(uapi.drm_asahi_attachment),
        .vdm_barrier = uapi.BARRIER_NONE,
        .cdm_barrier = uapi.BARRIER_NONE,
    };
    const att = uapi.drm_asahi_attachment{
        .pointer = color_va,
        .size = color_size,
        .pad = 0,
        .flags = 0,
    };
    {
        const hb = std.mem.asBytes(&set_hdr);
        @memcpy(buf[n..][0..hb.len], hb);
        n += hb.len;
        const ab = std.mem.asBytes(&att);
        @memcpy(buf[n..][0..ab.len], ab);
        n += ab.len;
    }

    // RENDER: header + drm_asahi_cmd_render body. First hardware command in the
    // submit -> barrier indices 0/0 (same as compute.zig).
    const ren_hdr = uapi.drm_asahi_cmd_header{
        .cmd_type = uapi.CMD_RENDER,
        .size = @sizeOf(uapi.drm_asahi_cmd_render),
        .vdm_barrier = 0,
        .cdm_barrier = 0,
    };
    _ = vdm_len;
    const body = uapi.drm_asahi_cmd_render{
        // PROCESS_EMPTY_TILES is mandatory when clearing (our bg clears).
        .flags = uapi.RENDER_PROCESS_EMPTY_TILES,
        .isp_zls_pixels = 0, // no depth/stencil
        .vdm_ctrl_stream_base = vdm_va,
        .vertex_helper = .{ .binary = 0, .cfg = 0, .data = 0 },
        .fragment_helper = .{ .binary = 0, .cfg = 0, .data = 0 },
        .isp_scissor_base = scissor_base,
        .isp_dbias_base = dbias_base,
        .isp_oclqry_base = 0,
        .depth = .{ .base = 0, .comp_base = 0, .stride = 0, .comp_stride = 0 },
        .stencil = .{ .base = 0, .comp_base = 0, .stride = 0, .comp_stride = 0 },
        .zls_ctrl = 0,
        .ppp_multisamplectl = PPP_MULTISAMPLECTL_1,
        .sampler_heap = 0,
        .ppp_ctrl = PPP_CTRL,
        // Mesa hk_queue.c:118-119 c->width_px = cs->cr.width / height_px = height.
        .width_px = @intCast(width),
        .height_px = @intCast(height),
        .layers = 1,
        .sampler_count = 0,
        // utile / sample config are SIZE-INDEPENDENT (the AGX TBDR tiles 32x32
        // regardless of the render dimensions) - left at the proven constants.
        .utile_width_px = UTILE_W,
        .utile_height_px = UTILE_H,
        .samples = 1,
        .sample_size_B = SAMPLE_SIZE_B,
        // isp_merge_upper = fui(tan(60)/dim), PER-AXIS, the reciprocal (Mesa
        // hk_queue.c:170-171 tan_60=1.732051f; isp_merge_upper_x = fui(tan_60 /
        // width); _y = fui(tan_60 / height)). Recompute per W and H at runtime.
        .isp_merge_upper_x = fui_rt(1.732051 / @as(f32, @floatFromInt(width))),
        .isp_merge_upper_y = fui_rt(1.732051 / @as(f32, @floatFromInt(height))),
        .bg = bg,
        .eot = eot,
        .partial_bg = bg,
        .partial_eot = eot,
        .isp_bgobjdepth = comptime fui(1.0), // clear depth (unused, no ZLS)
        // isp_bgobjvals: Mesa unconditionally programs 0x300 (hk_cmd_draw.c:634
        // `render->cr.isp_bgobjvals = 0x300;` and gallium agx_state.c). The bottom
        // 8 bits are the stencil clear (0 here, no ZLS); bits 8-9 (0x300) are a
        // hardware-required ISP background-object control we were previously
        // zeroing (we wrote 0xFF = stencil 255). Match Mesa's ground-truth value.
        .isp_bgobjvals = 0x300,
        .ts_vtx = .{ .start = .{ .handle = 0, .offset = 0 }, .end = .{ .handle = 0, .offset = 0 } },
        .ts_frag = .{ .start = .{ .handle = 0, .offset = 0 }, .end = .{ .handle = 0, .offset = 0 } },
    };
    {
        const hb = std.mem.asBytes(&ren_hdr);
        @memcpy(buf[n..][0..hb.len], hb);
        n += hb.len;
        const bb = std.mem.asBytes(&body);
        @memcpy(buf[n..][0..bb.len], bb);
        n += bb.len;
    }

    return n;
}

// ---------------------------------------------------------------------------
// CPU<->GPU coherency bracket (identical rationale to compute.zig: AGX is
// CPU-coherent, so these EL0 cache ops are belt-and-suspenders that make a
// surviving sentinel an honest "the GPU did not store" verdict).
// ---------------------------------------------------------------------------
const CACHE_LINE: usize = 64;

inline fn dsbSy() void {
    asm volatile ("dsb sy" ::: .{ .memory = true });
}

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

/// Read the packed RGBA8 word at the framebuffer center pixel from a CPU mapping
/// `cpu` of the color BO (row-major, `stride` bytes per row).
fn centerPixel(cpu: []const u8, width: u32, height: u32, stride: u32) u32 {
    const cx: usize = width / 2;
    const cy: usize = height / 2;
    const off = cy * stride + cx * 4;
    return std.mem.readInt(u32, cpu[off..][0..4], .little);
}

// ===========================================================================
// The live render.
// ===========================================================================

/// Run the solid-triangle render pass on a real AGX GPU and return the packed
/// RGBA8 word at the framebuffer center pixel. Allocates + VM-binds all BOs at
/// fixed PAGE-aligned VAs, fills the shaders/USC/VDM/PPP/PBE, pre-fills the color
/// BO with the sentinel, submits (P2 submit), waits the fence, then reads the
/// center pixel back. The caller passes an EXISTING vm_id + queue_id; the queue
/// MUST have been created with usc_exec_base = USC_EXEC_BASE.
///
/// NOTE: this performs a REAL GPU submit. It is the USER's M1 test - it cannot be
/// exercised on a box without an AGX GPU (allocBo/submit fail and the caller
/// treats that as SKIP/FAIL, not a crash). Decode the return three ways:
///   == kTriColor  -> the AGX rasterized our triangle to the center (SUCCESS)
///   == kClearColor-> the bg clear ran, the triangle missed the center
///   == kSentinel  -> nothing stored (encoding bug; coherency ruled out)
///   else          -> a partial/wrong store (note + iterate)
pub fn runTriangle(dev: *Device, vm_id: u32, queue_id: u32) Error!u32 {
    return runRender(dev, vm_id, queue_id, true);
}

/// CLEAR-ONLY discriminator. Builds + submits the SAME render pass as runTriangle
/// but WITHOUT the geometry draw (a Barrier+Terminate VDM stream, no VDM/PPP/VS/FS
/// draw). The per-tile background clear + end-of-tile store still run (they are
/// driven by the bg/eot programs + PROCESS_EMPTY_TILES in drm_asahi_cmd_render,
/// independent of the draw), so this isolates the tile-renderer INFRASTRUCTURE
/// (bg clear + eot store + tile/render config) from the DRAW. Returns the
/// framebuffer center pixel, decoded the same ways:
///   == kClearColor -> the tile renderer (bg clear + eot store + config) WORKS,
///                     and any triangle hang/miss is purely in the DRAW path.
///   == kSentinel   -> nothing stored: the bg/eot/PBE/tile config is the bug.
///   else           -> a partial/wrong store (note + iterate).
pub fn runClear(dev: *Device, vm_id: u32, queue_id: u32) Error!u32 {
    return runRender(dev, vm_id, queue_id, false);
}

/// PARAMETERIZED clear: run the PROVEN clear-only render pass (bg clear + eot
/// store, NO geometry draw) but bake the CALLER's `rgba` and write it into the
/// CALLER's output BO at `out_va`/`out_size` instead of the hardcoded clear color
/// and render.zig's own COLOR_VA. This is the asahi primitive the Prism HAL
/// `apple` driver's clear path calls: the recorded HAL Color goes to the recorded
/// render-target's Bo.
///
/// The render-infra BOs (shader/USC/ctrl/PBE/init-ppp/...) stay at their fixed
/// USC_EXEC_BASE VAs and are allocated here exactly as the self-test does; only
/// the bg-clear shader's baked color and the color attachment (PBE buffer +
/// cmd_render output) are the caller's. The proven encoding (bg/eot/PBE/tile
/// config, the fp32 st_tile) is UNCHANGED - only the color immediates and the
/// output VA are threaded through.
///
/// SCOPE: the proven clear pass now clears an ARBITRARY W x H linear RGBA8
/// framebuffer. The per-tile programs (bg-clear shader, eot store, USC blocks,
/// the VDM Barrier+Terminate clear stream) are SIZE-INDEPENDENT (the AGX TBDR
/// tiles 32x32 from the render dimensions); only the size-dependent cmd_render /
/// PBE / region-clip / isp_merge / scissor / output-size fields scale with W/H,
/// each grounded in Mesa (see buildPbe, buildScissor, buildViewportScissorPpp,
/// buildCmdbuf). `out_size` must cover the W x H RGBA8 image (W*4*H bytes). Does a
/// REAL GPU submit + fence; on a box without an AGX GPU the allocBo/submit fail
/// (the caller treats that as SKIP/FAIL, not a crash).
pub fn clearColorTo(
    dev: *Device,
    vm_id: u32,
    queue_id: u32,
    out_va: u64,
    out_size: u64,
    width: u32,
    height: u32,
    rgba: [4]f32,
) Error!void {
    if (width == 0 or height == 0) return error.MapFailed;
    if (out_size < @as(u64, width) * 4 * height) return error.MapFailed;
    _ = try runRenderCore(dev, vm_id, queue_id, out_va, out_size, width, height, rgba, false, false);
}

/// Shared render driver. `draw==true` issues the full triangle (VDM draw + VS/FS
/// + PPP); `draw==false` is the clear-only discriminator (Barrier+Terminate VDM
/// stream, bg clear + eot store only). Everything else (VA layout, the bg clear
/// program, the eot store, the PBE, the tile/render config, the sentinel
/// pre-fill, the cache bracket, the fence + readback) is identical so the two
/// readbacks differ ONLY by the presence of the draw.
fn runRender(dev: *Device, vm_id: u32, queue_id: u32, draw: bool) Error!u32 {
    // The self-test / triangle path uses render.zig's own COLOR_VA + the proven
    // dark-blue clear color, so its readback constants (kClearColor/kTriColor) are
    // unchanged. clearColorTo() reuses the same core with the caller's output +
    // color.
    return runRenderCore(dev, vm_id, queue_id, COLOR_VA, COLOR_SIZE, FB_WIDTH, FB_HEIGHT, clear_rgba, draw, true);
}

/// Shared, parameterized render core. Builds + submits the full render pass with
/// the bg-clear color `clear_in` baked into the bg shader and the color
/// attachment at `color_out_va`/`color_out_size`. `draw==true` adds the triangle
/// geometry (VS/FS/PPP/VDM draw); `draw==false` is clear-only (Barrier+Terminate).
/// All render-infra BOs stay at their fixed USC_EXEC_BASE VAs; only the bg color
/// and the output attachment vary. Returns the framebuffer center pixel.
fn runRenderCore(
    dev: *Device,
    vm_id: u32,
    queue_id: u32,
    color_out_va: u64,
    color_out_size: u64,
    // Framebuffer geometry. The self-test passes FB_WIDTH/FB_HEIGHT (64x64); the
    // HAL clear path passes the caller's target W/H. The size-dependent PBE /
    // scissor / region-clip / isp_merge / cmd_render fields all derive from these.
    width: u32,
    height: u32,
    clear_in: [4]f32,
    draw: bool,
    // When false, the CALLER owns the output BO at color_out_va (already
    // allocated + VM-bound, e.g. the Prism HAL render-target Resource): do NOT
    // allocate a second BO there, and skip the sentinel pre-fill + clean +
    // readback (the caller reads its own BO). When true, this path owns COLOR_VA
    // (the self-test / triangle).
    owns_output: bool,
) Error!u32 {
    // Allocate every BO at its fixed, PAGE-aligned VA (16-KiB alignment is
    // mandatory for VM_BIND - the P3 iter-1 lesson).
    var vs_code = try dev.allocBo(vm_id, PAGE, VS_CODE_VA);
    var ps_code = try dev.allocBo(vm_id, PAGE, PS_CODE_VA);
    var eot_code = try dev.allocBo(vm_id, PAGE, EOT_CODE_VA);
    var vs_usc = try dev.allocBo(vm_id, PAGE, VS_USC_VA);
    var ps_usc = try dev.allocBo(vm_id, PAGE, PS_USC_VA);
    var bg_usc = try dev.allocBo(vm_id, PAGE, BG_USC_VA);
    var eot_usc = try dev.allocBo(vm_id, PAGE, EOT_USC_VA);
    var vdm = try dev.allocBo(vm_id, PAGE, VDM_STREAM_VA);
    var ppp = try dev.allocBo(vm_id, PAGE, PPP_VA);
    var pbe = try dev.allocBo(vm_id, PAGE, PBE_VA);
    var clear_data = try dev.allocBo(vm_id, PAGE, CLEAR_DATA_VA);
    var bg_code = try dev.allocBo(vm_id, PAGE, BG_CODE_VA);
    var init_ppp = try dev.allocBo(vm_id, PAGE, INIT_PPP_VA);
    var scissor_bo = try dev.allocBo(vm_id, PAGE, SCISSOR_VA);
    var dbias_bo = try dev.allocBo(vm_id, PAGE, DBIAS_VA);
    var vps_ppp = try dev.allocBo(vm_id, PAGE, VPS_PPP_VA);
    // The MANDATORY graphics root descriptor BOs: the 1-entry sampler heap (the
    // reserved txf sampler #0) + the FS color uniform (4 fp16 halves).
    var sampler_heap = try dev.allocBo(vm_id, PAGE, SAMPLER_HEAP_VA);
    var fs_color = try dev.allocBo(vm_id, PAGE, FS_COLOR_VA);
    // The color attachment. When this path OWNS the output (the self-test /
    // triangle, COLOR_VA), allocate + bind it here. When the CALLER owns it (the
    // Prism HAL passes its render-target Resource's already-bound VA), do NOT
    // allocate a second GEM handle at that VA - that bind would shadow the
    // caller's BO, so the GPU would clear ours while the caller reads theirs (an
    // unwritten readback). The PBE + cmd_render below reference color_out_va
    // either way, so the GPU writes the right address regardless.
    const color_alloc: u64 = (color_out_size + PAGE - 1) & ~(PAGE - 1);
    const color: ?Bo = if (owns_output) try dev.allocBo(vm_id, color_alloc, color_out_va) else null;

    // Upload the shader binaries. The bg clear program now has its OWN dedicated,
    // self-contained, terminating shader (no longer aliasing the FS code). It
    // bakes the CALLER's clear color via the PROVEN buildColorStore encoding.
    const vs_len = buildVertexShader(vs_code.cpu[0..]);
    const ps_len = buildFragmentShader(ps_code.cpu[0..]);
    _ = buildEotShader(eot_code.cpu[0..]);
    _ = buildBgClearShaderColor(bg_code.cpu[0..], clear_in);
    _ = vs_len;
    _ = ps_len;

    // The clear color, 4 fp32 RGBA in [0,1]. Left in the CLEAR_DATA BO for
    // diagnostics/future uniform-sourced clear, but the baked clear shader does
    // NOT read it and the bg USC no longer binds it (R4 fix - see buildBgUsc: a
    // USC that binds a uniform the program never consumes, with a zero Counts
    // uniform_register_count, was dropping the background-clear dispatch's output).
    std.mem.writeInt(u32, clear_data.cpu[0..4], fui_rt(clear_in[0]), .little); // R
    std.mem.writeInt(u32, clear_data.cpu[4..8], fui_rt(clear_in[1]), .little); // G
    std.mem.writeInt(u32, clear_data.cpu[8..12], fui_rt(clear_in[2]), .little); // B
    std.mem.writeInt(u32, clear_data.cpu[12..16], fui_rt(clear_in[3]), .little); // A

    // The MANDATORY graphics root descriptor data: the reserved txf sampler #0
    // into the sampler heap, and the FS color (4 fp16 halves R,G,B,A) into the
    // color uniform BO the FS USC binds at uniform half 16 (regs u8,u9).
    _ = buildTxfSampler(sampler_heap.cpu[0..]);
    {
        const halves = fsColorHalves();
        var i: usize = 0;
        while (i < 4) : (i += 1) {
            std.mem.writeInt(u16, fs_color.cpu[i * 2 ..][0..2], halves[i], .little);
        }
    }

    // Build the USC pipeline blocks. Offsets are (code_va - USC_EXEC_BASE).
    const vs_code_off: u32 = @intCast(VS_CODE_VA - USC_EXEC_BASE);
    const ps_code_off: u32 = @intCast(PS_CODE_VA - USC_EXEC_BASE);
    const eot_code_off: u32 = @intCast(EOT_CODE_VA - USC_EXEC_BASE);
    const bg_code_off: u32 = @intCast(BG_CODE_VA - USC_EXEC_BASE);
    // VS uses r0..r3 + r5; binds the mandatory reserved txf sampler.
    _ = buildVsUsc(vs_usc.cpu[0..], vs_code_off, VS_GPR_COUNT, SAMPLER_HEAP_VA);
    // FS uses r4,r5 (fp16 color halves); binds the mandatory sampler + the color
    // uniform (4 fp16 halves at FS_COLOR_VA, read as u8,u9). 6 GPRs covers r5.
    _ = buildFsUsc(ps_usc.cpu[0..], ps_code_off, 6, SAMPLER_HEAP_VA, FS_COLOR_VA);
    // bg USC SHADER points at the dedicated BG clear code; SHARED+SHADER+REGISTERS+
    // NO_PRESHADER only (no UNIFORM word - the baked clear reads no uniform, so the
    // USC, the Counts, and the shader all agree on "no uniforms" - R4 fix).
    _ = buildBgUsc(bg_usc.cpu[0..], bg_code_off, 4); // bg uses r0..r3, no uniforms
    _ = buildEotUsc(eot_usc.cpu[0..], eot_code_off, 1, PBE_VA);

    // The PBE descriptor for the linear color attachment (caller's output VA).
    // Stride = width*4 (linear RGBA8), width/height = the framebuffer geometry.
    const color_stride: u32 = width * 4;
    _ = buildPbe(pbe.cpu[0..], color_out_va, color_stride, width, height);

    // The PPP word block (FS pipeline = ps_usc offset).
    const ppp_len = buildPpp(ppp.cpu[0..], @intCast(PS_USC_VA - USC_EXEC_BASE));
    const ppp_words: u32 = @intCast((ppp_len + 3) / 4);

    // The mandatory init PPP block (W-clamp setup - the draw-hang fix). Built
    // into its own BO; the draw VDM stream emits a PPP State Update pointing here
    // right after the barrier (matches Honeykrisp hk_cs_init_graphics).
    const init_ppp_len = buildInitPpp(init_ppp.cpu[0..]);
    const init_ppp_words: u32 = @intCast((init_ppp_len + 3) / 4);

    // The scissor + depth-bias arrays (isp_scissor_base / isp_dbias_base) and the
    // viewport/scissor/region-clip PPP State Update record (Mesa's separate
    // agx_upload_viewport_scissor block).
    _ = buildScissor(scissor_bo.cpu[0..], width, height);
    _ = buildDepthBias(dbias_bo.cpu[0..]);
    const vps_len = buildViewportScissorPpp(vps_ppp.cpu[0..], width, height);
    const vps_words: u32 = @intCast((vps_len + 3) / 4);

    // The VDM control stream. For the full triangle: VS pipeline + PPP + draw.
    // For the clear-only discriminator: a Barrier+Terminate stream with no draw.
    const vs_usc_off: u32 = @intCast(VS_USC_VA - USC_EXEC_BASE);
    const vdm_len = if (draw)
        buildVdmStream(vdm.cpu[0..], vs_usc_off, PPP_VA, ppp_words, INIT_PPP_VA, init_ppp_words, VPS_PPP_VA, vps_words)
    else
        buildVdmStreamClear(vdm.cpu[0..]);

    // The bg / eot program references. usc = (USC block VA - USC_EXEC_BASE) | 4
    // (the low-bit config tag Mesa ORs for a single-RT program). rsrc_spec = the
    // packed Counts word (bg sets the "Unknown" 0xFFFF; eot has 1 texture).
    const bg = uapi.drm_asahi_bg_eot{
        .usc = @as(u32, @intCast(BG_USC_VA - USC_EXEC_BASE)) | 4,
        .rsrc_spec = buildCounts(0, true),
    };
    const eot = uapi.drm_asahi_bg_eot{
        .usc = @as(u32, @intCast(EOT_USC_VA - USC_EXEC_BASE)) | 4,
        .rsrc_spec = buildCounts(1, false),
    };

    // Assemble the SUBMIT cmdbuf (SET_FRAGMENT_ATTACHMENTS + RENDER).
    var cmdbuf: [
        @sizeOf(uapi.drm_asahi_cmd_header) * 2 +
            @sizeOf(uapi.drm_asahi_attachment) +
            @sizeOf(uapi.drm_asahi_cmd_render)
    ]u8 = undefined;
    // For the full triangle, point isp_scissor_base/isp_dbias_base at the uploaded
    // arrays (Mesa always does). For the clear-only discriminator there is no draw
    // / per-draw PPP referencing them, so leave them null (keeps the proven clear
    // path byte-identical to the working reference).
    const cmd_len = if (draw)
        buildCmdbuf(cmdbuf[0..], VDM_STREAM_VA, vdm_len, color_out_va, color_out_size, bg, eot, SCISSOR_VA, DBIAS_VA, width, height)
    else
        buildCmdbuf(cmdbuf[0..], VDM_STREAM_VA, vdm_len, color_out_va, color_out_size, bg, eot, 0, 0, width, height);

    // Pre-fill the OWNED color BO with the sentinel (so a non-store is
    // distinguishable from a write-of-zero), then clean it out to RAM. Skipped
    // when the caller owns the output - it manages its own BO.
    if (color) |c| {
        var off: usize = 0;
        while (off + 4 <= c.cpu.len) : (off += 4) {
            std.mem.writeInt(u32, c.cpu[off..][0..4], kSentinel, .little);
        }
        cleanToPoC(c.cpu[0..]);
    }
    cleanToPoC(clear_data.cpu[0..16]);
    // Flush the mandatory graphics root descriptor BOs to RAM before the GPU
    // loads them (the sampler heap the USC Sampler word points at, and the FS
    // color uniform the USC Uniform word DMAs into u8,u9).
    cleanToPoC(sampler_heap.cpu[0..8]);
    cleanToPoC(fs_color.cpu[0..8]);

    // Out-syncobj the kernel signals on completion.
    const out_sync = try dev.syncobjCreate(false);
    defer dev.syncobjDestroy(out_sync);

    const out_syncs = [_]uapi.drm_asahi_sync{Device.binarySync(out_sync)};
    try dev.submit(queue_id, cmdbuf[0..cmd_len], &.{}, &out_syncs);

    // Wait up to 1 second for the GPU to finish the render pass. The fence MUST
    // signal before we free any BO below: the GPU is still reading the infra
    // (shaders/USC/streams/PBE) until it finishes, so unbinding/closing a BO
    // before the fence would race live GPU access.
    const handles = [_]u32{out_sync};
    const wait_result = dev.syncobjWait(&handles, 1_000_000_000);

    // Read the OWNED color BO before freeing the infra (the readback is a CPU
    // read of the GPU-written attachment; AGX is CPU-coherent, the
    // clean+invalidate is belt-and-suspenders). When the caller owns the output
    // it reads its own BO, so we have nothing to read here.
    var center: u32 = 0;
    if (color) |c| {
        cleanInvalidateToPoC(c.cpu[0..]);
        center = centerPixel(c.cpu[0..], width, height, color_stride);
    }

    // FREE EVERY infra BO this call allocated, AFTER the fence (the GPU is done)
    // and AFTER the readback. This is what makes the clear REPEATABLE: every
    // infra BO sits at a FIXED USC_EXEC_BASE-relative VA, so a second clear's
    // allocBo would EINVAL on VM_BIND ("VA already mapped") if these were left
    // bound. freeBo unbinds the VA + munmaps + GEM_CLOSEs each one, so the next
    // clearColorTo re-allocates the same VAs cleanly. The caller-owned `color`
    // (owns_output==false, the HAL's render-target Resource) is NOT freed here -
    // the HAL owns its lifetime; only the self-test's own COLOR_VA
    // (owns_output==true) is freed.
    //
    // FUTURE PERF: this re-allocates + rebinds ~18 infra BOs per clear. A
    // persistent-infra cache (alloc the fixed-VA infra ONCE per device/queue, keep
    // it bound, only re-bake the color + re-point the output PBE per clear) would
    // avoid the per-clear alloc/free churn. Deferred - correctness (repeatability)
    // first; this is a pure optimization that does not change behavior.
    dev.freeBo(vm_id, vs_code);
    dev.freeBo(vm_id, ps_code);
    dev.freeBo(vm_id, eot_code);
    dev.freeBo(vm_id, vs_usc);
    dev.freeBo(vm_id, ps_usc);
    dev.freeBo(vm_id, bg_usc);
    dev.freeBo(vm_id, eot_usc);
    dev.freeBo(vm_id, vdm);
    dev.freeBo(vm_id, ppp);
    dev.freeBo(vm_id, pbe);
    dev.freeBo(vm_id, clear_data);
    dev.freeBo(vm_id, bg_code);
    dev.freeBo(vm_id, init_ppp);
    dev.freeBo(vm_id, scissor_bo);
    dev.freeBo(vm_id, dbias_bo);
    dev.freeBo(vm_id, vps_ppp);
    dev.freeBo(vm_id, sampler_heap);
    dev.freeBo(vm_id, fs_color);
    // Free the OWNED color attachment (the self-test owns COLOR_VA). The HAL path
    // (owns_output==false) leaves `color` null and never frees the caller's BO.
    if (color) |c| dev.freeBo(vm_id, c);

    // Surface a fence timeout/failure AFTER cleanup so a failed clear still frees
    // its infra (otherwise the leaked binds would EINVAL the next clear too).
    try wait_result;
    return center;
}

// ===========================================================================
// RUNTIME GROUND-TRUTH DUMP. Rebuild OUR triangle render stream into LOCAL
// buffers (no GPU, no allocBo/submit - pure CPU) and print it in a DECODED,
// annotated, comparable form: each VDM control-stream block (Barrier / PPP-State
// / VDM-State / Index-List / Terminate) with raw bytes AND decoded meaning, the
// per-draw PPP record words (named), the init-PPP (W-clamp) block, and every
// drm_asahi_cmd_render field (esp. vertex_helper@16 / fragment_helper@32 and the
// vertex-pipeline fields). This mirrors Mesa's agxdecode_drm_cmd_render output
// (src/asahi/lib/decode.c agxdecode_drm_cmd_render + agxdecode_vdm) so OUR bytes
// can be diffed field-by-field against a real M1 capture
// (ASAHI_MESA_DEBUG=trace AGXDECODE_DUMP_FILE=stderr). READ-ONLY: it does not
// touch the GPU and does not change the clear/compute/triangle behaviour.
// ===========================================================================

/// Read a u32 LE from a byte slice at `off`.
fn rdU32(buf: []const u8, off: usize) u32 {
    return std.mem.readInt(u32, buf[off..][0..4], .little);
}

/// Print one 4-byte VDM/PPP word: "  +OFF  raw=0xWWWWWWWW  <annotation>".
fn dumpWord(w: *std.Io.Writer, base: usize, off: usize, buf: []const u8, note: []const u8) void {
    w.print("  +{d:0>3}  raw=0x{x:0>8}  {s}\n", .{ base + off, rdU32(buf, off), note }) catch {};
}

/// Hex-dump `buf` as space-separated bytes after a label (for shader binaries).
fn dumpBytes(w: *std.Io.Writer, label: []const u8, buf: []const u8) void {
    w.print("  {s} ({d} bytes):\n    ", .{ label, buf.len }) catch {};
    var i: usize = 0;
    while (i < buf.len) : (i += 1) {
        w.print("{x:0>2} ", .{buf[i]}) catch {};
        if ((i % 16) == 15 and i + 1 < buf.len) w.print("\n    ", .{}) catch {};
    }
    w.print("\n", .{}) catch {};
}

/// Decode + print the VDM control stream `buf[0..len]` block-by-block, exactly
/// the way Mesa's agxdecode_vdm walks it (block type = bits 29-31 of word0,
/// advance by each block's own encoded length). Annotates every word.
fn dumpVdm(w: *std.Io.Writer, buf: []const u8, len: usize) void {
    w.print("--- VDM control stream ({d} bytes @ VDM_STREAM_VA=0x{x}) ---\n", .{ len, VDM_STREAM_VA }) catch {};
    var off: usize = 0;
    while (off < len) {
        const word0 = rdU32(buf, off);
        const block_type: u3 = @truncate(word0 >> 29);
        switch (block_type) {
            VDM_BARRIER => {
                dumpWord(w, off, 0, buf[off..], "Barrier (type 1) usc_cache_inval@3");
                off += 4;
            },
            VDM_PPP_STATE => {
                // 8B: pointer_hi@0(8) | size_words@8(8) | type@29 ; word1 = ptr_lo
                const size_words = (word0 >> 8) & 0xFF;
                const ptr_lo = rdU32(buf, off + 4);
                const ptr_hi = word0 & 0xFF;
                const va = (@as(u64, ptr_hi) << 32) | ptr_lo;
                var nb: [96]u8 = undefined;
                const s = std.fmt.bufPrint(&nb, "PPP State Update (type 0) -> record VA=0x{x} size_words={d}", .{ va, size_words }) catch "PPP State Update";
                dumpWord(w, off, 0, buf[off..], s);
                dumpWord(w, off, 4, buf[off..], "  (pointer lo)");
                off += 8;
            },
            VDM_STATE => {
                // 4B header of present bits, then ALIGN_POT(len,8) of sub-words.
                const h = word0;
                var nb: [128]u8 = undefined;
                const s = std.fmt.bufPrint(&nb, "VDM State Update (type 2) present: w0={d} w1={d} outputs={d} unknown={d}", .{ (h >> 1) & 1, (h >> 2) & 1, (h >> 3) & 1, (h >> 5) & 1 }) catch "VDM State Update";
                dumpWord(w, off, 0, buf[off..], s);
                var p: usize = 4;
                if ((h >> 1) & 1 != 0) {
                    dumpWord(w, off, p, buf[off..], "  VS Word 0 (register/sampler counts)");
                    p += 4;
                }
                if ((h >> 2) & 1 != 0) {
                    const w1 = rdU32(buf, off + p);
                    var b2: [64]u8 = undefined;
                    const s2 = std.fmt.bufPrint(&b2, "  VS Word 1: Pipeline(>>6)@6 = 0x{x} (USC off 0x{x})", .{ (w1 >> 6), (w1 >> 6) << 6 }) catch "  VS Word 1";
                    dumpWord(w, off, p, buf[off..], s2);
                    p += 4;
                }
                if ((h >> 3) & 1 != 0) {
                    const vo = rdU32(buf, off + p);
                    var b3: [64]u8 = undefined;
                    const s3 = std.fmt.bufPrint(&b3, "  Vertex Outputs: count_1={d} count_2={d}", .{ vo & 0xFF, (vo >> 8) & 0xFF }) catch "  Vertex Outputs";
                    dumpWord(w, off, p, buf[off..], s3);
                    p += 4;
                }
                if ((h >> 5) & 1 != 0) {
                    const vu = rdU32(buf, off + p);
                    var b4: [64]u8 = undefined;
                    const s4 = std.fmt.bufPrint(&b4, "  Vertex Unknown: flat_shading_control@0(2)={d}", .{vu & 3}) catch "  Vertex Unknown";
                    dumpWord(w, off, p, buf[off..], s4);
                    p += 4;
                }
                // The whole VDM State run is ALIGN_POT(len, 8). Surface the pad.
                const aligned = (p + 7) & ~@as(usize, 7);
                if (aligned > p) {
                    dumpWord(w, off, p, buf[off..], "  (8-byte align pad, Mesa memsets 4)");
                }
                off += aligned;
            },
            VDM_INDEX_LIST => {
                const h = word0;
                var nb: [128]u8 = undefined;
                const prim = (h >> 8) & 0xFF;
                const s = std.fmt.bufPrint(&nb, "Index List (type 3): primitive@8={d} (6=Triangles) count_present={d} inst_present={d} start_present={d}", .{ prim, (h >> 22) & 1, (h >> 23) & 1, (h >> 24) & 1 }) catch "Index List";
                dumpWord(w, off, 0, buf[off..], s);
                var p: usize = 4;
                if ((h >> 22) & 1 != 0) {
                    dumpWord(w, off, p, buf[off..], "  Index count (vertices)");
                    p += 4;
                }
                if ((h >> 23) & 1 != 0) {
                    dumpWord(w, off, p, buf[off..], "  Instance count");
                    p += 4;
                }
                if ((h >> 24) & 1 != 0) {
                    dumpWord(w, off, p, buf[off..], "  Start (base vertex)");
                    p += 4;
                }
                off += p;
            },
            VDM_TERMINATE => {
                dumpWord(w, off, 0, buf[off..], "Stream Terminate (type 6) -> STOP");
                off += 4;
                break;
            },
            else => {
                var nb: [48]u8 = undefined;
                const s = std.fmt.bufPrint(&nb, "UNKNOWN block type {d} - DESYNC RISK", .{block_type}) catch "UNKNOWN block";
                dumpWord(w, off, 0, buf[off..], s);
                off += 4;
            },
        }
    }
}

/// Decode + print the per-draw PPP record `buf[0..len]`. A 4B PPP_HEADER of
/// present bits followed by exactly the present sub-words in bit order. Mirrors
/// Mesa decode.c agxdecode_record. We know our exact layout, so annotate it.
fn dumpPpp(w: *std.Io.Writer, buf: []const u8, len: usize) void {
    w.print("--- per-draw PPP record ({d} bytes @ PPP_VA=0x{x}) ---\n", .{ len, PPP_VA }) catch {};
    const h = rdU32(buf, 0);
    var nb: [128]u8 = undefined;
    const s = std.fmt.bufPrint(&nb, "PPP_HEADER present bits = 0x{x:0>8}", .{h}) catch "PPP_HEADER";
    dumpWord(w, 0, 0, buf, s);
    // Our buildPpp emits, in order: FragControl(scissor_enable=1), FragControl2,
    // FrontFace, FrontFace2, BackFace, BackFace2, OutputSelect, Varying32,
    // Varying16, Cull, Cull2, FS word0..3, OutputSize. (Viewport/scissor are in
    // their own PPP block now - see dumpViewportScissorPpp.)
    var p: usize = 4;
    dumpWord(w, 0, p, buf, "FRAGMENT_CONTROL (scissor_enable@16=1)");
    p += 4;
    dumpWord(w, 0, p, buf, "FRAGMENT_CONTROL_2 (tag_write_disable off)");
    p += 4;
    dumpWord(w, 0, p, buf, "FRAGMENT_FACE front (depth_write_disable@21, func Always@24)");
    p += 4;
    dumpWord(w, 0, p, buf, "FRAGMENT_FACE_2 front (object_type Triangle@28=0)");
    p += 4;
    dumpWord(w, 0, p, buf, "FRAGMENT_FACE back");
    p += 4;
    dumpWord(w, 0, p, buf, "FRAGMENT_FACE_2 back");
    p += 4;
    dumpWord(w, 0, p, buf, "OUTPUT_SELECT (0, position-only VS, no varyings)");
    p += 4;
    dumpWord(w, 0, p, buf, "VARYING_COUNTS_32 (0)");
    p += 4;
    dumpWord(w, 0, p, buf, "VARYING_COUNTS_16 (0)");
    p += 4;
    dumpWord(w, 0, p, buf, "CULL (flat_shading_vertex@7=3, depth_clip@10=1)");
    p += 4;
    dumpWord(w, 0, p, buf, "CULL_2 (clamp_w@5=1)");
    p += 4;
    dumpWord(w, 0, p, buf, "FRAGMENT_SHADER word_0 (uniform=64 sampler=4compact, Mesa noddtri)");
    p += 4;
    const fsw1 = rdU32(buf, p);
    var b1: [64]u8 = undefined;
    const sf = std.fmt.bufPrint(&b1, "FRAGMENT_SHADER word_1: Pipeline(>>6)@6 = 0x{x} (USC off 0x{x})", .{ fsw1 >> 6, (fsw1 >> 6) << 6 }) catch "FRAGMENT_SHADER word_1";
    dumpWord(w, 0, p, buf, sf);
    p += 4;
    dumpWord(w, 0, p, buf, "FRAGMENT_SHADER word_2 (CF bindings 0)");
    p += 4;
    dumpWord(w, 0, p, buf, "FRAGMENT_SHADER word_3 (0)");
    p += 4;
    dumpWord(w, 0, p, buf, "OUTPUT_SIZE (Count = UVS size = 4)");
    p += 4;
    if (p < len) w.print("  ... {d} trailing bytes\n", .{len - p}) catch {};
}

/// Decode + print the init (W-clamp) PPP block.
fn dumpInitPpp(w: *std.Io.Writer, buf: []const u8, len: usize) void {
    w.print("--- init PPP record (W-clamp setup, {d} bytes @ INIT_PPP_VA=0x{x}) ---\n", .{ len, INIT_PPP_VA }) catch {};
    const h = rdU32(buf, 0);
    var nb: [96]u8 = undefined;
    const s = std.fmt.bufPrint(&nb, "PPP_HEADER present = 0x{x:0>8} (w_clamp@16, occl_q_2@25, out_unknown@26, varying_2@28)", .{h}) catch "PPP_HEADER";
    dumpWord(w, 0, 0, buf, s);
    const wc = rdU32(buf, 4);
    const wcf: f32 = @bitCast(wc);
    var b: [64]u8 = undefined;
    const sv = std.fmt.bufPrint(&b, "W_CLAMP = {e} (Mesa uses 1e-10)", .{wcf}) catch "W_CLAMP";
    dumpWord(w, 0, 4, buf, sv);
    dumpWord(w, 0, 8, buf, "FRAGMENT_OCCLUSION_QUERY_2 (0)");
    dumpWord(w, 0, 12, buf, "OUTPUT_UNKNOWN (0)");
    w.print("  +016  VARYING_2 (8B): 0x{x:0>8} 0x{x:0>8}\n", .{ rdU32(buf, 16), rdU32(buf, 20) }) catch {};
}

/// Decode + print the viewport/scissor/region-clip PPP block (Mesa's separate
/// agx_upload_viewport_scissor block).
fn dumpViewportScissorPpp(w: *std.Io.Writer, buf: []const u8, len: usize) void {
    w.print("--- viewport/scissor PPP record ({d} bytes @ VPS_PPP_VA=0x{x}) ---\n", .{ len, VPS_PPP_VA }) catch {};
    const h = rdU32(buf, 0);
    var nb: [96]u8 = undefined;
    const s = std.fmt.bufPrint(&nb, "PPP_HEADER present = 0x{x:0>8} (depth_bias_scissor@8, region_clip@10, viewport@11, vp_count@12)", .{h}) catch "PPP_HEADER";
    dumpWord(w, 0, 0, buf, s);
    dumpWord(w, 0, 4, buf, "DEPTH_BIAS_SCISSOR (scissor idx 0, dbias idx 0)");
    const rc0 = rdU32(buf, 8);
    var b: [96]u8 = undefined;
    const src = std.fmt.bufPrint(&b, "REGION_CLIP (8B): max_x-1@0={d} min_x@16={d} enable@31={d} | hi=0x{x:0>8}", .{ rc0 & 0x1FF, (rc0 >> 16) & 0x1FF, (rc0 >> 31) & 1, rdU32(buf, 12) }) catch "REGION_CLIP";
    dumpWord(w, 0, 8, buf, src);
    dumpWord(w, 0, 16, buf, "VIEWPORT_CONTROL (0)");
    const vp_names = [_][]const u8{ "translate_x", "scale_x", "translate_y", "scale_y", "translate_z", "scale_z" };
    var p: usize = 20;
    for (vp_names) |name| {
        const fv: f32 = @bitCast(rdU32(buf, p));
        var bb: [64]u8 = undefined;
        const sv = std.fmt.bufPrint(&bb, "VIEWPORT {s} = {d}", .{ name, fv }) catch name;
        dumpWord(w, 0, p, buf, sv);
        p += 4;
    }
}

/// Decode + print the FS USC block byte-by-byte by tag, to confirm the FRAGMENT
/// PROPERTIES word is present (the prime draw-hang fix).
fn dumpFsUsc(w: *std.Io.Writer, buf: []const u8) void {
    var off: usize = 0;
    while (off < buf.len) {
        const tag = buf[off];
        switch (tag) {
            USC_TAG_SAMPLER => {
                const r = std.mem.readInt(u64, buf[off..][0..8], .little);
                w.print("  +{d:0>2}  SAMPLER (tag 0x9d, 8B): start={d} count={d} heap=0x{x} (reserved txf sampler)\n", .{ off, (r >> 8) & 0xFF, (r >> 20) & 0x7F, ((r >> 27) & 0xFFFFFFFFF) << 3 }) catch {};
                off += 8;
            },
            USC_TAG_UNIFORM => {
                const r = std.mem.readInt(u64, buf[off..][0..8], .little);
                w.print("  +{d:0>2}  UNIFORM (tag 0x1d, 8B): start_half={d} size_halfs={d} buffer=0x{x} (FS color)\n", .{ off, (r >> 8) & 0xFF, (r >> 20) & 0x3F, ((r >> 26) & 0x3FFFFFFFFF) << 2 }) catch {};
                off += 8;
            },
            USC_TAG_SHARED => {
                w.print("  +{d:0>2}  SHARED (tag 0x4d, 4B): tilebuffer 32x32\n", .{off}) catch {};
                off += 4;
            },
            USC_TAG_SHADER => {
                const lv = (buf[off + 1] >> 0) & 1;
                const unk2 = (rdU32(buf, off) >> 10) & 0x3F;
                w.print("  +{d:0>2}  SHADER (tag 0x0d, 6B): loads_varyings={d} unk2={d}\n", .{ off, lv, unk2 }) catch {};
                off += 6;
            },
            USC_TAG_REGISTERS => {
                const r = rdU32(buf, off);
                w.print("  +{d:0>2}  REGISTERS (tag 0x8d, 4B): count_groups={d} unk1@13={d} unk4@24=0x{x}\n", .{ off, (r >> 8) & 0x1F, (r >> 13) & 1, (r >> 24) & 0xFF }) catch {};
                off += 4;
            },
            USC_TAG_FRAGMENT_PROPERTIES => {
                const r = rdU32(buf, off);
                w.print("  +{d:0>2}  FRAGMENT_PROPERTIES (tag 0x58, 4B): early_z@8={d} unk2@9={d} unk3@12=0x{x} unk4@16=0x{x}\n", .{ off, (r >> 8) & 1, (r >> 9) & 1, (r >> 12) & 0xF, (r >> 16) & 0xFF }) catch {};
                off += 4;
            },
            USC_TAG_NO_PRESHADER => {
                w.print("  +{d:0>2}  NO_PRESHADER (tag 0x88, 2B)\n", .{off}) catch {};
                off += 2;
            },
            else => {
                w.print("  +{d:0>2}  UNKNOWN USC tag 0x{x:0>2}\n", .{ off, tag }) catch {};
                break;
            },
        }
    }
}

/// Print every drm_asahi_cmd_render field, mirroring Mesa's
/// agxdecode_drm_cmd_render DUMP_FIELD order so the fields line up against a real
/// M1 capture. Emphasises vertex_helper@16 / fragment_helper@32 and the
/// vertex-pipeline fields (the geometry-setup suspects).
fn dumpCmdRender(w: *std.Io.Writer, c: *const uapi.drm_asahi_cmd_render) void {
    w.print("--- drm_asahi_cmd_render ({d} bytes) ---\n", .{@sizeOf(uapi.drm_asahi_cmd_render)}) catch {};
    w.print("  flags                = 0x{x}\n", .{c.flags}) catch {};
    w.print("  isp_zls_pixels       = 0x{x}\n", .{c.isp_zls_pixels}) catch {};
    w.print("  vdm_ctrl_stream_base = 0x{x}\n", .{c.vdm_ctrl_stream_base}) catch {};
    w.print("  vertex_helper   @16  = .binary=0x{x} .cfg=0x{x} .data=0x{x}\n", .{ c.vertex_helper.binary, c.vertex_helper.cfg, c.vertex_helper.data }) catch {};
    w.print("  fragment_helper @32  = .binary=0x{x} .cfg=0x{x} .data=0x{x}\n", .{ c.fragment_helper.binary, c.fragment_helper.cfg, c.fragment_helper.data }) catch {};
    w.print("  isp_scissor_base     = 0x{x}\n", .{c.isp_scissor_base}) catch {};
    w.print("  isp_dbias_base       = 0x{x}\n", .{c.isp_dbias_base}) catch {};
    w.print("  isp_oclqry_base      = 0x{x}\n", .{c.isp_oclqry_base}) catch {};
    w.print("  zls_ctrl             = 0x{x}\n", .{c.zls_ctrl}) catch {};
    w.print("  ppp_multisamplectl   = 0x{x}\n", .{c.ppp_multisamplectl}) catch {};
    w.print("  ppp_ctrl             = 0x{x}\n", .{c.ppp_ctrl}) catch {};
    w.print("  width_px             = {d}\n", .{c.width_px}) catch {};
    w.print("  height_px            = {d}\n", .{c.height_px}) catch {};
    w.print("  layers               = {d}\n", .{c.layers}) catch {};
    w.print("  samples              = {d}\n", .{c.samples}) catch {};
    w.print("  sample_size_B        = {d}\n", .{c.sample_size_B}) catch {};
    w.print("  utile_width_px       = {d}\n", .{c.utile_width_px}) catch {};
    w.print("  utile_height_px      = {d}\n", .{c.utile_height_px}) catch {};
    w.print("  isp_merge_upper_x    = 0x{x}\n", .{c.isp_merge_upper_x}) catch {};
    w.print("  isp_merge_upper_y    = 0x{x}\n", .{c.isp_merge_upper_y}) catch {};
    w.print("  bg.usc               = 0x{x}  bg.rsrc_spec        = 0x{x}\n", .{ c.bg.usc, c.bg.rsrc_spec }) catch {};
    w.print("  eot.usc              = 0x{x}  eot.rsrc_spec       = 0x{x}\n", .{ c.eot.usc, c.eot.rsrc_spec }) catch {};
    w.print("  partial_bg.usc       = 0x{x}  partial_bg.rsrc_spec= 0x{x}\n", .{ c.partial_bg.usc, c.partial_bg.rsrc_spec }) catch {};
    w.print("  partial_eot.usc      = 0x{x}  partial_eot.rsrc    = 0x{x}\n", .{ c.partial_eot.usc, c.partial_eot.rsrc_spec }) catch {};
    w.print("  isp_bgobjdepth       = 0x{x}\n", .{c.isp_bgobjdepth}) catch {};
    w.print("  isp_bgobjvals        = 0x{x}\n", .{c.isp_bgobjvals}) catch {};
    w.print("  sampler_heap         = 0x{x}  sampler_count = {d}\n", .{ c.sampler_heap, c.sampler_count }) catch {};
}

/// RUNTIME ground-truth dump of OUR triangle render stream, decoded + annotated
/// for a field-by-field diff against a real M1 Mesa capture. Builds every block
/// into LOCAL buffers (NO GPU), so it runs on the dev box or the M1. Pass `draw`
/// = true for the triangle stream, false for the clear-only stream. Invoked from
/// asahi-info via `--dump-render`.
pub fn dumpStream(w: *std.Io.Writer, draw: bool) void {
    w.print("\n========================================================\n", .{}) catch {};
    w.print("OUR AGX render stream dump (draw={}) - decoded ground truth\n", .{draw}) catch {};
    w.print("Diff against Mesa: ASAHI_MESA_DEBUG=trace AGXDECODE_DUMP_FILE=stderr\n", .{}) catch {};
    w.print("========================================================\n", .{}) catch {};

    // Rebuild all the blocks into local buffers (mirrors runRender, no GPU).
    var vs_code: [PAGE]u8 = undefined;
    var ps_code: [PAGE]u8 = undefined;
    var eot_code: [PAGE]u8 = undefined;
    var bg_code: [PAGE]u8 = undefined;
    @memset(vs_code[0..], 0);
    @memset(ps_code[0..], 0);
    @memset(eot_code[0..], 0);
    @memset(bg_code[0..], 0);
    const vs_len = buildVertexShader(vs_code[0..]);
    const ps_len = buildFragmentShader(ps_code[0..]);
    const eot_len = buildEotShader(eot_code[0..]);
    const bg_len = buildBgClearShader(bg_code[0..]);

    var ppp_buf: [PAGE]u8 = undefined;
    var init_ppp_buf: [PAGE]u8 = undefined;
    var vps_buf: [PAGE]u8 = undefined;
    @memset(ppp_buf[0..], 0);
    @memset(init_ppp_buf[0..], 0);
    @memset(vps_buf[0..], 0);
    const ppp_len = buildPpp(ppp_buf[0..], @intCast(PS_USC_VA - USC_EXEC_BASE));
    const ppp_words: u32 = @intCast((ppp_len + 3) / 4);
    const init_ppp_len = buildInitPpp(init_ppp_buf[0..]);
    const init_ppp_words: u32 = @intCast((init_ppp_len + 3) / 4);
    const vps_len = buildViewportScissorPpp(vps_buf[0..], FB_WIDTH, FB_HEIGHT);
    const vps_words: u32 = @intCast((vps_len + 3) / 4);

    // The FS USC block (to surface the Fragment Properties word in the dump).
    var fs_usc_buf: [PAGE]u8 = undefined;
    @memset(fs_usc_buf[0..], 0);
    const fs_usc_len = buildFsUsc(fs_usc_buf[0..], @intCast(PS_CODE_VA - USC_EXEC_BASE), 6, SAMPLER_HEAP_VA, FS_COLOR_VA);

    var vdm_buf: [PAGE]u8 = undefined;
    @memset(vdm_buf[0..], 0);
    const vs_usc_off: u32 = @intCast(VS_USC_VA - USC_EXEC_BASE);
    const vdm_len = if (draw)
        buildVdmStream(vdm_buf[0..], vs_usc_off, PPP_VA, ppp_words, INIT_PPP_VA, init_ppp_words, VPS_PPP_VA, vps_words)
    else
        buildVdmStreamClear(vdm_buf[0..]);

    // The bg / eot program references + the cmd_render body (mirror runRender).
    const bg = uapi.drm_asahi_bg_eot{
        .usc = @as(u32, @intCast(BG_USC_VA - USC_EXEC_BASE)) | 4,
        .rsrc_spec = buildCounts(0, true),
    };
    const eot = uapi.drm_asahi_bg_eot{
        .usc = @as(u32, @intCast(EOT_USC_VA - USC_EXEC_BASE)) | 4,
        .rsrc_spec = buildCounts(1, false),
    };
    var cmdbuf: [
        @sizeOf(uapi.drm_asahi_cmd_header) * 2 +
            @sizeOf(uapi.drm_asahi_attachment) +
            @sizeOf(uapi.drm_asahi_cmd_render)
    ]u8 = undefined;
    _ = if (draw)
        buildCmdbuf(cmdbuf[0..], VDM_STREAM_VA, vdm_len, COLOR_VA, COLOR_SIZE, bg, eot, SCISSOR_VA, DBIAS_VA, FB_WIDTH, FB_HEIGHT)
    else
        buildCmdbuf(cmdbuf[0..], VDM_STREAM_VA, vdm_len, COLOR_VA, COLOR_SIZE, bg, eot, 0, 0, FB_WIDTH, FB_HEIGHT);
    // The drm_asahi_cmd_render body sits after SET_FRAGMENT_ATTACHMENTS (hdr 8 +
    // attachment 24) + the RENDER header (8).
    const body: *const uapi.drm_asahi_cmd_render = @ptrCast(@alignCast(&cmdbuf[8 + 24 + 8]));

    // 1) drm_asahi_cmd_render fields (the vertex-pipeline + helper suspects).
    dumpCmdRender(w, body);

    // 2) The VDM control stream, block-by-block.
    dumpVdm(w, vdm_buf[0..], vdm_len);

    if (draw) {
        // 3) The init (W-clamp) PPP block.
        dumpInitPpp(w, init_ppp_buf[0..], init_ppp_len);
        // 4) The viewport/scissor/region-clip PPP block (Mesa's separate block).
        dumpViewportScissorPpp(w, vps_buf[0..], vps_len);
        // 5) The per-draw PPP record.
        dumpPpp(w, ppp_buf[0..], ppp_len);
        // 6) The FS USC block (to confirm Fragment Properties is present).
        w.print("--- FS USC block ({d} bytes @ PS_USC_VA) ---\n", .{fs_usc_len}) catch {};
        dumpFsUsc(w, fs_usc_buf[0..fs_usc_len]);
    }

    // 7) The shader binaries (raw bytes for an applegpu round-trip / diff).
    w.print("--- shader binaries ---\n", .{}) catch {};
    dumpBytes(w, "VS @VS_CODE_VA", vs_code[0..vs_len]);
    dumpBytes(w, "FS @PS_CODE_VA", ps_code[0..ps_len]);
    dumpBytes(w, "EOT @EOT_CODE_VA", eot_code[0..eot_len]);
    dumpBytes(w, "BG-clear @BG_CODE_VA", bg_code[0..bg_len]);

    w.print("======================= end dump =======================\n\n", .{}) catch {};
}

// ===========================================================================
// Structural tests (no GPU on this box - these check the encoders' shapes/sizes,
// since execution is the M1's test).
// ===========================================================================

test "vertex shader binary shape: vertex_id ALU + 2 mov_imm + 4 st_var + stop = 84 bytes" {
    var buf: [128]u8 = undefined;
    const n = buildVertexShader(&buf);
    // iadd(8)+and(6)+and(6)+cvt(6)+cvt(6)+fmadd(8)+fmadd(8) = 48 ALU bytes,
    // + 2*6 (mov_imm z,w) + 4*4 (st_var) + 2 (stop) = 48+12+16+2 = 78.
    try std.testing.expectEqual(@as(usize, 78), n);
    // first instruction is the vertex_id*2 iadd (opcode 0x0E).
    try std.testing.expectEqual(@as(u8, 0x0E), buf[0]);
    // the ALU run reads vertex_id from r5 (the iadd src bytes are exact).
    try std.testing.expectEqualSlices(u8, &VS_IADD_R0_VID2, buf[0..8]);
    try std.testing.expectEqualSlices(u8, &VS_AND_R0_2, buf[8..14]);
    // z/w mov_imm (proven 32-bit form) at offset 48.
    try std.testing.expectEqual(@as(u8, 0x62), buf[48]);
    // first st_var opcode low byte 0x11 (not final) at offset 60.
    try std.testing.expectEqual(@as(u8, 0x11), buf[60]);
    // last st_var sets the final bit (low byte 0x91) at offset 72.
    try std.testing.expectEqual(@as(u8, 0x91), buf[72]);
    // stop at the end.
    try std.testing.expectEqual(@as(u8, 0x88), buf[76]);
}

test "vertex shader emits the 3 distinct fullscreen-triangle positions" {
    // Software model of the VS arithmetic: confirm the 3 vertex ids map to the
    // covering triangle (-1,-1),(3,-1),(-1,3) so the center (0,0) is inside.
    const model = struct {
        fn pos(vid: u32) [2]f32 {
            const xi: f32 = @floatFromInt((vid << 1) & 2);
            const yi: f32 = @floatFromInt(vid & 2);
            return .{ xi * 2.0 - 1.0, yi * 2.0 - 1.0 };
        }
    };
    try std.testing.expectEqual([2]f32{ -1.0, -1.0 }, model.pos(0));
    try std.testing.expectEqual([2]f32{ 3.0, -1.0 }, model.pos(1));
    try std.testing.expectEqual([2]f32{ -1.0, 3.0 }, model.pos(2));
    // the 3 are distinct (not a degenerate point).
    try std.testing.expect(model.pos(0)[0] != model.pos(1)[0]);
    try std.testing.expect(model.pos(0)[1] != model.pos(2)[1]);
}

test "fp16 conversion: the chosen float colors land on the expected RGBA8 readbacks" {
    // Each component f32 -> fp16 -> (PBE) round(component*255) -> RGBA8 byte.
    // The CPU reads the RGBA8 pixel as a little-endian u32 = R | G<<8 | B<<16 | A<<24.
    const expect = struct {
        fn rgba8(rgba: [4]f32) u32 {
            var out: u32 = 0;
            inline for (rgba, 0..) |c, i| {
                const f: f32 = @floatCast(@as(f16, @floatCast(c))); // round to fp16
                const b: u32 = @intFromFloat(@round(std.math.clamp(f, 0.0, 1.0) * 255.0));
                out |= b << @intCast(i * 8);
            }
            return out;
        }
    };
    try std.testing.expectEqual(kTriColor, expect.rgba8(tri_rgba)); // 0xFFFF00FF
    try std.testing.expectEqual(kClearColor, expect.rgba8(clear_rgba)); // 0xFF800000
}

test "fragment shader is Mesa noddtri: mov r4,u8 + mov r5,u9 + wait_pix + st_tile(r4l_r4h_r5l_r5h) + stop = 26 bytes" {
    var buf: [64]u8 = undefined;
    const n = buildFragmentShader(&buf);
    // mov(6) + mov(6) + wait_pix(4) + st_tile(8) + stop(2) = 26. This is the
    // config-matched Mesa noddtri FS (uniform-sourced fp16 color), NOT the old
    // baked 4x fp32 form.
    try std.testing.expectEqual(@as(usize, 26), n);
    // mov r4, u8 - byte-identical to Mesa's 7e1190098000 (mesa_noddtri.txt:287).
    try std.testing.expectEqualSlices(u8, &[_]u8{ 0x7e, 0x11, 0x90, 0x09, 0x80, 0x00 }, buf[0..6]);
    // mov r5, u9 - byte-identical to assembled 7e1592098000.
    try std.testing.expectEqualSlices(u8, &[_]u8{ 0x7e, 0x15, 0x92, 0x09, 0x80, 0x00 }, buf[6..12]);
    // wait_pix 0xC at offset 12: bytes 48 0C 00 00 (Mesa egltri/noddtri pixwait).
    try std.testing.expectEqualSlices(u8, &WAIT_PIX_C, buf[12..16]);
    // st_tile r4l_r4h_r5l_r5h u8norm xyzw 255 - byte-identical to Mesa's
    // 09100004f0fc0003 (mesa_noddtri.txt:291).
    try std.testing.expectEqualSlices(u8, &[_]u8{ 0x09, 0x10, 0x00, 0x04, 0xf0, 0xfc, 0x00, 0x03 }, buf[16..24]);
    try std.testing.expectEqual(@as(u8, 0x88), buf[24]); // stop
    const st = std.mem.readInt(u64, buf[16..24], .little);
    // half-register data group r4l_r4h_r5l_r5h: Rt@8=0 (16-bit), R@9..14 = 8.
    try std.testing.expectEqual(@as(u64, 0), (st >> 8) & 1); // Rt = 0 (halves)
    try std.testing.expectEqual(@as(u64, 8), (st >> 9) & 0x3F); // R = half base 8
    // broadcast sample mask 0xFF (bits 42..49) + u8norm (bits 24..27) + mask 0xF.
    const sample_lo = (st >> 42) & 0x3F;
    const sample_hi = (st >> 56) & 0x3;
    try std.testing.expectEqual(@as(u64, 0xFF), sample_lo | (sample_hi << 6));
    try std.testing.expectEqual(@as(u64, 4), (st >> 24) & 0xF);
    try std.testing.expectEqual(@as(u64, 0xF), (st >> 36) & 0xF);
}

test "FS color uniform: 4 fp16 RGBA halves for magenta = 3C00 0000 3C00 3C00" {
    const h = fsColorHalves();
    try std.testing.expectEqual(@as(u16, 0x3C00), h[0]); // R = 1.0
    try std.testing.expectEqual(@as(u16, 0x0000), h[1]); // G = 0.0
    try std.testing.expectEqual(@as(u16, 0x3C00), h[2]); // B = 1.0
    try std.testing.expectEqual(@as(u16, 0x3C00), h[3]); // A = 1.0
}

test "txf sampler descriptor matches Mesa noddtri (Min LOD 0, Max LOD 14, Nearest, clamp-to-border)" {
    var buf: [8]u8 = undefined;
    const n = buildTxfSampler(&buf);
    try std.testing.expectEqual(@as(usize, 8), n);
    // Byte-identical to the computed Mesa txf sampler: 00 00 0E 68 1B 00 00 00.
    try std.testing.expectEqualSlices(u8, &[_]u8{ 0x00, 0x00, 0x0E, 0x68, 0x1B, 0x00, 0x00, 0x00 }, &buf);
    const r = std.mem.readInt(u64, &buf, .little);
    try std.testing.expectEqual(@as(u64, 0), r & 0x3FF); // Min LOD = 0
    try std.testing.expectEqual(@as(u64, 14 * 64), (r >> 10) & 0x3FF); // Max LOD = 14.0
    try std.testing.expectEqual(@as(u64, 1), (r >> 27) & 0x3); // Mip = Nearest
    try std.testing.expectEqual(@as(u64, 3), (r >> 29) & 0x7); // Wrap S clamp-to-border
}

test "eot shader binary shape: mov_imm + block_image_store + stop + 8x trap" {
    var buf: [64]u8 = undefined;
    const n = buildEotShader(&buf);
    // 6 (mov_imm) + 10 (block_image_store) + 2 (stop) + 16 (8x trap) = 34.
    try std.testing.expectEqual(@as(usize, 34), n);
    try std.testing.expectEqual(@as(u8, 0xB1), buf[6]); // block_image_store opcode
    try std.testing.expectEqual(@as(u8, 0x88), buf[16]); // stop
    try std.testing.expectEqual(@as(u8, 0x08), buf[18]); // first trap
}

test "bg clear shader: 4 fp32 components + wait + st_tile(r0_r1_r2_r3) + stop = 36 bytes, distinct color from FS" {
    var buf: [64]u8 = undefined;
    const n = buildBgClearShader(&buf);
    // Same shape as the FS: 4*6 (mov_imm 32-bit) + 2 (wait) + 8 (st_tile) + 2 (stop) = 36.
    try std.testing.expectEqual(@as(usize, 36), n);
    try std.testing.expectEqual(@as(u8, 0x62), buf[0]); // first mov_imm
    // it writes the CLEAR color (dark blue), distinct from the triangle color.
    // R component (clear R = 0.0 -> 0x00000000) in the first mov's imm32.
    try std.testing.expectEqual(fui_rt(clear_rgba[0]), std.mem.readInt(u32, buf[2..6], .little));
    // B component (clear B = 0.5 -> 0x3F000000) in the THIRD mov's imm32 (offset 12+2).
    try std.testing.expectEqual(fui_rt(clear_rgba[2]), std.mem.readInt(u32, buf[14..18], .little));
    try std.testing.expect(kClearColor != kTriColor);
    try std.testing.expectEqual(@as(u8, 0x38), buf[24]); // wait
    try std.testing.expectEqual(@as(u8, 0x09), buf[26]); // st_tile opcode
    // 32-bit data register group r0_r1_r2_r3 (D=2): D low = 1.
    const st = std.mem.readInt(u64, buf[26..34], .little);
    try std.testing.expectEqual(@as(u64, 1), (st >> 8) & 0x7F);
    try std.testing.expectEqual(@as(u8, 0x88), buf[34]); // stop, no trap footer
}

test "buildBgClearShaderColor with kClearColor's rgba == buildBgClearShader (proven fp32 round-trip)" {
    // The parameterized clear shader, called with the SAME dark-blue clear_rgba
    // the self-test bakes, must be byte-for-byte the proven buildBgClearShader -
    // so clearColorTo with clear_rgba still hits the M1-proven 0xFF800000 readback.
    var a: [64]u8 = undefined;
    var b: [64]u8 = undefined;
    const na = buildBgClearShader(&a);
    const nb = buildBgClearShaderColor(&b, clear_rgba);
    try std.testing.expectEqual(na, nb);
    try std.testing.expectEqualSlices(u8, a[0..na], b[0..nb]);
    // And the proven fp32 -> u8norm round-trip of clear_rgba lands on kClearColor.
    const expect = struct {
        fn rgba8(rgba: [4]f32) u32 {
            var out: u32 = 0;
            inline for (rgba, 0..) |c, i| {
                const bb: u32 = @intFromFloat(@round(std.math.clamp(c, 0.0, 1.0) * 255.0));
                out |= bb << @intCast(i * 8);
            }
            return out;
        }
    };
    try std.testing.expectEqual(kClearColor, expect.rgba8(clear_rgba)); // 0xFF800000

    // A DIFFERENT caller color bakes DIFFERENT immediate words (only the color
    // varies; the wait/st_tile/stop encoding is byte-stable).
    var c: [64]u8 = undefined;
    const nc = buildBgClearShaderColor(&c, .{ 1.0, 0.0, 0.0, 1.0 }); // red
    try std.testing.expectEqual(nb, nc);
    try std.testing.expectEqual(fui_rt(1.0), std.mem.readInt(u32, c[2..6], .little)); // R imm
    // the non-color tail (wait + st_tile + stop) is identical to the dark-blue clear.
    try std.testing.expectEqualSlices(u8, b[24..nb], c[24..nc]);
}

test "clear-only VDM stream is exactly Barrier + Terminate (no draw) = 8 bytes" {
    var buf: [64]u8 = undefined;
    const n = buildVdmStreamClear(&buf);
    try std.testing.expectEqual(@as(usize, 8), n);
    const barrier = std.mem.readInt(u32, buf[0..4], .little);
    try std.testing.expectEqual(@as(u32, VDM_BARRIER), barrier >> 29);
    const term = std.mem.readInt(u32, buf[4..8], .little);
    try std.testing.expectEqual(@as(u32, VDM_TERMINATE), term >> 29);
    // NO index-list / draw block anywhere in the clear stream.
    try std.testing.expect((barrier >> 29) != VDM_INDEX_LIST);
    try std.testing.expect((term >> 29) != VDM_INDEX_LIST);
}

test "st_var encodes opcode, slot, final bit, and immediate-index" {
    var buf: [4]u8 = undefined;
    encodeStVar(&buf, 0, 0, false);
    const raw = std.mem.readInt(u32, &buf, .little);
    try std.testing.expectEqual(@as(u32, 0x11), raw & 0x3FF); // opcode low 10 bits
    try std.testing.expectEqual(@as(u32, 1), (raw >> 23) & 1); // immediate index
    try std.testing.expectEqual(@as(u32, 0x8), raw >> 28); // the XXX constant
    // final bit set for the last store.
    encodeStVar(&buf, 3, 3, true);
    const raw2 = std.mem.readInt(u32, &buf, .little);
    try std.testing.expectEqual(@as(u32, 1), (raw2 >> 7) & 1);
    try std.testing.expectEqual(@as(u32, 3), (raw2 >> 16) & 0x3F); // slot 3
}

test "USC blocks have the expected tag order and sizes" {
    var buf: [64]u8 = undefined;
    // VS USC: Sampler(8)+Shared(4)+Shader(6)+Registers(4)+NoPreshader(2) = 24.
    // The mandatory reserved-txf SAMPLER comes FIRST (the graphics root descriptor).
    const vn = buildVsUsc(&buf, 0x1000, 4, SAMPLER_HEAP_VA);
    try std.testing.expectEqual(@as(usize, 24), vn);
    try std.testing.expectEqual(USC_TAG_SAMPLER, buf[0]);
    // Sampler word: start 0, count 1, buffer = sampler heap >> 3.
    {
        const sw = std.mem.readInt(u64, buf[0..8], .little);
        try std.testing.expectEqual(@as(u64, 1), (sw >> 20) & 0x7F); // count = 1
        try std.testing.expectEqual(SAMPLER_HEAP_VA >> 3, (sw >> 27) & 0xFFFFFFFFF);
    }
    try std.testing.expectEqual(USC_TAG_SHARED, buf[8]);
    try std.testing.expectEqual(USC_TAG_SHADER, buf[12]);
    try std.testing.expectEqual(USC_TAG_REGISTERS, buf[18]);
    try std.testing.expectEqual(USC_TAG_NO_PRESHADER, buf[22]);

    // EOT USC: Texture(8)+Shared(4)+Shader(6)+Registers(4)+NoPreshader(2) = 24.
    const en = buildEotUsc(&buf, 0x2000, 1, PBE_VA);
    try std.testing.expectEqual(@as(usize, 24), en);
    try std.testing.expectEqual(USC_TAG_TEXTURE, buf[0]);
    // the PBE address is encoded shr(3) in the Texture Buffer field.
    const tword = std.mem.readInt(u64, buf[0..8], .little);
    try std.testing.expectEqual(PBE_VA >> 3, tword >> 27);

    // BG USC (R4 fix - NO uniform word; same shape as the FS USC):
    // Shared(4)+Shader(6)+Registers(4)+NoPreshader(2) = 16. The bg USC must NOT
    // bind a uniform the baked clear shader never reads (USC + Counts + shader
    // all agree on "no uniforms").
    const bn = buildBgUsc(&buf, 0x3000, 4);
    try std.testing.expectEqual(@as(usize, 16), bn);
    try std.testing.expectEqual(USC_TAG_SHARED, buf[0]); // no UNIFORM word
    try std.testing.expectEqual(USC_TAG_SHADER, buf[4]);
    try std.testing.expectEqual(USC_TAG_REGISTERS, buf[10]);
    try std.testing.expectEqual(USC_TAG_NO_PRESHADER, buf[14]);
    // the bg is a COMPUTE-style tile writer: shader unk2=0, no Fragment Properties.
    try std.testing.expectEqual(@as(u32, 0), (rdU32(&buf, 4) >> 10) & 0x3F); // unk2=0

    // FS USC: Sampler(8)+Uniform(8)+Shared(4)+Shader(6)+Registers(4)+
    // FragmentProperties(4)+NoPreshader(2) = 36. The FS now carries the MANDATORY
    // graphics root descriptor (reserved-txf SAMPLER + the color UNIFORM Mesa's
    // noddtri FS reads) AND the FRAGMENT PROPERTIES word.
    const fbuf_n = buildFsUsc(&buf, 0x3000, 6, SAMPLER_HEAP_VA, FS_COLOR_VA);
    try std.testing.expectEqual(@as(usize, 36), fbuf_n);
    // (1) reserved txf SAMPLER first.
    try std.testing.expectEqual(USC_TAG_SAMPLER, buf[0]);
    {
        const sw = std.mem.readInt(u64, buf[0..8], .little);
        try std.testing.expectEqual(@as(u64, 1), (sw >> 20) & 0x7F); // count = 1
    }
    // (2) the color UNIFORM at uniform half 16 (regs u8,u9), size 4 halves.
    try std.testing.expectEqual(USC_TAG_UNIFORM, buf[8]);
    {
        const uw = std.mem.readInt(u64, buf[8..16], .little);
        try std.testing.expectEqual(@as(u64, 16), (uw >> 8) & 0xFF); // start half 16
        try std.testing.expectEqual(@as(u64, 4), (uw >> 20) & 0x3F); // size 4 halves
        try std.testing.expectEqual(FS_COLOR_VA >> 2, (uw >> 26) & 0x3FFFFFFFFF);
    }
    try std.testing.expectEqual(USC_TAG_SHARED, buf[16]);
    try std.testing.expectEqual(USC_TAG_SHADER, buf[20]);
    // fragment SHADER unk2 = 2 (Mesa: fragment ? 2 : 3).
    try std.testing.expectEqual(@as(u32, 2), (rdU32(&buf, 20) >> 10) & 0x3F);
    try std.testing.expectEqual(USC_TAG_REGISTERS, buf[26]);
    // fragment REGISTERS unk1 @13 = true, unk4 @24 = 1.
    const fregs = rdU32(&buf, 26);
    try std.testing.expectEqual(@as(u32, 1), (fregs >> 13) & 1);
    try std.testing.expectEqual(@as(u32, 1), (fregs >> 24) & 0xFF);
    // the mandatory FRAGMENT PROPERTIES word.
    try std.testing.expectEqual(USC_TAG_FRAGMENT_PROPERTIES, buf[30]);
    const fprops = rdU32(&buf, 30);
    try std.testing.expectEqual(@as(u32, 1), (fprops >> 8) & 1); // early_z
    try std.testing.expectEqual(@as(u32, 0xF), (fprops >> 12) & 0xF); // unk3=0xf
    try std.testing.expectEqual(@as(u32, 0x2), (fprops >> 16) & 0xFF); // unk4=0x2
    try std.testing.expectEqual(USC_TAG_NO_PRESHADER, buf[34]);
}

test "VDM stream order: barrier, init-ppp, vdm-state, viewport/scissor-ppp, per-draw-ppp, index-list, terminate" {
    var buf: [128]u8 = undefined;
    const n = buildVdmStream(&buf, 0x4000, PPP_VA, 12, INIT_PPP_VA, 6, VPS_PPP_VA, 11);
    // Barrier(4)+InitPPP(8)+VDMState[hdr4+w0+w1+outputs+unknown+pad=24]+VPS-PPP(8)
    // +PerDrawPPP(8)+IndexList[hdr4+count4+inst4+start4=16]+Term(4) = 72.
    try std.testing.expectEqual(@as(usize, 72), n);
    // Barrier block type (bits 29..31) = 1 @0.
    const w0 = std.mem.readInt(u32, buf[0..4], .little);
    try std.testing.expectEqual(@as(u32, VDM_BARRIER), w0 >> 29);
    // The init/W-clamp PPP State Update at offset 4, block type 0, size 6 words.
    const init_ps = std.mem.readInt(u32, buf[4..8], .little);
    try std.testing.expectEqual(@as(u32, VDM_PPP_STATE), init_ps >> 29);
    try std.testing.expectEqual(@as(u32, 6), (init_ps >> 8) & 0xFF);
    try std.testing.expectEqual(@as(u32, @truncate(INIT_PPP_VA & 0xFFFF_FFFF)), std.mem.readInt(u32, buf[8..12], .little));
    // The VDM State Update comes NEXT (the order fix) at offset 12, block type 2.
    const vdm_state = std.mem.readInt(u32, buf[12..16], .little);
    try std.testing.expectEqual(@as(u32, VDM_STATE), vdm_state >> 29);
    // After the VDM State run (24B from off 12 -> 36): the viewport/scissor PPP
    // State Update @36, block type 0, pointing at VPS_PPP_VA.
    const vps_ps = std.mem.readInt(u32, buf[36..40], .little);
    try std.testing.expectEqual(@as(u32, VDM_PPP_STATE), vps_ps >> 29);
    try std.testing.expectEqual(@as(u32, @truncate(VPS_PPP_VA & 0xFFFF_FFFF)), std.mem.readInt(u32, buf[40..44], .little));
    // The per-draw PPP State Update @44, block type 0, pointing at PPP_VA.
    const ps = std.mem.readInt(u32, buf[44..48], .little);
    try std.testing.expectEqual(@as(u32, VDM_PPP_STATE), ps >> 29);
    try std.testing.expectEqual(@as(u32, @truncate(PPP_VA & 0xFFFF_FFFF)), std.mem.readInt(u32, buf[48..52], .little));
    // Index List @52.
    const il = std.mem.readInt(u32, buf[52..56], .little);
    try std.testing.expectEqual(@as(u32, VDM_INDEX_LIST), il >> 29);
    try std.testing.expectEqual(@as(u32, PRIMITIVE_TRIANGLES), (il >> 8) & 0xFF);
    try std.testing.expectEqual(@as(u32, 3), std.mem.readInt(u32, buf[56..60], .little)); // count
    // Terminate is the last word @68.
    const term = std.mem.readInt(u32, buf[68..72], .little);
    try std.testing.expectEqual(@as(u32, VDM_TERMINATE), term >> 29);
}

test "init PPP block is 24 bytes with w_clamp + 3 zero words and correct present bits" {
    var buf: [64]u8 = undefined;
    const n = buildInitPpp(&buf);
    try std.testing.expectEqual(@as(usize, 24), n);
    // Header present bits: w_clamp(16), occlusion_query_2(25), output_unknown(26),
    // varying_word_2(28).
    const hdr = std.mem.readInt(u32, buf[0..4], .little);
    try std.testing.expectEqual(@as(u32, 1), (hdr >> PPP_W_CLAMP) & 1);
    try std.testing.expectEqual(@as(u32, 1), (hdr >> PPP_OCCLUSION_QUERY_2) & 1);
    try std.testing.expectEqual(@as(u32, 1), (hdr >> PPP_OUTPUT_UNKNOWN) & 1);
    try std.testing.expectEqual(@as(u32, 1), (hdr >> PPP_VARYING_WORD_2) & 1);
    // W_CLAMP word = the f32 bits of 1e-10 (Mesa's value).
    try std.testing.expectEqual(fui(1e-10), std.mem.readInt(u32, buf[4..8], .little));
    // The remaining 3 words (occ_q2, output_unknown, varying_2 lo+hi) are zero.
    try std.testing.expectEqual(@as(u32, 0), std.mem.readInt(u32, buf[8..12], .little));
    try std.testing.expectEqual(@as(u32, 0), std.mem.readInt(u32, buf[12..16], .little));
    try std.testing.expectEqual(@as(u64, 0), std.mem.readInt(u64, buf[16..24], .little));
}

test "PBE descriptor encodes linear RGBA8 64x64 with shr(4) buffer and stride-4" {
    var buf: [32]u8 = undefined;
    const n = buildPbe(&buf, COLOR_VA, COLOR_STRIDE, FB_WIDTH, FB_HEIGHT);
    try std.testing.expectEqual(@as(usize, 32), n);
    const lo = std.mem.readInt(u64, buf[0..8], .little);
    try std.testing.expectEqual(@as(u64, 2), lo & 0xF); // Dimension 2D
    try std.testing.expectEqual(@as(u64, 0), (lo >> 4) & 0x3); // Layout Linear
    try std.testing.expectEqual(@as(u64, 0x28), (lo >> 6) & 0x7F); // Channels RGBA8
    try std.testing.expectEqual(@as(u64, FB_WIDTH - 1), (lo >> 24) & 0x3FFF); // Width-1
    // Buffer (bits 64..100) = COLOR_VA >> 4.
    const mid = std.mem.readInt(u64, buf[8..16], .little);
    try std.testing.expectEqual(COLOR_VA >> 4, mid & 0xF_FFFF_FFFF);
    // Stride field at bits 104..125 = stride - 4.
    const hi = std.mem.readInt(u64, buf[12..20], .little);
    const stride = (hi >> (104 - 96)) & 0x1F_FFFF;
    try std.testing.expectEqual(@as(u64, COLOR_STRIDE - 4), stride);
}

test "per-draw PPP header sets present bits, enables scissor, no viewport (own block)" {
    var buf: [256]u8 = undefined;
    const n = buildPpp(&buf, 0x5000);
    // header(4)+FragControl(4)+FragControl2(4)+FrontFace(4)+FrontFace2(4)
    // +BackFace(4)+BackFace2(4)+OutputSelect(4)+VaryingCounts*2(8)+Cull(4)
    // +Cull2(4)+FragShader*4(16)+OutputSize(4) = 68. (Viewport moved to its own
    // PPP block - buildViewportScissorPpp.)
    try std.testing.expectEqual(@as(usize, 68), n);
    const hdr = std.mem.readInt(u32, buf[0..4], .little);
    try std.testing.expectEqual(@as(u32, 1), (hdr >> PPP_FRAGMENT_CONTROL) & 1);
    try std.testing.expectEqual(@as(u32, 1), (hdr >> PPP_FRAGMENT_CONTROL_2) & 1);
    try std.testing.expectEqual(@as(u32, 1), (hdr >> PPP_FRAGMENT_FRONT_FACE_2) & 1);
    try std.testing.expectEqual(@as(u32, 1), (hdr >> PPP_FRAGMENT_BACK_FACE_2) & 1);
    try std.testing.expectEqual(@as(u32, 1), (hdr >> PPP_FRAGMENT_SHADER) & 1);
    try std.testing.expectEqual(@as(u32, 1), (hdr >> PPP_OUTPUT_SELECT) & 1);
    // viewport is NOT in this record now.
    try std.testing.expectEqual(@as(u32, 0), (hdr >> PPP_VIEWPORT) & 1);
    // FRAGMENT_CONTROL (first word after header) has scissor_enable @16 = 1.
    const fc = std.mem.readInt(u32, buf[4..8], .little);
    try std.testing.expectEqual(@as(u32, 1), (fc >> 16) & 1);
    // OUTPUT_SELECT word is 0 (position-only VS, no varyings) - the Mesa-correct
    // value for our shaders (osel.varyings = nr_cf_bindings>0 = 0).
    try std.testing.expectEqual(@as(u32, 0), std.mem.readInt(u32, buf[28..32], .little));
}

test "viewport/scissor PPP block: present bits, region clip enabled, viewport maps to FB" {
    var buf: [64]u8 = undefined;
    const n = buildViewportScissorPpp(&buf, FB_WIDTH, FB_HEIGHT);
    // header(4)+DepthBiasScissor(4)+RegionClip(8)+ViewportControl(4)+Viewport(24) = 44.
    try std.testing.expectEqual(@as(usize, 44), n);
    const hdr = std.mem.readInt(u32, buf[0..4], .little);
    try std.testing.expectEqual(@as(u32, 1), (hdr >> PPP_DEPTH_BIAS_SCISSOR) & 1);
    try std.testing.expectEqual(@as(u32, 1), (hdr >> PPP_REGION_CLIP) & 1);
    try std.testing.expectEqual(@as(u32, 1), (hdr >> PPP_VIEWPORT) & 1);
    // REGION_CLIP @8: Enable @31 = 1, Max X-1 = DIV_ROUND_UP(64,32)-1 = 1.
    const rc = std.mem.readInt(u32, buf[8..12], .little);
    try std.testing.expectEqual(@as(u32, 1), (rc >> 31) & 1);
    try std.testing.expectEqual(@as(u32, 1), rc & 0x1FF);
    // VIEWPORT @20: scale_x = FB_WIDTH/2 = 32.0.
    const scale_x: f32 = @bitCast(std.mem.readInt(u32, buf[24..28], .little));
    try std.testing.expectEqual(@as(f32, 32.0), scale_x);
}

test "scissor + depth-bias array entries are well sized" {
    var sbuf: [32]u8 = undefined;
    try std.testing.expectEqual(@as(usize, 16), buildScissor(&sbuf, FB_WIDTH, FB_HEIGHT));
    // Max X @0(16) = FB_WIDTH, Max Y @32(16) = FB_HEIGHT.
    try std.testing.expectEqual(@as(u16, FB_WIDTH), std.mem.readInt(u16, sbuf[0..2], .little));
    try std.testing.expectEqual(@as(u16, FB_HEIGHT), std.mem.readInt(u16, sbuf[4..6], .little));
    var dbuf: [16]u8 = undefined;
    try std.testing.expectEqual(@as(usize, 12), buildDepthBias(&dbuf));
}

test "size-dependent fields scale to an arbitrary 256x256 framebuffer" {
    const W: u32 = 256;
    const H: u32 = 256;

    // PBE: Width/Height minus(1) = 255; Stride = W*4 - 4.
    var pbe: [32]u8 = undefined;
    _ = buildPbe(&pbe, COLOR_VA, W * 4, W, H);
    const lo = std.mem.readInt(u64, pbe[0..8], .little);
    try std.testing.expectEqual(@as(u64, W - 1), (lo >> 24) & 0x3FFF); // Width-1
    try std.testing.expectEqual(@as(u64, H - 1), (lo >> 38) & 0x3FFF); // Height-1
    const hi = std.mem.readInt(u64, pbe[12..20], .little);
    const stride = (hi >> (104 - 96)) & 0x1F_FFFF;
    try std.testing.expectEqual(@as(u64, W * 4 - 4), stride);

    // Scissor: Max X = W, Max Y = H.
    var sb: [16]u8 = undefined;
    _ = buildScissor(&sb, W, H);
    try std.testing.expectEqual(@as(u16, @intCast(W)), std.mem.readInt(u16, sb[0..2], .little)); // Max X @0
    try std.testing.expectEqual(@as(u16, @intCast(H)), std.mem.readInt(u16, sb[4..6], .little)); // Max Y @32 bits = byte 4

    // Viewport/scissor PPP: region-clip Max X-1 = DIV_ROUND_UP(256,32)-1 = 7;
    // viewport scale_x = W/2 = 128.0.
    var vps: [64]u8 = undefined;
    _ = buildViewportScissorPpp(&vps, W, H);
    const rc = std.mem.readInt(u32, vps[8..12], .little);
    try std.testing.expectEqual(@as(u32, (W + 31) / 32 - 1), rc & 0x1FF); // 7
    const scale_x: f32 = @bitCast(std.mem.readInt(u32, vps[24..28], .little));
    try std.testing.expectEqual(@as(f32, 128.0), scale_x);

    // cmd_render: width_px/height_px = W/H; isp_merge per-axis = tan60/dim.
    var cb: [
        @sizeOf(uapi.drm_asahi_cmd_header) * 2 +
            @sizeOf(uapi.drm_asahi_attachment) +
            @sizeOf(uapi.drm_asahi_cmd_render)
    ]u8 = undefined;
    const bg = uapi.drm_asahi_bg_eot{ .usc = 0x6000 | 4, .rsrc_spec = 0 };
    const eot = uapi.drm_asahi_bg_eot{ .usc = 0x7000 | 4, .rsrc_spec = 0 };
    _ = buildCmdbuf(&cb, VDM_STREAM_VA, 56, COLOR_VA, @as(u64, W) * 4 * H, bg, eot, SCISSOR_VA, DBIAS_VA, W, H);
    const body: *const uapi.drm_asahi_cmd_render = @ptrCast(@alignCast(&cb[8 + 24 + 8]));
    try std.testing.expectEqual(@as(u16, @intCast(W)), body.width_px);
    try std.testing.expectEqual(@as(u16, @intCast(H)), body.height_px);
    try std.testing.expectEqual(fui_rt(1.732051 / @as(f32, 256.0)), body.isp_merge_upper_x);
    try std.testing.expectEqual(fui_rt(1.732051 / @as(f32, 256.0)), body.isp_merge_upper_y);
    // utile config stays size-independent (32x32).
    try std.testing.expectEqual(@as(u8, UTILE_W), body.utile_width_px);
}

test "Counts word: bg sets the unknown field, eot encodes 1 texture" {
    const bg_counts = buildCounts(0, true);
    const eot_counts = buildCounts(1, false);
    // bg has the high "Unknown" 0xFFFF field set.
    try std.testing.expectEqual(@as(u32, 0xFFFF), (bg_counts >> 16) & 0xFFFF);
    // eot does not set it.
    try std.testing.expectEqual(@as(u32, 0), (eot_counts >> 16) & 0xFFFF);
    // eot texture state register count (bits 4..9) groups(8): 1 -> 1.
    try std.testing.expectEqual(@as(u32, 1), (eot_counts >> 4) & 0x1F);
}

test "cmdbuf is SET_FRAGMENT_ATTACHMENTS + RENDER with correct headers" {
    var buf: [
        @sizeOf(uapi.drm_asahi_cmd_header) * 2 +
            @sizeOf(uapi.drm_asahi_attachment) +
            @sizeOf(uapi.drm_asahi_cmd_render)
    ]u8 = undefined;
    const bg = uapi.drm_asahi_bg_eot{ .usc = 0x6000 | 4, .rsrc_spec = 0 };
    const eot = uapi.drm_asahi_bg_eot{ .usc = 0x7000 | 4, .rsrc_spec = 0 };
    const n = buildCmdbuf(&buf, VDM_STREAM_VA, 56, COLOR_VA, COLOR_SIZE, bg, eot, SCISSOR_VA, DBIAS_VA, FB_WIDTH, FB_HEIGHT);
    // 8 (set hdr) + 24 (attachment) + 8 (render hdr) + 240 (render body) = 280.
    try std.testing.expectEqual(@as(usize, 8 + 24 + 8 + 240), n);
    // isp_scissor_base / isp_dbias_base are now wired (non-null).
    const body0: *const uapi.drm_asahi_cmd_render = @ptrCast(@alignCast(&buf[8 + 24 + 8]));
    try std.testing.expectEqual(SCISSOR_VA, body0.isp_scissor_base);
    try std.testing.expectEqual(DBIAS_VA, body0.isp_dbias_base);

    const set_hdr: *const uapi.drm_asahi_cmd_header = @ptrCast(@alignCast(&buf[0]));
    try std.testing.expectEqual(uapi.SET_FRAGMENT_ATTACHMENTS, set_hdr.cmd_type);
    try std.testing.expectEqual(uapi.BARRIER_NONE, set_hdr.vdm_barrier);

    const ren_hdr: *const uapi.drm_asahi_cmd_header = @ptrCast(@alignCast(&buf[8 + 24]));
    try std.testing.expectEqual(uapi.CMD_RENDER, ren_hdr.cmd_type);
    try std.testing.expectEqual(@as(u16, @sizeOf(uapi.drm_asahi_cmd_render)), ren_hdr.size);

    const body: *const uapi.drm_asahi_cmd_render = @ptrCast(@alignCast(&buf[8 + 24 + 8]));
    try std.testing.expectEqual(VDM_STREAM_VA, body.vdm_ctrl_stream_base);
    try std.testing.expectEqual(@as(u16, FB_WIDTH), body.width_px);
    try std.testing.expectEqual(@as(u8, UTILE_W), body.utile_width_px);
    try std.testing.expectEqual(@as(u8, SAMPLE_SIZE_B), body.sample_size_B);
    // PROCESS_EMPTY_TILES must be set (we clear).
    try std.testing.expect((body.flags & uapi.RENDER_PROCESS_EMPTY_TILES) != 0);
    try std.testing.expectEqual(@as(u64, PPP_MULTISAMPLECTL_1), body.ppp_multisamplectl);
    try std.testing.expectEqual(PPP_CTRL, body.ppp_ctrl);
    try std.testing.expectEqual(bg.usc, body.bg.usc);
    try std.testing.expectEqual(eot.usc, body.eot.usc);
}

test "VA layout is PAGE-aligned, distinct, and inside the USC window" {
    const vas = [_]u64{
        VS_CODE_VA,    PS_CODE_VA, EOT_CODE_VA,   VS_USC_VA,  PS_USC_VA,
        BG_USC_VA,     EOT_USC_VA, VDM_STREAM_VA, PPP_VA,     PBE_VA,
        CLEAR_DATA_VA, BG_CODE_VA, INIT_PPP_VA,   SCISSOR_VA, DBIAS_VA,
        VPS_PPP_VA,
    };
    for (vas) |va| {
        try std.testing.expectEqual(@as(u64, 0), va % PAGE);
        // inside the 4 GiB USC window (offset fits in 32 bits).
        try std.testing.expect(va - USC_EXEC_BASE < 0x1_0000_0000);
    }
    // the color BO is above the USC window.
    try std.testing.expect(COLOR_VA >= USC_EXEC_BASE + 0x1_0000_0000);
    try std.testing.expectEqual(@as(u64, 0), COLOR_VA % PAGE);
}

test "dumpStream produces decoded output for both draw and clear streams" {
    var buf: [16 * 1024]u8 = undefined;
    // Triangle stream: must mention the VDM stream, the cmd_render fields, the
    // helper fields (the geometry suspects), and the per-draw PPP record.
    {
        var fw = std.Io.Writer.fixed(buf[0..]);
        dumpStream(&fw, true);
        const out = fw.buffered();
        try std.testing.expect(std.mem.indexOf(u8, out, "VDM control stream") != null);
        try std.testing.expect(std.mem.indexOf(u8, out, "drm_asahi_cmd_render") != null);
        try std.testing.expect(std.mem.indexOf(u8, out, "vertex_helper") != null);
        try std.testing.expect(std.mem.indexOf(u8, out, "fragment_helper") != null);
        try std.testing.expect(std.mem.indexOf(u8, out, "Index List") != null);
        try std.testing.expect(std.mem.indexOf(u8, out, "Stream Terminate") != null);
        try std.testing.expect(std.mem.indexOf(u8, out, "per-draw PPP record") != null);
        try std.testing.expect(std.mem.indexOf(u8, out, "W_CLAMP") != null);
    }
    // Clear-only stream: a Barrier + Terminate VDM, no per-draw PPP.
    {
        var fw = std.Io.Writer.fixed(buf[0..]);
        dumpStream(&fw, false);
        const out = fw.buffered();
        try std.testing.expect(std.mem.indexOf(u8, out, "Barrier") != null);
        try std.testing.expect(std.mem.indexOf(u8, out, "Stream Terminate") != null);
        try std.testing.expect(std.mem.indexOf(u8, out, "per-draw PPP record") == null);
    }
}
