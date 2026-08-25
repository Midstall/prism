//! End-to-end SPIR-V compute path on a real NVIDIA GPU: SPIR-V binary parsed to
//! Vulcan IR (prism.spirv.parseSpirv), instruction-selected to SASS
//! (vulcan-target.nvidia.isel.compileKernel), dispatched as a 1x1x1 grid via
//! QMDV05_00 on the from-scratch RM driver. Result read back from GPU memory.
//! Proves all three layers work together (SPIR-V front end, Vulcan NVIDIA backend,
//! subproject compute dispatch). Parameters go through the QMD constant buffer
//! (bank 0, param_base). The kernel prologue reads them via LDC c[0][off].
//! Vulcan emits correct Blackwell non-uniform integer-ALU opcodes and schedules
//! LDC as variable-latency, so no fixup is needed here.

const std = @import("std");
const gpumem = @import("gpumem.zig");
const nvidia = @import("nvidia");
const isel = @import("vulcan-target").nvidia.isel;
const spirv = @import("../../spirv.zig");
const NvDevice = @import("device.zig").Device;
const nverr = @import("device.zig").nverr;

const sdk = nvidia.sdk;
const compute = nvidia.compute;

/// Write the kernel-ABI parameters into a constant buffer at the offset where the
/// kernel reads them (`param_base`): the 64-bit output pointer first, then each
/// 32-bit scalar argument in order. This matches the LDC prologue emitted by
/// vulcan-target/nvidia/isel.zig (out-ptr lo/hi, then one word per arg).
fn writeParams(cbuf: []u8, out_va: u64, args: []const i32) void {
    const words = std.mem.bytesAsSlice(u32, cbuf);
    var w: usize = isel.param_base / 4; // param_base is 4-byte aligned
    words[w] = @truncate(out_va); // output ptr low
    words[w + 1] = @truncate(out_va >> 32); // output ptr high
    w += 2;
    for (args) |a| {
        words[w] = @bitCast(a);
        w += 1;
    }
}

/// Run a SPIR-V compute kernel that returns a 32-bit value computed from the two
/// 32-bit integer parameters `x` and `y`, and return the value the GPU stored.
/// The kernel must match the value-returning ABI (one output pointer + two int
/// params in constant bank 0). It is launched as a single thread (1x1x1 grid,
/// 1x1x1 block).
pub fn runIntBinaryKernel(dev: *NvDevice, spirv_code: []const u8, x: i32, y: i32) !i32 {
    const gpa = dev.gpa;

    // SPIR-V -> Vulcan IR -> SASS compute kernel.
    var func = try spirv.parseSpirv(gpa, spirv_code);
    defer func.deinit();
    var kernel = try isel.compileKernel(gpa, &func);
    defer kernel.deinit(gpa);

    // GPU memory: the kernel code, the output word, and the parameter constant
    // buffer. allocGpu maps each into the GPU address space and CPU-maps it. freeGpu
    // releases it.
    const codem = try dev.allocGpu(.system_wc, 0x1000);
    defer dev.freeGpu(codem);
    const outm = try dev.allocGpu(.system, 0x1000);
    defer dev.freeGpu(outm);
    const cbufm = try dev.allocGpu(.system_wc, 0x1000);
    defer dev.freeGpu(cbufm);

    // Upload the SASS unmodified.
    gpumem.writeWords(codem.bytes, kernel.code);

    // Write the kernel-ABI parameters (output pointer + the two ints) into the
    // constant buffer at param_base. The kernel reads them via LDC c[0][off]. The
    // buffer must cover up to the highest offset read (param_base + 0x10 here).
    const args = [_]i32{ x, y };
    gpumem.zero(cbufm.bytes, cbufm.bytes.len);
    writeParams(cbufm.bytes, outm.va, &args);
    const cbuf_size: u32 = isel.param_base + 0x10;

    // Zero the output so the read-back is meaningful.
    const outp: *volatile i32 = @ptrCast(@alignCast(outm.bytes.ptr));
    outp.* = 0;

    // Build the QMD: a 1x1x1 grid of a 1x1x1 block at the kernel's register count,
    // with constant bank 0 bound to the parameter buffer. QMDV05_00 is 96 dwords
    // (384 bytes). buildQmd fills the constant-buffer fields up to dword 58.
    // The rest must be zeroed because the hardware reads the full descriptor.
    const QMD_DWORDS_V05: usize = 96;
    var qmd: [QMD_DWORDS_V05]u32 = [_]u32{0} ** QMD_DWORDS_V05;
    compute.buildQmd(qmd[0..compute.QMD_DWORDS], .{
        .prog_va = codem.va,
        .register_count = kernel.reg_count,
        .cbuf0_va = cbufm.va,
        .cbuf0_size = cbuf_size,
    });
    const qmdm = try dev.allocGpu(.system, 0x1000);
    defer dev.freeGpu(qmdm);
    gpumem.writeWords(qmdm.bytes, &qmd);

    // Channel infrastructure: USERD, GPFIFO ring, pushbuffer, completion sema.
    const userd = try dev.allocGpu(.vram, 0x1000);
    defer dev.freeGpu(userd);
    const gpfifo = try dev.allocGpu(.vram, 0x2000);
    defer dev.freeGpu(gpfifo);
    const pbuf = try dev.allocGpu(.system, 0x1000);
    defer dev.freeGpu(pbuf);
    const sem = try dev.allocGpu(.system, 0x1000);
    defer dev.freeGpu(sem);

    const ch = dev.client.allocChannel(dev.dev, sdk.BLACKWELL_CHANNEL_GPFIFO_B, dev.vaspace, gpfifo.va, 0x100, userd.mem) catch |e| return nverr(e);
    defer dev.client.rmFree(dev.dev.client, dev.dev.device, ch.handle);
    const object = dev.client.allocObject(dev.dev, ch, compute.BLACKWELL_COMPUTE_B) catch |e| return nverr(e);
    defer dev.client.rmFree(dev.dev.client, ch.handle, object);
    dev.client.bindChannel(dev.dev, ch, sdk.NV2080_ENGINE_TYPE_GRAPHICS) catch |e| return nverr(e);
    dev.client.scheduleChannel(dev.dev, ch, true) catch |e| return nverr(e);
    const token = dev.client.workSubmitToken(dev.dev, ch) catch |e| return nverr(e);
    const usermode = dev.client.allocUsermode(dev.dev, sdk.BLACKWELL_USERMODE_A) catch |e| return nverr(e);
    defer dev.client.rmFree(dev.dev.client, dev.dev.subdevice, usermode);
    const door = dev.client.mapMemory(dev.dev, .{ .handle = usermode, .size = 0x1000, .location = .vram }) catch |e| return nverr(e);
    // Release the doorbell CPU mapping before freeing the usermode object (LIFO):
    // the usermode aperture is limited, so leaking it exhausts the aperture across
    // repeated dispatches (NV status 0x36 on the next usermode alloc).
    defer dev.client.unmapMemory(door);

    // Build the compute launch method stream (subchannel 1): bind the compute
    // class + shader memory windows, dispatch the QMD, fence on a semaphore.
    const FENCE: u32 = 0xC0DE;
    var s = compute.Stream{ .buf = @as([*]u32, @ptrCast(@alignCast(pbuf.bytes.ptr)))[0 .. pbuf.bytes.len / 4] };
    s.setup();
    s.dispatch(qmdm.va);
    s.fence(sem.va, FENCE);

    const semp: *volatile u32 = @ptrCast(@alignCast(sem.bytes.ptr));
    semp.* = 0;
    var q = nvidia.Queue{ .channel = ch, .token = token, .userd = userd.bytes, .gpfifo = gpfifo.bytes, .doorbell = door.bytes };
    q.submit(pbuf.va, s.dwords());

    var spins: u64 = 0;
    while (spins < 500_000_000) : (spins += 1) {
        if (semp.* == FENCE) return outp.*;
    }
    return error.DeviceLost; // the grid never completed
}

test "SPIR-V compute kernel (x*y - x) runs on the NVIDIA GPU as SASS (skips without a GPU)" {
    const gpa = std.testing.allocator;
    const dev = NvDevice.create(gpa) catch return error.SkipZigTest;
    defer dev.deinit();
    const self: *NvDevice = @ptrCast(@alignCast(dev.ptr));

    // int f(int x, int y) { return x*y - x; }. Byte-for-byte the builder
    // sequence from lib/prism/spirv.zig's own test. f(3,4) = 3*4 - 3 = 9.
    const op = @import("vulcan-spirv").opcodes;
    var b = try @import("vulcan-spirv").binary.Builder.init(gpa, 9);
    defer b.deinit(gpa);
    try b.emit(gpa, op.TypeInt, &.{ 1, 32, 1 });
    try b.emit(gpa, op.TypeFunction, &.{ 2, 1, 1, 1 });
    try b.emit(gpa, op.Function, &.{ 1, 3, 0, 2 });
    try b.emit(gpa, op.FunctionParameter, &.{ 1, 4 });
    try b.emit(gpa, op.FunctionParameter, &.{ 1, 5 });
    try b.emit(gpa, op.Label, &.{6});
    try b.emit(gpa, op.IMul, &.{ 1, 7, 4, 5 });
    try b.emit(gpa, op.ISub, &.{ 1, 8, 7, 4 });
    try b.emit(gpa, op.ReturnValue, &.{8});
    try b.emit(gpa, op.FunctionEnd, &.{});

    const result = try runIntBinaryKernel(self, std.mem.sliceAsBytes(b.words.items), 3, 4);
    try std.testing.expectEqual(@as(i32, 9), result);
}
