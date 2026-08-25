//! CE detile path for the nvidia HAL driver. Owns a GPFIFO channel bound to the
//! async copy engine (NV2080_ENGINE_TYPE_COPY0, class BLACKWELL_DMA_COPY_B 0xCAB5)
//! and submits a LAUNCH_DMA that detiles a block-linear color RT in VRAM into a
//! pitch-linear destination buffer on the GPU.
//!
//! GPU-side replacement for device.deswizzleBlockLinear/graphics.blColorPixelOffset.
//! Byte-for-byte identical to the CPU ground truth. Wiring into the present path is
//! a later milestone.
//!
//! Channel bring-up mirrors context.zig but binds the COPY engine (no MME/3D-state
//! init needed). Everything alloced is freed on deinit, matching context.zig.

const std = @import("std");
const nvidia = @import("nvidia");
const hal = @import("../../hal.zig");
const Resource = @import("resource.zig").Resource;
const NvDevice = @import("device.zig").Device;
const nverr = @import("device.zig").nverr;

const sdk = nvidia.sdk;
const copy = nvidia.copy;

/// A copy-engine command context: a GPFIFO channel on the async CE, a USERMODE
/// doorbell, a pushbuffer, and a completion semaphore.
pub const CopyEngine = struct {
    dev: *NvDevice,
    channel: nvidia.Channel,
    object: sdk.NvHandle,
    usermode: sdk.NvHandle,
    userd: NvDevice.GpuBuf,
    gpfifo: NvDevice.GpuBuf,
    pbuf: NvDevice.GpuBuf,
    sem: NvDevice.GpuBuf,
    queue: nvidia.Queue,
    doorbell_map: nvidia.Mapping,
    /// The engine index that actually bound + scheduled (COPY0..COPY4). 0 = COPY0.
    engine_index: u32,

    const FENCE: u32 = 0xCE0F;

    pub fn create(dev: *NvDevice) hal.Error!*CopyEngine {
        const self = dev.gpa.create(CopyEngine) catch return error.OutOfMemory;
        errdefer dev.gpa.destroy(self);
        self.dev = dev;

        self.userd = try dev.allocGpu(.vram, 0x1000);
        errdefer dev.freeGpu(self.userd);
        self.gpfifo = try dev.allocGpu(.vram, 0x2000);
        errdefer dev.freeGpu(self.gpfifo);
        self.pbuf = try dev.allocGpu(.system, 0x4000);
        errdefer dev.freeGpu(self.pbuf);
        self.sem = try dev.allocGpu(.system, 0x1000);
        errdefer dev.freeGpu(self.sem);

        // Try the copy engines in order (COPY0..COPY4). On a contended/headless GPU
        // not every CE index may accept a channel. Whichever binds + schedules is
        // recorded in engine_index. The channel must be allocated on the same engine
        // runlist it is bound to, so re-alloc per attempt.
        var engine_idx: u32 = 0;
        var bound = false;
        var channel: nvidia.Channel = undefined;
        var object: sdk.NvHandle = 0;
        while (engine_idx < 5) : (engine_idx += 1) {
            const engine_type = sdk.NV2080_ENGINE_TYPE_COPY0 + engine_idx;
            channel = dev.client.allocChannelEngine(dev.dev, sdk.BLACKWELL_CHANNEL_GPFIFO_B, dev.vaspace, self.gpfifo.va, 0x100, self.userd.mem, engine_type) catch continue;
            object = dev.client.allocObject(dev.dev, channel, sdk.BLACKWELL_DMA_COPY_B) catch {
                dev.client.rmFree(dev.dev.client, dev.dev.device, channel.handle);
                continue;
            };
            dev.client.bindChannel(dev.dev, channel, engine_type) catch {
                dev.client.rmFree(dev.dev.client, channel.handle, object);
                dev.client.rmFree(dev.dev.client, dev.dev.device, channel.handle);
                continue;
            };
            dev.client.scheduleChannel(dev.dev, channel, true) catch {
                dev.client.rmFree(dev.dev.client, channel.handle, object);
                dev.client.rmFree(dev.dev.client, dev.dev.device, channel.handle);
                continue;
            };
            bound = true;
            break;
        }
        if (!bound) return error.InitializationFailed;
        self.channel = channel;
        self.object = object;
        self.engine_index = engine_idx;
        errdefer {
            dev.client.rmFree(dev.dev.client, self.channel.handle, self.object);
            dev.client.rmFree(dev.dev.client, dev.dev.device, self.channel.handle);
        }

        const token = dev.client.workSubmitToken(dev.dev, self.channel) catch |e| return nverr(e);
        self.usermode = dev.client.allocUsermode(dev.dev, sdk.BLACKWELL_USERMODE_A) catch |e| return nverr(e);
        errdefer dev.client.rmFree(dev.dev.client, dev.dev.subdevice, self.usermode);
        const door = dev.client.mapMemory(dev.dev, .{ .handle = self.usermode, .size = 0x1000, .location = .vram }) catch |e| return nverr(e);
        self.doorbell_map = door;
        self.queue = .{ .channel = self.channel, .token = token, .userd = self.userd.bytes, .gpfifo = self.gpfifo.bytes, .doorbell = door.bytes };
        return self;
    }

    /// Detile a block-linear color render target `src` into the pitch-linear
    /// destination `dst` (a sysmem GpuBuf), `dst_pitch` bytes per row. The CE
    /// copies bytes verbatim (A8R8G8B8 -> [B,G,R,A]). The caller reads dst.bytes
    /// after this returns. Blocks until the copy fences.
    pub fn detile(self: *CopyEngine, src: *Resource, dst: NvDevice.GpuBuf, dst_pitch: u32) hal.Error!void {
        var s = copy.Stream{ .buf = @alignCast(std.mem.bytesAsSlice(u32, self.pbuf.bytes)) };
        s.setup();
        s.detile(.{
            .src_va = src.gpu_va,
            .dst_va = dst.va,
            .width = src.width,
            .height = src.height,
            .dst_pitch = dst_pitch,
            .src_block_height_gobs = nvidia.graphics.ZT_BLOCK_HEIGHT_GOBS,
            .sem_va = self.sem.va,
            .sem_seq = FENCE,
        });

        const semp: *volatile u32 = @ptrCast(@alignCast(self.sem.bytes.ptr));
        semp.* = 0;
        self.queue.submit(self.pbuf.va, s.dwords());
        return waitFence(semp);
    }

    fn waitFence(semp: *volatile u32) hal.Error!void {
        var spins: u64 = 0;
        const start = nowNs();
        while (true) {
            if (semp.* == FENCE) return;
            spins += 1;
            if (spins >= 8192) {
                var req = std.os.linux.timespec{ .sec = 0, .nsec = 50_000 };
                _ = std.os.linux.nanosleep(&req, null);
                if (nowNs() - start > 5 * std.time.ns_per_s) return error.DeviceLost;
            }
        }
    }

    fn nowNs() u64 {
        var ts: std.os.linux.timespec = undefined;
        _ = std.os.linux.clock_gettime(.MONOTONIC, &ts);
        return @as(u64, @intCast(ts.sec)) * 1_000_000_000 + @as(u64, @intCast(ts.nsec));
    }

    pub fn deinit(self: *CopyEngine) void {
        const dev = self.dev;
        dev.client.rmFree(dev.dev.client, self.channel.handle, self.object);
        dev.client.unmapMemory(self.doorbell_map);
        dev.client.rmFree(dev.dev.client, dev.dev.subdevice, self.usermode);
        dev.client.rmFree(dev.dev.client, dev.dev.device, self.channel.handle);
        dev.freeGpu(self.sem);
        dev.freeGpu(self.pbuf);
        dev.freeGpu(self.gpfifo);
        dev.freeGpu(self.userd);
        dev.gpa.destroy(self);
    }
};

test "nvidia CE detile of a block-linear RT == the CPU de-swizzle byte-for-byte (skips without a GPU)" {
    const gpa = std.testing.allocator;
    const dev = NvDevice.create(gpa) catch return error.SkipZigTest;
    defer dev.deinit();
    const self: *NvDevice = @ptrCast(@alignCast(dev.ptr));

    // 800x600 is the glmark2 present size: 800*4=3200 is GOB-row-aligned but 600 is NOT
    // a multiple of the 128-row (16-GOB) block height. It exercises the partial last
    // block. A CE detile bug there scatters pixels (looks like a broken model on present).
    const W: u32 = 800;
    const H: u32 = 600;

    // Render a gradient triangle into a block-linear color RT, so the surface
    // holds a non-trivial GOB-tiled pattern (a flat clear could hide a swizzle bug).
    const sass = nvidia.sass;
    var vs_code: [256]u32 = undefined;
    var vsa = sass.Assembler{ .code = &vs_code };
    {
        var p: u32 = 0;
        while (p < 6) : (p += 1) vsa.movImm(12, p, .{});
    }
    vsa.ald(4, sass.ATTR_GENERIC0, 4, .{ .wr_barrier = 0 });
    vsa.ald(8, sass.ATTR_GENERIC0 + 0x10, 4, .{ .wr_barrier = 1 });
    vsa.ast(sass.ATTR_POSITION, 4, 4, .{ .wait_mask = 1 });
    vsa.ast(sass.ATTR_GENERIC0, 8, 4, .{ .wait_mask = 2 });
    vsa.exit(.{ .stall = 1 });
    var ps_code: [256]u32 = undefined;
    var psa = sass.Assembler{ .code = &ps_code };
    {
        var p: u32 = 0;
        while (p < 6) : (p += 1) psa.movImm(15, p, .{});
    }
    psa.ipa(4, sass.ATTR_GENERIC0, .{ .wr_barrier = 0 });
    psa.ipa(5, sass.ATTR_GENERIC0 + 4, .{ .wr_barrier = 1 });
    psa.ipa(6, sass.ATTR_GENERIC0 + 8, .{ .wr_barrier = 2 });
    psa.ipa(7, sass.ATTR_GENERIC0 + 12, .{ .wr_barrier = 3 });
    psa.movReg(0, 4, .{ .wait_mask = 0xf });
    psa.movReg(1, 5, .{});
    psa.movReg(2, 6, .{});
    psa.movReg(3, 7, .{});
    psa.exit(.{ .stall = 15 });

    const vs = try dev.createShaderModule(.{ .stage = .vertex, .code = std.mem.sliceAsBytes(vs_code[0..vsa.dwords()]) });
    defer dev.destroyShaderModule(vs);
    const ps = try dev.createShaderModule(.{ .stage = .fragment, .code = std.mem.sliceAsBytes(ps_code[0..psa.dwords()]) });
    defer dev.destroyShaderModule(ps);

    const rt = try dev.createResource(.{ .image = .{ .width = W, .height = H, .format = .bgra8_unorm, .usage = .{ .render_target = true } } });
    defer dev.destroyResource(rt);
    const vbuf = try dev.createResource(.{ .buffer = .{ .size = 3 * 32, .usage = .{ .vertex = true } } });
    defer dev.destroyResource(vbuf);
    const verts = [_]f32{
        0.0,  0.5,  0.0, 1.0, 1.0, 0.0, 0.0, 1.0,
        -0.5, -0.5, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0,
        0.5,  -0.5, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0,
    };
    @memcpy(std.mem.bytesAsSlice(f32, try dev.mapResource(vbuf))[0..verts.len], &verts);
    const attrs = [_]hal.VertexAttribute{
        .{ .location = 0, .format = .rgba8_unorm, .offset = 0 },
        .{ .location = 1, .format = .rgba8_unorm, .offset = 16 },
    };
    const pipe = try dev.createPipeline(.{
        .vertex = vs,
        .fragment = ps,
        .vertex_layout = .{ .stride = 32, .attributes = &attrs },
        .color_format = .bgra8_unorm,
    });
    defer dev.destroyPipeline(pipe);

    const ctx = try dev.createContext();
    defer ctx.deinit();
    const cb = try ctx.beginCommands();
    defer cb.deinit();
    try cb.setRenderTarget(rt);
    try cb.clear(.{ .r = 0.125, .g = 0.125, .b = 0.125, .a = 1 });
    try cb.bindPipeline(pipe);
    try cb.bindVertexBuffer(vbuf);
    try cb.draw(3, 0);
    try ctx.submit(cb);

    const rt_res: *Resource = @ptrCast(@alignCast(rt));

    // Path A (ground truth): the CPU de-swizzle, called DIRECTLY.
    //
    // It used to read `dev.mapResource(rt)` and rely on that returning the CPU
    // de-swizzle. `mapResource` now detiles on the CE itself, which is the whole
    // point of having one, so going through it would compare the copy engine
    // against the copy engine and assert nothing at all. The oracle only means
    // something while the two paths are genuinely independent.
    //
    // The RT is .bgra8_unorm, so the no-swap form (bgra_straight) is the exact
    // byte order the CE copies verbatim. `mapResource` is still called first,
    // for its side effect of establishing the CPU mapping the de-swizzle reads.
    _ = try dev.mapResource(rt);
    const ground = try gpa.alloc(u8, @as(usize, W) * H * 4);
    defer gpa.free(ground);
    try self.deswizzleBlockLinear(rt_res, ground, @as(usize, W) * 4, .bgra_straight, 4);

    // Path B (CE): detile the SAME block-linear RT into a pitch-linear sysmem buffer.
    const dst_pitch: u32 = W * 4;
    const dst = try self.allocGpu(.system, @as(u64, dst_pitch) * H);
    defer self.freeGpu(dst);
    @memset(dst.bytes[0 .. @as(usize, dst_pitch) * H], 0);

    const ce = CopyEngine.create(self) catch return error.SkipZigTest;
    defer ce.deinit();
    try ce.detile(rt_res, dst, dst_pitch);

    // Byte-for-byte equality of the visible W*H*4 image.
    const ce_out = dst.bytes[0 .. @as(usize, dst_pitch) * H];
    try std.testing.expectEqualSlices(u8, ground, ce_out);
}
