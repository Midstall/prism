const std = @import("std");
const asahi = @import("asahi");
const hal = @import("../../hal.zig");
const Resource = @import("resource.zig").Resource;
const AppleDevice = @import("device.zig").Device;
const aserr = @import("device.zig").aserr;
const Surface = @import("surface.zig").Surface;

/// A recorded command. The command buffer appends these. GPU work is built and
/// replayed in Context.submit (mirroring the nvidia / software drivers).
const Command = union(enum) {
    set_render_target: *Resource,
    clear: hal.Color,
    bind_pipeline: *hal.Pipeline,
    bind_vertex_buffer: *Resource,
    draw: struct { vertex_count: u32, first_vertex: u32 },
};

pub const CommandBuffer = struct {
    gpa: std.mem.Allocator,
    cmds: std.ArrayListUnmanaged(Command) = .empty,

    fn create(gpa: std.mem.Allocator) hal.Error!hal.CommandBuffer {
        const self = gpa.create(CommandBuffer) catch return error.OutOfMemory;
        self.* = .{ .gpa = gpa };
        return .{ .ptr = self, .vtable = &vtable };
    }

    fn setRenderTarget(ptr: *anyopaque, target: *hal.Resource) hal.Error!void {
        const self: *CommandBuffer = @ptrCast(@alignCast(ptr));
        self.cmds.append(self.gpa, .{ .set_render_target = @ptrCast(@alignCast(target)) }) catch return error.OutOfMemory;
    }
    fn clear(ptr: *anyopaque, color: hal.Color) hal.Error!void {
        const self: *CommandBuffer = @ptrCast(@alignCast(ptr));
        self.cmds.append(self.gpa, .{ .clear = color }) catch return error.OutOfMemory;
    }
    fn bindPipeline(ptr: *anyopaque, pipeline: *hal.Pipeline) hal.Error!void {
        const self: *CommandBuffer = @ptrCast(@alignCast(ptr));
        self.cmds.append(self.gpa, .{ .bind_pipeline = pipeline }) catch return error.OutOfMemory;
    }
    fn bindVertexBuffer(ptr: *anyopaque, buffer: *hal.Resource) hal.Error!void {
        const self: *CommandBuffer = @ptrCast(@alignCast(ptr));
        self.cmds.append(self.gpa, .{ .bind_vertex_buffer = @ptrCast(@alignCast(buffer)) }) catch return error.OutOfMemory;
    }
    fn draw(ptr: *anyopaque, vertex_count: u32, first_vertex: u32) hal.Error!void {
        const self: *CommandBuffer = @ptrCast(@alignCast(ptr));
        self.cmds.append(self.gpa, .{ .draw = .{ .vertex_count = vertex_count, .first_vertex = first_vertex } }) catch return error.OutOfMemory;
    }
    fn reset(ptr: *anyopaque) void {
        const self: *CommandBuffer = @ptrCast(@alignCast(ptr));
        self.cmds.clearRetainingCapacity();
    }
    fn deinit(ptr: *anyopaque) void {
        const self: *CommandBuffer = @ptrCast(@alignCast(ptr));
        const gpa = self.gpa;
        self.cmds.deinit(gpa);
        gpa.destroy(self);
    }

    const vtable = hal.CommandBuffer.VTable{
        .setRenderTarget = &setRenderTarget,
        .clear = &clear,
        .bindPipeline = &bindPipeline,
        .bindVertexBuffer = &bindVertexBuffer,
        .draw = &draw,
        .reset = &reset,
        .deinit = &deinit,
    };
};

/// A GPU command context for the AGX path. Records go to a CommandBuffer. submit()
/// replays them onto the GPU. The proven op is a clear over a render target via
/// asahi clearColorTo. A recorded draw or pipeline bind returns NotImplemented
/// (the AGX triangle draw is a documented hard open-item).
pub const Context = struct {
    dev: *AppleDevice,

    pub fn create(dev: *AppleDevice) hal.Error!hal.Context {
        const self = dev.gpa.create(Context) catch return error.OutOfMemory;
        self.* = .{ .dev = dev };
        return .{ .ptr = self, .vtable = &vtable };
    }

    fn beginCommands(ptr: *anyopaque) hal.Error!hal.CommandBuffer {
        const self: *Context = @ptrCast(@alignCast(ptr));
        return CommandBuffer.create(self.dev.gpa);
    }

    fn submit(ptr: *anyopaque, cb: hal.CommandBuffer) hal.Error!void {
        const self: *Context = @ptrCast(@alignCast(ptr));
        const buf: *CommandBuffer = @ptrCast(@alignCast(cb.ptr));

        // Replay the records: a bare clear runs the proven asahi clear render pass.
        // Any draw or pipeline bind is rejected (the AGX draw path is not implemented).
        var rt: ?*Resource = null;
        var color: ?hal.Color = null;
        var has_draw = false;
        for (buf.cmds.items) |cmd| switch (cmd) {
            .set_render_target => |t| rt = t,
            .clear => |c| color = c,
            // A draw or its pipeline/vertex bind means the unsupported graphics draw path.
            .bind_pipeline, .bind_vertex_buffer, .draw => has_draw = true,
        };
        if (has_draw) return error.NotImplemented;

        const target = rt orelse return error.InvalidArgument;
        const c = color orelse return error.InvalidArgument;

        // The clear render pass clears an arbitrary W x H RGBA8 image (the
        // size-dependent cmd_render/PBE/region-clip/isp_merge fields scale with
        // target.width/height). clearColorTo writes `c` to the target BO (its
        // gpu_va + size) on the GPU.
        asahi.clearColorTo(
            &self.dev.dev,
            self.dev.vm_id,
            self.dev.queue_id,
            target.bo.gpu_va,
            target.bo.size,
            target.width,
            target.height,
            .{ c.r, c.g, c.b, c.a },
        ) catch |e| return aserr(e);
    }

    fn present(ptr: *anyopaque, surface: *hal.Surface, source: *hal.Resource) hal.Error!void {
        const self: *Context = @ptrCast(@alignCast(ptr));
        _ = self;
        const surf: *Surface = @ptrCast(@alignCast(surface));
        const src: *Resource = @ptrCast(@alignCast(source));

        // The GPU wrote the source framebuffer (the clear's fence already flushed
        // it to memory). CPU-blit it into the platform surface's current buffer.
        const src_bytes = src.bo.cpu;
        const src_w = src.width;
        const src_stride: u32 = src.width * 4; // linear RGBA8, 4 bytes/pixel

        const dstbuf = try surf.platform.currentBuffer();
        // The source render target is linear RGBA8 (byte0=R,1=G,2=B,3=A). A bgra8
        // surface buffer wants B,G,R,A: swap the red/blue lanes. An rgba8 buffer
        // blits straight across.
        const swap = dstbuf.format == .bgra8_unorm;
        const copy_w = @min(src_w, dstbuf.width);
        const copy_h = @min(src.height, dstbuf.height);
        var y: u32 = 0;
        while (y < copy_h) : (y += 1) {
            const src_row = src_bytes[@as(usize, y) * src_stride ..][0 .. copy_w * 4];
            const dst_row = dstbuf.bytes[@as(usize, y) * dstbuf.stride ..][0 .. copy_w * 4];
            if (swap) {
                var x: usize = 0;
                while (x < copy_w) : (x += 1) {
                    const o = x * 4;
                    dst_row[o + 0] = src_row[o + 2]; // B <- R
                    dst_row[o + 1] = src_row[o + 1]; // G
                    dst_row[o + 2] = src_row[o + 0]; // R <- B
                    dst_row[o + 3] = src_row[o + 3]; // A
                }
            } else {
                @memcpy(dst_row, src_row);
            }
        }
        try surf.platform.commit();
    }

    fn deinit(ptr: *anyopaque) void {
        const self: *Context = @ptrCast(@alignCast(ptr));
        self.dev.gpa.destroy(self);
    }

    const vtable = hal.Context.VTable{
        .beginCommands = &beginCommands,
        .submit = &submit,
        .present = &present,
        .deinit = &deinit,
    };
};

test {
    _ = @import("surface.zig");
}

test "apple command buffer records and a recorded draw submit returns NotImplemented" {
    // No GPU needed: drive only the record + the submit-replay's draw rejection by
    // building a Context shell over a fake device pointer that submit never
    // dereferences before the has_draw check.
    const gpa = std.testing.allocator;

    // A CommandBuffer that records a pipeline bind + a draw.
    const cb = try CommandBuffer.create(gpa);
    defer cb.deinit();
    // bindPipeline / draw take opaque pointers we never dereference here.
    var dummy_pipe: u8 = 0;
    try cb.bindPipeline(@ptrCast(&dummy_pipe));
    try cb.draw(3, 0);

    // submit must see the draw record and reject the whole submission.
    var devshell: AppleDevice = undefined;
    var ctx = Context{ .dev = &devshell };
    try std.testing.expectError(error.NotImplemented, Context.submit(&ctx, cb));
}

test "apple submit with neither RT nor clear is InvalidArgument (no GPU touched)" {
    const gpa = std.testing.allocator;
    const cb = try CommandBuffer.create(gpa);
    defer cb.deinit();
    // empty command buffer: no render target -> InvalidArgument before any GPU op.
    var devshell: AppleDevice = undefined;
    var ctx = Context{ .dev = &devshell };
    try std.testing.expectError(error.InvalidArgument, Context.submit(&ctx, cb));
}
