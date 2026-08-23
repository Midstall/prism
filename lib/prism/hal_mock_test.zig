const std = @import("std");
const hal = @import("hal.zig");
const drv = @import("driver.zig");

// A minimal in-memory mock that implements the full HAL chain so the interface
// can be exercised end to end with no real driver. Handles are backed by real
// allocations cast to the opaque handle types.

const MockResource = struct { bytes: []u8 };
const MockShader = struct { stage: hal.ShaderStage };
const MockPipeline = struct {};
const MockSurface = struct {};

const MockCmd = struct {
    gpa: std.mem.Allocator,
    draws: u32 = 0,
    clears: u32 = 0,

    fn setRenderTarget(ptr: *anyopaque, target: *hal.Resource) hal.Error!void {
        _ = ptr;
        _ = target;
    }
    fn clear(ptr: *anyopaque, color: hal.Color) hal.Error!void {
        _ = color;
        const self: *MockCmd = @ptrCast(@alignCast(ptr));
        self.clears += 1;
    }
    fn bindPipeline(ptr: *anyopaque, p: *hal.Pipeline) hal.Error!void {
        _ = ptr;
        _ = p;
    }
    fn bindVertexBuffer(ptr: *anyopaque, b: *hal.Resource) hal.Error!void {
        _ = ptr;
        _ = b;
    }
    fn draw(ptr: *anyopaque, vertex_count: u32, first_vertex: u32) hal.Error!void {
        _ = vertex_count;
        _ = first_vertex;
        const self: *MockCmd = @ptrCast(@alignCast(ptr));
        self.draws += 1;
    }
    fn reset(ptr: *anyopaque) void {
        const self: *MockCmd = @ptrCast(@alignCast(ptr));
        self.draws = 0;
        self.clears = 0;
    }
    fn deinit(ptr: *anyopaque) void {
        const self: *MockCmd = @ptrCast(@alignCast(ptr));
        self.gpa.destroy(self);
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

const MockContext = struct {
    gpa: std.mem.Allocator,
    last: ?*MockCmd = null,

    fn beginCommands(ptr: *anyopaque) hal.Error!hal.CommandBuffer {
        const self: *MockContext = @ptrCast(@alignCast(ptr));
        const cmd = self.gpa.create(MockCmd) catch return error.OutOfMemory;
        cmd.* = .{ .gpa = self.gpa };
        self.last = cmd;
        return .{ .ptr = cmd, .vtable = &MockCmd.vtable };
    }
    fn submit(ptr: *anyopaque, cb: hal.CommandBuffer) hal.Error!void {
        _ = ptr;
        _ = cb;
    }
    fn present(ptr: *anyopaque, surface: *hal.Surface, source: *hal.Resource) hal.Error!void {
        _ = ptr;
        _ = surface;
        _ = source;
    }
    fn deinit(ptr: *anyopaque) void {
        const self: *MockContext = @ptrCast(@alignCast(ptr));
        self.gpa.destroy(self);
    }

    const vtable = hal.Context.VTable{
        .beginCommands = &beginCommands,
        .submit = &submit,
        .present = &present,
        .deinit = &deinit,
    };
};

const MockDevice = struct {
    gpa: std.mem.Allocator,

    fn createContext(ptr: *anyopaque) hal.Error!hal.Context {
        const self: *MockDevice = @ptrCast(@alignCast(ptr));
        const c = self.gpa.create(MockContext) catch return error.OutOfMemory;
        c.* = .{ .gpa = self.gpa };
        return .{ .ptr = c, .vtable = &MockContext.vtable };
    }
    fn createResource(ptr: *anyopaque, desc: hal.ResourceDesc) hal.Error!*hal.Resource {
        const self: *MockDevice = @ptrCast(@alignCast(ptr));
        const size: usize = switch (desc) {
            .buffer => |b| b.size,
            .image => |i| @as(usize, i.width) * i.height * i.format.bytesPerPixel(),
        };
        const r = self.gpa.create(MockResource) catch return error.OutOfMemory;
        errdefer self.gpa.destroy(r);
        r.* = .{ .bytes = self.gpa.alloc(u8, size) catch return error.OutOfMemory };
        return @ptrCast(r);
    }
    fn destroyResource(ptr: *anyopaque, resource: *hal.Resource) void {
        const self: *MockDevice = @ptrCast(@alignCast(ptr));
        const r: *MockResource = @ptrCast(@alignCast(resource));
        self.gpa.free(r.bytes);
        self.gpa.destroy(r);
    }
    fn mapResource(ptr: *anyopaque, resource: *hal.Resource) hal.Error![]u8 {
        _ = ptr;
        const r: *MockResource = @ptrCast(@alignCast(resource));
        return r.bytes;
    }
    fn unmapResource(ptr: *anyopaque, resource: *hal.Resource) void {
        _ = ptr;
        _ = resource;
    }
    fn createShaderModule(ptr: *anyopaque, desc: hal.ShaderModuleDesc) hal.Error!*hal.ShaderModule {
        const self: *MockDevice = @ptrCast(@alignCast(ptr));
        const m = self.gpa.create(MockShader) catch return error.OutOfMemory;
        m.* = .{ .stage = desc.stage };
        return @ptrCast(m);
    }
    fn destroyShaderModule(ptr: *anyopaque, module: *hal.ShaderModule) void {
        const self: *MockDevice = @ptrCast(@alignCast(ptr));
        const m: *MockShader = @ptrCast(@alignCast(module));
        self.gpa.destroy(m);
    }
    fn dispatchCompute(ptr: *anyopaque, d: hal.ComputeDispatch) hal.Error!void {
        _ = ptr;
        _ = d;
        return error.NotImplemented;
    }
    fn createPipeline(ptr: *anyopaque, desc: hal.PipelineDesc) hal.Error!*hal.Pipeline {
        _ = desc;
        const self: *MockDevice = @ptrCast(@alignCast(ptr));
        const p = self.gpa.create(MockPipeline) catch return error.OutOfMemory;
        p.* = .{};
        return @ptrCast(p);
    }
    fn destroyPipeline(ptr: *anyopaque, pipeline: *hal.Pipeline) void {
        const self: *MockDevice = @ptrCast(@alignCast(ptr));
        const p: *MockPipeline = @ptrCast(@alignCast(pipeline));
        self.gpa.destroy(p);
    }
    fn createSurface(ptr: *anyopaque, platform_surface: *anyopaque) hal.Error!*hal.Surface {
        _ = platform_surface;
        const self: *MockDevice = @ptrCast(@alignCast(ptr));
        const s = self.gpa.create(MockSurface) catch return error.OutOfMemory;
        s.* = .{};
        return @ptrCast(s);
    }
    fn destroySurface(ptr: *anyopaque, surface: *hal.Surface) void {
        const self: *MockDevice = @ptrCast(@alignCast(ptr));
        self.gpa.destroy(@as(*MockSurface, @ptrCast(@alignCast(surface))));
    }
    fn deinit(ptr: *anyopaque) void {
        const self: *MockDevice = @ptrCast(@alignCast(ptr));
        self.gpa.destroy(self);
    }

    const mock_formats = [_]hal.Format{ .rgba8_unorm, .bgra8_unorm };
    fn caps(ptr: *anyopaque) hal.DeviceCaps {
        _ = ptr;
        return .{
            .device_name = "mock device",
            .formats = &mock_formats,
            .hdr = hal.DeviceCaps.deriveHdr(&mock_formats),
            .spirv = true,
            .compute = true,
            .graphics = true,
            .present = true,
            .max_texture_dim = 4096,
        };
    }

    const vtable = hal.Device.VTable{
        .caps = &caps,
        .createContext = &createContext,
        .createResource = &createResource,
        .destroyResource = &destroyResource,
        .mapResource = &mapResource,
        .unmapResource = &unmapResource,
        .createShaderModule = &createShaderModule,
        .destroyShaderModule = &destroyShaderModule,
        .dispatchCompute = &dispatchCompute,
        .createPipeline = &createPipeline,
        .destroyPipeline = &destroyPipeline,
        .createSurface = &createSurface,
        .destroySurface = &destroySurface,
        .deinit = &deinit,
    };
};

fn mockCreateDevice(ptr: *anyopaque, gpa: std.mem.Allocator) drv.Error!hal.Device {
    _ = ptr;
    const dev = gpa.create(MockDevice) catch return error.OutOfMemory;
    dev.* = .{ .gpa = gpa };
    return .{ .ptr = dev, .vtable = &MockDevice.vtable };
}
fn mockAvailable(ptr: *anyopaque) bool {
    _ = ptr;
    return true;
}

test "exportResource returns Unsupported when vtable slot is null" {
    const gpa = std.testing.allocator;
    var state: u8 = 0;
    const vt = drv.Driver.VTable{ .isAvailable = &mockAvailable, .createDevice = &mockCreateDevice };
    const driver = drv.Driver{ .name = "mock", .ptr = &state, .vtable = &vt };

    const device = try driver.createDevice(gpa);
    defer device.deinit();

    const resource = try device.createResource(.{ .image = .{ .width = 4, .height = 4, .format = .rgba8_unorm, .usage = .{ .render_target = true, .scanout = true } } });
    defer device.destroyResource(resource);

    try std.testing.expectError(error.Unsupported, device.exportResource(resource));
}

test "mock driver drives the full HAL chain end to end" {
    const gpa = std.testing.allocator;
    var state: u8 = 0;
    const vt = drv.Driver.VTable{ .isAvailable = &mockAvailable, .createDevice = &mockCreateDevice };
    const driver = drv.Driver{ .name = "mock", .ptr = &state, .vtable = &vt };

    const device = try driver.createDevice(gpa);
    defer device.deinit();

    const target = try device.createResource(.{ .image = .{ .width = 8, .height = 8, .format = .rgba8_unorm, .usage = .{ .render_target = true } } });
    defer device.destroyResource(target);

    const vs = try device.createShaderModule(.{ .stage = .vertex, .code = "vs" });
    defer device.destroyShaderModule(vs);
    const fs = try device.createShaderModule(.{ .stage = .fragment, .code = "fs" });
    defer device.destroyShaderModule(fs);

    const vbuf = try device.createResource(.{ .buffer = .{ .size = 36, .usage = .{ .vertex = true } } });
    defer device.destroyResource(vbuf);

    const pipeline = try device.createPipeline(.{
        .vertex = vs,
        .fragment = fs,
        .vertex_layout = .{ .stride = 12, .attributes = &.{.{ .location = 0, .format = .rgba8_unorm, .offset = 0 }} },
        .color_format = .rgba8_unorm,
    });
    defer device.destroyPipeline(pipeline);

    const ctx = try device.createContext();
    defer ctx.deinit();

    const cb = try ctx.beginCommands();
    defer cb.deinit();
    try cb.setRenderTarget(target);
    try cb.clear(.{ .r = 0, .g = 0, .b = 0, .a = 1 });
    try cb.bindPipeline(pipeline);
    try cb.bindVertexBuffer(vbuf);
    try cb.draw(3, 0);
    try ctx.submit(cb);

    const mapped = try device.mapResource(target);
    try std.testing.expectEqual(@as(usize, 8 * 8 * 4), mapped.len);

    const mock_ctx: *MockContext = @ptrCast(@alignCast(ctx.ptr));
    const mock_cmd = mock_ctx.last.?;
    try std.testing.expectEqual(@as(u32, 1), mock_cmd.clears);
    try std.testing.expectEqual(@as(u32, 1), mock_cmd.draws);

    var dummy: u8 = 0;
    const hal_surface = try device.createSurface(@ptrCast(&dummy));
    defer device.destroySurface(hal_surface);
    try ctx.present(hal_surface, target);
}
