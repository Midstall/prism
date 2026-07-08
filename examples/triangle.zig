//! triangle: draws a colored triangle on the best available driver (real GPU
//! first, software last) and presents it to a Wayland compositor, a KMS/DRM
//! console, or a headless framebuffer. Run: zig build run-triangle.

const std = @import("std");
const prism = @import("prism");

const platform = prism.platform;
const hal = prism.hal;

const Vtx = extern struct { x: f32, y: f32, r: f32, g: f32, b: f32 };

const tri = [3]Vtx{
    .{ .x = -0.8, .y = -0.8, .r = 1, .g = 0, .b = 0 },
    .{ .x = 0.8, .y = -0.8, .r = 0, .g = 1, .b = 0 },
    .{ .x = 0.0, .y = 0.8, .r = 0, .g = 0, .b = 1 },
};

const vs_src =
    \\attribute vec2 aPos;
    \\attribute vec3 aColor;
    \\varying vec3 vColor;
    \\void main() { gl_Position = vec4(aPos, 0.0, 1.0); vColor = aColor; }
;
const fs_src =
    \\precision mediump float;
    \\varying vec3 vColor;
    \\void main() { gl_FragColor = vec4(vColor, 1.0); }
;

/// The vertex-attribute format for a GLSL attribute of `n` components.
fn attrFormat(n: u8) hal.Format {
    return switch (n) {
        2 => .r32g32_float,
        3 => .r32g32b32_float,
        else => .r32g32b32a32_float,
    };
}

/// Byte offset of an attribute within `Vtx`. Position leads, color follows.
fn attrOffset(name: []const u8) u32 {
    return if (std.mem.eql(u8, name, "aPos")) 0 else 8;
}

/// Render the triangle into a fresh w*h target and present it to the surface.
/// Called for the first frame and again on every resize.
fn drawAndPresent(
    device: prism.Device,
    ctx: prism.Context,
    hal_surface: *prism.hal.Surface,
    pipeline: *prism.hal.Pipeline,
    vbuf: *prism.hal.Resource,
    w: u32,
    h: u32,
) !void {
    const target = try device.createResource(.{ .image = .{
        .width = w,
        .height = h,
        .format = .rgba8_unorm,
        .usage = .{ .render_target = true },
    } });
    defer device.destroyResource(target);

    const cb = try ctx.beginCommands();
    defer cb.deinit();
    try cb.setRenderTarget(target);
    try cb.clear(.{ .r = 0.05, .g = 0.05, .b = 0.05, .a = 1 });
    try cb.bindPipeline(pipeline);
    try cb.bindVertexBuffer(vbuf);
    try cb.draw(3, 0);
    try ctx.submit(cb);
    try ctx.present(hal_surface, target);
}

/// The per-device render state: the device plus the size-independent resources
/// (vertex buffer, shaders, pipeline, context). Built on whichever driver wins
/// selection, torn down together.
const Renderer = struct {
    driver_name: []const u8,
    device: prism.Device,
    vbuf: *prism.hal.Resource,
    vs: *prism.hal.ShaderModule,
    fs: *prism.hal.ShaderModule,
    pipeline: *prism.hal.Pipeline,
    ctx: prism.Context,

    fn deinit(self: Renderer) void {
        self.ctx.deinit();
        self.device.destroyPipeline(self.pipeline);
        self.device.destroyShaderModule(self.fs);
        self.device.destroyShaderModule(self.vs);
        self.device.destroyResource(self.vbuf);
        self.device.deinit();
    }
};

/// Build the full triangle render state on one driver's device from the shared
/// SPIR-V. A driver that cannot build a graphics pipeline (the apple driver stubs
/// the shader path) errors out so the caller moves to the next driver. The
/// errdefer chain frees whatever was created. The software driver always
/// succeeds, so it is the guaranteed fallback.
fn buildRenderer(
    d: anytype,
    gpa: std.mem.Allocator,
    vs_spirv: []const u8,
    attributes: []const prism.glsl.AttributeMember,
    fs_spirv: []const u8,
) !Renderer {
    const device = try d.createDevice(gpa);
    errdefer device.deinit();

    const vbuf = try device.createResource(.{ .buffer = .{ .size = @sizeOf(@TypeOf(tri)), .usage = .{ .vertex = true } } });
    errdefer device.destroyResource(vbuf);
    @memcpy(try device.mapResource(vbuf), std.mem.asBytes(&tri));

    const vs = try device.createShaderModule(.{ .stage = .vertex, .code = vs_spirv });
    errdefer device.destroyShaderModule(vs);
    const fs = try device.createShaderModule(.{ .stage = .fragment, .code = fs_spirv });
    errdefer device.destroyShaderModule(fs);

    // The vertex layout follows the locations the GLSL front end assigned, so the
    // buffer's f32 fields feed the shader's attributes without any name guessing.
    var attrs: [4]hal.VertexAttribute = undefined;
    for (attributes, 0..) |a, i| {
        attrs[i] = .{ .location = a.location, .format = attrFormat(a.components), .offset = attrOffset(a.name) };
    }

    const pipeline = try device.createPipeline(.{
        .vertex = vs,
        .fragment = fs,
        .vertex_layout = .{ .stride = @sizeOf(Vtx), .attributes = attrs[0..attributes.len] },
        .color_format = .rgba8_unorm,
    });
    errdefer device.destroyPipeline(pipeline);

    const ctx = try device.createContext();
    errdefer ctx.deinit();

    return .{
        .driver_name = d.name,
        .device = device,
        .vbuf = vbuf,
        .vs = vs,
        .fs = fs,
        .pipeline = pipeline,
        .ctx = ctx,
    };
}

/// A display backend plus whether it has a windowing event source. Only the
/// Wayland backend delivers resize/close events, so the others render one frame.
const Selected = struct {
    display: platform.Display,
    interactive: bool,
    name: []const u8,
};

/// Open the best available display: a Wayland compositor if one is reachable,
/// else a KMS/DRM console, else an offscreen headless framebuffer that always
/// works. The render path is identical across all three.
fn openDisplay(gpa: std.mem.Allocator, io: std.Io, env: anytype, path_buf: []u8) !Selected {
    if (platform.wayland.resolveSocketPath(env, path_buf)) |socket_path| {
        if (platform.wayland.create(gpa, io, socket_path)) |display| {
            return .{ .display = display, .interactive = true, .name = "wayland" };
        } else |_| {}
    } else |_| {}

    if (platform.drm.create(gpa)) |display| {
        return .{ .display = display, .interactive = false, .name = "drm" };
    } else |_| {}

    return .{ .display = try platform.headless.create(gpa), .interactive = false, .name = "headless" };
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;

    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const sel = try openDisplay(gpa, init.io, init.environ_map, &path_buf);
    defer sel.display.deinit();

    var plat_surface = try sel.display.createSurface(.{ .width = 800, .height = 600 });
    defer plat_surface.deinit();

    // Compile the shaders once. Every driver consumes the same SPIR-V.
    var cvs = try prism.glsl.compileForStageWithLayout(gpa, vs_src, .vertex);
    defer cvs.deinit(gpa);
    const fs_spirv = try prism.glsl.compileForStage(gpa, fs_src, .fragment);
    defer gpa.free(fs_spirv);

    // Prefer a real GPU, fall back to the software rasterizer. Probe each compiled
    // driver in preference order by building the pipeline on it. The apple driver
    // creates a device but stubs the draw path, so on an M-series machine this
    // lands on software.
    const renderer = for (prism.drivers.all) |d| {
        if (!d.isAvailable()) continue;
        if (buildRenderer(d, gpa, cvs.spirv, cvs.attributes, fs_spirv)) |r| break r else |_| {}
    } else return error.InitializationFailed;
    defer renderer.deinit();
    std.debug.print("triangle: display '{s}', driver '{s}'\n", .{ sel.name, renderer.driver_name });

    const hal_surface = try renderer.device.createSurface(@ptrCast(&plat_surface));
    defer renderer.device.destroySurface(hal_surface);

    var sz = plat_surface.size();
    try drawAndPresent(renderer.device, renderer.ctx, hal_surface, renderer.pipeline, renderer.vbuf, sz[0], sz[1]);

    // A backend with no event source shows the one frame and returns.
    if (!sel.interactive) return;

    // Render until the user closes the window. processEvents blocks on the next
    // compositor event. On resize we re-render at the new size.
    while (true) {
        switch (plat_surface.processEvents() catch break) {
            .none => {},
            .resized => {
                sz = plat_surface.size();
                try drawAndPresent(renderer.device, renderer.ctx, hal_surface, renderer.pipeline, renderer.vbuf, sz[0], sz[1]);
            },
            .closed => break,
        }
    }
}

test "triangle module compiles" {
    _ = tri;
    _ = vs_src;
    _ = fs_src;
}
