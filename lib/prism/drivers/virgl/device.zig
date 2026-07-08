//! Prism HAL device over the virgl transport seam. Lowers the standard HAL flow to virgl commands.
//! Resources back onto transport-owned guest memory. Freestanding: DMA-coherent backing. Linux: mmap of each GEM BO.
//! Shaders are SPIR-V -> Vulcan IR -> TGSI text (virglrenderer's tgsi_text_translate).

const std = @import("std");
const builtin = @import("builtin");
const hal = @import("../../hal.zig");
const stream_mod = @import("virgl_stream.zig");
const spirv = @import("../../spirv.zig");
const tgsi = @import("vulcan-target").tgsi;
const enc = @import("encoding.zig");
const formats = @import("formats.zig");

const transport_mod = @import("transport.zig");
pub const Transport = transport_mod.Transport;
pub const InitArgs = transport_mod.InitArgs;
const TResource = transport_mod.Resource;

/// Map a HAL Format to the matching virgl color format (formats.zig). SDR
/// rgba8/bgra8 map to B8G8R8X8. HDR formats (rgba16_float / rgb10a2 / rgb10x2)
/// map to their dedicated high-precision pipe formats.
const virglColorFormat = formats.virglColorFormat;

const Resource = struct {
    kind: hal.ResourceKind,
    t: TResource,
    format: hal.Format = .rgba8_unorm,
    /// Created with usage.sampled: a sampler view over it reads R8G8B8A8_UNORM.
    sampled: bool = false,
    /// MSAA sample count (1 = single-sample). >1 means the host allocated a
    /// multisampled surface. The draw enables multisample rasterization and a
    /// resolve BLIT downsamples it to a single-sample image.
    samples: u8 = 1,
};

const ShaderModule = struct {
    stage: hal.ShaderStage,
    // The TGSI text bytes (owned, NUL-terminated, dword-padded).
    tgsi: []u8,
};

const Pipeline = struct {
    vs: *ShaderModule,
    fs: *ShaderModule,
    layout: hal.VertexLayout,
    color_format: hal.Format,
    /// Primitive topology (default triangle_list) -> the DRAW_VBO PIPE_PRIM mode.
    topology: hal.Topology = .triangle_list,
    /// Line width (glLineWidth, default 1) -> the rasterizer state's line_width.
    line_width: f32 = 1.0,
    /// Depth-test state (default: disabled) -> the DSA object's S0 when a depth
    /// attachment is bound at submit.
    depth: hal.DepthState = .{},
    /// Alpha-blend state (default: disabled) -> the blend object's S2.
    blend: hal.BlendState = .{},
    /// Back-face cull + front winding (default: none/ccw) -> the rasterizer S0.
    cull: hal.CullState = .{},
};

const Surface = struct {
    platform: *anyopaque,
};

pub const Device = struct {
    gpa: std.mem.Allocator,
    transport: Transport,
    stream: [1024]u32 = undefined,

    /// Create a HAL device over the virgl transport. `args` is the OS-specific
    /// construction argument (freestanding: the live Conduit `*Virtio` + DMA
    /// stream scratch. Linux: an optional render-node path). The device owns its
    /// 3D context, created lazily by the transport on first use.
    pub fn create(gpa: std.mem.Allocator, args: InitArgs) hal.Error!hal.Device {
        const self = gpa.create(Device) catch return error.OutOfMemory;
        errdefer gpa.destroy(self);
        self.* = .{ .gpa = gpa, .transport = Transport.init(gpa, args) catch return error.InitializationFailed };
        return .{ .ptr = self, .vtable = &vtable };
    }

    fn createResource(ptr: *anyopaque, desc: hal.ResourceDesc) hal.Error!*hal.Resource {
        const self: *Device = @ptrCast(@alignCast(ptr));
        const r = self.gpa.create(Resource) catch return error.OutOfMemory;
        errdefer self.gpa.destroy(r);

        switch (desc) {
            .buffer => |b| {
                const tr = self.transport.createBuffer(b.size) catch |e| return mapErr(e);
                r.* = .{ .kind = .buffer, .t = tr };
            },
            .image => |i| {
                // A sampled texture is stored as R8G8B8A8_UNORM so the sampler view
                // reads Prism's internal RGBA texel bytes in the right channel order
                // (the color/scanout path uses BGRA-ordered B8G8R8X8). Depth + plain
                // color targets keep their virglColorFormat mapping.
                const vformat = if (i.usage.sampled and i.format != .depth32_float)
                    formats.virglSampledFormat(i.format)
                else
                    virglColorFormat(i.format);
                const tr = self.transport.createImage(i.width, i.height, vformat, i.format.bytesPerPixel(), i.samples) catch |e| return mapErr(e);
                r.* = .{ .kind = .image, .t = tr, .format = i.format, .sampled = i.usage.sampled, .samples = i.samples };
            },
        }
        return @ptrCast(r);
    }

    fn destroyResource(ptr: *anyopaque, resource: *hal.Resource) void {
        const self: *Device = @ptrCast(@alignCast(ptr));
        const r: *Resource = @ptrCast(@alignCast(resource));
        self.transport.destroyResource(r.t);
        self.gpa.destroy(r);
    }

    fn mapResource(ptr: *anyopaque, resource: *hal.Resource) hal.Error![]u8 {
        const self: *Device = @ptrCast(@alignCast(ptr));
        const r: *Resource = @ptrCast(@alignCast(resource));
        return self.transport.map(r.t) catch |e| return mapErr(e);
    }

    fn unmapResource(ptr: *anyopaque, resource: *hal.Resource) void {
        // Uploading a vertex buffer to the host happens at submit time (we know
        // the byte count there). Nothing to flush on unmap.
        _ = ptr;
        _ = resource;
    }

    fn createShaderModule(ptr: *anyopaque, desc: hal.ShaderModuleDesc) hal.Error!*hal.ShaderModule {
        const self: *Device = @ptrCast(@alignCast(ptr));
        const m = self.gpa.create(ShaderModule) catch return error.OutOfMemory;
        errdefer self.gpa.destroy(m);

        // SPIR-V -> Vulcan IR -> TGSI text. The `code` bytes are a SPIR-V binary.
        // Parse it to a graphics IR function and lower to the TGSI text the
        // virgl shader-create command carries.
        var func = spirv.parseSpirv(self.gpa, desc.code) catch return error.InvalidArgument;
        defer func.deinit();
        const text = tgsi.lower(self.gpa, &func) catch return error.InvalidArgument;
        m.* = .{ .stage = desc.stage, .tgsi = text };
        return @ptrCast(m);
    }

    fn destroyShaderModule(ptr: *anyopaque, module: *hal.ShaderModule) void {
        const self: *Device = @ptrCast(@alignCast(ptr));
        const m: *ShaderModule = @ptrCast(@alignCast(module));
        self.gpa.free(m.tgsi);
        self.gpa.destroy(m);
    }

    /// One-shot HAL compute dispatch. Stubbed: the virgl driver only implements
    /// the graphics+present path (caps.compute = false). The field exists to
    /// complete the Device.VTable literal.
    fn dispatchCompute(ptr: *anyopaque, d: hal.ComputeDispatch) hal.Error!void {
        _ = ptr;
        _ = d;
        return error.NotImplemented;
    }

    fn createPipeline(ptr: *anyopaque, desc: hal.PipelineDesc) hal.Error!*hal.Pipeline {
        const self: *Device = @ptrCast(@alignCast(ptr));
        const p = self.gpa.create(Pipeline) catch return error.OutOfMemory;
        p.* = .{
            .vs = @ptrCast(@alignCast(desc.vertex)),
            .fs = @ptrCast(@alignCast(desc.fragment)),
            .layout = desc.vertex_layout,
            .color_format = desc.color_format,
            .topology = desc.topology,
            .line_width = desc.line_width,
            .depth = desc.depth,
            .blend = desc.blend,
            .cull = desc.cull,
        };
        return @ptrCast(p);
    }

    fn destroyPipeline(ptr: *anyopaque, pipeline: *hal.Pipeline) void {
        const self: *Device = @ptrCast(@alignCast(ptr));
        self.gpa.destroy(@as(*Pipeline, @ptrCast(@alignCast(pipeline))));
    }

    fn createSurface(ptr: *anyopaque, platform_surface: *anyopaque) hal.Error!*hal.Surface {
        const self: *Device = @ptrCast(@alignCast(ptr));
        const s = self.gpa.create(Surface) catch return error.OutOfMemory;
        s.* = .{ .platform = platform_surface };
        return @ptrCast(s);
    }

    fn destroySurface(ptr: *anyopaque, surface: *hal.Surface) void {
        const self: *Device = @ptrCast(@alignCast(ptr));
        self.gpa.destroy(@as(*Surface, @ptrCast(@alignCast(surface))));
    }

    /// SDR color formats the virgl render/scanout path handles (no HDR).
    const supported_formats = [_]hal.Format{ .rgba8_unorm, .bgra8_unorm };

    fn caps(ptr: *anyopaque) hal.DeviceCaps {
        _ = ptr;
        return .{
            .device_name = "virgl (host GPU)",
            .formats = &supported_formats,
            .hdr = hal.DeviceCaps.deriveHdr(&supported_formats),
            // Shaders enter as SPIR-V (lowered to TGSI for virglrenderer).
            .spirv = true,
            // The implemented path is the graphics+present triangle. No compute
            // submission is proven through virgl here.
            .compute = false,
            .graphics = true,
            .present = true,
            .max_texture_dim = 16384,
        };
    }

    fn createContext(ptr: *anyopaque) hal.Error!hal.Context {
        const self: *Device = @ptrCast(@alignCast(ptr));
        self.transport.ensureContext() catch |e| return mapErr(e);
        return @import("context.zig").Context.create(self);
    }

    fn deinit(ptr: *anyopaque) void {
        const self: *Device = @ptrCast(@alignCast(ptr));
        self.transport.deinit();
        self.gpa.destroy(self);
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

    /// What `adoptScanout` returns: the HAL render-target resource (drive the
    /// normal HAL flow into it) plus the GEM bo + res handles + the CPU mapping
    /// the display owner needs to ADDFB2 the bo as a KMS framebuffer and to read
    /// the scanned-out pixels back (the in-guest verification).
    pub const Scanout = struct {
        resource: *hal.Resource,
        bo_handle: u32,
        res_handle: u32,
        bytes: []u8,
        width: u32,
        height: u32,
    };

    /// Create a scanout-able render target on the transport's (shared card0) fd and
    /// wrap it as a HAL image resource. The same virtgpu resource is both the virgl
    /// render target (returned `resource`) and the KMS scanout framebuffer (the
    /// caller ADDFB2's `bo_handle`), with no copy. Only available on a transport that
    /// exposes `createScanoutResource` (the Linux DRM transport).
    pub fn adoptScanout(self: *Device, width: u32, height: u32, format: hal.Format) hal.Error!Scanout {
        const sr = self.transport.createScanoutResource(width, height, virglColorFormat(format)) catch |e| return mapErr(e);
        const r = self.gpa.create(Resource) catch return error.OutOfMemory;
        r.* = .{
            .kind = .image,
            .format = format,
            .t = .{
                .res_id = sr.res_handle,
                .bytes = sr.bytes,
                .bo_handle = sr.bo_handle,
                .width = width,
                .height = height,
            },
        };
        return .{
            .resource = @ptrCast(r),
            .bo_handle = sr.bo_handle,
            .res_handle = sr.res_handle,
            .bytes = sr.bytes,
            .width = width,
            .height = height,
        };
    }

    pub fn resourceOf(resource: *hal.Resource) *Resource {
        return @ptrCast(@alignCast(resource));
    }
    pub fn pipelineOf(pipeline: *hal.Pipeline) *Pipeline {
        return @ptrCast(@alignCast(pipeline));
    }
};

/// Map a transport error onto the HAL error set.
fn mapErr(e: transport_mod.Error) hal.Error {
    return switch (e) {
        error.OutOfMemory => error.OutOfMemory,
        error.InitializationFailed => error.InitializationFailed,
        error.DeviceLost => error.DeviceLost,
    };
}

pub const ResourceRef = Resource;
pub const PipelineRef = Pipeline;
pub const ShaderModuleRef = ShaderModule;
