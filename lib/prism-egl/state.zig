//! EGL state tracker: Display objects, the honest EGLConfig set, eglQueryString
//! data, eglChooseConfig filtering, and the per-thread EGLint error model.
//! C-ABI exports live in vendor.zig. Testable logic lives here.

const std = @import("std");
const builtin = @import("builtin");
const prism = @import("prism");
const egl = @import("egl.zig");
const wlw = @import("wl_egl_window.zig");

const EGLint = egl.EGLint;

/// Process-wide allocator for EGL objects. EGL has no allocator parameter, so
/// we own one (page allocator, like the Vulkan ICD's object allocations).
pub const gpa = std.heap.page_allocator;

/// HAL driver pin for new EGL displays. Null = auto-detect (real hardware first,
/// software last). There is no env knob. The in-tree software oracles pin "software"
/// to assert exact, GPU-independent values. Under a test build that is the default,
/// so `zig build test` never depends on the host GPU.
pub var pinned_driver: ?[]const u8 = if (builtin.is_test) "software" else null;

// --- The honest EGLConfig set ----------------------------------------------

/// One advertised EGL framebuffer config. Values are real and consistent.
/// eglGetConfigAttrib reports them and eglChooseConfig filters on them.
pub const Config = struct {
    id: EGLint,
    red: EGLint,
    green: EGLint,
    blue: EGLint,
    alpha: EGLint,
    depth: EGLint,
    stencil: EGLint,
    samples: EGLint,

    /// EGL_BUFFER_SIZE = total color bits.
    pub fn bufferSize(self: Config) EGLint {
        return self.red + self.green + self.blue + self.alpha;
    }

    /// Look up a single config attribute by its EGL enum. Returns null for an
    /// attribute we don't recognise (caller maps that to EGL_BAD_ATTRIBUTE).
    pub fn attrib(self: Config, attribute: EGLint) ?EGLint {
        return switch (attribute) {
            egl.EGL_CONFIG_ID => self.id,
            egl.EGL_RED_SIZE => self.red,
            egl.EGL_GREEN_SIZE => self.green,
            egl.EGL_BLUE_SIZE => self.blue,
            egl.EGL_ALPHA_SIZE => self.alpha,
            egl.EGL_DEPTH_SIZE => self.depth,
            egl.EGL_STENCIL_SIZE => self.stencil,
            egl.EGL_BUFFER_SIZE => self.bufferSize(),
            egl.EGL_LUMINANCE_SIZE => 0,
            egl.EGL_ALPHA_MASK_SIZE => 0,
            egl.EGL_SAMPLES => self.samples,
            egl.EGL_SAMPLE_BUFFERS => if (self.samples > 0) 1 else 0,
            egl.EGL_COLOR_BUFFER_TYPE => egl.EGL_RGB_BUFFER,
            egl.EGL_CONFIG_CAVEAT => egl.EGL_NONE,
            egl.EGL_SURFACE_TYPE => egl.EGL_WINDOW_BIT | egl.EGL_PBUFFER_BIT,
            egl.EGL_RENDERABLE_TYPE => egl.EGL_OPENGL_ES2_BIT | egl.EGL_OPENGL_ES3_BIT | egl.EGL_OPENGL_BIT,
            egl.EGL_CONFORMANT => egl.EGL_OPENGL_ES2_BIT | egl.EGL_OPENGL_ES3_BIT | egl.EGL_OPENGL_BIT,
            egl.EGL_NATIVE_RENDERABLE => egl.EGL_FALSE,
            egl.EGL_NATIVE_VISUAL_ID => 0,
            egl.EGL_NATIVE_VISUAL_TYPE => egl.EGL_NONE,
            egl.EGL_TRANSPARENT_TYPE => egl.EGL_NONE,
            egl.EGL_TRANSPARENT_RED_VALUE, egl.EGL_TRANSPARENT_GREEN_VALUE, egl.EGL_TRANSPARENT_BLUE_VALUE => 0,
            egl.EGL_LEVEL => 0,
            egl.EGL_MAX_PBUFFER_WIDTH => 4096,
            egl.EGL_MAX_PBUFFER_HEIGHT => 4096,
            egl.EGL_MAX_PBUFFER_PIXELS => 4096 * 4096,
            egl.EGL_MIN_SWAP_INTERVAL => 0,
            egl.EGL_MAX_SWAP_INTERVAL => 1,
            egl.EGL_BIND_TO_TEXTURE_RGB => egl.EGL_FALSE,
            egl.EGL_BIND_TO_TEXTURE_RGBA => egl.EGL_FALSE,
            else => null,
        };
    }
};

/// Prism's advertised configs: RGBA8888 in four depth/stencil flavours. Honest
/// and small, mirroring the ICD's "one software physical device" minimalism.
pub const configs = [_]Config{
    .{ .id = 1, .red = 8, .green = 8, .blue = 8, .alpha = 8, .depth = 0, .stencil = 0, .samples = 0 },
    .{ .id = 2, .red = 8, .green = 8, .blue = 8, .alpha = 8, .depth = 24, .stencil = 0, .samples = 0 },
    .{ .id = 3, .red = 8, .green = 8, .blue = 8, .alpha = 8, .depth = 24, .stencil = 8, .samples = 0 },
    .{ .id = 4, .red = 8, .green = 8, .blue = 8, .alpha = 0, .depth = 0, .stencil = 0, .samples = 0 },
    // MSAA configs (EGL_SAMPLES): 4x and 2x multisampled RGBA8+D24 surfaces. eglSwapBuffers
    // resolves the samples=N backbuffer into the single-sample image via box-average.
    .{ .id = 5, .red = 8, .green = 8, .blue = 8, .alpha = 8, .depth = 24, .stencil = 0, .samples = 4 },
    .{ .id = 6, .red = 8, .green = 8, .blue = 8, .alpha = 8, .depth = 24, .stencil = 0, .samples = 2 },
};

/// Stable EGLConfig handle for index `i` (the C ABI hands clients an opaque
/// pointer). We encode the 1-based config id as a non-null pointer value so the
/// handle round-trips without any allocation and validates cheaply.
pub fn configHandle(i: usize) egl.EGLConfig {
    return @ptrFromInt(@as(usize, @intCast(configs[i].id)));
}

/// Recover the config index from an EGLConfig handle, or null if invalid.
pub fn configIndex(handle: egl.EGLConfig) ?usize {
    const v = @intFromPtr(handle);
    for (configs, 0..) |c, i| {
        if (@as(usize, @intCast(c.id)) == v) return i;
    }
    return null;
}

// --- Display objects --------------------------------------------------------

/// A Prism EGL display. Wraps the chosen platform + (after eglInitialize) a
/// Prism platform.Display + a brought-up HAL Device (the software driver). The
/// handle the client sees is `*Display`.
pub const Display = struct {
    platform: egl.EGLenum,
    native: ?*anyopaque,
    initialized: bool = false,
    /// The backing platform display, created lazily at eglInitialize. Headless
    /// for surfaceless (the pbuffer/offscreen path needs no Wayland connection).
    backend: ?prism.platform.Display = null,
    /// The HAL device backing all contexts/surfaces on this display. EGL M2 uses
    /// the software driver, the same HAL path the Vulkan ICD's software renderer
    /// and the clear/present example use.
    device: ?prism.hal.Device = null,

    pub fn deinit(self: *Display) void {
        if (self.device) |dev| dev.deinit();
        if (self.backend) |b| b.deinit();
        self.* = undefined;
    }

    /// True when this display was created from a stock app's own libwayland
    /// `wl_display *` (eglGetPlatformDisplay(EGL_PLATFORM_WAYLAND, app_wl_display)
    /// or the legacy eglGetDisplay(wl_display)). The native window is then a
    /// `wl_egl_window *`, and present goes through the system libwayland
    /// (wl_shm on the app's display), not Prism's own platform layer.
    pub fn appWlDisplay(self: *Display) ?*wlw.wl_display {
        if ((self.platform == egl.EGL_PLATFORM_WAYLAND_KHR or self.platform == egl.EGL_PLATFORM_NONE)) {
            if (self.native) |n| return @ptrCast(n);
        }
        return null;
    }

    /// The app's gbm.Device for a GBM-platform display (the native pointer passed
    /// to eglGetPlatformDisplay(EGL_PLATFORM_GBM, gbm_device)), else null. The
    /// native window for such a display is a gbm.Surface.
    pub fn gbmDevice(self: *Display) ?*prism.platform.gbm.Device {
        if (self.platform == egl.EGL_PLATFORM_GBM_KHR) {
            if (self.native) |n| return @ptrCast(@alignCast(n));
        }
        return null;
    }
};

// --- Contexts ---------------------------------------------------------------

/// A Prism EGL render context: the bound HAL Context (over the display's HAL
/// Device) + the chosen config + the client API it was created for. eglMakeCurrent
/// makes one current per-thread. The minimal GLES tracker drives the HAL through it.
var batching_enabled_cache: ?bool = null;

pub const Context = struct {
    display: *Display,
    config: usize, // index into `configs`
    client_api: egl.EGLenum, // EGL_OPENGL_ES_API / EGL_OPENGL_API (the bound API at create)
    hal: prism.hal.Context,
    // EGL 1.5 3.7.1: eglDestroyContext on a current context defers destruction
    // until it is no longer current. We flag it here and tear it down at the next
    // eglMakeCurrent that releases it. The context stays usable until the app rebinds.
    pending_delete: bool = false,

    // Per-context pool of expanded-vertex scratch buffers. Every draw expands indices into
    // a tightly-interleaved vertex buffer. Allocating a fresh GPU Resource per draw dominated
    // the per-draw CPU cost, so we keep grow-only buffers and check one out per draw. When
    // draws are submitted immediately (default) the pool resets after each submit and behaves
    // as a single reused buffer. When batched, each draw needs its own buffer because several
    // draws' vertex data coexist in one deferred submit. vb_used advances per draw and resets
    // at flush. Each buffer is fully rewritten before its draw. The prior submit has fenced
    // before any buffer is reused, so reuse is safe.
    vb_pool: std.ArrayListUnmanaged(PoolVb) = .empty,
    vb_used: usize = 0,

    // Draw batching: when active, drawArraysUboTarget records into `batch_cb` instead of
    // submitting. Accumulated draws flush at swapBuffers, GPU-result readback, render-target
    // change, or resource free (see flushDraws + its call sites). `batch_target` is the
    // single render target the open batch draws into. A draw to a different target flushes first.
    // Gated behind PRISM_BATCH so the default path is the proven per-draw-submit behavior.
    batch_cb: ?prism.hal.CommandBuffer = null,
    batch_target: ?*prism.hal.Resource = null,
    batch_depth: ?*prism.hal.Resource = null,
    batch_stencil: ?*prism.hal.Resource = null,
    batch_draws: u32 = 0,

    const PoolVb = struct { res: *prism.hal.Resource, map: []u8, cap: usize };
    /// Max draws in one open batch before an automatic flush. Bounds pool and command-buffer
    /// growth for a never-flushing loop. Well above a normal frame's draw count.
    const BATCH_DRAW_CAP: u32 = 1024;

    pub fn deinit(self: *Context) void {
        unregisterContext(self);
        // A context torn down mid-batch: drop the open command buffer (its draws were never needed).
        if (self.batch_cb) |cb| cb.deinit();
        const dev = self.display.device.?;
        for (self.vb_pool.items) |p| dev.destroyResource(p.res);
        self.vb_pool.deinit(gpa);
        self.hal.deinit();
        gpa.destroy(self);
    }

    /// Whether draw batching is active this session. On by default. `PRISM_NOBATCH` disables it
    /// (escape hatch back to per-draw-submit). Batching accumulates many glDraw* to the same
    /// render target into one HAL submit instead of submit+fence per draw, hiding the per-draw
    /// fence roundtrip (~2-3x faster). Verified on the full suite + 7 demo apps, both drivers,
    /// Xid-clean. Correctness relies on: UBO byte snapshots at record time (both drivers), a
    /// distinct pooled vertex buffer per draw, flushDraws at every GPU-result read / retarget /
    /// sync / resource-free, and the BATCH_DRAW_CAP safety flush. Scoped to the GLES/EGL path.
    /// Cached on first query.
    pub fn batchingEnabled() bool {
        return batching_enabled_cache orelse blk: {
            const on = std.c.getenv("PRISM_NOBATCH") == null;
            batching_enabled_cache = on;
            break :blk on;
        };
    }

    /// Check out the next pooled vertex buffer (>= `size` bytes), growing the pool / a buffer as
    /// needed. Returns the Resource + its full mapping (caller writes the first `size`).
    /// Advances `vb_used`. In non-batched mode vb_used resets each submit and reuses pool[0].
    /// In batched mode successive draws take pool[0], pool[1], ...
    pub fn checkoutVertexBuffer(self: *Context, size: usize) prism.Error!PoolVb {
        const dev = self.display.device.?;
        // No batch open => the previous batch (if any) has flushed + fenced, so every pooled buffer
        // is free to reuse: reclaim them all. In non-batched mode batch_cb is always null, so this
        // resets to pool[0] every draw (the single reused-scratch behaviour). In batched mode it
        // resets only at the start of a fresh run of draws, bounding the pool to one batch's draws.
        if (self.batch_cb == null) self.vb_used = 0;
        if (self.vb_used < self.vb_pool.items.len) {
            const p = &self.vb_pool.items[self.vb_used];
            if (p.cap >= size) {
                self.vb_used += 1;
                return p.*;
            }
            // Grow this slot: free the too-small buffer and replace it in place.
            dev.destroyResource(p.res);
            const grown = try allocPoolVb(dev, size);
            p.* = grown;
            self.vb_used += 1;
            return grown;
        }
        const fresh = try allocPoolVb(dev, size);
        try self.vb_pool.append(gpa, fresh);
        self.vb_used += 1;
        return fresh;
    }

    fn allocPoolVb(dev: prism.hal.Device, size: usize) prism.Error!PoolVb {
        const cap = @max(size, 4096); // headroom to reduce reallocation churn as draws grow
        const vb = try dev.createResource(.{ .buffer = .{ .size = cap, .usage = .{ .vertex = true } } });
        const map = dev.mapResource(vb) catch |e| {
            dev.destroyResource(vb);
            return e;
        };
        return .{ .res = vb, .map = map, .cap = cap };
    }

    /// Submit any open batch of accumulated draws (one submit for all of them) and reset. A no-op
    /// when no batch is open. Call before anything reads GPU-rendered results, changes the render
    /// target, or frees a resource a pending draw references (see call sites). Pooled vertex
    /// buffers are reclaimed lazily by the next checkoutVertexBuffer (batch_cb is null after).
    pub fn flushDraws(self: *Context) prism.Error!void {
        const cb = self.batch_cb orelse return;
        self.batch_cb = null;
        self.batch_target = null;
        self.batch_depth = null;
        self.batch_stencil = null;
        self.batch_draws = 0;
        defer cb.deinit();
        try self.hal.submit(cb);
    }

    /// Clear `surface`'s HAL backbuffer image to `color` through the proven software
    /// HAL clear flow (beginCommands -> setRenderTarget -> clear -> submit). The result
    /// lands in the backbuffer's bytes, readable for a pbuffer and presentable for a
    /// window surface.
    pub fn clearBackbuffer(self: *Context, surface: *Surface, color: prism.hal.Color, scissor: ?prism.hal.ScissorRect) prism.Error!void {
        // Clear the render target (multisampled backbuffer for MSAA, else the plain backbuffer).
        // swapBuffers resolves it into the presented image.
        try self.clearTarget(surface.renderTarget(), color, scissor);
    }

    /// Clear an arbitrary render-target image (the surface backbuffer OR a bound FBO's
    /// color attachment) to `color`, through the same HAL clear flow. Render-to-texture
    /// (an FBO bound) routes its clear here with the FBO's color image. `scissor` clips the
    /// clear to a sub-rect when non-null (GL glClear honors an enabled scissor test).
    pub fn clearTarget(self: *Context, target: *prism.hal.Resource, color: prism.hal.Color, scissor: ?prism.hal.ScissorRect) prism.Error!void {
        const cb = try self.hal.beginCommands();
        defer cb.deinit();
        try cb.setRenderTarget(target);
        try cb.setScissor(scissor);
        try cb.clear(color);
        try self.hal.submit(cb);
    }

    /// Clear a depth attachment to `value` (GL_DEPTH_BUFFER_BIT clear of a bound FBO's depth
    /// texture/renderbuffer). A trivial render pass with color + depth bound, no draw. The
    /// software setDepthTarget clears all f32 depths. A color target is bound too so the
    /// nvidia render pass has a valid surface (CLEAR_SURFACE Z clear runs even with no draw).
    /// Software ignores the unused RT.
    pub fn clearDepthTarget(self: *Context, color: *prism.hal.Resource, depth: *prism.hal.Resource, value: f32) prism.Error!void {
        const cb = try self.hal.beginCommands();
        defer cb.deinit();
        try cb.setRenderTarget(color);
        try cb.setDepthTarget(depth, value);
        try self.hal.submit(cb);
    }

    /// Rasterize `vertex_count` vertices of `pipeline` from `vertex_buffer` into
    /// `surface`'s HAL backbuffer. Uses the same HAL flow as the triangle_hal example:
    /// beginCommands -> setRenderTarget -> [optional clear] -> bindPipeline ->
    /// bindVertexBuffer -> draw -> submit. The pipeline is built from the GLES program's
    /// SPIR-V VS+FS modules, so this rides spirv.zig + the Vulcan JIT verbatim. `clear`
    /// is applied as the render-pass clear color. Pass null to draw over existing contents.
    pub fn drawArrays(
        self: *Context,
        surface: *Surface,
        pipeline: *prism.hal.Pipeline,
        vertex_buffer: *prism.hal.Resource,
        first_vertex: u32,
        vertex_count: u32,
        clear: ?prism.hal.Color,
    ) prism.Error!void {
        const cb = try self.hal.beginCommands();
        defer cb.deinit();
        try cb.setRenderTarget(surface.backbuffer);
        if (clear) |c| try cb.clear(c);
        try cb.bindPipeline(pipeline);
        try cb.bindVertexBuffer(vertex_buffer);
        try cb.draw(vertex_count, first_vertex);
        try self.hal.submit(cb);
    }

    /// Like `drawArrays` but additionally binds a uniform buffer (default-uniform-block
    /// bytes as a HAL UBO) and an optional depth attachment (cleared to `depth_clear`).
    /// Same HAL flow: beginCommands -> setRenderTarget -> setDepthTarget -> [clear] ->
    /// bindPipeline -> bindUniformBuffer -> bindVertexBuffer -> draw -> submit.
    /// The pipeline carries depth-test + cull state. Rides the software driver verbatim.
    pub fn drawArraysUbo(
        self: *Context,
        surface: *Surface,
        pipeline: *prism.hal.Pipeline,
        vertex_buffer: *prism.hal.Resource,
        first_vertex: u32,
        vertex_count: u32,
        clear: ?prism.hal.Color,
        uniform_buffer: ?*prism.hal.Resource,
        depth: ?*prism.hal.Resource,
        depth_clear: ?f32,
        textures: []const prism.hal.TextureBinding,
    ) prism.Error!void {
        try self.drawArraysUboTarget(surface.backbuffer, pipeline, vertex_buffer, first_vertex, vertex_count, 1, 0, clear, uniform_buffer, depth, depth_clear, null, null, textures, null, null, &.{});
    }

    /// Like `drawArraysUbo` but draws into an explicit color render-target Resource (the
    /// surface backbuffer or a bound FBO's color attachment / depth-texture scratch image),
    /// for GLES render-to-texture. Same HAL flow. Nothing else changes.
    pub fn drawArraysUboTarget(
        self: *Context,
        target: *prism.hal.Resource,
        pipeline: *prism.hal.Pipeline,
        vertex_buffer: *prism.hal.Resource,
        first_vertex: u32,
        vertex_count: u32,
        instance_count: u32,
        first_instance: u32,
        clear: ?prism.hal.Color,
        vs_uniform_buffer: ?*prism.hal.Resource,
        fs_uniform_buffer: ?*prism.hal.Resource,
        depth: ?*prism.hal.Resource,
        depth_clear: ?f32,
        stencil: ?*prism.hal.Resource,
        stencil_clear: ?u8,
        textures: []const prism.hal.TextureBinding,
        scissor: ?prism.hal.ScissorRect,
        viewport: ?prism.hal.Viewport,
        extra_color_targets: []const *prism.hal.Resource,
    ) prism.Error!void {
        // Accumulate compatible draws into one command buffer, deferring submit+fence to the
        // next flush point (swap / readback / RT change / resource free). A draw joins the open
        // batch only if it targets the same color/depth/stencil surfaces, is not MRT, and does
        // not itself clear (a mid-frame clear must re-clear, and the one-submit model
        // applies that clear only at the front). Otherwise flush first, then start fresh.
        if (Context.batchingEnabled() and extra_color_targets.len == 0) {
            if (self.batch_cb != null) {
                const can_join = self.batch_target == target and self.batch_depth == depth and
                    self.batch_stencil == stencil and clear == null and depth_clear == null and stencil_clear == null;
                if (!can_join) try self.flushDraws();
            }
            if (self.batch_cb == null) {
                self.batch_cb = try self.hal.beginCommands();
                self.batch_target = target;
                self.batch_depth = depth;
                self.batch_stencil = stencil;
                self.batch_draws = 0;
            }
            try recordDrawInto(self.batch_cb.?, target, pipeline, vertex_buffer, first_vertex, vertex_count, instance_count, first_instance, clear, vs_uniform_buffer, fs_uniform_buffer, depth, depth_clear, stencil, stencil_clear, textures, scissor, viewport, extra_color_targets);
            self.batch_draws += 1;
            // Safety cap: bounds vertex-buffer pool + command-buffer growth. A normal frame flushes
            // at swap/readback well under this limit. A pathological loop that never flushes would
            // otherwise allocate one vertex buffer per draw forever. Each flush is a valid submit.
            // Real frames are unaffected.
            if (self.batch_draws >= BATCH_DRAW_CAP) try self.flushDraws();
            return; // deferred: submitted at the next flushDraws
        }
        // Non-batched: record one draw and submit immediately (fences per draw).
        const cb = try self.hal.beginCommands();
        defer cb.deinit();
        try recordDrawInto(cb, target, pipeline, vertex_buffer, first_vertex, vertex_count, instance_count, first_instance, clear, vs_uniform_buffer, fs_uniform_buffer, depth, depth_clear, stencil, stencil_clear, textures, scissor, viewport, extra_color_targets);
        try self.hal.submit(cb);
    }

    /// Record one draw (render-target + attachment binds, optional clear, pipeline/UBO/texture binds,
    /// vertex buffer, draw) into `cb`. Shared by the immediate and batched paths. Does not submit.
    fn recordDrawInto(
        cb: prism.hal.CommandBuffer,
        target: *prism.hal.Resource,
        pipeline: *prism.hal.Pipeline,
        vertex_buffer: *prism.hal.Resource,
        first_vertex: u32,
        vertex_count: u32,
        instance_count: u32,
        first_instance: u32,
        clear: ?prism.hal.Color,
        vs_uniform_buffer: ?*prism.hal.Resource,
        fs_uniform_buffer: ?*prism.hal.Resource,
        depth: ?*prism.hal.Resource,
        depth_clear: ?f32,
        stencil: ?*prism.hal.Resource,
        stencil_clear: ?u8,
        textures: []const prism.hal.TextureBinding,
        scissor: ?prism.hal.ScissorRect,
        viewport: ?prism.hal.Viewport,
        extra_color_targets: []const *prism.hal.Resource,
    ) prism.Error!void {
        try cb.setRenderTarget(target);
        // MRT: bind GL_COLOR_ATTACHMENT1..N as HAL extra color targets (index 0 = attachment 1).
        // A fragment shader's located `out`s write each. Unbound slots keep only attachment 0.
        for (extra_color_targets, 0..) |ect, i| try cb.setColorTarget(@intCast(i + 1), ect);
        // Set the scissor BEFORE the clear so glClear under an enabled scissor test clips
        // to the same rect (GL semantics). null disables clipping (full render target).
        try cb.setScissor(scissor);
        // glViewport: NDC maps into this window rect + rasterization clips to it. null = full RT.
        try cb.setViewport(viewport);
        // Bind the depth attachment and pass the clear value through verbatim. A null
        // depth_clear means "preserve": the HAL binds the depth target without clearing,
        // so depth accumulated by earlier draws this frame survives (the GLES clear-once-
        // then-draw-many contract). Earlier this forced `orelse 1.0`, which re-cleared the
        // depth buffer on every draw and broke multi-primitive depth occlusion.
        if (depth) |d| try cb.setDepthTarget(d, depth_clear);
        // Bind the stencil attachment with the same deferred-clear contract as depth.
        if (stencil) |s| try cb.setStencilTarget(s, stencil_clear);
        if (clear) |c| try cb.clear(c);
        try cb.bindPipeline(pipeline);
        // GLES2 default-uniform blocks are per-stage: the VS block binds at binding 0, the FS
        // block at binding 1 (samplers are at binding 2+). The shaders read each from its own
        // constant-bank slot, so a VS-only and an FS-only uniform never collide.
        if (vs_uniform_buffer) |ub| try cb.bindUniformBuffer(0, ub);
        if (fs_uniform_buffer) |ub| try cb.bindUniformBuffer(1, ub);
        // Bind combined-image-sampler textures (the ICD's vkcube texture path). The
        // software driver records each and builds a host TexDesc for the JITed FS.
        for (textures) |t| try cb.bindTexture(t);
        try cb.bindVertexBuffer(vertex_buffer);
        if (instance_count <= 1 and first_instance == 0) {
            try cb.draw(vertex_count, first_vertex);
        } else {
            // first_instance shifts gl_InstanceIndex (the per-instance-attribute-divisor
            // emulation submits one instance at a time, each with its own base index).
            try cb.drawInstanced(vertex_count, instance_count, first_vertex, first_instance);
        }
    }

    /// The HAL device backing this context (for GLES object allocation: shader
    /// modules, vertex buffers, pipelines).
    pub fn device(self: *Context) prism.hal.Device {
        return self.display.device.?;
    }

    /// Box-downsample a multisampled color image `src` (samples=N) into the single-sample `dst`
    /// via the HAL resolve command (one submit). Used to resolve a multisampled FBO color
    /// renderbuffer before glBlitFramebuffer / glReadPixels reads it.
    pub fn resolveMsaa(self: *Context, src: *prism.hal.Resource, dst: *prism.hal.Resource, w: u32, h: u32, format: prism.hal.Format, samples: u8) prism.Error!void {
        const cb = try self.hal.beginCommands();
        defer cb.deinit();
        try cb.resolve(src, dst, w, h, format, samples);
        try self.hal.submit(cb);
    }
};

/// Live-object registries. Contexts + surfaces validate by identity so a bogus
/// or misaligned handle yields null instead of a panic. Guarded by display_lock.
var contexts: std.ArrayListUnmanaged(*Context) = .empty;
var surfaces: std.ArrayListUnmanaged(*Surface) = .empty;

fn registerContext(c: *Context) bool {
    display_lock.lock();
    defer display_lock.unlock();
    contexts.append(gpa, c) catch return false;
    return true;
}
fn unregisterContext(c: *Context) void {
    display_lock.lock();
    defer display_lock.unlock();
    for (contexts.items, 0..) |live, i| if (live == c) {
        _ = contexts.swapRemove(i);
        return;
    };
}
fn registerSurface(s: *Surface) bool {
    display_lock.lock();
    defer display_lock.unlock();
    surfaces.append(gpa, s) catch return false;
    return true;
}
fn unregisterSurface(s: *Surface) void {
    display_lock.lock();
    defer display_lock.unlock();
    for (surfaces.items, 0..) |live, i| if (live == s) {
        _ = surfaces.swapRemove(i);
        return;
    };
}

/// Create a render context on `display` for `config_index`, bound to the current
/// client API. Brings up nothing new beyond a HAL Context over the display's device.
pub fn createContext(display: *Display, config_index: usize, client_api: egl.EGLenum) prism.Error!*Context {
    const dev = display.device orelse return error.InitializationFailed;
    const hal_ctx = try dev.createContext();
    errdefer hal_ctx.deinit();
    const c = gpa.create(Context) catch return error.OutOfMemory;
    c.* = .{ .display = display, .config = config_index, .client_api = client_api, .hal = hal_ctx };
    if (!registerContext(c)) {
        hal_ctx.deinit();
        gpa.destroy(c);
        return error.OutOfMemory;
    }
    return c;
}

// --- Surfaces ---------------------------------------------------------------

pub const SurfaceKind = enum {
    /// A Prism platform.Surface window (Prism's self-contained EGL_EXT_platform_
    /// wayland: the client owns the platform surface). Present via the HAL.
    window,
    /// A stock app's `wl_egl_window *` over its own libwayland `wl_display`.
    /// Present via the system libwayland (wl_shm on the app's wl_surface).
    wl_egl_window,
    pbuffer,
};

/// A Prism EGL surface: a HAL backbuffer image (the render target the GLES clear
/// writes) plus, for a window surface, the Prism platform.Surface to present to.
/// A pbuffer is offscreen (no platform surface): swap is a no-op and the client
/// reads the backbuffer back. A window surface presents the backbuffer through the
/// platform layer on eglSwapBuffers (the example's commit/present path).
pub const Surface = struct {
    kind: SurfaceKind,
    display: *Display,
    config: usize,
    width: u32,
    height: u32,
    /// The render-target image the GLES clear/draw writes (rgba8_unorm, CPU-readable). For an
    /// MSAA config this is the resolved single-sample image that is presented/read back. The
    /// multisampled render target is `msaa_backbuffer` and swapBuffers resolves into this.
    backbuffer: *prism.hal.Resource,
    /// EGL_SAMPLES from the config (1 = no MSAA). When >1 the surface renders into
    /// `msaa_backbuffer` (a samples=N image) and resolves into `backbuffer` on swap.
    samples: u8 = 1,
    /// The multisampled render target (a color image with samples=`samples`), only for
    /// an MSAA config. Draws/clears target this. swapBuffers box-resolves it into
    /// `backbuffer`. Null for a non-MSAA surface (draws target `backbuffer` directly).
    msaa_backbuffer: ?*prism.hal.Resource = null,
    /// The depth attachment (depth32_float-sized buffer, one f32 per pixel), allocated
    /// lazily the first time a depth-tested draw needs it. The config's EGL_DEPTH_SIZE
    /// advertises depth. Created on demand so a color-only surface pays nothing. Cleared
    /// by the render pass (setDepthTarget) on each depth-clearing draw.
    depth_buffer: ?*prism.hal.Resource = null,
    /// The stencil attachment (one u8 per pixel, a plain buffer Resource), allocated lazily
    /// the first time a stencil-tested draw needs it. The config's EGL_STENCIL_SIZE
    /// advertises stencil. The software driver reads its bytes as a u8-per-pixel buffer.
    stencil_buffer: ?*prism.hal.Resource = null,
    /// The HAL surface to present to (window surfaces only; null for a pbuffer).
    hal_surface: ?*prism.hal.Surface = null,
    /// The platform surface (window surfaces only). For a Wayland window it is the
    /// caller-owned EGL native window. For a GBM window it is a Prism-owned adapter
    /// over the app's gbm.Surface (see owns_plat_surface).
    plat_surface: ?*prism.platform.Surface = null,
    /// Whether this surface owns plat_surface and must free it on deinit. True for a
    /// GBM window (Prism heap-allocates the adapter), false for a Wayland window.
    owns_plat_surface: bool = false,
    /// The raw `wl_egl_window *` (wl_egl_window surfaces only). Caller-owned. Read
    /// fresh on each swap so a `wl_egl_window_resize` is honored.
    egl_window: ?*anyopaque = null,
    /// The libwayland present state (wl_egl_window surfaces only): wl_shm buffers
    /// on the app's wl_display, attached to the app's wl_surface on swap. Created
    /// lazily on the first present (and re-created on a window resize).
    wl_present: ?*wlw.WaylandPresent = null,

    pub fn deinit(self: *Surface) void {
        unregisterSurface(self);
        const dev = self.display.device.?;
        if (self.wl_present) |wp| wp.deinit();
        if (self.hal_surface) |hs| dev.destroySurface(hs);
        // A GBM window owns its platform.Surface adapter (freed after the HAL
        // surface that referenced it). A Wayland window's is caller-owned.
        if (self.owns_plat_surface) if (self.plat_surface) |ps| {
            ps.deinit();
            gpa.destroy(ps);
        };
        if (self.depth_buffer) |db| dev.destroyResource(db);
        if (self.stencil_buffer) |sb| dev.destroyResource(sb);
        if (self.msaa_backbuffer) |mb| dev.destroyResource(mb);
        dev.destroyResource(self.backbuffer);
        gpa.destroy(self);
    }

    /// The image GLES draws/clears write into: the multisampled render target for an MSAA
    /// surface, else the plain backbuffer. swapBuffers resolves the MSAA target into
    /// `backbuffer` (the presented/read-back image).
    pub fn renderTarget(self: *Surface) *prism.hal.Resource {
        return self.msaa_backbuffer orelse self.backbuffer;
    }

    /// Resolve the multisampled render target into `backbuffer` (a no-op for a non-MSAA
    /// surface). Called by swapBuffers before present / read-back so the single-sample
    /// `backbuffer` holds the box-averaged result. The MS target was rendered + fenced by
    /// the draw submits, so the HAL resolve is a pure downsample.
    pub fn resolveMSAA(self: *Surface, ctx: *Context) prism.Error!void {
        const ms = self.msaa_backbuffer orelse return;
        const cb = try ctx.hal.beginCommands();
        defer cb.deinit();
        try cb.resolve(ms, self.backbuffer, self.width, self.height, .rgba8_unorm, self.samples);
        try ctx.hal.submit(cb);
    }

    /// Get-or-create the surface's depth attachment (depth32_float, one f32 per pixel).
    /// Allocated lazily so a color-only surface never pays for it. The depth-tested draw
    /// path requests it. The software driver reinterprets its bytes as an f32 depth buffer.
    /// The render pass (setDepthTarget) clears it, matching the ICD's depth path.
    pub fn depthAttachment(self: *Surface) prism.Error!*prism.hal.Resource {
        if (self.depth_buffer) |db| return db;
        const dev = self.display.device.?;
        const db = try dev.createResource(.{
            .image = .{
                .width = self.width,
                .height = self.height,
                .format = .depth32_float,
                .samples = self.samples, // match the MSAA render dims (1 for a non-MSAA surface)
                .usage = .{ .render_target = true },
            },
        });
        self.depth_buffer = db;
        return db;
    }

    /// Get-or-create the surface's stencil attachment (one u8 per pixel, a buffer Resource).
    /// Allocated lazily so a non-stencil surface never pays for it. The software driver's
    /// setStencilTarget reads its bytes as a u8-per-pixel stencil buffer.
    pub fn stencilAttachment(self: *Surface) prism.Error!*prism.hal.Resource {
        if (self.stencil_buffer) |sb| return sb;
        const dev = self.display.device.?;
        const sb = try dev.createResource(.{ .buffer = .{ .size = @as(usize, self.width) * self.height } });
        self.stencil_buffer = sb;
        return sb;
    }
};

/// Allocate the common backbuffer image (rgba8_unorm, render-target + CPU-readable).
fn createBackbuffer(dev: prism.hal.Device, width: u32, height: u32) prism.Error!*prism.hal.Resource {
    return dev.createResource(.{ .image = .{
        .width = width,
        .height = height,
        .format = .rgba8_unorm,
        .usage = .{ .render_target = true, .copy_src = true },
    } });
}

/// If `config_index` selects an MSAA config (EGL_SAMPLES>1), allocate the surface's
/// multisampled render target (a samples=N color image) so draws render into it. The
/// single-sample `backbuffer` becomes the resolve/present target. A no-op otherwise.
/// Call after the surface is registered so its deinit reclaims the MS image on failure.
fn setupMSAA(s: *Surface) prism.Error!void {
    const samp = configs[s.config].samples;
    if (samp <= 0) return;
    const dev = s.display.device.?;
    const n: u8 = @intCast(samp);
    s.msaa_backbuffer = try dev.createResource(.{ .image = .{
        .width = s.width,
        .height = s.height,
        .format = .rgba8_unorm,
        .samples = n,
        .usage = .{ .render_target = true },
    } });
    s.samples = n;
}

/// Create an offscreen pbuffer surface (width/height from the attrib list). No
/// platform surface: eglSwapBuffers is a no-op and the client reads the backbuffer.
pub fn createPbufferSurface(display: *Display, config_index: usize, width: u32, height: u32) prism.Error!*Surface {
    const dev = display.device orelse return error.InitializationFailed;
    const bb = try createBackbuffer(dev, width, height);
    errdefer dev.destroyResource(bb);
    const s = gpa.create(Surface) catch return error.OutOfMemory;
    s.* = .{
        .kind = .pbuffer,
        .display = display,
        .config = config_index,
        .width = width,
        .height = height,
        .backbuffer = bb,
    };
    if (!registerSurface(s)) {
        dev.destroyResource(bb);
        gpa.destroy(s);
        return error.OutOfMemory;
    }
    setupMSAA(s) catch {
        s.deinit();
        return error.OutOfMemory;
    };
    return s;
}

/// Create a window surface over a Prism platform.Surface (the EGL native window
/// handle). The backbuffer is sized to the platform surface. eglSwapBuffers
/// presents it through the platform layer (the example's HAL present/commit path).
pub fn createWindowSurface(display: *Display, config_index: usize, plat_surface: *prism.platform.Surface) prism.Error!*Surface {
    return makeWindowSurface(display, config_index, plat_surface, false);
}

/// Create an EGL window surface over an app's gbm.Surface on a GBM-platform
/// display. Prism wraps the gbm.Surface in a platform.Surface adapter (which it
/// owns) and presents into the gbm back buffer on eglSwapBuffers. The app then
/// scans out the front buffer via gbm_surface_lock_front_buffer.
pub fn createGbmWindowSurface(display: *Display, config_index: usize, gbm_surface: *prism.platform.gbm.Surface) prism.Error!*Surface {
    const plat = gpa.create(prism.platform.Surface) catch return error.OutOfMemory;
    plat.* = prism.platform.gbm.wrapSurface(gpa, gbm_surface) catch |e| {
        gpa.destroy(plat);
        return e;
    };
    return makeWindowSurface(display, config_index, plat, true) catch |e| {
        plat.deinit();
        gpa.destroy(plat);
        return e;
    };
}

/// Build a window surface presenting through `plat_surface`. When `owns` is set
/// the surface frees `plat_surface` on deinit (the GBM adapter); otherwise the
/// caller owns it (a Wayland native window).
fn makeWindowSurface(display: *Display, config_index: usize, plat_surface: *prism.platform.Surface, owns: bool) prism.Error!*Surface {
    const dev = display.device orelse return error.InitializationFailed;
    const dims = plat_surface.size();
    const bb = try createBackbuffer(dev, dims[0], dims[1]);
    errdefer dev.destroyResource(bb);
    const hal_surface = try dev.createSurface(@ptrCast(plat_surface));
    errdefer dev.destroySurface(hal_surface);
    const s = gpa.create(Surface) catch return error.OutOfMemory;
    s.* = .{
        .kind = .window,
        .display = display,
        .config = config_index,
        .width = dims[0],
        .height = dims[1],
        .backbuffer = bb,
        .hal_surface = hal_surface,
        .plat_surface = plat_surface,
        .owns_plat_surface = owns,
    };
    if (!registerSurface(s)) {
        dev.destroySurface(hal_surface);
        dev.destroyResource(bb);
        gpa.destroy(s);
        return error.OutOfMemory;
    }
    setupMSAA(s) catch {
        // The caller frees plat_surface on any makeWindowSurface error, so drop
        // ownership before deinit to avoid a double free.
        s.owns_plat_surface = false;
        s.deinit();
        return error.OutOfMemory;
    };
    return s;
}

/// Create an EGL window surface over a stock app's `wl_egl_window *` on a display
/// bound to the app's own `wl_display`. The wl_egl_window is parsed for its
/// wl_surface + size; the backbuffer is sized to the window. On eglSwapBuffers
/// we present via the system libwayland (wl_shm on the app's display, attached
/// to the app's wl_surface), reusing the ICD's interop path.
pub fn createWaylandWindowSurface(
    display: *Display,
    config_index: usize,
    egl_window: *anyopaque,
) prism.Error!*Surface {
    const dev = display.device orelse return error.InitializationFailed;
    const parsed = wlw.parse(egl_window) orelse return error.InvalidArgument;
    const bb = try createBackbuffer(dev, parsed.width, parsed.height);
    errdefer dev.destroyResource(bb);
    const s = gpa.create(Surface) catch return error.OutOfMemory;
    s.* = .{
        .kind = .wl_egl_window,
        .display = display,
        .config = config_index,
        .width = parsed.width,
        .height = parsed.height,
        .backbuffer = bb,
        .egl_window = egl_window,
    };
    if (!registerSurface(s)) {
        dev.destroyResource(bb);
        gpa.destroy(s);
        return error.OutOfMemory;
    }
    setupMSAA(s) catch {
        s.deinit();
        return error.OutOfMemory;
    };
    return s;
}

/// Present a window surface's backbuffer through the platform layer (the proven
/// HAL present: blit the backbuffer into the platform buffer + commit). A pbuffer
/// swap is a no-op (offscreen). A wl_egl_window swap presents via the system
/// libwayland (wl_shm on the app's wl_display). Returns the EGL contract success.
pub fn swapBuffers(ctx: *Context, surface: *Surface) prism.Error!void {
    // Submit any batched draws before the frame is resolved / presented / read back.
    try ctx.flushDraws();
    // MSAA: resolve the multisampled render target into the single-sample backbuffer that is
    // presented / read back. A no-op for a non-MSAA surface. Done for a pbuffer too, so an
    // offscreen MSAA render is resolved before the client reads the backbuffer.
    try surface.resolveMSAA(ctx);
    switch (surface.kind) {
        .pbuffer => {}, // offscreen: nothing to present (the resolve above made backbuffer readable)
        .window => {
            const hs = surface.hal_surface orelse return error.InvalidArgument;
            try ctx.hal.present(hs, surface.backbuffer);
        },
        .wl_egl_window => try swapWaylandWindow(ctx, surface),
    }
}

/// Number of wl_shm buffers behind a wl_egl_window surface (double-buffered).
const WL_SWAPCHAIN_BUFFERS: u32 = 3; // triple-buffer: slack so present rarely blocks on release

/// Present a wl_egl_window surface: read the (possibly resized) window, ensure a
/// wl_shm swapchain on the app's display, blit the rendered HAL backbuffer into a
/// free buffer (RGBA8 -> XRGB8888), then attach + commit on the app's wl_surface.
fn swapWaylandWindow(ctx: *Context, surface: *Surface) prism.Error!void {
    const app_display = surface.display.appWlDisplay() orelse return error.InvalidArgument;
    const raw = surface.egl_window orelse return error.InvalidArgument;
    const parsed = wlw.parse(raw) orelse return error.InvalidArgument;

    // Honor a wl_egl_window_resize: if the window grew/shrank, rebuild the HAL
    // backbuffer + the wl_shm swapchain to the new size.
    if (parsed.width != surface.width or parsed.height != surface.height) {
        const dev = surface.display.device.?;
        const new_bb = try createBackbuffer(dev, parsed.width, parsed.height);
        dev.destroyResource(surface.backbuffer);
        surface.backbuffer = new_bb;
        if (surface.depth_buffer) |db| {
            dev.destroyResource(db);
            surface.depth_buffer = null;
        }
        // Rebuild the multisampled render target at the new size (MSAA surfaces only).
        if (surface.msaa_backbuffer) |mb| {
            dev.destroyResource(mb);
            surface.msaa_backbuffer = dev.createResource(.{ .image = .{
                .width = parsed.width,
                .height = parsed.height,
                .format = .rgba8_unorm,
                .samples = surface.samples,
                .usage = .{ .render_target = true },
            } }) catch null;
        }
        if (surface.wl_present) |wp| {
            wp.deinit();
            surface.wl_present = null;
        }
        surface.width = parsed.width;
        surface.height = parsed.height;
        return; // this frame was rendered at the old size. the app re-renders
    }

    // Lazily create the wl_shm swapchain on the app's display + wl_surface.
    if (surface.wl_present == null) {
        surface.wl_present = wlw.WaylandPresent.init(
            gpa,
            app_display,
            parsed.surface,
            surface.width,
            surface.height,
            WL_SWAPCHAIN_BUFFERS,
        ) catch return error.InitializationFailed;
    }
    const wp = surface.wl_present.?;

    const dev = ctx.display.device.?;
    const idx = wp.acquire();
    const dst = wp.buffers[idx].pixels;
    // Fast path: let the driver de-swizzle the rendered RT straight into the wl_shm
    // XRGB8888 buffer in ONE pass (no linear-copy intermediate, no separate blit).
    // Falls back to mapResource + a CPU RGBA->XRGB blit on drivers without it (software).
    if (!try dev.tryReadbackPresent(surface.backbuffer, dst, @as(usize, surface.width) * 4)) {
        const src = try dev.mapResource(surface.backbuffer);
        wlw.blitRgbaToXrgb(dst, src, surface.width, surface.height);
    }
    if (std.c.getenv("PRISM_DUMP") != null) dumpFrame(dst, surface.width, surface.height);
    wp.present(idx);
}

// Debug-only: when PRISM_DUMP is set, write every 20th presented frame to a numbered
// PPM under /tmp/prism_dump so the actual rendered output can be inspected/compared.
var dump_frame: u64 = 0;
var dump_written: u32 = 0;
fn dumpFrame(xrgb: []const u8, w: u32, h: u32) void {
    dump_frame += 1;
    if ((dump_frame % 4 != 0 and dump_frame > 2) or dump_written >= 80) return;
    const linux = std.os.linux;
    var namebuf: [64]u8 = undefined;
    const name = std.fmt.bufPrintZ(&namebuf, "/tmp/prism_dump/f{d:0>4}.ppm", .{dump_written}) catch return;
    const fd_us = linux.open(name.ptr, .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true }, 0o644);
    if (@as(isize, @bitCast(fd_us)) < 0) return;
    const fd: i32 = @intCast(fd_us);
    defer _ = linux.close(fd);
    var hdr: [32]u8 = undefined;
    const hs = std.fmt.bufPrint(&hdr, "P6\n{d} {d}\n255\n", .{ w, h }) catch return;
    _ = linux.write(fd, hs.ptr, hs.len);
    // XRGB8888 little-endian bytes are [B,G,R,X]. PPM wants R,G,B.
    var row: [4096 * 3]u8 = undefined;
    var y: u32 = 0;
    while (y < h) : (y += 1) {
        var x: u32 = 0;
        while (x < w) : (x += 1) {
            const si = (@as(usize, y) * w + x) * 4;
            row[x * 3 + 0] = xrgb[si + 2];
            row[x * 3 + 1] = xrgb[si + 1];
            row[x * 3 + 2] = xrgb[si + 0];
        }
        _ = linux.write(fd, &row, w * 3);
    }
    dump_written += 1;
}

// --- Per-thread current context + surfaces (eglMakeCurrent) -----------------

const Current = struct {
    ctx: ?*Context = null,
    draw: ?*Surface = null,
    read: ?*Surface = null,
};
threadlocal var current: Current = .{};

/// eglMakeCurrent: bind ctx + draw/read for this thread (or unbind all on null).
pub fn makeCurrent(ctx: ?*Context, draw: ?*Surface, read: ?*Surface) void {
    // Flush the outgoing context's batched draws before it stops being current. Pending draws
    // target the surface it was bound to and must not carry across a context switch or teardown.
    if (current.ctx) |old| old.flushDraws() catch {};
    current = .{ .ctx = ctx, .draw = draw, .read = read };
}

pub fn currentContext() ?*Context {
    return current.ctx;
}
pub fn currentDrawSurface() ?*Surface {
    return current.draw;
}
pub fn currentReadSurface() ?*Surface {
    return current.read;
}

/// Validate a handle is a live context by identity in the registry. No @alignCast
/// on an unvalidated handle: a bogus pointer yields null, never a safety panic.
pub fn lookupContext(handle: egl.EGLContext) ?*Context {
    if (handle == null) return null;
    const v = @intFromPtr(handle.?);
    display_lock.lock();
    defer display_lock.unlock();
    for (contexts.items) |live| if (@intFromPtr(live) == v) return live;
    return null;
}

/// Validate a handle is a live surface by identity in the registry.
pub fn lookupSurface(handle: egl.EGLSurface) ?*Surface {
    if (handle == null) return null;
    const v = @intFromPtr(handle.?);
    display_lock.lock();
    defer display_lock.unlock();
    for (surfaces.items) |live| if (@intFromPtr(live) == v) return live;
    return null;
}

/// Atomic spinlock for the display registry. EGL is callable from any thread and
/// we have no Io handle for std.Io.Mutex. The registry is small and rarely contended.
const SpinLock = struct {
    flag: std.atomic.Value(bool) = .init(false),
    fn lock(self: *SpinLock) void {
        while (self.flag.cmpxchgWeak(false, true, .acquire, .monotonic) != null) {
            std.atomic.spinLoopHint();
        }
    }
    fn unlock(self: *SpinLock) void {
        self.flag.store(false, .release);
    }
};

/// Registry of live displays. eglGetPlatformDisplay returns the same handle for
/// the same (platform, native) pair, per the EGL spec.
var display_lock: SpinLock = .{};
var displays: std.ArrayListUnmanaged(*Display) = .empty;

/// Whether a platform enum is one Prism owns. EGL_PLATFORM_NONE (from
/// eglGetDisplay(EGL_DEFAULT_DISPLAY)) is accepted and defaults to surfaceless.
pub fn supportsPlatform(platform: egl.EGLenum) bool {
    return switch (platform) {
        egl.EGL_PLATFORM_SURFACELESS_MESA,
        egl.EGL_PLATFORM_WAYLAND_KHR,
        egl.EGL_PLATFORM_GBM_KHR,
        egl.EGL_PLATFORM_NONE,
        => true,
        else => false,
    };
}

/// Find-or-create the Display for a (platform, native) pair. Returns null if the
/// platform is not one we support. Caller maps null to EGL_NO_DISPLAY without an
/// error (per the GLVND contract; another vendor may own it).
pub fn getPlatformDisplay(platform: egl.EGLenum, native: ?*anyopaque) ?*Display {
    if (!supportsPlatform(platform)) return null;

    display_lock.lock();
    defer display_lock.unlock();

    for (displays.items) |d| {
        if (d.platform == platform and d.native == native) return d;
    }
    const d = gpa.create(Display) catch return null;
    d.* = .{ .platform = platform, .native = native };
    displays.append(gpa, d) catch {
        gpa.destroy(d);
        return null;
    };
    return d;
}

/// Validate that a handle is a live Display we created.
pub fn lookupDisplay(handle: egl.EGLDisplay) ?*Display {
    if (handle == null) return null;
    // Compare by raw pointer value. No @alignCast on an unvalidated handle:
    // a bogus/misaligned EGLDisplay must yield null, not a safety panic.
    const v = @intFromPtr(handle.?);
    display_lock.lock();
    defer display_lock.unlock();
    for (displays.items) |live| {
        if (@intFromPtr(live) == v) return live;
    }
    return null;
}

/// eglInitialize: bring the display up and report EGL 1.5. Creates the backing
/// platform display (headless for surfaceless / no-display M1). Idempotent.
pub fn initialize(d: *Display) prism.Error!void {
    if (d.initialized) return;
    // Enumeration is display-free: back it with the headless platform display (no
    // Wayland connection needed to enumerate configs or to allocate offscreen
    // pbuffers). A window surface supplies its own platform.Surface via the EGL
    // native window handle, so the EGL display never opens a Wayland connection.
    d.backend = try prism.platform.headless.create(gpa);
    errdefer if (d.backend) |b| {
        b.deinit();
        d.backend = null;
    };
    // Bring up the HAL device that backs all contexts + surfaces (EGL M2 render
    // path). Auto-detect by default: the best available driver wins (real GPU first,
    // the software rasterizer last), so a real app on the RTX renders on the GPU with
    // no configuration, mirroring how the Vulkan ICD enumerates available drivers. The
    // deterministic oracles pin "software" via `pinned_driver` to stay GPU-independent.
    if (std.c.getenv("PRISM_FORCE_SW") != null) pinned_driver = "software"; // DIAG: force CPU driver
    if (std.c.getenv("PRISM_NOQUAD") != null) prism.drivers.software.pipeline.disable_quad_diag = true; // DIAG: scalar FS only
    if (pinned_driver) |name| {
        const drv = prism.drivers.select(name) orelse return error.InitializationFailed;
        d.device = try drv.createDevice(gpa);
    } else {
        const sel = prism.drivers.createBestDevice(gpa) orelse return error.InitializationFailed;
        d.device = sel.device;
    }
    d.initialized = true;
}

pub fn terminate(d: *Display) void {
    if (!d.initialized) return;
    if (d.device) |dev| {
        dev.deinit();
        d.device = null;
    }
    if (d.backend) |b| {
        b.deinit();
        d.backend = null;
    }
    d.initialized = false;
}

// --- Query strings ----------------------------------------------------------

pub const version_string = "1.5";
pub const vendor_string = "Prism";
pub const client_apis_string = "OpenGL_ES OpenGL";
/// Honest extension set. EGL_KHR_surfaceless_context costs nothing for
/// enumeration. The platform extensions match getPlatformDisplay support
/// (wayland, gbm, surfaceless).
pub const extensions_string = "EGL_KHR_platform_wayland EGL_EXT_platform_wayland EGL_KHR_platform_gbm EGL_MESA_platform_gbm EGL_MESA_platform_surfaceless EGL_KHR_surfaceless_context";
/// Returned to libEGL via getVendorString(__EGL_VENDOR_STRING_PLATFORM_EXTENSIONS):
/// the platform extensions libEGL itself advertises on our behalf.
pub const platform_extensions_string = "EGL_KHR_platform_wayland EGL_EXT_platform_wayland EGL_KHR_platform_gbm EGL_MESA_platform_gbm EGL_MESA_platform_surfaceless";

pub fn queryString(name: EGLint) ?[*:0]const u8 {
    return switch (name) {
        egl.EGL_VERSION => version_string,
        egl.EGL_VENDOR => vendor_string,
        egl.EGL_CLIENT_APIS => client_apis_string,
        egl.EGL_EXTENSIONS => extensions_string,
        else => null,
    };
}

// --- Client API binding -----------------------------------------------------

pub fn supportsAPI(api: egl.EGLenum) bool {
    return api == egl.EGL_OPENGL_ES_API or api == egl.EGL_OPENGL_API;
}

// --- eglChooseConfig filtering ---------------------------------------------

/// Does a config satisfy one (attribute, value) constraint from the attrib list?
/// EGL_DONT_CARE always matches. Size attributes are "at least" matches. Bitmask
/// attributes (SURFACE_TYPE/RENDERABLE_TYPE/CONFORMANT) are "all requested bits
/// present". Exact-match attributes compare equal.
fn satisfies(c: Config, attribute: EGLint, value: EGLint) bool {
    if (value == egl.EGL_DONT_CARE) return true;
    return switch (attribute) {
        // "At least" size constraints.
        egl.EGL_RED_SIZE => c.red >= value,
        egl.EGL_GREEN_SIZE => c.green >= value,
        egl.EGL_BLUE_SIZE => c.blue >= value,
        egl.EGL_ALPHA_SIZE => c.alpha >= value,
        egl.EGL_DEPTH_SIZE => c.depth >= value,
        egl.EGL_STENCIL_SIZE => c.stencil >= value,
        egl.EGL_BUFFER_SIZE => c.bufferSize() >= value,
        egl.EGL_SAMPLES => c.samples >= value,
        egl.EGL_SAMPLE_BUFFERS => (if (c.samples > 0) @as(EGLint, 1) else 0) >= value,
        // "All requested bits present" bitmask constraints.
        egl.EGL_SURFACE_TYPE => blk: {
            const have = egl.EGL_WINDOW_BIT | egl.EGL_PBUFFER_BIT;
            break :blk (have & value) == value;
        },
        egl.EGL_RENDERABLE_TYPE, egl.EGL_CONFORMANT => blk: {
            const have = egl.EGL_OPENGL_ES2_BIT | egl.EGL_OPENGL_ES3_BIT | egl.EGL_OPENGL_BIT;
            break :blk (have & value) == value;
        },
        // Exact-match constraints we model.
        egl.EGL_CONFIG_ID => c.id == value,
        egl.EGL_COLOR_BUFFER_TYPE => value == egl.EGL_RGB_BUFFER,
        egl.EGL_CONFIG_CAVEAT => value == egl.EGL_NONE,
        // Anything else with a non-DONT_CARE value: accept only its default (0
        // for the zero-valued attributes we expose); CONFIG_ID handled above.
        else => true,
    };
}

/// Result of validating + applying an attrib list to the config set.
pub const ChooseError = error{BadAttribute};

/// Apply an eglChooseConfig attrib list, writing up to `out.len` matching config
/// indices into `out` and returning the total match count. A null/empty list
/// matches all configs. The attrib list is `attr, value, attr, value, ..., EGL_NONE`.
pub fn chooseConfig(attrib_list: ?[*]const EGLint, out: ?[]usize) ChooseError!usize {
    // Parse the constraints, validating each attribute key.
    var constraints: [64]struct { a: EGLint, v: EGLint } = undefined;
    var n_con: usize = 0;
    if (attrib_list) |list| {
        var i: usize = 0;
        while (list[i] != egl.EGL_NONE) : (i += 2) {
            const a = list[i];
            const v = list[i + 1];
            if (!isValidChooseAttribute(a)) return error.BadAttribute;
            if (n_con < constraints.len) {
                constraints[n_con] = .{ .a = a, .v = v };
                n_con += 1;
            }
        }
    }

    var count: usize = 0;
    for (configs, 0..) |c, idx| {
        var ok = true;
        for (constraints[0..n_con]) |con| {
            if (!satisfies(c, con.a, con.v)) {
                ok = false;
                break;
            }
        }
        if (!ok) continue;
        if (out) |o| {
            if (count < o.len) o[count] = idx;
        }
        count += 1;
    }
    return count;
}

/// Is `attribute` a key eglChooseConfig accepts? (Reject unknown keys with
/// EGL_BAD_ATTRIBUTE, per the spec.)
fn isValidChooseAttribute(attribute: EGLint) bool {
    return switch (attribute) {
        egl.EGL_RED_SIZE,
        egl.EGL_GREEN_SIZE,
        egl.EGL_BLUE_SIZE,
        egl.EGL_ALPHA_SIZE,
        egl.EGL_DEPTH_SIZE,
        egl.EGL_STENCIL_SIZE,
        egl.EGL_BUFFER_SIZE,
        egl.EGL_LUMINANCE_SIZE,
        egl.EGL_ALPHA_MASK_SIZE,
        egl.EGL_SAMPLES,
        egl.EGL_SAMPLE_BUFFERS,
        egl.EGL_SURFACE_TYPE,
        egl.EGL_RENDERABLE_TYPE,
        egl.EGL_CONFORMANT,
        egl.EGL_CONFIG_ID,
        egl.EGL_COLOR_BUFFER_TYPE,
        egl.EGL_CONFIG_CAVEAT,
        egl.EGL_NATIVE_RENDERABLE,
        egl.EGL_NATIVE_VISUAL_TYPE,
        egl.EGL_TRANSPARENT_TYPE,
        egl.EGL_TRANSPARENT_RED_VALUE,
        egl.EGL_TRANSPARENT_GREEN_VALUE,
        egl.EGL_TRANSPARENT_BLUE_VALUE,
        egl.EGL_LEVEL,
        egl.EGL_MAX_PBUFFER_WIDTH,
        egl.EGL_MAX_PBUFFER_HEIGHT,
        egl.EGL_MAX_PBUFFER_PIXELS,
        egl.EGL_MIN_SWAP_INTERVAL,
        egl.EGL_MAX_SWAP_INTERVAL,
        egl.EGL_BIND_TO_TEXTURE_RGB,
        egl.EGL_BIND_TO_TEXTURE_RGBA,
        egl.EGL_MATCH_NATIVE_PIXMAP,
        => true,
        else => false,
    };
}

// Tests

test "config attrib queries are consistent" {
    const c = configs[2]; // RGBA8 + D24 + S8
    try std.testing.expectEqual(@as(EGLint, 3), c.attrib(egl.EGL_CONFIG_ID).?);
    try std.testing.expectEqual(@as(EGLint, 8), c.attrib(egl.EGL_RED_SIZE).?);
    try std.testing.expectEqual(@as(EGLint, 24), c.attrib(egl.EGL_DEPTH_SIZE).?);
    try std.testing.expectEqual(@as(EGLint, 8), c.attrib(egl.EGL_STENCIL_SIZE).?);
    try std.testing.expectEqual(@as(EGLint, 32), c.attrib(egl.EGL_BUFFER_SIZE).?);
    try std.testing.expectEqual(egl.EGL_RGB_BUFFER, c.attrib(egl.EGL_COLOR_BUFFER_TYPE).?);
    // RENDERABLE_TYPE advertises both GL and GLES2/3.
    const rt = c.attrib(egl.EGL_RENDERABLE_TYPE).?;
    try std.testing.expect((rt & egl.EGL_OPENGL_BIT) != 0);
    try std.testing.expect((rt & egl.EGL_OPENGL_ES2_BIT) != 0);
    // Unknown attribute -> null (mapped to EGL_BAD_ATTRIBUTE at the boundary).
    try std.testing.expectEqual(@as(?EGLint, null), c.attrib(0x9999));
}

test "config handle round-trips" {
    for (0..configs.len) |i| {
        const h = configHandle(i);
        try std.testing.expectEqual(@as(?usize, i), configIndex(h));
    }
    try std.testing.expectEqual(@as(?usize, null), configIndex(@ptrFromInt(0xdead)));
    try std.testing.expectEqual(@as(?usize, null), configIndex(null));
}

test "chooseConfig: null list returns every config" {
    var buf: [16]usize = undefined;
    const n = try chooseConfig(null, &buf);
    try std.testing.expectEqual(configs.len, n);
}

test "chooseConfig: depth>=24 filters out the no-depth configs" {
    const list = [_]EGLint{ egl.EGL_DEPTH_SIZE, 24, egl.EGL_NONE };
    var buf: [16]usize = undefined;
    const n = try chooseConfig(&list, &buf);
    // configs 2, 3, and the two MSAA configs (5, 6) have depth 24.
    try std.testing.expectEqual(@as(usize, 4), n);
    for (buf[0..n]) |idx| {
        try std.testing.expect(configs[idx].depth >= 24);
    }
}

test "chooseConfig: stencil>=8 selects only the D24S8 config" {
    const list = [_]EGLint{ egl.EGL_STENCIL_SIZE, 8, egl.EGL_NONE };
    var buf: [16]usize = undefined;
    const n = try chooseConfig(&list, &buf);
    try std.testing.expectEqual(@as(usize, 1), n);
    try std.testing.expectEqual(@as(EGLint, 3), configs[buf[0]].id);
}

test "chooseConfig: alpha>=8 drops the alpha-less config" {
    const list = [_]EGLint{ egl.EGL_ALPHA_SIZE, 8, egl.EGL_NONE };
    var buf: [16]usize = undefined;
    const n = try chooseConfig(&list, &buf);
    try std.testing.expectEqual(@as(usize, 5), n); // ids 1,2,3,5,6 have alpha 8
}

test "chooseConfig: DONT_CARE matches everything" {
    const list = [_]EGLint{ egl.EGL_ALPHA_SIZE, egl.EGL_DONT_CARE, egl.EGL_NONE };
    var buf: [16]usize = undefined;
    try std.testing.expectEqual(configs.len, try chooseConfig(&list, &buf));
}

test "chooseConfig: RENDERABLE_TYPE GL bit is satisfiable" {
    const list = [_]EGLint{ egl.EGL_RENDERABLE_TYPE, egl.EGL_OPENGL_BIT, egl.EGL_NONE };
    var buf: [16]usize = undefined;
    try std.testing.expectEqual(configs.len, try chooseConfig(&list, &buf));
}

test "chooseConfig: unknown attribute key is rejected" {
    const list = [_]EGLint{ 0x1234, 1, egl.EGL_NONE };
    try std.testing.expectError(error.BadAttribute, chooseConfig(&list, null));
}

test "chooseConfig: count without an output buffer" {
    const list = [_]EGLint{ egl.EGL_DEPTH_SIZE, 24, egl.EGL_NONE };
    try std.testing.expectEqual(@as(usize, 4), try chooseConfig(&list, null));
}

test "supportsPlatform accepts surfaceless, wayland, gbm, and default" {
    try std.testing.expect(supportsPlatform(egl.EGL_PLATFORM_SURFACELESS_MESA));
    try std.testing.expect(supportsPlatform(egl.EGL_PLATFORM_WAYLAND_KHR));
    try std.testing.expect(supportsPlatform(egl.EGL_PLATFORM_GBM_KHR));
    try std.testing.expect(supportsPlatform(egl.EGL_PLATFORM_NONE));
    // An unsupported platform (X11) stays another vendor's job.
    try std.testing.expect(!supportsPlatform(egl.EGL_PLATFORM_X11_KHR));
}

test "supportsAPI: GL and GLES only" {
    try std.testing.expect(supportsAPI(egl.EGL_OPENGL_ES_API));
    try std.testing.expect(supportsAPI(egl.EGL_OPENGL_API));
    try std.testing.expect(!supportsAPI(egl.EGL_OPENVG_API));
}

test "queryString reports Prism EGL 1.5" {
    try std.testing.expectEqualStrings("1.5", std.mem.span(queryString(egl.EGL_VERSION).?));
    try std.testing.expectEqualStrings("Prism", std.mem.span(queryString(egl.EGL_VENDOR).?));
    try std.testing.expect(std.mem.indexOf(u8, std.mem.span(queryString(egl.EGL_CLIENT_APIS).?), "OpenGL_ES") != null);
    try std.testing.expectEqual(@as(?[*:0]const u8, null), queryString(0x1234));
}

test "getPlatformDisplay is stable per (platform, native) and initialize reports 1.5" {
    const d1 = getPlatformDisplay(egl.EGL_PLATFORM_SURFACELESS_MESA, null).?;
    const d2 = getPlatformDisplay(egl.EGL_PLATFORM_SURFACELESS_MESA, null).?;
    try std.testing.expectEqual(d1, d2); // same handle
    try std.testing.expect(lookupDisplay(@ptrCast(d1)) != null);
    try initialize(d1);
    try std.testing.expect(d1.initialized);
    terminate(d1);
    try std.testing.expect(!d1.initialized);
    // An unsupported platform yields no display and no error (another vendor's job).
    try std.testing.expectEqual(@as(?*Display, null), getPlatformDisplay(egl.EGL_PLATFORM_X11_KHR, null));
}

test "gbm: getPlatformDisplay + createGbmWindowSurface build a window surface aliasing the gbm buffer" {
    var mb = prism.platform.gbm.MemoryBackend.init(gpa);
    defer mb.deinit();
    var dev = prism.platform.gbm.Device.init(mb.allocator());
    var surf = prism.platform.gbm.Surface.init(&dev, .{
        .width = 16,
        .height = 8,
        .format = prism.platform.gbm.format.DRM_FORMAT_XRGB8888,
        .usage = .{ .scanout = true },
    });
    defer surf.deinit();

    const d = getPlatformDisplay(egl.EGL_PLATFORM_GBM_KHR, @ptrCast(&dev)).?;
    try std.testing.expect(d.gbmDevice().? == &dev);
    try initialize(d);
    defer terminate(d);

    const s = try createGbmWindowSurface(d, 0, &surf);
    defer s.deinit();
    try std.testing.expectEqual(SurfaceKind.window, s.kind);
    try std.testing.expect(s.owns_plat_surface);
    try std.testing.expectEqual(@as(u32, 16), s.width);
    try std.testing.expectEqual(@as(u32, 8), s.height);

    // The EGL surface's platform buffer aliases the gbm back buffer. After a
    // commit (eglSwapBuffers posts it) the app locks the same bytes for scanout.
    const buf = try s.plat_surface.?.currentBuffer();
    @memset(buf.bytes, 0x5a);
    try s.plat_surface.?.commit();
    const front = surf.lockFrontBuffer().?;
    try std.testing.expectEqual(@as(u8, 0x5a), front.data[0]);
}
