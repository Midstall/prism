const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const version = b.option([]const u8, "version", "Prism version string") orelse "0.0.0-dev";

    // -Ddrivers selects which drivers are compiled in, passed to the code as
    // per-driver comptime flags in build_options. lib/prism/drivers.zig builds its
    // list from them. An unselected driver's source is never analyzed.
    const known_drivers = [_][]const u8{ "software", "apple", "nvidia", "virgl" };
    const drivers_opt = b.option([]const u8, "drivers", "Comma-separated drivers to compile in") orelse "software,apple,nvidia,virgl";
    var enabled = std.StringHashMap(void).init(b.allocator);
    {
        var it = std.mem.splitScalar(u8, drivers_opt, ',');
        while (it.next()) |raw| {
            const name = std.mem.trim(u8, raw, " ");
            if (name.len == 0) continue;
            var ok = false;
            for (known_drivers) |k| {
                if (std.mem.eql(u8, k, name)) ok = true;
            }
            if (!ok) std.debug.panic("unknown driver: {s}", .{name});
            enabled.put(name, {}) catch @panic("OOM");
        }
    }

    const options = b.addOptions();
    options.addOption([]const u8, "version", version);
    inline for (known_drivers) |name| {
        options.addOption(bool, "driver_" ++ name, enabled.contains(name));
    }

    const wl_dep = b.dependency("wayland", .{ .target = target, .optimize = optimize });
    const nvidia_dep = b.dependency("nvidia", .{ .target = target, .optimize = optimize });
    const asahi_dep = b.dependency("asahi", .{ .target = target, .optimize = optimize });
    const vulcan_dep = b.dependency("vulcan", .{ .target = target, .optimize = optimize });
    const drm_dep = b.dependency("drm", .{ .target = target, .optimize = optimize });
    const gbm_dep = b.dependency("gbm", .{ .target = target, .optimize = optimize });

    const prism = b.addModule("prism", .{
        .root_source_file = b.path("lib/prism.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "build-options", .module = options.createModule() },
            .{ .name = "wayland", .module = wl_dep.module("wayland") },
            .{ .name = "nvidia", .module = nvidia_dep.module("nvidia") },
            .{ .name = "asahi", .module = asahi_dep.module("asahi") },
            .{ .name = "drm", .module = drm_dep.module("drm") },
            .{ .name = "gbm", .module = gbm_dep.module("gbm") },
            .{ .name = "vulcan-ir", .module = vulcan_dep.module("vulcan-ir") },
            .{ .name = "vulcan-spirv", .module = vulcan_dep.module("vulcan-spirv") },
            .{ .name = "vulcan-target", .module = vulcan_dep.module("vulcan-target") },
            .{ .name = "vulcan-glsl", .module = vulcan_dep.module("vulcan-glsl") },
        },
    });

    const linkage = b.option(std.builtin.LinkMode, "linkage", "Core library linkage") orelse .dynamic;
    b.installArtifact(b.addLibrary(.{ .name = "prism", .root_module = prism, .linkage = linkage }));

    // One test runner: the prism module pulls in every internal test via its
    // file-path import graph (lib/prism.zig references the test-only files too).
    const test_step = b.step("test", "Run all Prism tests");
    const test_filter = b.option([]const u8, "test-filter", "Only run tests whose name contains this substring");
    const test_filters: []const []const u8 = if (test_filter) |f| &.{f} else &.{};
    test_step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = prism, .filters = test_filters })).step);

    // The GLX vendor, EGL/GLVND vendor, and Vulkan ICD shared libraries.
    if (!target.result.cpu.arch.isWasm() and target.result.os.tag != .uefi) {
        const Frontend = struct { name: []const u8, root: []const u8 };
        for ([_]Frontend{
            .{ .name = "GLX_prism", .root = "lib/prism-gl.zig" },
            .{ .name = "EGL_prism", .root = "lib/prism-egl.zig" },
            .{ .name = "prism-vk", .root = "lib/prism-vk.zig" },
        }) |fe| {
            const is_vk = std.mem.eql(u8, fe.name, "prism-vk");
            const is_egl = std.mem.eql(u8, fe.name, "EGL_prism");
            // The ICD's wsi_wayland.zig and the EGL vendor's wl_egl_window.zig both use
            // wayland.shm for the wl_shm present pool, so both need Prism's Zig wayland
            // library directly.
            const wl_imports = [_]std.Build.Module.Import{
                .{ .name = "prism", .module = prism },
                .{ .name = "wayland", .module = wl_dep.module("wayland") },
            };
            const base_imports = [_]std.Build.Module.Import{
                .{ .name = "prism", .module = prism },
            };
            const fmod = b.createModule(.{
                .root_source_file = b.path(fe.root),
                .target = target,
                .optimize = optimize,
                .imports = if (is_vk or is_egl) &wl_imports else &base_imports,
            });
            if (is_vk or is_egl) {
                // The libwayland present path links libwayland-client (the wl_* C API),
                // bound at runtime to the host app's already-loaded copy.
                fmod.link_libc = true;
                fmod.linkSystemLibrary("wayland-client", .{});
            }
            b.installArtifact(b.addLibrary(.{ .name = fe.name, .root_module = fmod, .linkage = .dynamic }));
            test_step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = fmod })).step);
        }
    }

    // prism-info CLI.
    const info_mod = b.createModule(.{
        .root_source_file = b.path("tools/prism-info.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{.{ .name = "prism", .module = prism }},
    });
    const info_exe = b.addExecutable(.{ .name = "prism-info", .root_module = info_mod });
    b.installArtifact(info_exe);
    b.step("run-info", "Run prism-info").dependOn(&b.addRunArtifact(info_exe).step);
    test_step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = info_mod })).step);

    // Triangle example: best available driver with a software fallback, presented
    // over Wayland.
    const triangle_mod = b.createModule(.{
        .root_source_file = b.path("examples/triangle.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{.{ .name = "prism", .module = prism }},
    });
    const triangle_exe = b.addExecutable(.{ .name = "triangle", .root_module = triangle_mod });
    b.installArtifact(triangle_exe);
    b.step("run-triangle", "Run the triangle on Wayland (needs a compositor)").dependOn(&b.addRunArtifact(triangle_exe).step);

    // Render the loader/vendor JSON templates, replacing @LIBDIR@ with the install lib dir.
    const libdir = b.getInstallPath(.lib, "");
    inline for (.{
        .{ "data/vulkan/icd.d/prism.json.in", "share/vulkan/icd.d/prism.json" },
        .{ "data/glvnd/egl_vendor.d/50_prism.json.in", "share/glvnd/egl_vendor.d/50_prism.json" },
    }) |pair| {
        const tmpl = b.build_root.handle.readFileAlloc(b.graph.io, pair[0], b.allocator, .limited(1 << 16)) catch @panic("read template");
        const rendered = std.mem.replaceOwned(u8, b.allocator, tmpl, "@LIBDIR@", libdir) catch @panic("OOM");
        const wf = b.addWriteFiles();
        const out = wf.add(std.fs.path.basename(pair[1]), rendered);
        b.getInstallStep().dependOn(&b.addInstallFile(out, pair[1]).step);
    }
}
