const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const root_module = b.addModule("asahi", .{
        .root_source_file = b.path("src/asahi.zig"),
        .target = target,
        .optimize = optimize,
    });

    // conduit (device discovery + MMIO + dtree) for the UEFI/baremetal AGX
    // probe. We reach dtree through `conduit.dtree`, so only one dtree instance
    // exists in the graph. conduit lazily resolves its own dtree (pinned in its
    // build.zig.zon) for the wanted device-tree backend (`-Ddtree=true`, the
    // default). Only the UEFI branch (uefiMain) references it; on the Linux
    // build Zig's lazy analysis never touches the conduit import.
    const conduit_dep = b.dependency("conduit", .{
        .target = target,
        .optimize = optimize,
    });
    const conduit_module = conduit_dep.module("conduit");

    // The shared EFI-console subproject (con_out writer + std.log/panic opt-in).
    const uefi_dep = b.dependency("uefi", .{
        .target = target,
        .optimize = optimize,
    });
    const uefi_module = uefi_dep.module("uefi");

    const test_step = b.step("test", "Run asahi subproject tests");
    test_step.dependOn(&b.addRunArtifact(b.addTest(.{
        .root_module = root_module,
    })).step);

    // The single dual-target asahi-info probe (parity with nvidia-info):
    //   * Linux  -> a normal aarch64-linux executable "asahi-info" in bin/, the
    //     user rsyncs it to the M1 Pro and runs it (the verified milestone-1 path).
    //   * UEFI   -> a UEFI application named BOOTAA64/BOOTX64 by arch, installed to
    //     EFI/BOOT/. Apple Silicon has no PCI; the UEFI path reads the FDT
    //     (devicetree) from the EFI config table. The M1 is aarch64, so
    //     `zig build asahi-info -Dtarget=aarch64-uefi` -> EFI/BOOT/BOOTAA64.efi.
    const is_uefi = target.result.os.tag == .uefi;

    const exe_name = if (is_uefi) switch (target.result.cpu.arch) {
        .x86_64 => "BOOTX64",
        .aarch64 => "BOOTAA64",
        else => "BOOTEFI",
    } else "asahi-info";

    const probe = b.addExecutable(.{
        .name = exe_name,
        .root_module = b.createModule(.{
            .root_source_file = b.path("examples/asahi-info.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "asahi", .module = root_module },
                .{ .name = "conduit", .module = conduit_module },
                .{ .name = "uefi", .module = uefi_module },
            },
        }),
    });

    // UEFI artifacts land in EFI/BOOT/ (the removable-media default boot path);
    // Linux artifacts land in bin/.
    const install = if (is_uefi)
        b.addInstallArtifact(probe, .{
            .dest_dir = .{ .override = .{ .custom = "EFI/BOOT" } },
        })
    else
        b.addInstallArtifact(probe, .{});

    b.getInstallStep().dependOn(&install.step);

    const probe_step = b.step("asahi-info", "Build the asahi-info probe (Linux bin / UEFI EFI/BOOT)");
    probe_step.dependOn(&install.step);

    // A convenience run step (Linux only - the UEFI build is run on the M1).
    if (!is_uefi) {
        const run_probe = b.addRunArtifact(probe);
        if (b.args) |args| run_probe.addArgs(args);
        const run_step = b.step("run-asahi-info", "Run the asahi-info probe (needs an AGX GPU)");
        run_step.dependOn(&run_probe.step);
    }
}
