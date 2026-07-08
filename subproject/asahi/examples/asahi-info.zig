//! asahi-info - the single dual-target AGX probe (parity with nvidia-info).
//!
//! ONE example, comptime target-gated. Zig only analyzes the referenced branch
//! (the same lazy-analysis trick the comptime driver registry uses), so the UEFI
//! branch's std.os.uefi usage does not break the Linux build and vice versa.
//!
//!   * On Linux: a normal aarch64-linux executable the user rsyncs to the M1 Pro
//!     (Asahi Linux). It opens the kernel `asahi` DRM render node (no root),
//!     queries GET_PARAMS, prints the GPU identity, creates a VM, allocs + maps +
//!     VM-binds a small GEM BO, does a CPU write/readback roundtrip, prints
//!     RESULT: OK. This is the verified milestone-1 path, UNCHANGED.
//!
//!   * On UEFI (the M1's U-Boot EFI layer): Apple Silicon has NO PCI and NO
//!     EFI_PCI_IO - the GPU is an on-SoC device described by the DEVICETREE (FDT),
//!     not PCI. So the UEFI path walks the EFI configuration table for the DTB
//!     table GUID, parses the FDT, and reports the SoC + AGX GPU identity, then
//!     halts so the output can be read off a real machine's console. The live
//!     FDT-from-EFI read is the user's M1 test (run BOOTAA64.efi via U-Boot EFI).
//!
//! Output style is shared: greppable "key: value" lines and a final
//! RESULT: OK / RESULT: SKIP / RESULT: FAIL.

const std = @import("std");
const builtin = @import("builtin");
const asahi = @import("asahi");
// conduit re-exports dtree as `conduit.dtree`, and ships the device-tree
// discovery backend (`conduit.backend.dtree.DtBackend`) + the normalized
// resource/MMIO model the UEFI/baremetal AGX probe reuses. Only the UEFI branch
// references it; Zig's lazy analysis keeps it out of the Linux build entirely.
const conduit = @import("conduit");
// The shared UEFI console subproject. On UEFI it gives the con_out std.Io.Writer
// (UTF-8 -> UTF-16, '\n' -> "\r\n") + the std.log / panic opt-in below; on Linux
// it is imported harmlessly (its std.os.uefi usage is comptime-gated) and unused.
const uefi_support = @import("uefi");

const is_uefi = builtin.os.tag == .uefi;

// Opt into UEFI console support from the ROOT, but ONLY in the uefi build branch
// (on Linux std.log + panic already work natively and we must not override them).
// These re-exports route std.log.* and panics to the EFI console; the greppable
// data lines below stay on a PLAIN writer (out()) so existing greps are intact.
// NOTE (honest, Zig 0.16): std.debug.print is NOT routable to con_out from the
// root - it goes to std.fs.File.stderr(), which has no UEFI fd. std.log IS the
// supported routable path. See subproject/uefi/src/uefi.zig.
pub const std_options: std.Options = if (is_uefi) uefi_support.std_options else .{};
pub const panic = if (is_uefi) uefi_support.panic else std.debug.FullPanic(std.debug.defaultPanic);

// Comptime target-gated entry. Zig analyzes only the referenced branch, so the
// UEFI std.os.uefi usage and the Linux std.os.linux usage never clash.
pub const main = if (is_uefi) uefiMain else linuxMain;

// The data sink for the greppable "key: value" / RESULT lines is the single
// per-target writer from the shared uefi subproject: `uefi_support.init()`
// returns con_out on UEFI and a buffered stdout writer on Linux. std.log + panic
// (routed to con_out on UEFI via the opt-in above) are reserved for diagnostics,
// NOT the data.

// ===========================================================================
// Linux path: the kernel asahi DRM probe (the existing, M1-verified milestone 1).
// ===========================================================================

fn linuxMain(init: std.process.Init.Minimal) void {
    // The single per-target writer (buffered stdout on Linux) for the greppable
    // data lines. flush on return so the buffered lines are not lost. std.log /
    // std.debug.print also work natively on Linux.
    const lw = uefi_support.init();
    defer uefi_support.flush();

    // Allow forcing a node path: `asahi-info /dev/dri/renderD128`.
    // Also support `--dump-render` (decode + print OUR triangle + clear render
    // streams with no GPU, for a field-by-field diff against a real M1 Mesa
    // capture - see asahi.dumpStream). It is checked first and works anywhere.
    var forced: ?[]const u8 = null;
    var dump_render = false;
    var it = init.args.iterate();
    _ = it.next(); // argv[0]
    while (it.next()) |arg| {
        if (std.mem.eql(u8, arg, "--dump-render")) {
            dump_render = true;
        } else {
            forced = arg;
        }
    }
    if (dump_render) {
        lw.print("=== asahi-info --dump-render: OUR triangle + clear streams ===\n", .{}) catch {};
        asahi.render.dumpStream(lw, true); // triangle stream
        asahi.render.dumpStream(lw, false); // clear-only stream
        uefi_support.flush();
        return;
    }

    var dev = blk: {
        if (forced) |p| {
            break :blk asahi.Device.openNode(p) catch {
                lw.print("RESULT: SKIP could-not-open {s}\n", .{p}) catch {};
                return;
            };
        }
        break :blk asahi.Device.open() catch {
            // Dump the render nodes that DO exist + their kernel drivers, so a
            // driver-name mismatch (or a missing node) is visible rather than a
            // bare SKIP. Force a node directly with: asahi-info /dev/dri/renderD128
            var n: u32 = 128;
            while (n < 140) : (n += 1) {
                var nb: [32]u8 = undefined;
                if (asahi.Device.renderNodeDriver(n, &nb)) |drv| {
                    lw.print("  renderD{d}: driver={s}\n", .{ n, drv }) catch {};
                }
            }
            lw.print("RESULT: SKIP no-asahi-device (no renderD* with driver 'asahi')\n", .{}) catch {};
            return;
        };
    };
    defer dev.deinit();
    lw.print("device: opened asahi render node\n", .{}) catch {};

    const info = dev.getParams() catch {
        lw.print("RESULT: FAIL get-params ioctl failed (check the running asahi UABI matches uapi.zig)\n", .{}) catch {};
        return;
    };

    lw.print("gpu_name: {s}\n", .{info.name}) catch {};
    lw.print("gpu_generation: {d}\n", .{info.gpu_generation}) catch {};
    lw.print("gpu_variant: {c} (0x{x})\n", .{ info.gpu_variant, info.gpu_variant }) catch {};
    lw.print("gpu_revision: 0x{x}\n", .{info.gpu_revision}) catch {};
    lw.print("chip_id: 0x{x}\n", .{info.chip_id}) catch {};
    lw.print("num_dies: {d}\n", .{info.num_dies}) catch {};
    lw.print("num_clusters_total: {d}\n", .{info.num_clusters_total}) catch {};
    lw.print("num_cores_per_cluster: {d}\n", .{info.num_cores_per_cluster}) catch {};
    lw.print("total_core_count: {d}\n", .{info.total_core_count}) catch {};
    lw.print("max_frequency_mhz: {d}\n", .{info.max_frequency_khz / 1000}) catch {};
    lw.print("features: 0x{x}\n", .{info.features}) catch {};
    lw.print("vm_start: 0x{x}\n", .{info.vm_start}) catch {};
    lw.print("vm_end: 0x{x}\n", .{info.vm_end}) catch {};
    lw.print("vm_kernel_min_size: 0x{x}\n", .{info.vm_kernel_min_size}) catch {};

    // Create a VM.
    const vm_id = dev.vmCreate(info) catch {
        lw.print("RESULT: FAIL vm-create\n", .{}) catch {};
        return;
    };
    lw.print("vm_id: {d}\n", .{vm_id}) catch {};
    defer dev.vmDestroy(vm_id);

    // Allocate + map + bind a small BO low in the usable VA window.
    const size: u64 = 16 * 1024;
    const gpu_va: u64 = info.vm_start + 0x10000;
    const bo = dev.allocBo(vm_id, size, gpu_va) catch {
        lw.print("RESULT: FAIL gem-alloc/map/bind\n", .{}) catch {};
        return;
    };
    lw.print("bo_handle: {d}\n", .{bo.handle}) catch {};
    lw.print("bo_gpu_va: 0x{x}\n", .{bo.gpu_va}) catch {};
    lw.print("bo_size: {d}\n", .{bo.size}) catch {};

    // CPU write + read-back roundtrip over the whole mapping.
    var ok = true;
    var i: usize = 0;
    while (i < bo.cpu.len) : (i += 1) {
        bo.cpu[i] = @truncate(i *% 31 +% 7);
    }
    i = 0;
    while (i < bo.cpu.len) : (i += 1) {
        const expect: u8 = @truncate(i *% 31 +% 7);
        if (bo.cpu[i] != expect) {
            ok = false;
            break;
        }
    }

    if (ok) {
        lw.print("roundtrip: {d} bytes write+readback OK\n", .{bo.cpu.len}) catch {};
    } else {
        lw.print("roundtrip: MISMATCH at byte {d}\n", .{i}) catch {};
        lw.print("RESULT: FAIL roundtrip\n", .{}) catch {};
        return;
    }

    // -----------------------------------------------------------------------
    // P2: command queue + syncobj roundtrip + SUBMIT wiring.
    // Each step degrades gracefully (SKIP/FAIL line) so a partial M1 result
    // still prints what DID work above. The final RESULT line is emitted at
    // the end regardless.
    // -----------------------------------------------------------------------

    // Create a submission queue on this VM. P3 dispatches a real compute shader,
    // so the queue's usc_exec_base MUST be the USC-heap base the compute module
    // resolves its USC-relative shader/pipeline addresses against.
    const queue_id = dev.queueCreate(vm_id, asahi.uapi.PRIORITY_MEDIUM, asahi.compute.USC_EXEC_BASE) catch {
        lw.print("queue: SKIP queue-create failed (QUEUE_CREATE ioctl)\n", .{}) catch {};
        lw.print("RESULT: FAIL queue-create\n", .{}) catch {};
        return;
    };
    lw.print("queue_id: {d}\n", .{queue_id}) catch {};
    defer dev.queueDestroy(queue_id);

    // Syncobj roundtrip: create -> signal -> wait (should return immediately
    // because we just signalled it). This proves the core DRM syncobj path the
    // SUBMIT fences ride on.
    var sync_ok = false;
    if (dev.syncobjCreate(false)) |so| {
        lw.print("syncobj_handle: {d}\n", .{so}) catch {};
        if (dev.syncobjSignal(so)) |_| {
            const handles = [_]u32{so};
            // 1ms timeout; the obj is already signalled so this returns at once.
            if (dev.syncobjWait(&handles, 1_000_000)) |_| {
                lw.print("syncobj: create+signal+wait OK\n", .{}) catch {};
                sync_ok = true;
            } else |werr| {
                lw.print("syncobj: SKIP wait failed ({s})\n", .{@errorName(werr)}) catch {};
            }
        } else |serr| {
            lw.print("syncobj: SKIP signal failed ({s})\n", .{@errorName(serr)}) catch {};
        }
        dev.syncobjDestroy(so);
    } else |cerr| {
        lw.print("syncobj: SKIP create failed ({s})\n", .{@errorName(cerr)}) catch {};
    }

    // -----------------------------------------------------------------------
    // P3: the from-scratch AGX COMPUTE launch. Build a trivial kernel that
    // stores a known constant (kComputeConstant) to a GPU buffer, submit it,
    // wait the fence, and read the buffer back. If the read-back word matches
    // the constant, the AGX cores executed our hand-assembled shader.
    //
    // This is the user's M1 test: it performs a REAL GPU submit, so any failure
    // (alloc / submit / fence timeout / mismatch) must be reported gracefully -
    // the RESULT line below distinguishes OK (GPU ran) from FAIL (it did not),
    // and a thrown error degrades to a SKIP/FAIL line rather than a crash.
    // -----------------------------------------------------------------------
    lw.print("compute_expected: 0x{X:0>8}\n", .{asahi.compute.kConstant}) catch {};

    var compute_ok = false;
    if (asahi.runComputeConstant(&dev, vm_id, queue_id)) |got| {
        lw.print("compute_readback: 0x{X:0>8}\n", .{got}) catch {};
        compute_ok = (got == asahi.compute.kConstant);
        if (compute_ok) {
            lw.print("compute: OK readback == kConstant (the AGX executed our kernel)\n", .{}) catch {};
        } else if (got == asahi.compute.kSentinel) {
            // The sentinel SURVIVED a successful submit (fence signalled, no
            // fault). AGX is cache-coherent, so this is NOT a stale read: the
            // store never landed -> the shader/USC/dispatch ENCODING is the bug.
            lw.print("compute: MISMATCH readback == kSentinel 0x{X:0>8} (store did NOT land - ENCODING bug, coherency ruled out)\n", .{asahi.compute.kSentinel}) catch {};
        } else {
            lw.print("compute: MISMATCH readback is neither kConstant nor kSentinel (store hit but wrote a wrong value - note it)\n", .{}) catch {};
        }
    } else |cerr| {
        // Diagnose WHICH ioctl was rejected + the errno. runComputeConstant runs
        // a chain of allocBo (GEM_CREATE -> GEM_MMAP_OFFSET -> VM_BIND), then
        // SUBMIT, then SYNCOBJ_WAIT - the bare error name hides which one failed.
        // device.zig records last_request/last_errno on the failing ioctl, so an
        // EINVAL (22) on SUBMIT vs on VM_BIND points at totally different bugs.
        lw.print("compute: SKIP/FAIL runComputeConstant ({s})\n", .{@errorName(cerr)}) catch {};
        if (cerr == error.IoctlFailed) {
            lw.print(
                "compute: FAIL ioctl={s} (0x{x}) errno={d}\n",
                .{ asahi.device.requestName(dev.last_request), dev.last_request, dev.last_errno },
            ) catch {};
        } else if (cerr == error.Timeout) {
            // SYNCOBJ_WAIT special-cases ETIME to Error.Timeout (no errno set):
            // the submit was accepted but the GPU never signalled the fence.
            lw.print("compute: FAIL ioctl=SYNCOBJ_WAIT (fence timeout - submit accepted, GPU did not signal)\n", .{}) catch {};
        }
    }

    // -----------------------------------------------------------------------
    // P4: the from-scratch AGX RENDER launch - the first TRIANGLE on Apple
    // Silicon via Prism. Clear a 64x64 linear RGBA8 buffer + rasterize one
    // solid-color triangle into it, submit, wait the fence, read a center pixel.
    //
    // This is the user's M1 test (a REAL GPU submit). The AGX render encoding is
    // large + partly under-documented, so this is a Mesa-grounded FIRST ATTEMPT
    // and WILL likely need iteration: a wrong cmdbuf GPU-faults (asahi dmesg "gpu
    // fault address 0x... reason:Unmapped"), exactly the signal the compute
    // bring-up used. The readback is decoded four ways below so a partial result
    // (clear ran but triangle missed, vs nothing stored) is distinguishable. Any
    // thrown error degrades to a SKIP/FAIL line rather than a crash.
    // -----------------------------------------------------------------------
    // CLEAR-ONLY DISCRIMINATOR first: the SAME render pass minus the geometry
    // draw (a Barrier+Terminate VDM stream). It runs the per-tile background clear
    // + end-of-tile store ONLY, isolating the tile-renderer INFRASTRUCTURE (bg
    // clear + eot store + tile/render config) from the DRAW (VDM/VS/FS). Decode:
    //   clear_readback == kClearColor -> infrastructure WORKS; any triangle
    //       hang/miss is purely in the DRAW path (next target VDM/VS/FS).
    //   clear_readback == kSentinel / a fence timeout -> the bg/eot/PBE/tile
    //       config hangs (next target the clear-only infrastructure).
    // A timeout (not a fault) means a non-terminating stream/shader: the
    // clear-only stream is a minimal Barrier+Terminate, so a timeout HERE points
    // squarely at the bg/eot shaders or the drm_asahi_cmd_render tile config.
    lw.print("clear_expected: 0x{X:0>8}\n", .{asahi.render.kClearColor}) catch {};
    if (asahi.render.runClear(&dev, vm_id, queue_id)) |cpx| {
        lw.print("clear_readback: 0x{X:0>8}\n", .{cpx}) catch {};
        if (cpx == asahi.render.kClearColor) {
            lw.print("clear: OK readback == kClearColor (the tile renderer WORKS: bg clear + eot store + config; any triangle hang is in the DRAW)\n", .{}) catch {};
        } else if (cpx == asahi.render.kSentinel) {
            lw.print("clear: MISMATCH readback == kSentinel 0x{X:0>8} (NOTHING stored - bg/eot/PBE/tile-config ENCODING bug, coherency ruled out)\n", .{asahi.render.kSentinel}) catch {};
        } else {
            lw.print("clear: MISMATCH readback is neither kClearColor nor kSentinel (partial/wrong store - note it)\n", .{}) catch {};
        }
    } else |cerr| {
        lw.print("clear: SKIP/FAIL runClear ({s})\n", .{@errorName(cerr)}) catch {};
        if (cerr == error.IoctlFailed) {
            lw.print(
                "clear: FAIL ioctl={s} (0x{x}) errno={d}\n",
                .{ asahi.device.requestName(dev.last_request), dev.last_request, dev.last_errno },
            ) catch {};
        } else if (cerr == error.Timeout) {
            lw.print("clear: FAIL ioctl=SYNCOBJ_WAIT (fence timeout - submit accepted, GPU did not signal; bg/eot/config HANG, check dmesg for fault-vs-timeout)\n", .{}) catch {};
        }
    }

    lw.print("triangle_expected: 0x{X:0>8}\n", .{asahi.render.kTriColor}) catch {};

    var triangle_ok = false;
    if (asahi.runTriangle(&dev, vm_id, queue_id)) |px| {
        lw.print("triangle_readback: 0x{X:0>8}\n", .{px}) catch {};
        triangle_ok = (px == asahi.render.kTriColor);
        if (triangle_ok) {
            lw.print("triangle: OK readback == kTriColor (the AGX rasterized our triangle to the center pixel)\n", .{}) catch {};
        } else if (px == asahi.render.kClearColor) {
            // The bg clear landed but the triangle did not cover the center
            // pixel -> a draw / VDM / PPP / viewport / vertex-position bug (NOT a
            // store-path bug: the eot store clearly ran to write the clear color).
            lw.print("triangle: PARTIAL readback == kClearColor 0x{X:0>8} (clear ran, triangle MISSED the center - draw/VDM/PPP/VS-position bug)\n", .{asahi.render.kClearColor}) catch {};
        } else if (px == asahi.render.kSentinel) {
            // The sentinel survived a successful submit (fence signalled, no
            // fault). AGX is cache-coherent, so this is NOT a stale read: nothing
            // was stored to the center pixel -> the bg clear AND the eot store
            // did not land -> an ENCODING bug (bg/eot/PBE/tile config).
            lw.print("triangle: MISMATCH readback == kSentinel 0x{X:0>8} (NOTHING stored - bg/eot/PBE ENCODING bug, coherency ruled out)\n", .{asahi.render.kSentinel}) catch {};
        } else {
            lw.print("triangle: MISMATCH readback is none of kTriColor/kClearColor/kSentinel (partial/wrong store - note it)\n", .{}) catch {};
        }
    } else |rerr| {
        // Same diagnostic chain as compute: runTriangle runs allocBo (GEM_CREATE
        // -> GEM_MMAP_OFFSET -> VM_BIND) x many BOs, then SUBMIT, then
        // SYNCOBJ_WAIT. device.zig records last_request/last_errno so an EINVAL on
        // SUBMIT vs VM_BIND points at different bugs.
        lw.print("triangle: SKIP/FAIL runTriangle ({s})\n", .{@errorName(rerr)}) catch {};
        if (rerr == error.IoctlFailed) {
            lw.print(
                "triangle: FAIL ioctl={s} (0x{x}) errno={d}\n",
                .{ asahi.device.requestName(dev.last_request), dev.last_request, dev.last_errno },
            ) catch {};
        } else if (rerr == error.Timeout) {
            lw.print("triangle: FAIL ioctl=SYNCOBJ_WAIT (fence timeout - submit accepted, GPU did not signal; likely a GPU fault, check dmesg)\n", .{}) catch {};
        }
    }

    // The final RESULT: OK iff the syncobj roundtrip, the compute readback, AND
    // the triangle readback all passed - i.e. the GPU executed our compute kernel
    // AND rasterized our triangle. The triangle is the new P4 milestone; the
    // earlier passes still print above so a partial M1 result is legible.
    if (sync_ok and compute_ok and triangle_ok) {
        lw.print("RESULT: OK\n", .{}) catch {};
    } else if (!sync_ok) {
        lw.print("RESULT: FAIL syncobj-roundtrip\n", .{}) catch {};
    } else if (!compute_ok) {
        lw.print("RESULT: FAIL compute-execution\n", .{}) catch {};
    } else {
        lw.print("RESULT: FAIL triangle-render\n", .{}) catch {};
    }
}

// ===========================================================================
// UEFI path: the baremetal FDT probe. Reads the DTB from the EFI configuration
// table (EFI_DTB_TABLE_GUID), parses it, reports the SoC + AGX GPU, then halts.
// ===========================================================================

const uefi = std.os.uefi;

// The device-tree reader + conduit's discovery backend used by the UEFI probe.
const Reader = conduit.dtree.Reader;
const DtBackend = conduit.backend.dtree.DtBackend;

// EFI_DTB_TABLE_GUID = b1b621d5-f19c-41a5-830b-d9152c69aae0. This is the GUID U-Boot
// (and other firmware) uses to publish the Flattened Device Tree in the EFI
// configuration table.
const EFI_DTB_TABLE_GUID = uefi.Guid{
    .time_low = 0xb1b621d5,
    .time_mid = 0xf19c,
    .time_high_and_version = 0x41a5,
    .clock_seq_high_and_reserved = 0x83,
    .clock_seq_low = 0x0b,
    .node = .{ 0xd9, 0x15, 0x2c, 0x69, 0xaa, 0xe0 },
};

// A pointer to the shared con_out writer, set up in uefiMain so every helper can
// format through it. The shared uefi subproject writer translates UTF-8 to UTF-16
// and '\n' to "\r\n" (so a "\n" terminator emits CRLF on con_out), exactly what
// the old per-subproject console.zig did.
var w: *std.Io.Writer = undefined;

fn uefiMain() uefi.Status {
    // Capture + reset con_out via the shared uefi subproject; init() returns the
    // con_out writer. After this, std.log.* and panics (opted in at the root
    // above) also reach con_out. pauseHalt() flushes before spinning.
    w = uefi_support.init();
    const st = uefi.system_table;

    w.print("\n", .{}) catch {};
    w.print("=== Prism UEFI AGX probe (FDT / devicetree) ===\n", .{}) catch {};
    w.print("Apple Silicon has no PCI; reading the SoC + GPU identity from the FDT.\n", .{}) catch {};
    w.print("\n", .{}) catch {};

    // Walk the EFI configuration table for the DTB table GUID.
    const blob = findDtb(st) orelse {
        w.print("fdt: no devicetree table in EFI config table\n", .{}) catch {};
        w.print("     (expected EFI_DTB_TABLE_GUID b1b621d5-f19c-41a5-830b-d9152c69aae0)\n", .{}) catch {};
        w.print("\n", .{}) catch {};
        w.print("RESULT: SKIP no-devicetree\n", .{}) catch {};
        pauseHalt();
    };

    w.print("fdt: found devicetree at 0x{X:0>16}  totalsize = 0x{X:0>8}\n", .{ @intFromPtr(blob.ptr), @as(u32, @intCast(blob.len)) }) catch {};

    // Parse the FDT with Midstall's dtree (reached via conduit.dtree, so there is
    // only one dtree instance in the graph). The Reader validates the magic +
    // bounds and never reads past the blob.
    const reader = Reader.initBuffer(blob) catch |e| {
        w.print("fdt: parse failed ({s})\n", .{@errorName(e)}) catch {};
        w.print("\n", .{}) catch {};
        w.print("RESULT: FAIL fdt-parse\n", .{}) catch {};
        pauseHalt();
    };

    // The root "compatible" is the SoC compatible list (e.g.
    // "apple,j314s\0apple,t6000\0apple,arm-platform\0"). Print it whole + name
    // the SoC from the first apple,tXXXX entry. Root properties live under the
    // root node, whose name is the empty string, so the dtree path is
    // {"", <prop>}.
    w.print("\n", .{}) catch {};
    w.print("--- SoC identity (root node) ---\n", .{}) catch {};
    if (reader.find(&.{ "", "compatible" })) |compat| {
        printCompatList("root compatible", compat);
        if (socName(compat)) |name| {
            w.print("soc_name: {s}\n", .{name}) catch {};
        } else {
            w.print("soc_name: (unrecognized apple,tXXXX - see root compatible above)\n", .{}) catch {};
        }
    } else |_| {
        w.print("root compatible: (absent)\n", .{}) catch {};
    }
    if (reader.find(&.{ "", "model" })) |model| {
        // model is a single NUL-terminated string.
        w.print("model: {s}\n", .{trimNul(model)}) catch {};
    } else |_| {}

    // Find the AGX GPU node + its MMIO register window via CONDUIT's device
    // discovery. The DtBackend walks the tree yielding one node per device, with
    // each node's `compatible` strings lowered into ids and its `reg` lowered
    // (cell-decoded + ranges-translated) into MMIO resources. We iterate the
    // backend directly and match the GPU by an "agx" substring over its ids,
    // because the AGX compatible is chip-specific ("apple,agx-t6000" /
    // "apple,agx-g13s..."), which a fixed-string conduit.Matcher cannot enumerate.
    // This is the same backend + resource API the baremetal driver will reuse to
    // mint conduit.Mmio.direct(reg_base) in a later phase.
    w.print("\n", .{}) catch {};
    w.print("--- AGX GPU node (via conduit discovery) ---\n", .{}) catch {};
    var be = DtBackend.init(&reader);
    const agx = findAgx(&be) orelse {
        // SELF-DIAGNOSING: no agx node found. Dump the immediate root children so
        // the real devicetree structure is visible (same spirit as the Linux
        // renderNodeDriver dump). Use dtree's nodeIterator at depth 1.
        w.print("agx: NOT FOUND - dumping the root's immediate child nodes so the\n", .{}) catch {};
        w.print("     real devicetree structure is visible:\n", .{}) catch {};
        dumpRootChildren(&reader);
        w.print("\n", .{}) catch {};
        w.print("RESULT: SKIP no-agx-node\n", .{}) catch {};
        pauseHalt();
    };

    w.print("agx_node: {s}\n", .{agx.name}) catch {};
    printIdList("agx compatible", agx.ids);

    // Report the MMIO register window(s) conduit lowered from the node's `reg`.
    if (agx.resources.mmio()) |m| {
        w.print("agx_reg_base: 0x{X:0>16}\n", .{m.base}) catch {};
        w.print("agx_reg_size: 0x{X:0>16}\n", .{m.size}) catch {};
        // Additional windows, if the GPU node carries more than one reg entry.
        var n: usize = 1;
        while (agx.resources.mmioAt(n)) |extra| : (n += 1) {
            w.print("agx_reg_base[{d}]: 0x{X:0>16}  size = 0x{X:0>16}\n", .{ n, extra.base, extra.size }) catch {};
        }
    } else {
        w.print("agx_reg: (no MMIO register window in the node's reg)\n", .{}) catch {};
    }

    w.print("\n", .{}) catch {};
    w.print("RESULT: OK\n", .{}) catch {};
    w.print("=== probe done ===\n", .{}) catch {};
    pauseHalt();
}

/// One discovered AGX device: its node name, its compatible ids, and the MMIO
/// resources conduit lowered from its `reg` property.
const AgxMatch = struct {
    name: []const u8,
    ids: conduit.IdList,
    resources: conduit.ResourceList,
};

/// Walk conduit's device-tree backend for the first node whose compatible ids
/// contain "agx", returning it with its lowered MMIO resources. null if none.
fn findAgx(be: *DtBackend) ?AgxMatch {
    while (be.next() catch return null) |node| {
        if (!idsContain(node.ids.slice(), "agx")) continue;
        var list = conduit.ResourceList{};
        be.resources(node, &list) catch {};
        return .{ .name = node.name, .ids = node.ids, .resources = list };
    }
    return null;
}

/// True if any id in the list contains `needle` as a substring.
fn idsContain(ids: []const conduit.match.Id, needle: []const u8) bool {
    for (ids) |id| {
        if (std.mem.indexOf(u8, id, needle) != null) return true;
    }
    return false;
}

/// Dump the root's immediate child node names (depth 1) using dtree's iterator.
fn dumpRootChildren(reader: *const Reader) void {
    var it = reader.nodeIterator();
    while (it.next() catch null) |node| {
        if (node == .begin and node.begin.depth == 1) {
            const name = node.begin.name;
            if (name.len == 0) {
                w.print("  child: (anonymous)\n", .{}) catch {};
            } else {
                w.print("  child: {s}\n", .{name}) catch {};
            }
        }
    }
}

/// Walk the EFI configuration table for EFI_DTB_TABLE_GUID and return the FDT blob
/// (its length is the totalsize read from the FDT header). null if not present.
fn findDtb(st: *uefi.tables.SystemTable) ?[]const u8 {
    const entries = st.configuration_table[0..st.number_of_table_entries];
    for (entries) |entry| {
        if (entry.vendor_guid.eql(EFI_DTB_TABLE_GUID)) {
            const ptr: [*]const u8 = @ptrCast(@alignCast(entry.vendor_table));
            // The FDT header carries its own totalsize (u32 BE at offset 4). Read
            // it directly (the blob length is not in the config table). Magic is
            // validated by Fdt.init, so a bad table is rejected there.
            const totalsize = std.mem.readInt(u32, ptr[4..8], .big);
            if (totalsize < 40 or totalsize > 16 * 1024 * 1024) {
                // Implausible header - still hand a header-sized slice so Fdt.init
                // reports BadMagic/Truncated rather than us trusting a wild size.
                return ptr[0..40];
            }
            return ptr[0..totalsize];
        }
    }
    return null;
}

/// Map the SoC compatible list to a human name, mirroring device.zig's chip table.
/// The list is NUL-separated; the apple,tXXXX entry is the SoC part number.
fn socName(compat: []const u8) ?[]const u8 {
    var it = std.mem.splitScalar(u8, compat, 0);
    while (it.next()) |entry| {
        if (entry.len == 0) continue;
        if (std.mem.eql(u8, entry, "apple,t8103")) return "Apple M1 (T8103)";
        if (std.mem.eql(u8, entry, "apple,t6000")) return "Apple M1 Pro (T6000)";
        if (std.mem.eql(u8, entry, "apple,t6001")) return "Apple M1 Max (T6001)";
        if (std.mem.eql(u8, entry, "apple,t6002")) return "Apple M1 Ultra (T6002)";
        if (std.mem.eql(u8, entry, "apple,t8112")) return "Apple M2 (T8112)";
        if (std.mem.eql(u8, entry, "apple,t6020")) return "Apple M2 Pro (T6020)";
        if (std.mem.eql(u8, entry, "apple,t6021")) return "Apple M2 Max (T6021)";
        if (std.mem.eql(u8, entry, "apple,t6022")) return "Apple M2 Ultra (T6022)";
        if (std.mem.eql(u8, entry, "apple,t8122")) return "Apple M3 (T8122)";
        if (std.mem.eql(u8, entry, "apple,t6030")) return "Apple M3 Pro (T6030)";
        if (std.mem.eql(u8, entry, "apple,t6031")) return "Apple M3 Max (T6031)";
        if (std.mem.eql(u8, entry, "apple,t8132")) return "Apple M4 (T8132)";
        if (std.mem.eql(u8, entry, "apple,t6040")) return "Apple M4 Pro (T6040)";
        if (std.mem.eql(u8, entry, "apple,t6041")) return "Apple M4 Max (T6041)";
    }
    return null;
}

fn trimNul(s: []const u8) []const u8 {
    var end = s.len;
    while (end > 0 and s[end - 1] == 0) end -= 1;
    return s[0..end];
}

/// Print a NUL-separated compatible string list, one entry per line.
fn printCompatList(label: []const u8, compat: []const u8) void {
    w.print("{s}:\n", .{label}) catch {};
    var it = std.mem.splitScalar(u8, compat, 0);
    while (it.next()) |entry| {
        if (entry.len == 0) continue;
        w.print("  - {s}\n", .{entry}) catch {};
    }
}

/// Print a conduit IdList (a node's compatible strings, already split), one per
/// line. The AGX node's ids come pre-split out of conduit's discovery.
fn printIdList(label: []const u8, ids: conduit.IdList) void {
    w.print("{s}:\n", .{label}) catch {};
    for (ids.slice()) |id| {
        if (id.len == 0) continue;
        w.print("  - {s}\n", .{id}) catch {};
    }
}

// Halt with the output left on screen, via the shared uefi subproject (it prints
// the same ">>> HALTED ... <<<" banner then spins forever). Without this the probe
// returns to the EFI firmware (U-Boot), which boots on before the identity can be
// read or photographed.
fn pauseHalt() noreturn {
    uefi_support.halt();
}
