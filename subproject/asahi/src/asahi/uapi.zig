//! drm/asahi UAPI bindings - the user-space ABI of the kernel's Apple AGX GPU
//! DRM driver (`asahi`), ported from include/uapi/drm/asahi_drm.h.
//!
//! VERSION PINNING: this mirrors the MAINLINED asahi UAPI (Linux kernel
//! ~6.14+, the version Alyssa Rosenzweig's series landed as - "agx-uapi v5",
//! March 2025). Source: torvalds/linux master include/uapi/drm/asahi_drm.h.
//! In this revision:
//!   - DRM_ASAHI_MAX_CLUSTERS = 64
//!   - the ioctl id ordering is GET_PARAMS=0, GET_TIME=1, VM_CREATE=2,
//!     VM_DESTROY=3, VM_BIND=4, GEM_CREATE=5, GEM_MMAP_OFFSET=6,
//!     GEM_BIND_OBJECT=7, QUEUE_CREATE=8, QUEUE_DESTROY=9, SUBMIT=10
//!   - GPU<->VM binding is done with VM_BIND (drm_asahi_vm_bind +
//!     drm_asahi_gem_bind_op), NOT a flat "GEM_BIND" ioctl.
//!
//! NOTE for the user: the structs here are version-pinned. An OLDER out-of-tree
//! revision (the AsahiLinux gpu/rust-wip branch) had MAX_CLUSTERS=32, a flat
//! DRM_ASAHI_GEM_BIND ioctl, and GET_PARAMS as DRM_IOWR. If a live GET_PARAMS on
//! the M1 returns garbage / EINVAL, check `uname -r` and the running asahi
//! driver's UABI against this mainlined layout.
//!
//! The DRM ioctl encoding is the standard Linux _IOC scheme over base 'd'
//! (0x64) with DRM_COMMAND_BASE = 0x40 (mirrors the virgl/virtgpu transport in
//! lib/prism/drivers/virgl/transport/linux.zig).

const std = @import("std");

// ---------------------------------------------------------------------------
// DRM ioctl encoding.
//   DRM_IOCTL_BASE = 'd' (0x64); DRM_COMMAND_BASE = 0x40.
//   _IOC builds a 32-bit request: dir(2) | size(14) | type(8) | nr(8).
//   GET_PARAMS / VM_DESTROY / VM_BIND / QUEUE_DESTROY / SUBMIT use _IOW
//   (write-only); the rest use _IOWR.
// ---------------------------------------------------------------------------

pub const DRM_IOCTL_BASE: u32 = 'd'; // 0x64
pub const DRM_COMMAND_BASE: u32 = 0x40;

const IOC_NONE: u32 = 0;
const IOC_WRITE: u32 = 1;
const IOC_READ: u32 = 2;
const IOC_NRSHIFT: u5 = 0;
const IOC_TYPESHIFT: u5 = 8;
const IOC_SIZESHIFT: u5 = 16;
const IOC_DIRSHIFT: u5 = 30;

fn ioc(dir: u32, typ: u32, nr: u32, size: u32) u32 {
    return (dir << IOC_DIRSHIFT) | (typ << IOC_TYPESHIFT) | (nr << IOC_NRSHIFT) | (size << IOC_SIZESHIFT);
}

/// DRM_IOW(DRM_COMMAND_BASE + nr, T): a write-only DRM ioctl request value.
pub fn drmIow(comptime nr: u32, comptime T: type) u32 {
    return ioc(IOC_WRITE, DRM_IOCTL_BASE, DRM_COMMAND_BASE + nr, @sizeOf(T));
}

/// DRM_IOWR(DRM_COMMAND_BASE + nr, T): a read-write DRM ioctl request value.
pub fn drmIowr(comptime nr: u32, comptime T: type) u32 {
    return ioc(IOC_READ | IOC_WRITE, DRM_IOCTL_BASE, DRM_COMMAND_BASE + nr, @sizeOf(T));
}

/// DRM_IOWR(nr, T) for a CORE DRM ioctl (drm.h, e.g. the syncobj group): the
/// nr is LITERAL - it is NOT offset by DRM_COMMAND_BASE the way the asahi
/// driver-private ioctls are.
pub fn drmCoreIowr(comptime nr: u32, comptime T: type) u32 {
    return ioc(IOC_READ | IOC_WRITE, DRM_IOCTL_BASE, nr, @sizeOf(T));
}

/// DRM_IOW(nr, T) for a CORE DRM ioctl (drm.h): write-only, LITERAL nr (not
/// DRM_COMMAND_BASE-offset). GEM_CLOSE is the one we need - it is _IOW in drm.h
/// (`#define DRM_IOCTL_GEM_CLOSE DRM_IOW(0x09, struct drm_gem_close)`).
pub fn drmCoreIow(comptime nr: u32, comptime T: type) u32 {
    return ioc(IOC_WRITE, DRM_IOCTL_BASE, nr, @sizeOf(T));
}

// ---------------------------------------------------------------------------
// Command numbers (the DRM_ASAHI_* enum in asahi_drm.h, mainlined order).
// ---------------------------------------------------------------------------

pub const DRM_ASAHI_GET_PARAMS: u32 = 0;
pub const DRM_ASAHI_GET_TIME: u32 = 1;
pub const DRM_ASAHI_VM_CREATE: u32 = 2;
pub const DRM_ASAHI_VM_DESTROY: u32 = 3;
pub const DRM_ASAHI_VM_BIND: u32 = 4;
pub const DRM_ASAHI_GEM_CREATE: u32 = 5;
pub const DRM_ASAHI_GEM_MMAP_OFFSET: u32 = 6;
pub const DRM_ASAHI_GEM_BIND_OBJECT: u32 = 7;
pub const DRM_ASAHI_QUEUE_CREATE: u32 = 8;
pub const DRM_ASAHI_QUEUE_DESTROY: u32 = 9;
pub const DRM_ASAHI_SUBMIT: u32 = 10;

pub const DRM_ASAHI_MAX_CLUSTERS: usize = 64;

// ---------------------------------------------------------------------------
// GEM / feature / bind flags.
// ---------------------------------------------------------------------------

/// enum drm_asahi_gem_flags.
pub const GEM_WRITEBACK: u32 = 1 << 0; // DRM_ASAHI_GEM_WRITEBACK (cacheable)
pub const GEM_VM_PRIVATE: u32 = 1 << 1; // DRM_ASAHI_GEM_VM_PRIVATE (bound to one VM)

/// enum drm_asahi_feature (the params_global.features bitset).
pub const FEATURE_SOFT_FAULTS: u64 = 1 << 0; // DRM_ASAHI_FEATURE_SOFT_FAULTS

/// enum drm_asahi_bind_flags (drm_asahi_gem_bind_op.flags).
pub const BIND_UNBIND: u32 = 1 << 0; // DRM_ASAHI_BIND_UNBIND
pub const BIND_READ: u32 = 1 << 1; // DRM_ASAHI_BIND_READ
pub const BIND_WRITE: u32 = 1 << 2; // DRM_ASAHI_BIND_WRITE
pub const BIND_SINGLE_PAGE: u32 = 1 << 3; // DRM_ASAHI_BIND_SINGLE_PAGE

// ---------------------------------------------------------------------------
// extern structs - kernel ABI, exact field order/types from asahi_drm.h.
// ---------------------------------------------------------------------------

/// struct drm_asahi_get_params - the GET_PARAMS ioctl argument. The kernel
/// writes the params struct to the user `pointer`, capped at `size` (so newer
/// kernels stay back-compatible with a smaller userspace buffer).
pub const drm_asahi_get_params = extern struct {
    /// Parameter group to fetch. MBZ (must be zero) for the global params.
    param_group: u32,
    /// MBZ.
    pad: u32,
    /// User pointer to write the parameter struct into.
    pointer: u64,
    /// Size of the user buffer at `pointer`.
    size: u64,
};

/// struct drm_asahi_params_global - the global GPU parameter block GET_PARAMS
/// fills in. This is THE GPU-identity + capability struct.
pub const drm_asahi_params_global = extern struct {
    /// Feature bitset (enum drm_asahi_feature, e.g. FEATURE_SOFT_FAULTS).
    features: u64,
    /// GPU architecture generation, e.g. 13 for "G13" (M1 family), 14 for
    /// "G14" (M2 family), 16 for "G16" (M3/M4 family).
    gpu_generation: u32,
    /// GPU variant as an ASCII character, e.g. 'G' (0x47) for G13G (base M1).
    /// Asahi variant letters: 'G' base, 'S' Pro/Max-class, 'C'/'D' Ultra dies.
    gpu_variant: u32,
    /// GPU revision in BCD, e.g. 0x00 for "A0", 0x21 for "C1".
    gpu_revision: u32,
    /// Chip ID in BCD, e.g. 0x8103 for T8103 (M1), 0x6000 for T6000 (M1 Pro).
    chip_id: u32,
    /// Number of GPU dies (1, or 2 for the Ultra parts).
    num_dies: u32,
    /// Total number of GPU clusters across all dies.
    num_clusters_total: u32,
    /// Cores per cluster.
    num_cores_per_cluster: u32,
    /// Max GPU frequency in kHz.
    max_frequency_khz: u32,
    /// Per-cluster active-core bitmasks (DRM_ASAHI_MAX_CLUSTERS entries).
    core_masks: [DRM_ASAHI_MAX_CLUSTERS]u64,
    /// Start of the usable GPU VA range.
    vm_start: u64,
    /// End of the usable GPU VA range.
    vm_end: u64,
    /// Minimum size of the kernel-reserved VA sub-range a VM must carve out.
    vm_kernel_min_size: u64,
    /// Max number of command buffers per SUBMIT.
    max_commands_per_submission: u32,
    /// Max render-pass attachments.
    max_attachments: u32,
    /// GPU command timestamp frequency in Hz.
    command_timestamp_frequency_hz: u64,
};

/// struct drm_asahi_get_time - GET_TIME ioctl argument (GPU timestamp).
pub const drm_asahi_get_time = extern struct {
    /// MBZ.
    flags: u64,
    /// Returned GPU timestamp.
    gpu_timestamp: u64,
};

/// struct drm_asahi_vm_create - VM_CREATE: allocate a GPU address space. The
/// kernel returns `vm_id`. `kernel_start`/`kernel_end` reserve a kernel-only VA
/// sub-range (>= vm_kernel_min_size) that userspace must not bind into.
pub const drm_asahi_vm_create = extern struct {
    kernel_start: u64,
    kernel_end: u64,
    /// Returned VM ID.
    vm_id: u32,
    /// MBZ.
    pad: u32,
};

/// struct drm_asahi_vm_destroy - VM_DESTROY.
pub const drm_asahi_vm_destroy = extern struct {
    /// VM ID to destroy.
    vm_id: u32,
    /// MBZ.
    pad: u32,
};

/// struct drm_asahi_gem_create - GEM_CREATE: allocate a GEM buffer object. The
/// kernel returns the GEM `handle`.
pub const drm_asahi_gem_create = extern struct {
    /// Size of the BO in bytes.
    size: u64,
    /// Combination of GEM_* flags (drm_asahi_gem_flags).
    flags: u32,
    /// VM ID to bind the BO to, when GEM_VM_PRIVATE is set in flags.
    vm_id: u32,
    /// Returned GEM handle.
    handle: u32,
    /// MBZ.
    pad: u32,
};

/// struct drm_asahi_gem_mmap_offset - GEM_MMAP_OFFSET: get the fake mmap offset
/// for a GEM handle, to pass as the mmap() file offset for a CPU mapping.
pub const drm_asahi_gem_mmap_offset = extern struct {
    /// GEM handle being mapped.
    handle: u32,
    /// MBZ.
    flags: u32,
    /// Returned fake offset to pass to mmap().
    offset: u64,
};

/// struct drm_asahi_gem_bind_op - one bind operation inside a VM_BIND batch.
/// Maps `range` bytes of GEM `handle` (from `offset`) into the VM at GPU VA
/// `addr`. `flags` is a drm_asahi_bind_flags combination (READ|WRITE to map,
/// UNBIND to remove).
pub const drm_asahi_gem_bind_op = extern struct {
    flags: u32,
    handle: u32,
    offset: u64,
    range: u64,
    addr: u64,
};

/// struct drm_asahi_vm_bind - VM_BIND: apply `num_binds` bind ops (a strided
/// array of drm_asahi_gem_bind_op at `userptr`) to VM `vm_id`.
pub const drm_asahi_vm_bind = extern struct {
    vm_id: u32,
    num_binds: u32,
    /// Stride in bytes between consecutive bind ops at `userptr`.
    stride: u32,
    /// MBZ.
    pad: u32,
    /// User pointer to the bind-op array.
    userptr: u64,
};

// ---------------------------------------------------------------------------
// P2: command queue + SUBMIT + sync.
//
// VERSION: same mainlined asahi UAPI pinned above (kernel ~6.16, the
// "agx-uapi v5/v6" series that landed). Source: torvalds/linux master
// include/uapi/drm/asahi_drm.h, cross-checked against the v6 patch on
// dri-devel (mail-archive.com msg537311). CONFIRMED field order/types below;
// the absolute @offsetOf/@sizeOf values are HAND-computed from the field
// types (no compiled C header on this box), exactly like the P1 structs -
// re-verify if a live QUEUE_CREATE/SUBMIT on the M1 misbehaves.
// ---------------------------------------------------------------------------

/// enum drm_asahi_priority - the QUEUE_CREATE priority field.
pub const PRIORITY_LOW: u32 = 0; // DRM_ASAHI_PRIORITY_LOW
pub const PRIORITY_MEDIUM: u32 = 1; // DRM_ASAHI_PRIORITY_MEDIUM
pub const PRIORITY_HIGH: u32 = 2; // DRM_ASAHI_PRIORITY_HIGH
pub const PRIORITY_REALTIME: u32 = 3; // DRM_ASAHI_PRIORITY_REALTIME

/// enum drm_asahi_sync_type - drm_asahi_sync.sync_type.
pub const SYNC_SYNCOBJ: u32 = 0; // DRM_ASAHI_SYNC_SYNCOBJ (binary syncobj)
pub const SYNC_TIMELINE_SYNCOBJ: u32 = 1; // DRM_ASAHI_SYNC_TIMELINE_SYNCOBJ

/// enum drm_asahi_cmd_type - drm_asahi_cmd_header.cmd_type. RENDER/COMPUTE are
/// the hardware commands; SET_*_ATTACHMENTS are synthetic software commands
/// that configure the following hardware command (they pass BARRIER_NONE).
pub const CMD_RENDER: u16 = 0; // DRM_ASAHI_CMD_RENDER
pub const CMD_COMPUTE: u16 = 1; // DRM_ASAHI_CMD_COMPUTE
pub const SET_VERTEX_ATTACHMENTS: u16 = 2; // DRM_ASAHI_SET_VERTEX_ATTACHMENTS
pub const SET_FRAGMENT_ATTACHMENTS: u16 = 3; // DRM_ASAHI_SET_FRAGMENT_ATTACHMENTS
pub const SET_COMPUTE_ATTACHMENTS: u16 = 4; // DRM_ASAHI_SET_COMPUTE_ATTACHMENTS

/// DRM_ASAHI_BARRIER_NONE - the no-barrier sentinel for the u16 vdm/cdm
/// barrier fields of drm_asahi_cmd_header. Software (SET_*_ATTACHMENTS)
/// commands MUST pass (NONE, NONE).
pub const BARRIER_NONE: u16 = 0xFFFF;

/// struct drm_asahi_queue_create - QUEUE_CREATE: create a GPU submission queue
/// bound to VM `vm_id`. The kernel returns `queue_id`. `usc_exec_base` is the
/// base GPU VA of the queue's USC (uniform/shader-code) executable region.
pub const drm_asahi_queue_create = extern struct {
    /// MBZ for now (no queue flags defined in this revision).
    flags: u32,
    /// VM ID this queue submits into.
    vm_id: u32,
    /// enum drm_asahi_priority.
    priority: u32,
    /// Returned queue ID.
    queue_id: u32,
    /// Base GPU VA of the USC executable region for this queue.
    usc_exec_base: u64,
};

/// struct drm_asahi_queue_destroy - QUEUE_DESTROY.
pub const drm_asahi_queue_destroy = extern struct {
    /// Queue ID to destroy.
    queue_id: u32,
    /// MBZ.
    pad: u32,
};

/// struct drm_asahi_sync - one in/out sync point for SUBMIT. `sync_type` is a
/// drm_asahi_sync_type; `handle` is a DRM syncobj handle (from
/// DRM_IOCTL_SYNCOBJ_CREATE); `timeline_value` is the point for a timeline
/// syncobj (0 for a binary SYNC_SYNCOBJ).
pub const drm_asahi_sync = extern struct {
    sync_type: u32,
    handle: u32,
    timeline_value: u64,
};

/// struct drm_asahi_cmd_header - the fixed-size header that prefixes each
/// command in a SUBMIT cmdbuf. The cmdbuf is a flat byte stream of
/// [header][payload][header][payload]... with no CPU pointers. `cmd_type` is a
/// drm_asahi_cmd_type, `size` is the payload size in bytes, and the
/// vdm_barrier/cdm_barrier are scheduling barriers (BARRIER_NONE = none).
pub const drm_asahi_cmd_header = extern struct {
    cmd_type: u16,
    size: u16,
    vdm_barrier: u16,
    cdm_barrier: u16,
};

// ---------------------------------------------------------------------------
// P3: the COMPUTE command body. After a drm_asahi_cmd_header{cmd_type=COMPUTE},
// the cmdbuf carries a drm_asahi_cmd_compute payload (size = @sizeOf below).
//
// VERSION/SOURCE: same mainlined asahi UAPI pin as P1/P2. The exact struct +
// its sub-structs are taken VERBATIM from torvalds/linux master
// include/uapi/drm/asahi_drm.h (the agx-uapi v5/v6 series that landed ~6.14+,
// the same revision the running 6.18.x kernel on the M1 carries). Field order
// and types are byte-confirmed against that header; the @offsetOf/@sizeOf
// asserts below are hand-walked from the C field types (no compiled C header on
// this box), exactly like the P1/P2 structs.
// ---------------------------------------------------------------------------

/// struct drm_asahi_timestamp - a (syncobj-backed) GPU timestamp write target:
/// `handle` is a DRM buffer/timestamp handle, `offset` a byte offset within it.
/// A zeroed timestamp (handle 0) means "no timestamp", which is what a minimal
/// compute submit uses.
pub const drm_asahi_timestamp = extern struct {
    handle: u32,
    offset: u32,
};

/// struct drm_asahi_timestamps - the start/end timestamp pair for a command.
pub const drm_asahi_timestamps = extern struct {
    start: drm_asahi_timestamp,
    end: drm_asahi_timestamp,
};

/// struct drm_asahi_helper_program - the "helper program" a command may use for
/// dynamic scratch/stack allocation. `binary` is a USC address (tagged pointer,
/// config in the low bits); `cfg` is extra config bits; `data` is an opaque
/// 64-bit sideband the kernel/firmware/hardware never interpret (userspace
/// usually puts a GPU VA of the helper's arguments here). A minimal compute
/// kernel that uses no scratch/spill leaves this entirely ZERO - Mesa only
/// fills it when `cs->scratch.cs.main || cs->scratch.cs.preamble` (hk_queue.c
/// asahi_fill_cdm_command), which our single-store kernel does not trigger.
pub const drm_asahi_helper_program = extern struct {
    binary: u32,
    cfg: u32,
    data: u64,
};

/// struct drm_asahi_cmd_compute - the COMPUTE command body. This describes one
/// CDM (compute data master) control stream of compute dispatches.
///   - `flags`: MBZ.
///   - `sampler_count`: number of samplers in the sampler heap (0 here).
///   - `cdm_ctrl_stream_base`: GPU VA of the start of the CDM control stream.
///   - `cdm_ctrl_stream_end`: GPU VA of the end of the FIRST contiguous segment
///     of the control stream (base + length, where length spans the launch
///     words through the stream-terminate word).
///   - `sampler_heap`: base GPU VA of the sampler heap (0 here).
///   - `helper`: the helper program (zeroed - no scratch).
///   - `ts`: start/end timestamps (zeroed - no timestamps).
/// Source: include/uapi/drm/asahi_drm.h (mainlined). Field order verbatim.
pub const drm_asahi_cmd_compute = extern struct {
    flags: u32,
    sampler_count: u32,
    cdm_ctrl_stream_base: u64,
    cdm_ctrl_stream_end: u64,
    sampler_heap: u64,
    helper: drm_asahi_helper_program,
    ts: drm_asahi_timestamps,
};

// ---------------------------------------------------------------------------
// P4: the RENDER command body + its sub-structs. After a
// drm_asahi_cmd_header{cmd_type=RENDER}, the cmdbuf carries a
// drm_asahi_cmd_render payload (size = @sizeOf below). A render pass also
// usually has a preceding SET_FRAGMENT_ATTACHMENTS / SET_VERTEX_ATTACHMENTS
// software command (drm_asahi_cmd_header{cmd_type=SET_*_ATTACHMENTS, size = N *
// @sizeOf(drm_asahi_attachment)} + an array of attachments). The attachments
// are documented as "purely a hint about the accessed memory regions ...
// optional to specify", so they are NOT the data path, but we emit a fragment
// attachment for the color buffer (the region the eot store writes) anyway.
//
// VERSION/SOURCE: byte-confirmed VERBATIM from torvalds/linux master
// include/uapi/drm/asahi_drm.h (the mainlined agx-uapi the running 6.18.x M1
// kernel carries) - the same fetch that confirmed the P1/P2/P3 structs. Every
// field name + C type + order matches; the @offsetOf/@sizeOf asserts below are
// hand-walked from the C types (no compiled C header on this box). Confirmed
// the struct has NO `fragment_attachments`/`attachment_count` members (those go
// through the separate SET_*_ATTACHMENTS software command), against an explicit
// re-fetch.
// ---------------------------------------------------------------------------

/// enum drm_asahi_render_flags - drm_asahi_cmd_render.flags bitset.
pub const RENDER_VERTEX_SCRATCH: u32 = 1 << 0; // DRM_ASAHI_RENDER_VERTEX_SCRATCH
/// Process even empty tiles. The kernel doc: "This must be set when clearing
/// render targets." Our bg program clears, so we MUST set this.
pub const RENDER_PROCESS_EMPTY_TILES: u32 = 1 << 1; // DRM_ASAHI_RENDER_PROCESS_EMPTY_TILES
pub const RENDER_NO_VERTEX_CLUSTERING: u32 = 1 << 2; // DRM_ASAHI_RENDER_NO_VERTEX_CLUSTERING
pub const RENDER_DBIAS_IS_INT: u32 = 1 << 18; // DRM_ASAHI_RENDER_DBIAS_IS_INT

/// struct drm_asahi_attachment - one render-pass attachment region. Carried by
/// a SET_*_ATTACHMENTS software command as an array. `pointer`/`size` describe
/// the memory region a shader (e.g. the eot store) touches; `pad`/`flags` MBZ.
/// "purely a hint about the accessed memory regions ... optional to specify".
pub const drm_asahi_attachment = extern struct {
    pointer: u64,
    size: u64,
    pad: u32,
    flags: u32,
};

/// struct drm_asahi_zls_buffer - a depth or stencil (Z load/store) buffer.
/// `base` is the buffer GPU VA, `comp_base` the compression-metadata VA (if
/// compressed), `stride`/`comp_stride` the per-layer byte strides. A render
/// pass with no depth/stencil leaves all four zero.
pub const drm_asahi_zls_buffer = extern struct {
    base: u64,
    comp_base: u64,
    stride: u32,
    comp_stride: u32,
};

/// struct drm_asahi_bg_eot - a background (load/clear) or end-of-tile (store)
/// program reference. `usc` is a USC address of the USC-words block (a 32-bit
/// offset relative to the queue's usc_exec_base, with configuration in the
/// bottom bits - Mesa ORs `| 4` for a single-RT program). `rsrc_spec` is the
/// packed "Counts" word (uniform/texture/sampler/preshader register counts).
/// The hardware auto-dispatches `bg` at tile start (to prime the tilebuffer)
/// and `eot` at tile end (to store the tilebuffer to memory); `partial_*` are
/// the variants the GPU uses when it splits a render pass.
pub const drm_asahi_bg_eot = extern struct {
    usc: u32,
    rsrc_spec: u32,
};

/// struct drm_asahi_cmd_render - the RENDER command body: one render pass
/// (clear -> draw(s) -> store) over a tile-based deferred-rendering framebuffer.
/// Field order/types VERBATIM from include/uapi/drm/asahi_drm.h (mainlined).
///   - `flags`: drm_asahi_render_flags.
///   - `isp_zls_pixels`: ISP_ZLS_PIXELS (depth/stencil dims, packed; 0 if none).
///   - `vdm_ctrl_stream_base`: GPU VA of the VDM (vertex/draw) control stream.
///   - `vertex_helper`/`fragment_helper`: scratch helper programs (zero unless a
///     shader spills - we use none).
///   - `isp_scissor_base`/`isp_dbias_base`/`isp_oclqry_base`: ISP register array
///     bases (scissor / depth-bias / occlusion-query; 0 if unused).
///   - `depth`/`stencil`: the ZLS buffers (zero - no depth/stencil here).
///   - `zls_ctrl`: ZLS_CTRL register (0 - no depth/stencil).
///   - `ppp_multisamplectl`: PPP_MULTISAMPLECTL (packed sample positions).
///   - `sampler_heap`/`sampler_count`: sampler heap (0 - no samplers).
///   - `ppp_ctrl`: PPP_CTRL register.
///   - `width_px`/`height_px`/`layers`: framebuffer dims.
///   - `utile_width_px`/`utile_height_px`: hardware tile dims.
///   - `samples`/`sample_size_B`: MSAA count + tilebuffer bytes-per-sample.
///   - `isp_merge_upper_x`/`_y`: f32 triangle-merge thresholds.
///   - `bg`/`eot`/`partial_bg`/`partial_eot`: the load/store programs.
///   - `isp_bgobjdepth`/`isp_bgobjvals`: background clear depth/stencil values.
///   - `ts_vtx`/`ts_frag`: vertex/fragment timestamps (zeroed - none).
pub const drm_asahi_cmd_render = extern struct {
    flags: u32,
    isp_zls_pixels: u32,
    vdm_ctrl_stream_base: u64,
    vertex_helper: drm_asahi_helper_program,
    fragment_helper: drm_asahi_helper_program,
    isp_scissor_base: u64,
    isp_dbias_base: u64,
    isp_oclqry_base: u64,
    depth: drm_asahi_zls_buffer,
    stencil: drm_asahi_zls_buffer,
    zls_ctrl: u64,
    ppp_multisamplectl: u64,
    sampler_heap: u64,
    ppp_ctrl: u32,
    width_px: u16,
    height_px: u16,
    layers: u16,
    sampler_count: u16,
    utile_width_px: u8,
    utile_height_px: u8,
    samples: u8,
    sample_size_B: u8,
    isp_merge_upper_x: u32,
    isp_merge_upper_y: u32,
    bg: drm_asahi_bg_eot,
    eot: drm_asahi_bg_eot,
    partial_bg: drm_asahi_bg_eot,
    partial_eot: drm_asahi_bg_eot,
    isp_bgobjdepth: u32,
    isp_bgobjvals: u32,
    ts_vtx: drm_asahi_timestamps,
    ts_frag: drm_asahi_timestamps,
};

/// struct drm_asahi_submit - SUBMIT: run the command stream at `cmdbuf`
/// (`cmdbuf_size` bytes, a sequence of drm_asahi_cmd_header + payload) on queue
/// `queue_id`, after waiting on the first `in_sync_count` drm_asahi_sync at
/// `syncs` and signalling the next `out_sync_count` on completion.
pub const drm_asahi_submit = extern struct {
    /// User pointer to a drm_asahi_sync array: in_sync_count waits followed by
    /// out_sync_count signals.
    syncs: u64,
    /// User pointer to the command-buffer byte stream.
    cmdbuf: u64,
    /// MBZ for now (no submit flags defined in this revision).
    flags: u32,
    /// Queue ID to submit on.
    queue_id: u32,
    /// Number of leading drm_asahi_sync at `syncs` to wait on.
    in_sync_count: u32,
    /// Number of trailing drm_asahi_sync at `syncs` to signal.
    out_sync_count: u32,
    /// Size of the `cmdbuf` byte stream.
    cmdbuf_size: u32,
    /// MBZ.
    pad: u32,
};

// ---------------------------------------------------------------------------
// Generic DRM core syncobj ioctls (NOT asahi-specific). SUBMIT waits on /
// signals these standard DRM syncobjs. Source: torvalds/linux master
// include/uapi/drm/drm.h. The nr values are LITERAL (0xBF..) - they are core
// DRM ioctls, so they are NOT offset by DRM_COMMAND_BASE the way the asahi
// driver ioctls are. All four are DRM_IOWR over base 'd'.
// ---------------------------------------------------------------------------

/// struct drm_gem_close - DRM_IOCTL_GEM_CLOSE (drm.h core ioctl, nr 0x09). Drops
/// this process's reference to GEM `handle`; when the last reference goes away
/// the BO (and its backing memory) is freed. `pad` is MBZ. This is the per-BO
/// release the asahi VM_BIND-unbind path pairs with (unbind the VA, then close
/// the handle).
pub const drm_gem_close = extern struct {
    handle: u32,
    pad: u32,
};

pub const DRM_SYNCOBJ_CREATE_SIGNALED: u32 = 1 << 0;

pub const DRM_SYNCOBJ_WAIT_FLAGS_WAIT_ALL: u32 = 1 << 0;
pub const DRM_SYNCOBJ_WAIT_FLAGS_WAIT_FOR_SUBMIT: u32 = 1 << 1;
pub const DRM_SYNCOBJ_WAIT_FLAGS_WAIT_AVAILABLE: u32 = 1 << 2;
pub const DRM_SYNCOBJ_WAIT_FLAGS_WAIT_DEADLINE: u32 = 1 << 3;

/// struct drm_syncobj_create - DRM_IOCTL_SYNCOBJ_CREATE. The kernel returns a
/// syncobj `handle`. flags may include DRM_SYNCOBJ_CREATE_SIGNALED.
pub const drm_syncobj_create = extern struct {
    handle: u32,
    flags: u32,
};

/// struct drm_syncobj_destroy - DRM_IOCTL_SYNCOBJ_DESTROY.
pub const drm_syncobj_destroy = extern struct {
    handle: u32,
    pad: u32,
};

/// struct drm_syncobj_array - the argument for DRM_IOCTL_SYNCOBJ_SIGNAL (and
/// RESET): `handles` is a user pointer to a u32 array of `count_handles`
/// syncobj handles to signal.
pub const drm_syncobj_array = extern struct {
    handles: u64,
    count_handles: u32,
    pad: u32,
};

/// struct drm_syncobj_wait - DRM_IOCTL_SYNCOBJ_WAIT. Wait for `count_handles`
/// syncobjs (a u32 array at `handles`) until `timeout_nsec` (an ABSOLUTE
/// CLOCK_MONOTONIC deadline). `deadline_nsec` is the recent scheduler-deadline
/// field (only consulted with WAIT_FLAGS_WAIT_DEADLINE), so its presence grows
/// the struct to 40 bytes - the ioctl size encoding must match the running
/// kernel's drm.h.
pub const drm_syncobj_wait = extern struct {
    handles: u64,
    timeout_nsec: i64,
    count_handles: u32,
    flags: u32,
    first_signaled: u32,
    pad: u32,
    deadline_nsec: u64,
};

// ---------------------------------------------------------------------------
// Encoded ioctl request values.
// ---------------------------------------------------------------------------

pub const IOCTL_GET_PARAMS = drmIow(DRM_ASAHI_GET_PARAMS, drm_asahi_get_params);
pub const IOCTL_GET_TIME = drmIowr(DRM_ASAHI_GET_TIME, drm_asahi_get_time);
pub const IOCTL_VM_CREATE = drmIowr(DRM_ASAHI_VM_CREATE, drm_asahi_vm_create);
pub const IOCTL_VM_DESTROY = drmIow(DRM_ASAHI_VM_DESTROY, drm_asahi_vm_destroy);
pub const IOCTL_VM_BIND = drmIow(DRM_ASAHI_VM_BIND, drm_asahi_vm_bind);
pub const IOCTL_GEM_CREATE = drmIowr(DRM_ASAHI_GEM_CREATE, drm_asahi_gem_create);
pub const IOCTL_GEM_MMAP_OFFSET = drmIowr(DRM_ASAHI_GEM_MMAP_OFFSET, drm_asahi_gem_mmap_offset);

// QUEUE_CREATE is _IOWR; QUEUE_DESTROY and SUBMIT are _IOW (write-only).
pub const IOCTL_QUEUE_CREATE = drmIowr(DRM_ASAHI_QUEUE_CREATE, drm_asahi_queue_create);
pub const IOCTL_QUEUE_DESTROY = drmIow(DRM_ASAHI_QUEUE_DESTROY, drm_asahi_queue_destroy);
pub const IOCTL_SUBMIT = drmIow(DRM_ASAHI_SUBMIT, drm_asahi_submit);

// Core DRM ioctl nrs (literal, from drm.h - NOT DRM_COMMAND_BASE-offset).
// GEM_CLOSE is the standard handle-release ioctl (DRM_IOW, nr 0x09).
pub const DRM_GEM_CLOSE: u32 = 0x09;
pub const IOCTL_GEM_CLOSE = drmCoreIow(DRM_GEM_CLOSE, drm_gem_close);

// Core DRM syncobj ioctl nrs (literal, from drm.h).
pub const DRM_SYNCOBJ_CREATE: u32 = 0xBF;
pub const DRM_SYNCOBJ_DESTROY: u32 = 0xC0;
pub const DRM_SYNCOBJ_WAIT: u32 = 0xC3;
pub const DRM_SYNCOBJ_SIGNAL: u32 = 0xC5;

pub const IOCTL_SYNCOBJ_CREATE = drmCoreIowr(DRM_SYNCOBJ_CREATE, drm_syncobj_create);
pub const IOCTL_SYNCOBJ_DESTROY = drmCoreIowr(DRM_SYNCOBJ_DESTROY, drm_syncobj_destroy);
pub const IOCTL_SYNCOBJ_WAIT = drmCoreIowr(DRM_SYNCOBJ_WAIT, drm_syncobj_wait);
pub const IOCTL_SYNCOBJ_SIGNAL = drmCoreIowr(DRM_SYNCOBJ_SIGNAL, drm_syncobj_array);

// ---------------------------------------------------------------------------
// Layout asserts (mirror nvidia sdk.zig's @sizeOf/@offsetOf pattern). These
// pin the struct layouts to the kernel header at comptime, so any accidental
// field reorder/type change fails the build instead of corrupting an ioctl.
// All values computed by hand against the mainlined asahi_drm.h field types.
// ---------------------------------------------------------------------------

comptime {
    // drm_asahi_get_params: u32,u32,u64,u64 -> 24 bytes (8-aligned).
    std.debug.assert(@sizeOf(drm_asahi_get_params) == 24);
    std.debug.assert(@offsetOf(drm_asahi_get_params, "pointer") == 8);
    std.debug.assert(@offsetOf(drm_asahi_get_params, "size") == 16);

    // drm_asahi_params_global. Layout walk (8-byte aligned struct):
    //   features              u64 @0
    //   gpu_generation        u32 @8
    //   gpu_variant           u32 @12
    //   gpu_revision          u32 @16
    //   chip_id               u32 @20
    //   num_dies              u32 @24
    //   num_clusters_total    u32 @28
    //   num_cores_per_cluster u32 @32
    //   max_frequency_khz     u32 @36
    //   core_masks[64] u64     @40 .. @40+512=552
    //   vm_start              u64 @552
    //   vm_end                u64 @560
    //   vm_kernel_min_size    u64 @568
    //   max_commands_per_submission u32 @576
    //   max_attachments       u32 @580
    //   command_timestamp_frequency_hz u64 @584 .. @592
    std.debug.assert(@offsetOf(drm_asahi_params_global, "gpu_generation") == 8);
    std.debug.assert(@offsetOf(drm_asahi_params_global, "gpu_variant") == 12);
    std.debug.assert(@offsetOf(drm_asahi_params_global, "gpu_revision") == 16);
    std.debug.assert(@offsetOf(drm_asahi_params_global, "chip_id") == 20);
    std.debug.assert(@offsetOf(drm_asahi_params_global, "num_dies") == 24);
    std.debug.assert(@offsetOf(drm_asahi_params_global, "max_frequency_khz") == 36);
    std.debug.assert(@offsetOf(drm_asahi_params_global, "core_masks") == 40);
    std.debug.assert(@offsetOf(drm_asahi_params_global, "vm_start") == 552);
    std.debug.assert(@offsetOf(drm_asahi_params_global, "vm_kernel_min_size") == 568);
    std.debug.assert(@offsetOf(drm_asahi_params_global, "max_commands_per_submission") == 576);
    std.debug.assert(@offsetOf(drm_asahi_params_global, "max_attachments") == 580);
    std.debug.assert(@offsetOf(drm_asahi_params_global, "command_timestamp_frequency_hz") == 584);
    std.debug.assert(@sizeOf(drm_asahi_params_global) == 592);

    // drm_asahi_vm_create: u64,u64,u32,u32 -> 24 bytes.
    std.debug.assert(@sizeOf(drm_asahi_vm_create) == 24);
    std.debug.assert(@offsetOf(drm_asahi_vm_create, "vm_id") == 16);

    // drm_asahi_vm_destroy: u32,u32 -> 8 bytes.
    std.debug.assert(@sizeOf(drm_asahi_vm_destroy) == 8);

    // drm_asahi_gem_create: u64,u32,u32,u32,u32 -> 24 bytes.
    std.debug.assert(@sizeOf(drm_asahi_gem_create) == 24);
    std.debug.assert(@offsetOf(drm_asahi_gem_create, "flags") == 8);
    std.debug.assert(@offsetOf(drm_asahi_gem_create, "vm_id") == 12);
    std.debug.assert(@offsetOf(drm_asahi_gem_create, "handle") == 16);

    // drm_asahi_gem_mmap_offset: u32,u32,u64 -> 16 bytes.
    std.debug.assert(@sizeOf(drm_asahi_gem_mmap_offset) == 16);
    std.debug.assert(@offsetOf(drm_asahi_gem_mmap_offset, "offset") == 8);

    // drm_asahi_gem_bind_op: u32,u32,u64,u64,u64 -> 32 bytes.
    std.debug.assert(@sizeOf(drm_asahi_gem_bind_op) == 32);
    std.debug.assert(@offsetOf(drm_asahi_gem_bind_op, "offset") == 8);
    std.debug.assert(@offsetOf(drm_asahi_gem_bind_op, "range") == 16);
    std.debug.assert(@offsetOf(drm_asahi_gem_bind_op, "addr") == 24);

    // drm_asahi_vm_bind: u32,u32,u32,u32,u64 -> 24 bytes.
    std.debug.assert(@sizeOf(drm_asahi_vm_bind) == 24);
    std.debug.assert(@offsetOf(drm_asahi_vm_bind, "userptr") == 16);

    // drm_asahi_get_time: u64,u64 -> 16 bytes.
    std.debug.assert(@sizeOf(drm_asahi_get_time) == 16);

    // P2 structs.
    // drm_asahi_queue_create: u32,u32,u32,u32,u64 -> 24 bytes, u64@16.
    std.debug.assert(@sizeOf(drm_asahi_queue_create) == 24);
    std.debug.assert(@offsetOf(drm_asahi_queue_create, "vm_id") == 4);
    std.debug.assert(@offsetOf(drm_asahi_queue_create, "priority") == 8);
    std.debug.assert(@offsetOf(drm_asahi_queue_create, "queue_id") == 12);
    std.debug.assert(@offsetOf(drm_asahi_queue_create, "usc_exec_base") == 16);

    // drm_asahi_queue_destroy: u32,u32 -> 8 bytes.
    std.debug.assert(@sizeOf(drm_asahi_queue_destroy) == 8);

    // drm_asahi_sync: u32,u32,u64 -> 16 bytes, timeline_value@8.
    std.debug.assert(@sizeOf(drm_asahi_sync) == 16);
    std.debug.assert(@offsetOf(drm_asahi_sync, "handle") == 4);
    std.debug.assert(@offsetOf(drm_asahi_sync, "timeline_value") == 8);

    // drm_asahi_cmd_header: u16,u16,u16,u16 -> 8 bytes.
    std.debug.assert(@sizeOf(drm_asahi_cmd_header) == 8);
    std.debug.assert(@offsetOf(drm_asahi_cmd_header, "size") == 2);
    std.debug.assert(@offsetOf(drm_asahi_cmd_header, "vdm_barrier") == 4);
    std.debug.assert(@offsetOf(drm_asahi_cmd_header, "cdm_barrier") == 6);

    // P3 compute structs.
    // drm_asahi_timestamp: u32,u32 -> 8 bytes.
    std.debug.assert(@sizeOf(drm_asahi_timestamp) == 8);
    std.debug.assert(@offsetOf(drm_asahi_timestamp, "offset") == 4);

    // drm_asahi_timestamps: two timestamps -> 16 bytes, end@8.
    std.debug.assert(@sizeOf(drm_asahi_timestamps) == 16);
    std.debug.assert(@offsetOf(drm_asahi_timestamps, "end") == 8);

    // drm_asahi_helper_program: u32,u32,u64 -> 16 bytes, data@8.
    std.debug.assert(@sizeOf(drm_asahi_helper_program) == 16);
    std.debug.assert(@offsetOf(drm_asahi_helper_program, "cfg") == 4);
    std.debug.assert(@offsetOf(drm_asahi_helper_program, "data") == 8);

    // drm_asahi_cmd_compute layout walk (8-byte aligned):
    //   flags                u32 @0
    //   sampler_count        u32 @4
    //   cdm_ctrl_stream_base u64 @8
    //   cdm_ctrl_stream_end  u64 @16
    //   sampler_heap         u64 @24
    //   helper (16B)             @32 .. @48
    //   ts     (16B)             @48 .. @64
    std.debug.assert(@offsetOf(drm_asahi_cmd_compute, "sampler_count") == 4);
    std.debug.assert(@offsetOf(drm_asahi_cmd_compute, "cdm_ctrl_stream_base") == 8);
    std.debug.assert(@offsetOf(drm_asahi_cmd_compute, "cdm_ctrl_stream_end") == 16);
    std.debug.assert(@offsetOf(drm_asahi_cmd_compute, "sampler_heap") == 24);
    std.debug.assert(@offsetOf(drm_asahi_cmd_compute, "helper") == 32);
    std.debug.assert(@offsetOf(drm_asahi_cmd_compute, "ts") == 48);
    std.debug.assert(@sizeOf(drm_asahi_cmd_compute) == 64);

    // P4 render structs.
    // drm_asahi_attachment: u64,u64,u32,u32 -> 24 bytes.
    std.debug.assert(@sizeOf(drm_asahi_attachment) == 24);
    std.debug.assert(@offsetOf(drm_asahi_attachment, "size") == 8);
    std.debug.assert(@offsetOf(drm_asahi_attachment, "pad") == 16);
    std.debug.assert(@offsetOf(drm_asahi_attachment, "flags") == 20);

    // drm_asahi_zls_buffer: u64,u64,u32,u32 -> 24 bytes.
    std.debug.assert(@sizeOf(drm_asahi_zls_buffer) == 24);
    std.debug.assert(@offsetOf(drm_asahi_zls_buffer, "comp_base") == 8);
    std.debug.assert(@offsetOf(drm_asahi_zls_buffer, "stride") == 16);
    std.debug.assert(@offsetOf(drm_asahi_zls_buffer, "comp_stride") == 20);

    // drm_asahi_bg_eot: u32,u32 -> 8 bytes.
    std.debug.assert(@sizeOf(drm_asahi_bg_eot) == 8);
    std.debug.assert(@offsetOf(drm_asahi_bg_eot, "rsrc_spec") == 4);

    // drm_asahi_cmd_render layout walk (8-byte aligned struct). The depth/stencil
    // ZLS buffers, the helper programs, the bg/eot programs, and the timestamps
    // are all sub-structs; the u16/u8 block (ppp_ctrl..sample_size_B) packs
    // tightly into 16 bytes @144..160.
    //   flags                u32 @0
    //   isp_zls_pixels       u32 @4
    //   vdm_ctrl_stream_base u64 @8
    //   vertex_helper (16B)      @16
    //   fragment_helper (16B)    @32
    //   isp_scissor_base     u64 @48
    //   isp_dbias_base       u64 @56
    //   isp_oclqry_base      u64 @64
    //   depth (zls 24B)          @72
    //   stencil (zls 24B)        @96
    //   zls_ctrl             u64 @120
    //   ppp_multisamplectl   u64 @128
    //   sampler_heap         u64 @136
    //   ppp_ctrl             u32 @144
    //   width_px             u16 @148
    //   height_px            u16 @150
    //   layers               u16 @152
    //   sampler_count        u16 @154
    //   utile_width_px        u8 @156
    //   utile_height_px       u8 @157
    //   samples               u8 @158
    //   sample_size_B         u8 @159
    //   isp_merge_upper_x    u32 @160
    //   isp_merge_upper_y    u32 @164
    //   bg (8B)                  @168
    //   eot (8B)                 @176
    //   partial_bg (8B)          @184
    //   partial_eot (8B)         @192
    //   isp_bgobjdepth       u32 @200
    //   isp_bgobjvals        u32 @204
    //   ts_vtx (16B)             @208
    //   ts_frag (16B)            @224 .. @240
    std.debug.assert(@offsetOf(drm_asahi_cmd_render, "isp_zls_pixels") == 4);
    std.debug.assert(@offsetOf(drm_asahi_cmd_render, "vdm_ctrl_stream_base") == 8);
    std.debug.assert(@offsetOf(drm_asahi_cmd_render, "vertex_helper") == 16);
    std.debug.assert(@offsetOf(drm_asahi_cmd_render, "fragment_helper") == 32);
    std.debug.assert(@offsetOf(drm_asahi_cmd_render, "isp_scissor_base") == 48);
    std.debug.assert(@offsetOf(drm_asahi_cmd_render, "isp_dbias_base") == 56);
    std.debug.assert(@offsetOf(drm_asahi_cmd_render, "isp_oclqry_base") == 64);
    std.debug.assert(@offsetOf(drm_asahi_cmd_render, "depth") == 72);
    std.debug.assert(@offsetOf(drm_asahi_cmd_render, "stencil") == 96);
    std.debug.assert(@offsetOf(drm_asahi_cmd_render, "zls_ctrl") == 120);
    std.debug.assert(@offsetOf(drm_asahi_cmd_render, "ppp_multisamplectl") == 128);
    std.debug.assert(@offsetOf(drm_asahi_cmd_render, "sampler_heap") == 136);
    std.debug.assert(@offsetOf(drm_asahi_cmd_render, "ppp_ctrl") == 144);
    std.debug.assert(@offsetOf(drm_asahi_cmd_render, "width_px") == 148);
    std.debug.assert(@offsetOf(drm_asahi_cmd_render, "height_px") == 150);
    std.debug.assert(@offsetOf(drm_asahi_cmd_render, "layers") == 152);
    std.debug.assert(@offsetOf(drm_asahi_cmd_render, "sampler_count") == 154);
    std.debug.assert(@offsetOf(drm_asahi_cmd_render, "utile_width_px") == 156);
    std.debug.assert(@offsetOf(drm_asahi_cmd_render, "utile_height_px") == 157);
    std.debug.assert(@offsetOf(drm_asahi_cmd_render, "samples") == 158);
    std.debug.assert(@offsetOf(drm_asahi_cmd_render, "sample_size_B") == 159);
    std.debug.assert(@offsetOf(drm_asahi_cmd_render, "isp_merge_upper_x") == 160);
    std.debug.assert(@offsetOf(drm_asahi_cmd_render, "isp_merge_upper_y") == 164);
    std.debug.assert(@offsetOf(drm_asahi_cmd_render, "bg") == 168);
    std.debug.assert(@offsetOf(drm_asahi_cmd_render, "eot") == 176);
    std.debug.assert(@offsetOf(drm_asahi_cmd_render, "partial_bg") == 184);
    std.debug.assert(@offsetOf(drm_asahi_cmd_render, "partial_eot") == 192);
    std.debug.assert(@offsetOf(drm_asahi_cmd_render, "isp_bgobjdepth") == 200);
    std.debug.assert(@offsetOf(drm_asahi_cmd_render, "isp_bgobjvals") == 204);
    std.debug.assert(@offsetOf(drm_asahi_cmd_render, "ts_vtx") == 208);
    std.debug.assert(@offsetOf(drm_asahi_cmd_render, "ts_frag") == 224);
    std.debug.assert(@sizeOf(drm_asahi_cmd_render) == 240);

    // drm_asahi_submit: u64,u64,u32*6 -> 40 bytes.
    std.debug.assert(@sizeOf(drm_asahi_submit) == 40);
    std.debug.assert(@offsetOf(drm_asahi_submit, "cmdbuf") == 8);
    std.debug.assert(@offsetOf(drm_asahi_submit, "flags") == 16);
    std.debug.assert(@offsetOf(drm_asahi_submit, "queue_id") == 20);
    std.debug.assert(@offsetOf(drm_asahi_submit, "in_sync_count") == 24);
    std.debug.assert(@offsetOf(drm_asahi_submit, "out_sync_count") == 28);
    std.debug.assert(@offsetOf(drm_asahi_submit, "cmdbuf_size") == 32);

    // Core DRM syncobj structs.
    // drm_syncobj_create / drm_syncobj_destroy: u32,u32 -> 8 bytes.
    std.debug.assert(@sizeOf(drm_syncobj_create) == 8);
    std.debug.assert(@sizeOf(drm_syncobj_destroy) == 8);

    // drm_gem_close: u32,u32 -> 8 bytes.
    std.debug.assert(@sizeOf(drm_gem_close) == 8);
    std.debug.assert(@offsetOf(drm_gem_close, "pad") == 4);

    // drm_syncobj_array: u64,u32,u32 -> 16 bytes.
    std.debug.assert(@sizeOf(drm_syncobj_array) == 16);
    std.debug.assert(@offsetOf(drm_syncobj_array, "count_handles") == 8);

    // drm_syncobj_wait: u64,i64,u32,u32,u32,u32,u64 -> 40 bytes.
    std.debug.assert(@sizeOf(drm_syncobj_wait) == 40);
    std.debug.assert(@offsetOf(drm_syncobj_wait, "timeout_nsec") == 8);
    std.debug.assert(@offsetOf(drm_syncobj_wait, "count_handles") == 16);
    std.debug.assert(@offsetOf(drm_syncobj_wait, "flags") == 20);
    std.debug.assert(@offsetOf(drm_syncobj_wait, "first_signaled") == 24);
    std.debug.assert(@offsetOf(drm_syncobj_wait, "deadline_nsec") == 32);
}

test "ioctl request encodings match the DRM _IOC scheme" {
    // _IOW('d', 0x40+0, drm_asahi_get_params): dir=1, size=24, type=0x64, nr=0x40.
    const expect_get_params: u32 = (IOC_WRITE << IOC_DIRSHIFT) |
        (@as(u32, 24) << IOC_SIZESHIFT) | (DRM_IOCTL_BASE << IOC_TYPESHIFT) |
        ((DRM_COMMAND_BASE + 0) << IOC_NRSHIFT);
    try std.testing.expectEqual(expect_get_params, IOCTL_GET_PARAMS);

    // VM_CREATE is _IOWR(0x40+2, 24-byte): dir=3.
    const expect_vm_create: u32 = ((IOC_READ | IOC_WRITE) << IOC_DIRSHIFT) |
        (@as(u32, 24) << IOC_SIZESHIFT) | (DRM_IOCTL_BASE << IOC_TYPESHIFT) |
        ((DRM_COMMAND_BASE + 2) << IOC_NRSHIFT);
    try std.testing.expectEqual(expect_vm_create, IOCTL_VM_CREATE);

    // GEM_CREATE is _IOWR(0x40+5, 24-byte).
    const expect_gem_create: u32 = ((IOC_READ | IOC_WRITE) << IOC_DIRSHIFT) |
        (@as(u32, 24) << IOC_SIZESHIFT) | (DRM_IOCTL_BASE << IOC_TYPESHIFT) |
        ((DRM_COMMAND_BASE + 5) << IOC_NRSHIFT);
    try std.testing.expectEqual(expect_gem_create, IOCTL_GEM_CREATE);
}

test "P2 queue/submit/sync struct sizes and ioctl encodings" {
    // Struct sizes the SUBMIT/queue path marshals.
    try std.testing.expectEqual(@as(usize, 24), @sizeOf(drm_asahi_queue_create));
    try std.testing.expectEqual(@as(usize, 8), @sizeOf(drm_asahi_queue_destroy));
    try std.testing.expectEqual(@as(usize, 16), @sizeOf(drm_asahi_sync));
    try std.testing.expectEqual(@as(usize, 8), @sizeOf(drm_asahi_cmd_header));
    try std.testing.expectEqual(@as(usize, 40), @sizeOf(drm_asahi_submit));
    try std.testing.expectEqual(@as(usize, 8), @sizeOf(drm_syncobj_create));
    try std.testing.expectEqual(@as(usize, 16), @sizeOf(drm_syncobj_array));
    try std.testing.expectEqual(@as(usize, 40), @sizeOf(drm_syncobj_wait));

    // QUEUE_CREATE is _IOWR(0x40+8, 24-byte): dir=3.
    const expect_queue_create: u32 = ((IOC_READ | IOC_WRITE) << IOC_DIRSHIFT) |
        (@as(u32, 24) << IOC_SIZESHIFT) | (DRM_IOCTL_BASE << IOC_TYPESHIFT) |
        ((DRM_COMMAND_BASE + 8) << IOC_NRSHIFT);
    try std.testing.expectEqual(expect_queue_create, IOCTL_QUEUE_CREATE);

    // QUEUE_DESTROY is _IOW(0x40+9, 8-byte): dir=1.
    const expect_queue_destroy: u32 = (IOC_WRITE << IOC_DIRSHIFT) |
        (@as(u32, 8) << IOC_SIZESHIFT) | (DRM_IOCTL_BASE << IOC_TYPESHIFT) |
        ((DRM_COMMAND_BASE + 9) << IOC_NRSHIFT);
    try std.testing.expectEqual(expect_queue_destroy, IOCTL_QUEUE_DESTROY);

    // SUBMIT is _IOW(0x40+10, 40-byte): dir=1.
    const expect_submit: u32 = (IOC_WRITE << IOC_DIRSHIFT) |
        (@as(u32, 40) << IOC_SIZESHIFT) | (DRM_IOCTL_BASE << IOC_TYPESHIFT) |
        ((DRM_COMMAND_BASE + 10) << IOC_NRSHIFT);
    try std.testing.expectEqual(expect_submit, IOCTL_SUBMIT);

    // Core syncobj ioctls: LITERAL nr (no DRM_COMMAND_BASE offset), all _IOWR.
    // SYNCOBJ_CREATE is _IOWR(0xBF, 8-byte).
    const expect_so_create: u32 = ((IOC_READ | IOC_WRITE) << IOC_DIRSHIFT) |
        (@as(u32, 8) << IOC_SIZESHIFT) | (DRM_IOCTL_BASE << IOC_TYPESHIFT) |
        (@as(u32, 0xBF) << IOC_NRSHIFT);
    try std.testing.expectEqual(expect_so_create, IOCTL_SYNCOBJ_CREATE);

    // SYNCOBJ_WAIT is _IOWR(0xC3, 40-byte).
    const expect_so_wait: u32 = ((IOC_READ | IOC_WRITE) << IOC_DIRSHIFT) |
        (@as(u32, 40) << IOC_SIZESHIFT) | (DRM_IOCTL_BASE << IOC_TYPESHIFT) |
        (@as(u32, 0xC3) << IOC_NRSHIFT);
    try std.testing.expectEqual(expect_so_wait, IOCTL_SYNCOBJ_WAIT);

    // SYNCOBJ_SIGNAL is _IOWR(0xC5, 16-byte drm_syncobj_array).
    const expect_so_signal: u32 = ((IOC_READ | IOC_WRITE) << IOC_DIRSHIFT) |
        (@as(u32, 16) << IOC_SIZESHIFT) | (DRM_IOCTL_BASE << IOC_TYPESHIFT) |
        (@as(u32, 0xC5) << IOC_NRSHIFT);
    try std.testing.expectEqual(expect_so_signal, IOCTL_SYNCOBJ_SIGNAL);

    // GEM_CLOSE is a core DRM ioctl: _IOW(0x09, 8-byte drm_gem_close), dir=1,
    // LITERAL nr (no DRM_COMMAND_BASE offset).
    const expect_gem_close: u32 = (IOC_WRITE << IOC_DIRSHIFT) |
        (@as(u32, 8) << IOC_SIZESHIFT) | (DRM_IOCTL_BASE << IOC_TYPESHIFT) |
        (@as(u32, 0x09) << IOC_NRSHIFT);
    try std.testing.expectEqual(expect_gem_close, IOCTL_GEM_CLOSE);
}

test "layout asserts are reachable at runtime too" {
    try std.testing.expectEqual(@as(usize, 592), @sizeOf(drm_asahi_params_global));
    try std.testing.expectEqual(@as(usize, 24), @sizeOf(drm_asahi_get_params));
}
