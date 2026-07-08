//! Prism Vulkan ICD. Maps each available compiled-in HAL driver to a VkPhysicalDevice.
//! Dispatch logic lives here. lib/prism-vk.zig re-exports the loader entry points.
//! All Vulkan-facing functions use callconv(.c) with C ABI types from vk.zig.

const std = @import("std");
const prism = @import("prism");
const vk = @import("vk.zig");
/// The libwayland-client WSI (standard-app present path via wl_shm).
const wsi_wl = @import("wsi_wayland.zig");

/// Loader ICD interface version Prism negotiates. v5 is the modern interface:
/// the loader calls vk_icdGetInstanceProcAddr for everything and expects the ICD
/// to stamp dispatchable handles with the loader magic.
pub const loader_icd_interface_version: u32 = 5;

/// Prism's chosen Vulkan vendor id. No PCI vendor exists for a software stack,
/// so we use a private non-PCI id. 0x10005 is arbitrary, reserved for "Prism".
/// Stable so tools that key off vendorID behave consistently.
pub const PRISM_VENDOR_ID: u32 = 0x10005;

/// driverVersion: pack Prism's semantic version into the Vulkan 22.10 layout used
/// for driverVersion (major.minor.patch). Prism 0.1.0 -> here as 0,1,0.
pub const PRISM_DRIVER_VERSION: u32 = vk.VK_MAKE_API_VERSION(0, 0, 1, 0);

// The ICD makes only a handful of small, long-lived allocations (the Instance and
// its physical-device list). page_allocator needs no libc so the .so stays
// libc-free. Over-allocation is negligible at this volume.
const allocator = std.heap.page_allocator;

/// Env-gated stderr trace (PRISM_VK_TRACE set): a printf to fd 2 via libc so the .so
/// needs no std fs/io layer. Used to instrument the texture path while bringing up
/// vkcube. A no-op (one getenv) when the env var is unset.
fn vtrace(comptime fmt: []const u8, args: anytype) void {
    if (std.c.getenv("PRISM_VK_TRACE") == null) return;
    var buf: [256]u8 = undefined;
    const s = std.fmt.bufPrint(&buf, "[prism-vk] " ++ fmt ++ "\n", args) catch return;
    _ = std.c.write(2, s.ptr, s.len);
}

/// One enumerated Prism physical device: a thin record over a compiled-in driver.
/// The first field is the loader magic (uintptr_t) so the loader accepts the
/// pointer as a dispatchable handle (see vk_icd.h set_loader_magic_value).
pub const PhysicalDevice = extern struct {
    loader_magic: usize,
    /// Index into prism.drivers.all for the backing driver.
    driver_index: u32,
};

/// A Prism VkInstance: holds the enumerated physical devices. First field is the
/// loader magic for the same reason as PhysicalDevice.
pub const Instance = struct {
    loader_magic: usize,
    phys: []PhysicalDevice,

    fn toHandle(self: *Instance) vk.VkInstance {
        return @ptrCast(self);
    }
    fn fromHandle(h: vk.VkInstance) *Instance {
        return @ptrCast(@alignCast(h.?));
    }
};

fn physFromHandle(h: vk.VkPhysicalDevice) *PhysicalDevice {
    return @ptrCast(@alignCast(h.?));
}

fn physToHandle(p: *PhysicalDevice) vk.VkPhysicalDevice {
    return @ptrCast(p);
}

/// Copy a Zig string into a fixed C char buffer, NUL-terminated + zero-padded.
fn copyCStr(dst: []u8, src: []const u8) void {
    @memset(dst, 0);
    const n = @min(dst.len - 1, src.len);
    @memcpy(dst[0..n], src[0..n]);
}

/// Build the list of physical devices: one per available compiled-in driver.
fn enumerateAvailable() ![]PhysicalDevice {
    var count: usize = 0;
    for (prism.drivers.all) |d| {
        if (d.isAvailable()) count += 1;
    }
    const list = try allocator.alloc(PhysicalDevice, count);
    var i: usize = 0;
    for (prism.drivers.all, 0..) |d, idx| {
        if (!d.isAvailable()) continue;
        list[i] = .{ .loader_magic = vk.ICD_LOADER_MAGIC, .driver_index = @intCast(idx) };
        i += 1;
    }
    return list;
}

/// Whether a driver name denotes real GPU hardware (DISCRETE_GPU) vs the software
/// rasterizer (CPU). Driven by drm_driver presence: only the software driver has
/// no DRM device association.
fn isHardwareDriver(d: prism.Driver) bool {
    return d.drm_driver != null;
}

// Implemented Vulkan entry points. Each is callconv(.c) and matches the Vulkan
// prototype, reached via vk_icdGetInstanceProcAddr (the loader's only required
// hook in v5) and the thin re-exports in lib/prism-vk.zig.

/// One instance extension entry: name + spec version.
const InstExt = struct { name: []const u8, ver: u32 };
const instance_extensions = [_]InstExt{
    .{ .name = "VK_KHR_surface", .ver = 25 },
    .{ .name = "VK_KHR_wayland_surface", .ver = 6 },
};

pub fn enumerateInstanceExtensionProperties(
    pLayerName: ?[*:0]const u8,
    pCount: *u32,
    pProperties: ?[*]vk.VkExtensionProperties,
) callconv(.c) vk.VkResult {
    _ = pLayerName;
    const n: u32 = instance_extensions.len;
    if (pProperties == null) {
        pCount.* = n;
        return .VK_SUCCESS;
    }
    const write = @min(pCount.*, n);
    var i: u32 = 0;
    while (i < write) : (i += 1) {
        var e = std.mem.zeroes(vk.VkExtensionProperties);
        copyCStr(&e.extensionName, instance_extensions[i].name);
        e.specVersion = instance_extensions[i].ver;
        pProperties.?[i] = e;
    }
    pCount.* = write;
    return if (write < n) .VK_INCOMPLETE else .VK_SUCCESS;
}

pub fn enumerateInstanceLayerProperties(
    pCount: *u32,
    pProperties: ?[*]vk.VkLayerProperties,
) callconv(.c) vk.VkResult {
    _ = pProperties;
    // ICDs do not expose layers; the loader owns layers.
    pCount.* = 0;
    return .VK_SUCCESS;
}

pub fn createInstance(
    pCreateInfo: ?*const vk.VkInstanceCreateInfo,
    pAllocator: ?*const vk.VkAllocationCallbacks,
    pInstance: *vk.VkInstance,
) callconv(.c) vk.VkResult {
    _ = pCreateInfo;
    _ = pAllocator;
    const inst = allocator.create(Instance) catch return .VK_ERROR_OUT_OF_HOST_MEMORY;
    const phys = enumerateAvailable() catch {
        allocator.destroy(inst);
        return .VK_ERROR_OUT_OF_HOST_MEMORY;
    };
    inst.* = .{ .loader_magic = vk.ICD_LOADER_MAGIC, .phys = phys };
    pInstance.* = inst.toHandle();
    return .VK_SUCCESS;
}

pub fn destroyInstance(
    instance: vk.VkInstance,
    pAllocator: ?*const vk.VkAllocationCallbacks,
) callconv(.c) void {
    _ = pAllocator;
    if (instance == null) return;
    const inst = Instance.fromHandle(instance);
    allocator.free(inst.phys);
    allocator.destroy(inst);
}

pub fn enumeratePhysicalDevices(
    instance: vk.VkInstance,
    pPhysicalDeviceCount: *u32,
    pPhysicalDevices: ?*vk.VkPhysicalDevice,
) callconv(.c) vk.VkResult {
    const inst = Instance.fromHandle(instance);
    const n: u32 = @intCast(inst.phys.len);
    if (pPhysicalDevices == null) {
        pPhysicalDeviceCount.* = n;
        return .VK_SUCCESS;
    }
    const out: [*]vk.VkPhysicalDevice = @ptrCast(pPhysicalDevices.?);
    const cap = pPhysicalDeviceCount.*;
    const write = @min(cap, n);
    var i: u32 = 0;
    while (i < write) : (i += 1) {
        out[i] = physToHandle(&inst.phys[i]);
    }
    pPhysicalDeviceCount.* = write;
    return if (write < n) .VK_INCOMPLETE else .VK_SUCCESS;
}

pub fn getPhysicalDeviceProperties(
    physicalDevice: vk.VkPhysicalDevice,
    pProperties: *vk.VkPhysicalDeviceProperties,
) callconv(.c) void {
    const pd = physFromHandle(physicalDevice);
    const d = prism.drivers.all[pd.driver_index];

    pProperties.* = std.mem.zeroes(vk.VkPhysicalDeviceProperties);
    pProperties.apiVersion = vk.VK_API_VERSION_1_3;
    pProperties.driverVersion = PRISM_DRIVER_VERSION;
    pProperties.vendorID = PRISM_VENDOR_ID;
    // A stable per-driver device id derived from the driver index.
    pProperties.deviceID = 0x9000 + pd.driver_index;
    pProperties.deviceType = if (isHardwareDriver(d))
        .VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU
    else
        .VK_PHYSICAL_DEVICE_TYPE_CPU;

    // deviceName: prefix "Prism " + the driver's name. We use the driver name
    // (not caps().device_name) to avoid bringing a full HAL device up just to
    // enumerate, which would fail/SKIP on boxes without that GPU.
    var namebuf: [vk.VK_MAX_PHYSICAL_DEVICE_NAME_SIZE]u8 = undefined;
    const name = std.fmt.bufPrint(&namebuf, "Prism ({s})", .{d.name}) catch d.name;
    copyCStr(&pProperties.deviceName, name);

    // A fixed pipeline-cache UUID tagged "PRISM" so it is recognizable + stable.
    pProperties.pipelineCacheUUID = .{ 'P', 'R', 'I', 'S', 'M', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 };
    // limits + sparseProperties left zero-filled (already zeroed above), except
    // maxPushConstantsSize: spec minimum is 128 bytes, we support PUSH_CONSTANT_SIZE
    // (256), so an app querying the limit sees a usable value. A zero would make
    // any push fail validation. The command-buffer push-constant block backs this size.
    pProperties.limits.maxPushConstantsSize = PUSH_CONSTANT_SIZE;
    // MSAA (breadth 4): advertise 1x/2x/4x for color + depth framebuffer attachments
    // (the software rasterizer evaluates coverage at the standard sample positions for
    // these counts and resolves). 8x+ is not implemented, so it is not advertised.
    const msaa_counts: vk.VkSampleCountFlags =
        vk.VK_SAMPLE_COUNT_1_BIT | vk.VK_SAMPLE_COUNT_2_BIT | vk.VK_SAMPLE_COUNT_4_BIT;
    pProperties.limits.framebufferColorSampleCounts = msaa_counts;
    pProperties.limits.framebufferDepthSampleCounts = msaa_counts;
    pProperties.limits.framebufferStencilSampleCounts = vk.VK_SAMPLE_COUNT_1_BIT;
    pProperties.limits.framebufferNoAttachmentsSampleCounts = msaa_counts;
    pProperties.limits.sampledImageColorSampleCounts = msaa_counts;
    pProperties.limits.sampledImageDepthSampleCounts = msaa_counts;
}

pub fn getPhysicalDeviceFeatures(
    physicalDevice: vk.VkPhysicalDevice,
    pFeatures: *vk.VkPhysicalDeviceFeatures,
) callconv(.c) void {
    _ = physicalDevice;
    pFeatures.* = std.mem.zeroes(vk.VkPhysicalDeviceFeatures);
}

pub fn getPhysicalDeviceQueueFamilyProperties(
    physicalDevice: vk.VkPhysicalDevice,
    pQueueFamilyPropertyCount: *u32,
    pQueueFamilyProperties: ?[*]vk.VkQueueFamilyProperties,
) callconv(.c) void {
    _ = physicalDevice;
    // One queue family: graphics + compute + transfer.
    if (pQueueFamilyProperties == null) {
        pQueueFamilyPropertyCount.* = 1;
        return;
    }
    if (pQueueFamilyPropertyCount.* < 1) {
        pQueueFamilyPropertyCount.* = 0;
        return;
    }
    pQueueFamilyProperties.?[0] = .{
        .queueFlags = vk.VK_QUEUE_GRAPHICS_BIT | vk.VK_QUEUE_COMPUTE_BIT | vk.VK_QUEUE_TRANSFER_BIT,
        .queueCount = 1,
        .timestampValidBits = 0,
        .minImageTransferGranularity = .{ .width = 1, .height = 1, .depth = 1 },
    };
    pQueueFamilyPropertyCount.* = 1;
}

pub fn getPhysicalDeviceMemoryProperties(
    physicalDevice: vk.VkPhysicalDevice,
    pMemoryProperties: *vk.VkPhysicalDeviceMemoryProperties,
) callconv(.c) void {
    _ = physicalDevice;
    pMemoryProperties.* = std.mem.zeroes(vk.VkPhysicalDeviceMemoryProperties);
    pMemoryProperties.memoryHeapCount = 1;
    pMemoryProperties.memoryHeaps[0] = .{
        .size = 256 * 1024 * 1024, // 256 MiB placeholder heap.
        .flags = vk.VK_MEMORY_HEAP_DEVICE_LOCAL_BIT,
    };
    pMemoryProperties.memoryTypeCount = 1;
    pMemoryProperties.memoryTypes[0] = .{
        .propertyFlags = vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
            vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
            vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        .heapIndex = 0,
    };
}

pub fn getPhysicalDeviceFormatProperties(
    physicalDevice: vk.VkPhysicalDevice,
    format: vk.VkFormat,
    pFormatProperties: *vk.VkFormatProperties,
) callconv(.c) void {
    _ = physicalDevice;
    var props = std.mem.zeroes(vk.VkFormatProperties);
    // Report the features the software path actually honors: a depth/stencil format
    // is a valid depth/stencil attachment. The color/vertex formats the render path
    // uses are color attachments / sampled. (Conservative but truthful.)
    if (vk.formatIsDepth(format)) {
        props.optimalTilingFeatures |= vk.VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT;
    } else switch (format) {
        // Color/sampleable formats: a color attachment that can also be sampled and is a
        // valid transfer destination (so a texture is filled via vkCmdCopyBufferToImage).
        // R8G8B8A8_SRGB is vkcube's texture format. It must be reported sampleable +
        // transfer-dst, or vkcube creates no texture image (its descriptor arrives with
        // a null image view, producing a black cube). These are advertised only in
        // optimalTilingFeatures, not linearTilingFeatures, so vkcube takes its staging
        // path (staging buffer -> vkCmdCopyBufferToImage), which the software path supports.
        vk.VK_FORMAT_R8G8B8A8_UNORM,
        vk.VK_FORMAT_R8G8B8A8_SRGB,
        vk.VK_FORMAT_B8G8R8A8_UNORM,
        vk.VK_FORMAT_B8G8R8A8_SRGB,
        // Float render-target + sampled formats (breadth 4): R16F/R32F + the half-vec
        // variants are valid color attachments and sampleable (a value renders into them
        // and reads back within precision). R8/RG8 are sampleable single/dual-channel.
        vk.VK_FORMAT_R16_SFLOAT,
        vk.VK_FORMAT_R32_SFLOAT,
        vk.VK_FORMAT_R16G16_SFLOAT,
        vk.VK_FORMAT_R16G16B16A16_SFLOAT,
        vk.VK_FORMAT_R32G32B32A32_SFLOAT,
        => {
            props.optimalTilingFeatures |= vk.VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT |
                vk.VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT |
                vk.VK_FORMAT_FEATURE_TRANSFER_DST_BIT;
        },
        // R8/RG8 unorm: sampleable textures (+ transfer dst). Not advertised as color
        // attachments (the common use is an uploaded single/dual-channel texture).
        vk.VK_FORMAT_R8_UNORM,
        vk.VK_FORMAT_R8G8_UNORM,
        => {
            props.optimalTilingFeatures |= vk.VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT |
                vk.VK_FORMAT_FEATURE_TRANSFER_DST_BIT |
                vk.VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT;
        },
        else => {},
    }
    pFormatProperties.* = props;
}

pub fn getPhysicalDeviceImageFormatProperties(
    physicalDevice: vk.VkPhysicalDevice,
    format: vk.VkFormat,
    image_type: vk.VkImageType,
    tiling: vk.VkImageTiling,
    usage: vk.VkImageUsageFlags,
    flags: vk.VkImageCreateFlags,
    pImageFormatProperties: *vk.VkImageFormatProperties,
) callconv(.c) vk.VkResult {
    _ = physicalDevice;
    _ = format;
    _ = image_type;
    _ = tiling;
    _ = usage;
    _ = flags;
    pImageFormatProperties.* = std.mem.zeroes(vk.VkImageFormatProperties);
    return .VK_ERROR_FORMAT_NOT_SUPPORTED;
}

pub fn getPhysicalDeviceSparseImageFormatProperties(
    physicalDevice: vk.VkPhysicalDevice,
    format: vk.VkFormat,
    image_type: vk.VkImageType,
    samples: vk.VkSampleCountFlagBits,
    usage: vk.VkImageUsageFlags,
    tiling: vk.VkImageTiling,
    pPropertyCount: *u32,
    pProperties: ?[*]vk.VkSparseImageFormatProperties,
) callconv(.c) void {
    _ = physicalDevice;
    _ = format;
    _ = image_type;
    _ = samples;
    _ = usage;
    _ = tiling;
    _ = pProperties;
    pPropertyCount.* = 0;
}

/// A real logical device. Its first field is the loader magic (dispatchable
/// handle). It owns a brought-up Prism HAL `hal.Device` (the backing driver's
/// createDevice), through which all Vulkan memory/buffer work is serviced. The
/// embedded `queue` is the single device queue we expose (family 0, index 0).
pub const LogicalDevice = extern struct {
    loader_magic: usize,
    queue: Queue,
    /// The Prism HAL device fat-pointer, split into its two words so the struct
    /// stays `extern` (Vulkan dispatchable handles must be extern with the magic
    /// first). Reassembled via `hal()`.
    hal_ptr: *anyopaque,
    hal_vtable: *const prism.hal.Device.VTable,

    fn hal(self: *LogicalDevice) prism.hal.Device {
        return .{ .ptr = self.hal_ptr, .vtable = self.hal_vtable };
    }
    fn fromHandle(h: vk.VkDevice) *LogicalDevice {
        return @ptrCast(@alignCast(h.?));
    }
};

/// A device queue. Also a dispatchable handle, so loader-magic first.
pub const Queue = extern struct {
    loader_magic: usize,
};

/// A Vulkan VkDeviceMemory: a Prism HAL Resource. vkAllocateMemory allocates a
/// HAL buffer resource through the device. The mapped pointer for vkMapMemory is
/// the HAL `mapResource` of this resource. Heap-allocated. u64 handle is
/// @intFromPtr of this struct.
pub const DeviceMemory = struct {
    resource: *prism.hal.Resource,
    size: u64,

    fn toHandle(self: *DeviceMemory) vk.VkDeviceMemory {
        return @intCast(@intFromPtr(self));
    }
    fn fromHandle(h: vk.VkDeviceMemory) *DeviceMemory {
        return @ptrFromInt(@as(usize, @intCast(h)));
    }
};

/// A Vulkan VkBuffer: a size + (after vkBindBufferMemory) the bound memory and
/// offset. Heap-allocated. The u64 handle is @intFromPtr of this struct.
pub const Buffer = struct {
    size: u64,
    memory: ?*DeviceMemory = null,
    offset: u64 = 0,

    fn toHandle(self: *Buffer) vk.VkBuffer {
        return @intCast(@intFromPtr(self));
    }
    fn fromHandle(h: vk.VkBuffer) *Buffer {
        return @ptrFromInt(@as(usize, @intCast(h)));
    }
};

/// Round `n` up to a multiple of `a` (a must be a power of two).
fn roundUp(n: u64, a: u64) u64 {
    return (n + a - 1) & ~(a - 1);
}

/// Buffer/memory alignment Prism advertises for VkMemoryRequirements.
pub const PRISM_BUFFER_ALIGNMENT: u64 = 256;

/// Real device creation: bring up the Prism HAL device for the selected physical
/// device's driver (the software driver works on this host) and stash it on a
/// heap LogicalDevice. All later memory/buffer calls run through this HAL device.
pub fn createDevice(
    physicalDevice: vk.VkPhysicalDevice,
    pCreateInfo: ?*const vk.VkDeviceCreateInfo,
    pAllocator: ?*const vk.VkAllocationCallbacks,
    pDevice: *vk.VkDevice,
) callconv(.c) vk.VkResult {
    _ = pCreateInfo;
    _ = pAllocator;
    const pd = physFromHandle(physicalDevice);
    const d = prism.drivers.all[pd.driver_index];
    // Bring up the real HAL device for this driver. The ICD .so is libc-free, so
    // the HAL device's backing allocator is page_allocator (same as our own).
    const hal_dev = d.createDevice(allocator) catch return .VK_ERROR_INITIALIZATION_FAILED;
    const dev = allocator.create(LogicalDevice) catch {
        hal_dev.deinit();
        return .VK_ERROR_OUT_OF_HOST_MEMORY;
    };
    dev.* = .{
        .loader_magic = vk.ICD_LOADER_MAGIC,
        .queue = .{ .loader_magic = vk.ICD_LOADER_MAGIC },
        .hal_ptr = hal_dev.ptr,
        .hal_vtable = hal_dev.vtable,
    };
    pDevice.* = @ptrCast(dev);
    return .VK_SUCCESS;
}

pub fn destroyDevice(
    device: vk.VkDevice,
    pAllocator: ?*const vk.VkAllocationCallbacks,
) callconv(.c) void {
    _ = pAllocator;
    if (device == null) return;
    const dev = LogicalDevice.fromHandle(device);
    dev.hal().deinit();
    allocator.destroy(dev);
}

pub fn getDeviceQueue(
    device: vk.VkDevice,
    queueFamilyIndex: u32,
    queueIndex: u32,
    pQueue: *vk.VkQueue,
) callconv(.c) void {
    _ = queueFamilyIndex;
    _ = queueIndex;
    const dev = LogicalDevice.fromHandle(device);
    pQueue.* = @ptrCast(&dev.queue);
}

// Buffer + memory entry points.

pub fn createBuffer(
    device: vk.VkDevice,
    pCreateInfo: ?*const vk.VkBufferCreateInfo,
    pAllocator: ?*const vk.VkAllocationCallbacks,
    pBuffer: *vk.VkBuffer,
) callconv(.c) vk.VkResult {
    _ = device;
    _ = pAllocator;
    const ci = pCreateInfo orelse return .VK_ERROR_INITIALIZATION_FAILED;
    const buf = allocator.create(Buffer) catch return .VK_ERROR_OUT_OF_HOST_MEMORY;
    buf.* = .{ .size = ci.size };
    pBuffer.* = buf.toHandle();
    return .VK_SUCCESS;
}

pub fn destroyBuffer(
    device: vk.VkDevice,
    buffer: vk.VkBuffer,
    pAllocator: ?*const vk.VkAllocationCallbacks,
) callconv(.c) void {
    _ = device;
    _ = pAllocator;
    if (buffer == vk.VK_NULL_HANDLE) return;
    allocator.destroy(Buffer.fromHandle(buffer));
}

pub fn getBufferMemoryRequirements(
    device: vk.VkDevice,
    buffer: vk.VkBuffer,
    pMemoryRequirements: *vk.VkMemoryRequirements,
) callconv(.c) void {
    _ = device;
    const buf = Buffer.fromHandle(buffer);
    pMemoryRequirements.* = .{
        .size = roundUp(buf.size, PRISM_BUFFER_ALIGNMENT),
        .alignment = PRISM_BUFFER_ALIGNMENT,
        // Only memory type 0 exists (see getPhysicalDeviceMemoryProperties).
        .memoryTypeBits = 0b1,
    };
}

pub fn allocateMemory(
    device: vk.VkDevice,
    pAllocateInfo: ?*const vk.VkMemoryAllocateInfo,
    pAllocator: ?*const vk.VkAllocationCallbacks,
    pMemory: *vk.VkDeviceMemory,
) callconv(.c) vk.VkResult {
    _ = pAllocator;
    const ai = pAllocateInfo orelse return .VK_ERROR_INITIALIZATION_FAILED;
    const dev = LogicalDevice.fromHandle(device);
    // Vulkan device memory is a Prism HAL buffer resource. Allocate it through the
    // HAL. The mapped pointer for vkMapMemory is its mapResource.
    const resource = dev.hal().createResource(.{ .buffer = .{
        .size = @intCast(ai.allocationSize),
        .usage = .{ .copy_dst = true, .copy_src = true },
    } }) catch return .VK_ERROR_OUT_OF_DEVICE_MEMORY;
    const mem = allocator.create(DeviceMemory) catch {
        dev.hal().destroyResource(resource);
        return .VK_ERROR_OUT_OF_HOST_MEMORY;
    };
    mem.* = .{ .resource = resource, .size = ai.allocationSize };
    pMemory.* = mem.toHandle();
    return .VK_SUCCESS;
}

pub fn freeMemory(
    device: vk.VkDevice,
    memory: vk.VkDeviceMemory,
    pAllocator: ?*const vk.VkAllocationCallbacks,
) callconv(.c) void {
    _ = pAllocator;
    if (memory == vk.VK_NULL_HANDLE) return;
    const dev = LogicalDevice.fromHandle(device);
    const mem = DeviceMemory.fromHandle(memory);
    dev.hal().destroyResource(mem.resource);
    allocator.destroy(mem);
}

pub fn bindBufferMemory(
    device: vk.VkDevice,
    buffer: vk.VkBuffer,
    memory: vk.VkDeviceMemory,
    memoryOffset: vk.VkDeviceSize,
) callconv(.c) vk.VkResult {
    _ = device;
    const buf = Buffer.fromHandle(buffer);
    buf.memory = DeviceMemory.fromHandle(memory);
    buf.offset = memoryOffset;
    return .VK_SUCCESS;
}

pub fn mapMemory(
    device: vk.VkDevice,
    memory: vk.VkDeviceMemory,
    offset: vk.VkDeviceSize,
    size: vk.VkDeviceSize,
    flags: vk.VkMemoryMapFlags,
    ppData: *?*anyopaque,
) callconv(.c) vk.VkResult {
    _ = size;
    _ = flags;
    const dev = LogicalDevice.fromHandle(device);
    const mem = DeviceMemory.fromHandle(memory);
    const mapped = dev.hal().mapResource(mem.resource) catch return .VK_ERROR_MEMORY_MAP_FAILED;
    ppData.* = @ptrFromInt(@intFromPtr(mapped.ptr) + @as(usize, @intCast(offset)));
    return .VK_SUCCESS;
}

pub fn unmapMemory(
    device: vk.VkDevice,
    memory: vk.VkDeviceMemory,
) callconv(.c) void {
    // The HAL holds the map for the resource's lifetime (no per-call unmap), so
    // there is nothing to undo here.
    _ = device;
    _ = memory;
}

pub fn flushMappedMemoryRanges(
    device: vk.VkDevice,
    memoryRangeCount: u32,
    pMemoryRanges: ?[*]const vk.VkMappedMemoryRange,
) callconv(.c) vk.VkResult {
    // Memory type 0 is HOST_COHERENT, so flush/invalidate are no-ops.
    _ = device;
    _ = memoryRangeCount;
    _ = pMemoryRanges;
    return .VK_SUCCESS;
}

pub fn invalidateMappedMemoryRanges(
    device: vk.VkDevice,
    memoryRangeCount: u32,
    pMemoryRanges: ?[*]const vk.VkMappedMemoryRange,
) callconv(.c) vk.VkResult {
    _ = device;
    _ = memoryRangeCount;
    _ = pMemoryRanges;
    return .VK_SUCCESS;
}

// Compute pipeline + descriptor + command objects (M3 compute path).
// Non-dispatchable handles encode @intFromPtr of a heap struct (LP64, the loader
// passes them through). VkCommandBuffer is dispatchable (loader-magic first).
// vkQueueSubmit replays the recorded dispatch by calling the backing HAL device's
// dispatchCompute over the bound buffers. The software driver lowers SPIR-V
// (parseSpirv), JITs it (spirv_jit), and runs the kernel into the HAL-backed
// storage buffers from M2.

/// A VkShaderModule: the raw SPIR-V bytes (copied, so the app may free pCode).
pub const ShaderModule = struct {
    spirv: []u8,
    fn toHandle(self: *ShaderModule) vk.VkShaderModule {
        return @intCast(@intFromPtr(self));
    }
    fn fromHandle(h: vk.VkShaderModule) *ShaderModule {
        return @ptrFromInt(@as(usize, @intCast(h)));
    }
};

/// A VkDescriptorSetLayout: the count of storage-buffer bindings (we only model
/// storage buffers). Binding indices are assumed 0..count-1 (the M3 contract).
pub const DescriptorSetLayout = struct {
    binding_count: u32,
    fn toHandle(self: *DescriptorSetLayout) vk.VkDescriptorSetLayout {
        return @intCast(@intFromPtr(self));
    }
    fn fromHandle(h: vk.VkDescriptorSetLayout) *DescriptorSetLayout {
        return @ptrFromInt(@as(usize, @intCast(h)));
    }
};

/// The push-constant block size the ICD backs (spec minimum is 128; we support 256).
/// vkCmdPushConstants writes into a per-command-buffer block of this size.
/// Pushed bytes reach the shader as a UBO-like pointer param at draw time.
pub const PUSH_CONSTANT_SIZE = 256;

/// A VkPipelineLayout: tracks the (single) descriptor set layout's binding count and
/// whether any push-constant range is declared (and the high-water byte the ranges
/// cover). The push-constant block reaches the shader as a UBO bound at the binding
/// index right after the descriptor bindings (binding_count), which matches where the
/// PushConstant-storage variable lands in Vulcan's lowered parameter order (it carries
/// no Binding decoration, so spirv.zig's binding sort places it after the descriptors).
pub const PipelineLayout = struct {
    binding_count: u32,
    push_constant_size: u32 = 0, // 0 = no push constants declared
    fn toHandle(self: *PipelineLayout) vk.VkPipelineLayout {
        return @intCast(@intFromPtr(self));
    }
    fn fromHandle(h: vk.VkPipelineLayout) *PipelineLayout {
        return @ptrFromInt(@as(usize, @intCast(h)));
    }
};

/// vkDestroyPipeline takes either a compute or a graphics VkPipeline behind the
/// shared u64 handle. Zig may reorder non-extern struct fields, so we can't read a
/// "kind" field at a fixed offset from a raw pointer. The kind is tagged in the
/// handle's low bit (heap pointers are >=8-aligned, so bit 0 is free).
/// A graphics pipeline handle has bit 0 set. fromHandle masks it off.
const GRAPHICS_PIPELINE_TAG: u64 = 1;

/// A compute VkPipeline: the HAL shader module (built from the stage's SPIR-V) and
/// its binding count. The HAL shader module is created once at pipeline creation.
pub const ComputePipeline = struct {
    hal_shader: *prism.hal.ShaderModule,
    binding_count: u32,
    fn toHandle(self: *ComputePipeline) vk.VkPipeline {
        return @intCast(@intFromPtr(self));
    }
    fn fromHandle(h: vk.VkPipeline) *ComputePipeline {
        return @ptrFromInt(@as(usize, @intCast(h)));
    }
};

/// The maximum storage buffers a descriptor set binds (matches the HAL dispatch cap).
pub const MAX_BINDINGS = 8;

/// A VkSampler: the filter + per-axis address modes (the software sampler reads them).
pub const Sampler = struct {
    filter: prism.hal.Filter = .nearest,
    address_u: prism.hal.AddressMode = .repeat,
    address_v: prism.hal.AddressMode = .repeat,
    max_anisotropy: f32 = 1,
    fn toHandle(self: *Sampler) vk.VkSampler {
        return @intCast(@intFromPtr(self));
    }
    fn fromHandle(h: vk.VkSampler) *Sampler {
        return @ptrFromInt(@as(usize, @intCast(h)));
    }
};

/// The kind of resource a descriptor binding holds: which the draw/dispatch path
/// uses to route storage buffers (compute) vs uniform buffers (graphics UBOs) vs a
/// combined-image-sampler (graphics texture).
pub const BindingKind = enum { none, storage_buffer, uniform_buffer, combined_image_sampler };

/// A VkDescriptorSet: the resources bound per binding index (filled by
/// vkUpdateDescriptorSets). buffers[i] is the VkBuffer at binding i, or null.
/// kinds[i] records whether it was bound as a storage/uniform buffer or a
/// combined-image-sampler. images[i]/samplers[i] hold the texture binding.
pub const DescriptorSet = struct {
    buffers: [MAX_BINDINGS]?*Buffer = .{null} ** MAX_BINDINGS,
    kinds: [MAX_BINDINGS]BindingKind = .{.none} ** MAX_BINDINGS,
    images: [MAX_BINDINGS]?*Image = .{null} ** MAX_BINDINGS,
    samplers: [MAX_BINDINGS]?*Sampler = .{null} ** MAX_BINDINGS,
    fn toHandle(self: *DescriptorSet) vk.VkDescriptorSet {
        return @intCast(@intFromPtr(self));
    }
    fn fromHandle(h: vk.VkDescriptorSet) *DescriptorSet {
        return @ptrFromInt(@as(usize, @intCast(h)));
    }
};

/// A VkDescriptorPool: owns the descriptor sets it allocates (freed on destroy).
pub const DescriptorPool = struct {
    sets: std.ArrayListUnmanaged(*DescriptorSet) = .empty,
    fn toHandle(self: *DescriptorPool) vk.VkDescriptorPool {
        return @intCast(@intFromPtr(self));
    }
    fn fromHandle(h: vk.VkDescriptorPool) *DescriptorPool {
        return @ptrFromInt(@as(usize, @intCast(h)));
    }
};

/// A VkCommandPool: owns its allocated command buffers (freed on destroy).
pub const CommandPool = struct {
    buffers: std.ArrayListUnmanaged(*CommandBuffer) = .empty,
    fn toHandle(self: *CommandPool) vk.VkCommandPool {
        return @intCast(@intFromPtr(self));
    }
    fn fromHandle(h: vk.VkCommandPool) *CommandPool {
        return @ptrFromInt(@as(usize, @intCast(h)));
    }
};

/// A recorded compute dispatch OR offscreen render pass on a VkCommandBuffer.
/// vkQueueSubmit executes whichever was recorded. Dispatchable handle, so
/// loader-magic first.
///
/// The graphics (M4) recording is flat single-shot state: one render pass (a
/// framebuffer + clear color), one bound graphics pipeline + vertex buffer, one
/// draw, and one image->buffer copy. That is exactly what the offscreen-triangle
/// test records, so a fixed set of fields (rather than a command list) is the
/// simplest faithful model.
/// One recorded draw within a render pass: its bound vertex buffer + graphics
/// pipeline + descriptor set + the draw range. Multiple of these run in order
/// against the same color + depth attachment (so depth testing across draws works).
pub const GfxDraw = extern struct {
    // Null for a vertex-pulling draw (zero vertex attributes; the VS pulls vertices
    // from a bound UBO by gl_VertexIndex). Non-null for an attribute-fed draw.
    vertex_buffer: ?*Buffer,
    pipeline: *GraphicsPipeline,
    descriptor_set: ?*DescriptorSet,
    vertex_count: u32,
    first_vertex: u32,
    // Instancing: render the draw `instance_count` times, the VS seeing gl_InstanceIndex =
    // first_instance + the instance. 1/0 for a non-instanced draw.
    instance_count: u32 = 1,
    first_instance: u32 = 0,
    // Snapshot of the command buffer's push-constant block at record time.
    // Push constants are draw-time state: a later vkCmdPushConstants does not
    // change an earlier draw, so each draw copies the current bytes here.
    // has_push_constants tells executeRender to bind these bytes (as a UBO at the
    // pipeline's pc_binding slot) before replaying this draw.
    push_constants: [PUSH_CONSTANT_SIZE]u8,
    has_push_constants: bool,
    // A snapshot of the command buffer's current scissor (vkCmdSetScissor) at this draw.
    // has_scissor false -> no clipping (full render target). Top-left origin, HAL-native.
    scissor: vk.VkRect2D,
    has_scissor: bool,
    // Snapshot of the command buffer's dynamic stencil ref/compare/write masks (front + back)
    // at this draw, when the pipeline declares them dynamic. executeRender rebuilds a per-draw
    // HAL pipeline variant with these live values overriding the pipeline's baked stencil.
    dyn_stencil: bool = false,
    st_ref_front: u8 = 0,
    st_ref_back: u8 = 0,
    st_cmp_front: u8 = 0xff,
    st_cmp_back: u8 = 0xff,
    st_wrt_front: u8 = 0xff,
    st_wrt_back: u8 = 0xff,
    // Indexed draw (vkCmdDrawIndexed): vertex_count holds indexCount, first_vertex holds firstIndex.
    // executeRender gathers vertex_buffer[index[i]+vertex_offset] into a temp stream, then draws it
    // non-indexed. index_buffer null / is_indexed false -> the ordinary non-indexed path.
    is_indexed: bool = false,
    index_buffer: ?*Buffer = null,
    index_offset: u64 = 0,
    index_u32: bool = false,
    vertex_offset: i32 = 0,
};

/// The maximum number of draws a single command buffer records, summed across all
/// render-pass instances (a small fixed cap that covers the overlapping-triangle depth
/// proof + typical multi-mesh / multi-pass frame graphs).
pub const MAX_GFX_DRAWS = 64;

/// Max vkCmdCopyBuffer regions recorded in one command buffer (staging uploads).
pub const MAX_BUF_COPIES = 32;

/// One recorded buffer-to-buffer copy region (vkCmdCopyBuffer).
pub const BufCopy = extern struct {
    src: *Buffer,
    dst: *Buffer,
    src_offset: u64,
    dst_offset: u64,
    size: u64,
};

/// The maximum number of render-pass instances (vkCmdBeginRenderPass..EndRenderPass
/// blocks) a single command buffer records. Real frame graphs (a shadow-map pass + a
/// scene pass, or an offscreen scene + a post-process pass) use a handful. This caps it.
pub const MAX_RENDER_PASSES = 8;

/// One render-pass instance: the framebuffer it renders into (an offscreen sampled image
/// or the swapchain image), its color + depth clear values, and the slice of the command
/// buffer's shared `draws` array that belongs to it ([draw_start, draw_start+draw_count)).
/// executeRender replays each instance in order, clearing then drawing into its framebuffer
/// image, so pass 1 can render into a texture that pass 2 then samples.
pub const RenderPassInstance = extern struct {
    framebuffer: ?*Framebuffer,
    clear_r: f32,
    clear_g: f32,
    clear_b: f32,
    clear_a: f32,
    depth_image: ?*Image,
    depth_clear: f32,
    has_depth: bool,
    /// The stencil clear value from pClearValues[depth_index].depthStencil.stencil (the
    /// depth/stencil attachment is one image; its stencil component clears here).
    stencil_clear: u8 = 0,
    draw_start: usize,
    draw_count: usize,
    /// MSAA: the single-sample image the multisampled color attachment resolves into at
    /// EndRenderPass (null when no resolve attachment). `samples` is the MSAA count.
    resolve_image: ?*Image = null,
    samples: u8 = 1,
};

pub const CommandBuffer = extern struct {
    loader_magic: usize,
    // The owning logical device (set at allocate). Needed by record-time HAL work
    // like vkCmdCopyBufferToImage (which fills a texture before the draw).
    device: vk.VkDevice = null,
    // Compute (M3).
    pipeline: ?*ComputePipeline = null,
    descriptor_set: ?*DescriptorSet = null,
    group_x: u32 = 0,
    group_y: u32 = 0,
    group_z: u32 = 0,
    has_dispatch: bool = false,
    // Graphics (M4).
    gfx_pipeline: ?*GraphicsPipeline = null,
    framebuffer: ?*Framebuffer = null,
    vertex_buffer: ?*Buffer = null,
    // Bound index buffer (vkCmdBindIndexBuffer) for vkCmdDrawIndexed. index_u32 = UINT32, else UINT16.
    index_buffer: ?*Buffer = null,
    index_offset: u64 = 0,
    index_u32: bool = false,
    clear_r: f32 = 0,
    clear_g: f32 = 0,
    clear_b: f32 = 0,
    clear_a: f32 = 1,
    // Depth attachment (feature 2): the depth image (from the framebuffer's depth
    // view) + the clear value, present only when the render pass has a depth
    // attachment. Null/0 -> the color-only path (no depth test).
    depth_image: ?*Image = null,
    depth_clear: f32 = 1.0,
    has_depth: bool = false,
    draw_vertex_count: u32 = 0,
    draw_first_vertex: u32 = 0,
    has_draw: bool = false,
    // A render pass may issue multiple draws (each with its own bound vertex buffer +
    // pipeline), all into the same color + depth attachment. We record them in order
    // so depth testing across draws works (the nearer surface occludes the farther
    // one regardless of draw order). draw_count == 0 -> the legacy single-draw fields
    // above drive a one-draw render pass (the offscreen test path, unchanged).
    draws: [MAX_GFX_DRAWS]GfxDraw = undefined,
    draw_count: usize = 0,
    // Multiple render-pass instances (one per vkCmdBeginRenderPass..EndRenderPass block).
    // Each instance owns its framebuffer + clear/depth + a slice of `draws`. pass_count==0
    // keeps the legacy single-pass path (the flat framebuffer/draws fields above) intact.
    // That path is used by the offscreen test that records draws without opening an instance.
    // cur_pass is the index of the currently-open instance (-1 = none open).
    passes: [MAX_RENDER_PASSES]RenderPassInstance = undefined,
    pass_count: usize = 0,
    cur_pass: isize = -1,
    // Image copy (vkCmdCopyImageToBuffer): source image + dest buffer.
    copy_src_image: ?*Image = null,
    copy_dst_buffer: ?*Buffer = null,
    has_copy: bool = false,
    // Buffer-to-buffer copies (vkCmdCopyBuffer): the staging-upload pattern (map a host-visible
    // staging buffer, write, then copy into the vertex/index/uniform buffer). Executed at submit.
    buf_copies: [MAX_BUF_COPIES]BufCopy = undefined,
    buf_copy_count: usize = 0,
    // The current push-constant block. vkCmdPushConstants writes into it at the given
    // offset. Each cmdDraw snapshots the whole block (PUSH_CONSTANT_SIZE bytes) so a
    // shader can read any offset. pc_dirty marks that at least one push happened.
    // Reset clears it.
    push_constants: [PUSH_CONSTANT_SIZE]u8 = .{0} ** PUSH_CONSTANT_SIZE,
    pc_dirty: bool = false,
    // The current scissor (vkCmdSetScissor dynamic state), snapshotted into each draw.
    // VkRect2D is framebuffer-pixel top-left origin, matching the HAL convention exactly,
    // so it passes through with no y flip. has_scissor stays false until vkCmdSetScissor
    // is called. Reset clears it.
    scissor: vk.VkRect2D = .{ .offset = .{ .x = 0, .y = 0 }, .extent = .{ .width = 0, .height = 0 } },
    has_scissor: bool = false,
    // Dynamic stencil ref/compare/write masks (vkCmdSetStencilReference/CompareMask/
    // WriteMask), per face. Snapshotted into each draw whose pipeline declares them dynamic.
    // Defaults match the GL/Vulkan reset (ref 0, full masks). vkCmdSetStencil* overrides them.
    st_ref_front: u8 = 0,
    st_ref_back: u8 = 0,
    st_cmp_front: u8 = 0xff,
    st_cmp_back: u8 = 0xff,
    st_wrt_front: u8 = 0xff,
    st_wrt_back: u8 = 0xff,

    fn toHandle(self: *CommandBuffer) vk.VkCommandBuffer {
        return @ptrCast(self);
    }
    fn fromHandle(h: vk.VkCommandBuffer) *CommandBuffer {
        return @ptrCast(@alignCast(h.?));
    }
};

// Graphics object model (M4).

/// Map a VkFormat to the Prism HAL Format used for a render target or vertex
/// attribute. Only the formats the offscreen-triangle path uses are mapped. The
/// default is rgba8_unorm (the render-target format).
fn halFormat(f: vk.VkFormat) prism.hal.Format {
    return switch (f) {
        vk.VK_FORMAT_R8G8B8A8_UNORM, vk.VK_FORMAT_R8G8B8A8_SRGB => .rgba8_unorm,
        vk.VK_FORMAT_R8_UNORM => .r8_unorm,
        vk.VK_FORMAT_R8G8_UNORM => .r8g8_unorm,
        vk.VK_FORMAT_R16_SFLOAT => .r16_float,
        vk.VK_FORMAT_R32_SFLOAT => .r32_float,
        vk.VK_FORMAT_R16G16_SFLOAT => .r16g16_float,
        vk.VK_FORMAT_R16G16B16A16_SFLOAT => .rgba16_float,
        vk.VK_FORMAT_R32G32_SFLOAT => .r32g32_float,
        vk.VK_FORMAT_R32G32B32_SFLOAT => .r32g32b32_float,
        vk.VK_FORMAT_R32G32B32A32_SFLOAT => .r32g32b32a32_float,
        // All depth/stencil formats map to the software depth32_float buffer (one
        // f32 per pixel). D16/D24S8 are accepted gracefully (same backing).
        vk.VK_FORMAT_D16_UNORM,
        vk.VK_FORMAT_X8_D24_UNORM_PACK32,
        vk.VK_FORMAT_D32_SFLOAT,
        vk.VK_FORMAT_D24_UNORM_S8_UINT,
        vk.VK_FORMAT_D32_SFLOAT_S8_UINT,
        => .depth32_float,
        else => .rgba8_unorm,
    };
}

/// Map a VkSampleCountFlagBits bitmask to an integer sample count the software path
/// understands (1/2/4). Higher counts are clamped to 4 (the max the rasterizer's
/// standard sample positions cover). 0 or unset maps to 1 (no MSAA).
fn sampleCount(bits: vk.VkSampleCountFlagBits) u8 {
    if (bits & vk.VK_SAMPLE_COUNT_4_BIT != 0) return 4;
    if (bits & vk.VK_SAMPLE_COUNT_2_BIT != 0) return 2;
    return 1;
}

/// Map a HAL compare op from VkCompareOp (the depth-test comparison).
fn halCompareOp(op: vk.VkCompareOp) prism.hal.CompareOp {
    return switch (op) {
        vk.VK_COMPARE_OP_NEVER => .never,
        vk.VK_COMPARE_OP_LESS => .less,
        vk.VK_COMPARE_OP_EQUAL => .equal,
        vk.VK_COMPARE_OP_LESS_OR_EQUAL => .less_or_equal,
        vk.VK_COMPARE_OP_GREATER => .greater,
        vk.VK_COMPARE_OP_NOT_EQUAL => .not_equal,
        vk.VK_COMPARE_OP_GREATER_OR_EQUAL => .greater_or_equal,
        vk.VK_COMPARE_OP_ALWAYS => .always,
        else => .less,
    };
}

fn halStencilOp(op: vk.VkStencilOp) prism.hal.StencilOp {
    return switch (op) {
        vk.VK_STENCIL_OP_KEEP => .keep,
        vk.VK_STENCIL_OP_ZERO => .zero,
        vk.VK_STENCIL_OP_REPLACE => .replace,
        vk.VK_STENCIL_OP_INCREMENT_AND_CLAMP => .incr_clamp,
        vk.VK_STENCIL_OP_DECREMENT_AND_CLAMP => .decr_clamp,
        vk.VK_STENCIL_OP_INVERT => .invert,
        vk.VK_STENCIL_OP_INCREMENT_AND_WRAP => .incr_wrap,
        vk.VK_STENCIL_OP_DECREMENT_AND_WRAP => .decr_wrap,
        else => .keep,
    };
}

/// A VkImage: an offscreen color target backed by a HAL image Resource.
/// Allocated at vkBindImageMemory time. The HAL image carries its own pixel
/// storage, so the bound VkDeviceMemory is unused for images. width/height/format come from create.
pub const Image = struct {
    width: u32,
    height: u32,
    format: prism.hal.Format,
    /// Whether the image is a sampled texture (VK_IMAGE_USAGE_SAMPLED_BIT). The HAL
    /// Resource gets `sampled` + `copy_dst` usage so vkCmdCopyBufferToImage can fill it
    /// and a fragment shader can sample it.
    sampled: bool = false,
    /// Whether the image is a color attachment (VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT). The
    /// HAL Resource gets `render_target` usage so a render pass can render into it. An
    /// image that is both a color attachment and sampled is a render-to-texture target:
    /// pass 1 renders into it, pass 2 samples it.
    color_attachment: bool = false,
    /// MSAA sample count (1/2/4) the image was created with. A multisampled image's HAL
    /// Resource is sized width*height*samples pixels (sample-minor) so the rasterizer can
    /// hold N samples per pixel. The render pass resolves it into a single-sample image.
    samples: u8 = 1,
    resource: ?*prism.hal.Resource = null,
    /// Lazily-allocated software stencil buffer (one u8 per pixel) for a depth/stencil
    /// attachment image. Vulkan packs depth+stencil into one image but the HAL keeps depth
    /// (f32) and stencil (u8) as separate buffers. A stencil-using render pass allocates
    /// this on demand and binds it via setStencilTarget. Freed in destroyImage.
    stencil_resource: ?*prism.hal.Resource = null,
    /// Get-or-create the per-pixel u8 stencil buffer (needs the HAL device to allocate).
    fn stencilResource(self: *Image, hal: prism.hal.Device) ?*prism.hal.Resource {
        if (self.stencil_resource) |s| return s;
        const s = hal.createResource(.{ .buffer = .{ .size = @as(usize, self.width) * self.height } }) catch return null;
        self.stencil_resource = s;
        return s;
    }
    fn toHandle(self: *Image) vk.VkImage {
        return @intCast(@intFromPtr(self));
    }
    fn fromHandle(h: vk.VkImage) *Image {
        return @ptrFromInt(@as(usize, @intCast(h)));
    }
};

/// A VkImageView over a VkImage (the framebuffer attachment). The software path
/// renders straight into the image, so the view is just a typed reference.
pub const ImageView = struct {
    image: *Image,
    fn toHandle(self: *ImageView) vk.VkImageView {
        return @intCast(@intFromPtr(self));
    }
    fn fromHandle(h: vk.VkImageView) *ImageView {
        return @ptrFromInt(@as(usize, @intCast(h)));
    }
};

/// A VkRenderPass: one color attachment (loadOp CLEAR, storeOp STORE) plus an
/// optional depth/stencil attachment (referenced by the subpass's
/// pDepthStencilAttachment). The software path always clears then stores, so only
/// the attachments' presence + the depth attachment's index are tracked. The depth
/// clear value is supplied at vkCmdBeginRenderPass via pClearValues[depth_index].
pub const RenderPass = struct {
    color_format: prism.hal.Format,
    /// True when the subpass references a depth/stencil attachment.
    has_depth: bool = false,
    /// The attachment index of the depth attachment (so cmdBeginRenderPass reads
    /// pClearValues[depth_index].depthStencil.depth).
    depth_index: u32 = 0,
    /// MSAA: true when the subpass declares a resolve attachment (pResolveAttachments[0]).
    /// The multisampled color attachment is resolved (box-averaged) into the
    /// single-sample image at `resolve_index` when the render pass ends.
    has_resolve: bool = false,
    resolve_index: u32 = 0,
    /// MSAA sample count of the (multisampled) color attachment, from its
    /// VkAttachmentDescription.samples. 1 = no MSAA.
    samples: u8 = 1,
    fn toHandle(self: *RenderPass) vk.VkRenderPass {
        return @intCast(@intFromPtr(self));
    }
    fn fromHandle(h: vk.VkRenderPass) *RenderPass {
        return @ptrFromInt(@as(usize, @intCast(h)));
    }
};

/// A VkFramebuffer: the color attachment's image view + dimensions, plus an
/// optional depth attachment view (when the render pass has a depth attachment).
pub const Framebuffer = struct {
    view: *ImageView,
    depth_view: ?*ImageView = null,
    /// MSAA: the single-sample resolve target's view (pAttachments[resolve_index]),
    /// when the render pass declares a resolve attachment. The multisampled color
    /// attachment (`view`) is box-averaged into this image at EndRenderPass.
    resolve_view: ?*ImageView = null,
    width: u32,
    height: u32,
    fn toHandle(self: *Framebuffer) vk.VkFramebuffer {
        return @intCast(@intFromPtr(self));
    }
    fn fromHandle(h: vk.VkFramebuffer) *Framebuffer {
        return @ptrFromInt(@as(usize, @intCast(h)));
    }
};

/// The vertex attributes a graphics pipeline reads. The software path needs the
/// per-vertex stride, which attribute location holds position vs color, and each
/// attribute's byte offset + format (so it can read a vec2 position + a vec3/vec4
/// color). The ICD derives position = the smallest attribute location, color =
/// the next, from VkPipelineVertexInputStateCreateInfo.
const MAX_VERTEX_ATTRS = 8;
pub const GraphicsPipeline = struct {
    /// The HAL shader modules + pipeline that drive the software rasterizer.
    hal_vs: *prism.hal.ShaderModule,
    hal_fs: *prism.hal.ShaderModule,
    hal_pipeline: *prism.hal.Pipeline,
    /// Owned copy of the HAL vertex attributes (the HAL pipeline borrows this).
    attrs: [MAX_VERTEX_ATTRS]prism.hal.VertexAttribute = undefined,
    attr_count: usize = 0,
    /// The depth-test state parsed from VkPipelineDepthStencilStateCreateInfo
    /// (disabled by default so the color-only path is unchanged).
    depth: prism.hal.DepthState = .{},
    /// The stencil-test state parsed from VkPipelineDepthStencilStateCreateInfo's `front`
    /// face (disabled by default). The HAL carries a single face. Vulkan's front state is
    /// used (UI/2D clip workloads draw front-facing geometry).
    stencil: prism.hal.StencilState = .{},
    /// Optional back-face stencil state (two-sided stencil, Vulkan's ds.back when it differs
    /// from ds.front). null means single-face (`stencil` for both windings).
    stencil_back: ?prism.hal.StencilState = null,
    /// Dynamic stencil state (VkPipelineDynamicStateCreateInfo). When set, the corresponding
    /// stencil field is not baked from the pipeline. It comes from vkCmdSetStencil* at draw
    /// time, and executeRender rebuilds a per-draw HAL pipeline variant with the live values.
    /// `stride`/`cull`/`color_format` are kept so the variant can be rebuilt.
    dyn_stencil_ref: bool = false,
    dyn_stencil_compare: bool = false,
    dyn_stencil_write: bool = false,
    stride: u32 = 0,
    cull: prism.hal.CullState = .{},
    /// Color write mask (colorWriteMask). Kept so a dynamic-stencil variant rebuild preserves it.
    blend: prism.hal.BlendState = .{},
    /// The descriptor binding count from the pipeline layout, and whether the layout
    /// declares push constants. When it does, the draw binds the snapshotted push-constant
    /// bytes as a UBO at binding index `pc_binding` (= binding_count), the slot right after
    /// the descriptors. This is where the PushConstant variable lands in the shader's
    /// lowered pointer-param order.
    pc_binding: u32 = 0,
    has_push_constants: bool = false,
    /// MSAA sample count (1/2/4) from VkPipelineMultisampleStateCreateInfo. Render pass
    /// instances created with this pipeline allocate N-sample attachments + resolve.
    samples: u8 = 1,
    /// The handle carries the graphics tag in bit 0 (see GRAPHICS_PIPELINE_TAG).
    fn toHandle(self: *GraphicsPipeline) vk.VkPipeline {
        return @as(u64, @intCast(@intFromPtr(self))) | GRAPHICS_PIPELINE_TAG;
    }
    fn fromHandle(h: vk.VkPipeline) *GraphicsPipeline {
        return @ptrFromInt(@as(usize, @intCast(h & ~GRAPHICS_PIPELINE_TAG)));
    }
};

/// A VkFence: a synchronous submit completes before returning, so a fence is
/// always signaled. We track the signaled bit for vkWaitForFences/vkGetFenceStatus.
pub const Fence = struct {
    signaled: bool,
    fn toHandle(self: *Fence) vk.VkFence {
        return @intCast(@intFromPtr(self));
    }
    fn fromHandle(h: vk.VkFence) *Fence {
        return @ptrFromInt(@as(usize, @intCast(h)));
    }
};

pub fn createShaderModule(
    device: vk.VkDevice,
    pCreateInfo: ?*const vk.VkShaderModuleCreateInfo,
    pAllocator: ?*const vk.VkAllocationCallbacks,
    pShaderModule: *vk.VkShaderModule,
) callconv(.c) vk.VkResult {
    _ = device;
    _ = pAllocator;
    const ci = pCreateInfo orelse return .VK_ERROR_INITIALIZATION_FAILED;
    const code = ci.pCode orelse return .VK_ERROR_INITIALIZATION_FAILED;
    if (ci.codeSize == 0 or ci.codeSize % 4 != 0) return .VK_ERROR_INITIALIZATION_FAILED;
    const bytes = std.mem.sliceAsBytes(code[0 .. ci.codeSize / 4]);
    const copy = allocator.alloc(u8, bytes.len) catch return .VK_ERROR_OUT_OF_HOST_MEMORY;
    @memcpy(copy, bytes);
    const m = allocator.create(ShaderModule) catch {
        allocator.free(copy);
        return .VK_ERROR_OUT_OF_HOST_MEMORY;
    };
    m.* = .{ .spirv = copy };
    pShaderModule.* = m.toHandle();
    return .VK_SUCCESS;
}

pub fn destroyShaderModule(
    device: vk.VkDevice,
    shaderModule: vk.VkShaderModule,
    pAllocator: ?*const vk.VkAllocationCallbacks,
) callconv(.c) void {
    _ = device;
    _ = pAllocator;
    if (shaderModule == vk.VK_NULL_HANDLE) return;
    const m = ShaderModule.fromHandle(shaderModule);
    allocator.free(m.spirv);
    allocator.destroy(m);
}

pub fn createDescriptorSetLayout(
    device: vk.VkDevice,
    pCreateInfo: ?*const vk.VkDescriptorSetLayoutCreateInfo,
    pAllocator: ?*const vk.VkAllocationCallbacks,
    pSetLayout: *vk.VkDescriptorSetLayout,
) callconv(.c) vk.VkResult {
    _ = device;
    _ = pAllocator;
    const ci = pCreateInfo orelse return .VK_ERROR_INITIALIZATION_FAILED;
    // Count storage and uniform-buffer bindings. The highest binding index + 1 sets
    // the arity (the dispatch/draw path scans bindings 0..count-1 for bound buffers).
    var max_binding: u32 = 0;
    var any = false;
    if (ci.pBindings) |binds| {
        for (binds[0..ci.bindingCount]) |bnd| {
            if (bnd.descriptorType == vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER or
                bnd.descriptorType == vk.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER or
                bnd.descriptorType == vk.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
            {
                if (bnd.binding + 1 > max_binding) max_binding = bnd.binding + 1;
                any = true;
            }
        }
    }
    const layout = allocator.create(DescriptorSetLayout) catch return .VK_ERROR_OUT_OF_HOST_MEMORY;
    layout.* = .{ .binding_count = if (any) max_binding else ci.bindingCount };
    pSetLayout.* = layout.toHandle();
    return .VK_SUCCESS;
}

pub fn destroyDescriptorSetLayout(
    device: vk.VkDevice,
    descriptorSetLayout: vk.VkDescriptorSetLayout,
    pAllocator: ?*const vk.VkAllocationCallbacks,
) callconv(.c) void {
    _ = device;
    _ = pAllocator;
    if (descriptorSetLayout == vk.VK_NULL_HANDLE) return;
    allocator.destroy(DescriptorSetLayout.fromHandle(descriptorSetLayout));
}

pub fn createPipelineLayout(
    device: vk.VkDevice,
    pCreateInfo: ?*const vk.VkPipelineLayoutCreateInfo,
    pAllocator: ?*const vk.VkAllocationCallbacks,
    pPipelineLayout: *vk.VkPipelineLayout,
) callconv(.c) vk.VkResult {
    _ = device;
    _ = pAllocator;
    const ci = pCreateInfo orelse return .VK_ERROR_INITIALIZATION_FAILED;
    var binding_count: u32 = 0;
    if (ci.pSetLayouts) |sl| {
        if (ci.setLayoutCount >= 1) {
            binding_count = DescriptorSetLayout.fromHandle(sl[0]).binding_count;
        }
    }
    // Push-constant ranges: track the high-water byte the declared ranges cover
    // (offset + size, capped at the backed block size). A nonzero result means the
    // pipeline reads a push-constant block, so the draw threads the pushed bytes into
    // the shader as the buffer pointer after the descriptors.
    var push_constant_size: u32 = 0;
    if (ci.pPushConstantRanges) |ranges| {
        var r: u32 = 0;
        while (r < ci.pushConstantRangeCount) : (r += 1) {
            const hi = @min(ranges[r].offset + ranges[r].size, PUSH_CONSTANT_SIZE);
            if (hi > push_constant_size) push_constant_size = hi;
        }
    }
    const pl = allocator.create(PipelineLayout) catch return .VK_ERROR_OUT_OF_HOST_MEMORY;
    pl.* = .{ .binding_count = binding_count, .push_constant_size = push_constant_size };
    pPipelineLayout.* = pl.toHandle();
    return .VK_SUCCESS;
}

pub fn destroyPipelineLayout(
    device: vk.VkDevice,
    pipelineLayout: vk.VkPipelineLayout,
    pAllocator: ?*const vk.VkAllocationCallbacks,
) callconv(.c) void {
    _ = device;
    _ = pAllocator;
    if (pipelineLayout == vk.VK_NULL_HANDLE) return;
    allocator.destroy(PipelineLayout.fromHandle(pipelineLayout));
}

pub fn createComputePipelines(
    device: vk.VkDevice,
    pipelineCache: vk.VkPipelineCache,
    createInfoCount: u32,
    pCreateInfos: ?[*]const vk.VkComputePipelineCreateInfo,
    pAllocator: ?*const vk.VkAllocationCallbacks,
    pPipelines: ?[*]vk.VkPipeline,
) callconv(.c) vk.VkResult {
    _ = pipelineCache;
    _ = pAllocator;
    const dev = LogicalDevice.fromHandle(device);
    const cis = pCreateInfos orelse return .VK_ERROR_INITIALIZATION_FAILED;
    const out = pPipelines orelse return .VK_ERROR_INITIALIZATION_FAILED;
    var i: u32 = 0;
    while (i < createInfoCount) : (i += 1) {
        const ci = cis[i];
        if (ci.stage.module == vk.VK_NULL_HANDLE) return .VK_ERROR_INITIALIZATION_FAILED;
        const m = ShaderModule.fromHandle(ci.stage.module);
        // Build the HAL compute shader module from the SPIR-V (the software driver
        // retains the bytes; it JITs them at dispatch time).
        const hal_shader = dev.hal().createShaderModule(.{
            .stage = .compute,
            .code = m.spirv,
        }) catch return .VK_ERROR_INITIALIZATION_FAILED;
        const pipe = allocator.create(ComputePipeline) catch {
            dev.hal().destroyShaderModule(hal_shader);
            return .VK_ERROR_OUT_OF_HOST_MEMORY;
        };
        const binding_count = if (ci.layout != vk.VK_NULL_HANDLE)
            PipelineLayout.fromHandle(ci.layout).binding_count
        else
            0;
        pipe.* = .{ .hal_shader = hal_shader, .binding_count = binding_count };
        out[i] = pipe.toHandle();
    }
    return .VK_SUCCESS;
}

pub fn destroyPipeline(
    device: vk.VkDevice,
    pipeline: vk.VkPipeline,
    pAllocator: ?*const vk.VkAllocationCallbacks,
) callconv(.c) void {
    _ = pAllocator;
    if (pipeline == vk.VK_NULL_HANDLE) return;
    const dev = LogicalDevice.fromHandle(device);
    // The handle's low bit tags compute (0) vs graphics (1). Pick the freer.
    if ((pipeline & GRAPHICS_PIPELINE_TAG) != 0) {
        const pipe = GraphicsPipeline.fromHandle(pipeline);
        dev.hal().destroyPipeline(pipe.hal_pipeline);
        dev.hal().destroyShaderModule(pipe.hal_vs);
        dev.hal().destroyShaderModule(pipe.hal_fs);
        allocator.destroy(pipe);
    } else {
        const pipe = ComputePipeline.fromHandle(pipeline);
        dev.hal().destroyShaderModule(pipe.hal_shader);
        allocator.destroy(pipe);
    }
}

// Graphics: image + view + render pass + framebuffer + pipeline (M4).

pub fn createImage(
    device: vk.VkDevice,
    pCreateInfo: ?*const vk.VkImageCreateInfo,
    pAllocator: ?*const vk.VkAllocationCallbacks,
    pImage: *vk.VkImage,
) callconv(.c) vk.VkResult {
    _ = device;
    _ = pAllocator;
    const ci = pCreateInfo orelse return .VK_ERROR_INITIALIZATION_FAILED;
    const img = allocator.create(Image) catch return .VK_ERROR_OUT_OF_HOST_MEMORY;
    img.* = .{
        .width = ci.extent.width,
        .height = ci.extent.height,
        .format = halFormat(ci.format),
        .sampled = (ci.usage & vk.VK_IMAGE_USAGE_SAMPLED_BIT) != 0,
        .color_attachment = (ci.usage & vk.VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT) != 0,
        .samples = sampleCount(ci.samples),
    };
    pImage.* = img.toHandle();
    vtrace("createImage {}x{} fmt={} tiling={} usage=0x{x} sampled={} -> 0x{x}", .{
        ci.extent.width, ci.extent.height, @as(i64, ci.format), @as(i64, ci.tiling), ci.usage, img.sampled, @intFromPtr(img),
    });
    return .VK_SUCCESS;
}

pub fn destroyImage(
    device: vk.VkDevice,
    image: vk.VkImage,
    pAllocator: ?*const vk.VkAllocationCallbacks,
) callconv(.c) void {
    _ = pAllocator;
    if (image == vk.VK_NULL_HANDLE) return;
    const dev = LogicalDevice.fromHandle(device);
    const img = Image.fromHandle(image);
    if (img.resource) |r| dev.hal().destroyResource(r);
    if (img.stencil_resource) |r| dev.hal().destroyResource(r);
    allocator.destroy(img);
}

pub fn getImageMemoryRequirements(
    device: vk.VkDevice,
    image: vk.VkImage,
    pMemoryRequirements: *vk.VkMemoryRequirements,
) callconv(.c) void {
    _ = device;
    const img = Image.fromHandle(image);
    const size = @as(u64, img.width) * img.height * @max(1, img.samples) * img.format.bytesPerPixel();
    pMemoryRequirements.* = .{
        .size = roundUp(size, PRISM_BUFFER_ALIGNMENT),
        .alignment = PRISM_BUFFER_ALIGNMENT,
        .memoryTypeBits = 0b1,
    };
}

pub fn bindImageMemory(
    device: vk.VkDevice,
    image: vk.VkImage,
    memory: vk.VkDeviceMemory,
    memoryOffset: vk.VkDeviceSize,
) callconv(.c) vk.VkResult {
    _ = memory;
    _ = memoryOffset;
    // A HAL image carries its own pixel storage, so we allocate the image's HAL
    // Resource here (at bind time) rather than aliasing the bound VkDeviceMemory.
    // The bound memory handle is accepted but unused for images.
    const dev = LogicalDevice.fromHandle(device);
    const img = Image.fromHandle(image);
    if (img.resource != null) return .VK_SUCCESS;
    // A depth image backs an f32-per-pixel depth buffer. Otherwise the usage is the union
    // of what the image was created for: `render_target` when it is a color attachment,
    // `sampled` when a shader samples it, plus copy bits. A render-to-texture image has
    // both color_attachment and sampled set: pass 1 renders into it (render_target) and
    // pass 2 samples it (sampled). The software Resource's pixels are the same bytes
    // regardless of flags, so one Resource serves both roles.
    const usage: prism.hal.ResourceUsage = if (img.format == .depth32_float)
        .{ .render_target = true }
    else
        .{
            .render_target = img.color_attachment or !img.sampled,
            .sampled = img.sampled,
            .copy_dst = img.sampled,
            .copy_src = true,
        };
    const r = dev.hal().createResource(.{ .image = .{
        .width = img.width,
        .height = img.height,
        .format = img.format,
        .usage = usage,
        .samples = img.samples,
    } }) catch return .VK_ERROR_OUT_OF_DEVICE_MEMORY;
    img.resource = r;
    return .VK_SUCCESS;
}

pub fn createImageView(
    device: vk.VkDevice,
    pCreateInfo: ?*const vk.VkImageViewCreateInfo,
    pAllocator: ?*const vk.VkAllocationCallbacks,
    pView: *vk.VkImageView,
) callconv(.c) vk.VkResult {
    _ = device;
    _ = pAllocator;
    const ci = pCreateInfo orelse return .VK_ERROR_INITIALIZATION_FAILED;
    if (ci.image == vk.VK_NULL_HANDLE) return .VK_ERROR_INITIALIZATION_FAILED;
    const view = allocator.create(ImageView) catch return .VK_ERROR_OUT_OF_HOST_MEMORY;
    view.* = .{ .image = Image.fromHandle(ci.image) };
    pView.* = view.toHandle();
    vtrace("createImageView img=0x{x} fmt={} sampled={} -> view=0x{x}", .{
        @intFromPtr(view.image), @as(i64, ci.format), view.image.sampled, @intFromPtr(view),
    });
    return .VK_SUCCESS;
}

pub fn destroyImageView(
    device: vk.VkDevice,
    imageView: vk.VkImageView,
    pAllocator: ?*const vk.VkAllocationCallbacks,
) callconv(.c) void {
    _ = device;
    _ = pAllocator;
    if (imageView == vk.VK_NULL_HANDLE) return;
    allocator.destroy(ImageView.fromHandle(imageView));
}

fn halFilter(f: vk.VkFilter) prism.hal.Filter {
    return switch (f) {
        vk.VK_FILTER_LINEAR => .linear,
        else => .nearest,
    };
}

fn halAddressMode(m: vk.VkSamplerAddressMode) prism.hal.AddressMode {
    return switch (m) {
        vk.VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE => .clamp_to_edge,
        vk.VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT => .mirrored_repeat,
        else => .repeat,
    };
}

pub fn createSampler(
    device: vk.VkDevice,
    pCreateInfo: ?*const vk.VkSamplerCreateInfo,
    pAllocator: ?*const vk.VkAllocationCallbacks,
    pSampler: *vk.VkSampler,
) callconv(.c) vk.VkResult {
    _ = device;
    _ = pAllocator;
    const ci = pCreateInfo orelse return .VK_ERROR_INITIALIZATION_FAILED;
    const s = allocator.create(Sampler) catch return .VK_ERROR_OUT_OF_HOST_MEMORY;
    s.* = .{
        .filter = halFilter(ci.magFilter),
        .address_u = halAddressMode(ci.addressModeU),
        .address_v = halAddressMode(ci.addressModeV),
        .max_anisotropy = if (ci.anisotropyEnable != vk.VK_FALSE) ci.maxAnisotropy else 1,
    };
    pSampler.* = s.toHandle();
    return .VK_SUCCESS;
}

pub fn destroySampler(
    device: vk.VkDevice,
    sampler: vk.VkSampler,
    pAllocator: ?*const vk.VkAllocationCallbacks,
) callconv(.c) void {
    _ = device;
    _ = pAllocator;
    if (sampler == vk.VK_NULL_HANDLE) return;
    allocator.destroy(Sampler.fromHandle(sampler));
}

/// vkGetImageSubresourceLayout: the software path stores a sampled image linearly,
/// tightly packed RGBA8 (offset 0, rowPitch = width*4).
pub fn getImageSubresourceLayout(
    device: vk.VkDevice,
    image: vk.VkImage,
    pSubresource: ?*const vk.VkImageSubresource,
    pLayout: *vk.VkSubresourceLayout,
) callconv(.c) void {
    _ = device;
    _ = pSubresource;
    const img = Image.fromHandle(image);
    const pitch = @as(u64, img.width) * 4;
    pLayout.* = .{
        .offset = 0,
        .size = pitch * img.height,
        .rowPitch = pitch,
        .arrayPitch = pitch * img.height,
        .depthPitch = pitch * img.height,
    };
}

pub fn createRenderPass(
    device: vk.VkDevice,
    pCreateInfo: ?*const vk.VkRenderPassCreateInfo,
    pAllocator: ?*const vk.VkAllocationCallbacks,
    pRenderPass: *vk.VkRenderPass,
) callconv(.c) vk.VkResult {
    _ = device;
    _ = pAllocator;
    const ci = pCreateInfo orelse return .VK_ERROR_INITIALIZATION_FAILED;
    var fmt: prism.hal.Format = .rgba8_unorm;
    if (ci.pAttachments) |atts| {
        if (ci.attachmentCount >= 1) fmt = halFormat(atts[0].format);
    }
    // Detect a depth/stencil attachment: the subpass's pDepthStencilAttachment
    // names the attachment index whose VkAttachmentDescription has a depth format.
    var has_depth = false;
    var depth_index: u32 = 0;
    if (ci.pSubpasses) |subs| {
        if (ci.subpassCount >= 1) {
            if (subs[0].pDepthStencilAttachment) |dsref| {
                const idx = dsref.attachment;
                if (idx != vk.VK_ATTACHMENT_UNUSED and ci.pAttachments != null and idx < ci.attachmentCount) {
                    if (vk.formatIsDepth(ci.pAttachments.?[idx].format)) {
                        has_depth = true;
                        depth_index = idx;
                    }
                }
            }
        }
    }
    // MSAA: the color attachment's sample count + an optional resolve attachment. The
    // subpass's pResolveAttachments[0] (when present and not VK_ATTACHMENT_UNUSED) names
    // the single-sample image the multisampled color attachment box-averages into at EndRenderPass.
    var samples: u8 = 1;
    if (ci.pAttachments) |atts| {
        if (ci.attachmentCount >= 1) samples = sampleCount(atts[0].samples);
    }
    var has_resolve = false;
    var resolve_index: u32 = 0;
    if (ci.pSubpasses) |subs| {
        if (ci.subpassCount >= 1) {
            if (subs[0].pResolveAttachments) |rrefs| {
                if (subs[0].colorAttachmentCount >= 1) {
                    const idx = rrefs[0].attachment;
                    if (idx != vk.VK_ATTACHMENT_UNUSED and ci.pAttachments != null and idx < ci.attachmentCount) {
                        has_resolve = true;
                        resolve_index = idx;
                    }
                }
            }
        }
    }
    const rp = allocator.create(RenderPass) catch return .VK_ERROR_OUT_OF_HOST_MEMORY;
    rp.* = .{
        .color_format = fmt,
        .has_depth = has_depth,
        .depth_index = depth_index,
        .has_resolve = has_resolve,
        .resolve_index = resolve_index,
        .samples = samples,
    };
    pRenderPass.* = rp.toHandle();
    return .VK_SUCCESS;
}

pub fn destroyRenderPass(
    device: vk.VkDevice,
    renderPass: vk.VkRenderPass,
    pAllocator: ?*const vk.VkAllocationCallbacks,
) callconv(.c) void {
    _ = device;
    _ = pAllocator;
    if (renderPass == vk.VK_NULL_HANDLE) return;
    allocator.destroy(RenderPass.fromHandle(renderPass));
}

pub fn createFramebuffer(
    device: vk.VkDevice,
    pCreateInfo: ?*const vk.VkFramebufferCreateInfo,
    pAllocator: ?*const vk.VkAllocationCallbacks,
    pFramebuffer: *vk.VkFramebuffer,
) callconv(.c) vk.VkResult {
    _ = device;
    _ = pAllocator;
    const ci = pCreateInfo orelse return .VK_ERROR_INITIALIZATION_FAILED;
    if (ci.pAttachments == null or ci.attachmentCount < 1) return .VK_ERROR_INITIALIZATION_FAILED;
    // The color attachment is index 0. If the render pass has a depth attachment,
    // its view is at pAttachments[depth_index]. Identify the depth view by either the
    // render pass's depth_index or by the view's image format.
    const view = ImageView.fromHandle(ci.pAttachments.?[0]);
    var depth_view: ?*ImageView = null;
    if (ci.renderPass != vk.VK_NULL_HANDLE) {
        const rp = RenderPass.fromHandle(ci.renderPass);
        if (rp.has_depth and rp.depth_index < ci.attachmentCount) {
            const dv = ImageView.fromHandle(ci.pAttachments.?[rp.depth_index]);
            if (dv.image.format == .depth32_float) depth_view = dv;
        }
    }
    // Fall back: if any attachment is a depth view, treat it as the depth attachment
    // (covers a render pass handle we could not resolve).
    if (depth_view == null) {
        var a: u32 = 1;
        while (a < ci.attachmentCount) : (a += 1) {
            const v = ImageView.fromHandle(ci.pAttachments.?[a]);
            if (v.image.format == .depth32_float) {
                depth_view = v;
                break;
            }
        }
    }
    // MSAA resolve target: the render pass's resolve_index names the single-sample image
    // the multisampled color attachment resolves into at EndRenderPass.
    var resolve_view: ?*ImageView = null;
    if (ci.renderPass != vk.VK_NULL_HANDLE) {
        const rp = RenderPass.fromHandle(ci.renderPass);
        if (rp.has_resolve and rp.resolve_index < ci.attachmentCount) {
            resolve_view = ImageView.fromHandle(ci.pAttachments.?[rp.resolve_index]);
        }
    }
    const fb = allocator.create(Framebuffer) catch return .VK_ERROR_OUT_OF_HOST_MEMORY;
    fb.* = .{ .view = view, .depth_view = depth_view, .resolve_view = resolve_view, .width = ci.width, .height = ci.height };
    pFramebuffer.* = fb.toHandle();
    return .VK_SUCCESS;
}

pub fn destroyFramebuffer(
    device: vk.VkDevice,
    framebuffer: vk.VkFramebuffer,
    pAllocator: ?*const vk.VkAllocationCallbacks,
) callconv(.c) void {
    _ = device;
    _ = pAllocator;
    if (framebuffer == vk.VK_NULL_HANDLE) return;
    allocator.destroy(Framebuffer.fromHandle(framebuffer));
}

/// Map a VkBlendFactor (canonical Vulkan integer value) to the HAL BlendFactor. Unknown -> one.
fn vkBlendFactor(v: i32) prism.hal.BlendFactor {
    return switch (v) {
        0 => .zero,
        1 => .one,
        2 => .src_color,
        3 => .one_minus_src_color,
        4 => .dst_color,
        5 => .one_minus_dst_color,
        6 => .src_alpha,
        7 => .one_minus_src_alpha,
        8 => .dst_alpha,
        9 => .one_minus_dst_alpha,
        10 => .constant_color,
        11 => .one_minus_constant_color,
        12 => .constant_alpha,
        13 => .one_minus_constant_alpha,
        14 => .src_alpha_saturate,
        else => .one,
    };
}

/// Map a VkBlendOp (canonical Vulkan integer value) to the HAL BlendOp. Unknown -> add.
fn vkBlendOp(v: i32) prism.hal.BlendOp {
    return switch (v) {
        0 => .add,
        1 => .subtract,
        2 => .reverse_subtract,
        3 => .min,
        4 => .max,
        else => .add,
    };
}

/// Build graphics pipelines from VkGraphicsPipelineCreateInfo. Derives vertex
/// attribute locations, stride, depth/stencil/blend/cull state, MSAA count, and
/// dynamic stencil flags. Passes real VS/FS SPIR-V to the HAL for JIT execution.
pub fn createGraphicsPipelines(
    device: vk.VkDevice,
    pipelineCache: vk.VkPipelineCache,
    createInfoCount: u32,
    pCreateInfos: ?[*]const vk.VkGraphicsPipelineCreateInfo,
    pAllocator: ?*const vk.VkAllocationCallbacks,
    pPipelines: ?[*]vk.VkPipeline,
) callconv(.c) vk.VkResult {
    _ = pipelineCache;
    _ = pAllocator;
    const dev = LogicalDevice.fromHandle(device);
    const cis = pCreateInfos orelse return .VK_ERROR_INITIALIZATION_FAILED;
    const out = pPipelines orelse return .VK_ERROR_INITIALIZATION_FAILED;
    const shader = prism.software.shader;
    var i: u32 = 0;
    while (i < createInfoCount) : (i += 1) {
        const ci = cis[i];
        const vis = ci.pVertexInputState orelse return .VK_ERROR_INITIALIZATION_FAILED;

        // Derive position/color attribute locations: position = the smallest
        // attribute location, color = the next-smallest. Copy each attribute's
        // location/format/offset into a HAL VertexAttribute (owned by the pipe).
        var attrs: [MAX_VERTEX_ATTRS]prism.hal.VertexAttribute = undefined;
        var n: usize = 0;
        if (vis.pVertexAttributeDescriptions) |va| {
            const count = @min(vis.vertexAttributeDescriptionCount, MAX_VERTEX_ATTRS);
            var k: u32 = 0;
            while (k < count) : (k += 1) {
                attrs[n] = .{
                    .location = va[k].location,
                    .format = halFormat(va[k].format),
                    .offset = va[k].offset,
                };
                n += 1;
            }
        }
        // Find the VS and FS stage SPIR-V from the pipeline's shader stages. Passing
        // the real SPIR-V to the HAL makes the software driver JIT + execute arbitrary
        // vertex/fragment shaders (real per-vertex / per-fragment SPIR-V execution).
        // If a stage's SPIR-V is missing, fall back to the declarative passthrough IR
        // for that stage (the software driver accepts either).
        // Vertex stride from the (single) binding description. A vertex-pulling pipeline
        // has no binding description, so stride stays 0 (no vertex buffer is read).
        var stride: u32 = 0;
        if (vis.pVertexBindingDescriptions) |vb| {
            if (vis.vertexBindingDescriptionCount >= 1) stride = vb[0].stride;
        }

        var vs_spirv: ?[]const u8 = null;
        var fs_spirv: ?[]const u8 = null;
        if (ci.pStages) |stages| {
            var s: u32 = 0;
            while (s < ci.stageCount) : (s += 1) {
                const st = stages[s];
                if (st.module == vk.VK_NULL_HANDLE) continue;
                const m = ShaderModule.fromHandle(st.module);
                if (st.stage & vk.VK_SHADER_STAGE_VERTEX_BIT != 0) vs_spirv = m.spirv;
                if (st.stage & vk.VK_SHADER_STAGE_FRAGMENT_BIT != 0) fs_spirv = m.spirv;
            }
        }

        // The declarative passthrough VS (used only when the pipeline has no vertex-shader
        // SPIR-V) needs the position + color attribute locations. With real VS SPIR-V the
        // JIT path runs and these are unused, so a zero-attribute (vertex-pulling) pipeline
        // is allowed. Only the declarative fallback requires two attributes.
        var pos_loc: u32 = if (n > 0) attrs[0].location else 0;
        var col_loc: u32 = std.math.maxInt(u32);
        for (attrs[0..n]) |a| {
            if (a.location < pos_loc) pos_loc = a.location;
        }
        for (attrs[0..n]) |a| {
            if (a.location != pos_loc and a.location < col_loc) col_loc = a.location;
        }
        if (vs_spirv == null) {
            // Declarative passthrough path: it needs a position + color attribute.
            if (n < 2 or col_loc == std.math.maxInt(u32)) return .VK_ERROR_INITIALIZATION_FAILED;
        } else if (col_loc == std.math.maxInt(u32)) {
            col_loc = pos_loc; // unused by the JIT path. Keep encodeVertex well-formed.
        }

        const hal_vs_bytes = shader.encodeVertex(.{ .position_attr = pos_loc, .color_attr = col_loc });
        const hal_vs = dev.hal().createShaderModule(.{
            .stage = .vertex,
            .code = vs_spirv orelse &hal_vs_bytes,
        }) catch return .VK_ERROR_INITIALIZATION_FAILED;
        const hal_fs_bytes = shader.encodeFragment(.{});
        const hal_fs = dev.hal().createShaderModule(.{
            .stage = .fragment,
            .code = fs_spirv orelse &hal_fs_bytes,
        }) catch {
            dev.hal().destroyShaderModule(hal_vs);
            return .VK_ERROR_INITIALIZATION_FAILED;
        };

        // Parse the depth-test state. Default (no pDepthStencilState or the test
        // disabled) = no depth test, so the M4/M5/UBO color-only path is unchanged.
        var depth_state: prism.hal.DepthState = .{};
        var stencil_state: prism.hal.StencilState = .{};
        var stencil_back: ?prism.hal.StencilState = null;
        if (ci.pDepthStencilState) |ds| {
            if (ds.depthTestEnable != vk.VK_FALSE) {
                depth_state = .{
                    .test_enable = true,
                    .write_enable = ds.depthWriteEnable != vk.VK_FALSE,
                    .compare_op = halCompareOp(ds.depthCompareOp),
                };
            }
            // Stencil from the front face (the HAL carries a single face). compareMask/
            // writeMask/reference are dynamic state in Vulkan, but the common case sets
            // them statically here. vkCmdSetStencil* overrides land in a later pass.
            if (ds.stencilTestEnable != vk.VK_FALSE) {
                const f = ds.front;
                stencil_state = .{
                    .test_enable = true,
                    .compare_op = halCompareOp(f.compareOp),
                    .fail_op = halStencilOp(f.failOp),
                    .depth_fail_op = halStencilOp(f.depthFailOp),
                    .pass_op = halStencilOp(f.passOp),
                    .compare_mask = @truncate(f.compareMask),
                    .write_mask = @truncate(f.writeMask),
                    .reference = @truncate(f.reference),
                };
                // Two-sided: Vulkan always carries a back VkStencilOpState. Set stencil_back
                // only when it differs from the front (equal means single-face, keeps the
                // front-only path unchanged). Front-facing fragments use `front`, back-facing `back`.
                const b = ds.back;
                const back_state: prism.hal.StencilState = .{
                    .test_enable = true,
                    .compare_op = halCompareOp(b.compareOp),
                    .fail_op = halStencilOp(b.failOp),
                    .depth_fail_op = halStencilOp(b.depthFailOp),
                    .pass_op = halStencilOp(b.passOp),
                    .compare_mask = @truncate(b.compareMask),
                    .write_mask = @truncate(b.writeMask),
                    .reference = @truncate(b.reference),
                };
                if (!std.meta.eql(back_state, stencil_state)) stencil_back = back_state;
            }
        }

        // Parse the back-face cull state. Default (no pRasterizationState) = no culling,
        // so existing pipelines render both windings exactly as before. vkcube culls back
        // faces (cullMode = BACK, frontFace = CCW) so its unlit back faces don't overdraw
        // the lit front faces (which made the cube render black).
        var cull_state: prism.hal.CullState = .{};
        if (ci.pRasterizationState) |rs| {
            cull_state.mode = switch (rs.cullMode) {
                vk.VK_CULL_MODE_BACK_BIT => .back,
                vk.VK_CULL_MODE_FRONT_BIT => .front,
                else => .none, // NONE or FRONT_AND_BACK (we don't discard everything)
            };
            cull_state.front_face = switch (rs.frontFace) {
                vk.VK_FRONT_FACE_CLOCKWISE => .clockwise,
                else => .counter_clockwise,
            };
            // Depth bias (glPolygonOffset equivalent). Only meaningful with a depth test. When
            // depth is disabled the software path ignores it (bias only shifts tested depth).
            if (rs.depthBiasEnable != vk.VK_FALSE) {
                depth_state.bias_enable = true;
                depth_state.bias_constant = rs.depthBiasConstantFactor;
                depth_state.bias_slope = rs.depthBiasSlopeFactor;
                depth_state.bias_clamp = rs.depthBiasClamp;
            }
        }
        // Parse the MSAA sample count. Default (no pMultisampleState) = 1 (no MSAA), so
        // existing pipelines take the single-sample fast path unchanged.
        var msaa_samples: u8 = 1;
        if (ci.pMultisampleState) |ms| msaa_samples = sampleCount(ms.rasterizationSamples);
        // Parse the color write mask + alpha-blend state from the first color-blend attachment
        // (the ICD is single-color-target). Default (no state or no attachments) = all channels
        // enabled, blending off. When blendEnable is set, src/dst factors + ops map to the HAL
        // BlendState the software raster and nvidia SET_BLEND already apply (transparency/UI/particles).
        var blend_state: prism.hal.BlendState = .{};
        if (ci.pColorBlendState) |cb| {
            if (cb.attachmentCount > 0) {
                if (cb.pAttachments) |ats| {
                    const a = ats[0];
                    const m = a.colorWriteMask;
                    blend_state.write_mask = .{
                        m & vk.VK_COLOR_COMPONENT_R_BIT != 0,
                        m & vk.VK_COLOR_COMPONENT_G_BIT != 0,
                        m & vk.VK_COLOR_COMPONENT_B_BIT != 0,
                        m & vk.VK_COLOR_COMPONENT_A_BIT != 0,
                    };
                    if (a.blendEnable != 0) {
                        blend_state.enable = true;
                        blend_state.src_color = vkBlendFactor(a.srcColorBlendFactor);
                        blend_state.dst_color = vkBlendFactor(a.dstColorBlendFactor);
                        blend_state.color_op = vkBlendOp(a.colorBlendOp);
                        blend_state.src_alpha = vkBlendFactor(a.srcAlphaBlendFactor);
                        blend_state.dst_alpha = vkBlendFactor(a.dstAlphaBlendFactor);
                        blend_state.alpha_op = vkBlendOp(a.alphaBlendOp);
                    }
                }
            }
            // The blend constant (for CONSTANT_COLOR/ALPHA factors), VkPipelineColorBlendStateCreateInfo.
            blend_state.constant = cb.blendConstants;
        }
        // Parse dynamic state: mark which stencil members come from vkCmdSetStencil* rather than
        // the baked pipeline. executeRender rebuilds a per-draw pipeline variant when any is set.
        var dyn_ref = false;
        var dyn_cmp = false;
        var dyn_wrt = false;
        if (ci.pDynamicState) |dsc| {
            if (dsc.pDynamicStates) |states| {
                for (states[0..dsc.dynamicStateCount]) |st| switch (st) {
                    vk.VK_DYNAMIC_STATE_STENCIL_REFERENCE => dyn_ref = true,
                    vk.VK_DYNAMIC_STATE_STENCIL_COMPARE_MASK => dyn_cmp = true,
                    vk.VK_DYNAMIC_STATE_STENCIL_WRITE_MASK => dyn_wrt = true,
                    else => {},
                };
            }
        }
        vtrace("createGraphicsPipeline cull mode={} front={} hasRS={} samples={}", .{ @intFromEnum(cull_state.mode), @intFromEnum(cull_state.front_face), ci.pRasterizationState != null, msaa_samples });

        const pipe = allocator.create(GraphicsPipeline) catch {
            dev.hal().destroyShaderModule(hal_vs);
            dev.hal().destroyShaderModule(hal_fs);
            return .VK_ERROR_OUT_OF_HOST_MEMORY;
        };
        pipe.* = .{ .hal_vs = hal_vs, .hal_fs = hal_fs, .hal_pipeline = undefined, .attr_count = n, .depth = depth_state, .stencil = stencil_state, .stencil_back = stencil_back, .samples = msaa_samples, .dyn_stencil_ref = dyn_ref, .dyn_stencil_compare = dyn_cmp, .dyn_stencil_write = dyn_wrt, .stride = stride, .cull = cull_state, .blend = blend_state };
        @memcpy(pipe.attrs[0..n], attrs[0..n]);

        // Record the pipeline layout's descriptor binding count + whether it declares
        // push constants. A push-constant block is bound at the slot right after the
        // descriptors (binding_count), matching the lowered shader's PC pointer-param slot.
        if (ci.layout != vk.VK_NULL_HANDLE) {
            const pl = PipelineLayout.fromHandle(ci.layout);
            pipe.pc_binding = pl.binding_count;
            pipe.has_push_constants = pl.push_constant_size > 0;
        }

        const hal_pipeline = dev.hal().createPipeline(.{
            .vertex = hal_vs,
            .fragment = hal_fs,
            .vertex_layout = .{ .stride = stride, .attributes = pipe.attrs[0..n] },
            .color_format = .rgba8_unorm,
            .depth = depth_state,
            .cull = cull_state,
            .samples = msaa_samples,
            .stencil = stencil_state,
            .stencil_back = stencil_back,
            .blend = blend_state,
        }) catch {
            dev.hal().destroyShaderModule(hal_vs);
            dev.hal().destroyShaderModule(hal_fs);
            allocator.destroy(pipe);
            return .VK_ERROR_INITIALIZATION_FAILED;
        };
        pipe.hal_pipeline = hal_pipeline;
        out[i] = pipe.toHandle();
    }
    return .VK_SUCCESS;
}

pub fn createDescriptorPool(
    device: vk.VkDevice,
    pCreateInfo: ?*const vk.VkDescriptorPoolCreateInfo,
    pAllocator: ?*const vk.VkAllocationCallbacks,
    pDescriptorPool: *vk.VkDescriptorPool,
) callconv(.c) vk.VkResult {
    _ = device;
    _ = pAllocator;
    _ = pCreateInfo;
    const pool = allocator.create(DescriptorPool) catch return .VK_ERROR_OUT_OF_HOST_MEMORY;
    pool.* = .{};
    pDescriptorPool.* = pool.toHandle();
    return .VK_SUCCESS;
}

pub fn destroyDescriptorPool(
    device: vk.VkDevice,
    descriptorPool: vk.VkDescriptorPool,
    pAllocator: ?*const vk.VkAllocationCallbacks,
) callconv(.c) void {
    _ = device;
    _ = pAllocator;
    if (descriptorPool == vk.VK_NULL_HANDLE) return;
    const pool = DescriptorPool.fromHandle(descriptorPool);
    for (pool.sets.items) |s| allocator.destroy(s);
    pool.sets.deinit(allocator);
    allocator.destroy(pool);
}

pub fn allocateDescriptorSets(
    device: vk.VkDevice,
    pAllocateInfo: ?*const vk.VkDescriptorSetAllocateInfo,
    pDescriptorSets: ?[*]vk.VkDescriptorSet,
) callconv(.c) vk.VkResult {
    _ = device;
    const ai = pAllocateInfo orelse return .VK_ERROR_INITIALIZATION_FAILED;
    const out = pDescriptorSets orelse return .VK_ERROR_INITIALIZATION_FAILED;
    const pool = DescriptorPool.fromHandle(ai.descriptorPool);
    var i: u32 = 0;
    while (i < ai.descriptorSetCount) : (i += 1) {
        const set = allocator.create(DescriptorSet) catch return .VK_ERROR_OUT_OF_HOST_MEMORY;
        set.* = .{};
        pool.sets.append(allocator, set) catch {
            allocator.destroy(set);
            return .VK_ERROR_OUT_OF_HOST_MEMORY;
        };
        out[i] = set.toHandle();
    }
    return .VK_SUCCESS;
}

pub fn updateDescriptorSets(
    device: vk.VkDevice,
    descriptorWriteCount: u32,
    pDescriptorWrites: ?[*]const vk.VkWriteDescriptorSet,
    descriptorCopyCount: u32,
    pDescriptorCopies: ?*const vk.VkCopyDescriptorSet,
) callconv(.c) void {
    _ = device;
    _ = descriptorCopyCount;
    _ = pDescriptorCopies;
    const writes = pDescriptorWrites orelse return;
    var i: u32 = 0;
    while (i < descriptorWriteCount) : (i += 1) {
        const w = writes[i];
        const set = DescriptorSet.fromHandle(w.dstSet);

        // A combined-image-sampler write: pImageInfo carries the (sampler, imageView)
        // pair per binding. The FS samples the bound image through the sampler.
        if (w.descriptorType == vk.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER) {
            const ii = w.pImageInfo orelse continue;
            const infos: [*]const vk.VkDescriptorImageInfo = @ptrCast(@alignCast(ii));
            var j: u32 = 0;
            while (j < w.descriptorCount) : (j += 1) {
                const binding = w.dstBinding + j;
                if (binding >= MAX_BINDINGS) continue;
                const info = infos[j];
                set.kinds[binding] = .combined_image_sampler;
                set.images[binding] = if (info.imageView != vk.VK_NULL_HANDLE) ImageView.fromHandle(info.imageView).image else null;
                set.samplers[binding] = if (info.sampler != vk.VK_NULL_HANDLE) Sampler.fromHandle(info.sampler) else null;
                vtrace("updateDescriptorSets CIS set=0x{x} binding={} view=0x{x} sampler=0x{x} -> img=0x{x}", .{
                    @intFromPtr(set), binding, @intFromPtr(@as(?*anyopaque, @ptrFromInt(@as(usize, @intCast(info.imageView))))), @intFromPtr(@as(?*anyopaque, @ptrFromInt(@as(usize, @intCast(info.sampler))))), if (set.images[binding]) |im| @intFromPtr(im) else 0,
                });
            }
            continue;
        }

        const kind: BindingKind = switch (w.descriptorType) {
            vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER => .storage_buffer,
            vk.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER => .uniform_buffer,
            else => continue,
        };
        const bi = w.pBufferInfo orelse continue;
        var j: u32 = 0;
        while (j < w.descriptorCount) : (j += 1) {
            const binding = w.dstBinding + j;
            if (binding >= MAX_BINDINGS) continue;
            const buf_handle = bi[j].buffer;
            if (buf_handle == vk.VK_NULL_HANDLE) {
                set.buffers[binding] = null;
                set.kinds[binding] = .none;
            } else {
                set.buffers[binding] = Buffer.fromHandle(buf_handle);
                set.kinds[binding] = kind;
            }
        }
    }
}

pub fn createCommandPool(
    device: vk.VkDevice,
    pCreateInfo: ?*const vk.VkCommandPoolCreateInfo,
    pAllocator: ?*const vk.VkAllocationCallbacks,
    pCommandPool: *vk.VkCommandPool,
) callconv(.c) vk.VkResult {
    _ = device;
    _ = pAllocator;
    _ = pCreateInfo;
    const pool = allocator.create(CommandPool) catch return .VK_ERROR_OUT_OF_HOST_MEMORY;
    pool.* = .{};
    pCommandPool.* = pool.toHandle();
    return .VK_SUCCESS;
}

pub fn destroyCommandPool(
    device: vk.VkDevice,
    commandPool: vk.VkCommandPool,
    pAllocator: ?*const vk.VkAllocationCallbacks,
) callconv(.c) void {
    _ = device;
    _ = pAllocator;
    if (commandPool == vk.VK_NULL_HANDLE) return;
    const pool = CommandPool.fromHandle(commandPool);
    for (pool.buffers.items) |cb| allocator.destroy(cb);
    pool.buffers.deinit(allocator);
    allocator.destroy(pool);
}

pub fn allocateCommandBuffers(
    device: vk.VkDevice,
    pAllocateInfo: ?*const vk.VkCommandBufferAllocateInfo,
    pCommandBuffers: ?[*]vk.VkCommandBuffer,
) callconv(.c) vk.VkResult {
    const ai = pAllocateInfo orelse return .VK_ERROR_INITIALIZATION_FAILED;
    const out = pCommandBuffers orelse return .VK_ERROR_INITIALIZATION_FAILED;
    const pool = CommandPool.fromHandle(ai.commandPool);
    var i: u32 = 0;
    while (i < ai.commandBufferCount) : (i += 1) {
        const cb = allocator.create(CommandBuffer) catch return .VK_ERROR_OUT_OF_HOST_MEMORY;
        cb.* = .{ .loader_magic = vk.ICD_LOADER_MAGIC, .device = device };
        pool.buffers.append(allocator, cb) catch {
            allocator.destroy(cb);
            return .VK_ERROR_OUT_OF_HOST_MEMORY;
        };
        out[i] = cb.toHandle();
    }
    return .VK_SUCCESS;
}

pub fn freeCommandBuffers(
    device: vk.VkDevice,
    commandPool: vk.VkCommandPool,
    commandBufferCount: u32,
    pCommandBuffers: ?[*]const vk.VkCommandBuffer,
) callconv(.c) void {
    // Command buffers are freed when their pool is destroyed. This is a no-op so
    // their pointers stay valid for the pool's bulk free (simplest correct model).
    _ = device;
    _ = commandPool;
    _ = commandBufferCount;
    _ = pCommandBuffers;
}

pub fn beginCommandBuffer(
    commandBuffer: vk.VkCommandBuffer,
    pBeginInfo: ?*const vk.VkCommandBufferBeginInfo,
) callconv(.c) vk.VkResult {
    _ = pBeginInfo;
    const cb = CommandBuffer.fromHandle(commandBuffer);
    // Reset the recorded state at begin (compute + graphics).
    cb.pipeline = null;
    cb.descriptor_set = null;
    cb.has_dispatch = false;
    cb.gfx_pipeline = null;
    cb.framebuffer = null;
    cb.vertex_buffer = null;
    cb.has_draw = false;
    cb.draw_count = 0;
    cb.pass_count = 0;
    cb.cur_pass = -1;
    cb.depth_image = null;
    cb.has_depth = false;
    cb.copy_src_image = null;
    cb.copy_dst_buffer = null;
    cb.has_copy = false;
    cb.buf_copy_count = 0;
    cb.pc_dirty = false;
    return .VK_SUCCESS;
}

pub fn endCommandBuffer(
    commandBuffer: vk.VkCommandBuffer,
) callconv(.c) vk.VkResult {
    _ = commandBuffer;
    return .VK_SUCCESS;
}

/// vkResetCommandBuffer: a standard render loop resets + re-records each frame.
/// Recording state is also cleared in beginCommandBuffer, so clear it here too.
pub fn resetCommandBuffer(
    commandBuffer: vk.VkCommandBuffer,
    flags: vk.VkCommandBufferResetFlags,
) callconv(.c) vk.VkResult {
    _ = flags;
    const cb = CommandBuffer.fromHandle(commandBuffer);
    cb.pipeline = null;
    cb.descriptor_set = null;
    cb.has_dispatch = false;
    cb.gfx_pipeline = null;
    cb.framebuffer = null;
    cb.vertex_buffer = null;
    cb.has_draw = false;
    cb.draw_count = 0;
    cb.pass_count = 0;
    cb.cur_pass = -1;
    cb.depth_image = null;
    cb.has_depth = false;
    cb.copy_src_image = null;
    cb.copy_dst_buffer = null;
    cb.has_copy = false;
    cb.buf_copy_count = 0;
    cb.pc_dirty = false;
    return .VK_SUCCESS;
}

/// vkResetCommandPool: pools allocate fixed CommandBuffer structs that are reset
/// per-begin, so resetting the pool is a no-op success here.
pub fn resetCommandPool(
    device: vk.VkDevice,
    commandPool: vk.VkCommandPool,
    flags: vk.VkCommandPoolResetFlags,
) callconv(.c) vk.VkResult {
    _ = device;
    _ = commandPool;
    _ = flags;
    return .VK_SUCCESS;
}

pub fn cmdBindPipeline(
    commandBuffer: vk.VkCommandBuffer,
    pipelineBindPoint: vk.VkPipelineBindPoint,
    pipeline: vk.VkPipeline,
) callconv(.c) void {
    const cb = CommandBuffer.fromHandle(commandBuffer);
    // Ignore a null pipeline handle (e.g. an app that binds VK_NULL_HANDLE, or a device
    // whose createGraphicsPipelines failed): the tag-mask + ptrFromInt would otherwise
    // fault on a null cast.
    if (pipeline == vk.VK_NULL_HANDLE) return;
    if (pipelineBindPoint == vk.VK_PIPELINE_BIND_POINT_COMPUTE) {
        cb.pipeline = ComputePipeline.fromHandle(pipeline);
    } else if (pipelineBindPoint == vk.VK_PIPELINE_BIND_POINT_GRAPHICS) {
        cb.gfx_pipeline = GraphicsPipeline.fromHandle(pipeline);
    }
}

pub fn cmdBindDescriptorSets(
    commandBuffer: vk.VkCommandBuffer,
    pipelineBindPoint: vk.VkPipelineBindPoint,
    layout: vk.VkPipelineLayout,
    firstSet: u32,
    descriptorSetCount: u32,
    pDescriptorSets: ?[*]const vk.VkDescriptorSet,
    dynamicOffsetCount: u32,
    pDynamicOffsets: ?[*]const u32,
) callconv(.c) void {
    _ = layout;
    _ = firstSet;
    _ = dynamicOffsetCount;
    _ = pDynamicOffsets;
    _ = pipelineBindPoint; // record the set for both compute (SSBO) and graphics (UBO)
    const cb = CommandBuffer.fromHandle(commandBuffer);
    const sets = pDescriptorSets orelse return;
    if (descriptorSetCount >= 1) cb.descriptor_set = DescriptorSet.fromHandle(sets[0]);
}

pub fn cmdDispatch(
    commandBuffer: vk.VkCommandBuffer,
    groupCountX: u32,
    groupCountY: u32,
    groupCountZ: u32,
) callconv(.c) void {
    const cb = CommandBuffer.fromHandle(commandBuffer);
    cb.group_x = groupCountX;
    cb.group_y = groupCountY;
    cb.group_z = groupCountZ;
    cb.has_dispatch = true;
}

// Graphics command recording (M4).

pub fn cmdBeginRenderPass(
    commandBuffer: vk.VkCommandBuffer,
    pRenderPassBegin: ?*const vk.VkRenderPassBeginInfo,
    contents: vk.VkSubpassContents,
) callconv(.c) void {
    _ = contents;
    const cb = CommandBuffer.fromHandle(commandBuffer);
    const bi = pRenderPassBegin orelse return;
    var depth_index: u32 = std.math.maxInt(u32);
    var rp_samples: u8 = 1;
    if (bi.renderPass != vk.VK_NULL_HANDLE) {
        const rp = RenderPass.fromHandle(bi.renderPass);
        if (rp.has_depth) depth_index = rp.depth_index;
        rp_samples = rp.samples;
    }

    // Open a new render-pass instance. Each instance owns its own framebuffer + clear +
    // depth + a slice of the shared `draws` array (subsequent cmdDraw appends to it). This
    // is what lets one command buffer record multiple passes (render into a texture, then
    // sample it). The draws this instance owns start at the current shared draw count.
    var inst: RenderPassInstance = .{
        .framebuffer = null,
        .clear_r = 0,
        .clear_g = 0,
        .clear_b = 0,
        .clear_a = 1,
        .depth_image = null,
        .depth_clear = 1.0,
        .has_depth = false,
        .draw_start = cb.draw_count,
        .draw_count = 0,
    };

    if (bi.framebuffer != vk.VK_NULL_HANDLE) {
        const fb = Framebuffer.fromHandle(bi.framebuffer);
        inst.framebuffer = fb;
        // MSAA: when the render pass declares a resolve attachment, record the resolve
        // target image + sample count so the pass resolves (box-averages) the
        // multisampled color attachment into it at the end of the pass.
        if (fb.resolve_view) |rv| {
            inst.resolve_image = rv.image;
            inst.samples = if (fb.view.image.samples > 1) fb.view.image.samples else rp_samples;
        } else if (fb.view.image.samples > 1) {
            // A multisampled color attachment with no explicit resolve view: still record
            // the sample count. The pass renders N samples. Without a resolve target the
            // resolved result is unavailable, but the count keeps the pipeline consistent.
            inst.samples = fb.view.image.samples;
        }
        // Keep the flat fields in sync with the most-recently-opened pass so any code
        // still reading them (legacy) sees the current framebuffer.
        cb.framebuffer = fb;
        // The depth attachment for this render pass (if any) comes from the
        // framebuffer's depth view.
        if (fb.depth_view) |dv| {
            inst.depth_image = dv.image;
            inst.has_depth = true;
            cb.depth_image = dv.image;
            cb.has_depth = true;
        } else {
            cb.depth_image = null;
            cb.has_depth = false;
        }
    }
    // The clear color (attachment 0). loadOp CLEAR + storeOp STORE is the only
    // mode the software path models, so we always clear to this then store.
    if (bi.pClearValues) |cv| {
        if (bi.clearValueCount >= 1) {
            const c = cv[0].color.float32;
            inst.clear_r = c[0];
            inst.clear_g = c[1];
            inst.clear_b = c[2];
            inst.clear_a = c[3];
            cb.clear_r = c[0];
            cb.clear_g = c[1];
            cb.clear_b = c[2];
            cb.clear_a = c[3];
        }
        // The depth clear value: pClearValues indexes by attachment index, so the depth
        // attachment's clear is at pClearValues[depth_index].depthStencil.depth.
        if (inst.has_depth and depth_index != std.math.maxInt(u32) and depth_index < bi.clearValueCount) {
            inst.depth_clear = cv[depth_index].depthStencil.depth;
            cb.depth_clear = inst.depth_clear;
            inst.stencil_clear = @truncate(cv[depth_index].depthStencil.stencil);
        }
    }

    if (cb.pass_count < MAX_RENDER_PASSES) {
        cb.passes[cb.pass_count] = inst;
        cb.cur_pass = @intCast(cb.pass_count);
        cb.pass_count += 1;
    } else {
        // Out of instance slots: leave cur_pass at -1 so cmdDraw falls back to the flat
        // single-pass fields (degraded but safe rather than corrupting an instance).
        cb.cur_pass = -1;
    }
}

pub fn cmdEndRenderPass(commandBuffer: vk.VkCommandBuffer) callconv(.c) void {
    // Close the current instance: its draw slice is fixed. Later draws need a new
    // vkCmdBeginRenderPass. The pass executes at submit. Nothing to flush here.
    const cb = CommandBuffer.fromHandle(commandBuffer);
    cb.cur_pass = -1;
}

/// vkCmdPipelineBarrier: the software path has no async GPU pipeline (submit is
/// synchronous + the framebuffer image is the swapchain image), so layout/memory
/// barriers are a recording no-op. Standard apps (vkcube) emit these for image
/// layout transitions. Honoring them as no-ops keeps the linear flow correct.
pub fn cmdPipelineBarrier(
    commandBuffer: vk.VkCommandBuffer,
    srcStageMask: vk.VkFlags,
    dstStageMask: vk.VkFlags,
    dependencyFlags: vk.VkFlags,
    memoryBarrierCount: u32,
    pMemoryBarriers: ?*const anyopaque,
    bufferMemoryBarrierCount: u32,
    pBufferMemoryBarriers: ?*const anyopaque,
    imageMemoryBarrierCount: u32,
    pImageMemoryBarriers: ?*const anyopaque,
) callconv(.c) void {
    _ = commandBuffer;
    _ = srcStageMask;
    _ = dstStageMask;
    _ = dependencyFlags;
    _ = memoryBarrierCount;
    _ = pMemoryBarriers;
    _ = bufferMemoryBarrierCount;
    _ = pBufferMemoryBarriers;
    _ = imageMemoryBarrierCount;
    _ = pImageMemoryBarriers;
}

/// vkCreatePipelineCache / vkDestroyPipelineCache: Prism builds pipelines eagerly
/// and caches nothing, so a cache is an opaque non-null token. Standard apps
/// (vkcube) always create one before vkCreateGraphicsPipelines.
pub fn createPipelineCache(
    device: vk.VkDevice,
    pCreateInfo: ?*const anyopaque,
    pAllocator: ?*const vk.VkAllocationCallbacks,
    pPipelineCache: *vk.VkPipelineCache,
) callconv(.c) vk.VkResult {
    _ = device;
    _ = pCreateInfo;
    _ = pAllocator;
    // A stable non-null token (apps may assert non-null). Never dereferenced.
    pPipelineCache.* = 0x9CACE000;
    return .VK_SUCCESS;
}

pub fn destroyPipelineCache(
    device: vk.VkDevice,
    pipelineCache: vk.VkPipelineCache,
    pAllocator: ?*const vk.VkAllocationCallbacks,
) callconv(.c) void {
    _ = device;
    _ = pipelineCache;
    _ = pAllocator;
}

pub fn cmdBindVertexBuffers(
    commandBuffer: vk.VkCommandBuffer,
    firstBinding: u32,
    bindingCount: u32,
    pBuffers: ?[*]const vk.VkBuffer,
    pOffsets: ?[*]const vk.VkDeviceSize,
) callconv(.c) void {
    _ = firstBinding;
    _ = pOffsets;
    const cb = CommandBuffer.fromHandle(commandBuffer);
    const bufs = pBuffers orelse return;
    if (bindingCount >= 1 and bufs[0] != vk.VK_NULL_HANDLE) cb.vertex_buffer = Buffer.fromHandle(bufs[0]);
}

pub fn cmdSetViewport(
    commandBuffer: vk.VkCommandBuffer,
    firstViewport: u32,
    viewportCount: u32,
    pViewports: ?[*]const vk.VkViewport,
) callconv(.c) void {
    // The software rasterizer maps NDC to the full render target, so a dynamic
    // viewport that matches the framebuffer is a no-op. Accepted for API completeness.
    _ = commandBuffer;
    _ = firstViewport;
    _ = viewportCount;
    _ = pViewports;
}

pub fn cmdSetScissor(
    commandBuffer: vk.VkCommandBuffer,
    firstScissor: u32,
    scissorCount: u32,
    pScissors: ?[*]const vk.VkRect2D,
) callconv(.c) void {
    const cb = CommandBuffer.fromHandle(commandBuffer);
    // We track a single scissor (the common case; firstScissor 0). The rect is captured
    // per-draw at cmdDraw and applied via the HAL before that draw.
    _ = firstScissor;
    if (scissorCount > 0) {
        if (pScissors) |s| {
            cb.scissor = s[0];
            cb.has_scissor = true;
        }
    }
}

/// vkCmdSetStencilReference: set the dynamic stencil reference for the given face(s). Applies to
/// draws whose pipeline declared VK_DYNAMIC_STATE_STENCIL_REFERENCE. Snapshotted per-draw.
pub fn cmdSetStencilReference(commandBuffer: vk.VkCommandBuffer, faceMask: vk.VkStencilFaceFlags, reference: u32) callconv(.c) void {
    const cb = CommandBuffer.fromHandle(commandBuffer);
    const v: u8 = @truncate(reference);
    if (faceMask & vk.VK_STENCIL_FACE_FRONT_BIT != 0) cb.st_ref_front = v;
    if (faceMask & vk.VK_STENCIL_FACE_BACK_BIT != 0) cb.st_ref_back = v;
}

/// vkCmdSetStencilCompareMask: set the dynamic stencil compare mask for the given face(s).
pub fn cmdSetStencilCompareMask(commandBuffer: vk.VkCommandBuffer, faceMask: vk.VkStencilFaceFlags, compareMask: u32) callconv(.c) void {
    const cb = CommandBuffer.fromHandle(commandBuffer);
    const v: u8 = @truncate(compareMask);
    if (faceMask & vk.VK_STENCIL_FACE_FRONT_BIT != 0) cb.st_cmp_front = v;
    if (faceMask & vk.VK_STENCIL_FACE_BACK_BIT != 0) cb.st_cmp_back = v;
}

/// vkCmdSetStencilWriteMask: set the dynamic stencil write mask for the given face(s).
pub fn cmdSetStencilWriteMask(commandBuffer: vk.VkCommandBuffer, faceMask: vk.VkStencilFaceFlags, writeMask: u32) callconv(.c) void {
    const cb = CommandBuffer.fromHandle(commandBuffer);
    const v: u8 = @truncate(writeMask);
    if (faceMask & vk.VK_STENCIL_FACE_FRONT_BIT != 0) cb.st_wrt_front = v;
    if (faceMask & vk.VK_STENCIL_FACE_BACK_BIT != 0) cb.st_wrt_back = v;
}

pub fn cmdDraw(
    commandBuffer: vk.VkCommandBuffer,
    vertexCount: u32,
    instanceCount: u32,
    firstVertex: u32,
    firstInstance: u32,
) callconv(.c) void {
    const cb = CommandBuffer.fromHandle(commandBuffer);
    cb.draw_vertex_count = vertexCount;
    cb.draw_first_vertex = firstVertex;
    cb.has_draw = true;
    // Record this draw with the currently bound vertex buffer + pipeline + set so a
    // render pass with multiple draws replays them in order (depth across draws). The
    // vertex buffer may be null for a vertex-pulling draw (the VS pulls vertices from a
    // bound UBO by gl_VertexIndex with no vertex buffer bound). Only a pipeline is required.
    if (cb.draw_count < MAX_GFX_DRAWS) {
        if (cb.gfx_pipeline) |gp| {
            var draw: GfxDraw = .{
                .vertex_buffer = cb.vertex_buffer,
                .pipeline = gp,
                .descriptor_set = cb.descriptor_set,
                .vertex_count = vertexCount,
                .first_vertex = firstVertex,
                .instance_count = instanceCount,
                .first_instance = firstInstance,
                .push_constants = undefined,
                .has_push_constants = gp.has_push_constants and cb.pc_dirty,
                .scissor = cb.scissor,
                .has_scissor = cb.has_scissor,
                // Snapshot the live dynamic stencil masks when this pipeline declares any dynamic.
                .dyn_stencil = gp.dyn_stencil_ref or gp.dyn_stencil_compare or gp.dyn_stencil_write,
                .st_ref_front = cb.st_ref_front,
                .st_ref_back = cb.st_ref_back,
                .st_cmp_front = cb.st_cmp_front,
                .st_cmp_back = cb.st_cmp_back,
                .st_wrt_front = cb.st_wrt_front,
                .st_wrt_back = cb.st_wrt_back,
            };
            // Snapshot the current push-constant block: push constants are draw-time state,
            // so each draw keeps its own copy (a later push does not change this draw).
            @memcpy(&draw.push_constants, &cb.push_constants);
            cb.draws[cb.draw_count] = draw;
            cb.draw_count += 1;
            // Attribute this draw to the currently-open render-pass instance so executeRender
            // replays it into that instance's framebuffer. When no instance is open (cur_pass
            // < 0, e.g. the out-of-slots fallback) the flat single-pass path picks it up.
            if (cb.cur_pass >= 0) {
                const pi: usize = @intCast(cb.cur_pass);
                cb.passes[pi].draw_count += 1;
            }
        }
    }
}

/// vkCmdBindIndexBuffer: bind the index buffer for subsequent vkCmdDrawIndexed.
/// `indexType` is UINT16 or UINT32. `offset` is a byte offset into the buffer.
pub fn cmdBindIndexBuffer(
    commandBuffer: vk.VkCommandBuffer,
    buffer: vk.VkBuffer,
    offset: vk.VkDeviceSize,
    indexType: vk.VkIndexType,
) callconv(.c) void {
    const cb = CommandBuffer.fromHandle(commandBuffer);
    cb.index_buffer = if (buffer == vk.VK_NULL_HANDLE) null else Buffer.fromHandle(buffer);
    cb.index_offset = offset;
    cb.index_u32 = indexType == vk.VK_INDEX_TYPE_UINT32;
}

/// vkCmdDrawIndexed: draw `indexCount` indices from the bound index buffer, each selecting a vertex
/// (index + vertexOffset) from the bound vertex buffer. Recorded like cmdDraw but flagged indexed.
/// executeRender expands the indices into a vertex stream (the HAL draws non-indexed). Attribute-fed
/// only. A vertex-pulling indexed draw is not expanded. gl_VertexIndex in the VS sees the sequential
/// position, not the index value, which is fine for the usual attribute-reading shaders.
pub fn cmdDrawIndexed(
    commandBuffer: vk.VkCommandBuffer,
    indexCount: u32,
    instanceCount: u32,
    firstIndex: u32,
    vertexOffset: i32,
    firstInstance: u32,
) callconv(.c) void {
    const cb = CommandBuffer.fromHandle(commandBuffer);
    cb.has_draw = true;
    if (cb.draw_count < MAX_GFX_DRAWS) {
        if (cb.gfx_pipeline) |gp| {
            var draw: GfxDraw = .{
                .vertex_buffer = cb.vertex_buffer,
                .pipeline = gp,
                .descriptor_set = cb.descriptor_set,
                .vertex_count = indexCount, // index count for the indexed path
                .first_vertex = firstIndex, // reused as the firstIndex offset
                .instance_count = instanceCount,
                .first_instance = firstInstance,
                .push_constants = undefined,
                .has_push_constants = gp.has_push_constants and cb.pc_dirty,
                .scissor = cb.scissor,
                .has_scissor = cb.has_scissor,
                .dyn_stencil = gp.dyn_stencil_ref or gp.dyn_stencil_compare or gp.dyn_stencil_write,
                .st_ref_front = cb.st_ref_front,
                .st_ref_back = cb.st_ref_back,
                .st_cmp_front = cb.st_cmp_front,
                .st_cmp_back = cb.st_cmp_back,
                .st_wrt_front = cb.st_wrt_front,
                .st_wrt_back = cb.st_wrt_back,
                .is_indexed = true,
                .index_buffer = cb.index_buffer,
                .index_offset = cb.index_offset,
                .index_u32 = cb.index_u32,
                .vertex_offset = vertexOffset,
            };
            @memcpy(&draw.push_constants, &cb.push_constants);
            cb.draws[cb.draw_count] = draw;
            cb.draw_count += 1;
            if (cb.cur_pass >= 0) {
                const pi: usize = @intCast(cb.cur_pass);
                cb.passes[pi].draw_count += 1;
            }
        }
    }
}

/// vkCmdPushConstants: write `size` bytes from `pValues` into the command buffer's
/// push-constant block at `offset`. The bytes are draw-time state. A later cmdDraw
/// snapshots the current block so a subsequent push does not affect an earlier draw.
/// executeRender threads each draw's snapshot into the shader as a UBO-like pointer
/// param at the pipeline's push-constant binding slot (right after the descriptors).
pub fn cmdPushConstants(
    commandBuffer: vk.VkCommandBuffer,
    layout: vk.VkPipelineLayout,
    stageFlags: vk.VkShaderStageFlags,
    offset: u32,
    size: u32,
    pValues: ?*const anyopaque,
) callconv(.c) void {
    _ = layout;
    _ = stageFlags;
    const cb = CommandBuffer.fromHandle(commandBuffer);
    const src = pValues orelse return;
    if (offset >= PUSH_CONSTANT_SIZE) return;
    const n = @min(size, PUSH_CONSTANT_SIZE - offset);
    const bytes: [*]const u8 = @ptrCast(src);
    @memcpy(cb.push_constants[offset..][0..n], bytes[0..n]);
    cb.pc_dirty = true;
}

/// vkCmdCopyBuffer: record buffer-to-buffer copy regions (the staging-upload pattern: an app maps a
/// host-visible staging buffer, writes vertex/index/uniform data, then copies it into the actual
/// buffer). Executed at submit (executeBufferCopies) before draws, so the data is in place when
/// draws read it. Overflowing MAX_BUF_COPIES silently drops extra regions.
pub fn cmdCopyBuffer(
    commandBuffer: vk.VkCommandBuffer,
    srcBuffer: vk.VkBuffer,
    dstBuffer: vk.VkBuffer,
    regionCount: u32,
    pRegions: ?[*]const vk.VkBufferCopy,
) callconv(.c) void {
    const cb = CommandBuffer.fromHandle(commandBuffer);
    if (srcBuffer == vk.VK_NULL_HANDLE or dstBuffer == vk.VK_NULL_HANDLE) return;
    const regions = pRegions orelse return;
    const src = Buffer.fromHandle(srcBuffer);
    const dst = Buffer.fromHandle(dstBuffer);
    var i: u32 = 0;
    while (i < regionCount and cb.buf_copy_count < MAX_BUF_COPIES) : (i += 1) {
        cb.buf_copies[cb.buf_copy_count] = .{
            .src = src,
            .dst = dst,
            .src_offset = regions[i].srcOffset,
            .dst_offset = regions[i].dstOffset,
            .size = regions[i].size,
        };
        cb.buf_copy_count += 1;
    }
}

/// Execute the recorded vkCmdCopyBuffer regions: memcpy each src range into the dst buffer (both are
/// host-mappable HAL resources). Best-effort: a region referencing an unbound buffer or exceeding a
/// buffer's bytes is clamped/skipped rather than faulting.
fn executeBufferCopies(dev: *LogicalDevice, cb: *CommandBuffer) void {
    var i: usize = 0;
    while (i < cb.buf_copy_count) : (i += 1) {
        const c = cb.buf_copies[i];
        const smem = c.src.memory orelse continue;
        const dmem = c.dst.memory orelse continue;
        const sbytes = dev.hal().mapResource(smem.resource) catch continue;
        const dbytes = dev.hal().mapResource(dmem.resource) catch continue;
        if (c.src_offset >= sbytes.len or c.dst_offset >= dbytes.len) continue;
        const n = @min(c.size, @min(sbytes.len - c.src_offset, dbytes.len - c.dst_offset));
        @memcpy(dbytes[@intCast(c.dst_offset)..][0..@intCast(n)], sbytes[@intCast(c.src_offset)..][0..@intCast(n)]);
    }
}

/// vkCmdCopyBufferToImage: fill a sampled image's texels from a staging buffer. The
/// software submit is synchronous and the staging buffer is already populated (mapped +
/// written by the app before recording), so the copy runs immediately and the image
/// Resource is ready for the FS to sample at draw time. Honors bufferRowLength
/// (0 = tightly packed to imageExtent.width) and imageExtent. RGBA8 (4 bytes/texel).
pub fn cmdCopyBufferToImage(
    commandBuffer: vk.VkCommandBuffer,
    srcBuffer: vk.VkBuffer,
    dstImage: vk.VkImage,
    dstImageLayout: vk.VkImageLayout,
    regionCount: u32,
    pRegions: ?[*]const vk.VkBufferImageCopy,
) callconv(.c) void {
    _ = dstImageLayout;
    const cb = CommandBuffer.fromHandle(commandBuffer);
    const dev = LogicalDevice.fromHandle(cb.device);
    if (srcBuffer == vk.VK_NULL_HANDLE or dstImage == vk.VK_NULL_HANDLE) return;
    const buf = Buffer.fromHandle(srcBuffer);
    const img = Image.fromHandle(dstImage);
    const regions = pRegions orelse return;
    vtrace("cmdCopyBufferToImage src=0x{x} dst=0x{x} {}x{} regions={} mem={} res={}", .{
        @intFromPtr(buf), @intFromPtr(img), img.width, img.height, regionCount, buf.memory != null, img.resource != null,
    });
    const mem = buf.memory orelse return;
    const dst_res = img.resource orelse return;
    const src_bytes = dev.hal().mapResource(mem.resource) catch return;
    const dst_bytes = dev.hal().mapResource(dst_res) catch return;
    const dst_pitch = @as(usize, img.width) * 4;

    var r: u32 = 0;
    while (r < regionCount) : (r += 1) {
        const reg = regions[r];
        const w = if (reg.imageExtent.width == 0) img.width else @min(reg.imageExtent.width, img.width);
        const h = if (reg.imageExtent.height == 0) img.height else @min(reg.imageExtent.height, img.height);
        // Source row stride: bufferRowLength texels (0 = tightly packed = w).
        const src_row_texels: usize = if (reg.bufferRowLength != 0) reg.bufferRowLength else w;
        const src_row_bytes = src_row_texels * 4;
        const x0: usize = @intCast(@max(reg.imageOffset.x, 0));
        const y0: usize = @intCast(@max(reg.imageOffset.y, 0));
        var y: usize = 0;
        while (y < h) : (y += 1) {
            const src_off = @as(usize, @intCast(reg.bufferOffset)) + y * src_row_bytes;
            const dst_off = (y0 + y) * dst_pitch + x0 * 4;
            const n = w * 4;
            if (src_off + n > src_bytes.len or dst_off + n > dst_bytes.len) break;
            @memcpy(dst_bytes[dst_off..][0..n], src_bytes[src_off..][0..n]);
        }
    }
}

pub fn cmdCopyImageToBuffer(
    commandBuffer: vk.VkCommandBuffer,
    srcImage: vk.VkImage,
    srcImageLayout: vk.VkImageLayout,
    dstBuffer: vk.VkBuffer,
    regionCount: u32,
    pRegions: ?[*]const vk.VkBufferImageCopy,
) callconv(.c) void {
    _ = srcImageLayout;
    _ = regionCount;
    _ = pRegions;
    const cb = CommandBuffer.fromHandle(commandBuffer);
    if (srcImage != vk.VK_NULL_HANDLE) cb.copy_src_image = Image.fromHandle(srcImage);
    if (dstBuffer != vk.VK_NULL_HANDLE) cb.copy_dst_buffer = Buffer.fromHandle(dstBuffer);
    cb.has_copy = true;
}

/// Bind a descriptor set's uniform buffers (the MVP UBO) onto a HAL command buffer
/// by binding index, so the JITed VS/FS receives each UBO's base pointer (std140).
fn bindUniformBuffers(hcb: prism.hal.CommandBuffer, set: *DescriptorSet) prism.hal.Error!void {
    var b: u32 = 0;
    while (b < MAX_BINDINGS) : (b += 1) {
        switch (set.kinds[b]) {
            .uniform_buffer => {
                const ubuf = set.buffers[b] orelse continue;
                const umem = ubuf.memory orelse continue;
                try hcb.bindUniformBuffer(b, umem.resource);
            },
            .combined_image_sampler => {
                // Bind the texture (image Resource + sampler state) for the FS to sample.
                const img = set.images[b] orelse continue;
                const res = img.resource orelse continue;
                const samp = set.samplers[b];
                try hcb.bindTexture(.{
                    .binding = b,
                    .image = res,
                    .filter = if (samp) |s| s.filter else .nearest,
                    .address_u = if (samp) |s| s.address_u else .repeat,
                    .address_v = if (samp) |s| s.address_v else .repeat,
                    .max_anisotropy = if (samp) |s| s.max_anisotropy else 1,
                });
            },
            else => {},
        }
    }
}

/// Replay one render-pass instance: clear its color (and optional depth) attachment, then
/// run the draws in [draw_start, draw_start+draw_count) in order into its framebuffer's
/// image Resource via one HAL command buffer. All draws share the same color and depth
/// attachment (so depth testing across draws works). Each instance has its own framebuffer
/// image, which is how an earlier pass renders into a texture a later pass then samples
/// (the offscreen image Resource pixels written here are exactly what the later sampler reads).
fn executePassInstance(
    dev: *LogicalDevice,
    cb: *CommandBuffer,
    target: *prism.hal.Resource,
    clear: prism.hal.Color,
    depth_image: ?*Image,
    has_depth: bool,
    depth_clear: f32,
    stencil_clear: u8,
    draw_start: usize,
    draw_count: usize,
    // MSAA: when `resolve_image` is non-null, the multisampled `target` (rendered at
    // `samples` samples/pixel) is box-averaged into the resolve image at the pass end.
    resolve_image: ?*Image,
    samples: u8,
    width: u32,
    height: u32,
    color_format: prism.hal.Format,
) vk.VkResult {
    const ctx = dev.hal().createContext() catch return .VK_ERROR_DEVICE_LOST;
    defer ctx.deinit();
    const hcb = ctx.beginCommands() catch return .VK_ERROR_DEVICE_LOST;
    defer hcb.deinit();
    hcb.setRenderTarget(target) catch return .VK_ERROR_DEVICE_LOST;
    hcb.clear(clear) catch return .VK_ERROR_DEVICE_LOST;
    // Bind + clear the depth attachment when the render pass has one. The HAL
    // setDepthTarget is optional/null on non-software drivers (no-op there). The
    // software driver allocates/clears its f32 depth buffer.
    if (has_depth) {
        if (depth_image) |di| {
            if (di.resource) |dres| {
                hcb.setDepthTarget(dres, depth_clear) catch return .VK_ERROR_DEVICE_LOST;
            }
        }
    }
    // Stencil attachment: if any draw in this instance uses stencil, bind the depth/stencil
    // image's u8 stencil buffer (lazily allocated) and clear it to the pass's stencil clear
    // value. Allocated only when needed, so non-stencil passes (e.g. vkcube) pay nothing.
    if (depth_image) |di| {
        var uses_stencil = false;
        var si: usize = draw_start;
        const send = draw_start + draw_count;
        while (si < send and si < MAX_GFX_DRAWS) : (si += 1) {
            if (cb.draws[si].pipeline.stencil.test_enable) {
                uses_stencil = true;
                break;
            }
        }
        if (uses_stencil) {
            if (di.stencilResource(dev.hal())) |sres| {
                hcb.setStencilTarget(sres, stencil_clear) catch return .VK_ERROR_DEVICE_LOST;
            }
        }
    }

    // Per-draw push-constant scratch HAL resources (kept valid until submit returns).
    var pc_res: [MAX_GFX_DRAWS]?*prism.hal.Resource = .{null} ** MAX_GFX_DRAWS;
    defer for (pc_res) |r| {
        if (r) |res| dev.hal().destroyResource(res);
    };
    // Per-draw expanded-index vertex streams (vkCmdDrawIndexed): indexed vertices gathered into a
    // flat non-indexed buffer, kept valid until submit returns.
    var idx_res: [MAX_GFX_DRAWS]?*prism.hal.Resource = .{null} ** MAX_GFX_DRAWS;
    defer for (idx_res) |r| {
        if (r) |res| dev.hal().destroyResource(res);
    };
    // Per-draw HAL pipeline variants for dynamic stencil (vkCmdSetStencil*): a draw whose pipeline
    // declares dynamic stencil ref/masks needs a HAL pipeline carrying the live values (the HAL
    // bakes stencil into the pipeline), rebuilt here and freed after submit. Null for static draws.
    var dyn_pipes: [MAX_GFX_DRAWS]?*prism.hal.Pipeline = .{null} ** MAX_GFX_DRAWS;
    defer for (dyn_pipes) |p| {
        if (p) |pl| dev.hal().destroyPipeline(pl);
    };

    var i: usize = draw_start;
    const end = draw_start + draw_count;
    while (i < end and i < MAX_GFX_DRAWS) : (i += 1) {
        const d = cb.draws[i];
        // Dynamic stencil: build a variant pipeline with the snapshotted ref/compare/write masks
        // overriding the pipeline's baked stencil (only the members marked dynamic change; a
        // masked-off member keeps the pipeline's value). Otherwise use the baked pipeline.
        var draw_pipe = d.pipeline.hal_pipeline;
        if (d.dyn_stencil and d.pipeline.stencil.test_enable) {
            const gp = d.pipeline;
            var front = gp.stencil;
            if (gp.dyn_stencil_ref) front.reference = d.st_ref_front;
            if (gp.dyn_stencil_compare) front.compare_mask = d.st_cmp_front;
            if (gp.dyn_stencil_write) front.write_mask = d.st_wrt_front;
            var back = gp.stencil_back;
            if (back) |*b| {
                if (gp.dyn_stencil_ref) b.reference = d.st_ref_back;
                if (gp.dyn_stencil_compare) b.compare_mask = d.st_cmp_back;
                if (gp.dyn_stencil_write) b.write_mask = d.st_wrt_back;
            }
            const variant = dev.hal().createPipeline(.{
                .vertex = gp.hal_vs,
                .fragment = gp.hal_fs,
                .vertex_layout = .{ .stride = gp.stride, .attributes = gp.attrs[0..gp.attr_count] },
                .color_format = .rgba8_unorm,
                .depth = gp.depth,
                .cull = gp.cull,
                .samples = gp.samples,
                .stencil = front,
                .stencil_back = back,
                .blend = gp.blend,
            }) catch return .VK_ERROR_DEVICE_LOST;
            dyn_pipes[i - draw_start] = variant;
            draw_pipe = variant;
        }
        hcb.bindPipeline(draw_pipe) catch return .VK_ERROR_DEVICE_LOST;
        if (d.is_indexed) {
            // Expand the indexed draw into a flat non-indexed vertex stream (the HAL has no indexed
            // draw), bind that, and draw index_count vertices. Attribute-fed only.
            const expanded = expandIndexedDraw(dev, d) orelse return .VK_ERROR_DEVICE_LOST;
            idx_res[i - draw_start] = expanded;
            hcb.bindVertexBuffer(expanded) catch return .VK_ERROR_DEVICE_LOST;
        } else if (d.vertex_buffer) |vb| {
            const vmem = vb.memory orelse return .VK_ERROR_INITIALIZATION_FAILED;
            hcb.bindVertexBuffer(vmem.resource) catch return .VK_ERROR_DEVICE_LOST;
        }
        if (d.descriptor_set) |set| bindUniformBuffers(hcb, set) catch return .VK_ERROR_DEVICE_LOST;
        if (d.has_push_constants) {
            const res = dev.hal().createResource(.{ .buffer = .{
                .size = PUSH_CONSTANT_SIZE,
                .usage = .{ .copy_dst = true, .copy_src = true },
            } }) catch return .VK_ERROR_DEVICE_LOST;
            pc_res[i - draw_start] = res;
            const pcbytes = dev.hal().mapResource(res) catch return .VK_ERROR_DEVICE_LOST;
            @memcpy(pcbytes[0..PUSH_CONSTANT_SIZE], &d.push_constants);
            hcb.bindUniformBuffer(d.pipeline.pc_binding, res) catch return .VK_ERROR_DEVICE_LOST;
        }
        // Scissor: Vulkan's VkRect2D is framebuffer-pixel top-left origin, matching the HAL
        // convention exactly, so it maps through directly. has_scissor false -> null (no clipping).
        if (d.has_scissor) {
            hcb.setScissor(.{
                .x = d.scissor.offset.x,
                .y = d.scissor.offset.y,
                .width = d.scissor.extent.width,
                .height = d.scissor.extent.height,
            }) catch return .VK_ERROR_DEVICE_LOST;
        } else {
            hcb.setScissor(null) catch return .VK_ERROR_DEVICE_LOST;
        }
        // Indexed: the expanded stream is already in index order, so draw from vertex 0.
        const fv: u32 = if (d.is_indexed) 0 else d.first_vertex;
        hcb.drawInstanced(d.vertex_count, d.instance_count, fv, d.first_instance) catch return .VK_ERROR_DEVICE_LOST;
    }
    // MSAA resolve: box-average the N samples of the multisampled target into the
    // single-sample resolve image, recorded as the last command in this pass's HAL
    // command buffer (so it runs after every draw has written the sample buffer).
    if (samples > 1) {
        if (resolve_image) |ri| {
            if (ri.resource) |rres| {
                hcb.resolve(target, rres, width, height, color_format, samples) catch return .VK_ERROR_DEVICE_LOST;
            }
        }
    }
    ctx.submit(hcb) catch return .VK_ERROR_DEVICE_LOST;
    return .VK_SUCCESS;
}

/// Expand a vkCmdDrawIndexed into a flat, non-indexed vertex stream: for each of the draw's
/// index_count indices (from the bound index buffer at first_vertex=firstIndex), gather the vertex
/// (index + vertexOffset) from the bound vertex buffer into a fresh HAL vertex buffer in draw order.
/// The caller binds it and draws index_count vertices non-indexed. Returns null on any failure (a
/// missing vertex/index buffer, e.g. an unsupported vertex-pulling indexed draw). The caller frees it.
fn expandIndexedDraw(dev: *LogicalDevice, d: GfxDraw) ?*prism.hal.Resource {
    const vb = d.vertex_buffer orelse return null; // indexed vertex-pulling not supported
    const vmem = vb.memory orelse return null;
    const ib = d.index_buffer orelse return null;
    const imem = ib.memory orelse return null;
    const stride: usize = d.pipeline.stride;
    if (stride == 0) return null;
    const vbytes = dev.hal().mapResource(vmem.resource) catch return null;
    const ibytes = dev.hal().mapResource(imem.resource) catch return null;
    const index_count: usize = d.vertex_count;
    const isz: u64 = if (d.index_u32) 4 else 2;
    const out_size = @max(index_count * stride, 1);
    const res = dev.hal().createResource(.{ .buffer = .{ .size = out_size, .usage = .{ .vertex = true } } }) catch return null;
    const obytes = dev.hal().mapResource(res) catch {
        dev.hal().destroyResource(res);
        return null;
    };
    @memset(obytes[0..out_size], 0);
    const ibase: u64 = d.index_offset + @as(u64, d.first_vertex) * isz; // first_vertex holds firstIndex
    var k: usize = 0;
    while (k < index_count) : (k += 1) {
        const ioff = ibase + @as(u64, k) * isz;
        if (ioff + isz > ibytes.len) break;
        const raw: u32 = if (d.index_u32)
            std.mem.readInt(u32, ibytes[@intCast(ioff)..][0..4], .little)
        else
            std.mem.readInt(u16, ibytes[@intCast(ioff)..][0..2], .little);
        const vi: i64 = @as(i64, raw) + d.vertex_offset;
        if (vi < 0) continue;
        const src: usize = @as(usize, @intCast(vi)) * stride;
        if (src + stride > vbytes.len) continue;
        @memcpy(obytes[k * stride ..][0..stride], vbytes[src..][0..stride]);
    }
    return res;
}

fn executeRender(dev: *LogicalDevice, cb: *CommandBuffer) vk.VkResult {
    // Buffer copies (staging uploads) run first, before any draws read the data, and even for a
    // pure-upload command buffer with no draws.
    executeBufferCopies(dev, cb);
    if (!cb.has_draw and !cb.has_copy) return .VK_SUCCESS;

    if (cb.has_draw) {
        if (cb.pass_count > 0) {
            // Multiple render-pass instances: replay each one in order into its own
            // framebuffer. Pass 1 may render into an offscreen sampled image. Pass 2's
            // descriptor set binds that image as a combined-image-sampler so its FS
            // samples pass 1's freshly-rendered pixels. The swapchain-image pass is the one
            // present later reads.
            var pidx: usize = 0;
            while (pidx < cb.pass_count) : (pidx += 1) {
                const inst = cb.passes[pidx];
                const fb = inst.framebuffer orelse continue;
                const target = fb.view.image.resource orelse return .VK_ERROR_INITIALIZATION_FAILED;
                const rc = executePassInstance(
                    dev,
                    cb,
                    target,
                    .{ .r = inst.clear_r, .g = inst.clear_g, .b = inst.clear_b, .a = inst.clear_a },
                    inst.depth_image,
                    inst.has_depth,
                    inst.depth_clear,
                    inst.stencil_clear,
                    inst.draw_start,
                    inst.draw_count,
                    inst.resolve_image,
                    inst.samples,
                    fb.view.image.width,
                    fb.view.image.height,
                    fb.view.image.format,
                );
                if (rc != .VK_SUCCESS) return rc;
            }
        } else {
            // Legacy single-pass fallback: no render-pass instance was opened (the in-tree
            // path that records the flat framebuffer + draws directly, or the out-of-slots
            // degraded path). Clears once and replays all recorded draws into the flat
            // framebuffer. If no draw list was recorded, the flat single-draw fields drive it.
            const fb = cb.framebuffer orelse return .VK_ERROR_INITIALIZATION_FAILED;
            const target = fb.view.image.resource orelse return .VK_ERROR_INITIALIZATION_FAILED;
            if (cb.draw_count == 0) {
                const ctx = dev.hal().createContext() catch return .VK_ERROR_DEVICE_LOST;
                defer ctx.deinit();
                const hcb = ctx.beginCommands() catch return .VK_ERROR_DEVICE_LOST;
                defer hcb.deinit();
                hcb.setRenderTarget(target) catch return .VK_ERROR_DEVICE_LOST;
                hcb.clear(.{ .r = cb.clear_r, .g = cb.clear_g, .b = cb.clear_b, .a = cb.clear_a }) catch return .VK_ERROR_DEVICE_LOST;
                if (cb.has_depth) {
                    if (cb.depth_image) |di| {
                        if (di.resource) |dres| {
                            hcb.setDepthTarget(dres, cb.depth_clear) catch return .VK_ERROR_DEVICE_LOST;
                        }
                    }
                }
                const pipe = cb.gfx_pipeline orelse return .VK_ERROR_INITIALIZATION_FAILED;
                const vbuf = cb.vertex_buffer orelse return .VK_ERROR_INITIALIZATION_FAILED;
                const vmem = vbuf.memory orelse return .VK_ERROR_INITIALIZATION_FAILED;
                hcb.bindPipeline(pipe.hal_pipeline) catch return .VK_ERROR_DEVICE_LOST;
                hcb.bindVertexBuffer(vmem.resource) catch return .VK_ERROR_DEVICE_LOST;
                if (cb.descriptor_set) |set| bindUniformBuffers(hcb, set) catch return .VK_ERROR_DEVICE_LOST;
                hcb.draw(cb.draw_vertex_count, cb.draw_first_vertex) catch return .VK_ERROR_DEVICE_LOST;
                ctx.submit(hcb) catch return .VK_ERROR_DEVICE_LOST;
            } else {
                const rc = executePassInstance(
                    dev,
                    cb,
                    target,
                    .{ .r = cb.clear_r, .g = cb.clear_g, .b = cb.clear_b, .a = cb.clear_a },
                    cb.depth_image,
                    cb.has_depth,
                    cb.depth_clear,
                    0, // stencil_clear (legacy single-pass path: stencil unused)
                    0, // draw_start
                    cb.draw_count,
                    null,
                    fb.view.image.samples,
                    fb.view.image.width,
                    fb.view.image.height,
                    fb.view.image.format,
                );
                if (rc != .VK_SUCCESS) return rc;
            }
        }
    }

    // The image->buffer copy: move the rendered image pixels into the readback
    // buffer's HAL memory (both are HAL Resources mapped to the same backing).
    if (cb.has_copy) {
        const src_img = cb.copy_src_image orelse return .VK_ERROR_INITIALIZATION_FAILED;
        const dst_buf = cb.copy_dst_buffer orelse return .VK_ERROR_INITIALIZATION_FAILED;
        const src_res = src_img.resource orelse return .VK_ERROR_INITIALIZATION_FAILED;
        const dst_mem = dst_buf.memory orelse return .VK_ERROR_INITIALIZATION_FAILED;
        const src_bytes = dev.hal().mapResource(src_res) catch return .VK_ERROR_DEVICE_LOST;
        const dst_bytes = dev.hal().mapResource(dst_mem.resource) catch return .VK_ERROR_DEVICE_LOST;
        const n = @min(src_bytes.len, dst_bytes.len);
        @memcpy(dst_bytes[0..n], src_bytes[0..n]);
    }
    return .VK_SUCCESS;
}

/// Execute one recorded command buffer's compute dispatch through the HAL.
/// The software HAL backend runs `groups.x * groups.y * groups.z` invocations, one
/// gid per call, where each thread processes its own element. Vulkan's launch covers
/// `groupCountX * local_size_x` x-invocations. We fold both into groups[0] by passing
/// the bound buffers' element count as the total invocation count (one thread per output
/// element), driven by the buffer span rather than re-parsing local_size_x here.
fn executeDispatch(dev: *LogicalDevice, cb: *CommandBuffer) vk.VkResult {
    if (!cb.has_dispatch) return .VK_SUCCESS;
    const pipe = cb.pipeline orelse return .VK_ERROR_INITIALIZATION_FAILED;
    const set = cb.descriptor_set orelse return .VK_ERROR_INITIALIZATION_FAILED;

    // Collect the bound storage buffers' HAL resources, in binding order, up to the
    // pipeline layout's binding count.
    var resources: [MAX_BINDINGS]*prism.hal.Resource = undefined;
    var n: usize = 0;
    var b: u32 = 0;
    const count = if (pipe.binding_count == 0) MAX_BINDINGS else pipe.binding_count;
    while (b < count and b < MAX_BINDINGS) : (b += 1) {
        const buf = set.buffers[b] orelse break;
        const mem = buf.memory orelse return .VK_ERROR_INITIALIZATION_FAILED;
        resources[n] = mem.resource;
        n += 1;
    }
    if (n == 0) return .VK_ERROR_INITIALIZATION_FAILED;

    // Total invocations = the dispatched grid times the shader's workgroup x-size.
    // To stay shader-driven we pass the buffer element count as the total invocations,
    // since the M3 kernel processes one element per thread.
    const total_elems = smallestBufferElems(set, n);
    const groups: [3]u32 = .{ @intCast(total_elems), 1, 1 };

    dev.hal().dispatchCompute(.{
        .shader = pipe.hal_shader,
        .buffers = resources[0..n],
        .groups = groups,
    }) catch return .VK_ERROR_DEVICE_LOST;
    return .VK_SUCCESS;
}

/// The element count (u32s) of the smallest bound storage buffer: the safe upper
/// bound on the number of one-element-per-thread invocations to run.
fn smallestBufferElems(set: *DescriptorSet, n: usize) u64 {
    var min_bytes: u64 = std.math.maxInt(u64);
    var i: usize = 0;
    while (i < n) : (i += 1) {
        const buf = set.buffers[i].?;
        if (buf.size < min_bytes) min_bytes = buf.size;
    }
    return min_bytes / 4;
}

pub fn queueSubmit(
    queue: vk.VkQueue,
    submitCount: u32,
    pSubmits: ?[*]const vk.VkSubmitInfo,
    fence: vk.VkFence,
) callconv(.c) vk.VkResult {
    // The queue is &dev.queue: recover the device from the queue's address.
    const q: *Queue = @ptrCast(@alignCast(queue.?));
    const dev: *LogicalDevice = @fieldParentPtr("queue", q);
    if (pSubmits) |submits| {
        var s: u32 = 0;
        while (s < submitCount) : (s += 1) {
            const sub = submits[s];
            if (sub.pCommandBuffers) |cbs| {
                var c: u32 = 0;
                while (c < sub.commandBufferCount) : (c += 1) {
                    const cb = CommandBuffer.fromHandle(cbs[c]);
                    // Compute dispatch (M3) and/or offscreen render + copy (M4).
                    const rc = executeDispatch(dev, cb);
                    if (rc != .VK_SUCCESS) return rc;
                    const rg = executeRender(dev, cb);
                    if (rg != .VK_SUCCESS) return rg;
                }
            }
        }
    }
    // Synchronous: the dispatch is complete, so signal the fence.
    if (fence != vk.VK_NULL_HANDLE) Fence.fromHandle(fence).signaled = true;
    return .VK_SUCCESS;
}

pub fn queueWaitIdle(queue: vk.VkQueue) callconv(.c) vk.VkResult {
    _ = queue;
    return .VK_SUCCESS; // submits are synchronous. The queue is always idle.
}

pub fn deviceWaitIdle(device: vk.VkDevice) callconv(.c) vk.VkResult {
    _ = device;
    return .VK_SUCCESS;
}

pub fn createFence(
    device: vk.VkDevice,
    pCreateInfo: ?*const vk.VkFenceCreateInfo,
    pAllocator: ?*const vk.VkAllocationCallbacks,
    pFence: *vk.VkFence,
) callconv(.c) vk.VkResult {
    _ = device;
    _ = pAllocator;
    const f = allocator.create(Fence) catch return .VK_ERROR_OUT_OF_HOST_MEMORY;
    // VK_FENCE_CREATE_SIGNALED_BIT = 0x1.
    const signaled = if (pCreateInfo) |ci| (ci.flags & 0x1) != 0 else false;
    f.* = .{ .signaled = signaled };
    pFence.* = f.toHandle();
    return .VK_SUCCESS;
}

pub fn destroyFence(
    device: vk.VkDevice,
    fence: vk.VkFence,
    pAllocator: ?*const vk.VkAllocationCallbacks,
) callconv(.c) void {
    _ = device;
    _ = pAllocator;
    if (fence == vk.VK_NULL_HANDLE) return;
    allocator.destroy(Fence.fromHandle(fence));
}

pub fn waitForFences(
    device: vk.VkDevice,
    fenceCount: u32,
    pFences: ?[*]const vk.VkFence,
    waitAll: vk.VkBool32,
    timeout: u64,
) callconv(.c) vk.VkResult {
    _ = device;
    _ = waitAll;
    _ = timeout;
    _ = fenceCount;
    _ = pFences;
    // Submits are synchronous, so any fence handed to waitForFences is already done.
    return .VK_SUCCESS;
}

pub fn resetFences(
    device: vk.VkDevice,
    fenceCount: u32,
    pFences: ?[*]const vk.VkFence,
) callconv(.c) vk.VkResult {
    _ = device;
    if (pFences) |fences| {
        var i: u32 = 0;
        while (i < fenceCount) : (i += 1) {
            if (fences[i] != vk.VK_NULL_HANDLE) Fence.fromHandle(fences[i]).signaled = false;
        }
    }
    return .VK_SUCCESS;
}

pub fn getFenceStatus(
    device: vk.VkDevice,
    fence: vk.VkFence,
) callconv(.c) vk.VkResult {
    _ = device;
    if (fence == vk.VK_NULL_HANDLE) return .VK_SUCCESS;
    return if (Fence.fromHandle(fence).signaled) .VK_SUCCESS else .VK_NOT_READY;
}

// WSI: surface + swapchain + present.
//
// Two surface kinds are supported behind one VkSurfaceKHR:
//
//  (A) Standard libwayland (the interop goal): vkCreateWaylandSurfaceKHR receives
//      a genuine libwayland `wl_display` (in `.display`) + `wl_surface` (in
//      `.surface`) that the host app (vkcube, a stock Vulkan triangle, DOTA)
//      already created on its own libwayland-client connection. The ICD does not
//      link libwayland. It declares the wl_* C API + interface DATA symbols as
//      extern (wsi_wayland.zig) and they resolve at runtime from the app's loaded
//      libwayland-client.so. Each swapchain image is backed by a wl_shm buffer
//      (memfd+mmap via wayland.zig's ShmPool). Present blits the rendered RGBA8
//      into the shm buffer's XRGB8888 layout, then attach+damage_buffer+commit+
//      flush on the app's wl_surface. This is the Mesa wsi_common_wayland model.
//
//  (B) Legacy Prism client (the M6 path, kept for the in-tree example): when
//      `.display` is null, `.surface` is a *prism.platform.Surface and present
//      reuses the platform.wayland path (ctx.present blits into Prism's own
//      surface's shm buffer). The example triangle_vulkan_wayland still uses this.
//
// The kind is chosen by whether `.display` is non-null at create time.

const platform = prism.platform;

/// A VkSurfaceKHR. Either a standard libwayland surface (display + surface
/// pointers from the app's libwayland), or the legacy Prism platform surface.
pub const Surface = union(enum) {
    /// Standard app: the app's real libwayland wl_display + wl_surface.
    libwayland: struct {
        display: *wsi_wl.wl_display,
        surface: *wsi_wl.wl_surface,
    },
    /// Legacy in-tree path: Prism's own platform surface.
    prism: *platform.Surface,

    fn toHandle(self: *Surface) vk.VkSurfaceKHR {
        return @intCast(@intFromPtr(self));
    }
    fn fromHandle(h: vk.VkSurfaceKHR) *Surface {
        return @ptrFromInt(@as(usize, @intCast(h)));
    }
};

/// A VkSwapchainKHR: N render-target image Resources at the surface extent, plus
/// the HAL surface + a Context used to present the acquired image. acquire round-robins
/// the image index. present blits image[i] into the platform surface.
pub const Swapchain = struct {
    dev: *LogicalDevice,
    surface: *Surface,
    width: u32,
    height: u32,
    /// The swapchain images, as ICD-side Image objects (each owns a HAL image
    /// Resource). Same Image type the offscreen path uses, so vkAcquireNextImage
    /// hands back a VkImage the render path already understands.
    images: []*Image,
    next: u32 = 0,
    /// Present backend: exactly one is set.
    /// (A) libwayland: the wl_shm WSI over the app's real wl_surface.
    wl: ?*wsi_wl.WaylandWsi = null,
    /// (B) legacy Prism: the HAL surface + context (platform.wayland present).
    hal_surface: ?*prism.hal.Surface = null,
    ctx: ?prism.hal.Context = null,

    fn toHandle(self: *Swapchain) vk.VkSwapchainKHR {
        return @intCast(@intFromPtr(self));
    }
    fn fromHandle(h: vk.VkSwapchainKHR) *Swapchain {
        return @ptrFromInt(@as(usize, @intCast(h)));
    }
};

/// Registry of live swapchains. A VkSwapchainKHR is `@intFromPtr` of the
/// Swapchain, so `fromHandle` is an unchecked `@ptrFromInt`. On a compositor
/// resize a WSI app (e.g. vkcube) recreates its swapchain and destroys the
/// retired one. A stale or double-destroyed handle would otherwise dereference
/// freed/unmapped memory and SIGSEGV inside destroySwapchainKHR. This set is the
/// single source of truth for which Swapchain pointers are currently valid.
/// create inserts, destroy atomically checks-and-removes, acquire/present
/// validate before use. An unknown handle is a safe no-op instead of a
/// use-after-free crash that takes down the host application.
/// Vulkan WSI calls on a given swapchain are externally synchronized per the spec
/// and the ICD services them on the app's render thread, so this set needs no lock.
const SwapchainRegistry = struct {
    var live: std.AutoHashMapUnmanaged(*Swapchain, void) = .{};

    fn add(sc: *Swapchain) void {
        // page_allocator-backed. Ignore OOM (the swapchain still works, it just
        // would not be double-free-guarded. Acceptable degradation.
        live.put(allocator, sc, {}) catch {};
    }

    /// True iff `sc` is currently a live swapchain. The pointer is only
    /// dereferenced by the caller after this returns true.
    fn isLive(sc: *Swapchain) bool {
        return live.contains(sc);
    }

    /// Remove `sc`. Returns true if it was live (the caller then owns teardown).
    /// False means it was already destroyed / never created, so the caller must
    /// not touch the (freed) memory.
    fn remove(sc: *Swapchain) bool {
        return live.remove(sc);
    }
};

/// A VkSemaphore: a no-op for the synchronous software present path. Tracked as a
/// heap struct so the handle is a stable non-null u64 (some apps assert non-null).
pub const Semaphore = struct {
    dummy: u8 = 0,
    fn toHandle(self: *Semaphore) vk.VkSemaphore {
        return @intCast(@intFromPtr(self));
    }
    fn fromHandle(h: vk.VkSemaphore) *Semaphore {
        return @ptrFromInt(@as(usize, @intCast(h)));
    }
};

pub fn createWaylandSurfaceKHR(
    instance: vk.VkInstance,
    pCreateInfo: ?*const vk.VkWaylandSurfaceCreateInfoKHR,
    pAllocator: ?*const vk.VkAllocationCallbacks,
    pSurface: *vk.VkSurfaceKHR,
) callconv(.c) vk.VkResult {
    _ = instance;
    _ = pAllocator;
    const ci = pCreateInfo orelse return .VK_ERROR_INITIALIZATION_FAILED;
    const ps = ci.surface orelse return .VK_ERROR_INITIALIZATION_FAILED;
    const s = allocator.create(Surface) catch return .VK_ERROR_OUT_OF_HOST_MEMORY;
    if (ci.display) |disp| {
        // (A) standard libwayland: genuine wl_display + wl_surface from the app.
        s.* = .{ .libwayland = .{
            .display = @ptrCast(disp),
            .surface = @ptrCast(ps),
        } };
    } else {
        // (B) legacy Prism: a *prism.platform.Surface in `.surface`.
        s.* = .{ .prism = @ptrCast(@alignCast(ps)) };
    }
    pSurface.* = s.toHandle();
    return .VK_SUCCESS;
}

pub fn destroySurfaceKHR(
    instance: vk.VkInstance,
    surface: vk.VkSurfaceKHR,
    pAllocator: ?*const vk.VkAllocationCallbacks,
) callconv(.c) void {
    _ = instance;
    _ = pAllocator;
    if (surface == 0) return;
    allocator.destroy(Surface.fromHandle(surface));
}

pub fn getPhysicalDeviceSurfaceSupportKHR(
    physicalDevice: vk.VkPhysicalDevice,
    queueFamilyIndex: u32,
    surface: vk.VkSurfaceKHR,
    pSupported: *vk.VkBool32,
) callconv(.c) vk.VkResult {
    _ = physicalDevice;
    _ = queueFamilyIndex;
    _ = surface;
    pSupported.* = vk.VK_TRUE;
    return .VK_SUCCESS;
}

/// vkGetPhysicalDeviceWaylandPresentationSupportKHR: a standard app (vkcube)
/// calls this to check a queue family can present to a given wl_display. The
/// software path can always present via wl_shm, so report VK_TRUE.
pub fn getPhysicalDeviceWaylandPresentationSupportKHR(
    physicalDevice: vk.VkPhysicalDevice,
    queueFamilyIndex: u32,
    display: ?*anyopaque,
) callconv(.c) vk.VkBool32 {
    _ = physicalDevice;
    _ = queueFamilyIndex;
    _ = display;
    return vk.VK_TRUE;
}

pub fn getPhysicalDeviceSurfaceCapabilitiesKHR(
    physicalDevice: vk.VkPhysicalDevice,
    surface: vk.VkSurfaceKHR,
    pCaps: *vk.VkSurfaceCapabilitiesKHR,
) callconv(.c) vk.VkResult {
    _ = physicalDevice;
    const s = Surface.fromHandle(surface);
    // Legacy Prism surfaces carry a fixed extent. Libwayland surfaces don't (the
    // app drives the size via xdg-shell), so report the "swapchain decides"
    // sentinel 0xFFFFFFFF for currentExtent and a generous max, exactly like
    // Mesa's wayland WSI.
    const current: vk.VkExtent2D = switch (s.*) {
        .prism => |p| blk: {
            const sz = p.size();
            break :blk .{ .width = sz[0], .height = sz[1] };
        },
        .libwayland => .{ .width = 0xFFFFFFFF, .height = 0xFFFFFFFF },
    };
    const min_ext: vk.VkExtent2D = switch (s.*) {
        .prism => current,
        .libwayland => .{ .width = 1, .height = 1 },
    };
    const max_ext: vk.VkExtent2D = switch (s.*) {
        .prism => current,
        .libwayland => .{ .width = 16384, .height = 16384 },
    };
    pCaps.* = .{
        .minImageCount = 2,
        .maxImageCount = 0, // 0 = no upper limit
        .currentExtent = current,
        .minImageExtent = min_ext,
        .maxImageExtent = max_ext,
        .maxImageArrayLayers = 1,
        .supportedTransforms = vk.VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR,
        .currentTransform = vk.VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR,
        .supportedCompositeAlpha = vk.VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
        .supportedUsageFlags = vk.VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | vk.VK_IMAGE_USAGE_TRANSFER_DST_BIT | vk.VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
    };
    return .VK_SUCCESS;
}

const surface_formats = [_]vk.VkSurfaceFormatKHR{
    .{ .format = vk.VK_FORMAT_B8G8R8A8_UNORM, .colorSpace = vk.VK_COLOR_SPACE_SRGB_NONLINEAR_KHR },
    .{ .format = vk.VK_FORMAT_B8G8R8A8_SRGB, .colorSpace = vk.VK_COLOR_SPACE_SRGB_NONLINEAR_KHR },
    .{ .format = vk.VK_FORMAT_R8G8B8A8_UNORM, .colorSpace = vk.VK_COLOR_SPACE_SRGB_NONLINEAR_KHR },
};

pub fn getPhysicalDeviceSurfaceFormatsKHR(
    physicalDevice: vk.VkPhysicalDevice,
    surface: vk.VkSurfaceKHR,
    pCount: *u32,
    pFormats: ?[*]vk.VkSurfaceFormatKHR,
) callconv(.c) vk.VkResult {
    _ = physicalDevice;
    _ = surface;
    const n: u32 = surface_formats.len;
    if (pFormats == null) {
        pCount.* = n;
        return .VK_SUCCESS;
    }
    const write = @min(pCount.*, n);
    var i: u32 = 0;
    while (i < write) : (i += 1) pFormats.?[i] = surface_formats[i];
    pCount.* = write;
    return if (write < n) .VK_INCOMPLETE else .VK_SUCCESS;
}

const present_modes = [_]vk.VkPresentModeKHR{
    vk.VK_PRESENT_MODE_FIFO_KHR,
    vk.VK_PRESENT_MODE_MAILBOX_KHR,
};

pub fn getPhysicalDeviceSurfacePresentModesKHR(
    physicalDevice: vk.VkPhysicalDevice,
    surface: vk.VkSurfaceKHR,
    pCount: *u32,
    pModes: ?[*]vk.VkPresentModeKHR,
) callconv(.c) vk.VkResult {
    _ = physicalDevice;
    _ = surface;
    const n: u32 = present_modes.len;
    if (pModes == null) {
        pCount.* = n;
        return .VK_SUCCESS;
    }
    const write = @min(pCount.*, n);
    var i: u32 = 0;
    while (i < write) : (i += 1) pModes.?[i] = present_modes[i];
    pCount.* = write;
    return if (write < n) .VK_INCOMPLETE else .VK_SUCCESS;
}

pub fn createSwapchainKHR(
    device: vk.VkDevice,
    pCreateInfo: ?*const vk.VkSwapchainCreateInfoKHR,
    pAllocator: ?*const vk.VkAllocationCallbacks,
    pSwapchain: *vk.VkSwapchainKHR,
) callconv(.c) vk.VkResult {
    _ = pAllocator;
    const dev = LogicalDevice.fromHandle(device);
    const ci = pCreateInfo orelse return .VK_ERROR_INITIALIZATION_FAILED;
    const s = Surface.fromHandle(ci.surface);

    const w = ci.imageExtent.width;
    const h = ci.imageExtent.height;
    const count: u32 = @max(ci.minImageCount, 2);

    // The N swapchain images: HAL render-target Resources the software path
    // renders into (same Image type the offscreen path uses). On present they
    // are blitted to the surface (a wl_shm buffer for libwayland, the platform
    // surface's shm for the legacy path).
    const sc = allocator.create(Swapchain) catch return .VK_ERROR_OUT_OF_HOST_MEMORY;
    errdefer allocator.destroy(sc);
    const images = allocator.alloc(*Image, count) catch return .VK_ERROR_OUT_OF_HOST_MEMORY;
    errdefer allocator.free(images);

    var made: usize = 0;
    errdefer {
        var k: usize = 0;
        while (k < made) : (k += 1) {
            if (images[k].resource) |r| dev.hal().destroyResource(r);
            allocator.destroy(images[k]);
        }
    }
    while (made < count) : (made += 1) {
        const img = allocator.create(Image) catch return .VK_ERROR_OUT_OF_HOST_MEMORY;
        const res = dev.hal().createResource(.{ .image = .{
            .width = w,
            .height = h,
            .format = .rgba8_unorm,
            .usage = .{ .render_target = true, .copy_src = true },
        } }) catch {
            allocator.destroy(img);
            return .VK_ERROR_INITIALIZATION_FAILED;
        };
        img.* = .{ .width = w, .height = h, .format = .rgba8_unorm, .resource = res };
        images[made] = img;
    }

    switch (s.*) {
        .libwayland => |lw| {
            // (A) standard libwayland: bind wl_shm on the app's display + back
            // each image with a wl_shm buffer (memfd+mmap via wayland.zig).
            const wsi = wsi_wl.WaylandWsi.init(allocator, lw.display, lw.surface, w, h, count) catch
                return .VK_ERROR_INITIALIZATION_FAILED;
            sc.* = .{
                .dev = dev,
                .surface = s,
                .width = w,
                .height = h,
                .images = images,
                .wl = wsi,
            };
        },
        .prism => |ps| {
            // (B) legacy Prism: the HAL surface + a context for ctx.present.
            const hal_surface = dev.hal().createSurface(@ptrCast(ps)) catch
                return .VK_ERROR_INITIALIZATION_FAILED;
            errdefer dev.hal().destroySurface(hal_surface);
            const ctx = dev.hal().createContext() catch return .VK_ERROR_INITIALIZATION_FAILED;
            errdefer ctx.deinit();
            sc.* = .{
                .dev = dev,
                .surface = s,
                .width = w,
                .height = h,
                .images = images,
                .hal_surface = hal_surface,
                .ctx = ctx,
            };
        },
    }
    SwapchainRegistry.add(sc);
    pSwapchain.* = sc.toHandle();
    return .VK_SUCCESS;
}

pub fn destroySwapchainKHR(
    device: vk.VkDevice,
    swapchain: vk.VkSwapchainKHR,
    pAllocator: ?*const vk.VkAllocationCallbacks,
) callconv(.c) void {
    _ = pAllocator;
    if (swapchain == 0) return;
    const dev = LogicalDevice.fromHandle(device);
    const sc = Swapchain.fromHandle(swapchain);
    // Atomically claim the swapchain. If it is not live (already destroyed, or a
    // stale handle from a WSI app's recreate-on-resize), this is a no-op rather
    // than a use-after-free that segfaults the host (the crash this guards).
    if (!SwapchainRegistry.remove(sc)) return;
    for (sc.images) |img| {
        if (img.resource) |r| dev.hal().destroyResource(r);
        allocator.destroy(img);
    }
    allocator.free(sc.images);
    if (sc.wl) |wsi| wsi.deinit();
    if (sc.ctx) |ctx| ctx.deinit();
    if (sc.hal_surface) |hs| dev.hal().destroySurface(hs);
    allocator.destroy(sc);
}

pub fn getSwapchainImagesKHR(
    device: vk.VkDevice,
    swapchain: vk.VkSwapchainKHR,
    pCount: *u32,
    pImages: ?[*]vk.VkImage,
) callconv(.c) vk.VkResult {
    _ = device;
    const sc = Swapchain.fromHandle(swapchain);
    if (!SwapchainRegistry.isLive(sc)) return .VK_ERROR_OUT_OF_DATE_KHR;
    const n: u32 = @intCast(sc.images.len);
    if (pImages == null) {
        pCount.* = n;
        return .VK_SUCCESS;
    }
    const write = @min(pCount.*, n);
    var i: u32 = 0;
    while (i < write) : (i += 1) pImages.?[i] = sc.images[i].toHandle();
    pCount.* = write;
    return if (write < n) .VK_INCOMPLETE else .VK_SUCCESS;
}

pub fn acquireNextImageKHR(
    device: vk.VkDevice,
    swapchain: vk.VkSwapchainKHR,
    timeout: u64,
    semaphore: vk.VkSemaphore,
    fence: vk.VkFence,
    pImageIndex: *u32,
) callconv(.c) vk.VkResult {
    _ = device;
    _ = timeout;
    _ = semaphore;
    const sc = Swapchain.fromHandle(swapchain);
    if (!SwapchainRegistry.isLive(sc)) return .VK_ERROR_OUT_OF_DATE_KHR;
    if (sc.wl) |wsi| {
        // libwayland (FIFO): return a buffer the compositor has released. Image
        // index == wl_buffer index (created in the same order).
        pImageIndex.* = wsi.acquire();
    } else {
        pImageIndex.* = sc.next;
        sc.next = (sc.next + 1) % @as(u32, @intCast(sc.images.len));
    }
    // Synchronous path: any signalled fence is immediately satisfied.
    if (fence != vk.VK_NULL_HANDLE) Fence.fromHandle(fence).signaled = true;
    return .VK_SUCCESS;
}

/// How many presents have happened, and whether the image has been dumped already.
var present_count: u64 = 0;
var present_dumped: bool = false;

/// When the env var `PRISM_VK_DUMP` is set to a path, write the rendered swapchain image
/// (RGBA8) to it as a binary PPM (P6). Compositor-independent present capture. It reads
/// the same image bytes the ICD is about to present. Dumps a later frame (after enough
/// presents that the scene is drawn, not the initial clear): the frame index is
/// `PRISM_VK_DUMP_FRAME` if set, else 60. Best-effort: any failure is silently ignored (it
/// must never disturb presentation).
fn dumpPresentImage(dev: *LogicalDevice, res: *prism.hal.Resource, width: u32, height: u32) void {
    const path_c = std.c.getenv("PRISM_VK_DUMP") orelse return;
    present_count += 1;
    // PRISM_VK_DUMP_FRAME: dump exactly that frame once. If unset, dump every present
    // (overwriting), so the file ends up holding the last frame vkcube presented, which is
    // robust when a compositor closes the window after only a few frames.
    if (std.c.getenv("PRISM_VK_DUMP_FRAME")) |fc| {
        if (present_dumped) return;
        const want_frame = std.fmt.parseInt(u64, std.mem.span(fc), 10) catch 1;
        if (present_count < want_frame) return;
    }
    if (path_c[0] == 0) return;
    const src = dev.hal().mapResource(res) catch return;
    if (src.len < @as(usize, width) * height * 4) return;
    // Use libc file I/O (the .so links libc but not std's fs layer).
    const f = std.c.fopen(path_c, "wb") orelse return;
    defer _ = std.c.fclose(f);
    var hdr_buf: [64]u8 = undefined;
    const hdr = std.fmt.bufPrint(&hdr_buf, "P6\n{d} {d}\n255\n", .{ width, height }) catch return;
    _ = std.c.fwrite(hdr.ptr, 1, hdr.len, f);
    // RGBA8 -> RGB (drop alpha), row by row.
    if (width > 4096) return;
    var row: [4096 * 3]u8 = undefined;
    var y: u32 = 0;
    while (y < height) : (y += 1) {
        var x: u32 = 0;
        while (x < width) : (x += 1) {
            const si = (@as(usize, y) * width + x) * 4;
            row[x * 3 + 0] = src[si + 0];
            row[x * 3 + 1] = src[si + 1];
            row[x * 3 + 2] = src[si + 2];
        }
        _ = std.c.fwrite(&row, 1, @as(usize, width) * 3, f);
    }
    present_dumped = true;
}

pub fn queuePresentKHR(
    queue: vk.VkQueue,
    pPresentInfo: ?*const vk.VkPresentInfoKHR,
) callconv(.c) vk.VkResult {
    _ = queue;
    const pi = pPresentInfo orelse return .VK_ERROR_INITIALIZATION_FAILED;
    if (pi.pSwapchains == null or pi.pImageIndices == null) return .VK_ERROR_INITIALIZATION_FAILED;
    var i: u32 = 0;
    while (i < pi.swapchainCount) : (i += 1) {
        const sc = Swapchain.fromHandle(pi.pSwapchains.?[i]);
        // A stale/retired swapchain handle (WSI recreate-on-resize): report
        // out-of-date instead of dereferencing freed memory.
        if (!SwapchainRegistry.isLive(sc)) {
            if (pi.pResults) |r| r[i] = .VK_ERROR_OUT_OF_DATE_KHR;
            continue;
        }
        const idx = pi.pImageIndices.?[i];
        if (idx >= sc.images.len) {
            if (pi.pResults) |r| r[i] = .VK_ERROR_INITIALIZATION_FAILED;
            return .VK_ERROR_INITIALIZATION_FAILED;
        }
        const res = sc.images[idx].resource orelse return .VK_ERROR_INITIALIZATION_FAILED;
        // Optional present capture: when PRISM_VK_DUMP is set, write the rendered
        // swapchain image (RGBA8) to that path as a PPM. Compositor-independent capture
        // (e.g. on Weston, which has no wlr-screencopy for grim).
        dumpPresentImage(sc.dev, res, sc.width, sc.height);
        if (sc.wl) |wsi| {
            // (A) standard libwayland: blit the rendered RGBA8 image into the
            // wl_shm buffer's XRGB8888 pixels, then attach+damage+commit+flush
            // on the app's real wl_surface.
            const src = sc.dev.hal().mapResource(res) catch {
                if (pi.pResults) |r| r[i] = .VK_ERROR_DEVICE_LOST;
                return .VK_ERROR_DEVICE_LOST;
            };
            wsi_wl.blitRgbaToXrgb(wsi.buffers[idx].pixels, src, sc.width, sc.height);
            wsi.present(idx);
        } else {
            // (B) legacy Prism: reuse the platform.wayland present path (blit the
            // rendered image into the platform surface's shm buffer + commit).
            sc.ctx.?.present(sc.hal_surface.?, res) catch {
                if (pi.pResults) |r| r[i] = .VK_ERROR_DEVICE_LOST;
                return .VK_ERROR_DEVICE_LOST;
            };
        }
        if (pi.pResults) |r| r[i] = .VK_SUCCESS;
    }
    return .VK_SUCCESS;
}

pub fn createSemaphore(
    device: vk.VkDevice,
    pCreateInfo: ?*const vk.VkSemaphoreCreateInfo,
    pAllocator: ?*const vk.VkAllocationCallbacks,
    pSemaphore: *vk.VkSemaphore,
) callconv(.c) vk.VkResult {
    _ = device;
    _ = pCreateInfo;
    _ = pAllocator;
    const s = allocator.create(Semaphore) catch return .VK_ERROR_OUT_OF_HOST_MEMORY;
    s.* = .{};
    pSemaphore.* = s.toHandle();
    return .VK_SUCCESS;
}

pub fn destroySemaphore(
    device: vk.VkDevice,
    semaphore: vk.VkSemaphore,
    pAllocator: ?*const vk.VkAllocationCallbacks,
) callconv(.c) void {
    _ = device;
    _ = pAllocator;
    if (semaphore == 0) return;
    allocator.destroy(Semaphore.fromHandle(semaphore));
}

/// The WSI (M6) entry points, shared by the device + instance proc-addr tables.
const wsi_dispatch_table = .{
    .{ "vkCreateWaylandSurfaceKHR", asVoidFn(createWaylandSurfaceKHR) },
    .{ "vkDestroySurfaceKHR", asVoidFn(destroySurfaceKHR) },
    .{ "vkGetPhysicalDeviceSurfaceSupportKHR", asVoidFn(getPhysicalDeviceSurfaceSupportKHR) },
    .{ "vkGetPhysicalDeviceWaylandPresentationSupportKHR", asVoidFn(getPhysicalDeviceWaylandPresentationSupportKHR) },
    .{ "vkGetPhysicalDeviceSurfaceCapabilitiesKHR", asVoidFn(getPhysicalDeviceSurfaceCapabilitiesKHR) },
    .{ "vkGetPhysicalDeviceSurfaceFormatsKHR", asVoidFn(getPhysicalDeviceSurfaceFormatsKHR) },
    .{ "vkGetPhysicalDeviceSurfacePresentModesKHR", asVoidFn(getPhysicalDeviceSurfacePresentModesKHR) },
    .{ "vkCreateSwapchainKHR", asVoidFn(createSwapchainKHR) },
    .{ "vkDestroySwapchainKHR", asVoidFn(destroySwapchainKHR) },
    .{ "vkGetSwapchainImagesKHR", asVoidFn(getSwapchainImagesKHR) },
    .{ "vkAcquireNextImageKHR", asVoidFn(acquireNextImageKHR) },
    .{ "vkQueuePresentKHR", asVoidFn(queuePresentKHR) },
    .{ "vkCreateSemaphore", asVoidFn(createSemaphore) },
    .{ "vkDestroySemaphore", asVoidFn(destroySemaphore) },
};

pub fn getDeviceProcAddr(
    device: vk.VkDevice,
    pName: ?[*:0]const u8,
) callconv(.c) vk.PFN_vkVoidFunction {
    _ = device;
    const name = std.mem.span(pName orelse return null);
    const map = .{
        .{ "vkGetDeviceProcAddr", asVoidFn(getDeviceProcAddr) },
        .{ "vkDestroyDevice", asVoidFn(destroyDevice) },
        .{ "vkGetDeviceQueue", asVoidFn(getDeviceQueue) },
        .{ "vkCreateBuffer", asVoidFn(createBuffer) },
        .{ "vkDestroyBuffer", asVoidFn(destroyBuffer) },
        .{ "vkGetBufferMemoryRequirements", asVoidFn(getBufferMemoryRequirements) },
        .{ "vkAllocateMemory", asVoidFn(allocateMemory) },
        .{ "vkFreeMemory", asVoidFn(freeMemory) },
        .{ "vkBindBufferMemory", asVoidFn(bindBufferMemory) },
        .{ "vkMapMemory", asVoidFn(mapMemory) },
        .{ "vkUnmapMemory", asVoidFn(unmapMemory) },
        .{ "vkFlushMappedMemoryRanges", asVoidFn(flushMappedMemoryRanges) },
        .{ "vkInvalidateMappedMemoryRanges", asVoidFn(invalidateMappedMemoryRanges) },
    } ++ compute_dispatch_table ++ wsi_dispatch_table;
    inline for (map) |entry| {
        if (std.mem.eql(u8, name, entry[0])) return entry[1];
    }
    return null;
}

/// The compute-dispatch device-level entry points (M3), shared by the device and
/// instance proc-addr tables.
const compute_dispatch_table = .{
    .{ "vkCreateShaderModule", asVoidFn(createShaderModule) },
    .{ "vkDestroyShaderModule", asVoidFn(destroyShaderModule) },
    .{ "vkCreateDescriptorSetLayout", asVoidFn(createDescriptorSetLayout) },
    .{ "vkDestroyDescriptorSetLayout", asVoidFn(destroyDescriptorSetLayout) },
    .{ "vkCreatePipelineLayout", asVoidFn(createPipelineLayout) },
    .{ "vkDestroyPipelineLayout", asVoidFn(destroyPipelineLayout) },
    .{ "vkCreateComputePipelines", asVoidFn(createComputePipelines) },
    .{ "vkDestroyPipeline", asVoidFn(destroyPipeline) },
    .{ "vkCreateDescriptorPool", asVoidFn(createDescriptorPool) },
    .{ "vkDestroyDescriptorPool", asVoidFn(destroyDescriptorPool) },
    .{ "vkAllocateDescriptorSets", asVoidFn(allocateDescriptorSets) },
    .{ "vkUpdateDescriptorSets", asVoidFn(updateDescriptorSets) },
    .{ "vkCreateCommandPool", asVoidFn(createCommandPool) },
    .{ "vkDestroyCommandPool", asVoidFn(destroyCommandPool) },
    .{ "vkAllocateCommandBuffers", asVoidFn(allocateCommandBuffers) },
    .{ "vkFreeCommandBuffers", asVoidFn(freeCommandBuffers) },
    .{ "vkBeginCommandBuffer", asVoidFn(beginCommandBuffer) },
    .{ "vkEndCommandBuffer", asVoidFn(endCommandBuffer) },
    .{ "vkResetCommandBuffer", asVoidFn(resetCommandBuffer) },
    .{ "vkResetCommandPool", asVoidFn(resetCommandPool) },
    .{ "vkCmdPipelineBarrier", asVoidFn(cmdPipelineBarrier) },
    .{ "vkCreatePipelineCache", asVoidFn(createPipelineCache) },
    .{ "vkDestroyPipelineCache", asVoidFn(destroyPipelineCache) },
    .{ "vkCmdBindPipeline", asVoidFn(cmdBindPipeline) },
    .{ "vkCmdBindDescriptorSets", asVoidFn(cmdBindDescriptorSets) },
    .{ "vkCmdDispatch", asVoidFn(cmdDispatch) },
    .{ "vkQueueSubmit", asVoidFn(queueSubmit) },
    .{ "vkQueueWaitIdle", asVoidFn(queueWaitIdle) },
    .{ "vkDeviceWaitIdle", asVoidFn(deviceWaitIdle) },
    .{ "vkCreateFence", asVoidFn(createFence) },
    .{ "vkDestroyFence", asVoidFn(destroyFence) },
    .{ "vkWaitForFences", asVoidFn(waitForFences) },
    .{ "vkResetFences", asVoidFn(resetFences) },
    .{ "vkGetFenceStatus", asVoidFn(getFenceStatus) },
} ++ graphics_dispatch_table;

/// The graphics device-level entry points (M4: the offscreen triangle), shared by
/// the device and instance proc-addr tables.
const graphics_dispatch_table = .{
    .{ "vkCreateImage", asVoidFn(createImage) },
    .{ "vkDestroyImage", asVoidFn(destroyImage) },
    .{ "vkGetImageMemoryRequirements", asVoidFn(getImageMemoryRequirements) },
    .{ "vkBindImageMemory", asVoidFn(bindImageMemory) },
    .{ "vkCreateImageView", asVoidFn(createImageView) },
    .{ "vkDestroyImageView", asVoidFn(destroyImageView) },
    .{ "vkCreateRenderPass", asVoidFn(createRenderPass) },
    .{ "vkDestroyRenderPass", asVoidFn(destroyRenderPass) },
    .{ "vkCreateFramebuffer", asVoidFn(createFramebuffer) },
    .{ "vkDestroyFramebuffer", asVoidFn(destroyFramebuffer) },
    .{ "vkCreateGraphicsPipelines", asVoidFn(createGraphicsPipelines) },
    .{ "vkCmdBeginRenderPass", asVoidFn(cmdBeginRenderPass) },
    .{ "vkCmdEndRenderPass", asVoidFn(cmdEndRenderPass) },
    .{ "vkCmdBindVertexBuffers", asVoidFn(cmdBindVertexBuffers) },
    .{ "vkCmdBindIndexBuffer", asVoidFn(cmdBindIndexBuffer) },
    .{ "vkCmdDrawIndexed", asVoidFn(cmdDrawIndexed) },
    .{ "vkCmdSetViewport", asVoidFn(cmdSetViewport) },
    .{ "vkCmdSetScissor", asVoidFn(cmdSetScissor) },
    .{ "vkCmdSetStencilReference", asVoidFn(cmdSetStencilReference) },
    .{ "vkCmdSetStencilCompareMask", asVoidFn(cmdSetStencilCompareMask) },
    .{ "vkCmdSetStencilWriteMask", asVoidFn(cmdSetStencilWriteMask) },
    .{ "vkCmdDraw", asVoidFn(cmdDraw) },
    .{ "vkCmdPushConstants", asVoidFn(cmdPushConstants) },
    .{ "vkCmdCopyImageToBuffer", asVoidFn(cmdCopyImageToBuffer) },
    // Textures / samplers (vkcube feature 3): combined-image-sampler path.
    .{ "vkCreateSampler", asVoidFn(createSampler) },
    .{ "vkDestroySampler", asVoidFn(destroySampler) },
    .{ "vkCmdCopyBufferToImage", asVoidFn(cmdCopyBufferToImage) },
    .{ "vkCmdCopyBuffer", asVoidFn(cmdCopyBuffer) },
    .{ "vkGetImageSubresourceLayout", asVoidFn(getImageSubresourceLayout) },
};

/// Lets the loader treat us as a 1.3 ICD instead of falling back to 1.0.
pub fn enumerateInstanceVersion(pApiVersion: *u32) callconv(.c) vk.VkResult {
    pApiVersion.* = vk.VK_API_VERSION_1_3;
    return .VK_SUCCESS;
}

pub fn enumerateDeviceExtensionProperties(
    physicalDevice: vk.VkPhysicalDevice,
    pLayerName: ?[*:0]const u8,
    pPropertyCount: *u32,
    pProperties: ?[*]vk.VkExtensionProperties,
) callconv(.c) vk.VkResult {
    _ = physicalDevice;
    _ = pLayerName;
    const n: u32 = 1; // VK_KHR_swapchain
    if (pProperties == null) {
        pPropertyCount.* = n;
        return .VK_SUCCESS;
    }
    if (pPropertyCount.* >= 1) {
        var e = std.mem.zeroes(vk.VkExtensionProperties);
        copyCStr(&e.extensionName, "VK_KHR_swapchain");
        e.specVersion = 70;
        pProperties.?[0] = e;
        pPropertyCount.* = 1;
        return .VK_SUCCESS;
    }
    pPropertyCount.* = 0;
    return .VK_INCOMPLETE;
}

// Proc-address dispatch.

/// Reinterpret a typed callconv(.c) fn pointer as the loader's opaque
/// PFN_vkVoidFunction.
fn asVoidFn(comptime f: anytype) vk.PFN_vkVoidFunction {
    return @ptrCast(&f);
}

/// The ICD's primary entry point. The loader calls this to resolve every function by
/// name. Returns a pointer to our implementation for each known name, else null.
pub fn getInstanceProcAddr(
    instance: vk.VkInstance,
    pName: ?[*:0]const u8,
) callconv(.c) vk.PFN_vkVoidFunction {
    _ = instance;
    const name = std.mem.span(pName orelse return null);
    const map = .{
        .{ "vkGetInstanceProcAddr", asVoidFn(getInstanceProcAddr) },
        .{ "vkEnumerateInstanceVersion", asVoidFn(enumerateInstanceVersion) },
        .{ "vkEnumerateInstanceExtensionProperties", asVoidFn(enumerateInstanceExtensionProperties) },
        .{ "vkEnumerateInstanceLayerProperties", asVoidFn(enumerateInstanceLayerProperties) },
        .{ "vkCreateInstance", asVoidFn(createInstance) },
        .{ "vkDestroyInstance", asVoidFn(destroyInstance) },
        .{ "vkEnumeratePhysicalDevices", asVoidFn(enumeratePhysicalDevices) },
        .{ "vkGetPhysicalDeviceProperties", asVoidFn(getPhysicalDeviceProperties) },
        .{ "vkGetPhysicalDeviceFeatures", asVoidFn(getPhysicalDeviceFeatures) },
        .{ "vkGetPhysicalDeviceFormatProperties", asVoidFn(getPhysicalDeviceFormatProperties) },
        .{ "vkGetPhysicalDeviceImageFormatProperties", asVoidFn(getPhysicalDeviceImageFormatProperties) },
        .{ "vkGetPhysicalDeviceSparseImageFormatProperties", asVoidFn(getPhysicalDeviceSparseImageFormatProperties) },
        .{ "vkGetPhysicalDeviceQueueFamilyProperties", asVoidFn(getPhysicalDeviceQueueFamilyProperties) },
        .{ "vkGetPhysicalDeviceMemoryProperties", asVoidFn(getPhysicalDeviceMemoryProperties) },
        .{ "vkCreateDevice", asVoidFn(createDevice) },
        .{ "vkDestroyDevice", asVoidFn(destroyDevice) },
        .{ "vkGetDeviceProcAddr", asVoidFn(getDeviceProcAddr) },
        .{ "vkGetDeviceQueue", asVoidFn(getDeviceQueue) },
        .{ "vkEnumerateDeviceExtensionProperties", asVoidFn(enumerateDeviceExtensionProperties) },
        // Device-level memory/buffer entry points. The loader also resolves these
        // through the instance proc addr, so list them here too (in addition to
        // getDeviceProcAddr) to be safe across loader versions.
        .{ "vkCreateBuffer", asVoidFn(createBuffer) },
        .{ "vkDestroyBuffer", asVoidFn(destroyBuffer) },
        .{ "vkGetBufferMemoryRequirements", asVoidFn(getBufferMemoryRequirements) },
        .{ "vkAllocateMemory", asVoidFn(allocateMemory) },
        .{ "vkFreeMemory", asVoidFn(freeMemory) },
        .{ "vkBindBufferMemory", asVoidFn(bindBufferMemory) },
        .{ "vkMapMemory", asVoidFn(mapMemory) },
        .{ "vkUnmapMemory", asVoidFn(unmapMemory) },
        .{ "vkFlushMappedMemoryRanges", asVoidFn(flushMappedMemoryRanges) },
        .{ "vkInvalidateMappedMemoryRanges", asVoidFn(invalidateMappedMemoryRanges) },
    } ++ compute_dispatch_table ++ wsi_dispatch_table;
    inline for (map) |entry| {
        if (std.mem.eql(u8, name, entry[0])) return entry[1];
    }
    return null;
}

// Tests.

test "icd advertises loader interface version 5" {
    try std.testing.expectEqual(@as(u32, 5), loader_icd_interface_version);
}

test "getInstanceProcAddr resolves implemented names and rejects unknowns" {
    try std.testing.expect(getInstanceProcAddr(null, "vkCreateInstance") != null);
    try std.testing.expect(getInstanceProcAddr(null, "vkEnumeratePhysicalDevices") != null);
    try std.testing.expect(getInstanceProcAddr(null, "vkGetInstanceProcAddr") != null);
    try std.testing.expect(getInstanceProcAddr(null, "vkNotARealFunction") == null);
}

test "instance enumerates a physical device per available driver" {
    var inst: vk.VkInstance = null;
    const ci = vk.VkInstanceCreateInfo{
        .sType = .VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pNext = null,
        .flags = 0,
        .pApplicationInfo = null,
        .enabledLayerCount = 0,
        .ppEnabledLayerNames = null,
        .enabledExtensionCount = 0,
        .ppEnabledExtensionNames = null,
    };
    try std.testing.expectEqual(vk.VkResult.VK_SUCCESS, createInstance(&ci, null, &inst));
    defer destroyInstance(inst, null);
    try std.testing.expect(inst != null);

    // Two-call idiom: count first.
    var count: u32 = 0;
    try std.testing.expectEqual(vk.VkResult.VK_SUCCESS, enumeratePhysicalDevices(inst, &count, null));
    // The software driver is always available, so at least one device.
    try std.testing.expect(count >= 1);

    var handles: [8]vk.VkPhysicalDevice = undefined;
    var got = count;
    try std.testing.expectEqual(vk.VkResult.VK_SUCCESS, enumeratePhysicalDevices(inst, &got, &handles[0]));
    try std.testing.expectEqual(count, got);

    // Each handle yields clean properties.
    var props: vk.VkPhysicalDeviceProperties = undefined;
    getPhysicalDeviceProperties(handles[0], &props);
    try std.testing.expectEqual(vk.VK_API_VERSION_1_3, props.apiVersion);
    try std.testing.expectEqual(PRISM_VENDOR_ID, props.vendorID);
    // deviceName starts with "Prism ".
    try std.testing.expect(std.mem.startsWith(u8, props.deviceName[0..6], "Prism "));
    // The loader magic survived into each handle.
    try std.testing.expectEqual(vk.ICD_LOADER_MAGIC, physFromHandle(handles[0]).loader_magic);
}

test "queue family and memory properties are well-formed" {
    var inst: vk.VkInstance = null;
    const ci = vk.VkInstanceCreateInfo{
        .sType = .VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pNext = null,
        .flags = 0,
        .pApplicationInfo = null,
        .enabledLayerCount = 0,
        .ppEnabledLayerNames = null,
        .enabledExtensionCount = 0,
        .ppEnabledExtensionNames = null,
    };
    _ = createInstance(&ci, null, &inst);
    defer destroyInstance(inst, null);
    var count: u32 = 1;
    var handle: vk.VkPhysicalDevice = null;
    _ = enumeratePhysicalDevices(inst, &count, &handle);

    var qcount: u32 = 0;
    getPhysicalDeviceQueueFamilyProperties(handle, &qcount, null);
    try std.testing.expectEqual(@as(u32, 1), qcount);
    var qfam: [1]vk.VkQueueFamilyProperties = undefined;
    getPhysicalDeviceQueueFamilyProperties(handle, &qcount, &qfam);
    try std.testing.expect((qfam[0].queueFlags & vk.VK_QUEUE_GRAPHICS_BIT) != 0);
    try std.testing.expectEqual(@as(u32, 1), qfam[0].queueCount);

    var mem: vk.VkPhysicalDeviceMemoryProperties = undefined;
    getPhysicalDeviceMemoryProperties(handle, &mem);
    try std.testing.expectEqual(@as(u32, 1), mem.memoryHeapCount);
    try std.testing.expectEqual(@as(u32, 1), mem.memoryTypeCount);
}

test "compute pipeline dispatch runs a SPIR-V kernel through the ICD" {
    const op = prism.spirv.opcodes;
    const sc = op.StorageClass;

    // The M3 kernel: out[i] = in[i] + 0x100, built as SPIR-V (in = buffer 0, out = 1).
    var spv_b = try prism.spirv.binary.Builder.init(allocator, 25);
    defer spv_b.deinit(allocator);
    try spv_b.emit(allocator, op.Decorate, &.{ 14, op.Decoration.builtin, op.BuiltIn.global_invocation_id });
    try spv_b.emit(allocator, op.TypeVoid, &.{1});
    try spv_b.emit(allocator, op.TypeInt, &.{ 2, 32, 1 });
    try spv_b.emit(allocator, op.TypeInt, &.{ 3, 32, 0 });
    try spv_b.emit(allocator, op.TypeVector, &.{ 4, 3, 3 });
    try spv_b.emit(allocator, op.TypePointer, &.{ 5, sc.input, 4 });
    try spv_b.emit(allocator, op.TypePointer, &.{ 6, sc.input, 3 });
    try spv_b.emit(allocator, op.TypeRuntimeArray, &.{ 7, 2 });
    try spv_b.emit(allocator, op.TypeStruct, &.{ 8, 7 });
    try spv_b.emit(allocator, op.TypePointer, &.{ 9, sc.storage_buffer, 8 });
    try spv_b.emit(allocator, op.TypePointer, &.{ 10, sc.storage_buffer, 2 });
    try spv_b.emit(allocator, op.TypeFunction, &.{ 11, 1 });
    try spv_b.emit(allocator, op.Constant, &.{ 3, 12, 0 });
    try spv_b.emit(allocator, op.Constant, &.{ 2, 13, 0x100 });
    try spv_b.emit(allocator, op.Variable, &.{ 5, 14, sc.input });
    try spv_b.emit(allocator, op.Variable, &.{ 9, 15, sc.storage_buffer });
    try spv_b.emit(allocator, op.Variable, &.{ 9, 16, sc.storage_buffer });
    try spv_b.emit(allocator, op.Function, &.{ 1, 17, 0, 11 });
    try spv_b.emit(allocator, op.Label, &.{18});
    try spv_b.emit(allocator, op.AccessChain, &.{ 6, 19, 14, 12 });
    try spv_b.emit(allocator, op.Load, &.{ 3, 20, 19 });
    try spv_b.emit(allocator, op.AccessChain, &.{ 10, 21, 15, 12, 20 });
    try spv_b.emit(allocator, op.Load, &.{ 2, 22, 21 });
    try spv_b.emit(allocator, op.IAdd, &.{ 2, 23, 22, 13 });
    try spv_b.emit(allocator, op.AccessChain, &.{ 10, 24, 16, 12, 20 });
    try spv_b.emit(allocator, op.Store, &.{ 24, 23 });
    try spv_b.emit(allocator, op.Return, &.{});
    try spv_b.emit(allocator, op.FunctionEnd, &.{});
    const spv = std.mem.sliceAsBytes(spv_b.words.items);

    // Instance + physical devices. Select the SOFTWARE physical device (the CPU/JIT
    // backend that runs this test's kernel): with multiple drivers compiled in,
    // device[0] may be a hardware driver whose compute path differs, so pick by name.
    var inst: vk.VkInstance = null;
    const ci = std.mem.zeroInit(vk.VkInstanceCreateInfo, .{ .sType = .VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO });
    _ = createInstance(&ci, null, &inst);
    defer destroyInstance(inst, null);
    var count: u32 = 0;
    _ = enumeratePhysicalDevices(inst, &count, null);
    if (count == 0) return error.SkipZigTest;
    var handles: [16]vk.VkPhysicalDevice = undefined;
    var got: u32 = @min(count, 16);
    _ = enumeratePhysicalDevices(inst, &got, &handles[0]);
    var phys: vk.VkPhysicalDevice = null;
    for (handles[0..got]) |h| {
        var props: vk.VkPhysicalDeviceProperties = undefined;
        getPhysicalDeviceProperties(h, &props);
        if (std.mem.indexOf(u8, &props.deviceName, "software") != null) {
            phys = h;
            break;
        }
    }
    if (phys == null) return error.SkipZigTest; // no software driver compiled in
    const dci = std.mem.zeroes(vk.VkDeviceCreateInfo);
    var device: vk.VkDevice = null;
    try std.testing.expectEqual(vk.VkResult.VK_SUCCESS, createDevice(phys, &dci, null, &device));
    defer destroyDevice(device, null);
    var queue: vk.VkQueue = null;
    getDeviceQueue(device, 0, 0, &queue);

    // Two storage buffers (N=256 u32s each), HAL-backed via M2.
    const N = 256;
    const bytes = N * 4;
    var in_buf: vk.VkBuffer = vk.VK_NULL_HANDLE;
    var out_buf: vk.VkBuffer = vk.VK_NULL_HANDLE;
    const bci = std.mem.zeroInit(vk.VkBufferCreateInfo, .{ .size = bytes, .usage = vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT });
    try std.testing.expectEqual(vk.VkResult.VK_SUCCESS, createBuffer(device, &bci, null, &in_buf));
    defer destroyBuffer(device, in_buf, null);
    try std.testing.expectEqual(vk.VkResult.VK_SUCCESS, createBuffer(device, &bci, null, &out_buf));
    defer destroyBuffer(device, out_buf, null);

    var reqs: vk.VkMemoryRequirements = undefined;
    getBufferMemoryRequirements(device, in_buf, &reqs);
    const mai = std.mem.zeroInit(vk.VkMemoryAllocateInfo, .{ .allocationSize = reqs.size });
    var in_mem: vk.VkDeviceMemory = vk.VK_NULL_HANDLE;
    var out_mem: vk.VkDeviceMemory = vk.VK_NULL_HANDLE;
    _ = allocateMemory(device, &mai, null, &in_mem);
    defer freeMemory(device, in_mem, null);
    _ = allocateMemory(device, &mai, null, &out_mem);
    defer freeMemory(device, out_mem, null);
    _ = bindBufferMemory(device, in_buf, in_mem, 0);
    _ = bindBufferMemory(device, out_buf, out_mem, 0);

    // Fill input[i] = i, zero output.
    var pin: ?*anyopaque = null;
    _ = mapMemory(device, in_mem, 0, vk.VK_WHOLE_SIZE, 0, &pin);
    const in_data: [*]u32 = @ptrCast(@alignCast(pin.?));
    for (0..N) |i| in_data[i] = @intCast(i);
    var pout: ?*anyopaque = null;
    _ = mapMemory(device, out_mem, 0, vk.VK_WHOLE_SIZE, 0, &pout);
    const out_data: [*]u32 = @ptrCast(@alignCast(pout.?));
    @memset(out_data[0..N], 0);

    // Descriptor set layout (2 storage buffers) + pipeline layout.
    const binds = [_]vk.VkDescriptorSetLayoutBinding{
        .{ .binding = 0, .descriptorType = vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1, .stageFlags = vk.VK_SHADER_STAGE_COMPUTE_BIT, .pImmutableSamplers = null },
        .{ .binding = 1, .descriptorType = vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1, .stageFlags = vk.VK_SHADER_STAGE_COMPUTE_BIT, .pImmutableSamplers = null },
    };
    const dslci = std.mem.zeroInit(vk.VkDescriptorSetLayoutCreateInfo, .{ .bindingCount = 2, .pBindings = &binds });
    var dsl: vk.VkDescriptorSetLayout = vk.VK_NULL_HANDLE;
    try std.testing.expectEqual(vk.VkResult.VK_SUCCESS, createDescriptorSetLayout(device, &dslci, null, &dsl));
    defer destroyDescriptorSetLayout(device, dsl, null);
    const plci = std.mem.zeroInit(vk.VkPipelineLayoutCreateInfo, .{ .setLayoutCount = 1, .pSetLayouts = @as([*]const vk.VkDescriptorSetLayout, @ptrCast(&dsl)) });
    var pl: vk.VkPipelineLayout = vk.VK_NULL_HANDLE;
    try std.testing.expectEqual(vk.VkResult.VK_SUCCESS, createPipelineLayout(device, &plci, null, &pl));
    defer destroyPipelineLayout(device, pl, null);

    // Shader module + compute pipeline.
    const smci = std.mem.zeroInit(vk.VkShaderModuleCreateInfo, .{ .codeSize = spv.len, .pCode = @as([*]const u32, @ptrCast(@alignCast(spv.ptr))) });
    var sm: vk.VkShaderModule = vk.VK_NULL_HANDLE;
    try std.testing.expectEqual(vk.VkResult.VK_SUCCESS, createShaderModule(device, &smci, null, &sm));
    defer destroyShaderModule(device, sm, null);
    const cpci = std.mem.zeroInit(vk.VkComputePipelineCreateInfo, .{
        .stage = std.mem.zeroInit(vk.VkPipelineShaderStageCreateInfo, .{ .stage = vk.VK_SHADER_STAGE_COMPUTE_BIT, .module = sm, .pName = "main" }),
        .layout = pl,
    });
    var pipe: vk.VkPipeline = vk.VK_NULL_HANDLE;
    try std.testing.expectEqual(vk.VkResult.VK_SUCCESS, createComputePipelines(device, vk.VK_NULL_HANDLE, 1, @ptrCast(&cpci), null, @ptrCast(&pipe)));
    defer destroyPipeline(device, pipe, null);

    // Descriptor pool + set + update (bind the 2 buffers).
    const psize = vk.VkDescriptorPoolSize{ .type = vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 2 };
    const dpci = std.mem.zeroInit(vk.VkDescriptorPoolCreateInfo, .{ .maxSets = 1, .poolSizeCount = 1, .pPoolSizes = @as([*]const vk.VkDescriptorPoolSize, @ptrCast(&psize)) });
    var pool: vk.VkDescriptorPool = vk.VK_NULL_HANDLE;
    try std.testing.expectEqual(vk.VkResult.VK_SUCCESS, createDescriptorPool(device, &dpci, null, &pool));
    defer destroyDescriptorPool(device, pool, null);
    const dsai = std.mem.zeroInit(vk.VkDescriptorSetAllocateInfo, .{ .descriptorPool = pool, .descriptorSetCount = 1, .pSetLayouts = @as([*]const vk.VkDescriptorSetLayout, @ptrCast(&dsl)) });
    var set: vk.VkDescriptorSet = vk.VK_NULL_HANDLE;
    try std.testing.expectEqual(vk.VkResult.VK_SUCCESS, allocateDescriptorSets(device, &dsai, @ptrCast(&set)));
    const bufinfos = [_]vk.VkDescriptorBufferInfo{
        .{ .buffer = in_buf, .offset = 0, .range = vk.VK_WHOLE_SIZE },
        .{ .buffer = out_buf, .offset = 0, .range = vk.VK_WHOLE_SIZE },
    };
    const writes = [_]vk.VkWriteDescriptorSet{
        std.mem.zeroInit(vk.VkWriteDescriptorSet, .{ .dstSet = set, .dstBinding = 0, .descriptorCount = 1, .descriptorType = vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = @as([*]const vk.VkDescriptorBufferInfo, @ptrCast(&bufinfos[0])) }),
        std.mem.zeroInit(vk.VkWriteDescriptorSet, .{ .dstSet = set, .dstBinding = 1, .descriptorCount = 1, .descriptorType = vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = @as([*]const vk.VkDescriptorBufferInfo, @ptrCast(&bufinfos[1])) }),
    };
    updateDescriptorSets(device, 2, &writes, 0, null);

    // Command pool + buffer: bind pipeline + descriptors, dispatch.
    const cpci2 = std.mem.zeroInit(vk.VkCommandPoolCreateInfo, .{});
    var cmd_pool: vk.VkCommandPool = vk.VK_NULL_HANDLE;
    try std.testing.expectEqual(vk.VkResult.VK_SUCCESS, createCommandPool(device, &cpci2, null, &cmd_pool));
    defer destroyCommandPool(device, cmd_pool, null);
    const cbai = std.mem.zeroInit(vk.VkCommandBufferAllocateInfo, .{ .commandPool = cmd_pool, .level = vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1 });
    var cmd: vk.VkCommandBuffer = null;
    try std.testing.expectEqual(vk.VkResult.VK_SUCCESS, allocateCommandBuffers(device, &cbai, @ptrCast(&cmd)));
    const bi = std.mem.zeroInit(vk.VkCommandBufferBeginInfo, .{});
    _ = beginCommandBuffer(cmd, &bi);
    cmdBindPipeline(cmd, vk.VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    cmdBindDescriptorSets(cmd, vk.VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, @ptrCast(&set), 0, null);
    cmdDispatch(cmd, N / 64, 1, 1);
    _ = endCommandBuffer(cmd);

    // Submit + wait, then verify output[i] == i + 0x100.
    const submit = std.mem.zeroInit(vk.VkSubmitInfo, .{ .commandBufferCount = 1, .pCommandBuffers = @as([*]const vk.VkCommandBuffer, @ptrCast(&cmd)) });
    try std.testing.expectEqual(vk.VkResult.VK_SUCCESS, queueSubmit(queue, 1, @ptrCast(&submit), vk.VK_NULL_HANDLE));
    _ = queueWaitIdle(queue);

    for (0..N) |i| {
        try std.testing.expectEqual(@as(u32, @as(u32, @intCast(i)) + 0x100), out_data[i]);
    }
}

test "device + memory + buffer roundtrip through the Prism HAL" {
    var inst: vk.VkInstance = null;
    const ci = vk.VkInstanceCreateInfo{
        .sType = .VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pNext = null,
        .flags = 0,
        .pApplicationInfo = null,
        .enabledLayerCount = 0,
        .ppEnabledLayerNames = null,
        .enabledExtensionCount = 0,
        .ppEnabledExtensionNames = null,
    };
    _ = createInstance(&ci, null, &inst);
    defer destroyInstance(inst, null);
    var count: u32 = 1;
    var phys: vk.VkPhysicalDevice = null;
    _ = enumeratePhysicalDevices(inst, &count, &phys);

    // Create the logical device (1 queue, family 0).
    const prio: f32 = 1.0;
    const qci = vk.VkDeviceQueueCreateInfo{
        .sType = .VK_STRUCTURE_TYPE_APPLICATION_INFO, // sType unchecked by us
        .pNext = null,
        .flags = 0,
        .queueFamilyIndex = 0,
        .queueCount = 1,
        .pQueuePriorities = @ptrCast(&prio),
    };
    const dci = vk.VkDeviceCreateInfo{
        .sType = .VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pNext = null,
        .flags = 0,
        .queueCreateInfoCount = 1,
        .pQueueCreateInfos = @ptrCast(&qci),
        .enabledLayerCount = 0,
        .ppEnabledLayerNames = null,
        .enabledExtensionCount = 0,
        .ppEnabledExtensionNames = null,
        .pEnabledFeatures = null,
    };
    var device: vk.VkDevice = null;
    try std.testing.expectEqual(vk.VkResult.VK_SUCCESS, createDevice(phys, &dci, null, &device));
    defer destroyDevice(device, null);
    try std.testing.expect(device != null);

    var queue: vk.VkQueue = null;
    getDeviceQueue(device, 0, 0, &queue);
    try std.testing.expect(queue != null);

    // Buffer.
    const bci = vk.VkBufferCreateInfo{
        .sType = .VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pNext = null,
        .flags = 0,
        .size = 1024,
        .usage = vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | vk.VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        .sharingMode = vk.VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices = null,
    };
    var buffer: vk.VkBuffer = vk.VK_NULL_HANDLE;
    try std.testing.expectEqual(vk.VkResult.VK_SUCCESS, createBuffer(device, &bci, null, &buffer));
    defer destroyBuffer(device, buffer, null);

    var reqs: vk.VkMemoryRequirements = undefined;
    getBufferMemoryRequirements(device, buffer, &reqs);
    try std.testing.expectEqual(@as(u64, 1024), reqs.size);
    try std.testing.expectEqual(@as(u64, 256), reqs.alignment);
    try std.testing.expectEqual(@as(u32, 0b1), reqs.memoryTypeBits);

    const mai = vk.VkMemoryAllocateInfo{
        .sType = .VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pNext = null,
        .allocationSize = reqs.size,
        .memoryTypeIndex = 0,
    };
    var memory: vk.VkDeviceMemory = vk.VK_NULL_HANDLE;
    try std.testing.expectEqual(vk.VkResult.VK_SUCCESS, allocateMemory(device, &mai, null, &memory));
    defer freeMemory(device, memory, null);

    try std.testing.expectEqual(vk.VkResult.VK_SUCCESS, bindBufferMemory(device, buffer, memory, 0));

    var ptr: ?*anyopaque = null;
    try std.testing.expectEqual(vk.VkResult.VK_SUCCESS, mapMemory(device, memory, 0, vk.VK_WHOLE_SIZE, 0, &ptr));
    try std.testing.expect(ptr != null);
    const data: [*]u32 = @ptrCast(@alignCast(ptr.?));
    data[0] = 0xABCD1234;
    data[1] = 0xCAFEF00D;
    // Read back through the SAME HAL-backed mapping.
    try std.testing.expectEqual(@as(u32, 0xABCD1234), data[0]);
    try std.testing.expectEqual(@as(u32, 0xCAFEF00D), data[1]);
    unmapMemory(device, memory);
}

test "offscreen triangle renders through the ICD and reads back" {
    // Select the software physical device (the rasterizer the graphics path drives).
    var inst: vk.VkInstance = null;
    const ci = std.mem.zeroInit(vk.VkInstanceCreateInfo, .{ .sType = .VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO });
    _ = createInstance(&ci, null, &inst);
    defer destroyInstance(inst, null);
    var count: u32 = 0;
    _ = enumeratePhysicalDevices(inst, &count, null);
    if (count == 0) return error.SkipZigTest;
    var handles: [16]vk.VkPhysicalDevice = undefined;
    var got: u32 = @min(count, 16);
    _ = enumeratePhysicalDevices(inst, &got, &handles[0]);
    var phys: vk.VkPhysicalDevice = null;
    for (handles[0..got]) |h| {
        var props: vk.VkPhysicalDeviceProperties = undefined;
        getPhysicalDeviceProperties(h, &props);
        if (std.mem.indexOf(u8, &props.deviceName, "software") != null) {
            phys = h;
            break;
        }
    }
    if (phys == null) return error.SkipZigTest;
    const dci = std.mem.zeroes(vk.VkDeviceCreateInfo);
    var device: vk.VkDevice = null;
    try std.testing.expectEqual(vk.VkResult.VK_SUCCESS, createDevice(phys, &dci, null, &device));
    defer destroyDevice(device, null);
    var queue: vk.VkQueue = null;
    getDeviceQueue(device, 0, 0, &queue);

    const W = 64;
    const H = 64;

    // Offscreen color image (RGBA8) + memory + bind + view.
    const ici = std.mem.zeroInit(vk.VkImageCreateInfo, .{
        .imageType = vk.VK_IMAGE_TYPE_2D,
        .format = vk.VK_FORMAT_R8G8B8A8_UNORM,
        .extent = .{ .width = W, .height = H, .depth = 1 },
        .mipLevels = 1,
        .arrayLayers = 1,
        .usage = vk.VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | vk.VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
    });
    var image: vk.VkImage = vk.VK_NULL_HANDLE;
    try std.testing.expectEqual(vk.VkResult.VK_SUCCESS, createImage(device, &ici, null, &image));
    defer destroyImage(device, image, null);
    var ireqs: vk.VkMemoryRequirements = undefined;
    getImageMemoryRequirements(device, image, &ireqs);
    const imai = std.mem.zeroInit(vk.VkMemoryAllocateInfo, .{ .allocationSize = ireqs.size });
    var imem: vk.VkDeviceMemory = vk.VK_NULL_HANDLE;
    _ = allocateMemory(device, &imai, null, &imem);
    defer freeMemory(device, imem, null);
    _ = bindImageMemory(device, image, imem, 0);
    const ivci = std.mem.zeroInit(vk.VkImageViewCreateInfo, .{ .image = image, .viewType = vk.VK_IMAGE_VIEW_TYPE_2D, .format = vk.VK_FORMAT_R8G8B8A8_UNORM });
    var view: vk.VkImageView = vk.VK_NULL_HANDLE;
    try std.testing.expectEqual(vk.VkResult.VK_SUCCESS, createImageView(device, &ivci, null, &view));
    defer destroyImageView(device, view, null);

    // Render pass (1 color attachment: CLEAR -> STORE) + framebuffer.
    const att = vk.VkAttachmentDescription{
        .flags = 0,
        .format = vk.VK_FORMAT_R8G8B8A8_UNORM,
        .samples = 1,
        .loadOp = vk.VK_ATTACHMENT_LOAD_OP_CLEAR,
        .storeOp = vk.VK_ATTACHMENT_STORE_OP_STORE,
        .stencilLoadOp = vk.VK_ATTACHMENT_LOAD_OP_CLEAR,
        .stencilStoreOp = vk.VK_ATTACHMENT_STORE_OP_STORE,
        .initialLayout = vk.VK_IMAGE_LAYOUT_UNDEFINED,
        .finalLayout = vk.VK_IMAGE_LAYOUT_UNDEFINED,
    };
    const ref = vk.VkAttachmentReference{ .attachment = 0, .layout = vk.VK_IMAGE_LAYOUT_UNDEFINED };
    const sub = std.mem.zeroInit(vk.VkSubpassDescription, .{ .colorAttachmentCount = 1, .pColorAttachments = @as([*]const vk.VkAttachmentReference, @ptrCast(&ref)) });
    const rpci = std.mem.zeroInit(vk.VkRenderPassCreateInfo, .{ .attachmentCount = 1, .pAttachments = @as([*]const vk.VkAttachmentDescription, @ptrCast(&att)), .subpassCount = 1, .pSubpasses = @as([*]const vk.VkSubpassDescription, @ptrCast(&sub)) });
    var rp: vk.VkRenderPass = vk.VK_NULL_HANDLE;
    try std.testing.expectEqual(vk.VkResult.VK_SUCCESS, createRenderPass(device, &rpci, null, &rp));
    defer destroyRenderPass(device, rp, null);
    const fbci = std.mem.zeroInit(vk.VkFramebufferCreateInfo, .{ .renderPass = rp, .attachmentCount = 1, .pAttachments = @as([*]const vk.VkImageView, @ptrCast(&view)), .width = W, .height = H, .layers = 1 });
    var fb: vk.VkFramebuffer = vk.VK_NULL_HANDLE;
    try std.testing.expectEqual(vk.VkResult.VK_SUCCESS, createFramebuffer(device, &fbci, null, &fb));
    defer destroyFramebuffer(device, fb, null);

    // Vertex buffer: 3 vertices, pos vec2 + color vec3, covering the center.
    const Vtx = extern struct { x: f32, y: f32, r: f32, g: f32, b: f32 };
    const verts = [3]Vtx{
        .{ .x = -0.8, .y = -0.8, .r = 1, .g = 0, .b = 0 },
        .{ .x = 0.8, .y = -0.8, .r = 0, .g = 1, .b = 0 },
        .{ .x = 0.0, .y = 0.8, .r = 0, .g = 0, .b = 1 },
    };
    var vbuf: vk.VkBuffer = vk.VK_NULL_HANDLE;
    const vbci = std.mem.zeroInit(vk.VkBufferCreateInfo, .{ .size = @sizeOf(@TypeOf(verts)), .usage = vk.VK_BUFFER_USAGE_VERTEX_BUFFER_BIT });
    try std.testing.expectEqual(vk.VkResult.VK_SUCCESS, createBuffer(device, &vbci, null, &vbuf));
    defer destroyBuffer(device, vbuf, null);
    var vreqs: vk.VkMemoryRequirements = undefined;
    getBufferMemoryRequirements(device, vbuf, &vreqs);
    const vmai = std.mem.zeroInit(vk.VkMemoryAllocateInfo, .{ .allocationSize = vreqs.size });
    var vmem: vk.VkDeviceMemory = vk.VK_NULL_HANDLE;
    _ = allocateMemory(device, &vmai, null, &vmem);
    defer freeMemory(device, vmem, null);
    _ = bindBufferMemory(device, vbuf, vmem, 0);
    var vp: ?*anyopaque = null;
    _ = mapMemory(device, vmem, 0, vk.VK_WHOLE_SIZE, 0, &vp);
    @memcpy(@as([*]u8, @ptrCast(vp.?))[0..@sizeOf(@TypeOf(verts))], std.mem.asBytes(&verts));

    // VS+FS shader modules: REAL passthrough SPIR-V. The software driver JITs and
    // executes these per-vertex / per-fragment (the general SPIR-V graphics path),
    // not a declarative mapping. VS: in pos(loc0)+color(loc1) -> gl_Position + out
    // color(loc0); FS: in color(loc0) -> out vec4(color, 1).
    const op = prism.spirv.opcodes;
    const sc = op.StorageClass;
    var vsb = try prism.spirv.binary.Builder.init(std.testing.allocator, 24);
    defer vsb.deinit(std.testing.allocator);
    try vsb.emit(std.testing.allocator, op.EntryPoint, &.{ op.ExecutionModel.vertex, 15, 0, 11, 12, 13, 14 });
    try vsb.emit(std.testing.allocator, op.Decorate, &.{ 13, op.Decoration.builtin, op.BuiltIn.position });
    try vsb.emit(std.testing.allocator, op.Decorate, &.{ 11, op.Decoration.location, 0 });
    try vsb.emit(std.testing.allocator, op.Decorate, &.{ 12, op.Decoration.location, 1 });
    try vsb.emit(std.testing.allocator, op.Decorate, &.{ 14, op.Decoration.location, 0 });
    try vsb.emit(std.testing.allocator, op.TypeVoid, &.{1});
    try vsb.emit(std.testing.allocator, op.TypeFunction, &.{ 2, 1 });
    try vsb.emit(std.testing.allocator, op.TypeFloat, &.{ 3, 32 });
    try vsb.emit(std.testing.allocator, op.TypeVector, &.{ 4, 3, 2 });
    try vsb.emit(std.testing.allocator, op.TypeVector, &.{ 5, 3, 3 });
    try vsb.emit(std.testing.allocator, op.TypeVector, &.{ 6, 3, 4 });
    try vsb.emit(std.testing.allocator, op.TypePointer, &.{ 7, sc.input, 4 });
    try vsb.emit(std.testing.allocator, op.TypePointer, &.{ 8, sc.input, 5 });
    try vsb.emit(std.testing.allocator, op.TypePointer, &.{ 9, sc.output, 6 });
    try vsb.emit(std.testing.allocator, op.TypePointer, &.{ 10, sc.output, 5 });
    try vsb.emit(std.testing.allocator, op.Constant, &.{ 3, 17, @bitCast(@as(f32, 0.0)) });
    try vsb.emit(std.testing.allocator, op.Constant, &.{ 3, 18, @bitCast(@as(f32, 1.0)) });
    try vsb.emit(std.testing.allocator, op.Variable, &.{ 7, 11, sc.input });
    try vsb.emit(std.testing.allocator, op.Variable, &.{ 8, 12, sc.input });
    try vsb.emit(std.testing.allocator, op.Variable, &.{ 9, 13, sc.output });
    try vsb.emit(std.testing.allocator, op.Variable, &.{ 10, 14, sc.output });
    try vsb.emit(std.testing.allocator, op.Function, &.{ 1, 15, 0, 2 });
    try vsb.emit(std.testing.allocator, op.Label, &.{16});
    try vsb.emit(std.testing.allocator, op.Load, &.{ 4, 19, 11 });
    try vsb.emit(std.testing.allocator, op.Load, &.{ 5, 20, 12 });
    try vsb.emit(std.testing.allocator, op.CompositeExtract, &.{ 3, 21, 19, 0 });
    try vsb.emit(std.testing.allocator, op.CompositeExtract, &.{ 3, 22, 19, 1 });
    try vsb.emit(std.testing.allocator, op.CompositeConstruct, &.{ 6, 23, 21, 22, 17, 18 });
    try vsb.emit(std.testing.allocator, op.Store, &.{ 13, 23 });
    try vsb.emit(std.testing.allocator, op.Store, &.{ 14, 20 });
    try vsb.emit(std.testing.allocator, op.Return, &.{});
    try vsb.emit(std.testing.allocator, op.FunctionEnd, &.{});

    var fsb = try prism.spirv.binary.Builder.init(std.testing.allocator, 18);
    defer fsb.deinit(std.testing.allocator);
    try fsb.emit(std.testing.allocator, op.EntryPoint, &.{ op.ExecutionModel.fragment, 10, 0, 8, 9 });
    try fsb.emit(std.testing.allocator, op.Decorate, &.{ 8, op.Decoration.location, 0 });
    try fsb.emit(std.testing.allocator, op.Decorate, &.{ 9, op.Decoration.location, 0 });
    try fsb.emit(std.testing.allocator, op.TypeVoid, &.{1});
    try fsb.emit(std.testing.allocator, op.TypeFunction, &.{ 2, 1 });
    try fsb.emit(std.testing.allocator, op.TypeFloat, &.{ 3, 32 });
    try fsb.emit(std.testing.allocator, op.TypeVector, &.{ 4, 3, 3 });
    try fsb.emit(std.testing.allocator, op.TypeVector, &.{ 5, 3, 4 });
    try fsb.emit(std.testing.allocator, op.TypePointer, &.{ 6, sc.input, 4 });
    try fsb.emit(std.testing.allocator, op.TypePointer, &.{ 7, sc.output, 5 });
    try fsb.emit(std.testing.allocator, op.Constant, &.{ 3, 12, @bitCast(@as(f32, 1.0)) });
    try fsb.emit(std.testing.allocator, op.Variable, &.{ 6, 8, sc.input });
    try fsb.emit(std.testing.allocator, op.Variable, &.{ 7, 9, sc.output });
    try fsb.emit(std.testing.allocator, op.Function, &.{ 1, 10, 0, 2 });
    try fsb.emit(std.testing.allocator, op.Label, &.{11});
    try fsb.emit(std.testing.allocator, op.Load, &.{ 4, 13, 8 });
    try fsb.emit(std.testing.allocator, op.CompositeExtract, &.{ 3, 14, 13, 0 });
    try fsb.emit(std.testing.allocator, op.CompositeExtract, &.{ 3, 15, 13, 1 });
    try fsb.emit(std.testing.allocator, op.CompositeExtract, &.{ 3, 16, 13, 2 });
    try fsb.emit(std.testing.allocator, op.CompositeConstruct, &.{ 5, 17, 14, 15, 16, 12 });
    try fsb.emit(std.testing.allocator, op.Store, &.{ 9, 17 });
    try fsb.emit(std.testing.allocator, op.Return, &.{});
    try fsb.emit(std.testing.allocator, op.FunctionEnd, &.{});

    const vs_smci = std.mem.zeroInit(vk.VkShaderModuleCreateInfo, .{ .codeSize = vsb.words.items.len * 4, .pCode = vsb.words.items.ptr });
    const fs_smci = std.mem.zeroInit(vk.VkShaderModuleCreateInfo, .{ .codeSize = fsb.words.items.len * 4, .pCode = fsb.words.items.ptr });
    var vs: vk.VkShaderModule = vk.VK_NULL_HANDLE;
    var fs: vk.VkShaderModule = vk.VK_NULL_HANDLE;
    _ = createShaderModule(device, &vs_smci, null, &vs);
    defer destroyShaderModule(device, vs, null);
    _ = createShaderModule(device, &fs_smci, null, &fs);
    defer destroyShaderModule(device, fs, null);

    // Pipeline layout (empty) + graphics pipeline.
    const plci = std.mem.zeroInit(vk.VkPipelineLayoutCreateInfo, .{});
    var pl: vk.VkPipelineLayout = vk.VK_NULL_HANDLE;
    _ = createPipelineLayout(device, &plci, null, &pl);
    defer destroyPipelineLayout(device, pl, null);
    const stages = [_]vk.VkPipelineShaderStageCreateInfo{
        std.mem.zeroInit(vk.VkPipelineShaderStageCreateInfo, .{ .stage = vk.VK_SHADER_STAGE_VERTEX_BIT, .module = vs, .pName = "main" }),
        std.mem.zeroInit(vk.VkPipelineShaderStageCreateInfo, .{ .stage = vk.VK_SHADER_STAGE_FRAGMENT_BIT, .module = fs, .pName = "main" }),
    };
    const bind = vk.VkVertexInputBindingDescription{ .binding = 0, .stride = @sizeOf(Vtx), .inputRate = 0 };
    const vattrs = [_]vk.VkVertexInputAttributeDescription{
        .{ .location = 0, .binding = 0, .format = vk.VK_FORMAT_R32G32_SFLOAT, .offset = 0 },
        .{ .location = 1, .binding = 0, .format = vk.VK_FORMAT_R32G32B32_SFLOAT, .offset = 8 },
    };
    const vis = std.mem.zeroInit(vk.VkPipelineVertexInputStateCreateInfo, .{
        .vertexBindingDescriptionCount = 1,
        .pVertexBindingDescriptions = @as([*]const vk.VkVertexInputBindingDescription, @ptrCast(&bind)),
        .vertexAttributeDescriptionCount = 2,
        .pVertexAttributeDescriptions = &vattrs,
    });
    const gpci = std.mem.zeroInit(vk.VkGraphicsPipelineCreateInfo, .{
        .stageCount = 2,
        .pStages = &stages,
        .pVertexInputState = &vis,
        .layout = pl,
        .renderPass = rp,
    });
    var gpipe: vk.VkPipeline = vk.VK_NULL_HANDLE;
    try std.testing.expectEqual(vk.VkResult.VK_SUCCESS, createGraphicsPipelines(device, vk.VK_NULL_HANDLE, 1, @ptrCast(&gpci), null, @ptrCast(&gpipe)));
    defer destroyPipeline(device, gpipe, null);

    // Readback buffer (W*H*4) + memory.
    const rb_size = W * H * 4;
    var rbuf: vk.VkBuffer = vk.VK_NULL_HANDLE;
    const rbci = std.mem.zeroInit(vk.VkBufferCreateInfo, .{ .size = rb_size, .usage = vk.VK_BUFFER_USAGE_TRANSFER_DST_BIT });
    try std.testing.expectEqual(vk.VkResult.VK_SUCCESS, createBuffer(device, &rbci, null, &rbuf));
    defer destroyBuffer(device, rbuf, null);
    var rreqs: vk.VkMemoryRequirements = undefined;
    getBufferMemoryRequirements(device, rbuf, &rreqs);
    const rmai = std.mem.zeroInit(vk.VkMemoryAllocateInfo, .{ .allocationSize = rreqs.size });
    var rmem: vk.VkDeviceMemory = vk.VK_NULL_HANDLE;
    _ = allocateMemory(device, &rmai, null, &rmem);
    defer freeMemory(device, rmem, null);
    _ = bindBufferMemory(device, rbuf, rmem, 0);

    // Record: beginRenderPass(clear black) / bindPipeline / bindVB / draw(3) /
    // endRenderPass / copyImageToBuffer.
    const cpci2 = std.mem.zeroInit(vk.VkCommandPoolCreateInfo, .{});
    var cmd_pool: vk.VkCommandPool = vk.VK_NULL_HANDLE;
    _ = createCommandPool(device, &cpci2, null, &cmd_pool);
    defer destroyCommandPool(device, cmd_pool, null);
    const cbai = std.mem.zeroInit(vk.VkCommandBufferAllocateInfo, .{ .commandPool = cmd_pool, .level = vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1 });
    var cmd: vk.VkCommandBuffer = null;
    _ = allocateCommandBuffers(device, &cbai, @ptrCast(&cmd));
    const bi = std.mem.zeroInit(vk.VkCommandBufferBeginInfo, .{});
    _ = beginCommandBuffer(cmd, &bi);
    var clearv: vk.VkClearValue = .{ .color = .{ .float32 = .{ 0, 0, 0, 1 } } };
    const rpbi = std.mem.zeroInit(vk.VkRenderPassBeginInfo, .{
        .renderPass = rp,
        .framebuffer = fb,
        .renderArea = .{ .offset = .{ .x = 0, .y = 0 }, .extent = .{ .width = W, .height = H } },
        .clearValueCount = 1,
        .pClearValues = @as([*]const vk.VkClearValue, @ptrCast(&clearv)),
    });
    cmdBeginRenderPass(cmd, &rpbi, vk.VK_SUBPASS_CONTENTS_INLINE);
    cmdBindPipeline(cmd, vk.VK_PIPELINE_BIND_POINT_GRAPHICS, gpipe);
    const voff: vk.VkDeviceSize = 0;
    cmdBindVertexBuffers(cmd, 0, 1, @ptrCast(&vbuf), @ptrCast(&voff));
    cmdDraw(cmd, 3, 1, 0, 0);
    cmdEndRenderPass(cmd);
    const region = std.mem.zeroInit(vk.VkBufferImageCopy, .{
        .imageSubresource = .{ .aspectMask = vk.VK_IMAGE_ASPECT_COLOR_BIT, .mipLevel = 0, .baseArrayLayer = 0, .layerCount = 1 },
        .imageExtent = .{ .width = W, .height = H, .depth = 1 },
    });
    cmdCopyImageToBuffer(cmd, image, vk.VK_IMAGE_LAYOUT_UNDEFINED, rbuf, 1, @ptrCast(&region));
    _ = endCommandBuffer(cmd);

    // Submit + wait.
    const submit = std.mem.zeroInit(vk.VkSubmitInfo, .{ .commandBufferCount = 1, .pCommandBuffers = @as([*]const vk.VkCommandBuffer, @ptrCast(&cmd)) });
    try std.testing.expectEqual(vk.VkResult.VK_SUCCESS, queueSubmit(queue, 1, @ptrCast(&submit), vk.VK_NULL_HANDLE));
    _ = queueWaitIdle(queue);

    // Map the readback buffer. Center must be non-black (the triangle), a corner
    // must be the clear color (black).
    var rp_ptr: ?*anyopaque = null;
    _ = mapMemory(device, rmem, 0, vk.VK_WHOLE_SIZE, 0, &rp_ptr);
    const px: [*]u8 = @ptrCast(rp_ptr.?);
    const center = ((H / 2) * W + (W / 2)) * 4;
    const corner = (1 * W + 1) * 4;
    const center_nonblack = px[center] != 0 or px[center + 1] != 0 or px[center + 2] != 0;
    const corner_black = px[corner] == 0 and px[corner + 1] == 0 and px[corner + 2] == 0;
    try std.testing.expect(center_nonblack);
    try std.testing.expect(corner_black);
}

test "vkCmdDrawIndexed: an indexed quad (4 verts, 6 indices) fills the screen, reusing shared vertices" {
    const device = makeSoftwareDevice() orelse return error.SkipZigTest;
    defer destroyDevice(device, null);
    var queue: vk.VkQueue = null;
    getDeviceQueue(device, 0, 0, &queue);
    const W = 64;
    const H = 64;

    // Offscreen color image + view.
    const ici = std.mem.zeroInit(vk.VkImageCreateInfo, .{ .imageType = vk.VK_IMAGE_TYPE_2D, .format = vk.VK_FORMAT_R8G8B8A8_UNORM, .extent = .{ .width = W, .height = H, .depth = 1 }, .mipLevels = 1, .arrayLayers = 1, .usage = vk.VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | vk.VK_IMAGE_USAGE_TRANSFER_SRC_BIT });
    var image: vk.VkImage = vk.VK_NULL_HANDLE;
    try std.testing.expectEqual(vk.VkResult.VK_SUCCESS, createImage(device, &ici, null, &image));
    defer destroyImage(device, image, null);
    var ireqs: vk.VkMemoryRequirements = undefined;
    getImageMemoryRequirements(device, image, &ireqs);
    const imai = std.mem.zeroInit(vk.VkMemoryAllocateInfo, .{ .allocationSize = ireqs.size });
    var imem: vk.VkDeviceMemory = vk.VK_NULL_HANDLE;
    _ = allocateMemory(device, &imai, null, &imem);
    defer freeMemory(device, imem, null);
    _ = bindImageMemory(device, image, imem, 0);
    const ivci = std.mem.zeroInit(vk.VkImageViewCreateInfo, .{ .image = image, .viewType = vk.VK_IMAGE_VIEW_TYPE_2D, .format = vk.VK_FORMAT_R8G8B8A8_UNORM });
    var view: vk.VkImageView = vk.VK_NULL_HANDLE;
    try std.testing.expectEqual(vk.VkResult.VK_SUCCESS, createImageView(device, &ivci, null, &view));
    defer destroyImageView(device, view, null);

    // Render pass (1 color: CLEAR->STORE) + framebuffer.
    const att = vk.VkAttachmentDescription{ .flags = 0, .format = vk.VK_FORMAT_R8G8B8A8_UNORM, .samples = 1, .loadOp = vk.VK_ATTACHMENT_LOAD_OP_CLEAR, .storeOp = vk.VK_ATTACHMENT_STORE_OP_STORE, .stencilLoadOp = vk.VK_ATTACHMENT_LOAD_OP_CLEAR, .stencilStoreOp = vk.VK_ATTACHMENT_STORE_OP_STORE, .initialLayout = vk.VK_IMAGE_LAYOUT_UNDEFINED, .finalLayout = vk.VK_IMAGE_LAYOUT_UNDEFINED };
    const ref = vk.VkAttachmentReference{ .attachment = 0, .layout = vk.VK_IMAGE_LAYOUT_UNDEFINED };
    const sub = std.mem.zeroInit(vk.VkSubpassDescription, .{ .colorAttachmentCount = 1, .pColorAttachments = @as([*]const vk.VkAttachmentReference, @ptrCast(&ref)) });
    const rpci = std.mem.zeroInit(vk.VkRenderPassCreateInfo, .{ .attachmentCount = 1, .pAttachments = @as([*]const vk.VkAttachmentDescription, @ptrCast(&att)), .subpassCount = 1, .pSubpasses = @as([*]const vk.VkSubpassDescription, @ptrCast(&sub)) });
    var rp: vk.VkRenderPass = vk.VK_NULL_HANDLE;
    try std.testing.expectEqual(vk.VkResult.VK_SUCCESS, createRenderPass(device, &rpci, null, &rp));
    defer destroyRenderPass(device, rp, null);
    const fbci = std.mem.zeroInit(vk.VkFramebufferCreateInfo, .{ .renderPass = rp, .attachmentCount = 1, .pAttachments = @as([*]const vk.VkImageView, @ptrCast(&view)), .width = W, .height = H, .layers = 1 });
    var fb: vk.VkFramebuffer = vk.VK_NULL_HANDLE;
    try std.testing.expectEqual(vk.VkResult.VK_SUCCESS, createFramebuffer(device, &fbci, null, &fb));
    defer destroyFramebuffer(device, fb, null);

    // Vertex buffer: 4 quad corners (red). A full-screen quad is 2 triangles = 6 indices reusing
    // verts 0 and 2, so a correct indexed draw reads only 4 vertices for 6 emitted.
    const Vtx = extern struct { x: f32, y: f32, r: f32, g: f32, b: f32 };
    const verts = [4]Vtx{
        .{ .x = -1, .y = -1, .r = 1, .g = 0, .b = 0 },
        .{ .x = 1, .y = -1, .r = 1, .g = 0, .b = 0 },
        .{ .x = 1, .y = 1, .r = 1, .g = 0, .b = 0 },
        .{ .x = -1, .y = 1, .r = 1, .g = 0, .b = 0 },
    };
    var vbuf: vk.VkBuffer = vk.VK_NULL_HANDLE;
    const vbci = std.mem.zeroInit(vk.VkBufferCreateInfo, .{ .size = @sizeOf(@TypeOf(verts)), .usage = vk.VK_BUFFER_USAGE_VERTEX_BUFFER_BIT });
    try std.testing.expectEqual(vk.VkResult.VK_SUCCESS, createBuffer(device, &vbci, null, &vbuf));
    defer destroyBuffer(device, vbuf, null);
    var vreqs: vk.VkMemoryRequirements = undefined;
    getBufferMemoryRequirements(device, vbuf, &vreqs);
    const vmai = std.mem.zeroInit(vk.VkMemoryAllocateInfo, .{ .allocationSize = vreqs.size });
    var vmem: vk.VkDeviceMemory = vk.VK_NULL_HANDLE;
    _ = allocateMemory(device, &vmai, null, &vmem);
    defer freeMemory(device, vmem, null);
    _ = bindBufferMemory(device, vbuf, vmem, 0);
    var vp: ?*anyopaque = null;
    _ = mapMemory(device, vmem, 0, vk.VK_WHOLE_SIZE, 0, &vp);
    @memcpy(@as([*]u8, @ptrCast(vp.?))[0..@sizeOf(@TypeOf(verts))], std.mem.asBytes(&verts));

    // Index buffer (uint16): two triangles (0,1,2) + (0,2,3).
    const indices = [6]u16{ 0, 1, 2, 0, 2, 3 };
    var ibuf: vk.VkBuffer = vk.VK_NULL_HANDLE;
    const ibci = std.mem.zeroInit(vk.VkBufferCreateInfo, .{ .size = @sizeOf(@TypeOf(indices)), .usage = vk.VK_BUFFER_USAGE_INDEX_BUFFER_BIT });
    try std.testing.expectEqual(vk.VkResult.VK_SUCCESS, createBuffer(device, &ibci, null, &ibuf));
    defer destroyBuffer(device, ibuf, null);
    var idxreqs: vk.VkMemoryRequirements = undefined;
    getBufferMemoryRequirements(device, ibuf, &idxreqs);
    const idxmai = std.mem.zeroInit(vk.VkMemoryAllocateInfo, .{ .allocationSize = idxreqs.size });
    var idxmem: vk.VkDeviceMemory = vk.VK_NULL_HANDLE;
    _ = allocateMemory(device, &idxmai, null, &idxmem);
    defer freeMemory(device, idxmem, null);
    _ = bindBufferMemory(device, ibuf, idxmem, 0);
    var ip: ?*anyopaque = null;
    _ = mapMemory(device, idxmem, 0, vk.VK_WHOLE_SIZE, 0, &ip);
    @memcpy(@as([*]u8, @ptrCast(ip.?))[0..@sizeOf(@TypeOf(indices))], std.mem.asBytes(&indices));

    // Shaders (in vec2 pos loc0 + vec3 color loc1) + pipeline.
    const sh = makePassthroughShaders(device) orelse return error.SkipZigTest;
    defer destroyShaderModule(device, sh.vs, null);
    defer destroyShaderModule(device, sh.fs, null);
    const plci = std.mem.zeroInit(vk.VkPipelineLayoutCreateInfo, .{});
    var pl: vk.VkPipelineLayout = vk.VK_NULL_HANDLE;
    _ = createPipelineLayout(device, &plci, null, &pl);
    defer destroyPipelineLayout(device, pl, null);
    const stages = [_]vk.VkPipelineShaderStageCreateInfo{
        std.mem.zeroInit(vk.VkPipelineShaderStageCreateInfo, .{ .stage = vk.VK_SHADER_STAGE_VERTEX_BIT, .module = sh.vs, .pName = "main" }),
        std.mem.zeroInit(vk.VkPipelineShaderStageCreateInfo, .{ .stage = vk.VK_SHADER_STAGE_FRAGMENT_BIT, .module = sh.fs, .pName = "main" }),
    };
    const bind = vk.VkVertexInputBindingDescription{ .binding = 0, .stride = @sizeOf(Vtx), .inputRate = 0 };
    const vattrs = [_]vk.VkVertexInputAttributeDescription{
        .{ .location = 0, .binding = 0, .format = vk.VK_FORMAT_R32G32_SFLOAT, .offset = 0 },
        .{ .location = 1, .binding = 0, .format = vk.VK_FORMAT_R32G32B32_SFLOAT, .offset = 8 },
    };
    const vis = std.mem.zeroInit(vk.VkPipelineVertexInputStateCreateInfo, .{ .vertexBindingDescriptionCount = 1, .pVertexBindingDescriptions = @as([*]const vk.VkVertexInputBindingDescription, @ptrCast(&bind)), .vertexAttributeDescriptionCount = 2, .pVertexAttributeDescriptions = &vattrs });
    const gpci = std.mem.zeroInit(vk.VkGraphicsPipelineCreateInfo, .{ .stageCount = 2, .pStages = &stages, .pVertexInputState = &vis, .layout = pl, .renderPass = rp });
    var gpipe: vk.VkPipeline = vk.VK_NULL_HANDLE;
    try std.testing.expectEqual(vk.VkResult.VK_SUCCESS, createGraphicsPipelines(device, vk.VK_NULL_HANDLE, 1, @ptrCast(&gpci), null, @ptrCast(&gpipe)));
    defer destroyPipeline(device, gpipe, null);

    // Readback buffer.
    const rb_size = W * H * 4;
    var rbuf: vk.VkBuffer = vk.VK_NULL_HANDLE;
    const rbci = std.mem.zeroInit(vk.VkBufferCreateInfo, .{ .size = rb_size, .usage = vk.VK_BUFFER_USAGE_TRANSFER_DST_BIT });
    try std.testing.expectEqual(vk.VkResult.VK_SUCCESS, createBuffer(device, &rbci, null, &rbuf));
    defer destroyBuffer(device, rbuf, null);
    var rreqs: vk.VkMemoryRequirements = undefined;
    getBufferMemoryRequirements(device, rbuf, &rreqs);
    const rmai = std.mem.zeroInit(vk.VkMemoryAllocateInfo, .{ .allocationSize = rreqs.size });
    var rmem: vk.VkDeviceMemory = vk.VK_NULL_HANDLE;
    _ = allocateMemory(device, &rmai, null, &rmem);
    defer freeMemory(device, rmem, null);
    _ = bindBufferMemory(device, rbuf, rmem, 0);

    // Record: beginRP / bindPipeline / bindVB / bindIB / drawIndexed(6) / endRP / copyImageToBuffer.
    const cpci2 = std.mem.zeroInit(vk.VkCommandPoolCreateInfo, .{});
    var cmd_pool: vk.VkCommandPool = vk.VK_NULL_HANDLE;
    _ = createCommandPool(device, &cpci2, null, &cmd_pool);
    defer destroyCommandPool(device, cmd_pool, null);
    const cbai = std.mem.zeroInit(vk.VkCommandBufferAllocateInfo, .{ .commandPool = cmd_pool, .level = vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1 });
    var cmd: vk.VkCommandBuffer = null;
    _ = allocateCommandBuffers(device, &cbai, @ptrCast(&cmd));
    const bi = std.mem.zeroInit(vk.VkCommandBufferBeginInfo, .{});
    _ = beginCommandBuffer(cmd, &bi);
    var clearv: vk.VkClearValue = .{ .color = .{ .float32 = .{ 0, 0, 0, 1 } } };
    const rpbi = std.mem.zeroInit(vk.VkRenderPassBeginInfo, .{ .renderPass = rp, .framebuffer = fb, .renderArea = .{ .offset = .{ .x = 0, .y = 0 }, .extent = .{ .width = W, .height = H } }, .clearValueCount = 1, .pClearValues = @as([*]const vk.VkClearValue, @ptrCast(&clearv)) });
    cmdBeginRenderPass(cmd, &rpbi, vk.VK_SUBPASS_CONTENTS_INLINE);
    cmdBindPipeline(cmd, vk.VK_PIPELINE_BIND_POINT_GRAPHICS, gpipe);
    const voff: vk.VkDeviceSize = 0;
    cmdBindVertexBuffers(cmd, 0, 1, @ptrCast(&vbuf), @ptrCast(&voff));
    cmdBindIndexBuffer(cmd, ibuf, 0, vk.VK_INDEX_TYPE_UINT16);
    cmdDrawIndexed(cmd, 6, 1, 0, 0, 0);
    cmdEndRenderPass(cmd);
    const region = std.mem.zeroInit(vk.VkBufferImageCopy, .{ .imageSubresource = .{ .aspectMask = vk.VK_IMAGE_ASPECT_COLOR_BIT, .mipLevel = 0, .baseArrayLayer = 0, .layerCount = 1 }, .imageExtent = .{ .width = W, .height = H, .depth = 1 } });
    cmdCopyImageToBuffer(cmd, image, vk.VK_IMAGE_LAYOUT_UNDEFINED, rbuf, 1, @ptrCast(&region));
    _ = endCommandBuffer(cmd);
    const submit = std.mem.zeroInit(vk.VkSubmitInfo, .{ .commandBufferCount = 1, .pCommandBuffers = @as([*]const vk.VkCommandBuffer, @ptrCast(&cmd)) });
    try std.testing.expectEqual(vk.VkResult.VK_SUCCESS, queueSubmit(queue, 1, @ptrCast(&submit), vk.VK_NULL_HANDLE));
    _ = queueWaitIdle(queue);

    // The indexed quad covers the whole screen: center + all four near-corners must be red. A broken
    // indexed path (drawing 6 sequential verts from a 4-vert buffer, or wrong index gather) would
    // leave holes or garbage.
    var rp_ptr: ?*anyopaque = null;
    _ = mapMemory(device, rmem, 0, vk.VK_WHOLE_SIZE, 0, &rp_ptr);
    const px: [*]u8 = @ptrCast(rp_ptr.?);
    const isRed = struct {
        fn at(p: [*]u8, x: usize, y: usize) bool {
            const i = (y * W + x) * 4;
            return p[i] > 200 and p[i + 1] < 60 and p[i + 2] < 60;
        }
    }.at;
    try std.testing.expect(isRed(px, W / 2, H / 2)); // center
    try std.testing.expect(isRed(px, 3, 3)); // near each corner (quad is fullscreen)
    try std.testing.expect(isRed(px, W - 4, 3));
    try std.testing.expect(isRed(px, 3, H - 4));
    try std.testing.expect(isRed(px, W - 4, H - 4));
}

test "alpha blending: a CONSTANT_ALPHA-blended red draw over a blue clear yields 50/50 purple" {
    const device = makeSoftwareDevice() orelse return error.SkipZigTest;
    defer destroyDevice(device, null);
    var queue: vk.VkQueue = null;
    getDeviceQueue(device, 0, 0, &queue);
    const W = 64;
    const H = 64;

    const ici = std.mem.zeroInit(vk.VkImageCreateInfo, .{ .imageType = vk.VK_IMAGE_TYPE_2D, .format = vk.VK_FORMAT_R8G8B8A8_UNORM, .extent = .{ .width = W, .height = H, .depth = 1 }, .mipLevels = 1, .arrayLayers = 1, .usage = vk.VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | vk.VK_IMAGE_USAGE_TRANSFER_SRC_BIT });
    var image: vk.VkImage = vk.VK_NULL_HANDLE;
    try std.testing.expectEqual(vk.VkResult.VK_SUCCESS, createImage(device, &ici, null, &image));
    defer destroyImage(device, image, null);
    var ireqs: vk.VkMemoryRequirements = undefined;
    getImageMemoryRequirements(device, image, &ireqs);
    const imai = std.mem.zeroInit(vk.VkMemoryAllocateInfo, .{ .allocationSize = ireqs.size });
    var imem: vk.VkDeviceMemory = vk.VK_NULL_HANDLE;
    _ = allocateMemory(device, &imai, null, &imem);
    defer freeMemory(device, imem, null);
    _ = bindImageMemory(device, image, imem, 0);
    const ivci = std.mem.zeroInit(vk.VkImageViewCreateInfo, .{ .image = image, .viewType = vk.VK_IMAGE_VIEW_TYPE_2D, .format = vk.VK_FORMAT_R8G8B8A8_UNORM });
    var view: vk.VkImageView = vk.VK_NULL_HANDLE;
    try std.testing.expectEqual(vk.VkResult.VK_SUCCESS, createImageView(device, &ivci, null, &view));
    defer destroyImageView(device, view, null);

    const att = vk.VkAttachmentDescription{ .flags = 0, .format = vk.VK_FORMAT_R8G8B8A8_UNORM, .samples = 1, .loadOp = vk.VK_ATTACHMENT_LOAD_OP_CLEAR, .storeOp = vk.VK_ATTACHMENT_STORE_OP_STORE, .stencilLoadOp = vk.VK_ATTACHMENT_LOAD_OP_CLEAR, .stencilStoreOp = vk.VK_ATTACHMENT_STORE_OP_STORE, .initialLayout = vk.VK_IMAGE_LAYOUT_UNDEFINED, .finalLayout = vk.VK_IMAGE_LAYOUT_UNDEFINED };
    const ref = vk.VkAttachmentReference{ .attachment = 0, .layout = vk.VK_IMAGE_LAYOUT_UNDEFINED };
    const sub = std.mem.zeroInit(vk.VkSubpassDescription, .{ .colorAttachmentCount = 1, .pColorAttachments = @as([*]const vk.VkAttachmentReference, @ptrCast(&ref)) });
    const rpci = std.mem.zeroInit(vk.VkRenderPassCreateInfo, .{ .attachmentCount = 1, .pAttachments = @as([*]const vk.VkAttachmentDescription, @ptrCast(&att)), .subpassCount = 1, .pSubpasses = @as([*]const vk.VkSubpassDescription, @ptrCast(&sub)) });
    var rp: vk.VkRenderPass = vk.VK_NULL_HANDLE;
    try std.testing.expectEqual(vk.VkResult.VK_SUCCESS, createRenderPass(device, &rpci, null, &rp));
    defer destroyRenderPass(device, rp, null);
    const fbci = std.mem.zeroInit(vk.VkFramebufferCreateInfo, .{ .renderPass = rp, .attachmentCount = 1, .pAttachments = @as([*]const vk.VkImageView, @ptrCast(&view)), .width = W, .height = H, .layers = 1 });
    var fb: vk.VkFramebuffer = vk.VK_NULL_HANDLE;
    try std.testing.expectEqual(vk.VkResult.VK_SUCCESS, createFramebuffer(device, &fbci, null, &fb));
    defer destroyFramebuffer(device, fb, null);

    // Fullscreen red triangle (covers the whole screen).
    const Vtx = extern struct { x: f32, y: f32, r: f32, g: f32, b: f32 };
    const verts = [3]Vtx{ .{ .x = -1, .y = -1, .r = 1, .g = 0, .b = 0 }, .{ .x = 3, .y = -1, .r = 1, .g = 0, .b = 0 }, .{ .x = -1, .y = 3, .r = 1, .g = 0, .b = 0 } };
    var vbuf: vk.VkBuffer = vk.VK_NULL_HANDLE;
    const vbci = std.mem.zeroInit(vk.VkBufferCreateInfo, .{ .size = @sizeOf(@TypeOf(verts)), .usage = vk.VK_BUFFER_USAGE_VERTEX_BUFFER_BIT });
    try std.testing.expectEqual(vk.VkResult.VK_SUCCESS, createBuffer(device, &vbci, null, &vbuf));
    defer destroyBuffer(device, vbuf, null);
    var vreqs: vk.VkMemoryRequirements = undefined;
    getBufferMemoryRequirements(device, vbuf, &vreqs);
    const vmai = std.mem.zeroInit(vk.VkMemoryAllocateInfo, .{ .allocationSize = vreqs.size });
    var vmem: vk.VkDeviceMemory = vk.VK_NULL_HANDLE;
    _ = allocateMemory(device, &vmai, null, &vmem);
    defer freeMemory(device, vmem, null);
    _ = bindBufferMemory(device, vbuf, vmem, 0);
    var vp: ?*anyopaque = null;
    _ = mapMemory(device, vmem, 0, vk.VK_WHOLE_SIZE, 0, &vp);
    @memcpy(@as([*]u8, @ptrCast(vp.?))[0..@sizeOf(@TypeOf(verts))], std.mem.asBytes(&verts));

    const sh = makePassthroughShaders(device) orelse return error.SkipZigTest;
    defer destroyShaderModule(device, sh.vs, null);
    defer destroyShaderModule(device, sh.fs, null);
    const plci = std.mem.zeroInit(vk.VkPipelineLayoutCreateInfo, .{});
    var pl: vk.VkPipelineLayout = vk.VK_NULL_HANDLE;
    _ = createPipelineLayout(device, &plci, null, &pl);
    defer destroyPipelineLayout(device, pl, null);
    const stages = [_]vk.VkPipelineShaderStageCreateInfo{
        std.mem.zeroInit(vk.VkPipelineShaderStageCreateInfo, .{ .stage = vk.VK_SHADER_STAGE_VERTEX_BIT, .module = sh.vs, .pName = "main" }),
        std.mem.zeroInit(vk.VkPipelineShaderStageCreateInfo, .{ .stage = vk.VK_SHADER_STAGE_FRAGMENT_BIT, .module = sh.fs, .pName = "main" }),
    };
    const bind = vk.VkVertexInputBindingDescription{ .binding = 0, .stride = @sizeOf(Vtx), .inputRate = 0 };
    const vattrs = [_]vk.VkVertexInputAttributeDescription{ .{ .location = 0, .binding = 0, .format = vk.VK_FORMAT_R32G32_SFLOAT, .offset = 0 }, .{ .location = 1, .binding = 0, .format = vk.VK_FORMAT_R32G32B32_SFLOAT, .offset = 8 } };
    const vis = std.mem.zeroInit(vk.VkPipelineVertexInputStateCreateInfo, .{ .vertexBindingDescriptionCount = 1, .pVertexBindingDescriptions = @as([*]const vk.VkVertexInputBindingDescription, @ptrCast(&bind)), .vertexAttributeDescriptionCount = 2, .pVertexAttributeDescriptions = &vattrs });
    // Blend: src = CONSTANT_ALPHA (12), dst = ONE_MINUS_CONSTANT_ALPHA (13), constant.a = 0.5, ADD.
    // -> result.rgb = red*0.5 + blueClear*0.5 = (0.5, 0, 0.5). Also exercises blendConstants.
    const blend_att = vk.VkPipelineColorBlendAttachmentState{ .blendEnable = 1, .srcColorBlendFactor = 12, .dstColorBlendFactor = 13, .colorBlendOp = 0, .srcAlphaBlendFactor = 12, .dstAlphaBlendFactor = 13, .alphaBlendOp = 0, .colorWriteMask = 0xf };
    const cbs = std.mem.zeroInit(vk.VkPipelineColorBlendStateCreateInfo, .{ .attachmentCount = 1, .pAttachments = @as([*]const vk.VkPipelineColorBlendAttachmentState, @ptrCast(&blend_att)), .blendConstants = .{ 0, 0, 0, 0.5 } });
    const gpci = std.mem.zeroInit(vk.VkGraphicsPipelineCreateInfo, .{ .stageCount = 2, .pStages = &stages, .pVertexInputState = &vis, .pColorBlendState = &cbs, .layout = pl, .renderPass = rp });
    var gpipe: vk.VkPipeline = vk.VK_NULL_HANDLE;
    try std.testing.expectEqual(vk.VkResult.VK_SUCCESS, createGraphicsPipelines(device, vk.VK_NULL_HANDLE, 1, @ptrCast(&gpci), null, @ptrCast(&gpipe)));
    defer destroyPipeline(device, gpipe, null);

    const rb_size = W * H * 4;
    var rbuf: vk.VkBuffer = vk.VK_NULL_HANDLE;
    const rbci = std.mem.zeroInit(vk.VkBufferCreateInfo, .{ .size = rb_size, .usage = vk.VK_BUFFER_USAGE_TRANSFER_DST_BIT });
    try std.testing.expectEqual(vk.VkResult.VK_SUCCESS, createBuffer(device, &rbci, null, &rbuf));
    defer destroyBuffer(device, rbuf, null);
    var rreqs: vk.VkMemoryRequirements = undefined;
    getBufferMemoryRequirements(device, rbuf, &rreqs);
    const rmai = std.mem.zeroInit(vk.VkMemoryAllocateInfo, .{ .allocationSize = rreqs.size });
    var rmem: vk.VkDeviceMemory = vk.VK_NULL_HANDLE;
    _ = allocateMemory(device, &rmai, null, &rmem);
    defer freeMemory(device, rmem, null);
    _ = bindBufferMemory(device, rbuf, rmem, 0);

    const cpci2 = std.mem.zeroInit(vk.VkCommandPoolCreateInfo, .{});
    var cmd_pool: vk.VkCommandPool = vk.VK_NULL_HANDLE;
    _ = createCommandPool(device, &cpci2, null, &cmd_pool);
    defer destroyCommandPool(device, cmd_pool, null);
    const cbai = std.mem.zeroInit(vk.VkCommandBufferAllocateInfo, .{ .commandPool = cmd_pool, .level = vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1 });
    var cmd: vk.VkCommandBuffer = null;
    _ = allocateCommandBuffers(device, &cbai, @ptrCast(&cmd));
    const bi = std.mem.zeroInit(vk.VkCommandBufferBeginInfo, .{});
    _ = beginCommandBuffer(cmd, &bi);
    var clearv: vk.VkClearValue = .{ .color = .{ .float32 = .{ 0, 0, 1, 1 } } }; // blue clear = the blend dst
    const rpbi = std.mem.zeroInit(vk.VkRenderPassBeginInfo, .{ .renderPass = rp, .framebuffer = fb, .renderArea = .{ .offset = .{ .x = 0, .y = 0 }, .extent = .{ .width = W, .height = H } }, .clearValueCount = 1, .pClearValues = @as([*]const vk.VkClearValue, @ptrCast(&clearv)) });
    cmdBeginRenderPass(cmd, &rpbi, vk.VK_SUBPASS_CONTENTS_INLINE);
    cmdBindPipeline(cmd, vk.VK_PIPELINE_BIND_POINT_GRAPHICS, gpipe);
    const voff: vk.VkDeviceSize = 0;
    cmdBindVertexBuffers(cmd, 0, 1, @ptrCast(&vbuf), @ptrCast(&voff));
    cmdDraw(cmd, 3, 1, 0, 0);
    cmdEndRenderPass(cmd);
    const region = std.mem.zeroInit(vk.VkBufferImageCopy, .{ .imageSubresource = .{ .aspectMask = vk.VK_IMAGE_ASPECT_COLOR_BIT, .mipLevel = 0, .baseArrayLayer = 0, .layerCount = 1 }, .imageExtent = .{ .width = W, .height = H, .depth = 1 } });
    cmdCopyImageToBuffer(cmd, image, vk.VK_IMAGE_LAYOUT_UNDEFINED, rbuf, 1, @ptrCast(&region));
    _ = endCommandBuffer(cmd);
    const submit = std.mem.zeroInit(vk.VkSubmitInfo, .{ .commandBufferCount = 1, .pCommandBuffers = @as([*]const vk.VkCommandBuffer, @ptrCast(&cmd)) });
    try std.testing.expectEqual(vk.VkResult.VK_SUCCESS, queueSubmit(queue, 1, @ptrCast(&submit), vk.VK_NULL_HANDLE));
    _ = queueWaitIdle(queue);

    var rp_ptr: ?*anyopaque = null;
    _ = mapMemory(device, rmem, 0, vk.VK_WHOLE_SIZE, 0, &rp_ptr);
    const px: [*]u8 = @ptrCast(rp_ptr.?);
    const c = ((H / 2) * W + (W / 2)) * 4;
    // 50/50 red-over-blue -> ~ (128, 0, 128). Without blending it would be pure red (255,0,0).
    try std.testing.expect(px[c + 0] > 100 and px[c + 0] < 160); // R ~128
    try std.testing.expect(px[c + 1] < 40); // G ~0
    try std.testing.expect(px[c + 2] > 100 and px[c + 2] < 160); // B ~128 (the dst blue survived)
}

test "vkCmdCopyBuffer: copies a sub-range (srcOffset + dstOffset + size) between buffers" {
    const device = makeSoftwareDevice() orelse return error.SkipZigTest;
    defer destroyDevice(device, null);
    var queue: vk.VkQueue = null;
    getDeviceQueue(device, 0, 0, &queue);

    const N = 64;
    const mkbuf = struct {
        fn make(dev: vk.VkDevice, usage: vk.VkBufferUsageFlags) struct { buf: vk.VkBuffer, mem: vk.VkDeviceMemory, ptr: [*]u8 } {
            var buf: vk.VkBuffer = vk.VK_NULL_HANDLE;
            const bci = std.mem.zeroInit(vk.VkBufferCreateInfo, .{ .size = N, .usage = usage });
            _ = createBuffer(dev, &bci, null, &buf);
            var reqs: vk.VkMemoryRequirements = undefined;
            getBufferMemoryRequirements(dev, buf, &reqs);
            const mai = std.mem.zeroInit(vk.VkMemoryAllocateInfo, .{ .allocationSize = reqs.size });
            var mem: vk.VkDeviceMemory = vk.VK_NULL_HANDLE;
            _ = allocateMemory(dev, &mai, null, &mem);
            _ = bindBufferMemory(dev, buf, mem, 0);
            var p: ?*anyopaque = null;
            _ = mapMemory(dev, mem, 0, vk.VK_WHOLE_SIZE, 0, &p);
            return .{ .buf = buf, .mem = mem, .ptr = @ptrCast(p.?) };
        }
    }.make;
    const s = mkbuf(device, vk.VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
    defer destroyBuffer(device, s.buf, null);
    defer freeMemory(device, s.mem, null);
    const d = mkbuf(device, vk.VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    defer destroyBuffer(device, d.buf, null);
    defer freeMemory(device, d.mem, null);
    for (0..N) |i| {
        s.ptr[i] = @intCast(i);
        d.ptr[i] = 0;
    }

    const cpci = std.mem.zeroInit(vk.VkCommandPoolCreateInfo, .{});
    var pool: vk.VkCommandPool = vk.VK_NULL_HANDLE;
    _ = createCommandPool(device, &cpci, null, &pool);
    defer destroyCommandPool(device, pool, null);
    const cbai = std.mem.zeroInit(vk.VkCommandBufferAllocateInfo, .{ .commandPool = pool, .level = vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1 });
    var cmd: vk.VkCommandBuffer = null;
    _ = allocateCommandBuffers(device, &cbai, @ptrCast(&cmd));
    const bi = std.mem.zeroInit(vk.VkCommandBufferBeginInfo, .{});
    _ = beginCommandBuffer(cmd, &bi);
    // Copy src[8..40] -> dst[16..48] (offsets + size all non-trivial).
    const region = vk.VkBufferCopy{ .srcOffset = 8, .dstOffset = 16, .size = 32 };
    cmdCopyBuffer(cmd, s.buf, d.buf, 1, @ptrCast(&region));
    _ = endCommandBuffer(cmd);
    const submit = std.mem.zeroInit(vk.VkSubmitInfo, .{ .commandBufferCount = 1, .pCommandBuffers = @as([*]const vk.VkCommandBuffer, @ptrCast(&cmd)) });
    try std.testing.expectEqual(vk.VkResult.VK_SUCCESS, queueSubmit(queue, 1, @ptrCast(&submit), vk.VK_NULL_HANDLE));
    _ = queueWaitIdle(queue);

    // dst[16..48] must equal src[8..40]. Everything outside stays 0.
    try std.testing.expectEqual(@as(u8, 0), d.ptr[15]); // just before the copied range
    try std.testing.expectEqual(@as(u8, 8), d.ptr[16]); // start = src[8]
    try std.testing.expectEqual(@as(u8, 39), d.ptr[47]); // end = src[8+31]
    try std.testing.expectEqual(@as(u8, 0), d.ptr[48]); // just after
}

/// Test helper: create a logical device on the software physical device (the
/// rasterizer the graphics/depth path drives), or null if unavailable. The
/// instance is leaked deliberately (process-lifetime in the test runner).
fn makeSoftwareDevice() ?vk.VkDevice {
    var inst: vk.VkInstance = null;
    const ci = std.mem.zeroInit(vk.VkInstanceCreateInfo, .{ .sType = .VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO });
    if (createInstance(&ci, null, &inst) != .VK_SUCCESS) return null;
    var count: u32 = 0;
    _ = enumeratePhysicalDevices(inst, &count, null);
    if (count == 0) return null;
    var handles: [16]vk.VkPhysicalDevice = undefined;
    var got: u32 = @min(count, 16);
    _ = enumeratePhysicalDevices(inst, &got, &handles[0]);
    var phys: vk.VkPhysicalDevice = null;
    for (handles[0..got]) |h| {
        var props: vk.VkPhysicalDeviceProperties = undefined;
        getPhysicalDeviceProperties(h, &props);
        if (std.mem.indexOf(u8, &props.deviceName, "software") != null) {
            phys = h;
            break;
        }
    }
    if (phys == null) return null;
    const dci = std.mem.zeroes(vk.VkDeviceCreateInfo);
    var device: vk.VkDevice = null;
    if (createDevice(phys, &dci, null, &device) != .VK_SUCCESS) return null;
    return device;
}

/// Test helper: build a passthrough VS (in vec2 pos[loc0] + vec3 color[loc1] ->
/// gl_Position=(pos,0,1), out color[loc0]) + FS (in color -> vec4(color,1)) as ICD
/// VkShaderModules. Caller destroys both. Returns null on a builder OOM.
fn makePassthroughShaders(device: vk.VkDevice) ?struct { vs: vk.VkShaderModule, fs: vk.VkShaderModule } {
    const a = std.testing.allocator;
    const op = prism.spirv.opcodes;
    const sc = op.StorageClass;
    var vsb = prism.spirv.binary.Builder.init(a, 24) catch return null;
    defer vsb.deinit(a);
    vsb.emit(a, op.EntryPoint, &.{ op.ExecutionModel.vertex, 15, 0, 11, 12, 13, 14 }) catch return null;
    vsb.emit(a, op.Decorate, &.{ 13, op.Decoration.builtin, op.BuiltIn.position }) catch return null;
    vsb.emit(a, op.Decorate, &.{ 11, op.Decoration.location, 0 }) catch return null;
    vsb.emit(a, op.Decorate, &.{ 12, op.Decoration.location, 1 }) catch return null;
    vsb.emit(a, op.Decorate, &.{ 14, op.Decoration.location, 0 }) catch return null;
    vsb.emit(a, op.TypeVoid, &.{1}) catch return null;
    vsb.emit(a, op.TypeFunction, &.{ 2, 1 }) catch return null;
    vsb.emit(a, op.TypeFloat, &.{ 3, 32 }) catch return null;
    vsb.emit(a, op.TypeVector, &.{ 4, 3, 2 }) catch return null;
    vsb.emit(a, op.TypeVector, &.{ 5, 3, 3 }) catch return null;
    vsb.emit(a, op.TypeVector, &.{ 6, 3, 4 }) catch return null;
    vsb.emit(a, op.TypePointer, &.{ 7, sc.input, 4 }) catch return null;
    vsb.emit(a, op.TypePointer, &.{ 8, sc.input, 5 }) catch return null;
    vsb.emit(a, op.TypePointer, &.{ 9, sc.output, 6 }) catch return null;
    vsb.emit(a, op.TypePointer, &.{ 10, sc.output, 5 }) catch return null;
    vsb.emit(a, op.Constant, &.{ 3, 17, @bitCast(@as(f32, 0.0)) }) catch return null;
    vsb.emit(a, op.Constant, &.{ 3, 18, @bitCast(@as(f32, 1.0)) }) catch return null;
    vsb.emit(a, op.Variable, &.{ 7, 11, sc.input }) catch return null;
    vsb.emit(a, op.Variable, &.{ 8, 12, sc.input }) catch return null;
    vsb.emit(a, op.Variable, &.{ 9, 13, sc.output }) catch return null;
    vsb.emit(a, op.Variable, &.{ 10, 14, sc.output }) catch return null;
    vsb.emit(a, op.Function, &.{ 1, 15, 0, 2 }) catch return null;
    vsb.emit(a, op.Label, &.{16}) catch return null;
    vsb.emit(a, op.Load, &.{ 4, 19, 11 }) catch return null;
    vsb.emit(a, op.Load, &.{ 5, 20, 12 }) catch return null;
    vsb.emit(a, op.CompositeExtract, &.{ 3, 21, 19, 0 }) catch return null;
    vsb.emit(a, op.CompositeExtract, &.{ 3, 22, 19, 1 }) catch return null;
    vsb.emit(a, op.CompositeConstruct, &.{ 6, 23, 21, 22, 17, 18 }) catch return null;
    vsb.emit(a, op.Store, &.{ 13, 23 }) catch return null;
    vsb.emit(a, op.Store, &.{ 14, 20 }) catch return null;
    vsb.emit(a, op.Return, &.{}) catch return null;
    vsb.emit(a, op.FunctionEnd, &.{}) catch return null;

    var fsb = prism.spirv.binary.Builder.init(a, 18) catch return null;
    defer fsb.deinit(a);
    fsb.emit(a, op.EntryPoint, &.{ op.ExecutionModel.fragment, 10, 0, 8, 9 }) catch return null;
    fsb.emit(a, op.Decorate, &.{ 8, op.Decoration.location, 0 }) catch return null;
    fsb.emit(a, op.Decorate, &.{ 9, op.Decoration.location, 0 }) catch return null;
    fsb.emit(a, op.TypeVoid, &.{1}) catch return null;
    fsb.emit(a, op.TypeFunction, &.{ 2, 1 }) catch return null;
    fsb.emit(a, op.TypeFloat, &.{ 3, 32 }) catch return null;
    fsb.emit(a, op.TypeVector, &.{ 4, 3, 3 }) catch return null;
    fsb.emit(a, op.TypeVector, &.{ 5, 3, 4 }) catch return null;
    fsb.emit(a, op.TypePointer, &.{ 6, sc.input, 4 }) catch return null;
    fsb.emit(a, op.TypePointer, &.{ 7, sc.output, 5 }) catch return null;
    fsb.emit(a, op.Constant, &.{ 3, 12, @bitCast(@as(f32, 1.0)) }) catch return null;
    fsb.emit(a, op.Variable, &.{ 6, 8, sc.input }) catch return null;
    fsb.emit(a, op.Variable, &.{ 7, 9, sc.output }) catch return null;
    fsb.emit(a, op.Function, &.{ 1, 10, 0, 2 }) catch return null;
    fsb.emit(a, op.Label, &.{11}) catch return null;
    fsb.emit(a, op.Load, &.{ 4, 13, 8 }) catch return null;
    fsb.emit(a, op.CompositeExtract, &.{ 3, 14, 13, 0 }) catch return null;
    fsb.emit(a, op.CompositeExtract, &.{ 3, 15, 13, 1 }) catch return null;
    fsb.emit(a, op.CompositeExtract, &.{ 3, 16, 13, 2 }) catch return null;
    fsb.emit(a, op.CompositeConstruct, &.{ 5, 17, 14, 15, 16, 12 }) catch return null;
    fsb.emit(a, op.Store, &.{ 9, 17 }) catch return null;
    fsb.emit(a, op.Return, &.{}) catch return null;
    fsb.emit(a, op.FunctionEnd, &.{}) catch return null;

    const vs_smci = std.mem.zeroInit(vk.VkShaderModuleCreateInfo, .{ .codeSize = vsb.words.items.len * 4, .pCode = vsb.words.items.ptr });
    const fs_smci = std.mem.zeroInit(vk.VkShaderModuleCreateInfo, .{ .codeSize = fsb.words.items.len * 4, .pCode = fsb.words.items.ptr });
    var vs: vk.VkShaderModule = vk.VK_NULL_HANDLE;
    var fs: vk.VkShaderModule = vk.VK_NULL_HANDLE;
    if (createShaderModule(device, &vs_smci, null, &vs) != .VK_SUCCESS) return null;
    if (createShaderModule(device, &fs_smci, null, &fs) != .VK_SUCCESS) return null;
    return .{ .vs = vs, .fs = fs };
}

test "push constants: layout range parsed, pushed bytes stored, per-draw snapshot is draw-time state" {
    // The reported maxPushConstantsSize must be usable (>= the spec minimum 128).
    var props: vk.VkPhysicalDeviceProperties = undefined;
    const insts = enumerateAvailable() catch return error.SkipZigTest;
    if (insts.len == 0) return error.SkipZigTest;
    // Find the software physical device.
    var phys: ?*PhysicalDevice = null;
    for (insts) |*p| {
        if (!isHardwareDriver(prism.drivers.all[p.driver_index])) phys = p;
    }
    const pd = phys orelse return error.SkipZigTest;
    getPhysicalDeviceProperties(physToHandle(pd), &props);
    try std.testing.expect(props.limits.maxPushConstantsSize >= 128);
    try std.testing.expectEqual(@as(u32, PUSH_CONSTANT_SIZE), props.limits.maxPushConstantsSize);

    const device = makeSoftwareDevice() orelse return error.SkipZigTest;
    defer destroyDevice(device, null);

    // A pipeline layout with a push-constant range (offset 0, 16 bytes, fragment stage) and
    // no descriptor sets: push_constant_size must be the range high-water, binding_count 0.
    const range = vk.VkPushConstantRange{ .stageFlags = vk.VK_SHADER_STAGE_FRAGMENT_BIT, .offset = 0, .size = 16 };
    const plci = std.mem.zeroInit(vk.VkPipelineLayoutCreateInfo, .{
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = @as([*]const vk.VkPushConstantRange, @ptrCast(&range)),
    });
    var pl: vk.VkPipelineLayout = vk.VK_NULL_HANDLE;
    try std.testing.expectEqual(vk.VkResult.VK_SUCCESS, createPipelineLayout(device, &plci, null, &pl));
    defer destroyPipelineLayout(device, pl, null);
    const pl_obj = PipelineLayout.fromHandle(pl);
    try std.testing.expectEqual(@as(u32, 16), pl_obj.push_constant_size);
    try std.testing.expectEqual(@as(u32, 0), pl_obj.binding_count);

    // A command buffer: push a known vec4, record a draw (which must snapshot it), then
    // push a different value and record a second draw. Each draw's snapshot must hold the
    // bytes current at its record time (a later push does not change the earlier draw).
    // This is the defining property of push-constant state. The draws need a graphics
    // pipeline bound (cmdDraw only records when one is); we set the cb fields directly to
    // exercise the snapshot path without building a full pipeline here.
    const cpci = std.mem.zeroInit(vk.VkCommandPoolCreateInfo, .{});
    var pool: vk.VkCommandPool = vk.VK_NULL_HANDLE;
    try std.testing.expectEqual(vk.VkResult.VK_SUCCESS, createCommandPool(device, &cpci, null, &pool));
    defer destroyCommandPool(device, pool, null);
    const cbai = std.mem.zeroInit(vk.VkCommandBufferAllocateInfo, .{ .commandPool = pool, .level = vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1 });
    var cmd: vk.VkCommandBuffer = null;
    try std.testing.expectEqual(vk.VkResult.VK_SUCCESS, allocateCommandBuffers(device, &cbai, @ptrCast(&cmd)));
    const cb = CommandBuffer.fromHandle(cmd);
    const bi = std.mem.zeroInit(vk.VkCommandBufferBeginInfo, .{});
    _ = beginCommandBuffer(cmd, &bi);

    // A graphics pipeline whose layout declares push constants, so the snapshot fires.
    var gp = GraphicsPipeline{ .hal_vs = undefined, .hal_fs = undefined, .hal_pipeline = undefined, .pc_binding = 0, .has_push_constants = true };
    cb.gfx_pipeline = &gp;

    const v0 = [4]f32{ 0.2, 0.4, 0.6, 1.0 };
    cmdPushConstants(cmd, pl, vk.VK_SHADER_STAGE_FRAGMENT_BIT, 0, @sizeOf(@TypeOf(v0)), &v0);
    try std.testing.expect(cb.pc_dirty);
    // The live block now holds v0.
    {
        const live = std.mem.bytesAsValue([4]f32, cb.push_constants[0..16]);
        try std.testing.expectEqual(v0, live.*);
    }
    cmdDraw(cmd, 3, 1, 0, 0); // draw 0 snapshots v0

    const v1 = [4]f32{ 0.8, 0.1, 0.1, 1.0 };
    cmdPushConstants(cmd, pl, vk.VK_SHADER_STAGE_FRAGMENT_BIT, 0, @sizeOf(@TypeOf(v1)), &v1);
    cmdDraw(cmd, 3, 1, 0, 0); // draw 1 snapshots v1

    try std.testing.expectEqual(@as(usize, 2), cb.draw_count);
    try std.testing.expect(cb.draws[0].has_push_constants and cb.draws[1].has_push_constants);
    // Draw 0 kept v0 even though a later push set v1 (draw-time state).
    const snap0 = std.mem.bytesAsValue([4]f32, cb.draws[0].push_constants[0..16]);
    const snap1 = std.mem.bytesAsValue([4]f32, cb.draws[1].push_constants[0..16]);
    try std.testing.expectEqual(v0, snap0.*);
    try std.testing.expectEqual(v1, snap1.*);
}

test "depth API: format props, render pass depth attachment, framebuffer + pipeline depth state" {
    // D32_SFLOAT must report as a valid depth/stencil attachment.
    var fp: vk.VkFormatProperties = undefined;
    getPhysicalDeviceFormatProperties(undefined, vk.VK_FORMAT_D32_SFLOAT, &fp);
    try std.testing.expect((fp.optimalTilingFeatures & vk.VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT) != 0);
    try std.testing.expect(vk.formatIsDepth(vk.VK_FORMAT_D32_SFLOAT));
    try std.testing.expect(vk.formatIsDepth(vk.VK_FORMAT_D24_UNORM_S8_UINT));
    try std.testing.expect(vk.formatIsDepth(vk.VK_FORMAT_D16_UNORM));
    try std.testing.expect(!vk.formatIsDepth(vk.VK_FORMAT_R8G8B8A8_UNORM));
    try std.testing.expectEqual(prism.hal.Format.depth32_float, halFormat(vk.VK_FORMAT_D32_SFLOAT));

    // A render pass with a color attachment (0) + a depth attachment (1) referenced
    // by the subpass's pDepthStencilAttachment must record has_depth + depth_index.
    const W = 16;
    const H = 16;
    const device = makeSoftwareDevice() orelse return error.SkipZigTest;
    defer destroyDevice(device, null);

    const atts = [_]vk.VkAttachmentDescription{
        .{ .flags = 0, .format = vk.VK_FORMAT_R8G8B8A8_UNORM, .samples = 1, .loadOp = 1, .storeOp = 0, .stencilLoadOp = 0, .stencilStoreOp = 0, .initialLayout = 0, .finalLayout = 0 },
        .{ .flags = 0, .format = vk.VK_FORMAT_D32_SFLOAT, .samples = 1, .loadOp = 1, .storeOp = 0, .stencilLoadOp = 0, .stencilStoreOp = 0, .initialLayout = 0, .finalLayout = 0 },
    };
    const color_ref = vk.VkAttachmentReference{ .attachment = 0, .layout = 0 };
    const depth_ref = vk.VkAttachmentReference{ .attachment = 1, .layout = 0 };
    const sub = std.mem.zeroInit(vk.VkSubpassDescription, .{
        .colorAttachmentCount = 1,
        .pColorAttachments = @as([*]const vk.VkAttachmentReference, @ptrCast(&color_ref)),
        .pDepthStencilAttachment = &depth_ref,
    });
    const rpci = std.mem.zeroInit(vk.VkRenderPassCreateInfo, .{
        .attachmentCount = 2,
        .pAttachments = @as([*]const vk.VkAttachmentDescription, @ptrCast(&atts)),
        .subpassCount = 1,
        .pSubpasses = @as([*]const vk.VkSubpassDescription, @ptrCast(&sub)),
    });
    var rp: vk.VkRenderPass = vk.VK_NULL_HANDLE;
    try std.testing.expectEqual(vk.VkResult.VK_SUCCESS, createRenderPass(device, &rpci, null, &rp));
    defer destroyRenderPass(device, rp, null);
    const rp_obj = RenderPass.fromHandle(rp);
    try std.testing.expect(rp_obj.has_depth);
    try std.testing.expectEqual(@as(u32, 1), rp_obj.depth_index);

    // A depth VkImage + view: bindImageMemory allocates a depth32_float HAL resource.
    const ici = std.mem.zeroInit(vk.VkImageCreateInfo, .{
        .format = vk.VK_FORMAT_D32_SFLOAT,
        .extent = vk.VkExtent3D{ .width = W, .height = H, .depth = 1 },
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = 1,
    });
    var dimg: vk.VkImage = vk.VK_NULL_HANDLE;
    try std.testing.expectEqual(vk.VkResult.VK_SUCCESS, createImage(device, &ici, null, &dimg));
    defer destroyImage(device, dimg, null);
    var dreqs: vk.VkMemoryRequirements = undefined;
    getImageMemoryRequirements(device, dimg, &dreqs);
    try std.testing.expect(dreqs.size >= W * H * 4);
    const dmai = std.mem.zeroInit(vk.VkMemoryAllocateInfo, .{ .allocationSize = dreqs.size });
    var dmem: vk.VkDeviceMemory = vk.VK_NULL_HANDLE;
    try std.testing.expectEqual(vk.VkResult.VK_SUCCESS, allocateMemory(device, &dmai, null, &dmem));
    defer freeMemory(device, dmem, null);
    try std.testing.expectEqual(vk.VkResult.VK_SUCCESS, bindImageMemory(device, dimg, dmem, 0));
    try std.testing.expect(Image.fromHandle(dimg).resource != null);
    try std.testing.expectEqual(prism.hal.Format.depth32_float, Image.fromHandle(dimg).format);

    const dvci = std.mem.zeroInit(vk.VkImageViewCreateInfo, .{
        .image = dimg,
        .format = vk.VK_FORMAT_D32_SFLOAT,
        .subresourceRange = .{ .aspectMask = vk.VK_IMAGE_ASPECT_DEPTH_BIT, .baseMipLevel = 0, .levelCount = 1, .baseArrayLayer = 0, .layerCount = 1 },
    });
    var dview: vk.VkImageView = vk.VK_NULL_HANDLE;
    try std.testing.expectEqual(vk.VkResult.VK_SUCCESS, createImageView(device, &dvci, null, &dview));
    defer destroyImageView(device, dview, null);
}

test "stencil through the ICD: a REPLACE mask clips a later EQUAL draw (Vulkan UI clip path)" {
    const device = makeSoftwareDevice() orelse return error.SkipZigTest;
    defer destroyDevice(device, null);
    var queue: vk.VkQueue = null;
    getDeviceQueue(device, 0, 0, &queue);
    const W = 64;
    const H = 64;

    // Color image (RGBA8) + view.
    const cici = std.mem.zeroInit(vk.VkImageCreateInfo, .{
        .imageType = vk.VK_IMAGE_TYPE_2D,
        .format = vk.VK_FORMAT_R8G8B8A8_UNORM,
        .extent = .{ .width = W, .height = H, .depth = 1 },
        .mipLevels = 1,
        .arrayLayers = 1,
        .usage = vk.VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | vk.VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
    });
    var cimg: vk.VkImage = vk.VK_NULL_HANDLE;
    try std.testing.expectEqual(vk.VkResult.VK_SUCCESS, createImage(device, &cici, null, &cimg));
    defer destroyImage(device, cimg, null);
    var creqs: vk.VkMemoryRequirements = undefined;
    getImageMemoryRequirements(device, cimg, &creqs);
    const cmai = std.mem.zeroInit(vk.VkMemoryAllocateInfo, .{ .allocationSize = creqs.size });
    var cmem: vk.VkDeviceMemory = vk.VK_NULL_HANDLE;
    _ = allocateMemory(device, &cmai, null, &cmem);
    defer freeMemory(device, cmem, null);
    _ = bindImageMemory(device, cimg, cmem, 0);
    const cvci = std.mem.zeroInit(vk.VkImageViewCreateInfo, .{ .image = cimg, .viewType = vk.VK_IMAGE_VIEW_TYPE_2D, .format = vk.VK_FORMAT_R8G8B8A8_UNORM });
    var cview: vk.VkImageView = vk.VK_NULL_HANDLE;
    try std.testing.expectEqual(vk.VkResult.VK_SUCCESS, createImageView(device, &cvci, null, &cview));
    defer destroyImageView(device, cview, null);

    // Depth/stencil image (D24S8) + view: the stencil component lives in the HAL u8 buffer.
    const dici = std.mem.zeroInit(vk.VkImageCreateInfo, .{
        .imageType = vk.VK_IMAGE_TYPE_2D,
        .format = vk.VK_FORMAT_D24_UNORM_S8_UINT,
        .extent = .{ .width = W, .height = H, .depth = 1 },
        .mipLevels = 1,
        .arrayLayers = 1,
    });
    var dimg: vk.VkImage = vk.VK_NULL_HANDLE;
    try std.testing.expectEqual(vk.VkResult.VK_SUCCESS, createImage(device, &dici, null, &dimg));
    defer destroyImage(device, dimg, null);
    var dreqs: vk.VkMemoryRequirements = undefined;
    getImageMemoryRequirements(device, dimg, &dreqs);
    const dmai = std.mem.zeroInit(vk.VkMemoryAllocateInfo, .{ .allocationSize = dreqs.size });
    var dmem: vk.VkDeviceMemory = vk.VK_NULL_HANDLE;
    _ = allocateMemory(device, &dmai, null, &dmem);
    defer freeMemory(device, dmem, null);
    _ = bindImageMemory(device, dimg, dmem, 0);
    const dvci = std.mem.zeroInit(vk.VkImageViewCreateInfo, .{ .image = dimg, .viewType = vk.VK_IMAGE_VIEW_TYPE_2D, .format = vk.VK_FORMAT_D24_UNORM_S8_UINT, .subresourceRange = .{ .aspectMask = vk.VK_IMAGE_ASPECT_DEPTH_BIT | vk.VK_IMAGE_ASPECT_STENCIL_BIT, .baseMipLevel = 0, .levelCount = 1, .baseArrayLayer = 0, .layerCount = 1 } });
    var dview: vk.VkImageView = vk.VK_NULL_HANDLE;
    try std.testing.expectEqual(vk.VkResult.VK_SUCCESS, createImageView(device, &dvci, null, &dview));
    defer destroyImageView(device, dview, null);

    // Render pass: color (0) + depth/stencil (1, stencilLoadOp CLEAR) + framebuffer.
    const atts = [_]vk.VkAttachmentDescription{
        .{ .flags = 0, .format = vk.VK_FORMAT_R8G8B8A8_UNORM, .samples = 1, .loadOp = vk.VK_ATTACHMENT_LOAD_OP_CLEAR, .storeOp = vk.VK_ATTACHMENT_STORE_OP_STORE, .stencilLoadOp = 0, .stencilStoreOp = 0, .initialLayout = 0, .finalLayout = 0 },
        .{ .flags = 0, .format = vk.VK_FORMAT_D24_UNORM_S8_UINT, .samples = 1, .loadOp = vk.VK_ATTACHMENT_LOAD_OP_CLEAR, .storeOp = vk.VK_ATTACHMENT_STORE_OP_STORE, .stencilLoadOp = vk.VK_ATTACHMENT_LOAD_OP_CLEAR, .stencilStoreOp = vk.VK_ATTACHMENT_STORE_OP_STORE, .initialLayout = 0, .finalLayout = 0 },
    };
    const color_ref = vk.VkAttachmentReference{ .attachment = 0, .layout = 0 };
    const depth_ref = vk.VkAttachmentReference{ .attachment = 1, .layout = 0 };
    const sub = std.mem.zeroInit(vk.VkSubpassDescription, .{ .colorAttachmentCount = 1, .pColorAttachments = @as([*]const vk.VkAttachmentReference, @ptrCast(&color_ref)), .pDepthStencilAttachment = &depth_ref });
    const rpci = std.mem.zeroInit(vk.VkRenderPassCreateInfo, .{ .attachmentCount = 2, .pAttachments = @as([*]const vk.VkAttachmentDescription, @ptrCast(&atts)), .subpassCount = 1, .pSubpasses = @as([*]const vk.VkSubpassDescription, @ptrCast(&sub)) });
    var rp: vk.VkRenderPass = vk.VK_NULL_HANDLE;
    try std.testing.expectEqual(vk.VkResult.VK_SUCCESS, createRenderPass(device, &rpci, null, &rp));
    defer destroyRenderPass(device, rp, null);
    const fb_views = [_]vk.VkImageView{ cview, dview };
    const fbci = std.mem.zeroInit(vk.VkFramebufferCreateInfo, .{ .renderPass = rp, .attachmentCount = 2, .pAttachments = &fb_views, .width = W, .height = H, .layers = 1 });
    var fb: vk.VkFramebuffer = vk.VK_NULL_HANDLE;
    try std.testing.expectEqual(vk.VkResult.VK_SUCCESS, createFramebuffer(device, &fbci, null, &fb));
    defer destroyFramebuffer(device, fb, null);

    // Vertex buffer: 6 mask verts (left half, black) then 6 content verts (full, red).
    const Vtx = extern struct { x: f32, y: f32, r: f32, g: f32, b: f32 };
    const quad = struct {
        fn make(x0: f32, x1: f32, r: f32, g: f32, b: f32) [6]Vtx {
            return .{
                .{ .x = x0, .y = -1, .r = r, .g = g, .b = b }, .{ .x = x1, .y = -1, .r = r, .g = g, .b = b }, .{ .x = x1, .y = 1, .r = r, .g = g, .b = b },
                .{ .x = x0, .y = -1, .r = r, .g = g, .b = b }, .{ .x = x1, .y = 1, .r = r, .g = g, .b = b },  .{ .x = x0, .y = 1, .r = r, .g = g, .b = b },
            };
        }
    };
    const verts: [12]Vtx = quad.make(-1, 0, 0, 0, 0) ++ quad.make(-1, 1, 1, 0, 0);
    var vbuf: vk.VkBuffer = vk.VK_NULL_HANDLE;
    const vbci = std.mem.zeroInit(vk.VkBufferCreateInfo, .{ .size = @sizeOf(@TypeOf(verts)), .usage = vk.VK_BUFFER_USAGE_VERTEX_BUFFER_BIT });
    try std.testing.expectEqual(vk.VkResult.VK_SUCCESS, createBuffer(device, &vbci, null, &vbuf));
    defer destroyBuffer(device, vbuf, null);
    var vreqs: vk.VkMemoryRequirements = undefined;
    getBufferMemoryRequirements(device, vbuf, &vreqs);
    const vmai = std.mem.zeroInit(vk.VkMemoryAllocateInfo, .{ .allocationSize = vreqs.size });
    var vmem: vk.VkDeviceMemory = vk.VK_NULL_HANDLE;
    _ = allocateMemory(device, &vmai, null, &vmem);
    defer freeMemory(device, vmem, null);
    _ = bindBufferMemory(device, vbuf, vmem, 0);
    var vp: ?*anyopaque = null;
    _ = mapMemory(device, vmem, 0, vk.VK_WHOLE_SIZE, 0, &vp);
    @memcpy(@as([*]u8, @ptrCast(vp.?))[0..@sizeOf(@TypeOf(verts))], std.mem.asBytes(&verts));

    const sh = makePassthroughShaders(device) orelse return error.SkipZigTest;
    defer destroyShaderModule(device, sh.vs, null);
    defer destroyShaderModule(device, sh.fs, null);
    const plci = std.mem.zeroInit(vk.VkPipelineLayoutCreateInfo, .{});
    var pl: vk.VkPipelineLayout = vk.VK_NULL_HANDLE;
    _ = createPipelineLayout(device, &plci, null, &pl);
    defer destroyPipelineLayout(device, pl, null);
    const stages = [_]vk.VkPipelineShaderStageCreateInfo{
        std.mem.zeroInit(vk.VkPipelineShaderStageCreateInfo, .{ .stage = vk.VK_SHADER_STAGE_VERTEX_BIT, .module = sh.vs, .pName = "main" }),
        std.mem.zeroInit(vk.VkPipelineShaderStageCreateInfo, .{ .stage = vk.VK_SHADER_STAGE_FRAGMENT_BIT, .module = sh.fs, .pName = "main" }),
    };
    const bind = vk.VkVertexInputBindingDescription{ .binding = 0, .stride = @sizeOf(Vtx), .inputRate = 0 };
    const vattrs = [_]vk.VkVertexInputAttributeDescription{
        .{ .location = 0, .binding = 0, .format = vk.VK_FORMAT_R32G32_SFLOAT, .offset = 0 },
        .{ .location = 1, .binding = 0, .format = vk.VK_FORMAT_R32G32B32_SFLOAT, .offset = 8 },
    };
    const vis = std.mem.zeroInit(vk.VkPipelineVertexInputStateCreateInfo, .{ .vertexBindingDescriptionCount = 1, .pVertexBindingDescriptions = @as([*]const vk.VkVertexInputBindingDescription, @ptrCast(&bind)), .vertexAttributeDescriptionCount = 2, .pVertexAttributeDescriptions = &vattrs });

    // Two pipelines sharing everything but the stencil state.
    const mkStencil = struct {
        fn ds(compare: vk.VkCompareOp, pass: vk.VkStencilOp) vk.VkPipelineDepthStencilStateCreateInfo {
            const face = vk.VkStencilOpState{ .failOp = vk.VK_STENCIL_OP_KEEP, .passOp = pass, .depthFailOp = vk.VK_STENCIL_OP_KEEP, .compareOp = compare, .compareMask = 0xFF, .writeMask = 0xFF, .reference = 1 };
            return std.mem.zeroInit(vk.VkPipelineDepthStencilStateCreateInfo, .{ .stencilTestEnable = vk.VK_TRUE, .front = face, .back = face });
        }
    };
    const mask_ds = mkStencil.ds(vk.VK_COMPARE_OP_ALWAYS, vk.VK_STENCIL_OP_REPLACE);
    const content_ds = mkStencil.ds(vk.VK_COMPARE_OP_EQUAL, vk.VK_STENCIL_OP_KEEP);
    const base_gpci = vk.VkGraphicsPipelineCreateInfo;
    const mask_gpci = std.mem.zeroInit(base_gpci, .{ .stageCount = 2, .pStages = &stages, .pVertexInputState = &vis, .pDepthStencilState = &mask_ds, .layout = pl, .renderPass = rp });
    const content_gpci = std.mem.zeroInit(base_gpci, .{ .stageCount = 2, .pStages = &stages, .pVertexInputState = &vis, .pDepthStencilState = &content_ds, .layout = pl, .renderPass = rp });
    var mask_pipe: vk.VkPipeline = vk.VK_NULL_HANDLE;
    var content_pipe: vk.VkPipeline = vk.VK_NULL_HANDLE;
    try std.testing.expectEqual(vk.VkResult.VK_SUCCESS, createGraphicsPipelines(device, vk.VK_NULL_HANDLE, 1, @ptrCast(&mask_gpci), null, @ptrCast(&mask_pipe)));
    defer destroyPipeline(device, mask_pipe, null);
    try std.testing.expectEqual(vk.VkResult.VK_SUCCESS, createGraphicsPipelines(device, vk.VK_NULL_HANDLE, 1, @ptrCast(&content_gpci), null, @ptrCast(&content_pipe)));
    defer destroyPipeline(device, content_pipe, null);
    // The mask pipeline's parsed stencil state must reflect the Vulkan struct (REPLACE/ALWAYS).
    {
        const gp = GraphicsPipeline.fromHandle(mask_pipe);
        try std.testing.expect(gp.stencil.test_enable);
        try std.testing.expectEqual(prism.hal.StencilOp.replace, gp.stencil.pass_op);
        try std.testing.expectEqual(prism.hal.CompareOp.always, gp.stencil.compare_op);
        try std.testing.expectEqual(@as(u8, 1), gp.stencil.reference);
    }

    // Readback buffer.
    const rb_size = W * H * 4;
    var rbuf: vk.VkBuffer = vk.VK_NULL_HANDLE;
    const rbci = std.mem.zeroInit(vk.VkBufferCreateInfo, .{ .size = rb_size, .usage = vk.VK_BUFFER_USAGE_TRANSFER_DST_BIT });
    _ = createBuffer(device, &rbci, null, &rbuf);
    defer destroyBuffer(device, rbuf, null);
    var rreqs: vk.VkMemoryRequirements = undefined;
    getBufferMemoryRequirements(device, rbuf, &rreqs);
    const rmai = std.mem.zeroInit(vk.VkMemoryAllocateInfo, .{ .allocationSize = rreqs.size });
    var rmem: vk.VkDeviceMemory = vk.VK_NULL_HANDLE;
    _ = allocateMemory(device, &rmai, null, &rmem);
    defer freeMemory(device, rmem, null);
    _ = bindBufferMemory(device, rbuf, rmem, 0);

    // Record: clear (color black + stencil 0); mask draw (left, REPLACE -> stencil 1);
    // content draw (full, red, EQUAL 1 -> only the left half passes).
    const cpci = std.mem.zeroInit(vk.VkCommandPoolCreateInfo, .{});
    var cmd_pool: vk.VkCommandPool = vk.VK_NULL_HANDLE;
    _ = createCommandPool(device, &cpci, null, &cmd_pool);
    defer destroyCommandPool(device, cmd_pool, null);
    const cbai = std.mem.zeroInit(vk.VkCommandBufferAllocateInfo, .{ .commandPool = cmd_pool, .level = vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1 });
    var cmd: vk.VkCommandBuffer = null;
    _ = allocateCommandBuffers(device, &cbai, @ptrCast(&cmd));
    const bi = std.mem.zeroInit(vk.VkCommandBufferBeginInfo, .{});
    _ = beginCommandBuffer(cmd, &bi);
    const clears = [_]vk.VkClearValue{
        .{ .color = .{ .float32 = .{ 0, 0, 0, 1 } } },
        .{ .depthStencil = .{ .depth = 1.0, .stencil = 0 } },
    };
    const rpbi = std.mem.zeroInit(vk.VkRenderPassBeginInfo, .{ .renderPass = rp, .framebuffer = fb, .renderArea = .{ .offset = .{ .x = 0, .y = 0 }, .extent = .{ .width = W, .height = H } }, .clearValueCount = 2, .pClearValues = &clears });
    cmdBeginRenderPass(cmd, &rpbi, vk.VK_SUBPASS_CONTENTS_INLINE);
    const voff: vk.VkDeviceSize = 0;
    cmdBindVertexBuffers(cmd, 0, 1, @ptrCast(&vbuf), @ptrCast(&voff));
    cmdBindPipeline(cmd, vk.VK_PIPELINE_BIND_POINT_GRAPHICS, mask_pipe);
    cmdDraw(cmd, 6, 1, 0, 0);
    cmdBindPipeline(cmd, vk.VK_PIPELINE_BIND_POINT_GRAPHICS, content_pipe);
    cmdDraw(cmd, 6, 1, 6, 0);
    cmdEndRenderPass(cmd);
    const region = std.mem.zeroInit(vk.VkBufferImageCopy, .{ .imageSubresource = .{ .aspectMask = vk.VK_IMAGE_ASPECT_COLOR_BIT, .mipLevel = 0, .baseArrayLayer = 0, .layerCount = 1 }, .imageExtent = .{ .width = W, .height = H, .depth = 1 } });
    cmdCopyImageToBuffer(cmd, cimg, vk.VK_IMAGE_LAYOUT_UNDEFINED, rbuf, 1, @ptrCast(&region));
    _ = endCommandBuffer(cmd);
    const submit = std.mem.zeroInit(vk.VkSubmitInfo, .{ .commandBufferCount = 1, .pCommandBuffers = @as([*]const vk.VkCommandBuffer, @ptrCast(&cmd)) });
    try std.testing.expectEqual(vk.VkResult.VK_SUCCESS, queueSubmit(queue, 1, @ptrCast(&submit), vk.VK_NULL_HANDLE));
    _ = queueWaitIdle(queue);

    var rp_ptr: ?*anyopaque = null;
    _ = mapMemory(device, rmem, 0, vk.VK_WHOLE_SIZE, 0, &rp_ptr);
    const px: [*]u8 = @ptrCast(rp_ptr.?);
    const left = (32 * W + 16) * 4; // inside the mask -> red content drew
    const right = (32 * W + 48) * 4; // outside the mask -> clipped, clear black
    try std.testing.expect(px[left] > 200 and px[left + 1] < 50 and px[left + 2] < 50);
    try std.testing.expect(px[right] < 50 and px[right + 1] < 50 and px[right + 2] < 50);
}

test "dynamic stencil through the ICD: vkCmdSetStencilReference/CompareMask/WriteMask override baked values" {
    const device = makeSoftwareDevice() orelse return error.SkipZigTest;
    defer destroyDevice(device, null);
    var queue: vk.VkQueue = null;
    getDeviceQueue(device, 0, 0, &queue);
    const W = 64;
    const H = 64;

    const cici = std.mem.zeroInit(vk.VkImageCreateInfo, .{
        .imageType = vk.VK_IMAGE_TYPE_2D,
        .format = vk.VK_FORMAT_R8G8B8A8_UNORM,
        .extent = .{ .width = W, .height = H, .depth = 1 },
        .mipLevels = 1,
        .arrayLayers = 1,
        .usage = vk.VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | vk.VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
    });
    var cimg: vk.VkImage = vk.VK_NULL_HANDLE;
    try std.testing.expectEqual(vk.VkResult.VK_SUCCESS, createImage(device, &cici, null, &cimg));
    defer destroyImage(device, cimg, null);
    var creqs: vk.VkMemoryRequirements = undefined;
    getImageMemoryRequirements(device, cimg, &creqs);
    const cmai = std.mem.zeroInit(vk.VkMemoryAllocateInfo, .{ .allocationSize = creqs.size });
    var cmem: vk.VkDeviceMemory = vk.VK_NULL_HANDLE;
    _ = allocateMemory(device, &cmai, null, &cmem);
    defer freeMemory(device, cmem, null);
    _ = bindImageMemory(device, cimg, cmem, 0);
    const cvci = std.mem.zeroInit(vk.VkImageViewCreateInfo, .{ .image = cimg, .viewType = vk.VK_IMAGE_VIEW_TYPE_2D, .format = vk.VK_FORMAT_R8G8B8A8_UNORM });
    var cview: vk.VkImageView = vk.VK_NULL_HANDLE;
    try std.testing.expectEqual(vk.VkResult.VK_SUCCESS, createImageView(device, &cvci, null, &cview));
    defer destroyImageView(device, cview, null);

    const dici = std.mem.zeroInit(vk.VkImageCreateInfo, .{
        .imageType = vk.VK_IMAGE_TYPE_2D,
        .format = vk.VK_FORMAT_D24_UNORM_S8_UINT,
        .extent = .{ .width = W, .height = H, .depth = 1 },
        .mipLevels = 1,
        .arrayLayers = 1,
    });
    var dimg: vk.VkImage = vk.VK_NULL_HANDLE;
    try std.testing.expectEqual(vk.VkResult.VK_SUCCESS, createImage(device, &dici, null, &dimg));
    defer destroyImage(device, dimg, null);
    var dreqs: vk.VkMemoryRequirements = undefined;
    getImageMemoryRequirements(device, dimg, &dreqs);
    const dmai = std.mem.zeroInit(vk.VkMemoryAllocateInfo, .{ .allocationSize = dreqs.size });
    var dmem: vk.VkDeviceMemory = vk.VK_NULL_HANDLE;
    _ = allocateMemory(device, &dmai, null, &dmem);
    defer freeMemory(device, dmem, null);
    _ = bindImageMemory(device, dimg, dmem, 0);
    const dvci = std.mem.zeroInit(vk.VkImageViewCreateInfo, .{ .image = dimg, .viewType = vk.VK_IMAGE_VIEW_TYPE_2D, .format = vk.VK_FORMAT_D24_UNORM_S8_UINT, .subresourceRange = .{ .aspectMask = vk.VK_IMAGE_ASPECT_DEPTH_BIT | vk.VK_IMAGE_ASPECT_STENCIL_BIT, .baseMipLevel = 0, .levelCount = 1, .baseArrayLayer = 0, .layerCount = 1 } });
    var dview: vk.VkImageView = vk.VK_NULL_HANDLE;
    try std.testing.expectEqual(vk.VkResult.VK_SUCCESS, createImageView(device, &dvci, null, &dview));
    defer destroyImageView(device, dview, null);

    const atts = [_]vk.VkAttachmentDescription{
        .{ .flags = 0, .format = vk.VK_FORMAT_R8G8B8A8_UNORM, .samples = 1, .loadOp = vk.VK_ATTACHMENT_LOAD_OP_CLEAR, .storeOp = vk.VK_ATTACHMENT_STORE_OP_STORE, .stencilLoadOp = 0, .stencilStoreOp = 0, .initialLayout = 0, .finalLayout = 0 },
        .{ .flags = 0, .format = vk.VK_FORMAT_D24_UNORM_S8_UINT, .samples = 1, .loadOp = vk.VK_ATTACHMENT_LOAD_OP_CLEAR, .storeOp = vk.VK_ATTACHMENT_STORE_OP_STORE, .stencilLoadOp = vk.VK_ATTACHMENT_LOAD_OP_CLEAR, .stencilStoreOp = vk.VK_ATTACHMENT_STORE_OP_STORE, .initialLayout = 0, .finalLayout = 0 },
    };
    const color_ref = vk.VkAttachmentReference{ .attachment = 0, .layout = 0 };
    const depth_ref = vk.VkAttachmentReference{ .attachment = 1, .layout = 0 };
    const sub = std.mem.zeroInit(vk.VkSubpassDescription, .{ .colorAttachmentCount = 1, .pColorAttachments = @as([*]const vk.VkAttachmentReference, @ptrCast(&color_ref)), .pDepthStencilAttachment = &depth_ref });
    const rpci = std.mem.zeroInit(vk.VkRenderPassCreateInfo, .{ .attachmentCount = 2, .pAttachments = @as([*]const vk.VkAttachmentDescription, @ptrCast(&atts)), .subpassCount = 1, .pSubpasses = @as([*]const vk.VkSubpassDescription, @ptrCast(&sub)) });
    var rp: vk.VkRenderPass = vk.VK_NULL_HANDLE;
    try std.testing.expectEqual(vk.VkResult.VK_SUCCESS, createRenderPass(device, &rpci, null, &rp));
    defer destroyRenderPass(device, rp, null);
    const fb_views = [_]vk.VkImageView{ cview, dview };
    const fbci = std.mem.zeroInit(vk.VkFramebufferCreateInfo, .{ .renderPass = rp, .attachmentCount = 2, .pAttachments = &fb_views, .width = W, .height = H, .layers = 1 });
    var fb: vk.VkFramebuffer = vk.VK_NULL_HANDLE;
    try std.testing.expectEqual(vk.VkResult.VK_SUCCESS, createFramebuffer(device, &fbci, null, &fb));
    defer destroyFramebuffer(device, fb, null);

    const Vtx = extern struct { x: f32, y: f32, r: f32, g: f32, b: f32 };
    const quad = struct {
        fn make(x0: f32, x1: f32, r: f32, g: f32, b: f32) [6]Vtx {
            return .{
                .{ .x = x0, .y = -1, .r = r, .g = g, .b = b }, .{ .x = x1, .y = -1, .r = r, .g = g, .b = b }, .{ .x = x1, .y = 1, .r = r, .g = g, .b = b },
                .{ .x = x0, .y = -1, .r = r, .g = g, .b = b }, .{ .x = x1, .y = 1, .r = r, .g = g, .b = b },  .{ .x = x0, .y = 1, .r = r, .g = g, .b = b },
            };
        }
    };
    const verts: [12]Vtx = quad.make(-1, 0, 0, 0, 0) ++ quad.make(-1, 1, 1, 0, 0);
    var vbuf: vk.VkBuffer = vk.VK_NULL_HANDLE;
    const vbci = std.mem.zeroInit(vk.VkBufferCreateInfo, .{ .size = @sizeOf(@TypeOf(verts)), .usage = vk.VK_BUFFER_USAGE_VERTEX_BUFFER_BIT });
    try std.testing.expectEqual(vk.VkResult.VK_SUCCESS, createBuffer(device, &vbci, null, &vbuf));
    defer destroyBuffer(device, vbuf, null);
    var vreqs: vk.VkMemoryRequirements = undefined;
    getBufferMemoryRequirements(device, vbuf, &vreqs);
    const vmai = std.mem.zeroInit(vk.VkMemoryAllocateInfo, .{ .allocationSize = vreqs.size });
    var vmem: vk.VkDeviceMemory = vk.VK_NULL_HANDLE;
    _ = allocateMemory(device, &vmai, null, &vmem);
    defer freeMemory(device, vmem, null);
    _ = bindBufferMemory(device, vbuf, vmem, 0);
    var vp: ?*anyopaque = null;
    _ = mapMemory(device, vmem, 0, vk.VK_WHOLE_SIZE, 0, &vp);
    @memcpy(@as([*]u8, @ptrCast(vp.?))[0..@sizeOf(@TypeOf(verts))], std.mem.asBytes(&verts));

    const sh = makePassthroughShaders(device) orelse return error.SkipZigTest;
    defer destroyShaderModule(device, sh.vs, null);
    defer destroyShaderModule(device, sh.fs, null);
    const plci = std.mem.zeroInit(vk.VkPipelineLayoutCreateInfo, .{});
    var pl: vk.VkPipelineLayout = vk.VK_NULL_HANDLE;
    _ = createPipelineLayout(device, &plci, null, &pl);
    defer destroyPipelineLayout(device, pl, null);
    const stages = [_]vk.VkPipelineShaderStageCreateInfo{
        std.mem.zeroInit(vk.VkPipelineShaderStageCreateInfo, .{ .stage = vk.VK_SHADER_STAGE_VERTEX_BIT, .module = sh.vs, .pName = "main" }),
        std.mem.zeroInit(vk.VkPipelineShaderStageCreateInfo, .{ .stage = vk.VK_SHADER_STAGE_FRAGMENT_BIT, .module = sh.fs, .pName = "main" }),
    };
    const bind = vk.VkVertexInputBindingDescription{ .binding = 0, .stride = @sizeOf(Vtx), .inputRate = 0 };
    const vattrs = [_]vk.VkVertexInputAttributeDescription{
        .{ .location = 0, .binding = 0, .format = vk.VK_FORMAT_R32G32_SFLOAT, .offset = 0 },
        .{ .location = 1, .binding = 0, .format = vk.VK_FORMAT_R32G32B32_SFLOAT, .offset = 8 },
    };
    const vis = std.mem.zeroInit(vk.VkPipelineVertexInputStateCreateInfo, .{ .vertexBindingDescriptionCount = 1, .pVertexBindingDescriptions = @as([*]const vk.VkVertexInputBindingDescription, @ptrCast(&bind)), .vertexAttributeDescriptionCount = 2, .pVertexAttributeDescriptions = &vattrs });

    // Both pipelines bake deliberately wrong stencil masks (ref=99, compareMask=0, writeMask=0)
    // and declare all three dynamic. If the dynamic override is honored, vkCmdSetStencil*(ref=1,
    // compareMask=0xff, writeMask=0xff) makes the clip correct (left red, right black). If any
    // dynamic member were dropped, the wrong baked value would break the result:
    //   ref 99   -> mask writes 99 / content compares vs 99 -> not (left red, right black)
    //   cmp 0    -> content EQUAL passes everywhere -> right also red
    //   write 0  -> mask writes nothing -> content fails everywhere -> all black
    const dyn_states = [_]vk.VkDynamicState{ vk.VK_DYNAMIC_STATE_STENCIL_REFERENCE, vk.VK_DYNAMIC_STATE_STENCIL_COMPARE_MASK, vk.VK_DYNAMIC_STATE_STENCIL_WRITE_MASK };
    const dyn_ci = vk.VkPipelineDynamicStateCreateInfo{ .sType = @enumFromInt(0), .pNext = null, .flags = 0, .dynamicStateCount = dyn_states.len, .pDynamicStates = &dyn_states };
    const mkStencil = struct {
        fn ds(compare: vk.VkCompareOp, pass: vk.VkStencilOp) vk.VkPipelineDepthStencilStateCreateInfo {
            const face = vk.VkStencilOpState{ .failOp = vk.VK_STENCIL_OP_KEEP, .passOp = pass, .depthFailOp = vk.VK_STENCIL_OP_KEEP, .compareOp = compare, .compareMask = 0, .writeMask = 0, .reference = 99 };
            return std.mem.zeroInit(vk.VkPipelineDepthStencilStateCreateInfo, .{ .stencilTestEnable = vk.VK_TRUE, .front = face, .back = face });
        }
    };
    const mask_ds = mkStencil.ds(vk.VK_COMPARE_OP_ALWAYS, vk.VK_STENCIL_OP_REPLACE);
    const content_ds = mkStencil.ds(vk.VK_COMPARE_OP_EQUAL, vk.VK_STENCIL_OP_KEEP);
    const base_gpci = vk.VkGraphicsPipelineCreateInfo;
    const mask_gpci = std.mem.zeroInit(base_gpci, .{ .stageCount = 2, .pStages = &stages, .pVertexInputState = &vis, .pDepthStencilState = &mask_ds, .pDynamicState = &dyn_ci, .layout = pl, .renderPass = rp });
    const content_gpci = std.mem.zeroInit(base_gpci, .{ .stageCount = 2, .pStages = &stages, .pVertexInputState = &vis, .pDepthStencilState = &content_ds, .pDynamicState = &dyn_ci, .layout = pl, .renderPass = rp });
    var mask_pipe: vk.VkPipeline = vk.VK_NULL_HANDLE;
    var content_pipe: vk.VkPipeline = vk.VK_NULL_HANDLE;
    try std.testing.expectEqual(vk.VkResult.VK_SUCCESS, createGraphicsPipelines(device, vk.VK_NULL_HANDLE, 1, @ptrCast(&mask_gpci), null, @ptrCast(&mask_pipe)));
    defer destroyPipeline(device, mask_pipe, null);
    try std.testing.expectEqual(vk.VkResult.VK_SUCCESS, createGraphicsPipelines(device, vk.VK_NULL_HANDLE, 1, @ptrCast(&content_gpci), null, @ptrCast(&content_pipe)));
    defer destroyPipeline(device, content_pipe, null);
    // The dynamic-state flags must be parsed.
    {
        const gp = GraphicsPipeline.fromHandle(mask_pipe);
        try std.testing.expect(gp.dyn_stencil_ref and gp.dyn_stencil_compare and gp.dyn_stencil_write);
    }

    const rb_size = W * H * 4;
    var rbuf: vk.VkBuffer = vk.VK_NULL_HANDLE;
    const rbci = std.mem.zeroInit(vk.VkBufferCreateInfo, .{ .size = rb_size, .usage = vk.VK_BUFFER_USAGE_TRANSFER_DST_BIT });
    _ = createBuffer(device, &rbci, null, &rbuf);
    defer destroyBuffer(device, rbuf, null);
    var rreqs: vk.VkMemoryRequirements = undefined;
    getBufferMemoryRequirements(device, rbuf, &rreqs);
    const rmai = std.mem.zeroInit(vk.VkMemoryAllocateInfo, .{ .allocationSize = rreqs.size });
    var rmem: vk.VkDeviceMemory = vk.VK_NULL_HANDLE;
    _ = allocateMemory(device, &rmai, null, &rmem);
    defer freeMemory(device, rmem, null);
    _ = bindBufferMemory(device, rbuf, rmem, 0);

    const cpci = std.mem.zeroInit(vk.VkCommandPoolCreateInfo, .{});
    var cmd_pool: vk.VkCommandPool = vk.VK_NULL_HANDLE;
    _ = createCommandPool(device, &cpci, null, &cmd_pool);
    defer destroyCommandPool(device, cmd_pool, null);
    const cbai = std.mem.zeroInit(vk.VkCommandBufferAllocateInfo, .{ .commandPool = cmd_pool, .level = vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1 });
    var cmd: vk.VkCommandBuffer = null;
    _ = allocateCommandBuffers(device, &cbai, @ptrCast(&cmd));
    const bi = std.mem.zeroInit(vk.VkCommandBufferBeginInfo, .{});
    _ = beginCommandBuffer(cmd, &bi);
    const clears = [_]vk.VkClearValue{
        .{ .color = .{ .float32 = .{ 0, 0, 0, 1 } } },
        .{ .depthStencil = .{ .depth = 1.0, .stencil = 0 } },
    };
    const rpbi = std.mem.zeroInit(vk.VkRenderPassBeginInfo, .{ .renderPass = rp, .framebuffer = fb, .renderArea = .{ .offset = .{ .x = 0, .y = 0 }, .extent = .{ .width = W, .height = H } }, .clearValueCount = 2, .pClearValues = &clears });
    cmdBeginRenderPass(cmd, &rpbi, vk.VK_SUBPASS_CONTENTS_INLINE);
    // Supply the correct stencil masks dynamically (overriding the wrong baked values).
    cmdSetStencilReference(cmd, vk.VK_STENCIL_FACE_FRONT_AND_BACK, 1);
    cmdSetStencilCompareMask(cmd, vk.VK_STENCIL_FACE_FRONT_AND_BACK, 0xFF);
    cmdSetStencilWriteMask(cmd, vk.VK_STENCIL_FACE_FRONT_AND_BACK, 0xFF);
    const voff: vk.VkDeviceSize = 0;
    cmdBindVertexBuffers(cmd, 0, 1, @ptrCast(&vbuf), @ptrCast(&voff));
    cmdBindPipeline(cmd, vk.VK_PIPELINE_BIND_POINT_GRAPHICS, mask_pipe);
    cmdDraw(cmd, 6, 1, 0, 0);
    cmdBindPipeline(cmd, vk.VK_PIPELINE_BIND_POINT_GRAPHICS, content_pipe);
    cmdDraw(cmd, 6, 1, 6, 0);
    cmdEndRenderPass(cmd);
    const region = std.mem.zeroInit(vk.VkBufferImageCopy, .{ .imageSubresource = .{ .aspectMask = vk.VK_IMAGE_ASPECT_COLOR_BIT, .mipLevel = 0, .baseArrayLayer = 0, .layerCount = 1 }, .imageExtent = .{ .width = W, .height = H, .depth = 1 } });
    cmdCopyImageToBuffer(cmd, cimg, vk.VK_IMAGE_LAYOUT_UNDEFINED, rbuf, 1, @ptrCast(&region));
    _ = endCommandBuffer(cmd);
    const submit = std.mem.zeroInit(vk.VkSubmitInfo, .{ .commandBufferCount = 1, .pCommandBuffers = @as([*]const vk.VkCommandBuffer, @ptrCast(&cmd)) });
    try std.testing.expectEqual(vk.VkResult.VK_SUCCESS, queueSubmit(queue, 1, @ptrCast(&submit), vk.VK_NULL_HANDLE));
    _ = queueWaitIdle(queue);

    var rp_ptr: ?*anyopaque = null;
    _ = mapMemory(device, rmem, 0, vk.VK_WHOLE_SIZE, 0, &rp_ptr);
    const px: [*]u8 = @ptrCast(rp_ptr.?);
    const left = (32 * W + 16) * 4;
    const right = (32 * W + 48) * 4;
    try std.testing.expect(px[left] > 200 and px[left + 1] < 50 and px[left + 2] < 50); // dynamic ref/masks -> red
    try std.testing.expect(px[right] < 50 and px[right + 1] < 50 and px[right + 2] < 50); // clipped -> black
}

test "MSAA + formats API: sample counts, float format props, MSAA image sizing + render-pass resolve" {
    // The device must advertise 1x/2x/4x for color + depth framebuffer attachments.
    var inst: vk.VkInstance = null;
    const ici0 = std.mem.zeroInit(vk.VkInstanceCreateInfo, .{ .sType = .VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO });
    try std.testing.expectEqual(vk.VkResult.VK_SUCCESS, createInstance(&ici0, null, &inst));
    defer destroyInstance(inst, null);
    var pcount: u32 = 0;
    _ = enumeratePhysicalDevices(inst, &pcount, null);
    if (pcount == 0) return error.SkipZigTest;
    var phandles: [16]vk.VkPhysicalDevice = undefined;
    var pgot: u32 = @min(pcount, 16);
    _ = enumeratePhysicalDevices(inst, &pgot, &phandles[0]);
    var props: vk.VkPhysicalDeviceProperties = undefined;
    getPhysicalDeviceProperties(phandles[0], &props);
    const want = vk.VK_SAMPLE_COUNT_1_BIT | vk.VK_SAMPLE_COUNT_2_BIT | vk.VK_SAMPLE_COUNT_4_BIT;
    try std.testing.expect((props.limits.framebufferColorSampleCounts & want) == want);
    try std.testing.expect((props.limits.framebufferDepthSampleCounts & want) == want);
    try std.testing.expectEqual(@as(u8, 4), sampleCount(vk.VK_SAMPLE_COUNT_4_BIT));
    try std.testing.expectEqual(@as(u8, 2), sampleCount(vk.VK_SAMPLE_COUNT_2_BIT));
    try std.testing.expectEqual(@as(u8, 1), sampleCount(vk.VK_SAMPLE_COUNT_1_BIT));

    // The new float + R8/RG8 formats map to HAL formats and report sampled/color features.
    try std.testing.expectEqual(prism.hal.Format.r16_float, halFormat(vk.VK_FORMAT_R16_SFLOAT));
    try std.testing.expectEqual(prism.hal.Format.r32_float, halFormat(vk.VK_FORMAT_R32_SFLOAT));
    try std.testing.expectEqual(prism.hal.Format.r16g16_float, halFormat(vk.VK_FORMAT_R16G16_SFLOAT));
    try std.testing.expectEqual(prism.hal.Format.rgba16_float, halFormat(vk.VK_FORMAT_R16G16B16A16_SFLOAT));
    try std.testing.expectEqual(prism.hal.Format.r8_unorm, halFormat(vk.VK_FORMAT_R8_UNORM));
    try std.testing.expectEqual(prism.hal.Format.r8g8_unorm, halFormat(vk.VK_FORMAT_R8G8_UNORM));
    inline for (.{ vk.VK_FORMAT_R16_SFLOAT, vk.VK_FORMAT_R32_SFLOAT, vk.VK_FORMAT_R16G16B16A16_SFLOAT }) |f| {
        var fp: vk.VkFormatProperties = undefined;
        getPhysicalDeviceFormatProperties(undefined, f, &fp);
        try std.testing.expect((fp.optimalTilingFeatures & vk.VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT) != 0);
        try std.testing.expect((fp.optimalTilingFeatures & vk.VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT) != 0);
    }
    {
        var fp: vk.VkFormatProperties = undefined;
        getPhysicalDeviceFormatProperties(undefined, vk.VK_FORMAT_R8G8_UNORM, &fp);
        try std.testing.expect((fp.optimalTilingFeatures & vk.VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT) != 0);
    }

    const device = makeSoftwareDevice() orelse return error.SkipZigTest;
    defer destroyDevice(device, null);
    const W = 16;
    const H = 16;

    // A 4x-MSAA color image must size its backing to W*H*4*bpp (sample-minor).
    const ici = std.mem.zeroInit(vk.VkImageCreateInfo, .{
        .format = vk.VK_FORMAT_R8G8B8A8_UNORM,
        .extent = vk.VkExtent3D{ .width = W, .height = H, .depth = 1 },
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = vk.VK_SAMPLE_COUNT_4_BIT,
        .usage = vk.VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
    });
    var img: vk.VkImage = vk.VK_NULL_HANDLE;
    try std.testing.expectEqual(vk.VkResult.VK_SUCCESS, createImage(device, &ici, null, &img));
    defer destroyImage(device, img, null);
    try std.testing.expectEqual(@as(u8, 4), Image.fromHandle(img).samples);
    var reqs: vk.VkMemoryRequirements = undefined;
    getImageMemoryRequirements(device, img, &reqs);
    try std.testing.expect(reqs.size >= W * H * 4 * 4);

    // A render pass with a 4x color attachment (0) + a single-sample resolve attachment
    // (1) referenced by pResolveAttachments must record has_resolve + samples=4.
    const atts = [_]vk.VkAttachmentDescription{
        .{ .flags = 0, .format = vk.VK_FORMAT_R8G8B8A8_UNORM, .samples = vk.VK_SAMPLE_COUNT_4_BIT, .loadOp = 1, .storeOp = 0, .stencilLoadOp = 0, .stencilStoreOp = 0, .initialLayout = 0, .finalLayout = 0 },
        .{ .flags = 0, .format = vk.VK_FORMAT_R8G8B8A8_UNORM, .samples = vk.VK_SAMPLE_COUNT_1_BIT, .loadOp = 1, .storeOp = 0, .stencilLoadOp = 0, .stencilStoreOp = 0, .initialLayout = 0, .finalLayout = 0 },
    };
    const color_ref = vk.VkAttachmentReference{ .attachment = 0, .layout = 0 };
    const resolve_ref = vk.VkAttachmentReference{ .attachment = 1, .layout = 0 };
    const sub = std.mem.zeroInit(vk.VkSubpassDescription, .{
        .colorAttachmentCount = 1,
        .pColorAttachments = @as([*]const vk.VkAttachmentReference, @ptrCast(&color_ref)),
        .pResolveAttachments = @as([*]const vk.VkAttachmentReference, @ptrCast(&resolve_ref)),
    });
    const rpci = std.mem.zeroInit(vk.VkRenderPassCreateInfo, .{
        .attachmentCount = 2,
        .pAttachments = @as([*]const vk.VkAttachmentDescription, @ptrCast(&atts)),
        .subpassCount = 1,
        .pSubpasses = @as([*]const vk.VkSubpassDescription, @ptrCast(&sub)),
    });
    var rp: vk.VkRenderPass = vk.VK_NULL_HANDLE;
    try std.testing.expectEqual(vk.VkResult.VK_SUCCESS, createRenderPass(device, &rpci, null, &rp));
    defer destroyRenderPass(device, rp, null);
    const rp_obj = RenderPass.fromHandle(rp);
    try std.testing.expect(rp_obj.has_resolve);
    try std.testing.expectEqual(@as(u32, 1), rp_obj.resolve_index);
    try std.testing.expectEqual(@as(u8, 4), rp_obj.samples);
}

// Pull the libwayland WSI module's tests (the RGBA->XRGB blit) into the runner.
/// Build the passthrough VS (in pos vec2 loc0 + color vec3 loc1 -> gl_Position + out
/// vc loc0) into `b`. Caller owns + frees `b`.
fn buildPassthroughVs(b: *prism.spirv.binary.Builder) !void {
    const a = std.testing.allocator;
    const op = prism.spirv.opcodes;
    const sc = op.StorageClass;
    try b.emit(a, op.EntryPoint, &.{ op.ExecutionModel.vertex, 15, 0, 11, 12, 13, 14 });
    try b.emit(a, op.Decorate, &.{ 13, op.Decoration.builtin, op.BuiltIn.position });
    try b.emit(a, op.Decorate, &.{ 11, op.Decoration.location, 0 });
    try b.emit(a, op.Decorate, &.{ 12, op.Decoration.location, 1 });
    try b.emit(a, op.Decorate, &.{ 14, op.Decoration.location, 0 });
    try b.emit(a, op.TypeVoid, &.{1});
    try b.emit(a, op.TypeFunction, &.{ 2, 1 });
    try b.emit(a, op.TypeFloat, &.{ 3, 32 });
    try b.emit(a, op.TypeVector, &.{ 4, 3, 2 });
    try b.emit(a, op.TypeVector, &.{ 5, 3, 3 });
    try b.emit(a, op.TypeVector, &.{ 6, 3, 4 });
    try b.emit(a, op.TypePointer, &.{ 7, sc.input, 4 });
    try b.emit(a, op.TypePointer, &.{ 8, sc.input, 5 });
    try b.emit(a, op.TypePointer, &.{ 9, sc.output, 6 });
    try b.emit(a, op.TypePointer, &.{ 10, sc.output, 5 });
    try b.emit(a, op.Constant, &.{ 3, 17, @bitCast(@as(f32, 0.0)) });
    try b.emit(a, op.Constant, &.{ 3, 18, @bitCast(@as(f32, 1.0)) });
    try b.emit(a, op.Variable, &.{ 7, 11, sc.input });
    try b.emit(a, op.Variable, &.{ 8, 12, sc.input });
    try b.emit(a, op.Variable, &.{ 9, 13, sc.output });
    try b.emit(a, op.Variable, &.{ 10, 14, sc.output });
    try b.emit(a, op.Function, &.{ 1, 15, 0, 2 });
    try b.emit(a, op.Label, &.{16});
    try b.emit(a, op.Load, &.{ 4, 19, 11 });
    try b.emit(a, op.Load, &.{ 5, 20, 12 });
    try b.emit(a, op.CompositeExtract, &.{ 3, 21, 19, 0 });
    try b.emit(a, op.CompositeExtract, &.{ 3, 22, 19, 1 });
    try b.emit(a, op.CompositeConstruct, &.{ 6, 23, 21, 22, 17, 18 });
    try b.emit(a, op.Store, &.{ 13, 23 });
    try b.emit(a, op.Store, &.{ 14, 20 });
    try b.emit(a, op.Return, &.{});
    try b.emit(a, op.FunctionEnd, &.{});
}

/// Build the passthrough FS (in vc vec3 loc0 -> out o = vec4(vc, 1)) into `b`.
fn buildPassthroughFs(b: *prism.spirv.binary.Builder) !void {
    const a = std.testing.allocator;
    const op = prism.spirv.opcodes;
    const sc = op.StorageClass;
    try b.emit(a, op.EntryPoint, &.{ op.ExecutionModel.fragment, 10, 0, 8, 9 });
    try b.emit(a, op.Decorate, &.{ 8, op.Decoration.location, 0 });
    try b.emit(a, op.Decorate, &.{ 9, op.Decoration.location, 0 });
    try b.emit(a, op.TypeVoid, &.{1});
    try b.emit(a, op.TypeFunction, &.{ 2, 1 });
    try b.emit(a, op.TypeFloat, &.{ 3, 32 });
    try b.emit(a, op.TypeVector, &.{ 4, 3, 3 });
    try b.emit(a, op.TypeVector, &.{ 5, 3, 4 });
    try b.emit(a, op.TypePointer, &.{ 6, sc.input, 4 });
    try b.emit(a, op.TypePointer, &.{ 7, sc.output, 5 });
    try b.emit(a, op.Constant, &.{ 3, 12, @bitCast(@as(f32, 1.0)) });
    try b.emit(a, op.Variable, &.{ 6, 8, sc.input });
    try b.emit(a, op.Variable, &.{ 7, 9, sc.output });
    try b.emit(a, op.Function, &.{ 1, 10, 0, 2 });
    try b.emit(a, op.Label, &.{11});
    try b.emit(a, op.Load, &.{ 4, 13, 8 });
    try b.emit(a, op.CompositeExtract, &.{ 3, 14, 13, 0 });
    try b.emit(a, op.CompositeExtract, &.{ 3, 15, 13, 1 });
    try b.emit(a, op.CompositeExtract, &.{ 3, 16, 13, 2 });
    try b.emit(a, op.CompositeConstruct, &.{ 5, 17, 14, 15, 16, 12 });
    try b.emit(a, op.Store, &.{ 9, 17 });
    try b.emit(a, op.Return, &.{});
    try b.emit(a, op.FunctionEnd, &.{});
}

test "multiple render passes: two instances in one command buffer each render into their own framebuffer" {
    const device = makeSoftwareDevice() orelse return error.SkipZigTest;
    defer destroyDevice(device, null);
    var queue: vk.VkQueue = null;
    getDeviceQueue(device, 0, 0, &queue);

    const W = 64;
    const H = 64;

    // Helper: create an image (color attachment, optionally sampled) + view + a single-color
    // attachment render pass + framebuffer. Returns the pieces the recording needs.
    const Target = struct {
        image: vk.VkImage,
        view: vk.VkImageView,
        rp: vk.VkRenderPass,
        fb: vk.VkFramebuffer,
    };
    const makeTarget = struct {
        fn f(dev: vk.VkDevice, sampled: bool) Target {
            const extra: vk.VkImageUsageFlags = if (sampled) vk.VK_IMAGE_USAGE_SAMPLED_BIT else 0;
            const ici = std.mem.zeroInit(vk.VkImageCreateInfo, .{
                .imageType = vk.VK_IMAGE_TYPE_2D,
                .format = vk.VK_FORMAT_R8G8B8A8_UNORM,
                .extent = .{ .width = W, .height = H, .depth = 1 },
                .mipLevels = 1,
                .arrayLayers = 1,
                .usage = vk.VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | vk.VK_IMAGE_USAGE_TRANSFER_SRC_BIT | extra,
            });
            var image: vk.VkImage = vk.VK_NULL_HANDLE;
            _ = createImage(dev, &ici, null, &image);
            var ireqs: vk.VkMemoryRequirements = undefined;
            getImageMemoryRequirements(dev, image, &ireqs);
            const imai = std.mem.zeroInit(vk.VkMemoryAllocateInfo, .{ .allocationSize = ireqs.size });
            var imem: vk.VkDeviceMemory = vk.VK_NULL_HANDLE;
            _ = allocateMemory(dev, &imai, null, &imem);
            _ = bindImageMemory(dev, image, imem, 0);
            const ivci = std.mem.zeroInit(vk.VkImageViewCreateInfo, .{ .image = image, .viewType = vk.VK_IMAGE_VIEW_TYPE_2D, .format = vk.VK_FORMAT_R8G8B8A8_UNORM });
            var view: vk.VkImageView = vk.VK_NULL_HANDLE;
            _ = createImageView(dev, &ivci, null, &view);
            const att = vk.VkAttachmentDescription{
                .flags = 0,
                .format = vk.VK_FORMAT_R8G8B8A8_UNORM,
                .samples = 1,
                .loadOp = vk.VK_ATTACHMENT_LOAD_OP_CLEAR,
                .storeOp = vk.VK_ATTACHMENT_STORE_OP_STORE,
                .stencilLoadOp = vk.VK_ATTACHMENT_LOAD_OP_CLEAR,
                .stencilStoreOp = vk.VK_ATTACHMENT_STORE_OP_STORE,
                .initialLayout = vk.VK_IMAGE_LAYOUT_UNDEFINED,
                .finalLayout = vk.VK_IMAGE_LAYOUT_UNDEFINED,
            };
            const ref = vk.VkAttachmentReference{ .attachment = 0, .layout = vk.VK_IMAGE_LAYOUT_UNDEFINED };
            const sub = std.mem.zeroInit(vk.VkSubpassDescription, .{ .colorAttachmentCount = 1, .pColorAttachments = @as([*]const vk.VkAttachmentReference, @ptrCast(&ref)) });
            const rpci = std.mem.zeroInit(vk.VkRenderPassCreateInfo, .{ .attachmentCount = 1, .pAttachments = @as([*]const vk.VkAttachmentDescription, @ptrCast(&att)), .subpassCount = 1, .pSubpasses = @as([*]const vk.VkSubpassDescription, @ptrCast(&sub)) });
            var rp: vk.VkRenderPass = vk.VK_NULL_HANDLE;
            _ = createRenderPass(dev, &rpci, null, &rp);
            const fbci = std.mem.zeroInit(vk.VkFramebufferCreateInfo, .{ .renderPass = rp, .attachmentCount = 1, .pAttachments = @as([*]const vk.VkImageView, @ptrCast(&view)), .width = W, .height = H, .layers = 1 });
            var fb: vk.VkFramebuffer = vk.VK_NULL_HANDLE;
            _ = createFramebuffer(dev, &fbci, null, &fb);
            return .{ .image = image, .view = view, .rp = rp, .fb = fb };
        }
    }.f;

    // Pass-1 target is sampled (render-to-texture: usage is COLOR_ATTACHMENT|SAMPLED so the
    // one HAL Resource is both a render target here and a sampleable texture later). Pass-2
    // target is the final readback image.
    const t1 = makeTarget(device, true);
    const t2 = makeTarget(device, false);

    // The sampled image's HAL Resource must carry both render_target (rendered into) and
    // sampled (sampled later) usage, which is the render-to-texture aliasing the feature requires.
    {
        const img1 = Image.fromHandle(t1.image);
        try std.testing.expect(img1.resource != null);
        try std.testing.expect(img1.color_attachment and img1.sampled);
    }

    // A vertex-color triangle (the same geometry for both passes; the point is that each
    // instance renders into its own framebuffer, not that the last one wins).
    const Vtx = extern struct { x: f32, y: f32, r: f32, g: f32, b: f32 };
    const makeVB = struct {
        fn f(dev: vk.VkDevice, verts: []const Vtx) vk.VkBuffer {
            var vbuf: vk.VkBuffer = vk.VK_NULL_HANDLE;
            const vbci = std.mem.zeroInit(vk.VkBufferCreateInfo, .{ .size = verts.len * @sizeOf(Vtx), .usage = vk.VK_BUFFER_USAGE_VERTEX_BUFFER_BIT });
            _ = createBuffer(dev, &vbci, null, &vbuf);
            var vreqs: vk.VkMemoryRequirements = undefined;
            getBufferMemoryRequirements(dev, vbuf, &vreqs);
            const vmai = std.mem.zeroInit(vk.VkMemoryAllocateInfo, .{ .allocationSize = vreqs.size });
            var vmem: vk.VkDeviceMemory = vk.VK_NULL_HANDLE;
            _ = allocateMemory(dev, &vmai, null, &vmem);
            _ = bindBufferMemory(dev, vbuf, vmem, 0);
            var vp: ?*anyopaque = null;
            _ = mapMemory(dev, vmem, 0, vk.VK_WHOLE_SIZE, 0, &vp);
            @memcpy(@as([*]u8, @ptrCast(vp.?))[0 .. verts.len * @sizeOf(Vtx)], std.mem.sliceAsBytes(verts));
            return vbuf;
        }
    }.f;
    // Pass 1: a red-cornered triangle. Pass 2: a blue-cornered triangle. Distinct so a
    // collapse to one framebuffer (the old flat model) would be visible.
    const v1 = [3]Vtx{
        .{ .x = -0.9, .y = -0.9, .r = 1, .g = 0, .b = 0 },
        .{ .x = 0.9, .y = -0.9, .r = 1, .g = 0, .b = 0 },
        .{ .x = 0.0, .y = 0.9, .r = 1, .g = 0, .b = 0 },
    };
    const v2 = [3]Vtx{
        .{ .x = -0.9, .y = -0.9, .r = 0, .g = 0, .b = 1 },
        .{ .x = 0.9, .y = -0.9, .r = 0, .g = 0, .b = 1 },
        .{ .x = 0.0, .y = 0.9, .r = 0, .g = 0, .b = 1 },
    };
    const vb1 = makeVB(device, &v1);
    const vb2 = makeVB(device, &v2);

    // Shared passthrough VS/FS + pipeline (one suffices for both passes here).
    var vsb = try prism.spirv.binary.Builder.init(std.testing.allocator, 24);
    defer vsb.deinit(std.testing.allocator);
    try buildPassthroughVs(&vsb);
    var fsb = try prism.spirv.binary.Builder.init(std.testing.allocator, 18);
    defer fsb.deinit(std.testing.allocator);
    try buildPassthroughFs(&fsb);
    const vs_smci = std.mem.zeroInit(vk.VkShaderModuleCreateInfo, .{ .codeSize = vsb.words.items.len * 4, .pCode = vsb.words.items.ptr });
    const fs_smci = std.mem.zeroInit(vk.VkShaderModuleCreateInfo, .{ .codeSize = fsb.words.items.len * 4, .pCode = fsb.words.items.ptr });
    var vs: vk.VkShaderModule = vk.VK_NULL_HANDLE;
    var fs: vk.VkShaderModule = vk.VK_NULL_HANDLE;
    _ = createShaderModule(device, &vs_smci, null, &vs);
    _ = createShaderModule(device, &fs_smci, null, &fs);
    const plci = std.mem.zeroInit(vk.VkPipelineLayoutCreateInfo, .{});
    var pl: vk.VkPipelineLayout = vk.VK_NULL_HANDLE;
    _ = createPipelineLayout(device, &plci, null, &pl);
    const stages = [_]vk.VkPipelineShaderStageCreateInfo{
        std.mem.zeroInit(vk.VkPipelineShaderStageCreateInfo, .{ .stage = vk.VK_SHADER_STAGE_VERTEX_BIT, .module = vs, .pName = "main" }),
        std.mem.zeroInit(vk.VkPipelineShaderStageCreateInfo, .{ .stage = vk.VK_SHADER_STAGE_FRAGMENT_BIT, .module = fs, .pName = "main" }),
    };
    const bind = vk.VkVertexInputBindingDescription{ .binding = 0, .stride = @sizeOf(Vtx), .inputRate = 0 };
    const vattrs = [_]vk.VkVertexInputAttributeDescription{
        .{ .location = 0, .binding = 0, .format = vk.VK_FORMAT_R32G32_SFLOAT, .offset = 0 },
        .{ .location = 1, .binding = 0, .format = vk.VK_FORMAT_R32G32B32_SFLOAT, .offset = 8 },
    };
    const vis = std.mem.zeroInit(vk.VkPipelineVertexInputStateCreateInfo, .{
        .vertexBindingDescriptionCount = 1,
        .pVertexBindingDescriptions = @as([*]const vk.VkVertexInputBindingDescription, @ptrCast(&bind)),
        .vertexAttributeDescriptionCount = 2,
        .pVertexAttributeDescriptions = &vattrs,
    });
    const gpci = std.mem.zeroInit(vk.VkGraphicsPipelineCreateInfo, .{
        .stageCount = 2,
        .pStages = &stages,
        .pVertexInputState = &vis,
        .layout = pl,
        .renderPass = t1.rp,
    });
    var gpipe: vk.VkPipeline = vk.VK_NULL_HANDLE;
    _ = createGraphicsPipelines(device, vk.VK_NULL_HANDLE, 1, @ptrCast(&gpci), null, @ptrCast(&gpipe));

    // Readback buffers for both targets.
    const rb_size = W * H * 4;
    const makeRB = struct {
        fn f(dev: vk.VkDevice) struct { buf: vk.VkBuffer, mem: vk.VkDeviceMemory } {
            var rbuf: vk.VkBuffer = vk.VK_NULL_HANDLE;
            const rbci = std.mem.zeroInit(vk.VkBufferCreateInfo, .{ .size = rb_size, .usage = vk.VK_BUFFER_USAGE_TRANSFER_DST_BIT });
            _ = createBuffer(dev, &rbci, null, &rbuf);
            var rreqs: vk.VkMemoryRequirements = undefined;
            getBufferMemoryRequirements(dev, rbuf, &rreqs);
            const rmai = std.mem.zeroInit(vk.VkMemoryAllocateInfo, .{ .allocationSize = rreqs.size });
            var rmem: vk.VkDeviceMemory = vk.VK_NULL_HANDLE;
            _ = allocateMemory(dev, &rmai, null, &rmem);
            _ = bindBufferMemory(dev, rbuf, rmem, 0);
            return .{ .buf = rbuf, .mem = rmem };
        }
    }.f;
    const rb1 = makeRB(device);
    const rb2 = makeRB(device);

    // One command buffer, two render-pass instances:
    //   instance 1 -> t1.fb (the sampled offscreen image), clear green, draw the red triangle
    //   instance 2 -> t2.fb (the final image),             clear green, draw the blue triangle
    // then copy each target to its readback buffer.
    const cpci = std.mem.zeroInit(vk.VkCommandPoolCreateInfo, .{});
    var cmd_pool: vk.VkCommandPool = vk.VK_NULL_HANDLE;
    _ = createCommandPool(device, &cpci, null, &cmd_pool);
    const cbai = std.mem.zeroInit(vk.VkCommandBufferAllocateInfo, .{ .commandPool = cmd_pool, .level = vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1 });
    var cmd: vk.VkCommandBuffer = null;
    _ = allocateCommandBuffers(device, &cbai, @ptrCast(&cmd));
    const bi = std.mem.zeroInit(vk.VkCommandBufferBeginInfo, .{});
    _ = beginCommandBuffer(cmd, &bi);

    var clearg: vk.VkClearValue = .{ .color = .{ .float32 = .{ 0, 1, 0, 1 } } }; // green bg
    const voff: vk.VkDeviceSize = 0;

    // Instance 1.
    const rpbi1 = std.mem.zeroInit(vk.VkRenderPassBeginInfo, .{
        .renderPass = t1.rp,
        .framebuffer = t1.fb,
        .renderArea = .{ .offset = .{ .x = 0, .y = 0 }, .extent = .{ .width = W, .height = H } },
        .clearValueCount = 1,
        .pClearValues = @as([*]const vk.VkClearValue, @ptrCast(&clearg)),
    });
    cmdBeginRenderPass(cmd, &rpbi1, vk.VK_SUBPASS_CONTENTS_INLINE);
    cmdBindPipeline(cmd, vk.VK_PIPELINE_BIND_POINT_GRAPHICS, gpipe);
    cmdBindVertexBuffers(cmd, 0, 1, @ptrCast(&vb1), @ptrCast(&voff));
    cmdDraw(cmd, 3, 1, 0, 0);
    cmdEndRenderPass(cmd);

    // Instance 2.
    const rpbi2 = std.mem.zeroInit(vk.VkRenderPassBeginInfo, .{
        .renderPass = t2.rp,
        .framebuffer = t2.fb,
        .renderArea = .{ .offset = .{ .x = 0, .y = 0 }, .extent = .{ .width = W, .height = H } },
        .clearValueCount = 1,
        .pClearValues = @as([*]const vk.VkClearValue, @ptrCast(&clearg)),
    });
    cmdBeginRenderPass(cmd, &rpbi2, vk.VK_SUBPASS_CONTENTS_INLINE);
    cmdBindPipeline(cmd, vk.VK_PIPELINE_BIND_POINT_GRAPHICS, gpipe);
    cmdBindVertexBuffers(cmd, 0, 1, @ptrCast(&vb2), @ptrCast(&voff));
    cmdDraw(cmd, 3, 1, 0, 0);
    cmdEndRenderPass(cmd);

    // Two instances must have been recorded, each owning one draw.
    {
        const cbp = CommandBuffer.fromHandle(cmd);
        try std.testing.expectEqual(@as(usize, 2), cbp.pass_count);
        try std.testing.expectEqual(@as(usize, 1), cbp.passes[0].draw_count);
        try std.testing.expectEqual(@as(usize, 1), cbp.passes[1].draw_count);
    }

    const region = std.mem.zeroInit(vk.VkBufferImageCopy, .{
        .imageSubresource = .{ .aspectMask = vk.VK_IMAGE_ASPECT_COLOR_BIT, .mipLevel = 0, .baseArrayLayer = 0, .layerCount = 1 },
        .imageExtent = .{ .width = W, .height = H, .depth = 1 },
    });
    cmdCopyImageToBuffer(cmd, t2.image, vk.VK_IMAGE_LAYOUT_UNDEFINED, rb2.buf, 1, @ptrCast(&region));
    _ = endCommandBuffer(cmd);

    const submit = std.mem.zeroInit(vk.VkSubmitInfo, .{ .commandBufferCount = 1, .pCommandBuffers = @as([*]const vk.VkCommandBuffer, @ptrCast(&cmd)) });
    try std.testing.expectEqual(vk.VkResult.VK_SUCCESS, queueSubmit(queue, 1, @ptrCast(&submit), vk.VK_NULL_HANDLE));
    _ = queueWaitIdle(queue);
    _ = rb1;

    // The pass-1 sampled image's HAL Resource holds the red triangle rendered into it by
    // instance 1. Read its pixels directly (this is exactly what a sampler in a later pass
    // would read, since it is the same Resource).
    {
        const img1 = Image.fromHandle(t1.image);
        const res1 = img1.resource.?;
        const dev1 = LogicalDevice.fromHandle(device);
        const px1 = dev1.hal().mapResource(res1) catch unreachable;
        const center = ((H / 2) * W + (W / 2)) * 4;
        // Center is inside the triangle: red dominant (pass 1's content), not the green bg.
        try std.testing.expect(px1[center + 0] > 200); // R high
        try std.testing.expect(px1[center + 2] < 60); // B low (not blue)
    }

    // The pass-2 final image (read back) holds the blue triangle, proving instance 2
    // rendered into its own framebuffer (not collapsed with instance 1).
    {
        var rp_ptr: ?*anyopaque = null;
        _ = mapMemory(device, rb2.mem, 0, vk.VK_WHOLE_SIZE, 0, &rp_ptr);
        const px2: [*]u8 = @ptrCast(rp_ptr.?);
        const center = ((H / 2) * W + (W / 2)) * 4;
        try std.testing.expect(px2[center + 2] > 200); // B high (pass 2's blue triangle)
        try std.testing.expect(px2[center + 0] < 60); // R low (not pass 1's red)
    }
}

test {
    _ = wsi_wl;
}

test "swapchain recreate (resize) is crash-free: create -> recreate(oldSwapchain) -> destroy old, looped" {
    // Regression for the destroySwapchainKHR bad-`sc` segfault hit on a real
    // compositor resize. vkcube on resize creates a new swapchain with the old
    // one set as ci.oldSwapchain, then destroys the old one, looped. This drives
    // that exact lifecycle on the legacy prism (headless) path so it needs no
    // compositor, exercising the same Swapchain alloc/free + per-create HAL
    // surface/context the live libwayland path uses. With page_allocator any
    // use-after-free or double-free of the Swapchain (or its `images`) faults here.
    const device = makeSoftwareDevice() orelse return error.SkipZigTest;
    defer destroyDevice(device, null);

    var hl = platform.headless.create(allocator) catch return error.SkipZigTest;
    defer hl.deinit();
    var plat_surface = hl.createSurface(.{ .width = 64, .height = 48 }) catch return error.SkipZigTest;
    defer plat_surface.deinit();

    const wsci = vk.VkWaylandSurfaceCreateInfoKHR{
        .sType = .VK_STRUCTURE_TYPE_WAYLAND_SURFACE_CREATE_INFO_KHR,
        .pNext = null,
        .flags = 0,
        .display = null, // legacy prism path: `.surface` is a *platform.Surface
        .surface = @ptrCast(&plat_surface),
    };
    var surface: vk.VkSurfaceKHR = 0;
    try std.testing.expectEqual(vk.VkResult.VK_SUCCESS, createWaylandSurfaceKHR(null, &wsci, null, &surface));
    defer destroySurfaceKHR(null, surface, null);

    const baseScci = struct {
        fn make(s: vk.VkSurfaceKHR, w: u32, h: u32, old: vk.VkSwapchainKHR) vk.VkSwapchainCreateInfoKHR {
            return std.mem.zeroInit(vk.VkSwapchainCreateInfoKHR, .{
                .sType = .VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
                .surface = s,
                .minImageCount = 2,
                .imageFormat = vk.VK_FORMAT_B8G8R8A8_UNORM,
                .imageColorSpace = vk.VK_COLOR_SPACE_SRGB_NONLINEAR_KHR,
                .imageExtent = .{ .width = w, .height = h },
                .imageArrayLayers = 1,
                .imageUsage = vk.VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
                .presentMode = vk.VK_PRESENT_MODE_FIFO_KHR,
                .clipped = vk.VK_TRUE,
                .oldSwapchain = old,
            });
        }
    }.make;

    // Initial swapchain.
    var ci0 = baseScci(surface, 64, 48, vk.VK_NULL_HANDLE);
    var cur: vk.VkSwapchainKHR = 0;
    try std.testing.expectEqual(vk.VkResult.VK_SUCCESS, createSwapchainKHR(device, &ci0, null, &cur));

    // Several resize cycles: each creates a new swapchain referencing the old one,
    // then destroys the old (the exact order a WSI app uses on configure/resize).
    const sizes = [_][2]u32{ .{ 80, 60 }, .{ 100, 75 }, .{ 64, 48 }, .{ 120, 90 } };
    for (sizes) |sz| {
        var ci = baseScci(surface, sz[0], sz[1], cur);
        var next: vk.VkSwapchainKHR = 0;
        try std.testing.expectEqual(vk.VkResult.VK_SUCCESS, createSwapchainKHR(device, &ci, null, &next));
        // Touch the old swapchain's images before destroying it (acquire reads
        // sc.images), mirroring an app that may still hold the retired handle.
        var idx: u32 = 0;
        _ = acquireNextImageKHR(device, cur, std.math.maxInt(u64), 0, vk.VK_NULL_HANDLE, &idx);
        // Destroy the retired (old) swapchain. This is the crash site.
        destroySwapchainKHR(device, cur, null);
        cur = next;
    }
    destroySwapchainKHR(device, cur, null);

    // The actual crash signature: a double-destroy of the same handle (and a
    // destroy of a stale handle) must be a safe no-op, not a use-after-free that
    // dereferences the freed `sc` (the `for (sc.images)` segfault). Before the
    // live-swapchain registry these would read freed/unmapped memory and crash.
    {
        var ci = baseScci(surface, 64, 48, vk.VK_NULL_HANDLE);
        var sc: vk.VkSwapchainKHR = 0;
        try std.testing.expectEqual(vk.VkResult.VK_SUCCESS, createSwapchainKHR(device, &ci, null, &sc));
        destroySwapchainKHR(device, sc, null); // frees sc
        destroySwapchainKHR(device, sc, null); // double-destroy: no-op, must not crash
        destroySwapchainKHR(device, sc, null); // and again
        // acquire / present / getImages on the stale handle are safe no-ops too.
        var idx: u32 = 0;
        try std.testing.expectEqual(vk.VkResult.VK_ERROR_OUT_OF_DATE_KHR, acquireNextImageKHR(device, sc, 0, 0, vk.VK_NULL_HANDLE, &idx));
        var n: u32 = 0;
        try std.testing.expectEqual(vk.VkResult.VK_ERROR_OUT_OF_DATE_KHR, getSwapchainImagesKHR(device, sc, &n, null));
        const swaps = [_]vk.VkSwapchainKHR{sc};
        const inds = [_]u32{0};
        var results = [_]vk.VkResult{.VK_SUCCESS};
        const pi = std.mem.zeroInit(vk.VkPresentInfoKHR, .{
            .sType = .VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
            .swapchainCount = 1,
            .pSwapchains = &swaps,
            .pImageIndices = &inds,
            .pResults = &results,
        });
        _ = queuePresentKHR(undefined, &pi);
        try std.testing.expectEqual(vk.VkResult.VK_ERROR_OUT_OF_DATE_KHR, results[0]);
    }
}
