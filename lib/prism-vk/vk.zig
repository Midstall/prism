//! Vulkan API types with exact C ABI layout (vulkan-headers 1.4.341.0). Every
//! struct is `extern` and mirrors vulkan_core.h field order. Wrong layout in
//! VkPhysicalDeviceProperties / VkPhysicalDeviceLimits crashes vulkaninfo.

const std = @import("std");

// --- Core scalar aliases ---------------------------------------------------

pub const VkBool32 = u32;
pub const VkDeviceSize = u64;
pub const VkFlags = u32;
pub const VkSampleCountFlags = VkFlags;
pub const VkQueueFlags = VkFlags;
pub const VkMemoryPropertyFlags = VkFlags;
pub const VkMemoryHeapFlags = VkFlags;
pub const VkInstanceCreateFlags = VkFlags;

pub const VK_TRUE: VkBool32 = 1;
pub const VK_FALSE: VkBool32 = 0;

// --- Constants (must match vulkan_core.h) ----------------------------------

pub const VK_MAX_PHYSICAL_DEVICE_NAME_SIZE: usize = 256;
pub const VK_UUID_SIZE: usize = 16;
pub const VK_MAX_EXTENSION_NAME_SIZE: usize = 256;
pub const VK_MAX_DESCRIPTION_SIZE: usize = 256;
pub const VK_MAX_MEMORY_TYPES: usize = 32;
pub const VK_MAX_MEMORY_HEAPS: usize = 16;

// Queue flag bits.
pub const VK_QUEUE_GRAPHICS_BIT: VkQueueFlags = 0x00000001;
pub const VK_QUEUE_COMPUTE_BIT: VkQueueFlags = 0x00000002;
pub const VK_QUEUE_TRANSFER_BIT: VkQueueFlags = 0x00000004;

// Memory property + heap flag bits.
pub const VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT: VkMemoryPropertyFlags = 0x00000001;
pub const VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT: VkMemoryPropertyFlags = 0x00000002;
pub const VK_MEMORY_PROPERTY_HOST_COHERENT_BIT: VkMemoryPropertyFlags = 0x00000004;
pub const VK_MEMORY_HEAP_DEVICE_LOCAL_BIT: VkMemoryHeapFlags = 0x00000001;

// --- API version helpers ---------------------------------------------------

/// VK_MAKE_API_VERSION(variant, major, minor, patch).
pub fn VK_MAKE_API_VERSION(variant: u32, major: u32, minor: u32, patch: u32) u32 {
    return (variant << 29) | (major << 22) | (minor << 12) | patch;
}

pub const VK_API_VERSION_1_0: u32 = VK_MAKE_API_VERSION(0, 1, 0, 0);
pub const VK_API_VERSION_1_3: u32 = VK_MAKE_API_VERSION(0, 1, 3, 0);

// --- Result + enums --------------------------------------------------------

pub const VkResult = enum(i32) {
    VK_SUCCESS = 0,
    VK_NOT_READY = 1,
    VK_TIMEOUT = 2,
    VK_EVENT_SET = 3,
    VK_EVENT_RESET = 4,
    VK_INCOMPLETE = 5,
    VK_ERROR_OUT_OF_HOST_MEMORY = -1,
    VK_ERROR_OUT_OF_DEVICE_MEMORY = -2,
    VK_ERROR_INITIALIZATION_FAILED = -3,
    VK_ERROR_DEVICE_LOST = -4,
    VK_ERROR_MEMORY_MAP_FAILED = -5,
    VK_ERROR_LAYER_NOT_PRESENT = -6,
    VK_ERROR_EXTENSION_NOT_PRESENT = -7,
    VK_ERROR_FEATURE_NOT_PRESENT = -8,
    VK_ERROR_INCOMPATIBLE_DRIVER = -9,
    VK_ERROR_TOO_MANY_OBJECTS = -10,
    VK_ERROR_FORMAT_NOT_SUPPORTED = -11,
    // WSI (VK_KHR_swapchain): returned when a swapchain handle is retired/stale so
    // the app recreates it instead of the ICD dereferencing freed memory.
    VK_SUBOPTIMAL_KHR = 1000001003,
    VK_ERROR_OUT_OF_DATE_KHR = -1000001004,
    _,
};

pub const VkStructureType = enum(i32) {
    VK_STRUCTURE_TYPE_APPLICATION_INFO = 0,
    VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO = 1,
    VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO = 9,
    VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR = 1000001000,
    VK_STRUCTURE_TYPE_PRESENT_INFO_KHR = 1000001001,
    VK_STRUCTURE_TYPE_WAYLAND_SURFACE_CREATE_INFO_KHR = 1000006000,
    _,
};

pub const VkPhysicalDeviceType = enum(i32) {
    VK_PHYSICAL_DEVICE_TYPE_OTHER = 0,
    VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU = 1,
    VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU = 2,
    VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU = 3,
    VK_PHYSICAL_DEVICE_TYPE_CPU = 4,
    _,
};

// --- Dispatchable + function-pointer handle types --------------------------

pub const VkInstance = ?*opaque {};
pub const VkPhysicalDevice = ?*opaque {};

/// Treated as opaque/ignored. The loader passes a pointer we never dereference.
pub const VkAllocationCallbacks = anyopaque;

pub const PFN_vkVoidFunction = ?*const fn () callconv(.c) void;

// --- Instance creation structs ---------------------------------------------

pub const VkApplicationInfo = extern struct {
    sType: VkStructureType,
    pNext: ?*const anyopaque,
    pApplicationName: ?[*:0]const u8,
    applicationVersion: u32,
    pEngineName: ?[*:0]const u8,
    engineVersion: u32,
    apiVersion: u32,
};

pub const VkInstanceCreateInfo = extern struct {
    sType: VkStructureType,
    pNext: ?*const anyopaque,
    flags: VkInstanceCreateFlags,
    pApplicationInfo: ?*const VkApplicationInfo,
    enabledLayerCount: u32,
    ppEnabledLayerNames: ?[*]const [*:0]const u8,
    enabledExtensionCount: u32,
    ppEnabledExtensionNames: ?[*]const [*:0]const u8,
};

// --- Enumeration-property structs ------------------------------------------

pub const VkExtensionProperties = extern struct {
    extensionName: [VK_MAX_EXTENSION_NAME_SIZE]u8,
    specVersion: u32,
};

pub const VkLayerProperties = extern struct {
    layerName: [VK_MAX_EXTENSION_NAME_SIZE]u8,
    specVersion: u32,
    implementationVersion: u32,
    description: [VK_MAX_DESCRIPTION_SIZE]u8,
};

// --- Physical-device property structs --------------------------------------

/// The full ~100-field limits struct. Layout/size must be exact. The values may
/// be zero-filled for the enumeration milestone.
pub const VkPhysicalDeviceLimits = extern struct {
    maxImageDimension1D: u32,
    maxImageDimension2D: u32,
    maxImageDimension3D: u32,
    maxImageDimensionCube: u32,
    maxImageArrayLayers: u32,
    maxTexelBufferElements: u32,
    maxUniformBufferRange: u32,
    maxStorageBufferRange: u32,
    maxPushConstantsSize: u32,
    maxMemoryAllocationCount: u32,
    maxSamplerAllocationCount: u32,
    bufferImageGranularity: VkDeviceSize,
    sparseAddressSpaceSize: VkDeviceSize,
    maxBoundDescriptorSets: u32,
    maxPerStageDescriptorSamplers: u32,
    maxPerStageDescriptorUniformBuffers: u32,
    maxPerStageDescriptorStorageBuffers: u32,
    maxPerStageDescriptorSampledImages: u32,
    maxPerStageDescriptorStorageImages: u32,
    maxPerStageDescriptorInputAttachments: u32,
    maxPerStageResources: u32,
    maxDescriptorSetSamplers: u32,
    maxDescriptorSetUniformBuffers: u32,
    maxDescriptorSetUniformBuffersDynamic: u32,
    maxDescriptorSetStorageBuffers: u32,
    maxDescriptorSetStorageBuffersDynamic: u32,
    maxDescriptorSetSampledImages: u32,
    maxDescriptorSetStorageImages: u32,
    maxDescriptorSetInputAttachments: u32,
    maxVertexInputAttributes: u32,
    maxVertexInputBindings: u32,
    maxVertexInputAttributeOffset: u32,
    maxVertexInputBindingStride: u32,
    maxVertexOutputComponents: u32,
    maxTessellationGenerationLevel: u32,
    maxTessellationPatchSize: u32,
    maxTessellationControlPerVertexInputComponents: u32,
    maxTessellationControlPerVertexOutputComponents: u32,
    maxTessellationControlPerPatchOutputComponents: u32,
    maxTessellationControlTotalOutputComponents: u32,
    maxTessellationEvaluationInputComponents: u32,
    maxTessellationEvaluationOutputComponents: u32,
    maxGeometryShaderInvocations: u32,
    maxGeometryInputComponents: u32,
    maxGeometryOutputComponents: u32,
    maxGeometryOutputVertices: u32,
    maxGeometryTotalOutputComponents: u32,
    maxFragmentInputComponents: u32,
    maxFragmentOutputAttachments: u32,
    maxFragmentDualSrcAttachments: u32,
    maxFragmentCombinedOutputResources: u32,
    maxComputeSharedMemorySize: u32,
    maxComputeWorkGroupCount: [3]u32,
    maxComputeWorkGroupInvocations: u32,
    maxComputeWorkGroupSize: [3]u32,
    subPixelPrecisionBits: u32,
    subTexelPrecisionBits: u32,
    mipmapPrecisionBits: u32,
    maxDrawIndexedIndexValue: u32,
    maxDrawIndirectCount: u32,
    maxSamplerLodBias: f32,
    maxSamplerAnisotropy: f32,
    maxViewports: u32,
    maxViewportDimensions: [2]u32,
    viewportBoundsRange: [2]f32,
    viewportSubPixelBits: u32,
    minMemoryMapAlignment: usize,
    minTexelBufferOffsetAlignment: VkDeviceSize,
    minUniformBufferOffsetAlignment: VkDeviceSize,
    minStorageBufferOffsetAlignment: VkDeviceSize,
    minTexelOffset: i32,
    maxTexelOffset: u32,
    minTexelGatherOffset: i32,
    maxTexelGatherOffset: u32,
    minInterpolationOffset: f32,
    maxInterpolationOffset: f32,
    subPixelInterpolationOffsetBits: u32,
    maxFramebufferWidth: u32,
    maxFramebufferHeight: u32,
    maxFramebufferLayers: u32,
    framebufferColorSampleCounts: VkSampleCountFlags,
    framebufferDepthSampleCounts: VkSampleCountFlags,
    framebufferStencilSampleCounts: VkSampleCountFlags,
    framebufferNoAttachmentsSampleCounts: VkSampleCountFlags,
    maxColorAttachments: u32,
    sampledImageColorSampleCounts: VkSampleCountFlags,
    sampledImageIntegerSampleCounts: VkSampleCountFlags,
    sampledImageDepthSampleCounts: VkSampleCountFlags,
    sampledImageStencilSampleCounts: VkSampleCountFlags,
    storageImageSampleCounts: VkSampleCountFlags,
    maxSampleMaskWords: u32,
    timestampComputeAndGraphics: VkBool32,
    timestampPeriod: f32,
    maxClipDistances: u32,
    maxCullDistances: u32,
    maxCombinedClipAndCullDistances: u32,
    discreteQueuePriorities: u32,
    pointSizeRange: [2]f32,
    lineWidthRange: [2]f32,
    pointSizeGranularity: f32,
    lineWidthGranularity: f32,
    strictLines: VkBool32,
    standardSampleLocations: VkBool32,
    optimalBufferCopyOffsetAlignment: VkDeviceSize,
    optimalBufferCopyRowPitchAlignment: VkDeviceSize,
    nonCoherentAtomSize: VkDeviceSize,
};

pub const VkPhysicalDeviceSparseProperties = extern struct {
    residencyStandard2DBlockShape: VkBool32,
    residencyStandard2DMultisampleBlockShape: VkBool32,
    residencyStandard3DBlockShape: VkBool32,
    residencyAlignedMipSize: VkBool32,
    residencyNonResidentStrict: VkBool32,
};

pub const VkPhysicalDeviceProperties = extern struct {
    apiVersion: u32,
    driverVersion: u32,
    vendorID: u32,
    deviceID: u32,
    deviceType: VkPhysicalDeviceType,
    deviceName: [VK_MAX_PHYSICAL_DEVICE_NAME_SIZE]u8,
    pipelineCacheUUID: [VK_UUID_SIZE]u8,
    limits: VkPhysicalDeviceLimits,
    sparseProperties: VkPhysicalDeviceSparseProperties,
};

/// The ~55 VkBool32 feature flags, exact order from vulkan_core.h.
pub const VkPhysicalDeviceFeatures = extern struct {
    robustBufferAccess: VkBool32,
    fullDrawIndexUint32: VkBool32,
    imageCubeArray: VkBool32,
    independentBlend: VkBool32,
    geometryShader: VkBool32,
    tessellationShader: VkBool32,
    sampleRateShading: VkBool32,
    dualSrcBlend: VkBool32,
    logicOp: VkBool32,
    multiDrawIndirect: VkBool32,
    drawIndirectFirstInstance: VkBool32,
    depthClamp: VkBool32,
    depthBiasClamp: VkBool32,
    fillModeNonSolid: VkBool32,
    depthBounds: VkBool32,
    wideLines: VkBool32,
    largePoints: VkBool32,
    alphaToOne: VkBool32,
    multiViewport: VkBool32,
    samplerAnisotropy: VkBool32,
    textureCompressionETC2: VkBool32,
    textureCompressionASTC_LDR: VkBool32,
    textureCompressionBC: VkBool32,
    occlusionQueryPrecise: VkBool32,
    pipelineStatisticsQuery: VkBool32,
    vertexPipelineStoresAndAtomics: VkBool32,
    fragmentStoresAndAtomics: VkBool32,
    shaderTessellationAndGeometryPointSize: VkBool32,
    shaderImageGatherExtended: VkBool32,
    shaderStorageImageExtendedFormats: VkBool32,
    shaderStorageImageMultisample: VkBool32,
    shaderStorageImageReadWithoutFormat: VkBool32,
    shaderStorageImageWriteWithoutFormat: VkBool32,
    shaderUniformBufferArrayDynamicIndexing: VkBool32,
    shaderSampledImageArrayDynamicIndexing: VkBool32,
    shaderStorageBufferArrayDynamicIndexing: VkBool32,
    shaderStorageImageArrayDynamicIndexing: VkBool32,
    shaderClipDistance: VkBool32,
    shaderCullDistance: VkBool32,
    shaderFloat64: VkBool32,
    shaderInt64: VkBool32,
    shaderInt16: VkBool32,
    shaderResourceResidency: VkBool32,
    shaderResourceMinLod: VkBool32,
    sparseBinding: VkBool32,
    sparseResidencyBuffer: VkBool32,
    sparseResidencyImage2D: VkBool32,
    sparseResidencyImage3D: VkBool32,
    sparseResidency2Samples: VkBool32,
    sparseResidency4Samples: VkBool32,
    sparseResidency8Samples: VkBool32,
    sparseResidency16Samples: VkBool32,
    sparseResidencyAliased: VkBool32,
    variableMultisampleRate: VkBool32,
    inheritedQueries: VkBool32,
};

pub const VkExtent3D = extern struct {
    width: u32,
    height: u32,
    depth: u32,
};

pub const VkQueueFamilyProperties = extern struct {
    queueFlags: VkQueueFlags,
    queueCount: u32,
    timestampValidBits: u32,
    minImageTransferGranularity: VkExtent3D,
};

pub const VkMemoryType = extern struct {
    propertyFlags: VkMemoryPropertyFlags,
    heapIndex: u32,
};

pub const VkMemoryHeap = extern struct {
    size: VkDeviceSize,
    flags: VkMemoryHeapFlags,
};

pub const VkPhysicalDeviceMemoryProperties = extern struct {
    memoryTypeCount: u32,
    memoryTypes: [VK_MAX_MEMORY_TYPES]VkMemoryType,
    memoryHeapCount: u32,
    memoryHeaps: [VK_MAX_MEMORY_HEAPS]VkMemoryHeap,
};

// Format-query types. The loader requires these entrypoints. Enumeration reports
// "format not supported" for everything but the types still need to be defined.

/// VkFormat is a large enum. We only pass it through, so a plain i32 alias keeps
/// the ABI (enum is int-sized) without enumerating ~250 members.
pub const VkFormat = i32;
pub const VkImageType = i32;
pub const VkImageTiling = i32;
pub const VkImageUsageFlags = VkFlags;
pub const VkImageCreateFlags = VkFlags;
pub const VkSampleCountFlagBits = VkFlags;
pub const VkFormatFeatureFlags = VkFlags;

pub const VkFormatProperties = extern struct {
    linearTilingFeatures: VkFormatFeatureFlags,
    optimalTilingFeatures: VkFormatFeatureFlags,
    bufferFeatures: VkFormatFeatureFlags,
};

pub const VkImageFormatProperties = extern struct {
    maxExtent: VkExtent3D,
    maxMipLevels: u32,
    maxArrayLayers: u32,
    sampleCounts: VkSampleCountFlags,
    maxResourceSize: VkDeviceSize,
};

pub const VkSparseImageFormatProperties = extern struct {
    aspectMask: VkFlags,
    imageGranularity: VkExtent3D,
    flags: VkFlags,
};

// --- Device-level handles --------------------------------------------------
// VkDevice + VkQueue are dispatchable: the loader stamps the loader magic into
// the first uintptr_t, so the backing structs (in icd.zig) are extern with
// loader_magic first.
pub const VkDevice = ?*opaque {};
pub const VkQueue = ?*opaque {};

// VkDeviceMemory + VkBuffer are non-dispatchable: 64-bit handles (on LP64 the
// loader passes them through untouched). We encode @intFromPtr of a heap struct
// directly. No loader magic.
pub const VkDeviceMemory = u64;
pub const VkBuffer = u64;
pub const VK_NULL_HANDLE: u64 = 0;

// --- Device creation structs -----------------------------------------------

pub const VkDeviceCreateFlags = VkFlags;
pub const VkDeviceQueueCreateFlags = VkFlags;

pub const VkDeviceQueueCreateInfo = extern struct {
    sType: VkStructureType,
    pNext: ?*const anyopaque,
    flags: VkDeviceQueueCreateFlags,
    queueFamilyIndex: u32,
    queueCount: u32,
    pQueuePriorities: ?[*]const f32,
};

pub const VkDeviceCreateInfo = extern struct {
    sType: VkStructureType,
    pNext: ?*const anyopaque,
    flags: VkDeviceCreateFlags,
    queueCreateInfoCount: u32,
    pQueueCreateInfos: ?[*]const VkDeviceQueueCreateInfo,
    enabledLayerCount: u32,
    ppEnabledLayerNames: ?[*]const [*:0]const u8,
    enabledExtensionCount: u32,
    ppEnabledExtensionNames: ?[*]const [*:0]const u8,
    pEnabledFeatures: ?*const VkPhysicalDeviceFeatures,
};

// --- Memory + buffer structs -----------------------------------------------

pub const VkMemoryAllocateFlags = VkFlags;
pub const VkBufferCreateFlags = VkFlags;
pub const VkBufferUsageFlags = VkFlags;
pub const VkSharingMode = i32;

// Buffer usage flag bits (subset we care about).
pub const VK_BUFFER_USAGE_TRANSFER_SRC_BIT: VkBufferUsageFlags = 0x00000001;
pub const VK_BUFFER_USAGE_TRANSFER_DST_BIT: VkBufferUsageFlags = 0x00000002;
pub const VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT: VkBufferUsageFlags = 0x00000004;
pub const VK_BUFFER_USAGE_STORAGE_TEXEL_BUFFER_BIT: VkBufferUsageFlags = 0x00000008;
pub const VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT: VkBufferUsageFlags = 0x00000010;
pub const VK_BUFFER_USAGE_STORAGE_BUFFER_BIT: VkBufferUsageFlags = 0x00000020;
pub const VK_BUFFER_USAGE_INDEX_BUFFER_BIT: VkBufferUsageFlags = 0x00000040;
pub const VK_BUFFER_USAGE_VERTEX_BUFFER_BIT: VkBufferUsageFlags = 0x00000080;
pub const VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT: VkBufferUsageFlags = 0x00000100;

pub const VK_SHARING_MODE_EXCLUSIVE: VkSharingMode = 0;

pub const VkMemoryAllocateInfo = extern struct {
    sType: VkStructureType,
    pNext: ?*const anyopaque,
    allocationSize: VkDeviceSize,
    memoryTypeIndex: u32,
};

pub const VkMemoryRequirements = extern struct {
    size: VkDeviceSize,
    alignment: VkDeviceSize,
    memoryTypeBits: u32,
};

pub const VkBufferCreateInfo = extern struct {
    sType: VkStructureType,
    pNext: ?*const anyopaque,
    flags: VkBufferCreateFlags,
    size: VkDeviceSize,
    usage: VkBufferUsageFlags,
    sharingMode: VkSharingMode,
    queueFamilyIndexCount: u32,
    pQueueFamilyIndices: ?[*]const u32,
};

pub const VkMemoryMapFlags = VkFlags;

pub const VkMappedMemoryRange = extern struct {
    sType: VkStructureType,
    pNext: ?*const anyopaque,
    memory: VkDeviceMemory,
    offset: VkDeviceSize,
    size: VkDeviceSize,
};

pub const VK_WHOLE_SIZE: VkDeviceSize = ~@as(u64, 0);

// --- Compute pipeline + descriptor + command types -------------------------
// The M3 compute-dispatch path. Non-dispatchable handles (u64 = @intFromPtr of a
// heap struct) for everything except VkCommandBuffer, which IS dispatchable (the
// loader stamps its first word), so its backing struct is extern with the magic.

pub const VkShaderModule = u64;
pub const VkDescriptorSetLayout = u64;
pub const VkPipelineLayout = u64;
pub const VkPipeline = u64;
pub const VkPipelineCache = u64;
pub const VkDescriptorPool = u64;
pub const VkDescriptorSet = u64;
pub const VkCommandPool = u64;
pub const VkCommandBuffer = ?*opaque {};
pub const VkFence = u64;
pub const VkEvent = u64;
pub const VkSemaphore = u64;

pub const VkShaderModuleCreateFlags = VkFlags;
pub const VkPipelineLayoutCreateFlags = VkFlags;
pub const VkDescriptorSetLayoutCreateFlags = VkFlags;
pub const VkPipelineShaderStageCreateFlags = VkFlags;
pub const VkPipelineCreateFlags = VkFlags;
pub const VkDescriptorPoolCreateFlags = VkFlags;
pub const VkCommandPoolCreateFlags = VkFlags;
pub const VkCommandPoolResetFlags = VkFlags;
pub const VkCommandBufferResetFlags = VkFlags;
pub const VkCommandBufferUsageFlags = VkFlags;
pub const VkFenceCreateFlags = VkFlags;
pub const VkShaderStageFlags = VkFlags;
pub const VkPipelineBindPoint = i32;
pub const VkDescriptorType = i32;
pub const VkCommandBufferLevel = i32;

pub const VK_SHADER_STAGE_COMPUTE_BIT: VkShaderStageFlags = 0x00000020;
pub const VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER: VkDescriptorType = 1;
pub const VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER: VkDescriptorType = 6;
pub const VK_DESCRIPTOR_TYPE_STORAGE_BUFFER: VkDescriptorType = 7;
pub const VK_PIPELINE_BIND_POINT_COMPUTE: VkPipelineBindPoint = 1;
pub const VK_COMMAND_BUFFER_LEVEL_PRIMARY: VkCommandBufferLevel = 0;

pub const VkShaderModuleCreateInfo = extern struct {
    sType: VkStructureType,
    pNext: ?*const anyopaque,
    flags: VkShaderModuleCreateFlags,
    codeSize: usize, // bytes
    pCode: ?[*]const u32,
};

pub const VkDescriptorSetLayoutBinding = extern struct {
    binding: u32,
    descriptorType: VkDescriptorType,
    descriptorCount: u32,
    stageFlags: VkShaderStageFlags,
    pImmutableSamplers: ?*const anyopaque,
};

pub const VkDescriptorSetLayoutCreateInfo = extern struct {
    sType: VkStructureType,
    pNext: ?*const anyopaque,
    flags: VkDescriptorSetLayoutCreateFlags,
    bindingCount: u32,
    pBindings: ?[*]const VkDescriptorSetLayoutBinding,
};

pub const VkPushConstantRange = extern struct {
    stageFlags: VkShaderStageFlags,
    offset: u32,
    size: u32,
};

pub const VkPipelineLayoutCreateInfo = extern struct {
    sType: VkStructureType,
    pNext: ?*const anyopaque,
    flags: VkPipelineLayoutCreateFlags,
    setLayoutCount: u32,
    pSetLayouts: ?[*]const VkDescriptorSetLayout,
    pushConstantRangeCount: u32,
    pPushConstantRanges: ?[*]const VkPushConstantRange,
};

pub const VkSpecializationInfo = anyopaque;

pub const VkPipelineShaderStageCreateInfo = extern struct {
    sType: VkStructureType,
    pNext: ?*const anyopaque,
    flags: VkPipelineShaderStageCreateFlags,
    stage: VkShaderStageFlags,
    module: VkShaderModule,
    pName: ?[*:0]const u8,
    pSpecializationInfo: ?*const VkSpecializationInfo,
};

pub const VkComputePipelineCreateInfo = extern struct {
    sType: VkStructureType,
    pNext: ?*const anyopaque,
    flags: VkPipelineCreateFlags,
    stage: VkPipelineShaderStageCreateInfo,
    layout: VkPipelineLayout,
    basePipelineHandle: VkPipeline,
    basePipelineIndex: i32,
};

pub const VkDescriptorPoolSize = extern struct {
    type: VkDescriptorType,
    descriptorCount: u32,
};

pub const VkDescriptorPoolCreateInfo = extern struct {
    sType: VkStructureType,
    pNext: ?*const anyopaque,
    flags: VkDescriptorPoolCreateFlags,
    maxSets: u32,
    poolSizeCount: u32,
    pPoolSizes: ?[*]const VkDescriptorPoolSize,
};

pub const VkDescriptorSetAllocateInfo = extern struct {
    sType: VkStructureType,
    pNext: ?*const anyopaque,
    descriptorPool: VkDescriptorPool,
    descriptorSetCount: u32,
    pSetLayouts: ?[*]const VkDescriptorSetLayout,
};

pub const VkDescriptorBufferInfo = extern struct {
    buffer: VkBuffer,
    offset: VkDeviceSize,
    range: VkDeviceSize,
};

pub const VkSampler = u64;

pub const VkDescriptorImageInfo = extern struct {
    sampler: VkSampler,
    imageView: VkImageView,
    imageLayout: VkImageLayout,
};

pub const VkWriteDescriptorSet = extern struct {
    sType: VkStructureType,
    pNext: ?*const anyopaque,
    dstSet: VkDescriptorSet,
    dstBinding: u32,
    dstArrayElement: u32,
    descriptorCount: u32,
    descriptorType: VkDescriptorType,
    pImageInfo: ?*const VkDescriptorImageInfo,
    pBufferInfo: ?[*]const VkDescriptorBufferInfo,
    pTexelBufferView: ?*const anyopaque,
};

pub const VkCopyDescriptorSet = anyopaque;

pub const VkCommandPoolCreateInfo = extern struct {
    sType: VkStructureType,
    pNext: ?*const anyopaque,
    flags: VkCommandPoolCreateFlags,
    queueFamilyIndex: u32,
};

pub const VkCommandBufferAllocateInfo = extern struct {
    sType: VkStructureType,
    pNext: ?*const anyopaque,
    commandPool: VkCommandPool,
    level: VkCommandBufferLevel,
    commandBufferCount: u32,
};

pub const VkCommandBufferInheritanceInfo = anyopaque;

pub const VkCommandBufferBeginInfo = extern struct {
    sType: VkStructureType,
    pNext: ?*const anyopaque,
    flags: VkCommandBufferUsageFlags,
    pInheritanceInfo: ?*const VkCommandBufferInheritanceInfo,
};

pub const VkFenceCreateInfo = extern struct {
    sType: VkStructureType,
    pNext: ?*const anyopaque,
    flags: VkFenceCreateFlags,
};

pub const VkPipelineStageFlags = VkFlags;

pub const VkSubmitInfo = extern struct {
    sType: VkStructureType,
    pNext: ?*const anyopaque,
    waitSemaphoreCount: u32,
    pWaitSemaphores: ?[*]const VkSemaphore,
    pWaitDstStageMask: ?[*]const VkPipelineStageFlags,
    commandBufferCount: u32,
    pCommandBuffers: ?[*]const VkCommandBuffer,
    signalSemaphoreCount: u32,
    pSignalSemaphores: ?[*]const VkSemaphore,
};

// --- Graphics: image + render pass + framebuffer + pipeline (M4) -----------
// The offscreen-triangle path. Non-dispatchable handles (u64 = @intFromPtr of a
// heap struct). All extern structs cross-checked vs gcc on vulkan-headers
// 1.4.341.0 (aarch64-linux LP64). See the comptime asserts below.

pub const VkImage = u64;
pub const VkImageView = u64;
pub const VkRenderPass = u64;
pub const VkFramebuffer = u64;

pub const VkImageCreateFlags2 = VkFlags;
pub const VkImageLayout = i32;
pub const VkImageViewType = i32;
pub const VkComponentSwizzle = i32;
pub const VkImageAspectFlags = VkFlags;
pub const VkAttachmentLoadOp = i32;
pub const VkAttachmentStoreOp = i32;
pub const VkRenderPassCreateFlags = VkFlags;
pub const VkFramebufferCreateFlags = VkFlags;
pub const VkSubpassDescriptionFlags = VkFlags;
pub const VkBufferCopy = extern struct {
    srcOffset: VkDeviceSize,
    dstOffset: VkDeviceSize,
    size: VkDeviceSize,
};
pub const VkPrimitiveTopology = i32;
pub const VkIndexType = i32;
pub const VK_INDEX_TYPE_UINT16: VkIndexType = 0;
pub const VK_INDEX_TYPE_UINT32: VkIndexType = 1;
pub const VkPolygonMode = i32;
pub const VkCullModeFlags = VkFlags;
pub const VkFrontFace = i32;
pub const VkColorComponentFlags = VkFlags;
pub const VkSubpassContents = i32;

// The VkFormat enum members the offscreen triangle uses (int values from
// vulkan_core.h). VkFormat itself is an i32 alias (see above).
pub const VK_FORMAT_R8_UNORM: VkFormat = 9;
pub const VK_FORMAT_R8G8_UNORM: VkFormat = 16;
pub const VK_FORMAT_R8G8B8A8_UNORM: VkFormat = 37;
pub const VK_FORMAT_R8G8B8A8_SRGB: VkFormat = 43;
// Single/dual/quad-channel float pixel formats (vulkan_core.h enum values).
pub const VK_FORMAT_R16_SFLOAT: VkFormat = 76;
pub const VK_FORMAT_R16G16_SFLOAT: VkFormat = 83;
pub const VK_FORMAT_R16G16B16A16_SFLOAT: VkFormat = 97;
pub const VK_FORMAT_R32_SFLOAT: VkFormat = 100;
pub const VK_FORMAT_R32G32_SFLOAT: VkFormat = 103;
pub const VK_FORMAT_R32G32B32_SFLOAT: VkFormat = 106;
pub const VK_FORMAT_R32G32B32A32_SFLOAT: VkFormat = 109;
// Depth/stencil formats (vulkan_core.h enum values). D32_SFLOAT is the depth
// attachment the depth-test path uses. D16_UNORM + D24_UNORM_S8_UINT are accepted
// gracefully (mapped to the same f32 depth buffer).
pub const VK_FORMAT_D16_UNORM: VkFormat = 124;
pub const VK_FORMAT_X8_D24_UNORM_PACK32: VkFormat = 125;
pub const VK_FORMAT_D32_SFLOAT: VkFormat = 126;
pub const VK_FORMAT_D24_UNORM_S8_UINT: VkFormat = 129;
pub const VK_FORMAT_D32_SFLOAT_S8_UINT: VkFormat = 130;

/// Whether a VkFormat is a depth/stencil format (drives the depth attachment path).
pub fn formatIsDepth(f: VkFormat) bool {
    return switch (f) {
        VK_FORMAT_D16_UNORM,
        VK_FORMAT_X8_D24_UNORM_PACK32,
        VK_FORMAT_D32_SFLOAT,
        VK_FORMAT_D24_UNORM_S8_UINT,
        VK_FORMAT_D32_SFLOAT_S8_UINT,
        => true,
        else => false,
    };
}

pub const VK_IMAGE_TYPE_2D: VkImageType = 1;
pub const VK_IMAGE_TILING_OPTIMAL: VkImageTiling = 0;
pub const VK_IMAGE_VIEW_TYPE_2D: VkImageViewType = 1;
pub const VK_IMAGE_LAYOUT_UNDEFINED: VkImageLayout = 0;
pub const VK_IMAGE_ASPECT_COLOR_BIT: VkImageAspectFlags = 0x00000001;
pub const VK_IMAGE_ASPECT_DEPTH_BIT: VkImageAspectFlags = 0x00000002;
pub const VK_IMAGE_ASPECT_STENCIL_BIT: VkImageAspectFlags = 0x00000004;
// VkFormatFeatureFlagBits: the depth/stencil attachment capability bit, reported
// by vkGetPhysicalDeviceFormatProperties for D32_SFLOAT.
pub const VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT: VkFormatFeatureFlags = 0x00000200;
pub const VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT: VkFormatFeatureFlags = 0x00000001;
pub const VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT: VkFormatFeatureFlags = 0x00000080;
pub const VK_FORMAT_FEATURE_TRANSFER_DST_BIT: VkFormatFeatureFlags = 0x00008000;
pub const VK_ATTACHMENT_UNUSED: u32 = 0xFFFFFFFF;
pub const VK_ATTACHMENT_LOAD_OP_CLEAR: VkAttachmentLoadOp = 1;
pub const VK_ATTACHMENT_STORE_OP_STORE: VkAttachmentStoreOp = 0;
pub const VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST: VkPrimitiveTopology = 3;
pub const VK_PIPELINE_BIND_POINT_GRAPHICS: VkPipelineBindPoint = 0;
pub const VK_SUBPASS_CONTENTS_INLINE: VkSubpassContents = 0;
pub const VK_SHADER_STAGE_VERTEX_BIT: VkShaderStageFlags = 0x00000001;
pub const VK_SHADER_STAGE_FRAGMENT_BIT: VkShaderStageFlags = 0x00000010;
pub const VK_IMAGE_USAGE_TRANSFER_SRC_BIT: VkImageUsageFlags = 0x00000001;
pub const VK_IMAGE_USAGE_SAMPLED_BIT: VkImageUsageFlags = 0x00000004;
pub const VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT: VkImageUsageFlags = 0x00000010;
pub const VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL: VkImageLayout = 5;
pub const VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL: VkImageLayout = 7;

// VkFilter / VkSamplerAddressMode (vulkan_core.h enum values).
pub const VkFilter = i32;
pub const VK_FILTER_NEAREST: VkFilter = 0;
pub const VK_FILTER_LINEAR: VkFilter = 1;
pub const VkSamplerAddressMode = i32;
pub const VK_SAMPLER_ADDRESS_MODE_REPEAT: VkSamplerAddressMode = 0;
pub const VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT: VkSamplerAddressMode = 1;
pub const VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE: VkSamplerAddressMode = 2;
pub const VkSamplerMipmapMode = i32;
pub const VkBorderColor = i32;
pub const VkSamplerCreateFlags = VkFlags;

pub const VkExtent2D = extern struct { width: u32, height: u32 };
pub const VkOffset2D = extern struct { x: i32, y: i32 };
pub const VkOffset3D = extern struct { x: i32, y: i32, z: i32 };
pub const VkRect2D = extern struct { offset: VkOffset2D, extent: VkExtent2D };

pub const VkImageCreateInfo = extern struct {
    sType: VkStructureType,
    pNext: ?*const anyopaque,
    flags: VkImageCreateFlags,
    imageType: VkImageType,
    format: VkFormat,
    extent: VkExtent3D,
    mipLevels: u32,
    arrayLayers: u32,
    samples: VkSampleCountFlagBits,
    tiling: VkImageTiling,
    usage: VkImageUsageFlags,
    sharingMode: VkSharingMode,
    queueFamilyIndexCount: u32,
    pQueueFamilyIndices: ?[*]const u32,
    initialLayout: VkImageLayout,
};

pub const VkComponentMapping = extern struct {
    r: VkComponentSwizzle,
    g: VkComponentSwizzle,
    b: VkComponentSwizzle,
    a: VkComponentSwizzle,
};

pub const VkImageSubresourceRange = extern struct {
    aspectMask: VkImageAspectFlags,
    baseMipLevel: u32,
    levelCount: u32,
    baseArrayLayer: u32,
    layerCount: u32,
};

pub const VkImageViewCreateFlags = VkFlags;
pub const VkImageViewCreateInfo = extern struct {
    sType: VkStructureType,
    pNext: ?*const anyopaque,
    flags: VkImageViewCreateFlags,
    image: VkImage,
    viewType: VkImageViewType,
    format: VkFormat,
    components: VkComponentMapping,
    subresourceRange: VkImageSubresourceRange,
};

pub const VkSamplerCreateInfo = extern struct {
    sType: VkStructureType,
    pNext: ?*const anyopaque,
    flags: VkSamplerCreateFlags,
    magFilter: VkFilter,
    minFilter: VkFilter,
    mipmapMode: VkSamplerMipmapMode,
    addressModeU: VkSamplerAddressMode,
    addressModeV: VkSamplerAddressMode,
    addressModeW: VkSamplerAddressMode,
    mipLodBias: f32,
    anisotropyEnable: VkBool32,
    maxAnisotropy: f32,
    compareEnable: VkBool32,
    compareOp: VkCompareOp,
    minLod: f32,
    maxLod: f32,
    borderColor: VkBorderColor,
    unnormalizedCoordinates: VkBool32,
};

/// A single image subresource (aspect + mip level + array layer), for
/// vkGetImageSubresourceLayout.
pub const VkImageSubresource = extern struct {
    aspectMask: VkImageAspectFlags,
    mipLevel: u32,
    arrayLayer: u32,
};

/// The memory layout of a linear image subresource.
pub const VkSubresourceLayout = extern struct {
    offset: VkDeviceSize,
    size: VkDeviceSize,
    rowPitch: VkDeviceSize,
    arrayPitch: VkDeviceSize,
    depthPitch: VkDeviceSize,
};

pub const VkAttachmentDescriptionFlags = VkFlags;
pub const VkAttachmentDescription = extern struct {
    flags: VkAttachmentDescriptionFlags,
    format: VkFormat,
    samples: VkSampleCountFlagBits,
    loadOp: VkAttachmentLoadOp,
    storeOp: VkAttachmentStoreOp,
    stencilLoadOp: VkAttachmentLoadOp,
    stencilStoreOp: VkAttachmentStoreOp,
    initialLayout: VkImageLayout,
    finalLayout: VkImageLayout,
};

pub const VkAttachmentReference = extern struct {
    attachment: u32,
    layout: VkImageLayout,
};

pub const VkSubpassDescription = extern struct {
    flags: VkSubpassDescriptionFlags,
    pipelineBindPoint: VkPipelineBindPoint,
    inputAttachmentCount: u32,
    pInputAttachments: ?[*]const VkAttachmentReference,
    colorAttachmentCount: u32,
    pColorAttachments: ?[*]const VkAttachmentReference,
    pResolveAttachments: ?[*]const VkAttachmentReference,
    pDepthStencilAttachment: ?*const VkAttachmentReference,
    preserveAttachmentCount: u32,
    pPreserveAttachments: ?[*]const u32,
};

pub const VkSubpassDependency = anyopaque;
pub const VkRenderPassCreateInfo = extern struct {
    sType: VkStructureType,
    pNext: ?*const anyopaque,
    flags: VkRenderPassCreateFlags,
    attachmentCount: u32,
    pAttachments: ?[*]const VkAttachmentDescription,
    subpassCount: u32,
    pSubpasses: ?[*]const VkSubpassDescription,
    dependencyCount: u32,
    pDependencies: ?*const VkSubpassDependency,
};

pub const VkFramebufferCreateInfo = extern struct {
    sType: VkStructureType,
    pNext: ?*const anyopaque,
    flags: VkFramebufferCreateFlags,
    renderPass: VkRenderPass,
    attachmentCount: u32,
    pAttachments: ?[*]const VkImageView,
    width: u32,
    height: u32,
    layers: u32,
};

// --- Graphics pipeline state structs ---------------------------------------

pub const VkVertexInputRate = i32;
pub const VkVertexInputBindingDescription = extern struct {
    binding: u32,
    stride: u32,
    inputRate: VkVertexInputRate,
};

pub const VkVertexInputAttributeDescription = extern struct {
    location: u32,
    binding: u32,
    format: VkFormat,
    offset: u32,
};

pub const VkPipelineVertexInputStateCreateFlags = VkFlags;
pub const VkPipelineVertexInputStateCreateInfo = extern struct {
    sType: VkStructureType,
    pNext: ?*const anyopaque,
    flags: VkPipelineVertexInputStateCreateFlags,
    vertexBindingDescriptionCount: u32,
    pVertexBindingDescriptions: ?[*]const VkVertexInputBindingDescription,
    vertexAttributeDescriptionCount: u32,
    pVertexAttributeDescriptions: ?[*]const VkVertexInputAttributeDescription,
};

pub const VkPipelineInputAssemblyStateCreateFlags = VkFlags;
pub const VkPipelineInputAssemblyStateCreateInfo = extern struct {
    sType: VkStructureType,
    pNext: ?*const anyopaque,
    flags: VkPipelineInputAssemblyStateCreateFlags,
    topology: VkPrimitiveTopology,
    primitiveRestartEnable: VkBool32,
};

pub const VkViewport = extern struct {
    x: f32,
    y: f32,
    width: f32,
    height: f32,
    minDepth: f32,
    maxDepth: f32,
};

pub const VkPipelineViewportStateCreateFlags = VkFlags;
pub const VkPipelineViewportStateCreateInfo = extern struct {
    sType: VkStructureType,
    pNext: ?*const anyopaque,
    flags: VkPipelineViewportStateCreateFlags,
    viewportCount: u32,
    pViewports: ?[*]const VkViewport,
    scissorCount: u32,
    pScissors: ?[*]const VkRect2D,
};

// The rasterization state carries the cull mode + front-face winding the software
// path honors (so vkcube's back faces don't overdraw its lit front faces). The exact
// vulkan_core.h layout, so the loader's pointer resolves each field at the right offset.
pub const VkPipelineRasterizationStateCreateInfo = extern struct {
    sType: VkStructureType,
    pNext: ?*const anyopaque,
    flags: u32,
    depthClampEnable: VkBool32,
    rasterizerDiscardEnable: VkBool32,
    polygonMode: i32,
    cullMode: VkCullModeFlags,
    frontFace: VkFrontFace,
    depthBiasEnable: VkBool32,
    depthBiasConstantFactor: f32,
    depthBiasClamp: f32,
    depthBiasSlopeFactor: f32,
    lineWidth: f32,
};
pub const VK_CULL_MODE_NONE: VkCullModeFlags = 0;
pub const VK_CULL_MODE_FRONT_BIT: VkCullModeFlags = 0x00000001;
pub const VK_CULL_MODE_BACK_BIT: VkCullModeFlags = 0x00000002;
pub const VK_FRONT_FACE_COUNTER_CLOCKWISE: VkFrontFace = 0;
pub const VK_FRONT_FACE_CLOCKWISE: VkFrontFace = 1;

// The multisample state carries the MSAA sample count (rasterizationSamples). The
// software path reads it to render N samples/pixel then resolve. Layout cross-checked
// vs vulkan-headers 1.4.341.0 (gcc): size 48, rasterizationSamples@16, pSampleMask@32.
pub const VkPipelineMultisampleStateCreateInfo = extern struct {
    sType: VkStructureType,
    pNext: ?*const anyopaque,
    flags: VkFlags,
    rasterizationSamples: VkSampleCountFlagBits,
    sampleShadingEnable: VkBool32,
    minSampleShading: f32,
    pSampleMask: ?*const u32,
    alphaToCoverageEnable: VkBool32,
    alphaToOneEnable: VkBool32,
};

// VkSampleCountFlagBits values (a bitmask; one bit = one sample count).
pub const VK_SAMPLE_COUNT_1_BIT: VkSampleCountFlags = 0x00000001;
pub const VK_SAMPLE_COUNT_2_BIT: VkSampleCountFlags = 0x00000002;
pub const VK_SAMPLE_COUNT_4_BIT: VkSampleCountFlags = 0x00000004;
pub const VK_SAMPLE_COUNT_8_BIT: VkSampleCountFlags = 0x00000008;

// The color-blend state is passed by pointer and the software path ignores its
// contents (it always writes the FS color opaque), so it is an opaque blob.
// Color blend attachment: we model the colorWriteMask (glColorMask equivalent). The blend
// factors/ops are parsed-and-ignored for now (the software path blends via its own defaults).
// (VkColorComponentFlags is already aliased above.)
pub const VK_COLOR_COMPONENT_R_BIT: VkColorComponentFlags = 0x1;
pub const VK_COLOR_COMPONENT_G_BIT: VkColorComponentFlags = 0x2;
pub const VK_COLOR_COMPONENT_B_BIT: VkColorComponentFlags = 0x4;
pub const VK_COLOR_COMPONENT_A_BIT: VkColorComponentFlags = 0x8;
pub const VkPipelineColorBlendAttachmentState = extern struct {
    blendEnable: VkBool32,
    srcColorBlendFactor: i32,
    dstColorBlendFactor: i32,
    colorBlendOp: i32,
    srcAlphaBlendFactor: i32,
    dstAlphaBlendFactor: i32,
    alphaBlendOp: i32,
    colorWriteMask: VkColorComponentFlags,
};
pub const VkPipelineColorBlendStateCreateInfo = extern struct {
    sType: VkStructureType,
    pNext: ?*const anyopaque,
    flags: VkFlags,
    logicOpEnable: VkBool32,
    logicOp: i32,
    attachmentCount: u32,
    pAttachments: ?[*]const VkPipelineColorBlendAttachmentState,
    blendConstants: [4]f32,
};

// VkCompareOp: the depth-test comparison. A plain i32 alias keeps the enum-int ABI.
// The named operators below cover the depth path the rasterizer implements.
pub const VkCompareOp = i32;
pub const VK_COMPARE_OP_NEVER: VkCompareOp = 0;
pub const VK_COMPARE_OP_LESS: VkCompareOp = 1;
pub const VK_COMPARE_OP_EQUAL: VkCompareOp = 2;
pub const VK_COMPARE_OP_LESS_OR_EQUAL: VkCompareOp = 3;
pub const VK_COMPARE_OP_GREATER: VkCompareOp = 4;
pub const VK_COMPARE_OP_NOT_EQUAL: VkCompareOp = 5;
pub const VK_COMPARE_OP_GREATER_OR_EQUAL: VkCompareOp = 6;
pub const VK_COMPARE_OP_ALWAYS: VkCompareOp = 7;

// VkStencilOp: the stencil update action (a plain u32 in VkStencilOpState).
pub const VkStencilOp = u32;
pub const VK_STENCIL_OP_KEEP: VkStencilOp = 0;
pub const VK_STENCIL_OP_ZERO: VkStencilOp = 1;
pub const VK_STENCIL_OP_REPLACE: VkStencilOp = 2;
pub const VK_STENCIL_OP_INCREMENT_AND_CLAMP: VkStencilOp = 3;
pub const VK_STENCIL_OP_DECREMENT_AND_CLAMP: VkStencilOp = 4;
pub const VK_STENCIL_OP_INVERT: VkStencilOp = 5;
pub const VK_STENCIL_OP_INCREMENT_AND_WRAP: VkStencilOp = 6;
pub const VK_STENCIL_OP_DECREMENT_AND_WRAP: VkStencilOp = 7;

// VkStencilOpState is two of these per face. The software path ignores stencil, so
// it is an opaque blob sized to the real struct (7 u32-ish fields = 28 bytes). It
// only needs to occupy the right space inside the depth-stencil state struct.
pub const VkStencilOpState = extern struct {
    failOp: u32,
    passOp: u32,
    depthFailOp: u32,
    compareOp: VkCompareOp,
    compareMask: u32,
    writeMask: u32,
    reference: u32,
};

pub const VkPipelineDepthStencilStateCreateFlags = VkFlags;
pub const VkPipelineDepthStencilStateCreateInfo = extern struct {
    sType: VkStructureType,
    pNext: ?*const anyopaque,
    flags: VkPipelineDepthStencilStateCreateFlags,
    depthTestEnable: VkBool32,
    depthWriteEnable: VkBool32,
    depthCompareOp: VkCompareOp,
    depthBoundsTestEnable: VkBool32,
    stencilTestEnable: VkBool32,
    front: VkStencilOpState,
    back: VkStencilOpState,
    minDepthBounds: f32,
    maxDepthBounds: f32,
};
// VkDynamicState: which pipeline state is set dynamically (vkCmdSet*) instead of baked into the
// pipeline. We honor the stencil-mask/reference members (the common two-sided-outline / portal
// pattern that changes the stencil ref per object without a new pipeline). Other values are
// parsed-and-ignored (viewport/scissor are already handled via vkCmdSetScissor).
pub const VkDynamicState = enum(i32) {
    viewport = 0,
    scissor = 1,
    line_width = 2,
    depth_bias = 3,
    blend_constants = 4,
    depth_bounds = 5,
    stencil_compare_mask = 6,
    stencil_write_mask = 7,
    stencil_reference = 8,
    _,
};
pub const VK_DYNAMIC_STATE_STENCIL_COMPARE_MASK = VkDynamicState.stencil_compare_mask;
pub const VK_DYNAMIC_STATE_STENCIL_WRITE_MASK = VkDynamicState.stencil_write_mask;
pub const VK_DYNAMIC_STATE_STENCIL_REFERENCE = VkDynamicState.stencil_reference;

pub const VkPipelineDynamicStateCreateInfo = extern struct {
    sType: VkStructureType,
    pNext: ?*const anyopaque,
    flags: VkFlags,
    dynamicStateCount: u32,
    pDynamicStates: ?[*]const VkDynamicState,
};

// VkStencilFaceFlagBits: which face(s) a vkCmdSetStencil* call targets.
pub const VkStencilFaceFlags = VkFlags;
pub const VK_STENCIL_FACE_FRONT_BIT: VkStencilFaceFlags = 0x1;
pub const VK_STENCIL_FACE_BACK_BIT: VkStencilFaceFlags = 0x2;
pub const VK_STENCIL_FACE_FRONT_AND_BACK: VkStencilFaceFlags = 0x3;

pub const VkPipelineTessellationStateCreateInfo = anyopaque;

pub const VkGraphicsPipelineCreateInfo = extern struct {
    sType: VkStructureType,
    pNext: ?*const anyopaque,
    flags: VkPipelineCreateFlags,
    stageCount: u32,
    pStages: ?[*]const VkPipelineShaderStageCreateInfo,
    pVertexInputState: ?*const VkPipelineVertexInputStateCreateInfo,
    pInputAssemblyState: ?*const VkPipelineInputAssemblyStateCreateInfo,
    pTessellationState: ?*const VkPipelineTessellationStateCreateInfo,
    pViewportState: ?*const VkPipelineViewportStateCreateInfo,
    pRasterizationState: ?*const VkPipelineRasterizationStateCreateInfo,
    pMultisampleState: ?*const VkPipelineMultisampleStateCreateInfo,
    pDepthStencilState: ?*const VkPipelineDepthStencilStateCreateInfo,
    pColorBlendState: ?*const VkPipelineColorBlendStateCreateInfo,
    pDynamicState: ?*const VkPipelineDynamicStateCreateInfo,
    layout: VkPipelineLayout,
    renderPass: VkRenderPass,
    subpass: u32,
    basePipelineHandle: VkPipeline,
    basePipelineIndex: i32,
};

// --- Render-pass recording + image copy ------------------------------------

pub const VkClearColorValue = extern union {
    float32: [4]f32,
    int32: [4]i32,
    uint32: [4]u32,
};

pub const VkClearDepthStencilValue = extern struct {
    depth: f32,
    stencil: u32,
};

pub const VkClearValue = extern union {
    color: VkClearColorValue,
    // depthStencil is { f32, u32 } = 8 bytes, smaller than color (16), so the
    // union size is governed by color. pClearValues indexes by attachment, so a
    // depth attachment's clear value is read via .depthStencil.depth.
    depthStencil: VkClearDepthStencilValue,
};

pub const VkRenderPassBeginInfo = extern struct {
    sType: VkStructureType,
    pNext: ?*const anyopaque,
    renderPass: VkRenderPass,
    framebuffer: VkFramebuffer,
    renderArea: VkRect2D,
    clearValueCount: u32,
    pClearValues: ?[*]const VkClearValue,
};

pub const VkImageSubresourceLayers = extern struct {
    aspectMask: VkImageAspectFlags,
    mipLevel: u32,
    baseArrayLayer: u32,
    layerCount: u32,
};

pub const VkExtent3DC = VkExtent3D;
pub const VkBufferImageCopy = extern struct {
    bufferOffset: VkDeviceSize,
    bufferRowLength: u32,
    bufferImageHeight: u32,
    imageSubresource: VkImageSubresourceLayers,
    imageOffset: VkOffset3D,
    imageExtent: VkExtent3D,
};

// WSI: surface + swapchain (VK_KHR_surface / VK_KHR_wayland_surface / VK_KHR_swapchain).
// Prism's WSI is wired to its own from-scratch Wayland client (lib/prism/platform/wayland.zig):
// vkCreateWaylandSurfaceKHR receives a *prism.platform.Surface (passed through
// the `surface` field of VkWaylandSurfaceCreateInfoKHR as an opaque pointer),
// not a libwayland wl_display/wl_surface. The swapchain present reuses the exact
// platform.wayland present path that examples/triangle_wayland uses.
// libwayland-client ABI interop is a documented future gap.

pub const VkSurfaceKHR = u64;
pub const VkSwapchainKHR = u64;
pub const VkColorSpaceKHR = i32;
pub const VkPresentModeKHR = i32;
pub const VkSurfaceTransformFlagsKHR = VkFlags;
pub const VkCompositeAlphaFlagsKHR = VkFlags;
pub const VkSwapchainCreateFlagsKHR = VkFlags;
pub const VkWaylandSurfaceCreateFlagsKHR = VkFlags;

pub const VK_COLOR_SPACE_SRGB_NONLINEAR_KHR: VkColorSpaceKHR = 0;
pub const VK_PRESENT_MODE_FIFO_KHR: VkPresentModeKHR = 2;
pub const VK_PRESENT_MODE_MAILBOX_KHR: VkPresentModeKHR = 1;
pub const VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR: VkSurfaceTransformFlagsKHR = 0x00000001;
pub const VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR: VkCompositeAlphaFlagsKHR = 0x00000001;
pub const VK_FORMAT_B8G8R8A8_UNORM: VkFormat = 44;
pub const VK_FORMAT_B8G8R8A8_SRGB: VkFormat = 50;
pub const VK_IMAGE_USAGE_TRANSFER_DST_BIT: VkImageUsageFlags = 0x00000002;

/// vkCreateWaylandSurfaceKHR create info. `display` and `surface` are declared
/// `void*` in vulkan_wayland.h (struct wl_display* / struct wl_surface*). Prism
/// passes a *prism.platform.Surface in `surface` (and `display` is unused).
pub const VkWaylandSurfaceCreateInfoKHR = extern struct {
    sType: VkStructureType,
    pNext: ?*const anyopaque,
    flags: VkWaylandSurfaceCreateFlagsKHR,
    display: ?*anyopaque,
    surface: ?*anyopaque,
};

pub const VkSurfaceCapabilitiesKHR = extern struct {
    minImageCount: u32,
    maxImageCount: u32,
    currentExtent: VkExtent2D,
    minImageExtent: VkExtent2D,
    maxImageExtent: VkExtent2D,
    maxImageArrayLayers: u32,
    supportedTransforms: VkSurfaceTransformFlagsKHR,
    currentTransform: VkSurfaceTransformFlagsKHR,
    supportedCompositeAlpha: VkCompositeAlphaFlagsKHR,
    supportedUsageFlags: VkImageUsageFlags,
};

pub const VkSurfaceFormatKHR = extern struct {
    format: VkFormat,
    colorSpace: VkColorSpaceKHR,
};

pub const VkSwapchainCreateInfoKHR = extern struct {
    sType: VkStructureType,
    pNext: ?*const anyopaque,
    flags: VkSwapchainCreateFlagsKHR,
    surface: VkSurfaceKHR,
    minImageCount: u32,
    imageFormat: VkFormat,
    imageColorSpace: VkColorSpaceKHR,
    imageExtent: VkExtent2D,
    imageArrayLayers: u32,
    imageUsage: VkImageUsageFlags,
    imageSharingMode: VkSharingMode,
    queueFamilyIndexCount: u32,
    pQueueFamilyIndices: ?[*]const u32,
    preTransform: VkSurfaceTransformFlagsKHR,
    compositeAlpha: VkCompositeAlphaFlagsKHR,
    presentMode: VkPresentModeKHR,
    clipped: VkBool32,
    oldSwapchain: VkSwapchainKHR,
};

pub const VkPresentInfoKHR = extern struct {
    sType: VkStructureType,
    pNext: ?*const anyopaque,
    waitSemaphoreCount: u32,
    pWaitSemaphores: ?[*]const VkSemaphore,
    swapchainCount: u32,
    pSwapchains: ?[*]const VkSwapchainKHR,
    pImageIndices: ?[*]const u32,
    pResults: ?[*]VkResult,
};

pub const VkSemaphoreCreateInfo = extern struct {
    sType: VkStructureType,
    pNext: ?*const anyopaque,
    flags: VkFlags,
};

// --- Loader ICD interface --------------------------------------------------

/// The loader stamps this magic into the first uintptr_t of every dispatchable
/// handle. An ICD that allocates its own dispatchable handles (VkInstance,
/// VkPhysicalDevice) must place this value there or the loader's
/// `valid_loader_magic_value` check fails and it segfaults. See vk_icd.h.
pub const ICD_LOADER_MAGIC: usize = 0x01CDC0DE;

comptime {
    // Guard the load-bearing layouts. A drift here would make vulkaninfo print
    // garbage or crash, so fail the build instead.
    // Cross-checked against gcc on vulkan-headers 1.4.341.0 (aarch64-linux LP64).
    std.debug.assert(@sizeOf(VkPhysicalDeviceLimits) == 504);
    std.debug.assert(@offsetOf(VkPhysicalDeviceProperties, "deviceName") == 20);
    std.debug.assert(@offsetOf(VkPhysicalDeviceProperties, "pipelineCacheUUID") == 276);
    std.debug.assert(@offsetOf(VkPhysicalDeviceProperties, "limits") == 296);
    std.debug.assert(@sizeOf(VkPhysicalDeviceProperties) == 824);
    std.debug.assert(@sizeOf(VkPhysicalDeviceFeatures) == 220);
    std.debug.assert(@offsetOf(VkPhysicalDeviceMemoryProperties, "memoryHeapCount") == 260);
    std.debug.assert(@sizeOf(VkPhysicalDeviceMemoryProperties) == 520);

    // Device + memory + buffer structs (cross-checked vs gcc, see /tmp/sz.c).
    std.debug.assert(@sizeOf(VkDeviceQueueCreateInfo) == 40);
    std.debug.assert(@sizeOf(VkDeviceCreateInfo) == 72);
    std.debug.assert(@offsetOf(VkDeviceCreateInfo, "pQueueCreateInfos") == 24);
    std.debug.assert(@offsetOf(VkDeviceCreateInfo, "pEnabledFeatures") == 64);
    std.debug.assert(@sizeOf(VkMemoryAllocateInfo) == 32);
    std.debug.assert(@offsetOf(VkMemoryAllocateInfo, "memoryTypeIndex") == 24);
    std.debug.assert(@sizeOf(VkMemoryRequirements) == 24);
    std.debug.assert(@sizeOf(VkBufferCreateInfo) == 56);
    std.debug.assert(@offsetOf(VkBufferCreateInfo, "usage") == 32);
    std.debug.assert(@sizeOf(VkMappedMemoryRange) == 40);

    // Compute pipeline + descriptor + command structs (cross-checked vs gcc on
    // vulkan-headers 1.4.341.0, aarch64-linux LP64; see /tmp/sz.c).
    std.debug.assert(@sizeOf(VkShaderModuleCreateInfo) == 40);
    std.debug.assert(@offsetOf(VkShaderModuleCreateInfo, "pCode") == 32);
    std.debug.assert(@sizeOf(VkDescriptorSetLayoutBinding) == 24);
    std.debug.assert(@offsetOf(VkDescriptorSetLayoutBinding, "stageFlags") == 12);
    std.debug.assert(@sizeOf(VkDescriptorSetLayoutCreateInfo) == 32);
    std.debug.assert(@offsetOf(VkDescriptorSetLayoutCreateInfo, "pBindings") == 24);
    std.debug.assert(@sizeOf(VkPipelineLayoutCreateInfo) == 48);
    std.debug.assert(@offsetOf(VkPipelineLayoutCreateInfo, "pSetLayouts") == 24);
    std.debug.assert(@offsetOf(VkPipelineLayoutCreateInfo, "pushConstantRangeCount") == 32);
    std.debug.assert(@sizeOf(VkPipelineShaderStageCreateInfo) == 48);
    std.debug.assert(@offsetOf(VkPipelineShaderStageCreateInfo, "module") == 24);
    std.debug.assert(@offsetOf(VkPipelineShaderStageCreateInfo, "pName") == 32);
    std.debug.assert(@sizeOf(VkComputePipelineCreateInfo) == 96);
    std.debug.assert(@offsetOf(VkComputePipelineCreateInfo, "stage") == 24);
    std.debug.assert(@offsetOf(VkComputePipelineCreateInfo, "layout") == 72);
    std.debug.assert(@sizeOf(VkDescriptorPoolSize) == 8);
    std.debug.assert(@sizeOf(VkDescriptorPoolCreateInfo) == 40);
    std.debug.assert(@offsetOf(VkDescriptorPoolCreateInfo, "pPoolSizes") == 32);
    std.debug.assert(@sizeOf(VkDescriptorSetAllocateInfo) == 40);
    std.debug.assert(@offsetOf(VkDescriptorSetAllocateInfo, "pSetLayouts") == 32);
    std.debug.assert(@sizeOf(VkDescriptorBufferInfo) == 24);
    std.debug.assert(@sizeOf(VkWriteDescriptorSet) == 64);
    std.debug.assert(@offsetOf(VkWriteDescriptorSet, "dstBinding") == 24);
    std.debug.assert(@offsetOf(VkWriteDescriptorSet, "descriptorType") == 36);
    std.debug.assert(@offsetOf(VkWriteDescriptorSet, "pBufferInfo") == 48);
    std.debug.assert(@sizeOf(VkCommandPoolCreateInfo) == 24);
    std.debug.assert(@offsetOf(VkCommandPoolCreateInfo, "queueFamilyIndex") == 20);
    std.debug.assert(@sizeOf(VkCommandBufferAllocateInfo) == 32);
    std.debug.assert(@offsetOf(VkCommandBufferAllocateInfo, "level") == 24);
    std.debug.assert(@offsetOf(VkCommandBufferAllocateInfo, "commandBufferCount") == 28);
    std.debug.assert(@sizeOf(VkCommandBufferBeginInfo) == 32);
    std.debug.assert(@offsetOf(VkCommandBufferBeginInfo, "pInheritanceInfo") == 24);
    std.debug.assert(@sizeOf(VkFenceCreateInfo) == 24);
    std.debug.assert(@sizeOf(VkSubmitInfo) == 72);
    std.debug.assert(@offsetOf(VkSubmitInfo, "commandBufferCount") == 40);
    std.debug.assert(@offsetOf(VkSubmitInfo, "pCommandBuffers") == 48);

    // Graphics: image + render pass + framebuffer + pipeline structs (M4),
    // cross-checked vs gcc on vulkan-headers 1.4.341.0 (aarch64-linux LP64).
    std.debug.assert(@sizeOf(VkExtent2D) == 8);
    std.debug.assert(@sizeOf(VkOffset2D) == 8);
    std.debug.assert(@sizeOf(VkOffset3D) == 12);
    std.debug.assert(@sizeOf(VkRect2D) == 16);
    std.debug.assert(@sizeOf(VkImageCreateInfo) == 88);
    std.debug.assert(@offsetOf(VkImageCreateInfo, "format") == 24);
    std.debug.assert(@offsetOf(VkImageCreateInfo, "extent") == 28);
    std.debug.assert(@offsetOf(VkImageCreateInfo, "usage") == 56);
    std.debug.assert(@offsetOf(VkImageCreateInfo, "initialLayout") == 80);
    std.debug.assert(@sizeOf(VkComponentMapping) == 16);
    std.debug.assert(@sizeOf(VkImageSubresourceRange) == 20);
    // Texture API (cross-checked vs vulkan_core.h LP64).
    std.debug.assert(@sizeOf(VkDescriptorImageInfo) == 24);
    std.debug.assert(@offsetOf(VkDescriptorImageInfo, "imageView") == 8);
    std.debug.assert(@offsetOf(VkDescriptorImageInfo, "imageLayout") == 16);
    std.debug.assert(@sizeOf(VkSamplerCreateInfo) == 80);
    std.debug.assert(@offsetOf(VkSamplerCreateInfo, "magFilter") == 20);
    std.debug.assert(@offsetOf(VkSamplerCreateInfo, "minFilter") == 24);
    std.debug.assert(@offsetOf(VkSamplerCreateInfo, "addressModeU") == 32);
    std.debug.assert(@offsetOf(VkSamplerCreateInfo, "addressModeV") == 36);
    std.debug.assert(@offsetOf(VkSamplerCreateInfo, "borderColor") == 72);
    std.debug.assert(@sizeOf(VkImageSubresource) == 12);
    std.debug.assert(@sizeOf(VkSubresourceLayout) == 40);
    std.debug.assert(@sizeOf(VkImageViewCreateInfo) == 80);
    std.debug.assert(@offsetOf(VkImageViewCreateInfo, "image") == 24);
    std.debug.assert(@offsetOf(VkImageViewCreateInfo, "format") == 36);
    std.debug.assert(@offsetOf(VkImageViewCreateInfo, "subresourceRange") == 56);
    std.debug.assert(@sizeOf(VkAttachmentDescription) == 36);
    std.debug.assert(@offsetOf(VkAttachmentDescription, "loadOp") == 12);
    std.debug.assert(@offsetOf(VkAttachmentDescription, "storeOp") == 16);
    std.debug.assert(@offsetOf(VkAttachmentDescription, "finalLayout") == 32);
    std.debug.assert(@sizeOf(VkPipelineMultisampleStateCreateInfo) == 48);
    std.debug.assert(@offsetOf(VkPipelineMultisampleStateCreateInfo, "rasterizationSamples") == 20);
    std.debug.assert(@offsetOf(VkPipelineMultisampleStateCreateInfo, "pSampleMask") == 32);
    std.debug.assert(@sizeOf(VkAttachmentReference) == 8);
    std.debug.assert(@sizeOf(VkSubpassDescription) == 72);
    std.debug.assert(@offsetOf(VkSubpassDescription, "colorAttachmentCount") == 24);
    std.debug.assert(@offsetOf(VkSubpassDescription, "pColorAttachments") == 32);
    std.debug.assert(@sizeOf(VkRenderPassCreateInfo) == 64);
    std.debug.assert(@offsetOf(VkRenderPassCreateInfo, "pAttachments") == 24);
    std.debug.assert(@offsetOf(VkRenderPassCreateInfo, "pSubpasses") == 40);
    std.debug.assert(@sizeOf(VkFramebufferCreateInfo) == 64);
    std.debug.assert(@offsetOf(VkFramebufferCreateInfo, "attachmentCount") == 32);
    std.debug.assert(@offsetOf(VkFramebufferCreateInfo, "pAttachments") == 40);
    std.debug.assert(@offsetOf(VkFramebufferCreateInfo, "width") == 48);
    std.debug.assert(@sizeOf(VkVertexInputBindingDescription) == 12);
    std.debug.assert(@sizeOf(VkVertexInputAttributeDescription) == 16);
    std.debug.assert(@offsetOf(VkVertexInputAttributeDescription, "format") == 8);
    std.debug.assert(@offsetOf(VkVertexInputAttributeDescription, "offset") == 12);
    std.debug.assert(@sizeOf(VkPipelineVertexInputStateCreateInfo) == 48);
    std.debug.assert(@offsetOf(VkPipelineVertexInputStateCreateInfo, "pVertexBindingDescriptions") == 24);
    std.debug.assert(@offsetOf(VkPipelineVertexInputStateCreateInfo, "pVertexAttributeDescriptions") == 40);
    std.debug.assert(@sizeOf(VkPipelineInputAssemblyStateCreateInfo) == 32);
    std.debug.assert(@offsetOf(VkPipelineInputAssemblyStateCreateInfo, "topology") == 20);
    std.debug.assert(@sizeOf(VkViewport) == 24);
    std.debug.assert(@sizeOf(VkPipelineViewportStateCreateInfo) == 48);
    std.debug.assert(@sizeOf(VkGraphicsPipelineCreateInfo) == 144);
    std.debug.assert(@offsetOf(VkGraphicsPipelineCreateInfo, "pStages") == 24);
    std.debug.assert(@offsetOf(VkGraphicsPipelineCreateInfo, "pVertexInputState") == 32);
    std.debug.assert(@offsetOf(VkGraphicsPipelineCreateInfo, "layout") == 104);
    std.debug.assert(@offsetOf(VkGraphicsPipelineCreateInfo, "renderPass") == 112);
    std.debug.assert(@offsetOf(VkGraphicsPipelineCreateInfo, "subpass") == 120);
    // Depth/stencil (feature 2). Cross-checked vs gcc on vulkan-headers 1.4.341.0.
    std.debug.assert(@sizeOf(VkStencilOpState) == 28);
    std.debug.assert(@sizeOf(VkPipelineDepthStencilStateCreateInfo) == 104);
    std.debug.assert(@offsetOf(VkPipelineDepthStencilStateCreateInfo, "depthTestEnable") == 20);
    std.debug.assert(@offsetOf(VkPipelineDepthStencilStateCreateInfo, "depthCompareOp") == 28);
    std.debug.assert(@offsetOf(VkPipelineDepthStencilStateCreateInfo, "front") == 40);
    std.debug.assert(@offsetOf(VkPipelineDepthStencilStateCreateInfo, "back") == 68);
    std.debug.assert(@offsetOf(VkPipelineDepthStencilStateCreateInfo, "minDepthBounds") == 96);
    std.debug.assert(@sizeOf(VkClearDepthStencilValue) == 8);
    std.debug.assert(@sizeOf(VkClearColorValue) == 16);
    std.debug.assert(@sizeOf(VkClearValue) == 16);
    std.debug.assert(@sizeOf(VkRenderPassBeginInfo) == 64);
    std.debug.assert(@offsetOf(VkRenderPassBeginInfo, "framebuffer") == 24);
    std.debug.assert(@offsetOf(VkRenderPassBeginInfo, "renderArea") == 32);
    std.debug.assert(@offsetOf(VkRenderPassBeginInfo, "clearValueCount") == 48);
    std.debug.assert(@offsetOf(VkRenderPassBeginInfo, "pClearValues") == 56);
    std.debug.assert(@sizeOf(VkImageSubresourceLayers) == 16);
    std.debug.assert(@sizeOf(VkBufferImageCopy) == 56);
    std.debug.assert(@offsetOf(VkBufferImageCopy, "imageSubresource") == 16);
    std.debug.assert(@offsetOf(VkBufferImageCopy, "imageOffset") == 32);
    std.debug.assert(@offsetOf(VkBufferImageCopy, "imageExtent") == 44);

    // WSI (M6). Cross-checked vs vulkan-headers 1.4.341.0 (aarch64-linux LP64).
    std.debug.assert(@sizeOf(VkWaylandSurfaceCreateInfoKHR) == 40);
    std.debug.assert(@offsetOf(VkWaylandSurfaceCreateInfoKHR, "display") == 24);
    std.debug.assert(@offsetOf(VkWaylandSurfaceCreateInfoKHR, "surface") == 32);
    std.debug.assert(@sizeOf(VkSurfaceCapabilitiesKHR) == 52);
    std.debug.assert(@sizeOf(VkSurfaceFormatKHR) == 8);
    std.debug.assert(@sizeOf(VkSwapchainCreateInfoKHR) == 104);
    std.debug.assert(@offsetOf(VkSwapchainCreateInfoKHR, "surface") == 24);
    std.debug.assert(@offsetOf(VkSwapchainCreateInfoKHR, "imageExtent") == 44);
    std.debug.assert(@offsetOf(VkSwapchainCreateInfoKHR, "imageUsage") == 56);
    std.debug.assert(@offsetOf(VkSwapchainCreateInfoKHR, "presentMode") == 88);
    std.debug.assert(@offsetOf(VkSwapchainCreateInfoKHR, "oldSwapchain") == 96);
    std.debug.assert(@sizeOf(VkPresentInfoKHR) == 64);
    std.debug.assert(@offsetOf(VkPresentInfoKHR, "pSwapchains") == 40);
    std.debug.assert(@offsetOf(VkPresentInfoKHR, "pImageIndices") == 48);
}

test "version helper packs like VK_MAKE_API_VERSION" {
    try std.testing.expectEqual(@as(u32, (1 << 22) | (3 << 12)), VK_API_VERSION_1_3);
}

test "result codes match the spec" {
    try std.testing.expectEqual(@as(i32, 0), @intFromEnum(VkResult.VK_SUCCESS));
    try std.testing.expectEqual(@as(i32, 5), @intFromEnum(VkResult.VK_INCOMPLETE));
    try std.testing.expectEqual(@as(i32, -3), @intFromEnum(VkResult.VK_ERROR_INITIALIZATION_FAILED));
}
