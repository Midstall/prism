//! Prism's Vulkan ICD entry surface (libprism-vk.so). The Vulkan loader dlopens
//! this library and calls the vk_icd* exports. Everything dispatches to prism-vk/icd.zig.

const std = @import("std");
const prism = @import("prism");
pub const vk = @import("prism-vk/vk.zig");
pub const icd = @import("prism-vk/icd.zig");

/// Loader <-> ICD interface-version negotiation. The loader passes the highest
/// version it supports. We clamp to the highest we support (v5) and write it back.
export fn vk_icdNegotiateLoaderICDInterfaceVersion(pSupportedVersion: *u32) callconv(.c) vk.VkResult {
    if (pSupportedVersion.* > icd.loader_icd_interface_version) {
        pSupportedVersion.* = icd.loader_icd_interface_version;
    }
    return .VK_SUCCESS;
}

/// The ICD's primary entry point (loader interface v2+). The loader resolves
/// every Vulkan function through this by name.
export fn vk_icdGetInstanceProcAddr(
    instance: vk.VkInstance,
    pName: ?[*:0]const u8,
) callconv(.c) vk.PFN_vkVoidFunction {
    return icd.getInstanceProcAddr(instance, pName);
}

/// Loader interface v4+ physical-device-level proc lookup. We have no
/// physical-device-level dispatchable extension functions, so defer to the same
/// table (it returns null for anything physical-device specific we don't list).
export fn vk_icdGetPhysicalDeviceProcAddr(
    instance: vk.VkInstance,
    pName: ?[*:0]const u8,
) callconv(.c) vk.PFN_vkVoidFunction {
    return icd.getInstanceProcAddr(instance, pName);
}

test "core is reachable from the vk frontend" {
    try std.testing.expect(prism.drivers.all.len >= 1);
}

test "negotiation clamps to our supported interface version" {
    var v: u32 = 6;
    try std.testing.expectEqual(vk.VkResult.VK_SUCCESS, vk_icdNegotiateLoaderICDInterfaceVersion(&v));
    try std.testing.expectEqual(icd.loader_icd_interface_version, v);
    var v2: u32 = 3;
    _ = vk_icdNegotiateLoaderICDInterfaceVersion(&v2);
    try std.testing.expectEqual(@as(u32, 3), v2);
}

test {
    _ = vk;
    _ = icd;
}
