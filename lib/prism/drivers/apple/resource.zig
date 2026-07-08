const asahi = @import("asahi");
const hal = @import("../../hal.zig");

/// A GPU-backed HAL resource on the Apple Silicon (AGX) path: an asahi GEM BO
/// (handle + GPU VA + CPU mapping), bound into the device's VM at a bump-allocated
/// 16-KiB-aligned GPU VA above the render-infra range. Buffers and images share
/// this type. An image carries its dimensions + format so the context can clear it.
pub const Resource = struct {
    kind: hal.ResourceKind,
    bo: asahi.Bo,
    desc: hal.ResourceDesc,
    // image-only metadata (zero for buffers), cached for the clear/present paths.
    width: u32 = 0,
    height: u32 = 0,
    format: hal.Format = .rgba8_unorm,
};
