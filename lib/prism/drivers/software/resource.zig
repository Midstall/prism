const std = @import("std");
const hal = @import("../../hal.zig");

pub const Resource = struct {
    kind: hal.ResourceKind,
    bytes: []u8,
    // image-only metadata (undefined for buffers)
    width: u32 = 0,
    height: u32 = 0,
    format: hal.Format = .rgba8_unorm,
    // Mip level count (1 = base only). When > 1 `bytes` holds a tightly-packed mip chain
    // (level 0 first, then w/2 x h/2, ... per hal.mipLevelOffset) the sampler can read.
    mip_levels: u8 = 1,
    // Cubemap: `bytes` holds 6 faces packed in GL order (each width*height*bpp). A cube
    // sampler picks the face from the sampled direction. Default false = ordinary 2D image.
    is_cube: bool = false,
    // 3D texture: the Z-slice count (default 1). When > 1 `bytes` holds width*height*depth
    // texels packed slice-major. A sampler3D trilinearly interpolates the volume. Also the layer
    // count of a 2D array (`is_array`).
    depth: u32 = 1,
    // 2D array texture: when true the `depth` layers are independent 2D images (sampler2DArray),
    // not a 3D volume. Same layer-major backing as 3D. Distinguishes the two (both depth > 1).
    is_array: bool = false,

    pub fn sizeOf(desc: hal.ResourceDesc) usize {
        return switch (desc) {
            .buffer => |b| b.size,
            // A multisampled image holds `samples` pixels per logical pixel (sample-minor).
            // A mip-chained image (mip_levels > 1) holds every level packed contiguously
            // (level 0 first). MSAA + mipmaps don't co-occur (samples>1 is render-target).
            // A cubemap holds 6 faces packed contiguously. Each face is one image or (with
            // mip_levels > 1) its own mip chain (prefiltered env maps). A 3D image holds
            // `depth` slices packed slice-major.
            .image => |i| if (i.cube)
                6 * (if (i.mip_levels > 1)
                    hal.mipChainBytes(i.width, i.height, i.mip_levels, i.format.bytesPerPixel())
                else
                    @as(usize, i.width) * i.height * i.format.bytesPerPixel())
            else if (i.depth > 1)
                @as(usize, i.width) * i.height * i.depth * i.format.bytesPerPixel()
            else if (i.mip_levels > 1)
                hal.mipChainBytes(i.width, i.height, i.mip_levels, i.format.bytesPerPixel())
            else
                @as(usize, i.width) * i.height * @max(1, i.samples) * i.format.bytesPerPixel(),
        };
    }
};

test "image size computation" {
    const desc = hal.ResourceDesc{ .image = .{ .width = 4, .height = 4, .format = .rgba8_unorm } };
    try std.testing.expectEqual(@as(usize, 4 * 4 * 4), Resource.sizeOf(desc));
}

test "mip-chained image size sums all levels" {
    // 4x4 RGBA8, 3 levels: 4x4 (64) + 2x2 (16) + 1x1 (4) = 84 bytes.
    const desc = hal.ResourceDesc{ .image = .{ .width = 4, .height = 4, .format = .rgba8_unorm, .mip_levels = 3 } };
    try std.testing.expectEqual(@as(usize, 64 + 16 + 4), Resource.sizeOf(desc));
}
