//! EGL 1.5 + libglvnd EGL-vendor C-ABI types in Zig (zero C deps). Mirrors
//! EGL/egl.h, EGL/eglext.h, and glvnd/libeglabi.h (libglvnd 1.7.0). Wrong
//! layout in EGLapiExports/EGLapiImports misaligns the vtable. Byte-faithful.

const std = @import("std");

// --- Core scalar aliases (EGL/egl.h) ---------------------------------------

pub const EGLBoolean = c_uint; // typedef unsigned int EGLBoolean;
pub const EGLint = i32; // typedef khronos_int32_t EGLint;
pub const EGLenum = c_uint; // typedef unsigned int EGLenum;
pub const EGLAttrib = isize; // typedef intptr_t EGLAttrib;

pub const EGLDisplay = ?*anyopaque; // typedef void *EGLDisplay;
pub const EGLConfig = ?*anyopaque; // typedef void *EGLConfig;
pub const EGLSurface = ?*anyopaque; // typedef void *EGLSurface;
pub const EGLContext = ?*anyopaque; // typedef void *EGLContext;
pub const EGLClientBuffer = ?*anyopaque;
pub const EGLDeviceEXT = ?*anyopaque;
pub const EGLNativeDisplayType = ?*anyopaque; // void* on the Wayland/surfaceless platforms

/// typedef void (*__eglMustCastToProperFunctionPointerType)(void);
pub const ProcFn = ?*const fn () callconv(.c) void;

pub const GLboolean = u8;

// --- Boolean + sentinel constants ------------------------------------------

pub const EGL_FALSE: EGLBoolean = 0;
pub const EGL_TRUE: EGLBoolean = 1;
pub const EGL_DONT_CARE: EGLint = -1;
pub const EGL_NO_DISPLAY: EGLDisplay = null;
pub const EGL_NO_CONTEXT: EGLContext = null;
pub const EGL_NO_SURFACE: EGLSurface = null;
pub const EGL_DEFAULT_DISPLAY: EGLNativeDisplayType = null;

// --- Error codes (returned by eglGetError) ---------------------------------

pub const EGL_SUCCESS: EGLint = 0x3000;
pub const EGL_NOT_INITIALIZED: EGLint = 0x3001;
pub const EGL_BAD_ACCESS: EGLint = 0x3002;
pub const EGL_BAD_ALLOC: EGLint = 0x3003;
pub const EGL_BAD_ATTRIBUTE: EGLint = 0x3004;
pub const EGL_BAD_CONFIG: EGLint = 0x3005;
pub const EGL_BAD_CONTEXT: EGLint = 0x3006;
pub const EGL_BAD_CURRENT_SURFACE: EGLint = 0x3007;
pub const EGL_BAD_DISPLAY: EGLint = 0x3008;
pub const EGL_BAD_MATCH: EGLint = 0x3009;
pub const EGL_BAD_NATIVE_PIXMAP: EGLint = 0x300A;
pub const EGL_BAD_NATIVE_WINDOW: EGLint = 0x300B;
pub const EGL_BAD_PARAMETER: EGLint = 0x300C;
pub const EGL_BAD_SURFACE: EGLint = 0x300D;

// --- eglQueryString names ---------------------------------------------------

pub const EGL_VENDOR: EGLint = 0x3053;
pub const EGL_VERSION: EGLint = 0x3054;
pub const EGL_EXTENSIONS: EGLint = 0x3055;
pub const EGL_CLIENT_APIS: EGLint = 0x308D;

// --- Config attributes ------------------------------------------------------

pub const EGL_BUFFER_SIZE: EGLint = 0x3020;
pub const EGL_ALPHA_SIZE: EGLint = 0x3021;
pub const EGL_BLUE_SIZE: EGLint = 0x3022;
pub const EGL_GREEN_SIZE: EGLint = 0x3023;
pub const EGL_RED_SIZE: EGLint = 0x3024;
pub const EGL_DEPTH_SIZE: EGLint = 0x3025;
pub const EGL_STENCIL_SIZE: EGLint = 0x3026;
pub const EGL_CONFIG_CAVEAT: EGLint = 0x3027;
pub const EGL_CONFIG_ID: EGLint = 0x3028;
pub const EGL_LEVEL: EGLint = 0x3029;
pub const EGL_MAX_PBUFFER_HEIGHT: EGLint = 0x302A;
pub const EGL_MAX_PBUFFER_PIXELS: EGLint = 0x302B;
pub const EGL_MAX_PBUFFER_WIDTH: EGLint = 0x302C;
pub const EGL_NATIVE_RENDERABLE: EGLint = 0x302D;
pub const EGL_NATIVE_VISUAL_ID: EGLint = 0x302E;
pub const EGL_NATIVE_VISUAL_TYPE: EGLint = 0x302F;
pub const EGL_SAMPLES: EGLint = 0x3031;
pub const EGL_SAMPLE_BUFFERS: EGLint = 0x3032;
pub const EGL_SURFACE_TYPE: EGLint = 0x3033;
pub const EGL_TRANSPARENT_TYPE: EGLint = 0x3034;
pub const EGL_TRANSPARENT_BLUE_VALUE: EGLint = 0x3035;
pub const EGL_TRANSPARENT_GREEN_VALUE: EGLint = 0x3036;
pub const EGL_TRANSPARENT_RED_VALUE: EGLint = 0x3037;
pub const EGL_NONE: EGLint = 0x3038;
pub const EGL_BIND_TO_TEXTURE_RGB: EGLint = 0x3039;
pub const EGL_BIND_TO_TEXTURE_RGBA: EGLint = 0x303A;
pub const EGL_MIN_SWAP_INTERVAL: EGLint = 0x303B;
pub const EGL_MAX_SWAP_INTERVAL: EGLint = 0x303C;
pub const EGL_LUMINANCE_SIZE: EGLint = 0x303D;
pub const EGL_ALPHA_MASK_SIZE: EGLint = 0x303E;
pub const EGL_COLOR_BUFFER_TYPE: EGLint = 0x303F;
pub const EGL_RENDERABLE_TYPE: EGLint = 0x3040;
pub const EGL_MATCH_NATIVE_PIXMAP: EGLint = 0x3041;
pub const EGL_CONFORMANT: EGLint = 0x3042;

// Config attribute VALUES.
pub const EGL_SLOW_CONFIG: EGLint = 0x3050;
pub const EGL_NON_CONFORMANT_CONFIG: EGLint = 0x3051;
pub const EGL_TRANSPARENT_RGB: EGLint = 0x3052;
pub const EGL_RGB_BUFFER: EGLint = 0x308E;
pub const EGL_LUMINANCE_BUFFER: EGLint = 0x308F;

// EGL_SURFACE_TYPE bits.
pub const EGL_PBUFFER_BIT: EGLint = 0x0001;
pub const EGL_PIXMAP_BIT: EGLint = 0x0002;
pub const EGL_WINDOW_BIT: EGLint = 0x0004;

// EGL_RENDERABLE_TYPE / EGL_CONFORMANT bits.
pub const EGL_OPENGL_ES_BIT: EGLint = 0x0001;
pub const EGL_OPENVG_BIT: EGLint = 0x0002;
pub const EGL_OPENGL_ES2_BIT: EGLint = 0x0004;
pub const EGL_OPENGL_BIT: EGLint = 0x0008;
pub const EGL_OPENGL_ES3_BIT: EGLint = 0x00000040;

// --- Surface attributes + queries -------------------------------------------

pub const EGL_HEIGHT: EGLint = 0x3056;
pub const EGL_WIDTH: EGLint = 0x3057;
pub const EGL_LARGEST_PBUFFER: EGLint = 0x3058;
pub const EGL_TEXTURE_FORMAT: EGLint = 0x3080;
pub const EGL_TEXTURE_TARGET: EGLint = 0x3081;
pub const EGL_MIPMAP_TEXTURE: EGLint = 0x3082;
pub const EGL_MIPMAP_LEVEL: EGLint = 0x3083;
pub const EGL_RENDER_BUFFER: EGLint = 0x3086;
pub const EGL_CONFIG_ID_SURFACE: EGLint = 0x3028; // same enum as config's EGL_CONFIG_ID

// eglQueryContext attributes.
pub const EGL_CONTEXT_CLIENT_TYPE: EGLint = 0x3097;
pub const EGL_CONTEXT_CLIENT_VERSION: EGLint = 0x3098;
pub const EGL_CONTEXT_MAJOR_VERSION: EGLint = 0x3098; // alias of CLIENT_VERSION

// --- eglBindAPI / eglQueryAPI -----------------------------------------------

pub const EGL_OPENGL_ES_API: EGLenum = 0x30A0;
pub const EGL_OPENVG_API: EGLenum = 0x30A1;
pub const EGL_OPENGL_API: EGLenum = 0x30A2;

// --- Platform enums (eglext.h) ----------------------------------------------

pub const EGL_PLATFORM_GBM_KHR: EGLenum = 0x31D7;
pub const EGL_PLATFORM_GBM_MESA: EGLenum = 0x31D7;
pub const EGL_PLATFORM_WAYLAND_KHR: EGLenum = 0x31D8;
pub const EGL_PLATFORM_WAYLAND_EXT: EGLenum = 0x31D8;
pub const EGL_PLATFORM_X11_KHR: EGLenum = 0x31D5;
pub const EGL_PLATFORM_DEVICE_EXT: EGLenum = 0x313F;
pub const EGL_PLATFORM_SURFACELESS_MESA: EGLenum = 0x31DD;

// `EGL_NONE` (0x3038) is also used as the "no platform" sentinel that libEGL
// passes to getPlatformDisplay when the app called eglGetDisplay(EGL_DEFAULT_DISPLAY).
pub const EGL_PLATFORM_NONE: EGLenum = @intCast(EGL_NONE);

// libglvnd EGL vendor ABI (glvnd/libeglabi.h)

/// EGL_VENDOR_ABI_VERSION = (0 << 16) | 2 in libglvnd 1.7.0.
pub const EGL_VENDOR_ABI_MAJOR_VERSION: u32 = 0;
pub const EGL_VENDOR_ABI_MINOR_VERSION: u32 = 2;

pub fn abiMajor(version: u32) u32 {
    return version >> 16;
}
pub fn abiMinor(version: u32) u32 {
    return version & 0xFFFF;
}

/// Opaque per-vendor handle libEGL hands us in __egl_Main. We never dereference it.
pub const EGLvendorInfo = anyopaque;

/// typedef GLboolean (*DispatchPatchLookupStubOffset)(const char *funcName,
///         void **writePtr, const void **execPtr);
pub const DispatchPatchLookupStubOffset = ?*const fn (
    funcName: ?[*:0]const u8,
    writePtr: ?*?*anyopaque,
    execPtr: ?*?*const anyopaque,
) callconv(.c) GLboolean;

/// __EGLapiExports: the table libEGL.so passes us. Field order is byte-faithful
/// to libeglabi.h. We only call setEGLError, but the whole struct must lay out
/// correctly so libEGL reads our imports at the right offsets.
pub const EGLapiExports = extern struct {
    threadInit: ?*const fn () callconv(.c) void,
    getCurrentApi: ?*const fn () callconv(.c) EGLenum,
    getCurrentVendor: ?*const fn () callconv(.c) ?*EGLvendorInfo,
    getCurrentContext: ?*const fn () callconv(.c) EGLContext,
    getCurrentDisplay: ?*const fn () callconv(.c) EGLDisplay,
    getCurrentSurface: ?*const fn (readDraw: EGLint) callconv(.c) EGLSurface,
    fetchDispatchEntry: ?*const fn (dynDispatch: ?*EGLvendorInfo, index: c_int) callconv(.c) ProcFn,
    setEGLError: ?*const fn (errorCode: EGLint) callconv(.c) void,
    setLastVendor: ?*const fn (vendor: ?*EGLvendorInfo) callconv(.c) EGLBoolean,
    getVendorFromDisplay: ?*const fn (dpy: EGLDisplay) callconv(.c) ?*EGLvendorInfo,
    getVendorFromDevice: ?*const fn (dev: EGLDeviceEXT) callconv(.c) ?*EGLvendorInfo,
    setVendorForDevice: ?*const fn (dev: EGLDeviceEXT, vendor: ?*EGLvendorInfo) callconv(.c) EGLBoolean,
};

/// __EGLapiImports: the table we fill in and return through __egl_Main. Field
/// order is byte-faithful to libeglabi.h. Non-optional fields (getPlatformDisplay,
/// getSupportsAPI, getProcAddress, getDispatchAddress, setDispatchIndex) must be
/// set. The patch/thread hooks are optional and left null.
pub const EGLapiImports = extern struct {
    getPlatformDisplay: ?*const fn (
        platform: EGLenum,
        nativeDisplay: ?*anyopaque,
        attrib_list: ?[*]const EGLAttrib,
    ) callconv(.c) EGLDisplay,
    getSupportsAPI: ?*const fn (api: EGLenum) callconv(.c) EGLBoolean,
    getVendorString: ?*const fn (name: c_int) callconv(.c) ?[*:0]const u8,
    getProcAddress: ?*const fn (procName: ?[*:0]const u8) callconv(.c) ?*anyopaque,
    getDispatchAddress: ?*const fn (procName: ?[*:0]const u8) callconv(.c) ?*anyopaque,
    setDispatchIndex: ?*const fn (procName: ?[*:0]const u8, index: c_int) callconv(.c) void,
    isPatchSupported: ?*const fn (type_: c_int, stubSize: c_int) callconv(.c) GLboolean,
    initiatePatch: ?*const fn (type_: c_int, stubSize: c_int, lookupStubOffset: DispatchPatchLookupStubOffset) callconv(.c) GLboolean,
    releasePatch: ?*const fn () callconv(.c) void,
    patchThreadAttach: ?*const fn () callconv(.c) void,
    findNativeDisplayPlatform: ?*const fn (native_display: ?*anyopaque) callconv(.c) EGLenum,
};

/// The value of `name` for getVendorString: __EGL_VENDOR_STRING_PLATFORM_EXTENSIONS.
pub const VENDOR_STRING_PLATFORM_EXTENSIONS: c_int = 0;

/// __egl_Main prototype:
/// EGLBoolean __egl_Main(uint32_t version, const __EGLapiExports *exports,
///         __EGLvendorInfo *vendor, __EGLapiImports *imports);
pub const PFNEGLMAIN = ?*const fn (
    version: u32,
    exports: ?*const EGLapiExports,
    vendor: ?*EGLvendorInfo,
    imports: ?*EGLapiImports,
) callconv(.c) EGLBoolean;

test "scalar layouts match the EGL/glvnd C ABI" {
    try std.testing.expectEqual(@as(usize, 4), @sizeOf(EGLBoolean));
    try std.testing.expectEqual(@as(usize, 4), @sizeOf(EGLint));
    try std.testing.expectEqual(@as(usize, 4), @sizeOf(EGLenum));
    try std.testing.expectEqual(@sizeOf(isize), @sizeOf(EGLAttrib));
    try std.testing.expectEqual(@sizeOf(usize), @sizeOf(EGLDisplay));
    // 12 pointer-sized exports, 11 pointer-sized imports.
    try std.testing.expectEqual(@as(usize, 12 * @sizeOf(usize)), @sizeOf(EGLapiExports));
    try std.testing.expectEqual(@as(usize, 11 * @sizeOf(usize)), @sizeOf(EGLapiImports));
}
