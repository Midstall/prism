const hal = @import("../hal.zig");

pub const Buffer = struct {
    bytes: []u8,
    width: u32,
    height: u32,
    stride: u32,
    format: hal.Format,
};

pub const SurfaceDesc = struct {
    width: u32,
    height: u32,
    format: hal.Format = .rgba8_unorm,
};

/// Result of processing one windowing-system event.
pub const WindowEvent = enum {
    /// Nothing actionable for the caller (a ping was answered, etc.).
    none,
    /// The window was resized. Caller should query size() and re-render.
    resized,
    /// The user asked to close the window. Caller should exit.
    closed,
};

pub const Surface = struct {
    ptr: *anyopaque,
    vtable: *const VTable,

    pub const VTable = struct {
        currentBuffer: *const fn (ptr: *anyopaque) hal.Error!Buffer,
        commit: *const fn (ptr: *anyopaque) hal.Error!void,
        /// Block on and handle the next windowing-system event. Backends with
        /// no event source (headless) return .none.
        processEvents: *const fn (ptr: *anyopaque) hal.Error!WindowEvent,
        /// Current surface dimensions, {width, height}.
        size: *const fn (ptr: *anyopaque) [2]u32,
        deinit: *const fn (ptr: *anyopaque) void,
    };

    pub fn currentBuffer(self: Surface) hal.Error!Buffer {
        return self.vtable.currentBuffer(self.ptr);
    }
    pub fn commit(self: Surface) hal.Error!void {
        return self.vtable.commit(self.ptr);
    }
    pub fn processEvents(self: Surface) hal.Error!WindowEvent {
        return self.vtable.processEvents(self.ptr);
    }
    pub fn size(self: Surface) [2]u32 {
        return self.vtable.size(self.ptr);
    }
    pub fn deinit(self: Surface) void {
        self.vtable.deinit(self.ptr);
    }
};

pub const Display = struct {
    ptr: *anyopaque,
    vtable: *const VTable,

    pub const VTable = struct {
        createSurface: *const fn (ptr: *anyopaque, desc: SurfaceDesc) hal.Error!Surface,
        deinit: *const fn (ptr: *anyopaque) void,
    };

    pub fn createSurface(self: Display, desc: SurfaceDesc) hal.Error!Surface {
        return self.vtable.createSurface(self.ptr, desc);
    }
    pub fn deinit(self: Display) void {
        self.vtable.deinit(self.ptr);
    }
};
