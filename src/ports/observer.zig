//! Observer port: reacts to engine events (visualization, logging, metrics,
//! invariant checks, trace recording). Several observers can be attached at
//! once through `ObserverList`.
//!
//! Contract for an implementation type `T`:
//!     pub fn onEvent(self: *T, event: Event, state: *const State) void
//! optional:
//!     pub fn deinit(self: *T) void
//!
//! Rules:
//!   * Read only: never change simulation state.
//!   * Must not influence results: a run with or without observers must
//!     produce the same placements.
//!   * Errors (e.g. a failed file write) are handled inside the observer
//!     (log, set a flag); `onEvent` cannot fail the simulation.

const std = @import("std");
const types = @import("../model/types.zig");
pub const State = types.State;
pub const Event = types.Event;

pub const Observer = struct {
    ptr: *anyopaque,
    vtable: *const VTable,

    pub const VTable = struct {
        onEvent: *const fn (ptr: *anyopaque, event: Event, state: *const State) void,
        deinit: *const fn (ptr: *anyopaque) void,
    };

    pub fn onEvent(self: Observer, event: Event, state: *const State) void {
        self.vtable.onEvent(self.ptr, event, state);
    }

    pub fn deinit(self: Observer) void {
        self.vtable.deinit(self.ptr);
    }

    /// Wrap `impl: *T` as an Observer.
    pub fn init(impl: anytype) Observer {
        const Ptr = @TypeOf(impl);
        const T = @typeInfo(Ptr).pointer.child;
        if (!@hasDecl(T, "onEvent"))
            @compileError(@typeName(T) ++ " is not an Observer: missing `pub fn onEvent(self: *T, event: Event, state: *const State) void`");
        const gen = struct {
            fn onEvent(p: *anyopaque, event: Event, state: *const State) void {
                const self: Ptr = @ptrCast(@alignCast(p));
                self.onEvent(event, state);
            }
            fn deinit(p: *anyopaque) void {
                const self: Ptr = @ptrCast(@alignCast(p));
                if (@hasDecl(T, "deinit")) self.deinit();
            }
        };
        const vt = VTable{ .onEvent = gen.onEvent, .deinit = gen.deinit };
        return .{ .ptr = impl, .vtable = &vt };
    }
};

/// Fixed-capacity list of observers; `emit` fans an event out to all of them.
/// No allocation: attaching observers never touches the heap.
pub const ObserverList = struct {
    pub const max = 8;

    items: [max]Observer = undefined,
    len: usize = 0,

    pub fn add(self: *ObserverList, o: Observer) error{TooManyObservers}!void {
        if (self.len == max) return error.TooManyObservers;
        self.items[self.len] = o;
        self.len += 1;
    }

    pub fn emit(self: *const ObserverList, event: Event, state: *const State) void {
        for (self.items[0..self.len]) |o| o.onEvent(event, state);
    }

    /// Calls each observer's deinit, in reverse order of attachment.
    pub fn deinitAll(self: *ObserverList) void {
        var i = self.len;
        while (i > 0) {
            i -= 1;
            self.items[i].deinit();
        }
        self.len = 0;
    }
};

test "observer port: fan-out to several observers" {
    const Counter = struct {
        placed: usize = 0,
        ticks: usize = 0,
        pub fn onEvent(self: *@This(), event: Event, state: *const State) void {
            _ = state;
            switch (event) {
                .job_placed => self.placed += 1,
                .tick_advanced => self.ticks += 1,
                else => {},
            }
        }
    };
    var a = Counter{};
    var b = Counter{};
    var list = ObserverList{};
    try list.add(Observer.init(&a));
    try list.add(Observer.init(&b));

    const state: State = undefined;
    list.emit(.{ .tick_advanced = .{ .now = 1 } }, &state);
    list.emit(.{ .job_placed = .{ .id = 0, .slot = 0, .machine = 0, .tick = 1, .arrival = 0, .dur = 3 } }, &state);
    try std.testing.expectEqual(@as(usize, 1), a.ticks);
    try std.testing.expectEqual(@as(usize, 1), b.placed);
    list.deinitAll();
}
