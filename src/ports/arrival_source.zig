//! ArrivalSource port: creates job metadata — which jobs exist and WHEN they
//! arrive. It does not create the usage profile; the engine asks the
//! UsageGenerator for that when a job arrives.
//!
//! Contract for an implementation type `T`:
//!     pub fn next(self: *T, now: i64) Error!?JobMeta
//!         Next job with arrival <= now, or null if none (yet) at this tick.
//!         Called repeatedly each tick until it returns null.
//!     pub fn done(self: *T) bool
//!         True when no further job will ever be returned.
//! optional:
//!     pub fn reset(self: *T, seed: u64) void
//!     pub fn deinit(self: *T) void
//!
//! Rules:
//!   * Non-decreasing arrival times; never return arrival > now.
//!   * Unique ids within an episode.
//!   * Deterministic for a given seed.
//!
//! Example adapters: fixed list (tests), Poisson / uniform random arrivals,
//! trace replay (CSV / SWF), externally pushed (Python).

const std = @import("std");
const types = @import("../model/types.zig");
pub const JobMeta = types.JobMeta;

pub const Error = error{ OutOfMemory, InvalidInput, ReadFailed };

pub const ArrivalSource = struct {
    ptr: *anyopaque,
    vtable: *const VTable,

    pub const VTable = struct {
        next: *const fn (ptr: *anyopaque, now: i64) Error!?JobMeta,
        done: *const fn (ptr: *anyopaque) bool,
        reset: *const fn (ptr: *anyopaque, seed: u64) void,
        deinit: *const fn (ptr: *anyopaque) void,
    };

    pub fn next(self: ArrivalSource, now: i64) Error!?JobMeta {
        return self.vtable.next(self.ptr, now);
    }
    pub fn done(self: ArrivalSource) bool {
        return self.vtable.done(self.ptr);
    }
    pub fn reset(self: ArrivalSource, seed: u64) void {
        self.vtable.reset(self.ptr, seed);
    }
    pub fn deinit(self: ArrivalSource) void {
        self.vtable.deinit(self.ptr);
    }

    pub fn init(impl: anytype) ArrivalSource {
        const Ptr = @TypeOf(impl);
        const T = @typeInfo(Ptr).pointer.child;
        inline for (.{ "next", "done" }) |name| {
            if (!@hasDecl(T, name))
                @compileError(@typeName(T) ++ " is not an ArrivalSource: missing `" ++ name ++ "`");
        }
        const gen = struct {
            fn cast(p: *anyopaque) Ptr {
                return @ptrCast(@alignCast(p));
            }
            fn next(p: *anyopaque, now: i64) Error!?JobMeta {
                return cast(p).next(now);
            }
            fn done(p: *anyopaque) bool {
                return cast(p).done();
            }
            fn reset(p: *anyopaque, seed: u64) void {
                if (@hasDecl(T, "reset")) cast(p).reset(seed);
            }
            fn deinit(p: *anyopaque) void {
                if (@hasDecl(T, "deinit")) cast(p).deinit();
            }
        };
        const vt = VTable{ .next = gen.next, .done = gen.done, .reset = gen.reset, .deinit = gen.deinit };
        return .{ .ptr = impl, .vtable = &vt };
    }
};

test "arrival source port: fixed list" {
    const Fixed = struct {
        items: []const JobMeta,
        i: usize = 0,
        pub fn next(self: *@This(), now: i64) Error!?JobMeta {
            if (self.i < self.items.len and self.items[self.i].arrival <= now) {
                defer self.i += 1;
                return self.items[self.i];
            }
            return null;
        }
        pub fn done(self: *@This()) bool {
            return self.i == self.items.len;
        }
        pub fn reset(self: *@This(), seed: u64) void {
            _ = seed;
            self.i = 0;
        }
    };

    var impl = Fixed{ .items = &.{
        .{ .id = 0, .arrival = 0 },
        .{ .id = 1, .arrival = 0 },
        .{ .id = 2, .arrival = 3 },
    } };
    const src = ArrivalSource.init(&impl);

    var n: usize = 0;
    while (try src.next(0)) |_| n += 1;
    try std.testing.expectEqual(@as(usize, 2), n);
    try std.testing.expect((try src.next(2)) == null);
    try std.testing.expectEqual(@as(u32, 2), (try src.next(3)).?.id);
    try std.testing.expect(src.done());
    src.reset(0);
    try std.testing.expect(!src.done());
}
