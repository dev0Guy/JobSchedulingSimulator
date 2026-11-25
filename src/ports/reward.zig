//! Reward port: turns one elapsed tick into a scalar reward (for RL and for
//! comparing policies on a common objective).
//!
//! Contract for an implementation type `T`:
//!     pub fn tick(self: *T, ctx: *const Context) f32
//!         Reward for the tick that is ending. Called once per elapsed tick,
//!         before time advances: `ctx.state.jobs` are the jobs still waiting
//!         and `ctx.running` the jobs running during that tick.
//! optional:
//!     pub fn reset(self: *T, seed: u64) void
//!     pub fn deinit(self: *T) void
//!
//! Rules:
//!   * Read only; deterministic.
//!   * Placements pay nothing; illegal actions are penalized by the engine
//!     (Config.illegal_penalty), not by this port.
//!   * The sum over an episode is what an RL agent maximizes, so a reward of
//!     −(something per tick) minimizes the time-integral of that something.
//!
//! With no Reward set, the engine uses −(waiting + running) per tick.

const std = @import("std");
const types = @import("../model/types.zig");
pub const State = types.State;
pub const RunningJob = types.RunningJob;

pub const Context = struct {
    state: *const State,
    running: []const RunningJob,
};

pub const Reward = struct {
    ptr: *anyopaque,
    vtable: *const VTable,

    pub const VTable = struct {
        tick: *const fn (ptr: *anyopaque, ctx: *const Context) f32,
        reset: *const fn (ptr: *anyopaque, seed: u64) void,
        deinit: *const fn (ptr: *anyopaque) void,
    };

    pub fn tick(self: Reward, ctx: *const Context) f32 {
        return self.vtable.tick(self.ptr, ctx);
    }
    pub fn reset(self: Reward, seed: u64) void {
        self.vtable.reset(self.ptr, seed);
    }
    pub fn deinit(self: Reward) void {
        self.vtable.deinit(self.ptr);
    }

    /// Wrap `impl: *T` as a Reward.
    pub fn init(impl: anytype) Reward {
        const Ptr = @TypeOf(impl);
        const T = @typeInfo(Ptr).pointer.child;
        if (!@hasDecl(T, "tick"))
            @compileError(@typeName(T) ++ " is not a Reward: missing `pub fn tick(self: *T, ctx: *const Context) f32`");
        const gen = struct {
            fn cast(p: *anyopaque) Ptr {
                return @ptrCast(@alignCast(p));
            }
            fn tick(p: *anyopaque, ctx: *const Context) f32 {
                return cast(p).tick(ctx);
            }
            fn reset(p: *anyopaque, seed: u64) void {
                if (@hasDecl(T, "reset")) cast(p).reset(seed);
            }
            fn deinit(p: *anyopaque) void {
                if (@hasDecl(T, "deinit")) cast(p).deinit();
            }
        };
        const vt = VTable{ .tick = gen.tick, .reset = gen.reset, .deinit = gen.deinit };
        return .{ .ptr = impl, .vtable = &vt };
    }
};

test "reward port: wrapper calls through" {
    const Const = struct {
        v: f32,
        resets: usize = 0,
        pub fn tick(self: *@This(), ctx: *const Context) f32 {
            return self.v - @as(f32, @floatFromInt(ctx.running.len));
        }
        pub fn reset(self: *@This(), seed: u64) void {
            _ = seed;
            self.resets += 1;
        }
    };
    var impl = Const{ .v = 2 };
    const r = Reward.init(&impl);
    const st: State = undefined; // not read
    const run = [_]RunningJob{.{ .id = 0, .machine = 0, .start = 0, .dur = 1 }};
    try std.testing.expectEqual(@as(f32, 1), r.tick(&.{ .state = &st, .running = &run }));
    r.reset(3);
    r.deinit(); // no deinit decl → no-op
    try std.testing.expectEqual(@as(usize, 1), impl.resets);
}
