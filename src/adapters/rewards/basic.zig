//! Simple per-tick rewards.
//!   neg_in_system : −(waiting + running)   minimizes mean time in system
//!   neg_waiting   : −waiting               minimizes mean waiting time
//!   zero          : 0                      heuristics / benchmarking
const ports = @import("../../ports.zig");
const Context = ports.reward.Context;

pub const NegInSystem = struct {
    pub fn tick(self: *NegInSystem, ctx: *const Context) f32 {
        _ = self;
        return -@as(f32, @floatFromInt(ctx.state.jobs.count + ctx.running.len));
    }
};

pub const NegWaiting = struct {
    pub fn tick(self: *NegWaiting, ctx: *const Context) f32 {
        _ = self;
        return -@as(f32, @floatFromInt(ctx.state.jobs.count));
    }
};

pub const Zero = struct {
    pub fn tick(self: *Zero, ctx: *const Context) f32 {
        _ = self;
        _ = ctx;
        return 0;
    }
};
