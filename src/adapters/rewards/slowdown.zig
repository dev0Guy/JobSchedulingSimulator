//! DeepRM reward (Mao et al., HotNets'16): −Σ 1/T_j per tick over every job
//! in the system (waiting or running), T_j = the job's duration.
//!
//! Summed over an episode, job j contributes −(wait_j + T_j)/T_j, i.e. minus
//! its slowdown, so maximizing the return minimizes the total (and mean)
//! slowdown: short jobs that wait are penalized much more than long ones.
//!
//! `min_dur` bounds the denominator (bounded slowdown, common in HPC work):
//! 1/max(T_j, min_dur). Default 1 = plain DeepRM.
const ports = @import("../../ports.zig");
const Context = ports.reward.Context;

pub const Slowdown = struct {
    min_dur: u16 = 1,

    pub fn tick(self: *Slowdown, ctx: *const Context) f32 {
        const lo: f32 = @floatFromInt(@max(1, self.min_dur));
        const jobs = ctx.state.jobs;
        var s: f32 = 0;
        var it = jobs.iter();
        while (it.next()) |slot| s += 1.0 / @max(lo, @as(f32, @floatFromInt(jobs.duration[slot])));
        for (ctx.running) |r| s += 1.0 / @max(lo, @as(f32, @floatFromInt(r.dur)));
        return -s;
    }
};
