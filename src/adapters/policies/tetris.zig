//! Tetris (Grandl et al., SIGCOMM'14): multi-resource packing by alignment.
//! Over all (job, machine) pairs that fit now, pick the one whose demand is
//! most aligned with the machine's free capacity:
//!
//!     a = Σ_r Σ_{t<dur} demand[r][t] · free[r][t] / cap_r²
//!
//! i.e. the dot product of the job's and the machine's normalized
//! [resource × time] grids over the job's whole duration (the paper uses
//! one vector at the start; here the time-varying profile is used, so a job
//! is steered to machines that will have room for it later in its run too).
//! Large jobs on roomy machines score high, so big work is packed first and
//! small jobs fill the gaps. Ties: oldest job.
//!
//! Pure alignment favours large jobs and so hurts slowdown. `TetrisSrpt`
//! adds the paper's shortest-remaining-time term:
//!
//!     score = a − weight · ε · dur,   ε = mean(a) / mean(dur) over the candidates
//!
//! ε puts both terms on the same scale every decision (the paper's choice);
//! `weight` trades packing (0 = pure Tetris) against job completion time
//! (large = close to SJF).
//!
//! Alignments are cached across decisions (`PairCache`): after a placement
//! only the chosen machine's column is recomputed.
const std = @import("std");
const model = @import("../../model.zig");
const PairCache = @import("pair_cache.zig").PairCache;

/// Alignment of slot `s` with machine `m` (see above).
fn alignment(_: void, st: *const model.State, s: usize, m: usize) f32 {
    const j = st.jobs;
    const dur = j.duration[s];
    const job = j.get(s);
    const mach = st.machines.get(m);
    var score: f32 = 0;
    for (0..st.machines.layout.n_res) |r| {
        const c: f32 = @floatFromInt(@max(1, mach.cap[r]));
        const free = mach.resource(r);
        const dem = job.row(r);
        var acc: f32 = 0;
        for (0..dur) |t| acc += @as(f32, @floatFromInt(dem[t])) * @as(f32, @floatFromInt(free[t]));
        score += acc / (c * c);
    }
    return score;
}

pub const Tetris = struct {
    pub const name = "tetris";

    cache: PairCache = .{},

    pub fn deinit(self: *Tetris) void {
        self.cache.deinit();
    }

    pub fn decide(self: *Tetris, st: *const model.State) model.Action {
        self.cache.update(st, alignment, {}) catch @panic("tetris: out of memory");
        return self.cache.pick(st, PairCache.noPen, {});
    }
};

pub const TetrisSrpt = struct {
    pub const name = "tetris_srpt";

    weight: f32 = 1.5,
    cache: PairCache = .{},

    pub fn deinit(self: *TetrisSrpt) void {
        self.cache.deinit();
    }

    fn pen(eps: f32, st: *const model.State, s: usize) f32 {
        return eps * @as(f32, @floatFromInt(st.jobs.duration[s]));
    }

    pub fn decide(self: *TetrisSrpt, st: *const model.State) model.Action {
        self.cache.update(st, alignment, {}) catch @panic("tetris_srpt: out of memory");
        // means of alignment and duration over the candidate pairs
        // (summed in slot, machine order: same rounding as a full scan)
        const j = st.jobs;
        var sum_a: f32 = 0;
        var sum_d: f32 = 0;
        var n: f32 = 0;
        var it = j.iter();
        while (it.next()) |s| {
            if (self.cache.bestOf(s) == null) continue;
            const d: f32 = @floatFromInt(j.duration[s]);
            for (self.cache.row(s)) |a| {
                if (std.math.isNan(a)) continue;
                sum_a += a;
                sum_d += d;
                n += 1;
            }
        }
        if (n == 0) return .wait;
        const eps = self.weight * sum_a / @max(sum_d, 1e-6);
        return self.cache.pick(st, pen, eps);
    }
};
