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
const model = @import("../../model.zig");
const rules = @import("../../engine.zig").rules;

/// Alignment of slot `s` with machine `m` (see above).
fn alignment(st: *const model.State, s: usize, m: usize) f32 {
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

/// Best (slot, machine) by `alignment − eps·dur`; eps = 0 → pure Tetris.
fn pick(st: *const model.State, eps: f32) model.Action {
    const j = st.jobs;
    var best: ?struct { s: usize, m: usize, score: f32 } = null;
    var it = j.iter();
    while (it.next()) |s| {
        const pen = eps * @as(f32, @floatFromInt(j.duration[s]));
        for (0..st.machines.n) |m| {
            if (!rules.fitsNow(st, s, m)) continue;
            const score = alignment(st, s, m) - pen;
            if (best) |b| {
                if (score < b.score) continue;
                if (score == b.score and j.arrival[s] >= j.arrival[b.s]) continue;
            }
            best = .{ .s = s, .m = m, .score = score };
        }
    }
    const b = best orelse return .wait;
    return .{ .place = .{ .slot = @intCast(b.s), .machine = @intCast(b.m) } };
}

pub const Tetris = struct {
    pub const name = "tetris";

    pub fn decide(self: *Tetris, st: *const model.State) model.Action {
        _ = self;
        return pick(st, 0);
    }
};

pub const TetrisSrpt = struct {
    pub const name = "tetris_srpt";

    weight: f32 = 1.5,

    pub fn decide(self: *TetrisSrpt, st: *const model.State) model.Action {
        // pass 1: means of alignment and duration over the candidate pairs
        const j = st.jobs;
        var sum_a: f32 = 0;
        var sum_d: f32 = 0;
        var n: f32 = 0;
        var it = j.iter();
        while (it.next()) |s| {
            for (0..st.machines.n) |m| {
                if (!rules.fitsNow(st, s, m)) continue;
                sum_a += alignment(st, s, m);
                sum_d += @floatFromInt(j.duration[s]);
                n += 1;
            }
        }
        if (n == 0) return .wait;
        const eps = self.weight * sum_a / @max(sum_d, 1e-6);
        // pass 2: best combined score
        return pick(st, eps);
    }
};
