//! Best fit: over all (job, machine) pairs that fit now, choose the pair
//! that leaves the least free capacity at tick 0 (normalized per resource),
//! packing machines tightly. Ties: oldest job.
//!
//! Scores are cached across decisions (`PairCache`): after a placement only
//! the chosen machine's scores are recomputed.
const std = @import("std");
const model = @import("../../model.zig");
const PairCache = @import("pair_cache.zig").PairCache;

pub const BestFit = struct {
    pub const name = "best_fit";

    cache: PairCache = .{},

    pub fn deinit(self: *BestFit) void {
        self.cache.deinit();
    }

    /// Negated free capacity left after placing (higher = tighter).
    fn score(_: void, st: *const model.State, s: usize, m: usize) f32 {
        const mach = st.machines.get(m);
        const job = st.jobs.get(s);
        var left: f32 = 0;
        for (0..st.machines.layout.n_res) |r| {
            const cap: f32 = @floatFromInt(@max(1, mach.cap[r]));
            const after: f32 = @floatFromInt(mach.freeNow(r) - job.row(r)[0]);
            left += after / cap;
        }
        return -left;
    }

    pub fn decide(self: *BestFit, st: *const model.State) model.Action {
        self.cache.update(st, score, {}) catch @panic("best_fit: out of memory");
        return self.cache.pick(st, PairCache.noPen, {});
    }
};
