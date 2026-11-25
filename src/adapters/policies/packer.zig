//! Packer, the packing baseline of DeepRM (Mao et al., HotNets'16; the
//! `get_packer_action` agent of the reference code): among the (job,
//! machine) pairs that fit now, pick the one with the largest dot product of
//! the machine's free capacity at the current tick and the job's demand
//! vector:
//!
//!     score = Σ_r free[r][now] · peak[r]
//!
//! Unlike `tetris` there is no normalization by capacity and only the current
//! tick of the machine is looked at (the fit check still covers the job's
//! whole duration). The demand vector is the job's per-resource peak, which
//! is DeepRM's constant `res_vec` for constant-demand jobs. Ties: first
//! machine, then oldest job (DeepRM: lowest slot index). With one machine
//! this is exactly DeepRM's Packer.
//!
//! The dot product is summed exactly in integers; it is stored as f32 by
//! `PairCache`, which is exact up to 2^24 (e.g. capacities up to ~2000 on
//! 2 resources), so ties are real ties for realistic capacities.
//!
//! Scores are cached across decisions (`PairCache`).
const model = @import("../../model.zig");
const PairCache = @import("pair_cache.zig").PairCache;

fn score(_: void, st: *const model.State, s: usize, m: usize) f32 {
    const peak = st.jobs.peakOf(s);
    const mach = st.machines.get(m);
    var acc: u64 = 0;
    for (0..st.machines.layout.n_res) |r| {
        acc += @as(u64, mach.resource(r)[0]) * @as(u64, peak[r]);
    }
    return @floatFromInt(acc);
}

pub const Packer = struct {
    pub const name = "packer";

    cache: PairCache = .{},

    pub fn deinit(self: *Packer) void {
        self.cache.deinit();
    }

    pub fn decide(self: *Packer, st: *const model.State) model.Action {
        self.cache.update(st, score, {}) catch @panic("packer: out of memory");
        return self.cache.pick(st, PairCache.noPen, {});
    }
};
