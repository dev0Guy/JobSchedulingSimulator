//! Random: a uniformly random legal placement (reservoir sampling, no
//! allocation); waits only when nothing fits. Sanity baseline.
const std = @import("std");
const model = @import("../../model.zig");
const rules = @import("../../engine.zig").rules;

pub const RandomPolicy = struct {
    pub const name = "random";

    prng: std.Random.DefaultPrng = .init(0),

    pub fn reset(self: *RandomPolicy, seed: u64) void {
        self.prng = .init(seed ^ 0x9e3779b97f4a7c15);
    }

    pub fn decide(self: *RandomPolicy, st: *const model.State) model.Action {
        const r = self.prng.random();
        var seen: usize = 0;
        var pick: model.Action = .wait;
        var it = st.jobs.iter();
        while (it.next()) |s| {
            for (0..st.machines.n) |m| {
                if (!rules.fitsNow(st, s, m)) continue;
                seen += 1;
                if (r.uintLessThan(usize, seen) == 0)
                    pick = .{ .place = .{ .slot = @intCast(s), .machine = @intCast(m) } };
            }
        }
        return pick;
    }
};
