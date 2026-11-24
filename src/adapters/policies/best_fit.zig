//! Best fit: over all (job, machine) pairs that fit now, choose the pair
//! that leaves the least free capacity at tick 0 (normalized per resource),
//! packing machines tightly. Ties: oldest job.
const model = @import("../../model.zig");
const rules = @import("../../engine.zig").rules;

pub const BestFit = struct {
    pub const name = "best_fit";

    pub fn decide(self: *BestFit, st: *const model.State) model.Action {
        _ = self;
        const j = st.jobs;
        var best: ?struct { s: usize, m: usize, score: f32 } = null;
        var it = j.iter();
        while (it.next()) |s| {
            if (rules.knownBlocked(st, s)) continue;
            var fits_any = false;
            for (0..st.machines.n) |m| {
                if (!rules.fitsNow(st, s, m)) continue;
                fits_any = true;
                const mach = st.machines.get(m);
                var left: f32 = 0;
                for (0..st.machines.layout.n_res) |r| {
                    const cap: f32 = @floatFromInt(@max(1, mach.cap[r]));
                    const after: f32 = @floatFromInt(mach.freeNow(r) - j.get(s).row(r)[0]);
                    left += after / cap;
                }
                if (best) |b| {
                    if (left > b.score) continue;
                    if (left == b.score and j.arrival[s] >= j.arrival[b.s]) continue;
                }
                best = .{ .s = s, .m = m, .score = left };
            }
            if (!fits_any) rules.markBlocked(st, s);
        }
        const b = best orelse return .wait;
        return .{ .place = .{ .slot = @intCast(b.s), .machine = @intCast(b.m) } };
    }
};
