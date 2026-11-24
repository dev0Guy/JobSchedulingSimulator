//! FIFO with backfill: place the oldest waiting job that fits now on any
//! machine (first machine that fits). If the oldest does not fit, younger
//! jobs may go first. Waits when nothing fits.
const model = @import("../../model.zig");
const rules = @import("../../engine.zig").rules;

pub const FifoBackfill = struct {
    pub const name = "fifo";

    pub fn decide(self: *FifoBackfill, st: *const model.State) model.Action {
        _ = self;
        var best: ?usize = null;
        var best_m: usize = 0;
        var it = st.jobs.iter();
        while (it.next()) |s| {
            if (best) |b| if (st.jobs.arrival[s] >= st.jobs.arrival[b]) continue;
            const m = rules.firstMachine(st, s) orelse continue;
            best = s;
            best_m = m;
        }
        const s = best orelse return .wait;
        return .{ .place = .{ .slot = @intCast(s), .machine = @intCast(best_m) } };
    }
};
