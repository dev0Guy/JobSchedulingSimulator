//! FIFO with backfill: place the oldest waiting job that fits now on any
//! machine (first machine that fits). If the oldest does not fit, younger
//! jobs may go first. Waits when nothing fits.
const model = @import("../../model.zig");
const rules = @import("../../engine.zig").rules;

pub const FifoBackfill = struct {
    pub const name = "fifo";

    pub fn decide(self: *FifoBackfill, st: *const model.State) model.Action {
        _ = self;
        // oldest first ((arrival, slot) order): the first job that fits anywhere
        var it = st.jobs.byArrival();
        while (it.next()) |s| {
            const m = rules.firstMachine(st, s) orelse continue;
            return .{ .place = .{ .slot = @intCast(s), .machine = @intCast(m) } };
        }
        return .wait;
    }
};
