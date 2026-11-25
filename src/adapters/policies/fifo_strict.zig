//! Strict FIFO: only the oldest waiting job may start. If it does not fit,
//! wait. No starvation, lower utilization.
const model = @import("../../model.zig");
const rules = @import("../../engine.zig").rules;

pub const FifoStrict = struct {
    pub const name = "fifo_strict";

    pub fn decide(self: *FifoStrict, st: *const model.State) model.Action {
        _ = self;
        const s = st.jobs.oldest() orelse return .wait;
        const m = rules.firstMachine(st, s) orelse return .wait;
        return .{ .place = .{ .slot = @intCast(s), .machine = @intCast(m) } };
    }
};
