//! Shortest job first: among jobs that fit now, the one with the smallest
//! duration (ties: oldest).
const model = @import("../../model.zig");
const rules = @import("../../engine.zig").rules;

pub const ShortestFirst = struct {
    pub const name = "sjf";

    pub fn decide(self: *ShortestFirst, st: *const model.State) model.Action {
        _ = self;
        const j = st.jobs;
        var best: ?usize = null;
        var best_m: usize = 0;
        var it = j.iter();
        while (it.next()) |s| {
            if (best) |b| {
                if (j.duration[s] > j.duration[b]) continue;
                if (j.duration[s] == j.duration[b] and j.arrival[s] >= j.arrival[b]) continue;
            }
            const m = rules.firstMachine(st, s) orelse continue;
            best = s;
            best_m = m;
        }
        const s = best orelse return .wait;
        return .{ .place = .{ .slot = @intCast(s), .machine = @intCast(best_m) } };
    }
};
