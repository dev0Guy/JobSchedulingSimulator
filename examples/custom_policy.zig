//! Example: write your own policy and run it against a built-in one on the
//! same workload. A policy is any type with
//!     pub fn decide(self: *Self, state: *const sched.State) sched.Action
const std = @import("std");
const sched = @import("sched");
const A = sched.adapters;
const rules = sched.engine.rules;

/// "Largest first": among jobs that fit now, the one with the biggest total
/// peak demand (pack big jobs early, fill gaps with small ones later).
const LargestFirst = struct {
    pub fn decide(self: *LargestFirst, st: *const sched.State) sched.Action {
        _ = self;
        var best: ?usize = null;
        var best_m: usize = 0;
        var best_size: u32 = 0;
        var it = st.jobs.iter();
        while (it.next()) |s| {
            var size: u32 = 0;
            for (st.jobs.peakOf(s)) |p| size += p;
            if (best != null and size <= best_size) continue;
            const m = rules.firstMachine(st, s) orelse continue;
            best = s;
            best_m = m;
            best_size = size;
        }
        const s = best orelse return .wait;
        return .{ .place = .{ .slot = @intCast(s), .machine = @intCast(best_m) } };
    }
};

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;

    var arr = A.arrivals.Poisson{ .total = 1000, .rate = 0.7 };
    var gen = A.generators.DeepRM{ .n_machines = 2 };
    var eng = try sched.Engine.init(gpa, .{ .n_res = 10, .n_ticks = 30 }, .init(&arr), .init(&gen));
    defer eng.deinit();

    try eng.reset(1);
    var mine = LargestFirst{};
    const a = try eng.run(&mine, 10_000_000);

    try eng.reset(1);
    var fifo = A.policies.FifoBackfill{};
    const b = try eng.run(&fifo, 10_000_000);

    std.debug.print("largest_first  makespan {d:>4}  avg wait {d:>6.2}\n", .{ @as(u64, @intCast(a.makespan)), a.avgWait() });
    std.debug.print("fifo           makespan {d:>4}  avg wait {d:>6.2}\n", .{ @as(u64, @intCast(b.makespan)), b.avgWait() });
}
