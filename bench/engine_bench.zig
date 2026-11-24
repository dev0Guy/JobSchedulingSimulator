//! Engine benchmark: full episodes with each policy, no observers.
//! Reports episodes/s, steps/s and placements/s.
const std = @import("std");
const sched = @import("sched");
const A = sched.adapters;

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;

    std.debug.print(
        "engine bench   L = {d}   Elem = {s}   (500 jobs, 4 machines, 10 res, 100 ticks, best of 5)\n\n",
        .{ sched.grid.simd.L, @typeName(sched.Elem) },
    );
    std.debug.print("{s:>12} | {s:>10} | {s:>12} | {s:>14} | {s:>9} | {s:>8}\n", .{ "policy", "µs/episode", "steps/s", "placements/s", "makespan", "avg wait" });
    std.debug.print("{s:->12}-+-{s:->10}-+-{s:->12}-+-{s:->14}-+-{s:->9}-+-{s:->8}\n", .{ "", "", "", "", "", "" });

    inline for (.{ A.policies.FifoBackfill, A.policies.FifoStrict, A.policies.ShortestFirst, A.policies.BestFit, A.policies.RandomPolicy }) |P| {
        var arr = A.arrivals.Poisson{ .total = 500, .rate = 1.0 };
        var gen = A.generators.RandomProfiles{};
        var eng = try sched.Engine.init(gpa, .{ .reward = .zero }, .init(&arr), .init(&gen));
        defer eng.deinit();

        var best: i64 = std.math.maxInt(i64);
        var stats: sched.Stats = undefined;
        for (0..5) |_| {
            try eng.reset(42);
            var p = P{};
            sched.ports.policy.resetIfAny(P, &p, 42);
            const t0 = std.Io.Timestamp.now(io, .awake);
            stats = try eng.run(&p, 100_000_000);
            best = @min(best, t0.durationTo(std.Io.Timestamp.now(io, .awake)).toMicroseconds());
        }
        const sec = @as(f64, @floatFromInt(@max(best, 1))) / 1e6;
        std.debug.print("{s:>12} | {d:>10} | {d:>12.0} | {d:>14.0} | {d:>9} | {d:>8.2}\n", .{
            P.name,
            @as(u64, @intCast(best)),
            @as(f64, @floatFromInt(stats.steps)) / sec,
            @as(f64, @floatFromInt(stats.placed)) / sec,
            @as(u64, @intCast(stats.makespan)),
            stats.avgWait(),
        });
    }
}
