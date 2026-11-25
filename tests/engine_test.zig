//! Integration tests: whole episodes through the public API.
const std = @import("std");
const sched = @import("sched");
const A = sched.adapters;
const rules = sched.engine.rules;

const testing = std.testing;

fn makeEngine(arr: *A.arrivals.Poisson, gen: *A.generators.RandomProfiles, cfg: sched.Config) !sched.Engine {
    return sched.Engine.init(testing.allocator, cfg, .init(arr), .init(gen));
}

/// Checks invariants after every event.
const InvariantChecker = struct {
    failures: usize = 0,
    placed_ids: std.AutoHashMapUnmanaged(u32, void) = .empty,
    dup: usize = 0,

    pub fn onEvent(self: *InvariantChecker, ev: sched.Event, st: *const sched.State) void {
        // job pool: count matches bitmask, padding zero beyond duration
        var n: usize = 0;
        var it = st.jobs.iter();
        while (it.next()) |s| {
            n += 1;
            const u = st.jobs.get(s);
            for (0..st.jobs.layout.n_res) |r| {
                for (u.row(r)[st.jobs.duration[s]..]) |v| if (v != 0) {
                    self.failures += 1;
                };
            }
            if (st.jobs.arrival[s] > st.now) self.failures += 1;
        }
        if (n != st.jobs.count) self.failures += 1;
        // free capacity never above capacity (unsigned: never below 0 either)
        for (0..st.machines.n) |m| {
            const mach = st.machines.get(m);
            for (0..st.machines.layout.n_res) |r|
                for (mach.resource(r)) |v| if (v > mach.cap[r]) {
                    self.failures += 1;
                };
        }
        switch (ev) {
            .job_placed => |p| {
                const gop = self.placed_ids.getOrPut(testing.allocator, p.id) catch return;
                if (gop.found_existing) self.dup += 1;
            },
            else => {},
        }
    }
};

test "episode: every job placed once, invariants hold, all policies" {
    inline for (.{ A.policies.FifoBackfill, A.policies.FifoStrict, A.policies.ShortestFirst, A.policies.BestFit, A.policies.RandomPolicy }) |P| {
        var arr = A.arrivals.Poisson{ .total = 200, .rate = 0.8 };
        var gen = A.generators.RandomProfiles{ .n_machines = 3, .varying = true };
        var eng = try makeEngine(&arr, &gen, .{ .n_res = 4, .n_ticks = 40, .max_jobs = 64 });
        defer eng.deinit();
        var chk = InvariantChecker{};
        defer chk.placed_ids.deinit(testing.allocator);
        try eng.addObserver(.init(&chk));

        try eng.reset(7);
        var p = P{};
        defer sched.ports.policy.deinitIfAny(P, &p);
        const st = try eng.run(&p, 1_000_000);

        try testing.expect(eng.done());
        try testing.expectEqual(@as(usize, 0), chk.failures);
        try testing.expectEqual(@as(usize, 0), chk.dup);
        try testing.expectEqual(st.arrived, st.placed);
        try testing.expectEqual(@as(usize, 200), st.arrived + st.rejected);
        try testing.expect(eng.machines.allIdle());
    }
}

test "determinism: same seed, same result; different seed, different result" {
    var results: [3]sched.Stats = undefined;
    for ([_]u64{ 1, 1, 2 }, 0..) |seed, i| {
        var arr = A.arrivals.Poisson{ .total = 300, .rate = 0.6 };
        var gen = A.generators.RandomProfiles{};
        var eng = try makeEngine(&arr, &gen, .{ .n_res = 5, .n_ticks = 50 });
        defer eng.deinit();
        try eng.reset(seed);
        var p = A.policies.FifoBackfill{};
        results[i] = try eng.run(&p, 1_000_000);
    }
    try testing.expectEqual(results[0].makespan, results[1].makespan);
    try testing.expectEqual(results[0].total_wait, results[1].total_wait);
    try testing.expect(results[0].total_wait != results[2].total_wait or results[0].makespan != results[2].makespan);
}

test "illegal action: penalized, treated as wait" {
    var arr = A.arrivals.Poisson{ .total = 5, .rate = 0 };
    var gen = A.generators.RandomProfiles{ .n_machines = 1 };
    var eng = try makeEngine(&arr, &gen, .{ .n_res = 2, .n_ticks = 30, .max_jobs = 8 });
    defer eng.deinit();
    try eng.reset(3);

    const t0 = eng.now;
    const r = try eng.step(.{ .place = .{ .slot = 7, .machine = 0 } }); // empty slot
    try testing.expect(r.illegal);
    try testing.expect(r.reward <= -1.0);
    try testing.expectEqual(t0 + 1, eng.now);
}

test "action encoding and mask" {
    var arr = A.arrivals.Poisson{ .total = 10, .rate = 0 };
    var gen = A.generators.RandomProfiles{ .n_machines = 2 };
    var eng = try makeEngine(&arr, &gen, .{ .n_res = 3, .n_ticks = 30, .max_jobs = 16 });
    defer eng.deinit();
    try eng.reset(5);

    const st = eng.state();
    const n = rules.actionCount(&st);
    try testing.expectEqual(@as(usize, 16 * 2 + 1), n);
    const mask = try testing.allocator.alloc(bool, n);
    defer testing.allocator.free(mask);
    rules.actionMask(&st, mask);
    try testing.expect(mask[n - 1]); // wait always legal
    for (0..n) |i| {
        const a = rules.decode(&st, i);
        try testing.expectEqual(i, rules.encode(&st, a));
        try testing.expectEqual(mask[i], rules.isLegal(&st, a));
    }
}

test "truncation" {
    var arr = A.arrivals.Poisson{ .total = 1000, .rate = 0.1 };
    var gen = A.generators.RandomProfiles{};
    var eng = try makeEngine(&arr, &gen, .{ .n_res = 2, .n_ticks = 20, .max_ticks = 50 });
    defer eng.deinit();
    try eng.reset(1);
    var p = A.policies.FifoBackfill{};
    _ = try eng.run(&p, 1_000_000);
    try testing.expect(eng.done());
    try testing.expectEqual(@as(i64, 50), eng.now);
}
