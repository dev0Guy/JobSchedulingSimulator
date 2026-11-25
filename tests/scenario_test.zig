//! Scenario tests: small hand-built workloads whose correct outcome is worked
//! out by hand (comments show the arithmetic), plus a property test that
//! compares the engine against a deliberately naive reference simulator.
//!
//! Reward convention checked here (default reward = neg_in_system): each
//! `.wait` pays −(waiting + running) for the tick that ends, counted before
//! time advances; `.place` pays 0.

const std = @import("std");
const sched = @import("sched");
const A = sched.adapters;
const rules = sched.engine.rules;
const Elem = sched.Elem;
const Action = sched.Action;
const testing = std.testing;

// ------------------------------------------------------------------ helpers

/// One scripted job: arrival, and its demand profile flat [n_res][dur].
const JobSpec = struct {
    arrival: i64 = 0,
    dur: u16,
    /// Flat [n_res][dur]; or n_res values for constant demand.
    demand: []const Elem,
};

/// UsageGenerator + ArrivalSource from a fixed script.
const Script = struct {
    caps: []const Elem, // flat [n_machines][n_res]
    n_res: usize,
    jobs: []const JobSpec,
    metas: [64]sched.JobMeta = undefined,
    fixed: A.arrivals.Fixed = .{ .items = &.{} },

    fn init(self: *Script) void {
        for (self.jobs, 0..) |j, i| self.metas[i] = .{ .id = @intCast(i), .arrival = j.arrival };
        self.fixed = .{ .items = self.metas[0..self.jobs.len] };
    }

    // UsageGenerator
    pub fn machineCount(self: *Script) usize {
        return self.caps.len / self.n_res;
    }
    pub fn machine(self: *Script, i: usize, cap: []Elem) sched.ports.usage_generator.Error!void {
        @memcpy(cap, self.caps[i * self.n_res ..][0..self.n_res]);
    }
    pub fn job(self: *Script, meta: sched.JobMeta, out: sched.Usage) sched.ports.usage_generator.Error!u16 {
        const j = self.jobs[meta.id];
        const n = @min(j.dur, out.layout.n_ticks); // never write past the horizon
        for (0..self.n_res) |r| {
            if (j.demand.len == self.n_res) {
                @memset(out.resourceMut(r)[0..n], j.demand[r]);
            } else {
                @memcpy(out.resourceMut(r)[0..n], j.demand[r * j.dur ..][0..n]);
            }
        }
        return j.dur;
    }
};

/// Records events for assertions.
const Recorder = struct {
    const Placed = struct { id: u32, machine: u32, tick: i64 };
    placed: std.ArrayList(Placed) = .empty,
    rejected: std.ArrayList(sched.model.RejectReason) = .empty,
    illegal: usize = 0,
    done: ?struct { now: i64, truncated: bool } = null,

    pub fn onEvent(self: *Recorder, ev: sched.Event, st: *const sched.State) void {
        _ = st;
        switch (ev) {
            .job_placed => |p| self.placed.append(testing.allocator, .{ .id = p.id, .machine = p.machine, .tick = p.tick }) catch {},
            .job_rejected => |r| self.rejected.append(testing.allocator, r.reason) catch {},
            .illegal_action => self.illegal += 1,
            .episode_done => |d| self.done = .{ .now = d.now, .truncated = d.truncated },
            else => {},
        }
    }
    fn deinit(self: *Recorder) void {
        self.placed.deinit(testing.allocator);
        self.rejected.deinit(testing.allocator);
    }
    fn tickOf(self: Recorder, id: u32) ?i64 {
        for (self.placed.items) |p| if (p.id == id) return p.tick;
        return null;
    }
    fn machineOf(self: Recorder, id: u32) ?u32 {
        for (self.placed.items) |p| if (p.id == id) return p.machine;
        return null;
    }
};

/// Engine + script + recorder, wired together.
const Sim = struct {
    script: Script,
    rec: Recorder = .{},
    eng: sched.Engine = undefined,

    fn start(self: *Sim, cfg: sched.Config) !void {
        self.script.init();
        self.eng = try sched.Engine.init(testing.allocator, cfg, .init(&self.script.fixed), .init(&self.script));
        try self.eng.addObserver(.init(&self.rec));
        try self.eng.reset(0);
    }
    fn deinit(self: *Sim) void {
        self.eng.deinit();
        self.rec.deinit();
    }
    fn slotOf(self: *Sim, id: u32) u32 {
        var it = self.eng.jobs.iter();
        while (it.next()) |s| if (self.eng.jobs.id[s] == id) return @intCast(s);
        @panic("job not waiting");
    }
    fn place(self: *Sim, id: u32, m: u32) !sched.engine.StepResult {
        return self.eng.step(.{ .place = .{ .slot = self.slotOf(id), .machine = m } });
    }
    fn wait(self: *Sim) !sched.engine.StepResult {
        return self.eng.step(.wait);
    }
    fn free(self: *Sim, m: usize, r: usize) []const Elem {
        return self.eng.machines.get(m).resource(r);
    }
};

fn cfg1(n_res: usize, n_ticks: usize) sched.Config {
    return .{ .n_res = n_res, .n_ticks = n_ticks, .max_jobs = 16 };
}

// ------------------------------------------------------------------ basic mechanics

test "scenario: one job — placement, aging, finish, reward, termination" {
    // cap 10/10; job0 needs 4 (res 0) and 1 (res 1) for 3 ticks.
    var sim = Sim{ .script = .{ .caps = &.{ 10, 10 }, .n_res = 2, .jobs = &.{
        .{ .dur = 3, .demand = &.{ 4, 1 } },
    } } };
    // skip_empty off: step through each tick (with it on, one wait would
    // fast-forward straight to the end, see the skip_empty scenario)
    try sim.start(.{ .n_res = 2, .n_ticks = 6, .max_jobs = 4, .skip_empty = false });
    defer sim.deinit();

    try testing.expectEqual(@as(usize, 1), sim.eng.jobs.count);
    const r0 = try sim.place(0, 0);
    try testing.expect(r0.placed and !r0.illegal);
    try testing.expectEqual(@as(f32, 0), r0.reward); // placing pays nothing
    try testing.expectEqualSlices(Elem, &.{ 6, 6, 6, 10, 10, 10 }, sim.free(0, 0));
    try testing.expectEqualSlices(Elem, &.{ 9, 9, 9, 10, 10, 10 }, sim.free(0, 1));

    // tick 0 ends: 1 running → −1; the profile slides left
    var r = try sim.wait();
    try testing.expectEqual(@as(f32, -1), r.reward);
    try testing.expectEqual(@as(i64, 1), sim.eng.now);
    try testing.expectEqualSlices(Elem, &.{ 6, 6, 10, 10, 10, 10 }, sim.free(0, 0));
    r = try sim.wait();
    try testing.expect(!r.terminated);
    r = try sim.wait(); // now = 3: job finished, nothing left → terminated
    try testing.expect(r.terminated and !r.truncated);
    try testing.expect(sim.eng.done());

    const s = sim.eng.stats;
    try testing.expectEqual(@as(i64, 3), s.makespan);
    try testing.expectEqual(@as(f64, -3), s.total_reward); // 3 ticks × −1
    try testing.expectEqual(@as(f64, 0), s.avgWait());
    try testing.expect(sim.eng.machines.allIdle());
    try testing.expectEqual(@as(i64, 3), sim.rec.done.?.now);
}

test "scenario: capacity blocks the second job until the first finishes" {
    // one machine cap 10; two jobs of demand 6 → cannot overlap.
    // job0 runs [0,4), job1 [4,6). waits 0 and 4.
    // reward: ticks 0..3 → −(1 waiting + 1 running) ×4 = −8; ticks 4,5 → −1 ×2 = −2.
    var sim = Sim{ .script = .{ .caps = &.{10}, .n_res = 1, .jobs = &.{
        .{ .dur = 4, .demand = &.{6} },
        .{ .dur = 2, .demand = &.{6} },
    } } };
    try sim.start(cfg1(1, 8));
    defer sim.deinit();
    var p = A.policies.FifoBackfill{};
    const s = try sim.eng.run(&p, 1000);

    try testing.expectEqual(@as(?i64, 0), sim.rec.tickOf(0));
    try testing.expectEqual(@as(?i64, 4), sim.rec.tickOf(1));
    try testing.expectEqual(@as(i64, 6), s.makespan);
    try testing.expectEqual(@as(i64, 4), s.max_wait);
    try testing.expectEqual(@as(f64, 2), s.avgWait());
    try testing.expectEqual(@as(f64, -10), s.total_reward);
}

test "scenario: a second machine removes the conflict" {
    var sim = Sim{ .script = .{ .caps = &.{ 10, 10 }, .n_res = 1, .jobs = &.{
        .{ .dur = 4, .demand = &.{6} },
        .{ .dur = 2, .demand = &.{6} },
    } } };
    try sim.start(cfg1(1, 8));
    defer sim.deinit();
    var p = A.policies.FifoBackfill{};
    const s = try sim.eng.run(&p, 1000);
    try testing.expectEqual(@as(?u32, 0), sim.rec.machineOf(0));
    try testing.expectEqual(@as(?u32, 1), sim.rec.machineOf(1));
    try testing.expectEqual(@as(i64, 0), s.max_wait);
    try testing.expectEqual(@as(i64, 4), s.makespan);
}

test "scenario: exact fit to the last unit is allowed, one more is not" {
    var sim = Sim{ .script = .{ .caps = &.{ 10, 7 }, .n_res = 2, .jobs = &.{
        .{ .dur = 2, .demand = &.{ 10, 7 } }, // exactly full
        .{ .dur = 1, .demand = &.{ 0, 1 } }, // 1 unit too many on res 1
    } } };
    try sim.start(cfg1(2, 4));
    defer sim.deinit();
    const st = sim.eng.state();
    try testing.expect(rules.fitsNow(&st, sim.slotOf(0), 0));
    _ = try sim.place(0, 0);
    const st2 = sim.eng.state();
    try testing.expect(!rules.fitsNow(&st2, sim.slotOf(1), 0));
    try testing.expectEqualSlices(Elem, &.{ 0, 0, 10, 10 }, sim.free(0, 0));
}

test "scenario: time-varying profiles — the whole window must fit, not just tick 0" {
    // job0 uses 8,8,2,2 → free 2,2,8,8. job1 needs 2,8: tick 0 fits (2 ≤ 2)
    // but tick 1 does not (8 > 2). One tick later free is 2,8,8,10 → fits.
    var sim = Sim{ .script = .{ .caps = &.{10}, .n_res = 1, .jobs = &.{
        .{ .dur = 4, .demand = &.{ 8, 8, 2, 2 } },
        .{ .dur = 2, .demand = &.{ 2, 8 } },
    } } };
    try sim.start(cfg1(1, 6));
    defer sim.deinit();
    _ = try sim.place(0, 0);
    try testing.expectEqualSlices(Elem, &.{ 2, 2, 8, 8, 10, 10 }, sim.free(0, 0));

    const m = sim.eng.machines.get(0);
    const s1 = sim.slotOf(1);
    try testing.expect(m.mayFit(sim.eng.jobs.get(s1), sim.eng.jobs.peakOf(s1), 0)); // cheap check passes
    const st = sim.eng.state();
    try testing.expect(!rules.fitsNow(&st, s1, 0)); // full check rejects
    try testing.expectEqual(@as(?usize, 1), m.earliestFit(sim.eng.jobs.get(s1), 2));

    var p = A.policies.FifoBackfill{};
    _ = try sim.eng.run(&p, 1000);
    try testing.expectEqual(@as(?i64, 1), sim.rec.tickOf(1));
    try testing.expectEqual(@as(i64, 4), sim.eng.stats.makespan);
}

test "scenario: resources are independent — one full resource blocks" {
    // res 0 has room, res 1 is full → no fit.
    var sim = Sim{ .script = .{ .caps = &.{ 10, 10 }, .n_res = 2, .jobs = &.{
        .{ .dur = 3, .demand = &.{ 1, 10 } },
        .{ .dur = 1, .demand = &.{ 1, 1 } },
    } } };
    try sim.start(cfg1(2, 5));
    defer sim.deinit();
    var p = A.policies.FifoBackfill{};
    _ = try sim.eng.run(&p, 1000);
    try testing.expectEqual(@as(?i64, 3), sim.rec.tickOf(1));
}

test "scenario: a job as long as the horizon fits; finishing frees exactly at dur" {
    var sim = Sim{ .script = .{ .caps = &.{5}, .n_res = 1, .jobs = &.{
        .{ .dur = 5, .demand = &.{5} },
        .{ .dur = 5, .demand = &.{5} },
    } } };
    try sim.start(cfg1(1, 5));
    defer sim.deinit();
    var p = A.policies.FifoBackfill{};
    const s = try sim.eng.run(&p, 1000);
    try testing.expectEqual(@as(?i64, 5), sim.rec.tickOf(1));
    try testing.expectEqual(@as(i64, 10), s.makespan);
}

// ------------------------------------------------------------------ policies

test "scenario: backfill lets a small job jump ahead; strict FIFO does not" {
    // cap 10. job0 (t0, 7 × 5 ticks) runs; job1 (t0, 5 × 3) must wait until 5;
    // job2 (t1, 3 × 2) fits next to job0 (7+3 = 10).
    // backfill: job2 starts at 1 → waits 0,5,0.  strict: job2 waits behind job1,
    // both start at 5 (5+3 = 8) → waits 0,5,4.
    const jobs = [_]JobSpec{
        .{ .dur = 5, .demand = &.{7} },
        .{ .dur = 3, .demand = &.{5} },
        .{ .arrival = 1, .dur = 2, .demand = &.{3} },
    };
    var bf = Sim{ .script = .{ .caps = &.{10}, .n_res = 1, .jobs = &jobs } };
    try bf.start(cfg1(1, 8));
    defer bf.deinit();
    var p1 = A.policies.FifoBackfill{};
    const a = try bf.eng.run(&p1, 1000);
    try testing.expectEqual(@as(?i64, 1), bf.rec.tickOf(2));
    try testing.expectEqual(@as(i64, 5), a.total_wait);

    var st = Sim{ .script = .{ .caps = &.{10}, .n_res = 1, .jobs = &jobs } };
    try st.start(cfg1(1, 8));
    defer st.deinit();
    var p2 = A.policies.FifoStrict{};
    const b = try st.eng.run(&p2, 1000);
    try testing.expectEqual(@as(?i64, 5), st.rec.tickOf(2));
    try testing.expectEqual(@as(i64, 9), b.total_wait);
    try testing.expectEqual(a.makespan, b.makespan); // both end at 8
    try testing.expectEqual(@as(i64, 8), b.makespan);
}

test "scenario: SJF orders by duration, FIFO by arrival" {
    // three exclusive jobs (demand = cap) at t0, durations 3,1,2.
    // SJF: 1 [0,1), 2 [1,3), 0 [3,6) → waits 0+1+3 = 4.
    // FIFO: 0 [0,3), 1 [3,4), 2 [4,6) → waits 0+3+4 = 7.
    const jobs = [_]JobSpec{
        .{ .dur = 3, .demand = &.{10} },
        .{ .dur = 1, .demand = &.{10} },
        .{ .dur = 2, .demand = &.{10} },
    };
    var a = Sim{ .script = .{ .caps = &.{10}, .n_res = 1, .jobs = &jobs } };
    try a.start(cfg1(1, 8));
    defer a.deinit();
    var sjf = A.policies.ShortestFirst{};
    const sa = try a.eng.run(&sjf, 1000);
    try testing.expectEqual(@as(i64, 4), sa.total_wait);
    try testing.expectEqual(@as(?i64, 3), a.rec.tickOf(0));

    var b = Sim{ .script = .{ .caps = &.{10}, .n_res = 1, .jobs = &jobs } };
    try b.start(cfg1(1, 8));
    defer b.deinit();
    var fifo = A.policies.FifoBackfill{};
    const sb = try b.eng.run(&fifo, 1000);
    try testing.expectEqual(@as(i64, 7), sb.total_wait);
    try testing.expectEqual(sa.makespan, sb.makespan);
}

test "scenario: best_fit packs the tighter machine, FIFO takes the first" {
    // machines cap 10 and 5; job demand 5 → leaves 0.5 vs 0.0 normalized.
    const jobs = [_]JobSpec{.{ .dur = 1, .demand = &.{5} }};
    var a = Sim{ .script = .{ .caps = &.{ 10, 5 }, .n_res = 1, .jobs = &jobs } };
    try a.start(cfg1(1, 4));
    defer a.deinit();
    var bf = A.policies.BestFit{};
    defer bf.deinit();
    _ = try a.eng.run(&bf, 100);
    try testing.expectEqual(@as(?u32, 1), a.rec.machineOf(0));

    var b = Sim{ .script = .{ .caps = &.{ 10, 5 }, .n_res = 1, .jobs = &jobs } };
    try b.start(cfg1(1, 4));
    defer b.deinit();
    var ff = A.policies.FifoBackfill{};
    _ = try b.eng.run(&ff, 100);
    try testing.expectEqual(@as(?u32, 0), b.rec.machineOf(0));
}

test "scenario: slowdown reward sums to minus the total slowdown" {
    // jobs (dur 1, 4, 2) exclusive on one machine, SJF order 0,2,1:
    // starts 0,1,3 → slowdowns (0+1)/1, (3+4)/4, (1+2)/2 = 1 + 1.75 + 1.5 = 4.25
    const jobs = [_]JobSpec{
        .{ .dur = 1, .demand = &.{10} },
        .{ .dur = 4, .demand = &.{10} },
        .{ .dur = 2, .demand = &.{10} },
    };
    var sim = Sim{ .script = .{ .caps = &.{10}, .n_res = 1, .jobs = &jobs } };
    try sim.start(cfg1(1, 8));
    defer sim.deinit();
    var slow = A.rewards.Slowdown{};
    sim.eng.setReward(.init(&slow));
    var p = A.policies.ShortestFirst{};
    const s = try sim.eng.run(&p, 100);
    try testing.expectApproxEqAbs(@as(f64, -4.25), s.total_reward, 1e-5);

    // bounded slowdown: denominators below min_dur are raised to it
    var bsim = Sim{ .script = .{ .caps = &.{10}, .n_res = 1, .jobs = &jobs } };
    try bsim.start(cfg1(1, 8));
    defer bsim.deinit();
    var bounded = A.rewards.Slowdown{ .min_dur = 4 };
    bsim.eng.setReward(.init(&bounded));
    const b = try bsim.eng.run(&p, 100);
    // (0+1)/4 + (3+4)/4 + (1+2)/4 = 11/4
    try testing.expectApproxEqAbs(@as(f64, -2.75), b.total_reward, 1e-5);
}

test "scenario: tetris aligns job shape with free capacity, best_fit packs tick 0" {
    // two resources. m0 has room on res 0, m1 on res 1 (both 10 at tick 0
    // after the blockers). job2 is res-1 heavy → tetris sends it to m1.
    //   m0 cap (10, 10), blocker uses (2, 8)  → free (8, 2)
    //   m1 cap (10, 10), blocker uses (8, 2)  → free (2, 8)
    //   job2 demand (1, 2): fits both; alignment m0 = 1·8+2·2 = 12, m1 = 1·2+2·8 = 18
    var sim = Sim{ .script = .{ .caps = &.{ 10, 10, 10, 10 }, .n_res = 2, .jobs = &.{
        .{ .dur = 6, .demand = &.{ 2, 8 } },
        .{ .dur = 6, .demand = &.{ 8, 2 } },
        .{ .arrival = 1, .dur = 2, .demand = &.{ 1, 2 } },
    } } };
    try sim.start(cfg1(2, 8));
    defer sim.deinit();
    _ = try sim.place(0, 0);
    _ = try sim.place(1, 1);
    _ = try sim.wait(); // job2 arrives
    var t = A.policies.Tetris{};
    defer t.deinit();
    const st = sim.eng.state();
    const a = t.decide(&st);
    try testing.expectEqual(@as(u32, 1), a.place.machine);
    // best_fit sees a tie at tick 0 (both leave 0.7 of capacity) → first machine
    var bf0 = A.policies.BestFit{};
    defer bf0.deinit();
    try testing.expectEqual(@as(u32, 0), bf0.decide(&st).place.machine);

    // time matters: a job whose demand is high later is sent to the machine
    // that frees up later in its run. m0 free (2,2,10,10), m1 free (6,6,6,6)
    // on one resource; job needs (2,2,6,6): fits both, aligns with m0.
    var sim2 = Sim{ .script = .{ .caps = &.{ 10, 10 }, .n_res = 1, .jobs = &.{
        .{ .dur = 2, .demand = &.{8} },
        .{ .dur = 6, .demand = &.{4} },
        .{ .arrival = 0, .dur = 4, .demand = &.{ 2, 2, 6, 6 } },
    } } };
    try sim2.start(cfg1(1, 8));
    defer sim2.deinit();
    _ = try sim2.place(0, 0);
    _ = try sim2.place(1, 1);
    const st2 = sim2.eng.state();
    const a2 = t.decide(&st2);
    try testing.expectEqual(@as(u32, 0), a2.place.machine); // 2·2+2·2+6·10+6·10=128 vs 2·6·2+6·6·2=96
    var bf = A.policies.BestFit{};
    defer bf.deinit();
    try testing.expectEqual(@as(u32, 0), bf.decide(&st2).place.machine); // tick 0: m0 leaves 0 vs m1 4
}

test "scenario: tetris_srpt trades packing for short jobs via its weight" {
    // empty machine cap 10. big job A (8 × 4 ticks) vs small job B (2 × 1).
    // alignment a = demand·free/cap² summed over time: A 3.2, B 0.2.
    // ε = weight · mean(a)/mean(dur) = weight · 1.7/2.5.
    //   weight 0   → scores 3.2, 0.2                 → A (pure tetris)
    //   weight 1.5 → 3.2 − 1.02·4 = −0.88, 0.2 − 1.02 = −0.82 → B
    const jobs = [_]JobSpec{
        .{ .dur = 4, .demand = &.{8} },
        .{ .dur = 1, .demand = &.{2} },
    };
    var sim = Sim{ .script = .{ .caps = &.{10}, .n_res = 1, .jobs = &jobs } };
    try sim.start(cfg1(1, 8));
    defer sim.deinit();
    const st = sim.eng.state();
    const big = sim.slotOf(0);
    const small = sim.slotOf(1);

    var pure = A.policies.Tetris{};
    defer pure.deinit();
    try testing.expectEqual(big, pure.decide(&st).place.slot);
    var w0 = A.policies.TetrisSrpt{ .weight = 0 };
    defer w0.deinit();
    try testing.expectEqual(big, w0.decide(&st).place.slot);
    var w = A.policies.TetrisSrpt{};
    defer w.deinit();
    try testing.expectEqual(small, w.decide(&st).place.slot);
    // nothing fits → wait
    _ = try sim.place(0, 0);
    _ = try sim.place(1, 0);
    const st2 = sim.eng.state();
    try testing.expect(w.decide(&st2) == .wait);
}

test "scenario: packer scores free-now x demand, unnormalized (DeepRM)" {
    // m0 cap (20, 20), m1 cap (10, 10), both empty. job (2, 2) x 2 ticks.
    //   packer: m0 = 20·2 + 20·2 = 80, m1 = 10·2 + 10·2 = 40 → m0
    //   tetris (normalized by cap²): m0 = 2·(2·20·2)/400 = 0.4, m1 = 2·(2·10·2)/100 = 0.8 → m1
    var sim = Sim{ .script = .{ .caps = &.{ 20, 20, 10, 10 }, .n_res = 2, .jobs = &.{
        .{ .dur = 2, .demand = &.{ 2, 2 } },
    } } };
    try sim.start(cfg1(2, 8));
    defer sim.deinit();
    const st = sim.eng.state();
    var p = A.policies.Packer{};
    defer p.deinit();
    try testing.expectEqual(@as(u32, 0), p.decide(&st).place.machine);
    var t = A.policies.Tetris{};
    defer t.deinit();
    try testing.expectEqual(@as(u32, 1), t.decide(&st).place.machine);

    // job choice, one machine cap (10, 10): A (6, 1) → 10·6 + 10·1 = 70,
    // B (3, 3) → 60, C (1, 1) → 20: A first; then free (4, 9): B (3,3) → 39,
    // C (1,1) → 13: B; then free (1, 6): C → 7; then nothing waits.
    var sim2 = Sim{ .script = .{ .caps = &.{ 10, 10 }, .n_res = 2, .jobs = &.{
        .{ .dur = 3, .demand = &.{ 3, 3 } },
        .{ .dur = 3, .demand = &.{ 1, 1 } },
        .{ .dur = 3, .demand = &.{ 6, 1 } },
    } } };
    try sim2.start(cfg1(2, 8));
    defer sim2.deinit();
    var p2 = A.policies.Packer{};
    defer p2.deinit();
    for ([_]u32{ 2, 0, 1 }) |want| {
        const s2 = sim2.eng.state();
        const a = p2.decide(&s2);
        try testing.expectEqual(sim2.slotOf(want), a.place.slot);
        _ = try sim2.place(want, 0);
    }
    const s3 = sim2.eng.state();
    try testing.expect(p2.decide(&s3) == .wait);
}

test "scenario: several placements in one tick, time does not move" {
    var sim = Sim{ .script = .{ .caps = &.{10}, .n_res = 1, .jobs = &.{
        .{ .dur = 2, .demand = &.{3} },
        .{ .dur = 2, .demand = &.{3} },
        .{ .dur = 2, .demand = &.{3} },
    } } };
    try sim.start(cfg1(1, 4));
    defer sim.deinit();
    for (0..3) |i| _ = try sim.place(@intCast(i), 0);
    try testing.expectEqual(@as(i64, 0), sim.eng.now);
    try testing.expectEqualSlices(Elem, &.{ 1, 1, 10, 10 }, sim.free(0, 0));
}

// ------------------------------------------------------------------ illegal actions

test "scenario: illegal actions are penalized, counted, and act as wait" {
    // job0 fills the machine; job1 cannot fit.
    var sim = Sim{ .script = .{ .caps = &.{10}, .n_res = 1, .jobs = &.{
        .{ .dur = 5, .demand = &.{10} },
        .{ .dur = 1, .demand = &.{1} },
    } } };
    try sim.start(cfg1(1, 8));
    defer sim.deinit();
    _ = try sim.place(0, 0);

    // does not fit → penalty −1 plus the tick: 1 waiting + 1 running = −2 → −3
    var r = try sim.place(1, 0);
    try testing.expect(r.illegal and !r.placed);
    try testing.expectEqual(@as(f32, -3), r.reward);
    try testing.expectEqual(@as(i64, 1), sim.eng.now);

    // empty slot / machine out of range
    r = try sim.eng.step(.{ .place = .{ .slot = 15, .machine = 0 } });
    try testing.expect(r.illegal);
    r = try sim.eng.step(.{ .place = .{ .slot = sim.slotOf(1), .machine = 7 } });
    try testing.expect(r.illegal);
    // slot index beyond the pool
    r = try sim.eng.step(.{ .place = .{ .slot = 999, .machine = 0 } });
    try testing.expect(r.illegal);

    try testing.expectEqual(@as(usize, 4), sim.eng.stats.illegal);
    try testing.expectEqual(@as(usize, 4), sim.rec.illegal);
    try testing.expectEqual(@as(i64, 4), sim.eng.now);
    try testing.expectEqual(@as(usize, 1), sim.eng.jobs.count); // job1 still waiting, untouched
    try testing.expectEqual(@as(usize, 1), sim.eng.stats.placed);
}

test "scenario: action mask agrees with legality for every action" {
    var sim = Sim{ .script = .{ .caps = &.{ 10, 4, 6, 6 }, .n_res = 2, .jobs = &.{
        .{ .dur = 2, .demand = &.{ 5, 5 } }, // fits m0 only
        .{ .dur = 2, .demand = &.{ 3, 3 } }, // fits both
        .{ .dur = 2, .demand = &.{ 11, 0 } }, // impossible → dropped
    } } };
    try sim.start(cfg1(2, 4));
    defer sim.deinit();
    const st = sim.eng.state();
    const n = rules.actionCount(&st);
    const mask = try testing.allocator.alloc(bool, n);
    defer testing.allocator.free(mask);
    rules.actionMask(&st, mask);
    var legal: usize = 0;
    for (0..n) |i| {
        const act = rules.decode(&st, i);
        try testing.expectEqual(rules.isLegal(&st, act), mask[i]);
        try testing.expectEqual(i, rules.encode(&st, act));
        if (mask[i]) legal += 1;
    }
    try testing.expectEqual(@as(usize, 1 + 2 + 1), legal); // job0→m0, job1→m0/m1, wait
}

// ------------------------------------------------------------------ admission

test "scenario: full pool rejects, impossible and invalid jobs are dropped" {
    var sim = Sim{ .script = .{ .caps = &.{10}, .n_res = 1, .jobs = &.{
        // the pool check comes first, so the bad jobs arrive while slots are free
        .{ .dur = 1, .demand = &.{11} }, // impossible on every machine
        .{ .dur = 9, .demand = &.{1} }, // longer than the horizon (4)
        .{ .dur = 0, .demand = &.{1} }, // zero duration
        .{ .dur = 1, .demand = &.{1} },
        .{ .dur = 1, .demand = &.{1} },
        .{ .dur = 1, .demand = &.{1} }, // pool (2 slots) now full
    } } };
    try sim.start(.{ .n_res = 1, .n_ticks = 4, .max_jobs = 2 });
    defer sim.deinit();

    try testing.expectEqual(@as(usize, 2), sim.eng.jobs.count);
    try testing.expectEqual(@as(usize, 2), sim.eng.stats.arrived);
    // rejected slots were handed back: the two admitted jobs got slots 0 and 1
    try testing.expect(sim.eng.jobs.isValid(0) and sim.eng.jobs.isValid(1));
    try testing.expectEqual(@as(usize, 4), sim.eng.stats.rejected);
    const R = sched.model.RejectReason;
    try testing.expectEqualSlices(R, &.{ .impossible, .invalid_profile, .invalid_profile, .pool_full }, sim.rec.rejected.items);

    var p = A.policies.FifoBackfill{};
    const s = try sim.eng.run(&p, 100);
    try testing.expectEqual(@as(usize, 2), s.placed);
    try testing.expect(sim.eng.done());
}

test "scenario: impossible job kept (drop_impossible = false) waits until truncation" {
    var sim = Sim{ .script = .{ .caps = &.{10}, .n_res = 1, .jobs = &.{
        .{ .dur = 1, .demand = &.{11} },
    } } };
    try sim.start(.{ .n_res = 1, .n_ticks = 4, .max_jobs = 2, .drop_impossible = false, .max_ticks = 20 });
    defer sim.deinit();
    var p = A.policies.FifoBackfill{};
    const s = try sim.eng.run(&p, 1000);
    try testing.expect(sim.rec.done.?.truncated);
    try testing.expectEqual(@as(i64, 20), sim.eng.now);
    try testing.expectEqual(@as(usize, 0), s.placed);
    try testing.expectEqual(@as(f64, -20), s.total_reward); // 1 waiting × 20 ticks
}

// ------------------------------------------------------------------ time

test "scenario: skip_empty fast-forwards an idle gap in one step" {
    // job0 [0,2); job1 arrives at 50. After placing job0, one wait jumps to 50.
    // reward: ticks 0,1 → −1 each (job0 running); 2..49 → 0.
    const jobs = [_]JobSpec{
        .{ .dur = 2, .demand = &.{1} },
        .{ .arrival = 50, .dur = 1, .demand = &.{1} },
    };
    var a = Sim{ .script = .{ .caps = &.{10}, .n_res = 1, .jobs = &jobs } };
    try a.start(cfg1(1, 4));
    defer a.deinit();
    _ = try a.place(0, 0);
    const r = try a.wait();
    try testing.expectEqual(@as(i64, 50), a.eng.now);
    try testing.expectEqual(@as(f32, -2), r.reward);
    try testing.expectEqual(@as(usize, 1), a.eng.jobs.count);

    var b = Sim{ .script = .{ .caps = &.{10}, .n_res = 1, .jobs = &jobs } };
    try b.start(.{ .n_res = 1, .n_ticks = 4, .max_jobs = 4, .skip_empty = false });
    defer b.deinit();
    _ = try b.place(0, 0);
    _ = try b.wait();
    try testing.expectEqual(@as(i64, 1), b.eng.now);
    var p = A.policies.FifoBackfill{};
    const sb = try b.eng.run(&p, 1000);
    // same outcome either way
    try testing.expectEqual(@as(i64, 51), sb.makespan);
    try testing.expectEqual(@as(f64, -3), sb.total_reward); // −2 for job0, −1 for job1
}

test "scenario: reward kinds" {
    // two exclusive jobs dur 2 each at t0 on one machine:
    // ticks 0,1: 1 waiting + 1 running; ticks 2,3: 0 waiting + 1 running.
    const jobs = [_]JobSpec{ .{ .dur = 2, .demand = &.{10} }, .{ .dur = 2, .demand = &.{10} } };
    var in_sys = A.rewards.NegInSystem{};
    var waiting = A.rewards.NegWaiting{};
    var zero = A.rewards.Zero{};
    var slow = A.rewards.Slowdown{};
    const want = [_]struct { r: ?sched.ports.Reward, v: f64 }{
        .{ .r = null, .v = -6 }, // default = neg_in_system
        .{ .r = .init(&in_sys), .v = -6 }, // 2+2+1+1
        .{ .r = .init(&waiting), .v = -2 }, // 1+1
        .{ .r = .init(&zero), .v = 0 },
        // slowdown: −Σ 1/T per job per tick in system; job0 2 ticks, job1 4 ticks,
        // T = 2 → −(2 + 4)/2 = −3 = −(slowdown 1 + slowdown 2)
        .{ .r = .init(&slow), .v = -3 },
    };
    for (want) |w| {
        var sim = Sim{ .script = .{ .caps = &.{10}, .n_res = 1, .jobs = &jobs } };
        try sim.start(.{ .n_res = 1, .n_ticks = 4, .max_jobs = 4 });
        defer sim.deinit();
        sim.eng.setReward(w.r);
        var p = A.policies.FifoBackfill{};
        const s = try sim.eng.run(&p, 100);
        try testing.expectEqual(w.v, s.total_reward);
    }
}

test "scenario: reset replays the same episode" {
    const jobs = [_]JobSpec{
        .{ .dur = 3, .demand = &.{6} },
        .{ .arrival = 1, .dur = 2, .demand = &.{6} },
        .{ .arrival = 2, .dur = 1, .demand = &.{4} },
    };
    var sim = Sim{ .script = .{ .caps = &.{10}, .n_res = 1, .jobs = &jobs } };
    try sim.start(cfg1(1, 4));
    defer sim.deinit();
    var p = A.policies.FifoBackfill{};
    const a = try sim.eng.run(&p, 100);
    try sim.eng.reset(0);
    try testing.expect(sim.eng.machines.allIdle());
    try testing.expectEqual(@as(i64, 0), sim.eng.now);
    const b = try sim.eng.run(&p, 100);
    try testing.expectEqual(a.total_wait, b.total_wait);
    try testing.expectEqual(a.makespan, b.makespan);
    try testing.expectEqual(a.total_reward, b.total_reward);
}

// ------------------------------------------------------------------ reference model

/// Naive reference: keeps a list of running jobs and recomputes every
/// machine's free capacity from scratch, tick by tick. No SIMD, no padding,
/// no sliding window — just the definition.
const Reference = struct {
    const Run = struct { machine: usize, start: i64, spec: JobSpec };
    running: std.ArrayList(Run) = .empty,

    fn deinit(self: *Reference) void {
        self.running.deinit(testing.allocator);
    }

    fn demandAt(spec: JobSpec, n_res: usize, r: usize, k: usize) Elem {
        if (k >= spec.dur) return 0;
        return if (spec.demand.len == n_res) spec.demand[r] else spec.demand[r * spec.dur + k];
    }

    fn free(self: Reference, caps: []const Elem, n_res: usize, m: usize, r: usize, now: i64, t: usize) i64 {
        var f: i64 = caps[m * n_res + r];
        for (self.running.items) |run| {
            if (run.machine != m) continue;
            const k = now + @as(i64, @intCast(t)) - run.start;
            if (k < 0 or k >= run.spec.dur) continue;
            f -= demandAt(run.spec, n_res, r, @intCast(k));
        }
        return f;
    }

    fn fits(self: Reference, caps: []const Elem, n_res: usize, n_ticks: usize, m: usize, spec: JobSpec, now: i64) bool {
        if (spec.dur > n_ticks) return false;
        for (0..n_res) |r| for (0..spec.dur) |k| {
            if (demandAt(spec, n_res, r, k) > self.free(caps, n_res, m, r, now, k)) return false;
        };
        return true;
    }

    fn runningAt(self: Reference, now: i64) usize {
        var n: usize = 0;
        for (self.running.items) |run| {
            if (run.start <= now and now < run.start + run.spec.dur) n += 1;
        }
        return n;
    }
};

test "property: engine matches the naive reference on random workloads" {
    const n_res = 3;
    const n_ticks = 8; // small horizon → the machine window compacts very often
    const caps = [_]Elem{ 12, 9, 15, 6, 10, 8 }; // 2 machines

    var prng = std.Random.DefaultPrng.init(2024);
    const rnd = prng.random();

    for (0..20) |episode| {
        // random script: 60 jobs, random arrivals, time-varying profiles
        var specs: [60]JobSpec = undefined;
        var demand_store: [60][n_res * n_ticks]Elem = undefined;
        var t: i64 = 0;
        for (&specs, 0..) |*s, i| {
            t += rnd.intRangeAtMost(i64, 0, 2);
            const dur = rnd.intRangeAtMost(u16, 1, n_ticks);
            for (0..n_res * dur) |k| demand_store[i][k] = rnd.intRangeAtMost(Elem, 0, 7);
            s.* = .{ .arrival = t, .dur = dur, .demand = demand_store[i][0 .. n_res * dur] };
        }

        var sim = Sim{ .script = .{ .caps = &caps, .n_res = n_res, .jobs = &specs } };
        try sim.start(.{ .n_res = n_res, .n_ticks = n_ticks, .max_jobs = 32, .skip_empty = false, .drop_impossible = false });
        defer sim.deinit();
        var ref = Reference{};
        defer ref.deinit();

        var expected_reward: f64 = 0;
        var steps: usize = 0;
        while (!sim.eng.done()) : (steps += 1) {
            try testing.expect(steps < 100_000);
            const now = sim.eng.now;

            // 1. free capacity equals the reference everywhere in the horizon
            for (0..2) |m| for (0..n_res) |r| {
                const row = sim.free(m, r);
                for (0..n_ticks) |k| try testing.expectEqual(ref.free(&caps, n_res, m, r, now, k), @as(i64, row[k]));
            };

            // 2. legality equals the reference for every (slot, machine)
            const st = sim.eng.state();
            var it = sim.eng.jobs.iter();
            var options: [64]Action = undefined;
            var n_opt: usize = 0;
            while (it.next()) |s| {
                var first: ?usize = null;
                for (0..2) |m| {
                    const spec = specs[sim.eng.jobs.id[s]];
                    const ok = ref.fits(&caps, n_res, n_ticks, m, spec, now);
                    try testing.expectEqual(ok, rules.fitsNow(&st, s, m));
                    if (ok) {
                        if (first == null) first = m;
                        options[n_opt] = .{ .place = .{ .slot = @intCast(s), .machine = @intCast(m) } };
                        n_opt += 1;
                    }
                }
                // the per-tick "fits nowhere" memo never hides a fit
                try testing.expectEqual(first, rules.firstMachine(&st, s));
                try testing.expectEqual(first == null, sim.eng.jobs.isBlocked(s));
            }

            // 3. random policy: mostly legal placements, sometimes wait / illegal
            const roll = rnd.uintLessThan(u32, 10);
            const act: Action = if (n_opt > 0 and roll < 7)
                options[rnd.uintLessThan(usize, n_opt)]
            else if (roll == 9)
                .{ .place = .{ .slot = rnd.uintLessThan(u32, 32), .machine = rnd.uintLessThan(u32, 3) } }
            else
                .wait;

            const legal = rules.isLegal(&st, act);
            const waiting = sim.eng.jobs.count;
            if (legal and act == .place) {
                const p = act.place;
                try ref.running.append(testing.allocator, .{ .machine = p.machine, .start = now, .spec = specs[sim.eng.jobs.id[p.slot]] });
            } else {
                expected_reward += -@as(f64, @floatFromInt(waiting + ref.runningAt(now)));
                if (!legal) expected_reward += -1.0;
            }
            const res = try sim.eng.step(act);
            try testing.expectEqual(!legal, res.illegal);
        }

        // 4. episode-level results
        const s = sim.eng.stats;
        try testing.expectEqual(@as(usize, 60), s.arrived);
        try testing.expectEqual(@as(usize, 60), s.placed);
        try testing.expectApproxEqAbs(expected_reward, s.total_reward, 1e-6);
        var makespan: i64 = 0;
        for (ref.running.items) |run| makespan = @max(makespan, run.start + run.spec.dur);
        try testing.expectEqual(makespan, s.makespan);
        try testing.expect(sim.eng.machines.allIdle());
        _ = episode;
    }
}

test "property: cached pair scores give the same decision as a fresh scan" {
    // A long-lived policy (incremental cache) against a fresh copy at every
    // step (full rescan), over many placements per tick and time advances.
    inline for (.{ A.policies.BestFit, A.policies.Tetris, A.policies.TetrisSrpt, A.policies.Packer }) |P| {
        for (0..4) |seed| {
            var arr = A.arrivals.Poisson{ .total = 400, .rate = 4 };
            var gen = A.generators.RandomProfiles{ .n_machines = 6, .varying = true };
            var eng = try sched.Engine.init(testing.allocator, .{ .n_res = 3, .n_ticks = 30, .max_jobs = 48 }, .init(&arr), .init(&gen));
            defer eng.deinit();
            try eng.reset(seed);
            var cached = P{ .cache = .{ .alloc = testing.allocator } };
            defer cached.deinit();
            var steps: usize = 0;
            while (!eng.finished and steps < 20_000) : (steps += 1) {
                const st = eng.state();
                var fresh = P{ .cache = .{ .alloc = testing.allocator } };
                defer fresh.deinit();
                const want = fresh.decide(&st);
                const got = cached.decide(&st);
                try testing.expectEqual(want, got);
                _ = try eng.step(got);
            }
        }
    }
}

// ------------------------------------------------------------------ seeds

/// `h`: every event plus the demand grid of each arriving job.
/// `work`: the workload only (arriving jobs: id, tick, duration, demand),
/// without the episode seed or anything the policy decides.
const EventHash = struct {
    h: std.hash.Wyhash = .init(0),
    work: std.hash.Wyhash = .init(0),

    pub fn onEvent(self: *EventHash, ev: sched.Event, st: *const sched.State) void {
        if (ev == .job_arrived) {
            const e = ev.job_arrived;
            std.hash.autoHash(&self.work, e.id);
            std.hash.autoHash(&self.work, e.tick);
            std.hash.autoHash(&self.work, st.jobs.duration[e.slot]);
            for (0..st.jobs.layout.n_res) |r| self.work.update(std.mem.sliceAsBytes(st.jobs.get(e.slot).row(r)));
        }
        self.h.update(@tagName(ev));
        switch (ev) {
            inline else => |e| std.hash.autoHash(&self.h, e),
        }
        if (ev == .job_arrived) {
            const s = ev.job_arrived.slot;
            const job = st.jobs.get(s);
            self.h.update(std.mem.asBytes(&st.jobs.duration[s]));
            for (0..st.jobs.layout.n_res) |r| self.h.update(std.mem.sliceAsBytes(job.row(r)));
        }
    }
};

const RunHash = struct { all: u64, work: u64 };

fn seededRun(comptime Arr: type, comptime Gen: type, arr0: Arr, seeds: []const u64, out: []RunHash) !void {
    var arr = arr0;
    var gen = Gen{ .n_machines = 3 };
    var eng = try sched.Engine.init(testing.allocator, .{ .n_res = 2, .n_ticks = 20, .max_jobs = 12 }, .init(&arr), .init(&gen));
    defer eng.deinit();
    var hash = EventHash{};
    try eng.addObserver(.init(&hash));
    for (seeds, out) |seed, *o| {
        hash = .{};
        try eng.reset(seed);
        var p = A.policies.RandomPolicy{};
        sched.ports.policy.resetIfAny(A.policies.RandomPolicy, &p, seed);
        _ = try eng.run(&p, 100_000);
        o.* = .{ .all = hash.h.final(), .work = hash.work.final() };
    }
}

test "property: same seed gives the same episode, on fresh or reused engines" {
    // Run A: seed 5 on a fresh engine. Run B: seeds 9 then 5 on another
    // engine, so seed 5 follows a different episode. Both seed-5 episodes
    // must produce the identical event stream (and job grids); seed 9 not.
    inline for (.{ A.generators.RandomProfiles, A.generators.DeepRM }) |G| {
        const arrivals = .{
            A.arrivals.Poisson{ .total = 150, .rate = 1.5 },
            A.arrivals.Mmpp{ .total = 150, .rate = 1.5 },
            A.arrivals.ParetoArrivals{ .total = 150, .rate = 1.5 },
            A.arrivals.SelfSimilar{ .total = 150, .rate = 1.5 },
        };
        inline for (arrivals) |arr| {
            var a: [1]RunHash = undefined;
            var b: [3]RunHash = undefined;
            try seededRun(@TypeOf(arr), G, arr, &.{5}, &a);
            try seededRun(@TypeOf(arr), G, arr, &.{ 9, 5, 9 }, &b);
            try testing.expectEqual(a[0], b[1]);
            try testing.expectEqual(b[0], b[2]);
            // different seed -> different workload (not just a different seed field)
            try testing.expect(a[0].work != b[0].work);
        }
    }
}

/// job id -> hash of its demand grid, for every admitted job.
const Demands = struct {
    map: std.AutoHashMapUnmanaged(u32, u64) = .empty,

    pub fn onEvent(self: *Demands, ev: sched.Event, st: *const sched.State) void {
        if (ev != .job_arrived) return;
        const s = ev.job_arrived.slot;
        var h = std.hash.Wyhash.init(st.jobs.duration[s]);
        for (0..st.jobs.layout.n_res) |r| h.update(std.mem.sliceAsBytes(st.jobs.get(s).row(r)));
        self.map.put(testing.allocator, ev.job_arrived.id, h.final()) catch unreachable;
    }
};

fn demandsWith(comptime P: type, d: *Demands) !usize {
    var arr = A.arrivals.Poisson{ .total = 300, .rate = 2 };
    var gen = A.generators.RandomProfiles{ .n_machines = 1 };
    // tiny pool: many pool_full rejections, which differ per policy
    var eng = try sched.Engine.init(testing.allocator, .{ .n_res = 2, .n_ticks = 20, .max_jobs = 3 }, .init(&arr), .init(&gen));
    defer eng.deinit();
    try eng.addObserver(.init(d));
    try eng.reset(11);
    var p = P{};
    defer sched.ports.policy.deinitIfAny(P, &p);
    return (try eng.run(&p, 100_000)).rejected;
}

test "property: a seed fixes every job's profile, even when the pool rejects jobs" {
    var a = Demands{};
    defer a.map.deinit(testing.allocator);
    var b = Demands{};
    defer b.map.deinit(testing.allocator);
    const ra = try demandsWith(A.policies.FifoStrict, &a);
    const rb = try demandsWith(A.policies.ShortestFirst, &b);
    try testing.expect(ra > 0 and rb > 0 and ra != rb); // the policies did reject differently
    var common: usize = 0;
    var it = a.map.iterator();
    while (it.next()) |e| if (b.map.get(e.key_ptr.*)) |h| {
        try testing.expectEqual(e.value_ptr.*, h);
        common += 1;
    };
    try testing.expect(common > 10);
}

/// Generator whose n-th call (from reset) returns a 1-tick job of demand n.
const Counting = struct {
    calls: Elem = 0,
    pub fn reset(self: *Counting, seed: u64) void {
        _ = seed;
        self.calls = 0;
    }
    pub fn machineCount(_: *Counting) usize {
        return 1;
    }
    pub fn machine(_: *Counting, _: usize, cap: []Elem) sched.ports.usage_generator.Error!void {
        @memset(cap, 10);
    }
    pub fn job(self: *Counting, _: sched.JobMeta, out: sched.Usage) sched.ports.usage_generator.Error!u16 {
        self.calls += 1;
        out.resourceMut(0)[0] = self.calls;
        return 1;
    }
};

test "scenario: a job rejected for a full pool still draws its profile" {
    // Pool of 1. Jobs 0 and 1 arrive at t=0, job 2 at t=1.
    //   t=0: job 0 admitted (draw 1), job 1 pool_full (draw 2)
    //   wait (job 0 never placed: still holds the slot)
    //   t=1: job 2 pool_full (draw 3)
    // Place job 0 at t=1, then job 3 arrives at t=2 and is admitted with
    // draw 4. Before the fix, rejected jobs drew nothing and job 3 got 2.
    var metas = [_]sched.JobMeta{
        .{ .id = 0, .arrival = 0 }, .{ .id = 1, .arrival = 0 },
        .{ .id = 2, .arrival = 1 }, .{ .id = 3, .arrival = 2 },
    };
    var arr = A.arrivals.Fixed{ .items = &metas };
    var gen = Counting{};
    var rec = Recorder{};
    defer rec.deinit();
    var eng = try sched.Engine.init(testing.allocator, .{ .n_res = 1, .n_ticks = 4, .max_jobs = 1, .skip_empty = false }, .init(&arr), .init(&gen));
    defer eng.deinit();
    try eng.addObserver(.init(&rec));
    try eng.reset(0);
    try testing.expectEqual(@as(Elem, 1), eng.jobs.get(0).row(0)[0]);
    _ = try eng.step(.wait);
    _ = try eng.step(.{ .place = .{ .slot = 0, .machine = 0 } });
    _ = try eng.step(.wait);
    try testing.expectEqual(@as(u32, 3), eng.jobs.id[0]);
    try testing.expectEqual(@as(Elem, 4), eng.jobs.get(0).row(0)[0]);
    try testing.expectEqual(@as(usize, 2), rec.rejected.items.len);
    for (rec.rejected.items) |r| try testing.expectEqual(sched.model.RejectReason.pool_full, r);
}
