//! C ABI: a driving adapter that exposes the engine to other languages
//! (Python via ctypes → python/sched_env). Built as libsched by
//! `zig build lib`.
//!
//! Lifecycle:
//!     h = sim_create(&cfg)          // NULL on bad config (message on stderr)
//!     sim_dims(h, &dims)            // sizes and strides of the shared buffers
//!     sim_reset(h, seed)
//!     loop: sim_action_mask(h, mask); sim_step(h, action, &res)
//!     sim_destroy(h)
//!
//! Zero-copy: sim_machines_ptr / sim_jobs_ptr / sim_duration_ptr /
//! sim_arrival_ptr point into engine memory. They stay valid until
//! sim_destroy and always show the current state (read-only for callers).
//! Grids are laid out [grid][res][stride] with `block` elements per grid.
//! Job rows: ticks [0, n_ticks) are real. Machine rows use their own
//! `m_stride` / `m_block` (sliding window): tick t of the horizon is at
//! row[sim_machine_head() + t], and the head moves as time advances.
//!
//! Action index (rules.encode): i < max_jobs·n_machines → place(slot =
//! i / n_machines, machine = i % n_machines); i == max_jobs·n_machines → wait.

const std = @import("std");
const sched = @import("sched");
const cli = @import("cli.zig");
const registry = @import("registry.zig");

const rules = sched.engine.rules;
const Elem = sched.Elem;

pub const SimConfig = extern struct {
    /// Generator spec, e.g. "deeprm:n=2" (same syntax as --machines).
    machines: [*:0]const u8,
    /// Arrival spec, e.g. "poisson:n=100,rate=0.5" (same as --jobs).
    jobs: [*:0]const u8,
    n_res: u32 = 10,
    n_ticks: u32 = 100,
    max_jobs: u32 = 128,
    /// 0 = neg_in_system, 1 = neg_waiting, 2 = zero.
    reward: u32 = 0,
    /// Truncate at this tick; < 0 = never.
    max_ticks: i64 = -1,
    illegal_penalty: f32 = -1.0,
    skip_empty: u8 = 1,
    drop_impossible: u8 = 1,
};

pub const SimDims = extern struct {
    n_machines: u32,
    n_res: u32,
    n_ticks: u32,
    max_jobs: u32,
    n_actions: u32,
    /// Elements per row (n_ticks + padding).
    stride: u32,
    /// Elements per grid (n_res · stride + alignment).
    block: u32,
    /// Largest capacity over all machines and resources (for normalizing).
    max_cap: u32,
    /// Machine rows: elements per row / per grid (n_ticks + slack + padding).
    m_stride: u32,
    m_block: u32,
};

pub const SimStep = extern struct {
    reward: f32 = 0,
    terminated: u8 = 0,
    truncated: u8 = 0,
    placed: u8 = 0,
    illegal: u8 = 0,
};

pub const SimStats = extern struct {
    now: i64,
    arrived: u64,
    placed: u64,
    rejected: u64,
    illegal: u64,
    steps: u64,
    waiting: u64,
    makespan: i64,
    avg_wait: f64,
    max_wait: i64,
    avg_util: f64,
    total_reward: f64,
};

const Sim = struct {
    arena: std.heap.ArenaAllocator,
    gen: sched.ports.UsageGenerator,
    arr: sched.ports.ArrivalSource,
    eng: sched.Engine,
    started: bool = false,
};

const gpa = std.heap.smp_allocator;

fn create(c: *const SimConfig) !*Sim {
    const sim = try gpa.create(Sim);
    errdefer gpa.destroy(sim);
    sim.arena = .init(gpa);
    errdefer sim.arena.deinit();
    const a = sim.arena.allocator();

    const reward: rules.Reward = switch (c.reward) {
        0 => .neg_in_system,
        1 => .neg_waiting,
        2 => .zero,
        else => return cli.fail("sim_create: reward must be 0, 1 or 2", .{}),
    };
    const cfg = sched.Config{
        .n_res = c.n_res,
        .n_ticks = c.n_ticks,
        .max_jobs = c.max_jobs,
        .reward = reward,
        .max_ticks = if (c.max_ticks < 0) null else c.max_ticks,
        .illegal_penalty = c.illegal_penalty,
        .skip_empty = c.skip_empty != 0,
        .drop_impossible = c.drop_impossible != 0,
    };
    if (c.max_jobs == 0) return cli.fail("sim_create: max_jobs must be > 0", .{});

    sim.gen = try registry.generator(a, try cli.Spec.parse(a, "machines", std.mem.span(c.machines)));
    errdefer sim.gen.deinit();
    sim.arr = try registry.arrivals(a, try cli.Spec.parse(a, "jobs", std.mem.span(c.jobs)));
    errdefer sim.arr.deinit();
    sim.eng = try sched.Engine.init(gpa, cfg, sim.arr, sim.gen);
    sim.started = false;
    return sim;
}

export fn sim_create(cfg: *const SimConfig) ?*Sim {
    return create(cfg) catch |e| {
        if (e != error.InvalidArgs) std.debug.print("sim_create: {s}\n", .{@errorName(e)});
        return null;
    };
}

export fn sim_destroy(sim: ?*Sim) void {
    const s = sim orelse return;
    s.eng.deinit();
    s.arr.deinit();
    s.gen.deinit();
    s.arena.deinit();
    gpa.destroy(s);
}

/// 0 = ok, -1 = error.
export fn sim_reset(sim: *Sim, seed: u64) c_int {
    sim.eng.reset(seed) catch return -1;
    sim.started = true;
    return 0;
}

export fn sim_dims(sim: *const Sim, out: *SimDims) void {
    const e = &sim.eng;
    var max_cap: Elem = 0;
    for (0..e.layout.n_res) |r| max_cap = @max(max_cap, e.machines.maxCap(r));
    out.* = .{
        .n_machines = @intCast(e.machines.n),
        .n_res = @intCast(e.layout.n_res),
        .n_ticks = @intCast(e.layout.n_ticks),
        .max_jobs = @intCast(e.jobs.cap),
        .n_actions = @intCast(e.jobs.cap * e.machines.n + 1),
        .stride = @intCast(e.layout.stride),
        .block = @intCast(e.layout.block),
        .max_cap = max_cap,
        .m_stride = @intCast(e.machines.ring.stride),
        .m_block = @intCast(e.machines.ring.block),
    };
}

/// 0 = ok, -1 = episode not running (call sim_reset), -2 = engine error.
export fn sim_step(sim: *Sim, action: u32, out: *SimStep) c_int {
    if (!sim.started or sim.eng.done()) return -1;
    const st = sim.eng.state();
    const r = sim.eng.step(rules.decode(&st, action)) catch return -2;
    out.* = .{
        .reward = r.reward,
        .terminated = @intFromBool(r.terminated),
        .truncated = @intFromBool(r.truncated),
        .placed = @intFromBool(r.placed),
        .illegal = @intFromBool(r.illegal),
    };
    return 0;
}

/// mask must hold n_actions bytes; 1 = legal.
export fn sim_action_mask(sim: *const Sim, mask: [*]u8) void {
    const st = sim.eng.state();
    const n = rules.actionCount(&st);
    rules.actionMask(&st, @as([*]bool, @ptrCast(mask))[0..n]);
}

/// valid must hold max_jobs bytes; 1 = slot holds a waiting job.
export fn sim_slot_valid(sim: *const Sim, valid: [*]u8) void {
    const jobs = &sim.eng.jobs;
    for (0..jobs.cap) |s| valid[s] = @intFromBool(jobs.isValid(s));
}

/// Free capacity, [n_machines][n_res][m_stride] u16; the horizon of each
/// row starts at sim_machine_head().
export fn sim_machines_ptr(sim: *const Sim) [*]const Elem {
    return sim.eng.machines.buf.ptr;
}
/// Offset of tick 0 (now) inside every machine row. Changes with time.
export fn sim_machine_head(sim: *const Sim) u32 {
    return @intCast(sim.eng.machines.head);
}
/// Capacity, [n_machines][n_res] u16.
export fn sim_capacity_ptr(sim: *const Sim) [*]const Elem {
    return sim.eng.machines.cap.ptr;
}
/// Waiting-job demand, [max_jobs][n_res][stride] u16 (0 for empty slots).
export fn sim_jobs_ptr(sim: *const Sim) [*]const Elem {
    return sim.eng.jobs.buf.ptr;
}
/// Duration per slot, [max_jobs] u16 (meaningful for valid slots only).
export fn sim_duration_ptr(sim: *const Sim) [*]const u16 {
    return sim.eng.jobs.duration.ptr;
}
/// Arrival tick per slot, [max_jobs] i64 (meaningful for valid slots only).
export fn sim_arrival_ptr(sim: *const Sim) [*]const i64 {
    return sim.eng.jobs.arrival.ptr;
}

export fn sim_now(sim: *const Sim) i64 {
    return sim.eng.now;
}

export fn sim_stats(sim: *const Sim, out: *SimStats) void {
    const s = sim.eng.stats;
    out.* = .{
        .now = sim.eng.now,
        .arrived = s.arrived,
        .placed = s.placed,
        .rejected = s.rejected,
        .illegal = s.illegal,
        .steps = s.steps,
        .waiting = sim.eng.jobs.count,
        .makespan = s.makespan,
        .avg_wait = s.avgWait(),
        .max_wait = s.max_wait,
        .avg_util = s.avgUtil(),
        .total_reward = s.total_reward,
    };
}

/// Reset with `seed` and run a whole episode with a built-in policy
/// ("fifo", "sjf", "best_fit", …) at native speed. 0 = ok, -1 = error.
export fn sim_run_policy(sim: *Sim, name: [*:0]const u8, seed: u64, out: *SimStats) c_int {
    sim.eng.reset(seed) catch return -1;
    sim.started = true;
    _ = registry.runPolicy(&sim.eng, std.mem.span(name), seed, std.math.maxInt(usize)) catch return -1;
    sim_stats(sim, out);
    return 0;
}

/// Action index a built-in policy would choose in the current state
/// (for stepping / visualizing heuristics from Python). -1 = unknown name.
/// Stateful policies (random) are reseeded from the clock and step count.
export fn sim_policy_action(sim: *Sim, name: [*:0]const u8) i64 {
    const n = std.mem.span(name);
    const st = sim.eng.state();
    inline for (registry.policy_types) |P| {
        if (std.mem.eql(u8, n, P.name)) {
            var p = P{};
            sched.ports.policy.resetIfAny(P, &p, sim.eng.seed ^ (sim.eng.stats.steps *% 0x9e3779b97f4a7c15));
            return @intCast(rules.encode(&st, p.decide(&st)));
        }
    }
    return -1;
}

test "capi: create, mask, step to the end, run_policy" {
    const cfg = SimConfig{
        .machines = "deeprm:n=2",
        .jobs = "poisson:n=30,rate=0.5",
        .n_res = 2,
        .n_ticks = 20,
        .max_jobs = 8,
    };
    const sim = sim_create(&cfg) orelse return error.CreateFailed;
    defer sim_destroy(sim);

    var d: SimDims = undefined;
    sim_dims(sim, &d);
    try std.testing.expectEqual(@as(u32, 8 * 2 + 1), d.n_actions);

    var res: SimStep = .{};
    try std.testing.expectEqual(@as(c_int, -1), sim_step(sim, 0, &res)); // before reset
    try std.testing.expectEqual(@as(c_int, 0), sim_reset(sim, 1));

    var mask: [17]u8 = undefined;
    var steps: usize = 0;
    while (true) : (steps += 1) {
        sim_action_mask(sim, &mask);
        try std.testing.expectEqual(@as(u8, 1), mask[16]); // wait always legal
        var a: u32 = 16;
        for (mask[0..16], 0..) |m, i| if (m == 1) {
            a = @intCast(i);
            break;
        };
        try std.testing.expectEqual(@as(c_int, 0), sim_step(sim, a, &res));
        try std.testing.expectEqual(@as(u8, 0), res.illegal);
        if (res.terminated == 1 or res.truncated == 1) break;
    }
    var st: SimStats = undefined;
    sim_stats(sim, &st);
    try std.testing.expectEqual(st.arrived, st.placed);
    try std.testing.expectEqual(@as(c_int, -1), sim_step(sim, 16, &res)); // after end

    try std.testing.expectEqual(@as(c_int, 0), sim_run_policy(sim, "sjf", 1, &st));
    const want = st;

    // stepping sjf one decision at a time gives the same episode
    try std.testing.expectEqual(@as(c_int, 0), sim_reset(sim, 1));
    while (true) {
        const a = sim_policy_action(sim, "sjf");
        try std.testing.expect(a >= 0);
        try std.testing.expectEqual(@as(c_int, 0), sim_step(sim, @intCast(a), &res));
        if (res.terminated == 1 or res.truncated == 1) break;
    }
    sim_stats(sim, &st);
    try std.testing.expectEqual(want.avg_wait, st.avg_wait);
    try std.testing.expectEqual(want.makespan, st.makespan);
    try std.testing.expectEqual(@as(i64, -1), sim_policy_action(sim, "nope"));
    try std.testing.expectEqual(st.arrived, st.placed);
}
