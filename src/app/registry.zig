//! Registry: maps adapter names typed by the user to constructors.
//! Adding an adapter = one entry here; the engine and other adapters do not
//! change. Adapters are allocated in the arena and live for the whole run.

const std = @import("std");
const sched = @import("sched");
const cli = @import("cli.zig");

const ports = sched.ports;
const A = sched.adapters;
const Elem = sched.Elem;
const Spec = cli.Spec;

// ---------------------------------------------------------------- generators

pub fn generator(arena: std.mem.Allocator, s: Spec) !ports.UsageGenerator {
    if (std.mem.eql(u8, s.name, "random")) {
        try s.only(&.{ "n", "caps", "dur", "demand", "varying" });
        const g = try arena.create(A.generators.RandomProfiles);
        const dur = try s.range(u16, "dur", 1, 20);
        const dem = try s.range(Elem, "demand", 0, 30);
        g.* = .{
            .n_machines = try s.int(usize, "n", 4),
            .caps = try s.list(arena, Elem, "caps", &.{ 100, 80, 60, 40 }),
            .dur_min = dur[0],
            .dur_max = dur[1],
            .demand_min = dem[0],
            .demand_max = dem[1],
            .varying = try s.boolean("varying", false),
        };
        return .init(g);
    }
    if (std.mem.eql(u8, s.name, "deeprm")) {
        try s.only(&.{ "n", "cap", "short", "long", "short_prob" });
        const g = try arena.create(A.generators.DeepRM);
        const long = try s.range(u16, "long", 10, 15);
        g.* = .{
            .n_machines = try s.int(usize, "n", 1),
            .cap = try s.int(Elem, "cap", 100),
            .short_max = try s.int(u16, "short", 3),
            .long_min = long[0],
            .long_max = long[1],
            .short_prob = try s.float("short_prob", 0.8),
        };
        return .init(g);
    }
    return cli.fail("--machines: unknown generator '{s}' (see --list)", .{s.name});
}

// ---------------------------------------------------------------- arrivals

pub fn arrivals(arena: std.mem.Allocator, s: Spec) !ports.ArrivalSource {
    if (std.mem.eql(u8, s.name, "poisson")) {
        try s.only(&.{ "n", "rate" });
        const p = try arena.create(A.arrivals.Poisson);
        p.* = .{ .total = try s.int(usize, "n", 500), .rate = try s.float("rate", 1.0) };
        return .init(p);
    }
    if (std.mem.eql(u8, s.name, "batch")) {
        try s.only(&.{"n"});
        const p = try arena.create(A.arrivals.Poisson);
        p.* = .{ .total = try s.int(usize, "n", 200), .rate = 0 };
        return .init(p);
    }
    return cli.fail("--jobs: unknown arrival source '{s}' (see --list)", .{s.name});
}

// ---------------------------------------------------------------- observers

pub fn observer(arena: std.mem.Allocator, gpa: std.mem.Allocator, io: std.Io, s: Spec) !ports.Observer {
    if (std.mem.eql(u8, s.name, "viz")) {
        try s.only(&.{ "delay", "per_resource", "on_place", "jobs", "cols", "color" });
        const v = try arena.create(A.observers.TerminalViz);
        v.* = .init(gpa, io, .{
            .delay_ms = try s.int(u32, "delay", 80),
            .per_resource = try s.boolean("per_resource", false),
            .on_place = try s.boolean("on_place", false),
            .max_jobs = try s.int(usize, "jobs", 20),
            .max_cols = try s.int(usize, "cols", 100),
            .color = try s.boolean("color", true),
        });
        return .init(v);
    }
    if (std.mem.eql(u8, s.name, "metrics")) {
        try s.only(&.{});
        const m = try arena.create(A.observers.Metrics);
        m.* = .init(gpa);
        return .init(m);
    }
    if (std.mem.eql(u8, s.name, "log")) {
        try s.only(&.{"ticks"});
        const l = try arena.create(A.observers.EventLog);
        l.* = .{ .ticks = try s.boolean("ticks", false) };
        return .init(l);
    }
    return cli.fail("--observe: unknown observer '{s}' (see --list)", .{s.name});
}

// ---------------------------------------------------------------- policies
// Chosen at run time, executed through a comptime-specialized loop per type.

pub const policy_types = .{
    A.policies.FifoBackfill,
    A.policies.FifoStrict,
    A.policies.ShortestFirst,
    A.policies.BestFit,
    A.policies.RandomPolicy,
};

pub fn hasPolicy(name: []const u8) bool {
    inline for (policy_types) |P| if (std.mem.eql(u8, name, P.name)) return true;
    return false;
}

pub fn runPolicy(eng: *sched.Engine, name: []const u8, seed: u64, max_steps: usize) !sched.Stats {
    inline for (policy_types) |P| {
        if (std.mem.eql(u8, name, P.name)) {
            comptime ports.assertPolicy(P);
            var p = P{};
            ports.policy.resetIfAny(P, &p, seed);
            return eng.run(&p, max_steps);
        }
    }
    return cli.fail("--policy: unknown policy '{s}' (see --list)", .{name});
}

pub const list_text =
    \\Machines / usage generators (--machines):
    \\  random   n=4  caps=100/80/60/40  dur=1-20  demand=0-30  varying=0
    \\           machines cycle through caps; uniform random duration and demand
    \\  deeprm   n=1  cap=100  short=3  long=10-15  short_prob=0.8
    \\           DeepRM workload: 80% short jobs, one dominant resource per job
    \\
    \\Job arrivals (--jobs):
    \\  poisson  n=500  rate=1.0      exponential gaps, mean 1/rate ticks
    \\  batch    n=200                all jobs at tick 0
    \\
    \\Policies (--policy):
    \\  fifo         oldest job that fits now (backfill)
    \\  fifo_strict  only the oldest job; wait if it does not fit
    \\  sjf          shortest job that fits now
    \\  best_fit     (job, machine) pair leaving least free capacity
    \\  random       random legal placement
    \\
    \\Observers (--observe, repeatable):
    \\  viz      delay=80  per_resource=0  on_place=0  jobs=20  cols=100  color=1
    \\  metrics  wait percentiles, utilization, jobs per machine
    \\  log      ticks=0   one line per event
    \\
;
