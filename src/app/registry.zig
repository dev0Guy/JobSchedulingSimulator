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

pub fn generator(arena: std.mem.Allocator, io: std.Io, s: Spec) !ports.UsageGenerator {
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
    if (std.mem.eql(u8, s.name, "psplib")) {
        try s.only(&.{ "path", "n", "cap", "order" });
        const path = s.raw("path") orelse return cli.fail("--machines psplib: path=<.sm file or directory> is required", .{});
        const g = try arena.create(A.generators.Psplib);
        const order_s = s.raw("order") orelse "random";
        const order = std.meta.stringToEnum(A.generators.psplib.Order, order_s) orelse
            return cli.fail("--machines psplib: order must be random or sequential, got '{s}'", .{order_s});
        var acts: std.ArrayList(A.generators.psplib.Activity) = .empty;
        const n_ren = try loadPsplib(arena, io, path, &acts);
        g.* = .{
            .acts = acts.items,
            .n_renewable = n_ren,
            .n_machines = try s.int(usize, "n", 1),
            .cap = try s.int(Elem, "cap", 100),
            .order = order,
        };
        return .init(g);
    }
    return cli.fail("--machines: unknown generator '{s}' (see --list)", .{s.name});
}

/// Read a PSPLIB instance file, or every .sm/.mm file of a directory (in
/// name order, so the template list does not depend on the file system).
/// Returns the smallest renewable-resource count over the instances.
fn loadPsplib(arena: std.mem.Allocator, io: std.Io, path: []const u8, acts: *std.ArrayList(A.generators.psplib.Activity)) !usize {
    const cwd = std.Io.Dir.cwd();
    const limit: std.Io.Limit = .limited(64 << 20);
    var names: std.ArrayList([]const u8) = .empty;
    var dir = cwd.openDir(io, path, .{ .iterate = true }) catch |e| switch (e) {
        error.NotDir => null,
        else => return cli.fail("--machines psplib: cannot open '{s}': {t}", .{ path, e }),
    };
    if (dir) |*d| {
        defer d.close(io);
        var it = d.iterate();
        while (it.next(io) catch |e| return cli.fail("--machines psplib: cannot list '{s}': {t}", .{ path, e })) |ent| {
            if (ent.kind != .file) continue;
            if (!std.mem.endsWith(u8, ent.name, ".sm") and !std.mem.endsWith(u8, ent.name, ".mm")) continue;
            try names.append(arena, try std.fs.path.join(arena, &.{ path, ent.name }));
        }
        std.mem.sort([]const u8, names.items, {}, struct {
            fn lt(_: void, a: []const u8, b: []const u8) bool {
                return std.mem.lessThan(u8, a, b);
            }
        }.lt);
        if (names.items.len == 0) return cli.fail("--machines psplib: no .sm/.mm files in '{s}'", .{path});
    } else try names.append(arena, path);

    var n_ren: usize = std.math.maxInt(usize);
    for (names.items) |name| {
        const text = cwd.readFileAlloc(io, name, arena, limit) catch |e|
            return cli.fail("--machines psplib: cannot read '{s}': {t}", .{ name, e });
        const r = A.generators.psplib.parse(arena, text, acts) catch
            return cli.fail("--machines psplib: '{s}' is not a valid PSPLIB instance", .{name});
        n_ren = @min(n_ren, r);
    }
    return n_ren;
}

test "psplib spec loads a file and a directory" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.createDirPath(io, "set");
    try tmp.dir.writeFile(io, .{ .sub_path = "set/a.sm", .data = A.generators.psplib.example_sm });
    try tmp.dir.writeFile(io, .{ .sub_path = "set/b.sm", .data = A.generators.psplib.example_sm });
    try tmp.dir.writeFile(io, .{ .sub_path = "set/notes.txt", .data = "ignored" });
    const base = try tmp.dir.realPathFileAlloc(io, "set", arena);

    var acts: std.ArrayList(A.generators.psplib.Activity) = .empty;
    try std.testing.expectEqual(@as(usize, 4), try loadPsplib(arena, io, base, &acts));
    try std.testing.expectEqual(@as(usize, 4), acts.items.len); // 2 files x 2 activities
    acts.clearRetainingCapacity();
    _ = try loadPsplib(arena, io, try std.fs.path.join(arena, &.{ base, "a.sm" }), &acts);
    try std.testing.expectEqual(@as(usize, 2), acts.items.len);

    const spec = try std.fmt.allocPrint(arena, "psplib:path={s},n=2,cap=40,order=sequential", .{base});
    const g = try generator(arena, io, try cli.Spec.parse(arena, "--machines", spec));
    try std.testing.expectEqual(@as(usize, 2), g.machineCount());
}

// ---------------------------------------------------------------- arrivals

pub fn arrivals(arena: std.mem.Allocator, s: Spec) !ports.ArrivalSource {
    if (std.mem.eql(u8, s.name, "poisson")) {
        try s.only(&.{ "n", "rate" });
        const p = try arena.create(A.arrivals.Poisson);
        p.* = .{ .total = try s.int(usize, "n", 500), .rate = try s.float("rate", 1.0) };
        return .init(p);
    }
    if (std.mem.eql(u8, s.name, "mmpp")) {
        try s.only(&.{ "n", "rate", "ratio", "calm", "burst" });
        const p = try arena.create(A.arrivals.Mmpp);
        p.* = .{
            .total = try s.int(usize, "n", 500),
            .rate = try s.float("rate", 1.0),
            .ratio = try s.float("ratio", 10),
            .calm = try s.float("calm", 40),
            .burst_len = try s.float("burst", 10),
        };
        if (p.rate <= 0 or p.ratio < 1) return cli.fail("--jobs mmpp: need rate > 0 and ratio >= 1", .{});
        return .init(p);
    }
    if (std.mem.eql(u8, s.name, "pareto")) {
        try s.only(&.{ "n", "rate", "alpha" });
        const p = try arena.create(A.arrivals.ParetoArrivals);
        p.* = .{ .total = try s.int(usize, "n", 500), .rate = try s.float("rate", 1.0), .alpha = try s.float("alpha", 1.5) };
        if (p.rate <= 0 or p.alpha <= 1) return cli.fail("--jobs pareto: need rate > 0 and alpha > 1", .{});
        return .init(p);
    }
    if (std.mem.eql(u8, s.name, "selfsimilar")) {
        try s.only(&.{ "n", "rate", "sources", "alpha", "period" });
        const p = try arena.create(A.arrivals.SelfSimilar);
        p.* = .{
            .total = try s.int(usize, "n", 500),
            .rate = try s.float("rate", 1.0),
            .sources = try s.int(usize, "sources", 8),
            .alpha = try s.float("alpha", 1.4),
            .period = try s.float("period", 50),
        };
        if (p.rate <= 0 or p.alpha <= 1 or p.sources == 0 or p.sources > 256)
            return cli.fail("--jobs selfsimilar: need rate > 0, alpha > 1, 1 <= sources <= 256", .{});
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

// ---------------------------------------------------------------- rewards

pub fn reward(arena: std.mem.Allocator, s: Spec) !ports.Reward {
    if (std.mem.eql(u8, s.name, "neg_in_system")) {
        try s.only(&.{});
        return .init(try arena.create(A.rewards.NegInSystem));
    }
    if (std.mem.eql(u8, s.name, "neg_waiting")) {
        try s.only(&.{});
        return .init(try arena.create(A.rewards.NegWaiting));
    }
    if (std.mem.eql(u8, s.name, "zero")) {
        try s.only(&.{});
        return .init(try arena.create(A.rewards.Zero));
    }
    if (std.mem.eql(u8, s.name, "slowdown")) {
        try s.only(&.{"min_dur"});
        const r = try arena.create(A.rewards.Slowdown);
        r.* = .{ .min_dur = try s.int(u16, "min_dur", 1) };
        return .init(r);
    }
    return cli.fail("--reward: unknown reward '{s}' (see --list)", .{s.name});
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
    if (std.mem.eql(u8, s.name, "trace")) {
        try s.only(&.{ "path", "ticks" });
        const t = try arena.create(A.observers.Trace);
        t.* = .{ .ticks = try s.boolean("ticks", true) };
        const path = s.raw("path") orelse "trace.jsonl";
        t.open(io, std.Io.Dir.cwd(), path) catch |e|
            return cli.fail("--observe trace: cannot create '{s}': {t}", .{ path, e });
        return .init(t);
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
    A.policies.Tetris,
    A.policies.TetrisSrpt,
};

pub fn hasPolicy(name: []const u8) bool {
    inline for (policy_types) |P| if (std.mem.eql(u8, name, P.name)) return true;
    return false;
}

/// Build policy P from a spec: every numeric field of P is an option
/// (e.g. `tetris_srpt:weight=2`); other fields keep their defaults.
pub fn makePolicy(comptime P: type, s: Spec) !P {
    const info = @typeInfo(P).@"struct";
    const numeric = comptime blk: {
        var names: []const []const u8 = &.{};
        for (info.field_names, info.field_types) |n, T| switch (@typeInfo(T)) {
            .float, .int => names = names ++ .{n},
            else => {},
        };
        break :blk names;
    };
    try s.only(numeric);
    var p = P{};
    inline for (info.field_names, info.field_types) |n, T| switch (@typeInfo(T)) {
        .float => @field(p, n) = @floatCast(try s.float(n, @field(p, n))),
        .int => @field(p, n) = try s.int(T, n, @field(p, n)),
        else => {},
    };
    return p;
}

pub fn runPolicy(eng: *sched.Engine, s: Spec, seed: u64, max_steps: usize) !sched.Stats {
    inline for (policy_types) |P| {
        if (std.mem.eql(u8, s.name, P.name)) {
            comptime ports.assertPolicy(P);
            var p = try makePolicy(P, s);
            defer ports.policy.deinitIfAny(P, &p);
            ports.policy.resetIfAny(P, &p, seed);
            return eng.run(&p, max_steps);
        }
    }
    return cli.fail("--policy: unknown policy '{s}' (see --list)", .{s.name});
}

/// The action policy `s` would take in the current state.
pub fn policyAction(eng: *sched.Engine, s: Spec, seed: u64) !sched.Action {
    const st = eng.state();
    inline for (policy_types) |P| {
        if (std.mem.eql(u8, s.name, P.name)) {
            var p = try makePolicy(P, s);
            defer ports.policy.deinitIfAny(P, &p);
            ports.policy.resetIfAny(P, &p, seed);
            return p.decide(&st);
        }
    }
    return cli.fail("policy: unknown policy '{s}' (see --list)", .{s.name});
}

test "policy specs set numeric fields" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const p = try makePolicy(A.policies.TetrisSrpt, try Spec.parse(a, "--policy", "tetris_srpt:weight=2.5"));
    try std.testing.expectEqual(@as(f32, 2.5), p.weight);
    const d = try makePolicy(A.policies.TetrisSrpt, try Spec.parse(a, "--policy", "tetris_srpt"));
    try std.testing.expectEqual(@as(f32, 1.5), d.weight);
}

pub const list_text =
    \\Machines / usage generators (--machines):
    \\  random   n=4  caps=100/80/60/40  dur=1-20  demand=0-30  varying=0
    \\           machines cycle through caps; uniform random duration and demand
    \\  deeprm   n=1  cap=100  short=3  long=10-15  short_prob=0.8
    \\           DeepRM workload: 80% short jobs, one dominant resource per job
    \\  psplib   path=<.sm file or dir>  n=1  cap=100  order=random
    \\           PSPLIB (RCPSP) activities as jobs: duration + demand / availability x cap;
    \\           order=random (seeded) or sequential; n_res <= renewable resources (4)
    \\
    \\Job arrivals (--jobs):
    \\  poisson      n=500  rate=1.0                       exponential gaps, mean 1/rate ticks
    \\  mmpp         n=500  rate=1.0  ratio=10  calm=40  burst=10
    \\               bursty: calm/burst states (mean lengths in ticks), burst rate = ratio x calm rate
    \\  pareto       n=500  rate=1.0  alpha=1.5             heavy-tailed gaps (infinite variance if alpha <= 2)
    \\  selfsimilar  n=500  rate=1.0  sources=8  alpha=1.4  period=50
    \\               Pareto ON/OFF sources, long-range dependent, Hurst H = (3 - alpha)/2
    \\  batch        n=200                                 all jobs at tick 0
    \\
    \\Policies (--policy):
    \\  fifo         oldest job that fits now (backfill)
    \\  fifo_strict  only the oldest job; wait if it does not fit
    \\  sjf          shortest job that fits now
    \\  best_fit     (job, machine) pair leaving least free capacity
    \\  random       random legal placement
    \\  tetris       (job, machine) pair whose demand best aligns with free capacity over time
    \\  tetris_srpt  weight=1.5   tetris alignment minus weight x shortest-job term (Grandl et al.)
    \\               0 = pure tetris (packing, makespan), ~2 = close to sjf (slowdown)
    \\
    \\Rewards (--reward), per elapsed tick:
    \\  neg_in_system           -(waiting + running)        (default)
    \\  neg_waiting             -waiting
    \\  slowdown  min_dur=1     -sum 1/max(T_j, min_dur) over jobs in system (DeepRM)
    \\  zero                    0
    \\
    \\Observers (--observe, repeatable):
    \\  viz      delay=80  per_resource=0  on_place=0  jobs=20  cols=100  color=1
    \\  metrics  wait percentiles, utilization, jobs per machine
    \\  log      ticks=0   one line per event
    \\  trace    path=trace.jsonl  ticks=1
    \\           every event + job profiles as JSON Lines, for replay / visualization
    \\           (python python/examples/replay.py trace.jsonl)
    \\
;
