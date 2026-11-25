//! Composition root: parse the command line, build the chosen adapters,
//! wire them into the engine, run.

const std = @import("std");
const sched = @import("sched");
const cli = @import("cli.zig");
const registry = @import("registry.zig");

const usage_text =
    \\sim — SIMD job-to-machine scheduling simulator
    \\
    \\usage: sim [options]
    \\  --machines <spec>   usage generator     (default random)
    \\  --jobs <spec>       arrival source      (default poisson)
    \\  --policy <spec>     scheduling policy   (default fifo), e.g. tetris_srpt:weight=2
    \\  --observe <spec>    observer, repeatable (viz, metrics, log)
    \\  --res <n>           resources per machine (default 10)
    \\  --ticks <n>         horizon in ticks      (default 100)
    \\  --pool <n>          waiting-job slots     (default 128)
    \\  --reward <spec>     neg_in_system | neg_waiting | slowdown | zero
    \\  --max-ticks <n>     truncate episodes at this tick
    \\  --seed <n>          random seed           (default 42)
    \\  --episodes <n>      episodes to run       (default 1)
    \\  --list              list adapters and their options
    \\  --help
    \\
    \\spec: name:key=value,key=value      e.g. random:n=8,caps=100/50,dur=1-10
    \\
    \\examples:
    \\  sim --observe viz
    \\  sim --machines deeprm:n=2 --jobs poisson:n=300,rate=0.8 --policy sjf --observe metrics
    \\  sim --policy best_fit --observe viz:delay=30,per_resource=1 --observe metrics
    \\
;

pub fn main(init: std.process.Init) !void {
    run(init) catch |e| switch (e) {
        error.InvalidArgs => std.process.exit(2),
        else => return e,
    };
}

fn run(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;
    const arena = init.arena.allocator();
    const args = try init.minimal.args.toSlice(arena);

    // ---------------- parse ----------------
    var machines_spec: []const u8 = "random";
    var jobs_spec: []const u8 = "poisson";
    var policy: []const u8 = "fifo";
    var reward_spec: ?[]const u8 = null;
    var observe: std.ArrayList([]const u8) = .empty;
    var cfg = sched.Config{};
    var seed: u64 = 42;
    var episodes: usize = 1;

    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        const a = args[i];
        if (std.mem.eql(u8, a, "--help") or std.mem.eql(u8, a, "-h")) {
            std.debug.print("{s}", .{usage_text});
            return;
        }
        if (std.mem.eql(u8, a, "--list")) {
            std.debug.print("{s}", .{registry.list_text});
            return;
        }
        if (i + 1 >= args.len) return cli.fail("{s} needs a value (see --help)", .{a});
        const v = args[i + 1];
        i += 1;
        if (std.mem.eql(u8, a, "--machines")) machines_spec = v
        else if (std.mem.eql(u8, a, "--jobs")) jobs_spec = v
        else if (std.mem.eql(u8, a, "--policy")) policy = v
        else if (std.mem.eql(u8, a, "--observe")) try observe.append(arena, v)
        else if (std.mem.eql(u8, a, "--res")) cfg.n_res = try parseNum(usize, a, v)
        else if (std.mem.eql(u8, a, "--ticks")) cfg.n_ticks = try parseNum(usize, a, v)
        else if (std.mem.eql(u8, a, "--pool")) cfg.max_jobs = try parseNum(usize, a, v)
        else if (std.mem.eql(u8, a, "--max-ticks")) cfg.max_ticks = try parseNum(i64, a, v)
        else if (std.mem.eql(u8, a, "--seed")) seed = try parseNum(u64, a, v)
        else if (std.mem.eql(u8, a, "--episodes")) episodes = try parseNum(usize, a, v)
        else if (std.mem.eql(u8, a, "--reward")) reward_spec = v
        else return cli.fail("unknown option '{s}' (see --help)", .{a});
    }
    const policy_spec = try cli.Spec.parse(arena, "--policy", policy);
    if (!registry.hasPolicy(policy_spec.name)) return cli.fail("--policy: unknown policy '{s}' (see --list)", .{policy_spec.name});

    // ---------------- build adapters ----------------
    const gen = try registry.generator(arena, io, try cli.Spec.parse(arena, "--machines", machines_spec));
    defer gen.deinit();
    const arr = try registry.arrivals(arena, try cli.Spec.parse(arena, "--jobs", jobs_spec));
    defer arr.deinit();

    var eng = sched.Engine.init(gpa, cfg, arr, gen) catch |e| switch (e) {
        error.ZeroResources, error.ZeroTicks => return cli.fail("--res and --ticks must be > 0", .{}),
        error.NoMachines => return cli.fail("--machines: n must be > 0", .{}),
        else => return e,
    };
    defer eng.deinit();
    if (reward_spec) |rs| eng.setReward(try registry.reward(arena, try cli.Spec.parse(arena, "--reward", rs)));

    for (observe.items) |o| try eng.addObserver(try registry.observer(arena, gpa, io, try cli.Spec.parse(arena, "--observe", o)));
    defer eng.observers.deinitAll();

    // ---------------- run ----------------
    for (0..episodes) |e| {
        const ep_seed = seed +% e;
        try eng.reset(ep_seed);
        const t0 = std.Io.Timestamp.now(io, .awake);
        const st = try registry.runPolicy(&eng, policy_spec, ep_seed, 100_000_000);
        const us = t0.durationTo(std.Io.Timestamp.now(io, .awake)).toMicroseconds();

        std.debug.print(
            "episode {d}  seed {d}  policy {s}  placed {d}/{d}  makespan {d}  avg wait {d:.2}  util {d:.1}%  reward {d:.0}  steps {d}  {d} us\n",
            .{ e, ep_seed, policy, st.placed, st.arrived, st.makespan, st.avgWait(), st.avgUtil() * 100, st.total_reward, st.steps, us },
        );
    }
}

fn parseNum(comptime T: type, flag: []const u8, v: []const u8) !T {
    return std.fmt.parseInt(T, v, 10) catch cli.fail("{s}: '{s}' is not a valid number", .{ flag, v });
}

test {
    _ = cli;
}
