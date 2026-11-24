//! Fit-mask benchmark, CPU side of the CPU-vs-GPU comparison.
//!
//! Workload (the same one bench/fitmask_gpu.py runs on the GPU):
//!   for every env e, job j, machine m:
//!       mask[e][j][m] = all(job[e][j][r][k] <= free[e][m][r][k])
//!                       for every resource r and tick k < duration
//! This is the action mask an RL agent needs every step.
//!
//! Output: fit tests per second for growing numbers of environments.
//! Run: zig build fitmask   (ReleaseFast)
//!      zig build fitmask -- --threads 8   (default: all CPUs)
//!      zig build fitmask -- --csv         (machine-readable, used by bench/compare.py)

const std = @import("std");
const sched = @import("sched");
const grid = sched.grid;
const Elem = grid.Elem;
const Usage = grid.Usage;

// Same sizes as fitmask_gpu.py
const J = 64; // job slots per env
const M = 4; // machines per env
const R = 10; // resources
const T = 100; // ticks
const envs_list = [_]usize{ 1, 4, 16, 64, 256, 1024 };

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    var csv = false;
    var n_threads: usize = std.Thread.getCpuCount() catch 1;
    var ai: usize = 1;
    while (ai < args.len) : (ai += 1) {
        if (std.mem.eql(u8, args[ai], "--csv")) csv = true;
        if (std.mem.eql(u8, args[ai], "--threads") and ai + 1 < args.len) {
            ai += 1;
            n_threads = try std.fmt.parseInt(usize, args[ai], 10);
        }
    }
    n_threads = @max(1, n_threads);

    const layout = try grid.Layout.init(R, T, grid.simd.padding);
    const max_e = envs_list[envs_list.len - 1];

    const machines = try gpa.alignedAlloc(Elem, grid.simd.ALIGN, max_e * M * layout.block);
    defer gpa.free(machines);
    const jobs = try gpa.alignedAlloc(Elem, grid.simd.ALIGN, max_e * J * layout.block);
    defer gpa.free(jobs);
    const dur = try gpa.alloc(u16, max_e * J);
    defer gpa.free(dur);
    const mask = try gpa.alloc(bool, max_e * J * M);
    defer gpa.free(mask);

    // Random data: machines partly used (free in [20, 100]), jobs with
    // duration 1..20 and demand 0..30 → a realistic mix of fits / no-fits.
    var prng = std.Random.DefaultPrng.init(42);
    const rnd = prng.random();
    @memset(machines, 0);
    @memset(jobs, 0);
    for (0..max_e * M) |i| {
        const u = Usage.fromPool(machines, layout, i);
        for (0..R) |r| for (u.row(r)) |*v| {
            v.* = rnd.intRangeAtMost(Elem, 20, 100);
        };
    }
    for (0..max_e * J) |i| {
        const d = rnd.intRangeAtMost(u16, 1, 20);
        dur[i] = d;
        const u = Usage.fromPool(jobs, layout, i);
        for (0..R) |r| @memset(u.resourceMut(r)[0..d], rnd.intRangeAtMost(Elem, 0, 30));
    }

    if (!csv) std.debug.print(
        "fit-mask bench (CPU)   L = {d} x {s}   J={d} M={d} R={d} T={d}   threads = {d}\n\n",
        .{ grid.simd.L, @typeName(Elem), J, M, R, T, n_threads },
    );
    if (!csv) std.debug.print("{s:>6} | {s:>12} | {s:>14} | {s:>12} | {s:>14} | {s:>8} | {s:>7}\n", .{ "envs", "1 thread µs", "tests / s", "N threads µs", "tests / s", "speedup", "fits %" });
    if (!csv) std.debug.print("{s:->6}-+-{s:->12}-+-{s:->14}-+-{s:->12}-+-{s:->14}-+-{s:->8}-+-{s:->7}\n", .{ "", "", "", "", "", "", "" });

    const data = Data{ .machines = machines, .jobs = jobs, .dur = dur, .mask = mask, .layout = layout };

    for (envs_list) |n_env| {
        const t1 = try bestOf(io, data, n_env, 1);
        const tn = try bestOf(io, data, n_env, n_threads);

        var fits: usize = 0;
        for (mask[0 .. n_env * J * M]) |b| fits += @intFromBool(b);
        const tests: f64 = @floatFromInt(n_env * J * M);
        if (csv) {
            std.debug.print("CSV,{d},{d},{d},{d}\n", .{ n_env, t1 * 1e6, tn * 1e6, n_threads });
            continue;
        }
        std.debug.print("{d:>6} | {d:>12.1} | {e:>14.3} | {d:>12.1} | {e:>14.3} | {d:>7.2}x | {d:>6.1}%\n", .{
            n_env, t1 * 1e6, tests / t1, tn * 1e6, tests / tn, t1 / tn,
            100 * @as(f64, @floatFromInt(fits)) / tests,
        });
    }
    if (!csv) std.debug.print("\nCompare with: python3 bench/fitmask_gpu.py\n", .{});
}

const Data = struct {
    machines: []align(grid.simd.CACHE) Elem,
    jobs: []align(grid.simd.CACHE) Elem,
    dur: []const u16,
    mask: []bool,
    layout: grid.Layout,
};

/// Best-of-7 seconds for one batch of `n_env` envs on `n_threads` threads.
fn bestOf(io: std.Io, d: Data, n_env: usize, n_threads: usize) !f64 {
    var best: i96 = std.math.maxInt(i96);
    for (0..7) |_| {
        const t0 = std.Io.Timestamp.now(io, .awake);
        try computeMaskThreaded(d, n_env, n_threads);
        std.mem.doNotOptimizeAway(d.mask.ptr);
        best = @min(best, t0.durationTo(std.Io.Timestamp.now(io, .awake)).nanoseconds);
    }
    return @as(f64, @floatFromInt(best)) / 1e9;
}

/// Split the envs into contiguous chunks, one per thread. Each thread
/// reads and writes only its own envs (no locks, no shared writes except
/// at chunk edges of `mask`). The calling thread does the last chunk.
/// Threads are spawned per batch, so small batches pay the spawn cost
/// (~10–50 µs); a persistent worker pool would remove it.
fn computeMaskThreaded(d: Data, n_env: usize, n_threads: usize) !void {
    const n = @min(n_threads, n_env);
    if (n <= 1) return computeMask(d, 0, n_env);

    var threads: [256]std.Thread = undefined;
    const per = (n_env + n - 1) / n;
    var spawned: usize = 0;
    var begin: usize = 0;
    while (begin + per < n_env and spawned < threads.len) : (begin += per) {
        threads[spawned] = try std.Thread.spawn(.{}, computeMask, .{ d, begin, begin + per });
        spawned += 1;
    }
    computeMask(d, begin, n_env);
    for (threads[0..spawned]) |t| t.join();
}

/// mask for envs [env_begin, env_end).
noinline fn computeMask(d: Data, env_begin: usize, env_end: usize) void {
    for (env_begin..env_end) |e| {
        for (0..J) |j| {
            const job = Usage.fromPool(d.jobs, d.layout, e * J + j);
            const du = d.dur[e * J + j];
            for (0..M) |m| {
                const mach = Usage.fromPool(d.machines, d.layout, e * M + m);
                d.mask[(e * J + j) * M + m] = mach.covers(job, 0, du);
            }
        }
    }
}
