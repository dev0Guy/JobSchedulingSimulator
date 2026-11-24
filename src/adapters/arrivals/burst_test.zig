//! Statistical checks for the arrival sources: mean rate, determinism,
//! and burstiness (index of dispersion = Var/Mean of counts per window;
//! Poisson ≈ 1, bursty sources ≫ 1, self-similar keeps growing with the
//! window size).
const std = @import("std");
const Poisson = @import("poisson.zig").Poisson;
const Mmpp = @import("mmpp.zig").Mmpp;
const ParetoArrivals = @import("pareto.zig").ParetoArrivals;
const SelfSimilar = @import("self_similar.zig").SelfSimilar;

const N = 60_000;
const RATE = 1.0;

/// Counts per tick for one run; returns the number of ticks used.
fn counts(src: anytype, seed: u64, out: []u32) !usize {
    src.reset(seed);
    @memset(out, 0);
    var now: i64 = 0;
    var last: i64 = -1;
    while (!src.done()) : (now += 1) {
        while (try src.next(now)) |m| {
            try std.testing.expect(m.arrival <= now and m.arrival >= last);
            last = m.arrival;
            out[@intCast(m.arrival)] += 1;
        }
    }
    return @intCast(now);
}

fn dispersion(c: []const u32, window: usize) f64 {
    const n = c.len / window;
    var sum: f64 = 0;
    var sq: f64 = 0;
    for (0..n) |w| {
        var x: f64 = 0;
        for (c[w * window ..][0..window]) |v| x += @floatFromInt(v);
        sum += x;
        sq += x * x;
    }
    const mean = sum / @as(f64, @floatFromInt(n));
    return (sq / @as(f64, @floatFromInt(n)) - mean * mean) / mean;
}

const Result = struct { rate: f64, d1: f64, d10: f64, d100: f64, d1000: f64 };

fn measure(src: anytype) !Result {
    const buf = try std.testing.allocator.alloc(u32, 4 * N);
    defer std.testing.allocator.free(buf);
    const ticks = try counts(src, 7, buf);
    // determinism
    const buf2 = try std.testing.allocator.alloc(u32, 4 * N);
    defer std.testing.allocator.free(buf2);
    _ = try counts(src, 7, buf2);
    try std.testing.expectEqualSlices(u32, buf, buf2);
    const c = buf[0..ticks];
    return .{
        .rate = @as(f64, @floatFromInt(N)) / @as(f64, @floatFromInt(ticks)),
        .d1 = dispersion(c, 1),
        .d10 = dispersion(c, 10),
        .d100 = dispersion(c, 100),
        .d1000 = dispersion(c, 1000),
    };
}

test "arrivals: mean rate, determinism and burstiness" {
    var p = Poisson{ .total = N, .rate = RATE };
    var m = Mmpp{ .total = N, .rate = RATE };
    var h = ParetoArrivals{ .total = N, .rate = RATE, .alpha = 1.3 };
    var s = SelfSimilar{ .total = N, .rate = RATE };
    const rp = try measure(&p);
    const rm = try measure(&m);
    const rh = try measure(&h);
    const rs = try measure(&s);

    // every source honours the mean rate (heavy tails converge slowly)
    try std.testing.expectApproxEqRel(RATE, rp.rate, 0.05);
    try std.testing.expectApproxEqRel(RATE, rm.rate, 0.15);
    try std.testing.expectApproxEqRel(RATE, rh.rate, 0.25);
    try std.testing.expectApproxEqRel(RATE, rs.rate, 0.15);

    // Poisson: dispersion ≈ 1 at every time scale
    try std.testing.expectApproxEqAbs(@as(f64, 1), rp.d1, 0.1);
    try std.testing.expect(rp.d1000 < 1.5);
    // MMPP: bursty, but short-range — dispersion grows, then levels off
    try std.testing.expect(rm.d10 > 3);
    try std.testing.expect(rm.d1000 < 2 * rm.d100);
    // heavy-tailed gaps: bursty and still growing at long scales
    try std.testing.expect(rh.d1000 > 2 * rh.d100);
    // self-similar: dispersion keeps growing with the window (long-range
    // dependence; d(m) ~ m^(2H-1), H = 0.8 for alpha = 1.4)
    try std.testing.expect(rs.d10 < rs.d100 and rs.d100 < rs.d1000);
    try std.testing.expect(rs.d1000 > 2 * rs.d100);
    try std.testing.expect(rs.d1000 > 10);
}
