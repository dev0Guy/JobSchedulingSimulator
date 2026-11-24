//! Random sampling helpers shared by arrival sources.
const std = @import("std");

/// Poisson(λ) count. Knuth for small λ, normal approximation for large λ.
pub fn poisson(r: std.Random, lambda: f64) usize {
    if (lambda <= 0) return 0;
    if (lambda > 30) {
        const x = lambda + @sqrt(lambda) * r.floatNorm(f64);
        return @intFromFloat(@max(0, @round(x)));
    }
    const l = @exp(-lambda);
    var k: usize = 0;
    var p: f64 = 1;
    while (true) {
        p *= r.float(f64);
        if (p <= l) return k;
        k += 1;
    }
}

/// Pareto(α) sample with the given mean (α > 1): x_m / U^(1/α),
/// x_m = mean·(α−1)/α. Heavy-tailed: infinite variance for α ≤ 2.
pub fn pareto(r: std.Random, alpha: f64, mean: f64) f64 {
    const xm = mean * (alpha - 1) / alpha;
    const u = 1.0 - r.float(f64); // (0, 1]
    return xm / std.math.pow(f64, u, 1.0 / alpha);
}

test "sample means" {
    var prng = std.Random.DefaultPrng.init(1);
    const r = prng.random();
    var sp: f64 = 0;
    var sq: f64 = 0;
    const n = 200_000;
    for (0..n) |_| {
        sp += @floatFromInt(poisson(r, 2.5));
        sq += pareto(r, 2.5, 4.0); // finite variance → mean converges fast
    }
    try std.testing.expectApproxEqAbs(@as(f64, 2.5), sp / n, 0.05);
    try std.testing.expectApproxEqAbs(@as(f64, 4.0), sq / n, 0.1);
    try std.testing.expectEqual(@as(usize, 0), poisson(r, 0));
}
