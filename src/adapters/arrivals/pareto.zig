//! Heavy-tailed arrivals: `total` jobs, inter-arrival gaps ~ Pareto(alpha)
//! with mean 1/rate ticks. Same mean rate as Poisson, but with α ≤ 2 the
//! gap variance is infinite: long silences followed by dense clusters.
//! α → ∞ approaches evenly spaced arrivals; α ≈ 1.2–1.6 is very bursty.
const std = @import("std");
const model = @import("../../model.zig");
const Error = @import("../../ports.zig").arrival_source.Error;
const sample = @import("sample.zig");

pub const ParetoArrivals = struct {
    total: usize,
    rate: f64,
    alpha: f64 = 1.5,
    prng: std.Random.DefaultPrng = .init(0),
    generated: usize = 0,
    t: f64 = 0,

    pub fn reset(self: *ParetoArrivals, seed: u64) void {
        self.prng = .init(seed +% 11);
        self.generated = 0;
        self.t = 0;
    }

    pub fn next(self: *ParetoArrivals, now: i64) Error!?model.JobMeta {
        if (self.generated >= self.total) return null;
        if (self.rate <= 0 or self.alpha <= 1) return error.InvalidInput;
        const at: i64 = @intFromFloat(@floor(self.t));
        if (at > now) return null;
        const meta = model.JobMeta{ .id = @intCast(self.generated), .arrival = at };
        self.generated += 1;
        self.t += sample.pareto(self.prng.random(), self.alpha, 1.0 / self.rate);
        return meta;
    }

    pub fn done(self: *ParetoArrivals) bool {
        return self.generated >= self.total;
    }
};
