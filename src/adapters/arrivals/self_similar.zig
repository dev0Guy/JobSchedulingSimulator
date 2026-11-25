//! Self-similar arrivals: superposition of `sources` ON/OFF sources whose
//! ON and OFF periods are Pareto(alpha) with mean `period` ticks (Willinger
//! et al., "Self-similarity through high-variability", 1997). While ON a
//! source emits Poisson arrivals; the aggregate is long-range dependent with
//! Hurst parameter H = (3 − α)/2 (α = 1.4 → H = 0.8): bursts and lulls at
//! every time scale, as measured in real cluster and network traces.
//!
//! Fewer sources and longer periods = burstier: the ON-count fluctuation
//! adds rate²/sources of variance per tick. Defaults (8 sources, period 50)
//! give an index of dispersion of ~6 over 100 ticks and ~24 over 1000
//! (Poisson: 1 at every scale).
//!
//! Each source is ON half the time on average, so the per-source ON rate is
//! 2·rate/sources and the long-run mean is `rate` jobs per tick.
const std = @import("std");
const model = @import("../../model.zig");
const Error = @import("../../ports.zig").arrival_source.Error;
const sample = @import("sample.zig");

pub const max_sources = 256;

pub const SelfSimilar = struct {
    total: usize,
    rate: f64,
    sources: usize = 8,
    alpha: f64 = 1.4,
    period: f64 = 50,
    prng: std.Random.DefaultPrng = .init(0),
    generated: usize = 0,
    cur: i64 = 0,
    pending: usize = 0,
    on: [max_sources]bool = undefined,
    left: [max_sources]f64 = undefined, // ticks left in the current period

    pub fn reset(self: *SelfSimilar, seed: u64) void {
        self.prng = .init(seed +% 13);
        self.generated = 0;
        self.cur = 0;
        const r = self.prng.random();
        for (0..@min(self.sources, max_sources)) |i| {
            self.on[i] = r.boolean();
            self.left[i] = sample.pareto(r, self.alpha, self.period) * r.float(f64); // random phase
        }
        self.pending = self.draw();
    }

    fn draw(self: *SelfSimilar) usize {
        const r = self.prng.random();
        const k = @min(self.sources, max_sources);
        var n_on: usize = 0;
        for (0..k) |i| {
            if (self.on[i]) n_on += 1;
            // advance this source by one tick
            self.left[i] -= 1;
            while (self.left[i] <= 0) {
                self.on[i] = !self.on[i];
                self.left[i] += sample.pareto(r, self.alpha, self.period);
            }
        }
        const per_on = 2.0 * self.rate / @as(f64, @floatFromInt(k));
        return sample.poisson(r, per_on * @as(f64, @floatFromInt(n_on)));
    }

    pub fn next(self: *SelfSimilar, now: i64) Error!?model.JobMeta {
        if (self.rate <= 0 or self.alpha <= 1 or self.sources == 0 or self.sources > max_sources)
            return error.InvalidInput;
        while (self.generated < self.total and self.cur <= now) {
            if (self.pending > 0) {
                self.pending -= 1;
                const meta = model.JobMeta{ .id = @intCast(self.generated), .arrival = self.cur };
                self.generated += 1;
                return meta;
            }
            self.cur += 1;
            self.pending = self.draw();
        }
        return null;
    }

    pub fn done(self: *SelfSimilar) bool {
        return self.generated >= self.total;
    }
};
