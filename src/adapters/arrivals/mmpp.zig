//! Bursty arrivals: a two-state Markov-modulated Poisson process (MMPP-2).
//! The source alternates between a calm state and a burst state; state
//! lengths are geometric with means `calm` and `burst_len` ticks; arrivals
//! per tick are Poisson with the state's rate. The burst rate is `ratio`
//! times the calm rate, and both are chosen so the long-run mean is `rate`
//! jobs per tick:
//!
//!     low  = rate·(calm + burst_len) / (calm + ratio·burst_len)
//!     high = ratio · low
//!
//! Short-range bursty (correlations die out after a few state changes);
//! compare `selfsimilar` for bursts at every time scale.
const std = @import("std");
const model = @import("../../model.zig");
const Error = @import("../../ports.zig").arrival_source.Error;
const sample = @import("sample.zig");

pub const Mmpp = struct {
    total: usize,
    rate: f64,
    ratio: f64 = 10,
    calm: f64 = 40,
    burst_len: f64 = 10,
    prng: std.Random.DefaultPrng = .init(0),
    generated: usize = 0,
    cur: i64 = 0, // tick whose arrivals are pending
    pending: usize = 0,
    in_burst: bool = false,

    pub fn reset(self: *Mmpp, seed: u64) void {
        self.prng = .init(seed +% 12);
        self.generated = 0;
        self.cur = 0;
        // start in the stationary distribution
        self.in_burst = self.prng.random().float(f64) < self.burst_len / (self.calm + self.burst_len);
        self.pending = self.draw();
    }

    fn draw(self: *Mmpp) usize {
        const r = self.prng.random();
        const low = self.rate * (self.calm + self.burst_len) / (self.calm + self.ratio * self.burst_len);
        const n = sample.poisson(r, if (self.in_burst) self.ratio * low else low);
        // state change at the end of the tick
        const mean = if (self.in_burst) self.burst_len else self.calm;
        if (r.float(f64) < 1.0 / @max(1.0, mean)) self.in_burst = !self.in_burst;
        return n;
    }

    pub fn next(self: *Mmpp, now: i64) Error!?model.JobMeta {
        if (self.rate <= 0 or self.ratio < 1) return error.InvalidInput;
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

    pub fn done(self: *Mmpp) bool {
        return self.generated >= self.total;
    }
};
