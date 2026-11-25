//! Poisson arrivals: `total` jobs, exponential inter-arrival gaps with mean
//! 1/rate ticks (rate = expected jobs per tick). Several jobs can arrive in
//! the same tick. rate = 0 → all jobs at tick 0 (batch).
const std = @import("std");
const model = @import("../../model.zig");
const Error = @import("../../ports.zig").arrival_source.Error;

pub const Poisson = struct {
    total: usize,
    rate: f64,
    prng: std.Random.DefaultPrng = .init(0),
    generated: usize = 0,
    t: f64 = 0, // continuous arrival clock

    pub fn reset(self: *Poisson, seed: u64) void {
        self.prng = .init(seed);
        self.generated = 0;
        self.t = 0;
    }

    pub fn next(self: *Poisson, now: i64) Error!?model.JobMeta {
        if (self.generated >= self.total) return null;
        const at: i64 = @intFromFloat(@floor(self.t));
        if (at > now) return null;
        const meta = model.JobMeta{ .id = @intCast(self.generated), .arrival = at };
        self.generated += 1;
        if (self.rate > 0) self.t += self.prng.random().floatExp(f64) / self.rate;
        return meta;
    }

    pub fn done(self: *Poisson) bool {
        return self.generated >= self.total;
    }
};

test "poisson: count, order, determinism" {
    var p = Poisson{ .total = 50, .rate = 0.7 };
    p.reset(3);
    var last: i64 = 0;
    var n: usize = 0;
    var now: i64 = 0;
    while (!p.done()) : (now += 1) {
        while (try p.next(now)) |m| {
            try std.testing.expect(m.arrival <= now and m.arrival >= last);
            last = m.arrival;
            n += 1;
        }
    }
    try std.testing.expectEqual(@as(usize, 50), n);
    const end1 = last;
    p.reset(3);
    now = 0;
    while (!p.done()) : (now += 1) while (try p.next(now)) |m| {
        last = m.arrival;
    };
    try std.testing.expectEqual(end1, last);
}
