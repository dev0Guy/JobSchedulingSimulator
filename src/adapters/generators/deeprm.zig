//! DeepRM-style workload (Mao et al., HotNets'16), adapted:
//!   * identical machines with capacity `cap` on every resource;
//!   * 80% short jobs (duration 1..short_max), 20% long (long_min..long_max);
//!   * one random dominant resource with demand in [0.25, 0.5]·cap,
//!     the others in [0.05, 0.1]·cap; constant over the job's duration.
const std = @import("std");
const grid = @import("../../grid.zig");
const model = @import("../../model.zig");
const Error = @import("../../ports.zig").usage_generator.Error;
const Elem = grid.Elem;
const Usage = grid.Usage;

pub const DeepRM = struct {
    n_machines: usize = 1,
    cap: Elem = 100,
    short_prob: f64 = 0.8,
    short_max: u16 = 3,
    long_min: u16 = 10,
    long_max: u16 = 15,
    prng: std.Random.DefaultPrng = .init(0),

    pub fn reset(self: *DeepRM, seed: u64) void {
        self.prng = .init(seed +% 2);
    }

    pub fn machineCount(self: *DeepRM) usize {
        return self.n_machines;
    }

    pub fn machine(self: *DeepRM, index: usize, cap: []Elem) Error!void {
        _ = index;
        @memset(cap, self.cap);
    }

    pub fn job(self: *DeepRM, meta: model.JobMeta, out: Usage) Error!u16 {
        _ = meta;
        const r = self.prng.random();
        const n_ticks: u16 = @intCast(out.layout.n_ticks);
        const dur = @min(n_ticks, if (r.float(f64) < self.short_prob)
            r.intRangeAtMost(u16, 1, self.short_max)
        else
            r.intRangeAtMost(u16, self.long_min, self.long_max));

        const c: f64 = @floatFromInt(self.cap);
        const dom = r.uintLessThan(usize, out.layout.n_res);
        for (0..out.layout.n_res) |res| {
            const frac = if (res == dom) 0.25 + 0.25 * r.float(f64) else 0.05 + 0.05 * r.float(f64);
            const d: Elem = @intFromFloat(@round(frac * c));
            @memset(out.resourceMut(res)[0..dur], d);
        }
        return dur;
    }
};
