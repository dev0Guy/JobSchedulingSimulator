//! Random profiles: machines with given capacities (cycled), jobs with a
//! uniform random duration and per-resource demand.
//!   * constant: one demand value per resource for the whole duration
//!   * varying : demand changes at 1–3 random points in time
const std = @import("std");
const grid = @import("../../grid.zig");
const model = @import("../../model.zig");
const Error = @import("../../ports.zig").usage_generator.Error;
const Elem = grid.Elem;
const Usage = grid.Usage;

pub const RandomProfiles = struct {
    n_machines: usize = 4,
    /// Capacity per machine, cycled: machine i gets caps[i % caps.len]
    /// on every resource.
    caps: []const Elem = &.{ 100, 80, 60, 40 },
    dur_min: u16 = 1,
    dur_max: u16 = 20,
    demand_min: Elem = 0,
    demand_max: Elem = 30,
    varying: bool = false,
    prng: std.Random.DefaultPrng = .init(0),

    pub fn reset(self: *RandomProfiles, seed: u64) void {
        self.prng = .init(seed +% 1);
    }

    pub fn machineCount(self: *RandomProfiles) usize {
        return self.n_machines;
    }

    pub fn machine(self: *RandomProfiles, index: usize, cap: []Elem) Error!void {
        if (self.caps.len == 0) return error.InvalidInput;
        @memset(cap, self.caps[index % self.caps.len]);
    }

    pub fn job(self: *RandomProfiles, meta: model.JobMeta, out: Usage) Error!u16 {
        _ = meta;
        const r = self.prng.random();
        const hi: u16 = @intCast(@min(self.dur_max, out.layout.n_ticks));
        const lo = @min(self.dur_min, hi);
        const dur = r.intRangeAtMost(u16, lo, hi);
        for (0..out.layout.n_res) |res| {
            const line = out.resourceMut(res)[0..dur];
            if (!self.varying) {
                @memset(line, r.intRangeAtMost(Elem, self.demand_min, self.demand_max));
                continue;
            }
            var t: usize = 0;
            while (t < dur) {
                const seg = r.intRangeAtMost(usize, 1, @max(1, dur / 2));
                const end = @min(dur, t + seg);
                @memset(line[t..end], r.intRangeAtMost(Elem, self.demand_min, self.demand_max));
                t = end;
            }
        }
        return dur;
    }
};
