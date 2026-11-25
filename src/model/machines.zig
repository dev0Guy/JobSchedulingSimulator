//! Machines: one pool owning N grids of FREE capacity (cap - used),
//! laid out [machine][res][stride], plus `Machine`, a per-machine view with
//! the fit test and placement operations.
//!
//! Time is a sliding window. Each row stores `n_ticks + slack` ticks
//! (the `ring` layout); the horizon [now, now + n_ticks) is the window
//! starting at `head`:
//!
//!     row r: | stale … | head: tick 0 … tick n_ticks-1 | cap cap cap … | pad |
//!
//! Advancing time is `head += k`: no data moves. Every slot at or after
//! head + n_ticks is always full capacity, so newly visible ticks are
//! already correct. When the window would run past the slack, the live
//! ticks are copied back to the start of the row once (`compact`), i.e.
//! one row copy every `slack` ticks: O(1) amortized per tick instead of
//! O(n_ticks). Windows stay contiguous, so the SIMD fit test is unchanged.

const std = @import("std");
const grid = @import("../grid.zig");
const simd = grid.simd;
const Elem = grid.Elem;
const Layout = grid.Layout;
const Usage = grid.Usage;

/// View of one machine. Owns nothing. Tick t of the horizon is
/// `free.row(r)[head + t]`.
pub const Machine = struct {
    free: Usage, // ring layout
    cap: []const Elem, // [n_res]
    head: usize,
    horizon: usize, // n_ticks

    /// Empty machine: every slot, slack and padding included, at full capacity.
    pub fn reset(self: Machine) void {
        for (0..self.free.layout.n_res) |r| @memset(self.free.row(r), self.cap[r]);
    }

    /// Free capacity of resource r over the horizon, ticks [0, n_ticks).
    pub fn resource(self: Machine, r: usize) []const Elem {
        return self.free.row(r)[self.head..][0..self.horizon];
    }

    /// Cheap necessary conditions: peak within capacity, first tick within
    /// free capacity at tick t. No grid scan.
    pub fn mayFit(self: Machine, job: Usage, peak: []const Elem, t: usize) bool {
        for (0..self.free.layout.n_res) |r| {
            if (peak[r] > self.cap[r]) return false;
            if (job.row(r)[0] > self.free.row(r)[self.head + t]) return false;
        }
        return true;
    }

    /// Could a job with this peak ever run here (on an empty machine)?
    pub fn canEverFit(self: Machine, peak: []const Elem) bool {
        for (self.cap, peak) |c, p| if (p > c) return false;
        return true;
    }

    /// Exact test: horizon + SIMD comparison against free capacity.
    pub fn canPlace(self: Machine, job: Usage, dur: usize, t: usize) bool {
        if (t + dur > self.horizon) return false;
        return self.free.covers(job, self.head + t, dur);
    }

    pub fn place(self: Machine, job: Usage, dur: usize, t: usize) void {
        std.debug.assert(self.canPlace(job, dur, t));
        self.free.subAt(job, self.head + t, dur);
    }

    pub fn unplace(self: Machine, job: Usage, dur: usize, t: usize) void {
        self.free.addAt(job, self.head + t, dur);
    }

    /// Earliest start tick at which the job fits, or null.
    pub fn earliestFit(self: Machine, job: Usage, dur: usize) ?usize {
        if (dur > self.horizon) return null;
        var t: usize = 0;
        while (t + dur <= self.horizon) : (t += 1)
            if (self.free.covers(job, self.head + t, dur)) return t;
        return null;
    }

    pub fn freeNow(self: Machine, r: usize) Elem {
        return self.free.row(r)[self.head];
    }

    /// Mean over resources of used/cap at tick 0, in [0, 1].
    pub fn utilizationNow(self: Machine) f32 {
        const n_res = self.free.layout.n_res;
        var s: f32 = 0;
        for (0..n_res) |r| {
            const c: f32 = @floatFromInt(self.cap[r]);
            if (c > 0) s += 1.0 - @as(f32, @floatFromInt(self.freeNow(r))) / c;
        }
        return s / @as(f32, @floatFromInt(n_res));
    }

    /// True if nothing is running (every tick of the horizon at full capacity).
    pub fn idle(self: Machine) bool {
        for (0..self.free.layout.n_res) |r|
            for (self.resource(r)) |v| if (v != self.cap[r]) return false;
        return true;
    }

    /// Move the live window [head, head + horizon) to the start of the row
    /// and refill everything after it with capacity.
    fn compact(self: Machine) void {
        const stride = self.free.layout.stride;
        const keep = @min(self.horizon, stride -| self.head);
        for (0..self.free.layout.n_res) |r| {
            const rw = self.free.row(r);
            std.mem.copyForwards(Elem, rw[0..keep], rw[self.head..][0..keep]);
            @memset(rw[keep..], self.cap[r]);
        }
    }
};

pub const Machines = struct {
    /// Logical layout: n_res, n_ticks (horizon). Shared with jobs.
    layout: Layout,
    /// Storage layout of each machine: n_ticks + slack ticks per row.
    ring: Layout,
    n: usize,
    buf: []align(simd.CACHE) Elem,
    cap: []Elem, // [machine * n_res + r]
    /// Start of the horizon inside every row (all machines share it).
    head: usize = 0,
    /// Change stamps for caches kept across decisions (see
    /// adapters/policies/pair_cache.zig). `epoch` changes whenever every
    /// machine may have changed (reset, time advance); `stamp[m]` changes
    /// when machine m alone changed (`touch`, called by the engine on every
    /// placement). Both come from one counter that never repeats.
    epoch: u64 = 0,
    stamp: []u64,
    clock: u64 = 0,

    /// `caps` is flat [n_machines][n_res].
    pub fn init(a: std.mem.Allocator, layout: Layout, n_machines: usize, caps: []const Elem) !Machines {
        std.debug.assert(n_machines > 0);
        std.debug.assert(caps.len == n_machines * layout.n_res);
        // slack = horizon (at least a few vectors): one compaction per
        // n_ticks advances, row memory doubled.
        const slack = @max(layout.n_ticks, 4 * layout.padding.lanes);
        const ring = try Layout.init(layout.n_res, layout.n_ticks + slack, layout.padding);
        const buf = try a.alignedAlloc(Elem, simd.ALIGN, n_machines * ring.block);
        errdefer a.free(buf);
        const cap = try a.dupe(Elem, caps);
        errdefer a.free(cap);
        const stamp = try a.alloc(u64, n_machines);
        @memset(stamp, 0);
        var self = Machines{ .layout = layout, .ring = ring, .n = n_machines, .buf = buf, .cap = cap, .stamp = stamp };
        self.resetAll();
        return self;
    }

    pub fn deinit(self: *Machines, a: std.mem.Allocator) void {
        a.free(self.buf);
        a.free(self.cap);
        a.free(self.stamp);
    }

    fn tick(self: *Machines) u64 {
        self.clock += 1;
        return self.clock;
    }

    /// Record that machine m changed (a job was placed on it).
    pub fn touch(self: *Machines, m: usize) void {
        self.stamp[m] = self.tick();
    }

    pub fn get(self: Machines, i: usize) Machine {
        std.debug.assert(i < self.n);
        return .{
            .free = Usage.fromPool(self.buf, self.ring, i),
            .cap = self.cap[i * self.layout.n_res ..][0..self.layout.n_res],
            .head = self.head,
            .horizon = self.layout.n_ticks,
        };
    }

    pub fn resetAll(self: *Machines) void {
        self.head = 0;
        self.epoch = self.tick();
        for (0..self.n) |i| self.get(i).reset();
    }

    /// Global time skip: every machine advances k ticks together.
    /// O(1) unless the window reaches the end of the slack.
    pub fn advanceAll(self: *Machines, k: usize) void {
        self.head += k;
        self.epoch = self.tick();
        if (self.head + self.layout.n_ticks <= self.ring.n_ticks) return;
        for (0..self.n) |i| self.get(i).compact();
        self.head = 0;
    }

    pub fn anyCanEverFit(self: Machines, peak: []const Elem) bool {
        for (0..self.n) |i| if (self.get(i).canEverFit(peak)) return true;
        return false;
    }

    pub fn allIdle(self: Machines) bool {
        for (0..self.n) |i| if (!self.get(i).idle()) return false;
        return true;
    }

    pub fn utilizationNow(self: Machines) f32 {
        var s: f32 = 0;
        for (0..self.n) |i| s += self.get(i).utilizationNow();
        return s / @as(f32, @floatFromInt(self.n));
    }

    /// Largest capacity of resource r over all machines.
    pub fn maxCap(self: Machines, r: usize) Elem {
        var m: Elem = 0;
        for (0..self.n) |i| m = @max(m, self.cap[i * self.layout.n_res + r]);
        return m;
    }
};

test "machines: place, earliestFit, advance" {
    const a = std.testing.allocator;
    const layout = try Layout.init(2, 10, simd.padding);
    var ms = try Machines.init(a, layout, 2, &.{ 10, 10, 5, 5 });
    defer ms.deinit(a);

    const jb = try a.alignedAlloc(Elem, simd.ALIGN, layout.block);
    defer a.free(jb);
    @memset(jb, 0);
    const job = Usage.fromSlice(jb, layout);
    job.set(0, &.{ 8, 8 });
    job.set(1, &.{ 1, 1 });

    try std.testing.expect(!ms.get(1).canPlace(job, 2, 0));
    try std.testing.expect(ms.get(0).canPlace(job, 2, 0));
    ms.get(0).place(job, 2, 0);
    try std.testing.expectEqual(@as(Elem, 2), ms.get(0).freeNow(0));
    try std.testing.expectEqual(@as(?usize, 2), ms.get(0).earliestFit(job, 2));
    try std.testing.expect(!ms.allIdle());
    ms.advanceAll(2);
    try std.testing.expectEqual(@as(Elem, 10), ms.get(0).freeNow(0));
    try std.testing.expect(ms.allIdle());
}

test "machines: sliding window matches a shifted reference across compactions" {
    const a = std.testing.allocator;
    const n_ticks = 7;
    const layout = try Layout.init(2, n_ticks, simd.padding);
    var ms = try Machines.init(a, layout, 1, &.{ 20, 9 });
    defer ms.deinit(a);

    const jb = try a.alignedAlloc(Elem, simd.ALIGN, layout.block);
    defer a.free(jb);

    // reference: plain arrays shifted by hand
    var ref: [2][n_ticks]Elem = .{ @splat(20), @splat(9) };
    var prng = std.Random.DefaultPrng.init(5);
    const rnd = prng.random();

    for (0..10 * ms.ring.n_ticks) |_| {
        // try to place a random job somewhere in the horizon
        @memset(jb, 0);
        const job = Usage.fromSlice(jb, layout);
        const dur = rnd.intRangeAtMost(usize, 1, n_ticks);
        for (0..2) |r| @memset(job.resourceMut(r)[0..dur], rnd.intRangeAtMost(Elem, 0, 6));
        const t = rnd.uintLessThan(usize, n_ticks - dur + 1);
        const m = ms.get(0);
        var fits = true;
        for (0..2) |r| for (0..dur) |k| {
            if (job.get(r, k) > ref[r][t + k]) fits = false;
        };
        try std.testing.expectEqual(fits, m.canPlace(job, dur, t));
        if (fits) {
            m.place(job, dur, t);
            for (0..2) |r| for (0..dur) |k| {
                ref[r][t + k] -= job.get(r, k);
            };
        }
        // advance 1 or 2 ticks
        const k = rnd.intRangeAtMost(usize, 1, 2);
        ms.advanceAll(k);
        for (0..2) |r| {
            std.mem.copyForwards(Elem, ref[r][0 .. n_ticks - k], ref[r][k..]);
            @memset(ref[r][n_ticks - k ..], ms.cap[r]);
            try std.testing.expectEqualSlices(Elem, &ref[r], ms.get(0).resource(r));
        }
    }

    // a jump longer than the horizon empties the machine
    ms.advanceAll(3 * n_ticks);
    try std.testing.expect(ms.allIdle());
}
