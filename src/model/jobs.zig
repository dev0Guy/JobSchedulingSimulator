//! Jobs: fixed slot pool [slot][res][stride] (== observation buffer),
//! struct-of-arrays metadata, a validity bitmask, and a doubly linked list
//! of the waiting jobs in (arrival, slot) order — so FIFO-style policies
//! visit the oldest job first and can stop early instead of scanning the
//! whole pool.
//!
//! Adding a job is two-phase so a generator can write the profile directly
//! into the pool without copies:
//!     const slot = jobs.acquire() orelse …;   // cleared, not yet visible
//!     … write profile into jobs.get(slot) …
//!     jobs.commit(slot, meta, dur);           // enforce invariants, peaks, visible
//! or jobs.abort(slot) to give the slot back.

const std = @import("std");
const grid = @import("../grid.zig");
const types = @import("types.zig");
const simd = grid.simd;
const Elem = grid.Elem;
const Layout = grid.Layout;
const Usage = grid.Usage;

pub const Jobs = struct {
    layout: Layout,
    cap: usize,
    buf: []align(simd.CACHE) Elem,
    duration: []u16,
    arrival: []i64,
    id: []u32,
    peak: []Elem, // [slot * n_res + r]
    valid: []u64, // committed, waiting jobs
    reserved: []u64, // acquired but not committed
    /// Memo: waiting jobs known to fit on no machine this tick. Capacity only
    /// shrinks within a tick, so the bit stays true until the engine clears
    /// it when time advances. Written through `*const Jobs` (slice), like a
    /// cache: it never changes what the simulation does.
    blocked: []u64,
    count: usize = 0,
    // waiting jobs ordered by (arrival, slot); none = end of list
    prev: []u32,
    next: []u32,
    head: u32 = none,
    tail: u32 = none,

    pub const none = std.math.maxInt(u32);

    pub fn init(a: std.mem.Allocator, layout: Layout, max_jobs: usize) !Jobs {
        std.debug.assert(max_jobs > 0);
        const words = (max_jobs + 63) / 64;
        const buf = try a.alignedAlloc(Elem, simd.ALIGN, max_jobs * layout.block);
        errdefer a.free(buf);
        const duration = try a.alloc(u16, max_jobs);
        errdefer a.free(duration);
        const arrival = try a.alloc(i64, max_jobs);
        errdefer a.free(arrival);
        const id = try a.alloc(u32, max_jobs);
        errdefer a.free(id);
        const peak = try a.alloc(Elem, max_jobs * layout.n_res);
        errdefer a.free(peak);
        const valid = try a.alloc(u64, words);
        errdefer a.free(valid);
        const reserved = try a.alloc(u64, words);
        errdefer a.free(reserved);
        const blocked = try a.alloc(u64, words);
        errdefer a.free(blocked);
        const prev = try a.alloc(u32, max_jobs);
        errdefer a.free(prev);
        const next = try a.alloc(u32, max_jobs);

        var self = Jobs{
            .layout = layout,
            .cap = max_jobs,
            .buf = buf,
            .duration = duration,
            .arrival = arrival,
            .id = id,
            .peak = peak,
            .valid = valid,
            .reserved = reserved,
            .blocked = blocked,
            .prev = prev,
            .next = next,
        };
        self.clearAll();
        return self;
    }

    pub fn deinit(self: *Jobs, a: std.mem.Allocator) void {
        a.free(self.buf);
        a.free(self.duration);
        a.free(self.arrival);
        a.free(self.id);
        a.free(self.peak);
        a.free(self.valid);
        a.free(self.reserved);
        a.free(self.blocked);
        a.free(self.prev);
        a.free(self.next);
    }

    /// Empty the pool.
    pub fn clearAll(self: *Jobs) void {
        @memset(self.buf, 0);
        @memset(self.valid, 0);
        @memset(self.reserved, 0);
        @memset(self.blocked, 0);
        self.count = 0;
        self.head = none;
        self.tail = none;
    }

    pub fn get(self: Jobs, slot: usize) Usage {
        return Usage.fromPool(self.buf, self.layout, slot);
    }

    pub fn isValid(self: Jobs, slot: usize) bool {
        return testBit(self.valid, slot);
    }

    pub fn peakOf(self: Jobs, slot: usize) []const Elem {
        return self.peak[slot * self.layout.n_res ..][0..self.layout.n_res];
    }

    pub fn full(self: Jobs) bool {
        return self.findFree() == null;
    }

    // ---------------- two-phase add ----------------

    /// Reserve a free, cleared slot. Not visible to iterators yet.
    pub fn acquire(self: *Jobs) ?usize {
        const slot = self.findFree() orelse return null;
        setBit(self.reserved, slot);
        clearBit(self.blocked, slot);
        self.get(slot).clear();
        return slot;
    }

    /// Give back an acquired slot without adding a job.
    pub fn abort(self: *Jobs, slot: usize) void {
        std.debug.assert(testBit(self.reserved, slot));
        clearBit(self.reserved, slot);
        self.get(slot).clear();
    }

    /// Make an acquired slot a waiting job. Enforces the pool invariants:
    /// zero beyond `dur`, then computes peaks.
    pub fn commit(self: *Jobs, slot: usize, meta: types.JobMeta, dur: u16) void {
        std.debug.assert(testBit(self.reserved, slot));
        std.debug.assert(dur >= 1 and dur <= self.layout.n_ticks);
        const u = self.get(slot);
        for (0..self.layout.n_res) |r| @memset(u.row(r)[dur..], 0);

        self.duration[slot] = dur;
        self.arrival[slot] = meta.arrival;
        self.id[slot] = meta.id;
        u.peak(dur, self.peak[slot * self.layout.n_res ..][0..self.layout.n_res]);

        clearBit(self.reserved, slot);
        setBit(self.valid, slot);
        self.count += 1;
        self.link(@intCast(slot));
    }

    /// One-shot add from a flat [n_res][dur] profile (tests, simple sources).
    pub fn add(self: *Jobs, meta: types.JobMeta, req: []const Elem, dur: u16) ?usize {
        std.debug.assert(req.len == self.layout.n_res * dur);
        const slot = self.acquire() orelse return null;
        const u = self.get(slot);
        for (0..self.layout.n_res) |r| u.set(r, req[r * dur ..][0..dur]);
        self.commit(slot, meta, dur);
        return slot;
    }

    /// Remove a waiting job (placed or dropped).
    pub fn remove(self: *Jobs, slot: usize) void {
        std.debug.assert(self.isValid(slot));
        clearBit(self.valid, slot);
        clearBit(self.blocked, slot);
        self.get(slot).clear();
        self.count -= 1;
        self.unlink(@intCast(slot));
    }

    // ---------------- per-tick "fits nowhere" memo ----------------

    pub inline fn isBlocked(self: Jobs, slot: usize) bool {
        return testBit(self.blocked, slot);
    }
    pub inline fn markBlocked(self: Jobs, slot: usize) void {
        setBit(self.blocked, slot);
    }
    /// Called by the engine when time advances (capacity may have freed up).
    pub fn clearBlocked(self: Jobs) void {
        @memset(self.blocked, 0);
    }

    // ---------------- iteration ----------------

    pub const Iter = struct {
        valid: []const u64,
        word: usize = 0,
        bits: u64,

        pub fn next(it: *Iter) ?usize {
            while (it.bits == 0) {
                it.word += 1;
                if (it.word >= it.valid.len) return null;
                it.bits = it.valid[it.word];
            }
            const b = @ctz(it.bits);
            it.bits &= it.bits - 1;
            return it.word * 64 + b;
        }
    };

    /// Waiting jobs in slot order.
    pub fn iter(self: Jobs) Iter {
        return .{ .valid = self.valid, .bits = self.valid[0] };
    }

    /// Slot of the oldest waiting job (lowest arrival, then lowest slot). O(1).
    pub fn oldest(self: Jobs) ?usize {
        return if (self.head == none) null else self.head;
    }

    pub const ArrivalIter = struct {
        next_slot: u32,
        links: []const u32,

        pub fn next(it: *ArrivalIter) ?usize {
            if (it.next_slot == none) return null;
            const s = it.next_slot;
            it.next_slot = it.links[s];
            return s;
        }
    };

    /// Waiting jobs, oldest first: (arrival, slot) order.
    pub fn byArrival(self: Jobs) ArrivalIter {
        return .{ .next_slot = self.head, .links = self.next };
    }

    /// Insert at its (arrival, slot) position. Arrivals come in time order,
    /// so this walks back over at most the jobs of the current tick.
    fn link(self: *Jobs, s: u32) void {
        var after = self.tail; // insert after this node
        while (after != none and self.before(s, after)) after = self.prev[after];
        const before_node = if (after == none) self.head else self.next[after];
        self.prev[s] = after;
        self.next[s] = before_node;
        if (after == none) self.head = s else self.next[after] = s;
        if (before_node == none) self.tail = s else self.prev[before_node] = s;
    }

    fn unlink(self: *Jobs, s: u32) void {
        const p = self.prev[s];
        const n = self.next[s];
        if (p == none) self.head = n else self.next[p] = n;
        if (n == none) self.tail = p else self.prev[n] = p;
    }

    /// Does slot a come before slot b in (arrival, slot) order?
    inline fn before(self: Jobs, a: u32, b: u32) bool {
        return self.arrival[a] < self.arrival[b] or (self.arrival[a] == self.arrival[b] and a < b);
    }

    // ---------------- bits ----------------

    fn findFree(self: Jobs) ?usize {
        for (self.valid, self.reserved, 0..) |v, r, i| {
            const used = v | r;
            if (used != ~@as(u64, 0)) {
                const slot = i * 64 + @ctz(~used);
                return if (slot < self.cap) slot else null;
            }
        }
        return null;
    }

    inline fn testBit(m: []const u64, i: usize) bool {
        return (m[i / 64] >> @intCast(i % 64)) & 1 == 1;
    }
    inline fn setBit(m: []u64, i: usize) void {
        m[i / 64] |= @as(u64, 1) << @intCast(i % 64);
    }
    inline fn clearBit(m: []u64, i: usize) void {
        m[i / 64] &= ~(@as(u64, 1) << @intCast(i % 64));
    }
};

test "jobs: two-phase add, iterate, remove" {
    const a = std.testing.allocator;
    const layout = try Layout.init(2, 8, simd.padding);
    var jobs = try Jobs.init(a, layout, 70);
    defer jobs.deinit(a);

    const s0 = jobs.add(.{ .id = 7, .arrival = 3 }, &.{ 1, 2, 3, 4 }, 2).?;
    try std.testing.expectEqualSlices(Elem, &.{ 2, 4 }, jobs.peakOf(s0));

    const s1 = jobs.acquire().?;
    try std.testing.expect(s1 != s0);
    jobs.get(s1).set(0, &.{ 9, 9, 9 });
    jobs.commit(s1, .{ .id = 8, .arrival = 1 }, 1); // tick 1,2 must be zeroed
    try std.testing.expectEqual(@as(Elem, 0), jobs.get(s1).get(0, 1));
    try std.testing.expectEqual(@as(?usize, s1), jobs.oldest());

    const s2 = jobs.acquire().?;
    jobs.abort(s2);

    var n: usize = 0;
    var it = jobs.iter();
    while (it.next()) |_| n += 1;
    try std.testing.expectEqual(@as(usize, 2), n);

    jobs.remove(s0);
    try std.testing.expectEqual(@as(usize, 1), jobs.count);
}

test "jobs: arrival order list stays sorted by (arrival, slot) under churn" {
    const a = std.testing.allocator;
    const layout = try Layout.init(1, 4, simd.padding);
    var jobs = try Jobs.init(a, layout, 40);
    defer jobs.deinit(a);
    var prng = std.Random.DefaultPrng.init(9);
    const r = prng.random();
    var t: i64 = 0;
    for (0..2000) |i| {
        // mostly in-order arrivals, sometimes out of order (allowed by the API)
        if (r.uintLessThan(u32, 3) == 0) t += 1;
        const at = if (r.uintLessThan(u32, 10) == 0) t - r.intRangeAtMost(i64, 0, 3) else t;
        if (!jobs.full() and r.boolean()) {
            _ = jobs.add(.{ .id = @intCast(i), .arrival = at }, &.{1}, 1);
        } else if (jobs.count > 0) {
            // remove a random waiting job
            var k = r.uintLessThan(usize, jobs.count);
            var it = jobs.iter();
            while (it.next()) |s| {
                if (k == 0) {
                    jobs.remove(s);
                    break;
                }
                k -= 1;
            }
        }
        // list == valid set, in (arrival, slot) order; oldest() == head
        var n: usize = 0;
        var last: ?usize = null;
        var it = jobs.byArrival();
        while (it.next()) |s| : (n += 1) {
            try std.testing.expect(jobs.isValid(s));
            if (last) |l| try std.testing.expect(jobs.arrival[l] < jobs.arrival[s] or
                (jobs.arrival[l] == jobs.arrival[s] and l < s));
            last = s;
        }
        try std.testing.expectEqual(jobs.count, n);
        // oldest() matches a brute-force scan
        var best: ?usize = null;
        var it2 = jobs.iter();
        while (it2.next()) |s| {
            if (best == null or jobs.arrival[s] < jobs.arrival[best.?]) best = s;
        }
        try std.testing.expectEqual(best, jobs.oldest());
    }
}
