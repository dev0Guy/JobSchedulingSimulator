//! Incremental (job, machine) scores for pair-scanning policies.
//!
//! Within one tick a placement changes only the machine it lands on (and
//! removes one job), so after the first decision of a tick only that
//! machine's column of scores needs recomputing. The cache keeps:
//!   * `score[s·n_m + m]`: score of slot s on machine m, NaN = does not fit;
//!   * `best[s]`: the first machine with the highest score in row s.
//! It is rebuilt in full when `Machines.epoch` changes (reset, time
//! advance) and per column when `Machines.stamp[m]` changes (placement).
//!
//! Results are bit-identical to scoring every pair each decision: the same
//! function computes each score, and ties resolve the same way (first
//! machine within a job; across jobs the strictly older job, else the
//! first slot). Higher scores win; negate to minimize.
const std = @import("std");
const model = @import("../../model.zig");
const rules = @import("../../engine.zig").rules;

const none = std.math.maxInt(u32);

pub const PairCache = struct {
    alloc: std.mem.Allocator = std.heap.page_allocator,
    score: []f32 = &.{},
    best: []u32 = &.{},
    ids: []u64 = &.{},
    seen: []u64 = &.{},
    epoch: u64 = std.math.maxInt(u64),
    n_m: usize = 0,

    pub fn deinit(self: *PairCache) void {
        self.free();
        self.* = .{ .alloc = self.alloc };
    }

    fn free(self: *PairCache) void {
        self.alloc.free(self.score);
        self.alloc.free(self.best);
        self.alloc.free(self.ids);
        self.alloc.free(self.seen);
    }

    fn ensure(self: *PairCache, n_s: usize, n_m: usize) !void {
        if (self.best.len == n_s and self.n_m == n_m) return;
        self.free();
        self.* = .{ .alloc = self.alloc };
        self.score = try self.alloc.alloc(f32, n_s * n_m);
        self.best = try self.alloc.alloc(u32, n_s);
        self.ids = try self.alloc.alloc(u64, n_s);
        self.seen = try self.alloc.alloc(u64, n_m);
        self.n_m = n_m;
    }

    pub fn row(self: *const PairCache, s: usize) []const f32 {
        return self.score[s * self.n_m ..][0..self.n_m];
    }

    /// Best machine of slot s, or null if it fits nowhere.
    pub fn bestOf(self: *const PairCache, s: usize) ?usize {
        return if (self.best[s] == none) null else self.best[s];
    }

    fn better(a: f32, b: f32) bool {
        return a > b;
    }

    fn rescan(self: *PairCache, s: usize) void {
        const r = self.row(s);
        var b: u32 = none;
        for (r, 0..) |x, m| {
            if (std.math.isNan(x)) continue;
            if (b == none or better(x, r[b])) b = @intCast(m);
        }
        self.best[s] = b;
    }

    fn fillRow(self: *PairCache, st: *const model.State, s: usize, comptime scoreFn: anytype, ctx: anytype) void {
        const r = self.score[s * self.n_m ..][0..self.n_m];
        if (rules.knownBlocked(st, s)) {
            @memset(r, std.math.nan(f32));
        } else for (0..self.n_m) |m| {
            r[m] = if (rules.fitsNow(st, s, m)) scoreFn(ctx, st, s, m) else std.math.nan(f32);
        }
        self.ids[s] = st.jobs.id[s];
        self.rescan(s);
    }

    /// Bring the cache up to date with `st`. `scoreFn(ctx, st, s, m) f32`
    /// is only called for pairs that fit now. Marks jobs that fit nowhere
    /// as blocked (`rules.markBlocked`).
    pub fn update(self: *PairCache, st: *const model.State, comptime scoreFn: anytype, ctx: anytype) !void {
        const ms = st.machines;
        const j = st.jobs;
        const full = self.best.len != j.cap or self.n_m != ms.n or self.epoch != ms.epoch;
        try self.ensure(j.cap, ms.n);
        if (full) {
            self.epoch = ms.epoch;
            @memcpy(self.seen, ms.stamp);
            var it = j.iter();
            while (it.next()) |s| self.fillRow(st, s, scoreFn, ctx);
        } else {
            // slots whose job changed (not expected within an epoch; safe anyway)
            var it = j.iter();
            while (it.next()) |s| if (self.ids[s] != j.id[s]) self.fillRow(st, s, scoreFn, ctx);
            for (0..ms.n) |m| {
                if (self.seen[m] == ms.stamp[m]) continue;
                self.seen[m] = ms.stamp[m];
                var it2 = j.iter();
                while (it2.next()) |s| {
                    const b = self.best[s];
                    if (b == none) continue; // capacity only shrinks within a tick
                    const cell = &self.score[s * self.n_m + m];
                    const old = cell.*;
                    cell.* = if (rules.fitsNow(st, s, m)) scoreFn(ctx, st, s, m) else std.math.nan(f32);
                    const new = cell.*;
                    if (b == m) {
                        if (std.math.isNan(new) or new < old) self.rescan(s);
                    } else if (!std.math.isNan(new)) {
                        const bs = self.score[s * self.n_m + b];
                        if (better(new, bs) or (new == bs and m < b)) self.best[s] = @intCast(m);
                    }
                }
            }
        }
        var it = j.iter();
        while (it.next()) |s| if (self.best[s] == none) rules.markBlocked(st, s);
    }

    /// Best (slot, machine) by `row score − pen(s)` (pen = 0 for none).
    pub fn pick(self: *const PairCache, st: *const model.State, comptime penFn: anytype, ctx: anytype) model.Action {
        const j = st.jobs;
        var best: ?struct { s: usize, score: f32 } = null;
        var it = j.iter();
        while (it.next()) |s| {
            const m = self.bestOf(s) orelse continue;
            // max over m of (x − pen) = (max x) − pen: subtraction is monotone
            const score = self.row(s)[m] - penFn(ctx, st, s);
            if (best) |b| {
                if (score < b.score) continue;
                if (score == b.score and j.arrival[s] >= j.arrival[b.s]) continue;
            }
            best = .{ .s = s, .score = score };
        }
        const b = best orelse return .wait;
        // rounding may make an earlier machine tie after subtracting pen
        const pen = penFn(ctx, st, b.s);
        const m = for (self.row(b.s), 0..) |x, m| {
            if (!std.math.isNan(x) and x - pen == b.score) break m;
        } else unreachable;
        return .{ .place = .{ .slot = @intCast(b.s), .machine = @intCast(m) } };
    }

    pub fn noPen(_: void, _: *const model.State, _: usize) f32 {
        return 0;
    }
};
