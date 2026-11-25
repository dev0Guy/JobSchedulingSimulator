//! Usage: CPU view of one [resource][stride] grid, with SIMD operations.
//!
//! Works on any buffer the CPU can read whose Layout padding is compatible
//! with the CPU SIMD rule: plain RAM (simd.padding) or unified CPU+GPU
//! memory (simd_gpu.unified_padding). A GPU-only buffer is not accessible
//! through Usage; GPU backends operate on whole pools with kernels instead.
//!
//! Elements are unsigned (Elem = u16). Subtraction below zero is a bug:
//! Debug builds panic, release builds are undefined. Callers must check
//! first (e.g. Machine.canPlace before subAt).

const std = @import("std");
const simd = @import("simd.zig");
const types = @import("types.zig");
const Layout = @import("layout.zig").Layout;

const Elem = types.Elem;
const L = simd.L;
const V = simd.V;
const load = simd.load;
const store = simd.store;
const span = simd.span;

pub const Usage = struct {
    ptr: [*]align(simd.CACHE) Elem,
    layout: Layout,

    /// True if CPU SIMD code may operate on grids with this layout.
    pub fn compatible(layout: Layout) bool {
        return layout.stride % L == 0 and layout.padding.align_bytes % simd.CACHE == 0;
    }

    /// View the grid that starts at buf[0].
    pub fn fromSlice(buf: []align(simd.CACHE) Elem, layout: Layout) Usage {
        std.debug.assert(compatible(layout));
        std.debug.assert(buf.len >= layout.block);
        return .{ .ptr = buf.ptr, .layout = layout };
    }

    /// View grid i of a pool laid out [grid][res][stride].
    pub fn fromPool(pool: []align(simd.CACHE) Elem, layout: Layout, i: usize) Usage {
        std.debug.assert(compatible(layout));
        const off = layout.gridOffset(i);
        std.debug.assert(off + layout.block <= pool.len);
        return .{ .ptr = @alignCast(pool[off..].ptr), .layout = layout };
    }

    /// Whole grid, padding included.
    pub fn all(self: Usage) []Elem {
        return self.ptr[0..self.layout.block];
    }

    /// Full padded row r.
    pub inline fn row(self: Usage, r: usize) []Elem {
        std.debug.assert(r < self.layout.n_res);
        return self.ptr[self.layout.rowOffset(r)..][0..self.layout.stride];
    }

    pub fn clear(self: Usage) void {
        @memset(self.all(), 0);
    }

    // ---------------- element / resource access ----------------

    pub inline fn get(self: Usage, r: usize, t: usize) Elem {
        std.debug.assert(t < self.layout.n_ticks);
        return self.row(r)[t];
    }

    /// Usage of resource r through time: real ticks [0, n_ticks).
    pub fn resource(self: Usage, r: usize) []const Elem {
        return self.row(r)[0..self.layout.n_ticks];
    }

    pub fn resourceMut(self: Usage, r: usize) []Elem {
        return self.row(r)[0..self.layout.n_ticks];
    }

    /// Copy values into resource r's line, ticks [0, values.len).
    pub fn set(self: Usage, r: usize, values: []const Elem) void {
        std.debug.assert(values.len <= self.layout.n_ticks);
        @memcpy(self.row(r)[0..values.len], values);
    }

    // ---------------- whole-grid arithmetic ----------------

    pub fn sub(self: Usage, other: Usage) void {
        std.debug.assert(self.layout.eql(other.layout));
        self.subAt(other, 0, self.layout.stride);
    }

    pub fn add(self: Usage, other: Usage) void {
        std.debug.assert(self.layout.eql(other.layout));
        self.addAt(other, 0, self.layout.stride);
    }

    // ---------------- offset arithmetic ----------------

    /// self[r][t+k] -= other[r][k], k < span(dur). Must not go below 0.
    pub fn subAt(self: Usage, other: Usage, t: usize, dur: usize) void {
        const n = span(dur);
        std.debug.assert(t + n <= self.layout.stride);
        for (0..self.layout.n_res) |r| {
            const m = self.row(r);
            const j = other.row(r);
            var k: usize = 0;
            while (k < n) : (k += L) store(m, t + k, load(m, t + k) - load(j, k));
        }
    }

    /// self[r][t+k] += other[r][k], k < span(dur).
    pub fn addAt(self: Usage, other: Usage, t: usize, dur: usize) void {
        const n = span(dur);
        std.debug.assert(t + n <= self.layout.stride);
        for (0..self.layout.n_res) |r| {
            const m = self.row(r);
            const j = other.row(r);
            var k: usize = 0;
            while (k < n) : (k += L) store(m, t + k, load(m, t + k) + load(j, k));
        }
    }

    /// other[r][k] <= self[r][t+k] for all r, k < span(dur)?
    pub fn covers(self: Usage, other: Usage, t: usize, dur: usize) bool {
        const n = span(dur);
        std.debug.assert(t + n <= self.layout.stride);
        for (0..self.layout.n_res) |r| {
            const m = self.row(r);
            const j = other.row(r);
            var k: usize = 0;
            while (k < n) : (k += L)
                if (@reduce(.Or, load(j, k) > load(m, t + k))) return false;
        }
        return true;
    }

    // ---------------- time and summaries ----------------

    /// Shift every row left by `k` ticks, filling the last k slots with tail[r].
    pub fn shiftLeft(self: Usage, k: usize, tail: []const Elem) void {
        std.debug.assert(tail.len == self.layout.n_res);
        const s = self.layout.stride;
        std.debug.assert(k <= s);
        for (0..self.layout.n_res) |r| {
            const rw = self.row(r);
            std.mem.copyForwards(Elem, rw[0 .. s - k], rw[k..s]);
            @memset(rw[s - k ..], tail[r]);
        }
    }

    /// out[r] = max over ticks [0, span(dur)) of resource r.
    pub fn peak(self: Usage, dur: usize, out: []Elem) void {
        std.debug.assert(out.len == self.layout.n_res);
        const n = span(dur);
        for (0..self.layout.n_res) |r| {
            var acc: V = @splat(0);
            var k: usize = 0;
            while (k < n) : (k += L) acc = @max(acc, load(self.row(r), k));
            out[r] = @reduce(.Max, acc);
        }
    }
};

// ------------------------------------------------------------------
// tests
// ------------------------------------------------------------------

const testing = std.testing;
const gpu = @import("simd_gpu.zig");

fn testPool(n: usize, layout: Layout, comptime alignment: std.mem.Alignment) ![]align(alignment.toByteUnits()) Elem {
    const buf = try testing.allocator.alignedAlloc(Elem, alignment, n * layout.block);
    @memset(buf, 0);
    return buf;
}

test "get / set / resource" {
    const layout = try Layout.init(3, 20, simd.padding);
    const buf = try testPool(1, layout, simd.ALIGN);
    defer testing.allocator.free(buf);
    const u = Usage.fromSlice(buf, layout);

    u.set(1, &.{ 5, 6, 7 });
    try testing.expectEqual(@as(Elem, 7), u.get(1, 2));
    try testing.expectEqual(@as(usize, 20), u.resource(0).len);
}

test "place / unplace / covers" {
    const layout = try Layout.init(2, 10, simd.padding);
    const buf = try testPool(2, layout, simd.ALIGN);
    defer testing.allocator.free(buf);
    const m = Usage.fromPool(buf, layout, 0);
    const j = Usage.fromPool(buf, layout, 1);

    for (0..2) |r| @memset(m.row(r), 10);
    j.set(0, &.{ 4, 4 });
    j.set(1, &.{ 1, 1 });

    try testing.expect(m.covers(j, 3, 2));
    m.subAt(j, 3, 2);
    try testing.expectEqualSlices(Elem, &.{ 10, 10, 10, 6, 6, 10 }, m.resource(0)[0..6]);
    m.addAt(j, 3, 2);
    try testing.expectEqual(@as(Elem, 10), m.get(0, 3));

    j.set(0, &.{11});
    try testing.expect(!m.covers(j, 0, 1));
}

test "shiftLeft by k / peak" {
    const layout = try Layout.init(2, 6, simd.padding);
    const buf = try testPool(1, layout, simd.ALIGN);
    defer testing.allocator.free(buf);
    const u = Usage.fromSlice(buf, layout);

    u.set(0, &.{ 1, 2, 3, 4 });
    u.set(1, &.{ 9, 0, 0 });
    var pk: [2]Elem = undefined;
    u.peak(4, &pk);
    try testing.expectEqualSlices(Elem, &.{ 4, 9 }, &pk);

    u.shiftLeft(2, &.{ 7, 7 });
    try testing.expectEqualSlices(Elem, &.{ 3, 4, 0 }, u.resource(0)[0..3]);
    try testing.expectEqual(@as(Elem, 7), u.row(0)[layout.stride - 1]);
    try testing.expectEqual(@as(Elem, 7), u.row(0)[layout.stride - 2]);
}

test "usage on a unified (CPU+GPU) layout" {
    const layout = try Layout.init(4, 100, gpu.unified_padding);
    try testing.expect(Usage.compatible(layout));
    const buf = try testPool(2, layout, gpu.UNIFIED_ALIGN);
    defer testing.allocator.free(buf);

    const m = Usage.fromPool(buf, layout, 0);
    const j = Usage.fromPool(buf, layout, 1);
    for (0..4) |r| @memset(m.row(r), 5);
    j.set(2, &.{ 3, 3, 3 });
    try testing.expect(m.covers(j, 50, 3));
    m.subAt(j, 50, 3);
    try testing.expectEqual(@as(Elem, 2), m.get(2, 51));
    // every grid in the pool starts on a GPU-aligned address
    try testing.expect(@intFromPtr(j.ptr) % gpu.ALIGN_BYTES == 0);
}
