//! PSPLIB workload (Kolisch & Sprecher, "PSPLIB – a project scheduling
//! problem library", EJOR 1997): job profiles taken from the activities of
//! RCPSP instances (`.sm` single-mode files; `.mm` files use mode 1).
//!
//!   * every non-dummy activity (duration > 0) becomes a job template with
//!     its duration and constant per-resource demand;
//!   * demands are normalized by the instance's resource availability and
//!     rescaled to the machine capacity `cap`, so templates from instances
//!     with different availabilities mix on the same machines;
//!   * resource r of the simulator is PSPLIB renewable resource r
//!     (n_res must be <= the number of renewable resources, usually 4);
//!   * `order = .random` draws a template uniformly per job (seeded);
//!     `.sequential` takes them in file order, cycling.
//!
//! Precedence relations are ignored: jobs are independent, as everywhere in
//! the simulator. Parsing is separate from file I/O (see `parse`), so this
//! adapter does no I/O; the registry reads the files.
const std = @import("std");
const grid = @import("../../grid.zig");
const model = @import("../../model.zig");
const Error = @import("../../ports.zig").usage_generator.Error;
const Elem = grid.Elem;
const Usage = grid.Usage;

pub const max_res = 8;

pub const Activity = struct {
    dur: u16,
    /// demand / availability per renewable resource, in [0, 1]
    frac: [max_res]f32 = @splat(0),
};

pub const Order = enum { random, sequential };

pub const Psplib = struct {
    acts: []const Activity,
    /// renewable resources available in every template (min over instances)
    n_renewable: usize,
    n_machines: usize = 1,
    cap: Elem = 100,
    order: Order = .random,
    next: usize = 0,
    prng: std.Random.DefaultPrng = .init(0),

    pub fn reset(self: *Psplib, seed: u64) void {
        self.prng = .init(seed +% 3);
        self.next = 0;
    }

    pub fn machineCount(self: *Psplib) usize {
        return self.n_machines;
    }

    pub fn machine(self: *Psplib, index: usize, cap: []Elem) Error!void {
        _ = index;
        if (cap.len > self.n_renewable) return error.InvalidInput;
        @memset(cap, self.cap);
    }

    pub fn job(self: *Psplib, meta: model.JobMeta, out: Usage) Error!u16 {
        _ = meta;
        if (self.acts.len == 0 or out.layout.n_res > self.n_renewable) return error.InvalidInput;
        const i = switch (self.order) {
            .random => self.prng.random().uintLessThan(usize, self.acts.len),
            .sequential => blk: {
                const k = self.next;
                self.next = (k + 1) % self.acts.len;
                break :blk k;
            },
        };
        const a = self.acts[i];
        const dur: u16 = @intCast(@min(a.dur, out.layout.n_ticks));
        const c: f32 = @floatFromInt(self.cap);
        for (0..out.layout.n_res) |res| {
            const d: Elem = @intFromFloat(@round(a.frac[res] * c));
            @memset(out.resourceMut(res)[0..dur], d);
        }
        return dur;
    }
};

/// Parse one PSPLIB instance (text of a `.sm` / `.mm` file) and append its
/// activities to `out`. Returns the number of renewable resources.
pub fn parse(gpa: std.mem.Allocator, text: []const u8, out: *std.ArrayList(Activity)) Error!usize {
    var n_ren: ?usize = null;
    var n_non: usize = 0;
    var raw: std.ArrayList(struct { dur: u16, dem: [max_res]u32 }) = .empty;
    defer raw.deinit(gpa);
    var avail: ?[max_res]u32 = null;

    var lines = std.mem.splitScalar(u8, text, '\n');
    while (lines.next()) |line_cr| {
        const line = std.mem.trimEnd(u8, line_cr, "\r");
        const t = std.mem.trim(u8, line, " \t");
        if (startsWith(t, "- renewable")) {
            n_ren = try lastNumberBefore(t, 'R');
        } else if (startsWith(t, "- nonrenewable")) {
            n_non = try lastNumberBefore(t, 'N');
        } else if (startsWith(t, "REQUESTS/DURATIONS")) {
            const r = n_ren orelse return error.InvalidInput;
            if (r == 0 or r > max_res) return error.InvalidInput;
            _ = lines.next(); // column header
            _ = lines.next(); // dashes
            while (lines.next()) |row_cr| {
                const row = std.mem.trim(u8, row_cr, " \t\r");
                if (row.len == 0) continue;
                if (row[0] == '*') break;
                var nums: [3 + 2 * max_res + 8]u32 = undefined;
                const n = try ints(row, &nums);
                // new activity: jobnr mode dur R.. N..; extra mode lines lack jobnr -> skipped (mode 1 only)
                if (n != 3 + r + n_non) continue;
                var e: @TypeOf(raw.items[0]) = .{ .dur = std.math.cast(u16, nums[2]) orelse return error.InvalidInput, .dem = @splat(0) };
                @memcpy(e.dem[0..r], nums[3 .. 3 + r]);
                try raw.append(gpa, e);
            }
        } else if (startsWith(t, "RESOURCEAVAILABILITIES")) {
            const r = n_ren orelse return error.InvalidInput;
            _ = lines.next(); // "R 1  R 2 ..."
            const row = lines.next() orelse return error.InvalidInput;
            var nums: [2 * max_res + 8]u32 = undefined;
            const n = try ints(row, &nums);
            if (n < r) return error.InvalidInput;
            var a: [max_res]u32 = @splat(0);
            @memcpy(a[0..r], nums[0..r]);
            avail = a;
        }
    }
    const r = n_ren orelse return error.InvalidInput;
    const av = avail orelse return error.InvalidInput;
    if (raw.items.len == 0) return error.InvalidInput;
    for (raw.items) |e| {
        if (e.dur == 0) continue; // dummy source / sink
        var act: Activity = .{ .dur = e.dur };
        for (0..r) |k| {
            if (e.dem[k] > av[k]) return error.InvalidInput; // can never fit
            act.frac[k] = if (av[k] == 0) 0 else @as(f32, @floatFromInt(e.dem[k])) / @as(f32, @floatFromInt(av[k]));
        }
        try out.append(gpa, act);
    }
    return r;
}

fn startsWith(s: []const u8, p: []const u8) bool {
    return std.mem.startsWith(u8, s, p);
}

/// "- renewable                 :  4   R" -> 4
fn lastNumberBefore(s: []const u8, unit: u8) Error!usize {
    const colon = std.mem.indexOfScalar(u8, s, ':') orelse return error.InvalidInput;
    var it = std.mem.tokenizeAny(u8, s[colon + 1 ..], " \t");
    const num = it.next() orelse return error.InvalidInput;
    const u = it.next() orelse return error.InvalidInput;
    if (u.len == 0 or u[0] != unit) return error.InvalidInput;
    return std.fmt.parseInt(usize, num, 10) catch error.InvalidInput;
}

/// Whitespace-separated unsigned ints of `row` into `buf`; returns the count.
fn ints(row: []const u8, buf: []u32) Error!usize {
    var it = std.mem.tokenizeAny(u8, row, " \t\r");
    var n: usize = 0;
    while (it.next()) |tok| {
        if (n == buf.len) return error.InvalidInput;
        buf[n] = std.fmt.parseInt(u32, tok, 10) catch return error.InvalidInput;
        n += 1;
    }
    return n;
}

// ----------------------------------------------------------------------------- tests

/// Small hand-made instance in the PSPLIB .sm layout (2 real activities + dummies).
pub const example_sm =
    \\************************************************************************
    \\file with basedata            : example.bas
    \\initial value random generator: 1
    \\************************************************************************
    \\projects                      :  1
    \\jobs (incl. supersource/sink ):  4
    \\horizon                       :  20
    \\RESOURCES
    \\  - renewable                 :  4   R
    \\  - nonrenewable              :  2   N
    \\  - doubly constrained        :  0   D
    \\************************************************************************
    \\PRECEDENCE RELATIONS:
    \\jobnr.    #modes  #successors   successors
    \\   1        1          2           2   3
    \\   2        1          1           4
    \\   3        1          1           4
    \\   4        1          0
    \\************************************************************************
    \\REQUESTS/DURATIONS:
    \\jobnr. mode duration  R 1  R 2  R 3  R 4  N 1  N 2
    \\------------------------------------------------------------------------
    \\  1      1     0       0    0    0    0    0    0
    \\  2      1     3       5    0   10    2    0    0
    \\  3      1     7       0   20    5    8    0    0
    \\  4      1     0       0    0    0    0    0    0
    \\************************************************************************
    \\RESOURCEAVAILABILITIES:
    \\  R 1  R 2  R 3  R 4  N 1  N 2
    \\   10   20   20   16    0    0
    \\************************************************************************
    \\
;

test "psplib: parse an .sm instance" {
    const a = std.testing.allocator;
    var acts: std.ArrayList(Activity) = .empty;
    defer acts.deinit(a);
    try std.testing.expectEqual(@as(usize, 4), try parse(a, example_sm, &acts));
    try std.testing.expectEqual(@as(usize, 2), acts.items.len); // dummies dropped
    try std.testing.expectEqual(@as(u16, 3), acts.items[0].dur);
    try std.testing.expectEqualSlices(f32, &.{ 0.5, 0, 0.5, 0.125 }, acts.items[0].frac[0..4]);
    try std.testing.expectEqual(@as(u16, 7), acts.items[1].dur);
    try std.testing.expectEqualSlices(f32, &.{ 0, 1, 0.25, 0.5 }, acts.items[1].frac[0..4]);
}

test "psplib: malformed input is rejected" {
    const a = std.testing.allocator;
    var acts: std.ArrayList(Activity) = .empty;
    defer acts.deinit(a);
    try std.testing.expectError(error.InvalidInput, parse(a, "not a psplib file", &acts));
    // demand above availability
    const bad = try std.mem.replaceOwned(u8, a, example_sm, "   10   20   20   16", "    1   20   20   16");
    defer a.free(bad);
    try std.testing.expectError(error.InvalidInput, parse(a, bad, &acts));
}

test "psplib: jobs follow the templates, scaled to cap; seeded and sequential" {
    const a = std.testing.allocator;
    var acts: std.ArrayList(Activity) = .empty;
    defer acts.deinit(a);
    const n_ren = try parse(a, example_sm, &acts);

    const layout = try grid.Layout.init(3, 8, grid.simd.padding);
    const buf = try a.alignedAlloc(Elem, grid.simd.ALIGN, layout.block);
    defer a.free(buf);

    var g = Psplib{ .acts = acts.items, .n_renewable = n_ren, .cap = 40, .order = .sequential };
    g.reset(1);
    var cap: [3]Elem = undefined;
    try g.machine(0, &cap);
    try std.testing.expectEqualSlices(Elem, &.{ 40, 40, 40 }, &cap);
    var too_many: [5]Elem = undefined;
    try std.testing.expectError(error.InvalidInput, g.machine(0, &too_many));

    for ([_]u16{ 3, 7, 3 }) |want| {
        @memset(buf, 0);
        const u = Usage.fromSlice(buf, layout);
        const dur = try g.job(.{ .id = 0, .arrival = 0 }, u);
        try std.testing.expectEqual(want, dur);
        const exp: [3]Elem = if (want == 3) .{ 20, 0, 20 } else .{ 0, 40, 10 };
        for (0..3) |r| {
            try std.testing.expectEqual(exp[r], u.resource(r)[0]);
            try std.testing.expectEqual(exp[r], u.resource(r)[dur - 1]);
            try std.testing.expectEqual(@as(Elem, 0), u.resource(r)[dur]);
        }
    }

    // random order: same seed -> same sequence
    g.order = .random;
    var seqs: [2][16]u16 = undefined;
    for (&seqs) |*s| {
        g.reset(7);
        for (s) |*d| {
            @memset(buf, 0);
            d.* = try g.job(.{ .id = 0, .arrival = 0 }, Usage.fromSlice(buf, layout));
        }
    }
    try std.testing.expectEqualSlices(u16, &seqs[0], &seqs[1]);
}
