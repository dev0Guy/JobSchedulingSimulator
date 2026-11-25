//! Terminal visualization observer. Redraws on every tick (and optionally on
//! every placement) and shows two colored matrices:
//!   MACHINES: per machine (and per resource, or the busiest resource),
//!             one column per tick of the horizon; cell = used / capacity.
//!   JOBS:     waiting jobs, oldest first; one column per tick of the job's
//!             profile; cell = max over resources of demand / max capacity.
//! A frame is built in memory and written with one print (no flicker).
const std = @import("std");
const model = @import("../../model.zig");
const Elem = @import("../../grid.zig").Elem;

pub const Options = struct {
    /// Max tick columns; longer horizons are compressed (max over bucket).
    max_cols: usize = 100,
    /// Max waiting jobs listed.
    max_jobs: usize = 20,
    /// One row per resource (true) or per machine (false).
    per_resource: bool = false,
    /// Redraw also after each placement (slower, shows every decision).
    on_place: bool = false,
    /// Delay after each frame, ms.
    delay_ms: u32 = 80,
    color: bool = true,
};

pub const TerminalViz = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    opts: Options,
    out: std.ArrayList(u8) = .empty,
    order: std.ArrayList(usize) = .empty,
    last_color: u8 = 0,
    placed: usize = 0,
    rejected: usize = 0,
    started: bool = false,

    pub fn init(gpa: std.mem.Allocator, io: std.Io, opts: Options) TerminalViz {
        return .{ .gpa = gpa, .io = io, .opts = opts };
    }

    pub fn deinit(self: *TerminalViz) void {
        if (self.started) std.debug.print("\x1b[0m\x1b[?25h", .{});
        self.out.deinit(self.gpa);
        self.order.deinit(self.gpa);
    }

    // ---------------- Observer contract ----------------

    pub fn onEvent(self: *TerminalViz, ev: model.Event, st: *const model.State) void {
        switch (ev) {
            .episode_started => {
                self.placed = 0;
                self.rejected = 0;
                if (!self.started) std.debug.print("\x1b[?25l\x1b[2J", .{});
                self.started = true;
                self.frame(st, "started");
            },
            .job_placed => {
                self.placed += 1;
                if (self.opts.on_place) self.frame(st, "placed");
            },
            .job_rejected => self.rejected += 1,
            .tick_advanced => self.frame(st, "tick"),
            .episode_done => |d| {
                self.frame(st, if (d.truncated) "truncated" else "done");
                std.debug.print("\x1b[0m\x1b[?25h", .{});
                self.started = false;
            },
            else => {},
        }
    }

    fn frame(self: *TerminalViz, st: *const model.State, label: []const u8) void {
        self.draw(st, label) catch return; // out of memory: skip the frame
        std.debug.print("{s}", .{self.out.items});
        if (self.opts.delay_ms > 0)
            self.io.sleep(.fromMilliseconds(self.opts.delay_ms), .awake) catch {};
    }

    // ---------------- drawing ----------------

    const glyphs = [_][]const u8{ "·", "░", "▒", "▓", "█" };
    const colors = [_]u8{ 240, 34, 178, 208, 196 };

    fn level(frac: f32) usize {
        if (frac <= 0) return 0;
        if (frac <= 0.25) return 1;
        if (frac <= 0.5) return 2;
        if (frac <= 0.75) return 3;
        return 4;
    }

    fn put(self: *TerminalViz, comptime fmt: []const u8, args: anytype) !void {
        try self.out.print(self.gpa, fmt, args);
    }

    fn esc(self: *TerminalViz, code: []const u8) !void {
        if (self.opts.color) try self.out.appendSlice(self.gpa, code);
        if (std.mem.eql(u8, code, "\x1b[0m")) self.last_color = 0;
    }

    fn cell(self: *TerminalViz, frac: f32) !void {
        const lv = level(frac);
        if (self.opts.color and colors[lv] != self.last_color) {
            try self.put("\x1b[38;5;{d}m", .{colors[lv]});
            self.last_color = colors[lv];
        }
        try self.out.appendSlice(self.gpa, glyphs[lv]);
    }

    fn usedFrac(free: []const Elem, cap: Elem, t0: usize, t1: usize) f32 {
        if (cap == 0) return 0;
        var used: Elem = 0;
        for (free[t0..t1]) |f| used = @max(used, cap -| f);
        return @as(f32, @floatFromInt(used)) / @as(f32, @floatFromInt(cap));
    }

    fn draw(self: *TerminalViz, st: *const model.State, label: []const u8) !void {
        self.out.clearRetainingCapacity();
        const ms = st.machines;
        const jobs = st.jobs;
        const lay = ms.layout;
        const n_ticks = lay.n_ticks;
        const bucket = @max(1, (n_ticks + self.opts.max_cols - 1) / self.opts.max_cols);
        const cols = (n_ticks + bucket - 1) / bucket;

        try self.out.appendSlice(self.gpa, "\x1b[H");

        // header
        try self.esc("\x1b[1m");
        try self.put("t = {d}", .{st.now});
        try self.esc("\x1b[0m");
        try self.put("  [{s}]   waiting = {d}   placed = {d}   rejected = {d}   util now = {d:.0}%\x1b[K\n", .{
            label, jobs.count, self.placed, self.rejected, ms.utilizationNow() * 100,
        });
        try self.esc("\x1b[2m");
        try self.put("legend: · 0%  ░ ≤25%  ▒ ≤50%  ▓ ≤75%  █ >75%    1 col = {d} tick(s)\x1b[K\n\n", .{bucket});
        try self.esc("\x1b[0m");

        // machines
        try self.esc("\x1b[1m");
        try self.put("MACHINES  (used / capacity over the horizon, tick 0 = now)\x1b[K\n", .{});
        try self.esc("\x1b[0m");
        try self.axis(cols, bucket, 15);

        for (0..ms.n) |m| {
            const mach = ms.get(m);
            if (self.opts.per_resource) {
                for (0..lay.n_res) |r| {
                    if (r == 0) try self.put("M{d:<3}", .{m}) else try self.put("    ", .{});
                    try self.put(" r{d:<2} {d:>5} |", .{ r, mach.cap[r] });
                    for (0..cols) |c| {
                        const t0 = c * bucket;
                        try self.cell(usedFrac(mach.resource(r), mach.cap[r], t0, @min(n_ticks, t0 + bucket)));
                    }
                    try self.esc("\x1b[0m");
                    try self.put("|\x1b[K\n", .{});
                }
            } else {
                try self.put("M{d:<3} cap {d:>5} |", .{ m, mach.cap[0] });
                for (0..cols) |c| {
                    const t0 = c * bucket;
                    const t1 = @min(n_ticks, t0 + bucket);
                    var frac: f32 = 0;
                    for (0..lay.n_res) |r| frac = @max(frac, usedFrac(mach.resource(r), mach.cap[r], t0, t1));
                    try self.cell(frac);
                }
                try self.esc("\x1b[0m");
                try self.put("| {d:>3.0}%\x1b[K\n", .{mach.utilizationNow() * 100});
            }
        }

        // jobs, oldest first
        self.order.clearRetainingCapacity();
        var it = jobs.iter();
        while (it.next()) |s| try self.order.append(self.gpa, s);
        const Ctx = struct {
            arrival: []const i64,
            fn lt(c: @This(), a: usize, b: usize) bool {
                return c.arrival[a] < c.arrival[b] or (c.arrival[a] == c.arrival[b] and a < b);
            }
        };
        std.mem.sort(usize, self.order.items, Ctx{ .arrival = jobs.arrival }, Ctx.lt);

        const shown = @min(self.order.items.len, self.opts.max_jobs);
        try self.esc("\x1b[1m");
        try self.put("\nJOBS  (waiting, oldest first: {d} of {d})\x1b[K\n", .{ shown, self.order.items.len });
        try self.esc("\x1b[0m");
        try self.put("    id   arr  wait dur |\x1b[K\n", .{});

        for (self.order.items[0..shown]) |s| {
            const u = jobs.get(s);
            const dur: usize = jobs.duration[s];
            try self.put("{d:>6} {d:>5} {d:>5} {d:>3} |", .{
                jobs.id[s], @as(u64, @intCast(jobs.arrival[s])), @as(u64, @intCast(st.now - jobs.arrival[s])), dur,
            });
            const jcols = (dur + bucket - 1) / bucket;
            for (0..jcols) |c| {
                const t0 = c * bucket;
                const t1 = @min(dur, t0 + bucket);
                var frac: f32 = 0;
                for (0..lay.n_res) |r| {
                    const mc: f32 = @floatFromInt(@max(1, ms.maxCap(r)));
                    for (u.resource(r)[t0..t1]) |v| frac = @max(frac, @as(f32, @floatFromInt(v)) / mc);
                }
                try self.cell(frac);
            }
            try self.esc("\x1b[0m");
            try self.put("\x1b[K\n", .{});
        }
        if (self.order.items.len > shown) {
            try self.esc("\x1b[2m");
            try self.put("  … {d} more\x1b[K\n", .{self.order.items.len - shown});
            try self.esc("\x1b[0m");
        }
        try self.put("\x1b[J", .{});
    }

    fn axis(self: *TerminalViz, cols: usize, bucket: usize, indent: usize) !void {
        try self.esc("\x1b[2m");
        for (0..indent) |_| try self.put(" ", .{});
        try self.put("|", .{});
        var c: usize = 0;
        while (c < cols) {
            if (c % 10 == 0) {
                var buf: [16]u8 = undefined;
                const s = try std.fmt.bufPrint(&buf, "{d}", .{c * bucket});
                if (c + s.len <= cols) {
                    try self.put("{s}", .{s});
                    c += s.len;
                    continue;
                }
            }
            try self.put(" ", .{});
            c += 1;
        }
        try self.put("|\x1b[K\n", .{});
        try self.esc("\x1b[0m");
    }
};
