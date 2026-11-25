//! Trace observer: writes every event of every episode to a JSON Lines file,
//! with enough data to rebuild the full simulation state offline (replay,
//! visualization, analysis) without running the simulator again.
//!
//! One JSON object per line; `type` says which. Format version 1:
//!
//!   {"type":"episode","v":1,"seed":S,"n_res":R,"n_ticks":T,"max_jobs":J,"caps":[[c..],..]}
//!   {"type":"arrive","t":T,"id":I,"slot":S,"dur":D,"demand":[[..r0..],[..r1..]]}
//!   {"type":"reject","t":T,"id":I,"reason":"pool_full|impossible|invalid_profile"}
//!   {"type":"place","t":T,"id":I,"slot":S,"machine":M,"arrival":A,"dur":D}
//!   {"type":"illegal","t":T}
//!   {"type":"tick","t":T}          time advanced to T (after arrivals at T)
//!   {"type":"done","t":T,"makespan":M,"truncated":false}
//!
//! `caps` is [machine][resource]; `demand` is [resource][tick < dur].
//! Ordering: within a tick, arrivals come before the `tick` line, so the
//! state right after a `tick` line is "start of tick T". The python side
//! (python/sched_env/replay.py) reads this format.
//!
//! I/O errors never stop the simulation: the first one is kept in `err`
//! and later events are dropped.

const std = @import("std");
const model = @import("../../model.zig");

pub const Trace = struct {
    io: std.Io = undefined,
    file: std.Io.File = undefined,
    writer: std.Io.File.Writer = undefined,
    buf: [64 * 1024]u8 = undefined,
    /// Also write one line per tick (needed for replay; turn off only for
    /// smaller files when you just want placements).
    ticks: bool = true,
    err: ?anyerror = null,

    /// Create (truncate) `path` relative to `dir`. `self` must not move
    /// afterwards (the writer points into `buf`).
    pub fn open(self: *Trace, io: std.Io, dir: std.Io.Dir, path: []const u8) !void {
        self.io = io;
        self.err = null;
        self.file = try dir.createFile(io, path, .{});
        self.writer = self.file.writer(io, &self.buf);
    }

    /// Flush and close. Returns the first I/O error, if any.
    pub fn close(self: *Trace) !void {
        self.writer.interface.flush() catch |e| {
            if (self.err == null) self.err = e;
        };
        self.file.close(self.io);
        if (self.err) |e| return e;
    }

    pub fn deinit(self: *Trace) void {
        self.close() catch {};
    }

    pub fn onEvent(self: *Trace, ev: model.Event, st: *const model.State) void {
        if (self.err != null) return;
        self.write(ev, st) catch |e| {
            self.err = e;
        };
    }

    fn write(self: *Trace, ev: model.Event, st: *const model.State) !void {
        const w = &self.writer.interface;
        switch (ev) {
            .episode_started => |e| {
                const ms = st.machines;
                const n_res = ms.layout.n_res;
                try w.print("{{\"type\":\"episode\",\"v\":1,\"seed\":{d},\"n_res\":{d},\"n_ticks\":{d},\"max_jobs\":{d},\"caps\":[", .{ e.seed, n_res, ms.layout.n_ticks, st.jobs.cap });
                for (0..ms.n) |m| {
                    if (m > 0) try w.writeByte(',');
                    try writeList(w, ms.cap[m * n_res ..][0..n_res]);
                }
                try w.writeAll("]}\n");
            },
            .job_arrived => |e| {
                const dur = st.jobs.duration[e.slot];
                try w.print("{{\"type\":\"arrive\",\"t\":{d},\"id\":{d},\"slot\":{d},\"dur\":{d},\"demand\":[", .{ e.tick, e.id, e.slot, dur });
                const job = st.jobs.get(e.slot);
                for (0..st.jobs.layout.n_res) |r| {
                    if (r > 0) try w.writeByte(',');
                    try writeList(w, job.row(r)[0..dur]);
                }
                try w.writeAll("]}\n");
            },
            .job_rejected => |e| try w.print("{{\"type\":\"reject\",\"t\":{d},\"id\":{d},\"reason\":\"{t}\"}}\n", .{ e.tick, e.id, e.reason }),
            .job_placed => |e| try w.print("{{\"type\":\"place\",\"t\":{d},\"id\":{d},\"slot\":{d},\"machine\":{d},\"arrival\":{d},\"dur\":{d}}}\n", .{ e.tick, e.id, e.slot, e.machine, e.arrival, e.dur }),
            .illegal_action => |e| try w.print("{{\"type\":\"illegal\",\"t\":{d}}}\n", .{e.tick}),
            .tick_advanced => |e| if (self.ticks) try w.print("{{\"type\":\"tick\",\"t\":{d}}}\n", .{e.now}),
            .episode_done => |e| {
                try w.print("{{\"type\":\"done\",\"t\":{d},\"makespan\":{d},\"truncated\":{}}}\n", .{ e.now, e.makespan, e.truncated });
                try w.flush();
            },
        }
    }

    fn writeList(w: *std.Io.Writer, xs: anytype) !void {
        try w.writeByte('[');
        for (xs, 0..) |x, i| {
            if (i > 0) try w.writeByte(',');
            try w.print("{d}", .{x});
        }
        try w.writeByte(']');
    }
};

test "trace: writes a parseable line per event" {
    const sched_ports = @import("../../ports.zig");
    const Engine = @import("../../engine.zig").Engine;
    const Poisson = @import("../arrivals/poisson.zig").Poisson;
    const DeepRM = @import("../generators/deeprm.zig").DeepRM;
    const Fifo = @import("../policies/fifo_backfill.zig").FifoBackfill;
    const t = std.testing;

    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();

    var arr = Poisson{ .total = 20, .rate = 0.5 };
    var gen = DeepRM{ .n_machines = 2 };
    var eng = try Engine.init(t.allocator, .{ .n_res = 2, .n_ticks = 10, .max_jobs = 8 }, .init(&arr), .init(&gen));
    defer eng.deinit();
    var tr = Trace{};
    try tr.open(t.io, tmp.dir, "run.jsonl");
    try eng.addObserver(sched_ports.Observer.init(&tr));
    try eng.reset(3);
    var p = Fifo{};
    const st = try eng.run(&p, 10_000);
    try tr.close();

    const text = try tmp.dir.readFileAlloc(t.io, "run.jsonl", t.allocator, .limited(1 << 20));
    defer t.allocator.free(text);

    const kinds = [_][]const u8{ "episode", "done", "arrive", "place", "tick" };
    var counts: [kinds.len]usize = @splat(0);
    var lines = std.mem.splitScalar(u8, std.mem.trimEnd(u8, text, "\n"), '\n');
    while (lines.next()) |line| {
        const parsed = try std.json.parseFromSlice(std.json.Value, t.allocator, line, .{});
        defer parsed.deinit();
        const ty = parsed.value.object.get("type").?.string;
        for (kinds, 0..) |k, i| if (std.mem.eql(u8, k, ty)) {
            counts[i] += 1;
        };
    }
    try t.expectEqual(@as(usize, 1), counts[0]);
    try t.expectEqual(@as(usize, 1), counts[1]);
    try t.expectEqual(st.arrived, counts[2]);
    try t.expectEqual(st.placed, counts[3]);
    try t.expectEqual(@as(usize, @intCast(st.ticks)), counts[4]);
}
