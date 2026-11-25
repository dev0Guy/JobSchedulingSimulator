//! Metrics observer: per-machine placements, waiting-time distribution;
//! prints a summary at the end of each episode.
const std = @import("std");
const model = @import("../../model.zig");

pub const Metrics = struct {
    gpa: std.mem.Allocator,
    waits: std.ArrayList(i64) = .empty,
    per_machine: std.ArrayList(usize) = .empty,
    rejected: usize = 0,
    illegal: usize = 0,
    /// Print the summary on episode_done.
    print_summary: bool = true,

    pub fn init(gpa: std.mem.Allocator) Metrics {
        return .{ .gpa = gpa };
    }

    pub fn deinit(self: *Metrics) void {
        self.waits.deinit(self.gpa);
        self.per_machine.deinit(self.gpa);
    }

    pub fn onEvent(self: *Metrics, ev: model.Event, st: *const model.State) void {
        switch (ev) {
            .episode_started => {
                self.waits.clearRetainingCapacity();
                self.per_machine.clearRetainingCapacity();
                self.per_machine.appendNTimes(self.gpa, 0, st.machines.n) catch {};
                self.rejected = 0;
                self.illegal = 0;
            },
            .job_placed => |p| {
                self.waits.append(self.gpa, p.tick - p.arrival) catch {};
                if (p.machine < self.per_machine.items.len) self.per_machine.items[p.machine] += 1;
            },
            .job_rejected => self.rejected += 1,
            .illegal_action => self.illegal += 1,
            .episode_done => |d| if (self.print_summary) self.summary(d.makespan, d.truncated),
            else => {},
        }
    }

    pub fn percentile(self: *Metrics, p: f64) i64 {
        const w = self.waits.items;
        if (w.len == 0) return 0;
        std.mem.sort(i64, w, {}, std.sort.asc(i64));
        const idx: usize = @intFromFloat(@round(p * @as(f64, @floatFromInt(w.len - 1))));
        return w[idx];
    }

    fn summary(self: *Metrics, makespan: i64, truncated: bool) void {
        var total: i64 = 0;
        for (self.waits.items) |x| total += x;
        const n = self.waits.items.len;
        const avg = if (n == 0) 0 else @as(f64, @floatFromInt(total)) / @as(f64, @floatFromInt(n));
        std.debug.print(
            \\
            \\── metrics {s}──────────────────────────────
            \\placed      {d}
            \\rejected    {d}
            \\illegal     {d}
            \\makespan    {d} ticks
            \\wait avg    {d:.2}   p50 {d}   p90 {d}   p99 {d}   max {d}
            \\per machine {any}
            \\
        , .{
            if (truncated) "(truncated) " else "",
            n,
            self.rejected,
            self.illegal,
            makespan,
            avg,
            self.percentile(0.5),
            self.percentile(0.9),
            self.percentile(0.99),
            self.percentile(1.0),
            self.per_machine.items,
        });
    }
};
