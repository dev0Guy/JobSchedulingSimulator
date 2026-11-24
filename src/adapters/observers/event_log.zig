//! Event log observer: one line per event on stderr. For debugging.
const std = @import("std");
const model = @import("../../model.zig");

pub const EventLog = struct {
    /// Also log every tick (noisy).
    ticks: bool = false,

    pub fn onEvent(self: *EventLog, ev: model.Event, st: *const model.State) void {
        _ = st;
        switch (ev) {
            .episode_started => |e| std.debug.print("[start] seed={d}\n", .{e.seed}),
            .job_arrived => |e| std.debug.print("[t={d}] arrive  job={d} slot={d}\n", .{ e.tick, e.id, e.slot }),
            .job_rejected => |e| std.debug.print("[t={d}] reject  job={d} reason={t}\n", .{ e.tick, e.id, e.reason }),
            .job_placed => |e| std.debug.print("[t={d}] place   job={d} machine={d} dur={d} wait={d}\n", .{ e.tick, e.id, e.machine, e.dur, e.tick - e.arrival }),
            .illegal_action => |e| std.debug.print("[t={d}] illegal {f}\n", .{ e.tick, e.action }),
            .tick_advanced => |e| if (self.ticks) std.debug.print("[t={d}] tick\n", .{e.now}),
            .episode_done => |e| std.debug.print("[t={d}] done    makespan={d}{s}\n", .{ e.now, e.makespan, if (e.truncated) " (truncated)" else "" }),
        }
    }
};
