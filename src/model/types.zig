//! Plain data shared by the model, the engine and the ports.

const std = @import("std");
const grid = @import("../grid.zig");
const Machines = @import("machines.zig").Machines;
const Jobs = @import("jobs.zig").Jobs;

pub const Elem = grid.Elem;

/// Read-only view of the simulation, handed to policies and observers.
/// Only the engine mutates the simulation.
pub const State = struct {
    now: i64,
    machines: *const Machines,
    jobs: *const Jobs,
};

/// A scheduling decision.
pub const Action = union(enum) {
    /// Start the job in `slot` on `machine` now (tick 0).
    place: struct { slot: u32, machine: u32 },
    /// Place nothing more this tick: time advances by one tick.
    wait,

    pub fn format(self: Action, w: *std.Io.Writer) std.Io.Writer.Error!void {
        switch (self) {
            .place => |p| try w.print("place(slot={d}, machine={d})", .{ p.slot, p.machine }),
            .wait => try w.writeAll("wait"),
        }
    }
};

/// Why a job could not enter the pool.
pub const RejectReason = enum { pool_full, impossible, invalid_profile };

/// Everything the engine announces to observers.
pub const Event = union(enum) {
    episode_started: struct { seed: u64 },
    job_arrived: struct { id: u32, slot: u32, tick: i64 },
    job_rejected: struct { id: u32, reason: RejectReason, tick: i64 },
    job_placed: struct { id: u32, slot: u32, machine: u32, tick: i64, arrival: i64, dur: u16 },
    illegal_action: struct { action: Action, tick: i64 },
    tick_advanced: struct { now: i64 },
    episode_done: struct { now: i64, makespan: i64, truncated: bool },
};

/// Job metadata from an ArrivalSource. The profile (grid) comes from a
/// UsageGenerator when the job arrives.
pub const JobMeta = struct {
    id: u32,
    arrival: i64,
    /// Optional class tag a generator may use to pick a distribution.
    class: u16 = 0,
};
