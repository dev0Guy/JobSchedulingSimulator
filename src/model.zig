//! Model layer: domain state (machines, jobs) built on the grid layer.
pub const types = @import("model/types.zig");
pub const State = types.State;
pub const Action = types.Action;
pub const Event = types.Event;
pub const JobMeta = types.JobMeta;
pub const RejectReason = types.RejectReason;

pub const Machines = @import("model/machines.zig").Machines;
pub const Machine = @import("model/machines.zig").Machine;
pub const Jobs = @import("model/jobs.zig").Jobs;

test {
    _ = @import("model/machines.zig");
    _ = @import("model/jobs.zig");
}
