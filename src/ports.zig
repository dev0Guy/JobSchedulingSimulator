//! Ports: the interfaces between the engine and the outside world.
//!   Policy         — decides the next Action             (hot, comptime or runtime)
//!   Observer       — reacts to Events                    (runtime, several at once)
//!   UsageGenerator — creates machine capacities and job profiles
//!   ArrivalSource  — creates job metadata (which jobs, when)
const model = @import("model.zig");
pub const State = model.State;
pub const Action = model.Action;
pub const Event = model.Event;
pub const JobMeta = model.JobMeta;

pub const policy = @import("ports/policy.zig");
pub const Policy = policy.Policy;
pub const assertPolicy = policy.assertPolicy;

pub const Observer = @import("ports/observer.zig").Observer;
pub const ObserverList = @import("ports/observer.zig").ObserverList;

pub const usage_generator = @import("ports/usage_generator.zig");
pub const UsageGenerator = usage_generator.UsageGenerator;

pub const arrival_source = @import("ports/arrival_source.zig");
pub const ArrivalSource = arrival_source.ArrivalSource;

test {
    _ = @import("ports/policy.zig");
    _ = @import("ports/observer.zig");
    _ = @import("ports/usage_generator.zig");
    _ = @import("ports/arrival_source.zig");
}
