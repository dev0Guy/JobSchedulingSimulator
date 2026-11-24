//! sched: a SIMD job-to-machine scheduling simulator.
//!
//!     const sched = @import("sched");
//!     var arrivals = sched.adapters.arrivals.Poisson{ .total = 500, .rate = 0.5 };
//!     var profiles = sched.adapters.generators.RandomProfiles{};
//!     var eng = try sched.Engine.init(gpa, .{}, .init(&arrivals), .init(&profiles));
//!     defer eng.deinit();
//!     try eng.reset(42);
//!     var policy = sched.adapters.policies.FifoBackfill{};
//!     const stats = try eng.run(&policy, 1_000_000);
//!
//! Layers (each imports only the ones above it):
//!   grid     — aligned [resource][tick] grids, SIMD operations
//!   model    — machines, jobs, State / Action / Event
//!   ports    — Policy, Observer, UsageGenerator, ArrivalSource
//!   engine   — rules and the step loop
//!   adapters — concrete policies, sources, generators, observers

const std = @import("std");

pub const version = std.SemanticVersion{ .major = 0, .minor = 1, .patch = 0 };

pub const grid = @import("grid.zig");
pub const model = @import("model.zig");
pub const ports = @import("ports.zig");
pub const engine = @import("engine.zig");
pub const adapters = @import("adapters.zig");

pub const Elem = grid.Elem;
pub const Layout = grid.Layout;
pub const Usage = grid.Usage;
pub const Machines = model.Machines;
pub const Jobs = model.Jobs;
pub const State = model.State;
pub const Action = model.Action;
pub const Event = model.Event;
pub const JobMeta = model.JobMeta;
pub const Engine = engine.Engine;
pub const Config = engine.Config;
pub const Stats = engine.Stats;

test {
    _ = grid;
    _ = model;
    _ = ports;
    _ = engine;
    _ = adapters;
}
