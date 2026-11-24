//! Adapters: concrete implementations of the ports.
pub const policies = struct {
    pub const FifoBackfill = @import("adapters/policies/fifo_backfill.zig").FifoBackfill;
    pub const FifoStrict = @import("adapters/policies/fifo_strict.zig").FifoStrict;
    pub const ShortestFirst = @import("adapters/policies/sjf.zig").ShortestFirst;
    pub const BestFit = @import("adapters/policies/best_fit.zig").BestFit;
    pub const RandomPolicy = @import("adapters/policies/random.zig").RandomPolicy;
};

pub const arrivals = struct {
    pub const Fixed = @import("adapters/arrivals/fixed.zig").Fixed;
    pub const Poisson = @import("adapters/arrivals/poisson.zig").Poisson;
};

pub const generators = struct {
    pub const RandomProfiles = @import("adapters/generators/random_profiles.zig").RandomProfiles;
    pub const DeepRM = @import("adapters/generators/deeprm.zig").DeepRM;
};

pub const observers = struct {
    pub const TerminalViz = @import("adapters/observers/terminal_viz.zig").TerminalViz;
    pub const VizOptions = @import("adapters/observers/terminal_viz.zig").Options;
    pub const Metrics = @import("adapters/observers/metrics.zig").Metrics;
    pub const EventLog = @import("adapters/observers/event_log.zig").EventLog;
};

test {
    _ = @import("adapters/arrivals/poisson.zig");
    @import("std").testing.refAllDecls(policies);
    @import("std").testing.refAllDecls(arrivals);
    @import("std").testing.refAllDecls(generators);
    @import("std").testing.refAllDecls(observers);
}
