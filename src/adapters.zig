//! Adapters: concrete implementations of the ports.
pub const policies = struct {
    pub const FifoBackfill = @import("adapters/policies/fifo_backfill.zig").FifoBackfill;
    pub const FifoStrict = @import("adapters/policies/fifo_strict.zig").FifoStrict;
    pub const ShortestFirst = @import("adapters/policies/sjf.zig").ShortestFirst;
    pub const BestFit = @import("adapters/policies/best_fit.zig").BestFit;
    pub const RandomPolicy = @import("adapters/policies/random.zig").RandomPolicy;
    pub const Tetris = @import("adapters/policies/tetris.zig").Tetris;
    pub const TetrisSrpt = @import("adapters/policies/tetris.zig").TetrisSrpt;
};

pub const arrivals = struct {
    pub const Fixed = @import("adapters/arrivals/fixed.zig").Fixed;
    pub const Poisson = @import("adapters/arrivals/poisson.zig").Poisson;
    pub const Mmpp = @import("adapters/arrivals/mmpp.zig").Mmpp;
    pub const ParetoArrivals = @import("adapters/arrivals/pareto.zig").ParetoArrivals;
    pub const SelfSimilar = @import("adapters/arrivals/self_similar.zig").SelfSimilar;
};

pub const generators = struct {
    pub const RandomProfiles = @import("adapters/generators/random_profiles.zig").RandomProfiles;
    pub const DeepRM = @import("adapters/generators/deeprm.zig").DeepRM;
};

pub const rewards = struct {
    pub const NegInSystem = @import("adapters/rewards/basic.zig").NegInSystem;
    pub const NegWaiting = @import("adapters/rewards/basic.zig").NegWaiting;
    pub const Zero = @import("adapters/rewards/basic.zig").Zero;
    pub const Slowdown = @import("adapters/rewards/slowdown.zig").Slowdown;
};

pub const observers = struct {
    pub const TerminalViz = @import("adapters/observers/terminal_viz.zig").TerminalViz;
    pub const VizOptions = @import("adapters/observers/terminal_viz.zig").Options;
    pub const Metrics = @import("adapters/observers/metrics.zig").Metrics;
    pub const EventLog = @import("adapters/observers/event_log.zig").EventLog;
};

test {
    _ = @import("adapters/arrivals/poisson.zig");
    _ = @import("adapters/arrivals/sample.zig");
    _ = @import("adapters/arrivals/burst_test.zig");
    @import("std").testing.refAllDecls(rewards);
    @import("std").testing.refAllDecls(policies);
    @import("std").testing.refAllDecls(arrivals);
    @import("std").testing.refAllDecls(generators);
    @import("std").testing.refAllDecls(observers);
}
