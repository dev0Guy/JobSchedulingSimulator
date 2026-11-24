//! Engine layer: rules and the step loop.
pub const rules = @import("engine/rules.zig");
pub const Config = rules.Config;
pub const Reward = rules.Reward;
pub const Engine = @import("engine/engine.zig").Engine;
pub const StepResult = @import("engine/engine.zig").StepResult;
pub const Stats = @import("engine/engine.zig").Stats;

test {
    _ = rules;
    _ = @import("engine/engine.zig");
}
