//! Policy port: decides what the engine does next.
//!
//! Two forms of the same contract:
//!   * Comptime (hot path, zero cost): any type `P` with
//!         pub fn decide(self: *P, state: *const State) Action
//!     and optionally
//!         pub fn reset(self: *P, seed: u64) void
//!     Check it with `assertPolicy(P)`; the engine is then specialized for P
//!     and `decide` can be inlined into the step loop.
//!   * Runtime (`Policy`): a type-erased handle for when the policy must be
//!     chosen or swapped at run time, or lives outside Zig (e.g. Python).
//!
//! Rules for implementers:
//!   * Never mutate the state (it is `*const`).
//!   * Return an action you believe is legal; the engine validates anyway
//!     and reports illegal ones as `Event.illegal_action`.
//!   * Several `.place` actions may be returned in a row within one tick;
//!     return `.wait` when you want time to advance.

const std = @import("std");
const types = @import("../model/types.zig");
pub const State = types.State;
pub const Action = types.Action;

/// Compile-time check that `P` implements the policy contract.
pub fn assertPolicy(comptime P: type) void {
    if (!@hasDecl(P, "decide"))
        @compileError(@typeName(P) ++ " is not a Policy: missing `pub fn decide(self: *" ++ @typeName(P) ++ ", state: *const State) Action`");
    const F = @TypeOf(P.decide);
    const info = @typeInfo(F).@"fn";
    if (info.param_types.len != 2 or info.return_type.? != Action)
        @compileError(@typeName(P) ++ ".decide must be fn(*Self, *const State) Action");
}

/// Call `reset` if the policy has one.
pub fn resetIfAny(comptime P: type, p: *P, seed: u64) void {
    if (@hasDecl(P, "reset")) p.reset(seed);
}

/// Runtime (type-erased) policy.
pub const Policy = struct {
    ptr: *anyopaque,
    vtable: *const VTable,

    pub const VTable = struct {
        decide: *const fn (ptr: *anyopaque, state: *const State) Action,
        reset: *const fn (ptr: *anyopaque, seed: u64) void,
    };

    pub fn decide(self: Policy, state: *const State) Action {
        return self.vtable.decide(self.ptr, state);
    }

    pub fn reset(self: Policy, seed: u64) void {
        self.vtable.reset(self.ptr, seed);
    }

    /// Wrap any comptime policy (`impl: *P`) as a runtime Policy.
    pub fn init(impl: anytype) Policy {
        const Ptr = @TypeOf(impl);
        const P = @typeInfo(Ptr).pointer.child;
        comptime assertPolicy(P);
        const gen = struct {
            fn decide(p: *anyopaque, state: *const State) Action {
                const self: Ptr = @ptrCast(@alignCast(p));
                return self.decide(state);
            }
            fn reset(p: *anyopaque, seed: u64) void {
                const self: Ptr = @ptrCast(@alignCast(p));
                resetIfAny(P, self, seed);
            }
        };
        const vt = VTable{ .decide = gen.decide, .reset = gen.reset };
        return .{ .ptr = impl, .vtable = &vt };
    }
};

test "policy port: comptime check and runtime wrapper" {
    const AlwaysWait = struct {
        calls: usize = 0,
        pub fn decide(self: *@This(), state: *const State) Action {
            _ = state;
            self.calls += 1;
            return .wait;
        }
    };
    comptime assertPolicy(AlwaysWait);

    var impl = AlwaysWait{};
    const p = Policy.init(&impl);
    const state: State = undefined; // not read by this policy
    try std.testing.expect(p.decide(&state) == .wait);
    p.reset(1); // no reset decl → no-op
    try std.testing.expectEqual(@as(usize, 1), impl.calls);
}
