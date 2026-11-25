//! Simulation rules: configuration, legality, default reward, action encoding.
//! Shared by the engine (enforcement) and by policies (queries), so both
//! use exactly the same definition of "fits".

const std = @import("std");
const grid = @import("../grid.zig");
const model = @import("../model.zig");
const State = model.State;
const Action = model.Action;

pub const Config = struct {
    n_res: usize = 10,
    n_ticks: usize = 100,
    /// Job slots in the waiting pool (observation size).
    max_jobs: usize = 128,
    /// Memory padding rule (CPU, or grid.simd_gpu.unified_padding).
    padding: grid.Padding = grid.simd.padding,
    /// Added to the reward of an illegal action (then treated as wait).
    illegal_penalty: f32 = -1.0,
    /// Truncate the episode at this tick (null = never).
    max_ticks: ?i64 = null,
    /// Drop arriving jobs that no machine could ever run.
    drop_impossible: bool = true,
    /// When the pool is empty, fast-forward time until a job arrives.
    skip_empty: bool = true,
};

/// Can `slot` start now (tick 0) on machine `m`? Cheap checks, then SIMD.
pub fn fitsNow(state: *const State, slot: usize, m: usize) bool {
    const jobs = state.jobs;
    const mach = state.machines.get(m);
    const job = jobs.get(slot);
    if (!mach.mayFit(job, jobs.peakOf(slot), 0)) return false;
    return mach.canPlace(job, jobs.duration[slot], 0);
}

/// Is `action` legal in `state`?
pub fn isLegal(state: *const State, action: Action) bool {
    return switch (action) {
        .wait => true,
        .place => |p| p.slot < state.jobs.cap and state.jobs.isValid(p.slot) and
            p.machine < state.machines.n and fitsNow(state, p.slot, p.machine),
    };
}

/// First machine on which `slot` fits now, or null.
pub fn firstMachine(state: *const State, slot: usize) ?usize {
    if (state.jobs.isBlocked(slot)) return null;
    for (0..state.machines.n) |m| if (fitsNow(state, slot, m)) return m;
    state.jobs.markBlocked(slot);
    return null;
}

/// Known (this tick) to fit on no machine? Policies that scan all
/// (job, machine) pairs skip such jobs; after a scan that found no machine
/// they call markBlocked. Valid until time advances (the engine clears it).
pub inline fn knownBlocked(state: *const State, slot: usize) bool {
    return state.jobs.isBlocked(slot);
}
pub inline fn markBlocked(state: *const State, slot: usize) void {
    state.jobs.markBlocked(slot);
}

/// Default tick reward when no Reward port is set: −(waiting + running).
pub fn defaultTickReward(waiting: usize, running: usize) f32 {
    return -@as(f32, @floatFromInt(waiting + running));
}

// ---------------- flat action encoding (for RL / Gymnasium) ----------------
// index < max_jobs * n_machines : place(slot = i / n_machines, machine = i % n_machines)
// index == max_jobs * n_machines: wait

pub fn actionCount(state: *const State) usize {
    return state.jobs.cap * state.machines.n + 1;
}

pub fn decode(state: *const State, index: usize) Action {
    const n_m = state.machines.n;
    if (index >= state.jobs.cap * n_m) return .wait;
    return .{ .place = .{ .slot = @intCast(index / n_m), .machine = @intCast(index % n_m) } };
}

pub fn encode(state: *const State, action: Action) usize {
    return switch (action) {
        .wait => state.jobs.cap * state.machines.n,
        .place => |p| p.slot * state.machines.n + p.machine,
    };
}

/// mask[i] = action i is legal. `mask.len == actionCount(state)`.
pub fn actionMask(state: *const State, mask: []bool) void {
    std.debug.assert(mask.len == actionCount(state));
    @memset(mask, false);
    mask[mask.len - 1] = true; // wait
    var it = state.jobs.iter();
    while (it.next()) |s| {
        if (state.jobs.isBlocked(s)) continue;
        var any = false;
        for (0..state.machines.n) |m| {
            const f = fitsNow(state, s, m);
            mask[s * state.machines.n + m] = f;
            any = any or f;
        }
        if (!any) state.jobs.markBlocked(s);
    }
}
