//! Engine: applies actions and enforces the rules. It makes no decisions
//! (Policy port), creates no data itself (ArrivalSource, UsageGenerator
//! ports) and does no I/O (Observer port).
//!
//! One step = one action:
//!   .place → validate, subtract the job from the machine, emit job_placed
//!   .wait  → advance time one tick (or more with skip_empty), pull arrivals,
//!            emit tick_advanced, return the tick reward
//! Illegal actions are reported, penalized and treated as .wait.

const std = @import("std");
const grid = @import("../grid.zig");
const model = @import("../model.zig");
const ports = @import("../ports.zig");
pub const rules = @import("rules.zig");

const Elem = grid.Elem;
const Layout = grid.Layout;
const Machines = model.Machines;
const Jobs = model.Jobs;
const State = model.State;
const Action = model.Action;
const Event = model.Event;

pub const Config = rules.Config;

pub const StepResult = struct {
    reward: f32 = 0,
    terminated: bool = false,
    truncated: bool = false,
    placed: bool = false,
    illegal: bool = false,
};

pub const Stats = struct {
    arrived: usize = 0,
    placed: usize = 0,
    rejected: usize = 0,
    illegal: usize = 0,
    steps: usize = 0,
    total_wait: i64 = 0,
    max_wait: i64 = 0,
    makespan: i64 = 0,
    total_reward: f64 = 0,
    util_sum: f64 = 0, // sum over ticks of utilization at tick 0
    ticks: i64 = 0,

    pub fn avgWait(s: Stats) f64 {
        if (s.placed == 0) return 0;
        return @as(f64, @floatFromInt(s.total_wait)) / @as(f64, @floatFromInt(s.placed));
    }
    pub fn avgUtil(s: Stats) f64 {
        if (s.ticks == 0) return 0;
        return s.util_sum / @as(f64, @floatFromInt(s.ticks));
    }
};

pub const Engine = struct {
    gpa: std.mem.Allocator,
    cfg: Config,
    layout: Layout,
    machines: Machines,
    jobs: Jobs,
    arrivals: ports.ArrivalSource,
    generator: ports.UsageGenerator,
    observers: ports.ObserverList = .{},
    /// Tick reward; null = rules.defaultTickReward (−(waiting + running)).
    reward: ?ports.Reward = null,

    now: i64 = 0,
    seed: u64 = 0,
    running: std.ArrayList(model.RunningJob) = .empty,
    stats: Stats = .{},
    finished: bool = false,

    pub fn init(
        gpa: std.mem.Allocator,
        cfg: Config,
        arrivals: ports.ArrivalSource,
        generator: ports.UsageGenerator,
    ) !Engine {
        const layout = try Layout.init(cfg.n_res, cfg.n_ticks, cfg.padding);

        const n_m = generator.machineCount();
        if (n_m == 0) return error.NoMachines;
        const caps = try gpa.alloc(Elem, n_m * cfg.n_res);
        defer gpa.free(caps);
        for (0..n_m) |m| try generator.machine(m, caps[m * cfg.n_res ..][0..cfg.n_res]);

        var machines = try Machines.init(gpa, layout, n_m, caps);
        errdefer machines.deinit(gpa);
        const jobs = try Jobs.init(gpa, layout, cfg.max_jobs);

        return .{
            .gpa = gpa,
            .cfg = cfg,
            .layout = layout,
            .machines = machines,
            .jobs = jobs,
            .arrivals = arrivals,
            .generator = generator,
        };
    }

    pub fn deinit(self: *Engine) void {
        self.machines.deinit(self.gpa);
        self.jobs.deinit(self.gpa);
        self.running.deinit(self.gpa);
    }

    pub fn setReward(self: *Engine, r: ?ports.Reward) void {
        self.reward = r;
    }

    /// Jobs currently running (started, not finished).
    pub fn runningJobs(self: *const Engine) []const model.RunningJob {
        return self.running.items;
    }

    pub fn addObserver(self: *Engine, o: ports.Observer) !void {
        try self.observers.add(o);
    }

    pub fn state(self: *const Engine) State {
        return .{ .now = self.now, .machines = &self.machines, .jobs = &self.jobs };
    }

    pub fn done(self: *const Engine) bool {
        return self.finished;
    }

    // ------------------------------------------------------------------
    // episode
    // ------------------------------------------------------------------

    pub fn reset(self: *Engine, seed: u64) !void {
        self.seed = seed;
        self.arrivals.reset(seed);
        self.generator.reset(seed);
        if (self.reward) |r| r.reset(seed);
        self.machines.resetAll();
        self.jobs.clearAll();
        self.running.clearRetainingCapacity();
        self.now = 0;
        self.stats = .{};
        self.finished = false;

        self.emit(.{ .episode_started = .{ .seed = seed } });
        try self.pullArrivals();
        _ = self.checkEnd();
    }

    /// Apply one action.
    pub fn step(self: *Engine, action: Action) !StepResult {
        std.debug.assert(!self.finished);
        var res = StepResult{};
        self.stats.steps += 1;

        var act = action;
        if (!rules.isLegal(&self.state(), act)) {
            res.illegal = true;
            res.reward += self.cfg.illegal_penalty;
            self.stats.illegal += 1;
            self.emit(.{ .illegal_action = .{ .action = act, .tick = self.now } });
            act = .wait;
        }

        switch (act) {
            .place => |p| {
                try self.placeNow(p.slot, p.machine);
                res.placed = true;
            },
            .wait => {
                res.reward += try self.advanceTick();
                if (self.cfg.skip_empty) {
                    while (self.jobs.count == 0 and !self.endReached())
                        res.reward += try self.advanceTick();
                }
            },
        }

        const end = self.checkEnd();
        res.terminated = end == .terminated;
        res.truncated = end == .truncated;
        self.stats.total_reward += res.reward;
        return res;
    }

    /// Run a full episode with any policy that has
    /// `decide(self, *const State) Action` (comptime-dispatched, inlinable).
    pub fn run(self: *Engine, policy: anytype, max_steps: usize) !Stats {
        while (!self.finished and self.stats.steps < max_steps) {
            const st = self.state();
            _ = try self.step(policy.decide(&st));
        }
        return self.stats;
    }

    // ------------------------------------------------------------------
    // internals
    // ------------------------------------------------------------------

    fn emit(self: *Engine, ev: Event) void {
        if (self.observers.len == 0) return;
        const st = self.state();
        self.observers.emit(ev, &st);
    }

    fn placeNow(self: *Engine, slot: usize, m: usize) !void {
        const dur = self.jobs.duration[slot];
        self.machines.get(m).place(self.jobs.get(slot), dur, 0);

        const wait = self.now - self.jobs.arrival[slot];
        const finish = self.now + dur;
        try self.running.append(self.gpa, .{ .id = self.jobs.id[slot], .machine = @intCast(m), .start = self.now, .dur = dur });
        self.stats.placed += 1;
        self.stats.total_wait += wait;
        self.stats.max_wait = @max(self.stats.max_wait, wait);
        self.stats.makespan = @max(self.stats.makespan, finish);

        self.emit(.{ .job_placed = .{
            .id = self.jobs.id[slot],
            .slot = @intCast(slot),
            .machine = @intCast(m),
            .tick = self.now,
            .arrival = self.jobs.arrival[slot],
            .dur = dur,
        } });
        self.jobs.remove(slot);
    }

    /// One tick of time. Returns that tick's reward.
    fn advanceTick(self: *Engine) !f32 {
        const reward = if (self.reward) |r| blk: {
            const st = self.state();
            break :blk r.tick(&.{ .state = &st, .running = self.running.items });
        } else rules.defaultTickReward(self.jobs.count, self.running.items.len);
        self.stats.util_sum += self.machines.utilizationNow();
        self.stats.ticks += 1;

        self.machines.advanceAll(1);
        self.now += 1;

        var i: usize = 0;
        while (i < self.running.items.len) {
            if (self.running.items[i].finish() <= self.now) {
                _ = self.running.swapRemove(i);
            } else i += 1;
        }

        try self.pullArrivals();
        self.jobs.clearBlocked(); // capacity freed up: re-check everything
        self.emit(.{ .tick_advanced = .{ .now = self.now } });
        return reward;
    }

    fn pullArrivals(self: *Engine) !void {
        while (try self.arrivals.next(self.now)) |meta| {
            const slot = self.jobs.acquire() orelse {
                self.reject(meta.id, .pool_full);
                continue;
            };
            const dur = self.generator.job(meta, self.jobs.get(slot)) catch |e| switch (e) {
                error.OutOfMemory => return e,
                else => {
                    self.jobs.abort(slot);
                    self.reject(meta.id, .invalid_profile);
                    continue;
                },
            };
            if (dur == 0 or dur > self.layout.n_ticks) {
                self.jobs.abort(slot);
                self.reject(meta.id, .invalid_profile);
                continue;
            }
            self.jobs.commit(slot, meta, dur);

            if (self.cfg.drop_impossible and !self.machines.anyCanEverFit(self.jobs.peakOf(slot))) {
                self.jobs.remove(slot);
                self.reject(meta.id, .impossible);
                continue;
            }
            self.stats.arrived += 1;
            self.emit(.{ .job_arrived = .{ .id = meta.id, .slot = @intCast(slot), .tick = self.now } });
        }
    }

    fn reject(self: *Engine, id: u32, reason: model.RejectReason) void {
        self.stats.rejected += 1;
        self.emit(.{ .job_rejected = .{ .id = id, .reason = reason, .tick = self.now } });
    }

    const End = enum { none, terminated, truncated };

    fn endReached(self: *const Engine) bool {
        const all_done = self.arrivals.done() and self.jobs.count == 0 and self.running.items.len == 0;
        const trunc = if (self.cfg.max_ticks) |mt| self.now >= mt else false;
        return all_done or trunc;
    }

    fn checkEnd(self: *Engine) End {
        if (self.finished) return .none;
        const all_done = self.arrivals.done() and self.jobs.count == 0 and self.running.items.len == 0;
        const trunc = if (self.cfg.max_ticks) |mt| self.now >= mt else false;
        if (!all_done and !trunc) return .none;
        self.finished = true;
        self.emit(.{ .episode_done = .{ .now = self.now, .makespan = self.stats.makespan, .truncated = !all_done } });
        return if (all_done) .terminated else .truncated;
    }
};
