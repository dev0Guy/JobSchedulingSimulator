//! UsageGenerator port: creates resource usage — job requirement profiles
//! (the [resource][tick] grid of a job) and machine capacities.
//!
//! The engine owns all memory. It hands the generator:
//!   * for a job: an already CLEARED Usage view of the job's slot; the
//!     generator writes the profile (via `set` / `resourceMut`) for ticks
//!     [0, dur) and returns `dur`. The engine then enforces the invariants
//!     (1 <= dur <= n_ticks, zero past dur, values >= 0) and computes peaks.
//!   * for a machine: an `n_res`-long slice to fill with capacities.
//!
//! Splitting "what a job needs" (this port) from "when a job arrives"
//! (ArrivalSource) lets you mix any arrival process with any profile
//! distribution, e.g. a real trace's arrival times with synthetic profiles.
//!
//! Contract for an implementation type `T`:
//!     pub fn machineCount(self: *T) usize
//!     pub fn machine(self: *T, index: usize, cap: []Elem) Error!void
//!     pub fn job(self: *T, meta: JobMeta, out: Usage) Error!u16
//! optional:
//!     pub fn reset(self: *T, seed: u64) void
//!     pub fn deinit(self: *T) void
//!
//! Rules:
//!   * Deterministic for a given seed (reproducible experiments).
//!   * Write only real ticks [0, dur) of `out`; never write padding.
//!   * Do not keep the `out` view after returning — it points into the pool.

const std = @import("std");
const types = @import("../model/types.zig");
const grid = @import("../grid.zig");
const Usage = grid.Usage;
const Elem = grid.Elem;
pub const JobMeta = types.JobMeta;

pub const Error = error{ OutOfMemory, InvalidInput, ReadFailed, EndOfData };

pub const UsageGenerator = struct {
    ptr: *anyopaque,
    vtable: *const VTable,

    pub const VTable = struct {
        machineCount: *const fn (ptr: *anyopaque) usize,
        machine: *const fn (ptr: *anyopaque, index: usize, cap: []Elem) Error!void,
        job: *const fn (ptr: *anyopaque, meta: JobMeta, out: Usage) Error!u16,
        reset: *const fn (ptr: *anyopaque, seed: u64) void,
        deinit: *const fn (ptr: *anyopaque) void,
    };

    pub fn machineCount(self: UsageGenerator) usize {
        return self.vtable.machineCount(self.ptr);
    }
    pub fn machine(self: UsageGenerator, index: usize, cap: []Elem) Error!void {
        return self.vtable.machine(self.ptr, index, cap);
    }
    pub fn job(self: UsageGenerator, meta: JobMeta, out: Usage) Error!u16 {
        return self.vtable.job(self.ptr, meta, out);
    }
    pub fn reset(self: UsageGenerator, seed: u64) void {
        self.vtable.reset(self.ptr, seed);
    }
    pub fn deinit(self: UsageGenerator) void {
        self.vtable.deinit(self.ptr);
    }

    pub fn init(impl: anytype) UsageGenerator {
        const Ptr = @TypeOf(impl);
        const T = @typeInfo(Ptr).pointer.child;
        inline for (.{ "machineCount", "machine", "job" }) |name| {
            if (!@hasDecl(T, name))
                @compileError(@typeName(T) ++ " is not a UsageGenerator: missing `" ++ name ++ "`");
        }
        const gen = struct {
            fn cast(p: *anyopaque) Ptr {
                return @ptrCast(@alignCast(p));
            }
            fn machineCount(p: *anyopaque) usize {
                return cast(p).machineCount();
            }
            fn machine(p: *anyopaque, index: usize, cap: []Elem) Error!void {
                return cast(p).machine(index, cap);
            }
            fn job(p: *anyopaque, meta: JobMeta, out: Usage) Error!u16 {
                return cast(p).job(meta, out);
            }
            fn reset(p: *anyopaque, seed: u64) void {
                if (@hasDecl(T, "reset")) cast(p).reset(seed);
            }
            fn deinit(p: *anyopaque) void {
                if (@hasDecl(T, "deinit")) cast(p).deinit();
            }
        };
        const vt = VTable{
                .machineCount = gen.machineCount,
                .machine = gen.machine,
                .job = gen.job,
                .reset = gen.reset,
                .deinit = gen.deinit,
            };
        return .{ .ptr = impl, .vtable = &vt };
    }
};

test "usage generator port" {
    const Layout = grid.Layout;
    const simd = grid.simd;

    // Example adapter: N identical machines, constant-demand random jobs.
    const Uniform = struct {
        n: usize,
        cap: Elem,
        prng: std.Random.DefaultPrng,

        pub fn machineCount(self: *@This()) usize {
            return self.n;
        }
        pub fn machine(self: *@This(), index: usize, cap: []Elem) Error!void {
            _ = index;
            @memset(cap, self.cap);
        }
        pub fn job(self: *@This(), meta: JobMeta, out: Usage) Error!u16 {
            _ = meta;
            const r = self.prng.random();
            const dur = r.intRangeAtMost(usize, 1, @min(5, out.layout.n_ticks));
            for (0..out.layout.n_res) |res|
                @memset(out.resourceMut(res)[0..dur], r.intRangeAtMost(Elem, 0, self.cap));
            return @intCast(dur);
        }
        pub fn reset(self: *@This(), seed: u64) void {
            self.prng = .init(seed);
        }
    };

    var impl = Uniform{ .n = 3, .cap = 10, .prng = .init(1) };
    const g = UsageGenerator.init(&impl);
    try std.testing.expectEqual(@as(usize, 3), g.machineCount());

    var cap: [2]Elem = undefined;
    try g.machine(0, &cap);
    try std.testing.expectEqual(@as(Elem, 10), cap[1]);

    const layout = try Layout.init(2, 8, simd.padding);
    const a = std.testing.allocator;
    const buf = try a.alignedAlloc(Elem, simd.ALIGN, layout.block);
    defer a.free(buf);
    @memset(buf, 0);
    const dur = try g.job(.{ .id = 0, .arrival = 0 }, Usage.fromSlice(buf, layout));
    try std.testing.expect(dur >= 1 and dur <= 5);
    g.reset(1);
    g.deinit();
}
