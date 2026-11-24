//! Benchmarks for grid/simd.zig.
//!
//!   1. SIMD vs scalar for the two hot kernels (compare = fit test, sub = place).
//!   2. Working-set sweep 4 KiB … 256 MiB: throughput drops at each cache
//!      level (L1 → L2 → L3 → RAM), which shows how much caching matters.
//!   3. Aligned vs unaligned vector loads (machine reads at offset t).
//!
//! Run:  zig build bench            (always ReleaseFast)
//!       zig build bench -Dlanes=4  (compare vector widths)

const std = @import("std");
const simd = @import("sched").grid.simd;
const Elem = simd.Elem;

const L = simd.L;
const V = simd.V;

// ------------------------------------------------------------------
// kernels (noinline so they can be found in the disassembly by name)
// ------------------------------------------------------------------

/// SIMD fit test: all need[i] <= have[i] ?
noinline fn coversSimd(need: []const Elem, have: []const Elem) bool {
    var i: usize = 0;
    while (i < need.len) : (i += L)
        if (@reduce(.Or, simd.load(need, i) > simd.load(have, i))) return false;
    return true;
}

/// Scalar fit test. The early exit per element keeps LLVM from
/// auto-vectorizing it, so it is a fair scalar baseline.
noinline fn coversScalar(need: []const Elem, have: []const Elem) bool {
    for (need, have) |n, h| if (n > h) return false;
    return true;
}

/// SIMD place: have[i] -= need[i]
noinline fn subSimd(have: []Elem, need: []const Elem) void {
    var i: usize = 0;
    while (i < need.len) : (i += L) simd.store(have, i, simd.load(have, i) - simd.load(need, i));
}

/// Same as subSimd but every vector load/store is shifted by one element
/// (4 bytes), so loads are unaligned and every L-th one splits a cache line.
noinline fn subSimdUnaligned(have: []Elem, need: []const Elem) void {
    var i: usize = 0;
    const n = need.len - L; // room for the +1 shift
    while (i < n) : (i += L) simd.store(have, i + 1, simd.load(have, i + 1) - simd.load(need, i + 1));
}

// ------------------------------------------------------------------
// timing
// ------------------------------------------------------------------

var io: std.Io = undefined;

fn nowNs() i96 {
    return std.Io.Timestamp.now(io, .awake).toNanoseconds();
}

const Kernel = enum { covers_simd, covers_scalar, sub_simd, sub_unaligned };

/// Best-of-5 nanoseconds per element for `k` over arrays of `n` elements.
fn measure(k: Kernel, need: []Elem, have: []Elem) f64 {
    const n = need.len;
    // Repeat so each sample touches ~256 MiB of data in total.
    const reps: usize = @max(1, (256 << 20) / (n * 2 * @sizeOf(Elem)));
    var best: f64 = std.math.inf(f64);
    for (0..5) |_| {
        const t0 = nowNs();
        for (0..reps) |_| {
            switch (k) {
                .covers_simd => std.mem.doNotOptimizeAway(coversSimd(need, have)),
                .covers_scalar => std.mem.doNotOptimizeAway(coversScalar(need, have)),
                .sub_simd => subSimd(have, need),
                .sub_unaligned => subSimdUnaligned(have, need),
            }
            std.mem.doNotOptimizeAway(have.ptr);
        }
        const dt: f64 = @floatFromInt(nowNs() - t0);
        best = @min(best, dt / @as(f64, @floatFromInt(reps * n)));
    }
    return best;
}

fn gbps(ns_per_elem: f64, bytes_per_elem: f64) f64 {
    return bytes_per_elem / ns_per_elem; // bytes/ns == GB/s
}

pub fn main(init: std.process.Init) !void {
    io = init.io;
    const gpa = init.gpa;

    std.debug.print(
        \\simd bench   Elem = {s}   L = {d} lanes ({d}-bit vectors)   CACHE = {d} B
        \\
        \\
    , .{ @typeName(Elem), L, L * @bitSizeOf(Elem), simd.CACHE });

    const sizes_kib = [_]usize{ 4, 16, 32, 64, 256, 1024, 4096, 16384, 65536, 262144 };
    const max_n = sizes_kib[sizes_kib.len - 1] * 1024 / @sizeOf(Elem) / 2; // two arrays share the size

    const need_all = try gpa.alignedAlloc(Elem, simd.ALIGN, max_n);
    defer gpa.free(need_all);
    const have_all = try gpa.alignedAlloc(Elem, simd.ALIGN, max_n);
    defer gpa.free(have_all);
    // need = 0, have = max → the fit test never exits early (full scan),
    // and sub never underflows across many repetitions.
    @memset(need_all, 0);
    @memset(have_all, std.math.maxInt(Elem));

    std.debug.print("working set = both arrays together; ns/elem (lower is better), GB/s (higher is better)\n\n", .{});
    std.debug.print("{s:>10} | {s:>22} | {s:>22} | {s:>8} | {s:>22} | {s:>22}\n", .{
        "set", "covers SIMD", "covers scalar", "speedup", "sub SIMD aligned", "sub SIMD unaligned",
    });
    std.debug.print("{s:->10}-+-{s:->22}-+-{s:->22}-+-{s:->8}-+-{s:->22}-+-{s:->22}\n", .{ "", "", "", "", "", "" });

    for (sizes_kib) |kib| {
        const n = std.mem.alignForward(usize, kib * 1024 / @sizeOf(Elem) / 2, L);
        const need = need_all[0..n];
        const have = have_all[0..n];

        const cs = measure(.covers_simd, need, have);
        const cr = measure(.covers_scalar, need, have);
        const ss = measure(.sub_simd, need, have);
        const su = measure(.sub_unaligned, need, have);

        var label: [16]u8 = undefined;
        const lbl = if (kib >= 1024)
            try std.fmt.bufPrint(&label, "{d} MiB", .{kib / 1024})
        else
            try std.fmt.bufPrint(&label, "{d} KiB", .{kib});

        std.debug.print("{s:>10} | {d:>8.3} ns {d:>7.1} GB/s | {d:>8.3} ns {d:>7.1} GB/s | {d:>7.1}x | {d:>8.3} ns {d:>7.1} GB/s | {d:>8.3} ns {d:>7.1} GB/s\n", .{
            lbl,
            cs, gbps(cs, 2 * @sizeOf(Elem)),
            cr, gbps(cr, 2 * @sizeOf(Elem)),
            cr / cs,
            ss, gbps(ss, 3 * @sizeOf(Elem)), // 2 loads + 1 store
            su, gbps(su, 3 * @sizeOf(Elem)),
        });
    }

    std.debug.print(
        \\
        \\How to read it:
        \\  * speedup  = scalar / SIMD for the fit test; close to L in L1, smaller
        \\               once memory bandwidth is the limit.
        \\  * rows where GB/s drops sharply mark cache boundaries (L1 → L2 → L3 → RAM).
        \\    Your simulator's hot data (one machine ≈ 5 KiB, metadata ≈ 25 KiB) sits
        \\    in the first rows; the 2.5 MiB job pool around the 4 MiB row.
        \\  * aligned vs unaligned: the cost of machine reads at an arbitrary tick t.
        \\
    , .{});
}
