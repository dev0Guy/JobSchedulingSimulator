//! CPU SIMD constants and vector helpers for the grid layer.
//!
//! Set at build time:
//!     zig build -Dlanes=<n> -Dcache-line=<bytes>
//! 0 (default) = derive from the compilation target.

const std = @import("std");
const opts = @import("build_options");
const types = @import("types.zig");

pub const Elem = types.Elem;

/// Cache-line size in bytes (target arch default if not set).
pub const CACHE: usize = if (opts.cache_line != 0) opts.cache_line else std.atomic.cache_line;

/// Elements per SIMD vector (target CPU default if not set).
/// For u16: NEON → 8, AVX2 → 16, AVX-512 → 32.
pub const L: usize = if (opts.lanes != 0) opts.lanes else (std.simd.suggestVectorLength(Elem) orelse 8);

/// One SIMD vector of grid elements.
pub const V = @Vector(L, Elem);

/// Allocator alignment for CPU grid buffers.
pub const ALIGN = std.mem.Alignment.fromByteUnits(CACHE);

/// Padding rule for grids processed by the CPU.
pub const padding = types.Padding{ .lanes = L, .align_bytes = CACHE };

/// Load L consecutive elements starting at p[i].
pub inline fn load(p: []const Elem, i: usize) V {
    return p[i..][0..L].*;
}

/// Store one vector at p[i].
pub inline fn store(p: []Elem, i: usize, v: V) void {
    p[i..][0..L].* = v;
}

/// Round n up to whole vectors.
pub inline fn span(n: usize) usize {
    return std.mem.alignForward(usize, n, L);
}

comptime {
    if (!std.math.isPowerOfTwo(CACHE))
        @compileError(std.fmt.comptimePrint("cache-line must be a power of two, got {d}", .{CACHE}));
    if (CACHE < 16)
        @compileError(std.fmt.comptimePrint("cache-line must be at least 16 bytes, got {d}", .{CACHE}));
    if (!std.math.isPowerOfTwo(L))
        @compileError(std.fmt.comptimePrint("lanes must be a power of two, got {d}", .{L}));
}

test "cpu simd helpers" {
    try std.testing.expect(std.math.isPowerOfTwo(L));
    try std.testing.expectEqual(@as(usize, 0), span(0));
    try std.testing.expectEqual(L, span(1));

    var a: [2 * L]Elem = @splat(1);
    store(&a, L, load(&a, 0) + @as(V, @splat(1)));
    try std.testing.expectEqual(@as(Elem, 2), a[L]);
    try std.testing.expectEqual(@as(Elem, 1), a[0]);
}
