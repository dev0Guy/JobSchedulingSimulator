//! GPU constants for the grid layer: how grids must be padded and aligned
//! so GPU kernels can process them efficiently.
//!
//! This file has no GPU API code. It defines the memory contract that a GPU
//! backend (Metal / CUDA / Vulkan, in adapters/) relies on:
//!   * rows padded to whole warps (SIMD-groups), so each thread of a warp
//!     reads one tick and no thread needs bounds checks;
//!   * grids aligned to `ALIGN_BYTES`, so loads are coalesced and buffers
//!     can be bound at any grid offset.
//!
//! Set at build time:
//!     zig build -Dgpu-warp=<threads> -Dgpu-align=<bytes>
//! 0 (default) = the defaults below.

const std = @import("std");
const opts = @import("build_options");
const types = @import("types.zig");
const cpu = @import("simd.zig");

pub const Elem = types.Elem;

/// Threads per warp / SIMD-group / wavefront.
/// NVIDIA: 32, Apple: 32, AMD: 32 or 64. Default 32.
pub const WARP: usize = if (opts.gpu_warp != 0) opts.gpu_warp else 32;

/// Alignment of every grid in a GPU buffer, in bytes.
/// 256 satisfies CUDA's device-pointer alignment and Metal/Vulkan
/// buffer-offset rules on common hardware.
pub const ALIGN_BYTES: usize = if (opts.gpu_align != 0) opts.gpu_align else 256;

/// Padding rule for grids processed only by the GPU.
pub const padding = types.Padding{ .lanes = WARP, .align_bytes = ALIGN_BYTES };

/// Padding rule for unified memory (Apple Silicon, CUDA managed memory):
/// the same buffer is used by CPU SIMD code (Usage views) and GPU kernels,
/// so it must satisfy both.
pub const unified_padding = types.Padding.combine(cpu.padding, padding);

/// Allocator alignment for unified buffers.
pub const UNIFIED_ALIGN = std.mem.Alignment.fromByteUnits(unified_padding.align_bytes);

/// Kernel launch helper: how many thread groups of `group_size` threads
/// cover `n` work items.
pub inline fn groups(n: usize, group_size: usize) usize {
    return (n + group_size - 1) / group_size;
}

comptime {
    if (!std.math.isPowerOfTwo(WARP))
        @compileError(std.fmt.comptimePrint("gpu-warp must be a power of two, got {d}", .{WARP}));
    if (!std.math.isPowerOfTwo(ALIGN_BYTES) or ALIGN_BYTES < 16)
        @compileError(std.fmt.comptimePrint("gpu-align must be a power of two >= 16, got {d}", .{ALIGN_BYTES}));
}

test "gpu padding" {
    try std.testing.expect(std.math.isPowerOfTwo(WARP));
    try unified_padding.validate();
    // Unified padding must be usable by the CPU SIMD code.
    try std.testing.expect(unified_padding.lanes % cpu.L == 0);
    try std.testing.expect(unified_padding.align_bytes % cpu.CACHE == 0);
    try std.testing.expectEqual(@as(usize, 4), groups(100, 32));
}
