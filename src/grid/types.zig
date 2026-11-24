//! Types shared by every part of the grid layer, CPU and GPU.

const std = @import("std");

/// Element type of every grid cell (demand / free capacity).
/// u16: 0 … 65535, half the memory of i32 and twice the values per vector.
/// Change it here only; everything else derives from it.
pub const Elem = u16;

/// How a grid is padded and aligned in memory. Each target (CPU SIMD,
/// GPU, unified CPU+GPU) provides its own; Layout is computed from it.
pub const Padding = struct {
    /// Every row length (stride) is a multiple of this many elements:
    /// SIMD lanes on the CPU, threads per warp / SIMD-group on the GPU.
    lanes: usize,
    /// Every grid starts at a multiple of this many bytes:
    /// cache line on the CPU, buffer/coalescing alignment on the GPU.
    align_bytes: usize,

    /// Padding that satisfies both a and b (for unified memory, where the
    /// CPU and the GPU work on the same buffer).
    pub fn combine(a: Padding, b: Padding) Padding {
        return .{
            .lanes = std.math.lcm(a.lanes, b.lanes),
            .align_bytes = @max(a.align_bytes, b.align_bytes),
        };
    }

    pub fn validate(self: Padding) error{InvalidPadding}!void {
        if (self.lanes == 0 or !std.math.isPowerOfTwo(self.lanes)) return error.InvalidPadding;
        if (!std.math.isPowerOfTwo(self.align_bytes)) return error.InvalidPadding;
        if (self.align_bytes < @sizeOf(Elem)) return error.InvalidPadding;
    }
};

test "padding combine" {
    const cpu = Padding{ .lanes = 8, .align_bytes = 128 };
    const gpu = Padding{ .lanes = 32, .align_bytes = 256 };
    const u = Padding.combine(cpu, gpu);
    try std.testing.expectEqual(@as(usize, 32), u.lanes);
    try std.testing.expectEqual(@as(usize, 256), u.align_bytes);
    try u.validate();
    try std.testing.expectError(error.InvalidPadding, (Padding{ .lanes = 12, .align_bytes = 64 }).validate());
}
