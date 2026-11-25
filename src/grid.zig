//! Grid layer: padded, aligned [resource][tick] grids of `Elem`.
//! CPU SIMD operations in `Usage`; padding rules for CPU, GPU and unified
//! memory in `simd` / `simd_gpu`. Knows nothing about jobs or machines.

pub const types = @import("grid/types.zig");
pub const Elem = types.Elem;
pub const Padding = types.Padding;

pub const simd = @import("grid/simd.zig");
pub const simd_gpu = @import("grid/simd_gpu.zig");

pub const Layout = @import("grid/layout.zig").Layout;
pub const Usage = @import("grid/usage.zig").Usage;

test {
    _ = types;
    _ = simd;
    _ = simd_gpu;
    _ = @import("grid/layout.zig");
    _ = @import("grid/usage.zig");
}
