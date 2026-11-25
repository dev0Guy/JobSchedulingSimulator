//! Runtime shape of a [resource][tick] grid, padded by a Padding rule
//! (CPU, GPU or unified). Every grid used together must share one Layout.
//!
//!     row r:  tick 0 … tick n_ticks-1 | padding |   ← stride elements
//!
//!   * stride: multiple of padding.lanes, with at least `lanes` slack after
//!     the real ticks, so windows rounded up to whole vectors / warps stay
//!     inside their row.
//!   * block: elements per grid, rounded so each grid in a pool starts at a
//!     multiple of padding.align_bytes.

const std = @import("std");
const types = @import("types.zig");
const Elem = types.Elem;
pub const Padding = types.Padding;

pub const Layout = struct {
    n_res: usize,
    n_ticks: usize,
    stride: usize,
    block: usize,
    padding: Padding,

    pub const Error = error{ ZeroResources, ZeroTicks, InvalidPadding };

    pub fn init(n_res: usize, n_ticks: usize, padding: Padding) Error!Layout {
        if (n_res == 0) return error.ZeroResources;
        if (n_ticks == 0) return error.ZeroTicks;
        try padding.validate();

        const stride = std.mem.alignForward(usize, n_ticks + padding.lanes, padding.lanes);
        const per_align = @max(1, padding.align_bytes / @sizeOf(Elem));
        return .{
            .n_res = n_res,
            .n_ticks = n_ticks,
            .stride = stride,
            .block = std.mem.alignForward(usize, n_res * stride, per_align),
            .padding = padding,
        };
    }

    pub fn bytesPerGrid(self: Layout) usize {
        return self.block * @sizeOf(Elem);
    }

    pub inline fn rowOffset(self: Layout, r: usize) usize {
        return r * self.stride;
    }

    pub inline fn gridOffset(self: Layout, i: usize) usize {
        return i * self.block;
    }

    pub fn eql(a: Layout, b: Layout) bool {
        return a.n_res == b.n_res and a.n_ticks == b.n_ticks and
            a.stride == b.stride and a.block == b.block;
    }
};

test "layout for cpu, gpu and unified padding" {
    const t = std.testing;
    const pads = [_]Padding{
        .{ .lanes = 8, .align_bytes = 128 },
        .{ .lanes = 32, .align_bytes = 256 },
        Padding.combine(.{ .lanes = 16, .align_bytes = 64 }, .{ .lanes = 32, .align_bytes = 256 }),
    };
    for (pads) |p| {
        for ([_]usize{ 1, 7, 100, 128 }) |n| {
            const lay = try Layout.init(10, n, p);
            try t.expect(lay.stride % p.lanes == 0);
            try t.expect(lay.stride >= n + p.lanes);
            try t.expect(lay.bytesPerGrid() % p.align_bytes == 0);
        }
    }
    try t.expectError(error.ZeroResources, Layout.init(0, 10, pads[0]));
    try t.expectError(error.InvalidPadding, Layout.init(1, 10, .{ .lanes = 3, .align_bytes = 64 }));
}
