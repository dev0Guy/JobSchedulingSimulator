//! Fixed: a hard-coded list of jobs (sorted by arrival). For tests and
//! reproducing specific scenarios.
const model = @import("../../model.zig");
const Error = @import("../../ports.zig").arrival_source.Error;

pub const Fixed = struct {
    items: []const model.JobMeta,
    i: usize = 0,

    pub fn next(self: *Fixed, now: i64) Error!?model.JobMeta {
        if (self.i < self.items.len and self.items[self.i].arrival <= now) {
            defer self.i += 1;
            return self.items[self.i];
        }
        return null;
    }
    pub fn done(self: *Fixed) bool {
        return self.i == self.items.len;
    }
    pub fn reset(self: *Fixed, seed: u64) void {
        _ = seed;
        self.i = 0;
    }
};
