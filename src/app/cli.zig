//! Command-line parsing: `--flag value` pairs and adapter specs of the form
//!     name:key=value,key=value
//! Values: integers, floats, booleans (1/0/true/false/yes/no),
//! ranges "lo-hi", lists "a/b/c".

const std = @import("std");

pub const Error = error{InvalidArgs};

pub fn fail(comptime fmt: []const u8, args: anytype) Error {
    std.debug.print("error: " ++ fmt ++ "\n", args);
    return error.InvalidArgs;
}

pub const Spec = struct {
    name: []const u8,
    keys: []const []const u8,
    values: []const []const u8,
    /// Flag this spec came from, for error messages.
    flag: []const u8,

    pub fn parse(arena: std.mem.Allocator, flag: []const u8, text: []const u8) !Spec {
        const colon = std.mem.indexOfScalar(u8, text, ':');
        const name = if (colon) |c| text[0..c] else text;
        if (name.len == 0) return fail("{s}: missing adapter name in '{s}'", .{ flag, text });

        var keys: std.ArrayList([]const u8) = .empty;
        var values: std.ArrayList([]const u8) = .empty;
        if (colon) |c| {
            var it = std.mem.splitScalar(u8, text[c + 1 ..], ',');
            while (it.next()) |kv| {
                if (kv.len == 0) continue;
                const eq = std.mem.indexOfScalar(u8, kv, '=') orelse
                    return fail("{s}: expected key=value, got '{s}'", .{ flag, kv });
                try keys.append(arena, kv[0..eq]);
                try values.append(arena, kv[eq + 1 ..]);
            }
        }
        return .{ .name = name, .keys = keys.items, .values = values.items, .flag = flag };
    }

    pub fn raw(self: Spec, key: []const u8) ?[]const u8 {
        for (self.keys, self.values) |k, v| if (std.mem.eql(u8, k, key)) return v;
        return null;
    }

    /// Reject keys not in `allowed` (catches typos).
    pub fn only(self: Spec, allowed: []const []const u8) Error!void {
        outer: for (self.keys) |k| {
            for (allowed) |a| if (std.mem.eql(u8, k, a)) continue :outer;
            return fail("{s} {s}: unknown option '{s}'", .{ self.flag, self.name, k });
        }
    }

    pub fn int(self: Spec, comptime T: type, key: []const u8, default: T) Error!T {
        const v = self.raw(key) orelse return default;
        return std.fmt.parseInt(T, v, 10) catch fail("{s} {s}: '{s}' is not a valid integer for {s}", .{ self.flag, self.name, v, key });
    }

    pub fn float(self: Spec, key: []const u8, default: f64) Error!f64 {
        const v = self.raw(key) orelse return default;
        return std.fmt.parseFloat(f64, v) catch fail("{s} {s}: '{s}' is not a valid number for {s}", .{ self.flag, self.name, v, key });
    }

    pub fn boolean(self: Spec, key: []const u8, default: bool) Error!bool {
        const v = self.raw(key) orelse return default;
        return parseBool(v) orelse fail("{s} {s}: '{s}' is not a boolean for {s}", .{ self.flag, self.name, v, key });
    }

    pub fn range(self: Spec, comptime T: type, key: []const u8, lo: T, hi: T) Error![2]T {
        const v = self.raw(key) orelse return .{ lo, hi };
        const dash = std.mem.indexOfScalar(u8, v, '-') orelse {
            const x = std.fmt.parseInt(T, v, 10) catch return fail("{s} {s}: bad range '{s}' for {s}", .{ self.flag, self.name, v, key });
            return .{ x, x };
        };
        const a = std.fmt.parseInt(T, v[0..dash], 10) catch return fail("{s} {s}: bad range '{s}' for {s}", .{ self.flag, self.name, v, key });
        const b = std.fmt.parseInt(T, v[dash + 1 ..], 10) catch return fail("{s} {s}: bad range '{s}' for {s}", .{ self.flag, self.name, v, key });
        if (a > b) return fail("{s} {s}: range '{s}' has lo > hi", .{ self.flag, self.name, v });
        return .{ a, b };
    }

    pub fn list(self: Spec, arena: std.mem.Allocator, comptime T: type, key: []const u8, default: []const T) ![]const T {
        const v = self.raw(key) orelse return default;
        var out: std.ArrayList(T) = .empty;
        var it = std.mem.splitScalar(u8, v, '/');
        while (it.next()) |p| try out.append(arena, std.fmt.parseInt(T, p, 10) catch
            return fail("{s} {s}: bad list '{s}' for {s}", .{ self.flag, self.name, v, key }));
        if (out.items.len == 0) return fail("{s} {s}: empty list for {s}", .{ self.flag, self.name, key });
        return out.items;
    }
};

pub fn parseBool(v: []const u8) ?bool {
    const t = [_][]const u8{ "1", "true", "yes", "on" };
    const f = [_][]const u8{ "0", "false", "no", "off" };
    for (t) |x| if (std.ascii.eqlIgnoreCase(v, x)) return true;
    for (f) |x| if (std.ascii.eqlIgnoreCase(v, x)) return false;
    return null;
}

test "spec parsing" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const s = try Spec.parse(a, "--machines", "random:n=4,caps=100/80,dur=2-9,varying=yes");
    try std.testing.expectEqualStrings("random", s.name);
    try std.testing.expectEqual(@as(usize, 4), try s.int(usize, "n", 1));
    try std.testing.expectEqualSlices(u16, &.{ 100, 80 }, try s.list(a, u16, "caps", &.{}));
    try std.testing.expectEqual([2]u16{ 2, 9 }, try s.range(u16, "dur", 1, 1));
    try std.testing.expect(try s.boolean("varying", false));
    try std.testing.expectEqual(@as(usize, 7), try s.int(usize, "missing", 7));
    try s.only(&.{ "n", "caps", "dur", "varying" });
    try std.testing.expectEqualStrings("fifo", (try Spec.parse(a, "--policy", "fifo")).name);
}
