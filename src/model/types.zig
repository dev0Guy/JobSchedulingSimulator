// A scheduling decision
pub const Action = union(enum) { place: struct { slot: u32, machine: u32 }, wait };
