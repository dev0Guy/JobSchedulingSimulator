const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // ---- build options (0 = auto / default) ----
    const options = b.addOptions();
    options.addOption(usize, "cache_line", b.option(usize, "cache-line", "CPU cache line in bytes (0 = auto)") orelse 0);
    options.addOption(usize, "lanes", b.option(usize, "lanes", "CPU SIMD lanes (0 = auto)") orelse 0);
    options.addOption(usize, "gpu_warp", b.option(usize, "gpu-warp", "GPU threads per warp (0 = 32)") orelse 0);
    options.addOption(usize, "gpu_align", b.option(usize, "gpu-align", "GPU grid alignment in bytes (0 = 256)") orelse 0);

    const lib = libModule(b, target, optimize, options);
    const lib_fast = libModule(b, target, .ReleaseFast, options);

    // ---- simulator: zig build run -- <args> ----
    const exe = b.addExecutable(.{
        .name = "sim",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/app/main.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{.{ .name = "sched", .module = lib }},
        }),
    });
    b.installArtifact(exe);
    const run = b.addRunArtifact(exe);
    run.addPassthruArgs(); // zig build run -- <args>
    b.step("run", "Run the simulator (args after --)").dependOn(&run.step);

    // ---- tests: zig build test ----
    const test_step = b.step("test", "Run unit and integration tests");
    const unit = b.addTest(.{ .root_module = lib });
    test_step.dependOn(&b.addRunArtifact(unit).step);
    const app_tests = b.addTest(.{ .root_module = exe.root_module });
    test_step.dependOn(&b.addRunArtifact(app_tests).step);
    const integ = b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("tests/engine_test.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{.{ .name = "sched", .module = lib }},
    }) });
    test_step.dependOn(&b.addRunArtifact(integ).step);

    // ---- C ABI shared library for Python: zig build lib ----
    // Default ReleaseFast (RL needs speed); override with -Doptimize.
    const capi_mod = b.createModule(.{
        .root_source_file = b.path("src/app/capi.zig"),
        .target = target,
        .optimize = if (optimize == .debug) .fast else optimize,
        .imports = &.{.{ .name = "sched", .module = if (optimize == .debug) lib_fast else lib }},
    });
    const capi = b.addLibrary(.{ .linkage = .dynamic, .name = "sched", .root_module = capi_mod });
    const capi_install = b.addInstallArtifact(capi, .{});
    b.step("lib", "Build libsched (C ABI for Python) into zig-out/lib").dependOn(&capi_install.step);
    const capi_tests = b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("src/app/capi.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{.{ .name = "sched", .module = lib }},
    }) });
    test_step.dependOn(&b.addRunArtifact(capi_tests).step);

    // ---- benchmarks: zig build bench (always ReleaseFast) ----
    const bench_step = b.step("bench", "Run benchmarks (ReleaseFast)");
    for ([_][]const u8{ "simd_bench", "engine_bench" }) |name| {
        const exe_b = b.addExecutable(.{
            .name = name,
            .root_module = b.createModule(.{
                .root_source_file = b.path(b.fmt("bench/{s}.zig", .{name})),
                .target = target,
                .optimize = .ReleaseFast,
                .imports = &.{.{ .name = "sched", .module = lib_fast }},
            }),
        });
        b.installArtifact(exe_b);
        bench_step.dependOn(&b.addRunArtifact(exe_b).step);
    }

    // ---- CPU side of the CPU-vs-GPU comparison: zig build fitmask ----
    const fm = b.addExecutable(.{
        .name = "fitmask_bench",
        .root_module = b.createModule(.{
            .root_source_file = b.path("bench/fitmask_bench.zig"),
            .target = target,
            .optimize = .ReleaseFast,
            .imports = &.{.{ .name = "sched", .module = lib_fast }},
        }),
    });
    b.installArtifact(fm);
    const fm_run = b.addRunArtifact(fm);
    fm_run.addPassthruArgs();
    b.step("fitmask", "CPU fit-mask benchmark (compare with bench/compare.py)").dependOn(&fm_run.step);

    // ---- examples: zig build example ----
    const ex = b.addExecutable(.{
        .name = "custom_policy",
        .root_module = b.createModule(.{
            .root_source_file = b.path("examples/custom_policy.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{.{ .name = "sched", .module = lib }},
        }),
    });
    b.installArtifact(ex);
    b.step("example", "Run examples/custom_policy.zig").dependOn(&b.addRunArtifact(ex).step);
}

fn libModule(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    options: *std.Build.Step.Options,
) *std.Build.Module {
    const m = b.createModule(.{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });
    m.addOptions("build_options", options);
    return m;
}
