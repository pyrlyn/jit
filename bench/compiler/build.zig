const std = @import("std");
// Standalone harness: builds bench_ir.zig against ../../src (this checkout), default options.
pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{}); // use -Doptimize=ReleaseFast
    const opts = b.addOptions();
    opts.addOption(bool, "x86_64", true);
    opts.addOption(bool, "aarch64", true);
    opts.addOption(bool, "validate", false);
    opts.addOption(bool, "hooks", true);
    opts.addOption(bool, "asmjit_tests", false);
    opts.addOption([]const u8, "x86_features", "");
    opts.addOption(bool, "gp_fuzz", false);
    opts.addOption(bool, "bench_counting", false);
    const pj = b.createModule(.{ .root_source_file = b.path("../../src/pyrlyn_jit.zig"), .target = target, .optimize = optimize, .link_libc = true });
    pj.addOptions("pyrlyn_jit_options", opts);
    const m = b.createModule(.{ .root_source_file = b.path("bench_ir.zig"), .target = target, .optimize = optimize, .link_libc = true });
    m.addImport("pyrlyn_jit", pj);
    b.installArtifact(b.addExecutable(.{ .name = "bench-ir", .root_module = m }));
}
