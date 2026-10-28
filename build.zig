const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const targets = b.option([]const u8, "targets", "Comma-separated backends to compile in (x86_64,aarch64)") orelse "x86_64,aarch64";

    const opts = b.addOptions();
    const want_x86_64 = std.mem.indexOf(u8, targets, "x86_64") != null;
    const want_aarch64 = std.mem.indexOf(u8, targets, "aarch64") != null;
    opts.addOption(bool, "x86_64", want_x86_64);
    opts.addOption(bool, "aarch64", want_aarch64);
    // Validation: on by default in Debug/ReleaseSafe, off in ReleaseFast/ReleaseSmall.
    const safe_default = optimize == .Debug or optimize == .ReleaseSafe;
    opts.addOption(bool, "validate", b.option(bool, "validate", "Validate operands before emission (default: on in Debug/ReleaseSafe)") orelse safe_default);
    // Hooks stay available in every mode (Ivan: not disabled in release); -Dhooks=false opts out.
    opts.addOption(bool, "hooks", b.option(bool, "hooks", "Default Assembler supports runtime emission hooks (default: on in all modes)") orelse true);
    // Benchmark only: attach a comptime CountingHooks instead of the default CallbackHooks.
    opts.addOption(bool, "asmjit_tests", b.option(bool, "asmjit-tests", "test: include AsmJit's ported assembler tests (src/asmjit_x64_tests.zig, src/asmjit_x86_tests.zig)") orelse false);
    opts.addOption(bool, "gp_fuzz", b.option(bool, "gp-fuzz", "test: include src/gp_fuzz.zig (written by scripts/fuzz_gp.py)") orelse false);
    opts.addOption(bool, "bench_counting", b.option(bool, "bench-counting", "bench: use comptime CountingHooks (implies validate setting as given)") orelse false);

    const zigjit = b.addModule("zigjit", .{
        .root_source_file = b.path("src/zigjit.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    zigjit.addOptions("zigjit_options", opts);

    const tests = b.addTest(.{ .root_module = zigjit });
    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&b.addRunArtifact(tests).step);

    inline for (.{ "example_x86_64", "example_aarch64", "example_hooks", "bench_x86_64" }) |name| {
        const enabled = if (comptime std.mem.eql(u8, name, "example_hooks")) want_x86_64 and want_aarch64 else if (comptime std.mem.endsWith(u8, name, "aarch64")) want_aarch64 else want_x86_64;
        const dir = if (comptime std.mem.startsWith(u8, name, "bench")) "bench/" else "examples/";
        if (enabled) {
            const exe = b.addExecutable(.{
                .name = name,
                .root_module = b.createModule(.{
                    .root_source_file = b.path(dir ++ name ++ ".zig"),
                    .target = target,
                    .optimize = optimize,
                    .link_libc = true,
                    .imports = &.{.{ .name = "zigjit", .module = zigjit }},
                }),
            });
            b.installArtifact(exe);
            if (comptime std.mem.startsWith(u8, name, "bench")) {
                // Harness tests: scenarios must stay byte-identical to AsmJit's output.
                const bt = b.addTest(.{ .root_module = exe.root_module });
                test_step.dependOn(&b.addRunArtifact(bt).step);
            }
            const run = b.addRunArtifact(exe);
            if (b.args) |args| run.addArgs(args);
            b.step("run-" ++ name, "Run " ++ name).dependOn(&run.step);
        }
    }
}
