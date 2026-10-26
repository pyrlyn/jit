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

    inline for (.{ "example_x86_64", "example_aarch64", "bench_x86_64" }) |name| {
        const enabled = if (comptime std.mem.endsWith(u8, name, "aarch64")) want_aarch64 else want_x86_64;
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
            const run = b.addRunArtifact(exe);
            if (b.args) |args| run.addArgs(args);
            b.step("run-" ++ name, "Run " ++ name).dependOn(&run.step);
        }
    }
}
