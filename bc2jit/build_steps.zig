//! Build steps for bc2jit, called from the root build.zig:
//!   zig build bc2jit        install zig-out/bin/bc2jit
//!   zig build test-bc2jit   unit tests (also part of `zig build test`)
//!   zig build e2e-bc2jit    run the CLI on KXBC fixtures with --arch = target arch and check results
//! Foreign targets run through the build system's emulator flags (`-frosetta`, `-fqemu`) and are
//! skipped when none is available.
const std = @import("std");

/// One end-to-end case: a module, its arguments and the expected stdout.
const Case = struct { name: []const u8, module: []const u8, args: []const []const u8, stdout: []const u8 };

/// KXBC v0 module bytes: header (magic, version 0, argc, u16 code length) + code.
fn module(comptime argc: u8, comptime code: []const u8) []const u8 {
    return "KXBC" ++ [_]u8{ 0, argc, @truncate(code.len), @truncate(code.len >> 8) } ++ code;
}
fn imm(comptime v: i64) []const u8 {
    var b: [8]u8 = undefined;
    std.mem.writeInt(i64, &b, v, .little);
    return &([_]u8{0x01} ++ b);
}
const ADD = "\x10";
const SUB = "\x11";
const MUL = "\x12";
const DUP = "\x20";
const DROP = "\x21";
const RET = "\xF0";
fn arg(comptime n: u8) []const u8 {
    return &[_]u8{ 0x02, n };
}

const cases = [_]Case{
    .{ .name = "add", .module = module(2, arg(0) ++ arg(1) ++ ADD ++ RET), .args = &.{ "40", "2" }, .stdout = "42\n" },
    .{ .name = "sub-negative", .module = module(2, arg(0) ++ arg(1) ++ SUB ++ RET), .args = &.{ "5", "9" }, .stdout = "-4\n" },
    .{ .name = "poly", .module = module(1, arg(0) ++ DUP ++ MUL ++ imm(3) ++ arg(0) ++ MUL ++ SUB ++ imm(7) ++ ADD ++ RET), .args = &.{"10"}, .stdout = "77\n" },
    .{ .name = "imm64", .module = module(0, imm(0x123456789abcdef0) ++ RET), .args = &.{}, .stdout = "1311768467463790320\n" },
    .{ .name = "depth5-drop", .module = module(0, imm(1) ++ imm(2) ++ imm(3) ++ imm(4) ++ imm(100) ++ DROP ++ ADD ++ ADD ++ ADD ++ RET), .args = &.{}, .stdout = "10\n" },
    .{ .name = "four-args", .module = module(4, arg(0) ++ arg(1) ++ MUL ++ arg(2) ++ arg(3) ++ MUL ++ ADD ++ RET), .args = &.{ "3", "4", "5", "-6" }, .stdout = "-18\n" },
};

/// Registers the bc2jit steps; unit tests are also attached to `test_step`.
pub fn add(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, pyrlyn_jit: *std.Build.Module, test_step: *std.Build.Step) void {
    const mod = b.createModule(.{
        .root_source_file = b.path("bc2jit/main.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{.{ .name = "pyrlyn_jit", .module = pyrlyn_jit }},
    });
    const exe = b.addExecutable(.{ .name = "bc2jit", .root_module = mod });
    b.step("bc2jit", "Build the bc2jit CLI (KXBC bytecode -> pyrlyn-jit machine code)").dependOn(&b.addInstallArtifact(exe, .{}).step);

    const unit = b.addRunArtifact(b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("bc2jit/tests.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{.{ .name = "pyrlyn_jit", .module = pyrlyn_jit }},
    }) }));
    unit.skip_foreign_checks = true;
    b.step("test-bc2jit", "Run the bc2jit unit tests").dependOn(&unit.step);
    test_step.dependOn(&unit.step);

    const e2e = b.step("e2e-bc2jit", "Run bc2jit on KXBC fixtures with --arch = target arch (-frosetta / -fqemu for foreign targets)");
    const files = b.addWriteFiles();
    // Execution cases need a host the lowering's ABI runs on; Windows x64 is not SysV.
    const arch_flag: ?[]const u8 = switch (target.result.cpu.arch) {
        .x86_64 => if (target.result.os.tag == .windows) null else "x86-64",
        .aarch64 => "aarch64",
        .riscv64 => "riscv",
        else => null,
    };
    if (arch_flag) |flag| for (cases) |c| {
        const run = b.addRunArtifact(exe);
        run.setName(b.fmt("bc2jit e2e {s} --arch {s}", .{ c.name, flag }));
        run.addArgs(&.{ "--arch", flag });
        run.addFileArg(files.add(b.fmt("{s}.kxbc", .{c.name}), c.module));
        run.addArgs(c.args);
        run.expectExitCode(0);
        run.expectStdOutEqual(c.stdout);
        run.skip_foreign_checks = true;
        e2e.dependOn(&run.step);
    };
    // CLI contract: usage errors go to stderr with exit status 2.
    const usage_cases = [_]struct { args: []const []const u8, stderr: []const u8 }{
        .{ .args = &.{"x.kxbc"}, .stderr = "bc2jit: missing required option --arch (allowed: x86-64, aarch64, riscv)\n" },
        .{ .args = &.{ "--arch", "mips", "x.kxbc" }, .stderr = "bc2jit: unknown --arch value 'mips' (allowed: x86-64, aarch64, riscv)\n" },
        .{ .args = &.{"--arch"}, .stderr = "bc2jit: option --arch requires a value (allowed: x86-64, aarch64, riscv)\n" },
    };
    for (usage_cases, 0..) |u, k| {
        const run = b.addRunArtifact(exe);
        run.setName(b.fmt("bc2jit e2e usage error {d}", .{k}));
        run.addArgs(u.args);
        run.expectExitCode(2);
        run.expectStdErrEqual(u.stderr);
        run.skip_foreign_checks = true;
        e2e.dependOn(&run.step);
    }
    test_step.dependOn(e2e);
}
