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
/// KXBC v1 module (control flow, locals, IR lowering).
fn moduleV1(comptime argc: u8, comptime code: []const u8) []const u8 {
    return "KXBC" ++ [_]u8{ 1, argc, @truncate(code.len), @truncate(code.len >> 8) } ++ code;
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
    .{ .name = "wrap-add", .module = module(1, arg(0) ++ imm(1) ++ ADD ++ RET), .args = &.{"9223372036854775807"}, .stdout = "-9223372036854775808\n" },
    .{ .name = "wrap-mul", .module = module(0, imm(std.math.minInt(i64)) ++ imm(-1) ++ MUL ++ RET), .args = &.{}, .stdout = "-9223372036854775808\n" },
    .{ .name = "imm-min", .module = module(0, imm(std.math.minInt(i64)) ++ RET), .args = &.{}, .stdout = "-9223372036854775808\n" },
    .{ .name = "imm-2048", .module = module(0, imm(2048) ++ imm(-2049) ++ SUB ++ RET), .args = &.{}, .stdout = "4097\n" },
    .{ .name = "dup-square", .module = module(1, arg(0) ++ DUP ++ MUL ++ RET), .args = &.{"-0x10"}, .stdout = "256\n" },
    .{ .name = "extra-args-ignored", .module = module(0, imm(5) ++ RET), .args = &.{ "1", "2", "3", "4" }, .stdout = "5\n" },
    .{ .name = "ret-top-of-deep-stack", .module = module(2, arg(0) ++ arg(1) ++ RET), .args = &.{ "1", "2" }, .stdout = "2\n" },
    // v1: gcd(a, b) = while (b != 0) (a, b) = (b, a % b)
    //   0 push_arg 0 | 2 push_arg 1 | 4 dup | 5 push 0 | 14 br_if eq 28 | 18 dup | 19 store_local 0
    //   21 rem | 22 load_local 0 | 24 swap | 25 jmp 4 | 28 drop | 29 ret
    .{ .name = "v1-gcd", .module = moduleV1(2, arg(0) ++ arg(1) ++ DUP ++ imm(0) ++ "\x41\x00\x1c\x00" ++ DUP ++ "\x31\x00" ++ "\x14" ++ "\x30\x00" ++ "\x22" ++ "\x40\x04\x00" ++ DROP ++ RET), .args = &.{ "1071", "462" }, .stdout = "21\n" },
    .{ .name = "v1-div-by-zero-is-zero", .module = moduleV1(2, arg(0) ++ arg(1) ++ "\x13" ++ RET), .args = &.{ "7", "0" }, .stdout = "0\n" },
    .{ .name = "v1-min-div-minus-one", .module = moduleV1(2, arg(0) ++ arg(1) ++ "\x13" ++ RET), .args = &.{ "-9223372036854775808", "-1" }, .stdout = "-9223372036854775808\n" },
    .{ .name = "four-args", .module = module(4, arg(0) ++ arg(1) ++ MUL ++ arg(2) ++ arg(3) ++ MUL ++ ADD ++ RET), .args = &.{ "3", "4", "5", "-6" }, .stdout = "-18\n" },
};

/// Registers the bc2jit steps; unit tests are also attached to `test_step`.
pub fn add(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, pyrlyn_jit: *std.Build.Module, bc2jit: *std.Build.Module, test_step: *std.Build.Step, llvm_enabled: bool) void {
    const mod = b.createModule(.{
        .root_source_file = b.path("bc2jit/main.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{ .{ .name = "pyrlyn_jit", .module = pyrlyn_jit }, .{ .name = "bc2jit", .module = bc2jit } },
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
    const llvm_arch: ?[]const u8 = if (target.result.cpu.arch == .x86_64) "x86-64" else arch_flag;
    if (llvm_arch) |flag| {
        if (llvm_enabled) {
            for (cases) |c| for ([_][]const u8{ "llvm-jit", "llvm-interpreter" }) |backend| for ([_]bool{ false, true }) |optimized| {
                const run = b.addRunArtifact(exe);
                run.setName(b.fmt("bc2jit {s} {s} optimize={}", .{ backend, c.name, optimized }));
                run.addArgs(&.{ "--arch", flag, "--backend", backend });
                if (!optimized) run.addArg("--no-opt");
                run.addFileArg(files.add(b.fmt("llvm-{s}.kxbc", .{c.name}), c.module));
                run.addArgs(c.args);
                run.expectExitCode(0);
                run.expectStdOutEqual(c.stdout);
                run.skip_foreign_checks = true;
                e2e.dependOn(&run.step);
            };
        } else {
            const run = b.addRunArtifact(exe);
            run.setName("bc2jit LLVM backend disabled");
            run.addArgs(&.{ "--arch", flag, "--backend", "llvm-jit" });
            run.addFileArg(files.add("llvm-disabled.kxbc", cases[0].module));
            run.expectExitCode(1);
            run.addCheck(.{ .expect_stderr_match = "-Dllvm-prefix=PREFIX" });
            run.skip_foreign_checks = true;
            e2e.dependOn(&run.step);
        }
    }
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
    // Cross-target output: `-o FILE` writes the raw machine code for every --arch (golden bytes,
    // see tests.zig `op_cases`), independent of the host.
    const out_module = module(2, arg(0) ++ arg(1) ++ ADD ++ RET);
    const out_cases = [_]struct { flag: []const u8, bytes: []const u8 }{
        .{ .flag = "x86-64", .bytes = "\x48\x89\xf8\x49\x89\xf2\x4c\x01\xd0\xc3" },
        .{ .flag = "aarch64", .bytes = "\xe9\x03\x00\xaa\xea\x03\x01\xaa\x29\x01\x0a\x8b\xe0\x03\x09\xaa\xc0\x03\x5f\xd6" },
        .{ .flag = "riscv", .bytes = "\x93\x02\x05\x00\x13\x83\x05\x00\xb3\x82\x62\x00\x13\x85\x02\x00\x67\x80\x00\x00" },
    };
    const in_file = files.add("out-add.kxbc", out_module);
    for (out_cases) |oc| {
        const run = b.addRunArtifact(exe);
        run.setName(b.fmt("bc2jit e2e -o add --arch {s}", .{oc.flag}));
        run.addArgs(&.{ "--arch", oc.flag, "-o" });
        const out = run.addOutputFileArg2(b.fmt("add-{s}.bin", .{oc.flag}), .{});
        run.addFileArg(in_file);
        run.addArgs(&.{ "1", "2" });
        run.expectExitCode(0);
        run.skip_foreign_checks = true;
        const check = b.addCheckFile(out, .{ .expected_exact = oc.bytes });
        e2e.dependOn(&check.step);
    }
    // Explicitly suppress JIT even for the host, and check the same golden code bytes.
    for (out_cases) |oc| {
        const run = b.addRunArtifact(exe);
        run.addArgs(&.{ "--arch", oc.flag, "--no-jit" });
        run.addFileArg(in_file);
        run.expectExitCode(0);
        run.expectStdOutEqual(b.fmt("{x}\n", .{oc.bytes}));
        run.skip_foreign_checks = true;
        e2e.dependOn(&run.step);
    }
    // Input errors: exit status 1 with the decoder's message and byte offset.
    const bad_cases = [_]struct { name: []const u8, module: []const u8, stderr: []const u8 }{
        .{ .name = "empty", .module = "", .stderr = "module truncated" },
        .{ .name = "bad-magic", .module = "KXBX\x00\x00\x01\x00\xF0", .stderr = "offset 0: not a KXBC module (bad magic)" },
        .{ .name = "unknown-op", .module = module(0, "\x99"), .stderr = "offset 8: unknown opcode" },
        .{ .name = "truncated-imm", .module = module(0, "\x01\x00\x00"), .stderr = "offset 8: module truncated" },
        .{ .name = "underflow", .module = module(0, ADD ++ RET), .stderr = "offset 8: operand stack underflow" },
        .{ .name = "too-deep", .module = module(1, arg(0) ++ DUP ++ DUP ++ DUP ++ DUP ++ DUP ++ RET), .stderr = "offset 14: operand stack deeper than 5" },
        .{ .name = "no-ret", .module = module(1, arg(0)), .stderr = "offset 10: function does not end with ret" },
        .{ .name = "len-overflow", .module = "KXBC\x00\x00\xff\xff", .stderr = "offset 6: module truncated" },
    };
    for (bad_cases) |bcase| for ([_][]const u8{ "x86-64", "aarch64", "riscv" }) |flag| {
        const run = b.addRunArtifact(exe);
        run.setName(b.fmt("bc2jit e2e bad input {s} --arch {s}", .{ bcase.name, flag }));
        run.addArgs(&.{ "--arch", flag });
        run.addFileArg(files.add(b.fmt("bad-{s}.kxbc", .{bcase.name}), bcase.module));
        run.expectExitCode(1);
        run.addCheck(.{ .expect_stderr_match = bcase.stderr });
        run.skip_foreign_checks = true;
        e2e.dependOn(&run.step);
    };
    {
        const run = b.addRunArtifact(exe);
        run.setName("bc2jit e2e missing input file");
        run.addArgs(&.{ "--arch", "riscv", "does-not-exist.kxbc" });
        run.expectExitCode(1);
        run.addCheck(.{ .expect_stderr_match = "cannot read 'does-not-exist.kxbc'" });
        e2e.dependOn(&run.step);
    }
    // Foreign --arch without -o prints the code as hex on stdout.
    const hex_cases = [_]struct { flag: []const u8, hex: []const u8 }{
        .{ .flag = "x86-64", .hex = "4889f84989f24c01d0c3\n" },
        .{ .flag = "aarch64", .hex = "e90300aaea0301aa29010a8be00309aac0035fd6\n" },
        .{ .flag = "riscv", .hex = "9302050013830500b38262001385020067800000\n" },
    };
    for (hex_cases) |hc| {
        if (arch_flag != null and std.mem.eql(u8, arch_flag.?, hc.flag)) continue; // host executes instead
        const run = b.addRunArtifact(exe);
        run.setName(b.fmt("bc2jit e2e hex stdout --arch {s}", .{hc.flag}));
        run.addArgs(&.{ "--arch", hc.flag });
        run.addFileArg(in_file);
        run.expectExitCode(0);
        run.expectStdOutEqual(hc.hex);
        run.skip_foreign_checks = true;
        e2e.dependOn(&run.step);
    }
    test_step.dependOn(e2e);
}
