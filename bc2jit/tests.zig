//! bc2jit unit tests: CLI parsing, decoder, per-opcode lowering bytes (cross-checked with
//! `llvm-mc --disassemble`), the stack register model, and host-only execution.
const std = @import("std");
const jit = @import("pyrlyn_jit");
const bc = @import("bytecode.zig");
const cli = @import("cli.zig");
const lower = @import("lower.zig");
const driver = @import("driver.zig");
const Target = lower.Target;
const t = std.testing;

fn parse(arena: *std.heap.ArenaAllocator, argv: []const []const u8) !cli.Parsed {
    return cli.parse(arena.allocator(), argv);
}

// ---- --arch parsing --------------------------------------------------------------------

test "cli: each --arch value, short/long/= forms" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const forms = [_][3][]const u8{
        .{ "--arch", "x86-64", "x86_64" }, .{ "-a", "aarch64", "aarch64" }, .{ "--arch", "riscv", "riscv64" },
    };
    for (forms) |f| {
        const want = std.meta.stringToEnum(Target, f[2]).?;
        const p = try parse(&arena, &.{ f[0], f[1], "in.kxbc" });
        try t.expectEqual(want, p.run.target);
        try t.expectEqualStrings("in.kxbc", p.run.input);
        const eq = try std.fmt.allocPrint(arena.allocator(), "--arch={s}", .{f[1]});
        try t.expectEqual(want, (try parse(&arena, &.{ "in.kxbc", eq })).run.target);
    }
}

test "cli: missing --arch" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const p = try parse(&arena, &.{"in.kxbc"});
    try t.expectEqualStrings("bc2jit: missing required option --arch (allowed: x86-64, aarch64, riscv)", p.usage_error);
    try t.expectEqual(2, cli.exit_usage);
}

test "cli: unknown and empty --arch values" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    try t.expectEqualStrings(
        "bc2jit: unknown --arch value 'x86_64' (allowed: x86-64, aarch64, riscv)",
        (try parse(&arena, &.{ "--arch", "x86_64", "in.kxbc" })).usage_error,
    );
    try t.expectEqualStrings(
        "bc2jit: unknown --arch value 'RISCV' (allowed: x86-64, aarch64, riscv)",
        (try parse(&arena, &.{ "-a", "RISCV", "in.kxbc" })).usage_error,
    );
    const no_value = "bc2jit: option --arch requires a value (allowed: x86-64, aarch64, riscv)";
    try t.expectEqualStrings(no_value, (try parse(&arena, &.{"--arch"})).usage_error);
    try t.expectEqualStrings(no_value, (try parse(&arena, &.{ "in.kxbc", "--arch=" })).usage_error);
}

test "cli: help, output, ARGs, other errors" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    try t.expect((try parse(&arena, &.{"-h"})) == .help);
    try t.expect((try parse(&arena, &.{ "--arch", "riscv", "--help" })) == .help);
    const p = (try parse(&arena, &.{ "-a", "aarch64", "-o", "out.bin", "f.kxbc", "7", "-3", "0x10" })).run;
    try t.expectEqualStrings("out.bin", p.output.?);
    try t.expectEqual(3, p.arg_count);
    try t.expectEqualSlices(i64, &.{ 7, -3, 16, 0 }, &p.args);
    try t.expectEqualStrings("bc2jit: missing INPUT file (see --help)", (try parse(&arena, &.{ "-a", "riscv" })).usage_error);
    try t.expectEqualStrings("bc2jit: unknown option '--bogus' (see --help)", (try parse(&arena, &.{ "-a", "riscv", "--bogus" })).usage_error);
    try t.expectEqualStrings("bc2jit: ARG 'x' is not a 64-bit integer", (try parse(&arena, &.{ "-a", "riscv", "f", "x" })).usage_error);
    try t.expectEqualStrings("bc2jit: at most 4 ARGs are allowed", (try parse(&arena, &.{ "-a", "riscv", "f", "1", "2", "3", "4", "5" })).usage_error);
}

// ---- decoder ---------------------------------------------------------------------------

const all_ops = [_]bc.Inst{
    .{ .push_arg = 0 }, .{ .push_i64 = std.math.minInt(i64) }, .add, .{ .push_i64 = -1 }, .sub,
    .{ .push_arg = 1 }, .mul,                                  .dup, .drop,               .ret,
};

test "decoder: round-trip of every opcode" {
    const f: bc.Function = .{ .arg_count = 2, .insts = &all_ops };
    const bytes = try bc.encode(t.allocator, f);
    defer t.allocator.free(bytes);
    try t.expectEqualSlices(u8, "KXBC\x00\x02", bytes[0..6]);
    try t.expectEqual(@as(u16, @intCast(bytes.len - bc.header_len)), std.mem.readInt(u16, bytes[6..8], .little));
    var diag: bc.Diagnostic = .{};
    const g = try bc.decode(t.allocator, bytes, &diag);
    defer g.deinit(t.allocator);
    try t.expectEqual(f.arg_count, g.arg_count);
    try t.expectEqualSlices(bc.Inst, f.insts, g.insts);
    const again = try bc.encode(t.allocator, g);
    defer t.allocator.free(again);
    try t.expectEqualSlices(u8, bytes, again);
}

fn expectDecodeError(want: bc.Error, offset: usize, bytes: []const u8) !void {
    var diag: bc.Diagnostic = .{};
    try t.expectError(want, bc.decode(t.allocator, bytes, &diag));
    try t.expectEqual(offset, diag.offset);
}

test "decoder: rejects malformed modules with offsets" {
    try expectDecodeError(error.Truncated, 0, "KXBC");
    try expectDecodeError(error.BadMagic, 0, "KXBX\x00\x00\x01\x00\xF0");
    try expectDecodeError(error.UnsupportedVersion, 4, "KXBC\x02\x00\x01\x00\xF0");
    try expectDecodeError(error.TooManyArgs, 5, "KXBC\x00\x05\x01\x00\xF0");
    try expectDecodeError(error.Truncated, 6, "KXBC\x00\x00\x02\x00\xF0");
    try expectDecodeError(error.TrailingBytes, 6, "KXBC\x00\x00\x01\x00\xF0\xF0");
    try expectDecodeError(error.UnknownOpcode, 8, "KXBC\x00\x00\x01\x00\x99");
    try expectDecodeError(error.ArgOutOfRange, 8, "KXBC\x00\x01\x03\x00\x02\x01\xF0");
    try expectDecodeError(error.StackUnderflow, 8, "KXBC\x00\x00\x01\x00\xF0");
    try expectDecodeError(error.Truncated, 8, "KXBC\x00\x00\x02\x00\x01\x00");
    try expectDecodeError(error.MissingRet, 10, "KXBC\x00\x01\x02\x00\x02\x00");
    try expectDecodeError(error.CodeAfterRet, 11, "KXBC\x00\x01\x04\x00\x02\x00\xF0\x20");
    try expectDecodeError(error.StackTooDeep, 14, "KXBC\x00\x01\x08\x00\x02\x00\x20\x20\x20\x20\x20\xF0");
}

// ---- lowering: one program per core opcode, exact bytes per arch ------------------------

fn lowerBytes(target: Target, argc: u8, insts: []const bc.Inst) ![]u8 {
    var cb = jit.CodeBuffer.init(t.allocator);
    defer cb.deinit();
    try lower.lower(target, .{ .arg_count = argc, .insts = insts }, &cb);
    return t.allocator.dupe(u8, cb.code());
}

fn hex(comptime s: []const u8) [s.len / 2]u8 {
    var out: [s.len / 2]u8 = undefined;
    _ = std.fmt.hexToBytes(&out, s) catch unreachable;
    return out;
}

const OpCase = struct { name: []const u8, argc: u8, insts: []const bc.Inst, x86: []const u8, a64: []const u8, rv: []const u8 };

// Disassembly (llvm-mc) is given per case; every program ends in `mov ret, slot0; ret`.
const op_cases = [_]OpCase{
    // movq $-2,%rax | movz/movk x9 (4) | li t0,-2
    .{ .name = "push_i64", .argc = 0, .insts = &.{ .{ .push_i64 = -2 }, .ret }, .x86 = &hex("48c7c0feffffffc3"), .a64 = &hex("c9ff9fd2e9ffbff2e9ffdff2e9fffff2e00309aac0035fd6"), .rv = &hex("9302e0ff1385020067800000") },
    // movq %rsi,%rax | mov x9,x1 | mv t0,a1
    .{ .name = "push_arg", .argc = 2, .insts = &.{ .{ .push_arg = 1 }, .ret }, .x86 = &hex("4889f0c3"), .a64 = &hex("e90301aae00309aac0035fd6"), .rv = &hex("938205001385020067800000") },
    // addq %r10,%rax | add x9,x9,x10 | add t0,t0,t1
    .{ .name = "add", .argc = 2, .insts = &.{ .{ .push_arg = 0 }, .{ .push_arg = 1 }, .add, .ret }, .x86 = &hex("4889f84989f24c01d0c3"), .a64 = &hex("e90300aaea0301aa29010a8be00309aac0035fd6"), .rv = &hex("9302050013830500b38262001385020067800000") },
    // subq %r10,%rax | sub x9,x9,x10 | sub t0,t0,t1
    .{ .name = "sub", .argc = 2, .insts = &.{ .{ .push_arg = 0 }, .{ .push_arg = 1 }, .sub, .ret }, .x86 = &hex("4889f84989f24c29d0c3"), .a64 = &hex("e90300aaea0301aa29010acbe00309aac0035fd6"), .rv = &hex("9302050013830500b38262401385020067800000") },
    // imulq %r10,%rax | mul x9,x9,x10 | mul t0,t0,t1
    .{ .name = "mul", .argc = 2, .insts = &.{ .{ .push_arg = 0 }, .{ .push_arg = 1 }, .mul, .ret }, .x86 = &hex("4889f84989f2490fafc2c3"), .a64 = &hex("e90300aaea0301aa297d0a9be00309aac0035fd6"), .rv = &hex("9302050013830500b38262021385020067800000") },
    // movq %rax,%r10 | mov x10,x9 | mv t1,t0
    .{ .name = "dup", .argc = 1, .insts = &.{ .{ .push_arg = 0 }, .dup, .add, .ret }, .x86 = &hex("4889f84989c24c01d0c3"), .a64 = &hex("e90300aaea0309aa29010a8be00309aac0035fd6"), .rv = &hex("9302050013830200b38262001385020067800000") },
    // drop emits nothing; ret returns slot 0
    .{ .name = "drop", .argc = 2, .insts = &.{ .{ .push_arg = 0 }, .{ .push_arg = 1 }, .drop, .ret }, .x86 = &hex("4889f84989f2c3"), .a64 = &hex("e90300aaea0301aae00309aac0035fd6"), .rv = &hex("93020500138305001385020067800000") },
};

fn expectLowering(target: Target, c: OpCase, want: []const u8) !void {
    const got = try lowerBytes(target, c.argc, c.insts);
    defer t.allocator.free(got);
    t.expectEqualSlices(u8, want, got) catch |e| {
        std.debug.print("opcode {s}, --arch {s}\n", .{ c.name, target.flagName() });
        return e;
    };
}

test "lowering: core opcodes on x86-64" {
    if (!Target.x86_64.isBuilt()) return error.SkipZigTest;
    for (op_cases) |c| try expectLowering(.x86_64, c, c.x86);
}

test "lowering: core opcodes on aarch64" {
    if (!Target.aarch64.isBuilt()) return error.SkipZigTest;
    for (op_cases) |c| try expectLowering(.aarch64, c, c.a64);
}

test "lowering: core opcodes on riscv" {
    for (op_cases) |c| try expectLowering(.riscv64, c, c.rv);
}

test "lowering: backend left out of the build is an error" {
    var cb = jit.CodeBuffer.init(t.allocator);
    defer cb.deinit();
    const f: bc.Function = .{ .arg_count = 0, .insts = &.{ .{ .push_i64 = 1 }, .ret } };
    for (std.meta.tags(Target)) |target| {
        if (target.isBuilt()) continue;
        try t.expectError(error.BackendNotBuilt, lower.lower(target, f, &cb));
    }
}

// ---- stack register model --------------------------------------------------------------

test "register model: stack slot i lives in slots[i] (riscv, depth 3)" {
    // a*a - 3*a: slot depths 1,2,1,2,3,2,1
    const prog = [_]bc.Inst{ .{ .push_arg = 0 }, .dup, .mul, .{ .push_i64 = 3 }, .{ .push_arg = 0 }, .mul, .sub, .ret };
    const got = try lowerBytes(.riscv64, 1, &prog);
    defer t.allocator.free(got);

    var cb = jit.CodeBuffer.init(t.allocator);
    defer cb.deinit();
    const a = jit.riscv.Assembler.init(&cb);
    const s = lower.Rv.slots;
    try t.expectEqual(jit.riscv.X.t0, s[0]);
    try a.mv(s[0], .a0);
    try a.mv(s[1], s[0]);
    try a.mul(s[0], s[0], s[1]);
    try a.li(s[1], 3);
    try a.mv(s[2], .a0);
    try a.mul(s[1], s[1], s[2]);
    try a.sub(s[0], s[0], s[1]);
    try a.mv(.a0, s[0]);
    try a.ret();
    try cb.finalize();
    try t.expectEqualSlices(u8, cb.code(), got);
}

test "register model: slot registers are distinct and never argument registers" {
    inline for (.{ lower.X86, lower.A64, lower.Rv }) |M| {
        if (M == void) continue;
        for (M.slots, 0..) |r, i| {
            for (M.slots[i + 1 ..]) |q| try t.expect(r != q);
            for (M.args) |q| try t.expect(r != q);
        }
    }
}

// ---- execution only when --arch is the host --------------------------------------------

test "driver: executable mapping only for the host arch, bytes for the others" {
    const f: bc.Function = .{ .arg_count = 2, .insts = &.{ .{ .push_arg = 0 }, .{ .push_arg = 1 }, .mul, .{ .push_i64 = 2 }, .add, .ret } };
    for (std.meta.tags(Target)) |target| {
        if (!target.isBuilt()) continue;
        var c = try driver.compile(t.allocator, target, f);
        defer c.deinit();
        try t.expect(c.code().len > 0);
        try t.expectEqual(target.canExecute(), c.exec != null);
        if (c.exec != null) try t.expectEqual(@as(i64, 42), c.call(.{ 8, 5, 0, 0 }));
    }
}

test "driver: canExecute is exactly target == host" {
    for (std.meta.tags(Target)) |target| {
        if (target.canExecute()) try t.expectEqual(Target.host(), target);
    }
}

test {
    _ = @import("tests_extra.zig");
    _ = @import("tests_v1.zig");
    _ = @import("tests_llvm.zig");
}
