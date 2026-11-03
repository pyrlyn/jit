//! Extended bc2jit test suite: decoder edge/negative cases, exhaustive opcode bytes, every
//! opcode at every stack depth on all three `--arch` targets, boundary immediates (golden bytes
//! cross-checked with `llvm-mc --disassemble`), differential checks against a reference
//! interpreter (host execution, plus a small RV64 simulator for RISC-V output), seeded
//! property/fuzz checks, and stress programs at the 64 KiB code-length limit.
const std = @import("std");
const jit = @import("pyrlyn_jit");
const bc = @import("bytecode.zig");
const lower = @import("lower.zig");
const driver = @import("driver.zig");
const Target = lower.Target;
const t = std.testing;

fn hex(comptime s: []const u8) [s.len / 2]u8 {
    @setEvalBranchQuota(100_000);
    var out: [s.len / 2]u8 = undefined;
    _ = std.fmt.hexToBytes(&out, s) catch unreachable;
    return out;
}

// ---- helpers ---------------------------------------------------------------------------

/// Reference semantics of a validated function (wrapping i64 arithmetic).
fn interpret(f: bc.Function, args: [4]i64) i64 {
    var st: [bc.max_depth]i64 = undefined;
    var d: usize = 0;
    for (f.insts) |i| switch (i) {
        .push_i64 => |v| {
            st[d] = v;
            d += 1;
        },
        .push_arg => |n| {
            st[d] = args[n];
            d += 1;
        },
        .add => {
            st[d - 2] +%= st[d - 1];
            d -= 1;
        },
        .sub => {
            st[d - 2] -%= st[d - 1];
            d -= 1;
        },
        .mul => {
            st[d - 2] *%= st[d - 1];
            d -= 1;
        },
        .dup => {
            st[d] = st[d - 1];
            d += 1;
        },
        .drop => d -= 1,
        .ret => return st[d - 1],
    };
    unreachable;
}

/// Minimal RV64IM simulator for the instruction subset the RISC-V lowering emits
/// (lui, addi, addiw, slli, add, sub, mul, jalr). Returns a0 at `ret`.
fn simRv64(code: []const u8, args: [4]i64) !i64 {
    var x: [32]u64 = @splat(0);
    for (args, 0..) |a, k| x[10 + k] = @bitCast(a);
    x[1] = 0xdead_0000; // ra sentinel
    var pc: usize = 0;
    var steps: usize = 0;
    while (true) : (steps += 1) {
        if (steps > 1_000_000 or pc + 4 > code.len) return error.SimOutOfBounds;
        const w = std.mem.readInt(u32, code[pc..][0..4], .little);
        const rd: u5 = @truncate(w >> 7);
        const rs1: u5 = @truncate(w >> 15);
        const rs2: u5 = @truncate(w >> 20);
        const f3: u3 = @truncate(w >> 12);
        const f7: u7 = @truncate(w >> 25);
        const imm_i: i64 = @as(i32, @bitCast(w)) >> 20;
        var v: u64 = 0;
        switch (w & 0x7f) {
            0x37 => v = @bitCast(@as(i64, @as(i32, @bitCast(w & 0xfffff000)))),
            0x13 => switch (f3) {
                0 => v = x[rs1] +% @as(u64, @bitCast(imm_i)),
                1 => v = x[rs1] << @as(u6, @truncate(w >> 20)),
                else => return error.SimUnsupported,
            },
            0x1b => {
                if (f3 != 0) return error.SimUnsupported;
                const r: u32 = @truncate(x[rs1] +% @as(u64, @bitCast(imm_i)));
                v = @bitCast(@as(i64, @as(i32, @bitCast(r))));
            },
            0x33 => {
                if (f3 != 0) return error.SimUnsupported;
                v = switch (f7) {
                    0x00 => x[rs1] +% x[rs2],
                    0x20 => x[rs1] -% x[rs2],
                    0x01 => x[rs1] *% x[rs2],
                    else => return error.SimUnsupported,
                };
            },
            0x67 => {
                if (rd == 0 and rs1 == 1 and imm_i == 0 and x[1] == 0xdead_0000) return @bitCast(x[10]);
                return error.SimUnsupported;
            },
            else => return error.SimUnsupported,
        }
        if (rd != 0) x[rd] = v;
        pc += 4;
    }
}

fn lowerBytes(target: Target, f: bc.Function) ![]u8 {
    var cb = jit.CodeBuffer.init(t.allocator);
    defer cb.deinit();
    try lower.lower(target, f, &cb);
    return t.allocator.dupe(u8, cb.code());
}

/// Encoding of the final `ret` instruction per target.
fn retBytes(target: Target) []const u8 {
    return switch (target) {
        .x86_64 => "\xc3",
        .aarch64 => "\xc0\x03\x5f\xd6",
        .riscv64 => "\x67\x80\x00\x00",
    };
}

/// Lowers `f` for every built target and checks the result against `interpret`: by executing
/// it when the target is the host, by simulation for RISC-V, and by shape otherwise.
/// Returns how many (target, check) pairs actually ran a value comparison.
fn checkAllTargets(f: bc.Function, args: [4]i64) !usize {
    const want = interpret(f, args);
    var compared: usize = 0;
    for (std.meta.tags(Target)) |target| {
        if (!target.isBuilt()) continue;
        var c = try driver.compile(t.allocator, target, f);
        defer c.deinit();
        const code = c.code();
        try t.expect(std.mem.endsWith(u8, code, retBytes(target)));
        if (c.exec != null) {
            try t.expectEqual(want, c.call(args));
            compared += 1;
        }
        if (target == .riscv64) {
            try t.expectEqual(want, try simRv64(code, args));
            compared += 1;
        }
    }
    return compared;
}

/// Random valid program: depth stays in 0..max_depth, ends with ret.
fn randomProgram(r: std.Random, buf: []bc.Inst, argc: u8) []bc.Inst {
    var n: usize = 0;
    var d: u8 = 0;
    while (n + 1 < buf.len) {
        if (d >= 1 and r.uintLessThan(u8, 8) == 0) break;
        var choices: [8]bc.Opcode = undefined;
        var k: usize = 0;
        if (d < bc.max_depth) {
            choices[k] = .push_i64;
            k += 1;
            if (argc > 0) {
                choices[k] = .push_arg;
                k += 1;
            }
        }
        if (d >= 1 and d < bc.max_depth) {
            choices[k] = .dup;
            k += 1;
        }
        if (d >= 2) {
            for ([_]bc.Opcode{ .add, .sub, .mul, .drop }) |o| {
                choices[k] = o;
                k += 1;
            }
        }
        const inst: bc.Inst = switch (choices[r.uintLessThan(usize, k)]) {
            .push_i64 => .{ .push_i64 = switch (r.uintLessThan(u8, 4)) {
                0 => r.int(i64),
                1 => r.intRangeAtMost(i64, -3000, 3000),
                2 => @bitCast(@as(u64, 1) << r.int(u6)),
                else => 0 -% @as(i64, @bitCast(@as(u64, 1) << r.int(u6))),
            } },
            .push_arg => .{ .push_arg = r.uintLessThan(u8, argc) },
            .add => .add,
            .sub => .sub,
            .mul => .mul,
            .dup => .dup,
            .drop => .drop,
            .ret => unreachable,
        };
        buf[n] = inst;
        n += 1;
        d = @intCast(@as(i16, d) + inst.stackEffect());
    }
    if (d == 0) {
        buf[n] = .{ .push_i64 = 7 };
        n += 1;
    }
    buf[n] = .ret;
    return buf[0 .. n + 1];
}

const ImmCase = struct { v: i64, x86: []const u8, a64: []const u8, rv: []const u8 };

// `push_i64 v; ret` on each target. Covers RISC-V li forms (12-bit, lui, lui+addiw, shifted
// sequences), AArch64 movz/movk chains, and x86-64 mov r32 / mov r64 sign-extended / movabs.
const imm_cases = [_]ImmCase{
    .{ .v = 0, .x86 = &hex("b800000000c3"), .a64 = &hex("090080d2e00309aac0035fd6"), .rv = &hex("930200001385020067800000") },
    .{ .v = 1, .x86 = &hex("b801000000c3"), .a64 = &hex("290080d2e00309aac0035fd6"), .rv = &hex("930210001385020067800000") },
    .{ .v = -1, .x86 = &hex("48c7c0ffffffffc3"), .a64 = &hex("e9ff9fd2e9ffbff2e9ffdff2e9fffff2e00309aac0035fd6"), .rv = &hex("9302f0ff1385020067800000") },
    .{ .v = 2047, .x86 = &hex("b8ff070000c3"), .a64 = &hex("e9ff80d2e00309aac0035fd6"), .rv = &hex("9302f07f1385020067800000") },
    .{ .v = 2048, .x86 = &hex("b800080000c3"), .a64 = &hex("090081d2e00309aac0035fd6"), .rv = &hex("b71200009b8202801385020067800000") },
    .{ .v = -2048, .x86 = &hex("48c7c000f8ffffc3"), .a64 = &hex("09009fd2e9ffbff2e9ffdff2e9fffff2e00309aac0035fd6"), .rv = &hex("930200801385020067800000") },
    .{ .v = -2049, .x86 = &hex("48c7c0fff7ffffc3"), .a64 = &hex("e9ff9ed2e9ffbff2e9ffdff2e9fffff2e00309aac0035fd6"), .rv = &hex("b7f2ffff9b82f27f1385020067800000") },
    .{ .v = 65535, .x86 = &hex("b8ffff0000c3"), .a64 = &hex("e9ff9fd2e00309aac0035fd6"), .rv = &hex("b70201009b82f2ff1385020067800000") },
    .{ .v = 65536, .x86 = &hex("b800000100c3"), .a64 = &hex("090080d22900a0f2e00309aac0035fd6"), .rv = &hex("b70201001385020067800000") },
    .{ .v = 2147483647, .x86 = &hex("b8ffffff7fc3"), .a64 = &hex("e9ff9fd2e9ffaff2e00309aac0035fd6"), .rv = &hex("b70200809b82f2ff1385020067800000") },
    .{ .v = -2147483648, .x86 = &hex("48c7c000000080c3"), .a64 = &hex("090080d20900b0f2e9ffdff2e9fffff2e00309aac0035fd6"), .rv = &hex("b70200801385020067800000") },
    .{ .v = 2147483648, .x86 = &hex("b800000080c3"), .a64 = &hex("090080d20900b0f2e00309aac0035fd6"), .rv = &hex("930210009392f2011385020067800000") },
    .{ .v = 4294967295, .x86 = &hex("b8ffffffffc3"), .a64 = &hex("e9ff9fd2e9ffbff2e00309aac0035fd6"), .rv = &hex("93021000939202029382f2ff1385020067800000") },
    .{ .v = 4294967296, .x86 = &hex("48b80000000001000000c3"), .a64 = &hex("090080d22900c0f2e00309aac0035fd6"), .rv = &hex("93021000939202021385020067800000") },
    .{ .v = 9223372036854775807, .x86 = &hex("48b8ffffffffffffff7fc3"), .a64 = &hex("e9ff9fd2e9ffbff2e9ffdff2e9ffeff2e00309aac0035fd6"), .rv = &hex("9302f0ff9392f2039382f2ff1385020067800000") },
    .{ .v = std.math.minInt(i64), .x86 = &hex("48b80000000000000080c3"), .a64 = &hex("090080d20900f0f2e00309aac0035fd6"), .rv = &hex("9302f0ff9392f2031385020067800000") },
    .{ .v = 1311768467463790320, .x86 = &hex("48b8f0debc9a78563412c3"), .a64 = &hex("09de9bd28957b3f209cfcaf28946e2f2e00309aac0035fd6"), .rv = &hex("b77224009b82d28a9392e2009382d2c49392c2009382725e9392d200938202ef1385020067800000") },
};

const all_targets = [_]Target{ .x86_64, .aarch64, .riscv64 };

// ---- decoder: per opcode, exhaustive bytes, edge and negative cases ----------------------

fn mod(comptime argc: u8, comptime code: []const u8) []const u8 {
    return "KXBC" ++ [_]u8{ 0, argc, @truncate(code.len), @truncate(code.len >> 8) } ++ code;
}

fn decodeOk(bytes: []const u8) !bc.Function {
    var diag: bc.Diagnostic = .{};
    return bc.decode(t.allocator, bytes, &diag);
}

fn expectErr(want: bc.Error, offset: usize, bytes: []const u8) !void {
    var diag: bc.Diagnostic = .{};
    try t.expectError(want, bc.decode(t.allocator, bytes, &diag));
    try t.expectEqual(offset, diag.offset);
}

test "decoder: each opcode decodes to its instruction" {
    const cases = .{
        .{ mod(0, "\x01\x2a\x00\x00\x00\x00\x00\x00\x00\xF0"), &[_]bc.Inst{ .{ .push_i64 = 42 }, .ret } },
        .{ mod(1, "\x02\x00\xF0"), &[_]bc.Inst{ .{ .push_arg = 0 }, .ret } },
        .{ mod(2, "\x02\x00\x02\x01\x10\xF0"), &[_]bc.Inst{ .{ .push_arg = 0 }, .{ .push_arg = 1 }, .add, .ret } },
        .{ mod(2, "\x02\x00\x02\x01\x11\xF0"), &[_]bc.Inst{ .{ .push_arg = 0 }, .{ .push_arg = 1 }, .sub, .ret } },
        .{ mod(2, "\x02\x00\x02\x01\x12\xF0"), &[_]bc.Inst{ .{ .push_arg = 0 }, .{ .push_arg = 1 }, .mul, .ret } },
        .{ mod(1, "\x02\x00\x20\xF0"), &[_]bc.Inst{ .{ .push_arg = 0 }, .dup, .ret } },
        .{ mod(1, "\x02\x00\x20\x21\xF0"), &[_]bc.Inst{ .{ .push_arg = 0 }, .dup, .drop, .ret } },
    };
    inline for (cases) |c| {
        const f = try decodeOk(c[0]);
        defer f.deinit(t.allocator);
        try t.expectEqualSlices(bc.Inst, c[1], f.insts);
    }
}

test "decoder: every byte that is not an opcode is UnknownOpcode at its offset" {
    var n: usize = 0;
    for (0..256) |b| {
        if (std.enums.fromInt(bc.Opcode, @as(u8, @intCast(b))) != null) continue;
        const bytes = [_]u8{ 'K', 'X', 'B', 'C', 0, 0, 1, 0, @intCast(b) };
        try expectErr(error.UnknownOpcode, 8, &bytes);
        // Also as the second instruction, after a valid push.
        const bytes2 = [_]u8{ 'K', 'X', 'B', 'C', 0, 1, 3, 0, 0x02, 0x00, @intCast(b) };
        try expectErr(error.UnknownOpcode, 10, &bytes2);
        n += 1;
    }
    try t.expectEqual(256 - std.meta.tags(bc.Opcode).len, n);
}

test "decoder: empty input and short headers are Truncated at offset 0" {
    for (0..bc.header_len) |len| try expectErr(error.Truncated, 0, "KXBC\x00\x00\x00\x00"[0..len]);
}

test "decoder: header-only module (zero code) is MissingRet" {
    try expectErr(error.MissingRet, 8, "KXBC\x00\x00\x00\x00");
    try expectErr(error.MissingRet, 8, "KXBC\x00\x04\x00\x00");
}

test "decoder: every bad magic byte, version and arg count" {
    for (0..4) |k| {
        var bytes = "KXBC\x00\x00\x01\x00\xF0".*;
        bytes[k] ^= 0x20; // e.g. lowercase
        try expectErr(error.BadMagic, 0, &bytes);
    }
    for (1..256) |v| {
        var bytes = "KXBC\x00\x00\x01\x00\xF0".*;
        bytes[4] = @intCast(v);
        try expectErr(error.UnsupportedVersion, 4, &bytes);
    }
    for (bc.max_args + 1..256) |a| {
        var bytes = "KXBC\x00\x00\x01\x00\xF0".*;
        bytes[5] = @intCast(a);
        try expectErr(error.TooManyArgs, 5, &bytes);
    }
}

test "decoder: push_arg index boundary for every arg count" {
    for (0..bc.max_args + 1) |argc| {
        for (0..256) |n| {
            const bytes = [_]u8{ 'K', 'X', 'B', 'C', 0, @intCast(argc), 3, 0, 0x02, @intCast(n), 0xF0 };
            if (n < argc) {
                const f = try decodeOk(&bytes);
                defer f.deinit(t.allocator);
                try t.expectEqual(@as(u8, @intCast(n)), f.insts[0].push_arg);
            } else try expectErr(error.ArgOutOfRange, 8, &bytes);
        }
    }
}

test "decoder: push_i64 boundary values round-trip" {
    const vals = [_]i64{ 0, 1, -1, 127, 128, -128, -129, 0x7fff_ffff, -0x8000_0000, 0x1_0000_0000, std.math.maxInt(i64), std.math.minInt(i64) };
    for (vals) |v| {
        const f0: bc.Function = .{ .arg_count = 0, .insts = &.{ .{ .push_i64 = v }, .ret } };
        const bytes = try bc.encode(t.allocator, f0);
        defer t.allocator.free(bytes);
        try t.expectEqual(v, std.mem.readInt(i64, bytes[9..17], .little));
        const f = try decodeOk(bytes);
        defer f.deinit(t.allocator);
        try t.expectEqual(v, f.insts[0].push_i64);
    }
}

test "decoder: truncation at every operand byte" {
    // push_i64 with 0..7 operand bytes, code_len matching what is present.
    for (0..8) |k| {
        var buf: [17]u8 = undefined;
        @memcpy(buf[0..8], "KXBC\x00\x00\x00\x00");
        buf[6] = @intCast(1 + k);
        buf[8] = 0x01;
        @memset(buf[9 .. 9 + k], 0xAA);
        try expectErr(error.Truncated, 8, buf[0 .. 9 + k]);
    }
    try expectErr(error.Truncated, 8, "KXBC\x00\x01\x01\x00\x02");
}

test "decoder: every proper prefix of a valid module is rejected, never accepted" {
    const bytes = mod(2, "\x02\x00\x01\x05\x00\x00\x00\x00\x00\x00\x00\x12\x02\x01\x20\x10\x21\x01\xff\xff\xff\xff\xff\xff\xff\xff\x10\xF0");
    _ = interpret; // keep helpers referenced in all configurations
    (try decodeOk(bytes)).deinit(t.allocator);
    for (0..bytes.len) |len| {
        var diag: bc.Diagnostic = .{};
        if (bc.decode(t.allocator, bytes[0..len], &diag)) |f| {
            f.deinit(t.allocator);
            return error.PrefixAccepted;
        } else |_| {}
        // Prefix with a corrected code length: must fail as truncated operand, missing ret or underflow.
        if (len > bc.header_len) {
            var copy = try t.allocator.dupe(u8, bytes[0..len]);
            defer t.allocator.free(copy);
            std.mem.writeInt(u16, copy[6..8], @intCast(len - bc.header_len), .little);
            if (bc.decode(t.allocator, copy, &diag)) |f| {
                f.deinit(t.allocator);
                return error.PrefixAccepted;
            } else |e| try t.expect(e == error.Truncated or e == error.MissingRet);
        }
    }
}

test "decoder: stack depth boundaries" {
    // Exactly max_depth is fine, one more is StackTooDeep.
    (try decodeOk(mod(1, "\x02\x00\x20\x20\x20\x20\x10\x10\x10\x10\xF0"))).deinit(t.allocator);
    try expectErr(error.StackTooDeep, 14, mod(1, "\x02\x00\x20\x20\x20\x20\x20\xF0"));
    // Underflow for each popping opcode on an empty / one-deep stack.
    for ([_]u8{ 0x10, 0x11, 0x12 }) |op| {
        try expectErr(error.StackUnderflow, 8, &[_]u8{ 'K', 'X', 'B', 'C', 0, 0, 2, 0, op, 0xF0 });
        try expectErr(error.StackUnderflow, 10, &[_]u8{ 'K', 'X', 'B', 'C', 0, 1, 4, 0, 0x02, 0x00, op, 0xF0 });
    }
    for ([_]u8{ 0x20, 0x21, 0xF0 }) |op| try expectErr(error.StackUnderflow, 8, &[_]u8{ 'K', 'X', 'B', 'C', 0, 0, 2, 0, op, 0xF0 });
    // drop to empty, then ret underflows.
    try expectErr(error.StackUnderflow, 11, mod(1, "\x02\x00\x21\xF0"));
}

test "decoder: ret placement" {
    try expectErr(error.CodeAfterRet, 11, mod(1, "\x02\x00\xF0\xF0"));
    try expectErr(error.CodeAfterRet, 11, mod(1, "\x02\x00\xF0\x99")); // after-ret wins over unknown
    try expectErr(error.MissingRet, 12, mod(1, "\x02\x00\x20\x10"));
    // ret with extra stack values is allowed (returns the top).
    const f = try decodeOk(mod(2, "\x02\x00\x02\x01\xF0"));
    defer f.deinit(t.allocator);
    try t.expectEqual(@as(i64, 9), interpret(f, .{ 3, 9, 0, 0 }));
}

test "decoder: code lengths near the u16 limit do not overflow (regression)" {
    // Header claims 65535 code bytes but none follow: Truncated, not an integer overflow.
    try expectErr(error.Truncated, 6, "KXBC\x00\x00\xff\xff");
    try expectErr(error.Truncated, 6, "KXBC\x00\x00\xf8\xff\xF0");
}

test "decoder: trailing bytes after declared length" {
    for (1..16) |extra| {
        var buf: [32]u8 = undefined;
        @memcpy(buf[0..9], "KXBC\x00\x00\x01\x00\xF0");
        @memset(buf[9 .. 9 + extra], 0xF0);
        try expectErr(error.TrailingBytes, 6, buf[0 .. 9 + extra]);
    }
}

test "decoder: maximum code length 65535 accepted, encode rejects 65536" {
    // push_arg (2 bytes), then (push_i64 + add) units, then dup/drop pairs, then ret: exactly 65535.
    var list: std.ArrayList(bc.Inst) = .empty;
    defer list.deinit(t.allocator);
    try list.append(t.allocator, .{ .push_arg = 0 });
    var len: usize = 2;
    while (len + 10 + 1 <= 65535) : (len += 10) {
        try list.append(t.allocator, .{ .push_i64 = 1 });
        try list.append(t.allocator, .add);
    }
    while (len + 2 + 1 <= 65535) : (len += 2) {
        try list.append(t.allocator, .dup);
        try list.append(t.allocator, .drop);
    }
    if (len + 1 < 65535) return error.BadFill;
    try list.append(t.allocator, .ret);
    len += 1;
    try t.expectEqual(@as(usize, 65535), len);
    const bytes = try bc.encode(t.allocator, .{ .arg_count = 1, .insts = list.items });
    defer t.allocator.free(bytes);
    try t.expectEqual(@as(usize, 65535 + bc.header_len), bytes.len);
    const f = try decodeOk(bytes);
    defer f.deinit(t.allocator);
    try t.expectEqual(list.items.len, f.insts.len);
    // One more dup/drop pair overflows u16.
    try list.insert(t.allocator, 1, .dup);
    try list.insert(t.allocator, 2, .drop);
    try t.expectError(error.CodeTooLarge, bc.encode(t.allocator, .{ .arg_count = 1, .insts = list.items }));
}

test "decoder: allocation failure is reported as OutOfMemory without leaks" {
    const bytes = mod(1, "\x02\x00\x20\x10\x02\x00\x12\xF0");
    var k: usize = 0;
    while (true) : (k += 1) {
        var fa = std.testing.FailingAllocator.init(t.allocator, .{ .fail_index = k });
        var diag: bc.Diagnostic = .{};
        if (bc.decode(fa.allocator(), bytes, &diag)) |f| {
            f.deinit(fa.allocator());
            break;
        } else |e| try t.expectEqual(error.OutOfMemory, e);
        if (k > 64) return error.NeverSucceeded;
    }
}

test "property: random valid programs encode/decode round-trip (seeded, 2000 programs)" {
    var prng = std.Random.DefaultPrng.init(0x6b78_6263);
    const r = prng.random();
    var buf: [64]bc.Inst = undefined;
    for (0..2000) |_| {
        const argc = r.uintAtMost(u8, bc.max_args);
        const p = randomProgram(r, &buf, argc);
        const bytes = try bc.encode(t.allocator, .{ .arg_count = argc, .insts = p });
        defer t.allocator.free(bytes);
        const f = try decodeOk(bytes);
        defer f.deinit(t.allocator);
        try t.expectEqualSlices(bc.Inst, p, f.insts);
    }
}

test "fuzz: mutated modules never crash; accepted ones re-encode byte-identically (seeded)" {
    var prng = std.Random.DefaultPrng.init(0xf022);
    const r = prng.random();
    var buf: [48]bc.Inst = undefined;
    var accepted: usize = 0;
    for (0..3000) |_| {
        const argc = r.uintAtMost(u8, bc.max_args);
        const bytes = try bc.encode(t.allocator, .{ .arg_count = argc, .insts = randomProgram(r, &buf, argc) });
        defer t.allocator.free(bytes);
        const flips = 1 + r.uintLessThan(usize, 3);
        for (0..flips) |_| bytes[r.uintLessThan(usize, bytes.len)] = r.int(u8);
        const cut = if (r.boolean()) bytes.len else r.uintAtMost(usize, bytes.len);
        var diag: bc.Diagnostic = .{};
        const f = bc.decode(t.allocator, bytes[0..cut], &diag) catch {
            try t.expect(diag.offset <= cut);
            continue;
        };
        defer f.deinit(t.allocator);
        accepted += 1;
        const again = try bc.encode(t.allocator, f);
        defer t.allocator.free(again);
        try t.expectEqualSlices(u8, bytes[0..cut], again);
        // Anything the decoder accepts must lower on every built target.
        _ = try checkAllTargets(f, .{ 1, -2, 3, -4 });
    }
    try t.expect(accepted > 0);
}

// ---- --arch flag ------------------------------------------------------------------------

test "arch flag: every target round-trips through its flag name" {
    for (std.meta.tags(Target)) |target| try t.expectEqual(target, Target.fromFlag(target.flagName()).?);
    try t.expectEqual(@as(usize, 3), std.meta.tags(Target).len);
}

test "arch flag: near-miss and invalid values are rejected" {
    const bad = [_][]const u8{ "", " ", "x86", "x86_64", "X86-64", "x86-64 ", " x86-64", "amd64", "x64", "i386", "arm64", "AArch64", "aarch64be", "aarch32", "arm", "riscv64", "RISCV", "rv64", "riscv32", "mips", "wasm32", "x86-64\x00", "aarch64,riscv" };
    for (bad) |s| try t.expectEqual(@as(?Target, null), Target.fromFlag(s));
}

// ---- every opcode at every stack depth on every arch ------------------------------------

/// Program that fills the stack to `depth` from args/immediates, applies `op`, then folds
/// the stack with `add` and returns. Valid for every opcode whose pops fit `depth`.
fn opAtDepth(buf: []bc.Inst, op: bc.Inst, depth: u8) ?[]bc.Inst {
    if (depth < op.pops()) return null;
    if (@as(i16, depth) + op.stackEffect() > bc.max_depth) return null;
    var n: usize = 0;
    for (0..depth) |k| {
        buf[n] = if (k % 2 == 0) .{ .push_arg = @intCast(k % 4) } else .{ .push_i64 = @as(i64, @intCast(k)) * -1_000_003 };
        n += 1;
    }
    if (op == .ret) {
        buf[n] = .ret;
        return buf[0 .. n + 1];
    }
    buf[n] = op;
    n += 1;
    var d: i16 = @as(i16, depth) + op.stackEffect();
    if (d == 0) {
        buf[n] = .{ .push_i64 = 5 };
        n += 1;
        d = 1;
    }
    while (d > 1) : (d -= 1) {
        buf[n] = .add;
        n += 1;
    }
    buf[n] = .ret;
    return buf[0 .. n + 1];
}

fn opSamples() [10]bc.Inst {
    return .{ .{ .push_i64 = 0 }, .{ .push_i64 = std.math.minInt(i64) }, .{ .push_arg = 0 }, .{ .push_arg = 3 }, .add, .sub, .mul, .dup, .drop, .ret };
}

fn opAllDepths(target: Target) !usize {
    if (!target.isBuilt()) return error.SkipZigTest;
    var buf: [16]bc.Inst = undefined;
    var n: usize = 0;
    const argsets = [_][4]i64{ .{ 0, 0, 0, 0 }, .{ 7, -3, 11, 2 }, .{ std.math.maxInt(i64), std.math.minInt(i64), -1, 1 } };
    for (opSamples()) |op| for (0..bc.max_depth + 1) |depth| {
        const p = opAtDepth(&buf, op, @intCast(depth)) orelse continue;
        const f: bc.Function = .{ .arg_count = 4, .insts = p };
        // The program must pass the decoder too.
        const bytes = try bc.encode(t.allocator, f);
        defer t.allocator.free(bytes);
        (try decodeOk(bytes)).deinit(t.allocator);
        var c = try driver.compile(t.allocator, target, f);
        defer c.deinit();
        try t.expect(std.mem.endsWith(u8, c.code(), retBytes(target)));
        for (argsets) |a| {
            const want = interpret(f, a);
            if (c.exec != null) try t.expectEqual(want, c.call(a));
            if (target == .riscv64) try t.expectEqual(want, try simRv64(c.code(), a));
        }
        n += 1;
    };
    return n;
}

test "every opcode at every stack depth: x86-64" {
    try t.expect(try opAllDepths(.x86_64) > 30);
}
test "every opcode at every stack depth: aarch64" {
    try t.expect(try opAllDepths(.aarch64) > 30);
}
test "every opcode at every stack depth: riscv" {
    try t.expect(try opAllDepths(.riscv64) > 30);
}

// ---- boundary immediates, golden bytes ---------------------------------------------------

fn immGolden(target: Target) !void {
    if (!target.isBuilt()) return error.SkipZigTest;
    for (imm_cases) |c| {
        const got = try lowerBytes(target, .{ .arg_count = 0, .insts = &.{ .{ .push_i64 = c.v }, .ret } });
        defer t.allocator.free(got);
        const want = switch (target) {
            .x86_64 => c.x86,
            .aarch64 => c.a64,
            .riscv64 => c.rv,
        };
        t.expectEqualSlices(u8, want, got) catch |e| {
            std.debug.print("push_i64 {d} --arch {s}\n", .{ c.v, target.flagName() });
            return e;
        };
    }
}

test "boundary immediates, golden bytes: x86-64" {
    try immGolden(.x86_64);
}
test "boundary immediates, golden bytes: aarch64" {
    try immGolden(.aarch64);
}
test "boundary immediates, golden bytes: riscv" {
    try immGolden(.riscv64);
}

test "boundary immediates: values reproduced (host execution / RV64 sim)" {
    for (imm_cases) |c| _ = try checkAllTargets(.{ .arg_count = 0, .insts = &.{ .{ .push_i64 = c.v }, .ret } }, .{ 0, 0, 0, 0 });
}

test "riscv li: random 64-bit immediates materialize exactly (sim, 5000 values)" {
    var prng = std.Random.DefaultPrng.init(0x11);
    const r = prng.random();
    for (0..5000) |k| {
        const v: i64 = switch (k % 4) {
            0 => r.int(i64),
            1 => r.int(i32),
            2 => @as(i64, r.int(i16)) << r.int(u6),
            else => @as(i64, @bitCast(@as(u64, 1) << r.int(u6))) +% r.intRangeAtMost(i64, -2049, 2049),
        };
        const code = try lowerBytes(.riscv64, .{ .arg_count = 0, .insts = &.{ .{ .push_i64 = v }, .ret } });
        defer t.allocator.free(code);
        try t.expectEqual(v, try simRv64(code, .{ 0, 0, 0, 0 }));
    }
}

// ---- wrapping arithmetic edge cases -----------------------------------------------------

test "arithmetic edge cases: overflow wraps identically on every arch" {
    const max = std.math.maxInt(i64);
    const min = std.math.minInt(i64);
    const pairs = [_][2]i64{ .{ max, 1 }, .{ min, -1 }, .{ min, min }, .{ max, max }, .{ 0, min }, .{ -1, -1 }, .{ 0x1_0000_0000, 0x1_0000_0000 }, .{ 3, -7 } };
    for ([_]bc.Inst{ .add, .sub, .mul }) |op| for (pairs) |p| {
        const f: bc.Function = .{ .arg_count = 2, .insts = &.{ .{ .push_arg = 0 }, .{ .push_arg = 1 }, op, .ret } };
        _ = try checkAllTargets(f, .{ p[0], p[1], 0, 0 });
        const g: bc.Function = .{ .arg_count = 0, .insts = &.{ .{ .push_i64 = p[0] }, .{ .push_i64 = p[1] }, op, .ret } };
        _ = try checkAllTargets(g, .{ 0, 0, 0, 0 });
    };
}

test "argument registers: each of the four args reaches the code on every arch" {
    for (0..4) |n| {
        const f: bc.Function = .{ .arg_count = 4, .insts = &.{ .{ .push_arg = @intCast(n) }, .ret } };
        _ = try checkAllTargets(f, .{ 101, -202, 303, -404 });
    }
    // Unused argument registers do not leak into the result.
    _ = try checkAllTargets(.{ .arg_count = 0, .insts = &.{ .{ .push_i64 = 9 }, .ret } }, .{ 1, 2, 3, 4 });
}

// ---- differential: random programs vs reference interpreter ------------------------------

test "differential: 1500 random programs agree with the interpreter on every arch (seeded)" {
    var prng = std.Random.DefaultPrng.init(0xb2c_17);
    const r = prng.random();
    var buf: [96]bc.Inst = undefined;
    var compared: usize = 0;
    for (0..1500) |_| {
        const argc = r.uintAtMost(u8, bc.max_args);
        const p = randomProgram(r, &buf, argc);
        const args = [4]i64{ r.int(i64), r.intRangeAtMost(i64, -100, 100), r.int(i32), r.int(i64) };
        compared += try checkAllTargets(.{ .arg_count = argc, .insts = p }, args);
    }
    // At least RISC-V (simulated) for every program.
    try t.expect(compared >= 1500);
}

// ---- stress: large functions --------------------------------------------------------------

fn stressProgram(gpa: std.mem.Allocator, kind: u8) ![]bc.Inst {
    var list: std.ArrayList(bc.Inst) = .empty;
    errdefer list.deinit(gpa);
    var len: usize = 0; // encoded code bytes
    var prng = std.Random.DefaultPrng.init(kind);
    const r = prng.random();
    switch (kind) {
        // Long chain of 64-bit immediates (worst-case materialization) folded with add/sub/mul.
        0 => {
            try list.append(gpa, .{ .push_i64 = r.int(i64) });
            len = 9;
            var k: usize = 0;
            while (len + 10 + 1 <= 65535) : ({
                len += 10;
                k += 1;
            }) {
                try list.append(gpa, .{ .push_i64 = r.int(i64) });
                try list.append(gpa, switch (k % 3) {
                    0 => .add,
                    1 => .sub,
                    else => .mul,
                });
            }
        },
        // Max-depth sawtooth over arguments: push to depth 5, fold back, repeat.
        1 => {
            try list.append(gpa, .{ .push_arg = 0 });
            len = 2;
            const unit = [_]bc.Inst{ .{ .push_arg = 1 }, .{ .push_arg = 2 }, .dup, .{ .push_arg = 3 }, .mul, .sub, .add, .dup, .drop, .add };
            const unit_len = 2 + 2 + 1 + 2 + 1 + 1 + 1 + 1 + 1 + 1;
            // (the trailing add consumes the dup'ed value... keep depth balanced)
            while (len + unit_len + 1 <= 65535) : (len += unit_len) try list.appendSlice(gpa, &unit);
        },
        else => unreachable,
    }
    try list.append(gpa, .ret);
    return list.toOwnedSlice(gpa);
}

fn stress(kind: u8) !void {
    const p = try stressProgram(t.allocator, kind);
    defer t.allocator.free(p);
    const f: bc.Function = .{ .arg_count = 4, .insts = p };
    const bytes = try bc.encode(t.allocator, f);
    defer t.allocator.free(bytes);
    try t.expect(bytes.len > 65000);
    const g = try decodeOk(bytes);
    defer g.deinit(t.allocator);
    const args = [4]i64{ 3, -5, 7, 11 };
    const want = interpret(g, args);
    for (all_targets) |target| {
        if (!target.isBuilt()) continue;
        var c = try driver.compile(t.allocator, target, g);
        defer c.deinit();
        try t.expect(c.code().len > p.len); // every non-drop op emits at least one instruction
        if (c.exec != null) try t.expectEqual(want, c.call(args));
        if (target == .riscv64) try t.expectEqual(want, try simRv64(c.code(), args));
    }
}

test "stress: 64 KiB of 64-bit immediates and arithmetic on every arch" {
    try stress(0);
}
test "stress: 64 KiB max-depth sawtooth over all arguments on every arch" {
    try stress(1);
}

test "stress: lowering is deterministic and code buffers are independent" {
    const p = try stressProgram(t.allocator, 0);
    defer t.allocator.free(p);
    for (all_targets) |target| {
        if (!target.isBuilt()) continue;
        const a = try lowerBytes(target, .{ .arg_count = 4, .insts = p });
        defer t.allocator.free(a);
        const b = try lowerBytes(target, .{ .arg_count = 4, .insts = p });
        defer t.allocator.free(b);
        try t.expectEqualSlices(u8, a, b);
    }
}

test "driver: many host compilations map and release cleanly" {
    const host = Target.host() orelse return error.SkipZigTest;
    if (!host.canExecute() or !host.isBuilt()) return error.SkipZigTest;
    for (0..200) |k| {
        const f: bc.Function = .{ .arg_count = 1, .insts = &.{ .{ .push_arg = 0 }, .{ .push_i64 = @intCast(k) }, .add, .ret } };
        var c = try driver.compile(t.allocator, host, f);
        defer c.deinit();
        try t.expectEqual(@as(i64, @intCast(k + 1)), c.call(.{ 1, 0, 0, 0 }));
    }
}
