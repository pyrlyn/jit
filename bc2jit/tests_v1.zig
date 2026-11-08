//! KXBC v1 tests: decoder (control flow validation), reference interpreter semantics, the IR
//! lowering on every target, host execution against `bytecode.eval`, v0 parity between the
//! direct and the IR lowering, and a random-program differential fuzz.
const std = @import("std");
const jit = @import("pyrlyn_jit");
const bc = @import("bytecode.zig");
const lower = @import("lower.zig");
const lower_ir = @import("lower_ir.zig");
const driver = @import("driver.zig");
const Target = lower.Target;
const t = std.testing;

/// Small assembler for test programs: instructions with symbolic jump targets.
const Prog = struct {
    insts: std.ArrayList(bc.Inst) = .empty,
    /// For each instruction: label it jumps to (index into `labels`), if any.
    refs: std.ArrayList(?u32) = .empty,
    labels: std.ArrayList(?u32) = .empty, // label -> instruction index
    fn deinit(p: *Prog) void {
        p.insts.deinit(t.allocator);
        p.refs.deinit(t.allocator);
        p.labels.deinit(t.allocator);
    }
    fn op(p: *Prog, i: bc.Inst) !void {
        try p.insts.append(t.allocator, i);
        try p.refs.append(t.allocator, null);
    }
    fn label(p: *Prog) !u32 {
        try p.labels.append(t.allocator, null);
        return @intCast(p.labels.items.len - 1);
    }
    fn bind(p: *Prog, l: u32) void {
        p.labels.items[l] = @intCast(p.insts.items.len);
    }
    fn jmp(p: *Prog, l: u32) !void {
        try p.insts.append(t.allocator, .{ .jmp = 0 });
        try p.refs.append(t.allocator, l);
    }
    fn brIf(p: *Prog, c: bc.Cond, l: u32) !void {
        try p.insts.append(t.allocator, .{ .br_if = .{ .cond = c, .target = 0 } });
        try p.refs.append(t.allocator, l);
    }
    /// Encoded module (caller frees).
    fn module(p: *Prog, argc: u8) ![]u8 {
        var offs = try t.allocator.alloc(u16, p.insts.items.len + 1);
        defer t.allocator.free(offs);
        var o: u16 = 0;
        for (p.insts.items, 0..) |i, k| {
            offs[k] = o;
            o += size(i);
        }
        offs[p.insts.items.len] = o;
        for (p.insts.items, p.refs.items) |*i, r| if (r) |l| {
            const tg = offs[p.labels.items[l].?];
            switch (i.*) {
                .jmp => i.* = .{ .jmp = tg },
                .br_if => |x| i.* = .{ .br_if = .{ .cond = x.cond, .target = tg } },
                else => unreachable,
            }
        };
        return bc.encode(t.allocator, .{ .arg_count = argc, .insts = p.insts.items, .version = bc.version_v1 });
    }
};

fn size(i: bc.Inst) u16 {
    return switch (i) {
        .push_i64 => 9,
        .push_arg, .load_local, .store_local => 2,
        .jmp => 3,
        .br_if => 4,
        else => 1,
    };
}

fn decodeOk(bytes: []const u8) !bc.Function {
    var d: bc.Diagnostic = .{};
    return bc.decode(t.allocator, bytes, &d) catch |e| {
        std.debug.print("decode failed at {d}: {s}\n", .{ d.offset, bc.errorMessage(e) });
        return e;
    };
}

/// Compiles for the host through the driver and checks against `eval` for every arg set.
fn checkHost(f: bc.Function, arg_sets: []const [4]i64) !void {
    const host = Target.host() orelse return error.SkipZigTest;
    if (!host.isBuilt()) return error.SkipZigTest;
    var c = try driver.compile(t.allocator, host, f);
    defer c.deinit();
    try t.expect(c.exec != null);
    for (arg_sets) |a| {
        const want = try bc.eval(f, a, 1 << 24);
        const got = c.call(a);
        if (got != want) {
            std.debug.print("args {any}: got {d} want {d}\n", .{ a, got, want });
            return error.TestUnexpectedResult;
        }
    }
}

/// factorial(n) with a loop over locals: r = 1; while (n > 0) { r *= n; n -= 1 } return r
fn factorial(p: *Prog) !void {
    const top = try p.label();
    const done = try p.label();
    try p.op(.{ .push_i64 = 1 });
    try p.op(.{ .store_local = 0 });
    try p.op(.{ .push_arg = 0 });
    try p.op(.{ .store_local = 1 });
    p.bind(top);
    try p.op(.{ .load_local = 1 });
    try p.op(.{ .push_i64 = 0 });
    try p.brIf(.le, done);
    try p.op(.{ .load_local = 0 });
    try p.op(.{ .load_local = 1 });
    try p.op(.mul);
    try p.op(.{ .store_local = 0 });
    try p.op(.{ .load_local = 1 });
    try p.op(.{ .push_i64 = 1 });
    try p.op(.sub);
    try p.op(.{ .store_local = 1 });
    try p.jmp(top);
    p.bind(done);
    try p.op(.{ .load_local = 0 });
    try p.op(.ret);
}

/// gcd(a, b) with rem and swap on the stack: while (b != 0) { (a, b) = (b, a % b) }
fn gcd(p: *Prog) !void {
    const top = try p.label();
    const done = try p.label();
    try p.op(.{ .push_arg = 0 });
    try p.op(.{ .push_arg = 1 });
    p.bind(top); // stack: a b
    try p.op(.dup);
    try p.op(.{ .push_i64 = 0 });
    try p.brIf(.eq, done);
    try p.op(.dup); // a b b
    try p.op(.{ .store_local = 0 }); // a b   (l0 = b)
    try p.op(.rem); // a%b
    try p.op(.{ .load_local = 0 }); // a%b b
    try p.op(.swap); // b a%b
    try p.jmp(top);
    p.bind(done);
    try p.op(.drop);
    try p.op(.ret);
}

test "v1: decode/encode round trip of every opcode, depths" {
    var p: Prog = .{};
    defer p.deinit();
    try gcd(&p);
    const m = try p.module(2);
    defer t.allocator.free(m);
    const f = try decodeOk(m);
    defer f.deinit(t.allocator);
    try t.expectEqual(bc.version_v1, f.version);
    // push_arg push_arg | dup push br_if | dup store rem load swap jmp | drop ret
    try t.expectEqualSlices(u8, &.{ 0, 1, 2, 3, 4, 2, 3, 2, 1, 2, 2, 2, 1 }, f.depths);
    const again = try bc.encode(t.allocator, f);
    defer t.allocator.free(again);
    try t.expectEqualSlices(u8, m, again);
}

fn expectErr(e: bc.Error, offset: usize, bytes: []const u8) !void {
    var d: bc.Diagnostic = .{};
    try t.expectError(e, bc.decode(t.allocator, bytes, &d));
    try t.expectEqual(offset, d.offset);
}

test "v1: decoder rejects bad control flow with offsets" {
    // jmp into the middle of push_i64
    try expectErr(error.BadJumpTarget, 8, "KXBC\x01\x00\x0d\x00" ++ "\x40\x04\x00" ++ "\x01\x01\x00\x00\x00\x00\x00\x00\x00" ++ "\xF0");
    // join with depth 1 (fall-through after push) and 0 (jump): push 1; br_if? simpler: jmp over a push
    //   0: push_arg 0   2: push_arg 0   4: br_if eq -> 9   8: (depth 0) ... target 9 sees depth 0 from br_if
    //   9 reached by fall-through from push_arg at 7 with depth 1 => mismatch at 9
    try expectErr(error.StackMismatch, 8 + 10, "KXBC\x01\x01\x0c\x00" ++ "\x02\x00" ++ "\x02\x00" ++ "\x41\x00\x0a\x00" ++ "\x02\x00" ++ "\xF0" ++ "\xF0");
    // unreachable instruction after ret
    try expectErr(error.UnreachableCode, 8 + 3, "KXBC\x01\x01\x04\x00" ++ "\x02\x00\xF0" ++ "\xF0");
    // falls off the end
    try expectErr(error.MissingRet, 8 + 2, "KXBC\x01\x01\x02\x00" ++ "\x02\x00");
    // local index and condition
    try expectErr(error.LocalOutOfRange, 8, "KXBC\x01\x00\x03\x00" ++ "\x30\x10\xF0");
    try expectErr(error.UnknownCondition, 8, "KXBC\x01\x00\x05\x00" ++ "\x41\x0a\x00\x00\xF0");
    // v1 opcodes are unknown in v0; version 2 is unsupported
    try expectErr(error.UnknownOpcode, 8 + 4, "KXBC\x00\x01\x06\x00" ++ "\x02\x00\x02\x00\x13\xF0");
    try expectErr(error.UnsupportedVersion, 4, "KXBC\x02\x00\x01\x00\xF0");
    // underflow inside a loop body, too deep (65 pushes)
    try expectErr(error.StackUnderflow, 8, "KXBC\x01\x00\x02\x00" ++ "\x10\xF0");
    var deep: [8 + 65 * 2 + 1]u8 = undefined;
    @memcpy(deep[0..8], "KXBC\x01\x01\x83\x00");
    for (0..65) |i| @memcpy(deep[8 + 2 * i ..][0..2], "\x02\x00");
    deep[deep.len - 1] = 0xF0;
    try expectErr(error.StackTooDeep, 8 + 64 * 2, &deep);
}

test "v1: interpreter semantics (total division, shifts mod 64, unsigned compares)" {
    const I = bc.Inst;
    try t.expectEqual(@as(i64, 0), bc.binop(I.div, 7, 0));
    try t.expectEqual(@as(i64, 7), bc.binop(I.rem, 7, 0));
    try t.expectEqual(@as(i64, std.math.minInt(i64)), bc.binop(I.div, std.math.minInt(i64), -1));
    try t.expectEqual(@as(i64, 0), bc.binop(I.rem, std.math.minInt(i64), -1));
    try t.expectEqual(@as(i64, -3), bc.binop(I.div, -7, 2));
    try t.expectEqual(@as(i64, -1), bc.binop(I.rem, -7, 2));
    try t.expectEqual(@as(i64, 2), bc.binop(I.shl, 1, 65));
    try t.expectEqual(@as(i64, 1), bc.binop(I.shr, -1, 63));
    try t.expectEqual(@as(i64, -1), bc.binop(I.sar, -1, 63));
    try t.expect(bc.condHolds(.ugt, -1, 1) and !bc.condHolds(.gt, -1, 1));
}

test "v1: factorial and gcd on the host match the interpreter" {
    inline for (.{ .{ factorial, 1 }, .{ gcd, 2 } }) |c| {
        var p: Prog = .{};
        defer p.deinit();
        try c[0](&p);
        const m = try p.module(c[1]);
        defer t.allocator.free(m);
        const f = try decodeOk(m);
        defer f.deinit(t.allocator);
        try checkHost(f, &.{ .{ 0, 0, 0, 0 }, .{ 1, 7, 0, 0 }, .{ 5, 15, 0, 0 }, .{ 20, 1071, 0, 0 }, .{ 1071, 462, 0, 0 }, .{ -12, 18, 0, 0 }, .{ 25, 0, 0, 0 } });
    }
    // factorial(20) = 2432902008176640000
    var p: Prog = .{};
    defer p.deinit();
    try factorial(&p);
    const m = try p.module(1);
    defer t.allocator.free(m);
    const f = try decodeOk(m);
    defer f.deinit(t.allocator);
    try t.expectEqual(@as(i64, 2432902008176640000), try bc.eval(f, .{ 20, 0, 0, 0 }, 1000));
}

test "v1: division edge cases and every binary op/condition execute like the interpreter" {
    const ops = [_]bc.Inst{ .add, .sub, .mul, .div, .rem, .@"and", .@"or", .xor, .shl, .shr, .sar };
    const vals = [_]i64{ 0, 1, -1, 2, -7, 63, 64, 65, std.math.minInt(i64), std.math.maxInt(i64), 0x1234_5678_9abc };
    var sets: [vals.len * vals.len][4]i64 = undefined;
    for (vals, 0..) |a, i| for (vals, 0..) |b, j| {
        sets[i * vals.len + j] = .{ a, b, 0, 0 };
    };
    for (ops) |o| {
        var p: Prog = .{};
        defer p.deinit();
        try p.op(.{ .push_arg = 0 });
        try p.op(.{ .push_arg = 1 });
        try p.op(o);
        try p.op(.ret);
        const m = try p.module(2);
        defer t.allocator.free(m);
        const f = try decodeOk(m);
        defer f.deinit(t.allocator);
        try checkHost(f, &sets);
    }
    for (std.meta.tags(bc.Cond)) |c| {
        var p: Prog = .{};
        defer p.deinit();
        const yes = try p.label();
        try p.op(.{ .push_arg = 0 });
        try p.op(.{ .push_arg = 1 });
        try p.brIf(c, yes);
        try p.op(.{ .push_i64 = 10 });
        try p.op(.ret);
        p.bind(yes);
        try p.op(.{ .push_i64 = 20 });
        try p.op(.ret);
        const m = try p.module(2);
        defer t.allocator.free(m);
        const f = try decodeOk(m);
        defer f.deinit(t.allocator);
        try checkHost(f, &sets);
    }
}

test "v1: 64-deep stack spills and runs" {
    var p: Prog = .{};
    defer p.deinit();
    for (0..64) |i| try p.op(if (i % 3 == 0) .{ .push_arg = @intCast(i % 4) } else .{ .push_i64 = @as(i64, @intCast(i)) * 7 - 100 });
    for (0..63) |i| try p.op(if (i % 2 == 0) .add else .xor);
    try p.op(.ret);
    const m = try p.module(4);
    defer t.allocator.free(m);
    const f = try decodeOk(m);
    defer f.deinit(t.allocator);
    try checkHost(f, &.{ .{ 1, 2, 3, 4 }, .{ -5, 100, std.math.maxInt(i64), 0 } });
}

test "v1: every target lowers through the IR compilers (bytes)" {
    var p: Prog = .{};
    defer p.deinit();
    try gcd(&p);
    const m = try p.module(2);
    defer t.allocator.free(m);
    const f = try decodeOk(m);
    defer f.deinit(t.allocator);
    inline for (std.meta.tags(Target)) |tg| if (tg.isBuilt()) {
        var cb = jit.CodeBuffer.init(t.allocator);
        defer cb.deinit();
        try lower_ir.lowerIr(t.allocator, tg, f, &cb);
        try t.expect(cb.code().len > 16);
    };
}

test "v1: v0 modules give the same results through the IR lowering" {
    const host = Target.host() orelse return error.SkipZigTest;
    if (!host.canExecute()) return error.SkipZigTest;
    const progs = [_][]const bc.Inst{
        &.{ .{ .push_arg = 0 }, .{ .push_arg = 1 }, .add, .ret },
        &.{ .{ .push_arg = 0 }, .dup, .mul, .{ .push_i64 = 3 }, .{ .push_arg = 0 }, .mul, .sub, .{ .push_i64 = 7 }, .add, .ret },
        &.{ .{ .push_i64 = 1 }, .{ .push_i64 = 2 }, .{ .push_i64 = 3 }, .{ .push_i64 = 4 }, .{ .push_i64 = 100 }, .drop, .add, .add, .add, .ret },
        &.{ .{ .push_arg = 0 }, .{ .push_arg = 1 }, .mul, .{ .push_arg = 2 }, .{ .push_arg = 3 }, .mul, .add, .ret },
    };
    for (progs) |insts| {
        const f: bc.Function = .{ .arg_count = 4, .insts = insts };
        var direct = try driver.compile(t.allocator, host, f);
        defer direct.deinit();
        var cb = jit.CodeBuffer.init(t.allocator);
        defer cb.deinit();
        try lower_ir.lowerIr(t.allocator, host, f, &cb);
        const fun = try jit.JitRuntime.add(&cb);
        defer fun.release();
        for ([_][4]i64{ .{ 1, 2, 3, 4 }, .{ -9, 8, std.math.minInt(i64), 3 } }) |a| {
            const via_ir = fun.ptr(driver.Fn)(a[0], a[1], a[2], a[3]);
            try t.expectEqual(direct.call(a), via_ir);
            try t.expectEqual(try bc.eval(f, a, 1000), via_ir);
        }
    }
}

/// Random terminating v1 program: straight-line segments with forward branches, one counted
/// loop (local 15 counts down from 1..8), locals, every op.
fn genProgram(rng: std.Random, p: *Prog) !void {
    const binops = [_]bc.Inst{ .add, .sub, .mul, .div, .rem, .@"and", .@"or", .xor, .shl, .shr, .sar };
    try p.op(.{ .push_i64 = rng.intRangeAtMost(i64, 1, 8) });
    try p.op(.{ .store_local = 15 });
    const top = try p.label();
    p.bind(top);
    var depth: usize = 0;
    var pending: ?u32 = null; // forward label bound at depth 0
    const n = rng.intRangeAtMost(usize, 3, 60);
    for (0..n) |_| {
        const r = rng.uintLessThan(u8, 12);
        if (depth < 2 or r < 4) {
            if (depth >= 60) break;
            try p.op(switch (rng.uintLessThan(u8, 4)) {
                0 => .{ .push_arg = rng.uintLessThan(u8, 4) },
                1 => .{ .load_local = rng.uintLessThan(u8, 15) },
                2 => .{ .push_i64 = rng.intRangeAtMost(i64, -100, 100) },
                else => .{ .push_i64 = @bitCast(rng.int(u64)) },
            });
            depth += 1;
        } else if (r < 9) {
            try p.op(binops[rng.uintLessThan(usize, binops.len)]);
            depth -= 1;
        } else if (r == 9) {
            try p.op(if (rng.boolean()) .swap else .dup);
            if (p.insts.items[p.insts.items.len - 1] == .dup) depth += 1;
        } else if (r == 10) {
            try p.op(.{ .store_local = rng.uintLessThan(u8, 15) });
            depth -= 1;
        } else if (depth == 2 and pending == null) {
            const l = try p.label();
            try p.brIf(rng.enumValue(bc.Cond), l);
            depth = 0;
            pending = l;
        }
        if (pending) |l| if (depth == 0 and rng.boolean()) {
            p.bind(l);
            pending = null;
        };
    }
    while (depth > 0) : (depth -= 1) try p.op(.{ .store_local = rng.uintLessThan(u8, 15) });
    if (pending) |l| p.bind(l);
    // loop: if (--l15 > 0) goto top
    try p.op(.{ .load_local = 15 });
    try p.op(.{ .push_i64 = 1 });
    try p.op(.sub);
    try p.op(.dup);
    try p.op(.{ .store_local = 15 });
    try p.op(.{ .push_i64 = 0 });
    try p.brIf(.gt, top);
    // result: xor of all locals
    try p.op(.{ .load_local = 0 });
    for (1..15) |k| {
        try p.op(.{ .load_local = @intCast(k) });
        try p.op(.xor);
    }
    try p.op(.ret);
}

test "v1: random programs match the interpreter on the host (fuzz)" {
    const host = Target.host() orelse return error.SkipZigTest;
    if (!host.isBuilt()) return error.SkipZigTest;
    var prng = std.Random.DefaultPrng.init(0xC0DE_B17E);
    const rng = prng.random();
    for (0..400) |_| {
        var p: Prog = .{};
        defer p.deinit();
        try genProgram(rng, &p);
        const m = try p.module(4);
        defer t.allocator.free(m);
        const f = try decodeOk(m);
        defer f.deinit(t.allocator);
        var sets: [3][4]i64 = undefined;
        for (&sets) |*s| for (s) |*v| {
            v.* = if (rng.boolean()) rng.intRangeAtMost(i64, -64, 64) else @bitCast(rng.int(u64));
        };
        try checkHost(f, &sets);
    }
}

test "v1 decoder: no leak on any allocation failure" {
    const bytes = try bc.encode(t.allocator, .{ .arg_count = 1, .version = bc.version_v1, .insts = &.{
        .{ .push_arg = 0 }, .{ .push_i64 = 2 }, .add, .ret,
    } });
    defer t.allocator.free(bytes);
    try t.checkAllAllocationFailures(t.allocator, struct {
        fn f(gpa: std.mem.Allocator, b: []const u8) !void {
            var d: bc.Diagnostic = .{};
            const g = try bc.decode(gpa, b, &d);
            g.deinit(gpa);
        }
    }.f, .{bytes});
}
