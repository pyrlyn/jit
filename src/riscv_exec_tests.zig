//! Test root for a riscv64-linux process (`zig build test-riscv64`, run under qemu-user in CI
//! with `-fqemu`): assembles RV64 code with `pyrlyn_jit.riscv`, maps it through JitRuntime and
//! calls it. Every test skips on other hosts.
const std = @import("std");
const builtin = @import("builtin");
const rv = @import("riscv/assembler.zig");
const CodeBuffer = @import("core/code_buffer.zig").CodeBuffer;
const JitRuntime = @import("core/jit_runtime.zig");
const t = std.testing;

test {
    _ = @import("riscv/compiler.zig"); // Compiler execution tests
    _ = @import("riscv/builder.zig");
}

const can_run = builtin.cpu.arch == .riscv64;
const Fn3 = *const fn (i64, i64, i64) callconv(.c) i64;

fn jit(cb: *CodeBuffer) !JitRuntime.Function {
    try cb.finalize();
    return JitRuntime.add(cb);
}

fn constFn(comptime A: type, v: i64) !i64 {
    var cb = CodeBuffer.init(t.allocator);
    defer cb.deinit();
    const a = A.init(&cb);
    try a.li(.a0, v);
    try a.ret();
    const f = try jit(&cb);
    defer f.release();
    return f.ptr(Fn3)(0, 0, 0);
}

test "riscv64 exec: li materializes any 64-bit constant (plain and compressed)" {
    if (!can_run) return error.SkipZigTest;
    const vals = [_]i64{ 0, 1, -1, 31, -32, 2047, -2048, 2048, 0x7FFFF800, 0x7FFFFFFF, -0x80000000, 0x80000000, 0xFFFFFFFF, 0x123456789ABCDEF0, std.math.maxInt(i64), std.math.minInt(i64), 0x0000_0800_0000_0000, -0x1234_5678_9ABC };
    for (vals) |v| {
        try t.expectEqual(v, try constFn(rv.Assembler, v));
        try t.expectEqual(v, try constFn(rv.AssemblerC, v));
    }
    var prng = std.Random.DefaultPrng.init(0x5EED);
    for (0..200) |_| {
        const v = prng.random().int(i64);
        try t.expectEqual(v, try constFn(rv.Assembler, v));
        try t.expectEqual(v, try constFn(rv.AssemblerC, v));
    }
}

test "riscv64 exec: M extension and W forms match Zig semantics" {
    if (!can_run) return error.SkipZigTest;
    const Op = enum { mul, div, rem, addw, sub, sll, sra };
    inline for (std.meta.tags(Op)) |op| {
        var cb = CodeBuffer.init(t.allocator);
        defer cb.deinit();
        const a = rv.Assembler.init(&cb);
        switch (op) {
            .mul => try a.mul(.a0, .a0, .a1),
            .div => try a.div(.a0, .a0, .a1),
            .rem => try a.rem(.a0, .a0, .a1),
            .addw => try a.addw(.a0, .a0, .a1),
            .sub => try a.sub(.a0, .a0, .a1),
            .sll => try a.sll(.a0, .a0, .a1),
            .sra => try a.sra(.a0, .a0, .a1),
        }
        try a.ret();
        const f = try jit(&cb);
        defer f.release();
        const cases = [_][2]i64{ .{ 7, 2 }, .{ -7, 2 }, .{ 7, -2 }, .{ std.math.minInt(i64), -1 }, .{ 5, 0 }, .{ 0x7FFFFFFF, 1 }, .{ 1, 63 } };
        for (cases) |c| {
            const x = c[0];
            const y = c[1];
            const want: i64 = switch (op) {
                .mul => x *% y,
                // RISC-V: x/0 = -1, x%0 = x, minInt/-1 = minInt, minInt%-1 = 0.
                .div => if (y == 0) -1 else if (y == -1) 0 -% x else @divTrunc(x, y),
                .rem => if (y == 0) x else if (y == -1) 0 else @rem(x, y),
                .addw => @as(i32, @truncate(x)) +% @as(i32, @truncate(y)),
                .sub => x -% y,
                .sll => x << @as(u6, @truncate(@as(u64, @bitCast(y)))),
                .sra => x >> @as(u6, @truncate(@as(u64, @bitCast(y)))),
            };
            try t.expectEqual(want, f.ptr(Fn3)(x, y, 0));
        }
    }
}

test "riscv64 exec: loop with branches and labels (sum 1..n, plain and far)" {
    if (!can_run) return error.SkipZigTest;
    inline for (.{ false, true }) |far| {
        var cb = CodeBuffer.init(t.allocator);
        defer cb.deinit();
        const a = rv.AssemblerC.init(&cb);
        const top = try a.newLabel();
        const done = try a.newLabel();
        try a.li(.t1, 0);
        try a.bind(top);
        if (far) try a.bFar(.ge, .zero, .a0, done) else try a.b(.ge, .zero, .a0, done); // n <= 0
        try a.add(.t1, .t1, .a0);
        try a.addi(.a0, .a0, -1);
        try a.j(top);
        try a.bind(done);
        try a.mv(.a0, .t1);
        try a.ret();
        const f = try jit(&cb);
        defer f.release();
        for ([_]i64{ 0, 1, 10, 1000, -5 }) |n| try t.expectEqual(if (n > 0) @divExact(n * (n + 1), 2) else 0, f.ptr(Fn3)(n, 0, 0));
    }
}

fn mulAdd(x: i64, y: i64, z: i64) callconv(.c) i64 {
    return x *% y +% z;
}

test "riscv64 exec: prolog/epilog, callee-saved registers and callAbs into Zig" {
    if (!can_run) return error.SkipZigTest;
    var cb = CodeBuffer.init(t.allocator);
    defer cb.deinit();
    const a = rv.Assembler.init(&cb);
    const saved = [_]rv.X{ .s1, .s2 };
    const n = try a.prolog(&saved, 16);
    try a.mv(.s1, .a0); // survives the call
    try a.mv(.s2, .a2);
    try a.sd(.a1, -32 - 8, .s0); // a local slot below the saved area
    try a.callAbs(@intFromPtr(&mulAdd)); // a0 = a0*a1 + a2
    try a.ld(.t2, -32 - 8, .s0);
    try a.add(.a0, .a0, .s1);
    try a.add(.a0, .a0, .s2);
    try a.add(.a0, .a0, .t2);
    try a.epilog(&saved, n);
    const f = try jit(&cb);
    defer f.release();
    // 6*7+5 + 6 + 5 + 7
    try t.expectEqual(@as(i64, 65), f.ptr(Fn3)(6, 7, 5));
    try t.expectEqual(@as(i64, -3 * 4 + 1 - 3 + 1 + 4), f.ptr(Fn3)(-3, 4, 1));
}

test "riscv64 exec: memory, AMOs and lr/sc" {
    if (!can_run) return error.SkipZigTest;
    var cb = CodeBuffer.init(t.allocator);
    defer cb.deinit();
    const a = rv.Assembler.init(&cb);
    // a0 = &mem: mem[1] = mem[0] + 1 (lw/sw), amoadd.d mem+8 += a1 -> returns old value
    try a.lw(.t0, 0, .a0);
    try a.addi(.t0, .t0, 1);
    try a.sw(.t0, 4, .a0);
    try a.addi(.t1, .a0, 8);
    try a.amo(.add, .d, .aqrl, .a0, .a1, .t1);
    try a.ret();
    const f = try jit(&cb);
    defer f.release();
    var mem: [2]i64 align(8) = .{ 41, 100 };
    try t.expectEqual(@as(i64, 100), f.ptr(Fn3)(@bitCast(@intFromPtr(&mem)), 23, 0));
    try t.expectEqual(@as(i64, 123), mem[1]);
    const words: *const [2]i32 = @ptrCast(&mem[0]);
    try t.expectEqual(@as(i32, 42), words[1]);
}

test "riscv64 exec: F/D arithmetic and conversions" {
    if (!can_run) return error.SkipZigTest;
    var cb = CodeBuffer.init(t.allocator);
    defer cb.deinit();
    const a = rv.Assembler.init(&cb);
    // (double(a0) * 0.5 + load f64 [a1]) -> fcvt.l.d rtz; also store the f32 version to [a1+8]
    try a.fcvtFromInt(.l, .d, .f0, .a0, .rne);
    try a.li(.t0, @bitCast(@as(f64, 0.5)));
    try a.fmvFromInt(.d, .f1, .t0);
    try a.farith(.mul, .d, .f0, .f0, .f1, .dyn);
    try a.fload(.d, .f2, 0, .a1);
    try a.farith(.add, .d, .f0, .f0, .f2, .dyn);
    try a.fcvtFmt(.s, .f3, .f0, .dyn);
    try a.fstore(.s, .f3, 8, .a1);
    try a.fcvtToInt(.l, .d, .a0, .f0, .rtz);
    try a.ret();
    const f = try jit(&cb);
    defer f.release();
    var mem: extern struct { d: f64, s: f32, pad: f32 } = .{ .d = 2.75, .s = 0, .pad = 0 };
    const p: i64 = @bitCast(@intFromPtr(&mem));
    try t.expectEqual(@as(i64, 7), f.ptr(Fn3)(9, p, 0)); // 4.5 + 2.75 = 7.25
    try t.expectEqual(@as(f32, 7.25), mem.s);
    try t.expectEqual(@as(i64, -2), f.ptr(Fn3)(-10, p, 0)); // -5 + 2.75 = -2.25
}
