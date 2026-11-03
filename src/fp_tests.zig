//! Scalar float lowering (f32/f64 IR on the x86-64 and AArch64 Compilers): execution on the host
//! against the reference interpreter (`ir.interpret`), bit-exact, including NaN, signed zeros,
//! infinities and subnormals; spill stress (one FP register); values live across calls; and a
//! random-program differential fuzz. Runs natively on aarch64 and on x86-64 (Rosetta / CI).
const std = @import("std");
const builtin = @import("builtin");
const ir = @import("core/ir.zig");
const rc = @import("core/reg_classes.zig");
const CodeBuffer = @import("core/code_buffer.zig").CodeBuffer;
const JitRuntime = @import("core/jit_runtime.zig");
const opts = @import("pyrlyn_jit_options");
const t = std.testing;

const host_compiler = switch (builtin.cpu.arch) {
    .x86_64 => if (opts.x86_64) @import("x86_64/compiler.zig") else void,
    .aarch64 => if (opts.aarch64) @import("aarch64/compiler.zig") else void,
    else => void,
};
const can_run = host_compiler != void;

/// Float types the compilers lower (grows with `reg_classes.support`).
const types = blk: {
    var n: usize = 0;
    var out: [2]ir.Type = undefined;
    if (rc.support.f32) {
        out[n] = .f32;
        n += 1;
    }
    if (rc.support.f64) {
        out[n] = .f64;
        n += 1;
    }
    const final = out;
    break :blk final[0..n];
};

const Mode = enum { normal, one_fp_reg };

/// Compiles `src` for the host and runs it as `i64 f(ptr)`; also runs the interpreter on a copy
/// of `mem` and checks the result and memory bit-exactly (any two NaNs compare equal).
fn checkProgram(src: []const u8, mem: []u64, tp: ir.Type, mode: Mode) !void {
    var f = try ir.parse(t.allocator, src);
    defer f.deinit();
    const want_mem = try t.allocator.dupe(u64, mem);
    defer t.allocator.free(want_mem);
    const want = try ir.interpret(&f, &.{@bitCast(@intFromPtr(want_mem.ptr))}, 1 << 20);

    var cb = CodeBuffer.init(t.allocator);
    defer cb.deinit();
    var o: host_compiler.Options = .{};
    if (mode == .one_fp_reg) o.fp_allocatable = 1;
    _ = try host_compiler.compile(t.allocator, &f, &cb, o);
    const fun = try JitRuntime.add(&cb);
    defer fun.release();
    const got = fun.ptr(*const fn (*u64) callconv(.c) i64)(&mem[0]);
    if (got != want) {
        std.debug.print("result mismatch: got {d} want {d}\n{s}\n", .{ got, want, src });
        return error.TestUnexpectedResult;
    }
    for (mem, want_mem, 0..) |g, w, i| if (!sameBits(tp, g, w)) {
        std.debug.print("mem[{d}] mismatch: got 0x{x} want 0x{x}\n{s}\n", .{ i, g, w, src });
        return error.TestUnexpectedResult;
    };
}

fn sameBits(tp: ir.Type, a: u64, b: u64) bool {
    if (a == b) return true;
    if (tp == .f32) {
        const x: f32 = @bitCast(@as(u32, @truncate(a)));
        const y: f32 = @bitCast(@as(u32, @truncate(b)));
        return std.math.isNan(x) and std.math.isNan(y) and a >> 32 == b >> 32;
    }
    return std.math.isNan(@as(f64, @bitCast(a))) and std.math.isNan(@as(f64, @bitCast(b)));
}

fn specials(comptime T: type) []const T {
    const inf = std.math.inf(T);
    return &[_]T{ 0.0, -0.0, 1.0, -1.5, 3.25, 1e-3, -7e5, inf, -inf, std.math.nan(T), std.math.floatTrueMin(T), std.math.floatMax(T), -std.math.floatMin(T), 0.1 };
}

fn bitsOf(tp: ir.Type, v: f64) u64 {
    return if (tp == .f32) @as(u32, @bitCast(@as(f32, @floatCast(v)))) else @bitCast(v);
}

fn valueBits(tp: ir.Type, i: usize) u64 {
    return if (tp == .f32) @as(u32, @bitCast(specials(f32)[i % specials(f32).len])) else @bitCast(specials(f64)[i % specials(f64).len]);
}

/// `out[k] = a op b` for every op, and a bitmask of every fcmp condition, for all pairs of
/// special values. Memory: slot 0 = a, slot 1 = b, slots 2.. = results.
fn arithSrc(buf: []u8, tp: ir.Type) ![]const u8 {
    const n = @tagName(tp);
    return std.fmt.bufPrint(buf,
        \\func arith 1
        \\e:
        \\  v1 = load.{0s} v0 0
        \\  v2 = load.{0s} v0 8
        \\  v3 = add.{0s} v1 v2
        \\  store.{0s} v0 16 v3
        \\  v4 = sub.{0s} v1 v2
        \\  store.{0s} v0 24 v4
        \\  v5 = mul.{0s} v1 v2
        \\  store.{0s} v0 32 v5
        \\  v6 = div.{0s} v1 v2
        \\  store.{0s} v0 40 v6
        \\  v7 = min.{0s} v1 v2
        \\  store.{0s} v0 48 v7
        \\  v8 = max.{0s} v1 v2
        \\  store.{0s} v0 56 v8
        \\  v9 = copy v8
        \\  store.{0s} v0 64 v9
        \\  v10 = fcmp.{0s} eq v1 v2
        \\  v11 = fcmp.{0s} ne v1 v2
        \\  v12 = fcmp.{0s} lt v1 v2
        \\  v13 = fcmp.{0s} le v1 v2
        \\  v14 = fcmp.{0s} gt v1 v2
        \\  v15 = fcmp.{0s} ge v1 v2
        \\  v16 = const 1
        \\  v17 = shl v11 v16
        \\  v18 = or v10 v17
        \\  v19 = const 2
        \\  v20 = shl v12 v19
        \\  v21 = or v18 v20
        \\  v22 = const 3
        \\  v23 = shl v13 v22
        \\  v24 = or v21 v23
        \\  v25 = const 4
        \\  v26 = shl v14 v25
        \\  v27 = or v24 v26
        \\  v28 = const 5
        \\  v29 = shl v15 v28
        \\  v30 = or v27 v29
        \\  ret v30
        \\
    , .{n});
}

test "fp: arithmetic, min/max and fcmp on special values match the interpreter" {
    if (!can_run) return error.SkipZigTest;
    inline for (types) |tp| for ([_]Mode{ .normal, .one_fp_reg }) |mode| {
        var buf: [2048]u8 = undefined;
        const src = try arithSrc(&buf, tp);
        const ns = if (tp == .f32) specials(f32).len else specials(f64).len;
        for (0..ns) |i| for (0..ns) |j| {
            var mem: [9]u64 = @splat(0);
            mem[0] = valueBits(tp, i);
            mem[1] = valueBits(tp, j);
            try checkProgram(src, &mem, tp, mode);
        };
    };
}

test "fp: constants, loops, branches on fcmp, live across calls" {
    if (!can_run) return error.SkipZigTest;
    const Helper = struct {
        fn clobber(a: i64) callconv(.c) i64 {
            // Touch every caller-saved FP register through a float computation the compiler
            // is free to place anywhere.
            var acc: [8]f64 = undefined;
            for (&acc, 0..) |*x, i| x.* = @floatFromInt(a + @as(i64, @intCast(i)));
            var s: f64 = 0;
            for (acc) |x| s += x * 1.5;
            return @intFromFloat(@mod(s, 1000));
        }
    };
    inline for (types) |tp| for ([_]Mode{ .normal, .one_fp_reg }) |mode| {
        // sum = 0; x = 0.5; loop: sum += x; x *= 1.5; while (x < 100); call clobber; store sum, x
        var buf: [1024]u8 = undefined;
        const n = @tagName(tp);
        const src = try std.fmt.bufPrint(&buf,
            \\func loop 1
            \\e:
            \\  v1 = fconst.{0s} 0
            \\  v2 = fconst.{0s} 0.5
            \\  v3 = fconst.{0s} 1.5
            \\  v4 = fconst.{0s} 100
            \\  v8 = const 0
            \\  v9 = const 1
            \\  v10 = const 0
            \\  br l
            \\l:
            \\  v1 = add.{0s} v1 v2
            \\  v2 = mul.{0s} v2 v3
            \\  v8 = add v8 v9
            \\  v5 = fcmp.{0s} lt v2 v4
            \\  cbr ne v5 v10 l done
            \\done:
            \\  v6 = call {1d} v8
            \\  store.{0s} v0 0 v1
            \\  store.{0s} v0 8 v2
            \\  v7 = add v6 v8
            \\  ret v7
            \\
        , .{ n, @intFromPtr(&Helper.clobber) });
        var mem: [2]u64 = @splat(0);
        try checkProgram(src, &mem, tp, mode);
        // Same loop in Zig: the interpreter check above is not the only oracle.
        const T = if (tp == .f32) f32 else f64;
        var sum: T = 0;
        var xx: T = 0.5;
        while (true) {
            sum += xx;
            xx *= 1.5;
            if (!(xx < 100)) break;
        }
        try t.expectEqual(bitsOf(tp, sum), mem[0]);
        try t.expectEqual(bitsOf(tp, xx), mem[1]);
    };
}

test "fp: memory displacements (negative, unaligned, large)" {
    if (!can_run) return error.SkipZigTest;
    inline for (types) |tp| {
        var buf: [512]u8 = undefined;
        const n = @tagName(tp);
        const src = try std.fmt.bufPrint(&buf,
            \\func m 1
            \\e:
            \\  v1 = const 40000
            \\  v2 = add v0 v1
            \\  v3 = load.{0s} v2 -40000
            \\  v4 = load.{0s} v0 4
            \\  v5 = add.{0s} v3 v4
            \\  store.{0s} v2 -39996 v5
            \\  store.{0s} v0 40000 v5
            \\  v6 = const 7
            \\  ret v6
            \\
        , .{n});
        const mem = try t.allocator.alloc(u64, 5008);
        defer t.allocator.free(mem);
        @memset(mem, 0);
        mem[0] = valueBits(tp, 2) | valueBits(tp, 4) << 32;
        try checkProgram(src, mem, tp, .normal);
    }
}

test "fp: f64 keeps double precision and f32/f64 values coexist" {
    if (!can_run) return error.SkipZigTest;
    // 0.1 + 0.2 in f64 (0x3FD3333333333334) and in f32, in one function; fcmp of each vs 0.3.
    const src =
        \\func p 1
        \\e:
        \\  v1 = fconst.f64 0.1
        \\  v2 = fconst.f64 0.2
        \\  v3 = add.f64 v1 v2
        \\  store.f64 v0 0 v3
        \\  v4 = fconst.f32 0.1
        \\  v5 = fconst.f32 0.2
        \\  v6 = add.f32 v4 v5
        \\  store.f32 v0 8 v6
        \\  v7 = fconst.f64 0.3
        \\  v8 = fcmp.f64 eq v3 v7
        \\  v9 = fconst.f32 0.3
        \\  v10 = fcmp.f32 eq v6 v9
        \\  v11 = const 1
        \\  v12 = shl v10 v11
        \\  v13 = or v8 v12
        \\  ret v13
    ;
    var mem: [2]u64 = @splat(0);
    try checkProgram(src, &mem, .f64, .normal);
    try t.expectEqual(@as(u64, 0x3FD3333333333334), mem[0]);
    try t.expectEqual(@as(u64, @as(u32, @bitCast(@as(f32, 0.1) + @as(f32, 0.2)))), mem[1] & 0xFFFF_FFFF);
    try checkProgram(src, &mem, .f64, .one_fp_reg);
}

test "fp: conversions (fpext, fptrunc, i64 <-> float) on special values" {
    if (!can_run) return error.SkipZigTest;
    // mem[0] = f32 input, mem[1] = f64 input, mem[2] = i64 input; outputs from mem[3].
    const src =
        \\func cvt 1
        \\e:
        \\  v1 = load.f32 v0 0
        \\  v2 = load.f64 v0 8
        \\  v3 = load v0 16
        \\  v4 = cvt.f64.f32 v1
        \\  store.f64 v0 24 v4
        \\  v5 = cvt.f32.f64 v2
        \\  store.f32 v0 32 v5
        \\  v6 = cvt.f32.i64 v3
        \\  store.f32 v0 40 v6
        \\  v7 = cvt.f64.i64 v3
        \\  store.f64 v0 48 v7
        \\  v8 = cvt.i64.f32 v1
        \\  store v0 56 v8
        \\  v9 = cvt.i64.f64 v2
        \\  store v0 64 v9
        \\  v10 = cvt.f32.f64 v4
        \\  v11 = fcmp.f32 eq v10 v1
        \\  v12 = xor v8 v9
        \\  v13 = add v12 v11
        \\  ret v13
    ;
    const f32s = specials(f32);
    const f64s = [_]f64{ 0, -0.0, 1.5, -2.5, 3.4e38, 3.5e38, -1e39, 1e-46, 9.2233720368547748e18, 9.2233720368547758e18, -9.2233720368547758e18, -9.2233720368547779e18, 1e300, std.math.nan(f64), std.math.inf(f64), 2.5, -0.5, 0x1.000001p0, 0x1.0000018p0 };
    const ints = [_]i64{ 0, 1, -1, 16777217, -16777217, 9007199254740993, std.math.maxInt(i64), std.math.minInt(i64), 123456789012345678 };
    const extra_f32 = [_]f32{ 9.2233715e18, 9.223372e18, -9.223372e18, -9.2233731e18, 2.5, -3.5 };
    for ([_]Mode{ .normal, .one_fp_reg }) |mode| {
        for (0..@max(f32s.len + extra_f32.len, f64s.len, ints.len)) |i| {
            var mem: [9]u64 = @splat(0);
            const a: f32 = if (i < f32s.len) f32s[i] else extra_f32[(i - f32s.len) % extra_f32.len];
            mem[0] = @as(u32, @bitCast(a));
            mem[1] = @bitCast(f64s[i % f64s.len]);
            mem[2] = @bitCast(ints[i % ints.len]);
            try checkProgram(src, &mem, .f64, mode);
            // Independent oracle for the f -> i64 rule (x86 cvtt: NaN / out of range -> minInt).
            const want_i: i64 = if (std.math.isNan(a) or a >= 9.223372036854775807e18 or a < -9.223372036854775808e18) std.math.minInt(i64) else @intFromFloat(a);
            try t.expectEqual(want_i, @as(i64, @bitCast(mem[7])));
            try t.expectEqual(@as(u64, @bitCast(@as(f64, a))), mem[3]);
        }
    }
}

test "fp: unsupported float IR is rejected, not miscompiled" {
    if (!can_run) return error.SkipZigTest;
    var f = try ir.parse(t.allocator,
        \\func v 1
        \\e:
        \\  v1 = fconst.f32 1
        \\  v2 = splat.v4f32 v1
        \\  v3 = extract.v4f32 v2 0
        \\  v4 = const 0
        \\  ret v4
    );
    defer f.deinit();
    var cb = CodeBuffer.init(t.allocator);
    defer cb.deinit();
    try t.expectError(error.UnsupportedType, host_compiler.compile(t.allocator, &f, &cb, .{}));
}

/// Random straight-line + branch programs over floats: loads from memory, typed ops, fcmp
/// folded into an i64 accumulator, stores of every value.
fn genProgram(rng: std.Random, tp: ir.Type, out: *std.ArrayList(u8)) !void {
    const n = @tagName(tp);
    const gpa = t.allocator;
    const nin = 4;
    try out.print(gpa, "func fz 1\ne:\n  v1 = const 0\n", .{});
    var next: u32 = 2;
    var fl: [64]u32 = undefined;
    var nf: usize = 0;
    for (0..nin) |i| {
        try out.print(gpa, "  v{d} = load.{s} v0 {d}\n", .{ next, n, 8 * i });
        fl[nf] = next;
        nf += 1;
        next += 1;
    }
    const ops = [_][]const u8{ "add", "sub", "mul", "div", "min", "max" };
    const steps = rng.intRangeAtMost(usize, 4, 40);
    var stores: u32 = 0;
    for (0..steps) |_| {
        const a = fl[rng.uintLessThan(usize, nf)];
        const b = fl[rng.uintLessThan(usize, nf)];
        switch (rng.uintLessThan(u8, 10)) {
            0...5 => {
                try out.print(gpa, "  v{d} = {s}.{s} v{d} v{d}\n", .{ next, ops[rng.uintLessThan(usize, ops.len)], n, a, b });
                if (nf < fl.len) {
                    fl[nf] = next;
                    nf += 1;
                } else fl[rng.uintLessThan(usize, nf)] = next;
                next += 1;
            },
            6 => {
                try out.print(gpa, "  v{d} = fconst.{s} {d}\n", .{ next, n, @as(f64, @floatFromInt(rng.intRangeAtMost(i32, -50, 50))) / 4.0 });
                if (nf < fl.len) {
                    fl[nf] = next;
                    nf += 1;
                }
                next += 1;
            },
            7, 8 => {
                const conds = [_][]const u8{ "eq", "ne", "lt", "le", "gt", "ge" };
                try out.print(gpa, "  v{d} = fcmp.{s} {s} v{d} v{d}\n  v{d} = const 1\n  v{d} = shl v1 v{d}\n  v1 = xor v{d} v{d}\n", .{ next, n, conds[rng.uintLessThan(usize, conds.len)], a, b, next + 1, next + 2, next + 1, next + 2, next });
                next += 3;
            },
            else => if (stores < 12) {
                try out.print(gpa, "  store.{s} v0 {d} v{d}\n", .{ n, 8 * (nin + stores), a });
                stores += 1;
            },
        }
    }
    for (0..@min(nf, 12 - stores)) |i| try out.print(gpa, "  store.{s} v0 {d} v{d}\n", .{ n, 8 * (nin + stores + i), fl[i] });
    try out.print(gpa, "  ret v1\n", .{});
}

test "fp: random programs match the interpreter (fuzz)" {
    if (!can_run) return error.SkipZigTest;
    var prng = std.Random.DefaultPrng.init(0x5EED_F10A7);
    const rng = prng.random();
    inline for (types) |tp| for (0..300) |iter| {
        var src: std.ArrayList(u8) = .empty;
        defer src.deinit(t.allocator);
        try genProgram(rng, tp, &src);
        var mem: [16]u64 = @splat(0);
        for (mem[0..4]) |*m| m.* = if (rng.boolean()) valueBits(tp, rng.uintLessThan(usize, 14)) else bitsOf(tp, @as(f64, @floatFromInt(rng.intRangeAtMost(i32, -1000, 1000))) / 8.0);
        try checkProgram(src.items, &mem, tp, if (iter % 3 == 0) .one_fp_reg else .normal);
    };
}

test "fp: both compilers lower float IR on any host (bytes only)" {
    inline for (types) |tp| {
        var buf: [2048]u8 = undefined;
        var f = try ir.parse(t.allocator, try arithSrc(&buf, tp));
        defer f.deinit();
        const cs = .{ if (opts.x86_64) @import("x86_64/compiler.zig") else void, if (opts.aarch64) @import("aarch64/compiler.zig") else void };
        inline for (cs) |C| if (C != void) {
            var cb = CodeBuffer.init(t.allocator);
            defer cb.deinit();
            _ = try C.compile(t.allocator, &f, &cb, .{});
            try t.expect(cb.code().len > 64);
        };
    }
}
