//! x86-64 IR Compiler code quality and SSE vector lowering: execution against `ir.interpret`
//! (Rosetta on Apple Silicon) and instruction-shape checks (X64 work package).
//!
//! Every program takes a memory pointer in v0 (its results are stored there) and is run under
//! several `Options` (`variants`): default, spill stress with one callee-saved register, only
//! caller-saved registers (call-crossing values saved or spilled, fixed-register clobbers
//! everywhere), no live-range splitting, the graph allocator, and the Builder path with the
//! SSE2 vector sequences. Results and memory compare bit-exactly; two NaNs compare equal
//! (fp_tests.zig style), per f64 word or per f32 half.
const std = @import("std");
const builtin = @import("builtin");
const ir = @import("core/ir.zig");
const ra = @import("core/regalloc.zig");
const x = @import("x86_64/assembler.zig");
const cx = @import("x86_64/compiler.zig");
const cpu = @import("x86_64/cpu.zig");
const CodeBuffer = @import("core/code_buffer.zig").CodeBuffer;
const JitRuntime = @import("core/jit_runtime.zig");
const t = std.testing;

const can_run = builtin.cpu.arch == .x86_64;
const sysv = x.CallConv.host.shadow == 0;

fn bit(r: x.Gp) ra.RegMask {
    return @as(ra.RegMask, 1) << @backingInt(r);
}

const variants = [_]cx.Options{
    .{},
    .{ .allocatable = bit(.rbx), .fp_allocatable = 1 },
    .{ .allocatable = bit(.rcx) | bit(.rdx), .fp_allocatable = 0b11 },
    .{ .allocatable = bit(.rax) | bit(.rsi) | bit(.r12), .split = false, .split_at_calls = false },
    .{ .algo = .graph },
    .{ .shorten_branches = false, .sse41 = false },
};

/// i64 edge values: 0, ±1, extremes, imm32 borders, 32-bit lane borders, shift counts >= 64.
const edges = [_]i64{ 0, 1, -1, std.math.minInt(i64), std.math.maxInt(i64), 0x7fffffff, 0x80000000, -0x80000000, -0x80000001, 0xffffffff, 0x100000000, 63, 64, 65, 127, -64, 3, 0x123456789 };

fn sameWord(a: u64, b: u64) bool {
    if (a == b) return true;
    if (std.math.isNan(@as(f64, @bitCast(a))) and std.math.isNan(@as(f64, @bitCast(b)))) return true;
    for ([_]u6{ 0, 32 }) |sh| {
        const p: u32 = @truncate(a >> sh);
        const q: u32 = @truncate(b >> sh);
        if (p != q and !(std.math.isNan(@as(f32, @bitCast(p))) and std.math.isNan(@as(f32, @bitCast(q))))) return false;
    }
    return true;
}

const Fn8 = *const fn (i64, i64, i64, i64, i64, i64, i64, i64) callconv(.c) i64;

/// Runs `f` (params: v0 = memory, then `in`) under every option set for every input against
/// the interpreter.
fn check(f: *const ir.Func, inputs: []const []const i64, mem_init: []const u64, opts: []const cx.Options) !void {
    const want_mem = try t.allocator.alloc(u64, mem_init.len);
    defer t.allocator.free(want_mem);
    const got_mem = try t.allocator.alloc(u64, mem_init.len);
    defer t.allocator.free(got_mem);
    for (opts, 0..) |o, oi| {
        var cb = CodeBuffer.init(t.allocator);
        defer cb.deinit();
        _ = try cx.compile(t.allocator, f, &cb, o);
        const fun = try JitRuntime.add(&cb);
        defer fun.release();
        for (inputs) |in| {
            @memcpy(want_mem, mem_init);
            @memcpy(got_mem, mem_init);
            var a: [8]i64 = @splat(0);
            a[0] = @bitCast(@intFromPtr(want_mem.ptr));
            for (in, 1..) |v, i| a[i] = v;
            const want = try ir.interpret(f, a[0 .. 1 + in.len], 1 << 20);
            a[0] = @bitCast(@intFromPtr(got_mem.ptr));
            const got = fun.ptr(Fn8)(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]);
            var ok = got == want;
            for (got_mem, want_mem) |g, w| ok = ok and sameWord(g, w);
            if (!ok) {
                std.debug.print("mismatch (options #{d}, input {any}): got {d} want {d}\n", .{ oi, in, got, want });
                for (got_mem, want_mem, 0..) |g, w, i| if (!sameWord(g, w)) std.debug.print("  mem[{d}] got 0x{x} want 0x{x}\n", .{ i, g, w });
                var out: std.ArrayList(u8) = .empty;
                defer out.deinit(t.allocator);
                try f.print(t.allocator, &out);
                std.debug.print("{s}\n", .{out.items});
                return error.TestUnexpectedResult;
            }
        }
    }
}

fn parse(src: []const u8) !ir.Func {
    return ir.parse(t.allocator, src);
}

fn divTraps(op: ir.Arith, a: i64, b: i64) bool {
    return (op == .sdiv or op == .srem) and (b == 0 or (a == std.math.minInt(i64) and b == -1));
}

/// Writes garbage into every caller-saved register (a callee that uses all of them).
fn clobberCallerSaved() void {
    if (comptime builtin.cpu.arch == .x86_64 and sysv) asm volatile (
        \\ movq $0x5a5a5a5a5a, %%rcx
        \\ movq %%rcx, %%rdx
        \\ movq %%rcx, %%rsi
        \\ movq %%rcx, %%rdi
        \\ movq %%rcx, %%r8
        \\ movq %%rcx, %%r9
        \\ movq %%rcx, %%r10
        \\ movq %%rcx, %%r11
        \\ pcmpeqd %%xmm0, %%xmm0
        \\ pcmpeqd %%xmm1, %%xmm1
        \\ pcmpeqd %%xmm2, %%xmm2
        \\ pcmpeqd %%xmm3, %%xmm3
        \\ pcmpeqd %%xmm4, %%xmm4
        \\ pcmpeqd %%xmm5, %%xmm5
        \\ pcmpeqd %%xmm6, %%xmm6
        \\ pcmpeqd %%xmm7, %%xmm7
        \\ pcmpeqd %%xmm8, %%xmm8
        \\ pcmpeqd %%xmm9, %%xmm9
        \\ pcmpeqd %%xmm10, %%xmm10
        \\ pcmpeqd %%xmm11, %%xmm11
        \\ pcmpeqd %%xmm12, %%xmm12
        \\ pcmpeqd %%xmm13, %%xmm13
        \\ pcmpeqd %%xmm14, %%xmm14
        \\ pcmpeqd %%xmm15, %%xmm15
        ::: .{ .rcx = true, .rdx = true, .rsi = true, .rdi = true, .r8 = true, .r9 = true, .r10 = true, .r11 = true, .xmm0 = true, .xmm1 = true, .xmm2 = true, .xmm3 = true, .xmm4 = true, .xmm5 = true, .xmm6 = true, .xmm7 = true, .xmm8 = true, .xmm9 = true, .xmm10 = true, .xmm11 = true, .xmm12 = true, .xmm13 = true, .xmm14 = true, .xmm15 = true });
}

fn mix8(a: i64, b: i64, c: i64, d: i64, e: i64, f: i64, g: i64, h: i64) callconv(.c) i64 {
    clobberCallerSaved();
    return (a *% 3) +% (b ^ 0x55) -% (c *% 7) +% (d *% 8) +% (e ^ -2) +% (f *% 11) -% g +% (h *% 13);
}

/// Callees by argument count (a call passes exactly the callee's parameters).
const Mix = struct {
    fn m0() callconv(.c) i64 {
        return mix8(0, 0, 0, 0, 0, 0, 0, 0);
    }
    fn m1(a: i64) callconv(.c) i64 {
        return mix8(a, 0, 0, 0, 0, 0, 0, 0);
    }
    fn m2(a: i64, b: i64) callconv(.c) i64 {
        return mix8(a, b, 0, 0, 0, 0, 0, 0);
    }
    fn m3(a: i64, b: i64, c: i64) callconv(.c) i64 {
        return mix8(a, b, c, 0, 0, 0, 0, 0);
    }
    fn m6(a: i64, b: i64, c: i64, d: i64, e: i64, f: i64) callconv(.c) i64 {
        return mix8(a, b, c, d, e, f, 0, 0);
    }
    fn m7(a: i64, b: i64, c: i64, d: i64, e: i64, f: i64, g: i64) callconv(.c) i64 {
        return mix8(a, b, c, d, e, f, g, 0);
    }
    const counts = [_]u32{ 0, 1, 2, 3, 6, 7, 8 };
    fn addr(n: u32) usize {
        return switch (n) {
            0 => @intFromPtr(&m0),
            1 => @intFromPtr(&m1),
            2 => @intFromPtr(&m2),
            3 => @intFromPtr(&m3),
            6 => @intFromPtr(&m6),
            7 => @intFromPtr(&m7),
            else => @intFromPtr(&mix8),
        };
    }
};

// ---- integer edge values --------------------------------------------------------------

const int_ops = [_]ir.Arith{ .add, .sub, .mul, .@"and", .@"or", .xor, .shl, .shr, .sar, .sdiv, .srem };

/// Every operand placement for `op` with the operand `K` either a constant (block-local, from
/// block 0 in a later block, on the left of a non-commutative op, in compares and constant
/// stores) or a loaded value (registers and spilled memory operands).
fn intProgram(buf: *std.ArrayList(u8), op: ir.Arith, k: i64, known: bool) !void {
    buf.clearRetainingCapacity();
    const o = @tagName(op);
    try buf.print(t.allocator,
        \\func f 3
        \\e:
        \\  {s}
        \\  v4 = {s} v1 v3
        \\  store v0 0 v4
        \\  v5 = {s} v3 v1
        \\  store v0 8 v5
        \\  v6 = copy v1
        \\  v6 = {s} v6 v3
        \\  store v0 16 v6
        \\  v12 = const {d}
        \\  store v0 48 v12
        \\  br n
        \\n:
        \\  v7 = {s} v2 v3
        \\  store v0 24 v7
        \\  v8 = {s} v3 v2
        \\  store v0 32 v8
        \\  v9 = {s} v1 v2
        \\  v13 = copy v1
        \\  v13 = {s} v2 v13
        \\  store v0 64 v13
        \\  v2 = {s} v2 v1
        \\  store v0 40 v9
        \\  cbr lt v1 v3 x y
        \\x:
        \\  v10 = sub v2 v3
        \\  ret v10
        \\y:
        \\  cbr ne v3 v2 z w
        \\z:
        \\  ret v2
        \\w:
        \\  v11 = const 0
        \\  ret v11
        \\
    , .{ "", o, o, o, k, o, o, o, o, o });
    // The first line defines v3 (patched in: a const or a load of mem[7] = K).
    const head = "func f 3\ne:\n  ";
    var line: [64]u8 = undefined;
    const def = if (known)
        try std.fmt.bufPrint(&line, "v3 = const {d}", .{k})
    else
        try std.fmt.bufPrint(&line, "v3 = load v0 56", .{});
    try buf.insertSlice(t.allocator, head.len, def);
}

test "x64 codegen: integer ops with edge values, constants and every operand placement" {
    if (!can_run) return error.SkipZigTest;
    var buf: std.ArrayList(u8) = .empty;
    defer buf.deinit(t.allocator);
    const ys = [_]i64{ 0, -1, std.math.minInt(i64), 0x7fffffff, 64, 5 };
    var inputs: std.ArrayList([]const i64) = .empty;
    defer inputs.deinit(t.allocator);
    var pairs: [edges.len * ys.len][2]i64 = undefined;
    for (int_ops) |op| for (edges) |k| for ([_]bool{ true, false }) |known| {
        try intProgram(&buf, op, k, known);
        var f = try parse(buf.items);
        defer f.deinit();
        inputs.clearRetainingCapacity();
        var n: usize = 0;
        for (edges) |a| for (ys) |b| {
            // Divisions that trap in generated code (the interpreter returns 0) are skipped.
            if (divTraps(op, a, k) or divTraps(op, k, a) or divTraps(op, b, k) or divTraps(op, k, b) or divTraps(op, a, b) or divTraps(op, b, a)) continue;
            pairs[n] = .{ a, b };
            try inputs.append(t.allocator, &pairs[n]);
            n += 1;
        };
        var mem: [9]u64 = @splat(0);
        mem[7] = @bitCast(k);
        try check(&f, inputs.items, &mem, if (known) &variants else variants[0..2]);
    };
}

test "x64 codegen: constant-only arithmetic folds like the interpreter" {
    if (!can_run) return error.SkipZigTest;
    var buf: std.ArrayList(u8) = .empty;
    defer buf.deinit(t.allocator);
    for (int_ops) |op| for (edges) |a| {
        buf.clearRetainingCapacity();
        try buf.print(t.allocator, "func f 1\ne:\n  v1 = const {d}\n", .{a});
        var off: u32 = 0;
        for (edges) |b| {
            if (divTraps(op, a, b)) continue;
            try buf.print(t.allocator, "  v2 = const {d}\n  v3 = {s} v1 v2\n  store v0 {d} v3\n", .{ b, @tagName(op), off });
            off += 8;
        }
        try buf.print(t.allocator, "  ret v3\n", .{});
        var f = try parse(buf.items);
        defer f.deinit();
        var mem: [edges.len]u64 = @splat(0);
        try check(&f, &.{&.{}}, &mem, variants[0..1]);
    };
}

test "x64 codegen: stack parameters, parallel moves and calls with stack arguments" {
    if (!can_run) return error.SkipZigTest;
    // Parameters swapped and rotated into each other's argument registers (move cycles).
    var buf: std.ArrayList(u8) = .empty;
    defer buf.deinit(t.allocator);
    try buf.print(t.allocator,
        \\func f 8
        \\e:
        \\  v8 = call {d} v2 v1 v3 v4 v5 v6 v7 v1
        \\  v9 = call {d} v7 v6 v5 v4 v3 v2 v1 v8
        \\  v10 = sub v9 v7
        \\  v11 = call {d} v10 v1 v2 v3 v4 v5 v6 v7
        \\  store v0 0 v11
        \\  v12 = add v1 v2
        \\  v12 = add v12 v6
        \\  v12 = add v12 v7
        \\  ret v12
        \\
    , .{ @intFromPtr(&mix8), @intFromPtr(&mix8), @intFromPtr(&mix8) });
    var f = try parse(buf.items);
    defer f.deinit();
    var mem: [1]u64 = .{0};
    try check(&f, &.{ &.{ 1, 2, 3, 4, 5, 6, 7 }, &.{ -1, std.math.minInt(i64), 0, 9, 0x7fffffff, -5, 1 << 40 } }, &mem, &variants);
}

test "x64 codegen: division and shifts keep values in rax, rdx and rcx alive" {
    if (!can_run) return error.SkipZigTest;
    // Many values live across idiv (rax:rdx) and variable shifts (rcx) with every register
    // allowed, and with only rcx/rdx allowed (all of them conflict).
    var f = try parse(
        \\func f 3
        \\e:
        \\  v3 = add v1 v2
        \\  v4 = sub v1 v2
        \\  v5 = xor v1 v2
        \\  v6 = const 255
        \\  v6 = and v2 v6
        \\  v7 = const 1
        \\  v6 = or v6 v7
        \\  v8 = sdiv v1 v6
        \\  v9 = srem v3 v6
        \\  v10 = shl v4 v2
        \\  v11 = sar v5 v1
        \\  v12 = shr v3 v4
        \\  v13 = sdiv v6 v6
        \\  v14 = shl v2 v2
        \\  store v0 0 v8
        \\  store v0 8 v9
        \\  store v0 16 v10
        \\  store v0 24 v11
        \\  store v0 32 v12
        \\  store v0 40 v13
        \\  store v0 48 v14
        \\  v15 = add v3 v4
        \\  v15 = add v15 v5
        \\  v15 = add v15 v6
        \\  ret v15
    );
    defer f.deinit();
    var mem: [7]u64 = @splat(0);
    var inputs: [edges.len * 4][2]i64 = undefined;
    var list: [inputs.len][]const i64 = undefined;
    for (edges, 0..) |a, i| for ([_]i64{ 0, -1, 65, std.math.minInt(i64) }, 0..) |b, j| {
        inputs[i * 4 + j] = .{ a, b };
        list[i * 4 + j] = &inputs[i * 4 + j];
    };
    try check(&f, &list, &mem, &variants);
}

// ---- floats ------------------------------------------------------------------------------

fn floatSpecials(comptime T: type) [14]T {
    const inf = std.math.inf(T);
    return .{ 0.0, -0.0, 1.0, -1.5, 3.25, 1e-3, -7e5, inf, -inf, std.math.nan(T), std.math.floatTrueMin(T), std.math.floatMax(T), -std.math.floatMin(T), 1e30 };
}

test "x64 codegen: scalar floats - two-address forms, memory operands, fcmp, cvt" {
    if (!can_run) return error.SkipZigTest;
    var buf: std.ArrayList(u8) = .empty;
    defer buf.deinit(t.allocator);
    inline for (.{ f32, f64 }) |T| {
        const tn = if (T == f32) "f32" else "f64";
        const other = if (T == f32) "f64" else "f32";
        buf.clearRetainingCapacity();
        try buf.print(t.allocator, "func f 1\ne:\n  v1 = load.{s} v0 0\n  v2 = load.{s} v0 8\n", .{ tn, tn });
        var off: u32 = 16;
        for ([_][]const u8{ "add", "sub", "mul", "div", "min", "max" }) |op| {
            // dst new; dst == a; dst == b; a == b.
            try buf.print(t.allocator, "  v3 = {s}.{s} v1 v2\n  store.{s} v0 {d} v3\n", .{ op, tn, tn, off });
            try buf.print(t.allocator, "  v4 = copy v1\n  v4 = {s}.{s} v4 v2\n  store.{s} v0 {d} v4\n", .{ op, tn, tn, off + 8 });
            try buf.print(t.allocator, "  v5 = copy v2\n  v5 = {s}.{s} v1 v5\n  store.{s} v0 {d} v5\n", .{ op, tn, tn, off + 16 });
            try buf.print(t.allocator, "  v6 = {s}.{s} v2 v2\n  store.{s} v0 {d} v6\n", .{ op, tn, tn, off + 24 });
            off += 32;
        }
        for ([_][]const u8{ "eq", "ne", "lt", "le", "gt", "ge" }) |c| {
            try buf.print(t.allocator, "  v7 = fcmp.{s} {s} v1 v2\n  store v0 {d} v7\n", .{ tn, c, off });
            off += 8;
        }
        try buf.print(t.allocator,
            \\  v8 = cvt.i64.{s} v1
            \\  store v0 {d} v8
            \\  v9 = cvt.{s}.i64 v8
            \\  store.{s} v0 {d} v9
            \\  v10 = cvt.{s}.{s} v2
            \\  store.{s} v0 {d} v10
            \\  v11 = cvt.{s}.{s} v10
            \\  store.{s} v0 {d} v11
            \\  v12 = fconst.{s} 0
            \\  v12 = sub.{s} v12 v1
            \\  store.{s} v0 {d} v12
            \\  v13 = fconst.{s} 2.5
            \\  v13 = mul.{s} v13 v2
            \\  store.{s} v0 {d} v13
            \\  ret v8
            \\
        , .{ tn, off, tn, tn, off + 8, other, tn, other, off + 16, tn, other, tn, off + 24, tn, tn, tn, off + 32, tn, tn, tn, off + 40 });
        off += 48;
        var f = try parse(buf.items);
        defer f.deinit();
        const sp = floatSpecials(T);
        const U = if (T == f32) u32 else u64;
        const mem = try t.allocator.alloc(u64, off / 8);
        defer t.allocator.free(mem);
        @memset(mem, 0);
        var opts_list = variants;
        opts_list[2].fp_allocatable = 1; // memory operands everywhere
        for (sp) |a| for (sp) |b| {
            mem[0] = @as(U, @bitCast(a));
            mem[1] = @as(U, @bitCast(b));
            try check(&f, &.{&.{}}, mem, &opts_list);
        };
    }
}

// ---- vectors -----------------------------------------------------------------------------

const vec_types = [_]ir.Type{ .v4i32, .v2i64, .v4f32, .v2f64 };

fn laneWords(rng: std.Random, tp: ir.Type, out: []u64) void {
    const i32s = [_]i32{ 0, 1, -1, std.math.maxInt(i32), std.math.minInt(i32), 0x10000, 0xffff, 46341, -7, 12345 };
    const f32s = floatSpecials(f32);
    const f64s = floatSpecials(f64);
    for (out) |*w| w.* = switch (tp) {
        .v4i32 => @as(u64, @as(u32, @bitCast(i32s[rng.uintLessThan(usize, i32s.len)]))) | @as(u64, @as(u32, @bitCast(i32s[rng.uintLessThan(usize, i32s.len)]))) << 32,
        .v2i64 => @bitCast(edges[rng.uintLessThan(usize, edges.len)]),
        .v4f32 => @as(u64, @as(u32, @bitCast(f32s[rng.uintLessThan(usize, f32s.len)]))) | @as(u64, @as(u32, @bitCast(f32s[rng.uintLessThan(usize, f32s.len)]))) << 32,
        else => @bitCast(f64s[rng.uintLessThan(usize, f64s.len)]),
    };
}

fn vecOpts() [8]cx.Options {
    var o: [8]cx.Options = undefined;
    @memcpy(o[0..variants.len], &variants);
    const has41 = if (cpu.host()) |h| h.contains(.SSE4_1) else false;
    o[variants.len] = .{ .sse41 = false };
    o[variants.len + 1] = .{ .sse41 = has41, .fp_allocatable = 1 };
    return o;
}

test "x64 codegen: SSE vectors - ops, operand placements, splat, extract, lane wrap" {
    if (!can_run) return error.SkipZigTest;
    var buf: std.ArrayList(u8) = .empty;
    defer buf.deinit(t.allocator);
    var prng = std.Random.DefaultPrng.init(0x5eed);
    const opts = vecOpts();
    for (vec_types) |tp| {
        const tn = @tagName(tp);
        const ops: []const []const u8 = if (tp.isFloat()) &.{ "add", "sub", "mul", "div", "min", "max" } else &.{ "add", "sub", "mul", "and", "or", "xor" };
        for (ops) |op| {
            buf.clearRetainingCapacity();
            try buf.print(t.allocator,
                \\func f 1
                \\e:
                \\  v1 = load.{s} v0 0
                \\  v2 = load.{s} v0 16
                \\  v3 = {s}.{s} v1 v2
                \\  store.{s} v0 32 v3
                \\  v4 = copy v2
                \\  v4 = {s}.{s} v1 v4
                \\  store.{s} v0 48 v4
                \\  v5 = copy v1
                \\  v5 = {s}.{s} v5 v2
                \\  store.{s} v0 64 v5
                \\  v6 = {s}.{s} v2 v2
                \\  store.{s} v0 80 v6
                \\
            , .{ tn, tn, op, tn, tn, op, tn, tn, op, tn, tn, op, tn, tn });
            const lane = @tagName(tp.lane());
            const lsz: u32 = if (tp == .v4f32) 4 else 8;
            for (0..tp.lanes()) |l| try buf.print(t.allocator, "  v7 = extract.{s} v3 {d}\n  store.{s} v0 {d} v7\n", .{ tn, l, lane, 96 + lsz * l });
            try buf.print(t.allocator, "  v8 = splat.{s} v7\n  store.{s} v0 128 v8\n", .{ tn, tn });
            if (!tp.isFloat()) {
                // splat truncates a known i64 (v4i32) / of a loaded i64.
                try buf.print(t.allocator, "  v9 = const 6442450945\n  v10 = splat.{s} v9\n  store.{s} v0 144 v10\n  v11 = load v0 8\n  v12 = splat.{s} v11\n  store.{s} v0 160 v12\n  v13 = extract.{s} v1 1\n  ret v13\n", .{ tn, tn, tn, tn, tn });
            } else {
                try buf.print(t.allocator, "  v9 = load.{s} v0 4\n  v10 = splat.{s} v9\n  store.{s} v0 144 v10\n  v11 = const 0\n  ret v11\n", .{ lane, tn, tn });
            }
            var f = try parse(buf.items);
            defer f.deinit();
            var mem: [22]u64 = @splat(0);
            for (0..12) |_| {
                laneWords(prng.random(), tp, mem[0..4]);
                try check(&f, &.{&.{}}, &mem, &opts);
            }
        }
    }
}

test "x64 codegen: vectors and floats live across calls (spilled) and undefined vectors" {
    if (!can_run) return error.SkipZigTest;
    var buf: std.ArrayList(u8) = .empty;
    defer buf.deinit(t.allocator);
    const a = @intFromPtr(&mix8);
    try buf.print(t.allocator,
        \\func f 2
        \\e:
        \\  v2 = load.v4i32 v0 0
        \\  v3 = load.v2f64 v0 16
        \\  v4 = load.f64 v0 32
        \\  v5 = load.v2i64 v0 0
        \\  v6 = load.v4f32 v0 16
        \\  v7 = const 3
        \\  cbr eq v1 v7 skip go
        \\go:
        \\  v20 = splat.v4i32 v1
        \\  br skip
        \\skip:
        \\  v8 = call {d} v7 v1 v7 v1 v7 v1 v7 v1
        \\  v9 = mul.v4i32 v2 v2
        \\  v10 = mul.v2f64 v3 v3
        \\  v11 = add.f64 v4 v4
        \\  v12 = mul.v2i64 v5 v5
        \\  v13 = min.v4f32 v6 v6
        \\  v14 = call {d} v8 v1 v7 v1 v7 v1 v7 v8
        \\  v9 = add.v4i32 v9 v2
        \\  v20 = add.v4i32 v20 v9
        \\  store.v4i32 v0 48 v20
        \\  store.v2f64 v0 64 v10
        \\  store.f64 v0 80 v11
        \\  store.v2i64 v0 88 v12
        \\  store.v4f32 v0 104 v13
        \\  v15 = extract.v2i64 v12 1
        \\  v15 = add v15 v14
        \\  ret v15
        \\
    , .{ a, a });
    var f = try parse(buf.items);
    defer f.deinit();
    var prng = std.Random.DefaultPrng.init(7);
    var mem: [16]u64 = @splat(0);
    const opts = vecOpts();
    for (0..8) |_| {
        laneWords(prng.random(), .v4i32, mem[0..2]);
        laneWords(prng.random(), .v2f64, mem[2..5]);
        try check(&f, &.{ &.{3}, &.{-9} }, &mem, &opts);
    }
}

// ---- random differential fuzz ---------------------------------------------------------------

/// Random programs over i64, f64, f32 and vector vregs: constants (edge values), memory,
/// calls with up to 8 arguments, safe divisions, variable shifts, forward branches and bounded
/// back edges.
const Fuzz = struct {
    rng: std.Random,
    out: *std.ArrayList(u8),
    nblocks: u32,

    // vreg pools (fixed types)
    const ints = .{ 1, 14 }; // v1..v14 (v1, v2 parameters; v13, v14 may be read undefined)
    const f64s = .{ 15, 18 };
    const v4i = .{ 19, 21 };
    const v2l = .{ 22, 23 };
    const v2d = .{ 24, 25 };
    const v4f = 26;
    const f32s = .{ 32, 33 };

    fn pick(z: *Fuzz, comptime r: anytype) u32 {
        return z.rng.intRangeAtMost(u32, r[0], r[1]);
    }
    fn line(z: *Fuzz, comptime fmt: []const u8, args: anytype) !void {
        try z.out.print(t.allocator, "  " ++ fmt ++ "\n", args);
    }
    fn imm(z: *Fuzz) i64 {
        return if (z.rng.boolean()) edges[z.rng.uintLessThan(usize, edges.len)] else z.rng.intRangeAtMost(i64, -300, 300);
    }

    fn op(z: *Fuzz) !void {
        const i = z.pick(ints);
        const a = z.pick(ints);
        const b = z.pick(ints);
        switch (z.rng.uintLessThan(u32, 22)) {
            0, 1 => try z.line("v{d} = const {d}", .{ i, z.imm() }),
            2...6 => {
                const ops = [_][]const u8{ "add", "sub", "mul", "and", "or", "xor", "shl", "shr", "sar" };
                try z.line("v{d} = {s} v{d} v{d}", .{ i, ops[z.rng.uintLessThan(usize, ops.len)], a, b });
            },
            7 => {
                try z.line("v30 = const 255", .{});
                try z.line("v30 = and v{d} v30", .{b});
                try z.line("v31 = const 1", .{});
                try z.line("v30 = or v30 v31", .{});
                try z.line("v{d} = {s} v{d} v30", .{ i, if (z.rng.boolean()) "sdiv" else "srem", a });
            },
            8 => try z.line("v{d} = copy v{d}", .{ i, a }),
            9 => try z.line("v{d} = load v0 {d}", .{ i, 8 * z.rng.uintLessThan(u32, 16) }),
            10 => try z.line("store v0 {d} v{d}", .{ 8 * z.rng.uintLessThan(u32, 16), a }),
            11 => {
                const n = Mix.counts[z.rng.uintLessThan(usize, Mix.counts.len)];
                try z.out.print(t.allocator, "  v{d} = call {d}", .{ i, Mix.addr(n) });
                for (0..n) |_| try z.out.print(t.allocator, " v{d}", .{z.pick(ints)});
                try z.out.append(t.allocator, '\n');
            },
            12 => try z.line("v{d} = cvt.f64.i64 v{d}", .{ z.pick(f64s), a }),
            13 => {
                const ops = [_][]const u8{ "add", "sub", "mul", "div", "min", "max" };
                try z.line("v{d} = {s}.f64 v{d} v{d}", .{ z.pick(f64s), ops[z.rng.uintLessThan(usize, ops.len)], z.pick(f64s), z.pick(f64s) });
            },
            14 => {
                const cs = [_][]const u8{ "eq", "ne", "lt", "le", "gt", "ge" };
                try z.line("v{d} = fcmp.f64 {s} v{d} v{d}", .{ i, cs[z.rng.uintLessThan(usize, cs.len)], z.pick(f64s), z.pick(f64s) });
            },
            15 => try z.line("v{d} = cvt.i64.f64 v{d}", .{ i, z.pick(f64s) }),
            16 => {
                const v = z.pick(v4i);
                try z.line("v{d} = splat.v4i32 v{d}", .{ v, a });
                const ops = [_][]const u8{ "add", "sub", "mul", "and", "or", "xor" };
                try z.line("v{d} = {s}.v4i32 v{d} v{d}", .{ z.pick(v4i), ops[z.rng.uintLessThan(usize, ops.len)], z.pick(v4i), v });
            },
            17 => try z.line("v{d} = extract.v4i32 v{d} {d}", .{ i, z.pick(v4i), z.rng.uintLessThan(u32, 4) }),
            18 => {
                const ops = [_][]const u8{ "add", "sub", "mul", "and", "xor" };
                try z.line("v{d} = splat.v2i64 v{d}", .{ z.pick(v2l), a });
                try z.line("v{d} = {s}.v2i64 v{d} v{d}", .{ z.pick(v2l), ops[z.rng.uintLessThan(usize, ops.len)], z.pick(v2l), z.pick(v2l) });
                try z.line("v{d} = extract.v2i64 v{d} {d}", .{ i, z.pick(v2l), z.rng.uintLessThan(u32, 2) });
            },
            19 => {
                const ops = [_][]const u8{ "add", "mul", "min", "max", "div" };
                try z.line("v{d} = splat.v2f64 v{d}", .{ z.pick(v2d), z.pick(f64s) });
                try z.line("v{d} = {s}.v2f64 v{d} v{d}", .{ z.pick(v2d), ops[z.rng.uintLessThan(usize, ops.len)], z.pick(v2d), z.pick(v2d) });
                try z.line("v{d} = extract.v2f64 v{d} {d}", .{ z.pick(f64s), z.pick(v2d), z.rng.uintLessThan(u32, 2) });
            },
            20 => {
                const s = z.pick(f32s);
                try z.line("v{d} = cvt.f32.f64 v{d}", .{ s, z.pick(f64s) });
                try z.line("v{d} = splat.v4f32 v{d}", .{ v4f, s });
                try z.line("v{d} = mul.v4f32 v{d} v{d}", .{ v4f, v4f, v4f });
                try z.line("v{d} = extract.v4f32 v{d} {d}", .{ z.pick(f32s), v4f, z.rng.uintLessThan(u32, 4) });
                try z.line("v{d} = cvt.f64.f32 v{d}", .{ z.pick(f64s), z.pick(f32s) });
            },
            else => {
                const d = 16 * z.rng.uintLessThan(u32, 7);
                switch (z.rng.uintLessThan(u32, 4)) {
                    0 => try z.line("store.v4i32 v0 {d} v{d}", .{ d, z.pick(v4i) }),
                    1 => try z.line("v{d} = load.v2i64 v0 {d}", .{ z.pick(v2l), d }),
                    2 => try z.line("store.v2f64 v0 {d} v{d}", .{ d, z.pick(v2d) }),
                    else => try z.line("store.f64 v0 {d} v{d}", .{ d, z.pick(f64s) }),
                }
            },
        }
    }

    fn program(z: *Fuzz) !void {
        z.out.clearRetainingCapacity();
        try z.out.print(t.allocator, "func f 3\nb0:\n", .{});
        for (3..13) |v| if (z.rng.boolean()) try z.line("v{d} = const {d}", .{ v, z.imm() }) else try z.line("v{d} = load v0 {d}", .{ v, 8 * z.rng.uintLessThan(u32, 16) });
        for (15..19) |v| try z.line("v{d} = load.f64 v0 {d}", .{ v, 8 * z.rng.uintLessThan(u32, 16) });
        for (32..34) |v| try z.line("v{d} = load.f32 v0 {d}", .{ v, 4 * z.rng.uintLessThan(u32, 32) });
        for (19..27) |v| {
            const tn = if (v < 22) "v4i32" else if (v < 24) "v2i64" else if (v < 26) "v2f64" else "v4f32";
            try z.line("v{d} = load.{s} v0 {d}", .{ v, tn, 16 * z.rng.uintLessThan(u32, 7) });
        }
        try z.line("v27 = const 0", .{});
        try z.line("v28 = const 1", .{});
        try z.line("v29 = const 3", .{});
        for (0..z.nblocks) |bi| {
            if (bi != 0) try z.out.print(t.allocator, "b{d}:\n", .{bi});
            for (0..z.rng.intRangeAtMost(u32, 3, 12)) |_| try z.op();
            if (bi + 1 == z.nblocks) {
                try z.line("v13 = const 5", .{});
                try z.line("v14 = add v{d} v{d}", .{ z.pick(ints), z.pick(ints) });
                try z.line("ret v{d}", .{z.pick(ints)});
                break;
            }
            const cs = [_][]const u8{ "eq", "ne", "lt", "le", "gt", "ge", "ult", "ule", "ugt", "uge" };
            switch (z.rng.uintLessThan(u32, 4)) {
                0 => try z.line("br b{d}", .{bi + 1}),
                1 => if (bi > 0) {
                    // Bounded back edge (never to b0, which resets it): v27 counts back edges taken (at
                    // most 3 in total).
                    try z.line("v27 = add v27 v28", .{});
                    try z.line("cbr lt v27 v29 b{d} b{d}", .{ z.rng.intRangeAtMost(u32, 1, @intCast(bi)), bi + 1 });
                } else try z.line("br b1", .{}),
                else => {
                    const a = z.pick(ints);
                    const b = if (z.rng.boolean()) z.pick(ints) else 3; // v3 is often a constant
                    const far = z.rng.intRangeAtMost(u32, @intCast(bi + 1), z.nblocks - 1);
                    if (z.rng.boolean())
                        try z.line("cbr {s} v{d} v{d} b{d} b{d}", .{ cs[z.rng.uintLessThan(usize, cs.len)], a, b, bi + 1, far })
                    else
                        try z.line("cbr {s} v{d} v{d} b{d} b{d}", .{ cs[z.rng.uintLessThan(usize, cs.len)], a, b, far, bi + 1 });
                },
            }
        }
    }
};

test "x64 codegen: random differential fuzz (constants, memory, calls, division, shifts, vectors)" {
    if (!can_run) return error.SkipZigTest;
    var buf: std.ArrayList(u8) = .empty;
    defer buf.deinit(t.allocator);
    var prng = std.Random.DefaultPrng.init(0x64c0de);
    const opts = vecOpts();
    var mem: [32]u64 = undefined;
    for (0..250) |_| {
        var z: Fuzz = .{ .rng = prng.random(), .out = &buf, .nblocks = prng.random().intRangeAtMost(u32, 2, 7) };
        try z.program();
        var f = try parse(buf.items);
        defer f.deinit();
        for (&mem) |*w| w.* = if (prng.random().boolean()) @bitCast(edges[prng.random().uintLessThan(usize, edges.len)]) else prng.random().int(u64);
        try check(&f, &.{ &.{ 7, -3 }, &.{ std.math.minInt(i64), 64 } }, &mem, &opts);
    }
}

// ---- shapes (bytes; any host, SysV register assignment) --------------------------------------

fn compileBytes(src: []const u8, opts: cx.Options, cb: *CodeBuffer) !cx.Stats {
    var f = try parse(src);
    defer f.deinit();
    return cx.compile(t.allocator, &f, cb, opts);
}

test "x64 codegen: leaf functions have no frame and compute in place" {
    var cb = CodeBuffer.init(t.allocator);
    defer cb.deinit();
    _ = try compileBytes("func f 2\ne:\n  v2 = add v0 v1\n  ret v2\n", .{}, &cb);
    if (sysv) try t.expectEqualSlices(u8, &.{ 0x48, 0x8D, 0x04, 0x37, 0xC3 }, cb.code()); // lea rax, [rdi+rsi]; ret

    var cb2 = CodeBuffer.init(t.allocator);
    defer cb2.deinit();
    const st = try compileBytes(
        \\func g 2
        \\e:
        \\  v2 = const 8
        \\  v3 = mul v0 v2
        \\  v4 = const 100
        \\  v6 = sub v4 v3
        \\  v5 = const 0
        \\  v7 = sdiv v6 v1
        \\  store v1 16 v5
        \\  ret v7
    , .{}, &cb2);
    try t.expectEqual(@as(u32, 3), st.excluded);
    if (sysv) try t.expectEqualSlices(u8, &.{
        0x48, 0xC1, 0xE7, 0x03, // shl rdi, 3 (v3 takes v0's register)
        0xB8, 0x64, 0x00, 0x00, 0x00, // mov eax, 100 (v6: the dividend, in rax)
        0x48, 0x29, 0xF8, // sub rax, rdi
        0x48, 0x99, // cqo
        0x48, 0xF7, 0xFE, // idiv rsi
        0x48, 0xC7, 0x46, 0x10, 0x00, 0x00, 0x00, 0x00, // mov qword [rsi+16], 0
        0xC3, // ret
    }, cb2.code());
}

test "x64 codegen: counted loop body and inverted branch" {
    var cb = CodeBuffer.init(t.allocator);
    defer cb.deinit();
    _ = try compileBytes(
        \\func sum 1
        \\e:
        \\  v1 = const 0
        \\  v2 = const 0
        \\  v3 = const 1
        \\  br l
        \\l:
        \\  cbr ge v2 v0 done body
        \\body:
        \\  v4 = load v2 0
        \\  v1 = add v1 v2
        \\  v2 = add v2 v3
        \\  br l
        \\done:
        \\  ret v1
    , .{}, &cb);
    // `then` (done) is not next: jcc + fall through; the loop is cmp/jcc/3 ops/jmp, all rel8.
    if (sysv) try t.expectEqualSlices(u8, &.{
        0x31, 0xC0, // xor eax, eax
        0x45, 0x31, 0xC0, // xor r8d, r8d
        0x49, 0x39, 0xF8, // l: cmp r8, rdi
        0x7D, 0x0C, // jge done
        0x4D, 0x8B, 0x08, // mov r9, [r8]
        0x4C, 0x01, 0xC0, // add rax, r8
        0x49, 0x83, 0xC0, 0x01, // add r8, 1
        0xEB, 0xEF, // jmp l
        0xC3, // done: ret
    }, cb.code());

    var cb2 = CodeBuffer.init(t.allocator);
    defer cb2.deinit();
    _ = try compileBytes(
        \\func m 2
        \\e:
        \\  cbr lt v0 v1 a b
        \\a:
        \\  ret v0
        \\b:
        \\  ret v1
    , .{}, &cb2);
    // `then` is the next block: the inverted condition jumps to `else`, no jmp.
    if (sysv) try t.expectEqualSlices(u8, &.{
        0x48, 0x39, 0xF7, // cmp rdi, rsi
        0x7D, 0x04, // jge b
        0x48, 0x89, 0xF8, // mov rax, rdi
        0xC3, // ret
        0x48, 0x89, 0xF0, // b: mov rax, rsi
        0xC3, // ret
    }, cb2.code());
}

test "x64 codegen: calls keep rsp aligned, callee-saved registers only when needed" {
    var cb = CodeBuffer.init(t.allocator);
    defer cb.deinit();
    _ = try compileBytes("func c 1\ne:\n  v1 = call 4096 v0\n  v2 = add v1 v0\n  ret v2\n", .{}, &cb);
    // v0 lives across the call in rbx; one push keeps rsp 16-aligned at the call.
    if (sysv) try t.expectEqualSlices(u8, &.{
        0x53, // push rbx
        0x48, 0x89, 0xFB, // mov rbx, rdi
        0x48, 0x89, 0xDF, // mov rdi, rbx
        0x41, 0xBB, 0x00, 0x10, 0x00, 0x00, // mov r11d, 4096
        0x41, 0xFF, 0xD3, // call r11
        0x48, 0x01, 0xD8, // add rax, rbx
        0x5B, // pop rbx
        0xC3, // ret
    }, cb.code());
}
