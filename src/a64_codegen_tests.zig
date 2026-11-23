//! AArch64 IR Compiler code quality and NEON vector lowering: execution against `ir.interpret`
//! and instruction-shape checks (A64 work package).
const std = @import("std");
const builtin = @import("builtin");
const C = @import("aarch64/compiler.zig");
const ir = @import("core/ir.zig");
const ra = @import("core/regalloc.zig");
const CodeBuffer = @import("core/code_buffer.zig").CodeBuffer;
const JitRuntime = @import("core/jit_runtime.zig");
const enc = C.enc;
const t = std.testing;
const can_run = builtin.cpu.arch == .aarch64;
const minI = std.math.minInt(i64);
const maxI = std.math.maxInt(i64);

/// Register masks for execution: default, heavy GP/FP spilling, parameters partly pinned.
const masks = [_]C.Options{
    .{},
    .{ .allocatable = 1 << 19, .fp_allocatable = 1 },
    .{ .allocatable = (1 << 0) | (1 << 9), .fp_allocatable = (1 << 16) | (1 << 17) },
    .{ .allocatable = (1 << 1) | (1 << 2) | (1 << 20) | (1 << 21), .fp_allocatable = (1 << 3) | (1 << 29) },
};

/// Compiled words of `f`.
fn codeOf(f: *const ir.Func, opts: C.Options) ![]u32 {
    var cb = CodeBuffer.init(t.allocator);
    defer cb.deinit();
    _ = try C.compile(t.allocator, f, &cb, opts);
    const c = cb.code();
    const w = try t.allocator.alloc(u32, c.len / 4);
    for (w, 0..) |*x, i| x.* = std.mem.readInt(u32, c[4 * i ..][0..4], .little);
    return w;
}
fn codeOfSrc(src: []const u8) ![]u32 {
    var f = try ir.parse(t.allocator, src);
    defer f.deinit();
    return codeOf(&f, .{});
}

fn run(f: *const ir.Func, opts: C.Options, args: []const i64) !i64 {
    var cb = CodeBuffer.init(t.allocator);
    defer cb.deinit();
    _ = try C.compile(t.allocator, f, &cb, opts);
    const fun = try JitRuntime.add(&cb);
    defer fun.release();
    var v: [8]i64 = @splat(0);
    for (args, 0..) |x, i| v[i] = x;
    const F = *const fn (i64, i64, i64, i64, i64, i64, i64, i64) callconv(.c) i64;
    return fun.ptr(F)(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]);
}

/// Bit-exact equality where any NaN equals any NaN (as f64, else per f32 half).
fn sameBits(x: u64, y: u64) bool {
    if (x == y) return true;
    if (std.math.isNan(@as(f64, @bitCast(x))) and std.math.isNan(@as(f64, @bitCast(y)))) return true;
    inline for (.{ 0, 32 }) |sh| {
        const a: u32 = @truncate(x >> sh);
        const b: u32 = @truncate(y >> sh);
        if (a != b and !(std.math.isNan(@as(f32, @bitCast(a))) and std.math.isNan(@as(f32, @bitCast(b))))) return false;
    }
    return true;
}

/// Runs `f` with `args` (args[0] may point into `mem`) under every mask and compares the
/// result and `mem` with the interpreter.
fn checkMem(f: *const ir.Func, args: []const i64, mem: []u64) !void {
    const init = try t.allocator.dupe(u64, mem);
    defer t.allocator.free(init);
    const want = try ir.interpret(f, args, 1 << 22);
    const want_mem = try t.allocator.dupe(u64, mem);
    defer t.allocator.free(want_mem);
    for (masks) |m| {
        @memcpy(mem, init);
        const got = try run(f, m, args);
        if (got != want) {
            std.debug.print("mask {x}/{x}: want {d} got {d}\n", .{ m.allocatable, m.fp_allocatable, want, got });
            return error.TestExpectedEqual;
        }
        for (want_mem, mem, 0..) |x, y, i| if (!sameBits(x, y)) {
            std.debug.print("mask {x}/{x}: mem[{d}] want {x} got {x}\n", .{ m.allocatable, m.fp_allocatable, i, x, y });
            return error.TestExpectedEqual;
        };
    }
}
fn check(f: *const ir.Func, args: []const i64) !void {
    var none: [0]u64 = .{};
    try checkMem(f, args, &none);
}
fn checkSrc(src: []const u8, args: []const i64) !void {
    var f = try ir.parse(t.allocator, src);
    defer f.deinit();
    try check(&f, args);
}

// ---- encodings ------------------------------------------------------------------------------

test "a64 codegen: encodings match LLVM (any host)" {
    // llvm-mc -triple=aarch64 -mattr=+neon -show-encoding, one line per word below.
    const got = [_]u32{
        enc.rrr(enc.add_r, 1, 2, 3), // add x1, x2, x3
        enc.rrr(enc.sub_r, 1, 31, 3), // neg x1, x3
        enc.cmpReg(2, 3), // cmp x2, x3
        enc.cmpImm(false, 2, 4095, 0), // cmp x2, #4095
        enc.cmpImm(false, 2, 1, 1), // cmp x2, #1, lsl #12
        enc.cmpImm(true, 2, 1, 0), // cmn x2, #1
        enc.addImm(false, 1, 2, 4095, 0), // add x1, x2, #4095
        enc.addImm(false, 1, 2, 1, 1), // add x1, x2, #1, lsl #12
        enc.addImm(true, 1, 2, 7, 0), // sub x1, x2, #7
        enc.rrr(enc.mul, 1, 2, 3), // mul x1, x2, x3
        enc.madd(false, 1, 2, 3, 4), // madd x1, x2, x3, x4
        enc.madd(true, 1, 2, 3, 4), // msub x1, x2, x3, x4
        enc.rrr(enc.sdiv, 1, 2, 3), // sdiv x1, x2, x3
        enc.logImm(enc.and_i, 1, 2, @import("aarch64/a64.zig").logicalImm(0xff, true).?), // and x1, x2, #0xff
        enc.logImm(enc.orr_i, 1, 31, @import("aarch64/a64.zig").logicalImm(0x5555555555555555, true).?), // orr x1, xzr, #0x5555555555555555
        enc.logImm(enc.eor_i, 1, 2, @import("aarch64/a64.zig").logicalImm(0xfffffffffffffffe, true).?), // eor x1, x2, #0xfffffffffffffffe
        enc.shiftImm(.shl, 1, 2, 3), // lsl x1, x2, #3
        enc.shiftImm(.shr, 1, 2, 63), // lsr x1, x2, #63
        enc.shiftImm(.sar, 1, 2, 1), // asr x1, x2, #1
        enc.shiftImm(.shl, 1, 2, 0), // lsr x1, x2, #0
        enc.rrr(enc.lslv, 1, 2, 3), // lsl x1, x2, x3
        enc.rrr(enc.lsrv, 1, 2, 3), // lsr x1, x2, x3
        enc.rrr(enc.asrv, 1, 2, 3), // asr x1, x2, x3
        enc.movw(enc.movz, 1, 0x1234, 1), // movz x1, #0x1234, lsl #16
        enc.movw(enc.movn, 1, 0, 0), // movn x1, #0
        enc.movw(enc.movn, 1, 0x1234, 3), // movn x1, #0x1234, lsl #48
        enc.movw(enc.movk, 1, 0xbeef, 2), // movk x1, #0xbeef, lsl #32
        enc.mov(1, 31), // mov x1, xzr
        enc.cset(1, enc.lt), // cset x1, lt
        enc.csel(1, 2, 3, enc.vs), // csel x1, x2, x3, vs
        enc.ccmnImm(16, 1, 1, enc.vc), // ccmn x16, #1, #1, vc
        enc.ldstImm(true, .x, 1, 2, 4095), // ldr x1, [x2, #32760]
        enc.ldstUnscaled(true, .x, 1, 2, -256), // ldur x1, [x2, #-256]
        enc.ldstUnscaled(false, .x, 1, 2, 255), // stur x1, [x2, #255]
        enc.ldstReg(true, .x, 1, 2, 8), // ldr x1, [x2, x8]
        enc.ldstReg(false, .x, 31, 31, 8), // str xzr, [sp, x8]
        enc.ldstImm(true, .d, 1, 2, 1), // ldr d1, [x2, #8]
        enc.ldstUnscaled(true, .d, 1, 2, -8), // ldur d1, [x2, #-8]
        enc.ldstReg(false, .s, 1, 2, 8), // str s1, [x2, x8]
        enc.ldstImm(true, .q, 1, 2, 4095), // ldr q1, [x2, #65520]
        enc.ldstImm(false, .q, 1, 31, 1), // str q1, [sp, #16]
        enc.ldstUnscaled(true, .q, 1, 2, -16), // ldur q1, [x2, #-16]
        enc.ldstUnscaled(false, .q, 1, 2, 255), // stur q1, [x2, #255]
        enc.ldstReg(true, .q, 1, 2, 8), // ldr q1, [x2, x8]
        enc.ldstReg(false, .q, 1, 2, 8), // str q1, [x2, x8]
        enc.fmovImm(.f64, 1, @import("aarch64/a64.zig").fpImm8(1.0).?), // fmov d1, #1.0
        enc.fmovImm(.f32, 1, @import("aarch64/a64.zig").fpImm8(-2.5).?), // fmov s1, #-2.5
        enc.fmovImm(.f64, 1, @import("aarch64/a64.zig").fpImm8(0.125).?), // fmov d1, #0.125
        enc.moviZero(1), // movi v1.2d, #0
        enc.rrr(enc.add_4s, 1, 2, 3), // add v1.4s, v2.4s, v3.4s
        enc.rrr(enc.add_2d, 1, 2, 3), // add v1.2d, v2.2d, v3.2d
        enc.rrr(enc.sub_4s, 1, 2, 3), // sub v1.4s, v2.4s, v3.4s
        enc.rrr(enc.sub_2d, 1, 2, 3), // sub v1.2d, v2.2d, v3.2d
        enc.rrr(enc.mul_4s, 1, 2, 3), // mul v1.4s, v2.4s, v3.4s
        enc.rrr(enc.fadd_4s, 1, 2, 3), // fadd v1.4s, v2.4s, v3.4s
        enc.rrr(enc.fadd_4s | enc.sz_2d, 1, 2, 3), // fadd v1.2d, v2.2d, v3.2d
        enc.rrr(enc.fsub_4s, 1, 2, 3), // fsub v1.4s, v2.4s, v3.4s
        enc.rrr(enc.fsub_4s | enc.sz_2d, 1, 2, 3), // fsub v1.2d, v2.2d, v3.2d
        enc.rrr(enc.fmul_4s, 1, 2, 3), // fmul v1.4s, v2.4s, v3.4s
        enc.rrr(enc.fmul_4s | enc.sz_2d, 1, 2, 3), // fmul v1.2d, v2.2d, v3.2d
        enc.rrr(enc.fdiv_4s, 1, 2, 3), // fdiv v1.4s, v2.4s, v3.4s
        enc.rrr(enc.fdiv_4s | enc.sz_2d, 1, 2, 3), // fdiv v1.2d, v2.2d, v3.2d
        enc.rrr(enc.and_16b, 1, 2, 3), // and v1.16b, v2.16b, v3.16b
        enc.rrr(enc.orr_16b, 1, 2, 3), // orr v1.16b, v2.16b, v3.16b
        enc.rrr(enc.eor_16b, 1, 2, 3), // eor v1.16b, v2.16b, v3.16b
        enc.rrr(enc.bsl_16b, 1, 2, 3), // bsl v1.16b, v2.16b, v3.16b
        enc.rrr(enc.bit_16b, 1, 2, 3), // bit v1.16b, v2.16b, v3.16b
        enc.rrr(enc.bif_16b, 1, 2, 3), // bif v1.16b, v2.16b, v3.16b
        enc.rrr(enc.fcmgt_4s, 1, 2, 3), // fcmgt v1.4s, v2.4s, v3.4s
        enc.rrr(enc.fcmgt_4s | enc.sz_2d, 1, 2, 3), // fcmgt v1.2d, v2.2d, v3.2d
        enc.dupGp(false, 1, 2), // dup v1.4s, w2
        enc.dupGp(true, 1, 2), // dup v1.2d, x2
        enc.dupGp(false, 1, 31), // dup v1.4s, wzr
        enc.dupElem0(false, 1, 2), // dup v1.4s, v2.s[0]
        enc.dupElem0(true, 1, 2), // dup v1.2d, v2.d[0]
        enc.smovS(1, 2, 3), // smov x1, v2.s[3]
        enc.umovD(1, 2, 1), // mov x1, v2.d[1]
        enc.dupScalar(false, 1, 2, 2), // mov s1, v2.s[2]
        enc.dupScalar(true, 1, 2, 1), // mov d1, v2.d[1]
        enc.insD(1, 1, 2), // mov v1.d[1], x2
        enc.insD(1, 0, 31), // mov v1.d[0], xzr
        enc.fmovFromGp(.f64, 1, 2), // fmov d1, x2
        enc.movV(1, 2), // mov v1.16b, v2.16b
        enc.bcondSkip(enc.ne), // b.ne #8
        enc.cbzSkip(false, 3), // cbz x3, #8
        enc.cbzSkip(true, 3), // cbnz x3, #8
        enc.blr(16), // blr x16
    };
    const want = [_]u32{
        0x8b030041, 0xcb0303e1, 0xeb03005f, 0xf13ffc5f, 0xf140045f, 0xb100045f, 0x913ffc41,
        0x91400441, 0xd1001c41, 0x9b037c41, 0x9b031041, 0x9b039041, 0x9ac30c41, 0x92401c41,
        0xb200f3e1, 0xd27ff841, 0xd37df041, 0xd37ffc41, 0x9341fc41, 0xd340fc41, 0x9ac32041,
        0x9ac32441, 0x9ac32841, 0xd2a24681, 0x92800001, 0x92e24681, 0xf2d7dde1, 0xaa1f03e1,
        0x9a9fa7e1, 0x9a836041, 0xba417a01, 0xf97ffc41, 0xf8500041, 0xf80ff041, 0xf8686841,
        0xf8286bff, 0xfd400441, 0xfc5f8041, 0xbc286841, 0x3dfffc41, 0x3d8007e1, 0x3cdf0041,
        0x3c8ff041, 0x3ce86841, 0x3ca86841, 0x1e6e1001, 0x1e309001, 0x1e681001, 0x6f00e401,
        0x4ea38441, 0x4ee38441, 0x6ea38441, 0x6ee38441, 0x4ea39c41, 0x4e23d441, 0x4e63d441,
        0x4ea3d441, 0x4ee3d441, 0x6e23dc41, 0x6e63dc41, 0x6e23fc41, 0x6e63fc41, 0x4e231c41,
        0x4ea31c41, 0x6e231c41, 0x6e631c41, 0x6ea31c41, 0x6ee31c41, 0x6ea3e441, 0x6ee3e441,
        0x4e040c41, 0x4e080c41, 0x4e040fe1, 0x4e040441, 0x4e080441, 0x4e1c2c41, 0x4e183c41,
        0x5e140441, 0x5e180441, 0x4e181c41, 0x4e081fe1, 0x9e670041, 0x4ea21c41, 0x54000041,
        0xb4000043, 0xb5000043, 0xd63f0200,
    };
    try t.expectEqualSlices(u32, &want, &got);
}

// ---- shapes ---------------------------------------------------------------------------------

fn count(code: []const u32, mask: u32, val: u32) usize {
    var n: usize = 0;
    for (code) |w| n += @intFromBool(w & mask == val);
    return n;
}
const is_bcond = .{ 0xFF000010, 0x54000000 };
const is_b = .{ 0xFC000000, 0x14000000 };
const is_cbz = .{ 0x7E000000, 0x34000000 };
const is_stp_fp = .{ 0xFFFFFFFF, 0xA9BF7BFD };

test "a64 codegen: a leaf add is two instructions" {
    const c = try codeOfSrc("func f 2\ne:\n  v2 = add v0 v1\n  ret v2\n");
    defer t.allocator.free(c);
    try t.expectEqualSlices(u32, &.{ enc.rrr(enc.add_r, 0, 0, 1), enc.ret }, c);
}

test "a64 codegen: constants fold into immediates and xzr" {
    const cases = [_]struct { src: []const u8, want: []const u32 }{
        // add/sub imm12, the shifted form, a negative constant as the opposite op
        .{ .src = "func f 1\ne:\n  v1 = const 4095\n  v2 = add v0 v1\n  ret v2\n", .want = &.{ enc.addImm(false, 0, 0, 4095, 0), enc.ret } },
        .{ .src = "func f 1\ne:\n  v1 = const 4096\n  v2 = add v1 v0\n  ret v2\n", .want = &.{ enc.addImm(false, 0, 0, 1, 1), enc.ret } },
        .{ .src = "func f 1\ne:\n  v1 = const -7\n  v2 = add v0 v1\n  ret v2\n", .want = &.{ enc.addImm(true, 0, 0, 7, 0), enc.ret } },
        .{ .src = "func f 1\ne:\n  v1 = const -7\n  v2 = sub v0 v1\n  ret v2\n", .want = &.{ enc.addImm(false, 0, 0, 7, 0), enc.ret } },
        // 0 - x reads xzr; logical immediates; shifts and power-of-two multiplies
        .{ .src = "func f 1\ne:\n  v1 = const 0\n  v2 = sub v1 v0\n  ret v2\n", .want = &.{ enc.rrr(enc.sub_r, 0, 31, 0), enc.ret } },
        .{ .src = "func f 1\ne:\n  v1 = const 255\n  v2 = and v0 v1\n  ret v2\n", .want = &.{ enc.logImm(enc.and_i, 0, 0, @import("aarch64/a64.zig").logicalImm(255, true).?), enc.ret } },
        .{ .src = "func f 1\ne:\n  v1 = const 67\n  v2 = shl v0 v1\n  ret v2\n", .want = &.{ enc.shiftImm(.shl, 0, 0, 3), enc.ret } },
        .{ .src = "func f 1\ne:\n  v1 = const 8\n  v2 = mul v1 v0\n  ret v2\n", .want = &.{ enc.shiftImm(.shl, 0, 0, 3), enc.ret } },
        // store of 0 via xzr; constant results; shortest movImm
        .{ .src = "func f 1\ne:\n  v1 = const 0\n  store v0 8 v1\n  ret v1\n", .want = &.{ enc.ldstImm(false, .x, 31, 0, 1), enc.movw(enc.movz, 0, 0, 0), enc.ret } },
        .{ .src = "func f 0\ne:\n  v1 = const 6\n  v2 = const 7\n  v3 = mul v1 v2\n  ret v3\n", .want = &.{ enc.movw(enc.movz, 0, 42, 0), enc.ret } },
        .{ .src = "func f 0\ne:\n  v1 = const -2\n  ret v1\n", .want = &.{ enc.movw(enc.movn, 0, 1, 0), enc.ret } },
        .{ .src = "func f 0\ne:\n  v1 = const 6148914691236517205\n  ret v1\n", .want = &.{ enc.logImm(enc.orr_i, 0, 31, @import("aarch64/a64.zig").logicalImm(0x5555555555555555, true).?), enc.ret } },
        .{ .src = "func f 0\ne:\n  v1 = const -4294901761\n  ret v1\n", .want = &.{ enc.movw(enc.movn, 0, 0xFFFF, 1), enc.ret } },
        .{ .src = "func f 0\ne:\n  v1 = const 81985529216486895\n  ret v1\n", .want = &.{ enc.movw(enc.movz, 0, 0xCDEF, 0), enc.movw(enc.movk, 0, 0x89AB, 1), enc.movw(enc.movk, 0, 0x4567, 2), enc.movw(enc.movk, 0, 0x0123, 3), enc.ret } },
    };
    for (cases) |cs| {
        const c = try codeOfSrc(cs.src);
        defer t.allocator.free(c);
        t.expectEqualSlices(u32, cs.want, c) catch |e| {
            std.debug.print("source:\n{s}\n", .{cs.src});
            return e;
        };
    }
}

test "a64 codegen: a counted loop" {
    const src =
        \\func f 1
        \\e:
        \\  v1 = const 0
        \\  v2 = const 0
        \\  v3 = const 1
        \\  br loop
        \\loop:
        \\  cbr lt v1 v0 body done
        \\body:
        \\  v2 = add v2 v1
        \\  v1 = add v1 v3
        \\  br loop
        \\done:
        \\  ret v2
    ;
    const c = try codeOfSrc(src);
    defer t.allocator.free(c);
    // e: 2 movz; loop: cmp + b.ge; body: add, add #1, b; done: mov x0 + ret.
    try t.expectEqual(@as(usize, 1), count(c, is_bcond[0], is_bcond[1]));
    try t.expectEqual(@as(usize, 1), count(c, is_b[0], is_b[1]));
    try t.expectEqual(@as(usize, 1), count(c, 0xFFFFFC00, 0x91000400)); // add xN, xM, #1
    try t.expectEqual(@as(usize, 0), count(c, is_stp_fp[0], is_stp_fp[1]));
    try t.expect(c.len <= 10);
    var f = try ir.parse(t.allocator, src);
    defer f.deinit();
    if (can_run) for ([_]i64{ 0, 1, 10, -5 }) |n| try check(&f, &.{n});
}

test "a64 codegen: cbr to the next block is one conditional branch" {
    const src =
        \\func f 2
        \\e:
        \\  cbr eq v0 v1 a b
        \\a:
        \\  ret v0
        \\b:
        \\  ret v1
    ;
    const c = try codeOfSrc(src);
    defer t.allocator.free(c);
    try t.expectEqual(@as(usize, 1), count(c, is_bcond[0], is_bcond[1]));
    try t.expectEqual(@as(usize, 0), count(c, is_b[0], is_b[1]));
    // eq against a constant 0: cbz/cbnz, no compare
    const z = try codeOfSrc("func f 1\ne:\n  v1 = const 0\n  cbr ne v0 v1 a b\na:\n  ret v1\nb:\n  ret v0\n");
    defer t.allocator.free(z);
    try t.expectEqual(@as(usize, 1), count(z, is_cbz[0], is_cbz[1]));
    try t.expectEqual(@as(usize, 0), count(z, 0xFF00001F, 0xEB00001F));
    if (!can_run) return;
    try checkSrc(src, &.{ 3, 3 });
    try checkSrc(src, &.{ 3, 4 });
}

// ---- edge values ----------------------------------------------------------------------------

const Fn8 = *const fn (i64, i64, i64, i64, i64, i64, i64, i64) callconv(.c) i64;

/// Compiles `f` once per mask in `ms`; for each case `i` (arguments `args[i % args.len]`,
/// memory image `mems[i % mems.len]` copied into `mem`) compares the result and `mem` with the
/// interpreter (NaN == NaN).
fn checkAll(f: *const ir.Func, args: []const []const i64, mem: []u64, mems: []const []const u64, ms: []const C.Options) !void {
    const n = @max(args.len, mems.len, 1);
    const want_mem = try t.allocator.alloc(u64, mem.len);
    defer t.allocator.free(want_mem);
    for (ms) |m| {
        var cb = CodeBuffer.init(t.allocator);
        defer cb.deinit();
        _ = try C.compile(t.allocator, f, &cb, m);
        const fun = try JitRuntime.add(&cb);
        defer fun.release();
        for (0..n) |i| {
            var v: [8]i64 = @splat(0);
            if (args.len != 0) for (args[i % args.len], 0..) |x, k| {
                v[k] = x;
            };
            if (mems.len != 0) @memcpy(mem, mems[i % mems.len]);
            const want = try ir.interpret(f, v[0..f.params], 1 << 24);
            @memcpy(want_mem, mem);
            if (mems.len != 0) @memcpy(mem, mems[i % mems.len]);
            const got = fun.ptr(Fn8)(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]);
            var ok = got == want;
            for (want_mem, mem) |x, y| ok = ok and sameBits(x, y);
            if (!ok) {
                var txt: std.ArrayList(u8) = .empty;
                defer txt.deinit(t.allocator);
                f.print(t.allocator, &txt) catch {};
                std.debug.print("mismatch (mask {x}/{x}, case {d}, args {any}): want {d} got {d}\n{s}\n", .{ m.allocatable, m.fp_allocatable, i, v[0..f.params], want, got, txt.items });
                for (want_mem, mem, 0..) |x, y, k| if (!sameBits(x, y)) std.debug.print("  mem[{d}] want {x} got {x}\n", .{ k, x, y });
                return error.TestExpectedEqual;
            }
        }
    }
}

const int_edges = [_]i64{ 0, 1, -1, 2, 7, 63, 64, 65, 255, 4095, 4096, 4097, -4095, -4096, -4097, 0xFFF000, 0x1000000, 0xFF, 0x5555555555555555, -2, 0x00FF00FF00FF00FF, 0x123456789, minI, maxI, 1 << 62, -98765 };
const int_inputs = [_][]const i64{ &.{0}, &.{1}, &.{-1}, &.{2}, &.{minI}, &.{maxI}, &.{4095}, &.{4096}, &.{-4096}, &.{0x123456789}, &.{-98765}, &.{63} };

test "a64 codegen: integer constants as operands (immediate forms) match the interpreter" {
    if (!can_run) return error.SkipZigTest;
    const ms = masks[0..2];
    var none: [0]u64 = .{};
    for (std.enums.values(ir.Arith)) |op| for (int_edges) |k| for (0..2) |side| for (0..3) |place| {
        const div = op == .sdiv or op == .srem;
        if (div and side == 0 and (k == 0 or k == -1)) continue;
        var buf: [512]u8 = undefined;
        const operands: []const u8 = if (side == 0) "v0 v1" else "v1 v0";
        const src = switch (place) {
            // function-wide constant, block-local constant, copy of a constant
            0 => try std.fmt.bufPrint(&buf, "func f 1\ne:\n  v1 = const {d}\n  v2 = {t} {s}\n  ret v2\n", .{ k, op, operands }),
            1 => try std.fmt.bufPrint(&buf, "func f 1\ne:\n  br b\nb:\n  v1 = const {d}\n  v2 = {t} {s}\n  v1 = const 0\n  cbr eq v2 v1 b x\nx:\n  ret v2\n", .{ k, op, operands }),
            else => try std.fmt.bufPrint(&buf, "func f 1\ne:\n  v3 = const {d}\n  v1 = copy v3\n  v2 = {t} {s}\n  v4 = add v2 v1\n  ret v4\n", .{ k, op, operands }),
        };
        var f = try ir.parse(t.allocator, src);
        defer f.deinit();
        var inputs: [int_inputs.len][]const i64 = undefined;
        var ni: usize = 0;
        for (int_inputs) |in| {
            if (div and side == 1 and (in[0] == 0 or in[0] == -1)) continue;
            // place 1 loops while the result is 0: keep inputs whose result is not 0
            if (place == 1 and ir.arith(op, if (side == 0) in[0] else k, if (side == 0) k else in[0]) == 0) continue;
            inputs[ni] = in;
            ni += 1;
        }
        if (ni == 0) continue;
        try checkAll(&f, inputs[0..ni], &none, &.{}, ms);
    };
}

test "a64 codegen: comparisons against constants (cmp/cmn imm, cbz) match the interpreter" {
    if (!can_run) return error.SkipZigTest;
    var none: [0]u64 = .{};
    for (std.enums.values(ir.CondCode)) |cc| for (int_edges) |k| for (0..2) |side| {
        var buf: [512]u8 = undefined;
        const operands: []const u8 = if (side == 0) "v0 v1" else "v1 v0";
        const src = try std.fmt.bufPrint(&buf, "func f 1\ne:\n  v1 = const {d}\n  cbr {t} {s} a b\na:\n  v2 = const 1\n  ret v2\nb:\n  v3 = const 2\n  ret v3\n", .{ k, cc, operands });
        var f = try ir.parse(t.allocator, src);
        defer f.deinit();
        var inputs: [int_inputs.len + 3][]const i64 = undefined;
        @memcpy(inputs[0..int_inputs.len], &int_inputs);
        const near = [3]i64{ k, k -% 1, k +% 1 };
        for (0..3) |i| inputs[int_inputs.len + i] = near[i .. i + 1];
        try checkAll(&f, &inputs, &none, &.{}, masks[0..1]);
    };
}

test "a64 codegen: shortest constant materialization" {
    if (!can_run) return error.SkipZigTest;
    const vals = int_edges ++ [_]i64{ 0xFFFF, 0xFFFF0000, 0x123400005678, -4294901761, 0x7FFFFFFF00000000, -0x10000, 0x0001000100010001, 81985529216486895, -81985529216486896, 0x00000000FFFFFFFF };
    var none: [0]u64 = .{};
    for (vals) |k| {
        var buf: [256]u8 = undefined;
        // `ret` of a constant, a constant kept in a register, a call argument
        const src = try std.fmt.bufPrint(&buf, "func f 1\ne:\n  v1 = const {d}\n  v2 = sdiv v0 v1\n  v3 = xor v2 v1\n  ret v3\n", .{k});
        var f = try ir.parse(t.allocator, src);
        defer f.deinit();
        if (k == 0 or k == -1) continue;
        try checkAll(&f, &int_inputs, &none, &.{}, masks[0..2]);
        const r = try std.fmt.bufPrint(&buf, "func f 0\ne:\n  v1 = const {d}\n  ret v1\n", .{k});
        var g = try ir.parse(t.allocator, r);
        defer g.deinit();
        try checkAll(&g, &.{}, &none, &.{}, masks[0..1]);
    }
}

const disps = [_]i32{ 0, 1, 3, 4, 8, -8, 16, -16, -255, -256, -257, 255, 256, 4092, 16380, 16384, 32760, 32764, 32768, 32776, 65520, 65536, 39984, -32768, -39992, -4096 };

test "a64 codegen: load/store displacements for every type" {
    if (!can_run) return error.SkipZigTest;
    const words = 20480; // 160 KiB, base in the middle
    const mem = try t.allocator.alloc(u64, words);
    defer t.allocator.free(mem);
    const image = try t.allocator.alloc(u64, words);
    defer t.allocator.free(image);
    var prng = std.Random.DefaultPrng.init(5);
    for (image) |*x| x.* = prng.random().int(u64);
    const base: i64 = @intCast(@intFromPtr(mem.ptr) + 8 * words / 2);
    for ([_]ir.Type{ .i64, .f32, .f64, .v4i32, .v2i64, .v4f32, .v2f64 }) |tp| for (disps) |d| {
        var buf: [512]u8 = undefined;
        const d2 = -d - 64;
        const src = try std.fmt.bufPrint(&buf,
            \\func f 2
            \\e:
            \\  v2 = load.{t} v0 {d}
            \\  store.{t} v0 {d} v2
            \\  v3 = add v0 v1
            \\  v4 = load.{t} v3 {d}
            \\  store.{t} v3 {d} v4
            \\  v5 = load v0 {d}
            \\  ret v5
            \\
        , .{ tp, d, tp, d2, tp, d, tp, d2 + 24, d });
        var f = try ir.parse(t.allocator, src);
        defer f.deinit();
        try checkAll(&f, &.{ &.{ base, 0 }, &.{ base, -8 }, &.{ base - 40, 40 } }, mem, &.{image}, &masks);
    };
}

const f64_edges = [_]f64{ 0.0, -0.0, 1.0, -1.0, 0.5, -2.5, 3.5, std.math.nan(f64), -std.math.nan(f64), std.math.inf(f64), -std.math.inf(f64), 5e-324, -1e-310, std.math.floatMax(f64), 1e19, -1e19, 9.223372036854775807e18, -9.223372036854775808e18, 9.2233720368547748e18, -9.2233720368547748e18, 4.5e15, -0.75, 1e300, 31.0, 0.1 };

fn fbits(tp: ir.Type, x: f64) u64 {
    return if (tp == .f32) @as(u32, @bitCast(@as(f32, @floatCast(x)))) else @bitCast(x);
}

test "a64 codegen: scalar float edge values (arith, min/max, fcmp, cvt) match the interpreter" {
    if (!can_run) return error.SkipZigTest;
    for ([_]ir.Type{ .f32, .f64 }) |tp| {
        const other: ir.Type = if (tp == .f32) .f64 else .f32;
        var buf: [2048]u8 = undefined;
        const src = try std.fmt.bufPrint(&buf,
            \\func f 1
            \\e:
            \\  v1 = load.{0t} v0 0
            \\  v2 = load.{0t} v0 8
            \\  v3 = add.{0t} v1 v2
            \\  store.{0t} v0 16 v3
            \\  v3 = sub.{0t} v1 v2
            \\  store.{0t} v0 24 v3
            \\  v3 = mul.{0t} v1 v2
            \\  store.{0t} v0 32 v3
            \\  v3 = div.{0t} v1 v2
            \\  store.{0t} v0 40 v3
            \\  v3 = min.{0t} v1 v2
            \\  store.{0t} v0 48 v3
            \\  v3 = max.{0t} v1 v2
            \\  store.{0t} v0 56 v3
            \\  v4 = cvt.i64.{0t} v1
            \\  store v0 64 v4
            \\  v5 = cvt.{1t}.{0t} v1
            \\  store.{1t} v0 72 v5
            \\  v6 = load v0 80
            \\  v7 = cvt.{0t}.i64 v6
            \\  store.{0t} v0 88 v7
            \\  v8 = fcmp.{0t} eq v1 v2
            \\  v9 = fcmp.{0t} ne v1 v2
            \\  v10 = fcmp.{0t} lt v1 v2
            \\  v11 = fcmp.{0t} le v1 v2
            \\  v12 = fcmp.{0t} gt v1 v2
            \\  v13 = fcmp.{0t} ge v1 v2
            \\  v14 = const 1
            \\  v9 = shl v9 v14
            \\  v8 = or v8 v9
            \\  v14 = const 2
            \\  v10 = shl v10 v14
            \\  v8 = or v8 v10
            \\  v14 = const 3
            \\  v11 = shl v11 v14
            \\  v8 = or v8 v11
            \\  v14 = const 4
            \\  v12 = shl v12 v14
            \\  v8 = or v8 v12
            \\  v14 = const 5
            \\  v13 = shl v13 v14
            \\  v8 = or v8 v13
            \\  ret v8
            \\
        , .{ tp, other });
        var f = try ir.parse(t.allocator, src);
        defer f.deinit();
        var mem: [12]u64 = undefined;
        var images: [f64_edges.len * f64_edges.len][12]u64 = undefined;
        var sl: [images.len][]const u64 = undefined;
        for (f64_edges, 0..) |x, i| for (f64_edges, 0..) |y, j| {
            const im = &images[i * f64_edges.len + j];
            @memset(im, 0);
            im[0] = fbits(tp, x);
            im[1] = fbits(tp, y);
            im[10] = @bitCast(@as(i64, @intFromFloat(std.math.clamp(x, -9.2e18, 9.2e18) * @as(f64, if (std.math.isNan(x)) 0 else 1))));
            sl[i * f64_edges.len + j] = im;
        };
        const p: i64 = @intCast(@intFromPtr(&mem));
        try checkAll(&f, &.{&.{p}}, &mem, &sl, &masks);
    }
}

test "a64 codegen: float constants (movi, fmov imm, via GP)" {
    if (!can_run) return error.SkipZigTest;
    for ([_]ir.Type{ .f32, .f64 }) |tp| for (f64_edges ++ [_]f64{ 0.125, -31.0, 1.9375, 0.1875, 16.0, -0.0 }) |x| {
        var f = ir.Func.init(t.allocator);
        defer f.deinit();
        f.params = 1;
        const e = try f.addBlock("e");
        try f.emit(e, .{ .fconst = .{ .dst = 1, .ty = tp, .bits = fbits(tp, x) } });
        try f.emit(e, .{ .store = .{ .base = 0, .disp = 0, .src = 1, .ty = tp } });
        try f.emit(e, .{ .typed = .{ .op = .add, .ty = tp, .dst = 2, .a = 1, .b = 1 } });
        try f.emit(e, .{ .store = .{ .base = 0, .disp = 8, .src = 2, .ty = tp } });
        try f.emit(e, .{ .ret = 0 });
        var mem: [2]u64 = .{ 0, 0 };
        const p: i64 = @intCast(@intFromPtr(&mem));
        const zero = [2]u64{ 0, 0 };
        try checkAll(&f, &.{&.{p}}, &mem, &.{&zero}, &masks);
    };
}

test "a64 codegen: vector edge values (lanes, wrap, NaN, min/max) match the interpreter" {
    if (!can_run) return error.SkipZigTest;
    const vec_src =
        \\func f 1
        \\e:
        \\  v1 = load.{0t} v0 0
        \\  v2 = load.{0t} v0 16
        \\  v3 = {1t}.{0t} v1 v2
        \\  store.{0t} v0 32 v3
        \\  v4 = {1t}.{0t} v2 v1
        \\  store.{0t} v0 48 v4
        \\  v5 = {1t}.{0t} v1 v1
        \\  store.{0t} v0 64 v5
        \\  v1 = {1t}.{0t} v1 v2
        \\  store.{0t} v0 80 v1
        \\  v2 = {1t}.{0t} v3 v2
        \\  store.{0t} v0 96 v2
        \\  ret v0
        \\
    ;
    const i32s = [_]u32{ 0x7fffffff, 0x80000000, 0xffffffff, 1, 0, 12345, 0xfffffffe, 0x40000000, 0x10001 };
    const i64s = [_]u64{ 0x7fffffffffffffff, 0x8000000000000000, 0xffffffffffffffff, 3, 0x100000001, 0, 0xfedcba9876543210, 0x8000000080000000 };
    var mem: [14]u64 = undefined;
    const p: i64 = @intCast(@intFromPtr(&mem));
    for ([_]ir.Type{ .v4i32, .v2i64, .v4f32, .v2f64 }) |tp| for (std.enums.values(ir.TArith)) |op| {
        const ok = switch (op) {
            .add, .sub, .mul => true,
            .div, .min, .max => tp.isFloat(),
            .@"and", .@"or", .xor => !tp.isFloat(),
        };
        if (!ok) continue;
        var buf: [1024]u8 = undefined;
        var f = try ir.parse(t.allocator, try std.fmt.bufPrint(&buf, vec_src, .{ tp, op }));
        defer f.deinit();
        var images: [64][14]u64 = undefined;
        var sl: [64][]const u64 = undefined;
        for (&images, 0..) |*im, i| {
            @memset(im, 0);
            for (0..4) |l| {
                const k = i * 3 + l * 5;
                switch (tp) {
                    .v4i32 => {
                        const a = i32s[k % i32s.len];
                        const b = i32s[(k + i / 7 + 1) % i32s.len];
                        im[l / 2] |= @as(u64, a) << @intCast(32 * (l % 2));
                        im[2 + l / 2] |= @as(u64, b) << @intCast(32 * (l % 2));
                    },
                    .v2i64 => if (l < 2) {
                        im[l] = i64s[k % i64s.len];
                        im[2 + l] = i64s[(k + i / 5 + 1) % i64s.len];
                    },
                    .v4f32 => {
                        im[l / 2] |= fbits(.f32, f64_edges[k % f64_edges.len]) << @intCast(32 * (l % 2));
                        im[2 + l / 2] |= fbits(.f32, f64_edges[(k + i / 3 + 1) % f64_edges.len]) << @intCast(32 * (l % 2));
                    },
                    else => if (l < 2) {
                        im[l] = fbits(.f64, f64_edges[k % f64_edges.len]);
                        im[2 + l] = fbits(.f64, f64_edges[(k + i / 3 + 1) % f64_edges.len]);
                    },
                }
            }
            sl[i] = im;
        }
        try checkAll(&f, &.{&.{p}}, &mem, &sl, &masks);
    };
}

test "a64 codegen: splat and extract on every lane" {
    if (!can_run) return error.SkipZigTest;
    var mem: [16]u64 = undefined;
    const p: i64 = @intCast(@intFromPtr(&mem));
    for ([_]ir.Type{ .v4i32, .v2i64, .v4f32, .v2f64 }) |tp| for (0..tp.lanes()) |lane| {
        var buf: [1024]u8 = undefined;
        const lt = tp.lane();
        const src = try std.fmt.bufPrint(&buf,
            \\func f 2
            \\e:
            \\  v2 = load.{0t} v0 0
            \\  v3 = extract.{0t} v2 {1d}
            \\  v4 = splat.{0t} v3
            \\  store.{0t} v0 16 v4
            \\  v5 = load.{2t} v0 32
            \\  v6 = splat.{0t} v5
            \\  store.{0t} v0 48 v6
            \\  v7 = extract.{0t} v6 {1d}
            \\  store.{2t} v0 64 v7
            \\  v8 = const 0
            \\  v9 = copy v8
            \\  v10 = {3s}
            \\  v11 = splat.{0t} v10
            \\  store.{0t} v0 80 v11
            \\  v12 = extract.{0t} v2 {1d}
            \\  store.{2t} v0 96 v12
            \\  v13 = load v0 0
            \\  ret v13
            \\
        , .{ tp, lane, lt, if (lt == .i64) "copy v9" else "load." ++ "f64 v0 32" });
        // the float splat source must have the lane type
        const fixed = if (tp == .v4f32) try std.mem.replaceOwned(u8, t.allocator, src, "v10 = load.f64", "v10 = load.f32") else try t.allocator.dupe(u8, src);
        defer t.allocator.free(fixed);
        var f = try ir.parse(t.allocator, fixed);
        defer f.deinit();
        var images: [8][16]u64 = undefined;
        var sl: [8][]const u64 = undefined;
        for (&images, 0..) |*im, i| {
            @memset(im, 0);
            im[0] = 0x80000001_7fffffff +% i *% 0x123456789;
            im[1] = 0xfffffffe_00000005 ^ (i << 31);
            im[4] = switch (lt) {
                .i64 => 0x1_8000_0001 *% (i + 1),
                .f32 => fbits(.f32, f64_edges[i * 3 % f64_edges.len]),
                else => fbits(.f64, f64_edges[i * 3 % f64_edges.len]),
            };
            sl[i] = im;
        }
        try checkAll(&f, &.{&.{ p, 0 }}, &mem, &sl, &masks);
    };
}

// ---- random differential fuzz ---------------------------------------------------------------

/// JIT-built call targets: `helpers[n]` takes `n` i64 arguments and returns
/// `sum(arg_i << i)` (77 for no arguments), then clobbers every caller-saved GP and FP register
/// so that values the caller keeps there are caught.
const CallHelpers = struct {
    funs: [11]JitRuntime.Function,

    fn init() !CallHelpers {
        var h: CallHelpers = undefined;
        for (&h.funs, 0..) |*fun, n| {
            var w: std.ArrayList(u32) = .empty;
            defer w.deinit(t.allocator);
            if (n == 0) try w.append(t.allocator, enc.movw(enc.movz, 0, 77, 0));
            for (1..@max(n, 1)) |i| {
                var r: u5 = @intCast(i);
                if (i >= 8) {
                    r = 9;
                    try w.append(t.allocator, enc.ldstImm(true, .x, 9, 31, @intCast(i - 8)));
                }
                // add x0, x0, x<r>, lsl #i
                try w.append(t.allocator, 0x8B000000 | @as(u32, r) << 16 | @as(u32, @intCast(i)) << 10);
            }
            for (1..18) |r| try w.append(t.allocator, enc.movw(enc.movn, @intCast(r), @intCast(r * 0x111), 0));
            for (0..32) |v| if (v < 8 or v >= 16) try w.append(t.allocator, enc.dupGp(true, @intCast(v), 9));
            try w.append(t.allocator, enc.ret);
            var cb = CodeBuffer.init(t.allocator);
            defer cb.deinit();
            try cb.emitBytes(std.mem.sliceAsBytes(w.items));
            try cb.finalize();
            fun.* = try JitRuntime.add(&cb);
        }
        return h;
    }
    fn deinit(h: *CallHelpers) void {
        for (&h.funs) |*fun| fun.release();
    }
};

/// Vreg pools of the generated programs.
const Pool = struct {
    lo: ir.VReg,
    hi: ir.VReg, // exclusive
};
const pool_int: Pool = .{ .lo = 3, .hi = 13 }; // writable i64
const pool_div: Pool = .{ .lo = 13, .hi = 16 }; // const-only, never 0 or -1 (sdiv/srem divisors)
const pool_k: Pool = .{ .lo = 16, .hi = 19 }; // const-only
fn poolOf(tp: ir.Type) Pool {
    return switch (tp) {
        .i64 => pool_int,
        .f64 => .{ .lo = 19, .hi = 23 },
        .f32 => .{ .lo = 23, .hi = 27 },
        .v4i32 => .{ .lo = 27, .hi = 30 },
        .v2i64 => .{ .lo = 30, .hi = 33 },
        .v4f32 => .{ .lo = 33, .hi = 36 },
        .v2f64 => .{ .lo = 36, .hi = 39 },
    };
}
const fuzz_mem_words = 64;

const Gen = struct {
    f: *ir.Func,
    r: std.Random,
    helpers: *const CallHelpers,
    calls: bool,

    fn pick(g: *Gen, p: Pool) ir.VReg {
        return g.r.intRangeLessThan(ir.VReg, p.lo, p.hi);
    }
    /// Any i64 value: parameters (incl. the base pointer), writable and constant vregs.
    fn anyInt(g: *Gen) ir.VReg {
        return switch (g.r.uintLessThan(u8, 10)) {
            0 => g.r.uintLessThan(ir.VReg, 3),
            1, 2 => g.pick(pool_div),
            3, 4 => g.pick(pool_k),
            else => g.pick(pool_int),
        };
    }
    fn konst(g: *Gen, div: bool) i64 {
        while (true) {
            const k = switch (g.r.uintLessThan(u8, 4)) {
                0 => g.r.int(i64),
                1 => g.r.intRangeAtMost(i64, -5000, 5000),
                else => int_edges[g.r.uintLessThan(usize, int_edges.len)],
            };
            if (!div or (k != 0 and k != -1)) return k;
        }
    }
    fn disp(g: *Gen, tp: ir.Type) i32 {
        const max = fuzz_mem_words * 8 - @as(i32, tp.size());
        const d: i32 = if (g.r.uintLessThan(u8, 8) == 0) g.r.intRangeAtMost(i32, 0, max) else 8 * g.r.intRangeAtMost(i32, 0, @divFloor(max, 8));
        return d;
    }
    fn typedOk(x: ir.TArith, tp: ir.Type) bool {
        return switch (x) {
            .add, .sub, .mul => true,
            .div, .min, .max => tp.isFloat(),
            .@"and", .@"or", .xor => tp.isVector() and !tp.isFloat(),
        };
    }

    fn op(g: *Gen, b: u32) !void {
        const tps = [_]ir.Type{ .f32, .f64, .v4i32, .v2i64, .v4f32, .v2f64 };
        const tp = tps[g.r.uintLessThan(usize, tps.len)];
        const e = g.f;
        switch (g.r.uintLessThan(u8, 22)) {
            0 => try e.emit(b, .{ .@"const" = .{ .dst = g.pick(pool_int), .imm = g.konst(false) } }),
            1 => if (g.r.boolean())
                try e.emit(b, .{ .@"const" = .{ .dst = g.pick(pool_k), .imm = g.konst(false) } })
            else
                try e.emit(b, .{ .@"const" = .{ .dst = g.pick(pool_div), .imm = g.konst(true) } }),
            2 => try e.emit(b, .{ .copy = .{ .dst = g.pick(pool_int), .src = g.anyInt() } }),
            3, 4, 5, 6 => {
                const a = std.enums.values(ir.Arith)[g.r.uintLessThan(usize, 11)];
                const div = a == .sdiv or a == .srem;
                try e.emit(b, .{ .arith = .{ .op = a, .dst = g.pick(pool_int), .a = g.anyInt(), .b = if (div) g.pick(pool_div) else g.anyInt() } });
            },
            7 => try e.emit(b, .{ .load = .{ .dst = g.pick(pool_int), .base = 0, .disp = g.disp(.i64) } }),
            8 => try e.emit(b, .{ .store = .{ .base = 0, .disp = g.disp(.i64), .src = g.anyInt() } }),
            9 => try e.emit(b, .{ .load = .{ .dst = g.pick(poolOf(tp)), .base = 0, .disp = g.disp(tp), .ty = tp } }),
            10 => try e.emit(b, .{ .store = .{ .base = 0, .disp = g.disp(tp), .src = g.pick(poolOf(tp)), .ty = tp } }),
            11, 12, 13 => {
                var a: ir.TArith = undefined;
                while (true) {
                    a = std.enums.values(ir.TArith)[g.r.uintLessThan(usize, 9)];
                    if (typedOk(a, tp)) break;
                }
                try e.emit(b, .{ .typed = .{ .op = a, .ty = tp, .dst = g.pick(poolOf(tp)), .a = g.pick(poolOf(tp)), .b = g.pick(poolOf(tp)) } });
            },
            14 => {
                const ft: ir.Type = if (g.r.boolean()) .f32 else .f64;
                const x = f64_edges[g.r.uintLessThan(usize, f64_edges.len)];
                const v = if (g.r.boolean()) x else @as(f64, @floatFromInt(g.r.intRangeAtMost(i32, -40, 40))) / 8;
                try e.emit(b, .{ .fconst = .{ .dst = g.pick(poolOf(ft)), .ty = ft, .bits = fbits(ft, v) } });
            },
            15 => {
                const ft: ir.Type = if (g.r.boolean()) .f32 else .f64;
                const c = std.enums.values(ir.FCond)[g.r.uintLessThan(usize, 6)];
                try e.emit(b, .{ .fcmp = .{ .cond = c, .ty = ft, .dst = g.pick(pool_int), .a = g.pick(poolOf(ft)), .b = g.pick(poolOf(ft)) } });
            },
            16 => {
                const sc = [_]ir.Type{ .i64, .f32, .f64 };
                const to = sc[g.r.uintLessThan(usize, 3)];
                var from = sc[g.r.uintLessThan(usize, 3)];
                if (from == to) from = if (to == .i64) .f64 else .i64;
                const src = if (from == .i64) g.anyInt() else g.pick(poolOf(from));
                try e.emit(b, .{ .cvt = .{ .to = to, .from = from, .dst = g.pick(poolOf(to)), .src = src } });
            },
            17 => {
                const vt = tps[2 + g.r.uintLessThan(usize, 4)];
                const src = if (vt.lane() == .i64) g.anyInt() else g.pick(poolOf(vt.lane()));
                try e.emit(b, .{ .splat = .{ .ty = vt, .dst = g.pick(poolOf(vt)), .src = src } });
            },
            18 => {
                const vt = tps[2 + g.r.uintLessThan(usize, 4)];
                try e.emit(b, .{ .extract = .{ .ty = vt, .dst = g.pick(poolOf(vt.lane())), .src = g.pick(poolOf(vt)), .lane = g.r.uintLessThan(u8, vt.lanes()) } });
            },
            else => if (g.calls) {
                const n = g.r.uintAtMost(usize, 10);
                const args = try e.alloc().alloc(ir.VReg, n);
                for (args) |*a| a.* = g.anyInt();
                try e.emit(b, .{ .call = .{ .dst = g.pick(pool_int), .addr = @intFromPtr(g.helpers.funs[n].mem.ptr), .args = args } });
            } else try e.emit(b, .{ .arith = .{ .op = .add, .dst = g.pick(pool_int), .a = g.anyInt(), .b = g.anyInt() } }),
        }
    }
};

fn genProgram(f: *ir.Func, r: std.Random, helpers: *const CallHelpers) !void {
    const names = [_][]const u8{ "b0", "b1", "b2", "b3", "b4", "b5", "b6", "b7" };
    var g: Gen = .{ .f = f, .r = r, .helpers = helpers, .calls = r.uintLessThan(u8, 3) != 0 };
    f.params = 3;
    const nb = r.intRangeAtMost(u32, 1, names.len);
    for (names[0..nb]) |n| _ = try f.addBlock(n);
    // Entry: constants, typed pools from memory, some (not all) writable ints.
    for (pool_div.lo..pool_div.hi) |v| try f.emit(0, .{ .@"const" = .{ .dst = @intCast(v), .imm = g.konst(true) } });
    for (pool_k.lo..pool_k.hi) |v| try f.emit(0, .{ .@"const" = .{ .dst = @intCast(v), .imm = g.konst(false) } });
    for ([_]ir.Type{ .f64, .f32, .v4i32, .v2i64, .v4f32, .v2f64 }) |tp| {
        const p = poolOf(tp);
        for (p.lo..p.hi) |v| try f.emit(0, .{ .load = .{ .dst = @intCast(v), .base = 0, .disp = g.disp(tp), .ty = tp } });
    }
    for (pool_int.lo..pool_int.hi) |v| if (r.boolean())
        try f.emit(0, .{ .load = .{ .dst = @intCast(v), .base = 0, .disp = g.disp(.i64) } });
    for (0..nb) |bi| {
        const b: u32 = @intCast(bi);
        for (0..r.intRangeAtMost(usize, 1, 24)) |_| try g.op(b);
        const last = b + 1 == nb;
        switch (if (last) 0 else r.uintLessThan(u8, 6)) {
            0 => {
                // Usually observe every pool: typed values go to memory, ints fold into the result.
                if (r.uintLessThan(u8, 4) != 0) {
                    var d: i32 = 8 * r.intRangeAtMost(i32, 0, 32);
                    for ([_]ir.Type{ .f64, .f32, .v4i32, .v2i64, .v4f32, .v2f64 }) |tp| {
                        const pl = poolOf(tp);
                        for (pl.lo..pl.hi) |v| {
                            try f.emit(b, .{ .store = .{ .base = 0, .disp = d, .src = @intCast(v), .ty = tp } });
                            d += tp.size();
                        }
                    }
                    const acc = g.pick(pool_int);
                    for (pool_int.lo..pool_k.hi) |v| if (v != acc)
                        try f.emit(b, .{ .arith = .{ .op = if (v % 2 == 0) .xor else .add, .dst = acc, .a = acc, .b = @intCast(v) } });
                    try f.emit(b, .{ .ret = acc });
                } else try f.emit(b, .{ .ret = g.anyInt() });
            },
            1 => try f.emit(b, .{ .br = r.intRangeLessThan(u32, b + 1, nb) }),
            else => try f.emit(b, .{ .cbr = .{
                .cond = std.enums.values(ir.CondCode)[r.uintLessThan(usize, 10)],
                .a = g.anyInt(),
                .b = g.anyInt(),
                .then = r.intRangeLessThan(u32, b + 1, nb),
                .@"else" = r.intRangeLessThan(u32, b + 1, nb),
            } }),
        }
    }
}

test "a64 codegen: random differential fuzz (vectors, calls, spills)" {
    if (!can_run) return error.SkipZigTest;
    var helpers = try CallHelpers.init();
    defer helpers.deinit();
    var prng = std.Random.DefaultPrng.init(0xa64);
    const r = prng.random();
    var mem: [fuzz_mem_words]u64 = undefined;
    var images: [3][fuzz_mem_words]u64 = undefined;
    for (&images) |*im| for (im) |*w| {
        w.* = switch (r.uintLessThan(u8, 5)) {
            0 => @bitCast(int_edges[r.uintLessThan(usize, int_edges.len)]),
            1 => fbits(.f64, f64_edges[r.uintLessThan(usize, f64_edges.len)]),
            2 => fbits(.f32, f64_edges[r.uintLessThan(usize, f64_edges.len)]) | fbits(.f32, f64_edges[r.uintLessThan(usize, f64_edges.len)]) << 32,
            3 => r.uintLessThan(u64, 100),
            else => r.int(u64),
        };
    };
    const sl = [_][]const u64{ &images[0], &images[1], &images[2] };
    const p: i64 = @intCast(@intFromPtr(&mem));
    const programs = 400;
    for (0..programs) |_| {
        var f = ir.Func.init(t.allocator);
        defer f.deinit();
        try genProgram(&f, r, &helpers);
        const a = int_edges[r.uintLessThan(usize, int_edges.len)];
        const b = int_edges[r.uintLessThan(usize, int_edges.len)];
        try checkAll(&f, &.{ &.{ p, a, b }, &.{ p, b, r.int(i64) }, &.{ p, r.intRangeAtMost(i64, -9, 9), a } }, &mem, &sl, &masks);
    }
}

test "a64 codegen: branches beyond +-1 MiB (cbz, b.cond forward, b.cond backward)" {
    if (!can_run) return error.SkipZigTest;
    var f = ir.Func.init(t.allocator);
    defer f.deinit();
    f.params = 2;
    const e = try f.addBlock("e");
    const head = try f.addBlock("head");
    const big = try f.addBlock("big");
    const tail = try f.addBlock("tail");
    const skip = try f.addBlock("skip");
    const zero = try f.addBlock("zero");
    try f.emit(e, .{ .@"const" = .{ .dst = 3, .imm = 1 } });
    try f.emit(e, .{ .@"const" = .{ .dst = 2, .imm = 0 } });
    try f.emit(e, .{ .@"const" = .{ .dst = 4, .imm = 0 } });
    try f.emit(e, .{ .cbr = .{ .cond = .eq, .a = 0, .b = 4, .then = zero, .@"else" = head } }); // cbz, far
    try f.emit(head, .{ .cbr = .{ .cond = .eq, .a = 0, .b = 1, .then = skip, .@"else" = big } }); // far forward
    for (0..270_000) |_| try f.emit(big, .{ .arith = .{ .op = .add, .dst = 2, .a = 2, .b = 3 } });
    try f.emit(big, .{ .br = tail });
    try f.emit(tail, .{ .arith = .{ .op = .sub, .dst = 0, .a = 0, .b = 3 } });
    try f.emit(tail, .{ .cbr = .{ .cond = .gt, .a = 0, .b = 4, .then = head, .@"else" = skip } }); // far backward
    try f.emit(skip, .{ .arith = .{ .op = .add, .dst = 5, .a = 2, .b = 0 } });
    try f.emit(skip, .{ .ret = 5 });
    try f.emit(zero, .{ .@"const" = .{ .dst = 6, .imm = 12345 } });
    try f.emit(zero, .{ .ret = 6 });
    const code = try codeOf(&f, .{});
    defer t.allocator.free(code);
    try t.expect(code.len * 4 > 1 << 20);
    var none: [0]u64 = .{};
    try checkAll(&f, &.{ &.{ 3, 1 }, &.{ 1, 1 }, &.{ 0, 7 }, &.{ 2, 1 }, &.{ -5, 1 } }, &none, &.{}, masks[0..2]);
}

test "a64 codegen: addressing modes, float immediates and vector ops are single instructions" {
    const Case = struct { src: []const u8, want: []const u32 };
    const cases = [_]Case{
        // scaled unsigned offset, unscaled negative offset, register offset for the rest
        .{ .src = "func f 1\ne:\n  v1 = load v0 32760\n  ret v1\n", .want = &.{ enc.ldstImm(true, .x, 0, 0, 4095), enc.ret } },
        .{ .src = "func f 1\ne:\n  v1 = load v0 -8\n  ret v1\n", .want = &.{ enc.ldstUnscaled(true, .x, 0, 0, -8), enc.ret } },
        .{ .src = "func f 1\ne:\n  v1 = load.v4i32 v0 65520\n  store.v4i32 v0 0 v1\n  ret v0\n", .want = &.{ enc.ldstImm(true, .q, 0, 0, 4095), enc.ldstImm(false, .q, 0, 0, 0), enc.ret } },
        .{ .src = "func f 1\ne:\n  v1 = load.f32 v0 3\n  store.f32 v0 16380 v1\n  ret v0\n", .want = &.{ enc.ldstUnscaled(true, .s, 0, 0, 3), enc.ldstImm(false, .s, 0, 0, 4095), enc.ret } },
        // fmov #imm8 and movi #0 instead of a GP round trip
        .{ .src = "func f 1\ne:\n  v1 = fconst.f64 1.0\n  store.f64 v0 0 v1\n  ret v0\n", .want = &.{ enc.fmovImm(.f64, 0, 0x70), enc.ldstImm(false, .d, 0, 0, 0), enc.ret } },
        .{ .src = "func f 1\ne:\n  v1 = fconst.f32 -0.5\n  store.f32 v0 0 v1\n  ret v0\n", .want = &.{ enc.fmovImm(.f32, 0, 0xE0), enc.ldstImm(false, .s, 0, 0, 0), enc.ret } },
        .{ .src = "func f 1\ne:\n  v1 = fconst.f64 0\n  store.f64 v0 0 v1\n  ret v0\n", .want = &.{ enc.moviZero(0), enc.ldstImm(false, .d, 0, 0, 0), enc.ret } },
        // three-address vector arithmetic
        .{ .src = "func f 1\ne:\n  v1 = load.v4i32 v0 0\n  v2 = load.v4i32 v0 16\n  v3 = add.v4i32 v1 v2\n  store.v4i32 v0 32 v3\n  ret v0\n", .want = &.{ enc.ldstImm(true, .q, 0, 0, 0), enc.ldstImm(true, .q, 1, 0, 1), enc.rrr(enc.add_4s, 0, 0, 1), enc.ldstImm(false, .q, 0, 0, 2), enc.ret } },
        .{ .src = "func f 1\ne:\n  v1 = load.v2f64 v0 0\n  v2 = load.v2f64 v0 16\n  v3 = div.v2f64 v1 v2\n  store.v2f64 v0 32 v3\n  ret v0\n", .want = &.{ enc.ldstImm(true, .q, 0, 0, 0), enc.ldstImm(true, .q, 1, 0, 1), enc.rrr(enc.fdiv_4s | enc.sz_2d, 0, 0, 1), enc.ldstImm(false, .q, 0, 0, 2), enc.ret } },
    };
    for (cases) |cs| {
        const c = try codeOfSrc(cs.src);
        defer t.allocator.free(c);
        t.expectEqualSlices(u32, cs.want, c) catch |err| {
            std.debug.print("source:\n{s}\n", .{cs.src});
            return err;
        };
    }
}
