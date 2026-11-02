//! RISC-V RV64IMA Assembler (TODO-ARCHES.md §2, base step). Fixed 32-bit little-endian words,
//! direct emission into a CodeBuffer like the other backends. Labels use the new
//! `rv_b12` (conditional branches, ±4 KiB) and `rv_j20` (jal, ±1 MiB) fixups.
//!
//! Covered: RV64I integer ops (R/I/W forms), loads/stores, lui/auipc, jal/jalr, branches,
//! M extension (mul/div/rem + W forms), pseudo-instructions `li` (any 64-bit constant,
//! lui/addiw/slli/addi chains), `mv`, `nop`, `ret`, `j`, `callAbs`, and an RV64 prolog/epilog.
//! A extension (RV64A): `lr`/`sc` and all AMOs for .w/.d with aq/rl ordering bits, `fence`.
//! F/D extensions: flw/fld/fsw/fsd, add/sub/mul/div/sqrt/min/max/sgnj, fused multiply-add,
//! compares, int<->fp conversions, fmv between register files, s<->d conversion.
//! Far control flow: `callFar`/`jFar`/`la` (auipc pairs, ±2 GiB, fixup `rv_pcrel_pair`) and
//! `bFar` (inverted branch over `jal`, ±1 MiB).
//! C extension (explicit `c*` emitters): CI/CR (c.nop/addi/addiw/li/lui/addi16sp/slli/mv/add/jr/
//! jalr/ret/ebreak), stack-relative (c.lwsp/swsp/ldsp/sdsp), CIW c.addi4spn, CL/CS on x8..x15
//! (c.lw/sw/ld/sd), CB (c.srli/srai/andi, c.beqz/bnez with label fixup `rv_cb9`), CA
//! (c.sub/xor/or/and/subw/addw), CJ (c.j, RV32 c.jal; fixup `rv_cj12`).
//! Auto-compress: `AssemblerT(.{ .compress = true })` makes the regular emitters pick the
//! 16-bit form when one encodes the same instruction (label branches stay 32-bit).
//! RV32 mode: `AssemblerT(.{ .xlen = .rv32 })` (comptime). Shift amounts are limited to 0..31,
//! RV64-only instructions (W forms, ld/sd/lwu, .d atomics, fcvt l/lu, fmv.x.d/fmv.d.x,
//! c.addiw/c.ldsp/c.sdsp) are compile errors, `li` uses lui+addi, and prolog/epilog use
//! 4-byte sw/lw slots (ILP32). `c.lwsp`/`c.swsp` are available in both modes.
const std = @import("std");
const core = @import("../core/code_buffer.zig");
const CodeBuffer = core.CodeBuffer;
pub const Label = core.Label;
pub const Error = core.Error || error{InvalidOperand};

/// Atomic operand width: `.w` (32-bit, sign-extended result) or `.d` (64-bit).
pub const Width = enum(u3) { w = 2, d = 3 };
/// Memory ordering bits of A-extension instructions.
pub const Order = enum(u2) { none = 0, rl = 1, aq = 2, aqrl = 3 };
/// AMO operations (funct5).
pub const Amo = enum(u5) {
    add = 0b00000,
    swap = 0b00001,
    xor = 0b00100,
    @"or" = 0b01000,
    @"and" = 0b01100,
    min = 0b10000,
    max = 0b10100,
    minu = 0b11000,
    maxu = 0b11100,
};

/// x0..x31 with ABI names.
pub const X = enum(u5) {
    zero,
    ra,
    sp,
    gp,
    tp,
    t0,
    t1,
    t2,
    s0,
    s1,
    a0,
    a1,
    a2,
    a3,
    a4,
    a5,
    a6,
    a7,
    s2,
    s3,
    s4,
    s5,
    s6,
    s7,
    s8,
    s9,
    s10,
    s11,
    t3,
    t4,
    t5,
    t6,
    inline fn n(r: X) u32 {
        return @backingInt(r);
    }
};

pub const Cond = enum(u3) { eq = 0, ne = 1, lt = 4, ge = 5, ltu = 6, geu = 7 };

/// LP64 calling convention.
pub const CallConv = struct {
    pub const args = [_]X{ .a0, .a1, .a2, .a3, .a4, .a5, .a6, .a7 };
    pub const ret: X = .a0;
    pub const callee_saved = [_]X{ .s0, .s1, .s2, .s3, .s4, .s5, .s6, .s7, .s8, .s9, .s10, .s11 };
};

fn fitsI12(v: i64) bool {
    return v >= -2048 and v <= 2047;
}

/// Base integer register width.
pub const Xlen = enum { rv32, rv64 };

pub const Options = struct {
    validate: bool = true,
    xlen: Xlen = .rv64,
    /// Emit 16-bit C forms from the regular emitters where possible.
    compress: bool = false,
};

/// f0..f31 (floating-point registers; ABI names ft0.., fa0.. are f0.., f10..).
pub const F = enum(u5) {
    f0,
    f1,
    f2,
    f3,
    f4,
    f5,
    f6,
    f7,
    f8,
    f9,
    f10,
    f11,
    f12,
    f13,
    f14,
    f15,
    f16,
    f17,
    f18,
    f19,
    f20,
    f21,
    f22,
    f23,
    f24,
    f25,
    f26,
    f27,
    f28,
    f29,
    f30,
    f31,
    pub fn n(r: F) u32 {
        return @backingInt(r);
    }
};
/// Floating-point format field: single or double.
pub const Fmt = enum(u2) { s = 0, d = 1 };
/// Rounding mode (`dyn` = use fcsr.frm).
pub const Rm = enum(u3) { rne = 0, rtz = 1, rdn = 2, rup = 3, rmm = 4, dyn = 7 };
pub const FArith = enum(u5) { add = 0b00000, sub = 0b00001, mul = 0b00010, div = 0b00011 };
pub const FCmp = enum(u3) { le = 0, lt = 1, eq = 2 };
/// Integer side of fcvt: w (i32), wu (u32), l (i64), lu (u64).
pub const IntKind = enum(u5) { w = 0, wu = 1, l = 2, lu = 3 };

pub fn AssemblerT(comptime opts: Options) type {
    return struct {
        const Self = @This();
        buf: *CodeBuffer,
        pub const xlen = opts.xlen;
        const is64 = opts.xlen == .rv64;
        /// Register width in bytes (8 on RV64, 4 on RV32).
        pub const xlen_bytes: u32 = if (is64) 8 else 4;
        const max_shamt: u32 = if (is64) 63 else 31;
        fn need64(comptime what: []const u8) void {
            if (!is64) @compileError(what ++ " is RV64-only");
        }

        pub fn init(buf: *CodeBuffer) Self {
            return .{ .buf = buf };
        }
        pub fn newLabel(a: Self) Error!Label {
            return a.buf.newLabel();
        }
        pub fn bind(a: Self, l: Label) Error!void {
            return a.buf.bind(l);
        }
        // ---- C extension (RV64C subset): explicit 16-bit forms, never chosen implicitly ----
        fn half(a: Self, h: u16) Error!void {
            var hb: [2]u8 = undefined;
            std.mem.writeInt(u16, &hb, h, .little);
            return a.buf.emitBytes(&hb);
        }
        fn imm6(v: i64) Error!u16 {
            if (v < -32 or v > 31) return error.InvalidOperand;
            return @truncate(@as(u64, @bitCast(v)) & 0x3F);
        }
        fn ci(a: Self, f3: u16, rd: X, imm: u16, op: u16) Error!void {
            return a.half(f3 << 13 | (imm >> 5 & 1) << 12 | @as(u16, @intCast(rd.n())) << 7 | (imm & 0x1F) << 2 | op);
        }
        /// `c.nop`.
        pub fn cNop(a: Self) Error!void {
            return a.half(0x0001);
        }
        /// `c.addi rd, imm` (rd != zero, imm != 0, -32..31).
        pub fn cAddi(a: Self, rd: X, imm: i64) Error!void {
            if (rd == .zero or imm == 0) return error.InvalidOperand;
            return a.ci(0b000, rd, try imm6(imm), 0b01);
        }
        /// `c.addiw rd, imm` (rd != zero, -32..31).
        pub fn cAddiw(a: Self, rd: X, imm: i64) Error!void {
            comptime need64("cAddiw");
            if (rd == .zero) return error.InvalidOperand;
            return a.ci(0b001, rd, try imm6(imm), 0b01);
        }
        /// `c.li rd, imm` (rd != zero, -32..31).
        pub fn cLi(a: Self, rd: X, imm: i64) Error!void {
            if (rd == .zero) return error.InvalidOperand;
            return a.ci(0b010, rd, try imm6(imm), 0b01);
        }
        /// `c.slli rd, shamt` (rd != zero, 1..63).
        pub fn cSlli(a: Self, rd: X, shamt: u6) Error!void {
            if (rd == .zero or shamt == 0 or shamt > max_shamt) return error.InvalidOperand;
            return a.ci(0b000, rd, shamt, 0b10);
        }
        /// `c.mv rd, rs2` (both != zero).
        pub fn cMv(a: Self, rd: X, rs2: X) Error!void {
            if (rd == .zero or rs2 == .zero) return error.InvalidOperand;
            return a.half(0b1000 << 12 | @as(u16, @intCast(rd.n())) << 7 | @as(u16, @intCast(rs2.n())) << 2 | 0b10);
        }
        /// `c.add rd, rs2` (both != zero).
        pub fn cAdd(a: Self, rd: X, rs2: X) Error!void {
            if (rd == .zero or rs2 == .zero) return error.InvalidOperand;
            return a.half(0b1001 << 12 | @as(u16, @intCast(rd.n())) << 7 | @as(u16, @intCast(rs2.n())) << 2 | 0b10);
        }
        /// `c.jr rs1` (rs1 != zero); `c.jr ra` is `c.ret`.
        pub fn cJr(a: Self, rs1: X) Error!void {
            if (rs1 == .zero) return error.InvalidOperand;
            return a.half(0b1000 << 12 | @as(u16, @intCast(rs1.n())) << 7 | 0b10);
        }
        pub fn cRet(a: Self) Error!void {
            return a.cJr(.ra);
        }
        /// `c.ldsp rd, off(sp)` (rd != zero, off multiple of 8, 0..504).
        pub fn cLdsp(a: Self, rd: X, off: u32) Error!void {
            comptime need64("cLdsp");
            if (rd == .zero or off % 8 != 0 or off > 504) return error.InvalidOperand;
            const o: u16 = @intCast(off);
            return a.half(0b011 << 13 | (o >> 5 & 1) << 12 | @as(u16, @intCast(rd.n())) << 7 | (o >> 3 & 3) << 5 | (o >> 6 & 7) << 2 | 0b10);
        }
        /// `c.sdsp rs2, off(sp)` (off multiple of 8, 0..504).
        /// `c.lwsp rd, off(sp)` (rd != zero, off multiple of 4, 0..252).
        pub fn cLwsp(a: Self, rd: X, off: u32) Error!void {
            if (rd == .zero or off % 4 != 0 or off > 252) return error.InvalidOperand;
            const o: u16 = @intCast(off);
            return a.half(0b010 << 13 | (o >> 5 & 1) << 12 | @as(u16, @intCast(rd.n())) << 7 | (o >> 2 & 7) << 4 | (o >> 6 & 3) << 2 | 0b10);
        }
        /// `c.swsp rs2, off(sp)` (off multiple of 4, 0..252).
        pub fn cSwsp(a: Self, rs2: X, off: u32) Error!void {
            if (off % 4 != 0 or off > 252) return error.InvalidOperand;
            const o: u16 = @intCast(off);
            return a.half(0b110 << 13 | (o >> 2 & 0xF) << 9 | (o >> 6 & 3) << 7 | @as(u16, @intCast(rs2.n())) << 2 | 0b10);
        }
        pub fn cSdsp(a: Self, rs2: X, off: u32) Error!void {
            comptime need64("cSdsp");
            if (off % 8 != 0 or off > 504) return error.InvalidOperand;
            const o: u16 = @intCast(off);
            return a.half(0b111 << 13 | (o >> 3 & 7) << 10 | (o >> 6 & 7) << 7 | @as(u16, @intCast(rs2.n())) << 2 | 0b10);
        }

        /// 3-bit register field of x8..x15 (s0, s1, a0..a5).
        fn creg(r: X) Error!u16 {
            const v = r.n();
            if (v < 8 or v > 15) return error.InvalidOperand;
            return @intCast(v - 8);
        }
        fn isC(r: X) bool {
            return r.n() >= 8 and r.n() <= 15;
        }
        /// `c.lui rd, imm` (rd != zero/sp, imm6 nonzero; rd = sext(imm) << 12).
        pub fn cLui(a: Self, rd: X, imm: i64) Error!void {
            if (rd == .zero or rd == .sp or imm == 0) return error.InvalidOperand;
            return a.ci(0b011, rd, try imm6(imm), 0b01);
        }
        /// `c.addi16sp imm` (multiple of 16, -512..496, nonzero).
        pub fn cAddi16sp(a: Self, imm: i64) Error!void {
            if (imm == 0 or @mod(imm, 16) != 0 or imm < -512 or imm > 496) return error.InvalidOperand;
            const v: u16 = @truncate(@as(u64, @bitCast(imm)));
            return a.half(0b011 << 13 | (v >> 9 & 1) << 12 | 2 << 7 | (v >> 4 & 1) << 6 | (v >> 6 & 1) << 5 | (v >> 7 & 3) << 3 | (v >> 5 & 1) << 2 | 0b01);
        }
        /// `c.addi4spn rd', sp, imm` (multiple of 4, 4..1020).
        pub fn cAddi4spn(a: Self, rd: X, imm: u32) Error!void {
            if (imm == 0 or imm % 4 != 0 or imm > 1020) return error.InvalidOperand;
            const v: u16 = @intCast(imm);
            return a.half((v >> 4 & 3) << 11 | (v >> 6 & 0xF) << 7 | (v >> 2 & 1) << 6 | (v >> 3 & 1) << 5 | try creg(rd) << 2);
        }
        fn clcs(a: Self, f3: u16, r: X, off: u32, base: X, comptime dbl: bool) Error!void {
            const lim: u32 = if (dbl) 248 else 124;
            if (off % (if (dbl) 8 else 4) != 0 or off > lim) return error.InvalidOperand;
            const o: u16 = @intCast(off);
            const lo: u16 = if (dbl) (o >> 6 & 3) << 5 else (o >> 2 & 1) << 6 | (o >> 6 & 1) << 5;
            return a.half(f3 << 13 | (o >> 3 & 7) << 10 | try creg(base) << 7 | lo | try creg(r) << 2);
        }
        /// `c.lw rd', off(rs1')` (off multiple of 4, 0..124).
        pub fn cLw(a: Self, rd: X, off: u32, rs1: X) Error!void {
            return a.clcs(0b010, rd, off, rs1, false);
        }
        /// `c.sw rs2', off(rs1')`.
        pub fn cSw(a: Self, rs2: X, off: u32, rs1: X) Error!void {
            return a.clcs(0b110, rs2, off, rs1, false);
        }
        /// `c.ld rd', off(rs1')` (RV64; off multiple of 8, 0..248).
        pub fn cLd(a: Self, rd: X, off: u32, rs1: X) Error!void {
            comptime need64("cLd");
            return a.clcs(0b011, rd, off, rs1, true);
        }
        /// `c.sd rs2', off(rs1')` (RV64).
        pub fn cSd(a: Self, rs2: X, off: u32, rs1: X) Error!void {
            comptime need64("cSd");
            return a.clcs(0b111, rs2, off, rs1, true);
        }
        fn cb(a: Self, f2: u16, rd: X, imm: u16) Error!void {
            return a.half(0b100 << 13 | (imm >> 5 & 1) << 12 | f2 << 10 | try creg(rd) << 7 | (imm & 0x1F) << 2 | 0b01);
        }
        /// `c.srli rd', shamt` (1..63; RV32 1..31).
        pub fn cSrli(a: Self, rd: X, shamt: u32) Error!void {
            if (shamt == 0 or shamt > max_shamt) return error.InvalidOperand;
            return a.cb(0b00, rd, @intCast(shamt));
        }
        /// `c.srai rd', shamt`.
        pub fn cSrai(a: Self, rd: X, shamt: u32) Error!void {
            if (shamt == 0 or shamt > max_shamt) return error.InvalidOperand;
            return a.cb(0b01, rd, @intCast(shamt));
        }
        /// `c.andi rd', imm` (-32..31).
        pub fn cAndi(a: Self, rd: X, imm: i64) Error!void {
            return a.cb(0b10, rd, try imm6(imm));
        }
        fn ca(a: Self, hi: u16, rd: X, f2: u16, rs2: X) Error!void {
            return a.half(hi << 10 | try creg(rd) << 7 | f2 << 5 | try creg(rs2) << 2 | 0b01);
        }
        /// `c.sub/xor/or/and rd', rs2'` (rd' = rd' op rs2').
        pub fn cSub(a: Self, rd: X, rs2: X) Error!void {
            return a.ca(0b100011, rd, 0b00, rs2);
        }
        pub fn cXor(a: Self, rd: X, rs2: X) Error!void {
            return a.ca(0b100011, rd, 0b01, rs2);
        }
        pub fn cOr(a: Self, rd: X, rs2: X) Error!void {
            return a.ca(0b100011, rd, 0b10, rs2);
        }
        pub fn cAnd(a: Self, rd: X, rs2: X) Error!void {
            return a.ca(0b100011, rd, 0b11, rs2);
        }
        /// `c.subw/c.addw rd', rs2'` (RV64).
        pub fn cSubw(a: Self, rd: X, rs2: X) Error!void {
            comptime need64("cSubw");
            return a.ca(0b100111, rd, 0b00, rs2);
        }
        pub fn cAddw(a: Self, rd: X, rs2: X) Error!void {
            comptime need64("cAddw");
            return a.ca(0b100111, rd, 0b01, rs2);
        }
        /// `c.jalr rs1` (rs1 != zero; links ra).
        pub fn cJalr(a: Self, rs1: X) Error!void {
            if (rs1 == .zero) return error.InvalidOperand;
            return a.half(0b1001 << 12 | @as(u16, @intCast(rs1.n())) << 7 | 0b10);
        }
        pub fn cEbreak(a: Self) Error!void {
            return a.half(0x9002);
        }
        fn cBranchTo(a: Self, h: u16, comptime kind: core.FixupKind, l: Label) Error!void {
            if (!a.buf.isValidLabel(l)) return error.InvalidLabel;
            const at = a.buf.offset();
            if (a.buf.labelOffset(l)) |target| {
                const d: i64 = @as(i64, target) - at;
                const lim: i64 = if (kind == .rv_cb9) 1 << 8 else 1 << 11;
                if (d < -lim or d >= lim) return error.DisplacementOutOfRange;
                return a.half(if (kind == .rv_cb9) core.rvPatchCB(h, d) else core.rvPatchCJ(h, d));
            }
            try a.half(h);
            try a.buf.addFixup(.{ .at = at, .base = at, .label = l.id, .kind = kind });
        }
        /// `c.beqz rs1', label` (±256 B).
        pub fn cBeqz(a: Self, rs1: X, l: Label) Error!void {
            return a.cBranchTo(0b110 << 13 | try creg(rs1) << 7 | 0b01, .rv_cb9, l);
        }
        /// `c.bnez rs1', label` (±256 B).
        pub fn cBnez(a: Self, rs1: X, l: Label) Error!void {
            return a.cBranchTo(0b111 << 13 | try creg(rs1) << 7 | 0b01, .rv_cb9, l);
        }
        /// `c.j label` (±2 KiB).
        pub fn cJ(a: Self, l: Label) Error!void {
            return a.cBranchTo(0b101 << 13 | 0b01, .rv_cj12, l);
        }
        /// `c.jal label` (RV32 only; links ra, ±2 KiB).
        pub fn cJal(a: Self, l: Label) Error!void {
            if (is64) @compileError("cJal is RV32-only (the encoding is c.addiw on RV64)");
            return a.cBranchTo(0b001 << 13 | 0b01, .rv_cj12, l);
        }

        fn word(a: Self, w: u32) Error!void {
            try a.buf.ensure(4);
            a.buf.put32(w);
        }
        fn imm12(v: i64) Error!u32 {
            if (opts.validate and !fitsI12(v)) return error.InvalidOperand;
            return @as(u32, @truncate(@as(u64, @bitCast(v)))) & 0xFFF;
        }

        /// Auto-compress helpers: null = no 16-bit form, use the 32-bit encoding.
        fn tryCompressAddi(a: Self, rd: X, rs1: X, imm: i64) ?Error!void {
            if (rd == .zero) return if (rs1 == .zero and imm == 0) a.cNop() else null;
            if (rs1 == .zero and imm >= -32 and imm <= 31) return a.cLi(rd, imm);
            if (imm == 0) return if (rs1 != .zero) a.cMv(rd, rs1) else null;
            if (rd == rs1 and imm >= -32 and imm <= 31) return a.cAddi(rd, imm);
            if (rd == .sp and rs1 == .sp and @mod(imm, 16) == 0 and imm >= -512 and imm <= 496) return a.cAddi16sp(imm);
            if (rs1 == .sp and isC(rd) and imm > 0 and imm <= 1020 and @mod(imm, 4) == 0) return a.cAddi4spn(rd, @intCast(imm));
            return null;
        }
        fn tryCompressMem(a: Self, r: X, off: i64, base: X, comptime size: u32, comptime store: bool) ?Error!void {
            if (off < 0 or @mod(off, size) != 0) return null;
            const o: u32 = @intCast(@min(off, 1 << 20));
            if (base == .sp and (store or r != .zero) and o <= 63 * size) {
                if (size == 8) return if (store) a.cSdsp(r, o) else a.cLdsp(r, o);
                return if (store) a.cSwsp(r, o) else a.cLwsp(r, o);
            }
            if (isC(base) and isC(r) and o <= 31 * size) {
                if (size == 8) return if (store) a.cSd(r, o, base) else a.cLd(r, o, base);
                return if (store) a.cSw(r, o, base) else a.cLw(r, o, base);
            }
            return null;
        }

        // ---- raw formats ----
        pub fn rType(a: Self, f7: u32, rs2: X, rs1: X, f3: u32, rd: X, op: u32) Error!void {
            return a.word(f7 << 25 | rs2.n() << 20 | rs1.n() << 15 | f3 << 12 | rd.n() << 7 | op);
        }
        pub fn iType(a: Self, imm: i64, rs1: X, f3: u32, rd: X, op: u32) Error!void {
            return a.word(try imm12(imm) << 20 | rs1.n() << 15 | f3 << 12 | rd.n() << 7 | op);
        }
        pub fn sType(a: Self, imm: i64, rs2: X, rs1: X, f3: u32) Error!void {
            const v = try imm12(imm);
            return a.word((v >> 5) << 25 | rs2.n() << 20 | rs1.n() << 15 | f3 << 12 | (v & 0x1F) << 7 | 0x23);
        }

        // ---- RV64I ----
        pub fn add(a: Self, rd: X, rs1: X, rs2: X) Error!void {
            if (opts.compress) if (rd != .zero and rd == rs1 and rs2 != .zero) return a.cAdd(rd, rs2) else if (rd != .zero and rs1 == .zero and rs2 != .zero) return a.cMv(rd, rs2) else if (rd != .zero and rd == rs2 and rs1 != .zero) return a.cAdd(rd, rs1);
            return a.rType(0, rs2, rs1, 0, rd, 0x33);
        }
        pub fn sub(a: Self, rd: X, rs1: X, rs2: X) Error!void {
            if (opts.compress) if (rd == rs1 and isC(rd) and isC(rs2)) return a.cSub(rd, rs2);
            return a.rType(0x20, rs2, rs1, 0, rd, 0x33);
        }
        pub fn sll(a: Self, rd: X, rs1: X, rs2: X) Error!void {
            return a.rType(0, rs2, rs1, 1, rd, 0x33);
        }
        pub fn slt(a: Self, rd: X, rs1: X, rs2: X) Error!void {
            return a.rType(0, rs2, rs1, 2, rd, 0x33);
        }
        pub fn sltu(a: Self, rd: X, rs1: X, rs2: X) Error!void {
            return a.rType(0, rs2, rs1, 3, rd, 0x33);
        }
        pub fn xor(a: Self, rd: X, rs1: X, rs2: X) Error!void {
            if (opts.compress) if (isC(rd) and isC(rs1) and isC(rs2) and (rd == rs1 or rd == rs2)) return a.cXor(rd, if (rd == rs1) rs2 else rs1);
            return a.rType(0, rs2, rs1, 4, rd, 0x33);
        }
        pub fn srl(a: Self, rd: X, rs1: X, rs2: X) Error!void {
            return a.rType(0, rs2, rs1, 5, rd, 0x33);
        }
        pub fn sra(a: Self, rd: X, rs1: X, rs2: X) Error!void {
            return a.rType(0x20, rs2, rs1, 5, rd, 0x33);
        }
        pub fn @"or"(a: Self, rd: X, rs1: X, rs2: X) Error!void {
            if (opts.compress) if (isC(rd) and isC(rs1) and isC(rs2) and (rd == rs1 or rd == rs2)) return a.cOr(rd, if (rd == rs1) rs2 else rs1);
            return a.rType(0, rs2, rs1, 6, rd, 0x33);
        }
        pub fn @"and"(a: Self, rd: X, rs1: X, rs2: X) Error!void {
            if (opts.compress) if (isC(rd) and isC(rs1) and isC(rs2) and (rd == rs1 or rd == rs2)) return a.cAnd(rd, if (rd == rs1) rs2 else rs1);
            return a.rType(0, rs2, rs1, 7, rd, 0x33);
        }
        pub fn addw(a: Self, rd: X, rs1: X, rs2: X) Error!void {
            comptime need64("addw");
            if (opts.compress) if (isC(rd) and isC(rs1) and isC(rs2) and (rd == rs1 or rd == rs2)) return a.cAddw(rd, if (rd == rs1) rs2 else rs1);
            return a.rType(0, rs2, rs1, 0, rd, 0x3B);
        }
        pub fn subw(a: Self, rd: X, rs1: X, rs2: X) Error!void {
            comptime need64("subw");
            if (opts.compress) if (rd == rs1 and isC(rd) and isC(rs2)) return a.cSubw(rd, rs2);
            return a.rType(0x20, rs2, rs1, 0, rd, 0x3B);
        }
        pub fn addi(a: Self, rd: X, rs1: X, imm: i64) Error!void {
            if (opts.compress) if (a.tryCompressAddi(rd, rs1, imm)) |r| return r;
            return a.iType(imm, rs1, 0, rd, 0x13);
        }
        pub fn slti(a: Self, rd: X, rs1: X, imm: i64) Error!void {
            return a.iType(imm, rs1, 2, rd, 0x13);
        }
        pub fn sltiu(a: Self, rd: X, rs1: X, imm: i64) Error!void {
            return a.iType(imm, rs1, 3, rd, 0x13);
        }
        pub fn xori(a: Self, rd: X, rs1: X, imm: i64) Error!void {
            return a.iType(imm, rs1, 4, rd, 0x13);
        }
        pub fn ori(a: Self, rd: X, rs1: X, imm: i64) Error!void {
            return a.iType(imm, rs1, 6, rd, 0x13);
        }
        pub fn andi(a: Self, rd: X, rs1: X, imm: i64) Error!void {
            if (opts.compress) if (rd == rs1 and isC(rd) and imm >= -32 and imm <= 31) return a.cAndi(rd, imm);
            return a.iType(imm, rs1, 7, rd, 0x13);
        }
        pub fn addiw(a: Self, rd: X, rs1: X, imm: i64) Error!void {
            comptime need64("addiw");
            if (opts.compress) if (rd == rs1 and rd != .zero and imm >= -32 and imm <= 31) return a.cAddiw(rd, imm);
            return a.iType(imm, rs1, 0, rd, 0x1B);
        }
        fn shiftImm(a: Self, rd: X, rs1: X, shamt: u32, f3: u32, hi: u32) Error!void {
            if (shamt > max_shamt) return error.InvalidOperand;
            return a.word(hi << 26 | shamt << 20 | rs1.n() << 15 | f3 << 12 | rd.n() << 7 | 0x13);
        }
        pub fn slli(a: Self, rd: X, rs1: X, shamt: u32) Error!void {
            if (opts.compress) if (rd == rs1 and rd != .zero and shamt != 0 and shamt <= max_shamt) return a.cSlli(rd, @intCast(shamt));
            return a.shiftImm(rd, rs1, shamt, 1, 0);
        }
        pub fn srli(a: Self, rd: X, rs1: X, shamt: u32) Error!void {
            if (opts.compress) if (rd == rs1 and isC(rd) and shamt != 0 and shamt <= max_shamt) return a.cSrli(rd, shamt);
            return a.shiftImm(rd, rs1, shamt, 5, 0);
        }
        pub fn srai(a: Self, rd: X, rs1: X, shamt: u32) Error!void {
            if (opts.compress) if (rd == rs1 and isC(rd) and shamt != 0 and shamt <= max_shamt) return a.cSrai(rd, shamt);
            return a.shiftImm(rd, rs1, shamt, 5, 0x10);
        }
        pub fn lui(a: Self, rd: X, imm20: u32) Error!void {
            if (opts.compress) if (rd != .zero and rd != .sp and imm20 != 0 and (imm20 < 32 or imm20 >= 0xFFFE0)) return a.cLui(rd, @as(i64, @as(i20, @bitCast(@as(u20, @intCast(imm20))))));
            if (opts.validate and imm20 > 0xFFFFF) return error.InvalidOperand;
            return a.word(imm20 << 12 | rd.n() << 7 | 0x37);
        }
        pub fn auipc(a: Self, rd: X, imm20: u32) Error!void {
            if (opts.validate and imm20 > 0xFFFFF) return error.InvalidOperand;
            return a.word(imm20 << 12 | rd.n() << 7 | 0x17);
        }
        // loads / stores: `ld rd, off(rs1)`
        pub fn lb(a: Self, rd: X, off: i64, rs1: X) Error!void {
            return a.iType(off, rs1, 0, rd, 0x03);
        }
        pub fn lh(a: Self, rd: X, off: i64, rs1: X) Error!void {
            return a.iType(off, rs1, 1, rd, 0x03);
        }
        pub fn lw(a: Self, rd: X, off: i64, rs1: X) Error!void {
            if (opts.compress) if (a.tryCompressMem(rd, off, rs1, 4, false)) |r| return r;
            return a.iType(off, rs1, 2, rd, 0x03);
        }
        pub fn ld(a: Self, rd: X, off: i64, rs1: X) Error!void {
            comptime need64("ld");
            if (opts.compress) if (a.tryCompressMem(rd, off, rs1, 8, false)) |r| return r;
            return a.iType(off, rs1, 3, rd, 0x03);
        }
        pub fn lbu(a: Self, rd: X, off: i64, rs1: X) Error!void {
            return a.iType(off, rs1, 4, rd, 0x03);
        }
        pub fn lhu(a: Self, rd: X, off: i64, rs1: X) Error!void {
            return a.iType(off, rs1, 5, rd, 0x03);
        }
        pub fn lwu(a: Self, rd: X, off: i64, rs1: X) Error!void {
            comptime need64("lwu");
            return a.iType(off, rs1, 6, rd, 0x03);
        }
        pub fn sb(a: Self, rs2: X, off: i64, rs1: X) Error!void {
            return a.sType(off, rs2, rs1, 0);
        }
        pub fn sh(a: Self, rs2: X, off: i64, rs1: X) Error!void {
            return a.sType(off, rs2, rs1, 1);
        }
        pub fn sw(a: Self, rs2: X, off: i64, rs1: X) Error!void {
            if (opts.compress) if (a.tryCompressMem(rs2, off, rs1, 4, true)) |r| return r;
            return a.sType(off, rs2, rs1, 2);
        }
        pub fn sd(a: Self, rs2: X, off: i64, rs1: X) Error!void {
            comptime need64("sd");
            if (opts.compress) if (a.tryCompressMem(rs2, off, rs1, 8, true)) |r| return r;
            return a.sType(off, rs2, rs1, 3);
        }
        pub fn jalr(a: Self, rd: X, off: i64, rs1: X) Error!void {
            if (opts.compress) if (off == 0 and rs1 != .zero and (rd == .zero or rd == .ra)) return if (rd == .zero) a.cJr(rs1) else a.cJalr(rs1);
            return a.iType(off, rs1, 0, rd, 0x67);
        }
        /// `jal rd, label` (±1 MiB).
        pub fn jal(a: Self, rd: X, l: Label) Error!void {
            return a.branchTo(0x6F | rd.n() << 7, .rv_j20, l);
        }
        /// `b<cond> rs1, rs2, label` (±4 KiB).
        pub fn b(a: Self, c: Cond, rs1: X, rs2: X, l: Label) Error!void {
            return a.branchTo(rs2.n() << 20 | rs1.n() << 15 | @as(u32, @backingInt(c)) << 12 | 0x63, .rv_b12, l);
        }
        // ---- far control flow ----------------------------------------------------------
        fn pcrelPair(a: Self, rd: X, second: u32, l: Label) Error!void {
            if (!a.buf.isValidLabel(l)) return error.InvalidLabel;
            const at = a.buf.offset();
            const first: u32 = rd.n() << 7 | 0x17;
            if (a.buf.labelOffset(l)) |target| {
                const ws = core.rvPatchPcrelPair(first, second, @as(i64, target) - at);
                try a.word(ws[0]);
                return a.word(ws[1]);
            }
            try a.word(first);
            try a.word(second);
            try a.buf.addFixup(.{ .at = at, .base = at, .label = l.id, .kind = .rv_pcrel_pair });
        }
        /// `auipc ra, hi; jalr ra, lo(ra)`: call a label up to ±2 GiB away.
        pub fn callFar(a: Self, l: Label) Error!void {
            return a.pcrelPair(.ra, X.ra.n() << 15 | X.ra.n() << 7 | 0x67, l);
        }
        /// `auipc tmp, hi; jalr zero, lo(tmp)`: jump to a label up to ±2 GiB away.
        pub fn jFar(a: Self, tmp: X, l: Label) Error!void {
            if (tmp == .zero) return error.InvalidOperand;
            return a.pcrelPair(tmp, tmp.n() << 15 | 0x67, l);
        }
        /// `auipc rd, hi; addi rd, rd, lo`: address of a label.
        pub fn la(a: Self, rd: X, l: Label) Error!void {
            if (rd == .zero) return error.InvalidOperand;
            return a.pcrelPair(rd, rd.n() << 15 | rd.n() << 7 | 0x13, l);
        }
        /// Conditional branch with ±1 MiB reach: `b<!cond> skip; jal zero, l; skip:`.
        pub fn bFar(a: Self, c: Cond, rs1: X, rs2: X, l: Label) Error!void {
            const inv: Cond = switch (c) {
                .eq => .ne,
                .ne => .eq,
                .lt => .ge,
                .ge => .lt,
                .ltu => .geu,
                .geu => .ltu,
            };
            const skip = try a.newLabel();
            try a.b(inv, rs1, rs2, skip);
            try a.jal(.zero, l);
            try a.bind(skip);
        }

        fn branchTo(a: Self, w: u32, comptime kind: core.FixupKind, l: Label) Error!void {
            if (!a.buf.isValidLabel(l)) return error.InvalidLabel;
            const at = a.buf.offset();
            if (a.buf.labelOffset(l)) |target| {
                const d: i64 = @as(i64, target) - at;
                const lim: i64 = if (kind == .rv_b12) 1 << 12 else 1 << 20;
                if (d < -lim or d >= lim) return error.DisplacementOutOfRange;
                return a.word(if (kind == .rv_b12) core.rvPatchB(w, d) else core.rvPatchJ(w, d));
            }
            try a.word(w);
            try a.buf.addFixup(.{ .at = at, .base = at, .label = l.id, .kind = kind });
        }

        // ---- M extension ----
        pub fn mul(a: Self, rd: X, rs1: X, rs2: X) Error!void {
            return a.rType(1, rs2, rs1, 0, rd, 0x33);
        }
        pub fn mulh(a: Self, rd: X, rs1: X, rs2: X) Error!void {
            return a.rType(1, rs2, rs1, 1, rd, 0x33);
        }
        pub fn mulhu(a: Self, rd: X, rs1: X, rs2: X) Error!void {
            return a.rType(1, rs2, rs1, 3, rd, 0x33);
        }
        pub fn div(a: Self, rd: X, rs1: X, rs2: X) Error!void {
            return a.rType(1, rs2, rs1, 4, rd, 0x33);
        }
        pub fn divu(a: Self, rd: X, rs1: X, rs2: X) Error!void {
            return a.rType(1, rs2, rs1, 5, rd, 0x33);
        }
        pub fn rem(a: Self, rd: X, rs1: X, rs2: X) Error!void {
            return a.rType(1, rs2, rs1, 6, rd, 0x33);
        }
        pub fn remu(a: Self, rd: X, rs1: X, rs2: X) Error!void {
            return a.rType(1, rs2, rs1, 7, rd, 0x33);
        }
        // ---- A extension (opcode 0x2F, funct7 = funct5 << 2 | aq << 1 | rl) ------------
        /// `amo<op>.<w|d><.aq|.rl> rd, rs2, (rs1)`: rd = old [rs1]; [rs1] = op(old, rs2).
        pub fn amo(a: Self, op: Amo, w: Width, ord: Order, rd: X, rs2: X, rs1: X) Error!void {
            if (!is64 and w == .d) return error.InvalidOperand;
            return a.rType(@as(u32, @backingInt(op)) << 2 | @backingInt(ord), rs2, rs1, @backingInt(w), rd, 0x2F);
        }
        /// `lr.<w|d> rd, (rs1)`: load-reserved.
        pub fn lr(a: Self, w: Width, ord: Order, rd: X, rs1: X) Error!void {
            if (!is64 and w == .d) return error.InvalidOperand;
            return a.rType(0b00010 << 2 | @as(u32, @backingInt(ord)), .zero, rs1, @backingInt(w), rd, 0x2F);
        }
        /// `sc.<w|d> rd, rs2, (rs1)`: store-conditional, rd = 0 on success.
        pub fn sc(a: Self, w: Width, ord: Order, rd: X, rs2: X, rs1: X) Error!void {
            if (!is64 and w == .d) return error.InvalidOperand;
            return a.rType(0b00011 << 2 | @as(u32, @backingInt(ord)), rs2, rs1, @backingInt(w), rd, 0x2F);
        }
        /// `fence pred, succ` (4-bit iorw sets; `fence rw, rw` = `fence(0b0011, 0b0011)`).
        pub fn fence(a: Self, pred: u4, succ: u4) Error!void {
            return a.word(@as(u32, pred) << 24 | @as(u32, succ) << 20 | 0x0F);
        }

        // ---- F/D extensions ----------------------------------------------------------
        fn fpR(a: Self, f5: u32, fmt: Fmt, rs2: u32, rs1: u32, f3: u32, rd: u32) Error!void {
            return a.word(f5 << 27 | @as(u32, @backingInt(fmt)) << 25 | rs2 << 20 | rs1 << 15 | f3 << 12 | rd << 7 | 0x53);
        }
        /// `flw/fld rd, off(rs1)`.
        pub fn fload(a: Self, fmt: Fmt, rd: F, off: i64, rs1: X) Error!void {
            return a.word(try imm12(off) << 20 | rs1.n() << 15 | (@as(u32, @backingInt(fmt)) + 2) << 12 | rd.n() << 7 | 0x07);
        }
        /// `fsw/fsd rs2, off(rs1)`.
        pub fn fstore(a: Self, fmt: Fmt, rs2: F, off: i64, rs1: X) Error!void {
            const v = try imm12(off);
            return a.word((v >> 5) << 25 | rs2.n() << 20 | rs1.n() << 15 | (@as(u32, @backingInt(fmt)) + 2) << 12 | (v & 0x1F) << 7 | 0x27);
        }
        /// `fadd/fsub/fmul/fdiv.<s|d> rd, rs1, rs2` (rounding `rm`).
        pub fn farith(a: Self, op: FArith, fmt: Fmt, rd: F, rs1: F, rs2: F, rm: Rm) Error!void {
            return a.fpR(@backingInt(op), fmt, rs2.n(), rs1.n(), @backingInt(rm), rd.n());
        }
        pub fn fsqrt(a: Self, fmt: Fmt, rd: F, rs1: F, rm: Rm) Error!void {
            return a.fpR(0b01011, fmt, 0, rs1.n(), @backingInt(rm), rd.n());
        }
        pub fn fmin(a: Self, fmt: Fmt, rd: F, rs1: F, rs2: F) Error!void {
            return a.fpR(0b00101, fmt, rs2.n(), rs1.n(), 0, rd.n());
        }
        pub fn fmax(a: Self, fmt: Fmt, rd: F, rs1: F, rs2: F) Error!void {
            return a.fpR(0b00101, fmt, rs2.n(), rs1.n(), 1, rd.n());
        }
        /// `fsgnj` (j = 0), `fsgnjn` (1), `fsgnjx` (2); `fmv.d rd, rs` = `fsgnj(.d, 0, rd, rs, rs)`.
        pub fn fsgnj(a: Self, fmt: Fmt, kind: u2, rd: F, rs1: F, rs2: F) Error!void {
            return a.fpR(0b00100, fmt, rs2.n(), rs1.n(), kind, rd.n());
        }
        /// `fmadd.<s|d> rd, rs1, rs2, rs3` = rs1 * rs2 + rs3.
        pub fn fmadd(a: Self, fmt: Fmt, rd: F, rs1: F, rs2: F, rs3: F, rm: Rm) Error!void {
            return a.word(rs3.n() << 27 | @as(u32, @backingInt(fmt)) << 25 | rs2.n() << 20 | rs1.n() << 15 | @as(u32, @backingInt(rm)) << 12 | rd.n() << 7 | 0x43);
        }
        /// `feq/flt/fle.<s|d> rd, rs1, rs2` (rd is an integer register).
        pub fn fcmp(a: Self, c: FCmp, fmt: Fmt, rd: X, rs1: F, rs2: F) Error!void {
            return a.fpR(0b10100, fmt, rs2.n(), rs1.n(), @backingInt(c), rd.n());
        }
        /// `fcvt.<w|wu|l|lu>.<s|d> rd, rs1` (float -> integer).
        pub fn fcvtToInt(a: Self, k: IntKind, fmt: Fmt, rd: X, rs1: F, rm: Rm) Error!void {
            if (!is64 and (k == .l or k == .lu)) return error.InvalidOperand;
            return a.fpR(0b11000, fmt, @backingInt(k), rs1.n(), @backingInt(rm), rd.n());
        }
        /// `fcvt.<s|d>.<w|wu|l|lu> rd, rs1` (integer -> float).
        pub fn fcvtFromInt(a: Self, k: IntKind, fmt: Fmt, rd: F, rs1: X, rm: Rm) Error!void {
            if (!is64 and (k == .l or k == .lu)) return error.InvalidOperand;
            return a.fpR(0b11010, fmt, @backingInt(k), rs1.n(), @backingInt(rm), rd.n());
        }
        /// `fcvt.s.d` (to = .s) or `fcvt.d.s` (to = .d).
        pub fn fcvtFmt(a: Self, to: Fmt, rd: F, rs1: F, rm: Rm) Error!void {
            const from: u32 = if (to == .s) 1 else 0;
            return a.fpR(0b01000, to, from, rs1.n(), @backingInt(rm), rd.n());
        }
        /// `fmv.x.w/fmv.x.d rd, rs1` (raw bits float -> integer register).
        pub fn fmvToInt(a: Self, fmt: Fmt, rd: X, rs1: F) Error!void {
            if (!is64 and fmt == .d) return error.InvalidOperand;
            return a.fpR(0b11100, fmt, 0, rs1.n(), 0, rd.n());
        }
        /// `fmv.w.x/fmv.d.x rd, rs1` (raw bits integer -> float register).
        pub fn fmvFromInt(a: Self, fmt: Fmt, rd: F, rs1: X) Error!void {
            if (!is64 and fmt == .d) return error.InvalidOperand;
            return a.fpR(0b11110, fmt, 0, rs1.n(), 0, rd.n());
        }

        pub fn mulw(a: Self, rd: X, rs1: X, rs2: X) Error!void {
            comptime need64("mulw");
            return a.rType(1, rs2, rs1, 0, rd, 0x3B);
        }

        // ---- pseudo-instructions ----
        pub fn nop(a: Self) Error!void {
            return a.addi(.zero, .zero, 0);
        }
        pub fn mv(a: Self, rd: X, rs: X) Error!void {
            return a.addi(rd, rs, 0);
        }
        pub fn ret(a: Self) Error!void {
            return a.jalr(.zero, 0, .ra);
        }
        pub fn j(a: Self, l: Label) Error!void {
            return a.jal(.zero, l);
        }
        /// Loads any 64-bit constant (same strategy as LLVM's RISCVMatInt for the base ISA):
        /// 32-bit values: `lui` + `addiw` (or `addi` from zero); larger: materialize the upper
        /// part recursively, `slli`, then `addi` the low 12 bits.
        /// RV32: `v` must fit in 32 bits (signed or unsigned); emits `lui` + `addi`.
        pub fn li(a: Self, rd: X, v: i64) Error!void {
            if (!is64) {
                if (v < std.math.minInt(i32) or v > std.math.maxInt(u32)) return error.InvalidOperand;
                const v32: i64 = @as(i32, @bitCast(@as(u32, @truncate(@as(u64, @bitCast(v))))));
                if (fitsI12(v32)) return a.addi(rd, .zero, v32);
                const lo = @as(i64, @as(i12, @truncate(v32)));
                try a.lui(rd, @truncate(@as(u64, @bitCast((v32 - lo) >> 12)) & 0xFFFFF));
                if (lo != 0) try a.addi(rd, rd, lo);
                return;
            }
            if (fitsI12(v)) return a.addi(rd, .zero, v);
            if (v >= std.math.minInt(i32) and v <= std.math.maxInt(i32)) {
                const lo = @as(i64, @as(i12, @truncate(v)));
                const hi: u32 = @truncate(@as(u64, @bitCast((v - lo) >> 12)) & 0xFFFFF);
                try a.lui(rd, hi);
                if (lo != 0) try a.addiw(rd, rd, lo);
                return;
            }
            return a.liWide(rd, v);
        }
        fn liWide(a: Self, rd: X, v: i64) Error!void {
            const lo = @as(i64, @as(i12, @truncate(v)));
            var hi = (v -% lo) >> 12;
            const tz: u6 = @intCast(@ctz(hi));
            hi >>= tz;
            try a.li(rd, hi);
            try a.slli(rd, rd, @as(u32, tz) + 12);
            if (lo != 0) try a.addi(rd, rd, lo);
        }
        /// `li t0, addr; jalr ra, 0(t0)`.
        pub fn callAbs(a: Self, addr: u64) Error!void {
            if (!is64 and addr > std.math.maxInt(u32)) return error.InvalidOperand;
            try a.li(.t0, @bitCast(addr));
            try a.jalr(.ra, 0, .t0);
        }
        /// Store/load one register-sized slot (`sd`/`ld` on RV64, `sw`/`lw` on RV32).
        pub fn storeX(a: Self, rs2: X, off: i64, rs1: X) Error!void {
            return a.sType(off, rs2, rs1, if (is64) 3 else 2);
        }
        pub fn loadX(a: Self, rd: X, off: i64, rs1: X) Error!void {
            return a.iType(off, rs1, if (is64) 3 else 2, rd, 0x03);
        }
        /// `addi sp, sp, -n; s{d,w} ra, n-X(sp); s{d,w} s0, n-2X(sp); addi s0, sp, n; <saved>...`,
        /// X = xlen_bytes, n = 16-aligned frame (ra + s0 + saved + locals). Returns n.
        pub fn prolog(a: Self, saved: []const X, locals: u32) Error!i32 {
            const w: i32 = @intCast(xlen_bytes);
            const n: i32 = @intCast(std.mem.alignForward(u32, 2 * xlen_bytes + xlen_bytes * @as(u32, @intCast(saved.len)) + locals, 16));
            try a.addi(.sp, .sp, -n);
            try a.storeX(.ra, n - w, .sp);
            try a.storeX(.s0, n - 2 * w, .sp);
            for (saved, 0..) |r, i| try a.storeX(r, n - 3 * w - w * @as(i32, @intCast(i)), .sp);
            try a.addi(.s0, .sp, n);
            return n;
        }
        pub fn epilog(a: Self, saved: []const X, n: i32) Error!void {
            const w: i32 = @intCast(xlen_bytes);
            for (saved, 0..) |r, i| try a.loadX(r, n - 3 * w - w * @as(i32, @intCast(i)), .sp);
            try a.loadX(.s0, n - 2 * w, .sp);
            try a.loadX(.ra, n - w, .sp);
            try a.addi(.sp, .sp, n);
            try a.ret();
        }
    };
}
pub const Assembler = AssemblerT(.{});
/// RV32 (ILP32) assembler.
pub const Assembler32 = AssemblerT(.{ .xlen = .rv32 });
/// RV64GC assembler that auto-compresses.
pub const AssemblerC = AssemblerT(.{ .compress = true });

// ---- tests ------------------------------------------------------------------------------
const t = std.testing;

fn words(cb: *const CodeBuffer) []align(1) const u32 {
    return std.mem.bytesAsSlice(u32, cb.code());
}

test "riscv: encodings (RISC-V spec / GNU as reference values)" {
    var cb = CodeBuffer.init(t.allocator);
    defer cb.deinit();
    const a = Assembler.init(&cb);
    try a.addi(.a0, .a0, 1);
    try a.add(.a0, .a0, .a1);
    try a.sub(.a0, .a0, .a1);
    try a.mul(.a0, .a0, .a1);
    try a.lui(.a0, 0x12345);
    try a.ld(.a0, 8, .sp);
    try a.sd(.ra, 8, .sp);
    try a.ret();
    try a.slli(.a0, .a0, 3);
    try a.srai(.a0, .a0, 63);
    try a.addiw(.a0, .a0, -1);
    try a.nop();
    try a.mv(.s0, .sp);
    const want = [_]u32{ 0x00150513, 0x00b50533, 0x40b50533, 0x02b50533, 0x12345537, 0x00813503, 0x00113423, 0x00008067, 0x00351513, 0x43f55513, 0xfff5051b, 0x00000013, 0x00010413 };
    for (want, words(&cb)) |w, g| try t.expectEqual(w, g);
}

test "riscv: A extension and fence (spec reference values)" {
    var cb = CodeBuffer.init(t.allocator);
    defer cb.deinit();
    const a = Assembler.init(&cb);
    try a.lr(.d, .none, .a0, .a1); // lr.d a0, (a1)
    try a.sc(.d, .none, .a2, .a3, .a1); // sc.d a2, a3, (a1)
    try a.amo(.add, .w, .none, .a0, .a1, .a2); // amoadd.w a0, a1, (a2)
    try a.amo(.swap, .d, .aqrl, .t0, .t1, .t2); // amoswap.d.aqrl t0, t1, (t2)
    try a.lr(.w, .aq, .a0, .a1); // lr.w.aq a0, (a1)
    try a.amo(.maxu, .d, .rl, .a0, .a1, .a2); // amomaxu.d.rl a0, a1, (a2)
    try a.fence(0b0011, 0b0011); // fence rw, rw
    const want = [_]u32{ 0x1005B52F, 0x18D5B62F, 0x00B6252F, 0x0E63B2AF, 0x1405A52F, 0xE2B6352F, 0x0330000F };
    for (want, 0..) |w, i| try t.expectEqual(w, std.mem.readInt(u32, cb.code()[4 * i ..][0..4], .little));
}

test "riscv: F/D extensions (GNU as reference values)" {
    var cb = CodeBuffer.init(t.allocator);
    defer cb.deinit();
    const a = Assembler.init(&cb);
    try a.farith(.add, .d, .f10, .f10, .f11, .dyn); // fadd.d fa0, fa0, fa1
    try a.fload(.d, .f10, 8, .a0); // fld fa0, 8(a0)
    try a.fstore(.d, .f10, 8, .a0); // fsd fa0, 8(a0)
    try a.fmvToInt(.d, .a0, .f10); // fmv.x.d a0, fa0
    try a.fcvtFromInt(.l, .d, .f10, .a0, .dyn); // fcvt.d.l fa0, a0
    try a.fcmp(.eq, .d, .a0, .f10, .f11); // feq.d a0, fa0, fa1
    try a.fload(.s, .f1, -4, .sp); // flw ft1, -4(sp)
    try a.fmadd(.d, .f0, .f1, .f2, .f3, .rne); // fmadd.d ft0, ft1, ft2, ft3, rne
    try a.fcvtToInt(.w, .s, .a1, .f2, .rtz); // fcvt.w.s a1, ft2, rtz
    try a.fcvtFmt(.d, .f4, .f5, .dyn); // fcvt.d.s ft4, ft5
    const want = [_]u32{ 0x02B57553, 0x00853507, 0x00A53427, 0xE2050553, 0xD2257553, 0xA2B52553, 0xFFC12087, 0x1A208043, 0xC00115D3, 0x4202F253 };
    for (want, 0..) |w, i| try t.expectEqual(w, std.mem.readInt(u32, cb.code()[4 * i ..][0..4], .little));
}

test "riscv: far branches (auipc pairs and bFar)" {
    var cb = CodeBuffer.init(t.allocator);
    defer cb.deinit();
    const a = Assembler.init(&cb);
    const far = try a.newLabel();
    try a.callFar(far); // 0
    try a.la(.a0, far); // 8
    try a.bFar(.eq, .a0, .a1, far); // 16: bne +8; 20: jal
    try a.jFar(.t1, far); // 24
    // Pad to put `far` 0x12345 + 0x800 bytes away (exercises the hi20 rounding).
    while (cb.offset() < 0x12B48) try a.nop();
    try a.bind(far);
    try a.ret();
    try cb.finalize();
    const c = cb.code();
    const rd = struct {
        fn w(code: []const u8, off: usize) u32 {
            return std.mem.readInt(u32, code[off..][0..4], .little);
        }
        fn simm(v: u32, comptime bits: u5, comptime shift: u5) i64 {
            const x: u32 = (v >> shift) & ((@as(u32, 1) << bits) - 1);
            const sx: i64 = x;
            return if (x >> (bits - 1) != 0) sx - (@as(i64, 1) << bits) else sx;
        }
    };
    const target: i64 = 0x12B48;
    // Decode each auipc pair back: pc + (hi << 12) + sext(lo) == target.
    for ([_]usize{ 0, 8, 24 }) |at| {
        const hi = rd.simm(rd.w(c, at), 20, 12);
        const lo = rd.simm(rd.w(c, at + 4), 12, 20);
        try t.expectEqual(target, @as(i64, @intCast(at)) + (hi << 12) + lo);
    }
    try t.expectEqual(@as(u32, 0x0000_0017 | 1 << 7), rd.w(c, 0) & 0xFFF); // auipc ra
    try t.expectEqual(@as(u32, 0x00B51463), rd.w(c, 16)); // bne a0, a1, +8
    try t.expectEqual(@as(u32, 0x6F), rd.w(c, 20) & 0xFFF); // jal zero
    // Backward and resolved-at-emit path.
    const back = try a.newLabel();
    try a.bind(back);
    try a.la(.a2, back);
    try t.expectEqual(@as(i64, 0), rd.simm(rd.w(cb.code(), cb.code().len - 8), 20, 12));
    try t.expectError(error.InvalidOperand, a.jFar(.zero, back));
}

test "riscv: C extension subset (GNU as reference values)" {
    var cb = CodeBuffer.init(t.allocator);
    defer cb.deinit();
    const a = Assembler.init(&cb);
    try a.cNop(); // 0001
    try a.cMv(.a0, .a1); // 852e
    try a.cAdd(.a0, .a1); // 952e
    try a.cLi(.a0, 1); // 4505
    try a.cAddi(.a0, 1); // 0505
    try a.cRet(); // 8082
    try a.cLdsp(.ra, 8); // 60a2
    try a.cSdsp(.ra, 8); // e406
    try a.cAddi(.sp, -16); // 1141
    try a.cSlli(.a0, 3); // 050e
    try a.cAddiw(.a0, -1); // 357d
    const want = [_]u16{ 0x0001, 0x852E, 0x952E, 0x4505, 0x0505, 0x8082, 0x60A2, 0xE406, 0x1141, 0x050E, 0x357D };
    for (want, 0..) |w, i| try t.expectEqual(w, std.mem.readInt(u16, cb.code()[2 * i ..][0..2], .little));
    try t.expectError(error.InvalidOperand, a.cAddi(.a0, 0));
    try t.expectError(error.InvalidOperand, a.cLi(.a0, 32));
    try t.expectError(error.InvalidOperand, a.cLdsp(.a0, 12));
    try t.expectError(error.InvalidOperand, a.cMv(.a0, .zero));
}

test "riscv: branches and jal with labels (forward fixups and backward)" {
    var cb = CodeBuffer.init(t.allocator);
    defer cb.deinit();
    const a = Assembler.init(&cb);
    const fwd = try a.newLabel();
    const top = try a.newLabel();
    try a.b(.eq, .a0, .a1, fwd); // +8
    try a.jal(.ra, fwd); // +4
    try a.bind(fwd);
    try a.bind(top);
    try a.b(.ne, .a0, .zero, top); // 0
    try a.j(top); // -4
    try cb.finalize();
    const w = words(&cb);
    try t.expectEqual(@as(u32, 0x00b50463), w[0]); // beq a0, a1, +8
    try t.expectEqual(@as(u32, 0x004000ef), w[1]); // jal ra, +4
    try t.expectEqual(@as(u32, 0x00051063), w[2]); // bne a0, zero, 0
    try t.expectEqual(@as(u32, 0xffdff06f), w[3]); // jal zero, -4
    try t.expectError(error.InvalidOperand, a.addi(.a0, .a0, 4096));
    try t.expectError(error.InvalidOperand, a.slli(.a0, .a0, 64));
}

/// Tiny RV64 interpreter for the ops `li` emits (verifies constant materialization).
fn evalLi(code: []align(1) const u32, rd: u5) i64 {
    var x: [32]i64 = @splat(0);
    for (code) |w| {
        const op = w & 0x7F;
        const d: u5 = @truncate(w >> 7);
        const s1: u5 = @truncate(w >> 15);
        const imm: i64 = @as(i32, @bitCast(w)) >> 20;
        switch (op) {
            0x37 => x[d] = @as(i32, @bitCast(w & 0xFFFFF000)),
            0x13 => x[d] = switch ((w >> 12) & 7) {
                0 => x[s1] +% imm,
                1 => @bitCast(@as(u64, @bitCast(x[s1])) << @intCast((w >> 20) & 63)),
                else => unreachable,
            },
            0x1B => x[d] = @as(i32, @truncate(x[s1] +% imm)),
            else => unreachable,
        }
        x[0] = 0;
    }
    return x[rd];
}

test "riscv: li materializes any 64-bit constant (interpreter check, 20000 values)" {
    var prng = std.Random.DefaultPrng.init(1);
    const r = prng.random();
    var cb = CodeBuffer.init(t.allocator);
    defer cb.deinit();
    const edge = [_]i64{ 0, 1, -1, 2047, -2048, 2048, -2049, 0x7FFFFFFF, -0x80000000, 0x80000000, 0xFFFFFFFF, std.math.maxInt(i64), std.math.minInt(i64), 0x123456789ABCDEF0 };
    for (0..20000 + edge.len) |i| {
        const v: i64 = if (i < edge.len) edge[i] else switch (i % 3) {
            0 => r.int(i64),
            1 => r.int(i32),
            else => r.int(i64) >> @intCast(r.uintLessThan(u32, 63)),
        };
        cb.reset();
        try Assembler.init(&cb).li(.a0, v);
        try t.expect(words(&cb).len <= 8);
        try t.expectEqual(v, evalLi(words(&cb), @backingInt(X.a0)));
    }
}

test "riscv: prolog/epilog shape" {
    var cb = CodeBuffer.init(t.allocator);
    defer cb.deinit();
    const a = Assembler.init(&cb);
    const n = try a.prolog(&.{.s1}, 0);
    try t.expectEqual(@as(i32, 32), n);
    try a.epilog(&.{.s1}, n);
    const w = words(&cb);
    try t.expectEqual(@as(u32, 0xfe010113), w[0]); // addi sp, sp, -32
    try t.expectEqual(@as(u32, 0x00008067), w[w.len - 1]); // ret
}

test "riscv: RV32 mode (GNU as -march=rv32imafd reference values)" {
    var cb = CodeBuffer.init(t.allocator);
    defer cb.deinit();
    const a = Assembler32.init(&cb);
    try a.slli(.a0, .a0, 31); // 01f51513
    try a.srai(.a0, .a0, 31); // 41f55513
    try a.srli(.a0, .a0, 1); // 00155513
    try a.lw(.a0, 4, .sp); // 00412503
    try a.sw(.ra, 12, .sp); // 00112623
    try a.amo(.add, .w, .none, .a0, .a1, .a2); // 00b6252f
    try a.fmvToInt(.s, .a0, .f10); // e0050553
    try a.cLwsp(.ra, 12); // 40b2
    try a.cSwsp(.ra, 12); // c606
    const want = [_]u32{ 0x01F51513, 0x41F55513, 0x00155513, 0x00412503, 0x00112623, 0x00B6252F, 0xE0050553 };
    for (want, 0..) |w, i| try t.expectEqual(w, std.mem.readInt(u32, cb.code()[4 * i ..][0..4], .little));
    try t.expectEqual(@as(u16, 0x40B2), std.mem.readInt(u16, cb.code()[28..30], .little));
    try t.expectEqual(@as(u16, 0xC606), std.mem.readInt(u16, cb.code()[30..32], .little));
    try t.expectError(error.InvalidOperand, a.slli(.a0, .a0, 32));
    try t.expectError(error.InvalidOperand, a.srai(.a0, .a0, 32));
    try t.expectError(error.InvalidOperand, a.cSlli(.a0, 32));
    try t.expectError(error.InvalidOperand, a.amo(.add, .d, .none, .a0, .a1, .a2));
    try t.expectError(error.InvalidOperand, a.lr(.d, .none, .a0, .a1));
    try t.expectError(error.InvalidOperand, a.fcvtToInt(.l, .d, .a0, .f0, .dyn));
    try t.expectError(error.InvalidOperand, a.fmvFromInt(.d, .f0, .a0));
    try t.expectError(error.InvalidOperand, a.li(.a0, 0x1_0000_0000));
    try t.expectError(error.InvalidOperand, a.li(.a0, -0x8000_0001));
    try t.expectError(error.InvalidOperand, a.callAbs(0x1_0000_0000));
    // RV64 accepts the same shift amount and still rejects 64.
    var cb64 = CodeBuffer.init(t.allocator);
    defer cb64.deinit();
    try Assembler.init(&cb64).slli(.a0, .a0, 32);
    try t.expectEqual(@as(u32, 0x02051513), words(&cb64)[0]);
    try t.expect(!@hasDecl(Assembler32, "is64") and Assembler32.xlen == .rv32 and Assembler32.xlen_bytes == 4);
}

/// Tiny RV32 interpreter for the ops RV32 `li` emits (lui, addi).
fn evalLi32(code: []align(1) const u32, rd: u5) u32 {
    var x: [32]u32 = @splat(0);
    for (code) |w| {
        const d: u5 = @truncate(w >> 7);
        const s1: u5 = @truncate(w >> 15);
        switch (w & 0x7F) {
            0x37 => x[d] = w & 0xFFFFF000,
            0x13 => {
                std.debug.assert((w >> 12) & 7 == 0);
                x[d] = x[s1] +% @as(u32, @bitCast(@as(i32, @bitCast(w)) >> 20));
            },
            else => unreachable,
        }
        x[0] = 0;
    }
    return x[rd];
}

test "riscv: RV32 li (lui+addi, interpreter check, 20000 values)" {
    var prng = std.Random.DefaultPrng.init(2);
    const r = prng.random();
    var cb = CodeBuffer.init(t.allocator);
    defer cb.deinit();
    const edge = [_]i64{ 0, 1, -1, 2047, -2048, 2048, -2049, 0x7FF, 0x800, 0x7FFFF800, 0x7FFFFFFF, -0x80000000, 0x80000000, 0xFFFFFFFF, 0xFFFFF800 };
    for (0..20000 + edge.len) |i| {
        const v: i64 = if (i < edge.len) edge[i] else if (i % 2 == 0) r.int(i32) else r.int(u32);
        cb.reset();
        try Assembler32.init(&cb).li(.a0, v);
        try t.expect(words(&cb).len <= 2);
        try t.expectEqual(@as(u32, @truncate(@as(u64, @bitCast(v)))), evalLi32(words(&cb), @backingInt(X.a0)));
    }
}

test "riscv: RV32 prolog/epilog use 4-byte slots" {
    var cb = CodeBuffer.init(t.allocator);
    defer cb.deinit();
    const a = Assembler32.init(&cb);
    const n = try a.prolog(&.{ .s1, .s2 }, 0);
    try t.expectEqual(@as(i32, 16), n);
    try a.epilog(&.{ .s1, .s2 }, n);
    const w = words(&cb);
    // addi sp,-16; sw ra,12(sp); sw s0,8(sp); sw s1,4(sp); sw s2,0(sp); addi s0,sp,16
    const want = [_]u32{ 0xff010113, 0x00112623, 0x00812423, 0x00912223, 0x01212023, 0x01010413 };
    for (want, w[0..want.len]) |e, g| try t.expectEqual(e, g);
    try t.expectEqual(@as(u32, 0x00c12083), w[w.len - 3]); // lw ra, 12(sp)
    try t.expectEqual(@as(u32, 0x00008067), w[w.len - 1]); // ret
}

fn halves(cb: *const CodeBuffer) []align(1) const u16 {
    return std.mem.bytesAsSlice(u16, cb.code());
}

test "riscv: C extension CL/CS/CB/CA/CIW forms (GNU as reference values)" {
    var cb = CodeBuffer.init(t.allocator);
    defer cb.deinit();
    const a = Assembler.init(&cb);
    try a.cLw(.a0, 0, .a1); // 4188
    try a.cSw(.a0, 0, .a1); // c188
    try a.cLd(.a0, 0, .a1); // 6188
    try a.cSd(.a0, 0, .a1); // e188
    try a.cLd(.a0, 8, .a1); // 6588
    try a.cSub(.a0, .a1); // 8d0d
    try a.cXor(.a0, .a1); // 8d2d
    try a.cOr(.a0, .a1); // 8d4d
    try a.cAnd(.a0, .a1); // 8d6d
    try a.cSubw(.a0, .a1); // 9d0d
    try a.cAddw(.a0, .a1); // 9d2d
    try a.cSrli(.a0, 1); // 8105
    try a.cSrai(.a0, 1); // 8505
    try a.cAndi(.a0, 1); // 8905
    try a.cLui(.a0, 1); // 6505
    try a.cAddi16sp(-64); // 7139
    try a.cAddi4spn(.s0, 16); // 0800
    try a.cAddi4spn(.a0, 16); // 0808
    try a.cJalr(.a0); // 9502
    try a.cEbreak(); // 9002
    const want = [_]u16{ 0x4188, 0xC188, 0x6188, 0xE188, 0x6588, 0x8D0D, 0x8D2D, 0x8D4D, 0x8D6D, 0x9D0D, 0x9D2D, 0x8105, 0x8505, 0x8905, 0x6505, 0x7139, 0x0800, 0x0808, 0x9502, 0x9002 };
    for (want, halves(&cb)) |w, g| try t.expectEqual(w, g);
    try t.expectEqual(want.len * 2, cb.code().len);
    try t.expectError(error.InvalidOperand, a.cLw(.a0, 0, .sp)); // base not x8..x15
    try t.expectError(error.InvalidOperand, a.cLw(.t0, 0, .a1));
    try t.expectError(error.InvalidOperand, a.cLw(.a0, 2, .a1));
    try t.expectError(error.InvalidOperand, a.cLd(.a0, 256, .a1));
    try t.expectError(error.InvalidOperand, a.cAddi16sp(8));
    try t.expectError(error.InvalidOperand, a.cLui(.sp, 1));
}

/// Decodes the target offset of a CB / CJ parcel (inverse of rvPatchCB / rvPatchCJ).
fn cOff(h: u16, comptime cj: bool) i64 {
    const v: u32 = h;
    var o: u32 = undefined;
    var bits: u5 = undefined;
    if (cj) {
        o = (v >> 12 & 1) << 11 | (v >> 11 & 1) << 4 | (v >> 9 & 3) << 8 | (v >> 8 & 1) << 10 | (v >> 7 & 1) << 6 | (v >> 6 & 1) << 7 | (v >> 3 & 7) << 1 | (v >> 2 & 1) << 5;
        bits = 12;
    } else {
        o = (v >> 12 & 1) << 8 | (v >> 10 & 3) << 3 | (v >> 5 & 3) << 6 | (v >> 3 & 3) << 1 | (v >> 2 & 1) << 5;
        bits = 9;
    }
    const sx: i64 = o;
    return if (o >> (bits - 1) != 0) sx - (@as(i64, 1) << bits) else sx;
}

test "riscv: compressed branches and jumps (labels, fixups, round-trip)" {
    var cb = CodeBuffer.init(t.allocator);
    defer cb.deinit();
    const a = Assembler.init(&cb);
    const here = try a.newLabel();
    try a.bind(here);
    try a.cJ(here); // a001
    try a.cBeqz(.a0, here); // -2
    try a.cBnez(.a0, here); // -4
    try t.expectEqual(@as(u16, 0xA001), halves(&cb)[0]);
    try t.expectEqual(@as(u16, 0xC101), halves(&cb)[1] & 0xE383);
    try t.expectEqual(@as(i64, -2), cOff(halves(&cb)[1], false));
    try t.expectEqual(@as(i64, -4), cOff(halves(&cb)[2], false));
    // Every even offset in range, forward (fixup) and backward (resolved at emit).
    var d: i64 = -2048;
    while (d < 2048) : (d += 2) {
        try t.expectEqual(d, cOff(core.rvPatchCJ(0xA001, d), true));
        if (d >= -256 and d < 256) try t.expectEqual(d, cOff(core.rvPatchCB(0xC101, d), false));
    }
    cb.reset();
    const fwd = try a.newLabel();
    try a.cBnez(.s1, fwd);
    try a.cJ(fwd);
    for (0..100) |_| try a.cNop();
    try a.bind(fwd);
    try cb.finalize();
    try t.expectEqual(@as(i64, 204), cOff(halves(&cb)[0], false));
    try t.expectEqual(@as(i64, 202), cOff(halves(&cb)[1], true));
    // Out of range.
    cb.reset();
    const far = try a.newLabel();
    try a.cBeqz(.a0, far);
    for (0..200) |_| try a.cNop();
    try a.bind(far);
    try t.expectError(error.DisplacementOutOfRange, cb.finalize());
    try t.expectError(error.InvalidOperand, a.cBeqz(.t0, far));
}

test "riscv: RV32 c.jal" {
    var cb = CodeBuffer.init(t.allocator);
    defer cb.deinit();
    const a = Assembler32.init(&cb);
    const l = try a.newLabel();
    try a.bind(l);
    try a.cJal(l); // 2001
    try t.expectEqual(@as(u16, 0x2001), halves(&cb)[0]);
}

test "riscv: auto-compress picks 16-bit forms only when equivalent" {
    var cb = CodeBuffer.init(t.allocator);
    defer cb.deinit();
    const a = AssemblerC.init(&cb);
    try a.addi(.sp, .sp, -16); // c.addi sp,-16: 1141
    try a.sd(.ra, 8, .sp); // c.sdsp: e406
    try a.addi(.s0, .sp, 16); // c.addi4spn: 0800
    try a.mv(.a0, .a1); // c.mv: 852e
    try a.li(.a0, 1); // c.li: 4505
    try a.add(.a0, .a0, .a1); // c.add: 952e
    try a.sub(.a0, .a0, .a1); // c.sub: 8d0d
    try a.ld(.a0, 8, .a1); // c.ld: 6588
    try a.lw(.a0, 0, .a1); // c.lw: 4188
    try a.ld(.ra, 8, .sp); // c.ldsp: 60a2
    try a.addi(.sp, .sp, -64); // c.addi16sp: 7139
    try a.nop(); // c.nop: 0001
    try a.ret(); // c.ret: 8082
    const want = [_]u16{ 0x1141, 0xE406, 0x0800, 0x852E, 0x4505, 0x952E, 0x8D0D, 0x6588, 0x4188, 0x60A2, 0x7139, 0x0001, 0x8082 };
    for (want, halves(&cb)) |w, g| try t.expectEqual(w, g);
    try t.expectEqual(want.len * 2, cb.code().len);
    // No 16-bit form: stays 32-bit (same words as the plain assembler).
    cb.reset();
    try a.addi(.a0, .a1, 100);
    try a.sub(.a0, .a1, .a2);
    try a.ld(.t0, 8, .a1);
    try a.ld(.a0, 4, .a1); // misaligned for c.ld
    try a.addi(.a0, .a0, 0x7FF);
    try t.expectEqual(@as(usize, 20), cb.code().len);
    var cb2 = CodeBuffer.init(t.allocator);
    defer cb2.deinit();
    const p = Assembler.init(&cb2);
    try p.addi(.a0, .a1, 100);
    try p.sub(.a0, .a1, .a2);
    try p.ld(.t0, 8, .a1);
    try p.ld(.a0, 4, .a1);
    try p.addi(.a0, .a0, 0x7FF);
    try t.expectEqualSlices(u8, cb2.code(), cb.code());
}

test "riscv: RV32 auto-compress (c.lwsp/c.swsp, c.lw, ILP32 prolog stays valid)" {
    var cb = CodeBuffer.init(t.allocator);
    defer cb.deinit();
    const a = AssemblerT(.{ .xlen = .rv32, .compress = true }).init(&cb);
    try a.sw(.ra, 12, .sp); // c.swsp: c606
    try a.lw(.ra, 12, .sp); // c.lwsp: 40b2
    try a.lw(.a0, 0, .a1); // c.lw: 4188
    try a.slli(.a0, .a0, 31); // c.slli a0,31: 057e
    const want = [_]u16{ 0xC606, 0x40B2, 0x4188, 0x057E };
    for (want, halves(&cb)) |w, g| try t.expectEqual(w, g);
    try a.slli(.a0, .a1, 3); // rd != rs1: 32-bit
    try t.expectEqual(@as(usize, 12), cb.code().len);
    _ = try a.prolog(&.{.s1}, 0);
}
