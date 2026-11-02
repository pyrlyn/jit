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
//! Not yet: C (compressed), RV32, far branches (auipc+jalr pairs).
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

pub const Options = struct { validate: bool = true };

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

        pub fn init(buf: *CodeBuffer) Self {
            return .{ .buf = buf };
        }
        pub fn newLabel(a: Self) Error!Label {
            return a.buf.newLabel();
        }
        pub fn bind(a: Self, l: Label) Error!void {
            return a.buf.bind(l);
        }
        fn word(a: Self, w: u32) Error!void {
            try a.buf.ensure(4);
            a.buf.put32(w);
        }
        fn imm12(v: i64) Error!u32 {
            if (opts.validate and !fitsI12(v)) return error.InvalidOperand;
            return @as(u32, @truncate(@as(u64, @bitCast(v)))) & 0xFFF;
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
            return a.rType(0, rs2, rs1, 0, rd, 0x33);
        }
        pub fn sub(a: Self, rd: X, rs1: X, rs2: X) Error!void {
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
            return a.rType(0, rs2, rs1, 4, rd, 0x33);
        }
        pub fn srl(a: Self, rd: X, rs1: X, rs2: X) Error!void {
            return a.rType(0, rs2, rs1, 5, rd, 0x33);
        }
        pub fn sra(a: Self, rd: X, rs1: X, rs2: X) Error!void {
            return a.rType(0x20, rs2, rs1, 5, rd, 0x33);
        }
        pub fn @"or"(a: Self, rd: X, rs1: X, rs2: X) Error!void {
            return a.rType(0, rs2, rs1, 6, rd, 0x33);
        }
        pub fn @"and"(a: Self, rd: X, rs1: X, rs2: X) Error!void {
            return a.rType(0, rs2, rs1, 7, rd, 0x33);
        }
        pub fn addw(a: Self, rd: X, rs1: X, rs2: X) Error!void {
            return a.rType(0, rs2, rs1, 0, rd, 0x3B);
        }
        pub fn subw(a: Self, rd: X, rs1: X, rs2: X) Error!void {
            return a.rType(0x20, rs2, rs1, 0, rd, 0x3B);
        }
        pub fn addi(a: Self, rd: X, rs1: X, imm: i64) Error!void {
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
            return a.iType(imm, rs1, 7, rd, 0x13);
        }
        pub fn addiw(a: Self, rd: X, rs1: X, imm: i64) Error!void {
            return a.iType(imm, rs1, 0, rd, 0x1B);
        }
        fn shiftImm(a: Self, rd: X, rs1: X, shamt: u32, f3: u32, hi: u32) Error!void {
            if (shamt > 63) return error.InvalidOperand;
            return a.word(hi << 26 | shamt << 20 | rs1.n() << 15 | f3 << 12 | rd.n() << 7 | 0x13);
        }
        pub fn slli(a: Self, rd: X, rs1: X, shamt: u32) Error!void {
            return a.shiftImm(rd, rs1, shamt, 1, 0);
        }
        pub fn srli(a: Self, rd: X, rs1: X, shamt: u32) Error!void {
            return a.shiftImm(rd, rs1, shamt, 5, 0);
        }
        pub fn srai(a: Self, rd: X, rs1: X, shamt: u32) Error!void {
            return a.shiftImm(rd, rs1, shamt, 5, 0x10);
        }
        pub fn lui(a: Self, rd: X, imm20: u32) Error!void {
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
            return a.iType(off, rs1, 2, rd, 0x03);
        }
        pub fn ld(a: Self, rd: X, off: i64, rs1: X) Error!void {
            return a.iType(off, rs1, 3, rd, 0x03);
        }
        pub fn lbu(a: Self, rd: X, off: i64, rs1: X) Error!void {
            return a.iType(off, rs1, 4, rd, 0x03);
        }
        pub fn lhu(a: Self, rd: X, off: i64, rs1: X) Error!void {
            return a.iType(off, rs1, 5, rd, 0x03);
        }
        pub fn lwu(a: Self, rd: X, off: i64, rs1: X) Error!void {
            return a.iType(off, rs1, 6, rd, 0x03);
        }
        pub fn sb(a: Self, rs2: X, off: i64, rs1: X) Error!void {
            return a.sType(off, rs2, rs1, 0);
        }
        pub fn sh(a: Self, rs2: X, off: i64, rs1: X) Error!void {
            return a.sType(off, rs2, rs1, 1);
        }
        pub fn sw(a: Self, rs2: X, off: i64, rs1: X) Error!void {
            return a.sType(off, rs2, rs1, 2);
        }
        pub fn sd(a: Self, rs2: X, off: i64, rs1: X) Error!void {
            return a.sType(off, rs2, rs1, 3);
        }
        pub fn jalr(a: Self, rd: X, off: i64, rs1: X) Error!void {
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
            return a.rType(@as(u32, @backingInt(op)) << 2 | @backingInt(ord), rs2, rs1, @backingInt(w), rd, 0x2F);
        }
        /// `lr.<w|d> rd, (rs1)`: load-reserved.
        pub fn lr(a: Self, w: Width, ord: Order, rd: X, rs1: X) Error!void {
            return a.rType(0b00010 << 2 | @as(u32, @backingInt(ord)), .zero, rs1, @backingInt(w), rd, 0x2F);
        }
        /// `sc.<w|d> rd, rs2, (rs1)`: store-conditional, rd = 0 on success.
        pub fn sc(a: Self, w: Width, ord: Order, rd: X, rs2: X, rs1: X) Error!void {
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
            return a.fpR(0b11000, fmt, @backingInt(k), rs1.n(), @backingInt(rm), rd.n());
        }
        /// `fcvt.<s|d>.<w|wu|l|lu> rd, rs1` (integer -> float).
        pub fn fcvtFromInt(a: Self, k: IntKind, fmt: Fmt, rd: F, rs1: X, rm: Rm) Error!void {
            return a.fpR(0b11010, fmt, @backingInt(k), rs1.n(), @backingInt(rm), rd.n());
        }
        /// `fcvt.s.d` (to = .s) or `fcvt.d.s` (to = .d).
        pub fn fcvtFmt(a: Self, to: Fmt, rd: F, rs1: F, rm: Rm) Error!void {
            const from: u32 = if (to == .s) 1 else 0;
            return a.fpR(0b01000, to, from, rs1.n(), @backingInt(rm), rd.n());
        }
        /// `fmv.x.w/fmv.x.d rd, rs1` (raw bits float -> integer register).
        pub fn fmvToInt(a: Self, fmt: Fmt, rd: X, rs1: F) Error!void {
            return a.fpR(0b11100, fmt, 0, rs1.n(), 0, rd.n());
        }
        /// `fmv.w.x/fmv.d.x rd, rs1` (raw bits integer -> float register).
        pub fn fmvFromInt(a: Self, fmt: Fmt, rd: F, rs1: X) Error!void {
            return a.fpR(0b11110, fmt, 0, rs1.n(), 0, rd.n());
        }

        pub fn mulw(a: Self, rd: X, rs1: X, rs2: X) Error!void {
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
        pub fn li(a: Self, rd: X, v: i64) Error!void {
            if (fitsI12(v)) return a.addi(rd, .zero, v);
            if (v >= std.math.minInt(i32) and v <= std.math.maxInt(i32)) {
                const lo = @as(i64, @as(i12, @truncate(v)));
                const hi: u32 = @truncate(@as(u64, @bitCast((v - lo) >> 12)) & 0xFFFFF);
                try a.lui(rd, hi);
                if (lo != 0) try a.addiw(rd, rd, lo);
                return;
            }
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
            try a.li(.t0, @bitCast(addr));
            try a.jalr(.ra, 0, .t0);
        }
        /// `addi sp, sp, -n; sd ra, n-8(sp); sd s0, n-16(sp); addi s0, sp, n; sd <saved>...`,
        /// n = 16-aligned frame (ra + s0 + saved + locals). Returns n.
        pub fn prolog(a: Self, saved: []const X, locals: u32) Error!i32 {
            const n: i32 = @intCast(std.mem.alignForward(u32, 16 + 8 * @as(u32, @intCast(saved.len)) + locals, 16));
            try a.addi(.sp, .sp, -n);
            try a.sd(.ra, n - 8, .sp);
            try a.sd(.s0, n - 16, .sp);
            for (saved, 0..) |r, i| try a.sd(r, n - 24 - 8 * @as(i32, @intCast(i)), .sp);
            try a.addi(.s0, .sp, n);
            return n;
        }
        pub fn epilog(a: Self, saved: []const X, n: i32) Error!void {
            for (saved, 0..) |r, i| try a.ld(r, n - 24 - 8 * @as(i32, @intCast(i)), .sp);
            try a.ld(.s0, n - 16, .sp);
            try a.ld(.ra, n - 8, .sp);
            try a.addi(.sp, .sp, n);
            try a.ret();
        }
    };
}
pub const Assembler = AssemblerT(.{});

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
