//! AArch64 Compiler: IR (`core/ir.zig`) -> linear-scan register allocation
//! (`core/regalloc.zig`) -> A64 machine code (AAPCS64), NEON for the 128-bit vector types.
//!
//! Conventions:
//! - GP allocatable: x0..x7 and x9..x15 (caller-saved), x19..x28 (callee-saved). Intervals
//!   that cross a call only get callee-saved registers; in call-free functions the register
//!   parameters are pinned to their argument registers (no entry moves). x18 (platform register
//!   on Apple) is never touched; x8, x16 and x17 are scratch (large offsets, spilled operands,
//!   call targets, the cycle temp of parallel moves).
//! - FP/vector allocatable: v0..v7 and v16..v29, all caller-saved (v8..v15 are only partly
//!   callee-saved in AAPCS64 and never used); v30/v31 are scratch, and v29 too in functions
//!   with a vector `min`/`max`. Scalars use the s/d views, vectors the full q register.
//! - Constants: a vreg is a known constant where its value is fixed by a `const` earlier in
//!   the same block, or everywhere when its only definition is a `const` in block 0 before
//!   any use there. Known operands become immediates (add/sub/cmp/cmn imm12, logical
//!   immediates, shifts, cbz/cbnz, xzr); a constant whose uses are all folded is never
//!   materialized and takes no register.
//! - frame: none at all for leaves without calls, spills or callee-saved registers;
//!   otherwise `Assembler.prolog(used callee-saved, out + slots)`: outgoing stack arguments
//!   at [sp + 8i], GP spill slot k at [sp + out + 8k], FP slot k at [sp + fbase + size*k]
//!   (16-byte slots when the function has a vector vreg); sp never moves inside the body.
//! - Blocks are laid out in IR order: no branch to the next block, a `ret` in the last block
//!   falls into the epilog, short epilogs are duplicated at each `ret`. Conditional branches
//!   that may be out of the ±1 MiB range go over an unconditional `b`.
//! - parameters and call arguments past x7 use 8-byte stack slots (up to `ir.max_args`);
//!   incoming ones are read at [x29 + 16 + 8i] (or [sp + 8i] without a frame).
const std = @import("std");
const a64 = @import("assembler.zig");
const imm = @import("a64.zig");
const ir = @import("../core/ir.zig");
const ra = @import("../core/regalloc.zig");
const rc = @import("../core/reg_classes.zig");
const CodeBuffer = @import("../core/code_buffer.zig").CodeBuffer;
const X = a64.X;
const CC = a64.CallConv;

/// GP scratch registers (never allocated).
const s0: u5 = 16;
const s1: u5 = 17;
const s2: u5 = 8;
/// Register number 31: xzr/wzr in data-processing operands, sp as a base or in add/sub imm.
const zr: u5 = 31;
const spr: u5 = 31;
/// FP/vector scratch registers.
const fs0: u5 = 30;
const fs1: u5 = 31;
const fs2: u5 = 29;

fn bit(r: X) ra.RegMask {
    return @as(ra.RegMask, 1) << @backingInt(r);
}
fn maskOf(comptime rs: []const X) ra.RegMask {
    var m: ra.RegMask = 0;
    for (rs) |r| m |= bit(r);
    return m;
}
pub const callee_saved: ra.RegMask = maskOf(&CC.callee_saved);
pub const allocatable: ra.RegMask = maskOf(&CC.args) | maskOf(&.{ .x9, .x10, .x11, .x12, .x13, .x14, .x15 }) | callee_saved;

/// FP class (f32/f64 scalars and 128-bit vectors in v registers, `core/reg_classes.zig`):
/// v0..v7 and v16..v29, all caller-saved; v30/v31 are scratch.
pub const fp_allocatable: ra.RegMask = 0xFF | (0x3FFF << 16);

pub const Options = struct {
    /// Restrict the allocatable set (spill stress tests); masked with `allocatable`.
    allocatable: ra.RegMask = allocatable,
    /// Restrict the FP allocatable set (spill stress tests); masked with `fp_allocatable`.
    fp_allocatable: ra.RegMask = fp_allocatable,
};

pub const Stats = struct { spills: u32 = 0, slots: u32 = 0 };

/// Raw A64 encodings used by the compiler (registers as numbers; 31 is xzr or sp depending on
/// the form, as in the architecture). Checked against LLVM (`llvm-mc`) in the tests.
pub const enc = struct {
    inline fn u(x: anytype) u32 {
        return @intCast(x);
    }
    // ---- condition codes (A64 `cond` field) ----
    pub const eq: u4 = 0;
    pub const ne: u4 = 1;
    pub const hs: u4 = 2;
    pub const lo: u4 = 3;
    pub const mi: u4 = 4;
    pub const vs: u4 = 6;
    pub const vc: u4 = 7;
    pub const hi: u4 = 8;
    pub const ls: u4 = 9;
    pub const ge: u4 = 10;
    pub const lt: u4 = 11;
    pub const gt: u4 = 12;
    pub const le: u4 = 13;

    // ---- GP data processing ----
    pub const add_r: u32 = 0x8B000000;
    pub const sub_r: u32 = 0xCB000000;
    pub const and_r: u32 = 0x8A000000;
    pub const orr_r: u32 = 0xAA000000;
    pub const eor_r: u32 = 0xCA000000;
    pub const lslv: u32 = 0x9AC02000;
    pub const lsrv: u32 = 0x9AC02400;
    pub const asrv: u32 = 0x9AC02800;
    pub const sdiv: u32 = 0x9AC00C00;
    pub const mul: u32 = 0x9B007C00;
    /// `<op> x<d>, x<n>, x<m>` (also NEON three-same forms with `v` registers).
    pub fn rrr(base: u32, d: u5, n: u5, m: u5) u32 {
        return base | u(m) << 16 | u(n) << 5 | d;
    }
    /// madd x<d>, x<n>, x<m>, x<a> (`sub`: msub).
    pub fn madd(sub: bool, d: u5, n: u5, m: u5, a: u5) u32 {
        return @as(u32, if (sub) 0x9B008000 else 0x9B000000) | u(m) << 16 | u(a) << 10 | u(n) << 5 | d;
    }
    /// add|sub x<d>, x<n>|sp, #imm12 {, lsl #12}.
    pub fn addImm(sub: bool, d: u5, n: u5, imm12: u12, sh12: u1) u32 {
        return @as(u32, if (sub) 0xD1000000 else 0x91000000) | u(sh12) << 22 | u(imm12) << 10 | u(n) << 5 | d;
    }
    /// cmp|cmn x<n>, #imm12 {, lsl #12}.
    pub fn cmpImm(cmn: bool, n: u5, imm12: u12, sh12: u1) u32 {
        return @as(u32, if (cmn) 0xB1000000 else 0xF1000000) | u(sh12) << 22 | u(imm12) << 10 | u(n) << 5 | 31;
    }
    /// cmp x<n>, x<m>.
    pub fn cmpReg(n: u5, m: u5) u32 {
        return 0xEB000000 | u(m) << 16 | u(n) << 5 | 31;
    }
    pub const and_i: u32 = 0x92000000;
    pub const orr_i: u32 = 0xB2000000;
    pub const eor_i: u32 = 0xD2000000;
    /// and|orr|eor x<d>, x<n>, #bitmask (`li` from `bitmask`).
    pub fn logImm(base: u32, d: u5, n: u5, li: anytype) u32 {
        return base | u(li.n) << 22 | u(li.immr) << 16 | u(li.imms) << 10 | u(n) << 5 | d;
    }
    /// lsl|lsr|asr x<d>, x<n>, #s (ubfm/sbfm aliases).
    pub fn shiftImm(op: ir.Arith, d: u5, n: u5, s: u6) u32 {
        const rn = u(n) << 5 | d;
        return switch (op) {
            .shl => 0xD3400000 | u((64 - @as(u7, s)) & 63) << 16 | u(63 - s) << 10 | rn,
            .shr => 0xD3400000 | u(s) << 16 | 63 << 10 | rn,
            .sar => 0x93400000 | u(s) << 16 | 63 << 10 | rn,
            else => unreachable,
        };
    }
    pub const movz: u32 = 0xD2800000;
    pub const movn: u32 = 0x92800000;
    pub const movk: u32 = 0xF2800000;
    /// movz|movn|movk x<d>, #imm16, lsl #(16 hw).
    pub fn movw(base: u32, d: u5, imm16: u16, hw: u2) u32 {
        return base | u(hw) << 21 | u(imm16) << 5 | d;
    }
    /// mov x<d>, x<m> (orr x<d>, xzr, x<m>).
    pub fn mov(d: u5, m: u5) u32 {
        return 0xAA0003E0 | u(m) << 16 | d;
    }
    /// cset x<d>, cond (csinc x<d>, xzr, xzr, !cond).
    pub fn cset(d: u5, cond: u4) u32 {
        return 0x9A9F07E0 | u(cond ^ 1) << 12 | d;
    }
    /// csel x<d>, x<n>, x<m>, cond.
    pub fn csel(d: u5, n: u5, m: u5, cond: u4) u32 {
        return 0x9A800000 | u(m) << 16 | u(cond) << 12 | u(n) << 5 | d;
    }
    /// ccmn x<n>, #imm5, #nzcv, cond.
    pub fn ccmnImm(n: u5, im5: u5, nzcv: u4, cond: u4) u32 {
        return 0xBA400800 | u(im5) << 16 | u(cond) << 12 | u(n) << 5 | nzcv;
    }
    pub fn blr(n: u5) u32 {
        return 0xD63F0000 | u(n) << 5;
    }
    pub const ret: u32 = 0xD65F03C0;
    /// b.cond / cbz / cbnz over the next instruction (+8).
    pub fn bcondSkip(cond: u4) u32 {
        return 0x54000040 | u(cond);
    }
    pub fn cbzSkip(nz: bool, rt: u5) u32 {
        return @as(u32, if (nz) 0xB5000040 else 0xB4000040) | rt;
    }

    // ---- loads and stores (x, s, d and q registers) ----
    pub const Mem = enum { x, s, d, q };
    fn memBase(k: Mem) u32 {
        return switch (k) {
            .x => 0xF8000000,
            .d => 0xFC000000,
            .s => 0xBC000000,
            .q => 0x3C800000,
        };
    }
    pub fn memSize(k: Mem) u32 {
        return switch (k) {
            .s => 4,
            .x, .d => 8,
            .q => 16,
        };
    }
    pub fn memOf(tp: ir.Type) Mem {
        return switch (tp) {
            .i64 => .x,
            .f32 => .s,
            .f64 => .d,
            else => .q,
        };
    }
    /// ldr|str <t>, [x<n>|sp, #imm12 * size].
    pub fn ldstImm(load: bool, k: Mem, rt: u5, n: u5, imm12: u12) u32 {
        return memBase(k) | @as(u32, if (load) 0x400000 else 0) | 0x01000000 | u(imm12) << 10 | u(n) << 5 | rt;
    }
    /// ldur|stur <t>, [x<n>|sp, #simm9].
    pub fn ldstUnscaled(load: bool, k: Mem, rt: u5, n: u5, simm9: i9) u32 {
        return memBase(k) | @as(u32, if (load) 0x400000 else 0) | u(@as(u9, @bitCast(simm9))) << 12 | u(n) << 5 | rt;
    }
    /// ldr|str <t>, [x<n>|sp, x<m>].
    pub fn ldstReg(load: bool, k: Mem, rt: u5, n: u5, m: u5) u32 {
        return memBase(k) | @as(u32, if (load) 0x400000 else 0) | 0x206800 | u(m) << 16 | u(n) << 5 | rt;
    }

    // ---- scalar FP (`ft` = 0 single, 1 double) ----
    fn ft(tp: ir.Type) u32 {
        return if (tp == .f64) 1 << 22 else 0;
    }
    pub const fadd: u32 = 0x1E202800;
    pub const fsub: u32 = 0x1E203800;
    pub const fmul: u32 = 0x1E200800;
    pub const fdiv: u32 = 0x1E201800;
    /// f<op> s|d <d>, <n>, <m>.
    pub fn frrr(base: u32, tp: ir.Type, d: u5, n: u5, m: u5) u32 {
        return rrr(base | ft(tp), d, n, m);
    }
    pub fn fcmp(tp: ir.Type, n: u5, m: u5) u32 {
        return 0x1E202000 | ft(tp) | u(m) << 16 | u(n) << 5;
    }
    pub fn fcsel(tp: ir.Type, d: u5, n: u5, m: u5, cond: u4) u32 {
        return 0x1E200C00 | ft(tp) | u(m) << 16 | u(cond) << 12 | u(n) << 5 | d;
    }
    /// fmov d<d>, d<n> (copies the low 64 bits).
    pub fn fmovRR(d: u5, n: u5) u32 {
        return 0x1E604000 | u(n) << 5 | d;
    }
    /// fmov s<d>, w<n> / fmov d<d>, x<n> (n = 31: zero register).
    pub fn fmovFromGp(tp: ir.Type, d: u5, n: u5) u32 {
        return @as(u32, if (tp == .f64) 0x9E670000 else 0x1E270000) | u(n) << 5 | d;
    }
    /// fmov s|d <d>, #imm8 (`a64.fpImm8`).
    pub fn fmovImm(tp: ir.Type, d: u5, imm8: u8) u32 {
        return 0x1E201000 | ft(tp) | u(imm8) << 13 | d;
    }
    /// movi v<d>.2d, #0.
    pub fn moviZero(d: u5) u32 {
        return 0x6F00E400 | u(d);
    }
    /// scvtf s|d <d>, x<n>.
    pub fn scvtf(tp: ir.Type, d: u5, n: u5) u32 {
        return @as(u32, if (tp == .f64) 0x9E620000 else 0x9E220000) | u(n) << 5 | d;
    }
    /// fcvtzs x<d>, s|d <n>.
    pub fn fcvtzs(tp: ir.Type, d: u5, n: u5) u32 {
        return @as(u32, if (tp == .f64) 0x9E780000 else 0x9E380000) | u(n) << 5 | d;
    }
    /// fcvt d<d>, s<n> (to = f64) or fcvt s<d>, d<n> (to = f32).
    pub fn fcvt(to: ir.Type, d: u5, n: u5) u32 {
        return @as(u32, if (to == .f64) 0x1E22C000 else 0x1E624000) | u(n) << 5 | d;
    }

    // ---- NEON (three-same forms take `rrr`) ----
    pub const add_4s: u32 = 0x4EA08400;
    pub const add_2d: u32 = 0x4EE08400;
    pub const sub_4s: u32 = 0x6EA08400;
    pub const sub_2d: u32 = 0x6EE08400;
    pub const mul_4s: u32 = 0x4EA09C00;
    pub const fadd_4s: u32 = 0x4E20D400;
    pub const fsub_4s: u32 = 0x4EA0D400;
    pub const fmul_4s: u32 = 0x6E20DC00;
    pub const fdiv_4s: u32 = 0x6E20FC00;
    pub const fcmgt_4s: u32 = 0x6EA0E400;
    /// `.4s` float form -> `.2d` (sz bit).
    pub const sz_2d: u32 = 0x00400000;
    pub const and_16b: u32 = 0x4E201C00;
    pub const orr_16b: u32 = 0x4EA01C00;
    pub const eor_16b: u32 = 0x6E201C00;
    pub const bsl_16b: u32 = 0x6E601C00;
    pub const bit_16b: u32 = 0x6EA01C00;
    pub const bif_16b: u32 = 0x6EE01C00;
    /// mov v<d>.16b, v<n>.16b.
    pub fn movV(d: u5, n: u5) u32 {
        return rrr(orr_16b, d, n, n);
    }
    /// dup v<d>.4s, w<n> (`d64` = false) / dup v<d>.2d, x<n>.
    pub fn dupGp(d64: bool, d: u5, n: u5) u32 {
        return @as(u32, if (d64) 0x4E080C00 else 0x4E040C00) | u(n) << 5 | d;
    }
    /// dup v<d>.4s, v<n>.s[0] / dup v<d>.2d, v<n>.d[0].
    pub fn dupElem0(d64: bool, d: u5, n: u5) u32 {
        return @as(u32, if (d64) 0x4E080400 else 0x4E040400) | u(n) << 5 | d;
    }
    fn imm5(d64: bool, lane: u8) u32 {
        return if (d64) (u(lane) << 4 | 8) << 16 else (u(lane) << 3 | 4) << 16;
    }
    /// smov x<d>, v<n>.s[lane].
    pub fn smovS(d: u5, n: u5, lane: u8) u32 {
        return 0x4E002C00 | imm5(false, lane) | u(n) << 5 | d;
    }
    /// umov x<d>, v<n>.d[lane].
    pub fn umovD(d: u5, n: u5, lane: u8) u32 {
        return 0x4E003C00 | imm5(true, lane) | u(n) << 5 | d;
    }
    /// mov s<d>, v<n>.s[lane] / mov d<d>, v<n>.d[lane] (dup, scalar).
    pub fn dupScalar(d64: bool, d: u5, n: u5, lane: u8) u32 {
        return 0x5E000400 | imm5(d64, lane) | u(n) << 5 | d;
    }
    /// mov v<d>.d[lane], x<n>.
    pub fn insD(d: u5, lane: u8, n: u5) u32 {
        return 0x4E001C00 | imm5(true, lane) | u(n) << 5 | d;
    }
};

// ---- constant analysis ------------------------------------------------------------------

const global: u32 = std.math.maxInt(u32);

/// Known constant vregs. `gen[v]` is `global` (known everywhere), the stamp of the block walk
/// that set it, or stale; `begin` starts a block.
const Consts = struct {
    val: []i64,
    gen: []u32,
    stamp: u32 = 0,
    inline fn begin(c: *Consts) void {
        c.stamp += 1;
    }
    inline fn get(c: *const Consts, v: ir.VReg) ?i64 {
        const g = c.gen[v];
        return if (g == global or g == c.stamp) c.val[v] else null;
    }
    inline fn set(c: *Consts, v: ir.VReg, k: ?i64) void {
        if (c.gen[v] == global) return;
        if (k) |x| {
            c.val[v] = x;
            c.gen[v] = c.stamp;
        } else c.gen[v] = 0;
    }
    inline fn zero(c: *const Consts, v: ir.VReg) bool {
        return (c.get(v) orelse return false) == 0;
    }
    /// Known and one instruction to materialize.
    inline fn cheap(c: *const Consts, v: ir.VReg) ?i64 {
        const k = c.get(v) orelse return null;
        return if (movLen(@bitCast(k)) == 1) k else null;
    }
};

fn ub(op: ir.Arith, a: i64, b: i64) bool {
    return (op == .sdiv or op == .srem) and (b == 0 or (a == std.math.minInt(i64) and b == -1));
}

const AddImm = struct { sub: bool, imm: u12, sh: u1 };
/// `x + k` as one add/sub immediate (imm12, optionally shifted by 12).
fn addImm(k: i64) ?AddImm {
    const P = struct {
        fn pos(v: i64) ?struct { u12, u1 } {
            if (v >= 0 and v < 4096) return .{ @intCast(v), 0 };
            if (v > 0 and v & 0xFFF == 0 and v >> 12 < 4096) return .{ @intCast(v >> 12), 1 };
            return null;
        }
    };
    if (P.pos(k)) |p| return .{ .sub = false, .imm = p[0], .sh = p[1] };
    if (k != std.math.minInt(i64)) if (P.pos(-k)) |p| return .{ .sub = true, .imm = p[0], .sh = p[1] };
    return null;
}

fn immOk(op: ir.Arith, k: i64) bool {
    return switch (op) {
        .add, .sub => addImm(k) != null,
        .mul => @popCount(@as(u64, @bitCast(k))) == 1,
        .@"and", .@"or", .xor => bitmask(@bitCast(k)) != null,
        .shl, .shr, .sar => true,
        .sdiv, .srem => false,
    };
}

const ArithOp = @FieldType(ir.Op, "arith");
const CbrOp = @FieldType(ir.Op, "cbr");

/// How an `arith` is lowered: a constant result, an immediate form (`swap`: the constant is
/// operand `a` of a commutative op), or registers (`za`/`zb`: operand is 0, read as xzr).
const APlan = union(enum) {
    konst: i64,
    imm: struct { swap: bool, k: i64 },
    reg: struct { za: bool, zb: bool },
};
fn aplan(c: *const Consts, x: ArithOp) APlan {
    const ka = c.get(x.a);
    const kb = c.get(x.b);
    if (ka != null and kb != null and !ub(x.op, ka.?, kb.?)) return .{ .konst = ir.arith(x.op, ka.?, kb.?) };
    if (kb) |k| if (immOk(x.op, k)) return .{ .imm = .{ .swap = false, .k = k } };
    if (ka) |k| switch (x.op) {
        .add, .mul, .@"and", .@"or", .xor => if (immOk(x.op, k)) return .{ .imm = .{ .swap = true, .k = k } },
        else => {},
    };
    return .{ .reg = .{ .za = ka != null and ka.? == 0, .zb = kb != null and kb.? == 0 } };
}

fn condOf(cc: ir.CondCode) u4 {
    return switch (cc) {
        .eq => enc.eq,
        .ne => enc.ne,
        .lt => enc.lt,
        .le => enc.le,
        .gt => enc.gt,
        .ge => enc.ge,
        .ult => enc.lo,
        .ule => enc.ls,
        .ugt => enc.hi,
        .uge => enc.hs,
    };
}
/// `b <cc> a` as `a <swapped cc> b`.
fn swapped(cc: ir.CondCode) ir.CondCode {
    return switch (cc) {
        .eq, .ne => cc,
        .lt => .gt,
        .gt => .lt,
        .le => .ge,
        .ge => .le,
        .ult => .ugt,
        .ugt => .ult,
        .ule => .uge,
        .uge => .ule,
    };
}

/// How a `cbr` is lowered: decided at compile time, cmp of registers, cmp/cmn of `v` with an
/// immediate (condition already adjusted for operand order), or cbz/cbnz of `v`.
const CPlan = union(enum) {
    static: bool,
    reg,
    imm: struct { v: ir.VReg, cond: u4, ai: AddImm },
    zero: struct { v: ir.VReg, nz: bool },
};
fn cplan(c: *const Consts, x: CbrOp) CPlan {
    const ka = c.get(x.a);
    const kb = c.get(x.b);
    if (ka != null and kb != null) return .{ .static = ir.cond(x.cond, ka.?, kb.?) };
    inline for (.{ false, true }) |sw| {
        const k = if (sw) ka else kb;
        const v = if (sw) x.b else x.a;
        const cc = if (sw) swapped(x.cond) else x.cond;
        if (k) |kk| {
            if (kk == 0 and (cc == .eq or cc == .ne)) return .{ .zero = .{ .v = v, .nz = cc == .ne } };
            if (addImm(kk)) |ai| return .{ .imm = .{ .v = v, .cond = condOf(cc), .ai = ai } };
        }
    }
    return .reg;
}

const LogImm = struct { n: u1, immr: u6, imms: u6 };
/// 64-bit bitmask immediate of `v` (same result as `a64.logicalImm(v, true)`, loop-free): `v`
/// repeats an element of 2..64 bits holding one (rotated) run of ones, so it has 128 / size
/// circular bit transitions and equals itself rotated by the element size.
fn bitmask(v: u64) ?LogImm {
    const tr = @popCount(v ^ std.math.rotr(u64, v, 1));
    if (tr == 0 or tr & (tr - 1) != 0) return null; // 0, ~0, or not 2 * 2^k transitions
    const size: u7 = @intCast(128 / @as(u32, tr));
    if (size < 64 and v != std.math.rotr(u64, v, size)) return null;
    const mask: u64 = if (size == 64) ~@as(u64, 0) else (@as(u64, 1) << @intCast(size)) - 1;
    const e = v & mask;
    const ones: u7 = @popCount(e);
    // Rotation `r` with rotr(e, r) == ones mask: the start of the run of ones.
    const r: u7 = if (e & 1 == 0) @ctz(e) else (@as(u7, @ctz(~e)) + size - ones) % size;
    return .{
        .n = @intFromBool(size == 64),
        .immr = @intCast((size - r) % size),
        .imms = @intCast(((~(size - 1) << 1) & 0x3F) | (ones - 1)),
    };
}

/// Instructions `movImm` needs for `v`.
fn movLen(v: u64) u32 {
    var z: u32 = 0;
    var o: u32 = 0;
    inline for (0..4) |hw| {
        const h: u16 = @truncate(v >> (16 * hw));
        z += @intFromBool(h == 0);
        o += @intFromBool(h == 0xFFFF);
    }
    const n = 4 - @max(z, o);
    if (n <= 1) return 1;
    return if (bitmask(v) != null) 1 else n;
}

/// Extra `analyze` mask bits of an `arith`: the `APlan` kind (`reg` when neither is set).
const plan_konst: u16 = 1 << 4;
const plan_imm: u16 = 1 << 5;

/// The `aplan` that `analyze` recorded in `m` (constant operands are read from `c`).
inline fn planOf(m: u16, c: *const Consts, x: ArithOp) APlan {
    if (m & plan_konst != 0) return .{ .konst = ir.arith(x.op, c.get(x.a).?, c.get(x.b).?) };
    if (m & plan_imm != 0) {
        const swap = m & 2 != 0;
        return .{ .imm = .{ .swap = swap, .k = c.get(if (swap) x.a else x.b).? } };
    }
    return .{ .reg = .{ .za = m & 1 == 0, .zb = m & 2 == 0 } };
}

/// The analysis of `o` at the constant state `an.cs`, in one dispatch: returns the operand
/// slots read from registers (uses not folded into immediates, xzr or constant results; bit
/// 0/1 = first/second operand (`a`/`b`, `src`, `base` then `src` for stores), bit i = argument
/// i of a call; plus `plan_*` bits for `arith`), marks them in `an.needs` and reports them to
/// `ctx.use`, then reports the definition to `ctx.def` and steps `an.cs` past `o`.
inline fn analyze(o: *const ir.Op, an: *Analysis, ctx: anytype) ra.Error!u16 {
    const c = &an.cs;
    const needs = an.needs;
    const U = struct {
        /// A use of any class.
        inline fn use(n: []bool, cx: @TypeOf(ctx), v: ir.VReg) ra.Error!void {
            n[v] = true;
            try cx.use(v);
        }
        /// A use of a GP operand.
        inline fn gp(n: []bool, cx: @TypeOf(ctx), v: ir.VReg) ra.Error!void {
            n[v] = true;
            try cx.useAs(v, false);
        }
        /// A use of an operand of type `ty`.
        inline fn typed(n: []bool, cx: @TypeOf(ctx), v: ir.VReg, ty: ir.Type) ra.Error!void {
            n[v] = true;
            try cx.useAs(v, ty != .i64);
        }
    };
    switch (o.*) {
        .@"const" => |x| {
            try ctx.defAs(x.dst, false);
            c.set(x.dst, x.imm);
        },
        .fconst => |x| {
            try ctx.defAs(x.dst, true);
            c.set(x.dst, null);
        },
        .br => {},
        .copy => |x| {
            const k = c.get(x.src);
            const reg = k == null or movLen(@bitCast(k.?)) != 1;
            if (reg) try U.use(needs, ctx, x.src);
            try ctx.def(x.dst);
            c.set(x.dst, k);
            return @intFromBool(reg);
        },
        .arith => |x| {
            const m: u16 = switch (aplan(c, x)) {
                .konst => |k| {
                    try ctx.defAs(x.dst, false);
                    c.set(x.dst, k);
                    return plan_konst;
                },
                .imm => |p| plan_imm | @as(u16, if (p.swap) 2 else 1),
                .reg => |p| @as(u16, @intFromBool(!p.za)) | @as(u16, @intFromBool(!p.zb)) << 1,
            };
            if (m & 1 != 0) try U.gp(needs, ctx, x.a);
            if (m & 2 != 0) try U.gp(needs, ctx, x.b);
            try ctx.defAs(x.dst, false);
            c.set(x.dst, null);
            return m;
        },
        .cbr => |x| {
            const m: u16 = switch (cplan(c, x)) {
                .static => 0,
                .reg => 3,
                inline .imm, .zero => |p| if (p.v == x.a) 1 else 2,
            };
            if (m & 1 != 0) try U.gp(needs, ctx, x.a);
            if (m & 2 != 0) try U.gp(needs, ctx, x.b);
            return m;
        },
        .load => |x| {
            try U.gp(needs, ctx, x.base);
            try ctx.defAs(x.dst, x.ty != .i64);
            c.set(x.dst, null);
            return 1;
        },
        .store => |x| {
            try U.gp(needs, ctx, x.base);
            if (x.ty == .i64 and c.zero(x.src)) return 1;
            try U.typed(needs, ctx, x.src, x.ty);
            return 3;
        },
        .call => |x| {
            var m: u16 = 0;
            for (x.args, 0..) |v, i| if (c.cheap(v) == null) {
                try U.gp(needs, ctx, v);
                m |= @as(u16, 1) << @intCast(i);
            };
            try ctx.defAs(x.dst, false);
            c.set(x.dst, null);
            return m;
        },
        .ret => |v| if (c.get(v) == null) {
            try U.gp(needs, ctx, v);
            return 1;
        },
        inline .typed, .fcmp => |x, tag| {
            if (tag == .typed and x.ty.isVector() and (x.op == .min or x.op == .max)) an.vminmax = true;
            try U.typed(needs, ctx, x.a, x.ty);
            try U.typed(needs, ctx, x.b, x.ty);
            try ctx.defAs(x.dst, tag == .typed);
            c.set(x.dst, null);
            return 3;
        },
        inline .cvt, .splat => |x, tag| {
            const src: ir.Type = if (tag == .cvt) x.from else x.ty.lane();
            const zero = src == .i64 and c.zero(x.src);
            if (!zero) try U.typed(needs, ctx, x.src, src);
            try ctx.defAs(x.dst, if (tag == .cvt) x.to != .i64 else true);
            c.set(x.dst, null);
            return @intFromBool(!zero);
        },
        .extract => |x| {
            try U.typed(needs, ctx, x.src, x.ty);
            try ctx.defAs(x.dst, x.ty.lane() != .i64);
            c.set(x.dst, null);
            return 1;
        },
        .bra, .cbra => unreachable, // removed by lowerBlockParams
    }
    return 0;
}

/// Reports every operand of `o` (lowered: no block arguments).
inline fn forUses(o: *const ir.Op, ctx: anytype) void {
    switch (o.*) {
        inline .copy, .cvt, .splat, .extract => |x| ctx.use(x.src),
        .load => |x| ctx.use(x.base),
        .ret => |v| ctx.use(v),
        .store => |x| {
            ctx.use(x.base);
            ctx.use(x.src);
        },
        inline .arith, .cbr, .typed, .fcmp => |x| {
            ctx.use(x.a);
            ctx.use(x.b);
        },
        .call => |x| for (x.args) |v| ctx.use(v),
        .@"const", .fconst, .br => {},
        .bra, .cbra => unreachable, // removed by lowerBlockParams
    }
}

const Analysis = struct {
    cs: Consts,
    /// `v` is read from a register somewhere (some use is not folded).
    needs: []bool,
    /// `analyze` mask of every op (function-wide op index): folding in the register
    /// allocation input and in the emitter agree.
    masks: []u16,
    types: []const ir.Type,
    params: u32,
    /// A vector min/max (it needs a third FP scratch register).
    vminmax: bool = false,
};

const Block = struct {
    insts: struct { ops: []const ir.Op, params: bool, len: usize },
    succs: []const u32,
    /// Function-wide index of the first op.
    first: u32,
};

/// Register-allocation input for one class: like `reg_classes.Source`, but uses folded into
/// immediates are not reported. The GP walk runs `analyze` (it completes `Analysis`); the FP
/// walk, after it, also skips definitions of unneeded vregs (emitted as nothing).
fn Source(comptime class: rc.Class) type {
    return struct {
        blocks: []const Block,
        num_vregs: u32,
        fixed: []const ?u8 = &.{},
        an: *Analysis,

        pub inline fn isCall(_: @This(), b: Block, i: usize) bool {
            if (b.insts.params and i == 0) return false;
            return b.insts.ops[i - @intFromBool(b.insts.params)] == .call;
        }
        pub inline fn collectBlock(s: @This(), b: Block, collector: anytype) ra.Error!void {
            const an = s.an;
            const C = struct {
                types: []const ir.Type,
                out: @TypeOf(collector),
                inline fn use(c: @This(), v: ir.VReg) ra.Error!void {
                    if (rc.classOf(c.types[v]) == class) try c.out.use(v);
                }
                inline fn def(c: @This(), v: ir.VReg) ra.Error!void {
                    if (rc.classOf(c.types[v]) == class) try c.out.def(v);
                }
                /// `use`/`def` of a vreg whose class the op fixes (`fp`: FP class).
                inline fn useAs(c: @This(), v: ir.VReg, fp: bool) ra.Error!void {
                    if (fp == (class == .fp)) try c.out.use(v);
                }
                inline fn defAs(c: @This(), v: ir.VReg, fp: bool) ra.Error!void {
                    if (fp == (class == .fp)) try c.out.def(v);
                }
            };
            const c: C = .{ .types = an.types, .out = collector };
            if (b.insts.params) {
                if (class == .gp) for (0..@min(an.params, s.num_vregs)) |p| try collector.def(@intCast(p));
                collector.next();
            }
            const masks = an.masks[b.first..][0..b.insts.ops.len];
            if (class == .gp) {
                // The analysis pass runs here: `needs` and `masks` are complete for the FP
                // class and the emitter. Every block is a fresh constant state, so the order
                // and a repeated walk (`ra.verifySource`) give the same result.
                an.cs.begin();
                for (b.insts.ops, masks) |*o, *m| {
                    m.* = try analyze(o, an, c);
                    collector.next();
                }
                return;
            }
            const needs = an.needs;
            for (b.insts.ops, masks) |*o, m| {
                // Uses selected by `m`, then the definition; pure definitions of unneeded
                // vregs (emitted as nothing) report neither.
                switch (o.*) {
                    .br, .@"const", .arith, .call, .cbr, .ret => {},
                    .fconst => |x| if (needs[x.dst]) try collector.def(x.dst),
                    .copy => |x| if (needs[x.dst]) {
                        if (m & 1 != 0) try c.use(x.src);
                        try c.def(x.dst);
                    },
                    inline .cvt, .splat, .extract => |x, tag| if (needs[x.dst]) {
                        const src: ir.Type, const dst: ir.Type = switch (tag) {
                            .cvt => .{ x.from, x.to },
                            .splat => .{ x.ty.lane(), x.ty },
                            else => .{ x.ty, x.ty.lane() },
                        };
                        if (m & 1 != 0) try c.useAs(x.src, src != .i64);
                        try c.defAs(x.dst, dst != .i64);
                    },
                    .typed => |x| if (needs[x.dst]) {
                        try collector.use(x.a);
                        try collector.use(x.b);
                        try collector.def(x.dst);
                    },
                    .fcmp => |x| if (needs[x.dst]) {
                        try collector.use(x.a);
                        try collector.use(x.b);
                    },
                    .load => |x| if (needs[x.dst] and x.ty != .i64) try collector.def(x.dst),
                    .store => |x| if (m & 2 != 0 and x.ty != .i64) try collector.use(x.src),
                    .bra, .cbra => unreachable, // removed by lowerBlockParams
                }
                collector.next();
            }
        }
    };
}

fn buildBlocks(A: std.mem.Allocator, f: *const ir.Func) ![]Block {
    var entry_target = false;
    for (f.blocks.items) |b| switch (b.ops.items[b.ops.items.len - 1]) {
        .br => |target| entry_target = entry_target or target == 0,
        .cbr => |c| entry_target = entry_target or c.then == 0 or c.@"else" == 0,
        else => {},
    };
    const pre: usize = @intFromBool(entry_target);
    const blocks = try A.alloc(Block, f.blocks.items.len + pre);
    const edges = try A.alloc(u32, 2 * blocks.len);
    var si: usize = 0;
    if (pre != 0) {
        edges[0] = 1;
        blocks[0] = .{ .insts = .{ .ops = &.{}, .params = true, .len = 1 }, .succs = edges[0..1], .first = 0 };
        si = 1;
    }
    var first: u32 = 0;
    for (f.blocks.items, 0..) |b, bi| {
        const start = si;
        switch (b.ops.items[b.ops.items.len - 1]) {
            .br => |target| {
                edges[si] = @intCast(target + pre);
                si += 1;
            },
            .cbr => |c| {
                edges[si] = @intCast(c.then + pre);
                edges[si + 1] = @intCast(c.@"else" + pre);
                si += 2;
            },
            else => {},
        }
        const params = pre == 0 and bi == 0;
        blocks[bi + pre] = .{ .insts = .{ .ops = b.ops.items, .params = params, .len = b.ops.items.len + @intFromBool(params) }, .succs = edges[start..si], .first = first };
        first += @intCast(b.ops.items.len);
    }
    return blocks;
}

/// Branch: `cond` (b.cond) or cbz/cbnz of a register.
const Br = union(enum) {
    cond: u4,
    cbz: struct { nz: bool, r: u5 },
    fn invert(b: Br) Br {
        return switch (b) {
            .cond => |c| .{ .cond = c ^ 1 },
            .cbz => |z| .{ .cbz = .{ .nz = !z.nz, .r = z.r } },
        };
    }
};

fn Emitter(comptime Asm: type) type {
    return struct {
        const Self = @This();
        a: Asm,
        locs: []const ra.Loc,
        flocs: []const ra.Loc,
        out: u32,
        fbase: u32,
        /// FP spill slots hold q registers (the function has vector vregs).
        qspill: bool,
        maybe_far: bool,
        /// Upper bound of the words before each block (and before the epilog).
        pre: []const u32,
        labels: []const a64.Label,
        cur: usize = 0,

        inline fn w(e: *const Self, x: u32) !void {
            try e.a.word(x);
        }
        inline fn xr(r: u5) X {
            return @fromBackingInt(r);
        }
        /// Shortest mov sequence for `v` (one movz/movn inline, the rest in `movImmLong`).
        inline fn movImm(e: *const Self, d: u5, v: u64) !void {
            if (v <= 0xFFFF) return e.w(enc.movw(enc.movz, d, @intCast(v), 0));
            if (~v <= 0xFFFF) return e.w(enc.movw(enc.movn, d, @intCast(~v), 0));
            return e.movImmLong(d, v);
        }
        fn movImmLong(e: *const Self, d: u5, v: u64) !void {
            var z: u32 = 0;
            var o: u32 = 0;
            inline for (0..4) |hw| {
                const h: u16 = @truncate(v >> (16 * hw));
                z += @intFromBool(h == 0);
                o += @intFromBool(h == 0xFFFF);
            }
            if (z == 4) return e.w(enc.movw(enc.movz, d, 0, 0));
            if (o == 4) return e.w(enc.movw(enc.movn, d, 0, 0));
            if (4 - @max(z, o) > 1) if (bitmask(v)) |li| return e.w(enc.logImm(enc.orr_i, d, zr, li));
            const inv = o > z;
            const skip: u16 = if (inv) 0xFFFF else 0;
            var first = true;
            inline for (0..4) |hw| {
                const h: u16 = @truncate(v >> (16 * hw));
                if (h != skip) {
                    if (first) {
                        try e.w(enc.movw(if (inv) enc.movn else enc.movz, d, if (inv) ~h else h, hw));
                        first = false;
                    } else try e.w(enc.movw(enc.movk, d, h, hw));
                }
            }
        }
        /// ldr|str `t` at `n + disp`: scaled unsigned offset, unscaled, or x8 register offset.
        inline fn mem(e: *const Self, load: bool, k: enc.Mem, rt: u5, n: u5, disp: i64) !void {
            const sh: u6 = @intCast(@ctz(enc.memSize(k)));
            if (disp >= 0 and disp & ((@as(i64, 1) << sh) - 1) == 0 and disp >> sh <= 4095)
                return e.w(enc.ldstImm(load, k, rt, n, @intCast(disp >> sh)));
            return e.memLong(load, k, rt, n, disp);
        }
        fn memLong(e: *const Self, load: bool, k: enc.Mem, rt: u5, n: u5, disp: i64) !void {
            if (disp >= -256 and disp <= 255) return e.w(enc.ldstUnscaled(load, k, rt, n, @intCast(disp)));
            try e.movImm(s2, @bitCast(disp));
            try e.w(enc.ldstReg(load, k, rt, n, s2));
        }
        /// Spill-slot access (`off` >= 0 and aligned): one scaled ldr/str when in range.
        inline fn slot(e: *const Self, load: bool, comptime k: enc.Mem, rt: u5, off: i64) !void {
            const sh = comptime @ctz(enc.memSize(k));
            if (off <= 4095 << sh) return e.w(enc.ldstImm(load, k, rt, spr, @intCast(off >> sh)));
            return e.memLong(load, k, rt, spr, off);
        }
        inline fn fslotAccess(e: *const Self, load: bool, rt: u5, k: u32) !void {
            return if (e.qspill) e.slot(load, .q, rt, e.fslot(k)) else e.slot(load, .d, rt, e.fslot(k));
        }
        inline fn gslot(e: *const Self, k: u32) i64 {
            return e.out + 8 * @as(i64, k);
        }
        inline fn fslot(e: *const Self, k: u32) i64 {
            return e.fbase + @as(i64, if (e.qspill) 16 else 8) * k;
        }
        /// Register holding GP `v`; a spilled value is loaded into `scratch`. A value without a
        /// location (never written) reads as 0: xzr where register 31 means xzr (`zr_ok`).
        inline fn get(e: *const Self, v: ir.VReg, scratch: u5, comptime zr_ok: bool) !u5 {
            switch (e.locs[v]) {
                .reg => |r| return @intCast(r),
                .stack => |k| {
                    try e.slot(true, .x, scratch, e.gslot(k));
                    return scratch;
                },
                .none => {
                    if (zr_ok) return zr;
                    try e.w(enc.movw(enc.movz, scratch, 0, 0));
                    return scratch;
                },
            }
        }
        /// Register to compute GP `v` into: its own, or s0 when spilled.
        inline fn dst(e: *const Self, v: ir.VReg) u5 {
            return switch (e.locs[v]) {
                .reg => |r| @intCast(r),
                else => s0,
            };
        }
        inline fn put(e: *const Self, v: ir.VReg, r: u5) !void {
            switch (e.locs[v]) {
                .reg => |x| if (x != r) try e.w(enc.mov(@intCast(x), r)),
                .stack => |k| try e.slot(false, .x, r, e.gslot(k)),
                .none => {},
            }
        }
        inline fn fget(e: *const Self, v: ir.VReg, scratch: u5) !u5 {
            switch (e.flocs[v]) {
                .reg => |r| return @intCast(r),
                .stack => |k| {
                    try e.fslotAccess(true, scratch, k);
                    return scratch;
                },
                .none => {
                    try e.w(enc.moviZero(scratch));
                    return scratch;
                },
            }
        }
        inline fn fdst(e: *const Self, v: ir.VReg, scratch: u5) u5 {
            return switch (e.flocs[v]) {
                .reg => |r| @intCast(r),
                else => scratch,
            };
        }
        inline fn fput(e: *const Self, v: ir.VReg, r: u5) !void {
            switch (e.flocs[v]) {
                .reg => |x| if (x != r) try e.w(enc.movV(@intCast(x), r)),
                .stack => |k| try e.fslotAccess(false, r, k),
                .none => {},
            }
        }
        /// Parallel register moves `dst[i] <- src[i]` (distinct destinations; x16 breaks cycles).
        fn parallelMove(e: *const Self, dsts: []u5, srcs: []u5) !void {
            var n = dsts.len;
            var i: usize = 0;
            while (i < n) {
                if (dsts[i] == srcs[i]) {
                    n -= 1;
                    dsts[i] = dsts[n];
                    srcs[i] = srcs[n];
                } else i += 1;
            }
            while (n > 0) {
                const free: ?usize = for (0..n) |k| {
                    const blocked = for (0..n) |j| {
                        if (j != k and srcs[j] == dsts[k]) break true;
                    } else false;
                    if (!blocked) break k;
                } else null;
                if (free) |k| {
                    try e.w(enc.mov(dsts[k], srcs[k]));
                    n -= 1;
                    dsts[k] = dsts[n];
                    srcs[k] = srcs[n];
                } else {
                    // Every destination is still read: move one source out of the cycle.
                    const r = srcs[0];
                    try e.w(enc.mov(s0, r));
                    for (srcs[0..n]) |*s| if (s.* == r) {
                        s.* = s0;
                    };
                }
            }
        }

        // ---- branches ----
        const reach: u32 = (1 << 18) - 16;
        /// The b.cond/cbz range (±1 MiB) may not reach block `target` (`labels.len` = epilog).
        fn far(e: *const Self, target: usize) bool {
            if (!e.maybe_far) return false;
            if (e.a.buf.labelOffset(e.labels[target])) |at| return (e.a.buf.offset() - at) / 4 >= reach;
            return e.pre[target] - e.pre[e.cur] >= reach;
        }
        fn jump(e: *const Self, target: usize) !void {
            if (target != e.cur + 1) try e.a.b(e.labels[target]);
        }
        fn condBr(e: *const Self, br: Br, target: usize) !void {
            const l = e.labels[target];
            if (e.far(target)) {
                switch (br.invert()) {
                    .cond => |c| try e.w(enc.bcondSkip(c)),
                    .cbz => |z| try e.w(enc.cbzSkip(z.nz, z.r)),
                }
                return e.a.b(l);
            }
            switch (br) {
                .cond => |c| switch (@as(a64.Cond, @fromBackingInt(c))) {
                    inline else => |cc| try e.a.bcond(cc, l),
                },
                .cbz => |z| if (z.nz) try e.a.cbnz(xr(z.r), l) else try e.a.cbz(xr(z.r), l),
            }
        }
        /// Two-way branch to `then`/`els` after the condition `br` was set up.
        fn branch2(e: *const Self, br: Br, then: usize, els: usize) !void {
            const next = e.cur + 1;
            if (then == next) return e.condBr(br.invert(), els);
            try e.condBr(br, then);
            try e.jump(els);
        }
    };
}

/// Compiles `f` into `cb` through `a` (an `a64.AssemblerT(...)` instance on `cb`), not finalized.
/// Functions with block parameters are first rewritten by `ir.Func.lowerBlockParams`.
pub fn lower(backing: std.mem.Allocator, f0: *const ir.Func, a: anytype, opts: Options) !Stats {
    var scratch_bytes: [2048]u8 = undefined;
    var temp_alloc = @import("../core/scratch.zig").Scratch.init(&scratch_bytes, backing);
    const gpa = temp_alloc.allocator();
    try f0.verify();
    var lowered: ?ir.Func = if (f0.hasBlockParams()) try f0.lowerBlockParams(gpa) else null;
    defer if (lowered) |*g| g.deinit();
    const f: *const ir.Func = if (lowered) |*g| g else f0;
    if (f.params > ir.max_args) return error.TooManyParams;

    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const A = arena.allocator();
    const types = try rc.checkWith(A, f, .{ .vectors = true });
    const nv: u32 = @max(f.num_vregs, 1);
    const nb = f.blocks.items.len;

    // ---- pass 0: calls, code size bounds, function-wide constants ----
    var has_call = false;
    var out_args: u32 = 0;
    var has_vec = false;
    var has_fp = false;
    for (types) |tp| {
        has_vec = has_vec or tp.isVector();
        has_fp = has_fp or rc.classOf(tp) == .fp;
    }
    var nops: usize = 0;
    // Upper bound of the words before each block (and before the epilog): at most 32 words
    // per op, plus 12 per call argument.
    const pre = try A.alloc(u32, nb + 1);
    var total: u32 = 0;
    for (f.blocks.items, 0..) |blk, bi| {
        pre[bi] = total;
        nops += blk.ops.items.len;
        total +|= 32 *| @as(u32, @intCast(blk.ops.items.len));
        for (blk.ops.items) |*o| if (o.* == .call) {
            const x = o.call;
            total +|= 12 *| @as(u32, @intCast(x.args.len));
            has_call = true;
            out_args = @max(out_args, @as(u32, @intCast(x.args.len -| CC.args.len)));
        };
    }
    pre[nb] = total;
    var an: Analysis = .{
        .cs = .{ .val = try A.alloc(i64, nv), .gen = try A.alloc(u32, nv) },
        .needs = try A.alloc(bool, nv),
        .masks = try A.alloc(u16, nops),
        .types = types,
        .params = f.params,
    };
    @memset(an.cs.gen, 0);
    if (nb != 0) {
        // Function-wide constants: the only definition is a `const` in block 0 before any use
        // there (block 0 runs first, so the value is known at every use). Candidates come
        // from block 0; definitions are then counted until each candidate has a second one.
        const seen = an.needs;
        @memset(seen, false);
        const Seen = struct {
            seen: []bool,
            inline fn use(n: @This(), v: ir.VReg) void {
                n.seen[v] = true;
            }
        };
        var live: usize = 0;
        for (f.blocks.items[0].ops.items) |*o| {
            forUses(o, Seen{ .seen = seen });
            if (o.* == .@"const") {
                const x = o.@"const";
                if (x.dst >= f.params and !seen[x.dst] and an.cs.gen[x.dst] != global) {
                    an.cs.gen[x.dst] = global;
                    an.cs.val[x.dst] = x.imm;
                    live += 1;
                }
            }
        }
        if (live != 0) {
            const ndefs = try A.alloc(u8, nv);
            @memset(ndefs, 0);
            count: for (f.blocks.items) |blk| for (blk.ops.items) |*o| {
                const d = switch (o.*) {
                    inline .@"const", .copy, .arith, .load, .call, .fconst, .typed, .fcmp, .cvt, .splat, .extract => |x| x.dst,
                    .store, .br, .cbr, .bra, .cbra, .ret => continue,
                };
                if (an.cs.gen[d] != global) continue;
                ndefs[d] += 1;
                if (ndefs[d] == 2) {
                    an.cs.gen[d] = 0;
                    live -= 1;
                    if (live == 0) break :count;
                }
            };
        }
    }
    @memset(an.needs, false);
    // ---- register allocation (one run per class) ----
    const gmask = opts.allocatable & allocatable;
    // Call-free functions keep register parameters where they arrive.
    var fixed: []const ?u8 = &.{};
    if (!has_call) {
        const fx = try A.alloc(?u8, @min(f.params, CC.args.len, nv));
        for (fx, 0..) |*x, i| x.* = if (gmask & (@as(ra.RegMask, 1) << @intCast(i)) != 0) @intCast(i) else null;
        fixed = fx;
    }
    const GS = Source(.gp);
    const blocks = try buildBlocks(A, f);
    const gsrc: GS = .{ .blocks = blocks, .num_vregs = nv, .fixed = fixed, .an = &an };
    const gcfg: ra.Config = .{ .allocatable = gmask, .callee_saved = callee_saved & gmask };
    var res = try ra.allocateLinearSource(gpa, gsrc, gcfg);
    defer res.deinit(gpa);
    if (std.debug.runtime_safety) std.debug.assert(ra.verifySource(gsrc, gcfg, res));
    var fres: ?ra.Result = null;
    defer if (fres) |*r| r.deinit(gpa);
    var flocs: []const ra.Loc = undefined;
    if (has_fp) {
        var fmask = opts.fp_allocatable & fp_allocatable;
        if (an.vminmax) fmask &= ~(@as(ra.RegMask, 1) << fs2);
        const FS = Source(.fp);
        const fsrc: FS = .{ .blocks = blocks, .num_vregs = nv, .an = &an };
        const fcfg: ra.Config = .{ .allocatable = fmask };
        fres = try ra.allocateLinearSource(gpa, fsrc, fcfg);
        if (std.debug.runtime_safety) std.debug.assert(ra.verifySource(fsrc, fcfg, fres.?));
        flocs = fres.?.locs;
    } else {
        const l = try A.alloc(ra.Loc, nv);
        @memset(l, .none);
        flocs = l;
    }
    const gslots = res.num_slots;
    const fslots = if (fres) |r| r.num_slots else 0;

    // ---- frame ----
    const used = res.usedRegs();
    var saved_buf: [CC.callee_saved.len]X = undefined;
    var nsaved: usize = 0;
    for (CC.callee_saved) |r| if (used & bit(r) != 0) {
        saved_buf[nsaved] = r;
        nsaved += 1;
    };
    const saved = saved_buf[0..nsaved];
    const fsz: u32 = if (has_vec) 16 else 8;
    const out: u32 = std.mem.alignForward(u32, 8 * out_args, 16);
    const fbase: u32 = std.mem.alignForward(u32, out + 8 * gslots, fsz);
    const frame64: u64 = std.mem.alignForward(u64, @as(u64, fbase) + @as(u64, fsz) * fslots, 16);
    // `sub sp` takes a 24-bit frame in two shifted 12-bit immediates (see the prolog below).
    if (frame64 > 0xFFF000) return error.TooManySpills;
    const frame: u32 = @intCast(frame64);
    const frameless = !has_call and frame == 0 and nsaved == 0;
    const big = frame > 4080;

    const labels = try A.alloc(a64.Label, nb + 1);
    for (labels) |*l| l.* = try a.newLabel();
    const E = Emitter(@TypeOf(a));
    var e: E = .{
        .a = a,
        .locs = res.locs,
        .flocs = flocs,
        .out = out,
        .fbase = fbase,
        .qspill = has_vec,
        .maybe_far = pre[nb] >= E.reach,
        .pre = pre,
        .labels = labels,
    };

    // ---- prolog ----
    // Up to 4080 bytes `prolog` subtracts the frame itself; a larger one is two `sub sp`
    // (imm12 << 12, then imm12) here, and the epilog resets sp from x29 instead.
    var n: u12 = 0;
    if (!frameless) {
        n = try a.prolog(saved, if (big) 0 else frame);
        if (big) {
            if (frame >> 12 != 0) try a.word(0xD14003FF | (frame >> 12) << 10); // sub sp, sp, #hi, lsl 12
            if (frame & 0xFFF != 0) try a.subImm(@fromBackingInt(spr), @fromBackingInt(spr), @intCast(frame & 0xFFF));
        }
    }
    const ep_words: u32 = if (frameless) 1 else @as(u32, @intFromBool(big)) + @intFromBool(n != 0) + @as(u32, @intCast((saved.len + 1) / 2)) + 2;
    const Ep = struct {
        fn emit(as: anytype, fl: bool, bg: bool, sv: []const X, nn: u12) !void {
            if (fl) return as.word(enc.ret);
            // sp back to just below the saved registers (16 bytes per pair or odd last one).
            if (bg) try as.subImm(@fromBackingInt(spr), .x29, @intCast(16 * ((sv.len + 1) / 2)));
            try as.epilog(sv, nn);
        }
    };

    // ---- parameters ----
    {
        const nreg = @min(f.params, CC.args.len);
        var md: [CC.args.len]u5 = undefined;
        var ms: [CC.args.len]u5 = undefined;
        var nm: usize = 0;
        for (0..nreg) |i| switch (res.locs[i]) {
            .stack => |k| try e.mem(false, .x, @intCast(i), spr, e.gslot(k)),
            .reg => |r| {
                md[nm] = @intCast(r);
                ms[nm] = @intCast(i);
                nm += 1;
            },
            .none => {},
        };
        try e.parallelMove(md[0..nm], ms[0..nm]);
        for (@min(f.params, CC.args.len)..f.params) |i| {
            const base: u5 = if (frameless) spr else 29;
            const off: i64 = @as(i64, if (frameless) 0 else 16) + 8 * @as(i64, @intCast(i - CC.args.len));
            switch (res.locs[i]) {
                .reg => |r| try e.mem(true, .x, @intCast(r), base, off),
                .stack => |k| {
                    try e.mem(true, .x, s0, base, off);
                    try e.mem(false, .x, s0, spr, e.gslot(k));
                },
                .none => {},
            }
        }
    }
    // Values read before any def on some path are 0 (IR semantics, as in `ir.interpret`).
    for (res.entry_undef) |v| try e.put(v, zr);
    if (fres) |r| if (r.entry_undef.len != 0) {
        try a.word(enc.moviZero(fs0));
        for (r.entry_undef) |v| switch (flocs[v]) {
            .reg => |x| try a.word(enc.moviZero(@intCast(x))),
            else => try e.fput(v, fs0),
        };
    };

    // ---- body ----
    var epilog_used = false;
    var oi: usize = 0; // function-wide op index (`an.masks`)
    for (f.blocks.items, 0..) |blk, bi| {
        try a.bind(labels[bi]);
        e.cur = bi;
        an.cs.begin();
        const cs = &an.cs;
        for (blk.ops.items) |*op| {
            defer oi += 1;
            switch (op.*) {
                .@"const" => |x| {
                    cs.set(x.dst, x.imm);
                    if (an.needs[x.dst]) {
                        const d = e.dst(x.dst);
                        try e.movImm(d, @bitCast(x.imm));
                        try e.put(x.dst, d);
                    }
                },
                .copy => |x| {
                    cs.set(x.dst, cs.get(x.src));
                    if (an.needs[x.dst]) if (rc.classOf(types[x.dst]) == .fp) {
                        try e.fput(x.dst, try e.fget(x.src, fs0));
                    } else if (cs.cheap(x.src)) |k| {
                        const d = e.dst(x.dst);
                        try e.movImm(d, @bitCast(k));
                        try e.put(x.dst, d);
                    } else try e.put(x.dst, try e.get(x.src, s0, true));
                },
                .arith => |x| {
                    const m = an.masks[oi];
                    if (an.needs[x.dst]) {
                        const d = e.dst(x.dst);
                        switch (planOf(m, cs, x)) {
                            .konst => |k| try e.movImm(d, @bitCast(k)),
                            .imm => |p| {
                                const r = try e.get(if (p.swap) x.b else x.a, s0, false);
                                switch (x.op) {
                                    .add, .sub => {
                                        const ai = addImm(if (x.op == .sub) -%p.k else p.k).?;
                                        try e.w(enc.addImm(ai.sub, d, r, ai.imm, ai.sh));
                                    },
                                    .mul => try e.w(enc.shiftImm(.shl, d, r, @intCast(@ctz(@as(u64, @bitCast(p.k)))))),
                                    .@"and", .@"or", .xor => {
                                        const li = bitmask(@bitCast(p.k)).?;
                                        try e.w(enc.logImm(switch (x.op) {
                                            .@"and" => enc.and_i,
                                            .@"or" => enc.orr_i,
                                            else => enc.eor_i,
                                        }, d, r, li));
                                    },
                                    .shl, .shr, .sar => try e.w(enc.shiftImm(x.op, d, r, @intCast(p.k & 63))),
                                    .sdiv, .srem => unreachable,
                                }
                            },
                            .reg => |p| {
                                const ra_ = if (p.za) zr else try e.get(x.a, s0, true);
                                const rb = if (p.zb) zr else try e.get(x.b, s1, true);
                                switch (x.op) {
                                    .srem => { // a - (a / b) * b; the quotient in x8 since ra_ may be d
                                        try e.w(enc.rrr(enc.sdiv, s2, ra_, rb));
                                        try e.w(enc.madd(true, d, s2, rb, ra_));
                                    },
                                    else => try e.w(enc.rrr(switch (x.op) {
                                        .add => enc.add_r,
                                        .sub => enc.sub_r,
                                        .mul => enc.mul,
                                        .@"and" => enc.and_r,
                                        .@"or" => enc.orr_r,
                                        .xor => enc.eor_r,
                                        .shl => enc.lslv,
                                        .shr => enc.lsrv,
                                        .sar => enc.asrv,
                                        .sdiv => enc.sdiv,
                                        .srem => unreachable,
                                    }, d, ra_, rb)),
                                }
                            },
                        }
                        try e.put(x.dst, d);
                    }
                    cs.set(x.dst, if (m & plan_konst != 0) ir.arith(x.op, cs.get(x.a).?, cs.get(x.b).?) else null);
                },
                .load => |x| {
                    cs.set(x.dst, null);
                    const base = try e.get(x.base, s1, false);
                    if (x.ty == .i64) {
                        const d = e.dst(x.dst);
                        try e.mem(true, .x, d, base, x.disp);
                        try e.put(x.dst, d);
                    } else {
                        const d = e.fdst(x.dst, fs0);
                        try e.mem(true, enc.memOf(x.ty), d, base, x.disp);
                        try e.fput(x.dst, d);
                    }
                },
                .store => |x| {
                    if (x.ty == .i64) {
                        const src = if (cs.zero(x.src)) zr else try e.get(x.src, s0, true);
                        try e.mem(false, .x, src, try e.get(x.base, s1, false), x.disp);
                    } else {
                        const src = try e.fget(x.src, fs0);
                        try e.mem(false, enc.memOf(x.ty), src, try e.get(x.base, s1, false), x.disp);
                    }
                },
                .call => |x| {
                    // Stack arguments first (x0..x7 still hold the register sources).
                    if (x.args.len > CC.args.len) for (x.args[CC.args.len..], 0..) |v, j| {
                        const r: u5 = if (cs.cheap(v)) |k| blk: {
                            if (k == 0) break :blk zr;
                            try e.movImm(s0, @bitCast(k));
                            break :blk s0;
                        } else try e.get(v, s0, true);
                        try e.mem(false, .x, r, spr, 8 * @as(i64, @intCast(j)));
                    };
                    const nreg = @min(x.args.len, CC.args.len);
                    var md: [CC.args.len]u5 = undefined;
                    var ms: [CC.args.len]u5 = undefined;
                    var nm: usize = 0;
                    for (x.args[0..nreg], 0..) |v, i| if (cs.cheap(v) == null) switch (res.locs[v]) {
                        .reg => |r| {
                            md[nm] = @intCast(i);
                            ms[nm] = @intCast(r);
                            nm += 1;
                        },
                        else => {},
                    };
                    try e.parallelMove(md[0..nm], ms[0..nm]);
                    for (x.args[0..nreg], 0..) |v, i| {
                        const ri: u5 = @intCast(i);
                        if (cs.cheap(v)) |k| {
                            try e.movImm(ri, @bitCast(k));
                        } else switch (res.locs[v]) {
                            .reg => {},
                            .stack => |k| try e.mem(true, .x, ri, spr, e.gslot(k)),
                            .none => try e.w(enc.movw(enc.movz, ri, 0, 0)),
                        }
                    }
                    try e.movImm(s0, x.addr);
                    try e.w(enc.blr(s0));
                    if (an.needs[x.dst]) try e.put(x.dst, 0);
                    cs.set(x.dst, null);
                },
                .br => |tg| try e.jump(tg),
                .cbr => |x| {
                    const p = cplan(cs, x);
                    if (p == .static) {
                        try e.jump(if (p.static) x.then else x.@"else");
                    } else if (x.then == x.@"else") {
                        try e.jump(x.then);
                    } else {
                        const br: Br = switch (p) {
                            .static => unreachable,
                            .reg => blk: {
                                try e.w(enc.cmpReg(try e.get(x.a, s0, true), try e.get(x.b, s1, true)));
                                break :blk .{ .cond = condOf(x.cond) };
                            },
                            .imm => |q| blk: {
                                const r = try e.get(q.v, s0, false);
                                try e.w(enc.cmpImm(q.ai.sub, r, q.ai.imm, q.ai.sh));
                                break :blk .{ .cond = q.cond };
                            },
                            .zero => |q| .{ .cbz = .{ .nz = q.nz, .r = try e.get(q.v, s0, false) } },
                        };
                        try e.branch2(br, x.then, x.@"else");
                    }
                },
                .ret => |v| {
                    if (cs.get(v)) |k| {
                        try e.movImm(0, @bitCast(k));
                    } else {
                        const r = try e.get(v, 0, true);
                        if (r != 0) try e.w(enc.mov(0, r));
                    }
                    if (bi + 1 == nb) {
                        epilog_used = true; // falls through
                    } else if (ep_words <= 3) {
                        try Ep.emit(a, frameless, big, saved, n);
                    } else {
                        epilog_used = true;
                        try a.b(labels[nb]);
                    }
                },
                .fconst => |x| {
                    cs.set(x.dst, null);
                    if (an.needs[x.dst]) {
                        const d = e.fdst(x.dst, fs0);
                        const bits: u64 = if (x.ty == .f32) @as(u32, @truncate(x.bits)) else x.bits;
                        const val: f64 = if (x.ty == .f32) @as(f32, @bitCast(@as(u32, @truncate(bits)))) else @bitCast(bits);
                        if (bits == 0) {
                            try e.w(enc.moviZero(d));
                        } else if (imm.fpImm8(val)) |i8_| {
                            try e.w(enc.fmovImm(x.ty, d, i8_));
                        } else {
                            try e.movImm(s0, bits);
                            try e.w(enc.fmovFromGp(x.ty, d, s0));
                        }
                        try e.fput(x.dst, d);
                    }
                },
                .typed => |x| {
                    cs.set(x.dst, null);
                    if (an.needs[x.dst]) {
                        const va = try e.fget(x.a, fs0);
                        const vb = try e.fget(x.b, fs1);
                        var d = e.fdst(x.dst, fs0);
                        if (!x.ty.isVector()) switch (x.op) {
                            .add, .sub, .mul, .div => try e.w(enc.frrr(switch (x.op) {
                                .add => enc.fadd,
                                .sub => enc.fsub,
                                .mul => enc.fmul,
                                else => enc.fdiv,
                            }, x.ty, d, va, vb)),
                            // IR (x86) semantics: `a < b ? a : b` / `a > b ? a : b`, the second operand
                            // when unordered or equal (A64 fmin/fmax differ on NaN and signed zeros).
                            .min, .max => {
                                try e.w(enc.fcmp(x.ty, va, vb));
                                try e.w(enc.fcsel(x.ty, d, va, vb, if (x.op == .min) enc.mi else enc.gt));
                            },
                            .@"and", .@"or", .xor => unreachable, // integer vectors only
                        } else switch (x.op) {
                            .@"and", .@"or", .xor => try e.w(enc.rrr(switch (x.op) {
                                .@"and" => enc.and_16b,
                                .@"or" => enc.orr_16b,
                                else => enc.eor_16b,
                            }, d, va, vb)),
                            .add, .sub, .mul, .div => if (x.ty == .v2i64 and x.op == .mul) {
                                // No 64-bit lane multiply in NEON: per lane through GP registers.
                                try e.w(enc.umovD(s0, va, 0));
                                try e.w(enc.umovD(s1, vb, 0));
                                try e.w(enc.rrr(enc.mul, s0, s0, s1));
                                try e.w(enc.umovD(s1, va, 1));
                                try e.w(enc.umovD(s2, vb, 1));
                                try e.w(enc.rrr(enc.mul, s1, s1, s2));
                                try e.w(enc.fmovFromGp(.f64, d, s0));
                                try e.w(enc.insD(d, 1, s1));
                            } else {
                                const base: u32 = switch (x.ty) {
                                    .v4i32 => switch (x.op) {
                                        .add => enc.add_4s,
                                        .sub => enc.sub_4s,
                                        .mul => enc.mul_4s,
                                        else => unreachable,
                                    },
                                    .v2i64 => switch (x.op) {
                                        .add => enc.add_2d,
                                        .sub => enc.sub_2d,
                                        else => unreachable,
                                    },
                                    else => @as(u32, switch (x.op) {
                                        .add => enc.fadd_4s,
                                        .sub => enc.fsub_4s,
                                        .mul => enc.fmul_4s,
                                        else => enc.fdiv_4s,
                                    }) | (if (x.ty == .v2f64) enc.sz_2d else 0),
                                };
                                try e.w(enc.rrr(base, d, va, vb));
                            },
                            // Per lane `x < y ? x : y` / `x > y ? x : y`: mask = fcmgt(y, x) (min)
                            // or fcmgt(x, y) (max), result = mask ? x : y (NaN or equal -> y).
                            .min, .max => {
                                const cmp = enc.fcmgt_4s | (if (x.ty == .v2f64) enc.sz_2d else 0);
                                const gt_n, const gt_m = if (x.op == .min) .{ vb, va } else .{ va, vb };
                                if (e.flocs[x.dst] != .reg) {
                                    // Spilled result: a scratch distinct from both operands.
                                    d = for ([_]u5{ fs0, fs1, fs2 }) |s| {
                                        if (s != va and s != vb) break s;
                                    } else unreachable;
                                }
                                if (va == vb) {
                                    if (d != va) try e.w(enc.movV(d, va));
                                } else if (d == va) {
                                    const tmp: u5 = if (vb != fs0) fs0 else fs1;
                                    try e.w(enc.rrr(cmp, tmp, gt_n, gt_m));
                                    try e.w(enc.rrr(enc.bif_16b, d, vb, tmp));
                                } else if (d == vb) {
                                    const tmp: u5 = if (va != fs0) fs0 else fs1;
                                    try e.w(enc.rrr(cmp, tmp, gt_n, gt_m));
                                    try e.w(enc.rrr(enc.bit_16b, d, va, tmp));
                                } else {
                                    try e.w(enc.rrr(cmp, d, gt_n, gt_m));
                                    try e.w(enc.rrr(enc.bsl_16b, d, va, vb));
                                }
                            },
                        }
                        try e.fput(x.dst, d);
                    }
                },
                .fcmp => |x| {
                    cs.set(x.dst, null);
                    if (an.needs[x.dst]) {
                        try e.w(enc.fcmp(x.ty, try e.fget(x.a, fs0), try e.fget(x.b, fs1)));
                        const d = e.dst(x.dst);
                        // Unordered sets C and V: mi/ls/gt/ge are false, ne is true (IR semantics).
                        try e.w(enc.cset(d, switch (x.cond) {
                            .eq => enc.eq,
                            .ne => enc.ne,
                            .lt => enc.mi,
                            .le => enc.ls,
                            .gt => enc.gt,
                            .ge => enc.ge,
                        }));
                        try e.put(x.dst, d);
                    }
                },
                .cvt => |x| {
                    cs.set(x.dst, null);
                    if (an.needs[x.dst]) switch (x.to) {
                        .i64 => {
                            // fcvtzs saturates (NaN -> 0); the IR follows x86 cvttsd2si: NaN and out of
                            // range give minInt(i64). Positive overflow saturates to maxInt, which no
                            // in-range float converts to: `ccmn d, #1` overflows (V) exactly for maxInt,
                            // and an unordered `fcmp` sets V, so `vs` selects minInt.
                            const v = try e.fget(x.src, fs0);
                            const d = e.dst(x.dst);
                            try e.w(enc.fcvtzs(x.from, d, v));
                            try e.w(enc.fcmp(x.from, v, v));
                            try e.w(enc.ccmnImm(d, 1, 1, enc.vc));
                            try e.w(enc.movw(enc.movz, s1, 0x8000, 3));
                            try e.w(enc.csel(d, s1, d, enc.vs));
                            try e.put(x.dst, d);
                        },
                        else => {
                            const d = e.fdst(x.dst, fs0);
                            if (x.from == .i64) {
                                try e.w(enc.scvtf(x.to, d, if (cs.zero(x.src)) zr else try e.get(x.src, s0, true)));
                            } else {
                                try e.w(enc.fcvt(x.to, d, try e.fget(x.src, fs1)));
                            }
                            try e.fput(x.dst, d);
                        },
                    };
                },
                .splat => |x| {
                    cs.set(x.dst, null);
                    if (an.needs[x.dst]) {
                        const d = e.fdst(x.dst, fs0);
                        const d64 = x.ty == .v2i64 or x.ty == .v2f64;
                        if (x.ty.lane() == .i64) {
                            try e.w(enc.dupGp(d64, d, if (cs.zero(x.src)) zr else try e.get(x.src, s0, true)));
                        } else {
                            try e.w(enc.dupElem0(d64, d, try e.fget(x.src, fs1)));
                        }
                        try e.fput(x.dst, d);
                    }
                },
                .extract => |x| {
                    cs.set(x.dst, null);
                    if (an.needs[x.dst]) {
                        const v = try e.fget(x.src, fs0);
                        switch (x.ty) {
                            .v4i32, .v2i64 => {
                                const d = e.dst(x.dst);
                                try e.w(if (x.ty == .v4i32) enc.smovS(d, v, x.lane) else enc.umovD(d, v, x.lane));
                                try e.put(x.dst, d);
                            },
                            else => {
                                const d = e.fdst(x.dst, fs1);
                                try e.w(enc.dupScalar(x.ty == .v2f64, d, v, x.lane));
                                try e.fput(x.dst, d);
                            },
                        }
                    }
                },
                .bra, .cbra => unreachable, // removed by lowerBlockParams
            }
        }
    }
    try a.bind(labels[nb]);
    if (epilog_used) try Ep.emit(a, frameless, big, saved, n);
    return .{ .spills = res.spills + if (fres) |r| r.spills else 0, .slots = gslots + fslots };
}

/// IR -> bytes in `cb` (finalized), with validation on.
pub fn compile(gpa: std.mem.Allocator, f: *const ir.Func, cb: *CodeBuffer, opts: Options) !Stats {
    const st = try lower(gpa, f, a64.AssemblerT(.{ .validate = true }).init(cb), opts);
    try cb.finalize();
    return st;
}

// ---- tests ------------------------------------------------------------------------------
const t = std.testing;
const builtin = @import("builtin");
const JitRuntime = @import("../core/jit_runtime.zig");
const can_run = builtin.cpu.arch == .aarch64;

fn run(f: *const ir.Func, opts: Options, args: []const i64) !i64 {
    var cb = CodeBuffer.init(t.allocator);
    defer cb.deinit();
    _ = try compile(t.allocator, f, &cb, opts);
    const fun = try JitRuntime.add(&cb);
    defer fun.release();
    var v: [8]i64 = @splat(0);
    for (args, 0..) |x, i| v[i] = x;
    const F = *const fn (i64, i64, i64, i64, i64, i64, i64, i64) callconv(.c) i64;
    return fun.ptr(F)(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]);
}

test "a64 compiler: bitmask matches a64.logicalImm" {
    const Li = @TypeOf(imm.logicalImm(0, true));
    const same = struct {
        fn f(v: u64) !void {
            const want: Li = imm.logicalImm(v, true);
            const got = bitmask(v);
            try t.expectEqual(want == null, got == null);
            if (got) |g| try t.expect(g.n == want.?.n and g.immr == want.?.immr and g.imms == want.?.imms);
        }
    }.f;
    // Every bitmask immediate (element size, run length, rotation), and its neighbours.
    var size: u8 = 2;
    while (size <= 64) : (size *= 2) for (1..size) |ones| for (0..size) |rot| {
        var e: u64 = (@as(u64, 1) << @intCast(ones)) - 1;
        if (rot != 0) e = ((e >> @intCast(rot)) | (e << @intCast(size - rot))) & if (size == 64) ~@as(u64, 0) else (@as(u64, 1) << @intCast(size)) - 1;
        var v = e;
        var w: u8 = size;
        while (w < 64) : (w *= 2) v |= v << @intCast(w);
        try same(v);
        try same(v +% 1);
        try same(v -% 1);
        try same(v ^ 1 << 63);
    };
    var prng = std.Random.DefaultPrng.init(0xb17);
    for (0..100_000) |_| try same(prng.random().int(u64));
    for ([_]u64{ 0, ~@as(u64, 0), 1, 1 << 63, 0x100006a3c0000 }) |v| try same(v);
}

test "a64 compiler: fib compiles (byte-level, any host)" {
    var f = try ir.parse(t.allocator, ir.fib_src);
    defer f.deinit();
    var cb = CodeBuffer.init(t.allocator);
    defer cb.deinit();
    const st = try compile(t.allocator, &f, &cb, .{});
    try t.expectEqual(@as(u32, 0), st.spills);
    const c = cb.code();
    try t.expectEqual(@as(usize, 0), c.len % 4);
    try t.expect(std.mem.readInt(u32, c[0..4], .little) != 0xA9BF7BFD); // a leaf without a frame
    try t.expectEqual(@as(u32, 0xD65F03C0), std.mem.readInt(u32, c[c.len - 4 ..][0..4], .little)); // ret
    var g = try ir.parse(t.allocator, "func f 17\na:\n  ret v0\n");
    defer g.deinit();
    try t.expectError(error.TooManyParams, compile(t.allocator, &g, &cb, .{}));
}

test "a64 compiler: execution - fib, calls, memory, spills" {
    if (!can_run) return error.SkipZigTest;
    var f = try ir.parse(t.allocator, ir.fib_src);
    defer f.deinit();
    for ([_]i64{ 0, 1, 2, 10, 50, 90 }) |nn| {
        const want = try ir.interpret(&f, &.{nn}, 1 << 20);
        try t.expectEqual(want, try run(&f, .{}, &.{nn}));
        try t.expectEqual(want, try run(&f, .{ .allocatable = bit(.x19) }, &.{nn})); // spills
    }
    const Helper = struct {
        fn mul3(x: i64, y: i64, z: i64) callconv(.c) i64 {
            return x * y * z;
        }
    };
    var mem: [2]i64 = .{ 0, 0 };
    var g = ir.Func.init(t.allocator);
    defer g.deinit();
    g.params = 3;
    const e = try g.addBlock("e");
    const args = try g.alloc().dupe(ir.VReg, &.{ 0, 1, 1 });
    try g.emit(e, .{ .call = .{ .dst = 3, .addr = @intFromPtr(&Helper.mul3), .args = args } });
    try g.emit(e, .{ .arith = .{ .op = .add, .dst = 4, .a = 3, .b = 0 } }); // v0 live across the call
    try g.emit(e, .{ .store = .{ .base = 2, .disp = 8, .src = 4 } });
    try g.emit(e, .{ .load = .{ .dst = 5, .base = 2, .disp = 8 } });
    try g.emit(e, .{ .ret = 5 });
    const p: i64 = @bitCast(@intFromPtr(&mem));
    for ([_]ra.RegMask{ allocatable, bit(.x19) | bit(.x9) }) |m| {
        mem = .{ 0, 0 };
        try t.expectEqual(@as(i64, 3 * 4 * 4 + 3), try run(&g, .{ .allocatable = m }, &.{ 3, 4, p }));
        try t.expectEqual(@as(i64, 51), mem[1]);
    }
}

test "a64 compiler: shifts and signed division match the interpreter" {
    if (!can_run) return error.SkipZigTest;
    const src =
        \\func f 2
        \\e:
        \\  v2 = shl v0 v1
        \\  v3 = shr v0 v1
        \\  v4 = sar v0 v1
        \\  v9 = const 7
        \\  v5 = sdiv v0 v9
        \\  v6 = srem v0 v9
        \\  v7 = sdiv v2 v0
        \\  v8 = add v2 v3
        \\  v8 = xor v8 v4
        \\  v8 = add v8 v5
        \\  v8 = sub v8 v6
        \\  v8 = add v8 v7
        \\  ret v8
    ;
    var f = try ir.parse(t.allocator, src);
    defer f.deinit();
    const cases = [_][2]i64{ .{ 1, 0 }, .{ -1, 62 }, .{ 12345, 3 }, .{ -98765, 65 }, .{ std.math.maxInt(i64), 1 }, .{ -7, 2 } };
    for (cases) |c| {
        const want = try ir.interpret(&f, &c, 1000);
        try t.expectEqual(want, try run(&f, .{}, &c));
        try t.expectEqual(want, try run(&f, .{ .allocatable = bit(.x9) | bit(.x19) }, &c));
        try t.expectEqual(want, try run(&f, .{ .allocatable = bit(.x19) }, &c)); // operands spilled
    }
}

test "a64 compiler: block parameters (SSA input)" {
    var f = try ir.parse(t.allocator, ir.fib_ssa_src);
    defer f.deinit();
    var cb = CodeBuffer.init(t.allocator);
    defer cb.deinit();
    _ = try compile(t.allocator, &f, &cb, .{});
    if (!can_run) return;
    for ([_]i64{ 0, 1, 10, 90 }) |n| {
        const want = try ir.interpret(&f, &.{n}, 1 << 20);
        try t.expectEqual(want, try run(&f, .{}, &.{n}));
        try t.expectEqual(want, try run(&f, .{ .allocatable = bit(.x19) }, &.{n}));
    }
}

test "a64 compiler: a block parameter live into its use block keeps its register" {
    // `lowerBlockParams` puts the edge copies for `a`'s parameter after `a`, so v2 is live
    // into `a` without being live out of it; the constants at the top of `a` must not take
    // its register.
    var f = try ir.parse(t.allocator,
        \\func f 1
        \\e:
        \\  v1 = const 3
        \\  cbr lt v0 v1 a:v0 b:v1
        \\a(v2):
        \\  v3 = const 100
        \\  v4 = const 1
        \\  v5 = add v3 v4
        \\  v6 = add v2 v5
        \\  ret v6
        \\b(v7):
        \\  ret v7
    );
    defer f.deinit();
    var cb = CodeBuffer.init(t.allocator);
    defer cb.deinit();
    _ = try compile(t.allocator, &f, &cb, .{});
    if (!can_run) return;
    for ([_]i64{ 0, 1, 2, 5 }) |n| {
        const want = try ir.interpret(&f, &.{n}, 1000);
        try t.expectEqual(want, try run(&f, .{}, &.{n}));
    }
}

test "a64 compiler: a branch back to the entry block keeps the parameters" {
    // v0 is used on every pass through `e`, including after the back edge e -> e.
    var f = try ir.parse(t.allocator,
        \\func f 2
        \\e:
        \\  v3 = load v0 0
        \\  v4 = add v3 v3
        \\  v5 = const 1
        \\  v1 = sub v1 v5
        \\  cbr gt v1 v5 e x
        \\x:
        \\  ret v4
    );
    defer f.deinit();
    var cb = CodeBuffer.init(t.allocator);
    defer cb.deinit();
    _ = try compile(t.allocator, &f, &cb, .{});
    if (!can_run) return;
    var cell: i64 = 21;
    const p: i64 = @intCast(@intFromPtr(&cell));
    for ([_]i64{ 1, 3, 10 }) |n| {
        const want = try ir.interpret(&f, &.{ p, n }, 1000);
        try t.expectEqual(want, try run(&f, .{}, &.{ p, n }));
        try t.expectEqual(want, try run(&f, .{ .allocatable = bit(.x19) | bit(.x9) }, &.{ p, n }));
    }
}

test "a64 compiler: a value undefined on some path reads as 0" {
    // e -> j skips the def of v2 (and of the float v4): the interpreter reads 0 there.
    // v1 is unused, so its argument register x1 still holds the caller's value.
    var f = try ir.parse(t.allocator,
        \\func f 2
        \\e:
        \\  v9 = const 7
        \\  cbr gt v0 v9 a j
        \\a:
        \\  v2 = const 5
        \\  v4 = fconst.f64 2.5
        \\  br j
        \\j:
        \\  v3 = add v2 v0
        \\  v5 = fconst.f64 0
        \\  v6 = fcmp.f64 eq v4 v5
        \\  v7 = add v3 v6
        \\  ret v7
    );
    defer f.deinit();
    if (!can_run) return;
    for ([_]i64{ 1, 10 }) |n| {
        const want = try ir.interpret(&f, &.{ n, 1000 }, 1000);
        for ([_]ra.RegMask{ allocatable, bit(.x1) | bit(.x9), bit(.x1) }) |m| {
            try t.expectEqual(want, try run(&f, .{ .allocatable = m }, &.{ n, 1000 }));
        }
    }
}

/// IR with `n` values live at once (then 8 floats): with few registers the spill area passes
/// the `sub sp` (4080) and scaled ldr/str (32760) immediate limits; FP slots come after.
fn bigFrameSource(gpa: std.mem.Allocator, n: u32) ![]u8 {
    var src: std.ArrayList(u8) = .empty;
    errdefer src.deinit(gpa);
    try src.appendSlice(gpa, "func f 1\ne:\n");
    for (1..n + 1) |i| try src.print(gpa, "  v{d} = add v{d} v0\n", .{ i, i - 1 });
    var nv: usize = n + 1;
    for (0..8) |j| try src.print(gpa, "  v{d} = cvt.f64.i64 v{d}\n", .{ nv + j, j + 1 });
    const fv = nv;
    nv += 8;
    var acc: usize = n;
    var i: usize = n - 1;
    while (i >= 1) : (i -= 1) {
        try src.print(gpa, "  v{d} = add v{d} v{d}\n", .{ nv, acc, i });
        acc = nv;
        nv += 1;
    }
    for (0..8) |j| {
        try src.print(gpa, "  v{d} = cvt.i64.f64 v{d}\n  v{d} = add v{d} v{d}\n", .{ nv, fv + j, nv + 1, acc, nv });
        acc = nv + 1;
        nv += 2;
    }
    try src.print(gpa, "  ret v{d}\n", .{acc});
    return src.toOwnedSlice(gpa);
}

test "a64 compiler: frames past the immediate offset ranges" {
    for ([_]u32{ 600, 4200 }) |n| {
        const src = try bigFrameSource(t.allocator, n);
        defer t.allocator.free(src);
        var f = try ir.parse(t.allocator, src);
        defer f.deinit();
        var cb = CodeBuffer.init(t.allocator);
        defer cb.deinit();
        const st = try compile(t.allocator, &f, &cb, .{ .allocatable = bit(.x9) });
        try t.expect(st.slots * 8 > if (n > 1000) @as(u32, 32760) else 4080);
        if (!can_run) continue;
        for ([_]i64{ 1, 3 }) |x| {
            const want = try ir.interpret(&f, &.{x}, 1_000_000);
            try t.expectEqual(want, try run(&f, .{ .allocatable = bit(.x9) }, &.{x}));
            try t.expectEqual(want, try run(&f, .{}, &.{x}));
        }
    }
}

test "a64 compiler: a spilled base with an out-of-range offset" {
    // disp -8 needs `base + disp` in a register; with spills the base is loaded into the
    // same scratch register the offset used to be built in.
    var f = try ir.parse(t.allocator,
        \\func g 1
        \\e:
        \\  v1 = const 8
        \\  v2 = add v0 v1
        \\  v5 = const 100
        \\  v3 = load v2 -8
        \\  store v2 -8 v5
        \\  v8 = load v0 0
        \\  v6 = add v3 v8
        \\  v7 = sub v6 v2
        \\  ret v7
    );
    defer f.deinit();
    var cb = CodeBuffer.init(t.allocator);
    defer cb.deinit();
    _ = try compile(t.allocator, &f, &cb, .{});
    if (!can_run) return;
    var cell: i64 = 7;
    const p: i64 = @intCast(@intFromPtr(&cell));
    const want: i64 = 7 + 100 - (p + 8);
    for ([_]ra.RegMask{ allocatable, bit(.x19) }) |m| {
        cell = 7;
        try t.expectEqual(want, try run(&f, .{ .allocatable = m }, &.{p}));
        try t.expectEqual(@as(i64, 100), cell);
    }
}

test "a64 compiler: stack-passed parameters and call arguments" {
    if (!can_run) return error.SkipZigTest;
    const Helper = struct {
        fn sum11(a: i64, b: i64, c: i64, d: i64, e: i64, f_: i64, g: i64, h: i64, i: i64, j: i64, k: i64) callconv(.c) i64 {
            return a + 2 * b + 3 * c + 4 * d + 5 * e + 6 * f_ + 7 * g + 8 * h + 9 * i + 10 * j + 11 * k;
        }
        fn alt10(a: i64, b: i64, c: i64, d: i64, e: i64, f_: i64, g: i64, h: i64, i: i64, j: i64) callconv(.c) i64 {
            return a - b + c - d + e - f_ + g - h + i - j;
        }
    };
    // 10 parameters (2 on the stack); calls with 11 and 10 arguments.
    var g = ir.Func.init(t.allocator);
    defer g.deinit();
    g.params = 10;
    const e = try g.addBlock("e");
    const a11 = try g.alloc().dupe(ir.VReg, &.{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9 });
    try g.emit(e, .{ .call = .{ .dst = 10, .addr = @intFromPtr(&Helper.sum11), .args = a11 } });
    const a10 = try g.alloc().dupe(ir.VReg, &.{ 10, 9, 8, 7, 6, 5, 4, 3, 2, 1 });
    try g.emit(e, .{ .call = .{ .dst = 11, .addr = @intFromPtr(&Helper.alt10), .args = a10 } });
    try g.emit(e, .{ .arith = .{ .op = .add, .dst = 12, .a = 11, .b = 0 } });
    try g.emit(e, .{ .ret = 12 });
    const p = [_]i64{ 1, 20, 300, 4000, 50000, 600000, 7000000, 80000000, 900000000, 10000000000 };
    const s11 = Helper.sum11(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9], p[9]);
    const want = Helper.alt10(s11, p[9], p[8], p[7], p[6], p[5], p[4], p[3], p[2], p[1]) + p[0];
    for ([_]ra.RegMask{ allocatable, bit(.x19) }) |mask| {
        var cb = CodeBuffer.init(t.allocator);
        defer cb.deinit();
        _ = try compile(t.allocator, &g, &cb, .{ .allocatable = mask });
        const fun = try JitRuntime.add(&cb);
        defer fun.release();
        const F = *const fn (i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) callconv(.c) i64;
        try t.expectEqual(want, fun.ptr(F)(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9]));
    }
}

test "a64 compiler: differential vs interpreter on random programs" {
    if (!can_run) return error.SkipZigTest;
    var prng = std.Random.DefaultPrng.init(11);
    const r = prng.random();
    for (0..300) |_| {
        var f = ir.Func.init(t.allocator);
        defer f.deinit();
        f.params = 3;
        const nv: u32 = 3 + r.uintLessThan(u32, 24);
        f.num_vregs = nv;
        const nb = 1 + r.uintLessThan(u32, 5);
        for (0..nb) |_| _ = try f.addBlock("b");
        for (3..nv) |v| try f.emit(0, .{ .@"const" = .{ .dst = @intCast(v), .imm = 0 } });
        for (0..nb) |bi| {
            const blk: u32 = @intCast(bi);
            for (0..r.uintLessThan(u32, 10)) |_| {
                const d = r.uintLessThan(u32, nv);
                if (r.boolean()) {
                    try f.emit(blk, .{ .@"const" = .{ .dst = d, .imm = r.int(i64) } });
                } else try f.emit(blk, .{ .arith = .{ .op = @fromBackingInt(@intCast(r.uintLessThan(u8, @backingInt(ir.Arith.sdiv)))), .dst = d, .a = r.uintLessThan(u32, nv), .b = r.uintLessThan(u32, nv) } });
            }
            if (bi + 1 == nb) {
                try f.emit(blk, .{ .ret = r.uintLessThan(u32, nv) });
            } else {
                const t1 = blk + 1 + r.uintLessThan(u32, nb - blk - 1);
                const t2 = blk + 1 + r.uintLessThan(u32, nb - blk - 1);
                try f.emit(blk, .{ .cbr = .{ .cond = r.enumValue(ir.CondCode), .a = r.uintLessThan(u32, nv), .b = r.uintLessThan(u32, nv), .then = t1, .@"else" = t2 } });
            }
        }
        const args = [_]i64{ r.int(i64), r.int(i64), r.int(i64) };
        const want = try ir.interpret(&f, &args, 1 << 16);
        try t.expectEqual(want, try run(&f, .{}, &args));
        try t.expectEqual(want, try run(&f, .{ .allocatable = bit(.x9) | bit(.x19) }, &args));
    }
}

test "compiler: FP encodings match LLVM (any host)" {
    const e = enc;
    const got = [_]u32{
        e.frrr(e.fadd, .f32, 30, 1, 2),    e.frrr(e.fsub, .f64, 30, 3, 31),
        e.frrr(e.fmul, .f32, 0, 7, 16),    e.frrr(e.fdiv, .f64, 29, 30, 31),
        e.fcmp(.f32, 30, 31),              e.fcmp(.f64, 1, 2),
        e.fcsel(.f64, 30, 1, 2, e.mi),     e.fcsel(.f32, 30, 1, 2, e.gt),
        e.fmovRR(5, 30),                   e.fmovFromGp(.f32, 30, 16),
        e.fmovFromGp(.f64, 30, 16),        e.fmovFromGp(.f64, 30, 31),
        e.ldstImm(true, .s, 30, 17, 1023), e.ldstImm(false, .d, 30, 31, 4095),
        e.ldstImm(true, .d, 0, 1, 1),      e.ldstImm(false, .s, 3, 2, 0),
        e.scvtf(.f32, 30, 16),             e.scvtf(.f64, 30, 9),
        e.fcvtzs(.f32, 16, 30),            e.fcvtzs(.f64, 16, 1),
        e.fcvt(.f64, 30, 31),              e.fcvt(.f32, 30, 31),
        e.cset(16, e.mi),                  e.cset(16, e.ls),
        e.csel(16, 17, 16, e.eq),          e.csel(16, 17, 16, e.vs),
    };
    // clang -target arm64-apple-macos: fadd s30, s1, s2 ... csel x16, x17, x16, vs
    const want = [_]u32{
        0x1e22283e, 0x1e7f387e, 0x1e3008e0, 0x1e7f1bdd, 0x1e3f23c0, 0x1e622020, 0x1e624c3e,
        0x1e22cc3e, 0x1e6043c5, 0x1e27021e, 0x9e67021e, 0x9e6703fe, 0xbd4ffe3e, 0xfd3ffffe,
        0xfd400420, 0xbd000043, 0x9e22021e, 0x9e62013e, 0x9e3803d0, 0x9e780030, 0x1e22c3fe,
        0x1e6243fe, 0x9a9f57f0, 0x9a9f87f0, 0x9a900230, 0x9a906230,
    };
    try std.testing.expectEqualSlices(u32, &want, &got);
}
