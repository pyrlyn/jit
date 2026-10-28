//! Table-driven AArch64 (A64) encoder for the GP and ASIMD forms of AsmJit's ISA database
//! (`db/isa_aarch64.json`, generated into a64_table.zig by scripts/gen_a64_tables.py).
//!
//!     try a64.emit(&buf, .add, &.{ a64.x(1), a64.x(2), a64.imm(16) });
//!     try a64.emit(&buf, .ld1, &.{ a64.vec(1, .a4s), a64.vec(2, .a4s), a64.ptrPost(a64.x(3), 32) });
//!
//! Each form lists its operand specs and bit fields; operands provide named values (register
//! numbers, immediates, arrangement size, ...) that are placed into the fields. The first form
//! (in DB order, with a few AsmJit preferences) that accepts the operands is used.
const std = @import("std");
const core = @import("../core/code_buffer.zig");
pub const tbl = @import("a64_table.zig");
pub const Mnemonic = tbl.Mnemonic;
pub const Name = tbl.Name;
pub const ImmFn = tbl.ImmFn;
const CodeBuffer = core.CodeBuffer;

pub const Error = core.Error || error{NoMatchingForm};

pub const Arr = enum(u8) { none, a8b, a16b, a4h, a8h, a2s, a4s, a1d, a2d, a1q, a4b, a2h, a2b, a1s };
pub const Et = enum(u3) { none, b, h, s, d, q };

pub const K = enum(u8) { w, x, wsp, xsp, rext, sb, sh, ss, sd, sq, vt, vta, vtb, vfix, velem, vlist, vlistelem, glist, immfix, rel, cond, fimm, imm, shift, mem, memlit };
pub const Sh = enum(u8) { none, shift3, shift4, lsl, extend, msl, lslmsl };
pub const Mode = enum(u8) { off, post, pre, any, offpre };
pub const MK = enum(u8) { none, base, imm, fixed, fixedsz, postreg, reg };

pub const OpSpec = struct {
    k: K,
    f: ?Name = null,
    f2: ?Name = null,
    arr: Arr = .none,
    et: Et = .none,
    sh: Sh = .none,
    mode: Mode = .off,
    mk: MK = .none,
    n: u8 = 0,
    w: bool = false,
    fixed: i64 = 0,
    scale: i8 = 0,
    signed: bool = false,
    opt: bool = false,
};

pub const Field = struct { n: Name, at: u5, w: u6, src: u6 };

pub const Form = struct {
    bits: u32,
    fields: []const Field,
    ops: []const OpSpec,
    t: []const Arr = &.{},
    ta: []const Arr = &.{},
    tb: []const Arr = &.{},
    imm: ImmFn = .none,
    args: []const i8 = &.{},
    /// Alias with an inverted condition (cset/csetm/cinc/cinv/cneg = csinc/csinv/csneg !cond).
    inv_cond: bool = false,
};

// ---- operands -------------------------------------------------------------------------------

/// General-purpose register: id 0..30, 31 = WZR/XZR, 32 = WSP/SP.
pub const Gp = struct { id: u6, x: bool };
pub const VKind = enum(u3) { b, h, s, d, q, v };
/// SIMD&FP register: scalar (`kind` b..q), vector (`kind` v + `arr`), or element (`et` + `idx`).
pub const V = struct { id: u5, kind: VKind, arr: Arr = .none, et: Et = .none, idx: ?u4 = null };
pub const ShiftOp = enum(u8) { lsl, lsr, asr, ror, msl, uxtb, uxth, uxtw, uxtx, sxtb, sxth, sxtw, sxtx };
pub const Shift = struct { op: ShiftOp, n: u8 };
pub const MemMode = enum(u2) { off, pre, post };
/// [base, #off] / [base, index{, ext #n}] / pre-index `[base, #off]!` / post-index `[base], #off|index`.
pub const Mem = struct { base: Gp, index: ?Gp = null, ext: ?Shift = null, off: i64 = 0, mode: MemMode = .off };
pub const Op = union(enum) { r: Gp, v: V, imm: i64, fimm: f64, shift: Shift, mem: Mem, cond: u4, rel: i64 };

pub fn w(i: u6) Op {
    return .{ .r = .{ .id = i, .x = false } };
}
pub fn x(i: u6) Op {
    return .{ .r = .{ .id = i, .x = true } };
}
pub const wzr: Op = .{ .r = .{ .id = 31, .x = false } };
pub const xzr: Op = .{ .r = .{ .id = 31, .x = true } };
pub const wsp: Op = .{ .r = .{ .id = 32, .x = false } };
pub const sp: Op = .{ .r = .{ .id = 32, .x = true } };
pub fn scalar(kind: VKind, i: u5) Op {
    return .{ .v = .{ .id = i, .kind = kind } };
}
pub fn vec(i: u5, arr: Arr) Op {
    return .{ .v = .{ .id = i, .kind = .v, .arr = arr } };
}
/// Element `Vn.<et>[idx]` (`arr` set for the 4B/2H dot-product element groups).
pub fn elem(i: u5, et: Et, idx: u4) Op {
    return .{ .v = .{ .id = i, .kind = .v, .et = et, .idx = idx } };
}
pub fn elemArr(i: u5, arr: Arr, idx: u4) Op {
    const et: Et = switch (arr) {
        .a4b, .a2b, .a8b, .a16b => .b,
        .a2h, .a4h, .a8h => .h,
        else => .s,
    };
    return .{ .v = .{ .id = i, .kind = .v, .arr = arr, .et = et, .idx = idx } };
}
pub fn imm(v: i64) Op {
    return .{ .imm = v };
}
pub fn fimm(v: f64) Op {
    return .{ .fimm = v };
}
pub fn shift(op: ShiftOp, n: u8) Op {
    return .{ .shift = .{ .op = op, .n = n } };
}
pub fn cond(c: u4) Op {
    return .{ .cond = c };
}
/// Branch displacement in bytes (relative to the instruction).
pub fn rel(d: i64) Op {
    return .{ .rel = d };
}
fn gpOf(o: Op) Gp {
    return o.r;
}
pub fn ptr(base: Op, off: i64) Op {
    return .{ .mem = .{ .base = gpOf(base), .off = off } };
}
pub fn ptrReg(base: Op, index: Op, ext: ?Op) Op {
    return .{ .mem = .{ .base = gpOf(base), .index = gpOf(index), .ext = if (ext) |e| e.shift else null } };
}
pub fn ptrPre(base: Op, off: i64) Op {
    return .{ .mem = .{ .base = gpOf(base), .off = off, .mode = .pre } };
}
pub fn ptrPost(base: Op, off: i64) Op {
    return .{ .mem = .{ .base = gpOf(base), .off = off, .mode = .post } };
}
pub fn ptrPostReg(base: Op, index: Op) Op {
    return .{ .mem = .{ .base = gpOf(base), .index = gpOf(index), .mode = .post } };
}

// ---- encoder --------------------------------------------------------------------------------

const n_names = @typeInfo(Name).@"enum".fields.len;

const Ctx = struct {
    vals: [n_names]i64 = undefined,
    set: std.StaticBitSet(n_names) = .initEmpty(),
    sz: ?u3 = null,
    /// log2 of the scalar/element register size for shifts and fbits (b=0 .. q=4)
    esz: ?u3 = null,
    rext_x: bool = false,
    /// smallest element size (log2) among SIMD operands: the `size`/`sz` field
    min_el: ?u3 = null,

    fn el(c: *Ctx, l: u3) void {
        c.min_el = if (c.min_el) |m| @min(m, l) else l;
    }

    fn put(c: *Ctx, n: Name, v: i64) void {
        c.vals[@intFromEnum(n)] = v;
        c.set.set(@intFromEnum(n));
    }
    fn get(c: *const Ctx, n: Name) ?i64 {
        return if (c.set.isSet(@intFromEnum(n))) c.vals[@intFromEnum(n)] else null;
    }
};

fn hasField(f: *const Form, n: Name) bool {
    for (f.fields) |fl| if (fl.n == n) return true;
    return false;
}
fn fieldWidth(f: *const Form, n: Name) u7 {
    var t: u7 = 0;
    for (f.fields) |fl| if (fl.n == n) {
        t += fl.w;
    };
    return t;
}

fn indexOf(list: []const Arr, a: Arr) ?u3 {
    for (list, 0..) |e, i| if (e == a and e != .none) return @intCast(i);
    return null;
}

fn regGp(o: Op, xw: bool, allow_sp: bool, allow_zr: bool) ?u5 {
    if (o != .r) return null;
    const r = o.r;
    if (r.x != xw) return null;
    if (r.id == 32) return if (allow_sp) 31 else null;
    if (r.id == 31) return if (allow_zr) 31 else null;
    return @intCast(r.id);
}

fn sizeLog2(kind: VKind) u3 {
    return switch (kind) {
        .b => 0,
        .h => 1,
        .s => 2,
        .d => 3,
        .q => 4,
        .v => 4,
    };
}
fn etLog2(et: Et) u3 {
    return switch (et) {
        .b => 0,
        .h => 1,
        .s => 2,
        .d => 3,
        .q => 4,
        .none => 0,
    };
}
fn arrEt(a: Arr) u3 {
    return switch (a) {
        .a8b, .a16b, .a4b, .a2b => 0,
        .a4h, .a8h, .a2h => 1,
        .a2s, .a4s, .a1s => 2,
        .a1d, .a2d => 3,
        .a1q => 4,
        .none => 0,
    };
}

fn extOption(op: ShiftOp, rm_x: bool) ?i64 {
    return switch (op) {
        .uxtb => 0,
        .uxth => 1,
        .uxtw => 2,
        .uxtx => 3,
        .sxtb => 4,
        .sxth => 5,
        .sxtw => 6,
        .sxtx => 7,
        .lsl => if (rm_x) 3 else 2,
        else => null,
    };
}

/// Register-offset memory access size (log2) for `ImmLDR*`-style forms.
fn accessLog2(f: *const Form, first: Op) i8 {
    return switch (f.imm) {
        .ImmLDRB, .ImmLDRB_STRB => 0,
        .ImmLDRH, .ImmLDRH_STRH => 1,
        .ImmLDRW => 2,
        .ImmLDR, .ImmLDR_STR => if (first == .r and first.r.x) 3 else 2,
        .ImmPRF => 3,
        else => if (first == .v) sizeLog2(first.v.kind) else if (first == .r and first.r.x) 3 else 2,
    };
}

fn matchOps(f: *const Form, ops: []const Op, c: *Ctx) bool {
    var oi: usize = 0;
    for (f.ops) |s| {
        const have = oi < ops.len;
        const o: Op = if (have) ops[oi] else .{ .imm = 0 };
        switch (s.k) {
            .w, .x => {
                const id = regGp(o, s.k == .x, false, true) orelse return false;
                if (s.f) |n| c.put(n, id) else if (id != s.fixed) return false; // fixed register (chkfeat X16)
                oi += 1;
            },
            .wsp, .xsp => {
                const id = regGp(o, s.k == .xsp, true, false) orelse return false;
                c.put(s.f.?, id);
                oi += 1;
            },
            .rext => {
                if (o != .r or o.r.id == 32) return false;
                c.rext_x = o.r.x;
                c.put(s.f.?, @intCast(o.r.id & 31));
                oi += 1;
            },
            .sb, .sh, .ss, .sd, .sq => {
                const want: VKind = switch (s.k) {
                    .sb => .b,
                    .sh => .h,
                    .ss => .s,
                    .sd => .d,
                    else => .q,
                };
                if (o != .v or o.v.kind != want) return false;
                c.put(s.f.?, o.v.id);
                if (c.esz == null) c.esz = sizeLog2(want);
                c.el(sizeLog2(want));
                oi += 1;
            },
            .vt, .vta, .vtb, .vfix => {
                if (o != .v or o.v.idx != null) return false;
                // AsmJit accepts scalar D/Q registers for the 1D/1Q arrangements.
                const a: Arr = if (o.v.kind == .v) o.v.arr else if (o.v.kind == .d and o.v.arr == .none) .a1d else if (o.v.kind == .q and o.v.arr == .none) .a1q else return false;
                if (a == .none) return false;
                switch (s.k) {
                    .vfix => if (a != s.arr) return false,
                    .vt, .vta => {
                        const list = if (s.k == .vt) f.t else f.ta;
                        const i = indexOf(list, a) orelse {
                            // SSHLL2/USHLL2: AsmJit derives the size from the source only.
                            if (s.k == .vta and f.tb.len > 0 and f.imm == .ASimdShiftPImm) {
                                c.put(s.f.?, o.v.id);
                                oi += 1;
                                continue;
                            }
                            return false;
                        };
                        if (c.sz) |z| if (z != i) return false;
                        c.sz = i;
                    },
                    else => {
                        const i = indexOf(f.tb, a) orelse return false;
                        if (c.sz) |z| {
                            if (f.tb[z] != a) return false;
                        } else c.sz = i;
                    },
                }
                if (c.esz == null) c.esz = arrEt(a);
                c.el(arrEt(a));
                c.put(s.f.?, o.v.id);
                oi += 1;
            },
            .velem => {
                if (o != .v or o.v.kind != .v or o.v.idx == null) return false;
                if (o.v.et != s.et) return false;
                if (s.arr != .none and o.v.arr != s.arr) return false;
                if (s.arr == .none and o.v.arr != .none) return false;
                if (s.f2) |n| c.put(n, o.v.idx.?) else if (o.v.idx.? != s.fixed) return false;
                c.put(s.f.?, o.v.id);
                if (c.esz == null) c.esz = etLog2(s.et);
                oi += 1;
            },
            .vlist, .vlistelem => {
                if (oi + s.n > ops.len) return false;
                const first = ops[oi];
                if (first != .v or first.v.kind != .v) return false;
                for (0..s.n) |j| {
                    const e = ops[oi + j];
                    if (e != .v or e.v.kind != .v) return false;
                    if (e.v.id != @as(u5, @truncate(first.v.id +% j))) return false;
                    if (e.v.arr != first.v.arr or e.v.et != first.v.et or !std.meta.eql(e.v.idx, first.v.idx)) return false;
                }
                if (s.k == .vlist) {
                    if (first.v.idx != null) return false;
                    if (s.arr != .none) {
                        if (first.v.arr != s.arr) return false;
                    } else {
                        const i = indexOf(f.t, first.v.arr) orelse return false;
                        c.sz = i;
                    }
                    if (c.esz == null) c.esz = arrEt(first.v.arr);
                    c.el(arrEt(first.v.arr));
                } else {
                    if (first.v.idx == null or first.v.et != s.et or first.v.arr != .none) return false;
                    c.put(s.f2.?, first.v.idx.?);
                    c.esz = etLog2(s.et);
                }
                c.put(s.f.?, first.v.id);
                oi += s.n;
            },
            .glist => {
                if (oi + s.n > ops.len) return false;
                const id0 = regGp(ops[oi], s.w, false, true) orelse return false;
                for (1..s.n) |j| {
                    const id = regGp(ops[oi + j], s.w, false, true) orelse return false;
                    if (id != id0 + j) return false;
                }
                c.put(s.f.?, id0);
                oi += s.n;
            },
            .immfix => {
                if (!have) return false;
                switch (o) {
                    .imm => |v| if (v != s.fixed) return false,
                    .fimm => |v| if (v != @as(f64, @floatFromInt(s.fixed))) return false,
                    else => return false,
                }
                oi += 1;
            },
            .rel => {
                if (!have or o != .rel) return false;
                const d = o.rel;
                const sc: u6 = @intCast(s.scale);
                if (@rem(d, @as(i64, 1) << sc) != 0) return false;
                c.put(s.f.?, d >> sc);
                oi += 1;
            },
            .cond => {
                if (!have or o != .cond) return false;
                if (f.inv_cond and o.cond >= 14) return false;
                c.put(s.f.?, if (f.inv_cond) o.cond ^ 1 else o.cond);
                oi += 1;
            },
            .fimm => {
                if (!have) return false;
                const v: f64 = switch (o) {
                    .fimm => |v| v,
                    .imm => |v| @floatFromInt(v),
                    else => return false,
                };
                c.put(s.f.?, @bitCast(v));
                oi += 1;
            },
            .imm => {
                if (have and o == .imm) {
                    const sc: u6 = @intCast(s.scale);
                    if (@rem(o.imm, @as(i64, 1) << sc) != 0) return false;
                    c.put(s.f.?, o.imm >> sc);
                    oi += 1;
                } else if (s.opt) {
                    c.put(s.f.?, s.fixed);
                } else return false;
            },
            .shift => {
                var op: ShiftOp = .lsl;
                var amt: i64 = 0;
                var given = false;
                if (have and o == .shift) {
                    op = o.shift.op;
                    amt = o.shift.n;
                    given = true;
                    oi += 1;
                }
                switch (s.sh) {
                    .shift3, .shift4 => {
                        const v: i64 = switch (op) {
                            .lsl => 0,
                            .lsr => 1,
                            .asr => 2,
                            .ror => if (s.sh == .shift4) 3 else return false,
                            else => return false,
                        };
                        c.put(.sop, v);
                    },
                    .lsl => if (op != .lsl) return false,
                    .msl => if (op != .msl or !given) return false,
                    .lslmsl => if (op != .lsl and op != .msl) return false,
                    .extend => {
                        const v = if (given) (extOption(op, c.rext_x) orelse return false) else (if (c.rext_x) @as(i64, 3) else 2);
                        c.put(.option, v);
                    },
                    .none => return false,
                }
                if (s.scale != 0) {
                    if (@rem(amt, s.scale) != 0) return false;
                    amt = @divTrunc(amt, s.scale);
                }
                c.put(s.f.?, amt);
            },
            .mem => {
                if (!have or o != .mem) return false;
                const m = o.mem;
                if (m.base.id == 31 or !m.base.x) return false;
                c.put(.Rn, if (m.base.id == 32) 31 else m.base.id);
                const mode_ok = switch (s.mode) {
                    .off => m.mode == .off,
                    .post => m.mode == .post,
                    .pre => m.mode == .pre,
                    .any => true,
                    .offpre => m.mode != .post,
                };
                if (!mode_ok) return false;
                if (s.mode == .any or s.mode == .offpre) {
                    if (hasField(f, .@"!post")) c.put(.@"!post", if (m.mode == .post) 0 else 1);
                    c.put(.W, if (m.mode == .off) 0 else 1);
                }
                switch (s.mk) {
                    .base => if (m.index != null or m.off != 0) return false,
                    .imm => {
                        if (m.index != null) return false;
                        const sc: u6 = @intCast(s.scale);
                        if (@rem(m.off, @as(i64, 1) << sc) != 0) return false;
                        c.put(s.f.?, m.off >> sc);
                    },
                    .fixed, .fixedsz => {
                        if (m.index != null) return false;
                        const want = if (s.mk == .fixed) s.fixed else s.fixed << (c.esz orelse 0);
                        if (m.off != want) return false;
                    },
                    .postreg => {
                        const ix = m.index orelse return false;
                        if (!ix.x or ix.id >= 31) return false;
                        c.put(.Rm, ix.id);
                    },
                    .reg => {
                        const ix = m.index orelse return false;
                        if (ix.id >= 31) {
                            if (ix.id != 31) return false;
                        }
                        c.put(.Rm, ix.id & 31);
                        const e: Shift = m.ext orelse Shift{ .op = .lsl, .n = 0 };
                        // AsmJit does not check the index width for UXTW/SXTW (`[x3, x4, sxtw]`).
                        if (ix.x and !(e.op == .lsl or e.op == .sxtx or e.op == .uxtw or e.op == .sxtw)) return false;
                        if (!ix.x and !(e.op == .uxtw or e.op == .sxtw)) return false;
                        c.put(.option, switch (e.op) {
                            .uxtw => 2,
                            .lsl => 3,
                            .sxtw => 6,
                            else => 7,
                        });
                        const size: i8 = if (s.scale > 0) (if (s.scale == 8) 3 else s.scale) else accessLog2(f, ops[0]);
                        if (e.n != 0 and e.n != size) return false;
                        const sbit: i64 = if (e.n != 0 or (m.ext != null and size == 0 and m.ext.?.op == .lsl and false)) 1 else 0;
                        if (hasField(f, .s)) c.put(.s, sbit) else c.put(.n, sbit);
                    },
                    .none => return false,
                }
                oi += 1;
            },
            .memlit => {
                if (!have or o != .rel) return false;
                if (@rem(o.rel, 4) != 0) return false;
                c.put(.offS, o.rel >> 2);
                oi += 1;
            },
        }
    }
    return oi == ops.len;
}

fn fitsU(v: i64, bits: u7) bool {
    return v >= 0 and (bits >= 63 or v < (@as(i64, 1) << @intCast(bits)));
}
fn fitsS(v: i64, bits: u7) bool {
    const b: u6 = @intCast(bits - 1);
    return v >= -(@as(i64, 1) << b) and v < (@as(i64, 1) << b);
}

/// Logical (bitmask) immediate -> N:immr:imms, or null.
pub fn logicalImm(value: u64, is64: bool) ?struct { n: u1, immr: u6, imms: u6 } {
    var v = value;
    if (!is64) {
        if (v >> 32 != 0) return null;
        v |= v << 32;
    }
    if (v == 0 or v == std.math.maxInt(u64)) return null;
    var size: u7 = 64;
    while (size > 2) {
        const half: u6 = @intCast(size / 2);
        const mask = (@as(u64, 1) << half) - 1;
        if ((v & mask) != ((v >> half) & mask)) break;
        size /= 2;
    }
    const mask: u64 = if (size == 64) std.math.maxInt(u64) else (@as(u64, 1) << @intCast(size)) - 1;
    const e = v & mask;
    // rotate so that the run of ones starts at bit 0
    const ones = @popCount(e);
    var r: u7 = 0;
    while (r < size) : (r += 1) {
        const rot = if (r == 0) e else ((e >> @intCast(r)) | (e << @intCast(size - r))) & mask;
        if (rot == (@as(u64, 1) << @intCast(ones)) - 1) break;
    }
    if (r == size) return null;
    const immr: u7 = (size - r) % size;
    const imms: u7 = ((~(size - 1) << 1) & 0x3F) | (ones - 1);
    return .{ .n = if (size == 64) 1 else 0, .immr = @intCast(immr), .imms = @intCast(imms & 0x3F) };
}

/// FP immediate -> imm8 (`a:b:c:d:e:f:g:h`, value = (-1)^a * (16 + efgh) / 16 * 2^e, e in -3..4).
pub fn fpImm8(v: f64) ?u8 {
    const bits: u64 = @bitCast(v);
    const sign: u8 = @intCast(bits >> 63);
    const e: i64 = @as(i64, @intCast((bits >> 52) & 0x7FF)) - 1023;
    const frac = bits & ((@as(u64, 1) << 52) - 1);
    if (frac & ((@as(u64, 1) << 48) - 1) != 0) return null;
    if (e < -3 or e > 4) return null;
    const bcd: u8 = if (e <= 0) @intCast(4 | (e + 3)) else @intCast(e - 1);
    return (sign << 7) | (bcd << 4) | @as(u8, @intCast(frac >> 48));
}

fn applyImm(f: *const Form, c: *Ctx, ops: []const Op, bits: *u32) bool {
    const arg = struct {
        fn a(ff: *const Form, i: usize) i8 {
            return if (i < ff.args.len) ff.args[i] else -3;
        }
    }.a;
    const is64 = ops.len > 0 and ops[0] == .r and ops[0].r.x;
    switch (f.imm) {
        .none, .ImmBFM, .ImmPRF, .ImmLDR, .ImmLDRB, .ImmLDRH, .ImmLDRW, .ImmLDR_STR, .ImmLDRB_STRB, .ImmLDRH_STRH => {},
        .LogicalImm, .ImmLogical => {
            const src: Name = if (c.get(.imm) != null) .imm else .log_imm;
            const v = c.get(src) orelse return false;
            const li = logicalImm(@bitCast(v), arg(f, 1) == 1) orelse return false;
            if (hasField(f, .immr)) {
                c.put(.immr, li.immr);
                c.put(.imms, li.imms);
                bits.* = (bits.* & ~@as(u32, 1 << 22)) | (@as(u32, li.n) << 22);
            } else c.put(.imm, (@as(i64, li.n) << 12) | (@as(i64, li.immr) << 6) | li.imms);
        },
        .ImmWide, .ImmWideInv => {
            var v: u64 = @bitCast(c.get(.imm) orelse return false);
            const wide = arg(f, 2) == 1;
            if (!wide) {
                if (v >> 32 != 0 and v >> 32 != 0xFFFFFFFF) return false;
                v &= 0xFFFFFFFF;
            }
            if (f.imm == .ImmWideInv) v = ~v & (if (wide) @as(u64, std.math.maxInt(u64)) else 0xFFFFFFFF);
            const nh: u6 = if (wide) 4 else 2;
            var hw: u6 = 0;
            while (hw < nh) : (hw += 1) {
                if (v & ~(@as(u64, 0xFFFF) << (hw * 16)) == 0) break;
            }
            if (hw == nh) return false;
            c.put(.hw, hw);
            c.put(.imm, @intCast((v >> (hw * 16)) & 0xFFFF));
        },
        .ShiftImm => {
            const n = c.get(.n) orelse 0;
            if (n >= (if (is64) @as(i64, 64) else 32)) return false;
        },
        .ImmLSL => {
            const n = c.get(.n) orelse return false;
            const size: i64 = if (arg(f, 1) == 1) 64 else 32;
            if (n < 0 or n >= size) return false;
            c.put(.immr, @mod(size - n, size));
            c.put(.imms, size - 1 - n);
        },
        .ImmISB, .ImmDataBarrier => {
            const v = c.get(.isb_op) orelse c.get(.barrier_op) orelse c.get(.imm) orelse return false;
            if (v < 0 or v > 15) return false;
            c.put(.CRm, v);
        },
        .ImmAt, .ImmDC, .ImmIC, .ImmTLBI => {
            // AsmJit packs op1:CRn:CRm:op2 as (op1 << 11) | (CRn << 7) | (CRm << 3) | op2.
            const v = c.get(.at_op) orelse c.get(.dc_op) orelse c.get(.ic_op) orelse c.get(.tlbi_op) orelse return false;
            if (v < 0 or v >= (1 << 14)) return false;
            if (((bits.* >> 12) & 15) != ((v >> 7) & 15)) return false; // CRn is fixed by the form
            c.put(.op1, (v >> 11) & 7);
            c.put(.CRm, (v >> 3) & 15);
            c.put(.op2, v & 7);
        },
        .ImmBTI => {
            const v = c.get(.targets) orelse 0;
            if (v < 0 or v > 3) return false;
            c.put(.op2, v << 1);
        },
        .ImmSysReg => {
            // op0:op1:CRn:CRm:op2 (16 bits); op0 is 2 or 3, its high bit is fixed in the opcode.
            const v = c.get(.sysreg) orelse return false;
            if (v < 0x8000 or v > 0xFFFF) return false;
            c.put(.sysreg, v & 0x7FFF);
        },
        .ASimdMovPImm, .ASimdMovNImm, .ASimdLogicalImm => if (!simdImm(f, c, ops)) return false,
        .ImmBFX, .ImmBFIZ => {
            const lsb = c.get(.lsb) orelse return false;
            const width = c.get(.width) orelse return false;
            const size: i64 = if (arg(f, 2) == 1) 64 else 32;
            if (lsb < 0 or width < 1 or lsb + width > size) return false;
            if (f.imm == .ImmBFX) {
                c.put(.immr, lsb);
                c.put(.imms, lsb + width - 1);
            } else {
                c.put(.immr, @mod(size - lsb, size));
                c.put(.imms, width - 1);
            }
        },
        .ASimdShiftNImm, .ASimdSHRN, .ASimdShiftPImm, .ASimdSHL => {
            const n = c.get(.n) orelse return false;
            const a = arg(f, 1);
            const l: u3 = if (a >= 0) @intCast(a) else (c.sz orelse return false);
            const es: i64 = @as(i64, 8) << l;
            const v = if (f.imm == .ASimdShiftNImm or f.imm == .ASimdSHRN) blk: {
                if (n < 1 or n > es) return false;
                break :blk 2 * es - n;
            } else blk: {
                if (n < 0 or n >= es) return false;
                break :blk es + n;
            };
            c.put(.immh, v >> 3);
            c.put(.immb, v & 7);
        },
        .ASimdXtlImm => {
            const l: u3 = c.sz orelse return false;
            c.put(.immh, (@as(i64, 8) << l) >> 3);
        },
        .ASimdFBitsHBImm => {
            const fb = c.get(.fbits) orelse c.get(.bits) orelse return false;
            const size: i64 = arg(f, 1);
            if (fb < 1 or fb > size) return false;
            const v = 2 * size - fb;
            c.put(.immh, v >> 3);
            c.put(.immb, v & 7);
        },
        .ASimdFBitsScaleImm => {
            const fb = c.get(.fbits) orelse return false;
            const size: i64 = arg(f, 1);
            if (fb < 1 or fb > size) return false;
            c.put(.scale, 64 - fb);
        },
        .ASimdFMovImm => {
            const v: f64 = @bitCast(c.get(.fimm) orelse return false);
            const i8v = fpImm8(v) orelse return false;
            if (hasField(f, .imm)) c.put(.imm, i8v) else {
                c.put(.abc, i8v >> 5);
                c.put(.defgh, i8v & 31);
            }
        },
        .ASimdRotateImm_0_90_180_270 => {
            const r = c.get(.rotate) orelse return false;
            if (@rem(r, 90) != 0 or r < 0 or r > 270) return false;
            c.put(.imm, @divTrunc(r, 90));
        },
        .ASimdRotateImm_90_270 => {
            const r = c.get(.rotate) orelse return false;
            if (r != 90 and r != 270) return false;
            c.put(.imm, if (r == 270) 1 else 0);
        },
        else => return false,
    }
    return true;
}

/// AdvSIMDExpandImm(op, cmode, imm8) -> 64-bit value.
fn expandImm(op: u1, cmode: u4, imm8: u8) u64 {
    const b: u64 = imm8;
    const r32 = struct {
        fn f(v: u64) u64 {
            return v | (v << 32);
        }
    }.f;
    const r16 = struct {
        fn f(v: u64) u64 {
            return v | (v << 16) | (v << 32) | (v << 48);
        }
    }.f;
    return switch (cmode >> 1) {
        0 => r32(b),
        1 => r32(b << 8),
        2 => r32(b << 16),
        3 => r32(b << 24),
        4 => r16(b),
        5 => r16(b << 8),
        6 => if (cmode & 1 == 0) r32((b << 8) | 0xFF) else r32((b << 16) | 0xFFFF),
        else => if (op == 0) b * 0x0101_0101_0101_0101 else blk: {
            var v: u64 = 0;
            for (0..8) |i| if ((imm8 >> @intCast(i)) & 1 != 0) {
                v |= @as(u64, 0xFF) << @intCast(i * 8);
            };
            break :blk v;
        },
    };
}

/// MOVI/MVNI/ORR/BIC (vector, immediate): finds op/cmode/imm8 whose expansion equals the
/// requested value (`imm << lsl`, or `(imm << n) | ones` for MSL) replicated over the lanes.
fn simdImm(f: *const Form, c: *Ctx, ops: []const Op) bool {
    const iv: u64 = @bitCast(c.get(.imm) orelse return false);
    const n = c.get(.n) orelse 0;
    if (n < 0 or n > 24) return false;
    var msl = false;
    for (ops) |o| if (o == .shift and o.shift.op == .msl) {
        msl = true;
    };
    const lane: u7 = if (ops.len > 0 and ops[0] == .v and ops[0].v.kind == .v) @as(u7, 8) << arrEt(ops[0].v.arr) else 64;
    const lmask: u64 = if (lane == 64) std.math.maxInt(u64) else (@as(u64, 1) << @intCast(lane)) - 1;
    const sh: u6 = @intCast(n);
    if (sh != 0 and (iv >> @intCast(@as(u7, lane) - sh)) != 0 and lane != 64) return false;
    var v = (iv << sh) & lmask;
    if (msl) v |= (@as(u64, 1) << sh) - 1;
    if (v & ~lmask != 0 or (iv & ~lmask) != 0) return false;
    var l: u7 = lane;
    while (l < 64) : (l *= 2) v |= v << @intCast(l);
    const has_op = hasField(f, .op);
    // Candidate cmodes in AsmJit's order of preference.
    const Cand = struct { op: u1, cmode: u4 };
    const mov = [_]Cand{ .{ .op = 0, .cmode = 0b1110 }, .{ .op = 0, .cmode = 0b1000 }, .{ .op = 0, .cmode = 0b1010 }, .{ .op = 0, .cmode = 0b0000 }, .{ .op = 0, .cmode = 0b0010 }, .{ .op = 0, .cmode = 0b0100 }, .{ .op = 0, .cmode = 0b0110 }, .{ .op = 0, .cmode = 0b1100 }, .{ .op = 0, .cmode = 0b1101 }, .{ .op = 1, .cmode = 0b1110 } };
    const mvn = [_]Cand{ .{ .op = 1, .cmode = 0b1000 }, .{ .op = 1, .cmode = 0b1010 }, .{ .op = 1, .cmode = 0b0000 }, .{ .op = 1, .cmode = 0b0010 }, .{ .op = 1, .cmode = 0b0100 }, .{ .op = 1, .cmode = 0b0110 }, .{ .op = 1, .cmode = 0b1100 }, .{ .op = 1, .cmode = 0b1101 } };
    const logic = [_]Cand{ .{ .op = 0, .cmode = 0b1001 }, .{ .op = 0, .cmode = 0b1011 }, .{ .op = 0, .cmode = 0b0001 }, .{ .op = 0, .cmode = 0b0011 }, .{ .op = 0, .cmode = 0b0101 }, .{ .op = 0, .cmode = 0b0111 } };
    const cands: []const Cand = switch (f.imm) {
        .ASimdMovPImm => &mov,
        .ASimdMovNImm => &mvn,
        else => &logic,
    };
    for (cands) |cd| {
        if (msl != (cd.cmode >> 1 == 6)) continue;
        // Lane size of the encoding must match the arrangement (except 8-bit / 64-bit patterns).
        const enc_lane: u7 = switch (cd.cmode >> 1) {
            0, 1, 2, 3, 6 => 32,
            4, 5 => 16,
            else => if (cd.op == 0) 8 else 64,
        };
        if (f.imm != .ASimdMovPImm and enc_lane != lane) continue;
        if (f.imm == .ASimdMovPImm and lane != 64 and enc_lane > lane) continue;
        // imm8 candidates: derive from the value for this cmode.
        var b: u16 = 0;
        while (b < 256) : (b += 1) {
            // MVNI takes the value before inversion (like AsmJit); only MOVI's 64-bit form uses op=1.
            const eop: u1 = if (cd.cmode == 0b1110) cd.op else 0;
            if (expandImm(eop, cd.cmode, @intCast(b)) != v) continue;
            if (has_op) c.put(.op, cd.op);
            c.put(.cmode, cd.cmode);
            c.put(.abc, b >> 5);
            c.put(.defgh, b & 31);
            return true;
        }
    }
    return false;
}

fn signedName(n: Name) bool {
    return n == .offS or n == .relS or n == .immS;
}

/// Rejects operand values no form can accept (instead of relying on each spec to catch them).
fn opsValid(ops: []const Op) bool {
    for (ops) |o| switch (o) {
        .r => |r| if (r.id > 32) return false,
        .mem => |m| {
            if (m.base.id > 32) return false;
            if (m.index) |i| if (i.id > 31) return false;
        },
        .fimm => |v| if (!std.math.isFinite(v)) return false,
        else => {},
    };
    return true;
}

pub fn encode(f: *const Form, ops: []const Op) ?u32 {
    if (!opsValid(ops)) return null;
    var c: Ctx = .{};
    if (!matchOps(f, ops, &c)) return null;
    var bits = f.bits;
    if (hasField(f, .sz)) {
        // `size`: log2 of the (smallest) element; 1-bit `sz` (FP): S = 0, D = 1.
        const l = c.min_el orelse return null;
        const wsz = fieldWidth(f, .sz);
        c.put(.sz, if (wsz == 1) @as(i64, l) - 2 else l);
    }
    if (hasField(f, .hw) and c.get(.hw) == null and f.imm == .none) {
        const n = c.get(.n) orelse 0;
        if (@rem(n, 16) != 0) return null;
        c.put(.hw, @divTrunc(n, 16));
    }
    if (!applyImm(f, &c, ops, &bits)) return null;
    for (f.fields) |fl| {
        const v = c.get(fl.n) orelse return null;
        const tw = fieldWidth(f, fl.n);
        if (signedName(fl.n)) {
            if (!fitsS(v, tw)) return null;
        } else if (!fitsU(v, tw)) return null;
        const u: u64 = @bitCast(v);
        const mask: u64 = (@as(u64, 1) << @intCast(fl.w)) - 1;
        bits |= @intCast(((u >> fl.src) & mask) << fl.at);
    }
    return bits;
}

pub fn encodeAny(forms: []const Form, ops: []const Op) ?u32 {
    return if (encodeIdx(forms, ops)) |r| r.word else null;
}

/// Encoded word and the index of the form that produced it.
pub fn encodeIdx(forms: []const Form, ops: []const Op) ?struct { word: u32, idx: usize } {
    for (forms, 0..) |*f, i| if (encode(f, ops)) |b| return .{ .word = b, .idx = i };
    return null;
}

const all_forms = blk: {
    @setEvalBranchQuota(1_000_000);
    const fs = @typeInfo(Mnemonic).@"enum".fields;
    var out: [fs.len][]const Form = undefined;
    for (fs, 0..) |f, i| out[i] = &@field(tbl.table, f.name);
    break :blk out;
};

pub fn formsOf(mn: Mnemonic) []const Form {
    return all_forms[@intFromEnum(mn)];
}

/// Encodes `mn`, applying AsmJit's assembler-level aliases when no form matches directly:
/// - LDR/STR (+B/H/SB/SH/SW) and PRFM with an unaligned or negative offset -> LDUR/STUR/PRFUM;
/// - BIC/BICS with an immediate -> AND/ANDS with the inverted immediate;
/// - ADD/SUB/CMP/CMN with an immediate that is a multiple of 4096 -> `#imm >> 12, lsl #12`.
pub fn encodeMnIdx(mn: Mnemonic, ops: []const Op) ?struct { word: u32, idx: usize } {
    if (encodeIdx(formsOf(mn), ops)) |r| return .{ .word = r.word, .idx = r.idx };
    const alt: ?Mnemonic = switch (mn) {
        .ldr => .ldur,
        .ldrb => .ldurb,
        .ldrh => .ldurh,
        .ldrsb => .ldursb,
        .ldrsh => .ldursh,
        .ldrsw => .ldursw,
        .str => .stur,
        .strb => .sturb,
        .strh => .sturh,
        .prfm => .prfum,
        else => null,
    };
    if (alt) |a| if (ops.len >= 2 and ops[ops.len - 1] == .mem and ops[ops.len - 1].mem.mode == .off and ops[ops.len - 1].mem.index == null) {
        if (encodeIdx(formsOf(a), ops)) |r| return .{ .word = r.word, .idx = 1000 + r.idx };
    };
    if ((mn == .bic or mn == .bics) and ops.len == 3 and ops[0] == .r and ops[2] == .imm) {
        var o = ops[0..3].*;
        const inv: u64 = ~@as(u64, @bitCast(o[2].imm));
        o[2] = .{ .imm = @bitCast(if (o[0].r.x) inv else inv & 0xFFFF_FFFF) };
        if (encodeIdx(formsOf(if (mn == .bic) .@"and" else .ands), &o)) |r| return .{ .word = r.word, .idx = 2000 + r.idx };
    }
    if ((mn == .add or mn == .adds or mn == .sub or mn == .subs or mn == .cmp or mn == .cmn) and ops.len >= 2 and ops.len <= 3 and ops[ops.len - 1] == .imm) {
        const v = ops[ops.len - 1].imm;
        if (v > 0 and v & 0xFFF == 0 and v >> 12 < 4096) {
            var o: [4]Op = undefined;
            @memcpy(o[0..ops.len], ops);
            o[ops.len - 1] = .{ .imm = v >> 12 };
            o[ops.len] = .{ .shift = .{ .op = .lsl, .n = 12 } };
            if (encodeIdx(formsOf(mn), o[0 .. ops.len + 1])) |r| return .{ .word = r.word, .idx = 3000 + r.idx };
        }
    }
    return null;
}

pub fn encodeMn(mn: Mnemonic, ops: []const Op) ?u32 {
    return if (encodeMnIdx(mn, ops)) |r| r.word else null;
}

pub fn emit(buf: *CodeBuffer, comptime mn: Mnemonic, ops: []const Op) Error!void {
    const word = encodeMn(mn, ops) orelse return error.NoMatchingForm;
    try buf.ensure(4);
    buf.put32(word);
}
