//! Table-driven x86-64 GP instruction encoder (forms generated from AsmJit's ISA database by
//! scripts/gen_gp_tables.py). Covers 8/16/32/64-bit register, memory and immediate operands,
//! including RIP-relative memory (raw displacement or label).
//!
//!     try gp.emit(&buf, .add, .{ gp.r32(.rcx), gp.dword(x.ptr(.rax, 8)) });
//!     try gp.emit(&buf, .mov, .{ gp.r64(.rax), gp.ripLabel(l, .q) });
//!
//! Form selection: among the forms whose operand kinds match, the shortest encoding wins;
//! ties go to the first form (MR before RM), which matches LLVM's choices.

const std = @import("std");
const core = @import("../core/code_buffer.zig");
const asm_ = @import("assembler.zig");
pub const tbl = @import("gp_table.zig");
const isa = @import("isa.zig");
pub const Mnemonic = tbl.Mnemonic;
const CodeBuffer = core.CodeBuffer;
const Label = core.Label;
const Gp = asm_.Gp;

pub const Error = core.Error || error{ InvalidOperand, NoMatchingForm };

pub const Kind = enum {
    r8, r16, r32, r64, m8, m16, m32, m64, m,
    i8, s8, u8, i16, u16, i32, s32, u32, i64, one,
    al, ax, eax, rax, cl, dx,
};

pub const Form = struct {
    ops: []const Kind,
    pre: []const u8,
    op: []const u8,
    rexw: bool,
    plus_r: bool,
    modrm: bool,
    digit: ?u3,
    imm: u8,
    reg: ?u2,
    rm: ?u2,
    opreg: ?u2,
    /// CPU features required (AsmJit `ext`), empty for baseline x86-64.
    features: []const isa.Feature,
};

/// Union of the features of all forms of `mn` (used for the comptime target/deny checks).
pub fn requiredFeatures(comptime mn: Mnemonic) isa.Features {
    const set = comptime blk: {
        @setEvalBranchQuota(100_000);
        var list: []const isa.Feature = &.{};
        for (@field(tbl.table, @tagName(mn))) |f| list = list ++ f.features;
        break :blk isa.featureSet(list);
    };
    return set;
}

pub const Size = enum(u8) { none = 0, b = 1, w = 2, d = 4, q = 8 };

pub const Reg = struct { id: Gp, size: Size };

/// Sized memory operand. `rip`: [rip + disp] or [rip + label] (base/index must be null).
pub const Mem = struct {
    base: ?Gp = null,
    index: ?Gp = null,
    scale: asm_.Scale = .x1,
    disp: i32 = 0,
    size: Size = .none,
    rip: bool = false,
    label: ?Label = null,
};

pub const Op = union(enum) { reg: Reg, mem: Mem, imm: i64 };

pub inline fn r8(g: Gp) Reg {
    return .{ .id = g, .size = .b };
}
pub inline fn r16(g: Gp) Reg {
    return .{ .id = g, .size = .w };
}
pub inline fn r32(g: Gp) Reg {
    return .{ .id = g, .size = .d };
}
pub inline fn r64(g: Gp) Reg {
    return .{ .id = g, .size = .q };
}
inline fn sized(m: asm_.Mem, s: Size) Mem {
    return .{ .base = m.base, .index = m.index, .scale = m.scale, .disp = m.disp, .size = s };
}
pub inline fn byte(m: asm_.Mem) Mem {
    return sized(m, .b);
}
pub inline fn word(m: asm_.Mem) Mem {
    return sized(m, .w);
}
pub inline fn dword(m: asm_.Mem) Mem {
    return sized(m, .d);
}
pub inline fn qword(m: asm_.Mem) Mem {
    return sized(m, .q);
}
/// Unsized memory (e.g. `lea`).
pub inline fn mem(m: asm_.Mem) Mem {
    return sized(m, .none);
}
/// [rip + disp], disp relative to the end of the instruction.
pub inline fn rip(disp: i32, s: Size) Mem {
    return .{ .rip = true, .disp = disp, .size = s };
}
/// [rip + label]; resolved at `finalize`.
pub inline fn ripLabel(l: Label, s: Size) Mem {
    return .{ .rip = true, .label = l, .size = s };
}

fn toOp(v: anytype) Op {
    const T = @TypeOf(v);
    return switch (T) {
        Reg => .{ .reg = v },
        Mem => .{ .mem = v },
        Op => v,
        comptime_int => .{ .imm = v },
        else => switch (@typeInfo(T)) {
            .int => .{ .imm = @intCast(v) },
            else => @compileError("unsupported operand type " ++ @typeName(T)),
        },
    };
}

fn regSize(k: Kind) Size {
    return switch (k) {
        .r8, .m8, .al, .cl => .b,
        .r16, .m16, .ax, .dx => .w,
        .r32, .m32, .eax => .d,
        .r64, .m64, .rax => .q,
        else => .none,
    };
}

/// Normalizes an immediate to an operand of `bytes` bytes: values in the unsigned range wrap
/// to their signed meaning (so `add eax, 0xFFFFFFFF` can use the imm8 form like LLVM).
fn normImm(v: i64, bytes: u8) ?i64 {
    if (bytes >= 8) return v;
    const bits: u6 = @intCast(bytes * 8);
    const lo = -(@as(i64, 1) << (bits - 1));
    const hi_u = (@as(i64, 1) << bits) - 1;
    if (v < lo or v > hi_u) return null;
    return if (v > (@as(i64, 1) << (bits - 1)) - 1) v - (@as(i64, 1) << bits) else v;
}

fn fits(v: i64, comptime T: type) bool {
    return std.math.cast(T, v) != null;
}

/// Operand size used for immediate normalization: the size of the first sized operand.
fn opSize(f: Form) u8 {
    for (f.ops) |k| {
        const s = regSize(k);
        if (s != .none) return @intFromEnum(s);
    }
    return 8;
}

fn matchKind(f: Form, k: Kind, o: Op) bool {
    switch (o) {
        .reg => |r| return switch (k) {
            .r8, .r16, .r32, .r64 => regSize(k) == r.size,
            .al, .ax, .eax, .rax => r.id == .rax and regSize(k) == r.size,
            .cl => r.id == .rcx and r.size == .b,
            .dx => r.id == .rdx and r.size == .w,
            else => false,
        },
        .mem => |m| return switch (k) {
            .m => true,
            .m8, .m16, .m32, .m64 => m.size == regSize(k),
            else => false,
        },
        .imm => |v| {
            const n = normImm(v, opSize(f));
            return switch (k) {
                .one => v == 1,
                .i8 => fits(v, i8) or fits(v, u8),
                .s8 => if (n) |x| fits(x, i8) else false,
                .u8 => fits(v, u8),
                .i16 => fits(v, i16) or fits(v, u16),
                .u16 => fits(v, u16),
                .i32 => fits(v, i32) or fits(v, u32),
                .s32 => fits(v, i32),
                .u32 => fits(v, u32),
                .i64 => true,
                else => false,
            };
        },
    }
}

fn matches(f: Form, ops: []const Op) bool {
    if (f.ops.len != ops.len) return false;
    for (f.ops, ops) |k, o| if (!matchKind(f, k, o)) return false;
    return true;
}

fn checkMem(m: Mem) Error!void {
    if (m.rip) {
        if (m.base != null or m.index != null) return error.InvalidOperand;
    } else {
        if (m.label != null) return error.InvalidOperand;
        if (m.index == .rsp) return error.InvalidOperand;
    }
}

pub const Buf = struct {
    b: [16]u8 = undefined,
    n: u8 = 0,
    fn u8_(s: *Buf, v: u8) void {
        s.b[s.n] = v;
        s.n += 1;
    }
    fn int(s: *Buf, v: i64, bytes: u8) void {
        var x: u64 = @bitCast(v);
        for (0..bytes) |_| {
            s.u8_(@truncate(x));
            x >>= 8;
        }
    }
};

fn rlo(g: Gp) u8 {
    return @intFromEnum(g) & 7;
}
fn rhi(g: Gp) u8 {
    return @intFromEnum(g) >> 3;
}

/// Encodes `ops` with form `f` into `out`. Returns the offset of a RIP disp32 within `out`, if any.
pub fn encodeForm(f: Form, ops: []const Op, out: *Buf) Error!?u8 {
    var rex: u8 = if (f.rexw) 0x48 else 0;
    var need_rex = false;
    for (ops) |o| switch (o) {
        .reg => |r| if (r.size == .b and @intFromEnum(r.id) >= 4 and @intFromEnum(r.id) < 8) {
            need_rex = true;
        },
        .mem => |m| try checkMem(m),
        else => {},
    };
    var reg_field: u8 = f.digit orelse 0;
    if (f.reg) |i| {
        const r = ops[i].reg.id;
        reg_field = rlo(r);
        rex |= rhi(r) << 2;
    }
    var rm_reg: ?Gp = null;
    var rm_mem: ?Mem = null;
    if (f.rm) |i| switch (ops[i]) {
        .reg => |r| {
            rm_reg = r.id;
            rex |= rhi(r.id);
        },
        .mem => |m| {
            rm_mem = m;
            if (m.index) |x| rex |= rhi(x) << 1;
            if (m.base) |b| rex |= rhi(b);
        },
        .imm => unreachable,
    };
    var plus: u8 = 0;
    if (f.opreg) |i| {
        const r = ops[i].reg.id;
        plus = rlo(r);
        rex |= rhi(r);
    }
    for (f.pre) |p| out.u8_(p);
    if (rex != 0 or need_rex) out.u8_(0x40 | rex);
    for (f.op, 0..) |b, k| out.u8_(if (k == f.op.len - 1) b + plus else b);
    var rip_at: ?u8 = null;
    if (f.modrm) {
        const r = (reg_field & 7) << 3;
        if (rm_reg) |g| {
            out.u8_(0xC0 | r | rlo(g));
        } else if (rm_mem) |m| {
            if (m.rip) {
                out.u8_(r | 5);
                rip_at = out.n;
                out.int(m.disp, 4);
            } else if (m.base) |base| {
                const mod: u8 = if (m.disp == 0 and rlo(base) != 5) 0x00 else if (fits(m.disp, i8)) 0x40 else 0x80;
                if (m.index != null or rlo(base) == 4) {
                    out.u8_(mod | r | 4);
                    out.u8_(@as(u8, @intFromEnum(m.scale)) << 6 | (if (m.index) |x| rlo(x) else 4) << 3 | rlo(base));
                } else out.u8_(mod | r | rlo(base));
                if (mod == 0x40) out.int(m.disp, 1) else if (mod == 0x80) out.int(m.disp, 4);
            } else {
                out.u8_(r | 4);
                out.u8_(@as(u8, @intFromEnum(m.scale)) << 6 | (if (m.index) |x| rlo(x) else 4) << 3 | 5);
                out.int(m.disp, 4);
            }
        }
    }
    if (f.imm != 0) for (f.ops, ops) |k, o| switch (k) {
        .i8, .s8, .u8, .i16, .u16, .i32, .s32, .u32, .i64 => out.int(o.imm, f.imm),
        else => {},
    };
    return rip_at;
}

/// Emits `mn` with operands `args` (tuple of Reg / Mem / integers) into `buf`.
pub fn emit(buf: *CodeBuffer, comptime mn: Mnemonic, args: anytype) Error!void {
    var ops: [args.len]Op = undefined;
    inline for (args, 0..) |a, i| ops[i] = toOp(a);
    return emitOps(buf, &@field(tbl.table, @tagName(mn)), &ops);
}

pub fn emitOps(buf: *CodeBuffer, forms: []const Form, ops: []const Op) Error!void {
    var best: ?Buf = null;
    var best_rip: ?u8 = null;
    for (forms) |f| {
        if (!matches(f, ops)) continue;
        var b: Buf = .{};
        const r = try encodeForm(f, ops, &b);
        if (best == null or b.n < best.?.n) {
            best = b;
            best_rip = r;
        }
    }
    const b = best orelse return error.NoMatchingForm;
    try buf.ensure(16);
    const start = buf.offset();
    for (b.b[0..b.n]) |x| buf.put8(x);
    if (best_rip) |at| for (ops) |o| switch (o) {
        .mem => |m| if (m.label) |l| {
            if (!buf.isValidLabel(l)) {
                buf.len = start;
                return error.InvalidLabel;
            }
            try buf.addFixup(.{ .at = start + at, .base = start + b.n, .label = l.id, .kind = .rel32 });
        },
        else => {},
    };
}
