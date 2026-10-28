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

/// `high`: AH/CH/DH/BH (id must be rax..rbx, size .b); cannot be used with a REX prefix.
pub const Reg = struct { id: Gp, size: Size, high: bool = false };

/// Register number in ModRM/opcode encoding (AH..BH are 4..7 without REX).
fn num(r: Reg) u8 {
    return if (r.high) @as(u8, @intFromEnum(r.id)) + 4 else @intFromEnum(r.id);
}

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
/// High 8-bit register: `r8hi(.rax)` = AH, `.rcx` = CH, `.rdx` = DH, `.rbx` = BH.
pub inline fn r8hi(g: Gp) Reg {
    return .{ .id = g, .size = .b, .high = true };
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
            .al, .ax, .eax, .rax => r.id == .rax and !r.high and regSize(k) == r.size,
            .cl => r.id == .rcx and !r.high and r.size == .b,
            .dx => r.id == .rdx and r.size == .w,
            else => false,
        },
        .mem => |m| return switch (k) {
            .m => true,
            // Unsized memory (AsmJit `ptr(...)`) takes the size the form needs; the shortest
            // encoding wins among sizes, so e.g. `ptwrite [m]` picks m32 like AsmJit.
            .m8, .m16, .m32, .m64 => m.size == .none or m.size == regSize(k),
            else => false,
        },
        .imm => |v| {
            const n = normImm(v, opSize(f));
            return switch (k) {
                .one => v == 1,
                // `push imm8` (6A) sign-extends to 64 bits, so 128..255 must take `push imm32`.
                .i8 => if (f.op.len == 1 and f.op[0] == 0x6a) fits(v, i8) else fits(v, i8) or fits(v, u8),
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
    // `xchg eax, eax` via 90+r would be `nop`, which does not zero-extend into rax: use 87 /r.
    if (f.opreg != null and f.op.len == 1 and f.op[0] == 0x90 and !f.rexw and f.pre.len == 0 and
        ops[0] == .reg and ops[1] == .reg and ops[0].reg.id == .rax and ops[1].reg.id == .rax) return false;
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
    pub fn u8_(s: *Buf, v: u8) void {
        s.b[s.n] = v;
        s.n += 1;
    }
    pub fn int(s: *Buf, v: i64, bytes: u8) void {
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
    var has_high = false;
    for (ops) |o| switch (o) {
        .reg => |r| if (r.high) {
            if (@intFromEnum(r.id) >= 4 or r.size != .b) return error.InvalidOperand;
            has_high = true;
        } else if (r.size == .b and @intFromEnum(r.id) >= 4 and @intFromEnum(r.id) < 8) {
            need_rex = true;
        },
        .mem => |m| try checkMem(m),
        else => {},
    };
    var reg_field: u8 = f.digit orelse 0;
    if (f.reg) |i| {
        const r = num(ops[i].reg);
        reg_field = r & 7;
        rex |= (r >> 3) << 2;
    }
    var rm_reg: ?u8 = null;
    var rm_mem: ?Mem = null;
    if (f.rm) |i| switch (ops[i]) {
        .reg => |r| {
            rm_reg = num(r);
            rex |= num(r) >> 3;
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
        const r = num(ops[i].reg);
        plus = r & 7;
        rex |= r >> 3;
    }
    // AH..BH are only addressable without REX.
    if (cur_opt.rex) need_rex = true;
    if (has_high and (rex != 0 or need_rex)) return error.InvalidOperand;
    for (f.pre) |p| out.u8_(p);
    if (rex != 0 or need_rex) out.u8_(0x40 | rex);
    for (f.op, 0..) |b, k| out.u8_(if (k == f.op.len - 1) b + plus else b);
    var rip_at: ?u8 = null;
    if (f.modrm) {
        const r = (reg_field & 7) << 3;
        if (rm_reg) |g| {
            out.u8_(0xC0 | r | (g & 7));
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

/// Per-instruction options (AsmJit `lock()`, `rex()`, `mod_mr()` / `mod_rm()`, `long_()`).
pub const Opt = struct {
    lock: bool = false,
    /// Always emit a REX prefix (0x40 if no bit is needed).
    rex: bool = false,
    /// reg,reg encoding direction: `.mr` (first operand in ModRM.rm) or `.rm`.
    mod: enum { auto, mr, rm } = .auto,
    /// `mov r64, imm`: always the 10-byte imm64 form.
    long: bool = false,
};
threadlocal var cur_opt: Opt = .{};

/// `emit` with options.
pub fn emitOpt(buf: *CodeBuffer, comptime mn: Mnemonic, args: anytype, opt: Opt) Error!void {
    cur_opt = opt;
    defer cur_opt = .{};
    return emit(buf, mn, args);
}

/// `mov` between the accumulator and a 64-bit absolute address (A0..A3 moffs64).
pub fn movMoffs(buf: *CodeBuffer, store: bool, acc: Reg, addr: u64) Error!void {
    if (acc.id != .rax or acc.high) return error.InvalidOperand;
    try buf.ensure(16);
    if (acc.size == .w) buf.put8(0x66);
    if (acc.size == .q) buf.put8(0x48);
    buf.put8(@as(u8, if (acc.size == .b) 0xA0 else 0xA1) + @as(u8, if (store) 2 else 0));
    var a = addr;
    for (0..8) |_| {
        buf.put8(@truncate(a));
        a >>= 8;
    }
}

/// Emits `mn` with operands `args` (tuple of Reg / Mem / integers) into `buf`.
pub fn emit(buf: *CodeBuffer, comptime mn: Mnemonic, args: anytype) Error!void {
    var ops: [args.len]Op = undefined;
    inline for (args, 0..) |a, i| ops[i] = toOp(a);
    return emitOps(buf, &@field(tbl.table, @tagName(mn)), &ops);
}

pub fn emitOps(buf: *CodeBuffer, forms: []const Form, ops: []const Op) Error!void {
    return emitExact(buf, forms, ops) catch |e| switch (e) {
        error.NoMatchingForm => emitAliased(buf, forms, ops) orelse e,
        else => e,
    };
}

fn isImplicit(o: Op) bool {
    return switch (o) {
        .reg => |r| if (r.high) r.id == .rax else (r.id == .rax or r.id == .rcx or r.id == .rdx or r.id == .rbx),
        // `[rax]` / `[rdi]` style implicit memory (clzero, monitor).
        .mem => |m| m.base != null and m.index == null and m.disp == 0 and !m.rip,
        .imm => false,
    };
}

fn withSize(ops: []const Op, s: Size, out: []Op) []const Op {
    for (ops, 0..) |o, i| out[i] = switch (o) {
        .mem => |m| if (m.size == .none) blk: {
            var x = m;
            x.size = s;
            break :blk .{ .mem = x };
        } else o,
        else => o,
    };
    return out[0..ops.len];
}

/// AsmJit spellings that ZigJIT's table writes differently:
/// - explicit implicit operands: `mul(rdx, rax, rcx)`, `div(ax, cl)`, `cqo(rdx, rax)`,
///   `rdmsr(edx, eax, ecx)`, `xsaves(m, edx, eax)`, `monitor(ptr(rax), ecx, edx)`;
///   the implicit `dx:ax` pair / `ax` also fixes the size of an unsized memory operand;
/// - `imul(r, imm)` -> `imul(r, r, imm)`.
fn emitAliased(buf: *CodeBuffer, forms: []const Form, ops: []const Op) ?Error!void {
    var tmp: [8]Op = undefined;
    if (ops.len > tmp.len) return null;
    if (ops.len == 2 and ops[0] == .reg and ops[1] == .imm) {
        const three = [_]Op{ ops[0], ops[0], ops[1] };
        if (emitExact(buf, forms, &three)) |_| return {} else |_| {}
    }
    // dx:ax pair (mul/div/imul one-operand forms): size comes from the pair.
    if (ops.len == 3 and ops[0] == .reg and ops[1] == .reg and ops[0].reg.id == .rdx and
        ops[1].reg.id == .rax and ops[0].reg.size == ops[1].reg.size)
        if (emitExact(buf, forms, withSize(ops[2..], ops[0].reg.size, &tmp))) |_| return {} else |_| {};
    // ax = al * r/m8.
    if (ops.len == 2 and ops[0] == .reg and ops[0].reg.id == .rax and ops[0].reg.size == .w)
        if (emitExact(buf, forms, withSize(ops[1..], .b, &tmp))) |_| return {} else |_| {};
    // Otherwise drop implicit operands from either end.
    var lo: usize = 0;
    while (lo <= ops.len) : (lo += 1) {
        var hi: usize = ops.len;
        while (hi >= lo) : (hi -= 1) {
            if (lo != 0 or hi != ops.len) {
                var ok = true;
                for (ops[0..lo]) |o| ok = ok and isImplicit(o);
                for (ops[hi..]) |o| ok = ok and isImplicit(o);
                if (ok) if (emitExact(buf, forms, ops[lo..hi])) |_| return {} else |_| {};
            }
            if (hi == lo) break;
        }
        if (lo < ops.len and !isImplicit(ops[lo])) break;
    }
    return null;
}

fn emitExact(buf: *CodeBuffer, forms: []const Form, ops: []const Op) Error!void {
    var best: ?Buf = null;
    var best_rip: ?u8 = null;
    var best_imm: u8 = 0;
    for (forms) |f| {
        if (!matches(f, ops)) continue;
        const o = cur_opt;
        if (o.mod == .mr and f.reg != null and f.reg.? == 0) continue;
        if (o.mod == .rm and f.rm != null and f.rm.? == 0 and f.reg != null) continue;
        if (o.long and !(f.imm == 8)) continue;
        var b: Buf = .{};
        const r = try encodeForm(f, ops, &b);
        // Shortest wins; on a tie prefer the narrower immediate (`66 83 /r ib` over `66 05 iw`), like LLVM.
        if (best == null or b.n < best.?.n or (b.n == best.?.n and f.imm < best_imm)) {
            best = b;
            best_rip = r;
            best_imm = f.imm;
        }
    }
    var b = best orelse return error.NoMatchingForm;
    // `xchg rax, rax` (REX.W 90) is a no-op; AsmJit and LLVM emit plain `90`.
    if (b.n == 2 and b.b[0] == 0x48 and b.b[1] == 0x90) {
        b.b[0] = 0x90;
        b.n = 1;
    }
    try buf.ensure(17);
    if (cur_opt.lock) buf.put8(0xF0);
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
