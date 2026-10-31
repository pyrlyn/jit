//! x86-64 Assembler: direct emission into a CodeBuffer, no IR, no dispatch tables.
//! Encodings follow the Intel SDM. Where AsmJit (with kOptimizeForSize) picks a shorter
//! form, e.g. `mov r64, u32` => `mov r32, imm32`, we pick the same one, so outputs are
//! byte-identical (see bench/).
//!
//! Hot path: each instruction is assembled into a 16-byte stack scratch (`Enc`, kept in
//! registers by LLVM) and committed with one unaligned 16-byte store + one length update.
const std = @import("std");
const core = @import("../core/code_buffer.zig");
const CodeBuffer = core.CodeBuffer;
pub const Label = core.Label;
pub const Error = core.Error || error{ InvalidOperand, InvalidLabel, UnsupportedCpuFeature, EncodingMismatch };
/// Generated from LLVM's X86.td (tools/gen_isa.zig, `zig build gen-tables`).
pub const isa = @import("isa.zig");
/// Host CPU feature detection.
pub const cpu = @import("cpu.zig");
/// Table-driven encoder for all generated GP forms (see `AssemblerT.inst`).
pub const gp = @import("gp.zig");
/// Table-driven SIMD/VEX/EVEX/x87/... encoder (all non-GP forms of AsmJit's ISA database).
pub const vec = @import("vec.zig");
const safety = @import("../core/safety.zig");
const build_opts = @import("pyrlyn_jit_options");
pub const Options = safety.Options;

pub const Gp = enum(u4) {
    rax,
    rcx,
    rdx,
    rbx,
    rsp,
    rbp,
    rsi,
    rdi,
    r8,
    r9,
    r10,
    r11,
    r12,
    r13,
    r14,
    r15,
    inline fn lo(r: Gp) u8 {
        return @backingInt(r) & 7;
    }
    inline fn hi(r: Gp) u8 {
        return @backingInt(r) >> 3;
    }
};

pub const Scale = enum(u2) { x1, x2, x4, x8 };

/// [base + index*scale + disp]. `base == null` => [index*scale + disp32] / absolute disp32.
pub const Mem = struct {
    base: ?Gp = null,
    index: ?Gp = null,
    scale: Scale = .x1,
    disp: i32 = 0,
};

pub inline fn ptr(base: Gp, disp: i32) Mem {
    return .{ .base = base, .disp = disp };
}
pub inline fn ptrIdx(base: Gp, index: Gp, scale: Scale, disp: i32) Mem {
    return .{ .base = base, .index = index, .scale = scale, .disp = disp };
}

pub const Cond = enum(u4) { o, no, b, ae, e, ne, be, a, s, ns, p, np, l, ge, le, g };

/// Classic ALU group: `0x83/0x81 /digit` for imm forms, `op*8+1` (r/m,r), `op*8+3` (r,r/m).
pub const Alu = enum(u3) { add = 0, @"or" = 1, adc = 2, sbb = 3, @"and" = 4, sub = 5, xor = 6, cmp = 7 };

const REX_W: u8 = 0x48;
const max_inst = 16;

inline fn fitsI8(v: i64) bool {
    return v >= -128 and v <= 127;
}
inline fn fitsI32(v: i64) bool {
    return v >= std.math.minInt(i32) and v <= std.math.maxInt(i32);
}

/// One instruction under construction.
const Enc = struct {
    /// Little-endian accumulator: byte k of the instruction lives in bits [8k, 8k+8).
    /// Kept as an integer so LLVM holds it in registers (writing bytes into a stack array
    /// and reloading it as 16 bytes caused store-forwarding stalls).
    lo: u64 = 0,
    hi: u64 = 0,
    n: u8 = 0,

    inline fn put(e: *Enc, v: u64, comptime size: u8) void {
        const sh: u7 = @intCast(@as(u32, e.n) * 8);
        if (e.n < 8) {
            e.lo |= v << @intCast(sh);
            if (size > 1 and e.n + size > 8) e.hi |= v >> @intCast(64 - @as(u32, sh));
        } else e.hi |= v << @intCast(sh - 64);
        e.n += size;
    }
    inline fn u8_(e: *Enc, v: u8) void {
        e.put(v, 1);
    }
    inline fn u32_(e: *Enc, v: u32) void {
        e.put(v, 4);
    }
    inline fn u64_(e: *Enc, v: u64) void {
        e.put(v, 8);
    }
    inline fn i8_(e: *Enc, v: i64) void {
        e.u8_(@bitCast(@as(i8, @intCast(v))));
    }
    inline fn i32_(e: *Enc, v: i64) void {
        e.u32_(@bitCast(@as(i32, @intCast(v))));
    }

    inline fn rex(e: *Enc, w: bool, r: u8, x: u8, b: u8) void {
        const v: u8 = (if (w) REX_W else 0x40) | (r >> 3) << 2 | x << 1 | b;
        if (v != 0x40) e.u8_(v);
    }
    inline fn rexRR(e: *Enc, w: bool, reg: u8, rm: Gp) void {
        e.rex(w, reg, 0, rm.hi());
    }
    inline fn rexRM(e: *Enc, w: bool, reg: u8, m: Mem) void {
        e.rex(w, reg, if (m.index) |i| i.hi() else 0, if (m.base) |r| r.hi() else 0);
    }
    inline fn modrmRR(e: *Enc, reg: u8, rm: Gp) void {
        e.u8_(0xC0 | (reg & 7) << 3 | rm.lo());
    }

    /// ModRM [+SIB] [+disp8/disp32] for a memory operand; `reg` is the /r or /digit field.
    inline fn modrmM(e: *Enc, reg: u8, m: Mem) void {
        const r: u8 = (reg & 7) << 3;
        const base = m.base orelse {
            const idx: u8 = if (m.index) |i| i.lo() else 4;
            e.u8_(0x04 | r);
            e.u8_(@as(u8, @backingInt(m.scale)) << 6 | idx << 3 | 5);
            e.u32_(@bitCast(m.disp));
            return;
        };
        // rbp/r13 as base cannot use mod=00 (that slot means disp32 / RIP-relative).
        const mod: u8 = if (m.disp == 0 and base.lo() != 5) 0x00 else if (fitsI8(m.disp)) 0x40 else 0x80;
        if (m.index != null or base.lo() == 4) {
            const idx: u8 = if (m.index) |i| i.lo() else 4;
            e.u8_(mod | r | 4);
            e.u8_(@as(u8, @backingInt(m.scale)) << 6 | idx << 3 | base.lo());
        } else e.u8_(mod | r | base.lo());
        switch (mod) {
            0x40 => e.i8_(m.disp),
            0x80 => e.u32_(@bitCast(m.disp)),
            else => {},
        }
    }
};

/// Instruction id seen by validation and hooks (one per public emitter).
pub const Inst = enum { mov_rr, mov_rm, mov_mr, mov_mi, lea, mov_ri, movabs, alu_rr, alu_rm, alu_mr, alu_ri, test_rr, imul_rr, imul_rm, imul_rri, push, pop, ret, nop, call_r, jmp_r, jmp, call, jcc, gp };

/// Operands of one instruction, as passed to validation and hooks. Unused fields are null.
/// `r0` is the first register operand in Intel order (destination when it is a register).
pub const Ops = struct {
    r0: ?Gp = null,
    r1: ?Gp = null,
    mem: ?Mem = null,
    imm: ?i64 = null,
    label: ?Label = null,
    alu: ?Alu = null,
    cond: ?Cond = null,
    /// `.gp` (the generated `inst` path): the mnemonic; operands are not decoded into `Ops`.
    mn: ?[]const u8 = null,
};
pub const PreEvent = safety.PreEvent(Inst, Ops);
pub const Event = safety.Event(Inst, Ops);
pub const CountingHooks = safety.CountingHooks;
pub const LogHooks = safety.LogHooks;
pub const CallbackHooks = safety.CallbackHooks(Event);

/// Logger hook: an Intel-syntax text listing with offsets and bytes, e.g.
/// `0000  48 89 c8              mov rax, rcx`. Use as `Options.Hooks = Logger(cap)` or forward
/// to it from a `CallbackHooks` callback. Text is truncated (flag set) when `cap` is exceeded.
pub fn Logger(comptime cap: usize) type {
    return struct {
        const Self = @This();
        buf: [cap]u8 = undefined,
        len: usize = 0,
        truncated: bool = false,
        pub fn post(self: *Self, ev: Event) void {
            self.print("{x:0>4} ", .{ev.offset});
            for (ev.bytes) |b| self.print(" {x:0>2}", .{b});
            var n = ev.bytes.len * 3;
            while (n < 22) : (n += 1) self.print(" ", .{});
            self.print(" {s}", .{mnemonic(ev.id, ev.ops)});
            var first = true;
            if (ev.ops.r0) |r| self.operand(&first, "{s}", .{@tagName(r)});
            if (ev.ops.mem) |m| {
                self.operand(&first, "[", .{});
                if (m.base) |b| self.print("{s}", .{@tagName(b)});
                if (m.index) |i| self.print("{s}{s}*{d}", .{ if (m.base != null) "+" else "", @tagName(i), @as(u8, 1) << @backingInt(m.scale) });
                if (m.disp != 0 or (m.base == null and m.index == null)) self.print("{s}{d}", .{ if (m.disp >= 0 and (m.base != null or m.index != null)) "+" else "", m.disp });
                self.print("]", .{});
            }
            if (ev.ops.r1) |r| self.operand(&first, "{s}", .{@tagName(r)});
            if (ev.ops.imm) |v| self.operand(&first, "{d}", .{v});
            if (ev.ops.label) |l| self.operand(&first, "L{d}", .{l.id});
            self.print("\n", .{});
        }
        fn operand(self: *Self, first: *bool, comptime fmt: []const u8, args: anytype) void {
            self.print("{s}", .{if (first.*) " " else ", "});
            first.* = false;
            self.print(fmt, args);
        }
        fn print(self: *Self, comptime fmt: []const u8, args: anytype) void {
            const out = std.fmt.bufPrint(self.buf[self.len..], fmt, args) catch {
                self.truncated = true;
                return;
            };
            self.len += out.len;
        }
        pub fn text(self: *const Self) []const u8 {
            return self.buf[0..self.len];
        }
    };
}

fn mnemonic(id: Inst, ops: Ops) []const u8 {
    return switch (id) {
        .mov_rr, .mov_rm, .mov_mr, .mov_mi, .mov_ri => "mov",
        .alu_rr, .alu_rm, .alu_mr, .alu_ri => @tagName(ops.alu.?),
        .test_rr => "test",
        .imul_rr, .imul_rm, .imul_rri => "imul",
        .call_r => "call",
        .jmp_r => "jmp",
        .jcc => switch (ops.cond.?) {
            inline else => |c| "j" ++ @tagName(c),
        },
        .gp => ops.mn orelse "?",
        else => @tagName(id),
    };
}

/// Options from `-Dvalidate` / `-Dhooks` (hooks => runtime `CallbackHooks`).
pub const default_options: Options = .{
    .validate = build_opts.validate,
    .Hooks = if (build_opts.hooks) CallbackHooks else void,
};
/// The default assembler, configured by build options.
pub const Assembler = AssemblerT(default_options);

/// Checks a memory operand: rsp cannot be an index (its SIB slot means "no index"),
/// and a scale without an index is meaningless.
/// Calling conventions: integer argument/return registers and callee-saved registers.
pub const CallConv = struct {
    args: []const Gp,
    ret: Gp = .rax,
    callee_saved: []const Gp,
    /// Bytes the caller reserves above the return address (Win64 home space).
    shadow: u32 = 0,

    pub const sysv: CallConv = .{ .args = &.{ .rdi, .rsi, .rdx, .rcx, .r8, .r9 }, .callee_saved = &.{ .rbx, .rbp, .r12, .r13, .r14, .r15 } };
    pub const win64: CallConv = .{ .args = &.{ .rcx, .rdx, .r8, .r9 }, .callee_saved = &.{ .rbx, .rbp, .rdi, .rsi, .r12, .r13, .r14, .r15 }, .shadow = 32 };
    /// [rbp + off] of integer argument `i` (`i >= args.len`) after `prolog`: return address,
    /// saved rbp, then the shadow space (Win64), then 8-byte stack slots. Same formula for SysV and Win64.
    pub fn stackArg(cc: CallConv, i: usize) Mem {
        std.debug.assert(i >= cc.args.len);
        return .{ .base = .rbp, .disp = @intCast(16 + cc.shadow + 8 * (i - cc.args.len)) };
    }
    pub const host: CallConv = if (@import("builtin").os.tag == .windows) win64 else sysv;
};

/// Recommended NOP sequences of length 1..9 (Intel SDM, Vol. 2B "NOP").
const nops = [9][9]u8{
    .{ 0x90, 0, 0, 0, 0, 0, 0, 0, 0 },
    .{ 0x66, 0x90, 0, 0, 0, 0, 0, 0, 0 },
    .{ 0x0F, 0x1F, 0x00, 0, 0, 0, 0, 0, 0 },
    .{ 0x0F, 0x1F, 0x40, 0x00, 0, 0, 0, 0, 0 },
    .{ 0x0F, 0x1F, 0x44, 0x00, 0x00, 0, 0, 0, 0 },
    .{ 0x66, 0x0F, 0x1F, 0x44, 0x00, 0x00, 0, 0, 0 },
    .{ 0x0F, 0x1F, 0x80, 0x00, 0x00, 0x00, 0x00, 0, 0 },
    .{ 0x0F, 0x1F, 0x84, 0x00, 0x00, 0x00, 0x00, 0x00, 0 },
    .{ 0x66, 0x0F, 0x1F, 0x84, 0x00, 0x00, 0x00, 0x00, 0x00 },
};

fn padding(off: u32, boundary: u32) u32 {
    std.debug.assert(std.math.isPowerOfTwo(boundary));
    return (boundary - (off & (boundary - 1))) & (boundary - 1);
}

pub fn validateMem(m: Mem) Error!void {
    if (m.index) |i| {
        if (i == .rsp) return error.InvalidOperand;
    } else if (m.scale != .x1) return error.InvalidOperand;
}

/// AsmJit forms for one emitted instruction (ALU ops and conditions select a sub-table).
pub fn formsFor(comptime id: Inst, ops: Ops) []const isa.Form {
    return switch (id) {
        .alu_rr, .alu_rm, .alu_mr, .alu_ri => @field(isa.table, @tagName(id))[@backingInt(ops.alu.?)],
        .jcc => isa.table.jcc[@backingInt(ops.cond.?)],
        else => &@field(isa.table, @tagName(id)),
    };
}

/// Union of the CPU features required by any form of `id` (comptime).
pub fn requiredFeatures(comptime id: Inst) isa.Features {
    const set = comptime blk: {
        @setEvalBranchQuota(100_000);
        var list: []const isa.Feature = &.{};
        const t = @field(isa.table, @tagName(id));
        switch (@typeInfo(@TypeOf(t)).array.child) {
            isa.Form => for (t) |f| {
                list = list ++ f.features;
            },
            else => for (t) |fs| for (fs) |f| {
                list = list ++ f.features;
            },
        }
        break :blk isa.featureSet(list);
    };
    return set;
}

/// Checks `need` against `have`; `error.UnsupportedCpuFeature` if anything is missing.
pub fn checkFeatureSet(need: isa.Features, have: isa.Features) Error!void {
    if (!need.subsetOf(have)) return error.UnsupportedCpuFeature;
}

fn immSize(k: isa.Imm) usize {
    return switch (k) {
        .none => 0,
        .imm8, .rel8 => 1,
        .imm32, .rel32 => 4,
        .imm64 => 8,
    };
}

/// True if `bytes` is an encoding of form `f`: opcode bytes (after REX), `+r` register
/// in the low 3 bits, ModRM `/digit`, and the immediate size for forms without ModRM.
pub fn matchForm(f: isa.Form, bytes: []const u8) bool {
    var i: usize = 0;
    while (i < bytes.len and bytes[i] & 0xF0 == 0x40) i += 1; // REX
    if (bytes.len < i + f.opcode.len) return false;
    for (f.opcode, 0..) |want, j| {
        const got = bytes[i + j];
        const last = j + 1 == f.opcode.len;
        if (if (last and f.plus == .reg) got & 0xF8 != want else got != want) return false;
    }
    i += f.opcode.len;
    if (f.modrm) {
        if (i >= bytes.len) return false;
        if (f.digit) |d| if ((bytes[i] >> 3) & 7 != d) return false;
        return bytes.len >= i + 1 + immSize(f.imm);
    }
    return bytes.len == i + immSize(f.imm);
}

pub fn AssemblerT(comptime opts: Options) type {
    return struct {
        const Self = @This();
        pub const options = opts;
        pub const Hooks = opts.Hooks;
        const has_hooks = Hooks != void;
        /// True when any safety layer is compiled in; when false, emitters call the raw
        /// encoders directly and no `Ops` value is ever built.
        const checked = opts.validate or has_hooks;

        buf: *CodeBuffer,
        /// Hook instance; null = no hook attached (only exists when `Hooks != void`).
        hooks: if (has_hooks) ?*Hooks else void = if (has_hooks) null else {},

        pub fn init(buf: *CodeBuffer) Self {
            return .{ .buf = buf };
        }
        pub fn initHooks(buf: *CodeBuffer, h: *Hooks) Self {
            return .{ .buf = buf, .hooks = h };
        }
        pub inline fn newLabel(a: Self) Error!Label {
            return a.buf.newLabel();
        }
        pub inline fn bind(a: Self, l: Label) Error!void {
            if (comptime opts.validate) try a.checkLabel(l);
            return a.buf.bind(l);
        }

        inline fn checkLabel(a: Self, l: Label) Error!void {
            if (l.id >= a.buf.labels.items.len) return error.InvalidLabel;
        }

        /// Operand check before encoding (only compiled when `opts.validate`).
        pub inline fn validate(a: Self, comptime id: Inst, ops: Ops) Error!void {
            if (ops.mem) |m| try validateMem(m);
            if (ops.label) |l| try a.checkLabel(l);
            try checkCpu(id);
        }

        /// CPU features required by `id` (from the AsmJit database) vs the configured target.
        inline fn checkCpu(comptime id: Inst) Error!void {
            comptime checkDeny(requiredFeatures(id), @tagName(id));
            switch (opts.cpu_check) {
                .none => {},
                .target => comptime {
                    const have = safety.featureSet(isa.Feature, opts.target_features);
                    if (!requiredFeatures(id).subsetOf(have)) @compileError("instruction " ++ @tagName(id) ++ " needs CPU features missing from target_features");
                },
                .host => try checkFeatureSet(requiredFeatures(id), cpu.host() orelse return error.UnsupportedCpuFeature),
            }
        }

        fn checkDeny(comptime need: isa.Features, comptime name: []const u8) void {
            comptime {
                const deny = safety.featureSet(isa.Feature, opts.deny_features);
                if (need.intersectWith(deny).count() != 0) @compileError("instruction " ++ name ++ " needs a CPU feature in deny_features");
            }
        }

        /// Any GP instruction form generated from AsmJit's database: 8/16/32/64-bit registers,
        /// sized memory (`gp.byte/word/dword/qword`), RIP-relative (`gp.rip`, `gp.ripLabel`) and
        /// immediates. Example: `a.inst(.movzx, .{ gp.r32(.rax), gp.byte(ptr(.rdi, 0)) })`.
        /// CPU checks: deny list and `.target` at compile time, `.host` at runtime (validate).
        /// Hooks see it as `Inst.gp` with `Ops.mn` set (operands are not decoded into `Ops`).
        pub fn inst(a: Self, comptime mn: gp.Mnemonic, args: anytype) (Error || gp.Error)!void {
            const need = comptime gp.requiredFeatures(mn);
            comptime checkDeny(need, @tagName(mn));
            if (comptime opts.validate) switch (opts.cpu_check) {
                .none => {},
                .target => comptime {
                    if (!need.subsetOf(safety.featureSet(isa.Feature, opts.target_features)))
                        @compileError("instruction " ++ @tagName(mn) ++ " needs CPU features missing from target_features");
                },
                .host => try checkFeatureSet(need, cpu.host() orelse return error.UnsupportedCpuFeature),
            };
            if (comptime !has_hooks) return gp.emit(a.buf, mn, args);
            const h = a.hooks orelse return gp.emit(a.buf, mn, args);
            const ops: Ops = .{ .mn = @tagName(mn) };
            const start: u32 = @intCast(a.buf.len);
            if (comptime @hasDecl(Hooks, "pre")) try h.pre(.{ .id = .gp, .ops = ops, .offset = start });
            try gp.emit(a.buf, mn, args);
            if (comptime @hasDecl(Hooks, "post")) h.post(.{ .id = .gp, .ops = ops, .offset = start, .bytes = a.buf.data[start..a.buf.len] });
        }

        /// Raw encoder + (validate mode) check of the emitted bytes against the AsmJit forms.
        /// On mismatch the instruction is removed and `EncodingMismatch` returned.
        inline fn encode(a: Self, comptime id: Inst, ops: Ops, comptime raw: anytype, args: anytype) Error!void {
            if (comptime !opts.validate) return @call(.always_inline, raw, .{a} ++ args);
            const start = a.buf.len;
            try @call(.always_inline, raw, .{a} ++ args);
            const bytes = a.buf.data[start..a.buf.len];
            for (formsFor(id, ops)) |f| {
                if (matchForm(f, bytes)) return;
            }
            a.buf.len = start;
            return error.EncodingMismatch;
        }

        /// Validation + hooks around a raw encoder. With both layers off this is just the call.
        /// False when no layer can act on this call: both layers compiled out, or hooks
        /// compiled in without an attached instance (one null check, no `Ops` built).
        inline fn active(a: Self) bool {
            if (comptime opts.validate) return true;
            if (comptime !has_hooks) return false;
            return a.hooks != null;
        }

        inline fn emit(a: Self, comptime id: Inst, ops: Ops, comptime raw: anytype, args: anytype) Error!void {
            if (comptime opts.validate) try a.validate(id, ops);
            if (comptime !has_hooks) return a.encode(id, ops, raw, args);
            if (a.hooks) |h| {
                @branchHint(.unlikely);
                return @call(.always_inline, emitHooked, .{ a, h, id, ops, raw, args });
            }
            return a.encode(id, ops, raw, args);
        }

        fn emitHooked(a: Self, h: *Hooks, comptime id: Inst, ops: Ops, comptime raw: anytype, args: anytype) Error!void {
            const start: u32 = @intCast(a.buf.len);
            if (comptime @hasDecl(Hooks, "pre")) try h.pre(.{ .id = id, .ops = ops, .offset = start });
            try a.encode(id, ops, raw, args);
            if (comptime @hasDecl(Hooks, "post")) h.post(.{ .id = id, .ops = ops, .offset = start, .bytes = a.buf.data[start..a.buf.len] });
        }

        /// Writes the scratch with one 16-byte store (capacity was ensured) and bumps len.
        inline fn commit(a: Self, e: *const Enc) void {
            const cb = a.buf;
            const p = cb.data[cb.len..][0..max_inst];
            std.mem.writeInt(u64, p[0..8], e.lo, .little);
            std.mem.writeInt(u64, p[8..16], e.hi, .little);
            cb.len += e.n;
        }

        inline fn opRR(a: Self, comptime op: u8, reg: Gp, rm: Gp) Error!void {
            try a.buf.ensure(max_inst);
            var e: Enc = .{};
            e.rexRR(true, @backingInt(reg), rm);
            e.u8_(op);
            e.modrmRR(@backingInt(reg), rm);
            a.commit(&e);
        }

        inline fn opRM(a: Self, comptime op: u8, reg: u8, m: Mem, comptime imm32: ?void, imm: i32) Error!void {
            try a.buf.ensure(max_inst);
            var e: Enc = .{};
            e.rexRM(true, reg, m);
            e.u8_(op);
            e.modrmM(reg, m);
            if (imm32 != null) e.u32_(@bitCast(imm));
            a.commit(&e);
        }

        // ---- mov --------------------------------------------------------------------

        inline fn rawMovRR(a: Self, dst: Gp, src: Gp) Error!void {
            return a.opRR(0x89, src, dst);
        }
        inline fn rawMovRM(a: Self, dst: Gp, src: Mem) Error!void {
            return a.opRM(0x8B, @backingInt(dst), src, null, 0);
        }
        inline fn rawMovMR(a: Self, dst: Mem, src: Gp) Error!void {
            return a.opRM(0x89, @backingInt(src), dst, null, 0);
        }
        /// mov qword [m], imm32 (sign-extended)
        inline fn rawMovMI(a: Self, dst: Mem, imm: i32) Error!void {
            return a.opRM(0xC7, 0, dst, {}, imm);
        }
        inline fn rawLea(a: Self, dst: Gp, src: Mem) Error!void {
            return a.opRM(0x8D, @backingInt(dst), src, null, 0);
        }

        /// Shortest encoding: u32 => `mov r32, imm32` (zero-extends), i32 => `mov r/m64, imm32`
        /// (sign-extends), otherwise `movabs r64, imm64`.
        inline fn rawMovRI(a: Self, dst: Gp, imm: i64) Error!void {
            try a.buf.ensure(max_inst);
            var e: Enc = .{};
            if (imm >= 0 and imm <= std.math.maxInt(u32)) {
                if (dst.hi() != 0) e.u8_(0x41);
                e.u8_(0xB8 + dst.lo());
                e.u32_(@intCast(imm));
            } else if (fitsI32(imm)) {
                e.rexRR(true, 0, dst);
                e.u8_(0xC7);
                e.modrmRR(0, dst);
                e.i32_(imm);
            } else {
                e.u8_(REX_W | dst.hi());
                e.u8_(0xB8 + dst.lo());
                e.u64_(@bitCast(imm));
            }
            a.commit(&e);
        }

        /// Always the 10-byte `movabs r64, imm64` (patchable constant slot).
        inline fn rawMovabs(a: Self, dst: Gp, imm: u64) Error!void {
            try a.buf.ensure(max_inst);
            var e: Enc = .{};
            e.u8_(REX_W | dst.hi());
            e.u8_(0xB8 + dst.lo());
            e.u64_(imm);
            a.commit(&e);
        }

        // ---- ALU ----------------------------------------------------------------------

        inline fn rawAluRR(a: Self, comptime op: Alu, dst: Gp, src: Gp) Error!void {
            return a.opRR(@as(u8, @backingInt(op)) * 8 + 1, src, dst);
        }
        inline fn rawAluRM(a: Self, comptime op: Alu, dst: Gp, src: Mem) Error!void {
            return a.opRM(@as(u8, @backingInt(op)) * 8 + 3, @backingInt(dst), src, null, 0);
        }
        inline fn rawAluMR(a: Self, comptime op: Alu, dst: Mem, src: Gp) Error!void {
            return a.opRM(@as(u8, @backingInt(op)) * 8 + 1, @backingInt(src), dst, null, 0);
        }
        inline fn rawAluRI(a: Self, comptime op: Alu, dst: Gp, imm: i32) Error!void {
            try a.buf.ensure(max_inst);
            var e: Enc = .{};
            e.rexRR(true, 0, dst);
            if (fitsI8(imm)) {
                e.u8_(0x83);
                e.modrmRR(@backingInt(op), dst);
                e.i8_(imm);
            } else if (dst == .rax) {
                e.u8_(@as(u8, @backingInt(op)) * 8 + 5); // accumulator short form
                e.i32_(imm);
            } else {
                e.u8_(0x81);
                e.modrmRR(@backingInt(op), dst);
                e.i32_(imm);
            }
            a.commit(&e);
        }

        pub inline fn addRR(a: Self, d: Gp, s: Gp) Error!void {
            return a.aluRR(.add, d, s);
        }
        pub inline fn addRI(a: Self, d: Gp, i: i32) Error!void {
            return a.aluRI(.add, d, i);
        }
        pub inline fn subRR(a: Self, d: Gp, s: Gp) Error!void {
            return a.aluRR(.sub, d, s);
        }
        pub inline fn subRI(a: Self, d: Gp, i: i32) Error!void {
            return a.aluRI(.sub, d, i);
        }
        pub inline fn cmpRR(a: Self, d: Gp, s: Gp) Error!void {
            return a.aluRR(.cmp, d, s);
        }
        pub inline fn cmpRI(a: Self, d: Gp, i: i32) Error!void {
            return a.aluRI(.cmp, d, i);
        }
        pub inline fn xorRR(a: Self, d: Gp, s: Gp) Error!void {
            return a.aluRR(.xor, d, s);
        }

        inline fn rawTestRR(a: Self, d: Gp, s: Gp) Error!void {
            return a.opRR(0x85, s, d);
        }

        inline fn rawImulRR(a: Self, dst: Gp, src: Gp) Error!void {
            try a.buf.ensure(max_inst);
            var e: Enc = .{};
            e.rexRR(true, @backingInt(dst), src);
            e.u8_(0x0F);
            e.u8_(0xAF);
            e.modrmRR(@backingInt(dst), src);
            a.commit(&e);
        }
        inline fn rawImulRM(a: Self, dst: Gp, src: Mem) Error!void {
            try a.buf.ensure(max_inst);
            var e: Enc = .{};
            e.rexRM(true, @backingInt(dst), src);
            e.u8_(0x0F);
            e.u8_(0xAF);
            e.modrmM(@backingInt(dst), src);
            a.commit(&e);
        }
        inline fn rawImulRRI(a: Self, dst: Gp, src: Gp, imm: i32) Error!void {
            try a.buf.ensure(max_inst);
            var e: Enc = .{};
            e.rexRR(true, @backingInt(dst), src);
            e.u8_(if (fitsI8(imm)) 0x6B else 0x69);
            e.modrmRR(@backingInt(dst), src);
            if (fitsI8(imm)) e.i8_(imm) else e.i32_(imm);
            a.commit(&e);
        }

        // ---- stack / control flow ---------------------------------------------------------

        /// One-byte-opcode-plus-register form (push/pop), REX.B for r8..r15.
        inline fn opPlusR(a: Self, comptime op: u8, r: Gp) Error!void {
            try a.buf.ensure(max_inst);
            var e: Enc = .{};
            if (r.hi() != 0) e.u8_(0x41);
            e.u8_(op + r.lo());
            a.commit(&e);
        }
        inline fn rawPush(a: Self, r: Gp) Error!void {
            return a.opPlusR(0x50, r);
        }
        inline fn rawPop(a: Self, r: Gp) Error!void {
            return a.opPlusR(0x58, r);
        }

        inline fn byte(a: Self, comptime b: u8) Error!void {
            try a.buf.ensure(1);
            a.buf.put8(b);
        }
        inline fn rawRet(a: Self) Error!void {
            return a.byte(0xC3);
        }
        inline fn rawNop(a: Self) Error!void {
            return a.byte(0x90);
        }

        /// FF /digit with a register operand (call r / jmp r).
        inline fn ffR(a: Self, comptime digit: u8, r: Gp) Error!void {
            try a.buf.ensure(max_inst);
            var e: Enc = .{};
            if (r.hi() != 0) e.u8_(0x41);
            e.u8_(0xFF);
            e.modrmRR(digit, r);
            a.commit(&e);
        }
        inline fn rawCallR(a: Self, r: Gp) Error!void {
            return a.ffR(2, r);
        }
        inline fn rawJmpR(a: Self, r: Gp) Error!void {
            return a.ffR(4, r);
        }

        /// Branch to a label. Bound (backward) labels get rel8 when it fits; unbound (forward)
        /// labels get rel32 + a fixup resolved in `CodeBuffer.finalize` — AsmJit's policy.
        inline fn branch(a: Self, comptime short_op: []const u8, comptime long_op: []const u8, l: Label) Error!void {
            if (!a.buf.isValidLabel(l)) return error.InvalidLabel; // always on: avoids an out-of-bounds read
            try a.buf.ensure(max_inst);
            const start: i64 = a.buf.offset();
            var e: Enc = .{};
            if (a.buf.labelOffset(l)) |target| {
                const d8 = target - (start + short_op.len + 1);
                if (short_op.len != 0 and fitsI8(d8)) {
                    inline for (short_op) |b| e.u8_(b);
                    e.i8_(d8);
                } else {
                    inline for (long_op) |b| e.u8_(b);
                    e.i32_(target - (start + long_op.len + 4));
                }
                return a.commit(&e);
            }
            inline for (long_op) |b| e.u8_(b);
            e.u32_(0);
            const at: u32 = @intCast(start + long_op.len);
            try a.buf.addFixup(.{ .at = at, .base = at + 4, .label = l.id, .kind = .rel32 });
            a.commit(&e);
        }

        inline fn rawJmp(a: Self, l: Label) Error!void {
            return a.branch(&.{0xEB}, &.{0xE9}, l);
        }
        inline fn rawCall(a: Self, l: Label) Error!void {
            return a.branch(&.{}, &.{0xE8}, l);
        }
        inline fn rawJcc(a: Self, comptime c: Cond, l: Label) Error!void {
            const cc: u8 = @backingInt(c);
            return a.branch(&.{0x70 + cc}, &.{ 0x0F, 0x80 + cc }, l);
        }

        /// Calls an absolute address: `mov r11, imm; call r11` (r11 is caller-saved scratch in SysV).
        pub inline fn callAbs(a: Self, addr: usize) Error!void {
            try a.movRI(.r11, @bitCast(@as(u64, addr)));
            try a.callR(.r11);
        }

        // ---- public emitters: validation + hooks around the raw encoders ----------------
        pub inline fn movRR(a: Self, dst: Gp, src: Gp) Error!void {
            if (!a.active()) return a.rawMovRR(dst, src);
            return a.emit(.mov_rr, .{ .r0 = dst, .r1 = src }, rawMovRR, .{ dst, src });
        }
        pub inline fn movRM(a: Self, dst: Gp, src: Mem) Error!void {
            if (!a.active()) return a.rawMovRM(dst, src);
            return a.emit(.mov_rm, .{ .r0 = dst, .mem = src }, rawMovRM, .{ dst, src });
        }
        pub inline fn movMR(a: Self, dst: Mem, src: Gp) Error!void {
            if (!a.active()) return a.rawMovMR(dst, src);
            return a.emit(.mov_mr, .{ .mem = dst, .r0 = src }, rawMovMR, .{ dst, src });
        }
        pub inline fn movMI(a: Self, dst: Mem, imm: i32) Error!void {
            if (!a.active()) return a.rawMovMI(dst, imm);
            return a.emit(.mov_mi, .{ .mem = dst, .imm = imm }, rawMovMI, .{ dst, imm });
        }
        pub inline fn lea(a: Self, dst: Gp, src: Mem) Error!void {
            if (!a.active()) return a.rawLea(dst, src);
            return a.emit(.lea, .{ .r0 = dst, .mem = src }, rawLea, .{ dst, src });
        }
        pub inline fn movRI(a: Self, dst: Gp, imm: i64) Error!void {
            if (!a.active()) return a.rawMovRI(dst, imm);
            return a.emit(.mov_ri, .{ .r0 = dst, .imm = imm }, rawMovRI, .{ dst, imm });
        }
        pub inline fn movabs(a: Self, dst: Gp, imm: u64) Error!void {
            if (!a.active()) return a.rawMovabs(dst, imm);
            return a.emit(.movabs, .{ .r0 = dst, .imm = @bitCast(imm) }, rawMovabs, .{ dst, imm });
        }
        pub fn aluRR(a: Self, comptime op: Alu, dst: Gp, src: Gp) Error!void {
            if (!a.active()) return a.rawAluRR(op, dst, src);
            return a.emit(.alu_rr, .{ .alu = op, .r0 = dst, .r1 = src }, rawAluRR, .{ op, dst, src });
        }
        pub inline fn aluRM(a: Self, comptime op: Alu, dst: Gp, src: Mem) Error!void {
            if (!a.active()) return a.rawAluRM(op, dst, src);
            return a.emit(.alu_rm, .{ .alu = op, .r0 = dst, .mem = src }, rawAluRM, .{ op, dst, src });
        }
        pub inline fn aluMR(a: Self, comptime op: Alu, dst: Mem, src: Gp) Error!void {
            if (!a.active()) return a.rawAluMR(op, dst, src);
            return a.emit(.alu_mr, .{ .alu = op, .mem = dst, .r0 = src }, rawAluMR, .{ op, dst, src });
        }
        pub fn aluRI(a: Self, comptime op: Alu, dst: Gp, imm: i32) Error!void {
            if (!a.active()) return a.rawAluRI(op, dst, imm);
            return a.emit(.alu_ri, .{ .alu = op, .r0 = dst, .imm = imm }, rawAluRI, .{ op, dst, imm });
        }
        pub inline fn testRR(a: Self, d: Gp, s: Gp) Error!void {
            if (!a.active()) return a.rawTestRR(d, s);
            return a.emit(.test_rr, .{ .r0 = d, .r1 = s }, rawTestRR, .{ d, s });
        }
        pub inline fn imulRR(a: Self, dst: Gp, src: Gp) Error!void {
            if (!a.active()) return a.rawImulRR(dst, src);
            return a.emit(.imul_rr, .{ .r0 = dst, .r1 = src }, rawImulRR, .{ dst, src });
        }
        pub inline fn imulRM(a: Self, dst: Gp, src: Mem) Error!void {
            if (!a.active()) return a.rawImulRM(dst, src);
            return a.emit(.imul_rm, .{ .r0 = dst, .mem = src }, rawImulRM, .{ dst, src });
        }
        pub inline fn imulRRI(a: Self, dst: Gp, src: Gp, imm: i32) Error!void {
            if (!a.active()) return a.rawImulRRI(dst, src, imm);
            return a.emit(.imul_rri, .{ .r0 = dst, .r1 = src, .imm = imm }, rawImulRRI, .{ dst, src, imm });
        }
        pub fn push(a: Self, r: Gp) Error!void {
            if (!a.active()) return a.rawPush(r);
            return a.emit(.push, .{ .r0 = r }, rawPush, .{r});
        }
        pub fn pop(a: Self, r: Gp) Error!void {
            if (!a.active()) return a.rawPop(r);
            return a.emit(.pop, .{ .r0 = r }, rawPop, .{r});
        }
        pub fn ret(a: Self) Error!void {
            if (!a.active()) return a.rawRet();
            return a.emit(.ret, .{}, rawRet, .{});
        }
        pub fn nop(a: Self) Error!void {
            if (!a.active()) return a.rawNop();
            return a.emit(.nop, .{}, rawNop, .{});
        }
        /// `int3` (0xCC): trap.
        pub fn int3(a: Self) Error!void {
            try a.buf.ensure(1);
            a.buf.put8(0xCC);
        }
        /// Pads with the recommended multi-byte NOPs up to a multiple of `boundary` (power of two).
        pub fn alignNop(a: Self, boundary: u32) Error!void {
            var pad = padding(a.buf.offset(), boundary);
            try a.buf.ensure(pad);
            while (pad > 0) {
                const n = @min(pad, nops.len);
                for (nops[n - 1][0..n]) |b| a.buf.put8(b);
                pad -= n;
            }
        }
        /// Pads with `int3` up to a multiple of `boundary` (power of two), e.g. between functions.
        pub fn alignTrap(a: Self, boundary: u32) Error!void {
            const pad = padding(a.buf.offset(), boundary);
            try a.buf.ensure(pad);
            for (0..pad) |_| a.buf.put8(0xCC);
        }
        /// Frame prolog: `push rbp; mov rbp, rsp; push <saved>...; sub rsp, n`, where `n` is
        /// `locals` rounded so that rsp stays 16-byte aligned (both SysV and Win64).
        /// Returns the stack adjustment to pass to `epilog`.
        pub fn prolog(a: Self, saved: []const Gp, locals: u32) Error!i32 {
            try a.push(.rbp);
            try a.movRR(.rbp, .rsp);
            for (saved) |r| try a.push(r);
            const used: u32 = @intCast(saved.len * 8);
            const adj = std.mem.alignForward(u32, used + locals, 16) - used;
            if (adj > std.math.maxInt(i32)) return error.InvalidOperand;
            if (adj != 0) try a.subRI(.rsp, @intCast(adj));
            return @intCast(adj);
        }
        /// Frame epilog matching `prolog(saved, ...)`: `add rsp, adj; pop <saved reversed>; pop rbp; ret`.
        pub fn epilog(a: Self, saved: []const Gp, adj: i32) Error!void {
            if (adj != 0) try a.addRI(.rsp, adj);
            var i = saved.len;
            while (i > 0) {
                i -= 1;
                try a.pop(saved[i]);
            }
            try a.pop(.rbp);
            try a.ret();
        }
        pub fn callR(a: Self, r: Gp) Error!void {
            if (!a.active()) return a.rawCallR(r);
            return a.emit(.call_r, .{ .r0 = r }, rawCallR, .{r});
        }
        pub fn jmpR(a: Self, r: Gp) Error!void {
            if (!a.active()) return a.rawJmpR(r);
            return a.emit(.jmp_r, .{ .r0 = r }, rawJmpR, .{r});
        }
        pub fn jmp(a: Self, l: Label) Error!void {
            if (!a.active()) return a.rawJmp(l);
            return a.emit(.jmp, .{ .label = l }, rawJmp, .{l});
        }
        pub fn call(a: Self, l: Label) Error!void {
            if (!a.active()) return a.rawCall(l);
            return a.emit(.call, .{ .label = l }, rawCall, .{l});
        }
        pub fn jcc(a: Self, comptime c: Cond, l: Label) Error!void {
            if (!a.active()) return a.rawJcc(c, l);
            return a.emit(.jcc, .{ .cond = c, .label = l }, rawJcc, .{ c, l });
        }
    };
}

// ---- tests: encodings verified against GNU as / AsmJit output ---------------------

fn expectEnc(expected: []const u8, comptime f: anytype, args: anytype) !void {
    var cb = CodeBuffer.init(std.testing.allocator);
    defer cb.deinit();
    const a = Assembler.init(&cb);
    try @call(.auto, f, .{a} ++ args);
    try std.testing.expectEqualSlices(u8, expected, cb.code());
}

test "mov" {
    try expectEnc(&.{ 0x48, 0x89, 0xC8 }, Assembler.movRR, .{ Gp.rax, Gp.rcx }); // mov rax, rcx
    try expectEnc(&.{ 0x4D, 0x89, 0xC8 }, Assembler.movRR, .{ Gp.r8, Gp.r9 }); // mov r8, r9
    try expectEnc(&.{ 0xB8, 42, 0, 0, 0 }, Assembler.movRI, .{ Gp.rax, @as(i64, 42) }); // mov eax, 42
    try expectEnc(&.{ 0x41, 0xBB, 1, 0, 0, 0 }, Assembler.movRI, .{ Gp.r11, @as(i64, 1) }); // mov r11d, 1
    try expectEnc(&.{ 0x48, 0xC7, 0xC1, 0xFF, 0xFF, 0xFF, 0xFF }, Assembler.movRI, .{ Gp.rcx, @as(i64, -1) }); // mov rcx, -1
    try expectEnc(&.{ 0x48, 0xB8, 0xEF, 0xCD, 0xAB, 0x89, 0x67, 0x45, 0x23, 0x01 }, Assembler.movRI, .{ Gp.rax, @as(i64, 0x0123456789ABCDEF) });
    try expectEnc(&.{ 0x49, 0xBF, 0, 0, 0, 0, 1, 0, 0, 0 }, Assembler.movRI, .{ Gp.r15, @as(i64, 1 << 32) });
}

test "mem operands" {
    try expectEnc(&.{ 0x48, 0x8B, 0x07 }, Assembler.movRM, .{ Gp.rax, ptr(.rdi, 0) }); // mov rax, [rdi]
    try expectEnc(&.{ 0x48, 0x8B, 0x04, 0x24 }, Assembler.movRM, .{ Gp.rax, ptr(.rsp, 0) }); // [rsp]
    try expectEnc(&.{ 0x48, 0x8B, 0x45, 0x00 }, Assembler.movRM, .{ Gp.rax, ptr(.rbp, 0) }); // [rbp]
    try expectEnc(&.{ 0x49, 0x8B, 0x45, 0x00 }, Assembler.movRM, .{ Gp.rax, ptr(.r13, 0) }); // [r13]
    try expectEnc(&.{ 0x49, 0x8B, 0x04, 0x24 }, Assembler.movRM, .{ Gp.rax, ptr(.r12, 0) }); // [r12]
    try expectEnc(&.{ 0x48, 0x8B, 0x47, 0x08 }, Assembler.movRM, .{ Gp.rax, ptr(.rdi, 8) }); // [rdi+8]
    try expectEnc(&.{ 0x48, 0x8B, 0x87, 0x00, 0x01, 0, 0 }, Assembler.movRM, .{ Gp.rax, ptr(.rdi, 256) });
    try expectEnc(&.{ 0x4A, 0x8B, 0x44, 0xC7, 0x10 }, Assembler.movRM, .{ Gp.rax, ptrIdx(.rdi, .r8, .x8, 16) }); // [rdi+r8*8+16]
    try expectEnc(&.{ 0x4C, 0x89, 0x4C, 0x24, 0xF8 }, Assembler.movMR, .{ ptr(.rsp, -8), Gp.r9 }); // mov [rsp-8], r9
    try expectEnc(&.{ 0x48, 0x8D, 0x04, 0x8E }, Assembler.lea, .{ Gp.rax, ptrIdx(.rsi, .rcx, .x4, 0) }); // lea rax,[rsi+rcx*4]
    try expectEnc(&.{ 0x48, 0x8B, 0x04, 0xCD, 0x10, 0, 0, 0 }, Assembler.movRM, .{ Gp.rax, Mem{ .index = .rcx, .scale = .x8, .disp = 16 } });
    try expectEnc(&.{ 0x48, 0xC7, 0x07, 5, 0, 0, 0 }, Assembler.movMI, .{ ptr(.rdi, 0), @as(i32, 5) }); // mov qword [rdi], 5
}

test "alu" {
    try expectEnc(&.{ 0x48, 0x01, 0xF7 }, Assembler.addRR, .{ Gp.rdi, Gp.rsi }); // add rdi, rsi
    try expectEnc(&.{ 0x48, 0x83, 0xC0, 0x01 }, Assembler.addRI, .{ Gp.rax, @as(i32, 1) }); // add rax, 1
    try expectEnc(&.{ 0x48, 0x05, 0x00, 0x01, 0, 0 }, Assembler.addRI, .{ Gp.rax, @as(i32, 256) }); // add rax, 256
    try expectEnc(&.{ 0x49, 0x81, 0xEA, 0x00, 0x01, 0, 0 }, Assembler.subRI, .{ Gp.r10, @as(i32, 256) }); // sub r10, 256
    try expectEnc(&.{ 0x4C, 0x29, 0xC1 }, Assembler.subRR, .{ Gp.rcx, Gp.r8 }); // sub rcx, r8
    try expectEnc(&.{ 0x48, 0x39, 0xF7 }, Assembler.cmpRR, .{ Gp.rdi, Gp.rsi }); // cmp rdi, rsi
    try expectEnc(&.{ 0x48, 0x0F, 0xAF, 0xC1 }, Assembler.imulRR, .{ Gp.rax, Gp.rcx }); // imul rax, rcx
    try expectEnc(&.{ 0x4D, 0x6B, 0xC1, 0x0A }, Assembler.imulRRI, .{ Gp.r8, Gp.r9, @as(i32, 10) }); // imul r8, r9, 10
    try expectEnc(&.{ 0x48, 0x03, 0x47, 0x08 }, Assembler.aluRM, .{ Alu.add, Gp.rax, ptr(.rdi, 8) }); // add rax,[rdi+8]
}

test "stack and calls" {
    try expectEnc(&.{0x55}, Assembler.push, .{Gp.rbp});
    try expectEnc(&.{ 0x41, 0x57 }, Assembler.push, .{Gp.r15});
    try expectEnc(&.{ 0x41, 0x5C }, Assembler.pop, .{Gp.r12});
    try expectEnc(&.{0xC3}, Assembler.ret, .{});
    try expectEnc(&.{ 0x41, 0xFF, 0xD3 }, Assembler.callR, .{Gp.r11}); // call r11
    try expectEnc(&.{ 0xFF, 0xD0 }, Assembler.callR, .{Gp.rax});
}

test "labels: forward rel32, backward rel8/rel32" {
    var cb = CodeBuffer.init(std.testing.allocator);
    defer cb.deinit();
    const a = Assembler.init(&cb);
    const top = try a.newLabel();
    const out = try a.newLabel();
    try a.bind(top);
    try a.jcc(.e, out); // 0F 84 rel32 (forward)
    try a.nop();
    try a.jmp(top); // EB rel8 backward: -(6+1+2) = -9
    try a.bind(out);
    try a.ret();
    try cb.finalize();
    try std.testing.expectEqualSlices(u8, &.{ 0x0F, 0x84, 0x03, 0, 0, 0, 0x90, 0xEB, 0xF7, 0xC3 }, cb.code());
}
