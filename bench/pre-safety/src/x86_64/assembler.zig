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
pub const Error = core.Error;

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
        return @intFromEnum(r) & 7;
    }
    inline fn hi(r: Gp) u8 {
        return @intFromEnum(r) >> 3;
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
            e.u8_(@as(u8, @intFromEnum(m.scale)) << 6 | idx << 3 | 5);
            e.u32_(@bitCast(m.disp));
            return;
        };
        // rbp/r13 as base cannot use mod=00 (that slot means disp32 / RIP-relative).
        const mod: u8 = if (m.disp == 0 and base.lo() != 5) 0x00 else if (fitsI8(m.disp)) 0x40 else 0x80;
        if (m.index != null or base.lo() == 4) {
            const idx: u8 = if (m.index) |i| i.lo() else 4;
            e.u8_(mod | r | 4);
            e.u8_(@as(u8, @intFromEnum(m.scale)) << 6 | idx << 3 | base.lo());
        } else e.u8_(mod | r | base.lo());
        switch (mod) {
            0x40 => e.i8_(m.disp),
            0x80 => e.u32_(@bitCast(m.disp)),
            else => {},
        }
    }
};

pub const Assembler = struct {
    buf: *CodeBuffer,

    pub fn init(buf: *CodeBuffer) Assembler {
        return .{ .buf = buf };
    }
    pub inline fn newLabel(a: Assembler) Error!Label {
        return a.buf.newLabel();
    }
    pub inline fn bind(a: Assembler, l: Label) Error!void {
        return a.buf.bind(l);
    }

    /// Writes the scratch with one 16-byte store (capacity was ensured) and bumps len.
    inline fn commit(a: Assembler, e: *const Enc) void {
        const cb = a.buf;
        const p = cb.data[cb.len..][0..max_inst];
        std.mem.writeInt(u64, p[0..8], e.lo, .little);
        std.mem.writeInt(u64, p[8..16], e.hi, .little);
        cb.len += e.n;
    }

    inline fn opRR(a: Assembler, comptime op: u8, reg: Gp, rm: Gp) Error!void {
        try a.buf.ensure(max_inst);
        var e: Enc = .{};
        e.rexRR(true, @intFromEnum(reg), rm);
        e.u8_(op);
        e.modrmRR(@intFromEnum(reg), rm);
        a.commit(&e);
    }

    inline fn opRM(a: Assembler, comptime op: u8, reg: u8, m: Mem, comptime imm32: ?void, imm: i32) Error!void {
        try a.buf.ensure(max_inst);
        var e: Enc = .{};
        e.rexRM(true, reg, m);
        e.u8_(op);
        e.modrmM(reg, m);
        if (imm32 != null) e.u32_(@bitCast(imm));
        a.commit(&e);
    }

    // ---- mov --------------------------------------------------------------------

    pub fn movRR(a: Assembler, dst: Gp, src: Gp) Error!void {
        return a.opRR(0x89, src, dst);
    }
    pub fn movRM(a: Assembler, dst: Gp, src: Mem) Error!void {
        return a.opRM(0x8B, @intFromEnum(dst), src, null, 0);
    }
    pub fn movMR(a: Assembler, dst: Mem, src: Gp) Error!void {
        return a.opRM(0x89, @intFromEnum(src), dst, null, 0);
    }
    /// mov qword [m], imm32 (sign-extended)
    pub fn movMI(a: Assembler, dst: Mem, imm: i32) Error!void {
        return a.opRM(0xC7, 0, dst, {}, imm);
    }
    pub fn lea(a: Assembler, dst: Gp, src: Mem) Error!void {
        return a.opRM(0x8D, @intFromEnum(dst), src, null, 0);
    }

    /// Shortest encoding: u32 => `mov r32, imm32` (zero-extends), i32 => `mov r/m64, imm32`
    /// (sign-extends), otherwise `movabs r64, imm64`.
    pub fn movRI(a: Assembler, dst: Gp, imm: i64) Error!void {
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
    pub fn movabs(a: Assembler, dst: Gp, imm: u64) Error!void {
        try a.buf.ensure(max_inst);
        var e: Enc = .{};
        e.u8_(REX_W | dst.hi());
        e.u8_(0xB8 + dst.lo());
        e.u64_(imm);
        a.commit(&e);
    }

    // ---- ALU ----------------------------------------------------------------------

    pub fn aluRR(a: Assembler, comptime op: Alu, dst: Gp, src: Gp) Error!void {
        return a.opRR(@as(u8, @intFromEnum(op)) * 8 + 1, src, dst);
    }
    pub fn aluRM(a: Assembler, comptime op: Alu, dst: Gp, src: Mem) Error!void {
        return a.opRM(@as(u8, @intFromEnum(op)) * 8 + 3, @intFromEnum(dst), src, null, 0);
    }
    pub fn aluMR(a: Assembler, comptime op: Alu, dst: Mem, src: Gp) Error!void {
        return a.opRM(@as(u8, @intFromEnum(op)) * 8 + 1, @intFromEnum(src), dst, null, 0);
    }
    pub fn aluRI(a: Assembler, comptime op: Alu, dst: Gp, imm: i32) Error!void {
        try a.buf.ensure(max_inst);
        var e: Enc = .{};
        e.rexRR(true, 0, dst);
        if (fitsI8(imm)) {
            e.u8_(0x83);
            e.modrmRR(@intFromEnum(op), dst);
            e.i8_(imm);
        } else if (dst == .rax) {
            e.u8_(@as(u8, @intFromEnum(op)) * 8 + 5); // accumulator short form
            e.i32_(imm);
        } else {
            e.u8_(0x81);
            e.modrmRR(@intFromEnum(op), dst);
            e.i32_(imm);
        }
        a.commit(&e);
    }

    pub inline fn addRR(a: Assembler, d: Gp, s: Gp) Error!void {
        return a.aluRR(.add, d, s);
    }
    pub inline fn addRI(a: Assembler, d: Gp, i: i32) Error!void {
        return a.aluRI(.add, d, i);
    }
    pub inline fn subRR(a: Assembler, d: Gp, s: Gp) Error!void {
        return a.aluRR(.sub, d, s);
    }
    pub inline fn subRI(a: Assembler, d: Gp, i: i32) Error!void {
        return a.aluRI(.sub, d, i);
    }
    pub inline fn cmpRR(a: Assembler, d: Gp, s: Gp) Error!void {
        return a.aluRR(.cmp, d, s);
    }
    pub inline fn cmpRI(a: Assembler, d: Gp, i: i32) Error!void {
        return a.aluRI(.cmp, d, i);
    }
    pub inline fn xorRR(a: Assembler, d: Gp, s: Gp) Error!void {
        return a.aluRR(.xor, d, s);
    }

    pub fn testRR(a: Assembler, d: Gp, s: Gp) Error!void {
        return a.opRR(0x85, s, d);
    }

    pub fn imulRR(a: Assembler, dst: Gp, src: Gp) Error!void {
        try a.buf.ensure(max_inst);
        var e: Enc = .{};
        e.rexRR(true, @intFromEnum(dst), src);
        e.u8_(0x0F);
        e.u8_(0xAF);
        e.modrmRR(@intFromEnum(dst), src);
        a.commit(&e);
    }
    pub fn imulRM(a: Assembler, dst: Gp, src: Mem) Error!void {
        try a.buf.ensure(max_inst);
        var e: Enc = .{};
        e.rexRM(true, @intFromEnum(dst), src);
        e.u8_(0x0F);
        e.u8_(0xAF);
        e.modrmM(@intFromEnum(dst), src);
        a.commit(&e);
    }
    pub fn imulRRI(a: Assembler, dst: Gp, src: Gp, imm: i32) Error!void {
        try a.buf.ensure(max_inst);
        var e: Enc = .{};
        e.rexRR(true, @intFromEnum(dst), src);
        e.u8_(if (fitsI8(imm)) 0x6B else 0x69);
        e.modrmRR(@intFromEnum(dst), src);
        if (fitsI8(imm)) e.i8_(imm) else e.i32_(imm);
        a.commit(&e);
    }

    // ---- stack / control flow ---------------------------------------------------------

    /// One-byte-opcode-plus-register form (push/pop), REX.B for r8..r15.
    inline fn opPlusR(a: Assembler, comptime op: u8, r: Gp) Error!void {
        try a.buf.ensure(max_inst);
        var e: Enc = .{};
        if (r.hi() != 0) e.u8_(0x41);
        e.u8_(op + r.lo());
        a.commit(&e);
    }
    pub fn push(a: Assembler, r: Gp) Error!void {
        return a.opPlusR(0x50, r);
    }
    pub fn pop(a: Assembler, r: Gp) Error!void {
        return a.opPlusR(0x58, r);
    }

    inline fn byte(a: Assembler, comptime b: u8) Error!void {
        try a.buf.ensure(1);
        a.buf.put8(b);
    }
    pub fn ret(a: Assembler) Error!void {
        return a.byte(0xC3);
    }
    pub fn nop(a: Assembler) Error!void {
        return a.byte(0x90);
    }

    /// FF /digit with a register operand (call r / jmp r).
    inline fn ffR(a: Assembler, comptime digit: u8, r: Gp) Error!void {
        try a.buf.ensure(max_inst);
        var e: Enc = .{};
        if (r.hi() != 0) e.u8_(0x41);
        e.u8_(0xFF);
        e.modrmRR(digit, r);
        a.commit(&e);
    }
    pub fn callR(a: Assembler, r: Gp) Error!void {
        return a.ffR(2, r);
    }
    pub fn jmpR(a: Assembler, r: Gp) Error!void {
        return a.ffR(4, r);
    }

    /// Branch to a label. Bound (backward) labels get rel8 when it fits; unbound (forward)
    /// labels get rel32 + a fixup resolved in `CodeBuffer.finalize` — AsmJit's policy.
    inline fn branch(a: Assembler, comptime short_op: []const u8, comptime long_op: []const u8, l: Label) Error!void {
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

    pub fn jmp(a: Assembler, l: Label) Error!void {
        return a.branch(&.{0xEB}, &.{0xE9}, l);
    }
    pub fn call(a: Assembler, l: Label) Error!void {
        return a.branch(&.{}, &.{0xE8}, l);
    }
    pub fn jcc(a: Assembler, comptime c: Cond, l: Label) Error!void {
        const cc: u8 = @intFromEnum(c);
        return a.branch(&.{0x70 + cc}, &.{ 0x0F, 0x80 + cc }, l);
    }

    /// Calls an absolute address: `mov r11, imm; call r11` (r11 is caller-saved scratch in SysV).
    pub fn callAbs(a: Assembler, addr: usize) Error!void {
        try a.movRI(.r11, @bitCast(@as(u64, addr)));
        try a.callR(.r11);
    }
};

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
