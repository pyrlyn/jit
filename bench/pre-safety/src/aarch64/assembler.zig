//! AArch64 Assembler (A64): fixed 32-bit words, emitted directly.
//! Subset: movz/movk/mov(imm64 via movz+movk)/mov reg, add/sub (reg, imm12), mul,
//! cmp, b, b.cond, cbz/cbnz, bl, blr, ret, ldr/str (unsigned imm offset).
const std = @import("std");
const core = @import("../core/code_buffer.zig");
const CodeBuffer = core.CodeBuffer;
pub const Label = core.Label;
pub const Error = core.Error || error{InvalidImmediate};

/// X0..X30; 31 is SP or XZR depending on the instruction.
pub const X = enum(u5) {
    x0,
    x1,
    x2,
    x3,
    x4,
    x5,
    x6,
    x7,
    x8,
    x9,
    x10,
    x11,
    x12,
    x13,
    x14,
    x15,
    x16,
    x17,
    x18,
    x19,
    x20,
    x21,
    x22,
    x23,
    x24,
    x25,
    x26,
    x27,
    x28,
    x29,
    x30,
    zr_sp,
    pub const fp: X = .x29;
    pub const lr: X = .x30;
    inline fn n(r: X) u32 {
        return @intFromEnum(r);
    }
};

pub const Cond = enum(u4) { eq, ne, hs, lo, mi, pl, vs, vc, hi, ls, ge, lt, gt, le, al, nv };

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

    pub inline fn word(a: Assembler, w: u32) Error!void {
        try a.buf.ensure(4);
        a.buf.put32(w);
    }

    pub fn movz(a: Assembler, rd: X, imm16: u16, comptime shift: u6) Error!void {
        comptime std.debug.assert(shift % 16 == 0);
        return a.word(0xD2800000 | @as(u32, shift / 16) << 21 | @as(u32, imm16) << 5 | rd.n());
    }
    pub fn movk(a: Assembler, rd: X, imm16: u16, comptime shift: u6) Error!void {
        comptime std.debug.assert(shift % 16 == 0);
        return a.word(0xF2800000 | @as(u32, shift / 16) << 21 | @as(u32, imm16) << 5 | rd.n());
    }
    /// Materializes any 64-bit constant with movz + up to three movk (skips zero halves).
    pub fn movImm(a: Assembler, rd: X, imm: u64) Error!void {
        try a.movz(rd, @truncate(imm), 0);
        inline for (.{ 16, 32, 48 }) |sh| {
            const h: u16 = @truncate(imm >> sh);
            if (h != 0) try a.movk(rd, h, sh);
        }
    }
    /// mov xd, xm  (orr xd, xzr, xm)
    pub fn mov(a: Assembler, rd: X, rm: X) Error!void {
        return a.word(0xAA0003E0 | rm.n() << 16 | rd.n());
    }

    pub fn add(a: Assembler, rd: X, rn: X, rm: X) Error!void {
        return a.word(0x8B000000 | rm.n() << 16 | rn.n() << 5 | rd.n());
    }
    pub fn sub(a: Assembler, rd: X, rn: X, rm: X) Error!void {
        return a.word(0xCB000000 | rm.n() << 16 | rn.n() << 5 | rd.n());
    }
    pub fn mul(a: Assembler, rd: X, rn: X, rm: X) Error!void {
        return a.word(0x9B007C00 | rm.n() << 16 | rn.n() << 5 | rd.n());
    }
    pub fn addImm(a: Assembler, rd: X, rn: X, imm12: u12) Error!void {
        return a.word(0x91000000 | @as(u32, imm12) << 10 | rn.n() << 5 | rd.n());
    }
    pub fn subImm(a: Assembler, rd: X, rn: X, imm12: u12) Error!void {
        return a.word(0xD1000000 | @as(u32, imm12) << 10 | rn.n() << 5 | rd.n());
    }
    /// cmp xn, xm  (subs xzr, xn, xm)
    pub fn cmp(a: Assembler, rn: X, rm: X) Error!void {
        return a.word(0xEB00001F | rm.n() << 16 | rn.n() << 5);
    }
    pub fn cmpImm(a: Assembler, rn: X, imm12: u12) Error!void {
        return a.word(0xF100001F | @as(u32, imm12) << 10 | rn.n() << 5);
    }

    /// ldr xt, [xn, #off] — off must be a multiple of 8, < 32768.
    pub fn ldr(a: Assembler, rt: X, rn: X, off: u15) Error!void {
        if (off % 8 != 0) return error.InvalidImmediate;
        return a.word(0xF9400000 | @as(u32, off / 8) << 10 | rn.n() << 5 | rt.n());
    }
    pub fn str(a: Assembler, rt: X, rn: X, off: u15) Error!void {
        if (off % 8 != 0) return error.InvalidImmediate;
        return a.word(0xF9000000 | @as(u32, off / 8) << 10 | rn.n() << 5 | rt.n());
    }

    pub fn ret(a: Assembler) Error!void {
        return a.word(0xD65F03C0);
    }
    pub fn blr(a: Assembler, rn: X) Error!void {
        return a.word(0xD63F0000 | rn.n() << 5);
    }

    inline fn branch(a: Assembler, base_word: u32, comptime kind: core.FixupKind, l: Label) Error!void {
        try a.buf.ensure(4);
        const at = a.buf.offset();
        if (a.buf.labelOffset(l)) |t| {
            const words: i64 = @divExact(@as(i64, t) - at, 4);
            const bits: u32 = @truncate(@as(u64, @bitCast(words)));
            a.buf.put32(base_word | if (kind == .a64_imm26) bits & 0x03FF_FFFF else (bits & 0x7FFFF) << 5);
            return;
        }
        a.buf.put32(base_word);
        try a.buf.addFixup(.{ .at = at, .base = at, .label = l.id, .kind = kind });
    }
    pub fn b(a: Assembler, l: Label) Error!void {
        return a.branch(0x14000000, .a64_imm26, l);
    }
    pub fn bl(a: Assembler, l: Label) Error!void {
        return a.branch(0x94000000, .a64_imm26, l);
    }
    pub fn bcond(a: Assembler, comptime c: Cond, l: Label) Error!void {
        return a.branch(0x54000000 | @as(u32, @intFromEnum(c)), .a64_imm19, l);
    }
    pub fn cbz(a: Assembler, rt: X, l: Label) Error!void {
        return a.branch(0xB4000000 | rt.n(), .a64_imm19, l);
    }
    pub fn cbnz(a: Assembler, rt: X, l: Label) Error!void {
        return a.branch(0xB5000000 | rt.n(), .a64_imm19, l);
    }
};

fn expectWords(expected: []const u32, cb: *CodeBuffer) !void {
    try cb.finalize();
    const c = cb.code();
    try std.testing.expectEqual(expected.len * 4, c.len);
    for (expected, 0..) |w, i| try std.testing.expectEqual(w, std.mem.readInt(u32, c[i * 4 ..][0..4], .little));
}

test "a64 encodings (llvm-mc reference)" {
    var cb = CodeBuffer.init(std.testing.allocator);
    defer cb.deinit();
    const a = Assembler.init(&cb);
    try a.movz(.x0, 42, 0); //            mov x0, #42
    try a.movk(.x1, 0x1234, 16); //       movk x1, #0x1234, lsl #16
    try a.add(.x0, .x0, .x1); //          add x0, x0, x1
    try a.addImm(.x2, .x3, 16); //        add x2, x3, #16
    try a.sub(.x4, .x5, .x6); //          sub x4, x5, x6
    try a.mul(.x0, .x1, .x2); //          mul x0, x1, x2
    try a.cmp(.x0, .x1); //               cmp x0, x1
    try a.mov(.x0, .x19); //              mov x0, x19
    try a.ldr(.x0, .x1, 8); //            ldr x0, [x1, #8]
    try a.str(.x2, .zr_sp, 16); //        str x2, [sp, #16]
    try a.ret(); //                       ret
    try expectWords(&.{ 0xD2800540, 0xF2A24681, 0x8B010000, 0x91004062, 0xCB0600A4, 0x9B027C20, 0xEB01001F, 0xAA1303E0, 0xF9400420, 0xF9000BE2, 0xD65F03C0 }, &cb);
}

test "a64 branches" {
    var cb = CodeBuffer.init(std.testing.allocator);
    defer cb.deinit();
    const a = Assembler.init(&cb);
    const top = try a.newLabel();
    const out = try a.newLabel();
    try a.bind(top);
    try a.cbz(.x0, out); // +3 words
    try a.subImm(.x0, .x0, 1);
    try a.b(top); // -2 words
    try a.bind(out);
    try a.ret();
    try expectWords(&.{ 0xB4000060, 0xD1000400, 0x17FFFFFE, 0xD65F03C0 }, &cb);
}
