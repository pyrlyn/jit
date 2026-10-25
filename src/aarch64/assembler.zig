//! AArch64 Assembler (A64): fixed 32-bit words, emitted directly.
//! Subset: movz/movk/mov(imm64 via movz+movk)/mov reg, add/sub (reg, imm12), mul,
//! cmp, b, b.cond, cbz/cbnz, bl, blr, ret, ldr/str (unsigned imm offset).
const std = @import("std");
const core = @import("../core/code_buffer.zig");
const CodeBuffer = core.CodeBuffer;
pub const Label = core.Label;
pub const Error = core.Error || error{ InvalidImmediate, InvalidOperand, InvalidLabel };
const safety = @import("../core/safety.zig");
const build_opts = @import("zigjit_options");
pub const Options = safety.Options;

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

/// Instruction id seen by validation and hooks (one per public emitter; `movImm` is
/// reported as its movz/movk parts).
pub const Inst = enum { word, movz, movk, mov, add, sub, mul, add_imm, sub_imm, cmp, cmp_imm, ldr, str, ret, blr, b, bl, bcond, cbz, cbnz };

/// Operands of one instruction for validation and hooks (unused fields are null).
pub const Ops = struct {
    rd: ?X = null,
    rn: ?X = null,
    rm: ?X = null,
    imm: ?i64 = null,
    label: ?Label = null,
    cond: ?Cond = null,
};
pub const PreEvent = safety.PreEvent(Inst, Ops);
pub const Event = safety.Event(Inst, Ops);
pub const CountingHooks = safety.CountingHooks;
pub const LogHooks = safety.LogHooks;
pub const CallbackHooks = safety.CallbackHooks(Event);

pub const default_options: Options = .{
    .validate = build_opts.validate,
    .Hooks = if (build_opts.hooks) CallbackHooks else void,
};
pub const Assembler = AssemblerT(default_options);

/// Register 31 means XZR or SP depending on the instruction. This layer accepts it only where
/// the meaning is unambiguous and useful: SP in add/sub/cmp-immediate and as ldr/str base,
/// XZR as the source of `mov` and in `word`. Everywhere else it is rejected.
fn sp31Allowed(id: Inst, comptime field: []const u8) bool {
    return switch (id) {
        .word => true,
        .add_imm, .sub_imm, .cmp_imm => true,
        .ldr, .str => comptime std.mem.eql(u8, field, "rn"),
        .mov => comptime std.mem.eql(u8, field, "rm"),
        else => false,
    };
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
            if (comptime opts.validate) if (l.id >= a.buf.labels.items.len) return error.InvalidLabel;
            return a.buf.bind(l);
        }

        /// Full operand check (only compiled when `opts.validate`).
        pub fn validate(a: Self, id: Inst, ops: Ops) Error!void {
            inline for (.{ "rd", "rn", "rm" }) |f| {
                if (@field(ops, f)) |r| if (r == .zr_sp and !sp31Allowed(id, f)) return error.InvalidOperand;
            }
            if (ops.label) |l| {
                if (l.id >= a.buf.labels.items.len) return error.InvalidLabel;
                // Backward (bound) branches are encoded immediately: check the range here.
                if (a.buf.labelOffset(l)) |t| {
                    const words = @divExact(@as(i64, t) - @as(i64, @intCast(a.buf.len)), 4);
                    const bits: u6 = if (id == .b or id == .bl) 26 else 19;
                    const lim = @as(i64, 1) << (bits - 1);
                    if (words < -lim or words >= lim) return error.DisplacementOutOfRange;
                }
            }
        }

        /// False when no layer can act on this call: both layers compiled out, or hooks
        /// compiled in without an attached instance (one null check, no `Ops` built).
        inline fn active(a: Self) bool {
            if (comptime opts.validate) return true;
            if (comptime !has_hooks) return false;
            return a.hooks != null;
        }

        inline fn emit(a: Self, comptime id: Inst, ops: Ops, comptime raw: anytype, args: anytype) Error!void {
            if (comptime opts.validate) try a.validate(id, ops);
            if (comptime !has_hooks) return @call(.always_inline, raw, .{a} ++ args);
            if (a.hooks) |h| {
                @branchHint(.unlikely);
                return @call(.always_inline, emitHooked, .{ a, h, id, ops, raw, args });
            }
            return @call(.always_inline, raw, .{a} ++ args);
        }

        fn emitHooked(a: Self, h: *Hooks, comptime id: Inst, ops: Ops, comptime raw: anytype, args: anytype) Error!void {
            const start: u32 = @intCast(a.buf.len);
            if (comptime @hasDecl(Hooks, "pre")) try h.pre(.{ .id = id, .ops = ops, .offset = start });
            try @call(.always_inline, raw, .{a} ++ args);
            if (comptime @hasDecl(Hooks, "post")) h.post(.{ .id = id, .ops = ops, .offset = start, .bytes = a.buf.data[start..a.buf.len] });
        }

        inline fn put(a: Self, w: u32) Error!void {
            try a.buf.ensure(4);
            a.buf.put32(w);
        }

        fn rawMovz(a: Self, rd: X, imm16: u16, comptime shift: u6) Error!void {
            comptime std.debug.assert(shift % 16 == 0);
            return a.put(0xD2800000 | @as(u32, shift / 16) << 21 | @as(u32, imm16) << 5 | rd.n());
        }
        fn rawMovk(a: Self, rd: X, imm16: u16, comptime shift: u6) Error!void {
            comptime std.debug.assert(shift % 16 == 0);
            return a.put(0xF2800000 | @as(u32, shift / 16) << 21 | @as(u32, imm16) << 5 | rd.n());
        }
        /// Materializes any 64-bit constant with movz + up to three movk (skips zero halves).
        pub fn movImm(a: Self, rd: X, imm: u64) Error!void {
            try a.movz(rd, @truncate(imm), 0);
            inline for (.{ 16, 32, 48 }) |sh| {
                const h: u16 = @truncate(imm >> sh);
                if (h != 0) try a.movk(rd, h, sh);
            }
        }
        /// mov xd, xm  (orr xd, xzr, xm)
        fn rawMov(a: Self, rd: X, rm: X) Error!void {
            return a.put(0xAA0003E0 | rm.n() << 16 | rd.n());
        }

        fn rawAdd(a: Self, rd: X, rn: X, rm: X) Error!void {
            return a.put(0x8B000000 | rm.n() << 16 | rn.n() << 5 | rd.n());
        }
        fn rawSub(a: Self, rd: X, rn: X, rm: X) Error!void {
            return a.put(0xCB000000 | rm.n() << 16 | rn.n() << 5 | rd.n());
        }
        fn rawMul(a: Self, rd: X, rn: X, rm: X) Error!void {
            return a.put(0x9B007C00 | rm.n() << 16 | rn.n() << 5 | rd.n());
        }
        fn rawAddImm(a: Self, rd: X, rn: X, imm12: u12) Error!void {
            return a.put(0x91000000 | @as(u32, imm12) << 10 | rn.n() << 5 | rd.n());
        }
        fn rawSubImm(a: Self, rd: X, rn: X, imm12: u12) Error!void {
            return a.put(0xD1000000 | @as(u32, imm12) << 10 | rn.n() << 5 | rd.n());
        }
        /// cmp xn, xm  (subs xzr, xn, xm)
        fn rawCmp(a: Self, rn: X, rm: X) Error!void {
            return a.put(0xEB00001F | rm.n() << 16 | rn.n() << 5);
        }
        fn rawCmpImm(a: Self, rn: X, imm12: u12) Error!void {
            return a.put(0xF100001F | @as(u32, imm12) << 10 | rn.n() << 5);
        }

        /// ldr xt, [xn, #off] — off must be a multiple of 8, < 32768.
        fn rawLdr(a: Self, rt: X, rn: X, off: u15) Error!void {
            if (off % 8 != 0) return error.InvalidImmediate;
            return a.put(0xF9400000 | @as(u32, off / 8) << 10 | rn.n() << 5 | rt.n());
        }
        fn rawStr(a: Self, rt: X, rn: X, off: u15) Error!void {
            if (off % 8 != 0) return error.InvalidImmediate;
            return a.put(0xF9000000 | @as(u32, off / 8) << 10 | rn.n() << 5 | rt.n());
        }

        fn rawRet(a: Self) Error!void {
            return a.put(0xD65F03C0);
        }
        fn rawBlr(a: Self, rn: X) Error!void {
            return a.put(0xD63F0000 | rn.n() << 5);
        }

        inline fn branch(a: Self, base_word: u32, comptime kind: core.FixupKind, l: Label) Error!void {
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
        fn rawB(a: Self, l: Label) Error!void {
            return a.branch(0x14000000, .a64_imm26, l);
        }
        fn rawBl(a: Self, l: Label) Error!void {
            return a.branch(0x94000000, .a64_imm26, l);
        }
        fn rawBcond(a: Self, comptime c: Cond, l: Label) Error!void {
            return a.branch(0x54000000 | @as(u32, @intFromEnum(c)), .a64_imm19, l);
        }
        fn rawCbz(a: Self, rt: X, l: Label) Error!void {
            return a.branch(0xB4000000 | rt.n(), .a64_imm19, l);
        }
        fn rawCbnz(a: Self, rt: X, l: Label) Error!void {
            return a.branch(0xB5000000 | rt.n(), .a64_imm19, l);
        }

        // ---- public emitters: validation + hooks around the raw encoders ----------------
        /// Emits a raw 32-bit instruction word (no operand checks beyond the hook layer).
        pub fn word(a: Self, w: u32) Error!void {
            if (!a.active()) return a.put(w);
            return a.emit(.word, .{ .imm = w }, put, .{w});
        }
        pub fn movz(a: Self, rd: X, imm16: u16, comptime shift: u6) Error!void {
            if (!a.active()) return a.rawMovz(rd, imm16, shift);
            return a.emit(.movz, .{ .rd = rd, .imm = imm16 }, rawMovz, .{ rd, imm16, shift });
        }
        pub fn movk(a: Self, rd: X, imm16: u16, comptime shift: u6) Error!void {
            if (!a.active()) return a.rawMovk(rd, imm16, shift);
            return a.emit(.movk, .{ .rd = rd, .imm = imm16 }, rawMovk, .{ rd, imm16, shift });
        }
        pub fn mov(a: Self, rd: X, rm: X) Error!void {
            if (!a.active()) return a.rawMov(rd, rm);
            return a.emit(.mov, .{ .rd = rd, .rm = rm }, rawMov, .{ rd, rm });
        }
        pub fn add(a: Self, rd: X, rn: X, rm: X) Error!void {
            if (!a.active()) return a.rawAdd(rd, rn, rm);
            return a.emit(.add, .{ .rd = rd, .rn = rn, .rm = rm }, rawAdd, .{ rd, rn, rm });
        }
        pub fn sub(a: Self, rd: X, rn: X, rm: X) Error!void {
            if (!a.active()) return a.rawSub(rd, rn, rm);
            return a.emit(.sub, .{ .rd = rd, .rn = rn, .rm = rm }, rawSub, .{ rd, rn, rm });
        }
        pub fn mul(a: Self, rd: X, rn: X, rm: X) Error!void {
            if (!a.active()) return a.rawMul(rd, rn, rm);
            return a.emit(.mul, .{ .rd = rd, .rn = rn, .rm = rm }, rawMul, .{ rd, rn, rm });
        }
        pub fn addImm(a: Self, rd: X, rn: X, imm12: u12) Error!void {
            if (!a.active()) return a.rawAddImm(rd, rn, imm12);
            return a.emit(.add_imm, .{ .rd = rd, .rn = rn, .imm = imm12 }, rawAddImm, .{ rd, rn, imm12 });
        }
        pub fn subImm(a: Self, rd: X, rn: X, imm12: u12) Error!void {
            if (!a.active()) return a.rawSubImm(rd, rn, imm12);
            return a.emit(.sub_imm, .{ .rd = rd, .rn = rn, .imm = imm12 }, rawSubImm, .{ rd, rn, imm12 });
        }
        pub fn cmp(a: Self, rn: X, rm: X) Error!void {
            if (!a.active()) return a.rawCmp(rn, rm);
            return a.emit(.cmp, .{ .rn = rn, .rm = rm }, rawCmp, .{ rn, rm });
        }
        pub fn cmpImm(a: Self, rn: X, imm12: u12) Error!void {
            if (!a.active()) return a.rawCmpImm(rn, imm12);
            return a.emit(.cmp_imm, .{ .rn = rn, .imm = imm12 }, rawCmpImm, .{ rn, imm12 });
        }
        pub fn ldr(a: Self, rt: X, rn: X, off: u15) Error!void {
            if (!a.active()) return a.rawLdr(rt, rn, off);
            return a.emit(.ldr, .{ .rd = rt, .rn = rn, .imm = off }, rawLdr, .{ rt, rn, off });
        }
        pub fn str(a: Self, rt: X, rn: X, off: u15) Error!void {
            if (!a.active()) return a.rawStr(rt, rn, off);
            return a.emit(.str, .{ .rd = rt, .rn = rn, .imm = off }, rawStr, .{ rt, rn, off });
        }
        pub fn ret(a: Self) Error!void {
            if (!a.active()) return a.rawRet();
            return a.emit(.ret, .{}, rawRet, .{});
        }
        pub fn blr(a: Self, rn: X) Error!void {
            if (!a.active()) return a.rawBlr(rn);
            return a.emit(.blr, .{ .rn = rn }, rawBlr, .{rn});
        }
        pub fn b(a: Self, l: Label) Error!void {
            if (!a.active()) return a.rawB(l);
            return a.emit(.b, .{ .label = l }, rawB, .{l});
        }
        pub fn bl(a: Self, l: Label) Error!void {
            if (!a.active()) return a.rawBl(l);
            return a.emit(.bl, .{ .label = l }, rawBl, .{l});
        }
        pub fn bcond(a: Self, comptime c: Cond, l: Label) Error!void {
            if (!a.active()) return a.rawBcond(c, l);
            return a.emit(.bcond, .{ .cond = c, .label = l }, rawBcond, .{ c, l });
        }
        pub fn cbz(a: Self, rt: X, l: Label) Error!void {
            if (!a.active()) return a.rawCbz(rt, l);
            return a.emit(.cbz, .{ .rd = rt, .label = l }, rawCbz, .{ rt, l });
        }
        pub fn cbnz(a: Self, rt: X, l: Label) Error!void {
            if (!a.active()) return a.rawCbnz(rt, l);
            return a.emit(.cbnz, .{ .rd = rt, .label = l }, rawCbnz, .{ rt, l });
        }
    };
}

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
