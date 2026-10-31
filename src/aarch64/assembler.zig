//! AArch64 Assembler (A64): fixed 32-bit words, emitted directly.
//! Subset: movz/movk/mov(imm64 via movz+movk)/mov reg, add/sub (reg, imm12), mul, madd/msub/mneg,
//! smulh/umulh, sdiv/udiv, neg, and/orr/eor (reg), lsl/lsr/asr (imm), csel/cset,
//! cmp, b, b.cond, cbz/cbnz, bl, blr, ret, ldr/str (unsigned imm offset).
const std = @import("std");
const core = @import("../core/code_buffer.zig");
const CodeBuffer = core.CodeBuffer;
pub const Label = core.Label;
pub const Error = core.Error || error{ InvalidImmediate, InvalidOperand, InvalidLabel, UnsupportedCpuFeature, EncodingMismatch };
/// Generated from LLVM's AArch64.td (tools/gen_isa.zig, `zig build gen-tables`).
pub const isa = @import("isa.zig");
/// Host CPU feature detection.
pub const cpu = @import("cpu.zig");
const safety = @import("../core/safety.zig");
const build_opts = @import("pyrlyn_jit_options");
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
        return @backingInt(r);
    }
};

pub const Cond = enum(u4) { eq, ne, hs, lo, mi, pl, vs, vc, hi, ls, ge, lt, gt, le, al, nv };

/// Instruction id seen by validation and hooks (one per public emitter; `movImm` is
/// reported as its movz/movk parts).
pub const Inst = enum { word, movz, movk, mov, add, sub, mul, add_imm, sub_imm, cmp, cmp_imm, ldr, str, ret, blr, b, bl, bcond, cbz, cbnz, sdiv, udiv, madd, msub, mneg, smulh, umulh, neg, and_, orr, eor, csel, cset, lsl_imm, lsr_imm, asr_imm };

/// Operands of one instruction for validation and hooks (unused fields are null).
pub const Ops = struct {
    rd: ?X = null,
    rn: ?X = null,
    rm: ?X = null,
    ra: ?X = null,
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

/// AsmJit form for an instruction id (`word` has none).
/// AAPCS64 (Apple arm64 is the same for integer args up to 8 registers; Apple packs stack
/// arguments by size, this helper assumes 8-byte slots).
pub const CallConv = struct {
    pub const args = [_]X{ .x0, .x1, .x2, .x3, .x4, .x5, .x6, .x7 };
    pub const ret: X = .x0;
    pub const callee_saved = [_]X{ .x19, .x20, .x21, .x22, .x23, .x24, .x25, .x26, .x27, .x28 };
    /// Offset from x29 (after `prolog`) of integer argument `i` (`i >= args.len`).
    pub fn stackArgOffset(i: usize) u15 {
        std.debug.assert(i >= args.len);
        return @intCast(16 + 8 * (i - args.len));
    }
};

pub fn formFor(comptime id: Inst) ?isa.Form {
    return if (id == .word) null else @field(isa.table, @tagName(id));
}

/// Register 31 means SP or XZR depending on the instruction. Allowed where AsmJit's database
/// marks the field `Xd|SP` / `Xn|SP` (SP), plus XZR as the source of `mov` and in `word`.
/// Everywhere else 31 would silently mean XZR, which is almost always a bug: rejected.
fn reg31Allowed(comptime id: Inst, comptime field: []const u8) bool {
    if (id == .word) return true;
    if (id == .mov and comptime std.mem.eql(u8, field, "rm")) return true;
    const f = formFor(id).?;
    if (comptime std.mem.eql(u8, field, "rd")) return f.rd_sp;
    if (comptime std.mem.eql(u8, field, "rn")) return f.rn_sp;
    return false;
}

pub fn checkFeatureSet(need: isa.Features, have: isa.Features) Error!void {
    if (!need.subsetOf(have)) return error.UnsupportedCpuFeature;
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
            return a.buf.bind(l); // InvalidLabel / LabelAlreadyBound are always checked
        }

        /// Operand check before encoding (only compiled when `opts.validate`), driven by the
        /// AsmJit form: register-31 meaning, immediate width/scale, CPU features.
        pub inline fn validate(a: Self, comptime id: Inst, ops: Ops) Error!void {
            inline for (.{ "rd", "rn", "rm", "ra" }) |f| {
                if (@field(ops, f)) |r| if (r == .zr_sp and !reg31Allowed(id, f)) return error.InvalidOperand;
            }
            if (ops.label) |l| if (!a.buf.isValidLabel(l)) return error.InvalidLabel;
            if (comptime formFor(id)) |form| {
                if (ops.imm) |v| if (form.imm_bits != 0 and ops.label == null and id != .movz and id != .movk) {
                    if (@mod(v, form.imm_scale) != 0) return error.InvalidImmediate;
                    if (@divFloor(v, form.imm_scale) >= (@as(i64, 1) << @intCast(form.imm_bits))) return error.InvalidImmediate;
                };
                comptime if (isa.featureSet(form.features).intersectWith(safety.featureSet(isa.Feature, opts.deny_features)).count() != 0)
                    @compileError("instruction " ++ @tagName(id) ++ " needs a CPU feature in deny_features");
                switch (opts.cpu_check) {
                    .none => {},
                    .target => comptime {
                        if (!isa.featureSet(form.features).subsetOf(safety.featureSet(isa.Feature, opts.target_features)))
                            @compileError("instruction " ++ @tagName(id) ++ " needs CPU features missing from target_features");
                    },
                    .host => try checkFeatureSet(comptime isa.featureSet(form.features), cpu.host() orelse return error.UnsupportedCpuFeature),
                }
            }
        }

        /// Raw encoder + (validate mode) check that the word matches the form's fixed bits.
        inline fn encode(a: Self, comptime id: Inst, comptime raw: anytype, args: anytype) Error!void {
            if (comptime !opts.validate or formFor(id) == null) return @call(.always_inline, raw, .{a} ++ args);
            const form = comptime formFor(id).?;
            const start = a.buf.len;
            try @call(.always_inline, raw, .{a} ++ args);
            const w = std.mem.readInt(u32, a.buf.data[start..][0..4], .little);
            if (w & form.mask != form.value) {
                a.buf.len = start;
                return error.EncodingMismatch;
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
            if (comptime !has_hooks) return a.encode(id, raw, args);
            if (a.hooks) |h| {
                @branchHint(.unlikely);
                return @call(.always_inline, emitHooked, .{ a, h, id, ops, raw, args });
            }
            return a.encode(id, raw, args);
        }

        fn emitHooked(a: Self, h: *Hooks, comptime id: Inst, ops: Ops, comptime raw: anytype, args: anytype) Error!void {
            const start: u32 = @intCast(a.buf.len);
            if (comptime @hasDecl(Hooks, "pre")) try h.pre(.{ .id = id, .ops = ops, .offset = start });
            try a.encode(id, raw, args);
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
        fn rawRRR(a: Self, comptime base: u32, rd: X, rn: X, rm: X) Error!void {
            return a.put(base | rm.n() << 16 | rn.n() << 5 | rd.n());
        }
        fn rawRRRR(a: Self, comptime base: u32, rd: X, rn: X, rm: X, ra: X) Error!void {
            return a.put(base | rm.n() << 16 | ra.n() << 10 | rn.n() << 5 | rd.n());
        }
        fn rawCsel(a: Self, rd: X, rn: X, rm: X, c: Cond) Error!void {
            return a.put(0x9A800000 | rm.n() << 16 | @as(u32, @backingInt(c)) << 12 | rn.n() << 5 | rd.n());
        }
        fn rawCset(a: Self, rd: X, c: Cond) Error!void {
            // csinc Xd, xzr, xzr, invert(c)
            return a.put(0x9A9F07E0 | @as(u32, @backingInt(c) ^ 1) << 12 | rd.n());
        }
        fn rawShiftImm(a: Self, comptime base: u32, rd: X, rn: X, immr: u6, imms: u6) Error!void {
            return a.put(base | @as(u32, immr) << 16 | @as(u32, imms) << 10 | rn.n() << 5 | rd.n());
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
            if (!a.buf.isValidLabel(l)) return error.InvalidLabel; // always on
            try a.buf.ensure(4);
            const at = a.buf.offset();
            if (a.buf.labelOffset(l)) |t| {
                const words: i64 = @divExact(@as(i64, t) - at, 4);
                // Always on: a backward branch out of range is an error, never truncated.
                const lim: i64 = if (kind == .a64_imm26) 1 << 25 else 1 << 18;
                if (words < -lim or words >= lim) return error.DisplacementOutOfRange;
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
            return a.branch(0x54000000 | @as(u32, @backingInt(c)), .a64_imm19, l);
        }
        fn rawCbz(a: Self, rt: X, l: Label) Error!void {
            return a.branch(0xB4000000 | rt.n(), .a64_imm19, l);
        }
        fn rawCbnz(a: Self, rt: X, l: Label) Error!void {
            return a.branch(0xB5000000 | rt.n(), .a64_imm19, l);
        }

        // ---- public emitters: validation + hooks around the raw encoders ----------------
        /// Emits a raw 32-bit instruction word (no operand checks beyond the hook layer).
        /// `udf #imm16`: permanently undefined (trap).
        pub fn udf(a: Self, imm16: u16) Error!void {
            return a.word(imm16);
        }
        /// Pads with `nop` up to a multiple of `boundary` bytes (power of two, >= 4).
        pub fn alignNop(a: Self, boundary: u32) Error!void {
            return a.alignWith(boundary, 0xD503201F);
        }
        /// Pads with `udf #0` up to a multiple of `boundary` bytes (power of two, >= 4).
        pub fn alignTrap(a: Self, boundary: u32) Error!void {
            return a.alignWith(boundary, 0);
        }
        fn alignWith(a: Self, boundary: u32, w: u32) Error!void {
            std.debug.assert(std.math.isPowerOfTwo(boundary) and boundary >= 4);
            const off = a.buf.offset();
            const pad = (boundary - (off & (boundary - 1))) & (boundary - 1);
            try a.buf.ensure(pad);
            for (0..pad / 4) |_| a.buf.put32(w);
        }
        /// AAPCS64 frame prolog: `stp x29, x30, [sp, #-16]!; mov x29, sp`, then the callee-saved
        /// registers in `saved` pushed in pairs (`stp`, or `str` for an odd last one, 16 bytes each
        /// slot), then `sub sp, sp, n` (`n` = `locals` rounded up to 16, at most 4080).
        /// Returns `n` for `epilog`.
        pub fn prolog(a: Self, saved: []const X, locals: u32) Error!u12 {
            const n = std.mem.alignForward(u32, locals, 16);
            if (n > 4080) return error.InvalidOperand;
            for (saved) |r| if (@backingInt(r) >= 29) return error.InvalidOperand;
            try a.word(0xA9BF7BFD); // stp x29, x30, [sp, #-16]!
            try a.word(0x910003FD); // mov x29, sp
            var i: usize = 0;
            while (i < saved.len) : (i += 2) {
                const r0: u32 = @backingInt(saved[i]);
                if (i + 1 < saved.len) {
                    try a.word(0xA9BF03E0 | @as(u32, @backingInt(saved[i + 1])) << 10 | r0); // stp r0, r1, [sp, #-16]!
                } else try a.word(0xF81F0FE0 | r0); // str r0, [sp, #-16]!
            }
            if (n != 0) try a.subImm(@fromBackingInt(@intCast(31)), @fromBackingInt(@intCast(31)), @intCast(n));
            return @intCast(n);
        }
        /// Frame epilog matching `prolog(saved, ...)`: `add sp, sp, n`, restore `saved` in reverse,
        /// `ldp x29, x30, [sp], #16; ret`.
        pub fn epilog(a: Self, saved: []const X, n: u12) Error!void {
            if (n != 0) try a.addImm(@fromBackingInt(@intCast(31)), @fromBackingInt(@intCast(31)), n);
            var i: usize = (saved.len + 1) / 2 * 2;
            while (i > 0) {
                i -= 2;
                const r0: u32 = @backingInt(saved[i]);
                if (i + 1 < saved.len) {
                    try a.word(0xA8C103E0 | @as(u32, @backingInt(saved[i + 1])) << 10 | r0); // ldp r0, r1, [sp], #16
                } else try a.word(0xF84107E0 | r0); // ldr r0, [sp], #16
            }
            try a.word(0xA8C17BFD); // ldp x29, x30, [sp], #16
            try a.ret();
        }
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
        pub fn sdiv(a: Self, rd: X, rn: X, rm: X) Error!void {
            if (!a.active()) return a.rawRRR(0x9AC00C00, rd, rn, rm);
            return a.emit(.sdiv, .{ .rd = rd, .rn = rn, .rm = rm }, rawRRR, .{ 0x9AC00C00, rd, rn, rm });
        }
        pub fn udiv(a: Self, rd: X, rn: X, rm: X) Error!void {
            if (!a.active()) return a.rawRRR(0x9AC00800, rd, rn, rm);
            return a.emit(.udiv, .{ .rd = rd, .rn = rn, .rm = rm }, rawRRR, .{ 0x9AC00800, rd, rn, rm });
        }
        pub fn mneg(a: Self, rd: X, rn: X, rm: X) Error!void {
            if (!a.active()) return a.rawRRR(0x9B00FC00, rd, rn, rm);
            return a.emit(.mneg, .{ .rd = rd, .rn = rn, .rm = rm }, rawRRR, .{ 0x9B00FC00, rd, rn, rm });
        }
        pub fn smulh(a: Self, rd: X, rn: X, rm: X) Error!void {
            if (!a.active()) return a.rawRRR(0x9B407C00, rd, rn, rm);
            return a.emit(.smulh, .{ .rd = rd, .rn = rn, .rm = rm }, rawRRR, .{ 0x9B407C00, rd, rn, rm });
        }
        pub fn umulh(a: Self, rd: X, rn: X, rm: X) Error!void {
            if (!a.active()) return a.rawRRR(0x9BC07C00, rd, rn, rm);
            return a.emit(.umulh, .{ .rd = rd, .rn = rn, .rm = rm }, rawRRR, .{ 0x9BC07C00, rd, rn, rm });
        }
        pub fn and_(a: Self, rd: X, rn: X, rm: X) Error!void {
            if (!a.active()) return a.rawRRR(0x8A000000, rd, rn, rm);
            return a.emit(.and_, .{ .rd = rd, .rn = rn, .rm = rm }, rawRRR, .{ 0x8A000000, rd, rn, rm });
        }
        pub fn orr(a: Self, rd: X, rn: X, rm: X) Error!void {
            if (!a.active()) return a.rawRRR(0xAA000000, rd, rn, rm);
            return a.emit(.orr, .{ .rd = rd, .rn = rn, .rm = rm }, rawRRR, .{ 0xAA000000, rd, rn, rm });
        }
        pub fn eor(a: Self, rd: X, rn: X, rm: X) Error!void {
            if (!a.active()) return a.rawRRR(0xCA000000, rd, rn, rm);
            return a.emit(.eor, .{ .rd = rd, .rn = rn, .rm = rm }, rawRRR, .{ 0xCA000000, rd, rn, rm });
        }
        /// madd Xd, Xn, Xm, Xa: Xd = Xa + Xn * Xm.
        pub fn madd(a: Self, rd: X, rn: X, rm: X, ra: X) Error!void {
            if (!a.active()) return a.rawRRRR(0x9B000000, rd, rn, rm, ra);
            return a.emit(.madd, .{ .rd = rd, .rn = rn, .rm = rm, .ra = ra }, rawRRRR, .{ 0x9B000000, rd, rn, rm, ra });
        }
        /// msub Xd, Xn, Xm, Xa: Xd = Xa - Xn * Xm.
        pub fn msub(a: Self, rd: X, rn: X, rm: X, ra: X) Error!void {
            if (!a.active()) return a.rawRRRR(0x9B008000, rd, rn, rm, ra);
            return a.emit(.msub, .{ .rd = rd, .rn = rn, .rm = rm, .ra = ra }, rawRRRR, .{ 0x9B008000, rd, rn, rm, ra });
        }
        /// neg Xd, Xm (= sub Xd, xzr, Xm).
        pub fn neg(a: Self, rd: X, rm: X) Error!void {
            if (!a.active()) return a.rawRRR(0xCB0003E0, rd, .x0, rm);
            return a.emit(.neg, .{ .rd = rd, .rm = rm }, rawRRR, .{ 0xCB0003E0, rd, X.x0, rm });
        }
        pub fn csel(a: Self, rd: X, rn: X, rm: X, c: Cond) Error!void {
            if (!a.active()) return a.rawCsel(rd, rn, rm, c);
            return a.emit(.csel, .{ .rd = rd, .rn = rn, .rm = rm, .cond = c }, rawCsel, .{ rd, rn, rm, c });
        }
        /// cset Xd, c: Xd = 1 if c holds, else 0 (`al`/`nv` are not valid conditions here).
        pub fn cset(a: Self, rd: X, c: Cond) Error!void {
            if (c == .al or c == .nv) return error.InvalidOperand;
            if (!a.active()) return a.rawCset(rd, c);
            return a.emit(.cset, .{ .rd = rd, .cond = c }, rawCset, .{ rd, c });
        }
        /// lsl Xd, Xn, #s (= ubfm Xd, Xn, #(-s mod 64), #(63-s)).
        pub fn lslImm(a: Self, rd: X, rn: X, s: u6) Error!void {
            const immr: u6 = @truncate(@as(u7, 64) - s);
            if (!a.active()) return a.rawShiftImm(0xD3400000, rd, rn, immr, 63 - s);
            return a.emit(.lsl_imm, .{ .rd = rd, .rn = rn, .imm = s }, rawShiftImm, .{ 0xD3400000, rd, rn, immr, 63 - s });
        }
        /// lsr Xd, Xn, #s (= ubfm Xd, Xn, #s, #63).
        pub fn lsrImm(a: Self, rd: X, rn: X, s: u6) Error!void {
            if (!a.active()) return a.rawShiftImm(0xD3400000, rd, rn, s, 63);
            return a.emit(.lsr_imm, .{ .rd = rd, .rn = rn, .imm = s }, rawShiftImm, .{ 0xD3400000, rd, rn, s, 63 });
        }
        /// asr Xd, Xn, #s (= sbfm Xd, Xn, #s, #63).
        pub fn asrImm(a: Self, rd: X, rn: X, s: u6) Error!void {
            if (!a.active()) return a.rawShiftImm(0x93400000, rd, rn, s, 63);
            return a.emit(.asr_imm, .{ .rd = rd, .rn = rn, .imm = s }, rawShiftImm, .{ 0x93400000, rd, rn, s, 63 });
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

test "a64 data processing (LLVM words; matches the AsmJit-parity encoder)" {
    var cb = CodeBuffer.init(std.testing.allocator);
    defer cb.deinit();
    const a = Assembler.init(&cb);
    try a.sdiv(.x0, .x1, .x2);
    try a.udiv(.x3, .x4, .x5);
    try a.madd(.x0, .x1, .x2, .x3);
    try a.msub(.x4, .x5, .x6, .x7);
    try a.mneg(.x8, .x9, .x10);
    try a.smulh(.x11, .x12, .x13);
    try a.umulh(.x14, .x15, .x16);
    try a.neg(.x17, .x18);
    try a.and_(.x0, .x1, .x2);
    try a.orr(.x3, .x4, .x5);
    try a.eor(.x6, .x7, .x8);
    try a.csel(.x0, .x1, .x2, .lt);
    try a.cset(.x3, .eq);
    try a.cset(.x4, .hi);
    try a.lslImm(.x5, .x6, 3);
    try a.lslImm(.x5, .x6, 63);
    try a.lsrImm(.x7, .x8, 1);
    try a.asrImm(.x9, .x10, 63);
    // clang -c -target aarch64 (LLVM's assembler) for the same instructions.
    const want = [_]u32{ 0x9AC20C20, 0x9AC50883, 0x9B020C20, 0x9B069CA4, 0x9B0AFD28, 0x9B4D7D8B, 0x9BD07DEE, 0xCB1203F1, 0x8A020020, 0xAA050083, 0xCA0800E6, 0x9A82B020, 0x9A9F17E3, 0x9A9F97E4, 0xD37DF0C5, 0xD34100C5, 0xD341FD07, 0x937FFD49 };
    // The generic encoder (AsmJit test-suite parity) must agree word for word.
    const a64 = @import("a64.zig");
    const generic = [_]struct { a64.Mnemonic, []const a64.Op }{
        .{ .sdiv, &.{ a64.x(0), a64.x(1), a64.x(2) } },
        .{ .udiv, &.{ a64.x(3), a64.x(4), a64.x(5) } },
        .{ .madd, &.{ a64.x(0), a64.x(1), a64.x(2), a64.x(3) } },
        .{ .msub, &.{ a64.x(4), a64.x(5), a64.x(6), a64.x(7) } },
        .{ .mneg, &.{ a64.x(8), a64.x(9), a64.x(10) } },
        .{ .smulh, &.{ a64.x(11), a64.x(12), a64.x(13) } },
        .{ .umulh, &.{ a64.x(14), a64.x(15), a64.x(16) } },
        .{ .neg, &.{ a64.x(17), a64.x(18) } },
        .{ .@"and", &.{ a64.x(0), a64.x(1), a64.x(2) } },
        .{ .orr, &.{ a64.x(3), a64.x(4), a64.x(5) } },
        .{ .eor, &.{ a64.x(6), a64.x(7), a64.x(8) } },
        .{ .csel, &.{ a64.x(0), a64.x(1), a64.x(2), a64.cond(@backingInt(Cond.lt)) } },
        .{ .cset, &.{ a64.x(3), a64.cond(@backingInt(Cond.eq)) } },
        .{ .cset, &.{ a64.x(4), a64.cond(@backingInt(Cond.hi)) } },
        .{ .lsl, &.{ a64.x(5), a64.x(6), a64.imm(3) } },
        .{ .lsl, &.{ a64.x(5), a64.x(6), a64.imm(63) } },
        .{ .lsr, &.{ a64.x(7), a64.x(8), a64.imm(1) } },
        .{ .asr, &.{ a64.x(9), a64.x(10), a64.imm(63) } },
    };
    for (generic, want) |g, w| try std.testing.expectEqual(w, (a64.encodeMnIdx(g[0], g[1]) orelse return error.NoMatchingForm).word);
    try expectWords(&want, &cb);
}

test "a64 data processing operand checks" {
    var cb = CodeBuffer.init(std.testing.allocator);
    defer cb.deinit();
    const a = AssemblerT(.{ .validate = true }).init(&cb);
    try std.testing.expectError(error.InvalidOperand, a.sdiv(.zr_sp, .x1, .x2));
    try std.testing.expectError(error.InvalidOperand, a.madd(.x0, .x1, .x2, .zr_sp));
    try std.testing.expectError(error.InvalidOperand, a.and_(.zr_sp, .x1, .x2));
    try std.testing.expectError(error.InvalidOperand, a.cset(.x0, .al));
    try std.testing.expectEqual(@as(usize, 0), cb.len);
}
