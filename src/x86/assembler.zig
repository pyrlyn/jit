//! x86 (32-bit) Assembler (TODO-ARCHES.md §1). Typed API over the shared table-driven GP
//! encoder (`x86_64/gp.zig`) run in 32-bit mode: no REX, eax..edi only, 32-bit ModRM/SIB,
//! absolute disp32 (no RIP-relative), one-byte `inc/dec r32`, moffs32 `mov eax, [abs]`.
//! Branches and labels reuse `CodeBuffer` fixups (rel8/rel32 have the same encoding as x86-64).
//!
//! The GP encoder selects the mode through the global `gp.mode32`; every emitter sets it for
//! the duration of one instruction and restores it (not thread-safe across concurrent 64-bit
//! and 32-bit emission on different threads; a comptime mode parameter is the follow-up).
const std = @import("std");
const core = @import("../core/code_buffer.zig");
const x64 = @import("../x86_64/assembler.zig");
pub const gp = @import("../x86_64/gp.zig");
const CodeBuffer = core.CodeBuffer;
pub const Label = core.Label;
pub const Error = core.Error || gp.Error || error{InvalidOperand};

pub const Gp = enum(u3) {
    eax,
    ecx,
    edx,
    ebx,
    esp,
    ebp,
    esi,
    edi,
    inline fn wide(r: Gp) x64.Gp {
        return @enumFromInt(@intFromEnum(r));
    }
};
pub const Scale = x64.Scale;
pub const Cond = x64.Cond;
pub const Alu = x64.Alu;

/// [base + index*scale + disp]; `base == null and index == null` => absolute [disp32].
pub const Mem = struct {
    base: ?Gp = null,
    index: ?Gp = null,
    scale: Scale = .x1,
    disp: i32 = 0,
    fn wide(m: Mem) x64.Mem {
        return .{ .base = if (m.base) |b| b.wide() else null, .index = if (m.index) |i| i.wide() else null, .scale = m.scale, .disp = m.disp };
    }
};
pub inline fn ptr(base: Gp, disp: i32) Mem {
    return .{ .base = base, .disp = disp };
}
pub inline fn ptrIdx(base: Gp, index: Gp, scale: Scale, disp: i32) Mem {
    return .{ .base = base, .index = index, .scale = scale, .disp = disp };
}
pub inline fn abs(addr: u32) Mem {
    return .{ .disp = @bitCast(addr) };
}

/// 32-bit calling conventions (argument registers for fastcall, who pops the arguments).
pub const CallConv = struct {
    reg_args: []const Gp = &.{},
    callee_pops: bool = false,
    callee_saved: []const Gp = &.{ .ebx, .esi, .edi, .ebp },
    ret: Gp = .eax,
    /// Stack argument i (after `push ebp; mov ebp, esp`): [ebp + 8 + 4*i].
    pub fn stackArg(_: CallConv, i: u32) Mem {
        return ptr(.ebp, @intCast(8 + 4 * i));
    }
    pub const cdecl: CallConv = .{};
    pub const sysv_i386: CallConv = .{};
    pub const stdcall: CallConv = .{ .callee_pops = true };
    pub const fastcall: CallConv = .{ .reg_args = &.{ .ecx, .edx }, .callee_pops = true };
};

pub const Options = struct { validate: bool = true };

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

        fn checkMem(m: Mem) Error!void {
            if (!opts.validate) return;
            if (m.index == .esp) return error.InvalidOperand; // SIB index 100 = none
            if (m.index == null and m.scale != .x1) return error.InvalidOperand;
        }

        /// Any GP form of the shared table in 32-bit mode (8/16/32-bit registers, sized
        /// memory, immediates): `a.inst(.movzx, .{ gp.r32(.rax), gp.byte(...) })`.
        /// 64-bit registers and REX-only registers are rejected by the encoder.
        pub fn inst(a: Self, comptime mn: gp.Mnemonic, args: anytype) Error!void {
            const prev = gp.mode32;
            gp.mode32 = true;
            defer gp.mode32 = prev;
            return gp.emit(a.buf, mn, args);
        }
        inline fn r(g: Gp) gp.Reg {
            return gp.r32(g.wide());
        }
        inline fn dw(m: Mem) gp.Mem {
            return gp.dword(m.wide());
        }

        pub fn movRR(a: Self, d: Gp, s: Gp) Error!void {
            return a.inst(.mov, .{ r(d), r(s) });
        }
        pub fn movRM(a: Self, d: Gp, s: Mem) Error!void {
            try checkMem(s);
            return a.inst(.mov, .{ r(d), dw(s) });
        }
        pub fn movMR(a: Self, d: Mem, s: Gp) Error!void {
            try checkMem(d);
            return a.inst(.mov, .{ dw(d), r(s) });
        }
        pub fn movMI(a: Self, d: Mem, i: i32) Error!void {
            try checkMem(d);
            return a.inst(.mov, .{ dw(d), @as(i64, i) });
        }
        /// `mov r32, imm32` (B8+r). Accepts i32 or u32 range.
        pub fn movRI(a: Self, d: Gp, i: i64) Error!void {
            if (i < std.math.minInt(i32) or i > std.math.maxInt(u32)) return error.InvalidOperand;
            const v: u32 = @truncate(@as(u64, @bitCast(i)));
            try a.buf.ensure(5);
            a.buf.put8(0xB8 + @as(u8, @intFromEnum(d)));
            a.buf.put32(v);
        }
        pub fn lea(a: Self, d: Gp, s: Mem) Error!void {
            try checkMem(s);
            return a.inst(.lea, .{ r(d), gp.mem(s.wide()) });
        }
        pub fn aluRR(a: Self, comptime op: Alu, d: Gp, s: Gp) Error!void {
            return a.inst(@field(gp.Mnemonic, @tagName(op)), .{ r(d), r(s) });
        }
        pub fn aluRM(a: Self, comptime op: Alu, d: Gp, s: Mem) Error!void {
            try checkMem(s);
            return a.inst(@field(gp.Mnemonic, @tagName(op)), .{ r(d), dw(s) });
        }
        pub fn aluMR(a: Self, comptime op: Alu, d: Mem, s: Gp) Error!void {
            try checkMem(d);
            return a.inst(@field(gp.Mnemonic, @tagName(op)), .{ dw(d), r(s) });
        }
        pub fn aluRI(a: Self, comptime op: Alu, d: Gp, i: i32) Error!void {
            return a.inst(@field(gp.Mnemonic, @tagName(op)), .{ r(d), @as(i64, i) });
        }
        pub fn addRR(a: Self, d: Gp, s: Gp) Error!void {
            return a.aluRR(.add, d, s);
        }
        pub fn addRI(a: Self, d: Gp, i: i32) Error!void {
            return a.aluRI(.add, d, i);
        }
        pub fn subRI(a: Self, d: Gp, i: i32) Error!void {
            return a.aluRI(.sub, d, i);
        }
        pub fn cmpRR(a: Self, d: Gp, s: Gp) Error!void {
            return a.aluRR(.cmp, d, s);
        }
        pub fn testRR(a: Self, d: Gp, s: Gp) Error!void {
            return a.inst(.@"test", .{ r(d), r(s) });
        }
        pub fn imulRR(a: Self, d: Gp, s: Gp) Error!void {
            return a.inst(.imul, .{ r(d), r(s) });
        }
        pub fn imulRRI(a: Self, d: Gp, s: Gp, i: i32) Error!void {
            return a.inst(.imul, .{ r(d), r(s), @as(i64, i) });
        }
        /// One-byte `inc r32` (0x40+r), legal only in 32-bit mode.
        pub fn inc(a: Self, d: Gp) Error!void {
            try a.buf.emitBytes(&.{0x40 + @as(u8, @intFromEnum(d))});
        }
        pub fn dec(a: Self, d: Gp) Error!void {
            try a.buf.emitBytes(&.{0x48 + @as(u8, @intFromEnum(d))});
        }
        pub fn push(a: Self, g: Gp) Error!void {
            try a.buf.emitBytes(&.{0x50 + @as(u8, @intFromEnum(g))});
        }
        pub fn pop(a: Self, g: Gp) Error!void {
            try a.buf.emitBytes(&.{0x58 + @as(u8, @intFromEnum(g))});
        }
        pub fn pushI(a: Self, i: i32) Error!void {
            if (std.math.cast(i8, i)) |b| return a.buf.emitBytes(&.{ 0x6A, @bitCast(b) });
            var e: [5]u8 = .{ 0x68, 0, 0, 0, 0 };
            std.mem.writeInt(i32, e[1..5], i, .little);
            return a.buf.emitBytes(&e);
        }
        pub fn ret(a: Self) Error!void {
            return a.buf.emitBytes(&.{0xC3});
        }
        /// `ret imm16` (callee pops `n` bytes: stdcall/fastcall).
        pub fn retN(a: Self, n: u16) Error!void {
            if (n == 0) return a.ret();
            return a.buf.emitBytes(&.{ 0xC2, @truncate(n), @truncate(n >> 8) });
        }
        pub fn nop(a: Self) Error!void {
            return a.buf.emitBytes(&.{0x90});
        }
        pub fn int3(a: Self) Error!void {
            return a.buf.emitBytes(&.{0xCC});
        }
        pub fn callR(a: Self, g: Gp) Error!void {
            return a.buf.emitBytes(&.{ 0xFF, 0xD0 + @as(u8, @intFromEnum(g)) });
        }
        pub fn jmpR(a: Self, g: Gp) Error!void {
            return a.buf.emitBytes(&.{ 0xFF, 0xE0 + @as(u8, @intFromEnum(g)) });
        }
        fn branch(a: Self, short_op: []const u8, long_op: []const u8, l: Label) Error!void {
            if (!a.buf.isValidLabel(l)) return error.InvalidLabel;
            const start: i64 = a.buf.offset();
            if (a.buf.labelOffset(l)) |target| {
                const d8 = target - (start + @as(i64, @intCast(short_op.len)) + 1);
                if (short_op.len != 0 and std.math.cast(i8, d8) != null) {
                    try a.buf.emitBytes(short_op);
                    return a.buf.emitBytes(&.{@bitCast(@as(i8, @intCast(d8)))});
                }
                try a.buf.emitBytes(long_op);
                var d: [4]u8 = undefined;
                std.mem.writeInt(i32, &d, @intCast(target - (start + @as(i64, @intCast(long_op.len)) + 4)), .little);
                return a.buf.emitBytes(&d);
            }
            try a.buf.emitBytes(long_op);
            const at = a.buf.offset();
            try a.buf.emitBytes(&.{ 0, 0, 0, 0 });
            try a.buf.addFixup(.{ .at = at, .base = at + 4, .label = l.id, .kind = .rel32 });
        }
        pub fn jmp(a: Self, l: Label) Error!void {
            return a.branch(&.{0xEB}, &.{0xE9}, l);
        }
        pub fn call(a: Self, l: Label) Error!void {
            return a.branch(&.{}, &.{0xE8}, l);
        }
        pub fn jcc(a: Self, c: Cond, l: Label) Error!void {
            const cc: u8 = @intFromEnum(c);
            return a.branch(&.{0x70 + cc}, &.{ 0x0F, 0x80 + cc }, l);
        }
        /// `mov eax, addr; call eax` (eax is caller-saved in every 32-bit convention).
        pub fn callAbs(a: Self, addr: u32) Error!void {
            try a.movRI(.eax, addr);
            try a.callR(.eax);
        }
        /// `push ebp; mov ebp, esp; push <saved>; sub esp, n` with esp 16-byte aligned at
        /// call sites (SysV i386 / modern cdecl: esp ≡ 12 mod 16 on entry). Returns `n`.
        pub fn prolog(a: Self, saved: []const Gp, locals: u32) Error!i32 {
            try a.push(.ebp);
            try a.movRR(.ebp, .esp);
            for (saved) |g| try a.push(g);
            const used: u32 = 8 + @as(u32, @intCast(saved.len)) * 4; // ret addr + ebp + saved
            const adj = std.mem.alignForward(u32, used + locals, 16) - used;
            if (adj != 0) try a.subRI(.esp, @intCast(adj));
            return @intCast(adj);
        }
        /// Matching epilog; `cc.callee_pops` emits `ret 4*stack_args`.
        pub fn epilog(a: Self, saved: []const Gp, adj: i32, cc: CallConv, stack_args: u16) Error!void {
            if (adj != 0) try a.addRI(.esp, adj);
            var i = saved.len;
            while (i > 0) {
                i -= 1;
                try a.pop(saved[i]);
            }
            try a.pop(.ebp);
            if (cc.callee_pops) try a.retN(stack_args * 4) else try a.ret();
        }
    };
}
pub const Assembler = AssemblerT(.{});

// ---- tests: expected bytes from `clang -target i386-linux-gnu` + objdump --------------
const t = std.testing;

test "x86: byte-exact vs LLVM (clang -target i386)" {
    var cb = CodeBuffer.init(t.allocator);
    defer cb.deinit();
    const a = Assembler.init(&cb);
    const mode_before = gp.mode32; // other test files leave the global set
    try a.movRR(.eax, .ecx);
    try a.movRM(.edi, ptr(.esp, 4));
    try a.movMR(ptr(.ebp, -8), .esi);
    try a.movMI(ptrIdx(.ebx, .eax, .x4, 16), 7);
    try a.lea(.edx, ptrIdx(.ecx, .edx, .x2, 100));
    try a.movRI(.eax, 42);
    try a.movRI(.ecx, -1);
    try a.addRR(.eax, .ecx);
    try a.subRI(.esp, 12);
    try a.addRI(.eax, 1000);
    try a.aluRM(.cmp, .edx, ptr(.ebp, 8));
    try a.aluRR(.xor, .ebx, .ebx);
    try a.aluMR(.@"and", ptr(.eax, 0), .edx);
    try a.testRR(.eax, .eax);
    try a.imulRR(.ecx, .edx);
    try a.imulRRI(.eax, .ecx, 10);
    try a.inc(.eax);
    try a.dec(.edi);
    try a.push(.ebp);
    try a.pop(.ebp);
    try a.pushI(1);
    try a.pushI(1000);
    try a.ret();
    try a.retN(8);
    try a.nop();
    try a.int3();
    try a.callR(.eax);
    try a.jmpR(.ecx);
    try a.movRM(.eax, abs(0x1000));
    try a.movRM(.ecx, abs(0x1000));
    const want = [_]u8{
        0x89, 0xc8, 0x8b, 0x7c, 0x24, 0x04, 0x89, 0x75, 0xf8, 0xc7, 0x44, 0x83, 0x10, 0x07, 0x00, 0x00, 0x00,
        0x8d, 0x54, 0x51, 0x64, 0xb8, 0x2a, 0x00, 0x00, 0x00, 0xb9, 0xff, 0xff, 0xff, 0xff, 0x01, 0xc8,
        0x83, 0xec, 0x0c, 0x05, 0xe8, 0x03, 0x00, 0x00, 0x3b, 0x55, 0x08, 0x31, 0xdb, 0x21, 0x10, 0x85,
        0xc0, 0x0f, 0xaf, 0xca, 0x6b, 0xc1, 0x0a, 0x40, 0x4f, 0x55, 0x5d, 0x6a, 0x01, 0x68, 0xe8, 0x03,
        0x00, 0x00, 0xc3, 0xc2, 0x08, 0x00, 0x90, 0xcc, 0xff, 0xd0, 0xff, 0xe1, 0xa1, 0x00, 0x10, 0x00,
        0x00, 0x8b, 0x0d, 0x00, 0x10, 0x00, 0x00,
    };
    try t.expectEqualSlices(u8, &want, cb.code());
    try t.expectEqual(mode_before, gp.mode32); // mode restored
}

test "x86: branches rel8/rel32, labels, validation" {
    var cb = CodeBuffer.init(t.allocator);
    defer cb.deinit();
    const a = Assembler.init(&cb);
    const top = try a.newLabel();
    const fwd = try a.newLabel();
    try a.bind(top);
    try a.jcc(.ne, top); // 75 FE
    try a.jmp(fwd); // E9 rel32 (forward)
    try a.call(top); // E8 rel32 backward
    try a.bind(fwd);
    try a.ret();
    try cb.finalize();
    try t.expectEqualSlices(u8, &.{ 0x75, 0xFE, 0xE9, 0x05, 0, 0, 0, 0xE8, 0xF4, 0xFF, 0xFF, 0xFF, 0xC3 }, cb.code());
    try t.expectError(error.InvalidOperand, a.movRM(.eax, ptrIdx(.ebx, .esp, .x1, 0)));
    try t.expectError(error.InvalidOperand, a.movRI(.eax, 1 << 33));
    // 64-bit registers are not encodable in 32-bit mode.
    try t.expectError(error.NoMatchingForm, a.inst(.mov, .{ gp.r64(.rax), gp.r64(.rcx) }));
    try t.expectError(error.InvalidLabel, a.jmp(.{ .id = 99 }));
}

test "x86: stdcall prolog/epilog and generic inst path" {
    var cb = CodeBuffer.init(t.allocator);
    defer cb.deinit();
    const a = Assembler.init(&cb);
    const adj = try a.prolog(&.{.ebx}, 4);
    try a.movRM(.eax, CallConv.stdcall.stackArg(0));
    try a.inst(.movzx, .{ gp.r32(.rcx), gp.byte(x64.ptr(.rax, 0)) });
    try a.epilog(&.{.ebx}, adj, CallConv.stdcall, 2);
    // push ebp; mov ebp,esp; push ebx; sub esp,4; mov eax,[ebp+8]; movzx ecx, byte [eax];
    // add esp,4; pop ebx; pop ebp; ret 8
    try t.expectEqualSlices(u8, &.{ 0x55, 0x89, 0xE5, 0x53, 0x83, 0xEC, 0x04, 0x8B, 0x45, 0x08, 0x0F, 0xB6, 0x08, 0x83, 0xC4, 0x04, 0x5B, 0x5D, 0xC2, 0x08, 0x00 }, cb.code());
}
