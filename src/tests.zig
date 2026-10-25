//! Tests for the safety layer (validation + hooks) and label/branch encodings on both backends.
//! Byte-exact per-form encodings are in the generated `enc_tests.zig`.
const std = @import("std");
const opts = @import("zigjit_options");
const CodeBuffer = @import("core/code_buffer.zig").CodeBuffer;
const safety = @import("core/safety.zig");
const t = std.testing;

// ---- x86-64 -------------------------------------------------------------------------------

test "x86_64 validate: rejects illegal memory operands and labels" {
    if (!opts.x86_64) return error.SkipZigTest;
    const x = @import("x86_64/assembler.zig");
    const A = x.AssemblerT(.{ .validate = true });
    var cb = CodeBuffer.init(t.allocator);
    defer cb.deinit();
    const a = A.init(&cb);
    // rsp cannot be an index register (SIB index=100 means "none").
    try t.expectError(error.InvalidOperand, a.movRM(.rax, x.ptrIdx(.rdi, .rsp, .x1, 0)));
    try t.expectError(error.InvalidOperand, a.lea(.rax, x.Mem{ .index = .rsp }));
    try t.expectError(error.InvalidOperand, a.movMI(x.ptrIdx(.rax, .rsp, .x8, 0), 1));
    // scale without index
    try t.expectError(error.InvalidOperand, a.movRM(.rax, x.Mem{ .base = .rdi, .scale = .x4 }));
    // label that does not belong to this buffer
    try t.expectError(error.InvalidLabel, a.jmp(.{ .id = 7 }));
    try t.expectError(error.InvalidLabel, a.bind(.{ .id = 0 }));
    // nothing was emitted by rejected instructions
    try t.expectEqual(@as(usize, 0), cb.code().len);
    // r12 as index is legal (only rsp is special)
    try a.movRM(.rax, x.ptrIdx(.rdi, .r12, .x1, 0));
    try t.expectEqualSlices(u8, &.{ 0x4A, 0x8B, 0x04, 0x27 }, cb.code());
}

test "x86_64 validate off: same bytes as validate on for valid code, no checks for invalid" {
    if (!opts.x86_64) return error.SkipZigTest;
    const x = @import("x86_64/assembler.zig");
    var on = CodeBuffer.init(t.allocator);
    defer on.deinit();
    var off = CodeBuffer.init(t.allocator);
    defer off.deinit();
    inline for (.{ &on, &off }, .{ true, false }) |cb, v| {
        const a = x.AssemblerT(.{ .validate = v }).init(cb);
        const l = try a.newLabel();
        try a.bind(l);
        try a.movRM(.rax, x.ptrIdx(.rsp, .r13, .x8, -4));
        try a.aluRI(.sub, .rsp, 40);
        try a.jcc(.ne, l);
        try a.ret();
        try cb.finalize();
    }
    try t.expectEqualSlices(u8, on.code(), off.code());
    // With validation off the bad operand is encoded as-is (rsp index => "no index").
    const a = x.AssemblerT(.{}).init(&off);
    try a.movRM(.rax, x.ptrIdx(.rdi, .rsp, .x1, 0));
}

test "x86_64 branches: rel8/rel32 forward and backward" {
    if (!opts.x86_64) return error.SkipZigTest;
    const x = @import("x86_64/assembler.zig");
    var cb = CodeBuffer.init(t.allocator);
    defer cb.deinit();
    const a = x.AssemblerT(.{ .validate = true }).init(&cb);
    const top = try a.newLabel();
    const fwd = try a.newLabel();
    try a.bind(top);
    try a.call(fwd); // E8 rel32 (forward, fixup)
    try a.jmp(fwd); // E9 rel32 (forward always long)
    try a.jcc(.l, top); // 7C rel8 backward: 0 - (10+2) = -12
    for (0..130) |_| try a.nop();
    try a.jmp(top); // E9 rel32 backward: 0 - (142+5) = -147
    try a.jcc(.g, top); // 0F 8F rel32: 0 - (147+6) = -153
    try a.bind(fwd);
    try a.ret();
    try cb.finalize();
    const c = cb.code();
    try t.expectEqualSlices(u8, &.{ 0xE8, 148, 0, 0, 0 }, c[0..5]); // 153 - 5
    try t.expectEqualSlices(u8, &.{ 0xE9, 143, 0, 0, 0 }, c[5..10]); // 153 - 10
    try t.expectEqualSlices(u8, &.{ 0x7C, 0xF4 }, c[10..12]);
    try t.expectEqualSlices(u8, &.{ 0xE9, 0x6D, 0xFF, 0xFF, 0xFF }, c[142..147]);
    try t.expectEqualSlices(u8, &.{ 0x0F, 0x8F, 0x67, 0xFF, 0xFF, 0xFF }, c[147..153]);
    try t.expectEqual(@as(u8, 0xC3), c[153]);
}

test "x86_64 callAbs = mov r11, imm; call r11" {
    if (!opts.x86_64) return error.SkipZigTest;
    const x = @import("x86_64/assembler.zig");
    var cb = CodeBuffer.init(t.allocator);
    defer cb.deinit();
    try x.AssemblerT(.{}).init(&cb).callAbs(0x1122334455);
    try t.expectEqualSlices(u8, &.{ 0x49, 0xBB, 0x55, 0x44, 0x33, 0x22, 0x11, 0, 0, 0, 0x41, 0xFF, 0xD3 }, cb.code());
}

test "x86_64 hooks: counting, logging, veto, runtime callback" {
    if (!opts.x86_64) return error.SkipZigTest;
    const x = @import("x86_64/assembler.zig");
    var cb = CodeBuffer.init(t.allocator);
    defer cb.deinit();

    var count: x.CountingHooks = .{};
    const ac = x.AssemblerT(.{ .Hooks = x.CountingHooks }).initHooks(&cb, &count);
    try ac.movRR(.rax, .rdi);
    try ac.addRI(.rax, 1);
    try ac.callAbs(0x1000); // two instructions
    try ac.ret();
    try t.expectEqual(@as(usize, 5), count.instructions);
    try t.expectEqual(cb.code().len, count.bytes);

    cb.reset();
    var log: x.LogHooks(256) = .{};
    const al = x.AssemblerT(.{ .Hooks = x.LogHooks(256) }).initHooks(&cb, &log);
    try al.movRR(.rax, .rcx);
    try al.ret();
    try t.expectEqualStrings("0000  mov_rr   48 89 c8\n0003  ret      c3\n", log.text());

    // A `pre` hook can inspect operands and veto emission.
    const NoR15 = struct {
        seen: usize = 0,
        pub fn pre(self: *@This(), ev: x.PreEvent) x.Error!void {
            self.seen += 1;
            if (ev.ops.r0 == .r15 or ev.ops.r1 == .r15) return error.InvalidOperand;
        }
    };
    cb.reset();
    var veto: NoR15 = .{};
    const av = x.AssemblerT(.{ .Hooks = NoR15 }).initHooks(&cb, &veto);
    try av.push(.rbx);
    try t.expectError(error.InvalidOperand, av.push(.r15));
    try t.expectEqual(@as(usize, 2), veto.seen);
    try t.expectEqualSlices(u8, &.{0x53}, cb.code());

    // Runtime callback adapter, with an event carrying id/offset/bytes/ops.
    const Rec = struct {
        var last: ?x.Event = null;
        var n: usize = 0;
        fn cb_(_: ?*anyopaque, ev: x.Event) void {
            last = ev;
            n += 1;
        }
    };
    cb.reset();
    var rt: x.CallbackHooks = .{ .post_fn = Rec.cb_ };
    const ar = x.AssemblerT(.{ .Hooks = x.CallbackHooks }).initHooks(&cb, &rt);
    try ar.nop();
    try ar.aluRI(.cmp, .rcx, 5);
    try t.expectEqual(@as(usize, 2), Rec.n);
    try t.expectEqual(x.Inst.alu_ri, Rec.last.?.id);
    try t.expectEqual(x.Alu.cmp, Rec.last.?.ops.alu.?);
    try t.expectEqual(@as(u32, 1), Rec.last.?.offset);
    try t.expectEqualSlices(u8, &.{ 0x48, 0x83, 0xF9, 0x05 }, Rec.last.?.bytes);

    // Hook type selected but no instance attached: plain emission.
    cb.reset();
    const an = x.AssemblerT(.{ .Hooks = x.CallbackHooks }).init(&cb);
    try an.ret();
    try t.expectEqualSlices(u8, &.{0xC3}, cb.code());
}

test "x86_64 hooks off: identical output to hooks on" {
    if (!opts.x86_64) return error.SkipZigTest;
    const x = @import("x86_64/assembler.zig");
    var a_cb = CodeBuffer.init(t.allocator);
    defer a_cb.deinit();
    var b_cb = CodeBuffer.init(t.allocator);
    defer b_cb.deinit();
    var count: x.CountingHooks = .{};
    const plain = x.AssemblerT(.{}).init(&a_cb);
    const hooked = x.AssemblerT(.{ .validate = true, .Hooks = x.CountingHooks }).initHooks(&b_cb, &count);
    inline for (.{ plain, hooked }) |a| {
        try a.push(.rbp);
        try a.movRR(.rbp, .rsp);
        try a.movRI(.rax, -5);
        try a.imulRRI(.rax, .rax, 1000);
        try a.movMR(x.ptr(.rbp, -8), .rax);
        try a.pop(.rbp);
        try a.ret();
    }
    try t.expectEqualSlices(u8, a_cb.code(), b_cb.code());
    try t.expectEqual(@as(usize, 7), count.instructions);
    // Disabled layers add no state to the assembler.
    try t.expectEqual(@sizeOf(*CodeBuffer), @sizeOf(x.AssemblerT(.{})));
}

// ---- AArch64 ------------------------------------------------------------------------------

test "aarch64 validate: register 31 rules, labels, branch range" {
    if (!opts.aarch64) return error.SkipZigTest;
    const m = @import("aarch64/assembler.zig");
    var cb = CodeBuffer.init(t.allocator);
    defer cb.deinit();
    const a = m.AssemblerT(.{ .validate = true }).init(&cb);
    // x31 is XZR in register forms: almost always a bug when SP was meant.
    try t.expectError(error.InvalidOperand, a.add(.zr_sp, .x0, .x1));
    try t.expectError(error.InvalidOperand, a.add(.x0, .zr_sp, .x1));
    try t.expectError(error.InvalidOperand, a.mov(.zr_sp, .x0));
    try t.expectError(error.InvalidOperand, a.movz(.zr_sp, 1, 0));
    try t.expectError(error.InvalidOperand, a.ldr(.zr_sp, .x0, 0));
    try t.expectError(error.InvalidOperand, a.blr(.zr_sp));
    try t.expectError(error.InvalidLabel, a.b(.{ .id = 3 }));
    try t.expectEqual(@as(usize, 0), cb.code().len);
    // SP-allowed forms
    try a.subImm(.zr_sp, .zr_sp, 16);
    try a.str(.x30, .zr_sp, 8);
    try a.mov(.x0, .zr_sp); // mov x0, xzr
    // misaligned offset is always rejected (encoder check)
    try t.expectError(error.InvalidImmediate, a.ldr(.x0, .x1, 4));

    // Bound label beyond imm19 range (±1 MiB) for b.cond; b (imm26) still fits.
    cb.reset();
    const top = try a.newLabel();
    try a.bind(top);
    for (0..(1 << 18) + 1) |_| try a.word(0xD503201F); // nop
    try t.expectError(error.DisplacementOutOfRange, a.bcond(.eq, top));
    try t.expectError(error.DisplacementOutOfRange, a.cbz(.x0, top));
    try a.b(top);
}

test "aarch64 branches: forward/backward imm26/imm19" {
    if (!opts.aarch64) return error.SkipZigTest;
    const m = @import("aarch64/assembler.zig");
    var cb = CodeBuffer.init(t.allocator);
    defer cb.deinit();
    const a = m.AssemblerT(.{ .validate = true }).init(&cb);
    const top = try a.newLabel();
    const out = try a.newLabel();
    try a.bind(top);
    try a.bcond(.ne, out); // +4 words
    try a.cbnz(.x3, out); // +3
    try a.bl(out); // +2
    try a.b(top); // -3
    try a.bind(out);
    try a.ret();
    try cb.finalize();
    const want = [_]u32{ 0x54000081, 0xB5000063, 0x94000002, 0x17FFFFFD, 0xD65F03C0 };
    for (want, 0..) |w, i| try t.expectEqual(w, std.mem.readInt(u32, cb.code()[i * 4 ..][0..4], .little));
}

test "aarch64 movImm expands to movz + non-zero movk" {
    if (!opts.aarch64) return error.SkipZigTest;
    const m = @import("aarch64/assembler.zig");
    var cb = CodeBuffer.init(t.allocator);
    defer cb.deinit();
    var count: m.CountingHooks = .{};
    const a = m.AssemblerT(.{ .Hooks = m.CountingHooks }).initHooks(&cb, &count);
    try a.movImm(.x0, 0x1234_0000_0000_5678);
    try t.expectEqual(@as(usize, 2), count.instructions); // movz #0x5678; movk #0x1234, lsl 48
    try t.expectEqual(@as(u32, 0xD28ACF00), std.mem.readInt(u32, cb.code()[0..4], .little));
    try t.expectEqual(@as(u32, 0xF2E24680), std.mem.readInt(u32, cb.code()[4..8], .little));
}

test "aarch64 hooks: log + identical output when off" {
    if (!opts.aarch64) return error.SkipZigTest;
    const m = @import("aarch64/assembler.zig");
    var cb = CodeBuffer.init(t.allocator);
    defer cb.deinit();
    var log: m.LogHooks(128) = .{};
    const a = m.AssemblerT(.{ .validate = true, .Hooks = m.LogHooks(128) }).initHooks(&cb, &log);
    try a.add(.x0, .x0, .x1);
    try a.ret();
    try t.expectEqualStrings("0000  add      00 00 01 8b\n0004  ret      c0 03 5f d6\n", log.text());
    var cb2 = CodeBuffer.init(t.allocator);
    defer cb2.deinit();
    const p = m.AssemblerT(.{}).init(&cb2);
    try p.add(.x0, .x0, .x1);
    try p.ret();
    try t.expectEqualSlices(u8, cb.code(), cb2.code());
    try t.expectEqual(@sizeOf(*CodeBuffer), @sizeOf(m.AssemblerT(.{})));
}

test "LogHooks truncates instead of overflowing" {
    var log: safety.LogHooks(8) = .{};
    const Ev = struct { id: enum { nop }, offset: u32, bytes: []const u8 };
    log.post(Ev{ .id = .nop, .offset = 0, .bytes = &.{ 0x90, 0x90, 0x90 } });
    try t.expect(log.truncated);
    try t.expect(log.len <= 8);
}

test "default options: hooks on in every mode, validation only in Debug/ReleaseSafe" {
    const builtin = @import("builtin");
    // Holds for builds without explicit -Dvalidate/-Dhooks (the test step's defaults).
    const safe = builtin.mode == .Debug or builtin.mode == .ReleaseSafe;
    if (opts.validate != safe or !opts.hooks) return error.SkipZigTest; // overridden on the command line
    if (opts.x86_64) {
        const x = @import("x86_64/assembler.zig");
        try t.expect(x.default_options.Hooks == x.CallbackHooks);
        try t.expectEqual(safe, x.default_options.validate);
        // hooks compiled in but nothing attached: plain emission
        var cb = CodeBuffer.init(t.allocator);
        defer cb.deinit();
        try x.Assembler.init(&cb).ret();
        try t.expectEqualSlices(u8, &.{0xC3}, cb.code());
    }
    if (opts.aarch64) {
        const m = @import("aarch64/assembler.zig");
        try t.expect(m.default_options.Hooks == m.CallbackHooks);
        try t.expectEqual(safe, m.default_options.validate);
    }
}
