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

// ---- ISA database (AsmJit db) driven validation ------------------------------------------

test "x86_64 isa: every instruction has AsmJit forms; matchForm accepts/rejects" {
    if (!opts.x86_64) return error.SkipZigTest;
    const x = @import("x86_64/assembler.zig");
    inline for (@typeInfo(x.Inst).@"enum".fields) |f| {
        const id: x.Inst = @enumFromInt(f.value);
        const ops: x.Ops = .{ .alu = .add, .cond = .e };
        try t.expect(x.formsFor(id, ops).len > 0);
        try t.expect(x.requiredFeatures(id).count() == 0); // all current forms are baseline x86-64
    }
    const push = x.isa.table.push[0];
    try t.expect(x.matchForm(push, &.{0x53})); // push rbx
    try t.expect(x.matchForm(push, &.{ 0x41, 0x57 })); // push r15
    try t.expect(!x.matchForm(push, &.{0x5B})); // pop rbx
    const add_i8 = x.isa.table.alu_ri[0][0]; // add r/m64, imms8 (83 /0 ib)
    try t.expect(x.matchForm(add_i8, &.{ 0x48, 0x83, 0xC0, 0x01 }));
    try t.expect(!x.matchForm(add_i8, &.{ 0x48, 0x83, 0xE8, 0x01 })); // /5 = sub
    try t.expect(!x.matchForm(x.isa.table.jmp[1], &.{ 0xE9, 0, 0 })); // rel32 too short
    try t.expectEqualStrings("add x:rv/mv, imms8", add_i8.asmjit);
}

test "x86_64 cpu features: set check, target option, host detection" {
    if (!opts.x86_64) return error.SkipZigTest;
    const x = @import("x86_64/assembler.zig");
    const builtin = @import("builtin");
    const need = x.isa.featureSet(&.{ .AVX2, .BMI2 });
    try t.expectError(error.UnsupportedCpuFeature, x.checkFeatureSet(need, x.isa.featureSet(&.{.AVX2})));
    try x.checkFeatureSet(need, x.isa.featureSet(&.{ .AVX2, .BMI2, .SSE2 }));

    var cb = CodeBuffer.init(t.allocator);
    defer cb.deinit();
    // Comptime target check: baseline forms need nothing, so this compiles and emits.
    const At = x.AssemblerT(.{ .validate = true, .cpu_check = .target, .target_features = &.{"SSE2"} });
    try At.init(&cb).movRR(.rax, .rcx);
    try t.expectEqualSlices(u8, &.{ 0x48, 0x89, 0xC8 }, cb.code());

    const Ah = x.AssemblerT(.{ .validate = true, .cpu_check = .host });
    if (builtin.cpu.arch == .x86_64) {
        const h = x.cpu.host().?;
        try t.expect(h.contains(.SSE2)); // baseline x86-64 (also under Rosetta)
        try Ah.init(&cb).ret();
    } else {
        try t.expect(x.cpu.host() == null);
        try t.expectError(error.UnsupportedCpuFeature, Ah.init(&cb).ret());
    }
}

test "aarch64 isa: forms match the encoder, DB register-31 rules, host features" {
    if (!opts.aarch64) return error.SkipZigTest;
    const m = @import("aarch64/assembler.zig");
    const builtin = @import("builtin");
    try t.expect(m.isa.table.add_imm.rd_sp and m.isa.table.add_imm.rn_sp);
    try t.expect(!m.isa.table.add.rd_sp and !m.isa.table.add.rn_sp);
    try t.expect(m.isa.table.ldr.rn_sp and !m.isa.table.ldr.rd_sp);
    try t.expectEqual(@as(u8, 12), m.isa.table.ldr.imm_bits);
    try t.expectEqual(@as(u8, 8), m.isa.table.ldr.imm_scale);
    try t.expectEqual(@as(u8, 19), m.isa.table.cbz.imm_bits);
    try t.expectEqual(@as(u32, 0xB4000000), m.isa.table.cbz.value); // DB fix-up (see generator)
    try t.expectEqual(@as(u32, 0xD65F0000), m.isa.table.ret.value); // DB fix-up
    if (builtin.cpu.arch == .aarch64 and builtin.os.tag == .macos) {
        try t.expect(m.cpu.host().?.contains(.ASIMD));
        var cb = CodeBuffer.init(t.allocator);
        defer cb.deinit();
        try m.AssemblerT(.{ .validate = true, .cpu_check = .host }).init(&cb).ret();
    } else try t.expect(m.cpu.host() == null);
    try t.expectError(error.UnsupportedCpuFeature, m.checkFeatureSet(m.isa.featureSet(&.{.SVE2}), m.isa.featureSet(&.{.ASIMD})));
}

// ---- finalize: offset overflow checks (always on) ------------------------------------------

test "finalize: out-of-range fixups are errors and leave the code untouched" {
    var cb = CodeBuffer.init(t.allocator);
    defer cb.deinit();
    const l = try cb.newLabel();
    try cb.emitBytes(&.{ 0xEB, 0x00 }); // jmp rel8 with a fixup to a far label
    try cb.addFixup(.{ .at = 1, .base = 2, .label = l.id, .kind = .rel8 });
    try cb.emitBytes(&([_]u8{0x90} ** 200));
    try cb.bind(l);
    try t.expectError(error.DisplacementOutOfRange, cb.finalize());
    try t.expectEqual(@as(u8, 0x00), cb.code()[1]); // not patched
    // Code size limit (2 GiB): checked before any allocation.
    try t.expectError(error.CodeTooLarge, cb.ensure(@import("core/code_buffer.zig").max_code_size));
}

test "x86_64 finalize: forward branch to an unbound label; label errors" {
    if (!opts.x86_64) return error.SkipZigTest;
    const x = @import("x86_64/assembler.zig");
    var cb = CodeBuffer.init(t.allocator);
    defer cb.deinit();
    const a = x.AssemblerT(.{}).init(&cb); // no validate: these checks are always on
    const l0 = try a.newLabel();
    const l1 = try a.newLabel();
    const l2 = try a.newLabel();
    try a.jmp(l1);
    try a.jcc(.e, l2);
    try a.call(l1);
    try a.bind(l0);
    try t.expectError(error.LabelAlreadyBound, a.bind(l0));
    try t.expectError(error.InvalidLabel, a.bind(.{ .id = 99 }));
    try t.expectError(error.InvalidLabel, a.jmp(.{ .id = 99 }));
    try t.expectError(error.UnboundLabel, cb.finalize());
    var ids: [8]u32 = undefined;
    try t.expectEqualSlices(u32, &.{ l1.id, l2.id }, cb.unresolvedLabels(&ids));
    try a.bind(l1);
    try a.bind(l2);
    try cb.finalize();
}

test "aarch64 finalize and branches: imm19 range forward/backward, labels (no validate)" {
    if (!opts.aarch64) return error.SkipZigTest;
    const m = @import("aarch64/assembler.zig");
    var cb = CodeBuffer.init(t.allocator);
    defer cb.deinit();
    const a = m.AssemblerT(.{}).init(&cb);
    const top = try a.newLabel();
    try a.bind(top);
    const fwd = try a.newLabel();
    try a.cbz(.x0, fwd);
    for (0..(1 << 18)) |_| try a.word(0xD503201F);
    try a.bind(fwd);
    try t.expectError(error.DisplacementOutOfRange, cb.finalize()); // +2^18 words: imm19 overflow
    try t.expectEqual(@as(u32, 0xB4000000), std.mem.readInt(u32, cb.code()[0..4], .little)); // untouched
    // backward: rejected at emit time even without validation (never truncated)
    try t.expectError(error.DisplacementOutOfRange, a.bcond(.eq, top));
    try a.b(top); // imm26 still fits
    cb.reset();
    const l = try a.newLabel();
    try a.b(l);
    try t.expectError(error.UnboundLabel, cb.finalize());
    var ids: [4]u32 = undefined;
    try t.expectEqualSlices(u32, &.{l.id}, cb.unresolvedLabels(&ids));
    try a.bind(l);
    try t.expectError(error.LabelAlreadyBound, a.bind(l));
    try t.expectError(error.InvalidLabel, a.cbnz(.x1, .{ .id = 5 }));
    try cb.finalize();
}

// ---- align, trap fill, prolog/epilog ------------------------------------------------------

test "x86_64 alignNop/alignTrap/int3 and prolog/epilog" {
    if (!opts.x86_64) return error.SkipZigTest;
    const x = @import("x86_64/assembler.zig");
    inline for (.{ x.AssemblerT(.{}), x.AssemblerT(.{ .validate = true }) }) |A| {
        var cb = CodeBuffer.init(t.allocator);
        defer cb.deinit();
        const a = A.init(&cb);
        try a.int3();
        try a.alignNop(16); // 15 bytes: 9 + 6
        try t.expectEqual(@as(usize, 16), cb.code().len);
        try t.expectEqualSlices(u8, &.{ 0x66, 0x0F, 0x1F, 0x84, 0, 0, 0, 0, 0, 0x66, 0x0F, 0x1F, 0x44, 0, 0 }, cb.code()[1..16]);
        try a.alignNop(16); // already aligned: nothing
        try t.expectEqual(@as(usize, 16), cb.code().len);
        try a.nop();
        try a.alignTrap(8);
        try t.expectEqualSlices(u8, &.{ 0x90, 0xCC, 0xCC, 0xCC, 0xCC, 0xCC, 0xCC, 0xCC }, cb.code()[16..24]);

        cb.reset();
        const saved = [_]x.Gp{ .rbx, .r12, .r13 };
        const adj = try a.prolog(&saved, 20);
        try t.expectEqual(@as(i32, 24), adj); // 24 saved + 24 = 48 >= 24 + 20
        try a.epilog(&saved, adj);
        try t.expectEqualSlices(u8, &.{
            0x55, 0x48, 0x89, 0xE5, 0x53, 0x41, 0x54, 0x41, 0x55, 0x48, 0x83, 0xEC, 0x18,
            0x48, 0x83, 0xC4, 0x18, 0x41, 0x5D, 0x41, 0x5C, 0x5B, 0x5D, 0xC3,
        }, cb.code());
    }
    try t.expectEqualSlices(x.Gp, &.{ .rdi, .rsi, .rdx, .rcx, .r8, .r9 }, x.CallConv.sysv.args);
    try t.expectEqual(@as(u32, 32), x.CallConv.win64.shadow);
}

test "aarch64 alignNop/alignTrap/udf and prolog/epilog" {
    const x = @import("aarch64/assembler.zig");
    inline for (.{ x.AssemblerT(.{}), x.AssemblerT(.{ .validate = true }) }) |A| {
        var cb = CodeBuffer.init(t.allocator);
        defer cb.deinit();
        const a = A.init(&cb);
        try a.udf(7);
        try a.alignNop(16);
        try a.alignTrap(32);
        const w = std.mem.bytesAsSlice(u32, cb.code());
        try t.expectEqual(@as(usize, 8), w.len);
        try t.expectEqual(@as(u32, 7), w[0]);
        try t.expectEqual(@as(u32, 0xD503201F), w[1]);
        try t.expectEqual(@as(u32, 0xD503201F), w[3]);
        try t.expectEqual(@as(u32, 0), w[4]);

        cb.reset();
        const n = try a.prolog(&.{}, 20);
        try t.expectEqual(@as(u12, 32), n);
        try a.epilog(&.{}, n);
        const p = std.mem.bytesAsSlice(u32, cb.code());
        try t.expectEqualSlices(u32, &.{ 0xA9BF7BFD, 0x910003FD, 0xD10083FF, 0x910083FF, 0xA8C17BFD, 0xD65F03C0 }, @alignCast(p));
        try t.expectError(error.InvalidOperand, a.prolog(&.{}, 5000));
        try t.expectError(error.InvalidOperand, a.prolog(&.{.x29}, 0));
        cb.reset();
        const sv = [_]x.X{ .x19, .x20, .x21 };
        const m = try a.prolog(&sv, 0);
        try a.epilog(&sv, m);
        try t.expectEqualSlices(u32, &.{
            0xA9BF7BFD, 0x910003FD, 0xA9BF53F3, 0xF81F0FF5, // stp x19, x20, [sp,#-16]!; str x21, [sp,#-16]!
            0xF84107F5, 0xA8C153F3, 0xA8C17BFD, 0xD65F03C0, // ldr x21, [sp],#16; ldp x19, x20, [sp],#16
        }, @alignCast(std.mem.bytesAsSlice(u32, cb.code())));
    }
}

test "jit runtime: prolog/epilog function runs; tail is trap-filled" {
    const builtin = @import("builtin");
    const rt = @import("core/jit_runtime.zig");
    var cb = CodeBuffer.init(t.allocator);
    defer cb.deinit();
    switch (builtin.cpu.arch) {
        .x86_64 => {
            const x = @import("x86_64/assembler.zig");
            const a = x.Assembler.init(&cb);
            const cc = x.CallConv.host;
            const saved = [_]x.Gp{.rbx};
            const adj = try a.prolog(&saved, 8);
            try a.movRR(.rbx, cc.args[0]);
            try a.addRR(.rbx, cc.args[1]);
            try a.movRR(cc.ret, .rbx);
            try a.epilog(&saved, adj);
        },
        .aarch64 => {
            const x = @import("aarch64/assembler.zig");
            const a = x.Assembler.init(&cb);
            const n = try a.prolog(&.{}, 16);
            try a.add(.x0, .x0, .x1);
            try a.epilog(&.{}, n);
        },
        else => return error.SkipZigTest,
    }
    try cb.finalize();
    const f = try rt.add(&cb);
    defer f.release();
    try t.expectEqual(@as(i64, 42), f.ptr(*const fn (i64, i64) callconv(.c) i64)(40, 2));
    const trap: u8 = if (builtin.cpu.arch == .x86_64) 0xCC else 0;
    for (f.mem[cb.code().len..]) |b| try t.expectEqual(trap, b);
}

test "code buffer limits: max_size and max_labels" {
    var cb = CodeBuffer.init(t.allocator);
    defer cb.deinit();
    cb.max_size = 64;
    cb.max_labels = 2;
    _ = try cb.newLabel();
    _ = try cb.newLabel();
    try t.expectError(error.TooManyLabels, cb.newLabel());
    var big: [100]u8 = @splat(0x90);
    try t.expectError(error.CodeTooLarge, cb.emitBytes(&big));
    try cb.emitBytes(big[0..64]);
    // Writes that fit the existing capacity are caught at finalize.
    try cb.emitBytes(big[0..1]);
    try t.expectError(error.CodeTooLarge, cb.finalize());
}

// ---- generated GP encoder: sizes, RIP-relative, features -----------------------------------

test "x86_64 gp: 8/16/32-bit operands, RIP-relative label, errors" {
    if (!opts.x86_64) return error.SkipZigTest;
    const x = @import("x86_64/assembler.zig");
    const gp = x.gp;
    var cb = CodeBuffer.init(t.allocator);
    defer cb.deinit();
    const a = x.AssemblerT(.{ .validate = true }).init(&cb);
    const data = try a.newLabel();
    try a.inst(.mov, .{ gp.r8(.rsi), gp.byte(x.ptr(.rdi, 1)) }); // 40 8A 77 01
    try a.inst(.mov, .{ gp.r16(.r9), @as(i64, 0x1234) }); // 66 41 B9 34 12
    try a.inst(.add, .{ gp.dword(x.ptr(.rax, 0)), @as(i64, 0xFFFFFFFF) }); // 83 00 FF
    try a.inst(.mov, .{ gp.r64(.rax), gp.ripLabel(data, .q) }); // 48 8B 05 <rel32>
    try a.inst(.lea, .{ gp.r64(.rcx), gp.rip(-7, .none) }); // 48 8D 0D F9 FF FF FF
    try a.bind(data);
    try cb.emitBytes(&.{ 1, 2, 3, 4, 5, 6, 7, 8 });
    try cb.finalize();
    try t.expectEqualSlices(u8, &.{
        0x40, 0x8A, 0x77, 0x01, 0x66, 0x41, 0xB9, 0x34, 0x12, 0x83, 0x00, 0xFF,
        0x48, 0x8B, 0x05, 0x07, 0x00, 0x00, 0x00, 0x48, 0x8D, 0x0D, 0xF9, 0xFF, 0xFF, 0xFF,
    }, cb.code()[0..26]);
    // errors
    try t.expectError(error.NoMatchingForm, a.inst(.mov, .{ gp.r32(.rax), gp.r64(.rcx) }));
    try t.expectError(error.NoMatchingForm, a.inst(.add, .{ gp.r8(.rax), @as(i64, 300) }));
    try t.expectError(error.InvalidOperand, a.inst(.mov, .{ gp.r64(.rax), gp.Mem{ .rip = true, .base = .rbx, .size = .q } }));
    try t.expectError(error.InvalidLabel, a.inst(.mov, .{ gp.r64(.rax), gp.ripLabel(.{ .id = 99 }, .q) }));
    const len = cb.code().len;
    try t.expectEqual(len, cb.code().len); // failed emits leave nothing behind
}

test "x86_64 gp: CPU features from the database" {
    if (!opts.x86_64) return error.SkipZigTest;
    const gp = @import("x86_64/assembler.zig").gp;
    try t.expect(gp.requiredFeatures(.lzcnt).contains(.LZCNT));
    try t.expect(gp.requiredFeatures(.popcnt).contains(.POPCNT));
    try t.expectEqual(@as(usize, 0), gp.requiredFeatures(.add).count());
    // A target that has LZCNT accepts it at compile time; deny/target misses are compile errors.
    const x = @import("x86_64/assembler.zig");
    var cb = CodeBuffer.init(t.allocator);
    defer cb.deinit();
    const a = x.AssemblerT(.{ .validate = true, .cpu_check = .target, .target_features = &.{"LZCNT"}, .deny_features = &.{"RTM"} }).init(&cb);
    try a.inst(.lzcnt, .{ x.gp.r32(.rax), x.gp.r32(.rcx) });
    try t.expectEqualSlices(u8, &.{ 0xF3, 0x0F, 0xBD, 0xC1 }, cb.code());
}

test "jit runtime: generated GP encoder (movzx, RIP-relative constant) runs" {
    const builtin = @import("builtin");
    if (builtin.cpu.arch != .x86_64 or !opts.x86_64) return error.SkipZigTest;
    const x = @import("x86_64/assembler.zig");
    const gp = x.gp;
    const rt = @import("core/jit_runtime.zig");
    var cb = CodeBuffer.init(t.allocator);
    defer cb.deinit();
    const a = x.Assembler.init(&cb);
    const k = try a.newLabel();
    const arg = x.CallConv.host.args[0];
    try a.inst(.movzx, .{ gp.r32(.rax), gp.r8(arg) }); // zero-extend low byte of arg
    try a.inst(.add, .{ gp.r64(.rax), gp.ripLabel(k, .q) });
    try a.ret();
    try a.alignTrap(8);
    try a.bind(k);
    try cb.emitBytes(&std.mem.toBytes(@as(u64, 1000)));
    try cb.finalize();
    const f = try rt.add(&cb);
    defer f.release();
    try t.expectEqual(@as(u64, 1000 + 0x34), f.ptr(*const fn (u64) callconv(.c) u64)(0x1234));
}

test "jit runtime: callee-saved registers and stack arguments" {
    const builtin = @import("builtin");
    const rt = @import("core/jit_runtime.zig");
    var cb = CodeBuffer.init(t.allocator);
    defer cb.deinit();
    // f(a0..a9) = a0 + a8 + a9, using a callee-saved register as the accumulator.
    switch (builtin.cpu.arch) {
        .x86_64 => {
            const x = @import("x86_64/assembler.zig");
            const a = x.Assembler.init(&cb);
            const cc = x.CallConv.host;
            const sv = [_]x.Gp{ .rbx, .r12 };
            const adj = try a.prolog(&sv, 0);
            try a.movRR(.rbx, cc.args[0]);
            try a.movRM(.r12, cc.stackArg(8));
            try a.addRR(.rbx, .r12);
            try a.movRM(.r12, cc.stackArg(9));
            try a.addRR(.rbx, .r12);
            try a.movRR(cc.ret, .rbx);
            try a.epilog(&sv, adj);
        },
        .aarch64 => {
            const x = @import("aarch64/assembler.zig");
            const a = x.Assembler.init(&cb);
            const sv = [_]x.X{ .x19, .x20, .x21 };
            const n = try a.prolog(&sv, 32);
            try a.mov(.x19, .x0);
            try a.ldr(.x20, .x29, x.CallConv.stackArgOffset(8));
            try a.add(.x19, .x19, .x20);
            try a.ldr(.x21, .x29, x.CallConv.stackArgOffset(9));
            try a.add(.x0, .x19, .x21);
            try a.epilog(&sv, n);
        },
        else => return error.SkipZigTest,
    }
    try cb.finalize();
    const f = try rt.add(&cb);
    defer f.release();
    const F = *const fn (i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) callconv(.c) i64;
    try t.expectEqual(@as(i64, 1 + 900 + 10000), f.ptr(F)(1, 2, 3, 4, 5, 6, 7, 8, 900, 10000));
}
