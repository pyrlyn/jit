//! x86-64 30-bit immediate helpers (TODO-ARCHES §7): boundaries byte-exact vs llvm-mc 23.1.4
//! (`-triple=x86_64 -show-encoding`), validation, tagged-int helpers and execution.
const std = @import("std");
const builtin = @import("builtin");
const t = std.testing;
const x = @import("x86_64/assembler.zig");
const CodeBuffer = @import("core/code_buffer.zig").CodeBuffer;
const JitRuntime = @import("core/jit_runtime.zig");
const A = x.AssemblerT(.{ .validate = true });

fn expectBytes(want: []const u8, comptime f: anytype, args: anytype) !void {
    var cb = CodeBuffer.init(t.allocator);
    defer cb.deinit();
    try @call(.auto, f, .{A.init(&cb)} ++ args);
    try t.expectEqualSlices(u8, want, cb.code());
}

test "imm30: boundaries byte-exact vs llvm-mc" {
    try expectBytes(&.{ 0xb8, 0, 0, 0, 0 }, A.movI30, .{ .rax, 0 }); // mov eax, 0
    try expectBytes(&.{ 0x48, 0xc7, 0xc0, 0xff, 0xff, 0xff, 0xff }, A.movI30, .{ .rax, -1 });
    try expectBytes(&.{ 0xb8, 0xff, 0xff, 0xff, 0x1f }, A.movI30, .{ .rax, (1 << 29) - 1 });
    try expectBytes(&.{ 0x48, 0xc7, 0xc0, 0x00, 0x00, 0x00, 0xe0 }, A.movI30, .{ .rax, -(1 << 29) });
    try expectBytes(&.{ 0x49, 0xc7, 0xc1, 0x00, 0x00, 0x00, 0xe0 }, A.movI30, .{ .r9, -(1 << 29) });
    try expectBytes(&.{ 0xb8, 0xff, 0xff, 0xff, 0x3f }, A.movU30, .{ .rax, (1 << 30) - 1 });
    try expectBytes(&.{ 0x41, 0xb9, 0xff, 0xff, 0xff, 0x3f }, A.movU30, .{ .r9, (1 << 30) - 1 });
    try expectBytes(&.{ 0x48, 0x83, 0xc0, 0x7f }, A.aluI30, .{ .add, .rax, 127 });
    try expectBytes(&.{ 0x48, 0x83, 0xc0, 0x80 }, A.aluI30, .{ .add, .rax, -128 });
    try expectBytes(&.{ 0x48, 0x81, 0xc1, 0x00, 0x00, 0x00, 0xe0 }, A.aluI30, .{ .add, .rcx, -(1 << 29) });
    try expectBytes(&.{ 0x48, 0x83, 0xeb, 0xfe }, A.aluI30, .{ .sub, .rbx, -2 });
}

test "imm30: rax short form matches llvm-mc (add rax, imm32 = 48 05)" {
    try expectBytes(&.{ 0x48, 0x05, 0xff, 0xff, 0xff, 0x1f }, A.aluI30, .{ .add, .rax, (1 << 29) - 1 });
}

test "imm30: validation and the general-path boundary" {
    try t.expectEqual(@as(i30, (1 << 29) - 1), try x.checkI30((1 << 29) - 1));
    try t.expectEqual(@as(i30, -(1 << 29)), try x.checkI30(-(1 << 29)));
    try t.expectError(error.InvalidImmediate, x.checkI30(1 << 29));
    try t.expectError(error.InvalidImmediate, x.checkI30(1 << 30));
    try t.expectError(error.InvalidImmediate, x.checkI30(-(1 << 29) - 1));
    try t.expect(x.fitsI30(0) and x.fitsI30(-1) and !x.fitsI30(1 << 30));
    var cb = CodeBuffer.init(t.allocator);
    defer cb.deinit();
    const a = A.init(&cb);
    try t.expectError(error.InvalidImmediate, a.movI30Checked(.rax, 1 << 30));
    try t.expectError(error.InvalidImmediate, a.aluI30Checked(.add, .rax, -(1 << 30)));
    try t.expectEqual(@as(usize, 0), cb.code().len);
    // 2^30 falls back to the general path (still imm32 here, via movRI).
    try a.movRI(.rax, 1 << 30);
    try t.expectEqualSlices(u8, &.{ 0xb8, 0, 0, 0, 0x40 }, cb.code());
}

test "imm30: tagged small integers" {
    const L: x.Smi = .{}; // shift 1, tag 0
    try t.expectEqual(@as(i32, 6), x.smi(L, 3));
    try t.expectEqual(@as(i32, -(1 << 30)), x.smi(L, -(1 << 29)));
    try t.expectEqual(@as(i32, 13), x.smi(.{ .shift = 2, .tag = 1 }, 3));
    try expectBytes(&.{ 0xb8, 0x06, 0, 0, 0 }, A.movSmi, .{ L, .rax, 3 });
    try expectBytes(&.{ 0x48, 0x83, 0xc0, 0x04 }, A.addSmi, .{ L, .rax, 2 });
}

test "imm30: tagged-int helpers execute" {
    if (builtin.cpu.arch != .x86_64) return error.SkipZigTest;
    const L: x.Smi = .{ .shift = 1, .tag = 1 };
    var cb = CodeBuffer.init(t.allocator);
    defer cb.deinit();
    const a = A.init(&cb);
    // rax = smi(-(2^29)); rax += 2^29 - 1 (tag kept); return rax
    try a.movSmi(L, .rax, -(1 << 29));
    try a.addSmi(L, .rax, (1 << 29) - 1);
    try a.ret();
    try cb.finalize();
    const fun = try JitRuntime.add(&cb);
    defer fun.release();
    const got = fun.ptr(*const fn () callconv(.c) i64)();
    try t.expectEqual(@as(i64, x.smi(L, -1)), got);
    try t.expectEqual(@as(i64, -1), got >> 1);
}
