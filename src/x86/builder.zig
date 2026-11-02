//! x86 (32-bit) Builder: records instructions as nodes on the generic `core/builder.zig`
//! layer, runs passes, then replays them through an `x86.AssemblerT(opts)` (byte-identical to
//! calling the Assembler directly; validation and hooks unchanged).
const std = @import("std");
const x86 = @import("assembler.zig");
const core = @import("../core/code_buffer.zig");
const generic = @import("../core/builder.zig");
const Gp = x86.Gp;
const Mem = x86.Mem;
const Xmm = x86.Xmm;
const Alu = x86.Alu;
const Cond = x86.Cond;

pub const Error = x86.Error || generic.Error;
pub const Label = generic.Label;

/// One recorded instruction: tag = Assembler method name, payload = its arguments.
pub const Instr = union(enum) {
    movRR: struct { Gp, Gp },
    movRM: struct { Gp, Mem },
    movMR: struct { Mem, Gp },
    movMI: struct { Mem, i32 },
    movRI: struct { Gp, i64 },
    lea: struct { Gp, Mem },
    aluRR: struct { Alu, Gp, Gp },
    aluRM: struct { Alu, Gp, Mem },
    aluMR: struct { Alu, Mem, Gp },
    aluRI: struct { Alu, Gp, i32 },
    addRR: struct { Gp, Gp },
    addRI: struct { Gp, i32 },
    subRI: struct { Gp, i32 },
    cmpRR: struct { Gp, Gp },
    testRR: struct { Gp, Gp },
    imulRR: struct { Gp, Gp },
    imulRRI: struct { Gp, Gp, i32 },
    inc: struct { Gp },
    dec: struct { Gp },
    push: struct { Gp },
    pop: struct { Gp },
    pushI: struct { i32 },
    ret,
    retN: struct { u16 },
    nop,
    int3,
    callR: struct { Gp },
    jmpR: struct { Gp },
    jmp: struct { Label },
    call: struct { Label },
    jcc: struct { Cond, Label },
    callAbs: struct { u32 },
    movupsRM: struct { Xmm, Mem },
    movupsMR: struct { Mem, Xmm },
    movsdRM: struct { Xmm, Mem },
    movsdMR: struct { Mem, Xmm },
    movssRM: struct { Xmm, Mem },
    movssMR: struct { Mem, Xmm },
    cvtsi2sd: struct { Xmm, Gp },
    cvttsd2si: struct { Gp, Xmm },

    /// The ALU emitters take the operation at comptime.
    pub const special: []const []const u8 = &.{ "aluRR", "aluRM", "aluMR", "aluRI" };
    pub fn emitSpecial(a: anytype, i: Instr, _: []const core.Label) !void {
        switch (i) {
            .aluRR => |o| switch (o[0]) {
                inline else => |op| try a.aluRR(op, o[1], o[2]),
            },
            .aluRM => |o| switch (o[0]) {
                inline else => |op| try a.aluRM(op, o[1], o[2]),
            },
            .aluMR => |o| switch (o[0]) {
                inline else => |op| try a.aluMR(op, o[1], o[2]),
            },
            .aluRI => |o| switch (o[0]) {
                inline else => |op| try a.aluRI(op, o[1], o[2]),
            },
            else => unreachable,
        }
    }
    pub fn isTerminator(i: Instr) bool {
        return switch (i) {
            .ret, .retN, .jmp, .jmpR, .int3 => true,
            else => false,
        };
    }
    pub fn jumpTarget(i: Instr) ?Label {
        return if (i == .jmp) i.jmp[0] else null;
    }
};

pub const Builder = generic.BuilderT(Instr);
pub const Node = Builder.Node;
/// Removes instructions after `ret`/`jmp`/`int3` up to the next label.
pub const removeDeadCode = Builder.removeDeadCode;
/// Removes `jmp L` when `L` is bound right after it.
pub const removeJumpsToNext = Builder.removeJumpsToNext;

// ---- tests ------------------------------------------------------------------------------
const t = std.testing;
const CodeBuffer = core.CodeBuffer;
const ptr = x86.ptr;

/// cdecl `u32 f(u32 n)`: sum of 3*i for i in 1..n.
fn buildLoop(b: *Builder) !void {
    const loop = try b.newLabel();
    const done = try b.newLabel();
    try b.comment("sum of 3*i");
    try b.emit(.movRM, .{ .ecx, ptr(.esp, 4) });
    try b.emit(.aluRR, .{ .xor, .eax, .eax });
    try b.bind(loop);
    try b.emit(.testRR, .{ .ecx, .ecx });
    try b.emit(.jcc, .{ .e, done });
    try b.emit(.imulRRI, .{ .edx, .ecx, 3 });
    try b.emit(.addRR, .{ .eax, .edx });
    try b.emit(.dec, .{.ecx});
    try b.emit(.jmp, .{loop});
    try b.bind(done);
    try b.add(.ret);
}

test "x86 builder: serialize is byte-identical to the Assembler" {
    var b = Builder.init(t.allocator);
    defer b.deinit();
    try buildLoop(&b);
    var cb1 = CodeBuffer.init(t.allocator);
    defer cb1.deinit();
    try b.serialize(x86.Assembler.init(&cb1));
    try cb1.finalize();

    var cb2 = CodeBuffer.init(t.allocator);
    defer cb2.deinit();
    const a = x86.Assembler.init(&cb2);
    const loop = try a.newLabel();
    const done = try a.newLabel();
    try a.movRM(.ecx, ptr(.esp, 4));
    try a.aluRR(.xor, .eax, .eax);
    try a.bind(loop);
    try a.testRR(.ecx, .ecx);
    try a.jcc(.e, done);
    try a.imulRRI(.edx, .ecx, 3);
    try a.addRR(.eax, .edx);
    try a.dec(.ecx);
    try a.jmp(loop);
    try a.bind(done);
    try a.ret();
    try cb2.finalize();
    try t.expectEqualSlices(u8, cb2.code(), cb1.code());
}

test "x86 builder: passes, SIMD nodes, hooks through serialize, formatter" {
    var b = Builder.init(t.allocator);
    defer b.deinit();
    const next = try b.newLabel();
    try b.emit(.movsdRM, .{ .xmm0, ptr(.esp, 4) });
    try b.emit(.cvttsd2si, .{ .eax, .xmm0 });
    try b.emit(.jmp, .{next}); // removed
    try b.bind(next);
    try b.add(.ret);
    try b.emit(.nop, {}); // dead
    try b.alignTrap(8);
    try b.runPasses(.{ removeDeadCode, removeJumpsToNext });
    var cb = CodeBuffer.init(t.allocator);
    defer cb.deinit();
    var h: x86.CountingHooks = .{};
    try b.serialize(x86.AssemblerT(.{ .Hooks = x86.CountingHooks }).initHooks(&cb, &h));
    try cb.finalize();
    // movsd xmm0, [esp+4]; cvttsd2si eax, xmm0; ret; int3 x3 (clang -target i386)
    try t.expectEqualSlices(u8, &.{ 0xf2, 0x0f, 0x10, 0x44, 0x24, 0x04, 0xf2, 0x0f, 0x2c, 0xc0, 0xc3, 0xcc, 0xcc, 0xcc, 0xcc, 0xcc }, cb.code());
    try t.expectEqual(@as(usize, 3 + 5), h.instructions);
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(t.allocator);
    try b.format(&out);
    try t.expect(std.mem.startsWith(u8, out.items, "  movsdRM xmm0, "));
    try t.expect(std.mem.indexOf(u8, out.items, "  cvttsd2si eax, xmm0\nL0:\n  ret\n  .align 8\n") != null);
}
