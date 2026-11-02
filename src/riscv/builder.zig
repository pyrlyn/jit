//! RISC-V Builder: records RV64 instructions as nodes, runs passes over the node list, then
//! replays it through an `AssemblerT(opts)` (like `aarch64/builder.zig`).
//!
//! - `Instr` has one variant per Assembler emitter, named after it, whose payload is the
//!   emitter's argument tuple; `serialize` calls the emitter by name, so output is
//!   byte-identical to calling the Assembler directly (auto-compress with
//!   `AssemblerT(.{ .compress = true })` included).
//! - Branches are fixed-size (`b` +-4 KiB, `bFar`/`j` +-1 MiB, `jFar` +-2 GiB), so there is no
//!   branch shortening; passes are `removeDeadCode` and `removeJumpsToNext`.
const std = @import("std");
const rv = @import("assembler.zig");
const core = @import("../core/code_buffer.zig");
const X = rv.X;
const Cond = rv.Cond;

const generic = @import("../core/builder.zig");
pub const Error = rv.Error || error{LabelAlreadyBound};

/// Builder-side label (mapped to buffer labels at serialize).
pub const Label = generic.Label;

const R3 = struct { X, X, X };
const RI = struct { X, X, i64 };
const Sh = struct { X, X, u32 };
const Mem = struct { X, i64, X };

/// One recorded instruction: tag = Assembler method name, payload = its arguments.
pub const Instr = union(enum) {
    li: struct { X, i64 },
    mv: struct { X, X },
    lui: struct { X, u32 },
    auipc: struct { X, u32 },
    add: R3,
    sub: R3,
    addw: R3,
    subw: R3,
    sll: R3,
    srl: R3,
    sra: R3,
    slt: R3,
    sltu: R3,
    xor: R3,
    @"or": R3,
    @"and": R3,
    mul: R3,
    mulh: R3,
    mulhu: R3,
    div: R3,
    divu: R3,
    rem: R3,
    remu: R3,
    addi: RI,
    slti: RI,
    xori: RI,
    ori: RI,
    andi: RI,
    slli: Sh,
    srli: Sh,
    srai: Sh,
    lb: Mem,
    lh: Mem,
    lw: Mem,
    ld: Mem,
    lbu: Mem,
    lhu: Mem,
    lwu: Mem,
    sb: Mem,
    sh: Mem,
    sw: Mem,
    sd: Mem,
    jalr: struct { X, i64, X },
    callAbs: struct { u64 },
    nop,
    ret,
    j: struct { Label },
    jal: struct { X, Label },
    jFar: struct { X, Label },
    b: struct { Cond, X, X, Label },
    bFar: struct { Cond, X, X, Label },

    pub const min_align: u32 = 2;
    pub const comment_prefix: []const u8 = "#";
    /// GNU-style text: `addi a0, a0, -1`, `ld t0, 8(sp)`, `b ne, a0, zero, L0`.
    pub fn formatInst(gpa: std.mem.Allocator, out: *std.ArrayList(u8), i: Instr) !void {
        switch (i) {
            inline else => |args, tag| {
                try out.appendSlice(gpa, @tagName(tag));
                const T = @TypeOf(args);
                if (T == void) return;
                const mem_form = T == Mem or tag == .jalr;
                inline for (args, 0..) |v, k| {
                    try out.appendSlice(gpa, if (k == 0) " " else if (mem_form and k == 2) "(" else ", ");
                    switch (@TypeOf(v)) {
                        X, Cond => try out.appendSlice(gpa, @tagName(v)),
                        Label => try out.print(gpa, "L{d}", .{v.id}),
                        u64 => try out.print(gpa, "0x{x}", .{v}),
                        else => try out.print(gpa, "{d}", .{v}),
                    }
                    if (mem_form and k == 2) try out.append(gpa, ')');
                }
            },
        }
    }
    pub fn jumpTarget(i: Instr) ?Label {
        return switch (i) {
            .j => |x| x[0],
            .jFar => |x| x[1],
            else => null,
        };
    }
    /// Control never falls through to the next node.
    pub fn isTerminator(i: Instr) bool {
        return switch (i) {
            .j, .jFar, .ret => true,
            else => false,
        };
    }
};

pub const Builder = generic.BuilderT(Instr);
pub const Node = Builder.Node;

// ---- passes ---------------------------------------------------------------------------
/// Removes instructions after `j`/`jFar`/`ret` up to the next label.
pub const removeDeadCode = Builder.removeDeadCode;
/// Removes `j L` / `jFar L` when `L` is bound right after it (only labels/comments in between).
pub const removeJumpsToNext = Builder.removeJumpsToNext;

// ---- tests ------------------------------------------------------------------------------
const t = std.testing;
const builtin = @import("builtin");
const CodeBuffer = core.CodeBuffer;

/// sum = 0; while (n != 0) { sum += n * 3; n -= 1 } return sum
fn buildLoop(b: *Builder) !void {
    const loop = try b.newLabel();
    const done = try b.newLabel();
    try b.comment("sum of 3*i");
    try b.emit(.li, .{ .t1, 0 });
    try b.emit(.li, .{ .t2, 3 });
    try b.bind(loop);
    try b.emit(.b, .{ .eq, .a0, .zero, done });
    try b.emit(.mul, .{ .t3, .a0, .t2 });
    try b.emit(.add, .{ .t1, .t1, .t3 });
    try b.emit(.addi, .{ .a0, .a0, -1 });
    try b.emit(.bFar, .{ .ne, .a0, .zero, loop });
    try b.emit(.j, .{done});
    try b.bind(done);
    try b.emit(.mv, .{ .a0, .t1 });
    try b.add(.ret);
}

fn assembleLoop(a: anytype) !void {
    const loop = try a.newLabel();
    const done = try a.newLabel();
    try a.li(.t1, 0);
    try a.li(.t2, 3);
    try a.bind(loop);
    try a.b(.eq, .a0, .zero, done);
    try a.mul(.t3, .a0, .t2);
    try a.add(.t1, .t1, .t3);
    try a.addi(.a0, .a0, -1);
    try a.bFar(.ne, .a0, .zero, loop);
    try a.j(done);
    try a.bind(done);
    try a.mv(.a0, .t1);
    try a.ret();
}

test "riscv builder: serialize is byte-identical to the Assembler (plain and compressed)" {
    inline for (.{ rv.Assembler, rv.AssemblerC }) |A| {
        var b = Builder.init(t.allocator);
        defer b.deinit();
        try buildLoop(&b);
        var cb1 = CodeBuffer.init(t.allocator);
        defer cb1.deinit();
        try b.serialize(A.init(&cb1));
        try cb1.finalize();
        var cb2 = CodeBuffer.init(t.allocator);
        defer cb2.deinit();
        try assembleLoop(A.init(&cb2));
        try cb2.finalize();
        try t.expectEqualSlices(u8, cb2.code(), cb1.code());
    }
}

test "riscv builder: passes, cursor insertion, formatter" {
    var b = Builder.init(t.allocator);
    defer b.deinit();
    try buildLoop(&b);
    try b.emit(.li, .{ .a0, 99 }); // dead: after ret, before any label
    const n0 = b.count();
    try b.runPasses(.{ removeDeadCode, removeJumpsToNext });
    try t.expectEqual(n0 - 2, b.count()); // dead li + `j done` before `done:`

    b.setCursor(Node.none);
    try b.emit(.addi, .{ .a0, .a0, 0 });
    try t.expectEqual(n0 - 1, b.count());

    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(t.allocator);
    try b.format(&out);
    try t.expect(std.mem.startsWith(u8, out.items, "  addi a0, a0, 0\n  # sum of 3*i\n"));
    try t.expect(std.mem.indexOf(u8, out.items, "  bFar ne, a0, zero, L0\n") != null);
    try t.expect(std.mem.indexOf(u8, out.items, "  addi a0, a0, -1\n") != null);
    var b2 = Builder.init(t.allocator);
    defer b2.deinit();
    try b2.emit(.ld, .{ .t0, 8, .sp });
    try b2.emit(.jalr, .{ .ra, 0, .t0 });
    try b2.emit(.@"and", .{ .a0, .a1, .a2 });
    var out2: std.ArrayList(u8) = .empty;
    defer out2.deinit(t.allocator);
    try b2.format(&out2);
    try t.expectEqualStrings("  ld t0, 8(sp)\n  jalr ra, 0(t0)\n  and a0, a1, a2\n", out2.items);
    try t.expect(std.mem.indexOf(u8, out.items, "  j L1") == null);

    // align: 2-aligned offset -> c.nop + nop; trap padding is zero bytes.
    var cb = CodeBuffer.init(t.allocator);
    defer cb.deinit();
    const a = rv.Assembler.init(&cb);
    try a.buf.emitBytes(&.{ 0x01, 0x00 });
    try a.alignNop(8);
    try t.expectEqualSlices(u8, &.{ 0x01, 0x00, 0x01, 0x00, 0x13, 0x00, 0x00, 0x00 }, cb.code());
    try a.alignTrap(8);
    try t.expectEqual(@as(usize, 8), cb.code().len);
    try a.nop();
    try a.alignTrap(16);
    try t.expectEqualSlices(u8, &.{ 0, 0, 0, 0 }, cb.code()[12..16]);
    try t.expectError(error.InvalidOperand, a.alignNop(3));

    try t.expectError(error.InvalidLabel, b.emit(.j, .{.{ .id = 7 }}));
    try t.expectError(error.LabelAlreadyBound, b.bind(.{ .id = 0 }));
}

test "riscv builder: execution (RV64 host / qemu)" {
    if (builtin.cpu.arch != .riscv64) return error.SkipZigTest;
    const JitRuntime = @import("../core/jit_runtime.zig");
    inline for (.{ rv.Assembler, rv.AssemblerC }) |A| {
        var b = Builder.init(t.allocator);
        defer b.deinit();
        try buildLoop(&b);
        try b.runPasses(.{ removeDeadCode, removeJumpsToNext });
        var cb = CodeBuffer.init(t.allocator);
        defer cb.deinit();
        try b.serialize(A.init(&cb));
        try cb.finalize();
        const fun = try JitRuntime.add(&cb);
        defer fun.release();
        const F = *const fn (u64) callconv(.c) u64;
        for ([_]u64{ 0, 1, 10, 1000 }) |n| try t.expectEqual(3 * n * (n + 1) / 2, fun.ptr(F)(n));
    }
}
