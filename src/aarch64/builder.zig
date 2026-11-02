//! AArch64 Builder: records A64 instructions as nodes, runs passes over the node list, then
//! replays it through an `AssemblerT(opts)` (like `x86_64/builder.zig`).
//!
//! - `Instr` has one variant per Assembler emitter, named after it, whose payload is the
//!   emitter's argument tuple; `serialize` calls the emitter by name, so output is
//!   byte-identical to calling the Assembler directly (validation and hooks unchanged).
//! - A64 words are fixed-size, so there is no branch shortening; passes are
//!   `removeDeadCode` and `removeJumpsToNext`.
const std = @import("std");
const a64 = @import("assembler.zig");
const core = @import("../core/code_buffer.zig");
const X = a64.X;
const Cond = a64.Cond;

const generic = @import("../core/builder.zig");
pub const Error = a64.Error || error{LabelAlreadyBound};

/// Builder-side label (mapped to buffer labels at serialize).
pub const Label = generic.Label;

/// One recorded instruction: tag = Assembler method name, payload = its arguments.
pub const Instr = union(enum) {
    movImm: struct { X, u64 },
    mov: struct { X, X },
    add: struct { X, X, X },
    sub: struct { X, X, X },
    mul: struct { X, X, X },
    sdiv: struct { X, X, X },
    udiv: struct { X, X, X },
    mneg: struct { X, X, X },
    smulh: struct { X, X, X },
    umulh: struct { X, X, X },
    and_: struct { X, X, X },
    orr: struct { X, X, X },
    eor: struct { X, X, X },
    madd: struct { X, X, X, X },
    msub: struct { X, X, X, X },
    neg: struct { X, X },
    csel: struct { X, X, X, Cond },
    cset: struct { X, Cond },
    lslImm: struct { X, X, u6 },
    lsrImm: struct { X, X, u6 },
    asrImm: struct { X, X, u6 },
    addImm: struct { X, X, u12 },
    subImm: struct { X, X, u12 },
    cmp: struct { X, X },
    cmpImm: struct { X, u12 },
    ldr: struct { X, X, u15 },
    str: struct { X, X, u15 },
    word: struct { u32 },
    udf: struct { u16 },
    ret,
    blr: struct { X },
    b: struct { Label },
    bl: struct { Label },
    bcond: struct { Cond, Label },
    cbz: struct { X, Label },
    cbnz: struct { X, Label },

    pub const min_align: u32 = 4;
    pub fn jumpTarget(i: Instr) ?Label {
        return if (i == .b) i.b[0] else null;
    }
    /// `bcond` takes the condition at comptime.
    pub const special: []const []const u8 = &.{"bcond"};
    pub fn emitSpecial(a: anytype, i: Instr, labels: []const core.Label) !void {
        switch (i.bcond[0]) {
            inline else => |c| try a.bcond(c, labels[i.bcond[1].id]),
        }
    }
    /// Control never falls through to the next node.
    pub fn isTerminator(i: Instr) bool {
        return switch (i) {
            .b, .ret, .udf => true,
            else => false,
        };
    }
};

pub const Builder = generic.BuilderT(Instr);
pub const Node = Builder.Node;

// ---- passes ---------------------------------------------------------------------------
/// Removes instructions after `b`/`ret`/`udf` up to the next label.
pub const removeDeadCode = Builder.removeDeadCode;
/// Removes `b L` when `L` is bound right after it (only labels/comments in between).
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
    try b.emit(.movImm, .{ .x1, 0 });
    try b.emit(.movImm, .{ .x2, 3 });
    try b.bind(loop);
    try b.emit(.cbz, .{ .x0, done });
    try b.emit(.madd, .{ .x1, .x0, .x2, .x1 });
    try b.emit(.subImm, .{ .x0, .x0, 1 });
    try b.emit(.cmpImm, .{ .x0, 0 });
    try b.emit(.bcond, .{ .ne, loop });
    try b.emit(.b, .{done});
    try b.bind(done);
    try b.emit(.mov, .{ .x0, .x1 });
    try b.add(.ret);
}

test "a64 builder: serialize is byte-identical to the Assembler" {
    var b = Builder.init(t.allocator);
    defer b.deinit();
    try buildLoop(&b);
    var cb1 = CodeBuffer.init(t.allocator);
    defer cb1.deinit();
    try b.serialize(a64.AssemblerT(.{ .validate = true }).init(&cb1));
    try cb1.finalize();

    var cb2 = CodeBuffer.init(t.allocator);
    defer cb2.deinit();
    const a = a64.AssemblerT(.{ .validate = true }).init(&cb2);
    const loop = try a.newLabel();
    const done = try a.newLabel();
    try a.movImm(.x1, 0);
    try a.movImm(.x2, 3);
    try a.bind(loop);
    try a.cbz(.x0, done);
    try a.madd(.x1, .x0, .x2, .x1);
    try a.subImm(.x0, .x0, 1);
    try a.cmpImm(.x0, 0);
    try a.bcond(.ne, loop);
    try a.b(done);
    try a.bind(done);
    try a.mov(.x0, .x1);
    try a.ret();
    try cb2.finalize();
    try t.expectEqualSlices(u8, cb2.code(), cb1.code());
}

test "a64 builder: passes, cursor insertion, formatter" {
    var b = Builder.init(t.allocator);
    defer b.deinit();
    try buildLoop(&b);
    try b.emit(.movImm, .{ .x0, 99 }); // dead: after ret, before any label
    const n0 = b.count();
    try b.runPasses(.{ removeDeadCode, removeJumpsToNext });
    try t.expectEqual(n0 - 2, b.count()); // dead movImm + `b done` before `done:`

    // Insert at the front.
    b.setCursor(Node.none);
    try b.emit(.addImm, .{ .x0, .x0, 0 });
    try t.expectEqual(n0 - 1, b.count());

    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(t.allocator);
    try b.format(&out);
    try t.expect(std.mem.startsWith(u8, out.items, "  addImm x0, x0, #0\n  // sum of 3*i\n"));
    try t.expect(std.mem.indexOf(u8, out.items, "  bcond ne, L0\n") != null);
    try t.expect(std.mem.indexOf(u8, out.items, "  b L1") == null);

    try t.expectError(error.InvalidLabel, b.emit(.b, .{.{ .id = 7 }}));
    try t.expectError(error.LabelAlreadyBound, b.bind(.{ .id = 0 }));
}

test "a64 builder: execution (AArch64 host)" {
    if (builtin.cpu.arch != .aarch64) return error.SkipZigTest;
    const JitRuntime = @import("../core/jit_runtime.zig");
    var b = Builder.init(t.allocator);
    defer b.deinit();
    try buildLoop(&b);
    try b.runPasses(.{ removeDeadCode, removeJumpsToNext });
    var cb = CodeBuffer.init(t.allocator);
    defer cb.deinit();
    try b.serialize(a64.AssemblerT(.{ .validate = true }).init(&cb));
    try cb.finalize();
    const fun = try JitRuntime.add(&cb);
    defer fun.release();
    const F = *const fn (u64) callconv(.c) u64;
    for ([_]u64{ 0, 1, 10, 1000 }) |n| try t.expectEqual(3 * n * (n + 1) / 2, fun.ptr(F)(n));
}
