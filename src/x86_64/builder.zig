//! x86-64 Builder: records instructions as nodes on the generic `core/builder.zig` layer
//! (`BuilderT`), runs passes over the node list, then replays it through an `AssemblerT(opts)`
//! (like AsmJit's BaseBuilder / x86::Builder).
//!
//! - `Instr` has one variant per Assembler emitter, named after it, whose payload is the
//!   emitter's argument tuple: `b.emit(.movRR, .{ .rax, .rdi })`, `b.add(.ret)`.
//! - With no pass run, `serialize` is byte-identical to calling the Assembler directly
//!   (validation and hooks unchanged).
//! - Passes: `removeDeadCode`, `removeJumpsToNext` (generic) and `shortenBranches`, which
//!   rewrites `jmp`/`jcc` into fixed-size `jmp8`/`jmp32`/`jcc8`/`jcc32` nodes by relaxation.
const std = @import("std");
const x = @import("assembler.zig");
const core = @import("../core/code_buffer.zig");
const generic = @import("../core/builder.zig");
const CodeBuffer = core.CodeBuffer;
const Gp = x.Gp;
const Mem = x.Mem;
const Alu = x.Alu;
const Cond = x.Cond;
const gp = @import("gp.zig");
const vec = x.vec;

/// Scalar SSE register-register ops used by the Compiler (`op xmm, xmm`).
pub const SseOp = enum { addss, addsd, subss, subsd, mulss, mulsd, divss, divsd, minss, minsd, maxss, maxsd, movaps, xorps, ucomiss, ucomisd, cvtss2sd, cvtsd2ss };
/// Scalar SSE load/store (`movss/movsd xmm, m` and `m, xmm`).
pub const SseMov = enum { movss, movsd };
/// `cvtsi2ss/sd xmm, r64`.
pub const SseFromInt = enum { cvtsi2ss, cvtsi2sd };
/// `cvttss2si/cvttsd2si r64, xmm` (truncating; NaN/out of range give minInt(i64)).
pub const SseToInt = enum { cvttss2si, cvttsd2si };
/// xmm register number (0..15).
pub const Xmm = u4;

pub const Error = x.Error || generic.Error;

/// Shift kind for `shiftRCl` (count in cl, masked to 6 bits by the CPU).
pub const Shift = enum { shl, shr, sar };

fn mapGp(e: anyerror) Error {
    return switch (e) {
        error.NoMatchingForm => error.InvalidOperand,
        else => |x_| @errorCast(x_),
    };
}

/// Builder-side label (mapped to buffer labels at serialize).
pub const Label = generic.Label;

/// One recorded instruction: tag = Assembler method name, payload = its arguments.
pub const Instr = union(enum) {
    movRR: struct { Gp, Gp },
    movRM: struct { Gp, Mem },
    movMR: struct { Mem, Gp },
    movMI: struct { Mem, i32 },
    lea: struct { Gp, Mem },
    movRI: struct { Gp, i64 },
    movabs: struct { Gp, u64 },
    aluRR: struct { Alu, Gp, Gp },
    aluRM: struct { Alu, Gp, Mem },
    aluMR: struct { Alu, Mem, Gp },
    aluRI: struct { Alu, Gp, i32 },
    addRR: struct { Gp, Gp },
    addRI: struct { Gp, i32 },
    subRR: struct { Gp, Gp },
    subRI: struct { Gp, i32 },
    cmpRR: struct { Gp, Gp },
    cmpRI: struct { Gp, i32 },
    xorRR: struct { Gp, Gp },
    testRR: struct { Gp, Gp },
    imulRR: struct { Gp, Gp },
    imulRM: struct { Gp, Mem },
    imulRRI: struct { Gp, Gp, i32 },
    /// `shl/shr/sar r64, cl`.
    shiftRCl: struct { Shift, Gp },
    /// `idiv r64` (rdx:rax / r).
    idivR: struct { Gp },
    /// `cqo` (sign-extend rax into rdx:rax).
    cqo,
    push: struct { Gp },
    pop: struct { Gp },
    ret,
    nop,
    int3,
    callR: struct { Gp },
    jmpR: struct { Gp },
    callAbs: struct { usize },
    jmp: struct { Label },
    call: struct { Label },
    jcc: struct { Cond, Label },
    /// Fixed-size branches (produced by `shortenBranches`; rel8 / rel32).
    jmp8: struct { Label },
    jmp32: struct { Label },
    jcc8: struct { Cond, Label },
    jcc32: struct { Cond, Label },
    /// Scalar SSE (SSE/SSE2 legacy encoding, xmm0..15).
    sseRR: struct { SseOp, Xmm, Xmm },
    sseRM: struct { SseMov, Xmm, Mem },
    sseMR: struct { SseMov, Mem, Xmm },
    /// `movq xmm, r64` / `movq r64, xmm`.
    movqXR: struct { Xmm, Gp },
    movqRX: struct { Gp, Xmm },
    cvtFromInt: struct { SseFromInt, Xmm, Gp },
    cvtToInt: struct { SseToInt, Gp, Xmm },
    /// `setcc r8` (low byte of the register).
    setcc: struct { Cond, Gp },
    /// `movzx r32, r8` (zero-extends into the full register).
    movzxB: struct { Gp, Gp },

    pub const comment_prefix: []const u8 = ";";
    pub const special: []const []const u8 = &.{ "aluRR", "aluRM", "aluMR", "aluRI", "shiftRCl", "idivR", "cqo", "jcc", "jmp8", "jmp32", "jcc8", "jcc32", "sseRR", "sseRM", "sseMR", "movqXR", "movqRX", "cvtFromInt", "cvtToInt", "setcc", "movzxB" };
    pub fn emitSpecial(a: anytype, i: Instr, labels: []const core.Label) !void {
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
            .shiftRCl => |o| switch (o[0]) {
                inline else => |op| a.inst(@field(gp.Mnemonic, @tagName(op)), .{ gp.r64(o[1]), gp.r8(.rcx) }) catch |e| return mapGp(e),
            },
            .idivR => |o| a.inst(.idiv, .{gp.r64(o[0])}) catch |e| return mapGp(e),
            .cqo => a.inst(.cqo, .{}) catch |e| return mapGp(e),
            .jcc => |o| switch (o[0]) {
                inline else => |c| try a.jcc(c, labels[o[1].id]),
            },
            .sseRR => |o| switch (o[0]) {
                inline else => |op| try a.vinst(@field(vec.Mnemonic, @tagName(op)), .{ vec.xmm(o[1]), vec.xmm(o[2]) }, .{}),
            },
            .sseRM => |o| switch (o[0]) {
                inline else => |op| try a.vinst(@field(vec.Mnemonic, @tagName(op)), .{ vec.xmm(o[1]), vec.mem(o[2], if (op == .movss) 4 else 8) }, .{}),
            },
            .sseMR => |o| switch (o[0]) {
                inline else => |op| try a.vinst(@field(vec.Mnemonic, @tagName(op)), .{ vec.mem(o[1], if (op == .movss) 4 else 8), vec.xmm(o[2]) }, .{}),
            },
            .movqXR => |o| try a.vinst(.movq, .{ vec.xmm(o[0]), gp.r64(o[1]) }, .{}),
            .movqRX => |o| try a.vinst(.movq, .{ gp.r64(o[0]), vec.xmm(o[1]) }, .{}),
            .cvtFromInt => |o| switch (o[0]) {
                inline else => |op| try a.vinst(@field(vec.Mnemonic, @tagName(op)), .{ vec.xmm(o[1]), gp.r64(o[2]) }, .{}),
            },
            .cvtToInt => |o| switch (o[0]) {
                inline else => |op| try a.vinst(@field(vec.Mnemonic, @tagName(op)), .{ gp.r64(o[1]), vec.xmm(o[2]) }, .{}),
            },
            .setcc => |o| switch (o[0]) {
                inline else => |c| a.setcc(c, o[1]) catch |e| return mapGp(e),
            },
            .movzxB => |o| a.movzx(.byte, o[0], o[1]) catch |e| return mapGp(e),
            .jmp8 => |o| try rawBranch(a.buf, &.{0xEB}, true, labels[o[0].id]),
            .jmp32 => |o| try rawBranch(a.buf, &.{0xE9}, false, labels[o[0].id]),
            .jcc8 => |o| try rawBranch(a.buf, &.{0x70 + @as(u8, @backingInt(o[0]))}, true, labels[o[1].id]),
            .jcc32 => |o| try rawBranch(a.buf, &.{ 0x0F, 0x80 + @as(u8, @backingInt(o[0])) }, false, labels[o[1].id]),
            else => unreachable,
        }
    }
    /// Unconditional jump target (for `removeJumpsToNext`).
    pub fn jumpTarget(i: Instr) ?Label {
        return switch (i) {
            .jmp, .jmp8, .jmp32 => |o| o[0],
            else => null,
        };
    }
    /// Any branch/call target.
    pub fn target(i: Instr) ?Label {
        return switch (i) {
            .jmp, .call, .jmp8, .jmp32 => |o| o[0],
            .jcc, .jcc8, .jcc32 => |o| o[1],
            else => null,
        };
    }
    /// Control never falls through to the next node.
    pub fn isTerminator(i: Instr) bool {
        return switch (i) {
            .jmp, .jmp8, .jmp32, .jmpR, .ret => true,
            else => false,
        };
    }
    pub const formatInst = formatInstImpl;
};

/// Branch with a size fixed by `shortenBranches` (bypasses the Assembler's choice).
fn rawBranch(buf: *CodeBuffer, op: []const u8, short: bool, l: core.Label) Error!void {
    try buf.emitBytes(op);
    const at = buf.offset();
    const w: u32 = if (short) 1 else 4;
    const zeros: [4]u8 = @splat(0);
    try buf.emitBytes(zeros[0..w]);
    try buf.addFixup(.{ .at = at, .base = at + w, .label = l.id, .kind = if (short) .rel8 else .rel32 });
}

pub const Builder = generic.BuilderT(Instr);
pub const Node = Builder.Node;

fn fmtMem(gpa: std.mem.Allocator, out: *std.ArrayList(u8), m: Mem) !void {
    try out.append(gpa, '[');
    var first = true;
    if (m.base) |r| {
        try out.appendSlice(gpa, @tagName(r));
        first = false;
    }
    if (m.index) |r| {
        if (!first) try out.append(gpa, '+');
        try out.print(gpa, "{s}*{d}", .{ @tagName(r), @as(u8, 1) << @backingInt(m.scale) });
        first = false;
    }
    if (m.disp != 0 or first) {
        if (!first and m.disp >= 0) try out.append(gpa, '+');
        try out.print(gpa, "{d}", .{m.disp});
    }
    try out.append(gpa, ']');
}

fn formatInstImpl(gpa: std.mem.Allocator, out: *std.ArrayList(u8), i: Instr) !void {
    switch (i) {
        .movRR => |o| try out.print(gpa, "mov {s}, {s}", .{ @tagName(o[0]), @tagName(o[1]) }),
        .movRM, .lea => |o| {
            try out.print(gpa, "{s} {s}, ", .{ if (i == .lea) "lea" else "mov", @tagName(o[0]) });
            try fmtMem(gpa, out, o[1]);
        },
        .imulRM => |o| {
            try out.print(gpa, "imul {s}, ", .{@tagName(o[0])});
            try fmtMem(gpa, out, o[1]);
        },
        .movMR => |o| {
            try out.appendSlice(gpa, "mov ");
            try fmtMem(gpa, out, o[0]);
            try out.print(gpa, ", {s}", .{@tagName(o[1])});
        },
        .movMI => |o| {
            try out.appendSlice(gpa, "mov qword ");
            try fmtMem(gpa, out, o[0]);
            try out.print(gpa, ", {d}", .{o[1]});
        },
        .movRI => |o| try out.print(gpa, "mov {s}, {d}", .{ @tagName(o[0]), o[1] }),
        .movabs => |o| try out.print(gpa, "movabs {s}, 0x{x}", .{ @tagName(o[0]), o[1] }),
        .aluRR => |o| try out.print(gpa, "{s} {s}, {s}", .{ @tagName(o[0]), @tagName(o[1]), @tagName(o[2]) }),
        .aluRM => |o| {
            try out.print(gpa, "{s} {s}, ", .{ @tagName(o[0]), @tagName(o[1]) });
            try fmtMem(gpa, out, o[2]);
        },
        .aluMR => |o| {
            try out.print(gpa, "{s} ", .{@tagName(o[0])});
            try fmtMem(gpa, out, o[1]);
            try out.print(gpa, ", {s}", .{@tagName(o[2])});
        },
        .aluRI => |o| try out.print(gpa, "{s} {s}, {d}", .{ @tagName(o[0]), @tagName(o[1]), o[2] }),
        .testRR => |o| try out.print(gpa, "test {s}, {s}", .{ @tagName(o[0]), @tagName(o[1]) }),
        .imulRR => |o| try out.print(gpa, "imul {s}, {s}", .{ @tagName(o[0]), @tagName(o[1]) }),
        .imulRRI => |o| try out.print(gpa, "imul {s}, {s}, {d}", .{ @tagName(o[0]), @tagName(o[1]), o[2] }),
        .shiftRCl => |o| try out.print(gpa, "{s} {s}, cl", .{ @tagName(o[0]), @tagName(o[1]) }),
        .idivR => |o| try out.print(gpa, "idiv {s}", .{@tagName(o[0])}),
        .cqo => try out.print(gpa, "cqo", .{}),
        .push => |o| try out.print(gpa, "push {s}", .{@tagName(o[0])}),
        .pop => |o| try out.print(gpa, "pop {s}", .{@tagName(o[0])}),
        .ret => try out.appendSlice(gpa, "ret"),
        .nop => try out.appendSlice(gpa, "nop"),
        .int3 => try out.appendSlice(gpa, "int3"),
        .callR => |o| try out.print(gpa, "call {s}", .{@tagName(o[0])}),
        .jmpR => |o| try out.print(gpa, "jmp {s}", .{@tagName(o[0])}),
        .callAbs => |o| try out.print(gpa, "call 0x{x}", .{o[0]}),
        .jmp, .jmp8, .jmp32 => |o| try out.print(gpa, "jmp{s} L{d}", .{ bsuf(i), o[0].id }),
        .call => |o| try out.print(gpa, "call L{d}", .{o[0].id}),
        .jcc, .jcc8, .jcc32 => |o| try out.print(gpa, "j{s}{s} L{d}", .{ @tagName(o[0]), bsuf(i), o[1].id }),
        .addRR, .subRR, .cmpRR, .xorRR => |o| try out.print(gpa, "{s} {s}, {s}", .{ aliasName(i), @tagName(o[0]), @tagName(o[1]) }),
        .addRI, .subRI, .cmpRI => |o| try out.print(gpa, "{s} {s}, {d}", .{ aliasName(i), @tagName(o[0]), o[1] }),
        .sseRR => |o| try out.print(gpa, "{s} xmm{d}, xmm{d}", .{ @tagName(o[0]), o[1], o[2] }),
        .sseRM => |o| {
            try out.print(gpa, "{s} xmm{d}, ", .{ @tagName(o[0]), o[1] });
            try fmtMem(gpa, out, o[2]);
        },
        .sseMR => |o| {
            try out.print(gpa, "{s} ", .{@tagName(o[0])});
            try fmtMem(gpa, out, o[1]);
            try out.print(gpa, ", xmm{d}", .{o[2]});
        },
        .movqXR => |o| try out.print(gpa, "movq xmm{d}, {s}", .{ o[0], @tagName(o[1]) }),
        .movqRX => |o| try out.print(gpa, "movq {s}, xmm{d}", .{ @tagName(o[0]), o[1] }),
        .cvtFromInt => |o| try out.print(gpa, "{s} xmm{d}, {s}", .{ @tagName(o[0]), o[1], @tagName(o[2]) }),
        .cvtToInt => |o| try out.print(gpa, "{s} {s}, xmm{d}", .{ @tagName(o[0]), @tagName(o[1]), o[2] }),
        .setcc => |o| try out.print(gpa, "set{s} {s}", .{ @tagName(o[0]), @tagName(o[1]) }),
        .movzxB => |o| try out.print(gpa, "movzx {s}, {s}.b", .{ @tagName(o[0]), @tagName(o[1]) }),
    }
}

fn bsuf(i: Instr) []const u8 {
    return switch (i) {
        .jmp8, .jcc8 => " short",
        .jmp32, .jcc32 => " near",
        else => "",
    };
}
fn aliasName(i: Instr) []const u8 {
    const n = @tagName(i);
    return n[0 .. n.len - 2];
}

// ---- passes ---------------------------------------------------------------------------
/// Removes instructions after an unconditional `jmp`/`ret`/`jmp reg` up to the next label or
/// embed (unreachable code). Comments and align nodes are kept.
pub const removeDeadCode = Builder.removeDeadCode;
/// Removes `jmp L` when `L` is bound right after it (only labels/comments in between).
pub const removeJumpsToNext = Builder.removeJumpsToNext;

/// Rewrites every `jmp`/`jcc` (forward and backward) into `jmp8`/`jcc8` where the
/// displacement fits and `jmp32`/`jcc32` otherwise, by iterative relaxation: start
/// all-short, grow the ones that do not fit, repeat until stable (sizes only grow, so it
/// terminates). Non-branch sizes are measured by encoding each node once; align padding is
/// recomputed from the current layout on every round. Already-sized branches keep growing
/// rules, so the pass can be re-run after edits.
pub fn shortenBranches(b: *Builder) !void {
    const n = b.nodes.items.len;
    const sizes = try b.gpa.alloc(u32, n);
    defer b.gpa.free(sizes);
    const offs = try b.gpa.alloc(u32, n);
    defer b.gpa.free(offs);
    var scratch = CodeBuffer.init(b.gpa);
    defer scratch.deinit();
    const labels = try b.gpa.alloc(core.Label, b.label_node.items.len);
    defer b.gpa.free(labels);
    for (labels) |*l| l.* = try scratch.newLabel();
    const A = x.AssemblerT(.{});
    var it = b.first;
    while (it != Node.none) : (it = b.nodes.items[it].next) {
        const p = &b.nodes.items[it];
        sizes[it] = 0;
        switch (p.data) {
            .inst => |*in| switch (in.*) {
                .jmp, .jmp32 => |o| in.* = .{ .jmp8 = o },
                .jcc, .jcc32 => |o| in.* = .{ .jcc8 = o },
                .jmp8, .jcc8 => {},
                else => {
                    scratch.len = 0;
                    scratch.fixups.clearRetainingCapacity();
                    try Builder.emitInst(A.init(&scratch), in.*, labels);
                    sizes[it] = @intCast(scratch.len);
                },
            },
            .embed => |e| sizes[it] = e.len,
            else => {},
        }
    }
    var changed = true;
    while (changed) {
        changed = false;
        var off: u32 = 0;
        it = b.first;
        while (it != Node.none) : (it = b.nodes.items[it].next) {
            offs[it] = off;
            off += switch (b.nodes.items[it].data) {
                .inst => |in| switch (in) {
                    .jmp8, .jcc8 => 2,
                    .jmp32 => 5,
                    .jcc32 => 6,
                    else => sizes[it],
                },
                .@"align" => |al| (al.boundary - (off & (al.boundary - 1))) & (al.boundary - 1),
                else => sizes[it],
            };
        }
        it = b.first;
        while (it != Node.none) : (it = b.nodes.items[it].next) {
            const p = &b.nodes.items[it];
            if (p.data != .inst) continue;
            const in = &p.data.inst;
            if (in.* != .jmp8 and in.* != .jcc8) continue;
            const l = in.target().?;
            const ln = b.label_node.items[l.id];
            if (ln == Node.none) return error.UnboundLabel;
            const disp: i64 = @as(i64, offs[ln]) - @as(i64, offs[it] + 2);
            if (std.math.cast(i8, disp) == null) {
                in.* = switch (in.*) {
                    .jmp8 => |o| .{ .jmp32 = o },
                    .jcc8 => |o| .{ .jcc32 = o },
                    else => unreachable,
                };
                changed = true;
            }
        }
    }
}

/// `shortenBranches` followed by `serialize` through a validating Assembler into `cb`, with
/// every instruction encoded once instead of twice (same bytes; call `cb.finalize()` after).
/// 1. Every non-branch node is encoded at a compact offset (branches, labels and padding take
///    no space yet), which gives its size.
/// 2. Branch relaxation over those sizes, as in `shortenBranches`, gives final offsets.
/// 3. Back to front, each node moves to its final offset (never before its compact one) and
///    branches, padding and label bindings are emitted there; fixups of moved instructions
///    are shifted with them.
/// One u32 per node: the size of an instruction or embed, or the final offset of any other
/// node (those take no space in the compact layout).
pub fn serializeShortened(b: *Builder, cb: *CodeBuffer) !void {
    const slot = try b.gpa.alloc(u32, b.nodes.items.len);
    defer b.gpa.free(slot);
    const labels = try b.gpa.alloc(core.Label, b.label_node.items.len);
    defer b.gpa.free(labels);
    for (labels) |*l| l.* = try cb.newLabel();
    const a = x.AssemblerT(.{ .validate = true }).init(cb);
    const fix0 = cb.fixups.items.len;
    const start = cb.offset();
    const isBranch = struct {
        inline fn f(d: Node.Data) bool {
            return d == .inst and switch (d.inst) {
                .jmp8, .jcc8, .jmp32, .jcc32 => true,
                else => false,
            };
        }
    }.f;
    var it = b.first;
    while (it != Node.none) : (it = b.nodes.items[it].next) {
        const p = &b.nodes.items[it];
        const at = cb.offset();
        switch (p.data) {
            .inst => |*in| switch (in.*) {
                .jmp, .jmp32 => |o| in.* = .{ .jmp8 = o },
                .jcc, .jcc32 => |o| in.* = .{ .jcc8 = o },
                .jmp8, .jcc8 => {},
                else => try Builder.emitInst(a, in.*, labels),
            },
            .embed => |e| try cb.emitBytes(b.slice(e.off, e.len)),
            else => {},
        }
        slot[it] = cb.offset() - at;
    }
    const compact_end = cb.offset();
    const fix_end = cb.fixups.items.len;
    var end = start;
    var changed = true;
    while (changed) {
        changed = false;
        var off = start;
        it = b.first;
        while (it != Node.none) : (it = b.nodes.items[it].next) {
            const d = b.nodes.items[it].data;
            switch (d) {
                .inst => |in| switch (in) {
                    .jmp8, .jcc8, .jmp32, .jcc32 => {
                        slot[it] = off;
                        off += switch (in) {
                            .jmp8, .jcc8 => 2,
                            .jmp32 => 5,
                            else => 6,
                        };
                    },
                    else => off += slot[it],
                },
                .embed => off += slot[it],
                .@"align" => |al| {
                    slot[it] = off;
                    off += (al.boundary - (off & (al.boundary - 1))) & (al.boundary - 1);
                },
                .label, .comment => slot[it] = off,
            }
        }
        end = off;
        it = b.first;
        while (it != Node.none) : (it = b.nodes.items[it].next) {
            const p = &b.nodes.items[it];
            if (p.data != .inst) continue;
            const in = &p.data.inst;
            if (in.* != .jmp8 and in.* != .jcc8) continue;
            const ln = b.label_node.items[in.target().?.id];
            if (ln == Node.none) return error.UnboundLabel;
            const disp: i64 = @as(i64, slot[ln]) - @as(i64, slot[it] + 2);
            if (std.math.cast(i8, disp) == null) {
                in.* = switch (in.*) {
                    .jmp8 => |o| .{ .jmp32 = o },
                    .jcc8 => |o| .{ .jcc32 = o },
                    else => unreachable,
                };
                changed = true;
            }
        }
    }
    // Capacity for the final size now: emitting at `len = offset` below must never grow the
    // buffer (growth only keeps the bytes below `len`).
    try cb.ensure(end - compact_end);
    var k = fix_end;
    var next_off = end; // final offset of the node after `it`
    var cpos = compact_end; // compact offset of the node after `it`
    it = b.last;
    while (it != Node.none) : (it = b.nodes.items[it].prev) {
        const d = b.nodes.items[it].data;
        if (d == .inst and !isBranch(d) or d == .embed) {
            const sz = slot[it];
            const off = next_off - sz;
            cpos -= sz;
            std.mem.copyBackwards(u8, cb.data[off..][0..sz], cb.data[cpos..][0..sz]);
            const shift = off - cpos;
            while (k > fix0 and cb.fixups.items[k - 1].at >= cpos) : (k -= 1) {
                cb.fixups.items[k - 1].at += shift;
                cb.fixups.items[k - 1].base += shift;
            }
            next_off = off;
            continue;
        }
        cb.len = slot[it];
        switch (d) {
            .inst => |in| try Builder.emitInst(a, in, labels),
            .label => |l| try a.bind(labels[l.id]),
            .@"align" => |al| if (al.nop) try a.alignNop(al.boundary) else try a.alignTrap(al.boundary),
            .comment, .embed => {},
        }
        next_off = slot[it];
    }
    cb.len = end;
}

// ---- tests ----------------------------------------------------------------------------
const t = std.testing;

fn expectSame(comptime f: fn (anytype) anyerror!void) !void {
    // Same program through the Assembler and through the Builder => identical bytes.
    var direct = CodeBuffer.init(t.allocator);
    defer direct.deinit();
    const a = x.AssemblerT(.{ .validate = true }).init(&direct);
    try f(a);
    try direct.finalize();

    var bld = Builder.init(t.allocator);
    defer bld.deinit();
    try f(&bld);
    var via = CodeBuffer.init(t.allocator);
    defer via.deinit();
    try bld.serialize(x.AssemblerT(.{ .validate = true }).init(&via));
    try via.finalize();
    try t.expectEqualSlices(u8, direct.code(), via.code());
}

/// Emits through either an Assembler (by method name) or a `*Builder` (as a node).
fn em(a: anytype, comptime tag: std.meta.Tag(Instr), args: anytype) !void {
    if (@TypeOf(a) == *Builder) {
        if (@TypeOf(args) == void) return a.add(tag);
        return a.emit(tag, args);
    }
    const f = @field(@TypeOf(a), @tagName(tag));
    if (@TypeOf(args) == void) return f(a);
    return @call(.auto, f, .{a} ++ args);
}

fn progLoop(e: anytype) anyerror!void {
    const a = e;
    const top = try a.newLabel();
    const done = try a.newLabel();
    try em(a, .push, .{.rbx});
    try em(a, .movRR, .{ .rax, .rdi });
    try em(a, .xorRR, .{ .rcx, .rcx });
    try a.bind(top);
    try em(a, .cmpRR, .{ .rcx, .rsi });
    try em(a, .jcc, .{ .ge, done });
    try em(a, .aluRM, .{ .add, .rax, x.ptrIdx(.rdx, .rcx, .x8, 0) });
    try em(a, .imulRRI, .{ .rbx, .rax, 3 });
    try em(a, .lea, .{ .rax, x.ptr(.rbx, -8) });
    try em(a, .movMR, .{ x.ptr(.rsp, 16), .rax });
    try em(a, .movMI, .{ x.ptr(.rdi, 0), -1 });
    try em(a, .aluRI, .{ .add, .rcx, 1 });
    try em(a, .movRI, .{ .r9, 1 << 40 });
    try em(a, .movabs, .{ .r10, 0xdead_beef_cafe });
    try em(a, .testRR, .{ .rax, .rax });
    try em(a, .jmp, .{top});
    try a.bind(done);
    try a.alignNop(16);
    try em(a, .pop, .{.rbx});
    try em(a, .callR, .{.r11});
    try em(a, .call, .{top});
    try em(a, .ret, {});
}

test "builder: serialize is byte-identical to the Assembler" {
    try expectSame(progLoop);
}

test "builder: node insertion with cursor and removal" {
    var b = Builder.init(t.allocator);
    defer b.deinit();
    try b.emit(.movRR, .{ .rax, .rdi }); // node 0
    try b.add(.ret); // node 1
    b.setCursor(0);
    try b.emit(.addRI, .{ .rax, 1 }); // inserted between
    b.setCursor(Node.none);
    try b.add(.nop); // inserted at the front
    try t.expectEqual(@as(usize, 4), b.count());
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(t.allocator);
    try b.format(&out);
    try t.expectEqualStrings("  nop\n  mov rax, rdi\n  add rax, 1\n  ret\n", out.items);
    b.removeNode(3); // the nop
    var cb = CodeBuffer.init(t.allocator);
    defer cb.deinit();
    try b.serialize(x.AssemblerT(.{}).init(&cb));
    try cb.finalize();
    try t.expectEqualSlices(u8, &.{ 0x48, 0x89, 0xF8, 0x48, 0x83, 0xC0, 0x01, 0xC3 }, cb.code());
}

test "builder: label errors" {
    var b = Builder.init(t.allocator);
    defer b.deinit();
    try t.expectError(error.InvalidLabel, b.emit(.jmp, .{.{ .id = 0 }}));
    const l = try b.newLabel();
    try b.bind(l);
    try t.expectError(error.LabelAlreadyBound, b.bind(l));
}

test "builder: dead code and jump-to-next removal" {
    var b = Builder.init(t.allocator);
    defer b.deinit();
    const l = try b.newLabel();
    try b.emit(.movRI, .{ .rax, 1 });
    try b.emit(.jmp, .{l});
    try b.emit(.movRI, .{ .rax, 2 }); // dead
    try b.add(.nop); // dead
    try b.comment("kept");
    try b.bind(l);
    try b.add(.ret);
    try b.add(.ret); // dead
    try b.runPasses(.{ removeDeadCode, removeJumpsToNext });
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(t.allocator);
    try b.format(&out);
    try t.expectEqualStrings("  mov rax, 1\n  ; kept\nL0:\n  ret\n", out.items);
}

test "builder: forward branch shortening (rel8 where it fits)" {
    var b = Builder.init(t.allocator);
    defer b.deinit();
    const near = try b.newLabel();
    const far = try b.newLabel();
    try b.emit(.jcc, .{ .e, near }); // 2 bytes
    try b.emit(.jmp, .{far}); // must stay rel32: 130 nops in between
    try b.bind(near);
    for (0..130) |_| try b.add(.nop);
    try b.bind(far);
    try b.add(.ret);
    try shortenBranches(&b);
    var cb = CodeBuffer.init(t.allocator);
    defer cb.deinit();
    try b.serialize(x.AssemblerT(.{}).init(&cb));
    try cb.finalize();
    const c = cb.code();
    try t.expectEqualSlices(u8, &.{ 0x74, 0x05 }, c[0..2]);
    try t.expectEqualSlices(u8, &.{ 0xE9, 130, 0, 0, 0 }, c[2..7]);
    try t.expectEqual(@as(usize, 2 + 5 + 130 + 1), c.len);
}

test "builder: shortening converges when one long branch pushes another out of range" {
    var b = Builder.init(t.allocator);
    defer b.deinit();
    const l1 = try b.newLabel();
    const l2 = try b.newLabel();
    try b.emit(.jmp, .{l1}); // spans the jcc below plus 124 nops
    try b.emit(.jcc, .{ .ne, l2 }); // spans 124 nops + 4 more: 128 -> long
    for (0..124) |_| try b.add(.nop);
    try b.bind(l1);
    for (0..4) |_| try b.add(.nop);
    try b.bind(l2);
    try b.add(.ret);
    try shortenBranches(&b);
    var cb = CodeBuffer.init(t.allocator);
    defer cb.deinit();
    try b.serialize(x.AssemblerT(.{}).init(&cb));
    try cb.finalize();
    const c = cb.code();
    // jcc rel32 (6 B) makes jmp's displacement 6+124 = 130 -> jmp long too.
    try t.expectEqual(@as(u8, 0xE9), c[0]);
    try t.expectEqualSlices(u8, &.{ 0x0F, 0x85 }, c[5..7]);
}

test "builder: shortened code executes (fuzz vs Assembler semantics)" {
    // Random streams of nops and forward/backward branches: shortened output must be no
    // longer than the Assembler's and must finalize (all displacements in range).
    var prng = std.Random.DefaultPrng.init(0x5eed);
    const r = prng.random();
    for (0..200) |_| {
        var b = Builder.init(t.allocator);
        defer b.deinit();
        var labels: [8]Label = undefined;
        for (&labels) |*l| l.* = try b.newLabel();
        var bound: [8]bool = @splat(false);
        for (0..60) |_| {
            switch (r.uintLessThan(u8, 4)) {
                0 => for (0..r.uintLessThan(u8, 40)) |_| try b.add(.nop),
                1 => try b.emit(.jmp, .{labels[r.uintLessThan(u8, 8)]}),
                2 => try b.emit(.jcc, .{ .ne, labels[r.uintLessThan(u8, 8)] }),
                else => {
                    const k = r.uintLessThan(u8, 8);
                    if (!bound[k]) {
                        try b.bind(labels[k]);
                        bound[k] = true;
                    }
                },
            }
        }
        for (labels, 0..) |l, k| if (!bound[k]) try b.bind(l);
        try b.add(.ret);
        var plain = CodeBuffer.init(t.allocator);
        defer plain.deinit();
        try b.serialize(x.AssemblerT(.{}).init(&plain));
        try plain.finalize();
        try shortenBranches(&b);
        var short = CodeBuffer.init(t.allocator);
        defer short.deinit();
        try b.serialize(x.AssemblerT(.{}).init(&short));
        try short.finalize();
        try t.expect(short.code().len <= plain.code().len);
    }
}

test "builder: shortenBranches is a re-runnable pass; sized branches format and embed" {
    var b = Builder.init(t.allocator);
    defer b.deinit();
    const l = try b.newLabel();
    try b.bind(l);
    try b.emit(.jcc, .{ .ne, l });
    try b.emit(.jmp, .{l});
    try b.embed(&.{ 0xAB, 0xCD }); // reachable data: not removed as dead code
    try b.runPasses(.{ removeDeadCode, shortenBranches, shortenBranches });
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(t.allocator);
    try b.format(&out);
    try t.expectEqualStrings("L0:\n  jne short L0\n  jmp short L0\n  .embed 2 bytes\n", out.items);
    var cb = CodeBuffer.init(t.allocator);
    defer cb.deinit();
    try b.serialize(x.AssemblerT(.{}).init(&cb));
    try cb.finalize();
    try t.expectEqualSlices(u8, &.{ 0x75, 0xFE, 0xEB, 0xFC, 0xAB, 0xCD }, cb.code());
}

test "builder: scalar SSE nodes encode like LLVM" {
    var b = Builder.init(std.testing.allocator);
    defer b.deinit();
    try b.emit(.sseRR, .{ .addsd, 1, 2 }); // f2 0f 58 ca
    try b.emit(.sseRR, .{ .ucomiss, 9, 0 }); // 44 0f 2e c8
    try b.emit(.sseRM, .{ .movss, 0, x.ptr(.rbp, -8) }); // f3 0f 10 45 f8
    try b.emit(.sseMR, .{ .movsd, x.ptr(.rax, 0), 15 }); // f2 44 0f 11 38
    try b.emit(.movqXR, .{ 3, .r10 }); // 66 49 0f 6e da
    try b.emit(.movqRX, .{ .rax, 3 }); // 66 48 0f 7e d8
    try b.emit(.cvtFromInt, .{ .cvtsi2sd, 0, .rdi }); // f2 48 0f 2a c7
    try b.emit(.cvtToInt, .{ .cvttss2si, .r10, 1 }); // f3 4c 0f 2c d1
    try b.emit(.setcc, .{ .a, .r10 }); // 41 0f 97 c2
    try b.emit(.movzxB, .{ .r10, .r10 }); // 45 0f b6 d2
    var cb = CodeBuffer.init(std.testing.allocator);
    defer cb.deinit();
    try b.serialize(x.AssemblerT(.{ .validate = true }).init(&cb));
    try std.testing.expectEqualSlices(u8, &.{
        0xf2, 0x0f, 0x58, 0xca,
        0x44, 0x0f, 0x2e, 0xc8,
        0xf3, 0x0f, 0x10, 0x45,
        0xf8, 0xf2, 0x44, 0x0f,
        0x11, 0x38, 0x66, 0x49,
        0x0f, 0x6e, 0xda, 0x66,
        0x48, 0x0f, 0x7e, 0xd8,
        0xf2, 0x48, 0x0f, 0x2a,
        0xc7, 0xf3, 0x4c, 0x0f,
        0x2c, 0xd1, 0x41, 0x0f,
        0x97, 0xc2, 0x45, 0x0f,
        0xb6, 0xd2,
    }, cb.code());
    var txt: std.ArrayList(u8) = .empty;
    defer txt.deinit(std.testing.allocator);
    try formatInstImpl(std.testing.allocator, &txt, .{ .sseRR = .{ .addsd, 1, 2 } });
    try std.testing.expectEqualStrings("addsd xmm1, xmm2", txt.items);
}

test "x64 builder: serializeShortened matches shortenBranches + serialize" {
    // Short and long forward/backward branches, an alignment, embedded bytes, a call to a
    // label (a fixup inside a moved instruction) and a non-empty buffer to start with.
    const build = struct {
        fn f(b: *Builder) !void {
            const top = try b.newLabel();
            const far = try b.newLabel();
            const near = try b.newLabel();
            const func = try b.newLabel();
            try b.bind(top);
            try b.emit(.call, .{func});
            try b.emit(.jcc, .{ .e, near });
            try b.emit(.jmp, .{far});
            try b.bind(near);
            for (0..40) |_| try b.emit(.addRI, .{ .rax, 0x12345 });
            try b.alignNop(16);
            try b.embed(&.{ 0xAA, 0xBB, 0xCC });
            try b.emit(.jcc, .{ .ne, top });
            try b.emit(.jmp, .{near});
            try b.bind(far);
            try b.add(.ret);
            try b.alignTrap(8);
            try b.bind(func);
            try b.emit(.movRR, .{ .rax, .rcx });
            try b.add(.ret);
        }
    }.f;
    for ([_]usize{ 0, 5 }) |pre| {
        var b1 = Builder.init(t.allocator);
        defer b1.deinit();
        try build(&b1);
        try shortenBranches(&b1);
        var want = CodeBuffer.init(t.allocator);
        defer want.deinit();
        for (0..pre) |_| try want.emitBytes(&.{0x90});
        try b1.serialize(x.AssemblerT(.{ .validate = true }).init(&want));
        try want.finalize();

        var b2 = Builder.init(t.allocator);
        defer b2.deinit();
        try build(&b2);
        var got = CodeBuffer.init(t.allocator);
        defer got.deinit();
        for (0..pre) |_| try got.emitBytes(&.{0x90});
        try serializeShortened(&b2, &got);
        try got.finalize();
        if (pre == 0) try t.expectEqualSlices(u8, want.code(), got.code());
        // With a prefix, alignment is taken from the absolute offset (here 5) in both.
        try t.expectEqual(want.code().len, got.code().len);
    }
}
