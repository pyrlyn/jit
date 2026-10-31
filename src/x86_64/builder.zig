//! x86-64 Builder: records instructions as nodes instead of bytes, runs passes over the node
//! list, then serializes it to an Assembler (like AsmJit's BaseBuilder / x86::Builder).
//!
//! - Nodes live in one array and are linked by u32 indices (`prev`/`next`), so inserting and
//!   removing is O(1) and nothing is reallocated per node.
//! - The emitter API mirrors the Assembler (`b.movRR(...)`, `b.aluRI(.add, ...)`, `b.jcc(.ne, l)`).
//! - Instructions are inserted after the cursor (`setCursor`), so code can be added anywhere.
//! - `serialize` replays nodes through an `AssemblerT(opts)`: with no pass run, output is
//!   byte-identical to calling the Assembler directly (validation and hooks work unchanged).
//! - Passes are plain functions taking `*Builder`, run with `runPasses(.{ ... })`.
const std = @import("std");
const x = @import("assembler.zig");
const core = @import("../core/code_buffer.zig");
const CodeBuffer = core.CodeBuffer;
const Gp = x.Gp;
const Mem = x.Mem;
const Alu = x.Alu;
const Cond = x.Cond;

pub const Error = x.Error;

/// Builder-side label (index into the builder's label table; mapped to buffer labels at serialize).
pub const Label = struct { id: u32 };

/// One recorded instruction. Same set as the Assembler's public emitters (`x.Inst`).
pub const Instr = union(enum) {
    mov_rr: struct { Gp, Gp },
    mov_rm: struct { Gp, Mem },
    mov_mr: struct { Mem, Gp },
    mov_mi: struct { Mem, i32 },
    lea: struct { Gp, Mem },
    mov_ri: struct { Gp, i64 },
    movabs: struct { Gp, u64 },
    alu_rr: struct { Alu, Gp, Gp },
    alu_rm: struct { Alu, Gp, Mem },
    alu_mr: struct { Alu, Mem, Gp },
    alu_ri: struct { Alu, Gp, i32 },
    test_rr: struct { Gp, Gp },
    imul_rr: struct { Gp, Gp },
    imul_rm: struct { Gp, Mem },
    imul_rri: struct { Gp, Gp, i32 },
    push: Gp,
    pop: Gp,
    ret,
    nop,
    int3,
    call_r: Gp,
    jmp_r: Gp,
    call_abs: usize,
    jmp: Label,
    call: Label,
    jcc: struct { Cond, Label },

    pub fn target(i: Instr) ?Label {
        return switch (i) {
            .jmp, .call => |l| l,
            .jcc => |c| c[1],
            else => null,
        };
    }
    /// Control never falls through to the next node.
    pub fn isTerminator(i: Instr) bool {
        return switch (i) {
            .jmp, .jmp_r, .ret => true,
            else => false,
        };
    }
};

pub const BranchSize = enum(u8) { auto, short, long };

pub const Node = struct {
    pub const none: u32 = std.math.maxInt(u32);
    data: Data,
    prev: u32 = none,
    next: u32 = none,

    pub const Data = union(enum) {
        inst: struct { i: Instr, size: BranchSize = .auto },
        label: Label,
        /// Pad with NOPs (`nop = true`) or int3 to a power-of-two boundary.
        @"align": struct { boundary: u32, nop: bool = true },
        /// Raw bytes (copied into the builder's byte arena).
        embed: struct { off: u32, len: u32 },
        comment: struct { off: u32, len: u32 },
    };
};

pub const Builder = struct {
    gpa: std.mem.Allocator,
    nodes: std.ArrayList(Node) = .empty,
    bytes: std.ArrayList(u8) = .empty,
    first: u32 = Node.none,
    last: u32 = Node.none,
    /// New nodes are inserted after this node (`none` = at the front).
    cursor: u32 = Node.none,
    label_count: u32 = 0,
    /// Label -> node that binds it (`none` until bound).
    label_node: std.ArrayList(u32) = .empty,

    pub fn init(gpa: std.mem.Allocator) Builder {
        return .{ .gpa = gpa };
    }
    pub fn deinit(b: *Builder) void {
        b.nodes.deinit(b.gpa);
        b.bytes.deinit(b.gpa);
        b.label_node.deinit(b.gpa);
    }

    // ---- node list --------------------------------------------------------------------
    pub fn newLabel(b: *Builder) Error!Label {
        try b.label_node.append(b.gpa, Node.none);
        b.label_count += 1;
        return .{ .id = b.label_count - 1 };
    }

    /// Inserts `d` after the cursor and moves the cursor to it. Returns the node index.
    pub fn addNode(b: *Builder, d: Node.Data) Error!u32 {
        const id: u32 = @intCast(b.nodes.items.len);
        try b.nodes.append(b.gpa, .{ .data = d });
        const n = &b.nodes.items[id];
        if (b.cursor == Node.none) {
            n.next = b.first;
            if (b.first != Node.none) b.nodes.items[b.first].prev = id else b.last = id;
            b.first = id;
        } else {
            const c = &b.nodes.items[b.cursor];
            n.prev = b.cursor;
            n.next = c.next;
            if (c.next != Node.none) b.nodes.items[c.next].prev = id else b.last = id;
            c.next = id;
        }
        b.cursor = id;
        return id;
    }

    /// Unlinks node `id` (its slot stays allocated). A cursor on it moves to its predecessor.
    pub fn removeNode(b: *Builder, id: u32) void {
        const n = b.nodes.items[id];
        if (n.prev != Node.none) b.nodes.items[n.prev].next = n.next else b.first = n.next;
        if (n.next != Node.none) b.nodes.items[n.next].prev = n.prev else b.last = n.prev;
        if (b.cursor == id) b.cursor = n.prev;
        if (n.data == .label) b.label_node.items[n.data.label.id] = Node.none;
        b.nodes.items[id].prev = Node.none;
        b.nodes.items[id].next = Node.none;
    }

    /// Next inserts go after `id` (`Node.none` = before the first node).
    pub fn setCursor(b: *Builder, id: u32) void {
        b.cursor = id;
    }
    pub fn node(b: *const Builder, id: u32) *Node {
        return &b.nodes.items[id];
    }
    /// Number of linked nodes.
    pub fn count(b: *const Builder) usize {
        var n: usize = 0;
        var it = b.first;
        while (it != Node.none) : (it = b.nodes.items[it].next) n += 1;
        return n;
    }

    pub fn bind(b: *Builder, l: Label) Error!void {
        if (l.id >= b.label_count) return error.InvalidLabel;
        if (b.label_node.items[l.id] != Node.none) return error.LabelAlreadyBound;
        const id = try b.addNode(.{ .label = l });
        b.label_node.items[l.id] = id;
    }
    pub fn alignNop(b: *Builder, boundary: u32) Error!void {
        if (!std.math.isPowerOfTwo(boundary)) return error.InvalidOperand;
        _ = try b.addNode(.{ .@"align" = .{ .boundary = boundary } });
    }
    pub fn alignTrap(b: *Builder, boundary: u32) Error!void {
        if (!std.math.isPowerOfTwo(boundary)) return error.InvalidOperand;
        _ = try b.addNode(.{ .@"align" = .{ .boundary = boundary, .nop = false } });
    }
    pub fn embed(b: *Builder, data: []const u8) Error!void {
        const off: u32 = @intCast(b.bytes.items.len);
        try b.bytes.appendSlice(b.gpa, data);
        _ = try b.addNode(.{ .embed = .{ .off = off, .len = @intCast(data.len) } });
    }
    pub fn comment(b: *Builder, text: []const u8) Error!void {
        const off: u32 = @intCast(b.bytes.items.len);
        try b.bytes.appendSlice(b.gpa, text);
        _ = try b.addNode(.{ .comment = .{ .off = off, .len = @intCast(text.len) } });
    }
    pub fn slice(b: *const Builder, off: u32, len: u32) []const u8 {
        return b.bytes.items[off..][0..len];
    }

    /// Records any instruction (the typed emitters below are thin wrappers).
    pub fn add(b: *Builder, i: Instr) Error!void {
        if (i.target()) |l| if (l.id >= b.label_count) return error.InvalidLabel;
        _ = try b.addNode(.{ .inst = .{ .i = i } });
    }

    // ---- emitters (same names and operand order as the Assembler) ---------------------
    pub fn movRR(b: *Builder, d: Gp, s: Gp) Error!void {
        return b.add(.{ .mov_rr = .{ d, s } });
    }
    pub fn movRM(b: *Builder, d: Gp, s: Mem) Error!void {
        return b.add(.{ .mov_rm = .{ d, s } });
    }
    pub fn movMR(b: *Builder, d: Mem, s: Gp) Error!void {
        return b.add(.{ .mov_mr = .{ d, s } });
    }
    pub fn movMI(b: *Builder, d: Mem, i: i32) Error!void {
        return b.add(.{ .mov_mi = .{ d, i } });
    }
    pub fn lea(b: *Builder, d: Gp, s: Mem) Error!void {
        return b.add(.{ .lea = .{ d, s } });
    }
    pub fn movRI(b: *Builder, d: Gp, i: i64) Error!void {
        return b.add(.{ .mov_ri = .{ d, i } });
    }
    pub fn movabs(b: *Builder, d: Gp, i: u64) Error!void {
        return b.add(.{ .movabs = .{ d, i } });
    }
    pub fn aluRR(b: *Builder, op: Alu, d: Gp, s: Gp) Error!void {
        return b.add(.{ .alu_rr = .{ op, d, s } });
    }
    pub fn aluRM(b: *Builder, op: Alu, d: Gp, s: Mem) Error!void {
        return b.add(.{ .alu_rm = .{ op, d, s } });
    }
    pub fn aluMR(b: *Builder, op: Alu, d: Mem, s: Gp) Error!void {
        return b.add(.{ .alu_mr = .{ op, d, s } });
    }
    pub fn aluRI(b: *Builder, op: Alu, d: Gp, i: i32) Error!void {
        return b.add(.{ .alu_ri = .{ op, d, i } });
    }
    pub fn addRR(b: *Builder, d: Gp, s: Gp) Error!void {
        return b.aluRR(.add, d, s);
    }
    pub fn addRI(b: *Builder, d: Gp, i: i32) Error!void {
        return b.aluRI(.add, d, i);
    }
    pub fn subRR(b: *Builder, d: Gp, s: Gp) Error!void {
        return b.aluRR(.sub, d, s);
    }
    pub fn subRI(b: *Builder, d: Gp, i: i32) Error!void {
        return b.aluRI(.sub, d, i);
    }
    pub fn cmpRR(b: *Builder, d: Gp, s: Gp) Error!void {
        return b.aluRR(.cmp, d, s);
    }
    pub fn cmpRI(b: *Builder, d: Gp, i: i32) Error!void {
        return b.aluRI(.cmp, d, i);
    }
    pub fn xorRR(b: *Builder, d: Gp, s: Gp) Error!void {
        return b.aluRR(.xor, d, s);
    }
    pub fn testRR(b: *Builder, d: Gp, s: Gp) Error!void {
        return b.add(.{ .test_rr = .{ d, s } });
    }
    pub fn imulRR(b: *Builder, d: Gp, s: Gp) Error!void {
        return b.add(.{ .imul_rr = .{ d, s } });
    }
    pub fn imulRM(b: *Builder, d: Gp, s: Mem) Error!void {
        return b.add(.{ .imul_rm = .{ d, s } });
    }
    pub fn imulRRI(b: *Builder, d: Gp, s: Gp, i: i32) Error!void {
        return b.add(.{ .imul_rri = .{ d, s, i } });
    }
    pub fn push(b: *Builder, r: Gp) Error!void {
        return b.add(.{ .push = r });
    }
    pub fn pop(b: *Builder, r: Gp) Error!void {
        return b.add(.{ .pop = r });
    }
    pub fn ret(b: *Builder) Error!void {
        return b.add(.ret);
    }
    pub fn nop(b: *Builder) Error!void {
        return b.add(.nop);
    }
    pub fn int3(b: *Builder) Error!void {
        return b.add(.int3);
    }
    pub fn callR(b: *Builder, r: Gp) Error!void {
        return b.add(.{ .call_r = r });
    }
    pub fn jmpR(b: *Builder, r: Gp) Error!void {
        return b.add(.{ .jmp_r = r });
    }
    pub fn callAbs(b: *Builder, addr: usize) Error!void {
        return b.add(.{ .call_abs = addr });
    }
    pub fn jmp(b: *Builder, l: Label) Error!void {
        return b.add(.{ .jmp = l });
    }
    pub fn call(b: *Builder, l: Label) Error!void {
        return b.add(.{ .call = l });
    }
    pub fn jcc(b: *Builder, c: Cond, l: Label) Error!void {
        return b.add(.{ .jcc = .{ c, l } });
    }

    // ---- passes -----------------------------------------------------------------------
    /// Runs each pass in order: `b.runPasses(.{ removeDeadCode, shortenBranches })`.
    pub fn runPasses(b: *Builder, comptime passes: anytype) !void {
        inline for (passes) |p| try p(b);
    }

    // ---- serialization ----------------------------------------------------------------
    /// Replays all nodes through `A` (an `x.AssemblerT(...)` instance) into `a.buf`.
    /// Builder labels get fresh buffer labels. Call `a.buf.finalize()` afterwards.
    pub fn serialize(b: *const Builder, a: anytype) !void {
        const buf = a.buf;
        const labels = try b.gpa.alloc(core.Label, b.label_count);
        defer b.gpa.free(labels);
        for (labels) |*l| l.* = try a.newLabel();
        var it = b.first;
        while (it != Node.none) : (it = b.nodes.items[it].next) {
            switch (b.nodes.items[it].data) {
                .label => |l| try a.bind(labels[l.id]),
                .@"align" => |al| if (al.nop) try a.alignNop(al.boundary) else try a.alignTrap(al.boundary),
                .embed => |e| try buf.emitBytes(b.slice(e.off, e.len)),
                .comment => {},
                .inst => |n| try emitInst(a, n.i, n.size, labels),
            }
        }
    }

    fn emitInst(a: anytype, i: Instr, size: BranchSize, labels: []const core.Label) !void {
        @setEvalBranchQuota(100_000); // many inlined encoders in one function
        switch (i) {
            .mov_rr => |o| try a.movRR(o[0], o[1]),
            .mov_rm => |o| try a.movRM(o[0], o[1]),
            .mov_mr => |o| try a.movMR(o[0], o[1]),
            .mov_mi => |o| try a.movMI(o[0], o[1]),
            .lea => |o| try a.lea(o[0], o[1]),
            .mov_ri => |o| try a.movRI(o[0], o[1]),
            .movabs => |o| try a.movabs(o[0], o[1]),
            .alu_rr => |o| switch (o[0]) {
                inline else => |op| try a.aluRR(op, o[1], o[2]),
            },
            .alu_rm => |o| switch (o[0]) {
                inline else => |op| try a.aluRM(op, o[1], o[2]),
            },
            .alu_mr => |o| switch (o[0]) {
                inline else => |op| try a.aluMR(op, o[1], o[2]),
            },
            .alu_ri => |o| switch (o[0]) {
                inline else => |op| try a.aluRI(op, o[1], o[2]),
            },
            .test_rr => |o| try a.testRR(o[0], o[1]),
            .imul_rr => |o| try a.imulRR(o[0], o[1]),
            .imul_rm => |o| try a.imulRM(o[0], o[1]),
            .imul_rri => |o| try a.imulRRI(o[0], o[1], o[2]),
            .push => |r| try a.push(r),
            .pop => |r| try a.pop(r),
            .ret => try a.ret(),
            .nop => try a.nop(),
            .int3 => try a.int3(),
            .call_r => |r| try a.callR(r),
            .jmp_r => |r| try a.jmpR(r),
            .call_abs => |p| try a.callAbs(p),
            .call => |l| try a.call(labels[l.id]),
            .jmp => |l| if (size == .auto) try a.jmp(labels[l.id]) else try rawBranch(a.buf, &.{0xEB}, &.{0xE9}, size, labels[l.id]),
            .jcc => |o| if (size == .auto) switch (o[0]) {
                inline else => |c| try a.jcc(c, labels[o[1].id]),
            } else {
                const cc: u8 = @backingInt(o[0]);
                try rawBranch(a.buf, &.{0x70 + cc}, &.{ 0x0F, 0x80 + cc }, size, labels[o[1].id]);
            },
        }
    }

    /// Branch with a size fixed by `shortenBranches` (bypasses the Assembler's choice).
    fn rawBranch(buf: *CodeBuffer, short_op: []const u8, long_op: []const u8, size: BranchSize, l: core.Label) Error!void {
        const op = if (size == .short) short_op else long_op;
        try buf.emitBytes(op);
        const at = buf.offset();
        const w: u32 = if (size == .short) 1 else 4;
        const zeros: [4]u8 = @splat(0);
        try buf.emitBytes(zeros[0..w]);
        try buf.addFixup(.{ .at = at, .base = at + w, .label = l.id, .kind = if (size == .short) .rel8 else .rel32 });
    }

    // ---- formatter --------------------------------------------------------------------
    /// Appends one line per node (`L0:`, `  mov rax, rdi`, `  ; text`) to `out`.
    pub fn format(b: *const Builder, out: *std.ArrayList(u8)) !void {
        var it = b.first;
        while (it != Node.none) : (it = b.nodes.items[it].next) {
            switch (b.nodes.items[it].data) {
                .label => |l| try out.print(b.gpa, "L{d}:\n", .{l.id}),
                .@"align" => |al| try out.print(b.gpa, "  .align {d}\n", .{al.boundary}),
                .embed => |e| try out.print(b.gpa, "  .embed {d} bytes\n", .{e.len}),
                .comment => |c| try out.print(b.gpa, "  ; {s}\n", .{b.slice(c.off, c.len)}),
                .inst => |n| {
                    try out.appendSlice(b.gpa, "  ");
                    try formatInst(b.gpa, out, n.i);
                    try out.append(b.gpa, '\n');
                },
            }
        }
    }
};

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

fn formatInst(gpa: std.mem.Allocator, out: *std.ArrayList(u8), i: Instr) !void {
    switch (i) {
        .mov_rr => |o| try out.print(gpa, "mov {s}, {s}", .{ @tagName(o[0]), @tagName(o[1]) }),
        .mov_rm, .lea => |o| {
            try out.print(gpa, "{s} {s}, ", .{ if (i == .lea) "lea" else "mov", @tagName(o[0]) });
            try fmtMem(gpa, out, o[1]);
        },
        .imul_rm => |o| {
            try out.print(gpa, "imul {s}, ", .{@tagName(o[0])});
            try fmtMem(gpa, out, o[1]);
        },
        .mov_mr => |o| {
            try out.appendSlice(gpa, "mov ");
            try fmtMem(gpa, out, o[0]);
            try out.print(gpa, ", {s}", .{@tagName(o[1])});
        },
        .mov_mi => |o| {
            try out.appendSlice(gpa, "mov qword ");
            try fmtMem(gpa, out, o[0]);
            try out.print(gpa, ", {d}", .{o[1]});
        },
        .mov_ri => |o| try out.print(gpa, "mov {s}, {d}", .{ @tagName(o[0]), o[1] }),
        .movabs => |o| try out.print(gpa, "movabs {s}, 0x{x}", .{ @tagName(o[0]), o[1] }),
        .alu_rr => |o| try out.print(gpa, "{s} {s}, {s}", .{ @tagName(o[0]), @tagName(o[1]), @tagName(o[2]) }),
        .alu_rm => |o| {
            try out.print(gpa, "{s} {s}, ", .{ @tagName(o[0]), @tagName(o[1]) });
            try fmtMem(gpa, out, o[2]);
        },
        .alu_mr => |o| {
            try out.print(gpa, "{s} ", .{@tagName(o[0])});
            try fmtMem(gpa, out, o[1]);
            try out.print(gpa, ", {s}", .{@tagName(o[2])});
        },
        .alu_ri => |o| try out.print(gpa, "{s} {s}, {d}", .{ @tagName(o[0]), @tagName(o[1]), o[2] }),
        .test_rr => |o| try out.print(gpa, "test {s}, {s}", .{ @tagName(o[0]), @tagName(o[1]) }),
        .imul_rr => |o| try out.print(gpa, "imul {s}, {s}", .{ @tagName(o[0]), @tagName(o[1]) }),
        .imul_rri => |o| try out.print(gpa, "imul {s}, {s}, {d}", .{ @tagName(o[0]), @tagName(o[1]), o[2] }),
        .push => |r| try out.print(gpa, "push {s}", .{@tagName(r)}),
        .pop => |r| try out.print(gpa, "pop {s}", .{@tagName(r)}),
        .ret => try out.appendSlice(gpa, "ret"),
        .nop => try out.appendSlice(gpa, "nop"),
        .int3 => try out.appendSlice(gpa, "int3"),
        .call_r => |r| try out.print(gpa, "call {s}", .{@tagName(r)}),
        .jmp_r => |r| try out.print(gpa, "jmp {s}", .{@tagName(r)}),
        .call_abs => |p| try out.print(gpa, "call 0x{x}", .{p}),
        .jmp => |l| try out.print(gpa, "jmp L{d}", .{l.id}),
        .call => |l| try out.print(gpa, "call L{d}", .{l.id}),
        .jcc => |o| try out.print(gpa, "j{s} L{d}", .{ @tagName(o[0]), o[1].id }),
    }
}

// ---- passes ---------------------------------------------------------------------------

/// Removes instructions after an unconditional `jmp`/`ret`/`jmp reg` up to the next label
/// (unreachable code). Comments, align and embed nodes are kept.
pub fn removeDeadCode(b: *Builder) !void {
    var dead = false;
    var it = b.first;
    while (it != Node.none) {
        const nx = b.nodes.items[it].next;
        switch (b.nodes.items[it].data) {
            .label => dead = false,
            .embed => dead = false,
            .inst => |n| {
                if (dead) b.removeNode(it) else if (n.i.isTerminator()) {
                    dead = true;
                }
            },
            else => {},
        }
        it = nx;
    }
}

/// Removes `jmp L` when `L` is bound right after it (only labels/comments in between).
pub fn removeJumpsToNext(b: *Builder) !void {
    var it = b.first;
    while (it != Node.none) {
        const nx = b.nodes.items[it].next;
        const d = b.nodes.items[it].data;
        if (d == .inst and d.inst.i == .jmp) {
            const l = d.inst.i.jmp;
            var j = nx;
            while (j != Node.none) : (j = b.nodes.items[j].next) {
                const dj = b.nodes.items[j].data;
                if (dj == .label) {
                    if (dj.label.id == l.id) {
                        b.removeNode(it);
                        break;
                    }
                } else if (dj != .comment) break;
            }
        }
        it = nx;
    }
}

/// Picks rel8 for every `jmp`/`jcc` (forward and backward) whose displacement fits, by
/// iterative relaxation: start all-short, grow the ones that do not fit, repeat until stable.
/// Non-branch sizes are measured by encoding each node once; align padding is recomputed
/// from the current layout on every round.
pub fn shortenBranches(b: *Builder) !void {
    const n = b.nodes.items.len;
    const sizes = try b.gpa.alloc(u32, n);
    defer b.gpa.free(sizes);
    const offs = try b.gpa.alloc(u32, n + 1);
    defer b.gpa.free(offs);
    // Measure fixed-size nodes.
    var scratch = CodeBuffer.init(b.gpa);
    defer scratch.deinit();
    const labels = try b.gpa.alloc(core.Label, b.label_count);
    defer b.gpa.free(labels);
    for (labels) |*l| l.* = try scratch.newLabel();
    const A = x.AssemblerT(.{});
    var it = b.first;
    while (it != Node.none) : (it = b.nodes.items[it].next) {
        const p = &b.nodes.items[it];
        sizes[it] = 0;
        switch (p.data) {
            .inst => |*in| switch (in.i) {
                .jmp, .jcc => in.size = .short,
                else => {
                    scratch.len = 0;
                    scratch.fixups.clearRetainingCapacity();
                    try Builder.emitInst(A.init(&scratch), in.i, .auto, labels);
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
        // Layout pass: offsets of nodes and label positions.
        var off: u32 = 0;
        it = b.first;
        while (it != Node.none) : (it = b.nodes.items[it].next) {
            offs[it] = off;
            const d = b.nodes.items[it].data;
            off += switch (d) {
                .inst => |in| switch (in.i) {
                    .jmp => if (in.size == .short) 2 else 5,
                    .jcc => if (in.size == .short) 2 else 6,
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
            if (in.size != .short) continue;
            const l = in.i.target() orelse continue;
            const ln = b.label_node.items[l.id];
            if (ln == Node.none) return error.UnboundLabel;
            const disp: i64 = @as(i64, offs[ln]) - @as(i64, offs[it] + 2);
            if (std.math.cast(i8, disp) == null) {
                in.size = .long;
                changed = true;
            }
        }
    }
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

fn progLoop(e: anytype) anyerror!void {
    const a = e;
    const top = try a.newLabel();
    const done = try a.newLabel();
    try a.push(.rbx);
    try a.movRR(.rax, .rdi);
    try a.xorRR(.rcx, .rcx);
    try a.bind(top);
    try a.cmpRR(.rcx, .rsi);
    try a.jcc(.ge, done);
    try a.aluRM(.add, .rax, x.ptrIdx(.rdx, .rcx, .x8, 0));
    try a.imulRRI(.rbx, .rax, 3);
    try a.lea(.rax, x.ptr(.rbx, -8));
    try a.movMR(x.ptr(.rsp, 16), .rax);
    try a.movMI(x.ptr(.rdi, 0), -1);
    try a.aluRI(.add, .rcx, 1);
    try a.movRI(.r9, 1 << 40);
    try a.movabs(.r10, 0xdead_beef_cafe);
    try a.testRR(.rax, .rax);
    try a.jmp(top);
    try a.bind(done);
    try a.alignNop(16);
    try a.pop(.rbx);
    try a.callR(.r11);
    try a.call(top);
    try a.ret();
}

test "builder: serialize is byte-identical to the Assembler" {
    try expectSame(progLoop);
}

test "builder: node insertion with cursor and removal" {
    var b = Builder.init(t.allocator);
    defer b.deinit();
    try b.movRR(.rax, .rdi); // node 0
    try b.ret(); // node 1
    b.setCursor(0);
    try b.addRI(.rax, 1); // inserted between
    b.setCursor(Node.none);
    try b.nop(); // inserted at the front
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
    try t.expectError(error.InvalidLabel, b.jmp(.{ .id = 0 }));
    const l = try b.newLabel();
    try b.bind(l);
    try t.expectError(error.LabelAlreadyBound, b.bind(l));
}

test "builder: dead code and jump-to-next removal" {
    var b = Builder.init(t.allocator);
    defer b.deinit();
    const l = try b.newLabel();
    try b.movRI(.rax, 1);
    try b.jmp(l);
    try b.movRI(.rax, 2); // dead
    try b.nop(); // dead
    try b.comment("kept");
    try b.bind(l);
    try b.ret();
    try b.ret(); // dead
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
    try b.jcc(.e, near); // 2 bytes
    try b.jmp(far); // must stay rel32: 130 nops in between
    try b.bind(near);
    for (0..130) |_| try b.nop();
    try b.bind(far);
    try b.ret();
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
    try b.jmp(l1); // spans the jcc below plus 124 nops
    try b.jcc(.ne, l2); // spans 124 nops + 4 more: 128 -> long
    for (0..124) |_| try b.nop();
    try b.bind(l1);
    for (0..4) |_| try b.nop();
    try b.bind(l2);
    try b.ret();
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
                0 => for (0..r.uintLessThan(u8, 40)) |_| try b.nop(),
                1 => try b.jmp(labels[r.uintLessThan(u8, 8)]),
                2 => try b.jcc(.ne, labels[r.uintLessThan(u8, 8)]),
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
        try b.ret();
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
