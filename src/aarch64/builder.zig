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

pub const Error = a64.Error || error{LabelAlreadyBound};

/// Builder-side label (mapped to buffer labels at serialize).
pub const Label = struct { id: u32 };

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

    pub fn target(i: Instr) ?Label {
        return switch (i) {
            .b, .bl => |o| o[0],
            .bcond => |o| o[1],
            .cbz, .cbnz => |o| o[1],
            else => null,
        };
    }
    /// Control never falls through to the next node.
    pub fn isTerminator(i: Instr) bool {
        return switch (i) {
            .b, .ret, .udf => true,
            else => false,
        };
    }
};

pub const Node = struct {
    pub const none: u32 = std.math.maxInt(u32);
    data: Data,
    prev: u32 = none,
    next: u32 = none,

    pub const Data = union(enum) {
        inst: Instr,
        label: Label,
        /// Pad with NOPs (`nop = true`) or UDF to a power-of-two boundary (multiple of 4).
        @"align": struct { boundary: u32, nop: bool = true },
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
        return .{ .id = @intCast(b.label_node.items.len - 1) };
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
    /// Unlinks node `id`. A cursor on it moves to its predecessor.
    pub fn removeNode(b: *Builder, id: u32) void {
        const n = b.nodes.items[id];
        if (n.prev != Node.none) b.nodes.items[n.prev].next = n.next else b.first = n.next;
        if (n.next != Node.none) b.nodes.items[n.next].prev = n.prev else b.last = n.prev;
        if (b.cursor == id) b.cursor = n.prev;
        if (n.data == .label) b.label_node.items[n.data.label.id] = Node.none;
        b.nodes.items[id].prev = Node.none;
        b.nodes.items[id].next = Node.none;
    }
    pub fn setCursor(b: *Builder, id: u32) void {
        b.cursor = id;
    }
    /// Number of linked nodes.
    pub fn count(b: *const Builder) usize {
        var n: usize = 0;
        var it = b.first;
        while (it != Node.none) : (it = b.nodes.items[it].next) n += 1;
        return n;
    }
    pub fn bind(b: *Builder, l: Label) Error!void {
        if (l.id >= b.label_node.items.len) return error.InvalidLabel;
        if (b.label_node.items[l.id] != Node.none) return error.LabelAlreadyBound;
        b.label_node.items[l.id] = try b.addNode(.{ .label = l });
    }
    pub fn alignNop(b: *Builder, boundary: u32) Error!void {
        if (!std.math.isPowerOfTwo(boundary) or boundary < 4) return error.InvalidOperand;
        _ = try b.addNode(.{ .@"align" = .{ .boundary = boundary } });
    }
    pub fn alignTrap(b: *Builder, boundary: u32) Error!void {
        if (!std.math.isPowerOfTwo(boundary) or boundary < 4) return error.InvalidOperand;
        _ = try b.addNode(.{ .@"align" = .{ .boundary = boundary, .nop = false } });
    }
    pub fn comment(b: *Builder, text: []const u8) Error!void {
        const off: u32 = @intCast(b.bytes.items.len);
        try b.bytes.appendSlice(b.gpa, text);
        _ = try b.addNode(.{ .comment = .{ .off = off, .len = @intCast(text.len) } });
    }
    /// Records any instruction: `b.add(.{ .add = .{ .x0, .x1, .x2 } })`.
    pub fn add(b: *Builder, i: Instr) Error!void {
        if (i.target()) |l| if (l.id >= b.label_node.items.len) return error.InvalidLabel;
        _ = try b.addNode(.{ .inst = i });
    }
    /// Typed emitter by name with the Assembler's argument order:
    /// `b.emit(.add, .{ .x0, .x1, .x2 })`, `b.emit(.bcond, .{ .ne, loop })`.
    pub fn emit(b: *Builder, comptime tag: std.meta.Tag(Instr), args: @FieldType(Instr, @tagName(tag))) Error!void {
        return b.add(@unionInit(Instr, @tagName(tag), args));
    }

    pub fn runPasses(b: *Builder, comptime passes: anytype) !void {
        inline for (passes) |p| try p(b);
    }

    // ---- serialization ----------------------------------------------------------------
    /// Replays all nodes through `a` (an `a64.AssemblerT(...)` instance). Call
    /// `a.buf.finalize()` afterwards.
    pub fn serialize(b: *const Builder, a: anytype) !void {
        const labels = try b.gpa.alloc(core.Label, b.label_node.items.len);
        defer b.gpa.free(labels);
        for (labels) |*l| l.* = try a.newLabel();
        var it = b.first;
        while (it != Node.none) : (it = b.nodes.items[it].next) {
            switch (b.nodes.items[it].data) {
                .label => |l| try a.bind(labels[l.id]),
                .@"align" => |al| if (al.nop) try a.alignNop(al.boundary) else try a.alignTrap(al.boundary),
                .comment => {},
                .inst => |i| try emitInst(a, i, labels),
            }
        }
    }

    fn emitInst(a: anytype, i: Instr, labels: []const core.Label) !void {
        @setEvalBranchQuota(100_000);
        switch (i) {
            // `bcond` takes the condition at comptime.
            .bcond => |o| switch (o[0]) {
                inline else => |c| try a.bcond(c, labels[o[1].id]),
            },
            .ret => try a.ret(),
            .b => |o| try a.b(labels[o[0].id]),
            .bl => |o| try a.bl(labels[o[0].id]),
            .cbz => |o| try a.cbz(o[0], labels[o[1].id]),
            .cbnz => |o| try a.cbnz(o[0], labels[o[1].id]),
            inline else => |o, tag| try @call(.auto, @field(@TypeOf(a), @tagName(tag)), .{a} ++ o),
        }
    }

    // ---- formatter --------------------------------------------------------------------
    /// Appends one line per node (`L0:`, `  add x0, x1, x2`, `  // text`) to `out`.
    pub fn format(b: *const Builder, out: *std.ArrayList(u8)) !void {
        var it = b.first;
        while (it != Node.none) : (it = b.nodes.items[it].next) {
            switch (b.nodes.items[it].data) {
                .label => |l| try out.print(b.gpa, "L{d}:\n", .{l.id}),
                .@"align" => |al| try out.print(b.gpa, "  .align {d}\n", .{al.boundary}),
                .comment => |c| try out.print(b.gpa, "  // {s}\n", .{b.bytes.items[c.off..][0..c.len]}),
                .inst => |i| switch (i) {
                    .ret => try out.appendSlice(b.gpa, "  ret\n"),
                    inline else => |o, tag| {
                        try out.print(b.gpa, "  {s}", .{@tagName(tag)});
                        inline for (o, 0..) |v, k| {
                            try out.appendSlice(b.gpa, if (k == 0) " " else ", ");
                            const T = @TypeOf(v);
                            if (T == Label) {
                                try out.print(b.gpa, "L{d}", .{v.id});
                            } else if (@typeInfo(T) == .@"enum") {
                                try out.appendSlice(b.gpa, @tagName(v));
                            } else try out.print(b.gpa, "#{d}", .{v});
                        }
                        try out.append(b.gpa, '\n');
                    },
                },
            }
        }
    }
};

// ---- passes ---------------------------------------------------------------------------

/// Removes instructions after `b`/`ret`/`udf` up to the next label.
pub fn removeDeadCode(b: *Builder) !void {
    var dead = false;
    var it = b.first;
    while (it != Node.none) {
        const nx = b.nodes.items[it].next;
        switch (b.nodes.items[it].data) {
            .label => dead = false,
            .inst => |i| if (dead) b.removeNode(it) else if (i.isTerminator()) {
                dead = true;
            },
            else => {},
        }
        it = nx;
    }
}

/// Removes `b L` when `L` is bound right after it (only labels/comments in between).
pub fn removeJumpsToNext(b: *Builder) !void {
    var it = b.first;
    while (it != Node.none) {
        const nx = b.nodes.items[it].next;
        const d = b.nodes.items[it].data;
        if (d == .inst and d.inst == .b) {
            const l = d.inst.b[0];
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
