//! Generic Builder: an arch-independent node list (instructions, labels, alignment, comments)
//! with cursor insertion, passes, a text formatter and serialization through an Assembler.
//!
//! `BuilderT(Instr)` takes the backend's instruction union. Requirements on `Instr`:
//! - one variant per Assembler emitter, named after it, whose payload is the emitter's
//!   argument tuple (`Label` fields are builder labels, mapped at serialize);
//! - `pub fn isTerminator(Instr) bool` (control never falls through);
//! - `pub fn jumpTarget(Instr) ?Label` (unconditional jump; for `removeJumpsToNext`).
//! Optional:
//! - `pub const special: []const []const u8` + `pub fn emitSpecial(a, Instr, []const core.Label)
//!   !void` for emitters that take comptime arguments (the listed tags go to `emitSpecial`);
//! - `pub const min_align: u32` (default 1);
//! - `pub fn formatInst(gpa, *std.ArrayList(u8), Instr) !void` (assembly-style text; the
//!   default prints `tag arg, arg`) and `pub const comment_prefix: []const u8` (default "//").
//! Serialization calls the Assembler emitter by tag name, so output is byte-identical to
//! calling the Assembler directly (validation and hooks unchanged).
const std = @import("std");
const core = @import("code_buffer.zig");

/// Builder-side label (index into the builder's label table; mapped to buffer labels at serialize).
pub const Label = struct { id: u32 };

pub const Error = core.Error || error{ LabelAlreadyBound, InvalidOperand };

pub fn BuilderT(comptime InstrT: type) type {
    return struct {
        const Self = @This();
        pub const Instr = InstrT;
        const comment_prefix: []const u8 = if (@hasDecl(Instr, "comment_prefix")) Instr.comment_prefix else "//";
        const min_align: u32 = if (@hasDecl(Instr, "min_align")) Instr.min_align else 1;

        pub const Node = struct {
            pub const none: u32 = std.math.maxInt(u32);
            data: Data,
            prev: u32 = none,
            next: u32 = none,

            pub const Data = union(enum) {
                inst: Instr,
                label: Label,
                /// Pad with NOPs (`nop = true`) or trap instructions to a power-of-two boundary.
                @"align": struct { boundary: u32, nop: bool = true },
                comment: struct { off: u32, len: u32 },
                /// Raw bytes (copied into the builder's byte arena).
                embed: struct { off: u32, len: u32 },
            };
        };

        gpa: std.mem.Allocator,
        nodes: std.ArrayList(Node) = .empty,
        bytes: std.ArrayList(u8) = .empty,
        first: u32 = Node.none,
        last: u32 = Node.none,
        /// New nodes are inserted after this node (`none` = at the front).
        cursor: u32 = Node.none,
        /// Label -> node that binds it (`none` until bound).
        label_node: std.ArrayList(u32) = .empty,

        pub fn init(gpa: std.mem.Allocator) Self {
            return .{ .gpa = gpa };
        }
        pub fn deinit(b: *Self) void {
            b.nodes.deinit(b.gpa);
            b.bytes.deinit(b.gpa);
            b.label_node.deinit(b.gpa);
        }

        // ---- node list ------------------------------------------------------------------
        pub fn newLabel(b: *Self) Error!Label {
            try b.label_node.append(b.gpa, Node.none);
            return .{ .id = @intCast(b.label_node.items.len - 1) };
        }
        /// Inserts `d` after the cursor and moves the cursor to it. Returns the node index.
        pub fn addNode(b: *Self, d: Node.Data) Error!u32 {
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
        pub fn removeNode(b: *Self, id: u32) void {
            const n = b.nodes.items[id];
            if (n.prev != Node.none) b.nodes.items[n.prev].next = n.next else b.first = n.next;
            if (n.next != Node.none) b.nodes.items[n.next].prev = n.prev else b.last = n.prev;
            if (b.cursor == id) b.cursor = n.prev;
            if (n.data == .label) b.label_node.items[n.data.label.id] = Node.none;
            b.nodes.items[id].prev = Node.none;
            b.nodes.items[id].next = Node.none;
        }
        pub fn setCursor(b: *Self, id: u32) void {
            b.cursor = id;
        }
        /// Number of linked nodes.
        pub fn count(b: *const Self) usize {
            var n: usize = 0;
            var it = b.first;
            while (it != Node.none) : (it = b.nodes.items[it].next) n += 1;
            return n;
        }
        pub fn bind(b: *Self, l: Label) Error!void {
            if (l.id >= b.label_node.items.len) return error.InvalidLabel;
            if (b.label_node.items[l.id] != Node.none) return error.LabelAlreadyBound;
            b.label_node.items[l.id] = try b.addNode(.{ .label = l });
        }
        pub fn alignNop(b: *Self, boundary: u32) Error!void {
            if (!std.math.isPowerOfTwo(boundary) or boundary < min_align) return error.InvalidOperand;
            _ = try b.addNode(.{ .@"align" = .{ .boundary = boundary } });
        }
        pub fn alignTrap(b: *Self, boundary: u32) Error!void {
            if (!std.math.isPowerOfTwo(boundary) or boundary < min_align) return error.InvalidOperand;
            _ = try b.addNode(.{ .@"align" = .{ .boundary = boundary, .nop = false } });
        }
        pub fn comment(b: *Self, text: []const u8) Error!void {
            const off: u32 = @intCast(b.bytes.items.len);
            try b.bytes.appendSlice(b.gpa, text);
            _ = try b.addNode(.{ .comment = .{ .off = off, .len = @intCast(text.len) } });
        }
        /// Embeds raw bytes (code or data); dead-code removal stops at embeds.
        pub fn embed(b: *Self, data: []const u8) Error!void {
            const off: u32 = @intCast(b.bytes.items.len);
            try b.bytes.appendSlice(b.gpa, data);
            _ = try b.addNode(.{ .embed = .{ .off = off, .len = @intCast(data.len) } });
        }
        pub fn slice(b: *const Self, off: u32, len: u32) []const u8 {
            return b.bytes.items[off..][0..len];
        }
        pub fn node(b: *const Self, id: u32) *Node {
            return &b.nodes.items[id];
        }
        /// Records any instruction: `b.add(.{ .add = .{ .x0, .x1, .x2 } })`.
        pub fn add(b: *Self, i: Instr) Error!void {
            switch (i) {
                inline else => |o| if (@TypeOf(o) != void) inline for (o) |v| if (@TypeOf(v) == Label) {
                    if (v.id >= b.label_node.items.len) return error.InvalidLabel;
                },
            }
            _ = try b.addNode(.{ .inst = i });
        }
        /// Typed emitter by name with the Assembler's argument order:
        /// `b.emit(.add, .{ .x0, .x1, .x2 })`.
        pub fn emit(b: *Self, comptime tag: std.meta.Tag(Instr), args: @FieldType(Instr, @tagName(tag))) Error!void {
            return b.add(@unionInit(Instr, @tagName(tag), args));
        }

        pub fn runPasses(b: *Self, comptime passes: anytype) !void {
            inline for (passes) |p| try p(b);
        }

        // ---- serialization --------------------------------------------------------------
        /// Replays all nodes through Assembler instance `a`. Call `a.buf.finalize()` afterwards.
        pub fn serialize(b: *const Self, a: anytype) !void {
            const labels = try b.gpa.alloc(core.Label, b.label_node.items.len);
            defer b.gpa.free(labels);
            for (labels) |*l| l.* = try a.newLabel();
            var it = b.first;
            while (it != Node.none) : (it = b.nodes.items[it].next) {
                switch (b.nodes.items[it].data) {
                    .label => |l| try a.bind(labels[l.id]),
                    .@"align" => |al| if (al.nop) try a.alignNop(al.boundary) else try a.alignTrap(al.boundary),
                    .comment => {},
                    .embed => |e| try a.buf.emitBytes(b.slice(e.off, e.len)),
                    .inst => |i| try emitInst(a, i, labels),
                }
            }
        }

        /// Emits one instruction through Assembler `a` (`labels` maps builder labels).
        pub fn emitInst(a: anytype, i: Instr, labels: []const core.Label) !void {
            @setEvalBranchQuota(100_000);
            switch (i) {
                inline else => |o, tag| {
                    if (comptime isSpecial(@tagName(tag))) return Instr.emitSpecial(a, i, labels);
                    const f = @field(@TypeOf(a), @tagName(tag));
                    if (@TypeOf(o) == void) return @call(.auto, f, .{a});
                    var args: MapArgs(@TypeOf(o)) = undefined;
                    inline for (o, 0..) |v, k| args[k] = if (@TypeOf(v) == Label) labels[v.id] else v;
                    try @call(.auto, f, .{a} ++ args);
                },
            }
        }
        fn isSpecial(comptime name: []const u8) bool {
            if (!@hasDecl(Instr, "special")) return false;
            for (Instr.special) |sp| if (std.mem.eql(u8, sp, name)) return true;
            return false;
        }
        fn MapArgs(comptime T: type) type {
            const fs = @typeInfo(T).@"struct".field_types;
            var out: [fs.len]type = undefined;
            for (fs, 0..) |ft, k| out[k] = if (ft == Label) core.Label else ft;
            return @Tuple(&out);
        }

        // ---- formatter ------------------------------------------------------------------
        /// Appends one line per node (`L0:`, `  add x0, x1, x2`, `  // text`) to `out`.
        pub fn format(b: *const Self, out: *std.ArrayList(u8)) !void {
            var it = b.first;
            while (it != Node.none) : (it = b.nodes.items[it].next) {
                switch (b.nodes.items[it].data) {
                    .label => |l| try out.print(b.gpa, "L{d}:\n", .{l.id}),
                    .@"align" => |al| try out.print(b.gpa, "  .align {d}\n", .{al.boundary}),
                    .comment => |c| try out.print(b.gpa, "  {s} {s}\n", .{ comment_prefix, b.slice(c.off, c.len) }),
                    .embed => |e| try out.print(b.gpa, "  .embed {d} bytes\n", .{e.len}),
                    .inst => |i| if (comptime @hasDecl(Instr, "formatInst")) {
                        try out.appendSlice(b.gpa, "  ");
                        try Instr.formatInst(b.gpa, out, i);
                        try out.append(b.gpa, '\n');
                    } else switch (i) {
                        inline else => |o, tag| {
                            try out.print(b.gpa, "  {s}", .{@tagName(tag)});
                            if (@TypeOf(o) != void) inline for (o, 0..) |v, k| {
                                try out.appendSlice(b.gpa, if (k == 0) " " else ", ");
                                const T = @TypeOf(v);
                                if (T == Label) {
                                    try out.print(b.gpa, "L{d}", .{v.id});
                                } else if (@typeInfo(T) == .@"enum") {
                                    try out.appendSlice(b.gpa, @tagName(v));
                                } else if (@typeInfo(T) == .int or @typeInfo(T) == .comptime_int) {
                                    try out.print(b.gpa, "#{d}", .{v});
                                } else try out.print(b.gpa, "{any}", .{v});
                            };
                            try out.append(b.gpa, '\n');
                        },
                    },
                }
            }
        }

        // ---- generic passes -------------------------------------------------------------
        /// Removes instructions after a terminator up to the next label.
        pub fn removeDeadCode(b: *Self) !void {
            var dead = false;
            var it = b.first;
            while (it != Node.none) {
                const nx = b.nodes.items[it].next;
                switch (b.nodes.items[it].data) {
                    .label, .embed => dead = false,
                    .inst => |i| if (dead) b.removeNode(it) else if (i.isTerminator()) {
                        dead = true;
                    },
                    else => {},
                }
                it = nx;
            }
        }

        /// Removes an unconditional jump to `L` when `L` is bound right after it (only
        /// labels/comments in between).
        pub fn removeJumpsToNext(b: *Self) !void {
            var it = b.first;
            while (it != Node.none) {
                const nx = b.nodes.items[it].next;
                const d = b.nodes.items[it].data;
                if (d == .inst) if (d.inst.jumpTarget()) |l| {
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
                };
                it = nx;
            }
        }
    };
}

// ---- tests (toy backend: one byte per instruction) --------------------------------------
const t = std.testing;
const Toy = struct {
    buf: *core.CodeBuffer,
    pub fn newLabel(a: Toy) !core.Label {
        return a.buf.newLabel();
    }
    pub fn bind(a: Toy, l: core.Label) !void {
        return a.buf.bind(l);
    }
    pub fn alignNop(a: Toy, n: u32) !void {
        while (a.buf.offset() % n != 0) try a.buf.emitBytes(&.{0x90});
    }
    pub fn alignTrap(a: Toy, n: u32) !void {
        while (a.buf.offset() % n != 0) try a.buf.emitBytes(&.{0xCC});
    }
    pub fn put(a: Toy, v: u8) !void {
        return a.buf.emitBytes(&.{v});
    }
    pub fn halt(a: Toy) !void {
        return a.buf.emitBytes(&.{0xFF});
    }
    pub fn jmp(a: Toy, l: core.Label) !void {
        try a.buf.emitBytes(&.{0xE9});
        try a.buf.emitBytes(&.{@intCast(a.buf.labelOffset(l) orelse 0xEE)});
    }
    pub fn twice(a: Toy, comptime v: u8) !void {
        return a.buf.emitBytes(&.{ v, v });
    }
};
const ToyInstr = union(enum) {
    put: struct { u8 },
    halt,
    jmp: struct { Label },
    twice: struct { u8 },
    pub const special: []const []const u8 = &.{"twice"};
    pub fn emitSpecial(a: anytype, i: ToyInstr, _: []const core.Label) !void {
        switch (i.twice[0]) {
            inline 1, 2 => |v| try a.twice(v),
            else => return error.InvalidOperand,
        }
    }
    pub fn isTerminator(i: ToyInstr) bool {
        return i == .halt or i == .jmp;
    }
    pub fn jumpTarget(i: ToyInstr) ?Label {
        return if (i == .jmp) i.jmp[0] else null;
    }
};

test "core builder: generic node list, passes, serialize, format" {
    const B = BuilderT(ToyInstr);
    var b = B.init(t.allocator);
    defer b.deinit();
    const top = try b.newLabel();
    const next = try b.newLabel();
    try b.bind(top);
    try b.emit(.put, .{1});
    try b.emit(.twice, .{2});
    try b.emit(.jmp, .{next}); // removed: next is bound right after
    try b.comment("x");
    try b.bind(next);
    try b.emit(.jmp, .{top});
    try b.emit(.put, .{9}); // dead
    try b.add(.halt); // dead
    try b.alignNop(4);
    try b.runPasses(.{ B.removeDeadCode, B.removeJumpsToNext });
    var cb = core.CodeBuffer.init(t.allocator);
    defer cb.deinit();
    try b.serialize(Toy{ .buf = &cb });
    try t.expectEqualSlices(u8, &.{ 1, 2, 2, 0xE9, 0, 0x90, 0x90, 0x90 }, cb.code());
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(t.allocator);
    try b.format(&out);
    try t.expectEqualStrings("L0:\n  put #1\n  twice #2\n  // x\nL1:\n  jmp L0\n  .align 4\n", out.items);
    try t.expectError(error.InvalidLabel, b.emit(.jmp, .{.{ .id = 5 }}));
    try t.expectError(error.LabelAlreadyBound, b.bind(top));
}
