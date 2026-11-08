//! Append-only sink for Compiler lowering. Encode ordinary instructions immediately;
//! retain only labels and branches for dead-code/jump removal and branch relaxation.
//! The general editable Builder remains the sink for the public `lower` API.
const std = @import("std");
const bld = @import("builder.zig");
const x = @import("assembler.zig");
const core = @import("../core/code_buffer.zig");
const A = x.AssemblerT(.{ .validate = true });

pub const Buffer = struct {
    const Event = struct {
        compact: u32,
        off: u32 = 0,
        label: u32,
        kind: enum { label, jmp, jcc },
        cond: x.Cond = .e,
        long: bool = false,
        fn size(e: Event) u32 {
            return if (e.kind == .label) 0 else if (!e.long) 2 else if (e.kind == .jmp) 5 else 6;
        }
    };
    const Label = struct { code: core.Label, off: u32 = 0, bound: bool = false };
    gpa: std.mem.Allocator,
    cb: *core.CodeBuffer,
    events: std.ArrayList(Event) = .empty,
    labels: std.ArrayList(Label) = .empty,
    nodes: usize = 0,
    dead: bool = false,
    fix0: usize,

    pub fn init(gpa: std.mem.Allocator, cb: *core.CodeBuffer) Buffer {
        return .{ .gpa = gpa, .cb = cb, .fix0 = cb.fixups.items.len };
    }
    pub fn deinit(b: *Buffer) void {
        b.events.deinit(b.gpa);
        b.labels.deinit(b.gpa);
    }
    pub fn count(b: *const Buffer) usize {
        return b.nodes;
    }
    pub fn comment(b: *Buffer, _: []const u8) !void {
        b.nodes += 1;
    }
    pub fn newLabel(b: *Buffer) !bld.Label {
        const code = try b.cb.newLabel();
        try b.labels.append(b.gpa, .{ .code = code });
        return .{ .id = @intCast(b.labels.items.len - 1) };
    }
    pub fn bind(b: *Buffer, l: bld.Label) !void {
        if (l.id >= b.labels.items.len) return error.InvalidLabel;
        if (b.labels.items[l.id].bound) return error.LabelAlreadyBound;
        // A jump to a following label disappears, including intervening labels.
        var i = b.events.items.len;
        while (i > 0) {
            i -= 1;
            const e = b.events.items[i];
            if (e.compact != b.cb.offset()) break;
            if (e.kind == .label) continue;
            if (e.kind == .jmp and e.label == l.id) {
                _ = b.events.orderedRemove(i);
                b.nodes -= 1;
            }
            break;
        }
        try b.events.append(b.gpa, .{ .compact = b.cb.offset(), .kind = .label, .label = l.id });
        b.labels.items[l.id].bound = true;
        b.dead = false;
        b.nodes += 1;
    }
    pub inline fn emit(b: *Buffer, comptime tag: std.meta.Tag(bld.Instr), args: @FieldType(bld.Instr, @tagName(tag))) !void {
        if (b.dead) return;
        switch (tag) {
            .jmp, .jcc => {
                const l = if (tag == .jmp) args[0] else args[1];
                if (l.id >= b.labels.items.len) return error.InvalidLabel;
                try b.events.append(b.gpa, .{
                    .compact = b.cb.offset(),
                    .kind = if (tag == .jmp) .jmp else .jcc,
                    .label = l.id,
                    .cond = if (tag == .jmp) .e else args[0],
                });
                if (tag == .jmp) b.dead = true;
            },
            .call, .jmp8, .jmp32, .jcc8, .jcc32 => @compileError("Compiler sink expects unsized branches and absolute calls"),
            else => {
                const special = comptime blk: {
                    for (bld.Instr.special) |name| if (std.mem.eql(u8, name, @tagName(tag))) break :blk true;
                    break :blk false;
                };
                const a = A.init(b.cb);
                if (special) {
                    try bld.Instr.emitSpecialTyped(a, tag, args, &.{});
                } else if (@TypeOf(args) == void) {
                    try @call(.always_inline, @field(A, @tagName(tag)), .{a});
                } else {
                    try @call(.always_inline, @field(A, @tagName(tag)), .{a} ++ args);
                }
                if (tag == .ret or tag == .jmpR) b.dead = true;
            },
        }
        b.nodes += 1;
    }
    pub fn add(b: *Buffer, in: bld.Instr) !void {
        // The compiler uses `add` only for operand-free instructions.
        switch (in) {
            .ret => try b.emit(.ret, {}),
            .cqo => try b.emit(.cqo, {}),
            else => unreachable,
        }
    }
    pub fn finish(b: *Buffer) !void {
        const compact_end = b.cb.offset();
        const fix_end = b.cb.fixups.items.len;
        var end = compact_end;
        var changed = true;
        while (changed) {
            changed = false;
            var extra: u32 = 0;
            for (b.events.items) |*e| {
                e.off = e.compact + extra;
                if (e.kind == .label) b.labels.items[e.label].off = e.off;
                extra += e.size();
            }
            end = compact_end + extra;
            for (b.events.items) |*e| {
                if (e.kind == .label) continue;
                const l = b.labels.items[e.label];
                if (!l.bound) return error.UnboundLabel;
                if (!e.long and std.math.cast(i8, @as(i64, l.off) - @as(i64, e.off + 2)) == null) {
                    e.long = true;
                    changed = true;
                }
            }
        }
        try b.cb.ensure(end - compact_end);
        var compact = compact_end;
        var final = end;
        var fix = fix_end;
        var i = b.events.items.len;
        while (i > 0) {
            i -= 1;
            const e = b.events.items[i];
            const len = compact - e.compact;
            const dest = final - len;
            std.mem.copyBackwards(u8, b.cb.data[dest..][0..len], b.cb.data[e.compact..][0..len]);
            const shift = dest - e.compact;
            while (fix > b.fix0 and b.cb.fixups.items[fix - 1].at >= e.compact) : (fix -= 1) {
                b.cb.fixups.items[fix - 1].at += shift;
                b.cb.fixups.items[fix - 1].base += shift;
            }
            compact = e.compact;
            b.cb.len = e.off;
            const label = b.labels.items[e.label].code;
            if (e.kind == .label) {
                try b.cb.bind(label);
            } else {
                if (e.kind == .jmp) {
                    try b.cb.emitBytes(&.{if (e.long) 0xE9 else 0xEB});
                } else if (e.long) {
                    try b.cb.emitBytes(&.{ 0x0F, 0x80 + @as(u8, @backingInt(e.cond)) });
                } else {
                    try b.cb.emitBytes(&.{0x70 + @as(u8, @backingInt(e.cond))});
                }
                const at = b.cb.offset();
                const width: u32 = if (e.long) 4 else 1;
                const zeros: [4]u8 = @splat(0);
                try b.cb.emitBytes(zeros[0..width]);
                try b.cb.addFixup(.{ .at = at, .base = at + width, .label = label.id, .kind = if (e.long) .rel32 else .rel8 });
            }
            final = e.off;
        }
        b.cb.len = end;
    }
};

fn testProgram(sink: anytype, seed: u64) !void {
    var random = std.Random.DefaultPrng.init(seed);
    const rnd = random.random();
    var labels: [12]bld.Label = undefined;
    for (&labels) |*l| l.* = try sink.newLabel();
    try sink.comment("function");
    for (labels, 0..) |label, bi| {
        try sink.bind(label);
        // Empty blocks, short/long branches, backwards edges, dead instructions,
        // and comments between an unconditional jump and its target.
        for (0..rnd.uintLessThan(usize, 90)) |_| try sink.emit(.nop, {});
        try sink.emit(.movRI, .{ .r10, @as(i64, @intCast(bi)) });
        try sink.emit(.jcc, .{ .ne, labels[rnd.uintLessThan(usize, labels.len)] });
        try sink.emit(.jmp, .{labels[if (bi + 1 < labels.len) bi + 1 else 0]});
        try sink.emit(.movRI, .{ .rax, -1 }); // dead
        try sink.comment("edge");
    }
    // This is dead as well (the last block jumps to entry).
    try sink.add(.ret);
}

fn testPrefix(cb: *core.CodeBuffer) !void {
    // A pre-existing fixup must remain valid when appending and relocating code.
    const label = try cb.newLabel();
    try cb.emitBytes(&.{ 0, 0x90, 0x90 });
    try cb.addFixup(.{ .at = 0, .base = 1, .label = label.id, .kind = .rel8 });
    try cb.bind(label);
}

test "compiler buffer: bytes and node counts match editable Builder passes" {
    const t = std.testing;
    for (0..80) |seed| {
        var expected = core.CodeBuffer.init(t.allocator);
        defer expected.deinit();
        var actual = core.CodeBuffer.init(t.allocator);
        defer actual.deinit();
        try testPrefix(&expected);
        try testPrefix(&actual);
        var builder = bld.Builder.init(t.allocator);
        defer builder.deinit();
        var stream = Buffer.init(t.allocator, &actual);
        defer stream.deinit();
        try testProgram(&builder, seed);
        try testProgram(&stream, seed);
        try builder.runPasses(.{ bld.removeDeadCode, bld.removeJumpsToNext });
        try bld.serializeShortened(&builder, &expected);
        try stream.finish();
        try expected.finalize();
        try actual.finalize();
        try t.expectEqualSlices(u8, expected.code(), actual.code());
        try t.expectEqual(builder.count(), stream.count());
    }
}

test "compiler buffer: allocation failures release scratch storage" {
    const Case = struct {
        fn run(gpa: std.mem.Allocator) !void {
            var cb = core.CodeBuffer.init(gpa);
            defer cb.deinit();
            var stream = Buffer.init(gpa, &cb);
            defer stream.deinit();
            try testProgram(&stream, 42);
            try stream.finish();
            try cb.finalize();
        }
    };
    try std.testing.checkAllAllocationFailures(std.testing.allocator, Case.run, .{});
}
