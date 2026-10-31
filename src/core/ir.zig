//! Minimal compiler IR (docs/ROADMAP.md §2): i64 values in virtual registers, functions of
//! basic blocks, three-address instructions, a text parser, a verifier, a printer and a
//! reference interpreter (used for differential tests against generated code).
//!
//! Values are mutable virtual registers (`v0`, `v1`, ...) rather than SSA with block
//! parameters: the linear-scan allocator works on def/use sets, so no phi lowering is needed.
//! Parameters are `v0 .. v(params-1)`.
//!
//! Text format (one instruction per line, `#` starts a comment):
//!   func <name> <params>
//!   <label>:
//!     vD = const <int>          vD = copy vA
//!     vD = add|sub|mul|and|or|xor vA vB
//!     vD = load vA <disp>       store vA <disp> vB         (64-bit memory at vA+disp)
//!     vD = call <addr> vA ...   (up to 6 args, C calling convention)
//!     br <label>                cbr <cond> vA vB <then> <else>
//!     ret vA
//!   conds: eq ne lt le gt ge ult ule ugt uge
const std = @import("std");

pub const VReg = u32;
pub const Arith = enum { add, sub, mul, @"and", @"or", xor };
pub const CondCode = enum { eq, ne, lt, le, gt, ge, ult, ule, ugt, uge };

pub const Op = union(enum) {
    @"const": struct { dst: VReg, imm: i64 },
    copy: struct { dst: VReg, src: VReg },
    arith: struct { op: Arith, dst: VReg, a: VReg, b: VReg },
    load: struct { dst: VReg, base: VReg, disp: i32 },
    store: struct { base: VReg, disp: i32, src: VReg },
    call: struct { dst: VReg, addr: usize, args: []const VReg },
    br: u32,
    cbr: struct { cond: CondCode, a: VReg, b: VReg, then: u32, @"else": u32 },
    ret: VReg,

    pub fn isTerminator(o: Op) bool {
        return switch (o) {
            .br, .cbr, .ret => true,
            else => false,
        };
    }
    pub fn def(o: Op) ?VReg {
        return switch (o) {
            inline .@"const", .copy, .arith, .load, .call => |x| x.dst,
            else => null,
        };
    }
    /// Writes used vregs into `buf`, returns the slice.
    pub fn uses(o: Op, buf: *[8]VReg) []const VReg {
        return switch (o) {
            .@"const", .br => buf[0..0],
            .copy => |x| blk: {
                buf[0] = x.src;
                break :blk buf[0..1];
            },
            .arith => |x| blk: {
                buf[0] = x.a;
                buf[1] = x.b;
                break :blk buf[0..2];
            },
            .load => |x| blk: {
                buf[0] = x.base;
                break :blk buf[0..1];
            },
            .store => |x| blk: {
                buf[0] = x.base;
                buf[1] = x.src;
                break :blk buf[0..2];
            },
            .call => |x| blk: {
                for (x.args, 0..) |a, i| buf[i] = a;
                break :blk buf[0..x.args.len];
            },
            .cbr => |x| blk: {
                buf[0] = x.a;
                buf[1] = x.b;
                break :blk buf[0..2];
            },
            .ret => |v| blk: {
                buf[0] = v;
                break :blk buf[0..1];
            },
        };
    }
};

pub const Block = struct {
    name: []const u8,
    ops: std.ArrayList(Op) = .empty,
};

pub const Func = struct {
    arena: std.heap.ArenaAllocator,
    name: []const u8 = "",
    params: u32 = 0,
    num_vregs: u32 = 0,
    blocks: std.ArrayList(Block) = .empty,

    pub fn init(gpa: std.mem.Allocator) Func {
        return .{ .arena = std.heap.ArenaAllocator.init(gpa) };
    }
    pub fn deinit(f: *Func) void {
        f.arena.deinit();
    }
    pub fn alloc(f: *Func) std.mem.Allocator {
        return f.arena.allocator();
    }

    // ---- IrBuilder-style API ----------------------------------------------------------
    pub fn newVReg(f: *Func) VReg {
        f.num_vregs += 1;
        return f.num_vregs - 1;
    }
    pub fn addBlock(f: *Func, name: []const u8) !u32 {
        try f.blocks.append(f.alloc(), .{ .name = try f.alloc().dupe(u8, name) });
        return @intCast(f.blocks.items.len - 1);
    }
    pub fn emit(f: *Func, block: u32, op: Op) !void {
        try f.blocks.items[block].ops.append(f.alloc(), op);
        if (op.def()) |d| f.num_vregs = @max(f.num_vregs, d + 1);
        var buf: [8]VReg = undefined;
        for (op.uses(&buf)) |u| f.num_vregs = @max(f.num_vregs, u + 1);
    }

    pub const VerifyError = error{ EmptyFunction, MissingTerminator, TerminatorNotLast, BadBlock, BadVReg, TooManyArgs };
    /// Structural checks: every block ends with exactly one terminator, branch targets exist,
    /// vregs are in range, calls have at most 6 arguments.
    pub fn verify(f: *const Func) VerifyError!void {
        if (f.blocks.items.len == 0) return error.EmptyFunction;
        const nb = f.blocks.items.len;
        for (f.blocks.items) |b| {
            if (b.ops.items.len == 0 or !b.ops.items[b.ops.items.len - 1].isTerminator()) return error.MissingTerminator;
            for (b.ops.items, 0..) |o, i| {
                if (o.isTerminator() and i + 1 != b.ops.items.len) return error.TerminatorNotLast;
                switch (o) {
                    .br => |tg| if (tg >= nb) return error.BadBlock,
                    .cbr => |c| if (c.then >= nb or c.@"else" >= nb) return error.BadBlock,
                    .call => |c| if (c.args.len > 6) return error.TooManyArgs,
                    else => {},
                }
                var buf: [8]VReg = undefined;
                for (o.uses(&buf)) |u| if (u >= f.num_vregs) return error.BadVReg;
                if (o.def()) |d| if (d >= f.num_vregs) return error.BadVReg;
            }
        }
        if (f.params > f.num_vregs and f.params > 0) return error.BadVReg;
    }

    /// Prints the function in the text format accepted by `parse`.
    pub fn print(f: *const Func, gpa: std.mem.Allocator, out: *std.ArrayList(u8)) !void {
        try out.print(gpa, "func {s} {d}\n", .{ f.name, f.params });
        for (f.blocks.items) |b| {
            try out.print(gpa, "{s}:\n", .{b.name});
            for (b.ops.items) |o| {
                try out.appendSlice(gpa, "  ");
                switch (o) {
                    .@"const" => |x| try out.print(gpa, "v{d} = const {d}", .{ x.dst, x.imm }),
                    .copy => |x| try out.print(gpa, "v{d} = copy v{d}", .{ x.dst, x.src }),
                    .arith => |x| try out.print(gpa, "v{d} = {s} v{d} v{d}", .{ x.dst, @tagName(x.op), x.a, x.b }),
                    .load => |x| try out.print(gpa, "v{d} = load v{d} {d}", .{ x.dst, x.base, x.disp }),
                    .store => |x| try out.print(gpa, "store v{d} {d} v{d}", .{ x.base, x.disp, x.src }),
                    .call => |x| {
                        try out.print(gpa, "v{d} = call 0x{x}", .{ x.dst, x.addr });
                        for (x.args) |a| try out.print(gpa, " v{d}", .{a});
                    },
                    .br => |tg| try out.print(gpa, "br {s}", .{f.blocks.items[tg].name}),
                    .cbr => |x| try out.print(gpa, "cbr {s} v{d} v{d} {s} {s}", .{ @tagName(x.cond), x.a, x.b, f.blocks.items[x.then].name, f.blocks.items[x.@"else"].name }),
                    .ret => |v| try out.print(gpa, "ret v{d}", .{v}),
                }
                try out.append(gpa, '\n');
            }
        }
    }
};

pub const ParseError = error{ Syntax, UnknownLabel, DuplicateLabel, OutOfMemory, InvalidCharacter, Overflow };

fn parseVReg(tok: ?[]const u8) ParseError!VReg {
    const s = tok orelse return error.Syntax;
    if (s.len < 2 or s[0] != 'v') return error.Syntax;
    return std.fmt.parseInt(VReg, s[1..], 10);
}
fn parseInt(comptime T: type, tok: ?[]const u8) ParseError!T {
    return std.fmt.parseInt(T, tok orelse return error.Syntax, 0);
}

/// Parses one function from the text format (two passes: labels, then instructions).
pub fn parse(gpa: std.mem.Allocator, src: []const u8) ParseError!Func {
    var f = Func.init(gpa);
    errdefer f.deinit();
    // Pass 1: block names.
    var names: std.StringHashMapUnmanaged(u32) = .empty;
    defer names.deinit(gpa);
    var lines = std.mem.splitScalar(u8, src, '\n');
    while (lines.next()) |raw| {
        const line = std.mem.trim(u8, raw[0 .. std.mem.indexOfScalar(u8, raw, '#') orelse raw.len], " \t\r");
        if (line.len > 1 and line[line.len - 1] == ':') {
            const name = line[0 .. line.len - 1];
            if (names.contains(name)) return error.DuplicateLabel;
            try names.put(gpa, name, try f.addBlock(name));
        }
    }
    var cur: ?u32 = null;
    lines = std.mem.splitScalar(u8, src, '\n');
    while (lines.next()) |raw| {
        const line = std.mem.trim(u8, raw[0 .. std.mem.indexOfScalar(u8, raw, '#') orelse raw.len], " \t\r");
        if (line.len == 0) continue;
        if (line[line.len - 1] == ':') {
            cur = names.get(line[0 .. line.len - 1]).?;
            continue;
        }
        var it = std.mem.tokenizeAny(u8, line, " \t,");
        const first = it.next().?;
        if (std.mem.eql(u8, first, "func")) {
            f.name = try f.alloc().dupe(u8, it.next() orelse return error.Syntax);
            f.params = try parseInt(u32, it.next());
            f.num_vregs = @max(f.num_vregs, f.params);
            continue;
        }
        const blk = cur orelse return error.Syntax;
        const label = struct {
            fn get(n: *const std.StringHashMapUnmanaged(u32), tok: ?[]const u8) ParseError!u32 {
                return n.get(tok orelse return error.Syntax) orelse error.UnknownLabel;
            }
        }.get;
        const op: Op = if (std.mem.eql(u8, first, "br"))
            .{ .br = try label(&names, it.next()) }
        else if (std.mem.eql(u8, first, "cbr")) blk2: {
            const c = std.meta.stringToEnum(CondCode, it.next() orelse return error.Syntax) orelse return error.Syntax;
            const a = try parseVReg(it.next());
            const b = try parseVReg(it.next());
            break :blk2 .{ .cbr = .{ .cond = c, .a = a, .b = b, .then = try label(&names, it.next()), .@"else" = try label(&names, it.next()) } };
        } else if (std.mem.eql(u8, first, "ret"))
            .{ .ret = try parseVReg(it.next()) }
        else if (std.mem.eql(u8, first, "store")) blk2: {
            const base = try parseVReg(it.next());
            const disp = try parseInt(i32, it.next());
            break :blk2 .{ .store = .{ .base = base, .disp = disp, .src = try parseVReg(it.next()) } };
        } else blk2: {
            const dst = try parseVReg(first);
            if (!std.mem.eql(u8, it.next() orelse "", "=")) return error.Syntax;
            const mn = it.next() orelse return error.Syntax;
            if (std.mem.eql(u8, mn, "const")) break :blk2 .{ .@"const" = .{ .dst = dst, .imm = try parseInt(i64, it.next()) } };
            if (std.mem.eql(u8, mn, "copy")) break :blk2 .{ .copy = .{ .dst = dst, .src = try parseVReg(it.next()) } };
            if (std.mem.eql(u8, mn, "load")) {
                const base = try parseVReg(it.next());
                break :blk2 .{ .load = .{ .dst = dst, .base = base, .disp = try parseInt(i32, it.next()) } };
            }
            if (std.mem.eql(u8, mn, "call")) {
                const addr = try parseInt(usize, it.next());
                var args: std.ArrayList(VReg) = .empty;
                while (it.next()) |a| try args.append(f.alloc(), try parseVReg(a));
                break :blk2 .{ .call = .{ .dst = dst, .addr = addr, .args = args.items } };
            }
            const ar = std.meta.stringToEnum(Arith, mn) orelse return error.Syntax;
            const a = try parseVReg(it.next());
            break :blk2 .{ .arith = .{ .op = ar, .dst = dst, .a = a, .b = try parseVReg(it.next()) } };
        };
        if (it.next() != null) return error.Syntax;
        try f.emit(blk, op);
    }
    return f;
}

/// Reference interpreter. `mem` backs load/store (vA is an address inside it). Calls go to
/// real function pointers. Returns error.StepLimit after `max_steps` instructions.
pub fn interpret(f: *const Func, args: []const i64, max_steps: usize) !i64 {
    var regs = try std.heap.page_allocator.alloc(i64, @max(f.num_vregs, 1));
    defer std.heap.page_allocator.free(regs);
    @memset(regs, 0);
    for (args, 0..) |a, i| regs[i] = a;
    var b: u32 = 0;
    var steps: usize = 0;
    while (true) {
        for (f.blocks.items[b].ops.items) |o| {
            steps += 1;
            if (steps > max_steps) return error.StepLimit;
            switch (o) {
                .@"const" => |x| regs[x.dst] = x.imm,
                .copy => |x| regs[x.dst] = regs[x.src],
                .arith => |x| regs[x.dst] = arith(x.op, regs[x.a], regs[x.b]),
                .load => |x| regs[x.dst] = @as(*const i64, @ptrFromInt(@as(usize, @bitCast(regs[x.base] +% x.disp)))).*,
                .store => |x| @as(*i64, @ptrFromInt(@as(usize, @bitCast(regs[x.base] +% x.disp)))).* = regs[x.src],
                .call => |x| {
                    var a: [6]i64 = @splat(0);
                    for (x.args, 0..) |v, i| a[i] = regs[v];
                    const F = *const fn (i64, i64, i64, i64, i64, i64) callconv(.c) i64;
                    regs[x.dst] = @as(F, @ptrFromInt(x.addr))(a[0], a[1], a[2], a[3], a[4], a[5]);
                },
                .br => |tg| {
                    b = tg;
                    break;
                },
                .cbr => |x| {
                    b = if (cond(x.cond, regs[x.a], regs[x.b])) x.then else x.@"else";
                    break;
                },
                .ret => |v| return regs[v],
            }
        }
    }
}

pub fn arith(op: Arith, a: i64, b: i64) i64 {
    return switch (op) {
        .add => a +% b,
        .sub => a -% b,
        .mul => a *% b,
        .@"and" => a & b,
        .@"or" => a | b,
        .xor => a ^ b,
    };
}
pub fn cond(c: CondCode, a: i64, b: i64) bool {
    const ua: u64 = @bitCast(a);
    const ub: u64 = @bitCast(b);
    return switch (c) {
        .eq => a == b,
        .ne => a != b,
        .lt => a < b,
        .le => a <= b,
        .gt => a > b,
        .ge => a >= b,
        .ult => ua < ub,
        .ule => ua <= ub,
        .ugt => ua > ub,
        .uge => ua >= ub,
    };
}

// ---- tests ------------------------------------------------------------------------------
const t = std.testing;

pub const fib_src =
    \\func fib 1
    \\entry:
    \\  v1 = const 0      # a
    \\  v2 = const 1      # b
    \\  v3 = const 0      # i
    \\  v5 = const 1
    \\  br loop
    \\loop:
    \\  cbr lt v3 v0 body done
    \\body:
    \\  v4 = add v1 v2
    \\  v1 = copy v2
    \\  v2 = copy v4
    \\  v3 = add v3 v5
    \\  br loop
    \\done:
    \\  ret v1
;

test "ir: parse, verify, print round-trip, interpret" {
    var f = try parse(t.allocator, fib_src);
    defer f.deinit();
    try f.verify();
    try t.expectEqual(@as(i64, 55), try interpret(&f, &.{10}, 10_000));
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(t.allocator);
    try f.print(t.allocator, &out);
    var g = try parse(t.allocator, out.items);
    defer g.deinit();
    var out2: std.ArrayList(u8) = .empty;
    defer out2.deinit(t.allocator);
    try g.print(t.allocator, &out2);
    try t.expectEqualStrings(out.items, out2.items);
}

test "ir: verifier and parser reject bad input" {
    try t.expectError(error.UnknownLabel, parse(t.allocator, "func f 0\na:\n  br nowhere\n"));
    try t.expectError(error.Syntax, parse(t.allocator, "func f 0\na:\n  v0 = frob v1 v2\n"));
    try t.expectError(error.DuplicateLabel, parse(t.allocator, "func f 0\na:\na:\n"));
    var f = try parse(t.allocator, "func f 1\na:\n  v1 = add v0 v0\n");
    defer f.deinit();
    try t.expectError(error.MissingTerminator, f.verify());
    var g = Func.init(t.allocator);
    defer g.deinit();
    const b = try g.addBlock("a");
    try g.emit(b, .{ .ret = 0 });
    try g.emit(b, .{ .ret = 0 });
    try t.expectError(error.TerminatorNotLast, g.verify());
}
