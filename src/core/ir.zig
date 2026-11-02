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
//!     vD = add|sub|mul|and|or|xor|shl|shr|sar|sdiv|srem vA vB
//!     vD = load vA <disp>       store vA <disp> vB         (64-bit memory at vA+disp)
//!     vD = call <addr> vA ...   (up to 16 args, C calling convention)
//!     br <label>                cbr <cond> vA vB <then> <else>
//!     ret vA
//!
//! Block parameters (SSA style, optional): a block header `name(vP vQ):` declares
//! parameters; edges pass arguments with `br name vA vB` and `cbr <cond> vA vB then:vC else:vD:vE`
//! (`:`-separated argument lists). Arguments are assigned in parallel on the edge.
//! `verifySsa` checks single definitions; `lowerBlockParams` rewrites edges into copies
//! (splitting `cbr` edges that carry arguments) so the compilers see plain mutable vregs.
//!   conds: eq ne lt le gt ge ult ule ugt uge
const std = @import("std");

pub const VReg = u32;
/// Maximum call arguments and function parameters (extra ones are passed on the stack).
pub const max_args = 16;
/// `shl/shr/sar` mask the count to 6 bits; `sdiv/srem` truncate toward zero and are
/// undefined (trap in generated code) for a zero divisor or `minInt(i64) / -1`.
pub const Arith = enum { add, sub, mul, @"and", @"or", xor, shl, shr, sar, sdiv, srem };
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
    /// `br` with block arguments.
    bra: struct { target: u32, args: []const VReg },
    /// `cbr` with block arguments on each edge.
    cbra: struct { cond: CondCode, a: VReg, b: VReg, then: u32, @"else": u32, then_args: []const VReg, else_args: []const VReg },
    ret: VReg,

    pub fn isTerminator(o: Op) bool {
        return switch (o) {
            .br, .cbr, .bra, .cbra, .ret => true,
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
    pub fn uses(o: Op, buf: *[max_args]VReg) []const VReg {
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
            .bra => |x| blk: {
                for (x.args, 0..) |a, i| buf[i] = a;
                break :blk buf[0..x.args.len];
            },
            .cbra => |x| blk: {
                buf[0] = x.a;
                buf[1] = x.b;
                for (x.then_args, 2..) |a, i| buf[i] = a;
                for (x.else_args, 2 + x.then_args.len..) |a, i| buf[i] = a;
                break :blk buf[0 .. 2 + x.then_args.len + x.else_args.len];
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
    /// Block parameters (assigned by incoming `bra`/`cbra` edges).
    params: []const VReg = &.{},
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
        var buf: [max_args]VReg = undefined;
        for (op.uses(&buf)) |u| f.num_vregs = @max(f.num_vregs, u + 1);
    }

    pub const VerifyError = error{ EmptyFunction, MissingTerminator, TerminatorNotLast, BadBlock, BadVReg, TooManyArgs, ArgCountMismatch, MultipleDefs };
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
                    .br => |tg| if (tg >= nb or f.blocks.items[tg].params.len != 0) return if (tg >= nb) error.BadBlock else error.ArgCountMismatch,
                    .cbr => |c| if (c.then >= nb or c.@"else" >= nb) return error.BadBlock else if (f.blocks.items[c.then].params.len != 0 or f.blocks.items[c.@"else"].params.len != 0) return error.ArgCountMismatch,
                    .call => |c| if (c.args.len > max_args) return error.TooManyArgs,
                    .bra => |c| {
                        if (c.target >= nb) return error.BadBlock;
                        if (c.args.len > max_args) return error.TooManyArgs;
                        if (c.args.len != f.blocks.items[c.target].params.len) return error.ArgCountMismatch;
                    },
                    .cbra => |c| {
                        if (c.then >= nb or c.@"else" >= nb) return error.BadBlock;
                        if (2 + c.then_args.len + c.else_args.len > max_args) return error.TooManyArgs;
                        if (c.then_args.len != f.blocks.items[c.then].params.len or
                            c.else_args.len != f.blocks.items[c.@"else"].params.len) return error.ArgCountMismatch;
                    },

                    else => {},
                }
                var buf: [max_args]VReg = undefined;
                for (o.uses(&buf)) |u| if (u >= f.num_vregs) return error.BadVReg;
                if (o.def()) |d| if (d >= f.num_vregs) return error.BadVReg;
            }
        }
        for (f.blocks.items) |b| for (b.params) |p| if (p >= f.num_vregs) return error.BadVReg;
        if (f.params > f.num_vregs and f.params > 0) return error.BadVReg;
    }

    /// `verify` plus SSA form: every vreg is defined once (function parameters, block
    /// parameters and instruction results count as definitions).
    pub fn verifySsa(f: *const Func, gpa: std.mem.Allocator) (VerifyError || error{OutOfMemory})!void {
        try f.verify();
        const seen = try gpa.alloc(bool, f.num_vregs);
        defer gpa.free(seen);
        @memset(seen, false);
        for (0..f.params) |i| seen[i] = true;
        for (f.blocks.items) |b| {
            for (b.params) |p| {
                if (seen[p]) return error.MultipleDefs;
                seen[p] = true;
            }
            for (b.ops.items) |o| if (o.def()) |d| {
                if (seen[d]) return error.MultipleDefs;
                seen[d] = true;
            };
        }
    }

    pub fn hasBlockParams(f: *const Func) bool {
        for (f.blocks.items) |b| if (b.params.len != 0) return true;
        return false;
    }

    /// Returns an equivalent function without block parameters: each edge becomes
    /// `copy tmp_i = arg_i; copy param_i = tmp_i` (parallel assignment), and `cbra` edges
    /// that carry arguments go through a new edge block. Caller owns the result.
    pub fn lowerBlockParams(f: *const Func, gpa: std.mem.Allocator) !Func {
        var g = Func.init(gpa);
        errdefer g.deinit();
        g.name = try g.alloc().dupe(u8, f.name);
        g.params = f.params;
        g.num_vregs = f.num_vregs;
        for (f.blocks.items) |b| _ = try g.addBlock(b.name);
        const Edge = struct {
            fn copies(h: *Func, blk: u32, params: []const VReg, args: []const VReg) !void {
                const base = h.num_vregs;
                h.num_vregs += @intCast(args.len);
                for (args, 0..) |a, i| try h.emit(blk, .{ .copy = .{ .dst = base + @as(VReg, @intCast(i)), .src = a } });
                for (params, 0..) |p, i| try h.emit(blk, .{ .copy = .{ .dst = p, .src = base + @as(VReg, @intCast(i)) } });
            }
            fn split(h: *Func, src: *const Func, target: u32, args: []const VReg) !u32 {
                if (args.len == 0) return target;
                const e = try h.addBlock("edge");
                try copies(h, e, src.blocks.items[target].params, args);
                try h.emit(e, .{ .br = target });
                return e;
            }
        };
        for (f.blocks.items, 0..) |b, bi| {
            const blk: u32 = @intCast(bi);
            for (b.ops.items) |o| switch (o) {
                .bra => |x| {
                    try Edge.copies(&g, blk, f.blocks.items[x.target].params, x.args);
                    try g.emit(blk, .{ .br = x.target });
                },
                .cbra => |x| {
                    const th = try Edge.split(&g, f, x.then, x.then_args);
                    const el = try Edge.split(&g, f, x.@"else", x.else_args);
                    try g.emit(blk, .{ .cbr = .{ .cond = x.cond, .a = x.a, .b = x.b, .then = th, .@"else" = el } });
                },
                else => try g.emit(blk, o),
            };
        }
        return g;
    }

    /// Prints the function in the text format accepted by `parse`.
    pub fn print(f: *const Func, gpa: std.mem.Allocator, out: *std.ArrayList(u8)) !void {
        try out.print(gpa, "func {s} {d}\n", .{ f.name, f.params });
        for (f.blocks.items) |b| {
            try out.appendSlice(gpa, b.name);
            if (b.params.len != 0) {
                try out.append(gpa, '(');
                for (b.params, 0..) |p, i| try out.print(gpa, "{s}v{d}", .{ if (i == 0) "" else " ", p });
                try out.append(gpa, ')');
            }
            try out.appendSlice(gpa, ":\n");
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
                    .bra => |x| {
                        try out.print(gpa, "br {s}", .{f.blocks.items[x.target].name});
                        for (x.args) |a| try out.print(gpa, " v{d}", .{a});
                    },
                    .cbra => |x| {
                        try out.print(gpa, "cbr {s} v{d} v{d} {s}", .{ @tagName(x.cond), x.a, x.b, f.blocks.items[x.then].name });
                        for (x.then_args) |a| try out.print(gpa, ":v{d}", .{a});
                        try out.print(gpa, " {s}", .{f.blocks.items[x.@"else"].name});
                        for (x.else_args) |a| try out.print(gpa, ":v{d}", .{a});
                    },
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

fn headerName(line: []const u8) []const u8 {
    const end = std.mem.indexOfScalar(u8, line, '(') orelse line.len - 1;
    return std.mem.trim(u8, line[0..end], " \t");
}

/// `name` or `name:vA:vB` (edge with block arguments).
fn edgeTarget(names: *const std.StringHashMapUnmanaged(u32), a: std.mem.Allocator, tok: ?[]const u8, args: *std.ArrayList(VReg)) ParseError!u32 {
    var parts = std.mem.splitScalar(u8, tok orelse return error.Syntax, ':');
    const tg = names.get(parts.next().?) orelse return error.UnknownLabel;
    while (parts.next()) |p| try args.append(a, try parseVReg(p));
    return tg;
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
            const name = headerName(line);
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
            cur = names.get(headerName(line)).?;
            if (std.mem.indexOfScalar(u8, line, '(')) |lp| {
                const rp = std.mem.lastIndexOfScalar(u8, line, ')') orelse return error.Syntax;
                var ps: std.ArrayList(VReg) = .empty;
                var pit = std.mem.tokenizeAny(u8, line[lp + 1 .. rp], " \t,");
                while (pit.next()) |tok| {
                    const v = try parseVReg(tok);
                    try ps.append(f.alloc(), v);
                    f.num_vregs = @max(f.num_vregs, v + 1);
                }
                f.blocks.items[cur.?].params = ps.items;
            }
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
        const op: Op = if (std.mem.eql(u8, first, "br")) blk2: {
            const tg = try label(&names, it.next());
            var args: std.ArrayList(VReg) = .empty;
            while (it.next()) |a| try args.append(f.alloc(), try parseVReg(a));
            break :blk2 if (args.items.len == 0) .{ .br = tg } else .{ .bra = .{ .target = tg, .args = args.items } };
        } else if (std.mem.eql(u8, first, "cbr")) blk2: {
            const c = std.meta.stringToEnum(CondCode, it.next() orelse return error.Syntax) orelse return error.Syntax;
            const a = try parseVReg(it.next());
            const b = try parseVReg(it.next());
            var ta: std.ArrayList(VReg) = .empty;
            var ea: std.ArrayList(VReg) = .empty;
            const th = try edgeTarget(&names, f.alloc(), it.next(), &ta);
            const el = try edgeTarget(&names, f.alloc(), it.next(), &ea);
            if (ta.items.len == 0 and ea.items.len == 0) break :blk2 .{ .cbr = .{ .cond = c, .a = a, .b = b, .then = th, .@"else" = el } };
            break :blk2 .{ .cbra = .{ .cond = c, .a = a, .b = b, .then = th, .@"else" = el, .then_args = ta.items, .else_args = ea.items } };
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
                .load => |x| regs[x.dst] = @as(*const i64, @ptrFromInt(@as(usize, @truncate(@as(u64, @bitCast(regs[x.base] +% x.disp)))))).*,
                .store => |x| @as(*i64, @ptrFromInt(@as(usize, @truncate(@as(u64, @bitCast(regs[x.base] +% x.disp)))))).* = regs[x.src],
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
                .bra => |x| {
                    assignParams(regs, f.blocks.items[x.target].params, x.args);
                    b = x.target;
                    break;
                },
                .cbra => |x| {
                    const taken = cond(x.cond, regs[x.a], regs[x.b]);
                    b = if (taken) x.then else x.@"else";
                    assignParams(regs, f.blocks.items[b].params, if (taken) x.then_args else x.else_args);
                    break;
                },
                .ret => |v| return regs[v],
            }
        }
    }
}

fn assignParams(regs: []i64, params: []const VReg, args: []const VReg) void {
    var tmp: [max_args]i64 = undefined;
    for (args, 0..) |a, i| tmp[i] = regs[a];
    for (params, 0..) |p, i| regs[p] = tmp[i];
}

pub fn arith(op: Arith, a: i64, b: i64) i64 {
    return switch (op) {
        .add => a +% b,
        .sub => a -% b,
        .mul => a *% b,
        .@"and" => a & b,
        .@"or" => a | b,
        .xor => a ^ b,
        .shl => @bitCast(@as(u64, @bitCast(a)) << @intCast(b & 63)),
        .shr => @bitCast(@as(u64, @bitCast(a)) >> @intCast(b & 63)),
        .sar => a >> @intCast(b & 63),
        // Undefined cases (b == 0, minInt / -1) return 0 here; callers must avoid them.
        .sdiv => if (b == 0 or (a == std.math.minInt(i64) and b == -1)) 0 else @divTrunc(a, b),
        .srem => if (b == 0 or (a == std.math.minInt(i64) and b == -1)) 0 else @rem(a, b),
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

pub const fib_ssa_src =
    \\func fib 1
    \\entry:
    \\  v1 = const 0
    \\  v2 = const 1
    \\  v3 = const 0
    \\  v4 = const 1
    \\  br loop v1 v2 v3
    \\loop(v5 v6 v7):
    \\  cbr lt v7 v0 body done:v5
    \\body:
    \\  v8 = add v5 v6
    \\  v9 = add v7 v4
    \\  br loop v6 v8 v9
    \\done(v10):
    \\  ret v10
;

test "ir: block parameters - parse, SSA verify, print round-trip, interpret, lower" {
    var f = try parse(t.allocator, fib_ssa_src);
    defer f.deinit();
    try f.verifySsa(t.allocator);
    try t.expect(f.hasBlockParams());
    for ([_]i64{ 0, 1, 2, 10, 50 }) |n| {
        var p = try parse(t.allocator, fib_src);
        defer p.deinit();
        const want = try interpret(&p, &.{n}, 1 << 20);
        try t.expectEqual(want, try interpret(&f, &.{n}, 1 << 20));
        var g = try f.lowerBlockParams(t.allocator);
        defer g.deinit();
        try g.verify();
        try t.expect(!g.hasBlockParams());
        try t.expectEqual(want, try interpret(&g, &.{n}, 1 << 20));
    }
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(t.allocator);
    try f.print(t.allocator, &out);
    try t.expect(std.mem.indexOf(u8, out.items, "loop(v5 v6 v7):") != null);
    var g = try parse(t.allocator, out.items);
    defer g.deinit();
    var out2: std.ArrayList(u8) = .empty;
    defer out2.deinit(t.allocator);
    try g.print(t.allocator, &out2);
    try t.expectEqualStrings(out.items, out2.items);
}

test "ir: block parameters - parallel swap and verifier errors" {
    // b(v1 v2) receives (v2 v1): a swap that sequential copies would get wrong.
    const src =
        \\func f 2
        \\e:
        \\  br s v1 v0
        \\s(v2 v3):
        \\  br b v3 v2
        \\b(v4 v5):
        \\  v6 = const 10
        \\  v7 = mul v4 v6
        \\  v8 = add v7 v5
        \\  ret v8
    ;
    var f = try parse(t.allocator, src);
    defer f.deinit();
    try f.verifySsa(t.allocator);
    try t.expectEqual(@as(i64, 12), try interpret(&f, &.{ 1, 2 }, 100));
    var g = try f.lowerBlockParams(t.allocator);
    defer g.deinit();
    try t.expectEqual(@as(i64, 12), try interpret(&g, &.{ 1, 2 }, 100));

    var bad = try parse(t.allocator, "func f 1\na:\n  br b v0\nb(v1 v2):\n  ret v1\n");
    defer bad.deinit();
    try t.expectError(error.ArgCountMismatch, bad.verify());
    var nossa = try parse(t.allocator, "func f 1\na:\n  v1 = const 1\n  v1 = const 2\n  ret v1\n");
    defer nossa.deinit();
    try nossa.verify();
    try t.expectError(error.MultipleDefs, nossa.verifySsa(t.allocator));
}
