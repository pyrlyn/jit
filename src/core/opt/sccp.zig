//! Sparse conditional constant propagation (Wegman and Zadeck) on SSA form.
//!
//! Every vreg gets a lattice value: unknown (not yet seen to execute), a constant (the
//! interpreter's `ir.Val` bits for its type) or overdefined. Block parameters are the phis:
//! the meet of the arguments on the incoming edges found executable so far. Two worklists
//! drive the analysis: CFG edges that became executable and vregs whose value went down; a
//! def->uses index (CSR over vregs, built once) finds the ops to re-evaluate. Each vreg's value
//! goes down at most twice, so the analysis is linear in ops + edge arguments.
//!
//! Folding uses the interpreter's helpers (`ir.arith`, `ir.cond`, `ir.typedArith`, `ir.fcond`,
//! `ir.convert`, `ir.getLane`, `ir.setLane`), so results are bit-identical to `ir.interpret`.
//! Never folded: `sdiv`/`srem` with a zero divisor or `minInt / -1` (undefined: the generated
//! code traps), float ops, `fcmp` and `cvt` with a NaN operand or result lane (NaN payloads
//! differ between the interpreter and hardware), `splat`/`extract` of a NaN lane, `load` and
//! `call`.
//!
//! Rewrite: the uses of every vreg proven constant (other than `const`/`fconst` results) are
//! renamed to one constant vreg per (type, bits), defined right after the constants that
//! already lead block 0 (`const` for i64, `fconst` for floats, a lane constant + `splat` for a
//! vector whose lanes are all equal; other vectors are left alone). A `cbr`/`cbra` of an
//! executable block with one executable edge becomes `br`/`bra` (the dropped edge's arguments
//! disappear, `c.taken` becomes unknown). Dead ops are left to dce and blocks that never execute
//! to simplifycfg (their terminators are untouched, so they stay valid IR).
const std = @import("std");
const ir = @import("../ir.zig");
const opt = @import("opt.zig");
const cfg = @import("cfg.zig");
const ssa = @import("ssa.zig");
const testgen = @import("testgen.zig");

const none = cfg.none;

const State = enum(u8) { unknown, constant, over };
const Lat = struct { st: State, val: ir.Val = 0 };
const unknown: Lat = .{ .st = .unknown };
const over: Lat = .{ .st = .over };

fn constant(v: ir.Val) Lat {
    return .{ .st = .constant, .val = v };
}

/// A read of a vreg by op `op` of block `b`: an operand (`arg == none`) or edge argument `pos`
/// of successor `index` (`arg == index << 31 | pos`).
const Use = struct { b: u32, op: u32, arg: u32 };

const Solver = struct {
    gpa: std.mem.Allocator,
    f: *ir.Func,
    lat: []Lat,
    /// Uses of vreg `v`: `uses[use_start[v]..use_start[v + 1]]`.
    use_start: []u32,
    uses: []Use,
    block_exec: []bool,
    /// Edge `2 * b + index` was found executable.
    edge_exec: []bool,
    cfg_work: std.ArrayList(u32) = .empty,
    ssa_work: std.ArrayList(ir.VReg) = .empty,

    fn init(gpa: std.mem.Allocator, f: *ir.Func) opt.Error!Solver {
        const nv = f.num_vregs;
        const nb = f.blocks.items.len;
        var s: Solver = .{
            .gpa = gpa,
            .f = f,
            .lat = &.{},
            .use_start = &.{},
            .uses = &.{},
            .block_exec = &.{},
            .edge_exec = &.{},
        };
        errdefer s.deinit();
        s.lat = try gpa.alloc(Lat, nv);
        @memset(s.lat, unknown);
        for (s.lat[0..f.params]) |*l| l.* = over;
        s.block_exec = try gpa.alloc(bool, nb);
        @memset(s.block_exec, false);
        s.edge_exec = try gpa.alloc(bool, 2 * nb);
        @memset(s.edge_exec, false);

        // def -> uses index: count per vreg, prefix sums, fill.
        s.use_start = try gpa.alloc(u32, nv + 1);
        @memset(s.use_start, 0);
        for (f.blocks.items) |b| for (b.ops.items) |o| forEachUse(o, s.use_start, countUse);
        var sum: u32 = 0;
        for (s.use_start) |*n| {
            const k = n.*;
            n.* = sum;
            sum += k;
        }
        s.uses = try gpa.alloc(Use, sum);
        const fill = try gpa.dupe(u32, s.use_start[0..nv]);
        defer gpa.free(fill);
        for (f.blocks.items, 0..) |b, bi| for (b.ops.items, 0..) |o, oi| {
            const at: Use = .{ .b = @intCast(bi), .op = @intCast(oi), .arg = none };
            var buf: [ir.max_args]ir.VReg = undefined;
            switch (o) {
                .bra => |x| for (x.args, 0..) |v, p| s.put(fill, v, .{ .b = at.b, .op = at.op, .arg = @intCast(p) }),
                .cbra => |x| {
                    s.put(fill, x.a, at);
                    s.put(fill, x.b, at);
                    for (x.then_args, 0..) |v, p| s.put(fill, v, .{ .b = at.b, .op = at.op, .arg = @intCast(p) });
                    for (x.else_args, 0..) |v, p| s.put(fill, v, .{ .b = at.b, .op = at.op, .arg = 1 << 31 | @as(u32, @intCast(p)) });
                },
                else => for (o.uses(&buf)) |v| s.put(fill, v, at),
            }
        };
        return s;
    }

    fn deinit(s: *Solver) void {
        const gpa = s.gpa;
        gpa.free(s.lat);
        gpa.free(s.use_start);
        gpa.free(s.uses);
        gpa.free(s.block_exec);
        gpa.free(s.edge_exec);
        s.cfg_work.deinit(gpa);
        s.ssa_work.deinit(gpa);
    }

    fn countUse(n: []u32, v: ir.VReg) void {
        n[v] += 1;
    }
    /// Calls `f(ctx, v)` for every vreg `o` reads (operands, call and edge arguments).
    fn forEachUse(o: ir.Op, ctx: anytype, comptime f: fn (@TypeOf(ctx), ir.VReg) void) void {
        var buf: [ir.max_args]ir.VReg = undefined;
        switch (o) {
            .bra => |x| for (x.args) |v| f(ctx, v),
            .cbra => |x| {
                f(ctx, x.a);
                f(ctx, x.b);
                for (x.then_args) |v| f(ctx, v);
                for (x.else_args) |v| f(ctx, v);
            },
            else => for (o.uses(&buf)) |v| f(ctx, v),
        }
    }
    fn put(s: *Solver, fill: []u32, v: ir.VReg, u: Use) void {
        s.uses[fill[v]] = u;
        fill[v] += 1;
    }

    fn solve(s: *Solver) opt.Error!void {
        s.block_exec[0] = true;
        try s.visitBlock(0);
        while (true) {
            if (s.cfg_work.pop()) |e| {
                try s.visitEdge(e);
            } else if (s.ssa_work.pop()) |v| {
                try s.visitUses(v);
            } else break;
        }
    }

    /// Lowers `v` to the meet of its value and `l`; queues its uses when it changed.
    fn lower(s: *Solver, v: ir.VReg, l: Lat) opt.Error!void {
        const cur = s.lat[v];
        if (cur.st == .over or l.st == .unknown) return;
        if (cur.st == .constant and l.st == .constant and cur.val == l.val) return;
        s.lat[v] = if (cur.st == .constant or l.st == .over) over else l;
        if (s.use_start[v] != s.use_start[v + 1]) try s.ssa_work.append(s.gpa, v);
    }

    fn markEdge(s: *Solver, b: u32, index: u1) opt.Error!void {
        const e = 2 * b + index;
        if (s.edge_exec[e]) return;
        s.edge_exec[e] = true;
        try s.cfg_work.append(s.gpa, e);
    }

    fn visitEdge(s: *Solver, e: u32) opt.Error!void {
        const ed = cfg.edgeOf(cfg.terminator(s.f, e / 2).*, @intCast(e % 2));
        for (s.f.blocks.items[ed.to].params, ed.args) |p, a| try s.lower(p, s.lat[a]);
        if (s.block_exec[ed.to]) return;
        s.block_exec[ed.to] = true;
        try s.visitBlock(ed.to);
    }

    fn visitBlock(s: *Solver, b: u32) opt.Error!void {
        for (s.f.blocks.items[b].ops.items) |o| {
            if (o.def()) |d| try s.lower(d, s.eval(o)) else if (o.isTerminator()) try s.visitTerminator(b);
        }
    }

    fn visitUses(s: *Solver, v: ir.VReg) opt.Error!void {
        for (s.uses[s.use_start[v]..s.use_start[v + 1]]) |u| {
            if (!s.block_exec[u.b]) continue;
            const o = s.f.blocks.items[u.b].ops.items[u.op];
            if (u.arg == none) {
                if (o.def()) |d| try s.lower(d, s.eval(o)) else if (o.isTerminator()) try s.visitTerminator(u.b);
                continue;
            }
            const index: u1 = @intCast(u.arg >> 31);
            if (!s.edge_exec[2 * u.b + index]) continue;
            const to = cfg.edgeOf(o, index).to;
            try s.lower(s.f.blocks.items[to].params[u.arg & ~(@as(u32, 1) << 31)], s.lat[v]);
        }
    }

    fn visitTerminator(s: *Solver, b: u32) opt.Error!void {
        switch (cfg.terminator(s.f, b).*) {
            .br, .bra => try s.markEdge(b, 0),
            .cbr => |x| try s.branch(b, x.cond, x.a, x.b),
            .cbra => |x| try s.branch(b, x.cond, x.a, x.b),
            else => {},
        }
    }

    fn branch(s: *Solver, b: u32, cc: ir.CondCode, a: ir.VReg, bv: ir.VReg) opt.Error!void {
        switch (s.decide(cc, a, bv)) {
            .unknown => {},
            .then => try s.markEdge(b, 0),
            .@"else" => try s.markEdge(b, 1),
            .both => {
                try s.markEdge(b, 0);
                try s.markEdge(b, 1);
            },
        }
    }

    const Decision = enum { unknown, then, @"else", both };

    /// Which edges of `cbr cc a b` can execute given the current lattice.
    fn decide(s: *const Solver, cc: ir.CondCode, a: ir.VReg, b: ir.VReg) Decision {
        const la = s.lat[a];
        const lb = s.lat[b];
        if (la.st == .unknown or lb.st == .unknown) return .unknown;
        // `x cc x` has the same outcome for every x.
        const taken = if (a == b) ir.cond(cc, 0, 0) else if (la.st == .constant and lb.st == .constant)
            ir.cond(cc, ir.toI(la.val), ir.toI(lb.val))
        else
            return .both;
        return if (taken) .then else .@"else";
    }

    /// Lattice value of the result of `o` (an op with a definition).
    fn eval(s: *const Solver, o: ir.Op) Lat {
        return switch (o) {
            .@"const" => |x| constant(ir.fromI(x.imm)),
            .fconst => |x| constant(x.bits),
            .copy => |x| s.lat[x.src],
            .arith => |x| s.evalArith(x.op, x.a, x.b),
            .typed => |x| blk: {
                const a = s.lat[x.a];
                const b = s.lat[x.b];
                if (a.st == .over or b.st == .over) break :blk over;
                if (a.st == .unknown or b.st == .unknown) break :blk unknown;
                const r = ir.typedArith(x.op, x.ty, a.val, b.val);
                if (hasNan(x.ty, a.val) or hasNan(x.ty, b.val) or hasNan(x.ty, r)) break :blk over;
                break :blk constant(r);
            },
            .fcmp => |x| blk: {
                const a = s.lat[x.a];
                const b = s.lat[x.b];
                if (a.st == .over or b.st == .over) break :blk over;
                if (a.st == .unknown or b.st == .unknown) break :blk unknown;
                if (hasNan(x.ty, a.val) or hasNan(x.ty, b.val)) break :blk over;
                const r = if (x.ty == .f32)
                    ir.fcond(x.cond, ir.getF32(a.val), ir.getF32(b.val))
                else
                    ir.fcond(x.cond, ir.getF64(a.val), ir.getF64(b.val));
                break :blk constant(@intFromBool(r));
            },
            .cvt => |x| blk: {
                const a = s.lat[x.src];
                if (a.st != .constant) break :blk a;
                if (hasNan(x.from, a.val)) break :blk over;
                const r = ir.convert(x.to, x.from, a.val);
                break :blk if (hasNan(x.to, r)) over else constant(r);
            },
            .splat => |x| blk: {
                const a = s.lat[x.src];
                if (a.st != .constant) break :blk a;
                if (hasNan(x.ty.lane(), a.val)) break :blk over;
                break :blk constant(splatVal(x.ty, a.val));
            },
            .extract => |x| blk: {
                const a = s.lat[x.src];
                if (a.st != .constant) break :blk a;
                const r = ir.getLane(x.ty, a.val, x.lane);
                break :blk if (hasNan(x.ty.lane(), r)) over else constant(r);
            },
            .load, .call => over,
            else => unreachable,
        };
    }

    fn evalArith(s: *const Solver, op: ir.Arith, av: ir.VReg, bv: ir.VReg) Lat {
        const a = s.lat[av];
        const b = s.lat[bv];
        const ka: ?i64 = if (a.st == .constant) ir.toI(a.val) else null;
        const kb: ?i64 = if (b.st == .constant) ir.toI(b.val) else null;
        // Results fixed by one operand alone (whatever the other one holds).
        switch (op) {
            .@"and", .mul => if (ka == 0 or kb == 0) return constant(0),
            .@"or" => if (ka == -1 or kb == -1) return constant(ir.fromI(-1)),
            .shl, .shr, .sar => if (ka == 0) return constant(0),
            .sub, .xor => if (av == bv and a.st != .unknown) return constant(0),
            else => {},
        }
        if (a.st == .over or b.st == .over) return over;
        if (a.st == .unknown or b.st == .unknown) return unknown;
        const x = ka.?;
        const y = kb.?;
        if ((op == .sdiv or op == .srem) and (y == 0 or (x == std.math.minInt(i64) and y == -1))) return over;
        return constant(ir.fromI(ir.arith(op, x, y)));
    }
};

/// Whether a float lane of `v` (of type `ty`; never for integer types) is a NaN.
fn hasNan(ty: ir.Type, v: ir.Val) bool {
    if (!ty.isFloat()) return false;
    for (0..ty.lanes()) |l| {
        const x = ir.getLane(ty, v, @intCast(l));
        if (if (ty.lane() == .f32) std.math.isNan(ir.getF32(x)) else std.math.isNan(ir.getF64(x))) return true;
    }
    return false;
}

/// The interpreter's `splat.<ty>` of the scalar `x`.
fn splatVal(ty: ir.Type, x: ir.Val) ir.Val {
    var r: ir.Val = 0;
    for (0..ty.lanes()) |l| r = ir.setLane(ty, r, @intCast(l), x);
    return r;
}

const Key = struct { ty: ir.Type, val: ir.Val };

/// Canonical constant vregs (one per type and bits), defined at the top of block 0.
const Consts = struct {
    map: std.AutoHashMapUnmanaged(Key, ir.VReg) = .empty,
    /// New constant ops (inserted into block 0 after the existing leading constants).
    ops: std.ArrayList(ir.Op) = .empty,

    fn deinit(k: *Consts, gpa: std.mem.Allocator) void {
        k.map.deinit(gpa);
        k.ops.deinit(gpa);
    }

    /// The vreg holding `val` of type `ty`, created if needed; null for a vector whose lanes
    /// differ (no single op materializes it).
    fn get(k: *Consts, c: *opt.Ctx, ty: ir.Type, val: ir.Val) opt.Error!?ir.VReg {
        if (k.map.get(.{ .ty = ty, .val = val })) |v| return v;
        var lane: ?ir.VReg = null;
        if (ty.isVector()) {
            const l0 = ir.getLane(ty, val, 0);
            if (splatVal(ty, l0) != val) return null;
            lane = (try k.get(c, ty.lane(), l0)).?;
        }
        const v = try c.newVReg(ty);
        try k.ops.append(c.gpa, switch (ty) {
            .i64 => .{ .@"const" = .{ .dst = v, .imm = ir.toI(val) } },
            .f32, .f64 => .{ .fconst = .{ .dst = v, .ty = ty, .bits = @truncate(val) } },
            else => .{ .splat = .{ .ty = ty, .dst = v, .src = lane.? } },
        });
        try k.map.put(c.gpa, .{ .ty = ty, .val = val }, v);
        return v;
    }
};

pub fn run(c: *opt.Ctx) opt.Error!bool {
    const gpa = c.gpa;
    const f = c.f;
    var s = try Solver.init(gpa, f);
    defer s.deinit();
    try s.solve();

    var changed = false;
    // Branches with a single executable edge.
    for (f.blocks.items, 0..) |*blk, bi| {
        if (!s.block_exec[bi]) continue;
        const term = &blk.ops.items[blk.ops.items.len - 1];
        if (term.* != .cbr and term.* != .cbra) continue;
        const then = s.edge_exec[2 * bi];
        if (then == s.edge_exec[2 * bi + 1]) continue;
        const ed = cfg.edgeOf(term.*, if (then) 0 else 1);
        term.* = if (ed.args.len == 0) .{ .br = ed.to } else .{ .bra = .{ .target = ed.to, .args = ed.args } };
        c.taken.items[bi] = null;
        changed = true;
    }

    // Constants that already lead block 0 (e.g. from ssa.build or an earlier run) are reused.
    var consts: Consts = .{};
    defer consts.deinit(gpa);
    const entry = f.blocks.items[0].ops.items;
    var lead: usize = 0;
    while (lead < entry.len) : (lead += 1) {
        const o = entry[lead];
        if (o != .@"const" and o != .fconst and o != .splat) break;
        const d = o.def().?;
        if (s.lat[d].st != .constant) break;
        const gop = try consts.map.getOrPut(gpa, .{ .ty = c.types.items[d], .val = s.lat[d].val });
        if (!gop.found_existing) gop.value_ptr.* = d;
    }

    // Rename the uses of every other constant vreg.
    const nv = f.num_vregs;
    const is_const_def = try gpa.alloc(bool, nv);
    defer gpa.free(is_const_def);
    @memset(is_const_def, false);
    for (f.blocks.items) |b| for (b.ops.items) |o| switch (o) {
        inline .@"const", .fconst => |x| is_const_def[x.dst] = true,
        else => {},
    };
    var map: std.ArrayList(ir.VReg) = .empty;
    defer map.deinit(gpa);
    var renamed = false;
    for (0..nv) |vi| {
        const v: ir.VReg = @intCast(vi);
        if (s.lat[v].st != .constant or is_const_def[v] or s.use_start[v] == s.use_start[v + 1]) continue;
        const k = try consts.get(c, c.types.items[v], s.lat[v].val) orelse continue;
        if (k == v) continue;
        if (!renamed) {
            try map.ensureTotalCapacity(gpa, nv);
            for (0..nv) |i| map.appendAssumeCapacity(@intCast(i));
            renamed = true;
        }
        map.items[v] = k;
    }
    if (consts.ops.items.len != 0) {
        try f.blocks.items[0].ops.insertSlice(c.irAlloc(), lead, consts.ops.items);
        changed = true;
    }
    if (renamed) {
        for (nv..f.num_vregs) |i| try map.append(gpa, @intCast(i));
        try opt.renameUses(c, map.items);
        changed = true;
    }
    return changed;
}

// ---- tests -------------------------------------------------------------------------------

/// Helpers shared by the sccp and instcombine tests.
pub const test_util = struct {
    /// A leak-checking allocator that records no stack traces: `std.testing.allocator`
    /// unwinds the stack on every allocation, which dominates random-program tests.
    pub const FastAlloc = std.heap.DebugAllocator(.{ .stack_trace_frames = 0 });

    /// Parses `src`, builds SSA, runs `pass` until it reports no change (at most 4 times; a
    /// fifth run must change nothing), checking the IR after each run, compares the result
    /// with the original on `testgen.default_inputs`, deletes dead ops (`dropDead`) and prints
    /// the function into `out`.
    pub fn runOn(comptime pass: testgen.Pass, src: []const u8, out: *std.ArrayList(u8)) !void {
        const gpa = std.testing.allocator;
        var ref = try ir.parse(gpa, src);
        defer ref.deinit();
        var f = try ir.parse(gpa, src);
        defer f.deinit();
        const options: opt.Options = .{ .verify_each = true };
        var stats: opt.Stats = .{};
        var c = try opt.Ctx.init(gpa, &f, &options, &stats);
        defer c.deinit();
        try ssa.build(&c);
        try c.check("ssa.build");
        for (0..4) |_| {
            const changed = try pass(&c);
            c.check("pass") catch |e| {
                var dbg: std.ArrayList(u8) = .empty;
                defer dbg.deinit(gpa);
                try f.print(gpa, &dbg);
                std.debug.print("invalid IR after the pass:\n{s}\n", .{dbg.items});
                return e;
            };
            if (!changed) break;
        } else {
            try std.testing.expect(!try pass(&c));
            try c.check("pass");
        }
        try testgen.expectSame(&ref, &f, &testgen.default_inputs, 20_000);
        dropDead(&f);
        try c.check("dropDead");
        try testgen.expectSame(&ref, &f, &testgen.default_inputs, 20_000);
        try f.print(gpa, out);
    }

    /// Deletes ops whose result nothing reads (except calls), repeatedly, so tests can assert
    /// on the live IR without depending on the dce pass.
    pub fn dropDead(f: *ir.Func) void {
        var changed = true;
        while (changed) {
            changed = false;
            const n = opt.useCounts(std.testing.allocator, f) catch unreachable;
            defer std.testing.allocator.free(n);
            for (f.blocks.items) |*b| {
                var k: usize = 0;
                for (b.ops.items) |o| {
                    if (o.def()) |d| if (n[d] == 0 and o != .call) {
                        changed = true;
                        continue;
                    };
                    b.ops.items[k] = o;
                    k += 1;
                }
                b.ops.items.len = k;
            }
        }
    }

    pub fn expectContains(text: []const u8, needle: []const u8) !void {
        if (std.mem.indexOf(u8, text, needle) != null) return;
        std.debug.print("expected \"{s}\" in:\n{s}\n", .{ needle, text });
        return error.TestExpectedEqual;
    }
    pub fn expectLacks(text: []const u8, needle: []const u8) !void {
        if (std.mem.indexOf(u8, text, needle) == null) return;
        std.debug.print("did not expect \"{s}\" in:\n{s}\n", .{ needle, text });
        return error.TestExpectedEqual;
    }

    /// The function printed in `text` returns a `const imm` vreg (from every `ret`).
    pub fn expectRetConst(text: []const u8, imm: i64) !void {
        var f = try ir.parse(std.testing.allocator, text);
        defer f.deinit();
        var found = false;
        for (f.blocks.items) |b| {
            const term = b.ops.items[b.ops.items.len - 1];
            if (term != .ret) continue;
            found = false;
            for (f.blocks.items) |d| for (d.ops.items) |o| if (o == .@"const" and o.@"const".dst == term.ret) {
                try std.testing.expectEqual(imm, o.@"const".imm);
                found = true;
            };
            if (!found) break;
        }
        if (found) return;
        std.debug.print("expected every ret to return const {d} in:\n{s}\n", .{ imm, text });
        return error.TestExpectedEqual;
    }
};

const t = std.testing;
const runOn = test_util.runOn;
const expectContains = test_util.expectContains;
const expectLacks = test_util.expectLacks;

fn expectFolds(src: []const u8, imm: i64) !void {
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(t.allocator);
    try runOn(run, src, &out);
    try test_util.expectRetConst(out.items, imm);
}

test "sccp: straight-line integer folding and absorbing operands" {
    try expectFolds(
        \\func s 2
        \\e:
        \\  v2 = const 6
        \\  v3 = const 7
        \\  v4 = mul v2 v3
        \\  v5 = const -1
        \\  v6 = shl v4 v5
        \\  v7 = sar v6 v5
        \\  v8 = add v7 v4
        \\  ret v8
    , ir.arith(.sar, ir.arith(.shl, 42, -1), -1) +% 42);
    // and 0, mul 0, or -1, x - x, x ^ x, 0 << x fold even though x is a parameter.
    inline for (.{
        .{ "and v0 v2", 0 }, .{ "mul v2 v1", 0 }, .{ "or v0 v3", -1 }, .{ "sub v1 v1", 0 },
        .{ "xor v0 v0", 0 }, .{ "shl v2 v0", 0 }, .{ "sar v2 v1", 0 },
    }) |case| {
        try expectFolds("func s 2\ne:\n  v2 = const 0\n  v3 = const -1\n  v4 = " ++ case[0] ++ "\n  ret v4\n", case[1]);
    }
}

test "sccp: branches on constants; unreachable paths do not pollute parameters" {
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(t.allocator);
    try runOn(run,
        \\func b 2
        \\e:
        \\  v2 = const 3
        \\  v3 = const 5
        \\  cbr lt v2 v3 yes no
        \\yes:
        \\  v4 = const 10
        \\  br join
        \\no:
        \\  v4 = call 0x1000 v0
        \\  br join
        \\join:
        \\  v5 = add v4 v2
        \\  ret v5
    , &out);
    try expectLacks(out.items, "cbr");
    try test_util.expectRetConst(out.items, 13);
}

test "sccp: x cc x branches are decided whatever x holds" {
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(t.allocator);
    try runOn(run,
        \\func b 2
        \\e:
        \\  cbr ule v0 v0 yes no
        \\yes:
        \\  v2 = const 1
        \\  ret v2
        \\no:
        \\  v3 = const 2
        \\  ret v3
    , &out);
    try expectLacks(out.items, "cbr");
    try expectContains(out.items, "br yes");
}

test "sccp: constant loop trip folds only what is sound" {
    // i is a loop-carried counter (overdefined, the exit test stays); k = phi(7, k + 0) is 7.
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(t.allocator);
    try runOn(run,
        \\func l 2
        \\e:
        \\  v2 = const 0
        \\  v3 = const 7
        \\  v4 = const 3
        \\  v6 = const 0
        \\  br h
        \\h:
        \\  cbr lt v2 v4 body exit
        \\body:
        \\  v5 = const 1
        \\  v2 = add v2 v5
        \\  v3 = add v3 v6
        \\  br h
        \\exit:
        \\  v7 = mul v3 v2
        \\  store v1 0 v7
        \\  ret v3
    , &out);
    try expectContains(out.items, "cbr lt");
    try expectContains(out.items, "add v");
    try test_util.expectRetConst(out.items, 7);
}

test "sccp: a loop whose first exit test is constant never runs its body" {
    // The back edge never becomes executable: the header parameter is the entry value.
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(t.allocator);
    try runOn(run,
        \\func l 1
        \\e:
        \\  v1 = const 5
        \\  br h
        \\h:
        \\  v2 = const 9
        \\  cbr lt v1 v2 exit body
        \\body:
        \\  v1 = call 0x1000 v1
        \\  br h
        \\exit:
        \\  v3 = add v1 v1
        \\  ret v3
    , &out);
    try expectLacks(out.items, "cbr");
    try test_util.expectRetConst(out.items, 10);
}

test "sccp: block parameters merge constants only on executable edges" {
    // `a`'s cbra folds; its dropped edge and the never-executed block `b` carry 2.
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(t.allocator);
    try runOn(run,
        \\func p 2
        \\e:
        \\  v2 = const 1
        \\  v3 = const 2
        \\  cbr eq v2 v2 a b
        \\a:
        \\  cbr ne v2 v3 j:v2 j:v3
        \\b:
        \\  br j v3
        \\j(v4):
        \\  v5 = mul v4 v0
        \\  store v1 0 v5
        \\  ret v4
    , &out);
    try expectLacks(out.items, "cbr");
    try expectContains(out.items, "mul v2 v0");
    try test_util.expectRetConst(out.items, 1);
}

test "sccp: sdiv/srem by zero and minInt / -1 are never folded" {
    inline for (.{ .{ "sdiv", -3 }, .{ "srem", -2 } }) |case| {
        const op = case[0];
        var out: std.ArrayList(u8) = .empty;
        defer out.deinit(t.allocator);
        try runOn(run, "func d 2\ne:\n  v2 = const 0\n  v3 = const -9223372036854775808\n  v4 = const -1\n  v8 = const -17\n" ++
            "  v5 = " ++ op ++ " v8 v2\n  v6 = " ++ op ++ " v3 v4\n  store v1 0 v5\n  store v1 8 v6\n" ++
            "  v9 = const 5\n  v10 = " ++ op ++ " v8 v9\n  ret v10\n", &out);
        try expectContains(out.items, op ++ " v8 v2");
        try expectContains(out.items, op ++ " v3 v4");
        try test_util.expectRetConst(out.items, case[1]);
    }
}

test "sccp: floats fold exactly, NaN never, -0.0 and +0.0 kept apart" {
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(t.allocator);
    try runOn(run,
        \\func f 2
        \\e:
        \\  v2 = fconst.f64 -0.0
        \\  v3 = fconst.f64 0.0
        \\  v4 = min.f64 v2 v3
        \\  v5 = add.f64 v2 v2
        \\  v6 = fconst.f64 -1.0
        \\  v7 = mul.f64 v3 v6
        \\  store.f64 v1 0 v4
        \\  store.f64 v1 8 v5
        \\  store.f64 v1 16 v7
        \\  v8 = div.f64 v3 v3
        \\  store.f64 v1 24 v8
        \\  v9 = fcmp.f64 lt v8 v3
        \\  v10 = fcmp.f64 eq v2 v3
        \\  v11 = add v9 v10
        \\  ret v11
    , &out);
    // min(-0, +0) = +0 (the second operand when equal), -0 + -0 = -0, +0 * -1 = -0; 0/0 is a
    // NaN: kept, like the fcmp reading it; -0 == +0 folds to 1.
    try expectContains(out.items, "store.f64 v1 0 v3");
    try expectContains(out.items, "store.f64 v1 8 v2");
    try expectContains(out.items, "store.f64 v1 16 v2");
    try expectContains(out.items, "div.f64 v3 v3");
    try expectContains(out.items, "fcmp.f64 lt v8 v3");
    try expectLacks(out.items, "fcmp.f64 eq");
}

test "sccp: cvt folds out-of-range and rounding exactly, never NaN" {
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(t.allocator);
    try runOn(run,
        \\func c 2
        \\e:
        \\  v2 = fconst.f64 1e300
        \\  v3 = cvt.i64.f64 v2
        \\  v4 = fconst.f32 -2.75
        \\  v5 = cvt.i64.f32 v4
        \\  v6 = const 9007199254740993
        \\  v7 = cvt.f64.i64 v6
        \\  v8 = cvt.f32.f64 v2
        \\  store.f64 v1 0 v7
        \\  store.f32 v1 8 v8
        \\  v9 = fconst.f64 0.0
        \\  v10 = div.f64 v9 v9
        \\  v11 = cvt.i64.f64 v10
        \\  v12 = add v3 v5
        \\  v13 = add v12 v11
        \\  ret v13
    , &out);
    try expectContains(out.items, "cvt.i64.f64 v10");
    try expectLacks(out.items, "cvt.i64.f32");
    try expectLacks(out.items, "cvt.f64.i64");
    try expectLacks(out.items, "cvt.f32.f64");
    try expectLacks(out.items, "cvt.i64.f64 v2");
}

test "sccp: vectors fold lane-wise; uniform ones become splats of a constant" {
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(t.allocator);
    try runOn(run,
        \\func v 2
        \\e:
        \\  v2 = const 4294967298
        \\  v3 = splat.v4i32 v2
        \\  v4 = add.v4i32 v3 v3
        \\  v5 = extract.v4i32 v4 3
        \\  v6 = const -3
        \\  v7 = splat.v2i64 v6
        \\  v8 = xor.v2i64 v7 v7
        \\  store.v2i64 v1 0 v8
        \\  v9 = fconst.f32 1.5
        \\  v10 = splat.v4f32 v9
        \\  v11 = mul.v4f32 v10 v10
        \\  store.v4f32 v1 16 v11
        \\  v12 = extract.v4f32 v11 2
        \\  v13 = cvt.i64.f32 v12
        \\  v14 = add v5 v13
        \\  ret v14
    , &out);
    try test_util.expectRetConst(out.items, 4 + 2);
    try expectLacks(out.items, "mul.v4f32");
    try expectLacks(out.items, "xor.v2i64");
    try expectContains(out.items, "fconst.f32 2.25");
}

test "sccp: random programs" {
    var da: test_util.FastAlloc = .init;
    defer if (da.deinit() == .leak) @panic("sccp: memory leak");
    const gpa = da.allocator();
    try testgen.fuzzPass(gpa, 1, 150, .{}, &.{}, run);
    try testgen.fuzzPass(gpa, 2, 100, .{ .loops = false }, &.{}, run);
    try testgen.fuzzPass(gpa, 3, 100, .{ .blocks = 10, .ops = 10, .calls = false, .memory = false }, &.{}, run);
    // A second run on its own output.
    try testgen.fuzzPass(gpa, 4, 100, .{ .blocks = 4, .ops = 12 }, &.{run}, run);
}
