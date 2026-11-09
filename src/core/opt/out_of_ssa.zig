//! Out of SSA with copy coalescing (Boissinot et al., "Revisiting Out-of-SSA Translation for
//! Correctness, Code Quality, and Efficiency", CGO 2009; interference test of Budimlić et al.,
//! "Fast Copy Coalescing and Live-Range Identification", PLDI 2002).
//!
//! Every block parameter and every argument passed to it start in their own congruence class
//! (union-find over vregs). Edges are visited hottest first (`Ctx.edgeFreq`, else the loop
//! depth of the edge) and each (parameter, argument) pair merges the two classes unless a
//! member of one interferes with a member of the other, so the copies that remain land on cold
//! edges. Each class becomes one mutable vreg; the arguments still in another class than their
//! parameter form the parallel copy of their edge, sequentialized at the end of the source
//! block (one successor) or in a block splitting the edge (`cfg.splitEdge`), with one
//! temporary per copy cycle. A conditional edge's copies go before the branch instead (no edge
//! block, no jump) when the other edge needs a subset of them, is not hotter, and nothing the
//! branch or the other successor reads is overwritten (`Hoist`): typically the guard of a
//! rotated loop, whose exit edge passes the same initial values as the entry edge. A counted loop's induction variable thus becomes one vreg updated
//! in place, and an unchanged loop-carried value costs no copy.
//!
//! Interference: two SSA values interfere when the one whose definition dominates the other's
//! is live just after that other definition (liveness per block from the SSA def-use chains:
//! edge arguments are uses at the end of their source block, parameters are defined at the top
//! of their block). Parameters of one block (and the function parameters) are defined
//! together by the incoming parallel copy, so they always interfere. Two classes are compared
//! by one walk over their members in dominance order with a stack (the "dominance forest"):
//! each member is checked against its nearest dominating member only, which is exact because
//! no class contains two interfering values.
//!
//! Classes never merge when both contain a function parameter (each keeps its number) or when
//! one is a `const`/`fconst` result (the IR Compilers fold single-definition constants into
//! immediates, so a copy from the constant is cheaper than a register for it). Arguments of
//! parameters nothing reads are dropped without a copy.
//!
//! The result has no block parameters, its vregs are renumbered densely (function parameters
//! keep `v0..v(params-1)`, then in order of first appearance) with `Ctx.types` to match, and
//! `c.in_ssa` is false. Unreachable blocks are deleted first.
const std = @import("std");
const ir = @import("../ir.zig");
const opt = @import("opt.zig");
const cfg = @import("cfg.zig");
const ssa = @import("ssa.zig");

const none = cfg.none;
const EdgeArgs = @TypeOf(cfg.edgeOf(.{ .br = 0 }, 0));

/// Merges whose classes together exceed this many members count as interfering (bounds the
/// merge walk on pathological inputs; real classes are small).
const max_class = 1024;

pub fn run(c: *opt.Ctx) opt.Error!void {
    std.debug.assert(c.in_ssa);
    const gpa = c.gpa;
    const f = c.f;
    {
        var g = try cfg.Cfg.init(gpa, f);
        defer g.deinit();
        if (g.rpo.len != f.blocks.items.len) {
            const keep = try gpa.alloc(bool, f.blocks.items.len);
            defer gpa.free(keep);
            for (keep, 0..) |*k, b| k.* = g.reachable(@intCast(b));
            try c.compactBlocks(keep);
        }
    }
    var g = try cfg.Cfg.init(gpa, f);
    defer g.deinit();
    var lv = try Liveness.init(c, &g);
    defer lv.deinit(gpa);
    var cls = try Classes.init(gpa, c, &lv);
    defer cls.deinit();

    // Edges with arguments, hottest first.
    var w = try Weights.init(c, &g);
    defer w.deinit();
    var edges: std.ArrayList(WEdge) = .empty;
    defer edges.deinit(gpa);
    for (f.blocks.items, 0..) |b, bi| {
        const term = b.ops.items[b.ops.items.len - 1];
        for (0..ssa.numSuccs(term)) |i| if (cfg.edgeOf(term, @intCast(i)).args.len != 0)
            try edges.append(gpa, .{ .from = @intCast(bi), .index = @intCast(i), .key = w.of(@intCast(bi), @intCast(i)) });
    }
    std.mem.sort(WEdge, edges.items, {}, WEdge.hotterFirst);
    for (edges.items) |e| {
        const x = cfg.edgeOf(cfg.terminator(f, e.from).*, e.index);
        for (f.blocks.items[x.to].params, x.args) |p, a| if (lv.used[p]) try cls.tryMerge(p, a);
    }

    // Split the conditional edges that still need copies, then place every edge's parallel
    // copy before the `br` that ends its block.
    const n0: u32 = @intCast(f.blocks.items.len);
    const after = try gpa.alloc(u32, n0); // edge block placed right after block b, or none
    defer gpa.free(after);
    @memset(after, none);
    const hoisted = try gpa.alloc(?u1, n0); // edge whose copies go before the branch
    defer gpa.free(hoisted);
    @memset(hoisted, null);
    var hoist: Hoist = try .init(gpa, f);
    defer hoist.deinit(gpa);
    var moved = false;
    for (0..n0) |bi| {
        const b: u32 = @intCast(bi);
        const term = cfg.terminator(f, b).*;
        if (term != .cbra) continue;
        var split = [2]bool{ cls.needsCopies(cfg.edgeOf(term, 0), f), cls.needsCopies(cfg.edgeOf(term, 1), f) };
        if (split[0] or split[1]) for ([2]u1{ 0, 1 }) |i| {
            if (split[i] and !w.hotter(b, 1 - i) and try hoist.check(&cls, &lv, b, i)) {
                hoisted[b] = i;
                split = .{ false, false };
                break;
            }
        };
        // An edge block goes right after its source when that keeps a fall-through into the
        // edge's target (the next block), else when it is on the hotter edge; else at the end.
        var next_to: ?u1 = null;
        for (0..2) |i| if (split[i] and cfg.edgeOf(term, @intCast(i)).to == b + 1) {
            next_to = @intCast(i);
        };
        if (next_to == null) for (0..2) |i| if (split[i] and w.hotter(b, @intCast(i))) {
            next_to = @intCast(i);
        };
        for (0..2) |i| if (split[i]) {
            const nb = try cfg.splitEdge(c, b, @intCast(i));
            if (next_to != null and next_to.? == i) {
                after[b] = nb;
                moved = true;
            }
        };
    }
    var pc: ParallelCopy = try .init(gpa, f.num_vregs);
    defer pc.deinit(gpa);
    for (f.blocks.items, 0..) |*b, bi| {
        const term = &b.ops.items[b.ops.items.len - 1];
        const e: ?EdgeArgs = switch (term.*) {
            .bra => cfg.edgeOf(term.*, 0),
            .cbra => if (bi < n0) if (hoisted[bi]) |i| cfg.edgeOf(term.*, i) else null else null,
            else => continue,
        };
        pc.clear();
        if (e) |x| for (f.blocks.items[x.to].params, x.args) |p, a| if (lv.used[p]) {
            const d = cls.find(p);
            const s = cls.find(a);
            if (d != s) try pc.add(gpa, d, s);
        };
        switch (term.*) {
            .bra => |x| term.* = .{ .br = x.target },
            else => ssa.setEdgeArgs(term, .{ &.{}, &.{} }), // copies before the branch, or none
        }
        if (pc.copies.items.len != 0) {
            const seq = try pc.sequentialize(c);
            try b.ops.insertSlice(c.irAlloc(), b.ops.items.len - 1, seq);
        }
    }
    for (f.blocks.items) |*b| b.params = &.{};
    try renumber(c, &cls);
    c.in_ssa = false;

    if (moved) {
        const order = try gpa.alloc(u32, f.blocks.items.len);
        defer gpa.free(order);
        const placed = try gpa.alloc(bool, f.blocks.items.len);
        defer gpa.free(placed);
        @memset(placed, false);
        var k: usize = 0;
        for (0..n0) |b| {
            order[k] = @intCast(b);
            k += 1;
            if (after[b] != none) {
                order[k] = after[b];
                placed[after[b]] = true;
                k += 1;
            }
        }
        for (n0..f.blocks.items.len) |b| if (!placed[b]) {
            order[k] = @intCast(b);
            k += 1;
        };
        try c.reorderBlocks(order);
    }
}

/// Whether the parallel copy of conditional edge `i` of block `b` can run before the branch,
/// on both edges: every copy of the other edge `j` is also one of edge `i` (same class pair),
/// the branch operands are not overwritten, and every class written only for edge `i` is dead
/// on edge `j` (no member live into its target, no copy-free argument of edge `j` reads it).
/// Classes with more than `max_members` members are not scanned (no hoisting), which keeps the
/// test linear.
const Hoist = struct {
    f: *const ir.Func,
    /// Source class of the copy into class `d` on edge `i`, or `none`.
    src: []ir.VReg,
    dsts: std.ArrayList(ir.VReg) = .empty,

    const max_members = 64;

    fn init(gpa: std.mem.Allocator, f: *const ir.Func) std.mem.Allocator.Error!Hoist {
        const src = try gpa.alloc(ir.VReg, f.num_vregs);
        @memset(src, none);
        return .{ .f = f, .src = src };
    }
    fn deinit(h: *Hoist, gpa: std.mem.Allocator) void {
        gpa.free(h.src);
        h.dsts.deinit(gpa);
    }

    fn check(h: *Hoist, cls: *Classes, lv: *const Liveness, b: u32, i: u1) std.mem.Allocator.Error!bool {
        const ops = h.f.blocks.items[b].ops.items;
        const term = ops[ops.len - 1];
        const ei = cfg.edgeOf(term, i);
        const ej = cfg.edgeOf(term, 1 - i);
        for (h.f.blocks.items[ei.to].params, ei.args) |p, a| if (lv.used[p]) {
            const d = cls.find(p);
            const s = cls.find(a);
            if (d == s) continue;
            h.src[d] = s;
            try h.dsts.append(cls.gpa, d);
        };
        defer {
            for (h.dsts.items) |d| h.src[d] = none;
            h.dsts.clearRetainingCapacity();
        }
        if (h.src[cls.find(term.cbra.a)] != none or h.src[cls.find(term.cbra.b)] != none) return false;
        // Edge `j`: its copies must be among edge `i`'s (they then run as part of the same
        // parallel copy); a copy-free argument must not read a class edge `i` writes.
        for (h.f.blocks.items[ej.to].params, ej.args) |p, a| if (lv.used[p]) {
            const d = cls.find(p);
            const s = cls.find(a);
            if (d == s) {
                if (h.src[s] != none) return false;
            } else if (h.src[d] != s) return false;
        };
        // No member of a class edge `i` writes may be live into `j`'s target (a parameter of
        // that target is defined there, not live in; its copy was matched above).
        for (h.dsts.items) |d| {
            var root = d;
            const members = cls.list(&root);
            if (members.len > max_members) return false;
            for (members) |m| if (lv.liveIn(m, ej.to)) return false;
        }
        return true;
    }
};

/// An edge with arguments and its sort key (larger is hotter).
const WEdge = struct {
    from: u32,
    index: u1,
    key: [2]u64,

    fn hotterFirst(_: void, a: WEdge, b: WEdge) bool {
        if (a.key[0] != b.key[0]) return a.key[0] > b.key[0];
        if (a.key[1] != b.key[1]) return a.key[1] > b.key[1];
        if (a.from != b.from) return a.from < b.from;
        return a.index < b.index;
    }
};

/// Edge weights: the profile count when every block has one, else (loop depth of the edge,
/// profile count or 0).
const Weights = struct {
    c: *opt.Ctx,
    loops: ?cfg.Loops,

    fn init(c: *opt.Ctx, g: *const cfg.Cfg) opt.Error!Weights {
        var known = true;
        for (c.freq.items) |q| known = known and q != null;
        return .{ .c = c, .loops = if (known) null else try cfg.Loops.init(c.gpa, c.f, g) };
    }
    fn deinit(w: *Weights) void {
        if (w.loops) |*l| l.deinit();
    }
    fn of(w: *const Weights, from: u32, index: u1) [2]u64 {
        const fr = w.c.edgeFreq(from, index) orelse 0;
        const l = w.loops orelse return .{ fr, 0 };
        const to = cfg.edgeOf(cfg.terminator(w.c.f, from).*, index).to;
        // Depth of the innermost loop containing both ends.
        var a = l.loop_of[from];
        var b = l.loop_of[to];
        while (a != b) {
            const da = if (a == none) 0 else l.loops[a].depth;
            const db = if (b == none) 0 else l.loops[b].depth;
            if (da >= db) a = l.loops[a].parent else b = l.loops[b].parent;
        }
        return .{ if (a == none) 0 else l.loops[a].depth, fr };
    }
    /// Edge `index` of conditional block `from` is strictly hotter than the other one.
    fn hotter(w: *const Weights, from: u32, index: u1) bool {
        const x = w.of(from, index);
        const y = w.of(from, 1 - index);
        return x[0] > y[0] or (x[0] == y[0] and x[1] > y[1]);
    }
};

/// Definition points, per-block live-out sets and last uses of an SSA function without
/// unreachable blocks. Positions within block `b`: entry definitions (block parameters, or
/// the function parameters in block 0) are `0..base[b]`, op `i` is at `base[b] + i`.
const Liveness = struct {
    g: *const cfg.Cfg,
    def_block: []u32,
    def_pos: []u32,
    base: []u32,
    /// Read anywhere (operands, call or edge arguments).
    used: []bool,
    /// Live-out vregs of block `b`: `out_vars[out_start[b]..out_start[b + 1]]`, ascending.
    out_start: []u32,
    out_vars: []ir.VReg,
    /// Last read of each vreg read in block `b` (ascending by vreg): `last[last_start[b]..]`.
    last_start: []u32,
    last: []Use,

    const Use = struct { v: ir.VReg, pos: u32 };

    fn init(c: *opt.Ctx, g: *const cfg.Cfg) opt.Error!Liveness {
        const gpa = c.gpa;
        const f = c.f;
        const nv = f.num_vregs;
        const nb: u32 = @intCast(f.blocks.items.len);
        var lv: Liveness = .{
            .g = g,
            .def_block = try gpa.alloc(u32, nv),
            .def_pos = &.{},
            .base = &.{},
            .used = &.{},
            .out_start = &.{},
            .out_vars = &.{},
            .last_start = &.{},
            .last = &.{},
        };
        errdefer lv.deinit(gpa);
        lv.def_pos = try gpa.alloc(u32, nv);
        lv.base = try gpa.alloc(u32, nb);
        lv.used = try gpa.alloc(bool, nv);
        lv.last_start = try gpa.alloc(u32, nb + 1);
        lv.out_start = try gpa.alloc(u32, nb + 1);
        @memset(lv.def_block, none);
        @memset(lv.used, false);
        for (0..f.params) |v| {
            lv.def_block[v] = 0;
            lv.def_pos[v] = @intCast(v);
        }
        for (f.blocks.items, 0..) |b, bi| {
            const base: u32 = if (bi == 0) f.params else @intCast(b.params.len);
            lv.base[bi] = base;
            for (b.params, 0..) |p, i| {
                lv.def_block[p] = @intCast(bi);
                lv.def_pos[p] = @intCast(i);
            }
            for (b.ops.items, 0..) |o, i| if (o.def()) |d| {
                lv.def_block[d] = @intCast(bi);
                lv.def_pos[d] = base + @as(u32, @intCast(i));
            };
        }

        // Last read per (block, vreg), sorted by vreg within the block.
        var last: std.ArrayList(Use) = .empty;
        defer last.deinit(gpa);
        const slot = try gpa.alloc(u32, nv); // index in `last` of v's entry for this block
        defer gpa.free(slot);
        const stamp = try gpa.alloc(u32, nv);
        defer gpa.free(stamp);
        @memset(stamp, none);
        var scratch: std.ArrayList(ir.VReg) = .empty;
        defer scratch.deinit(gpa);
        lv.last_start[0] = 0;
        for (f.blocks.items, 0..) |b, bi| {
            const blk: u32 = @intCast(bi);
            const start = last.items.len;
            for (b.ops.items, 0..) |o, i| for (try ssa.usesOf(gpa, o, &scratch)) |u| {
                const pos = lv.base[bi] + @as(u32, @intCast(i));
                if (stamp[u] == blk) {
                    last.items[slot[u]].pos = pos;
                } else {
                    stamp[u] = blk;
                    slot[u] = @intCast(last.items.len);
                    try last.append(gpa, .{ .v = u, .pos = pos });
                }
                lv.used[u] = true;
            };
            std.mem.sort(Use, last.items[start..], {}, struct {
                fn lt(_: void, x: Use, y: Use) bool {
                    return x.v < y.v;
                }
            }.lt);
            lv.last_start[bi + 1] = @intCast(last.items.len);
        }
        lv.last = try last.toOwnedSlice(gpa);

        // Upward-exposed reads (a read in a block other than the definition's) per vreg, then
        // live-in/live-out by walking predecessors from each until the definition's block.
        const up_start = try gpa.alloc(u32, nv + 1);
        defer gpa.free(up_start);
        @memset(up_start, 0);
        for (0..nb) |b| for (lv.last[lv.last_start[b]..lv.last_start[b + 1]]) |u| {
            if (lv.def_block[u.v] == none) return error.InvalidSsa;
            if (lv.def_block[u.v] != b) up_start[u.v + 1] += 1;
        };
        for (1..nv + 1) |i| up_start[i] += up_start[i - 1];
        const up = try gpa.alloc(u32, up_start[nv]);
        defer gpa.free(up);
        {
            const fill = try gpa.dupe(u32, up_start[0..nv]);
            defer gpa.free(fill);
            for (0..nb) |b| for (lv.last[lv.last_start[b]..lv.last_start[b + 1]]) |u| if (lv.def_block[u.v] != b) {
                up[fill[u.v]] = @intCast(b);
                fill[u.v] += 1;
            };
        }
        const Pair = struct { b: u32, v: ir.VReg };
        var outs: std.ArrayList(Pair) = .empty;
        defer outs.deinit(gpa);
        const in_stamp = try gpa.alloc(u32, nb);
        defer gpa.free(in_stamp);
        const out_stamp = try gpa.alloc(u32, nb);
        defer gpa.free(out_stamp);
        @memset(in_stamp, none);
        @memset(out_stamp, none);
        var work: std.ArrayList(u32) = .empty;
        defer work.deinit(gpa);
        for (0..nv) |vi| {
            const v: ir.VReg = @intCast(vi);
            for (up[up_start[v]..up_start[v + 1]]) |b0| {
                if (in_stamp[b0] == v) continue;
                in_stamp[b0] = v;
                try work.append(gpa, b0);
                while (work.pop()) |b| for (g.preds[b]) |p| {
                    if (out_stamp[p] != v) {
                        out_stamp[p] = v;
                        try outs.append(gpa, .{ .b = p, .v = v });
                    }
                    if (p != lv.def_block[v] and in_stamp[p] != v) {
                        in_stamp[p] = v;
                        try work.append(gpa, p);
                    }
                };
            }
        }
        // Group by block (stable: ascending vregs within a block).
        @memset(lv.out_start, 0);
        for (outs.items) |o| lv.out_start[o.b + 1] += 1;
        for (1..nb + 1) |i| lv.out_start[i] += lv.out_start[i - 1];
        lv.out_vars = try gpa.alloc(ir.VReg, outs.items.len);
        const fill = try gpa.dupe(u32, lv.out_start[0..nb]);
        defer gpa.free(fill);
        for (outs.items) |o| {
            lv.out_vars[fill[o.b]] = o.v;
            fill[o.b] += 1;
        }
        return lv;
    }

    fn deinit(lv: *Liveness, gpa: std.mem.Allocator) void {
        gpa.free(lv.def_block);
        gpa.free(lv.def_pos);
        gpa.free(lv.base);
        gpa.free(lv.used);
        gpa.free(lv.out_start);
        gpa.free(lv.out_vars);
        gpa.free(lv.last_start);
        gpa.free(lv.last);
    }

    fn isEntry(lv: *const Liveness, v: ir.VReg) bool {
        return lv.def_pos[v] < lv.base[lv.def_block[v]];
    }

    /// Dominance-tree preorder of definition points.
    fn before(lv: *const Liveness, a: ir.VReg, b: ir.VReg) bool {
        const pa = lv.g.pre[lv.def_block[a]];
        const pb = lv.g.pre[lv.def_block[b]];
        return pa < pb or (pa == pb and lv.def_pos[a] < lv.def_pos[b]);
    }

    /// The definition of `a` dominates (or is) the definition of `b`.
    fn dominates(lv: *const Liveness, a: ir.VReg, b: ir.VReg) bool {
        const ba = lv.def_block[a];
        const bb = lv.def_block[b];
        if (ba == bb) return lv.def_pos[a] <= lv.def_pos[b];
        return lv.g.dominates(ba, bb);
    }

    /// `v` is live on entry to block `blk` (read there or live through it, defined elsewhere).
    fn liveIn(lv: *const Liveness, v: ir.VReg, blk: u32) bool {
        if (lv.def_block[v] == blk) return false;
        const outs = lv.out_vars[lv.out_start[blk]..lv.out_start[blk + 1]];
        if (std.sort.binarySearch(ir.VReg, outs, v, order) != null) return true;
        const uses = lv.last[lv.last_start[blk]..lv.last_start[blk + 1]];
        return std.sort.binarySearch(Use, uses, v, orderUse) != null;
    }

    /// `a` (whose definition dominates `b`'s) is live just after `b` is defined.
    fn interferes(lv: *const Liveness, a: ir.VReg, b: ir.VReg) bool {
        const blk = lv.def_block[b];
        if (lv.def_block[a] == blk and lv.isEntry(a) and lv.isEntry(b)) return true;
        const outs = lv.out_vars[lv.out_start[blk]..lv.out_start[blk + 1]];
        if (std.sort.binarySearch(ir.VReg, outs, a, order) != null) return true;
        const uses = lv.last[lv.last_start[blk]..lv.last_start[blk + 1]];
        const i = std.sort.binarySearch(Use, uses, a, orderUse) orelse return false;
        return uses[i].pos > lv.def_pos[b];
    }

    fn order(k: ir.VReg, x: ir.VReg) std.math.Order {
        return std.math.order(k, x);
    }
    fn orderUse(k: ir.VReg, x: Use) std.math.Order {
        return std.math.order(k, x.v);
    }
};

/// Congruence classes: union-find over the vregs that existed before copies were added, with
/// the members of each class in dominance order (`Liveness.before`).
const Classes = struct {
    gpa: std.mem.Allocator,
    lv: *const Liveness,
    parent: []ir.VReg,
    /// Members of a class (at its root); empty means just the root itself.
    members: []std.ArrayList(ir.VReg),
    /// The class contains a function parameter / is a constant (never merged).
    fparam: []bool,
    fixed: []bool,
    merged: std.ArrayList(ir.VReg) = .empty,
    stack: std.ArrayList(struct { v: ir.VReg, side: u1 }) = .empty,

    fn init(gpa: std.mem.Allocator, c: *opt.Ctx, lv: *const Liveness) opt.Error!Classes {
        const nv = c.f.num_vregs;
        var cls: Classes = .{
            .gpa = gpa,
            .lv = lv,
            .parent = try gpa.alloc(ir.VReg, nv),
            .members = &.{},
            .fparam = &.{},
            .fixed = &.{},
        };
        errdefer cls.deinit();
        cls.members = try gpa.alloc(std.ArrayList(ir.VReg), nv);
        @memset(cls.members, .empty);
        cls.fparam = try gpa.alloc(bool, nv);
        cls.fixed = try gpa.alloc(bool, nv);
        for (cls.parent, 0..) |*p, v| p.* = @intCast(v);
        @memset(cls.fparam, false);
        @memset(cls.fparam[0..c.f.params], true);
        @memset(cls.fixed, false);
        for (c.f.blocks.items) |b| for (b.ops.items) |o| switch (o) {
            .@"const" => |x| cls.fixed[x.dst] = true,
            .fconst => |x| cls.fixed[x.dst] = true,
            else => {},
        };
        return cls;
    }

    fn deinit(cls: *Classes) void {
        for (cls.members) |*m| m.deinit(cls.gpa);
        cls.gpa.free(cls.members);
        cls.gpa.free(cls.parent);
        cls.gpa.free(cls.fparam);
        cls.gpa.free(cls.fixed);
        cls.merged.deinit(cls.gpa);
        cls.stack.deinit(cls.gpa);
    }

    /// Class representative of `v` (vregs created after `init`, i.e. copy temporaries, are
    /// their own class).
    fn find(cls: *Classes, v: ir.VReg) ir.VReg {
        if (v >= cls.parent.len) return v;
        var r = v;
        while (cls.parent[r] != r) r = cls.parent[r];
        var x = v;
        while (cls.parent[x] != r) {
            const next = cls.parent[x];
            cls.parent[x] = r;
            x = next;
        }
        return r;
    }

    fn list(cls: *Classes, root: *const ir.VReg) []const ir.VReg {
        const m = cls.members[root.*].items;
        return if (m.len == 0) root[0..1] else m;
    }

    /// Merges the classes of `p` and `a` when allowed and interference-free.
    fn tryMerge(cls: *Classes, p: ir.VReg, a: ir.VReg) std.mem.Allocator.Error!void {
        var rp = cls.find(p);
        var ra = cls.find(a);
        if (rp == ra or cls.fixed[rp] or cls.fixed[ra] or (cls.fparam[rp] and cls.fparam[ra])) return;
        const xs = cls.list(&rp);
        const ys = cls.list(&ra);
        if (xs.len + ys.len > max_class) return;
        // Dominance-forest walk over both member lists in dominance order: each member is
        // checked against its nearest dominating member when that one is in the other class.
        const lv = cls.lv;
        cls.merged.clearRetainingCapacity();
        cls.stack.clearRetainingCapacity();
        var i: usize = 0;
        var j: usize = 0;
        while (i < xs.len or j < ys.len) {
            const take_x = j == ys.len or (i < xs.len and lv.before(xs[i], ys[j]));
            const v = if (take_x) xs[i] else ys[j];
            const side: u1 = if (take_x) 0 else 1;
            if (take_x) i += 1 else j += 1;
            while (cls.stack.items.len != 0 and !lv.dominates(cls.stack.items[cls.stack.items.len - 1].v, v)) _ = cls.stack.pop();
            if (cls.stack.items.len != 0) {
                const top = cls.stack.items[cls.stack.items.len - 1];
                if (top.side != side and lv.interferes(top.v, v)) return;
            }
            try cls.stack.append(cls.gpa, .{ .v = v, .side = side });
            try cls.merged.append(cls.gpa, v);
        }
        // Union: the root with more members stays the root.
        if (xs.len < ys.len) std.mem.swap(ir.VReg, &rp, &ra);
        cls.parent[ra] = rp;
        cls.fparam[rp] = cls.fparam[rp] or cls.fparam[ra];
        cls.members[rp].clearRetainingCapacity();
        try cls.members[rp].appendSlice(cls.gpa, cls.merged.items);
        cls.members[ra].clearAndFree(cls.gpa);
    }

    /// Some argument of edge `e` lands in another class than its (read) parameter.
    fn needsCopies(cls: *Classes, e: anytype, f: *const ir.Func) bool {
        for (f.blocks.items[e.to].params, e.args) |p, a| {
            if (cls.lv.used[p] and cls.find(p) != cls.find(a)) return true;
        }
        return false;
    }
};

/// Sequentializes a parallel copy `dst_i = src_i` (distinct destinations, `dst != src`;
/// after Boissinot et al., Algorithm 1): copies into a location no pending copy reads go
/// first; after `d = s`, `d` also holds the value of `s`, so the location of `s` is free for
/// its own copy (later readers of `s` then read `d`; while `s` is intact they read `s`, which
/// keeps copies from constants direct). When only cycles remain, one destination is saved to
/// a new temporary (`Ctx.newVReg`): one per cycle.
const ParallelCopy = struct {
    copies: std.ArrayList(Copy) = .empty,
    out: std.ArrayList(ir.Op) = .empty,
    ready: std.ArrayList(u32) = .empty,
    /// Per vreg (meaningful for the current copy's operands only): is a source, the copy
    /// writing it, where the value it held before the copy is now, a destination already
    /// holding that value (or `none`).
    is_src: []bool,
    writer: []u32,
    loc: []ir.VReg,
    alt: []ir.VReg,

    const Copy = struct { dst: ir.VReg, src: ir.VReg, done: bool = false };

    fn init(gpa: std.mem.Allocator, nv: u32) std.mem.Allocator.Error!ParallelCopy {
        const is_src = try gpa.alloc(bool, nv);
        errdefer gpa.free(is_src);
        const writer = try gpa.alloc(u32, nv);
        errdefer gpa.free(writer);
        const loc = try gpa.alloc(ir.VReg, nv);
        errdefer gpa.free(loc);
        const alt = try gpa.alloc(ir.VReg, nv);
        @memset(is_src, false);
        @memset(writer, none);
        return .{ .is_src = is_src, .writer = writer, .loc = loc, .alt = alt };
    }
    fn deinit(pc: *ParallelCopy, gpa: std.mem.Allocator) void {
        pc.copies.deinit(gpa);
        pc.out.deinit(gpa);
        pc.ready.deinit(gpa);
        gpa.free(pc.is_src);
        gpa.free(pc.writer);
        gpa.free(pc.loc);
        gpa.free(pc.alt);
    }
    fn clear(pc: *ParallelCopy) void {
        for (pc.copies.items) |x| {
            pc.is_src[x.src] = false;
            pc.writer[x.dst] = none;
        }
        pc.copies.clearRetainingCapacity();
    }
    fn add(pc: *ParallelCopy, gpa: std.mem.Allocator, dst: ir.VReg, src: ir.VReg) std.mem.Allocator.Error!void {
        std.debug.assert(dst != src and pc.writer[dst] == none);
        pc.writer[dst] = @intCast(pc.copies.items.len);
        pc.is_src[src] = true;
        pc.loc[src] = src;
        pc.alt[src] = none;
        try pc.copies.append(gpa, .{ .dst = dst, .src = src });
    }

    /// The copy ops (valid until the next call).
    fn sequentialize(pc: *ParallelCopy, c: *opt.Ctx) opt.Error![]const ir.Op {
        const gpa = c.gpa;
        pc.out.clearRetainingCapacity();
        pc.ready.clearRetainingCapacity();
        for (pc.copies.items, 0..) |x, i| if (!pc.is_src[x.dst]) try pc.ready.append(gpa, @intCast(i));
        var left = pc.copies.items.len;
        var next_pending: usize = 0;
        while (true) {
            while (pc.ready.pop()) |i| {
                const x = &pc.copies.items[i];
                x.done = true;
                left -= 1;
                // x.dst is about to change: its old value lives on in `alt` (or a temporary).
                if (pc.is_src[x.dst] and pc.loc[x.dst] == x.dst) pc.loc[x.dst] = pc.alt[x.dst];
                const from = pc.loc[x.src];
                try pc.out.append(gpa, .{ .copy = .{ .dst = x.dst, .src = from } });
                if (from == x.src and pc.alt[x.src] == none) {
                    pc.alt[x.src] = x.dst; // final: every location is written once
                    const w = pc.writer[x.src];
                    if (w != none and !pc.copies.items[w].done) try pc.ready.append(gpa, w);
                }
            }
            if (left == 0) break;
            // Only cycles are left: save one destination and let its copy go.
            while (pc.copies.items[next_pending].done) next_pending += 1;
            const x = pc.copies.items[next_pending];
            const tmp = try c.newVReg(c.types.items[x.dst]);
            try pc.out.append(gpa, .{ .copy = .{ .dst = tmp, .src = x.dst } });
            pc.loc[x.dst] = tmp;
            try pc.ready.append(gpa, @intCast(next_pending));
        }
        return pc.out.items;
    }
};

/// Renames every vreg to its class and renumbers densely: function parameters keep their
/// numbers, the other classes are numbered in order of first appearance. Drops the copies
/// that became `v = copy v`.
fn renumber(c: *opt.Ctx, cls: *Classes) opt.Error!void {
    const gpa = c.gpa;
    const f = c.f;
    const nv = f.num_vregs;
    const num = try gpa.alloc(ir.VReg, nv); // new number per class root
    defer gpa.free(num);
    @memset(num, none);
    var next: ir.VReg = f.params;
    for (0..f.params) |v| num[cls.find(@intCast(v))] = @intCast(v);
    const map = try gpa.alloc(ir.VReg, nv);
    defer gpa.free(map);
    const types = try gpa.alloc(ir.Type, @max(nv, f.params, 1));
    defer gpa.free(types);
    for (0..f.params) |v| types[v] = .i64;
    var buf: [ir.max_args]ir.VReg = undefined; // no edge arguments are left
    for (f.blocks.items) |b| for (b.ops.items) |o| {
        const us = o.uses(&buf);
        for (0..us.len + 1) |k| {
            const v = if (k < us.len) us[k] else o.def() orelse break;
            const r = cls.find(v);
            if (num[r] == none) {
                num[r] = next;
                next += 1;
            }
            map[v] = num[r];
            types[num[r]] = c.types.items[v]; // every member of a class has the same type
        }
    };
    for (f.blocks.items) |*b| {
        var k: usize = 0;
        for (b.ops.items) |*o| {
            try opt.mapUses(o, c.irAlloc(), map);
            if (o.def()) |d| ssa.setDef(o, map[d]);
            if (o.* == .copy and o.copy.dst == o.copy.src) continue;
            b.ops.items[k] = o.*;
            k += 1;
        }
        b.ops.items.len = k;
    }
    f.num_vregs = next;
    c.types.clearRetainingCapacity();
    try c.types.appendSlice(gpa, types[0..next]);
}

// ---- tests -------------------------------------------------------------------------------

const t = std.testing;
const testgen = @import("testgen.zig");

/// Parses `src`, builds SSA, runs `pre` and `run` (all checked), compares the result with the
/// original on `testgen.default_inputs` and returns it (caller deinits).
fn lower(src: []const u8, comptime pre: []const testgen.Pass, profile: ?*const opt.Profile) !ir.Func {
    var ref = try ir.parse(t.allocator, src);
    defer ref.deinit();
    var f = try ir.parse(t.allocator, src);
    errdefer f.deinit();
    const options: opt.Options = .{ .verify_each = true, .profile = profile };
    var stats: opt.Stats = .{};
    var c = try opt.Ctx.init(t.allocator, &f, &options, &stats);
    defer c.deinit();
    try ssa.build(&c);
    try c.check("ssa.build");
    inline for (pre) |p| {
        _ = try p(&c);
        try c.check("pre");
    }
    try run(&c);
    try c.check("out_of_ssa");
    try t.expect(!c.in_ssa and !f.hasBlockParams());
    try t.expectEqual(c.f.num_vregs, @as(u32, @intCast(c.types.items.len)));
    if (std.c.getenv("PJIT_DUMP") != null) {
        var out: std.ArrayList(u8) = .empty;
        defer out.deinit(t.allocator);
        try f.print(t.allocator, &out);
        std.debug.print("{s}\n", .{out.items});
    }
    try testgen.expectSame(&ref, &f, &testgen.default_inputs, 20_000);
    return f;
}

fn blockNamed(f: *const ir.Func, name: []const u8) u32 {
    for (f.blocks.items, 0..) |b, i| if (std.mem.eql(u8, b.name, name)) return @intCast(i);
    unreachable;
}

fn copiesIn(f: *const ir.Func, b: u32) usize {
    var n: usize = 0;
    for (f.blocks.items[b].ops.items) |o| n += @intFromBool(o == .copy);
    return n;
}

fn countCopies(f: *const ir.Func) usize {
    var n: usize = 0;
    for (0..f.blocks.items.len) |b| n += copiesIn(f, @intCast(b));
    return n;
}

/// Copies the simple out-of-SSA translation would emit: two per edge argument.
fn simpleCopies(f: *const ir.Func) usize {
    var n: usize = 0;
    for (f.blocks.items) |b| {
        const term = b.ops.items[b.ops.items.len - 1];
        for (0..ssa.numSuccs(term)) |i| n += 2 * cfg.edgeOf(term, @intCast(i)).args.len;
    }
    return n;
}

test "out_of_ssa: counted loop updates its induction variable and sum in place" {
    var f = try lower(
        \\func sum 1
        \\e:
        \\  v1 = const 0
        \\  v2 = const 1
        \\  br h v1 v1
        \\h(v3 v4):
        \\  cbr lt v3 v0 b x
        \\b:
        \\  v5 = add v4 v3
        \\  v6 = add v3 v2
        \\  br h v6 v5
        \\x:
        \\  ret v4
    , &.{}, null);
    defer f.deinit();
    try t.expectEqual(@as(usize, 2), countCopies(&f)); // i = 0, s = 0 on the entry edge
    try t.expectEqual(@as(usize, 2), copiesIn(&f, 0));
    for (f.blocks.items[blockNamed(&f, "b")].ops.items) |o| switch (o) {
        .arith => |x| try t.expect(x.dst == x.a), // both updates in place
        .br => {},
        else => return error.TestUnexpectedResult,
    };
    // Constants keep a single definition (the IR Compilers fold them into immediates).
    var defs: [16]u32 = @splat(0);
    for (f.blocks.items) |b| for (b.ops.items) |o| if (o.def()) |d| {
        defs[d] += 1;
    };
    for (f.blocks.items[0].ops.items) |o| if (o == .@"const") try t.expectEqual(@as(u32, 1), defs[o.@"const".dst]);
    try t.expectEqual(@as(u32, 5), f.num_vregs); // v0, two consts, i, s
}

test "out_of_ssa: unchanged loop-carried values and function parameters need no copy" {
    var f = try lower(
        \\func cnt 2
        \\e:
        \\  br h v0 v1 v1
        \\h(v2 v3 v8):
        \\  v4 = const 1
        \\  v6 = add v3 v2
        \\  v5 = sub v2 v4
        \\  v9 = xor v8 v6
        \\  cbr gt v5 v4 h:v5:v6:v8 x:v9
        \\x(v7):
        \\  ret v7
    , &.{}, null);
    defer f.deinit();
    // v2 joins v0 and v3 joins v1; v8 (an unchanged copy of v1, which v3 overwrites) needs
    // the one entry copy; the loop itself has none.
    try t.expectEqual(@as(usize, 1), countCopies(&f));
    try t.expectEqual(@as(usize, 1), copiesIn(&f, 0));
    try t.expectEqual(@as(usize, 3), f.blocks.items.len); // no edge block
}

test "out_of_ssa: swap problem" {
    var f = try lower(
        \\func swap 2
        \\e:
        \\  v2 = const 0
        \\  br h v0 v1 v2
        \\h(v3 v4 v5):
        \\  v6 = const 3
        \\  cbr lt v5 v6 b x
        \\b:
        \\  v7 = const 1
        \\  v8 = add v5 v7
        \\  br h v4 v3 v8
        \\x:
        \\  v9 = const 3
        \\  v10 = mul v3 v9
        \\  v11 = add v10 v4
        \\  ret v11
    , &.{}, null);
    defer f.deinit();
    try t.expectEqual(@as(usize, 3), copiesIn(&f, blockNamed(&f, "b"))); // one temporary
}

test "out_of_ssa: lost-copy problem, with the back edge split and placed after the latch" {
    var f = try lower(
        \\func lost 1
        \\e:
        \\  v1 = const 1
        \\  br h v1
        \\h(v2):
        \\  v3 = add v2 v1
        \\  cbr lt v3 v0 h:v3 x
        \\x:
        \\  ret v2
    , &.{}, null);
    defer f.deinit();
    try t.expectEqual(@as(usize, 4), f.blocks.items.len);
    try t.expect(std.mem.startsWith(u8, f.blocks.items[2].name, "edge")); // h, edge, x
    try t.expectEqual(@as(usize, 1), copiesIn(&f, 2));
    var g = try lower(
        \\func lost2 1
        \\e:
        \\  v1 = const 1
        \\  br h v1
        \\h(v2):
        \\  v3 = add v2 v1
        \\  cbr lt v3 v0 l x
        \\l:
        \\  br h v3
        \\x:
        \\  ret v2
    , &.{}, null);
    defer g.deinit();
    try t.expectEqual(@as(usize, 1), copiesIn(&g, blockNamed(&g, "l")));
}

test "out_of_ssa: critical edges with arguments, same target twice" {
    var f = try lower(
        \\func crit 2
        \\e:
        \\  cbr lt v0 v1 a:v0 b:v1
        \\a(v2):
        \\  v4 = const 2
        \\  v5 = mul v2 v4
        \\  cbr eq v5 v1 m:v5:v2 m:v2:v5
        \\b(v3):
        \\  cbr ult v3 v0 m:v3:v0 a:v0
        \\m(v6 v7):
        \\  v8 = sub v6 v7
        \\  ret v8
    , &.{}, null);
    defer f.deinit();
    var g = try lower(
        \\func twice 2
        \\e:
        \\  v2 = const 9
        \\  cbr lt v0 v1 m:v0:v1 m:v1:v2
        \\m(v6 v7):
        \\  v8 = sub v6 v7
        \\  ret v8
    , &.{}, null);
    defer g.deinit();
}

test "out_of_ssa: parallel copy cycles of length 1 to 4, two cycles, fan-out" {
    const Case = struct { perm: []const u8, copies: usize };
    for ([_]Case{
        .{ .perm = "v3 v4 v5 v6", .copies = 0 }, // identity
        .{ .perm = "v4 v3 v5 v6", .copies = 3 }, // swap
        .{ .perm = "v4 v5 v3 v6", .copies = 4 }, // 3-cycle
        .{ .perm = "v4 v5 v6 v3", .copies = 5 }, // 4-cycle
        .{ .perm = "v4 v3 v6 v5", .copies = 6 }, // two swaps
        .{ .perm = "v4 v4 v3 v3", .copies = 3 }, // fan-out (a <- b, c <- a, d <- a; b stays)
        .{ .perm = "v5 v3 v3 v4", .copies = 4 }, // a 2-cycle broken through a fan-out copy
        .{ .perm = "v6 v6 v6 v6", .copies = 3 }, // one source for all
    }) |cs| {
        const src = try std.fmt.allocPrint(t.allocator,
            \\func perm 1
            \\e:
            \\  v1 = const 10
            \\  v2 = const 20
            \\  v7 = const 30
            \\  v8 = const 40
            \\  v9 = const 0
            \\  br h v1 v2 v7 v8 v9
            \\h(v3 v4 v5 v6 v10):
            \\  v11 = const 3
            \\  cbr lt v10 v11 b x
            \\b:
            \\  v12 = const 1
            \\  v13 = add v10 v12
            \\  br h {s} v13
            \\x:
            \\  v14 = const 1000
            \\  v15 = mul v3 v14
            \\  v15 = add v15 v4
            \\  v15 = mul v15 v14
            \\  v15 = add v15 v5
            \\  v15 = mul v15 v14
            \\  v15 = add v15 v6
            \\  ret v15
        , .{cs.perm});
        defer t.allocator.free(src);
        var f = try lower(src, &.{}, null);
        defer f.deinit();
        try t.expectEqual(cs.copies, copiesIn(&f, blockNamed(&f, "b")));
    }
}

test "out_of_ssa: the hotter edge gets the coalesced value (profile), the colder the copy" {
    const src =
        \\func hot 2
        \\e:
        \\  v2 = const 1
        \\  br h v0
        \\h(v3):
        \\  v4 = add v3 v2
        \\  v5 = mul v4 v4
        \\  cbr lt v4 v1 a b
        \\a:
        \\  br h v4
        \\b:
        \\  cbr lt v5 v1 h:v5 x
        \\x:
        \\  ret v4
    ;
    for ([_]u64{ 90, 10 }) |taken| {
        const prof: opt.Profile = .{ .blocks = &.{ 1, 100, taken, 100 - taken, 1 }, .taken = &.{ 0, taken, 0, 100 - taken - 1, 0 } };
        var f = try lower(src, &.{}, &prof);
        defer f.deinit();
        const in_a = copiesIn(&f, blockNamed(&f, "a"));
        try t.expectEqual(@as(usize, if (taken == 90) 0 else 1), in_a);
        try t.expectEqual(@as(usize, if (taken == 90) 4 else 3), f.blocks.items.len - @as(usize, 2)); // + edge block when b's edge copies
    }
}

test "out_of_ssa: block parameters of every type, after dce" {
    const dce = @import("dce.zig");
    var f = try lower(
        \\func ty 2
        \\e:
        \\  v2 = fconst.f64 0.5
        \\  v3 = fconst.f32 1.5
        \\  v4 = splat.v2f64 v2
        \\  v5 = const 0
        \\  br h v2 v3 v4 v5 v5
        \\h(v6 v7 v8 v9 v15):
        \\  v10 = add.f64 v6 v6
        \\  v11 = cvt.f64.f32 v7
        \\  v12 = splat.v2f64 v11
        \\  v13 = add.v2f64 v8 v12
        \\  v14 = const 1
        \\  v16 = add v9 v14
        \\  v17 = mul v15 v15
        \\  cbr lt v16 v1 h:v10:v7:v13:v16:v17 x
        \\x:
        \\  v18 = extract.v2f64 v8 1
        \\  v19 = add.f64 v18 v6
        \\  v20 = cvt.i64.f64 v19
        \\  ret v20
    , &.{dce.run}, null);
    defer f.deinit();
    for (f.blocks.items[0].ops.items) |o| try t.expect(o != .arith); // v17 is gone
}

test "out_of_ssa: a loop guard's copies go before its branch (no edge blocks)" {
    // Both edges of the guard pass the initial sum; the loop edge also the initial counter.
    var f = try lower(
        \\func f 2
        \\e:
        \\  v2 = const 0
        \\  v3 = const 0
        \\  v4 = const 1
        \\  cbr ge v2 v0 x l
        \\l:
        \\  v3 = add v3 v2
        \\  v2 = add v2 v4
        \\  cbr lt v2 v0 l x
        \\x:
        \\  ret v3
    , &.{}, null);
    defer f.deinit();
    try t.expectEqual(@as(usize, 3), f.blocks.items.len);
    try t.expectEqual(@as(usize, 2), copiesIn(&f, 0));
    try t.expectEqual(@as(usize, 2), countCopies(&f));
}

test "out_of_ssa: copies stay on their edge when the other successor reads the class" {
    // `v4` and `v2` share a class (edge b -> h, hotter by the profile); the copy `v4 = v3` on
    // a -> h would clobber `v2`, which `y` reads, if it ran before the branch in `a`.
    const prof: opt.Profile = .{ .blocks = &.{ 100, 10, 90, 5, 95 }, .taken = &.{ 10, 5, 0, 0, 0 } };
    var f = try lower(
        \\func f 2
        \\e:
        \\  v2 = add v0 v1
        \\  v3 = mul v0 v1
        \\  cbr lt v0 v1 a b
        \\a:
        \\  cbr lt v1 v0 h:v3 y
        \\b:
        \\  br h v2
        \\y:
        \\  ret v2
        \\h(v4):
        \\  v5 = add v4 v4
        \\  ret v5
    , &.{}, &prof);
    defer f.deinit();
    try t.expectEqual(@as(usize, 1), countCopies(&f));
    try t.expectEqual(@as(usize, 0), copiesIn(&f, blockNamed(&f, "a")));
    try t.expectEqual(@as(usize, 0), copiesIn(&f, blockNamed(&f, "b")));
    try t.expectEqual(@as(usize, 6), f.blocks.items.len); // a -> h split
}

test "out_of_ssa: copies stay on their edge when the other edge passes the class unchanged" {
    // `v2`, `v4` and `v5` share a class; in `c`, the edge to `hj` passes it as is, so the copy
    // `v4 = v3` of the edge to `hi` must not run before the branch.
    const prof: opt.Profile = .{ .blocks = &.{ 100, 10, 90, 5, 95 }, .taken = &.{ 10, 5, 0, 0, 0 } };
    var f = try lower(
        \\func f 2
        \\e:
        \\  v2 = add v0 v1
        \\  v3 = mul v0 v1
        \\  cbr lt v0 v1 c b
        \\c:
        \\  cbr lt v1 v0 hi:v3 hj:v2
        \\b:
        \\  br hi v2
        \\hj(v5):
        \\  ret v5
        \\hi(v4):
        \\  v6 = add v4 v4
        \\  ret v6
    , &.{}, &prof);
    defer f.deinit();
    try t.expectEqual(@as(usize, 1), countCopies(&f));
    try t.expectEqual(@as(usize, 0), copiesIn(&f, blockNamed(&f, "c")));
}

test "out_of_ssa: random programs, copies versus the simple translation" {
    const gpa = t.allocator;
    const dce = @import("dce.zig");
    var prng = std.Random.DefaultPrng.init(0x0055a_c0a1);
    var src: std.ArrayList(u8) = .empty;
    defer src.deinit(gpa);
    var simple: usize = 0;
    var coalesced: usize = 0;
    var edge_blocks: usize = 0;
    const configs = [_]testgen.Config{ .{}, .{ .blocks = 10, .ops = 10 }, .{ .loops = false }, .{ .calls = false, .floats = false, .ops = 12 } };
    for (0..400) |i| {
        src.clearRetainingCapacity();
        try testgen.program(prng.random(), gpa, configs[i % configs.len], &src);
        inline for (.{ &[_]testgen.Pass{}, &[_]testgen.Pass{dce.run} }) |pre| {
            var f = lower(src.items, pre, null) catch |e| {
                std.debug.print("out_of_ssa failure on:\n{s}\n", .{src.items});
                return e;
            };
            defer f.deinit();
            if (pre.len == 1) {
                // Edge arguments in SSA form (what the simple version copies twice each).
                var g = try ir.parse(gpa, src.items);
                defer g.deinit();
                const options: opt.Options = .{};
                var stats: opt.Stats = .{};
                var c = try opt.Ctx.init(gpa, &g, &options, &stats);
                defer c.deinit();
                try ssa.build(&c);
                _ = try dce.run(&c);
                simple += simpleCopies(&g);
                coalesced += countCopies(&f);
                edge_blocks += f.blocks.items.len - g.blocks.items.len;
            }
        }
    }
    if (std.c.getenv("PJIT_OPT_STATS") != null) std.debug.print("out_of_ssa copies: simple {d}, coalesced {d}; edge blocks {d}\n", .{ simple, coalesced, edge_blocks });
    try t.expect(coalesced * 3 <= simple);
}

/// Test pre-pass: replaces every use of a `copy` result with the copy's source (SSA copy
/// propagation), so edges pass permutations of block parameters (swap/lost-copy shapes).
fn copyProp(c: *opt.Ctx) opt.Error!bool {
    const map = try c.gpa.alloc(ir.VReg, c.f.num_vregs);
    defer c.gpa.free(map);
    for (map, 0..) |*m, v| m.* = @intCast(v);
    var any = false;
    for (c.f.blocks.items) |b| for (b.ops.items) |o| if (o == .copy) {
        map[o.copy.dst] = o.copy.src;
        any = true;
    };
    for (map) |*m| while (map[m.*] != m.*) {
        m.* = map[m.*];
    };
    try opt.renameUses(c, map);
    return any;
}

/// Random mutable programs dominated by permutations of the live values (swaps, rotations
/// through temporaries) around loops, with i64 and f64 values: v2..v6 i64, v7/v8 f64, v9
/// step counter, v10 scratch.
fn permProgram(rng: std.Random, out: *std.ArrayList(u8)) !void {
    const gpa = t.allocator;
    const n = rng.intRangeAtMost(u32, 1, 7);
    try out.print(gpa, "func p 2\n", .{});
    for (0..n) |b| {
        try out.print(gpa, "b{d}:\n  v10 = const 1\n  v9 = add v9 v10\n", .{b});
        if (b == 0) try out.print(gpa, "  v7 = fconst.f64 0.5\n  v8 = cvt.f64.i64 v0\n", .{});
        for (0..rng.uintAtMost(u32, 5)) |_| switch (rng.uintLessThan(u32, 8)) {
            0, 1, 2 => { // rotate k of the i64 values through a temporary
                var vs = [_]u32{ 2, 3, 4, 5, 6 };
                rng.shuffle(u32, &vs);
                const k = rng.intRangeAtMost(usize, 2, 5);
                try out.print(gpa, "  v10 = copy v{d}\n", .{vs[0]});
                for (0..k - 1) |i| try out.print(gpa, "  v{d} = copy v{d}\n", .{ vs[i], vs[i + 1] });
                try out.print(gpa, "  v{d} = copy v10\n", .{vs[k - 1]});
            },
            3 => try out.print(gpa, "  v11 = copy v7\n  v7 = copy v8\n  v8 = copy v11\n", .{}),
            4 => try out.print(gpa, "  v{d} = copy v{d}\n", .{ rng.intRangeAtMost(u32, 2, 6), rng.intRangeAtMost(u32, 0, 6) }),
            5 => try out.print(gpa, "  v{d} = {s} v{d} v{d}\n", .{ rng.intRangeAtMost(u32, 2, 6), @tagName(rng.enumValue(enum { add, sub, mul, xor })), rng.intRangeAtMost(u32, 0, 6), rng.intRangeAtMost(u32, 0, 6) }),
            6 => try out.print(gpa, "  v{d} = const {d}\n", .{ rng.intRangeAtMost(u32, 2, 6), rng.intRangeAtMost(i64, -3, 3) }),
            else => try out.print(gpa, "  v7 = add.f64 v7 v8\n", .{}),
        };
        const t1 = rng.uintLessThan(u32, n);
        const t2 = rng.uintLessThan(u32, n);
        switch (rng.uintLessThan(u32, 4)) {
            0 => try out.print(gpa, "  v12 = cvt.i64.f64 v{d}\n  v12 = mul v12 v{d}\n  v12 = add v12 v{d}\n  ret v12\n", .{ rng.intRangeAtMost(u32, 7, 8), rng.intRangeAtMost(u32, 2, 6), rng.intRangeAtMost(u32, 2, 6) }),
            1 => try out.print(gpa, "  br b{d}\n", .{t1}),
            2 => try out.print(gpa, "  v10 = const {d}\n  cbr lt v9 v10 b{d} b{d}\n", .{ rng.intRangeAtMost(i64, 1, 12), t1, t2 }),
            else => try out.print(gpa, "  cbr {s} v{d} v{d} b{d} b{d}\n", .{ @tagName(rng.enumValue(ir.CondCode)), rng.intRangeAtMost(u32, 0, 6), rng.intRangeAtMost(u32, 0, 6), t1, t2 }),
        }
    }
}

test "out_of_ssa: random permutation-heavy programs after copy propagation" {
    const dce = @import("dce.zig");
    var prng = std.Random.DefaultPrng.init(0x9e7_3a7e);
    var src: std.ArrayList(u8) = .empty;
    defer src.deinit(t.allocator);
    for (0..500) |_| {
        src.clearRetainingCapacity();
        try permProgram(prng.random(), &src);
        var f = lower(src.items, &.{ copyProp, dce.run }, null) catch |e| {
            std.debug.print("out_of_ssa failure on:\n{s}\n", .{src.items});
            return e;
        };
        f.deinit();
    }
}

test "out_of_ssa: random programs with random profiles" {
    const dce = @import("dce.zig");
    var prng = std.Random.DefaultPrng.init(0x9e7_f00d);
    const rng = prng.random();
    var src: std.ArrayList(u8) = .empty;
    defer src.deinit(t.allocator);
    var blocks: [16]u64 = undefined;
    var taken: [16]u64 = undefined;
    for (0..300) |i| {
        src.clearRetainingCapacity();
        if (i % 2 == 0) try permProgram(rng, &src) else try testgen.program(rng, t.allocator, .{ .blocks = 8 }, &src);
        for (&blocks, &taken) |*b, *k| {
            b.* = rng.uintAtMost(u64, 1000);
            k.* = rng.uintAtMost(u64, 1100); // sometimes more than the block count (racy)
        }
        const prof: opt.Profile = .{ .blocks = &blocks, .taken = &taken };
        var f = lower(src.items, &.{ copyProp, dce.run }, &prof) catch |e| {
            std.debug.print("out_of_ssa failure on:\n{s}\n", .{src.items});
            return e;
        };
        f.deinit();
    }
}
