//! Control-flow analysis over `ir.Func` for the optimizer: successors and predecessors,
//! reverse postorder, the dominator tree (Cooper-Harvey-Kennedy), dominance frontiers,
//! natural loops, and the two CFG edits the passes share (edge splitting, loop preheaders).
//!
//! `Cfg` and `Loops` are snapshots: any edit of blocks or terminators invalidates them
//! (recompute after editing). Both work on any function whose blocks end in a terminator;
//! unreachable blocks are tolerated (they have no RPO position and no dominator).
const std = @import("std");
const ir = @import("../ir.zig");
const opt = @import("opt.zig");

/// "No block / no loop" marker in the index arrays below.
pub const none: u32 = std.math.maxInt(u32);

/// Successors of block `b` in terminator order: `br`/`bra` give 1, `cbr`/`cbra` give 2
/// (then, else; both even when equal), `ret` gives 0. The slice points into `buf`.
pub fn succs(f: *const ir.Func, b: u32, buf: *[2]u32) []const u32 {
    const ops = f.blocks.items[b].ops.items;
    if (ops.len == 0) return buf[0..0];
    switch (ops[ops.len - 1]) {
        .br => |to| buf[0] = to,
        .bra => |x| buf[0] = x.target,
        .cbr => |x| {
            buf[0] = x.then;
            buf[1] = x.@"else";
            return buf[0..2];
        },
        .cbra => |x| {
            buf[0] = x.then;
            buf[1] = x.@"else";
            return buf[0..2];
        },
        else => return buf[0..0],
    }
    return buf[0..1];
}

/// A CFG edge: successor `index` (0 = then / only, 1 = else) of block `from`, entering `to`.
pub const Edge = struct { from: u32, to: u32, index: u1 };

pub const Cfg = struct {
    arena: std.heap.ArenaAllocator,
    /// Incoming edges of each block as source blocks, one entry per edge (a `cbr` whose two
    /// targets are equal contributes two), sorted ascending. Unreachable sources included.
    preds: []const []const u32,
    /// Reachable blocks in reverse postorder of a DFS that visits successors in terminator
    /// order; `rpo[0] == 0`.
    rpo: []const u32,
    /// Position of each block in `rpo`; `none` if unreachable.
    rpo_index: []const u32,
    /// Immediate dominator; `idom[0] == 0`; `none` for unreachable blocks.
    idom: []const u32,
    /// Children of each block in the dominator tree, sorted by `rpo_index`.
    dom_children: []const []const u32,
    /// Dominator-tree preorder / postorder numbers (`dominates` is O(1)); `none` if unreachable.
    pre: []const u32,
    post: []const u32,
    /// Lazily computed by `frontiers`.
    df: ?[]const []const u32 = null,

    pub fn init(gpa: std.mem.Allocator, f: *const ir.Func) std.mem.Allocator.Error!Cfg {
        var arena = std.heap.ArenaAllocator.init(gpa);
        errdefer arena.deinit();
        const a = arena.allocator();
        const n: u32 = @intCast(f.blocks.items.len);
        var buf: [2]u32 = undefined;

        // Predecessors: count, then fill one flat buffer (sources ascending by construction).
        const npred = try gpa.alloc(u32, n + 1);
        defer gpa.free(npred);
        @memset(npred, 0);
        for (0..n) |b| for (succs(f, @intCast(b), &buf)) |s| {
            npred[s + 1] += 1;
        };
        for (1..n + 1) |i| npred[i] += npred[i - 1];
        const flat = try a.alloc(u32, npred[n]);
        const fill = try gpa.dupe(u32, npred[0..n]);
        defer gpa.free(fill);
        for (0..n) |b| for (succs(f, @intCast(b), &buf)) |s| {
            flat[fill[s]] = @intCast(b);
            fill[s] += 1;
        };
        const preds = try a.alloc([]const u32, n);
        for (preds, 0..) |*p, i| p.* = flat[npred[i]..npred[i + 1]];

        // Reverse postorder by an iterative DFS from block 0.
        const rpo_index = try a.alloc(u32, n);
        @memset(rpo_index, none);
        const post_order = try gpa.alloc(u32, n);
        defer gpa.free(post_order);
        var npost: u32 = 0;
        if (n > 0) {
            const Frame = struct { b: u32, next: u8 };
            var stack: std.ArrayList(Frame) = .empty;
            defer stack.deinit(gpa);
            const seen = try gpa.alloc(bool, n);
            defer gpa.free(seen);
            @memset(seen, false);
            seen[0] = true;
            try stack.append(gpa, .{ .b = 0, .next = 0 });
            while (stack.items.len > 0) {
                const top = &stack.items[stack.items.len - 1];
                const ss = succs(f, top.b, &buf);
                if (top.next < ss.len) {
                    const s = ss[top.next];
                    top.next += 1;
                    if (!seen[s]) {
                        seen[s] = true;
                        try stack.append(gpa, .{ .b = s, .next = 0 });
                    }
                } else {
                    post_order[npost] = top.b;
                    npost += 1;
                    _ = stack.pop();
                }
            }
        }
        const rpo = try a.alloc(u32, npost);
        for (rpo, 0..) |*r, i| {
            r.* = post_order[npost - 1 - i];
            rpo_index[r.*] = @intCast(i);
        }

        // Immediate dominators (Cooper, Harvey, Kennedy: "A Simple, Fast Dominance Algorithm").
        const idom = try a.alloc(u32, n);
        @memset(idom, none);
        if (n > 0) idom[0] = 0;
        var changed = npost > 1;
        while (changed) {
            changed = false;
            for (rpo[1..]) |b| {
                var new: u32 = none;
                for (preds[b]) |p| {
                    if (idom[p] == none) continue; // unreachable or not processed yet
                    new = if (new == none) p else intersect(idom, rpo_index, p, new);
                }
                if (idom[b] != new) {
                    idom[b] = new;
                    changed = true;
                }
            }
        }

        // Dominator tree children (in RPO order) and pre/post numbers.
        const nchild = try gpa.alloc(u32, n + 1);
        defer gpa.free(nchild);
        @memset(nchild, 0);
        for (rpo[@min(1, rpo.len)..]) |b| nchild[idom[b] + 1] += 1;
        for (1..n + 1) |i| nchild[i] += nchild[i - 1];
        const cflat = try a.alloc(u32, nchild[n]);
        const cfill = try gpa.dupe(u32, nchild[0..n]);
        defer gpa.free(cfill);
        for (rpo[@min(1, rpo.len)..]) |b| {
            cflat[cfill[idom[b]]] = b;
            cfill[idom[b]] += 1;
        }
        const children = try a.alloc([]const u32, n);
        for (children, 0..) |*ch, i| ch.* = cflat[nchild[i]..nchild[i + 1]];

        const pre = try a.alloc(u32, n);
        const post = try a.alloc(u32, n);
        @memset(pre, none);
        @memset(post, none);
        if (npost > 0) {
            const Frame = struct { b: u32, next: u32 };
            var stack: std.ArrayList(Frame) = .empty;
            defer stack.deinit(gpa);
            var npre: u32 = 0;
            var npo: u32 = 0;
            pre[0] = npre;
            npre += 1;
            try stack.append(gpa, .{ .b = 0, .next = 0 });
            while (stack.items.len > 0) {
                const top = &stack.items[stack.items.len - 1];
                const ch = children[top.b];
                if (top.next < ch.len) {
                    const c = ch[top.next];
                    top.next += 1;
                    pre[c] = npre;
                    npre += 1;
                    try stack.append(gpa, .{ .b = c, .next = 0 });
                } else {
                    post[top.b] = npo;
                    npo += 1;
                    _ = stack.pop();
                }
            }
        }

        return .{
            .arena = arena,
            .preds = preds,
            .rpo = rpo,
            .rpo_index = rpo_index,
            .idom = idom,
            .dom_children = children,
            .pre = pre,
            .post = post,
        };
    }

    fn intersect(idom: []const u32, rpo_index: []const u32, x: u32, y: u32) u32 {
        var b1 = x;
        var b2 = y;
        while (b1 != b2) {
            while (rpo_index[b1] > rpo_index[b2]) b1 = idom[b1];
            while (rpo_index[b2] > rpo_index[b1]) b2 = idom[b2];
        }
        return b1;
    }

    pub fn deinit(c: *Cfg) void {
        c.arena.deinit();
    }
    pub fn reachable(c: *const Cfg, b: u32) bool {
        return c.rpo_index[b] != none;
    }
    /// `a` dominates `b` (reflexive); false when either is unreachable.
    pub fn dominates(c: *const Cfg, a: u32, b: u32) bool {
        if (c.pre[a] == none or c.pre[b] == none) return false;
        return c.pre[a] <= c.pre[b] and c.post[b] <= c.post[a];
    }
    /// Dominance frontier of every block (empty for unreachable ones), sorted ascending. The
    /// entry block counts as a join of its predecessors and the function entry, so it is in the
    /// frontier of every block on a cycle through it. Allocated in the Cfg's arena on first use.
    pub fn frontiers(c: *Cfg) std.mem.Allocator.Error![]const []const u32 {
        if (c.df) |d| return d;
        const a = c.arena.allocator();
        const n = c.rpo_index.len;
        const lists = try a.alloc(std.ArrayList(u32), n);
        for (lists) |*l| l.* = .empty;
        for (c.rpo) |b| {
            var live: u32 = 0;
            for (c.preds[b]) |p| live += @intFromBool(c.reachable(p));
            if (live == 0 or (b != 0 and live < 2)) continue;
            const stop = if (b == 0) none else c.idom[b];
            for (c.preds[b]) |p| {
                if (!c.reachable(p)) continue;
                var r = p;
                while (r != stop) {
                    const l = &lists[r];
                    if (l.items.len == 0 or l.items[l.items.len - 1] != b) try l.append(a, b);
                    r = if (r == 0) none else c.idom[r];
                }
            }
        }
        const out = try a.alloc([]const u32, n);
        for (out, lists) |*o, *l| {
            std.mem.sort(u32, l.items, {}, std.sort.asc(u32));
            o.* = l.items;
        }
        c.df = out;
        return out;
    }
};

pub const Loop = struct {
    header: u32,
    /// Sources of the back edges into `header` (edges from a block the header dominates),
    /// sorted ascending.
    latches: []const u32,
    /// Every block of the natural loop, header included, sorted ascending.
    blocks: []const u32,
    /// Edges from a block of the loop to a block outside it, in block order.
    exits: []const Edge,
    /// Innermost enclosing loop (index into `Loops.loops`), or `none`.
    parent: u32,
    /// 1 for an outermost loop.
    depth: u32,
    /// The unique predecessor of `header` outside the loop if its terminator is `br`/`bra`
    /// to `header` (its only successor); otherwise `none` (see `ensurePreheader`). Always
    /// `none` when the header is block 0 (the function entry is an outside edge).
    preheader: u32,
};

pub const Loops = struct {
    arena: std.heap.ArenaAllocator,
    /// Natural loops of the reachable CFG; back edges with the same header form one loop.
    /// Ordered so that every loop comes after the loops it contains (innermost first).
    loops: []const Loop,
    /// Innermost loop containing each block, or `none`.
    loop_of: []const u32,
    /// Some cycle of the reachable CFG is entered other than through a block dominating it
    /// (a retreating edge whose target does not dominate its source). Such cycles are not
    /// natural loops; passes that need reducible control flow should skip the function.
    irreducible: bool,

    pub fn init(gpa: std.mem.Allocator, f: *const ir.Func, c: *const Cfg) std.mem.Allocator.Error!Loops {
        var arena = std.heap.ArenaAllocator.init(gpa);
        errdefer arena.deinit();
        const a = arena.allocator();
        const n: u32 = @intCast(f.blocks.items.len);
        var buf: [2]u32 = undefined;

        // Headers in RPO order, with their latches; irreducibility from retreating edges.
        var irreducible = false;
        var headers: std.ArrayList(u32) = .empty;
        defer headers.deinit(gpa);
        const latch_count = try gpa.alloc(u32, n);
        defer gpa.free(latch_count);
        @memset(latch_count, 0);
        for (c.rpo) |b| for (succs(f, b, &buf)) |s| {
            if (c.rpo_index[s] > c.rpo_index[b]) continue; // forward or cross edge
            if (c.dominates(s, b)) {
                if (latch_count[s] == 0) try headers.append(gpa, s);
                latch_count[s] += 1;
            } else irreducible = true;
        };
        std.mem.sort(u32, headers.items, c, struct {
            fn lt(g: *const Cfg, x: u32, y: u32) bool {
                return g.rpo_index[x] < g.rpo_index[y];
            }
        }.lt);

        const in_loop = try gpa.alloc(bool, n);
        defer gpa.free(in_loop);
        @memset(in_loop, false);
        var work: std.ArrayList(u32) = .empty;
        defer work.deinit(gpa);
        var members: std.ArrayList(u32) = .empty;
        defer members.deinit(gpa);

        const loops = try a.alloc(Loop, headers.items.len);
        for (headers.items, loops) |h, *lp| {
            // Latches (distinct, ascending): preds of h that h dominates.
            var latches: std.ArrayList(u32) = .empty;
            for (c.preds[h]) |p| {
                if (!c.dominates(h, p)) continue;
                if (latches.items.len == 0 or latches.items[latches.items.len - 1] != p) try latches.append(a, p);
            }
            // Body: h plus everything that reaches a latch without passing through h.
            members.clearRetainingCapacity();
            work.clearRetainingCapacity();
            in_loop[h] = true;
            try members.append(gpa, h);
            for (latches.items) |l| if (!in_loop[l]) {
                in_loop[l] = true;
                try members.append(gpa, l);
                try work.append(gpa, l);
            };
            while (work.pop()) |b| for (c.preds[b]) |p| {
                if (in_loop[p] or !c.reachable(p)) continue;
                in_loop[p] = true;
                try members.append(gpa, p);
                try work.append(gpa, p);
            };
            const blocks = try a.dupe(u32, members.items);
            std.mem.sort(u32, blocks, {}, std.sort.asc(u32));
            var exits: std.ArrayList(Edge) = .empty;
            for (blocks) |b| for (succs(f, b, &buf), 0..) |s, i| {
                if (!in_loop[s]) try exits.append(a, .{ .from = b, .to = s, .index = @intCast(i) });
            };
            for (blocks) |b| in_loop[b] = false;
            // Preheader: the single outside predecessor, ending in an unconditional branch.
            var pre: u32 = none;
            if (h != 0) {
                var outside: u32 = 0;
                var cand: u32 = none;
                for (c.preds[h]) |p| {
                    if (std.sort.binarySearch(u32, blocks, p, order) != null) continue;
                    outside += 1;
                    cand = p;
                }
                if (outside == 1 and succs(f, cand, &buf).len == 1) pre = cand;
            }
            lp.* = .{
                .header = h,
                .latches = latches.items,
                .blocks = blocks,
                .exits = exits.items,
                .parent = none,
                .depth = 1,
                .preheader = pre,
            };
        }

        // Innermost first: a loop containing another one has strictly more blocks.
        std.mem.sort(Loop, loops, c, struct {
            fn lt(g: *const Cfg, x: Loop, y: Loop) bool {
                if (x.blocks.len != y.blocks.len) return x.blocks.len < y.blocks.len;
                return g.rpo_index[x.header] < g.rpo_index[y.header];
            }
        }.lt);
        // Innermost-first, the first loop to claim a block is its innermost loop, and the first
        // loop to contain an already claimed block is the parent of that block's outermost
        // loop so far. `top` finds that outermost loop (union-find with path halving: a root
        // is a loop without a parent yet), so deep nests stay near-linear.
        const loop_of = try a.alloc(u32, n);
        @memset(loop_of, none);
        const top = try a.alloc(u32, loops.len);
        for (top, 0..) |*t_, k| t_.* = @intCast(k);
        for (loops, 0..) |lp, li| for (lp.blocks) |b| {
            if (loop_of[b] == none) {
                loop_of[b] = @intCast(li);
                continue;
            }
            var x = loop_of[b];
            while (top[x] != x) {
                top[x] = top[top[x]];
                x = top[x];
            }
            if (x != li) {
                loops[x].parent = @intCast(li);
                top[x] = @intCast(li);
            }
        };
        var i = loops.len; // parents come after their children
        while (i > 0) {
            i -= 1;
            if (loops[i].parent != none) loops[i].depth = loops[loops[i].parent].depth + 1;
        }
        return .{ .arena = arena, .loops = loops, .loop_of = loop_of, .irreducible = irreducible };
    }

    pub fn deinit(l: *Loops) void {
        l.arena.deinit();
    }
    /// Block `b` belongs to loop `loop` (directly or through a nested loop).
    pub fn contains(l: *const Loops, loop: u32, b: u32) bool {
        return std.sort.binarySearch(u32, l.loops[loop].blocks, b, order) != null;
    }
};

fn order(k: u32, x: u32) std.math.Order {
    return std.math.order(k, x);
}

/// Points edge `index` of `term` (a terminator) at `to` with edge arguments `args` (a
/// `cbr`/`br` stays argument-free when `args` is empty and the other edge has none).
pub fn setEdge(term: *ir.Op, index: u1, to: u32, args: []const ir.VReg) void {
    switch (term.*) {
        .br => |*target| {
            std.debug.assert(index == 0);
            if (args.len == 0) target.* = to else term.* = .{ .bra = .{ .target = to, .args = args } };
        },
        .bra => |*x| {
            std.debug.assert(index == 0);
            if (args.len == 0) term.* = .{ .br = to } else x.* = .{ .target = to, .args = args };
        },
        .cbr => |x| {
            var y: ir.Op = .{ .cbra = .{ .cond = x.cond, .a = x.a, .b = x.b, .then = x.then, .@"else" = x.@"else", .then_args = &.{}, .else_args = &.{} } };
            setEdge(&y, index, to, args);
            term.* = y;
        },
        .cbra => |*x| {
            if (index == 0) {
                x.then = to;
                x.then_args = args;
            } else {
                x.@"else" = to;
                x.else_args = args;
            }
            if (x.then_args.len == 0 and x.else_args.len == 0)
                term.* = .{ .cbr = .{ .cond = x.cond, .a = x.a, .b = x.b, .then = x.then, .@"else" = x.@"else" } };
        },
        else => unreachable,
    }
}

/// Target and arguments of edge `index` of terminator `term`.
pub fn edgeOf(term: ir.Op, index: u1) struct { to: u32, args: []const ir.VReg } {
    return switch (term) {
        .br => |to| .{ .to = to, .args = &.{} },
        .bra => |x| .{ .to = x.target, .args = x.args },
        .cbr => |x| .{ .to = if (index == 0) x.then else x.@"else", .args = &.{} },
        .cbra => |x| if (index == 0) .{ .to = x.then, .args = x.then_args } else .{ .to = x.@"else", .args = x.else_args },
        else => unreachable,
    };
}

/// The terminator (last op) of block `b`.
pub fn terminator(f: *ir.Func, b: u32) *ir.Op {
    const ops = f.blocks.items[b].ops.items;
    return &ops[ops.len - 1];
}

/// Splits edge `index` of block `from`: appends a block (`Ctx.addBlock`, frequency of the
/// edge when known) that branches to the old target, moving the edge's arguments (if any) to
/// the new block's `bra`, and retargets `from`'s terminator to it (an edge without arguments
/// stays `br`/`cbr`). Returns the new block. Valid in and out of SSA.
pub fn splitEdge(c: *opt.Ctx, from: u32, index: u1) opt.Error!u32 {
    const e = edgeOf(terminator(c.f, from).*, index);
    const nb = try c.addBlock("edge", c.edgeFreq(from, index));
    try c.f.emit(nb, if (e.args.len == 0) .{ .br = e.to } else .{ .bra = .{ .target = e.to, .args = e.args } });
    setEdge(terminator(c.f, from), index, nb, &.{});
    return nb;
}

/// Returns loop `l`'s preheader, creating one when `loops.loops[l].preheader == none`: a new
/// block that every predecessor of the header outside the loop branches to instead, and that
/// branches to the header. If the header has block parameters, the preheader gets fresh
/// parameters of the same types (`Ctx.newVReg`), the outside edges pass their arguments to
/// them and the preheader passes them on. When the header is block 0, its ops move to a new
/// block that becomes the header (the back edges follow them) and block 0 (which must have no
/// parameters) becomes the preheader; edges from unreachable blocks keep entering block 0.
/// `g`/`loops` describe `c.f` before the edit and are stale afterwards.
pub fn ensurePreheader(c: *opt.Ctx, g: *const Cfg, loops: *const Loops, l: u32) opt.Error!u32 {
    const lp = loops.loops[l];
    if (lp.preheader != none) return lp.preheader;
    const h = lp.header;
    if (h == 0) {
        if (c.f.blocks.items[0].params.len != 0) return error.InvalidSsa;
        const nh = try c.addBlock("entry", c.freq.items[0]);
        const blocks = c.f.blocks.items;
        blocks[nh].ops = blocks[0].ops;
        blocks[0].ops = .empty;
        c.taken.items[nh] = c.taken.items[0];
        c.taken.items[0] = null;
        c.freq.items[0] = null;
        var prev: u32 = none;
        for (g.preds[0]) |p| { // back edges only; unreachable blocks keep entering block 0
            if (p == prev or !loops.contains(l, p)) continue;
            prev = p;
            retargetAll(c.f, if (p == 0) nh else p, 0, nh);
        }
        try c.f.emit(0, .{ .br = nh });
        return 0;
    }
    const params = c.f.blocks.items[h].params;
    var fr: ?u64 = 0;
    var prev: u32 = none;
    for (g.preds[h]) |p| {
        if (p == prev or loops.contains(l, p)) continue; // duplicates are adjacent (sorted)
        prev = p;
        var buf: [2]u32 = undefined;
        for (succs(c.f, p, &buf), 0..) |s, i| if (s == h) {
            const ef = c.edgeFreq(p, @intCast(i));
            fr = if (fr != null and ef != null) fr.? +| ef.? else null;
        };
    }
    const ph = try c.addBlock("preheader", fr);
    const fresh = try c.irAlloc().alloc(ir.VReg, params.len);
    for (fresh, params) |*v, p| v.* = try c.newVReg(c.types.items[p]);
    c.f.blocks.items[ph].params = fresh;
    try c.f.emit(ph, if (fresh.len == 0) .{ .br = h } else .{ .bra = .{ .target = h, .args = fresh } });
    prev = none;
    for (g.preds[h]) |p| {
        if (p == prev or loops.contains(l, p)) continue;
        prev = p;
        retargetAll(c.f, p, h, ph);
    }
    return ph;
}

/// Makes block 0 an entry block without predecessors or parameters (as SSA form requires):
/// when it has either, its ops and parameters move to a new block that every edge into block
/// 0 now enters, and block 0 becomes a branch to it that passes each parameter's value on entry:
/// a function parameter's vreg holds the call argument, any other parameter not yet assigned
/// reads the zero of its type. Returns whether `c.f` changed.
pub fn isolateEntry(c: *opt.Ctx) opt.Error!bool {
    const f = c.f;
    var entered = false;
    var buf: [2]u32 = undefined;
    for (0..f.blocks.items.len) |b| for (succs(f, @intCast(b), &buf)) |s| {
        entered = entered or s == 0;
    };
    if (!entered and f.blocks.items[0].params.len == 0) return false;
    const nb = try c.addBlock("entry", c.freq.items[0]);
    const blocks = f.blocks.items;
    blocks[nb].ops = blocks[0].ops;
    blocks[nb].params = blocks[0].params;
    blocks[0].ops = .empty;
    blocks[0].params = &.{};
    c.taken.items[nb] = c.taken.items[0];
    c.taken.items[0] = null;
    for (1..blocks.len) |b| retargetAll(f, @intCast(b), 0, nb);
    const params = blocks[nb].params;
    const args = try c.irAlloc().alloc(ir.VReg, params.len);
    for (args, params) |*a, p| a.* = if (p < f.params) p else try c.zeroValue(c.irAlloc(), &f.blocks.items[0].ops, c.types.items[p]);
    try f.emit(0, if (args.len == 0) .{ .br = nb } else .{ .bra = .{ .target = nb, .args = args } });
    return true;
}

/// Points every edge of block `b` that enters `from` at `to`, keeping its arguments.
pub fn retargetAll(f: *ir.Func, b: u32, from: u32, to: u32) void {
    const term = terminator(f, b);
    var buf: [2]u32 = undefined;
    const ss = succs(f, b, &buf);
    for (ss, 0..) |s, i| if (s == from) {
        const e = edgeOf(term.*, @intCast(i));
        setEdge(term, @intCast(i), to, e.args);
    };
}

// ---- tests -------------------------------------------------------------------------------

const t = std.testing;

fn parse(src: []const u8) !ir.Func {
    var f = try ir.parse(t.allocator, src);
    errdefer f.deinit();
    try f.verify();
    return f;
}

fn blockIndex(f: *const ir.Func, name: []const u8) u32 {
    for (f.blocks.items, 0..) |b, i| if (std.mem.eql(u8, b.name, name)) return @intCast(i);
    unreachable;
}

test "cfg: diamond, unreachable block, dominators and frontiers" {
    var f = try parse(
        \\func d 2
        \\e:
        \\  cbr lt v0 v1 l r
        \\l:
        \\  v2 = add v0 v1
        \\  br j
        \\r:
        \\  v2 = sub v0 v1
        \\  br j
        \\dead:
        \\  br j
        \\j:
        \\  ret v2
    );
    defer f.deinit();
    var g = try Cfg.init(t.allocator, &f);
    defer g.deinit();
    const e = 0;
    const l = blockIndex(&f, "l");
    const r = blockIndex(&f, "r");
    const dead = blockIndex(&f, "dead");
    const j = blockIndex(&f, "j");
    try t.expectEqualSlices(u32, &.{ l, r, dead }, g.preds[j]);
    try t.expectEqual(@as(usize, 4), g.rpo.len);
    try t.expectEqual(@as(u32, 0), g.rpo[0]);
    try t.expect(!g.reachable(dead));
    try t.expectEqual(none, g.idom[dead]);
    try t.expectEqual(@as(u32, e), g.idom[j]);
    try t.expectEqual(@as(u32, e), g.idom[l]);
    try t.expect(g.dominates(e, j) and g.dominates(j, j) and !g.dominates(l, j) and !g.dominates(dead, j) and !g.dominates(j, dead));
    const df = try g.frontiers();
    try t.expectEqualSlices(u32, &.{j}, df[l]);
    try t.expectEqualSlices(u32, &.{j}, df[r]);
    try t.expectEqualSlices(u32, &.{}, df[e]);
    try t.expectEqualSlices(u32, &.{}, df[j]);
    try t.expectEqualSlices(u32, &.{}, df[dead]);
    try t.expectEqual(@as(usize, 3), g.dom_children[e].len);
}

test "cfg: cbr with equal targets gives two predecessor entries" {
    var f = try parse(
        \\func s 1
        \\e:
        \\  cbr eq v0 v0 x x
        \\x:
        \\  ret v0
    );
    defer f.deinit();
    var g = try Cfg.init(t.allocator, &f);
    defer g.deinit();
    try t.expectEqualSlices(u32, &.{ 0, 0 }, g.preds[1]);
    try t.expectEqual(@as(u32, 0), g.idom[1]);
    const df = try g.frontiers();
    try t.expectEqual(@as(usize, 0), df[1].len);
}

test "cfg: nested loops, latches, exits, preheaders, frontiers of a loop" {
    var f = try parse(
        \\func n 1
        \\e:
        \\  v1 = const 0
        \\  br oh
        \\oh:
        \\  cbr lt v1 v0 ib done
        \\ib:
        \\  v2 = const 0
        \\  br ih
        \\ih:
        \\  cbr lt v2 v1 ibody olatch
        \\ibody:
        \\  v2 = add v2 v0
        \\  br ih
        \\olatch:
        \\  v3 = const 1
        \\  v1 = add v1 v3
        \\  br oh
        \\done:
        \\  ret v1
    );
    defer f.deinit();
    var g = try Cfg.init(t.allocator, &f);
    defer g.deinit();
    var ls = try Loops.init(t.allocator, &f, &g);
    defer ls.deinit();
    const oh = blockIndex(&f, "oh");
    const ib = blockIndex(&f, "ib");
    const ih = blockIndex(&f, "ih");
    const ibody = blockIndex(&f, "ibody");
    const olatch = blockIndex(&f, "olatch");
    const done = blockIndex(&f, "done");
    try t.expect(!ls.irreducible);
    try t.expectEqual(@as(usize, 2), ls.loops.len);
    const inner = ls.loops[0];
    const outer = ls.loops[1];
    try t.expectEqual(ih, inner.header);
    try t.expectEqual(oh, outer.header);
    try t.expectEqualSlices(u32, &.{ ih, ibody }, inner.blocks);
    try t.expectEqualSlices(u32, &.{ oh, ib, ih, ibody, olatch }, outer.blocks);
    try t.expectEqualSlices(u32, &.{ibody}, inner.latches);
    try t.expectEqualSlices(u32, &.{olatch}, outer.latches);
    try t.expectEqual(@as(u32, 1), inner.parent);
    try t.expectEqual(none, outer.parent);
    try t.expectEqual(@as(u32, 2), inner.depth);
    try t.expectEqual(@as(u32, 1), outer.depth);
    try t.expectEqual(ib, inner.preheader);
    try t.expectEqual(@as(u32, 0), outer.preheader);
    try t.expectEqual(@as(usize, 1), inner.exits.len);
    try t.expectEqual(Edge{ .from = ih, .to = olatch, .index = 1 }, inner.exits[0]);
    try t.expectEqual(Edge{ .from = oh, .to = done, .index = 1 }, outer.exits[0]);
    try t.expectEqual(@as(u32, 0), ls.loop_of[ibody]);
    try t.expectEqual(@as(u32, 1), ls.loop_of[olatch]);
    try t.expectEqual(none, ls.loop_of[done]);
    try t.expect(ls.contains(1, ibody) and !ls.contains(0, olatch));
    const df = try g.frontiers();
    try t.expectEqualSlices(u32, &.{ih}, df[ibody]);
    try t.expectEqualSlices(u32, &.{ oh, ih }, df[ih]);
    try t.expectEqualSlices(u32, &.{oh}, df[olatch]);
}

test "cfg: irreducible cycle is flagged, not a loop" {
    var f = try parse(
        \\func i 1
        \\e:
        \\  cbr lt v0 v0 a b
        \\a:
        \\  cbr lt v0 v0 b x
        \\b:
        \\  cbr lt v0 v0 a x
        \\x:
        \\  ret v0
    );
    defer f.deinit();
    var g = try Cfg.init(t.allocator, &f);
    defer g.deinit();
    var ls = try Loops.init(t.allocator, &f, &g);
    defer ls.deinit();
    try t.expect(ls.irreducible);
    try t.expectEqual(@as(usize, 0), ls.loops.len);
}

test "cfg: loop at the entry block and self loops" {
    var f = try parse(
        \\func s 1
        \\e:
        \\  v0 = sub v0 v0
        \\  cbr ne v0 v0 e x
        \\x:
        \\  br x2
        \\x2:
        \\  cbr eq v0 v0 x2 y
        \\y:
        \\  ret v0
    );
    defer f.deinit();
    var g = try Cfg.init(t.allocator, &f);
    defer g.deinit();
    var ls = try Loops.init(t.allocator, &f, &g);
    defer ls.deinit();
    try t.expectEqual(@as(usize, 2), ls.loops.len);
    for (ls.loops) |lp| {
        try t.expectEqual(@as(usize, 1), lp.blocks.len);
        try t.expectEqualSlices(u32, &.{lp.header}, lp.latches);
    }
    const x2 = blockIndex(&f, "x2");
    const lx2: u32 = if (ls.loops[0].header == x2) 0 else 1;
    try t.expectEqual(blockIndex(&f, "x"), ls.loops[lx2].preheader);
    try t.expectEqual(none, ls.loops[1 - lx2].preheader);
    const df = try g.frontiers();
    try t.expectEqualSlices(u32, &.{0}, df[0]);
}

fn interpretAll(f: *const ir.Func, g: *const ir.Func, inputs: []const [2]i64) !void {
    for (inputs) |in| try t.expectEqual(try ir.interpret(f, &in, 100_000), try ir.interpret(g, &in, 100_000));
}

const loop_params_src =
    \\func p 2
    \\e:
    \\  v2 = const 1
    \\  cbr lt v0 v1 pre1 pre2:v2
    \\pre1:
    \\  v3 = const 7
    \\  br h v3
    \\pre2(v9):
    \\  br h v9
    \\h(v4):
    \\  v5 = add v4 v2
    \\  cbr lt v5 v1 h:v5 out
    \\out:
    \\  ret v5
;

test "cfg: splitEdge moves edge arguments and keeps semantics" {
    var f = try parse(loop_params_src);
    defer f.deinit();
    var ref = try parse(loop_params_src);
    defer ref.deinit();
    var o: opt.Options = .{};
    var st: opt.Stats = .{};
    var c = try opt.Ctx.init(t.allocator, &f, &o, &st);
    defer c.deinit();
    const h = blockIndex(&f, "h");
    const back = try splitEdge(&c, h, 0); // h -> h carries v5
    const to_out = try splitEdge(&c, h, 1); // h -> out, no arguments
    const e2 = try splitEdge(&c, 0, 1); // e -> pre2 carries v2
    try c.check("split");
    try f.verifySsa(t.allocator);
    try t.expect(f.blocks.items[back].ops.items[0] == .bra);
    try t.expect(f.blocks.items[to_out].ops.items[0] == .br);
    try t.expect(f.blocks.items[0].ops.items[1] == .cbr); // both edges argument-free now
    try t.expect(f.blocks.items[e2].ops.items[0] == .bra);
    try t.expect(f.blocks.items[h].ops.items[1] == .cbr);
    try interpretAll(&ref, &f, &.{ .{ 0, 10 }, .{ 5, 3 }, .{ -4, 40 }, .{ 9, 9 } });
}

test "cfg: ensurePreheader with block parameters and several outside edges" {
    var f = try parse(loop_params_src);
    defer f.deinit();
    var ref = try parse(loop_params_src);
    defer ref.deinit();
    var o: opt.Options = .{};
    var st: opt.Stats = .{};
    var c = try opt.Ctx.init(t.allocator, &f, &o, &st);
    defer c.deinit();
    {
        var g = try Cfg.init(t.allocator, &f);
        defer g.deinit();
        var ls = try Loops.init(t.allocator, &f, &g);
        defer ls.deinit();
        try t.expectEqual(@as(usize, 1), ls.loops.len);
        try t.expectEqual(none, ls.loops[0].preheader);
        const ph = try ensurePreheader(&c, &g, &ls, 0);
        try t.expectEqual(@as(usize, 1), f.blocks.items[ph].params.len);
    }
    try c.check("preheader");
    try f.verifySsa(t.allocator);
    var g = try Cfg.init(t.allocator, &f);
    defer g.deinit();
    var ls = try Loops.init(t.allocator, &f, &g);
    defer ls.deinit();
    try t.expect(ls.loops[0].preheader != none);
    try interpretAll(&ref, &f, &.{ .{ 0, 10 }, .{ 5, 3 }, .{ -4, 40 }, .{ 9, 9 } });
}

test "cfg: ensurePreheader on a loop headed by the entry block" {
    const src =
        \\func s 2
        \\e:
        \\  v0 = add v0 v1
        \\  cbr lt v0 v1 e x
        \\x:
        \\  ret v0
    ;
    var f = try parse(src);
    defer f.deinit();
    var ref = try parse(src);
    defer ref.deinit();
    var o: opt.Options = .{};
    var st: opt.Stats = .{};
    var c = try opt.Ctx.init(t.allocator, &f, &o, &st);
    defer c.deinit();
    {
        var g = try Cfg.init(t.allocator, &f);
        defer g.deinit();
        var ls = try Loops.init(t.allocator, &f, &g);
        defer ls.deinit();
        try t.expectEqual(@as(u32, 0), try ensurePreheader(&c, &g, &ls, 0));
    }
    try c.check("preheader");
    var g = try Cfg.init(t.allocator, &f);
    defer g.deinit();
    try t.expectEqual(@as(usize, 0), g.preds[0].len);
    var ls = try Loops.init(t.allocator, &f, &g);
    defer ls.deinit();
    try t.expectEqual(@as(u32, 0), ls.loops[0].preheader);
    try interpretAll(&ref, &f, &.{ .{ 1, 10 }, .{ -3, 2 }, .{ 0, 0 }, .{ 50, 1 } });
}

/// Random CFG over `n` blocks: each block bumps a step counter (v1) and ends in `ret`, `br`
/// or a `cbr` on the counter, so every program terminates and both edges get exercised.
fn randomCfg(rng: std.Random, n: u32, out: *std.ArrayList(u8)) !void {
    const w = struct {
        fn print(o: *std.ArrayList(u8), comptime fmt: []const u8, args: anytype) !void {
            try o.print(t.allocator, fmt, args);
        }
    }.print;
    try w(out, "func r 1\n", .{});
    for (0..n) |b| {
        try w(out, "b{d}:\n  v2 = const {d}\n  v1 = add v1 v2\n", .{ b, rng.intRangeAtMost(i64, 1, 3) });
        switch (if (b == 0 and n > 1) rng.intRangeLessThan(u32, 1, 3) else rng.uintLessThan(u32, 4)) {
            0 => try w(out, "  ret v1\n", .{}),
            1 => try w(out, "  br b{d}\n", .{rng.uintLessThan(u32, n)}),
            else => try w(out, "  v3 = const {d}\n  cbr lt v1 v3 b{d} b{d}\n", .{
                rng.intRangeAtMost(i64, 0, 40), rng.uintLessThan(u32, n), rng.uintLessThan(u32, n),
            }),
        }
    }
}

test "cfg: random CFGs match naive dominators, frontiers and natural loops" {
    var prng = std.Random.DefaultPrng.init(0x5eed_cf9);
    const rng = prng.random();
    var src: std.ArrayList(u8) = .empty;
    defer src.deinit(t.allocator);
    for (0..400) |iter| {
        const n: u32 = 1 + @as(u32, @intCast(iter % 24));
        src.clearRetainingCapacity();
        try randomCfg(rng, n, &src);
        var f = try parse(src.items);
        defer f.deinit();
        var g = try Cfg.init(t.allocator, &f);
        defer g.deinit();

        // Naive dominator sets: dom[b] = {b} + intersection of dom[p] over reachable preds.
        var dom: [24]std.bit_set.IntegerBitSet(24) = undefined;
        for (&dom, 0..) |*d, b| {
            d.* = .full;
            if (b == 0) d.* = .empty;
        }
        dom[0].set(0);
        var changed = true;
        while (changed) {
            changed = false;
            for (1..n) |b| {
                if (!g.reachable(@intCast(b))) continue;
                var d: std.bit_set.IntegerBitSet(24) = .full;
                for (g.preds[b]) |p| if (g.reachable(p)) d.setIntersection(dom[p]);
                d.set(b);
                if (!d.eql(dom[b])) {
                    dom[b] = d;
                    changed = true;
                }
            }
        }
        for (0..n) |b| for (0..n) |a| {
            const want = g.reachable(@intCast(a)) and g.reachable(@intCast(b)) and dom[b].isSet(a);
            try t.expectEqual(want, g.dominates(@intCast(a), @intCast(b)));
        };
        for (g.rpo[1..]) |b| { // idom: the strict dominator dominated by all others
            const i = g.idom[b];
            try t.expect(i != b and dom[b].isSet(i));
            try t.expectEqual(dom[b].count() - 1, dom[i].count());
        }
        // Frontier definition (entry has a virtual incoming edge, so 0 is a join).
        const df = try g.frontiers();
        for (0..n) |x| {
            var want: std.bit_set.IntegerBitSet(24) = .empty;
            if (g.reachable(@intCast(x))) for (g.rpo) |y| {
                const strict = y != x and g.dominates(@intCast(x), y);
                for (g.preds[y]) |p| if (g.dominates(@intCast(x), p) and !strict) want.set(y);
            };
            var got: std.bit_set.IntegerBitSet(24) = .empty;
            for (df[x], 0..) |y, k| {
                if (k > 0) try t.expect(df[x][k - 1] < y);
                got.set(y);
            }
            try t.expect(want.eql(got));
        }
        // Natural loops: one per header of back edges; bodies by backward reachability.
        var ls = try Loops.init(t.allocator, &f, &g);
        defer ls.deinit();
        var headers: std.bit_set.IntegerBitSet(24) = .empty;
        var buf: [2]u32 = undefined;
        for (g.rpo) |b| for (succs(&f, b, &buf)) |s| {
            if (g.dominates(s, b)) headers.set(s);
        };
        try t.expectEqual(headers.count(), ls.loops.len);
        for (ls.loops, 0..) |lp, li| {
            try t.expect(headers.isSet(lp.header));
            var body: std.bit_set.IntegerBitSet(24) = .empty;
            body.set(lp.header);
            var grew = true;
            for (g.preds[lp.header]) |p| if (g.dominates(lp.header, p)) body.set(p);
            while (grew) {
                grew = false;
                for (0..n) |b| if (body.isSet(b) and b != lp.header) for (g.preds[b]) |p| {
                    if (g.reachable(p) and !body.isSet(p)) {
                        body.set(p);
                        grew = true;
                    }
                };
            }
            try t.expectEqual(body.count(), lp.blocks.len);
            for (lp.blocks) |b| try t.expect(body.isSet(b) and ls.contains(@intCast(li), b));
            // Parent: the smallest other loop containing the header.
            var best: u32 = none;
            for (ls.loops, 0..) |o, oi| {
                if (oi == li or !ls.contains(@intCast(oi), lp.header)) continue;
                if (best == none or o.blocks.len < ls.loops[best].blocks.len) best = @intCast(oi);
            }
            try t.expectEqual(best, lp.parent);
            try t.expectEqual(if (best == none) 1 else ls.loops[best].depth + 1, lp.depth);
            if (lp.parent != none) try t.expect(lp.parent > li);
        }
        for (0..n) |b| {
            var best: u32 = none;
            for (ls.loops, 0..) |o, oi| if (ls.contains(@intCast(oi), @intCast(b))) {
                if (best == none or o.blocks.len < ls.loops[best].blocks.len) best = @intCast(oi);
            };
            try t.expectEqual(best, ls.loop_of[b]);
        }

        // Give every loop a preheader (analyses recomputed after each edit) and compare runs.
        if (ls.irreducible) continue;
        var ref = try parse(src.items);
        defer ref.deinit();
        var o: opt.Options = .{};
        var st: opt.Stats = .{};
        var c = try opt.Ctx.init(t.allocator, &f, &o, &st);
        defer c.deinit();
        const nloops = ls.loops.len;
        for (0..nloops) |_| {
            var g2 = try Cfg.init(t.allocator, &f);
            defer g2.deinit();
            var l2 = try Loops.init(t.allocator, &f, &g2);
            defer l2.deinit();
            try t.expectEqual(nloops, l2.loops.len);
            for (l2.loops, 0..) |lp, li| if (lp.preheader == none) {
                _ = try ensurePreheader(&c, &g2, &l2, @intCast(li));
                break;
            };
        }
        try c.check("preheaders");
        var g3 = try Cfg.init(t.allocator, &f);
        defer g3.deinit();
        var l3 = try Loops.init(t.allocator, &f, &g3);
        defer l3.deinit();
        try t.expectEqual(nloops, l3.loops.len);
        try t.expect(!l3.irreducible);
        for (l3.loops) |lp| try t.expect(lp.preheader != none);
        for ([_]i64{ 0, 5, -3, 17 }) |a| try t.expectEqual(
            ir.interpret(&ref, &.{a}, 10_000) catch |e| @as(anyerror!i64, e),
            ir.interpret(&f, &.{a}, 10_000) catch |e| @as(anyerror!i64, e),
        );
    }
}

test "cfg: isolateEntry moves a looping, parameterized entry block" {
    const src =
        \\func s 2
        \\e(v2 v3):
        \\  v4 = fconst.f64 1.5
        \\  v5 = add.f64 v3 v4
        \\  v2 = add v2 v1
        \\  cbr lt v2 v0 e:v2:v5 x
        \\x:
        \\  v6 = cvt.i64.f64 v5
        \\  v7 = add v6 v2
        \\  ret v7
    ;
    var f = try parse(src);
    defer f.deinit();
    var ref = try parse(src);
    defer ref.deinit();
    var o: opt.Options = .{};
    var st: opt.Stats = .{};
    var c = try opt.Ctx.init(t.allocator, &f, &o, &st);
    defer c.deinit();
    try t.expect(try isolateEntry(&c));
    try c.check("isolateEntry");
    try t.expect(!try isolateEntry(&c));
    var g = try Cfg.init(t.allocator, &f);
    defer g.deinit();
    try t.expectEqual(@as(usize, 0), g.preds[0].len);
    try t.expectEqual(@as(usize, 0), f.blocks.items[0].params.len);
    try interpretAll(&ref, &f, &.{ .{ 3, 1 }, .{ -5, 2 }, .{ 100, 7 }, .{ 0, 0 } });
}

test "cfg: Loops of a deep perfect nest (parents and depths)" {
    // h0 ⊃ h1 ⊃ … ⊃ h(d-1) (a self loop); t(i) closes loop h(i-1). Parent assignment walks a
    // union-find, so deep nests stay near-linear (a plain parent-chain walk is cubic here).
    const gpa = t.allocator;
    const d = 120;
    var src: std.ArrayList(u8) = .empty;
    defer src.deinit(gpa);
    try src.print(gpa, "func n 1\ne:\n  v1 = const 0\n  br h0\n", .{});
    for (0..d) |i| {
        const inner = if (i + 1 < d) i + 1 else i;
        try src.print(gpa, "h{d}:\n  v1 = add v1 v0\n  cbr lt v1 v0 h{d} t{d}\n", .{ i, inner, i });
    }
    for (0..d) |i| {
        if (i == 0) try src.print(gpa, "t0:\n  ret v1\n", .{}) else try src.print(gpa, "t{d}:\n  br h{d}\n", .{ i, i - 1 });
    }
    var f = try ir.parse(gpa, src.items);
    defer f.deinit();
    var g = try Cfg.init(gpa, &f);
    defer g.deinit();
    var ls = try Loops.init(gpa, &f, &g);
    defer ls.deinit();
    try t.expectEqual(@as(usize, d), ls.loops.len);
    // Block indices follow the source: e = 0, h(i) = 1 + i.
    var by_header: [d]u32 = undefined;
    for (ls.loops, 0..) |lp, li| by_header[lp.header - 1] = @intCast(li);
    for (0..d) |i| {
        const lp = ls.loops[by_header[i]];
        try t.expectEqual(@as(u32, @intCast(i + 1)), lp.depth);
        try t.expectEqual(if (i == 0) none else by_header[i - 1], lp.parent);
    }
}
