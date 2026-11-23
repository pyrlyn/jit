//! Loop rotation (top-tested loops become guarded bottom-tested loops) and loop helpers
//! shared with `licm`. Runs in SSA.
//!
//! A top-tested loop `H: cbr c B X; B: ...; L: br H` executes two branches per iteration and
//! its body does not dominate the exit. `rotate` keeps H as a guard that runs once per loop
//! entry and replaces the latch's jump back to H by a copy of H's ops and conditional branch
//! (`L: ...; H'; cbr c' B X`), so each iteration runs one conditional branch and the body
//! dominates the loop's exiting test (which lets `licm` hoist divisions and loads). Values H
//! defines become block parameters of the body entry and of the exit, fed by the guard
//! (original values) and by the latch (the copies).
const std = @import("std");
const ir = @import("../ir.zig");
const opt = @import("opt.zig");
const cfg = @import("cfg.zig");
const ssa = @import("ssa.zig");

const none = cfg.none;

/// Largest header (ops besides its terminator) that `rotate` duplicates into the latch.
pub const max_header_ops = 16;

/// Block defining each vreg (function parameters: block 0; `cfg.none` when nothing defines
/// it). Caller frees.
pub fn defBlocks(gpa: std.mem.Allocator, f: *const ir.Func) std.mem.Allocator.Error![]u32 {
    const d = try gpa.alloc(u32, f.num_vregs);
    @memset(d, none);
    for (0..f.params) |v| d[v] = 0;
    for (f.blocks.items, 0..) |b, bi| {
        for (b.params) |p| d[p] = @intCast(bi);
        for (b.ops.items) |o| if (o.def()) |v| {
            d[v] = @intCast(bi);
        };
    }
    return d;
}

/// Nearest common dominator of the reachable blocks `a` and `b`.
pub fn commonDominator(g: *const cfg.Cfg, a: u32, b: u32) u32 {
    var x = a;
    var y = b;
    while (x != y) {
        while (g.rpo_index[x] > g.rpo_index[y]) x = g.idom[x];
        while (g.rpo_index[y] > g.rpo_index[x]) y = g.idom[y];
    }
    return x;
}

/// Edge index (0: then, 1: else) of the header's conditional branch that stays in loop `li`
/// when the loop is a rotation candidate: the header is not a latch (else the loop is already
/// bottom-tested), has no call, at most `max_header_ops` ops, and ends in `cbr`/`cbra` with
/// exactly one successor inside the loop.
fn headerShape(f: *const ir.Func, ls: *const cfg.Loops, li: u32) ?u1 {
    const lp = ls.loops[li];
    const h = lp.header;
    if (h == 0) return null; // never in SSA (block 0 has no predecessors)
    for (lp.latches) |l| if (l == h) return null;
    const ops = f.blocks.items[h].ops.items;
    if (ops.len - 1 > max_header_ops) return null;
    for (ops) |o| if (o == .call) return null;
    switch (ops[ops.len - 1]) {
        .cbr, .cbra => {},
        else => return null,
    }
    var buf: [2]u32 = undefined;
    const ss = cfg.succs(f, h, &buf);
    const in0 = ls.contains(li, ss[0]);
    if (in0 == ls.contains(li, ss[1])) return null;
    return if (in0) 0 else 1;
}

fn isJump(op: ir.Op) bool {
    return op == .br or op == .bra;
}

/// Points every back edge of `lp` at a new block (`latch.N`, parameters like the header's)
/// that jumps to the header, so the loop has one latch ending in `br`/`bra`. Its frequency is
/// the sum of the back edges'. Returns the new block.
fn mergeLatches(c: *opt.Ctx, lp: cfg.Loop) opt.Error!u32 {
    const h = lp.header;
    var fr: ?u64 = 0;
    var buf: [2]u32 = undefined;
    for (lp.latches) |l| for (cfg.succs(c.f, l, &buf), 0..) |s, i| if (s == h) {
        const ef = c.edgeFreq(l, @intCast(i));
        fr = if (fr != null and ef != null) fr.? +| ef.? else null;
    };
    const ll = try c.addBlock("latch", fr);
    const params = c.f.blocks.items[h].params;
    const fresh = try c.irAlloc().alloc(ir.VReg, params.len);
    for (fresh, params) |*v, p| v.* = try c.newVReg(c.types.items[p]);
    c.f.blocks.items[ll].params = fresh;
    try c.f.emit(ll, if (fresh.len == 0) .{ .br = h } else .{ .bra = .{ .target = h, .args = fresh } });
    for (lp.latches) |l| cfg.retargetAll(c.f, l, h, ll);
    return ll;
}

/// A loop selected for rotation (block indices of the current CFG snapshot).
const Cand = struct {
    h: u32,
    latch: u32,
    /// Edge index of H's branch into the loop body (the other one leaves the loop).
    body: u1,
    /// Successors of H: `succ[0]` in the loop (B), `succ[1]` outside (X).
    succ: [2]u32,
    /// The successor has predecessors besides H, so the edge from H gets its own block.
    split: [2]bool,
    /// Every reachable predecessor of X except H is dominated by X, so the edge H->X
    /// dominates X's dominator subtree (values from H used there can be renamed).
    x_ok: bool,
    rejected: bool = false,
};

/// Uses of H's values in the dominator subtree of `block` read `params` instead of `vals`.
const Region = struct { block: u32, vals: []const ir.VReg, params: []const ir.VReg };

/// Rotates every top-tested loop of a reducible function whose header qualifies (see
/// `headerShape`) and whose header values (parameters included) are used, outside the
/// header, only in blocks dominated by the body entry B or by the exit X (when the edge
/// H->X dominates X's subtree). A loop with several latches, or a latch ending in a
/// conditional branch, first gets a merged latch block. The header is not given a preheader:
/// it becomes the guard and keeps its outside predecessors.
///
/// Profile estimate: the guard runs `entries = freq(H) - freq(latch)` times and enters the body
/// `min(entries, body edge count)` times (a loop that ran at all ran from its first entry); the
/// latch's new branch continues the remaining body-edge count. Body and exit blocks keep their
/// counts (edge blocks created by splitting get the old edge counts).
///
/// Returns true iff the function changed (a loop rotated or a latch block was added).
pub fn rotate(c: *opt.Ctx) opt.Error!bool {
    const gpa = c.gpa;
    var g = try cfg.Cfg.init(gpa, c.f);
    defer g.deinit();
    var ls = try cfg.Loops.init(gpa, c.f, &g);
    defer ls.deinit();
    if (ls.irreducible or ls.loops.len == 0) return false;

    // Single unconditional latches. The snapshot stays valid for the other loops: merging
    // only redirects edges into the merged loop's own header.
    var changed = false;
    for (ls.loops, 0..) |lp, li| {
        if (headerShape(c.f, &ls, @intCast(li)) == null) continue;
        if (lp.latches.len == 1 and isJump(cfg.terminator(c.f, lp.latches[0]).*)) continue;
        _ = try mergeLatches(c, lp);
        changed = true;
    }
    if (changed) {
        var g2 = try cfg.Cfg.init(gpa, c.f);
        errdefer g2.deinit();
        const ls2 = try cfg.Loops.init(gpa, c.f, &g2);
        ls.deinit();
        g.deinit();
        g = g2;
        ls = ls2;
    }

    // Candidates; `owner[v]`: candidate whose header defines v.
    const nv = c.f.num_vregs;
    const owner = try gpa.alloc(u32, nv);
    defer gpa.free(owner);
    @memset(owner, none);
    var cands: std.ArrayList(Cand) = .empty;
    defer cands.deinit(gpa);
    for (ls.loops, 0..) |lp, li| {
        const body = headerShape(c.f, &ls, @intCast(li)) orelse continue;
        if (lp.latches.len != 1 or !isJump(cfg.terminator(c.f, lp.latches[0]).*)) continue;
        const h = lp.header;
        const term = cfg.terminator(c.f, h).*;
        const b = cfg.edgeOf(term, body).to;
        const x = cfg.edgeOf(term, body ^ 1).to;
        var x_ok = true;
        for (g.preds[x]) |p| {
            if (p != h and g.reachable(p) and !g.dominates(x, p)) x_ok = false;
        }
        const k: u32 = @intCast(cands.items.len);
        try cands.append(gpa, .{
            .h = h,
            .latch = lp.latches[0],
            .body = body,
            .succ = .{ b, x },
            .split = .{ g.preds[b].len != 1, g.preds[x].len != 1 },
            .x_ok = x_ok,
        });
        for (c.f.blocks.items[h].params) |p| owner[p] = k;
        for (c.f.blocks.items[h].ops.items) |o| if (o.def()) |d| {
            owner[d] = k;
        };
    }
    if (cands.items.len == 0) return changed;

    // Where header values are used: `need[0][v]` in B's subtree, `need[1][v]` in X's.
    const need_b = try gpa.alloc(bool, nv);
    defer gpa.free(need_b);
    const need_x = try gpa.alloc(bool, nv);
    defer gpa.free(need_x);
    @memset(need_b, false);
    @memset(need_x, false);
    var scratch: std.ArrayList(ir.VReg) = .empty;
    defer scratch.deinit(gpa);
    for (c.f.blocks.items, 0..) |blk, bi| {
        const u: u32 = @intCast(bi);
        if (!g.reachable(u)) continue; // never executed; SSA does not constrain its uses
        for (blk.ops.items) |o| for (try ssa.usesOf(gpa, o, &scratch)) |v| {
            const k = owner[v];
            if (k == none) continue;
            const cd = &cands.items[k];
            if (u == cd.h) continue;
            if (g.dominates(cd.succ[0], u)) {
                need_b[v] = true;
            } else if (cd.x_ok and g.dominates(cd.succ[1], u)) {
                need_x[v] = true;
            } else cd.rejected = true;
        };
    }
    // Arguments of a split edge move into the new edge block, which renames them too.
    for (cands.items, 0..) |cd, k| {
        const term = cfg.terminator(c.f, cd.h).*;
        for ([2]u1{ cd.body, cd.body ^ 1 }, [2][]bool{ need_b, need_x }, cd.split) |idx, need, split| {
            if (split) for (cfg.edgeOf(term, idx).args) |v| {
                if (owner[v] == k) need[v] = true;
            };
        }
    }

    const sig = try gpa.alloc(ir.VReg, nv);
    defer gpa.free(sig);
    for (sig, 0..) |*s, i| s.* = @intCast(i);
    var regions: std.ArrayList(Region) = .empty;
    defer regions.deinit(gpa);
    for (cands.items) |cd| {
        if (cd.rejected) continue;
        try rotateOne(c, cd, .{ need_b, need_x }, sig, &regions);
        changed = true;
    }
    if (regions.items.len != 0) try renameRegions(c, regions.items);
    return changed;
}

/// Rotates one candidate (see `rotate`). `sig` is an identity map over the vregs that existed
/// when the candidates were chosen; it is restored before returning.
fn rotateOne(c: *opt.Ctx, cd: Cand, need: [2][]const bool, sig: []ir.VReg, regions: *std.ArrayList(Region)) opt.Error!void {
    const gpa = c.gpa;
    const ia = c.irAlloc();
    const f = c.f;
    const h = cd.h;

    // Profile inputs, before any edit.
    const f_latch = c.freq.items[cd.latch];
    const entries: ?u64 = if (c.freq.items[h] != null and f_latch != null) c.freq.items[h].? -| f_latch.? else null;
    const t_body = c.edgeFreq(h, cd.body);

    // Values H defines, in definition order.
    var hvals: std.ArrayList(ir.VReg) = .empty;
    defer hvals.deinit(gpa);
    try hvals.appendSlice(gpa, f.blocks.items[h].params);
    {
        const hops = f.blocks.items[h].ops.items;
        for (hops[0 .. hops.len - 1]) |o| if (o.def()) |d| try hvals.append(gpa, d);
    }

    // Body entry (side 0) and exit (side 1): split if shared, add parameters for the header
    // values used below them, and pass the original values from H.
    const old = cfg.terminator(f, h).*;
    var target: [2]u32 = undefined; // by edge index
    var edge_args: [2][]const ir.VReg = undefined; // by edge index, H's edges after the edit
    for (0..2) |side| {
        const idx: u1 = if (side == 0) cd.body else cd.body ^ 1;
        const e = cfg.edgeOf(old, idx);
        var blk = e.to;
        var base = e.args;
        var old_params = f.blocks.items[blk].params;
        if (cd.split[side]) {
            blk = try cfg.splitEdge(c, h, idx); // moves `e.args` to the new block's branch
            base = &.{};
            old_params = &.{};
        }
        var n: usize = 0;
        for (hvals.items) |v| n += @intFromBool(need[side][v]);
        const args = try ia.alloc(ir.VReg, base.len + n);
        const params = try ia.alloc(ir.VReg, old_params.len + n);
        @memcpy(args[0..base.len], base);
        @memcpy(params[0..old_params.len], old_params);
        var i: usize = 0;
        for (hvals.items) |v| if (need[side][v]) {
            args[base.len + i] = v;
            params[old_params.len + i] = try c.newVReg(c.types.items[v]);
            i += 1;
        };
        f.blocks.items[blk].params = params;
        cfg.setEdge(cfg.terminator(f, h), idx, blk, args);
        target[idx] = blk;
        edge_args[idx] = args;
        if (n != 0) try regions.append(gpa, .{ .block = blk, .vals = args[base.len..], .params = params[old_params.len..] });
    }

    // The latch's jump to H becomes a copy of H: parameters read the latch's arguments,
    // results get fresh vregs, and the branch passes the copies to the new parameters.
    const latch_ops = &f.blocks.items[cd.latch].ops;
    const jump = latch_ops.pop().?;
    const hparams = f.blocks.items[h].params;
    for (hparams, cfg.edgeOf(jump, 0).args) |p, a| sig[p] = a;
    const hops = f.blocks.items[h].ops.items;
    try latch_ops.ensureUnusedCapacity(ia, hops.len);
    for (hops[0 .. hops.len - 1]) |o| {
        var cp = o;
        try opt.mapUses(&cp, ia, sig);
        if (o.def()) |d| {
            const nd = try c.newVReg(c.types.items[d]);
            ssa.setDef(&cp, nd);
            sig[d] = nd;
        }
        latch_ops.appendAssumeCapacity(cp);
    }
    var cargs: [2][]const ir.VReg = undefined;
    for (&cargs, edge_args) |*ca, ea| {
        const out = try ia.alloc(ir.VReg, ea.len);
        for (out, ea) |*o, v| o.* = sig[v];
        ca.* = out;
    }
    const cb: struct { cond: ir.CondCode, a: ir.VReg, b: ir.VReg } = switch (old) {
        .cbr => |x| .{ .cond = x.cond, .a = x.a, .b = x.b },
        .cbra => |x| .{ .cond = x.cond, .a = x.a, .b = x.b },
        else => unreachable,
    };
    try f.emit(cd.latch, if (cargs[0].len == 0 and cargs[1].len == 0)
        .{ .cbr = .{ .cond = cb.cond, .a = sig[cb.a], .b = sig[cb.b], .then = target[0], .@"else" = target[1] } }
    else
        .{ .cbra = .{ .cond = cb.cond, .a = sig[cb.a], .b = sig[cb.b], .then = target[0], .@"else" = target[1], .then_args = cargs[0], .else_args = cargs[1] } });
    for (hvals.items) |v| sig[v] = v;

    c.freq.items[h] = entries;
    if (entries != null and t_body != null and f_latch != null) {
        const g_body = @min(entries.?, t_body.?);
        const l_body = @min(t_body.? - g_body, f_latch.?);
        c.taken.items[h] = if (cd.body == 0) g_body else entries.? - g_body;
        c.taken.items[cd.latch] = if (cd.body == 0) l_body else f_latch.? - l_body;
    } else {
        c.taken.items[h] = null;
        c.taken.items[cd.latch] = null;
    }
}

/// Renames the uses of each region's values to its parameters in the region block's
/// dominator subtree (a walk of the dominator tree with a scoped map).
fn renameRegions(c: *opt.Ctx, regions: []const Region) opt.Error!void {
    const gpa = c.gpa;
    const ia = c.irAlloc();
    var g = try cfg.Cfg.init(gpa, c.f);
    defer g.deinit();
    const nb = c.f.blocks.items.len;
    const map = try gpa.alloc(ir.VReg, c.f.num_vregs);
    defer gpa.free(map);
    for (map, 0..) |*m, i| m.* = @intCast(i);
    // Regions starting at each block, as linked lists.
    const head = try gpa.alloc(u32, nb);
    defer gpa.free(head);
    @memset(head, none);
    const next = try gpa.alloc(u32, regions.len);
    defer gpa.free(next);
    for (regions, 0..) |r, i| {
        next[i] = head[r.block];
        head[r.block] = @intCast(i);
    }
    const Undo = struct { v: ir.VReg, old: ir.VReg };
    var undo: std.ArrayList(Undo) = .empty;
    defer undo.deinit(gpa);
    const Frame = struct { b: u32, child: u32, mark: usize };
    var stack: std.ArrayList(Frame) = .empty;
    defer stack.deinit(gpa);
    var b: u32 = 0;
    while (true) {
        // Enter `b`.
        try stack.append(gpa, .{ .b = b, .child = 0, .mark = undo.items.len });
        var r = head[b];
        while (r != none) : (r = next[r]) for (regions[r].vals, regions[r].params) |v, p| {
            try undo.append(gpa, .{ .v = v, .old = map[v] });
            map[v] = p;
        };
        for (c.f.blocks.items[b].ops.items) |*op| try opt.mapUses(op, ia, map);
        // Next block: first unvisited child of the deepest frame that has one.
        while (stack.items.len > 0) {
            const top = &stack.items[stack.items.len - 1];
            const ch = g.dom_children[top.b];
            if (top.child < ch.len) {
                b = ch[top.child];
                top.child += 1;
                break;
            }
            while (undo.items.len > top.mark) {
                const u = undo.pop().?;
                map[u.v] = u.old;
            }
            _ = stack.pop();
        } else return;
    }
}

// ---- tests -------------------------------------------------------------------------------

const t = std.testing;
const testgen = @import("testgen.zig");
/// Multiplies the iterations of the random-program tests (raise it for long fuzz runs).
const fuzz_scale = 1;
const licm = @import("licm.zig");

fn blockIndex(f: *const ir.Func, name: []const u8) u32 {
    for (f.blocks.items, 0..) |b, i| if (std.mem.eql(u8, b.name, name)) return @intCast(i);
    unreachable;
}

/// Parses `src`, builds SSA and runs `rotate` (and `licm` when `then_licm`) with checks;
/// compares with the original on `inputs` (`testgen.default_inputs` when empty). Caller
/// deinits `out.f` (the transformed function) and `out.c`.
const Rotated = struct {
    f: *ir.Func,
    c: opt.Ctx,
    options: *opt.Options,
    stats: *opt.Stats,
    changed: bool,

    fn deinit(r: *Rotated) void {
        r.c.deinit();
        r.f.deinit();
        t.allocator.destroy(r.f);
        t.allocator.destroy(r.options);
        t.allocator.destroy(r.stats);
    }
};

fn rotated(src: []const u8, profile: ?*const opt.Profile, then_licm: bool) !Rotated {
    const gpa = t.allocator;
    var ref = try ir.parse(gpa, src);
    defer ref.deinit();
    const f = try gpa.create(ir.Func);
    errdefer gpa.destroy(f);
    f.* = try ir.parse(gpa, src);
    errdefer f.deinit();
    const options = try gpa.create(opt.Options);
    errdefer gpa.destroy(options);
    options.* = .{ .verify_each = true, .profile = profile };
    const stats = try gpa.create(opt.Stats);
    errdefer gpa.destroy(stats);
    stats.* = .{};
    var c = try opt.Ctx.init(gpa, f, options, stats);
    errdefer c.deinit();
    try ssa.build(&c);
    try c.check("ssa.build");
    const changed = try rotate(&c);
    try c.check("rotate");
    if (then_licm) {
        _ = try licm.run(&c);
        try c.check("licm");
    }
    try testgen.expectSame(&ref, f, &testgen.default_inputs, 20_000);
    return .{ .f = f, .c = c, .options = options, .stats = stats, .changed = changed };
}

const Branches = struct { cond: u32, jump: u32, blocks: usize, loops: usize };

/// Number of `cbr`/`cbra` and of `br`/`bra` terminators in the blocks of loop `li`.
fn loopBranches(f: *const ir.Func, li: u32) !Branches {
    var g = try cfg.Cfg.init(t.allocator, f);
    defer g.deinit();
    var ls = try cfg.Loops.init(t.allocator, f, &g);
    defer ls.deinit();
    var r: Branches = .{ .cond = 0, .jump = 0, .blocks = 0, .loops = ls.loops.len };
    if (li >= ls.loops.len) return r;
    r.blocks = ls.loops[li].blocks.len;
    for (ls.loops[li].blocks) |b| {
        const ops = f.blocks.items[b].ops.items;
        switch (ops[ops.len - 1]) {
            .cbr, .cbra => r.cond += 1,
            .br, .bra => r.jump += 1,
            else => {},
        }
    }
    return r;
}

fn countOp(f: *const ir.Func, block: u32, comptime tag: std.meta.Tag(ir.Op)) usize {
    var n: usize = 0;
    for (f.blocks.items[block].ops.items) |o| n += @intFromBool(o == tag);
    return n;
}

const counted_src =
    \\func count 2
    \\e:
    \\  v2 = const 0
    \\  v3 = const 0
    \\  br h
    \\h:
    \\  cbr lt v3 v0 body done
    \\body:
    \\  v2 = add v2 v3
    \\  v4 = const 1
    \\  v3 = add v3 v4
    \\  br h
    \\done:
    \\  ret v2
;

test "loops: counted top-tested loop becomes one bottom-tested block" {
    var r = try rotated(counted_src, null, false);
    defer r.deinit();
    try t.expect(r.changed);
    const lb = try loopBranches(r.f, 0);
    try t.expectEqual(@as(usize, 1), lb.loops);
    try t.expectEqual(@as(usize, 1), lb.blocks); // the body block is its own latch
    try t.expectEqual(@as(u32, 1), lb.cond);
    try t.expectEqual(@as(u32, 0), lb.jump);
    // The guard (old header) is not in any loop anymore.
    try t.expect(cfg.terminator(r.f, blockIndex(r.f, "h")).* == .cbra);
    try t.expectEqual(@as(usize, 1), countOp(r.f, blockIndex(r.f, "body"), .cbra));
}

test "loops: rotation of a loop without block parameters or header values" {
    // Memory-carried state only: the header defines nothing, nothing needs a parameter.
    const src =
        \\func m 2
        \\e:
        \\  v5 = const 0
        \\  store v1 0 v5
        \\  br h
        \\h:
        \\  v2 = load v1 0
        \\  cbr ge v2 v0 done body
        \\body:
        \\  v3 = const 3
        \\  v4 = add v2 v3
        \\  store v1 0 v4
        \\  br h
        \\done:
        \\  v6 = load v1 0
        \\  ret v6
    ;
    var r = try rotated(src, null, false);
    defer r.deinit();
    try t.expect(r.changed);
    const lb = try loopBranches(r.f, 0);
    try t.expectEqual(@as(u32, 1), lb.cond);
    try t.expectEqual(@as(u32, 0), lb.jump);
    // The header's load is duplicated into the latch (one per iteration, as before).
    try t.expectEqual(@as(usize, 1), countOp(r.f, blockIndex(r.f, "body"), .load));
}

test "loops: header computing several values used in the body and after the loop" {
    const src =
        \\func s 2
        \\e:
        \\  v2 = const 0
        \\  v3 = const 1
        \\  br h
        \\h:
        \\  v4 = mul v2 v2
        \\  v5 = add v4 v3
        \\  v6 = xor v5 v0
        \\  v7 = const 9
        \\  cbr lt v2 v7 body done
        \\body:
        \\  v3 = add v3 v6
        \\  v8 = const 1
        \\  v2 = add v2 v8
        \\  br h
        \\done:
        \\  v9 = add v6 v5
        \\  v9 = add v9 v3
        \\  ret v9
    ;
    var r = try rotated(src, null, false);
    defer r.deinit();
    try t.expect(r.changed);
    const lb = try loopBranches(r.f, 0);
    try t.expectEqual(@as(u32, 1), lb.cond);
    try t.expectEqual(@as(u32, 0), lb.jump);
    // `done` receives the header values it reads as parameters.
    try t.expect(r.f.blocks.items[blockIndex(r.f, "done")].params.len >= 2);
}

test "loops: zero-trip loops and exits on the then edge" {
    const src =
        \\func z 2
        \\e:
        \\  v2 = const 0
        \\  v3 = const 5
        \\  br h
        \\h:
        \\  v4 = add v2 v3
        \\  cbr ge v2 v0 done body
        \\body:
        \\  v5 = const 2
        \\  v2 = add v2 v5
        \\  v3 = sub v3 v4
        \\  br h
        \\done:
        \\  v6 = mul v4 v3
        \\  ret v6
    ;
    var r = try rotated(src, null, false);
    defer r.deinit();
    try t.expect(r.changed);
    for ([_]i64{ -5, 0, 1, 2, 3, 8 }) |n| {
        var ref = try ir.parse(t.allocator, src);
        defer ref.deinit();
        try t.expectEqual(try ir.interpret(&ref, &.{ n, 0 }, 10_000), try ir.interpret(r.f, &.{ n, 0 }, 10_000));
    }
}

test "loops: skipped loops (call or too many ops in the header, bottom-tested, header value used past another exit)" {
    const call_src = try std.fmt.allocPrint(t.allocator,
        \\func k 2
        \\e:
        \\  v2 = const 0
        \\  br h
        \\h:
        \\  v3 = call 0x{x} v2 v0
        \\  cbr lt v2 v0 body done
        \\body:
        \\  v4 = const 1
        \\  v2 = add v2 v4
        \\  br h
        \\done:
        \\  ret v3
    , .{@intFromPtr(&testgen.callee)});
    defer t.allocator.free(call_src);
    var big: std.ArrayList(u8) = .empty;
    defer big.deinit(t.allocator);
    try big.appendSlice(t.allocator, "func b 2\ne:\n  v2 = const 0\n  br h\nh:\n");
    for (0..max_header_ops + 1) |_| try big.appendSlice(t.allocator, "  v3 = add v2 v0\n");
    try big.appendSlice(t.allocator, "  cbr lt v2 v0 body done\nbody:\n  v4 = const 1\n  v2 = add v2 v4\n  br h\ndone:\n  ret v3\n");
    const bottom_src =
        \\func d 2
        \\e:
        \\  v2 = const 0
        \\  v3 = const 1
        \\  cbr ge v2 v0 done loop
        \\loop:
        \\  v2 = add v2 v3
        \\  cbr lt v2 v0 loop done
        \\done:
        \\  ret v2
    ;
    // v3 (defined in h) is used in `join`, which both exits reach: not dominated by the
    // body entry nor by the exit edge.
    const join_src =
        \\func j 2
        \\e:
        \\  v2 = const 0
        \\  br h
        \\h:
        \\  v3 = add v2 v0
        \\  cbr lt v2 v0 body x
        \\body:
        \\  v4 = const 1
        \\  v2 = add v2 v4
        \\  cbr eq v2 v1 join h
        \\x:
        \\  br join
        \\join:
        \\  ret v3
    ;
    for ([_][]const u8{ call_src, big.items, bottom_src, join_src }) |src| {
        var r = try rotated(src, null, false);
        defer r.deinit();
        if (src.ptr == join_src.ptr) {
            // The conditional latch got merged into a latch block, but nothing rotated.
            try t.expect(r.changed);
            try t.expectEqual(@as(u32, 2), (try loopBranches(r.f, 0)).cond);
        } else try t.expect(!r.changed);
    }
}

test "loops: several latches are merged, then the loop rotates" {
    const src =
        \\func ml 2
        \\e:
        \\  v2 = const 0
        \\  v3 = const 0
        \\  br h
        \\h:
        \\  cbr lt v2 v0 body done
        \\body:
        \\  v4 = const 1
        \\  v2 = add v2 v4
        \\  v5 = const 3
        \\  v6 = and v2 v5
        \\  cbr eq v6 v4 odd h
        \\odd:
        \\  v3 = add v3 v2
        \\  br h
        \\done:
        \\  v7 = add v3 v2
        \\  ret v7
    ;
    var r = try rotated(src, null, false);
    defer r.deinit();
    try t.expect(r.changed);
    var g = try cfg.Cfg.init(t.allocator, r.f);
    defer g.deinit();
    var ls = try cfg.Loops.init(t.allocator, r.f, &g);
    defer ls.deinit();
    try t.expectEqual(@as(usize, 1), ls.loops.len);
    try t.expectEqual(@as(usize, 1), ls.loops[0].latches.len);
    // The merged latch ends in the copied header test; the guard is outside the loop.
    const latch = ls.loops[0].latches[0];
    try t.expect(cfg.terminator(r.f, latch).* == .cbra or cfg.terminator(r.f, latch).* == .cbr);
    try t.expect(!ls.contains(0, blockIndex(r.f, "h")));
}

test "loops: nested loops rotate inner and outer" {
    const src =
        \\func n 2
        \\e:
        \\  v2 = const 0
        \\  v5 = const 0
        \\  br oh
        \\oh:
        \\  cbr lt v2 v0 ob done
        \\ob:
        \\  v3 = const 0
        \\  br ih
        \\ih:
        \\  v6 = add v3 v2
        \\  cbr lt v3 v1 ib olatch
        \\ib:
        \\  v5 = add v5 v6
        \\  v4 = const 1
        \\  v3 = add v3 v4
        \\  br ih
        \\olatch:
        \\  v5 = xor v5 v6
        \\  v4 = const 1
        \\  v2 = add v2 v4
        \\  br oh
        \\done:
        \\  ret v5
    ;
    var r = try rotated(src, null, false);
    defer r.deinit();
    try t.expect(r.changed);
    var g = try cfg.Cfg.init(t.allocator, r.f);
    defer g.deinit();
    var ls = try cfg.Loops.init(t.allocator, r.f, &g);
    defer ls.deinit();
    try t.expectEqual(@as(usize, 2), ls.loops.len);
    // Both loops are bottom-tested: each header is its loop's (only) latch or the latch
    // ends in the loop's only exiting test.
    for (ls.loops) |lp| {
        try t.expectEqual(@as(usize, 1), lp.latches.len);
        const term = cfg.terminator(r.f, lp.latches[0]).*;
        try t.expect(term == .cbr or term == .cbra);
    }
    // The original headers are guards, outside their loops.
    try t.expect(!ls.contains(0, blockIndex(r.f, "ih")) and !ls.contains(1, blockIndex(r.f, "oh")));
    try t.expect(ls.contains(1, blockIndex(r.f, "ih"))); // the inner guard runs per outer iteration
    for ([_][2]i64{ .{ 3, 4 }, .{ 0, 4 }, .{ 3, 0 }, .{ -1, -1 }, .{ 7, 1 } }) |in| {
        var ref = try ir.parse(t.allocator, src);
        defer ref.deinit();
        try t.expectEqual(try ir.interpret(&ref, &in, 100_000), try ir.interpret(r.f, &in, 100_000));
    }
}

test "loops: rotation lets licm hoist a division from the body" {
    const src =
        \\func dv 2
        \\e:
        \\  v2 = const 0
        \\  v3 = const 0
        \\  br h
        \\h:
        \\  cbr lt v2 v0 body done
        \\body:
        \\  v4 = sdiv v0 v1
        \\  v3 = add v3 v4
        \\  v5 = const 1
        \\  v2 = add v2 v5
        \\  br h
        \\done:
        \\  ret v3
    ;
    // Without rotation the division block does not dominate the exiting header.
    {
        var o: opt.Options = .{ .verify_each = true };
        var st: opt.Stats = .{};
        var f = try ir.parse(t.allocator, src);
        defer f.deinit();
        var c = try opt.Ctx.init(t.allocator, &f, &o, &st);
        defer c.deinit();
        try ssa.build(&c);
        _ = try licm.run(&c);
        try c.check("licm");
        var divs: usize = 0;
        for (f.blocks.items[blockIndex(&f, "body")].ops.items) |op| divs += @intFromBool(op == .arith and op.arith.op == .sdiv);
        try t.expectEqual(@as(usize, 1), divs);
    }
    var r = try rotated(src, null, true);
    defer r.deinit();
    const body = blockIndex(r.f, "body");
    for (r.f.blocks.items[body].ops.items) |o| if (o == .arith) try t.expect(o.arith.op != .sdiv);
    try t.expect(r.stats.hoisted >= 2); // the division and the constant
}

test "loops: profile counts stay consistent after rotation" {
    // count(10, _): e once, h 11 times (10 into the body), body 10 times, done once.
    const p: opt.Profile = .{ .blocks = &.{ 1, 11, 10, 1 }, .taken = &.{ 0, 10, 0, 0 } };
    var r = try rotated(counted_src, &p, false);
    defer r.deinit();
    const h = blockIndex(r.f, "h");
    const body = blockIndex(r.f, "body");
    // Entered once: the guard runs once and enters the body; the latch loops 9 times.
    try t.expectEqual(@as(?u64, 1), r.c.freq.items[h]);
    try t.expectEqual(@as(?u64, 1), r.c.edgeFreq(h, 0));
    try t.expectEqual(@as(?u64, 0), r.c.edgeFreq(h, 1));
    try t.expectEqual(@as(?u64, 10), r.c.freq.items[body]);
    try t.expectEqual(@as(?u64, 9), r.c.edgeFreq(body, 0));
    try t.expectEqual(@as(?u64, 1), r.c.edgeFreq(body, 1));
}

test "loops: fuzz rotation (random programs with loops)" {
    const configs = [_]testgen.Config{
        .{},
        .{ .blocks = 10, .ops = 4 },
        .{ .blocks = 4, .ops = 10, .floats = false },
        .{ .blocks = 12, .ops = 3, .calls = false, .memory = false },
    };
    for (configs, 0..) |config, i| {
        try testgen.fuzzPass(t.allocator, 0x10075 + i, 12 * fuzz_scale, config, &.{}, rotate);
        try testgen.fuzzPass(t.allocator, 0x20075 + i, 8 * fuzz_scale, config, &.{rotate}, licm.run);
    }
}

test "loops: rotated random programs through the o2 pipeline" {
    var prng = std.Random.DefaultPrng.init(0x5eed_10);
    var src: std.ArrayList(u8) = .empty;
    defer src.deinit(t.allocator);
    for (0..20 * fuzz_scale) |i| {
        src.clearRetainingCapacity();
        try testgen.program(prng.random(), t.allocator, .{ .blocks = 8, .calls = i % 2 == 0 }, &src);
        _ = testgen.checkOptimize(t.allocator, src.items, .{ .level = .o2, .verify_each = true }) catch |e| {
            std.debug.print("o2 failure on:\n{s}\n", .{src.items});
            return e;
        };
    }
    _ = try testgen.checkOptimize(t.allocator, counted_src, .{ .level = .o2, .verify_each = true });
}

test "loops: huge profile counts saturate instead of overflowing" {
    // Two latches whose counts sum past maxInt(u64) (profiles are caller data, not checked).
    const src =
        \\func r 2
        \\e:
        \\  v2 = const 0
        \\  br h
        \\h:
        \\  v3 = const 6
        \\  cbr lt v2 v3 b x
        \\b:
        \\  v4 = const 1
        \\  v2 = add v2 v4
        \\  cbr eq v2 v4 h l2
        \\l2:
        \\  store v1 0 v2
        \\  br h
        \\x:
        \\  ret v2
    ;
    const big = std.math.maxInt(u64) / 2 + 10;
    const prof: opt.Profile = .{ .blocks = &.{ 1, big, big, big, 1 }, .taken = &.{ 0, big, big, 0, 0 } };
    var f = try ir.parse(t.allocator, src);
    defer f.deinit();
    _ = try opt.optimize(t.allocator, &f, .{ .level = .o2, .verify_each = true, .profile = &prof });
}
