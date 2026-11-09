//! CFG simplification on SSA form: deletes unreachable blocks, removes trivial block
//! parameters, folds conditional branches whose two edges agree, threads edges through empty
//! blocks and merges blocks into their only predecessor.
//!
//! Each iteration of `run` (repeated while it changes something, at most `max_iterations`):
//! 1. Unreachable blocks are deleted (`cfg.Cfg.reachable`, `Ctx.compactBlocks`), so every
//!    block below is reachable and the dominance arguments hold.
//! 2. Trivial parameters (Braun et al.): a parameter whose incoming arguments are all the same
//!    value `V` or the parameter itself is `V` (V's definition then dominates the block), so it
//!    is dropped with its arguments and its uses read `V`. A worklist revisits the blocks that
//!    receive a dropped parameter as an edge argument, so chains collapse in one go.
//! 3. A sweep in postorder (successors first, so chains of empty blocks collapse in one sweep):
//!    - `cbr`/`cbra` whose edges enter the same block with the same arguments -> `br`/`bra`
//!      (`c.taken` becomes null);
//!    - a block ending in `br`/`bra S`, where `S` (not block 0, not itself) has no other
//!      incoming edge, absorbs `S`: `S`'s parameters read the edge's arguments, `S`'s ops are
//!      appended and `S`'s `taken` count moves along;
//!    - an empty block `E` (not block 0; only op `br T` / `bra T args`, `T != E`) is threaded:
//!      every edge into `E` enters `T` directly, with `E`'s arguments in which `E`'s parameters
//!      are replaced by the edge's own arguments. `E`'s parameters must have no use outside its
//!      terminator. Values `E` passes that are defined elsewhere dominate `E`, hence each of its
//!      predecessors, so SSA holds. Predecessors keep their `taken` counts (edge order kept).
//!
//! Renamed values (dropped and merged parameters) are tracked in a union-find map and all uses
//! are rewritten once at the end. Within the sweep, predecessor sets and use counts are kept up
//! to date incrementally, so every iteration is linear in the size of the function.
const std = @import("std");
const ir = @import("../ir.zig");
const opt = @import("opt.zig");
const cfg = @import("cfg.zig");
const ssa = @import("ssa.zig");

const none = cfg.none;

/// Bound on the outer iterations of `run` (each one is linear; usually two suffice).
const max_iterations = 8;

pub fn run(c: *opt.Ctx) opt.Error!bool {
    const gpa = c.gpa;
    // `alias[v]`: what `v` was replaced by (`v` itself if nothing); resolved by `find`.
    const alias = try gpa.alloc(ir.VReg, c.f.num_vregs);
    defer gpa.free(alias);
    for (alias, 0..) |*a, i| a.* = @intCast(i);
    var renamed = false;
    var changed = false;
    var iteration: u32 = 0;
    while (iteration < max_iterations) : (iteration += 1) {
        var step = false;
        var g = g: {
            var g0 = try cfg.Cfg.init(gpa, c.f);
            if (g0.rpo.len == c.f.blocks.items.len) break :g g0;
            defer g0.deinit();
            try removeUnreachable(c, &g0);
            step = true;
            break :g try cfg.Cfg.init(gpa, c.f);
        };
        defer g.deinit();
        if (try removeTrivialParams(c, &g, alias)) {
            step = true;
            renamed = true;
        }
        var s: Sweep = try .init(c, &g, alias);
        defer s.deinit();
        const swept = try s.run();
        renamed = renamed or s.renamed;
        changed = changed or step or swept;
        // Without sweep changes, phases 1 and 2 of another iteration would find nothing.
        if (!swept) break;
    }
    if (renamed) {
        for (0..alias.len) |v| _ = find(alias, @intCast(v));
        try opt.renameUses(c, alias);
    }
    return changed;
}

/// Representative of `v` in the alias forest (with path compression).
fn find(alias: []ir.VReg, v: ir.VReg) ir.VReg {
    var r = v;
    while (alias[r] != r) r = alias[r];
    var x = v;
    while (alias[x] != r) {
        const n = alias[x];
        alias[x] = r;
        x = n;
    }
    return r;
}

/// Deletes the blocks that `g` (the current CFG) marks unreachable.
fn removeUnreachable(c: *opt.Ctx, g: *const cfg.Cfg) opt.Error!void {
    const n = c.f.blocks.items.len;
    const keep = try c.gpa.alloc(bool, n);
    defer c.gpa.free(keep);
    for (keep, 0..) |*k, b| k.* = g.reachable(@intCast(b));
    try c.compactBlocks(keep);
}

/// Drops the parameter at `keep[i] == false` from the arguments of every edge `p -> b`.
fn dropEdgeArgs(c: *opt.Ctx, p: u32, b: u32, keep: []const bool) opt.Error!void {
    const term = cfg.terminator(c.f, p);
    for (0..ssa.numSuccs(term.*)) |i| {
        const e = cfg.edgeOf(term.*, @intCast(i));
        if (e.to != b) continue;
        var n: usize = 0;
        for (keep) |k| n += @intFromBool(k);
        const args = try c.irAlloc().alloc(ir.VReg, n);
        n = 0;
        for (e.args, keep) |a, k| if (k) {
            args[n] = a;
            n += 1;
        };
        cfg.setEdge(term, @intCast(i), b, args);
    }
}

/// Phase 2 (module comment). `g` describes the current CFG (all blocks reachable).
fn removeTrivialParams(c: *opt.Ctx, g: *const cfg.Cfg, alias: []ir.VReg) opt.Error!bool {
    const gpa = c.gpa;
    const f = c.f;
    const nb: u32 = @intCast(f.blocks.items.len);
    for (f.blocks.items) |blk| {
        if (blk.params.len != 0) break;
    } else return false;
    // Who receives a value as an edge argument (block and parameter index): linked lists per
    // representative vreg.
    const head = try gpa.alloc(u32, f.num_vregs);
    defer gpa.free(head);
    const tail = try gpa.alloc(u32, f.num_vregs);
    defer gpa.free(tail);
    @memset(head, none);
    var link: std.ArrayList(struct { block: u32, param: u32, next: u32 }) = .empty;
    defer link.deinit(gpa);
    for (f.blocks.items) |blk| {
        const term = blk.ops.items[blk.ops.items.len - 1];
        for (0..ssa.numSuccs(term)) |i| {
            const e = cfg.edgeOf(term, @intCast(i));
            for (e.args, 0..) |a, j| {
                const r = find(alias, a);
                const id: u32 = @intCast(link.items.len);
                try link.append(gpa, .{ .block = e.to, .param = @intCast(j), .next = none });
                if (head[r] == none) head[r] = id else link.items[tail[r]].next = id;
                tail[r] = id;
            }
        }
    }

    var work: std.ArrayList(u32) = .empty;
    defer work.deinit(gpa);
    const queued = try gpa.alloc(bool, nb);
    defer gpa.free(queued);
    @memset(queued, false);
    var i = g.rpo.len; // popped from the end: blocks in RPO
    while (i > 0) {
        i -= 1;
        const b = g.rpo[i];
        if (f.blocks.items[b].params.len == 0) continue;
        queued[b] = true;
        try work.append(gpa, b);
    }
    // Which parameters stay, per block at `off[b]..` (indices are the original ones: params
    // and edge arguments are compacted only after the fixpoint, so `link` stays valid).
    const off = try gpa.alloc(u32, nb + 1);
    defer gpa.free(off);
    off[0] = 0;
    for (f.blocks.items, 0..) |blk, b| off[b + 1] = off[b] + @as(u32, @intCast(blk.params.len));
    const keep_all = try gpa.alloc(bool, off[nb]);
    defer gpa.free(keep_all);
    @memset(keep_all, true);
    const dropped = try gpa.alloc(bool, nb);
    defer gpa.free(dropped);
    @memset(dropped, false);
    // Parameter indices of the current block still to (re)check.
    var local: std.ArrayList(u32) = .empty;
    defer local.deinit(gpa);
    while (work.pop()) |b| {
        queued[b] = false;
        const blk = &f.blocks.items[b];
        const keep = keep_all[off[b]..off[b + 1]];
        local.clearRetainingCapacity();
        var pi = blk.params.len; // popped from the end: in parameter order
        while (pi > 0) {
            pi -= 1;
            if (keep[pi]) try local.append(gpa, @intCast(pi));
        }
        // A dropped parameter that `b` receives can make only the parameters it is passed to
        // trivial: recheck just those (linear, also on long chains of self-fed parameters).
        while (local.pop()) |qi| {
            if (!keep[qi]) continue;
            const p = blk.params[qi];
            const v = trivialValue(c, g, alias, b, qi, p) orelse continue;
            keep[qi] = false;
            alias[p] = v;
            dropped[b] = true;
            // Blocks receiving `p` now receive `v`: recheck them; `v` inherits the list.
            var it = head[p];
            while (it != none) : (it = link.items[it].next) {
                const to = link.items[it].block;
                if (to == b) {
                    if (keep[link.items[it].param]) try local.append(gpa, link.items[it].param);
                } else if (!queued[to]) {
                    queued[to] = true;
                    try work.append(gpa, to);
                }
            }
            if (head[p] != none) {
                if (head[v] == none) head[v] = head[p] else link.items[tail[v]].next = head[p];
                tail[v] = tail[p];
                head[p] = none;
            }
        }
    }
    var changed = false;
    for (f.blocks.items, 0..) |*blk, b| {
        if (!dropped[b]) continue;
        changed = true;
        const k = keep_all[off[b]..off[b + 1]];
        var prev: u32 = none;
        for (g.preds[b]) |p| {
            if (p == prev) continue; // duplicates are adjacent
            prev = p;
            try dropEdgeArgs(c, p, @intCast(b), k);
        }
        var n: usize = 0;
        for (k) |x| n += @intFromBool(x);
        const params = try c.irAlloc().alloc(ir.VReg, n);
        n = 0;
        for (blk.params, k) |p, x| if (x) {
            params[n] = p;
            n += 1;
        };
        blk.params = params;
    }
    return changed;
}

/// The value parameter `p` (index `pi` of block `b`) always equals, or `null`.
fn trivialValue(c: *opt.Ctx, g: *const cfg.Cfg, alias: []ir.VReg, b: u32, pi: u32, p: ir.VReg) ?ir.VReg {
    var v: ir.VReg = none;
    var prev: u32 = none;
    for (g.preds[b]) |q| {
        if (q == prev) continue;
        prev = q;
        const term = cfg.terminator(c.f, q).*;
        for (0..ssa.numSuccs(term)) |i| {
            const e = cfg.edgeOf(term, @intCast(i));
            if (e.to != b) continue;
            const a = find(alias, e.args[pi]);
            if (a == p) continue;
            if (v == none) v = a else if (v != a) return null;
        }
    }
    return if (v == none) null else v;
}

/// Phase 3 (module comment): one postorder sweep over the blocks of `g`.
const Sweep = struct {
    c: *opt.Ctx,
    g: *const cfg.Cfg,
    alias: []ir.VReg,
    /// Incoming edges per block (exact, maintained).
    npred: []u32,
    /// Predecessors gained during the sweep (`g.preds` has the original ones; stale entries
    /// are filtered by checking the predecessor's terminator).
    extra: []std.ArrayList(u32),
    dead: []bool,
    /// Reads of every representative vreg (operands and edge arguments; maintained). Only
    /// needed (and allocated) when some empty block has parameters: `paramsLocal`.
    uses: []u32,
    /// Scratch: per-vreg counter / parameter position, kept at 0 / `none` between uses.
    cnt: []u32,
    pos: []u32,
    /// Scratch: distinct current predecessors, and a stamp per block to dedupe them.
    preds: std.ArrayList(u32) = .empty,
    stamp: []u32,
    epoch: u32 = 0,
    renamed: bool = false,

    fn init(c: *opt.Ctx, g: *const cfg.Cfg, alias: []ir.VReg) opt.Error!Sweep {
        const gpa = c.gpa;
        const nb = c.f.blocks.items.len;
        const nv = c.f.num_vregs;
        var s: Sweep = .{
            .c = c,
            .g = g,
            .alias = alias,
            .npred = &.{},
            .extra = &.{},
            .dead = &.{},
            .uses = &.{},
            .cnt = &.{},
            .pos = &.{},
            .stamp = &.{},
        };
        errdefer s.deinit();
        s.npred = try gpa.alloc(u32, nb);
        for (s.npred, g.preds) |*n, p| n.* = @intCast(p.len);
        s.extra = try gpa.alloc(std.ArrayList(u32), nb);
        @memset(s.extra, .empty);
        s.dead = try gpa.alloc(bool, nb);
        @memset(s.dead, false);
        s.stamp = try gpa.alloc(u32, nb);
        @memset(s.stamp, 0);
        // Blocks never lose ops other than a terminator replaced by another (folds, merges), so
        // the blocks `paramsLocal` will see have a single op (and parameters) from the start.
        const counting = for (c.f.blocks.items, 0..) |b, i| {
            if (i != 0 and b.params.len != 0 and b.ops.items.len == 1) break true;
        } else false;
        if (!counting) return s;
        s.uses = try gpa.alloc(u32, nv);
        @memset(s.uses, 0);
        s.cnt = try gpa.alloc(u32, nv);
        @memset(s.cnt, 0);
        s.pos = try gpa.alloc(u32, nv);
        @memset(s.pos, none);
        var scratch: [ir.max_args]ir.VReg = undefined;
        for (c.f.blocks.items) |b| for (b.ops.items) |op| switch (op) {
            .bra => |x| s.count(x.args, 1),
            .cbra => |x| {
                s.count(&.{ x.a, x.b }, 1);
                s.count(x.then_args, 1);
                s.count(x.else_args, 1);
            },
            else => s.count(op.uses(&scratch), 1),
        };
        return s;
    }

    fn deinit(s: *Sweep) void {
        const gpa = s.c.gpa;
        for (s.extra) |*l| l.deinit(gpa);
        gpa.free(s.extra);
        gpa.free(s.npred);
        gpa.free(s.dead);
        gpa.free(s.uses);
        gpa.free(s.cnt);
        gpa.free(s.pos);
        gpa.free(s.stamp);
        s.preds.deinit(gpa);
    }

    fn count(s: *Sweep, vs: []const ir.VReg, comptime delta: i2) void {
        if (s.uses.len == 0) return;
        for (vs) |v| {
            const r = find(s.alias, v);
            if (delta > 0) s.uses[r] += 1 else s.uses[r] -= 1;
        }
    }

    fn run(s: *Sweep) opt.Error!bool {
        var changed = false;
        var i = s.g.rpo.len;
        while (i > 0) {
            i -= 1;
            const b = s.g.rpo[i];
            if (s.dead[b]) continue;
            if (try s.block(b)) changed = true;
        }
        if (!changed) return false;
        const keep = try s.c.gpa.alloc(bool, s.dead.len);
        defer s.c.gpa.free(keep);
        for (keep, s.dead) |*k, d| k.* = !d;
        try s.c.compactBlocks(keep);
        return true;
    }

    fn block(s: *Sweep, b: u32) opt.Error!bool {
        const f = s.c.f;
        var changed = false;
        while (true) {
            if (s.foldSameTarget(b)) changed = true;
            const term = cfg.terminator(f, b).*;
            const to = switch (term) {
                .br => |to| to,
                .bra => |x| x.target,
                else => break,
            };
            if (to == b or to == 0 or s.npred[to] != 1) break;
            try s.merge(b, to);
            changed = true;
        }
        if (b != 0 and f.blocks.items[b].ops.items.len == 1) {
            const term = cfg.terminator(f, b).*;
            switch (term) {
                .br, .bra => if (cfg.edgeOf(term, 0).to != b and s.paramsLocal(b)) {
                    try s.thread(b);
                    changed = true;
                },
                else => {},
            }
        }
        return changed;
    }

    /// `cbr`/`cbra` of `b` with both edges entering one block with equal arguments -> `br`/`bra`.
    fn foldSameTarget(s: *Sweep, b: u32) bool {
        const term = cfg.terminator(s.c.f, b);
        switch (term.*) {
            .cbr => |x| {
                if (x.then != x.@"else") return false;
                s.count(&.{ x.a, x.b }, -1);
                term.* = .{ .br = x.then };
            },
            .cbra => |x| {
                if (x.then != x.@"else") return false;
                for (x.then_args, x.else_args) |p, q| if (find(s.alias, p) != find(s.alias, q)) return false;
                s.count(&.{ x.a, x.b }, -1);
                s.count(x.else_args, -1);
                term.* = .{ .bra = .{ .target = x.then, .args = x.then_args } };
            },
            else => return false,
        }
        s.npred[cfg.edgeOf(term.*, 0).to] -= 1;
        s.c.taken.items[b] = null;
        return true;
    }

    /// Appends block `m` (whose only incoming edge is `p`'s `br`/`bra`) to `p`.
    fn merge(s: *Sweep, p: u32, m: u32) opt.Error!void {
        const f = s.c.f;
        const args = cfg.edgeOf(cfg.terminator(f, p).*, 0).args;
        const params = f.blocks.items[m].params;
        s.count(args, -1);
        for (params, args) |x, a| {
            const r = find(s.alias, a);
            std.debug.assert(r != x); // `m` does not dominate `p` (both are reachable)
            s.alias[x] = r;
            if (s.uses.len != 0) {
                s.uses[r] += s.uses[x];
                s.uses[x] = 0;
            }
        }
        if (params.len != 0) s.renamed = true;
        const pb = &f.blocks.items[p];
        pb.ops.items.len -= 1;
        try pb.ops.appendSlice(s.c.irAlloc(), f.blocks.items[m].ops.items);
        f.blocks.items[m].params = &.{};
        s.dead[m] = true;
        s.npred[m] = 0;
        s.c.taken.items[p] = s.c.taken.items[m];
        var buf: [2]u32 = undefined;
        for (cfg.succs(f, p, &buf)) |x| try s.extra[x].append(s.c.gpa, p);
    }

    /// The parameters of `e` are read only by its own terminator.
    fn paramsLocal(s: *Sweep, e: u32) bool {
        const params = s.c.f.blocks.items[e].params;
        if (params.len == 0) return true;
        std.debug.assert(s.uses.len != 0);
        const args = cfg.edgeOf(cfg.terminator(s.c.f, e).*, 0).args;
        for (args) |a| s.cnt[find(s.alias, a)] += 1;
        var ok = true;
        for (params) |p| ok = ok and s.cnt[p] == s.uses[p];
        for (args) |a| s.cnt[find(s.alias, a)] = 0;
        return ok;
    }

    /// Distinct current predecessors of `b` (in `s.preds`).
    fn currentPreds(s: *Sweep, b: u32) opt.Error![]const u32 {
        s.epoch += 1;
        s.preds.clearRetainingCapacity();
        const lists: [2][]const u32 = .{ s.g.preds[b], s.extra[b].items };
        var buf: [2]u32 = undefined;
        for (lists) |l| for (l) |p| {
            if (s.dead[p] or s.stamp[p] == s.epoch) continue;
            s.stamp[p] = s.epoch;
            for (cfg.succs(s.c.f, p, &buf)) |x| if (x == b) {
                try s.preds.append(s.c.gpa, p);
                break;
            };
        };
        return s.preds.items;
    }

    /// Retargets every edge into the empty block `e` to its successor (module comment).
    fn thread(s: *Sweep, e: u32) opt.Error!void {
        const f = s.c.f;
        const out = cfg.edgeOf(cfg.terminator(f, e).*, 0);
        const params = f.blocks.items[e].params;
        for (params, 0..) |p, i| s.pos[p] = @intCast(i);
        defer for (params) |p| {
            s.pos[p] = none;
        };
        var refs = false; // `e`'s arguments mention its parameters
        if (params.len != 0) for (out.args) |a| {
            refs = refs or s.pos[find(s.alias, a)] != none;
        };
        s.count(out.args, -1);
        s.npred[out.to] -= 1;
        var buf: [2]u32 = undefined;
        for (try s.currentPreds(e)) |p| {
            const term = cfg.terminator(f, p);
            for (cfg.succs(f, p, &buf), 0..) |x, i| {
                if (x != e) continue;
                const in = cfg.edgeOf(term.*, @intCast(i)).args;
                s.count(in, -1);
                var args = out.args;
                if (refs) {
                    const a = try s.c.irAlloc().alloc(ir.VReg, out.args.len);
                    for (a, out.args) |*y, v| {
                        const k = s.pos[find(s.alias, v)];
                        y.* = if (k != none) in[k] else v;
                    }
                    args = a;
                }
                s.count(args, 1);
                cfg.setEdge(term, @intCast(i), out.to, args);
                s.npred[out.to] += 1;
            }
            try s.extra[out.to].append(s.c.gpa, p);
        }
        s.dead[e] = true;
        s.npred[e] = 0;
    }
};

// ---- tests -------------------------------------------------------------------------------

const t = std.testing;
const testgen = @import("testgen.zig");

/// Like `testgen.checkPassAfter` (`pre` passes, then `pass`, checked and compared with the
/// source on `testgen.default_inputs`), and a second `pass` must find nothing left to do.
/// Returns the transformed function (caller deinits it) and whether the first run changed it.
fn transform(src: []const u8, comptime pre: []const testgen.Pass, comptime pass: fn (*opt.Ctx) opt.Error!bool) !struct { ir.Func, bool } {
    const gpa = t.allocator;
    var ref = try ir.parse(gpa, src);
    defer ref.deinit();
    var f = try ir.parse(gpa, src);
    errdefer f.deinit();
    const options: opt.Options = .{ .verify_each = true };
    var stats: opt.Stats = .{};
    var c = try opt.Ctx.init(gpa, &f, &options, &stats);
    defer c.deinit();
    try ssa.build(&c);
    try c.check("ssa.build");
    inline for (pre) |p| {
        _ = try p(&c);
        try c.check("pre");
    }
    const changed = try pass(&c);
    try c.check("pass");
    try testgen.expectSame(&ref, &f, &testgen.default_inputs, 20_000);
    try t.expect(!try pass(&c));
    return .{ f, changed };
}

fn expectIr(f: *const ir.Func, needle: []const u8) !void {
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(t.allocator);
    try f.print(t.allocator, &out);
    if (std.mem.indexOf(u8, out.items, needle) == null) {
        std.debug.print("expected `{s}` in:\n{s}\n", .{ needle, out.items });
        return error.TestExpectedEqual;
    }
}

/// Test pre-pass standing in for sccp: a conditional branch comparing a value with itself
/// takes its statically known edge.
fn foldBranches(c: *opt.Ctx) opt.Error!bool {
    var changed = false;
    for (c.f.blocks.items, 0..) |*blk, b| {
        const term = &blk.ops.items[blk.ops.items.len - 1];
        const x: struct { cond: ir.CondCode, a: ir.VReg, b: ir.VReg } = switch (term.*) {
            inline .cbr, .cbra => |y| .{ .cond = y.cond, .a = y.a, .b = y.b },
            else => continue,
        };
        if (x.a != x.b) continue;
        const e = cfg.edgeOf(term.*, if (ir.cond(x.cond, 0, 0)) 0 else 1);
        term.* = if (e.args.len == 0) .{ .br = e.to } else .{ .bra = .{ .target = e.to, .args = e.args } };
        c.taken.items[b] = null;
        changed = true;
    }
    return changed;
}

test "simplifycfg: unreachable blocks are deleted, then chains merge" {
    var r = try transform(
        \\func u 2
        \\e:
        \\  v2 = add v0 v1
        \\  cbr le v2 v2 a b
        \\a:
        \\  v3 = mul v2 v2
        \\  br c
        \\b:
        \\  v4 = sub v2 v0
        \\  br c
        \\c:
        \\  ret v2
    , &.{foldBranches}, run);
    defer r[0].deinit();
    try t.expect(r[1]);
    try t.expectEqual(@as(usize, 1), r[0].blocks.items.len);
    try expectIr(&r[0], "v3 = mul v2 v2\n  ret v2");
}

test "simplifycfg: merged blocks' parameters read the edge arguments" {
    var r = try transform(
        \\func m 2
        \\e:
        \\  v2 = add v0 v1
        \\  br a v2
        \\a(v3):
        \\  v4 = mul v3 v3
        \\  br b v4 v3
        \\b(v5 v6):
        \\  v7 = sub v5 v6
        \\  ret v7
    , &.{}, run);
    defer r[0].deinit();
    try t.expectEqual(@as(usize, 1), r[0].blocks.items.len);
    try expectIr(&r[0], "v4 = mul v2 v2\n  v7 = sub v4 v2\n  ret v7");
}

test "simplifycfg: threading through empty blocks; cbr edges to one block with different arguments" {
    var r = try transform(
        \\func th 2
        \\e:
        \\  cbr lt v0 v1 x y
        \\x:
        \\  br j v0
        \\y:
        \\  br j v1
        \\j(v2):
        \\  ret v2
    , &.{}, run);
    defer r[0].deinit();
    try t.expectEqual(@as(usize, 2), r[0].blocks.items.len);
    try expectIr(&r[0], "cbr lt v0 v1 j:v0 j:v1");
}

test "simplifycfg: both edges to one block with equal arguments become br, then merge" {
    var r = try transform(
        \\func eq 2
        \\e:
        \\  cbr lt v0 v1 x y:v0
        \\x:
        \\  br j v0
        \\y(v2):
        \\  br j v2
        \\j(v3):
        \\  v4 = add v3 v3
        \\  ret v4
    , &.{}, run);
    defer r[0].deinit();
    try t.expectEqual(@as(usize, 1), r[0].blocks.items.len);
    try expectIr(&r[0], "v4 = add v0 v0");
}

test "simplifycfg: trivial loop parameter removed, empty latch threaded into the header" {
    var r = try transform(
        \\func tp 2
        \\e:
        \\  v2 = const 0
        \\  v8 = const 5
        \\  br h v0 v2
        \\h(v3 v4):
        \\  v5 = const 1
        \\  v6 = add v4 v5
        \\  cbr lt v6 v8 l x
        \\l:
        \\  br h v3 v6
        \\x:
        \\  v7 = add v3 v6
        \\  ret v7
    , &.{}, run);
    defer r[0].deinit();
    try expectIr(&r[0], "h(v4):");
    try expectIr(&r[0], "cbr lt v6 v8 h:v6 x");
    try expectIr(&r[0], "v7 = add v0 v6");
    try t.expectEqual(@as(usize, 3), r[0].blocks.items.len);
}

test "simplifycfg: trivial parameters through nested loops (worklist) and self-arguments" {
    // v3 (outer) is always v0; the inner header passes v5 = v3 around, so it is v0 too.
    var r = try transform(
        \\func nl 2
        \\e:
        \\  v2 = const 0
        \\  v9 = const 3
        \\  br oh v0 v2
        \\oh(v3 v4):
        \\  br ih v3 v2
        \\ih(v5 v6):
        \\  v7 = const 1
        \\  v8 = add v6 v7
        \\  cbr lt v8 v9 ih:v5:v8 ol
        \\ol:
        \\  v10 = add v4 v7
        \\  cbr lt v10 v9 oh:v5:v10 x
        \\x:
        \\  v11 = add v5 v10
        \\  ret v11
    , &.{}, run);
    defer r[0].deinit();
    try expectIr(&r[0], "oh(v4):");
    try expectIr(&r[0], "ih(v6):");
    try expectIr(&r[0], "v11 = add v0 v10");
}

test "simplifycfg: empty latch with a parameter, header whose only op is a cbr" {
    var r = try transform(
        \\func sl 2
        \\e:
        \\  v2 = const 0
        \\  v9 = const 4
        \\  br h v2
        \\h(v3):
        \\  cbr lt v3 v9 body x
        \\body:
        \\  v4 = const 1
        \\  v5 = add v3 v4
        \\  br latch v5
        \\latch(v6):
        \\  br h v6
        \\x:
        \\  ret v3
    , &.{}, run);
    defer r[0].deinit();
    try t.expectEqual(@as(usize, 4), r[0].blocks.items.len);
    try expectIr(&r[0], "v5 = add v3 v4\n  br h v5");
}

test "simplifycfg: self loops are not merged; a block whose parameter is read elsewhere is not threaded" {
    var r = try transform(
        \\func ep 2
        \\e:
        \\  v8 = const 3
        \\  cbr lt v0 v1 E:v0 E:v1
        \\E(v2):
        \\  br T v8
        \\T(v3):
        \\  v4 = add v3 v2
        \\  v5 = const 1
        \\  v6 = sub v3 v5
        \\  cbr gt v6 v5 T:v6 x
        \\x:
        \\  ret v4
    , &.{}, run);
    defer r[0].deinit();
    try t.expect(!r[1]);
    try t.expectEqual(@as(usize, 4), r[0].blocks.items.len);
}

test "simplifycfg: profile counts follow merges, threading keeps predecessor counts" {
    const gpa = t.allocator;
    const src =
        \\func pr 2
        \\e:
        \\  v2 = const 3
        \\  br a
        \\a:
        \\  cbr lt v0 v2 b c
        \\b:
        \\  store v1 0 v0
        \\  br d
        \\c:
        \\  br d
        \\d:
        \\  ret v0
    ;
    var ref = try ir.parse(gpa, src);
    defer ref.deinit();
    var f = try ir.parse(gpa, src);
    defer f.deinit();
    const prof: opt.Profile = .{ .blocks = &.{ 10, 10, 4, 6, 10 }, .taken = &.{ 0, 4, 0, 0, 0 } };
    const options: opt.Options = .{ .verify_each = true, .profile = &prof };
    var stats: opt.Stats = .{};
    var c = try opt.Ctx.init(gpa, &f, &options, &stats);
    defer c.deinit();
    try ssa.build(&c);
    try t.expect(try run(&c));
    try c.check("simplifycfg");
    try testgen.expectSame(&ref, &f, &testgen.default_inputs, 20_000);
    try t.expectEqual(@as(usize, 3), f.blocks.items.len); // e+a, b, d
    try expectIr(&f, "cbr lt v0 v2 b d");
    try t.expectEqual(@as(?u64, 4), c.edgeFreq(0, 0));
    try t.expectEqual(@as(?u64, 6), c.edgeFreq(0, 1));
    try t.expectEqual(@as(?u64, 4), c.freq.items[1]);
    try t.expectEqual(@as(?u64, 10), c.freq.items[2]);

    // Both edges into one block: the cbr becomes a br (no taken count), then d merges.
    var g = try ir.parse(gpa,
        \\func pr2 1
        \\e:
        \\  cbr lt v0 v0 x y
        \\x:
        \\  br d
        \\y:
        \\  br d
        \\d:
        \\  v1 = const 1
        \\  cbr lt v0 v1 r s
        \\r:
        \\  ret v0
        \\s:
        \\  ret v1
    );
    defer g.deinit();
    const prof2: opt.Profile = .{ .blocks = &.{ 9, 0, 9, 9, 5, 4 }, .taken = &.{ 0, 0, 0, 5, 0, 0 } };
    const options2: opt.Options = .{ .verify_each = true, .profile = &prof2 };
    var c2 = try opt.Ctx.init(gpa, &g, &options2, &stats);
    defer c2.deinit();
    try ssa.build(&c2);
    try t.expect(try run(&c2));
    try c2.check("simplifycfg");
    try t.expectEqual(@as(usize, 3), g.blocks.items.len);
    try t.expectEqual(@as(?u64, 5), c2.edgeFreq(0, 0));
    try t.expectEqual(@as(?u64, 4), c2.edgeFreq(0, 1));
}

/// Random programs with many empty blocks (only a terminator) and join points: every cycle
/// passes a block that counts its executions in `v10` and leaves the cycle after a few rounds.
fn cfgProgram(rng: std.Random, gpa: std.mem.Allocator, out: *std.ArrayList(u8)) !void {
    const n = rng.intRangeAtMost(u32, 2, 12);
    try out.print(gpa, "func r 2\n", .{});
    for (0..n) |bi| {
        const b: u32 = @intCast(bi);
        try out.print(gpa, "b{d}:\n", .{b});
        const empty = b != 0 and rng.uintLessThan(u32, 3) != 0;
        if (!empty) {
            try out.print(gpa, "  v11 = const 1\n  v10 = add v10 v11\n", .{});
            for (0..rng.uintAtMost(u32, 3)) |_| try out.print(gpa, "  v{d} = {s} v{d} v{d}\n", .{
                rng.intRangeAtMost(u32, 2, 5), @tagName(rng.enumValue(enum { add, xor, sub, mul })), rng.uintAtMost(u32, 5), rng.uintAtMost(u32, 5),
            });
        }
        const fwd = struct {
            fn pick(r: std.Random, from: u32, count: u32) u32 {
                return r.intRangeLessThan(u32, from + 1, count);
            }
        }.pick;
        if (b + 1 == n or rng.uintLessThan(u32, 8) == 0) {
            try out.print(gpa, "  ret v{d}\n", .{rng.uintAtMost(u32, 5)});
        } else if (!empty and rng.uintLessThan(u32, 3) == 0) {
            try out.print(gpa, "  v11 = const {d}\n  cbr lt v10 v11 b{d} b{d}\n", .{ rng.intRangeAtMost(u32, 1, 6), rng.uintAtMost(u32, b), fwd(rng, b, n) });
        } else if (rng.boolean()) {
            try out.print(gpa, "  br b{d}\n", .{fwd(rng, b, n)});
        } else {
            const t1 = fwd(rng, b, n);
            const t2 = if (rng.uintLessThan(u32, 4) == 0) t1 else fwd(rng, b, n);
            const x = rng.uintAtMost(u32, 5);
            const y = if (rng.uintLessThan(u32, 3) == 0) x else rng.uintAtMost(u32, 5);
            try out.print(gpa, "  cbr {s} v{d} v{d} b{d} b{d}\n", .{ @tagName(rng.enumValue(ir.CondCode)), x, y, t1, t2 });
        }
    }
}

test "simplifycfg: random programs with empty blocks" {
    const gpa = t.allocator;
    var prng = std.Random.DefaultPrng.init(0x5cf6);
    var src: std.ArrayList(u8) = .empty;
    defer src.deinit(gpa);
    var blocks: [2]usize = .{ 0, 0 };
    for (0..400) |i| {
        src.clearRetainingCapacity();
        try cfgProgram(prng.random(), gpa, &src);
        var r = (if (i % 2 == 0) transform(src.items, &.{}, run) else transform(src.items, &.{foldBranches}, run)) catch |e| {
            std.debug.print("simplifycfg failure on:\n{s}\n", .{src.items});
            return e;
        };
        defer r[0].deinit();
        var g = try ir.parse(gpa, src.items);
        defer g.deinit();
        blocks[0] += g.blocks.items.len;
        blocks[1] += r[0].blocks.items.len;
    }
    try t.expect(2 * blocks[1] < blocks[0]);
}

test "simplifycfg: random programs (testgen)" {
    const gpa = t.allocator;
    const gvn = @import("gvn.zig");
    const configs = [_]testgen.Config{
        .{},
        .{ .blocks = 12 },
        .{ .blocks = 12, .ops = 2, .loops = false },
        .{ .blocks = 4, .ops = 12, .floats = false },
    };
    for (configs, 0..) |config, i| {
        try testgen.fuzzPass(gpa, 0xcf9 + i, 40, config, &.{}, run);
        try testgen.fuzzPass(gpa, 0x1cf9 + i, 40, config, &.{ foldBranches, gvn.run }, run);
    }
}

test "simplifycfg: long chains of loop parameters fed by each other (both orders)" {
    // Header parameters v10..v(10+n-1) where each one is passed the next: dropping one makes
    // the next trivial. Exercises the per-parameter worklist (it used to rescan the block).
    const gpa = t.allocator;
    var src: std.ArrayList(u8) = .empty;
    defer src.deinit(gpa);
    for ([_]bool{ false, true }) |reverse| {
        const n = 300;
        src.clearRetainingCapacity();
        try src.print(gpa, "func r 2\ne:\n", .{});
        for (0..n) |i| try src.print(gpa, "  v{d} = copy v0\n", .{10 + i});
        try src.print(gpa, "  v2 = const 0\n  br h\nh:\n  v3 = const 3\n  cbr lt v2 v3 b x\nb:\n  v5 = const 1\n  v2 = add v2 v5\n", .{});
        for (0..n - 1) |k| {
            const i = if (reverse) n - 2 - k else k;
            try src.print(gpa, "  v{d} = copy v{d}\n", .{ 10 + i, 11 + i });
        }
        try src.print(gpa, "  br h\nx:\n  v6 = const 0\n", .{});
        for (0..n) |i| try src.print(gpa, "  v6 = xor v6 v{d}\n", .{10 + i});
        try src.print(gpa, "  ret v6\n", .{});
        _ = try testgen.checkOptimize(gpa, src.items, .{ .level = .o1, .verify_each = true });
    }
}
