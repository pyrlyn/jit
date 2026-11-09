//! SSA construction and verification (the SSA invariants are in opt.zig's module comment).
//!
//! `build` turns a function with mutable vregs (and possibly block parameters) into strict SSA
//! with block parameters: unreachable blocks are deleted, block 0 loses its predecessors and
//! parameters (`cfg.isolateEntry`), block parameters are placed at the iterated dominance
//! frontiers of the definitions of every vreg that is read in a block before being written
//! there (semi-pruned SSA, Briggs et al.), vregs are renamed in a dominator-tree walk (the
//! first definition keeps its number), reads of never-written values become zero constants at
//! the top of block 0, and block parameters that nothing reads are removed.
const std = @import("std");
const ir = @import("../ir.zig");
const opt = @import("opt.zig");
const cfg = @import("cfg.zig");

const none = cfg.none;

/// Uses of `o` (operands, call and edge arguments), via `scratch` when `o` has many.
pub fn usesOf(gpa: std.mem.Allocator, o: ir.Op, scratch: *std.ArrayList(ir.VReg)) std.mem.Allocator.Error![]const ir.VReg {
    try scratch.resize(gpa, @max(o.numUses(), ir.max_args));
    return o.uses(scratch.items);
}

/// Sets the result vreg of `o` (an op with a definition).
pub fn setDef(o: *ir.Op, v: ir.VReg) void {
    switch (o.*) {
        inline .@"const", .copy, .arith, .load, .call, .fconst, .typed, .fcmp, .cvt, .splat, .extract => |*x| x.dst = v,
        else => unreachable,
    }
}

/// Replaces the edge arguments of terminator `term` (`args[i]` for successor `i`).
pub fn setEdgeArgs(term: *ir.Op, args: [2][]const ir.VReg) void {
    switch (term.*) {
        .br, .bra => {
            const to = cfg.edgeOf(term.*, 0).to;
            term.* = if (args[0].len == 0) .{ .br = to } else .{ .bra = .{ .target = to, .args = args[0] } };
        },
        .cbr, .cbra => {
            const x: struct { cond: ir.CondCode, a: ir.VReg, b: ir.VReg, then: u32, @"else": u32 } = switch (term.*) {
                .cbr => |y| .{ .cond = y.cond, .a = y.a, .b = y.b, .then = y.then, .@"else" = y.@"else" },
                .cbra => |y| .{ .cond = y.cond, .a = y.a, .b = y.b, .then = y.then, .@"else" = y.@"else" },
                else => unreachable,
            };
            term.* = if (args[0].len == 0 and args[1].len == 0)
                .{ .cbr = .{ .cond = x.cond, .a = x.a, .b = x.b, .then = x.then, .@"else" = x.@"else" } }
            else
                .{ .cbra = .{ .cond = x.cond, .a = x.a, .b = x.b, .then = x.then, .@"else" = x.@"else", .then_args = args[0], .else_args = args[1] } };
        },
        else => unreachable,
    }
}

/// Number of successors of terminator `term` (both edges of a `cbr`/`cbra`, even when equal).
pub fn numSuccs(term: ir.Op) usize {
    return switch (term) {
        .br, .bra => 1,
        .cbr, .cbra => 2,
        else => 0,
    };
}

pub fn build(c: *opt.Ctx) opt.Error!void {
    std.debug.assert(!c.in_ssa);
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
    _ = try cfg.isolateEntry(c);
    var g = try cfg.Cfg.init(gpa, f);
    defer g.deinit();
    const df = try g.frontiers();
    const nb: u32 = @intCast(f.blocks.items.len);
    const nv: u32 = f.num_vregs; // original names; everything created below is an SSA name
    var scratch: std.ArrayList(ir.VReg) = .empty;
    defer scratch.deinit(gpa);

    // Definition sites (block, whether a block parameter) per vreg, and the "global" vregs:
    // read in some block before any definition in that block.
    const global = try gpa.alloc(bool, nv);
    defer gpa.free(global);
    @memset(global, false);
    const stamp = try gpa.alloc(u32, nv);
    defer gpa.free(stamp);
    const def_start = try gpa.alloc(u32, nv + 1);
    defer gpa.free(def_start);
    @memset(def_start, 0);
    var sites: Sites = .{ .stamp = stamp, .start = def_start };
    defer sites.deinit(gpa);
    for (0..2) |pass| { // pass 0 counts definitions per vreg (one per block), pass 1 records them
        @memset(stamp, none);
        for (f.blocks.items, 0..) |blk, bi| {
            const b: u32 = @intCast(bi);
            if (b == 0) for (0..f.params) |v| sites.note(@intCast(v), 0, false);
            for (blk.params) |p| sites.note(p, b, true);
            for (blk.ops.items) |o| {
                if (pass == 0) for (try usesOf(gpa, o, &scratch)) |u| {
                    if (stamp[u] != b) global[u] = true;
                };
                if (o.def()) |d| sites.note(d, b, false);
            }
        }
        if (pass == 0) try sites.startRecording(gpa);
    }
    const def_sites = sites.blocks;
    const def_param = sites.is_param;

    // New block parameters ("phis") at iterated dominance frontiers, as (block, vreg, name).
    const Phi = struct { b: u32, v: ir.VReg, name: ir.VReg };
    var phis: std.ArrayList(Phi) = .empty;
    defer phis.deinit(gpa);
    {
        const has = try gpa.alloc(u32, nb);
        defer gpa.free(has);
        const queued = try gpa.alloc(u32, nb);
        defer gpa.free(queued);
        @memset(has, none);
        @memset(queued, none);
        var work: std.ArrayList(u32) = .empty;
        defer work.deinit(gpa);
        for (0..nv) |vi| {
            const v: ir.VReg = @intCast(vi);
            const at = def_sites[def_start[v]..def_start[v + 1]];
            if (!global[v] or at.len == 0) continue;
            work.clearRetainingCapacity();
            for (at, def_param[def_start[v]..def_start[v + 1]]) |b, is_param| {
                if (is_param) has[b] = v; // the existing parameter already defines v there
                queued[b] = v;
                try work.append(gpa, b);
            }
            while (work.pop()) |x| for (df[x]) |y| {
                if (has[y] == v) continue;
                has[y] = v;
                try phis.append(gpa, .{ .b = y, .v = v, .name = try c.newVReg(c.types.items[v]) });
                if (queued[y] != v) {
                    queued[y] = v;
                    try work.append(gpa, y);
                }
            };
        }
    }
    // Group by block (stable: `phis` is in vreg order per block).
    const phi_start = try gpa.alloc(u32, nb + 1);
    defer gpa.free(phi_start);
    @memset(phi_start, 0);
    for (phis.items) |p| phi_start[p.b + 1] += 1;
    for (1..nb + 1) |i| phi_start[i] += phi_start[i - 1];
    const by_block = try gpa.alloc(Phi, phis.items.len);
    defer gpa.free(by_block);
    {
        const fill = try gpa.dupe(u32, phi_start[0..nb]);
        defer gpa.free(fill);
        for (phis.items) |p| {
            by_block[fill[p.b]] = p;
            fill[p.b] += 1;
        }
    }
    const phisOf = struct {
        fn get(start: []const u32, all: []const Phi, b: u32) []const Phi {
            return all[start[b]..start[b + 1]];
        }
    }.get;

    // Rename in a dominator-tree preorder walk.
    const cur = try gpa.alloc(ir.VReg, nv);
    const first_done = gpa.alloc(bool, nv) catch |err| {
        gpa.free(cur);
        return err;
    };
    var r: Renamer = .{ .c = c, .cur = cur, .first_done = first_done };
    defer r.deinit();
    @memset(r.cur, none);
    @memset(r.first_done, false);
    for (0..f.params) |v| r.first_done[v] = true;
    const Frame = struct { b: u32, next: u32, mark: usize };
    var stack: std.ArrayList(Frame) = .empty;
    defer stack.deinit(gpa);
    var next_block: u32 = 0;
    var mark: usize = 0;
    while (true) {
        if (next_block != none) {
            const b = next_block;
            const blk = &f.blocks.items[b];
            if (b == 0) for (0..f.params) |v| try r.define(@intCast(v), @intCast(v));
            const bp = phisOf(phi_start, by_block, b);
            if (blk.params.len + bp.len > ir.max_edge_args) return error.TooManyArgs;
            if (blk.params.len + bp.len != 0) {
                const params = try c.irAlloc().alloc(ir.VReg, blk.params.len + bp.len);
                for (blk.params, params[0..blk.params.len]) |p, *q| {
                    q.* = try r.newName(p);
                    try r.define(p, q.*);
                }
                for (bp, params[blk.params.len..]) |p, *q| {
                    q.* = p.name;
                    try r.define(p.v, p.name);
                }
                blk.params = params;
            }
            for (blk.ops.items) |*o| {
                for (try usesOf(gpa, o.*, &scratch)) |u| _ = try r.get(u);
                try opt.mapUses(o, c.irAlloc(), r.cur);
                if (o.def()) |d| {
                    const name = try r.newName(d);
                    setDef(o, name);
                    try r.define(d, name);
                }
            }
            // Arguments for the successors' new parameters, from the names at the block end.
            const term = &blk.ops.items[blk.ops.items.len - 1];
            var args: [2][]const ir.VReg = .{ &.{}, &.{} };
            var extended = false;
            for (0..numSuccs(term.*)) |i| {
                const e = cfg.edgeOf(term.*, @intCast(i));
                const sp = phisOf(phi_start, by_block, e.to);
                args[i] = e.args;
                if (sp.len == 0) continue;
                const a = try c.irAlloc().alloc(ir.VReg, e.args.len + sp.len);
                @memcpy(a[0..e.args.len], e.args);
                for (sp, a[e.args.len..]) |p, *x| x.* = try r.get(p.v);
                args[i] = a;
                extended = true;
            }
            if (extended) setEdgeArgs(term, args);
            try stack.append(gpa, .{ .b = b, .next = 0, .mark = mark });
        }
        if (stack.items.len == 0) break;
        const top = &stack.items[stack.items.len - 1];
        const ch = g.dom_children[top.b];
        if (top.next < ch.len) {
            next_block = ch[top.next];
            top.next += 1;
            mark = r.undo.items.len;
        } else {
            r.unwind(top.mark);
            _ = stack.pop();
            next_block = none;
            if (stack.items.len == 0) break;
        }
    }
    try f.blocks.items[0].ops.insertSlice(c.irAlloc(), 0, r.zero_ops.items);
    try pruneParams(c, &g);
    c.in_ssa = true;
}

/// Definition sites per vreg in CSR form: `blocks[start[v]..start[v + 1]]`, each block once.
const Sites = struct {
    stamp: []u32,
    start: []u32,
    blocks: []u32 = &.{},
    is_param: []bool = &.{},
    fill: []u32 = &.{},
    recording: bool = false,

    fn deinit(s: *Sites, gpa: std.mem.Allocator) void {
        gpa.free(s.blocks);
        gpa.free(s.is_param);
        gpa.free(s.fill);
    }
    fn note(s: *Sites, v: ir.VReg, b: u32, param: bool) void {
        if (s.stamp[v] == b) return;
        s.stamp[v] = b;
        if (!s.recording) {
            s.start[v + 1] += 1;
            return;
        }
        s.blocks[s.fill[v]] = b;
        s.is_param[s.fill[v]] = param;
        s.fill[v] += 1;
    }
    fn startRecording(s: *Sites, gpa: std.mem.Allocator) std.mem.Allocator.Error!void {
        const nv = s.start.len - 1;
        for (1..nv + 1) |i| s.start[i] += s.start[i - 1];
        s.blocks = try gpa.alloc(u32, s.start[nv]);
        s.is_param = try gpa.alloc(bool, s.start[nv]);
        s.fill = try gpa.dupe(u32, s.start[0..nv]);
        s.recording = true;
    }
};

const Renamer = struct {
    c: *opt.Ctx,
    /// Current SSA name of every original vreg (`none`: not written on this path, reads 0).
    cur: []ir.VReg,
    first_done: []bool,
    undo: std.ArrayList(struct { v: ir.VReg, old: ir.VReg }) = .empty,
    /// Zero constants (one per type), spliced into the top of block 0 at the end.
    zero_ops: std.ArrayList(ir.Op) = .empty,
    zeros: [@typeInfo(ir.Type).@"enum".field_names.len]ir.VReg = @splat(none),

    fn deinit(r: *Renamer) void {
        const gpa = r.c.gpa;
        gpa.free(r.cur);
        gpa.free(r.first_done);
        r.undo.deinit(gpa);
        r.zero_ops.deinit(gpa);
    }
    /// SSA name for a new definition of `v`: its own number the first time, then fresh ones.
    fn newName(r: *Renamer, v: ir.VReg) opt.Error!ir.VReg {
        if (!r.first_done[v]) {
            r.first_done[v] = true;
            return v;
        }
        return r.c.newVReg(r.c.types.items[v]);
    }
    fn define(r: *Renamer, v: ir.VReg, name: ir.VReg) opt.Error!void {
        try r.undo.append(r.c.gpa, .{ .v = v, .old = r.cur[v] });
        r.cur[v] = name;
    }
    /// Current name of `v`; a value never written on this path is the zero of its type, which
    /// is valid everywhere (block 0 dominates all blocks), so it needs no undo entry.
    fn get(r: *Renamer, v: ir.VReg) opt.Error!ir.VReg {
        if (r.cur[v] == none) r.cur[v] = try r.zero(r.c.types.items[v]);
        return r.cur[v];
    }
    fn zero(r: *Renamer, ty: ir.Type) opt.Error!ir.VReg {
        const slot = &r.zeros[@backingInt(ty)];
        if (slot.* == none) {
            if (ty.isVector()) _ = try r.zero(ty.lane()); // scalar first, in op order
            if (ty.isVector()) {
                const v = try r.c.newVReg(ty);
                try r.zero_ops.append(r.c.gpa, .{ .splat = .{ .ty = ty, .dst = v, .src = r.zeros[@backingInt(ty.lane())] } });
                slot.* = v;
            } else slot.* = try r.c.zeroValue(r.c.gpa, &r.zero_ops, ty);
        }
        return slot.*;
    }
    fn unwind(r: *Renamer, mark: usize) void {
        while (r.undo.items.len > mark) {
            const u = r.undo.pop().?;
            r.cur[u.v] = u.old;
        }
    }
};

/// Removes block parameters that no op reads (directly or through edges into live
/// parameters), with their edge arguments. `g` must describe the current CFG.
fn pruneParams(c: *opt.Ctx, g: *const cfg.Cfg) opt.Error!void {
    const gpa = c.gpa;
    const f = c.f;
    const n = f.num_vregs;
    const pblock = try gpa.alloc(u32, n);
    defer gpa.free(pblock);
    const pidx = try gpa.alloc(u32, n);
    defer gpa.free(pidx);
    const live = try gpa.alloc(bool, n);
    defer gpa.free(live);
    @memset(pblock, none);
    @memset(live, false);
    var any = false;
    for (f.blocks.items, 0..) |b, bi| for (b.params, 0..) |p, i| {
        pblock[p] = @intCast(bi);
        pidx[p] = @intCast(i);
        any = true;
    };
    if (!any) return;
    var work: std.ArrayList(ir.VReg) = .empty;
    defer work.deinit(gpa);
    var buf: [ir.max_args]ir.VReg = undefined;
    for (f.blocks.items) |b| for (b.ops.items) |o| {
        const direct: []const ir.VReg = switch (o) {
            .bra => &.{},
            .cbra => |x| &.{ x.a, x.b },
            else => o.uses(&buf),
        };
        for (direct) |u| if (pblock[u] != none and !live[u]) {
            live[u] = true;
            try work.append(gpa, u);
        };
    };
    while (work.pop()) |p| {
        const s = pblock[p];
        var prev: u32 = none;
        for (g.preds[s]) |pred| {
            if (pred == prev) continue;
            prev = pred;
            const ops = f.blocks.items[pred].ops.items;
            const term = ops[ops.len - 1];
            for (0..numSuccs(term)) |i| {
                const e = cfg.edgeOf(term, @intCast(i));
                if (e.to != s) continue;
                const a = e.args[pidx[p]];
                if (pblock[a] != none and !live[a]) {
                    live[a] = true;
                    try work.append(gpa, a);
                }
            }
        }
    }
    // Drop dead parameters and the matching arguments of every edge.
    const keep_args = struct {
        fn f_(a: std.mem.Allocator, args: []const ir.VReg, params: []const ir.VReg, alive: []const bool) std.mem.Allocator.Error![]const ir.VReg {
            var k: usize = 0;
            for (params) |p| k += @intFromBool(alive[p]);
            if (k == args.len) return args;
            const out = try a.alloc(ir.VReg, k);
            k = 0;
            for (args, params) |x, p| if (alive[p]) {
                out[k] = x;
                k += 1;
            };
            return out;
        }
    }.f_;
    for (f.blocks.items) |*b| {
        const ops = b.ops.items;
        const term = &ops[ops.len - 1];
        const ns = numSuccs(term.*);
        if (ns == 0) continue;
        var args: [2][]const ir.VReg = .{ &.{}, &.{} };
        var changed = false;
        for (0..ns) |i| {
            const e = cfg.edgeOf(term.*, @intCast(i));
            args[i] = try keep_args(c.irAlloc(), e.args, f.blocks.items[e.to].params, live);
            changed = changed or args[i].len != e.args.len;
        }
        if (changed) setEdgeArgs(term, args);
    }
    for (f.blocks.items) |*b| {
        var k: usize = 0;
        for (b.params) |p| k += @intFromBool(live[p]);
        if (k == b.params.len) continue;
        const params = try c.irAlloc().alloc(ir.VReg, k);
        k = 0;
        for (b.params) |p| if (live[p]) {
            params[k] = p;
            k += 1;
        };
        b.params = params;
    }
}

/// Strict SSA check used by `Ctx.check` while `c.in_ssa`: single definitions, no parameters
/// or predecessors on block 0, every read defined, and every use in a reachable block
/// dominated by its definition (edge arguments are uses at the end of their block).
pub fn verify(c: *opt.Ctx) opt.Error!void {
    const gpa = c.gpa;
    const f = c.f;
    try f.verifySsa(gpa);
    if (f.blocks.items[0].params.len != 0) return error.InvalidSsa;
    var g = try cfg.Cfg.init(gpa, f);
    defer g.deinit();
    if (g.preds[0].len != 0) return error.InvalidSsa;
    const def_block = try gpa.alloc(u32, f.num_vregs);
    defer gpa.free(def_block);
    const def_pos = try gpa.alloc(u32, f.num_vregs);
    defer gpa.free(def_pos);
    @memset(def_block, none);
    for (0..f.params) |v| {
        def_block[v] = 0;
        def_pos[v] = 0;
    }
    for (f.blocks.items, 0..) |b, bi| {
        for (b.params) |p| {
            def_block[p] = @intCast(bi);
            def_pos[p] = 0;
        }
        for (b.ops.items, 1..) |o, i| if (o.def()) |d| {
            def_block[d] = @intCast(bi);
            def_pos[d] = @intCast(i);
        };
    }
    var scratch: std.ArrayList(ir.VReg) = .empty;
    defer scratch.deinit(gpa);
    for (f.blocks.items, 0..) |b, bi| {
        const blk: u32 = @intCast(bi);
        if (!g.reachable(blk)) continue;
        for (b.ops.items, 1..) |o, i| for (try usesOf(gpa, o, &scratch)) |u| {
            const d = def_block[u];
            if (d == none) return error.InvalidSsa;
            const ok = if (d == blk) def_pos[u] < i else g.dominates(d, blk);
            if (!ok) return error.InvalidSsa;
        };
    }
}

// ---- tests -------------------------------------------------------------------------------

const t = std.testing;

fn buildAndCompare(src: []const u8, inputs: []const []const i64) !void {
    var f = try ir.parse(t.allocator, src);
    defer f.deinit();
    var ref = try ir.parse(t.allocator, src);
    defer ref.deinit();
    var o: opt.Options = .{ .verify_each = true };
    var st: opt.Stats = .{};
    var c = try opt.Ctx.init(t.allocator, &f, &o, &st);
    defer c.deinit();
    try build(&c);
    try c.check("ssa.build");
    for (inputs) |in| {
        const want = ir.interpret(&ref, in, 100_000) catch |e| {
            try t.expectEqual(error.StepLimit, e);
            continue;
        };
        try t.expectEqual(want, try ir.interpret(&f, in, 200_000));
    }
}

test "ssa: counted loop gets loop-carried block parameters" {
    const src =
        \\func sum 1
        \\e:
        \\  v1 = const 0
        \\  v2 = const 0
        \\  br h
        \\h:
        \\  cbr lt v2 v0 b x
        \\b:
        \\  v1 = add v1 v2
        \\  v3 = const 1
        \\  v2 = add v2 v3
        \\  br h
        \\x:
        \\  ret v1
    ;
    try buildAndCompare(src, &.{ &.{0}, &.{1}, &.{10}, &.{-3} });
    var f = try ir.parse(t.allocator, src);
    defer f.deinit();
    var o: opt.Options = .{};
    var st: opt.Stats = .{};
    var c = try opt.Ctx.init(t.allocator, &f, &o, &st);
    defer c.deinit();
    try build(&c);
    try t.expectEqual(@as(usize, 2), f.blocks.items[1].params.len); // h(v1, v2); x needs none
    try t.expectEqual(@as(usize, 0), f.blocks.items[3].params.len);
}

test "ssa: values unwritten on some path read typed zeros" {
    try buildAndCompare(
        \\func z 2
        \\e:
        \\  cbr lt v0 v1 a b
        \\a:
        \\  v5 = const 7
        \\  v6 = fconst.f64 2.5
        \\  v8 = splat.v2f64 v6
        \\  br j
        \\b:
        \\  v7 = add v9 v1
        \\  br j
        \\j:
        \\  v10 = cvt.i64.f64 v6
        \\  v11 = extract.v2f64 v8 1
        \\  v12 = cvt.i64.f64 v11
        \\  v13 = add v5 v10
        \\  v13 = add v13 v7
        \\  v13 = add v13 v12
        \\  ret v13
    , &.{ &.{ 0, 1 }, &.{ 1, 0 }, &.{ 5, 5 } });
}

test "ssa: existing block parameters, redefinitions, entry loop, unreachable blocks" {
    try buildAndCompare(
        \\func p 2
        \\e(v2):
        \\  v3 = const 1
        \\  v2 = add v2 v3
        \\  cbr lt v2 v0 e:v2 m:v2:v1
        \\dead:
        \\  v4 = const 99
        \\  br m v4 v4
        \\m(v4 v5):
        \\  v4 = add v4 v5
        \\  v6 = copy v4
        \\  cbr eq v6 v1 m:v6:v6 x
        \\x:
        \\  ret v4
    , &.{ &.{ 3, 1 }, &.{ -2, 5 }, &.{ 10, 3 }, &.{ 0, 0 } });
}

test "ssa: random mutable programs stay equivalent" {
    var prng = std.Random.DefaultPrng.init(0x55a_b111d);
    const rng = prng.random();
    var src: std.ArrayList(u8) = .empty;
    defer src.deinit(t.allocator);
    for (0..300) |_| {
        src.clearRetainingCapacity();
        const n = rng.intRangeAtMost(u32, 1, 9);
        const w = struct {
            fn print(o: *std.ArrayList(u8), comptime fmt: []const u8, args: anytype) !void {
                try o.print(t.allocator, fmt, args);
            }
        }.print;
        // v0, v1: params; v2..v7 i64 pool; v8..v9 f64 pool; v10 step counter; v11 one.
        try w(&src, "func r 2\n", .{});
        for (0..n) |b| {
            try w(&src, "b{d}:\n  v11 = const 1\n  v10 = add v10 v11\n", .{b});
            if (b == n - 1) try w(&src, "  v8 = fconst.f64 1.25\n  v9 = fconst.f64 -3.5\n", .{});
            for (0..rng.intRangeAtMost(u32, 0, 5)) |_| {
                const d = rng.intRangeAtMost(u32, 2, 7);
                const a = rng.intRangeAtMost(u32, 0, 7);
                const bb = rng.intRangeAtMost(u32, 0, 7);
                switch (rng.uintLessThan(u32, 7)) {
                    0 => try w(&src, "  v{d} = const {d}\n", .{ d, rng.intRangeAtMost(i64, -5, 5) }),
                    1 => try w(&src, "  v{d} = add v{d} v{d}\n", .{ d, a, bb }),
                    2 => try w(&src, "  v{d} = mul v{d} v{d}\n", .{ d, a, bb }),
                    3 => try w(&src, "  v{d} = copy v{d}\n", .{ d, a }),
                    4 => try w(&src, "  v{d} = cvt.f64.i64 v{d}\n", .{ rng.intRangeAtMost(u32, 8, 9), a }),
                    5 => try w(&src, "  v{d} = add.f64 v8 v9\n", .{rng.intRangeAtMost(u32, 8, 9)}),
                    else => try w(&src, "  v{d} = cvt.i64.f64 v{d}\n", .{ d, rng.intRangeAtMost(u32, 8, 9) }),
                }
            }
            switch (rng.uintLessThan(u32, 4)) {
                0 => try w(&src, "  ret v{d}\n", .{rng.intRangeAtMost(u32, 0, 7)}),
                1 => try w(&src, "  br b{d}\n", .{rng.uintLessThan(u32, n)}),
                2 => try w(&src, "  v11 = const {d}\n  cbr lt v10 v11 b{d} b{d}\n", .{ rng.intRangeAtMost(i64, 1, 12), rng.uintLessThan(u32, n), rng.uintLessThan(u32, n) }),
                else => try w(&src, "  cbr {s} v{d} v{d} b{d} b{d}\n", .{
                    @tagName(rng.enumValue(ir.CondCode)), rng.intRangeAtMost(u32, 0, 7), rng.intRangeAtMost(u32, 0, 7), rng.uintLessThan(u32, n), rng.uintLessThan(u32, n),
                }),
            }
        }
        buildAndCompare(src.items, &.{ &.{ 0, 0 }, &.{ 3, -1 }, &.{ -7, 9 }, &.{ 100, 2 } }) catch |e| {
            std.debug.print("ssa random failure on:\n{s}\n", .{src.items});
            return e;
        };
    }
}

fn optimizeWith(gpa: std.mem.Allocator, src: []const u8) !void {
    // The IR lives in its own allocator: only the optimizer's scratch allocations fail.
    var f = try ir.parse(std.heap.page_allocator, src);
    defer f.deinit();
    _ = try opt.optimize(gpa, &f, .{ .level = .o2, .vector = opt.VectorTarget.host, .verify_each = false });
}

test "ssa: no leak when an allocation fails anywhere in the o2 pipeline" {
    const src =
        \\func esc 3
        \\e:
        \\  v2 = const 0
        \\  v3 = const 8
        \\  v4 = const 1
        \\  v5 = const 0
        \\  br loop
        \\loop:
        \\  v6 = mul v2 v3
        \\  v7 = add v0 v6
        \\  v13 = add v12 v6
        \\  v8 = load v7 0
        \\  store v13 0 v8
        \\  v5 = xor v5 v8
        \\  v9 = mul v1 v3
        \\  v2 = add v2 v4
        \\  cbr lt v2 v1 loop done
        \\done:
        \\  v10 = add v5 v2
        \\  v11 = add v10 v9
        \\  v12 = add v11 v6
        \\  ret v12
    ;
    try t.checkAllAllocationFailures(t.allocator, optimizeWith, .{src});
}
