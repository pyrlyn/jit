//! Loop-invariant code motion (in SSA). An op is invariant in a loop when every operand is
//! defined outside it (or by an op already hoisted out of it); invariant ops move to the end
//! of the loop's preheader, in their original relative order. Loops are processed innermost
//! first, so an op hoisted into an inner loop's preheader (a block of the outer loop) can move
//! again for the outer loop.
//!
//! Pure ops (const, fconst, copy, arith except sdiv/srem, typed, fcmp, cvt, splat, extract)
//! are hoisted whenever invariant. `sdiv`/`srem` and `load` may fault, so they are hoisted
//! only when they are certain to execute whenever the preheader does: their block is in the
//! loop but in no inner loop, dominates every latch and every exiting block, is not reached
//! from the header through an inner loop, the loop contains no `call` (which might not return)
//! and the function is reducible. Loads additionally require a loop without `store` (any store
//! may alias). Stores, calls, terminators and block parameters never move.
const std = @import("std");
const ir = @import("../ir.zig");
const opt = @import("opt.zig");
const cfg = @import("cfg.zig");
const loops = @import("loops.zig");

const none = cfg.none;

const Kind = enum { fixed, pure, div, load };

fn kind(o: ir.Op) Kind {
    return switch (o) {
        .@"const", .fconst, .copy, .typed, .fcmp, .cvt, .splat, .extract => .pure,
        .arith => |x| if (x.op == .sdiv or x.op == .srem) .div else .pure,
        .load => .load,
        else => .fixed,
    };
}

/// Every operand of `o` (a movable op: at most two operands) is defined outside the loop.
fn invariant(o: ir.Op, def_block: []const u32, in_loop: []const bool) bool {
    var buf: [ir.max_args]ir.VReg = undefined;
    for (o.uses(&buf)) |u| {
        const d = def_block[u];
        if (d == none or in_loop[d]) return false;
    }
    return true;
}

/// Some op of `blocks` (a loop marked in `in_loop`) can move: a chain of invariant ops starts
/// with one whose operands are all defined outside the loop.
fn hasCandidate(f: *const ir.Func, blocks: []const u32, def_block: []const u32, in_loop: []const bool) bool {
    for (blocks) |b| for (f.blocks.items[b].ops.items) |o| {
        if (kind(o) != .fixed and invariant(o, def_block, in_loop)) return true;
    };
    return false;
}

/// Hoists loop-invariant ops (see the module comment). Creates a preheader
/// (`cfg.ensurePreheader`) for each loop that lacks one and has an invariant candidate.
/// Returns true iff the function changed.
pub fn run(c: *opt.Ctx) opt.Error!bool {
    const gpa = c.gpa;
    var changed = false;
    // Preheaders first (the CFG snapshot stays valid for the other loops: `ensurePreheader`
    // only redirects edges into its own loop's header), then fresh analyses that include them.
    {
        var g = try cfg.Cfg.init(gpa, c.f);
        defer g.deinit();
        var ls = try cfg.Loops.init(gpa, c.f, &g);
        defer ls.deinit();
        if (ls.loops.len == 0) return false;
        var missing = false;
        for (ls.loops) |lp| missing = missing or lp.preheader == none;
        if (missing) {
            const def_block = try loops.defBlocks(gpa, c.f);
            defer gpa.free(def_block);
            const in_loop = try gpa.alloc(bool, c.f.blocks.items.len);
            defer gpa.free(in_loop);
            @memset(in_loop, false);
            var want: std.ArrayList(u32) = .empty;
            defer want.deinit(gpa);
            for (ls.loops, 0..) |lp, li| {
                if (lp.preheader != none) continue;
                for (lp.blocks) |b| in_loop[b] = true;
                if (hasCandidate(c.f, lp.blocks, def_block, in_loop)) try want.append(gpa, @intCast(li));
                for (lp.blocks) |b| in_loop[b] = false;
            }
            for (want.items) |li| _ = try cfg.ensurePreheader(c, &g, &ls, li);
            changed = want.items.len != 0;
        }
    }

    var g = try cfg.Cfg.init(gpa, c.f);
    defer g.deinit();
    var ls = try cfg.Loops.init(gpa, c.f, &g);
    defer ls.deinit();
    const nb = c.f.blocks.items.len;
    const def_block = try loops.defBlocks(gpa, c.f);
    defer gpa.free(def_block);
    const in_loop = try gpa.alloc(bool, nb);
    defer gpa.free(in_loop);
    @memset(in_loop, false);
    // `safe[b]`: a faulting op in b may be hoisted out of the loop being processed.
    const safe = try gpa.alloc(bool, nb);
    defer gpa.free(safe);
    @memset(safe, false);
    const seen = try gpa.alloc(bool, nb);
    defer gpa.free(seen);
    @memset(seen, false);
    const counted = try gpa.alloc(bool, c.f.num_vregs);
    defer gpa.free(counted);
    @memset(counted, false);
    var order: std.ArrayList(u32) = .empty;
    defer order.deinit(gpa);
    var moved: std.ArrayList(ir.Op) = .empty;
    defer moved.deinit(gpa);
    var work: std.ArrayList(u32) = .empty;
    defer work.deinit(gpa);
    var path: std.ArrayList(u32) = .empty;
    defer path.deinit(gpa);

    for (ls.loops, 0..) |lp, li_| {
        const li: u32 = @intCast(li_);
        const ph = lp.preheader;
        if (ph == none) continue; // nothing invariant when it was created
        for (lp.blocks) |b| in_loop[b] = true;
        defer for (lp.blocks) |b| {
            in_loop[b] = false;
        };
        var has_call = false;
        var has_store = false;
        for (lp.blocks) |b| for (c.f.blocks.items[b].ops.items) |o| switch (o) {
            .call => has_call = true,
            .store => has_store = true,
            else => {},
        };
        // Blocks in dominance order, so an op's invariant operands are decided before it.
        order.clearRetainingCapacity();
        try order.appendSlice(gpa, lp.blocks);
        std.mem.sort(u32, order.items, &g, struct {
            fn lt(gg: *const cfg.Cfg, x: u32, y: u32) bool {
                return gg.rpo_index[x] < gg.rpo_index[y];
            }
        }.lt);
        var safe_done = false;
        defer if (safe_done) for (lp.blocks) |b| {
            safe[b] = false;
            seen[b] = false;
        };
        moved.clearRetainingCapacity();
        for (order.items) |b| {
            const ops = &c.f.blocks.items[b].ops;
            var w: usize = 0;
            for (ops.items) |o| {
                const k = kind(o);
                const hoist = k != .fixed and invariant(o, def_block, in_loop) and switch (k) {
                    .pure => true,
                    .div, .load => blk: {
                        if (has_call or ls.irreducible or (k == .load and has_store)) break :blk false;
                        if (!safe_done) {
                            try guaranteed(gpa, &g, &ls, li, in_loop, safe, seen, &path, &work);
                            safe_done = true;
                        }
                        break :blk safe[b];
                    },
                    .fixed => unreachable,
                };
                if (hoist) {
                    const d = o.def().?;
                    try moved.append(gpa, o);
                    def_block[d] = ph;
                    if (!counted[d]) c.stats.hoisted += 1;
                    counted[d] = true;
                } else {
                    ops.items[w] = o;
                    w += 1;
                }
            }
            ops.shrinkRetainingCapacity(w);
        }
        if (moved.items.len == 0) continue;
        const pops = &c.f.blocks.items[ph].ops;
        try pops.insertSlice(c.irAlloc(), pops.items.len - 1, moved.items);
        changed = true;
    }
    return changed;
}

/// Sets `safe[x]` for the blocks x of loop `li` that execute whenever its header does (before
/// the loop is left or repeated): the blocks on the dominator-tree path from the header to the
/// nearest common dominator of all latches and exiting blocks, from the top down, as long as
/// they belong to no inner loop and no inner-loop block reaches them from the header without
/// passing it (an inner loop might not terminate). `seen` marks the blocks found by the
/// backward searches from those blocks to the header; the set only grows along the path, so
/// the searches share it. `safe` and `seen` must be false on the loop's blocks (the caller
/// resets them).
fn guaranteed(
    gpa: std.mem.Allocator,
    g: *const cfg.Cfg,
    ls: *const cfg.Loops,
    li: u32,
    in_loop: []const bool,
    safe: []bool,
    seen: []bool,
    path: *std.ArrayList(u32),
    work: *std.ArrayList(u32),
) std.mem.Allocator.Error!void {
    const lp = ls.loops[li];
    var d = lp.latches[0];
    for (lp.latches[1..]) |l| d = loops.commonDominator(g, d, l);
    for (lp.exits) |e| d = loops.commonDominator(g, d, e.from);
    path.clearRetainingCapacity();
    while (true) : (d = g.idom[d]) {
        try path.append(gpa, d);
        if (d == lp.header) break;
    }
    var i = path.items.len;
    while (i > 0) {
        i -= 1;
        const x = path.items[i];
        if (ls.loop_of[x] != li) return;
        if (!seen[x]) {
            seen[x] = true;
            work.clearRetainingCapacity();
            try work.append(gpa, x);
            while (work.pop()) |b| {
                if (b == lp.header) continue;
                for (g.preds[b]) |p| {
                    if (!in_loop[p] or seen[p]) continue; // outside: unreachable predecessors
                    if (ls.loop_of[p] != li) return;
                    seen[p] = true;
                    try work.append(gpa, p);
                }
            }
        }
        safe[x] = true;
    }
}

// ---- tests -------------------------------------------------------------------------------

const t = std.testing;
const ssa = @import("ssa.zig");
/// Multiplies the iterations of the random-program tests (raise it for long fuzz runs).
const fuzz_scale = 1;
const testgen = @import("testgen.zig");

fn blockIndex(f: *const ir.Func, name: []const u8) u32 {
    for (f.blocks.items, 0..) |b, i| if (std.mem.eql(u8, b.name, name)) return @intCast(i);
    unreachable;
}

fn countArith(f: *const ir.Func, block: u32, op: ir.Arith) usize {
    var n: usize = 0;
    for (f.blocks.items[block].ops.items) |o| n += @intFromBool(o == .arith and o.arith.op == op);
    return n;
}

fn countTag(f: *const ir.Func, block: u32, comptime tag: std.meta.Tag(ir.Op)) usize {
    var n: usize = 0;
    for (f.blocks.items[block].ops.items) |o| n += @intFromBool(o == tag);
    return n;
}

/// `src` after SSA construction and `pre` passes, then licm (all checked, compared with the
/// original on `testgen.default_inputs`).
const Hoisted = struct {
    f: ir.Func,
    options: opt.Options,
    stats: opt.Stats = .{},
    changed: bool = false,
    freq: []?u64 = &.{},
    taken: []?u64 = &.{},

    fn init(h: *Hoisted, src: []const u8, profile: ?*const opt.Profile, comptime pre: []const testgen.Pass) !void {
        h.* = .{ .f = try ir.parse(t.allocator, src), .options = .{ .verify_each = true, .profile = profile } };
        errdefer h.f.deinit();
        var ref = try ir.parse(t.allocator, src);
        defer ref.deinit();
        var c = try opt.Ctx.init(t.allocator, &h.f, &h.options, &h.stats);
        defer c.deinit();
        try ssa.build(&c);
        try c.check("ssa.build");
        inline for (pre) |p| {
            _ = try p(&c);
            try c.check("pre");
        }
        h.changed = try run(&c);
        try c.check("licm");
        try testgen.expectSame(&ref, &h.f, &testgen.default_inputs, 20_000);
        h.freq = try t.allocator.dupe(?u64, c.freq.items);
        errdefer t.allocator.free(h.freq);
        h.taken = try t.allocator.dupe(?u64, c.taken.items);
    }
    fn deinit(h: *Hoisted) void {
        t.allocator.free(h.freq);
        t.allocator.free(h.taken);
        h.f.deinit();
    }
    /// The single block whose name starts with `prefix`.
    fn named(h: *const Hoisted, prefix: []const u8) u32 {
        var r: u32 = none;
        for (h.f.blocks.items, 0..) |b, i| if (std.mem.startsWith(u8, b.name, prefix)) {
            std.debug.assert(r == none);
            r = @intCast(i);
        };
        return r;
    }
};

test "licm: invariant chain out of a top-tested loop into the existing preheader" {
    var h: Hoisted = undefined;
    try h.init(
        \\func a 2
        \\e:
        \\  v2 = const 0
        \\  v3 = const 0
        \\  br h
        \\h:
        \\  cbr lt v2 v0 body done
        \\body:
        \\  v4 = mul v0 v1
        \\  v5 = add v4 v1
        \\  v6 = fconst.f64 1.5
        \\  v7 = cvt.f64.i64 v5
        \\  v7 = mul.f64 v7 v6
        \\  v8 = cvt.i64.f64 v7
        \\  v3 = add v3 v8
        \\  v9 = const 1
        \\  v2 = add v2 v9
        \\  br h
        \\done:
        \\  ret v3
    , null, &.{});
    defer h.deinit();
    try t.expect(h.changed);
    const body = blockIndex(&h.f, "body");
    try t.expectEqual(@as(u32, 7), h.stats.hoisted); // mul, add, fconst, cvt, mul.f64, cvt, const
    try t.expectEqual(@as(usize, 2), h.f.blocks.items[body].ops.items.len - 1); // the two adds
    try t.expectEqual(@as(usize, 1), countArith(&h.f, 0, .mul));
    try t.expectEqual(@as(usize, 1), countTag(&h.f, 0, .typed));
    try t.expectEqual(@as(usize, 2), h.f.blocks.items.len - 2); // no block added
}

test "licm: bottom-tested loop gets a preheader; division and load are hoisted" {
    var h: Hoisted = undefined;
    try h.init(
        \\func b 2
        \\e:
        \\  v2 = const 0
        \\  v3 = const 0
        \\  cbr ge v2 v0 done loop
        \\loop:
        \\  v4 = load v1 8
        \\  v5 = sdiv v4 v0
        \\  v6 = srem v0 v5
        \\  v3 = add v3 v6
        \\  v7 = const 1
        \\  v2 = add v2 v7
        \\  cbr lt v2 v0 loop done
        \\done:
        \\  ret v3
    , null, &.{});
    defer h.deinit();
    const loop = blockIndex(&h.f, "loop");
    const ph = h.named("preheader");
    try t.expect(ph != none);
    try t.expectEqual(@as(usize, 0), countArith(&h.f, loop, .sdiv) + countArith(&h.f, loop, .srem) + countTag(&h.f, loop, .load));
    try t.expectEqual(@as(usize, 1), countArith(&h.f, ph, .sdiv));
    try t.expectEqual(@as(usize, 1), countArith(&h.f, ph, .srem));
    try t.expectEqual(@as(usize, 1), countTag(&h.f, ph, .load));
    try t.expectEqual(@as(u32, 4), h.stats.hoisted);
}

test "licm: no speculation of a division whose block does not dominate the exiting header" {
    var h: Hoisted = undefined;
    try h.init(
        \\func c 2
        \\e:
        \\  v2 = const 0
        \\  v3 = const 0
        \\  br h
        \\h:
        \\  v8 = sdiv v1 v0
        \\  cbr lt v2 v0 body done
        \\body:
        \\  v4 = sdiv v0 v1
        \\  v5 = load v1 0
        \\  v3 = add v3 v4
        \\  v3 = add v3 v5
        \\  v6 = const 1
        \\  v2 = add v2 v6
        \\  br h
        \\done:
        \\  v3 = add v3 v8
        \\  ret v3
    , null, &.{});
    defer h.deinit();
    const body = blockIndex(&h.f, "body");
    try t.expectEqual(@as(usize, 1), countArith(&h.f, body, .sdiv));
    try t.expectEqual(@as(usize, 1), countTag(&h.f, body, .load));
    // The header's division runs whenever the loop is entered.
    try t.expectEqual(@as(usize, 0), countArith(&h.f, blockIndex(&h.f, "h"), .sdiv));
    try t.expectEqual(@as(usize, 1), countArith(&h.f, 0, .sdiv));
}

test "licm: loads stay when the loop stores or calls; divisions stay when it calls" {
    const fmt =
        \\func m 2
        \\e:
        \\  v2 = const 0
        \\  v3 = const 0
        \\  cbr ge v2 v0 done loop
        \\loop:
        \\  v4 = load v1 0
        \\  v5 = sdiv v0 v4
        \\  v3 = add v3 v5
        \\  {s}
        \\  v7 = const 1
        \\  v2 = add v2 v7
        \\  cbr lt v2 v0 loop done
        \\done:
        \\  ret v3
    ;
    const call_line = try std.fmt.allocPrint(t.allocator, "v6 = call 0x{x} v2 v3", .{@intFromPtr(&testgen.callee)});
    defer t.allocator.free(call_line);
    // { extra op, load hoisted, division hoisted }
    const cases = [_]struct { []const u8, bool, bool }{
        .{ "store v1 16 v3", false, false }, // the division reads the load's result
        .{ "store v1 0 v3", false, false },
        .{ call_line, false, false },
        .{ "v6 = add v3 v2", true, true },
    };
    for (cases) |cs| {
        const src = try std.fmt.allocPrint(t.allocator, fmt, .{cs[0]});
        defer t.allocator.free(src);
        var h: Hoisted = undefined;
        try h.init(src, null, &.{});
        defer h.deinit();
        const loop = blockIndex(&h.f, "loop");
        try t.expectEqual(@as(usize, @intFromBool(!cs[1])), countTag(&h.f, loop, .load));
        try t.expectEqual(@as(usize, @intFromBool(!cs[2])), countArith(&h.f, loop, .sdiv));
    }
    // A division with invariant operands is hoisted from a loop that stores (no call).
    var h: Hoisted = undefined;
    try h.init(
        \\func s 2
        \\e:
        \\  v2 = const 0
        \\  cbr ge v2 v0 done loop
        \\loop:
        \\  v5 = srem v1 v0
        \\  store v1 0 v5
        \\  v7 = const 1
        \\  v2 = add v2 v7
        \\  cbr lt v2 v0 loop done
        \\done:
        \\  ret v2
    , null, &.{});
    defer h.deinit();
    try t.expectEqual(@as(usize, 0), countArith(&h.f, blockIndex(&h.f, "loop"), .srem));
}

const nested_src =
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
    \\  cbr lt v3 v0 ib olatch
    \\ib:
    \\  v6 = mul v0 v1
    \\  v7 = add v6 v2
    \\  v5 = add v5 v7
    \\  v4 = const 1
    \\  v3 = add v3 v4
    \\  br ih
    \\olatch:
    \\  v8 = const 1
    \\  v2 = add v2 v8
    \\  br oh
    \\done:
    \\  ret v5
;

test "licm: nested loops hoist two levels" {
    var h: Hoisted = undefined;
    try h.init(nested_src, null, &.{});
    defer h.deinit();
    const ob = blockIndex(&h.f, "ob");
    const ib = blockIndex(&h.f, "ib");
    try t.expectEqual(@as(usize, 1), countArith(&h.f, 0, .mul)); // invariant in both loops
    try t.expectEqual(@as(usize, 1), countArith(&h.f, ob, .add)); // depends on the outer counter
    try t.expectEqual(@as(usize, 0), countArith(&h.f, ib, .mul));
    try t.expectEqual(@as(usize, 2), countArith(&h.f, ib, .add)); // the accumulator and counter
    // The constants of both latches/bodies end up in the entry block.
    try t.expectEqual(@as(usize, 0), countTag(&h.f, ib, .@"const") + countTag(&h.f, blockIndex(&h.f, "olatch"), .@"const"));
}

test "licm: division after an inner loop is not hoisted, before it is" {
    const fmt =
        \\func q 2
        \\e:
        \\  v2 = const 0
        \\  v3 = const 0
        \\  cbr ge v2 v0 done oh
        \\oh:
        \\  v4 = const 0
        \\  {s}
        \\  br ih
        \\ih:
        \\  v5 = const 1
        \\  v4 = add v4 v5
        \\  cbr lt v4 v1 ih ox
        \\ox:
        \\  {s}
        \\  v3 = add v3 v6
        \\  v7 = const 1
        \\  v2 = add v2 v7
        \\  cbr lt v2 v0 oh done
        \\done:
        \\  ret v3
    ;
    for ([_]bool{ false, true }) |before| {
        const div = "v6 = sdiv v0 v1";
        const src = try std.fmt.allocPrint(t.allocator, fmt, if (before) .{ div, "" } else .{ "", div });
        defer t.allocator.free(src);
        var h: Hoisted = undefined;
        try h.init(src, null, &.{});
        defer h.deinit();
        const blk = blockIndex(&h.f, if (before) "oh" else "ox");
        try t.expectEqual(@as(usize, @intFromBool(!before)), countArith(&h.f, blk, .sdiv));
    }
}

test "licm: division reached through an optional inner loop is not hoisted" {
    // `j` dominates the latch and the exit, but the inner loop (which might not terminate)
    // lies on a path from the header to it: hoisting could trap where the loop would spin.
    var h: Hoisted = undefined;
    try h.init(
        \\func r 2
        \\e:
        \\  v2 = const 0
        \\  v3 = const 0
        \\  cbr ge v2 v0 done oh
        \\oh:
        \\  cbr lt v1 v0 inner j
        \\inner:
        \\  v4 = const 1
        \\  v5 = add v5 v4
        \\  cbr lt v5 v1 inner j
        \\j:
        \\  v6 = sdiv v0 v1
        \\  v3 = add v3 v6
        \\  v7 = const 1
        \\  v2 = add v2 v7
        \\  cbr lt v2 v0 oh done
        \\done:
        \\  ret v3
    , null, &.{});
    defer h.deinit();
    try t.expectEqual(@as(usize, 1), countArith(&h.f, blockIndex(&h.f, "j"), .sdiv));
}

test "licm: irreducible function hoists pure ops only" {
    var h: Hoisted = undefined;
    try h.init(
        \\func i 2
        \\e:
        \\  v2 = const 0
        \\  v3 = const 0
        \\  cbr lt v0 v1 a b
        \\a:
        \\  v2 = add v2 v0
        \\  cbr lt v2 v1 b x
        \\b:
        \\  v2 = sub v2 v1
        \\  cbr lt v2 v0 a x
        \\x:
        \\  cbr ge v3 v0 done loop
        \\loop:
        \\  v4 = sdiv v0 v1
        \\  v5 = mul v0 v1
        \\  v3 = add v3 v4
        \\  v3 = add v3 v5
        \\  v6 = const 1
        \\  v3 = add v3 v6
        \\  cbr lt v3 v0 loop done
        \\done:
        \\  ret v3
    , null, &.{});
    defer h.deinit();
    const loop = blockIndex(&h.f, "loop");
    try t.expectEqual(@as(usize, 1), countArith(&h.f, loop, .sdiv));
    try t.expectEqual(@as(usize, 0), countArith(&h.f, loop, .mul));
}

test "licm: created preheaders carry the entry edge's profile count" {
    const src =
        \\func p 2
        \\e:
        \\  v2 = const 0
        \\  cbr ge v2 v0 done loop
        \\loop:
        \\  v3 = mul v0 v1
        \\  v2 = add v2 v3
        \\  v4 = const 1
        \\  v2 = add v2 v4
        \\  cbr lt v2 v0 loop done
        \\done:
        \\  ret v2
    ;
    const profile: opt.Profile = .{ .blocks = &.{ 3, 40, 3 }, .taken = &.{ 1, 38, 0 } };
    var h: Hoisted = undefined;
    try h.init(src, &profile, &.{});
    defer h.deinit();
    const ph = h.named("preheader");
    try t.expectEqual(@as(?u64, 2), h.freq[ph]); // the entry's else edge: 3 - 1
    try t.expectEqual(@as(?u64, 40), h.freq[blockIndex(&h.f, "loop")]);
    try t.expectEqual(@as(?u64, 1), h.taken[0]);
    try t.expectEqual(@as(?u64, 38), h.taken[blockIndex(&h.f, "loop")]);
    try t.expectEqual(@as(usize, 1), countArith(&h.f, ph, .mul));
}

test "licm: loops without invariant ops get no preheader" {
    var h: Hoisted = undefined;
    try h.init(
        \\func z 2
        \\e:
        \\  v2 = const 0
        \\  cbr ge v2 v0 done loop
        \\loop:
        \\  v2 = add v2 v2
        \\  v2 = add v2 v0
        \\  cbr lt v2 v0 loop done
        \\done:
        \\  ret v2
    , null, &.{});
    defer h.deinit();
    try t.expect(!h.changed);
    try t.expectEqual(@as(usize, 3), h.f.blocks.items.len);
}

test "licm: fuzz (random programs with loops)" {
    const configs = [_]testgen.Config{
        .{},
        .{ .blocks = 10, .ops = 5 },
        .{ .blocks = 3, .ops = 12, .calls = false },
        .{ .blocks = 8, .ops = 8, .calls = false, .floats = false },
    };
    for (configs, 0..) |config, i| try testgen.fuzzPass(t.allocator, 0x11c0 + i, 15 * fuzz_scale, config, &.{}, run);
}

// Speculation checks: `ir.interpret` cannot tell a hoisted faulting op from the original
// (a zero divisor reads 0, loads read any address), so these tests run a trapping
// interpreter on random programs whose divisions may divide by 0 or -1 and whose loads may
// read past the buffer: whenever the original does not trap, the result must not trap either
// and must compute the same result, memory and calls.

var strict_mem: [testgen.mem_slots]u64 align(16) = undefined;

const Strict = struct {
    result: anyerror!i64,
    mem: [testgen.mem_slots]u64,
    calls: [64][2]i64 = undefined,
    ncalls: usize = 0,
};

fn strictRun(f: *const ir.Func, input: i64, max_steps: usize) !Strict {
    var s: Strict = .{ .result = 0, .mem = undefined };
    for (&strict_mem, 0..) |*m, i| m.* = @as(u64, @bitCast(input)) *% 0x9e3779b97f4a7c15 +% i;
    s.result = strictExec(f, input, max_steps, &s);
    s.mem = strict_mem;
    return s;
}

fn strictAddr(base: ir.Val, disp: i32, size: u8) ?[]u8 {
    const addr = @as(u64, @bitCast(ir.toI(base) +% disp));
    const off = addr -% @intFromPtr(&strict_mem);
    if (off > @sizeOf(@TypeOf(strict_mem)) - size) return null;
    return std.mem.asBytes(&strict_mem)[@intCast(off)..][0..size];
}

fn strictExec(f: *const ir.Func, input: i64, max_steps: usize, s: *Strict) !i64 {
    var regs_buf: [2048]ir.Val = undefined;
    if (f.num_vregs > regs_buf.len) return error.TooManyVRegs;
    const regs = regs_buf[0..@max(f.num_vregs, 2)];
    @memset(regs, 0);
    var tmp: [ir.max_edge_args]ir.Val = undefined;
    regs[0] = ir.fromI(input);
    regs[1] = ir.fromI(@intCast(@intFromPtr(&strict_mem)));
    var b: u32 = 0;
    var steps: usize = 0;
    while (true) {
        for (f.blocks.items[b].ops.items) |o| {
            steps += 1;
            if (steps > max_steps) return error.StepLimit;
            switch (o) {
                .@"const" => |x| regs[x.dst] = ir.fromI(x.imm),
                .copy => |x| regs[x.dst] = regs[x.src],
                .arith => |x| {
                    const a = ir.toI(regs[x.a]);
                    const d = ir.toI(regs[x.b]);
                    if ((x.op == .sdiv or x.op == .srem) and (d == 0 or (a == std.math.minInt(i64) and d == -1))) return error.Trap;
                    regs[x.dst] = ir.fromI(ir.arith(x.op, a, d));
                },
                .load => |x| {
                    const p = strictAddr(regs[x.base], x.disp, x.ty.size()) orelse return error.Trap;
                    var v: ir.Val = 0;
                    @memcpy(std.mem.asBytes(&v)[0..p.len], p);
                    regs[x.dst] = v;
                },
                .store => |x| {
                    const p = strictAddr(regs[x.base], x.disp, x.ty.size()) orelse return error.Trap;
                    @memcpy(p, std.mem.asBytes(&regs[x.src])[0..p.len]);
                },
                .call => |x| {
                    // `testgen.callee`, evaluated here so the call sequence is recorded per run.
                    std.debug.assert(x.addr == @intFromPtr(&testgen.callee) and x.args.len == 2);
                    const a = ir.toI(regs[x.args[0]]);
                    const d = ir.toI(regs[x.args[1]]);
                    if (s.ncalls < s.calls.len) s.calls[s.ncalls] = .{ a, d };
                    s.ncalls += 1;
                    regs[x.dst] = ir.fromI(a *% 31 +% (d ^ @as(i64, @intCast(s.ncalls))));
                },
                .fconst => |x| regs[x.dst] = x.bits,
                .typed => |x| regs[x.dst] = ir.typedArith(x.op, x.ty, regs[x.a], regs[x.b]),
                .fcmp => |x| regs[x.dst] = @intFromBool(if (x.ty == .f32)
                    ir.fcond(x.cond, ir.getF32(regs[x.a]), ir.getF32(regs[x.b]))
                else
                    ir.fcond(x.cond, ir.getF64(regs[x.a]), ir.getF64(regs[x.b]))),
                .cvt => |x| regs[x.dst] = ir.convert(x.to, x.from, regs[x.src]),
                .splat => |x| {
                    var v: ir.Val = 0;
                    for (0..x.ty.lanes()) |l| v = ir.setLane(x.ty, v, @intCast(l), regs[x.src]);
                    regs[x.dst] = v;
                },
                .extract => |x| regs[x.dst] = ir.getLane(x.ty, regs[x.src], x.lane),
                .br => |to| {
                    b = to;
                    break;
                },
                .cbr => |x| {
                    b = if (ir.cond(x.cond, ir.toI(regs[x.a]), ir.toI(regs[x.b]))) x.then else x.@"else";
                    break;
                },
                .bra, .cbra => {
                    const e: u1 = if (o == .bra or ir.cond(o.cbra.cond, ir.toI(regs[o.cbra.a]), ir.toI(regs[o.cbra.b]))) 0 else 1;
                    const edge = cfg.edgeOf(o, e);
                    for (edge.args, 0..) |a, i| tmp[i] = regs[a];
                    for (f.blocks.items[edge.to].params, 0..) |p, i| regs[p] = tmp[i];
                    b = edge.to;
                    break;
                },
                .ret => |v| return ir.toI(regs[v]),
            }
        }
    }
}

/// Makes a `testgen.program` hazardous: divisors become `(x & 3) - 1` (0 and -1 possible)
/// and loads read `v13 = v1 + 8 * ((v0 >> 2) & 15)` (+ disp), out of bounds for many inputs.
fn hazardous(src: []const u8, out: *std.ArrayList(u8)) !void {
    const gpa = t.allocator;
    out.clearRetainingCapacity();
    var rest = src;
    const entry = "b0:\n";
    std.debug.assert(std.mem.startsWith(u8, std.mem.trimStart(u8, rest[std.mem.indexOfScalar(u8, rest, '\n').? + 1 ..], ""), entry));
    const head = std.mem.indexOf(u8, rest, entry).? + entry.len;
    try out.appendSlice(gpa, rest[0..head]);
    try out.appendSlice(gpa,
        \\  v13 = const 2
        \\  v13 = sar v0 v13
        \\  v14 = const 15
        \\  v13 = and v13 v14
        \\  v14 = const 8
        \\  v13 = mul v13 v14
        \\  v13 = add v1 v13
        \\
    );
    rest = rest[head..];
    var lines = std.mem.splitScalar(u8, rest, '\n');
    while (lines.next()) |line| {
        if (line.len == 0) continue;
        if (std.mem.eql(u8, line, "  v12 = const 255")) {
            try out.appendSlice(gpa, "  v12 = const 3\n");
        } else if (std.mem.eql(u8, line, "  v12 = add v12 v11")) {
            try out.appendSlice(gpa, "  v12 = sub v12 v11\n");
        } else if (std.mem.indexOf(u8, line, " = load v1 ") orelse std.mem.indexOf(u8, line, " = load.f64 v1 ")) |i| {
            const at = std.mem.indexOfPos(u8, line, i, "v1 ").?;
            try out.print(gpa, "{s}v13 {s}\n", .{ line[0..at], line[at + 3 ..] });
        } else {
            try out.appendSlice(gpa, line);
            try out.append(gpa, '\n');
        }
    }
}

const Mode = enum { licm, rotate_licm, o2 };

/// Transforms `src` (`mode`) and checks it with the trapping interpreter: on every input where
/// the original finishes without trapping, the result does too, with the same result, memory
/// and calls. Returns the number of hoisted ops.
fn expectNoSpeculation(src: []const u8, mode: Mode) !u32 {
    const gpa = t.allocator;
    var ref = try ir.parse(gpa, src);
    defer ref.deinit();
    var f = try ir.parse(gpa, src);
    defer f.deinit();
    const options: opt.Options = .{ .verify_each = true };
    var stats: opt.Stats = .{};
    if (mode == .o2) {
        stats = try opt.optimize(gpa, &f, .{ .level = .o2, .verify_each = true });
    } else {
        var c = try opt.Ctx.init(gpa, &f, &options, &stats);
        defer c.deinit();
        try ssa.build(&c);
        if (mode == .rotate_licm) {
            _ = try @import("loops.zig").rotate(&c);
            try c.check("rotate");
        }
        _ = try run(&c);
        try c.check("licm");
    }
    const extra = [_]i64{ 3, 4, 5, 12, 13, 28, 29, 33, -2, -7, -64, 255, 256, 1 << 33 };
    for (testgen.default_inputs ++ extra) |in| {
        const want = try strictRun(&ref, in, 20_000);
        _ = want.result catch continue; // trapped or did not finish: anything goes
        const got = try strictRun(&f, in, 80_000);
        try t.expectEqual(try want.result, try got.result);
        for (want.mem, got.mem) |x, y| if (x != y) {
            const nan = std.math.isNan(@as(f64, @bitCast(x))) and std.math.isNan(@as(f64, @bitCast(y)));
            if (!nan) try t.expectEqual(x, y);
        };
        try t.expectEqual(want.ncalls, got.ncalls);
        const k = @min(want.ncalls, want.calls.len);
        try t.expectEqualSlices([2]i64, want.calls[0..k], got.calls[0..k]);
    }
    return stats.hoisted;
}

test "licm: no speculation of faulting ops (trapping interpreter, random programs)" {
    const gpa = t.allocator;
    const configs = [_]testgen.Config{
        .{ .floats = false },
        .{ .blocks = 10, .ops = 6, .calls = false },
        .{ .blocks = 4, .ops = 10, .calls = false, .floats = false },
    };
    var src: std.ArrayList(u8) = .empty;
    defer src.deinit(gpa);
    var hz: std.ArrayList(u8) = .empty;
    defer hz.deinit(gpa);
    for (configs, 0..) |config, ci| {
        var prng = std.Random.DefaultPrng.init(0x5bec + ci);
        for (0..12 * fuzz_scale) |i| {
            src.clearRetainingCapacity();
            try testgen.program(prng.random(), gpa, config, &src);
            try hazardous(src.items, &hz);
            _ = expectNoSpeculation(hz.items, if (i % 2 == 0) .licm else .rotate_licm) catch |e| {
                std.debug.print("licm speculation failure on:\n{s}\n", .{hz.items});
                return e;
            };
        }
    }
}

/// Random structured programs (nested if/else and counted loops, top- or bottom-tested, that
/// run 0..3 times) whose divisions and loads have loop-invariant operands and fault for some
/// inputs: `v8 = (v0 & 3) - 1` is a divisor, `v9 = v1 + 8 * ((v0 >> 2) & 15)` a load base.
/// Pool: v2..v7; loop counters v20+depth, bounds v30+depth, constants v40+depth.
const Structured = struct {
    rng: std.Random,
    out: *std.ArrayList(u8),
    stores: bool,
    calls: bool,
    labels: u32 = 0,

    fn print(g: *Structured, comptime fmt: []const u8, args: anytype) !void {
        try g.out.print(t.allocator, fmt, args);
    }
    fn label(g: *Structured) u32 {
        g.labels += 1;
        return g.labels;
    }
    /// An i64 operand: a parameter, the pool, or (often) a loop-invariant hazard.
    fn operand(g: *Structured) u32 {
        return switch (g.rng.uintLessThan(u32, 10)) {
            0, 1 => 0,
            2 => 1,
            3, 4 => 8,
            else => g.rng.intRangeAtMost(u32, 2, 7),
        };
    }
    /// A load/store base: the buffer or the hazard base. Few displacements, so accesses alias.
    fn base(g: *Structured) u32 {
        return if (g.rng.boolean()) 1 else 9;
    }
    fn disp(g: *Structured) u32 {
        return 8 * g.rng.uintLessThan(u32, 3);
    }
    fn program(g: *Structured) !void {
        try g.print(
            \\func s 2
            \\e:
            \\  v8 = const 3
            \\  v8 = and v0 v8
            \\  v14 = const 1
            \\  v8 = sub v8 v14
            \\  v9 = const 2
            \\  v9 = sar v0 v9
            \\  v14 = const 15
            \\  v9 = and v9 v14
            \\  v14 = const 8
            \\  v9 = mul v9 v14
            \\  v9 = add v1 v9
            \\
        , .{});
        try g.stmts(0, 6);
        try g.print("  v15 = xor v2 v3\n  v15 = xor v15 v4\n  v15 = add v15 v5\n  v15 = xor v15 v6\n  v15 = sub v15 v7\n  ret v15\n", .{});
    }
    fn stmts(g: *Structured, depth: u32, max: u32) anyerror!void {
        for (0..g.rng.intRangeAtMost(u32, 1, max)) |_| try g.stmt(depth);
    }
    fn stmt(g: *Structured, depth: u32) !void {
        const d = g.rng.intRangeAtMost(u32, 2, 7);
        const nested = depth < 3;
        switch (g.rng.uintLessThan(u32, 14)) {
            0 => try g.print("  v{d} = {s} v{d} v{d}\n", .{ d, @tagName(g.rng.enumValue(enum { add, sub, mul, xor, shl, sar })), g.operand(), g.operand() }),
            2 => try g.print("  v{d} = const {d}\n", .{ d, g.rng.intRangeAtMost(i64, -3, 3) }),
            3, 4 => try g.print("  v{d} = {s} v{d} v{d}\n", .{ d, if (g.rng.boolean()) "sdiv" else "srem", g.operand(), g.operand() }),
            1, 5, 6 => try g.print("  v{d} = load v{d} {d}\n", .{ d, g.base(), g.disp() }),
            7, 12 => if (g.stores) try g.print("  store v{d} {d} v{d}\n", .{ g.base(), g.disp(), g.operand() }),
            8 => if (g.calls) try g.print("  v{d} = call 0x{x} v{d} v{d}\n", .{ d, @intFromPtr(&testgen.callee), g.operand(), g.operand() }),
            9 => if (nested) {
                const th = g.label();
                const el = g.label();
                const join = g.label();
                try g.print("  cbr {s} v{d} v{d} L{d} L{d}\nL{d}:\n", .{ @tagName(g.rng.enumValue(ir.CondCode)), g.operand(), g.operand(), th, el, th });
                try g.stmts(depth + 1, 3);
                try g.print("  br L{d}\nL{d}:\n", .{ join, el });
                if (g.rng.boolean()) try g.stmts(depth + 1, 2);
                try g.print("  br L{d}\nL{d}:\n", .{ join, join });
            },
            else => if (nested) { // 10, 11, 13
                const top = g.rng.boolean();
                const hd = g.label();
                const body = g.label();
                const exit = g.label();
                try g.print("  v{d} = const 0\n  v{d} = const 3\n  v{d} = and v{d} v{d}\n", .{ 20 + depth, 30 + depth, 30 + depth, g.operand(), 30 + depth });
                if (top) {
                    try g.print("  br L{d}\nL{d}:\n  cbr lt v{d} v{d} L{d} L{d}\nL{d}:\n", .{ hd, hd, 20 + depth, 30 + depth, body, exit, body });
                } else try g.print("  cbr ge v{d} v{d} L{d} L{d}\nL{d}:\n", .{ 20 + depth, 30 + depth, exit, body, body });
                try g.stmts(depth + 1, 5);
                try g.print("  v{d} = const 1\n  v{d} = add v{d} v{d}\n", .{ 40 + depth, 20 + depth, 20 + depth, 40 + depth });
                if (top) {
                    try g.print("  br L{d}\nL{d}:\n", .{ hd, exit });
                } else try g.print("  cbr lt v{d} v{d} L{d} L{d}\nL{d}:\n", .{ 20 + depth, 30 + depth, body, exit, exit });
            },
        }
    }
};

test "licm: no speculation of faulting ops (trapping interpreter, structured programs)" {
    const gpa = t.allocator;
    var src: std.ArrayList(u8) = .empty;
    defer src.deinit(gpa);
    var hoisted: u32 = 0;
    for (0..4) |ci| {
        var prng = std.Random.DefaultPrng.init(0x57bc + ci);
        for (0..20 * fuzz_scale) |i| {
            src.clearRetainingCapacity();
            var g: Structured = .{ .rng = prng.random(), .out = &src, .stores = ci & 1 == 0, .calls = ci == 3 };
            try g.program();
            const mode: Mode = switch (i % 3) {
                0 => .licm,
                1 => .rotate_licm,
                else => .o2,
            };
            hoisted += expectNoSpeculation(src.items, mode) catch |e| {
                std.debug.print("licm speculation failure ({s}) on:\n{s}\n", .{ @tagName(mode), src.items });
                return e;
            };
        }
    }
    try t.expect(hoisted > 60 * fuzz_scale);
}

test "licm: o2 pipeline on hand-written loops" {
    for ([_][]const u8{ nested_src, ir.fib_src, ir.fib_ssa_src }) |src| {
        const st = try testgen.checkOptimize(t.allocator, src, .{ .level = .o2, .verify_each = true });
        _ = st;
    }
}
