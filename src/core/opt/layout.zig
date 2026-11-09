//! Block layout: orders blocks so hot edges fall through (the IR Compilers emit blocks in IR
//! order: no jump for a `br` to the next block, one inverted conditional branch for a `cbr`
//! with a target next), loops stay contiguous and cold blocks go last. Blocks are only
//! permuted (`Ctx.reorderBlocks`): no block, edge or terminator changes, so the pass is valid
//! in and out of SSA.
//!
//! Edge weights. With a profile (`Ctx.freq` known for some reachable block) an edge weighs
//! `Ctx.edgeFreq`; a block without a count gets the sum of its forward incoming edges (in
//! reverse postorder), and a `cbr` without a `taken` count splits its block's weight with the
//! static probabilities below. Without a profile a block weighs 8^d inside a loop of depth d
//! (relative units) and a `cbr` splits it: an edge into a block that only returns (`ret`, an
//! early exit or error path, possibly behind a split edge) gets 1/16 unless both do; otherwise, in a loop, the edge staying
//! in the innermost loop (back edge or body) gets 90% against a loop exit; otherwise 50/50.
//!
//! Chains (Pettis–Hansen): edges by decreasing weight (ties: an edge to the block that
//! already follows its source first, then block order, so equal weights keep the current
//! order) link `a -> b` when `a` ends a chain and `b` starts another; nothing links into
//! block 0, and no edge links a cold block (never executed in the profile; statically, a
//! return block entered only by cold edges) behind a block that is not cold. Placement: the
//! chain of block 0 first, then repeatedly the chain entered by the heaviest edge from already
//! placed blocks (ties: lowest block index), all-cold chains after all others, unreachable
//! blocks last in their current order. Loops come out contiguous because the static weights
//! favour staying in a loop (an edge from depth d weighs at least 8^d / 2 when it stays in the
//! loop, at most 8^d / 10 when it leaves only that loop): on random programs 99% of the
//! natural loops are contiguous afterwards (78% before); the rest leave several loops at once
//! or are entered through odd nests. O((blocks + edges) log edges).
const std = @import("std");
const ir = @import("../ir.zig");
const opt = @import("opt.zig");
const cfg = @import("cfg.zig");

const none = cfg.none;

/// Static probability of a `cbr` edge into a block that only returns, against one that does not.
const cold_prob = 1.0 / 16.0;
/// Static probability of the `cbr` edge staying in the innermost loop, against a loop exit.
const loop_prob = 0.9;
/// Static frequency factor per loop nesting level.
const loop_scale = 8.0;
/// Deeper nesting does not raise the static estimate further (keeps weights finite).
const max_depth = 32;

/// Reorders the blocks; returns whether the order changed.
pub fn run(c: *opt.Ctx) opt.Error!bool {
    const f = c.f;
    const gpa = c.gpa;
    const n: u32 = @intCast(f.blocks.items.len);
    if (n <= 2) return false; // block 0 stays first: nothing to choose

    var g = try cfg.Cfg.init(gpa, f);
    defer g.deinit();
    var loops = try cfg.Loops.init(gpa, f, &g);
    defer loops.deinit();

    // Weight of edge `2 * b + index`, and whether each block is cold.
    const w = try gpa.alloc(f64, 2 * n);
    defer gpa.free(w);
    @memset(w, 0);
    const cold = try gpa.alloc(bool, n);
    defer gpa.free(cold);
    try weigh(c, &g, &loops, w, cold);

    // Candidate fallthrough edges of reachable blocks, heaviest first.
    var edges: std.ArrayList(Edge) = .empty;
    defer edges.deinit(gpa);
    var buf: [2]u32 = undefined;
    for (g.rpo) |b| for (cfg.succs(f, b, &buf), 0..) |s, i| {
        if (s == b or s == 0) continue;
        try edges.append(gpa, .{ .from = b, .to = s, .w = w[2 * b + i] });
    };
    std.mem.sort(Edge, edges.items, {}, Edge.before); // stable: then (block, index) order

    // Chains as doubly linked lists; `end[x]` is the other end of the chain `x` starts or ends.
    const next = try gpa.alloc(u32, n);
    defer gpa.free(next);
    const prev = try gpa.alloc(u32, n);
    defer gpa.free(prev);
    const end = try gpa.alloc(u32, n);
    defer gpa.free(end);
    @memset(next, none);
    @memset(prev, none);
    for (end, 0..) |*e, i| e.* = @intCast(i);
    for (edges.items) |e| {
        if (next[e.from] != none or prev[e.to] != none or end[e.from] == e.to) continue;
        if (cold[e.to] and !cold[e.from]) continue; // cold code stays out of hot chains
        const head = end[e.from];
        const tail = end[e.to];
        next[e.from] = e.to;
        prev[e.to] = e.from;
        end[head] = tail;
        end[tail] = head;
    }

    // Chain ids in order of their lowest block (the tie-break order); unreachable blocks are
    // cold singleton chains.
    const chain_of = try gpa.alloc(u32, n);
    defer gpa.free(chain_of);
    @memset(chain_of, none);
    var heads: std.ArrayList(u32) = .empty; // chain id -> head block
    defer heads.deinit(gpa);
    var chain_cold: std.ArrayList(bool) = .empty;
    defer chain_cold.deinit(gpa);
    for (0..n) |bi| {
        if (chain_of[bi] != none) continue;
        var h: u32 = @intCast(bi);
        while (prev[h] != none) h = prev[h];
        const id: u32 = @intCast(heads.items.len);
        var all_cold = true;
        var x = h;
        while (x != none) : (x = next[x]) {
            chain_of[x] = id;
            all_cold = all_cold and cold[x];
        }
        try heads.append(gpa, h);
        try chain_cold.append(gpa, all_cold);
    }
    std.debug.assert(heads.items[chain_of[0]] == 0);

    // Placement.
    const placed = try gpa.alloc(bool, heads.items.len);
    defer gpa.free(placed);
    @memset(placed, false);
    var order: std.ArrayList(u32) = .empty;
    defer order.deinit(gpa);
    try order.ensureTotalCapacity(gpa, n);
    var queue: Queue = .empty;
    defer queue.deinit(gpa);
    var pending: ?u32 = chain_of[0];
    while (pending) |id| {
        placed[id] = true;
        var x = heads.items[id];
        while (x != none) : (x = next[x]) {
            order.appendAssumeCapacity(x);
            for (cfg.succs(f, x, &buf), 0..) |s, i| {
                const to = chain_of[s];
                if (!placed[to]) try queue.push(gpa, .{ .chain = to, .cold = chain_cold.items[to], .w = w[2 * x + i] });
            }
        }
        pending = while (queue.pop()) |cand| {
            if (!placed[cand.chain]) break cand.chain;
        } else null;
    }
    // What no placed block branches to: unreachable blocks (and their chains), in order.
    for (heads.items, 0..) |h, id| if (!placed[id]) {
        var x = h;
        while (x != none) : (x = next[x]) order.appendAssumeCapacity(x);
    };
    std.debug.assert(order.items.len == n and order.items[0] == 0);

    for (order.items, 0..) |b, i| {
        if (b != i) break;
    } else return false;
    try c.reorderBlocks(order.items);
    return true;
}

const Edge = struct {
    from: u32,
    to: u32,
    w: f64,

    /// Heavier first; among equal weights an edge into the block that already follows its
    /// source first, so ties keep the current order.
    fn before(_: void, a: Edge, b: Edge) bool {
        if (a.w != b.w) return a.w > b.w;
        return a.to == a.from + 1 and b.to != b.from + 1;
    }
};

const Candidate = struct {
    chain: u32,
    cold: bool,
    w: f64,

    /// Not cold first, then heavier, then the lower chain id (lowest block index).
    fn order(_: void, a: Candidate, b: Candidate) std.math.Order {
        if (a.cold != b.cold) return if (a.cold) .gt else .lt;
        if (a.w != b.w) return if (a.w > b.w) .lt else .gt;
        return std.math.order(a.chain, b.chain);
    }
};
const Queue = std.PriorityQueue(Candidate, void, Candidate.order);

/// Fills `w[2 * b + index]` with the weight of every edge of a reachable block and `cold`
/// for every block (see the module comment). Unreachable blocks are cold.
fn weigh(c: *const opt.Ctx, g: *const cfg.Cfg, loops: *const cfg.Loops, w: []f64, cold: []bool) opt.Error!void {
    const f = c.f;
    const gpa = c.gpa;
    const n = f.blocks.items.len;
    var profiled = false;
    for (g.rpo) |b| profiled = profiled or c.freq.items[b] != null;

    // Profile: estimated count of blocks without one (sum of forward incoming edges).
    // Static: number of incoming edges and of cold incoming edges (for `cold`).
    const est = try gpa.alloc(f64, n);
    defer gpa.free(est);
    @memset(est, 0);
    const ins = try gpa.alloc([2]u32, n);
    defer gpa.free(ins);
    @memset(ins, .{ 0, 0 });

    var buf: [2]u32 = undefined;
    for (g.rpo) |b| {
        const bf: f64 = if (profiled)
            if (c.freq.items[b]) |x| @floatFromInt(x) else if (b == 0) 1 else est[b]
        else blk: {
            const l = loops.loop_of[b];
            const d = if (l == none) 0 else @min(loops.loops[l].depth, max_depth);
            break :blk std.math.pow(f64, loop_scale, @floatFromInt(d));
        };
        const ss = cfg.succs(f, b, &buf);
        const p0: f64 = if (ss.len == 2) thenProb(f, loops, b, ss) else 1;
        for (ss, 0..) |s, i| {
            const p = if (i == 0) p0 else 1 - p0;
            const known: ?u64 = if (profiled) c.edgeFreq(b, @intCast(i)) else null;
            const wt = if (known) |k| @as(f64, @floatFromInt(k)) else bf * p;
            w[2 * b + i] = wt;
            if (g.rpo_index[s] > g.rpo_index[b]) est[s] += wt;
            ins[s][0] += 1;
            ins[s][1] += @intFromBool(ss.len == 2 and returns(f, s) and !returns(f, ss[1 - i]));
        }
        cold[b] = if (profiled) bf == 0 else false;
    }
    for (cold, 0..) |*cb, b| {
        if (!g.reachable(@intCast(b))) {
            cb.* = true;
        } else if (!profiled) {
            cb.* = returns(f, @intCast(b)) and ins[b][0] != 0 and ins[b][1] == ins[b][0];
        }
    }
}

/// Static probability of the `then` edge of the `cbr`/`cbra` ending block `b` (`ss`: its
/// two successors).
fn thenProb(f: *const ir.Func, loops: *const cfg.Loops, b: u32, ss: []const u32) f64 {
    if (ss[0] == ss[1]) return 0.5;
    const r0 = returns(f, ss[0]);
    if (r0 != returns(f, ss[1])) return if (r0) cold_prob else 1 - cold_prob;
    const l = loops.loop_of[b];
    if (l != none) {
        const in0 = loops.contains(l, ss[0]);
        if (in0 != loops.contains(l, ss[1])) return if (in0) loop_prob else 1 - loop_prob;
    }
    return 0.5;
}

/// Block `b` ends in `ret`, possibly behind a few blocks that only branch on (edges split by
/// other passes or by out of SSA).
fn returns(f: *const ir.Func, b: u32) bool {
    var x = b;
    for (0..4) |_| {
        const ops = f.blocks.items[x].ops.items;
        switch (ops[ops.len - 1]) {
            .ret => return true,
            .br => |to| x = to,
            .bra => |y| x = y.target,
            else => return false,
        }
        if (ops.len != 1) return false;
    }
    return false;
}

// ---- tests -------------------------------------------------------------------------------

const t = std.testing;
const testgen = @import("testgen.zig");
const ssa = @import("ssa.zig");
const out_of_ssa = @import("out_of_ssa.zig");

const Form = enum {
    /// `ssa.build` (or the text as strict SSA when it has block parameters).
    ssa,
    /// Mutable vregs as parsed (layout after out of SSA).
    plain,
};

/// Parses `src` (with `profile`, indexed by the parsed blocks), brings it into `form`, runs the
/// layout (checked; `expect_changed` when not null) and compares with the original on
/// `testgen.default_inputs`. Checks that the result is a permutation of the blocks with block 0
/// first and returns the block names in the new order, space-separated (caller frees).
fn layoutOf(src: []const u8, profile: ?opt.Profile, form: Form, expect_changed: ?bool) ![]u8 {
    const gpa = t.allocator;
    var ref = try ir.parse(gpa, src);
    defer ref.deinit();
    var f = try ir.parse(gpa, src);
    defer f.deinit();
    const options: opt.Options = .{ .verify_each = true, .profile = if (profile) |*p| p else null };
    var stats: opt.Stats = .{};
    var c = try opt.Ctx.init(gpa, &f, &options, &stats);
    defer c.deinit();
    switch (form) {
        .ssa => if (f.hasBlockParams()) {
            c.in_ssa = true;
        } else try ssa.build(&c),
        .plain => {},
    }
    try c.check("setup");
    const before = try gpa.alloc([]const u8, f.blocks.items.len);
    defer gpa.free(before);
    const freq_before = try gpa.dupe(?u64, c.freq.items);
    defer gpa.free(freq_before);
    for (before, f.blocks.items) |*nm, b| nm.* = b.name;
    const changed = try run(&c);
    if (expect_changed) |e| try t.expectEqual(e, changed);
    try c.check("layout");
    try testgen.expectSame(&ref, &f, &testgen.default_inputs, 20_000);
    // A permutation with the entry first; profile counts follow their blocks.
    try t.expectEqualStrings(before[0], f.blocks.items[0].name);
    try t.expectEqual(before.len, f.blocks.items.len);
    for (before, freq_before) |nm, fr| {
        var hits: usize = 0;
        for (f.blocks.items, c.freq.items) |b, fr2| if (b.name.ptr == nm.ptr) {
            hits += 1;
            try t.expectEqual(fr, fr2);
        };
        try t.expectEqual(@as(usize, 1), hits);
    }
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(gpa);
    for (f.blocks.items, 0..) |b, i| try out.print(gpa, "{s}{s}", .{ if (i == 0) "" else " ", b.name });
    return out.toOwnedSlice(gpa);
}

fn expectLayout(want: []const u8, src: []const u8, profile: ?opt.Profile, form: Form) !void {
    const got = try layoutOf(src, profile, form, null);
    defer t.allocator.free(got);
    try t.expectEqualStrings(want, got);
}

/// A counted loop whose body has an error exit (`err`, returns -1) checked every iteration.
const loop_err_src =
    \\func s 2
    \\e:
    \\  v2 = const 0
    \\  v3 = const 0
    \\  br h
    \\h:
    \\  v9 = const 8
    \\  cbr lt v2 v9 body done
    \\err:
    \\  v8 = const -1
    \\  ret v8
    \\body:
    \\  v4 = const 8
    \\  v5 = mul v2 v4
    \\  v5 = add v1 v5
    \\  v6 = load v5 0
    \\  v7 = const 0
    \\  cbr eq v6 v7 err cont
    \\done:
    \\  ret v3
    \\cont:
    \\  v3 = xor v3 v6
    \\  v7 = const 1
    \\  v2 = add v2 v7
    \\  br h
;

test "layout: loop with a cold error path, static estimate" {
    // The latch falls into the header, the header into the body (the loop is rotated: one
    // taken branch per iteration); the cold returns go last.
    try expectLayout("e cont h body err done", loop_err_src, null, .ssa);
    try expectLayout("e cont h body err done", loop_err_src, null, .plain);
}

test "layout: loop with a cold error path, profile" {
    // Blocks: e h err body done cont; 8 iterations, the error path never taken.
    const p: opt.Profile = .{ .blocks = &.{ 1, 9, 0, 8, 1, 8 }, .taken = &.{ 0, 8, 0, 0, 0, 0 } };
    // Equal weights on the loop's edges: the current order decides the rotation.
    try expectLayout("e h body cont done err", loop_err_src, p, .ssa);
    // The error path always taken on the first iteration: it is hot, the loop is not.
    const q: opt.Profile = .{ .blocks = &.{ 1, 1, 1, 1, 0, 0 }, .taken = &.{ 0, 1, 0, 1, 0, 0 } };
    try expectLayout("e h body err done cont", loop_err_src, q, .ssa);
}

const diamond_src =
    \\func d 2
    \\e:
    \\  cbr lt v0 v1 l r
    \\l:
    \\  v2 = add v0 v1
    \\  br j
    \\r:
    \\  v2 = sub v0 v1
    \\  br j
    \\j:
    \\  ret v2
;

test "layout: diamond keeps its order without a profile, biased branch with one" {
    const same = try layoutOf(diamond_src, null, .ssa, false);
    t.allocator.free(same);
    try expectLayout("e r j l", diamond_src, .{ .blocks = &.{ 100, 10, 90, 100 }, .taken = &.{10} }, .ssa);
    try expectLayout("e l j r", diamond_src, .{ .blocks = &.{ 100, 90, 10, 100 }, .taken = &.{90} }, .ssa);
    // Unknown counts (shorter arrays) for some blocks: the profile still decides.
    try expectLayout("e r j l", diamond_src, .{ .blocks = &.{100}, .taken = &.{3} }, .plain);
}

test "layout: nested loops stay contiguous" {
    const src =
        \\func n 2
        \\e:
        \\  v2 = const 0
        \\  v3 = const 0
        \\  br oh
        \\oh:
        \\  v9 = const 6
        \\  cbr lt v2 v9 ob done
        \\done:
        \\  ret v3
        \\ob:
        \\  v4 = const 0
        \\  br ih
        \\ih:
        \\  cbr lt v4 v2 ib olatch
        \\olatch:
        \\  v5 = const 1
        \\  v2 = add v2 v5
        \\  br oh
        \\ib:
        \\  v3 = add v3 v4
        \\  v6 = const 1
        \\  v4 = add v4 v6
        \\  br ih
    ;
    try expectLayout("e ib ih olatch oh ob done", src, null, .ssa);
    try expectLayout("e ib ih olatch oh ob done", src, null, .plain);
}

test "layout: cold return blocks go last" {
    const src =
        \\func r 2
        \\e:
        \\  v9 = const 0
        \\  cbr lt v0 v9 neg a
        \\neg:
        \\  v8 = const -1
        \\  ret v8
        \\a:
        \\  v2 = load v1 0
        \\  cbr eq v2 v0 same b
        \\same:
        \\  ret v9
        \\b:
        \\  v3 = add v2 v0
        \\  ret v3
    ;
    // `same` and `b` both return (50/50): the existing fallthrough `a -> same` stays.
    try expectLayout("e a same b neg", src, null, .ssa);
    // Profile: `b` is the common case, `neg` and `same` never happen.
    try expectLayout("e a b neg same", src, .{ .blocks = &.{ 10, 0, 10, 0, 10 }, .taken = &.{ 0, 0, 0 } }, .ssa);
}

test "layout: a never-executed loop goes last with a profile, rotated without" {
    const src =
        \\func z 2
        \\e:
        \\  v2 = const 0
        \\  br h
        \\h:
        \\  cbr lt v2 v0 body done
        \\body:
        \\  v3 = const 1
        \\  v2 = add v2 v3
        \\  br h
        \\done:
        \\  ret v2
    ;
    try expectLayout("e body h done", src, null, .ssa);
    try expectLayout("e h done body", src, .{ .blocks = &.{ 1, 1, 0, 1 }, .taken = &.{ 0, 0 } }, .ssa);
}

test "layout: unreachable blocks go last, edge arguments, equal targets" {
    try expectLayout("e h x dead",
        \\func u 2
        \\e:
        \\  v2 = const 1
        \\  cbr eq v0 v2 h:v2 x
        \\dead:
        \\  br x
        \\h(v3):
        \\  v4 = add v3 v2
        \\  cbr lt v4 v0 h:v4 h:v2
        \\x:
        \\  ret v2
    , null, .ssa);
}

test "layout: two blocks and straight lines are left alone" {
    for ([_][]const u8{
        \\func a 2
        \\e:
        \\  ret v0
        ,
        \\func b 2
        \\e:
        \\  br x
        \\x:
        \\  ret v0
        ,
        \\func c 2
        \\e:
        \\  br x
        \\x:
        \\  br y
        \\y:
        \\  ret v0
    }) |src| {
        const got = try layoutOf(src, null, .ssa, false);
        t.allocator.free(got);
    }
}

/// Random program `src` with a random profile (counts may be missing, zero, or racy
/// `taken > freq`), laid out in both forms.
fn fuzzProfiled(rng: std.Random, src: []const u8) !void {
    var f = try ir.parse(t.allocator, src);
    defer f.deinit();
    const n = f.blocks.items.len;
    var blocks: [64]u64 = undefined;
    var taken: [64]u64 = undefined;
    for (blocks[0..n], taken[0..n]) |*b, *tk| {
        b.* = switch (rng.uintLessThan(u8, 4)) {
            0 => 0,
            1 => rng.uintAtMost(u64, 3),
            else => rng.uintAtMost(u64, 1000),
        };
        tk.* = if (rng.uintLessThan(u8, 8) == 0) b.* + 1 else rng.uintAtMost(u64, b.*);
    }
    const p: opt.Profile = .{ .blocks = blocks[0..rng.uintAtMost(usize, n)], .taken = taken[0..rng.uintAtMost(usize, n)] };
    for ([_]Form{ .ssa, .plain }) |form| {
        const got = try layoutOf(src, p, form, null);
        t.allocator.free(got);
    }
}

test "layout: random programs" {
    const gpa = t.allocator;
    const configs = [_]testgen.Config{
        .{},
        .{ .loops = false },
        .{ .blocks = 16, .ops = 3 },
        .{ .blocks = 10, .ops = 2, .floats = false, .memory = false, .calls = false },
    };
    for (configs, 0..) |config, ci| try testgen.fuzzPass(gpa, 0x1a70 + ci, 20, config, &.{}, run);
    var prng = std.Random.DefaultPrng.init(0x1a7_0a7);
    var src: std.ArrayList(u8) = .empty;
    defer src.deinit(gpa);
    for (0..60) |i| {
        src.clearRetainingCapacity();
        try testgen.program(prng.random(), gpa, configs[i % configs.len], &src);
        fuzzProfiled(prng.random(), src.items) catch |e| {
            std.debug.print("layout failure on:\n{s}\n", .{src.items});
            return e;
        };
    }
}

/// Total Collatz steps for 1..n (`kx_collatz` of bench/exec_bench.zig as IR).
const collatz_src =
    \\func collatz 2
    \\e:
    \\  v2 = const 1
    \\  v4 = const 0
    \\  br outer
    \\outer:
    \\  cbr gt v2 v0 exit oinit
    \\oinit:
    \\  v3 = copy v2
    \\  br inner
    \\inner:
    \\  v5 = const 1
    \\  cbr eq v3 v5 onext ibody
    \\ibody:
    \\  v6 = const 2
    \\  v7 = srem v3 v6
    \\  v8 = const 0
    \\  cbr ne v7 v8 odd even
    \\even:
    \\  v3 = sdiv v3 v6
    \\  br step
    \\odd:
    \\  v9 = const 3
    \\  v3 = mul v3 v9
    \\  v3 = add v3 v5
    \\  br step
    \\step:
    \\  v4 = add v4 v5
    \\  br inner
    \\onext:
    \\  v2 = add v2 v5
    \\  br outer
    \\exit:
    \\  ret v4
;

/// `ir_calls` of bench/exec_bench.zig (the callee is `testgen.callee`, so v1 is passed too).
const calls_src_fmt =
    \\func calls 2
    \\e:
    \\  v1 = const 0
    \\  v2 = const 0
    \\  v3 = const 1
    \\  cbr ge v1 v0 done loop
    \\loop:
    \\  v4 = call 0x{x} v1 v3
    \\  v2 = add v2 v4
    \\  v1 = add v1 v3
    \\  cbr lt v1 v0 loop done
    \\done:
    \\  ret v2
;

test "layout: bench loops through the optimizer pipeline" {
    const gpa = t.allocator;
    // Collatz, blocks: e outer oinit inner ibody even odd step onext exit (n = 7).
    const prof: opt.Profile = .{
        .blocks = &.{ 1, 8, 7, 23, 16, 10, 6, 16, 7, 1 },
        .taken = &.{ 0, 1, 0, 7, 6, 0, 0, 0, 0, 0 },
    };
    const calls_src = try std.fmt.allocPrint(gpa, calls_src_fmt, .{@intFromPtr(&testgen.callee)});
    defer gpa.free(calls_src);
    for ([_][]const u8{ collatz_src, calls_src }) |src| for ([_]opt.Level{ .o1, .o2 }) |level| {
        _ = try testgen.checkOptimize(gpa, src, .{ .level = level, .verify_each = true });
        if (src.ptr == collatz_src.ptr)
            _ = try testgen.checkOptimize(gpa, src, .{ .level = level, .verify_each = true, .profile = &prof });
    };
    // Static layout of the nest: both loops contiguous and rotated (`step` falls into
    // `inner`, `onext` into `outer`), one taken branch per inner iteration on either path,
    // the exit last.
    try expectLayout("e onext outer oinit odd step inner ibody even exit", collatz_src, null, .plain);
}
