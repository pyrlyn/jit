//! Loop vectorizer (128-bit vectors, SSA form). Turns innermost counted loops over
//! consecutive array elements into a vector loop that runs VF = 16 / e scalar iterations per
//! trip (e = element size: 8 for i64/f64, 4 for f32), followed by the original loop as the
//! scalar epilogue.
//!
//! Accepted loops (everything else is left alone):
//! - shape `bottom`: one block L with parameters, straight-line body, ending in
//!   `cbr/cbra cond a b` with one edge back to L and one to an exit X; shape `top`: a header
//!   H (uniform and address arithmetic plus the exit test) whose `cbr` enters a single body
//!   block B or leaves to X, and B ends in `br/bra H`. A preheader P (`br/bra` to the header)
//!   is created when missing.
//! - Every header parameter is an induction variable (`q' = add q k` / `sub q k`, k a
//!   constant), an integer reduction (`r' = op r x`, op add/sub/and/or/xor, mul only with
//!   `mul_v2i64`, r used by nothing else in the loop) or loop invariant (`q' = q`).
//! - Each value of the body is uniform (loop invariant: constants, pure scalar ops on
//!   invariants; cloned into P), affine (`value_j = value_0 + j*stride`: induction variables
//!   and add/sub/mul-by-constant/shl-by-constant chains over them; kept scalar, lane 0's
//!   value) or data (lane-wise: loads, stores, add/sub/mul/and/or/xor on i64 and
//!   add/sub/mul/div/min/max on f32/f64; invariant operands are `splat`). Affine values are
//!   only used as addresses, in other affine values, in the exit test and as exit values.
//! - Every load/store has an affine address whose stride equals its size, all of one size e.
//! - The exit test compares an affine `x = q + t` (q an induction variable with a power-of-two
//!   step S > 0) with an invariant n; normalized to "continue iff x REL n" it is one of
//!   lt, ult, le, ule, ne.
//! - No calls, divisions, shifts/compares/conversions of data, float reductions, vector ops,
//!   or values of the loop used after it other than induction variables, reductions,
//!   invariants and affine values (bottom shape; the top shape re-runs its header, which
//!   recomputes them).
//!
//! Generated structure (P keeps its ops and ends in the first guard):
//!   P:  invariant clones; y0 = q0 + u; [lt/ult/le/ule: cbr REL y0 n -> count : scalar]
//!   count: R = n - y0; cnt = lower bound on the iterations still to run; kv = cnt & -VF;
//!          cbr eq kv 0 -> scalar : alias checks...
//!   alias: per pair of accesses with different symbolic bases (at least one store):
//!          d = addr_x - addr_y at iteration 0; scalar unless d == 0 or |d| >= 16
//!   vec.ph: q_end = q0 + kv*S, splats, accumulator identities; br vec.body
//!   vec.body(ivs, accs): the body on vectors; ivs += VF*step; cbr ne q' q_end -> vec.body : mid
//!   mid: reduce accumulators (extract + scalar ops, starting from the reduction's initial
//!        value); top: br H(state); bottom: the original exit test on the state, to L(state)
//!        or to X (through a merge block when values of L are used outside it).
//!   scalar: br header(initial values): the original loop.
//!
//! Iteration count (why the guards are exact). Let q_j = q0 + j*S be the induction variable at
//! the start of iteration j and y_j = q_j + u the value that decides whether iteration j runs
//! (top: u = t, the header tests x_j; bottom: iteration 0 always runs and iteration j >= 1 runs
//! iff x_{j-1} = q_j + t - S passes, so u = t - S and y_0 is a virtual extra test that only
//! makes the guard stricter). If REL(y_0 + k*S, n) holds for all k < kv, iterations 0..kv-1
//! all run. With R = n - y0 as u64 (exact once y0 REL n holds for lt/ult/le/ule, because then
//! y0 <= n in that order): lt/ult hold for k*S < R, i.e. k < ceil(R/S) = ((R-1) >> log S) + 1;
//! le/ule for k*S <= R, i.e. k < (R >> log S) + 1 (wraps to 0 only for R = 2^64-1, S = 1, which
//! just skips the vector loop); ne for k*S < R, i.e. k < R >> log S (no guard; R = 0 gives 0).
//! No y_0 + k*S used wraps (each is at most n in the compare's order) and kv*S <= 2^64, so
//! q0 + m*VF*S differs from q_end for 0 < m < kv/VF: `ne` ends the vector loop exactly.
//! After kv iterations the epilogue sees the state of iteration kv: the top shape's header
//! re-tests it; the bottom shape's mid block evaluates the original test on x_{kv-1} = q_kv + u.
//!
//! Dependences. All accesses advance e bytes per iteration. Two accesses with addresses
//! A + j*e and A' + j*e (A' - A = d constant) touch the same bytes in iterations j, j' only
//! if |(j' - j)*e - d| < e. If d == 0 that means j == j' (each element is accessed by one
//! iteration, and the vector body keeps the scalar program order of the accesses); if
//! |d| >= VF*e = 16 then |j' - j| >= VF, so the two iterations lie in different vector trips,
//! which run in order. Hence a pair is safe iff d == 0 or |d| >= 16: checked statically when
//! both addresses have the same symbolic base (equal linear forms up to a constant), else at
//! run time on d = addr_x(0) - addr_y(0) (wrapping): unsafe iff d != 0 and
//! (d + 15) as u64 < 31.
//!
//! Everything the vector loop executes (loads, stores) is exactly what iterations 0..kv-1
//! execute, so nothing is speculated; integer reductions are reassociated (exact with
//! wrapping), float ones are rejected.
const std = @import("std");
const ir = @import("../ir.zig");
const opt = @import("opt.zig");
const cfg = @import("cfg.zig");
const ssa = @import("ssa.zig");

const VReg = ir.VReg;
const none = cfg.none;
/// Vector width in bytes.
const width = 16;
/// Bail-out limits (keep the analysis linear and the guard code small).
const max_params = 32;
const max_accesses = 32;
const max_checks = 8;

pub fn run(c: *opt.Ctx) opt.Error!bool {
    if (!c.options.vector.enabled()) return false;
    const gpa = c.gpa;
    var changed = false;
    {
        // Preheaders first (one snapshot: each call only redirects edges into its own header).
        var g = try cfg.Cfg.init(gpa, c.f);
        defer g.deinit();
        var loops = try cfg.Loops.init(gpa, c.f, &g);
        defer loops.deinit();
        if (loops.irreducible) return false;
        for (loops.loops, 0..) |lp, l| {
            if (lp.preheader != none or lp.header == 0 or shapeOf(c.f, &g, &loops, @intCast(l)) == null) continue;
            _ = try cfg.ensurePreheader(c, &g, &loops, @intCast(l));
            changed = true;
        }
    }
    var cands: std.ArrayList(Cand) = .empty;
    defer cands.deinit(gpa);
    {
        var g = try cfg.Cfg.init(gpa, c.f);
        defer g.deinit();
        var loops = try cfg.Loops.init(gpa, c.f, &g);
        defer loops.deinit();
        for (loops.loops, 0..) |lp, l| {
            if (lp.preheader == none) continue;
            if (shapeOf(c.f, &g, &loops, @intCast(l))) |cd| try cands.append(gpa, cd);
        }
    }
    if (cands.items.len == 0) return changed;
    // Candidates are disjoint single- or two-block loops; transforming one only appends blocks
    // and rewrites its own preheader and exit edge, so the others' analysis stays valid.
    var gl = try Global.init(c);
    defer gl.deinit(gpa);
    var ren: Renames = .{};
    defer ren.deinit(gpa);
    var a: Loop = .{ .c = c, .gl = &gl, .cd = undefined };
    defer a.deinit();
    for (cands.items) |cd| {
        a.reset(cd);
        if (!try a.analyze()) continue;
        try a.transform(&ren);
        c.stats.vectorized_loops += 1;
        changed = true;
    }
    try ren.apply(c);
    return changed;
}

const Shape = enum { bottom, top };

/// A loop of an accepted shape: `hdr` holds the parameters and the exit test, `body` is the
/// latch (`hdr` itself for the bottom shape).
const Cand = struct {
    shape: Shape,
    hdr: u32,
    body: u32,
    pre: u32,
    exit: u32,
    /// Edge of `hdr`'s terminator that leaves the loop.
    exit_idx: u1,
    /// The loop's blocks in execution order: `hdr`, then `body` for the top shape.
    blocks: [2]u32,
};

fn shapeOf(f: *ir.Func, g: *const cfg.Cfg, loops: *const cfg.Loops, l: u32) ?Cand {
    const lp = loops.loops[l];
    if (lp.latches.len != 1 or lp.exits.len != 1 or lp.exits[0].from != lp.header) return null;
    const h = lp.header;
    const term = cfg.terminator(f, h).*;
    if (term != .cbr and term != .cbra) return null;
    const ex = lp.exits[0];
    const inside = cfg.edgeOf(term, 1 - ex.index).to;
    var cd: Cand = .{ .shape = .bottom, .hdr = h, .body = h, .pre = lp.preheader, .exit = ex.to, .exit_idx = ex.index, .blocks = .{ h, inside } };
    if (lp.blocks.len == 1) return if (inside == h) cd else null;
    if (lp.blocks.len != 2 or inside == h or lp.latches[0] != inside) return null;
    if (g.preds[inside].len != 1 or f.blocks.items[inside].params.len != 0) return null;
    switch (cfg.terminator(f, inside).*) {
        .br, .bra => {},
        else => return null,
    }
    cd.shape = .top;
    cd.body = inside;
    return cd;
}

/// Per-function tables, built once per `run` (over the vregs that existed then).
const Global = struct {
    def_block: []u32,
    /// Index of the defining op in its block; `none` for parameters.
    def_op: []u32,
    /// Read in a block other than the one defining it.
    escapes: std.DynamicBitSetUnmanaged,

    fn init(c: *opt.Ctx) opt.Error!Global {
        const f = c.f;
        const gpa = c.gpa;
        const nv = f.num_vregs;
        const db = try gpa.alloc(u32, nv);
        errdefer gpa.free(db);
        const dop = try gpa.alloc(u32, nv);
        errdefer gpa.free(dop);
        @memset(db, none);
        @memset(dop, none);
        for (0..f.params) |v| db[v] = 0;
        for (f.blocks.items, 0..) |b, bi| {
            for (b.params) |p| db[p] = @intCast(bi);
            for (b.ops.items, 0..) |o, i| if (o.def()) |d| {
                db[d] = @intCast(bi);
                dop[d] = @intCast(i);
            };
        }
        var esc = try std.DynamicBitSetUnmanaged.initEmpty(gpa, nv);
        errdefer esc.deinit(gpa);
        var scratch: std.ArrayList(VReg) = .empty;
        defer scratch.deinit(gpa);
        for (f.blocks.items, 0..) |b, bi| for (b.ops.items) |o| {
            for (try ssa.usesOf(gpa, o, &scratch)) |u| if (db[u] != bi) esc.set(u);
        };
        return .{ .def_block = db, .def_op = dop, .escapes = esc };
    }
    fn deinit(gl: *Global, gpa: std.mem.Allocator) void {
        gpa.free(gl.def_block);
        gpa.free(gl.def_op);
        gl.escapes.deinit(gpa);
    }
    fn defOp(gl: *const Global, f: *const ir.Func, v: VReg) ?ir.Op {
        if (v >= gl.def_op.len or gl.def_op[v] == none) return null;
        return f.blocks.items[gl.def_block[v]].ops.items[gl.def_op[v]];
    }
    fn constOf(gl: *const Global, f: *const ir.Func, v: VReg) ?i64 {
        const o = gl.defOp(f, v) orelse return null;
        return if (o == .@"const") o.@"const".imm else null;
    }
};

/// `c + sum(k[i] * v[i])` (wrapping), terms sorted by vreg with nonzero coefficients.
const Lin = struct {
    const max_terms = 4;
    n: u8 = 0,
    v: [max_terms]VReg = undefined,
    k: [max_terms]i64 = undefined,
    c: i64 = 0,

    fn term(v: VReg) Lin {
        var l: Lin = .{ .n = 1 };
        l.v[0] = v;
        l.k[0] = 1;
        return l;
    }
    /// `a + m*b`; null when the result has too many terms.
    fn addScaled(a: Lin, b: Lin, m: i64) ?Lin {
        var r: Lin = .{ .c = a.c +% b.c *% m };
        var i: usize = 0;
        var j: usize = 0;
        while (i < a.n or j < b.n) {
            var v: VReg = undefined;
            var k: i64 = undefined;
            if (j >= b.n or (i < a.n and a.v[i] < b.v[j])) {
                v = a.v[i];
                k = a.k[i];
                i += 1;
            } else if (i >= a.n or b.v[j] < a.v[i]) {
                v = b.v[j];
                k = b.k[j] *% m;
                j += 1;
            } else {
                v = a.v[i];
                k = a.k[i] +% b.k[j] *% m;
                i += 1;
                j += 1;
            }
            if (k == 0) continue;
            if (r.n == max_terms) return null;
            r.v[r.n] = v;
            r.k[r.n] = k;
            r.n += 1;
        }
        return r;
    }
    fn sameTerms(a: Lin, b: Lin) bool {
        if (a.n != b.n) return false;
        for (0..a.n) |i| if (a.v[i] != b.v[i] or a.k[i] != b.k[i]) return false;
        return true;
    }
};

const Kind = enum { outside, uniform, affine, data, red_param, red_next };

/// Classification of a vreg defined in the loop (or a read of one defined outside).
const Val = struct {
    kind: Kind,
    /// Type (uniform, outside) or lane type (data, reductions).
    ty: ir.Type = .i64,
    /// Uniform/outside: the value as a linear form; affine: the value in iteration 0.
    lin: Lin = .{},
    /// Affine: increment per iteration.
    stride: i64 = 0,
    /// Index into `Loop.params` for header parameters.
    param: u32 = none,
};

const Param = struct {
    v: VReg,
    init: VReg,
    next: VReg,
    kind: enum { iv, red, uniform },
    /// iv: increment per iteration.
    step: i64 = 0,
    /// red: the operation and its other operand; `seen` once the op was classified.
    op: ir.Arith = .add,
    x: VReg = 0,
    seen: bool = false,
    /// Transform: parameter of the vector body, its next value, the value after it (red).
    vl: VReg = 0,
    vl_next: VReg = 0,
    mid: VReg = 0,
};

const Access = struct { base: VReg, lin: Lin, disp: i32, store: bool, size: u8 };

/// How `scalar` maps a value of the loop to a value usable elsewhere.
const Mode = enum {
    /// In P (uniform values only).
    pre,
    /// In the vector body: lane 0's value.
    body,
    /// In an alias-check block: the value in iteration 0.
    iter0,
    /// In the mid block: the value in iteration kv-1 (the last one the vector loop ran).
    last,
};

const Loop = struct {
    c: *opt.Ctx,
    gl: *const Global,
    cd: Cand,
    vals: std.AutoHashMapUnmanaged(VReg, Val) = .empty,
    params: std.ArrayList(Param) = .empty,
    accesses: std.ArrayList(Access) = .empty,
    /// Pairs of `accesses` indices needing a run-time alias check.
    checks: std.ArrayList([2]u32) = .empty,
    /// Bottom shape: values of L used outside L (they then reach X through a merge block).
    escaping: std.ArrayList(VReg) = .empty,
    /// Lane types of data and reductions (bit per `ir.Type`).
    lanes: u8 = 0,
    init: []const VReg = &.{},
    // Exit test: continue iff `x REL n`, x = q + t with q = params[prim]; y = q + u decides.
    prim: u32 = 0,
    rel: ir.CondCode = .lt,
    n: VReg = 0,
    u: i64 = 0,
    x_is_a: bool = true,
    vf: u8 = 0,
    // Transform maps: uniform clones in P, vector-body values, iteration-0 and last-iteration
    // values, splats (by scalar source).
    cmap: std.AutoHashMapUnmanaged(VReg, VReg) = .empty,
    vmap: std.AutoHashMapUnmanaged(VReg, VReg) = .empty,
    m0: std.AutoHashMapUnmanaged(VReg, VReg) = .empty,
    mlast: std.AutoHashMapUnmanaged(VReg, VReg) = .empty,
    splats: std.AutoHashMapUnmanaged(VReg, VReg) = .empty,
    // Blocks the transform emits into for `scalar`'s modes and splats.
    iter0_block: u32 = 0,
    mid_block: u32 = 0,
    vph_block: u32 = 0,
    body_block: u32 = 0,

    fn deinit(a: *Loop) void {
        const gpa = a.c.gpa;
        a.vals.deinit(gpa);
        a.params.deinit(gpa);
        a.accesses.deinit(gpa);
        a.checks.deinit(gpa);
        a.escaping.deinit(gpa);
        a.cmap.deinit(gpa);
        a.vmap.deinit(gpa);
        a.m0.deinit(gpa);
        a.mlast.deinit(gpa);
        a.splats.deinit(gpa);
    }

    /// Starts on candidate `cd`, keeping the containers' capacity.
    fn reset(a: *Loop, cd: Cand) void {
        var fresh: Loop = .{ .c = a.c, .gl = a.gl, .cd = cd };
        inline for (.{ "vals", "params", "accesses", "checks", "escaping", "cmap", "vmap", "m0", "mlast", "splats" }) |name| {
            @field(fresh, name) = @field(a, name);
            @field(fresh, name).clearRetainingCapacity();
        }
        a.* = fresh;
    }

    fn inLoop(a: *const Loop, b: u32) bool {
        return b == a.cd.hdr or b == a.cd.body;
    }

    fn get(a: *const Loop, v: VReg) Val {
        if (a.vals.get(v)) |x| return x;
        if (a.gl.constOf(a.c.f, v)) |k| return .{ .kind = .outside, .lin = .{ .c = k } };
        return .{ .kind = .outside, .ty = a.c.types.items[v], .lin = Lin.term(v) };
    }

    fn put(a: *Loop, v: VReg, x: Val) opt.Error!void {
        try a.vals.put(a.c.gpa, v, x);
    }

    fn isUni(k: Kind) bool {
        return k == .outside or k == .uniform;
    }

    fn uniform(v: VReg, ty: ir.Type) Val {
        return .{ .kind = .uniform, .ty = ty, .lin = Lin.term(v) };
    }

    fn data(a: *Loop, ty: ir.Type) Val {
        a.lanes |= @as(u8, 1) << @intCast(@backingInt(ty));
        return .{ .kind = .data, .ty = ty };
    }

    /// Blocks of the loop body in execution order.
    fn bodyBlocks(a: *const Loop) []const u32 {
        return a.cd.blocks[0..if (a.cd.shape == .bottom) 1 else 2];
    }

    /// The `(edge index, args)` of the back edge into the header.
    fn backArgs(a: *const Loop) []const VReg {
        const lt = cfg.terminator(a.c.f, a.cd.body).*;
        return cfg.edgeOf(lt, if (a.cd.shape == .bottom) 1 - a.cd.exit_idx else 0).args;
    }

    fn analyze(a: *Loop) opt.Error!bool {
        const f = a.c.f;
        const cd = a.cd;
        const pt = cfg.terminator(f, cd.pre).*;
        if (pt != .br and pt != .bra) return false;
        const pe = cfg.edgeOf(pt, 0);
        if (pe.to != cd.hdr) return false;
        a.init = pe.args;
        const hparams = f.blocks.items[cd.hdr].params;
        const back = a.backArgs();
        if (hparams.len > max_params or a.init.len != hparams.len or back.len != hparams.len) return false;

        // Header parameters: induction variables, reductions, invariants.
        for (hparams, a.init, back, 0..) |p, iv0, nx, pi| {
            var prm: Param = .{ .v = p, .init = iv0, .next = nx, .kind = .uniform };
            var val: Val = undefined;
            if (nx == p) {
                val = a.get(iv0);
                val.kind = .uniform;
                val.ty = a.c.types.items[p];
            } else {
                const d = a.gl.defOp(f, nx) orelse return false;
                if (!a.inLoop(a.gl.def_block[nx]) or d != .arith) return false;
                const x = d.arith;
                if (x.a != p and x.b != p) return false;
                const other = if (x.a == p) x.b else x.a;
                const step: ?i64 = if (other == p) null else if (a.gl.constOf(f, other)) |k| switch (x.op) {
                    .add => k,
                    .sub => if (x.a == p) 0 -% k else null,
                    else => null,
                } else null;
                if (step) |s| {
                    prm.kind = .iv;
                    prm.step = s;
                    val = .{ .kind = .affine, .lin = a.get(iv0).lin, .stride = s };
                } else {
                    prm.kind = .red;
                    val = .{ .kind = .red_param };
                }
            }
            val.param = @intCast(pi);
            try a.put(p, val);
            try a.params.append(a.c.gpa, prm);
        }

        // Body ops in order (SSA: operands defined in the loop are classified before use).
        for (a.bodyBlocks()) |b| {
            const ops = f.blocks.items[b].ops.items;
            const header_only = cd.shape == .top and b == cd.hdr;
            for (ops[0 .. ops.len - 1]) |op| {
                if (!try a.classify(op, header_only)) return false;
            }
        }
        for (a.params.items) |p| if (p.kind == .red and !p.seen) return false;
        if (!a.exitTest()) return false;

        // Values the bottom shape must provide after the vector loop.
        if (cd.shape == .bottom) {
            const term = cfg.terminator(f, cd.hdr).*;
            for (cfg.edgeOf(term, cd.exit_idx).args) |v| if (!a.lastMappable(v)) return false;
            for (hparams) |p| if (a.gl.escapes.isSet(p)) try a.escape(p);
            const ops = f.blocks.items[cd.hdr].ops.items;
            for (ops[0 .. ops.len - 1]) |o| if (o.def()) |d| if (a.gl.escapes.isSet(d)) try a.escape(d);
            for (a.escaping.items) |v| if (!a.lastMappable(v)) return false;
            if (a.escaping.items.len != 0) for (cfg.edgeOf(term, cd.exit_idx).args) |v| {
                if (a.vals.contains(v) and std.mem.indexOfScalar(VReg, a.escaping.items, v) == null) try a.escape(v);
            };
        }

        // Element size, vectorization factor, vector types.
        if (a.accesses.items.len == 0) return false;
        const vf: u8 = width / a.accesses.items[0].size;
        a.vf = vf;
        for (std.enums.values(ir.Type)) |ty| if (a.lanes & (@as(u8, 1) << @intCast(@backingInt(ty))) != 0) {
            const vt = vecType(ty, vf) orelse return false;
            if (!a.c.options.vector.supports(vt)) return false;
        };

        // Dependences between accesses (at least one a store).
        const acc = a.accesses.items;
        for (acc, 0..) |x, i| for (acc[i + 1 ..], i + 1..) |y, j| {
            if (!x.store and !y.store) continue;
            if (Lin.sameTerms(x.lin, y.lin)) {
                const d = (x.lin.c +% x.disp) -% (y.lin.c +% y.disp);
                if (d != 0 and @abs(d) < width) return false;
            } else {
                if (a.checks.items.len == max_checks) return false;
                try a.checks.append(a.c.gpa, .{ @intCast(i), @intCast(j) });
            }
        };
        return true;
    }

    fn escape(a: *Loop, v: VReg) opt.Error!void {
        try a.escaping.append(a.c.gpa, v);
    }

    /// The value can be recomputed in the mid block for the last vectorized iteration.
    fn lastMappable(a: *const Loop, v: VReg) bool {
        return switch (a.get(v).kind) {
            .outside, .uniform, .affine, .red_next => true,
            .data, .red_param => false,
        };
    }

    fn redIndex(a: *const Loop, v: VReg) ?u32 {
        for (a.params.items, 0..) |p, i| if (p.kind == .red and p.next == v) return @intCast(i);
        return null;
    }

    /// Classifies one body op; false rejects the loop.
    fn classify(a: *Loop, op: ir.Op, header_only: bool) opt.Error!bool {
        switch (op) {
            .@"const" => |x| try a.put(x.dst, .{ .kind = .uniform, .lin = .{ .c = x.imm } }),
            .fconst => |x| try a.put(x.dst, uniform(x.dst, x.ty)),
            .copy => |x| {
                var s = a.get(x.src);
                switch (s.kind) {
                    .red_param, .red_next => return false,
                    .outside => s.kind = .uniform,
                    .data => if (header_only) return false,
                    else => {},
                }
                s.param = none;
                try a.put(x.dst, s);
            },
            .arith => |x| {
                if (a.redIndex(x.dst)) |ri| {
                    if (header_only) return false;
                    return a.reduction(ri, x.op, x.a, x.b);
                }
                const va = a.get(x.a);
                const vb = a.get(x.b);
                if (va.kind == .red_param or va.kind == .red_next or vb.kind == .red_param or vb.kind == .red_next) return false;
                if (x.op == .sdiv or x.op == .srem) return false;
                if (isUni(va.kind) and isUni(vb.kind)) {
                    const both: ?i64 = if (va.lin.n == 0 and vb.lin.n == 0) ir.arith(x.op, va.lin.c, vb.lin.c) else null;
                    try a.put(x.dst, if (both) |k| .{ .kind = .uniform, .lin = .{ .c = k } } else uniform(x.dst, .i64));
                } else if (va.kind == .affine or vb.kind == .affine) {
                    if (va.kind == .data or vb.kind == .data) return false;
                    const r: Val = switch (x.op) {
                        .add, .sub => blk: {
                            const m: i64 = if (x.op == .add) 1 else -1;
                            const lin = va.lin.addScaled(vb.lin, m) orelse return false;
                            break :blk .{ .kind = .affine, .lin = lin, .stride = va.stride +% vb.stride *% m };
                        },
                        .mul, .shl => blk: {
                            // mul: either operand may be the constant; shl: only the count.
                            const aff, const k = if (va.kind == .affine) .{ va, vb } else .{ vb, va };
                            if (k.kind == .affine or k.lin.n != 0) return false;
                            if (x.op == .shl and va.kind != .affine) return false;
                            const m: i64 = if (x.op == .mul) k.lin.c else @bitCast(@as(u64, 1) << @intCast(k.lin.c & 63));
                            const lin = (Lin{}).addScaled(aff.lin, m) orelse return false;
                            break :blk .{ .kind = .affine, .lin = lin, .stride = aff.stride *% m };
                        },
                        else => return false,
                    };
                    try a.put(x.dst, r);
                } else {
                    if (header_only) return false;
                    switch (x.op) {
                        .add, .sub, .@"and", .@"or", .xor => {},
                        .mul => if (!a.c.options.vector.mul_v2i64) return false,
                        else => return false,
                    }
                    if (va.kind == .affine or vb.kind == .affine) return false;
                    try a.put(x.dst, a.data(.i64));
                }
            },
            .typed => |x| {
                if (x.ty != .f32 and x.ty != .f64) return false;
                const va = a.get(x.a);
                const vb = a.get(x.b);
                const ok = (va.kind == .data or isUni(va.kind)) and (vb.kind == .data or isUni(vb.kind));
                if (!ok) return false;
                if (isUni(va.kind) and isUni(vb.kind)) {
                    try a.put(x.dst, uniform(x.dst, x.ty));
                } else {
                    if (header_only) return false;
                    try a.put(x.dst, a.data(x.ty));
                }
            },
            .fcmp, .cvt => {
                var buf: [2]VReg = undefined;
                for (op.uses(&buf)) |v| if (!isUni(a.get(v).kind)) return false;
                const d = op.def().?;
                try a.put(d, uniform(d, a.c.types.items[d]));
            },
            .load => |x| {
                if (header_only or !try a.access(x.base, x.disp, x.ty, false)) return false;
                try a.put(x.dst, a.data(x.ty));
            },
            .store => |x| {
                if (header_only) return false;
                const vs = a.get(x.src);
                if (vs.kind != .data and !isUni(vs.kind)) return false;
                if (isUni(vs.kind)) _ = a.data(x.ty); // splat of the stored invariant
                return a.access(x.base, x.disp, x.ty, true);
            },
            else => return false,
        }
        return true;
    }

    fn access(a: *Loop, base: VReg, disp: i32, ty: ir.Type, store: bool) opt.Error!bool {
        if (ty != .i64 and ty != .f64 and ty != .f32) return false;
        const vb = a.get(base);
        if (vb.kind != .affine or vb.stride != ty.size()) return false;
        if (a.accesses.items.len != 0 and a.accesses.items[0].size != ty.size()) return false;
        if (a.accesses.items.len == max_accesses) return false;
        try a.accesses.append(a.c.gpa, .{ .base = base, .lin = vb.lin, .disp = disp, .store = store, .size = ty.size() });
        return true;
    }

    /// `r' = op r x` for reduction parameter `params[ri]`.
    fn reduction(a: *Loop, ri: u32, op: ir.Arith, x: VReg, y: VReg) opt.Error!bool {
        const p = &a.params.items[ri];
        if (p.seen) return false;
        switch (op) {
            .add, .@"and", .@"or", .xor => {},
            .sub => if (x != p.v) return false,
            .mul => if (!a.c.options.vector.mul_v2i64) return false,
            else => return false,
        }
        const other = if (x == p.v) y else x;
        if (other == p.v) return false;
        const vo = a.get(other);
        if (vo.kind != .data and !isUni(vo.kind)) return false;
        p.op = op;
        p.x = other;
        p.seen = true;
        _ = a.data(.i64);
        try a.put(p.next, .{ .kind = .red_next, .param = ri });
        return true;
    }

    // ---- transformation ----------------------------------------------------------------

    fn konst(a: *Loop, b: u32, k: i64) opt.Error!VReg {
        const v = try a.c.newVReg(.i64);
        try a.c.f.emit(b, .{ .@"const" = .{ .dst = v, .imm = k } });
        return v;
    }

    fn arith(a: *Loop, b: u32, op: ir.Arith, x: VReg, y: VReg) opt.Error!VReg {
        const v = try a.c.newVReg(.i64);
        try a.c.f.emit(b, .{ .arith = .{ .op = op, .dst = v, .a = x, .b = y } });
        return v;
    }

    fn arithK(a: *Loop, b: u32, op: ir.Arith, x: VReg, k: i64) opt.Error!VReg {
        return a.arith(b, op, x, try a.konst(b, k));
    }

    fn branch(a: *Loop, b: u32, cond: ir.CondCode, x: VReg, y: VReg, then: u32, targs: []const VReg, els: u32, eargs: []const VReg) opt.Error!void {
        try a.c.f.emit(b, if (targs.len == 0 and eargs.len == 0)
            .{ .cbr = .{ .cond = cond, .a = x, .b = y, .then = then, .@"else" = els } }
        else
            .{ .cbra = .{ .cond = cond, .a = x, .b = y, .then = then, .@"else" = els, .then_args = targs, .else_args = eargs } });
    }

    fn jump(a: *Loop, b: u32, to: u32, args: []const VReg) opt.Error!void {
        try a.c.f.emit(b, if (args.len == 0) .{ .br = to } else .{ .bra = .{ .target = to, .args = args } });
    }

    /// Loop value `v` as a scalar for `mode` (see `Mode`), cloning affine ops on demand.
    fn scalar(a: *Loop, mode: Mode, v: VReg) opt.Error!VReg {
        if (a.cmap.get(v)) |w| return w;
        const val = a.vals.get(v) orelse return v; // defined outside the loop
        const memo = switch (mode) {
            .pre => unreachable, // only uniform values (all in `cmap`) are used in P
            .body => return a.vmap.get(v).?,
            .iter0 => &a.m0,
            .last => &a.mlast,
        };
        if (memo.get(v)) |w| return w;
        std.debug.assert(val.kind == .affine);
        const w = if (val.param != none) blk: {
            const p = a.params.items[val.param];
            break :blk if (mode == .iter0) p.init else try a.arithK(a.mid_block, .sub, p.vl_next, p.step);
        } else try a.cloneOp(if (mode == .iter0) a.iter0_block else a.mid_block, a.gl.defOp(a.c.f, v).?, mode);
        try memo.put(a.c.gpa, v, w);
        return w;
    }

    /// Emits a copy of scalar op `op` into block `b` with operands mapped by `mode`.
    fn cloneOp(a: *Loop, b: u32, op: ir.Op, mode: Mode) opt.Error!VReg {
        var o = op;
        switch (o) {
            .@"const", .fconst => {},
            .copy => |*x| x.src = try a.scalar(mode, x.src),
            .cvt => |*x| x.src = try a.scalar(mode, x.src),
            .arith => |*x| {
                x.a = try a.scalar(mode, x.a);
                x.b = try a.scalar(mode, x.b);
            },
            .typed => |*x| {
                x.a = try a.scalar(mode, x.a);
                x.b = try a.scalar(mode, x.b);
            },
            .fcmp => |*x| {
                x.a = try a.scalar(mode, x.a);
                x.b = try a.scalar(mode, x.b);
            },
            else => unreachable,
        }
        const d = try a.c.newVReg(a.c.types.items[op.def().?]);
        ssa.setDef(&o, d);
        try a.c.f.emit(b, o);
        return d;
    }

    /// Loop value `v` as a vector of `lane`s in the vector body (invariants are splat in the
    /// vector preheader, once per value).
    fn vec(a: *Loop, v: VReg, lane: ir.Type) opt.Error!VReg {
        const k = a.get(v).kind;
        if (k == .data or k == .red_param) return a.vmap.get(v).?;
        const s = try a.scalar(.body, v);
        if (a.splats.get(s)) |w| return w;
        const vt = vecType(lane, a.vf).?;
        const w = try a.c.newVReg(vt);
        try a.c.f.emit(a.vph_block, .{ .splat = .{ .ty = vt, .dst = w, .src = s } });
        try a.splats.put(a.c.gpa, s, w);
        return w;
    }

    fn typedOp(a: *Loop, op: ir.TArith, ty: ir.Type, x: VReg, y: VReg) opt.Error!VReg {
        const r = try a.c.newVReg(ty);
        try a.c.f.emit(a.body_block, .{ .typed = .{ .op = op, .ty = ty, .dst = r, .a = x, .b = y } });
        return r;
    }

    /// Emits body op `op` into the vector body.
    fn vectorOp(a: *Loop, op: ir.Op) opt.Error!void {
        const f = a.c.f;
        const body = a.body_block;
        if (op == .store) {
            const x = op.store;
            const src = try a.vec(x.src, x.ty);
            try f.emit(body, .{ .store = .{ .base = try a.scalar(.body, x.base), .disp = x.disp, .src = src, .ty = vecType(x.ty, a.vf).? } });
            return;
        }
        const d = op.def().?;
        const val = a.vals.get(d).?;
        const w: VReg = switch (val.kind) {
            .uniform => return, // cloned into P
            .affine => try a.cloneOp(body, op, .body),
            .data => switch (op) {
                .copy => |x| a.vmap.get(x.src).?,
                .load => |x| blk: {
                    const vt = vecType(x.ty, a.vf).?;
                    const r = try a.c.newVReg(vt);
                    try f.emit(body, .{ .load = .{ .dst = r, .base = try a.scalar(.body, x.base), .disp = x.disp, .ty = vt } });
                    break :blk r;
                },
                .arith => |x| try a.typedOp(tarith(x.op), .v2i64, try a.vec(x.a, .i64), try a.vec(x.b, .i64)),
                .typed => |x| blk: {
                    const vt = vecType(x.ty, a.vf).?;
                    break :blk try a.typedOp(x.op, vt, try a.vec(x.a, x.ty), try a.vec(x.b, x.ty));
                },
                else => unreachable,
            },
            .red_next => blk: {
                const p = &a.params.items[val.param];
                p.vl_next = try a.typedOp(tarith(p.op), .v2i64, p.vl, try a.vec(p.x, .i64));
                break :blk p.vl_next;
            },
            .outside, .red_param => unreachable,
        };
        try a.vmap.put(a.c.gpa, d, w);
    }

    fn transform(a: *Loop, ren: *Renames) opt.Error!void {
        const c = a.c;
        const f = c.f;
        const gpa = c.gpa;
        const ira = c.irAlloc();
        const cd = a.cd;
        const vf = a.vf;
        const prim = a.params.items[a.prim];
        const s = prim.step;
        const log_s: i64 = @ctz(@as(u64, @bitCast(s)));
        const pre = cd.pre;
        const fe = c.freq.items[pre];
        const fl = c.freq.items[cd.body];
        const fv: ?u64 = if (fl) |x| x / vf else null;
        const term = cfg.terminator(f, cd.hdr).*; // the original exit test
        const exit_args = cfg.edgeOf(term, cd.exit_idx).args;

        const scalar_b = try c.addBlock("vec.scalar", null);
        const count_b = if (a.rel == .ne) pre else try c.addBlock("vec.count", fe);
        var check_buf: [2 * max_checks]u32 = undefined;
        const check_b = check_buf[0 .. 2 * a.checks.items.len];
        for (check_b) |*b| b.* = try c.addBlock("vec.alias", fe);
        const vph = try c.addBlock("vec.ph", fe);
        const body = try c.addBlock("vec.body", fv);
        const mid = try c.addBlock("vec.mid", fe);
        a.vph_block = vph;
        a.body_block = body;
        a.mid_block = mid;
        try a.jump(scalar_b, cd.hdr, a.init);

        // P: uniform values of the loop, y0 and the first guard.
        f.blocks.items[pre].ops.items.len -= 1;
        for (a.params.items) |p| if (p.kind == .uniform) try a.cmap.put(gpa, p.v, p.init);
        for (a.bodyBlocks()) |b| {
            const ops = f.blocks.items[b].ops.items;
            for (ops[0 .. ops.len - 1]) |op| if (op.def()) |d| if (a.vals.get(d).?.kind == .uniform) {
                try a.cmap.put(gpa, d, try a.cloneOp(pre, op, .pre));
            };
        }
        const n = try a.scalar(.pre, a.n);
        const y0 = if (a.u == 0) prim.init else try a.arithK(pre, .add, prim.init, a.u);
        if (a.rel != .ne) try a.branch(pre, a.rel, y0, n, count_b, &.{}, scalar_b, &.{});

        // kv: a lower bound on the iterations that run (y0 + k*s REL n for k < cnt), rounded
        // down to a multiple of VF (module comment).
        const r = try a.arith(count_b, .sub, n, y0);
        const cnt = switch (a.rel) {
            .lt, .ult => if (s == 1) r else try a.arithK(count_b, .add, try a.arithK(count_b, .shr, try a.arithK(count_b, .sub, r, 1), log_s), 1),
            .le, .ule => if (s == 1) try a.arithK(count_b, .add, r, 1) else try a.arithK(count_b, .add, try a.arithK(count_b, .shr, r, log_s), 1),
            .ne => if (s == 1) r else try a.arithK(count_b, .shr, r, log_s),
            else => unreachable,
        };
        const kv = try a.arithK(count_b, .@"and", cnt, -@as(i64, vf));
        try a.branch(count_b, .eq, kv, try a.konst(count_b, 0), scalar_b, &.{}, if (check_b.len != 0) check_b[0] else vph, &.{});

        // Run-time alias checks: scalar unless d == 0 or |d| >= 16.
        for (a.checks.items, 0..) |pair, i| {
            const b = check_b[2 * i];
            const bz = check_b[2 * i + 1];
            const next = if (2 * i + 2 < check_b.len) check_b[2 * i + 2] else vph;
            a.iter0_block = b;
            const x = a.accesses.items[pair[0]];
            const y = a.accesses.items[pair[1]];
            var d = try a.arith(b, .sub, try a.scalar(.iter0, x.base), try a.scalar(.iter0, y.base));
            const dd = @as(i64, x.disp) - y.disp;
            if (dd != 0) d = try a.arithK(b, .add, d, dd);
            const w = try a.arithK(b, .add, d, width - 1);
            try a.branch(b, .ult, w, try a.konst(b, 2 * width - 1), bz, &.{}, next, &.{});
            try a.branch(bz, .eq, d, try a.konst(bz, 0), next, &.{}, scalar_b, &.{});
        }

        // Vector preheader and body.
        const q_end = try a.arith(vph, .add, prim.init, if (s == 1) kv else try a.arithK(vph, .shl, kv, log_s));
        var nvl: usize = 0;
        for (a.params.items) |p| nvl += @intFromBool(p.kind != .uniform);
        const vl_params = try ira.alloc(VReg, nvl);
        const vl_init = try ira.alloc(VReg, nvl);
        const vl_next = try ira.alloc(VReg, nvl);
        var k: usize = 0;
        for (a.params.items) |*p| {
            switch (p.kind) {
                .uniform => continue,
                .iv => {
                    p.vl = try c.newVReg(.i64);
                    vl_init[k] = p.init;
                },
                .red => {
                    p.vl = try c.newVReg(.v2i64);
                    vl_init[k] = try c.newVReg(.v2i64);
                    const id: i64 = switch (p.op) {
                        .@"and" => -1,
                        .mul => 1,
                        else => 0,
                    };
                    try f.emit(vph, .{ .splat = .{ .ty = .v2i64, .dst = vl_init[k], .src = try a.konst(vph, id) } });
                },
            }
            vl_params[k] = p.vl;
            try a.vmap.put(gpa, p.v, p.vl);
            k += 1;
        }
        f.blocks.items[body].params = vl_params;
        for (a.bodyBlocks()) |b| {
            const ops = f.blocks.items[b].ops.items;
            for (ops[0 .. ops.len - 1]) |op| try a.vectorOp(op);
        }
        k = 0;
        for (a.params.items) |*p| {
            if (p.kind == .uniform) continue;
            if (p.kind == .iv) p.vl_next = try a.arith(body, .add, p.vl, try a.konst(vph, p.step *% vf));
            vl_next[k] = p.vl_next;
            k += 1;
        }
        const prim_next = a.params.items[a.prim].vl_next;
        try a.branch(body, .ne, prim_next, q_end, body, vl_next, mid, &.{});
        try a.jump(vph, body, vl_init);

        // Mid block: reduce the accumulators, then continue with the scalar loop.
        const state = try ira.alloc(VReg, a.params.items.len);
        for (a.params.items, state) |*p, *st| {
            if (p.kind == .red) {
                var acc = p.init;
                for (0..vf) |lane| {
                    const e = try c.newVReg(.i64);
                    try f.emit(mid, .{ .extract = .{ .ty = .v2i64, .dst = e, .src = p.vl_next, .lane = @intCast(lane) } });
                    acc = try a.arith(mid, if (p.op == .sub) .add else p.op, acc, e);
                }
                p.mid = acc;
            }
            st.* = switch (p.kind) {
                .iv => p.vl_next,
                .red => p.mid,
                .uniform => p.init,
            };
        }
        if (cd.shape == .top) {
            try a.jump(mid, cd.hdr, state);
        } else {
            // The original test on x_{kv-1} = q_kv + u decides whether iteration kv runs.
            for (a.params.items) |p| switch (p.kind) {
                .iv => try a.mlast.put(gpa, p.next, p.vl_next),
                .red => try a.mlast.put(gpa, p.next, p.mid),
                .uniform => {},
            };
            const xq = if (a.u == 0) prim_next else try a.arithK(mid, .add, prim_next, a.u);
            const cond = switch (term) {
                .cbr => |x| x.cond,
                .cbra => |x| x.cond,
                else => unreachable,
            };
            var exit_to = cd.exit;
            var out: []VReg = undefined;
            if (a.escaping.items.len == 0) {
                out = try ira.alloc(VReg, exit_args.len);
                for (exit_args, out) |v, *o| o.* = try a.scalar(.last, v);
            } else {
                // Values of L are used outside it: L and the mid block reach X through a
                // merge block whose parameters replace those values outside L.
                const esc = try ira.dupe(VReg, a.escaping.items);
                const m = try c.addBlock("vec.exit", fe);
                const mparams = try ira.alloc(VReg, esc.len);
                out = try ira.alloc(VReg, esc.len);
                const start: u32 = @intCast(ren.pairs.items.len);
                for (esc, mparams, out) |v, *mp, *o| {
                    mp.* = try c.newVReg(c.types.items[v]);
                    o.* = try a.scalar(.last, v);
                    try ren.pairs.append(gpa, .{ v, mp.* });
                }
                try ren.owner.put(gpa, cd.hdr, .{ start, @intCast(ren.pairs.items.len) });
                f.blocks.items[m].params = mparams;
                const margs = try ira.alloc(VReg, exit_args.len);
                for (exit_args, margs) |v, *ma| ma.* = if (std.mem.indexOfScalar(VReg, esc, v)) |i| mparams[i] else v;
                try a.jump(m, cd.exit, margs);
                cfg.setEdge(cfg.terminator(f, cd.hdr), cd.exit_idx, m, esc);
                exit_to = m;
            }
            const xa, const xb = if (a.x_is_a) .{ xq, n } else .{ n, xq };
            if (cd.exit_idx == 1)
                try a.branch(mid, cond, xa, xb, cd.hdr, state, exit_to, out)
            else
                try a.branch(mid, cond, xa, xb, exit_to, out, cd.hdr, state);
        }

        // Profile estimates: the scalar loop now runs about VF/2 iterations per entry.
        if (fe) |entries| if (fl) |iters| {
            const epi = @min(iters, entries *| (vf / 2));
            c.freq.items[cd.body] = epi;
            c.taken.items[body] = fv.? -| entries;
            if (cd.shape == .top) {
                c.freq.items[cd.hdr] = epi +| entries;
                c.taken.items[cd.hdr] = if (cd.exit_idx == 1) epi else entries;
            } else {
                const exits = @min(entries, epi);
                c.taken.items[cd.hdr] = if (cd.exit_idx == 1) epi - exits else exits;
            }
        };
    }

    /// Recognizes the exit test (see the module comment); false rejects the loop.
    fn exitTest(a: *Loop) bool {
        const term = cfg.terminator(a.c.f, a.cd.hdr).*;
        const cond: ir.CondCode, const ta: VReg, const tb: VReg = switch (term) {
            .cbr => |x| .{ x.cond, x.a, x.b },
            .cbra => |x| .{ x.cond, x.a, x.b },
            else => return false,
        };
        const va = a.get(ta);
        const vb = a.get(tb);
        var x: Val = undefined;
        if (va.kind == .affine and isUni(vb.kind)) {
            x = va;
            a.n = tb;
            a.rel = cond;
            a.x_is_a = true;
        } else if (vb.kind == .affine and isUni(va.kind)) {
            x = vb;
            a.n = ta;
            a.rel = mirror(cond);
            a.x_is_a = false;
        } else return false;
        if (a.cd.exit_idx == 0) a.rel = negate(a.rel); // the loop continues on `else`
        switch (a.rel) {
            .lt, .ult, .le, .ule, .ne => {},
            else => return false,
        }
        const s = x.stride;
        if (s <= 0 or s > 1 << 24 or !std.math.isPowerOfTwo(s)) return false;
        for (a.params.items, 0..) |p, i| {
            if (p.kind != .iv or p.step != s) continue;
            const q = a.get(p.v);
            if (!Lin.sameTerms(q.lin, x.lin)) continue;
            a.prim = @intCast(i);
            const off = x.lin.c -% q.lin.c;
            a.u = if (a.cd.shape == .top) off else off -% s;
            return true;
        }
        return false;
    }
};

fn vecType(lane: ir.Type, vf: u8) ?ir.Type {
    return switch (lane) {
        .i64 => if (vf == 2) .v2i64 else null,
        .f64 => if (vf == 2) .v2f64 else null,
        .f32 => if (vf == 4) .v4f32 else null,
        else => null,
    };
}

fn tarith(op: ir.Arith) ir.TArith {
    return switch (op) {
        .add => .add,
        .sub => .sub,
        .mul => .mul,
        .@"and" => .@"and",
        .@"or" => .@"or",
        .xor => .xor,
        else => unreachable,
    };
}

fn mirror(cc: ir.CondCode) ir.CondCode {
    return switch (cc) {
        .lt => .gt,
        .gt => .lt,
        .le => .ge,
        .ge => .le,
        .ult => .ugt,
        .ugt => .ult,
        .ule => .uge,
        .uge => .ule,
        .eq, .ne => cc,
    };
}

fn negate(cc: ir.CondCode) ir.CondCode {
    return switch (cc) {
        .eq => .ne,
        .ne => .eq,
        .lt => .ge,
        .ge => .lt,
        .le => .gt,
        .gt => .le,
        .ult => .uge,
        .uge => .ult,
        .ule => .ugt,
        .ugt => .ule,
    };
}

/// Deferred renames of escaping values to merge-block parameters (applied once, after all
/// loops, so the pass stays linear).
const Renames = struct {
    pairs: std.ArrayList([2]VReg) = .empty,
    /// Bottom-shape loop block -> range of `pairs` whose values it defines (not renamed there).
    owner: std.AutoHashMapUnmanaged(u32, [2]u32) = .empty,

    fn deinit(r: *Renames, gpa: std.mem.Allocator) void {
        r.pairs.deinit(gpa);
        r.owner.deinit(gpa);
    }

    fn apply(r: *Renames, c: *opt.Ctx) opt.Error!void {
        if (r.pairs.items.len == 0) return;
        const map = try c.gpa.alloc(VReg, c.f.num_vregs);
        defer c.gpa.free(map);
        for (map, 0..) |*m, i| m.* = @intCast(i);
        for (r.pairs.items) |p| map[p[0]] = p[1];
        for (c.f.blocks.items, 0..) |*b, bi| {
            const own = r.owner.get(@intCast(bi));
            if (own) |o| for (r.pairs.items[o[0]..o[1]]) |p| {
                map[p[0]] = p[0];
            };
            for (b.ops.items) |*op| try opt.mapUses(op, c.irAlloc(), map);
            if (own) |o| for (r.pairs.items[o[0]..o[1]]) |p| {
                map[p[0]] = p[1];
            };
        }
    }
};

// ---- tests -------------------------------------------------------------------------------

const t = std.testing;
const testgen = @import("testgen.zig");

/// Memory the test functions address: pointer arguments are byte offsets into it.
const buf_words = 512;
const Buf = [buf_words]u64;
const Fill = enum { i64, f64, f32 };

fn fillBuf(buf: *align(16) Buf, seed: u64, fill: Fill) void {
    var prng = std.Random.DefaultPrng.init(seed);
    const r = prng.random();
    for (buf) |*w| w.* = switch (fill) {
        .i64 => if (r.boolean()) @bitCast(r.intRangeAtMost(i64, -1000, 1000)) else r.int(u64),
        .f64 => @bitCast(@as(f64, @floatFromInt(r.intRangeAtMost(i32, -64, 64))) / 8),
        .f32 => @as(u64, @as(u32, @bitCast(@as(f32, @floatFromInt(r.intRangeAtMost(i32, -64, 64))) / 4))) |
            @as(u64, @as(u32, @bitCast(@as(f32, @floatFromInt(r.intRangeAtMost(i32, -64, 64))) / 4))) << 32,
    };
}

fn exec(f: *const ir.Func, args: []const i64, ptrs: u8, buf: *align(16) Buf) anyerror!i64 {
    var a: [8]i64 = undefined;
    for (args, 0..) |x, i| a[i] = if (ptrs >> @intCast(i) & 1 != 0) @intCast(@intFromPtr(buf) + @as(usize, @intCast(x))) else x;
    return ir.interpret(f, a[0..args.len], 200_000);
}

fn sameFloat(comptime F: type, x: F, y: F) bool {
    const U = @Int(.unsigned, @bitSizeOf(F));
    return @as(U, @bitCast(x)) == @as(U, @bitCast(y)) or (std.math.isNan(x) and std.math.isNan(y));
}

/// `f` computes what `ref` computes (result and memory, any NaN equal to any NaN) for every
/// argument set on which `ref` finishes.
fn expectSameOn(ref: *const ir.Func, f: *const ir.Func, ptrs: u8, arg_sets: []const []const i64, fill: Fill) !void {
    var want_buf: Buf align(16) = undefined;
    var got_buf: Buf align(16) = undefined;
    for (arg_sets, 0..) |args, k| {
        fillBuf(&want_buf, k, fill);
        fillBuf(&got_buf, k, fill);
        const want = exec(ref, args, ptrs, &want_buf) catch |e| {
            if (e == error.StepLimit) continue;
            return e;
        };
        const got = try exec(f, args, ptrs, &got_buf);
        errdefer std.debug.print("vectorize: mismatch for args {any}\n", .{args});
        try t.expectEqual(want, got);
        for (want_buf, got_buf) |x, y| if (x != y) switch (fill) {
            .i64 => try t.expectEqual(x, y),
            .f64 => try t.expect(sameFloat(f64, @bitCast(x), @bitCast(y))),
            .f32 => for (0..2) |h| {
                const sh: u6 = @intCast(32 * h);
                try t.expect(sameFloat(f32, @bitCast(@as(u32, @truncate(x >> sh))), @bitCast(@as(u32, @truncate(y >> sh)))));
            },
        };
    }
}

/// Builds SSA, runs the vectorizer for `target` (checking the IR after it), compares with
/// the original on `arg_sets` and expects `want` vectorized loops; then the same through
/// the whole `.o2` pipeline. Returns the vectorized function printed (caller frees).
fn checkVec(src: []const u8, target: opt.VectorTarget, ptrs: u8, arg_sets: []const []const i64, fill: Fill, want: u32) ![]u8 {
    return checkVecFull(t.allocator, src, target, ptrs, arg_sets, fill, want, true);
}

fn checkVecFull(gpa: std.mem.Allocator, src: []const u8, target: opt.VectorTarget, ptrs: u8, arg_sets: []const []const i64, fill: Fill, want: u32, full: bool) ![]u8 {
    var ref = try ir.parse(gpa, src);
    defer ref.deinit();
    var f = try ir.parse(gpa, src);
    defer f.deinit();
    const options: opt.Options = .{ .vector = target, .verify_each = true };
    var stats: opt.Stats = .{};
    {
        var c = try opt.Ctx.init(gpa, &f, &options, &stats);
        defer c.deinit();
        try ssa.build(&c);
        try c.check("ssa.build");
        _ = try run(&c);
        try c.check("vectorize");
    }
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(gpa);
    try f.print(gpa, &out);
    errdefer std.debug.print("vectorize: failure on\n{s}\nvectorized:\n{s}\n", .{ src, out.items });
    try t.expectEqual(want, stats.vectorized_loops);
    try expectSameOn(&ref, &f, ptrs, arg_sets, fill);
    if (!full) return out.toOwnedSlice(gpa);
    var g = try ir.parse(gpa, src);
    defer g.deinit();
    _ = try opt.optimize(gpa, &g, .{ .level = .o2, .vector = target, .verify_each = true });
    try t.expect(!g.hasBlockParams());
    try expectSameOn(&ref, &g, ptrs, arg_sets, fill);
    return out.toOwnedSlice(gpa);
}

fn expectVec(src: []const u8, target: opt.VectorTarget, ptrs: u8, arg_sets: []const []const i64, fill: Fill, want: u32) !void {
    t.allocator.free(try checkVec(src, target, ptrs, arg_sets, fill, want));
}

const sum_i64_src =
    \\func sum_i64 2
    \\e:
    \\  v2 = const 0
    \\  v3 = const 0
    \\  v4 = const 8
    \\  v5 = const 1
    \\  cbr ge v2 v1 done loop
    \\loop:
    \\  v6 = mul v2 v4
    \\  v7 = add v0 v6
    \\  v8 = load v7 0
    \\  v3 = add v3 v8
    \\  v2 = add v2 v5
    \\  cbr lt v2 v1 loop done
    \\done:
    \\  ret v3
;
const add_f64_src =
    \\func add_f64 4
    \\e:
    \\  v4 = const 0
    \\  v5 = const 8
    \\  v6 = const 1
    \\  cbr ge v4 v3 done loop
    \\loop:
    \\  v7 = mul v4 v5
    \\  v8 = add v0 v7
    \\  v9 = add v1 v7
    \\  v10 = add v2 v7
    \\  v11 = load.f64 v8 0
    \\  v12 = load.f64 v9 0
    \\  v13 = add.f64 v11 v12
    \\  store.f64 v10 0 v13
    \\  v4 = add v4 v6
    \\  cbr lt v4 v3 loop done
    \\done:
    \\  ret v4
;
const saxpy_src =
    \\func saxpy 3
    \\e:
    \\  v3 = const 0
    \\  v4 = const 4
    \\  v5 = const 1
    \\  v6 = fconst.f32 2.5
    \\  cbr ge v3 v2 done loop
    \\loop:
    \\  v7 = mul v3 v4
    \\  v8 = add v0 v7
    \\  v9 = add v1 v7
    \\  v10 = load.f32 v8 0
    \\  v11 = load.f32 v9 0
    \\  v12 = mul.f32 v10 v6
    \\  v13 = add.f32 v12 v11
    \\  store.f32 v9 0 v13
    \\  v3 = add v3 v5
    \\  cbr lt v3 v2 loop done
    \\done:
    \\  ret v3
;
const dot_f64_src =
    \\func dot_f64 3
    \\e:
    \\  v3 = const 0
    \\  v4 = const 8
    \\  v5 = const 1
    \\  v6 = fconst.f64 0
    \\  cbr ge v3 v2 done loop
    \\loop:
    \\  v7 = mul v3 v4
    \\  v8 = add v0 v7
    \\  v9 = add v1 v7
    \\  v10 = load.f64 v8 0
    \\  v11 = load.f64 v9 0
    \\  v12 = mul.f64 v10 v11
    \\  v6 = add.f64 v6 v12
    \\  v3 = add v3 v5
    \\  cbr lt v3 v2 loop done
    \\done:
    \\  v13 = cvt.i64.f64 v6
    \\  ret v13
;

/// Trip counts around the vectorization factors: zero, one, VF-1, VF, VF+1, ...
const trips = [_]i64{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 40, -3, std.math.minInt(i64) };

fn argSets(comptime n_ptrs: usize, comptime offs: [n_ptrs]i64) [trips.len][n_ptrs + 1]i64 {
    var sets: [trips.len][n_ptrs + 1]i64 = undefined;
    for (&sets, trips) |*s, n| {
        s[0..n_ptrs].* = offs;
        s[n_ptrs] = n;
    }
    return sets;
}

fn slices(comptime N: usize, comptime M: usize, sets: *const [N][M]i64) [N][]const i64 {
    var out: [N][]const i64 = undefined;
    for (&out, sets) |*o, *s| o.* = s;
    return out;
}

test "vectorize: sum_i64 becomes a v2i64 reduction" {
    const sets = argSets(1, .{512});
    const ir_text = try checkVec(sum_i64_src, opt.VectorTarget.aarch64, 1, &slices(trips.len, 2, &sets), .i64, 1);
    defer t.allocator.free(ir_text);
    try t.expect(std.mem.indexOf(u8, ir_text, "load.v2i64") != null);
    try t.expect(std.mem.indexOf(u8, ir_text, "add.v2i64") != null);
    try t.expect(std.mem.indexOf(u8, ir_text, "extract.v2i64") != null);
}

test "vectorize: add_f64 with disjoint, equal and overlapping arrays" {
    // c = a + 8 overlaps by one element: the run-time check must take the scalar loop.
    const offs = [_][3]i64{ .{ 512, 1024, 1536 }, .{ 512, 1024, 512 }, .{ 512, 1024, 520 }, .{ 512, 1024, 504 }, .{ 512, 1024, 528 }, .{ 512, 520, 1536 }, .{ 512, 512, 512 } };
    inline for (offs) |o| {
        const sets = argSets(3, o);
        const ir_text = try checkVec(add_f64_src, opt.VectorTarget.aarch64, 7, &slices(trips.len, 4, &sets), .f64, 1);
        defer t.allocator.free(ir_text);
        try t.expect(std.mem.indexOf(u8, ir_text, "add.v2f64") != null);
    }
}

test "vectorize: saxpy_f32 uses v4f32 and a splat of the invariant" {
    const sets = argSets(2, .{ 512, 1024 });
    const ir_text = try checkVec(saxpy_src, opt.VectorTarget.aarch64, 3, &slices(trips.len, 3, &sets), .f32, 1);
    defer t.allocator.free(ir_text);
    try t.expect(std.mem.indexOf(u8, ir_text, "splat.v4f32") != null);
    try t.expect(std.mem.indexOf(u8, ir_text, "store.v4f32") != null);
    // Same array (in place, d == 0) and arrays 4 bytes apart (unsafe, scalar at run time).
    const s2 = argSets(2, .{ 512, 512 });
    try expectVec(saxpy_src, opt.VectorTarget.aarch64, 3, &slices(trips.len, 3, &s2), .f32, 1);
    const s3 = argSets(2, .{ 512, 516 });
    try expectVec(saxpy_src, opt.VectorTarget.aarch64, 3, &slices(trips.len, 3, &s3), .f32, 1);
}

test "vectorize: dot_f64 (float reduction) and unsupported targets stay scalar" {
    const sets = argSets(2, .{ 512, 1024 });
    try expectVec(dot_f64_src, opt.VectorTarget.aarch64, 3, &slices(trips.len, 3, &sets), .f64, 0);
    const s1 = argSets(1, .{512});
    try expectVec(sum_i64_src, .{ .v2f64 = true }, 1, &slices(trips.len, 2, &s1), .i64, 0);
    const s3 = argSets(3, .{ 512, 1024, 1536 });
    try expectVec(add_f64_src, .{ .v2i64 = true, .v4f32 = true }, 7, &slices(trips.len, 4, &s3), .f64, 0);
}

/// Trip counts without negative `n` (for unsigned exit tests, where they mean "forever").
const trips_pos = trips[0..14];

fn shiftSrc(comptime disp: i32) []const u8 {
    return std.fmt.comptimePrint(
        \\func shift 2
        \\e:
        \\  v2 = const 0
        \\  v3 = const 8
        \\  v4 = const 1
        \\  cbr ge v2 v1 done loop
        \\loop:
        \\  v5 = mul v2 v3
        \\  v6 = add v0 v5
        \\  v7 = load v6 {d}
        \\  store v6 0 v7
        \\  v2 = add v2 v4
        \\  cbr lt v2 v1 loop done
        \\done:
        \\  ret v2
    , .{disp});
}

test "vectorize: same-array distances below one vector are rejected statically" {
    const sets = argSets(1, .{1024});
    const sl = slices(trips.len, 2, &sets);
    inline for (.{ .{ -8, 0 }, .{ 8, 0 }, .{ -15, 0 }, .{ 0, 1 }, .{ 16, 1 }, .{ -16, 1 }, .{ 24, 1 } }) |c| {
        try expectVec(shiftSrc(c[0]), opt.VectorTarget.aarch64, 1, &sl, .i64, c[1]);
    }
}

test "vectorize: bottom-tested loop without guard, values used after it" {
    // Runs once for n <= 1; the reduction, the counter, an affine and an invariant value of
    // the loop reach `done` through the merge block.
    const src =
        \\func esc 2
        \\e:
        \\  v2 = const 0
        \\  v3 = const 8
        \\  v4 = const 1
        \\  v5 = const 0
        \\  br loop
        \\loop:
        \\  v6 = mul v2 v3
        \\  v7 = add v0 v6
        \\  v8 = load v7 0
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
    const sets = argSets(1, .{1024});
    const ir_text = try checkVec(src, opt.VectorTarget.aarch64, 1, &slices(trips.len, 2, &sets), .i64, 1);
    defer t.allocator.free(ir_text);
    try t.expect(std.mem.indexOf(u8, ir_text, "vec.exit") != null);
    try t.expect(std.mem.indexOf(u8, ir_text, "xor.v2i64") != null);
}

test "vectorize: top-tested product needs mul_v2i64" {
    const src =
        \\func prod 2
        \\e:
        \\  v2 = const 0
        \\  v3 = const 3
        \\  v4 = const 1
        \\  v5 = const 1
        \\  br head
        \\head:
        \\  cbr ult v2 v1 body done
        \\body:
        \\  v6 = shl v2 v3
        \\  v7 = add v0 v6
        \\  v8 = load v7 0
        \\  v9 = or v8 v4
        \\  v5 = mul v5 v9
        \\  v2 = add v2 v4
        \\  br head
        \\done:
        \\  v10 = sub v5 v2
        \\  ret v10
    ;
    const sets = argSets(1, .{1024});
    const sl = slices(trips.len, 2, &sets);
    try expectVec(src, opt.VectorTarget.aarch64, 1, sl[0..trips_pos.len], .i64, 0);
    var mul = opt.VectorTarget.aarch64;
    mul.mul_v2i64 = true;
    const ir_text = try checkVec(src, mul, 1, sl[0..trips_pos.len], .i64, 1);
    defer t.allocator.free(ir_text);
    try t.expect(std.mem.indexOf(u8, ir_text, "mul.v2i64") != null);
}

test "vectorize: loops with calls stay scalar" {
    const src = try std.fmt.allocPrint(t.allocator,
        \\func callee_loop 2
        \\e:
        \\  v2 = const 0
        \\  v3 = const 8
        \\  v4 = const 1
        \\  cbr ge v2 v1 done loop
        \\loop:
        \\  v5 = mul v2 v3
        \\  v6 = add v0 v5
        \\  v7 = load v6 0
        \\  v8 = call {d} v7 v4
        \\  store v6 0 v7
        \\  v2 = add v2 v4
        \\  cbr lt v2 v1 loop done
        \\done:
        \\  ret v2
    , .{@intFromPtr(&testgen.callee)});
    defer t.allocator.free(src);
    const sets = argSets(1, .{1024});
    try expectVec(src, opt.VectorTarget.aarch64, 1, &slices(trips.len, 2, &sets), .i64, 0);
}

test "vectorize: two loops in a row, the first one's results feed the second" {
    // l1 exits straight into l2's header; l2 reads l1's reduction (an invariant there).
    const src =
        \\func seq 3
        \\e:
        \\  v3 = const 0
        \\  v4 = const 8
        \\  v5 = const 1
        \\  v6 = const 0
        \\  v14 = add v2 v2
        \\  br l1
        \\l1:
        \\  v7 = mul v3 v4
        \\  v8 = add v0 v7
        \\  v9 = load v8 0
        \\  v6 = add v6 v9
        \\  v3 = add v3 v5
        \\  cbr lt v3 v2 l1 l2
        \\l2:
        \\  v10 = mul v3 v4
        \\  v11 = add v1 v10
        \\  v12 = load v11 0
        \\  v13 = xor v12 v6
        \\  store v11 0 v13
        \\  v3 = add v3 v5
        \\  cbr lt v3 v14 l2 done
        \\done:
        \\  v15 = sub v6 v3
        \\  ret v15
    ;
    for ([_]i64{ 512, 520, 2048 }) |b| {
        var sets: [trips.len][3]i64 = undefined;
        for (&sets, trips) |*st, n| st.* = .{ 512, b, n };
        const ir_text = try checkVec(src, opt.VectorTarget.aarch64, 3, &slices(trips.len, 3, &sets), .i64, 2);
        defer t.allocator.free(ir_text);
        try t.expect(std.mem.indexOf(u8, ir_text, "vec.exit") != null); // l1's sum leaves through it
    }
}

test "vectorize: inner loop of a nest, with the outer counter as an invariant" {
    const src =
        \\func nest 3
        \\e:
        \\  v3 = const 0
        \\  v4 = const 0
        \\  v5 = const 8
        \\  v6 = const 1
        \\  cbr ge v3 v1 done orow
        \\orow:
        \\  v7 = mul v3 v2
        \\  v7 = mul v7 v5
        \\  v8 = add v0 v7
        \\  v9 = const 0
        \\  v10 = const 0
        \\  cbr ge v9 v2 olatch inner
        \\inner:
        \\  v11 = mul v9 v5
        \\  v12 = add v8 v11
        \\  v13 = load v12 0
        \\  v14 = add v13 v3
        \\  store v12 0 v14
        \\  v10 = add v10 v14
        \\  v9 = add v9 v6
        \\  cbr lt v9 v2 inner olatch
        \\olatch:
        \\  v4 = add v4 v10
        \\  v3 = add v3 v6
        \\  cbr lt v3 v1 orow done
        \\done:
        \\  ret v4
    ;
    const shapes = [_][2]i64{ .{ 0, 5 }, .{ 3, 0 }, .{ 3, 1 }, .{ 2, 3 }, .{ 4, 7 }, .{ 5, 40 }, .{ 3, -2 }, .{ -1, 9 }, .{ 6, 17 } };
    var sets: [shapes.len][3]i64 = undefined;
    for (&sets, shapes) |*st, rc| st.* = .{ 512, rc[0], rc[1] };
    const ir_text = try checkVec(src, opt.VectorTarget.aarch64, 1, &slices(shapes.len, 3, &sets), .i64, 1);
    t.allocator.free(ir_text);
}

/// `run` with the aarch64 vector target (the testgen harness builds options without one).
fn runAarch64(c: *opt.Ctx) opt.Error!bool {
    const saved = c.options;
    defer c.options = saved;
    var o = saved.*;
    o.vector = opt.VectorTarget.aarch64;
    c.options = &o;
    return run(c);
}

test "vectorize: testgen programs, alone and in the pipeline" {
    var da: std.heap.DebugAllocator(.{ .stack_trace_frames = 0 }) = .init;
    defer t.expect(da.deinit() == .ok) catch @panic("vectorize: leak");
    const gpa = da.allocator();
    try testgen.fuzzPass(gpa, 0x7ec, 40, .{}, &.{}, runAarch64);
    try testgen.fuzzPass(gpa, 0x7ed, 40, .{ .calls = false, .floats = false }, &.{}, runAarch64);
    var prng = std.Random.DefaultPrng.init(0x7ee);
    var src: std.ArrayList(u8) = .empty;
    defer src.deinit(gpa);
    for (0..40) |i| {
        src.clearRetainingCapacity();
        try testgen.program(prng.random(), gpa, .{ .calls = i % 2 == 0 }, &src);
        _ = testgen.checkOptimize(gpa, src.items, .{ .level = .o2, .vector = opt.VectorTarget.aarch64, .verify_each = true }) catch |e| {
            std.debug.print("vectorize: pipeline failure on\n{s}\n", .{src.items});
            return e;
        };
    }
}

// ---- random array loops ------------------------------------------------------------------

/// A random loop written by `genLoop`: what the analysis should decide and how to bound it.
const GenCase = struct {
    /// Every feature is one the vectorizer accepts (target support aside).
    ok: bool = true,
    elem: ir.Type,
    uses_mul: bool = false,
    shape: Shape,
    guarded: bool,
    /// Induction variable the exit test reads: step, test offset, guard offset.
    step: i64,
    toff: i64 = 0,
    gtoff: i64 = 0,
    /// Continue iff `cont(x, n)`.
    cont: ir.CondCode,
    /// Constant start of the counter (else the argument v4).
    i0: ?i64 = null,
    /// The exit test compares the first array's pointer with v0 + n*e.
    ptr_test: bool = false,

    fn want(cs: GenCase, target: opt.VectorTarget) u32 {
        const vt = vecType(cs.elem, @intCast(width / cs.elem.size())).?;
        return @intFromBool(cs.ok and target.supports(vt) and (!cs.uses_mul or target.mul_v2i64));
    }

    /// Iterations the loop runs for `n` and start `start` (v4); null when more than 48 (or endless).
    fn trips(cs: GenCase, n: i64, start: i64) ?u32 {
        const base: i64 = 1 << 20; // pointers compare like small offsets from any real address
        var q: i64 = if (cs.ptr_test) base else cs.i0 orelse start;
        const lim = if (cs.ptr_test) base + n * cs.elem.size() else n;
        var k: u32 = 0;
        if (cs.shape == .top) {
            while (ir.cond(cs.cont, q +% cs.toff, lim)) : (q +%= cs.step) {
                k += 1;
                if (k > 48) return null;
            }
            return k;
        }
        if (cs.guarded and !ir.cond(cs.cont, q +% cs.gtoff, lim)) return 0;
        while (true) {
            k += 1;
            if (k > 48) return null;
            q +%= cs.step;
            if (!ir.cond(cs.cont, q +% cs.toff, lim)) return k;
        }
    }

    fn fill(cs: GenCase) Fill {
        return switch (cs.elem) {
            .i64 => .i64,
            .f64 => .f64,
            else => .f32,
        };
    }
};

/// Writer of one random loop (see `genLoop`).
const Gen = struct {
    r: std.Random,
    gpa: std.mem.Allocator,
    out: *std.ArrayList(u8),
    next: VReg = 27,

    fn fresh(g: *Gen) VReg {
        g.next += 1;
        return g.next - 1;
    }
    fn line(g: *Gen, comptime fmt: []const u8, args: anytype) !void {
        try g.out.print(g.gpa, "  " ++ fmt ++ "\n", args);
    }
    fn pick(g: *Gen, xs: []const VReg) VReg {
        return xs[g.r.uintLessThan(usize, xs.len)];
    }
    /// `cbr` continuing to `in` iff `cont(x, n)`, written in one of four equivalent ways.
    fn testBr(g: *Gen, x: VReg, n: VReg, cont: ir.CondCode, in: []const u8, ex: []const u8) !void {
        switch (g.r.uintLessThan(u8, 4)) {
            0 => try g.line("cbr {t} v{d} v{d} {s} {s}", .{ cont, x, n, in, ex }),
            1 => try g.line("cbr {t} v{d} v{d} {s} {s}", .{ mirror(cont), n, x, in, ex }),
            2 => try g.line("cbr {t} v{d} v{d} {s} {s}", .{ negate(cont), x, n, ex, in }),
            else => try g.line("cbr {t} v{d} v{d} {s} {s}", .{ negate(mirror(cont)), n, x, ex, in }),
        }
    }
};

const GenAccess = struct { array: u8, disp: i64, store: bool };

/// Writes a random loop over arrays v0..v2 (pointer arguments) with n = v3, start v4 and an
/// invariant v5, and returns what it is. Vregs: v6 counter, v7 reduction, v8..v10 array
/// pointers, v11 end pointer, v12..v16 constants, v17 entry invariant, v18 loop invariant,
/// v19/v20 offsets, v21..v23 addresses, v24 test value, v25 guard value, v26 result, v27+
/// temporaries. Rejected features are mixed in now and then (`GenCase.ok`).
fn genLoop(r: std.Random, gpa: std.mem.Allocator, out: *std.ArrayList(u8)) !GenCase {
    var g: Gen = .{ .r = r, .gpa = gpa, .out = out };
    const elem: ir.Type = switch (r.uintLessThan(u8, 3)) {
        0 => .i64,
        1 => .f64,
        else => .f32,
    };
    const e: i64 = elem.size();
    const sfx: []const u8 = switch (elem) {
        .i64 => "",
        .f64 => ".f64",
        else => ".f32",
    };
    const style = r.uintLessThan(u8, 3); // 0: index * (e / step), 1: byte offset, 2: pointers
    var cs: GenCase = .{
        .elem = elem,
        .shape = if (r.boolean()) .top else .bottom,
        .guarded = r.uintLessThan(u8, 4) != 0,
        .step = switch (style) {
            0 => @as(i64, 1) << r.uintAtMost(u6, if (e == 8) 3 else 2),
            1 => e,
            else => @as(i64, 1) << r.uintAtMost(u6, 3),
        },
        .cont = if (r.uintLessThan(u8, 8) == 0)
            r.enumValue(ir.CondCode)
        else
            ([_]ir.CondCode{ .lt, .ult, .le, .ule, .ne })[r.uintLessThan(u8, 5)],
    };
    cs.ptr_test = style == 2 and r.boolean();
    if (cs.ptr_test) cs.step = e;
    switch (cs.cont) {
        .lt, .ult, .le, .ule, .ne => {},
        else => cs.ok = false,
    }
    const offs = [_]i64{ 0, 0, cs.step, -cs.step, 1 };
    cs.toff = offs[r.uintLessThan(usize, offs.len)];
    cs.gtoff = offs[r.uintLessThan(usize, offs.len)];
    if (!cs.ptr_test and r.uintLessThan(u8, 4) == 0) cs.i0 = if (r.boolean()) 0 else 3;
    const mult = @divExact(e, if (style == 0) cs.step else e); // bytes per counter unit
    const red_i64 = elem == .i64 and r.boolean();
    const red_float = elem != .i64 and r.uintLessThan(u8, 6) == 0;
    if (red_float) cs.ok = false;
    const narrays = r.intRangeAtMost(u8, 1, 3);
    const unguarded_bottom = cs.shape == .bottom and !cs.guarded;

    // Entry.
    try out.print(gpa, "func g 6\ne:\n", .{});
    try g.line("v12 = const {d}", .{e});
    try g.line("v13 = const {d}", .{cs.step});
    try g.line("v14 = const {d}", .{mult});
    try g.line("v15 = const {d}", .{cs.toff});
    try g.line("v16 = const {d}", .{cs.gtoff});
    const shift = g.fresh();
    try g.line("v{d} = const {d}", .{ shift, std.math.log2_int(u64, @intCast(mult)) });
    if (elem == .i64) {
        try g.line("v17 = const {d}", .{r.intRangeAtMost(i64, -9, 9)});
    } else try g.line("v17 = fconst{s} {d}", .{ sfx, @as(f64, @floatFromInt(r.intRangeAtMost(i32, -9, 9))) / 4 });
    if (!cs.ptr_test) {
        if (cs.i0) |k| try g.line("v6 = const {d}", .{k}) else try g.line("v6 = copy v4", .{});
    }
    if (red_i64) try g.line("v7 = {s}", .{if (r.boolean()) "copy v5" else "const 3"});
    if (red_float) try g.line("v7 = fconst{s} 0.5", .{sfx});
    if (style == 2) for (0..narrays) |k| try g.line("v{d} = copy v{d}", .{ 8 + k, k });
    if (cs.ptr_test) try g.line("v11 = mul v3 v12\n  v11 = add v0 v11", .{});
    const q: VReg = if (cs.ptr_test) 8 else 6;
    const n: VReg = if (cs.ptr_test) 11 else 3;
    if (cs.shape == .top) {
        try out.print(gpa, "  br head\nhead:\n", .{});
        var x = q;
        if (cs.toff != 0) {
            try g.line("v24 = add v{d} v15", .{q});
            x = 24;
        }
        try g.testBr(x, n, cs.cont, "body", "done");
        try out.print(gpa, "body:\n", .{});
    } else if (cs.guarded) {
        try g.line("v25 = add v{d} v16", .{q});
        try g.testBr(25, n, cs.cont, "loop", "done");
        try out.print(gpa, "loop:\n", .{});
    } else try out.print(gpa, "  br loop\nloop:\n", .{});

    // Addresses.
    var addr: [3]VReg = .{ 8, 9, 10 };
    var off: VReg = 6;
    var doubled: ?u8 = null;
    if (style != 2) {
        if (cs.i0 == null) {
            try g.line("v19 = sub v6 v4", .{});
            off = 19;
        }
        if (mult != 1) {
            if (r.boolean()) try g.line("v20 = mul v{d} v14", .{off}) else try g.line("v20 = shl v{d} v{d}", .{ off, shift });
            off = 20;
        }
        for (0..narrays) |k| {
            addr[k] = @intCast(21 + k);
            try g.line("v{d} = add v{d} v{d}", .{ 21 + k, k, off });
        }
        if (r.uintLessThan(u8, 12) == 0) { // the last array with twice the stride
            const t2 = g.fresh();
            try g.line("v{d} = add v{d} v{d}\n  v{d} = add v{d} v{d}", .{ t2, off, off, addr[narrays - 1], narrays - 1, t2 });
            doubled = narrays - 1;
        }
    }

    // Invariants for data operands.
    var inv_buf: [4]VReg = undefined;
    var inv: std.ArrayList(VReg) = .initBuffer(&inv_buf);
    inv.appendAssumeCapacity(17);
    if (elem == .i64) inv.appendAssumeCapacity(5);
    if (r.boolean()) {
        const k = g.fresh();
        if (elem == .i64) try g.line("v{d} = const {d}", .{ k, r.intRangeAtMost(i64, -5, 5) }) else try g.line("v{d} = fconst{s} -1.5", .{ k, sfx });
        inv.appendAssumeCapacity(k);
    }
    const has18 = r.boolean();
    if (has18) {
        if (elem == .i64) try g.line("v18 = mul v5 v17", .{}) else try g.line("v18 = cvt{s}.i64 v5", .{sfx});
        inv.appendAssumeCapacity(18);
    }

    // Loads, lane-wise ops, stores.
    var acc_buf: [8]GenAccess = undefined;
    var acc: std.ArrayList(GenAccess) = .initBuffer(&acc_buf);
    var pool_buf: [8]VReg = undefined;
    var pool: std.ArrayList(VReg) = .initBuffer(&pool_buf);
    const disps = [_]i64{ 0, 0, 0, e, -e, 2 * e, 16, -16, 3 * e, 32 };
    var nl = r.uintAtMost(u8, 3);
    const ns = r.uintAtMost(u8, 2);
    if (nl + ns == 0) nl = 1;
    for (0..nl) |_| {
        const a: GenAccess = .{ .array = r.uintLessThan(u8, narrays), .disp = disps[r.uintLessThan(usize, disps.len)], .store = false };
        const v = g.fresh();
        try g.line("v{d} = load{s} v{d} {d}", .{ v, sfx, addr[a.array], a.disp });
        acc.appendAssumeCapacity(a);
        pool.appendAssumeCapacity(v);
    }
    if (pool.items.len != 0) for (0..r.uintAtMost(u8, 3)) |_| {
        const x = g.pick(pool.items);
        const y = if (r.boolean()) g.pick(pool.items) else g.pick(inv.items);
        const xy: [2]VReg = if (r.boolean()) .{ x, y } else .{ y, x };
        const v = g.fresh();
        if (elem == .i64) {
            const op = ([_][]const u8{ "add", "sub", "and", "or", "xor", "mul" })[r.uintLessThan(u8, 6)];
            if (op[0] == 'm') cs.uses_mul = true;
            try g.line("v{d} = {s} v{d} v{d}", .{ v, op, xy[0], xy[1] });
        } else {
            const op = ([_][]const u8{ "add", "sub", "mul", "div", "min", "max" })[r.uintLessThan(u8, 6)];
            try g.line("v{d} = {s}{s} v{d} v{d}", .{ v, op, sfx, xy[0], xy[1] });
        }
        pool.appendAssumeCapacity(v);
    };
    // Rejected features.
    if (r.uintLessThan(u8, 16) == 0) {
        const v = g.fresh();
        try g.line("v{d} = call {d} v5 v5", .{ v, @intFromPtr(&testgen.callee) });
        cs.ok = false;
    }
    if (pool.items.len != 0 and r.uintLessThan(u8, 10) == 0) {
        const v = g.fresh();
        const k = g.fresh();
        const x = g.pick(pool.items);
        if (elem == .i64) {
            try g.line("v{d} = const 7", .{k});
            try g.line("v{d} = {s} v{d} v{d}", .{ v, if (r.boolean()) "sdiv" else "shl", x, k });
        } else switch (r.uintLessThan(u8, 3)) {
            0 => try g.line("v{d} = fcmp{s} lt v{d} v{d}", .{ v, sfx, x, g.pick(pool.items) }),
            1 => try g.line("v{d} = cvt.i64{s} v{d}", .{ v, sfx, x }),
            else => try g.line("v{d} = load{s} v{d} 0", .{ v, if (elem == .f64) ".f32" else "", addr[0] }),
        }
        cs.ok = false;
    }
    for (0..ns) |_| {
        const a: GenAccess = .{ .array = r.uintLessThan(u8, narrays), .disp = disps[r.uintLessThan(usize, disps.len)], .store = true };
        const src = if (pool.items.len != 0 and r.uintLessThan(u8, 4) != 0) g.pick(pool.items) else g.pick(inv.items);
        try g.line("store{s} v{d} {d} v{d}", .{ sfx, addr[a.array], a.disp, src });
        acc.appendAssumeCapacity(a);
    }
    if (elem == .i64 and style != 2 and r.uintLessThan(u8, 16) == 0) {
        try g.line("store v{d} 0 v{d}", .{ addr[0], off }); // an affine value as data
        cs.ok = false;
    }
    if (red_i64) {
        const op = ([_][]const u8{ "add", "sub", "and", "or", "xor", "mul" })[r.uintLessThan(u8, 6)];
        const x = if (pool.items.len != 0 and r.uintLessThan(u8, 4) != 0) g.pick(pool.items) else 5;
        if (op[0] == 'm') cs.uses_mul = true;
        if (r.uintLessThan(u8, 4) == 0) {
            try g.line("v7 = {s} v{d} v7", .{ op, x });
            if (op[0] == 's') cs.ok = false; // x - r is no reduction
        } else try g.line("v7 = {s} v7 v{d}", .{ op, x });
    }
    if (red_float) try g.line("v7 = add{s} v7 v{d}", .{ sfx, if (pool.items.len != 0) g.pick(pool.items) else 17 });

    // Induction variables and the exit test.
    if (!cs.ptr_test) {
        if (r.uintLessThan(u8, 4) == 0) {
            const k = g.fresh();
            try g.line("v{d} = const {d}\n  v6 = sub v6 v{d}", .{ k, -cs.step, k });
        } else try g.line("v6 = add v6 v13", .{});
    }
    if (style == 2) for (0..narrays) |k| try g.line("v{d} = add v{d} v12", .{ 8 + k, 8 + k });
    if (cs.shape == .top) {
        try g.line("br head", .{});
    } else {
        var x = q;
        if (cs.toff != 0) {
            try g.line("v24 = add v{d} v15", .{q});
            x = 24;
        }
        try g.testBr(x, n, cs.cont, "loop", "done");
    }

    // Results.
    try out.print(gpa, "done:\n", .{});
    const count_res: VReg = if (cs.ptr_test) blk: {
        try g.line("v26 = sub v8 v0", .{});
        break :blk 26;
    } else 6;
    switch (r.uintLessThan(u8, 6)) {
        0 => try g.line("ret v{d}", .{count_res}),
        1 => if (red_i64) {
            try g.line("v26 = add v7 v{d}\n  ret v26", .{count_res});
        } else if (red_float) {
            try g.line("v26 = cvt.i64{s} v7\n  ret v26", .{sfx});
        } else try g.line("ret v5", .{}),
        2 => if (unguarded_bottom and has18) {
            if (elem == .i64) try g.line("ret v18", .{}) else try g.line("v26 = cvt.i64{s} v18\n  ret v26", .{sfx});
        } else try g.line("ret v{d}", .{count_res}),
        3 => if (unguarded_bottom and style != 2) {
            try g.line("ret v{d}", .{off}); // an affine value of the last iteration
        } else try g.line("ret v5", .{}),
        4 => if (unguarded_bottom and elem == .i64 and pool.items.len != 0) {
            try g.line("ret v{d}", .{pool.items[pool.items.len - 1]}); // data: rejected
            cs.ok = false;
        } else try g.line("ret v{d}", .{count_res}),
        else => if (red_i64) try g.line("ret v7", .{}) else try g.line("ret v{d}", .{count_res}),
    }

    // Dependences: same array closer than one vector apart (static), or too many checks.
    var checks: u32 = 0;
    for (acc.items) |x| if (x.array == doubled) {
        cs.ok = false;
    };
    for (acc.items, 0..) |x, i| for (acc.items[i + 1 ..]) |y| {
        if (!x.store and !y.store) continue;
        if (x.array != y.array) {
            checks += 1;
        } else if (x.disp != y.disp and @abs(x.disp - y.disp) < width) cs.ok = false;
    };
    if (checks > max_checks) cs.ok = false;
    return cs;
}

/// Up to 8 argument sets on which the loop runs at most 48 iterations (so it stays in the
/// buffer): arrays near one another or apart, counters near 0 and the ends of the range.
fn genArgs(r: std.Random, cs: GenCase, sets: *[8][6]i64) usize {
    const e = cs.elem.size();
    const starts = [_]i64{ 0, 1, -1, 7, -100, -20, std.math.maxInt(i64) - 30, std.math.maxInt(i64) - 3, std.math.maxInt(i64), std.math.minInt(i64), std.math.minInt(i64) + 5 };
    const ends = [_]i64{ 0, -1, std.math.maxInt(i64), std.math.minInt(i64) };
    var count: usize = 0;
    var tries: usize = 0;
    while (count < sets.len and tries < 40) : (tries += 1) {
        var s: [6]i64 = undefined;
        s[0] = 1024 + e * r.intRangeAtMost(i64, -2, 2);
        for (1..3) |k| s[k] = if (r.uintLessThan(u8, 3) == 0) 1024 * @as(i64, @intCast(k + 1)) else s[0] + e * r.intRangeAtMost(i64, -5, 5);
        const delta = r.intRangeAtMost(i64, -3, 44);
        if (cs.ptr_test) {
            s[3] = delta;
            s[4] = 0;
        } else {
            s[4] = starts[r.uintLessThan(usize, starts.len)];
            s[3] = if (r.uintLessThan(u8, 6) == 0) ends[r.uintLessThan(usize, ends.len)] else s[4] +% delta *% (if (r.boolean()) cs.step else 1);
        }
        s[5] = r.intRangeAtMost(i64, -50, 50);
        if (cs.trips(s[3], s[4]) == null) continue;
        sets[count] = s;
        count += 1;
    }
    return count;
}

test "vectorize: random array loops match the interpreter" {
    // Leak-checked like `t.allocator`, without its per-allocation stack traces (6x faster).
    var da: std.heap.DebugAllocator(.{ .stack_trace_frames = 0 }) = .init;
    defer t.expect(da.deinit() == .ok) catch @panic("vectorize: leak");
    const gpa = da.allocator();
    var src: std.ArrayList(u8) = .empty;
    defer src.deinit(gpa);
    const targets = [_]opt.VectorTarget{
        opt.VectorTarget.aarch64,
        .{ .v2i64 = true, .v4f32 = true, .v2f64 = true, .mul_v2i64 = true },
        .{ .v2i64 = true, .v2f64 = true },
    };
    var vectorized: u32 = 0;
    var total: u32 = 0;
    for ([_]u64{ 1, 2, 3, 0x5eed }) |seed| {
        var prng = std.Random.DefaultPrng.init(seed);
        const r = prng.random();
        for (0..80) |i| {
            src.clearRetainingCapacity();
            const cs = try genLoop(r, gpa, &src);
            var sets: [8][6]i64 = undefined;
            const ns = genArgs(r, cs, &sets);
            var sl: [8][]const i64 = undefined;
            for (sl[0..ns], sets[0..ns]) |*o, *s| o.* = s;
            const target = targets[i % targets.len];
            const want = cs.want(target);
            const printed = checkVecFull(gpa, src.items, target, 7, sl[0..ns], cs.fill(), want, i % 2 == 0) catch |err| {
                std.debug.print("vectorize: random loop seed {d} #{d} (want {d}) failed:\n{s}\n", .{ seed, i, want, src.items });
                return err;
            };
            gpa.free(printed);
            vectorized += want;
            total += 1;
        }
    }
    try t.expect(vectorized * 3 >= total);
}
