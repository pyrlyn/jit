//! IR optimizer (`pyrlyn_jit.opt`): runtime inlining, SSA-based scalar optimizations, a loop
//! vectorizer and profile-guided block layout over `ir.Func` (docs/OPTIMIZER.md).
//!
//! `optimize(gpa, f, options)` rewrites `f` in place into an equivalent function that any IR
//! Compiler accepts. Equivalent means: the same result, the same memory effects and the same
//! calls in the same order, for every input on which `f` is defined. `f` is undefined where it
//! executes `sdiv`/`srem` with a zero divisor or `minInt / -1`, or a load/store that faults;
//! the optimizer never introduces such operations on paths that did not execute them (no
//! speculation of `sdiv`/`srem`/`load`/`store`/`call`), but may delete them when their result
//! is unused (`sdiv`/`srem`/`load` only; stores and calls are never deleted or reordered with
//! each other). The reference interpreter `ir.interpret` is the oracle: every pass is tested
//! differentially against it.
//!
//! Pipeline (`Level.o2`): verify/typeCheck -> inline (if `resolver`) -> SSA construction ->
//! scalar rounds (sccp, instcombine, gvn, dce, simplifycfg) -> licm -> vectorize (if
//! `vector`; top-tested loops) -> loop rotation (+ licm again) -> scalar cleanup -> constants
//! to the entry block (remat) -> out of SSA (copy coalescing, compact vregs, no block
//! parameters) -> block layout.
//! `Level.o1` skips inlining, loop passes and vectorization; `Level.o0` only validates.
//!
//! Between `ssa.build` and `out_of_ssa.run` the function is in strict SSA form: function
//! parameters `v0..v(params-1)`, block parameters and op results are the only definitions,
//! each vreg has exactly one, every use is dominated by its definition, and values that were
//! read before being written (which read 0) are explicit zero constants.
const std = @import("std");
const builtin = @import("builtin");
const ir = @import("../ir.zig");

pub const cfg = @import("cfg.zig");
pub const testgen = @import("testgen.zig");
pub const ssa = @import("ssa.zig");
pub const out_of_ssa = @import("out_of_ssa.zig");
pub const dce = @import("dce.zig");
pub const sccp = @import("sccp.zig");
pub const instcombine = @import("instcombine.zig");
pub const gvn = @import("gvn.zig");
pub const simplifycfg = @import("simplifycfg.zig");
pub const licm = @import("licm.zig");
pub const loops = @import("loops.zig");
pub const remat = @import("remat.zig");
pub const layout = @import("layout.zig");
pub const inline_ = @import("inline.zig");
pub const profile = @import("profile.zig");
pub const vectorize = @import("vectorize.zig");

pub const Level = enum { o0, o1, o2 };

/// Runtime inlining: maps a `call` target address to the IR of the function at that address.
/// The IR must compute exactly what the machine code at `addr` computes for every input (C
/// calling convention, i64 arguments and result, same memory effects and calls). Returning
/// `null` keeps the call. The returned function is only read, during `optimize`.
pub const Resolver = struct {
    ctx: ?*anyopaque = null,
    lookup: *const fn (ctx: ?*anyopaque, addr: usize) ?*const ir.Func,
};

/// Execution counts collected by instrumented code (`profile.instrument`), indexed by the
/// block index of the function that was instrumented (the one passed to `optimize`).
pub const Profile = struct {
    /// Times each block started executing.
    blocks: []const u64 = &.{},
    /// For a block ending in `cbr`/`cbra`: times its `then` edge was taken (else unused).
    taken: []const u64 = &.{},
};

/// 128-bit vector support of the code generator that will compile the result. The loop
/// vectorizer only emits vector types (and `mul` on 64-bit lanes) the target lowers.
pub const VectorTarget = struct {
    v2i64: bool = false,
    v4i32: bool = false,
    v4f32: bool = false,
    v2f64: bool = false,
    /// `mul.v2i64` is a single native instruction (otherwise it is not worth vectorizing).
    mul_v2i64: bool = false,

    pub const none: VectorTarget = .{};
    /// AArch64 NEON and x86-64 SSE2/SSE4.1 lowering in this library's IR Compilers.
    pub const aarch64: VectorTarget = .{ .v2i64 = true, .v4i32 = true, .v4f32 = true, .v2f64 = true };
    pub const x86_64: VectorTarget = .{ .v2i64 = true, .v4i32 = true, .v4f32 = true, .v2f64 = true };
    pub const host: VectorTarget = switch (builtin.cpu.arch) {
        .aarch64 => aarch64,
        .x86_64 => x86_64,
        else => none,
    };

    pub fn enabled(v: VectorTarget) bool {
        return v.v2i64 or v.v4i32 or v.v4f32 or v.v2f64;
    }
    pub fn supports(v: VectorTarget, ty: ir.Type) bool {
        return switch (ty) {
            .v2i64 => v.v2i64,
            .v4i32 => v.v4i32,
            .v4f32 => v.v4f32,
            .v2f64 => v.v2f64,
            else => false,
        };
    }
};

pub const InlineOptions = struct {
    /// Largest callee (ops) inlined at a call site without profile data (or a cold one).
    max_callee_ops: u32 = 40,
    /// Largest callee inlined at a hot call site (`hot_count` executions or more).
    max_hot_callee_ops: u32 = 200,
    hot_count: u64 = 64,
    /// Stop inlining when the caller reaches this many ops.
    max_caller_ops: u32 = 4000,
    /// Nesting limit (a recursive function is unrolled at most this many levels).
    max_depth: u8 = 3,
};

pub const Options = struct {
    level: Level = .o2,
    resolver: ?Resolver = null,
    profile: ?*const Profile = null,
    vector: VectorTarget = .none,
    inlining: InlineOptions = .{},
    /// Upper bound on scalar optimization rounds (each runs every scalar pass once).
    max_rounds: u8 = 4,
    /// Re-verify the IR (structure, types, SSA) after every pass; on in tests and Debug.
    verify_each: bool = builtin.mode == .debug,
};

pub const Stats = struct {
    ops_before: u32 = 0,
    ops_after: u32 = 0,
    blocks_after: u32 = 0,
    inlined_calls: u32 = 0,
    vectorized_loops: u32 = 0,
    hoisted: u32 = 0,
    rounds: u32 = 0,
};

pub const Error = std.mem.Allocator.Error || ir.Func.VerifyError || ir.Func.TypeError || error{InvalidSsa};

/// Shared state of one `optimize` run. Passes edit `f` directly (ops, block params, edges)
/// but add, delete and reorder blocks only through `addBlock`, `compactBlocks` and
/// `reorderBlocks`, and create vregs only through `newVReg`, so `types`, `freq` and `taken`
/// stay parallel to `f`'s vregs and blocks. IR slices (edge args, params, call args) must be
/// allocated with `c.irAlloc()` (the function's arena); scratch data with `c.gpa`.
pub const Ctx = struct {
    gpa: std.mem.Allocator,
    f: *ir.Func,
    options: *const Options,
    stats: *Stats,
    /// Type of every vreg (`types.items.len == f.num_vregs`).
    types: std.ArrayList(ir.Type) = .empty,
    /// Execution-frequency hint per block (`null`: unknown), from `Options.profile`.
    freq: std.ArrayList(?u64) = .empty,
    /// Times the `then` edge of the block's `cbr`/`cbra` was taken (`null`: unknown or not a
    /// conditional branch), from `Options.profile`. A pass that moves a conditional terminator
    /// to another block, swaps its edges or replaces it updates the entry (see `edgeFreq`).
    taken: std.ArrayList(?u64) = .empty,
    /// True between `ssa.build` and `out_of_ssa.run`.
    in_ssa: bool = false,
    /// Every block name used so far (new blocks get unique names, so printed IR parses).
    names: std.StringHashMapUnmanaged(void) = .empty,
    name_seq: u32 = 0,

    pub fn init(gpa: std.mem.Allocator, f: *ir.Func, options: *const Options, stats: *Stats) Error!Ctx {
        var c: Ctx = .{ .gpa = gpa, .f = f, .options = options, .stats = stats };
        errdefer c.deinit();
        const ty = try f.typeCheck(gpa);
        defer gpa.free(ty);
        try c.types.appendSlice(gpa, ty[0..f.num_vregs]);
        for (f.blocks.items) |b| try c.names.put(gpa, b.name, {});
        try c.freq.resize(gpa, f.blocks.items.len);
        @memset(c.freq.items, null);
        try c.taken.resize(gpa, f.blocks.items.len);
        @memset(c.taken.items, null);
        if (options.profile) |p| for (c.freq.items, c.taken.items, f.blocks.items, 0..) |*q, *tk, b, i| {
            if (i < p.blocks.len) q.* = p.blocks[i];
            const term = b.ops.items[b.ops.items.len - 1];
            if (i < p.taken.len and (term == .cbr or term == .cbra)) tk.* = p.taken[i];
        };
        return c;
    }
    pub fn deinit(c: *Ctx) void {
        c.types.deinit(c.gpa);
        c.freq.deinit(c.gpa);
        c.taken.deinit(c.gpa);
        c.names.deinit(c.gpa);
    }

    /// Allocator for slices stored in the IR (lives as long as the function).
    pub fn irAlloc(c: *Ctx) std.mem.Allocator {
        return c.f.alloc();
    }

    pub fn newVReg(c: *Ctx, ty: ir.Type) Error!ir.VReg {
        if (c.f.num_vregs >= std.math.maxInt(ir.VReg) - 1) return error.BadVReg;
        try c.types.append(c.gpa, ty);
        return c.f.newVReg();
    }

    /// Appends to `ops` (a list allocated with `a`) ops that set a new vreg to the zero of
    /// `ty` (what a never-written vreg reads: 0, +0.0, all-zero lanes) and returns the vreg.
    pub fn zeroValue(c: *Ctx, a: std.mem.Allocator, ops: *std.ArrayList(ir.Op), ty: ir.Type) Error!ir.VReg {
        const v = try c.newVReg(ty);
        switch (ty) {
            .i64 => try ops.append(a, .{ .@"const" = .{ .dst = v, .imm = 0 } }),
            .f32, .f64 => try ops.append(a, .{ .fconst = .{ .dst = v, .ty = ty, .bits = 0 } }),
            else => {
                const s = try c.zeroValue(a, ops, ty.lane());
                try ops.append(a, .{ .splat = .{ .ty = ty, .dst = v, .src = s } });
            },
        }
        return v;
    }

    /// Appends a block named `<base>.<n>` (unique) with frequency hint `freq` (and an unknown
    /// `taken` count).
    pub fn addBlock(c: *Ctx, base: []const u8, freq: ?u64) Error!u32 {
        const name = while (true) {
            c.name_seq += 1;
            const n = try std.fmt.allocPrint(c.irAlloc(), "{s}.{d}", .{ base, c.name_seq });
            const gop = try c.names.getOrPut(c.gpa, n);
            if (!gop.found_existing) break n;
        };
        try c.freq.ensureUnusedCapacity(c.gpa, 1);
        try c.taken.ensureUnusedCapacity(c.gpa, 1);
        try c.f.blocks.append(c.irAlloc(), .{ .name = name });
        c.freq.appendAssumeCapacity(freq);
        c.taken.appendAssumeCapacity(null);
        return @intCast(c.f.blocks.items.len - 1);
    }

    /// Deletes the blocks with `keep[b] == false` (block 0 must be kept; no kept block may
    /// branch to a deleted one) and renumbers every branch target.
    pub fn compactBlocks(c: *Ctx, keep: []const bool) Error!void {
        const blocks = c.f.blocks.items;
        std.debug.assert(keep.len == blocks.len and keep[0]);
        const map = try c.gpa.alloc(u32, blocks.len);
        defer c.gpa.free(map);
        var n: u32 = 0;
        for (keep, 0..) |k, i| {
            map[i] = n;
            if (k) {
                blocks[n] = blocks[i];
                c.freq.items[n] = c.freq.items[i];
                c.taken.items[n] = c.taken.items[i];
                n += 1;
            }
        }
        c.f.blocks.items.len = n;
        c.freq.items.len = n;
        c.taken.items.len = n;
        for (c.f.blocks.items) |*b| retarget(b, map);
    }

    /// Puts the blocks in `order` (a permutation of all block indices with `order[0] == 0`)
    /// and renumbers every branch target.
    pub fn reorderBlocks(c: *Ctx, order: []const u32) Error!void {
        const blocks = c.f.blocks.items;
        std.debug.assert(order.len == blocks.len and order[0] == 0);
        const map = try c.gpa.alloc(u32, blocks.len);
        defer c.gpa.free(map);
        const old = try c.gpa.dupe(ir.Block, blocks);
        defer c.gpa.free(old);
        const old_freq = try c.gpa.alloc(?u64, 2 * blocks.len);
        defer c.gpa.free(old_freq);
        @memcpy(old_freq[0..blocks.len], c.freq.items);
        @memcpy(old_freq[blocks.len..], c.taken.items);
        for (order, 0..) |o, i| {
            map[o] = @intCast(i);
            blocks[i] = old[o];
            c.freq.items[i] = old_freq[o];
            c.taken.items[i] = old_freq[blocks.len + o];
        }
        for (c.f.blocks.items) |*b| retarget(b, map);
    }

    fn retarget(b: *ir.Block, map: []const u32) void {
        const t = &b.ops.items[b.ops.items.len - 1];
        switch (t.*) {
            .br => |*x| x.* = map[x.*],
            .cbr => |*x| {
                x.then = map[x.then];
                x.@"else" = map[x.@"else"];
            },
            .bra => |*x| x.target = map[x.target],
            .cbra => |*x| {
                x.then = map[x.then];
                x.@"else" = map[x.@"else"];
            },
            else => {},
        }
    }

    /// Estimated executions of edge `index` (0: `then` or the only successor, 1: `else`) of
    /// block `from`: `freq` for `br`/`bra`; `taken` and `freq - taken` for `cbr`/`cbra`
    /// (counts are racy, so `taken` is clamped to `freq`); `null` when unknown or for `ret`.
    pub fn edgeFreq(c: *const Ctx, from: u32, index: u1) ?u64 {
        const fr = c.freq.items[from] orelse return null;
        const ops = c.f.blocks.items[from].ops.items;
        return switch (ops[ops.len - 1]) {
            .br, .bra => fr,
            .cbr, .cbra => {
                const tk = @min(c.taken.items[from] orelse return null, fr);
                return if (index == 0) tk else fr - tk;
            },
            else => null,
        };
    }

    /// Structural, type and (in SSA) single-definition checks after `pass`, when enabled.
    pub fn check(c: *Ctx, pass: []const u8) Error!void {
        if (!c.options.verify_each) return;
        errdefer std.log.err("pyrlyn_jit.opt: IR invalid after {s}", .{pass});
        try c.f.verify();
        const nb = c.f.blocks.items.len;
        if (c.types.items.len != c.f.num_vregs or c.freq.items.len != nb or c.taken.items.len != nb) return error.InvalidSsa;
        // In SSA, blocks unreachable from the entry (left for simplifycfg) may read parameters
        // whose incoming arguments are all gone, which leaves them untyped: type-check the
        // reachable blocks only (the others are hidden for the duration of `typeCheck`; nothing
        // reachable uses their definitions). Out of SSA a vreg's type may come from any block.
        const reached = try c.gpa.alloc(bool, nb);
        defer c.gpa.free(reached);
        @memset(reached, !c.in_ssa);
        var stack: std.ArrayList(u32) = .empty;
        defer stack.deinit(c.gpa);
        reached[0] = true;
        try stack.append(c.gpa, 0);
        while (stack.pop()) |b| {
            const ops = c.f.blocks.items[b].ops.items;
            const term = ops[ops.len - 1];
            for (0..ssa.numSuccs(term)) |i| {
                const to = cfg.edgeOf(term, @intCast(i)).to;
                if (reached[to]) continue;
                reached[to] = true;
                try stack.append(c.gpa, to);
            }
        }
        const hidden = try c.gpa.alloc(struct { ops: usize, params: []const ir.VReg }, nb);
        defer c.gpa.free(hidden);
        for (c.f.blocks.items, reached, hidden) |*b, r, *h| {
            h.* = .{ .ops = b.ops.items.len, .params = b.params };
            if (r) continue;
            b.ops.items.len = 0;
            b.params = &.{};
        }
        const typed = c.f.typeCheck(c.gpa);
        for (c.f.blocks.items, hidden) |*b, h| {
            b.ops.items.len = h.ops;
            b.params = h.params;
        }
        const ty = try typed;
        defer c.gpa.free(ty);
        // `types` must agree with the inferred type of every defined vreg.
        for (0..c.f.params) |v| if (c.types.items[v] != .i64) return error.TypeMismatch;
        for (c.f.blocks.items, reached) |b, r| {
            if (!r) continue;
            for (b.params) |p| if (c.types.items[p] != ty[p]) return error.TypeMismatch;
            for (b.ops.items) |o| if (o.def()) |d| if (c.types.items[d] != ty[d]) return error.TypeMismatch;
        }
        if (c.in_ssa) try ssa.verify(c);
    }
};

pub fn countOps(f: *const ir.Func) u32 {
    var n: u32 = 0;
    for (f.blocks.items) |b| n += @intCast(b.ops.items.len);
    return n;
}

/// Rewrites every vreg `op` reads (operands, call arguments, edge arguments) `v` to `map[v]`;
/// definitions and block parameters are not touched. Argument slices are copied into `a` (the
/// function arena, `Ctx.irAlloc()`) only when one of their entries changes.
pub fn mapUses(op: *ir.Op, a: std.mem.Allocator, map: []const ir.VReg) std.mem.Allocator.Error!void {
    switch (op.*) {
        .@"const", .fconst, .br => {},
        .copy => |*x| x.src = map[x.src],
        .arith => |*x| {
            x.a = map[x.a];
            x.b = map[x.b];
        },
        .load => |*x| x.base = map[x.base],
        .store => |*x| {
            x.base = map[x.base];
            x.src = map[x.src];
        },
        .typed => |*x| {
            x.a = map[x.a];
            x.b = map[x.b];
        },
        .fcmp => |*x| {
            x.a = map[x.a];
            x.b = map[x.b];
        },
        .cvt => |*x| x.src = map[x.src],
        .splat => |*x| x.src = map[x.src],
        .extract => |*x| x.src = map[x.src],
        .call => |*x| x.args = try mapSlice(a, x.args, map),
        .cbr => |*x| {
            x.a = map[x.a];
            x.b = map[x.b];
        },
        .bra => |*x| x.args = try mapSlice(a, x.args, map),
        .cbra => |*x| {
            x.a = map[x.a];
            x.b = map[x.b];
            x.then_args = try mapSlice(a, x.then_args, map);
            x.else_args = try mapSlice(a, x.else_args, map);
        },
        .ret => |*x| x.* = map[x.*],
    }
}

fn mapSlice(a: std.mem.Allocator, s: []const ir.VReg, map: []const ir.VReg) std.mem.Allocator.Error![]const ir.VReg {
    for (s, 0..) |v, i| if (map[v] != v) {
        const out = try a.dupe(ir.VReg, s);
        for (out[i..]) |*w| w.* = map[w.*];
        return out;
    };
    return s;
}

/// `mapUses` over every op of every block (`map.len == f.num_vregs`).
pub fn renameUses(c: *Ctx, map: []const ir.VReg) Error!void {
    std.debug.assert(map.len >= c.f.num_vregs);
    for (c.f.blocks.items) |*b| for (b.ops.items) |*op| try mapUses(op, c.irAlloc(), map);
}

/// Number of reads of every vreg (operands, call and edge arguments); caller frees.
pub fn useCounts(gpa: std.mem.Allocator, f: *const ir.Func) std.mem.Allocator.Error![]u32 {
    const n = try gpa.alloc(u32, f.num_vregs);
    @memset(n, 0);
    var buf: [ir.max_args]ir.VReg = undefined;
    for (f.blocks.items) |b| for (b.ops.items) |op| switch (op) {
        .bra => |x| for (x.args) |v| {
            n[v] += 1;
        },
        .cbra => |x| {
            n[x.a] += 1;
            n[x.b] += 1;
            for (x.then_args) |v| n[v] += 1;
            for (x.else_args) |v| n[v] += 1;
        },
        else => for (op.uses(&buf)) |v| {
            n[v] += 1;
        },
    };
    return n;
}

/// Optimizes `f` in place (see the module comment). `gpa` is used for scratch memory only;
/// the function keeps its own arena. On error `f` may be partially transformed but is
/// still owned by the caller (call `f.deinit()`); it is not guaranteed to be valid IR.
pub fn optimize(gpa: std.mem.Allocator, f: *ir.Func, options: Options) Error!Stats {
    var stats: Stats = .{ .ops_before = countOps(f) };
    try f.verify();
    var c = try Ctx.init(gpa, f, &options, &stats);
    defer c.deinit();
    if (options.level != .o0) {
        if (options.level == .o2 and options.resolver != null) {
            if (try inline_.run(&c)) try c.check("inline");
        }
        try ssa.build(&c);
        try c.check("ssa.build");
        try scalarRounds(&c);
        if (options.level == .o2) {
            var loop_changed = false;
            if (try licm.run(&c)) {
                try c.check("licm");
                loop_changed = true;
            }
            if (options.vector.enabled()) if (try vectorize.run(&c)) {
                try c.check("vectorize");
                loop_changed = true;
            };
            if (try loops.rotate(&c)) { // bottom-tested bodies now run at least once: hoist more
                try c.check("loops.rotate");
                loop_changed = true;
                if (try licm.run(&c)) try c.check("licm");
            }
            if (loop_changed) try scalarRounds(&c);
        }
        if (try remat.run(&c)) try c.check("remat");
        try out_of_ssa.run(&c);
        try c.check("out_of_ssa");
        // Last, so the edge blocks out_of_ssa adds are placed too (layout only permutes).
        if (try layout.run(&c)) try c.check("layout");
    }
    stats.ops_after = countOps(f);
    stats.blocks_after = @intCast(f.blocks.items.len);
    return stats;
}

fn scalarRounds(c: *Ctx) Error!void {
    var round: u8 = 0;
    while (round < c.options.max_rounds) : (round += 1) {
        c.stats.rounds += 1;
        var changed = false;
        inline for (.{ .{ sccp, "sccp" }, .{ instcombine, "instcombine" }, .{ gvn, "gvn" }, .{ dce, "dce" }, .{ simplifycfg, "simplifycfg" } }) |p| {
            if (try p[0].run(c)) {
                try c.check(p[1]);
                changed = true;
            }
        }
        if (!changed) break;
    }
}

test {
    _ = cfg;
    _ = testgen;
    _ = ssa;
    _ = out_of_ssa;
    _ = dce;
    _ = sccp;
    _ = instcombine;
    _ = gvn;
    _ = simplifycfg;
    _ = licm;
    _ = loops;
    _ = remat;
    _ = layout;
    _ = inline_;
    _ = profile;
    _ = vectorize;
}

test "opt: mapUses/renameUses rewrite operands and edge arguments, not definitions" {
    const t = std.testing;
    var f = try ir.parse(t.allocator,
        \\func m 2
        \\e:
        \\  v2 = add v0 v1
        \\  cbr lt v2 v1 a:v2:v0 b:v1
        \\a(v3 v4):
        \\  v5 = call 0x10 v3 v4
        \\  ret v5
        \\b(v6):
        \\  store v6 8 v0
        \\  ret v6
    );
    defer f.deinit();
    const before = try useCounts(t.allocator, &f);
    defer t.allocator.free(before);
    try t.expectEqual(@as(u32, 3), before[0]); // add, then-edge argument, store value
    var opts: Options = .{};
    var stats: Stats = .{};
    var c = try Ctx.init(t.allocator, &f, &opts, &stats);
    defer c.deinit();
    var map: [7]ir.VReg = .{ 0, 1, 2, 3, 4, 5, 6 };
    map[0] = 1; // every read of v0 now reads v1
    try renameUses(&c, &map);
    const after = try useCounts(t.allocator, &f);
    defer t.allocator.free(after);
    try t.expectEqual(@as(u32, 0), after[0]);
    try t.expectEqual(before[1] + before[0], after[1]);
    try t.expectEqual(@as(ir.VReg, 2), f.blocks.items[0].ops.items[0].arith.dst);
    try t.expectEqual(@as(ir.VReg, 6), f.blocks.items[2].params[0]);
}

test "opt: profile counts follow blocks through addBlock, reorderBlocks and compactBlocks" {
    const t = std.testing;
    var f = try ir.parse(t.allocator,
        \\func p 1
        \\e:
        \\  cbr lt v0 v0 a b
        \\a:
        \\  br b
        \\b:
        \\  ret v0
    );
    defer f.deinit();
    const prof: Profile = .{ .blocks = &.{ 10, 3, 10 }, .taken = &.{ 3, 99, 0 } };
    var opts: Options = .{ .profile = &prof };
    var stats: Stats = .{};
    var c = try Ctx.init(t.allocator, &f, &opts, &stats);
    defer c.deinit();
    try t.expectEqual(@as(?u64, 3), c.edgeFreq(0, 0));
    try t.expectEqual(@as(?u64, 7), c.edgeFreq(0, 1));
    try t.expectEqual(@as(?u64, null), c.taken.items[1]); // `br` block: count ignored
    try t.expectEqual(@as(?u64, 3), c.edgeFreq(1, 0));
    try t.expectEqual(@as(?u64, null), c.edgeFreq(2, 0));
    const x = try c.addBlock("x", 5);
    try f.emit(x, .{ .ret = 0 });
    try t.expectEqual(@as(?u64, null), c.taken.items[x]);
    try c.reorderBlocks(&.{ 0, 2, 3, 1 });
    try t.expectEqual(@as(?u64, 3), c.edgeFreq(0, 0)); // then-edge now enters block 3 (old a)
    try t.expectEqual(@as(u32, 3), f.blocks.items[0].ops.items[0].cbr.then);
    try c.compactBlocks(&.{ true, true, false, true });
    try t.expectEqual(@as(usize, 3), c.taken.items.len);
    try t.expectEqual(@as(?u64, 3), c.freq.items[2]);
    try t.expectEqual(@as(?u64, 3), c.taken.items[0]);
    c.taken.items[0] = 50; // racy counts: taken > freq clamps
    try t.expectEqual(@as(?u64, 10), c.edgeFreq(0, 0));
    try t.expectEqual(@as(?u64, 0), c.edgeFreq(0, 1));
    try c.check("test");
}

test "optimize: o0 validates, o2 preserves fib" {
    const t = std.testing;
    for ([_]Level{ .o0, .o1, .o2 }) |lv| {
        var f = try ir.parse(t.allocator, ir.fib_src);
        defer f.deinit();
        _ = try optimize(t.allocator, &f, .{ .level = lv, .verify_each = true });
        try f.verify();
        try t.expect(!f.hasBlockParams());
        for ([_]i64{ 0, 1, 2, 10, 50 }) |n| {
            var g = try ir.parse(t.allocator, ir.fib_src);
            defer g.deinit();
            try t.expectEqual(try ir.interpret(&g, &.{n}, 1_000_000), try ir.interpret(&f, &.{n}, 1_000_000));
        }
    }
}
