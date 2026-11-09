//! Instruction combining: local algebraic simplification and strength reduction on SSA form.
//!
//! Blocks are visited in reverse postorder (so the definition of every operand of a reachable
//! op was simplified first) and each op is rewritten until no rule applies. A per-vreg table
//! of what defines it (an integer constant, an `arith` op, a `splat`) and approximate use
//! counts drive the rules. Rewritten ops keep their result vreg; ops whose result equals one
//! of their operands (`x + 0`, `copy`, ...) are deleted and their uses renamed (once, through
//! one map). New constants are defined right before the op that needs them, or reused when
//! the block or the constants leading block 0 already hold them. Ops made dead are left to dce
//! and blocks made unreachable to simplifycfg.
//!
//! Integer rules (i64, wrapping):
//!   both operands constant -> folded (`ir.arith`; never `sdiv`/`srem` by 0 or `minInt / -1`)
//!   constant left operand of add/mul/and/or/xor -> moved right
//!   x+0, x-0, x|0, x^0, x&-1, x*1, x<<0, x>>0 (counts mod 64), sdiv x 1 -> x
//!   x*0, x&0, x-x, x^x, srem x 1, 0<<x, 0>>x -> 0;  x|-1 -> -1;  x&x, x|x -> x
//!   x*2^k -> x<<k;  shift counts outside 0..63 -> count mod 64
//!   sub x c -> add x -c (c != minInt);  (x+c1)+c2, (x-c1)+c2, (x*c1)*c2, (x&c1)&c2,
//!   (x|c1)|c2, (x^c1)^c2 -> one op with the folded constant, and (x<<c1)<<c2 (likewise
//!   `shr`, `sar`) -> one shift, when the inner op has a single use
//!   sdiv/srem x 2^k (1 <= k <= 62) -> shifts: t = (x >>s 63) >>u (64-k); q = (x+t) >>s k;
//!   srem: x - ((x+t) & -2^k), exact for every x (divisors 0, -1, minInt are never touched)
//! Branches: `cbr`/`cbra` on equal or constant operands -> `br`/`bra` (`c.taken` unknown);
//! a constant left operand is swapped right with the mirrored condition.
//! Float ops get no identities (x+0.0, x*1.0, x-x change -0.0, NaN or signalling bits).
//! Integer vectors: x&x, x|x -> x; x^x, x-x -> splat 0. `extract` of a `splat` -> its scalar
//! (v4i32 only for a constant, since the lane is truncated and sign-extended).
//! No compare rewrites that could change ordering under overflow (e.g. `x + c < d`).
const std = @import("std");
const ir = @import("../ir.zig");
const opt = @import("opt.zig");
const cfg = @import("cfg.zig");
const ssa = @import("ssa.zig");

/// What defines a vreg, as far as the rules care.
const Def = union(enum) {
    other,
    iconst: i64,
    arith: struct { op: ir.Arith, a: ir.VReg, b: ir.VReg },
    splat: struct { ty: ir.Type, src: ir.VReg },
};

const Result = union(enum) {
    none,
    changed,
    /// The op computes this (existing) vreg: delete it and rename its uses.
    forward: ir.VReg,
};

const max_steps_per_op = 8;

const Combiner = struct {
    c: *opt.Ctx,
    gpa: std.mem.Allocator,
    defs: std.ArrayList(Def) = .empty,
    /// Reads of each vreg; approximate after rewrites (it only steers reassociation).
    nuses: std.ArrayList(u32) = .empty,
    /// Replacement of each vreg (identity unless its defining op was deleted).
    map: std.ArrayList(ir.VReg) = .empty,
    forwarded: bool = false,
    /// Constants defined by the ops leading block 0 (they dominate every block).
    entry_consts: std.AutoHashMapUnmanaged(i64, ir.VReg) = .empty,
    /// Constants defined so far in the block being rewritten.
    local_consts: std.AutoHashMapUnmanaged(i64, ir.VReg) = .empty,
    /// The block being rewritten.
    out: std.ArrayList(ir.Op) = .empty,
    scratch: std.ArrayList(ir.VReg) = .empty,

    fn deinit(k: *Combiner) void {
        const gpa = k.gpa;
        k.defs.deinit(gpa);
        k.nuses.deinit(gpa);
        k.map.deinit(gpa);
        k.entry_consts.deinit(gpa);
        k.local_consts.deinit(gpa);
        k.out.deinit(gpa);
        k.scratch.deinit(gpa);
    }

    fn constOf(k: *const Combiner, v: ir.VReg) ?i64 {
        return switch (k.defs.items[v]) {
            .iconst => |x| x,
            else => null,
        };
    }

    fn newVReg(k: *Combiner, ty: ir.Type) opt.Error!ir.VReg {
        const v = try k.c.newVReg(ty);
        try k.defs.append(k.gpa, .other);
        try k.nuses.append(k.gpa, 0);
        try k.map.append(k.gpa, v);
        return v;
    }

    /// Appends `o` (fully simplified) to the block being rebuilt and records its definition.
    fn emit(k: *Combiner, o: ir.Op) opt.Error!void {
        try k.out.append(k.gpa, o);
        switch (o) {
            .@"const" => |x| {
                k.defs.items[x.dst] = .{ .iconst = x.imm };
                const gop = try k.local_consts.getOrPut(k.gpa, x.imm);
                if (!gop.found_existing) gop.value_ptr.* = x.dst;
            },
            .arith => |x| k.defs.items[x.dst] = .{ .arith = .{ .op = x.op, .a = x.a, .b = x.b } },
            .splat => |x| k.defs.items[x.dst] = .{ .splat = .{ .ty = x.ty, .src = x.src } },
            else => if (o.def()) |d| {
                k.defs.items[d] = .other;
            },
        }
    }

    /// A vreg holding `imm` that dominates the op being rewritten (defined before it if new).
    fn constant(k: *Combiner, imm: i64) opt.Error!ir.VReg {
        if (k.local_consts.get(imm)) |v| return v;
        if (k.entry_consts.get(imm)) |v| return v;
        const v = try k.newVReg(.i64);
        try k.emit(.{ .@"const" = .{ .dst = v, .imm = imm } });
        return v;
    }

    /// Defines a new vreg as `op a b` before the op being rewritten.
    fn arith(k: *Combiner, op: ir.Arith, a: ir.VReg, b: ir.VReg) opt.Error!ir.VReg {
        const v = try k.newVReg(.i64);
        k.nuses.items[a] += 1;
        k.nuses.items[b] += 1;
        try k.emit(.{ .arith = .{ .op = op, .dst = v, .a = a, .b = b } });
        return v;
    }

    fn count(k: *Combiner, o: ir.Op, comptime add: bool) opt.Error!void {
        for (try ssa.usesOf(k.gpa, o, &k.scratch)) |u| {
            if (add) k.nuses.items[u] += 1 else k.nuses.items[u] -|= 1;
        }
    }

    /// Rewrites block `b` in place; returns whether it changed.
    fn block(k: *Combiner, b: u32) opt.Error!bool {
        const c = k.c;
        k.out.clearRetainingCapacity();
        k.local_consts.clearRetainingCapacity();
        var changed = false;
        for (c.f.blocks.items[b].ops.items) |orig| {
            var o = orig;
            if (k.forwarded) {
                for (try ssa.usesOf(k.gpa, o, &k.scratch)) |u| {
                    if (k.map.items[u] != u) changed = true;
                }
                try opt.mapUses(&o, c.irAlloc(), k.map.items);
            }
            const deleted = for (0..max_steps_per_op) |_| {
                const before = o;
                switch (try k.step(b, &o)) {
                    .none => break false,
                    .changed => {
                        try k.count(before, false);
                        try k.count(o, true);
                        changed = true;
                    },
                    .forward => |x| {
                        try k.count(before, false);
                        const d = o.def().?;
                        k.map.items[d] = x;
                        k.nuses.items[x] += k.nuses.items[d];
                        k.nuses.items[d] = 0;
                        k.forwarded = true;
                        changed = true;
                        break true;
                    },
                }
            } else false;
            if (!deleted) try k.emit(o);
        }
        if (changed) {
            const ops = &c.f.blocks.items[b].ops;
            ops.clearRetainingCapacity();
            try ops.appendSlice(c.irAlloc(), k.out.items);
        }
        return changed;
    }

    /// Applies the first rule that matches `o` (an op of block `b`).
    fn step(k: *Combiner, b: u32, o: *ir.Op) opt.Error!Result {
        switch (o.*) {
            .copy => |x| return .{ .forward = x.src },
            .arith => return k.stepArith(o),
            .typed => |x| {
                if (x.ty.isFloat() or x.a != x.b) return .none;
                switch (x.op) {
                    .@"and", .@"or" => return .{ .forward = x.a },
                    .sub, .xor => {
                        const z = try k.constant(0);
                        o.* = .{ .splat = .{ .ty = x.ty, .dst = x.dst, .src = z } };
                        return .changed;
                    },
                    else => return .none,
                }
            },
            .extract => |x| switch (k.defs.items[x.src]) {
                .splat => |s| {
                    // The lane of a splat is its scalar bit for bit, except that v4i32 lanes
                    // are truncated to 32 bits (and sign-extended back by extract).
                    if (x.ty != .v4i32) return .{ .forward = s.src };
                    const v = k.constOf(s.src) orelse return .none;
                    o.* = .{ .@"const" = .{ .dst = x.dst, .imm = @as(i32, @truncate(v)) } };
                    return .changed;
                },
                else => return .none,
            },
            inline .cbr, .cbra => |*x| {
                const ka = k.constOf(x.a);
                const kb = k.constOf(x.b);
                const taken = if (x.a == x.b)
                    ir.cond(x.cond, 0, 0)
                else if (ka != null and kb != null)
                    ir.cond(x.cond, ka.?, kb.?)
                else {
                    if (ka == null or kb != null) return .none;
                    std.mem.swap(ir.VReg, &x.a, &x.b);
                    x.cond = mirror(x.cond);
                    return .changed;
                };
                const ed = cfg.edgeOf(o.*, if (taken) 0 else 1);
                o.* = if (ed.args.len == 0) .{ .br = ed.to } else .{ .bra = .{ .target = ed.to, .args = ed.args } };
                k.c.taken.items[b] = null;
                return .changed;
            },
            else => return .none,
        }
    }

    fn stepArith(k: *Combiner, o: *ir.Op) opt.Error!Result {
        const x = &o.arith;
        const dst = x.dst;
        const ka = k.constOf(x.a);
        const kb = k.constOf(x.b);
        if (ka != null and kb != null) {
            if (undefinedDiv(x.op, ka.?, kb.?)) return .none;
            return setConst(o, ir.arith(x.op, ka.?, kb.?));
        }
        if (ka != null and commutative(x.op)) {
            std.mem.swap(ir.VReg, &x.a, &x.b);
            return .changed;
        }
        if (x.a == x.b) switch (x.op) {
            .sub, .xor => return setConst(o, 0),
            .@"and", .@"or" => return .{ .forward = x.a },
            else => {},
        };
        if (ka == 0 and (x.op == .shl or x.op == .shr or x.op == .sar)) return setConst(o, 0);
        const c2 = kb orelse return .none;
        switch (x.op) {
            .add, .@"or", .xor => if (c2 == 0) return .{ .forward = x.a },
            .sub => {
                if (c2 == 0) return .{ .forward = x.a };
                if (c2 == std.math.minInt(i64)) return .none;
                x.op = .add;
                x.b = try k.constant(-c2);
                return .changed;
            },
            .@"and" => {
                if (c2 == 0) return setConst(o, 0);
                if (c2 == -1) return .{ .forward = x.a };
            },
            .mul => {
                if (c2 == 0) return setConst(o, 0);
                if (c2 == 1) return .{ .forward = x.a };
            },
            .shl, .shr, .sar => {
                const m = c2 & 63;
                if (m == 0) return .{ .forward = x.a };
                if (m != c2) {
                    x.b = try k.constant(m);
                    return .changed;
                }
            },
            .sdiv => if (c2 == 1) return .{ .forward = x.a },
            .srem => if (c2 == 1) return setConst(o, 0),
        }
        if (x.op == .@"or" and c2 == -1) return setConst(o, -1);

        // Reassociation with a single-use inner op of the same kind and a constant operand.
        if (k.nuses.items[x.a] == 1) switch (k.defs.items[x.a]) {
            .arith => |in| if (k.constOf(in.b)) |c1| {
                const folded: ?i64 = switch (x.op) {
                    .add => switch (in.op) {
                        .add => c1 +% c2,
                        .sub => c2 -% c1,
                        else => null,
                    },
                    .mul, .@"and", .@"or", .xor => if (in.op == x.op) ir.arith(x.op, c1, c2) else null,
                    .shl, .shr, .sar => if (in.op == x.op and c1 >= 0 and c1 <= 63) blk: {
                        const s = c1 + c2; // both in 1..63 (c2 is canonical here)
                        if (s <= 63) break :blk s;
                        if (x.op == .sar) break :blk 63; // sign fill: the same as by 63
                        return setConst(o, 0); // every bit shifted out
                    } else null,
                    else => null,
                };
                if (folded) |c| {
                    x.a = in.a;
                    x.b = try k.constant(c);
                    return .changed;
                }
            },
            else => {},
        };

        // Powers of two.
        const u: u64 = @bitCast(c2);
        if (u & (u -% 1) != 0 or u == 0) return .none;
        const sh: i64 = @ctz(u);
        switch (x.op) {
            .mul => {
                x.op = .shl;
                x.b = try k.constant(sh);
                return .changed;
            },
            .sdiv, .srem => {
                if (sh < 1 or sh > 62) return .none;
                // Bias negative dividends by 2^k - 1 so the arithmetic shift truncates.
                const c63 = try k.constant(63);
                const bias = if (sh == 1)
                    try k.arith(.shr, x.a, c63)
                else
                    try k.arith(.shr, try k.arith(.sar, x.a, c63), try k.constant(64 - sh));
                const biased = try k.arith(.add, x.a, bias);
                if (x.op == .sdiv) {
                    const by = try k.constant(sh);
                    o.* = .{ .arith = .{ .op = .sar, .dst = dst, .a = biased, .b = by } };
                } else {
                    const dividend = x.a;
                    const masked = try k.arith(.@"and", biased, try k.constant(-c2));
                    o.* = .{ .arith = .{ .op = .sub, .dst = dst, .a = dividend, .b = masked } };
                }
                return .changed;
            },
            else => return .none,
        }
    }
};

fn setConst(o: *ir.Op, imm: i64) Result {
    const dst = o.def().?; // read before `o.*` is overwritten (result location)
    o.* = .{ .@"const" = .{ .dst = dst, .imm = imm } };
    return .changed;
}

fn undefinedDiv(op: ir.Arith, a: i64, b: i64) bool {
    return (op == .sdiv or op == .srem) and (b == 0 or (a == std.math.minInt(i64) and b == -1));
}

fn commutative(op: ir.Arith) bool {
    return switch (op) {
        .add, .mul, .@"and", .@"or", .xor => true,
        else => false,
    };
}

/// The condition `cc'` with `b cc' a == a cc b`.
fn mirror(cc: ir.CondCode) ir.CondCode {
    return switch (cc) {
        .eq, .ne => cc,
        .lt => .gt,
        .gt => .lt,
        .le => .ge,
        .ge => .le,
        .ult => .ugt,
        .ugt => .ult,
        .ule => .uge,
        .uge => .ule,
    };
}

pub fn run(c: *opt.Ctx) opt.Error!bool {
    const gpa = c.gpa;
    const f = c.f;
    var k: Combiner = .{ .c = c, .gpa = gpa };
    defer k.deinit();
    const nv = f.num_vregs;
    try k.defs.appendNTimes(gpa, .other, nv);
    try k.nuses.appendNTimes(gpa, 0, nv);
    try k.map.ensureTotalCapacity(gpa, nv);
    for (0..nv) |v| k.map.appendAssumeCapacity(@intCast(v));
    for (f.blocks.items) |b| for (b.ops.items) |o| try k.count(o, true);
    for (f.blocks.items[0].ops.items) |o| {
        if (o != .@"const") break;
        const gop = try k.entry_consts.getOrPut(gpa, o.@"const".imm);
        if (!gop.found_existing) gop.value_ptr.* = o.@"const".dst;
    }

    var g = try cfg.Cfg.init(gpa, f);
    defer g.deinit();
    var changed = false;
    for (g.rpo) |b| {
        if (try k.block(b)) changed = true;
    }
    // Uses in unreachable blocks (reachable ones were renamed while rewriting).
    if (k.forwarded) try opt.renameUses(c, k.map.items);
    return changed;
}

// ---- tests -------------------------------------------------------------------------------

const t = std.testing;
const testgen = @import("testgen.zig");
const sccp = @import("sccp.zig");
const util = sccp.test_util;

/// Runs instcombine (to a fixpoint) on `src` and returns the live IR, printed.
fn combine(src: []const u8, out: *std.ArrayList(u8)) !void {
    try util.runOn(run, src, out);
}

/// The live IR after instcombine is exactly `want` (after the `func` line).
fn expectCombined(src: []const u8, want: []const u8) !void {
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(t.allocator);
    try combine(src, &out);
    const body = out.items[std.mem.indexOfScalar(u8, out.items, '\n').? + 1 ..];
    t.expectEqualStrings(want, body) catch |e| {
        std.debug.print("from:\n{s}\n", .{src});
        return e;
    };
}

fn expectConst(src: []const u8, imm: i64) !void {
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(t.allocator);
    try combine(src, &out);
    try util.expectRetConst(out.items, imm);
}

test "instcombine: identities forward the operand" {
    try expectCombined(
        \\func i 2
        \\e:
        \\  v2 = const 0
        \\  v3 = const 1
        \\  v4 = const -1
        \\  v12 = const 64
        \\  v5 = add v0 v2
        \\  v6 = sub v5 v2
        \\  v7 = or v6 v2
        \\  v8 = xor v7 v2
        \\  v9 = mul v8 v3
        \\  v10 = and v9 v4
        \\  v11 = shl v10 v2
        \\  v13 = shr v11 v12
        \\  v14 = sar v13 v12
        \\  v15 = sdiv v14 v3
        \\  v16 = copy v15
        \\  v17 = and v16 v16
        \\  v18 = or v17 v17
        \\  v19 = add v2 v18
        \\  ret v19
    , "e:\n  ret v0\n");
}

test "instcombine: results fixed by the operands become constants" {
    inline for (.{
        .{ "mul v0 v2", 0 },   .{ "mul v2 v0", 0 },  .{ "and v0 v2", 0 }, .{ "sub v0 v0", 0 },
        .{ "xor v0 v0", 0 },   .{ "srem v0 v3", 0 }, .{ "shl v2 v0", 0 }, .{ "shr v2 v0", 0 },
        .{ "sar v2 v0", 0 },   .{ "or v0 v4", -1 },  .{ "or v4 v0", -1 }, .{ "add v3 v4", 0 },
        .{ "sdiv v4 v3", -1 }, .{ "srem v5 v3", 0 },
    }) |case| {
        try expectConst("func z 2\ne:\n  v2 = const 0\n  v3 = const 1\n  v4 = const -1\n" ++
            "  v5 = const -9223372036854775808\n  v6 = " ++ case[0] ++ "\n  ret v6\n", case[1]);
    }
}

test "instcombine: strength reduction and shift counts" {
    try expectCombined(
        \\func s 2
        \\e:
        \\  v2 = const 8
        \\  v3 = mul v0 v2
        \\  v4 = const -9223372036854775808
        \\  v5 = mul v4 v0
        \\  v6 = const 65
        \\  v7 = shr v5 v6
        \\  v8 = const -1
        \\  v9 = sar v7 v8
        \\  v10 = add v9 v3
        \\  ret v10
    ,
        \\e:
        \\  v11 = const 3
        \\  v3 = shl v0 v11
        \\  v12 = const 63
        \\  v5 = shl v0 v12
        \\  v13 = const 1
        \\  v7 = shr v5 v13
        \\  v9 = sar v7 v12
        \\  v10 = add v9 v3
        \\  ret v10
        \\
    );
}

test "instcombine: reassociation through single-use inner ops" {
    // Each pair collapses into one op with the folded constant (and the inner op dies).
    inline for (.{
        .{ "add", "add", 3, 4, "add", 7 },                                      .{ "sub", "add", 3, 10, "add", 7 },
        .{ "add", "sub", 3, 10, "add", -7 },                                    .{ "mul", "mul", 3, 5, "mul", 15 },
        .{ "and", "and", 12, 10, "and", 8 },                                    .{ "or", "or", 1, 2, "or", 3 },
        .{ "xor", "xor", 5, 3, "xor", 6 },                                      .{ "shl", "shl", 3, 4, "shl", 7 },
        .{ "shr", "shr", 30, 33, "shr", 63 },                                   .{ "sar", "sar", 40, 30, "sar", 63 },
        .{ "mul", "mul", 4, 4, "shl", 4 },                                      .{ "sub", "sub", 5, -9223372036854775807, "add", 9223372036854775802 },
        .{ "add", "add", 9223372036854775807, 2, "add", -9223372036854775807 },
    }) |case| {
        var out: std.ArrayList(u8) = .empty;
        defer out.deinit(t.allocator);
        try combine(std.fmt.comptimePrint("func r 2\ne:\n  v2 = const {d}\n  v3 = const {d}\n  v4 = {s} v0 v2\n  v5 = {s} v4 v3\n  ret v5\n", .{ case[2], case[3], case[0], case[1] }), &out);
        const want = std.fmt.comptimePrint("e:\n  v6 = const {d}\n  v7 = {s} v0 v6\n  ret v7\n", .{ case[5], case[4] });
        // Exactly two live ops: the folded constant and one `op v0 c`.
        var lines = std.mem.tokenizeScalar(u8, out.items, '\n');
        var n: usize = 0;
        while (lines.next()) |_| n += 1;
        if (n != 5) {
            std.debug.print("want shape:\n{s}\ngot:\n{s}\n", .{ want, out.items });
            return error.TestUnexpectedResult;
        }
        try util.expectContains(out.items, case[4] ++ " v0 v");
        try util.expectContains(out.items, std.fmt.comptimePrint("const {d}\n", .{case[5]}));
    }
    // Shifting every bit out folds to 0.
    try expectConst("func r 2\ne:\n  v2 = const 40\n  v3 = shl v0 v2\n  v4 = shl v3 v2\n  ret v4\n", 0);
    try expectConst("func r 2\ne:\n  v2 = const 40\n  v3 = shr v0 v2\n  v4 = shr v3 v2\n  ret v4\n", 0);
}

test "instcombine: no reassociation when the inner op has other uses" {
    try expectCombined(
        \\func r 2
        \\e:
        \\  v2 = const 3
        \\  v3 = add v0 v2
        \\  v4 = add v3 v2
        \\  store v1 0 v3
        \\  ret v4
    ,
        \\e:
        \\  v2 = const 3
        \\  v3 = add v0 v2
        \\  v4 = add v3 v2
        \\  store v1 0 v3
        \\  ret v4
        \\
    );
}

test "instcombine: canonical forms (sub of a constant, constants on the right)" {
    try expectCombined(
        \\func c 2
        \\e:
        \\  v2 = const 5
        \\  v3 = sub v0 v2
        \\  v4 = mul v2 v3
        \\  v5 = const -9223372036854775808
        \\  v6 = sub v4 v5
        \\  ret v6
    ,
        \\e:
        \\  v2 = const 5
        \\  v7 = const -5
        \\  v3 = add v0 v7
        \\  v4 = mul v3 v2
        \\  v5 = const -9223372036854775808
        \\  v6 = sub v4 v5
        \\  ret v6
        \\
    );
}

test "instcombine: division by 0, -1 and minInt is left alone" {
    inline for (.{ "0", "-1", "-9223372036854775808" }) |d| inline for (.{ "sdiv", "srem" }) |op| {
        var out: std.ArrayList(u8) = .empty;
        defer out.deinit(t.allocator);
        try combine("func d 2\ne:\n  v2 = const " ++ d ++ "\n  v3 = " ++ op ++ " v0 v2\n  ret v3\n", &out);
        try util.expectContains(out.items, op ++ " v0 v2");
    };
    // Constant operands: minInt / -1 and x / 0 are not folded either.
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(t.allocator);
    try combine("func d 2\ne:\n  v2 = const -9223372036854775808\n  v3 = const -1\n  v4 = sdiv v2 v3\n  v5 = const 0\n  v6 = srem v3 v5\n  v7 = add v4 v6\n  ret v7\n", &out);
    try util.expectContains(out.items, "sdiv v2 v3");
    try util.expectContains(out.items, "srem v3 v5");
}

test "instcombine: sdiv/srem by every power of two 2^1..2^62 are exact shift sequences" {
    const gpa = t.allocator;
    for (1..63) |k| {
        const p: i64 = @as(i64, 1) << @intCast(k);
        var src: std.ArrayList(u8) = .empty;
        defer src.deinit(gpa);
        try src.print(gpa, "func p 2\ne:\n  v2 = const {d}\n  v3 = sdiv v0 v2\n  v4 = srem v0 v2\n  store v1 0 v4\n  ret v3\n", .{p});
        var ref = try ir.parse(gpa, src.items);
        defer ref.deinit();
        var f = try ir.parse(gpa, src.items);
        defer f.deinit();
        const options: opt.Options = .{ .verify_each = true };
        var stats: opt.Stats = .{};
        var c = try opt.Ctx.init(gpa, &f, &options, &stats);
        defer c.deinit();
        try ssa.build(&c);
        try t.expect(try run(&c));
        try c.check("instcombine");
        try t.expect(!try run(&c));
        for (f.blocks.items[0].ops.items) |o| if (o == .arith) try t.expect(o.arith.op != .sdiv and o.arith.op != .srem);
        const min = std.math.minInt(i64);
        const max = std.math.maxInt(i64);
        const inputs = [_]i64{ 0, 1, -1, 2, -2, min, max, min + 1, max - 1, p, -p, p + 1, p - 1, -p + 1, -p - 1, 3 *% p, -%(3 *% p) +% 1, @divTrunc(min, 2), min + p, max - p + 1 };
        try testgen.expectSame(&ref, &f, &inputs, 1000);
        for (inputs) |x| { // and against the definition, not only the interpreter run above
            var buf: [testgen.mem_slots]u64 align(16) = undefined;
            const q = try ir.interpret(&f, &.{ x, @intCast(@intFromPtr(&buf)) }, 1000);
            try t.expectEqual(@divTrunc(x, p), q);
            try t.expectEqual(@rem(x, p), @as(i64, @bitCast(buf[0])));
        }
    }
}

test "instcombine: branches on equal or constant operands; constant left operands" {
    try expectCombined(
        \\func b 2
        \\e:
        \\  v2 = const 4
        \\  cbr lt v2 v0 x y
        \\x:
        \\  cbr uge v1 v1 y z
        \\y:
        \\  v3 = const 7
        \\  cbr eq v2 v3 z w
        \\z:
        \\  ret v0
        \\w:
        \\  ret v1
    ,
        \\e:
        \\  v2 = const 4
        \\  cbr gt v0 v2 x y
        \\x:
        \\  br y
        \\y:
        \\  br w
        \\z:
        \\  ret v0
        \\w:
        \\  ret v1
        \\
    );
    // cbra keeps the arguments of the edge it takes.
    try expectCombined(
        \\func b 2
        \\e:
        \\  cbr ne v0 v0 j:v0 j:v1
        \\j(v2):
        \\  ret v2
    ,
        \\e:
        \\  br j v1
        \\j(v2):
        \\  ret v2
        \\
    );
}

test "instcombine: folding a branch forgets its taken count, swapping keeps it" {
    const gpa = t.allocator;
    var f = try ir.parse(gpa,
        \\func b 1
        \\e:
        \\  v1 = const 4
        \\  cbr lt v1 v0 x y
        \\x:
        \\  cbr eq v0 v0 y z
        \\y:
        \\  ret v0
        \\z:
        \\  ret v1
    );
    defer f.deinit();
    const prof: opt.Profile = .{ .blocks = &.{ 10, 6, 10, 0 }, .taken = &.{ 6, 6, 0, 0 } };
    const options: opt.Options = .{ .verify_each = true, .profile = &prof };
    var stats: opt.Stats = .{};
    var c = try opt.Ctx.init(gpa, &f, &options, &stats);
    defer c.deinit();
    try ssa.build(&c);
    try t.expect(try run(&c));
    try c.check("instcombine");
    try t.expectEqual(@as(?u64, 6), c.taken.items[0]);
    try t.expectEqual(@as(?u64, null), c.taken.items[1]);
    try t.expect(cfg.terminator(&f, 1).* == .br);
}

test "instcombine: float ops get no identities" {
    const src =
        \\func f 2
        \\e:
        \\  v2 = load.f64 v1 0
        \\  v3 = fconst.f64 0.0
        \\  v4 = fconst.f64 1.0
        \\  v5 = add.f64 v2 v3
        \\  v6 = mul.f64 v5 v4
        \\  v7 = sub.f64 v6 v6
        \\  v8 = min.f64 v7 v7
        \\  store.f64 v1 8 v8
        \\  v9 = const 0
        \\  ret v9
    ;
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(t.allocator);
    try combine(src, &out);
    inline for (.{ "add.f64 v2 v3", "mul.f64 v5 v4", "sub.f64 v6 v6", "min.f64 v7 v7" }) |s| try util.expectContains(out.items, s);
}

test "instcombine: integer vector rules and extract of splat" {
    try expectCombined(
        \\func v 2
        \\e:
        \\  v2 = splat.v2i64 v0
        \\  v3 = and.v2i64 v2 v2
        \\  v4 = or.v2i64 v3 v3
        \\  store.v2i64 v1 0 v4
        \\  v5 = splat.v4i32 v0
        \\  v6 = xor.v4i32 v5 v5
        \\  v7 = sub.v4i32 v5 v5
        \\  store.v4i32 v1 16 v6
        \\  store.v4i32 v1 32 v7
        \\  v8 = extract.v2i64 v4 1
        \\  v9 = const 4294967301
        \\  v10 = splat.v4i32 v9
        \\  v11 = extract.v4i32 v10 2
        \\  v12 = extract.v4i32 v5 0
        \\  v13 = add v8 v11
        \\  v14 = add v13 v12
        \\  ret v14
    ,
        \\e:
        \\  v2 = splat.v2i64 v0
        \\  store.v2i64 v1 0 v2
        \\  v5 = splat.v4i32 v0
        \\  v15 = const 0
        \\  v6 = splat.v4i32 v15
        \\  v7 = splat.v4i32 v15
        \\  store.v4i32 v1 16 v6
        \\  store.v4i32 v1 32 v7
        \\  v11 = const 5
        \\  v12 = extract.v4i32 v5 0
        \\  v13 = add v0 v11
        \\  v14 = add v13 v12
        \\  ret v14
        \\
    );
}

/// A random straight-line function over `v0` with many constant operands (powers of two,
/// 0, ±1, minInt, maxInt, shift counts out of range), every `ir.Arith` op, copies, and
/// branches on constants or equal operands; results are stored to memory and returned.
fn arithProgram(rng: std.Random, gpa: std.mem.Allocator, out: *std.ArrayList(u8)) !void {
    const specials = [_]i64{ 0, 1, -1, 2, 3, 8, 64, 65, -64, 1 << 20, 1 << 62, std.math.minInt(i64), std.math.maxInt(i64), -(1 << 33) };
    try out.print(gpa, "func a 2\nb0:\n", .{});
    var nv: u32 = 2;
    var blk: u32 = 0;
    const n = rng.intRangeAtMost(u32, 4, 24);
    for (0..n) |_| {
        const pick = struct {
            fn f(r: std.Random, w: *std.ArrayList(u8), g: std.mem.Allocator, v: *u32, sp: []const i64) !u32 {
                // Existing values, the newest one most often (chains for reassociation).
                if (r.uintLessThan(u32, 3) != 0) return switch (if (v.* == 2) 0 else r.uintLessThan(u32, 4)) {
                    0 => 0,
                    1 => r.intRangeLessThan(u32, 2, v.*),
                    else => v.* - 1,
                };
                const val = if (r.boolean()) sp[r.uintLessThan(usize, sp.len)] else if (r.boolean())
                    @as(i64, 1) << r.int(u6)
                else
                    r.intRangeAtMost(i64, -9, 9);
                try w.print(g, "  v{d} = const {d}\n", .{ v.*, val });
                v.* += 1;
                return v.* - 1;
            }
        }.f;
        const a = try pick(rng, out, gpa, &nv, &specials);
        const b = try pick(rng, out, gpa, &nv, &specials);
        switch (rng.uintLessThan(u32, 10)) {
            0 => try out.print(gpa, "  v{d} = copy v{d}\n", .{ nv, a }),
            1 => { // a branch on constants / equal operands, joining right after
                try out.print(gpa, "  cbr {s} v{d} v{d} b{d} b{d}\nb{d}:\n  br b{d}\nb{d}:\n", .{ @tagName(rng.enumValue(ir.CondCode)), a, if (rng.boolean()) a else b, blk + 1, blk + 2, blk + 1, blk + 2, blk + 2 });
                blk += 2;
                continue;
            },
            2 => try out.print(gpa, "  store v1 {d} v{d}\n", .{ 8 * rng.uintLessThan(u32, testgen.mem_slots), a }),
            else => try out.print(gpa, "  v{d} = {s} v{d} v{d}\n", .{ nv, @tagName(rng.enumValue(ir.Arith)), a, b }),
        }
        nv += 1;
    }
    try out.print(gpa, "  store v1 0 v{d}\n  ret v{d}\n", .{ nv - 1, nv - 2 });
}

/// `arithProgram`s through `pre` + `pass` on `default_inputs` plus random inputs.
pub fn fuzzArith(gpa: std.mem.Allocator, seed: u64, iterations: usize, comptime pre: []const testgen.Pass, comptime pass: testgen.Pass) !void {
    var prng = std.Random.DefaultPrng.init(seed);
    const rng = prng.random();
    var src: std.ArrayList(u8) = .empty;
    defer src.deinit(gpa);
    for (0..iterations) |_| {
        src.clearRetainingCapacity();
        try arithProgram(rng, gpa, &src);
        checkArith(gpa, rng, src.items, pre, pass) catch |e| {
            std.debug.print("fuzzArith failure on:\n{s}\n", .{src.items});
            return e;
        };
    }
}

fn checkArith(gpa: std.mem.Allocator, rng: std.Random, src: []const u8, comptime pre: []const testgen.Pass, comptime pass: testgen.Pass) !void {
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
    inline for (pre) |p| {
        _ = try p(&c);
        try c.check("pre");
    }
    _ = try pass(&c);
    try c.check("pass");
    var inputs: [16]i64 = undefined;
    @memcpy(inputs[0..testgen.default_inputs.len], &testgen.default_inputs);
    for (inputs[testgen.default_inputs.len..]) |*x| x.* = if (rng.boolean()) rng.int(i64) else rng.intRangeAtMost(i64, -300, 300);
    // Like `testgen.expectSame` (these programs make no calls), with the interpreter's scratch
    // from `gpa` rather than the page allocator (much faster in Debug).
    for (inputs) |x| {
        var want_mem: [testgen.mem_slots]u64 align(16) = undefined;
        var got_mem: [testgen.mem_slots]u64 align(16) = undefined;
        for (&want_mem, &got_mem, 0..) |*w, *g, i| {
            w.* = @as(u64, @bitCast(x)) *% 0x9e3779b97f4a7c15 +% i;
            g.* = w.*;
        }
        const want = try ir.interpretWithAllocator(gpa, &ref, &.{ x, @intCast(@intFromPtr(&want_mem)) }, 10_000);
        const got = try ir.interpretWithAllocator(gpa, &f, &.{ x, @intCast(@intFromPtr(&got_mem)) }, 10_000);
        try t.expectEqual(want, got);
        try t.expectEqualSlices(u64, &want_mem, &got_mem);
    }
}

test "instcombine: random arithmetic with constants" {
    var da: util.FastAlloc = .init;
    defer if (da.deinit() == .leak) @panic("instcombine: memory leak");
    const gpa = da.allocator();
    try fuzzArith(gpa, 11, 200, &.{}, run);
    try fuzzArith(gpa, 12, 150, &.{run}, run);
    try fuzzArith(gpa, 13, 150, &.{sccp.run}, run);
    try fuzzArith(gpa, 14, 150, &.{}, sccp.run);
}

test "instcombine: random programs" {
    var da: util.FastAlloc = .init;
    defer if (da.deinit() == .leak) @panic("instcombine: memory leak");
    const gpa = da.allocator();
    try testgen.fuzzPass(gpa, 21, 150, .{}, &.{}, run);
    try testgen.fuzzPass(gpa, 22, 100, .{ .loops = false }, &.{sccp.run}, run);
    try testgen.fuzzPass(gpa, 23, 100, .{ .blocks = 10, .ops = 10, .floats = false }, &.{ sccp.run, run }, run);
    try testgen.fuzzPass(gpa, 24, 100, .{ .blocks = 3, .ops = 16, .calls = false }, &.{run}, sccp.run);
}

/// Optimizes `src` (the pipeline at `level`), checks it against the original and prints the
/// result into `out`.
fn optimizeText(src: []const u8, level: opt.Level, out: *std.ArrayList(u8)) !opt.Stats {
    const gpa = t.allocator;
    var ref = try ir.parse(gpa, src);
    defer ref.deinit();
    var f = try ir.parse(gpa, src);
    defer f.deinit();
    const stats = try opt.optimize(gpa, &f, .{ .level = level, .verify_each = true });
    try f.verify();
    try testgen.expectSame(&ref, &f, &testgen.default_inputs, 20_000);
    try f.print(gpa, out);
    return stats;
}

test "instcombine: bench-like loops through the pipeline (mul by 8 becomes shl)" {
    // The bench kernels (bench/exec_bench.zig) with the testgen buffer at v1 as the array.
    inline for (.{
        \\func sum_i64 2
        \\e:
        \\  v2 = const 0
        \\  v3 = const 0
        \\  v4 = const 8
        \\  v5 = const 1
        \\  v9 = const 8
        \\  cbr ge v2 v9 done loop
        \\loop:
        \\  v6 = mul v2 v4
        \\  v7 = add v1 v6
        \\  v8 = load v7 0
        \\  v3 = add v3 v8
        \\  v2 = add v2 v5
        \\  cbr lt v2 v9 loop done
        \\done:
        \\  ret v3
        ,
        \\func add_f64 2
        \\e:
        \\  v4 = const 0
        \\  v5 = const 8
        \\  v6 = const 1
        \\  v3 = const 4
        \\  cbr ge v4 v3 done loop
        \\loop:
        \\  v7 = mul v4 v5
        \\  v8 = add v1 v7
        \\  v14 = const 32
        \\  v9 = add v8 v14
        \\  v11 = load.f64 v8 0
        \\  v12 = load.f64 v9 0
        \\  v13 = add.f64 v11 v12
        \\  store.f64 v8 0 v13
        \\  v4 = add v4 v6
        \\  cbr lt v4 v3 loop done
        \\done:
        \\  ret v4
        ,
        \\func fib_like 2
        \\e:
        \\  v2 = const 2
        \\  v8 = const 1
        \\  v9 = and v0 v8
        \\  cbr lt v9 v2 base rec
        \\base:
        \\  ret v0
        \\rec:
        \\  v3 = const 1
        \\  v4 = sub v0 v3
        \\  v5 = sub v4 v3
        \\  v6 = mul v5 v2
        \\  v7 = add v4 v6
        \\  ret v7
    }) |src| inline for (.{ .o1, .o2 }) |level| {
        var out: std.ArrayList(u8) = .empty;
        defer out.deinit(t.allocator);
        _ = try optimizeText(src, level, &out);
        try util.expectLacks(out.items, "mul");
        try util.expectContains(out.items, "shl");
    };
}
