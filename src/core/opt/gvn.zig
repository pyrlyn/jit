//! Global value numbering (dominator-based) and redundant load elimination.
//!
//! A preorder walk of the dominator tree keeps a scoped hash table from "what an op computes"
//! (op kind, immediates, types and the value numbers of its operands) to the first vreg that
//! computed it; entries are removed again when the walk leaves the block that added them, so
//! a hit is always an op that dominates the current one. A pure op whose value is already in
//! the table (`const`, `fconst`, `arith` including `sdiv`/`srem` - the dominating copy
//! already executed with the same operands -, `typed`, `fcmp`, `cvt`, `splat`, `extract`) gets
//! the earlier vreg as its value number; a `copy` gets its source's. All uses are rewritten
//! at the end in one pass; the redundant ops stay for dce.
//!
//! Commutative ops are keyed with sorted operands: integer `add`/`mul`/`and`/`or`/`xor`,
//! typed `add`/`mul` (IEEE add and mul are commutative up to the payload of a NaN result, which
//! tests treat as "any NaN") and typed `and`/`or`/`xor`. Float `min`/`max` (the second operand
//! wins on NaN and on equal values), `sub`, `div` and shifts are not commutative. `fcmp eq/ne`
//! are commutative; `gt`/`ge` are keyed as `lt`/`le` with swapped operands (exact, also on NaN).
//!
//! Memory: a small table of known memory contents (base value number, displacement, type ->
//! value) is kept per block. A `load` that matches an entry reuses its value (an earlier load
//! of the same location, or the value an earlier store wrote there: store-to-load
//! forwarding); otherwise it is recorded. A `store` drops every entry it may overlap - all
//! entries with another base (different vregs may alias) and those with the same base whose
//! byte range intersects the stored one - and records the stored value. A `call` clears the
//! table. The table starts empty at a block entry, except that a block whose only predecessor
//! is its immediate dominator `P` starts with `P`'s table at its end (control comes straight
//! from there, so the memory state is the same). The table is bounded (`mem_cap`, oldest
//! entries dropped) to stay linear on long blocks.
const std = @import("std");
const ir = @import("../ir.zig");
const opt = @import("opt.zig");
const cfg = @import("cfg.zig");
const ssa = @import("ssa.zig");

const none = cfg.none;

/// What a pure op computes, with operands as value numbers (commutative ones sorted).
/// Three words without padding, hashed and compared as such (`KeyContext`).
const Key = extern struct {
    a: ir.VReg = 0,
    b: ir.VReg = 0,
    /// `const` immediate, `fconst` bits.
    imm: u64 = 0,
    kind: Kind,
    /// Arith / typed op, fcmp condition.
    sub: u8 = 0,
    /// Type (`cvt`: destination type).
    ty: u8 = 0,
    /// `cvt`: source type; `extract`: lane.
    aux: u8 = 0,
    pad: u32 = 0,

    const Kind = enum(u8) { @"const", fconst, arith, typed, fcmp, cvt, splat, extract };
    fn words(k: *const Key) [3]u64 {
        comptime std.debug.assert(@sizeOf(Key) == 24);
        return @as(*const [3]u64, @ptrCast(k)).*;
    }
};

const KeyContext = struct {
    pub fn hash(_: KeyContext, k: Key) u64 {
        const w = k.words();
        var h = (w[0] ^ (w[2] << 7)) *% 0x9e3779b97f4a7c15;
        h = (h ^ (h >> 29) ^ w[1]) *% 0xbf58476d1ce4e5b9;
        return h ^ (h >> 32);
    }
    pub fn eql(_: KeyContext, x: Key, y: Key) bool {
        const a = x.words();
        const b = y.words();
        return a[0] == b[0] and a[1] == b[1] and a[2] == b[2];
    }
};

/// One known memory content: `ty` at `base + disp` holds `value`.
const Mem = struct { base: ir.VReg, disp: i32, ty: ir.Type, value: ir.VReg };

/// Entries kept per block (a store or a load beyond this forgets the oldest entry).
const mem_cap = 32;

/// Memory table of the block being walked (see the module comment).
const MemState = struct {
    items: [mem_cap]Mem = undefined,
    len: usize = 0,

    fn find(s: *const MemState, base: ir.VReg, disp: i32, ty: ir.Type) ?ir.VReg {
        for (s.items[0..s.len]) |e| if (e.base == base and e.disp == disp and e.ty == ty) return e.value;
        return null;
    }
    fn add(s: *MemState, e: Mem) void {
        if (s.len == mem_cap) {
            std.mem.copyForwards(Mem, s.items[0 .. mem_cap - 1], s.items[1..mem_cap]);
            s.len -= 1;
        }
        s.items[s.len] = e;
        s.len += 1;
    }
    /// Effect of `store.ty base disp value`: forget what it may overwrite, remember the value.
    fn store(s: *MemState, base: ir.VReg, disp: i32, ty: ir.Type, value: ir.VReg) void {
        const lo: i64 = disp;
        const hi: i64 = lo + ty.size();
        var k: usize = 0;
        for (s.items[0..s.len]) |e| {
            const elo: i64 = e.disp;
            const ehi: i64 = elo + e.ty.size();
            if (e.base == base and (ehi <= lo or hi <= elo)) {
                s.items[k] = e;
                k += 1;
            }
        }
        s.len = k;
        s.add(.{ .base = base, .disp = disp, .ty = ty, .value = value });
    }
};

fn commutes(a: ir.Arith) bool {
    return switch (a) {
        .add, .mul, .@"and", .@"or", .xor => true,
        else => false,
    };
}

fn tcommutes(a: ir.TArith) bool {
    return switch (a) {
        .add, .mul, .@"and", .@"or", .xor => true,
        .sub, .div, .min, .max => false,
    };
}

/// Key of a pure op `o` (operands through `vn`), `null` for ops that are not numbered.
fn keyOf(o: ir.Op, vn: []const ir.VReg) ?Key {
    var k: Key = switch (o) {
        .@"const" => |x| .{ .kind = .@"const", .imm = @bitCast(x.imm) },
        .fconst => |x| .{ .kind = .fconst, .ty = @backingInt(x.ty), .imm = x.bits },
        .arith => |x| .{ .kind = .arith, .sub = @backingInt(x.op), .a = vn[x.a], .b = vn[x.b] },
        .typed => |x| .{ .kind = .typed, .sub = @backingInt(x.op), .ty = @backingInt(x.ty), .a = vn[x.a], .b = vn[x.b] },
        .fcmp => |x| switch (x.cond) {
            .gt => .{ .kind = .fcmp, .sub = @backingInt(ir.FCond.lt), .ty = @backingInt(x.ty), .a = vn[x.b], .b = vn[x.a] },
            .ge => .{ .kind = .fcmp, .sub = @backingInt(ir.FCond.le), .ty = @backingInt(x.ty), .a = vn[x.b], .b = vn[x.a] },
            else => .{ .kind = .fcmp, .sub = @backingInt(x.cond), .ty = @backingInt(x.ty), .a = vn[x.a], .b = vn[x.b] },
        },
        .cvt => |x| .{ .kind = .cvt, .ty = @backingInt(x.to), .aux = @backingInt(x.from), .a = vn[x.src] },
        .splat => |x| .{ .kind = .splat, .ty = @backingInt(x.ty), .a = vn[x.src] },
        .extract => |x| .{ .kind = .extract, .ty = @backingInt(x.ty), .aux = x.lane, .a = vn[x.src] },
        else => return null,
    };
    const sorted = switch (o) {
        .arith => |x| commutes(x.op),
        .typed => |x| tcommutes(x.op),
        .fcmp => |x| x.cond == .eq or x.cond == .ne,
        else => false,
    };
    if (sorted and k.a > k.b) std.mem.swap(ir.VReg, &k.a, &k.b);
    return k;
}

pub fn run(c: *opt.Ctx) opt.Error!bool {
    const gpa = c.gpa;
    const f = c.f;
    var g = try cfg.Cfg.init(gpa, f);
    defer g.deinit();

    // Value number of every vreg: itself, or an equal vreg whose definition dominates it.
    const vn = try gpa.alloc(ir.VReg, f.num_vregs);
    defer gpa.free(vn);
    for (vn, 0..) |*v, i| v.* = @intCast(i);

    var table: std.HashMapUnmanaged(Key, ir.VReg, KeyContext, std.hash_map.default_max_load_percentage) = .empty;
    defer table.deinit(gpa);
    try table.ensureTotalCapacity(gpa, @intCast(@min(opt.countOps(f), f.num_vregs)));
    var undo: std.ArrayList(Key) = .empty; // keys added by the blocks on the walk stack
    defer undo.deinit(gpa);
    // Memory tables at the end of the blocks on the stack (for single-predecessor children).
    var saved: std.ArrayList(Mem) = .empty;
    defer saved.deinit(gpa);

    const Frame = struct { b: u32, next: u32, undo: usize, saved: usize, saved_len: usize };
    var stack: std.ArrayList(Frame) = .empty;
    defer stack.deinit(gpa);
    var hs: std.ArrayList(ParamHash) = .empty;
    defer hs.deinit(gpa);
    var mem: MemState = .{};
    var any = false;
    var next: u32 = 0; // block to enter (`none`: continue with the stack top)
    while (true) {
        if (next != none) {
            const b = next;
            const undo_mark = undo.items.len;
            if (f.blocks.items[b].params.len > 1 and try equalParams(c, &g, vn, b, &hs)) any = true;
            for (f.blocks.items[b].ops.items) |o| switch (o) {
                .copy => |x| {
                    vn[x.dst] = vn[x.src];
                    any = true;
                },
                .load => |x| {
                    const base = vn[x.base];
                    if (mem.find(base, x.disp, x.ty)) |v| {
                        vn[x.dst] = v;
                        any = true;
                    } else mem.add(.{ .base = base, .disp = x.disp, .ty = x.ty, .value = x.dst });
                },
                .store => |x| mem.store(vn[x.base], x.disp, x.ty, vn[x.src]),
                .call => mem.len = 0,
                else => if (keyOf(o, vn)) |k| {
                    const gop = try table.getOrPut(gpa, k);
                    if (gop.found_existing) {
                        vn[o.def().?] = gop.value_ptr.*;
                        any = true;
                    } else {
                        gop.value_ptr.* = o.def().?;
                        try undo.append(gpa, k);
                    }
                },
            };
            const saved_mark = saved.items.len;
            try saved.appendSlice(gpa, mem.items[0..mem.len]);
            try stack.append(gpa, .{ .b = b, .next = 0, .undo = undo_mark, .saved = saved_mark, .saved_len = mem.len });
        }
        if (stack.items.len == 0) break;
        const top = &stack.items[stack.items.len - 1];
        const ch = g.dom_children[top.b];
        if (top.next < ch.len) {
            next = ch[top.next];
            top.next += 1;
            mem.len = 0;
            if (onlyPred(&g, next, top.b)) {
                @memcpy(mem.items[0..top.saved_len], saved.items[top.saved .. top.saved + top.saved_len]);
                mem.len = top.saved_len;
            }
        } else {
            for (undo.items[top.undo..]) |k| _ = table.remove(k);
            undo.items.len = top.undo;
            saved.items.len = top.saved;
            _ = stack.pop();
            next = none;
        }
    }
    if (!any) return false;
    return renameChanged(c, vn);
}

/// Parameters of block `b` that receive equal values (by the current `vn`, which is sound at
/// any time) on every reachable incoming edge are equal: later ones get the first one's value
/// number. Run when the walk enters `b`, when every forward-edge predecessor has been numbered;
/// pessimistic for back edges (a parameter passed back to itself around a loop only matches
/// itself). Parameters are grouped by a hash of their arguments, so this is linear in the
/// edge arguments (plus sorting). `hs` is scratch.
fn equalParams(c: *opt.Ctx, g: *const cfg.Cfg, vn: []ir.VReg, b: u32, hs: *std.ArrayList(ParamHash)) opt.Error!bool {
    const f = c.f;
    const params = f.blocks.items[b].params;
    hs.clearRetainingCapacity();
    for (0..params.len) |i| {
        var it: ArgIter = .{ .f = f, .g = g, .b = b, .i = @intCast(i) };
        var h: u64 = 0x2545f4914f6cdd1d;
        while (it.next()) |a| h = (h ^ vn[a]) *% 0x9e3779b97f4a7c15;
        try hs.append(c.gpa, .{ .h = h, .i = @intCast(i) });
    }
    std.mem.sort(ParamHash, hs.items, {}, struct {
        fn lt(_: void, x: ParamHash, y: ParamHash) bool {
            return if (x.h != y.h) x.h < y.h else x.i < y.i;
        }
    }.lt);
    var any = false;
    var first: usize = 0; // first parameter of the current run of equal hashes
    for (hs.items, 0..) |x, k| {
        if (x.h != hs.items[first].h) first = k;
        if (k == first) continue;
        const p = hs.items[first].i;
        if (!sameArgs(f, g, vn, b, p, x.i)) continue;
        vn[params[x.i]] = params[p];
        any = true;
    }
    return any;
}

const ParamHash = struct { h: u64, i: u32 };

/// Arguments for parameter `i` of block `b` on its reachable incoming edges, in a fixed order.
const ArgIter = struct {
    f: *const ir.Func,
    g: *const cfg.Cfg,
    b: u32,
    i: u32,
    /// Position in `g.preds[b]` and next edge of that predecessor's terminator.
    pred: usize = 0,
    edge: u8 = 0,

    fn next(it: *ArgIter) ?ir.VReg {
        const preds = it.g.preds[it.b];
        while (it.pred < preds.len) {
            const q = preds[it.pred];
            if (it.g.reachable(q)) {
                const ops = it.f.blocks.items[q].ops.items;
                const term = ops[ops.len - 1];
                while (it.edge < ssa.numSuccs(term)) {
                    const e = cfg.edgeOf(term, @intCast(it.edge));
                    it.edge += 1;
                    if (e.to == it.b) return e.args[it.i];
                }
            }
            it.edge = 0;
            while (it.pred < preds.len and preds[it.pred] == q) it.pred += 1; // one entry per edge
        }
        return null;
    }
};

fn sameArgs(f: *const ir.Func, g: *const cfg.Cfg, vn: []const ir.VReg, b: u32, i: u32, j: u32) bool {
    var x: ArgIter = .{ .f = f, .g = g, .b = b, .i = i };
    var y: ArgIter = .{ .f = f, .g = g, .b = b, .i = j };
    while (x.next()) |a| if (vn[a] != vn[y.next().?]) return false;
    return true;
}

/// Every reachable predecessor edge of `b` comes from `p`.
fn onlyPred(g: *const cfg.Cfg, b: u32, p: u32) bool {
    for (g.preds[b]) |q| if (q != p and g.reachable(q)) return false;
    return true;
}

/// `opt.renameUses` that only touches ops reading a renamed vreg; returns whether any did.
fn renameChanged(c: *opt.Ctx, map: []const ir.VReg) opt.Error!bool {
    var changed = false;
    var buf: [ir.max_args]ir.VReg = undefined;
    for (c.f.blocks.items) |*b| for (b.ops.items) |*op| {
        const reads = switch (op.*) {
            .bra => |x| renamed(x.args, map),
            .cbra => |x| map[x.a] != x.a or map[x.b] != x.b or renamed(x.then_args, map) or renamed(x.else_args, map),
            else => renamed(op.uses(&buf), map),
        };
        if (!reads) continue;
        try opt.mapUses(op, c.irAlloc(), map);
        changed = true;
    };
    return changed;
}

fn renamed(vs: []const ir.VReg, map: []const ir.VReg) bool {
    for (vs) |v| if (map[v] != v) return true;
    return false;
}

// ---- tests -------------------------------------------------------------------------------

const t = std.testing;
const testgen = @import("testgen.zig");

/// `testgen.checkPass` that hands back the transformed function (caller deinits it) and
/// whether the pass reported a change.
fn transform(src: []const u8, comptime pass: fn (*opt.Ctx) opt.Error!bool) !struct { ir.Func, bool } {
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
    const changed = try pass(&c);
    try c.check("pass");
    try testgen.expectSame(&ref, &f, &testgen.default_inputs, 20_000);
    return .{ f, changed };
}

/// Reads of every vreg of `src` after GVN.
fn usesAfterGvn(src: []const u8) ![]u32 {
    var r = try transform(src, run);
    defer r[0].deinit();
    return opt.useCounts(t.allocator, &r[0]);
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

test "gvn: redundant arithmetic in dominated blocks, commutative operands, sub kept" {
    var r = try transform(
        \\func g 2
        \\e:
        \\  v2 = add v0 v1
        \\  v20 = const 5
        \\  cbr lt v0 v1 a b
        \\a:
        \\  v3 = add v1 v0
        \\  v4 = mul v3 v2
        \\  v21 = const 5
        \\  v22 = xor v4 v21
        \\  ret v22
        \\b:
        \\  v5 = sub v0 v1
        \\  v6 = sub v1 v0
        \\  v7 = sub v0 v1
        \\  v8 = add v6 v7
        \\  v9 = add v8 v5
        \\  v10 = copy v9
        \\  v11 = add v8 v5
        \\  v12 = mul v10 v11
        \\  ret v12
    , run);
    defer r[0].deinit();
    try t.expect(r[1]);
    try expectIr(&r[0], "v4 = mul v2 v2");
    try expectIr(&r[0], "v22 = xor v4 v20");
    try expectIr(&r[0], "v8 = add v6 v5");
    try expectIr(&r[0], "v12 = mul v9 v9");
}

test "gvn: no reuse across sibling blocks or at a join" {
    var r = try transform(
        \\func s 2
        \\e:
        \\  cbr lt v0 v1 a b
        \\a:
        \\  v2 = mul v0 v1
        \\  v3 = const 3
        \\  v4 = sdiv v2 v3
        \\  br j v4
        \\b:
        \\  v5 = mul v0 v1
        \\  v6 = const 3
        \\  v7 = srem v5 v6
        \\  br j v7
        \\j(v8):
        \\  v9 = mul v0 v1
        \\  v10 = const 3
        \\  v11 = sdiv v9 v10
        \\  v12 = add v11 v8
        \\  ret v12
    , run);
    defer r[0].deinit();
    try t.expect(!r[1]);
}

test "gvn: sdiv/srem reused only under dominance" {
    var r = try transform(
        \\func d 2
        \\e:
        \\  v2 = const 7
        \\  v3 = sdiv v0 v2
        \\  cbr lt v0 v2 a b
        \\a:
        \\  v4 = sdiv v0 v2
        \\  v5 = srem v0 v2
        \\  v6 = add v4 v5
        \\  br j v6
        \\b:
        \\  v7 = srem v0 v2
        \\  br j v7
        \\j(v8):
        \\  v9 = srem v0 v2
        \\  v10 = mul v9 v8
        \\  v11 = add v10 v3
        \\  ret v11
    , run);
    defer r[0].deinit();
    try t.expect(r[1]);
    try expectIr(&r[0], "v6 = add v3 v5");
    try expectIr(&r[0], "v10 = mul v9 v8");
}

test "gvn: float ops - add/mul/fcmp eq commute, min/max and lt do not, gt is swapped lt" {
    var r = try transform(
        \\func fl 2
        \\e:
        \\  v2 = fconst.f64 -0.0
        \\  v3 = fconst.f64 0.0
        \\  v4 = min.f64 v2 v3
        \\  v5 = min.f64 v3 v2
        \\  v6 = max.f64 v2 v3
        \\  v7 = max.f64 v3 v2
        \\  store.f64 v1 0 v4
        \\  store.f64 v1 8 v5
        \\  store.f64 v1 16 v6
        \\  store.f64 v1 24 v7
        \\  v8 = cvt.f64.i64 v0
        \\  v9 = fconst.f64 1.5
        \\  v10 = add.f64 v8 v9
        \\  v11 = add.f64 v9 v8
        \\  v12 = mul.f64 v8 v9
        \\  v13 = mul.f64 v9 v8
        \\  v14 = sub.f64 v9 v8
        \\  v15 = div.f64 v8 v9
        \\  store.f64 v1 32 v11
        \\  store.f64 v1 40 v13
        \\  store.f64 v1 48 v14
        \\  store.f64 v1 56 v15
        \\  v16 = fcmp.f64 gt v8 v9
        \\  v17 = fcmp.f64 lt v9 v8
        \\  v18 = fcmp.f64 lt v8 v9
        \\  v19 = fcmp.f64 eq v8 v9
        \\  v20 = fcmp.f64 eq v9 v8
        \\  v21 = fcmp.f64 ge v8 v9
        \\  v22 = fcmp.f64 le v9 v8
        \\  v23 = const 2
        \\  v24 = mul v17 v23
        \\  v25 = add v24 v18
        \\  v26 = mul v25 v23
        \\  v27 = add v26 v20
        \\  v28 = mul v27 v23
        \\  v29 = add v28 v22
        \\  v30 = mul v29 v23
        \\  v31 = add v30 v16
        \\  v32 = add v31 v19
        \\  v33 = add v32 v21
        \\  ret v33
    , run);
    defer r[0].deinit();
    try t.expect(r[1]);
    try expectIr(&r[0], "store.f64 v1 8 v5");
    try expectIr(&r[0], "store.f64 v1 24 v7");
    try expectIr(&r[0], "store.f64 v1 32 v10");
    try expectIr(&r[0], "store.f64 v1 40 v12");
    try expectIr(&r[0], "v24 = mul v16 v23");
    try expectIr(&r[0], "v25 = add v24 v18");
    try expectIr(&r[0], "v27 = add v26 v19");
    try expectIr(&r[0], "v29 = add v28 v21");
}

test "gvn: constants, conversions, splat and extract by type" {
    var r = try transform(
        \\func ty 2
        \\e:
        \\  v2 = fconst.f32 1.5
        \\  v3 = fconst.f64 1.5
        \\  v4 = fconst.f32 1.5
        \\  v5 = cvt.i64.f32 v2
        \\  v6 = cvt.i64.f64 v3
        \\  v7 = cvt.i64.f32 v4
        \\  v8 = splat.v2i64 v0
        \\  v9 = splat.v4i32 v0
        \\  v10 = splat.v2i64 v0
        \\  v11 = extract.v2i64 v8 1
        \\  v12 = extract.v2i64 v10 0
        \\  v13 = extract.v2i64 v10 1
        \\  v14 = extract.v4i32 v9 1
        \\  v15 = add.v2i64 v8 v10
        \\  v16 = add.v2i64 v10 v8
        \\  v17 = xor.v2i64 v15 v16
        \\  v18 = extract.v2i64 v17 0
        \\  v19 = add v5 v6
        \\  v20 = add v19 v7
        \\  v21 = add v20 v11
        \\  v22 = add v21 v12
        \\  v23 = add v22 v13
        \\  v24 = add v23 v14
        \\  v25 = add v24 v18
        \\  ret v25
    , run);
    defer r[0].deinit();
    try expectIr(&r[0], "v20 = add v19 v5");
    try expectIr(&r[0], "v22 = add v21 v12");
    try expectIr(&r[0], "v23 = add v22 v11");
    try expectIr(&r[0], "v17 = xor.v2i64 v15 v15");
    try expectIr(&r[0], "v12 = extract.v2i64 v8 0");
    try expectIr(&r[0], "v14 = extract.v4i32 v9 1");
}

test "gvn: redundant loads, store-to-load forwarding and kills by overlapping stores" {
    const u = try usesAfterGvn(
        \\func m 2
        \\e:
        \\  v2 = load v1 0
        \\  v3 = load v1 0
        \\  store v1 8 v0
        \\  v4 = load v1 8
        \\  v5 = load v1 0
        \\  v6 = cvt.f32.i64 v0
        \\  store.f32 v1 20 v6
        \\  v7 = load v1 0
        \\  v8 = load v1 8
        \\  v9 = load v1 16
        \\  store.f32 v1 4 v6
        \\  v10 = load v1 0
        \\  v11 = load v1 8
        \\  v12 = load.f32 v1 4
        \\  v13 = load.f64 v1 8
        \\  v14 = const 8
        \\  v15 = add v1 v14
        \\  store v15 0 v3
        \\  v16 = load v1 8
        \\  v17 = load v15 0
        \\  v18 = load v1 0
        \\  v40 = cvt.i64.f32 v12
        \\  v41 = cvt.i64.f64 v13
        \\  v19 = xor v2 v3
        \\  v20 = xor v19 v4
        \\  v21 = xor v20 v5
        \\  v22 = xor v21 v7
        \\  v23 = xor v22 v8
        \\  v24 = xor v23 v9
        \\  v25 = xor v24 v10
        \\  v26 = xor v25 v11
        \\  v27 = xor v26 v40
        \\  v28 = xor v27 v41
        \\  v29 = xor v28 v16
        \\  v30 = xor v29 v17
        \\  v31 = xor v30 v18
        \\  ret v31
    );
    defer t.allocator.free(u);
    try t.expectEqual(@as(u32, 0), u[3]); // load v1 0 again
    try t.expectEqual(@as(u32, 0), u[4]); // forwarded from `store v1 8 v0`
    try t.expectEqual(@as(u32, 0), u[5]);
    try t.expectEqual(@as(u32, 0), u[7]); // f32 store at 20..24 is disjoint
    try t.expectEqual(@as(u32, 0), u[8]);
    try t.expectEqual(@as(u32, 1), u[9]); // overlaps the f32 store: reloaded
    try t.expectEqual(@as(u32, 1), u[10]); // f32 store at 4..8 kills 0..8
    try t.expectEqual(@as(u32, 0), u[11]); // ... but not 8..16
    try t.expectEqual(@as(u32, 0), u[12]); // forwarded f32
    try t.expectEqual(@as(u32, 1), u[13]); // f64 at 8 is not the i64 at 8
    try t.expectEqual(@as(u32, 1), u[16]); // other base may alias (it does): reloaded
    try t.expectEqual(@as(u32, 0), u[17]); // forwarded from `store v15 0 v3`
    try t.expectEqual(@as(u32, 1), u[18]);
}

test "gvn: a call clears the memory table; loads reused only along single-predecessor edges" {
    const gpa = t.allocator;
    const src = try std.fmt.allocPrint(gpa,
        \\func c 2
        \\e:
        \\  v2 = load v1 0
        \\  v3 = call 0x{x} v0 v2
        \\  v4 = load v1 0
        \\  store v1 16 v3
        \\  v5 = load v1 8
        \\  cbr lt v0 v4 a b
        \\a:
        \\  v6 = load v1 8
        \\  v7 = load v1 16
        \\  store v1 8 v0
        \\  br j
        \\b:
        \\  v8 = load v1 16
        \\  br j
        \\j:
        \\  v9 = load v1 8
        \\  br k
        \\k:
        \\  v10 = load v1 8
        \\  v11 = xor v4 v5
        \\  v12 = xor v11 v6
        \\  v13 = xor v12 v7
        \\  v14 = xor v13 v8
        \\  v15 = xor v14 v9
        \\  v16 = xor v15 v10
        \\  ret v16
    , .{@intFromPtr(&testgen.callee)});
    defer gpa.free(src);
    const u = try usesAfterGvn(src);
    defer gpa.free(u);
    try t.expectEqual(@as(u32, 2), u[4]); // after the call: reloaded (and it feeds the cbr)
    try t.expectEqual(@as(u32, 0), u[6]); // a's only predecessor is e
    try t.expectEqual(@as(u32, 0), u[7]);
    try t.expectEqual(@as(u32, 0), u[8]);
    try t.expectEqual(@as(u32, 2), u[9]); // j is a join: reloaded (a stored to 8); also replaces v10
    try t.expectEqual(@as(u32, 0), u[10]); // k's only predecessor is j
}

test "gvn: random programs" {
    const gpa = t.allocator;
    const configs = [_]testgen.Config{
        .{},
        .{ .blocks = 12, .ops = 10 },
        .{ .loops = false, .ops = 12 },
        .{ .blocks = 3, .ops = 16, .calls = false },
        .{ .floats = false, .division = false, .blocks = 8 },
    };
    for (configs, 0..) |config, i| try testgen.fuzzPass(gpa, 0x9e7 + i, 60, config, &.{}, run);
    // Twice in a row: the second run finds nothing new.
    try testgen.fuzzPass(gpa, 0x51, 40, .{ .blocks = 8 }, &.{run}, run);
}

/// Random forward-branching programs dense in loads and stores of every scalar and a vector
/// type through two possibly aliasing bases (`v1` and `v13 = v1 + 0/4/8/16`), at overlapping
/// displacements, with a small operand pool so that redundant ops are common.
fn memoryProgram(rng: std.Random, gpa: std.mem.Allocator, out: *std.ArrayList(u8)) !void {
    const n = rng.intRangeAtMost(u32, 1, 5);
    try out.print(gpa, "func m 2\n", .{});
    const Ty = struct { name: []const u8, dst: []const u8 };
    const types = [_]Ty{
        .{ .name = "", .dst = "v2" },       .{ .name = "", .dst = "v3" },
        .{ .name = ".f32", .dst = "v6" },   .{ .name = ".f64", .dst = "v7" },
        .{ .name = ".v2i64", .dst = "v8" },
    };
    for (0..n) |bi| {
        try out.print(gpa, "b{d}:\n", .{bi});
        if (bi == 0) try out.print(gpa, "  v13 = const {d}\n  v13 = add v1 v13\n  v6 = fconst.f32 0.5\n  v7 = fconst.f64 2.5\n  v8 = splat.v2i64 v0\n", .{4 * rng.uintAtMost(u32, 4)});
        for (0..rng.uintAtMost(u32, 14)) |_| {
            const ty = types[rng.uintLessThan(usize, types.len)];
            const base: u32 = if (rng.uintLessThan(u32, 3) == 0) 13 else 1;
            const disp = 4 * rng.uintAtMost(u32, 8);
            const x = rng.intRangeAtMost(u32, 2, 4);
            const y = rng.intRangeAtMost(u32, 0, 4);
            switch (rng.uintLessThan(u32, 9)) {
                0, 1, 2 => try out.print(gpa, "  {s} = load{s} v{d} {d}\n", .{ ty.dst, ty.name, base, disp }),
                3, 4 => try out.print(gpa, "  store{s} v{d} {d} {s}\n", .{ ty.name, base, disp, ty.dst }),
                5 => try out.print(gpa, "  v{d} = {s} v{d} v{d}\n", .{ rng.intRangeAtMost(u32, 2, 5), @tagName(rng.enumValue(enum { add, xor, mul, sub })), x, y }),
                6 => try out.print(gpa, "  v8 = splat.v2i64 v{d}\n", .{x}),
                7 => try out.print(gpa, "  v{d} = extract.v2i64 v8 {d}\n", .{ x, rng.uintAtMost(u32, 1) }),
                else => if (rng.uintLessThan(u32, 3) == 0)
                    try out.print(gpa, "  v{d} = call 0x{x} v{d} v{d}\n", .{ x, @intFromPtr(&testgen.callee), x, y })
                else
                    try out.print(gpa, "  v6 = cvt.f32.i64 v{d}\n  v7 = cvt.f64.f32 v6\n", .{x}),
            }
        }
        if (bi + 1 == n or rng.uintLessThan(u32, 5) == 0) {
            try out.print(gpa, "  v5 = cvt.i64.f64 v7\n  v4 = xor v2 v3\n  v4 = add v4 v5\n  ret v4\n", .{});
        } else {
            const t1 = rng.intRangeLessThan(usize, bi + 1, n);
            const t2 = rng.intRangeLessThan(usize, bi + 1, n);
            try out.print(gpa, "  cbr {s} v{d} v{d} b{d} b{d}\n", .{ @tagName(rng.enumValue(ir.CondCode)), rng.intRangeAtMost(u32, 2, 4), rng.intRangeAtMost(u32, 0, 4), t1, t2 });
        }
    }
}

test "gvn: random memory-heavy programs with aliasing bases" {
    const gpa = t.allocator;
    var prng = std.Random.DefaultPrng.init(0x6e4_31);
    var src: std.ArrayList(u8) = .empty;
    defer src.deinit(gpa);
    var removed: usize = 0;
    for (0..300) |_| {
        src.clearRetainingCapacity();
        try memoryProgram(prng.random(), gpa, &src);
        var r = transform(src.items, run) catch |e| {
            std.debug.print("gvn failure on:\n{s}\n", .{src.items});
            return e;
        };
        defer r[0].deinit();
        removed += @intFromBool(r[1]);
    }
    try t.expect(removed > 100); // the generator does produce redundancy
}

test "gvn: block parameters with equal arguments on every edge are equal" {
    var r = try transform(
        \\func ep 2
        \\e:
        \\  v2 = const 0
        \\  v12 = const 9
        \\  cbr lt v0 v1 a b
        \\a:
        \\  v3 = add v0 v1
        \\  v4 = add v1 v0
        \\  br j v3 v4 v0
        \\b:
        \\  v5 = mul v0 v0
        \\  br j v5 v5 v1
        \\j(v6 v7 v8):
        \\  v9 = sub v6 v7
        \\  v10 = sub v7 v6
        \\  br h v9 v10 v2
        \\h(v13 v14 v15):
        \\  v16 = const 1
        \\  v17 = add v15 v16
        \\  cbr lt v17 v12 h:v13:v14:v17 x
        \\x:
        \\  v18 = sub v13 v14
        \\  v19 = add v18 v8
        \\  ret v19
    , run);
    defer r[0].deinit();
    try t.expect(r[1]);
    try expectIr(&r[0], "v10 = sub v6 v6");
    try expectIr(&r[0], "v9 = sub v6 v6"); // v7 is v6
    try expectIr(&r[0], "br h v9 v9 v2"); // v10 is v9
    try expectIr(&r[0], "v18 = sub v13 v14"); // loop parameters: pessimistic, kept
}
