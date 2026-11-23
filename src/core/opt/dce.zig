//! Dead code elimination on SSA: aggressive mark and sweep. Roots are the operands of `store`,
//! `call` (kept even when its result is unused), conditional branches and `ret`. A live value
//! makes the operands of its defining op live; a live block parameter makes the matching
//! argument of every incoming edge live. Edge arguments are not roots, so values that only feed
//! dead parameters (e.g. a loop-carried accumulator nothing reads) die with them. Every other op
//! that defines a dead vreg is deleted (`load`, `sdiv` and `srem` included: deleting a
//! possibly-faulting op is allowed), and so are dead block parameters with their edge
//! arguments. Linear in ops + edge arguments (one worklist visit per vreg).
const std = @import("std");
const ir = @import("../ir.zig");
const opt = @import("opt.zig");
const cfg = @import("cfg.zig");
const ssa = @import("ssa.zig");

const none = cfg.none;

pub fn run(c: *opt.Ctx) opt.Error!bool {
    const gpa = c.gpa;
    const f = c.f;
    const nv = f.num_vregs;
    const nb: u32 = @intCast(f.blocks.items.len);

    // Definition site of every vreg: (block, op index) or (block, parameter index);
    // `none` for function parameters and never-defined vregs.
    const def_block = try gpa.alloc(u32, nv);
    defer gpa.free(def_block);
    const def_index = try gpa.alloc(u32, nv);
    defer gpa.free(def_index);
    const is_param = try gpa.alloc(bool, nv);
    defer gpa.free(is_param);
    @memset(def_block, none);
    // Incoming edges of every block as (source block, edge index), in CSR form.
    const in_start = try gpa.alloc(u32, nb + 1);
    defer gpa.free(in_start);
    @memset(in_start, 0);
    for (f.blocks.items, 0..) |b, bi| {
        for (b.params, 0..) |p, i| {
            def_block[p] = @intCast(bi);
            def_index[p] = @intCast(i);
            is_param[p] = true;
        }
        for (b.ops.items, 0..) |o, i| if (o.def()) |d| {
            def_block[d] = @intCast(bi);
            def_index[d] = @intCast(i);
            is_param[d] = false;
        };
        const term = b.ops.items[b.ops.items.len - 1];
        for (0..ssa.numSuccs(term)) |i| in_start[cfg.edgeOf(term, @intCast(i)).to + 1] += 1;
    }
    for (1..nb + 1) |i| in_start[i] += in_start[i - 1];
    const InEdge = struct { from: u32, index: u1 };
    const in_edges = try gpa.alloc(InEdge, in_start[nb]);
    defer gpa.free(in_edges);
    {
        const fill = try gpa.dupe(u32, in_start[0..nb]);
        defer gpa.free(fill);
        for (f.blocks.items, 0..) |b, bi| {
            const term = b.ops.items[b.ops.items.len - 1];
            for (0..ssa.numSuccs(term)) |i| {
                const to = cfg.edgeOf(term, @intCast(i)).to;
                in_edges[fill[to]] = .{ .from = @intCast(bi), .index = @intCast(i) };
                fill[to] += 1;
            }
        }
    }

    // Mark.
    const live = try gpa.alloc(bool, nv);
    defer gpa.free(live);
    @memset(live, false);
    var work: std.ArrayList(ir.VReg) = .empty;
    defer work.deinit(gpa);
    const Marker = struct {
        live: []bool,
        work: *std.ArrayList(ir.VReg),
        gpa: std.mem.Allocator,
        fn mark(m: @This(), vs: []const ir.VReg) std.mem.Allocator.Error!void {
            for (vs) |v| if (!m.live[v]) {
                m.live[v] = true;
                try m.work.append(m.gpa, v);
            };
        }
    };
    const m: Marker = .{ .live = live, .work = &work, .gpa = gpa };
    var buf: [ir.max_args]ir.VReg = undefined;
    for (f.blocks.items) |b| for (b.ops.items) |o| switch (o) {
        .store, .call, .cbr, .ret => try m.mark(o.uses(&buf)),
        .cbra => |x| try m.mark(&.{ x.a, x.b }),
        else => {},
    };
    while (work.pop()) |v| {
        const db = def_block[v];
        if (db == none) continue;
        if (is_param[v]) {
            for (in_edges[in_start[db]..in_start[db + 1]]) |e| {
                const args = cfg.edgeOf(cfg.terminator(f, e.from).*, e.index).args;
                try m.mark(args[def_index[v] .. def_index[v] + 1]);
            }
        } else {
            const o = f.blocks.items[db].ops.items[def_index[v]];
            if (o != .call) try m.mark(o.uses(&buf)); // a call's arguments are roots already
        }
    }

    // Sweep ops (calls stay), then dead parameters and their edge arguments.
    var changed = false;
    for (f.blocks.items) |*b| {
        var k: usize = 0;
        for (b.ops.items) |o| {
            if (o.def()) |d| if (!live[d] and o != .call) continue;
            b.ops.items[k] = o;
            k += 1;
        }
        changed = changed or k != b.ops.items.len;
        b.ops.items.len = k;
    }
    var dead_params = false;
    for (f.blocks.items) |b| for (b.params) |p| {
        dead_params = dead_params or !live[p];
    };
    if (!dead_params) return changed;
    for (f.blocks.items) |*b| {
        const term = &b.ops.items[b.ops.items.len - 1];
        const ns = ssa.numSuccs(term.*);
        var args: [2][]const ir.VReg = .{ &.{}, &.{} };
        var edited = false;
        for (0..ns) |i| {
            const e = cfg.edgeOf(term.*, @intCast(i));
            args[i] = try liveOnly(c.irAlloc(), e.args, f.blocks.items[e.to].params, live);
            edited = edited or args[i].len != e.args.len;
        }
        if (edited) ssa.setEdgeArgs(term, args);
    }
    for (f.blocks.items) |*b| b.params = try liveOnly(c.irAlloc(), b.params, b.params, live);
    return true;
}

/// `vs` without the entries whose matching `params` entry is dead (`vs` itself if none is).
fn liveOnly(a: std.mem.Allocator, vs: []const ir.VReg, params: []const ir.VReg, live: []const bool) std.mem.Allocator.Error![]const ir.VReg {
    var k: usize = 0;
    for (params) |p| k += @intFromBool(live[p]);
    if (k == vs.len) return vs;
    const out = try a.alloc(ir.VReg, k);
    k = 0;
    for (vs, params) |v, p| if (live[p]) {
        out[k] = v;
        k += 1;
    };
    return out;
}

// ---- tests -------------------------------------------------------------------------------

const t = std.testing;
const testgen = @import("testgen.zig");

/// Runs `ssa.build` + `run` on `src` (checked and compared with the original like
/// `testgen.checkPass`) and returns the printed result; caller frees.
fn dceText(src: []const u8) ![]u8 {
    try t.expect(try testgen.checkPass(t.allocator, src, run));
    var f = try ir.parse(t.allocator, src);
    defer f.deinit();
    const options: opt.Options = .{ .verify_each = true };
    var stats: opt.Stats = .{};
    var c = try opt.Ctx.init(t.allocator, &f, &options, &stats);
    defer c.deinit();
    try ssa.build(&c);
    _ = try run(&c);
    try c.check("dce");
    try t.expect(!try run(&c)); // a fixpoint after one run
    var out: std.ArrayList(u8) = .empty;
    try f.print(t.allocator, &out);
    return out.toOwnedSlice(t.allocator);
}

/// Number of parameters in the header `<name>(...):` of the printed function `text`.
fn paramCount(text: []const u8, name: []const u8) usize {
    var lines = std.mem.splitScalar(u8, text, '\n');
    while (lines.next()) |l| if (std.mem.startsWith(u8, l, name) and l.len > name.len and l[name.len] == '(') {
        return std.mem.count(u8, l, " ") + 1;
    };
    return 0;
}

fn countTag(text: []const u8, needle: []const u8) usize {
    return std.mem.count(u8, text, needle);
}

test "dce: dead loop-carried accumulator loses its parameter and arguments" {
    const out = try dceText(
        \\func f 1
        \\e:
        \\  v1 = const 0
        \\  v2 = const 0
        \\  v3 = const 1
        \\  br h
        \\h:
        \\  cbr lt v2 v0 b x
        \\b:
        \\  v1 = add v1 v2
        \\  v1 = mul v1 v1
        \\  v2 = add v2 v3
        \\  br h
        \\x:
        \\  ret v2
    );
    defer t.allocator.free(out);
    try t.expectEqual(@as(usize, 0), countTag(out, "mul"));
    try t.expectEqual(@as(usize, 1), countTag(out, "add")); // only the counter
    try t.expectEqual(@as(usize, 1), paramCount(out, "h")); // only the counter is carried
}

test "dce: dead load, sdiv, srem, float ops and chains go; unused call and store stay" {
    const src = try std.fmt.allocPrint(t.allocator,
        \\func f 2
        \\e:
        \\  v2 = load v1 0
        \\  v3 = const 7
        \\  v4 = sdiv v0 v3
        \\  v5 = srem v4 v2
        \\  v6 = add v5 v5
        \\  v7 = mul v6 v6
        \\  v8 = load v1 8
        \\  v9 = xor v8 v0
        \\  v10 = call 0x{x} v0 v9
        \\  v11 = const 3
        \\  store v1 16 v11
        \\  v12 = fconst.f64 1.5
        \\  v13 = cvt.i64.f64 v12
        \\  ret v0
    , .{@intFromPtr(&testgen.callee)});
    defer t.allocator.free(src);
    const out = try dceText(src);
    defer t.allocator.free(out);
    for ([_][]const u8{ "sdiv", "srem", "mul", "add", "cvt", "fconst" }) |gone| try t.expectEqual(@as(usize, 0), countTag(out, gone));
    try t.expectEqual(@as(usize, 1), countTag(out, "load")); // the call's argument chain stays
    try t.expectEqual(@as(usize, 1), countTag(out, "xor"));
    try t.expectEqual(@as(usize, 1), countTag(out, "call"));
    try t.expectEqual(@as(usize, 1), countTag(out, "store"));
}

test "dce: values only used by dead parameters, through several joins" {
    const out = try dceText(
        \\func f 2
        \\e:
        \\  v2 = const 5
        \\  v3 = mul v0 v2
        \\  cbr lt v0 v1 a:v3 b
        \\a(v4):
        \\  v5 = add v4 v3
        \\  br m v5 v4
        \\b:
        \\  br m v3 v2
        \\m(v6 v7):
        \\  v8 = sub v6 v7
        \\  cbr eq v1 v2 m:v8:v6 x
        \\x:
        \\  ret v1
    );
    defer t.allocator.free(out);
    for ([_][]const u8{ "mul", "add", "sub", "(" }) |gone| try t.expectEqual(@as(usize, 0), countTag(out, gone));
}

test "dce: live parameter keeps its arguments; branch operands are roots" {
    const out = try dceText(
        \\func f 2
        \\e:
        \\  v2 = const 0
        \\  v3 = const 1
        \\  v9 = const 9
        \\  br h v2 v3 v9
        \\h(v4 v5 v6):
        \\  v7 = add v4 v5
        \\  v8 = add v5 v3
        \\  v10 = mul v6 v6
        \\  cbr lt v8 v0 h:v7:v8:v10 x
        \\x:
        \\  ret v7
    );
    defer t.allocator.free(out);
    try t.expectEqual(@as(usize, 0), countTag(out, "mul"));
    try t.expectEqual(@as(usize, 2), countTag(out, "add"));
}

test "dce: random programs (several configurations)" {
    const gpa = t.allocator;
    try testgen.fuzzPass(gpa, 1, 120, .{}, &.{}, run);
    try testgen.fuzzPass(gpa, 2, 80, .{ .blocks = 10, .ops = 10 }, &.{}, run);
    try testgen.fuzzPass(gpa, 3, 80, .{ .loops = false }, &.{}, run);
    try testgen.fuzzPass(gpa, 4, 80, .{ .calls = false, .memory = false }, &.{}, run);
    try testgen.fuzzPass(gpa, 5, 60, .{ .floats = false, .division = false, .ops = 12 }, &.{run}, run);
}
