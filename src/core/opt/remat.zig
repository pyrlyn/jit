//! Constants to the entry block: every i64 `const` of a reachable block moves to the top of
//! block 0, one op per distinct value (duplicates are renamed to the first one). Block 0
//! dominates every reachable block, so the moved definitions dominate all their uses, and the
//! IR Compilers fold a single-definition block-0 constant into immediates function-wide
//! (emitting nothing for it when every use folds).
//!
//! Float and vector constants stay where they are (decision, see the WP report): the IR
//! Compilers cannot fold them into instructions, so moving them to block 0 only stretches FP
//! register live ranges over the whole function (more spills, and caller-saved FP registers
//! across calls), while gvn already merges dominated duplicates and licm already hoists
//! loop-invariant ones out of loops. Constants of unreachable blocks are left alone (nothing
//! reachable can use them; simplifycfg deletes such blocks).
const std = @import("std");
const ir = @import("../ir.zig");
const opt = @import("opt.zig");
const cfg = @import("cfg.zig");

/// Returns whether any constant moved or was merged. Linear in ops + blocks (one hash lookup
/// per constant; uses are renamed in one pass only when duplicates exist).
pub fn run(c: *opt.Ctx) opt.Error!bool {
    const gpa = c.gpa;
    const f = c.f;
    const reach = try reachable(gpa, f);
    defer gpa.free(reach);

    var first: std.AutoHashMapUnmanaged(i64, ir.VReg) = .empty;
    defer first.deinit(gpa);
    var hoisted: std.ArrayList(ir.Op) = .empty;
    defer hoisted.deinit(gpa);
    // Duplicate -> first definition; allocated on the first duplicate.
    var map: ?[]ir.VReg = null;
    defer if (map) |m| gpa.free(m);
    var changed = false;

    for (f.blocks.items, 0..) |*b, bi| {
        if (!reach[bi]) continue;
        const ops = b.ops.items;
        var keep: usize = 0;
        for (ops) |o| {
            if (o != .@"const") {
                ops[keep] = o;
                keep += 1;
                continue;
            }
            // A constant moves unless it is already part of block 0's leading constants.
            if (bi != 0 or keep != 0) changed = true;
            const gop = try first.getOrPut(gpa, o.@"const".imm);
            if (!gop.found_existing) {
                gop.value_ptr.* = o.@"const".dst;
                try hoisted.append(gpa, o);
                continue;
            }
            const m = map orelse blk: {
                const m = try gpa.alloc(ir.VReg, f.num_vregs);
                for (m, 0..) |*v, i| v.* = @intCast(i);
                map = m;
                break :blk m;
            };
            m[o.@"const".dst] = gop.value_ptr.*;
            changed = true;
        }
        b.ops.items.len = keep;
    }
    // Block 0 lost exactly its constants above; put the hoisted ones back in front.
    try f.blocks.items[0].ops.insertSlice(c.irAlloc(), 0, hoisted.items);
    if (map) |m| try opt.renameUses(c, m);
    return changed;
}

/// Blocks reachable from block 0 (caller frees).
fn reachable(gpa: std.mem.Allocator, f: *const ir.Func) std.mem.Allocator.Error![]bool {
    const n = f.blocks.items.len;
    const seen = try gpa.alloc(bool, n);
    errdefer gpa.free(seen);
    @memset(seen, false);
    var stack: std.ArrayList(u32) = .empty;
    defer stack.deinit(gpa);
    seen[0] = true;
    try stack.append(gpa, 0);
    var buf: [2]u32 = undefined;
    while (stack.pop()) |b| for (cfg.succs(f, b, &buf)) |s| if (!seen[s]) {
        seen[s] = true;
        try stack.append(gpa, s);
    };
    return seen;
}

// ---- tests -------------------------------------------------------------------------------

const t = std.testing;
const testgen = @import("testgen.zig");
const ssa = @import("ssa.zig");

/// Parses `src`, builds SSA (`src` is taken as strict SSA as it is when it has block
/// parameters), runs remat (checked; `expect_changed` when not null), compares with the
/// original on `testgen.default_inputs` and returns the result (caller deinits).
fn remat(src: []const u8, expect_changed: ?bool) !ir.Func {
    const gpa = t.allocator;
    var ref = try ir.parse(gpa, src);
    defer ref.deinit();
    var f = try ir.parse(gpa, src);
    errdefer f.deinit();
    const options: opt.Options = .{ .verify_each = true };
    var stats: opt.Stats = .{};
    var c = try opt.Ctx.init(gpa, &f, &options, &stats);
    defer c.deinit();
    if (f.hasBlockParams()) c.in_ssa = true else try ssa.build(&c);
    try c.check("ssa.build");
    const changed = try run(&c);
    if (expect_changed) |e| try t.expectEqual(e, changed);
    try c.check("remat");
    try testgen.expectSame(&ref, &f, &testgen.default_inputs, 20_000);
    try t.expect(!try run(&c)); // idempotent
    try expectHoisted(&f);
    return f;
}

/// Every `const` of a reachable block is in block 0's leading run, one per value.
fn expectHoisted(f: *const ir.Func) !void {
    const reach = try reachable(t.allocator, f);
    defer t.allocator.free(reach);
    var seen: std.AutoHashMapUnmanaged(i64, void) = .empty;
    defer seen.deinit(t.allocator);
    for (f.blocks.items, reach, 0..) |b, r, bi| {
        if (!r) continue;
        var leading = bi == 0;
        for (b.ops.items) |o| {
            if (o != .@"const") {
                leading = false;
                continue;
            }
            try t.expect(leading);
            try t.expect(try seen.fetchPut(t.allocator, o.@"const".imm, {}) == null);
        }
    }
}

/// Number of `const` ops (with value `imm`, or any) in block `name`.
fn countConsts(f: *const ir.Func, name: []const u8, imm: ?i64) usize {
    var n: usize = 0;
    for (f.blocks.items) |b| if (std.mem.eql(u8, b.name, name)) for (b.ops.items) |o| switch (o) {
        .@"const" => |x| n += @intFromBool(imm == null or x.imm == imm.?),
        else => {},
    };
    return n;
}

test "remat: duplicates across blocks merge into one entry constant" {
    var f = try remat(
        \\func d 2
        \\e:
        \\  v2 = add v0 v1
        \\  v3 = const 5
        \\  v4 = add v2 v3
        \\  cbr lt v0 v1 l r
        \\l:
        \\  v5 = const 5
        \\  v6 = mul v4 v5
        \\  v7 = const -1
        \\  v8 = add v6 v7
        \\  br j
        \\r:
        \\  v9 = const 5
        \\  v10 = sub v4 v9
        \\  v7 = const 5
        \\  v8 = xor v10 v7
        \\  br j
        \\j:
        \\  ret v8
    , true);
    defer f.deinit();
    const e = f.blocks.items[0].ops.items;
    try t.expectEqual(@as(i64, 5), e[0].@"const".imm);
    try t.expectEqual(@as(i64, -1), e[1].@"const".imm);
    try t.expect(e[2] == .arith);
    try t.expectEqual(@as(usize, 2), countConsts(&f, "e", null));
    for ([_][]const u8{ "l", "r", "j" }) |b| try t.expectEqual(@as(usize, 0), countConsts(&f, b, null));
}

test "remat: constants used only inside a loop move to the entry block" {
    var f = try remat(
        \\func s 2
        \\e:
        \\  v2 = const 0
        \\  v3 = const 0
        \\  br h
        \\h:
        \\  cbr lt v2 v0 body done
        \\body:
        \\  v4 = const 3
        \\  v5 = mul v2 v4
        \\  v6 = const 1000000007
        \\  v5 = srem v5 v6
        \\  v3 = add v3 v5
        \\  v7 = const 1
        \\  v2 = add v2 v7
        \\  br h
        \\done:
        \\  v8 = const 3
        \\  v9 = add v3 v8
        \\  ret v9
    , true);
    defer f.deinit();
    try t.expectEqual(@as(usize, 0), countConsts(&f, "body", null));
    try t.expectEqual(@as(usize, 0), countConsts(&f, "done", null));
    for ([_]i64{ 0, 3, 1000000007, 1 }) |k| try t.expectEqual(@as(usize, 1), countConsts(&f, "e", k));
}

test "remat: already hoisted entry constants are left as they are" {
    var f = try remat(
        \\func k 1
        \\e:
        \\  v1 = const 7
        \\  v2 = const 9
        \\  v3 = add v0 v1
        \\  cbr lt v3 v2 a b
        \\a:
        \\  ret v3
        \\b:
        \\  v4 = mul v3 v2
        \\  ret v4
    , false);
    f.deinit();
}

test "remat: an entry constant after other ops moves to the top, floats stay" {
    var f = try remat(
        \\func m 1
        \\e:
        \\  v1 = fconst.f64 2.5
        \\  v2 = cvt.f64.i64 v0
        \\  v3 = mul.f64 v2 v1
        \\  v4 = cvt.i64.f64 v3
        \\  v5 = const 4
        \\  v6 = add v4 v5
        \\  cbr gt v6 v5 a b
        \\a:
        \\  v7 = fconst.f64 2.5
        \\  v8 = add.f64 v3 v7
        \\  v9 = cvt.i64.f64 v8
        \\  ret v9
        \\b:
        \\  ret v6
    , true);
    defer f.deinit();
    const e = f.blocks.items[0].ops.items;
    try t.expectEqual(@as(i64, 4), e[0].@"const".imm);
    try t.expect(e[1] == .fconst);
    try t.expect(f.blocks.items[1].ops.items[0] == .fconst);
}

test "remat: constants of unreachable blocks and edge arguments" {
    var f = try remat(
        \\func u 1
        \\e:
        \\  v1 = const 1
        \\  v7 = const 0
        \\  br h v1 v7
        \\h(v2 v8):
        \\  v3 = const 1
        \\  v4 = add v2 v3
        \\  v5 = const 10
        \\  v9 = const 10
        \\  cbr lt v4 v5 h:v4:v9 x
        \\dead:
        \\  v6 = const 1
        \\  br x
        \\x:
        \\  ret v4
    , true);
    defer f.deinit();
    try t.expectEqual(@as(usize, 3), countConsts(&f, "e", null));
    try t.expectEqual(@as(usize, 0), countConsts(&f, "h", null));
    try t.expectEqual(@as(usize, 1), countConsts(&f, "dead", 1));
}

test "remat: random programs" {
    // `remat` checks equivalence, the hoisted shape and idempotence of every program.
    const gpa = t.allocator;
    const configs = [_]testgen.Config{
        .{},
        .{ .loops = false },
        .{ .blocks = 12, .ops = 10 },
        .{ .floats = false, .calls = false, .blocks = 3, .ops = 16 },
    };
    var prng = std.Random.DefaultPrng.init(0x7e3a7);
    var src: std.ArrayList(u8) = .empty;
    defer src.deinit(gpa);
    for (0..96) |i| {
        src.clearRetainingCapacity();
        try testgen.program(prng.random(), gpa, configs[i % configs.len], &src);
        var f = remat(src.items, null) catch |e| {
            std.debug.print("remat failure on:\n{s}\n", .{src.items});
            return e;
        };
        f.deinit();
    }
}
