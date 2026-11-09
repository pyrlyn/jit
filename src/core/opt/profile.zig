//! Profiling: tier-0 instrumentation (block-entry and taken-edge counters, an optional tier-up
//! hook) that produces the `opt.Profile` `optimize` uses for inlining and block layout. The
//! instrumentation is plain IR, so every IR Compiler and the interpreter run it unchanged.
const std = @import("std");
const ir = @import("../ir.zig");
const opt = @import("opt.zig");

pub const Profile = opt.Profile;

/// Counter storage for `instrument`, indexed by the block numbers of the instrumented
/// function (normally `f.blocks.items.len` long each). Blocks at or past the end of a slice
/// are not counted, so `blocks.len == 1, taken.len == 0` counts only entries of block 0.
pub const Counters = struct {
    /// Times each block started executing (block 0: calls, unless a branch targets it).
    blocks: []u64,
    /// Times the `then` edge of the `cbr`/`cbra` ending each block was taken.
    taken: []u64,
};

/// Tier-up hook: the entry block calls `addr(arg0, arg1)` (C ABI, i64 result ignored) when
/// `blocks[0]` reaches `at` (`at >= 1`).
pub const Hook = struct { at: u64, addr: usize, arg0: i64, arg1: i64 };

/// Returns a new function (owned by the caller; `f` is only read) that computes what `f`
/// computes and also counts block entries in `counters.blocks[b]` and taken `then` edges of
/// `cbr`/`cbra` in `counters.taken[b]`, indexed by `f`'s block numbers. Blocks keep their
/// numbers and new blocks are appended: the rest of block 0 when there is a hook, and one
/// block per counted `then` edge (it counts, then branches on with the edge's arguments).
///
/// A count is `v = load base d; one = const 1; v1 = add v one; store base d v1` on fresh
/// vregs, where `base = const <lowest counter slice address>` starts block 0 (which runs before
/// every other block) and `d` is the counter's offset from it; a counter more than an i32 away
/// is addressed by its own `const` instead. One base register instead of a 64-bit constant per
/// count keeps tier-0 loops about twice as fast. Increments are not atomic: concurrent calls may lose counts
/// but never corrupt anything else. With `hook`, the call that stores exactly `at` into
/// `blocks[0]` calls the hook: once in a single thread; under concurrency at least once
/// when the count passes `at` (racing calls that both store `at` both call it).
pub fn instrument(gpa: std.mem.Allocator, f: *const ir.Func, counters: Counters, hook: ?Hook) !ir.Func {
    std.debug.assert(hook == null or counters.blocks.len != 0);
    var g = try clone(gpa, f);
    errdefer g.deinit();
    const a = g.alloc();
    const base: Base = .{ .vreg = try fresh(&g, 1), .addr = baseAddress(counters) };
    for (0..f.blocks.items.len) |bi| {
        const b: u32 = @intCast(bi);
        const body = g.blocks.items[b].ops;
        var ops: std.ArrayList(ir.Op) = .empty;
        // Block that ends with the original terminator (block 0's rest when hooked).
        var last = b;
        if (b == 0) try ops.append(a, .{ .@"const" = .{ .dst = base.vreg, .imm = @bitCast(@as(u64, base.addr)) } });
        if (b < counters.blocks.len) {
            const v1 = try count(&g, &ops, base, &counters.blocks[b]);
            if (b == 0) if (hook) |h| {
                last = try addBlock(&g, "prof.rest");
                const call = try addBlock(&g, "prof.hook");
                const at = try fresh(&g, 1);
                try ops.appendSlice(a, &.{
                    .{ .@"const" = .{ .dst = at, .imm = @bitCast(h.at) } },
                    .{ .cbr = .{ .cond = .eq, .a = v1, .b = at, .then = call, .@"else" = last } },
                });
                const x = try fresh(&g, 3);
                const args = try a.dupe(ir.VReg, &.{ x, x + 1 });
                try g.blocks.items[call].ops.appendSlice(a, &.{
                    .{ .@"const" = .{ .dst = x, .imm = h.arg0 } },
                    .{ .@"const" = .{ .dst = x + 1, .imm = h.arg1 } },
                    .{ .call = .{ .dst = x + 2, .addr = h.addr, .args = args } },
                    .{ .br = last },
                });
            };
        }
        if (last == b) {
            try ops.appendSlice(a, body.items);
            g.blocks.items[b].ops = ops;
        } else {
            g.blocks.items[b].ops = ops;
            g.blocks.items[last].ops = body;
        }
        if (b < counters.taken.len) try countThen(&g, last, base, &counters.taken[b]);
    }
    return g;
}

/// The profile `instrument`ed code records into `counters` (a live view, not a copy: tier-0
/// code may still be updating it).
pub fn snapshot(counters: Counters) opt.Profile {
    return .{ .blocks = counters.blocks, .taken = counters.taken };
}

/// Deep copy of `f` (names, ops, argument and parameter slices) in a new arena owned by the
/// caller; `f` is only read.
pub fn clone(gpa: std.mem.Allocator, f: *const ir.Func) !ir.Func {
    var g = ir.Func.init(gpa);
    errdefer g.deinit();
    const a = g.alloc();
    g.name = try a.dupe(u8, f.name);
    g.params = f.params;
    g.num_vregs = f.num_vregs;
    try g.blocks.ensureTotalCapacity(a, f.blocks.items.len);
    for (f.blocks.items) |b| {
        var ops = try std.ArrayList(ir.Op).initCapacity(a, b.ops.items.len);
        for (b.ops.items) |op| ops.appendAssumeCapacity(switch (op) {
            .call => |x| .{ .call = .{ .dst = x.dst, .addr = x.addr, .args = try a.dupe(ir.VReg, x.args) } },
            .bra => |x| .{ .bra = .{ .target = x.target, .args = try a.dupe(ir.VReg, x.args) } },
            .cbra => |x| blk: {
                var y = x;
                y.then_args = try a.dupe(ir.VReg, x.then_args);
                y.else_args = try a.dupe(ir.VReg, x.else_args);
                break :blk .{ .cbra = y };
            },
            else => op,
        });
        g.blocks.appendAssumeCapacity(.{ .name = try a.dupe(u8, b.name), .ops = ops, .params = try a.dupe(ir.VReg, b.params) });
    }
    return g;
}

/// `n` new consecutive vregs.
fn fresh(g: *ir.Func, n: u32) error{BadVReg}!ir.VReg {
    const first = g.num_vregs;
    // maxInt(VReg) stays reserved (as in `ir.Func.emit`).
    if (first >= std.math.maxInt(ir.VReg) - n) return error.BadVReg;
    g.num_vregs += n;
    return first;
}

/// Appends a block named `<kind>.<index>` (unique, so printed IR parses again).
fn addBlock(g: *ir.Func, kind: []const u8) !u32 {
    var buf: [32]u8 = undefined;
    return g.addBlock(std.fmt.bufPrint(&buf, "{s}.{d}", .{ kind, g.blocks.items.len }) catch unreachable);
}

/// Appends the increment of `ctr` to `ops`; returns the vreg holding the new count.
/// Counter base register (defined at the start of block 0) and the address it holds.
const Base = struct { vreg: ir.VReg, addr: usize };

/// Lowest start of the two counter slices (an empty slice does not count), so typical
/// offsets are small and positive (cheap addressing modes).
fn baseAddress(counters: Counters) usize {
    var lo: usize = std.math.maxInt(usize);
    if (counters.blocks.len != 0) lo = @intFromPtr(counters.blocks.ptr);
    if (counters.taken.len != 0) lo = @min(lo, @intFromPtr(counters.taken.ptr));
    return if (lo == std.math.maxInt(usize)) 0 else lo;
}

/// Appends an increment of `ctr` to `ops`; returns the vreg holding the new count.
fn count(g: *ir.Func, ops: *std.ArrayList(ir.Op), base: Base, ctr: *u64) !ir.VReg {
    const v = try fresh(g, 4);
    const off = @as(i128, @intFromPtr(ctr)) - base.addr;
    const near = std.math.cast(i32, off);
    // Far counter: its own address constant in `v` (otherwise unused).
    if (near == null) try ops.append(g.alloc(), .{ .@"const" = .{ .dst = v, .imm = @bitCast(@as(u64, @intFromPtr(ctr))) } });
    const ptr = if (near == null) v else base.vreg;
    const disp = near orelse 0;
    try ops.appendSlice(g.alloc(), &.{
        .{ .load = .{ .dst = v + 1, .base = ptr, .disp = disp } },
        .{ .@"const" = .{ .dst = v + 2, .imm = 1 } },
        .{ .arith = .{ .op = .add, .dst = v + 3, .a = v + 1, .b = v + 2 } },
        .{ .store = .{ .base = ptr, .disp = disp, .src = v + 3 } },
    });
    return v + 3;
}

/// Routes the `then` edge of `blk`'s `cbr`/`cbra` through a new block that counts it in `ctr`
/// and branches on with the edge's arguments (the counting writes fresh vregs only, so the
/// arguments read the same values there).
fn countThen(g: *ir.Func, blk: u32, base: Base, ctr: *u64) !void {
    switch (g.blocks.items[blk].ops.getLast()) {
        .cbr, .cbra => {},
        else => return,
    }
    const e = try addBlock(g, "prof.then");
    var ops: std.ArrayList(ir.Op) = .empty;
    _ = try count(g, &ops, base, ctr);
    const term = &g.blocks.items[blk].ops.items[g.blocks.items[blk].ops.items.len - 1];
    switch (term.*) {
        .cbr => |*x| {
            try ops.append(g.alloc(), .{ .br = x.then });
            x.then = e;
        },
        .cbra => |x| {
            try ops.append(g.alloc(), if (x.then_args.len == 0) .{ .br = x.then } else .{ .bra = .{ .target = x.then, .args = x.then_args } });
            term.* = if (x.else_args.len == 0)
                .{ .cbr = .{ .cond = x.cond, .a = x.a, .b = x.b, .then = e, .@"else" = x.@"else" } }
            else
                .{ .cbra = .{ .cond = x.cond, .a = x.a, .b = x.b, .then = e, .@"else" = x.@"else", .then_args = &.{}, .else_args = x.else_args } };
        },
        else => unreachable,
    }
    g.blocks.items[e].ops = ops;
}

// ---- tests ------------------------------------------------------------------------------
const t = std.testing;

/// Interprets `f` and its instrumented version on `args`; checks they agree, returns the result.
fn runBoth(f: *const ir.Func, g: *const ir.Func, args: []const i64) !i64 {
    const want = try ir.interpret(f, args, 1 << 20);
    try t.expectEqual(want, try ir.interpret(g, args, 1 << 20));
    return want;
}

fn checkValid(g: *const ir.Func) !void {
    try g.verify();
    t.allocator.free(try g.typeCheck(t.allocator));
    // Unique block names: the printed function parses again.
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(t.allocator);
    try g.print(t.allocator, &out);
    var h = try ir.parse(t.allocator, out.items);
    h.deinit();
}

test "profile: loop block counts and taken edges are exact, results unchanged" {
    var f = try ir.parse(t.allocator, ir.fib_src); // entry, loop, body, done
    defer f.deinit();
    var blocks: [4]u64 = @splat(0);
    var taken: [4]u64 = @splat(0);
    const ctr: Counters = .{ .blocks = &blocks, .taken = &taken };
    var g = try instrument(t.allocator, &f, ctr, null);
    defer g.deinit();
    try checkValid(&g);
    try t.expectEqual(f.blocks.items.len + 1, g.blocks.items.len); // one counted then-edge
    var sum: u64 = 0;
    for ([_]i64{ 0, 1, 7, 30 }) |n| {
        _ = try runBoth(&f, &g, &.{n});
        sum += @intCast(n);
    }
    try t.expectEqualSlices(u64, &.{ 4, sum + 4, sum, 4 }, &blocks);
    try t.expectEqualSlices(u64, &.{ 0, sum, 0, 0 }, &taken);
    const p = snapshot(ctr);
    try t.expectEqual(sum, p.taken[1]);
    try t.expectEqual(@as(u64, 4), p.blocks[0]);
}

test "profile: both cbr directions and then-edges carrying block arguments" {
    // h's then-edge carries two arguments; its else-edge one (to a parameter block).
    var f = try ir.parse(t.allocator,
        \\func tri 2
        \\e:
        \\  v2 = const 0
        \\  br h v2 v0
        \\h(v3 v4):
        \\  cbr gt v4 v2 body:v3:v4 out:v3
        \\body(v5 v6):
        \\  v7 = add v5 v6
        \\  v8 = const 1
        \\  v9 = sub v6 v8
        \\  cbr lt v1 v2 neg pos
        \\neg:
        \\  v7 = sub v7 v1
        \\  br h v7 v9
        \\pos:
        \\  br h v7 v9
        \\out(v10):
        \\  ret v10
    );
    defer f.deinit();
    var blocks: [6]u64 = @splat(0);
    var taken: [6]u64 = @splat(0);
    var g = try instrument(t.allocator, &f, .{ .blocks = &blocks, .taken = &taken }, null);
    defer g.deinit();
    try checkValid(&g);
    try t.expectEqual(@as(i64, 15), try runBoth(&f, &g, &.{ 5, 1 }));
    try t.expectEqual(@as(i64, 6 + 3 * 2), try runBoth(&f, &g, &.{ 3, -2 }));
    try t.expectEqual(@as(i64, 0), try runBoth(&f, &g, &.{ -4, 0 }));
    // h: 6 + 4 + 1 entries; then taken 5 + 3 times; body splits 3 times to neg, 5 to pos.
    try t.expectEqualSlices(u64, &.{ 3, 11, 8, 3, 5, 3 }, &blocks);
    try t.expectEqualSlices(u64, &.{ 0, 8, 3, 0, 0, 0 }, &taken);
}

const HookLog = struct {
    var calls: u32 = 0;
    var args: [2]i64 = .{ 0, 0 };
    var count_seen: u64 = 0;
    var counter: *const u64 = undefined;
    fn hook(a0: i64, a1: i64) callconv(.c) i64 {
        calls += 1;
        args = .{ a0, a1 };
        count_seen = counter.*;
        return -1; // ignored
    }
};

test "profile: hook runs exactly once, when the entry count reaches `at`" {
    var f = try ir.parse(t.allocator, ir.fib_src);
    defer f.deinit();
    var blocks: [4]u64 = @splat(0);
    var taken: [4]u64 = @splat(0);
    HookLog.calls = 0;
    HookLog.counter = &blocks[0];
    const h: Hook = .{ .at = 3, .addr = @intFromPtr(&HookLog.hook), .arg0 = -7, .arg1 = 1 << 40 };
    var g = try instrument(t.allocator, &f, .{ .blocks = &blocks, .taken = &taken }, h);
    defer g.deinit();
    try checkValid(&g);
    for (0..6) |i| {
        try t.expectEqual(@as(u32, @intFromBool(i > 2)), HookLog.calls);
        _ = try runBoth(&f, &g, &.{@intCast(i)});
    }
    try t.expectEqual(@as(u32, 1), HookLog.calls);
    try t.expectEqual(@as(u64, 3), HookLog.count_seen);
    try t.expectEqual([2]i64{ -7, 1 << 40 }, HookLog.args);
    try t.expectEqualSlices(u64, &.{ 6, 15 + 6, 15, 6 }, &blocks);
    try t.expectEqual(@as(u64, 15), taken[1]);
}

test "profile: short counter slices count only the covered blocks; entry loops count entries" {
    // Block 0 is a loop header: its count includes the back-edge entries.
    var f = try ir.parse(t.allocator,
        \\func down 1
        \\top:
        \\  v1 = const 1
        \\  v0 = sub v0 v1
        \\  v2 = add v2 v1
        \\  cbr gt v0 v3 top end
        \\end:
        \\  ret v2
    );
    defer f.deinit();
    var blocks: [1]u64 = @splat(0);
    var g = try instrument(t.allocator, &f, .{ .blocks = &blocks, .taken = &.{} }, null);
    defer g.deinit();
    try checkValid(&g);
    try t.expectEqual(f.blocks.items.len, g.blocks.items.len);
    try t.expectEqual(@as(i64, 5), try runBoth(&f, &g, &.{5}));
    try t.expectEqual(@as(i64, 1), try runBoth(&f, &g, &.{-3}));
    try t.expectEqual(@as(u64, 6), blocks[0]);
}
