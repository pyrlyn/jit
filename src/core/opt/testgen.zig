//! Random valid IR for differential tests of the optimizer and the IR Compilers, and a
//! harness that runs a function before and after a transformation through `ir.interpret`
//! (result, memory and the sequence of calls must match).
//!
//! Generated functions take `v0` (data) and `v1` (address of a 16-byte aligned buffer of
//! `mem_slots` u64, see `Run`). Vregs: v2..v7 i64 pool, v8..v9 f64 pool (defined in the last
//! block, so they read +0.0 until it runs), v10 step counter (incremented at the top of every
//! block), v11/v12 scratch. Loops are bounded by the interpreter's step limit: programs that
//! hit it in the reference run are skipped by `expectSame`.
const std = @import("std");
const ir = @import("../ir.zig");
const opt = @import("opt.zig");
const ssa = @import("ssa.zig");

pub const mem_slots = 8;

pub const Config = struct {
    /// Blocks (1..blocks) and ops per block (0..ops, besides the counter and terminator).
    blocks: u32 = 6,
    ops: u32 = 6,
    floats: bool = true,
    /// `load`/`store` (i64 and f64) at `v1 + 8*k`, k < mem_slots.
    memory: bool = true,
    /// `call` of `callee` (deterministic; records every call, see `Run`).
    calls: bool = true,
    /// `sdiv`/`srem` with divisors in 1..256 (never undefined).
    division: bool = true,
    /// Branches to any block (else only forward: no cycles).
    loops: bool = true,
};

/// Writes a random function `r` in the text format.
pub fn program(rng: std.Random, gpa: std.mem.Allocator, config: Config, out: *std.ArrayList(u8)) !void {
    const n = rng.intRangeAtMost(u32, 1, @max(config.blocks, 1));
    try out.print(gpa, "func r 2\n", .{});
    for (0..n) |bi| {
        const b: u32 = @intCast(bi);
        try out.print(gpa, "b{d}:\n  v11 = const 1\n  v10 = add v10 v11\n", .{b});
        if (config.floats and b == n - 1) try out.print(gpa, "  v8 = fconst.f64 1.25\n  v9 = fconst.f64 -3.5\n", .{});
        for (0..rng.uintAtMost(u32, config.ops)) |_| try op(rng, gpa, config, out);
        const target = struct {
            fn pick(r: std.Random, cf: Config, from: u32, count: u32) ?u32 {
                if (cf.loops) return r.uintLessThan(u32, count);
                if (from + 1 >= count) return null;
                return r.intRangeLessThan(u32, from + 1, count);
            }
        }.pick;
        const t1 = target(rng, config, b, n);
        const t2 = target(rng, config, b, n);
        if (t1 == null or t2 == null or rng.uintLessThan(u32, 5) == 0) {
            try out.print(gpa, "  ret v{d}\n", .{pool(rng)});
        } else switch (rng.uintLessThan(u32, 3)) {
            0 => try out.print(gpa, "  br b{d}\n", .{t1.?}),
            1 => try out.print(gpa, "  v11 = const {d}\n  cbr lt v10 v11 b{d} b{d}\n", .{ rng.intRangeAtMost(i64, 1, 12), t1.?, t2.? }),
            else => try out.print(gpa, "  cbr {s} v{d} v{d} b{d} b{d}\n", .{ @tagName(rng.enumValue(ir.CondCode)), pool(rng), pool(rng), t1.?, t2.? }),
        }
    }
}

/// An i64 vreg to read: a parameter or the pool.
fn pool(rng: std.Random) u32 {
    return rng.uintAtMost(u32, 7);
}

fn op(rng: std.Random, gpa: std.mem.Allocator, config: Config, out: *std.ArrayList(u8)) !void {
    const d = rng.intRangeAtMost(u32, 2, 7);
    const a = pool(rng);
    const b = pool(rng);
    switch (rng.uintLessThan(u32, 12)) {
        0 => try out.print(gpa, "  v{d} = const {d}\n", .{ d, switch (rng.uintLessThan(u32, 4)) {
            0 => rng.intRangeAtMost(i64, -4, 4),
            1 => @as(i64, 1) << rng.intRangeAtMost(u6, 0, 63),
            2 => rng.int(i64),
            else => rng.intRangeAtMost(i64, -70, 70),
        } }),
        1 => try out.print(gpa, "  v{d} = copy v{d}\n", .{ d, a }),
        2, 3, 4 => try out.print(gpa, "  v{d} = {s} v{d} v{d}\n", .{ d, @tagName(rng.enumValue(enum { add, sub, mul, @"and", @"or", xor, shl, shr, sar })), a, b }),
        5 => if (config.division) try out.print(
            gpa,
            "  v12 = const 255\n  v12 = and v{d} v12\n  v11 = const 1\n  v12 = add v12 v11\n  v{d} = {s} v{d} v12\n",
            .{ b, d, if (rng.boolean()) "sdiv" else "srem", a },
        ),
        6, 7 => if (config.memory) {
            const disp = 8 * rng.uintLessThan(u32, mem_slots);
            if (rng.boolean()) {
                try out.print(gpa, "  v{d} = load v1 {d}\n", .{ d, disp });
            } else try out.print(gpa, "  store v1 {d} v{d}\n", .{ disp, a });
        },
        8 => if (config.calls) try out.print(gpa, "  v{d} = call 0x{x} v{d} v{d}\n", .{ d, @intFromPtr(&callee), a, b }),
        9 => if (config.floats) {
            const fd = rng.intRangeAtMost(u32, 8, 9);
            switch (rng.uintLessThan(u32, 4)) {
                0 => try out.print(gpa, "  v{d} = cvt.f64.i64 v{d}\n", .{ fd, a }),
                1 => try out.print(gpa, "  v{d} = {s}.f64 v8 v9\n", .{ fd, @tagName(rng.enumValue(enum { add, sub, mul, div, min, max })) }),
                2 => try out.print(gpa, "  v{d} = fconst.f64 {d}\n", .{ fd, @as(f64, @floatFromInt(rng.intRangeAtMost(i32, -8, 8))) / 4 }),
                else => if (config.memory) {
                    const disp = 8 * rng.uintLessThan(u32, mem_slots);
                    if (rng.boolean()) {
                        try out.print(gpa, "  v{d} = load.f64 v1 {d}\n", .{ fd, disp });
                    } else try out.print(gpa, "  store.f64 v1 {d} v{d}\n", .{ disp, fd });
                },
            }
        },
        10 => if (config.floats) switch (rng.uintLessThan(u32, 2)) {
            0 => try out.print(gpa, "  v{d} = cvt.i64.f64 v{d}\n", .{ d, rng.intRangeAtMost(u32, 8, 9) }),
            else => try out.print(gpa, "  v{d} = fcmp.f64 {s} v8 v9\n", .{ d, @tagName(rng.enumValue(ir.FCond)) }),
        },
        else => try out.print(gpa, "  v{d} = add v{d} v{d}\n", .{ d, a, b }),
    }
}

/// What `callee` saw, in order (reset by `run`).
var calls: [64][2]i64 = undefined;
var ncalls: usize = 0;

/// Deterministic C-ABI function for generated `call`s.
pub fn callee(a: i64, b: i64) callconv(.c) i64 {
    if (ncalls < calls.len) calls[ncalls] = .{ a, b };
    ncalls += 1;
    return a *% 31 +% (b ^ @as(i64, @intCast(ncalls)));
}

/// One execution: result, final buffer and call sequence.
pub const Run = struct {
    result: anyerror!i64,
    mem: [mem_slots]u64,
    ncalls: usize,
    calls: [64][2]i64,

    /// Interprets `f(input, &buffer)` with a buffer filled from `input`.
    pub fn of(f: *const ir.Func, input: i64, max_steps: usize) Run {
        var buf: [mem_slots]u64 align(16) = undefined;
        for (&buf, 0..) |*m, i| m.* = @as(u64, @bitCast(input)) *% 0x9e3779b97f4a7c15 +% i;
        ncalls = 0;
        const result = ir.interpret(f, &.{ input, @intCast(@intFromPtr(&buf)) }, max_steps);
        return .{ .result = result, .mem = buf, .ncalls = ncalls, .calls = calls };
    }
    /// Same result, memory and calls (two f64 NaNs in a slot count as equal).
    pub fn expectEqual(want: Run, got: Run) !void {
        const r = want.result catch |e| return std.testing.expectError(e, got.result);
        try std.testing.expectEqual(r, try got.result);
        for (want.mem, got.mem) |x, y| if (x != y) {
            const nan = std.math.isNan(@as(f64, @bitCast(x))) and std.math.isNan(@as(f64, @bitCast(y)));
            if (!nan) return std.testing.expectEqual(x, y);
        };
        try std.testing.expectEqual(want.ncalls, got.ncalls);
        const k = @min(want.ncalls, want.calls.len);
        try std.testing.expectEqualSlices([2]i64, want.calls[0..k], got.calls[0..k]);
    }
};

pub const default_inputs = [_]i64{ 0, 1, -1, 7, -100, 1 << 40, std.math.minInt(i64), std.math.maxInt(i64) };

/// For every input on which `ref` finishes within `max_steps`, `f` behaves the same (it may
/// take up to 4x the steps).
pub fn expectSame(ref: *const ir.Func, f: *const ir.Func, inputs: []const i64, max_steps: usize) !void {
    for (inputs) |in| {
        const want = Run.of(ref, in, max_steps);
        if (want.result) |_| {} else |e| if (e == error.StepLimit) continue;
        try want.expectEqual(Run.of(f, in, 4 * max_steps));
    }
}

/// Parses `src`, builds SSA, runs `pass` and `Ctx.check`, and compares before and after on
/// `default_inputs`. Returns whether the pass reported a change. For pass unit tests.
pub fn checkPass(gpa: std.mem.Allocator, src: []const u8, comptime pass: fn (*opt.Ctx) opt.Error!bool) !bool {
    return checkPassAfter(gpa, src, &.{}, pass);
}

/// `checkPass` with `pre` passes run (and checked) between `ssa.build` and `pass`.
pub fn checkPassAfter(
    gpa: std.mem.Allocator,
    src: []const u8,
    comptime pre: []const Pass,
    comptime pass: fn (*opt.Ctx) opt.Error!bool,
) !bool {
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
    const changed = try pass(&c);
    try c.check("pass");
    try expectSame(&ref, &f, &default_inputs, 20_000);
    return changed;
}

/// An optimizer pass (`run` of a pass module).
pub const Pass = *const fn (*opt.Ctx) opt.Error!bool;

/// `checkPass` on `iterations` random programs (`config`, seed `seed`); prints the failing
/// program. `pre` passes run first (after `ssa.build`), e.g. to give the pass under test
/// the input shape it sees in the pipeline.
pub fn fuzzPass(
    gpa: std.mem.Allocator,
    seed: u64,
    iterations: usize,
    config: Config,
    comptime pre: []const Pass,
    comptime pass: fn (*opt.Ctx) opt.Error!bool,
) !void {
    var prng = std.Random.DefaultPrng.init(seed);
    var src: std.ArrayList(u8) = .empty;
    defer src.deinit(gpa);
    for (0..iterations) |_| {
        src.clearRetainingCapacity();
        try program(prng.random(), gpa, config, &src);
        _ = checkPassAfter(gpa, src.items, pre, pass) catch |e| {
            std.debug.print("fuzzPass failure on:\n{s}\n", .{src.items});
            return e;
        };
    }
}

/// Parses `src`, runs `opt.optimize` with `options` and compares before and after.
pub fn checkOptimize(gpa: std.mem.Allocator, src: []const u8, options: opt.Options) !opt.Stats {
    var ref = try ir.parse(gpa, src);
    defer ref.deinit();
    var f = try ir.parse(gpa, src);
    defer f.deinit();
    const stats = try opt.optimize(gpa, &f, options);
    try f.verify();
    if (options.level != .o0) try std.testing.expect(!f.hasBlockParams());
    try expectSame(&ref, &f, &default_inputs, 20_000);
    return stats;
}

test "testgen: fuzzPass with an identity pass" {
    const identity = struct {
        fn run(_: *opt.Ctx) opt.Error!bool {
            return false;
        }
    }.run;
    try fuzzPass(std.testing.allocator, 1, 40, .{}, &.{identity}, identity);
}

test "testgen: random programs survive the optimizer pipeline" {
    const gpa = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(0x7e57_9e4);
    var src: std.ArrayList(u8) = .empty;
    defer src.deinit(gpa);
    for (0..150) |i| {
        src.clearRetainingCapacity();
        const config: Config = .{ .loops = i % 3 != 0, .calls = i % 4 != 0 };
        try program(prng.random(), gpa, config, &src);
        for ([_]opt.Level{ .o1, .o2 }) |level| {
            _ = checkOptimize(gpa, src.items, .{ .level = level, .verify_each = true }) catch |e| {
                std.debug.print("testgen failure ({s}) on:\n{s}\n", .{ @tagName(level), src.items });
                return e;
            };
        }
    }
}
