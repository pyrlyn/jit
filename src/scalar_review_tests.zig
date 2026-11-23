//! Review tests (scalar SSA passes): a trap-aware interpreter and a fuzzer whose programs
//! contain *guarded* faulting ops (sdiv/srem by arbitrary values, loads/stores at arbitrary
//! offsets), so speculation by the optimizer shows up as a trap the reference never hits.
const std = @import("std");
const ir = @import("core/ir.zig");
const opt = @import("core/opt/opt.zig");
const testgen = opt.testgen;
const t = std.testing;
const builtin = @import("builtin");
const CodeBuffer = @import("core/code_buffer.zig").CodeBuffer;
const JitRuntime = @import("core/jit_runtime.zig");
const host_compiler = switch (builtin.cpu.arch) {
    .x86_64 => @import("x86_64/compiler.zig"),
    .aarch64 => @import("aarch64/compiler.zig"),
    else => void,
};

/// Native code for `f` and an interpreted wrapper that calls it.
pub const Native = struct {
    fun: JitRuntime.Function,
    wrap: ir.Func,
    pub fn init(gpa: std.mem.Allocator, f: *const ir.Func) !Native {
        var cb = CodeBuffer.init(gpa);
        defer cb.deinit();
        _ = try host_compiler.compile(gpa, f, &cb, .{});
        const fun = try JitRuntime.add(&cb);
        var w = ir.Func.init(gpa);
        w.params = f.params;
        w.num_vregs = f.params;
        _ = try w.addBlock("e");
        const args = try w.alloc().alloc(ir.VReg, f.params);
        for (args, 0..) |*a, i| a.* = @intCast(i);
        try w.emit(0, .{ .call = .{ .dst = f.params, .addr = @intFromPtr(fun.mem.ptr), .args = args } });
        try w.emit(0, .{ .ret = f.params });
        return .{ .fun = fun, .wrap = w };
    }
    pub fn deinit(n: *Native) void {
        n.fun.release();
        n.wrap.deinit();
    }
};
pub var check_native = true;

pub var prof_on = false;
pub var prof_blocks: [64]u64 = @splat(0);
pub var prof_taken: [64]u64 = @splat(0);

pub const Trap = error{ Trap, StepLimit, OutOfMemory };

pub const slots = 8;

var calls: [64][2]i64 = undefined;
var ncalls: usize = 0;
pub fn callee(a: i64, b: i64) callconv(.c) i64 {
    if (ncalls < calls.len) calls[ncalls] = .{ a, b };
    ncalls += 1;
    return a *% 31 +% (b ^ @as(i64, @intCast(ncalls)));
}

/// `ir.interpret` that reports undefined behaviour (division traps, any memory access outside
/// `[lo, lo + len)`) as `error.Trap` instead of executing it.
pub fn interpretChecked(gpa: std.mem.Allocator, f: *const ir.Func, args: []const i64, max_steps: usize, lo: usize, len: usize) Trap!i64 {
    const regs = try gpa.alloc(ir.Val, @max(f.num_vregs, 1));
    defer gpa.free(regs);
    @memset(regs, 0);
    for (args, 0..) |a, i| regs[i] = ir.fromI(a);
    var max_params: usize = 1;
    for (f.blocks.items) |blk| max_params = @max(max_params, blk.params.len);
    const tmp = try gpa.alloc(ir.Val, max_params);
    defer gpa.free(tmp);
    var b: u32 = 0;
    var steps: usize = 0;
    while (true) {
        if (prof_on and b < prof_blocks.len) prof_blocks[b] += 1;
        for (f.blocks.items[b].ops.items) |o| {
            steps += 1;
            if (steps > max_steps) return error.StepLimit;
            switch (o) {
                .@"const" => |x| regs[x.dst] = ir.fromI(x.imm),
                .copy => |x| regs[x.dst] = regs[x.src],
                .arith => |x| {
                    const av = ir.toI(regs[x.a]);
                    const bv = ir.toI(regs[x.b]);
                    if ((x.op == .sdiv or x.op == .srem) and (bv == 0 or (av == std.math.minInt(i64) and bv == -1))) return error.Trap;
                    regs[x.dst] = ir.fromI(ir.arith(x.op, av, bv));
                },
                .load => |x| {
                    const addr: u64 = @bitCast(ir.toI(regs[x.base]) +% x.disp);
                    if (addr < lo or addr > lo + len - x.ty.size()) return error.Trap;
                    const p: [*]const u8 = @ptrFromInt(@as(usize, addr));
                    var v: ir.Val = 0;
                    @memcpy(std.mem.asBytes(&v)[0..x.ty.size()], p[0..x.ty.size()]);
                    regs[x.dst] = v;
                },
                .store => |x| {
                    const addr: u64 = @bitCast(ir.toI(regs[x.base]) +% x.disp);
                    if (addr < lo or addr > lo + len - x.ty.size()) return error.Trap;
                    const p: [*]u8 = @ptrFromInt(@as(usize, addr));
                    @memcpy(p[0..x.ty.size()], std.mem.asBytes(&regs[x.src])[0..x.ty.size()]);
                },
                .call => |x| {
                    const F = fn (i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) callconv(.c) i64;
                    var a: std.meta.ArgsTuple(F) = undefined;
                    inline for (0..ir.max_args) |i| a[i] = if (i < x.args.len) ir.toI(regs[x.args[i]]) else 0;
                    regs[x.dst] = ir.fromI(@call(.auto, @as(*const F, @ptrFromInt(x.addr)), a));
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
                .br => |tg| {
                    b = tg;
                    break;
                },
                .cbr => |x| {
                    const tk = ir.cond(x.cond, ir.toI(regs[x.a]), ir.toI(regs[x.b]));
                    if (prof_on and tk and b < prof_taken.len) prof_taken[b] += 1;
                    b = if (tk) x.then else x.@"else";
                    break;
                },
                .bra => |x| {
                    assign(regs, tmp, f.blocks.items[x.target].params, x.args);
                    b = x.target;
                    break;
                },
                .cbra => |x| {
                    const taken = ir.cond(x.cond, ir.toI(regs[x.a]), ir.toI(regs[x.b]));
                    b = if (taken) x.then else x.@"else";
                    assign(regs, tmp, f.blocks.items[b].params, if (taken) x.then_args else x.else_args);
                    break;
                },
                .ret => |v| return ir.toI(regs[v]),
            }
        }
    }
}

fn assign(regs: []ir.Val, tmp: []ir.Val, params: []const ir.VReg, args: []const ir.VReg) void {
    for (args, 0..) |a, i| tmp[i] = regs[a];
    for (params, 0..) |p, i| regs[p] = tmp[i];
}

pub const Run = struct {
    result: Trap!i64,
    mem: [slots]u64,
    ncalls: usize,
    calls: [64][2]i64,

    pub fn of(f: *const ir.Func, input: i64, max_steps: usize) Run {
        var buf: [slots]u64 align(16) = undefined;
        for (&buf, 0..) |*m, i| m.* = @as(u64, @bitCast(input)) *% 0x9e3779b97f4a7c15 +% i;
        ncalls = 0;
        const lo = @intFromPtr(&buf);
        const result = interpretChecked(std.heap.page_allocator, f, &.{ input, @intCast(lo) }, max_steps, lo, @sizeOf(@TypeOf(buf)));
        return .{ .result = result, .mem = buf, .ncalls = ncalls, .calls = calls };
    }
    /// Same result, memory and calls. A NaN anywhere matches any NaN.
    pub fn same(want: Run, got: Run) bool {
        const r = want.result catch return true;
        const g = got.result catch return false;
        if (r != g) return false;
        for (want.mem, got.mem) |x, y| if (x != y) {
            const nan64 = std.math.isNan(@as(f64, @bitCast(x))) and std.math.isNan(@as(f64, @bitCast(y)));
            const lo_nan = std.math.isNan(@as(f32, @bitCast(@as(u32, @truncate(x))))) and std.math.isNan(@as(f32, @bitCast(@as(u32, @truncate(y))))) and (x >> 32) == (y >> 32);
            const hi_nan = std.math.isNan(@as(f32, @bitCast(@as(u32, @truncate(x >> 32))))) and std.math.isNan(@as(f32, @bitCast(@as(u32, @truncate(y >> 32))))) and @as(u32, @truncate(x)) == @as(u32, @truncate(y));
            if (!nan64 and !lo_nan and !hi_nan) return false;
        };
        if (want.ncalls != got.ncalls) return false;
        const k = @min(want.ncalls, want.calls.len);
        return std.mem.eql([2]i64, want.calls[0..k], got.calls[0..k]);
    }
};

pub const inputs = [_]i64{ 0, 1, -1, 2, 3, 7, 8, 16, 24, 40, 56, 57, 63, 64, -8, -100, 1 << 40, std.math.minInt(i64), std.math.maxInt(i64), 0x55 };

pub const Gen = struct {
    loops: bool = true,
    calls: bool = true,
    floats: bool = true,
    blocks: u32 = 7,
    ops: u32 = 7,
    params: bool = false,
};

/// Random program with a "guard" register G (one of the pool), guarded faulting ops that use
/// G (divisor, memory offset), and branches that test G.
pub fn program(rng: std.Random, gpa: std.mem.Allocator, cfg: Gen, out: *std.ArrayList(u8)) !void {
    const n = rng.intRangeAtMost(u32, 2, @max(cfg.blocks, 2));
    const g = rng.intRangeAtMost(u32, 0, 7);
    var np: [64]u32 = @splat(0);
    if (cfg.params) for (np[0..n]) |*k| {
        k.* = rng.uintAtMost(u32, 3);
    };
    try out.print(gpa, "func r 2\n", .{});
    for (0..n) |bi| {
        const b: u32 = @intCast(bi);
        try out.print(gpa, "b{d}", .{b});
        if (np[b] != 0) {
            try out.print(gpa, "(", .{});
            for (0..np[b]) |j| try out.print(gpa, "{s}v{d}", .{ if (j == 0) "" else " ", 20 + 3 * b + @as(u32, @intCast(j)) });
            try out.print(gpa, ")", .{});
        }
        try out.print(gpa, ":\n  v11 = const 1\n  v10 = add v10 v11\n", .{});
        for (0..np[b]) |j| if (rng.boolean()) try out.print(gpa, "  v{d} = {s} v{d} v{d}\n", .{ pool(rng), if (rng.boolean()) "add" else "xor", pool(rng), 20 + 3 * b + @as(u32, @intCast(j)) });
        if (cfg.floats and b == n - 1) try out.print(gpa, "  v8 = fconst.f64 1.25\n  v9 = fconst.f64 -0.0\n  v13 = fconst.f32 2.5\n  v14 = fconst.f32 -0.0\n", .{});
        for (0..rng.uintAtMost(u32, cfg.ops)) |_| try op(rng, gpa, cfg, g, out);
        const tgt = struct {
            fn pick(r: std.Random, c: Gen, from: u32, count: u32) ?u32 {
                if (c.loops and r.uintLessThan(u32, 3) == 0) return r.uintLessThan(u32, count);
                if (from + 1 >= count) return null;
                return r.intRangeLessThan(u32, from + 1, count);
            }
        }.pick;
        const t1 = tgt(rng, cfg, b, n);
        const t2 = tgt(rng, cfg, b, n);
        var e1: [48]u8 = undefined;
        var e2: [48]u8 = undefined;
        const s1 = if (t1) |x| try edgeText(rng, &e1, x, np[x], np[b], 20 + 3 * b) else "";
        const s2 = if (t2) |x| try edgeText(rng, &e2, x, np[x], np[b], 20 + 3 * b) else "";
        if (t1 == null or t2 == null or rng.uintLessThan(u32, 6) == 0) {
            try out.print(gpa, "  ret v{d}\n", .{pool(rng)});
        } else switch (rng.uintLessThan(u32, 6)) {
            0 => {
                try out.print(gpa, "  br b{d}", .{t1.?});
                var it = std.mem.splitScalar(u8, s1, ':');
                _ = it.next();
                while (it.next()) |a| try out.print(gpa, " {s}", .{a});
                try out.print(gpa, "\n", .{});
            },
            1 => try out.print(gpa, "  v11 = const {d}\n  cbr lt v10 v11 {s} {s}\n", .{ rng.intRangeAtMost(i64, 1, 12), s1, s2 }),
            2 => try out.print(gpa, "  v11 = const 0\n  cbr {s} v{d} v11 {s} {s}\n", .{ if (rng.boolean()) "eq" else "ne", g, s1, s2 }),
            3 => try out.print(gpa, "  v11 = const 57\n  cbr {s} v{d} v11 {s} {s}\n", .{ if (rng.boolean()) "ult" else "uge", g, s1, s2 }),
            4 => try out.print(gpa, "  v11 = const -1\n  cbr {s} v{d} v11 {s} {s}\n", .{ if (rng.boolean()) "eq" else "ne", g, s1, s2 }),
            else => try out.print(gpa, "  cbr {s} v{d} v{d} {s} {s}\n", .{ @tagName(rng.enumValue(ir.CondCode)), pool(rng), pool(rng), s1, s2 }),
        }
    }
}

/// `b<to>` or `b<to>:vA:vB` with arguments from the pool or the current block's parameters.
fn edgeText(rng: std.Random, buf: []u8, to: u32, nargs: u32, own: u32, own_base: u32) ![]const u8 {
    var w: std.Io.Writer = .fixed(buf);
    try w.print("b{d}", .{to});
    for (0..nargs) |_| {
        const v = if (own != 0 and rng.boolean()) own_base + rng.uintLessThan(u32, own) else pool(rng);
        try w.print(":v{d}", .{v});
    }
    return w.buffered();
}

fn pool(rng: std.Random) u32 {
    return rng.uintAtMost(u32, 7);
}

fn op(rng: std.Random, gpa: std.mem.Allocator, cfg: Gen, g: u32, out: *std.ArrayList(u8)) !void {
    const d = rng.intRangeAtMost(u32, 2, 7);
    const a = pool(rng);
    const b = pool(rng);
    switch (rng.uintLessThan(u32, 16)) {
        0 => try out.print(gpa, "  v{d} = const {d}\n", .{ d, switch (rng.uintLessThan(u32, 5)) {
            0 => rng.intRangeAtMost(i64, -4, 4),
            1 => @as(i64, 1) << rng.intRangeAtMost(u6, 0, 63),
            2 => rng.int(i64),
            3 => -(@as(i64, 1) << rng.intRangeAtMost(u6, 0, 62)),
            else => rng.intRangeAtMost(i64, -70, 70),
        } }),
        1 => try out.print(gpa, "  v{d} = copy v{d}\n", .{ d, a }),
        2, 3 => try out.print(gpa, "  v{d} = {s} v{d} v{d}\n", .{ d, @tagName(rng.enumValue(ir.Arith)), a, b }),
        // Guarded division: by G (maybe 0 / -1) or by a power of two / constant.
        4 => try out.print(gpa, "  v{d} = {s} v{d} v{d}\n", .{ d, if (rng.boolean()) "sdiv" else "srem", a, g }),
        5 => {
            const k: i64 = switch (rng.uintLessThan(u32, 4)) {
                0 => @as(i64, 1) << rng.intRangeAtMost(u6, 0, 63),
                1 => -(@as(i64, 1) << rng.intRangeAtMost(u6, 0, 62)),
                2 => rng.intRangeAtMost(i64, -9, 9),
                else => rng.int(i64),
            };
            if (k == 0) return;
            try out.print(gpa, "  v12 = const {d}\n  v{d} = {s} v{d} v12\n", .{ k, d, if (rng.boolean()) "sdiv" else "srem", a });
        },
        // Guarded memory: v1 + G.
        6 => try out.print(gpa, "  v12 = add v1 v{d}\n  v{d} = load v12 0\n", .{ g, d }),
        7 => try out.print(gpa, "  v12 = add v1 v{d}\n  store v12 0 v{d}\n", .{ g, a }),
        8, 9 => {
            const disp = 4 * rng.uintLessThan(u32, 15);
            switch (rng.uintLessThan(u32, 6)) {
                0 => try out.print(gpa, "  v{d} = load v1 {d}\n", .{ d, disp }),
                1 => try out.print(gpa, "  store v1 {d} v{d}\n", .{ disp, a }),
                2 => if (cfg.floats) try out.print(gpa, "  v13 = load.f32 v1 {d}\n", .{disp + 4 * rng.uintLessThan(u32, 2)}),
                3 => if (cfg.floats) try out.print(gpa, "  store.f32 v1 {d} v{d}\n", .{ disp + 4 * rng.uintLessThan(u32, 2), 13 + rng.uintLessThan(u32, 2) }),
                4 => if (cfg.floats) try out.print(gpa, "  v8 = load.f64 v1 {d}\n", .{disp}),
                else => if (cfg.floats) try out.print(gpa, "  store.f64 v1 {d} v{d}\n", .{ disp, 8 + rng.uintLessThan(u32, 2) }),
            }
        },
        10 => if (cfg.calls) try out.print(gpa, "  v{d} = call 0x{x} v{d} v{d}\n", .{ d, @intFromPtr(&callee), a, b }),
        11 => if (cfg.floats) {
            const fd = rng.intRangeAtMost(u32, 8, 9);
            switch (rng.uintLessThan(u32, 3)) {
                0 => try out.print(gpa, "  v{d} = cvt.f64.i64 v{d}\n", .{ fd, a }),
                1 => try out.print(gpa, "  v{d} = {s}.f64 v{d} v{d}\n", .{ fd, @tagName(rng.enumValue(enum { add, sub, mul, div, min, max })), 8 + rng.uintLessThan(u32, 2), 8 + rng.uintLessThan(u32, 2) }),
                else => try out.print(gpa, "  v{d} = fconst.f64 {d}\n", .{ fd, @as(f64, @floatFromInt(rng.intRangeAtMost(i32, -8, 8))) / 4 }),
            }
        },
        12 => if (cfg.floats) {
            const fd = rng.intRangeAtMost(u32, 13, 14);
            switch (rng.uintLessThan(u32, 3)) {
                0 => try out.print(gpa, "  v{d} = cvt.f32.i64 v{d}\n", .{ fd, a }),
                1 => try out.print(gpa, "  v{d} = {s}.f32 v{d} v{d}\n", .{ fd, @tagName(rng.enumValue(enum { add, sub, mul, div, min, max })), 13 + rng.uintLessThan(u32, 2), 13 + rng.uintLessThan(u32, 2) }),
                else => try out.print(gpa, "  v{d} = cvt.f32.f64 v{d}\n", .{ fd, 8 + rng.uintLessThan(u32, 2) }),
            }
        },
        13 => if (cfg.floats) switch (rng.uintLessThan(u32, 3)) {
            0 => try out.print(gpa, "  v{d} = cvt.i64.f64 v{d}\n", .{ d, rng.intRangeAtMost(u32, 8, 9) }),
            1 => try out.print(gpa, "  v{d} = fcmp.f64 {s} v{d} v{d}\n", .{ d, @tagName(rng.enumValue(ir.FCond)), 8 + rng.uintLessThan(u32, 2), 8 + rng.uintLessThan(u32, 2) }),
            else => try out.print(gpa, "  v{d} = cvt.i64.f32 v{d}\n", .{ d, rng.intRangeAtMost(u32, 13, 14) }),
        },
        14 => try out.print(gpa, "  v12 = const {d}\n  v{d} = {s} v{d} v12\n", .{ rng.intRangeAtMost(i64, -3, 70), d, @tagName(rng.enumValue(enum { shl, shr, sar })), a }),
        else => try out.print(gpa, "  v12 = const {d}\n  v{d} = {s} v{d} v12\n  v{d} = {s} v{d} v12\n", .{
            rng.int(i64) >> rng.intRangeAtMost(u6, 0, 63),
            d,
            @tagName(rng.enumValue(enum { add, sub, mul, @"and", @"or", xor })),
            a,
            d,
            @tagName(rng.enumValue(enum { add, sub, mul, @"and", @"or", xor })),
            d,
        }),
    }
}

pub fn printFunc(f: *const ir.Func) void {
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(std.heap.page_allocator);
    f.print(std.heap.page_allocator, &out) catch return;
    std.debug.print("{s}\n", .{out.items});
}

/// Returns true on mismatch (prints details).
pub var skipped: usize = 0;
pub var compared: usize = 0;
pub fn checkSrc(gpa: std.mem.Allocator, src: []const u8, levels: []const opt.Level, vector: opt.VectorTarget, profile: ?*const opt.Profile) !bool {
    var ref = try ir.parse(gpa, src);
    defer ref.deinit();
    for (levels) |lv| {
        var f = try ir.parse(gpa, src);
        defer f.deinit();
        _ = opt.optimize(gpa, &f, .{ .level = lv, .verify_each = true, .vector = vector, .profile = profile }) catch |e| {
            std.debug.print("optimize error {s} at {s} on:\n{s}\n", .{ @errorName(e), @tagName(lv), src });
            return true;
        };
        for (inputs) |in| {
            const want = Run.of(&ref, in, 20_000);
            _ = want.result catch {
                skipped += 1;
                continue;
            };
            compared += 1;
            const got = Run.of(&f, in, 80_000);
            var native_ok = true;
            var nat_got: ?Run = null;
            if (want.same(got) and check_native) {
                var n = try Native.init(gpa, &f);
                defer n.deinit();
                nat_got = Run.of(&n.wrap, in, 80_000);
                native_ok = want.same(nat_got.?);
            }
            if (!native_ok) {
                std.debug.print("NATIVE MISMATCH level {s} input {d}: want {any} got {any}\nwant mem {any}\ngot mem {any}\nsrc:\n{s}\nopt:\n", .{ @tagName(lv), in, want.result, nat_got.?.result, want.mem, nat_got.?.mem, src });
                printFunc(&f);
                return true;
            }
            if (!want.same(got)) {
                std.debug.print("MISMATCH level {s} input {d}: want {any} got {any}\nwant mem {any}\ngot mem {any}\nwant calls {d} got {d}\nsrc:\n{s}\nopt:\n", .{ @tagName(lv), in, want.result, got.result, want.mem, got.mem, want.ncalls, got.ncalls, src });
                printFunc(&f);
                return true;
            }
        }
    }
    return false;
}

fn envCount(name: []const u8, default: usize) usize {
    const v = std.c.getenv(@ptrCast(name.ptr)) orelse return default;
    return std.fmt.parseInt(usize, std.mem.span(v), 10) catch default;
}

test "review-scalar: guarded faulting ops fuzz" {
    const gpa = t.allocator;
    const iters = envCount("REV_ITERS\x00", 300);
    const seed = envCount("REV_SEED\x00", 1);
    var prng = std.Random.DefaultPrng.init(seed);
    var src: std.ArrayList(u8) = .empty;
    defer src.deinit(gpa);
    var fails: usize = 0;
    for (0..iters) |i| {
        src.clearRetainingCapacity();
        const cfg: Gen = .{ .loops = i % 4 != 0, .calls = i % 3 != 0, .floats = i % 5 != 0, .params = i % 2 == 1 };
        try program(prng.random(), gpa, cfg, &src);
        var prof: opt.Profile = .{};
        var pb: [64]u64 = undefined;
        var pt: [64]u64 = undefined;
        if (i % 2 == 0) {
            var ref = try ir.parse(gpa, src.items);
            defer ref.deinit();
            prof_blocks = @splat(0);
            prof_taken = @splat(0);
            prof_on = true;
            for ([_]i64{ 0, 1, 7, -1 }) |in| _ = Run.of(&ref, in, 20_000);
            prof_on = false;
            const nb = ref.blocks.items.len;
            pb = prof_blocks;
            pt = prof_taken;
            if (i % 6 == 0) for (pb[0..nb], pt[0..nb]) |*x, *y| {
                x.* = prng.random().uintAtMost(u64, 50);
                y.* = prng.random().uintAtMost(u64, 60);
            };
            prof = .{ .blocks = pb[0..nb], .taken = pt[0..nb] };
        }
        if (try checkSrc(gpa, src.items, &.{ .o0, .o1, .o2 }, opt.VectorTarget.host, if (i % 2 == 0) &prof else null)) {
            fails += 1;
            if (fails >= 3) break;
        }
    }
    std.debug.print("review fuzz: compared {d} skipped {d}\n", .{ compared, skipped });
    try t.expectEqual(@as(usize, 0), fails);
}

fn expectAll(src: []const u8) !void {
    try t.expect(!try checkSrc(t.allocator, src, &.{ .o0, .o1, .o2 }, opt.VectorTarget.host, null));
}

test "review-scalar: float edge folding vs native (cvt out of range, -0.0 min/max, NaN)" {
    // 1e30, -1e30, NaN (0/0), -0.0 and +0.0 computed at run time (o0) or folded (o1/o2).
    inline for (.{ "min", "max" }) |m| {
        inline for (.{ .{ "v2", "v3" }, .{ "v3", "v2" }, .{ "v4", "v5" }, .{ "v5", "v4" }, .{ "v2", "v4" }, .{ "v4", "v2" } }) |ab| {
            try expectAll("func r 2\ne:\n  v2 = fconst.f64 -0.0\n  v3 = fconst.f64 0.0\n  v6 = fconst.f64 1.0\n  v4 = div.f64 v3 v3\n  v5 = copy v6\n" ++
                "  v7 = " ++ m ++ ".f64 " ++ ab[0] ++ " " ++ ab[1] ++ "\n  store.f64 v1 0 v7\n  v8 = load v1 0\n  ret v8\n");
            try expectAll("func r 2\ne:\n  v2 = fconst.f32 -0.0\n  v3 = fconst.f32 0.0\n  v6 = fconst.f32 1.0\n  v4 = div.f32 v3 v3\n  v5 = copy v6\n" ++
                "  v7 = " ++ m ++ ".f32 " ++ ab[0] ++ " " ++ ab[1] ++ "\n  store.f32 v1 0 v7\n  v8 = load v1 0\n  ret v8\n");
        }
    }
    inline for (.{ "1e30", "-1e30", "9.3e18", "-9.3e18", "-9.223372036854775808e18" }) |k| {
        try expectAll("func r 2\ne:\n  v2 = fconst.f64 " ++ k ++ "\n  v3 = cvt.i64.f64 v2\n  ret v3\n");
        try expectAll("func r 2\ne:\n  v2 = fconst.f32 " ++ k ++ "\n  v3 = cvt.i64.f32 v2\n  ret v3\n");
    }
}

/// Optimizes `src` twice (o2 then `lv2`) and compares with the original.
fn checkTwice(gpa: std.mem.Allocator, src: []const u8, lv2: opt.Level, vector: opt.VectorTarget) !bool {
    var ref = try ir.parse(gpa, src);
    defer ref.deinit();
    var f = try ir.parse(gpa, src);
    defer f.deinit();
    _ = opt.optimize(gpa, &f, .{ .level = .o2, .verify_each = true, .vector = vector }) catch |e| {
        std.debug.print("optimize#1 error {s} on:\n{s}\n", .{ @errorName(e), src });
        return true;
    };
    _ = opt.optimize(gpa, &f, .{ .level = lv2, .verify_each = true, .vector = vector }) catch |e| {
        std.debug.print("optimize#2 error {s} on:\n{s}\n", .{ @errorName(e), src });
        printFunc(&f);
        return true;
    };
    for (inputs) |in| {
        const want = Run.of(&ref, in, 20_000);
        _ = want.result catch continue;
        const got = Run.of(&f, in, 160_000);
        if (!want.same(got)) {
            std.debug.print("TWICE MISMATCH input {d}: want {any} got {any}\nsrc:\n{s}\nopt:\n", .{ in, want.result, got.result, src });
            printFunc(&f);
            return true;
        }
    }
    return false;
}

test "review-scalar: large programs, no vectors, re-optimized output" {
    const gpa = t.allocator;
    const iters = envCount("REV_ITERS\x00", 300);
    const seed = envCount("REV_SEED\x00", 1);
    var prng = std.Random.DefaultPrng.init(seed ^ 0xabcdef);
    var src: std.ArrayList(u8) = .empty;
    defer src.deinit(gpa);
    var fails: usize = 0;
    for (0..iters) |i| {
        src.clearRetainingCapacity();
        const cfg: Gen = .{ .loops = i % 4 != 0, .calls = i % 3 != 0, .floats = i % 5 != 0, .blocks = 14, .ops = 14 };
        try program(prng.random(), gpa, cfg, &src);
        if (try checkSrc(gpa, src.items, &.{ .o1, .o2 }, .none, null)) fails += 1;
        if (try checkTwice(gpa, src.items, if (i % 2 == 0) .o2 else .o1, opt.VectorTarget.host)) fails += 1;
        if (fails >= 3) break;
    }
    try t.expectEqual(@as(usize, 0), fails);
}

test "review-scalar: hand-written loops (rotation, exits, irreducible, guarded faults)" {
    const cases = [_][]const u8{
        // (a) header value used after the loop; exit also reached by a break and from outside.
        \\func r 2
        \\e:
        \\  v2 = const 0
        \\  v3 = const 3
        \\  cbr eq v0 v3 x h
        \\h:
        \\  v4 = load v1 0
        \\  v5 = add v4 v2
        \\  v6 = const 5
        \\  cbr lt v2 v6 body x
        \\body:
        \\  v7 = const 1
        \\  v2 = add v2 v7
        \\  store v1 0 v2
        \\  cbr eq v2 v0 x h
        \\x:
        \\  v8 = add v5 v2
        \\  ret v8
        ,
        // (b) nested loops sharing the exit; header values used in the inner loop and after.
        \\func r 2
        \\e:
        \\  v2 = const 0
        \\  br h1
        \\h1:
        \\  v3 = const 4
        \\  v9 = mul v2 v2
        \\  cbr lt v2 v3 h2 x
        \\h2:
        \\  v4 = const 0
        \\  br h2b
        \\h2b:
        \\  v5 = const 3
        \\  v10 = add v9 v4
        \\  cbr lt v4 v5 b2 l1
        \\b2:
        \\  v6 = const 1
        \\  v4 = add v4 v6
        \\  store v1 8 v10
        \\  br h2b
        \\l1:
        \\  v7 = const 1
        \\  v2 = add v2 v7
        \\  br h1
        \\x:
        \\  v8 = add v9 v10
        \\  ret v8
        ,
        // (c) guarded division in a top-tested loop: g = v0, header tests g != 0 every iteration.
        \\func r 2
        \\e:
        \\  v2 = const 0
        \\  v3 = const 0
        \\  br h
        \\h:
        \\  v12 = const 0
        \\  cbr eq v0 v12 x b
        \\b:
        \\  v4 = sdiv v1 v0
        \\  v5 = srem v1 v0
        \\  v3 = add v3 v4
        \\  v3 = add v3 v5
        \\  v6 = const 1
        \\  v2 = add v2 v6
        \\  v7 = const 4
        \\  cbr lt v2 v7 h x
        \\x:
        \\  ret v3
        ,
        // (d) loop-invariant guarded load: offset v0 checked in the header.
        \\func r 2
        \\e:
        \\  v2 = const 0
        \\  v3 = const 0
        \\  br h
        \\h:
        \\  v12 = const 57
        \\  cbr uge v0 v12 x b
        \\b:
        \\  v13 = add v1 v0
        \\  v4 = load v13 0
        \\  v3 = add v3 v4
        \\  v6 = const 1
        \\  v2 = add v2 v6
        \\  v7 = const 4
        \\  cbr lt v2 v7 h x
        \\x:
        \\  ret v3
        ,
        // (e) irreducible: two entries into a cycle, guarded division inside.
        \\func r 2
        \\e:
        \\  v2 = const 0
        \\  v3 = const 1
        \\  cbr lt v0 v3 a b
        \\a:
        \\  v2 = add v2 v3
        \\  v4 = const 9
        \\  cbr gt v2 v4 x b
        \\b:
        \\  v5 = const 0
        \\  cbr eq v0 v5 a c
        \\c:
        \\  v6 = sdiv v2 v0
        \\  v2 = add v2 v6
        \\  v2 = add v2 v3
        \\  br a
        \\x:
        \\  ret v2
        ,
        // (f) latch ending in cbr (merged latch), division in the header, value after loop.
        \\func r 2
        \\e:
        \\  v2 = const 0
        \\  v9 = const 0
        \\  br h
        \\h:
        \\  v3 = const 7
        \\  v4 = srem v2 v3
        \\  v10 = const 6
        \\  cbr lt v2 v10 b x
        \\b:
        \\  v5 = const 1
        \\  v2 = add v2 v5
        \\  v9 = add v9 v4
        \\  cbr eq v4 v5 h h2
        \\h2:
        \\  v9 = add v9 v9
        \\  br h
        \\x:
        \\  v11 = add v9 v4
        \\  ret v11
        ,
        // (g) f32/i64 overlapping stores and loads on one base, across a single-pred chain.
        \\func r 2
        \\e:
        \\  v2 = const 12345
        \\  store v1 0 v2
        \\  v3 = fconst.f32 1.5
        \\  store.f32 v1 4 v3
        \\  v4 = load v1 0
        \\  v5 = load.f32 v1 4
        \\  v6 = load.f32 v1 0
        \\  br n
        \\n:
        \\  store.f32 v1 0 v5
        \\  v7 = load v1 0
        \\  v8 = load.f32 v1 4
        \\  v9 = cvt.i64.f32 v8
        \\  v10 = add v7 v9
        \\  v11 = add v10 v4
        \\  v12 = cvt.i64.f32 v6
        \\  v13 = add v11 v12
        \\  ret v13
        ,
        // (h) store through an aliasing base between a store and a load of v1.
        \\func r 2
        \\e:
        \\  v2 = const 77
        \\  store v1 8 v2
        \\  v3 = const 4
        \\  v4 = add v1 v3
        \\  v5 = fconst.f32 -1.0
        \\  store.f32 v4 4 v5
        \\  v6 = load v1 8
        \\  ret v6
        ,
    };
    for (cases) |src| try expectAll(src);
}

fn dumpO2(src: []const u8) !void {
    var f = try ir.parse(t.allocator, src);
    defer f.deinit();
    _ = try opt.optimize(t.allocator, &f, .{ .level = .o2, .verify_each = true });
    printFunc(&f);
}

test "review-scalar: dump" {
    try dumpO2(
        \\func r 2
        \\e:
        \\  v2 = const 0
        \\  v3 = const 0
        \\  br h
        \\h:
        \\  v12 = const 0
        \\  cbr eq v0 v12 x b
        \\b:
        \\  v4 = sdiv v1 v0
        \\  v5 = srem v1 v0
        \\  v3 = add v3 v4
        \\  v3 = add v3 v5
        \\  v6 = const 1
        \\  v2 = add v2 v6
        \\  v7 = const 4
        \\  cbr lt v2 v7 h x
        \\x:
        \\  ret v3
    );
}

extern "c" fn clock_gettime_nsec_np(clock: c_int) u64;
fn timeOpt(src: []const u8, label: []const u8) !void {
    const gpa = std.heap.page_allocator;
    var f = try ir.parse(gpa, src);
    defer f.deinit();
    const t0 = clock_gettime_nsec_np(4);
    const st = try opt.optimize(gpa, &f, .{ .level = .o2, .verify_each = false, .vector = opt.VectorTarget.host });
    std.debug.print("{s}: {d} ms, ops {d} -> {d}, blocks {d}\n", .{ label, (clock_gettime_nsec_np(4) - t0) / 1_000_000, st.ops_before, st.ops_after, st.blocks_after });
}

test "review-scalar: compile-time scaling" {
    const gpa = std.heap.page_allocator;
    for ([_]usize{ 250, 500, 1000, 2000 }) |n| {
        // Chain of diamonds with loads/stores and arithmetic (many blocks, one long dom path).
        var src: std.ArrayList(u8) = .empty;
        defer src.deinit(gpa);
        try src.print(gpa, "func r 2\ne:\n  v2 = const 0\n  br d0\n", .{});
        for (0..n) |i| {
            try src.print(gpa, "d{d}:\n  v3 = load v1 {d}\n  v2 = add v2 v3\n  v4 = const {d}\n  cbr lt v2 v4 a{d} b{d}\na{d}:\n  store v1 {d} v2\n  v2 = mul v2 v3\n  br d{d}\nb{d}:\n  v2 = xor v2 v4\n  br d{d}\n", .{ i, 8 * (i % 8), i, i, i, i, 8 * ((i + 3) % 8), i + 1, i, i + 1 });
        }
        try src.print(gpa, "d{d}:\n  ret v2\n", .{n});
        try timeOpt(src.items, "diamonds");
        // Deep loop nest.
        src.clearRetainingCapacity();
        const depth = n / 10;
        try src.print(gpa, "func r 2\ne:\n  v2 = const 0\n", .{});
        for (0..depth) |i| try src.print(gpa, "  v{d} = const 0\n  br h{d}\nh{d}:\n  v{d} = load v1 0\n  v{d} = const 2\n  cbr ge v{d} v{d} x{d} b{d}\nb{d}:\n", .{ 10 + 3 * i, i, i, 11 + 3 * i, 12 + 3 * i, 10 + 3 * i, 12 + 3 * i, i, i, i });
        try src.print(gpa, "  v2 = add v2 v0\n", .{});
        var i = depth;
        while (i > 0) {
            i -= 1;
            try src.print(gpa, "  v9 = const 1\n  v{d} = add v{d} v9\n  br h{d}\nx{d}:\n", .{ 10 + 3 * i, 10 + 3 * i, i, i });
        }
        try src.print(gpa, "  ret v2\n", .{});
        try timeOpt(src.items, "loop nest");
    }
}

fn nest(gpa: std.mem.Allocator, depth: usize, src: *std.ArrayList(u8)) !void {
    try src.print(gpa, "func r 2\ne:\n  v2 = const 0\n", .{});
    for (0..depth) |i| try src.print(gpa, "  v{d} = const 0\n  br h{d}\nh{d}:\n  v{d} = load v1 0\n  v{d} = const 2\n  cbr ge v{d} v{d} x{d} b{d}\nb{d}:\n", .{ 10 + 3 * i, i, i, 11 + 3 * i, 12 + 3 * i, 10 + 3 * i, 12 + 3 * i, i, i, i });
    try src.print(gpa, "  v2 = add v2 v0\n", .{});
    var i = depth;
    while (i > 0) {
        i -= 1;
        try src.print(gpa, "  v9 = const 1\n  v{d} = add v{d} v9\n  br h{d}\nx{d}:\n", .{ 10 + 3 * i, 10 + 3 * i, i, i });
    }
    try src.print(gpa, "  ret v2\n", .{});
}

test "review-scalar: per-pass time on a loop nest" {
    const gpa = std.heap.page_allocator;
    for ([_]usize{ 100, 200, 400 }) |depth| {
        var src: std.ArrayList(u8) = .empty;
        defer src.deinit(gpa);
        try nest(gpa, depth, &src);
        var f = try ir.parse(gpa, src.items);
        defer f.deinit();
        const options: opt.Options = .{ .level = .o2, .verify_each = false };
        var stats: opt.Stats = .{};
        var c = try opt.Ctx.init(gpa, &f, &options, &stats);
        defer c.deinit();
        var t0 = clock_gettime_nsec_np(4);
        try opt.ssa.build(&c);
        std.debug.print("depth {d}: ssa.build {d} us\n", .{ depth, (clock_gettime_nsec_np(4) - t0) / 1000 });
        inline for (.{ .{ opt.sccp.run, "sccp" }, .{ opt.instcombine.run, "instcombine" }, .{ opt.gvn.run, "gvn" }, .{ opt.dce.run, "dce" }, .{ opt.simplifycfg.run, "simplifycfg" }, .{ opt.licm.run, "licm" }, .{ opt.loops.rotate, "rotate" }, .{ opt.licm.run, "licm2" }, .{ opt.sccp.run, "sccp" }, .{ opt.instcombine.run, "instcombine" }, .{ opt.gvn.run, "gvn" }, .{ opt.dce.run, "dce" }, .{ opt.simplifycfg.run, "simplifycfg" }, .{ opt.remat.run, "remat" } }) |p| {
            t0 = clock_gettime_nsec_np(4);
            _ = try p[0](&c);
            std.debug.print("  {s}: {d} us\n", .{ p[1], (clock_gettime_nsec_np(4) - t0) / 1000 });
        }
        t0 = clock_gettime_nsec_np(4);
        try opt.out_of_ssa.run(&c);
        std.debug.print("  out_of_ssa: {d} us\n", .{(clock_gettime_nsec_np(4) - t0) / 1000});
        t0 = clock_gettime_nsec_np(4);
        _ = try opt.layout.run(&c);
        std.debug.print("  layout: {d} us\n", .{(clock_gettime_nsec_np(4) - t0) / 1000});
    }
}

test "review-scalar: Loops.init scaling" {
    const gpa = std.heap.page_allocator;
    for ([_]usize{ 200, 400, 800 }) |depth| {
        var src: std.ArrayList(u8) = .empty;
        defer src.deinit(gpa);
        try nest(gpa, depth, &src);
        var f = try ir.parse(gpa, src.items);
        defer f.deinit();
        var g = try opt.cfg.Cfg.init(gpa, &f);
        defer g.deinit();
        const t0 = clock_gettime_nsec_np(4);
        var ls = try opt.cfg.Loops.init(gpa, &f, &g);
        defer ls.deinit();
        const t1 = clock_gettime_nsec_np(4);
        var f2 = try ir.parse(gpa, src.items);
        defer f2.deinit();
        _ = try opt.optimize(gpa, &f2, .{ .level = .o2, .verify_each = false });
        const t2 = clock_gettime_nsec_np(4);
        var f3 = try ir.parse(gpa, src.items);
        defer f3.deinit();
        _ = try opt.optimize(gpa, &f3, .{ .level = .o1, .verify_each = false });
        const t3 = clock_gettime_nsec_np(4);
        std.debug.print("depth {d} ({d} ops): Loops.init {d} us, optimize o2 {d} ms, o1 {d} ms\n", .{ depth, opt.countOps(&f), (t1 - t0) / 1000, (t2 - t1) / 1_000_000, (t3 - t2) / 1_000_000 });
    }
}

test "review-scalar: huge profile counts (two latches) do not overflow" {
    const src =
        \\func r 2
        \\e:
        \\  v2 = const 0
        \\  br h
        \\h:
        \\  v3 = const 6
        \\  cbr lt v2 v3 b x
        \\b:
        \\  v4 = const 1
        \\  v2 = add v2 v4
        \\  cbr eq v2 v4 h l2
        \\l2:
        \\  store v1 0 v2
        \\  br h
        \\x:
        \\  ret v2
    ;
    const big = std.math.maxInt(u64) / 2 + 10;
    const prof: opt.Profile = .{ .blocks = &.{ 1, big, big, big, 1 }, .taken = &.{ 0, big, big, 0, 0 } };
    var f = try ir.parse(t.allocator, src);
    defer f.deinit();
    _ = try opt.optimize(t.allocator, &f, .{ .level = .o2, .verify_each = true, .profile = &prof });
}

test "review-scalar: few scalar rounds (max_rounds 0/1) before the loop passes" {
    const gpa = t.allocator;
    const iters = envCount("REV_ITERS\x00", 300);
    const seed = envCount("REV_SEED\x00", 1);
    var prng = std.Random.DefaultPrng.init(seed ^ 0x5151);
    var src: std.ArrayList(u8) = .empty;
    defer src.deinit(gpa);
    var fails: usize = 0;
    for (0..iters) |i| {
        src.clearRetainingCapacity();
        const cfg: Gen = .{ .loops = true, .calls = i % 3 == 0, .floats = i % 5 != 0, .blocks = 9, .ops = 8 };
        try program(prng.random(), gpa, cfg, &src);
        var ref = try ir.parse(gpa, src.items);
        defer ref.deinit();
        for ([_]u8{ 0, 1 }) |mr| {
            var f = try ir.parse(gpa, src.items);
            defer f.deinit();
            _ = opt.optimize(gpa, &f, .{ .level = .o2, .verify_each = true, .max_rounds = mr, .vector = opt.VectorTarget.host }) catch |e| {
                std.debug.print("max_rounds {d}: optimize error {s} on:\n{s}\n", .{ mr, @errorName(e), src.items });
                fails += 1;
                continue;
            };
            for (inputs) |in| {
                const want = Run.of(&ref, in, 20_000);
                _ = want.result catch continue;
                const got = Run.of(&f, in, 80_000);
                if (!want.same(got)) {
                    std.debug.print("max_rounds {d} MISMATCH input {d}: want {any} got {any}\nsrc:\n{s}\nopt:\n", .{ mr, in, want.result, got.result, src.items });
                    printFunc(&f);
                    fails += 1;
                    break;
                }
            }
        }
        if (fails >= 2) break;
    }
    try t.expectEqual(@as(usize, 0), fails);
}

fn chain(gpa: std.mem.Allocator, n: usize, reverse: bool, src: *std.ArrayList(u8)) !void {
    try src.print(gpa, "func r 2\ne:\n", .{});
    for (0..n) |i| try src.print(gpa, "  v{d} = copy v0\n", .{10 + i});
    try src.print(gpa, "  v2 = const 0\n  br h\nh:\n  v3 = const 3\n  cbr lt v2 v3 b x\nb:\n  v5 = const 1\n  v2 = add v2 v5\n", .{});
    for (0..n - 1) |k| {
        const i = if (reverse) n - 2 - k else k;
        // v_i = v_{i+1} (in reverse order: v_{i} = v_{i+1} with i descending)
        try src.print(gpa, "  v{d} = copy v{d}\n", .{ 10 + i, 11 + i });
    }
    try src.print(gpa, "  br h\nx:\n  v6 = const 0\n", .{});
    for (0..n) |i| try src.print(gpa, "  v6 = xor v6 v{d}\n", .{10 + i});
    try src.print(gpa, "  ret v6\n", .{});
}

test "review-scalar: parameter chains scaling" {
    const gpa = std.heap.page_allocator;
    for ([_]usize{ 500, 1000, 2000, 4000 }) |n| for ([_]bool{ false, true }) |rev| {
        var src: std.ArrayList(u8) = .empty;
        defer src.deinit(gpa);
        try chain(gpa, n, rev, &src);
        var f = try ir.parse(gpa, src.items);
        defer f.deinit();
        const t0 = clock_gettime_nsec_np(4);
        const st = try opt.optimize(gpa, &f, .{ .level = .o2, .verify_each = false });
        std.debug.print("chain n={d} rev={}: {d} ms ({d} -> {d} ops)\n", .{ n, rev, (clock_gettime_nsec_np(4) - t0) / 1_000_000, st.ops_before, st.ops_after });
    };
}

test "review-scalar: per-pass time on a parameter chain" {
    const gpa = std.heap.page_allocator;
    var src: std.ArrayList(u8) = .empty;
    defer src.deinit(gpa);
    try chain(gpa, 4000, false, &src);
    var f = try ir.parse(gpa, src.items);
    defer f.deinit();
    const options: opt.Options = .{ .level = .o2, .verify_each = false };
    var stats: opt.Stats = .{};
    var c = try opt.Ctx.init(gpa, &f, &options, &stats);
    defer c.deinit();
    var t0 = clock_gettime_nsec_np(4);
    try opt.ssa.build(&c);
    std.debug.print("ssa.build {d} us\n", .{(clock_gettime_nsec_np(4) - t0) / 1000});
    inline for (.{ .{ opt.sccp.run, "sccp" }, .{ opt.instcombine.run, "instcombine" }, .{ opt.gvn.run, "gvn" }, .{ opt.dce.run, "dce" }, .{ opt.simplifycfg.run, "simplifycfg" }, .{ opt.sccp.run, "sccp" }, .{ opt.instcombine.run, "instcombine" }, .{ opt.gvn.run, "gvn" }, .{ opt.dce.run, "dce" }, .{ opt.simplifycfg.run, "simplifycfg" } }) |p| {
        t0 = clock_gettime_nsec_np(4);
        const ch = try p[0](&c);
        std.debug.print("  {s}: {d} us changed={}\n", .{ p[1], (clock_gettime_nsec_np(4) - t0) / 1000, ch });
    }
}
