//! Temporary review tests: vectorized .o2 code (host vectors) run natively vs ir.interpret.
const std = @import("std");
const builtin = @import("builtin");
const ir = @import("core/ir.zig");
const opt = @import("core/opt/opt.zig");
const CodeBuffer = @import("core/code_buffer.zig").CodeBuffer;
const JitRuntime = @import("core/jit_runtime.zig");
const opts = @import("pyrlyn_jit_options");
const t = std.testing;

const host_compiler = switch (builtin.cpu.arch) {
    .x86_64 => if (opts.x86_64) @import("x86_64/compiler.zig") else void,
    .aarch64 => if (opts.aarch64) @import("aarch64/compiler.zig") else void,
    else => void,
};
const can_run = host_compiler != void;
pub const can_run_pub = can_run;
const gpa = std.heap.page_allocator;

const words = 4096;
const Buf = [words]u64;
pub const Fill = enum { i64, f64, f32, f64s, f32s };

fn fillBuf(buf: *align(16) Buf, seed: u64, fill: Fill) void {
    var prng = std.Random.DefaultPrng.init(seed);
    const r = prng.random();
    const sp64 = [_]f64{ 0.0, -0.0, std.math.nan(f64), std.math.inf(f64), -std.math.inf(f64), 1.5, -2.25, 3.0 };
    const sp32 = [_]f32{ 0.0, -0.0, std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32), 1.5, -2.25, 3.0 };
    for (buf, 0..) |*w, i| w.* = switch (fill) {
        .i64 => if (r.boolean()) @bitCast(r.intRangeAtMost(i64, -1000, 1000)) else r.int(u64) ^ i,
        .f64 => @bitCast(@as(f64, @floatFromInt(r.intRangeAtMost(i32, -64, 64))) / 8),
        .f32 => @as(u64, @as(u32, @bitCast(@as(f32, @floatFromInt(r.intRangeAtMost(i32, -64, 64))) / 4))) |
            @as(u64, @as(u32, @bitCast(@as(f32, @floatFromInt(r.intRangeAtMost(i32, -64, 64))) / 4))) << 32,
        .f64s => @bitCast(sp64[r.intRangeLessThan(usize, 0, sp64.len)]),
        .f32s => @as(u64, @as(u32, @bitCast(sp32[r.intRangeLessThan(usize, 0, sp32.len)]))) |
            @as(u64, @as(u32, @bitCast(sp32[r.intRangeLessThan(usize, 0, sp32.len)]))) << 32,
    };
}

fn sameFloat(comptime F: type, x: F, y: F) bool {
    const U = @Int(.unsigned, @bitSizeOf(F));
    return @as(U, @bitCast(x)) == @as(U, @bitCast(y)) or (std.math.isNan(x) and std.math.isNan(y));
}

fn wrapper(addr: usize, params: u32) !ir.Func {
    var w = ir.Func.init(gpa);
    errdefer w.deinit();
    w.params = params;
    w.num_vregs = params;
    _ = try w.addBlock("e");
    const args = try w.alloc().alloc(ir.VReg, params);
    for (args, 0..) |*a, i| a.* = @intCast(i);
    try w.emit(0, .{ .call = .{ .dst = params, .addr = addr, .args = args } });
    try w.emit(0, .{ .ret = params });
    return w;
}

fn exec(f: *const ir.Func, args: []const i64, ptrs: u8, buf: *align(16) Buf, steps: usize) anyerror!i64 {
    var a: [8]i64 = undefined;
    for (args, 0..) |x, i| a[i] = if (ptrs >> @intCast(i) & 1 != 0) @as(i64, @intCast(@intFromPtr(buf))) + x else x;
    return ir.interpret(f, a[0..args.len], steps);
}

fn printF(name: []const u8, f: *const ir.Func) void {
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(gpa);
    f.print(gpa, &out) catch return;
    std.debug.print("-- {s}:\n{s}\n", .{ name, out.items });
}

pub const Opt = struct { min_vec: u32 = 1, target: opt.VectorTarget = opt.VectorTarget.host, profile: ?*const opt.Profile = null, twice: bool = false };

/// Reference (interpreted original) vs: interpreted .o2, native original, native .o1, native .o2.
pub var vec_count: usize = 0;
pub var run_count: usize = 0;
pub var native_count: usize = 0;
pub fn check(src: []const u8, ptrs: u8, sets: []const []const i64, fill: Fill, o: Opt) !void {
    var ref = try ir.parse(gpa, src);
    defer ref.deinit();
    var fs: [2]ir.Func = undefined;
    var st: opt.Stats = undefined;
    for (&fs, [_]opt.Level{ .o1, .o2 }) |*f, lv| {
        f.* = try ir.parse(gpa, src);
        const s = try opt.optimize(gpa, f, .{ .level = lv, .vector = o.target, .verify_each = true, .profile = o.profile });
        if (lv == .o2) st = s;
        if (o.twice) {
            const s2 = try opt.optimize(gpa, f, .{ .level = lv, .vector = o.target, .verify_each = true, .profile = o.profile });
            if (lv == .o2) st.vectorized_loops += s2.vectorized_loops;
        }
    }
    defer for (&fs) |*f| f.deinit();
    if (st.vectorized_loops < o.min_vec) {
        printF("o2 (not vectorized)", &fs[1]);
        return error.NotVectorized;
    }
    vec_count += st.vectorized_loops;
    var natives: [3]?JitRuntime.Function = @splat(null);
    var wraps: [3]?ir.Func = @splat(null);
    defer for (&natives, &wraps) |*n, *w| {
        if (n.*) |x| x.release();
        if (w.*) |*x| x.deinit();
    };
    if (can_run) for ([_]*const ir.Func{ &ref, &fs[0], &fs[1] }, 0..) |f, i| {
        var cb = CodeBuffer.init(gpa);
        defer cb.deinit();
        _ = try host_compiler.compile(gpa, f, &cb, .{});
        natives[i] = try JitRuntime.add(&cb);
        wraps[i] = try wrapper(@intFromPtr(natives[i].?.mem.ptr), f.params);
    };
    const names = [_][]const u8{ "o2-interp", "native-o0", "native-o1", "native-o2" };
    const S = struct {
        var want_buf: Buf align(16) = undefined;
        var got_buf: Buf align(16) = undefined;
        var tmpl: [8]Buf align(16) = undefined;
        var tmpl_fill: ?Fill = null;
    };
    const want_buf = &S.want_buf;
    const got_buf = &S.got_buf;
    if (S.tmpl_fill != fill) {
        for (&S.tmpl, 0..) |*b, i| fillBuf(b, i, fill);
        S.tmpl_fill = fill;
    }
    for (sets, 0..) |args, k| {
        want_buf.* = S.tmpl[k % 8];
        const want = exec(&ref, args, ptrs, want_buf, 10_000) catch |e| {
            if (e == error.StepLimit) continue;
            return e;
        };
        run_count += 1;
        const runners = [_]?*const ir.Func{ &fs[1], if (wraps[0]) |*w| w else null, if (wraps[1]) |*w| w else null, if (wraps[2]) |*w| w else null };
        for (runners, names) |rf, name| {
            const f = rf orelse continue;
            if (!std.mem.eql(u8, name, "o2-interp")) native_count += 1;
            got_buf.* = S.tmpl[k % 8];
            const got = try exec(f, args, ptrs, got_buf, 3_000_000);
            var bad = got != want;
            var at: usize = 0;
            for (want_buf, got_buf, 0..) |x, y, i| if (x != y) {
                const ok = switch (fill) {
                    .i64 => false,
                    .f64, .f64s => sameFloat(f64, @bitCast(x), @bitCast(y)),
                    .f32, .f32s => sameFloat(f32, @bitCast(@as(u32, @truncate(x))), @bitCast(@as(u32, @truncate(y)))) and
                        sameFloat(f32, @bitCast(@as(u32, @truncate(x >> 32))), @bitCast(@as(u32, @truncate(y >> 32)))),
                };
                if (!ok and !bad) {
                    bad = true;
                    at = i;
                }
            };
            if (bad) {
                std.debug.print("MISMATCH {s} args {any}: want {d} got {d}; first mem diff word {d}: want {x} got {x}\n", .{ name, args, want, got, at, want_buf[at], got_buf[at] });
                std.debug.print("src:\n{s}\n", .{src});
                printF("o2", &fs[1]);
                return error.Mismatch;
            }
        }
    }
}

pub fn sl(comptime N: usize, comptime M: usize, sets: *const [N][M]i64) [N][]const i64 {
    var out: [N][]const i64 = undefined;
    for (&out, sets) |*o, *s| o.* = s;
    return out;
}

test {
    _ = @import("vec_native_cases.zig");
}

pub fn compileNative(src: []const u8) !JitRuntime.Function {
    var f = try ir.parse(gpa, src);
    defer f.deinit();
    var cb = CodeBuffer.init(gpa);
    defer cb.deinit();
    _ = try host_compiler.compile(gpa, &f, &cb, .{});
    return JitRuntime.add(&cb);
}
