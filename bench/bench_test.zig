//! Benchmark regression tests: `zig build bench-test` (check) / `zig build bench-update` (rewrite baselines).
//! One test per case. Each case: warmup, then SAMPLES samples of BATCH iterations; the metric is the
//! median ns per iteration. Fails loudly when median > baseline * (1 + tolerance).
//! Env: PJIT_BENCH_BASELINE (file path, set by build.zig), PJIT_BENCH_UPDATE=1, PJIT_BENCH_TOL (percent, default 5).
const std = @import("std");
const builtin = @import("builtin");
const pjit = @import("pyrlyn_jit");
const bo = pjit.build_options;

const WARMUP = 200;
const SAMPLES = 15;
const BATCH = 300;

extern "c" fn clock_gettime_nsec_np(clock: c_int) u64;
/// Monotonic ns. Zig 0.17 std has no `std.time.Timer`, so use the clock directly (as bench_x86_64.zig does).
fn now() u64 {
    if (comptime builtin.os.tag.isDarwin()) return clock_gettime_nsec_np(4); // CLOCK_UPTIME_RAW
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}

fn env(name: [*:0]const u8) ?[]const u8 {
    const p = std.c.getenv(name) orelse return null;
    return std.mem.span(p);
}

/// Median ns/iteration of `body(ctx)`.
fn measure(ctx: anytype, comptime body: fn (@TypeOf(ctx)) anyerror!void) !f64 {
    for (0..WARMUP) |_| try body(ctx);
    var s: [SAMPLES]f64 = undefined;
    for (&s) |*v| {
        const t0 = now();
        for (0..BATCH) |_| try body(ctx);
        v.* = @as(f64, @floatFromInt(now() - t0)) / BATCH;
    }
    std.mem.sort(f64, &s, {}, std.sort.asc(f64));
    return s[SAMPLES / 2];
}

// ---- baseline file: lines `name ns_per_iter`, '#' comments ----

fn baselineFor(name: []const u8) !?f64 {
    const path = env("PJIT_BENCH_BASELINE") orelse return error.NoBaselinePath;
    var pbuf: [1024]u8 = undefined;
    const pz = try std.fmt.bufPrintSentinel(&pbuf, "{s}", .{path}, 0);
    const fp = std.c.fopen(pz, "rb") orelse return null;
    defer _ = std.c.fclose(fp);
    var buf: [16384]u8 = undefined;
    const n = std.c.fread(&buf, 1, buf.len, fp);
    var it = std.mem.tokenizeScalar(u8, buf[0..n], '\n');
    while (it.next()) |ln| {
        if (ln.len == 0 or ln[0] == '#') continue;
        var f = std.mem.tokenizeAny(u8, ln, " \t");
        const k = f.next() orelse continue;
        const v = f.next() orelse continue;
        if (std.mem.eql(u8, k, name)) return try std.fmt.parseFloat(f64, v);
    }
    return null;
}

/// Appends `name value` (update mode). bench-update truncates the file first (see build.zig).
fn recordBaseline(name: []const u8, v: f64) !void {
    const path = env("PJIT_BENCH_BASELINE") orelse return error.NoBaselinePath;
    var pbuf: [1024]u8 = undefined;
    const pz = try std.fmt.bufPrintSentinel(&pbuf, "{s}", .{path}, 0);
    const fp = std.c.fopen(pz, "ab") orelse return error.OpenFailed;
    defer _ = std.c.fclose(fp);
    var lb: [256]u8 = undefined;
    const ln = try std.fmt.bufPrint(&lb, "{s} {d:.1}\n", .{ name, v });
    _ = std.c.fwrite(ln.ptr, 1, ln.len, fp);
}

/// Measure and compare. A regression is re-measured up to ATTEMPTS times (best median kept) so a
/// single load spike doesn't fail the run; a persistent regression still fails.
const ATTEMPTS = 3;
/// Default tolerance in percent; cases dominated by mmap/mprotect syscalls (JitRuntime) are
/// noisier on a loaded Mac and use SYSCALL_TOL. PJIT_BENCH_TOL overrides both.
const DEFAULT_TOL = 5.0;
const SYSCALL_TOL = 15.0;
fn tolerance(case_tol: f64) !f64 {
    return if (env("PJIT_BENCH_TOL")) |t| try std.fmt.parseFloat(f64, t) else case_tol;
}

fn bench(name: []const u8, case_tol: f64, ctx: anytype, comptime body: fn (@TypeOf(ctx)) anyerror!void) !void {
    var best = try measure(ctx, body);
    if (env("PJIT_BENCH_UPDATE") != null) {
        // Baseline = median of ATTEMPTS measurements, so one lucky or unlucky run doesn't set it.
        var m: [ATTEMPTS]f64 = undefined;
        m[0] = best;
        for (m[1..]) |*v| v.* = try measure(ctx, body);
        std.mem.sort(f64, &m, {}, std.sort.asc(f64));
        best = m[ATTEMPTS / 2];
    } else {
        const tol = try tolerance(case_tol);
        if (try baselineFor(name)) |base| {
            var i: usize = 1;
            while (i < ATTEMPTS and best > base * (1 + tol / 100)) : (i += 1) best = @min(best, try measure(ctx, body));
        }
    }
    try check(name, best, try tolerance(case_tol));
}

fn check(name: []const u8, measured: f64, tol: f64) !void {
    if (env("PJIT_BENCH_UPDATE") != null) {
        try recordBaseline(name, measured);
        std.debug.print("[bench] {s:<22} recorded baseline {d:.1} ns/iter\n", .{ name, measured });
        return;
    }
    const base = (try baselineFor(name)) orelse {
        std.debug.print("\n!!! BENCH {s}: no baseline (measured {d:.1} ns/iter). Run `zig build bench-update`.\n", .{ name, measured });
        return error.MissingBaseline;
    };
    const delta = (measured - base) / base * 100.0;
    if (delta > tol) {
        std.debug.print(
            \\
            \\!!! BENCH REGRESSION: {s}
            \\!!!   baseline {d:.1} ns/iter, measured {d:.1} ns/iter, delta +{d:.1}% (tolerance {d:.1}%)
            \\!!!   check machine load (uptime) and re-run before trusting this.
            \\
        , .{ name, base, measured, delta, tol });
        return error.BenchRegression;
    }
    const note = if (delta < -tol) "  <-- IMPROVED: run `zig build bench-update` and commit the new baseline" else "";
    std.debug.print("[bench] {s:<22} baseline {d:>9.1}  measured {d:>9.1} ns/iter  {s}{d:.1}%{s}\n", .{ name, base, measured, if (delta >= 0) "+" else "", delta, note });
}

// ---- x86-64 codegen: the six scenarios of bench/bench_x86_64.zig (vs AsmJit), default Assembler ----

const X = struct {
    const x = if (bo.x86_64) pjit.x86_64 else void;
    const S = if (bo.x86_64) @import("bench_x86_64.zig").Scenarios(x.Assembler) else void;
    const Ctx = struct { cb: *pjit.CodeBuffer, a: if (bo.x86_64) x.Assembler else void };
    fn run(comptime name: []const u8) !void {
        if (!bo.x86_64) return error.SkipZigTest;
        var cb = pjit.CodeBuffer.init(std.heap.c_allocator);
        defer cb.deinit();
        const ctx: Ctx = .{ .cb = &cb, .a = x.Assembler.init(&cb) };
        try bench("x86_64_" ++ name, DEFAULT_TOL, ctx, struct {
            fn f(c: Ctx) anyerror!void {
                c.cb.reset();
                _ = try @field(S, name)(c.a);
                try c.cb.finalize();
            }
        }.f);
    }
};

test "bench x86_64_straight" {
    try X.run("straight");
}
test "bench x86_64_calls" {
    try X.run("calls");
}
test "bench x86_64_loops" {
    try X.run("loops");
}
test "bench x86_64_mem" {
    try X.run("mem");
}
test "bench x86_64_avx" {
    try X.run("avx");
}
test "bench x86_64_avx512" {
    try X.run("avx512");
}

// ---- AArch64: emission, JitRuntime add/release, end-to-end emit+map+call ----

const A = if (bo.aarch64) pjit.aarch64 else void;
const ACtx = struct { cb: *pjit.CodeBuffer, a: if (bo.aarch64) A.Assembler else void };

/// 256 copies of the sum-loop from examples/example_aarch64.zig (labels, loads, ALU, branches).
fn a64SumLoops(a: anytype, units: usize) !void {
    for (0..units) |_| {
        const loop = try a.newLabel();
        const done = try a.newLabel();
        try a.movz(.x2, 0, 0);
        try a.bind(loop);
        try a.cbz(.x1, done);
        try a.ldr(.x3, .x0, 0);
        try a.add(.x2, .x2, .x3);
        try a.addImm(.x0, .x0, 8);
        try a.subImm(.x1, .x1, 1);
        try a.b(loop);
        try a.bind(done);
        try a.movImm(.x4, 0x0123_4567_89AB_CDEF);
    }
    try a.mov(.x0, .x2);
    try a.ret();
}

test "bench aarch64_emit" {
    if (!bo.aarch64) return error.SkipZigTest;
    var cb = pjit.CodeBuffer.init(std.heap.c_allocator);
    defer cb.deinit();
    const ctx: ACtx = .{ .cb = &cb, .a = A.Assembler.init(&cb) };
    try bench("aarch64_emit", DEFAULT_TOL, ctx, struct {
        fn f(c: ACtx) anyerror!void {
            c.cb.reset();
            try a64SumLoops(c.a, 256);
            try c.cb.finalize();
        }
    }.f);
}

test "bench jit_runtime_add_release" {
    if (!bo.aarch64) return error.SkipZigTest;
    var cb = pjit.CodeBuffer.init(std.heap.c_allocator);
    defer cb.deinit();
    const a = A.Assembler.init(&cb);
    try a64SumLoops(a, 16);
    try cb.finalize();
    try bench("jit_runtime_add_release", SYSCALL_TOL, &cb, struct {
        fn f(c: *pjit.CodeBuffer) anyerror!void {
            const fun = try pjit.JitRuntime.add(c);
            fun.release();
        }
    }.f);
}

const arr = blk: {
    var v: [64]i64 = undefined;
    for (&v, 0..) |*e, i| e.* = @intCast(i);
    break :blk v;
};

test "bench e2e_aarch64_compile_run" {
    if (!bo.aarch64 or builtin.cpu.arch != .aarch64) return error.SkipZigTest;
    var cb = pjit.CodeBuffer.init(std.heap.c_allocator);
    defer cb.deinit();
    const ctx: ACtx = .{ .cb = &cb, .a = A.Assembler.init(&cb) };
    try bench("e2e_aarch64_compile_run", SYSCALL_TOL, ctx, struct {
        fn f(c: ACtx) anyerror!void {
            c.cb.reset();
            try a64SumLoops(c.a, 1);
            try c.cb.finalize();
            const fun = try pjit.JitRuntime.add(c.cb);
            defer fun.release();
            const r = fun.ptr(*const fn ([*]const i64, i64) callconv(.c) i64)(&arr, arr.len);
            if (r != 63 * 64 / 2) return error.WrongResult;
        }
    }.f);
}
