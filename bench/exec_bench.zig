//! Generated-code execution benchmark (`zig build bench-exec`): how fast the machine code from
//! the host IR Compiler runs, per workload, against the same algorithm compiled natively by Zig
//! (ReleaseFast). Compile time is outside the timed region.
//!
//! Workloads: hand-written IR loops over memory (i64/f64/f32 arrays: map, reduction, saxpy),
//! a loop calling a small function, recursive calls, and KXBC v1 programs lowered by bc2jit
//! (stack-machine IR with locals, guarded `div`/`rem`).
//!
//! Variants: `o0` compiles the IR as given. When the library has the IR optimizer
//! (`pyrlyn_jit.opt`), `o2` runs it first and `o2+inline`/`o2+vec` add runtime inlining and
//! loop vectorization. Every variant's result is checked against the native result before timing.
//!
//! Env: `PJIT_BENCH_MS` (time budget per sample, default 60), `PJIT_BENCH_ONLY` (one workload),
//! `PJIT_BENCH_SAMPLES` (default 7).
const std = @import("std");
const builtin = @import("builtin");
const pjit = @import("pyrlyn_jit");
const bc2jit = @import("bc2jit");
const ir = pjit.ir;

extern "c" fn clock_gettime_nsec_np(clock: c_int) u64;
fn now() u64 {
    if (comptime builtin.os.tag.isDarwin()) return clock_gettime_nsec_np(4);
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}

const has_opt = @hasDecl(pjit, "opt");
const host_ok = switch (builtin.cpu.arch) {
    .aarch64 => pjit.build_options.aarch64,
    .x86_64 => pjit.build_options.x86_64,
    else => false,
};

const Fn4 = *const fn (i64, i64, i64, i64) callconv(.c) i64;

// ---- data -----------------------------------------------------------------------------
const n_elems = 4096;
var arr_a: [n_elems]i64 = undefined;
var arr_b: [n_elems]i64 = undefined;
var f64_a: [n_elems]f64 = undefined;
var f64_b: [n_elems]f64 = undefined;
var f64_c: [n_elems]f64 = undefined;
var f32_x: [n_elems]f32 = undefined;
var f32_y: [n_elems]f32 = undefined;

fn initData() void {
    var prng = std.Random.DefaultPrng.init(0x5eed);
    const r = prng.random();
    for (0..n_elems) |i| {
        arr_a[i] = r.int(i32);
        arr_b[i] = r.int(i32);
        f64_a[i] = @floatFromInt(r.intRangeAtMost(i32, -1000, 1000));
        f64_b[i] = @as(f64, @floatFromInt(r.intRangeAtMost(i32, -1000, 1000))) / 8.0;
        f32_x[i] = @as(f32, @floatFromInt(r.intRangeAtMost(i32, -100, 100))) / 4.0;
        f32_y[i] = 0;
    }
}

fn addr(p: anytype) i64 {
    return @bitCast(@intFromPtr(p));
}

// ---- native references ------------------------------------------------------------------
fn nativeSumI64(p: i64, n: i64, _: i64, _: i64) callconv(.c) i64 {
    const a: [*]const i64 = @ptrFromInt(@as(usize, @bitCast(p)));
    var s: i64 = 0;
    var i: usize = 0;
    while (i < @as(usize, @intCast(n))) : (i += 1) s +%= a[i];
    return s;
}
fn nativeAddF64(pa: i64, pb: i64, pc: i64, n: i64) callconv(.c) i64 {
    const a: [*]const f64 = @ptrFromInt(@as(usize, @bitCast(pa)));
    const b: [*]const f64 = @ptrFromInt(@as(usize, @bitCast(pb)));
    const c: [*]f64 = @ptrFromInt(@as(usize, @bitCast(pc)));
    for (0..@intCast(n)) |i| c[i] = a[i] + b[i];
    return n;
}
fn nativeSaxpy(px: i64, py: i64, n: i64, _: i64) callconv(.c) i64 {
    const x: [*]const f32 = @ptrFromInt(@as(usize, @bitCast(px)));
    const y: [*]f32 = @ptrFromInt(@as(usize, @bitCast(py)));
    for (0..@intCast(n)) |i| y[i] = x[i] * 2.5 + y[i];
    return n;
}
fn nativeDotF64(pa: i64, pb: i64, n: i64, _: i64) callconv(.c) i64 {
    const a: [*]const f64 = @ptrFromInt(@as(usize, @bitCast(pa)));
    const b: [*]const f64 = @ptrFromInt(@as(usize, @bitCast(pb)));
    var s: f64 = 0;
    for (0..@intCast(n)) |i| s += a[i] * b[i];
    return @intFromFloat(s);
}
fn square(x: i64) callconv(.c) i64 {
    return x *% x +% 1;
}
fn nativeCalls(n: i64, _: i64, _: i64, _: i64) callconv(.c) i64 {
    var s: i64 = 0;
    var i: i64 = 0;
    while (i < n) : (i += 1) s +%= @call(.never_inline, square, .{i});
    return s;
}
fn nativeFib(n: i64, _: i64, _: i64, _: i64) callconv(.c) i64 {
    return fibRec(n);
}
fn fibRec(n: i64) i64 {
    return if (n < 2) n else fibRec(n - 1) +% fibRec(n - 2);
}
fn nativeKxSum(n: i64, _: i64, _: i64, _: i64) callconv(.c) i64 {
    var s: i64 = 0;
    var i: i64 = 0;
    while (i < n) : (i += 1) s +%= i *% 3 ^ (i >> 2);
    return s;
}
fn nativeKxCollatz(n: i64, _: i64, _: i64, _: i64) callconv(.c) i64 {
    var steps: i64 = 0;
    var k: i64 = 1;
    while (k <= n) : (k += 1) {
        var x = k;
        while (x != 1) {
            x = if (@rem(x, 2) == 0) @divTrunc(x, 2) else x *% 3 +% 1;
            steps += 1;
        }
    }
    return steps;
}
fn nativeKxGcd(n: i64, _: i64, _: i64, _: i64) callconv(.c) i64 {
    var s: i64 = 0;
    var i: i64 = 1;
    while (i <= n) : (i += 1) {
        var a: i64 = i *% 7919;
        var b: i64 = 104729;
        while (b != 0) {
            const t = @rem(a, b);
            a = b;
            b = t;
        }
        s +%= a;
    }
    return s;
}

// Recursive IR calls go through this shim (the IR needs the callee address before the callee
// itself is compiled).
var fib_ptr: ?*const fn (i64) callconv(.c) i64 = null;
fn fibShim(n: i64) callconv(.c) i64 {
    return fib_ptr.?(n);
}

// ---- IR workloads -------------------------------------------------------------------------
const sum_i64_src =
    \\func sum_i64 2
    \\e:
    \\  v2 = const 0
    \\  v3 = const 0
    \\  v4 = const 8
    \\  v5 = const 1
    \\  cbr ge v2 v1 done loop
    \\loop:
    \\  v6 = mul v2 v4
    \\  v7 = add v0 v6
    \\  v8 = load v7 0
    \\  v3 = add v3 v8
    \\  v2 = add v2 v5
    \\  cbr lt v2 v1 loop done
    \\done:
    \\  ret v3
;
const add_f64_src =
    \\func add_f64 4
    \\e:
    \\  v4 = const 0
    \\  v5 = const 8
    \\  v6 = const 1
    \\  cbr ge v4 v3 done loop
    \\loop:
    \\  v7 = mul v4 v5
    \\  v8 = add v0 v7
    \\  v9 = add v1 v7
    \\  v10 = add v2 v7
    \\  v11 = load.f64 v8 0
    \\  v12 = load.f64 v9 0
    \\  v13 = add.f64 v11 v12
    \\  store.f64 v10 0 v13
    \\  v4 = add v4 v6
    \\  cbr lt v4 v3 loop done
    \\done:
    \\  ret v4
;
const saxpy_src =
    \\func saxpy 3
    \\e:
    \\  v3 = const 0
    \\  v4 = const 4
    \\  v5 = const 1
    \\  v6 = fconst.f32 2.5
    \\  cbr ge v3 v2 done loop
    \\loop:
    \\  v7 = mul v3 v4
    \\  v8 = add v0 v7
    \\  v9 = add v1 v7
    \\  v10 = load.f32 v8 0
    \\  v11 = load.f32 v9 0
    \\  v12 = mul.f32 v10 v6
    \\  v13 = add.f32 v12 v11
    \\  store.f32 v9 0 v13
    \\  v3 = add v3 v5
    \\  cbr lt v3 v2 loop done
    \\done:
    \\  ret v3
;
const dot_f64_src =
    \\func dot_f64 3
    \\e:
    \\  v3 = const 0
    \\  v4 = const 8
    \\  v5 = const 1
    \\  v6 = fconst.f64 0
    \\  cbr ge v3 v2 done loop
    \\loop:
    \\  v7 = mul v3 v4
    \\  v8 = add v0 v7
    \\  v9 = add v1 v7
    \\  v10 = load.f64 v8 0
    \\  v11 = load.f64 v9 0
    \\  v12 = mul.f64 v10 v11
    \\  v6 = add.f64 v6 v12
    \\  v3 = add v3 v5
    \\  cbr lt v3 v2 loop done
    \\done:
    \\  v13 = cvt.i64.f64 v6
    \\  ret v13
;
/// `{d}` is the callee address.
const calls_fmt =
    \\func calls 1
    \\e:
    \\  v1 = const 0
    \\  v2 = const 0
    \\  v3 = const 1
    \\  cbr ge v1 v0 done loop
    \\loop:
    \\  v4 = call {d} v1
    \\  v2 = add v2 v4
    \\  v1 = add v1 v3
    \\  cbr lt v1 v0 loop done
    \\done:
    \\  ret v2
;
/// IR body of `square` (what runtime inlining sees for its address).
const square_src =
    \\func square 1
    \\e:
    \\  v1 = mul v0 v0
    \\  v2 = const 1
    \\  v3 = add v1 v2
    \\  ret v3
;
const fib_fmt =
    \\func fib 1
    \\e:
    \\  v1 = const 2
    \\  cbr lt v0 v1 base rec
    \\base:
    \\  ret v0
    \\rec:
    \\  v2 = const 1
    \\  v3 = sub v0 v2
    \\  v4 = call {d} v3
    \\  v5 = sub v3 v2
    \\  v6 = call {d} v5
    \\  v7 = add v4 v6
    \\  ret v7
;

// ---- KXBC assembler (labels resolved to code offsets) ---------------------------------------
const Asm = struct {
    const Item = union(enum) { inst: bc2jit.bytecode.Inst, label: u8, jmp: u8, br: struct { cond: bc2jit.bytecode.Cond, label: u8 } };
    fn size(i: bc2jit.bytecode.Inst) u16 {
        return switch (i) {
            .push_i64 => 9,
            .push_arg, .load_local, .store_local => 2,
            .jmp => 3,
            .br_if => 4,
            else => 1,
        };
    }
    fn build(gpa: std.mem.Allocator, args: u8, items: []const Item) ![]u8 {
        var labels: [16]u16 = @splat(0);
        var off: u16 = 0;
        for (items) |it| switch (it) {
            .inst => |i| off += size(i),
            .label => |l| labels[l] = off,
            .jmp => off += 3,
            .br => off += 4,
        };
        var insts: std.ArrayList(bc2jit.bytecode.Inst) = .empty;
        defer insts.deinit(gpa);
        for (items) |it| switch (it) {
            .inst => |i| try insts.append(gpa, i),
            .label => {},
            .jmp => |l| try insts.append(gpa, .{ .jmp = labels[l] }),
            .br => |b| try insts.append(gpa, .{ .br_if = .{ .cond = b.cond, .target = labels[b.label] } }),
        };
        return bc2jit.bytecode.encode(gpa, .{ .arg_count = args, .insts = insts.items, .version = bc2jit.bytecode.version_v1 });
    }
};

fn I(i: bc2jit.bytecode.Inst) Asm.Item {
    return .{ .inst = i };
}

/// sum_{i<n} (i*3 ^ (i >> 2)), locals: 0 = i, 1 = s.
const kx_sum = [_]Asm.Item{
    .{ .label = 0 },
    I(.{ .load_local = 0 }),
    I(.{ .push_arg = 0 }),
    .{ .br = .{ .cond = .ge, .label = 1 } },
    I(.{ .load_local = 1 }),
    I(.{ .load_local = 0 }),
    I(.{ .push_i64 = 3 }),
    I(.mul),
    I(.{ .load_local = 0 }),
    I(.{ .push_i64 = 2 }),
    I(.sar),
    I(.xor),
    I(.add),
    I(.{ .store_local = 1 }),
    I(.{ .load_local = 0 }),
    I(.{ .push_i64 = 1 }),
    I(.add),
    I(.{ .store_local = 0 }),
    .{ .jmp = 0 },
    .{ .label = 1 },
    I(.{ .load_local = 1 }),
    I(.ret),
};
/// Total Collatz steps for 1..n using `rem`/`div` by 2 (guarded division in the IR).
/// locals: 0 = k, 1 = x, 2 = steps.
const kx_collatz = [_]Asm.Item{
    I(.{ .push_i64 = 1 }), I(.{ .store_local = 0 }),
    .{ .label = 0 }, // outer
    I(.{ .load_local = 0 }),
    I(.{ .push_arg = 0 }),
    .{ .br = .{ .cond = .gt, .label = 5 } },
    I(.{ .load_local = 0 }),
    I(.{ .store_local = 1 }),
    .{ .label = 1 }, // inner
    I(.{ .load_local = 1 }),
    I(.{ .push_i64 = 1 }),
    .{ .br = .{ .cond = .eq, .label = 4 } },
    I(.{ .load_local = 1 }),
    I(.{ .push_i64 = 2 }),
    I(.rem),
    I(.{ .push_i64 = 0 }),
    .{ .br = .{ .cond = .ne, .label = 2 } },
    I(.{ .load_local = 1 }),
    I(.{ .push_i64 = 2 }),
    I(.div),
    I(.{ .store_local = 1 }),
    .{ .jmp = 3 },            .{ .label = 2 }, // odd
    I(.{ .load_local = 1 }),  I(.{ .push_i64 = 3 }),
    I(.mul),                  I(.{ .push_i64 = 1 }),
    I(.add),                  I(.{ .store_local = 1 }),
    .{ .label = 3 },          I(.{ .load_local = 2 }),
    I(.{ .push_i64 = 1 }),    I(.add),
    I(.{ .store_local = 2 }),
    .{ .jmp = 1 },           .{ .label = 4 }, // x == 1
    I(.{ .load_local = 0 }), I(.{ .push_i64 = 1 }),
    I(.add),                 I(.{ .store_local = 0 }),
    .{ .jmp = 0 },           .{ .label = 5 },
    I(.{ .load_local = 2 }), I(.ret),
};
/// sum_{i=1..n} gcd(i*7919, 104729) with `rem` by a variable. locals: 0 = i, 1 = a, 2 = b, 3 = s.
const kx_gcd = [_]Asm.Item{
    I(.{ .push_i64 = 1 }),      I(.{ .store_local = 0 }),
    .{ .label = 0 },            I(.{ .load_local = 0 }),
    I(.{ .push_arg = 0 }),      .{ .br = .{ .cond = .gt, .label = 3 } },
    I(.{ .load_local = 0 }),    I(.{ .push_i64 = 7919 }),
    I(.mul),                    I(.{ .store_local = 1 }),
    I(.{ .push_i64 = 104729 }), I(.{ .store_local = 2 }),
    .{ .label = 1 },            I(.{ .load_local = 2 }),
    I(.{ .push_i64 = 0 }),      .{ .br = .{ .cond = .eq, .label = 2 } },
    I(.{ .load_local = 1 }),    I(.{ .load_local = 2 }),
    I(.rem),                    I(.{ .load_local = 2 }),
    I(.{ .store_local = 1 }),   I(.{ .store_local = 2 }),
    .{ .jmp = 1 },              .{ .label = 2 },
    I(.{ .load_local = 3 }),    I(.{ .load_local = 1 }),
    I(.add),                    I(.{ .store_local = 3 }),
    I(.{ .load_local = 0 }),    I(.{ .push_i64 = 1 }),
    I(.add),                    I(.{ .store_local = 0 }),
    .{ .jmp = 0 },              .{ .label = 3 },
    I(.{ .load_local = 3 }),    I(.ret),
};

// ---- harness -----------------------------------------------------------------------------------
const Workload = struct {
    name: []const u8,
    /// IR text (`{d}` placeholders filled with `callee`), or KXBC items.
    src: ?[]const u8 = null,
    kx: ?[]const Asm.Item = null,
    callee: enum { none, square, fib_shim } = .none,
    native: Fn4,
    args: *const fn () [4]i64,
    /// Restore mutated input before each checked call.
    reset: ?*const fn () void = null,
};

fn argsSum() [4]i64 {
    return .{ addr(&arr_a), n_elems, 0, 0 };
}
fn argsAddF64() [4]i64 {
    return .{ addr(&f64_a), addr(&f64_b), addr(&f64_c), n_elems };
}
fn argsSaxpy() [4]i64 {
    return .{ addr(&f32_x), addr(&f32_y), n_elems, 0 };
}
fn argsDot() [4]i64 {
    return .{ addr(&f64_a), addr(&f64_b), n_elems, 0 };
}
fn argsCalls() [4]i64 {
    return .{ 1000, 0, 0, 0 };
}
fn argsFib() [4]i64 {
    return .{ 20, 0, 0, 0 };
}
fn argsKxSum() [4]i64 {
    return .{ 10000, 0, 0, 0 };
}
fn argsKxCollatz() [4]i64 {
    return .{ 300, 0, 0, 0 };
}
fn argsKxGcd() [4]i64 {
    return .{ 300, 0, 0, 0 };
}
fn resetY() void {
    @memset(&f32_y, 0);
}

const workloads = [_]Workload{
    .{ .name = "ir_sum_i64", .src = sum_i64_src, .native = nativeSumI64, .args = argsSum },
    .{ .name = "ir_add_f64", .src = add_f64_src, .native = nativeAddF64, .args = argsAddF64 },
    .{ .name = "ir_saxpy_f32", .src = saxpy_src, .native = nativeSaxpy, .args = argsSaxpy, .reset = resetY },
    .{ .name = "ir_dot_f64", .src = dot_f64_src, .native = nativeDotF64, .args = argsDot },
    .{ .name = "ir_calls", .src = calls_fmt, .callee = .square, .native = nativeCalls, .args = argsCalls },
    .{ .name = "ir_fib", .src = fib_fmt, .callee = .fib_shim, .native = nativeFib, .args = argsFib },
    .{ .name = "kx_sum", .kx = &kx_sum, .native = nativeKxSum, .args = argsKxSum },
    .{ .name = "kx_collatz", .kx = &kx_collatz, .native = nativeKxCollatz, .args = argsKxCollatz },
    .{ .name = "kx_gcd", .kx = &kx_gcd, .native = nativeKxGcd, .args = argsKxGcd },
};

const Variant = enum { o0, o2, o2_inline, o2_vec };

fn variantName(v: Variant) []const u8 {
    return switch (v) {
        .o0 => "o0",
        .o2 => "o2",
        .o2_inline => "o2+inline",
        .o2_vec => "o2+vec",
    };
}

fn envInt(name: [:0]const u8, default: u64) u64 {
    const v = std.c.getenv(name) orelse return default;
    return std.fmt.parseInt(u64, std.mem.span(v), 10) catch default;
}

/// Median ns per call of `f(args)` over `samples` samples of about `budget_ns` each.
fn timeCalls(f: Fn4, args: [4]i64, samples: usize, budget_ns: u64, reset: ?*const fn () void) f64 {
    // Calibrate iterations per sample.
    var iters: u64 = 1;
    while (true) {
        if (reset) |r| r();
        const t0 = now();
        for (0..iters) |_| std.mem.doNotOptimizeAway(f(args[0], args[1], args[2], args[3]));
        const dt = now() - t0;
        if (dt >= budget_ns / 4 or iters >= 1 << 30) {
            iters = @max(1, iters * budget_ns / @max(dt, 1));
            break;
        }
        iters *= 2;
    }
    var res: [32]f64 = undefined;
    const ns = @min(samples, res.len);
    for (0..ns) |s| {
        if (reset) |r| r();
        const t0 = now();
        for (0..iters) |_| std.mem.doNotOptimizeAway(f(args[0], args[1], args[2], args[3]));
        res[s] = @as(f64, @floatFromInt(now() - t0)) / @as(f64, @floatFromInt(iters));
    }
    std.mem.sort(f64, res[0..ns], {}, std.sort.asc(f64));
    return res[ns / 2];
}

fn compileHost(gpa: std.mem.Allocator, f: *ir.Func, cb: *pjit.CodeBuffer) !void {
    switch (builtin.cpu.arch) {
        .aarch64 => _ = try pjit.aarch64_compiler.compile(gpa, f, cb, .{}),
        .x86_64 => _ = try pjit.x86_64_compiler.compile(gpa, f, cb, .{}),
        else => return error.Unsupported,
    }
}

/// IR resolver for runtime inlining: maps known callee addresses to their IR.
const Callees = struct {
    square: ir.Func,
    fn lookup(ctx: ?*anyopaque, a: usize) ?*const ir.Func {
        const c: *Callees = @ptrCast(@alignCast(ctx.?));
        return if (a == @intFromPtr(&square)) &c.square else null;
    }
};

/// Applies the variant's optimization to `f` (no-op for `o0` or without `pyrlyn_jit.opt`).
fn prepare(gpa: std.mem.Allocator, f: *ir.Func, v: Variant, callees: *Callees) !bool {
    if (v == .o0) return true;
    if (comptime has_opt) {
        const opt = pjit.opt;
        var o: opt.Options = .{ .level = .o2 };
        if (v == .o2_inline) o.resolver = .{ .ctx = callees, .lookup = Callees.lookup };
        if (v == .o2_vec) o.vector = opt.VectorTarget.host;
        _ = try opt.optimize(gpa, f, o);
        return true;
    } else return false;
}

fn buildIr(gpa: std.mem.Allocator, w: Workload) !ir.Func {
    if (w.kx) |items| {
        const bytes = try Asm.build(gpa, 1, items);
        defer gpa.free(bytes);
        var diag: bc2jit.bytecode.Diagnostic = .{};
        const fun = try bc2jit.bytecode.decode(gpa, bytes, &diag);
        defer fun.deinit(gpa);
        return bc2jit.lower_ir.toIr(gpa, fun);
    }
    const text = switch (w.callee) {
        .none => return ir.parse(gpa, w.src.?),
        .square => try std.fmt.allocPrint(gpa, calls_fmt, .{@intFromPtr(&square)}),
        .fib_shim => try std.fmt.allocPrint(gpa, fib_fmt, .{ @intFromPtr(&fibShim), @intFromPtr(&fibShim) }),
    };
    defer gpa.free(text);
    return ir.parse(gpa, text);
}

pub fn main() !void {
    if (comptime !host_ok) {
        std.debug.print("bench-exec: host architecture has no IR Compiler in this build\n", .{});
        return;
    }
    const gpa = std.heap.c_allocator;
    initData();
    const budget = envInt("PJIT_BENCH_MS", 60) * std.time.ns_per_ms;
    const samples = envInt("PJIT_BENCH_SAMPLES", 7);
    const only: ?[]const u8 = if (std.c.getenv("PJIT_BENCH_ONLY")) |p| std.mem.span(p) else null;

    var callees: Callees = .{ .square = try ir.parse(gpa, square_src) };
    defer callees.square.deinit();

    std.debug.print("{s:<14} {s:<10} {s:>12} {s:>12} {s:>8} {s:>8}\n", .{ "workload", "variant", "jit ns/call", "native ns", "x native", "code B" });
    for (workloads) |w| {
        if (only) |o| if (!std.mem.eql(u8, o, w.name)) continue;
        if (w.reset) |r| r();
        const want = w.native(w.args()[0], w.args()[1], w.args()[2], w.args()[3]);
        const native_ns = timeCalls(w.native, w.args(), samples, budget, w.reset);
        for (std.enums.values(Variant)) |v| {
            var f = try buildIr(gpa, w);
            defer f.deinit();
            if (!try prepare(gpa, &f, v, &callees)) continue;
            var cb = pjit.CodeBuffer.init(gpa);
            defer cb.deinit();
            try compileHost(gpa, &f, &cb);
            if (std.c.getenv("PJIT_BENCH_DUMP") != null) {
                var text: std.ArrayList(u8) = .empty;
                defer text.deinit(gpa);
                try f.print(gpa, &text);
                std.debug.print("=== {s} {s} IR\n{s}=== code\n{x}\n", .{ w.name, variantName(v), text.items, cb.code() });
            }
            const fun = try pjit.JitRuntime.add(&cb);
            defer fun.release();
            const entry = fun.ptr(Fn4);
            if (w.callee == .fib_shim) fib_ptr = @ptrCast(entry);
            if (w.reset) |r| r();
            const a = w.args();
            const got = entry(a[0], a[1], a[2], a[3]);
            if (got != want) {
                std.debug.print("{s} {s}: WRONG RESULT {d}, native {d}\n", .{ w.name, variantName(v), got, want });
                return error.WrongResult;
            }
            const ns = timeCalls(entry, a, samples, budget, w.reset);
            std.debug.print("{s:<14} {s:<10} {d:>12.1} {d:>12.1} {d:>8.2} {d:>8}\n", .{ w.name, variantName(v), ns, native_ns, ns / native_ns, cb.code().len });
        }
    }
}
