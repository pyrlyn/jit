//! Differential tests of the IR optimizer (`core/opt`) end to end, the project's main safety
//! net for it: random programs (`opt.testgen`, several configurations) and hand-written
//! workloads (those of bench/exec_bench.zig and a few more shapes) must give bit-exactly the
//! same result, memory and call sequence through
//!   - `ir.interpret` on the original (the reference),
//!   - `ir.interpret` on `opt.optimize` output at `.o1` and `.o2` (`verify_each`, host vectors),
//!   - the host IR Compiler (AArch64 natively, x86-64 natively or under Rosetta) on the
//!     original and on both optimized functions.
//! Native code runs through a one-op interpreted wrapper (`call <code> v0 ...`), so it gets the
//! same buffer and call log setup as the interpreter (`testgen.Run`). Programs whose reference
//! run hits the step limit are skipped.
//!
//! Env: `PJIT_OPT_FUZZ=<n>` sets the random programs per configuration (default 200; e.g.
//! `PJIT_OPT_FUZZ=5000 zig build test-core -Dtest-filter=opt_tests` for a long local run);
//! `PJIT_OPT_STATS=1` prints what was compared.
const std = @import("std");
const builtin = @import("builtin");
const ir = @import("core/ir.zig");
const opt = @import("core/opt/opt.zig");
const rc = @import("core/reg_classes.zig");
const CodeBuffer = @import("core/code_buffer.zig").CodeBuffer;
const JitRuntime = @import("core/jit_runtime.zig");
const opts = @import("pyrlyn_jit_options");
const testgen = opt.testgen;
const t = std.testing;

const host_compiler = switch (builtin.cpu.arch) {
    .x86_64 => if (opts.x86_64) @import("x86_64/compiler.zig") else void,
    .aarch64 => if (opts.aarch64) @import("aarch64/compiler.zig") else void,
    else => void,
};
const can_run = host_compiler != void;

const max_steps = 20_000;

/// Leak-checking allocator of these tests. Unlike `std.testing.allocator` it records no stack
/// trace per allocation (unwinding dominated this file's Debug runtime by ~10x).
var leak_check: std.heap.SafeAllocator = undefined;
var gpa: std.mem.Allocator = undefined;

fn beginLeakCheck() void {
    leak_check = .init(std.heap.page_allocator, .{ .stack_trace_frames = 0, .canary = 0x0b7e57ed });
    gpa = leak_check.allocator();
}
/// Fails on leaks since `beginLeakCheck` (call it last; the test's
/// `errdefer _ = leak_check.deinit()` then releases a fresh, empty instance).
fn endLeakCheck() !void {
    const leaks = leak_check.deinit();
    if (leaks == 0) return;
    beginLeakCheck();
    std.debug.print("opt_tests: {d} leaked allocations\n", .{leaks});
    return error.MemoryLeakDetected;
}

/// Host machine code for an IR function.
const Native = struct {
    fun: JitRuntime.Function,

    fn compile(f: *const ir.Func) !Native {
        var cb = CodeBuffer.init(gpa);
        defer cb.deinit();
        _ = try host_compiler.compile(gpa, f, &cb, .{});
        return .{ .fun = try JitRuntime.add(&cb) };
    }
    fn release(n: Native) void {
        n.fun.release();
    }
    fn addr(n: Native) usize {
        return @intFromPtr(n.fun.mem.ptr);
    }
};

/// `func w <params>: v<params> = call <addr> v0 ...; ret v<params>`: interpreting it runs the
/// code at `addr` (C ABI) with the interpreter's arguments. Caller deinits.
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

/// The variants every program runs as, besides the reference (interpreted original).
const Variant = enum { o1, o2, native, native_o1, native_o2 };
const levels = [_]opt.Level{ .o1, .o2 };

/// `src` optimized at `level` (checked after every pass); caller deinits.
fn optimized(src: []const u8, level: opt.Level, resolver: ?opt.Resolver) !ir.Func {
    var f = try ir.parse(gpa, src);
    errdefer f.deinit();
    _ = try opt.optimize(gpa, &f, .{ .level = level, .vector = opt.VectorTarget.host, .verify_each = true, .resolver = resolver });
    try f.verify();
    try t.expect(!f.hasBlockParams());
    return f;
}

/// The original and its optimized versions, interpreted and (when the host has an IR
/// Compiler) compiled; `runner(v)` is the function to interpret for variant `v`.
const Program = struct {
    ref: ir.Func,
    opt: [2]ir.Func,
    native: [3]?Native = @splat(null),
    wrap: [3]?ir.Func = @splat(null),

    fn init(src: []const u8, resolver: ?opt.Resolver) !Program {
        var p: Program = .{ .ref = try ir.parse(gpa, src), .opt = undefined };
        errdefer p.ref.deinit();
        p.opt[0] = try optimized(src, .o1, resolver);
        errdefer p.opt[0].deinit();
        p.opt[1] = try optimized(src, .o2, resolver);
        errdefer p.opt[1].deinit();
        if (can_run) {
            errdefer p.deinitNative();
            for ([_]*const ir.Func{ &p.ref, &p.opt[0], &p.opt[1] }, 0..) |f, i| {
                p.native[i] = try Native.compile(f);
                p.wrap[i] = try wrapper(p.native[i].?.addr(), f.params);
            }
        }
        return p;
    }
    fn deinitNative(p: *Program) void {
        for (&p.native, &p.wrap) |*n, *w| {
            if (n.*) |x| x.release();
            if (w.*) |*x| x.deinit();
            n.* = null;
            w.* = null;
        }
    }
    fn deinit(p: *Program) void {
        p.deinitNative();
        p.ref.deinit();
        for (&p.opt) |*f| f.deinit();
    }
    /// What to interpret for `v` (null: no host compiler).
    fn runner(p: *Program, v: Variant) ?*const ir.Func {
        return switch (v) {
            .o1 => &p.opt[0],
            .o2 => &p.opt[1],
            .native => if (p.wrap[0]) |*w| w else null,
            .native_o1 => if (p.wrap[1]) |*w| w else null,
            .native_o2 => if (p.wrap[2]) |*w| w else null,
        };
    }
};

fn printProgram(what: []const u8, src: []const u8, p: ?*const Program) void {
    std.debug.print("opt_tests: {s} on:\n{s}\n", .{ what, src });
    const f = if (p) |q| q else return;
    for (f.opt, [_][]const u8{ "o1", "o2" }) |g, name| {
        var out: std.ArrayList(u8) = .empty;
        defer out.deinit(gpa);
        g.print(gpa, &out) catch return;
        std.debug.print("-- {s}:\n{s}\n", .{ name, out.items });
    }
}

/// What the random-program test exercised.
const Coverage = struct {
    /// (program, input) pairs compared, and those skipped (reference hit the step limit).
    runs: usize = 0,
    skipped: usize = 0,
    /// Variant runs compared with the reference (interpreted and native).
    compared: usize = 0,
    native: usize = 0,
};

/// Checks a random program (`testgen` shape: `v0` data, `v1` buffer) on every variant.
fn checkRandom(src: []const u8, cov: *Coverage) !void {
    var p = Program.init(src, null) catch |e| {
        printProgram("optimize/compile failed", src, null);
        return e;
    };
    defer p.deinit();
    for (testgen.default_inputs) |in| {
        const want = testgen.Run.of(&p.ref, in, max_steps);
        if (want.result) |_| {} else |e| if (e == error.StepLimit) {
            cov.skipped += 1;
            continue;
        }
        cov.runs += 1;
        for (std.enums.values(Variant)) |v| {
            const f = p.runner(v) orelse continue;
            cov.compared += 1;
            cov.native += @intFromBool(v != .o1 and v != .o2);
            want.expectEqual(testgen.Run.of(f, in, 4 * max_steps)) catch |e| {
                std.debug.print("variant {s}, input {d}\n", .{ @tagName(v), in });
                printProgram("mismatch", src, &p);
                return e;
            };
        }
    }
}

/// Programs per random configuration (`PJIT_OPT_FUZZ`).
fn fuzzCount(default: usize) usize {
    if (!builtin.link_libc) return default;
    const s = std.c.getenv("PJIT_OPT_FUZZ") orelse return default;
    return std.fmt.parseInt(usize, std.mem.span(s), 10) catch default;
}

test "opt_tests: random programs agree across interpreter, optimizer levels and host code" {
    beginLeakCheck();
    errdefer _ = leak_check.deinit();
    const Named = struct { name: []const u8, config: testgen.Config };
    const fl = rc.support.f64; // the host compilers lower f64 (testgen's only float type)
    const configs = [_]Named{
        .{ .name = "default", .config = .{ .floats = fl } },
        .{ .name = "no loops", .config = .{ .floats = fl, .loops = false } },
        .{ .name = "integers only", .config = .{ .floats = false } },
        .{ .name = "no memory, no calls", .config = .{ .floats = fl, .memory = false, .calls = false } },
        .{ .name = "no division", .config = .{ .floats = fl, .division = false } },
        .{ .name = "large", .config = .{ .floats = fl, .blocks = 12, .ops = 12 } },
    };
    const n = fuzzCount(200);
    var cov: Coverage = .{};
    var src: std.ArrayList(u8) = .empty;
    errdefer src.deinit(gpa);
    for (configs, 0..) |cf, ci| {
        var prng = std.Random.DefaultPrng.init(0x0b7_7e57 + ci);
        for (0..n) |_| {
            src.clearRetainingCapacity();
            try testgen.program(prng.random(), gpa, cf.config, &src);
            checkRandom(src.items, &cov) catch |e| {
                std.debug.print("configuration: {s}\n", .{cf.name});
                return e;
            };
        }
    }
    if (builtin.link_libc and std.c.getenv("PJIT_OPT_STATS") != null)
        std.debug.print("opt_tests: {d} programs/configuration x {d}: {d} runs ({d} skipped), {d} compared ({d} native)\n", .{ n, configs.len, cov.runs, cov.skipped, cov.compared, cov.native });
    // Guard against a harness that silently checks nothing.
    try t.expect(cov.runs * 4 >= cov.runs + cov.skipped); // testgen loops often never exit
    if (can_run) try t.expectEqual(cov.runs * 5, cov.compared);
    src.deinit(gpa);
    try endLeakCheck();
}

// ---- hand-written workloads ----------------------------------------------------------------

fn square(x: i64) callconv(.c) i64 {
    return x *% x +% 1;
}
fn mix(a: i64, b: i64) callconv(.c) i64 {
    return (a *% 0x9e37) ^ (b >> 3);
}
fn fibRef(n: i64) callconv(.c) i64 {
    return if (n < 2) n else fibRef(n - 1) +% fibRef(n - 2);
}
/// Recursive IR calls go through this shim: the interpreted variants recurse into `fibRef`,
/// each compiled variant into itself.
var fib_ptr: *const fn (i64) callconv(.c) i64 = &fibRef;
fn fibShim(n: i64) callconv(.c) i64 {
    return fib_ptr(n);
}

const square_src =
    \\func square 1
    \\e:
    \\  v1 = mul v0 v0
    \\  v2 = const 1
    \\  v3 = add v1 v2
    \\  ret v3
;

/// Resolver of the `calls` workloads: `square`'s address maps to its IR (runtime inlining).
const Callees = struct {
    square: ir.Func,
    fn lookup(ctx: ?*anyopaque, addr: usize) ?*const ir.Func {
        const self: *Callees = @ptrCast(@alignCast(ctx.?));
        return if (addr == @intFromPtr(&square)) &self.square else null;
    }
};

/// Buffers the workloads read and write: `slots` u64 each, filled per element type.
const slots = 72;
const Elem = enum { i64, f64, f32 };
const Mem = struct {
    buf: [3][slots]u64 align(16),

    fn init(elems: [3]Elem) Mem {
        var m: Mem = undefined;
        for (&m.buf, elems, 0..) |*b, e, k| for (b, 0..) |*x, i| {
            const fi: f64 = @floatFromInt(@as(i64, @intCast(i)) - 20 + @as(i64, @intCast(k)));
            x.* = switch (e) {
                .i64 => @as(u64, i) *% 0x9e3779b97f4a7c15 +% k,
                .f64 => @bitCast(fi * 0.75 + 0.125),
                .f32 => @as(u64, @as(u32, @bitCast(@as(f32, @floatCast(fi * 0.5))))) |
                    @as(u64, @as(u32, @bitCast(@as(f32, @floatCast(fi * -0.25 + 1.0))))) << 32,
            };
        };
        return m;
    }
};

/// An argument: the address of buffer `k`, the trip count, or a constant.
const Arg = union(enum) { buf: u2, n, imm: i64 };

const Workload = struct {
    name: []const u8,
    /// IR text; `{d}` is replaced by the callee's address when `callee != .none`.
    src: []const u8,
    args: []const Arg,
    elems: [3]Elem = .{ .i64, .i64, .i64 },
    callee: enum { none, square, fib } = .none,
    trips: []const i64 = &.{ 0, 1, 2, 3, 7, 16, 33, -5 },
};

const workloads = [_]Workload{
    .{ .name = "sum_i64", .args = &.{ .{ .buf = 0 }, .n }, .src =
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
    },
    .{ .name = "add_f64", .args = &.{ .{ .buf = 0 }, .{ .buf = 1 }, .{ .buf = 2 }, .n }, .elems = .{ .f64, .f64, .f64 }, .src =
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
    },
    .{ .name = "saxpy_f32", .args = &.{ .{ .buf = 0 }, .{ .buf = 1 }, .n }, .elems = .{ .f32, .f32, .i64 }, .src =
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
    },
    .{ .name = "dot_f64", .args = &.{ .{ .buf = 0 }, .{ .buf = 1 }, .n }, .elems = .{ .f64, .f64, .i64 }, .src =
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
    },
    .{ .name = "calls", .args = &.{.n}, .callee = .square, .src =
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
    },
    .{ .name = "fib", .args = &.{.n}, .callee = .fib, .trips = &.{ 0, 1, 2, 3, 10, 15, -3 }, .src =
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
    },
    .{ .name = "top-tested loop", .args = &.{ .{ .buf = 0 }, .n }, .src =
    \\func top 2
    \\e:
    \\  v2 = const 0
    \\  v3 = const 0
    \\  v4 = const 1
    \\  v9 = const 8
    \\  br h
    \\h:
    \\  cbr ge v2 v1 x b
    \\b:
    \\  v5 = mul v2 v9
    \\  v6 = add v0 v5
    \\  v7 = load v6 0
    \\  v10 = const 3
    \\  v8 = sar v7 v10
    \\  v3 = xor v3 v8
    \\  v11 = add v7 v2
    \\  store v6 0 v11
    \\  v2 = add v2 v4
    \\  br h
    \\x:
    \\  ret v3
    },
    .{ .name = "nested loops", .args = &.{ .{ .buf = 0 }, .n }, .src =
    \\func nest 2
    \\e:
    \\  v2 = const 0
    \\  v3 = const 0
    \\  v4 = const 1
    \\  v10 = const 7
    \\  v11 = const 8
    \\  br oh
    \\oh:
    \\  cbr ge v2 v1 x ob
    \\ob:
    \\  v5 = const 0
    \\  br ih
    \\ih:
    \\  cbr ge v5 v2 ox ib
    \\ib:
    \\  v6 = mul v2 v5
    \\  v3 = add v3 v6
    \\  v5 = add v5 v4
    \\  br ih
    \\ox:
    \\  v7 = and v2 v10
    \\  v8 = mul v7 v11
    \\  v9 = add v0 v8
    \\  store v9 0 v3
    \\  v2 = add v2 v4
    \\  br oh
    \\x:
    \\  ret v3
    },
    .{ .name = "loop with a call", .args = &.{ .{ .buf = 2 }, .n, .{ .imm = 5 } }, .elems = .{ .i64, .i64, .f64 }, .src =
    \\func lcall 3
    \\e:
    \\  v3 = const 0
    \\  v4 = const 1
    \\  v5 = fconst.f64 0.5
    \\  v6 = const 0
    \\  v7 = copy v2
    \\  br h
    \\h:
    \\  cbr ge v3 v1 x b
    \\b:
    \\  v8 = call {d} v7 v3
    \\  v6 = add v6 v8
    \\  v9 = cvt.f64.i64 v8
    \\  v5 = add.f64 v5 v9
    \\  v10 = const 8
    \\  v11 = and v3 v4
    \\  v11 = mul v11 v10
    \\  v12 = add v0 v11
    \\  store.f64 v12 0 v5
    \\  v7 = xor v7 v3
    \\  v3 = add v3 v4
    \\  br h
    \\x:
    \\  v13 = cvt.i64.f64 v5
    \\  v13 = add v13 v6
    \\  ret v13
    },
    // Regression (core/regalloc.zig `crossesCall`): `l` starts with a call and is entered only
    // from later blocks, so `v6` and `v10` are live into it from its first position and must
    // not get caller-saved registers.
    .{ .name = "call at the head of a block entered from below", .args = &.{ .n, .{ .imm = 3 } }, .src =
    \\func head 2
    \\e:
    \\  v2 = const 0
    \\  v4 = const 3
    \\  v8 = const 1
    \\  br b
    \\l:
    \\  v3 = call {d} v6 v0
    \\  v9 = add v9 v3
    \\  v2 = add v2 v8
    \\  cbr lt v2 v0 l x
    \\b:
    \\  v6 = add v1 v4
    \\  v10 = mul v1 v1
    \\  br l
    \\x:
    \\  v7 = add v6 v9
    \\  v7 = xor v7 v10
    \\  ret v7
    },
    // Same bug in the entry block: the parameter is live into a call at position 0.
    .{ .name = "call first in the entry block", .args = &.{.n}, .src =
    \\func entry 1
    \\e:
    \\  v1 = call {d} v0 v0
    \\  v2 = add v1 v0
    \\  ret v2
    },
    .{ .name = "block parameters (fib_ssa)", .args = &.{.n}, .trips = &.{ 0, 1, 2, 10, 50, -1 }, .src = ir.fib_ssa_src },
    .{ .name = "block parameters (swap, lost copy)", .args = &.{.n}, .src =
    \\func swp 1
    \\e:
    \\  v1 = const 1
    \\  v2 = const 2
    \\  v3 = const 0
    \\  br h v1 v2 v3
    \\h(v4 v5 v6):
    \\  v7 = const 1
    \\  v8 = add v6 v7
    \\  v9 = mul v4 v2
    \\  cbr lt v8 v0 h:v5:v9:v8 x:v4
    \\x(v10):
    \\  v11 = sub v10 v5
    \\  ret v11
    },
};

/// Interprets `f` (a workload or a wrapper with the same parameters) on fresh buffers.
fn runWorkload(w: *const Workload, f: *const ir.Func, n: i64) struct { r: anyerror!i64, mem: Mem } {
    var mem = Mem.init(w.elems);
    var args: [4]i64 = undefined;
    for (w.args, 0..) |a, i| args[i] = switch (a) {
        .buf => |k| @intCast(@intFromPtr(&mem.buf[k])),
        .n => n,
        .imm => |x| x,
    };
    const r = ir.interpret(f, args[0..w.args.len], 1_000_000);
    return .{ .r = r, .mem = mem };
}

fn checkWorkload(w: *const Workload, callees: *Callees) !void {
    const callee_addr: usize = switch (w.callee) {
        .none => @intFromPtr(&mix),
        .square => @intFromPtr(&square),
        .fib => @intFromPtr(&fibShim),
    };
    const src = if (std.mem.indexOf(u8, w.src, "{d}") == null) try gpa.dupe(u8, w.src) else blk: {
        const addr = try std.fmt.allocPrint(gpa, "{d}", .{callee_addr});
        defer gpa.free(addr);
        break :blk try std.mem.replaceOwned(u8, gpa, w.src, "{d}", addr);
    };
    defer gpa.free(src);
    const resolver: opt.Resolver = .{ .ctx = callees, .lookup = Callees.lookup };
    var p = try Program.init(src, resolver);
    defer p.deinit();
    for (w.trips) |n| {
        fib_ptr = &fibRef;
        const want = runWorkload(w, &p.ref, n);
        const wr = try want.r;
        for (std.enums.values(Variant)) |v| {
            const f = p.runner(v) orelse continue;
            fib_ptr = switch (v) {
                .native => @ptrFromInt(p.native[0].?.addr()),
                .native_o1 => @ptrFromInt(p.native[1].?.addr()),
                .native_o2 => @ptrFromInt(p.native[2].?.addr()),
                else => &fibRef,
            };
            const got = runWorkload(w, f, n);
            errdefer std.debug.print("workload {s}, variant {s}, n = {d}\n", .{ w.name, @tagName(v), n });
            try t.expectEqual(wr, try got.r);
            for (want.mem.buf, got.mem.buf) |x, y| try t.expectEqualSlices(u64, &x, &y);
        }
    }
    fib_ptr = &fibRef;
}

test "opt_tests: hand-written workloads agree across interpreter, optimizer levels and host code" {
    beginLeakCheck();
    errdefer _ = leak_check.deinit();
    var callees: Callees = .{ .square = try ir.parse(gpa, square_src) };
    errdefer callees.square.deinit();
    for (&workloads) |*w| {
        if (!rc.support.f32 and w.elems[0] == .f32) continue;
        checkWorkload(w, &callees) catch |e| {
            std.debug.print("workload {s} failed\n", .{w.name});
            return e;
        };
    }
    callees.square.deinit();
    try endLeakCheck();
}

test "opt_tests: optimized workloads keep their loops free of copies" {
    beginLeakCheck();
    errdefer _ = leak_check.deinit();
    // The counted loops of the bench workloads come out of SSA with every loop-carried value
    // updated in place (the IR Compilers' register allocators see no copies to coalesce).
    for ([_][]const u8{ workloads[0].src, workloads[1].src, workloads[2].src, workloads[3].src }) |src| {
        for (levels) |lv| {
            var f = try optimized(src, lv, null);
            defer f.deinit();
            var loops: usize = 0;
            for (f.blocks.items, 0..) |b, bi| {
                // The loop body: the block whose (non-SSA) conditional branch targets itself.
                const term = b.ops.items[b.ops.items.len - 1];
                if (term != .cbr or (term.cbr.then != bi and term.cbr.@"else" != bi)) continue;
                loops += 1;
                for (b.ops.items) |o| if (o == .copy) {
                    var out: std.ArrayList(u8) = .empty;
                    defer out.deinit(gpa);
                    try f.print(gpa, &out);
                    std.debug.print("copy left in a loop ({s}):\n{s}\n", .{ @tagName(lv), out.items });
                    return error.TestUnexpectedResult;
                };
            }
            // .o2 may add a vector loop in front of the scalar one (its epilogue).
            try t.expect(loops == 1 or (lv == .o2 and loops == 2));
            if (builtin.link_libc and std.c.getenv("PJIT_OPT_STATS") != null) {
                var copies: usize = 0;
                for (f.blocks.items) |b| for (b.ops.items) |o| {
                    copies += @intFromBool(o == .copy);
                };
                std.debug.print("opt_tests: {s} at {s}: {d} blocks, {d} copies\n", .{ f.name, @tagName(lv), f.blocks.items.len, copies });
            }
        }
    }
    try endLeakCheck();
}

// ---- Regressions from the adversarial review ----------------------------------------------

fn revCheck(src: []const u8, inputs: []const []const i64) !void {
    var p = Program.init(src, null) catch |e| {
        printProgram("optimize/compile failed", src, null);
        return e;
    };
    defer p.deinit();
    for (inputs) |in| {
        const want = ir.interpret(&p.ref, in, 1_000_000) catch |e| {
            if (e == error.StepLimit) continue;
            return e;
        };
        for (std.enums.values(Variant)) |v| {
            const f = p.runner(v) orelse continue;
            const got = ir.interpret(f, in, 4_000_000) catch |e| {
                std.debug.print("variant {s} input {any}: error {s}\n", .{ @tagName(v), in, @errorName(e) });
                printProgram("error", src, &p);
                return e;
            };
            if (got != want) {
                std.debug.print("variant {s} input {any}: want {d} got {d}\n", .{ @tagName(v), in, want, got });
                printProgram("mismatch", src, &p);
                return error.TestUnexpectedResult;
            }
        }
    }
}

test "opt_tests: review: entry block parameter that is a function parameter" {
    beginLeakCheck();
    errdefer _ = leak_check.deinit();
    try revCheck(
        \\func ep 1
        \\e(v0):
        \\  v1 = const 1
        \\  v2 = sub v0 v1
        \\  v3 = const 0
        \\  cbr gt v2 v3 e:v2 x
        \\x:
        \\  ret v0
    , &.{ &.{5}, &.{1}, &.{0} });
    try endLeakCheck();
}

/// Callee that really overwrites every caller-saved GP register before returning.
fn trash(a: i64) callconv(.c) i64 {
    switch (builtin.cpu.arch) {
        .x86_64 => asm volatile (
            \\ movq $0x1111, %%rcx
            \\ movq $0x2222, %%rdx
            \\ movq $0x3333, %%rsi
            \\ movq $0x4444, %%rdi
            \\ movq $0x5555, %%r8
            \\ movq $0x6666, %%r9
            \\ movq $0x7777, %%r10
            \\ movq $0x8888, %%r11
            ::: .{ .rcx = true, .rdx = true, .rsi = true, .rdi = true, .r8 = true, .r9 = true, .r10 = true, .r11 = true }),
        .aarch64 => asm volatile (
            \\ mov x1, #0x1111
            \\ mov x2, #0x2222
            \\ mov x3, #0x3333
            \\ mov x4, #0x4444
            \\ mov x5, #0x5555
            \\ mov x6, #0x6666
            \\ mov x7, #0x7777
            \\ mov x8, #0x8888
            \\ mov x9, #0x9999
            \\ mov x10, #0xaaaa
            \\ mov x11, #0xbbbb
            \\ mov x12, #0xcccc
            \\ mov x13, #0xdddd
            \\ mov x14, #0xeeee
            \\ mov x15, #0xffff
            \\ mov x16, #0x1234
            \\ mov x17, #0x4321
            ::: .{ .x1 = true, .x2 = true, .x3 = true, .x4 = true, .x5 = true, .x6 = true, .x7 = true, .x8 = true, .x9 = true, .x10 = true, .x11 = true, .x12 = true, .x13 = true, .x14 = true, .x15 = true, .x16 = true, .x17 = true }),
        else => {},
    }
    return a +% 1;
}

fn revWithAddr(comptime fmt: []const u8, inputs: []const []const i64) !void {
    const src = try std.fmt.allocPrint(gpa, fmt, .{@intFromPtr(&trash)});
    defer gpa.free(src);
    try revCheck(src, inputs);
}

test "opt_tests: review: params live across a call first in the entry block (pressure)" {
    beginLeakCheck();
    errdefer _ = leak_check.deinit();
    try revWithAddr(
        \\func h 8
        \\e:
        \\  v8 = call {d} v0
        \\  v9 = add v0 v1
        \\  v9 = mul v9 v2
        \\  v9 = add v9 v3
        \\  v9 = mul v9 v4
        \\  v9 = add v9 v5
        \\  v9 = mul v9 v6
        \\  v9 = add v9 v7
        \\  v9 = xor v9 v8
        \\  ret v9
    , &.{ &.{ 1, 2, 3, 4, 5, 6, 7, 8 }, &.{ -1, 9, 3, 100, 5, -6, 7, 1 } });
    try endLeakCheck();
}

test "opt_tests: review: values live into a block that starts with a call, entered from below (pressure)" {
    beginLeakCheck();
    errdefer _ = leak_check.deinit();
    try revWithAddr(
        \\func hb 2
        \\e:
        \\  v2 = const 0
        \\  v30 = const 1
        \\  br b
        \\l:
        \\  v3 = call {d} v2
        \\  v20 = add v20 v3
        \\  v2 = add v2 v30
        \\  cbr lt v2 v0 l x
        \\b:
        \\  v10 = add v1 v1
        \\  v11 = mul v1 v10
        \\  v12 = add v11 v1
        \\  v13 = mul v12 v1
        \\  v14 = add v13 v10
        \\  v15 = mul v14 v1
        \\  v16 = add v15 v11
        \\  v17 = mul v16 v12
        \\  v18 = add v17 v13
        \\  br l
        \\x:
        \\  v21 = add v20 v10
        \\  v21 = mul v21 v11
        \\  v21 = add v21 v12
        \\  v21 = mul v21 v13
        \\  v21 = add v21 v14
        \\  v21 = mul v21 v15
        \\  v21 = add v21 v16
        \\  v21 = mul v21 v17
        \\  v21 = add v21 v18
        \\  v21 = xor v21 v1
        \\  ret v21
    , &.{ &.{ 3, 5 }, &.{ 1, -2 }, &.{ 7, 11 } });
    try endLeakCheck();
}

test "opt_tests: review: duplicate block parameter and entry params without predecessors" {
    beginLeakCheck();
    errdefer _ = leak_check.deinit();
    try revCheck(
        \\func dp 2
        \\e:
        \\  v2 = const 1
        \\  br h v0 v1
        \\h(v3 v3):
        \\  v4 = add v3 v2
        \\  ret v4
    , &.{ &.{ 5, 9 }, &.{ -1, 2 } });
    try revCheck(
        \\func ep2 1
        \\e(v0):
        \\  v1 = const 3
        \\  v2 = add v0 v1
        \\  ret v2
    , &.{ &.{5}, &.{-1} });
    try endLeakCheck();
}
