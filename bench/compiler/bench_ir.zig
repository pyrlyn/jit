//! IR compiler benchmark: sum/fact (same IR as ~/Grok/cranelift-vs-pyrlyn-jit and
//! asmjit_compiler_bench.cpp) compiled with `optimize = false` (codegen before the opt passes)
//! and `optimize = true`, installed with JitRuntime.add (one mapping per function) or JitPool.
//! Usage: [BENCH_DUMP=dir] bench-ir
const std = @import("std");
const pj = @import("pyrlyn_jit");
const ir = pj.ir;
const gpa = std.heap.c_allocator;

extern "c" fn clock_gettime_nsec_np(clock: c_int) u64;
inline fn now() u64 {
    return clock_gettime_nsec_np(4); // CLOCK_UPTIME_RAW
}

// sum(v0=p, v1=n): end = p + (n << 3); s = 0; while p <u end { s += *p; p += 8 }
fn buildSum(f: *ir.Func) !void {
    f.params = 2;
    const e = try f.addBlock("e");
    const lp = try f.addBlock("loop");
    const body = try f.addBlock("body");
    const done = try f.addBlock("done");
    try f.emit(e, .{ .@"const" = .{ .dst = 4, .imm = 3 } });
    try f.emit(e, .{ .arith = .{ .op = .shl, .dst = 2, .a = 1, .b = 4 } });
    try f.emit(e, .{ .arith = .{ .op = .add, .dst = 2, .a = 0, .b = 2 } });
    try f.emit(e, .{ .@"const" = .{ .dst = 3, .imm = 0 } });
    try f.emit(e, .{ .@"const" = .{ .dst = 5, .imm = 8 } });
    try f.emit(e, .{ .br = lp });
    try f.emit(lp, .{ .cbr = .{ .cond = .ult, .a = 0, .b = 2, .then = body, .@"else" = done } });
    try f.emit(body, .{ .load = .{ .dst = 6, .base = 0, .disp = 0 } });
    try f.emit(body, .{ .arith = .{ .op = .add, .dst = 3, .a = 3, .b = 6 } });
    try f.emit(body, .{ .arith = .{ .op = .add, .dst = 0, .a = 0, .b = 5 } });
    try f.emit(body, .{ .br = lp });
    try f.emit(done, .{ .ret = 3 });
}

// fact(v0=n): i = 1; r = 1; while i <=u n { r *= i; i += 1 }
fn buildFact(f: *ir.Func) !void {
    f.params = 1;
    const e = try f.addBlock("e");
    const lp = try f.addBlock("loop");
    const body = try f.addBlock("body");
    const done = try f.addBlock("done");
    try f.emit(e, .{ .@"const" = .{ .dst = 1, .imm = 1 } });
    try f.emit(e, .{ .@"const" = .{ .dst = 2, .imm = 1 } });
    try f.emit(e, .{ .@"const" = .{ .dst = 3, .imm = 1 } });
    try f.emit(e, .{ .br = lp });
    try f.emit(lp, .{ .cbr = .{ .cond = .ule, .a = 1, .b = 0, .then = body, .@"else" = done } });
    try f.emit(body, .{ .arith = .{ .op = .mul, .dst = 2, .a = 2, .b = 1 } });
    try f.emit(body, .{ .arith = .{ .op = .add, .dst = 1, .a = 1, .b = 3 } });
    try f.emit(body, .{ .br = lp });
    try f.emit(done, .{ .ret = 2 });
}

fn median(v: []u64) u64 {
    std.mem.sort(u64, v, {}, std.sort.asc(u64));
    return v[v.len / 2];
}

fn dump(dir: ?[:0]const u8, name: []const u8, code: []const u8) void {
    const d = dir orelse return;
    var pbuf: [512]u8 = undefined;
    const path = std.fmt.bufPrintSentinel(&pbuf, "{s}/{s}.bin", .{ d, name }, 0) catch return;
    const fp = std.c.fopen(path, "wb") orelse return;
    _ = std.c.fwrite(code.ptr, 1, code.len, fp);
    _ = std.c.fclose(fp);
}

const N = 1000;

pub fn main() !void {
    const dir: ?[:0]const u8 = if (std.c.getenv("BENCH_DUMP")) |d| std.mem.span(d) else null;
    const n: usize = 10_000_000;
    const data = try gpa.alloc(i64, n);
    for (data, 0..) |*d, i| {
        const x: i64 = @intCast(i);
        d.* = (x *% 2654435761) ^ (x >> 3);
    }
    var cb = pj.CodeBuffer.init(gpa);
    defer cb.deinit();
    var ts: [N]u64 = undefined;
    var tf: [N]u64 = undefined;
    var tp: [N]u64 = undefined;
    const builders = .{ buildSum, buildFact };
    const names = [_][]const u8{ "sum", "fact" };
    for ([_]bool{ false, true }) |opt| {
        const tag = if (opt) "opt" else "unopt";
        var fns: [2]pj.JitRuntime.Function = undefined;
        inline for (builders, 0..) |build, k| {
            var st: pj.aarch64_compiler.Stats = .{};
            for (&ts) |*t| {
                const t0 = now();
                var f = ir.Func.init(gpa);
                try build(&f);
                cb.reset();
                st = try pj.aarch64_compiler.compile(gpa, &f, &cb, .{ .optimize = opt });
                f.deinit();
                t.* = now() - t0;
            }
            for (&tf) |*t| {
                const t0 = now();
                var f = ir.Func.init(gpa);
                try build(&f);
                cb.reset();
                _ = try pj.aarch64_compiler.compile(gpa, &f, &cb, .{ .optimize = opt });
                f.deinit();
                const fun = try pj.JitRuntime.add(&cb);
                t.* = now() - t0;
                fun.release();
            }
            {
                var pool = pj.JitPool.Pool.init(gpa, .{});
                defer pool.deinit();
                for (&tp) |*t| {
                    const t0 = now();
                    var f = ir.Func.init(gpa);
                    try build(&f);
                    cb.reset();
                    _ = try pj.aarch64_compiler.compile(gpa, &f, &cb, .{ .optimize = opt });
                    f.deinit();
                    _ = try pool.add(&cb);
                    t.* = now() - t0;
                }
            }
            fns[k] = try pj.JitRuntime.add(&cb);
            var nb: [64]u8 = undefined;
            dump(dir, try std.fmt.bufPrint(&nb, "pyrlyn_{s}_{s}", .{ tag, names[k] }), cb.code());
            std.debug.print("a64 {s} {s} compile_ns={d} full_runtime_ns={d} full_pool_ns={d} bytes={d} spills={d} frameless={}\n", .{ tag, names[k], median(&ts), median(&tf), median(&tp), cb.code().len, st.spills, st.frameless });
        }
        const sum = fns[0].ptr(*const fn ([*]const i64, i64) callconv(.c) i64);
        const fact = fns[1].ptr(*const fn (i64) callconv(.c) i64);
        var te: [21]u64 = undefined;
        var res: i64 = 0;
        for (&te) |*t| {
            const t0 = now();
            res = sum(data.ptr, @intCast(n));
            t.* = now() - t0;
        }
        std.debug.print("a64 {s} sum exec_ns={d} result={d}\n", .{ tag, median(&te), res });
        var acc: i64 = 0;
        for (&te) |*t| {
            const t0 = now();
            acc = 0;
            var k: i64 = 0;
            while (k < 1_000_000) : (k += 1) {
                var a: i64 = 20 + (k & 1);
                std.mem.doNotOptimizeAway(&a);
                acc +%= fact(a);
            }
            t.* = now() - t0;
        }
        std.debug.print("a64 {s} fact exec_ns={d} result={d}\n", .{ tag, median(&te), acc });
        fns[0].release();
        fns[1].release();

        // x86-64 code from the same IR (compile time and size only; arm64 host).
        inline for (builders, 0..) |build, k| {
            var st: pj.x86_64_compiler.Stats = .{};
            for (&ts) |*t| {
                const t0 = now();
                var f = ir.Func.init(gpa);
                try build(&f);
                cb.reset();
                st = try pj.x86_64_compiler.compile(gpa, &f, &cb, .{ .optimize = opt });
                f.deinit();
                t.* = now() - t0;
            }
            var nb: [64]u8 = undefined;
            dump(dir, try std.fmt.bufPrint(&nb, "pyrlyn_x64_{s}_{s}", .{ tag, names[k] }), cb.code());
            std.debug.print("x64 {s} {s} compile_ns={d} bytes={d} spills={d} frameless={}\n", .{ tag, names[k], median(&ts), cb.code().len, st.spills, st.frameless });
        }
    }
    // Install cost alone: 1000 x 48-byte functions.
    var code: [48]u8 = @splat(0);
    std.mem.writeInt(u32, code[44..48], 0xD65F03C0, .little);
    cb.reset();
    try cb.ensure(48);
    for (0..12) |i| cb.put32(std.mem.readInt(u32, code[i * 4 ..][0..4], .little));
    for (&tf) |*t| {
        const t0 = now();
        const fun = try pj.JitRuntime.add(&cb);
        t.* = now() - t0;
        fun.release();
    }
    var pool = pj.JitPool.Pool.init(gpa, .{});
    defer pool.deinit();
    for (&tp) |*t| {
        const t0 = now();
        _ = try pool.add(&cb);
        t.* = now() - t0;
    }
    std.debug.print("install 48B runtime_add_ns={d} pool_add_ns={d} pool_mapped_bytes={d} (runtime maps 16384 per function)\n", .{ median(&tf), median(&tp), pool.mapped });
}
