//! ZigJIT side of the codegen benchmark; mirrors asmjit_bench.cpp instruction-for-instruction.
//! usage: bench_x86_64 [iters] [dump_dir]
const std = @import("std");
const zj = @import("zigjit");
const x = zj.x86_64;
/// Safety layer follows build options: `-Dvalidate` turns on operand validation, `-Dhooks`
/// attaches a CountingHooks instance (the cheapest useful hook) to measure hook overhead.
const bo = zj.build_options;
/// Configurations:
/// - default build options: `x.Assembler` (hooks = runtime `CallbackHooks`, none attached unless
///   env `ZJ_HOOK=1`, which attaches a counting callback through the function pointer);
/// - `-Dbench-counting=true`: comptime `CountingHooks`, always attached (inlined hook).
const A = if (bo.bench_counting) x.AssemblerT(.{ .validate = bo.validate, .Hooks = x.CountingHooks }) else x.Assembler;

var cb_count: usize = 0;
fn countCb(_: ?*anyopaque, _: x.Event) void {
    cb_count += 1;
}
const Gp = x.Gp;
const Error = x.Error || x.vec.Error;

const R = [15]Gp{ .rax, .rcx, .rdx, .rbx, .rbp, .rsi, .rdi, .r8, .r9, .r10, .r11, .r12, .r13, .r14, .r15 };
const UNITS = 256;

/// The four scenarios, instantiated per assembler configuration (timing loop and tests).
fn Scenarios(comptime Asm: type) type {
    return struct {
        pub fn straight(a: Asm) Error!u32 {
            @setEvalBranchQuota(100_000);
            var n: u32 = 0;
            for (0..UNITS) |i| {
                const d = R[i % 15];
                const s = R[(i * 7 + 3) % 15];
                try a.movRI(d, @intCast(i * 1000));
                try a.addRR(d, s);
                try a.subRI(d, 5);
                try a.movRR(s, d);
                try a.addRI(d, 100000);
                try a.subRR(s, d);
                if (i % 8 == 0) {
                    try a.movRI(d, 0x123456789 * @as(i64, @intCast(i)));
                    n += 1;
                }
                n += 6;
            }
            try a.ret();
            return n + 1;
        }

        pub fn calls(a: Asm) Error!u32 {
            @setEvalBranchQuota(100_000);
            var n: u32 = 0;
            const f = try a.newLabel();
            for (0..UNITS) |i| {
                try a.push(.rbx);
                try a.movRR(.rdi, R[i % 15]);
                try a.call(f);
                try a.pop(.rbx);
                try a.callR(.r11);
                n += 5;
            }
            try a.ret();
            try a.bind(f);
            try a.ret();
            return n + 2;
        }

        pub fn loops(a: Asm) Error!u32 {
            @setEvalBranchQuota(100_000);
            var n: u32 = 0;
            for (0..UNITS) |_| {
                const top = try a.newLabel();
                const out = try a.newLabel();
                try a.bind(top);
                try a.cmpRR(.rcx, .rdx);
                try a.jcc(.ge, out);
                try a.addRI(.rax, 1);
                try a.subRI(.rdx, 1);
                try a.jmp(top);
                try a.bind(out);
                n += 5;
            }
            try a.ret();
            return n + 1;
        }

        pub fn mem(a: Asm) Error!u32 {
            @setEvalBranchQuota(100_000);
            var n: u32 = 0;
            for (0..UNITS) |iu| {
                const i: i32 = @intCast(iu);
                try a.movRM(.rax, x.ptrIdx(.rdi, .rcx, .x8, 16 + i));
                try a.movMR(x.ptr(.rsp, 8), .r9);
                try a.lea(.r10, x.ptrIdx(.r12, .r13, .x2, 300 + i * 8));
                try a.aluRM(.add, .rax, x.ptr(.rbp, 0));
                try a.imulRM(.rax, x.ptr(.r13, 8));
                try a.movMI(x.ptr(.rdi, i * 8), i);
                n += 6;
            }
            try a.ret();
            return n + 1;
        }

        /// AVX/AVX2/FMA (VEX) through the table-driven `vec` encoder.
        pub fn avx(a: Asm) Error!u32 {
            const v = x.vec;
            var n: u32 = 0;
            for (0..UNITS) |iu| {
                const i: i32 = @intCast(iu);
                try v.emit(a.buf, .vmovups, .{ v.ymm(@intCast(iu % 16)), v.mem(x.ptr(.rdi, i * 32), 0) }, .{});
                try v.emit(a.buf, .vfmadd231ps, .{ v.ymm(0), v.ymm(1), v.ymm(2) }, .{});
                try v.emit(a.buf, .vaddps, .{ v.ymm(3), v.ymm(4), v.mem(x.ptrIdx(.rsi, .rcx, .x4, 64), 0) }, .{});
                try v.emit(a.buf, .vpshufb, .{ v.xmm(5), v.xmm(6), v.xmm(7) }, .{});
                try v.emit(a.buf, .vmovups, .{ v.mem(x.ptr(.rdx, i * 32), 0), v.ymm(8) }, .{});
                try v.emit(a.buf, .vxorps, .{ v.ymm(9), v.ymm(9), v.ymm(9) }, .{});
                n += 6;
            }
            try a.ret();
            return n + 1;
        }

        /// AVX-512 (EVEX): opmask/zeroing, broadcast, {er}, compare into k, masked store.
        pub fn avx512(a: Asm) Error!u32 {
            const v = x.vec;
            var n: u32 = 0;
            for (0..UNITS) |iu| {
                const i: i32 = @intCast(iu);
                try v.emit(a.buf, .vaddps, .{ v.zmm(@intCast(iu % 32)), v.zmm(1), v.mem(x.ptr(.rdi, i * 64), 0) }, .{ .k = 1, .z = true });
                try v.emit(a.buf, .vfmadd231pd, .{ v.zmm(2), v.zmm(3), v.bcst(v.mem(x.ptr(.rsi, 0), 8), 8) }, .{});
                try v.emit(a.buf, .vpternlogd, .{ v.zmm(4), v.zmm(5), v.zmm(6), 0xAA }, .{});
                try v.emit(a.buf, .vaddps, .{ v.zmm(7), v.zmm(8), v.zmm(9) }, .{ .rc = .rn });
                try v.emit(a.buf, .vcmpps, .{ v.k(2), v.zmm(10), v.zmm(11), 1 }, .{});
                try v.emit(a.buf, .vmovdqu32, .{ v.mem(x.ptr(.rdx, i * 64), 0), v.zmm(12) }, .{ .k = 3 });
                n += 6;
            }
            try a.ret();
            return n + 1;
        }
    };
}
const S = Scenarios(A);

pub fn main(init: std.process.Init) !void {
    @setEvalBranchQuota(100_000);
    const gpa = std.heap.c_allocator;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    const iters: usize = if (args.len > 1) try std.fmt.parseInt(usize, args[1], 10) else 2000;
    const dump: ?[]const u8 = if (args.len > 2) args[2] else null;
    const names = [_][]const u8{ "straight", "calls", "loops", "mem", "avx", "avx512" };
    const fns = .{ S.straight, S.calls, S.loops, S.mem, S.avx, S.avx512 };
    // Scenarios are inlined into the timing loop (as LLVM did on its own before the safety
    // layer); this keeps the comparison with the earlier baseline like-for-like.
    inline for (names, fns) |name, f| {
        var cb = zj.CodeBuffer.init(gpa);
        defer cb.deinit();
        var counter: x.CountingHooks = .{};
        var cbh: x.CallbackHooks = .{ .post_fn = countCb };
        const attach_cb = std.c.getenv("ZJ_HOOK") != null;
        const a = if (comptime bo.bench_counting) A.initHooks(&cb, &counter) else if (comptime A.Hooks == x.CallbackHooks) (if (attach_cb) A.initHooks(&cb, &cbh) else A.init(&cb)) else A.init(&cb);
        cb_count = 0;
        var n: u32 = 0;
        var best: u64 = std.math.maxInt(u64);
        var total: u64 = 0;
        for (0..iters) |_| {
            const t0 = now();
            cb.reset();
            n = try @call(.always_inline, f, .{a});
            try cb.finalize();
            const ns = now() - t0;
            total += ns;
            best = @min(best, ns);
        }
        const nf: f64 = @floatFromInt(n);
        std.debug.print("zigjit {s:<9} instrs={d} bytes={d} avg_ns_per_instr={d:.2} min_ns_per_instr={d:.2}\n", .{
            name, n, cb.code().len, @as(f64, @floatFromInt(total)) / @as(f64, @floatFromInt(iters)) / nf, @as(f64, @floatFromInt(best)) / nf,
        });
        if (comptime bo.bench_counting) std.debug.assert(counter.instructions == n * iters);
        if (comptime A.Hooks == x.CallbackHooks) if (attach_cb) std.debug.assert(cb_count == n * iters);
        if (dump) |d| {
            var pbuf: [512]u8 = undefined;
            const p = try std.fmt.bufPrintZ(&pbuf, "{s}/zigjit_{s}.bin", .{ d, name });
            const fp = std.c.fopen(p, "wb") orelse return error.OpenFailed;
            _ = std.c.fwrite(cb.code().ptr, 1, cb.code().len, fp);
            _ = std.c.fclose(fp);
        }
    }
}

extern "c" fn clock_gettime_nsec_np(clock: c_int) u64;
inline fn now() u64 {
    return clock_gettime_nsec_np(4); // CLOCK_UPTIME_RAW
}

// ---- tests: the harness emits exactly what AsmJit emits (golden files from asmjit_bench) ----

fn expectScenario(comptime Asm: type, f: anytype, golden: []const u8, n_expected: ?u32) !void {
    var cb = zj.CodeBuffer.init(std.testing.allocator);
    defer cb.deinit();
    var counter: x.CountingHooks = .{};
    const a = if (comptime Asm.Hooks != void) Asm.initHooks(&cb, &counter) else Asm.init(&cb);
    // Run twice with reset to check buffer reuse (what the timing loop does).
    for (0..2) |_| {
        cb.reset();
        counter = .{};
        const n = try f(a);
        try cb.finalize();
        try std.testing.expectEqualSlices(u8, golden, cb.code());
        if (n_expected) |e| try std.testing.expectEqual(e, n);
        if (comptime Asm.Hooks != void) {
            try std.testing.expectEqual(@as(usize, n), counter.instructions);
            try std.testing.expectEqual(golden.len, counter.bytes);
        }
    }
}

const goldens = .{
    .{ "straight", @embedFile("ref/asmjit_straight.bin"), 1569 },
    .{ "calls", @embedFile("ref/asmjit_calls.bin"), 1282 },
    .{ "loops", @embedFile("ref/asmjit_loops.bin"), 1281 },
    .{ "mem", @embedFile("ref/asmjit_mem.bin"), 1537 },
};

test "bench scenarios: byte-identical to AsmJit, all safety-layer configurations" {
    inline for (.{
        x.AssemblerT(.{}),
        x.AssemblerT(.{ .validate = true }),
        x.AssemblerT(.{ .validate = true, .Hooks = x.CountingHooks }),
    }) |Asm| {
        inline for (goldens) |g| {
            try expectScenario(Asm, @field(Scenarios(Asm), g[0]), g[1], g[2]);
        }
    }
}

// SIMD scenarios go through `x.vec.emit`, which bypasses assembler hooks, so only the
// hook-free configurations are checked here.
test "SIMD bench scenarios: byte-identical to AsmJit" {
    inline for (.{ x.AssemblerT(.{}), x.AssemblerT(.{ .validate = true }) }) |Asm| {
        inline for (.{
            .{ "avx", @embedFile("ref/asmjit_avx.bin") },
            .{ "avx512", @embedFile("ref/asmjit_avx512.bin") },
        }) |g| try expectScenario(Asm, @field(Scenarios(Asm), g[0]), g[1], 1537);
    }
}
