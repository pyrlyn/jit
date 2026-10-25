//! ZigJIT side of the codegen benchmark; mirrors asmjit_bench.cpp instruction-for-instruction.
//! usage: bench_x86_64 [iters] [dump_dir]
const std = @import("std");
const zj = @import("zigjit");
const x = zj.x86_64;
/// Safety layer follows build options: `-Dvalidate` turns on operand validation, `-Dhooks`
/// attaches a CountingHooks instance (the cheapest useful hook) to measure hook overhead.
const bo = struct { const hooks = false; };
const A = x.Assembler;
const Gp = x.Gp;
const Error = x.Error;

const R = [15]Gp{ .rax, .rcx, .rdx, .rbx, .rbp, .rsi, .rdi, .r8, .r9, .r10, .r11, .r12, .r13, .r14, .r15 };
const UNITS = 256;

fn straight(a: anytype) Error!u32 {
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

fn calls(a: anytype) Error!u32 {
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

fn loops(a: anytype) Error!u32 {
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

fn mem(a: anytype) Error!u32 {
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

pub fn main(init: std.process.Init) !void {
    const gpa = std.heap.c_allocator;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    const iters: usize = if (args.len > 1) try std.fmt.parseInt(usize, args[1], 10) else 2000;
    const dump: ?[]const u8 = if (args.len > 2) args[2] else null;
    const names = [_][]const u8{ "straight", "calls", "loops", "mem" };
    const fns = .{ straight, calls, loops, mem };
    inline for (names, fns) |name, f| {
        var cb = zj.CodeBuffer.init(gpa);
        defer cb.deinit();
        const a = A.init(&cb);
        var n: u32 = 0;
        var best: u64 = std.math.maxInt(u64);
        var total: u64 = 0;
        for (0..iters) |_| {
            const t0 = now();
            cb.reset();
            n = try f(a);
            try cb.finalize();
            const ns = now() - t0;
            total += ns;
            best = @min(best, ns);
        }
        const nf: f64 = @floatFromInt(n);
        std.debug.print("zigjit {s:<9} instrs={d} bytes={d} avg_ns_per_instr={d:.2} min_ns_per_instr={d:.2}\n", .{
            name, n, cb.code().len, @as(f64, @floatFromInt(total)) / @as(f64, @floatFromInt(iters)) / nf, @as(f64, @floatFromInt(best)) / nf,
        });
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

