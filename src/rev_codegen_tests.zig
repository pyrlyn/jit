//! Review fuzz: scalar code generation of the host IR compiler vs `ir.interpret`.
const std = @import("std");
const builtin = @import("builtin");
const ir = @import("core/ir.zig");
const opt = @import("core/opt/opt.zig");
const CodeBuffer = @import("core/code_buffer.zig").CodeBuffer;
const JitRuntime = @import("core/jit_runtime.zig");
const H = @import("rev_helpers.zig");
const t = std.testing;

const is_x86 = builtin.cpu.arch == .x86_64;
const host = switch (builtin.cpu.arch) {
    .x86_64 => @import("x86_64/compiler.zig"),
    .aarch64 => @import("aarch64/compiler.zig"),
    else => void,
};
const can_run = host != void;
const gpa = std.heap.page_allocator;

fn xbit(comptime r: anytype) u64 {
    return @as(u64, 1) << @backingInt(r);
}

const host_opts: []const host.Options = if (is_x86) &.{
    .{},
    .{ .allocatable = xbit(@import("x86_64/assembler.zig").Gp.rbx), .fp_allocatable = 1 },
    .{ .allocatable = xbit(@import("x86_64/assembler.zig").Gp.rcx) | xbit(@import("x86_64/assembler.zig").Gp.rdx) | xbit(@import("x86_64/assembler.zig").Gp.rax), .fp_allocatable = 0b11 },
    .{ .allocatable = xbit(@import("x86_64/assembler.zig").Gp.rax) | xbit(@import("x86_64/assembler.zig").Gp.rsi) | xbit(@import("x86_64/assembler.zig").Gp.r12), .split = false, .split_at_calls = false },
    .{ .algo = .graph },
    .{ .shorten_branches = false },
    .{ .allocatable = xbit(@import("x86_64/assembler.zig").Gp.rax) | xbit(@import("x86_64/assembler.zig").Gp.rcx) | xbit(@import("x86_64/assembler.zig").Gp.rdx) | xbit(@import("x86_64/assembler.zig").Gp.rsi) | xbit(@import("x86_64/assembler.zig").Gp.rdi) },
    .{ .algo = .graph, .allocatable = xbit(@import("x86_64/assembler.zig").Gp.rax) | xbit(@import("x86_64/assembler.zig").Gp.rcx) | xbit(@import("x86_64/assembler.zig").Gp.r8) | xbit(@import("x86_64/assembler.zig").Gp.rbx) },
} else &.{
    .{},
    .{ .allocatable = (1 << 19) | (1 << 20), .fp_allocatable = 1 },
    .{ .allocatable = (1 << 0) | (1 << 9), .fp_allocatable = (1 << 16) | (1 << 17) },
    .{ .allocatable = (1 << 1) | (1 << 2) | (1 << 3) | (1 << 20), .fp_allocatable = (1 << 3) | (1 << 29) },
    .{ .allocatable = (1 << 5) | (1 << 6) | (1 << 7) },
    .{ .allocatable = 0xFE00 },
};

const mem_bytes = 1 << 17;
var mem_ref: [mem_bytes / 8]u64 align(16) = undefined;
var mem_got: [mem_bytes / 8]u64 align(16) = undefined;
var mem_init: [mem_bytes / 8]u64 = undefined;

const int_edges = [_]i64{ 0, 1, -1, 2, -2, 3, 7, 8, 63, 64, 65, 127, 128, -64, 255, 256, 4095, 4096, 4097, -4095, -4096, -4097, 0xFFF000, 0xFFF001, 0x1000000, -0xFFF000, 0xFFFF, 0x10000, -0x10000, 0x7fffffff, 0x80000000, -0x80000000, -0x80000001, 0xffffffff, 0x100000000, 0x5555555555555555, -0x5555555555555556, 0x00FF00FF00FF00FF, 0x0F0F0F0F0F0F0F0F, -0x100000000, 0x7FFFFFFFFFFFFFFE, std.math.minInt(i64), std.math.maxInt(i64), std.math.minInt(i64) + 1, 0x123456789ABCDEF, -0x123456789ABCDEF, 0xFFFF0000FFFF, 0x1234000000000000, -0x1234000000000001, 0x0000FFFF00000000, 1 << 62, -(1 << 62) };
const f64_edges = [_]f64{ 0, -0.0, 1, -1, 0.5, 1.5, 2.5, -2.5, 1e19, -1e19, 9.223372036854775807e18, -9.223372036854775808e18, 9.2233720368547748e18, -9.2233720368547758e18, std.math.nan(f64), -std.math.nan(f64), std.math.inf(f64), -std.math.inf(f64), 1e-300, 3.75, 123456789.75, -123456789.75, 4.5e15, 1.8446744073709552e19, 0.1 };

var max_ops: usize = 30;
const Gen = struct {
    r: std.Random,
    out: std.ArrayList(u8) = .empty,
    params: u32,
    ni: u32, // int pool: params .. params + ni
    // special vregs
    vc: u32, // loop counter
    vone: u32,
    vzero: u32,
    vm4: u32,
    vtwo: u32,
    bases: [4]u32,
    base_k: [4]i64 = .{ 0, 70000, 4096, 32 },
    f64p: u32, // 4 f64 vregs
    f32p: u32, // 4 f32 vregs
    tmp: u32, // temp int vregs (2)
    nb: u32,

    fn p(g: *Gen, comptime fmt: []const u8, args: anytype) !void {
        try g.out.print(gpa, fmt, args);
    }
    fn konst(g: *Gen) i64 {
        return switch (g.r.uintLessThan(u8, 5)) {
            0 => g.r.int(i64),
            1 => g.r.intRangeAtMost(i64, -70000, 70000),
            else => int_edges[g.r.uintLessThan(usize, int_edges.len)],
        };
    }
    fn intDst(g: *Gen) u32 {
        return g.params + g.r.uintLessThan(u32, g.ni);
    }
    fn anyInt(g: *Gen) u32 {
        return switch (g.r.uintLessThan(u8, 12)) {
            0, 1 => g.r.uintLessThan(u32, g.params),
            2 => g.vone,
            3 => g.vzero,
            4 => g.vm4,
            else => g.intDst(),
        };
    }
    fn fv(g: *Gen, f32_: bool) u32 {
        return (if (f32_) g.f32p else g.f64p) + g.r.uintLessThan(u32, 4);
    }
    /// base vreg + disp for an access of `size` bytes.
    fn addr(g: *Gen, size: u32) struct { u32, i64 } {
        const bi = g.r.uintLessThan(usize, 4);
        const k = g.base_k[bi];
        const interesting = [_]i64{ -70000, -4097, -4096, -257, -256, -255, -16, -9, -8, -1, 0, 1, 3, 4, 7, 8, 12, 16, 255, 256, 257, 4095, 4096, 16380, 16384, 32760, 32764, 32768, 65520, 65528, 65536, 70000, 100000 };
        var tries: u32 = 0;
        while (tries < 20) : (tries += 1) {
            const d = interesting[g.r.uintLessThan(usize, interesting.len)] + (if (g.r.uintLessThan(u8, 4) == 0) g.r.intRangeAtMost(i64, -3, 3) else 0);
            const off = d + k;
            if (off >= 0 and off + size <= mem_bytes) return .{ g.bases[bi], d };
        }
        const off = g.r.intRangeAtMost(i64, 0, mem_bytes - size);
        return .{ g.bases[bi], off - k };
    }

    fn op(g: *Gen) !void {
        switch (g.r.uintLessThan(u8, 30)) {
            0, 1 => try g.p("  v{d} = const {d}\n", .{ g.intDst(), g.konst() }),
            2 => try g.p("  v{d} = copy v{d}\n", .{ g.intDst(), g.anyInt() }),
            3...8 => {
                const ops = [_][]const u8{ "add", "sub", "mul", "and", "or", "xor", "shl", "shr", "sar" };
                try g.p("  v{d} = {s} v{d} v{d}\n", .{ g.intDst(), ops[g.r.uintLessThan(usize, ops.len)], g.anyInt(), g.anyInt() });
            },
            9, 10 => {
                // one known operand (block-local constant)
                const ops = [_][]const u8{ "add", "sub", "mul", "and", "or", "xor", "shl", "shr", "sar" };
                const k = g.konst();
                const tv = g.tmp;
                try g.p("  v{d} = const {d}\n", .{ tv, k });
                if (g.r.boolean())
                    try g.p("  v{d} = {s} v{d} v{d}\n", .{ g.intDst(), ops[g.r.uintLessThan(usize, ops.len)], g.anyInt(), tv })
                else
                    try g.p("  v{d} = {s} v{d} v{d}\n", .{ g.intDst(), ops[g.r.uintLessThan(usize, ops.len)], tv, g.anyInt() });
            },
            11, 12 => {
                // division with a safe divisor: (x & -4) | 2, or a known one
                const dv = g.tmp + 1;
                const opn = if (g.r.boolean()) "sdiv" else "srem";
                switch (g.r.uintLessThan(u8, 3)) {
                    0 => {
                        try g.p("  v{d} = and v{d} v{d}\n", .{ dv, g.anyInt(), g.vm4 });
                        try g.p("  v{d} = or v{d} v{d}\n", .{ dv, dv, g.vtwo });
                    },
                    else => {
                        var k = g.konst();
                        if (k == 0 or k == -1) k = 3;
                        try g.p("  v{d} = const {d}\n", .{ dv, k });
                    },
                }
                if (g.r.uintLessThan(u8, 3) == 0) {
                    try g.p("  v{d} = const {d}\n", .{ g.tmp, g.konst() });
                    try g.p("  v{d} = {s} v{d} v{d}\n", .{ g.intDst(), opn, g.tmp, dv });
                } else try g.p("  v{d} = {s} v{d} v{d}\n", .{ g.intDst(), opn, g.anyInt(), dv });
            },
            13, 14 => {
                const a = g.addr(8);
                try g.p("  v{d} = load v{d} {d}\n", .{ g.intDst(), a[0], a[1] });
            },
            15, 16 => {
                const a = g.addr(8);
                if (g.r.uintLessThan(u8, 3) == 0) {
                    try g.p("  v{d} = const {d}\n", .{ g.tmp, g.konst() });
                    try g.p("  store v{d} {d} v{d}\n", .{ a[0], a[1], g.tmp });
                } else try g.p("  store v{d} {d} v{d}\n", .{ a[0], a[1], g.anyInt() });
            },
            17 => {
                const f32_ = g.r.boolean();
                const a = g.addr(if (f32_) 4 else 8);
                if (g.r.boolean())
                    try g.p("  v{d} = load.{s} v{d} {d}\n", .{ g.fv(f32_), if (f32_) "f32" else "f64", a[0], a[1] })
                else
                    try g.p("  store.{s} v{d} {d} v{d}\n", .{ if (f32_) "f32" else "f64", a[0], a[1], g.fv(f32_) });
            },
            18 => {
                const f32_ = g.r.boolean();
                const x = f64_edges[g.r.uintLessThan(usize, f64_edges.len)];
                const bits: u64 = if (f32_) @as(u32, @bitCast(@as(f32, @floatCast(x)))) else @bitCast(x);
                try g.fconst(g.fv(f32_), f32_, bits);
            },
            19, 20 => {
                const f32_ = g.r.boolean();
                try g.p("  v{d} = cvt.i64.{s} v{d}\n", .{ g.intDst(), if (f32_) "f32" else "f64", g.fv(f32_) });
            },
            21 => {
                const f32_ = g.r.boolean();
                try g.p("  v{d} = cvt.{s}.i64 v{d}\n", .{ g.fv(f32_), if (f32_) "f32" else "f64", g.anyInt() });
            },
            22 => {
                const f32_ = g.r.boolean();
                try g.p("  v{d} = cvt.{s}.{s} v{d}\n", .{ g.fv(f32_), if (f32_) "f32" else "f64", if (f32_) "f64" else "f32", g.fv(!f32_) });
            },
            23 => {
                const f32_ = g.r.boolean();
                const ops = [_][]const u8{ "add", "sub", "mul", "div", "min", "max" };
                const ty = if (f32_) "f32" else "f64";
                try g.p("  v{d} = {s}.{s} v{d} v{d}\n", .{ g.fv(f32_), ops[g.r.uintLessThan(usize, ops.len)], ty, g.fv(f32_), g.fv(f32_) });
            },
            24 => {
                const f32_ = g.r.boolean();
                const cs = [_][]const u8{ "eq", "ne", "lt", "le", "gt", "ge" };
                try g.p("  v{d} = fcmp.{s} {s} v{d} v{d}\n", .{ g.intDst(), if (f32_) "f32" else "f64", cs[g.r.uintLessThan(usize, cs.len)], g.fv(f32_), g.fv(f32_) });
            },
            25, 26 => {
                const n = g.r.uintAtMost(usize, 16);
                try g.p("  v{d} = call {d}", .{ g.intDst(), H.addr(n) });
                for (0..n) |_| {
                    if (g.r.uintLessThan(u8, 4) == 0) {
                        // a constant argument via a global constant vreg
                        const c = [_]u32{ g.vone, g.vzero, g.vm4, g.vtwo };
                        try g.p(" v{d}", .{c[g.r.uintLessThan(usize, 4)]});
                    } else try g.p(" v{d}", .{g.anyInt()});
                }
                try g.p("\n", .{});
            },
            else => {
                const ops = [_][]const u8{ "add", "sub", "xor", "shl", "sar", "shr" };
                try g.p("  v{d} = {s} v{d} v{d}\n", .{ g.intDst(), ops[g.r.uintLessThan(usize, ops.len)], g.anyInt(), g.anyInt() });
            },
        }
    }
    fn fconst(g: *Gen, v: u32, f32_: bool, bits: u64) !void {
        // fconst text format: parse a float literal; use the bit pattern via a hex float when possible
        const x: f64 = if (f32_) @as(f32, @bitCast(@as(u32, @truncate(bits)))) else @bitCast(bits);
        if (std.math.isNan(x)) {
            try g.p("  v{d} = fconst.{s} nan\n", .{ v, if (f32_) "f32" else "f64" });
        } else if (std.math.isInf(x)) {
            try g.p("  v{d} = fconst.{s} {s}inf\n", .{ v, if (f32_) "f32" else "f64", if (x < 0) "-" else "" });
        } else try g.p("  v{d} = fconst.{s} {e}\n", .{ v, if (f32_) "f32" else "f64", x });
    }

    fn cmpOperand(g: *Gen) !u32 {
        if (g.r.uintLessThan(u8, 2) == 0) {
            try g.p("  v{d} = const {d}\n", .{ g.tmp, g.konst() });
            return g.tmp;
        }
        return g.anyInt();
    }

    fn program(g: *Gen) !void {
        try g.p("func fz {d}\n", .{g.params});
        try g.p("b0:\n", .{});
        try g.p("  v{d} = const 1\n  v{d} = const 0\n  v{d} = const -4\n  v{d} = const 2\n", .{ g.vone, g.vzero, g.vm4, g.vtwo });
        try g.p("  v{d} = const {d}\n", .{ g.vc, g.r.intRangeAtMost(i64, 1, 4) });
        for (g.bases[1..], g.base_k[1..]) |bv, k| {
            try g.p("  v{d} = const {d}\n  v{d} = add v0 v{d}\n", .{ g.tmp, k, bv, g.tmp });
        }
        for (0..4) |i| {
            try g.fconst(g.f64p + @as(u32, @intCast(i)), false, @bitCast(f64_edges[g.r.uintLessThan(usize, f64_edges.len)]));
            try g.fconst(g.f32p + @as(u32, @intCast(i)), true, @as(u32, @bitCast(@as(f32, @floatCast(f64_edges[g.r.uintLessThan(usize, f64_edges.len)])))));
        }
        var bi: u32 = 0;
        while (bi < g.nb) : (bi += 1) {
            if (bi != 0) try g.p("b{d}:\n", .{bi});
            for (0..g.r.intRangeAtMost(usize, 1, max_ops)) |_| try g.op();
            const last = bi + 1 == g.nb;
            const kind: u8 = if (last) 0 else g.r.uintLessThan(u8, 10);
            switch (kind) {
                0 => {
                    // observe floats and ints
                    var d: i64 = 8 * g.r.intRangeAtMost(i64, 0, 64);
                    for (0..4) |i| {
                        try g.p("  store.f64 v0 {d} v{d}\n", .{ d, g.f64p + @as(u32, @intCast(i)) });
                        d += 8;
                        try g.p("  store.f32 v0 {d} v{d}\n", .{ d, g.f32p + @as(u32, @intCast(i)) });
                        d += 4;
                    }
                    const acc = g.tmp;
                    try g.p("  v{d} = const 0\n", .{acc});
                    var v = g.params;
                    while (v < g.params + g.ni) : (v += 1) {
                        try g.p("  v{d} = mul v{d} v{d}\n  v{d} = xor v{d} v{d}\n", .{ acc, acc, g.vm4, acc, acc, v });
                    }
                    try g.p("  ret v{d}\n", .{acc});
                },
                1 => try g.p("  ret v{d}\n", .{g.anyInt()}),
                2, 3 => try g.p("  br b{d}\n", .{g.r.intRangeLessThan(u32, bi + 1, g.nb)}),
                4, 5 => if (bi >= 1) {
                    // loop back edge
                    try g.p("  v{d} = sub v{d} v{d}\n", .{ g.vc, g.vc, g.vone });
                    try g.p("  cbr gt v{d} v{d} b{d} b{d}\n", .{ g.vc, g.vzero, g.r.intRangeAtMost(u32, 1, bi), g.r.intRangeLessThan(u32, bi + 1, g.nb) });
                } else try g.p("  br b{d}\n", .{bi + 1}),
                else => {
                    const conds = [_][]const u8{ "eq", "ne", "lt", "le", "gt", "ge", "ult", "ule", "ugt", "uge" };
                    const c = conds[g.r.uintLessThan(usize, conds.len)];
                    const a = try g.cmpOperand();
                    const b = if (a == g.tmp) g.anyInt() else try g.cmpOperand();
                    const swap = g.r.boolean();
                    try g.p("  cbr {s} v{d} v{d} b{d} b{d}\n", .{ c, if (swap) b else a, if (swap) a else b, g.r.intRangeLessThan(u32, bi + 1, g.nb), g.r.intRangeLessThan(u32, bi + 1, g.nb) });
                },
            }
        }
    }
};

fn genSrc(r: std.Random) ![]u8 {
    const params = r.intRangeAtMost(u32, 1, 16);
    const big = r.uintLessThan(u8, 4) == 0;
    const ni = if (big) r.intRangeAtMost(u32, 40, 140) else r.intRangeAtMost(u32, 4, 40);
    max_ops = if (big) 80 else 30;
    var next = params + ni;
    var g: Gen = .{ .r = r, .params = params, .ni = ni, .vc = 0, .vone = 0, .vzero = 0, .vm4 = 0, .vtwo = 0, .bases = undefined, .f64p = 0, .f32p = 0, .tmp = 0, .nb = r.intRangeAtMost(u32, 1, if (big) 16 else 8) };
    g.vc = next;
    g.vone = next + 1;
    g.vzero = next + 2;
    g.vm4 = next + 3;
    g.vtwo = next + 4;
    g.bases = .{ 0, next + 5, next + 6, next + 7 };
    g.f64p = next + 8;
    g.f32p = next + 12;
    g.tmp = next + 16;
    next += 18;
    try g.program();
    return g.out.toOwnedSlice(gpa);
}

fn sameWord(a: u64, b: u64) bool {
    if (a == b) return true;
    if (std.math.isNan(@as(f64, @bitCast(a))) and std.math.isNan(@as(f64, @bitCast(b)))) return true;
    for ([_]u6{ 0, 32 }) |sh| {
        const p: u32 = @truncate(a >> sh);
        const q: u32 = @truncate(b >> sh);
        if (p != q and !(std.math.isNan(@as(f32, @bitCast(p))) and std.math.isNan(@as(f32, @bitCast(q))))) return false;
    }
    return true;
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

const Outcome = struct { ret: i64, calls: i64 };

fn runInterp(f: *const ir.Func, args: []i64, mem: []u64) !Outcome {
    @memcpy(&mem_got, &mem_init);
    H.calls = 0;
    args[0] = @bitCast(@intFromPtr(&mem_got));
    const r = try ir.interpret(f, args, 1 << 20);
    if (mem.ptr != &mem_got) @memcpy(mem, &mem_got);
    return .{ .ret = r, .calls = H.calls };
}

var quiet = false;
var opts_limit: usize = 100;
fn dbg(comptime fmt: []const u8, args: anytype) void {
    if (!quiet) std.debug.print(fmt, args);
}

fn checkFunc(src: []const u8, f: *const ir.Func, label: []const u8, inputs: []const []i64, want: []const Outcome, want_mems: []const []const u64, o: host.Options, oi: usize) !void {
    var cb = CodeBuffer.init(gpa);
    defer cb.deinit();
    _ = host.compile(gpa, f, &cb, o) catch |e| {
        dbg("compile failed {s} opts #{d}: {s}\n{s}\n", .{ label, oi, @errorName(e), src });
        return e;
    };
    const fun = try JitRuntime.add(&cb);
    defer fun.release();
    var w = try wrapper(@intFromPtr(fun.mem.ptr), f.params);
    defer w.deinit();
    for (inputs, want, want_mems, 0..) |in, wnt, wm, ii| {
        const got = try runInterp(&w, in, &mem_got);
        var ok = got.ret == wnt.ret and got.calls == wnt.calls;
        var bad: usize = 0;
        for (mem_got, wm, 0..) |a, b, i| if (!sameWord(a, b)) {
            ok = false;
            if (bad < 5) dbg("  mem[{d}] got 0x{x} want 0x{x}\n", .{ i, a, b });
            bad += 1;
        };
        if (!ok) {
            dbg("MISMATCH {s} opts #{d} input #{d} {any}: got {d} (calls {d}) want {d} (calls {d})\n", .{ label, oi, ii, in[1..], got.ret, got.calls, wnt.ret, wnt.calls });
            var out: std.ArrayList(u8) = .empty;
            defer out.deinit(gpa);
            f.print(gpa, &out) catch {};
            dbg("--- source:\n{s}\n--- compiled func:\n{s}\n", .{ src, out.items });
            return error.Mismatch;
        }
    }
}

const ninputs = 3;
fn genInputs(r: std.Random) [ninputs][16]i64 {
    var inputs_buf: [ninputs][16]i64 = undefined;
    for (&inputs_buf) |*b| {
        for (b) |*v| v.* = switch (r.uintLessThan(u8, 3)) {
            0 => r.int(i64),
            1 => r.intRangeAtMost(i64, -100, 100),
            else => int_edges[r.uintLessThan(usize, int_edges.len)],
        };
    }
    return inputs_buf;
}

fn checkSrc(src: []const u8, inputs_buf0: [ninputs][16]i64, do_opt: bool) !void {
    var inputs_buf = inputs_buf0;
    var f = ir.parse(gpa, src) catch |e| {
        dbg("parse failed: {s}\n{s}\n", .{ @errorName(e), src });
        return e;
    };
    defer f.deinit();
    f.verify() catch return error.Invalid;
    _ = ir.Func.typeCheck(&f, gpa) catch return error.Invalid;
    var inputs: [ninputs][]i64 = undefined;
    for (&inputs_buf, &inputs) |*b, *s| s.* = b[0..f.params];
    var want: [ninputs]Outcome = undefined;
    var wmems: [ninputs][]u64 = undefined;
    for (&wmems) |*m| m.* = try gpa.alloc(u64, mem_bytes / 8);
    defer for (wmems) |m| gpa.free(m);
    for (inputs, &want, wmems) |in, *w, m| {
        w.* = runInterp(&f, in, m) catch |e| {
            if (e == error.StepLimit) return;
            return e;
        };
    }
    var wm_const: [ninputs][]const u64 = undefined;
    for (&wm_const, wmems) |*a, b| a.* = b;
    for (host_opts[0..@min(host_opts.len, opts_limit)], 0..) |o, oi| try checkFunc(src, &f, "o0", &inputs, &want, &wm_const, o, oi);
    if (do_opt) {
        for ([_]opt.Level{ .o1, .o2 }) |lv| {
            var g = try ir.parse(gpa, src);
            defer g.deinit();
            _ = try opt.optimize(gpa, &g, .{ .level = lv, .vector = opt.VectorTarget.host, .verify_each = true });
            for (host_opts[0..2], 0..) |o, oi| try checkFunc(src, &g, @tagName(lv), &inputs, &want, &wm_const, o, oi);
        }
    }
}

fn getenv(name: [*:0]const u8) ?[*:0]const u8 {
    if (!builtin.link_libc) return null;
    return std.c.getenv(name);
}
fn fuzzN() usize {
    const s = getenv("REV_FUZZ") orelse return 60;
    return std.fmt.parseInt(usize, std.mem.span(s), 10) catch 300;
}
fn fuzzSeed() u64 {
    const s = getenv("REV_SEED") orelse return 0x5eed;
    return std.fmt.parseInt(u64, std.mem.span(s), 10) catch 0x5eed;
}

test "rev_codegen: random scalar differential fuzz" {
    if (!can_run) return error.SkipZigTest;
    var init_rng = std.Random.DefaultPrng.init(42);
    for (&mem_init) |*w| w.* = init_rng.random().int(u64);
    var prng = std.Random.DefaultPrng.init(fuzzSeed());
    const r = prng.random();
    const n = fuzzN();
    for (0..n) |i| {
        const src = try genSrc(r);
        defer gpa.free(src);
        const ins = genInputs(r);
        const do_opt = getenv("REV_NOOPT") == null;
        checkSrc(src, ins, do_opt) catch |e| {
            std.debug.print("program #{d} failed\n", .{i});
            if (e == error.Mismatch) try reduce(src, ins, do_opt);
            return e;
        };
    }
}

/// Removes lines (not labels / `func`) while the program still mismatches.
fn reduce(src0: []const u8, ins: [ninputs][16]i64, do_opt: bool) !void {
    quiet = true;
    defer quiet = false;
    var lines: std.ArrayList([]const u8) = .empty;
    var it = std.mem.splitScalar(u8, src0, '\n');
    while (it.next()) |l| if (l.len != 0) try lines.append(gpa, l);
    var keep: usize = 0;
    for (lines.items, 0..) |l, i| if (std.mem.indexOf(u8, l, " = add v0 ") != null) {
        keep = i + 1;
    };
    var changed = true;
    while (changed) {
        changed = false;
        var i: usize = lines.items.len;
        while (i > keep) {
            i -= 1;
            const l = lines.items[i];
            if (l[0] != ' ') continue; // label
            const removed = lines.orderedRemove(i);
            var buf: std.ArrayList(u8) = .empty;
            defer buf.deinit(gpa);
            for (lines.items) |x| {
                try buf.appendSlice(gpa, x);
                try buf.append(gpa, '\n');
            }
            const still = if (checkSrc(buf.items, ins, do_opt)) |_| false else |e| e == error.Mismatch;
            if (still) changed = true else try lines.insert(gpa, i, removed);
        }
    }
    var buf: std.ArrayList(u8) = .empty;
    for (lines.items) |x| {
        try buf.appendSlice(gpa, x);
        try buf.append(gpa, '\n');
    }
    std.debug.print("=== REDUCED (inputs {any}):\n{s}\n", .{ ins, buf.items });
    quiet = false;
    checkSrc(buf.items, ins, do_opt) catch {};
}

/// `call hN` -> `call <address of helper N>`.
fn resolveHelpers(src: []const u8) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    var i: usize = 0;
    while (i < src.len) {
        if (std.mem.startsWith(u8, src[i..], "call h")) {
            var j = i + 6;
            while (j < src.len and std.ascii.isDigit(src[j])) j += 1;
            const n = try std.fmt.parseInt(usize, src[i + 6 .. j], 10);
            try out.print(gpa, "call {d}", .{H.addr(n)});
            i = j;
        } else {
            try out.append(gpa, src[i]);
            i += 1;
        }
    }
    return out.toOwnedSlice(gpa);
}

fn dumpCode(src: []const u8, path: []const u8, o: host.Options) !void {
    var f = try ir.parse(gpa, src);
    defer f.deinit();
    var cb = CodeBuffer.init(gpa);
    defer cb.deinit();
    _ = try host.compile(gpa, &f, &cb, o);
    const io = std.testing.io;
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = cb.code() });
}

/// Checks a program (`call hN` names helpers) on the host with fixed inputs; dumps the code.
fn repro(src0: []const u8, ins: []const i64, dump: ?[]const u8) !void {
    const src = try resolveHelpers(src0);
    defer gpa.free(src);
    var init_rng = std.Random.DefaultPrng.init(42);
    for (&mem_init) |*w| w.* = init_rng.random().int(u64);
    var b: [ninputs][16]i64 = @splat(@splat(0));
    for (&b) |*row| @memcpy(row[0..ins.len], ins);
    if (dump) |p| try dumpCode(src, p, .{});
    try checkSrc(src, b, false);
}

test "rev_codegen: x86 call result clobbered (reduced fuzz case 1)" {
    if (!can_run) return error.SkipZigTest;
    try repro(
        \\func fz 2
        \\b0:
        \\  v36 = const 1
        \\  v37 = const 0
        \\  v38 = const -4
        \\  v39 = const 2
        \\  v35 = const 2
        \\  v51 = const 70000
        \\  v40 = add v0 v51
        \\  v51 = const 4096
        \\  v41 = add v0 v51
        \\  v51 = const 32
        \\  v42 = add v0 v51
        \\  br b2
        \\b1:
        \\  v8 = call h13 v15 v38 v23 v33 v29 v12 v36 v36 v1 v36 v39 v2 v37
        \\  v24 = add v38 v17
        \\  cbr lt v51 v28 b3 b3
        \\b2:
        \\  br b3
        \\b3:
        \\  v17 = call h4 v37 v18 v18 v4
        \\  br b5
        \\b4:
        \\  cbr gt v35 v37 b4 b6
        \\b5:
        \\  v35 = sub v35 v36
        \\  cbr gt v35 v37 b1 b6
        \\b6:
        \\  v51 = xor v51 v24
        \\  ret v51
        \\
    , &.{ 0, -445999435603779272 }, null);
}

test "rev_codegen: value live into a block that starts with a call is saved around it" {
    if (!can_run) return error.SkipZigTest;
    // v10..v16 are defined in b2 and used in b1 after the call that starts b1 (loop entered
    // at b2): with more call-crossing values than callee-saved registers, some live in
    // caller-saved registers and must be saved around the call.
    try repro(
        \\func f 2
        \\b0:
        \\  v1 = const 3
        \\  v20 = const 0
        \\  v21 = const 0
        \\  v22 = const 1
        \\  v5 = copy v1
        \\  br b2
        \\b1:
        \\  v9 = call h0
        \\  v20 = add v20 v10
        \\  v20 = add v20 v11
        \\  v20 = add v20 v12
        \\  v20 = add v20 v13
        \\  v20 = add v20 v14
        \\  v20 = add v20 v15
        \\  v20 = add v20 v16
        \\  v20 = xor v20 v9
        \\  br b2
        \\b2:
        \\  v10 = mul v5 v5
        \\  v11 = add v10 v5
        \\  v12 = mul v11 v5
        \\  v13 = add v12 v5
        \\  v14 = mul v13 v5
        \\  v15 = add v14 v5
        \\  v16 = mul v15 v5
        \\  v5 = add v5 v0
        \\  v1 = sub v1 v22
        \\  cbr gt v1 v21 b1 b3
        \\b3:
        \\  ret v20
        \\
    , &.{ 0, 12345 }, null);
}

fn bigFrameSrc(n: usize, with_call: bool) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    try out.print(gpa, "func big 16\nb0:\n", .{});
    // n values, each from a parameter, all live until the end
    for (0..n) |i| try out.print(gpa, "  v{d} = add v{d} v{d}\n", .{ 16 + i, (i % 15) + 1, if (i == 0) 1 else 16 + i - 1 });
    if (with_call) {
        try out.print(gpa, "  v{d} = call h16", .{16 + n});
        for (0..16) |i| try out.print(gpa, " v{d}", .{if (i % 2 == 0) i + 1 else 16 + (i * 37) % n});
        try out.print(gpa, "\n", .{});
    } else try out.print(gpa, "  v{d} = const 5\n", .{16 + n});
    try out.print(gpa, "  v{d} = copy v{d}\n", .{ 17 + n, 16 + n });
    var i: usize = 0;
    while (i < n) : (i += 1) try out.print(gpa, "  v{d} = xor v{d} v{d}\n  v{d} = mul v{d} v15\n", .{ 17 + n, 17 + n, 16 + i, 17 + n, 17 + n });
    for (1..16) |p| try out.print(gpa, "  v{d} = add v{d} v{d}\n", .{ 17 + n, 17 + n, p });
    try out.print(gpa, "  ret v{d}\n", .{17 + n});
    return out.toOwnedSlice(gpa);
}

test "rev_codegen: big frames (spill slots past the scaled range, stack params, stack args)" {
    if (!can_run) return error.SkipZigTest;
    const ins = [_]i64{ 0, 3, -5, 7, 11, -13, 17, 19, -23, 29, 31, 37, -41, 43, 47, 53 };
    opts_limit = 4; // not the graph allocator (quadratic on 5000 live values)
    defer opts_limit = 100;
    for ([_]usize{ 20, 600, 5000 }) |n| for ([_]bool{ false, true }) |wc| {
        const src = try bigFrameSrc(n, wc);
        defer gpa.free(src);
        repro(src, &ins, null) catch |e| {
            std.debug.print("big frame n={d} call={}\n", .{ n, wc });
            return e;
        };
    };
}
