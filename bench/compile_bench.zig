//! IR compile benchmark (`zig build bench-compile`): time and memory of the register-allocating
//! Compilers (IR -> regalloc -> machine code) on generated functions of several sizes.
//! Per case: median ns per `compile` call, peak live heap bytes and total bytes / allocations
//! requested during one call (counting allocator over the C allocator).
const std = @import("std");
const builtin = @import("builtin");
const pjit = @import("pyrlyn_jit");
const ir = pjit.ir;

extern "c" fn clock_gettime_nsec_np(clock: c_int) u64;
fn now() u64 {
    if (comptime builtin.os.tag.isDarwin()) return clock_gettime_nsec_np(4);
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}

/// Counting allocator: live bytes, peak live bytes, total requested bytes and call count.
const Counting = struct {
    child: std.mem.Allocator,
    live: usize = 0,
    peak: usize = 0,
    total: usize = 0,
    count: usize = 0,

    fn allocator(c: *Counting) std.mem.Allocator {
        return .{ .ptr = c, .vtable = &.{ .alloc = alloc, .resize = resize, .remap = remap, .free = free } };
    }
    fn grow(c: *Counting, n: usize) void {
        c.live += n;
        c.total += n;
        c.peak = @max(c.peak, c.live);
    }
    fn alloc(ctx: *anyopaque, len: usize, al: std.mem.Alignment, ra: usize) ?[*]u8 {
        const c: *Counting = @ptrCast(@alignCast(ctx));
        const p = c.child.rawAlloc(len, al, ra) orelse return null;
        c.count += 1;
        c.grow(len);
        return p;
    }
    fn resize(ctx: *anyopaque, m: []u8, al: std.mem.Alignment, n: usize, ra: usize) bool {
        const c: *Counting = @ptrCast(@alignCast(ctx));
        if (!c.child.rawResize(m, al, n, ra)) return false;
        if (n > m.len) c.grow(n - m.len) else c.live -= m.len - n;
        return true;
    }
    fn remap(ctx: *anyopaque, m: []u8, al: std.mem.Alignment, n: usize, ra: usize) ?[*]u8 {
        const c: *Counting = @ptrCast(@alignCast(ctx));
        const p = c.child.rawRemap(m, al, n, ra) orelse return null;
        if (n > m.len) c.grow(n - m.len) else c.live -= m.len - n;
        return p;
    }
    fn free(ctx: *anyopaque, m: []u8, al: std.mem.Alignment, ra: usize) void {
        const c: *Counting = @ptrCast(@alignCast(ctx));
        c.child.rawFree(m, al, ra);
        c.live -= m.len;
    }
};

fn helper(a: i64, b: i64) callconv(.c) i64 {
    return a +% b;
}

/// Deterministic random function: `nb` blocks of `ops` ops over `ni` integer and `nf` float
/// vregs, with loads/stores, calls, float ops and forward/backward conditional branches.
fn gen(gpa: std.mem.Allocator, seed: u64, nb: u32, ops: u32, ni: u32, nf: u32) !ir.Func {
    var prng = std.Random.DefaultPrng.init(seed);
    const r = prng.random();
    var f = ir.Func.init(gpa);
    errdefer f.deinit();
    f.params = 4;
    var name_buf: [16]u8 = undefined;
    for (0..nb) |i| _ = try f.addBlock(try std.fmt.bufPrint(&name_buf, "b{d}", .{i}));
    const iv = struct {
        fn int(rr: std.Random, n: u32) ir.VReg {
            return rr.uintLessThan(u32, n);
        }
        fn flt(rr: std.Random, n: u32, base: u32) ir.VReg {
            return base + rr.uintLessThan(u32, n);
        }
    };
    // Entry: define every integer and float vreg so all of them are live from the start.
    for (4..ni) |v| try f.emit(0, .{ .@"const" = .{ .dst = @intCast(v), .imm = @intCast(v * 7) } });
    for (0..nf) |k| try f.emit(0, .{ .fconst = .{ .dst = ni + @as(u32, @intCast(k)), .ty = .f64, .bits = @bitCast(@as(f64, @floatFromInt(k))) } });
    for (0..nb) |bi| {
        const b: u32 = @intCast(bi);
        for (0..ops) |_| {
            const d = 4 + r.uintLessThan(u32, ni - 4); // keep params intact (base address)
            switch (r.uintLessThan(u8, 16)) {
                0...6 => try f.emit(b, .{ .arith = .{ .op = r.enumValue(ir.Arith), .dst = d, .a = iv.int(r, ni), .b = iv.int(r, ni) } }),
                7 => try f.emit(b, .{ .@"const" = .{ .dst = d, .imm = r.int(i32) } }),
                8 => try f.emit(b, .{ .copy = .{ .dst = d, .src = iv.int(r, ni) } }),
                9 => try f.emit(b, .{ .load = .{ .dst = d, .base = 0, .disp = 8 * @as(i32, @intCast(r.uintLessThan(u32, 16))) } }),
                10 => try f.emit(b, .{ .store = .{ .base = 0, .disp = 8 * @as(i32, @intCast(r.uintLessThan(u32, 16))), .src = iv.int(r, ni) } }),
                11 => {
                    const args = try f.alloc().dupe(ir.VReg, &.{ iv.int(r, ni), iv.int(r, ni) });
                    try f.emit(b, .{ .call = .{ .dst = d, .addr = @intFromPtr(&helper), .args = args } });
                },
                12, 13 => if (nf > 0) {
                    try f.emit(b, .{ .typed = .{ .op = toT(r.enumValue(enum { add, sub, mul, div, min, max })), .ty = .f64, .dst = iv.flt(r, nf, ni), .a = iv.flt(r, nf, ni), .b = iv.flt(r, nf, ni) } });
                },
                14 => if (nf > 0) try f.emit(b, .{ .cvt = .{ .to = .i64, .from = .f64, .dst = d, .src = iv.flt(r, nf, ni) } }),
                else => if (nf > 0) try f.emit(b, .{ .cvt = .{ .to = .f64, .from = .i64, .dst = iv.flt(r, nf, ni), .src = iv.int(r, ni) } }),
            }
        }
        if (b + 1 == nb) {
            try f.emit(b, .{ .ret = iv.int(r, ni) });
        } else {
            const back = if (b > 0) r.uintLessThan(u32, b) else 0; // loops
            try f.emit(b, .{ .cbr = .{ .cond = r.enumValue(ir.CondCode), .a = iv.int(r, ni), .b = iv.int(r, ni), .then = if (r.boolean()) back else b + 1, .@"else" = b + 1 } });
        }
    }
    try f.verify();
    return f;
}

// `r.enumValue` over a local enum, mapped to ir.TArith (float-valid ops only).
fn toT(e: anytype) ir.TArith {
    return std.meta.stringToEnum(ir.TArith, @tagName(e)).?;
}

const Case = struct { name: []const u8, nb: u32, ops: u32, ni: u32, nf: u32 };
const cases = [_]Case{
    .{ .name = "small", .nb = 4, .ops = 12, .ni = 12, .nf = 4 },
    .{ .name = "medium", .nb = 32, .ops = 24, .ni = 48, .nf = 16 },
    .{ .name = "large", .nb = 200, .ops = 32, .ni = 160, .nf = 32 },
    .{ .name = "huge", .nb = 800, .ops = 40, .ni = 400, .nf = 64 },
};

const Backend = enum { aarch64, x86_64, riscv64 };

fn compileOnce(gpa: std.mem.Allocator, be: Backend, f: *const ir.Func) !usize {
    var cb = pjit.CodeBuffer.init(gpa);
    defer cb.deinit();
    switch (be) {
        .aarch64 => _ = try pjit.aarch64_compiler.compile(gpa, f, &cb, .{}),
        .x86_64 => _ = try pjit.x86_64_compiler.compile(gpa, f, &cb, .{}),
        .riscv64 => _ = try pjit.riscv_compiler.compile(gpa, f, &cb, .{}),
    }
    return cb.code().len;
}

fn envInt(name: [*:0]const u8, default: usize) usize {
    const p = std.c.getenv(name) orelse return default;
    return std.fmt.parseInt(usize, std.mem.span(p), 10) catch default;
}

pub fn main() !void {
    const gpa = std.heap.c_allocator;
    const budget_ns: u64 = envInt("PJIT_BENCH_MS", 300) * std.time.ns_per_ms;
    std.debug.print("\n{s:<8} {s:<8} {s:>12} {s:>12} {s:>12} {s:>8} {s:>8}\n", .{ "backend", "case", "us/compile", "peak KiB", "total KiB", "allocs", "code B" });
    inline for (std.meta.tags(Backend)) |be| {
        for (cases) |cs| {
            // PJIT_BENCH_ONLY=<backend>/<case> runs one case (for profiling).
            if (std.c.getenv("PJIT_BENCH_ONLY")) |only| {
                var buf: [64]u8 = undefined;
                const want = std.fmt.bufPrint(&buf, "{s}/{s}", .{ @tagName(be), cs.name }) catch continue;
                if (!std.mem.eql(u8, std.mem.span(only), want)) continue;
            }
            // RV64 Compiler is integer-only.
            var f = try gen(gpa, 0x5eed + cs.nb, cs.nb, cs.ops, cs.ni, if (be == .riscv64) 0 else cs.nf);
            defer f.deinit();
            var cnt: Counting = .{ .child = gpa };
            const code_len = compileOnce(cnt.allocator(), be, &f) catch |e| {
                std.debug.print("{s:<8} {s:<8} skipped: {t}\n", .{ @tagName(be), cs.name, e });
                continue;
            };
            // Time: median of samples, each sample one compile; run until the budget is spent.
            var samples: [100001]u64 = undefined;
            var ns: usize = 0;
            const t_end = now() + budget_ns;
            while (ns < samples.len and (ns < 5 or now() < t_end)) : (ns += 1) {
                const t0 = now();
                _ = try compileOnce(gpa, be, &f);
                samples[ns] = now() - t0;
            }
            std.mem.sort(u64, samples[0..ns], {}, std.sort.asc(u64));
            const med = samples[ns / 2];
            std.debug.print("{s:<8} {s:<8} {d:>12.1} {d:>12.1} {d:>12.1} {d:>8} {d:>8}\n", .{
                @tagName(be),                                cs.name,
                @as(f64, @floatFromInt(med)) / 1000.0,       @as(f64, @floatFromInt(cnt.peak)) / 1024.0,
                @as(f64, @floatFromInt(cnt.total)) / 1024.0, cnt.count,
                code_len,
            });
        }
    }
}
