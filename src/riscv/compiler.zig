//! RISC-V 64 Compiler: IR (`core/ir.zig`) -> linear-scan register allocation
//! (`core/regalloc.zig`) -> RISC-V Assembler -> machine code (LP64, RV64IM).
//!
//! Conventions:
//! - allocatable: t3..t6 (caller-saved) and s1..s11 (callee-saved); a0..a7 (arguments) are
//!   not allocated, so parameter and call-argument moves never conflict; s0 is the frame
//!   pointer; t0 is clobbered by `callAbs` and holds large load/store offsets; t1/t2 are scratch for spilled operands, large
//!   offsets and immediates;
//! - frame: `Assembler.prolog(used callee-saved, out + slots)`: outgoing stack arguments at
//!   [sp + 8i], spill slot k at [sp + out + 8k] (sp never moves inside the body);
//!   one shared epilog label;
//! - RISC-V is three-address, so `dst = a op b` needs no two-address fix-up;
//! - conditional branches use `bFar` (inverted branch over `jal`, +-1 MiB), so block layout
//!   never overflows the 12-bit branch range; le/gt/ule/ugt swap the operands;
//! - parameters and call arguments past a7 use 8-byte stack slots (up to `ir.max_args`);
//!   incoming ones are read at [s0 + 8i] (the caller's sp);
//! - `sdiv`/`srem` by zero follow the hardware (-1 / dividend); the IR leaves them undefined.
const std = @import("std");
const rv = @import("assembler.zig");
const ir = @import("../core/ir.zig");
const ra = @import("../core/regalloc.zig");
const rc = @import("../core/reg_classes.zig");
const CodeBuffer = @import("../core/code_buffer.zig").CodeBuffer;
const X = rv.X;
const CC = rv.CallConv;

const s0: X = .t1;
const s1: X = .t2;
const sp: X = .sp;

fn bit(r: X) ra.RegMask {
    return @as(ra.RegMask, 1) << @backingInt(r);
}
fn maskOf(comptime rs: []const X) ra.RegMask {
    var m: ra.RegMask = 0;
    for (rs) |r| m |= bit(r);
    return m;
}
/// Callee-saved registers the allocator may use (s0 is the frame pointer).
const saved_regs = [_]X{ .s1, .s2, .s3, .s4, .s5, .s6, .s7, .s8, .s9, .s10, .s11 };
pub const callee_saved: ra.RegMask = maskOf(&saved_regs);
pub const allocatable: ra.RegMask = maskOf(&.{ .t3, .t4, .t5, .t6 }) | callee_saved;

pub const Options = struct {
    /// Restrict the allocatable set (spill stress tests).
    allocatable: ra.RegMask = allocatable,
};

pub const Stats = struct { spills: u32 = 0, slots: u32 = 0 };

/// Compiles `f` into `cb` through `a` (an `rv.AssemblerT(...)` RV64 instance on `cb`), not finalized.
/// Functions with block parameters are first rewritten by `ir.Func.lowerBlockParams`.
pub fn lower(backing: std.mem.Allocator, f0: *const ir.Func, a: anytype, opts: Options) !Stats {
    var scratch_bytes: [2048]u8 = undefined;
    var temp_alloc = @import("../core/scratch.zig").Scratch.init(&scratch_bytes, backing);
    const gpa = temp_alloc.allocator();
    try f0.verify();
    var lowered: ?ir.Func = if (f0.hasBlockParams()) try f0.lowerBlockParams(gpa) else null;
    defer if (lowered) |*g| g.deinit();
    const f: *const ir.Func = if (lowered) |*g| g else f0;
    if (f.params > ir.max_args) return error.TooManyParams;
    // Outgoing stack-argument area: the largest number of stack-passed call arguments.
    var out_args: u32 = 0;
    for (f.blocks.items) |blk| for (blk.ops.items) |o| if (o == .call) {
        out_args = @max(out_args, @as(u32, @intCast(o.call.args.len -| CC.args.len)));
    };
    const out: u32 = std.mem.alignForward(u32, 8 * out_args, 16);

    // ---- register allocation input ----
    const rcfg: ra.Config = .{ .allocatable = opts.allocatable, .callee_saved = callee_saved & opts.allocatable };
    var res = try rc.allocateClass(gpa, f, null, .gp, false, rcfg);
    defer res.deinit(gpa);

    // ---- frame ----
    var used: ra.RegMask = 0;
    for (res.locs) |l| if (l == .reg) {
        used |= @as(ra.RegMask, 1) << @intCast(l.reg);
    };
    var saved_buf: [saved_regs.len]X = undefined;
    var nsaved: usize = 0;
    for (saved_regs) |r| if (used & bit(r) != 0) {
        saved_buf[nsaved] = r;
        nsaved += 1;
    };
    const saved = saved_buf[0..nsaved];
    const locals: u32 = out + res.num_slots * 8;
    if (locals > 0x7FFF_0000) return error.TooManySpills;

    const L = struct {
        locs: []const ra.Loc,
        out: u32,
        inline fn reg(r: u8) X {
            return @fromBackingInt(@intCast(r));
        }
        /// `ld`/`sd` of `rt` at frame slot `k`, sp-relative. Past the signed 12-bit offset
        /// range the address `sp + offset` is built in t0 (never allocated).
        inline fn slotAccess(s: @This(), as: anytype, load: bool, rt: X, k: u32) !void {
            const off: i64 = @as(i64, s.out) + 8 * @as(i64, k);
            var base: X = sp;
            var o = off;
            if (off > 2047) {
                try as.li(.t0, off);
                try as.add(.t0, sp, .t0);
                base = .t0;
                o = 0;
            }
            return if (load) as.ld(rt, o, base) else as.sd(rt, o, base);
        }
        /// Register holding `v` (loads spilled values into `scratch`).
        inline fn get(s: @This(), as: anytype, v: ir.VReg, scratch: X) !X {
            return switch (s.locs[v]) {
                .reg => |r| reg(r),
                .stack => |k| blk: {
                    try s.slotAccess(as, true, scratch, k);
                    break :blk scratch;
                },
                .none => blk: { // undefined value: reads as 0
                    try as.li(scratch, 0);
                    break :blk scratch;
                },
            };
        }
        inline fn put(s: @This(), as: anytype, v: ir.VReg, src: X) !void {
            switch (s.locs[v]) {
                .reg => |r| if (reg(r) != src) try as.mv(reg(r), src),
                .stack => |k| try s.slotAccess(as, false, src, k),
                .none => {},
            }
        }
        /// Address `base + disp` as (register, signed 12-bit offset).
        inline fn addr(as: anytype, base: X, disp: i32) !struct { X, i64 } {
            if (disp >= -2048 and disp <= 2047) return .{ base, disp };
            // The offset goes in t0 (never allocated): `base` may be s1 (a spilled base).
            try as.li(.t0, disp);
            try as.add(s1, base, .t0);
            return .{ s1, 0 };
        }
    };
    const ctx: L = .{ .locs = res.locs, .out = out };

    // ---- prolog ----
    // Up to 1920 bytes of locals `prolog` allocates the whole frame with one `addi` (12-bit
    // immediates); a larger area is subtracted here, and the epilog resets sp from s0.
    const big = locals > 1920;
    const big_locals: u32 = std.mem.alignForward(u32, locals, 16);
    const n = try a.prolog(saved, if (big) 0 else locals);
    if (big) {
        try a.li(.t0, big_locals);
        try a.sub(sp, sp, .t0);
    }
    for (0..f.params) |i| {
        if (i < CC.args.len) {
            try ctx.put(a, @intCast(i), CC.args[i]);
        } else {
            try a.ld(s0, @intCast(8 * (i - CC.args.len)), .s0);
            try ctx.put(a, @intCast(i), s0);
        }
    }
    // Values read before any def on some path are 0 (IR semantics, as in `ir.interpret`).
    if (res.entry_undef.len != 0) try a.li(s0, 0);
    for (res.entry_undef) |v| try ctx.put(a, v, s0);

    const labels = try gpa.alloc(rv.Label, f.blocks.items.len);
    defer gpa.free(labels);
    for (labels) |*l| l.* = try a.newLabel();
    const epilog = try a.newLabel();

    for (f.blocks.items, 0..) |blk, bi| {
        try a.bind(labels[bi]);
        for (blk.ops.items) |o| switch (o) {
            .@"const" => |c| switch (res.locs[c.dst]) {
                .reg => |r| try a.li(L.reg(r), c.imm),
                else => {
                    try a.li(s0, c.imm);
                    try ctx.put(a, c.dst, s0);
                },
            },
            .copy => |c| try ctx.put(a, c.dst, try ctx.get(a, c.src, s0)),
            .arith => |c| {
                const ra_ = try ctx.get(a, c.a, s0);
                const rb = try ctx.get(a, c.b, s1);
                switch (c.op) {
                    .add => try a.add(s0, ra_, rb),
                    .sub => try a.sub(s0, ra_, rb),
                    .mul => try a.mul(s0, ra_, rb),
                    .@"and" => try a.@"and"(s0, ra_, rb),
                    .@"or" => try a.@"or"(s0, ra_, rb),
                    .xor => try a.xor(s0, ra_, rb),
                    // sll/srl/sra take the count mod 64, like the IR
                    .shl => try a.sll(s0, ra_, rb),
                    .shr => try a.srl(s0, ra_, rb),
                    .sar => try a.sra(s0, ra_, rb),
                    .sdiv => try a.div(s0, ra_, rb),
                    .srem => try a.rem(s0, ra_, rb),
                }
                try ctx.put(a, c.dst, s0);
            },
            .load => |c| {
                const m = try L.addr(a, try ctx.get(a, c.base, s1), c.disp);
                try a.ld(s0, m[1], m[0]);
                try ctx.put(a, c.dst, s0);
            },
            .store => |c| {
                const v = try ctx.get(a, c.src, s0);
                const m = try L.addr(a, try ctx.get(a, c.base, s1), c.disp);
                try a.sd(v, m[1], m[0]);
            },
            .call => |c| {
                for (c.args, 0..) |arg, i| {
                    const r = try ctx.get(a, arg, s0);
                    if (i < CC.args.len) {
                        if (r != CC.args[i]) try a.mv(CC.args[i], r);
                    } else try a.sd(r, @intCast(8 * (i - CC.args.len)), sp);
                }
                try a.callAbs(c.addr);
                try ctx.put(a, c.dst, CC.ret);
            },
            .bra, .cbra => unreachable, // removed by lowerBlockParams
            .br => |tg| try a.j(labels[tg]),
            .cbr => |c| {
                const x = try ctx.get(a, c.a, s0);
                const y = try ctx.get(a, c.b, s1);
                const tg = labels[c.then];
                switch (c.cond) {
                    .eq => try a.bFar(.eq, x, y, tg),
                    .ne => try a.bFar(.ne, x, y, tg),
                    .lt => try a.bFar(.lt, x, y, tg),
                    .ge => try a.bFar(.ge, x, y, tg),
                    .le => try a.bFar(.ge, y, x, tg),
                    .gt => try a.bFar(.lt, y, x, tg),
                    .ult => try a.bFar(.ltu, x, y, tg),
                    .uge => try a.bFar(.geu, x, y, tg),
                    .ule => try a.bFar(.geu, y, x, tg),
                    .ugt => try a.bFar(.ltu, y, x, tg),
                }
                try a.j(labels[c.@"else"]);
            },
            .fconst, .typed, .fcmp, .cvt, .splat, .extract => unreachable, // rejected by isIntOnly
            .ret => |v| {
                const r = try ctx.get(a, v, s0);
                if (r != CC.ret) try a.mv(CC.ret, r);
                try a.j(epilog);
            },
        };
    }
    try a.bind(epilog);
    if (big) try a.addi(sp, .s0, -n);
    try a.epilog(saved, n);
    return .{ .spills = res.spills, .slots = res.num_slots };
}

/// IR -> bytes in `cb` (finalized), with validation on.
pub fn compile(gpa: std.mem.Allocator, f: *const ir.Func, cb: *CodeBuffer, opts: Options) !Stats {
    if (!f.isIntOnly()) return error.UnsupportedType;
    const st = try lower(gpa, f, rv.Assembler.init(cb), opts);
    try cb.finalize();
    return st;
}

// ---- tests ------------------------------------------------------------------------------
const t = std.testing;
const builtin = @import("builtin");
const JitRuntime = @import("../core/jit_runtime.zig");
const can_run = builtin.cpu.arch == .riscv64;

fn run(f: *const ir.Func, opts: Options, args: []const i64) !i64 {
    var cb = CodeBuffer.init(t.allocator);
    defer cb.deinit();
    _ = try compile(t.allocator, f, &cb, opts);
    const fun = try JitRuntime.add(&cb);
    defer fun.release();
    var v: [8]i64 = @splat(0);
    for (args, 0..) |x, i| v[i] = x;
    const F = *const fn (i64, i64, i64, i64, i64, i64, i64, i64) callconv(.c) i64;
    return fun.ptr(F)(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]);
}

test "riscv64 compiler: fib compiles (byte-level, any host)" {
    var f = try ir.parse(t.allocator, ir.fib_src);
    defer f.deinit();
    var cb = CodeBuffer.init(t.allocator);
    defer cb.deinit();
    const st = try compile(t.allocator, &f, &cb, .{});
    try t.expectEqual(@as(u32, 0), st.spills);
    const c = cb.code();
    try t.expectEqual(@as(usize, 0), c.len % 4);
    const w0 = std.mem.readInt(u32, c[0..4], .little);
    try t.expectEqual(@as(u32, 0x00010113), w0 & 0x000FFFFF); // addi sp, sp, -n
    try t.expectEqual(@as(u32, 0x00008067), std.mem.readInt(u32, c[c.len - 4 ..][0..4], .little)); // ret
    var g = try ir.parse(t.allocator, "func f 17\na:\n  ret v0\n");
    defer g.deinit();
    try t.expectError(error.TooManyParams, compile(t.allocator, &g, &cb, .{}));
    var h = try ir.parse(t.allocator, "func f 0\na:\n  v0 = fconst.f64 1\n  v1 = cvt.i64.f64 v0\n  ret v1\n");
    defer h.deinit();
    try t.expectError(error.UnsupportedType, compile(t.allocator, &h, &cb, .{}));
}

test "riscv64 compiler: frames past the 12-bit sp offsets" {
    // n values live at once with one allocatable register: the spill area passes 2047 bytes.
    for ([_]u32{ 300, 1500 }) |n| {
        var src: std.ArrayList(u8) = .empty;
        defer src.deinit(t.allocator);
        try src.appendSlice(t.allocator, "func f 1\ne:\n");
        for (1..n + 1) |i| try src.print(t.allocator, "  v{d} = add v{d} v0\n", .{ i, i - 1 });
        var acc: usize = n;
        var nv: usize = n + 1;
        var i: usize = n - 1;
        while (i >= 1) : (i -= 1) {
            try src.print(t.allocator, "  v{d} = add v{d} v{d}\n", .{ nv, acc, i });
            acc = nv;
            nv += 1;
        }
        try src.print(t.allocator, "  ret v{d}\n", .{acc});
        var f = try ir.parse(t.allocator, src.items);
        defer f.deinit();
        var cb = CodeBuffer.init(t.allocator);
        defer cb.deinit();
        const st = try compile(t.allocator, &f, &cb, .{ .allocatable = bit(.s1) });
        try t.expect(st.slots * 8 > 2047);
        // Large frames: `sub sp, sp, t0` in the prolog, slots past 2047 via `add t0, sp, t0`.
        var have_sub = false;
        var have_add = false;
        for (std.mem.bytesAsSlice(u32, cb.code()[0 .. cb.code().len / 4 * 4])) |w| {
            have_sub = have_sub or w == 0x40510133;
            have_add = have_add or w == 0x005102B3;
        }
        try t.expect(have_sub and have_add);
        if (!can_run) continue;
        for ([_]i64{ 1, 3 }) |x| try t.expectEqual(try ir.interpret(&f, &.{x}, 1_000_000), try run(&f, .{ .allocatable = bit(.s1) }, &.{x}));
    }
}

test "riscv64 compiler: execution - fib, calls, memory, spills" {
    if (!can_run) return error.SkipZigTest;
    var f = try ir.parse(t.allocator, ir.fib_src);
    defer f.deinit();
    for ([_]i64{ 0, 1, 2, 10, 50, 90 }) |nn| {
        const want = try ir.interpret(&f, &.{nn}, 1 << 20);
        try t.expectEqual(want, try run(&f, .{}, &.{nn}));
        try t.expectEqual(want, try run(&f, .{ .allocatable = bit(.s1) }, &.{nn})); // spills
    }
    const Helper = struct {
        fn mul3(x: i64, y: i64, z: i64) callconv(.c) i64 {
            return x * y * z;
        }
    };
    var mem: [2]i64 = .{ 0, 0 };
    var g = ir.Func.init(t.allocator);
    defer g.deinit();
    g.params = 3;
    const e = try g.addBlock("e");
    const args = try g.alloc().dupe(ir.VReg, &.{ 0, 1, 1 });
    try g.emit(e, .{ .call = .{ .dst = 3, .addr = @intFromPtr(&Helper.mul3), .args = args } });
    try g.emit(e, .{ .arith = .{ .op = .add, .dst = 4, .a = 3, .b = 0 } }); // v0 live across the call
    try g.emit(e, .{ .store = .{ .base = 2, .disp = 8, .src = 4 } });
    try g.emit(e, .{ .load = .{ .dst = 5, .base = 2, .disp = 8 } });
    try g.emit(e, .{ .ret = 5 });
    const p: i64 = @bitCast(@intFromPtr(&mem));
    for ([_]ra.RegMask{ allocatable, bit(.s1) | bit(.t3) }) |m| {
        mem = .{ 0, 0 };
        try t.expectEqual(@as(i64, 3 * 4 * 4 + 3), try run(&g, .{ .allocatable = m }, &.{ 3, 4, p }));
        try t.expectEqual(@as(i64, 51), mem[1]);
    }
}

test "riscv64 compiler: shifts and signed division match the interpreter" {
    if (!can_run) return error.SkipZigTest;
    const src =
        \\func f 2
        \\e:
        \\  v2 = shl v0 v1
        \\  v3 = shr v0 v1
        \\  v4 = sar v0 v1
        \\  v9 = const 7
        \\  v5 = sdiv v0 v9
        \\  v6 = srem v0 v9
        \\  v7 = sdiv v2 v0
        \\  v8 = add v2 v3
        \\  v8 = xor v8 v4
        \\  v8 = add v8 v5
        \\  v8 = sub v8 v6
        \\  v8 = add v8 v7
        \\  ret v8
    ;
    var f = try ir.parse(t.allocator, src);
    defer f.deinit();
    const cases = [_][2]i64{ .{ 1, 0 }, .{ -1, 62 }, .{ 12345, 3 }, .{ -98765, 65 }, .{ std.math.maxInt(i64), 1 }, .{ -7, 2 } };
    for (cases) |c| {
        const want = try ir.interpret(&f, &c, 1000);
        try t.expectEqual(want, try run(&f, .{}, &c));
        try t.expectEqual(want, try run(&f, .{ .allocatable = bit(.t3) | bit(.s1) }, &c));
        try t.expectEqual(want, try run(&f, .{ .allocatable = bit(.s1) }, &c)); // operands spilled
    }
}

test "riscv64 compiler: block parameters (SSA input)" {
    var f = try ir.parse(t.allocator, ir.fib_ssa_src);
    defer f.deinit();
    var cb = CodeBuffer.init(t.allocator);
    defer cb.deinit();
    _ = try compile(t.allocator, &f, &cb, .{});
    if (!can_run) return;
    for ([_]i64{ 0, 1, 10, 90 }) |n| {
        const want = try ir.interpret(&f, &.{n}, 1 << 20);
        try t.expectEqual(want, try run(&f, .{}, &.{n}));
        try t.expectEqual(want, try run(&f, .{ .allocatable = bit(.s1) }, &.{n}));
    }
}

test "riscv64 compiler: stack-passed parameters and call arguments" {
    if (!can_run) return error.SkipZigTest;
    const Helper = struct {
        fn sum11(a: i64, b: i64, c: i64, d: i64, e: i64, f_: i64, g: i64, h: i64, i: i64, j: i64, k: i64) callconv(.c) i64 {
            return a + 2 * b + 3 * c + 4 * d + 5 * e + 6 * f_ + 7 * g + 8 * h + 9 * i + 10 * j + 11 * k;
        }
        fn alt10(a: i64, b: i64, c: i64, d: i64, e: i64, f_: i64, g: i64, h: i64, i: i64, j: i64) callconv(.c) i64 {
            return a - b + c - d + e - f_ + g - h + i - j;
        }
    };
    // 10 parameters (2 on the stack); calls with 11 and 10 arguments.
    var g = ir.Func.init(t.allocator);
    defer g.deinit();
    g.params = 10;
    const e = try g.addBlock("e");
    const a11 = try g.alloc().dupe(ir.VReg, &.{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9 });
    try g.emit(e, .{ .call = .{ .dst = 10, .addr = @intFromPtr(&Helper.sum11), .args = a11 } });
    const a10 = try g.alloc().dupe(ir.VReg, &.{ 10, 9, 8, 7, 6, 5, 4, 3, 2, 1 });
    try g.emit(e, .{ .call = .{ .dst = 11, .addr = @intFromPtr(&Helper.alt10), .args = a10 } });
    try g.emit(e, .{ .arith = .{ .op = .add, .dst = 12, .a = 11, .b = 0 } });
    try g.emit(e, .{ .ret = 12 });
    const p = [_]i64{ 1, 20, 300, 4000, 50000, 600000, 7000000, 80000000, 900000000, 10000000000 };
    const s11 = Helper.sum11(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9], p[9]);
    const want = Helper.alt10(s11, p[9], p[8], p[7], p[6], p[5], p[4], p[3], p[2], p[1]) + p[0];
    for ([_]ra.RegMask{ allocatable, bit(.s1) }) |mask| {
        var cb = CodeBuffer.init(t.allocator);
        defer cb.deinit();
        _ = try compile(t.allocator, &g, &cb, .{ .allocatable = mask });
        const fun = try JitRuntime.add(&cb);
        defer fun.release();
        const F = *const fn (i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) callconv(.c) i64;
        try t.expectEqual(want, fun.ptr(F)(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9]));
    }
}

test "riscv64 compiler: differential vs interpreter on random programs" {
    if (!can_run) return error.SkipZigTest;
    var prng = std.Random.DefaultPrng.init(11);
    const r = prng.random();
    for (0..300) |_| {
        var f = ir.Func.init(t.allocator);
        defer f.deinit();
        f.params = 3;
        const nv: u32 = 3 + r.uintLessThan(u32, 24);
        f.num_vregs = nv;
        const nb = 1 + r.uintLessThan(u32, 5);
        for (0..nb) |_| _ = try f.addBlock("b");
        for (3..nv) |v| try f.emit(0, .{ .@"const" = .{ .dst = @intCast(v), .imm = 0 } });
        for (0..nb) |bi| {
            const blk: u32 = @intCast(bi);
            for (0..r.uintLessThan(u32, 10)) |_| {
                const d = r.uintLessThan(u32, nv);
                if (r.boolean()) {
                    try f.emit(blk, .{ .@"const" = .{ .dst = d, .imm = r.int(i64) } });
                } else try f.emit(blk, .{ .arith = .{ .op = @fromBackingInt(@intCast(r.uintLessThan(u8, @backingInt(ir.Arith.sdiv)))), .dst = d, .a = r.uintLessThan(u32, nv), .b = r.uintLessThan(u32, nv) } });
            }
            if (bi + 1 == nb) {
                try f.emit(blk, .{ .ret = r.uintLessThan(u32, nv) });
            } else {
                const t1 = blk + 1 + r.uintLessThan(u32, nb - blk - 1);
                const t2 = blk + 1 + r.uintLessThan(u32, nb - blk - 1);
                try f.emit(blk, .{ .cbr = .{ .cond = r.enumValue(ir.CondCode), .a = r.uintLessThan(u32, nv), .b = r.uintLessThan(u32, nv), .then = t1, .@"else" = t2 } });
            }
        }
        const args = [_]i64{ r.int(i64), r.int(i64), r.int(i64) };
        const want = try ir.interpret(&f, &args, 1 << 16);
        try t.expectEqual(want, try run(&f, .{}, &args));
        try t.expectEqual(want, try run(&f, .{ .allocatable = bit(.t3) | bit(.s1) }, &args));
    }
}
