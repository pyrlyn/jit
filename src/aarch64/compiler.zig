//! AArch64 Compiler: IR (`core/ir.zig`) -> linear-scan register allocation
//! (`core/regalloc.zig`) -> AArch64 Assembler -> machine code (AAPCS64).
//!
//! Conventions:
//! - allocatable: x9..x15 (caller-saved) and x19..x28 (callee-saved); x0..x7 (arguments)
//!   are not allocated, so parameter and call-argument moves never conflict; x18 (platform
//!   register on Apple) is never touched;
//! - x16/x17 (IP0/IP1) are scratch for spilled operands, large offsets and call targets;
//! - frame: `Assembler.prolog(used callee-saved, slots)`; spill slot k at [sp + 8k];
//!   one shared epilog label.
//! - A64 is three-address, so `dst = a op b` needs no two-address fix-up.
//! Limits: up to 8 parameters and call arguments (no stack-passed arguments yet).
const std = @import("std");
const a64 = @import("assembler.zig");
const ir = @import("../core/ir.zig");
const ra = @import("../core/regalloc.zig");
const CodeBuffer = @import("../core/code_buffer.zig").CodeBuffer;
const X = a64.X;
const CC = a64.CallConv;

const s0: X = .x16;
const s1: X = .x17;
const sp: X = @fromBackingInt(31);

fn bit(r: X) ra.RegMask {
    return @as(ra.RegMask, 1) << @backingInt(r);
}
fn maskOf(comptime rs: []const X) ra.RegMask {
    var m: ra.RegMask = 0;
    for (rs) |r| m |= bit(r);
    return m;
}
pub const callee_saved: ra.RegMask = maskOf(&CC.callee_saved);
pub const allocatable: ra.RegMask = maskOf(&.{ .x9, .x10, .x11, .x12, .x13, .x14, .x15 }) | callee_saved;

pub const Options = struct {
    /// Restrict the allocatable set (spill stress tests).
    allocatable: ra.RegMask = allocatable,
};

pub const Stats = struct { spills: u32 = 0, slots: u32 = 0 };

/// Compiles `f` into `cb` through `a` (an `a64.AssemblerT(...)` instance on `cb`), not finalized.
pub fn lower(gpa: std.mem.Allocator, f: *const ir.Func, a: anytype, opts: Options) !Stats {
    try f.verify();
    if (f.params > CC.args.len) return error.TooManyParams;
    for (f.blocks.items) |blk| for (blk.ops.items) |o| if (o == .call and o.call.args.len > CC.args.len) return error.TooManyArgs;

    // ---- register allocation input ----
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const A = arena.allocator();
    const blocks = try A.alloc(ra.Block, f.blocks.items.len);
    for (f.blocks.items, 0..) |blk, bi| {
        const extra: usize = if (bi == 0) 1 else 0;
        const insts = try A.alloc(ra.Inst, blk.ops.items.len + extra);
        if (bi == 0) {
            const params = try A.alloc(ra.VReg, f.params);
            for (params, 0..) |*p, i| p.* = @intCast(i);
            insts[0] = .{ .defs = params };
        }
        for (blk.ops.items, 0..) |o, i| {
            var buf: [8]ir.VReg = undefined;
            const u = try A.dupe(ra.VReg, o.uses(&buf));
            const d: []const ra.VReg = if (o.def()) |dv| try A.dupe(ra.VReg, &.{dv}) else &.{};
            insts[i + extra] = .{ .uses = u, .defs = d, .is_call = o == .call };
        }
        const succs: []const u32 = switch (blk.ops.items[blk.ops.items.len - 1]) {
            .br => |tg| try A.dupe(u32, &.{tg}),
            .cbr => |c| try A.dupe(u32, &.{ c.then, c.@"else" }),
            else => &.{},
        };
        blocks[bi] = .{ .insts = insts, .succs = succs };
    }
    const rf: ra.Func = .{ .blocks = blocks, .num_vregs = @max(f.num_vregs, 1) };
    const rcfg: ra.Config = .{ .allocatable = opts.allocatable, .callee_saved = callee_saved & opts.allocatable };
    var res = try ra.allocate(gpa, rf, rcfg);
    defer res.deinit(gpa);
    std.debug.assert(ra.verify(rf, rcfg, res));

    // ---- frame ----
    var used: ra.RegMask = 0;
    for (res.locs) |l| if (l == .reg) {
        used |= @as(ra.RegMask, 1) << @intCast(l.reg);
    };
    var saved_buf: [CC.callee_saved.len]X = undefined;
    var nsaved: usize = 0;
    for (CC.callee_saved) |r| if (used & bit(r) != 0) {
        saved_buf[nsaved] = r;
        nsaved += 1;
    };
    const saved = saved_buf[0..nsaved];
    if (res.num_slots * 8 > 4080) return error.TooManySpills;

    const L = struct {
        locs: []const ra.Loc,
        fn reg(r: u8) X {
            return @fromBackingInt(@intCast(r));
        }
        fn slot(k: u32) u15 {
            return @intCast(8 * k);
        }
        /// Register holding `v` (loads spilled values into `scratch`).
        fn get(s: @This(), as: anytype, v: ir.VReg, scratch: X) !X {
            return switch (s.locs[v]) {
                .reg => |r| reg(r),
                .stack => |k| blk: {
                    try as.ldr(scratch, sp, slot(k));
                    break :blk scratch;
                },
                .none => blk: { // undefined value: reads as 0
                    try as.movImm(scratch, 0);
                    break :blk scratch;
                },
            };
        }
        fn put(s: @This(), as: anytype, v: ir.VReg, src: X) !void {
            switch (s.locs[v]) {
                .reg => |r| if (reg(r) != src) try as.mov(reg(r), src),
                .stack => |k| try as.str(src, sp, slot(k)),
                .none => {},
            }
        }
        /// Address `base + disp` as (register, unsigned scaled offset).
        fn addr(as: anytype, base: X, disp: i32) !struct { X, u15 } {
            if (disp >= 0 and disp <= 32760 and @mod(disp, 8) == 0) return .{ base, @intCast(disp) };
            try as.movImm(s1, @bitCast(@as(i64, disp)));
            try as.add(s1, base, s1);
            return .{ s1, 0 };
        }
    };
    const ctx: L = .{ .locs = res.locs };

    // ---- prolog ----
    const n = try a.prolog(saved, res.num_slots * 8);
    for (0..f.params) |i| try ctx.put(a, @intCast(i), CC.args[i]);

    const labels = try gpa.alloc(a64.Label, f.blocks.items.len);
    defer gpa.free(labels);
    for (labels) |*l| l.* = try a.newLabel();
    const epilog = try a.newLabel();

    for (f.blocks.items, 0..) |blk, bi| {
        try a.bind(labels[bi]);
        for (blk.ops.items) |o| switch (o) {
            .@"const" => |c| switch (res.locs[c.dst]) {
                .reg => |r| try a.movImm(L.reg(r), @bitCast(c.imm)),
                else => {
                    try a.movImm(s0, @bitCast(c.imm));
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
                    .@"and" => try a.and_(s0, ra_, rb),
                    .@"or" => try a.orr(s0, ra_, rb),
                    .xor => try a.eor(s0, ra_, rb),
                }
                try ctx.put(a, c.dst, s0);
            },
            .load => |c| {
                const m = try L.addr(a, try ctx.get(a, c.base, s1), c.disp);
                try a.ldr(s0, m[0], m[1]);
                try ctx.put(a, c.dst, s0);
            },
            .store => |c| {
                const v = try ctx.get(a, c.src, s0);
                const m = try L.addr(a, try ctx.get(a, c.base, s1), c.disp);
                try a.str(v, m[0], m[1]);
            },
            .call => |c| {
                for (c.args, 0..) |arg, i| {
                    const r = try ctx.get(a, arg, s0);
                    if (r != CC.args[i]) try a.mov(CC.args[i], r);
                }
                try a.movImm(s0, c.addr);
                try a.blr(s0);
                try ctx.put(a, c.dst, CC.ret);
            },
            .br => |tg| try a.b(labels[tg]),
            .cbr => |c| {
                try a.cmp(try ctx.get(a, c.a, s0), try ctx.get(a, c.b, s1));
                switch (c.cond) {
                    inline else => |cc| try a.bcond(comptime switch (cc) {
                        .eq => .eq,
                        .ne => .ne,
                        .lt => .lt,
                        .le => .le,
                        .gt => .gt,
                        .ge => .ge,
                        .ult => .lo,
                        .ule => .ls,
                        .ugt => .hi,
                        .uge => .hs,
                    }, labels[c.then]),
                }
                try a.b(labels[c.@"else"]);
            },
            .ret => |v| {
                const r = try ctx.get(a, v, s0);
                if (r != CC.ret) try a.mov(CC.ret, r);
                try a.b(epilog);
            },
        };
    }
    try a.bind(epilog);
    try a.epilog(saved, n);
    return .{ .spills = res.spills, .slots = res.num_slots };
}

/// IR -> bytes in `cb` (finalized), with validation on.
pub fn compile(gpa: std.mem.Allocator, f: *const ir.Func, cb: *CodeBuffer, opts: Options) !Stats {
    const st = try lower(gpa, f, a64.AssemblerT(.{ .validate = true }).init(cb), opts);
    try cb.finalize();
    return st;
}

// ---- tests ------------------------------------------------------------------------------
const t = std.testing;
const builtin = @import("builtin");
const JitRuntime = @import("../core/jit_runtime.zig");
const can_run = builtin.cpu.arch == .aarch64;

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

test "a64 compiler: fib compiles (byte-level, any host)" {
    var f = try ir.parse(t.allocator, ir.fib_src);
    defer f.deinit();
    var cb = CodeBuffer.init(t.allocator);
    defer cb.deinit();
    const st = try compile(t.allocator, &f, &cb, .{});
    try t.expectEqual(@as(u32, 0), st.spills);
    const c = cb.code();
    try t.expectEqual(@as(usize, 0), c.len % 4);
    try t.expectEqual(@as(u32, 0xA9BF7BFD), std.mem.readInt(u32, c[0..4], .little)); // stp x29, x30
    try t.expectEqual(@as(u32, 0xD65F03C0), std.mem.readInt(u32, c[c.len - 4 ..][0..4], .little)); // ret
    var g = try ir.parse(t.allocator, "func f 9\na:\n  ret v0\n");
    defer g.deinit();
    try t.expectError(error.TooManyParams, compile(t.allocator, &g, &cb, .{}));
}

test "a64 compiler: execution - fib, calls, memory, spills" {
    if (!can_run) return error.SkipZigTest;
    var f = try ir.parse(t.allocator, ir.fib_src);
    defer f.deinit();
    for ([_]i64{ 0, 1, 2, 10, 50, 90 }) |nn| {
        const want = try ir.interpret(&f, &.{nn}, 1 << 20);
        try t.expectEqual(want, try run(&f, .{}, &.{nn}));
        try t.expectEqual(want, try run(&f, .{ .allocatable = bit(.x19) }, &.{nn})); // spills
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
    for ([_]ra.RegMask{ allocatable, bit(.x19) | bit(.x9) }) |m| {
        mem = .{ 0, 0 };
        try t.expectEqual(@as(i64, 3 * 4 * 4 + 3), try run(&g, .{ .allocatable = m }, &.{ 3, 4, p }));
        try t.expectEqual(@as(i64, 51), mem[1]);
    }
}

test "a64 compiler: differential vs interpreter on random programs" {
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
                } else try f.emit(blk, .{ .arith = .{ .op = r.enumValue(ir.Arith), .dst = d, .a = r.uintLessThan(u32, nv), .b = r.uintLessThan(u32, nv) } });
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
        try t.expectEqual(want, try run(&f, .{ .allocatable = bit(.x9) | bit(.x19) }, &args));
    }
}
