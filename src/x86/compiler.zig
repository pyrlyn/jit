//! x86 (32-bit) Compiler: IR (`core/ir.zig`) -> register allocation (`core/regalloc.zig`)
//! -> x86 Builder nodes (`builder.zig`) -> passes -> 32-bit Assembler -> machine code.
//! The 32-bit counterpart of `x86_64/compiler.zig`.
//!
//! Semantics: IR values are 32-bit (`add`/`mul` wrap at 32 bits, shift counts are masked to
//! 5 bits, comparisons are 32-bit, `const` keeps the low 32 bits, `load`/`store` move 4 bytes).
//! `interpret32` is the matching reference interpreter.
//!
//! Conventions (cdecl / SysV i386):
//! - allocatable: eax (caller-saved) + ebx, esi, edi (callee-saved);
//! - edx/ecx are scratch (`s0`/`s1`): `dst = a op b` becomes `mov edx, a; op edx, b; mov dst, edx`;
//!   shift counts go through ecx; division saves eax on the stack around `cdq; idiv ecx`;
//! - parameters are read from [ebp + 8 + 4*i]; calls push arguments right to left with esp
//!   16-aligned at the call (caller pops); results come back in eax;
//! - frame: `push ebp; mov ebp, esp; push <used callee-saved>; sub esp, n`, spill slot k at
//!   [ebp - 4*(saved+1+k)]; one shared epilog label;
//! - folding: a right-hand `const` operand from the same block becomes an imm32, a spilled
//!   right-hand operand of add/sub/and/or/xor a memory operand.
const std = @import("std");
const x = @import("assembler.zig");
const bld = @import("builder.zig");
const ir = @import("../core/ir.zig");
const ra = @import("../core/regalloc.zig");
const CodeBuffer = @import("../core/code_buffer.zig").CodeBuffer;
const Gp = x.Gp;

pub const allocatable: ra.RegMask = bit(.eax) | bit(.ebx) | bit(.esi) | bit(.edi);
pub const callee_saved: ra.RegMask = bit(.ebx) | bit(.esi) | bit(.edi);
const s0: Gp = .edx;
const s1: Gp = .ecx;

fn bit(r: Gp) ra.RegMask {
    return @as(ra.RegMask, 1) << @backingInt(r);
}

pub const Options = struct {
    /// Restrict the allocatable set (spill stress tests).
    allocatable: ra.RegMask = allocatable,
    split_at_calls: bool = true,
    algo: ra.Algo = .linear,
};

pub const Stats = struct {
    spills: u32 = 0,
    slots: u32 = 0,
    nodes: usize = 0,
    folded_imm: u32 = 0,
    folded_mem: u32 = 0,
};

fn imm32(v: i64) i32 {
    return @bitCast(@as(u32, @truncate(@as(u64, @bitCast(v)))));
}

/// Compiles `f` into `b` (an x86 Builder). Run passes and serialize separately, or use `compile`.
pub fn lower(gpa: std.mem.Allocator, f0: *const ir.Func, b: *bld.Builder, opts: Options) !Stats {
    try f0.verify();
    var lowered: ?ir.Func = if (f0.hasBlockParams()) try f0.lowerBlockParams(gpa) else null;
    defer if (lowered) |*g| g.deinit();
    const f: *const ir.Func = if (lowered) |*g| g else f0;
    const cc = x.CallConv.cdecl;

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
            var buf: [ir.max_args]ir.VReg = undefined;
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
    const rcfg: ra.Config = .{ .allocatable = opts.allocatable, .callee_saved = callee_saved & opts.allocatable, .split_at_calls = opts.split_at_calls, .algo = opts.algo };
    var res = try ra.allocate(gpa, rf, rcfg);
    defer res.deinit(gpa);
    std.debug.assert(ra.verify(rf, rcfg, res));

    var used: ra.RegMask = 0;
    for (res.locs) |l| if (l == .reg) {
        used |= @as(ra.RegMask, 1) << @intCast(l.reg);
    };
    var saved: std.ArrayList(Gp) = .empty;
    defer saved.deinit(gpa);
    for ([_]Gp{ .ebx, .esi, .edi }) |r| if (used & bit(r) != 0) try saved.append(gpa, r);
    const nsaved: u32 = @intCast(saved.items.len);
    // ret addr + ebp + saved + slots + adj = 0 mod 16 => esp 16-aligned in the body.
    const fixed: u32 = 8 + 4 * nsaved;
    const adj = std.mem.alignForward(u32, fixed + res.num_slots * 4, 16) - fixed;

    const L = struct {
        locs: []const ra.Loc,
        nsaved: u32,
        b: *bld.Builder,
        fn slot(s: @This(), k: u32) x.Mem {
            return x.ptr(.ebp, -@as(i32, @intCast(4 * (s.nsaved + 1 + k))));
        }
        fn reg(r: u32) Gp {
            return @fromBackingInt(@intCast(r));
        }
        fn get(s: @This(), v: ir.VReg, scratch: Gp) !Gp {
            return switch (s.locs[v]) {
                .reg => |r| reg(r),
                .stack => |k| blk: {
                    try s.b.emit(.movRM, .{ scratch, s.slot(k) });
                    break :blk scratch;
                },
                .none => blk: {
                    try s.b.emit(.movRI, .{ scratch, 0 });
                    break :blk scratch;
                },
            };
        }
        fn put(s: @This(), v: ir.VReg, src: Gp) !void {
            switch (s.locs[v]) {
                .reg => |r| if (reg(r) != src) try s.b.emit(.movRR, .{ reg(r), src }),
                .stack => |k| try s.b.emit(.movMR, .{ s.slot(k), src }),
                .none => {},
            }
        }
    };
    const ctx: L = .{ .locs = res.locs, .nsaved = nsaved, .b = b };

    try b.comment(f.name);
    try b.emit(.push, .{.ebp});
    try b.emit(.movRR, .{ .ebp, .esp });
    for (saved.items) |r| try b.emit(.push, .{r});
    if (adj != 0) try b.emit(.subRI, .{ .esp, @intCast(adj) });
    for (0..f.params) |i| {
        if (res.locs[i] == .none) continue;
        try b.emit(.movRM, .{ s0, cc.stackArg(@intCast(i)) });
        try ctx.put(@intCast(i), s0);
    }

    const labels = try gpa.alloc(bld.Label, f.blocks.items.len);
    defer gpa.free(labels);
    for (labels) |*l| l.* = try b.newLabel();
    const epilog = try b.newLabel();

    const known = try A.alloc(?i32, @max(f.num_vregs, 1));
    var folded_imm: u32 = 0;
    var folded_mem: u32 = 0;
    var pos: u32 = 1;
    for (f.blocks.items, 0..) |blk, bi| {
        try b.bind(labels[bi]);
        @memset(known, null);
        for (blk.ops.items) |o| {
            defer pos += 1;
            defer if (o.def()) |d| {
                known[d] = if (o == .@"const") imm32(o.@"const".imm) else null;
            };
            switch (o) {
                .@"const" => |c| switch (res.locs[c.dst]) {
                    .reg => |r| try b.emit(.movRI, .{ L.reg(r), imm32(c.imm) }),
                    else => {
                        try b.emit(.movRI, .{ s0, imm32(c.imm) });
                        try ctx.put(c.dst, s0);
                    },
                },
                .copy => |c| try ctx.put(c.dst, try ctx.get(c.src, s0)),
                .arith => |c| fold: {
                    const alu: ?x.Alu = switch (c.op) {
                        .add => .add,
                        .sub => .sub,
                        .@"and" => .@"and",
                        .@"or" => .@"or",
                        .xor => .xor,
                        else => null,
                    };
                    if (alu != null or c.op == .mul) {
                        if (known[c.b]) |imm| {
                            const ra0 = try ctx.get(c.a, s0);
                            if (alu) |op| {
                                if (ra0 != s0) try b.emit(.movRR, .{ s0, ra0 });
                                try b.emit(.aluRI, .{ op, s0, imm });
                            } else try b.emit(.imulRRI, .{ s0, ra0, imm });
                            try ctx.put(c.dst, s0);
                            folded_imm += 1;
                            break :fold;
                        }
                        if (alu != null and res.locs[c.b] == .stack) {
                            const ra0 = try ctx.get(c.a, s0);
                            if (ra0 != s0) try b.emit(.movRR, .{ s0, ra0 });
                            try b.emit(.aluRM, .{ alu.?, s0, ctx.slot(res.locs[c.b].stack) });
                            try ctx.put(c.dst, s0);
                            folded_mem += 1;
                            break :fold;
                        }
                    }
                    const ra_ = try ctx.get(c.a, s0);
                    const rb = try ctx.get(c.b, s1);
                    switch (c.op) {
                        .sdiv, .srem => {
                            // eax may hold a live value: keep it on the stack around idiv.
                            if (rb != s1) try b.emit(.movRR, .{ s1, rb });
                            try b.emit(.push, .{.eax});
                            if (ra_ != .eax) try b.emit(.movRR, .{ .eax, ra_ });
                            try b.add(.cdq);
                            try b.emit(.idivR, .{s1});
                            if (c.op == .sdiv) try b.emit(.movRR, .{ s0, .eax });
                            try b.emit(.pop, .{.eax});
                            try ctx.put(c.dst, s0);
                            break :fold;
                        },
                        else => {},
                    }
                    if (ra_ != s0) try b.emit(.movRR, .{ s0, ra_ });
                    switch (c.op) {
                        .shl, .shr, .sar => {
                            if (rb != s1) try b.emit(.movRR, .{ s1, rb });
                            try b.emit(.shiftRCl, .{ switch (c.op) {
                                .shl => .shl,
                                .shr => .shr,
                                else => .sar,
                            }, s0 });
                        },
                        .sdiv, .srem => unreachable,
                        .add => try b.emit(.aluRR, .{ .add, s0, rb }),
                        .sub => try b.emit(.aluRR, .{ .sub, s0, rb }),
                        .@"and" => try b.emit(.aluRR, .{ .@"and", s0, rb }),
                        .@"or" => try b.emit(.aluRR, .{ .@"or", s0, rb }),
                        .xor => try b.emit(.aluRR, .{ .xor, s0, rb }),
                        .mul => try b.emit(.imulRR, .{ s0, rb }),
                    }
                    try ctx.put(c.dst, s0);
                },
                .load => |c| {
                    const base = try ctx.get(c.base, s1);
                    try b.emit(.movRM, .{ s0, x.ptr(base, c.disp) });
                    try ctx.put(c.dst, s0);
                },
                .store => |c| {
                    const base = try ctx.get(c.base, s1);
                    const v = try ctx.get(c.src, s0);
                    try b.emit(.movMR, .{ x.ptr(base, c.disp), v });
                },
                .call => |c| {
                    for (0..res.locs.len) |v| if (res.savedAcross(@intCast(v), pos)) {
                        try b.emit(.movMR, .{ ctx.slot(res.save_slot[v].?), L.reg(res.locs[v].reg) });
                    };
                    const n: u32 = @intCast(c.args.len);
                    const pad: u32 = (16 - (4 * n) % 16) % 16;
                    if (pad != 0) try b.emit(.subRI, .{ .esp, @intCast(pad) });
                    var k = c.args.len;
                    while (k > 0) {
                        k -= 1;
                        try b.emit(.push, .{try ctx.get(c.args[k], s0)});
                    }
                    try b.emit(.movRI, .{ s0, @as(i64, @intCast(@as(u32, @truncate(c.addr)))) });
                    try b.emit(.callR, .{s0});
                    if (4 * n + pad != 0) try b.emit(.addRI, .{ .esp, @intCast(4 * n + pad) });
                    var restored = false;
                    for (0..res.locs.len) |v| if (res.savedAcross(@intCast(v), pos)) {
                        if (!restored) try b.emit(.movRR, .{ s0, .eax });
                        restored = true;
                        try b.emit(.movRM, .{ L.reg(res.locs[v].reg), ctx.slot(res.save_slot[v].?) });
                    };
                    try ctx.put(c.dst, if (restored) s0 else .eax);
                },
                .bra, .cbra => unreachable, // removed by lowerBlockParams
                .br => |tg| try b.emit(.jmp, .{labels[tg]}),
                .cbr => |c| {
                    const ra_ = try ctx.get(c.a, s0);
                    if (known[c.b]) |imm| {
                        try b.emit(.aluRI, .{ .cmp, ra_, imm });
                        folded_imm += 1;
                    } else try b.emit(.cmpRR, .{ ra_, try ctx.get(c.b, s1) });
                    const cond: x.Cond = switch (c.cond) {
                        .eq => .e,
                        .ne => .ne,
                        .lt => .l,
                        .le => .le,
                        .gt => .g,
                        .ge => .ge,
                        .ult => .b,
                        .ule => .be,
                        .ugt => .a,
                        .uge => .ae,
                    };
                    try b.emit(.jcc, .{ cond, labels[c.then] });
                    try b.emit(.jmp, .{labels[c.@"else"]});
                },
                .ret => |v| {
                    const r = try ctx.get(v, s0);
                    if (r != .eax) try b.emit(.movRR, .{ .eax, r });
                    try b.emit(.jmp, .{epilog});
                },
            }
        }
    }
    try b.bind(epilog);
    if (adj != 0) try b.emit(.addRI, .{ .esp, @intCast(adj) });
    var i = saved.items.len;
    while (i > 0) {
        i -= 1;
        try b.emit(.pop, .{saved.items[i]});
    }
    try b.emit(.pop, .{.ebp});
    try b.add(.ret);
    return .{ .spills = res.spills, .slots = res.num_slots, .nodes = b.count(), .folded_imm = folded_imm, .folded_mem = folded_mem };
}

/// IR -> bytes in `cb` (finalized). Runs dead-code and jump-to-next passes.
pub fn compile(gpa: std.mem.Allocator, f: *const ir.Func, cb: *CodeBuffer, opts: Options) !Stats {
    var b = bld.Builder.init(gpa);
    defer b.deinit();
    var st = try lower(gpa, f, &b, opts);
    try b.runPasses(.{ bld.removeDeadCode, bld.removeJumpsToNext });
    try b.serialize(x.AssemblerT(.{ .validate = true }).init(cb));
    try cb.finalize();
    st.nodes = b.count();
    return st;
}

/// Reference interpreter with the compiler's 32-bit semantics (no load/store/call).
pub fn interpret32(f: *const ir.Func, args: []const i32, max_steps: usize) !i32 {
    const regs = try std.heap.page_allocator.alloc(i32, @max(f.num_vregs, 1));
    defer std.heap.page_allocator.free(regs);
    @memset(regs, 0);
    for (args, 0..) |a, i| regs[i] = a;
    var blk: u32 = 0;
    var steps: usize = 0;
    while (true) {
        for (f.blocks.items[blk].ops.items) |o| {
            steps += 1;
            if (steps > max_steps) return error.StepLimit;
            switch (o) {
                .@"const" => |c| regs[c.dst] = imm32(c.imm),
                .copy => |c| regs[c.dst] = regs[c.src],
                .arith => |c| regs[c.dst] = arith32(c.op, regs[c.a], regs[c.b]),
                .br => |tg| {
                    blk = tg;
                    break;
                },
                .cbr => |c| {
                    blk = if (cond32(c.cond, regs[c.a], regs[c.b])) c.then else c.@"else";
                    break;
                },
                .ret => |v| return regs[v],
                else => return error.Unsupported,
            }
        }
    }
}
pub fn arith32(op: ir.Arith, a: i32, b: i32) i32 {
    const ua: u32 = @bitCast(a);
    const sh: u5 = @truncate(@as(u32, @bitCast(b)));
    return switch (op) {
        .add => a +% b,
        .sub => a -% b,
        .mul => a *% b,
        .@"and" => a & b,
        .@"or" => a | b,
        .xor => a ^ b,
        .shl => @bitCast(ua << sh),
        .shr => @bitCast(ua >> sh),
        .sar => a >> sh,
        .sdiv => if (b == 0 or (a == std.math.minInt(i32) and b == -1)) 0 else @divTrunc(a, b),
        .srem => if (b == 0 or (a == std.math.minInt(i32) and b == -1)) 0 else @rem(a, b),
    };
}
pub fn cond32(c: ir.CondCode, a: i32, b: i32) bool {
    const ua: u32 = @bitCast(a);
    const ub: u32 = @bitCast(b);
    return switch (c) {
        .eq => a == b,
        .ne => a != b,
        .lt => a < b,
        .le => a <= b,
        .gt => a > b,
        .ge => a >= b,
        .ult => ua < ub,
        .ule => ua <= ub,
        .ugt => ua > ub,
        .uge => ua >= ub,
    };
}

// ---- tests ------------------------------------------------------------------------------
const t = std.testing;
const builtin = @import("builtin");
const JitRuntime = @import("../core/jit_runtime.zig");
/// Compiled 32-bit code runs only in a 32-bit x86 process (e.g. `-Dtarget=x86-linux`).
const can_run = builtin.cpu.arch == .x86;

fn run(f: *const ir.Func, opts: Options, args: []const i32) !i32 {
    var cb = CodeBuffer.init(t.allocator);
    defer cb.deinit();
    _ = try compile(t.allocator, f, &cb, opts);
    const fun = try JitRuntime.add(&cb);
    defer fun.release();
    var a: [6]i32 = @splat(0);
    for (args, 0..) |v, i| a[i] = v;
    return fun.ptr(*const fn (i32, i32, i32, i32, i32, i32) callconv(.c) i32)(a[0], a[1], a[2], a[3], a[4], a[5]);
}

test "x86 compiler: fib compiles to the expected shape (byte-level, any host)" {
    var f = try ir.parse(t.allocator, ir.fib_src);
    defer f.deinit();
    var cb = CodeBuffer.init(t.allocator);
    defer cb.deinit();
    const st = try compile(t.allocator, &f, &cb, .{});
    try t.expect(st.spills <= 2); // 4 allocatable registers for fib's 6 values
    const c = cb.code();
    try t.expectEqualSlices(u8, &.{ 0x55, 0x89, 0xE5 }, c[0..3]); // push ebp; mov ebp, esp
    try t.expectEqual(@as(u8, 0xC3), c[c.len - 1]);
    // All allocations spill with a single register; still encodes.
    var cb2 = CodeBuffer.init(t.allocator);
    defer cb2.deinit();
    const st2 = try compile(t.allocator, &f, &cb2, .{ .allocatable = bit(.ebx) });
    try t.expect(st2.spills > 0);
    var cb3 = CodeBuffer.init(t.allocator);
    defer cb3.deinit();
    _ = try compile(t.allocator, &f, &cb3, .{ .algo = .graph });
}

test "x86 compiler: 32-bit reference semantics" {
    try t.expectEqual(@as(i32, std.math.minInt(i32)), arith32(.add, std.math.maxInt(i32), 1));
    try t.expectEqual(@as(i32, 2), arith32(.shl, 1, 33)); // count masked to 5 bits
    try t.expect(cond32(.ugt, -1, 1));
    try t.expectEqual(@as(i32, -3), arith32(.sdiv, -7, 2));
    try t.expectEqual(@as(i32, -1), arith32(.srem, -7, 2));
}

test "x86 compiler: execution - fib, division, shifts, calls, memory, spills (x86 host)" {
    if (!can_run) return error.SkipZigTest;
    var f = try ir.parse(t.allocator, ir.fib_src);
    defer f.deinit();
    for ([_]i32{ 0, 1, 2, 10, 40, 50 }) |n| {
        const want = try interpret32(&f, &.{n}, 1 << 20);
        try t.expectEqual(want, try run(&f, .{}, &.{n}));
        try t.expectEqual(want, try run(&f, .{ .allocatable = bit(.ebx) }, &.{n}));
        try t.expectEqual(want, try run(&f, .{ .algo = .graph }, &.{n}));
    }
    const Helper = struct {
        fn mul3(a: i32, b: i32, c: i32) callconv(.c) i32 {
            return a *% b *% c;
        }
    };
    var mem: [2]i32 = .{ 0, 0 };
    var g = ir.Func.init(t.allocator);
    defer g.deinit();
    g.params = 3;
    const e = try g.addBlock("e");
    const args = try g.alloc().dupe(ir.VReg, &.{ 0, 1, 1 });
    try g.emit(e, .{ .call = .{ .dst = 3, .addr = @intFromPtr(&Helper.mul3), .args = args } });
    try g.emit(e, .{ .arith = .{ .op = .add, .dst = 4, .a = 3, .b = 0 } });
    try g.emit(e, .{ .store = .{ .base = 2, .disp = 4, .src = 4 } });
    try g.emit(e, .{ .load = .{ .dst = 5, .base = 2, .disp = 4 } });
    try g.emit(e, .{ .ret = 5 });
    const p: i32 = @bitCast(@as(u32, @intCast(@intFromPtr(&mem))));
    for ([_]ra.RegMask{ allocatable, bit(.eax), bit(.ebx) }) |m| {
        mem = .{ 0, 0 };
        try t.expectEqual(@as(i32, 3 * 4 * 4 + 3), try run(&g, .{ .allocatable = m }, &.{ 3, 4, p }));
        try t.expectEqual(@as(i32, 51), mem[1]);
    }
}

test "x86 compiler: random programs match interpret32 (x86 host)" {
    if (!can_run) return error.SkipZigTest;
    var prng = std.Random.DefaultPrng.init(0x86);
    const r = prng.random();
    for (0..300) |_| {
        var f = ir.Func.init(t.allocator);
        defer f.deinit();
        f.params = 3;
        const nv: u32 = 3 + r.uintLessThan(u32, 20);
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
                } else {
                    const op: ir.Arith = @fromBackingInt(@intCast(r.uintLessThan(u8, @backingInt(ir.Arith.srem) + 1)));
                    var bv = r.uintLessThan(u32, nv);
                    if (op == .sdiv or op == .srem) {
                        // Divisor: a fresh non-zero, non -1 constant.
                        bv = r.uintLessThan(u32, nv);
                        try f.emit(blk, .{ .@"const" = .{ .dst = bv, .imm = 2 + r.uintLessThan(u32, 1000) } });
                    }
                    try f.emit(blk, .{ .arith = .{ .op = op, .dst = d, .a = r.uintLessThan(u32, nv), .b = bv } });
                }
            }
            if (bi + 1 == nb) {
                try f.emit(blk, .{ .ret = r.uintLessThan(u32, nv) });
            } else {
                const t1 = blk + 1 + r.uintLessThan(u32, nb - blk - 1);
                const t2 = blk + 1 + r.uintLessThan(u32, nb - blk - 1);
                try f.emit(blk, .{ .cbr = .{ .cond = r.enumValue(ir.CondCode), .a = r.uintLessThan(u32, nv), .b = r.uintLessThan(u32, nv), .then = t1, .@"else" = t2 } });
            }
        }
        const args = [_]i32{ r.int(i32), r.int(i32), r.int(i32) };
        const want = try interpret32(&f, &args, 1 << 16);
        try t.expectEqual(want, try run(&f, .{}, &args));
        try t.expectEqual(want, try run(&f, .{ .allocatable = bit(.eax) | bit(.ebx) }, &args));
        try t.expectEqual(want, try run(&f, .{ .algo = .graph }, &args));
    }
}
