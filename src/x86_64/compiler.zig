//! x86-64 Compiler: IR (`core/ir.zig`) -> linear-scan register allocation (`core/regalloc.zig`)
//! -> Builder nodes (`builder.zig`) -> passes -> Assembler -> machine code.
//! Like AsmJit's x86::Compiler but over a small typed IR instead of virtual-register operands.
//!
//! Conventions (host C ABI: SysV on macOS/Linux, Win64 on Windows):
//! - allocatable: rax (caller-saved) + rbx, r12..r15 (callee-saved); argument registers are
//!   not allocated, so parameter and call-argument moves never conflict;
//! - r10/r11 are scratch for spilled operands and two-address fix-up (`dst = a op b` becomes
//!   `mov r10, a; op r10, b; mov dst, r10`);
//! - frame: `push rbp; mov rbp, rsp; push <used callee-saved>; sub rsp, slots` (16-aligned),
//!   spill slot k at [rbp - 8*(saved+1+k)]; one shared epilog label.
//! Not yet: AArch64 lowering, shifts/div (fixed rcx/rax/rdx), stack-passed arguments,
//! immediate and memory-operand folding, cmp/branch fusion beyond `cbr`.
const std = @import("std");
const x = @import("assembler.zig");
const bld = @import("builder.zig");
const ir = @import("../core/ir.zig");
const ra = @import("../core/regalloc.zig");
const CodeBuffer = @import("../core/code_buffer.zig").CodeBuffer;
const Gp = x.Gp;

pub const allocatable: ra.RegMask = bit(.rax) | bit(.rbx) | bit(.r12) | bit(.r13) | bit(.r14) | bit(.r15);
pub const callee_saved: ra.RegMask = bit(.rbx) | bit(.r12) | bit(.r13) | bit(.r14) | bit(.r15);
const s0: Gp = .r10;
const s1: Gp = .r11;

fn bit(r: Gp) ra.RegMask {
    return @as(ra.RegMask, 1) << @intFromEnum(r);
}

pub const Options = struct {
    /// Restrict the allocatable set (spill stress tests).
    allocatable: ra.RegMask = allocatable,
    shorten_branches: bool = true,
};

pub const Stats = struct { spills: u32 = 0, slots: u32 = 0, nodes: usize = 0 };

/// Compiles `f` into `b` (a Builder). Run passes and serialize separately, or use `compile`.
pub fn lower(gpa: std.mem.Allocator, f: *const ir.Func, b: *bld.Builder, opts: Options) !Stats {
    try f.verify();
    const cc = x.CallConv.host;
    if (f.params > cc.args.len) return error.TooManyParams;

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
    var saved: std.ArrayList(Gp) = .empty;
    defer saved.deinit(gpa);
    for ([_]Gp{ .rbx, .r12, .r13, .r14, .r15 }) |r| if (used & bit(r) != 0) try saved.append(gpa, r);
    const nsaved: u32 = @intCast(saved.items.len);
    const slots_bytes = res.num_slots * 8;
    const adj = std.mem.alignForward(u32, nsaved * 8 + slots_bytes, 16) - nsaved * 8;

    const L = struct {
        locs: []const ra.Loc,
        nsaved: u32,
        b: *bld.Builder,
        fn slot(s: @This(), k: u32) x.Mem {
            return x.ptr(.rbp, -@as(i32, @intCast(8 * (s.nsaved + 1 + k))));
        }
        /// Register holding `v` (loads spilled values into `scratch`).
        fn get(s: @This(), v: ir.VReg, scratch: Gp) !Gp {
            return switch (s.locs[v]) {
                .reg => |r| @enumFromInt(r),
                .stack => |k| blk: {
                    try s.b.movRM(scratch, s.slot(k));
                    break :blk scratch;
                },
                .none => blk: { // undefined value: reads as 0
                    try s.b.movRI(scratch, 0);
                    break :blk scratch;
                },
            };
        }
        fn put(s: @This(), v: ir.VReg, src: Gp) !void {
            switch (s.locs[v]) {
                .reg => |r| if (@as(Gp, @enumFromInt(r)) != src) try s.b.movRR(@enumFromInt(r), src),
                .stack => |k| try s.b.movMR(s.slot(k), src),
                .none => {},
            }
        }
    };
    const ctx: L = .{ .locs = res.locs, .nsaved = nsaved, .b = b };

    // ---- prolog ----
    try b.comment(f.name);
    try b.push(.rbp);
    try b.movRR(.rbp, .rsp);
    for (saved.items) |r| try b.push(r);
    if (adj != 0) try b.subRI(.rsp, @intCast(adj));
    for (0..f.params) |i| try ctx.put(@intCast(i), cc.args[i]);

    const labels = try gpa.alloc(bld.Label, f.blocks.items.len);
    defer gpa.free(labels);
    for (labels) |*l| l.* = try b.newLabel();
    const epilog = try b.newLabel();

    for (f.blocks.items, 0..) |blk, bi| {
        try b.bind(labels[bi]);
        for (blk.ops.items) |o| switch (o) {
            .@"const" => |c| switch (res.locs[c.dst]) {
                .reg => |r| try b.movRI(@enumFromInt(r), c.imm),
                else => {
                    try b.movRI(s0, c.imm);
                    try ctx.put(c.dst, s0);
                },
            },
            .copy => |c| try ctx.put(c.dst, try ctx.get(c.src, s0)),
            .arith => |c| {
                const ra_ = try ctx.get(c.a, s0);
                const rb = try ctx.get(c.b, s1);
                if (ra_ != s0) try b.movRR(s0, ra_);
                switch (c.op) {
                    .add => try b.aluRR(.add, s0, rb),
                    .sub => try b.aluRR(.sub, s0, rb),
                    .@"and" => try b.aluRR(.@"and", s0, rb),
                    .@"or" => try b.aluRR(.@"or", s0, rb),
                    .xor => try b.aluRR(.xor, s0, rb),
                    .mul => try b.imulRR(s0, rb),
                }
                try ctx.put(c.dst, s0);
            },
            .load => |c| {
                const base = try ctx.get(c.base, s1);
                try b.movRM(s0, x.ptr(base, c.disp));
                try ctx.put(c.dst, s0);
            },
            .store => |c| {
                const base = try ctx.get(c.base, s1);
                const v = try ctx.get(c.src, s0);
                try b.movMR(x.ptr(base, c.disp), v);
            },
            .call => |c| {
                // Argument registers are never allocated: plain moves, no parallel-move cycles.
                for (c.args, 0..) |a, i| {
                    const r = try ctx.get(a, s0);
                    if (r != cc.args[i]) try b.movRR(cc.args[i], r);
                }
                if (cc.shadow != 0) try b.subRI(.rsp, @intCast(cc.shadow));
                try b.callAbs(c.addr);
                if (cc.shadow != 0) try b.addRI(.rsp, @intCast(cc.shadow));
                try ctx.put(c.dst, .rax);
            },
            .br => |tg| try b.jmp(labels[tg]),
            .cbr => |c| {
                const ra_ = try ctx.get(c.a, s0);
                const rb = try ctx.get(c.b, s1);
                try b.cmpRR(ra_, rb);
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
                try b.jcc(cond, labels[c.then]);
                try b.jmp(labels[c.@"else"]);
            },
            .ret => |v| {
                const r = try ctx.get(v, s0);
                if (r != .rax) try b.movRR(.rax, r);
                try b.jmp(epilog);
            },
        };
    }
    try b.bind(epilog);
    if (adj != 0) try b.addRI(.rsp, @intCast(adj));
    var i = saved.items.len;
    while (i > 0) {
        i -= 1;
        try b.pop(saved.items[i]);
    }
    try b.pop(.rbp);
    try b.ret();
    return .{ .spills = res.spills, .slots = res.num_slots, .nodes = b.count() };
}

/// IR -> bytes in `cb` (finalized). Runs dead-code, jump-to-next and branch-shortening passes.
pub fn compile(gpa: std.mem.Allocator, f: *const ir.Func, cb: *CodeBuffer, opts: Options) !Stats {
    var b = bld.Builder.init(gpa);
    defer b.deinit();
    var st = try lower(gpa, f, &b, opts);
    try b.runPasses(.{ bld.removeDeadCode, bld.removeJumpsToNext });
    if (opts.shorten_branches) try bld.shortenBranches(&b);
    try b.serialize(x.AssemblerT(.{ .validate = true }).init(cb));
    try cb.finalize();
    st.nodes = b.count();
    return st;
}

// ---- tests ------------------------------------------------------------------------------
const t = std.testing;
const builtin = @import("builtin");
const JitRuntime = @import("../core/jit_runtime.zig");
const can_run = builtin.cpu.arch == .x86_64;

fn run(f: *const ir.Func, opts: Options, args: []const i64) !i64 {
    var cb = CodeBuffer.init(t.allocator);
    defer cb.deinit();
    _ = try compile(t.allocator, f, &cb, opts);
    const fun = try JitRuntime.add(&cb);
    defer fun.release();
    var a: [6]i64 = .{0} ** 6;
    for (args, 0..) |v, i| a[i] = v;
    return fun.ptr(*const fn (i64, i64, i64, i64, i64, i64) callconv(.c) i64)(a[0], a[1], a[2], a[3], a[4], a[5]);
}

test "compiler: fib compiles to expected shape (byte-level, any host)" {
    var f = try ir.parse(t.allocator, ir.fib_src);
    defer f.deinit();
    var cb = CodeBuffer.init(t.allocator);
    defer cb.deinit();
    const st = try compile(t.allocator, &f, &cb, .{});
    try t.expectEqual(@as(u32, 0), st.spills);
    const c = cb.code();
    try t.expectEqualSlices(u8, &.{ 0x55, 0x48, 0x89, 0xE5 }, c[0..4]); // push rbp; mov rbp, rsp
    try t.expectEqual(@as(u8, 0xC3), c[c.len - 1]);
    // Every loop branch fits in rel8 after shortening: no 0F 8x / E9 long forms.
    var long: usize = 0;
    for (c, 0..) |byte, i| {
        if (byte == 0xE9) long += 1;
        if (byte == 0x0F and i + 1 < c.len and c[i + 1] & 0xF0 == 0x80) long += 1;
    }
    try t.expectEqual(@as(usize, 0), long);
}

test "compiler: execution - fib, loops, calls, memory, spills" {
    if (!can_run) return error.SkipZigTest;
    var f = try ir.parse(t.allocator, ir.fib_src);
    defer f.deinit();
    for ([_]i64{ 0, 1, 2, 10, 50, 90 }) |n| {
        const want = try ir.interpret(&f, &.{n}, 1 << 20);
        try t.expectEqual(want, try run(&f, .{}, &.{n}));
        try t.expectEqual(want, try run(&f, .{ .allocatable = bit(.rbx) }, &.{n})); // 1 register: spills
    }
    const Helper = struct {
        fn mul3(a: i64, b: i64, c: i64) callconv(.c) i64 {
            return a * b * c;
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
    try t.expectEqual(@as(i64, 3 * 4 * 4 + 3), try run(&g, .{}, &.{ 3, 4, p }));
    try t.expectEqual(@as(i64, 51), mem[1]);
}

test "compiler: differential vs interpreter on random programs" {
    if (!can_run) return error.SkipZigTest;
    var prng = std.Random.DefaultPrng.init(7);
    const r = prng.random();
    for (0..300) |_| {
        var f = ir.Func.init(t.allocator);
        defer f.deinit();
        f.params = 3;
        const nv: u32 = 3 + r.uintLessThan(u32, 20);
        f.num_vregs = nv;
        const nb = 1 + r.uintLessThan(u32, 5);
        for (0..nb) |_| _ = try f.addBlock("b");
        // Reading a never-written vreg is undefined in compiled code: define all up front.
        for (3..nv) |v| try f.emit(0, .{ .@"const" = .{ .dst = @intCast(v), .imm = 0 } });
        for (0..nb) |bi| {
            const blk: u32 = @intCast(bi);
            for (0..r.uintLessThan(u32, 10)) |_| {
                const d = r.uintLessThan(u32, nv);
                if (r.boolean()) {
                    try f.emit(blk, .{ .@"const" = .{ .dst = d, .imm = r.int(i32) } });
                } else try f.emit(blk, .{ .arith = .{ .op = r.enumValue(ir.Arith), .dst = d, .a = r.uintLessThan(u32, nv), .b = r.uintLessThan(u32, nv) } });
            }
            // forward-only control flow => always terminates
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
        try t.expectEqual(want, try run(&f, .{ .allocatable = bit(.rax) | bit(.rbx) }, &args));
    }
}
