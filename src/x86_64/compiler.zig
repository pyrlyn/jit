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
//! - shifts use `rcx` and division uses `rax`/`rdx`: rcx/rdx are argument registers (never
//!   allocated), and rax is saved in r11 around `cqo; idiv`.
//! - folding: a right-hand operand defined by `const` earlier in the same block (and not
//!   redefined since) that fits imm32 becomes an immediate; a spilled one becomes a memory operand.
//! Not yet: AArch64 lowering, stack-passed arguments,
//! immediate and memory-operand folding, cmp/branch fusion beyond `cbr`.
const std = @import("std");
const x = @import("assembler.zig");
const bld = @import("builder.zig");
const ir = @import("../core/ir.zig");
const ra = @import("../core/regalloc.zig");
const rc = @import("../core/reg_classes.zig");
const CodeBuffer = @import("../core/code_buffer.zig").CodeBuffer;
const Gp = x.Gp;

pub const allocatable: ra.RegMask = bit(.rax) | bit(.rbx) | bit(.r12) | bit(.r13) | bit(.r14) | bit(.r15);
pub const callee_saved: ra.RegMask = bit(.rbx) | bit(.r12) | bit(.r13) | bit(.r14) | bit(.r15);
const s0: Gp = .r10;
const s1: Gp = .r11;

/// FP class (scalar f32/f64 in xmm registers, `core/reg_classes.zig`). Every xmm register is
/// caller-saved on SysV; on Win64 xmm6..15 are callee-saved, so only xmm0..5 are used there.
const win = x.CallConv.host.shadow != 0;
pub const fp_allocatable: ra.RegMask = if (win) 0b1111 else 0x3FFF; // xmm0..3 / xmm0..13
const fs0: bld.Xmm = if (win) 4 else 14;
const fs1: bld.Xmm = if (win) 5 else 15;

fn bit(r: Gp) ra.RegMask {
    return @as(ra.RegMask, 1) << @backingInt(r);
}

pub const Options = struct {
    /// Restrict the allocatable set (spill stress tests).
    allocatable: ra.RegMask = allocatable,
    /// Keep call-crossing values in caller-saved registers when no callee-saved one is free,
    /// saving/restoring them around each call (`regalloc.Config.split_at_calls`).
    split_at_calls: bool = true,
    shorten_branches: bool = true,
    algo: ra.Algo = .linear,
    /// Per-block register segments for spilled values (`regalloc.Config.split`).
    split: bool = true,
    /// Restrict the FP (xmm) allocatable set (spill stress tests).
    fp_allocatable: ra.RegMask = fp_allocatable,
};

pub const Stats = struct {
    spills: u32 = 0,
    slots: u32 = 0,
    nodes: usize = 0,
    /// Operands folded into an imm32 (`add r, imm`, `cmp r, imm`, `imul r, r, imm`).
    folded_imm: u32 = 0,
    /// Spilled right-hand operands used directly as memory (`add r, [rbp-off]`).
    folded_mem: u32 = 0,
    /// Register segments given to spilled values (live-range splitting).
    segments: u32 = 0,
    /// Copies removed by coalescing (`.graph`).
    coalesced: u32 = 0,
};

/// Compiles `f` into `b` (a Builder). Run passes and serialize separately, or use `compile`.
/// Functions with block parameters are first rewritten by `ir.Func.lowerBlockParams`.
pub fn lower(gpa: std.mem.Allocator, f0: *const ir.Func, b: *bld.Builder, opts: Options) !Stats {
    try f0.verify();
    var lowered: ?ir.Func = if (f0.hasBlockParams()) try f0.lowerBlockParams(gpa) else null;
    defer if (lowered) |*g| g.deinit();
    const f: *const ir.Func = if (lowered) |*g| g else f0;
    const cc = x.CallConv.host;
    if (f.params > ir.max_args) return error.TooManyParams;

    // ---- register allocation input (one run per register class) ----
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const A = arena.allocator();
    const types = try rc.check(A, f);
    const rcfg: ra.Config = .{ .allocatable = opts.allocatable, .callee_saved = callee_saved & opts.allocatable, .split_at_calls = opts.split_at_calls, .algo = opts.algo, .split = opts.split };
    var res = try rc.allocateClass(gpa, f, types, .gp, true, rcfg);
    defer res.deinit(gpa);
    var fres = try rc.allocateClass(gpa, f, types, .fp, false, .{ .allocatable = opts.fp_allocatable });
    defer fres.deinit(gpa);
    // Vregs saved/restored around calls (split ranges), ascending.
    var save_vregs: std.ArrayList(ir.VReg) = .empty;
    for (res.save_slot, 0..) |s, v| if (s != null) try save_vregs.append(A, @intCast(v));

    // ---- frame ----
    const used = res.usedRegs();
    var saved: std.ArrayList(Gp) = .empty;
    defer saved.deinit(gpa);
    for ([_]Gp{ .rbx, .r12, .r13, .r14, .r15 }) |r| if (used & bit(r) != 0) try saved.append(gpa, r);
    const nsaved: u32 = @intCast(saved.items.len);
    const slots_bytes = (res.num_slots + fres.num_slots) * 8;
    const adj = std.mem.alignForward(u32, nsaved * 8 + slots_bytes, 16) - nsaved * 8;

    const L = struct {
        locs: []const ra.Loc,
        res: *const ra.Result,
        pos: *const u32,
        nsaved: u32,
        b: *bld.Builder,
        flocs: []const ra.Loc,
        /// First FP spill slot (after the GP slots).
        fbase: u32,
        /// xmm register holding FP `v` (loads spilled values into `scratch`; undefined reads as +0.0).
        fn fget(s: @This(), v: ir.VReg, scratch: bld.Xmm) !bld.Xmm {
            return switch (s.flocs[v]) {
                .reg => |r| @intCast(r),
                .stack => |k| blk: {
                    try s.b.emit(.sseRM, .{ .movsd, scratch, s.slot(s.fbase + k) });
                    break :blk scratch;
                },
                .none => blk: {
                    try s.b.emit(.sseRR, .{ .xorps, scratch, scratch });
                    break :blk scratch;
                },
            };
        }
        fn fput(s: @This(), v: ir.VReg, src: bld.Xmm) !void {
            switch (s.flocs[v]) {
                .reg => |r| if (r != src) try s.b.emit(.sseRR, .{ .movaps, @intCast(r), src }),
                .stack => |k| try s.b.emit(.sseMR, .{ .movsd, s.slot(s.fbase + k), src }),
                .none => {},
            }
        }
        fn slot(s: @This(), k: u32) x.Mem {
            return x.ptr(.rbp, -@as(i32, @intCast(8 * (s.nsaved + 1 + k))));
        }
        /// Register holding `v` (loads spilled values into `scratch`).
        fn get(s: @This(), v: ir.VReg, scratch: Gp) !Gp {
            return switch (s.locs[v]) {
                .reg => |r| @fromBackingInt(@intCast(r)),
                .stack => |k| blk: {
                    if (s.res.regAt(v, s.pos.*)) |r| break :blk @fromBackingInt(@intCast(r));
                    try s.b.emit(.movRM, .{ scratch, s.slot(k) });
                    break :blk scratch;
                },
                .none => blk: { // undefined value: reads as 0
                    try s.b.emit(.movRI, .{ scratch, 0 });
                    break :blk scratch;
                },
            };
        }
        fn load(s: @This(), v: ir.VReg, r: u8) !void {
            try s.b.emit(.movRM, .{ @fromBackingInt(@intCast(r)), s.slot(s.locs[v].stack) });
        }
        fn put(s: @This(), v: ir.VReg, src: Gp) !void {
            switch (s.locs[v]) {
                .reg => |r| if (@as(Gp, @fromBackingInt(@intCast(r))) != src) try s.b.emit(.movRR, .{ @fromBackingInt(@intCast(r)), src }),
                .stack => |k| {
                    // Inside a split segment: write through to the register and the slot.
                    if (s.res.regAt(v, s.pos.*)) |r| if (@as(Gp, @fromBackingInt(@intCast(r))) != src) try s.b.emit(.movRR, .{ @fromBackingInt(@intCast(r)), src });
                    try s.b.emit(.movMR, .{ s.slot(k), src });
                },
                .none => {},
            }
        }
    };
    var pos: u32 = 0;
    const ctx: L = .{ .locs = res.locs, .res = &res, .pos = &pos, .nsaved = nsaved, .b = b, .flocs = fres.locs, .fbase = res.num_slots };

    // ---- prolog ----
    try b.comment(f.name);
    try b.emit(.push, .{.rbp});
    try b.emit(.movRR, .{ .rbp, .rsp });
    for (saved.items) |r| try b.emit(.push, .{r});
    if (adj != 0) try b.emit(.subRI, .{ .rsp, @intCast(adj) });
    for (0..f.params) |i| {
        if (i < cc.args.len) {
            try ctx.put(@intCast(i), cc.args[i]);
        } else {
            try b.emit(.movRM, .{ s0, cc.stackArg(i) });
            try ctx.put(@intCast(i), s0);
        }
    }
    // Values read before any def on some path are 0 (IR semantics, as in `ir.interpret`).
    if (res.entry_undef.len != 0) try b.emit(.movRI, .{ s0, 0 });
    for (res.entry_undef) |v| try ctx.put(v, s0);
    if (fres.entry_undef.len != 0) try b.emit(.sseRR, .{ .xorps, fs0, fs0 });
    for (fres.entry_undef) |v| try ctx.fput(v, fs0);

    const labels = try gpa.alloc(bld.Label, f.blocks.items.len);
    defer gpa.free(labels);
    for (labels) |*l| l.* = try b.newLabel();
    const epilog = try b.newLabel();

    // Block-local constants for immediate folding: known[v] = value of the last `const` def.
    const known = try A.alloc(?i64, @max(f.num_vregs, 1));
    var folded_imm: u32 = 0;
    var folded_mem: u32 = 0;
    pos = 1; // regalloc position: entry block starts with the parameter-def inst
    for (f.blocks.items, 0..) |blk, bi| {
        try b.bind(labels[bi]);
        @memset(known, null);
        for (blk.ops.items) |o| {
            defer pos += 1;
            try res.loadsAt(pos, ctx, L.load);
            defer if (o.def()) |d| {
                known[d] = if (o == .@"const") o.@"const".imm else null;
            };
            switch (o) {
                .@"const" => |c| switch (res.locs[c.dst]) {
                    .reg => |r| try b.emit(.movRI, .{ @fromBackingInt(@intCast(r)), c.imm }),
                    else => {
                        try b.emit(.movRI, .{ s0, c.imm });
                        try ctx.put(c.dst, s0);
                    },
                },
                .copy => |c| if (rc.classOf(types[c.dst]) == .fp)
                    try ctx.fput(c.dst, try ctx.fget(c.src, fs0))
                else
                    try ctx.put(c.dst, try ctx.get(c.src, s0)),
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
                        if (known[c.b]) |k| if (std.math.cast(i32, k)) |imm| {
                            const ra0 = try ctx.get(c.a, s0);
                            if (alu) |op| {
                                if (ra0 != s0) try b.emit(.movRR, .{ s0, ra0 });
                                try b.emit(.aluRI, .{ op, s0, imm });
                            } else try b.emit(.imulRRI, .{ s0, ra0, imm });
                            try ctx.put(c.dst, s0);
                            folded_imm += 1;
                            break :fold;
                        };
                        if (res.locs[c.b] == .stack and res.regAt(c.b, pos) == null) {
                            const ra0 = try ctx.get(c.a, s0);
                            if (ra0 != s0) try b.emit(.movRR, .{ s0, ra0 });
                            const m = ctx.slot(res.locs[c.b].stack);
                            if (alu) |op| try b.emit(.aluRM, .{ op, s0, m }) else try b.emit(.imulRM, .{ s0, m });
                            try ctx.put(c.dst, s0);
                            folded_mem += 1;
                            break :fold;
                        }
                    }
                    const ra_ = try ctx.get(c.a, s0);
                    const rb = try ctx.get(c.b, s1);
                    if (ra_ != s0) try b.emit(.movRR, .{ s0, ra_ });
                    switch (c.op) {
                        .shl, .shr, .sar => {
                            try b.emit(.movRR, .{ .rcx, rb });
                            try b.emit(.shiftRCl, .{ switch (c.op) {
                                .shl => .shl,
                                .shr => .shr,
                                else => .sar,
                            }, s0 });
                        },
                        .sdiv, .srem => {
                            try b.emit(.movRR, .{ .rcx, rb });
                            try b.emit(.movRR, .{ s1, .rax });
                            try b.emit(.movRR, .{ .rax, s0 });
                            try b.add(.cqo);
                            try b.emit(.idivR, .{.rcx});
                            try b.emit(.movRR, .{ s0, if (c.op == .sdiv) .rax else .rdx });
                            try b.emit(.movRR, .{ .rax, s1 });
                        },
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
                    if (c.ty != .i64) {
                        try b.emit(.sseRM, .{ sseMov(c.ty), fs0, x.ptr(base, c.disp) });
                        try ctx.fput(c.dst, fs0);
                    } else {
                        try b.emit(.movRM, .{ s0, x.ptr(base, c.disp) });
                        try ctx.put(c.dst, s0);
                    }
                },
                .store => |c| {
                    const base = try ctx.get(c.base, s1);
                    if (c.ty != .i64) {
                        try b.emit(.sseMR, .{ sseMov(c.ty), x.ptr(base, c.disp), try ctx.fget(c.src, fs0) });
                    } else {
                        const v = try ctx.get(c.src, s0);
                        try b.emit(.movMR, .{ x.ptr(base, c.disp), v });
                    }
                },
                .fconst => |c| {
                    try b.emit(.movRI, .{ s0, @bitCast(c.bits) });
                    try b.emit(.movqXR, .{ fs0, s0 });
                    try ctx.fput(c.dst, fs0);
                },
                .typed => |c| {
                    const d64 = c.ty == .f64;
                    const op: bld.SseOp = switch (c.op) {
                        .add => if (d64) .addsd else .addss,
                        .sub => if (d64) .subsd else .subss,
                        .mul => if (d64) .mulsd else .mulss,
                        .div => if (d64) .divsd else .divss,
                        // x86 semantics = IR semantics: the second operand when unordered or equal.
                        .min => if (d64) .minsd else .minss,
                        .max => if (d64) .maxsd else .maxss,
                        .@"and", .@"or", .xor => unreachable, // integer vectors only (rejected)
                    };
                    const xa = try ctx.fget(c.a, fs0);
                    const xb = try ctx.fget(c.b, fs1);
                    if (xa != fs0) try b.emit(.sseRR, .{ .movaps, fs0, xa });
                    try b.emit(.sseRR, .{ op, fs0, xb });
                    try ctx.fput(c.dst, fs0);
                },
                .fcmp => |c| {
                    const ucomi: bld.SseOp = if (c.ty == .f64) .ucomisd else .ucomiss;
                    const xa = try ctx.fget(c.a, fs0);
                    const xb = try ctx.fget(c.b, fs1);
                    // ucomis: unordered sets ZF=PF=CF=1. lt/le swap operands so that `a`/`ae`
                    // (CF=0) are false on NaN; eq/ne combine with PF.
                    switch (c.cond) {
                        .lt, .le => try b.emit(.sseRR, .{ ucomi, xb, xa }),
                        else => try b.emit(.sseRR, .{ ucomi, xa, xb }),
                    }
                    switch (c.cond) {
                        .eq, .ne => {
                            try b.emit(.setcc, .{ if (c.cond == .eq) .e else .ne, s0 });
                            try b.emit(.setcc, .{ if (c.cond == .eq) .np else .p, s1 });
                            try b.emit(.movzxB, .{ s0, s0 });
                            try b.emit(.movzxB, .{ s1, s1 });
                            try b.emit(.aluRR, .{ if (c.cond == .eq) .@"and" else .@"or", s0, s1 });
                        },
                        .lt, .gt => {
                            try b.emit(.setcc, .{ .a, s0 });
                            try b.emit(.movzxB, .{ s0, s0 });
                        },
                        .le, .ge => {
                            try b.emit(.setcc, .{ .ae, s0 });
                            try b.emit(.movzxB, .{ s0, s0 });
                        },
                    }
                    try ctx.put(c.dst, s0);
                },
                .cvt => |c| switch (c.to) {
                    .i64 => {
                        try b.emit(.cvtToInt, .{ if (c.from == .f64) .cvttsd2si else .cvttss2si, s0, try ctx.fget(c.src, fs0) });
                        try ctx.put(c.dst, s0);
                    },
                    else => {
                        if (c.from == .i64) {
                            const r = try ctx.get(c.src, s0);
                            try b.emit(.sseRR, .{ .xorps, fs0, fs0 }); // break the false dependency
                            try b.emit(.cvtFromInt, .{ if (c.to == .f64) .cvtsi2sd else .cvtsi2ss, fs0, r });
                        } else {
                            const xs = try ctx.fget(c.src, fs1);
                            try b.emit(.sseRR, .{ if (c.to == .f64) .cvtss2sd else .cvtsd2ss, fs0, xs });
                        }
                        try ctx.fput(c.dst, fs0);
                    },
                },
                .call => |c| {
                    // Split ranges: save caller-saved registers live across this call.
                    for (save_vregs.items) |v| if (res.savedAcross(v, pos)) {
                        try b.emit(.movMR, .{ ctx.slot(res.save_slot[v].?), @fromBackingInt(@intCast(res.locs[v].reg)) });
                    };
                    // Argument registers are never allocated: plain moves, no parallel-move cycles.
                    // Stack arguments: pushed right to left, padded so rsp stays 16-aligned.
                    const nreg = @min(c.args.len, cc.args.len);
                    const nstack = c.args.len - nreg;
                    const pad: u32 = if (nstack % 2 == 1) 8 else 0;
                    if (pad != 0) try b.emit(.subRI, .{ .rsp, @intCast(pad) });
                    var k = c.args.len;
                    while (k > nreg) {
                        k -= 1;
                        try b.emit(.push, .{try ctx.get(c.args[k], s0)});
                    }
                    for (c.args[0..nreg], 0..) |a, i| {
                        const r = try ctx.get(a, s0);
                        if (r != cc.args[i]) try b.emit(.movRR, .{ cc.args[i], r });
                    }
                    if (cc.shadow != 0) try b.emit(.subRI, .{ .rsp, @intCast(cc.shadow) });
                    try b.emit(.callAbs, .{c.addr});
                    const pop: u32 = cc.shadow + pad + 8 * @as(u32, @intCast(nstack));
                    if (pop != 0) try b.emit(.addRI, .{ .rsp, @intCast(pop) });
                    var restored = false;
                    for (save_vregs.items) |v| if (res.savedAcross(v, pos)) {
                        if (!restored) try b.emit(.movRR, .{ s1, .rax });
                        restored = true;
                        try b.emit(.movRM, .{ @fromBackingInt(@intCast(res.locs[v].reg)), ctx.slot(res.save_slot[v].?) });
                    };
                    try ctx.put(c.dst, if (restored) s1 else .rax);
                },
                .bra, .cbra => unreachable, // removed by lowerBlockParams
                .br => |tg| try b.emit(.jmp, .{labels[tg]}),
                .cbr => |c| {
                    const ra_ = try ctx.get(c.a, s0);
                    if (known[c.b]) |k| if (std.math.cast(i32, k)) |imm| {
                        try b.emit(.cmpRI, .{ ra_, imm });
                        folded_imm += 1;
                    } else try b.emit(.cmpRR, .{ ra_, try ctx.get(c.b, s1) }) else try b.emit(.cmpRR, .{ ra_, try ctx.get(c.b, s1) });
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
                .splat, .extract => unreachable, // rejected by reg_classes.check
                .ret => |v| {
                    const r = try ctx.get(v, s0);
                    if (r != .rax) try b.emit(.movRR, .{ .rax, r });
                    try b.emit(.jmp, .{epilog});
                },
            }
        }
    }
    try b.bind(epilog);
    if (adj != 0) try b.emit(.addRI, .{ .rsp, @intCast(adj) });
    var i = saved.items.len;
    while (i > 0) {
        i -= 1;
        try b.emit(.pop, .{saved.items[i]});
    }
    try b.emit(.pop, .{.rbp});
    try b.add(.ret);
    return .{ .spills = res.spills, .slots = res.num_slots, .nodes = b.count(), .folded_imm = folded_imm, .folded_mem = folded_mem, .segments = @intCast(res.segs.len), .coalesced = res.coalesced };
}

fn sseMov(tp: ir.Type) bld.SseMov {
    return if (tp == .f32) .movss else .movsd;
}

/// IR -> bytes in `cb` (finalized). Runs dead-code, jump-to-next and branch-shortening passes.
pub fn compile(gpa: std.mem.Allocator, f: *const ir.Func, cb: *CodeBuffer, opts: Options) !Stats {
    var b = bld.Builder.init(gpa);
    defer b.deinit();
    var st = try lower(gpa, f, &b, opts);
    try b.runPasses(.{ bld.removeDeadCode, bld.removeJumpsToNext });
    // The node list is complete: drop its growth slack before the output buffer grows.
    b.nodes.shrinkAndFree(gpa, b.nodes.items.len);
    if (opts.shorten_branches)
        try bld.serializeShortened(&b, cb)
    else
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
    var a: [6]i64 = @splat(0);
    for (args, 0..) |v, i| a[i] = v;
    return fun.ptr(*const fn (i64, i64, i64, i64, i64, i64) callconv(.c) i64)(a[0], a[1], a[2], a[3], a[4], a[5]);
}

test "x64 compiler: a value undefined on some path reads as 0" {
    // e -> j skips the def of v2 (and of the float v4): the interpreter reads 0 there.
    var f = try ir.parse(t.allocator,
        \\func f 2
        \\e:
        \\  v9 = const 7
        \\  cbr gt v0 v9 a j
        \\a:
        \\  v2 = const 5
        \\  v4 = fconst.f64 2.5
        \\  br j
        \\j:
        \\  v3 = add v2 v0
        \\  v5 = fconst.f64 0
        \\  v6 = fcmp.f64 eq v4 v5
        \\  v7 = add v3 v6
        \\  ret v7
    );
    defer f.deinit();
    if (!can_run) return error.SkipZigTest;
    for ([_]i64{ 1, 10 }) |n| {
        const want = try ir.interpret(&f, &.{ n, 1000 }, 1000);
        for ([_]ra.RegMask{ allocatable, bit(.rsi) | bit(.rax), bit(.rsi) }) |m| {
            try t.expectEqual(want, try run(&f, .{ .allocatable = m }, &.{ n, 1000 }));
        }
    }
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
    // Only caller-saved rax: v0/v2 cross the call in rax, split (saved/restored) around it.
    for ([_]bool{ true, false }) |split| {
        mem = .{ 0, 0 };
        try t.expectEqual(@as(i64, 3 * 4 * 4 + 3), try run(&g, .{ .allocatable = bit(.rax), .split_at_calls = split }, &.{ 3, 4, p }));
        try t.expectEqual(@as(i64, 51), mem[1]);
    }
}

test "compiler: stack-passed parameters and call arguments" {
    if (!can_run) return error.SkipZigTest;
    const Helper = struct {
        fn sum9(a: i64, b: i64, c: i64, d: i64, e: i64, f_: i64, g: i64, h: i64, i: i64) callconv(.c) i64 {
            return a + 2 * b + 3 * c + 4 * d + 5 * e + 6 * f_ + 7 * g + 8 * h + 9 * i;
        }
        fn sum8(a: i64, b: i64, c: i64, d: i64, e: i64, f_: i64, g: i64, h: i64) callconv(.c) i64 {
            return a - b + c - d + e - f_ + g - h;
        }
    };
    // 8 parameters (2+ on the stack), calls with 9 (odd: padding) and 8 stack-heavy args.
    var g = ir.Func.init(t.allocator);
    defer g.deinit();
    g.params = 8;
    const e = try g.addBlock("e");
    const a9 = try g.alloc().dupe(ir.VReg, &.{ 0, 1, 2, 3, 4, 5, 6, 7, 7 });
    try g.emit(e, .{ .call = .{ .dst = 8, .addr = @intFromPtr(&Helper.sum9), .args = a9 } });
    const a8 = try g.alloc().dupe(ir.VReg, &.{ 8, 7, 6, 5, 4, 3, 2, 1 });
    try g.emit(e, .{ .call = .{ .dst = 9, .addr = @intFromPtr(&Helper.sum8), .args = a8 } });
    try g.emit(e, .{ .arith = .{ .op = .add, .dst = 10, .a = 9, .b = 0 } }); // v0 live across both calls
    try g.emit(e, .{ .ret = 10 });
    const p = [_]i64{ 1, 20, 300, 4000, 50000, 600000, 7000000, 80000000 };
    const s9 = Helper.sum9(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[7]);
    const want = Helper.sum8(s9, p[7], p[6], p[5], p[4], p[3], p[2], p[1]) + p[0];
    for ([_]ra.RegMask{ allocatable, bit(.rbx) }) |mask| {
        var cb = CodeBuffer.init(t.allocator);
        defer cb.deinit();
        _ = try compile(t.allocator, &g, &cb, .{ .allocatable = mask });
        const fun = try JitRuntime.add(&cb);
        defer fun.release();
        const F = *const fn (i64, i64, i64, i64, i64, i64, i64, i64) callconv(.c) i64;
        try t.expectEqual(want, fun.ptr(F)(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7]));
    }
}

test "compiler: shifts and signed division match the interpreter" {
    if (!can_run) return error.SkipZigTest;
    // Many values live at once so rax holds one across the idiv.
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
        \\  v8 = add v8 v1
        \\  ret v8
    ;
    var f = try ir.parse(t.allocator, src);
    defer f.deinit();
    try f.verify();
    const cases = [_][2]i64{ .{ 1, 0 }, .{ -1, 62 }, .{ 12345, 3 }, .{ -98765, 65 }, .{ std.math.maxInt(i64), 1 }, .{ -7, 2 } };
    for (cases) |c| {
        const want = try ir.interpret(&f, &c, 1000);
        try t.expectEqual(want, try run(&f, .{}, &c));
        try t.expectEqual(want, try run(&f, .{ .allocatable = bit(.rax) | bit(.rbx) }, &c));
    }
}

test "compiler: immediate and memory operand folding (byte-level, any host)" {
    const src =
        \\func f 1
        \\e:
        \\  v1 = const 5
        \\  v2 = add v0 v1
        \\  v3 = const 3000000000
        \\  v2 = mul v2 v3
        \\  v4 = const 7
        \\  v2 = mul v2 v4
        \\  cbr lt v2 v4 a b
        \\a:
        \\  v2 = sub v2 v1
        \\  ret v2
        \\b:
        \\  ret v2
    ;
    var f = try ir.parse(t.allocator, src);
    defer f.deinit();
    var cb = CodeBuffer.init(t.allocator);
    defer cb.deinit();
    const st = try compile(t.allocator, &f, &cb, .{});
    // add 5, imul 7, cmp 7 fold; 3e9 does not fit imm32; `sub v1` in block a is not block-local.
    try t.expectEqual(@as(u32, 3), st.folded_imm);
    try t.expect(std.mem.indexOf(u8, cb.code(), &.{ 0x49, 0x83, 0xC2, 0x05 }) != null); // add r10, 5

    // One allocatable register: spilled right-hand operands become memory operands.
    var cb2 = CodeBuffer.init(t.allocator);
    defer cb2.deinit();
    var g = try ir.parse(t.allocator, ir.fib_src);
    defer g.deinit();
    const st2 = try compile(t.allocator, &g, &cb2, .{ .allocatable = bit(.rbx) });
    try t.expect(st2.folded_mem > 0);
}

test "compiler: block parameters (SSA input)" {
    var f = try ir.parse(t.allocator, ir.fib_ssa_src);
    defer f.deinit();
    var cb = CodeBuffer.init(t.allocator);
    defer cb.deinit();
    _ = try compile(t.allocator, &f, &cb, .{});
    if (!can_run) return;
    for ([_]i64{ 0, 1, 10, 90 }) |n| {
        const want = try ir.interpret(&f, &.{n}, 1 << 20);
        try t.expectEqual(want, try run(&f, .{}, &.{n}));
        try t.expectEqual(want, try run(&f, .{ .allocatable = bit(.rbx) }, &.{n}));
    }
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
                } else try f.emit(blk, .{ .arith = .{ .op = @fromBackingInt(@intCast(r.uintLessThan(u8, @backingInt(ir.Arith.sdiv)))), .dst = d, .a = r.uintLessThan(u32, nv), .b = r.uintLessThan(u32, nv) } });
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
        try t.expectEqual(want, try run(&f, .{ .allocatable = bit(.rax) | bit(.rbx) | bit(.r12), .split = false }, &args));
        try t.expectEqual(want, try run(&f, .{ .allocatable = bit(.rax) | bit(.rbx) | bit(.r12), .algo = .graph }, &args));
        try t.expectEqual(want, try run(&f, .{ .algo = .graph }, &args));
    }
}

test "compiler: splitting and coalescing are exercised and execute correctly" {
    // 8 values live through a loop with 2 registers: spilled values used several times per
    // block get register segments; the copies in `done` coalesce under `.graph` (the
    // loop-carried `v6 = copy v7` does not: hole-free intervals make them interfere).
    const src =
        \\func k 2
        \\entry:
        \\  v2 = const 3
        \\  v3 = const 5
        \\  v4 = const 7
        \\  v5 = const 11
        \\  v6 = const 0
        \\  v8 = const 0
        \\  br loop
        \\loop:
        \\  v7 = add v6 v2
        \\  v7 = add v7 v2
        \\  v7 = mul v7 v3
        \\  v7 = add v7 v3
        \\  v7 = xor v7 v4
        \\  v7 = add v7 v4
        \\  v7 = add v7 v5
        \\  v7 = sub v7 v5
        \\  v6 = copy v7
        \\  v1 = sub v1 v0
        \\  cbr ne v1 v8 loop done
        \\done:
        \\  v9 = copy v6
        \\  v10 = add v9 v2
        \\  v11 = copy v10
        \\  ret v11
    ;
    var f = try ir.parse(t.allocator, src);
    defer f.deinit();
    var cb = CodeBuffer.init(t.allocator);
    defer cb.deinit();
    const st = try compile(t.allocator, &f, &cb, .{ .allocatable = bit(.rax) | bit(.rbx) });
    try t.expect(st.spills > 0 and st.segments > 0);
    var cb2 = CodeBuffer.init(t.allocator);
    defer cb2.deinit();
    const st2 = try compile(t.allocator, &f, &cb2, .{ .algo = .graph });
    try t.expect(st2.coalesced > 0);
    if (!can_run) return;
    for ([_]i64{ 1, 2, 7 }) |n| {
        const args = [_]i64{ 1, n };
        const want = try ir.interpret(&f, &args, 1 << 16);
        try t.expectEqual(want, try run(&f, .{ .allocatable = bit(.rax) | bit(.rbx) }, &args));
        try t.expectEqual(want, try run(&f, .{ .allocatable = bit(.rax) | bit(.rbx), .algo = .graph }, &args));
        try t.expectEqual(want, try run(&f, .{ .algo = .graph }, &args));
    }
}
